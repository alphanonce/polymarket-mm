"""
Insight-Driven Quote Model

Quote model based on empirical market making insights from Polymarket analysis:
- TTE Effect: Spreads widen 5-6x near expiry (276->1399 bps for BTC)
- Moneyness: ATM (~0.5) has tightest spreads, ITM/OTM wider
- Asset Volatility: SOL/XRP ~60% vol vs BTC ~32%
- Order Imbalance: Near-balanced flow (+-0.5%)
- Depth: 20k-46k average depth, safe to quote 500-2000 units

This model is stateless - it doesn't need price history like TpBS.
"""

from dataclasses import dataclass, field

from strategy.models.base import NormalizationConfig, QuoteModel, QuoteResult, StrategyState
from strategy.utils.polymarket import MAX_PRICE, MIN_PRICE, get_tick_size


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to [min_val, max_val] range."""
    return max(min_val, min(max_val, value))


@dataclass
class InsightQuoteConfig:
    """Configuration for Insight-driven quote model."""

    # Asset-specific base spreads (from empirical medians, in basis points)
    asset_base_spreads_bps: dict = field(
        default_factory=lambda: {
            "btc": 225.0,  # Tightest (most liquid)
            "eth": 250.0,
            "sol": 420.0,
            "xrp": 515.0,  # Widest (least liquid)
        }
    )
    default_base_spread_bps: float = 300.0  # Fallback for unknown assets

    # TTE multipliers (from empirical TTE analysis)
    # Format: (minutes_threshold, multiplier)
    # TTE >= threshold gets the multiplier, checked in order
    tte_multipliers: list = field(
        default_factory=lambda: [
            (12, 1.0),  # 12-15 min: 1.0x (baseline)
            (7, 1.3),  # 7-12 min: 1.3x
            (3, 2.5),  # 3-7 min: 2.5x
            (0, 5.0),  # 0-3 min: 5.0x (max widening)
        ]
    )

    # Moneyness adjustment (wider away from 0.5)
    moneyness_base_mult: float = 1.0  # At ATM (mid=0.5)
    moneyness_edge_mult: float = 3.0  # At deep ITM/OTM (mid=0.1 or 0.9)
    atm_center: float = 0.5  # ATM price level
    moneyness_range: float = 0.4  # Distance from ATM to edge

    # Inventory skew parameters
    max_inventory_skew_bps: float = 50.0  # Max skew in basis points
    max_position_pct: float = 0.10  # Max position as % of equity

    # Spread bounds
    min_spread_bps: float = 50.0  # 0.5% minimum spread
    max_spread_bps: float = 5000.0  # 50% maximum spread

    # Minimum edge from market
    min_edge: float = 0.005  # 0.5 cents minimum edge from BBO

    # Maker enforcement
    enforce_maker: bool = True  # Adjust to avoid crossing spread
    maker_offset_ticks: int = 1  # Ticks to offset from BBO

    # Market timing (for TTE calculation)
    market_duration_minutes: float = 15.0  # Default 15-minute markets


class InsightQuoteModel(QuoteModel):
    """
    Insight-driven quote model based on empirical market making analysis.

    Key features:
    - Dynamic spread based on time-to-expiry (TTE)
    - Moneyness adjustment (wider spreads away from ATM)
    - Asset-specific base spreads
    - Inventory skew for position management
    - Stateless model (no price history needed)

    TTE is calculated from the market slug timestamp which contains the
    market start time. For 15-minute markets, expiry is start + 15 minutes.
    """

    def __init__(
        self,
        config: InsightQuoteConfig | None = None,
        normalization: NormalizationConfig | None = None,
    ):
        super().__init__(normalization)
        self.config = config or InsightQuoteConfig()

    def copy(self) -> "InsightQuoteModel":
        """Create a copy of this model (stateless, so just recreate)."""
        return InsightQuoteModel(config=self.config, normalization=self._norm_config)

    def reset(self) -> None:
        """Reset model state (no-op for stateless model)."""
        pass

    def compute_raw(self, state: StrategyState) -> QuoteResult:
        """
        Compute raw bid and ask prices using insight-driven model.

        Args:
            state: Current strategy state

        Returns:
            QuoteResult with raw bid/ask prices
        """
        mid = state.mid_price

        # Validate mid price
        if mid <= 0 or mid >= 1:
            return QuoteResult(
                bid_price=0,
                ask_price=0,
                should_quote=False,
                reason="Invalid mid price",
            )

        # 1. Get asset-specific base spread
        asset = self._get_asset_from_state(state)
        base_spread_bps = self.config.asset_base_spreads_bps.get(
            asset.lower(), self.config.default_base_spread_bps
        )

        # 2. Calculate TTE multiplier
        tte_minutes = self._get_tte_minutes(state)
        tte_mult = self._get_tte_multiplier(tte_minutes)

        # 3. Calculate moneyness multiplier (wider away from ATM)
        moneyness_mult = self._get_moneyness_multiplier(mid)

        # 4. Combined spread calculation
        spread_bps = base_spread_bps * tte_mult * moneyness_mult
        spread_bps = _clamp(
            spread_bps, self.config.min_spread_bps, self.config.max_spread_bps
        )
        spread = spread_bps / 10000.0

        # 5. Inventory skew (shift quotes based on position)
        skew = self._compute_inventory_skew(state)

        # 6. Calculate bid and ask
        half_spread = spread / 2
        bid = mid - half_spread + skew
        ask = mid + half_spread + skew

        # 7. Clamp to valid price range
        bid = _clamp(bid, MIN_PRICE, MAX_PRICE)
        ask = _clamp(ask, MIN_PRICE, MAX_PRICE)

        # 8. Ensure bid < ask
        if bid >= ask:
            # Reset to symmetric spread around mid
            bid = mid - half_spread
            ask = mid + half_spread
            bid = _clamp(bid, MIN_PRICE, MAX_PRICE)
            ask = _clamp(ask, MIN_PRICE, MAX_PRICE)

        # Compute confidence based on TTE and position
        confidence = self._compute_confidence(tte_minutes, state)

        return QuoteResult(
            bid_price=bid,
            ask_price=ask,
            confidence=confidence,
            should_quote=True,
        )

    def compute(self, state: StrategyState) -> QuoteResult:
        """
        Compute bid and ask prices with normalization and maker enforcement.

        Overrides base class to add maker enforcement after normalization.

        Args:
            state: Current strategy state

        Returns:
            QuoteResult with normalized and maker-enforced bid/ask prices
        """
        result = super().compute(state)

        if not result.should_quote:
            return result

        bid, ask = result.bid_price, result.ask_price

        # Apply maker enforcement
        if self.config.enforce_maker:
            bid, ask = self._enforce_maker(bid, ask, state)

        return QuoteResult(
            bid_price=bid,
            ask_price=ask,
            confidence=result.confidence,
            should_quote=result.should_quote,
            reason=result.reason,
        )

    def _get_asset_from_state(self, state: StrategyState) -> str:
        """Extract asset name from state."""
        # Token ID or asset ID might contain the asset name
        asset_id = state.token_id or ""

        # Check for known assets in the ID
        for asset in ["btc", "eth", "sol", "xrp"]:
            if asset in asset_id.lower():
                return asset

        # Fallback: return as-is or default
        return asset_id.split("-")[0] if "-" in asset_id else "btc"

    def _get_tte_minutes(self, state: StrategyState) -> float:
        """
        Calculate time-to-expiry in minutes.

        The TTE is typically calculated from the market slug timestamp.
        For backtesting, this may be set in the state's market data.

        Returns:
            Time to expiry in minutes (defaults to market duration if not available)
        """
        market = state.market

        # Check if market has explicit TTE (>= 0 means it was set)
        if hasattr(market, "tte_minutes") and market.tte_minutes >= 0:
            return market.tte_minutes

        # Check if we have expiry timestamp
        if hasattr(market, "expiry_ts") and market.expiry_ts > 0:
            current_ts = market.timestamp_ns / 1e9  # Convert ns to seconds
            tte_seconds = market.expiry_ts - current_ts
            return max(0, tte_seconds / 60)

        # Default: return mid-range TTE
        return self.config.market_duration_minutes / 2

    def _get_tte_multiplier(self, tte_minutes: float) -> float:
        """
        Get TTE-based spread multiplier.

        Args:
            tte_minutes: Time to expiry in minutes

        Returns:
            Spread multiplier based on TTE
        """
        for threshold, mult in self.config.tte_multipliers:
            if tte_minutes >= threshold:
                return mult

        # Return the last (highest) multiplier if TTE is below all thresholds
        return self.config.tte_multipliers[-1][1] if self.config.tte_multipliers else 1.0

    def _get_moneyness_multiplier(self, mid: float) -> float:
        """
        Calculate moneyness-based spread multiplier.

        Spreads are tighter at ATM (mid~0.5) and wider at ITM/OTM.

        Args:
            mid: Current mid price

        Returns:
            Spread multiplier based on moneyness
        """
        # Distance from ATM center (0.5)
        distance = abs(mid - self.config.atm_center)

        # Normalize distance to [0, 1] range
        normalized_distance = min(1.0, distance / self.config.moneyness_range)

        # Interpolate between base_mult (at ATM) and edge_mult (at extremes)
        mult = (
            self.config.moneyness_base_mult
            + (self.config.moneyness_edge_mult - self.config.moneyness_base_mult)
            * normalized_distance
        )

        return mult

    def _compute_inventory_skew(self, state: StrategyState) -> float:
        """
        Compute inventory-based quote skew.

        Positive position -> shift quotes up (encourage sells)
        Negative position -> shift quotes down (encourage buys)

        Args:
            state: Current strategy state

        Returns:
            Quote skew in price units
        """
        position = state.current_position
        equity = state.total_equity

        if equity <= 0:
            return 0.0

        # Calculate max position
        max_position = equity * self.config.max_position_pct
        if max_position <= 0:
            return 0.0

        # Position ratio [-1, 1]
        position_ratio = _clamp(position / max_position, -1.0, 1.0)

        # Convert max skew from bps to price units
        max_skew = self.config.max_inventory_skew_bps / 10000.0

        # Skew is positive for long positions (shift quotes up to sell)
        return position_ratio * max_skew

    def _enforce_maker(
        self, bid: float, ask: float, state: StrategyState
    ) -> tuple[float, float]:
        """
        Adjust quotes to avoid crossing the spread (maker only).

        Args:
            bid: Current bid price
            ask: Current ask price
            state: Strategy state with market BBO

        Returns:
            Adjusted (bid, ask) tuple
        """
        market_bid = state.best_bid
        market_ask = state.best_ask

        # Don't cross the market ask with our bid
        if market_ask > 0 and bid >= market_ask:
            tick = get_tick_size(market_ask)
            bid = market_ask - tick * self.config.maker_offset_ticks

        # Don't cross the market bid with our ask
        if market_bid > 0 and ask <= market_bid:
            tick = get_tick_size(market_bid)
            ask = market_bid + tick * self.config.maker_offset_ticks

        return bid, ask

    def _compute_confidence(self, tte_minutes: float, state: StrategyState) -> float:
        """
        Compute quote confidence based on TTE and position.

        Lower confidence when:
        - TTE is very low (near expiry)
        - Position is large

        Args:
            tte_minutes: Time to expiry in minutes
            state: Strategy state

        Returns:
            Confidence score [0, 1]
        """
        confidence = 1.0

        # TTE factor: reduce confidence near expiry
        if tte_minutes < 1:
            confidence *= 0.5  # Very low TTE
        elif tte_minutes < 3:
            confidence *= 0.7  # Low TTE

        # Position factor: reduce confidence with large position
        if state.total_equity > 0:
            max_position = state.total_equity * self.config.max_position_pct
            if max_position > 0:
                position_ratio = abs(state.current_position) / max_position
                confidence *= 1.0 - 0.3 * min(1.0, position_ratio)

        return max(0.1, confidence)
