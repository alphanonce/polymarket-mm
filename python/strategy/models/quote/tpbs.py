"""
Take-Profit Black-Scholes Quote Model

Z-score-based Black-Scholes binary option pricing with position-aware
take-profit/stop-loss logic for Polymarket market-making.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from strategy.models.base import NormalizationConfig, QuoteModel, QuoteResult, StrategyState
from strategy.utils.black_scholes import bs_binary_call
from strategy.utils.polymarket import get_tick_size
from strategy.utils.volatility import SECONDS_PER_YEAR, PriceHistory


@dataclass
class TpBSQuoteConfig:
    """Configuration for TpBS quote model."""

    # Z-score parameters
    max_z: float = 4.0  # Z at neutral position (wide spread)
    min_z: float = 2.0  # Z at full position (tight spread for exit)

    # Position parameters
    max_position_pct: float = 0.20  # Max position as % of balance

    # Time parameters
    tau_seconds: float = 5.0  # Unhedgeable horizon in seconds

    # Volatility parameters
    vol_mode: Literal["iv", "rv", "max", "min"] = "max"
    vol_floor: float = 0.10  # Minimum volatility (10%)
    vol_cap: float = 3.0  # Maximum volatility (300%)
    implied_volatility: float = 0.5  # IV to use when vol_mode="iv" (fallback)
    iv_symbol: str | None = None  # Symbol for live IV lookup (e.g., "BTCUSDT")

    # Take-profit parameters
    tp_ticks: int = 2  # Ticks for take-profit offset

    # Spread parameters
    min_spread: float = 0.01  # Minimum bid-ask spread

    # Maker enforcement
    enforce_maker: bool = True  # Adjust to avoid crossing spread
    maker_offset_ticks: int = 1  # Ticks to offset from BBO

    # Market parameters
    strike: float = 0.5  # Option strike price
    time_to_expiry_years: float = 0.1  # T in years
    reference_price_symbol: str | None = None  # External price symbol

    # Real-time expiry calculation (overrides time_to_expiry_years if set)
    end_ts_ms: int | None = None  # Market end timestamp in milliseconds

    # Price history parameters
    price_history_max_size: int = 1000
    price_history_max_age_seconds: float = 300.0  # 5 minutes


class TpBSQuoteModel(QuoteModel):
    """
    Take-Profit Black-Scholes Quote Model.

    Uses z-score-based Black-Scholes binary option pricing to generate quotes.
    Incorporates position-aware take-profit/stop-loss logic for the ask side.

    Key features:
    - BID: Uses InvBS-style z interpolation based on position
    - ASK: Uses take-profit logic when in profit, aggressive z when in loss
    - Supports both realized and implied volatility modes
    - Maker enforcement to avoid crossing BBO
    - Normalization (rounding, clamping) is handled by the base class
    """

    def __init__(
        self,
        config: TpBSQuoteConfig | None = None,
        normalization: NormalizationConfig | None = None,
    ):
        super().__init__(normalization)
        self.config = config or TpBSQuoteConfig()
        self.price_history = PriceHistory(
            max_size=self.config.price_history_max_size,
            max_age_ns=int(self.config.price_history_max_age_seconds * 1e9),
        )

    def copy(self) -> "TpBSQuoteModel":
        """
        Create a deep copy of this model.

        Thread-safe: the returned copy is independent of the original.
        Required for parallel backtesting with ProcessPoolExecutor.

        Returns:
            A new TpBSQuoteModel with copied state
        """
        new = TpBSQuoteModel(config=self.config, normalization=self._norm_config)
        new.price_history = self.price_history.copy()
        return new

    def reset(self) -> None:
        """Reset model to initial state by clearing price history."""
        self.price_history.clear()

    def compute_raw(self, state: StrategyState) -> QuoteResult:
        """
        Compute raw bid and ask prices using TpBS model.

        This returns raw prices before normalization (rounding/clamping).
        The base class compute() method will apply normalization.

        Args:
            state: Current strategy state

        Returns:
            QuoteResult with raw bid/ask prices
        """
        # Step 1: Get spot price
        spot = self._get_spot_price(state)
        if spot <= 0:
            return QuoteResult(
                bid_price=0,
                ask_price=0,
                should_quote=False,
                reason="Invalid spot price",
            )

        # Step 2: Update price history and compute volatility
        timestamp_ns = state.market.timestamp_ns
        self.price_history.add(spot, timestamp_ns)
        vol = self._get_volatility(state)

        # Step 3: Calculate position ratio
        position_ratio = self._get_position_ratio(state)
        long_position = state.current_position

        # Step 4: Compute sigma_spot
        tau_years = self.config.tau_seconds / SECONDS_PER_YEAR
        sigma_spot = spot * vol * np.sqrt(tau_years)

        # Step 4b: Zero volatility fallback
        # When sigma_spot is extremely small, BS pricing becomes unreliable
        if sigma_spot < 1e-8:
            half_spread = self.config.min_spread / 2
            # Let normalize_quote handle boundary clamping
            return QuoteResult(
                bid_price=spot - half_spread,
                ask_price=spot + half_spread,
                confidence=0.5,
                reason="Zero volatility fallback",
            )

        # Step 5: Calculate BID (InvBS-style z interpolation)
        bid = self._compute_bid(spot, sigma_spot, vol, position_ratio)

        # Step 6: Calculate ASK (TP/SL logic)
        ask = self._compute_ask(spot, sigma_spot, vol, position_ratio, long_position, state)

        # Step 7: Apply minimum spread (pre-normalization)
        bid, ask = self._enforce_min_spread(bid, ask)

        # Step 8: Maker enforcement (pre-normalization)
        if self.config.enforce_maker:
            bid, ask = self._enforce_maker(bid, ask, state)

        # Compute confidence
        confidence = self._compute_confidence(vol, position_ratio)

        # Return raw prices - base class will normalize
        return QuoteResult(
            bid_price=bid,
            ask_price=ask,
            confidence=confidence,
            should_quote=True,
        )

    def _get_spot_price(self, state: StrategyState) -> float:
        """Get spot price from external source or market mid."""
        if self.config.reference_price_symbol:
            ext_price = state.get_external_price(self.config.reference_price_symbol)
            if ext_price is not None:
                return ext_price
        return state.mid_price

    def _get_time_to_expiry(self, state: StrategyState) -> float:
        """Get time to expiry in years.

        If end_ts_ms is set, calculates T in real-time from market timestamp.
        Otherwise falls back to static time_to_expiry_years.
        """
        if self.config.end_ts_ms is not None:
            now_ms = state.market.timestamp_ns // 1_000_000
            remaining_ms = self.config.end_ts_ms - now_ms
            if remaining_ms <= 0:
                return 0.0  # Expired
            return remaining_ms / 1000 / SECONDS_PER_YEAR
        return self.config.time_to_expiry_years

    def _get_volatility(self, state: StrategyState) -> float:
        """Get volatility based on configured mode.

        Args:
            state: Current strategy state (for live IV lookup)
        """
        rv = self.price_history.compute_volatility()

        # Get IV: try live IV from SHM first, fall back to config
        iv = self.config.implied_volatility
        if self.config.iv_symbol:
            # Get time to expiry for interpolation
            T = self._get_time_to_expiry(state)
            tte_days = T * 365.0
            live_iv = state.get_interpolated_iv(self.config.iv_symbol, tte_days)
            if live_iv is not None and live_iv > 0:
                iv = live_iv

        if self.config.vol_mode == "iv":
            vol = iv
        elif self.config.vol_mode == "rv":
            vol = rv if rv > 0 else iv  # Fallback to iv if rv is 0
        elif self.config.vol_mode == "max":
            vol = max(iv, rv) if rv > 0 else iv
        else:  # min
            vol = min(iv, rv) if rv > 0 else iv

        # Clamp volatility
        return max(self.config.vol_floor, min(self.config.vol_cap, vol))

    def _get_position_ratio(self, state: StrategyState) -> float:
        """Calculate position ratio (0 to 1)."""
        max_position = state.total_equity * self.config.max_position_pct
        if max_position <= 0:
            return 0.0

        long_position = max(0, state.current_position)
        return min(1.0, long_position / max_position)

    def _compute_bid(
        self,
        spot: float,
        sigma_spot: float,
        vol: float,
        position_ratio: float,
    ) -> float:
        """
        Compute bid price using InvBS-style z interpolation.

        position=0 → bid_z=max_z (wide spread)
        position=max → bid_z=min_z (tight spread for exit)
        """
        bid_z = self.config.max_z - (self.config.max_z - self.config.min_z) * position_ratio
        spot_down = spot - bid_z * sigma_spot
        # Clamp spot to valid range for BS pricing
        spot_down = max(0.001, min(0.999, spot_down))

        price = bs_binary_call(
            S=spot_down,
            K=self.config.strike,
            T=self.config.time_to_expiry_years,
            r=0.0,
            sigma=vol,
        )
        return max(0.0, price)  # Let normalize_quote handle boundary clamping

    def _compute_ask(
        self,
        spot: float,
        sigma_spot: float,
        vol: float,
        position_ratio: float,
        long_position: float,
        state: StrategyState,
    ) -> float:
        """
        Compute ask price with take-profit/stop-loss logic.

        Case A: No position → use max_z (wide)
        Case B: Position in profit → interpolate to take-profit price
        Case C: Position in loss → aggressive z for exit
        """
        # First compute z_ask price (what we'd quote without TP logic)
        z_ask_spot = spot + self.config.max_z * sigma_spot
        # Clamp spot to valid range for BS pricing
        z_ask_spot = max(0.001, min(0.999, z_ask_spot))
        z_ask = bs_binary_call(
            S=z_ask_spot,
            K=self.config.strike,
            T=self.config.time_to_expiry_years,
            r=0.0,
            sigma=vol,
        )
        z_ask = max(0.0, z_ask)  # Let normalize_quote handle boundary clamping

        # Case A: No position
        if position_ratio < 1e-6:
            return z_ask

        # Get average entry price
        avg_entry_price = 0.0
        if state.position:
            avg_entry_price = state.position.avg_entry_price

        # If no valid avg entry, use z_ask
        if avg_entry_price <= 0:
            return z_ask

        # Case B: Position in PROFIT (z_ask >= avg_entry_price)
        if z_ask >= avg_entry_price:
            # Calculate take-profit price
            tick = get_tick_size(avg_entry_price)
            tp_price = avg_entry_price + tick * self.config.tp_ticks

            # Check if TP would cross market bid
            market_bid = state.best_bid
            if market_bid > 0 and tp_price <= market_bid:
                closest_ask = market_bid + tick * self.config.maker_offset_ticks
            else:
                closest_ask = tp_price

            # Interpolate: no position → z_ask, full position → closest_ask
            ask = closest_ask + (z_ask - closest_ask) * (1 - position_ratio)
            return max(0.0, ask)  # Let normalize_quote handle boundary clamping

        # Case C: Position in LOSS (z_ask < avg_entry_price)
        # Use InvBS-style z interpolation (aggressive exit)
        sl_z = self.config.max_z - (self.config.max_z - self.config.min_z) * position_ratio
        sl_spot = spot + sl_z * sigma_spot
        # Clamp spot to valid range for BS pricing
        sl_spot = max(0.001, min(0.999, sl_spot))
        price = bs_binary_call(
            S=sl_spot,
            K=self.config.strike,
            T=self.config.time_to_expiry_years,
            r=0.0,
            sigma=vol,
        )
        return max(0.0, price)  # Let normalize_quote handle boundary clamping

    def _enforce_min_spread(self, bid: float, ask: float) -> tuple[float, float]:
        """Ensure minimum spread between bid and ask."""
        current_spread = ask - bid
        if current_spread < self.config.min_spread:
            half_add = (self.config.min_spread - current_spread) / 2
            ask += half_add
            bid -= half_add
        return bid, ask

    def _enforce_maker(
        self,
        bid: float,
        ask: float,
        state: StrategyState,
    ) -> tuple[float, float]:
        """Adjust quotes to avoid crossing the spread (maker only)."""
        market_bid = state.best_bid
        market_ask = state.best_ask

        if market_ask > 0 and bid >= market_ask:
            tick = get_tick_size(bid)
            bid = market_ask - tick * self.config.maker_offset_ticks

        if market_bid > 0 and ask <= market_bid:
            tick = get_tick_size(ask)
            ask = market_bid + tick * self.config.maker_offset_ticks

        return bid, ask

    def _compute_confidence(self, vol: float, position_ratio: float) -> float:
        """
        Compute quote confidence based on volatility and position.

        Lower confidence when:
        - Volatility is at floor/cap (unreliable estimate)
        - Position is near maximum
        - Price history is sparse
        """
        confidence = 1.0

        # Vol quality factor
        if vol <= self.config.vol_floor or vol >= self.config.vol_cap:
            confidence *= 0.8

        # Position factor
        confidence *= 1.0 - 0.3 * position_ratio

        # History size factor
        history_size = len(self.price_history)
        if history_size < 10:
            confidence *= 0.5
        elif history_size < 50:
            confidence *= 0.8

        return max(0.1, confidence)
