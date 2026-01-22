"""
TPSL-BS Quote Model

Take-Profit/Stop-Loss Black-Scholes quote model with:
- Single z-score (no min/max interpolation)
- Separate TP and SL tick offsets
- Market-based P/L determination (uses market_bid/ask vs avg_price)
- Separate long/short position tracking with individual avg prices
- TP floor/ceiling protection during interpolation
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from strategy.models.base import NormalizationConfig, QuoteModel, QuoteResult, StrategyState
from strategy.utils.black_scholes import bs_binary_call
from strategy.utils.polymarket import MAX_PRICE, MIN_PRICE, get_tick_size
from strategy.utils.volatility import SECONDS_PER_YEAR, PriceHistory

# Numerical thresholds
SIGMA_SPOT_EPSILON = 1e-8
POSITION_RATIO_EPSILON = 1e-6

# Confidence parameters
CONFIDENCE_FLOOR = 0.1
VOL_QUALITY_PENALTY = 0.8
POSITION_CONFIDENCE_REDUCTION = 0.3
HISTORY_MIN_PENALTY = 0.5
HISTORY_LOW_PENALTY = 0.8
HISTORY_MIN_SIZE = 10
HISTORY_LOW_SIZE = 50


def clamp_price(price: float) -> float:
    """Clamp price to valid quote range [0.01, 0.99]."""
    return max(MIN_PRICE, min(MAX_PRICE, price))


@dataclass
class TpslBSQuoteConfig:
    """Configuration for TPSL-BS quote model."""

    # Z-score parameter (single z, not min/max)
    z: float = 4.0  # Z-score for base spread

    # Take-profit/Stop-loss tick offsets
    tp_ticks: int = 2  # Take-profit offset in ticks
    sl_ticks: int = 3  # Stop-loss offset in ticks

    # Position parameters
    max_position_pct: float = 0.20  # Max position as % of balance

    # Time parameters
    tau_seconds: float = 5.0  # Unhedgeable horizon in seconds

    # Volatility parameters
    vol_mode: Literal["iv", "rv", "max", "min"] = "max"
    vol_floor: float = 0.10  # Minimum volatility (10%)
    vol_cap: float = 3.0  # Maximum volatility (300%)
    implied_volatility: float = 0.5  # IV to use when vol_mode="iv"

    # Spread parameters
    min_spread: float = 0.01  # Minimum bid-ask spread

    # Maker enforcement
    enforce_maker: bool = True  # Adjust to avoid crossing spread
    maker_offset_ticks: int = 1  # Ticks to offset from BBO

    # Market parameters
    strike: float = 0.5  # Option strike price
    time_to_expiry_years: float = 0.1  # T in years
    reference_price_symbol: str | None = None  # External price symbol

    # Price history parameters
    price_history_max_size: int = 1000
    price_history_max_age_seconds: float = 300.0  # 5 minutes


class TpslBSQuoteModel(QuoteModel):
    """
    TPSL-BS Quote Model.

    Uses single z-score Black-Scholes binary option pricing with
    separate take-profit/stop-loss logic for long and short positions.

    Key differences from TpBSQuoteModel:
    - Single z-score (not min/max interpolation)
    - Separate TP and SL tick offsets
    - Market-based P/L determination (uses market_bid/ask vs avg_price)
    - Separate long/short position tracking with individual avg prices
    - TP floor/ceiling protection during interpolation

    Position logic:
    - LONG position (UP token): Want to SELL (ASK) to exit
      - Profit: market_bid >= up_avg_price
      - Loss: market_bid < up_avg_price
    - SHORT position (DOWN token): Want to BUY (BID) to exit
      - Profit: market_ask <= short_avg_price (where short_avg_price = 1 - down_avg_price)
      - Loss: market_ask > short_avg_price
    """

    def __init__(
        self,
        config: TpslBSQuoteConfig | None = None,
        normalization: NormalizationConfig | None = None,
    ):
        super().__init__(normalization)
        self.config = config or TpslBSQuoteConfig()
        self.price_history = PriceHistory(
            max_size=self.config.price_history_max_size,
            max_age_ns=int(self.config.price_history_max_age_seconds * 1e9),
        )

        # Internal position state for separate long/short tracking
        self._long_position: float = 0.0  # UP token position
        self._up_avg_price: float = 0.0  # Average entry price for UP token

        self._short_position: float = 0.0  # DOWN token position (positive = short UP)
        self._down_avg_price: float = 0.0  # Average entry price for DOWN token

    def copy(self) -> "TpslBSQuoteModel":
        """
        Create a deep copy of this model.

        Thread-safe: the returned copy is independent of the original.
        Required for parallel backtesting with ProcessPoolExecutor.

        Returns:
            A new TpslBSQuoteModel with copied state
        """
        new = TpslBSQuoteModel(config=self.config, normalization=self._norm_config)
        new.price_history = self.price_history.copy()
        new._long_position = self._long_position
        new._up_avg_price = self._up_avg_price
        new._short_position = self._short_position
        new._down_avg_price = self._down_avg_price
        return new

    def reset(self) -> None:
        """Reset model to initial state."""
        self.price_history.clear()
        self._long_position = 0.0
        self._up_avg_price = 0.0
        self._short_position = 0.0
        self._down_avg_price = 0.0

    def update(
        self,
        state: StrategyState,
        result: QuoteResult,
        fill_info: dict | None = None,
    ) -> None:
        """
        Update model state after a fill.

        Updates internal position tracking for separate long/short positions.

        Args:
            state: Strategy state used for decision
            result: The quote result that was produced
            fill_info: Information about the fill with keys:
                - side: 1 for buy, -1 for sell
                - price: Fill price
                - size: Fill size
        """
        if fill_info is None:
            return

        side = fill_info.get("side", 0)
        price = fill_info.get("price", 0.0)
        size = fill_info.get("size", 0.0)

        if side == 0 or price <= 0 or size <= 0:
            return

        if side == 1:  # BUY - increasing long or decreasing short
            if self._short_position > 0:
                # Closing short position first
                close_size = min(size, self._short_position)
                self._short_position -= close_size
                size -= close_size

                # If short fully closed, reset down avg price
                if self._short_position <= 0:
                    self._short_position = 0.0
                    self._down_avg_price = 0.0

            if size > 0:
                # Opening/adding to long position
                if self._long_position == 0:
                    self._up_avg_price = price
                else:
                    # Weighted average
                    total_value = self._long_position * self._up_avg_price + size * price
                    self._long_position += size
                    self._up_avg_price = total_value / self._long_position
                    return
                self._long_position += size

        elif side == -1:  # SELL - decreasing long or increasing short
            if self._long_position > 0:
                # Closing long position first
                close_size = min(size, self._long_position)
                self._long_position -= close_size
                size -= close_size

                # If long fully closed, reset up avg price
                if self._long_position <= 0:
                    self._long_position = 0.0
                    self._up_avg_price = 0.0

            if size > 0:
                # Opening/adding to short position
                if self._short_position == 0:
                    self._down_avg_price = price
                else:
                    # Weighted average
                    total_value = self._short_position * self._down_avg_price + size * price
                    self._short_position += size
                    self._down_avg_price = total_value / self._short_position
                    return
                self._short_position += size

    def compute_raw(self, state: StrategyState) -> QuoteResult:
        """
        Compute raw bid and ask prices using TPSL-BS model.

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
        vol = self._get_volatility()

        # Step 3: Compute sigma_spot
        tau_years = self.config.tau_seconds / SECONDS_PER_YEAR
        sigma_spot = spot * vol * np.sqrt(tau_years)

        # Step 4: Zero volatility fallback
        if sigma_spot < SIGMA_SPOT_EPSILON:
            half_spread = self.config.min_spread / 2
            return QuoteResult(
                bid_price=max(MIN_PRICE, spot - half_spread),
                ask_price=min(MAX_PRICE, spot + half_spread),
                confidence=0.5,
                reason="Zero volatility fallback",
            )

        # Step 5: Compute z-based quotes (neutral position baseline)
        z_bid, z_ask = self._compute_z_quotes(spot, sigma_spot, vol)

        # Step 6: Get market prices for P/L determination
        market_bid = state.best_bid if state.best_bid > 0 else spot
        market_ask = state.best_ask if state.best_ask > 0 else spot

        # Step 7: Compute final bid/ask with TPSL logic
        bid = self._compute_tpsl_bid(z_bid, market_ask, state)
        ask = self._compute_tpsl_ask(z_ask, market_bid, state)

        # Note: Maker enforcement is applied in compute() after normalization

        # Compute confidence
        position_ratio = self._get_position_ratio(state)
        confidence = self._compute_confidence(vol, position_ratio)

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
        # Get normalized result from base class
        result = super().compute(state)

        if not result.should_quote:
            return result

        # Apply maker enforcement after normalization
        bid, ask = result.bid_price, result.ask_price
        if self.config.enforce_maker:
            bid, ask = self._enforce_maker(bid, ask, state)

        return QuoteResult(
            bid_price=bid,
            ask_price=ask,
            confidence=result.confidence,
            should_quote=result.should_quote,
            reason=result.reason,
        )

    def _get_spot_price(self, state: StrategyState) -> float:
        """Get spot price from external source (Chainlink/RTDS). Required for BS."""
        # Try specific reference symbol first
        if self.config.reference_price_symbol:
            ext_price = state.get_external_price(self.config.reference_price_symbol)
            if ext_price is not None:
                return ext_price

        # Auto-detect: use first available external price
        if state.external_prices:
            for symbol, price_state in state.external_prices.items():
                if price_state.price > 0:
                    return price_state.price

        # No fallback to mid_price - BS requires real spot
        return 0.0

    def _get_volatility(self) -> float:
        """Get volatility based on configured mode."""
        rv = self.price_history.compute_volatility()
        iv = self.config.implied_volatility

        if self.config.vol_mode == "iv":
            vol = iv
        elif self.config.vol_mode == "rv":
            vol = rv if rv > 0 else iv
        elif self.config.vol_mode == "max":
            vol = max(iv, rv) if rv > 0 else iv
        else:  # min
            vol = min(iv, rv) if rv > 0 else iv

        return max(self.config.vol_floor, min(self.config.vol_cap, vol))

    def _get_position_ratio(self, state: StrategyState) -> float:
        """Calculate position ratio (0 to 1) for the dominant side."""
        max_position = state.total_equity * self.config.max_position_pct
        if max_position <= 0:
            return 0.0

        # Use absolute position to handle both LONG and SHORT
        return min(1.0, abs(state.current_position) / max_position)

    def _get_long_position_ratio(self, state: StrategyState) -> float:
        """Calculate LONG position ratio (0 to 1) based on internal tracking."""
        max_position = state.total_equity * self.config.max_position_pct
        if max_position <= 0:
            return 0.0

        return min(1.0, self._long_position / max_position)

    def _get_short_position_ratio(self, state: StrategyState) -> float:
        """Calculate SHORT position ratio (0 to 1) based on internal tracking."""
        max_position = state.total_equity * self.config.max_position_pct
        if max_position <= 0:
            return 0.0

        return min(1.0, self._short_position / max_position)

    def _compute_z_quotes(
        self,
        spot: float,
        sigma_spot: float,
        vol: float,
    ) -> tuple[float, float]:
        """
        Compute z-score based bid/ask for neutral position.

        Args:
            spot: Current spot price
            sigma_spot: Spot volatility (spot * vol * sqrt(tau))
            vol: Annualized volatility

        Returns:
            Tuple of (z_bid, z_ask)
        """
        z = self.config.z

        # Compute spot adjustments
        spot_up = spot + z * sigma_spot
        spot_down = spot - z * sigma_spot

        # BS binary call prices
        z_ask = clamp_price(bs_binary_call(
            S=spot_up,
            K=self.config.strike,
            T=self.config.time_to_expiry_years,
            r=0.0,
            sigma=vol,
        ))

        z_bid = clamp_price(bs_binary_call(
            S=spot_down,
            K=self.config.strike,
            T=self.config.time_to_expiry_years,
            r=0.0,
            sigma=vol,
        ))

        return z_bid, z_ask

    def _compute_tpsl_ask(
        self,
        z_ask: float,
        market_bid: float,
        state: StrategyState,
    ) -> float:
        """
        Compute ASK price with TP/SL logic for closing LONG positions.

        For LONG position:
        - Profit: market_bid >= up_avg_price
          - closest_ask = max(tp_price, mkt_price)
          - TP floor protection: if z_ask >= closest_ask, interpolate; else use closest_ask
        - Loss: market_bid < up_avg_price
          - closest_ask = min(sl_price, mkt_price)
          - Interpolate towards closest_ask based on position_ratio

        Args:
            z_ask: Z-score based ask price
            market_bid: Current market bid price
            state: Strategy state

        Returns:
            Final ask price
        """
        # No long position - use z_ask
        if self._long_position < POSITION_RATIO_EPSILON:
            return z_ask

        position_ratio = self._get_long_position_ratio(state)
        tick = get_tick_size(self._up_avg_price)

        if market_bid >= self._up_avg_price:
            # PROFIT scenario
            tp_price = self._up_avg_price + tick * self.config.tp_ticks
            mkt_price = market_bid + tick  # One tick above market bid
            closest_ask = max(tp_price, mkt_price)

            # Always use quote that's further from mid to protect profits
            if z_ask <= closest_ask:
                # z_ask is closer to mid (lower), but we want to protect profits
                # Use closest_ask instead (the further one)
                final_ask = closest_ask
            else:
                # z_ask is further from mid, interpolate from z_ask toward closest_ask
                final_ask = closest_ask + (z_ask - closest_ask) * (1 - position_ratio)
        else:
            # LOSS scenario
            sl_price = self._up_avg_price - tick * self.config.sl_ticks
            mkt_price = market_bid + tick  # One tick above market bid
            closest_ask = min(sl_price, mkt_price)

            # Interpolate towards SL to exit losing position
            final_ask = closest_ask + (z_ask - closest_ask) * (1 - position_ratio)

        return clamp_price(final_ask)

    def _compute_tpsl_bid(
        self,
        z_bid: float,
        market_ask: float,
        state: StrategyState,
    ) -> float:
        """
        Compute BID price with TP/SL logic for closing SHORT positions.

        For SHORT position (we sold UP, want to buy back):
        - short_avg_price = 1 - down_avg_price (the UP price we sold at)
        - Profit: market_ask <= short_avg_price
          - closest_bid = min(tp_price, mkt_price)
          - TP ceiling protection: if z_bid <= closest_bid, interpolate; else use closest_bid
        - Loss: market_ask > short_avg_price
          - closest_bid = max(sl_price, mkt_price)
          - Interpolate towards closest_bid based on position_ratio

        Args:
            z_bid: Z-score based bid price
            market_ask: Current market ask price
            state: Strategy state

        Returns:
            Final bid price
        """
        # No short position - use z_bid
        if self._short_position < POSITION_RATIO_EPSILON:
            return z_bid

        position_ratio = self._get_short_position_ratio(state)

        # For shorts, we sold at down_avg_price which corresponds to UP price
        # In binary markets: if we sold UP at 0.6, we need to think of our
        # short entry as 0.6 on the UP side
        short_avg_price = self._down_avg_price
        tick = get_tick_size(short_avg_price)

        if market_ask <= short_avg_price:
            # PROFIT scenario (can buy back cheaper)
            tp_price = short_avg_price - tick * self.config.tp_ticks
            mkt_price = market_ask - tick  # One tick below market ask
            closest_bid = min(tp_price, mkt_price)

            # Always use quote that's further from mid to protect profits
            if z_bid >= closest_bid:
                # z_bid is closer to mid (higher), but we want to protect profits
                # Use closest_bid instead (the further one)
                final_bid = closest_bid
            else:
                # z_bid is further from mid, interpolate from z_bid toward closest_bid
                final_bid = closest_bid + (z_bid - closest_bid) * (1 - position_ratio)
        else:
            # LOSS scenario (price moved up)
            sl_price = short_avg_price + tick * self.config.sl_ticks
            mkt_price = market_ask - tick  # One tick below market ask
            closest_bid = max(sl_price, mkt_price)

            # Interpolate towards SL to exit losing position
            final_bid = closest_bid + (z_bid - closest_bid) * (1 - position_ratio)

        return clamp_price(final_bid)

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
            # Use tick size at target price (market_ask), not original bid
            tick = get_tick_size(market_ask)
            bid = market_ask - tick * self.config.maker_offset_ticks

        if market_bid > 0 and ask <= market_bid:
            # Use tick size at target price (market_bid), not original ask
            tick = get_tick_size(market_bid)
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
            confidence *= VOL_QUALITY_PENALTY

        # Position factor
        confidence *= 1.0 - POSITION_CONFIDENCE_REDUCTION * position_ratio

        # History size factor
        history_size = len(self.price_history)
        if history_size < HISTORY_MIN_SIZE:
            confidence *= HISTORY_MIN_PENALTY
        elif history_size < HISTORY_LOW_SIZE:
            confidence *= HISTORY_LOW_PENALTY

        return max(CONFIDENCE_FLOOR, confidence)
