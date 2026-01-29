"""
Z-Spread Quote Model

Uses z-score based spot shifting for spread calculation.
More sophisticated and volatility-aware than simple percentage-based spreads.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import t as t_dist  # type: ignore[import-untyped]

from strategy.models.base import NormalizationConfig, QuoteModel, QuoteResult, StrategyState
from strategy.utils.black_scholes import bs_binary_call
from strategy.utils.polymarket import get_tick_size
from strategy.utils.volatility import SECONDS_PER_YEAR, PriceHistory


def _t_cdf(x: float, df: float) -> float:
    """Student's t-distribution CDF."""
    return float(t_dist.cdf(x, df))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute Black-Scholes d2 term."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = np.sqrt(T)
    return float((np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * sqrt_T))


def bs_binary_call_t(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    df: float,
) -> float:
    """
    Price a binary call option using t-distribution instead of normal.

    Uses the d2 formula but replaces norm_cdf with t_cdf, scaling
    for t-distribution variance.
    """
    if T <= 0:
        return 1.0 if S >= K else 0.0
    if sigma <= 0:
        future_S = S * np.exp(r * T)
        return np.exp(-r * T) if future_S >= K else 0.0
    if S <= 0 or K <= 0:
        return 0.0

    d2_val = _d2(S, K, T, r, sigma)

    # Scale for t-distribution variance: Var(t) = df/(df-2) for df > 2
    scale = np.sqrt((df - 2) / df) if df > 2 else 1.0
    scaled_d2 = float(np.clip(d2_val * scale, -8.0, 8.0))

    return float(np.exp(-r * T)) * _t_cdf(scaled_d2, df)


@dataclass
class ZSpreadQuoteConfig:
    """Configuration for Z-spread quote model."""

    # Z-spread parameter (number of standard deviations)
    z: float = 1.0

    # Distribution parameters
    distribution: Literal["normal", "t"] = "t"
    t_df: float = 3.0  # Degrees of freedom (heavier tails than normal)

    # Volatility parameters
    vol_mode: Literal["iv", "rv", "max", "min"] = "rv"
    vol_floor: float = 0.10  # Minimum volatility - prevents near-zero vol issues
    implied_volatility: float = 0.5  # IV to use when vol_mode="iv" (fallback)
    iv_symbol: str | None = None  # Symbol for live IV lookup (e.g., "BTCUSDT")

    # Time parameters
    tau_seconds: float = 0.1  # Unhedgeable horizon (time to order replacement)

    # Market parameters
    strike: float = 0.5  # Option strike price
    time_to_expiry_years: float = 0.1  # T in years (for BS pricing, used as fallback)
    reference_price_symbol: str | None = None  # External price symbol

    # Real-time expiry calculation (overrides time_to_expiry_years if set)
    end_ts_ms: int | None = None  # Market end timestamp in milliseconds

    # Maker enforcement
    enforce_maker: bool = True  # Adjust to avoid crossing spread
    maker_offset_ticks: int = 1  # Ticks to offset from BBO

    # Price history parameters
    price_history_max_size: int = 1000
    price_history_max_age_seconds: float = 300.0  # 5 minutes


class ZSpreadQuoteModel(QuoteModel):
    """
    Z-Spread Quote Model.

    Uses z-score based spot shifting for spread calculation. The spread is
    derived from the actual option price difference at shifted spot levels,
    naturally producing wider spreads when:
    - Higher volatility (σ is larger)
    - Spot is near strike (gamma is highest)
    - Higher z parameter

    Algorithm:
    1. Calculate spot shift: shift = z × σ × √tau (tau = order replacement time)
    2. Calculate shifted spot prices: upper = S*(1+shift), lower = S*(1-shift)
    3. Price binary option at each shifted spot using T (time to expiry)
    4. bid = price(lower_spot), ask = price(upper_spot)
    """

    def __init__(
        self,
        config: ZSpreadQuoteConfig | None = None,
        normalization: NormalizationConfig | None = None,
    ):
        super().__init__(normalization)
        self.config = config or ZSpreadQuoteConfig()
        self.price_history = PriceHistory(
            max_size=self.config.price_history_max_size,
            max_age_ns=int(self.config.price_history_max_age_seconds * 1e9),
        )

    def copy(self) -> "ZSpreadQuoteModel":
        """Create a deep copy of this model."""
        new = ZSpreadQuoteModel(config=self.config, normalization=self._norm_config)
        new.price_history = self.price_history.copy()
        return new

    def reset(self) -> None:
        """Reset model to initial state by clearing price history."""
        self.price_history.clear()

    def _get_time_to_expiry(self, state: StrategyState) -> float | None:
        """
        Get time to expiry in years.

        If end_ts_ms is set, calculates T in real-time from market timestamp.
        Otherwise falls back to static time_to_expiry_years.

        Returns:
            Time to expiry in years, or None if market has expired.
        """
        if self.config.end_ts_ms is not None:
            # Real-time T calculation from market end timestamp
            now_ms = state.market.timestamp_ns // 1_000_000
            remaining_ms = self.config.end_ts_ms - now_ms
            if remaining_ms <= 0:
                return None  # Expired
            return remaining_ms / 1000 / SECONDS_PER_YEAR
        else:
            # Static T from config
            return self.config.time_to_expiry_years

    def compute_raw(self, state: StrategyState) -> QuoteResult:
        """Compute raw bid and ask prices using Z-spread model."""
        # Get spot price
        spot = self._get_spot_price(state)
        if spot <= 0:
            return QuoteResult(
                bid_price=0,
                ask_price=0,
                should_quote=False,
                reason="Invalid spot price",
            )

        # Calculate time to expiry
        T = self._get_time_to_expiry(state)
        if T is None or T < 1e-8:
            return QuoteResult(
                bid_price=0,
                ask_price=0,
                should_quote=False,
                reason="Market expired",
            )

        # Update price history and compute volatility
        timestamp_ns = state.market.timestamp_ns
        self.price_history.add(spot, timestamp_ns)
        vol = self._get_volatility(state, T)

        # Calculate spot shift using tau (order replacement time)
        # shift = z × σ × √tau
        tau_years = self.config.tau_seconds / SECONDS_PER_YEAR
        shift = self.config.z * vol * np.sqrt(tau_years)

        # Calculate shifted spot prices
        upper_spot = spot * (1 + shift)
        lower_spot = spot * (1 - shift)
        lower_spot = max(lower_spot, spot * 0.001)  # Ensure positive

        # Price binary option at each spot level using T (time to expiry)
        bid = self._price_binary(lower_spot, vol, T)
        ask = self._price_binary(upper_spot, vol, T)

        # Only prevent negative prices from numerical issues
        # Quote price clamping is handled by normalize_quote()
        bid = max(0.0, bid)
        ask = max(0.0, ask)

        # Maker enforcement
        if self.config.enforce_maker:
            bid, ask = self._enforce_maker(bid, ask, state)

        return QuoteResult(bid_price=bid, ask_price=ask)

    def _get_spot_price(self, state: StrategyState) -> float:
        """Get spot price from external source or market mid."""
        if self.config.reference_price_symbol:
            ext_price = state.get_external_price(self.config.reference_price_symbol)
            if ext_price is not None:
                return ext_price
        return state.mid_price

    def _get_volatility(self, state: StrategyState, T: float) -> float:
        """Get volatility based on configured mode.

        Args:
            state: Current strategy state (for live IV lookup)
            T: Time to expiry in years (for IV interpolation)
        """
        rv = self.price_history.compute_volatility()

        # Get IV: try live IV from SHM first, fall back to config
        iv = self.config.implied_volatility
        if self.config.iv_symbol:
            # Convert T from years to days for interpolation
            tte_days = T * 365.0
            live_iv = state.get_interpolated_iv(self.config.iv_symbol, tte_days)
            if live_iv is not None and live_iv > 0:
                iv = live_iv

        if self.config.vol_mode == "iv":
            vol = iv
        elif self.config.vol_mode == "rv":
            vol = rv if rv > 0 else iv
        elif self.config.vol_mode == "max":
            vol = max(iv, rv) if rv > 0 else iv
        else:  # min
            vol = min(iv, rv) if rv > 0 else iv

        return max(self.config.vol_floor, vol)

    def _price_binary(self, spot: float, vol: float, T: float) -> float:
        """Price binary call option using configured distribution."""
        K = self.config.strike

        if self.config.distribution == "t":
            price = bs_binary_call_t(S=spot, K=K, T=T, r=0.0, sigma=vol, df=self.config.t_df)
        else:
            price = bs_binary_call(S=spot, K=K, T=T, r=0.0, sigma=vol)

        return max(0.0, price)  # Let normalize_quote handle boundary clamping

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
