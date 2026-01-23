"""
Volatility Calculation Utilities

Functions for computing realized volatility from price history.
Optimized for high-frequency backtesting with O(1) volatility updates.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Constants
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
NANOSECONDS_PER_SECOND = 1_000_000_000


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Compute log returns from price series.

    Args:
        prices: Array of prices

    Returns:
        Array of log returns (length = len(prices) - 1)
    """
    if len(prices) < 2:
        return np.array([])
    return np.diff(np.log(prices))


def compute_realized_volatility(
    prices: np.ndarray,
    timestamps_ns: np.ndarray,
) -> float:
    """
    Compute annualized realized volatility from price and timestamp series.

    Uses the formula:
        rv = std(log_returns) / sqrt(avg_dt) * sqrt(seconds_per_year)

    This annualizes the volatility by scaling based on the average time
    between observations.

    Args:
        prices: Array of prices
        timestamps_ns: Array of timestamps in nanoseconds

    Returns:
        Annualized realized volatility (0 if insufficient data)
    """
    if len(prices) < 2 or len(timestamps_ns) < 2:
        return 0.0

    # Compute log returns
    log_returns = compute_log_returns(prices)
    if len(log_returns) == 0:
        return 0.0

    # Compute time deltas in seconds
    dt_ns = np.diff(timestamps_ns)
    dt_seconds = dt_ns / NANOSECONDS_PER_SECOND

    # Filter out zero or negative time deltas
    valid_mask = dt_seconds > 0
    if not np.any(valid_mask):
        return 0.0

    valid_returns = log_returns[valid_mask]
    valid_dt = dt_seconds[valid_mask]

    if len(valid_returns) < 2:
        return 0.0

    # Average time delta in seconds
    avg_dt = np.mean(valid_dt)
    if avg_dt <= 0:
        return 0.0

    # Standard deviation of log returns
    std_returns = np.std(valid_returns, ddof=1)

    # Annualize: scale by sqrt(seconds_per_year / avg_dt)
    # This converts from per-observation volatility to annual volatility
    annualized_vol = std_returns * np.sqrt(SECONDS_PER_YEAR / avg_dt)

    return float(annualized_vol)


@dataclass
class PriceHistory:
    """
    Rolling window buffer for price history with O(1) volatility computation.

    Uses a circular numpy buffer with Welford's online algorithm for
    incremental variance updates. Optimized for high-frequency backtesting.

    Performance:
        - add(): O(1) amortized (O(n) only when pruning)
        - compute_volatility(): O(1) (returns cached value)

    Attributes:
        max_size: Maximum number of observations to keep
        max_age_ns: Maximum age of observations in nanoseconds (default 5 min)
    """

    max_size: int = 1000
    max_age_ns: int = 300 * NANOSECONDS_PER_SECOND  # 5 minutes

    # Circular buffer (pre-allocated numpy arrays)
    _prices: np.ndarray = field(default=None, repr=False)
    _timestamps: np.ndarray = field(default=None, repr=False)
    _head: int = field(default=0, repr=False)
    _count: int = field(default=0, repr=False)

    # Welford's online stats (O(1) updates)
    _n_returns: int = field(default=0, repr=False)
    _mean_return: float = field(default=0.0, repr=False)
    _M2: float = field(default=0.0, repr=False)  # Sum of squared deviations
    _sum_dt: float = field(default=0.0, repr=False)  # Sum of time deltas in seconds

    # Cache
    _cached_vol: float = field(default=0.0, repr=False)
    _cache_valid: bool = field(default=False, repr=False)
    _last_price: float = field(default=0.0, repr=False)

    def __post_init__(self):
        # Initialize numpy arrays
        self._prices = np.zeros(self.max_size, dtype=np.float64)
        self._timestamps = np.zeros(self.max_size, dtype=np.int64)
        self._head = 0
        self._count = 0
        self._n_returns = 0
        self._mean_return = 0.0
        self._M2 = 0.0
        self._sum_dt = 0.0
        self._cached_vol = 0.0
        self._cache_valid = False
        self._last_price = 0.0

    def add(self, price: float, timestamp_ns: int) -> None:
        """
        Add a price observation to the history.

        Uses Welford's online algorithm for O(1) variance updates.
        Skips duplicate prices to avoid zero returns.

        Args:
            price: Price value
            timestamp_ns: Timestamp in nanoseconds
        """
        # Skip if price is same as last (avoids zero returns)
        if self._count > 0 and abs(price - self._last_price) < 1e-12:
            return

        # Auto-prune old observations (invalidates cache if pruning occurs)
        pruned = self._prune_old(timestamp_ns)

        # If we have existing data, compute the new return and update Welford stats
        if self._count > 0:
            prev_idx = (self._head - 1) % self.max_size
            prev_price = self._prices[prev_idx]
            prev_ts = self._timestamps[prev_idx]

            if prev_price > 0 and price > 0:
                log_return = np.log(price / prev_price)
                dt_seconds = (timestamp_ns - prev_ts) / NANOSECONDS_PER_SECOND

                if dt_seconds > 0:
                    self._welford_add(log_return, dt_seconds)
                    self._cache_valid = False  # Invalidate cache

        # Check if buffer is full (need to remove oldest and adjust stats)
        if self._count == self.max_size:
            self._remove_oldest_from_stats()

        # Add new observation to circular buffer
        self._prices[self._head] = price
        self._timestamps[self._head] = timestamp_ns
        self._head = (self._head + 1) % self.max_size
        self._count = min(self._count + 1, self.max_size)
        self._last_price = price

    def _welford_add(self, log_return: float, dt_seconds: float) -> None:
        """Add a return to Welford's online stats."""
        self._n_returns += 1
        delta = log_return - self._mean_return
        self._mean_return += delta / self._n_returns
        delta2 = log_return - self._mean_return
        self._M2 += delta * delta2
        self._sum_dt += dt_seconds

    def _remove_oldest_from_stats(self) -> None:
        """
        Remove the oldest return from Welford stats when buffer is full.
        This is an approximation - we recompute stats periodically for accuracy.
        """
        if self._n_returns > 0:
            # Get the oldest return that will be removed
            oldest_idx = self._head  # This is where the new value will go
            oldest_price = self._prices[oldest_idx]
            next_idx = (oldest_idx + 1) % self.max_size

            if next_idx != self._head and self._count > 1:
                next_price = self._prices[next_idx]
                oldest_ts = self._timestamps[oldest_idx]
                next_ts = self._timestamps[next_idx]

                if oldest_price > 0 and next_price > 0:
                    old_return = np.log(next_price / oldest_price)
                    old_dt = (next_ts - oldest_ts) / NANOSECONDS_PER_SECOND

                    if old_dt > 0 and self._n_returns > 1:
                        # Reverse Welford update (approximate)
                        self._n_returns -= 1
                        delta = old_return - self._mean_return
                        self._mean_return -= delta / self._n_returns if self._n_returns > 0 else 0
                        delta2 = old_return - self._mean_return
                        self._M2 -= delta * delta2
                        self._M2 = max(0, self._M2)  # Prevent negative variance
                        self._sum_dt -= old_dt
                        self._sum_dt = max(0, self._sum_dt)

    def _prune_old(self, current_time_ns: int) -> bool:
        """
        Remove observations older than max_age_ns.
        Returns True if any pruning occurred (requires stats recomputation).
        """
        if self._count == 0:
            return False

        cutoff = current_time_ns - self.max_age_ns
        pruned = False

        # Find oldest entry index
        oldest_idx = (self._head - self._count) % self.max_size

        while self._count > 0:
            if self._timestamps[oldest_idx] >= cutoff:
                break

            # Remove oldest entry
            self._count -= 1
            oldest_idx = (oldest_idx + 1) % self.max_size
            pruned = True

        if pruned:
            # Recompute Welford stats from scratch (expensive but rare)
            self._recompute_stats()

        return pruned

    def _recompute_stats(self) -> None:
        """Recompute Welford stats from scratch after pruning."""
        self._n_returns = 0
        self._mean_return = 0.0
        self._M2 = 0.0
        self._sum_dt = 0.0
        self._cache_valid = False

        if self._count < 2:
            return

        # Get ordered prices and timestamps
        prices = self.get_prices()
        timestamps = self.get_timestamps()

        # Recompute stats
        for i in range(1, len(prices)):
            if prices[i - 1] > 0 and prices[i] > 0:
                log_return = np.log(prices[i] / prices[i - 1])
                dt_seconds = (timestamps[i] - timestamps[i - 1]) / NANOSECONDS_PER_SECOND
                if dt_seconds > 0:
                    self._welford_add(log_return, dt_seconds)

    def compute_volatility(self) -> float:
        """
        Compute annualized realized volatility from current buffer.

        Returns cached value for O(1) performance.

        Returns:
            Annualized realized volatility, or 0 if insufficient data
        """
        if self._cache_valid:
            return self._cached_vol

        if self._n_returns < 2:
            self._cached_vol = 0.0
            self._cache_valid = True
            return 0.0

        # Compute variance from Welford stats
        variance = self._M2 / (self._n_returns - 1)
        if variance <= 0:
            self._cached_vol = 0.0
            self._cache_valid = True
            return 0.0

        std_returns = np.sqrt(variance)

        # Average time delta
        avg_dt = self._sum_dt / self._n_returns
        if avg_dt <= 0:
            self._cached_vol = 0.0
            self._cache_valid = True
            return 0.0

        # Annualize
        annualized_vol = std_returns * np.sqrt(SECONDS_PER_YEAR / avg_dt)

        self._cached_vol = float(annualized_vol)
        self._cache_valid = True
        return self._cached_vol

    def get_prices(self) -> np.ndarray:
        """Get current price buffer as numpy array (ordered oldest to newest)."""
        if self._count == 0:
            return np.array([])

        start_idx = (self._head - self._count) % self.max_size
        if start_idx + self._count <= self.max_size:
            return self._prices[start_idx : start_idx + self._count].copy()
        else:
            # Wrap around
            first_part = self._prices[start_idx:]
            second_part = self._prices[: self._head]
            return np.concatenate([first_part, second_part])

    def get_timestamps(self) -> np.ndarray:
        """Get current timestamp buffer as numpy array (ordered oldest to newest)."""
        if self._count == 0:
            return np.array([])

        start_idx = (self._head - self._count) % self.max_size
        if start_idx + self._count <= self.max_size:
            return self._timestamps[start_idx : start_idx + self._count].copy()
        else:
            # Wrap around
            first_part = self._timestamps[start_idx:]
            second_part = self._timestamps[: self._head]
            return np.concatenate([first_part, second_part])

    def __len__(self) -> int:
        """Return number of observations in buffer."""
        return self._count

    def clear(self) -> None:
        """Clear all observations and reset stats."""
        self._head = 0
        self._count = 0
        self._n_returns = 0
        self._mean_return = 0.0
        self._M2 = 0.0
        self._sum_dt = 0.0
        self._cached_vol = 0.0
        self._cache_valid = False
        self._last_price = 0.0

    def copy(self) -> "PriceHistory":
        """
        Create a deep copy of this PriceHistory.

        Thread-safe: the returned copy is independent of the original.

        Returns:
            A new PriceHistory with copied state
        """
        new = PriceHistory.__new__(PriceHistory)
        new.max_size = self.max_size
        new.max_age_ns = self.max_age_ns
        new._prices = self._prices.copy()
        new._timestamps = self._timestamps.copy()
        new._head = self._head
        new._count = self._count
        new._n_returns = self._n_returns
        new._mean_return = self._mean_return
        new._M2 = self._M2
        new._sum_dt = self._sum_dt
        new._cached_vol = self._cached_vol
        new._cache_valid = self._cache_valid
        new._last_price = self._last_price
        return new
