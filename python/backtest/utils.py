"""
Backtest Utilities

Shared utility functions for backtest engines to avoid code duplication.
Includes Numba-optimized functions for performance-critical inner loops.
"""


import numpy as np
import pandas as pd
from numba import njit

# Annualization factor for 24/7 crypto markets
ANNUALIZATION_FACTOR = np.sqrt(365.25 * 24)

# Timestamp magnitude thresholds for auto-detection
# Based on approximate year 2025 values
TS_THRESHOLD_NS = 1e18  # Nanoseconds: ~1.7e18 for 2025
TS_THRESHOLD_MS = 1e15  # Microseconds: ~1.7e15 for 2025
TS_THRESHOLD_S = 1e12   # Milliseconds: ~1.7e12 for 2025

# Conversion multipliers
MS_TO_NS = 1000
US_TO_NS = 1_000_000
S_TO_NS = 1_000_000_000


def compute_unrealized_pnl(
    position: float,
    avg_entry: float,
    current_price: float,
) -> float:
    """
    Calculate unrealized PnL for a position.

    Args:
        position: Current position size (positive = long, negative = short)
        avg_entry: Average entry price
        current_price: Current market price

    Returns:
        Unrealized PnL value
    """
    if position > 0:
        return (current_price - avg_entry) * position
    elif position < 0:
        return (avg_entry - current_price) * (-position)
    return 0.0


def compute_sharpe_ratio(
    pnl_samples: list[float] | np.ndarray,
    annualization: float = ANNUALIZATION_FACTOR,
) -> float:
    """
    Calculate annualized Sharpe ratio from PnL samples.

    Args:
        pnl_samples: List or array of PnL values over time
        annualization: Annualization factor (default: sqrt(365.25*24) for 24/7 markets)

    Returns:
        Sharpe ratio (0 if insufficient data or zero volatility)
    """
    pnl_arr = np.asarray(pnl_samples)
    if len(pnl_arr) <= 1:
        return 0.0

    returns = np.diff(pnl_arr)
    std = np.std(returns)
    if std <= 0:
        return 0.0

    return float(np.mean(returns) / std * annualization)


def compute_max_drawdown(pnl_arr: list[float] | np.ndarray) -> float:
    """
    Calculate maximum drawdown from PnL history.

    Args:
        pnl_arr: Array of cumulative PnL values

    Returns:
        Maximum drawdown value (positive number)
    """
    pnl = np.asarray(pnl_arr)
    if len(pnl) == 0:
        return 0.0

    cummax = np.maximum.accumulate(pnl)
    drawdown = cummax - pnl
    return float(np.max(drawdown))


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Find a column from a list of candidate names.

    Performs case-sensitive search first, then case-insensitive.

    Args:
        df: DataFrame to search
        candidates: List of possible column names

    Returns:
        Matching column name or None
    """
    # Case-sensitive search
    for col in candidates:
        if col in df.columns:
            return col

    # Case-insensitive search
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]

    return None


def detect_timestamp_unit(timestamp: float) -> str:
    """
    Auto-detect timestamp unit based on magnitude.

    Thresholds based on approximate ranges for 2025:
        - Nanoseconds: ~1.7e18
        - Microseconds: ~1.7e15
        - Milliseconds: ~1.7e12
        - Seconds: ~1.7e9

    Args:
        timestamp: Sample timestamp value

    Returns:
        Unit string for pandas: "ns", "us", "ms", or "s"
    """
    if timestamp > TS_THRESHOLD_MS:
        return "ns"
    elif timestamp > TS_THRESHOLD_S:
        return "ms"
    else:
        return "s"


def to_timestamp_ns(val) -> int:
    """
    Convert various timestamp formats to nanoseconds.

    Handles:
        - Nanoseconds (>1e18)
        - Microseconds (>1e15)
        - Milliseconds (>1e12)
        - Seconds (else)
        - pd.Timestamp objects
        - numpy integer/float types

    Args:
        val: Timestamp value in various formats

    Returns:
        Timestamp in nanoseconds (0 if conversion fails)
    """
    if val is None:
        return 0

    # Handle numeric types (including numpy int64, float64, etc.)
    if isinstance(val, (int, float, np.integer, np.floating)):
        num_val = float(val)
        if num_val > TS_THRESHOLD_NS:
            return int(num_val)
        elif num_val > TS_THRESHOLD_MS:
            return int(num_val * MS_TO_NS)
        elif num_val > TS_THRESHOLD_S:
            return int(num_val * US_TO_NS)
        else:
            return int(num_val * S_TO_NS)

    if isinstance(val, pd.Timestamp):
        return int(val.value)

    return 0


class PositionTracker:
    """
    Tracks position state and calculates PnL.

    Handles position updates for both long and short positions,
    including position flips (short→long and long→short).
    """

    def __init__(self):
        self.position: float = 0.0
        self.avg_entry: float = 0.0
        self.realized_pnl: float = 0.0

    def reset(self) -> None:
        """Reset tracker to initial state."""
        self.position = 0.0
        self.avg_entry = 0.0
        self.realized_pnl = 0.0

    def update_buy(self, fill_price: float, fill_size: float) -> float:
        """
        Process a buy fill.

        Args:
            fill_price: Price of the fill
            fill_size: Size of the fill (positive)

        Returns:
            Realized PnL from this fill (if covering short)
        """
        realized = 0.0
        old_position = self.position

        if self.position >= 0:
            # Adding to long or opening long
            total_cost = self.avg_entry * self.position + fill_price * fill_size
            self.position += fill_size
            self.avg_entry = total_cost / self.position if self.position > 0 else 0
        else:
            # Covering short
            cover = min(-self.position, fill_size)
            realized = (self.avg_entry - fill_price) * cover
            self.realized_pnl += realized
            self.position += fill_size

            # Handle position flip: short → long
            if old_position < 0 and self.position > 0:
                self.avg_entry = fill_price

        return realized

    def update_sell(self, fill_price: float, fill_size: float) -> float:
        """
        Process a sell fill.

        Args:
            fill_price: Price of the fill
            fill_size: Size of the fill (positive)

        Returns:
            Realized PnL from this fill (if closing long)
        """
        realized = 0.0
        old_position = self.position

        if self.position <= 0:
            # Adding to short or opening short
            total_cost = self.avg_entry * (-self.position) + fill_price * fill_size
            self.position -= fill_size
            self.avg_entry = total_cost / (-self.position) if self.position < 0 else 0
        else:
            # Closing long
            close = min(self.position, fill_size)
            realized = (fill_price - self.avg_entry) * close
            self.realized_pnl += realized
            self.position -= fill_size

            # Handle position flip: long → short
            if old_position > 0 and self.position < 0:
                self.avg_entry = fill_price

        return realized

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL at current price."""
        return compute_unrealized_pnl(self.position, self.avg_entry, current_price)

    def total_pnl(self, current_price: float) -> float:
        """Calculate total PnL (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl(current_price)


# =============================================================================
# Numba-optimized functions for performance-critical inner loops (Phase 3)
# =============================================================================


@njit(cache=True)
def compute_pnl_numba(position: float, avg_entry: float, price: float) -> float:
    """
    Numba-optimized unrealized PnL computation.

    Args:
        position: Current position size (positive = long, negative = short)
        avg_entry: Average entry price
        price: Current market price

    Returns:
        Unrealized PnL value
    """
    if position > 0:
        return (price - avg_entry) * position
    elif position < 0:
        return (avg_entry - price) * (-position)
    return 0.0


@njit(cache=True)
def update_position_buy_numba(
    position: float,
    avg_entry: float,
    realized_pnl: float,
    fill_price: float,
    fill_size: float,
) -> tuple[float, float, float, float]:
    """
    Numba-optimized buy position update.

    Args:
        position: Current position
        avg_entry: Current average entry price
        realized_pnl: Current realized PnL
        fill_price: Fill price
        fill_size: Fill size (positive)

    Returns:
        Tuple of (new_position, new_avg_entry, new_realized_pnl, fill_realized_pnl)
    """
    realized = 0.0

    if position >= 0:
        # Adding to long or opening long
        total_cost = avg_entry * position + fill_price * fill_size
        new_position = position + fill_size
        new_avg_entry = total_cost / new_position if new_position > 0 else 0.0
        new_realized = realized_pnl
    else:
        # Covering short
        cover = min(-position, fill_size)
        realized = (avg_entry - fill_price) * cover
        new_realized = realized_pnl + realized
        new_position = position + fill_size

        # Handle position flip: short → long
        if position < 0 and new_position > 0:
            new_avg_entry = fill_price
        else:
            new_avg_entry = avg_entry

    return new_position, new_avg_entry, new_realized, realized


@njit(cache=True)
def update_position_sell_numba(
    position: float,
    avg_entry: float,
    realized_pnl: float,
    fill_price: float,
    fill_size: float,
) -> tuple[float, float, float, float]:
    """
    Numba-optimized sell position update.

    Args:
        position: Current position
        avg_entry: Current average entry price
        realized_pnl: Current realized PnL
        fill_price: Fill price
        fill_size: Fill size (positive)

    Returns:
        Tuple of (new_position, new_avg_entry, new_realized_pnl, fill_realized_pnl)
    """
    realized = 0.0

    if position <= 0:
        # Adding to short or opening short
        total_cost = avg_entry * (-position) + fill_price * fill_size
        new_position = position - fill_size
        new_avg_entry = total_cost / (-new_position) if new_position < 0 else 0.0
        new_realized = realized_pnl
    else:
        # Closing long
        close = min(position, fill_size)
        realized = (fill_price - avg_entry) * close
        new_realized = realized_pnl + realized
        new_position = position - fill_size

        # Handle position flip: long → short
        if position > 0 and new_position < 0:
            new_avg_entry = fill_price
        else:
            new_avg_entry = avg_entry

    return new_position, new_avg_entry, new_realized, realized


@njit(cache=True)
def check_fill_buy_numba(
    order_price: float,
    order_size: float,
    best_ask_price: float,
    best_ask_size: float,
) -> tuple[bool, float, float]:
    """
    Check if a buy order fills against the orderbook.

    Args:
        order_price: Buy order price
        order_size: Buy order size
        best_ask_price: Best ask price
        best_ask_size: Best ask size

    Returns:
        Tuple of (filled, fill_price, fill_size)
    """
    if order_price >= best_ask_price and best_ask_size > 0:
        fill_size = min(order_size, best_ask_size)
        return True, best_ask_price, fill_size
    return False, 0.0, 0.0


@njit(cache=True)
def check_fill_sell_numba(
    order_price: float,
    order_size: float,
    best_bid_price: float,
    best_bid_size: float,
) -> tuple[bool, float, float]:
    """
    Check if a sell order fills against the orderbook.

    Args:
        order_price: Sell order price
        order_size: Sell order size
        best_bid_price: Best bid price
        best_bid_size: Best bid size

    Returns:
        Tuple of (filled, fill_price, fill_size)
    """
    if order_price <= best_bid_price and best_bid_size > 0:
        fill_size = min(order_size, best_bid_size)
        return True, best_bid_price, fill_size
    return False, 0.0, 0.0


@njit(cache=True)
def compute_maker_rebate_numba(price: float, size: float, rebate_pct: float = 0.20) -> float:
    """
    Numba-optimized maker rebate computation.

    Formula: shares * price * 0.25 * (price * (1 - price))^2 * rebate_pct

    Args:
        price: Fill price (0-1 range for binary options)
        size: Fill size in shares
        rebate_pct: Rebate percentage (default 20%)

    Returns:
        Rebate amount in USDC
    """
    fee_equivalent = size * price * 0.25 * (price * (1 - price)) ** 2
    return fee_equivalent * rebate_pct


@njit(cache=True)
def compute_sharpe_numba(pnl_arr: np.ndarray, annualization: float) -> float:
    """
    Numba-optimized Sharpe ratio computation.

    Args:
        pnl_arr: Array of PnL values
        annualization: Annualization factor

    Returns:
        Sharpe ratio
    """
    n = len(pnl_arr)
    if n <= 1:
        return 0.0

    # Compute returns
    returns = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        returns[i] = pnl_arr[i + 1] - pnl_arr[i]

    # Compute mean and std
    mean_ret = 0.0
    for i in range(len(returns)):
        mean_ret += returns[i]
    mean_ret /= len(returns)

    var = 0.0
    for i in range(len(returns)):
        diff = returns[i] - mean_ret
        var += diff * diff
    var /= len(returns)

    std = np.sqrt(var)
    if std <= 0:
        return 0.0

    return mean_ret / std * annualization


@njit(cache=True)
def compute_max_drawdown_numba(pnl_arr: np.ndarray) -> float:
    """
    Numba-optimized maximum drawdown computation.

    Args:
        pnl_arr: Array of cumulative PnL values

    Returns:
        Maximum drawdown value (positive number)
    """
    n = len(pnl_arr)
    if n == 0:
        return 0.0

    peak = pnl_arr[0]
    max_dd = 0.0

    for i in range(n):
        if pnl_arr[i] > peak:
            peak = pnl_arr[i]
        dd = peak - pnl_arr[i]
        if dd > max_dd:
            max_dd = dd

    return max_dd
