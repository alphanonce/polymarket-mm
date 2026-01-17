"""
Utility Functions

Reusable math and helper functions for strategy models.
"""

from strategy.utils.black_scholes import bs_binary_call, bs_binary_put, norm_cdf
from strategy.utils.polymarket import (
    DEFAULT_MIN_SIZE,
    DEFAULT_SIZE_TICK,
    MarketInfo,
    TickInfo,
    clamp_price,
    clamp_quantity,
    clear_cache,
    fetch_tick_info,
    fetch_tick_info_sync,
    format_price,
    format_quantity,
    get_tick_info,
    get_tick_size,
    is_valid_price,
    is_valid_quantity,
    round_ask,
    round_bid,
    round_quantity,
    round_to_tick,
    set_tick_info,
    ticks_between,
)
from strategy.utils.volatility import (
    PriceHistory,
    compute_log_returns,
    compute_realized_volatility,
)

__all__ = [
    # Black-Scholes
    "norm_cdf",
    "bs_binary_call",
    "bs_binary_put",
    # Volatility
    "compute_log_returns",
    "compute_realized_volatility",
    "PriceHistory",
    # Polymarket - Classes
    "TickInfo",
    "MarketInfo",
    # Polymarket - Constants
    "DEFAULT_SIZE_TICK",
    "DEFAULT_MIN_SIZE",
    # Polymarket - Price rounding
    "get_tick_size",
    "round_to_tick",
    "round_bid",
    "round_ask",
    "clamp_price",
    # Polymarket - Quantity rounding
    "round_quantity",
    "clamp_quantity",
    # Polymarket - Formatting
    "format_price",
    "format_quantity",
    # Polymarket - Validation
    "is_valid_price",
    "is_valid_quantity",
    "ticks_between",
    # Polymarket - Tick info management
    "get_tick_info",
    "set_tick_info",
    "fetch_tick_info",
    "fetch_tick_info_sync",
    "clear_cache",
]
