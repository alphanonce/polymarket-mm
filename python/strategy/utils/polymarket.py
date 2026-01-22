"""
Polymarket Price and Quantity Utilities

Tick size management, price/quantity rounding, and formatting for Polymarket.
Supports fetching tick info from the Polymarket CLOB API.
"""

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional

# Polymarket price bounds
MIN_PRICE = 0.01
MAX_PRICE = 0.99

# Polymarket tick size boundaries
FINE_TICK_LOWER = 0.04
FINE_TICK_UPPER = 0.96

# Default tick sizes
FINE_TICK = 0.001  # Near boundaries (0-0.04, 0.96-1.0)
COARSE_TICK = 0.01  # Normal range (0.04-0.96)

# Default quantity tick size (Polymarket uses 0.01 for most markets)
DEFAULT_SIZE_TICK = 0.01

# Minimum order size (Polymarket enforces this)
DEFAULT_MIN_SIZE = 5.0


@dataclass
class TickInfo:
    """Tick size information for a market."""

    price_tick: float = COARSE_TICK  # Price increment
    size_tick: float = DEFAULT_SIZE_TICK  # Quantity increment
    min_size: float = DEFAULT_MIN_SIZE  # Minimum order size
    price_decimals: int = 2  # Decimal places for price
    size_decimals: int = 2  # Decimal places for size

    @classmethod
    def from_api_response(cls, data: dict) -> "TickInfo":
        """
        Create TickInfo from Polymarket API response.

        Args:
            data: API response dict with tick size info

        Returns:
            TickInfo instance
        """
        # Polymarket API returns tick sizes as strings
        price_tick = float(data.get("minimum_tick_size", COARSE_TICK))
        size_tick = float(data.get("minimum_order_size", DEFAULT_SIZE_TICK))
        min_size = float(data.get("min_incentive_size", DEFAULT_MIN_SIZE))

        # Derive decimal places from tick sizes
        price_decimals = _decimals_from_tick(price_tick)
        size_decimals = _decimals_from_tick(size_tick)

        return cls(
            price_tick=price_tick,
            size_tick=size_tick,
            min_size=min_size,
            price_decimals=price_decimals,
            size_decimals=size_decimals,
        )


@dataclass
class MarketInfo:
    """Market information including tick sizes."""

    token_id: str
    tick_info: TickInfo
    condition_id: Optional[str] = None
    outcome: Optional[str] = None  # "Yes" or "No"

    @classmethod
    def default(cls, token_id: str) -> "MarketInfo":
        """Create MarketInfo with default tick sizes."""
        return cls(token_id=token_id, tick_info=TickInfo())


# Global cache for market tick info
_market_cache: Dict[str, MarketInfo] = {}


def _decimals_from_tick(tick: float) -> int:
    """Derive decimal places from tick size."""
    if tick <= 0:
        return 2
    # Count decimals needed to represent tick
    s = f"{tick:.10f}".rstrip("0")
    if "." not in s:
        return 0
    return len(s.split(".")[1])


def get_tick_size(price: float) -> float:
    """
    Get the appropriate price tick size for a given price.

    Polymarket uses finer tick sizes near price boundaries (0 and 1)
    where small price changes represent larger probability movements.

    Args:
        price: Price value (should be in [0, 1])

    Returns:
        Tick size (0.001 near boundaries, 0.01 in normal range)
    """
    if price < FINE_TICK_LOWER or price > FINE_TICK_UPPER:
        return FINE_TICK
    return COARSE_TICK


def get_tick_info(token_id: str) -> TickInfo:
    """
    Get tick info for a market.

    Args:
        token_id: Polymarket token ID

    Returns:
        TickInfo for the market (from cache or default)
    """
    if token_id in _market_cache:
        return _market_cache[token_id].tick_info
    return TickInfo()


def set_tick_info(token_id: str, tick_info: TickInfo) -> None:
    """
    Cache tick info for a market.

    Args:
        token_id: Polymarket token ID
        tick_info: Tick info to cache
    """
    if token_id in _market_cache:
        _market_cache[token_id].tick_info = tick_info
    else:
        _market_cache[token_id] = MarketInfo(token_id=token_id, tick_info=tick_info)


def round_bid(price: float, tick: Optional[float] = None) -> float:
    """
    Round bid price DOWN (floor) for profitable buying.

    Lower bid = better price for buyer.

    Args:
        price: Price to round
        tick: Tick size (auto-detected if None)

    Returns:
        Price floored to tick
    """
    if tick is None:
        tick = get_tick_size(price)
    return math.floor(price / tick) * tick


def round_ask(price: float, tick: Optional[float] = None) -> float:
    """
    Round ask price UP (ceil) for profitable selling.

    Higher ask = better price for seller.

    Args:
        price: Price to round
        tick: Tick size (auto-detected if None)

    Returns:
        Price ceiled to tick
    """
    if tick is None:
        tick = get_tick_size(price)
    return math.ceil(price / tick) * tick


def round_to_tick(
    price: float,
    direction: Literal["down", "up", "nearest"] = "nearest",
    tick: Optional[float] = None,
) -> float:
    """
    Round price to valid Polymarket tick.

    Args:
        price: Price to round
        direction: Rounding direction
            - "down": Floor (for bids)
            - "up": Ceil (for asks)
            - "nearest": Round to nearest tick
        tick: Tick size (auto-detected if None)

    Returns:
        Price rounded to valid tick
    """
    if tick is None:
        tick = get_tick_size(price)

    if direction == "down":
        return math.floor(price / tick) * tick
    elif direction == "up":
        return math.ceil(price / tick) * tick
    else:  # nearest
        return round(price / tick) * tick


def round_quantity(
    quantity: float,
    tick: float = DEFAULT_SIZE_TICK,
    direction: Literal["down", "up", "nearest"] = "nearest",
) -> float:
    """
    Round quantity to valid Polymarket size tick.

    Args:
        quantity: Quantity to round
        tick: Size tick (default 0.01)
        direction: Rounding direction

    Returns:
        Quantity rounded to valid tick
    """
    if direction == "down":
        return math.floor(quantity / tick) * tick
    elif direction == "up":
        return math.ceil(quantity / tick) * tick
    else:  # nearest
        return round(quantity / tick) * tick


def clamp_price(price: float, decimals: int = 4) -> float:
    """
    Clamp price to valid Polymarket range and round to decimals.

    Args:
        price: Price to clamp
        decimals: Decimal places to round to

    Returns:
        Price clamped to [0.01, 0.99] and rounded
    """
    clamped = max(MIN_PRICE, min(MAX_PRICE, price))
    return round(clamped, decimals)


def clamp_quantity(quantity: float, min_size: float = DEFAULT_MIN_SIZE) -> float:
    """
    Ensure quantity meets minimum size requirement.

    Args:
        quantity: Quantity to check
        min_size: Minimum order size

    Returns:
        Quantity (0 if below minimum, otherwise unchanged)
    """
    if quantity < min_size:
        return 0.0
    return quantity


def format_price(price: float, tick: Optional[float] = None) -> str:
    """
    Format price for display/API with correct decimals.

    Args:
        price: Price value
        tick: Tick size for decimal inference

    Returns:
        Formatted price string
    """
    if tick is None:
        tick = get_tick_size(price)
    decimals = _decimals_from_tick(tick)
    return f"{price:.{decimals}f}"


def format_quantity(quantity: float, tick: float = DEFAULT_SIZE_TICK) -> str:
    """
    Format quantity for display/API with correct decimals.

    Args:
        quantity: Quantity value
        tick: Size tick for decimal inference

    Returns:
        Formatted quantity string
    """
    decimals = _decimals_from_tick(tick)
    return f"{quantity:.{decimals}f}"


def is_valid_price(price: float) -> bool:
    """
    Check if price is valid for Polymarket.

    Args:
        price: Price to check

    Returns:
        True if price is in valid range [0.01, 0.99]
    """
    return MIN_PRICE <= price <= MAX_PRICE


def is_valid_quantity(quantity: float, min_size: float = DEFAULT_MIN_SIZE) -> bool:
    """
    Check if quantity meets minimum size requirement.

    Args:
        quantity: Quantity to check
        min_size: Minimum order size

    Returns:
        True if quantity >= min_size
    """
    return quantity >= min_size


def ticks_between(price1: float, price2: float) -> int:
    """
    Calculate number of ticks between two prices.

    Note: This is an approximation since tick size varies with price.
    Uses the tick size at the midpoint.

    Args:
        price1: First price
        price2: Second price

    Returns:
        Approximate number of ticks between prices
    """
    mid_price = (price1 + price2) / 2
    tick = get_tick_size(mid_price)
    return int(abs(price2 - price1) / tick)


def price_to_probability(price: float) -> float:
    """
    Convert Polymarket price to probability.

    For binary markets, price equals probability directly.

    Args:
        price: Polymarket price

    Returns:
        Probability (same as price for binary markets)
    """
    return price


async def fetch_tick_info(token_id: str, api_url: str = "https://clob.polymarket.com") -> TickInfo:
    """
    Fetch tick info from Polymarket CLOB API.

    Args:
        token_id: Polymarket token ID
        api_url: CLOB API base URL

    Returns:
        TickInfo from API

    Raises:
        Exception if API call fails
    """
    import aiohttp

    url = f"{api_url}/markets/{token_id}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"API error: {response.status}")
            data = await response.json()

    tick_info = TickInfo.from_api_response(data)

    # Cache the result
    set_tick_info(token_id, tick_info)

    return tick_info


def fetch_tick_info_sync(token_id: str, api_url: str = "https://clob.polymarket.com") -> TickInfo:
    """
    Fetch tick info from Polymarket CLOB API (synchronous).

    Args:
        token_id: Polymarket token ID
        api_url: CLOB API base URL

    Returns:
        TickInfo from API

    Raises:
        Exception if API call fails
    """
    import requests

    url = f"{api_url}/markets/{token_id}"
    response = requests.get(url, timeout=10)

    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code}")

    data = response.json()
    tick_info = TickInfo.from_api_response(data)

    # Cache the result
    set_tick_info(token_id, tick_info)

    return tick_info


def clear_cache() -> None:
    """Clear the market info cache."""
    _market_cache.clear()
