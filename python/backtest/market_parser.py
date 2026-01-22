"""
Market Slug Parser

Parses Polymarket market_slug strings to extract market parameters:
- Asset (BTC, ETH, SOL, etc.)
- Direction (above, below)
- Strike price
- Expiry time

Example slugs:
- "btc-above-100000-jan-20-1pm"
- "eth-below-3500-jan-21-12am"
- "sol-above-200-feb-1-6pm"
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

logger = structlog.get_logger()

# Month name to number mapping
MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Supported assets
SUPPORTED_ASSETS = {"btc", "eth", "sol", "xrp", "doge", "ada", "avax", "link", "dot", "matic"}


@dataclass
class MarketParams:
    """Parsed market parameters."""

    asset: str  # BTC, ETH, SOL, etc. (uppercase)
    direction: str  # "above" or "below"
    strike: float  # Strike price (e.g., 100000)
    expiry_time: datetime  # Expiry datetime (UTC)
    raw_slug: str  # Original slug

    @property
    def binance_symbol(self) -> str:
        """Get Binance symbol for this asset."""
        return f"{self.asset}USDT"

    def time_to_expiry_years(self, current_time: datetime) -> float:
        """Calculate time to expiry in years."""
        delta = self.expiry_time - current_time
        return max(0.0, delta.total_seconds() / (365.25 * 24 * 3600))

    def time_to_expiry_hours(self, current_time: datetime) -> float:
        """Calculate time to expiry in hours."""
        delta = self.expiry_time - current_time
        return max(0.0, delta.total_seconds() / 3600)


def parse_market_slug(slug: str, reference_year: int = 2025) -> MarketParams | None:
    """
    Parse a Polymarket market slug into MarketParams.

    Args:
        slug: Market slug string (e.g., "btc-above-100000-jan-20-1pm")
        reference_year: Year to use for expiry (default 2025)

    Returns:
        MarketParams if parsing succeeds, None otherwise

    Examples:
        >>> parse_market_slug("btc-above-100000-jan-20-1pm")
        MarketParams(asset='BTC', direction='above', strike=100000.0, ...)

        >>> parse_market_slug("eth-below-3500-jan-21-12am")
        MarketParams(asset='ETH', direction='below', strike=3500.0, ...)
    """
    if not slug:
        return None

    slug_lower = slug.lower().strip()

    # Pattern: {asset}-{direction}-{strike}-{month}-{day}-{time}
    # Examples:
    #   btc-above-100000-jan-20-1pm
    #   eth-below-3500-jan-21-12am
    #   sol-above-200-feb-1-6pm

    # Try main pattern
    pattern = r"^([a-z]+)-(above|below)-(\d+(?:\.\d+)?)-([a-z]{3})-(\d{1,2})-(\d{1,2})(am|pm)$"
    match = re.match(pattern, slug_lower)

    if not match:
        # Try alternate pattern without dash before am/pm
        pattern2 = r"^([a-z]+)-(above|below)-(\d+(?:\.\d+)?)-([a-z]{3})-(\d{1,2})-(\d{1,2})\s*(am|pm)$"
        match = re.match(pattern2, slug_lower)

    if not match:
        logger.warning("Failed to parse market slug", slug=slug)
        return None

    asset = match.group(1).upper()
    direction = match.group(2)
    strike = float(match.group(3))
    month_str = match.group(4)
    day = int(match.group(5))
    hour = int(match.group(6))
    ampm = match.group(7)

    # Validate asset
    if asset.lower() not in SUPPORTED_ASSETS:
        logger.warning("Unsupported asset in slug", slug=slug, asset=asset)
        # Still continue, might be a new asset

    # Validate month
    if month_str not in MONTH_MAP:
        logger.warning("Invalid month in slug", slug=slug, month=month_str)
        return None

    month = MONTH_MAP[month_str]

    # Convert 12-hour to 24-hour
    if ampm == "am":
        if hour == 12:
            hour = 0
    else:  # pm
        if hour != 12:
            hour += 12

    # Build expiry datetime
    try:
        # Handle year rollover (e.g., parsing "jan" in December should use next year)
        year = reference_year
        current_month = datetime.now(UTC).month
        if month < current_month and current_month >= 11:
            year += 1

        expiry_time = datetime(year, month, day, hour, 0, 0, tzinfo=UTC)
    except ValueError as e:
        logger.warning("Invalid date in slug", slug=slug, error=str(e))
        return None

    return MarketParams(
        asset=asset,
        direction=direction,
        strike=strike,
        expiry_time=expiry_time,
        raw_slug=slug,
    )


def parse_asset_from_path(path: str) -> str | None:
    """
    Extract asset from a file path.

    Common patterns:
    - data/s3_cache/btc-above-100000-jan-20/...
    - polymarket/trades/crypto/1h/btc-above-100000-jan-20/...

    Args:
        path: File path string

    Returns:
        Asset string (uppercase) if found, None otherwise
    """
    # Look for known assets in path segments
    path_lower = path.lower()

    for asset in SUPPORTED_ASSETS:
        # Check for asset at start of a path segment
        patterns = [
            f"/{asset}-",  # /btc-above-...
            f"/{asset}/",  # /btc/
            f"/{asset}.",  # /btc.parquet
        ]
        for pattern in patterns:
            if pattern in path_lower:
                return asset.upper()

    return None


def parse_slug_from_path(path: str) -> str | None:
    """
    Extract market_slug from a file path.

    Args:
        path: File path string

    Returns:
        Slug string if found, None otherwise
    """
    # Common pattern: the slug is a directory name
    # e.g., data/s3_cache/btc-above-100000-jan-20-1pm/2025-01.parquet

    parts = path.replace("\\", "/").split("/")

    for part in parts:
        # Try to parse as slug
        params = parse_market_slug(part)
        if params:
            return part

    return None


def normalize_strike_for_bs(strike: float, spot: float) -> float:
    """
    Normalize strike price for Black-Scholes calculation.

    For binary options, the strike is typically expressed as an absolute price.
    BS model uses spot/strike ratio normalized to [0, 1].

    Args:
        strike: Absolute strike price (e.g., 100000 for BTC)
        spot: Current spot price

    Returns:
        Normalized ratio for BS calculation
    """
    if strike <= 0 or spot <= 0:
        return 0.5

    # Return spot/strike ratio (probability of ending above strike)
    return spot / strike


class MarketParamsCache:
    """Cache for parsed market parameters."""

    def __init__(self):
        self._cache: dict[str, MarketParams | None] = {}

    def get(self, slug: str, reference_year: int = 2025) -> MarketParams | None:
        """Get or parse market params for a slug."""
        cache_key = f"{slug}_{reference_year}"
        if cache_key not in self._cache:
            self._cache[cache_key] = parse_market_slug(slug, reference_year)
        return self._cache[cache_key]

    def clear(self):
        """Clear the cache."""
        self._cache.clear()


# Global cache instance
_params_cache = MarketParamsCache()


def get_market_params(slug: str, reference_year: int = 2025) -> MarketParams | None:
    """Get market params with caching."""
    return _params_cache.get(slug, reference_year)
