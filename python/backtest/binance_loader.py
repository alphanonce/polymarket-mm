"""
Binance Kline Data Loader

Fetches 1-second candle (kline) data from Binance REST API with local caching.
Used to provide external spot prices for TpBS backtest.
"""

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()

# Lazy import requests to avoid breaking import when not installed
_requests = None


def _get_requests():
    """Lazy load requests module."""
    global _requests
    if _requests is None:
        import requests
        _requests = requests
    return _requests

# Binance API endpoint
BINANCE_API_BASE = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"

# Rate limiting
MAX_KLINES_PER_REQUEST = 1000
REQUEST_DELAY_SECONDS = 0.1  # 100ms between requests

# Cache directory
DEFAULT_CACHE_DIR = "data/binance_cache"


@dataclass
class KlineData:
    """Single kline data point."""

    timestamp_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class BinanceKlineLoader:
    """
    Loads Binance kline (candle) data with local caching.

    Supports fetching 1s, 1m, 5m, 15m, 1h intervals.
    Data is cached as parquet files: {cache_dir}/{symbol}/{YYYY-MM}.parquet
    """

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(component="binance_loader")

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> pd.DataFrame:
        """
        Fetch klines from Binance API.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Kline interval (e.g., "1s", "1m", "1h")
            start_time_ms: Start time in milliseconds
            end_time_ms: End time in milliseconds

        Returns:
            DataFrame with columns: timestamp_ns, open, high, low, close, volume
        """
        all_klines = []
        current_start = start_time_ms

        while current_start < end_time_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_time_ms,
                "limit": MAX_KLINES_PER_REQUEST,
            }

            try:
                requests = _get_requests()
                response = requests.get(
                    f"{BINANCE_API_BASE}{KLINES_ENDPOINT}",
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
                klines = response.json()

                if not klines:
                    break

                all_klines.extend(klines)

                # Update start time for next batch
                last_time = klines[-1][0]  # Open time of last kline
                current_start = last_time + 1

                # Rate limiting
                time.sleep(REQUEST_DELAY_SECONDS)

            except Exception as e:
                self.logger.error("Failed to fetch klines", error=str(e), symbol=symbol)
                break

        if not all_klines:
            return pd.DataFrame()

        # Parse klines into DataFrame
        # Binance kline format: [open_time, open, high, low, close, volume, ...]
        df = pd.DataFrame(
            all_klines,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        # Convert to proper types
        df["timestamp_ns"] = df["open_time"].astype(np.int64) * 1_000_000  # ms -> ns
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # Select only needed columns
        return df[["timestamp_ns", "open", "high", "low", "close", "volume"]]

    def get_cache_path(self, symbol: str, year_month: str) -> Path:
        """Get cache file path for a symbol and month."""
        return self.cache_dir / symbol / f"{year_month}.parquet"

    def save_cache(self, df: pd.DataFrame, symbol: str, year_month: str) -> None:
        """Save DataFrame to cache."""
        cache_path = self.get_cache_path(symbol, year_month)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        self.logger.info("Saved cache", path=str(cache_path), rows=len(df))

    def load_cache(self, symbol: str, year_month: str) -> pd.DataFrame | None:
        """Load DataFrame from cache if exists."""
        cache_path = self.get_cache_path(symbol, year_month)
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return None

    def load_klines(
        self,
        symbol: str,
        start_time_ns: int,
        end_time_ns: int,
        interval: str = "1s",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load klines for a time range, using cache when available.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            start_time_ns: Start time in nanoseconds
            end_time_ns: End time in nanoseconds
            interval: Kline interval (default "1s")
            use_cache: Whether to use local cache

        Returns:
            DataFrame with kline data
        """
        start_ms = start_time_ns // 1_000_000
        end_ms = end_time_ns // 1_000_000

        # Convert to datetime to determine months to load
        start_dt = datetime.fromtimestamp(start_ms / 1000, tz=UTC)
        end_dt = datetime.fromtimestamp(end_ms / 1000, tz=UTC)

        # Generate list of months to load
        months = []
        current = start_dt.replace(day=1)
        while current <= end_dt:
            months.append(current.strftime("%Y-%m"))
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        all_dfs = []
        for year_month in months:
            # Try cache first
            if use_cache:
                cached = self.load_cache(symbol, year_month)
                if cached is not None:
                    self.logger.debug("Loaded from cache", symbol=symbol, month=year_month)
                    all_dfs.append(cached)
                    continue

            # Fetch from API
            year, month = map(int, year_month.split("-"))
            month_start = datetime(year, month, 1, tzinfo=UTC)
            if month == 12:
                month_end = datetime(year + 1, 1, 1, tzinfo=UTC)
            else:
                month_end = datetime(year, month + 1, 1, tzinfo=UTC)

            month_start_ms = int(month_start.timestamp() * 1000)
            month_end_ms = int(month_end.timestamp() * 1000) - 1

            self.logger.info("Fetching from API", symbol=symbol, month=year_month)
            df = self.fetch_klines(symbol, interval, month_start_ms, month_end_ms)

            if not df.empty:
                # Save to cache
                if use_cache:
                    self.save_cache(df, symbol, year_month)
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        # Concatenate and filter to exact time range
        result = pd.concat(all_dfs, ignore_index=True)
        result = result[
            (result["timestamp_ns"] >= start_time_ns) & (result["timestamp_ns"] <= end_time_ns)
        ]
        result = result.sort_values("timestamp_ns").reset_index(drop=True)

        self.logger.info(
            "Loaded klines",
            symbol=symbol,
            rows=len(result),
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
        )

        return result

    def get_price_at_timestamp(
        self,
        df: pd.DataFrame,
        timestamp_ns: int,
        method: str = "close",
    ) -> float | None:
        """
        Get price at a specific timestamp using binary search.

        Args:
            df: Kline DataFrame (must be sorted by timestamp_ns)
            timestamp_ns: Target timestamp in nanoseconds
            method: Price to return ("open", "high", "low", "close", "mid")

        Returns:
            Price at or before the timestamp, or None if not found
        """
        if df.empty:
            return None

        timestamps = df["timestamp_ns"].values

        # Binary search for insertion point
        idx = np.searchsorted(timestamps, timestamp_ns, side="right") - 1

        if idx < 0:
            return None

        row = df.iloc[idx]

        if method == "mid":
            return (row["high"] + row["low"]) / 2
        return row[method]


def asset_to_binance_symbol(asset: str) -> str:
    """
    Map Polymarket asset to Binance symbol.

    Args:
        asset: Polymarket asset (e.g., "btc", "BTC")

    Returns:
        Binance symbol (e.g., "BTCUSDT")
    """
    asset_upper = asset.upper()

    # Map common assets
    mapping = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "SOL": "SOLUSDT",
        "XRP": "XRPUSDT",
        "DOGE": "DOGEUSDT",
        "ADA": "ADAUSDT",
        "AVAX": "AVAXUSDT",
        "LINK": "LINKUSDT",
        "DOT": "DOTUSDT",
        "MATIC": "MATICUSDT",
        "SHIB": "SHIBUSDT",
        "LTC": "LTCUSDT",
        "BCH": "BCHUSDT",
        "UNI": "UNIUSDT",
        "ATOM": "ATOMUSDT",
    }

    return mapping.get(asset_upper, f"{asset_upper}USDT")


class BinancePriceIndex:
    """
    Pre-indexed Binance price data for fast lookups during backtest.

    Builds an index of prices by timestamp for O(1) lookups.
    """

    def __init__(self, df: pd.DataFrame, resolution_ns: int = 1_000_000_000):
        """
        Build price index from kline DataFrame.

        Args:
            df: Kline DataFrame with timestamp_ns column
            resolution_ns: Index resolution in nanoseconds (default 1 second)
        """
        self.resolution_ns = resolution_ns
        self._index: dict[int, float] = {}
        self._min_ts: int = 0
        self._max_ts: int = 0

        if df.empty:
            return

        # Build index using close prices
        self._min_ts = int(df["timestamp_ns"].min())
        self._max_ts = int(df["timestamp_ns"].max())

        # Use itertuples (3-5x faster than iterrows)
        for row in df.itertuples(index=False):
            ts_key = int(row.timestamp_ns) // resolution_ns
            self._index[ts_key] = row.close

    def get_price(self, timestamp_ns: int) -> float | None:
        """Get price at timestamp."""
        if not self._index:
            return None

        ts_key = timestamp_ns // self.resolution_ns

        # Exact match
        if ts_key in self._index:
            return self._index[ts_key]

        # Search backwards for nearest price
        for offset in range(1, 60):  # Look back up to 60 seconds
            if ts_key - offset in self._index:
                return self._index[ts_key - offset]

        return None

    @property
    def is_empty(self) -> bool:
        """Check if index has data."""
        return len(self._index) == 0

    @property
    def time_range(self) -> tuple[int, int]:
        """Get min and max timestamps."""
        return self._min_ts, self._max_ts
