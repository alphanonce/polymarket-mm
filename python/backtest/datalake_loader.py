"""
Datalake Data Loader

Loads historical market data from processed datalake parquet files.
Data structure: data/datalake/processed/15m/{symbol}/{year}/{month}/{day}/{slug}/
"""

import gzip
import io
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

from backtest.data_loader import OrderBook, PriceLevel, Trade
from backtest.utils import to_timestamp_ns


@dataclass
class MarketInfo:
    """Information about a 15-minute market."""

    slug: str
    symbol: str
    date: str  # YYYY-MM-DD format
    start_timestamp: int  # Unix timestamp from slug
    path: Path


@dataclass
class DatalakeMarketData:
    """Container for single 15-min market data."""

    info: MarketInfo
    books: list[OrderBook] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)

    @property
    def slug(self) -> str:
        """Get market slug."""
        return self.info.slug

    @property
    def symbol(self) -> str:
        """Get symbol."""
        return self.info.symbol

    @property
    def has_data(self) -> bool:
        """Check if market has any data."""
        return len(self.books) > 0 or len(self.trades) > 0


# Numpy structured array dtype for orderbook levels (Phase 5 optimization)
LEVEL_DTYPE = np.dtype([("price", np.float64), ("size", np.float64)])


def parse_orderbook_json(
    asks_json: str | bytes,
    bids_json: str | bytes,
) -> tuple[list[PriceLevel], list[PriceLevel]]:
    """
    Parse orderbook JSON strings into PriceLevel lists.

    Args:
        asks_json: JSON string like '[["0.99","11801.45"],["0.98","3670.02"],...]'
        bids_json: JSON string like '[["0.01","9053.69"],["0.02","2775.47"],...]'

    Returns:
        Tuple of (asks, bids) as sorted PriceLevel lists
    """
    asks_raw = orjson.loads(asks_json) if asks_json else []
    bids_raw = orjson.loads(bids_json) if bids_json else []

    # Parse to PriceLevel objects
    asks = [PriceLevel(float(a[0]), float(a[1])) for a in asks_raw]
    bids = [PriceLevel(float(b[0]), float(b[1])) for b in bids_raw]

    # Sort: asks ascending by price, bids descending by price
    asks.sort(key=lambda x: x.price)
    bids.sort(key=lambda x: -x.price)

    return asks, bids


def parse_levels_to_array(json_str: str | bytes | None) -> np.ndarray:
    """
    Parse orderbook JSON to numpy structured array (optimized).

    Args:
        json_str: JSON string like '[["0.99","11801.45"],["0.98","3670.02"],...]'

    Returns:
        Numpy structured array with ('price', 'size') fields
    """
    if not json_str:
        return np.array([], dtype=LEVEL_DTYPE)

    levels = orjson.loads(json_str)
    if not levels:
        return np.array([], dtype=LEVEL_DTYPE)

    # Pre-allocate and fill array
    arr = np.empty(len(levels), dtype=LEVEL_DTYPE)
    for i, (price, size) in enumerate(levels):
        arr[i] = (float(price), float(size))

    return arr


def parse_levels_to_tuples(json_str: str | bytes | None) -> list[tuple[float, float]]:
    """
    Parse orderbook JSON to list of tuples (optimized with orjson).

    Args:
        json_str: JSON string like '[["0.99","11801.45"],["0.98","3670.02"],...]'

    Returns:
        List of (price, size) tuples
    """
    if not json_str:
        return []

    levels = orjson.loads(json_str)
    return [(float(p), float(s)) for p, s in levels]


class DatalakeLoader:
    """
    Loads and iterates through markets from processed datalake parquet files.

    Data structure:
        data/datalake/processed/15m/{symbol}/{year}/{month}/{day}/{slug}/
            - book.parquet.gz: orderbook snapshots
            - last_trade_price.parquet.gz: trade events
    """

    def __init__(
        self,
        data_dir: str = "data/datalake/processed/15m",
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        side: str = "up",  # "up", "down", or "both"
    ):
        """
        Initialize datalake loader.

        Args:
            data_dir: Base directory for processed data
            symbols: List of symbols to load (e.g., ["btc", "eth"]). None = all
            start_date: Filter start date (YYYY-MM-DD)
            end_date: Filter end date (YYYY-MM-DD)
            side: Which side to load ("up", "down", or "both")
        """
        self.data_dir = Path(data_dir)
        self.symbols = [s.lower() for s in symbols] if symbols else None
        self.start_date = start_date
        self.end_date = end_date
        self.side = side.lower()

        self._markets: list[MarketInfo] | None = None

    def discover_markets(self) -> list[MarketInfo]:
        """
        Discover all available markets in the data directory.

        Returns:
            List of MarketInfo for available markets
        """
        if self._markets is not None:
            return self._markets

        markets = []

        if not self.data_dir.exists():
            return markets

        # Iterate through symbol directories
        for symbol_dir in sorted(self.data_dir.iterdir()):
            if not symbol_dir.is_dir():
                continue

            symbol = symbol_dir.name.lower()

            # Filter by symbols if specified
            if self.symbols and symbol not in self.symbols:
                continue

            # Find all market directories
            for market_path in symbol_dir.rglob("*"):
                if not market_path.is_dir():
                    continue

                # Check if this directory contains the expected parquet files
                book_file = market_path / "book.parquet.gz"
                if not book_file.exists():
                    continue

                slug = market_path.name

                # Filter by side
                if self.side != "both":
                    # Slugs are like "btc-updown-15m-1768872600"
                    # The side is determined by which token_id we're looking at
                    # For now, we'll filter during data loading
                    pass

                # Extract date from path (e.g., btc/2026/01/20/slug)
                try:
                    parts = list(market_path.relative_to(self.data_dir).parts)
                    if len(parts) >= 4:
                        year, month, day = parts[1], parts[2], parts[3]
                        date_str = f"{year}-{month}-{day}"

                        # Filter by date
                        if self.start_date and date_str < self.start_date:
                            continue
                        if self.end_date and date_str > self.end_date:
                            continue
                    else:
                        date_str = "unknown"
                except (ValueError, IndexError):
                    date_str = "unknown"

                # Extract start timestamp from slug
                try:
                    start_ts = int(slug.split("-")[-1])
                except (ValueError, IndexError):
                    start_ts = 0

                markets.append(
                    MarketInfo(
                        slug=slug,
                        symbol=symbol,
                        date=date_str,
                        start_timestamp=start_ts,
                        path=market_path,
                    )
                )

        # Sort by timestamp
        markets.sort(key=lambda m: (m.date, m.start_timestamp))
        self._markets = markets

        return markets

    def load_market(self, market: MarketInfo) -> DatalakeMarketData:
        """
        Load data for a single market.

        Args:
            market: MarketInfo to load

        Returns:
            DatalakeMarketData containing orderbooks and trades
        """
        data = DatalakeMarketData(info=market)

        # Load orderbook data
        book_path = market.path / "book.parquet.gz"
        if book_path.exists():
            data.books = self._load_orderbooks(book_path)

        # Load trade data
        trades_path = market.path / "last_trade_price.parquet.gz"
        if trades_path.exists():
            data.trades = self._load_trades(trades_path)

        return data

    def _load_orderbooks(self, path: Path) -> list[OrderBook]:
        """Load orderbook snapshots from parquet file."""
        df = pd.read_parquet(path)

        if df.empty:
            return []

        # Sort by timestamp
        if "ts" in df.columns:
            df = df.sort_values("ts").reset_index(drop=True)

        # Pre-allocate list for better performance
        books: list[OrderBook] = []
        books_append = books.append  # Local reference for speed

        # Use itertuples (3-5x faster than iterrows)
        for row in df.itertuples(index=False):
            timestamp_ns = to_timestamp_ns(getattr(row, "ts", 0))
            asks_json = getattr(row, "asks", "[]")
            bids_json = getattr(row, "bids", "[]")

            asks, bids = parse_orderbook_json(asks_json, bids_json)

            books_append(
                OrderBook(
                    timestamp_ns=timestamp_ns,
                    asks=asks,
                    bids=bids,
                )
            )

        return books

    def _load_trades(self, path: Path) -> list[Trade]:
        """Load trades from parquet file."""
        df = pd.read_parquet(path)

        if df.empty:
            return []

        # Sort by timestamp
        if "ts" in df.columns:
            df = df.sort_values("ts").reset_index(drop=True)

        # Pre-allocate list for better performance
        trades: list[Trade] = []
        trades_append = trades.append  # Local reference for speed

        # Use itertuples (3-5x faster than iterrows)
        for row in df.itertuples(index=False):
            timestamp_ns = to_timestamp_ns(getattr(row, "ts", 0))
            price = float(getattr(row, "price", 0))
            size = float(getattr(row, "size", 0))

            # Parse side: 0 = sell, 1 = buy (based on data inspection)
            side_val = getattr(row, "side", 0)
            if isinstance(side_val, (int, float)):
                side = 1 if side_val == 1 else -1
            else:
                side = 1 if str(side_val).upper() in ("BUY", "1") else -1

            trades_append(
                Trade(
                    timestamp_ns=timestamp_ns,
                    price=price,
                    size=size,
                    side=side,
                )
            )

        return trades

    def iter_markets(self) -> Iterator[DatalakeMarketData]:
        """
        Iterate through all markets, loading data for each.

        Yields:
            DatalakeMarketData for each market
        """
        markets = self.discover_markets()

        for market in markets:
            data = self.load_market(market)
            if data.has_data:
                yield data

    def load_all(self) -> dict[str, DatalakeMarketData]:
        """
        Load all markets into memory.

        Returns:
            Dict mapping slug to DatalakeMarketData
        """
        result = {}
        for data in self.iter_markets():
            result[data.slug] = data
        return result

    def get_stats(self) -> dict[str, int]:
        """Get statistics about available data."""
        markets = self.discover_markets()

        symbols = set()
        dates = set()
        total_markets = len(markets)

        for m in markets:
            symbols.add(m.symbol)
            dates.add(m.date)

        return {
            "total_markets": total_markets,
            "unique_symbols": len(symbols),
            "unique_dates": len(dates),
            "symbols": sorted(symbols),
            "dates": sorted(dates),
        }


class RTDSPriceLoader:
    """
    Loads RTDS (Chainlink) price data from datalake.

    Data structure: data/datalake/global/rtds_crypto_prices/{year}/{month}/{day}/*.parquet.gz
    """

    # Symbol mapping from market symbol to RTDS symbol
    SYMBOL_MAP = {
        "btc": "btc/usd",
        "eth": "eth/usd",
        "sol": "sol/usd",
        "xrp": "xrp/usd",
        "doge": "doge/usd",
        "ada": "ada/usd",
        "avax": "avax/usd",
        "link": "link/usd",
    }

    def __init__(self, data_dir: str = "data/datalake/global/rtds_crypto_prices"):
        """
        Initialize RTDS price loader.

        Args:
            data_dir: Base directory for RTDS price data
        """
        self.data_dir = Path(data_dir)
        self._prices: pd.DataFrame | None = None
        self._price_index: dict[str, np.ndarray] = {}  # symbol -> sorted timestamps
        self._price_values: dict[str, np.ndarray] = {}  # symbol -> prices

    def load(self) -> bool:
        """
        Load all RTDS price data.

        Returns:
            True if data was loaded successfully
        """
        if not self.data_dir.exists():
            return False

        parquet_files = list(self.data_dir.rglob("*.parquet.gz"))
        if not parquet_files:
            return False

        dfs = []
        for path in parquet_files:
            try:
                with gzip.open(path, "rb") as f:
                    df = pd.read_parquet(io.BytesIO(f.read()))
                    dfs.append(df)
            except Exception:
                continue

        if not dfs:
            return False

        self._prices = pd.concat(dfs, ignore_index=True)
        self._prices["price"] = self._prices["price"].astype(float)
        self._prices["ts"] = self._prices["ts"].astype(np.int64)
        self._prices = self._prices.sort_values("ts").reset_index(drop=True)
        self._prices = self._prices.drop_duplicates(subset=["ts", "symbol"], keep="first")

        # Build index for fast lookups
        for symbol in self._prices["symbol"].unique():
            mask = self._prices["symbol"] == symbol
            self._price_index[symbol] = self._prices.loc[mask, "ts"].values
            self._price_values[symbol] = self._prices.loc[mask, "price"].values

        return True

    def get_price(self, symbol: str, timestamp_ms: int) -> float | None:
        """
        Get price for symbol at or before timestamp.

        Args:
            symbol: Symbol to look up (e.g., "BTC", "ETH")
            timestamp_ms: Timestamp in milliseconds

        Returns:
            Price if found, None otherwise
        """
        # Map market symbol to RTDS symbol
        rtds_symbol = self.SYMBOL_MAP.get(symbol.lower(), symbol.upper())

        if rtds_symbol not in self._price_index:
            return None

        timestamps = self._price_index[rtds_symbol]
        prices = self._price_values[rtds_symbol]

        # Binary search for closest timestamp <= target
        idx = np.searchsorted(timestamps, timestamp_ms, side="right") - 1

        if idx < 0:
            return None

        return float(prices[idx])

    def get_price_ns(self, symbol: str, timestamp_ns: int) -> float | None:
        """
        Get price for symbol at or before timestamp (nanoseconds).

        Args:
            symbol: Symbol to look up (e.g., "BTC", "ETH")
            timestamp_ns: Timestamp in nanoseconds

        Returns:
            Price if found, None otherwise
        """
        return self.get_price(symbol, timestamp_ns // 1_000_000)

    @property
    def is_loaded(self) -> bool:
        """Check if data is loaded."""
        return self._prices is not None and len(self._prices) > 0

    @property
    def symbols(self) -> list[str]:
        """Get list of available symbols."""
        return list(self._price_index.keys())
