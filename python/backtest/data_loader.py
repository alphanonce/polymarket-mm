"""
Data Loader

Loads historical market data for backtesting.
Supports both local JSON snapshots and S3 parquet trades data.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd


@dataclass
class PriceLevel:
    """Single price level."""

    price: float
    size: float


@dataclass
class OrderBook:
    """Orderbook snapshot."""

    timestamp_ns: int
    bids: List[PriceLevel]
    asks: List[PriceLevel]

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0].price + self.asks[0].price) / 2

    @property
    def spread(self) -> float:
        """Calculate spread."""
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0].price - self.bids[0].price

    @property
    def best_bid(self) -> float:
        """Get best bid price."""
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        """Get best ask price."""
        return self.asks[0].price if self.asks else 0.0


@dataclass
class Trade:
    """Trade event."""

    timestamp_ns: int
    price: float
    size: float
    side: int  # 1 = buy, -1 = sell


@dataclass
class ExternalPrice:
    """External price snapshot."""

    timestamp_ns: int
    symbol: str
    price: float
    bid: float
    ask: float


@dataclass
class MarketData:
    """Market data for a single timestamp."""

    timestamp_ns: int
    orderbook: OrderBook
    trades: List[Trade] = field(default_factory=list)
    external_prices: Dict[str, ExternalPrice] = field(default_factory=dict)


class DataLoader:
    """Loads and iterates through historical market data."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._files: List[Path] = []
        self._current_file_idx: int = 0
        self._current_data: List[dict] = []
        self._current_idx: int = 0

    def load_files(self, pattern: str = "snapshots_*.json") -> int:
        """
        Load data files matching pattern.

        Args:
            pattern: Glob pattern for files

        Returns:
            Number of files found
        """
        self._files = sorted(self.data_dir.glob(pattern))
        self._current_file_idx = 0
        self._current_data = []
        self._current_idx = 0
        return len(self._files)

    def __iter__(self) -> Iterator[MarketData]:
        """Iterate through all market data."""
        for file_path in self._files:
            yield from self._load_file(file_path)

    def _load_file(self, path: Path) -> Iterator[MarketData]:
        """Load and iterate through a single file."""
        with open(path) as f:
            snapshots = json.load(f)

        for snap in snapshots:
            yield self._parse_snapshot(snap)

    def _parse_snapshot(self, snap: dict) -> MarketData:
        """Parse a snapshot into MarketData."""
        timestamp_ns = snap.get("timestamp_ns", 0)

        # Parse orderbook (take first market for simplicity)
        poly_books = snap.get("poly_books", {})
        orderbook = OrderBook(timestamp_ns=timestamp_ns, bids=[], asks=[])

        for asset_id, book_data in poly_books.items():
            bids = [
                PriceLevel(price=float(level["price"]), size=float(level["size"]))
                for level in book_data.get("bids", [])
            ]
            asks = [
                PriceLevel(price=float(level["price"]), size=float(level["size"]))
                for level in book_data.get("asks", [])
            ]
            # Sort bids descending, asks ascending
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)
            orderbook = OrderBook(timestamp_ns=timestamp_ns, bids=bids, asks=asks)
            break  # Only use first market for now

        # Parse external prices
        external_prices: Dict[str, ExternalPrice] = {}
        binance_prices = snap.get("binance_prices", {})
        for symbol, price_data in binance_prices.items():
            external_prices[symbol] = ExternalPrice(
                timestamp_ns=timestamp_ns,
                symbol=symbol,
                price=price_data.get("price", 0),
                bid=price_data.get("bid_price", 0),
                ask=price_data.get("ask_price", 0),
            )

        return MarketData(
            timestamp_ns=timestamp_ns,
            orderbook=orderbook,
            external_prices=external_prices,
        )


class S3TradesLoader:
    """
    Loads trades data from S3 parquet files.

    S3 structure:
        s3://an-trading-research/polymarket/trades/crypto/1h/{asset}/{YYYY-MM}.parquet
    """

    def __init__(
        self,
        bucket: str = "an-trading-research",
        prefix: str = "polymarket/trades/crypto/1h",
        local_cache_dir: Optional[str] = None,
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.local_cache_dir = Path(local_cache_dir) if local_cache_dir else None
        self._s3_client = None

    @property
    def s3_client(self):
        """Lazy S3 client initialization."""
        if self._s3_client is None:
            import boto3
            self._s3_client = boto3.client("s3")
        return self._s3_client

    def list_assets(self) -> List[str]:
        """List all available crypto assets."""
        assets = set()
        paginator = self.s3_client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix, Delimiter="/"):
            for prefix_info in page.get("CommonPrefixes", []):
                # Extract asset name from prefix like "polymarket/trades/crypto/1h/BTC/"
                asset = prefix_info["Prefix"].rstrip("/").split("/")[-1]
                assets.add(asset)

        return sorted(assets)

    def list_files(self, asset: Optional[str] = None) -> List[str]:
        """List parquet files for an asset (or all if asset is None)."""
        prefix = f"{self.prefix}/{asset}" if asset else self.prefix
        files = []

        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    files.append(obj["Key"])

        return sorted(files)

    def load_parquet(self, s3_key: str) -> pd.DataFrame:
        """Load a parquet file from S3 (with optional local caching)."""
        # Check local cache first
        if self.local_cache_dir:
            local_path = self.local_cache_dir / s3_key.replace(f"{self.prefix}/", "")
            if local_path.exists():
                return pd.read_parquet(local_path)

            # Download to cache
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket, s3_key, str(local_path))
            return pd.read_parquet(local_path)

        # Direct read from S3
        s3_uri = f"s3://{self.bucket}/{s3_key}"
        return pd.read_parquet(s3_uri)

    def load_trades(
        self,
        asset: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load trades data for an asset.

        Args:
            asset: Asset symbol (e.g., "BTC"). If None, loads all assets.
            start_date: Filter start date (YYYY-MM format)
            end_date: Filter end date (YYYY-MM format)

        Returns:
            DataFrame with trades data
        """
        files = self.list_files(asset)

        # Filter by date if specified
        if start_date or end_date:
            filtered = []
            for f in files:
                # Extract YYYY-MM from filename
                filename = f.split("/")[-1]
                file_month = filename.replace(".parquet", "")

                if start_date and file_month < start_date:
                    continue
                if end_date and file_month > end_date:
                    continue
                filtered.append(f)
            files = filtered

        if not files:
            return pd.DataFrame()

        # Load and concatenate all files
        dfs = []
        for f in files:
            try:
                df = self.load_parquet(f)
                # Add asset column if not present
                if "asset" not in df.columns:
                    asset_name = f.split("/")[-2]  # Get asset from path
                    df["asset"] = asset_name
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)

        # Sort by timestamp
        time_col = self._find_time_column(result)
        if time_col:
            result = result.sort_values(time_col).reset_index(drop=True)

        return result

    def _find_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the timestamp column in the dataframe."""
        for col in ["timestamp", "time", "ts", "timestamp_ns", "datetime"]:
            if col in df.columns:
                return col
        # Check for columns containing 'time'
        for col in df.columns:
            if "time" in col.lower():
                return col
        return None

    def iter_trades(
        self,
        asset: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Iterator[Trade]:
        """
        Iterate through trades as Trade objects.

        Args:
            asset: Asset symbol (e.g., "BTC")
            start_date: Filter start date (YYYY-MM format)
            end_date: Filter end date (YYYY-MM format)

        Yields:
            Trade objects
        """
        df = self.load_trades(asset, start_date, end_date)

        if df.empty:
            return

        # Map column names (Polymarket schema)
        time_col = self._find_column(df, ["timestamp", "time", "ts", "timestamp_ns"])
        price_col = self._find_column(df, ["price", "trade_price"])
        size_col = self._find_column(df, ["outcome_tokens_amount", "trade_amount", "size", "amount", "quantity"])
        side_col = self._find_column(df, ["trade_type", "side", "direction", "taker_side"])

        for _, row in df.iterrows():
            timestamp_ns = self._to_timestamp_ns(row.get(time_col)) if time_col else 0
            price = float(row.get(price_col, 0)) if price_col else 0.0
            size = float(row.get(size_col, 0)) if size_col else 0.0

            # Parse side (Polymarket uses BUY/SELL in trade_type)
            side = 0
            if side_col and side_col in row:
                side_val = row[side_col]
                if isinstance(side_val, str):
                    side = 1 if side_val.upper() in ("BUY", "BID", "B") else -1
                else:
                    side = int(side_val) if side_val else 0

            yield Trade(
                timestamp_ns=timestamp_ns,
                price=price,
                size=size,
                side=side,
            )

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find a column from a list of candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _to_timestamp_ns(self, val) -> int:
        """Convert various timestamp formats to nanoseconds."""
        if val is None:
            return 0
        if isinstance(val, (int, float)):
            # Assume it's already in a reasonable format
            if val > 1e18:  # Already nanoseconds
                return int(val)
            elif val > 1e15:  # Microseconds
                return int(val * 1000)
            elif val > 1e12:  # Milliseconds
                return int(val * 1_000_000)
            else:  # Seconds
                return int(val * 1_000_000_000)
        if isinstance(val, pd.Timestamp):
            return int(val.value)
        return 0


class LocalParquetLoader:
    """Load trades from local parquet files."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def list_files(self, pattern: str = "**/*.parquet") -> List[Path]:
        """List all parquet files."""
        return sorted(self.data_dir.glob(pattern))

    def load_trades(self, pattern: str = "**/*.parquet") -> pd.DataFrame:
        """Load all trades from matching files."""
        files = self.list_files(pattern)

        if not files:
            return pd.DataFrame()

        dfs = []
        for f in files:
            try:
                df = pd.read_parquet(f)
                if "asset" not in df.columns:
                    # Extract asset from path
                    asset = f.parent.name
                    df["asset"] = asset
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)


def generate_synthetic_data(
    n_ticks: int = 10000,
    initial_price: float = 0.50,
    volatility: float = 0.001,
    tick_interval_ns: int = 100_000_000,  # 100ms
) -> Iterator[MarketData]:
    """
    Generate synthetic market data for testing.

    Args:
        n_ticks: Number of ticks to generate
        initial_price: Starting price
        volatility: Price volatility per tick
        tick_interval_ns: Time between ticks in nanoseconds

    Yields:
        MarketData for each tick
    """
    rng = np.random.default_rng(42)
    price = initial_price
    timestamp = 0

    for _ in range(n_ticks):
        # Random walk price
        price += rng.normal(0, volatility)
        price = max(0.01, min(0.99, price))  # Clamp to valid range

        # Generate orderbook
        spread = rng.uniform(0.01, 0.03)
        half_spread = spread / 2

        bids = []
        asks = []

        # Generate 5 levels each side
        for i in range(5):
            bid_price = price - half_spread - i * 0.005
            ask_price = price + half_spread + i * 0.005

            bid_size = rng.uniform(10, 100)
            ask_size = rng.uniform(10, 100)

            bids.append(PriceLevel(price=round(bid_price, 4), size=round(bid_size, 2)))
            asks.append(PriceLevel(price=round(ask_price, 4), size=round(ask_size, 2)))

        orderbook = OrderBook(timestamp_ns=timestamp, bids=bids, asks=asks)

        # Generate external price
        external_prices = {
            "BTCUSDT": ExternalPrice(
                timestamp_ns=timestamp,
                symbol="BTCUSDT",
                price=50000 + rng.normal(0, 100),
                bid=50000 - 5,
                ask=50000 + 5,
            )
        }

        yield MarketData(
            timestamp_ns=timestamp,
            orderbook=orderbook,
            external_prices=external_prices,
        )

        timestamp += tick_interval_ns
