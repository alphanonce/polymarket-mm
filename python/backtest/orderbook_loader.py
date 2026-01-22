"""
Orderbook CSV Data Loader

Loads orderbook tick data from CSV files with format:
    time,side,best_bid,best_ask,mid
    2026-01-13 00:00:28.938804,up,0.54,0.55,0.545

Used for tick-by-tick backtesting with crossing-based fill simulation.
"""
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


@dataclass
class OrderbookTick:
    """Single orderbook tick for one side (up or down)."""
    timestamp_ns: int
    side: str  # "up" or "down"
    best_bid: float | None
    best_ask: float | None
    mid: float | None

    @property
    def has_bid(self) -> bool:
        return self.best_bid is not None and not np.isnan(self.best_bid)

    @property
    def has_ask(self) -> bool:
        return self.best_ask is not None and not np.isnan(self.best_ask)

    @property
    def has_both(self) -> bool:
        return self.has_bid and self.has_ask

    @property
    def spread(self) -> float | None:
        if self.has_both:
            return self.best_ask - self.best_bid
        return None


@dataclass
class MarketSnapshot:
    """Combined market state with both up and down sides."""
    timestamp_ns: int
    up: OrderbookTick | None = None
    down: OrderbookTick | None = None

    @property
    def has_up(self) -> bool:
        return self.up is not None and self.up.has_both

    @property
    def has_down(self) -> bool:
        return self.down is not None and self.down.has_both


class OrderbookCSVLoader:
    """
    Loads orderbook tick data from CSV files.

    Expected CSV format:
        time,side,best_bid,best_ask,mid
        2026-01-13 00:00:28.938804,up,0.54,0.55,0.545
        2026-01-13 00:00:28.941203,down,0.45,0.46,0.455
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.logger = logger.bind(component="orderbook_loader", path=csv_path)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """
        Load CSV and convert timestamps to nanoseconds.

        Returns:
            DataFrame with columns: timestamp_ns, side, best_bid, best_ask, mid
        """
        if self._df is not None:
            return self._df

        self.logger.info("Loading orderbook CSV")

        df = pd.read_csv(self.csv_path)

        # Convert time to nanoseconds
        df['timestamp_ns'] = pd.to_datetime(df['time']).astype(np.int64)

        # Sort by timestamp
        df = df.sort_values('timestamp_ns').reset_index(drop=True)

        self.logger.info(
            "Loaded orderbook data",
            rows=len(df),
            start=df['time'].iloc[0] if len(df) > 0 else None,
            end=df['time'].iloc[-1] if len(df) > 0 else None,
        )

        self._df = df
        return df

    def get_side_df(self, side: str) -> pd.DataFrame:
        """Get DataFrame filtered to a specific side (up or down)."""
        df = self.load()
        return df[df['side'] == side].reset_index(drop=True)

    def iter_ticks(self, side: str) -> Iterator[OrderbookTick]:
        """
        Iterate over ticks for a specific side.

        Args:
            side: "up" or "down"

        Yields:
            OrderbookTick for each row
        """
        df = self.get_side_df(side)

        for _, row in df.iterrows():
            yield OrderbookTick(
                timestamp_ns=row['timestamp_ns'],
                side=row['side'],
                best_bid=row['best_bid'] if pd.notna(row['best_bid']) else None,
                best_ask=row['best_ask'] if pd.notna(row['best_ask']) else None,
                mid=row['mid'] if pd.notna(row['mid']) else None,
            )

    def iter_snapshots(self) -> Iterator[MarketSnapshot]:
        """
        Iterate over market snapshots combining up and down sides.

        Groups consecutive up/down ticks that are within 10ms of each other
        into a single MarketSnapshot.

        Yields:
            MarketSnapshot with both up and down ticks when available
        """
        df = self.load()

        # Group by rounded timestamp (10ms buckets)
        df['bucket'] = (df['timestamp_ns'] // 10_000_000) * 10_000_000

        for bucket, group in df.groupby('bucket'):
            snapshot = MarketSnapshot(timestamp_ns=int(bucket))

            for _, row in group.iterrows():
                tick = OrderbookTick(
                    timestamp_ns=row['timestamp_ns'],
                    side=row['side'],
                    best_bid=row['best_bid'] if pd.notna(row['best_bid']) else None,
                    best_ask=row['best_ask'] if pd.notna(row['best_ask']) else None,
                    mid=row['mid'] if pd.notna(row['mid']) else None,
                )

                if row['side'] == 'up':
                    snapshot.up = tick
                elif row['side'] == 'down':
                    snapshot.down = tick

            yield snapshot

    def get_stats(self) -> dict[str, any]:
        """Get statistics about the loaded data."""
        df = self.load()

        up_df = df[df['side'] == 'up']
        down_df = df[df['side'] == 'down']

        stats = {
            'total_rows': len(df),
            'up_rows': len(up_df),
            'down_rows': len(down_df),
            'start_time': df['time'].iloc[0] if len(df) > 0 else None,
            'end_time': df['time'].iloc[-1] if len(df) > 0 else None,
            'duration_hours': (
                (df['timestamp_ns'].iloc[-1] - df['timestamp_ns'].iloc[0]) / 3600e9
                if len(df) > 1 else 0
            ),
        }

        # Up side stats
        if len(up_df) > 0:
            valid_up = up_df[up_df['best_bid'].notna() & up_df['best_ask'].notna()]
            if len(valid_up) > 0:
                stats['up_avg_mid'] = valid_up['mid'].mean()
                stats['up_avg_spread'] = (valid_up['best_ask'] - valid_up['best_bid']).mean()
                stats['up_valid_rows'] = len(valid_up)

        # Down side stats
        if len(down_df) > 0:
            valid_down = down_df[down_df['best_bid'].notna() & down_df['best_ask'].notna()]
            if len(valid_down) > 0:
                stats['down_avg_mid'] = valid_down['mid'].mean()
                stats['down_avg_spread'] = (valid_down['best_ask'] - valid_down['best_bid']).mean()
                stats['down_valid_rows'] = len(valid_down)

        return stats


def load_multiple_assets(
    csv_dir: str,
    assets: list[str],
    filename_pattern: str = "orderbook_{asset}_jan13_19.csv",
) -> dict[str, OrderbookCSVLoader]:
    """
    Load orderbook data for multiple assets.

    Args:
        csv_dir: Directory containing CSV files
        assets: List of asset names (e.g., ["btc", "eth", "sol", "xrp"])
        filename_pattern: Pattern for CSV filenames with {asset} placeholder

    Returns:
        Dict mapping asset name to OrderbookCSVLoader
    """
    loaders = {}

    for asset in assets:
        filename = filename_pattern.format(asset=asset)
        csv_path = f"{csv_dir}/{filename}"

        try:
            loader = OrderbookCSVLoader(csv_path)
            loader.load()  # Verify file can be loaded
            loaders[asset] = loader
        except Exception as e:
            logger.warning("Failed to load asset", asset=asset, error=str(e))

    return loaders
