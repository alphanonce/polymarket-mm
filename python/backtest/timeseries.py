"""
Timeseries Data Structures for Backtest Dashboard

Defines data structures for storing and analyzing time-series backtest data.
Supports hourly period aggregation for visualization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesPoint:
    """Single point in the backtest timeseries."""

    timestamp_ns: int  # Nanosecond timestamp
    trade_price: float  # Market trade price
    bid_quote: float  # Our bid quote
    ask_quote: float  # Our ask quote
    position: float  # Current position
    realized_pnl: float  # Realized PnL so far
    unrealized_pnl: float  # Unrealized PnL at this point
    fill_side: int  # 0=no fill, 1=bid fill (buy), -1=ask fill (sell)
    fill_price: float  # Price at which fill occurred (0 if no fill)
    fill_size: float  # Size of fill (0 if no fill)


@dataclass
class HourlyPeriodResult:
    """Result for a single 1-hour period."""

    asset: str
    period_start: datetime  # e.g., 2024-01-15 14:00:00 UTC
    period_end: datetime  # e.g., 2024-01-15 15:00:00 UTC

    # Position metrics
    start_position: float
    final_position: float

    # PnL metrics
    total_pnl: float  # Realized + Unrealized at end
    realized_pnl: float  # Only realized PnL
    unrealized_pnl: float  # Mark-to-market at period end

    # Volume metrics
    volume: float  # Total volume traded
    n_fills: int  # Number of fills
    n_trades: int  # Number of market trades observed

    # Quote metrics
    avg_spread: float  # Average spread we quoted
    n_quote_updates: int  # Number of quote updates

    # Time series data for charting
    timeseries: List[TimeSeriesPoint] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization (without timeseries)."""
        return {
            "asset": self.asset,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "start_position": self.start_position,
            "final_position": self.final_position,
            "total_pnl": self.total_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "volume": self.volume,
            "n_fills": self.n_fills,
            "n_trades": self.n_trades,
            "avg_spread": self.avg_spread,
            "n_quote_updates": self.n_quote_updates,
        }


@dataclass
class EnhancedAssetResult:
    """Enhanced asset result with hourly breakdown."""

    asset: str

    # Aggregate metrics (same as AssetResult)
    n_trades: int
    n_fills: int
    total_volume: float
    total_pnl: float
    realized_pnl: float
    max_position: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    avg_fill_price: float

    # Hourly breakdown
    hourly_periods: List[HourlyPeriodResult] = field(default_factory=list)

    # Legacy pnl_history for compatibility
    pnl_history: np.ndarray = field(default_factory=lambda: np.array([]))

    def get_period_dataframe(self) -> pd.DataFrame:
        """Convert hourly periods to DataFrame for analysis."""
        if not self.hourly_periods:
            return pd.DataFrame()

        rows = [p.to_dict() for p in self.hourly_periods]
        df = pd.DataFrame(rows)
        df["period_start"] = pd.to_datetime(df["period_start"])
        df["period_end"] = pd.to_datetime(df["period_end"])
        return df.sort_values("period_start").reset_index(drop=True)

    def get_summary_stats(self) -> dict:
        """Get summary statistics across all periods."""
        if not self.hourly_periods:
            return {}

        pnls = [p.total_pnl for p in self.hourly_periods]
        volumes = [p.volume for p in self.hourly_periods]
        fills = [p.n_fills for p in self.hourly_periods]

        return {
            "n_periods": len(self.hourly_periods),
            "avg_pnl_per_period": np.mean(pnls),
            "std_pnl_per_period": np.std(pnls),
            "total_pnl": sum(pnls),
            "avg_volume_per_period": np.mean(volumes),
            "total_volume": sum(volumes),
            "avg_fills_per_period": np.mean(fills),
            "total_fills": sum(fills),
            "profitable_periods": sum(1 for p in pnls if p > 0),
            "period_win_rate": sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0,
        }


@dataclass
class EnhancedBacktestReport:
    """Enhanced backtest report with hourly analysis."""

    run_id: str  # Unique identifier for this run
    config: dict  # Config as dict for JSON serialization
    asset_results: dict  # Dict[str, EnhancedAssetResult]

    # Aggregate metrics
    total_trades: int = 0
    total_fills: int = 0
    total_volume: float = 0.0
    total_pnl: float = 0.0
    avg_sharpe: float = 0.0
    avg_win_rate: float = 0.0

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    def compute_aggregates(self):
        """Compute aggregate metrics from asset results."""
        if not self.asset_results:
            return

        self.total_trades = sum(r.n_trades for r in self.asset_results.values())
        self.total_fills = sum(r.n_fills for r in self.asset_results.values())
        self.total_volume = sum(r.total_volume for r in self.asset_results.values())
        self.total_pnl = sum(r.total_pnl for r in self.asset_results.values())

        sharpes = [
            r.sharpe_ratio for r in self.asset_results.values() if not np.isnan(r.sharpe_ratio)
        ]
        win_rates = [r.win_rate for r in self.asset_results.values() if r.win_rate > 0]

        self.avg_sharpe = np.mean(sharpes) if sharpes else 0.0
        self.avg_win_rate = np.mean(win_rates) if win_rates else 0.0

    def get_all_periods(self) -> pd.DataFrame:
        """Get all hourly periods across all assets as DataFrame."""
        all_periods = []
        for asset, result in self.asset_results.items():
            for period in result.hourly_periods:
                row = period.to_dict()
                all_periods.append(row)

        if not all_periods:
            return pd.DataFrame()

        df = pd.DataFrame(all_periods)
        df["period_start"] = pd.to_datetime(df["period_start"])
        df["period_end"] = pd.to_datetime(df["period_end"])
        return df.sort_values(["asset", "period_start"]).reset_index(drop=True)

    def to_summary_dict(self) -> dict:
        """Convert to summary dict for JSON (excludes timeseries)."""
        return {
            "run_id": self.run_id,
            "config": self.config,
            "total_trades": self.total_trades,
            "total_fills": self.total_fills,
            "total_volume": self.total_volume,
            "total_pnl": self.total_pnl,
            "avg_sharpe": self.avg_sharpe,
            "avg_win_rate": self.avg_win_rate,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "assets": list(self.asset_results.keys()),
            "asset_summaries": {
                asset: result.get_summary_stats()
                for asset, result in self.asset_results.items()
            },
        }


def timeseries_to_dataframe(timeseries: List[TimeSeriesPoint]) -> pd.DataFrame:
    """Convert timeseries points to DataFrame for analysis/plotting."""
    if not timeseries:
        return pd.DataFrame()

    rows = [
        {
            "timestamp_ns": p.timestamp_ns,
            "trade_price": p.trade_price,
            "bid_quote": p.bid_quote,
            "ask_quote": p.ask_quote,
            "position": p.position,
            "realized_pnl": p.realized_pnl,
            "unrealized_pnl": p.unrealized_pnl,
            "total_pnl": p.realized_pnl + p.unrealized_pnl,
            "fill_side": p.fill_side,
            "fill_price": p.fill_price,
            "fill_size": p.fill_size,
        }
        for p in timeseries
    ]
    df = pd.DataFrame(rows)

    # Auto-detect timestamp unit based on magnitude
    if len(df) > 0:
        sample_ts = df["timestamp_ns"].iloc[0]
        if sample_ts > 1e15:
            unit = "ns"
        elif sample_ts > 1e12:
            unit = "ms"
        else:
            unit = "s"
        df["timestamp"] = pd.to_datetime(df["timestamp_ns"], unit=unit)
    else:
        df["timestamp"] = pd.to_datetime([])

    return df


def split_trades_by_hour(trades_df: pd.DataFrame) -> dict:
    """
    Split trades DataFrame by hour.

    Returns dict mapping hour start datetime to trades DataFrame.
    """
    if trades_df.empty:
        return {}

    # Convert timestamp to datetime and floor to hour
    trades_df = trades_df.copy()

    # Auto-detect timestamp unit based on magnitude
    # Nanoseconds: ~1.7e18 for 2025
    # Seconds: ~1.7e9 for 2025
    sample_ts = trades_df["timestamp"].iloc[0]
    if sample_ts > 1e15:
        unit = "ns"
    elif sample_ts > 1e12:
        unit = "ms"
    else:
        unit = "s"

    trades_df["_datetime"] = pd.to_datetime(trades_df["timestamp"], unit=unit)
    trades_df["_hour"] = trades_df["_datetime"].dt.floor("h")

    result = {}
    for hour, group in trades_df.groupby("_hour"):
        group = group.drop(columns=["_datetime", "_hour"])
        result[hour.to_pydatetime()] = group.reset_index(drop=True)

    return result
