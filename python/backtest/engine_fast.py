"""
Fast Backtest Engine

Vectorized backtest engine for trades-based backtesting.
Uses numpy for efficient computation instead of row-by-row iteration.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
import yaml

from backtest.timeseries import (
    EnhancedAssetResult,
    EnhancedBacktestReport,
    HourlyPeriodResult,
    TimeSeriesPoint,
    split_trades_by_hour,
)

logger = structlog.get_logger()


@dataclass
class FastBacktestConfig:
    """Configuration for fast backtest."""

    # Data source
    s3_bucket: str = "an-trading-research"
    s3_prefix: str = "polymarket/trades/crypto/1h"
    local_cache_dir: Optional[str] = "data/s3_cache"

    # Asset filter
    assets: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Simulation
    maker_fee: float = 0.0
    taker_fee: float = 0.001

    # Strategy
    max_position: float = 100.0
    quote_refresh_sec: float = 60.0
    base_spread: float = 0.02
    base_size: float = 10.0

    # Performance
    n_workers: int = 4  # Parallel workers

    @classmethod
    def from_yaml(cls, path: str = "data/config/backtest.yaml") -> "FastBacktestConfig":
        """Create config from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        data_cfg = cfg.get("data", {})
        filter_cfg = cfg.get("filter", {})
        sim_cfg = cfg.get("simulation", {})
        strat_cfg = cfg.get("strategy", {})

        return cls(
            s3_bucket=data_cfg.get("s3_bucket", "an-trading-research"),
            s3_prefix=data_cfg.get("s3_prefix", "polymarket/trades/crypto/1h"),
            local_cache_dir=data_cfg.get("local_cache_dir", "data/s3_cache"),
            assets=filter_cfg.get("assets"),
            start_date=filter_cfg.get("start_date"),
            end_date=filter_cfg.get("end_date"),
            maker_fee=float(sim_cfg.get("maker_fee", 0.0)),
            taker_fee=float(sim_cfg.get("taker_fee", 0.001)),
            max_position=float(strat_cfg.get("max_position", 100.0)),
            quote_refresh_sec=float(strat_cfg.get("quote_refresh_sec", 60)),
            base_spread=float(strat_cfg.get("base_spread", 0.02)),
            base_size=float(strat_cfg.get("base_size", 10.0)),
        )


@dataclass
class AssetResult:
    """Result for a single asset."""

    asset: str
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
    pnl_history: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class BacktestReport:
    """Complete backtest report."""

    config: FastBacktestConfig
    asset_results: Dict[str, AssetResult]

    # Aggregate metrics
    total_trades: int = 0
    total_fills: int = 0
    total_volume: float = 0.0
    total_pnl: float = 0.0
    avg_sharpe: float = 0.0
    avg_win_rate: float = 0.0

    def compute_aggregates(self):
        """Compute aggregate metrics from asset results."""
        if not self.asset_results:
            return

        self.total_trades = sum(r.n_trades for r in self.asset_results.values())
        self.total_fills = sum(r.n_fills for r in self.asset_results.values())
        self.total_volume = sum(r.total_volume for r in self.asset_results.values())
        self.total_pnl = sum(r.total_pnl for r in self.asset_results.values())

        sharpes = [r.sharpe_ratio for r in self.asset_results.values() if not np.isnan(r.sharpe_ratio)]
        win_rates = [r.win_rate for r in self.asset_results.values() if r.win_rate > 0]

        self.avg_sharpe = np.mean(sharpes) if sharpes else 0.0
        self.avg_win_rate = np.mean(win_rates) if win_rates else 0.0


def run_vectorized_backtest(
    trades_df: pd.DataFrame,
    asset: str,
    config: FastBacktestConfig,
) -> AssetResult:
    """
    Run vectorized backtest for a single asset.

    Uses numpy operations for efficiency.
    """
    if trades_df.empty:
        return AssetResult(
            asset=asset, n_trades=0, n_fills=0, total_volume=0,
            total_pnl=0, realized_pnl=0, max_position=0, max_drawdown=0,
            sharpe_ratio=0, win_rate=0, avg_fill_price=0,
        )

    # Extract arrays
    timestamps = trades_df["timestamp"].values
    prices = trades_df["price"].values.astype(np.float64)
    sizes = trades_df["outcome_tokens_amount"].values.astype(np.float64)

    n_trades = len(prices)

    # Initialize state
    position = 0.0
    avg_entry = 0.0
    realized_pnl = 0.0
    total_volume = 0.0
    n_fills = 0

    # Track for metrics
    max_position = 0.0
    pnl_history = []
    fill_prices = []

    # Quote state
    last_quote_time = 0
    bid_price = 0.0
    ask_price = 0.0
    bid_size = config.base_size
    ask_size = config.base_size

    quote_refresh_ns = int(config.quote_refresh_sec * 1e9)
    half_spread = config.base_spread / 2

    # Process trades
    for i in range(n_trades):
        ts = timestamps[i]
        price = prices[i]
        size = sizes[i]

        # Update quotes periodically
        if ts - last_quote_time >= quote_refresh_ns or bid_price == 0:
            mid = price
            bid_price = mid * (1 - half_spread)
            ask_price = mid * (1 + half_spread)

            # Adjust sizes based on position
            pos_ratio = abs(position) / config.max_position if config.max_position > 0 else 0
            bid_size = config.base_size * (1 - pos_ratio) if position >= 0 else config.base_size
            ask_size = config.base_size * (1 - pos_ratio) if position <= 0 else config.base_size

            last_quote_time = ts

        # Check for fills using crossing logic
        # trade_price < our_bid → bid fills (we buy)
        if price < bid_price and bid_size > 0:
            fill_size = min(bid_size, size, config.max_position - position)
            if fill_size > 0:
                # Update position (buying)
                if position >= 0:
                    total_cost = avg_entry * position + bid_price * fill_size
                    position += fill_size
                    avg_entry = total_cost / position if position > 0 else 0
                else:
                    # Covering short
                    cover = min(-position, fill_size)
                    realized_pnl += (avg_entry - bid_price) * cover
                    position += fill_size
                    if position > 0:
                        avg_entry = bid_price

                total_volume += fill_size
                n_fills += 1
                fill_prices.append(bid_price)
                bid_size -= fill_size

        # trade_price > our_ask → ask fills (we sell)
        elif price > ask_price and ask_size > 0:
            fill_size = min(ask_size, size, config.max_position + position)
            if fill_size > 0:
                # Update position (selling)
                if position <= 0:
                    total_cost = avg_entry * (-position) + ask_price * fill_size
                    position -= fill_size
                    avg_entry = total_cost / (-position) if position < 0 else 0
                else:
                    # Closing long
                    close = min(position, fill_size)
                    realized_pnl += (ask_price - avg_entry) * close
                    position -= fill_size
                    if position < 0:
                        avg_entry = ask_price

                total_volume += fill_size
                n_fills += 1
                fill_prices.append(ask_price)
                ask_size -= fill_size

        # Track metrics
        max_position = max(max_position, abs(position))

        # Sample PnL periodically
        if i % 1000 == 0:
            unrealized = 0.0
            if position > 0:
                unrealized = (price - avg_entry) * position
            elif position < 0:
                unrealized = (avg_entry - price) * (-position)
            pnl_history.append(realized_pnl + unrealized)

    # Final PnL
    final_price = prices[-1] if len(prices) > 0 else 0
    unrealized_pnl = 0.0
    if position > 0:
        unrealized_pnl = (final_price - avg_entry) * position
    elif position < 0:
        unrealized_pnl = (avg_entry - final_price) * (-position)

    total_pnl = realized_pnl + unrealized_pnl

    # Compute metrics
    pnl_arr = np.array(pnl_history) if pnl_history else np.array([0.0])

    # Sharpe ratio
    if len(pnl_arr) > 1:
        returns = np.diff(pnl_arr)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365.25 * 24) if np.std(returns) > 0 else 0
    else:
        sharpe = 0.0

    # Max drawdown
    cummax = np.maximum.accumulate(pnl_arr)
    drawdown = cummax - pnl_arr
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    # Win rate (simplified: profitable fills)
    win_rate = 0.5  # Placeholder - would need more tracking

    avg_fill = np.mean(fill_prices) if fill_prices else 0

    return AssetResult(
        asset=asset,
        n_trades=n_trades,
        n_fills=n_fills,
        total_volume=total_volume,
        total_pnl=total_pnl,
        realized_pnl=realized_pnl,
        max_position=max_position,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        win_rate=win_rate,
        avg_fill_price=avg_fill,
        pnl_history=pnl_arr,
    )


def _process_asset(args: Tuple) -> AssetResult:
    """Worker function for parallel processing."""
    asset, df, config = args
    return run_vectorized_backtest(df, asset, config)


class FastBacktestEngine:
    """
    Fast backtest engine with vectorized operations and parallel processing.
    """

    def __init__(self, config: FastBacktestConfig):
        self.config = config
        self.logger = logger.bind(component="fast_backtest")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from S3 grouped by asset."""
        import re
        import boto3
        import io

        s3 = boto3.client("s3")
        bucket = self.config.s3_bucket
        prefix = self.config.s3_prefix

        # Pattern for monthly files: YYYY-MM.parquet (not combined files like btc.parquet)
        monthly_pattern = re.compile(r"^\d{4}-\d{2}(_delta_\d+)?\.parquet$")

        # List all parquet files
        paginator = s3.get_paginator("list_objects_v2")
        files_by_asset: Dict[str, List[str]] = {}

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".parquet"):
                    continue

                parts = key.split("/")
                filename = parts[-1]

                # Skip combined files (e.g., btc.parquet), only use monthly files
                if not monthly_pattern.match(filename):
                    continue

                # Skip delta files for now (incremental updates)
                if "_delta_" in filename:
                    continue

                if len(parts) >= 5:
                    asset = parts[-2]  # e.g., btc

                    # Filter by asset
                    if self.config.assets and asset not in self.config.assets:
                        continue

                    # Filter by date (extract YYYY-MM from filename)
                    file_month = filename.replace(".parquet", "").split("_")[0]
                    if self.config.start_date and file_month < self.config.start_date:
                        continue
                    if self.config.end_date and file_month > self.config.end_date:
                        continue

                    if asset not in files_by_asset:
                        files_by_asset[asset] = []
                    files_by_asset[asset].append(key)

        self.logger.info("Files to load", files={k: len(v) for k, v in files_by_asset.items()})

        # Load data
        data: Dict[str, pd.DataFrame] = {}
        for asset, keys in files_by_asset.items():
            dfs = []
            for key in sorted(keys):
                self.logger.info("Loading", key=key)
                obj = s3.get_object(Bucket=bucket, Key=key)
                df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
                dfs.append(df)
                self.logger.info("Loaded", key=key, rows=len(df))

            if dfs:
                data[asset] = pd.concat(dfs, ignore_index=True)
                data[asset] = data[asset].sort_values("timestamp").reset_index(drop=True)
                self.logger.info("Asset loaded", asset=asset, total_rows=len(data[asset]))

        return data

    def run(self) -> BacktestReport:
        """Run backtest across all assets."""
        self.logger.info("Loading data...")
        data = self.load_data()
        self.logger.info("Data loaded", assets=list(data.keys()))

        # Run backtest for each asset
        asset_results: Dict[str, AssetResult] = {}

        if self.config.n_workers > 1 and len(data) > 1:
            # Parallel processing
            self.logger.info("Running parallel backtest", workers=self.config.n_workers)
            args_list = [(asset, df, self.config) for asset, df in data.items()]

            with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
                futures = {executor.submit(_process_asset, args): args[0] for args in args_list}
                for future in as_completed(futures):
                    asset = futures[future]
                    try:
                        result = future.result()
                        asset_results[asset] = result
                        self.logger.info("Asset done", asset=asset, fills=result.n_fills, pnl=result.total_pnl)
                    except Exception as e:
                        self.logger.error("Asset failed", asset=asset, error=str(e))
        else:
            # Sequential processing
            for asset, df in data.items():
                self.logger.info("Processing", asset=asset, rows=len(df))
                result = run_vectorized_backtest(df, asset, self.config)
                asset_results[asset] = result
                self.logger.info("Asset done", asset=asset, fills=result.n_fills, pnl=result.total_pnl)

        report = BacktestReport(config=self.config, asset_results=asset_results)
        report.compute_aggregates()

        return report


def run_enhanced_backtest_for_period(
    trades_df: pd.DataFrame,
    asset: str,
    period_start: datetime,
    config: FastBacktestConfig,
    initial_position: float = 0.0,
    initial_avg_entry: float = 0.0,
    initial_realized_pnl: float = 0.0,
    sample_rate: int = 100,  # Sample 1 in N ticks (fills always sampled)
) -> Tuple[HourlyPeriodResult, float, float, float]:
    """
    Run enhanced backtest for a single period with timeseries collection.

    Returns:
        - HourlyPeriodResult with timeseries data
        - Final position (for next period)
        - Final avg_entry (for next period)
        - Final realized_pnl (for next period)
    """
    period_end = period_start + timedelta(hours=1)

    if trades_df.empty:
        return (
            HourlyPeriodResult(
                asset=asset,
                period_start=period_start,
                period_end=period_end,
                start_position=initial_position,
                final_position=initial_position,
                total_pnl=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                volume=0.0,
                n_fills=0,
                n_trades=0,
                avg_spread=config.base_spread,
                n_quote_updates=0,
                timeseries=[],
            ),
            initial_position,
            initial_avg_entry,
            initial_realized_pnl,
        )

    # Extract arrays
    timestamps = trades_df["timestamp"].values
    prices = trades_df["price"].values.astype(np.float64)
    sizes = trades_df["outcome_tokens_amount"].values.astype(np.float64)

    n_trades = len(prices)

    # Initialize state from previous period
    position = initial_position
    avg_entry = initial_avg_entry
    realized_pnl = initial_realized_pnl
    period_start_realized = initial_realized_pnl  # Track pnl delta for this period

    total_volume = 0.0
    n_fills = 0

    # Quote state
    last_quote_time = 0
    bid_price = 0.0
    ask_price = 0.0
    bid_size = config.base_size
    ask_size = config.base_size

    quote_refresh_ns = int(config.quote_refresh_sec * 1e9)
    half_spread = config.base_spread / 2

    # Tracking
    spreads = []
    n_quote_updates = 0
    start_position = position

    # Timeseries collection
    timeseries: List[TimeSeriesPoint] = []

    # Process trades
    for i in range(n_trades):
        ts = timestamps[i]
        price = prices[i]
        size = sizes[i]

        # Update quotes periodically
        quote_updated = False
        if ts - last_quote_time >= quote_refresh_ns or bid_price == 0:
            mid = price
            bid_price = mid * (1 - half_spread)
            ask_price = mid * (1 + half_spread)

            # Adjust sizes based on position
            pos_ratio = abs(position) / config.max_position if config.max_position > 0 else 0
            bid_size = config.base_size * (1 - pos_ratio) if position >= 0 else config.base_size
            ask_size = config.base_size * (1 - pos_ratio) if position <= 0 else config.base_size

            last_quote_time = ts
            spreads.append(ask_price - bid_price)
            n_quote_updates += 1
            quote_updated = True

        # Track fill for this tick
        fill_side = 0
        fill_price = 0.0
        fill_size = 0.0

        # Check for fills using crossing logic
        if price < bid_price and bid_size > 0:
            fill_qty = min(bid_size, size, config.max_position - position)
            if fill_qty > 0:
                # Update position (buying)
                if position >= 0:
                    total_cost = avg_entry * position + bid_price * fill_qty
                    position += fill_qty
                    avg_entry = total_cost / position if position > 0 else 0
                else:
                    cover = min(-position, fill_qty)
                    realized_pnl += (avg_entry - bid_price) * cover
                    position += fill_qty
                    if position > 0:
                        avg_entry = bid_price

                total_volume += fill_qty
                n_fills += 1
                bid_size -= fill_qty
                fill_side = 1
                fill_price = bid_price
                fill_size = fill_qty

        elif price > ask_price and ask_size > 0:
            fill_qty = min(ask_size, size, config.max_position + position)
            if fill_qty > 0:
                if position <= 0:
                    total_cost = avg_entry * (-position) + ask_price * fill_qty
                    position -= fill_qty
                    avg_entry = total_cost / (-position) if position < 0 else 0
                else:
                    close = min(position, fill_qty)
                    realized_pnl += (ask_price - avg_entry) * close
                    position -= fill_qty
                    if position < 0:
                        avg_entry = ask_price

                total_volume += fill_qty
                n_fills += 1
                ask_size -= fill_qty
                fill_side = -1
                fill_price = ask_price
                fill_size = fill_qty

        # Calculate unrealized PnL
        unrealized = 0.0
        if position > 0:
            unrealized = (price - avg_entry) * position
        elif position < 0:
            unrealized = (avg_entry - price) * (-position)

        # Sample timeseries: always sample fills, otherwise sample at rate
        if fill_side != 0 or i % sample_rate == 0:
            timeseries.append(
                TimeSeriesPoint(
                    timestamp_ns=int(ts),
                    trade_price=price,
                    bid_quote=bid_price,
                    ask_quote=ask_price,
                    position=position,
                    realized_pnl=realized_pnl,
                    unrealized_pnl=unrealized,
                    fill_side=fill_side,
                    fill_price=fill_price,
                    fill_size=fill_size,
                )
            )

    # Final calculations
    final_price = prices[-1] if len(prices) > 0 else 0
    unrealized_pnl = 0.0
    if position > 0:
        unrealized_pnl = (final_price - avg_entry) * position
    elif position < 0:
        unrealized_pnl = (avg_entry - final_price) * (-position)

    period_realized = realized_pnl - period_start_realized
    total_pnl = period_realized + unrealized_pnl

    avg_spread = np.mean(spreads) if spreads else config.base_spread

    result = HourlyPeriodResult(
        asset=asset,
        period_start=period_start,
        period_end=period_end,
        start_position=start_position,
        final_position=position,
        total_pnl=total_pnl,
        realized_pnl=period_realized,
        unrealized_pnl=unrealized_pnl,
        volume=total_volume,
        n_fills=n_fills,
        n_trades=n_trades,
        avg_spread=avg_spread,
        n_quote_updates=n_quote_updates,
        timeseries=timeseries,
    )

    return result, position, avg_entry, realized_pnl


def run_enhanced_backtest(
    trades_df: pd.DataFrame,
    asset: str,
    config: FastBacktestConfig,
    sample_rate: int = 100,
) -> EnhancedAssetResult:
    """
    Run enhanced backtest with hourly period breakdown and timeseries collection.
    """
    if trades_df.empty:
        return EnhancedAssetResult(
            asset=asset,
            n_trades=0,
            n_fills=0,
            total_volume=0,
            total_pnl=0,
            realized_pnl=0,
            max_position=0,
            max_drawdown=0,
            sharpe_ratio=0,
            win_rate=0,
            avg_fill_price=0,
            hourly_periods=[],
            pnl_history=np.array([]),
        )

    # Split trades by hour
    hourly_trades = split_trades_by_hour(trades_df)

    if not hourly_trades:
        return EnhancedAssetResult(
            asset=asset,
            n_trades=len(trades_df),
            n_fills=0,
            total_volume=0,
            total_pnl=0,
            realized_pnl=0,
            max_position=0,
            max_drawdown=0,
            sharpe_ratio=0,
            win_rate=0,
            avg_fill_price=0,
            hourly_periods=[],
            pnl_history=np.array([]),
        )

    # Sort hours chronologically
    sorted_hours = sorted(hourly_trades.keys())

    # State carried across periods
    position = 0.0
    avg_entry = 0.0
    realized_pnl = 0.0

    hourly_periods: List[HourlyPeriodResult] = []
    all_fill_prices = []
    max_position = 0.0
    pnl_samples = []

    for hour in sorted_hours:
        hour_df = hourly_trades[hour]

        period_result, position, avg_entry, realized_pnl = run_enhanced_backtest_for_period(
            hour_df,
            asset,
            hour,
            config,
            initial_position=position,
            initial_avg_entry=avg_entry,
            initial_realized_pnl=realized_pnl,
            sample_rate=sample_rate,
        )

        hourly_periods.append(period_result)
        max_position = max(max_position, abs(period_result.final_position))

        # Collect fill prices from timeseries
        for ts in period_result.timeseries:
            if ts.fill_side != 0:
                all_fill_prices.append(ts.fill_price)
            # Sample PnL for sharpe calculation
            pnl_samples.append(ts.realized_pnl + ts.unrealized_pnl)

    # Aggregate metrics
    total_trades = sum(p.n_trades for p in hourly_periods)
    total_fills = sum(p.n_fills for p in hourly_periods)
    total_volume = sum(p.volume for p in hourly_periods)
    total_pnl = sum(p.total_pnl for p in hourly_periods)
    total_realized = sum(p.realized_pnl for p in hourly_periods)

    # Sharpe ratio from PnL samples
    pnl_arr = np.array(pnl_samples) if pnl_samples else np.array([0.0])
    if len(pnl_arr) > 1:
        returns = np.diff(pnl_arr)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365.25 * 24) if np.std(returns) > 0 else 0
    else:
        sharpe = 0.0

    # Max drawdown
    cummax = np.maximum.accumulate(pnl_arr)
    drawdown = cummax - pnl_arr
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    # Win rate (periods with positive PnL)
    profitable = sum(1 for p in hourly_periods if p.total_pnl > 0)
    win_rate = profitable / len(hourly_periods) if hourly_periods else 0

    avg_fill = np.mean(all_fill_prices) if all_fill_prices else 0

    return EnhancedAssetResult(
        asset=asset,
        n_trades=total_trades,
        n_fills=total_fills,
        total_volume=total_volume,
        total_pnl=total_pnl,
        realized_pnl=total_realized,
        max_position=max_position,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        win_rate=win_rate,
        avg_fill_price=avg_fill,
        hourly_periods=hourly_periods,
        pnl_history=pnl_arr,
    )


def _process_enhanced_asset(args: Tuple) -> EnhancedAssetResult:
    """Worker function for parallel processing of enhanced backtest."""
    asset, df, config, sample_rate = args
    return run_enhanced_backtest(df, asset, config, sample_rate)


class EnhancedBacktestEngine:
    """
    Enhanced backtest engine with hourly period breakdown and timeseries collection.
    """

    def __init__(self, config: FastBacktestConfig, sample_rate: int = 100):
        self.config = config
        self.sample_rate = sample_rate
        self.logger = logger.bind(component="enhanced_backtest")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from S3 grouped by asset. (Delegates to FastBacktestEngine)"""
        engine = FastBacktestEngine(self.config)
        return engine.load_data()

    def run(self, run_id: Optional[str] = None) -> EnhancedBacktestReport:
        """Run enhanced backtest across all assets."""
        import uuid

        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

        start_time = datetime.now()
        self.logger.info("Starting enhanced backtest", run_id=run_id)

        self.logger.info("Loading data...")
        data = self.load_data()
        self.logger.info("Data loaded", assets=list(data.keys()))

        # Run backtest for each asset
        asset_results: Dict[str, EnhancedAssetResult] = {}

        if self.config.n_workers > 1 and len(data) > 1:
            self.logger.info("Running parallel enhanced backtest", workers=self.config.n_workers)
            args_list = [(asset, df, self.config, self.sample_rate) for asset, df in data.items()]

            with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
                futures = {
                    executor.submit(_process_enhanced_asset, args): args[0] for args in args_list
                }
                for future in as_completed(futures):
                    asset = futures[future]
                    try:
                        result = future.result()
                        asset_results[asset] = result
                        self.logger.info(
                            "Asset done",
                            asset=asset,
                            fills=result.n_fills,
                            pnl=result.total_pnl,
                            periods=len(result.hourly_periods),
                        )
                    except Exception as e:
                        self.logger.error("Asset failed", asset=asset, error=str(e))
        else:
            for asset, df in data.items():
                self.logger.info("Processing", asset=asset, rows=len(df))
                result = run_enhanced_backtest(df, asset, self.config, self.sample_rate)
                asset_results[asset] = result
                self.logger.info(
                    "Asset done",
                    asset=asset,
                    fills=result.n_fills,
                    pnl=result.total_pnl,
                    periods=len(result.hourly_periods),
                )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Convert config to dict
        config_dict = {
            "s3_bucket": self.config.s3_bucket,
            "s3_prefix": self.config.s3_prefix,
            "assets": self.config.assets,
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "maker_fee": self.config.maker_fee,
            "taker_fee": self.config.taker_fee,
            "max_position": self.config.max_position,
            "quote_refresh_sec": self.config.quote_refresh_sec,
            "base_spread": self.config.base_spread,
            "base_size": self.config.base_size,
            "sample_rate": self.sample_rate,
        }

        report = EnhancedBacktestReport(
            run_id=run_id,
            config=config_dict,
            asset_results=asset_results,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
        )
        report.compute_aggregates()

        self.logger.info(
            "Enhanced backtest complete",
            run_id=run_id,
            duration_seconds=duration,
            total_pnl=report.total_pnl,
            total_fills=report.total_fills,
        )

        return report
