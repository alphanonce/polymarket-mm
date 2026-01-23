"""
Backtest Storage

Save and load backtest results to/from disk.
Uses JSON for metadata and Parquet for timeseries data.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import structlog

from backtest.timeseries import (
    EnhancedAssetResult,
    EnhancedBacktestReport,
    HourlyPeriodResult,
    TimeSeriesPoint,
)

logger = structlog.get_logger()


class BacktestStorage:
    """
    Storage manager for backtest results.

    Directory structure:
        data/backtest_results/{run_id}/
        ├── summary.json          # Config + aggregate metrics
        ├── {asset}_metrics.json  # Per-asset metrics and period summary
        └── {asset}_timeseries.parquet  # Timeseries data for each asset
    """

    def __init__(self, base_dir: str = "data/backtest_results"):
        self.base_dir = Path(base_dir)
        self.logger = logger.bind(component="backtest_storage")

    def save(self, report: EnhancedBacktestReport) -> Path:
        """Save backtest report to disk."""
        run_dir = self.base_dir / report.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = run_dir / "summary.json"
        summary = report.to_summary_dict()
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info("Saved summary", path=str(summary_path))

        # Save each asset's data
        for asset, result in report.asset_results.items():
            self._save_asset(run_dir, asset, result)

        self.logger.info("Saved backtest results", run_id=report.run_id, path=str(run_dir))
        return run_dir

    def _save_asset(self, run_dir: Path, asset: str, result: EnhancedAssetResult):
        """Save single asset's data."""
        # Save metrics JSON (without timeseries)
        metrics_path = run_dir / f"{asset}_metrics.json"
        metrics = {
            "asset": result.asset,
            "n_trades": result.n_trades,
            "n_fills": result.n_fills,
            "total_volume": result.total_volume,
            "total_pnl": result.total_pnl,
            "realized_pnl": result.realized_pnl,
            "max_position": result.max_position,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "win_rate": result.win_rate,
            "avg_fill_price": result.avg_fill_price,
            "n_periods": len(result.hourly_periods),
            "periods": [p.to_dict() for p in result.hourly_periods],
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Save timeseries as parquet (all periods combined)
        ts_path = run_dir / f"{asset}_timeseries.parquet"
        self._save_timeseries(ts_path, result.hourly_periods)

    def _save_timeseries(self, path: Path, periods: List[HourlyPeriodResult]):
        """Save all timeseries data from periods to parquet."""
        rows = []
        for period in periods:
            period_start = period.period_start
            for ts in period.timeseries:
                rows.append(
                    {
                        "period_start": period_start,
                        "timestamp_ns": ts.timestamp_ns,
                        "trade_price": ts.trade_price,
                        "bid_quote": ts.bid_quote,
                        "ask_quote": ts.ask_quote,
                        "position": ts.position,
                        "realized_pnl": ts.realized_pnl,
                        "unrealized_pnl": ts.unrealized_pnl,
                        "fill_side": ts.fill_side,
                        "fill_price": ts.fill_price,
                        "fill_size": ts.fill_size,
                    }
                )

        if rows:
            df = pd.DataFrame(rows)
            df.to_parquet(path, index=False)
            self.logger.debug("Saved timeseries", path=str(path), rows=len(df))
        else:
            # Save empty parquet with correct schema
            df = pd.DataFrame(
                columns=[
                    "period_start",
                    "timestamp_ns",
                    "trade_price",
                    "bid_quote",
                    "ask_quote",
                    "position",
                    "realized_pnl",
                    "unrealized_pnl",
                    "fill_side",
                    "fill_price",
                    "fill_size",
                ]
            )
            df.to_parquet(path, index=False)

    def load(self, run_id: str) -> Optional[EnhancedBacktestReport]:
        """Load backtest report from disk."""
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            self.logger.warning("Run not found", run_id=run_id)
            return None

        # Load summary
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            self.logger.warning("Summary not found", run_id=run_id)
            return None

        with open(summary_path) as f:
            summary = json.load(f)

        # Load each asset
        asset_results = {}
        for asset in summary.get("assets", []):
            result = self._load_asset(run_dir, asset)
            if result:
                asset_results[asset] = result

        # Reconstruct report
        report = EnhancedBacktestReport(
            run_id=run_id,
            config=summary.get("config", {}),
            asset_results=asset_results,
            total_trades=summary.get("total_trades", 0),
            total_fills=summary.get("total_fills", 0),
            total_volume=summary.get("total_volume", 0.0),
            total_pnl=summary.get("total_pnl", 0.0),
            avg_sharpe=summary.get("avg_sharpe", 0.0),
            avg_win_rate=summary.get("avg_win_rate", 0.0),
            start_time=(
                datetime.fromisoformat(summary["start_time"]) if summary.get("start_time") else None
            ),
            end_time=(
                datetime.fromisoformat(summary["end_time"]) if summary.get("end_time") else None
            ),
            duration_seconds=summary.get("duration_seconds", 0.0),
        )

        self.logger.info("Loaded backtest results", run_id=run_id)
        return report

    def _load_asset(self, run_dir: Path, asset: str) -> Optional[EnhancedAssetResult]:
        """Load single asset's data."""
        metrics_path = run_dir / f"{asset}_metrics.json"
        ts_path = run_dir / f"{asset}_timeseries.parquet"

        if not metrics_path.exists():
            return None

        with open(metrics_path) as f:
            metrics = json.load(f)

        # Load timeseries
        timeseries_by_period = {}
        if ts_path.exists():
            ts_df = pd.read_parquet(ts_path)
            if not ts_df.empty:
                for period_start, group in ts_df.groupby("period_start"):
                    timeseries_by_period[period_start] = [
                        TimeSeriesPoint(
                            timestamp_ns=int(row["timestamp_ns"]),
                            trade_price=row["trade_price"],
                            bid_quote=row["bid_quote"],
                            ask_quote=row["ask_quote"],
                            position=row["position"],
                            realized_pnl=row["realized_pnl"],
                            unrealized_pnl=row["unrealized_pnl"],
                            fill_side=int(row["fill_side"]),
                            fill_price=row["fill_price"],
                            fill_size=row["fill_size"],
                        )
                        for _, row in group.iterrows()
                    ]

        # Reconstruct periods
        hourly_periods = []
        for period_data in metrics.get("periods", []):
            period_start = datetime.fromisoformat(period_data["period_start"])
            period_end = datetime.fromisoformat(period_data["period_end"])

            # Get timeseries for this period
            ts_key = None
            for key in timeseries_by_period.keys():
                if isinstance(key, str):
                    key_dt = datetime.fromisoformat(key)
                else:
                    key_dt = pd.Timestamp(key).to_pydatetime()
                if key_dt == period_start:
                    ts_key = key
                    break

            timeseries = timeseries_by_period.get(ts_key, []) if ts_key else []

            period = HourlyPeriodResult(
                asset=asset,
                period_start=period_start,
                period_end=period_end,
                start_position=period_data.get("start_position", 0),
                final_position=period_data.get("final_position", 0),
                total_pnl=period_data.get("total_pnl", 0),
                realized_pnl=period_data.get("realized_pnl", 0),
                unrealized_pnl=period_data.get("unrealized_pnl", 0),
                volume=period_data.get("volume", 0),
                n_fills=period_data.get("n_fills", 0),
                n_trades=period_data.get("n_trades", 0),
                avg_spread=period_data.get("avg_spread", 0),
                n_quote_updates=period_data.get("n_quote_updates", 0),
                timeseries=timeseries,
            )
            hourly_periods.append(period)

        return EnhancedAssetResult(
            asset=asset,
            n_trades=metrics.get("n_trades", 0),
            n_fills=metrics.get("n_fills", 0),
            total_volume=metrics.get("total_volume", 0),
            total_pnl=metrics.get("total_pnl", 0),
            realized_pnl=metrics.get("realized_pnl", 0),
            max_position=metrics.get("max_position", 0),
            max_drawdown=metrics.get("max_drawdown", 0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0),
            win_rate=metrics.get("win_rate", 0),
            avg_fill_price=metrics.get("avg_fill_price", 0),
            hourly_periods=hourly_periods,
            pnl_history=np.array([]),  # Not stored
        )

    def list_runs(self) -> List[dict]:
        """List all available backtest runs."""
        runs = []
        if not self.base_dir.exists():
            return runs

        for run_dir in sorted(self.base_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue

            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue

            try:
                with open(summary_path) as f:
                    summary = json.load(f)
                    runs.append(
                        {
                            "run_id": run_dir.name,
                            "start_time": summary.get("start_time"),
                            "total_pnl": summary.get("total_pnl", 0),
                            "total_fills": summary.get("total_fills", 0),
                            "assets": summary.get("assets", []),
                            "config": summary.get("config", {}),
                        }
                    )
            except Exception as e:
                self.logger.warning("Failed to load run", run_id=run_dir.name, error=str(e))

        return runs

    def get_latest_run_id(self) -> Optional[str]:
        """Get the most recent run ID."""
        runs = self.list_runs()
        return runs[0]["run_id"] if runs else None
