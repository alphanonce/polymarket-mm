"""
Backtest Package

Provides backtesting functionality for market-making strategies.
"""

from backtest.data_loader import (
    DataLoader,
    LocalParquetLoader,
    MarketData,
    S3TradesLoader,
    Trade,
)
from backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    TradesBacktestConfig,
    TradesBacktestEngine,
    TradesBacktestResult,
)
from backtest.engine_fast import (
    AssetResult,
    BacktestReport,
    EnhancedBacktestEngine,
    FastBacktestConfig,
    FastBacktestEngine,
)
from backtest.simulator import Fill, MarketSimulator
from backtest.storage import BacktestStorage
from backtest.timeseries import (
    EnhancedAssetResult,
    EnhancedBacktestReport,
    HourlyPeriodResult,
    TimeSeriesPoint,
)
from backtest.visualize import (
    generate_report,
    plot_pnl_curves,
    plot_results,
    print_summary_table,
    results_to_dataframe,
)

__all__ = [
    "AssetResult",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestReport",
    "BacktestResult",
    "BacktestStorage",
    "DataLoader",
    "EnhancedAssetResult",
    "EnhancedBacktestEngine",
    "EnhancedBacktestReport",
    "FastBacktestConfig",
    "FastBacktestEngine",
    "Fill",
    "HourlyPeriodResult",
    "LocalParquetLoader",
    "MarketData",
    "MarketSimulator",
    "S3TradesLoader",
    "TimeSeriesPoint",
    "Trade",
    "TradesBacktestConfig",
    "TradesBacktestEngine",
    "TradesBacktestResult",
    "generate_report",
    "plot_pnl_curves",
    "plot_results",
    "print_summary_table",
    "results_to_dataframe",
]
