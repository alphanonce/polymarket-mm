#!/usr/bin/env python3
"""
Datalake Backtest Runner

CLI tool for running backtests using processed datalake data.

Usage:
    python python/backtest/run_datalake_backtest.py \
        --symbols btc eth \
        --start-date 2026-01-18 \
        --side up \
        --model inventory \
        --base-spread 0.02 \
        --output results.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.datalake_engine import (
    DatalakeBacktestConfig,
    DatalakeBacktestEngine,
    DatalakeBacktestResult,
    print_summary,
)
from backtest.datalake_loader import DatalakeLoader
from strategy.models.quote import (
    InventoryAdjustedQuoteConfig,
    InventoryAdjustedQuoteModel,
    SpreadQuoteConfig,
    SpreadQuoteModel,
)
from strategy.models.quote_tpbs import TpBSQuoteConfig, TpBSQuoteModel
from strategy.models.quote_tpsl_bs import TpslBSQuoteConfig, TpslBSQuoteModel
from strategy.models.size import (
    FixedSizeConfig,
    FixedSizeModel,
    InventoryBasedSizeConfig,
    InventoryBasedSizeModel,
)

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


def create_quote_model(
    model_type: str,
    base_spread: float = 0.02,
    inventory_skew: float = 0.5,
    max_inventory: float = 100.0,
    max_position_pct: float = 0.20,
    max_z: float = 4.0,
    min_z: float = 2.0,
    implied_volatility: float = 0.5,
    z: float = 0.5,
    tp_ticks: int = 5,
    sl_ticks: int = 0,
) -> Any:
    """Create a quote model based on type."""
    if model_type == "spread":
        config = SpreadQuoteConfig(
            base_spread=base_spread,
            min_spread=0.01,
            max_spread=0.10,
        )
        return SpreadQuoteModel(config=config)

    elif model_type == "inventory":
        config = InventoryAdjustedQuoteConfig(
            base_spread=base_spread,
            min_spread=0.01,
            max_spread=0.15,
            inventory_skew=inventory_skew,
            max_inventory=max_inventory,
        )
        return InventoryAdjustedQuoteModel(config=config)

    elif model_type == "tpbs":
        config = TpBSQuoteConfig(
            max_z=max_z,
            min_z=min_z,
            max_position_pct=max_position_pct,
            implied_volatility=implied_volatility,
            vol_mode="max",
            min_spread=0.01,
            enforce_maker=True,
        )
        return TpBSQuoteModel(config=config)

    elif model_type == "tpsl_bs":
        config = TpslBSQuoteConfig(
            z=z,
            tp_ticks=tp_ticks,
            sl_ticks=sl_ticks,
            max_position_pct=max_position_pct,
            tau_seconds=0.1,  # 100ms unhedgeable horizon
            implied_volatility=implied_volatility,
            vol_mode="rv",  # Use realized volatility from price history
            time_to_expiry_years=15.0 / 60 / 24 / 365,  # 15 minutes
        )
        return TpslBSQuoteModel(config=config)

    else:
        raise ValueError(f"Unknown quote model type: {model_type}")


def create_size_model(
    model_type: str,
    base_size: float = 10.0,
    max_position: float = 100.0,
) -> Any:
    """Create a size model based on type."""
    if model_type == "fixed":
        config = FixedSizeConfig(
            base_size=base_size,
            max_size=max_position,
            min_size=5.0,
        )
        return FixedSizeModel(config=config)

    elif model_type == "inventory":
        config = InventoryBasedSizeConfig(
            base_size=base_size,
            max_size=max_position,
            min_size=5.0,
            max_position=max_position,
            size_reduction_rate=0.5,
            asymmetric_scaling=True,
        )
        return InventoryBasedSizeModel(config=config)

    else:
        raise ValueError(f"Unknown size model type: {model_type}")


def result_to_dict(result: DatalakeBacktestResult) -> dict[str, Any]:
    """Convert backtest result to JSON-serializable dict."""
    return {
        "config": {
            "data_dir": result.config.data_dir,
            "symbols": result.config.symbols,
            "start_date": result.config.start_date,
            "end_date": result.config.end_date,
            "side": result.config.side,
            "maker_fee": result.config.maker_fee,
            "max_position": result.config.max_position,
            "initial_equity": result.config.initial_equity,
        },
        "aggregate": {
            "total_markets": result.total_markets,
            "total_pnl": result.total_pnl,
            "maker_rebate": result.total_maker_rebate,
            "pnl_after_rebate": result.pnl_after_rebate,
            "total_fills": result.total_fills,
            "total_volume": result.total_volume,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
        },
        "markets": {
            slug: {
                "symbol": mr.info.symbol,
                "date": mr.info.date,
                "pnl": mr.metrics.total_pnl,
                "maker_rebate": mr.metrics.maker_rebate,
                "realized_pnl": mr.metrics.realized_pnl,
                "unrealized_pnl": mr.metrics.unrealized_pnl,
                "fills": mr.metrics.total_fills,
                "buy_fills": mr.metrics.buy_fills,
                "sell_fills": mr.metrics.sell_fills,
                "volume": mr.metrics.total_volume,
                "max_position": mr.metrics.max_position,
                "min_position": mr.metrics.min_position,
                "max_drawdown": mr.metrics.max_drawdown,
                "avg_spread": mr.metrics.avg_spread,
                "book_ticks": mr.metrics.book_ticks,
                "trade_ticks": mr.metrics.trade_ticks,
            }
            for slug, mr in result.market_results.items()
        },
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run backtest using processed datalake data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data selection
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/datalake/processed/15m",
        help="Path to processed datalake data",
    )
    parser.add_argument(
        "--rtds-price-dir",
        type=str,
        default="data/datalake/global/rtds_crypto_prices",
        help="Path to RTDS price data (for external prices)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        type=str,
        default=None,
        help="Symbols to backtest (e.g., btc eth sol). Default: all",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["up", "down", "both"],
        default="up",
        help="Which side to trade (default: up)",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        choices=["spread", "inventory", "tpbs", "tpsl_bs"],
        default="inventory",
        help="Quote model type (default: inventory)",
    )
    parser.add_argument(
        "--size-model",
        type=str,
        choices=["fixed", "inventory"],
        default="inventory",
        help="Size model type (default: inventory)",
    )
    parser.add_argument(
        "--base-spread",
        type=float,
        default=0.02,
        help="Base spread (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--inventory-skew",
        type=float,
        default=0.5,
        help="Inventory skew factor (default: 0.5)",
    )
    parser.add_argument(
        "--base-size",
        type=float,
        default=10.0,
        help="Base order size (default: 10)",
    )
    parser.add_argument(
        "--max-position",
        type=float,
        default=100.0,
        help="Maximum position size (default: 100)",
    )

    # TpBS model parameters
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.20,
        help="Max position as %% of equity for TpBS (default: 0.20 = 20%%)",
    )
    parser.add_argument(
        "--max-z",
        type=float,
        default=4.0,
        help="Max z-score for TpBS (wide spread at neutral, default: 4.0)",
    )
    parser.add_argument(
        "--min-z",
        type=float,
        default=2.0,
        help="Min z-score for TpBS (tight spread at full position, default: 2.0)",
    )
    parser.add_argument(
        "--implied-vol",
        type=float,
        default=0.5,
        help="Implied volatility for TpBS (default: 0.5 = 50%%)",
    )

    # TpSL-BS model parameters
    parser.add_argument(
        "--z",
        type=float,
        default=0.5,
        help="Z-score for base spread in TpSL-BS (default: 0.5)",
    )
    parser.add_argument(
        "--tp-ticks",
        type=int,
        default=5,
        help="Take-profit tick offset for TpSL-BS (default: 5)",
    )
    parser.add_argument(
        "--sl-ticks",
        type=int,
        default=0,
        help="Stop-loss tick offset for TpSL-BS (default: 0, disabled)",
    )

    # Fees and parameters
    parser.add_argument(
        "--maker-fee",
        type=float,
        default=0.0,
        help="Maker fee rate (default: 0.0)",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=10000.0,
        help="Initial equity (default: 10000)",
    )
    parser.add_argument(
        "--quote-interval",
        type=int,
        default=100,
        help="Quote refresh interval in ms (default: 100)",
    )

    # Output
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # Discovery mode
    parser.add_argument(
        "--list-markets",
        action="store_true",
        help="List available markets and exit",
    )

    args = parser.parse_args()

    # List markets mode
    if args.list_markets:
        loader = DatalakeLoader(
            data_dir=args.data_dir,
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        stats = loader.get_stats()
        print("\nAvailable Data:")
        print(f"  Total Markets: {stats['total_markets']}")
        print(f"  Symbols: {', '.join(stats['symbols'])}")
        print(f"  Dates: {', '.join(stats['dates'][:5])}{'...' if len(stats['dates']) > 5 else ''}")
        return 0

    # Create models
    quote_model = create_quote_model(
        model_type=args.model,
        base_spread=args.base_spread,
        inventory_skew=args.inventory_skew,
        max_inventory=args.max_position,
        max_position_pct=args.max_position_pct,
        max_z=args.max_z,
        min_z=args.min_z,
        implied_volatility=args.implied_vol,
        z=args.z,
        tp_ticks=args.tp_ticks,
        sl_ticks=args.sl_ticks,
    )

    size_model = create_size_model(
        model_type=args.size_model,
        base_size=args.base_size,
        max_position=args.max_position,
    )

    # Create config
    config = DatalakeBacktestConfig(
        data_dir=args.data_dir,
        rtds_price_dir=args.rtds_price_dir,
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        side=args.side,
        maker_fee=args.maker_fee,
        initial_equity=args.initial_equity,
        max_position=args.max_position,
        quote_refresh_interval_ns=args.quote_interval * 1_000_000,  # ms to ns
        verbose=args.verbose,
    )

    # Run backtest
    engine = DatalakeBacktestEngine(
        config=config,
        quote_model=quote_model,
        size_model=size_model,
    )

    result = engine.run()

    # Print summary
    print_summary(result)

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(result_to_dict(result), f, indent=2)

        logger.info("Results saved", path=str(output_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
