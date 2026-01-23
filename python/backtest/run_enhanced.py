"""
Enhanced Backtest Runner

Entry point for running enhanced backtest with timeseries collection
and saving results for dashboard visualization.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def main():
    parser = argparse.ArgumentParser(description="Run enhanced backtest with timeseries collection")
    parser.add_argument(
        "--config",
        type=str,
        default="data/config/backtest.yaml",
        help="Path to backtest config YAML",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=100,
        help="Timeseries sampling rate (1 in N ticks, fills always sampled)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/backtest_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID (auto-generated if not provided)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        default=True,
        help="Print summary table",
    )

    args = parser.parse_args()

    # Load environment
    load_env()

    # Import after env setup
    from backtest.engine_fast import EnhancedBacktestEngine, FastBacktestConfig
    from backtest.storage import BacktestStorage
    from backtest.visualize import print_summary_table
    import structlog

    logger = structlog.get_logger()

    # Resolve config path relative to project root
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root.parent / config_path

    logger.info("Loading config", path=str(config_path))

    # Load config
    config = FastBacktestConfig.from_yaml(str(config_path))
    logger.info(
        "Config loaded",
        assets=config.assets,
        start_date=config.start_date,
        end_date=config.end_date,
        spread=config.base_spread,
        size=config.base_size,
    )

    # Run enhanced backtest
    engine = EnhancedBacktestEngine(config, sample_rate=args.sample_rate)
    report = engine.run(run_id=args.run_id)

    # Print summary
    if args.print_summary:
        print("\n" + "=" * 80)
        print("ENHANCED BACKTEST RESULTS")
        print("=" * 80)
        print(f"\nRun ID: {report.run_id}")
        print(f"Duration: {report.duration_seconds:.1f}s")
        print()

        # Asset summary
        print(f"{'Asset':<10} {'Trades':>10} {'Fills':>8} {'Volume':>12} {'PnL':>12} {'Periods':>8}")
        print("-" * 70)
        for asset, result in report.asset_results.items():
            print(
                f"{asset.upper():<10} {result.n_trades:>10,} {result.n_fills:>8,} "
                f"{result.total_volume:>12,.0f} ${result.total_pnl:>11,.2f} "
                f"{len(result.hourly_periods):>8}"
            )
        print("-" * 70)
        print(
            f"{'TOTAL':<10} {report.total_trades:>10,} {report.total_fills:>8,} "
            f"{report.total_volume:>12,.0f} ${report.total_pnl:>11,.2f}"
        )
        print("=" * 80)

        # Period summary
        print("\nHOURLY PERIOD SUMMARY")
        print("-" * 50)
        for asset, result in report.asset_results.items():
            stats = result.get_summary_stats()
            print(f"\n{asset.upper()}:")
            print(f"  Periods: {stats.get('n_periods', 0)}")
            print(f"  Avg PnL/Period: ${stats.get('avg_pnl_per_period', 0):.2f}")
            print(f"  Profitable Periods: {stats.get('profitable_periods', 0)}/{stats.get('n_periods', 0)}")
            print(f"  Period Win Rate: {stats.get('period_win_rate', 0):.1%}")

    # Save results
    if not args.no_save:
        storage = BacktestStorage(args.output_dir)
        save_path = storage.save(report)
        print(f"\nResults saved to: {save_path}")
        print("\nTo view the dashboard, run:")
        print("  make run-dashboard")

    return report


if __name__ == "__main__":
    main()
