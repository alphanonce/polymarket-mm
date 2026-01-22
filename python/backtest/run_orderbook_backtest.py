#!/usr/bin/env python3
"""
Orderbook Backtest Runner

Runs tick-by-tick orderbook backtest using CSV data from:
    /tmp/polymarket-cache/orderbook_{asset}_jan13_19.csv

Usage:
    python3 backtest/run_orderbook_backtest.py
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.orderbook_engine import (
    OrderbookBacktestConfig,
    OrderbookBacktestEngine,
    OrderbookBacktestResult,
    print_summary,
    run_multi_asset_backtest,
)
from strategy.models.quote import (
    InventoryAdjustedQuoteConfig,
    InventoryAdjustedQuoteModel,
)
from strategy.models.size import FixedSizeConfig, FixedSizeModel


def run_simple_backtest(
    csv_dir: str = "/tmp/polymarket-cache",
    assets: list | None = None,
    side: str = "up",
) -> dict:
    """
    Run a simple inventory-adjusted market-making backtest.

    Args:
        csv_dir: Directory containing CSV files
        assets: List of assets to test (default: all available)
        side: Which side to trade ("up" or "down")

    Returns:
        Dict of results by asset
    """
    if assets is None:
        assets = ["btc", "eth", "sol", "xrp"]

    # Simple inventory-adjusted quote model
    quote_config = InventoryAdjustedQuoteConfig(
        base_spread=0.02,  # 2% base spread
        inventory_skew=0.1,  # Inventory skew
        min_spread=0.01,   # 1% minimum spread
        max_inventory=100.0,  # Max inventory for full skew
    )
    quote_model = InventoryAdjustedQuoteModel(config=quote_config)

    # Fixed size model
    size_config = FixedSizeConfig(base_size=10.0)
    size_model = FixedSizeModel(config=size_config)

    # Config overrides
    config_overrides = {
        'maker_fee': 0.0,
        'taker_fee': 0.0,
        'initial_equity': 10000.0,
        'max_position': 100.0,
        'quote_refresh_interval_ns': 100_000_000,  # 100ms
        'log_interval': 500_000,
    }

    print("Running orderbook backtest...")
    print(f"  CSV Dir: {csv_dir}")
    print(f"  Assets: {assets}")
    print(f"  Side: {side}")
    print(f"  Base Spread: {quote_config.base_spread * 100:.1f}%")
    print(f"  Min Spread: {quote_config.min_spread * 100:.1f}%")
    print(f"  Inventory Skew: {quote_config.inventory_skew}")
    print(f"  Base Size: {size_config.base_size}")
    print()

    results = run_multi_asset_backtest(
        csv_dir=csv_dir,
        assets=assets,
        quote_model=quote_model,
        size_model=size_model,
        side=side,
        config_overrides=config_overrides,
    )

    return results


def run_single_asset_backtest(
    asset: str = "btc",
    csv_dir: str = "/tmp/polymarket-cache",
    side: str = "up",
) -> OrderbookBacktestResult:
    """Run backtest for a single asset with detailed output."""
    csv_path = f"{csv_dir}/orderbook_{asset}_jan13_19.csv"

    # Quote model
    quote_config = InventoryAdjustedQuoteConfig(
        base_spread=0.02,
        inventory_skew=0.1,
        min_spread=0.01,
        max_inventory=100.0,
    )
    quote_model = InventoryAdjustedQuoteModel(config=quote_config)

    # Size model
    size_config = FixedSizeConfig(base_size=10.0)
    size_model = FixedSizeModel(config=size_config)

    # Config
    config = OrderbookBacktestConfig(
        csv_path=csv_path,
        asset=asset,
        side=side,
        maker_fee=0.0,
        taker_fee=0.0,
        initial_equity=10000.0,
        max_position=100.0,
        quote_refresh_interval_ns=100_000_000,  # 100ms
        log_interval=100_000,
    )

    print(f"\nRunning single asset backtest: {asset.upper()} ({side})")
    print(f"  CSV: {csv_path}")
    print()

    engine = OrderbookBacktestEngine(
        config=config,
        quote_model=quote_model,
        size_model=size_model,
    )

    result = engine.run()

    # Print detailed results
    m = result.metrics
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {asset.upper()} ({side})")
    print(f"{'=' * 60}")
    print("\nPERFORMANCE:")
    print(f"  Total PnL: ${m.total_pnl:,.2f}")
    print(f"  Realized PnL: ${m.realized_pnl:,.2f}")
    print(f"  Unrealized PnL: ${m.unrealized_pnl:,.2f}")
    print(f"  Max Drawdown: ${m.max_drawdown:,.2f}")

    print("\nEXECUTION:")
    print(f"  Total Fills: {m.total_fills:,}")
    print(f"  Buy Fills: {m.buy_fills:,}")
    print(f"  Sell Fills: {m.sell_fills:,}")
    print(f"  Total Volume: {m.total_volume:,.2f}")
    print(f"  Buy Volume: {m.buy_volume:,.2f}")
    print(f"  Sell Volume: {m.sell_volume:,.2f}")

    print("\nPOSITION:")
    print(f"  Final Position: {result.final_position.size:,.2f}")
    print(f"  Max Position: {m.max_position:,.2f}")
    print(f"  Min Position: {m.min_position:,.2f}")

    print("\nMARKET:")
    print(f"  Total Ticks: {m.total_ticks:,}")
    print(f"  Valid Ticks: {m.valid_ticks:,}")
    print(f"  Avg Spread: {m.avg_spread * 100:.3f}%")

    # Print sample fills
    if result.fills:
        print("\nSAMPLE FILLS (first 5):")
        for fill in result.fills[:5]:
            side_str = "BUY" if fill.side == 1 else "SELL"
            print(f"  {side_str} {fill.size:.2f} @ {fill.price:.4f}")

    return result


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run orderbook backtest")
    parser.add_argument(
        "--asset",
        type=str,
        default=None,
        help="Single asset to test (btc, eth, sol, xrp). If not specified, tests all.",
    )
    parser.add_argument(
        "--side",
        type=str,
        default="up",
        choices=["up", "down"],
        help="Which side to trade (up or down)",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default="/tmp/polymarket-cache",
        help="Directory containing orderbook CSV files",
    )

    args = parser.parse_args()

    if args.asset:
        # Single asset test
        run_single_asset_backtest(
            asset=args.asset,
            csv_dir=args.csv_dir,
            side=args.side,
        )
    else:
        # Multi-asset test
        results = run_simple_backtest(
            csv_dir=args.csv_dir,
            side=args.side,
        )
        print_summary(results)


if __name__ == "__main__":
    main()
