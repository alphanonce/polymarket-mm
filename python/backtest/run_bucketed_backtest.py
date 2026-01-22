#!/usr/bin/env python3
"""
Time-Bucketed Backtest Runner

Runs time-bucketed backtest that prevents look-ahead bias by:
1. Aggregating data into 100ms buckets
2. Using last BBO from time T to set quotes
3. Checking fills against quotes at time T+1

Supports both orderbook and trades data sources.
Uses pluggable QuoteModel and SizeModel for strategy logic.

Usage:
    # Run with default models (2% spread)
    cd python && uv run python -m backtest.run_bucketed_backtest --source trades

    # Run with custom spread
    cd python && uv run python -m backtest.run_bucketed_backtest --source trades --spread 0.01

    # Run with inventory-adjusted quoting
    cd python && uv run python -m backtest.run_bucketed_backtest --source trades --spread 0.01 --inventory-skew 0.5
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine_bucketed import (
    BucketedBacktestConfig,
    BucketedBacktestEngine,
    BucketedBacktestResult,
    DataSource,
    create_default_quote_model,
    create_default_size_model,
    print_bucketed_summary,
    run_bucketed_backtest,
)
from strategy.models.base import NormalizationConfig, QuoteModel, SizeModel
from strategy.models.quote import (
    InventoryAdjustedQuoteConfig,
    InventoryAdjustedQuoteModel,
    SpreadQuoteConfig,
    SpreadQuoteModel,
)
from strategy.models.size import (
    FixedSizeConfig,
    FixedSizeModel,
    InventoryBasedSizeConfig,
    InventoryBasedSizeModel,
)


def create_quote_model(
    base_spread: float,
    inventory_skew: float,
    max_position: float,
) -> QuoteModel:
    """Create quote model based on parameters."""
    normalization = NormalizationConfig(
        clamp_prices=False,
        use_dynamic_tick=False,
    )

    if inventory_skew > 0:
        config = InventoryAdjustedQuoteConfig(
            base_spread=base_spread,
            inventory_skew=inventory_skew,
            max_inventory=max_position,
        )
        return InventoryAdjustedQuoteModel(config=config, normalization=normalization)
    else:
        config = SpreadQuoteConfig(base_spread=base_spread)
        return SpreadQuoteModel(config=config, normalization=normalization)


def create_size_model(
    base_size: float,
    max_position: float,
    use_inventory_sizing: bool = True,
) -> SizeModel:
    """Create size model based on parameters."""
    normalization = NormalizationConfig(
        enforce_min_size=False,
        round_sizes=False,
    )

    if use_inventory_sizing:
        config = InventoryBasedSizeConfig(
            base_size=base_size,
            max_position=max_position,
            max_size=base_size * 2,
            min_size=1.0,
        )
        return InventoryBasedSizeModel(config=config, normalization=normalization)
    else:
        config = FixedSizeConfig(
            base_size=base_size,
            max_size=base_size * 2,
            min_size=1.0,
        )
        return FixedSizeModel(config=config, normalization=normalization)


def run_datalake_backtest(
    data_dir: str,
    source: DataSource,
    symbols: list[str] | None = None,
    bucket_ms: float = 100.0,
    base_spread: float = 0.02,
    base_size: float = 10.0,
    max_position: float = 100.0,
    inventory_skew: float = 0.0,
) -> dict[str, BucketedBacktestResult]:
    """
    Run bucketed backtest on datalake data.

    Args:
        data_dir: Path to datalake timebased data
        source: Data source type (orderbook or trades)
        symbols: List of symbols to backtest
        bucket_ms: Bucket size in milliseconds
        base_spread: Base spread for quotes
        base_size: Base order size
        max_position: Maximum position
        inventory_skew: Inventory skew factor for quote model

    Returns:
        Dict mapping slug to BucketedBacktestResult
    """
    bucket_ns = int(bucket_ms * 1_000_000)  # Convert ms to ns

    config = BucketedBacktestConfig(
        bucket_size_ns=bucket_ns,
        data_source=source,
    )

    # Create models
    quote_model = create_quote_model(base_spread, inventory_skew, max_position)
    size_model = create_size_model(base_size, max_position)

    # Get model type names for display
    quote_type = "InventoryAdjusted" if inventory_skew > 0 else "Spread"
    size_type = "InventoryBased"

    print("=" * 60)
    print("TIME-BUCKETED BACKTEST")
    print("=" * 60)
    print(f"Data Directory: {data_dir}")
    print(f"Data Source: {source.value}")
    print(f"Symbols: {symbols or 'all'}")
    print(f"Bucket Size: {bucket_ms}ms")
    print()
    print("QUOTE MODEL:")
    print(f"  Type: {quote_type}QuoteModel")
    print(f"  Base Spread: {base_spread * 100:.1f}%")
    if inventory_skew > 0:
        print(f"  Inventory Skew: {inventory_skew}")
    print()
    print("SIZE MODEL:")
    print(f"  Type: {size_type}SizeModel")
    print(f"  Base Size: {base_size}")
    print(f"  Max Position: {max_position}")
    print()

    engine = BucketedBacktestEngine(
        config=config,
        quote_model=quote_model,
        size_model=size_model,
        data_dir=data_dir,
        symbols=symbols,
    )

    results = engine.run()
    return results


def run_single_file_backtest(
    file_path: str,
    source: DataSource,
    bucket_ms: float = 100.0,
    base_spread: float = 0.02,
    base_size: float = 10.0,
    max_position: float = 100.0,
    inventory_skew: float = 0.0,
) -> BucketedBacktestResult:
    """
    Run bucketed backtest on a single file.

    Args:
        file_path: Path to parquet file
        source: Data source type
        bucket_ms: Bucket size in milliseconds
        base_spread: Base spread for quotes
        base_size: Base order size
        max_position: Maximum position
        inventory_skew: Inventory skew factor

    Returns:
        BucketedBacktestResult
    """
    import gzip
    import io

    import pandas as pd

    bucket_ns = int(bucket_ms * 1_000_000)

    config = BucketedBacktestConfig(
        bucket_size_ns=bucket_ns,
        data_source=source,
    )

    # Create models
    quote_model = create_quote_model(base_spread, inventory_skew, max_position)
    size_model = create_size_model(base_size, max_position)

    # Extract asset/slug from path
    path = Path(file_path)
    filename = path.name
    slug = filename.replace("book_", "").replace("last_trade_price_", "").replace(".parquet.gz", "").replace(".parquet", "")
    asset = slug.split("-")[0] if "-" in slug else path.stem

    # Get model type names for display
    quote_type = "InventoryAdjusted" if inventory_skew > 0 else "Spread"

    print("=" * 60)
    print("TIME-BUCKETED BACKTEST (Single File)")
    print("=" * 60)
    print(f"File: {file_path}")
    print(f"Asset: {asset}")
    print(f"Slug: {slug}")
    print(f"Data Source: {source.value}")
    print(f"Bucket Size: {bucket_ms}ms")
    print()
    print("QUOTE MODEL:")
    print(f"  Type: {quote_type}QuoteModel")
    print(f"  Base Spread: {base_spread * 100:.1f}%")
    if inventory_skew > 0:
        print(f"  Inventory Skew: {inventory_skew}")
    print()

    # Handle gzipped parquet files
    if file_path.endswith(".gz"):
        with gzip.open(file_path, "rb") as f:
            df = pd.read_parquet(io.BytesIO(f.read()))
    else:
        df = pd.read_parquet(file_path)

    print(f"Loaded {len(df)} rows")

    result = run_bucketed_backtest(
        df=df,
        config=config,
        quote_model=quote_model,
        size_model=size_model,
        asset=asset,
        slug=slug,
    )

    return result


def print_single_result(result: BucketedBacktestResult) -> None:
    """Print detailed results for a single backtest."""
    print("\n" + "=" * 60)
    print(f"RESULTS: {result.asset} / {result.slug}")
    print("=" * 60)

    print("\nPERFORMANCE:")
    print(f"  Total PnL: ${result.total_pnl:,.4f}")
    print(f"  Realized PnL: ${result.realized_pnl:,.4f}")
    print(f"  Unrealized PnL: ${result.unrealized_pnl:,.4f}")
    print(f"  Max Drawdown: ${result.max_drawdown:,.4f}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")

    print("\nEXECUTION:")
    print(f"  Buckets Processed: {result.n_buckets:,}")
    print(f"  Total Fills: {result.n_fills}")
    print(f"  Buy Fills: {result.buy_fills}")
    print(f"  Sell Fills: {result.sell_fills}")
    print(f"  Total Volume: {result.total_volume:,.2f}")

    print("\nPOSITION:")
    print(f"  Final Position: {result.final_position:,.2f}")
    print(f"  Max Position: {result.max_position:,.2f}")

    # Print sample fills
    if result.fills:
        print("\nSAMPLE FILLS (first 10):")
        for fill in result.fills[:10]:
            side_str = "BUY " if fill.side == 1 else "SELL"
            print(
                f"  {side_str} {fill.size:6.2f} @ {fill.price:.4f} "
                f"-> pos={fill.position_after:6.2f} realized={fill.realized_pnl:+.4f}"
            )

        if len(result.fills) > 10:
            print(f"  ... and {len(result.fills) - 10} more fills")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run time-bucketed backtest with pluggable quote/size models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default models (2% spread)
  python -m backtest.run_bucketed_backtest --source trades

  # Run with 1% spread
  python -m backtest.run_bucketed_backtest --source trades --spread 0.01

  # Run with inventory-adjusted quoting
  python -m backtest.run_bucketed_backtest --source trades --spread 0.01 --inventory-skew 0.5

  # Run on specific symbols
  python -m backtest.run_bucketed_backtest --source trades --symbols btc eth

  # Run on a single file
  python -m backtest.run_bucketed_backtest --source trades --file path/to/file.parquet.gz
        """,
    )

    # Data source options
    parser.add_argument(
        "--source",
        type=str,
        default="trades",
        choices=["orderbook", "trades"],
        help="Data source type (default: trades)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/datalake/timebased/crypto/updown/15m",
        help="Path to datalake timebased data directory",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Single parquet file to backtest (overrides --data-dir)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Symbols to backtest (e.g., btc eth sol)",
    )

    # Strategy parameters
    parser.add_argument(
        "--bucket-ms",
        type=float,
        default=100.0,
        help="Bucket size in milliseconds (default: 100)",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=0.02,
        help="Base spread for quote model (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=10.0,
        help="Base order size (default: 10.0)",
    )
    parser.add_argument(
        "--max-position",
        type=float,
        default=100.0,
        help="Maximum position (default: 100.0)",
    )
    parser.add_argument(
        "--inventory-skew",
        type=float,
        default=0.0,
        help="Inventory skew factor for quote model (default: 0.0 = no skew)",
    )

    args = parser.parse_args()

    # Parse data source
    source = DataSource.ORDERBOOK if args.source == "orderbook" else DataSource.TRADES

    if args.file:
        # Single file mode
        result = run_single_file_backtest(
            file_path=args.file,
            source=source,
            bucket_ms=args.bucket_ms,
            base_spread=args.spread,
            base_size=args.size,
            max_position=args.max_position,
            inventory_skew=args.inventory_skew,
        )
        print_single_result(result)
    else:
        # Directory mode
        results = run_datalake_backtest(
            data_dir=args.data_dir,
            source=source,
            symbols=args.symbols,
            bucket_ms=args.bucket_ms,
            base_spread=args.spread,
            base_size=args.size,
            max_position=args.max_position,
            inventory_skew=args.inventory_skew,
        )
        print_bucketed_summary(results)


if __name__ == "__main__":
    main()
