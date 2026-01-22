"""
TPSL-BS Backtest Runner

Test the TPSL-BS quote model with real trade data from S3/local parquet files.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dataclasses import dataclass

import numpy as np

from backtest.engine import TradesBacktestConfig, TradesBacktestEngine, TradesBacktestResult
from backtest.market_parser import parse_market_slug
from strategy.models.quote_tpsl_bs import TpslBSQuoteConfig, TpslBSQuoteModel
from strategy.models.size import FixedSizeConfig, FixedSizeModel


@dataclass
class TpslBSBacktestConfig:
    """Configuration for TPSL-BS backtest."""

    # Data source
    local_data_dir: str = "data/s3_cache"  # Local parquet files
    assets: list[str] | None = None  # None = all assets
    start_date: str | None = None  # YYYY-MM format
    end_date: str | None = None

    # Simulation
    maker_fee: float = 0.0
    taker_fee: float = 0.001
    max_position: float = 50.0
    quote_refresh_interval_ns: int = 100_000_000  # 0.1 seconds (100ms)
    initial_equity: float = 10000.0

    # TPSL-BS model parameters
    z: float = 4.0  # Single z-score
    tp_ticks: int = 2  # Take-profit tick offset
    sl_ticks: int = 3  # Stop-loss tick offset
    max_position_pct: float = 0.20
    tau_seconds: float = 5.0
    vol_mode: str = "iv"
    implied_volatility: float = 0.5
    min_spread: float = 0.01
    strike: float = 0.5
    time_to_expiry_years: float = 0.1

    # Size model
    base_size: float = 5.0
    base_spread: float = 0.02

    # External price configuration (Binance)
    load_external_prices: bool = True  # Enable Binance price loading
    binance_cache_dir: str = "data/binance_cache"
    binance_interval: str = "1s"  # 1-second klines
    reference_price_symbol: str | None = None  # Auto-detect from market_slug if None

    # Market parameters (can be auto-detected from market_slug)
    market_slug: str | None = None  # e.g., "btc-above-100000-jan-20-1pm"


def run_tpsl_bs_backtest(config: TpslBSBacktestConfig) -> TradesBacktestResult:
    """Run backtest with TPSL-BS quote model using real trade data."""

    # Parse market_slug to get strike/expiry if available
    strike = config.strike
    time_to_expiry_years = config.time_to_expiry_years
    reference_price_symbol = config.reference_price_symbol

    if config.market_slug:
        market_params = parse_market_slug(config.market_slug)
        if market_params:
            # Use parsed strike (need to normalize for BS)
            strike = market_params.strike
            reference_price_symbol = market_params.binance_symbol

            # Calculate time to expiry based on expiry_time
            # Default to 1 hour for crypto/1h markets
            time_to_expiry_years = 1.0 / 24 / 365  # 1 hour

            print(f"Parsed market_slug: {config.market_slug}")
            print(f"  Asset: {market_params.asset}")
            print(f"  Direction: {market_params.direction}")
            print(f"  Strike: {strike}")
            print(f"  Binance symbol: {reference_price_symbol}")
            print(f"  Expiry: {market_params.expiry_time}")

    # Create TPSL-BS quote model
    tpsl_bs_config = TpslBSQuoteConfig(
        z=config.z,
        tp_ticks=config.tp_ticks,
        sl_ticks=config.sl_ticks,
        max_position_pct=config.max_position_pct,
        tau_seconds=config.tau_seconds,
        vol_mode=config.vol_mode,
        implied_volatility=config.implied_volatility,
        min_spread=config.min_spread,
        strike=strike,
        time_to_expiry_years=time_to_expiry_years,
        reference_price_symbol=reference_price_symbol,
    )
    quote_model = TpslBSQuoteModel(config=tpsl_bs_config)

    # Create size model
    size_config = FixedSizeConfig(base_size=config.base_size)
    size_model = FixedSizeModel(config=size_config)

    # Create trades backtest config
    bt_config = TradesBacktestConfig(
        local_data_dir=config.local_data_dir,
        assets=config.assets,
        start_date=config.start_date,
        end_date=config.end_date,
        maker_fee=config.maker_fee,
        taker_fee=config.taker_fee,
        max_position=config.max_position,
        quote_refresh_interval_ns=config.quote_refresh_interval_ns,
        base_spread=config.base_spread,
        base_size=config.base_size,
        load_external_prices=config.load_external_prices,
        binance_cache_dir=config.binance_cache_dir,
        binance_interval=config.binance_interval,
        initial_equity=config.initial_equity,
    )

    # Create and run engine
    engine = TradesBacktestEngine(
        config=bt_config,
        quote_model=quote_model,
        size_model=size_model,
    )

    return engine.run()


def print_results(result: TradesBacktestResult, config: TpslBSBacktestConfig):
    """Print backtest results."""
    metrics = result.metrics

    print("\n" + "=" * 70)
    print("TPSL-BS BACKTEST RESULTS")
    print("=" * 70)

    # Config summary
    print("\nCONFIG:")
    print(f"  Data Dir: {config.local_data_dir}")
    print(f"  Assets: {config.assets or 'All'}")
    print(f"  Max Position: {config.max_position}")
    print(f"  TPSL-BS z: {config.z}")
    print(f"  TP Ticks: {config.tp_ticks}")
    print(f"  SL Ticks: {config.sl_ticks}")
    print(f"  Vol Mode: {config.vol_mode} (IV={config.implied_volatility})")
    print(f"  Strike: {config.strike}")
    print(f"  Time to Expiry: {config.time_to_expiry_years:.6f} years")
    if config.market_slug:
        print(f"  Market Slug: {config.market_slug}")
    print(f"  External Prices: {'Enabled' if config.load_external_prices else 'Disabled'}")
    if config.reference_price_symbol:
        print(f"  Reference Symbol: {config.reference_price_symbol}")

    # Performance summary
    print("\nPERFORMANCE:")
    print(f"  Total PnL: ${metrics.total_pnl:,.2f}")
    print(f"  Realized PnL: ${metrics.realized_pnl:,.2f}")
    print(f"  Unrealized PnL: ${metrics.unrealized_pnl:,.2f}")
    print(f"  Max Drawdown: ${metrics.max_drawdown:,.2f}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

    # Trading summary
    buy_fills = sum(1 for f in result.fills if f.side == 1)
    sell_fills = sum(1 for f in result.fills if f.side == -1)
    print("\nTRADING:")
    print(f"  Total Fills: {metrics.total_fills:,}")
    print(f"  Buy Fills: {buy_fills:,}")
    print(f"  Sell Fills: {sell_fills:,}")
    print(f"  Total Volume: {metrics.total_volume:,.2f}")
    print(f"  Max Position: {metrics.max_position:,.2f}")
    print(f"  Win Rate: {metrics.win_rate:.1%}")

    # Position analysis (calculate from fills)
    print("\nPOSITION ANALYSIS:")
    position = 0.0
    positions = []
    for fill in result.fills:
        position += fill.size * fill.side
        positions.append(position)

    if positions:
        max_long = max(positions)
        max_short = min(positions)
        print(f"  Max Long Position: {max_long:,.2f}")
        print(f"  Max Short Position: {max_short:,.2f}")

        # Count position flips
        long_to_short = sum(1 for i in range(1, len(positions))
                          if positions[i-1] > 0 and positions[i] < 0)
        short_to_long = sum(1 for i in range(1, len(positions))
                          if positions[i-1] < 0 and positions[i] > 0)
        print(f"  Long->Short Flips: {long_to_short}")
        print(f"  Short->Long Flips: {short_to_long}")

    # Fill analysis
    if result.fills:
        print("\nFILL ANALYSIS:")
        buy_prices = [f.price for f in result.fills if f.side == 1]
        sell_prices = [f.price for f in result.fills if f.side == -1]

        if buy_prices:
            print(f"  Avg Buy Price: {np.mean(buy_prices):.4f}")
        if sell_prices:
            print(f"  Avg Sell Price: {np.mean(sell_prices):.4f}")
        if buy_prices and sell_prices:
            print(f"  Avg Spread Captured: {np.mean(sell_prices) - np.mean(buy_prices):.4f}")

    # Per-asset results
    if result.asset_metrics:
        print("\nPER-ASSET RESULTS:")
        for asset, asset_metrics in sorted(result.asset_metrics.items()):
            print(f"  {asset}: PnL=${asset_metrics.total_pnl:.2f}, Fills={asset_metrics.total_fills}")

    print("=" * 70)


def main():
    """Run TPSL-BS backtest with real data."""

    print("Running TPSL-BS Backtest with real trade data...")

    # Example market_slug - update this based on actual data
    market_slug = None  # Set to actual market slug if known

    config = TpslBSBacktestConfig(
        # Data source (local parquet files)
        local_data_dir="data/s3_cache",
        assets=None,  # Use all available assets
        start_date=None,
        end_date=None,

        # Trading limits
        max_position=100.0,
        quote_refresh_interval_ns=100_000_000,  # 0.1 seconds (100ms)
        initial_equity=10000.0,

        # TPSL-BS parameters
        z=4.0,  # Single z-score
        tp_ticks=2,  # Take-profit offset
        sl_ticks=3,  # Stop-loss offset
        max_position_pct=0.01,  # 1% of equity = 100 max position
        tau_seconds=5.0,
        vol_mode="iv",
        implied_volatility=0.5,
        min_spread=0.01,

        # Strike/Expiry (can be auto-detected from market_slug)
        strike=100000.0,  # BTC strike example
        time_to_expiry_years=1.0 / 24 / 365,  # 1 hour for crypto/1h markets

        # Size
        base_size=10.0,
        base_spread=0.02,

        # External prices (Binance)
        load_external_prices=True,
        binance_cache_dir="data/binance_cache",
        binance_interval="1s",
        reference_price_symbol="BTCUSDT",  # Auto-detect if market_slug is set

        # Market slug (optional - enables auto-detection of strike/expiry)
        market_slug=market_slug,
    )

    result = run_tpsl_bs_backtest(config)
    print_results(result, config)

    # Additional position analysis
    if result.fills:
        print("\nPOSITION PROGRESSION (first 20 fills):")
        position = 0.0
        for i, fill in enumerate(result.fills[:20]):
            position += fill.size * fill.side
            side_str = "BUY" if fill.side == 1 else "SELL"
            print(f"  Fill {i+1}: {side_str} {fill.size}@{fill.price:.4f} -> position={position:.0f}")

        # Track zero-crossings and SHORT territory
        position = 0.0
        crossings = []
        short_entries = []
        max_short_reached = None
        for i, fill in enumerate(result.fills):
            old_pos = position
            position += fill.size * fill.side
            if (old_pos > 0 and position <= 0) or (old_pos < 0 and position >= 0):
                crossings.append((i, old_pos, position))
            if old_pos >= 0 and position < 0:
                short_entries.append((i, old_pos, position))
            if position <= -90:
                if max_short_reached is None or position < max_short_reached[1]:
                    max_short_reached = (i, position, fill.price, fill.side)

        print(f"\nZERO CROSSINGS: {len(crossings)}")
        for i, old, new in crossings[:5]:
            print(f"  Fill {i}: {old:.0f} -> {new:.0f}")

        print(f"\nSHORT ENTRIES: {len(short_entries)}")
        for i, old, new in short_entries[:5]:
            print(f"  Fill {i}: {old:.0f} -> {new:.0f}")

        if max_short_reached:
            i, pos, price, side = max_short_reached
            side_str = "BUY" if side == 1 else "SELL"
            print(f"\nMAX SHORT at Fill {i}: position={pos:.0f}, {side_str}@{price:.4f}")

    return result


if __name__ == "__main__":
    main()
