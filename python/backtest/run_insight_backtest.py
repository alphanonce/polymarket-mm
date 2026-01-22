#!/usr/bin/env python3
"""
Insight-Driven Quote Model Backtest Runner

Run backtests using the InsightQuoteModel against datalake orderbook data.
Uses empirical market making insights for spread calculation:
- TTE-based spread widening
- Moneyness-based adjustment
- Asset-specific base spreads

Usage:
    cd python && uv run python backtest/run_insight_backtest.py
"""

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.datalake_loader import (
    DatalakeOrderbookLoader,
    MarketInfo,
    OrderbookMidPrice,
    ResolvedMarketInfo,
)
from backtest.simulator import (
    SIDE_BUY,
    SIDE_SELL,
    Fill,
    MarketSimulator,
    Position,
)
from strategy.models.quote_insight import InsightQuoteConfig, InsightQuoteModel
from strategy.models.size import InventoryBasedSizeConfig, InventoryBasedSizeModel
from strategy.models.base import StrategyState
from strategy.shm.types import MarketState, PositionState


@dataclass
class InsightBacktestConfig:
    """Configuration for Insight backtest."""

    # Data source
    datalake_root: str = "data/datalake"
    assets: list[str] = field(default_factory=lambda: ["btc", "eth", "sol", "xrp"])
    timeframe: str = "15m"

    # Simulation parameters
    initial_equity: float = 10000.0
    max_position: float = 100.0  # Max position size
    maker_fee: float = 0.0

    # Quote refresh interval (ms)
    quote_refresh_ms: int = 100  # Update quotes every 100ms

    # Size model parameters
    base_size: float = 10.0
    max_size: float = 50.0
    min_size: float = 5.0

    # Insight model parameters (override defaults)
    max_inventory_skew_bps: float = 50.0
    min_spread_bps: float = 50.0
    max_spread_bps: float = 5000.0


@dataclass
class MarketBacktestMetrics:
    """Metrics for a single market backtest."""

    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_fills: int = 0
    buy_fills: int = 0
    sell_fills: int = 0
    total_volume: float = 0.0
    max_position: float = 0.0
    min_position: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_spread_bps: float = 0.0
    tick_count: int = 0
    quotes_placed: int = 0


@dataclass
class MarketBacktestResult:
    """Result for a single market."""

    slug: str
    asset: str
    outcome: str  # "up" or "down"
    metrics: MarketBacktestMetrics
    fills: list[Fill]
    final_position: Position
    pnl_samples: list[float] = field(default_factory=list)


@dataclass
class AggregateBacktestResult:
    """Aggregate result across all markets."""

    config: InsightBacktestConfig
    market_results: dict[str, MarketBacktestResult]
    total_pnl: float = 0.0
    total_fills: int = 0
    total_volume: float = 0.0
    total_markets: int = 0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_pnl_per_market: float = 0.0

    # Per-asset breakdown
    asset_metrics: dict[str, dict] = field(default_factory=dict)


class InsightBacktestEngine:
    """
    Backtest engine for InsightQuoteModel using datalake data.

    This engine:
    1. Loads orderbook data from the datalake
    2. Computes quotes using InsightQuoteModel
    3. Simulates fills when trades cross our quotes
    4. Tracks PnL and metrics per market
    """

    def __init__(self, config: InsightBacktestConfig):
        self.config = config

        # Initialize datalake loader
        self.loader = DatalakeOrderbookLoader(
            datalake_root=config.datalake_root,
            cache_parsed_data=True,
        )

        # Create quote model
        insight_config = InsightQuoteConfig(
            max_inventory_skew_bps=config.max_inventory_skew_bps,
            min_spread_bps=config.min_spread_bps,
            max_spread_bps=config.max_spread_bps,
            max_position_pct=config.max_position / config.initial_equity,
        )
        self.quote_model = InsightQuoteModel(config=insight_config)

        # Create size model
        size_config = InventoryBasedSizeConfig(
            base_size=config.base_size,
            max_size=config.max_size,
            min_size=config.min_size,
            max_position=config.max_position,
            size_reduction_rate=0.5,
            asymmetric_scaling=True,
        )
        self.size_model = InventoryBasedSizeModel(config=size_config)

        # Market state
        self.simulator: MarketSimulator | None = None

    def run(self) -> AggregateBacktestResult:
        """Run backtest across all markets."""
        start_time = time.time()

        market_results: dict[str, MarketBacktestResult] = {}

        for asset in self.config.assets:
            print(f"\n{'='*60}")
            print(f"Processing {asset.upper()} markets...")
            print(f"{'='*60}")

            # Try loading resolved markets first
            resolved_markets = self.loader.load_resolved_markets(
                asset, self.config.timeframe
            )

            if resolved_markets:
                print(f"  Found {len(resolved_markets)} resolved markets")

                # Run backtest for each resolved market
                for i, market in enumerate(resolved_markets):
                    result = self._run_single_market(market)
                    if result and result.metrics.tick_count > 0:
                        market_results[market.slug] = result

                    if (i + 1) % 50 == 0:
                        print(f"  Progress: {i + 1}/{len(resolved_markets)} markets")
            else:
                # Fallback: use available markets from orderbook data
                print(f"  No resolved markets found, using available orderbook data...")
                available_markets = self.loader.list_available_markets(
                    asset, self.config.timeframe
                )

                if not available_markets:
                    print(f"  No orderbook data found for {asset}")
                    continue

                print(f"  Found {len(available_markets)} markets in orderbook data")

                for i, (slug, expiry_ts, expiry_dt) in enumerate(available_markets):
                    # Create a minimal market info for simulation
                    market_info = self.loader._parse_slug(slug)
                    if not market_info:
                        continue

                    result = self._run_market_from_slug(market_info, asset)
                    if result and result.metrics.tick_count > 0:
                        market_results[slug] = result

                    if (i + 1) % 50 == 0:
                        print(f"  Progress: {i + 1}/{len(available_markets)} markets")

        elapsed = time.time() - start_time
        print(f"\nBacktest completed in {elapsed:.2f} seconds")

        # Create aggregate result
        result = AggregateBacktestResult(
            config=self.config,
            market_results=market_results,
        )
        self._compute_aggregates(result)

        return result

    def _run_market_from_slug(self, market_info: MarketInfo, asset: str) -> MarketBacktestResult | None:
        """Run backtest for a market using just slug info (no resolution data)."""
        # Load orderbook mid prices for this market
        mid_prices = self.loader.get_mid_prices_for_market(
            asset, market_info.slug_ts, self.config.timeframe
        )

        if not mid_prices:
            return None

        # Filter to market period
        start_ts = market_info.start_ts * 1000  # Convert to ms
        expiry_ts = market_info.expiry_ts * 1000  # Convert to ms

        filtered_prices = [
            mp for mp in mid_prices
            if start_ts <= mp.timestamp_ms <= expiry_ts
        ]

        if len(filtered_prices) < 10:
            return None

        # Initialize simulator
        self.simulator = MarketSimulator(maker_fee=self.config.maker_fee)

        # Track metrics
        metrics = MarketBacktestMetrics()
        pnl_samples: list[float] = []
        peak_pnl = 0.0

        # Quote refresh tracking
        last_quote_ms = 0
        spread_sum = 0.0
        spread_count = 0

        # Process each tick
        for mp in filtered_prices:
            metrics.tick_count += 1

            # Calculate TTE in minutes
            tte_seconds = (expiry_ts - mp.timestamp_ms) / 1000
            tte_minutes = max(0, tte_seconds / 60)

            # Build strategy state
            state = self._build_state(
                mp, asset, tte_minutes, metrics.tick_count
            )

            # Update quotes at refresh interval
            if mp.timestamp_ms - last_quote_ms >= self.config.quote_refresh_ms:
                self._update_quotes(state, mp.timestamp_ms * 1_000_000)
                last_quote_ms = mp.timestamp_ms
                metrics.quotes_placed += 1

            # Simulate fills using mid price as "trade" price
            fills = self._check_fills(mp.mid_price, mp.timestamp_ms * 1_000_000)
            self._update_metrics(fills, metrics)

            # Track spread
            spread_bps = mp.spread * 10000
            spread_sum += spread_bps
            spread_count += 1

            # Track position extremes
            pos_size = self.simulator.position.size
            metrics.max_position = max(metrics.max_position, pos_size)
            metrics.min_position = min(metrics.min_position, pos_size)

            # Track PnL
            current_pnl = self.simulator.get_total_pnl(mp.mid_price)
            pnl_samples.append(current_pnl)
            peak_pnl = max(peak_pnl, current_pnl)
            drawdown = peak_pnl - current_pnl
            metrics.max_drawdown = max(metrics.max_drawdown, drawdown)

        # Final metrics
        if filtered_prices:
            final_mid = filtered_prices[-1].mid_price
            metrics.unrealized_pnl = self.simulator.get_unrealized_pnl(final_mid)
            metrics.realized_pnl = self.simulator.position.realized_pnl
            metrics.total_pnl = metrics.realized_pnl + metrics.unrealized_pnl

        metrics.avg_spread_bps = spread_sum / spread_count if spread_count > 0 else 0

        return MarketBacktestResult(
            slug=market_info.slug,
            asset=asset,
            outcome="unknown",  # No resolution data
            metrics=metrics,
            fills=self.simulator.fills.copy(),
            final_position=self.simulator.position,
            pnl_samples=pnl_samples,
        )

    def _run_single_market(self, market: ResolvedMarketInfo) -> MarketBacktestResult | None:
        """Run backtest for a single market."""
        # Parse expiry from slug
        market_info = self.loader._parse_slug(market.slug)
        if not market_info:
            return None

        # Load orderbook mid prices for this market
        # market_info.slug_ts is the market START time, expiry is slug_ts + period
        mid_prices = self.loader.get_mid_prices_for_market(
            market.asset, market_info.slug_ts, self.config.timeframe
        )

        if not mid_prices:
            return None

        # Filter to market period (start_ts to expiry_ts)
        start_ts = market_info.start_ts * 1000  # Convert to ms
        expiry_ts = market_info.expiry_ts * 1000  # Convert to ms

        filtered_prices = [
            mp for mp in mid_prices
            if start_ts <= mp.timestamp_ms <= expiry_ts
        ]

        if len(filtered_prices) < 10:
            return None

        # Initialize simulator
        self.simulator = MarketSimulator(maker_fee=self.config.maker_fee)

        # Track metrics
        metrics = MarketBacktestMetrics()
        pnl_samples: list[float] = []
        peak_pnl = 0.0

        # Quote refresh tracking
        last_quote_ms = 0
        spread_sum = 0.0
        spread_count = 0

        # Process each tick
        for mp in filtered_prices:
            metrics.tick_count += 1

            # Calculate TTE in minutes
            tte_seconds = (expiry_ts - mp.timestamp_ms) / 1000
            tte_minutes = max(0, tte_seconds / 60)

            # Build strategy state
            state = self._build_state(
                mp, market.asset, tte_minutes, metrics.tick_count
            )

            # Update quotes at refresh interval
            if mp.timestamp_ms - last_quote_ms >= self.config.quote_refresh_ms:
                self._update_quotes(state, mp.timestamp_ms * 1_000_000)  # Convert to ns
                last_quote_ms = mp.timestamp_ms
                metrics.quotes_placed += 1

            # Simulate fills using mid price as "trade" price
            # Fill logic: if mid price crosses our quote
            fills = self._check_fills(mp.mid_price, mp.timestamp_ms * 1_000_000)
            self._update_metrics(fills, metrics)

            # Track spread
            spread_bps = mp.spread * 10000
            spread_sum += spread_bps
            spread_count += 1

            # Track position extremes
            pos_size = self.simulator.position.size
            metrics.max_position = max(metrics.max_position, pos_size)
            metrics.min_position = min(metrics.min_position, pos_size)

            # Track PnL
            current_pnl = self.simulator.get_total_pnl(mp.mid_price)
            pnl_samples.append(current_pnl)
            peak_pnl = max(peak_pnl, current_pnl)
            drawdown = peak_pnl - current_pnl
            metrics.max_drawdown = max(metrics.max_drawdown, drawdown)

        # Final metrics
        if filtered_prices:
            final_mid = filtered_prices[-1].mid_price
            metrics.unrealized_pnl = self.simulator.get_unrealized_pnl(final_mid)
            metrics.realized_pnl = self.simulator.position.realized_pnl
            metrics.total_pnl = metrics.realized_pnl + metrics.unrealized_pnl

        metrics.avg_spread_bps = spread_sum / spread_count if spread_count > 0 else 0

        # Calculate win rate from fills
        if metrics.total_fills > 0:
            wins = sum(1 for f in self.simulator.fills if self._is_winning_fill(f, market.outcome))
            metrics.win_rate = wins / metrics.total_fills

        return MarketBacktestResult(
            slug=market.slug,
            asset=market.asset,
            outcome=market.outcome,
            metrics=metrics,
            fills=self.simulator.fills.copy(),
            final_position=self.simulator.position,
            pnl_samples=pnl_samples,
        )

    def _build_state(
        self,
        mp: OrderbookMidPrice,
        asset: str,
        tte_minutes: float,
        timestamp_ns: int,
    ) -> StrategyState:
        """Build strategy state from orderbook mid price."""
        # Create minimal bid/ask from spread
        half_spread = mp.spread / 2
        bids = [(mp.best_bid, 1000.0)]  # Dummy size
        asks = [(mp.best_ask, 1000.0)]  # Dummy size

        # Create a MarketState with TTE info
        market = MarketState(
            asset_id=f"{asset}-updown-{self.config.timeframe}",
            timestamp_ns=timestamp_ns,
            mid_price=mp.mid_price,
            spread=mp.spread,
            bids=bids,
            asks=asks,
            last_trade_price=mp.mid_price,
            last_trade_size=0.0,
            last_trade_side=0,
            tte_minutes=tte_minutes,  # Pass TTE directly to MarketState
        )

        # Position state
        pos = self.simulator.position
        position = PositionState(
            asset_id=market.asset_id,
            position=pos.size,
            avg_entry_price=pos.avg_entry_price,
            unrealized_pnl=self.simulator.get_unrealized_pnl(mp.mid_price),
            realized_pnl=pos.realized_pnl,
        )

        # Calculate equity
        total_pnl = self.simulator.get_total_pnl(mp.mid_price)
        total_equity = self.config.initial_equity + total_pnl

        return StrategyState(
            market=market,
            position=position,
            total_equity=total_equity,
        )

    def _update_quotes(self, state: StrategyState, timestamp_ns: int) -> None:
        """Update quotes based on current state."""
        # Compute quote
        quote_result = self.quote_model.compute(state)

        if not quote_result.should_quote:
            return

        # Compute size
        size_result = self.size_model.compute(state, quote_result)

        # Cancel existing orders
        self.simulator.cancel_all()

        # Place new orders
        current_pos = self.simulator.position.size

        # Buy order
        if quote_result.bid_price > 0 and size_result.bid_size > 0:
            if current_pos + size_result.bid_size <= self.config.max_position:
                self.simulator.place_limit_order(
                    side=SIDE_BUY,
                    price=quote_result.bid_price,
                    size=size_result.bid_size,
                    timestamp_ns=timestamp_ns,
                )

        # Sell order
        if quote_result.ask_price > 0 and size_result.ask_size > 0:
            if current_pos - size_result.ask_size >= -self.config.max_position:
                self.simulator.place_limit_order(
                    side=SIDE_SELL,
                    price=quote_result.ask_price,
                    size=size_result.ask_size,
                    timestamp_ns=timestamp_ns,
                )

    def _check_fills(self, trade_price: float, timestamp_ns: int) -> list[Fill]:
        """Check for fills using trade price crossing logic."""
        tick_fills: list[Fill] = []
        orders_to_remove: list[int] = []

        for order_id, order in self.simulator._orders.items():
            # Buy fills if trade_price < bid_price (we buy when market sells low)
            # Sell fills if trade_price > ask_price (we sell when market buys high)
            should_fill = (
                (order.side == SIDE_BUY and trade_price < order.price) or
                (order.side == SIDE_SELL and trade_price > order.price)
            )

            if should_fill:
                fill_size = order.remaining_size
                order.filled_size += fill_size

                fill = Fill(
                    order_id=order.order_id,
                    side=order.side,
                    price=order.price,
                    size=fill_size,
                    timestamp_ns=timestamp_ns,
                    is_maker=True,
                )
                tick_fills.append(fill)
                self.simulator._fills.append(fill)
                self.simulator._update_position(fill)

                if order.is_filled:
                    orders_to_remove.append(order_id)

        for order_id in orders_to_remove:
            del self.simulator._orders[order_id]

        return tick_fills

    def _update_metrics(self, fills: list[Fill], metrics: MarketBacktestMetrics) -> None:
        """Update metrics based on fills."""
        for fill in fills:
            metrics.total_fills += 1
            metrics.total_volume += fill.size

            if fill.side == SIDE_BUY:
                metrics.buy_fills += 1
            else:
                metrics.sell_fills += 1

    def _is_winning_fill(self, fill: Fill, outcome: str) -> bool:
        """Determine if a fill was profitable given the market outcome."""
        # For "up" markets: YES token wins if price ends at 1.0
        # For "down" markets: NO token wins if price ends at 1.0
        # This is simplified - actual win depends on entry/exit prices
        return True  # Simplified for now

    def _compute_aggregates(self, result: AggregateBacktestResult) -> None:
        """Compute aggregate metrics from market results."""
        result.total_markets = len(result.market_results)

        if result.total_markets == 0:
            return

        result.total_pnl = sum(r.metrics.total_pnl for r in result.market_results.values())
        result.total_fills = sum(r.metrics.total_fills for r in result.market_results.values())
        result.total_volume = sum(r.metrics.total_volume for r in result.market_results.values())
        result.avg_pnl_per_market = result.total_pnl / result.total_markets

        # Per-asset breakdown
        for asset in self.config.assets:
            asset_results = [r for r in result.market_results.values() if r.asset == asset]
            if asset_results:
                result.asset_metrics[asset] = {
                    "markets": len(asset_results),
                    "total_pnl": sum(r.metrics.total_pnl for r in asset_results),
                    "total_fills": sum(r.metrics.total_fills for r in asset_results),
                    "total_volume": sum(r.metrics.total_volume for r in asset_results),
                    "avg_pnl": sum(r.metrics.total_pnl for r in asset_results) / len(asset_results),
                    "avg_spread_bps": np.mean([r.metrics.avg_spread_bps for r in asset_results]),
                }

        # Compute Sharpe ratio from PnL samples
        all_samples: list[float] = []
        for r in result.market_results.values():
            all_samples.extend(r.pnl_samples)

        if len(all_samples) > 1:
            returns = np.diff(all_samples)
            if len(returns) > 0 and np.std(returns) > 0:
                # Annualize assuming ~100ms samples, 8 hours per day
                samples_per_year = 8 * 3600 * 10 * 252  # 8hr * 3600s * 10 samples/s * 252 days
                result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(samples_per_year)

        # Max drawdown
        result.max_drawdown = max(
            (r.metrics.max_drawdown for r in result.market_results.values()),
            default=0.0
        )

        # Win rate
        total_fills = sum(r.metrics.total_fills for r in result.market_results.values())
        winning_markets = sum(1 for r in result.market_results.values() if r.metrics.total_pnl > 0)
        result.win_rate = winning_markets / result.total_markets if result.total_markets > 0 else 0


def print_results(result: AggregateBacktestResult) -> None:
    """Print backtest results summary."""
    print("\n" + "=" * 80)
    print("INSIGHT QUOTE MODEL BACKTEST RESULTS")
    print("=" * 80)

    print("\nConfiguration:")
    print(f"  Assets: {result.config.assets}")
    print(f"  Timeframe: {result.config.timeframe}")
    print(f"  Initial Equity: ${result.config.initial_equity:,.2f}")
    print(f"  Max Position: {result.config.max_position}")

    print("\nAggregate Metrics:")
    print(f"  Total Markets: {result.total_markets}")
    print(f"  Total PnL: ${result.total_pnl:,.2f}")
    print(f"  Avg PnL/Market: ${result.avg_pnl_per_market:,.2f}")
    print(f"  Total Fills: {result.total_fills:,}")
    print(f"  Total Volume: {result.total_volume:,.2f}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: ${result.max_drawdown:,.2f}")
    print(f"  Win Rate: {result.win_rate:.1%}")

    print("\nPer-Asset Breakdown:")
    print(f"{'Asset':<8} {'Markets':>8} {'PnL':>12} {'Fills':>8} {'Volume':>12} {'Avg PnL':>10} {'Spread':>10}")
    print("-" * 80)

    for asset in sorted(result.asset_metrics.keys()):
        m = result.asset_metrics[asset]
        print(
            f"{asset.upper():<8} {m['markets']:>8} "
            f"${m['total_pnl']:>10,.2f} {m['total_fills']:>8,} "
            f"{m['total_volume']:>12,.0f} ${m['avg_pnl']:>8,.2f} "
            f"{m['avg_spread_bps']:>8.1f}bp"
        )

    print("=" * 80)


def save_results(result: AggregateBacktestResult, output_path: str) -> None:
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "assets": result.config.assets,
            "timeframe": result.config.timeframe,
            "initial_equity": result.config.initial_equity,
            "max_position": result.config.max_position,
        },
        "aggregate": {
            "total_markets": result.total_markets,
            "total_pnl": result.total_pnl,
            "avg_pnl_per_market": result.avg_pnl_per_market,
            "total_fills": result.total_fills,
            "total_volume": result.total_volume,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
        },
        "per_asset": result.asset_metrics,
        "markets": {
            slug: {
                "asset": r.asset,
                "outcome": r.outcome,
                "pnl": r.metrics.total_pnl,
                "fills": r.metrics.total_fills,
                "volume": r.metrics.total_volume,
                "max_dd": r.metrics.max_drawdown,
                "avg_spread_bps": r.metrics.avg_spread_bps,
            }
            for slug, r in result.market_results.items()
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Run Insight quote model backtest."""
    print("Insight-Driven Quote Model Backtest")
    print("=" * 60)

    config = InsightBacktestConfig(
        datalake_root="../data/datalake",
        assets=["btc", "eth", "sol", "xrp"],
        timeframe="15m",
        initial_equity=10000.0,
        max_position=100.0,
        maker_fee=0.0,
        quote_refresh_ms=100,
        base_size=10.0,
        max_size=50.0,
        min_size=5.0,
        max_inventory_skew_bps=50.0,
        min_spread_bps=50.0,
        max_spread_bps=5000.0,
    )

    engine = InsightBacktestEngine(config)
    result = engine.run()

    print_results(result)

    # Save results
    output_path = "data/backtest_insight_results.json"
    save_results(result, output_path)

    return result


if __name__ == "__main__":
    main()
