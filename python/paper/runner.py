"""
Paper Trading Runner

Main entry point for the paper trading Python logic layer.
Coordinates strategy, execution simulation, position tracking, and SHM updates.
"""

import argparse
import asyncio
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog
import yaml

from paper.executor import PaperExecutor
from paper.metrics import MetricsCalculator
from paper.position_tracker import PositionTracker
from paper.shm_paper import PaperSHMWriter
from paper.supabase_store import SupabaseConfig, SupabaseStore
from strategy.shm.reader import SHMReader
from strategy.shm.types import SIDE_BUY, SIDE_SELL, MarketState

logger = structlog.get_logger()


@dataclass
class PaperConfig:
    """Paper trading configuration."""

    initial_equity: float = 10000.0
    tick_interval_ms: int = 100
    equity_snapshot_interval_s: int = 60
    metrics_sync_interval_s: int = 60
    position_sync_interval_s: int = 10
    market_snapshot_interval_s: int = 5
    use_shm: bool = True  # Write to SHM for Go persistence
    use_direct_supabase: bool = False  # Direct Python->Supabase (alternative)
    supabase_url: str = ""
    supabase_key: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> "PaperConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        paper_data = data.get("paper", {})
        return cls(
            initial_equity=paper_data.get("initial_equity", 10000.0),
            tick_interval_ms=paper_data.get("tick_interval_ms", 100),
            equity_snapshot_interval_s=paper_data.get("equity_snapshot_interval_s", 60),
            metrics_sync_interval_s=paper_data.get("metrics_sync_interval_s", 60),
            position_sync_interval_s=paper_data.get("position_sync_interval_s", 10),
            market_snapshot_interval_s=paper_data.get("market_snapshot_interval_s", 5),
            use_shm=paper_data.get("use_shm", True),
            use_direct_supabase=paper_data.get("use_direct_supabase", False),
            supabase_url=data.get("supabase", {}).get("url", os.environ.get("SUPABASE_URL", "")),
            supabase_key=data.get("supabase", {}).get("api_key", os.environ.get("SUPABASE_KEY", "")),
        )


class PaperTradingRunner:
    """Main paper trading runner."""

    def __init__(self, config: PaperConfig):
        self.config = config
        self._logger = logger.bind(component="paper_runner")
        self._running = False

        # Core components
        self._shm_reader: Optional[SHMReader] = None
        self._paper_shm: Optional[PaperSHMWriter] = None
        self._supabase_store: Optional[SupabaseStore] = None
        self._position_tracker: Optional[PositionTracker] = None
        self._paper_executor: Optional[PaperExecutor] = None
        self._metrics: Optional[MetricsCalculator] = None

        # Tracking
        self._last_equity_snapshot = 0.0
        self._last_metrics_sync = 0.0
        self._last_position_sync = 0.0
        self._last_market_snapshot = 0.0

    def setup(self) -> None:
        """Initialize all components."""
        self._logger.info("Setting up paper trading runner")

        # Connect to main SHM (Go aggregator writes market data)
        self._shm_reader = SHMReader()
        try:
            self._shm_reader.connect()
            self._logger.info("Connected to main SHM")
        except FileNotFoundError:
            self._logger.warning("Main SHM not found - will retry")

        # Initialize position tracker
        self._position_tracker = PositionTracker(initial_equity=self.config.initial_equity)

        # Initialize metrics calculator
        self._metrics = MetricsCalculator(initial_equity=self.config.initial_equity)

        # Initialize Supabase store (for executor)
        if self.config.use_direct_supabase and self.config.supabase_url:
            self._supabase_store = SupabaseStore(
                SupabaseConfig(
                    url=self.config.supabase_url,
                    api_key=self.config.supabase_key,
                )
            )
            if not self._supabase_store.health_check():
                self._logger.error("Supabase health check failed")
                self._supabase_store = None
        else:
            # Create a dummy store that just logs
            self._supabase_store = _DummyStore()

        # Initialize paper executor
        self._paper_executor = PaperExecutor(
            position_tracker=self._position_tracker,
            supabase_store=self._supabase_store,
        )

        # Initialize paper trading SHM (for Go to read)
        if self.config.use_shm:
            self._paper_shm = PaperSHMWriter()
            try:
                self._paper_shm.connect()
                self._logger.info("Connected to paper trading SHM")
            except Exception as e:
                self._logger.error("Failed to connect to paper SHM", error=str(e))
                self._paper_shm = None

    def teardown(self) -> None:
        """Cleanup all components."""
        self._logger.info("Tearing down paper trading runner")

        if self._paper_shm:
            self._paper_shm.close()

        if self._shm_reader:
            self._shm_reader.close()

        if self._supabase_store and hasattr(self._supabase_store, "close"):
            self._supabase_store.close()

    async def run_async(self) -> None:
        """Run the paper trading loop asynchronously."""
        self._running = True
        tick_interval = self.config.tick_interval_ms / 1000.0

        self._logger.info("Starting paper trading loop", tick_interval_ms=self.config.tick_interval_ms)

        while self._running:
            try:
                await self._tick()
            except Exception as e:
                self._logger.error("Error in tick", error=str(e))

            await asyncio.sleep(tick_interval)

    async def _tick(self) -> None:
        """Single tick of the paper trading loop."""
        now = time.time()

        # Try to connect to SHM if not connected
        if self._shm_reader is None or self._shm_reader._mm is None:
            try:
                self._shm_reader = SHMReader()
                self._shm_reader.connect()
            except FileNotFoundError:
                return

        # Refresh SHM state
        self._shm_reader.refresh()

        # Get current markets
        markets = self._shm_reader.get_markets_dict()
        if not markets:
            return

        # Check open orders against current market
        for asset_id, market in markets.items():
            fills = self._paper_executor.check_open_orders(market)
            for fill in fills:
                self._metrics.record_trade(fill.pnl)

        # Update unrealized PnL for all positions
        for asset_id, position in self._position_tracker.positions.items():
            if position.size > 0:
                market = markets.get(asset_id)
                if market and market.bids:
                    mid_price = (market.bids[0][0] + market.asks[0][0]) / 2 if market.asks else market.bids[0][0]
                    self._position_tracker.update_unrealized_pnl(asset_id, mid_price)

        # Sync to paper SHM
        if self._paper_shm:
            self._sync_to_shm(markets)

        # Periodic tasks
        if now - self._last_equity_snapshot > self.config.equity_snapshot_interval_s:
            self._snapshot_equity()
            self._last_equity_snapshot = now

        if now - self._last_metrics_sync > self.config.metrics_sync_interval_s:
            self._sync_metrics()
            self._last_metrics_sync = now

        if now - self._last_position_sync > self.config.position_sync_interval_s:
            self._sync_positions()
            self._last_position_sync = now

        if now - self._last_market_snapshot > self.config.market_snapshot_interval_s:
            self._snapshot_markets(markets)
            self._last_market_snapshot = now

    def _sync_to_shm(self, markets: dict[str, MarketState]) -> None:
        """Sync current state to paper trading SHM."""
        if not self._paper_shm or not self._position_tracker:
            return

        # Update positions
        for asset_id, position in self._position_tracker.positions.items():
            self._paper_shm.update_position(
                asset_id=position.asset_id,
                slug=position.slug,
                side=position.side,
                size=position.size,
                avg_entry_price=position.avg_entry_price,
                unrealized_pnl=position.unrealized_pnl,
                realized_pnl=position.realized_pnl,
            )

        # Update equity
        equity = self._position_tracker.total_equity
        cash = self._position_tracker.cash
        position_value = equity - cash
        self._paper_shm.update_equity(equity, cash, position_value)

        # Update metrics
        if self._metrics:
            state = self._metrics.get_state()
            self._paper_shm.update_metrics(
                total_pnl=state.total_pnl,
                realized_pnl=state.realized_pnl,
                unrealized_pnl=state.unrealized_pnl,
                total_trades=state.total_trades,
                win_count=state.win_count,
                sharpe_ratio=state.sharpe_ratio,
                max_drawdown=state.max_drawdown,
            )

        # Update quotes (our current positions as context)
        for asset_id, market in markets.items():
            position = self._position_tracker.positions.get(asset_id)
            inventory = position.size if position else 0.0

            if market.bids and market.asks:
                best_bid = market.bids[0][0]
                best_ask = market.asks[0][0]
                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid

                # Get slug from position or derive from asset_id
                slug = position.slug if position else asset_id[:32]

                self._paper_shm.update_quote(
                    slug=slug,
                    our_bid=0.0,  # We don't have active quotes in paper trading
                    our_ask=0.0,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    mid_price=mid_price,
                    spread=spread,
                    inventory=inventory,
                )

        self._paper_shm.flush()

    def _snapshot_equity(self) -> None:
        """Take equity snapshot."""
        if not self._position_tracker:
            return

        equity = self._position_tracker.total_equity
        cash = self._position_tracker.cash
        position_value = equity - cash

        self._position_tracker.snapshot_equity()

        if self._metrics:
            self._metrics.update_equity(
                equity=equity,
                realized_pnl=self._position_tracker.realized_pnl,
                unrealized_pnl=self._position_tracker.unrealized_pnl,
            )

        # Direct Supabase write if configured
        if self.config.use_direct_supabase and self._supabase_store:
            self._supabase_store.insert_equity_snapshot(equity, cash, position_value)

        self._logger.debug("Equity snapshot", equity=equity, cash=cash)

    def _sync_metrics(self) -> None:
        """Sync metrics to Supabase."""
        if not self._metrics:
            return

        state = self._metrics.get_state()

        # Direct Supabase write if configured
        if self.config.use_direct_supabase and self._supabase_store:
            self._supabase_store.upsert_metrics(
                total_pnl=state.total_pnl,
                realized_pnl=state.realized_pnl,
                unrealized_pnl=state.unrealized_pnl,
                total_trades=state.total_trades,
                win_rate=state.win_rate,
                sharpe_ratio=state.sharpe_ratio,
                max_drawdown=state.max_drawdown,
            )

    def _sync_positions(self) -> None:
        """Sync positions to Supabase."""
        if not self._position_tracker:
            return

        # Direct Supabase write if configured
        if self.config.use_direct_supabase and self._supabase_store:
            for position in self._position_tracker.get_all_positions():
                self._supabase_store.upsert_position(position)

    def _snapshot_markets(self, markets: dict[str, MarketState]) -> None:
        """Snapshot current market state."""
        if not self.config.use_direct_supabase or not self._supabase_store:
            return

        for asset_id, market in markets.items():
            if not market.bids or not market.asks:
                continue

            position = self._position_tracker.positions.get(asset_id) if self._position_tracker else None
            inventory = position.size if position else 0.0
            slug = position.slug if position else asset_id[:32]

            best_bid = market.bids[0][0]
            best_ask = market.asks[0][0]
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            self._supabase_store.insert_market_snapshot(
                slug=slug,
                best_bid=best_bid,
                best_ask=best_ask,
                mid_price=mid_price,
                spread=spread,
                inventory=inventory,
                inventory_value=inventory * mid_price if inventory else 0.0,
            )

    def stop(self) -> None:
        """Stop the runner."""
        self._running = False

    # Public API for placing orders (called by strategy)
    def place_order(
        self,
        asset_id: str,
        slug: str,
        side: int,
        price: float,
        size: float,
    ) -> Optional["Fill"]:
        """Place a paper order."""
        if not self._paper_executor or not self._shm_reader:
            return None

        # Get current market
        market = self._shm_reader.get_market(asset_id)
        if not market:
            return None

        # Create and process signal
        signal = self._paper_executor.create_signal(
            asset_id=asset_id,
            slug=slug,
            side=side,
            price=price,
            size=size,
        )

        fill = self._paper_executor.process_signal(signal, market)

        if fill:
            self._metrics.record_trade(fill.pnl)
            # Write trade to SHM
            if self._paper_shm:
                self._paper_shm.add_trade(
                    asset_id=fill.asset_id,
                    slug=fill.slug,
                    side=fill.side,
                    price=fill.price,
                    size=fill.size,
                    pnl=fill.pnl,
                )

        return fill


class _DummyStore:
    """Dummy store that does nothing (used when Supabase is disabled)."""

    def insert_trade(self, trade: dict) -> None:
        pass

    def upsert_position(self, position: dict) -> None:
        pass


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Paper Trading Runner")
    parser.add_argument("--config", default="data/config/paper.yaml", help="Config file path")
    parser.add_argument("--mock", action="store_true", help="Run with mock data")
    args = parser.parse_args()

    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = PaperConfig.from_yaml(str(config_path))
    else:
        logger.warning("Config file not found, using defaults", path=args.config)
        config = PaperConfig()

    # Create and run
    runner = PaperTradingRunner(config)

    # Handle shutdown signals
    def handle_signal(sig: int, frame: object) -> None:
        logger.info("Received shutdown signal")
        runner.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        runner.setup()
        asyncio.run(runner.run_async())
    finally:
        runner.teardown()


if __name__ == "__main__":
    main()
