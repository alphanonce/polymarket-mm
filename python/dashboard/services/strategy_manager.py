"""
Strategy Manager

Manages multiple strategy worker processes.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import time
from multiprocessing import Queue
from typing import Any

import structlog

from dashboard.config import DashboardConfig, StrategyConfig
from dashboard.models.schemas import StrategyCard, StrategySummary
from dashboard.services.broadcast import BroadcastHub
from dashboard.services.market_discovery import MarketDiscoveryService, MarketInfo
from dashboard.services.strategy_worker import StrategyWorker
from paper.supabase_store import SupabaseConfig

logger = structlog.get_logger()


def _run_worker_process(
    config: StrategyConfig,
    update_queue: Queue[dict[str, Any]],
    supabase_config: SupabaseConfig | None,
) -> None:
    """Entry point for worker subprocess."""
    import asyncio

    worker = StrategyWorker(
        config=config,
        update_queue=update_queue,
        supabase_config=supabase_config,
    )

    worker.setup()
    try:
        asyncio.run(worker.run())
    finally:
        worker.teardown()


class ManagedStrategy:
    """Container for a managed strategy and its process."""

    def __init__(
        self,
        config: StrategyConfig,
        update_queue: Queue[dict[str, Any]],
    ) -> None:
        self.config = config
        self.strategy_id = config.strategy_id
        self.update_queue = update_queue
        self.process: mp.Process | None = None
        self.worker: StrategyWorker | None = None  # For in-process mode

        # State tracking
        self.status = "stopped"
        self.last_state: dict[str, Any] = {}
        self.last_update_time = 0.0
        self.created_at = time.time()


class StrategyManager:
    """
    Manages multiple strategy workers.

    Can run workers as subprocesses (production) or in-process async tasks (testing).
    """

    def __init__(
        self,
        config: DashboardConfig,
        broadcast_hub: BroadcastHub,
        market_discovery: MarketDiscoveryService | None = None,
        use_processes: bool = False,
    ) -> None:
        self._config = config
        self._broadcast_hub = broadcast_hub
        self._market_discovery = market_discovery
        self._use_processes = use_processes

        self._strategies: dict[str, ManagedStrategy] = {}
        self._update_queue: Queue[dict[str, Any]] = mp.Queue()
        self._running = False
        self._consumer_task: asyncio.Task[None] | None = None

        self._logger = logger.bind(component="strategy_manager")

        # Supabase config
        self._supabase_config: SupabaseConfig | None = None
        if config.supabase.url:
            self._supabase_config = SupabaseConfig(
                url=config.supabase.url,
                api_key=config.supabase.api_key,
            )

    async def start(self) -> None:
        """Start the manager and load default strategies."""
        self._running = True

        # Start consuming updates from workers
        self._consumer_task = asyncio.create_task(self._consume_updates())

        # Load default strategies
        for strategy_config in self._config.default_strategies:
            await self.create_strategy(strategy_config)

        self._logger.info(
            "Strategy manager started",
            default_strategies=len(self._config.default_strategies),
        )

    async def stop(self) -> None:
        """Stop all strategies and cleanup."""
        self._running = False

        # Stop all strategies
        for strategy_id in list(self._strategies.keys()):
            await self.stop_strategy(strategy_id)

        # Cancel consumer task
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        self._logger.info("Strategy manager stopped")

    async def create_strategy(
        self,
        config: StrategyConfig,
        start: bool = True,
    ) -> str:
        """Create a new strategy. Returns strategy_id."""
        if len(self._strategies) >= self._config.max_concurrent_strategies:
            raise ValueError("Maximum concurrent strategies reached")

        strategy = ManagedStrategy(
            config=config,
            update_queue=self._update_queue,
        )
        self._strategies[config.strategy_id] = strategy

        self._logger.info(
            "Strategy created",
            strategy_id=config.strategy_id,
            name=config.name,
        )

        if start:
            await self.start_strategy(config.strategy_id)

        return config.strategy_id

    async def start_strategy(self, strategy_id: str) -> None:
        """Start a strategy."""
        strategy = self._strategies.get(strategy_id)
        if not strategy:
            raise KeyError(f"Strategy {strategy_id} not found")

        if strategy.status == "running":
            return

        if self._use_processes:
            # Run as subprocess
            strategy.process = mp.Process(
                target=_run_worker_process,
                args=(
                    strategy.config,
                    self._update_queue,
                    self._supabase_config,
                ),
                daemon=True,
            )
            strategy.process.start()
        else:
            # Run as async task in same process
            strategy.worker = StrategyWorker(
                config=strategy.config,
                update_queue=self._update_queue,
                supabase_config=self._supabase_config,
            )
            strategy.worker.setup()

            # Subscribe to market discovery
            if self._market_discovery:
                self._market_discovery.subscribe(strategy.worker.on_new_market)

                # Subscribe to existing active markets
                for market in self._market_discovery.get_active_markets():
                    await strategy.worker.on_new_market(market)

            # Start worker loop as background task
            asyncio.create_task(strategy.worker.run())

        strategy.status = "running"
        self._logger.info("Strategy started", strategy_id=strategy_id)

    async def stop_strategy(self, strategy_id: str) -> None:
        """Stop a strategy."""
        strategy = self._strategies.get(strategy_id)
        if not strategy:
            raise KeyError(f"Strategy {strategy_id} not found")

        if strategy.status == "stopped":
            return

        if self._use_processes and strategy.process:
            strategy.process.terminate()
            strategy.process.join(timeout=5.0)
            if strategy.process.is_alive():
                strategy.process.kill()
            strategy.process = None
        elif strategy.worker:
            strategy.worker.stop()
            strategy.worker.teardown()
            strategy.worker = None

        strategy.status = "stopped"
        self._logger.info("Strategy stopped", strategy_id=strategy_id)

    async def delete_strategy(self, strategy_id: str) -> None:
        """Delete a strategy."""
        await self.stop_strategy(strategy_id)
        del self._strategies[strategy_id]
        self._logger.info("Strategy deleted", strategy_id=strategy_id)

    def get_strategy(self, strategy_id: str) -> ManagedStrategy | None:
        """Get a managed strategy by ID."""
        return self._strategies.get(strategy_id)

    def get_all_strategies(self) -> list[ManagedStrategy]:
        """Get all managed strategies."""
        return list(self._strategies.values())

    def get_strategy_cards(self) -> list[StrategyCard]:
        """Get summary cards for all strategies."""
        cards = []
        for strategy in self._strategies.values():
            state = strategy.last_state

            pnl = state.get("total_pnl", 0.0)
            starting = strategy.config.starting_capital
            pnl_percent = (pnl / starting * 100) if starting > 0 else 0.0

            cards.append(
                StrategyCard(
                    strategy_id=strategy.strategy_id,
                    name=strategy.config.name,
                    assets=strategy.config.assets,
                    timeframe=strategy.config.timeframe,
                    total_pnl=pnl,
                    pnl_percent=pnl_percent,
                    status=strategy.status,
                    active_markets=len(state.get("quotes", [])),
                    position_count=len(state.get("positions", [])),
                    pnl_by_asset=state.get("pnl_by_asset", {}),
                )
            )

        return cards

    def get_strategy_summary(self, strategy_id: str) -> StrategySummary | None:
        """Get detailed summary for a strategy."""
        strategy = self._strategies.get(strategy_id)
        if not strategy:
            return None

        state = strategy.last_state

        return StrategySummary(
            strategy_id=strategy.strategy_id,
            name=strategy.config.name,
            config=strategy.config.to_dict(),
            status=strategy.status,
            starting_capital=strategy.config.starting_capital,
            current_equity=state.get("current_equity", strategy.config.starting_capital),
            total_pnl=state.get("total_pnl", 0.0),
            total_trades=state.get("total_trades", 0),
        )

    async def _consume_updates(self) -> None:
        """Consume updates from worker processes and broadcast to WebSocket clients."""
        self._logger.info("Starting update consumer")

        while self._running:
            try:
                # Non-blocking check with timeout
                try:
                    message = self._update_queue.get(timeout=0.1)
                except Exception:
                    await asyncio.sleep(0.01)
                    continue

                strategy_id: str | None = message.get("strategy_id")
                msg_type = message.get("type")
                data = message.get("data", {})

                if not strategy_id:
                    continue

                # Update last state
                strategy = self._strategies.get(strategy_id)
                if strategy:
                    strategy.last_state = data
                    strategy.last_update_time = time.time()

                # Broadcast to WebSocket clients
                if msg_type == "state":
                    await self._broadcast_hub.broadcast_state(strategy_id, data)
                elif msg_type == "fill":
                    await self._broadcast_hub.broadcast_fill(strategy_id, data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Error consuming update", error=str(e))

    async def on_new_market(self, market: MarketInfo) -> None:
        """Handle new market discovery - propagate to all strategies."""
        for strategy in self._strategies.values():
            if strategy.worker:
                await strategy.worker.on_new_market(market)
