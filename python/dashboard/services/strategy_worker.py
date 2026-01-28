"""
Strategy Worker

Isolated worker that runs a single strategy in a subprocess.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import Any

import structlog

from dashboard.config import StrategyConfig
from dashboard.models.schemas import (
    DashboardState,
    FillInfo,
    LevelInfo,
    MarketQuote,
    PositionInfo,
    QuoteHistoryPoint,
)
from dashboard.services.market_discovery import MarketInfo
from dashboard.services.model_factory import (
    create_quote_model,
    create_size_model,
)
from paper.executor import Fill, PaperExecutor
from paper.metrics import MetricsCalculator
from paper.position_tracker import PositionTracker
from paper.supabase_store import SupabaseConfig, SupabaseStore
from strategy.models.base import QuoteModel, QuoteResult, SizeModel, StrategyState
from strategy.shm.reader import SHMReader
from strategy.shm.types import SIDE_BUY, SIDE_SELL, MarketState, PositionState

logger = structlog.get_logger()


@dataclass
class WorkerState:
    """Internal state tracking for the worker."""

    running: bool = True
    tick_count: int = 0
    last_broadcast_time: float = 0.0
    last_equity_snapshot: float = 0.0
    inventory_history: list[tuple[int, float]] = field(default_factory=list)
    quote_history: list[tuple[int, str, float | None, float | None, float, float, float]] = field(
        default_factory=list
    )


class _DummyStore:
    """Dummy store when Supabase is disabled."""

    def insert_trade(self, trade: dict[str, object]) -> None:
        pass

    def upsert_position(self, position: dict[str, object]) -> None:
        pass


class StrategyWorker:
    """
    Worker that runs a single strategy in isolation.

    Designed to run in a subprocess and communicate via queues.
    """

    def __init__(
        self,
        config: StrategyConfig,
        update_queue: Queue[dict[str, Any]] | None = None,
        supabase_config: SupabaseConfig | None = None,
    ) -> None:
        self.config = config
        self.strategy_id = config.strategy_id
        self._update_queue = update_queue
        self._supabase_config = supabase_config

        self._logger = logger.bind(
            component="strategy_worker",
            strategy_id=self.strategy_id,
            strategy_name=config.name,
        )

        # State
        self._state = WorkerState()

        # Active markets for this strategy
        self._active_markets: dict[str, MarketInfo] = {}

        # Core components (initialized in setup)
        self._shm_reader: SHMReader | None = None
        self._position_tracker: PositionTracker | None = None
        self._paper_executor: PaperExecutor | None = None
        self._metrics: MetricsCalculator | None = None
        self._supabase_store: SupabaseStore | _DummyStore | None = None

        # Quote and size models (initialized in setup)
        self._quote_model: QuoteModel | None = None
        self._size_model: SizeModel | None = None

        # Per-market quote models (for models that need market-specific config like end_ts)
        self._market_quote_models: dict[str, QuoteModel] = {}

        # Configuration
        self._tick_interval_ms = 100
        self._broadcast_interval_ms = 200
        self._equity_snapshot_interval_s = 60
        self._quote_interval_ms = 500  # Generate quotes every 500ms
        self._last_quote_time: float = 0.0

        # Store last quote results for display
        self._last_quote_results: dict[str, QuoteResult] = {}

    def setup(self) -> None:
        """Initialize all components."""
        self._logger.info("Setting up strategy worker")

        # Connect to SHM
        self._shm_reader = SHMReader()
        try:
            self._shm_reader.connect()
            self._logger.info("Connected to SHM")
        except FileNotFoundError:
            self._logger.warning("SHM not found, will retry")

        # Initialize position tracker
        self._position_tracker = PositionTracker(initial_equity=self.config.starting_capital)

        # Initialize metrics
        self._metrics = MetricsCalculator(initial_equity=self.config.starting_capital)

        # Initialize Supabase store
        if self._supabase_config and self._supabase_config.url:
            self._supabase_store = SupabaseStore(self._supabase_config)
        else:
            self._supabase_store = _DummyStore()

        # Initialize paper executor
        self._paper_executor = PaperExecutor(
            position_tracker=self._position_tracker,
            supabase_store=self._supabase_store,  # type: ignore[arg-type]
        )

        # Initialize quote and size models
        self._quote_model = create_quote_model(self.config.quote_model)
        self._size_model = create_size_model(self.config.size_model)
        self._logger.info(
            "Models initialized",
            quote_model=self.config.quote_model.type,
            size_model=self.config.size_model.type,
        )

    def teardown(self) -> None:
        """Cleanup resources."""
        self._logger.info("Tearing down strategy worker")
        if self._shm_reader:
            self._shm_reader.close()
        if isinstance(self._supabase_store, SupabaseStore):
            self._supabase_store.close()

    async def on_new_market(self, market: MarketInfo) -> None:
        """Handle new market discovery."""
        if self._matches_config(market):
            self._active_markets[market.slug] = market

            # Create market-specific quote model with end_ts for real-time T calculation
            market_quote_model = create_quote_model(
                self.config.quote_model,
                end_ts_ms=market.end_ts,
            )
            self._market_quote_models[market.slug] = market_quote_model

            self._logger.info(
                "Subscribed to market",
                slug=market.slug,
                asset=market.asset,
                timeframe=market.timeframe,
                end_ts=market.end_ts,
                time_to_expiry_s=market.time_to_expiry_s,
            )

    def _matches_config(self, market: MarketInfo) -> bool:
        """Check if market matches this strategy's config."""
        return market.asset in self.config.assets and market.timeframe == self.config.timeframe

    async def run(self) -> None:
        """Main worker loop."""
        self._state.running = True
        tick_interval = self._tick_interval_ms / 1000.0

        self._logger.info("Starting strategy worker loop")

        while self._state.running:
            try:
                await self._tick()
            except Exception as e:
                self._logger.error("Error in tick", error=str(e))

            await asyncio.sleep(tick_interval)

    async def _tick(self) -> None:
        """Single tick of the strategy loop."""
        now = time.time()
        self._state.tick_count += 1

        # Ensure SHM connection
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

        # Process fills for open orders
        fills: list[Fill] = []
        if self._paper_executor and self._metrics:
            for market in markets.values():
                market_fills = self._paper_executor.check_open_orders(market)
                fills.extend(market_fills)
                for fill in market_fills:
                    self._metrics.record_trade(fill.pnl)

        # Update unrealized PnL
        self._update_unrealized_pnl(markets)

        # Generate and place quotes (throttled)
        quote_interval = self._quote_interval_ms / 1000.0
        if now - self._last_quote_time >= quote_interval:
            self._generate_quotes(markets)
            self._last_quote_time = now

        # Broadcast to dashboard (throttled)
        broadcast_interval = self._broadcast_interval_ms / 1000.0
        if now - self._state.last_broadcast_time >= broadcast_interval:
            await self._broadcast_state(markets, fills)
            self._state.last_broadcast_time = now

        # Periodic equity snapshot
        if now - self._state.last_equity_snapshot >= self._equity_snapshot_interval_s:
            self._snapshot_equity()
            self._state.last_equity_snapshot = now

    def _update_unrealized_pnl(self, markets: dict[str, MarketState]) -> None:
        """Update unrealized PnL for all positions."""
        if not self._position_tracker:
            return

        for asset_id, position in self._position_tracker.positions.items():
            if position.size > 0:
                market = markets.get(asset_id)
                if market and market.bids and market.asks:
                    mid_price = (market.bids[0][0] + market.asks[0][0]) / 2
                    self._position_tracker.update_unrealized_pnl(asset_id, mid_price)

    async def _broadcast_state(
        self,
        markets: dict[str, MarketState],
        recent_fills: list[Fill],
    ) -> None:
        """Send current state to dashboard."""
        if not self._update_queue or not self._position_tracker or not self._metrics:
            return

        # Build positions list
        positions: list[PositionInfo] = []
        for pos in self._position_tracker.get_all_positions():
            if pos:
                positions.append(
                    PositionInfo(
                        asset_id=pos["asset_id"],
                        slug=pos["slug"],
                        side=pos["side"],
                        size=pos["size"],
                        avg_entry_price=pos["avg_entry_price"],
                        unrealized_pnl=pos["unrealized_pnl"],
                        realized_pnl=pos["realized_pnl"],
                    )
                )

        # Build quotes list
        quotes: list[MarketQuote] = []
        for asset_id, market in markets.items():
            if not market.bids or not market.asks:
                continue

            # Parse slug info
            slug = asset_id[:32]  # Simplified
            asset = ""
            timeframe = ""

            # Get active market info if available
            for mkt in self._active_markets.values():
                if mkt.token_id_up == asset_id or mkt.token_id_down == asset_id:
                    slug = mkt.slug
                    asset = mkt.asset
                    timeframe = mkt.timeframe
                    break

            position = self._position_tracker.positions.get(asset_id)
            inventory = position.size if position else 0.0

            best_bid = market.bids[0][0]
            best_ask = market.asks[0][0]
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            # Get our quotes if available
            our_bid: float | None = None
            our_ask: float | None = None
            quote_result = self._last_quote_results.get(slug)
            if quote_result and quote_result.should_quote:
                our_bid = quote_result.bid_price
                our_ask = quote_result.ask_price

            # Get time to expiry
            time_to_expiry_s = 0.0
            for mkt in self._active_markets.values():
                if mkt.slug == slug:
                    time_to_expiry_s = mkt.time_to_expiry_s
                    break

            quotes.append(
                MarketQuote(
                    slug=slug,
                    asset=asset,
                    timeframe=timeframe,
                    our_bid=our_bid,
                    our_ask=our_ask,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    mid_price=mid_price,
                    spread=spread,
                    bids=[LevelInfo(price=p, size=s) for p, s in market.bids[:5]],
                    asks=[LevelInfo(price=p, size=s) for p, s in market.asks[:5]],
                    inventory=inventory,
                    time_to_expiry_s=time_to_expiry_s,
                )
            )

        # Build fills list
        fill_infos: list[FillInfo] = []
        for fill in recent_fills:
            fill_infos.append(
                FillInfo(
                    signal_id=fill.signal_id,
                    asset_id=fill.asset_id,
                    slug=fill.slug,
                    side="BUY" if fill.side == SIDE_BUY else "SELL",
                    price=fill.price,
                    size=fill.size,
                    pnl=fill.pnl,
                    timestamp_ms=fill.timestamp_ns // 1_000_000,
                )
            )

        # Also include recent fills from paper executor
        if not self._paper_executor:
            return
        # Use signal_id set for deduplication instead of object identity
        recent_signal_ids = {f.signal_id for f in recent_fills}
        for fill in self._paper_executor.get_recent_fills(20):
            if fill.signal_id not in recent_signal_ids:
                fill_infos.append(
                    FillInfo(
                        signal_id=fill.signal_id,
                        asset_id=fill.asset_id,
                        slug=fill.slug,
                        side="BUY" if fill.side == SIDE_BUY else "SELL",
                        price=fill.price,
                        size=fill.size,
                        pnl=fill.pnl,
                        timestamp_ms=fill.timestamp_ns // 1_000_000,
                    )
                )

        # Get metrics state
        metrics_state = self._metrics.get_state()

        # Calculate total inventory
        total_inventory = sum(pos.size * (1 if pos.side == "up" else -1) for pos in positions)

        # Update inventory history
        now_ms = int(time.time() * 1000)
        self._state.inventory_history.append((now_ms, total_inventory))
        # Keep only last 172800 points (24 hours @ 500ms interval)
        if len(self._state.inventory_history) > 172800:
            self._state.inventory_history = self._state.inventory_history[-172800:]

        # Build quote history points (send full history)
        quote_history_points = [
            QuoteHistoryPoint(
                timestamp_ms=ts,
                slug=slug,
                our_bid=bid,
                our_ask=ask,
                mid_price=mid,
                best_bid=best_bid,
                best_ask=best_ask,
            )
            for ts, slug, bid, ask, mid, best_bid, best_ask in self._state.quote_history
        ]

        # Build state
        state = DashboardState(
            strategy_id=self.strategy_id,
            strategy_name=self.config.name,
            starting_capital=self.config.starting_capital,
            current_equity=self._position_tracker.total_equity,
            cash=self._position_tracker.cash,
            total_pnl=self._position_tracker.total_pnl,
            realized_pnl=self._position_tracker.realized_pnl,
            unrealized_pnl=self._position_tracker.unrealized_pnl,
            positions=positions,
            total_inventory=total_inventory,
            quotes=quotes,
            recent_fills=fill_infos[:20],
            total_trades=metrics_state.total_trades,
            win_count=metrics_state.win_count,
            win_rate=metrics_state.win_rate,
            sharpe_ratio=metrics_state.sharpe_ratio,
            max_drawdown=metrics_state.max_drawdown,
            current_drawdown=metrics_state.current_drawdown,
            equity_history=self._position_tracker.get_equity_history(172800),
            inventory_history=self._state.inventory_history,
            quote_history=quote_history_points,
            status="running",
            timestamp_ms=now_ms,
        )

        # Send to queue
        try:
            self._update_queue.put_nowait(
                {
                    "type": "state",
                    "strategy_id": self.strategy_id,
                    "data": state.model_dump(),
                }
            )
        except Exception as e:
            self._logger.debug("Failed to put state in queue", error=str(e))

    def _generate_quotes(self, markets: dict[str, MarketState]) -> None:
        """Generate quotes for active markets and place orders."""
        if not self._paper_executor or not self._position_tracker or not self._size_model:
            return

        for slug, market_info in list(self._active_markets.items()):
            # Skip expired markets
            if market_info.is_expired:
                self._logger.debug("Skipping expired market", slug=slug)
                # Clean up expired market
                del self._active_markets[slug]
                if slug in self._market_quote_models:
                    del self._market_quote_models[slug]
                if slug in self._last_quote_results:
                    del self._last_quote_results[slug]
                continue

            # Get the UP token market data (we quote on UP token)
            asset_id = market_info.token_id_up
            market = markets.get(asset_id)
            if not market or not market.bids or not market.asks:
                continue

            # Get market-specific quote model or fallback to default
            quote_model = self._market_quote_models.get(slug, self._quote_model)
            if not quote_model:
                continue

            # Build strategy state with net position across UP/DOWN tokens
            state = self._build_strategy_state(market_info, market)

            # Compute quote
            try:
                quote_result = quote_model.compute(state)
            except Exception as e:
                self._logger.debug("Quote computation failed", slug=slug, error=str(e))
                continue

            # Store quote result for display
            self._last_quote_results[slug] = quote_result

            # Record quote history if we should quote
            if quote_result.should_quote:
                now_ms = int(time.time() * 1000)
                best_bid = market.bids[0][0]
                best_ask = market.asks[0][0]
                mid_price = (best_bid + best_ask) / 2
                self._state.quote_history.append(
                    (
                        now_ms,
                        slug,
                        quote_result.bid_price,
                        quote_result.ask_price,
                        mid_price,
                        best_bid,
                        best_ask,
                    )
                )
                # Keep only last 172800 points (24 hours @ 500ms interval)
                if len(self._state.quote_history) > 172800:
                    self._state.quote_history = self._state.quote_history[-172800:]

            if not quote_result.should_quote:
                self._logger.debug(
                    "Not quoting",
                    slug=slug,
                    reason=quote_result.reason,
                )
                continue

            # Compute sizes
            try:
                size_result = self._size_model.compute(state, quote_result)
            except Exception as e:
                self._logger.debug("Size computation failed", slug=slug, error=str(e))
                continue

            # Place bid order (want to go long)
            if size_result.bid_size > 0:
                self._place_quote_order(
                    slug=slug,
                    market_info=market_info,
                    quote_side=SIDE_BUY,
                    price=quote_result.bid_price,
                    size=size_result.bid_size,
                    markets=markets,
                )

            # Place ask order (want to go short)
            if size_result.ask_size > 0:
                self._place_quote_order(
                    slug=slug,
                    market_info=market_info,
                    quote_side=SIDE_SELL,
                    price=quote_result.ask_price,
                    size=size_result.ask_size,
                    markets=markets,
                )

    def _get_net_position(self, market_info: MarketInfo) -> tuple[float, float, float]:
        """
        Calculate net position for a market across both UP and DOWN tokens.

        Returns:
            Tuple of (net_position, avg_entry_price, unrealized_pnl)
            - net_position: +value = long, -value = short
            - UP tokens contribute positive position
            - DOWN tokens contribute negative position
        """
        if not self._position_tracker:
            return 0.0, 0.0, 0.0

        up_pos = self._position_tracker.get_position(market_info.token_id_up)
        down_pos = self._position_tracker.get_position(market_info.token_id_down)

        up_size = up_pos["size"] if up_pos else 0.0
        down_size = down_pos["size"] if down_pos else 0.0

        # Net position: UP is positive, DOWN is negative
        net_position = up_size - down_size

        # Weighted average entry price
        up_entry = up_pos["avg_entry_price"] if up_pos else 0.0
        down_entry = down_pos["avg_entry_price"] if down_pos else 0.0

        if abs(net_position) > 0.001:
            if net_position > 0:
                avg_entry = up_entry
            else:
                avg_entry = down_entry
        else:
            avg_entry = 0.0

        # Sum unrealized PnL from both positions
        up_unrealized = up_pos["unrealized_pnl"] if up_pos else 0.0
        down_unrealized = down_pos["unrealized_pnl"] if down_pos else 0.0
        unrealized_pnl = up_unrealized + down_unrealized

        return net_position, avg_entry, unrealized_pnl

    def _build_strategy_state(self, market_info: MarketInfo, market: MarketState) -> StrategyState:
        """Build StrategyState for quote/size model computation."""
        # Get net position across UP and DOWN tokens
        position: PositionState | None = None
        if self._position_tracker:
            net_position, avg_entry, unrealized_pnl = self._get_net_position(market_info)

            # Calculate realized PnL from both tokens
            up_pos = self._position_tracker.get_position(market_info.token_id_up)
            down_pos = self._position_tracker.get_position(market_info.token_id_down)
            realized_pnl = (up_pos["realized_pnl"] if up_pos else 0.0) + (
                down_pos["realized_pnl"] if down_pos else 0.0
            )

            if abs(net_position) > 0.001 or abs(realized_pnl) > 0.001:
                position = PositionState(
                    asset_id=market_info.token_id_up,  # Use UP token as primary identifier
                    position=net_position,
                    avg_entry_price=avg_entry,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=realized_pnl,
                )

        return StrategyState(
            market=market,
            position=position,
            total_equity=self._position_tracker.total_equity if self._position_tracker else 0.0,
            available_margin=self._position_tracker.cash if self._position_tracker else 0.0,
        )

    def _place_quote_order(
        self,
        slug: str,
        market_info: MarketInfo,
        quote_side: int,
        price: float,
        size: float,
        markets: dict[str, MarketState],
    ) -> None:
        """
        Place a quote order, choosing UP or DOWN token based on position.

        Args:
            slug: Market slug
            market_info: MarketInfo with both token_id_up and token_id_down
            quote_side: SIDE_BUY (bid) or SIDE_SELL (ask) - the quote direction
            price: Quote price
            size: Quote size
            markets: Dict of all markets for orderbook lookup
        """
        if not self._paper_executor or not self._metrics or not self._position_tracker:
            return

        # Get current positions for UP and DOWN tokens
        up_position = self._position_tracker.get_position(market_info.token_id_up)
        down_position = self._position_tracker.get_position(market_info.token_id_down)

        up_size = up_position["size"] if up_position else 0.0
        down_size = down_position["size"] if down_position else 0.0

        # Determine which token to trade and which side
        if quote_side == SIDE_BUY:  # BID quote
            # Bid = want to go long
            # If holding DOWN tokens >= order size, sell DOWN (close short)
            # Otherwise, buy UP (open long)
            if down_size >= size:
                asset_id = market_info.token_id_down
                trade_side = SIDE_SELL
                token_side = "down"
            else:
                asset_id = market_info.token_id_up
                trade_side = SIDE_BUY
                token_side = "up"
        else:  # ASK quote (SIDE_SELL)
            # Ask = want to go short
            # If holding UP tokens >= order size, sell UP (close long)
            # Otherwise, buy DOWN (open short)
            if up_size >= size:
                asset_id = market_info.token_id_up
                trade_side = SIDE_SELL
                token_side = "up"
            else:
                asset_id = market_info.token_id_down
                trade_side = SIDE_BUY
                token_side = "down"

        # Get market data for the selected token
        market = markets.get(asset_id)
        if not market or not market.bids or not market.asks:
            self._logger.debug(
                "No market data for token",
                slug=slug,
                token_side=token_side,
                asset_id=asset_id[:16] + "...",
            )
            return

        # Cancel any existing orders for this asset/side first
        existing_orders = self._paper_executor.get_open_orders(asset_id)
        for order in existing_orders:
            if order.signal.side == trade_side:
                self._paper_executor.cancel_order(order.signal.signal_id)

        # Create and submit new order
        signal = self._paper_executor.create_signal(
            asset_id=asset_id,
            slug=slug,
            side=trade_side,
            price=price,
            size=size,
            token_side=token_side,
        )

        fill = self._paper_executor.process_signal(signal, market)
        if fill:
            self._metrics.record_trade(fill.pnl)
            self._logger.info(
                "Quote filled",
                slug=slug,
                quote_side="BID" if quote_side == SIDE_BUY else "ASK",
                token_side=token_side,
                trade_side="BUY" if trade_side == SIDE_BUY else "SELL",
                price=fill.price,
                size=fill.size,
                pnl=fill.pnl,
            )

    def _snapshot_equity(self) -> None:
        """Take an equity snapshot."""
        if not self._position_tracker or not self._metrics:
            return

        equity = self._position_tracker.total_equity
        self._position_tracker.snapshot_equity()
        self._metrics.update_equity(
            equity=equity,
            realized_pnl=self._position_tracker.realized_pnl,
            unrealized_pnl=self._position_tracker.unrealized_pnl,
        )

    def stop(self) -> None:
        """Stop the worker."""
        self._state.running = False

    # Public API for placing orders
    def place_order(
        self,
        asset_id: str,
        slug: str,
        side: int,
        price: float,
        size: float,
    ) -> Fill | None:
        """Place a paper order."""
        if not self._paper_executor or not self._shm_reader:
            return None

        market = self._shm_reader.get_market(asset_id)
        if not market:
            return None

        signal = self._paper_executor.create_signal(
            asset_id=asset_id,
            slug=slug,
            side=side,
            price=price,
            size=size,
        )

        fill = self._paper_executor.process_signal(signal, market)
        if fill and self._metrics:
            self._metrics.record_trade(fill.pnl)

        return fill

    def get_state(self) -> dict[str, Any]:
        """Get current state as dict."""
        if not self._position_tracker or not self._metrics:
            return {}

        metrics_state = self._metrics.get_state()

        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.config.name,
            "starting_capital": self.config.starting_capital,
            "current_equity": self._position_tracker.total_equity,
            "cash": self._position_tracker.cash,
            "total_pnl": self._position_tracker.total_pnl,
            "realized_pnl": self._position_tracker.realized_pnl,
            "unrealized_pnl": self._position_tracker.unrealized_pnl,
            "total_trades": metrics_state.total_trades,
            "win_rate": metrics_state.win_rate,
            "sharpe_ratio": metrics_state.sharpe_ratio,
            "max_drawdown": metrics_state.max_drawdown,
            "status": "running" if self._state.running else "stopped",
        }
