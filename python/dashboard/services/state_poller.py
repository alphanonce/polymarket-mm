"""
State Poller Service

Polls Supabase for strategy state and broadcasts updates via WebSocket.
This allows the dashboard to operate independently from the strategy executor.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from dashboard.models.schemas import (
    DashboardState,
    FillInfo,
    LevelInfo,
    MarketQuote,
    PositionInfo,
    QuoteHistoryPoint,
)
from paper.supabase_store import SupabaseConfig, SupabaseStore
from paper.token_registry import get_global_registry
from strategy.shm.reader import SHMReader
from strategy.shm.types import MarketState

if TYPE_CHECKING:
    from dashboard.services.broadcast import BroadcastHub

logger = structlog.get_logger()


@dataclass
class StatePollerConfig:
    """Configuration for the state poller."""

    poll_interval_ms: int = 500
    equity_history_limit: int = 500
    trade_history_limit: int = 50
    quote_history_limit: int = 500


class StrategyStateCache:
    """Cached state for a single strategy."""

    def __init__(self, strategy_id: str) -> None:
        self.strategy_id = strategy_id
        self.name: str = strategy_id  # Will be updated from config if available
        self.last_state: DashboardState | None = None
        self.last_update_time: float = 0.0
        self.status: str = "unknown"


class StatePoller:
    """
    Polls Supabase for strategy state and broadcasts to WebSocket clients.

    Architecture:
    - Strategies are defined in config (paper.yaml)
    - Runtime data (PnL, positions, trades) comes from Supabase
    - Live orderbook data comes from SHM
    - Broadcasts state updates via BroadcastHub
    """

    def __init__(
        self,
        supabase_config: SupabaseConfig,
        broadcast_hub: BroadcastHub,
        config: StatePollerConfig | None = None,
        strategy_configs: list[Any] | None = None,  # List of StrategyConfig
    ) -> None:
        self._supabase_config = supabase_config
        self._broadcast_hub = broadcast_hub
        self._config = config or StatePollerConfig()
        self._strategy_configs = strategy_configs or []

        self._store: SupabaseStore | None = None
        self._strategies: dict[str, StrategyStateCache] = {}
        self._running = False
        self._poll_task: asyncio.Task[None] | None = None
        self._logger = logger.bind(component="state_poller")

        # SHM reader for live orderbook data
        self._shm_reader: SHMReader | None = None
        self._shm_markets: dict[str, MarketState] = {}

        # Pre-populate strategies from config
        for sc in self._strategy_configs:
            cache = StrategyStateCache(sc.strategy_id)
            cache.name = sc.name
            cache.status = "configured"
            self._strategies[sc.strategy_id] = cache
            self._logger.debug("Loaded strategy from config", strategy_id=sc.strategy_id)

    async def start(self) -> None:
        """Start the state poller."""
        self._logger.info("Starting state poller")

        # Initialize Supabase store
        self._store = SupabaseStore(self._supabase_config)
        if not self._store.health_check():
            self._logger.error("Supabase health check failed")
            return

        # Try to connect to SHM for live orderbook data
        try:
            self._shm_reader = SHMReader()
            self._shm_reader.connect()
            self._logger.info("Connected to SHM for live orderbook data")
        except Exception as e:
            self._logger.warning("SHM not available, orderbook will be empty", error=str(e))
            self._shm_reader = None

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())

        self._logger.info(
            "State poller started",
            poll_interval_ms=self._config.poll_interval_ms,
        )

    async def stop(self) -> None:
        """Stop the state poller."""
        self._running = False

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        if self._shm_reader:
            self._shm_reader.close()
            self._shm_reader = None

        if self._store:
            self._store.close()
            self._store = None

        self._logger.info("State poller stopped")

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        poll_interval_s = self._config.poll_interval_ms / 1000.0

        while self._running:
            try:
                await self._poll_once()
            except Exception as e:
                self._logger.error("Error in poll loop", error=str(e))

            await asyncio.sleep(poll_interval_s)

    async def _poll_once(self) -> None:
        """Single poll iteration."""
        if not self._store:
            return

        # Refresh SHM data for live orderbook
        if self._shm_reader:
            try:
                self._shm_reader.refresh()
                self._shm_markets = self._shm_reader.get_markets_dict()
            except Exception as e:
                self._logger.debug("Failed to refresh SHM", error=str(e))
                self._shm_markets = {}

        # Use config-defined strategies (already in self._strategies from __init__)
        # Also discover additional strategies from Supabase metrics table
        db_strategies = self._store.get_strategies()
        excluded_strategies = {"test-integration", "default", "test", "test-live"}

        for strategy_info in db_strategies:
            strategy_id = strategy_info.get("strategy_id")
            if not strategy_id or strategy_id in excluded_strategies:
                continue

            # Add to cache if not from config (discovered from DB)
            if strategy_id not in self._strategies:
                self._strategies[strategy_id] = StrategyStateCache(strategy_id)
                self._logger.info("Discovered strategy from DB", strategy_id=strategy_id)

        # Fetch and broadcast state for ALL strategies (config + discovered)
        for strategy_id, cache in self._strategies.items():
            # Fetch and broadcast state
            try:
                state = await self._fetch_strategy_state(strategy_id)
                if state:
                    cache.last_state = state
                    cache.last_update_time = time.time()
                    cache.status = state.status

                    # Broadcast to WebSocket clients
                    await self._broadcast_hub.broadcast_state(
                        strategy_id, state.model_dump()
                    )
            except Exception as e:
                self._logger.error(
                    "Failed to fetch strategy state",
                    strategy_id=strategy_id,
                    error=str(e),
                )

    async def _fetch_strategy_state(self, strategy_id: str) -> DashboardState | None:
        """Fetch complete state for a strategy from Supabase."""
        if not self._store:
            return None

        # Get strategy config if available
        strategy_config = None
        for sc in self._strategy_configs:
            if sc.strategy_id == strategy_id:
                strategy_config = sc
                break

        # Fetch metrics (may be empty for new strategies)
        metrics = self._store.get_latest_metrics(strategy_id) or {}

        # For config-defined strategies, we can return state even without metrics
        # (they'll have default values until paper runner populates data)

        # Fetch positions
        positions_data = self._store.get_positions(strategy_id)

        # Fetch equity history
        equity_history_data = self._store.get_equity_history(
            strategy_id, limit=self._config.equity_history_limit
        )

        # Fetch recent trades
        trades_data = self._store.get_recent_trades(
            strategy_id, limit=self._config.trade_history_limit
        )

        # Convert to schema
        positions = [
            PositionInfo(
                asset_id=p.get("asset_id", ""),
                slug=p.get("slug", ""),
                side=p.get("side", ""),
                size=p.get("size", 0.0),
                avg_entry_price=p.get("avg_entry_price", 0.0),
                unrealized_pnl=p.get("unrealized_pnl", 0.0),
                realized_pnl=p.get("realized_pnl", 0.0),
                current_price=p.get("current_price", 0.0),
            )
            for p in positions_data
        ]

        recent_fills = [
            FillInfo(
                signal_id=t.get("id", 0),
                asset_id=t.get("asset_id", ""),
                slug=t.get("slug", ""),
                side=self._convert_side(t.get("side")),
                price=t.get("price", 0.0),
                size=t.get("size", 0.0),
                pnl=t.get("pnl", 0.0),
                timestamp_ms=self._parse_timestamp_ms(t.get("timestamp")),
            )
            for t in trades_data
        ]

        # Equity history: convert to (timestamp_ms, equity) tuples
        # Reverse to get chronological order
        equity_history = [
            (self._parse_timestamp_ms(e.get("timestamp")), e.get("equity", 0.0))
            for e in reversed(equity_history_data)
        ]

        # Calculate derived metrics
        total_inventory = sum(p.size for p in positions)
        # Use config starting_capital if available, else from metrics, else default
        if strategy_config:
            starting_capital = strategy_config.starting_capital
        else:
            starting_capital = metrics.get("starting_capital", 10000.0)
        win_count = metrics.get("win_count", 0)
        total_trades = metrics.get("total_trades", 0)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0

        # Calculate PnL from positions (more reliable than metrics table)
        pnl_by_asset: dict[str, float] = {}
        total_realized_pnl = 0.0
        total_unrealized_pnl = 0.0
        for p in positions:
            asset = self._extract_asset(p.slug)
            pnl = p.realized_pnl + p.unrealized_pnl
            pnl_by_asset[asset] = pnl_by_asset.get(asset, 0.0) + pnl
            total_realized_pnl += p.realized_pnl
            total_unrealized_pnl += p.unrealized_pnl

        # Use positions-based PnL if metrics are 0 or NULL (fallback)
        # Coalesce NULL values from Supabase to 0.0
        metrics_total_pnl = metrics.get("total_pnl") or 0.0
        calculated_total_pnl = total_realized_pnl + total_unrealized_pnl
        total_pnl = calculated_total_pnl if metrics_total_pnl == 0 else metrics_total_pnl
        metrics_realized = metrics.get("realized_pnl") or 0.0
        metrics_unrealized = metrics.get("unrealized_pnl") or 0.0
        realized_pnl = total_realized_pnl if metrics_realized == 0 else metrics_realized
        unrealized_pnl = total_unrealized_pnl if metrics_unrealized == 0 else metrics_unrealized
        current_equity = total_pnl + starting_capital

        # Build quotes from SHM (live orderbook) or market snapshots (historical)
        quotes: list[MarketQuote] = []
        quote_history: list[QuoteHistoryPoint] = []

        # Build position lookup by asset_id
        position_by_asset: dict[str, PositionInfo] = {p.asset_id: p for p in positions}

        # Token registry for slug/asset lookup
        token_registry = get_global_registry()

        # Build quotes from live SHM data
        for asset_id, market in self._shm_markets.items():
            if not market.bids and not market.asks:
                continue

            # Get position for inventory
            position = position_by_asset.get(asset_id)
            inventory = position.size if position else 0.0

            # Get slug from registry or position
            registry_slug = token_registry.get_slug(asset_id)
            slug = registry_slug or (position.slug if position else asset_id)

            # Extract asset and timeframe - prefer registry, then parse slug
            registry_asset = token_registry.get_asset(asset_id)
            registry_timeframe = token_registry.get_timeframe(asset_id)

            if registry_asset:
                asset = registry_asset
                timeframe = registry_timeframe or "15m"
            else:
                # Parse from slug if available
                parts = slug.split("-") if slug and not slug.isdigit() else []
                asset = parts[0] if parts else "unknown"
                timeframe = parts[2] if len(parts) > 2 else "15m"

            # Build orderbook levels
            bids = [LevelInfo(price=p, size=s) for p, s in (market.bids or [])[:10]]
            asks = [LevelInfo(price=p, size=s) for p, s in (market.asks or [])[:10]]

            best_bid = market.bids[0][0] if market.bids else 0.0
            best_ask = market.asks[0][0] if market.asks else 0.0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
            spread = best_ask - best_bid if best_bid and best_ask else 0.0

            quotes.append(
                MarketQuote(
                    slug=slug,
                    asset=asset,
                    timeframe=timeframe,
                    our_bid=None,
                    our_ask=None,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    mid_price=mid_price,
                    spread=spread,
                    bids=bids,
                    asks=asks,
                    inventory=inventory,
                )
            )

        # If no SHM data, fallback to market snapshots
        if not quotes:
            slugs = {p.slug for p in positions if p.slug}
            for slug in slugs:
                snapshots = self._store.get_market_snapshots(
                    slug, limit=self._config.quote_history_limit
                )
                if snapshots:
                    latest = snapshots[0]
                    quotes.append(
                        MarketQuote(
                            slug=slug,
                            asset=slug.split("-")[0] if "-" in slug else slug,
                            timeframe=slug.split("-")[2] if len(slug.split("-")) > 2 else "15m",
                            our_bid=latest.get("our_bid"),
                            our_ask=latest.get("our_ask"),
                            best_bid=latest.get("best_bid", 0.0),
                            best_ask=latest.get("best_ask", 0.0),
                            mid_price=latest.get("mid_price", 0.0),
                            spread=latest.get("spread", 0.0),
                            inventory=latest.get("inventory", 0.0),
                        )
                    )

                    # Quote history (reversed for chronological order)
                    for snap in reversed(snapshots):
                        quote_history.append(
                            QuoteHistoryPoint(
                                timestamp_ms=self._parse_timestamp_ms(snap.get("timestamp")),
                                slug=slug,
                                our_bid=snap.get("our_bid"),
                                our_ask=snap.get("our_ask"),
                                mid_price=snap.get("mid_price", 0.0),
                                best_bid=snap.get("best_bid", 0.0),
                                best_ask=snap.get("best_ask", 0.0),
                            )
                        )

        # Determine strategy name and status
        if strategy_config:
            strategy_name = strategy_config.name
            status = "running" if metrics.get("updated_at") else "configured"
        else:
            strategy_name = self._strategies.get(strategy_id, StrategyStateCache(strategy_id)).name
            status = "running" if metrics.get("updated_at") else "unknown"

        return DashboardState(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            starting_capital=starting_capital,
            current_equity=current_equity,
            cash=current_equity - sum(p.size * p.avg_entry_price for p in positions),
            total_pnl=total_pnl,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            pnl_by_asset=pnl_by_asset,
            positions=positions,
            total_inventory=total_inventory,
            quotes=quotes,
            recent_fills=recent_fills,
            total_trades=total_trades,
            win_count=win_count,
            win_rate=win_rate,
            sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
            max_drawdown=metrics.get("max_drawdown", 0.0),
            current_drawdown=metrics.get("current_drawdown", 0.0),
            equity_history=equity_history,
            quote_history=quote_history,
            status=status,
            timestamp_ms=int(time.time() * 1000),
        )

    def _parse_timestamp_ms(self, ts: Any) -> int:
        """Parse timestamp to milliseconds."""
        if ts is None:
            return 0
        if isinstance(ts, int):
            return ts if ts > 1e12 else int(ts * 1000)
        if isinstance(ts, float):
            return int(ts * 1000) if ts < 1e12 else int(ts)
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return int(dt.timestamp() * 1000)
            except Exception:
                return 0
        return 0

    def _convert_side(self, side: Any) -> str:
        """Convert side value to string."""
        if side is None:
            return ""
        if isinstance(side, str):
            return side
        if isinstance(side, int):
            # 1 = BUY, -1 = SELL (common convention)
            if side == 1:
                return "BUY"
            elif side == -1:
                return "SELL"
            return str(side)
        return str(side)

    @staticmethod
    def _extract_asset(slug: str) -> str:
        """Extract asset name from slug (e.g., btc-updown-15m -> btc)."""
        if slug:
            parts = slug.lower().split("-")
            if parts:
                return parts[0]
        return "unknown"

    def get_strategy(self, strategy_id: str) -> StrategyStateCache | None:
        """Get cached strategy state."""
        return self._strategies.get(strategy_id)

    def get_all_strategies(self) -> list[StrategyStateCache]:
        """Get all cached strategies."""
        return list(self._strategies.values())

    def get_strategy_ids(self) -> list[str]:
        """Get list of known strategy IDs."""
        return list(self._strategies.keys())
