"""
Paper Executor

Intercepts order signals and simulates execution against live orderbook.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import structlog

from strategy.shm.types import SIDE_BUY, SIDE_SELL, MarketState

if TYPE_CHECKING:
    from paper.position_tracker import PositionTracker
    from paper.supabase_store import SupabaseStore

logger = structlog.get_logger()


@dataclass
class OrderSignal:
    """Order signal for paper trading."""

    signal_id: int
    asset_id: str
    slug: str
    side: int  # SIDE_BUY or SIDE_SELL
    price: float
    size: float
    timestamp_ns: int


@dataclass
class Fill:
    """Represents a fill (executed trade)."""

    signal_id: int
    asset_id: str
    slug: str
    side: int
    price: float
    size: float
    timestamp_ns: int
    pnl: float = 0.0


@dataclass
class PaperOrder:
    """Open paper order."""

    signal: OrderSignal
    created_at_ns: int
    status: str = "open"  # open, filled, cancelled


class PaperExecutor:
    """
    Intercepts order signals and simulates execution.

    Fill logic:
    - BUY orders fill if order price >= best ask
    - SELL orders fill if order price <= best bid
    - Fills at the touch (best bid/ask), not at limit price
    """

    def __init__(
        self,
        position_tracker: "PositionTracker",
        supabase_store: "SupabaseStore",
        fill_delay_ms: float = 0.0,
    ):
        self.position_tracker = position_tracker
        self.store = supabase_store
        self.fill_delay_ms = fill_delay_ms

        self.open_orders: Dict[int, PaperOrder] = {}
        self.fills: List[Fill] = []
        self.signal_id_counter: int = 0

        self.logger = logger.bind(component="paper_executor")

    def create_signal(
        self,
        asset_id: str,
        slug: str,
        side: int,
        price: float,
        size: float,
    ) -> OrderSignal:
        """Create an order signal."""
        self.signal_id_counter += 1
        return OrderSignal(
            signal_id=self.signal_id_counter,
            asset_id=asset_id,
            slug=slug,
            side=side,
            price=price,
            size=size,
            timestamp_ns=time.time_ns(),
        )

    def process_signal(
        self,
        signal: OrderSignal,
        market: MarketState,
    ) -> Optional[Fill]:
        """
        Process an order signal against the current orderbook.

        Returns Fill if order was executed, None otherwise.
        """
        if not market.bids or not market.asks:
            self.logger.debug("No orderbook data", asset_id=signal.asset_id)
            return None

        fill = None

        if signal.side == SIDE_BUY:
            # Buy at best ask if price crosses
            best_ask_price, best_ask_size = market.asks[0]
            if signal.price >= best_ask_price and best_ask_size > 0:
                fill_price = best_ask_price
                fill_size = min(signal.size, best_ask_size)
                fill = self._execute_fill(signal, fill_price, fill_size)
        else:
            # Sell at best bid if price crosses
            best_bid_price, best_bid_size = market.bids[0]
            if signal.price <= best_bid_price and best_bid_size > 0:
                fill_price = best_bid_price
                fill_size = min(signal.size, best_bid_size)
                fill = self._execute_fill(signal, fill_price, fill_size)

        if fill is None:
            # Order didn't fill, add to open orders
            self.open_orders[signal.signal_id] = PaperOrder(
                signal=signal,
                created_at_ns=time.time_ns(),
            )
            self.logger.debug(
                "Order queued",
                signal_id=signal.signal_id,
                side="BUY" if signal.side == SIDE_BUY else "SELL",
                price=signal.price,
                size=signal.size,
            )

        return fill

    def check_open_orders(self, market: MarketState) -> List[Fill]:
        """
        Check if any open orders can be filled against current market.

        Returns list of fills.
        """
        fills = []
        filled_ids = []

        for signal_id, order in self.open_orders.items():
            if order.signal.asset_id != market.asset_id:
                continue

            if not market.bids or not market.asks:
                continue

            signal = order.signal
            fill = None

            if signal.side == SIDE_BUY:
                best_ask_price, best_ask_size = market.asks[0]
                if signal.price >= best_ask_price and best_ask_size > 0:
                    fill_price = best_ask_price
                    fill_size = min(signal.size, best_ask_size)
                    fill = self._execute_fill(signal, fill_price, fill_size)
            else:
                best_bid_price, best_bid_size = market.bids[0]
                if signal.price <= best_bid_price and best_bid_size > 0:
                    fill_price = best_bid_price
                    fill_size = min(signal.size, best_bid_size)
                    fill = self._execute_fill(signal, fill_price, fill_size)

            if fill:
                fills.append(fill)
                filled_ids.append(signal_id)

        # Remove filled orders
        for signal_id in filled_ids:
            del self.open_orders[signal_id]

        return fills

    def _execute_fill(
        self,
        signal: OrderSignal,
        fill_price: float,
        fill_size: float,
    ) -> Fill:
        """
        Execute a fill, update position tracker, and store to Supabase.
        """
        # Record trade in position tracker
        trade = self.position_tracker.record_trade(
            asset_id=signal.asset_id,
            slug=signal.slug,
            side=signal.side,
            price=fill_price,
            size=fill_size,
        )

        fill = Fill(
            signal_id=signal.signal_id,
            asset_id=signal.asset_id,
            slug=signal.slug,
            side=signal.side,
            price=fill_price,
            size=fill_size,
            timestamp_ns=time.time_ns(),
            pnl=trade.get("pnl", 0.0),
        )

        self.fills.append(fill)

        # Store to Supabase
        self.store.insert_trade(trade)

        position = self.position_tracker.get_position(signal.asset_id)
        if position:
            self.store.upsert_position(position)

        self.logger.info(
            "Fill executed",
            signal_id=signal.signal_id,
            asset_id=signal.asset_id,
            side="BUY" if signal.side == SIDE_BUY else "SELL",
            price=fill_price,
            size=fill_size,
            pnl=fill.pnl,
        )

        return fill

    def cancel_order(self, signal_id: int) -> bool:
        """Cancel an open order."""
        if signal_id in self.open_orders:
            self.open_orders[signal_id].status = "cancelled"
            del self.open_orders[signal_id]
            return True
        return False

    def cancel_all_orders(self, asset_id: Optional[str] = None) -> int:
        """Cancel all open orders, optionally for a specific asset."""
        cancelled = 0
        ids_to_remove = []

        for signal_id, order in self.open_orders.items():
            if asset_id is None or order.signal.asset_id == asset_id:
                ids_to_remove.append(signal_id)
                cancelled += 1

        for signal_id in ids_to_remove:
            del self.open_orders[signal_id]

        return cancelled

    def get_open_orders(self, asset_id: Optional[str] = None) -> List[PaperOrder]:
        """Get open orders, optionally filtered by asset."""
        if asset_id is None:
            return list(self.open_orders.values())
        return [o for o in self.open_orders.values() if o.signal.asset_id == asset_id]

    def get_recent_fills(self, limit: int = 100) -> List[Fill]:
        """Get recent fills."""
        return self.fills[-limit:]
