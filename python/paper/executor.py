"""
Paper Executor

Intercepts order signals and simulates execution against live orderbook.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import structlog

from paper.token_registry import get_global_registry
from strategy.shm.types import SIDE_BUY, MarketState
from strategy.utils.polymarket import round_ask, round_bid, round_quantity

if TYPE_CHECKING:
    from paper.position_tracker import PositionTracker


class TradeStore(Protocol):
    """Protocol for trade storage."""

    def insert_trade(self, trade: dict[str, Any]) -> None: ...
    def upsert_position(self, position: dict[str, Any]) -> None: ...


logger = structlog.get_logger()

# Maximum age of a signal relative to market data timestamp (5 seconds)
MAX_SIGNAL_AGE_NS = 5_000_000_000


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
    token_side: str = "up"  # "up" or "down" - the token's outcome side


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
    token_side: str = "up"  # "up" or "down" - which token was traded


@dataclass
class PaperOrder:
    """Open paper order."""

    signal: OrderSignal
    created_at_ns: int
    status: str = "open"  # open, partially_filled, filled, cancelled
    remaining_size: float = 0.0  # Remaining size to fill (initialized from signal.size)
    post_only_check_at_ns: int = 0  # POST_ONLY verification time (created_at + 100ms)
    post_only_verified: bool = False  # True once POST_ONLY check passed

    def __post_init__(self) -> None:
        """Initialize remaining_size from signal if not set."""
        if self.remaining_size == 0.0:
            self.remaining_size = self.signal.size
        # Set POST_ONLY check time if not set (100ms after creation)
        if self.post_only_check_at_ns == 0:
            self.post_only_check_at_ns = self.created_at_ns + 100_000_000


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
        supabase_store: TradeStore,
        fill_delay_ms: float = 0.0,
    ):
        self.position_tracker = position_tracker
        self.store = supabase_store
        self.fill_delay_ms = fill_delay_ms

        # Orders indexed by asset_id for O(1) lookup, then by signal_id
        # Structure: {asset_id: {signal_id: PaperOrder}}
        self._orders_by_asset: dict[str, dict[int, PaperOrder]] = {}
        # Bounded deque to prevent unbounded memory growth
        self.fills: deque[Fill] = deque(maxlen=10000)
        self.signal_id_counter: int = 0

        self.logger = logger.bind(component="paper_executor")

    def create_signal(
        self,
        asset_id: str,
        slug: str,
        side: int,
        price: float,
        size: float,
        token_side: str | None = None,
    ) -> OrderSignal:
        """Create an order signal with price/size normalization."""
        self.signal_id_counter += 1

        # Normalize price based on side
        # BUY (bid) -> floor to tick, SELL (ask) -> ceil to tick
        if side == SIDE_BUY:
            normalized_price = round_bid(price)  # floors
        else:
            normalized_price = round_ask(price)  # ceils

        # Normalize quantity (floor to avoid over-sizing)
        normalized_size = round_quantity(size, direction="down")

        # Log when normalization changes values
        if normalized_price != price:
            self.logger.debug(
                "Price normalized",
                original=price,
                normalized=normalized_price,
                side="BUY" if side == SIDE_BUY else "SELL",
            )
        if normalized_size != size:
            self.logger.debug(
                "Size normalized",
                original=size,
                normalized=normalized_size,
            )

        # Look up token_side from registry if not provided
        if token_side is None:
            registry = get_global_registry()
            token_side = registry.get_outcome(asset_id) or "up"

        return OrderSignal(
            signal_id=self.signal_id_counter,
            asset_id=asset_id,
            slug=slug,
            side=side,
            price=normalized_price,
            size=normalized_size,
            timestamp_ns=time.time_ns(),
            token_side=token_side,
        )

    def process_signal(
        self,
        signal: OrderSignal,
        market: MarketState,
    ) -> Fill | None:
        """
        Process an order signal against the current orderbook.

        Returns Fill if order was executed (fully or partially), None otherwise.
        Partial fills keep the order open with remaining_size updated.
        """
        if not market.bids or not market.asks:
            self.logger.debug("No orderbook data", asset_id=signal.asset_id)
            return None

        # Validate signal is not stale relative to market data
        signal_age_ns = market.timestamp_ns - signal.timestamp_ns
        if signal_age_ns > MAX_SIGNAL_AGE_NS:
            self.logger.warning(
                "Signal is stale",
                signal_id=signal.signal_id,
                signal_age_ms=signal_age_ns // 1_000_000,
            )
            return None

        fill = None
        fill_size = 0.0

        if signal.side == SIDE_BUY:
            # Buy at best ask if price crosses
            best_ask_price, best_ask_size = market.asks[0]
            if signal.price >= best_ask_price and best_ask_size > 0:
                fill_price = best_ask_price
                fill_size = min(signal.size, best_ask_size)
                fill = self._execute_fill(signal, fill_price, fill_size, market.timestamp_ns)
        else:
            # Sell at best bid if price crosses
            best_bid_price, best_bid_size = market.bids[0]
            if signal.price <= best_bid_price and best_bid_size > 0:
                fill_price = best_bid_price
                fill_size = min(signal.size, best_bid_size)
                fill = self._execute_fill(signal, fill_price, fill_size, market.timestamp_ns)

        # Calculate remaining size after fill
        remaining = signal.size - fill_size

        if fill is None:
            # Order didn't fill at all, add to open orders
            order = PaperOrder(
                signal=signal,
                created_at_ns=time.time_ns(),
                remaining_size=signal.size,
            )
            self._add_order(order)
            self.logger.debug(
                "Order queued",
                signal_id=signal.signal_id,
                side="BUY" if signal.side == SIDE_BUY else "SELL",
                price=signal.price,
                size=signal.size,
            )
        elif remaining > 0.001:
            # Partial fill - keep order open with remaining size
            order = PaperOrder(
                signal=signal,
                created_at_ns=time.time_ns(),
                status="partially_filled",
                remaining_size=remaining,
            )
            self._add_order(order)
            self.logger.debug(
                "Order partially filled",
                signal_id=signal.signal_id,
                filled_size=fill_size,
                remaining_size=remaining,
            )

        return fill

    def submit_order(
        self,
        signal: OrderSignal,
        market: MarketState,
    ) -> None:
        """
        Submit order as POST_ONLY (no immediate execution).

        Order is registered as pending POST_ONLY verification.
        Verification happens at t+100ms in check_open_orders().
        If order would cross at verification time, it's cancelled.
        After verification, fills only occur when counterparty crosses our price.
        """
        if not market.bids or not market.asks:
            self.logger.debug("No orderbook data", asset_id=signal.asset_id)
            return

        # Validate signal is not stale relative to market data
        signal_age_ns = market.timestamp_ns - signal.timestamp_ns
        if signal_age_ns > MAX_SIGNAL_AGE_NS:
            self.logger.warning(
                "Signal is stale",
                signal_id=signal.signal_id,
                signal_age_ms=signal_age_ns // 1_000_000,
            )
            return

        # Register as open order (POST_ONLY pending verification)
        now_ns = time.time_ns()
        order = PaperOrder(
            signal=signal,
            created_at_ns=now_ns,
            post_only_check_at_ns=now_ns + 100_000_000,  # +100ms
            post_only_verified=False,
            remaining_size=signal.size,
        )
        self._add_order(order)

        self.logger.debug(
            "Order submitted (POST_ONLY pending)",
            signal_id=signal.signal_id,
            side="BUY" if signal.side == SIDE_BUY else "SELL",
            price=signal.price,
            size=signal.size,
            post_only_check_at_ms=order.post_only_check_at_ns // 1_000_000,
        )

    def check_open_orders(self, market: MarketState) -> list[Fill]:
        """
        Check open orders: POST_ONLY verification and fill detection.

        POST_ONLY simulation:
        1. Orders submitted via submit_order() are pending POST_ONLY verification
        2. At t+100ms, check if order would cross - if yes, cancel (POST_ONLY rejected)
        3. After verification, fills only occur when counterparty strictly crosses our price
        4. Fill quantity is sum of all crossing levels, not just best level

        Returns list of fills. Partial fills update remaining_size and keep order open.
        Uses O(1) lookup by asset_id instead of iterating all orders.
        """
        fills: list[Fill] = []
        fully_filled_ids: list[int] = []
        cancelled_ids: list[int] = []

        # O(1) lookup: only get orders for this specific asset
        asset_orders = self._orders_by_asset.get(market.asset_id, {})
        if not asset_orders:
            return fills

        if not market.bids or not market.asks:
            return fills

        now_ns = time.time_ns()

        for signal_id, order in list(asset_orders.items()):
            signal = order.signal

            # Check for stale orders (give open orders more leeway than new signals)
            order_age_ns = market.timestamp_ns - signal.timestamp_ns
            if order_age_ns > MAX_SIGNAL_AGE_NS * 2:
                self.logger.info(
                    "Cancelling stale order",
                    signal_id=signal_id,
                    order_age_ms=order_age_ns // 1_000_000,
                )
                cancelled_ids.append(signal_id)
                continue

            # Step 1: POST_ONLY verification (at t+100ms)
            if not order.post_only_verified:
                if now_ns >= order.post_only_check_at_ns:
                    # Time to verify POST_ONLY
                    if self._would_cross_for_post_only(signal, market):
                        # Would cross - reject order (POST_ONLY failure)
                        self.logger.debug(
                            "POST_ONLY rejected - would cross",
                            signal_id=signal_id,
                            side="BUY" if signal.side == SIDE_BUY else "SELL",
                            price=signal.price,
                            best_ask=market.asks[0][0],
                            best_bid=market.bids[0][0],
                        )
                        cancelled_ids.append(signal_id)
                        continue
                    else:
                        # Passed POST_ONLY check - order is now active
                        order.post_only_verified = True
                        self.logger.debug(
                            "POST_ONLY verified",
                            signal_id=signal_id,
                            side="BUY" if signal.side == SIDE_BUY else "SELL",
                            price=signal.price,
                        )
                else:
                    # Not yet time to verify, skip fill check
                    continue

            # Step 2: Fill check (only for verified orders)
            # Counterparty must strictly cross our price for fill
            crossing_qty = self._get_crossing_quantity(signal, market)
            if crossing_qty > 0.001:
                fill_size = min(order.remaining_size, crossing_qty)
                # Fill at best available price
                if signal.side == SIDE_BUY:
                    fill_price = market.asks[0][0]
                else:
                    fill_price = market.bids[0][0]

                fill = self._execute_fill(signal, fill_price, fill_size, market.timestamp_ns)
                fills.append(fill)

                # Update remaining size
                order.remaining_size -= fill.size
                if order.remaining_size <= 0.001:
                    order.status = "filled"
                    fully_filled_ids.append(signal_id)
                else:
                    order.status = "partially_filled"
                    self.logger.debug(
                        "Order partially filled via crossing",
                        signal_id=signal_id,
                        filled_size=fill.size,
                        remaining_size=order.remaining_size,
                    )

        # Remove fully filled and cancelled orders
        for signal_id in fully_filled_ids + cancelled_ids:
            self._remove_order(market.asset_id, signal_id)

        return fills

    def _execute_fill(
        self,
        signal: OrderSignal,
        fill_price: float,
        fill_size: float,
        reference_timestamp_ns: int | None = None,
    ) -> Fill:
        """
        Execute a fill, update position tracker, and store to Supabase.

        Args:
            signal: The order signal being filled
            fill_price: Price at which the fill occurs
            fill_size: Size of the fill
            reference_timestamp_ns: Optional market timestamp for consistency

        Returns:
            Fill object with execution details
        """
        # Use reference timestamp if provided, otherwise fall back to current time
        fill_timestamp_ns = reference_timestamp_ns or time.time_ns()

        # Record trade in position tracker
        trade = self.position_tracker.record_trade(
            asset_id=signal.asset_id,
            slug=signal.slug,
            side=signal.side,
            price=fill_price,
            size=fill_size,
            token_side=signal.token_side,
            timestamp_ns=fill_timestamp_ns,
        )

        fill = Fill(
            signal_id=signal.signal_id,
            asset_id=signal.asset_id,
            slug=signal.slug,
            side=signal.side,
            price=fill_price,
            size=fill_size,
            timestamp_ns=fill_timestamp_ns,
            pnl=trade.get("pnl", 0.0),
            token_side=signal.token_side,
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

    def _add_order(self, order: PaperOrder) -> None:
        """Add order to the index."""
        asset_id = order.signal.asset_id
        if asset_id not in self._orders_by_asset:
            self._orders_by_asset[asset_id] = {}
        self._orders_by_asset[asset_id][order.signal.signal_id] = order

    def _remove_order(self, asset_id: str, signal_id: int) -> None:
        """Remove order from the index."""
        if asset_id in self._orders_by_asset:
            self._orders_by_asset[asset_id].pop(signal_id, None)
            # Clean up empty asset buckets
            if not self._orders_by_asset[asset_id]:
                del self._orders_by_asset[asset_id]

    def _would_cross_for_post_only(self, signal: OrderSignal, market: MarketState) -> bool:
        """
        Check if order would cross for POST_ONLY validation.

        POST_ONLY orders are rejected if they would immediately match.
        Same price counts as crossing (would be matched immediately).

        Args:
            signal: The order signal to check
            market: Current market state

        Returns:
            True if order would cross (should be cancelled), False otherwise
        """
        if not market.bids or not market.asks:
            return False

        if signal.side == SIDE_BUY:
            # BUY @ price would cross if price >= best_ask (same or better)
            best_ask_price = market.asks[0][0]
            return signal.price >= best_ask_price
        else:
            # SELL @ price would cross if price <= best_bid (same or better)
            best_bid_price = market.bids[0][0]
            return signal.price <= best_bid_price

    def _get_crossing_quantity(self, signal: OrderSignal, market: MarketState) -> float:
        """
        Calculate total quantity that crosses our order price.

        For fills, counterparty must strictly cross our price (not equal).
        Sums quantities from all crossing price levels.

        Args:
            signal: The order signal
            market: Current market state

        Returns:
            Total quantity available to fill (sum of all crossing levels)
        """
        if not market.bids or not market.asks:
            return 0.0

        total_qty = 0.0

        if signal.side == SIDE_BUY:
            # Our BUY order fills when asks strictly cross (ask < our_bid)
            for price, size in market.asks:
                if price < signal.price:  # Strictly less than
                    total_qty += size
                else:
                    break  # Asks are sorted ascending
        else:
            # Our SELL order fills when bids strictly cross (bid > our_ask)
            for price, size in market.bids:
                if price > signal.price:  # Strictly greater than
                    total_qty += size
                else:
                    break  # Bids are sorted descending

        return total_qty

    def cancel_order(self, signal_id: int) -> bool:
        """Cancel an open order."""
        for asset_id, orders in self._orders_by_asset.items():
            if signal_id in orders:
                orders[signal_id].status = "cancelled"
                self._remove_order(asset_id, signal_id)
                return True
        return False

    def cancel_all_orders(self, asset_id: str | None = None) -> int:
        """Cancel all open orders, optionally for a specific asset."""
        cancelled = 0

        if asset_id is not None:
            # O(1) lookup for specific asset
            if asset_id in self._orders_by_asset:
                cancelled = len(self._orders_by_asset[asset_id])
                del self._orders_by_asset[asset_id]
        else:
            # Cancel all orders across all assets
            for orders in self._orders_by_asset.values():
                cancelled += len(orders)
            self._orders_by_asset.clear()

        return cancelled

    def get_open_orders(self, asset_id: str | None = None) -> list[PaperOrder]:
        """Get open orders, optionally filtered by asset."""
        if asset_id is not None:
            # O(1) lookup for specific asset
            return list(self._orders_by_asset.get(asset_id, {}).values())
        # Return all orders
        result: list[PaperOrder] = []
        for orders in self._orders_by_asset.values():
            result.extend(orders.values())
        return result

    @property
    def open_orders(self) -> dict[int, PaperOrder]:
        """
        Backwards-compatible property returning flat dict of all open orders.

        Deprecated: Use get_open_orders() for better performance.
        """
        result: dict[int, PaperOrder] = {}
        for orders in self._orders_by_asset.values():
            result.update(orders)
        return result

    def get_recent_fills(self, limit: int = 100) -> list[Fill]:
        """Get recent fills."""
        # deque supports slicing but returns a new deque; convert to list for consistency
        if limit >= len(self.fills):
            return list(self.fills)
        return list(self.fills)[-limit:]
