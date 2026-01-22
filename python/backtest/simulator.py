"""
Market Simulator

Simulates order matching and fills for backtesting.
Supports both orderbook-based and trades-based fill logic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backtest.data_loader import MarketData, OrderBook, Trade


@dataclass
class Order:
    """Open order."""

    order_id: int
    side: int  # 1 = buy, -1 = sell
    price: float
    size: float
    filled_size: float = 0.0
    created_at_ns: int = 0

    @property
    def remaining_size(self) -> float:
        """Get remaining unfilled size."""
        return self.size - self.filled_size

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.filled_size >= self.size


@dataclass
class Fill:
    """Order fill event."""

    order_id: int
    side: int
    price: float
    size: float
    timestamp_ns: int
    is_maker: bool = True


@dataclass
class Position:
    """Current position."""

    size: float = 0.0
    avg_entry_price: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0


class MarketSimulator:
    """
    Simulates a market for backtesting.

    Supports limit orders with maker/taker fill simulation.
    """

    def __init__(
        self,
        maker_fee: float = 0.0,
        taker_fee: float = 0.0,  # Kept for backwards compatibility, not used
    ):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee  # Deprecated: post-only orders only

        self._order_id_counter: int = 0
        self._orders: Dict[int, Order] = {}
        self._position = Position()
        self._fills: List[Fill] = []

    def reset(self) -> None:
        """Reset simulator state."""
        self._order_id_counter = 0
        self._orders = {}
        self._position = Position()
        self._fills = []

    @property
    def position(self) -> Position:
        """Get current position."""
        return self._position

    @property
    def fills(self) -> List[Fill]:
        """Get all fills."""
        return self._fills.copy()

    @property
    def open_orders(self) -> List[Order]:
        """Get all open orders."""
        return list(self._orders.values())

    def place_limit_order(
        self,
        side: int,
        price: float,
        size: float,
        timestamp_ns: int = 0,
    ) -> int:
        """
        Place a limit order.

        Args:
            side: 1 for buy, -1 for sell
            price: Limit price
            size: Order size
            timestamp_ns: Order timestamp

        Returns:
            Order ID
        """
        self._order_id_counter += 1
        order_id = self._order_id_counter

        order = Order(
            order_id=order_id,
            side=side,
            price=price,
            size=size,
            created_at_ns=timestamp_ns,
        )

        self._orders[order_id] = order
        return order_id

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order to cancel

        Returns:
            True if order was found and cancelled
        """
        if order_id in self._orders:
            del self._orders[order_id]
            return True
        return False

    def cancel_all(self) -> int:
        """
        Cancel all orders.

        Returns:
            Number of orders cancelled
        """
        count = len(self._orders)
        self._orders = {}
        return count

    def process_tick(self, data: MarketData) -> List[Fill]:
        """
        Process a market tick and check for fills (orderbook-based).

        Args:
            data: Market data for this tick

        Returns:
            List of fills that occurred
        """
        tick_fills: List[Fill] = []
        orders_to_remove: List[int] = []

        for order_id, order in self._orders.items():
            fill = self._check_fill(order, data.orderbook, data.timestamp_ns)
            if fill:
                tick_fills.append(fill)
                self._fills.append(fill)
                self._update_position(fill)

                if order.is_filled:
                    orders_to_remove.append(order_id)

        for order_id in orders_to_remove:
            del self._orders[order_id]

        return tick_fills

    def process_trade(self, trade: Trade) -> List[Fill]:
        """
        Process a trade and check for fills using crossing logic.

        Fill Logic (strictly less/greater, NOT equal):
            - trade_price < our_bid → bid fills (we buy)
            - trade_price > our_ask → ask fills (we sell)

        Args:
            trade: Trade event to process

        Returns:
            List of fills that occurred
        """
        tick_fills: List[Fill] = []
        orders_to_remove: List[int] = []

        for order_id, order in self._orders.items():
            fill = self._check_fill_by_trade(order, trade)
            if fill:
                tick_fills.append(fill)
                self._fills.append(fill)
                self._update_position(fill)

                if order.is_filled:
                    orders_to_remove.append(order_id)

        for order_id in orders_to_remove:
            del self._orders[order_id]

        return tick_fills

    def _check_fill_by_trade(
        self,
        order: Order,
        trade: Trade,
    ) -> Optional[Fill]:
        """
        Check if an order should be filled based on a trade.

        Uses crossing logic:
            - trade_price < bid_price → bid fills (we buy)
            - trade_price > ask_price → ask fills (we sell)
        """
        if order.side == 1:  # Buy order (bid)
            # trade_price < our_bid → bid fills
            if trade.price < order.price:
                fill_size = min(order.remaining_size, trade.size)
                order.filled_size += fill_size

                return Fill(
                    order_id=order.order_id,
                    side=order.side,
                    price=order.price,  # Fill at our limit price
                    size=fill_size,
                    timestamp_ns=trade.timestamp_ns,
                    is_maker=True,
                )

        else:  # Sell order (ask)
            # trade_price > our_ask → ask fills
            if trade.price > order.price:
                fill_size = min(order.remaining_size, trade.size)
                order.filled_size += fill_size

                return Fill(
                    order_id=order.order_id,
                    side=order.side,
                    price=order.price,  # Fill at our limit price
                    size=fill_size,
                    timestamp_ns=trade.timestamp_ns,
                    is_maker=True,
                )

        return None

    def _check_fill(
        self,
        order: Order,
        book: OrderBook,
        timestamp_ns: int,
    ) -> Optional[Fill]:
        """
        Check if an order should be filled (orderbook-based, maker-only).

        Fill logic (strictly crossing, NOT equal):
            - best_ask < our_bid -> bid fills (we buy)
            - best_bid > our_ask -> ask fills (we sell)
        """
        if order.side == 1:  # Buy order (bid)
            # best_ask < our_bid -> bid fills
            if book.asks and book.asks[0].price < order.price:
                fill_size = min(order.remaining_size, book.asks[0].size)
                order.filled_size += fill_size

                return Fill(
                    order_id=order.order_id,
                    side=order.side,
                    price=order.price,  # Fill at our limit price (maker)
                    size=fill_size,
                    timestamp_ns=timestamp_ns,
                    is_maker=True,
                )

        else:  # Sell order (ask)
            # best_bid > our_ask -> ask fills
            if book.bids and book.bids[0].price > order.price:
                fill_size = min(order.remaining_size, book.bids[0].size)
                order.filled_size += fill_size

                return Fill(
                    order_id=order.order_id,
                    side=order.side,
                    price=order.price,  # Fill at our limit price (maker)
                    size=fill_size,
                    timestamp_ns=timestamp_ns,
                    is_maker=True,
                )

        return None

    def _update_position(self, fill: Fill) -> None:
        """Update position after a fill."""
        # Calculate fee (always maker fee for post-only orders)
        fee = fill.price * fill.size * self.maker_fee

        self._position.total_fees += fee

        old_size = self._position.size
        fill_value = fill.price * fill.size

        if fill.side == 1:  # Buy
            # Update average entry price
            if old_size >= 0:
                # Adding to long or opening long
                total_cost = self._position.avg_entry_price * old_size + fill_value
                new_size = old_size + fill.size
                self._position.avg_entry_price = total_cost / new_size if new_size > 0 else 0
            else:
                # Covering short
                covered = min(-old_size, fill.size)
                pnl = (self._position.avg_entry_price - fill.price) * covered
                self._position.realized_pnl += pnl - fee

            self._position.size += fill.size

            # Handle position flip: if we were short and now long, set new entry price
            if old_size < 0 and self._position.size > 0:
                self._position.avg_entry_price = fill.price

        else:  # Sell
            if old_size <= 0:
                # Adding to short or opening short
                total_cost = self._position.avg_entry_price * (-old_size) + fill_value
                new_size = -old_size + fill.size
                self._position.avg_entry_price = total_cost / new_size if new_size > 0 else 0
            else:
                # Closing long
                closed = min(old_size, fill.size)
                pnl = (fill.price - self._position.avg_entry_price) * closed
                self._position.realized_pnl += pnl - fee

            self._position.size -= fill.size

            # Handle position flip: if we were long and now short, set new entry price
            if old_size > 0 and self._position.size < 0:
                self._position.avg_entry_price = fill.price

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL at current price."""
        if self._position.size > 0:
            return (current_price - self._position.avg_entry_price) * self._position.size
        elif self._position.size < 0:
            return (self._position.avg_entry_price - current_price) * (-self._position.size)
        return 0.0

    def get_total_pnl(self, current_price: float) -> float:
        """Calculate total PnL (realized + unrealized)."""
        return self._position.realized_pnl + self.get_unrealized_pnl(current_price)
