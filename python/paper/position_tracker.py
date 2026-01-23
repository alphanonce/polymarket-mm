"""
Position Tracker

Tracks positions, calculates PnL, and maintains equity curve.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import structlog

from strategy.shm.types import SIDE_BUY, SIDE_SELL

logger = structlog.get_logger()


@dataclass
class Position:
    """Position in a market."""

    asset_id: str
    slug: str
    side: str  # 'up' or 'down' (derived from asset_id)
    size: float
    avg_entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    trade_count: int = 0
    updated_at_ns: int = 0


@dataclass
class Trade:
    """Recorded trade."""

    asset_id: str
    slug: str
    side: int  # SIDE_BUY or SIDE_SELL
    price: float
    size: float
    pnl: float
    timestamp_ns: int


class PositionTracker:
    """
    Tracks positions across markets and calculates PnL.

    Binary option settlement:
    - If position is long (bought tokens) and market resolves to that outcome: payout = 1.0 per token
    - If position is long and market resolves to opposite outcome: payout = 0.0
    - PnL = payout - cost_basis
    """

    def __init__(self, initial_equity: float = 10000.0):
        self.initial_equity = initial_equity
        self.cash = initial_equity
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[tuple[int, float]] = []

        self.logger = logger.bind(component="position_tracker")

        # Record initial equity
        self.equity_history.append((time.time_ns(), initial_equity))

    def record_trade(
        self,
        asset_id: str,
        slug: str,
        side: int,
        price: float,
        size: float,
    ) -> Dict:
        """
        Record a trade and update position.

        Returns trade dict for storage.
        """
        timestamp_ns = time.time_ns()
        pnl = 0.0

        # Derive side from asset_id (e.g., contains 'up' or 'down')
        token_side = "up" if "up" in asset_id.lower() or asset_id.endswith("1") else "down"

        position = self.positions.get(asset_id)

        if position is None:
            # New position
            position = Position(
                asset_id=asset_id,
                slug=slug,
                side=token_side,
                size=0.0,
                avg_entry_price=0.0,
                updated_at_ns=timestamp_ns,
            )
            self.positions[asset_id] = position

        if side == SIDE_BUY:
            # Buying tokens
            cost = price * size
            self.cash -= cost

            # Update average entry price
            total_size = position.size + size
            if total_size > 0:
                position.avg_entry_price = (
                    (position.avg_entry_price * position.size + price * size) / total_size
                )
            position.size = total_size

        else:  # SIDE_SELL
            # Selling tokens
            revenue = price * size
            self.cash += revenue

            # Calculate realized PnL on the portion sold
            if position.size > 0:
                pnl = (price - position.avg_entry_price) * size
                position.realized_pnl += pnl

            position.size -= size

            # Clean up if position is closed
            if position.size <= 0.001:
                position.size = 0.0
                position.avg_entry_price = 0.0

        position.trade_count += 1
        position.updated_at_ns = timestamp_ns

        # Record trade
        trade = Trade(
            asset_id=asset_id,
            slug=slug,
            side=side,
            price=price,
            size=size,
            pnl=pnl,
            timestamp_ns=timestamp_ns,
        )
        self.trades.append(trade)

        self.logger.debug(
            "Trade recorded",
            asset_id=asset_id,
            side="BUY" if side == SIDE_BUY else "SELL",
            price=price,
            size=size,
            pnl=pnl,
            position_size=position.size,
        )

        return {
            "asset_id": asset_id,
            "slug": slug,
            "side": side,
            "price": price,
            "size": size,
            "pnl": pnl,
            "timestamp": timestamp_ns,
        }

    def update_unrealized_pnl(self, asset_id: str, current_price: float) -> None:
        """Update unrealized PnL for a position based on current market price."""
        position = self.positions.get(asset_id)
        if position and position.size > 0:
            position.unrealized_pnl = (current_price - position.avg_entry_price) * position.size

    def settle_position(self, asset_id: str, outcome: str) -> float:
        """
        Settle a position when market resolves.

        Args:
            asset_id: The asset to settle
            outcome: 'up' or 'down' - the winning outcome

        Returns:
            Settlement PnL
        """
        position = self.positions.get(asset_id)
        if position is None or position.size <= 0:
            return 0.0

        # Binary settlement: 1.0 if position side matches outcome, else 0.0
        settlement_price = 1.0 if position.side == outcome else 0.0

        pnl = (settlement_price - position.avg_entry_price) * position.size
        self.cash += settlement_price * position.size

        position.realized_pnl += pnl
        position.size = 0.0
        position.avg_entry_price = 0.0
        position.unrealized_pnl = 0.0

        self.logger.info(
            "Position settled",
            asset_id=asset_id,
            outcome=outcome,
            settlement_price=settlement_price,
            pnl=pnl,
        )

        return pnl

    def get_position(self, asset_id: str) -> Optional[Dict]:
        """Get position as dict for storage."""
        position = self.positions.get(asset_id)
        if position is None:
            return None

        return {
            "asset_id": position.asset_id,
            "slug": position.slug,
            "side": position.side,
            "size": position.size,
            "avg_entry_price": position.avg_entry_price,
            "unrealized_pnl": position.unrealized_pnl,
            "realized_pnl": position.realized_pnl,
        }

    def get_all_positions(self) -> List[Dict]:
        """Get all positions with non-zero size."""
        return [
            self.get_position(asset_id)
            for asset_id, pos in self.positions.items()
            if pos.size > 0
        ]

    @property
    def total_equity(self) -> float:
        """Calculate total equity (cash + position value)."""
        position_value = sum(
            pos.size * pos.avg_entry_price + pos.unrealized_pnl
            for pos in self.positions.values()
            if pos.size > 0
        )
        return self.cash + position_value

    @property
    def total_pnl(self) -> float:
        """Total PnL (realized + unrealized)."""
        realized = sum(pos.realized_pnl for pos in self.positions.values())
        unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        return realized + unrealized

    @property
    def realized_pnl(self) -> float:
        """Total realized PnL."""
        return sum(pos.realized_pnl for pos in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized PnL."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def snapshot_equity(self) -> tuple[int, float]:
        """Take a snapshot of current equity."""
        timestamp_ns = time.time_ns()
        equity = self.total_equity
        self.equity_history.append((timestamp_ns, equity))
        return timestamp_ns, equity

    def get_equity_history(self, limit: int = 1000) -> List[tuple[int, float]]:
        """Get equity history."""
        return self.equity_history[-limit:]

    def get_trade_history(self, limit: int = 100) -> List[Trade]:
        """Get recent trades."""
        return self.trades[-limit:]
