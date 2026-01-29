"""
Position Tracker

Tracks positions, calculates PnL, and maintains equity curve.
"""

import time
from dataclasses import dataclass
from typing import Any

import structlog

from strategy.shm.types import SIDE_BUY

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
    - If position is long (bought tokens) and market resolves to that outcome: payout = 1.0/token
    - If position is long and market resolves to opposite outcome: payout = 0.0
    - PnL = payout - cost_basis
    """

    def __init__(self, initial_equity: float = 10000.0, strategy_id: str | None = None):
        self.initial_equity = initial_equity
        self.strategy_id = strategy_id
        self.cash = initial_equity
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_history: list[tuple[int, float]] = []

        self.logger = logger.bind(
            component="position_tracker",
            strategy_id=strategy_id,
        )

        # Record initial equity
        self.equity_history.append((time.time_ns(), initial_equity))

    def record_trade(
        self,
        asset_id: str,
        slug: str,
        side: int,
        price: float,
        size: float,
        token_side: str | None = None,
    ) -> dict[str, Any]:
        """
        Record a trade and update position.

        Returns trade dict for storage.
        """
        timestamp_ns = time.time_ns()
        pnl = 0.0

        # Use provided token_side or default to "up"
        # (caller should use TokenIdRegistry for proper lookup)
        effective_token_side = token_side if token_side else "up"

        position = self.positions.get(asset_id)

        if position is None:
            # New position
            position = Position(
                asset_id=asset_id,
                slug=slug,
                side=effective_token_side,
                size=0.0,
                avg_entry_price=0.0,
                updated_at_ns=timestamp_ns,
            )
            self.positions[asset_id] = position

        if side == SIDE_BUY:
            # Buying tokens
            cost = price * size
            self.cash -= cost

            # Calculate realized PnL if closing short position
            if position.size < 0:
                buy_to_close = min(size, abs(position.size))
                pnl = (position.avg_entry_price - price) * buy_to_close
                position.realized_pnl += pnl

            # Update average entry price
            new_size = position.size + size
            if new_size > 0 and position.size <= 0:
                # Transitioning from short/flat to long - set entry price from new buys only
                long_portion = new_size  # portion that is now long
                if position.size < 0:
                    # Some of the buy closed the short, rest opens long
                    long_portion = size - abs(position.size)
                if long_portion > 0:
                    position.avg_entry_price = price
            elif new_size > 0 and position.size > 0:
                # Adding to existing long position - weighted average
                position.avg_entry_price = (
                    position.avg_entry_price * position.size + price * size
                ) / new_size
            # If still short after buy, keep existing avg_entry_price

            position.size = new_size

            # Reset if flat
            if abs(position.size) <= 0.001:
                position.size = 0.0
                position.avg_entry_price = 0.0

        else:  # SIDE_SELL
            # Selling tokens
            revenue = price * size
            self.cash += revenue

            # Calculate realized PnL on the portion sold (only if closing long)
            if position.size > 0:
                sell_from_long = min(size, position.size)
                pnl = (price - position.avg_entry_price) * sell_from_long
                position.realized_pnl += pnl

            new_size = position.size - size

            # Update average entry price for short positions
            if new_size < 0 and position.size >= 0:
                # Transitioning from long/flat to short - set entry price from new sells only
                short_portion = abs(new_size)
                if position.size > 0:
                    # Some of the sell closed the long, rest opens short
                    short_portion = size - position.size
                if short_portion > 0:
                    position.avg_entry_price = price
            elif new_size < 0 and position.size < 0:
                # Adding to existing short position - weighted average
                old_short_size = abs(position.size)
                new_short_size = abs(new_size)
                position.avg_entry_price = (
                    position.avg_entry_price * old_short_size + price * size
                ) / new_short_size
            # If still long after sell, keep existing avg_entry_price

            position.size = new_size

            # Reset avg_entry_price only if position is flat (within tolerance)
            if abs(position.size) <= 0.001:
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

        result = {
            "asset_id": asset_id,
            "slug": slug,
            "side": side,
            "price": price,
            "size": size,
            "pnl": pnl,
            "timestamp": timestamp_ns,
        }
        if self.strategy_id:
            result["strategy_id"] = self.strategy_id
        return result

    def update_unrealized_pnl(self, asset_id: str, current_price: float) -> None:
        """Update unrealized PnL for a position based on current market price."""
        position = self.positions.get(asset_id)
        if position and abs(position.size) > 0.001:
            # For long: profit when price goes up
            # For short: profit when price goes down
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
        if position is None or abs(position.size) <= 0.001:
            return 0.0

        # Binary settlement: 1.0 if position side matches outcome, else 0.0
        settlement_price = 1.0 if position.side == outcome else 0.0

        if position.size > 0:
            # Long position: receive settlement_price per token
            pnl = (settlement_price - position.avg_entry_price) * position.size
            self.cash += settlement_price * position.size
        else:
            # Short position: must buy back at settlement_price
            pnl = (position.avg_entry_price - settlement_price) * abs(position.size)
            self.cash -= settlement_price * abs(position.size)

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

    def get_position(self, asset_id: str) -> dict[str, Any] | None:
        """Get position as dict for storage."""
        position = self.positions.get(asset_id)
        if position is None:
            return None

        result = {
            "asset_id": position.asset_id,
            "slug": position.slug,
            "side": position.side,
            "size": position.size,
            "avg_entry_price": position.avg_entry_price,
            "unrealized_pnl": position.unrealized_pnl,
            "realized_pnl": position.realized_pnl,
        }
        if self.strategy_id:
            result["strategy_id"] = self.strategy_id
        return result

    def get_all_positions(self) -> list[dict[str, Any]]:
        """Get all positions with non-zero size (long or short)."""
        positions = [
            self.get_position(asset_id)
            for asset_id, pos in self.positions.items()
            if abs(pos.size) > 0.001
        ]
        return [p for p in positions if p is not None]

    @property
    def total_equity(self) -> float:
        """Calculate total equity (cash + position value)."""
        position_value = sum(
            pos.size * pos.avg_entry_price + pos.unrealized_pnl
            for pos in self.positions.values()
            if abs(pos.size) > 0.001
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

    def get_equity_history(self, limit: int = 1000) -> list[tuple[int, float]]:
        """Get equity history."""
        return self.equity_history[-limit:]

    def get_trade_history(self, limit: int = 100) -> list[Trade]:
        """Get recent trades."""
        return self.trades[-limit:]

    def get_pnl_by_asset(self) -> dict[str, float]:
        """Calculate PnL grouped by asset."""
        pnl_by_asset: dict[str, float] = {}
        for pos in self.positions.values():
            asset = self._extract_asset(pos.slug)
            pnl = pos.realized_pnl + pos.unrealized_pnl
            pnl_by_asset[asset] = pnl_by_asset.get(asset, 0.0) + pnl
        return pnl_by_asset

    @staticmethod
    def _extract_asset(slug: str) -> str:
        """Extract asset name from slug (e.g., btc-updown-15m -> btc)."""
        if slug:
            parts = slug.lower().split("-")
            if parts:
                return parts[0]
        return "unknown"
