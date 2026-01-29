"""
Position Tracker

Tracks positions, calculates PnL, and maintains equity curve.
"""

import time
from collections import deque
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
        # Bounded deques to prevent unbounded memory growth
        self.trades: deque[Trade] = deque(maxlen=10000)
        self.equity_history: deque[tuple[int, float]] = deque(maxlen=10000)
        # Per-asset equity history for independent asset tracking (smaller per-asset limit)
        self.equity_history_by_asset: dict[str, deque[tuple[int, float]]] = {}

        # Track active positions (non-zero size) for O(1) iteration
        self._active_positions: set[str] = set()

        self.logger = logger.bind(
            component="position_tracker",
            strategy_id=strategy_id,
        )

        # Cached metrics (invalidated on state changes)
        self._cached_equity: float | None = None
        self._cached_total_pnl: float | None = None
        self._cached_realized_pnl: float | None = None
        self._cached_unrealized_pnl: float | None = None

        # Record initial equity
        self.equity_history.append((time.time_ns(), initial_equity))

    def _invalidate_cache(self) -> None:
        """Invalidate cached metrics. Call this after any state change."""
        self._cached_equity = None
        self._cached_total_pnl = None
        self._cached_realized_pnl = None
        self._cached_unrealized_pnl = None

    def record_trade(
        self,
        asset_id: str,
        slug: str,
        side: int,
        price: float,
        size: float,
        token_side: str | None = None,
        timestamp_ns: int | None = None,
    ) -> dict[str, Any]:
        """
        Record a trade and update position.

        Args:
            asset_id: The asset identifier
            slug: Market slug
            side: SIDE_BUY or SIDE_SELL
            price: Trade price
            size: Trade size
            token_side: 'up' or 'down' token type
            timestamp_ns: Optional timestamp (uses current time if not provided)

        Returns trade dict for storage.
        """
        timestamp_ns = timestamp_ns or time.time_ns()
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

            # Reset if flat and update active positions set
            if abs(position.size) <= 0.001:
                position.size = 0.0
                position.avg_entry_price = 0.0
                self._active_positions.discard(asset_id)
            else:
                self._active_positions.add(asset_id)

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
            # and update active positions set
            if abs(position.size) <= 0.001:
                position.size = 0.0
                position.avg_entry_price = 0.0
                self._active_positions.discard(asset_id)
            else:
                self._active_positions.add(asset_id)

        position.trade_count += 1
        position.updated_at_ns = timestamp_ns

        # Invalidate cached metrics since position changed
        self._invalidate_cache()

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
            new_pnl = (current_price - position.avg_entry_price) * position.size
            if new_pnl != position.unrealized_pnl:
                position.unrealized_pnl = new_pnl
                self._invalidate_cache()

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

        # Invalidate cached metrics since position changed
        self._invalidate_cache()

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
            for asset_id in self._active_positions
        ]
        return [p for p in positions if p is not None]

    @property
    def active_position_ids(self) -> set[str]:
        """Get set of asset IDs with active (non-zero) positions."""
        return self._active_positions

    @property
    def total_equity(self) -> float:
        """Calculate total equity (cash + position value). Cached."""
        if self._cached_equity is not None:
            return self._cached_equity
        position_value = sum(
            pos.size * pos.avg_entry_price + pos.unrealized_pnl
            for pos in self.positions.values()
            if abs(pos.size) > 0.001
        )
        self._cached_equity = self.cash + position_value
        return self._cached_equity

    @property
    def total_pnl(self) -> float:
        """Total PnL (realized + unrealized). Cached."""
        if self._cached_total_pnl is not None:
            return self._cached_total_pnl
        self._cached_total_pnl = self.realized_pnl + self.unrealized_pnl
        return self._cached_total_pnl

    @property
    def realized_pnl(self) -> float:
        """Total realized PnL. Cached."""
        if self._cached_realized_pnl is not None:
            return self._cached_realized_pnl
        self._cached_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        return self._cached_realized_pnl

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized PnL. Cached."""
        if self._cached_unrealized_pnl is not None:
            return self._cached_unrealized_pnl
        self._cached_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self._cached_unrealized_pnl

    def snapshot_equity(self) -> tuple[int, float]:
        """Take a snapshot of current equity (total and per-asset)."""
        timestamp_ns = time.time_ns()
        equity = self.total_equity
        self.equity_history.append((timestamp_ns, equity))

        # Also snapshot per-asset equity (initial_equity + pnl)
        # Each asset has independent capital allocation of initial_equity
        pnl_by_asset = self.get_pnl_by_asset()

        for asset, pnl in pnl_by_asset.items():
            if asset not in self.equity_history_by_asset:
                # Smaller limit per-asset to bound total memory
                self.equity_history_by_asset[asset] = deque(maxlen=1000)
            # Each asset has full initial_equity as base capital
            asset_equity = self.initial_equity + pnl
            self.equity_history_by_asset[asset].append((timestamp_ns, asset_equity))

        return timestamp_ns, equity

    def get_equity_history(self, limit: int = 1000) -> list[tuple[int, float]]:
        """Get equity history."""
        # Convert deque to list for slicing
        if limit >= len(self.equity_history):
            return list(self.equity_history)
        return list(self.equity_history)[-limit:]

    def get_equity_history_by_asset(self, limit: int = 1000) -> dict[str, list[tuple[int, float]]]:
        """Get per-asset equity history."""
        result: dict[str, list[tuple[int, float]]] = {}
        for asset, history in self.equity_history_by_asset.items():
            if limit >= len(history):
                result[asset] = list(history)
            else:
                result[asset] = list(history)[-limit:]
        return result

    def get_trade_history(self, limit: int = 100) -> list[Trade]:
        """Get recent trades."""
        # Convert deque to list for slicing
        if limit >= len(self.trades):
            return list(self.trades)
        return list(self.trades)[-limit:]

    def get_pnl_by_asset(self) -> dict[str, float]:
        """Calculate PnL grouped by asset."""
        pnl_by_asset: dict[str, float] = {}
        for pos in self.positions.values():
            asset = self._extract_asset(pos.slug)
            pnl = pos.realized_pnl + pos.unrealized_pnl
            pnl_by_asset[asset] = pnl_by_asset.get(asset, 0.0) + pnl
        return pnl_by_asset

    def get_equity_for_asset(self, asset: str) -> float:
        """
        Get equity for a specific asset.

        Each asset has its own independent capital allocation:
        - Base capital = initial_equity (e.g., $10,000 per asset)
        - Asset equity = initial_equity + asset_pnl

        This allows independent sizing per asset.
        """
        pnl_by_asset = self.get_pnl_by_asset()
        asset_pnl = pnl_by_asset.get(asset.lower(), 0.0)
        return self.initial_equity + asset_pnl

    def get_cash_for_asset(self, asset: str) -> float:
        """
        Get available cash for a specific asset.

        Approximates available margin as initial_equity minus position value for this asset.
        """
        # Calculate position value for this asset
        position_value = 0.0
        for pos in self.positions.values():
            if self._extract_asset(pos.slug) == asset.lower():
                if pos.size > 0:
                    position_value += pos.size * pos.avg_entry_price
        return self.initial_equity - position_value

    @staticmethod
    def _extract_asset(slug: str) -> str:
        """Extract asset name from slug (e.g., btc-updown-15m -> btc)."""
        if slug:
            parts = slug.lower().split("-")
            if parts:
                return parts[0]
        return "unknown"
