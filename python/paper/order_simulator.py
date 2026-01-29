"""
Order Simulator

Provides various fill simulation modes for paper trading.
"""

import random
from dataclasses import dataclass
from enum import Enum

from strategy.shm.types import SIDE_BUY, SIDE_SELL, MarketState


class FillMode(Enum):
    """Fill simulation modes."""

    IMMEDIATE = "immediate"       # Fill at touch if price crosses
    QUEUE = "queue"               # Simulate queue position
    PROBABILISTIC = "probabilistic"  # Fill with probability based on depth


@dataclass
class FillResult:
    """Result of fill simulation."""

    filled: bool
    fill_price: float = 0.0
    fill_size: float = 0.0
    partial: bool = False
    slippage: float = 0.0


class OrderSimulator:
    """
    Simulates order fills against live orderbook.

    Provides different fill modes for varying levels of realism.
    """

    def __init__(
        self,
        fill_mode: FillMode = FillMode.IMMEDIATE,
        queue_position_ratio: float = 0.5,  # Assumed queue position (0=front, 1=back)
        fill_probability_base: float = 0.8,  # Base fill probability
    ):
        self.fill_mode = fill_mode
        self.queue_position_ratio = queue_position_ratio
        self.fill_probability_base = fill_probability_base

    def simulate_fill(
        self,
        side: int,
        price: float,
        size: float,
        market: MarketState,
    ) -> FillResult:
        """
        Simulate fill based on current mode.

        Args:
            side: SIDE_BUY or SIDE_SELL
            price: Limit price
            size: Order size
            market: Current market state

        Returns:
            FillResult with fill details
        """
        if self.fill_mode == FillMode.IMMEDIATE:
            return self._simulate_immediate(side, price, size, market)
        elif self.fill_mode == FillMode.QUEUE:
            return self._simulate_queue(side, price, size, market)
        else:
            return self._simulate_probabilistic(side, price, size, market)

    def _simulate_immediate(
        self,
        side: int,
        price: float,
        size: float,
        market: MarketState,
    ) -> FillResult:
        """
        Immediate fill at touch if price crosses.

        Most optimistic fill assumption - fills immediately at best price
        if limit price crosses the spread.
        """
        if side == SIDE_BUY:
            if not market.asks:
                return FillResult(filled=False)

            best_ask, ask_size = market.asks[0]
            if price >= best_ask:
                fill_size = min(size, ask_size)
                return FillResult(
                    filled=True,
                    fill_price=best_ask,
                    fill_size=fill_size,
                    partial=fill_size < size,
                    slippage=0.0,
                )
        else:
            if not market.bids:
                return FillResult(filled=False)

            best_bid, bid_size = market.bids[0]
            if price <= best_bid:
                fill_size = min(size, bid_size)
                return FillResult(
                    filled=True,
                    fill_price=best_bid,
                    fill_size=fill_size,
                    partial=fill_size < size,
                    slippage=0.0,
                )

        return FillResult(filled=False)

    def _simulate_queue(
        self,
        side: int,
        price: float,
        size: float,
        market: MarketState,
    ) -> FillResult:
        """
        Queue-based fill simulation.

        Considers queue position - order fills only if sufficient volume
        trades through our position in the queue.
        """
        if side == SIDE_BUY:
            if not market.asks:
                return FillResult(filled=False)

            best_ask, ask_size = market.asks[0]

            # If price crosses, fill immediately (aggressing)
            if price > best_ask:
                fill_size = min(size, ask_size)
                return FillResult(
                    filled=True,
                    fill_price=best_ask,
                    fill_size=fill_size,
                    partial=fill_size < size,
                )

            # If posting at best bid, need to wait in queue
            if price == best_ask:
                # Assume we're at queue_position_ratio of the way through the queue
                queue_ahead = ask_size * self.queue_position_ratio

                # Check if last trade would have filled us
                if (market.last_trade_side == SIDE_BUY and
                    market.last_trade_size > queue_ahead):
                    fill_size = min(size, market.last_trade_size - queue_ahead)
                    return FillResult(
                        filled=True,
                        fill_price=best_ask,
                        fill_size=fill_size,
                        partial=fill_size < size,
                    )
        else:
            if not market.bids:
                return FillResult(filled=False)

            best_bid, bid_size = market.bids[0]

            if price < best_bid:
                fill_size = min(size, bid_size)
                return FillResult(
                    filled=True,
                    fill_price=best_bid,
                    fill_size=fill_size,
                    partial=fill_size < size,
                )

            if price == best_bid:
                queue_ahead = bid_size * self.queue_position_ratio

                if (market.last_trade_side == SIDE_SELL and
                    market.last_trade_size > queue_ahead):
                    fill_size = min(size, market.last_trade_size - queue_ahead)
                    return FillResult(
                        filled=True,
                        fill_price=best_bid,
                        fill_size=fill_size,
                        partial=fill_size < size,
                    )

        return FillResult(filled=False)

    def _simulate_probabilistic(
        self,
        side: int,
        price: float,
        size: float,
        market: MarketState,
    ) -> FillResult:
        """
        Probabilistic fill simulation.

        Fill probability depends on:
        - Price improvement (aggressive pricing increases probability)
        - Order size relative to available liquidity
        """
        if side == SIDE_BUY:
            if not market.asks:
                return FillResult(filled=False)

            best_ask, ask_size = market.asks[0]

            if price < best_ask:
                return FillResult(filled=False)

            # Price improvement bonus
            price_improvement = (price - best_ask) / best_ask if best_ask > 0 else 0

            # Size penalty (larger orders harder to fill)
            size_ratio = min(size / ask_size, 1.0) if ask_size > 0 else 1.0

            fill_prob = self.fill_probability_base + (price_improvement * 0.1) - (size_ratio * 0.2)
            fill_prob = max(0.1, min(0.99, fill_prob))

            if random.random() < fill_prob:
                fill_size = min(size, ask_size)
                return FillResult(
                    filled=True,
                    fill_price=best_ask,
                    fill_size=fill_size,
                    partial=fill_size < size,
                )
        else:
            if not market.bids:
                return FillResult(filled=False)

            best_bid, bid_size = market.bids[0]

            if price > best_bid:
                return FillResult(filled=False)

            price_improvement = (best_bid - price) / best_bid if best_bid > 0 else 0
            size_ratio = min(size / bid_size, 1.0) if bid_size > 0 else 1.0

            fill_prob = self.fill_probability_base + (price_improvement * 0.1) - (size_ratio * 0.2)
            fill_prob = max(0.1, min(0.99, fill_prob))

            if random.random() < fill_prob:
                fill_size = min(size, bid_size)
                return FillResult(
                    filled=True,
                    fill_price=best_bid,
                    fill_size=fill_size,
                    partial=fill_size < size,
                )

        return FillResult(filled=False)

    def estimate_fill_price_with_impact(
        self,
        side: int,
        size: float,
        market: MarketState,
    ) -> tuple[float, float]:
        """
        Estimate fill price including market impact for large orders.

        Returns:
            Tuple of (average_fill_price, total_slippage)
        """
        levels = market.asks if side == SIDE_BUY else market.bids

        if not levels:
            return 0.0, 0.0

        remaining = size
        total_cost = 0.0
        first_price = levels[0][0]

        for price, level_size in levels:
            if remaining <= 0:
                break

            fill_at_level = min(remaining, level_size)
            total_cost += fill_at_level * price
            remaining -= fill_at_level

        if remaining > 0:
            # Not enough liquidity
            return 0.0, float('inf')

        avg_price = total_cost / size
        slippage = abs(avg_price - first_price)

        return avg_price, slippage
