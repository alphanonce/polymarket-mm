"""
Inventory Adjusted Quote Model

Quote model that adjusts prices based on inventory and external reference prices.
"""

from dataclasses import dataclass

import numpy as np

from strategy.models.base import (
    NormalizationConfig,
    QuoteModel,
    QuoteResult,
    StrategyState,
)


@dataclass
class InventoryAdjustedQuoteConfig:
    """Configuration for inventory-adjusted quote model."""

    base_spread: float = 0.02  # Base spread (2%)
    min_spread: float = 0.01  # Minimum spread
    max_spread: float = 0.15  # Maximum spread
    inventory_skew: float = 0.5  # How much to skew based on inventory
    max_inventory: float = 1000.0  # Max inventory for full skew
    reference_price_symbol: str | None = None  # External price to use as reference
    reference_price_weight: float = 0.0  # Weight of external price (0-1)


class InventoryAdjustedQuoteModel(QuoteModel):
    """
    Quote model that adjusts prices based on inventory.

    Skews quotes away from current position to reduce inventory risk.
    Also optionally incorporates external reference prices.
    Normalization (rounding, clamping) is handled by the base class.
    """

    def __init__(
        self,
        config: InventoryAdjustedQuoteConfig | None = None,
        normalization: NormalizationConfig | None = None,
    ):
        super().__init__(normalization)
        self.config = config or InventoryAdjustedQuoteConfig()
        self._ewma_spread: float = self.config.base_spread

    def compute_raw(self, state: StrategyState) -> QuoteResult:
        """Compute raw bid/ask prices before normalization."""
        mid = state.mid_price

        if mid <= 0:
            return QuoteResult(
                bid_price=0,
                ask_price=0,
                should_quote=False,
                reason="Invalid mid price",
            )

        # Adjust mid price with external reference if configured
        fair_value = self._compute_fair_value(state, mid)

        # Compute base spread (could incorporate volatility)
        spread = self._compute_spread(state)

        # Compute inventory skew
        skew = self._compute_inventory_skew(state)

        # Apply skew to quotes
        half_spread = spread / 2
        bid_price = fair_value - half_spread + skew
        ask_price = fair_value + half_spread + skew

        # Ensure bid < ask (pre-normalization)
        if bid_price >= ask_price:
            mid_adj = (bid_price + ask_price) / 2
            bid_price = mid_adj - self.config.min_spread / 2
            ask_price = mid_adj + self.config.min_spread / 2

        return QuoteResult(
            bid_price=bid_price,
            ask_price=ask_price,
            confidence=self._compute_confidence(state),
        )

    def _compute_fair_value(self, state: StrategyState, market_mid: float) -> float:
        """Compute fair value, optionally using external reference."""
        if self.config.reference_price_symbol and self.config.reference_price_weight > 0:
            ext_price = state.get_external_price(self.config.reference_price_symbol)
            if ext_price is not None:
                # Blend market mid with external reference
                weight = self.config.reference_price_weight
                return market_mid * (1 - weight) + ext_price * weight

        return market_mid

    def _compute_spread(self, state: StrategyState) -> float:
        """Compute the spread to quote."""
        # Start with base spread
        spread = self.config.base_spread

        # Could add volatility adjustment here
        # spread *= (1 + volatility_factor)

        # Clamp
        return max(self.config.min_spread, min(self.config.max_spread, spread))

    def _compute_inventory_skew(self, state: StrategyState) -> float:
        """
        Compute price skew based on inventory.

        Positive position -> negative skew (lower prices to sell)
        Negative position -> positive skew (higher prices to buy)
        """
        position = state.current_position
        if abs(position) < 1e-8:
            return 0.0

        # Normalize position to [-1, 1]
        normalized: float = float(np.clip(position / self.config.max_inventory, -1, 1))

        # Skew is opposite to position direction
        skew: float = -normalized * self.config.inventory_skew * state.mid_price

        return skew

    def _compute_confidence(self, state: StrategyState) -> float:
        """Compute confidence in the quote."""
        # Lower confidence when inventory is high
        position = abs(state.current_position)
        inv_ratio = min(1.0, position / self.config.max_inventory)

        # Lower confidence when spread is wide
        spread_ratio = (
            state.spread / self.config.base_spread if self.config.base_spread > 0 else 1.0
        )

        confidence = 1.0
        confidence *= 1.0 - 0.3 * inv_ratio  # Up to 30% reduction from inventory
        confidence *= 1.0 - 0.2 * min(1.0, spread_ratio - 1.0)  # Up to 20% from wide spread

        return max(0.1, confidence)
