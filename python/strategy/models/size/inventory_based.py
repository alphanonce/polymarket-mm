"""
Inventory Based Size Model

Size model that adjusts order sizes based on current inventory position.
"""

from dataclasses import dataclass

import numpy as np

from strategy.models.base import (
    NormalizationConfig,
    QuoteResult,
    SizeModel,
    SizeResult,
    StrategyState,
)


@dataclass
class InventoryBasedSizeConfig:
    """Configuration for inventory-based size model."""

    base_size: float = 10.0  # Base order size
    max_size: float = 100.0  # Maximum order size
    min_size: float = 5.0  # Minimum order size
    max_position: float = 500.0  # Maximum position
    size_reduction_rate: float = 0.5  # How fast to reduce size near max position
    asymmetric_scaling: bool = True  # Scale sizes asymmetrically based on position


class InventoryBasedSizeModel(SizeModel):
    """
    Size model that adjusts sizes based on inventory.

    Reduces size on the side that would increase inventory.
    Increases size on the side that would reduce inventory.
    Normalization (rounding, min size enforcement) is handled by the base class.
    """

    def __init__(
        self,
        config: InventoryBasedSizeConfig | None = None,
        normalization: NormalizationConfig | None = None,
    ):
        super().__init__(normalization)
        self.config = config or InventoryBasedSizeConfig()

    def compute_raw(self, state: StrategyState, quote: QuoteResult) -> SizeResult:
        """Compute raw bid/ask sizes before normalization."""
        if not quote.should_quote:
            return SizeResult(bid_size=0, ask_size=0)

        position = state.current_position
        max_pos = self.config.max_position

        # Base sizes
        bid_size = self.config.base_size
        ask_size = self.config.base_size

        if self.config.asymmetric_scaling:
            # Compute inventory ratio [-1, 1]
            inv_ratio = np.clip(position / max_pos, -1, 1)

            # Adjust bid size (buying)
            # Positive position -> reduce bid size
            # Negative position -> increase bid size
            bid_mult = 1.0 - inv_ratio * self.config.size_reduction_rate
            bid_size *= max(0.1, bid_mult)

            # Adjust ask size (selling)
            # Positive position -> increase ask size
            # Negative position -> reduce ask size
            ask_mult = 1.0 + inv_ratio * self.config.size_reduction_rate
            ask_size *= max(0.1, ask_mult)

        # Check position limits
        remaining_long = max_pos - position
        remaining_short = max_pos + position

        if remaining_long <= 0:
            bid_size = 0  # Can't buy more
        else:
            bid_size = min(bid_size, remaining_long)

        if remaining_short <= 0:
            ask_size = 0  # Can't sell more
        else:
            ask_size = min(ask_size, remaining_short)

        # Apply max constraint (min is handled by normalization)
        bid_size = min(self.config.max_size, bid_size)
        ask_size = min(self.config.max_size, ask_size)

        return SizeResult(
            bid_size=bid_size,
            ask_size=ask_size,
            max_position=max_pos,
        )
