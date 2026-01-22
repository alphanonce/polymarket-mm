"""
Size Models

Implementations of size models for market making.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from strategy.models.base import (
    NormalizationConfig,
    QuoteResult,
    SizeModel,
    SizeResult,
    StrategyState,
)


@dataclass
class FixedSizeConfig:
    """Configuration for fixed size model."""

    base_size: float = 10.0  # Base order size
    max_size: float = 100.0  # Maximum order size
    min_size: float = 5.0  # Minimum order size


class FixedSizeModel(SizeModel):
    """
    Simple fixed size model.

    Quotes fixed sizes on both sides.
    Normalization (rounding, min size enforcement) is handled by the base class.
    """

    def __init__(
        self,
        config: Optional[FixedSizeConfig] = None,
        normalization: Optional[NormalizationConfig] = None,
    ):
        super().__init__(normalization)
        self.config = config or FixedSizeConfig()

    def compute_raw(self, state: StrategyState, quote: QuoteResult) -> SizeResult:
        """Compute raw bid/ask sizes before normalization."""
        if not quote.should_quote:
            return SizeResult(bid_size=0, ask_size=0)

        size = self.config.base_size

        return SizeResult(
            bid_size=size,
            ask_size=size,
            max_position=self.config.max_size,
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
        config: Optional[InventoryBasedSizeConfig] = None,
        normalization: Optional[NormalizationConfig] = None,
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


@dataclass
class ConfidenceBasedSizeConfig:
    """Configuration for confidence-based size model."""

    base_size: float = 10.0
    max_size: float = 100.0
    min_size: float = 5.0
    max_position: float = 500.0
    confidence_exponent: float = 2.0  # How aggressively to scale with confidence


class ConfidenceBasedSizeModel(SizeModel):
    """
    Size model that scales sizes based on quote confidence.

    Higher confidence -> larger sizes.
    Normalization (rounding, min size enforcement) is handled by the base class.
    """

    def __init__(
        self,
        config: Optional[ConfidenceBasedSizeConfig] = None,
        normalization: Optional[NormalizationConfig] = None,
    ):
        super().__init__(normalization)
        self.config = config or ConfidenceBasedSizeConfig()

    def compute_raw(self, state: StrategyState, quote: QuoteResult) -> SizeResult:
        """Compute raw bid/ask sizes before normalization."""
        if not quote.should_quote:
            return SizeResult(bid_size=0, ask_size=0)

        # Scale size by confidence
        confidence_mult = quote.confidence ** self.config.confidence_exponent
        size = self.config.base_size * confidence_mult

        # Apply max constraint
        size = min(self.config.max_size, size)

        # Check position limits
        position = state.current_position
        max_pos = self.config.max_position

        bid_size = min(size, max_pos - position) if position < max_pos else 0
        ask_size = min(size, max_pos + position) if position > -max_pos else 0

        return SizeResult(
            bid_size=max(0, bid_size),
            ask_size=max(0, ask_size),
            max_position=max_pos,
        )
