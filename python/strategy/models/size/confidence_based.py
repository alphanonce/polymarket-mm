"""
Confidence Based Size Model

Size model that scales order sizes based on quote confidence.
"""

from dataclasses import dataclass

from strategy.models.base import (
    NormalizationConfig,
    QuoteResult,
    SizeModel,
    SizeResult,
    StrategyState,
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
        config: ConfidenceBasedSizeConfig | None = None,
        normalization: NormalizationConfig | None = None,
    ):
        super().__init__(normalization)
        self.config = config or ConfidenceBasedSizeConfig()

    def compute_raw(self, state: StrategyState, quote: QuoteResult) -> SizeResult:
        """Compute raw bid/ask sizes before normalization."""
        if not quote.should_quote:
            return SizeResult(bid_size=0, ask_size=0)

        # Scale size by confidence
        confidence_mult = quote.confidence**self.config.confidence_exponent
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
