"""
Fixed Size Model

Simple fixed size model that quotes the same size on both sides.
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
        config: FixedSizeConfig | None = None,
        normalization: NormalizationConfig | None = None,
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
