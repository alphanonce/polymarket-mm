"""
Spread Quote Model

Simple spread-based quote model that quotes a fixed spread around the mid price.
"""

from dataclasses import dataclass

from strategy.models.base import (
    NormalizationConfig,
    QuoteModel,
    QuoteResult,
    StrategyState,
)


@dataclass
class SpreadQuoteConfig:
    """Configuration for spread-based quote model."""

    min_spread: float = 0.01  # Minimum spread (1%)
    max_spread: float = 0.10  # Maximum spread (10%)
    base_spread: float = 0.02  # Base spread (2%)
    volatility_multiplier: float = 1.0  # Spread adjustment for volatility
    min_edge: float = 0.005  # Minimum edge over fair value


class SpreadQuoteModel(QuoteModel):
    """
    Simple spread-based quote model.

    Quotes a fixed spread around the mid price.
    Normalization (rounding, clamping) is handled by the base class.
    """

    def __init__(
        self,
        config: SpreadQuoteConfig | None = None,
        normalization: NormalizationConfig | None = None,
    ):
        super().__init__(normalization)
        self.config = config or SpreadQuoteConfig()

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

        half_spread = self.config.base_spread / 2

        bid_price = mid * (1 - half_spread)
        ask_price = mid * (1 + half_spread)

        # Ensure minimum spread (raw, before normalization)
        if ask_price - bid_price < self.config.min_spread:
            adjustment = (self.config.min_spread - (ask_price - bid_price)) / 2
            bid_price -= adjustment
            ask_price += adjustment

        return QuoteResult(
            bid_price=bid_price,
            ask_price=ask_price,
            confidence=1.0,
        )
