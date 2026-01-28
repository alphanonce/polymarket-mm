"""
Proportional Size Model

Position sizing model that calculates order sizes as a fixed percentage of bankroll.
Supports period-based bankroll management to avoid mid-period fluctuations.
"""

import math
from dataclasses import dataclass

from strategy.models.base import (
    NormalizationConfig,
    QuoteResult,
    SizeModel,
    SizeResult,
    StrategyState,
)


def floor_to_token_size(value: float, precision: int = 2) -> float:
    """Floor value to token precision (default 2 decimal places)."""
    multiplier: int = 10**precision
    floored: int = math.floor(value * multiplier)
    return floored / multiplier


@dataclass
class ProportionalSizeConfig:
    """Configuration for proportional size model."""

    order_size_pct: float = 0.05  # 5% of bankroll per order
    max_position_pct: float = 0.20  # 20% max position
    min_order_size: float = 5.0  # Minimum tokens per order
    min_order_value: float = 1.0  # Minimum $1 order value


class ProportionalSizeModel(SizeModel):
    """
    Proportional size model based on bankroll percentage.

    Calculates order sizes as a fixed percentage of bankroll (in tokens, not price-adjusted).
    Uses period-based bankroll management to avoid mid-period fluctuations.

    Key features:
    - Order size = bankroll * order_size_pct
    - Max position = bankroll * max_position_pct
    - Captures bankroll at period start to avoid mid-period volatility
    - Enforces minimum order size and value thresholds
    - Inventory-aware: reduces size when approaching position limits

    Normalization (rounding, min size enforcement) is handled by the base class.
    """

    def __init__(
        self,
        config: ProportionalSizeConfig | None = None,
        normalization: NormalizationConfig | None = None,
    ):
        super().__init__(normalization)
        self.config = config or ProportionalSizeConfig()
        self._period_bankroll: float | None = None

    def copy(self) -> "ProportionalSizeModel":
        """
        Create a deep copy of this model.

        Thread-safe: the returned copy is independent of the original.
        Required for parallel backtesting with ProcessPoolExecutor.

        Returns:
            A new ProportionalSizeModel with copied state
        """
        new = ProportionalSizeModel(config=self.config, normalization=self._norm_config)
        new._period_bankroll = self._period_bankroll
        return new

    def reset(self) -> None:
        """Reset model to initial state by clearing period bankroll."""
        self._period_bankroll = None

    def reset_period(self) -> None:
        """
        Reset period bankroll capture.

        Call this at the start of each trading period to capture a fresh bankroll.
        The bankroll will be captured on the next compute() call.
        """
        self._period_bankroll = None

    def _get_effective_bankroll(self, state: StrategyState) -> float:
        """
        Get effective bankroll for sizing calculations.

        Captures bankroll on first call of each period and reuses it
        to avoid mid-period fluctuations affecting order sizes.

        Args:
            state: Current strategy state

        Returns:
            Effective bankroll (total equity)
        """
        if self._period_bankroll is None:
            self._period_bankroll = state.total_equity
        return self._period_bankroll

    def compute_raw(self, state: StrategyState, quote: QuoteResult) -> SizeResult:
        """
        Compute raw bid/ask sizes before normalization.

        Calculates sizes based on bankroll percentage with position limits
        and viability checks for minimum order requirements.

        Args:
            state: Current strategy state
            quote: Quote result from quote model

        Returns:
            SizeResult with raw bid/ask sizes
        """
        if not quote.should_quote:
            return SizeResult(bid_size=0, ask_size=0)

        # Get effective bankroll (captured at period start)
        bankroll = self._get_effective_bankroll(state)

        if bankroll <= 0:
            return SizeResult(bid_size=0, ask_size=0)

        # Calculate max position (in tokens)
        max_position = bankroll * self.config.max_position_pct
        max_position = max(max_position, self.config.min_order_size)  # Floor to allow trading

        # Calculate base order size
        order_size = bankroll * self.config.order_size_pct
        order_size = max(order_size, self.config.min_order_size)

        # Floor to token precision
        order_size = floor_to_token_size(order_size)
        max_position = floor_to_token_size(max_position)

        # Get current position
        position = state.current_position

        # Calculate remaining capacity for each side
        remaining_long = max(0, max_position - position)
        remaining_short = max(0, max_position + position)

        # Calculate bid size (buying)
        bid_size = min(order_size, remaining_long)
        bid_size = floor_to_token_size(bid_size)

        # Calculate ask size (selling)
        ask_size = min(order_size, remaining_short)
        ask_size = floor_to_token_size(ask_size)

        # Viability checks
        bid_price = quote.bid_price
        ask_price = quote.ask_price

        # Check bid viability
        if bid_size < self.config.min_order_size:
            bid_size = 0
        elif bid_price > 0 and bid_size * bid_price < self.config.min_order_value:
            bid_size = 0
        elif bid_price < 0.01:  # Price bounds check
            bid_size = 0

        # Check ask viability
        if ask_size < self.config.min_order_size:
            ask_size = 0
        elif ask_price > 0 and ask_size * (1 - ask_price) < self.config.min_order_value:
            # For asks, value is based on DOWN price (1 - ask_price)
            ask_size = 0
        elif ask_price > 0.99:  # Price bounds check
            ask_size = 0

        return SizeResult(
            bid_size=bid_size,
            ask_size=ask_size,
            max_position=max_position,
        )
