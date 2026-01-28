"""
Strategy Models

Contains quote and size models for the market-making strategy.
"""

from strategy.models.base import (
    NormalizationConfig,
    QuoteModel,
    QuoteResult,
    SizeModel,
    SizeResult,
    StrategyState,
)
from strategy.models.quote import (
    InventoryAdjustedQuoteConfig,
    InventoryAdjustedQuoteModel,
    SpreadQuoteConfig,
    SpreadQuoteModel,
    TpBSQuoteConfig,
    TpBSQuoteModel,
    ZSpreadQuoteConfig,
    ZSpreadQuoteModel,
)
from strategy.models.size import (
    ConfidenceBasedSizeConfig,
    ConfidenceBasedSizeModel,
    FixedSizeConfig,
    FixedSizeModel,
    InventoryBasedSizeConfig,
    InventoryBasedSizeModel,
    ProportionalSizeConfig,
    ProportionalSizeModel,
)

__all__ = [
    # Base classes and configs
    "QuoteModel",
    "SizeModel",
    "QuoteResult",
    "SizeResult",
    "StrategyState",
    "NormalizationConfig",
    # Quote models
    "SpreadQuoteModel",
    "SpreadQuoteConfig",
    "InventoryAdjustedQuoteModel",
    "InventoryAdjustedQuoteConfig",
    "TpBSQuoteModel",
    "TpBSQuoteConfig",
    "ZSpreadQuoteModel",
    "ZSpreadQuoteConfig",
    # Size models
    "FixedSizeModel",
    "FixedSizeConfig",
    "InventoryBasedSizeModel",
    "InventoryBasedSizeConfig",
    "ConfidenceBasedSizeModel",
    "ConfidenceBasedSizeConfig",
    "ProportionalSizeModel",
    "ProportionalSizeConfig",
]
