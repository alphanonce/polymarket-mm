"""
Model Factory

Factory functions for creating quote and size models from configuration.
"""

from dashboard.config import QuoteModelConfig, SizeModelConfig
from strategy.models.base import QuoteModel, SizeModel
from strategy.models.quote import (
    InventoryAdjustedQuoteConfig,
    InventoryAdjustedQuoteModel,
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


def create_quote_model(
    config: QuoteModelConfig,
    end_ts_ms: int | None = None,
) -> QuoteModel:
    """
    Create a quote model from configuration.

    Args:
        config: Quote model configuration
        end_ts_ms: Market end timestamp in milliseconds (for real-time T calculation)

    Returns:
        Instantiated quote model
    """
    if config.type in ("zspread", "zscore"):
        return ZSpreadQuoteModel(
            ZSpreadQuoteConfig(
                z=config.z,
                distribution=config.distribution,  # type: ignore[arg-type]
                t_df=config.t_df,
                vol_mode=config.vol_mode,  # type: ignore[arg-type]
                vol_floor=config.vol_floor,
                implied_volatility=config.implied_volatility,
                tau_seconds=config.tau_seconds,
                strike=config.strike,
                end_ts_ms=end_ts_ms,
                price_history_max_age_seconds=config.price_history_max_age_seconds,
                enforce_maker=config.enforce_maker,
                maker_offset_ticks=config.maker_offset_ticks,
                reference_price_symbol=config.reference_price_symbol,
            )
        )
    elif config.type == "tpbs":
        return TpBSQuoteModel(
            TpBSQuoteConfig(
                min_z=config.min_z,
                max_z=config.max_z,
            )
        )
    else:  # inventory_adjusted (default)
        return InventoryAdjustedQuoteModel(
            InventoryAdjustedQuoteConfig(
                base_spread=config.base_spread,
                inventory_skew=config.inventory_skew,
            )
        )


def create_size_model(config: SizeModelConfig) -> SizeModel:
    """
    Create a size model from configuration.

    Args:
        config: Size model configuration

    Returns:
        Instantiated size model
    """
    if config.type == "proportional":
        return ProportionalSizeModel(
            ProportionalSizeConfig(
                order_size_pct=config.order_size_pct,
                max_position_pct=config.max_position_pct,
                min_order_size=config.min_order_size,
                min_order_value=config.min_order_value,
            )
        )
    elif config.type == "confidence_based":
        return ConfidenceBasedSizeModel(
            ConfidenceBasedSizeConfig(
                base_size=config.base_size,
                max_position=config.max_position,
            )
        )
    elif config.type == "fixed":
        return FixedSizeModel(
            FixedSizeConfig(
                base_size=config.base_size,
            )
        )
    else:  # inventory_based (default)
        return InventoryBasedSizeModel(
            InventoryBasedSizeConfig(
                base_size=config.base_size,
                max_position=config.max_position,
            )
        )


def update_quote_model_end_ts(model: QuoteModel, end_ts_ms: int) -> None:
    """
    Update the end_ts_ms on a ZSpreadQuoteModel.

    Args:
        model: Quote model (must be ZSpreadQuoteModel)
        end_ts_ms: Market end timestamp in milliseconds
    """
    if isinstance(model, ZSpreadQuoteModel):
        model.config.end_ts_ms = end_ts_ms
