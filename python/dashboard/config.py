"""
Dashboard Configuration

Dataclasses for dashboard and strategy configuration.
"""

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class QuoteModelConfig:
    """Quote model configuration."""

    type: str = "inventory_adjusted"

    # InventoryAdjusted parameters
    base_spread: float = 0.04
    inventory_skew: float = 0.3

    # TpBS parameters
    min_z: float = 0.5
    max_z: float = 2.0

    # ZSpread parameters
    z: float = 1.0
    distribution: str = "t"
    t_df: float = 3.0
    vol_mode: str = "rv"
    vol_floor: float = 0.10
    implied_volatility: float = 0.5
    tau_seconds: float = 0.1
    strike: float = 0.5
    price_history_max_age_seconds: float = 300.0
    enforce_maker: bool = True
    maker_offset_ticks: int = 1
    reference_price_symbol: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuoteModelConfig":
        return cls(
            type=data.get("type", "inventory_adjusted"),
            # InventoryAdjusted
            base_spread=data.get("base_spread", 0.04),
            inventory_skew=data.get("inventory_skew", 0.3),
            # TpBS
            min_z=data.get("min_z", 0.5),
            max_z=data.get("max_z", 2.0),
            # ZSpread
            z=data.get("z", 1.0),
            distribution=data.get("distribution", "t"),
            t_df=data.get("t_df", 3.0),
            vol_mode=data.get("vol_mode", "rv"),
            vol_floor=data.get("vol_floor", 0.10),
            implied_volatility=data.get("implied_volatility", 0.5),
            tau_seconds=data.get("tau_seconds", 0.1),
            strike=data.get("strike", 0.5),
            price_history_max_age_seconds=data.get("price_history_max_age_seconds", 300.0),
            enforce_maker=data.get("enforce_maker", True),
            maker_offset_ticks=data.get("maker_offset_ticks", 1),
            reference_price_symbol=data.get("reference_price_symbol"),
        )


@dataclass
class SizeModelConfig:
    """Size model configuration."""

    type: str = "inventory_based"

    # InventoryBased / ConfidenceBased parameters
    base_size: float = 10.0
    max_position: float = 100.0

    # Proportional parameters
    order_size_pct: float = 0.05
    max_position_pct: float = 0.20
    min_order_size: float = 5.0
    min_order_value: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SizeModelConfig":
        return cls(
            type=data.get("type", "inventory_based"),
            # InventoryBased / ConfidenceBased
            base_size=data.get("base_size", 10.0),
            max_position=data.get("max_position", 100.0),
            # Proportional
            order_size_pct=data.get("order_size_pct", 0.05),
            max_position_pct=data.get("max_position_pct", 0.20),
            min_order_size=data.get("min_order_size", 5.0),
            min_order_value=data.get("min_order_value", 1.0),
        )


@dataclass
class StrategyConfig:
    """Configuration for a single strategy."""

    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Default Strategy"
    assets: list[str] = field(default_factory=lambda: ["btc"])
    timeframe: str = "15m"
    auto_subscribe: bool = True
    starting_capital: float = 10000.0
    quote_model: QuoteModelConfig = field(default_factory=QuoteModelConfig)
    size_model: SizeModelConfig = field(default_factory=SizeModelConfig)
    max_position_per_market: float = 100.0
    max_total_exposure: float = 1000.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyConfig":
        return cls(
            strategy_id=data.get("strategy_id", str(uuid.uuid4())),
            name=data.get("name", "Default Strategy"),
            assets=data.get("assets", ["btc"]),
            timeframe=data.get("timeframe", "15m"),
            auto_subscribe=data.get("auto_subscribe", True),
            starting_capital=data.get("starting_capital", 10000.0),
            quote_model=QuoteModelConfig.from_dict(data.get("quote_model", {})),
            size_model=SizeModelConfig.from_dict(data.get("size_model", {})),
            max_position_per_market=data.get("max_position_per_market", 100.0),
            max_total_exposure=data.get("max_total_exposure", 1000.0),
        )

    def to_dict(self) -> dict[str, Any]:
        quote_dict: dict[str, Any] = {"type": self.quote_model.type}
        if self.quote_model.type == "inventory_adjusted":
            quote_dict.update(
                {
                    "base_spread": self.quote_model.base_spread,
                    "inventory_skew": self.quote_model.inventory_skew,
                }
            )
        elif self.quote_model.type == "zspread":
            quote_dict.update(
                {
                    "z": self.quote_model.z,
                    "distribution": self.quote_model.distribution,
                    "t_df": self.quote_model.t_df,
                    "vol_mode": self.quote_model.vol_mode,
                    "vol_floor": self.quote_model.vol_floor,
                    "implied_volatility": self.quote_model.implied_volatility,
                    "tau_seconds": self.quote_model.tau_seconds,
                    "strike": self.quote_model.strike,
                    "price_history_max_age_seconds": self.quote_model.price_history_max_age_seconds,
                    "enforce_maker": self.quote_model.enforce_maker,
                    "maker_offset_ticks": self.quote_model.maker_offset_ticks,
                }
            )
        else:  # tpbs or others
            quote_dict.update(
                {
                    "min_z": self.quote_model.min_z,
                    "max_z": self.quote_model.max_z,
                }
            )

        size_dict: dict[str, Any] = {"type": self.size_model.type}
        if self.size_model.type == "proportional":
            size_dict.update(
                {
                    "order_size_pct": self.size_model.order_size_pct,
                    "max_position_pct": self.size_model.max_position_pct,
                    "min_order_size": self.size_model.min_order_size,
                    "min_order_value": self.size_model.min_order_value,
                }
            )
        else:  # inventory_based, confidence_based, fixed
            size_dict.update(
                {
                    "base_size": self.size_model.base_size,
                    "max_position": self.size_model.max_position,
                }
            )

        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "assets": self.assets,
            "timeframe": self.timeframe,
            "auto_subscribe": self.auto_subscribe,
            "starting_capital": self.starting_capital,
            "quote_model": quote_dict,
            "size_model": size_dict,
            "max_position_per_market": self.max_position_per_market,
            "max_total_exposure": self.max_total_exposure,
        }


@dataclass
class MarketDiscoveryConfig:
    """Market discovery configuration."""

    enabled: bool = True
    poll_interval_s: float = 30.0
    supported_timeframes: list[str] = field(default_factory=lambda: ["15m", "1h", "4h"])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarketDiscoveryConfig":
        return cls(
            enabled=data.get("enabled", True),
            poll_interval_s=data.get("poll_interval_s", 30.0),
            supported_timeframes=data.get("supported_timeframes", ["15m", "1h", "4h"]),
        )


@dataclass
class SupabaseConfig:
    """Supabase configuration."""

    url: str = ""
    api_key: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SupabaseConfig":
        # Use YAML value if non-empty, otherwise fall back to environment variable
        return cls(
            url=data.get("url") or os.environ.get("SUPABASE_URL", ""),
            api_key=data.get("api_key") or os.environ.get("SUPABASE_KEY", ""),
        )


@dataclass
class StatePollerConfig:
    """State poller configuration."""

    poll_interval_ms: int = 500
    equity_history_limit: int = 500
    trade_history_limit: int = 50
    quote_history_limit: int = 500

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StatePollerConfig":
        return cls(
            poll_interval_ms=data.get("poll_interval_ms", 500),
            equity_history_limit=data.get("equity_history_limit", 500),
            trade_history_limit=data.get("trade_history_limit", 50),
            quote_history_limit=data.get("quote_history_limit", 500),
        )


@dataclass
class DashboardConfig:
    """Dashboard server configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    external_url: str = ""
    ws_update_interval_ms: int = 200
    supabase: SupabaseConfig = field(default_factory=SupabaseConfig)
    state_poller: StatePollerConfig = field(default_factory=StatePollerConfig)
    strategies_config_path: str = ""  # Path to paper.yaml or strategies.yaml
    # Strategies loaded from config
    strategies: list[StrategyConfig] = field(default_factory=list)
    # Legacy fields for backward compatibility with StrategyManager
    default_strategies: list[StrategyConfig] = field(default_factory=list)
    max_concurrent_strategies: int = 20

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DashboardConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        dashboard_data = data.get("dashboard", {})
        strategies_config_path = dashboard_data.get("strategies_config_path", "")

        # Load strategies from paper.yaml if path specified
        strategies: list[StrategyConfig] = []
        if strategies_config_path:
            strategies = load_strategies_from_config(strategies_config_path)

        return cls(
            host=dashboard_data.get("host", "0.0.0.0"),
            port=dashboard_data.get("port", 8080),
            external_url=dashboard_data.get("external_url", ""),
            ws_update_interval_ms=dashboard_data.get("ws_update_interval_ms", 200),
            supabase=SupabaseConfig.from_dict(data.get("supabase", {})),
            state_poller=StatePollerConfig.from_dict(
                dashboard_data.get("state_poller", {})
            ),
            strategies_config_path=strategies_config_path,
            strategies=strategies,
        )

    @classmethod
    def from_env(cls) -> "DashboardConfig":
        """Create configuration from environment variables."""
        strategies_config_path = os.environ.get("STRATEGIES_CONFIG_PATH", "")
        strategies: list[StrategyConfig] = []
        if strategies_config_path:
            strategies = load_strategies_from_config(strategies_config_path)

        return cls(
            host=os.environ.get("DASHBOARD_HOST", "0.0.0.0"),
            port=int(os.environ.get("DASHBOARD_PORT", "8080")),
            external_url=os.environ.get("DASHBOARD_EXTERNAL_URL", ""),
            supabase=SupabaseConfig(
                url=os.environ.get("SUPABASE_URL", ""),
                api_key=os.environ.get("SUPABASE_KEY", ""),
            ),
            strategies_config_path=strategies_config_path,
            strategies=strategies,
        )


def load_strategies_from_config(config_path: str) -> list[StrategyConfig]:
    """Load strategy definitions from a YAML config file (paper.yaml)."""
    path = Path(config_path)
    if not path.exists():
        return []

    with open(path) as f:
        data = yaml.safe_load(f)

    strategies_data = data.get("strategies", [])
    strategies: list[StrategyConfig] = []

    for s in strategies_data:
        if not s.get("enabled", True):
            continue

        strategies.append(
            StrategyConfig(
                strategy_id=s.get("id", str(uuid.uuid4())),
                name=s.get("name", s.get("id", "Unknown")),
                assets=s.get("assets", ["btc"]),
                timeframe=s.get("timeframe", "15m"),
                auto_subscribe=s.get("auto_subscribe", True),
                starting_capital=s.get("starting_capital", 10000.0),
                quote_model=QuoteModelConfig.from_dict(s.get("quote_model", {})),
                size_model=SizeModelConfig.from_dict(s.get("size_model", {})),
                max_position_per_market=s.get("max_position_per_market", 500.0),
                max_total_exposure=s.get("max_total_exposure", 5000.0),
            )
        )

    return strategies
