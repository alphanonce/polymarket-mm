"""
Base Model Interfaces

Abstract interfaces for quote and size models with built-in normalization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from strategy.shm.types import ExternalPriceState, MarketState, PositionState
from strategy.utils.polymarket import (
    DEFAULT_MIN_SIZE,
    DEFAULT_SIZE_TICK,
    TickInfo,
    clamp_price,
    clamp_quantity,
    get_tick_info,
    get_tick_size,
    round_ask,
    round_bid,
    round_quantity,
)


@dataclass
class QuoteResult:
    """Result from a quote model."""

    bid_price: float
    ask_price: float
    confidence: float = 1.0
    should_quote: bool = True
    reason: str = ""


@dataclass
class SizeResult:
    """Result from a size model."""

    bid_size: float
    ask_size: float
    max_position: float = 0.0


@dataclass
class StrategyState:
    """Complete state available to strategy models."""

    # Market data
    market: MarketState
    external_prices: Dict[str, ExternalPriceState] = field(default_factory=dict)

    # Position info
    position: Optional[PositionState] = None

    # Account info
    total_equity: float = 0.0
    available_margin: float = 0.0

    # Computed features
    features: Optional[np.ndarray] = None

    @property
    def current_position(self) -> float:
        """Get current position size."""
        return self.position.position if self.position else 0.0

    @property
    def mid_price(self) -> float:
        """Get current mid price."""
        return self.market.mid_price

    @property
    def spread(self) -> float:
        """Get current spread."""
        return self.market.spread

    @property
    def best_bid(self) -> float:
        """Get best bid price."""
        if self.market.bids:
            return self.market.bids[0][0]
        return 0.0

    @property
    def best_ask(self) -> float:
        """Get best ask price."""
        if self.market.asks:
            return self.market.asks[0][0]
        return 0.0

    @property
    def token_id(self) -> str:
        """Get market token/asset ID."""
        return self.market.asset_id

    def get_external_price(self, symbol: str) -> Optional[float]:
        """Get external price for a symbol."""
        if symbol in self.external_prices:
            return self.external_prices[symbol].price
        return None


@dataclass
class NormalizationConfig:
    """Configuration for price/size normalization."""

    # Price normalization
    round_bid_down: bool = True  # Floor bid prices (profitable)
    round_ask_up: bool = True  # Ceil ask prices (profitable)
    clamp_prices: bool = True  # Clamp to [0.01, 0.99]
    use_dynamic_tick: bool = True  # Use price-dependent tick size

    # Size normalization
    round_sizes: bool = True  # Round sizes to tick
    enforce_min_size: bool = True  # Enforce minimum order size
    size_tick: float = DEFAULT_SIZE_TICK
    min_size: float = DEFAULT_MIN_SIZE

    # Override tick info (if None, uses market defaults)
    price_tick: Optional[float] = None


class QuoteModel(ABC):
    """Abstract base class for quote models with normalization."""

    def __init__(self, normalization: Optional[NormalizationConfig] = None):
        self._norm_config = normalization or NormalizationConfig()

    @property
    def normalization(self) -> NormalizationConfig:
        """Get normalization config."""
        return self._norm_config

    @abstractmethod
    def compute_raw(self, state: StrategyState) -> QuoteResult:
        """
        Compute raw bid and ask prices (before normalization).

        Subclasses must implement this method.

        Args:
            state: Current strategy state

        Returns:
            QuoteResult with raw bid/ask prices
        """
        pass

    def compute(self, state: StrategyState) -> QuoteResult:
        """
        Compute bid and ask prices with normalization.

        Calls compute_raw() and applies price normalization.

        Args:
            state: Current strategy state

        Returns:
            QuoteResult with normalized bid/ask prices
        """
        result = self.compute_raw(state)

        if not result.should_quote:
            return result

        # Normalize prices
        bid, ask = self.normalize_quote(
            result.bid_price,
            result.ask_price,
            state,
        )

        return QuoteResult(
            bid_price=bid,
            ask_price=ask,
            confidence=result.confidence,
            should_quote=result.should_quote,
            reason=result.reason,
        )

    def normalize_quote(
        self,
        bid: float,
        ask: float,
        state: Optional[StrategyState] = None,
    ) -> tuple[float, float]:
        """
        Normalize bid/ask prices with rounding and clamping.

        Args:
            bid: Raw bid price
            ask: Raw ask price
            state: Optional state for market-specific tick info

        Returns:
            Tuple of (normalized_bid, normalized_ask)
        """
        config = self._norm_config

        # Get tick size
        if config.price_tick is not None:
            tick = config.price_tick
        elif config.use_dynamic_tick:
            # Use price-dependent tick (different near boundaries)
            bid_tick = get_tick_size(bid)
            ask_tick = get_tick_size(ask)
        else:
            bid_tick = ask_tick = get_tick_size(0.5)  # Default

        # Round prices
        if config.price_tick is not None:
            tick = config.price_tick
            if config.round_bid_down:
                bid = round_bid(bid, tick)
            if config.round_ask_up:
                ask = round_ask(ask, tick)
        else:
            if config.round_bid_down:
                bid = round_bid(bid, bid_tick if config.use_dynamic_tick else None)
            if config.round_ask_up:
                ask = round_ask(ask, ask_tick if config.use_dynamic_tick else None)

        # Clamp to valid range
        if config.clamp_prices:
            bid = clamp_price(bid)
            ask = clamp_price(ask)

        return bid, ask

    def update(
        self,
        state: StrategyState,
        result: QuoteResult,
        fill_info: Optional[dict] = None,
    ) -> None:
        """
        Update model state after a decision.

        Override this method for stateful models.

        Args:
            state: Strategy state used for decision
            result: The quote result that was produced
            fill_info: Information about any fills that occurred
        """
        pass

    def copy(self) -> "QuoteModel":
        """
        Create a deep copy of this model.

        Stateful models must implement this for thread-safe usage in
        parallel backtesting (e.g., ProcessPoolExecutor).

        Returns:
            A new QuoteModel with copied state

        Raises:
            NotImplementedError: If model is stateful but doesn't implement copy
        """
        raise NotImplementedError("Stateful models must implement copy()")

    def reset(self) -> None:
        """
        Reset model to initial state.

        Override this method for stateful models to clear accumulated state.
        Default implementation does nothing (for stateless models).
        """
        pass


class SizeModel(ABC):
    """Abstract base class for size models with normalization."""

    def __init__(self, normalization: Optional[NormalizationConfig] = None):
        self._norm_config = normalization or NormalizationConfig()

    @property
    def normalization(self) -> NormalizationConfig:
        """Get normalization config."""
        return self._norm_config

    @abstractmethod
    def compute_raw(self, state: StrategyState, quote: QuoteResult) -> SizeResult:
        """
        Compute raw bid and ask sizes (before normalization).

        Subclasses must implement this method.

        Args:
            state: Current strategy state
            quote: Quote result from quote model

        Returns:
            SizeResult with raw bid/ask sizes
        """
        pass

    def compute(self, state: StrategyState, quote: QuoteResult) -> SizeResult:
        """
        Compute bid and ask sizes with normalization.

        Calls compute_raw() and applies size normalization.

        Args:
            state: Current strategy state
            quote: Quote result from quote model

        Returns:
            SizeResult with normalized bid/ask sizes
        """
        result = self.compute_raw(state, quote)

        # Normalize sizes
        bid_size, ask_size = self.normalize_size(
            result.bid_size,
            result.ask_size,
            state,
        )

        return SizeResult(
            bid_size=bid_size,
            ask_size=ask_size,
            max_position=result.max_position,
        )

    def normalize_size(
        self,
        bid_size: float,
        ask_size: float,
        state: Optional[StrategyState] = None,
    ) -> tuple[float, float]:
        """
        Normalize bid/ask sizes with rounding and min size enforcement.

        Args:
            bid_size: Raw bid size
            ask_size: Raw ask size
            state: Optional state for market-specific tick info

        Returns:
            Tuple of (normalized_bid_size, normalized_ask_size)
        """
        config = self._norm_config

        # Get tick info from market if available
        size_tick = config.size_tick
        min_size = config.min_size

        if state is not None:
            tick_info = get_tick_info(state.token_id)
            size_tick = tick_info.size_tick
            min_size = tick_info.min_size

        # Round sizes
        if config.round_sizes:
            bid_size = round_quantity(bid_size, tick=size_tick, direction="down")
            ask_size = round_quantity(ask_size, tick=size_tick, direction="down")

        # Enforce minimum size
        if config.enforce_min_size:
            bid_size = clamp_quantity(bid_size, min_size)
            ask_size = clamp_quantity(ask_size, min_size)

        return bid_size, ask_size

    def update(
        self,
        state: StrategyState,
        result: SizeResult,
        fill_info: Optional[dict] = None,
    ) -> None:
        """
        Update model state after a decision.

        Override this method for stateful models.

        Args:
            state: Strategy state used for decision
            result: The size result that was produced
            fill_info: Information about any fills that occurred
        """
        pass

    def copy(self) -> "SizeModel":
        """
        Create a deep copy of this model.

        Stateful models must implement this for thread-safe usage in
        parallel backtesting (e.g., ProcessPoolExecutor).

        Returns:
            A new SizeModel with copied state

        Raises:
            NotImplementedError: If model is stateful but doesn't implement copy
        """
        raise NotImplementedError("Stateful models must implement copy()")

    def reset(self) -> None:
        """
        Reset model to initial state.

        Override this method for stateful models to clear accumulated state.
        Default implementation does nothing (for stateless models).
        """
        pass


class FeatureExtractor:
    """Extracts features from market state for model input."""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.n_features = len(feature_names)

    def extract(self, state: StrategyState) -> np.ndarray:
        """
        Extract features from state.

        Args:
            state: Current strategy state

        Returns:
            Feature vector as numpy array
        """
        features = np.zeros(self.n_features)

        for i, name in enumerate(self.feature_names):
            features[i] = self._extract_feature(name, state)

        return features

    def _extract_feature(self, name: str, state: StrategyState) -> float:
        """Extract a single feature by name."""
        if name == "mid_price":
            return state.mid_price
        elif name == "spread":
            return state.spread
        elif name == "position":
            return state.current_position
        elif name == "bid_depth":
            return sum(size for _, size in state.market.bids[:5])
        elif name == "ask_depth":
            return sum(size for _, size in state.market.asks[:5])
        elif name == "imbalance":
            bid_depth = sum(size for _, size in state.market.bids[:5])
            ask_depth = sum(size for _, size in state.market.asks[:5])
            total = bid_depth + ask_depth
            return (bid_depth - ask_depth) / total if total > 0 else 0.0
        elif name == "last_trade_side":
            return float(state.market.last_trade_side)
        elif name.startswith("ext_"):
            symbol = name[4:]
            price = state.get_external_price(symbol)
            return price if price else 0.0
        else:
            return 0.0
