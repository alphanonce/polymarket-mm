"""
Feature Extraction

Feature extraction utilities for strategy models.
"""

from dataclasses import dataclass, field

import numpy as np

from strategy.models.base import StrategyState


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    # Price features
    include_mid_price: bool = True
    include_spread: bool = True
    include_relative_spread: bool = True

    # Orderbook features
    include_book_imbalance: bool = True
    include_depth: bool = True
    depth_levels: int = 5

    # Trade features
    include_last_trade: bool = True

    # Position features
    include_position: bool = True
    include_normalized_position: bool = True
    max_position: float = 500.0

    # External price features
    external_symbols: list[str] = field(default_factory=list)
    include_external_spread: bool = True

    # Time features
    include_time_features: bool = False


class AdvancedFeatureExtractor:
    """
    Advanced feature extractor with configurable features.
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self._feature_names: list[str] = []
        self._build_feature_list()

    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names."""
        return self._feature_names.copy()

    @property
    def n_features(self) -> int:
        """Get number of features."""
        return len(self._feature_names)

    def _build_feature_list(self) -> None:
        """Build the list of features based on config."""
        self._feature_names = []

        if self.config.include_mid_price:
            self._feature_names.append("mid_price")

        if self.config.include_spread:
            self._feature_names.append("spread")

        if self.config.include_relative_spread:
            self._feature_names.append("relative_spread")

        if self.config.include_book_imbalance:
            self._feature_names.append("book_imbalance")

        if self.config.include_depth:
            self._feature_names.append("bid_depth")
            self._feature_names.append("ask_depth")
            self._feature_names.append("depth_ratio")

        if self.config.include_last_trade:
            self._feature_names.append("last_trade_price")
            self._feature_names.append("last_trade_size")
            self._feature_names.append("last_trade_side")

        if self.config.include_position:
            self._feature_names.append("position")

        if self.config.include_normalized_position:
            self._feature_names.append("normalized_position")

        for symbol in self.config.external_symbols:
            self._feature_names.append(f"ext_{symbol}")
            if self.config.include_external_spread:
                self._feature_names.append(f"ext_{symbol}_spread")

    def extract(self, state: StrategyState) -> np.ndarray:
        """
        Extract all configured features.

        Args:
            state: Current strategy state

        Returns:
            Feature vector as numpy array
        """
        features: dict[str, float] = {}

        # Price features
        if self.config.include_mid_price:
            features["mid_price"] = state.mid_price

        if self.config.include_spread:
            features["spread"] = state.spread

        if self.config.include_relative_spread:
            features["relative_spread"] = (
                state.spread / state.mid_price if state.mid_price > 0 else 0.0
            )

        # Orderbook features
        if self.config.include_book_imbalance:
            features["book_imbalance"] = self._compute_imbalance(state)

        if self.config.include_depth:
            bid_depth, ask_depth = self._compute_depth(state)
            features["bid_depth"] = bid_depth
            features["ask_depth"] = ask_depth
            total_depth = bid_depth + ask_depth
            features["depth_ratio"] = bid_depth / total_depth if total_depth > 0 else 0.5

        # Trade features
        if self.config.include_last_trade:
            features["last_trade_price"] = state.market.last_trade_price
            features["last_trade_size"] = state.market.last_trade_size
            features["last_trade_side"] = float(state.market.last_trade_side)

        # Position features
        if self.config.include_position:
            features["position"] = state.current_position

        if self.config.include_normalized_position:
            features["normalized_position"] = (
                state.current_position / self.config.max_position
                if self.config.max_position > 0
                else 0.0
            )

        # External price features
        for symbol in self.config.external_symbols:
            ext = state.external_prices.get(symbol)
            if ext:
                features[f"ext_{symbol}"] = ext.price
                if self.config.include_external_spread:
                    ext_spread = ext.ask - ext.bid if ext.ask > ext.bid else 0.0
                    features[f"ext_{symbol}_spread"] = ext_spread
            else:
                features[f"ext_{symbol}"] = 0.0
                if self.config.include_external_spread:
                    features[f"ext_{symbol}_spread"] = 0.0

        # Build output array in correct order
        result = np.zeros(len(self._feature_names))
        for i, name in enumerate(self._feature_names):
            result[i] = features.get(name, 0.0)

        return result

    def _compute_imbalance(self, state: StrategyState) -> float:
        """Compute bid/ask imbalance."""
        n_levels = self.config.depth_levels

        bid_volume = sum(
            size for _, size in state.market.bids[:n_levels]
        )
        ask_volume = sum(
            size for _, size in state.market.asks[:n_levels]
        )

        total = bid_volume + ask_volume
        if total == 0:
            return 0.0

        return (bid_volume - ask_volume) / total

    def _compute_depth(self, state: StrategyState) -> tuple[float, float]:
        """Compute bid and ask depth."""
        n_levels = self.config.depth_levels

        bid_depth = sum(size for _, size in state.market.bids[:n_levels])
        ask_depth = sum(size for _, size in state.market.asks[:n_levels])

        return bid_depth, ask_depth


def compute_vwap(levels: list[tuple[float, float]], depth: float = 100.0) -> float:
    """
    Compute volume-weighted average price.

    Args:
        levels: List of (price, size) tuples
        depth: Amount of volume to consider

    Returns:
        VWAP price
    """
    total_value = 0.0
    total_volume = 0.0

    for price, size in levels:
        if total_volume >= depth:
            break
        remaining = min(size, depth - total_volume)
        total_value += price * remaining
        total_volume += remaining

    return total_value / total_volume if total_volume > 0 else 0.0


def compute_microprice(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
) -> float:
    """
    Compute the microprice (volume-weighted mid).

    Args:
        bids: List of (price, size) tuples for bids
        asks: List of (price, size) tuples for asks

    Returns:
        Microprice
    """
    if not bids or not asks:
        return 0.0

    best_bid_price, best_bid_size = bids[0]
    best_ask_price, best_ask_size = asks[0]

    total_size = best_bid_size + best_ask_size
    if total_size == 0:
        return (best_bid_price + best_ask_price) / 2

    # Weight by opposite side's size
    microprice = (
        best_bid_price * best_ask_size + best_ask_price * best_bid_size
    ) / total_size

    return microprice
