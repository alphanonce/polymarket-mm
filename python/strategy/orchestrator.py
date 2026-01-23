"""
Strategy Orchestrator

Main strategy loop that reads market state, computes quotes/sizes,
and writes order signals to shared memory.
"""

import signal
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import structlog

from strategy.models.base import QuoteModel, QuoteResult, SizeModel, SizeResult, StrategyState
from strategy.models.features import AdvancedFeatureExtractor, FeatureConfig
from strategy.models.quote import InventoryAdjustedQuoteModel
from strategy.models.size import InventoryBasedSizeModel
from strategy.shm import SHMReader, SHMWriter
from strategy.shm.types import SIDE_BUY, SIDE_SELL

logger = structlog.get_logger()


@dataclass
class OrchestratorConfig:
    """Configuration for the strategy orchestrator."""

    # Loop timing
    tick_interval_ms: float = 10.0  # Target tick interval in milliseconds
    min_state_age_ms: float = 1.0  # Minimum age of state update to process

    # Markets to trade
    market_asset_ids: List[str] = field(default_factory=list)

    # External price mappings (asset_id -> external symbol)
    external_price_mappings: Dict[str, str] = field(default_factory=dict)

    # Risk limits
    max_open_orders: int = 10
    max_position_per_market: float = 500.0
    max_total_exposure: float = 5000.0

    # Quote management
    quote_refresh_interval_ms: float = 1000.0  # Refresh quotes every N ms
    cancel_stale_quotes_ms: float = 5000.0  # Cancel quotes older than N ms

    # Feature extraction
    feature_config: Optional[FeatureConfig] = None


@dataclass
class MarketQuote:
    """Represents current quotes for a market."""

    asset_id: str
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    timestamp_ns: int
    signal_ids: tuple[int, int] = (0, 0)  # (bid_signal_id, ask_signal_id)


class Orchestrator:
    """
    Main strategy orchestrator.

    Runs the strategy loop that:
    1. Reads market state from shared memory
    2. Computes quotes and sizes using models
    3. Writes order signals to shared memory
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        quote_model: Optional[QuoteModel] = None,
        size_model: Optional[SizeModel] = None,
    ):
        self.config = config
        self.logger = logger.bind(component="orchestrator")

        # Shared memory
        self.shm_reader = SHMReader()
        self.shm_writer = SHMWriter()

        # Models
        self.quote_model = quote_model or InventoryAdjustedQuoteModel()
        self.size_model = size_model or InventoryBasedSizeModel()

        # Feature extraction
        feature_config = config.feature_config or FeatureConfig(
            external_symbols=list(config.external_price_mappings.values())
        )
        self.feature_extractor = AdvancedFeatureExtractor(feature_config)

        # State
        self.current_quotes: Dict[str, MarketQuote] = {}
        self.last_state_sequence: int = 0
        self.last_quote_time: Dict[str, int] = {}

        # Control
        self.running = False
        self._setup_signal_handlers()

        # Metrics
        self.tick_count: int = 0
        self.signal_count: int = 0
        self.last_tick_duration_ns: int = 0

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame: object) -> None:
        """Handle shutdown signals."""
        self.logger.info("Received signal, stopping", signal=signum)
        self.running = False

    def start(self) -> None:
        """Start the strategy loop."""
        self.logger.info("Starting orchestrator", config=self.config)

        try:
            self.shm_reader.connect()
            self.shm_writer.connect()
            self.logger.info("Connected to shared memory")
        except Exception as e:
            self.logger.error("Failed to connect to shared memory", error=str(e))
            raise

        self.running = True
        self._run_loop()

    def stop(self) -> None:
        """Stop the strategy loop."""
        self.running = False

    def _run_loop(self) -> None:
        """Main strategy loop."""
        tick_interval_ns = int(self.config.tick_interval_ms * 1_000_000)

        while self.running:
            tick_start = time.time_ns()

            try:
                self._tick()
            except Exception as e:
                self.logger.error("Error in tick", error=str(e))

            # Sleep for remaining time
            tick_duration = time.time_ns() - tick_start
            self.last_tick_duration_ns = tick_duration

            sleep_ns = tick_interval_ns - tick_duration
            if sleep_ns > 0:
                time.sleep(sleep_ns / 1_000_000_000)

            self.tick_count += 1

        # Cleanup
        self._cleanup()

    def _tick(self) -> None:
        """Single tick of the strategy loop."""
        # Refresh shared memory state
        state_changed = self.shm_reader.refresh()

        if not state_changed:
            return

        # Check if trading is enabled
        if not self.shm_reader.trading_enabled:
            return

        current_time_ns = time.time_ns()

        # Process each market
        for asset_id in self.config.market_asset_ids:
            market = self.shm_reader.get_market(asset_id)
            if market is None:
                continue

            # Build strategy state
            state = self._build_strategy_state(asset_id, market)

            # Check if we need to refresh quotes
            last_quote = self.last_quote_time.get(asset_id, 0)
            quote_age_ms = (current_time_ns - last_quote) / 1_000_000

            if quote_age_ms >= self.config.quote_refresh_interval_ms:
                self._update_quotes(asset_id, state)
                self.last_quote_time[asset_id] = current_time_ns

    def _build_strategy_state(self, asset_id: str, market: "MarketState") -> StrategyState:
        """Build strategy state for a market."""
        from strategy.shm.types import MarketState as SHMMarketState

        # Get position
        position = self.shm_reader.get_position(asset_id)

        # Get external prices
        external_prices = {}
        for aid, symbol in self.config.external_price_mappings.items():
            if aid == asset_id:
                ext_price = self.shm_reader.get_external_price(symbol)
                if ext_price:
                    external_prices[symbol] = ext_price

        state = StrategyState(
            market=market,
            external_prices=external_prices,
            position=position,
            total_equity=self.shm_reader.total_equity,
            available_margin=self.shm_reader.available_margin,
        )

        # Extract features
        state.features = self.feature_extractor.extract(state)

        return state

    def _update_quotes(self, asset_id: str, state: StrategyState) -> None:
        """Update quotes for a market."""
        # Compute quote
        quote_result = self.quote_model.compute(state)

        if not quote_result.should_quote:
            self.logger.debug(
                "Not quoting",
                asset_id=asset_id,
                reason=quote_result.reason,
            )
            return

        # Compute sizes
        size_result = self.size_model.compute(state, quote_result)

        # Check if quotes changed significantly
        current = self.current_quotes.get(asset_id)
        if current and not self._quotes_changed(current, quote_result, size_result):
            return

        # Cancel existing quotes
        if current:
            self._cancel_quotes(asset_id, current)

        # Place new quotes
        new_quote = self._place_quotes(asset_id, quote_result, size_result)
        if new_quote:
            self.current_quotes[asset_id] = new_quote

        self.logger.debug(
            "Updated quotes",
            asset_id=asset_id,
            bid_price=quote_result.bid_price,
            bid_size=size_result.bid_size,
            ask_price=quote_result.ask_price,
            ask_size=size_result.ask_size,
        )

    def _quotes_changed(
        self,
        current: MarketQuote,
        quote: QuoteResult,
        size: SizeResult,
    ) -> bool:
        """Check if quotes changed significantly."""
        price_threshold = 0.0001
        size_threshold = 0.01

        if abs(current.bid_price - quote.bid_price) > price_threshold:
            return True
        if abs(current.ask_price - quote.ask_price) > price_threshold:
            return True
        if abs(current.bid_size - size.bid_size) > size_threshold:
            return True
        if abs(current.ask_size - size.ask_size) > size_threshold:
            return True

        return False

    def _cancel_quotes(self, asset_id: str, quote: MarketQuote) -> None:
        """Cancel existing quotes."""
        # In practice, you'd want to track order IDs and cancel them
        # For now, we'll let the Go executor handle order management
        pass

    def _place_quotes(
        self,
        asset_id: str,
        quote: QuoteResult,
        size: SizeResult,
    ) -> Optional[MarketQuote]:
        """Place new quotes via shared memory signals."""
        bid_signal_id = 0
        ask_signal_id = 0

        try:
            # Place bid
            if size.bid_size > 0 and quote.bid_price > 0:
                bid_signal_id = self.shm_writer.place_limit_order(
                    asset_id=asset_id,
                    side=SIDE_BUY,
                    price=quote.bid_price,
                    size=size.bid_size,
                )
                self.signal_count += 1

            # Place ask
            if size.ask_size > 0 and quote.ask_price > 0:
                ask_signal_id = self.shm_writer.place_limit_order(
                    asset_id=asset_id,
                    side=SIDE_SELL,
                    price=quote.ask_price,
                    size=size.ask_size,
                )
                self.signal_count += 1

            return MarketQuote(
                asset_id=asset_id,
                bid_price=quote.bid_price,
                bid_size=size.bid_size,
                ask_price=quote.ask_price,
                ask_size=size.ask_size,
                timestamp_ns=time.time_ns(),
                signal_ids=(bid_signal_id, ask_signal_id),
            )

        except Exception as e:
            self.logger.error("Failed to place quotes", asset_id=asset_id, error=str(e))
            return None

    def _cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info(
            "Cleaning up",
            tick_count=self.tick_count,
            signal_count=self.signal_count,
        )

        try:
            self.shm_reader.close()
            self.shm_writer.close()
        except Exception as e:
            self.logger.error("Error during cleanup", error=str(e))

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        return {
            "tick_count": self.tick_count,
            "signal_count": self.signal_count,
            "last_tick_duration_us": self.last_tick_duration_ns / 1000,
            "active_markets": len(self.current_quotes),
        }


def main() -> None:
    """Main entry point."""
    import yaml

    # Load config
    config_path = "data/config/strategy.yaml"
    try:
        with open(config_path) as f:
            cfg_dict = yaml.safe_load(f)
    except FileNotFoundError:
        cfg_dict = {}

    config = OrchestratorConfig(
        market_asset_ids=cfg_dict.get("markets", []),
        external_price_mappings=cfg_dict.get("external_prices", {}),
        tick_interval_ms=cfg_dict.get("tick_interval_ms", 10.0),
    )

    orchestrator = Orchestrator(config)
    orchestrator.start()


if __name__ == "__main__":
    main()
