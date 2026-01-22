"""
Datalake Backtest Engine

Runs tick-by-tick backtests using processed datalake parquet data.
Uses crossing logic: fills when our quote crosses the market BBO.
"""

import time as time_module
from dataclasses import dataclass, field

import numpy as np
import structlog

from backtest.data_loader import MarketData, OrderBook, Trade
from backtest.datalake_loader import DatalakeLoader, DatalakeMarketData, MarketInfo, RTDSPriceLoader
from backtest.simulator import (
    SIDE_BUY,
    SIDE_SELL,
    Fill,
    MarketSimulator,
    Position,
)
from backtest.utils import (
    ANNUALIZATION_FACTOR,
    compute_maker_rebate_numba,
    compute_max_drawdown_numba,
    compute_pnl_numba,
    compute_sharpe_numba,
)
from strategy.models.base import (
    QuoteModel,
    SizeModel,
    StrategyState,
)
from strategy.shm.types import ExternalPriceState, MarketState, PositionState

logger = structlog.get_logger()


@dataclass
class DatalakeBacktestConfig:
    """Configuration for datalake-based backtest."""

    # Data source
    data_dir: str = "data/datalake/processed/15m"
    rtds_price_dir: str | None = "data/datalake/global/rtds_crypto_prices"
    symbols: list[str] | None = None  # None = all symbols
    start_date: str | None = None  # YYYY-MM-DD
    end_date: str | None = None  # YYYY-MM-DD
    side: str = "up"  # "up", "down", or "both"

    # Fees
    maker_fee: float = 0.0
    taker_fee: float = 0.0

    # Position limits
    initial_equity: float = 10000.0
    max_position: float = 100.0

    # Quote update frequency
    quote_refresh_interval_ns: int = 100_000_000  # 100ms

    # Logging
    log_interval: int = 100_000  # Log every N ticks
    verbose: bool = False


@dataclass
class MarketBacktestMetrics:
    """Metrics for a single 15-min market."""

    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_fills: int = 0
    buy_fills: int = 0
    sell_fills: int = 0
    total_volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    total_fees: float = 0.0
    maker_rebate: float = 0.0  # Fee-curve weighted maker rebate
    max_position: float = 0.0
    min_position: float = 0.0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    total_ticks: int = 0
    book_ticks: int = 0
    trade_ticks: int = 0
    avg_spread: float = 0.0
    quotes_placed: int = 0


@dataclass
class MarketBacktestResult:
    """Result for a single 15-min market."""

    info: MarketInfo
    metrics: MarketBacktestMetrics
    fills: list[Fill]
    final_position: Position
    pnl_samples: list[float] = field(default_factory=list)


@dataclass
class DatalakeBacktestResult:
    """Aggregate result across all markets."""

    config: DatalakeBacktestConfig
    market_results: dict[str, MarketBacktestResult]

    # Aggregate metrics
    total_pnl: float = 0.0
    total_fills: int = 0
    total_volume: float = 0.0
    total_markets: int = 0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_maker_rebate: float = 0.0
    pnl_after_rebate: float = 0.0

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from individual market results."""
        self.total_markets = len(self.market_results)
        self.total_pnl = sum(r.metrics.total_pnl for r in self.market_results.values())
        self.total_fills = sum(r.metrics.total_fills for r in self.market_results.values())
        self.total_volume = sum(r.metrics.total_volume for r in self.market_results.values())
        self.total_maker_rebate = sum(r.metrics.maker_rebate for r in self.market_results.values())
        self.pnl_after_rebate = self.total_pnl + self.total_maker_rebate

        # Compute Sharpe and max drawdown using Numba-optimized functions
        all_pnl_samples: list[float] = []
        for r in self.market_results.values():
            all_pnl_samples.extend(r.pnl_samples)

        if all_pnl_samples:
            pnl_arr = np.array(all_pnl_samples, dtype=np.float64)
            self.sharpe_ratio = compute_sharpe_numba(pnl_arr, ANNUALIZATION_FACTOR)
            self.max_drawdown = compute_max_drawdown_numba(pnl_arr)


@dataclass
class TickEvent:
    """Unified tick event for orderbook or trade."""

    timestamp_ns: int
    is_book: bool  # True = orderbook update, False = trade
    book: OrderBook | None = None
    trade: Trade | None = None


class DatalakeBacktestEngine:
    """
    Datalake-based backtest engine.

    Processes tick-by-tick data from datalake parquet files and
    simulates fill execution based on crossing logic.

    Fill Logic:
        - BUY order fills if: trade_price < order_price (crossing)
        - SELL order fills if: trade_price > order_price (crossing)
    """

    def __init__(
        self,
        config: DatalakeBacktestConfig,
        quote_model: QuoteModel,
        size_model: SizeModel,
    ):
        self.config = config
        self.quote_model = quote_model
        self.size_model = size_model
        self.logger = logger.bind(
            component="datalake_backtest",
            side=config.side,
        )

        # Initialize loader
        self.loader = DatalakeLoader(
            data_dir=config.data_dir,
            symbols=config.symbols,
            start_date=config.start_date,
            end_date=config.end_date,
            side=config.side,
        )

        # Initialize RTDS price loader
        self.rtds_loader: RTDSPriceLoader | None = None
        if config.rtds_price_dir:
            self.rtds_loader = RTDSPriceLoader(data_dir=config.rtds_price_dir)
            if self.rtds_loader.load():
                self.logger.info(
                    "RTDS prices loaded",
                    symbols=self.rtds_loader.symbols,
                )
            else:
                self.logger.warning("Failed to load RTDS prices")
                self.rtds_loader = None

        # Per-market state (reset for each market)
        self.simulator: MarketSimulator | None = None
        self._last_quote_time_ns: int = 0
        self._current_mid_price: float = 0.0
        self._current_book: OrderBook | None = None
        self._current_symbol: str = ""
        self._external_prices: dict[str, ExternalPriceState] = {}

        # Pre-allocated buffers for state building (Phase 2b optimization)
        self._bids_buffer: list[tuple[float, float]] = [(0.0, 0.0)] * 20
        self._asks_buffer: list[tuple[float, float]] = [(0.0, 0.0)] * 20
        self._pnl_samples_buffer: np.ndarray = np.zeros(10000, dtype=np.float64)
        self._pnl_sample_idx: int = 0

    def run(self) -> DatalakeBacktestResult:
        """
        Run backtest across all markets.

        Returns:
            DatalakeBacktestResult with aggregate metrics
        """
        start_time = time_module.time()

        markets = self.loader.discover_markets()
        self.logger.info(
            "Starting datalake backtest",
            total_markets=len(markets),
            symbols=self.config.symbols,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )

        market_results: dict[str, MarketBacktestResult] = {}

        for i, market in enumerate(markets):
            data = self.loader.load_market(market)

            if not data.has_data:
                continue

            result = self.run_single_market(data)
            market_results[market.slug] = result

            if self.config.verbose and (i + 1) % 10 == 0:
                self.logger.info(
                    "Progress",
                    completed=i + 1,
                    total=len(markets),
                    pct=f"{(i + 1) / len(markets) * 100:.1f}%",
                )

        elapsed = time_module.time() - start_time

        # Create aggregate result
        result = DatalakeBacktestResult(
            config=self.config,
            market_results=market_results,
        )
        result.compute_aggregates()

        self.logger.info(
            "Backtest completed",
            elapsed_seconds=f"{elapsed:.2f}",
            total_markets=result.total_markets,
            total_pnl=f"${result.total_pnl:.2f}",
            total_fills=result.total_fills,
            sharpe_ratio=f"{result.sharpe_ratio:.2f}",
        )

        return result

    def run_single_market(self, data: DatalakeMarketData) -> MarketBacktestResult:
        """
        Run backtest for a single 15-min market.

        Args:
            data: Market data to process

        Returns:
            MarketBacktestResult with metrics and fills
        """
        # Reset state
        self.simulator = MarketSimulator(
            maker_fee=self.config.maker_fee,
            taker_fee=self.config.taker_fee,
        )
        self._last_quote_time_ns = 0
        self._current_mid_price = 0.0
        self._current_book = None
        self._current_symbol = data.info.symbol.upper()
        self._pnl_sample_idx = 0  # Reset PnL sample buffer index

        # Reset quote model state for stateful models
        if hasattr(self.quote_model, 'reset'):
            self.quote_model.reset()

        # Merge and sort events by timestamp
        events = self._merge_events(data)

        # Set strike price from RTDS at market start
        if events and self.rtds_loader and hasattr(self.quote_model, 'config'):
            first_timestamp_ns = events[0].timestamp_ns
            strike_price = self.rtds_loader.get_price_ns(self._current_symbol, first_timestamp_ns)
            if strike_price and hasattr(self.quote_model.config, 'strike'):
                self.quote_model.config.strike = strike_price

        if not events:
            return MarketBacktestResult(
                info=data.info,
                metrics=MarketBacktestMetrics(),
                fills=[],
                final_position=Position(),
            )

        # Initialize metrics
        metrics = MarketBacktestMetrics()
        metrics.total_ticks = len(events)
        spread_sum = 0.0
        spread_count = 0

        # Local references for faster access in hot loop
        simulator = self.simulator
        config = self.config
        quote_refresh_ns = config.quote_refresh_interval_ns
        pnl_buffer = self._pnl_samples_buffer
        pnl_idx = 0

        # Main loop (optimized)
        for idx, event in enumerate(events):
            if event.is_book and event.book is not None:
                metrics.book_ticks += 1
                self._process_book_update(event.book, metrics)
            elif event.trade is not None:
                metrics.trade_ticks += 1
                self._process_trade(event.trade, metrics)

            # Track spread
            if self._current_book and self._current_book.spread > 0:
                spread_sum += self._current_book.spread
                spread_count += 1

            # Update quotes if needed
            if event.timestamp_ns - self._last_quote_time_ns >= quote_refresh_ns:
                if self._current_book and self._current_mid_price > 0:
                    self._update_quotes(event.timestamp_ns)
                    self._last_quote_time_ns = event.timestamp_ns
                    metrics.quotes_placed += 1

            # Track PnL using Numba-optimized function
            if self._current_mid_price > 0:
                pos = simulator.position
                unrealized = compute_pnl_numba(pos.size, pos.avg_entry_price, self._current_mid_price)
                total_pnl = pos.realized_pnl + unrealized
                metrics.peak_pnl = max(metrics.peak_pnl, total_pnl)
                drawdown = metrics.peak_pnl - total_pnl
                metrics.max_drawdown = max(metrics.max_drawdown, drawdown)

                # Sample PnL for Sharpe calculation (every 100 ticks)
                if idx % 100 == 0 and pnl_idx < len(pnl_buffer):
                    pnl_buffer[pnl_idx] = total_pnl
                    pnl_idx += 1

            # Track position extremes
            pos_size = simulator.position.size
            metrics.max_position = max(metrics.max_position, pos_size)
            metrics.min_position = min(metrics.min_position, pos_size)

        # Store final PnL sample index
        self._pnl_sample_idx = pnl_idx

        # Final metrics
        if self._current_mid_price > 0:
            pos = simulator.position
            metrics.unrealized_pnl = compute_pnl_numba(pos.size, pos.avg_entry_price, self._current_mid_price)
            metrics.total_pnl = pos.realized_pnl + metrics.unrealized_pnl
        metrics.realized_pnl = simulator.position.realized_pnl
        metrics.total_fees = simulator.position.total_fees
        metrics.avg_spread = spread_sum / spread_count if spread_count > 0 else 0.0

        # Convert PnL samples to list for result (only used samples)
        pnl_samples = pnl_buffer[:pnl_idx].tolist() if pnl_idx > 0 else []

        return MarketBacktestResult(
            info=data.info,
            metrics=metrics,
            fills=simulator.fills,
            final_position=simulator.position,
            pnl_samples=pnl_samples,
        )

    def _merge_events(self, data: DatalakeMarketData) -> list[TickEvent]:
        """Merge orderbook and trade events, sorted by timestamp."""
        events: list[TickEvent] = []

        # Add book events
        for book in data.books:
            events.append(TickEvent(
                timestamp_ns=book.timestamp_ns,
                is_book=True,
                book=book,
            ))

        # Add trade events
        for trade in data.trades:
            events.append(TickEvent(
                timestamp_ns=trade.timestamp_ns,
                is_book=False,
                trade=trade,
            ))

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp_ns)

        return events

    def _process_book_update(self, book: OrderBook, metrics: MarketBacktestMetrics) -> None:
        """Process an orderbook update."""
        self._current_book = book

        # Update mid price
        if book.bids and book.asks:
            self._current_mid_price = (book.bids[0].price + book.asks[0].price) / 2

        # Check for fills (orderbook crossing)
        data = MarketData(timestamp_ns=book.timestamp_ns, orderbook=book)
        fills = self.simulator.process_tick(data)

        self._update_fill_metrics(fills, metrics)

    def _process_trade(self, trade: Trade, metrics: MarketBacktestMetrics) -> None:
        """Process a trade event."""
        # Update last trade price as mid price estimate
        if trade.price > 0:
            self._current_mid_price = trade.price

        # Check for fills (trade crossing)
        fills = self.simulator.process_trade(trade)

        self._update_fill_metrics(fills, metrics)

    def _update_fill_metrics(self, fills: list[Fill], metrics: MarketBacktestMetrics) -> None:
        """Update metrics based on fills."""
        for fill in fills:
            metrics.total_fills += 1
            metrics.total_volume += fill.size

            # Calculate maker rebate using Numba-optimized function
            metrics.maker_rebate += compute_maker_rebate_numba(fill.price, fill.size)

            if fill.side == SIDE_BUY:
                metrics.buy_fills += 1
                metrics.buy_volume += fill.size
            else:
                metrics.sell_fills += 1
                metrics.sell_volume += fill.size

            # Call model.update() for stateful models (e.g., TPSL-BS)
            if hasattr(self.quote_model, 'update'):
                fill_info = {"side": fill.side, "price": fill.price, "size": fill.size}
                state = self._build_state(fill.timestamp_ns) if self._current_book else None
                self.quote_model.update(state, None, fill_info)

    def _update_quotes(self, timestamp_ns: int) -> None:
        """Update strategy quotes based on current market."""
        if not self._current_book or self._current_mid_price <= 0:
            return

        # Build strategy state
        state = self._build_state(timestamp_ns)

        # Compute quote
        quote_result = self.quote_model.compute(state)

        if not quote_result.should_quote:
            return

        # Compute size
        size_result = self.size_model.compute(state, quote_result)

        # Cancel all existing orders
        self.simulator.cancel_all()

        # Place new orders if within position limits
        current_pos = self.simulator.position.size

        # Buy order (if not at max long position)
        if quote_result.bid_price > 0 and size_result.bid_size > 0:
            if current_pos + size_result.bid_size <= self.config.max_position:
                self.simulator.place_limit_order(
                    side=SIDE_BUY,
                    price=quote_result.bid_price,
                    size=size_result.bid_size,
                    timestamp_ns=timestamp_ns,
                )

        # Sell order (if not at max short position)
        if quote_result.ask_price > 0 and size_result.ask_size > 0:
            if current_pos - size_result.ask_size >= -self.config.max_position:
                self.simulator.place_limit_order(
                    side=SIDE_SELL,
                    price=quote_result.ask_price,
                    size=size_result.ask_size,
                    timestamp_ns=timestamp_ns,
                )

    def _build_state(self, timestamp_ns: int) -> StrategyState:
        """Build strategy state for quote computation."""
        book = self._current_book

        # Extract bid/ask levels as tuples
        bids = [(level.price, level.size) for level in book.bids[:10]]
        asks = [(level.price, level.size) for level in book.asks[:10]]

        market = MarketState(
            asset_id=self._get_asset_id(),
            timestamp_ns=timestamp_ns,
            mid_price=self._current_mid_price,
            spread=book.spread,
            bids=bids,
            asks=asks,
            last_trade_price=self._current_mid_price,
            last_trade_size=0.0,
            last_trade_side=0,
        )

        pos = self.simulator.position
        position = PositionState(
            asset_id=self._get_asset_id(),
            position=pos.size,
            avg_entry_price=pos.avg_entry_price,
            unrealized_pnl=self.simulator.get_unrealized_pnl(self._current_mid_price),
            realized_pnl=pos.realized_pnl,
        )

        # Calculate total equity
        total_pnl = self.simulator.get_total_pnl(self._current_mid_price)
        total_equity = self.config.initial_equity + total_pnl

        # Update external prices from RTDS
        if self.rtds_loader and self._current_symbol:
            rtds_price = self.rtds_loader.get_price_ns(self._current_symbol, timestamp_ns)
            if rtds_price is not None:
                self._external_prices[self._current_symbol] = ExternalPriceState(
                    symbol=self._current_symbol,
                    price=rtds_price,
                    bid=rtds_price,  # RTDS provides single price
                    ask=rtds_price,
                    timestamp_ns=timestamp_ns,
                )

        return StrategyState(
            market=market,
            external_prices=self._external_prices,
            position=position,
            total_equity=total_equity,
        )

    def _get_asset_id(self) -> str:
        """Get current asset ID for strategy state."""
        # For now, return symbol as asset ID
        return self.config.side


def print_summary(result: DatalakeBacktestResult) -> None:
    """Print summary of backtest results."""
    print("\n" + "=" * 80)
    print("DATALAKE BACKTEST RESULTS SUMMARY")
    print("=" * 80)

    print("\nConfiguration:")
    print(f"  Symbols: {result.config.symbols or 'all'}")
    print(f"  Date Range: {result.config.start_date or 'start'} to {result.config.end_date or 'end'}")
    print(f"  Side: {result.config.side}")
    print(f"  Maker Fee: {result.config.maker_fee * 100:.2f}%")

    print("\nAggregate Metrics:")
    print(f"  Total Markets: {result.total_markets}")
    print(f"  Total PnL: ${result.total_pnl:,.2f}")
    print(f"  Maker Rebate (20%): ${result.total_maker_rebate:,.2f}")
    print(f"  PnL After Rebate: ${result.pnl_after_rebate:,.2f}")
    print(f"  Total Fills: {result.total_fills:,}")
    print(f"  Total Volume: {result.total_volume:,.2f}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: ${result.max_drawdown:,.2f}")

    # Per-symbol breakdown
    symbol_pnl: dict[str, float] = {}
    symbol_rebate: dict[str, float] = {}
    symbol_fills: dict[str, int] = {}

    for slug, mr in result.market_results.items():
        symbol = mr.info.symbol
        symbol_pnl[symbol] = symbol_pnl.get(symbol, 0) + mr.metrics.total_pnl
        symbol_rebate[symbol] = symbol_rebate.get(symbol, 0) + mr.metrics.maker_rebate
        symbol_fills[symbol] = symbol_fills.get(symbol, 0) + mr.metrics.total_fills

    if symbol_pnl:
        print("\nPer-Symbol Breakdown:")
        for symbol in sorted(symbol_pnl.keys()):
            pnl = symbol_pnl[symbol]
            rebate = symbol_rebate[symbol]
            net = pnl + rebate
            fills = symbol_fills[symbol]
            print(f"  {symbol.upper()}: PnL=${pnl:,.2f}, Rebate=${rebate:,.2f}, "
                  f"Net=${net:,.2f}, Fills={fills:,}")

    print("=" * 80)
