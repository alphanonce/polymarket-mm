"""
Orderbook-Based Backtest Engine

Runs tick-by-tick backtests using orderbook CSV data.
Uses crossing logic: fills when our quote crosses the market BBO.
"""
import time as time_module
from dataclasses import dataclass, field

import pandas as pd
import structlog

from backtest.data_loader import MarketData, OrderBook, PriceLevel
from backtest.orderbook_loader import OrderbookCSVLoader
from backtest.simulator import (
    SIDE_BUY,
    SIDE_SELL,
    Fill,
    MarketSimulator,
    Position,
)
from strategy.models.base import (
    ExternalPriceState,
    MarketState,
    PositionState,
    QuoteModel,
    QuoteResult,
    SizeModel,
    SizeResult,
    StrategyState,
)

logger = structlog.get_logger()


@dataclass
class OrderbookBacktestConfig:
    """Configuration for orderbook-based backtest."""

    # Data source
    csv_path: str
    asset: str  # "btc", "eth", "sol", "xrp"
    side: str = "up"  # Which side to trade ("up" or "down")

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


@dataclass
class BacktestMetrics:
    """Metrics collected during backtest."""

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
    max_position: float = 0.0
    min_position: float = 0.0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    total_ticks: int = 0
    valid_ticks: int = 0
    avg_spread: float = 0.0
    quotes_placed: int = 0


@dataclass
class OrderbookBacktestResult:
    """Result of an orderbook-based backtest."""

    config: OrderbookBacktestConfig
    metrics: BacktestMetrics
    fills: list[Fill]
    final_position: Position
    pnl_timeseries: list[tuple[int, float]] = field(default_factory=list)


class OrderbookBacktestEngine:
    """
    Orderbook-based backtest engine.

    Processes tick-by-tick orderbook data and simulates fill
    execution based on crossing logic.

    Fill Logic:
        - BUY order fills if: market_ask < order_price (crossing)
        - SELL order fills if: market_bid > order_price (crossing)
    """

    def __init__(
        self,
        config: OrderbookBacktestConfig,
        quote_model: QuoteModel,
        size_model: SizeModel,
    ):
        self.config = config
        self.quote_model = quote_model
        self.size_model = size_model
        self.logger = logger.bind(
            component="orderbook_backtest",
            asset=config.asset,
            side=config.side,
        )

        # Initialize simulator
        self.simulator = MarketSimulator(
            maker_fee=config.maker_fee,
            taker_fee=config.taker_fee,
        )

        # State tracking
        self._last_quote_time_ns: int = 0
        self._current_mid_price: float = 0.0
        self._current_quote: tuple[QuoteResult, SizeResult] | None = None

        # External prices (for TpBS model)
        self._external_prices: dict[str, ExternalPriceState] = {}

    def set_external_price(self, symbol: str, price: float, timestamp_ns: int = 0) -> None:
        """Set external price (e.g., Binance spot price) for strategy."""
        self._external_prices[symbol] = ExternalPriceState(
            symbol=symbol,
            price=price,
            bid=price,
            ask=price,
            timestamp_ns=timestamp_ns,
        )

    def run(self) -> OrderbookBacktestResult:
        """
        Run the backtest.

        Returns:
            OrderbookBacktestResult with metrics and fills
        """
        start_time = time_module.time()

        # Reset state
        self.simulator.reset()
        self._last_quote_time_ns = 0
        self._current_mid_price = 0.0
        self._current_quote = None

        # Load data
        loader = OrderbookCSVLoader(self.config.csv_path)
        df = loader.load()

        # Filter to target side
        df = df[df['side'] == self.config.side].reset_index(drop=True)

        self.logger.info(
            "Starting orderbook backtest",
            total_ticks=len(df),
            side=self.config.side,
        )

        # Initialize metrics
        metrics = BacktestMetrics()
        metrics.total_ticks = len(df)
        pnl_timeseries: list[tuple[int, float]] = []
        spread_sum = 0.0
        spread_count = 0

        # Main loop
        for idx, row in df.iterrows():
            timestamp_ns = row['timestamp_ns']
            best_bid = row['best_bid'] if pd.notna(row['best_bid']) else None
            best_ask = row['best_ask'] if pd.notna(row['best_ask']) else None
            mid = row['mid'] if pd.notna(row['mid']) else None

            # Skip ticks without valid data
            if best_bid is None or best_ask is None:
                continue

            metrics.valid_ticks += 1

            # Track spread
            spread = best_ask - best_bid
            spread_sum += spread
            spread_count += 1

            # Update mid price
            self._current_mid_price = mid if mid is not None else (best_bid + best_ask) / 2

            # Build MarketData for simulator
            orderbook = OrderBook(
                timestamp_ns=timestamp_ns,
                bids=[PriceLevel(price=best_bid, size=1000.0)],
                asks=[PriceLevel(price=best_ask, size=1000.0)],
            )
            data = MarketData(timestamp_ns=timestamp_ns, orderbook=orderbook)

            # Process tick (check for fills)
            fills = self.simulator.process_tick(data)

            # Update fill metrics
            for fill in fills:
                metrics.total_fills += 1
                metrics.total_volume += fill.size

                if fill.side == SIDE_BUY:
                    metrics.buy_fills += 1
                    metrics.buy_volume += fill.size
                else:
                    metrics.sell_fills += 1
                    metrics.sell_volume += fill.size

            # Track position extremes
            pos_size = self.simulator.position.size
            metrics.max_position = max(metrics.max_position, pos_size)
            metrics.min_position = min(metrics.min_position, pos_size)

            # Update quotes if needed
            if timestamp_ns - self._last_quote_time_ns >= self.config.quote_refresh_interval_ns:
                self._update_quotes(timestamp_ns, best_bid, best_ask)
                self._last_quote_time_ns = timestamp_ns

            # Track PnL
            total_pnl = self.simulator.get_total_pnl(self._current_mid_price)
            metrics.peak_pnl = max(metrics.peak_pnl, total_pnl)
            drawdown = metrics.peak_pnl - total_pnl
            metrics.max_drawdown = max(metrics.max_drawdown, drawdown)

            # Sample PnL timeseries (every 10k ticks)
            if idx % 10000 == 0:
                pnl_timeseries.append((timestamp_ns, total_pnl))

            # Progress logging
            if idx > 0 and idx % self.config.log_interval == 0:
                self.logger.info(
                    "Backtest progress",
                    progress=f"{idx / len(df) * 100:.1f}%",
                    fills=metrics.total_fills,
                    pnl=f"${total_pnl:.2f}",
                    position=f"{pos_size:.2f}",
                )

        # Final metrics
        final_pnl = self.simulator.get_total_pnl(self._current_mid_price)
        metrics.total_pnl = final_pnl
        metrics.realized_pnl = self.simulator.position.realized_pnl
        metrics.unrealized_pnl = self.simulator.get_unrealized_pnl(self._current_mid_price)
        metrics.total_fees = self.simulator.position.total_fees
        metrics.avg_spread = spread_sum / spread_count if spread_count > 0 else 0.0

        elapsed = time_module.time() - start_time

        self.logger.info(
            "Backtest completed",
            elapsed_seconds=f"{elapsed:.2f}",
            total_pnl=f"${metrics.total_pnl:.2f}",
            total_fills=metrics.total_fills,
            total_volume=f"{metrics.total_volume:.2f}",
            final_position=f"{self.simulator.position.size:.2f}",
        )

        return OrderbookBacktestResult(
            config=self.config,
            metrics=metrics,
            fills=self.simulator.fills,
            final_position=self.simulator.position,
            pnl_timeseries=pnl_timeseries,
        )

    def _update_quotes(
        self,
        timestamp_ns: int,
        best_bid: float,
        best_ask: float,
    ) -> None:
        """Update strategy quotes based on current market."""
        if self._current_mid_price <= 0:
            return

        # Build strategy state
        state = self._build_state(timestamp_ns, best_bid, best_ask)

        # Compute quote
        quote_result = self.quote_model.compute(state)

        # Compute size
        size_result = self.size_model.compute(state, quote_result)

        self._current_quote = (quote_result, size_result)

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

    def _build_state(
        self,
        timestamp_ns: int,
        best_bid: float,
        best_ask: float,
    ) -> StrategyState:
        """Build strategy state for quote computation."""
        market = MarketState(
            asset_id=self.config.asset,
            timestamp_ns=timestamp_ns,
            mid_price=self._current_mid_price,
            spread=best_ask - best_bid,
            bids=[(best_bid, 1000.0)],
            asks=[(best_ask, 1000.0)],
            last_trade_price=self._current_mid_price,
            last_trade_size=0.0,
            last_trade_side=0,
        )

        pos = self.simulator.position
        position = PositionState(
            asset_id=self.config.asset,
            position=pos.size,
            avg_entry_price=pos.avg_entry_price,
            unrealized_pnl=self.simulator.get_unrealized_pnl(self._current_mid_price),
            realized_pnl=pos.realized_pnl,
        )

        # Calculate total equity
        total_pnl = self.simulator.get_total_pnl(self._current_mid_price)
        total_equity = self.config.initial_equity + total_pnl

        return StrategyState(
            market=market,
            external_prices=self._external_prices,
            position=position,
            total_equity=total_equity,
        )


def run_multi_asset_backtest(
    csv_dir: str,
    assets: list[str],
    quote_model: QuoteModel,
    size_model: SizeModel,
    side: str = "up",
    config_overrides: dict | None = None,
) -> dict[str, OrderbookBacktestResult]:
    """
    Run backtest across multiple assets.

    Args:
        csv_dir: Directory containing CSV files
        assets: List of asset names
        quote_model: Quote model to use
        size_model: Size model to use
        side: Which side to trade ("up" or "down")
        config_overrides: Optional config overrides

    Returns:
        Dict mapping asset name to BacktestResult
    """
    results = {}

    for asset in assets:
        csv_path = f"{csv_dir}/orderbook_{asset}_jan13_19.csv"

        config_dict = {
            'csv_path': csv_path,
            'asset': asset,
            'side': side,
        }

        if config_overrides:
            config_dict.update(config_overrides)

        config = OrderbookBacktestConfig(**config_dict)

        engine = OrderbookBacktestEngine(
            config=config,
            quote_model=quote_model,
            size_model=size_model,
        )

        try:
            result = engine.run()
            results[asset] = result
        except Exception as e:
            logger.error("Failed to run backtest", asset=asset, error=str(e))

    return results


def print_summary(results: dict[str, OrderbookBacktestResult]) -> None:
    """Print summary of backtest results."""
    print("\n" + "=" * 80)
    print("ORDERBOOK BACKTEST RESULTS SUMMARY")
    print("=" * 80)

    total_pnl = 0.0
    total_fills = 0
    total_volume = 0.0

    for asset, result in results.items():
        m = result.metrics
        print(f"\n{asset.upper()} ({result.config.side}):")
        print(f"  PnL: ${m.total_pnl:,.2f} (Realized: ${m.realized_pnl:,.2f}, "
              f"Unrealized: ${m.unrealized_pnl:,.2f})")
        print(f"  Fills: {m.total_fills} (Buy: {m.buy_fills}, Sell: {m.sell_fills})")
        print(f"  Volume: {m.total_volume:,.2f} (Buy: {m.buy_volume:,.2f}, Sell: {m.sell_volume:,.2f})")
        print(f"  Position: {result.final_position.size:,.2f} "
              f"(Max: {m.max_position:,.2f}, Min: {m.min_position:,.2f})")
        print(f"  Max Drawdown: ${m.max_drawdown:,.2f}")
        print(f"  Avg Spread: {m.avg_spread * 100:.3f}%")
        print(f"  Valid Ticks: {m.valid_ticks:,} / {m.total_ticks:,}")

        total_pnl += m.total_pnl
        total_fills += m.total_fills
        total_volume += m.total_volume

    print("\n" + "-" * 80)
    print(f"TOTAL: PnL=${total_pnl:,.2f}, Fills={total_fills:,}, Volume={total_volume:,.2f}")
    print("=" * 80)
