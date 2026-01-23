"""
Backtest Engine

Runs backtests using historical data and strategy models.
Supports both orderbook-based and trades-based backtesting.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
import yaml

from backtest.data_loader import (
    DataLoader,
    LocalParquetLoader,
    MarketData,
    S3TradesLoader,
    Trade,
    generate_synthetic_data,
)
from backtest.simulator import Fill, MarketSimulator
from strategy.models.base import QuoteModel, QuoteResult, SizeModel, SizeResult, StrategyState
from strategy.models.quote import InventoryAdjustedQuoteModel
from strategy.models.size import InventoryBasedSizeModel
from strategy.shm.types import ExternalPriceState, MarketState, PositionState

logger = structlog.get_logger()


@dataclass
class BacktestConfig:
    """Configuration for backtest."""

    data_dir: str = "data/snapshots"
    use_synthetic_data: bool = True
    n_synthetic_ticks: int = 10000

    # Simulation
    maker_fee: float = 0.0
    taker_fee: float = 0.001

    # Strategy
    max_position: float = 100.0
    quote_refresh_ticks: int = 10


@dataclass
class TradesBacktestConfig:
    """Configuration for trades-based backtest."""

    # Data source
    s3_bucket: str = "an-trading-research"
    s3_prefix: str = "polymarket/trades/crypto/1h"
    local_cache_dir: Optional[str] = "data/s3_cache"
    local_data_dir: Optional[str] = None  # Use local files instead of S3

    # Asset filter
    assets: Optional[List[str]] = None  # None = all assets
    start_date: Optional[str] = None  # YYYY-MM format
    end_date: Optional[str] = None

    # Simulation
    maker_fee: float = 0.0
    taker_fee: float = 0.001

    # Strategy
    max_position: float = 100.0
    quote_refresh_interval_ns: int = 60_000_000_000  # 60 seconds
    base_spread: float = 0.02  # 2% spread
    base_size: float = 10.0

    @classmethod
    def from_yaml(cls, path: str = "data/config/backtest.yaml") -> "TradesBacktestConfig":
        """Create config from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            # Return defaults if config doesn't exist
            return cls()

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        data_cfg = cfg.get("data", {})
        filter_cfg = cfg.get("filter", {})
        sim_cfg = cfg.get("simulation", {})
        strat_cfg = cfg.get("strategy", {})

        return cls(
            s3_bucket=data_cfg.get("s3_bucket", "an-trading-research"),
            s3_prefix=data_cfg.get("s3_prefix", "polymarket/trades/crypto/1h"),
            local_cache_dir=data_cfg.get("local_cache_dir", "data/s3_cache"),
            local_data_dir=data_cfg.get("local_data_dir"),
            assets=filter_cfg.get("assets"),
            start_date=filter_cfg.get("start_date"),
            end_date=filter_cfg.get("end_date"),
            maker_fee=float(sim_cfg.get("maker_fee", 0.0)),
            taker_fee=float(sim_cfg.get("taker_fee", 0.001)),
            max_position=float(strat_cfg.get("max_position", 100.0)),
            quote_refresh_interval_ns=int(
                float(strat_cfg.get("quote_interval_sec", 0.5)) * 1_000_000_000
            ),
            base_spread=float(strat_cfg.get("base_spread", 0.02)),
            base_size=float(strat_cfg.get("base_size", 10.0)),
        )


@dataclass
class BacktestMetrics:
    """Metrics from a backtest run."""

    total_ticks: int = 0
    total_fills: int = 0
    total_volume: float = 0.0
    total_fees: float = 0.0

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0

    max_drawdown: float = 0.0
    max_position: float = 0.0

    sharpe_ratio: float = 0.0
    win_rate: float = 0.0

    pnl_history: List[float] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    config: BacktestConfig
    metrics: BacktestMetrics
    fills: List[Fill]


class BacktestEngine:
    """
    Engine for running backtests.

    Uses the same strategy models as live trading.
    """

    def __init__(
        self,
        config: BacktestConfig,
        quote_model: Optional[QuoteModel] = None,
        size_model: Optional[SizeModel] = None,
    ):
        self.config = config
        self.logger = logger.bind(component="backtest")

        # Models
        self.quote_model = quote_model or InventoryAdjustedQuoteModel()
        self.size_model = size_model or InventoryBasedSizeModel()

        # Simulator
        self.simulator = MarketSimulator(
            maker_fee=config.maker_fee,
            taker_fee=config.taker_fee,
        )

        # State
        self._current_quote: Optional[Tuple[QuoteResult, SizeResult]] = None
        self._ticks_since_quote: int = 0

    def run(self) -> BacktestResult:
        """
        Run the backtest.

        Returns:
            BacktestResult with metrics and fills
        """
        self.logger.info("Starting backtest", config=self.config)

        self.simulator.reset()
        metrics = BacktestMetrics()
        all_fills: List[Fill] = []

        # Get data iterator
        if self.config.use_synthetic_data:
            data_iter = generate_synthetic_data(
                n_ticks=self.config.n_synthetic_ticks,
            )
        else:
            loader = DataLoader(self.config.data_dir)
            n_files = loader.load_files()
            self.logger.info("Loaded data files", count=n_files)
            data_iter = iter(loader)

        # Track PnL for metrics
        pnl_samples: List[float] = []
        peak_pnl: float = 0.0
        last_data: Optional[MarketData] = None

        # Main backtest loop
        for data in data_iter:
            last_data = data
            # Process tick
            fills = self._process_tick(data)

            # Update metrics
            metrics.total_ticks += 1

            for fill in fills:
                all_fills.append(fill)
                metrics.total_fills += 1
                metrics.total_volume += fill.size

            # Track position
            pos_size = abs(self.simulator.position.size)
            if pos_size > metrics.max_position:
                metrics.max_position = pos_size

            # Track PnL
            mid_price = data.orderbook.mid_price
            if mid_price > 0:
                total_pnl = self.simulator.get_total_pnl(mid_price)
                pnl_samples.append(total_pnl)
                metrics.pnl_history.append(total_pnl)

                # Track drawdown
                if total_pnl > peak_pnl:
                    peak_pnl = total_pnl
                drawdown = peak_pnl - total_pnl
                if drawdown > metrics.max_drawdown:
                    metrics.max_drawdown = drawdown

        # Final metrics
        if last_data:
            mid_price = last_data.orderbook.mid_price
            metrics.realized_pnl = self.simulator.position.realized_pnl
            metrics.unrealized_pnl = self.simulator.get_unrealized_pnl(mid_price)
            metrics.total_pnl = self.simulator.get_total_pnl(mid_price)
            metrics.total_fees = self.simulator.position.total_fees

        # Calculate Sharpe ratio
        if len(pnl_samples) > 1:
            returns = np.diff(pnl_samples)
            if np.std(returns) > 0:
                metrics.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365.25 * 24)

        # Calculate win rate
        winning_fills = sum(1 for f in all_fills if self._is_winning_fill(f))
        if all_fills:
            metrics.win_rate = winning_fills / len(all_fills)

        self.logger.info(
            "Backtest complete",
            ticks=metrics.total_ticks,
            fills=metrics.total_fills,
            total_pnl=metrics.total_pnl,
            sharpe=metrics.sharpe_ratio,
        )

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            fills=all_fills,
        )

    def _process_tick(self, data: MarketData) -> List[Fill]:
        """Process a single tick."""
        # Check for fills on existing orders
        fills = self.simulator.process_tick(data)

        # Update quotes if needed
        self._ticks_since_quote += 1
        if self._ticks_since_quote >= self.config.quote_refresh_ticks:
            self._update_quotes(data)
            self._ticks_since_quote = 0

        return fills

    def _update_quotes(self, data: MarketData) -> None:
        """Update strategy quotes."""
        # Build strategy state
        state = self._build_state(data)

        # Compute quote
        quote_result = self.quote_model.compute(state)

        if not quote_result.should_quote:
            self.simulator.cancel_all()
            self._current_quote = None
            return

        # Compute size
        size_result = self.size_model.compute(state, quote_result)

        # Cancel existing orders
        self.simulator.cancel_all()

        # Place new orders
        if size_result.bid_size > 0 and quote_result.bid_price > 0:
            self.simulator.place_limit_order(
                side=1,
                price=quote_result.bid_price,
                size=size_result.bid_size,
                timestamp_ns=data.timestamp_ns,
            )

        if size_result.ask_size > 0 and quote_result.ask_price > 0:
            self.simulator.place_limit_order(
                side=-1,
                price=quote_result.ask_price,
                size=size_result.ask_size,
                timestamp_ns=data.timestamp_ns,
            )

        self._current_quote = (quote_result, size_result)

    def _build_state(self, data: MarketData) -> StrategyState:
        """Build strategy state from market data."""
        # Convert orderbook
        market = MarketState(
            asset_id="backtest",
            timestamp_ns=data.timestamp_ns,
            mid_price=data.orderbook.mid_price,
            spread=data.orderbook.spread,
            bids=[(level.price, level.size) for level in data.orderbook.bids],
            asks=[(level.price, level.size) for level in data.orderbook.asks],
            last_trade_price=data.orderbook.mid_price,
            last_trade_size=0,
            last_trade_side=0,
        )

        # Convert external prices
        external_prices: Dict[str, ExternalPriceState] = {}
        for symbol, ext in data.external_prices.items():
            external_prices[symbol] = ExternalPriceState(
                symbol=ext.symbol,
                price=ext.price,
                bid=ext.bid,
                ask=ext.ask,
                timestamp_ns=ext.timestamp_ns,
            )

        # Build position state
        pos = self.simulator.position
        position = PositionState(
            asset_id="backtest",
            position=pos.size,
            avg_entry_price=pos.avg_entry_price,
            unrealized_pnl=self.simulator.get_unrealized_pnl(data.orderbook.mid_price),
            realized_pnl=pos.realized_pnl,
        )

        return StrategyState(
            market=market,
            external_prices=external_prices,
            position=position,
        )

    def _is_winning_fill(self, fill: Fill) -> bool:
        """Check if a fill was profitable (simplified)."""
        # This is a simplified check - in reality you'd track round-trip trades
        return fill.is_maker


@dataclass
class TradesBacktestResult:
    """Result of a trades-based backtest run."""

    config: TradesBacktestConfig
    metrics: BacktestMetrics
    fills: List[Fill]
    asset_metrics: Dict[str, BacktestMetrics]


class TradesBacktestEngine:
    """
    Engine for running trades-based backtests.

    Uses crossing logic for fills:
        - trade_price < our_bid → bid fills (we buy)
        - trade_price > our_ask → ask fills (we sell)
    """

    def __init__(
        self,
        config: TradesBacktestConfig,
        quote_model: Optional[QuoteModel] = None,
        size_model: Optional[SizeModel] = None,
    ):
        self.config = config
        self.logger = logger.bind(component="trades_backtest")

        # Models
        self.quote_model = quote_model or InventoryAdjustedQuoteModel()
        self.size_model = size_model or InventoryBasedSizeModel()

        # Simulator
        self.simulator = MarketSimulator(
            maker_fee=config.maker_fee,
            taker_fee=config.taker_fee,
        )

        # State
        self._last_quote_time_ns: int = 0
        self._current_mid_price: float = 0.0
        self._current_quote: Optional[Tuple[QuoteResult, SizeResult]] = None

    def _load_trades(self) -> pd.DataFrame:
        """Load trades data from S3 or local files."""
        if self.config.local_data_dir:
            loader = LocalParquetLoader(self.config.local_data_dir)
            return loader.load_trades()
        else:
            loader = S3TradesLoader(
                bucket=self.config.s3_bucket,
                prefix=self.config.s3_prefix,
                local_cache_dir=self.config.local_cache_dir,
            )
            return loader.load_trades(
                asset=None,  # Load all, filter later
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )

    def run(self) -> TradesBacktestResult:
        """
        Run the trades-based backtest.

        Returns:
            TradesBacktestResult with metrics and fills
        """
        self.logger.info("Starting trades-based backtest", config=self.config)

        # Load all trades data
        trades_df = self._load_trades()
        if trades_df.empty:
            self.logger.error("No trades data loaded")
            return TradesBacktestResult(
                config=self.config,
                metrics=BacktestMetrics(),
                fills=[],
                asset_metrics={},
            )

        self.logger.info("Loaded trades data", rows=len(trades_df))

        # Filter assets if specified
        if self.config.assets:
            trades_df = trades_df[trades_df["asset"].isin(self.config.assets)]
            self.logger.info("Filtered to assets", assets=self.config.assets, rows=len(trades_df))

        # Get list of assets to process
        assets = trades_df["asset"].unique().tolist()
        self.logger.info("Processing assets", count=len(assets), assets=assets[:10])

        # Run backtest for each asset
        asset_metrics: Dict[str, BacktestMetrics] = {}
        all_fills: List[Fill] = []

        for asset in assets:
            asset_df = trades_df[trades_df["asset"] == asset].copy()
            result = self._run_single_asset(asset, asset_df)
            asset_metrics[asset] = result.metrics
            all_fills.extend(result.fills)

        # Aggregate metrics
        total_metrics = self._aggregate_metrics(asset_metrics)
        total_metrics.pnl_history = []  # Don't aggregate history

        self.logger.info(
            "Backtest complete",
            assets=len(assets),
            total_fills=total_metrics.total_fills,
            total_pnl=total_metrics.total_pnl,
        )

        return TradesBacktestResult(
            config=self.config,
            metrics=total_metrics,
            fills=all_fills,
            asset_metrics=asset_metrics,
        )

    def _run_single_asset(self, asset: str, trades_df: pd.DataFrame) -> BacktestResult:
        """Run backtest for a single asset."""
        self.simulator.reset()
        self._last_quote_time_ns = 0
        self._current_mid_price = 0.0
        self._current_quote = None

        metrics = BacktestMetrics()
        asset_fills: List[Fill] = []

        # Find time and price columns
        time_col = self._find_column(trades_df, ["timestamp", "time", "ts", "timestamp_ns"])
        price_col = self._find_column(trades_df, ["price", "trade_price"])
        size_col = self._find_column(trades_df, ["size", "amount", "quantity", "qty"])

        if not time_col or not price_col:
            self.logger.warning("Missing required columns", asset=asset)
            return BacktestResult(
                config=BacktestConfig(),
                metrics=metrics,
                fills=[],
            )

        # Track PnL
        pnl_samples: List[float] = []
        peak_pnl: float = 0.0

        # Pre-extract numpy arrays for performance (10-50x faster than iterrows)
        timestamps_raw = trades_df[time_col].values
        prices = trades_df[price_col].values.astype(np.float64)
        if size_col and size_col in trades_df.columns:
            sizes = trades_df[size_col].values.astype(np.float64)
        else:
            sizes = np.ones(len(prices), dtype=np.float64)

        n_trades = len(prices)

        for i in range(n_trades):
            timestamp_ns = self._to_timestamp_ns(timestamps_raw[i])
            trade_price = prices[i]
            trade_size = sizes[i]

            trade = Trade(
                timestamp_ns=timestamp_ns,
                price=trade_price,
                size=trade_size,
                side=0,  # Unknown, not needed for crossing logic
            )

            # Update mid price estimate
            self._current_mid_price = trade_price

            # Process trade through simulator
            fills = self.simulator.process_trade(trade)

            # Update quotes if needed
            if timestamp_ns - self._last_quote_time_ns >= self.config.quote_refresh_interval_ns:
                self._update_quotes(asset, timestamp_ns)
                self._last_quote_time_ns = timestamp_ns

            # Update metrics
            metrics.total_ticks += 1

            for fill in fills:
                asset_fills.append(fill)
                metrics.total_fills += 1
                metrics.total_volume += fill.size

            # Track position
            pos_size = abs(self.simulator.position.size)
            if pos_size > metrics.max_position:
                metrics.max_position = pos_size

            # Track PnL periodically
            if metrics.total_ticks % 100 == 0 and self._current_mid_price > 0:
                total_pnl = self.simulator.get_total_pnl(self._current_mid_price)
                pnl_samples.append(total_pnl)
                metrics.pnl_history.append(total_pnl)

                # Track drawdown
                if total_pnl > peak_pnl:
                    peak_pnl = total_pnl
                drawdown = peak_pnl - total_pnl
                if drawdown > metrics.max_drawdown:
                    metrics.max_drawdown = drawdown

        # Final metrics
        if self._current_mid_price > 0:
            metrics.realized_pnl = self.simulator.position.realized_pnl
            metrics.unrealized_pnl = self.simulator.get_unrealized_pnl(self._current_mid_price)
            metrics.total_pnl = self.simulator.get_total_pnl(self._current_mid_price)
            metrics.total_fees = self.simulator.position.total_fees

        # Calculate Sharpe ratio
        if len(pnl_samples) > 1:
            returns = np.diff(pnl_samples)
            if np.std(returns) > 0:
                metrics.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365.25 * 24)

        return BacktestResult(
            config=BacktestConfig(),
            metrics=metrics,
            fills=asset_fills,
        )

    def _update_quotes(self, asset: str, timestamp_ns: int) -> None:
        """Update strategy quotes based on current price."""
        if self._current_mid_price <= 0:
            return

        # Build strategy state
        state = self._build_state(asset, timestamp_ns)

        # Compute quote
        quote_result = self.quote_model.compute(state)

        if not quote_result.should_quote:
            self.simulator.cancel_all()
            self._current_quote = None
            return

        # Compute size
        size_result = self.size_model.compute(state, quote_result)

        # Cancel existing orders
        self.simulator.cancel_all()

        # Place new orders
        if size_result.bid_size > 0 and quote_result.bid_price > 0:
            self.simulator.place_limit_order(
                side=1,
                price=quote_result.bid_price,
                size=size_result.bid_size,
                timestamp_ns=timestamp_ns,
            )

        if size_result.ask_size > 0 and quote_result.ask_price > 0:
            self.simulator.place_limit_order(
                side=-1,
                price=quote_result.ask_price,
                size=size_result.ask_size,
                timestamp_ns=timestamp_ns,
            )

        self._current_quote = (quote_result, size_result)

    def _build_state(self, asset: str, timestamp_ns: int) -> StrategyState:
        """Build strategy state for quote computation."""
        market = MarketState(
            asset_id=asset,
            timestamp_ns=timestamp_ns,
            mid_price=self._current_mid_price,
            spread=self._current_mid_price * self.config.base_spread,
            bids=[],
            asks=[],
            last_trade_price=self._current_mid_price,
            last_trade_size=0,
            last_trade_side=0,
        )

        pos = self.simulator.position
        position = PositionState(
            asset_id=asset,
            position=pos.size,
            avg_entry_price=pos.avg_entry_price,
            unrealized_pnl=self.simulator.get_unrealized_pnl(self._current_mid_price),
            realized_pnl=pos.realized_pnl,
        )

        return StrategyState(
            market=market,
            position=position,
        )

    def _aggregate_metrics(self, asset_metrics: Dict[str, BacktestMetrics]) -> BacktestMetrics:
        """Aggregate metrics from all assets."""
        total = BacktestMetrics()

        for metrics in asset_metrics.values():
            total.total_ticks += metrics.total_ticks
            total.total_fills += metrics.total_fills
            total.total_volume += metrics.total_volume
            total.total_fees += metrics.total_fees
            total.realized_pnl += metrics.realized_pnl
            total.unrealized_pnl += metrics.unrealized_pnl
            total.total_pnl += metrics.total_pnl

            if metrics.max_position > total.max_position:
                total.max_position = metrics.max_position
            if metrics.max_drawdown > total.max_drawdown:
                total.max_drawdown = metrics.max_drawdown

        # Average Sharpe and win rate
        sharpes = [m.sharpe_ratio for m in asset_metrics.values() if m.sharpe_ratio != 0]
        win_rates = [m.win_rate for m in asset_metrics.values() if m.win_rate != 0]

        if sharpes:
            total.sharpe_ratio = np.mean(sharpes)
        if win_rates:
            total.win_rate = np.mean(win_rates)

        return total

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find a column from a list of candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        # Check case-insensitive
        for col in df.columns:
            for cand in candidates:
                if col.lower() == cand.lower():
                    return col
        return None

    def _to_timestamp_ns(self, val) -> int:
        """Convert various timestamp formats to nanoseconds."""
        if val is None:
            return 0
        if isinstance(val, (int, float)):
            if val > 1e18:
                return int(val)
            elif val > 1e15:
                return int(val * 1000)
            elif val > 1e12:
                return int(val * 1_000_000)
            else:
                return int(val * 1_000_000_000)
        if isinstance(val, pd.Timestamp):
            return int(val.value)
        return 0


def main() -> None:
    """Run a sample backtest."""
    config = BacktestConfig(
        use_synthetic_data=True,
        n_synthetic_ticks=10000,
    )

    engine = BacktestEngine(config)
    result = engine.run()

    print(f"\nBacktest Results:")
    print(f"  Total Ticks: {result.metrics.total_ticks}")
    print(f"  Total Fills: {result.metrics.total_fills}")
    print(f"  Total Volume: {result.metrics.total_volume:.2f}")
    print(f"  Total PnL: ${result.metrics.total_pnl:.2f}")
    print(f"  Realized PnL: ${result.metrics.realized_pnl:.2f}")
    print(f"  Unrealized PnL: ${result.metrics.unrealized_pnl:.2f}")
    print(f"  Max Drawdown: ${result.metrics.max_drawdown:.2f}")
    print(f"  Max Position: {result.metrics.max_position:.2f}")
    print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    print(f"  Win Rate: {result.metrics.win_rate:.1%}")


def run_trades_backtest(config_path: str = "data/config/backtest.yaml") -> None:
    """Run a trades-based backtest."""
    config = TradesBacktestConfig.from_yaml(config_path)

    engine = TradesBacktestEngine(config)
    result = engine.run()

    print(f"\nTrades-Based Backtest Results:")
    print(f"  Assets Processed: {len(result.asset_metrics)}")
    print(f"  Total Ticks: {result.metrics.total_ticks}")
    print(f"  Total Fills: {result.metrics.total_fills}")
    print(f"  Total Volume: {result.metrics.total_volume:.2f}")
    print(f"  Total PnL: ${result.metrics.total_pnl:.2f}")
    print(f"  Realized PnL: ${result.metrics.realized_pnl:.2f}")
    print(f"  Unrealized PnL: ${result.metrics.unrealized_pnl:.2f}")
    print(f"  Max Drawdown: ${result.metrics.max_drawdown:.2f}")
    print(f"  Max Position: {result.metrics.max_position:.2f}")
    print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")

    print(f"\nPer-Asset Results:")
    for asset, metrics in sorted(result.asset_metrics.items()):
        print(f"  {asset}: PnL=${metrics.total_pnl:.2f}, Fills={metrics.total_fills}")


if __name__ == "__main__":
    main()
