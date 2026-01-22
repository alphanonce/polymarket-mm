"""
Time-Bucketed Backtest Engine

Prevents look-ahead bias by:
1. Aggregating data into configurable time buckets (default 100ms)
2. Using last BBO from time T to set quotes
3. Checking fills against quotes at time T+1
4. Only using BBO (best bid/offer), not full orderbook depth

Supports both orderbook BBO and trades data sources.
Uses pluggable QuoteModel and SizeModel for strategy logic.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
import structlog

from backtest.utils import (
    PositionTracker,
    compute_max_drawdown,
    compute_sharpe_ratio,
    to_timestamp_ns,
)
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
)
from strategy.models.size import (
    FixedSizeConfig,
    FixedSizeModel,
    InventoryBasedSizeConfig,
    InventoryBasedSizeModel,
)
from strategy.shm.types import MarketState, PositionState

logger = structlog.get_logger()

# Default bucket size: 100ms in nanoseconds
DEFAULT_BUCKET_SIZE_NS = 100_000_000


class DataSource(Enum):
    """Data source type for backtest."""

    ORDERBOOK = "orderbook"  # book_*.parquet.gz with bids/asks
    TRADES = "trades"  # trade events


@dataclass
class BucketedBacktestConfig:
    """Configuration for time-bucketed backtest."""

    # Time bucketing
    bucket_size_ns: int = DEFAULT_BUCKET_SIZE_NS  # 100ms default

    # Data source
    data_source: DataSource = DataSource.ORDERBOOK

    # Fees
    maker_fee: float = 0.0  # Maker fee (rebate if negative)


@dataclass
class BucketedFill:
    """Record of a fill in bucketed backtest."""

    bucket_ts: int  # Bucket timestamp (ns)
    side: int  # 1 = buy, -1 = sell
    price: float
    size: float
    position_after: float
    realized_pnl: float


@dataclass
class BucketedBacktestResult:
    """Result from bucketed backtest."""

    # Identification
    asset: str = ""
    slug: str = ""

    # Trade metrics
    n_buckets: int = 0
    n_fills: int = 0
    buy_fills: int = 0
    sell_fills: int = 0
    total_volume: float = 0.0

    # PnL metrics
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

    # Position metrics
    final_position: float = 0.0
    max_position: float = 0.0

    # Fill records
    fills: list[BucketedFill] = field(default_factory=list)

    # PnL timeseries for analysis
    pnl_history: np.ndarray = field(default_factory=lambda: np.array([]))


def create_default_quote_model(
    base_spread: float = 0.02,
    inventory_skew: float = 0.0,
    max_position: float = 100.0,
) -> QuoteModel:
    """Create a default quote model with given parameters."""
    if inventory_skew > 0:
        config = InventoryAdjustedQuoteConfig(
            base_spread=base_spread,
            inventory_skew=inventory_skew,
            max_inventory=max_position,
        )
        return InventoryAdjustedQuoteModel(
            config=config,
            normalization=NormalizationConfig(
                clamp_prices=False,  # Don't clamp in backtest
                use_dynamic_tick=False,
            ),
        )
    else:
        config = SpreadQuoteConfig(base_spread=base_spread)
        return SpreadQuoteModel(
            config=config,
            normalization=NormalizationConfig(
                clamp_prices=False,
                use_dynamic_tick=False,
            ),
        )


def create_default_size_model(
    base_size: float = 10.0,
    max_position: float = 100.0,
) -> SizeModel:
    """Create a default size model with given parameters."""
    config = InventoryBasedSizeConfig(
        base_size=base_size,
        max_position=max_position,
        max_size=base_size * 2,
        min_size=1.0,
    )
    return InventoryBasedSizeModel(
        config=config,
        normalization=NormalizationConfig(
            enforce_min_size=False,  # Don't enforce min in backtest
            round_sizes=False,
        ),
    )


def parse_orderbook_bbo(row: pd.Series) -> tuple[float, float]:
    """
    Extract best bid/ask from orderbook row.

    Expects 'bids' and 'asks' columns with JSON-encoded price levels.
    Returns (best_bid, best_ask).
    """
    import orjson

    best_bid = 0.0
    best_ask = 1.0  # Default to 1.0 for binary options

    # Parse bids (sorted descending by price - first is best)
    bids_json = row.get("bids", "[]")
    if bids_json and bids_json != "[]":
        try:
            bids = orjson.loads(bids_json) if isinstance(bids_json, (str, bytes)) else bids_json
            if bids:
                best_bid = float(bids[0][0])
        except (ValueError, IndexError, TypeError):
            pass

    # Parse asks (sorted ascending by price - first is best)
    asks_json = row.get("asks", "[]")
    if asks_json and asks_json != "[]":
        try:
            asks = orjson.loads(asks_json) if isinstance(asks_json, (str, bytes)) else asks_json
            if asks:
                best_ask = float(asks[0][0])
        except (ValueError, IndexError, TypeError):
            pass

    return best_bid, best_ask


def aggregate_orderbook_buckets(df: pd.DataFrame, bucket_size_ns: int) -> pd.DataFrame:
    """
    Aggregate orderbook data to BBO per bucket.

    Uses the LAST observation in each bucket (most recent state).

    Args:
        df: DataFrame with 'ts', 'bids', 'asks' columns
        bucket_size_ns: Bucket size in nanoseconds

    Returns:
        DataFrame with columns: bucket_ts, best_bid, best_ask
    """
    if df.empty:
        return pd.DataFrame(columns=["bucket_ts", "best_bid", "best_ask"])

    # Convert timestamp to nanoseconds if needed
    df = df.copy()
    if "ts" in df.columns:
        df["timestamp_ns"] = df["ts"].apply(to_timestamp_ns)
    elif "timestamp_ns" not in df.columns:
        raise ValueError("DataFrame must have 'ts' or 'timestamp_ns' column")

    # Create bucket column
    df["bucket_ts"] = (df["timestamp_ns"] // bucket_size_ns) * bucket_size_ns

    # Parse BBO for each row
    bbo_data = []
    for _, row in df.iterrows():
        best_bid, best_ask = parse_orderbook_bbo(row)
        bbo_data.append({"best_bid": best_bid, "best_ask": best_ask})

    bbo_df = pd.DataFrame(bbo_data)
    df["best_bid"] = bbo_df["best_bid"]
    df["best_ask"] = bbo_df["best_ask"]

    # Group by bucket and take last observation
    buckets = (
        df.groupby("bucket_ts")
        .agg({"best_bid": "last", "best_ask": "last"})
        .reset_index()
    )

    return buckets


def aggregate_trades_buckets(df: pd.DataFrame, bucket_size_ns: int) -> pd.DataFrame:
    """
    Aggregate trades data to implied BBO per bucket.

    Derives implied BBO from trades:
    - Use low/high as conservative BBO estimate
    - best_bid = low (lowest trade someone was willing to sell at)
    - best_ask = high (highest trade someone was willing to buy at)

    Note: This is a rough approximation since we don't know true BBO from trades.

    Args:
        df: DataFrame with 'ts'/'timestamp_ns', 'price' columns
        bucket_size_ns: Bucket size in nanoseconds

    Returns:
        DataFrame with columns: bucket_ts, best_bid, best_ask
    """
    if df.empty:
        return pd.DataFrame(columns=["bucket_ts", "best_bid", "best_ask"])

    df = df.copy()

    # Convert timestamp to nanoseconds if needed
    if "ts" in df.columns:
        df["timestamp_ns"] = df["ts"].apply(to_timestamp_ns)
    elif "timestamp" in df.columns:
        df["timestamp_ns"] = df["timestamp"].apply(to_timestamp_ns)
    elif "timestamp_ns" not in df.columns:
        raise ValueError("DataFrame must have 'ts', 'timestamp', or 'timestamp_ns' column")

    # Create bucket column
    df["bucket_ts"] = (df["timestamp_ns"] // bucket_size_ns) * bucket_size_ns

    # Get price column and convert to numeric
    price_col = "price"
    if price_col not in df.columns:
        for col in ["trade_price", "last_price"]:
            if col in df.columns:
                price_col = col
                break

    # Convert price to numeric (handle object dtype)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # Drop rows with invalid prices
    df = df.dropna(subset=[price_col])

    if df.empty:
        return pd.DataFrame(columns=["bucket_ts", "best_bid", "best_ask"])

    # Group by bucket and derive implied BBO from trade range
    buckets = (
        df.groupby("bucket_ts")
        .agg(
            best_bid=(price_col, "min"),  # Low = conservative bid estimate
            best_ask=(price_col, "max"),  # High = conservative ask estimate
        )
        .reset_index()
    )

    return buckets[["bucket_ts", "best_bid", "best_ask"]]


def create_strategy_state(
    bucket_ts: int,
    market_bid: float,
    market_ask: float,
    position: float,
    avg_entry: float,
    realized_pnl: float,
    asset_id: str = "",
) -> StrategyState:
    """
    Create a StrategyState from bucket data.

    Args:
        bucket_ts: Bucket timestamp in nanoseconds
        market_bid: Best bid price
        market_ask: Best ask price
        position: Current position
        avg_entry: Average entry price
        realized_pnl: Realized PnL
        asset_id: Asset identifier

    Returns:
        StrategyState for model input
    """
    mid_price = (market_bid + market_ask) / 2
    spread = market_ask - market_bid

    market = MarketState(
        asset_id=asset_id,
        timestamp_ns=bucket_ts,
        mid_price=mid_price,
        spread=spread,
        bids=[(market_bid, 1000.0)],  # Synthetic depth
        asks=[(market_ask, 1000.0)],
        last_trade_price=mid_price,
        last_trade_size=0.0,
        last_trade_side=0,
    )

    position_state = PositionState(
        asset_id=asset_id,
        position=position,
        avg_entry_price=avg_entry,
        unrealized_pnl=0.0,  # Computed by tracker
        realized_pnl=realized_pnl,
    )

    return StrategyState(
        market=market,
        position=position_state,
    )


def run_bucketed_backtest(
    df: pd.DataFrame,
    config: BucketedBacktestConfig,
    quote_model: QuoteModel | None = None,
    size_model: SizeModel | None = None,
    asset: str = "",
    slug: str = "",
) -> BucketedBacktestResult:
    """
    Run time-bucketed backtest preventing look-ahead bias.

    Core logic:
    1. Aggregate data into time buckets
    2. At each bucket T, check if previous quotes (set at T-1) fill against current BBO
    3. Update quotes for next bucket using current bucket's BBO via QuoteModel/SizeModel

    Fill logic (strict crossing):
    - BUY fill: market_ask at T < our_bid (set at T-1) -> someone sells to us cheaper
    - SELL fill: market_bid at T > our_ask (set at T-1) -> someone buys from us higher

    Args:
        df: Raw data DataFrame
        config: Backtest configuration
        quote_model: Quote model for computing bid/ask prices (default: SpreadQuoteModel)
        size_model: Size model for computing bid/ask sizes (default: InventoryBasedSizeModel)
        asset: Asset identifier
        slug: Market slug

    Returns:
        BucketedBacktestResult with metrics and fills
    """
    log = logger.bind(asset=asset, slug=slug)

    # Use default models if not provided
    if quote_model is None:
        quote_model = create_default_quote_model()
    if size_model is None:
        size_model = create_default_size_model()

    # Aggregate to buckets based on data source
    if config.data_source == DataSource.ORDERBOOK:
        buckets = aggregate_orderbook_buckets(df, config.bucket_size_ns)
    else:
        buckets = aggregate_trades_buckets(df, config.bucket_size_ns)

    if buckets.empty:
        log.warning("No buckets after aggregation")
        return BucketedBacktestResult(asset=asset, slug=slug)

    n_buckets = len(buckets)
    log.info("Aggregated to buckets", n_buckets=n_buckets, bucket_size_ms=config.bucket_size_ns / 1e6)

    # Initialize position tracker
    tracker = PositionTracker()

    # Quote state (quotes are set at T, used for fill checking at T+1)
    quote_bid = 0.0
    quote_ask = 0.0
    quote_bid_size = 0.0
    quote_ask_size = 0.0
    quote_valid = False
    max_position = 100.0  # Default, updated from size model

    # Tracking
    fills: list[BucketedFill] = []
    pnl_samples: list[float] = []
    max_position_seen = 0.0
    buy_fills = 0
    sell_fills = 0
    total_volume = 0.0

    # Iterate through buckets with T/T+1 separation
    for i, row in buckets.iterrows():
        bucket_ts = int(row["bucket_ts"])
        market_bid = row["best_bid"]
        market_ask = row["best_ask"]

        # Skip invalid BBO
        if market_bid <= 0 or market_ask <= 0 or market_bid >= market_ask:
            continue

        # Check fills using PREVIOUS bucket's quotes against CURRENT bucket's BBO
        if quote_valid:
            # BUY fill: market_ask < our_bid (someone sells to us cheaper than our bid)
            if market_ask < quote_bid and quote_bid_size > 0:
                # Can we buy given position limits?
                available_size = max_position - tracker.position
                fill_size = min(quote_bid_size, available_size)

                if fill_size > 0:
                    realized = tracker.update_buy(quote_bid, fill_size)
                    total_volume += fill_size
                    buy_fills += 1

                    fills.append(
                        BucketedFill(
                            bucket_ts=bucket_ts,
                            side=1,
                            price=quote_bid,
                            size=fill_size,
                            position_after=tracker.position,
                            realized_pnl=realized,
                        )
                    )

            # SELL fill: market_bid > our_ask (someone buys from us higher than our ask)
            if market_bid > quote_ask and quote_ask_size > 0:
                # Can we sell given position limits?
                available_size = max_position + tracker.position
                fill_size = min(quote_ask_size, available_size)

                if fill_size > 0:
                    realized = tracker.update_sell(quote_ask, fill_size)
                    total_volume += fill_size
                    sell_fills += 1

                    fills.append(
                        BucketedFill(
                            bucket_ts=bucket_ts,
                            side=-1,
                            price=quote_ask,
                            size=fill_size,
                            position_after=tracker.position,
                            realized_pnl=realized,
                        )
                    )

        # Track max position
        max_position_seen = max(max_position_seen, abs(tracker.position))

        # Sample PnL for metrics
        mid = (market_bid + market_ask) / 2
        pnl_samples.append(tracker.total_pnl(mid))

        # Create strategy state for models
        state = create_strategy_state(
            bucket_ts=bucket_ts,
            market_bid=market_bid,
            market_ask=market_ask,
            position=tracker.position,
            avg_entry=tracker.avg_entry,
            realized_pnl=tracker.realized_pnl,
            asset_id=asset,
        )

        # Get quotes from model
        quote_result = quote_model.compute(state)

        if quote_result.should_quote:
            quote_bid = quote_result.bid_price
            quote_ask = quote_result.ask_price

            # Get sizes from model
            size_result = size_model.compute(state, quote_result)
            quote_bid_size = size_result.bid_size
            quote_ask_size = size_result.ask_size
            max_position = size_result.max_position if size_result.max_position > 0 else 100.0

            quote_valid = True
        else:
            quote_valid = False

    # Final metrics
    final_mid = (buckets.iloc[-1]["best_bid"] + buckets.iloc[-1]["best_ask"]) / 2 if len(buckets) > 0 else 0
    unrealized_pnl = tracker.unrealized_pnl(final_mid)
    total_pnl = tracker.realized_pnl + unrealized_pnl

    pnl_arr = np.array(pnl_samples) if pnl_samples else np.array([0.0])
    sharpe = compute_sharpe_ratio(pnl_arr)
    max_dd = compute_max_drawdown(pnl_arr)

    log.info(
        "Backtest complete",
        n_buckets=n_buckets,
        n_fills=len(fills),
        buy_fills=buy_fills,
        sell_fills=sell_fills,
        total_pnl=f"{total_pnl:.4f}",
        realized_pnl=f"{tracker.realized_pnl:.4f}",
        final_position=f"{tracker.position:.2f}",
    )

    return BucketedBacktestResult(
        asset=asset,
        slug=slug,
        n_buckets=n_buckets,
        n_fills=len(fills),
        buy_fills=buy_fills,
        sell_fills=sell_fills,
        total_volume=total_volume,
        total_pnl=total_pnl,
        realized_pnl=tracker.realized_pnl,
        unrealized_pnl=unrealized_pnl,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        final_position=tracker.position,
        max_position=max_position_seen,
        fills=fills,
        pnl_history=pnl_arr,
    )


class BucketedBacktestEngine:
    """
    Engine for running bucketed backtests on datalake data.

    Loads data from datalake parquet files and runs time-bucketed backtest.
    """

    def __init__(
        self,
        config: BucketedBacktestConfig,
        quote_model: QuoteModel | None = None,
        size_model: SizeModel | None = None,
        data_dir: str = "data/datalake/timebased/crypto/updown/15m",
        symbols: list[str] | None = None,
    ):
        """
        Initialize bucketed backtest engine.

        Args:
            config: Backtest configuration
            quote_model: Quote model for computing prices
            size_model: Size model for computing sizes
            data_dir: Path to datalake timebased data
            symbols: List of symbols to backtest (e.g., ["btc", "eth"])
        """
        self.config = config
        self.quote_model = quote_model
        self.size_model = size_model
        self.data_dir = data_dir
        self.symbols = symbols
        self.logger = logger.bind(component="bucketed_engine")

    def run(self) -> dict[str, BucketedBacktestResult]:
        """
        Run backtest across all available markets.

        Returns:
            Dict mapping slug to BucketedBacktestResult
        """
        from pathlib import Path

        results: dict[str, BucketedBacktestResult] = {}
        data_path = Path(self.data_dir)

        if not data_path.exists():
            self.logger.error("Data directory not found", path=self.data_dir)
            return results

        # Find parquet files based on data source
        if self.config.data_source == DataSource.ORDERBOOK:
            pattern = "book_*.parquet.gz"
        else:
            pattern = "last_trade_price_*.parquet.gz"

        data_files = list(data_path.rglob(pattern))

        self.logger.info("Found data files", count=len(data_files), pattern=pattern)

        for data_file in data_files:
            # Extract symbol and slug from path
            parts = data_file.parts
            filename = data_file.name

            try:
                # Find symbol in path
                symbol = None
                for i, part in enumerate(parts):
                    if part.lower() in ["btc", "eth", "sol", "xrp", "doge", "ada", "avax", "link"]:
                        symbol = part.lower()
                        break

                if not symbol:
                    if filename.startswith("book_"):
                        slug_part = filename.split("_")[1]
                        symbol = slug_part.split("-")[0]
                    elif filename.startswith("last_trade_price_"):
                        slug_part = filename.split("_")[3]
                        symbol = slug_part.split("-")[0]

                if not symbol:
                    self.logger.warning("Could not parse symbol", path=str(data_file))
                    continue

                slug = filename.replace("book_", "").replace("last_trade_price_", "").replace(".parquet.gz", "")

            except (IndexError, ValueError):
                self.logger.warning("Could not parse path", path=str(data_file))
                continue

            # Filter by symbols
            if self.symbols and symbol.lower() not in [s.lower() for s in self.symbols]:
                continue

            self.logger.debug("Processing", slug=slug, symbol=symbol)

            try:
                import gzip
                import io

                if str(data_file).endswith(".gz"):
                    with gzip.open(data_file, "rb") as f:
                        df = pd.read_parquet(io.BytesIO(f.read()))
                else:
                    df = pd.read_parquet(data_file)

                if df.empty:
                    self.logger.warning("Empty data", slug=slug)
                    continue

                result = run_bucketed_backtest(
                    df=df,
                    config=self.config,
                    quote_model=self.quote_model,
                    size_model=self.size_model,
                    asset=symbol,
                    slug=slug,
                )

                results[slug] = result

            except Exception as e:
                self.logger.error("Failed to process", slug=slug, error=str(e))

        return results

    def run_single_file(self, file_path: str, asset: str = "", slug: str = "") -> BucketedBacktestResult:
        """
        Run backtest on a single parquet file.

        Args:
            file_path: Path to parquet file
            asset: Asset name
            slug: Market slug

        Returns:
            BucketedBacktestResult
        """
        import gzip
        import io

        if file_path.endswith(".gz"):
            with gzip.open(file_path, "rb") as f:
                df = pd.read_parquet(io.BytesIO(f.read()))
        else:
            df = pd.read_parquet(file_path)

        return run_bucketed_backtest(
            df=df,
            config=self.config,
            quote_model=self.quote_model,
            size_model=self.size_model,
            asset=asset,
            slug=slug,
        )


def print_bucketed_summary(results: dict[str, BucketedBacktestResult]) -> None:
    """Print summary of bucketed backtest results."""
    if not results:
        print("No results to display")
        return

    print("\n" + "=" * 80)
    print("BUCKETED BACKTEST SUMMARY")
    print("=" * 80)

    total_pnl = 0.0
    total_fills = 0
    total_volume = 0.0

    for slug, result in sorted(results.items()):
        total_pnl += result.total_pnl
        total_fills += result.n_fills
        total_volume += result.total_volume

        print(f"\n{slug}:")
        print(f"  Buckets: {result.n_buckets:,}")
        print(f"  Fills: {result.n_fills} (buy={result.buy_fills}, sell={result.sell_fills})")
        print(f"  Volume: {result.total_volume:,.2f}")
        print(f"  PnL: ${result.total_pnl:,.4f} (realized=${result.realized_pnl:,.4f})")
        print(f"  Final Position: {result.final_position:,.2f}")
        print(f"  Sharpe: {result.sharpe_ratio:.3f}")

    print("\n" + "-" * 80)
    print("AGGREGATE:")
    print(f"  Markets: {len(results)}")
    print(f"  Total Fills: {total_fills:,}")
    print(f"  Total Volume: {total_volume:,.2f}")
    print(f"  Total PnL: ${total_pnl:,.4f}")
    print("=" * 80)
