"""
Paper Trading Shared Memory Writer

Writes paper trading state to a dedicated SHM for Go to read and persist.
"""

import ctypes
import mmap
import os
import time
from pathlib import Path

import structlog

logger = structlog.get_logger()

# Constants matching Go types
PAPER_SHM_MAGIC = 0x50415052  # "PAPR"
PAPER_SHM_VERSION = 1
PAPER_SHM_NAME = "/polymarket_paper_shm"

MAX_PAPER_POSITIONS = 64
MAX_PAPER_QUOTES = 64
MAX_PAPER_TRADES = 128

ASSET_ID_LEN = 66
SLUG_LEN = 64


class PaperPosition(ctypes.Structure):
    """Paper position in SHM."""

    # No _pack_ - use natural alignment to match Go
    _fields_ = [
        ("asset_id", ctypes.c_char * ASSET_ID_LEN),
        ("slug", ctypes.c_char * SLUG_LEN),
        ("side", ctypes.c_char * 8),  # "up" or "down"
        ("size", ctypes.c_double),
        ("avg_entry_price", ctypes.c_double),
        ("unrealized_pnl", ctypes.c_double),
        ("realized_pnl", ctypes.c_double),
        ("updated_at_ns", ctypes.c_uint64),
        ("_padding", ctypes.c_uint8 * 6),
    ]


class PaperQuote(ctypes.Structure):
    """Paper quote in SHM."""

    # No _pack_ - use natural alignment to match Go
    _fields_ = [
        ("slug", ctypes.c_char * SLUG_LEN),
        ("our_bid", ctypes.c_double),
        ("our_ask", ctypes.c_double),
        ("best_bid", ctypes.c_double),
        ("best_ask", ctypes.c_double),
        ("mid_price", ctypes.c_double),
        ("spread", ctypes.c_double),
        ("inventory", ctypes.c_double),
        ("updated_ns", ctypes.c_uint64),
    ]


class PaperTrade(ctypes.Structure):
    """Paper trade in SHM (ring buffer)."""

    # No _pack_ - use natural alignment to match Go
    _fields_ = [
        ("asset_id", ctypes.c_char * ASSET_ID_LEN),
        ("slug", ctypes.c_char * SLUG_LEN),
        ("side", ctypes.c_int8),
        ("price", ctypes.c_double),
        ("size", ctypes.c_double),
        ("pnl", ctypes.c_double),
        ("timestamp_ns", ctypes.c_uint64),
        ("persisted", ctypes.c_uint8),  # 0=pending, 1=persisted
        ("_padding", ctypes.c_uint8 * 6),
    ]


class PaperTradingState(ctypes.Structure):
    """Paper trading SHM layout."""

    # No _pack_ - use natural alignment to match Go
    _fields_ = [
        # Magic number for validation
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        # Sequence number for change detection
        ("state_sequence", ctypes.c_uint32),
        ("_padding0", ctypes.c_uint32),
        # Positions
        ("num_positions", ctypes.c_uint32),
        ("_padding1", ctypes.c_uint32),
        ("positions", PaperPosition * MAX_PAPER_POSITIONS),
        # Quotes
        ("num_quotes", ctypes.c_uint32),
        ("_padding2", ctypes.c_uint32),
        ("quotes", PaperQuote * MAX_PAPER_QUOTES),
        # Trades (ring buffer)
        ("num_trades", ctypes.c_uint32),
        ("trades_head", ctypes.c_uint32),  # Next to consume
        ("trades_tail", ctypes.c_uint32),  # Next to write
        ("_padding3", ctypes.c_uint32),
        ("trades", PaperTrade * MAX_PAPER_TRADES),
        # Equity state
        ("total_equity", ctypes.c_double),
        ("cash", ctypes.c_double),
        ("position_value", ctypes.c_double),
        # Metrics
        ("total_pnl", ctypes.c_double),
        ("realized_pnl", ctypes.c_double),
        ("unrealized_pnl", ctypes.c_double),
        ("total_trades", ctypes.c_uint32),
        ("win_count", ctypes.c_uint32),
        ("sharpe_ratio", ctypes.c_double),
        ("max_drawdown", ctypes.c_double),
        # Timestamps
        ("last_update_ns", ctypes.c_uint64),
        ("last_snapshot_ns", ctypes.c_uint64),
        ("last_metrics_ns", ctypes.c_uint64),
    ]


def paper_shm_size() -> int:
    """Return the size of the paper trading SHM layout."""
    return ctypes.sizeof(PaperTradingState)


class PaperSHMWriter:
    """Writes paper trading state to SHM for Go to persist."""

    def __init__(self) -> None:
        self._mm: mmap.mmap | None = None
        self._layout: PaperTradingState | None = None
        self._fd: int | None = None
        self._position_map: dict[str, int] = {}  # asset_id -> index
        self._quote_map: dict[str, int] = {}  # slug -> index
        self._logger = logger.bind(component="paper_shm_writer")

    def connect(self) -> None:
        """Create and connect to paper trading SHM."""
        # Try /dev/shm first, fall back to /tmp
        shm_path = Path("/dev/shm") / PAPER_SHM_NAME.lstrip("/")
        if not shm_path.parent.exists():
            shm_path = Path("/tmp") / PAPER_SHM_NAME.lstrip("/")

        size = paper_shm_size()

        # Create the file if it doesn't exist
        self._fd = os.open(str(shm_path), os.O_RDWR | os.O_CREAT, 0o666)
        os.ftruncate(self._fd, size)

        # Memory map with read/write access
        self._mm = mmap.mmap(self._fd, size, access=mmap.ACCESS_WRITE)

        # Create ctypes structure from mmap (writable)
        self._layout = PaperTradingState.from_buffer(self._mm)

        # Initialize header
        self._layout.magic = PAPER_SHM_MAGIC
        self._layout.version = PAPER_SHM_VERSION

        # Always reset quote and position counts since our in-memory maps are empty
        # (positions/quotes will be re-populated from the paper trading engine)
        self._layout.num_positions = 0
        self._layout.num_quotes = 0

        # Only reset trades if magic was different (new file)
        if self._layout.state_sequence == 0:
            self._layout.num_trades = 0
            self._layout.trades_head = 0
            self._layout.trades_tail = 0

        self._mm.flush()

        self._logger.info("Connected to paper trading SHM", path=str(shm_path))

    def close(self) -> None:
        """Close SHM connection."""
        if self._mm is not None:
            self._mm.flush()
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._layout = None

    def _increment_sequence(self) -> None:
        """Increment state sequence number."""
        if self._layout:
            self._layout.state_sequence += 1
            self._layout.last_update_ns = time.time_ns()

    def update_position(
        self,
        asset_id: str,
        slug: str,
        side: str,
        size: float,
        avg_entry_price: float,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
    ) -> None:
        """Update a position in SHM."""
        if self._layout is None:
            return

        # Find existing or allocate new slot
        if asset_id in self._position_map:
            idx = self._position_map[asset_id]
        else:
            idx = self._layout.num_positions
            if idx >= MAX_PAPER_POSITIONS:
                self._logger.warning("Max positions reached")
                return
            self._position_map[asset_id] = idx
            self._layout.num_positions = idx + 1

        pos = self._layout.positions[idx]
        pos.asset_id = asset_id.encode("utf-8")[:ASSET_ID_LEN]
        pos.slug = slug.encode("utf-8")[:SLUG_LEN]
        pos.side = side.encode("utf-8")[:8]
        pos.size = size
        pos.avg_entry_price = avg_entry_price
        pos.unrealized_pnl = unrealized_pnl
        pos.realized_pnl = realized_pnl
        pos.updated_at_ns = time.time_ns()

        self._increment_sequence()

    def update_quote(
        self,
        slug: str,
        our_bid: float,
        our_ask: float,
        best_bid: float,
        best_ask: float,
        mid_price: float,
        spread: float,
        inventory: float,
    ) -> None:
        """Update a quote in SHM."""
        if self._layout is None:
            return

        # Find existing or allocate new slot
        if slug in self._quote_map:
            idx = self._quote_map[slug]
        else:
            idx = self._layout.num_quotes
            if idx >= MAX_PAPER_QUOTES:
                self._logger.warning("Max quotes reached")
                return
            self._quote_map[slug] = idx
            self._layout.num_quotes = idx + 1

        quote = self._layout.quotes[idx]
        quote.slug = slug.encode("utf-8")[:SLUG_LEN]
        quote.our_bid = our_bid
        quote.our_ask = our_ask
        quote.best_bid = best_bid
        quote.best_ask = best_ask
        quote.mid_price = mid_price
        quote.spread = spread
        quote.inventory = inventory
        quote.updated_ns = time.time_ns()

        self._increment_sequence()

    def add_trade(
        self,
        asset_id: str,
        slug: str,
        side: int,
        price: float,
        size: float,
        pnl: float = 0.0,
    ) -> None:
        """Add a trade to the ring buffer for Go to persist."""
        if self._layout is None:
            return

        # Write to ring buffer at tail
        tail = self._layout.trades_tail % MAX_PAPER_TRADES
        trade = self._layout.trades[tail]
        trade.asset_id = asset_id.encode("utf-8")[:ASSET_ID_LEN]
        trade.slug = slug.encode("utf-8")[:SLUG_LEN]
        trade.side = side
        trade.price = price
        trade.size = size
        trade.pnl = pnl
        trade.timestamp_ns = time.time_ns()
        trade.persisted = 0  # Not yet persisted

        # Advance tail
        self._layout.trades_tail += 1
        self._layout.num_trades += 1

        self._increment_sequence()

    def update_equity(
        self,
        total_equity: float,
        cash: float,
        position_value: float,
    ) -> None:
        """Update equity state."""
        if self._layout is None:
            return

        self._layout.total_equity = total_equity
        self._layout.cash = cash
        self._layout.position_value = position_value
        self._layout.last_snapshot_ns = time.time_ns()

        self._increment_sequence()

    def update_metrics(
        self,
        total_pnl: float,
        realized_pnl: float,
        unrealized_pnl: float,
        total_trades: int,
        win_count: int,
        sharpe_ratio: float,
        max_drawdown: float,
    ) -> None:
        """Update metrics."""
        if self._layout is None:
            return

        self._layout.total_pnl = total_pnl
        self._layout.realized_pnl = realized_pnl
        self._layout.unrealized_pnl = unrealized_pnl
        self._layout.total_trades = total_trades
        self._layout.win_count = win_count
        self._layout.sharpe_ratio = sharpe_ratio
        self._layout.max_drawdown = max_drawdown
        self._layout.last_metrics_ns = time.time_ns()

        self._increment_sequence()

    def flush(self) -> None:
        """Flush changes to SHM."""
        if self._mm is not None:
            self._mm.flush()

    def __enter__(self) -> "PaperSHMWriter":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
