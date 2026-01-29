"""
Shared Memory Types

ctypes struct definitions matching shared/shm_layout.h
This must be kept in sync with the C header file.
"""

import ctypes
from dataclasses import dataclass

# Constants matching shared/shm_layout.h
SHM_MAGIC = 0x504D4D4D  # "PMMM"
SHM_VERSION = 2
SHM_NAME = "/polymarket_mm_shm"

MAX_MARKETS = 64
MAX_ORDERBOOK_LEVELS = 20
MAX_EXTERNAL_PRICES = 32
MAX_POSITIONS = 64
MAX_SIGNALS = 16
MAX_OPEN_ORDERS = 128
MAX_IV_DATA = 8
MAX_IV_TENORS = 6

ASSET_ID_LEN = 78  # Polymarket token IDs are up to 77 digits + null terminator
SYMBOL_LEN = 16
ORDER_ID_LEN = 64

# Side constants
SIDE_BUY = 1
SIDE_SELL = -1

# Order type constants
ORDER_TYPE_LIMIT = 0
ORDER_TYPE_MARKET = 1

# Signal action constants
ACTION_PLACE = 0
ACTION_CANCEL = 1
ACTION_MODIFY = 2

# Order status constants
ORDER_STATUS_PENDING = 0
ORDER_STATUS_OPEN = 1
ORDER_STATUS_FILLED = 2
ORDER_STATUS_CANCELLED = 3
ORDER_STATUS_REJECTED = 4


class PriceLevel(ctypes.Structure):
    """Single price level in orderbook."""

    # No _pack_ - use natural alignment to match Go
    _fields_ = [
        ("price", ctypes.c_double),
        ("size", ctypes.c_double),
    ]


class MarketBook(ctypes.Structure):
    """Market orderbook state for a single market."""

    # No _pack_ - use natural alignment to match Go
    _fields_ = [
        ("asset_id", ctypes.c_char * ASSET_ID_LEN),
        ("timestamp_ns", ctypes.c_uint64),
        ("mid_price", ctypes.c_double),
        ("spread", ctypes.c_double),
        ("bids", PriceLevel * MAX_ORDERBOOK_LEVELS),
        ("asks", PriceLevel * MAX_ORDERBOOK_LEVELS),
        ("bid_levels", ctypes.c_uint32),
        ("ask_levels", ctypes.c_uint32),
        ("last_trade_price", ctypes.c_double),
        ("last_trade_size", ctypes.c_double),
        ("last_trade_side", ctypes.c_int8),
        ("_padding", ctypes.c_uint8 * 7),
    ]

    def get_asset_id(self) -> str:
        """Get asset ID as string."""
        return self.asset_id.decode("utf-8").rstrip("\x00")

    def get_bids(self) -> list[tuple[float, float]]:
        """Get bid levels as list of (price, size) tuples."""
        # Clamp to MAX_ORDERBOOK_LEVELS to avoid index errors from corrupted data
        levels = min(self.bid_levels, MAX_ORDERBOOK_LEVELS)
        return [(self.bids[i].price, self.bids[i].size) for i in range(levels)]

    def get_asks(self) -> list[tuple[float, float]]:
        """Get ask levels as list of (price, size) tuples."""
        # Clamp to MAX_ORDERBOOK_LEVELS to avoid index errors from corrupted data
        levels = min(self.ask_levels, MAX_ORDERBOOK_LEVELS)
        return [(self.asks[i].price, self.asks[i].size) for i in range(levels)]


class ExternalPrice(ctypes.Structure):
    """External price feed."""

    # No _pack_ - use natural alignment to match Go
    _fields_ = [
        ("symbol", ctypes.c_char * SYMBOL_LEN),
        ("price", ctypes.c_double),
        ("bid", ctypes.c_double),
        ("ask", ctypes.c_double),
        ("timestamp_ns", ctypes.c_uint64),
    ]

    def get_symbol(self) -> str:
        """Get symbol as string."""
        return self.symbol.decode("utf-8").rstrip("\x00")


class Position(ctypes.Structure):
    """Position in a market."""

    # No _pack_ - use natural alignment to match Go
    _fields_ = [
        ("asset_id", ctypes.c_char * ASSET_ID_LEN),
        ("position", ctypes.c_double),
        ("avg_entry_price", ctypes.c_double),
        ("unrealized_pnl", ctypes.c_double),
        ("realized_pnl", ctypes.c_double),
    ]

    def get_asset_id(self) -> str:
        """Get asset ID as string."""
        return self.asset_id.decode("utf-8").rstrip("\x00")


class OpenOrder(ctypes.Structure):
    """Open order tracking."""

    # No _pack_ - use natural alignment to match Go
    _fields_ = [
        ("order_id", ctypes.c_char * ORDER_ID_LEN),
        ("asset_id", ctypes.c_char * ASSET_ID_LEN),
        ("side", ctypes.c_int8),
        ("price", ctypes.c_double),
        ("size", ctypes.c_double),
        ("filled_size", ctypes.c_double),
        ("status", ctypes.c_uint8),
        ("created_at_ns", ctypes.c_uint64),
        ("updated_at_ns", ctypes.c_uint64),
        ("_padding", ctypes.c_uint8 * 6),
    ]

    def get_order_id(self) -> str:
        """Get order ID as string."""
        return self.order_id.decode("utf-8").rstrip("\x00")

    def get_asset_id(self) -> str:
        """Get asset ID as string."""
        return self.asset_id.decode("utf-8").rstrip("\x00")


class OrderSignal(ctypes.Structure):
    """Order signal from strategy to executor."""

    # No _pack_ - use natural alignment to match Go
    _fields_ = [
        ("signal_id", ctypes.c_uint64),
        ("asset_id", ctypes.c_char * ASSET_ID_LEN),
        ("side", ctypes.c_int8),
        ("price", ctypes.c_double),
        ("size", ctypes.c_double),
        ("order_type", ctypes.c_uint8),
        ("action", ctypes.c_uint8),
        ("cancel_order_id", ctypes.c_char * ORDER_ID_LEN),
        ("_padding", ctypes.c_uint8 * 5),
    ]

    def set_asset_id(self, asset_id: str) -> None:
        """Set asset ID from string."""
        self.asset_id = asset_id.encode("utf-8")

    def set_cancel_order_id(self, order_id: str) -> None:
        """Set cancel order ID from string."""
        self.cancel_order_id = order_id.encode("utf-8")


class IVTenorPoint(ctypes.Structure):
    """Single point in IV term structure."""

    # No _pack_ - use natural alignment to match Go
    _fields_ = [
        ("days_to_expiry", ctypes.c_double),
        ("atm_iv", ctypes.c_double),
    ]


class ImpliedVolData(ctypes.Structure):
    """Implied volatility data for a single underlying."""

    # No _pack_ - use natural alignment to match Go
    _fields_ = [
        ("symbol", ctypes.c_char * SYMBOL_LEN),
        ("atm_iv", ctypes.c_double),
        ("timestamp_ns", ctypes.c_uint64),
        ("term_structure", IVTenorPoint * MAX_IV_TENORS),
        ("num_tenors", ctypes.c_uint32),
        ("_padding", ctypes.c_uint8 * 4),
    ]

    def get_symbol(self) -> str:
        """Get symbol as string."""
        return self.symbol.decode("utf-8").rstrip("\x00")

    def get_term_structure(self) -> list[tuple[float, float]]:
        """Get term structure as list of (days_to_expiry, iv) tuples."""
        tenors = min(self.num_tenors, MAX_IV_TENORS)
        return [
            (self.term_structure[i].days_to_expiry, self.term_structure[i].atm_iv)
            for i in range(tenors)
        ]


class SharedMemoryLayout(ctypes.Structure):
    """Main shared memory layout."""

    # No _pack_ - use natural alignment to match Go
    _fields_ = [
        # Header
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        # Synchronization
        ("state_sequence", ctypes.c_uint32),
        ("signal_sequence", ctypes.c_uint32),
        # Timestamps
        ("state_timestamp_ns", ctypes.c_uint64),
        ("signal_timestamp_ns", ctypes.c_uint64),
        # Market state
        ("num_markets", ctypes.c_uint32),
        ("_padding1", ctypes.c_uint32),
        ("markets", MarketBook * MAX_MARKETS),
        # External prices
        ("num_external_prices", ctypes.c_uint32),
        ("_padding2", ctypes.c_uint32),
        ("external_prices", ExternalPrice * MAX_EXTERNAL_PRICES),
        # Positions
        ("num_positions", ctypes.c_uint32),
        ("_padding3", ctypes.c_uint32),
        ("positions", Position * MAX_POSITIONS),
        # Open orders
        ("num_open_orders", ctypes.c_uint32),
        ("_padding4", ctypes.c_uint32),
        ("open_orders", OpenOrder * MAX_OPEN_ORDERS),
        # Implied volatility data
        ("num_iv_data", ctypes.c_uint32),
        ("_padding_iv", ctypes.c_uint32),
        ("iv_data", ImpliedVolData * MAX_IV_DATA),
        # Strategy state
        ("total_equity", ctypes.c_double),
        ("available_margin", ctypes.c_double),
        ("trading_enabled", ctypes.c_uint8),
        ("_padding5", ctypes.c_uint8 * 7),
        # Order signals
        ("num_signals", ctypes.c_uint32),
        ("signals_processed", ctypes.c_uint32),
        ("signals", OrderSignal * MAX_SIGNALS),
    ]


def shm_size() -> int:
    """Return the size of the shared memory layout in bytes."""
    return ctypes.sizeof(SharedMemoryLayout)


# Python-friendly dataclasses for strategy use


@dataclass
class MarketState:
    """Python-friendly market state."""

    asset_id: str
    timestamp_ns: int
    mid_price: float
    spread: float
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    last_trade_price: float
    last_trade_size: float
    last_trade_side: int

    @classmethod
    def from_ctypes(cls, book: MarketBook) -> "MarketState":
        """Create from ctypes MarketBook."""
        return cls(
            asset_id=book.get_asset_id(),
            timestamp_ns=book.timestamp_ns,
            mid_price=book.mid_price,
            spread=book.spread,
            bids=book.get_bids(),
            asks=book.get_asks(),
            last_trade_price=book.last_trade_price,
            last_trade_size=book.last_trade_size,
            last_trade_side=book.last_trade_side,
        )


@dataclass
class ExternalPriceState:
    """Python-friendly external price state."""

    symbol: str
    price: float
    bid: float
    ask: float
    timestamp_ns: int

    @classmethod
    def from_ctypes(cls, price: ExternalPrice) -> "ExternalPriceState":
        """Create from ctypes ExternalPrice."""
        return cls(
            symbol=price.get_symbol(),
            price=price.price,
            bid=price.bid,
            ask=price.ask,
            timestamp_ns=price.timestamp_ns,
        )


@dataclass
class PositionState:
    """Python-friendly position state."""

    asset_id: str
    position: float
    avg_entry_price: float
    unrealized_pnl: float
    realized_pnl: float

    @classmethod
    def from_ctypes(cls, pos: Position) -> "PositionState":
        """Create from ctypes Position."""
        return cls(
            asset_id=pos.get_asset_id(),
            position=pos.position,
            avg_entry_price=pos.avg_entry_price,
            unrealized_pnl=pos.unrealized_pnl,
            realized_pnl=pos.realized_pnl,
        )


@dataclass
class IVState:
    """Python-friendly implied volatility state."""

    symbol: str
    atm_iv: float
    timestamp_ns: int
    term_structure: list[tuple[float, float]]  # [(days_to_expiry, iv), ...]

    @classmethod
    def from_ctypes(cls, iv_data: ImpliedVolData) -> "IVState":
        """Create from ctypes ImpliedVolData."""
        return cls(
            symbol=iv_data.get_symbol(),
            atm_iv=iv_data.atm_iv,
            timestamp_ns=iv_data.timestamp_ns,
            term_structure=iv_data.get_term_structure(),
        )

    def interpolate_iv(self, tte_days: float) -> float:
        """
        Interpolate IV for given time-to-expiry in days.

        Args:
            tte_days: Target time-to-expiry in days (e.g., 0.01 for 15 minutes)

        Returns:
            Interpolated IV, or atm_iv as fallback
        """
        if not self.term_structure:
            return self.atm_iv

        # For very short TTL (15m = 0.0104 days), extrapolate from shortest tenor
        if tte_days <= self.term_structure[0][0]:
            return self.term_structure[0][1]

        # Linear interpolation between tenors
        for i in range(len(self.term_structure) - 1):
            t1, iv1 = self.term_structure[i]
            t2, iv2 = self.term_structure[i + 1]
            if t1 <= tte_days <= t2:
                # Guard against division by zero (identical tenor points)
                if t2 == t1:
                    return iv1
                weight = (tte_days - t1) / (t2 - t1)
                return iv1 + weight * (iv2 - iv1)

        # Beyond longest tenor, use longest
        return self.term_structure[-1][1]
