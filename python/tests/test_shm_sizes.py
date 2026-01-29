"""
Tests for shared memory struct sizes.

These tests ensure Python ctypes structures match Go's natural alignment.
Go doesn't support packed structs, so Python must use the same alignment
to share memory correctly.

Note: The C header uses #pragma pack(push, 1), but Go cannot implement this.
Python matches Go's actual implementation (unpacked) rather than C's design.
"""

import ctypes

from strategy.shm.types import (
    ASSET_ID_LEN,
    MAX_EXTERNAL_PRICES,
    MAX_MARKETS,
    MAX_OPEN_ORDERS,
    MAX_ORDERBOOK_LEVELS,
    MAX_POSITIONS,
    MAX_SIGNALS,
    ORDER_ID_LEN,
    SYMBOL_LEN,
    ExternalPrice,
    MarketBook,
    OpenOrder,
    OrderSignal,
    Position,
    PriceLevel,
    SharedMemoryLayout,
)


def test_price_level_size() -> None:
    """PriceLevel: 2 doubles = 16 bytes (no padding needed)."""
    assert ctypes.sizeof(PriceLevel) == 16


def test_market_book_size() -> None:
    """MarketBook with natural alignment matches Go's 776 bytes.

    Go adds 2 bytes padding after asset_id[78] to align timestamp_ns (uint64).
    """
    # Go's actual size (unpacked with alignment)
    assert ctypes.sizeof(MarketBook) == 776


def test_external_price_size() -> None:
    """ExternalPrice: symbol(16) + 3 doubles(24) + timestamp(8) = 48 bytes."""
    assert ctypes.sizeof(ExternalPrice) == 48


def test_position_size() -> None:
    """Position with natural alignment = 112 bytes.

    Go adds 2 bytes padding after asset_id[78] to align position (float64).
    """
    assert ctypes.sizeof(Position) == 112


def test_open_order_size() -> None:
    """OpenOrder with natural alignment = 200 bytes."""
    assert ctypes.sizeof(OpenOrder) == 200


def test_order_signal_size() -> None:
    """OrderSignal with natural alignment = 176 bytes."""
    assert ctypes.sizeof(OrderSignal) == 176


def test_shared_memory_layout_size() -> None:
    """SharedMemoryLayout total size must match Go's 86880 bytes.

    This is the critical test - if this fails, Python and Go will
    have misaligned views of shared memory.
    """
    # This must match the actual SHM file size created by Go
    assert ctypes.sizeof(SharedMemoryLayout) == 86880


def test_no_pack_attribute() -> None:
    """Verify no struct has _pack_ set (should use natural alignment)."""
    structs = [
        PriceLevel,
        MarketBook,
        ExternalPrice,
        Position,
        OpenOrder,
        OrderSignal,
        SharedMemoryLayout,
    ]
    for struct in structs:
        # _pack_ should either not exist or be 0 (default alignment)
        pack_value = getattr(struct, "_pack_", 0)
        assert pack_value == 0, f"{struct.__name__} has _pack_={pack_value}, should be unset"
