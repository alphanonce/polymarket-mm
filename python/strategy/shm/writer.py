"""
Shared Memory Writer

Provides write access to the signal section of shared memory
for the Python strategy layer.
"""

import ctypes
import mmap
import os
import time
from pathlib import Path
from typing import Optional

from strategy.shm.types import (
    ACTION_CANCEL,
    ACTION_MODIFY,
    ACTION_PLACE,
    MAX_SIGNALS,
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_MARKET,
    OrderSignal,
    SharedMemoryLayout,
    SHM_MAGIC,
    SHM_NAME,
    SIDE_BUY,
    SIDE_SELL,
    shm_size,
)


class SHMWriter:
    """Write access to the signal section of shared memory."""

    def __init__(self) -> None:
        self._mm: Optional[mmap.mmap] = None
        self._layout: Optional[SharedMemoryLayout] = None
        self._signal_id_counter: int = 0

    def connect(self) -> None:
        """Connect to shared memory."""
        # Try /dev/shm first, fall back to /tmp
        shm_path = Path("/dev/shm") / SHM_NAME.lstrip("/")
        if not shm_path.exists():
            shm_path = Path("/tmp") / SHM_NAME.lstrip("/")

        if not shm_path.exists():
            raise FileNotFoundError(f"Shared memory file not found: {shm_path}")

        fd = os.open(str(shm_path), os.O_RDWR)
        try:
            self._mm = mmap.mmap(fd, shm_size(), access=mmap.ACCESS_WRITE)
        finally:
            os.close(fd)

        # Verify magic by reading header
        self._mm.seek(0)
        magic = int.from_bytes(self._mm.read(4), byteorder="little")
        if magic != SHM_MAGIC:
            self.close()
            raise ValueError(f"Invalid shared memory magic: got 0x{magic:X}, expected 0x{SHM_MAGIC:X}")

    def close(self) -> None:
        """Close shared memory connection."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None

    def _read_layout(self) -> SharedMemoryLayout:
        """Read current layout from shared memory."""
        if self._mm is None:
            raise RuntimeError("Not connected to shared memory")
        self._mm.seek(0)
        return SharedMemoryLayout.from_buffer_copy(self._mm)

    def _write_signals_section(self, num_signals: int, signals: list[OrderSignal]) -> None:
        """Write only the signals section to shared memory (not the entire layout)."""
        if self._mm is None:
            raise RuntimeError("Not connected to shared memory")

        # Read current layout to get current signal_sequence
        layout = self._read_layout()
        new_signal_sequence = layout.signal_sequence + 1
        new_signal_timestamp_ns = time.time_ns()

        # Calculate offsets using ctypes
        signal_sequence_offset = SharedMemoryLayout.signal_sequence.offset
        signal_timestamp_ns_offset = SharedMemoryLayout.signal_timestamp_ns.offset
        num_signals_offset = SharedMemoryLayout.num_signals.offset
        signals_offset = SharedMemoryLayout.signals.offset

        # Write signal_sequence (uint32, 4 bytes)
        self._mm.seek(signal_sequence_offset)
        self._mm.write(new_signal_sequence.to_bytes(4, byteorder="little"))

        # Write signal_timestamp_ns (uint64, 8 bytes)
        self._mm.seek(signal_timestamp_ns_offset)
        self._mm.write(new_signal_timestamp_ns.to_bytes(8, byteorder="little"))

        # Write num_signals (uint32, 4 bytes)
        self._mm.seek(num_signals_offset)
        self._mm.write(num_signals.to_bytes(4, byteorder="little"))

        # Write signals array
        self._mm.seek(signals_offset)
        for i in range(MAX_SIGNALS):
            if i < len(signals):
                self._mm.write(bytes(signals[i]))
            else:
                # Write empty signal
                self._mm.write(bytes(OrderSignal()))

    def _next_signal_id(self) -> int:
        """Generate next signal ID."""
        self._signal_id_counter += 1
        return self._signal_id_counter

    def place_limit_order(
        self,
        asset_id: str,
        side: int,
        price: float,
        size: float,
    ) -> int:
        """
        Add a limit order signal.

        Args:
            asset_id: The market asset ID
            side: SIDE_BUY or SIDE_SELL
            price: Limit price
            size: Order size

        Returns:
            Signal ID
        """
        if side not in (SIDE_BUY, SIDE_SELL):
            raise ValueError(f"Invalid side: {side}")

        signal = OrderSignal()
        signal.signal_id = self._next_signal_id()
        signal.set_asset_id(asset_id)
        signal.side = side
        signal.price = price
        signal.size = size
        signal.order_type = ORDER_TYPE_LIMIT
        signal.action = ACTION_PLACE

        return self._add_signal(signal)

    def place_market_order(
        self,
        asset_id: str,
        side: int,
        size: float,
    ) -> int:
        """
        Add a market order signal.

        Args:
            asset_id: The market asset ID
            side: SIDE_BUY or SIDE_SELL
            size: Order size

        Returns:
            Signal ID
        """
        if side not in (SIDE_BUY, SIDE_SELL):
            raise ValueError(f"Invalid side: {side}")

        signal = OrderSignal()
        signal.signal_id = self._next_signal_id()
        signal.set_asset_id(asset_id)
        signal.side = side
        signal.price = 0.0
        signal.size = size
        signal.order_type = ORDER_TYPE_MARKET
        signal.action = ACTION_PLACE

        return self._add_signal(signal)

    def cancel_order(self, asset_id: str, order_id: str) -> int:
        """
        Add a cancel order signal.

        Args:
            asset_id: The market asset ID
            order_id: The order ID to cancel

        Returns:
            Signal ID
        """
        signal = OrderSignal()
        signal.signal_id = self._next_signal_id()
        signal.set_asset_id(asset_id)
        signal.action = ACTION_CANCEL
        signal.set_cancel_order_id(order_id)

        return self._add_signal(signal)

    def modify_order(
        self,
        asset_id: str,
        order_id: str,
        new_price: float,
        new_size: float,
    ) -> int:
        """
        Add a modify order signal.

        Args:
            asset_id: The market asset ID
            order_id: The order ID to modify
            new_price: New limit price
            new_size: New size

        Returns:
            Signal ID
        """
        signal = OrderSignal()
        signal.signal_id = self._next_signal_id()
        signal.set_asset_id(asset_id)
        signal.price = new_price
        signal.size = new_size
        signal.action = ACTION_MODIFY
        signal.set_cancel_order_id(order_id)

        return self._add_signal(signal)

    def _add_signal(self, signal: OrderSignal) -> int:
        """Add a signal to the shared memory."""
        layout = self._read_layout()

        if layout.num_signals >= MAX_SIGNALS:
            raise RuntimeError("Signal buffer full")

        # Read existing signals
        signals = [layout.signals[i] for i in range(layout.num_signals)]
        signals.append(signal)

        self._write_signals_section(len(signals), signals)
        return signal.signal_id

    def clear_signals(self) -> None:
        """Clear all signals."""
        self._write_signals_section(0, [])

    def get_signals_processed(self) -> int:
        """Get number of signals that have been processed by executor."""
        layout = self._read_layout()
        return layout.signals_processed

    def wait_for_signals_processed(self, timeout_ms: int = 1000) -> bool:
        """
        Wait for all signals to be processed.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            True if all signals processed, False if timeout
        """
        start = time.time()
        while (time.time() - start) * 1000 < timeout_ms:
            layout = self._read_layout()
            if layout.signals_processed >= layout.num_signals:
                return True
            time.sleep(0.0001)  # 100 microseconds
        return False

    def __enter__(self) -> "SHMWriter":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
