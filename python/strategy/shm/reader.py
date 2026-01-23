"""
Shared Memory Reader

Provides read access to the shared memory for the Python strategy layer.
"""

import ctypes
import mmap
import os
from pathlib import Path
from typing import Dict, List, Optional

from strategy.shm.types import (
    ExternalPriceState,
    MarketState,
    PositionState,
    SharedMemoryLayout,
    SHM_MAGIC,
    SHM_NAME,
    shm_size,
)


class SHMReader:
    """Read-only access to shared memory."""

    def __init__(self) -> None:
        self._mm: Optional[mmap.mmap] = None
        self._layout: Optional[SharedMemoryLayout] = None
        self._last_sequence: int = 0

    def connect(self) -> None:
        """Connect to shared memory."""
        # Try /dev/shm first, fall back to /tmp
        shm_path = Path("/dev/shm") / SHM_NAME.lstrip("/")
        if not shm_path.exists():
            shm_path = Path("/tmp") / SHM_NAME.lstrip("/")

        if not shm_path.exists():
            raise FileNotFoundError(f"Shared memory file not found: {shm_path}")

        fd = os.open(str(shm_path), os.O_RDONLY)
        try:
            self._mm = mmap.mmap(fd, shm_size(), access=mmap.ACCESS_READ)
        finally:
            os.close(fd)

        # Create ctypes structure from mmap
        self._layout = SharedMemoryLayout.from_buffer_copy(self._mm)

        # Verify magic
        if self._layout.magic != SHM_MAGIC:
            self.close()
            raise ValueError(
                f"Invalid shared memory magic: got 0x{self._layout.magic:X}, "
                f"expected 0x{SHM_MAGIC:X}"
            )

    def close(self) -> None:
        """Close shared memory connection."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None
            self._layout = None

    def refresh(self) -> bool:
        """
        Refresh the layout from shared memory.

        Returns True if state has changed since last refresh.
        """
        if self._mm is None:
            raise RuntimeError("Not connected to shared memory")

        self._mm.seek(0)
        self._layout = SharedMemoryLayout.from_buffer_copy(self._mm)

        changed = self._layout.state_sequence != self._last_sequence
        self._last_sequence = self._layout.state_sequence
        return changed

    @property
    def state_sequence(self) -> int:
        """Get current state sequence number."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")
        return self._layout.state_sequence

    @property
    def state_timestamp_ns(self) -> int:
        """Get state timestamp in nanoseconds."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")
        return self._layout.state_timestamp_ns

    @property
    def trading_enabled(self) -> bool:
        """Check if trading is enabled."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")
        return self._layout.trading_enabled == 1

    @property
    def total_equity(self) -> float:
        """Get total equity."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")
        return self._layout.total_equity

    @property
    def available_margin(self) -> float:
        """Get available margin."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")
        return self._layout.available_margin

    def get_markets(self) -> List[MarketState]:
        """Get all markets."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")

        markets = []
        for i in range(self._layout.num_markets):
            markets.append(MarketState.from_ctypes(self._layout.markets[i]))
        return markets

    def get_market(self, asset_id: str) -> Optional[MarketState]:
        """Get a specific market by asset ID."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")

        for i in range(self._layout.num_markets):
            if self._layout.markets[i].get_asset_id() == asset_id:
                return MarketState.from_ctypes(self._layout.markets[i])
        return None

    def get_markets_dict(self) -> Dict[str, MarketState]:
        """Get all markets as a dictionary keyed by asset ID."""
        return {m.asset_id: m for m in self.get_markets()}

    def get_external_prices(self) -> List[ExternalPriceState]:
        """Get all external prices."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")

        prices = []
        for i in range(self._layout.num_external_prices):
            prices.append(ExternalPriceState.from_ctypes(self._layout.external_prices[i]))
        return prices

    def get_external_price(self, symbol: str) -> Optional[ExternalPriceState]:
        """Get a specific external price by symbol."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")

        for i in range(self._layout.num_external_prices):
            if self._layout.external_prices[i].get_symbol() == symbol:
                return ExternalPriceState.from_ctypes(self._layout.external_prices[i])
        return None

    def get_external_prices_dict(self) -> Dict[str, ExternalPriceState]:
        """Get all external prices as a dictionary keyed by symbol."""
        return {p.symbol: p for p in self.get_external_prices()}

    def get_positions(self) -> List[PositionState]:
        """Get all positions."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")

        positions = []
        for i in range(self._layout.num_positions):
            positions.append(PositionState.from_ctypes(self._layout.positions[i]))
        return positions

    def get_position(self, asset_id: str) -> Optional[PositionState]:
        """Get a specific position by asset ID."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")

        for i in range(self._layout.num_positions):
            if self._layout.positions[i].get_asset_id() == asset_id:
                return PositionState.from_ctypes(self._layout.positions[i])
        return None

    def get_positions_dict(self) -> Dict[str, PositionState]:
        """Get all positions as a dictionary keyed by asset ID."""
        return {p.asset_id: p for p in self.get_positions()}

    def __enter__(self) -> "SHMReader":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
