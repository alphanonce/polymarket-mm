"""
Shared Memory Reader

Provides read access to the shared memory for the Python strategy layer.
"""

import mmap
import os
from pathlib import Path

from strategy.shm.types import (
    SHM_MAGIC,
    SHM_NAME,
    ExternalPriceState,
    IVState,
    MarketState,
    PositionState,
    SharedMemoryLayout,
    shm_size,
)


class SHMReader:
    """Read-only access to shared memory."""

    def __init__(self) -> None:
        self._mm: mmap.mmap | None = None
        self._layout: SharedMemoryLayout | None = None
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

    def get_markets(self) -> list[MarketState]:
        """Get all markets."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")

        markets = []
        for i in range(self._layout.num_markets):
            markets.append(MarketState.from_ctypes(self._layout.markets[i]))
        return markets

    def get_market(self, asset_id: str) -> MarketState | None:
        """Get a specific market by asset ID."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")

        for i in range(self._layout.num_markets):
            if self._layout.markets[i].get_asset_id() == asset_id:
                return MarketState.from_ctypes(self._layout.markets[i])
        return None

    def get_markets_dict(self) -> dict[str, MarketState]:
        """Get all markets as a dictionary keyed by asset ID."""
        return {m.asset_id: m for m in self.get_markets()}

    def get_external_prices(self) -> list[ExternalPriceState]:
        """Get all external prices."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")

        prices = []
        for i in range(self._layout.num_external_prices):
            prices.append(ExternalPriceState.from_ctypes(self._layout.external_prices[i]))
        return prices

    def get_external_price(self, symbol: str) -> ExternalPriceState | None:
        """Get a specific external price by symbol."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")

        for i in range(self._layout.num_external_prices):
            if self._layout.external_prices[i].get_symbol() == symbol:
                return ExternalPriceState.from_ctypes(self._layout.external_prices[i])
        return None

    def get_external_prices_dict(self) -> dict[str, ExternalPriceState]:
        """Get all external prices as a dictionary keyed by symbol."""
        return {p.symbol: p for p in self.get_external_prices()}

    def get_positions(self) -> list[PositionState]:
        """Get all positions."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")

        positions = []
        for i in range(self._layout.num_positions):
            positions.append(PositionState.from_ctypes(self._layout.positions[i]))
        return positions

    def get_position(self, asset_id: str) -> PositionState | None:
        """Get a specific position by asset ID."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")

        for i in range(self._layout.num_positions):
            if self._layout.positions[i].get_asset_id() == asset_id:
                return PositionState.from_ctypes(self._layout.positions[i])
        return None

    def get_positions_dict(self) -> dict[str, PositionState]:
        """Get all positions as a dictionary keyed by asset ID."""
        return {p.asset_id: p for p in self.get_positions()}

    def get_iv_data(self) -> list[IVState]:
        """Get all implied volatility data."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")

        iv_list = []
        for i in range(self._layout.num_iv_data):
            iv_list.append(IVState.from_ctypes(self._layout.iv_data[i]))
        return iv_list

    def get_iv(self, symbol: str) -> IVState | None:
        """Get implied volatility data for a specific symbol."""
        if self._layout is None:
            raise RuntimeError("Not connected to shared memory")

        for i in range(self._layout.num_iv_data):
            if self._layout.iv_data[i].get_symbol() == symbol:
                return IVState.from_ctypes(self._layout.iv_data[i])
        return None

    def get_iv_dict(self) -> dict[str, IVState]:
        """Get all IV data as a dictionary keyed by symbol."""
        return {iv.symbol: iv for iv in self.get_iv_data()}

    def __enter__(self) -> "SHMReader":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
