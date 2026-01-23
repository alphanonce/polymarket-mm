"""
Shared Memory Module

Provides Python access to the shared memory layout for inter-process
communication with the Go executor.
"""

from strategy.shm.types import (
    PriceLevel,
    MarketBook,
    ExternalPrice,
    Position,
    OpenOrder,
    OrderSignal,
    SharedMemoryLayout,
    SHM_MAGIC,
    SHM_VERSION,
    SHM_NAME,
    SIDE_BUY,
    SIDE_SELL,
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_MARKET,
    ACTION_PLACE,
    ACTION_CANCEL,
    ACTION_MODIFY,
)
from strategy.shm.reader import SHMReader
from strategy.shm.writer import SHMWriter

__all__ = [
    "PriceLevel",
    "MarketBook",
    "ExternalPrice",
    "Position",
    "OpenOrder",
    "OrderSignal",
    "SharedMemoryLayout",
    "SHMReader",
    "SHMWriter",
    "SHM_MAGIC",
    "SHM_VERSION",
    "SHM_NAME",
    "SIDE_BUY",
    "SIDE_SELL",
    "ORDER_TYPE_LIMIT",
    "ORDER_TYPE_MARKET",
    "ACTION_PLACE",
    "ACTION_CANCEL",
    "ACTION_MODIFY",
]
