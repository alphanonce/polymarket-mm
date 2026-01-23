"""
Paper Trading Module

Simulates order execution against live orderbook data and tracks
positions, PnL, and metrics. Stores all data in Supabase for
real-time dashboard visualization.
"""

from paper.executor import PaperExecutor
from paper.position_tracker import PositionTracker
from paper.supabase_store import SupabaseStore
from paper.metrics import MetricsCalculator
from paper.shm_paper import PaperSHMWriter
from paper.runner import PaperTradingRunner, PaperConfig

__all__ = [
    "PaperExecutor",
    "PositionTracker",
    "SupabaseStore",
    "MetricsCalculator",
    "PaperSHMWriter",
    "PaperTradingRunner",
    "PaperConfig",
]
