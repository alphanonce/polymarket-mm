"""
Paper Trading Module

Simulates order execution against live orderbook data and tracks
positions, PnL, and metrics. Stores all data in Supabase for
real-time dashboard visualization.
"""

from paper.executor import PaperExecutor
from paper.metrics import MetricsCalculator
from paper.position_tracker import PositionTracker
from paper.runner import PaperConfig, PaperTradingRunner
from paper.shm_paper import PaperSHMWriter
from paper.supabase_store import SupabaseStore

__all__ = [
    "PaperExecutor",
    "PositionTracker",
    "SupabaseStore",
    "MetricsCalculator",
    "PaperSHMWriter",
    "PaperTradingRunner",
    "PaperConfig",
]
