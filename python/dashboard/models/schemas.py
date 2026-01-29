"""
Dashboard Pydantic Schemas

Data models for API responses and WebSocket messages.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LevelInfo(BaseModel):
    """Orderbook level."""

    price: float
    size: float


class PositionInfo(BaseModel):
    """Position information."""

    asset_id: str
    slug: str
    side: str  # 'up' or 'down'
    size: float
    avg_entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    current_price: float = 0.0


class FillInfo(BaseModel):
    """Fill (trade execution) information."""

    signal_id: int
    asset_id: str
    slug: str
    asset: str = ""  # e.g., 'btc', 'eth' - extracted from slug for filtering
    side: str  # 'BUY' or 'SELL'
    token_side: str = "up"  # 'up' or 'down' - which token was traded
    price: float
    size: float
    pnl: float = 0.0
    timestamp_ms: int


class MarketQuote(BaseModel):
    """Current market quote with our quotes overlaid."""

    slug: str
    asset: str
    timeframe: str
    our_bid: float | None = None
    our_ask: float | None = None
    best_bid: float
    best_ask: float
    mid_price: float
    spread: float
    bids: list[LevelInfo] = Field(default_factory=list)
    asks: list[LevelInfo] = Field(default_factory=list)
    inventory: float = 0.0
    time_to_expiry_s: float = 0.0


class QuoteHistoryPoint(BaseModel):
    """Historical quote data point."""

    timestamp_ms: int
    slug: str
    asset: str = ""  # e.g., 'btc', 'eth' - extracted from slug for filtering
    our_bid: float | None = None
    our_ask: float | None = None
    mid_price: float
    best_bid: float
    best_ask: float


class AssetMetrics(BaseModel):
    """Per-asset performance metrics."""

    asset: str
    total_trades: int = 0
    win_count: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    inventory: float = 0.0


class DashboardState(BaseModel):
    """Complete dashboard state for a strategy."""

    strategy_id: str
    strategy_name: str

    # Capital & PnL
    starting_capital: float
    current_equity: float
    cash: float
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    pnl_by_asset: dict[str, float] = Field(default_factory=dict)

    # Positions
    positions: list[PositionInfo] = Field(default_factory=list)
    total_inventory: float = 0.0

    # Current market quotes
    quotes: list[MarketQuote] = Field(default_factory=list)

    # Recent fills
    recent_fills: list[FillInfo] = Field(default_factory=list)

    # Metrics
    total_trades: int = 0
    win_count: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0

    # Equity history for chart (last N points)
    equity_history: list[tuple[int, float]] = Field(default_factory=list)

    # Per-asset equity history for independent asset tracking
    equity_history_by_asset: dict[str, list[tuple[int, float]]] = Field(default_factory=dict)

    # Per-asset inventory history for chart
    inventory_history_by_asset: dict[str, list[tuple[int, float]]] = Field(default_factory=dict)

    # Quote history for chart (already has asset field for filtering)
    quote_history: list[QuoteHistoryPoint] = Field(default_factory=list)

    # Per-asset metrics
    metrics_by_asset: dict[str, AssetMetrics] = Field(default_factory=dict)

    # Per-asset fill history
    fills_by_asset: dict[str, list[FillInfo]] = Field(default_factory=dict)

    # Status
    status: str = "running"  # running, stopped, error
    error_message: str | None = None
    timestamp_ms: int = 0

    class Config:
        from_attributes = True


class StrategyCard(BaseModel):
    """Summary card for strategy list view."""

    strategy_id: str
    name: str
    assets: list[str]
    timeframe: str
    total_pnl: float
    pnl_percent: float
    status: str
    active_markets: int = 0
    position_count: int = 0
    pnl_by_asset: dict[str, float] = Field(default_factory=dict)


class StrategySummary(BaseModel):
    """Strategy summary for API responses."""

    strategy_id: str
    name: str
    config: dict[str, Any]
    status: str
    starting_capital: float
    current_equity: float
    total_pnl: float
    total_trades: int
    created_at: datetime | None = None


class CreateStrategyRequest(BaseModel):
    """Request to create a new strategy."""

    name: str
    assets: list[str] = Field(default_factory=lambda: ["btc"])
    timeframe: str = "15m"
    starting_capital: float = 10000.0
    quote_model: dict[str, Any] = Field(default_factory=dict)
    size_model: dict[str, Any] = Field(default_factory=dict)
    max_position_per_market: float = 100.0
    max_total_exposure: float = 1000.0


class UpdateStrategyRequest(BaseModel):
    """Request to update a strategy."""

    name: str | None = None
    quote_model: dict[str, Any] | None = None
    size_model: dict[str, Any] | None = None
    max_position_per_market: float | None = None
    max_total_exposure: float | None = None


class StrategyActionRequest(BaseModel):
    """Request for strategy actions (start/stop)."""

    action: str  # 'start', 'stop', 'restart'


class WebSocketMessage(BaseModel):
    """WebSocket message wrapper."""

    type: str  # 'state', 'fill', 'error', 'connected'
    data: dict[str, Any] | None = None
    error: str | None = None
    timestamp_ms: int = 0
