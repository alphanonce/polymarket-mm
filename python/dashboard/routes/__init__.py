"""Dashboard routes."""

from dashboard.routes.api import router as api_router
from dashboard.routes.websocket import router as ws_router

__all__ = ["api_router", "ws_router"]
