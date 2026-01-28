"""
WebSocket Broadcast Hub

Manages WebSocket connections and broadcasts updates to subscribers.
"""

import asyncio
import time
from collections import defaultdict
from typing import Any

import structlog
from fastapi import WebSocket

logger = structlog.get_logger()


class BroadcastHub:
    """
    Manages WebSocket connections per strategy and broadcasts updates.

    Thread-safe for concurrent connection/disconnection.
    """

    def __init__(self) -> None:
        # strategy_id -> set of connected WebSocket clients
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._logger = logger.bind(component="broadcast_hub")

    async def connect(self, strategy_id: str, websocket: WebSocket) -> None:
        """Register a WebSocket connection for a strategy."""
        async with self._lock:
            self._connections[strategy_id].add(websocket)
            self._logger.info(
                "WebSocket connected",
                strategy_id=strategy_id,
                total_connections=len(self._connections[strategy_id]),
            )

    async def disconnect(self, strategy_id: str, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            self._connections[strategy_id].discard(websocket)
            if not self._connections[strategy_id]:
                del self._connections[strategy_id]
            self._logger.info(
                "WebSocket disconnected",
                strategy_id=strategy_id,
            )

    async def broadcast(self, strategy_id: str, data: dict[str, Any]) -> None:
        """Broadcast data to all clients subscribed to a strategy."""
        async with self._lock:
            clients = list(self._connections.get(strategy_id, set()))

        if not clients:
            return

        # Add timestamp if not present
        if "timestamp_ms" not in data:
            data["timestamp_ms"] = int(time.time() * 1000)

        # Send to all clients, removing disconnected ones
        disconnected: list[WebSocket] = []
        for websocket in clients:
            try:
                await websocket.send_json(data)
            except Exception:
                disconnected.append(websocket)

        # Clean up disconnected clients
        if disconnected:
            async with self._lock:
                for ws in disconnected:
                    self._connections[strategy_id].discard(ws)

    async def broadcast_all(self, data: dict[str, Any]) -> None:
        """Broadcast data to all connected clients across all strategies."""
        async with self._lock:
            all_strategies = list(self._connections.keys())

        for strategy_id in all_strategies:
            await self.broadcast(strategy_id, data)

    async def broadcast_fill(self, strategy_id: str, fill_data: dict[str, Any]) -> None:
        """Broadcast a fill event to subscribers."""
        message = {
            "type": "fill",
            "data": fill_data,
            "timestamp_ms": int(time.time() * 1000),
        }
        await self.broadcast(strategy_id, message)

    async def broadcast_state(self, strategy_id: str, state_data: dict[str, Any]) -> None:
        """Broadcast state update to subscribers."""
        message = {
            "type": "state",
            "data": state_data,
            "timestamp_ms": int(time.time() * 1000),
        }
        await self.broadcast(strategy_id, message)

    async def broadcast_error(self, strategy_id: str, error: str) -> None:
        """Broadcast an error to subscribers."""
        message = {
            "type": "error",
            "error": error,
            "timestamp_ms": int(time.time() * 1000),
        }
        await self.broadcast(strategy_id, message)

    def get_connection_count(self, strategy_id: str | None = None) -> int:
        """Get the number of connected clients."""
        if strategy_id:
            return len(self._connections.get(strategy_id, set()))
        return sum(len(clients) for clients in self._connections.values())

    def get_connected_strategies(self) -> list[str]:
        """Get list of strategies with active connections."""
        return list(self._connections.keys())
