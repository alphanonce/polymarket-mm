"""
WebSocket Routes

Real-time WebSocket endpoints for dashboard updates.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    from dashboard.services.broadcast import BroadcastHub
    from dashboard.services.state_poller import StatePoller

router = APIRouter(tags=["websocket"])

logger = structlog.get_logger()

# These will be set by the server on startup
_broadcast_hub: BroadcastHub | None = None
_state_poller: StatePoller | None = None


def set_broadcast_hub(hub: Any) -> None:
    """Set the broadcast hub instance."""
    global _broadcast_hub
    _broadcast_hub = hub


def set_state_poller(poller: StatePoller | None) -> None:
    """Set the state poller instance."""
    global _state_poller
    _state_poller = poller


@router.websocket("/ws/{strategy_id}")
async def websocket_endpoint(websocket: WebSocket, strategy_id: str) -> None:
    """
    WebSocket endpoint for real-time strategy updates.

    Clients connect to /ws/{strategy_id} to receive:
    - state: Full strategy state updates (polled from Supabase)
    - error: Error messages

    Message format:
    {
        "type": "state" | "error" | "connected",
        "data": {...},
        "error": "string" (only for error type),
        "timestamp_ms": 1234567890
    }
    """
    if not _broadcast_hub or not _state_poller:
        await websocket.close(code=1011, reason="Service not ready")
        return

    # Check if strategy exists in cache (will be discovered on next poll if new)
    cache = _state_poller.get_strategy(strategy_id)

    await websocket.accept()
    await _broadcast_hub.connect(strategy_id, websocket)

    log = logger.bind(
        component="websocket",
        strategy_id=strategy_id,
    )
    log.info("WebSocket client connected")

    try:
        # Send initial state
        await websocket.send_json(
            {
                "type": "connected",
                "data": {
                    "strategy_id": strategy_id,
                    "strategy_name": cache.name if cache else strategy_id,
                    "status": cache.status if cache else "unknown",
                },
                "timestamp_ms": int(time.time() * 1000),
            }
        )

        # Send current state if available
        if cache and cache.last_state:
            await websocket.send_json(
                {
                    "type": "state",
                    "data": cache.last_state.model_dump(),
                    "timestamp_ms": int(time.time() * 1000),
                }
            )

        # Keep connection alive and wait for disconnect
        while True:
            try:
                # Wait for client messages (ping/pong, commands)
                data = await websocket.receive_json()

                # Handle client commands
                msg_type = data.get("type", "")

                if msg_type == "ping":
                    await websocket.send_json(
                        {
                            "type": "pong",
                            "timestamp_ms": int(time.time() * 1000),
                        }
                    )
                elif msg_type == "get_state":
                    # Send current state on demand
                    cache = _state_poller.get_strategy(strategy_id)
                    if cache and cache.last_state:
                        await websocket.send_json(
                            {
                                "type": "state",
                                "data": cache.last_state.model_dump(),
                                "timestamp_ms": int(time.time() * 1000),
                            }
                        )

            except WebSocketDisconnect:
                break

    except Exception as e:
        log.error("WebSocket error", error=str(e))
    finally:
        await _broadcast_hub.disconnect(strategy_id, websocket)
        log.info("WebSocket client disconnected")


@router.websocket("/ws/all")
async def websocket_all_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for all strategies (overview).

    Receives aggregated updates from all running strategies.
    """
    if not _broadcast_hub or not _state_poller:
        await websocket.close(code=1011, reason="Service not ready")
        return

    await websocket.accept()

    # Subscribe to all known strategies
    for strategy_id in _state_poller.get_strategy_ids():
        await _broadcast_hub.connect(strategy_id, websocket)

    log = logger.bind(component="websocket", endpoint="all")
    log.info("WebSocket client connected to all strategies")

    try:
        # Send initial state for all strategies
        strategies_data = []
        for cache in _state_poller.get_all_strategies():
            strategies_data.append(
                {
                    "strategy_id": cache.strategy_id,
                    "name": cache.name,
                    "status": cache.status,
                }
            )

        await websocket.send_json(
            {
                "type": "connected",
                "data": {"strategies": strategies_data},
                "timestamp_ms": int(time.time() * 1000),
            }
        )

        while True:
            try:
                data = await websocket.receive_json()
                msg_type = data.get("type", "")

                if msg_type == "ping":
                    await websocket.send_json(
                        {
                            "type": "pong",
                            "timestamp_ms": int(time.time() * 1000),
                        }
                    )

            except WebSocketDisconnect:
                break

    except Exception as e:
        log.error("WebSocket error", error=str(e))
    finally:
        # Disconnect from all strategies
        for strategy_id in _state_poller.get_strategy_ids():
            await _broadcast_hub.disconnect(strategy_id, websocket)
        log.info("WebSocket client disconnected from all")
