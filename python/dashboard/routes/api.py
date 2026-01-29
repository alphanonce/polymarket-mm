"""
REST API Routes

API endpoints for strategy monitoring (read-only from Supabase).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Query

from dashboard.models.schemas import StrategyCard, StrategySummary

if TYPE_CHECKING:
    from dashboard.services.state_poller import StatePoller
    from dashboard.services.strategy_manager import StrategyManager

router = APIRouter(prefix="/api", tags=["api"])

# This will be set by the server on startup
_state_poller: StatePoller | None = None
_strategy_manager: StrategyManager | None = None

# Available assets
AVAILABLE_ASSETS = ["btc", "eth", "sol", "xrp"]


def set_state_poller(poller: StatePoller | None) -> None:
    """Set the state poller instance."""
    global _state_poller
    _state_poller = poller


def set_strategy_manager(manager: StrategyManager | None) -> None:
    """Set the strategy manager instance."""
    global _strategy_manager
    _strategy_manager = manager


def _extract_assets_from_positions(positions: list[dict[str, Any]]) -> list[str]:
    """Extract unique assets from position slugs."""
    assets = set()
    for p in positions:
        slug = p.get("slug", "")
        if slug:
            # slug format: btc-updown-15m-down -> extract btc
            parts = slug.split("-")
            if parts:
                assets.add(parts[0].lower())
    return sorted(assets)


@router.get("/assets")
async def list_assets() -> dict[str, Any]:
    """List available assets."""
    return {
        "assets": AVAILABLE_ASSETS,
        "description": "Available assets for filtering",
    }


@router.get("/strategies", response_model=list[StrategyCard])
async def list_strategies(
    asset: str | None = Query(None, description="Filter by asset (btc, eth, sol, xrp)"),
) -> list[StrategyCard]:
    """List all strategies with summary cards. Optionally filter by asset."""
    # Try strategy manager first (trading mode), then state poller (monitoring mode)
    if _strategy_manager:
        return _strategy_manager.get_strategy_cards()
    if not _state_poller:
        raise HTTPException(status_code=503, detail="Service not ready")

    cards: list[StrategyCard] = []
    for cache in _state_poller.get_all_strategies():
        state = cache.last_state
        if state:
            # Extract assets from positions
            position_assets = [
                p.slug.split("-")[0].lower()
                for p in state.positions
                if p.slug and "-" in p.slug
            ]
            unique_assets = sorted(set(position_assets)) if position_assets else []

            # Filter by asset if specified
            if asset and asset.lower() not in unique_assets:
                # Check if this strategy has any data for this asset
                # by looking at positions and recent_fills
                has_asset = any(
                    asset.lower() in (f.slug or "").lower()
                    for f in state.recent_fills
                )
                if not has_asset and not unique_assets:
                    # Include strategies with no positions (they might trade all assets)
                    pass
                elif not has_asset:
                    continue

            pnl = state.total_pnl
            starting = state.starting_capital
            pnl_percent = (pnl / starting * 100) if starting > 0 else 0.0
            cards.append(
                StrategyCard(
                    strategy_id=cache.strategy_id,
                    name=state.strategy_name,
                    assets=unique_assets if unique_assets else AVAILABLE_ASSETS,
                    timeframe="15m",
                    total_pnl=pnl,
                    pnl_percent=pnl_percent,
                    status=cache.status,
                    active_markets=len(state.quotes),
                    position_count=len(state.positions),
                    pnl_by_asset=state.pnl_by_asset,
                )
            )
        else:
            # Include strategies without state data
            if not asset:  # Only include if no filter
                cards.append(
                    StrategyCard(
                        strategy_id=cache.strategy_id,
                        name=cache.name,
                        assets=AVAILABLE_ASSETS,
                        timeframe="15m",
                        total_pnl=0.0,
                        pnl_percent=0.0,
                        status=cache.status,
                        active_markets=0,
                        position_count=0,
                    )
                )

    return cards


@router.get("/strategies/grouped")
async def list_strategies_grouped() -> dict[str, Any]:
    """List strategies grouped by asset for navigation."""
    if not _state_poller:
        raise HTTPException(status_code=503, detail="Service not ready")

    # Group strategies by their traded assets
    grouped: dict[str, list[dict[str, Any]]] = {asset: [] for asset in AVAILABLE_ASSETS}

    for cache in _state_poller.get_all_strategies():
        state = cache.last_state
        strategy_info = {
            "strategy_id": cache.strategy_id,
            "name": cache.name,
            "status": cache.status,
            "total_pnl": state.total_pnl if state else 0.0,
            "total_trades": state.total_trades if state else 0,
        }

        if state and state.positions:
            # Extract unique assets from positions
            position_assets = {
                p.slug.split("-")[0].lower()
                for p in state.positions
                if p.slug and "-" in p.slug
            }
            for asset in position_assets:
                if asset in grouped:
                    grouped[asset].append(strategy_info)
        else:
            # Strategy with no positions - add to all assets
            for asset in AVAILABLE_ASSETS:
                grouped[asset].append(strategy_info)

    return {
        "assets": AVAILABLE_ASSETS,
        "grouped": grouped,
        "total_strategies": len(_state_poller.get_all_strategies()),
    }


@router.get("/strategies/{strategy_id}", response_model=StrategySummary)
async def get_strategy(strategy_id: str) -> StrategySummary:
    """Get strategy details."""
    # Try strategy manager first (trading mode)
    if _strategy_manager:
        summary = _strategy_manager.get_strategy_summary(strategy_id)
        if summary:
            return summary
        raise HTTPException(status_code=404, detail="Strategy not found")

    if not _state_poller:
        raise HTTPException(status_code=503, detail="Service not ready")

    cache = _state_poller.get_strategy(strategy_id)
    if not cache:
        raise HTTPException(status_code=404, detail="Strategy not found")

    state = cache.last_state
    if not state:
        return StrategySummary(
            strategy_id=strategy_id,
            name=cache.name,
            config={},
            status=cache.status,
            starting_capital=10000.0,
            current_equity=10000.0,
            total_pnl=0.0,
            total_trades=0,
        )

    return StrategySummary(
        strategy_id=strategy_id,
        name=state.strategy_name,
        config={},  # Config not available in monitoring mode
        status=cache.status,
        starting_capital=state.starting_capital,
        current_equity=state.current_equity,
        total_pnl=state.total_pnl,
        total_trades=state.total_trades,
    )


@router.get("/strategies/{strategy_id}/state")
async def get_strategy_state(
    strategy_id: str,
    asset: str | None = Query(None, description="Filter by asset (btc, eth, sol, xrp)"),
) -> dict[str, Any]:
    """Get current strategy state (snapshot). Optionally filter by asset."""
    # Try strategy manager first (trading mode)
    if _strategy_manager:
        strategy = _strategy_manager.get_strategy(strategy_id)
        if strategy:
            state_dict = strategy.last_state.copy() if strategy.last_state else {}
            if not state_dict:
                return {"status": strategy.status, "strategy_id": strategy_id}
            # Continue with filtering below
        else:
            raise HTTPException(status_code=404, detail="Strategy not found")
    elif _state_poller:
        cache = _state_poller.get_strategy(strategy_id)
        if not cache:
            raise HTTPException(status_code=404, detail="Strategy not found")
        if not cache.last_state:
            return {}
        state_dict = cache.last_state.model_dump()
    else:
        raise HTTPException(status_code=503, detail="Service not ready")

    # Filter by asset if specified
    if asset:
        asset_lower = asset.lower()

        # Filter positions by asset
        state_dict["positions"] = [
            p for p in state_dict.get("positions", [])
            if asset_lower in (p.get("slug") or "").lower()
        ]

        # Filter recent_fills by asset
        state_dict["recent_fills"] = [
            f for f in state_dict.get("recent_fills", [])
            if asset_lower in (f.get("slug") or "").lower()
        ]

        # Filter quotes by asset
        state_dict["quotes"] = [
            q for q in state_dict.get("quotes", [])
            if asset_lower in (q.get("slug") or "").lower()
        ]

        # Filter quote_history by asset
        state_dict["quote_history"] = [
            qh for qh in state_dict.get("quote_history", [])
            if asset_lower in (qh.get("slug") or "").lower()
        ]

        # Recalculate totals for filtered data
        state_dict["position_count"] = len(state_dict["positions"])
        state_dict["total_inventory"] = sum(
            p.get("size", 0) for p in state_dict["positions"]
        )

    return state_dict


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    strategy_count = len(_state_poller.get_all_strategies()) if _state_poller else 0

    return {
        "status": "healthy",
        "mode": "monitoring",
        "strategies": strategy_count,
        "assets": AVAILABLE_ASSETS,
    }
