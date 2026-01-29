"""
Dashboard Server

FastAPI application for the paper trading dashboard.
Supports two modes:
- Monitoring mode: reads strategy state from Supabase (default)
- Trading mode: runs strategy workers that execute trades (--trading flag)
"""

import argparse
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dashboard.config import DashboardConfig
from dashboard.routes import api as api_routes
from dashboard.routes import websocket as ws_routes
from dashboard.services.broadcast import BroadcastHub
from dashboard.services.market_discovery import MarketDiscoveryService
from dashboard.services.state_poller import StatePoller, StatePollerConfig
from dashboard.services.strategy_manager import StrategyManager
from paper.supabase_store import SupabaseConfig

logger = structlog.get_logger()

# Global instances
_config: DashboardConfig | None = None
_broadcast_hub: BroadcastHub | None = None
_state_poller: StatePoller | None = None
_strategy_manager: StrategyManager | None = None
_market_discovery: MarketDiscoveryService | None = None
_trading_mode: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global _broadcast_hub, _state_poller, _strategy_manager, _market_discovery

    log = logger.bind(component="server")
    mode = "trading" if _trading_mode else "monitoring"
    log.info(f"Starting dashboard server ({mode} mode)")

    # Initialize broadcast hub
    _broadcast_hub = BroadcastHub()

    if _trading_mode and _config and _config.strategies:
        # Trading mode: run strategy workers
        log.info("Initializing trading mode components")

        # Initialize market discovery
        _market_discovery = MarketDiscoveryService()
        await _market_discovery.start()

        # Initialize strategy manager
        _strategy_manager = StrategyManager(
            config=_config,
            broadcast_hub=_broadcast_hub,
            market_discovery=_market_discovery,
            use_processes=False,  # Run in-process for now
        )
        await _strategy_manager.start()

        # Create and start all configured strategies
        for strategy_config in _config.strategies:
            await _strategy_manager.create_strategy(strategy_config, start=True)
            log.info(
                "Started strategy",
                strategy_id=strategy_config.strategy_id,
                name=strategy_config.name,
            )

        log.info(f"Started {len(_config.strategies)} strategies in trading mode")

        # Also start state poller for Supabase sync
        if _config.supabase.url:
            supabase_config = SupabaseConfig(
                url=_config.supabase.url,
                api_key=_config.supabase.api_key,
            )
            poller_config = StatePollerConfig(
                poll_interval_ms=_config.state_poller.poll_interval_ms,
                equity_history_limit=_config.state_poller.equity_history_limit,
                trade_history_limit=_config.state_poller.trade_history_limit,
                quote_history_limit=_config.state_poller.quote_history_limit,
            )
            _state_poller = StatePoller(
                supabase_config=supabase_config,
                broadcast_hub=_broadcast_hub,
                config=poller_config,
                strategy_configs=_config.strategies,
            )
            await _state_poller.start()

    else:
        # Monitoring mode: just read from Supabase
        if _config and _config.supabase.url:
            supabase_config = SupabaseConfig(
                url=_config.supabase.url,
                api_key=_config.supabase.api_key,
            )
            poller_config = StatePollerConfig(
                poll_interval_ms=_config.state_poller.poll_interval_ms,
                equity_history_limit=_config.state_poller.equity_history_limit,
                trade_history_limit=_config.state_poller.trade_history_limit,
                quote_history_limit=_config.state_poller.quote_history_limit,
            )
            _state_poller = StatePoller(
                supabase_config=supabase_config,
                broadcast_hub=_broadcast_hub,
                config=poller_config,
                strategy_configs=_config.strategies,  # Pass strategies from config
            )
            await _state_poller.start()
            log.info(f"Loaded {len(_config.strategies)} strategies from config")
        else:
            log.warning("Supabase not configured - dashboard will show no data")

    # Set instances for routes
    api_routes.set_state_poller(_state_poller)
    api_routes.set_strategy_manager(_strategy_manager)
    ws_routes.set_broadcast_hub(_broadcast_hub)
    ws_routes.set_state_poller(_state_poller)

    log.info(
        "Dashboard server started",
        host=_config.host if _config else "0.0.0.0",
        port=_config.port if _config else 8080,
        mode=mode,
    )

    yield

    # Shutdown
    log.info("Shutting down dashboard server")

    if _strategy_manager:
        await _strategy_manager.stop()

    if _market_discovery:
        await _market_discovery.stop()

    if _state_poller:
        await _state_poller.stop()


def create_app(config: DashboardConfig | None = None) -> FastAPI:
    """Create the FastAPI application."""
    global _config
    _config = config or DashboardConfig()

    app = FastAPI(
        title="Paper Trading Dashboard",
        description="Real-time monitoring dashboard for paper trading strategies",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(api_routes.router)
    app.include_router(ws_routes.router)

    # Static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Serve index.html at root
    @app.get("/")
    async def root() -> FileResponse:
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return FileResponse(str(static_dir / "index.html"))

    return app


def main() -> None:
    """Main entry point."""
    global _trading_mode

    parser = argparse.ArgumentParser(description="Paper Trading Dashboard")
    parser.add_argument(
        "--config",
        default="data/config/dashboard.yaml",
        help="Configuration file path",
    )
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument(
        "--trading",
        action="store_true",
        help="Enable trading mode (run strategy workers)",
    )
    args = parser.parse_args()

    _trading_mode = args.trading

    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = DashboardConfig.from_yaml(str(config_path))
        logger.info("Loaded config from file", path=args.config)
    else:
        config = DashboardConfig.from_env()
        logger.warning("Config file not found, using defaults", path=args.config)

    # Override with CLI args
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port

    # Create app
    app = create_app(config)

    # Run server
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
