"""
Market Discovery Service

Automatically discovers new updown markets from Polymarket API.
"""

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class MarketInfo:
    """Information about a discovered market."""

    slug: str
    condition_id: str
    token_id_up: str
    token_id_down: str
    asset: str
    timeframe: str
    start_ts: int  # Unix timestamp in ms
    end_ts: int  # Unix timestamp in ms
    status: str = "active"  # active, resolved
    outcome: str | None = None  # up, down (only when resolved)

    @property
    def time_to_expiry_s(self) -> float:
        """Time until market expiry in seconds."""
        now_ms = int(time.time() * 1000)
        return max(0, (self.end_ts - now_ms) / 1000)

    @property
    def is_expired(self) -> bool:
        """Check if market has expired."""
        return self.time_to_expiry_s <= 0


@dataclass
class MarketDiscoveryConfig:
    """Configuration for market discovery."""

    poll_interval_s: float = 30.0
    supported_timeframes: list[str] = field(default_factory=lambda: ["15m", "1h", "4h"])
    supported_assets: list[str] = field(default_factory=lambda: ["btc", "eth", "sol", "xrp"])
    polymarket_api_url: str = "https://gamma-api.polymarket.com"


# Timeframe to seconds mapping
TIMEFRAME_SECONDS: dict[str, int] = {
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
}


SubscriberCallback = Callable[[MarketInfo], Awaitable[None]]


class MarketDiscoveryService:
    """
    Periodically discovers new updown markets from Polymarket API.

    Notifies subscribers when new markets are found.
    """

    def __init__(self, config: MarketDiscoveryConfig | None = None) -> None:
        self._config = config or MarketDiscoveryConfig()
        self._known_markets: dict[str, MarketInfo] = {}
        self._subscribers: list[SubscriberCallback] = []
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._logger = logger.bind(component="market_discovery")

        # HTTP client
        self._client = httpx.AsyncClient(
            base_url=self._config.polymarket_api_url,
            timeout=30.0,
        )

        # Slug pattern: btc-updown-15m-1234567890
        self._slug_pattern = re.compile(r"^([a-z]+)-updown-(\d+[mh])-(\d+)$")

    async def start(self) -> None:
        """Start the discovery loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._discovery_loop())
        self._logger.info("Market discovery started")

    async def stop(self) -> None:
        """Stop the discovery loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._client.aclose()
        self._logger.info("Market discovery stopped")

    def subscribe(self, callback: SubscriberCallback) -> None:
        """Subscribe to new market notifications."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: SubscriberCallback) -> None:
        """Unsubscribe from notifications."""
        self._subscribers.remove(callback)

    def get_active_markets(
        self,
        asset: str | None = None,
        timeframe: str | None = None,
    ) -> list[MarketInfo]:
        """Get currently active markets, optionally filtered."""
        markets = [
            m for m in self._known_markets.values() if m.status == "active" and not m.is_expired
        ]

        if asset:
            markets = [m for m in markets if m.asset == asset]
        if timeframe:
            markets = [m for m in markets if m.timeframe == timeframe]

        return markets

    def get_market(self, slug: str) -> MarketInfo | None:
        """Get a specific market by slug."""
        return self._known_markets.get(slug)

    async def _discovery_loop(self) -> None:
        """Main discovery loop."""
        while self._running:
            try:
                await self._discover_markets()
            except Exception as e:
                self._logger.error("Discovery error", error=str(e))

            await asyncio.sleep(self._config.poll_interval_s)

    async def _discover_markets(self) -> None:
        """Fetch and process active markets from Polymarket API.

        Uses timestamp-based slug calculation since gamma API's slug_contains
        doesn't return crypto updown markets reliably.
        """
        now = int(time.time())

        for asset in self._config.supported_assets:
            for timeframe in self._config.supported_timeframes:
                interval_s = TIMEFRAME_SECONDS.get(timeframe)
                if not interval_s:
                    continue

                # Calculate current and next market timestamps
                current_ts = (now // interval_s) * interval_s
                next_ts = current_ts + interval_s

                # Try both current and next market slots
                for market_ts in [current_ts, next_ts]:
                    slug = f"{asset}-updown-{timeframe}-{market_ts}"

                    # Skip if already known
                    if slug in self._known_markets:
                        continue

                    await self._fetch_market_by_slug(slug)

    async def _fetch_market_by_slug(self, slug: str) -> None:
        """Fetch a specific market by its slug."""
        try:
            response = await self._client.get(
                "/markets",
                params={"slug": slug},
            )

            if response.status_code != 200:
                return

            data = response.json()
            markets = data if isinstance(data, list) else []

            if markets:
                await self._process_market(markets[0])

        except httpx.HTTPError as e:
            self._logger.debug("Failed to fetch market", slug=slug, error=str(e))

    async def _process_market(self, market_data: dict[str, Any]) -> None:
        """Process a single market from API response."""
        import json as json_module

        slug: str = market_data.get("slug", "")

        # Parse slug to extract asset and timeframe
        match = self._slug_pattern.match(slug)
        if not match:
            return

        asset = match.group(1)
        timeframe = match.group(2)
        market_ts = int(match.group(3))

        # Check if timeframe is supported
        if timeframe not in self._config.supported_timeframes:
            return

        # Extract token IDs from gamma API format
        # gamma API returns clobTokenIds and outcomes as JSON strings
        token_id_up = ""
        token_id_down = ""

        clob_token_ids_str = market_data.get("clobTokenIds", "[]")
        outcomes_str = market_data.get("outcomes", "[]")

        try:
            clob_token_ids: list[str] = json_module.loads(clob_token_ids_str)
            outcomes: list[str] = json_module.loads(outcomes_str)

            for token_id, outcome in zip(clob_token_ids, outcomes):
                if outcome.lower() == "up":
                    token_id_up = token_id
                elif outcome.lower() == "down":
                    token_id_down = token_id
        except (json_module.JSONDecodeError, TypeError):
            # Fallback: try old format with tokens list
            tokens: list[dict[str, Any]] = market_data.get("tokens", [])
            for token in tokens:
                outcome_val: str = token.get("outcome", "").lower()
                if outcome_val == "up":
                    token_id_up = token.get("token_id", "")
                elif outcome_val == "down":
                    token_id_down = token.get("token_id", "")

        if not token_id_up or not token_id_down:
            return

        # Calculate timestamps
        interval_s = TIMEFRAME_SECONDS.get(timeframe, 900)
        start_ts_ms = market_ts * 1000
        end_ts_ms = (market_ts + interval_s) * 1000

        # Create market info
        market = MarketInfo(
            slug=slug,
            condition_id=str(market_data.get("conditionId", "")),
            token_id_up=token_id_up,
            token_id_down=token_id_down,
            asset=asset,
            timeframe=timeframe,
            start_ts=start_ts_ms,
            end_ts=end_ts_ms,
            status="active",
        )

        # Check if this is a new market
        if slug not in self._known_markets:
            self._known_markets[slug] = market
            self._logger.info(
                "New market discovered",
                slug=slug,
                asset=asset,
                timeframe=timeframe,
                expires_in_s=market.time_to_expiry_s,
            )

            # Notify subscribers
            await self._notify_subscribers(market)

    async def _notify_subscribers(self, market: MarketInfo) -> None:
        """Notify all subscribers of a new market."""
        for callback in self._subscribers:
            try:
                await callback(market)
            except Exception as e:
                self._logger.error(
                    "Subscriber callback failed",
                    error=str(e),
                    slug=market.slug,
                )

    def mark_resolved(self, slug: str, outcome: str) -> None:
        """Mark a market as resolved."""
        market = self._known_markets.get(slug)
        if market:
            market.status = "resolved"
            market.outcome = outcome
            self._logger.info(
                "Market resolved",
                slug=slug,
                outcome=outcome,
            )

    def cleanup_expired(self) -> int:
        """Remove expired markets from tracking. Returns count removed."""
        expired = [
            slug
            for slug, market in self._known_markets.items()
            if market.is_expired and market.status == "resolved"
        ]

        for slug in expired:
            del self._known_markets[slug]

        if expired:
            self._logger.info("Cleaned up expired markets", count=len(expired))

        return len(expired)

    async def force_refresh(self) -> None:
        """Force an immediate discovery refresh."""
        await self._discover_markets()
