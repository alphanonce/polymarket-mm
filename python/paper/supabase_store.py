"""
Supabase Store for Paper Trading

Direct Python client for Supabase (alternative to Go persistence layer).
Can be used standalone or alongside Go service.
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class SupabaseConfig:
    """Supabase configuration."""

    url: str
    api_key: str
    max_retries: int = 3
    timeout: float = 10.0


class SupabaseStore:
    """Direct Supabase client for paper trading data."""

    def __init__(self, config: SupabaseConfig | None = None):
        if config is None:
            config = SupabaseConfig(
                url=os.environ.get("SUPABASE_URL", ""),
                api_key=os.environ.get("SUPABASE_KEY", ""),
            )

        self._config = config
        self._client = httpx.Client(
            base_url=config.url,
            timeout=config.timeout,
            headers={
                "apikey": config.api_key,
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
        )
        self._logger = logger.bind(component="supabase_store")

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Make HTTP request with retries."""
        req_headers = dict(self._client.headers)
        if headers:
            req_headers.update(headers)

        last_error: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                resp = self._client.request(method, path, json=json, headers=req_headers)
                if resp.status_code >= 400:
                    if resp.status_code >= 500 and attempt < self._config.max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    resp.raise_for_status()
                return resp
            except httpx.HTTPError as e:
                last_error = e
                if attempt < self._config.max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))

        raise last_error or Exception("Request failed")

    def health_check(self) -> bool:
        """Check Supabase connectivity."""
        try:
            self._request("GET", "/rest/v1/metrics?select=id&limit=1")
            return True
        except Exception as e:
            self._logger.error("Health check failed", error=str(e))
            return False

    def insert_trade(self, trade: dict) -> None:
        """Insert a trade record."""
        # Convert timestamp if needed
        if "timestamp" in trade and isinstance(trade["timestamp"], (int, float)):
            trade = dict(trade)
            ts = trade["timestamp"]
            if ts > 1e12:  # Nanoseconds
                ts = ts / 1e9
            trade["timestamp"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        try:
            self._request("POST", "/rest/v1/trades", json=trade)
            self._logger.debug("Inserted trade", asset_id=trade.get("asset_id"))
        except Exception as e:
            self._logger.error("Failed to insert trade", error=str(e))

    def upsert_position(self, position: dict) -> None:
        """Insert or update a position."""
        try:
            self._request(
                "POST",
                "/rest/v1/positions?on_conflict=asset_id",
                json=position,
                headers={"Prefer": "resolution=merge-duplicates"},
            )
            self._logger.debug("Upserted position", asset_id=position.get("asset_id"))
        except Exception as e:
            self._logger.error("Failed to upsert position", error=str(e))

    def insert_equity_snapshot(
        self,
        equity: float,
        cash: float,
        position_value: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Insert an equity snapshot."""
        if timestamp is None:
            timestamp = datetime.now(tz=timezone.utc)

        try:
            self._request(
                "POST",
                "/rest/v1/equity_snapshots",
                json={
                    "equity": equity,
                    "cash": cash,
                    "position_value": position_value,
                    "timestamp": timestamp.isoformat(),
                },
            )
            self._logger.debug("Inserted equity snapshot", equity=equity)
        except Exception as e:
            self._logger.error("Failed to insert equity snapshot", error=str(e))

    def upsert_metrics(
        self,
        total_pnl: float,
        realized_pnl: float,
        unrealized_pnl: float,
        total_trades: int,
        win_rate: float,
        sharpe_ratio: float,
        max_drawdown: float,
    ) -> None:
        """Update the metrics singleton."""
        try:
            self._request(
                "POST",
                "/rest/v1/metrics?on_conflict=id",
                json={
                    "id": 1,
                    "total_pnl": total_pnl,
                    "realized_pnl": realized_pnl,
                    "unrealized_pnl": unrealized_pnl,
                    "total_trades": total_trades,
                    "win_rate": win_rate,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "updated_at": datetime.now(tz=timezone.utc).isoformat(),
                },
                headers={"Prefer": "resolution=merge-duplicates"},
            )
            self._logger.debug("Upserted metrics", total_pnl=total_pnl)
        except Exception as e:
            self._logger.error("Failed to upsert metrics", error=str(e))

    def upsert_market(
        self,
        slug: str,
        asset: str,
        timeframe: str,
        status: str = "active",
        outcome: str | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
        period_pnl: float = 0.0,
    ) -> None:
        """Insert or update a market record."""
        data: dict[str, Any] = {
            "slug": slug,
            "asset": asset,
            "timeframe": timeframe,
            "status": status,
            "period_pnl": period_pnl,
        }
        if outcome is not None:
            data["outcome"] = outcome
        if start_ts is not None:
            data["start_ts"] = start_ts
        if end_ts is not None:
            data["end_ts"] = end_ts

        try:
            self._request(
                "POST",
                "/rest/v1/markets?on_conflict=slug",
                json=data,
                headers={"Prefer": "resolution=merge-duplicates"},
            )
            self._logger.debug("Upserted market", slug=slug)
        except Exception as e:
            self._logger.error("Failed to upsert market", error=str(e))

    def insert_market_snapshot(
        self,
        slug: str,
        best_bid: float,
        best_ask: float,
        mid_price: float,
        spread: float,
        our_bid: float | None = None,
        our_ask: float | None = None,
        inventory: float = 0.0,
        inventory_value: float = 0.0,
        period_pnl: float = 0.0,
        timestamp: datetime | None = None,
    ) -> None:
        """Insert a market snapshot."""
        if timestamp is None:
            timestamp = datetime.now(tz=timezone.utc)

        data: dict[str, Any] = {
            "slug": slug,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread": spread,
            "inventory": inventory,
            "inventory_value": inventory_value,
            "period_pnl": period_pnl,
            "timestamp": timestamp.isoformat(),
        }
        if our_bid is not None:
            data["our_bid"] = our_bid
        if our_ask is not None:
            data["our_ask"] = our_ask

        try:
            self._request("POST", "/rest/v1/market_snapshots", json=data)
            self._logger.debug("Inserted market snapshot", slug=slug)
        except Exception as e:
            self._logger.error("Failed to insert market snapshot", error=str(e))

    def __enter__(self) -> "SupabaseStore":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
