"""
Supabase Store for Paper Trading

Direct Python client for Supabase (alternative to Go persistence layer).
Can be used standalone or alongside Go service.
"""

import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

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

    def __init__(
        self,
        config: SupabaseConfig | None = None,
        strategy_id: str | None = None,
    ):
        if config is None:
            config = SupabaseConfig(
                url=os.environ.get("SUPABASE_URL", ""),
                api_key=os.environ.get("SUPABASE_KEY", ""),
            )

        self._config = config
        self._strategy_id = strategy_id
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
        self._logger = logger.bind(
            component="supabase_store",
            strategy_id=strategy_id,
        )

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
            # Use model_id which exists in the metrics table schema
            self._request("GET", "/rest/v1/metrics?select=model_id&limit=1")
            return True
        except Exception as e:
            self._logger.error("Health check failed", error=str(e))
            return False

    def insert_trade(self, trade: dict[str, Any]) -> None:
        """Insert a trade record."""
        # Convert timestamp if needed
        if "timestamp" in trade and isinstance(trade["timestamp"], (int, float)):
            trade = dict(trade)
            ts = trade["timestamp"]
            if ts > 1e12:  # Nanoseconds
                ts = ts / 1e9
            trade["timestamp"] = datetime.fromtimestamp(ts, tz=UTC).isoformat()

        # Add model_id if set (maps from strategy_id)
        if self._strategy_id and "model_id" not in trade:
            trade = dict(trade)
            trade["model_id"] = self._strategy_id

        try:
            self._request("POST", "/rest/v1/trades", json=trade)
            self._logger.debug("Inserted trade", asset_id=trade.get("asset_id"))
        except Exception as e:
            self._logger.error("Failed to insert trade", error=str(e))

    def upsert_position(self, position: dict[str, Any]) -> None:
        """Insert or update a position."""
        # Add model_id if set (maps from strategy_id)
        if self._strategy_id and "model_id" not in position:
            position = dict(position)
            position["model_id"] = self._strategy_id

        # Build conflict resolution path (model_id changes uniqueness)
        conflict_columns = "asset_id"
        if self._strategy_id:
            conflict_columns = "asset_id,model_id"

        try:
            self._request(
                "POST",
                f"/rest/v1/positions?on_conflict={conflict_columns}",
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
            timestamp = datetime.now(tz=UTC)

        data: dict[str, Any] = {
            "equity": equity,
            "cash": cash,
            "position_value": position_value,
            "timestamp": timestamp.isoformat(),
        }

        if self._strategy_id:
            data["model_id"] = self._strategy_id

        try:
            self._request("POST", "/rest/v1/equity_snapshots", json=data)
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
        """Update the metrics singleton (or per-strategy if strategy_id set)."""
        data: dict[str, Any] = {
            "total_pnl": total_pnl,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "updated_at": datetime.now(tz=UTC).isoformat(),
        }

        # Use model_id as key if set, otherwise singleton with model_id='default'
        if self._strategy_id:
            data["model_id"] = self._strategy_id
        else:
            data["model_id"] = "default"
        conflict_key = "model_id"

        try:
            self._request(
                "POST",
                f"/rest/v1/metrics?on_conflict={conflict_key}",
                json=data,
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
            timestamp = datetime.now(tz=UTC)

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

    # ========== Read Methods (for Dashboard) ==========

    def get_strategies(self) -> list[dict[str, Any]]:
        """Get all registered strategies from metrics table."""
        try:
            resp = self._request(
                "GET",
                "/rest/v1/metrics?select=model_id,updated_at&order=updated_at.desc",
                headers={"Prefer": "return=representation"},
            )
            # Map model_id to strategy_id for consistency in the interface
            result: list[dict[str, Any]] = resp.json()
            for item in result:
                if "model_id" in item:
                    item["strategy_id"] = item["model_id"]
            return result
        except Exception as e:
            self._logger.error("Failed to get strategies", error=str(e))
            return []

    def get_latest_metrics(self, strategy_id: str) -> dict[str, Any] | None:
        """Get latest metrics for a strategy."""
        try:
            resp = self._request(
                "GET",
                f"/rest/v1/metrics?model_id=eq.{strategy_id}&limit=1",
                headers={"Prefer": "return=representation"},
            )
            data = resp.json()
            return data[0] if data else None
        except Exception as e:
            self._logger.error("Failed to get metrics", error=str(e), strategy_id=strategy_id)
            return None

    def get_equity_history(
        self, strategy_id: str, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Get equity snapshots history for a strategy."""
        try:
            resp = self._request(
                "GET",
                f"/rest/v1/equity_snapshots?model_id=eq.{strategy_id}"
                f"&order=timestamp.desc&limit={limit}",
                headers={"Prefer": "return=representation"},
            )
            result: list[dict[str, Any]] = resp.json()
            return result
        except Exception as e:
            self._logger.error(
                "Failed to get equity history", error=str(e), strategy_id=strategy_id
            )
            return []

    def get_positions(self, strategy_id: str) -> list[dict[str, Any]]:
        """Get current positions for a strategy."""
        try:
            resp = self._request(
                "GET",
                f"/rest/v1/positions?model_id=eq.{strategy_id}",
                headers={"Prefer": "return=representation"},
            )
            result: list[dict[str, Any]] = resp.json()
            return result
        except Exception as e:
            self._logger.error(
                "Failed to get positions", error=str(e), strategy_id=strategy_id
            )
            return []

    def get_recent_trades(
        self, strategy_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get recent trades for a strategy."""
        try:
            resp = self._request(
                "GET",
                f"/rest/v1/trades?model_id=eq.{strategy_id}"
                f"&order=timestamp.desc&limit={limit}",
                headers={"Prefer": "return=representation"},
            )
            result: list[dict[str, Any]] = resp.json()
            return result
        except Exception as e:
            self._logger.error(
                "Failed to get recent trades", error=str(e), strategy_id=strategy_id
            )
            return []

    def get_market_snapshots(
        self, slug: str, limit: int = 500
    ) -> list[dict[str, Any]]:
        """Get market snapshots (quote history) for a slug."""
        try:
            resp = self._request(
                "GET",
                f"/rest/v1/market_snapshots?slug=eq.{slug}"
                f"&order=timestamp.desc&limit={limit}",
                headers={"Prefer": "return=representation"},
            )
            result: list[dict[str, Any]] = resp.json()
            return result
        except Exception as e:
            self._logger.error(
                "Failed to get market snapshots", error=str(e), slug=slug
            )
            return []

    def get_active_markets(self) -> list[dict[str, Any]]:
        """Get active markets from markets table."""
        try:
            resp = self._request(
                "GET",
                "/rest/v1/markets?status=eq.active&order=slug.asc",
                headers={"Prefer": "return=representation"},
            )
            result: list[dict[str, Any]] = resp.json()
            return result
        except Exception as e:
            self._logger.error("Failed to get active markets", error=str(e))
            return []

    def __enter__(self) -> "SupabaseStore":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


class AsyncSupabaseStore(SupabaseStore):
    """
    Async wrapper for SupabaseStore that processes writes in a background thread.

    This prevents tick overruns by making insert_trade/upsert_position non-blocking.
    Writes are queued and processed asynchronously by a background writer thread.
    """

    def __init__(
        self,
        config: SupabaseConfig | None = None,
        strategy_id: str | None = None,
        max_queue_size: int = 1000,
    ):
        super().__init__(config, strategy_id)
        self._write_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue(
            maxsize=max_queue_size
        )
        self._writer_thread: threading.Thread | None = None
        self._running = False
        self._pending_count = 0
        self._pending_lock = threading.Lock()

    def start(self) -> None:
        """Start the background writer thread."""
        if self._running:
            return

        self._running = True
        self._writer_thread = threading.Thread(
            target=self._write_loop,
            name="supabase-writer",
            daemon=True,
        )
        self._writer_thread.start()
        self._logger.info("Started async Supabase writer thread")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background writer thread and wait for pending writes."""
        if not self._running:
            return

        # First, wait for queue to drain (while writer is still running)
        pending = self._write_queue.qsize()
        if pending > 0:
            self._logger.info("Waiting for pending writes to complete", pending=pending)
            try:
                # Use queue.join() with a timeout check loop
                start = time.time()
                while self._write_queue.qsize() > 0 and (time.time() - start) < timeout:
                    time.sleep(0.05)
            except Exception:
                pass

        # Now stop the writer thread
        self._running = False
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=1.0)
            if self._writer_thread.is_alive():
                self._logger.warning("Writer thread did not stop cleanly")

        # Log any remaining items in queue
        remaining = self._write_queue.qsize()
        if remaining > 0:
            self._logger.warning("Shutdown with pending writes", pending_count=remaining)

    def close(self) -> None:
        """Close the store, stopping the writer thread first."""
        self.stop()
        super().close()

    def _write_loop(self) -> None:
        """Background loop that processes write requests from the queue."""
        while self._running:
            try:
                item = self._write_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._process_write(item)
            except Exception as e:
                self._logger.error("Error processing write", error=str(e))
            finally:
                self._write_queue.task_done()
                with self._pending_lock:
                    self._pending_count -= 1

    def _process_write(self, item: tuple[str, dict[str, Any]]) -> None:
        """Process a single write request."""
        write_type, data = item

        if write_type == "trade":
            super().insert_trade(data)
        elif write_type == "position":
            super().upsert_position(data)
        elif write_type == "equity_snapshot":
            super().insert_equity_snapshot(
                equity=data["equity"],
                cash=data["cash"],
                position_value=data["position_value"],
                timestamp=data.get("timestamp"),
            )
        elif write_type == "metrics":
            super().upsert_metrics(**data)
        elif write_type == "market_snapshot":
            super().insert_market_snapshot(**data)
        elif write_type == "market":
            super().upsert_market(**data)
        else:
            self._logger.warning("Unknown write type", write_type=write_type)

    def _enqueue(self, write_type: str, data: dict[str, Any]) -> bool:
        """
        Enqueue a write request. Returns True if queued, False if queue is full.
        """
        try:
            # Acquire lock before queue put to prevent race condition
            # where pending_count is checked between put and increment
            with self._pending_lock:
                self._write_queue.put_nowait((write_type, data))
                self._pending_count += 1
            return True
        except queue.Full:
            self._logger.warning(
                "Write queue full, dropping write",
                write_type=write_type,
                queue_size=self._write_queue.qsize(),
            )
            return False

    # Override write methods to be non-blocking

    def queue_trade(self, trade: dict[str, Any]) -> bool:
        """Non-blocking trade insert. Returns True if queued."""
        return self._enqueue("trade", trade)

    def queue_position(self, position: dict[str, Any]) -> bool:
        """Non-blocking position upsert. Returns True if queued."""
        return self._enqueue("position", position)

    def insert_trade(self, trade: dict[str, Any]) -> None:
        """Override to use queue (non-blocking)."""
        self.queue_trade(trade)

    def upsert_position(self, position: dict[str, Any]) -> None:
        """Override to use queue (non-blocking)."""
        self.queue_position(position)

    def insert_equity_snapshot(
        self,
        equity: float,
        cash: float,
        position_value: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Override to use queue (non-blocking)."""
        self._enqueue(
            "equity_snapshot",
            {
                "equity": equity,
                "cash": cash,
                "position_value": position_value,
                "timestamp": timestamp,
            },
        )

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
        """Override to use queue (non-blocking)."""
        self._enqueue(
            "metrics",
            {
                "total_pnl": total_pnl,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
            },
        )

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
        """Override to use queue (non-blocking)."""
        self._enqueue(
            "market_snapshot",
            {
                "slug": slug,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": mid_price,
                "spread": spread,
                "our_bid": our_bid,
                "our_ask": our_ask,
                "inventory": inventory,
                "inventory_value": inventory_value,
                "period_pnl": period_pnl,
                "timestamp": timestamp,
            },
        )

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
        """Override to use queue (non-blocking)."""
        self._enqueue(
            "market",
            {
                "slug": slug,
                "asset": asset,
                "timeframe": timeframe,
                "status": status,
                "outcome": outcome,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_pnl": period_pnl,
            },
        )

    @property
    def pending_writes(self) -> int:
        """Number of pending writes in the queue."""
        with self._pending_lock:
            return self._pending_count

    def wait_for_writes(self, timeout: float | None = None) -> bool:
        """
        Wait for all pending writes to complete.
        Returns True if all writes completed, False if timeout.
        """
        try:
            self._write_queue.join()
            return True
        except Exception:
            return False
