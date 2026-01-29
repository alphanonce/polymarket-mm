"""
Tests for AsyncSupabaseStore.

Verifies that the async store queues writes without blocking.
"""

import time
from unittest.mock import MagicMock, patch

import pytest


class TestAsyncSupabaseStore:
    """Tests for AsyncSupabaseStore non-blocking behavior."""

    def test_queue_trade_is_nonblocking(self) -> None:
        """Verify queue_trade returns immediately without waiting for HTTP."""
        from paper.supabase_store import AsyncSupabaseStore, SupabaseConfig

        # Create store with mock config
        config = SupabaseConfig(url="http://localhost:54321", api_key="test-key")

        with patch("paper.supabase_store.httpx.Client") as mock_client:
            mock_client.return_value = MagicMock()
            store = AsyncSupabaseStore(config)

            # Mock the parent's insert_trade to simulate slow HTTP
            call_times: list[float] = []

            def slow_insert(self: object, trade: dict[str, object]) -> None:
                time.sleep(0.1)  # Simulate 100ms HTTP request
                call_times.append(time.time())

            with patch.object(
                store.__class__.__bases__[0], "insert_trade", slow_insert
            ):
                store.start()

                # Queue multiple trades
                start = time.time()
                for i in range(5):
                    store.queue_trade({"trade_id": i, "asset_id": f"asset_{i}"})
                queue_time = time.time() - start

                # Queuing should be near-instant (< 10ms for 5 trades)
                assert queue_time < 0.01, f"Queuing took {queue_time*1000:.1f}ms, expected <10ms"

                # Wait for background writes to complete
                time.sleep(0.6)

                # All 5 trades should have been processed
                assert store._write_queue.qsize() == 0

            store.stop()

    def test_pending_writes_counter(self) -> None:
        """Verify pending_writes tracks queued items correctly."""
        from paper.supabase_store import AsyncSupabaseStore, SupabaseConfig

        config = SupabaseConfig(url="http://localhost:54321", api_key="test-key")

        with patch("paper.supabase_store.httpx.Client") as mock_client:
            mock_client.return_value = MagicMock()
            store = AsyncSupabaseStore(config)

            # Don't start the writer thread - items will stay in queue
            store.queue_trade({"trade_id": 1})
            store.queue_trade({"trade_id": 2})
            store.queue_position({"asset_id": "test"})

            assert store.pending_writes == 3

            store.stop()

    def test_queue_full_drops_writes(self) -> None:
        """Verify writes are dropped when queue is full."""
        from paper.supabase_store import AsyncSupabaseStore, SupabaseConfig

        config = SupabaseConfig(url="http://localhost:54321", api_key="test-key")

        with patch("paper.supabase_store.httpx.Client") as mock_client:
            mock_client.return_value = MagicMock()
            # Create store with tiny queue
            store = AsyncSupabaseStore(config, max_queue_size=2)

            # Don't start writer - queue will fill up
            assert store.queue_trade({"trade_id": 1}) is True
            assert store.queue_trade({"trade_id": 2}) is True
            # Third should be dropped
            assert store.queue_trade({"trade_id": 3}) is False

            store.stop()

    def test_graceful_shutdown_waits_for_pending(self) -> None:
        """Verify shutdown waits for pending writes."""
        from paper.supabase_store import AsyncSupabaseStore, SupabaseConfig

        config = SupabaseConfig(url="http://localhost:54321", api_key="test-key")

        processed_count = 0

        with patch("paper.supabase_store.httpx.Client") as mock_client:
            mock_client.return_value = MagicMock()
            store = AsyncSupabaseStore(config)

            def count_insert(self: object, trade: dict[str, object]) -> None:
                nonlocal processed_count
                time.sleep(0.05)
                processed_count += 1

            with patch.object(
                store.__class__.__bases__[0], "insert_trade", count_insert
            ):
                store.start()

                # Queue some trades
                for i in range(3):
                    store.queue_trade({"trade_id": i})

                # Stop and wait - should process all pending
                store.stop(timeout=5.0)

            # All trades should have been processed
            assert processed_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
