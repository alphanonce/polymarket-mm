"""
Shared pytest fixtures for polymarket-mm tests.
"""

from typing import List, Tuple

import pytest


@pytest.fixture
def sample_bids() -> List[Tuple[float, float]]:
    """Sample bid levels (price, size) for orderbook testing."""
    return [
        (0.50, 100.0),
        (0.49, 200.0),
        (0.48, 150.0),
        (0.47, 300.0),
        (0.46, 250.0),
    ]


@pytest.fixture
def sample_asks() -> List[Tuple[float, float]]:
    """Sample ask levels (price, size) for orderbook testing."""
    return [
        (0.51, 100.0),
        (0.52, 200.0),
        (0.53, 150.0),
        (0.54, 300.0),
        (0.55, 250.0),
    ]


@pytest.fixture
def sample_market_state() -> dict:
    """Sample market state for testing."""
    return {
        "asset_id": "test_asset_123",
        "mid_price": 0.505,
        "spread": 0.01,
        "bid_price": 0.50,
        "ask_price": 0.51,
        "timestamp_ns": 1700000000000000000,
    }


def assert_price_equal(actual: float, expected: float, tick: float = 0.01) -> None:
    """
    Assert two prices are equal within tick tolerance.

    Args:
        actual: Actual price value
        expected: Expected price value
        tick: Tick size for tolerance (default 0.01)
    """
    assert abs(actual - expected) < tick / 2, f"Price {actual} != {expected} (tick={tick})"


def assert_approx(actual: float, expected: float, rel_tol: float = 1e-6) -> None:
    """
    Assert two floats are approximately equal.

    Args:
        actual: Actual value
        expected: Expected value
        rel_tol: Relative tolerance
    """
    if expected == 0:
        assert abs(actual) < rel_tol, f"{actual} != {expected} (tol={rel_tol})"
    else:
        rel_diff = abs(actual - expected) / abs(expected)
        assert rel_diff < rel_tol, f"{actual} != {expected} (rel_diff={rel_diff}, tol={rel_tol})"
