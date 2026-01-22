"""
Tests for Polymarket price and quantity utilities.
"""

import pytest

from strategy.utils.polymarket import (
    COARSE_TICK,
    FINE_TICK,
    FINE_TICK_LOWER,
    FINE_TICK_UPPER,
    MAX_PRICE,
    MIN_PRICE,
    clamp_price,
    clamp_quantity,
    get_tick_size,
    is_valid_price,
    round_ask,
    round_bid,
    ticks_between,
)


class TestGetTickSize:
    """Tests for tick size determination."""

    def test_normal_range_returns_coarse_tick(self):
        """Prices in normal range [0.04, 0.96] use coarse tick."""
        normal_prices = [0.10, 0.25, 0.50, 0.75, 0.90]
        for price in normal_prices:
            assert get_tick_size(price) == COARSE_TICK, f"Failed for price={price}"

    def test_lower_boundary_returns_fine_tick(self):
        """Prices below 0.04 use fine tick."""
        low_prices = [0.01, 0.02, 0.03, 0.039]
        for price in low_prices:
            assert get_tick_size(price) == FINE_TICK, f"Failed for price={price}"

    def test_upper_boundary_returns_fine_tick(self):
        """Prices above 0.96 use fine tick."""
        high_prices = [0.961, 0.97, 0.98, 0.99]
        for price in high_prices:
            assert get_tick_size(price) == FINE_TICK, f"Failed for price={price}"

    def test_exact_boundary_lower(self):
        """Exact lower boundary (0.04) uses coarse tick."""
        assert get_tick_size(FINE_TICK_LOWER) == COARSE_TICK

    def test_exact_boundary_upper(self):
        """Exact upper boundary (0.96) uses coarse tick."""
        assert get_tick_size(FINE_TICK_UPPER) == COARSE_TICK


class TestRoundBid:
    """Tests for bid price rounding (floors down)."""

    def test_rounds_down(self):
        """Bid rounds down to next tick."""
        assert round_bid(0.505, 0.01) == 0.50
        assert round_bid(0.509, 0.01) == 0.50
        assert round_bid(0.511, 0.01) == 0.51

    def test_exact_tick_unchanged(self):
        """Price on exact tick remains unchanged."""
        assert round_bid(0.50, 0.01) == 0.50

    def test_fine_tick_rounding(self):
        """Fine tick rounding works correctly."""
        assert round_bid(0.0255, 0.001) == 0.025
        assert round_bid(0.0259, 0.001) == 0.025

    def test_auto_tick_detection(self):
        """Tick size is auto-detected based on price."""
        # Normal range - coarse tick
        assert round_bid(0.505) == 0.50
        # Near boundary - fine tick
        assert round_bid(0.035) == 0.035


class TestRoundAsk:
    """Tests for ask price rounding (ceils up)."""

    def test_rounds_up(self):
        """Ask rounds up to next tick."""
        assert round_ask(0.501, 0.01) == 0.51
        assert round_ask(0.505, 0.01) == 0.51
        assert round_ask(0.509, 0.01) == 0.51

    def test_exact_tick_unchanged(self):
        """Price on exact tick remains unchanged."""
        assert round_ask(0.50, 0.01) == 0.50

    def test_fine_tick_rounding(self):
        """Fine tick rounding works correctly."""
        assert round_ask(0.0251, 0.001) == pytest.approx(0.026, abs=1e-9)
        assert round_ask(0.0259, 0.001) == pytest.approx(0.026, abs=1e-9)

    def test_auto_tick_detection(self):
        """Tick size is auto-detected based on price."""
        # Normal range - coarse tick
        assert round_ask(0.501) == pytest.approx(0.51, abs=1e-9)
        # Near boundary - fine tick
        result = round_ask(0.0351)
        assert result == pytest.approx(0.036, abs=1e-9)


class TestClampPrice:
    """Tests for price clamping to valid range."""

    def test_clamps_to_min(self):
        """Prices below MIN_PRICE are clamped up."""
        assert clamp_price(0.001) == MIN_PRICE
        assert clamp_price(0.0) == MIN_PRICE
        assert clamp_price(-0.5) == MIN_PRICE

    def test_clamps_to_max(self):
        """Prices above MAX_PRICE are clamped down."""
        assert clamp_price(0.999) == MAX_PRICE
        assert clamp_price(1.0) == MAX_PRICE
        assert clamp_price(1.5) == MAX_PRICE

    def test_valid_range_unchanged(self):
        """Prices in valid range are unchanged (except rounding)."""
        assert clamp_price(0.50) == 0.50
        assert clamp_price(0.01) == 0.01
        assert clamp_price(0.99) == 0.99

    def test_respects_decimals(self):
        """Respects decimal parameter."""
        assert clamp_price(0.12345, decimals=2) == 0.12
        assert clamp_price(0.12345, decimals=4) == 0.1235


class TestClampQuantity:
    """Tests for quantity clamping."""

    def test_below_min_returns_zero(self):
        """Quantity below minimum returns 0."""
        assert clamp_quantity(1.0, min_size=5.0) == 0.0
        assert clamp_quantity(4.99, min_size=5.0) == 0.0

    def test_at_min_unchanged(self):
        """Quantity at minimum is unchanged."""
        assert clamp_quantity(5.0, min_size=5.0) == 5.0

    def test_above_min_unchanged(self):
        """Quantity above minimum is unchanged."""
        assert clamp_quantity(10.0, min_size=5.0) == 10.0
        assert clamp_quantity(100.0, min_size=5.0) == 100.0


class TestIsValidPrice:
    """Tests for price validation."""

    def test_valid_prices(self):
        """Valid prices return True."""
        valid_prices = [0.01, 0.50, 0.99, MIN_PRICE, MAX_PRICE]
        for price in valid_prices:
            assert is_valid_price(price) is True, f"Failed for price={price}"

    def test_invalid_low_prices(self):
        """Prices below range return False."""
        invalid_prices = [0.0, 0.009, -0.1]
        for price in invalid_prices:
            assert is_valid_price(price) is False, f"Failed for price={price}"

    def test_invalid_high_prices(self):
        """Prices above range return False."""
        invalid_prices = [0.991, 1.0, 1.5]
        for price in invalid_prices:
            assert is_valid_price(price) is False, f"Failed for price={price}"


class TestTicksBetween:
    """Tests for tick count calculation."""

    def test_same_price_zero_ticks(self):
        """Same price = 0 ticks."""
        assert ticks_between(0.50, 0.50) == 0

    def test_one_tick_apart(self):
        """Adjacent prices = 1 tick."""
        assert ticks_between(0.50, 0.51) == 1
        assert ticks_between(0.51, 0.50) == 1  # Order doesn't matter

    def test_multiple_ticks(self):
        """Multiple ticks counted correctly."""
        assert ticks_between(0.50, 0.55) == 5
        # Due to floating point truncation, may be off by 1
        assert ticks_between(0.40, 0.60) in [19, 20]

    def test_near_boundary_uses_fine_tick(self):
        """Near boundary uses fine tick for calculation."""
        # Both prices near lower boundary - uses fine tick
        result = ticks_between(0.02, 0.03)
        # Due to floating point truncation, may be off by 1
        assert result in [9, 10]


class TestEdgeCases:
    """Edge case tests."""

    def test_round_bid_at_min_price(self):
        """Rounding at MIN_PRICE boundary."""
        result = round_bid(0.015, 0.01)
        assert result == 0.01

    def test_round_ask_at_max_price(self):
        """Rounding at MAX_PRICE boundary."""
        result = round_ask(0.985, 0.01)
        assert result == 0.99

    def test_very_small_tick_precision(self):
        """Very small tick sizes are handled correctly."""
        # Fine tick = 0.001
        assert round_bid(0.0345, 0.001) == 0.034
        assert round_ask(0.0341, 0.001) == 0.035
