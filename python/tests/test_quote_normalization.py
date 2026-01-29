"""Tests for quote price normalization logic."""

import pytest

from strategy.models.base import NormalizationConfig, QuoteModel, QuoteResult, StrategyState
from strategy.models.quote.zspread import ZSpreadQuoteConfig, ZSpreadQuoteModel
from strategy.utils.polymarket import MAX_PRICE, MIN_PRICE


class TestNormalizeQuote:
    """Tests for normalize_quote boundary handling."""

    @pytest.fixture
    def model(self) -> ZSpreadQuoteModel:
        """Create a quote model for testing."""
        return ZSpreadQuoteModel(ZSpreadQuoteConfig())

    def test_ask_rounds_above_max_returns_zero(self, model: ZSpreadQuoteModel) -> None:
        """Ask that rounds up to > MAX_PRICE (0.99) should return 0."""
        # 0.9995 with tick=0.001 → ceil → 1.0 > 0.99 → 0
        bid, ask = model.normalize_quote(0.50, 0.9995, None)
        assert ask == 0.0
        assert bid > 0  # bid should still be valid

    def test_bid_rounds_below_min_returns_zero(self, model: ZSpreadQuoteModel) -> None:
        """Bid that rounds down to < MIN_PRICE (0.01) should return 0."""
        # 0.005 with tick=0.001 → floor → 0.005 < 0.01 → 0
        bid, ask = model.normalize_quote(0.005, 0.50, None)
        assert bid == 0.0
        assert ask > 0  # ask should still be valid

    def test_bid_rounds_to_zero_returns_zero(self, model: ZSpreadQuoteModel) -> None:
        """Bid that rounds down to 0 should return 0."""
        # 0.0005 with tick=0.001 → floor → 0.0 → 0
        bid, ask = model.normalize_quote(0.0005, 0.50, None)
        assert bid == 0.0

    def test_valid_prices_unchanged(self, model: ZSpreadQuoteModel) -> None:
        """Valid prices in normal range remain valid."""
        bid, ask = model.normalize_quote(0.45, 0.55, None)
        assert bid == 0.45
        assert ask == 0.55

    def test_boundary_prices_valid(self, model: ZSpreadQuoteModel) -> None:
        """Prices exactly at boundaries should be valid."""
        bid, ask = model.normalize_quote(0.01, 0.99, None)
        assert bid == 0.01
        assert ask == 0.99

    def test_both_sides_can_be_invalid(self, model: ZSpreadQuoteModel) -> None:
        """Both bid and ask can be invalidated."""
        bid, ask = model.normalize_quote(0.0001, 0.9999, None)
        assert bid == 0.0
        assert ask == 0.0

    def test_fine_tick_near_upper_boundary(self, model: ZSpreadQuoteModel) -> None:
        """Near 0.99, tick=0.001 is used."""
        # 0.9991 → ceil with tick=0.001 → 1.0 → invalid
        bid, ask = model.normalize_quote(0.50, 0.9991, None)
        assert ask == 0.0

    def test_fine_tick_near_lower_boundary(self, model: ZSpreadQuoteModel) -> None:
        """Near 0.01, tick=0.001 is used."""
        # 0.0099 → floor with tick=0.001 → 0.009 < 0.01 → invalid
        bid, ask = model.normalize_quote(0.0099, 0.50, None)
        assert bid == 0.0

    def test_ask_exactly_at_boundary_after_round(self, model: ZSpreadQuoteModel) -> None:
        """Ask that rounds exactly to MAX_PRICE should remain valid."""
        # 0.99 with any tick → ceil → 0.99 = MAX_PRICE → valid
        bid, ask = model.normalize_quote(0.50, 0.99, None)
        assert ask == 0.99

    def test_bid_exactly_at_boundary_after_round(self, model: ZSpreadQuoteModel) -> None:
        """Bid that rounds exactly to MIN_PRICE should remain valid."""
        # 0.01 → floor → 0.01 = MIN_PRICE → valid
        bid, ask = model.normalize_quote(0.01, 0.50, None)
        assert bid == 0.01

    def test_min_max_price_constants(self) -> None:
        """Verify MIN_PRICE and MAX_PRICE are as expected."""
        assert MIN_PRICE == 0.01
        assert MAX_PRICE == 0.99


class TestNormalizationWithClampDisabled:
    """Tests for normalize_quote with clamp_prices=False."""

    @pytest.fixture
    def model(self) -> ZSpreadQuoteModel:
        """Create a quote model with clamping disabled."""
        config = NormalizationConfig(clamp_prices=False)
        return ZSpreadQuoteModel(ZSpreadQuoteConfig(), normalization=config)

    def test_ask_still_invalidated_without_clamp(self, model: ZSpreadQuoteModel) -> None:
        """Ask > MAX_PRICE is still invalidated even without clamping."""
        bid, ask = model.normalize_quote(0.50, 0.9995, None)
        assert ask == 0.0

    def test_bid_still_invalidated_without_clamp(self, model: ZSpreadQuoteModel) -> None:
        """Bid < MIN_PRICE is still invalidated even without clamping."""
        bid, ask = model.normalize_quote(0.005, 0.50, None)
        assert bid == 0.0


class TestNormalizationWithRoundingDisabled:
    """Tests for normalize_quote with rounding disabled."""

    @pytest.fixture
    def model(self) -> ZSpreadQuoteModel:
        """Create a quote model with rounding disabled."""
        config = NormalizationConfig(round_bid_down=False, round_ask_up=False)
        return ZSpreadQuoteModel(ZSpreadQuoteConfig(), normalization=config)

    def test_no_rounding_valid_prices_pass_through(self, model: ZSpreadQuoteModel) -> None:
        """Valid prices pass through when rounding is disabled."""
        bid, ask = model.normalize_quote(0.456, 0.567, None)
        # Clamping still applies but these are valid
        assert bid == 0.456
        assert ask == 0.567

    def test_no_rounding_out_of_range_still_invalidated(self, model: ZSpreadQuoteModel) -> None:
        """Out-of-range prices still invalidated without rounding."""
        bid, ask = model.normalize_quote(0.005, 1.05, None)
        assert bid == 0.0
        assert ask == 0.0
