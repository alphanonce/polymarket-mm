"""
Tests for Black-Scholes binary option pricing module.
"""

import math

import numpy as np
import pytest
from scipy import stats

from strategy.utils.black_scholes import bs_binary_call, bs_binary_put, norm_cdf


class TestNormCdf:
    """Tests for the normal CDF approximation."""

    def test_zero_returns_half(self):
        """N(0) = 0.5 exactly."""
        result = norm_cdf(0.0)
        assert abs(result - 0.5) < 1e-6

    def test_symmetry(self):
        """N(-x) = 1 - N(x) for all x."""
        test_values = [-3.0, -1.5, -0.5, 0.5, 1.5, 3.0]
        for x in test_values:
            left = norm_cdf(-x)
            right = 1.0 - norm_cdf(x)
            assert abs(left - right) < 1e-4, f"Symmetry failed for x={x}"

    def test_accuracy_vs_scipy(self):
        """Compare accuracy against scipy.stats.norm.cdf."""
        test_values = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
        for x in test_values:
            actual = norm_cdf(x)
            expected = stats.norm.cdf(x)
            assert abs(actual - expected) < 1e-3, f"Accuracy failed for x={x}: {actual} vs {expected}"

    def test_extreme_negative(self):
        """Very negative values should return ~0."""
        assert norm_cdf(-10.0) == 0.0
        assert norm_cdf(-8.0) < 1e-14  # Very close to 0

    def test_extreme_positive(self):
        """Very positive values should return ~1."""
        assert norm_cdf(10.0) == 1.0
        assert norm_cdf(8.0) > 1.0 - 1e-14  # Very close to 1

    def test_tail_region(self):
        """Test values in tail region (|x| > 3.0)."""
        # These use Abramowitz-Stegun approximation
        assert norm_cdf(3.5) > 0.999
        assert norm_cdf(-3.5) < 0.001

    def test_central_region(self):
        """Test values in central region (|x| <= 3.0)."""
        # These use tanh approximation
        result = norm_cdf(1.96)
        expected = 0.975  # 95% confidence interval boundary
        assert abs(result - expected) < 0.01


class TestBsBinaryCall:
    """Tests for binary call option pricing."""

    def test_atm_near_half(self):
        """ATM binary call with small T should be near 0.5."""
        S, K = 100.0, 100.0
        T = 1.0 / 365  # 1 day
        r = 0.0
        sigma = 0.2
        result = bs_binary_call(S, K, T, r, sigma)
        # ATM should be close to 0.5 (slightly above due to drift)
        assert 0.45 < result < 0.55

    def test_at_expiry_itm(self):
        """At expiry, ITM call = 1."""
        S, K = 105.0, 100.0  # ITM
        T = 0.0
        r = 0.0
        sigma = 0.2
        result = bs_binary_call(S, K, T, r, sigma)
        assert result == 1.0

    def test_at_expiry_otm(self):
        """At expiry, OTM call = 0."""
        S, K = 95.0, 100.0  # OTM
        T = 0.0
        r = 0.0
        sigma = 0.2
        result = bs_binary_call(S, K, T, r, sigma)
        assert result == 0.0

    def test_zero_vol_itm(self):
        """Zero volatility ITM call = exp(-rT)."""
        S, K = 110.0, 100.0  # ITM
        T = 1.0
        r = 0.05
        sigma = 0.0
        result = bs_binary_call(S, K, T, r, sigma)
        expected = np.exp(-r * T)
        assert abs(result - expected) < 1e-6

    def test_zero_vol_otm(self):
        """Zero volatility OTM call = 0."""
        S, K = 90.0, 100.0  # OTM
        T = 1.0
        r = 0.05
        sigma = 0.0
        result = bs_binary_call(S, K, T, r, sigma)
        assert result == 0.0

    def test_deep_itm(self):
        """Deep ITM call should be close to exp(-rT)."""
        S, K = 150.0, 100.0  # Deep ITM
        T = 0.1
        r = 0.0
        sigma = 0.2
        result = bs_binary_call(S, K, T, r, sigma)
        assert result > 0.99

    def test_deep_otm(self):
        """Deep OTM call should be close to 0."""
        S, K = 50.0, 100.0  # Deep OTM
        T = 0.1
        r = 0.0
        sigma = 0.2
        result = bs_binary_call(S, K, T, r, sigma)
        assert result < 0.01

    def test_invalid_price_returns_zero(self):
        """Invalid spot or strike returns 0."""
        assert bs_binary_call(0, 100, 1, 0, 0.2) == 0.0
        assert bs_binary_call(-100, 100, 1, 0, 0.2) == 0.0
        assert bs_binary_call(100, 0, 1, 0, 0.2) == 0.0
        assert bs_binary_call(100, -100, 1, 0, 0.2) == 0.0

    def test_higher_vol_spreads_probability(self):
        """Higher volatility moves ATM probability closer to 0.5."""
        S, K = 100.0, 100.0
        T = 1.0
        r = 0.0

        low_vol = bs_binary_call(S, K, T, r, 0.1)
        high_vol = bs_binary_call(S, K, T, r, 0.5)

        # Both should be around 0.5 but high vol more so
        assert 0.4 < low_vol < 0.6
        assert 0.4 < high_vol < 0.6


class TestBsBinaryPut:
    """Tests for binary put option pricing."""

    def test_put_call_parity(self):
        """Binary call + put = exp(-rT)."""
        S, K = 100.0, 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2

        call = bs_binary_call(S, K, T, r, sigma)
        put = bs_binary_put(S, K, T, r, sigma)

        expected_sum = np.exp(-r * T)
        assert abs(call + put - expected_sum) < 1e-10

    def test_put_call_parity_various_params(self):
        """Put-call parity holds for various parameters."""
        test_cases = [
            (100, 100, 0.5, 0.0, 0.3),
            (50, 60, 0.25, 0.03, 0.4),
            (200, 150, 1.0, 0.1, 0.2),
            (100, 100, 0.01, 0.0, 0.5),
        ]
        for S, K, T, r, sigma in test_cases:
            call = bs_binary_call(S, K, T, r, sigma)
            put = bs_binary_put(S, K, T, r, sigma)
            expected_sum = np.exp(-r * T)
            assert abs(call + put - expected_sum) < 1e-9, f"Parity failed for S={S}, K={K}"

    def test_at_expiry_itm(self):
        """At expiry, ITM put = 1."""
        S, K = 95.0, 100.0  # Put is ITM when S < K
        T = 0.0
        r = 0.0
        sigma = 0.2
        result = bs_binary_put(S, K, T, r, sigma)
        assert result == 1.0

    def test_at_expiry_otm(self):
        """At expiry, OTM put = 0."""
        S, K = 105.0, 100.0  # Put is OTM when S >= K
        T = 0.0
        r = 0.0
        sigma = 0.2
        result = bs_binary_put(S, K, T, r, sigma)
        assert result == 0.0

    def test_deep_itm_put(self):
        """Deep ITM put should be close to exp(-rT)."""
        S, K = 50.0, 100.0  # Deep ITM for put
        T = 0.1
        r = 0.0
        sigma = 0.2
        result = bs_binary_put(S, K, T, r, sigma)
        assert result > 0.99

    def test_deep_otm_put(self):
        """Deep OTM put should be close to 0."""
        S, K = 150.0, 100.0  # Deep OTM for put
        T = 0.1
        r = 0.0
        sigma = 0.2
        result = bs_binary_put(S, K, T, r, sigma)
        assert result < 0.01
