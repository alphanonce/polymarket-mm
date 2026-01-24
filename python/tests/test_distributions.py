"""
Tests for Alternative Distribution Binary Option Pricing Models.

Tests cover:
1. CDF implementations against scipy reference
2. Binary option pricing edge cases
3. Numerical stability
4. Unified interface functionality
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Import fast implementations
from strategy.utils.distributions import (
    d2_from_prices,
    laplace_binary_call,
    laplace_binary_put,
    laplace_cdf,
    logistic_binary_call,
    logistic_binary_put,
    logistic_cdf,
    mixture_binary_call,
    mixture_binary_put,
    mixture_normal_cdf,
    nig_binary_call,
    nig_binary_put,
    nig_cdf,
    norm_cdf,
    vg_binary_call,
    vg_binary_put,
    vg_cdf,
)

# Import unified interface directly (bypassing analysis/__init__.py to avoid boto3 dependency)
_distribution_models_path = Path(__file__).parent.parent / "analysis" / "distribution_models.py"
_spec = importlib.util.spec_from_file_location("distribution_models", _distribution_models_path)
_distribution_models = importlib.util.module_from_spec(_spec)
sys.modules["distribution_models"] = _distribution_models
_spec.loader.exec_module(_distribution_models)

BinaryOptionModel = _distribution_models.BinaryOptionModel
DistributionParams = _distribution_models.DistributionParams
LaplaceModel = _distribution_models.LaplaceModel
LogisticModel = _distribution_models.LogisticModel
MixtureNormalModel = _distribution_models.MixtureNormalModel
NIGModel = _distribution_models.NIGModel
NormalModel = _distribution_models.NormalModel
StudentTModel = _distribution_models.StudentTModel
VarianceGammaModel = _distribution_models.VarianceGammaModel
compare_models = _distribution_models.compare_models
get_all_models = _distribution_models.get_all_models
get_model = _distribution_models.get_model
list_models = _distribution_models.list_models
model_summary = _distribution_models.model_summary


# =============================================================================
# CDF Tests
# =============================================================================


class TestNormCDF:
    """Tests for normal CDF implementation."""

    def test_symmetry(self):
        """CDF should satisfy N(-x) = 1 - N(x)."""
        for x in [0.5, 1.0, 2.0, 3.5]:
            assert abs(norm_cdf(-x) - (1.0 - norm_cdf(x))) < 1e-6

    def test_bounds(self):
        """CDF should be in [0, 1]."""
        for x in np.linspace(-10, 10, 100):
            cdf = norm_cdf(x)
            assert 0.0 <= cdf <= 1.0

    def test_known_values(self):
        """Test against known values."""
        assert abs(norm_cdf(0.0) - 0.5) < 1e-6
        assert abs(norm_cdf(1.96) - 0.975) < 0.001
        assert abs(norm_cdf(-1.96) - 0.025) < 0.001

    def test_extreme_values(self):
        """Extreme values should not cause overflow."""
        assert norm_cdf(-10.0) == 0.0
        assert norm_cdf(10.0) == 1.0


class TestLogisticCDF:
    """Tests for logistic CDF implementation."""

    def test_symmetry(self):
        """Logistic CDF should be symmetric around 0."""
        for x in [0.5, 1.0, 2.0]:
            assert abs(logistic_cdf(-x) - (1.0 - logistic_cdf(x))) < 1e-10

    def test_bounds(self):
        """CDF should be in [0, 1]."""
        for x in np.linspace(-20, 20, 100):
            cdf = logistic_cdf(x)
            assert 0.0 <= cdf <= 1.0

    def test_center(self):
        """CDF at 0 should be 0.5."""
        assert abs(logistic_cdf(0.0) - 0.5) < 1e-10

    def test_scale_parameter(self):
        """Larger scale should flatten the curve."""
        x = 1.0
        cdf_s1 = logistic_cdf(x, scale=1.0)
        cdf_s2 = logistic_cdf(x, scale=2.0)
        # With larger scale, CDF at same x should be closer to 0.5
        assert abs(cdf_s2 - 0.5) < abs(cdf_s1 - 0.5)

    def test_scipy_comparison(self):
        """Compare against scipy reference."""
        pytest.importorskip("scipy")
        from scipy.stats import logistic as scipy_logistic

        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            for scale in [0.5, 1.0, 2.0]:
                expected = scipy_logistic.cdf(x, scale=scale)
                actual = logistic_cdf(x, scale=scale)
                assert abs(actual - expected) < 1e-10, f"Failed at x={x}, scale={scale}"


class TestLaplaceCDF:
    """Tests for Laplace CDF implementation."""

    def test_symmetry(self):
        """Laplace CDF should be symmetric around 0."""
        for x in [0.5, 1.0, 2.0]:
            assert abs(laplace_cdf(-x) - (1.0 - laplace_cdf(x))) < 1e-10

    def test_bounds(self):
        """CDF should be in [0, 1]."""
        for x in np.linspace(-30, 30, 100):
            cdf = laplace_cdf(x)
            assert 0.0 <= cdf <= 1.0

    def test_center(self):
        """CDF at 0 should be 0.5."""
        assert abs(laplace_cdf(0.0) - 0.5) < 1e-10

    def test_scipy_comparison(self):
        """Compare against scipy reference."""
        pytest.importorskip("scipy")
        from scipy.stats import laplace as scipy_laplace

        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            for scale in [0.5, 1.0, 2.0]:
                expected = scipy_laplace.cdf(x, scale=scale)
                actual = laplace_cdf(x, scale=scale)
                assert abs(actual - expected) < 1e-10, f"Failed at x={x}, scale={scale}"


class TestMixtureCDF:
    """Tests for mixture of normals CDF."""

    def test_bounds(self):
        """CDF should be in [0, 1]."""
        for x in np.linspace(-5, 5, 50):
            cdf = mixture_normal_cdf(x, w=0.8, sigma1=1.0, sigma2=2.0)
            assert 0.0 <= cdf <= 1.0

    def test_degenerate_cases(self):
        """w=1 should give normal CDF with sigma1."""
        for x in [-1.0, 0.0, 1.0]:
            mix = mixture_normal_cdf(x, w=0.999, sigma1=1.0, sigma2=2.0)
            normal = norm_cdf(x)
            assert abs(mix - normal) < 0.01

    def test_symmetry(self):
        """Symmetric mixture should be symmetric."""
        for x in [0.5, 1.0, 2.0]:
            cdf_pos = mixture_normal_cdf(x, w=0.8, sigma1=1.0, sigma2=2.0)
            cdf_neg = mixture_normal_cdf(-x, w=0.8, sigma1=1.0, sigma2=2.0)
            assert abs(cdf_pos + cdf_neg - 1.0) < 1e-6


class TestVGCDF:
    """Tests for Variance Gamma CDF."""

    def test_bounds(self):
        """CDF should be in [0, 1]."""
        for x in np.linspace(-3, 3, 20):
            cdf = vg_cdf(x, sigma=1.0, theta=0.0, nu=0.5)
            assert 0.0 <= cdf <= 1.0

    def test_symmetric_case(self):
        """With theta=0, distribution should be symmetric."""
        for x in [0.5, 1.0, 2.0]:
            cdf_pos = vg_cdf(x, sigma=1.0, theta=0.0, nu=0.5)
            cdf_neg = vg_cdf(-x, sigma=1.0, theta=0.0, nu=0.5)
            assert abs(cdf_pos + cdf_neg - 1.0) < 0.05  # Numerical tolerance


class TestNIGCDF:
    """Tests for Normal Inverse Gaussian CDF."""

    def test_bounds(self):
        """CDF should be in [0, 1]."""
        for x in np.linspace(-3, 3, 20):
            cdf = nig_cdf(x, alpha=1.0, beta=0.0, delta=1.0)
            assert 0.0 <= cdf <= 1.0

    def test_symmetric_case(self):
        """With beta=0, distribution should be symmetric."""
        for x in [0.5, 1.0, 2.0]:
            cdf_pos = nig_cdf(x, alpha=1.0, beta=0.0, delta=1.0)
            cdf_neg = nig_cdf(-x, alpha=1.0, beta=0.0, delta=1.0)
            assert abs(cdf_pos + cdf_neg - 1.0) < 0.05


# =============================================================================
# Binary Option Pricing Tests
# =============================================================================


class TestBinaryOptionPricing:
    """Tests for binary option pricing functions."""

    # Common test parameters
    S = 100.0  # Spot
    K = 100.0  # Strike (ATM)
    T = 1.0 / 12  # 1 month
    r = 0.0  # Zero rate for crypto
    sigma = 0.5  # 50% vol

    def test_atm_prices_near_half(self):
        """ATM binary call with r=0 should be near 0.5."""
        for pricing_func in [
            logistic_binary_call,
            laplace_binary_call,
        ]:
            price = pricing_func(self.S, self.K, self.T, self.r, self.sigma)
            assert 0.4 < price < 0.6, f"Failed for {pricing_func.__name__}"

    def test_put_call_parity(self):
        """Binary call + put should equal discount factor."""
        discount = math.exp(-self.r * self.T)

        for call_func, put_func in [
            (logistic_binary_call, logistic_binary_put),
            (laplace_binary_call, laplace_binary_put),
            (mixture_binary_call, mixture_binary_put),
            (vg_binary_call, vg_binary_put),
            (nig_binary_call, nig_binary_put),
        ]:
            call = call_func(self.S, self.K, self.T, self.r, self.sigma)
            put = put_func(self.S, self.K, self.T, self.r, self.sigma)
            assert abs(call + put - discount) < 1e-10, f"Failed for {call_func.__name__}"

    def test_extreme_itm(self):
        """Deep ITM call should be near 1."""
        S_itm = 150.0
        K = 100.0

        for pricing_func in [
            logistic_binary_call,
            laplace_binary_call,
        ]:
            price = pricing_func(S_itm, K, self.T, self.r, self.sigma)
            assert price > 0.9, f"ITM failed for {pricing_func.__name__}"

    def test_extreme_otm(self):
        """Deep OTM call should be near 0."""
        S_otm = 50.0
        K = 100.0

        for pricing_func in [
            logistic_binary_call,
            laplace_binary_call,
        ]:
            price = pricing_func(S_otm, K, self.T, self.r, self.sigma)
            assert price < 0.1, f"OTM failed for {pricing_func.__name__}"

    def test_zero_time_to_expiry(self):
        """At expiry, price should be 0 or 1."""
        # ITM at expiry
        assert logistic_binary_call(110, 100, 0, 0, 0.5) == 1.0
        # OTM at expiry
        assert logistic_binary_call(90, 100, 0, 0, 0.5) == 0.0

    def test_zero_volatility(self):
        """With zero vol, price is deterministic."""
        # ITM
        price_itm = logistic_binary_call(110, 100, 0.1, 0, 0)
        assert price_itm == 1.0
        # OTM
        price_otm = logistic_binary_call(90, 100, 0.1, 0, 0)
        assert price_otm == 0.0

    def test_mixture_params(self):
        """Mixture model should accept custom parameters."""
        price1 = mixture_binary_call(100, 100, 0.1, 0, 0.5, w=0.8, sigma_ratio=2.0)
        price2 = mixture_binary_call(100, 100, 0.1, 0, 0.5, w=0.5, sigma_ratio=3.0)
        # Different params should give different prices
        assert price1 != price2
        # Both should be valid probabilities
        assert 0.0 < price1 < 1.0
        assert 0.0 < price2 < 1.0

    def test_vg_params(self):
        """VG model should accept custom parameters."""
        price1 = vg_binary_call(100, 100, 0.1, 0, 0.5, theta=0.0, nu=0.5)
        price2 = vg_binary_call(100, 100, 0.1, 0, 0.5, theta=0.1, nu=1.0)
        # Both should be valid probabilities
        assert 0.0 < price1 < 1.0
        assert 0.0 < price2 < 1.0

    def test_nig_params(self):
        """NIG model should accept custom parameters."""
        price1 = nig_binary_call(100, 100, 0.1, 0, 0.5, alpha=1.0, beta=0.0)
        price2 = nig_binary_call(100, 100, 0.1, 0, 0.5, alpha=0.5, beta=0.1)
        # Both should be valid probabilities
        assert 0.0 < price1 < 1.0
        assert 0.0 < price2 < 1.0


class TestD2Calculation:
    """Tests for d2 calculation utility."""

    def test_atm_d2(self):
        """ATM with r=0 should give d2 = -0.5*sigma*sqrt(T)."""
        S = K = 100.0
        T = 1.0
        r = 0.0
        sigma = 0.2

        d2 = d2_from_prices(S, K, T, r, sigma)
        expected = -0.5 * sigma * math.sqrt(T)
        assert abs(d2 - expected) < 1e-10

    def test_edge_cases(self):
        """Edge cases should return 0."""
        assert d2_from_prices(100, 100, 0, 0, 0.5) == 0.0  # T=0
        assert d2_from_prices(100, 100, 1, 0, 0) == 0.0  # sigma=0
        assert d2_from_prices(0, 100, 1, 0, 0.5) == 0.0  # S=0


# =============================================================================
# Unified Interface Tests
# =============================================================================


class TestUnifiedInterface:
    """Tests for the unified model interface."""

    def test_get_model(self):
        """Should return correct model instances."""
        assert isinstance(get_model("normal"), NormalModel)
        assert isinstance(get_model("student_t"), StudentTModel)
        assert isinstance(get_model("logistic"), LogisticModel)
        assert isinstance(get_model("laplace"), LaplaceModel)
        assert isinstance(get_model("mixture"), MixtureNormalModel)
        assert isinstance(get_model("variance_gamma"), VarianceGammaModel)
        assert isinstance(get_model("nig"), NIGModel)

    def test_get_model_invalid(self):
        """Should raise error for unknown model."""
        with pytest.raises(ValueError):
            get_model("unknown_model")

    def test_get_all_models(self):
        """Should return all models."""
        models = get_all_models()
        assert len(models) == 7
        assert "normal" in models
        assert "student_t" in models
        assert "logistic" in models

    def test_list_models(self):
        """Should list all model names."""
        names = list_models()
        assert len(names) == 7
        assert "normal" in names

    def test_model_interface(self):
        """All models should implement the interface."""
        for name, model in get_all_models().items():
            assert isinstance(model, BinaryOptionModel)
            assert hasattr(model, "name")
            assert hasattr(model, "estimate_params")
            assert hasattr(model, "binary_call_price")
            assert hasattr(model, "binary_put_price")

    def test_model_pricing_consistency(self):
        """All models should give similar prices for typical inputs."""
        S, K, T, vol, r = 100.0, 100.0, 1 / 12, 0.5, 0.0

        prices = compare_models(S, K, T, vol, r)

        # All prices should be between 0 and 1
        for name, price in prices.items():
            assert 0.0 <= price <= 1.0, f"Invalid price for {name}: {price}"

        # ATM prices should be roughly around 0.5
        for name, price in prices.items():
            assert 0.3 < price < 0.7, f"ATM price too extreme for {name}: {price}"

    def test_parameter_estimation(self):
        """Parameter estimation should work on synthetic returns."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)  # Daily returns

        for name, model in get_all_models().items():
            params = model.estimate_params(returns)
            assert isinstance(params, DistributionParams)
            assert params.name == name

    def test_model_summary(self):
        """Model summary should return params for all models."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)

        summary = model_summary(returns)
        assert len(summary) == 7
        for name, params in summary.items():
            assert isinstance(params, DistributionParams)


class TestDistributionParams:
    """Tests for DistributionParams dataclass."""

    def test_creation(self):
        """Should create params correctly."""
        params = DistributionParams(name="test", params={"a": 1.0, "b": 2.0})
        assert params.name == "test"
        assert params.params["a"] == 1.0
        assert params.params["b"] == 2.0

    def test_repr(self):
        """Should have readable repr."""
        params = DistributionParams(name="test", params={"df": 5.0})
        repr_str = repr(params)
        assert "test" in repr_str
        assert "df" in repr_str


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability under extreme inputs."""

    def test_very_small_time(self):
        """Should handle very small time to expiry."""
        T = 1e-10
        for func in [logistic_binary_call, laplace_binary_call, mixture_binary_call]:
            price = func(100, 100, T, 0, 0.5)
            assert not math.isnan(price)
            assert not math.isinf(price)

    def test_very_large_volatility(self):
        """Should handle very large volatility."""
        sigma = 10.0  # 1000% vol
        for func in [logistic_binary_call, laplace_binary_call, mixture_binary_call]:
            price = func(100, 100, 0.1, 0, sigma)
            assert not math.isnan(price)
            assert 0.0 <= price <= 1.0

    def test_very_small_volatility(self):
        """Should handle very small volatility."""
        sigma = 1e-10
        for func in [logistic_binary_call, laplace_binary_call, mixture_binary_call]:
            price = func(100, 100, 0.1, 0, sigma)
            assert not math.isnan(price)

    def test_extreme_moneyness(self):
        """Should handle extreme spot/strike ratios."""
        # Very ITM
        for func in [logistic_binary_call, laplace_binary_call]:
            price = func(1000, 100, 0.1, 0, 0.5)
            assert not math.isnan(price)
            assert price > 0.99

        # Very OTM
        for func in [logistic_binary_call, laplace_binary_call]:
            price = func(10, 100, 0.1, 0, 0.5)
            assert not math.isnan(price)
            assert price < 0.01

    def test_no_nan_in_pricing(self):
        """Pricing should never return NaN for valid inputs."""
        test_cases = [
            (100, 100, 0.1, 0, 0.5),
            (100, 100, 1.0, 0.05, 0.2),
            (50, 100, 0.5, 0, 1.0),
            (150, 100, 0.25, 0, 0.3),
        ]

        for S, K, T, r, sigma in test_cases:
            for func in [
                logistic_binary_call,
                laplace_binary_call,
                mixture_binary_call,
                vg_binary_call,
                nig_binary_call,
            ]:
                price = func(S, K, T, r, sigma)
                assert not math.isnan(price), f"NaN from {func.__name__} with {(S, K, T, r, sigma)}"
                assert not math.isinf(price), f"Inf from {func.__name__} with {(S, K, T, r, sigma)}"


# =============================================================================
# Comparison Tests
# =============================================================================


class TestModelComparison:
    """Tests comparing models to each other."""

    def test_all_models_bounded(self):
        """All models should give prices in [0, 1]."""
        params_list = [
            (100, 100, 0.1, 0, 0.5),  # ATM
            (110, 100, 0.1, 0, 0.5),  # ITM
            (90, 100, 0.1, 0, 0.5),  # OTM
            (100, 100, 0.01, 0, 0.5),  # Short expiry
            (100, 100, 1.0, 0, 0.5),  # Long expiry
        ]

        for S, K, T, r, sigma in params_list:
            prices = compare_models(S, K, T, sigma, r)
            for name, price in prices.items():
                assert (
                    0.0 <= price <= 1.0
                ), f"Invalid price {price} for {name} with params {(S, K, T, r, sigma)}"

    def test_fat_tails_effect(self):
        """Fat-tailed distributions should differ from normal for OTM options."""
        S, K, T, r, sigma = 90, 100, 0.1, 0, 0.5  # OTM call

        normal_price = get_model("normal").binary_call_price(S, K, T, sigma, rate=r)
        logistic_price = get_model("logistic").binary_call_price(S, K, T, sigma, rate=r)
        laplace_price = get_model("laplace").binary_call_price(S, K, T, sigma, rate=r)

        # All prices should be in valid range
        assert 0.0 < normal_price < 1.0
        assert 0.0 < logistic_price < 1.0
        assert 0.0 < laplace_price < 1.0

        # Logistic (kurtosis ~1.2) gives prices closer to normal than Laplace
        # Laplace (kurtosis ~3) has sharper peak, so OTM prices may differ
        # The key test is that models give different prices
        assert abs(normal_price - laplace_price) > 0.01 or abs(normal_price - logistic_price) > 0.01
