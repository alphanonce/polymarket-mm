"""
Unified Interface for Binary Option Pricing Models

Provides a common interface for different probability distributions
used in binary option pricing, enabling easy comparison and backtesting.

Available models:
- Normal (Black-Scholes)
- Student's t
- Logistic (with variable kurtosis)
- Laplace/GND (with variable kurtosis)
- Mixture of Normals
- Variance Gamma
- Normal Inverse Gaussian (NIG)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from strategy.utils.distributions import (
    LAPLACE_NATURAL_KURTOSIS,
    LOGISTIC_NATURAL_KURTOSIS,
    estimate_kurtosis,
    laplace_binary_call,
    logistic_binary_call,
    mixture_binary_call,
    nig_binary_call,
    norm_cdf,
    vg_binary_call,
)


@dataclass
class DistributionParams:
    """Parameters for a probability distribution."""

    name: str
    params: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v:.4f}" for k, v in self.params.items())
        return f"DistributionParams({self.name}: {params_str})"


class BinaryOptionModel(ABC):
    """Abstract base class for binary option pricing models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        pass

    @abstractmethod
    def estimate_params(self, returns: np.ndarray) -> DistributionParams:
        """
        Estimate distribution parameters from historical returns.

        Args:
            returns: Array of log returns

        Returns:
            DistributionParams with estimated values
        """
        pass

    @abstractmethod
    def binary_call_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        vol: float,
        params: Optional[DistributionParams] = None,
        rate: float = 0.0,
    ) -> float:
        """
        Price a binary call option.

        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            vol: Annualized volatility
            params: Optional distribution parameters (uses defaults if None)
            rate: Risk-free rate (default 0 for crypto)

        Returns:
            Binary call price (probability of S > K at expiry, discounted)
        """
        pass

    def binary_put_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        vol: float,
        params: Optional[DistributionParams] = None,
        rate: float = 0.0,
    ) -> float:
        """Price a binary put option (1 - call)."""
        import math

        discount = math.exp(-rate * time_to_expiry) if time_to_expiry > 0 else 1.0
        return discount - self.binary_call_price(
            spot, strike, time_to_expiry, vol, params, rate
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# =============================================================================
# Normal (Black-Scholes) Model
# =============================================================================


class NormalModel(BinaryOptionModel):
    """Standard Black-Scholes model using normal distribution."""

    @property
    def name(self) -> str:
        return "normal"

    def estimate_params(self, returns: np.ndarray) -> DistributionParams:
        return DistributionParams(name=self.name, params={})

    def binary_call_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        vol: float,
        params: Optional[DistributionParams] = None,
        rate: float = 0.0,
    ) -> float:
        import math

        if time_to_expiry <= 0:
            return 1.0 if spot > strike else 0.0
        if vol <= 0 or spot <= 0 or strike <= 0:
            return 0.0

        sqrt_t = math.sqrt(time_to_expiry)
        d2 = (math.log(spot / strike) + (rate - 0.5 * vol**2) * time_to_expiry) / (
            vol * sqrt_t
        )
        return math.exp(-rate * time_to_expiry) * norm_cdf(d2)


# =============================================================================
# Student's t Model
# =============================================================================


class StudentTModel(BinaryOptionModel):
    """Student's t-distribution model for fat-tailed returns."""

    def __init__(self, default_df: float = 5.0):
        self.default_df = default_df

    @property
    def name(self) -> str:
        return "student_t"

    def estimate_params(self, returns: np.ndarray) -> DistributionParams:
        """Estimate degrees of freedom from kurtosis."""
        clean_returns = returns[np.isfinite(returns)]
        if len(clean_returns) < 10:
            return DistributionParams(name=self.name, params={"df": self.default_df})

        k = estimate_kurtosis(clean_returns)

        if k <= 0:
            df = 100.0
        else:
            df = 6.0 / k + 4.0

        df = float(np.clip(df, 2.1, 100.0))
        return DistributionParams(name=self.name, params={"df": df})

    def binary_call_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        vol: float,
        params: Optional[DistributionParams] = None,
        rate: float = 0.0,
    ) -> float:
        import math

        from scipy.stats import t as t_dist

        if time_to_expiry <= 0:
            return 1.0 if spot > strike else 0.0
        if vol <= 0 or spot <= 0 or strike <= 0:
            return 0.0

        df = self.default_df
        if params and "df" in params.params:
            df = params.params["df"]

        sqrt_t = math.sqrt(time_to_expiry)
        d2 = (math.log(spot / strike) + (rate - 0.5 * vol**2) * time_to_expiry) / (
            vol * sqrt_t
        )

        if df > 2:
            scale_factor = math.sqrt((df - 2) / df)
            d2_scaled = d2 * scale_factor
        else:
            d2_scaled = d2

        prob = float(t_dist.cdf(d2_scaled, df))
        return math.exp(-rate * time_to_expiry) * prob


# =============================================================================
# Logistic Model (with kurtosis parameter)
# =============================================================================


class LogisticModel(BinaryOptionModel):
    """
    Logistic distribution model with variable kurtosis.

    - kurtosis = 0: Normal distribution
    - kurtosis = 1.2: Standard logistic (default)
    - kurtosis > 1.2: Heavier tails
    """

    def __init__(self, default_kurtosis: float = LOGISTIC_NATURAL_KURTOSIS):
        self.default_kurtosis = default_kurtosis

    @property
    def name(self) -> str:
        return "logistic"

    def estimate_params(self, returns: np.ndarray) -> DistributionParams:
        """Estimate kurtosis from historical returns."""
        k = estimate_kurtosis(returns)
        # Clamp kurtosis to reasonable range for logistic
        k = float(np.clip(k, 0.0, 6.0))
        return DistributionParams(name=self.name, params={"kurtosis": k})

    def binary_call_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        vol: float,
        params: Optional[DistributionParams] = None,
        rate: float = 0.0,
    ) -> float:
        kurtosis = self.default_kurtosis
        if params and "kurtosis" in params.params:
            kurtosis = params.params["kurtosis"]

        return logistic_binary_call(spot, strike, time_to_expiry, rate, vol, kurtosis)


# =============================================================================
# Laplace Model (with kurtosis parameter)
# =============================================================================


class LaplaceModel(BinaryOptionModel):
    """
    Laplace/Generalized Normal distribution model with variable kurtosis.

    - kurtosis = 0: Normal distribution
    - kurtosis = 3: Standard Laplace (default)
    - kurtosis > 3: Heavier tails than Laplace
    """

    def __init__(self, default_kurtosis: float = LAPLACE_NATURAL_KURTOSIS):
        self.default_kurtosis = default_kurtosis

    @property
    def name(self) -> str:
        return "laplace"

    def estimate_params(self, returns: np.ndarray) -> DistributionParams:
        """Estimate kurtosis from historical returns."""
        k = estimate_kurtosis(returns)
        # Clamp kurtosis to reasonable range for GND
        k = float(np.clip(k, 0.0, 6.0))
        return DistributionParams(name=self.name, params={"kurtosis": k})

    def binary_call_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        vol: float,
        params: Optional[DistributionParams] = None,
        rate: float = 0.0,
    ) -> float:
        kurtosis = self.default_kurtosis
        if params and "kurtosis" in params.params:
            kurtosis = params.params["kurtosis"]

        return laplace_binary_call(spot, strike, time_to_expiry, rate, vol, kurtosis)


# =============================================================================
# Mixture of Normals Model
# =============================================================================


class MixtureNormalModel(BinaryOptionModel):
    """Mixture of two normal distributions for jump/regime behavior."""

    def __init__(self, default_w: float = 0.8, default_sigma_ratio: float = 2.0):
        self.default_w = default_w
        self.default_sigma_ratio = default_sigma_ratio

    @property
    def name(self) -> str:
        return "mixture"

    def estimate_params(self, returns: np.ndarray) -> DistributionParams:
        """Estimate mixture parameters using kurtosis."""
        k = estimate_kurtosis(returns)

        if k <= 0:
            w, sigma_ratio = 0.95, 1.5
        elif k < 1:
            w, sigma_ratio = 0.9, 1.8
        elif k < 3:
            w, sigma_ratio = 0.8, 2.0
        elif k < 6:
            w, sigma_ratio = 0.7, 2.5
        else:
            w, sigma_ratio = 0.6, 3.0

        return DistributionParams(
            name=self.name, params={"w": w, "sigma_ratio": sigma_ratio}
        )

    def binary_call_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        vol: float,
        params: Optional[DistributionParams] = None,
        rate: float = 0.0,
    ) -> float:
        w = self.default_w
        sigma_ratio = self.default_sigma_ratio

        if params:
            w = params.params.get("w", w)
            sigma_ratio = params.params.get("sigma_ratio", sigma_ratio)

        return mixture_binary_call(
            spot, strike, time_to_expiry, rate, vol, w, sigma_ratio
        )


# =============================================================================
# Variance Gamma Model
# =============================================================================


class VarianceGammaModel(BinaryOptionModel):
    """Variance Gamma distribution model."""

    def __init__(self, default_theta: float = 0.0, default_nu: float = 0.5):
        self.default_theta = default_theta
        self.default_nu = default_nu

    @property
    def name(self) -> str:
        return "variance_gamma"

    def estimate_params(self, returns: np.ndarray) -> DistributionParams:
        """Estimate VG parameters from moments."""
        clean_returns = returns[np.isfinite(returns)]
        if len(clean_returns) < 30:
            return DistributionParams(
                name=self.name,
                params={"theta": self.default_theta, "nu": self.default_nu},
            )

        from scipy.stats import skew as calc_skew

        var = np.var(clean_returns)
        skewness = calc_skew(clean_returns)
        excess_kurt = estimate_kurtosis(clean_returns)

        if var <= 0:
            return DistributionParams(
                name=self.name,
                params={"theta": self.default_theta, "nu": self.default_nu},
            )

        nu = max(0.1, excess_kurt / 3.0) if excess_kurt > 0 else 0.5

        sigma_approx = np.sqrt(var)
        if abs(skewness) > 0.01 and nu > 0:
            theta = skewness * sigma_approx / np.sqrt(nu)
        else:
            theta = 0.0

        theta = float(np.clip(theta, -1.0, 1.0))
        nu = float(np.clip(nu, 0.1, 2.0))

        return DistributionParams(name=self.name, params={"theta": theta, "nu": nu})

    def binary_call_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        vol: float,
        params: Optional[DistributionParams] = None,
        rate: float = 0.0,
    ) -> float:
        theta = self.default_theta
        nu = self.default_nu

        if params:
            theta = params.params.get("theta", theta)
            nu = params.params.get("nu", nu)

        return vg_binary_call(spot, strike, time_to_expiry, rate, vol, theta, nu)


# =============================================================================
# Normal Inverse Gaussian (NIG) Model
# =============================================================================


class NIGModel(BinaryOptionModel):
    """Normal Inverse Gaussian distribution model."""

    def __init__(self, default_alpha: float = 1.0, default_beta: float = 0.0):
        self.default_alpha = default_alpha
        self.default_beta = default_beta

    @property
    def name(self) -> str:
        return "nig"

    def estimate_params(self, returns: np.ndarray) -> DistributionParams:
        """Estimate NIG parameters from moments."""
        clean_returns = returns[np.isfinite(returns)]
        if len(clean_returns) < 30:
            return DistributionParams(
                name=self.name,
                params={"alpha": self.default_alpha, "beta": self.default_beta},
            )

        from scipy.stats import skew as calc_skew

        var = np.var(clean_returns)
        skewness = calc_skew(clean_returns)
        excess_kurt = estimate_kurtosis(clean_returns)

        if var <= 0:
            return DistributionParams(
                name=self.name,
                params={"alpha": self.default_alpha, "beta": self.default_beta},
            )

        if excess_kurt <= 0:
            alpha = 2.0
        elif excess_kurt < 1:
            alpha = 1.5
        elif excess_kurt < 3:
            alpha = 1.0
        elif excess_kurt < 6:
            alpha = 0.7
        else:
            alpha = 0.5

        if abs(skewness) > 0.1:
            beta = np.sign(skewness) * min(abs(skewness) * 0.3, alpha * 0.9)
        else:
            beta = 0.0

        if abs(beta) >= alpha:
            beta = np.sign(beta) * alpha * 0.5

        alpha = float(np.clip(alpha, 0.3, 3.0))
        beta = float(np.clip(beta, -alpha * 0.9, alpha * 0.9))

        return DistributionParams(name=self.name, params={"alpha": alpha, "beta": beta})

    def binary_call_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        vol: float,
        params: Optional[DistributionParams] = None,
        rate: float = 0.0,
    ) -> float:
        alpha = self.default_alpha
        beta = self.default_beta

        if params:
            alpha = params.params.get("alpha", alpha)
            beta = params.params.get("beta", beta)

        return nig_binary_call(spot, strike, time_to_expiry, rate, vol, alpha, beta)


# =============================================================================
# Model Registry and Factory
# =============================================================================

MODELS: Dict[str, type] = {
    "normal": NormalModel,
    "student_t": StudentTModel,
    "logistic": LogisticModel,
    "laplace": LaplaceModel,
    "mixture": MixtureNormalModel,
    "variance_gamma": VarianceGammaModel,
    "nig": NIGModel,
}


def get_model(name: str) -> BinaryOptionModel:
    """Get a binary option pricing model by name."""
    if name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODELS[name]()


def get_all_models() -> Dict[str, BinaryOptionModel]:
    """Get all available binary option pricing models."""
    return {name: cls() for name, cls in MODELS.items()}


def list_models() -> List[str]:
    """List all available model names."""
    return list(MODELS.keys())


def compare_models(
    spot: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
    rate: float = 0.0,
    returns: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compare binary call prices across all models."""
    results = {}

    for name, model in get_all_models().items():
        params = None
        if returns is not None:
            params = model.estimate_params(returns)

        price = model.binary_call_price(
            spot, strike, time_to_expiry, vol, params, rate
        )
        results[name] = price

    return results


def model_summary(returns: np.ndarray) -> Dict[str, DistributionParams]:
    """Get estimated parameters for all models from historical returns."""
    return {
        name: model.estimate_params(returns) for name, model in get_all_models().items()
    }
