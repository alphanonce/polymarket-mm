"""
Alternative Distribution Binary Option Pricing

Distributions with variable kurtosis parameters:
- Logistic: kurtosis adjustable (default 1.2)
- Laplace/GND: kurtosis adjustable (default 3.0)
- Mixture of Normals, Variance Gamma, NIG
"""

import math
from typing import Tuple
import numpy as np

_SQRT_2 = math.sqrt(2.0)
_SQRT_3 = math.sqrt(3.0)
_PI = math.pi

LOGISTIC_NATURAL_KURTOSIS = 1.2
LAPLACE_NATURAL_KURTOSIS = 3.0


def norm_cdf(x: float) -> float:
    """Fast normal CDF approximation."""
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0
    if abs(x) > 3.0:
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        sign = 1.0 if x >= 0 else -1.0
        ax = abs(x)
        t = 1.0 / (1.0 + p * ax)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-ax * ax / 2.0)
        return 0.5 * (1.0 + sign * y)
    return 0.5 * (1.0 + math.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def bs_binary_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes binary call (standard normal distribution)."""
    if T <= 0:
        return 1.0 if S >= K else 0.0
    if S <= 0 or K <= 0:
        return 0.0
    if sigma <= 0:
        return math.exp(-r * T) if S >= K else 0.0
    d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return math.exp(-r * T) * norm_cdf(d2)


def bs_binary_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes binary put (standard normal distribution)."""
    return math.exp(-r * T) - bs_binary_call(S, K, T, r, sigma)


def logistic_cdf(x: float, scale: float = 1.0) -> float:
    """Logistic distribution CDF."""
    if scale <= 0:
        return 0.5 if x == 0 else (1.0 if x > 0 else 0.0)
    z = x / scale
    if z < -20.0:
        return 0.0
    if z > 20.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def logistic_cdf_with_kurtosis(x: float, kurtosis: float = 1.2) -> float:
    """Logistic CDF with adjustable kurtosis (0=normal, 1.2=standard logistic)."""
    if kurtosis <= 0:
        return norm_cdf(x)
    w = kurtosis / LOGISTIC_NATURAL_KURTOSIS
    if w >= 1.0:
        scale = (_SQRT_3 / _PI) * math.sqrt(LOGISTIC_NATURAL_KURTOSIS / kurtosis)
        return logistic_cdf(x, scale)
    scale = _SQRT_3 / _PI
    return (1.0 - w) * norm_cdf(x) + w * logistic_cdf(x, scale)


def logistic_binary_call(S: float, K: float, T: float, r: float, sigma: float, kurtosis: float = 1.2) -> float:
    """Binary call using logistic with variable kurtosis."""
    if T <= 0:
        return 1.0 if S >= K else 0.0
    if S <= 0 or K <= 0:
        return 0.0
    if sigma <= 0:
        return math.exp(-r * T) if S >= K else 0.0
    d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return math.exp(-r * T) * logistic_cdf_with_kurtosis(d2, kurtosis)


def logistic_binary_put(S: float, K: float, T: float, r: float, sigma: float, kurtosis: float = 1.2) -> float:
    """Binary put using logistic with variable kurtosis."""
    return math.exp(-r * T) - logistic_binary_call(S, K, T, r, sigma, kurtosis)


def laplace_cdf(x: float, scale: float = 1.0) -> float:
    """Laplace distribution CDF."""
    if scale <= 0:
        return 0.5 if x == 0 else (1.0 if x > 0 else 0.0)
    z = x / scale
    if z < -30.0:
        return 0.0
    if z > 30.0:
        return 1.0
    return 0.5 * math.exp(z) if x < 0 else 1.0 - 0.5 * math.exp(-z)


def _gnd_cdf(x: float, beta: float) -> float:
    """Generalized Normal Distribution CDF."""
    if beta >= 2.0:
        return norm_cdf(x)
    try:
        from scipy.stats import gennorm
        from scipy.special import gamma as gamma_func
        scale = math.sqrt(gamma_func(1.0 / beta) / gamma_func(3.0 / beta))
        return float(gennorm.cdf(x, beta, scale=scale))
    except ImportError:
        if beta >= 1.5:
            w = (2.0 - beta)
            return (1.0 - w) * norm_cdf(x) + w * laplace_cdf(x, 1.0 / _SQRT_2)
        return laplace_cdf(x, 1.0 / _SQRT_2)


def laplace_cdf_with_kurtosis(x: float, kurtosis: float = 3.0) -> float:
    """GND CDF with adjustable kurtosis (0=normal, 3=Laplace)."""
    if kurtosis <= 0:
        return norm_cdf(x)
    if kurtosis >= 6.0:
        beta = 0.5
    elif kurtosis <= 3.0:
        beta = 2.0 - kurtosis / 3.0
    else:
        beta = max(0.5, 1.0 - (kurtosis - 3.0) / 6.0)
    return _gnd_cdf(x, beta)


def laplace_binary_call(S: float, K: float, T: float, r: float, sigma: float, kurtosis: float = 3.0) -> float:
    """Binary call using GND with variable kurtosis."""
    if T <= 0:
        return 1.0 if S >= K else 0.0
    if S <= 0 or K <= 0:
        return 0.0
    if sigma <= 0:
        return math.exp(-r * T) if S >= K else 0.0
    d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return math.exp(-r * T) * laplace_cdf_with_kurtosis(d2, kurtosis)


def laplace_binary_put(S: float, K: float, T: float, r: float, sigma: float, kurtosis: float = 3.0) -> float:
    """Binary put using GND with variable kurtosis."""
    return math.exp(-r * T) - laplace_binary_call(S, K, T, r, sigma, kurtosis)


def mixture_normal_cdf(x: float, w: float = 0.8, sigma1: float = 1.0, sigma2: float = 2.0) -> float:
    """Mixture of two normals CDF."""
    w = max(0.01, min(0.99, w))
    return w * norm_cdf(x / max(0.01, sigma1)) + (1.0 - w) * norm_cdf(x / max(0.01, sigma2))


def mixture_binary_call(S: float, K: float, T: float, r: float, sigma: float, w: float = 0.8, sigma_ratio: float = 2.0) -> float:
    """Binary call using mixture of normals."""
    if T <= 0:
        return 1.0 if S >= K else 0.0
    if S <= 0 or K <= 0:
        return 0.0
    if sigma <= 0:
        return math.exp(-r * T) if S >= K else 0.0
    d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    denom = w + (1 - w) * sigma_ratio**2
    sigma1_std = 1.0 / math.sqrt(denom)
    sigma2_std = sigma_ratio / math.sqrt(denom)
    return math.exp(-r * T) * mixture_normal_cdf(d2, w, sigma1_std, sigma2_std)


def mixture_binary_put(S: float, K: float, T: float, r: float, sigma: float, w: float = 0.8, sigma_ratio: float = 2.0) -> float:
    """Binary put using mixture of normals."""
    return math.exp(-r * T) - mixture_binary_call(S, K, T, r, sigma, w, sigma_ratio)


def vg_cdf(x: float, sigma: float = 1.0, theta: float = 0.0, nu: float = 0.5) -> float:
    """Variance Gamma CDF."""
    if sigma <= 0:
        sigma = 1.0
    if nu <= 0:
        return norm_cdf((x - theta) / sigma)
    try:
        from scipy.stats import norm as scipy_norm
        from scipy.integrate import quad
        def integrand(g):
            if g <= 1e-10:
                return 0.0
            k = 1.0 / nu
            log_gamma_pdf = (k - 1) * math.log(g) - g / nu - k * math.log(nu) - math.lgamma(k)
            gamma_pdf = math.exp(log_gamma_pdf) if log_gamma_pdf > -700 else 0.0
            z = (x - theta * g) / (sigma * math.sqrt(g))
            return scipy_norm.cdf(z) * gamma_pdf
        result, _ = quad(integrand, 1e-8, 50.0, limit=100)
        return float(np.clip(result, 0.0, 1.0))
    except:
        return norm_cdf(x / math.sqrt(sigma**2 + nu * theta**2))


def vg_binary_call(S: float, K: float, T: float, r: float, sigma: float, theta: float = 0.0, nu: float = 0.5) -> float:
    """Binary call using Variance Gamma."""
    if T <= 0:
        return 1.0 if S >= K else 0.0
    if S <= 0 or K <= 0:
        return 0.0
    if sigma <= 0:
        return math.exp(-r * T) if S >= K else 0.0
    log_m = math.log(S / K) + (r - 0.5 * sigma**2) * T
    target_var = sigma**2 * T
    sigma_vg = math.sqrt(max(target_var - nu * theta**2, target_var * 0.25))
    prob = 1.0 - vg_cdf(-log_m, sigma_vg, theta * math.sqrt(T), nu)
    return math.exp(-r * T) * prob


def vg_binary_put(S: float, K: float, T: float, r: float, sigma: float, theta: float = 0.0, nu: float = 0.5) -> float:
    """Binary put using Variance Gamma."""
    return math.exp(-r * T) - vg_binary_call(S, K, T, r, sigma, theta, nu)


def nig_cdf(x: float, alpha: float = 1.0, beta: float = 0.0, delta: float = 1.0) -> float:
    """Normal Inverse Gaussian CDF."""
    alpha = max(0.1, alpha)
    if abs(beta) >= alpha:
        beta = 0.0
    delta = max(0.01, delta)
    gamma = math.sqrt(alpha**2 - beta**2)
    if delta < 0.1:
        var = delta / gamma if gamma > 0 else 1.0
        mean = delta * beta / gamma if gamma > 0 else 0.0
        return norm_cdf((x - mean) / math.sqrt(max(var, 1e-10)))
    try:
        from scipy.stats import norminvgauss
        a = delta * gamma
        b = beta / alpha if alpha > 0 else 0.0
        if a <= 0.01:
            var = delta / gamma
            mean = delta * beta / gamma
            return norm_cdf((x - mean) / math.sqrt(var))
        result = float(norminvgauss.cdf(x, a, b, loc=0, scale=delta))
        return result if not math.isnan(result) else norm_cdf(x)
    except:
        var = delta / gamma if gamma > 0 else 1.0
        mean = delta * beta / gamma if gamma > 0 else 0.0
        return norm_cdf((x - mean) / math.sqrt(max(var, 1e-10)))


def nig_binary_call(S: float, K: float, T: float, r: float, sigma: float, alpha: float = 1.0, beta: float = 0.0) -> float:
    """Binary call using NIG."""
    if T <= 0:
        return 1.0 if S >= K else 0.0
    if S <= 0 or K <= 0:
        return 0.0
    if sigma <= 0:
        return math.exp(-r * T) if S >= K else 0.0
    log_m = math.log(S / K) + (r - 0.5 * sigma**2) * T
    if abs(beta) >= alpha:
        beta = 0.0
    gamma = math.sqrt(alpha**2 - beta**2)
    delta = sigma**2 * T * gamma
    prob = 1.0 - nig_cdf(-log_m, alpha, beta, delta)
    return math.exp(-r * T) * prob


def nig_binary_put(S: float, K: float, T: float, r: float, sigma: float, alpha: float = 1.0, beta: float = 0.0) -> float:
    """Binary put using NIG."""
    return math.exp(-r * T) - nig_binary_call(S, K, T, r, sigma, alpha, beta)


def d2_from_prices(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate BS d2 parameter."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def estimate_kurtosis(returns: np.ndarray) -> float:
    """Estimate excess kurtosis from returns."""
    clean = returns[np.isfinite(returns)]
    if len(clean) < 10:
        return 0.0
    try:
        from scipy.stats import kurtosis
        return float(kurtosis(clean, fisher=True))
    except:
        mean, std = np.mean(clean), np.std(clean)
        if std <= 0:
            return 0.0
        return float(np.mean(((clean - mean) / std)**4) - 3.0)
