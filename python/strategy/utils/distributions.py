"""
Alternative Distribution Binary Option Pricing

Fast implementations of CDFs and binary option pricing for distributions
beyond the standard normal used in Black-Scholes.

Distributions implemented:
- Logistic: Heavier tails than normal (kurtosis=1.2)
- Laplace: Sharp peak + heavy tails (kurtosis=3)
- Mixture of Normals: Captures jumps/regime changes
- Variance Gamma: Pure-jump Levy process, captures skew+kurtosis
- Normal Inverse Gaussian (NIG): Flexible skew+kurtosis
"""

import math
from typing import Tuple

import numpy as np

# Constants
_SQRT_2 = math.sqrt(2.0)
_SQRT_3 = math.sqrt(3.0)
_PI = math.pi
_SQRT_PI = math.sqrt(_PI)


# =============================================================================
# Normal CDF (from black_scholes.py for reference)
# =============================================================================


def norm_cdf(x: float) -> float:
    """
    Fast normal CDF approximation using hybrid approach.

    Central region (|x| <= 3.0): Uses tanh approximation
    Tails (|x| > 3.0): Uses Abramowitz & Stegun 26.2.17

    Args:
        x: Standard normal z-score

    Returns:
        Cumulative probability P(Z <= x)
    """
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0

    if abs(x) > 3.0:
        return _norm_cdf_tail(x)

    return 0.5 * (1.0 + math.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def _norm_cdf_tail(x: float) -> float:
    """Normal CDF for tail values using Abramowitz & Stegun 26.2.17."""
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(
        -x * x / 2.0
    )

    return 0.5 * (1.0 + sign * y)


# =============================================================================
# Logistic Distribution (with variable kurtosis)
# =============================================================================

# Natural excess kurtosis of standard logistic distribution
LOGISTIC_NATURAL_KURTOSIS = 1.2


def logistic_cdf(x: float, scale: float = 1.0) -> float:
    """
    Logistic distribution CDF.

    CDF(x) = 1 / (1 + exp(-x/s))

    The logistic distribution has heavier tails than normal (kurtosis = 1.2).
    Variance = pi^2 * s^2 / 3

    Args:
        x: Input value
        scale: Scale parameter s (default 1.0)

    Returns:
        Cumulative probability P(X <= x)
    """
    if scale <= 0:
        return 0.5 if x == 0 else (1.0 if x > 0 else 0.0)

    z = x / scale

    # Handle extreme values to avoid overflow
    if z < -20.0:
        return 0.0
    if z > 20.0:
        return 1.0

    return 1.0 / (1.0 + math.exp(-z))


def logistic_cdf_with_kurtosis(x: float, kurtosis: float = 1.2) -> float:
    """
    Logistic-Normal mixture CDF with adjustable kurtosis.

    Uses a mixture of logistic and normal distributions to achieve
    the target excess kurtosis through interpolation/extrapolation.

    - kurtosis = 0: Pure normal distribution
    - kurtosis = 1.2: Pure logistic distribution (natural kurtosis)
    - kurtosis > 1.2: Extrapolated heavier tails

    Args:
        x: Input value (standardized, variance=1)
        kurtosis: Target excess kurtosis (default 1.2)

    Returns:
        Cumulative probability P(X <= x)
    """
    if kurtosis <= 0:
        return norm_cdf(x)

    # Weight for logistic component
    # w = 1 gives pure logistic (kurtosis = 1.2)
    # w = 0 gives pure normal (kurtosis = 0)
    # w > 1 extrapolates beyond logistic
    w = kurtosis / LOGISTIC_NATURAL_KURTOSIS

    if w >= 1.0:
        # For kurtosis >= 1.2, use pure logistic with adjusted scale
        # Higher kurtosis -> smaller scale -> heavier tails
        # Approximate: kurtosis ∝ 1/scale for tail behavior
        scale_adj = LOGISTIC_NATURAL_KURTOSIS / kurtosis
        scale = (_SQRT_3 / _PI) * math.sqrt(scale_adj)
        return logistic_cdf(x, scale)
    else:
        # Interpolate between normal and logistic
        scale = _SQRT_3 / _PI  # Standard logistic scale for variance=1
        logistic_prob = logistic_cdf(x, scale)
        normal_prob = norm_cdf(x)
        return (1.0 - w) * normal_prob + w * logistic_prob


def logistic_scale_from_vol(sigma: float, T: float) -> float:
    """
    Convert annualized volatility to logistic scale parameter.

    For logistic distribution: variance = pi^2 * s^2 / 3
    For log-returns: variance = sigma^2 * T

    Solving: s = sigma * sqrt(T) * sqrt(3) / pi

    Args:
        sigma: Annualized volatility
        T: Time to expiry in years

    Returns:
        Logistic scale parameter
    """
    return sigma * math.sqrt(T) * _SQRT_3 / _PI


def logistic_binary_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kurtosis: float = 1.2,
) -> float:
    """
    Binary call option price using logistic distribution with variable kurtosis.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        kurtosis: Excess kurtosis (default 1.2 = standard logistic)
                  0 = normal, >1.2 = heavier tails

    Returns:
        Binary call price (probability of finishing ITM, discounted)
    """
    if T <= 0:
        return 1.0 if S >= K else 0.0
    if sigma <= 0:
        future_S = S * math.exp(r * T)
        return math.exp(-r * T) if future_S >= K else 0.0
    if S <= 0 or K <= 0:
        return 0.0

    # d2 equivalent for log-returns (standardized)
    sqrt_T = math.sqrt(T)
    d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * sqrt_T)

    return math.exp(-r * T) * logistic_cdf_with_kurtosis(d2, kurtosis)


def logistic_binary_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kurtosis: float = 1.2,
) -> float:
    """
    Binary put option price using logistic distribution with variable kurtosis.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        kurtosis: Excess kurtosis (default 1.2)

    Returns:
        Binary put price
    """
    return math.exp(-r * T) - logistic_binary_call(S, K, T, r, sigma, kurtosis)


# =============================================================================
# Laplace / Generalized Normal Distribution (with variable kurtosis)
# =============================================================================

# Natural excess kurtosis of standard Laplace distribution
LAPLACE_NATURAL_KURTOSIS = 3.0


def laplace_cdf(x: float, scale: float = 1.0) -> float:
    """
    Laplace (double exponential) distribution CDF.

    CDF(x) = 0.5 * exp(x/b) for x < 0
           = 1 - 0.5 * exp(-x/b) for x >= 0

    The Laplace distribution has a sharp peak and heavy tails (kurtosis = 3).
    Variance = 2 * b^2

    Args:
        x: Input value
        scale: Scale parameter b (default 1.0)

    Returns:
        Cumulative probability P(X <= x)
    """
    if scale <= 0:
        return 0.5 if x == 0 else (1.0 if x > 0 else 0.0)

    z = x / scale

    # Handle extreme values
    if z < -30.0:
        return 0.0
    if z > 30.0:
        return 1.0

    if x < 0:
        return 0.5 * math.exp(z)
    else:
        return 1.0 - 0.5 * math.exp(-z)


def _gnd_beta_from_kurtosis(kurtosis: float) -> float:
    """
    Calculate Generalized Normal Distribution shape parameter beta from kurtosis.

    For GND: excess_kurtosis = Γ(5/β)Γ(1/β) / Γ(3/β)² - 3

    Key values:
    - β = 2: Normal (kurtosis = 0)
    - β = 1: Laplace (kurtosis = 3)
    - β < 1: Heavier tails (kurtosis > 3)
    - β > 2: Lighter tails (kurtosis < 0, platykurtic)

    Uses numerical approximation for inverse mapping.

    Args:
        kurtosis: Target excess kurtosis

    Returns:
        Shape parameter β
    """
    # Handle boundary cases
    if kurtosis <= 0:
        return 2.0  # Normal

    if kurtosis >= 6.0:
        return 0.5  # Very heavy tails, cap at β=0.5

    # Approximate inverse mapping using polynomial fit
    # This is derived from numerical computation of the relationship
    # kurtosis ≈ 3/β - 3 + correction terms
    # Simplified: β ≈ 3 / (kurtosis + 3) with refinement

    # Initial estimate
    beta = 3.0 / (kurtosis + 3.0)

    # Refinement for better accuracy
    # For kurtosis=3 (Laplace): β should be 1
    # For kurtosis=0 (Normal): β should be 2
    if kurtosis <= 3.0:
        # Interpolate between normal (β=2) and Laplace (β=1)
        t = kurtosis / 3.0
        beta = 2.0 - t  # Linear interpolation
    else:
        # Extrapolate beyond Laplace
        # β = 1 - (kurtosis - 3) / 6  approximately
        beta = 1.0 - (kurtosis - 3.0) / 6.0
        beta = max(beta, 0.5)  # Cap at 0.5

    return beta


def _gnd_cdf(x: float, beta: float) -> float:
    """
    Generalized Normal Distribution CDF.

    PDF: f(x; β) = β / (2σΓ(1/β)) * exp(-(|x|/σ)^β)

    For standardized (variance=1) GND:
    σ = sqrt(Γ(1/β) / Γ(3/β))

    Args:
        x: Input value
        beta: Shape parameter (2=normal, 1=Laplace, <1=heavier)

    Returns:
        CDF value
    """
    if beta <= 0:
        beta = 0.5
    if beta >= 2.0:
        return norm_cdf(x)

    try:
        from scipy.stats import gennorm
        from scipy.special import gamma as gamma_func

        # Scale to have unit variance
        # Variance of GND = Γ(3/β) / Γ(1/β)
        # So for unit variance: scale = sqrt(Γ(1/β) / Γ(3/β))
        scale = math.sqrt(gamma_func(1.0 / beta) / gamma_func(3.0 / beta))
        return float(gennorm.cdf(x, beta, scale=scale))
    except ImportError:
        # Fallback: interpolate between normal and Laplace
        if beta >= 1.5:
            # Closer to normal
            w = (2.0 - beta) / 1.0  # w=0 at β=2, w=0.5 at β=1.5
            return (1.0 - w) * norm_cdf(x) + w * laplace_cdf(x, 1.0 / _SQRT_2)
        else:
            # Closer to Laplace or heavier
            return laplace_cdf(x, 1.0 / _SQRT_2)


def laplace_cdf_with_kurtosis(x: float, kurtosis: float = 3.0) -> float:
    """
    Generalized Normal Distribution CDF with specified kurtosis.

    Uses the Generalized Normal Distribution (GND) to achieve
    variable kurtosis while maintaining the Laplace-like shape.

    - kurtosis = 0: Normal distribution
    - kurtosis = 3: Standard Laplace distribution
    - kurtosis > 3: Heavier tails than Laplace

    Args:
        x: Input value (standardized, variance=1)
        kurtosis: Target excess kurtosis (default 3.0)

    Returns:
        Cumulative probability P(X <= x)
    """
    if kurtosis <= 0:
        return norm_cdf(x)

    beta = _gnd_beta_from_kurtosis(kurtosis)
    return _gnd_cdf(x, beta)


def laplace_scale_from_vol(sigma: float, T: float) -> float:
    """
    Convert annualized volatility to Laplace scale parameter.

    For Laplace distribution: variance = 2 * b^2
    For log-returns: variance = sigma^2 * T

    Solving: b = sigma * sqrt(T) / sqrt(2)

    Args:
        sigma: Annualized volatility
        T: Time to expiry in years

    Returns:
        Laplace scale parameter
    """
    return sigma * math.sqrt(T) / _SQRT_2


def laplace_binary_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kurtosis: float = 3.0,
) -> float:
    """
    Binary call option price using Generalized Normal distribution.

    Uses the Generalized Normal Distribution (GND) which encompasses
    both normal (kurtosis=0) and Laplace (kurtosis=3) as special cases.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        kurtosis: Excess kurtosis (default 3.0 = standard Laplace)
                  0 = normal, >3 = heavier tails

    Returns:
        Binary call price (probability of finishing ITM, discounted)
    """
    if T <= 0:
        return 1.0 if S >= K else 0.0
    if sigma <= 0:
        future_S = S * math.exp(r * T)
        return math.exp(-r * T) if future_S >= K else 0.0
    if S <= 0 or K <= 0:
        return 0.0

    # d2 equivalent for log-returns (standardized)
    sqrt_T = math.sqrt(T)
    d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * sqrt_T)

    return math.exp(-r * T) * laplace_cdf_with_kurtosis(d2, kurtosis)


def laplace_binary_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kurtosis: float = 3.0,
) -> float:
    """
    Binary put option price using Generalized Normal distribution.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        kurtosis: Excess kurtosis (default 3.0)

    Returns:
        Binary put price
    """
    return math.exp(-r * T) - laplace_binary_call(S, K, T, r, sigma, kurtosis)


# =============================================================================
# Mixture of Normals Distribution
# =============================================================================


def mixture_normal_cdf(
    x: float, w: float = 0.8, sigma1: float = 1.0, sigma2: float = 2.0
) -> float:
    """
    Mixture of two normal distributions CDF.

    CDF(x) = w * N(x/sigma1) + (1-w) * N(x/sigma2)

    This captures jump behavior: normal regime (weight w) and jump regime.

    Args:
        x: Input value
        w: Weight of first (low-vol) component (0 < w < 1)
        sigma1: Std dev of first component
        sigma2: Std dev of second component (typically larger)

    Returns:
        Cumulative probability P(X <= x)
    """
    if not (0 < w < 1):
        w = 0.8

    if sigma1 <= 0:
        sigma1 = 1.0
    if sigma2 <= 0:
        sigma2 = 2.0

    # Standardize by each component's sigma
    z1 = x / sigma1 if sigma1 > 0 else 0.0
    z2 = x / sigma2 if sigma2 > 0 else 0.0

    return w * norm_cdf(z1) + (1.0 - w) * norm_cdf(z2)


def mixture_params_from_vol(
    sigma: float, T: float, w: float = 0.8, sigma_ratio: float = 2.0
) -> Tuple[float, float, float]:
    """
    Calculate mixture component sigmas to match overall variance.

    For mixture: variance = w * sigma1^2 + (1-w) * sigma2^2
    With constraint: sigma2 = sigma_ratio * sigma1

    Solving for sigma1:
    var = sigma^2 * T = w * s1^2 + (1-w) * (ratio * s1)^2
        = s1^2 * (w + (1-w) * ratio^2)
    s1 = sigma * sqrt(T) / sqrt(w + (1-w) * ratio^2)

    Args:
        sigma: Annualized volatility
        T: Time to expiry in years
        w: Weight of first component
        sigma_ratio: Ratio sigma2/sigma1

    Returns:
        (w, sigma1, sigma2) tuple
    """
    total_var = sigma**2 * T
    denom = w + (1 - w) * sigma_ratio**2
    sigma1 = math.sqrt(total_var / denom) if denom > 0 else sigma * math.sqrt(T)
    sigma2 = sigma_ratio * sigma1

    return w, sigma1, sigma2


def mixture_binary_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    w: float = 0.8,
    sigma_ratio: float = 2.0,
) -> float:
    """
    Binary call option price using mixture of normals.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        w: Weight of normal regime (default 0.8)
        sigma_ratio: Jump regime vol multiplier (default 2.0)

    Returns:
        Binary call price (probability of finishing ITM, discounted)
    """
    if T <= 0:
        return 1.0 if S >= K else 0.0
    if sigma <= 0:
        future_S = S * math.exp(r * T)
        return math.exp(-r * T) if future_S >= K else 0.0
    if S <= 0 or K <= 0:
        return 0.0

    # d2 equivalent for log-returns
    sqrt_T = math.sqrt(T)
    d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * sqrt_T)

    # Get component sigmas that match overall variance
    w_adj, sigma1, sigma2 = mixture_params_from_vol(sigma, T, w, sigma_ratio)

    # For standardized d2 (variance=1), component sigmas scale inversely
    # We need sigma1_std and sigma2_std such that overall variance = 1
    total_std = sigma * sqrt_T
    sigma1_std = sigma1 / total_std if total_std > 0 else 1.0
    sigma2_std = sigma2 / total_std if total_std > 0 else sigma_ratio

    prob = mixture_normal_cdf(d2, w_adj, sigma1_std, sigma2_std)

    return math.exp(-r * T) * prob


def mixture_binary_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    w: float = 0.8,
    sigma_ratio: float = 2.0,
) -> float:
    """
    Binary put option price using mixture of normals.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        w: Weight of normal regime
        sigma_ratio: Jump regime vol multiplier

    Returns:
        Binary put price
    """
    return math.exp(-r * T) - mixture_binary_call(S, K, T, r, sigma, w, sigma_ratio)


# =============================================================================
# Variance Gamma Distribution
# =============================================================================


def _vg_cdf_numerical(x: float, sigma: float, theta: float, nu: float) -> float:
    """
    Variance Gamma CDF via numerical integration.

    The VG distribution is the distribution of X(t) where:
    X(t) = theta * G(t) + sigma * W(G(t))
    G(t) is a gamma process with mean t and variance nu*t
    W is a standard Brownian motion

    For unit time (t=1):
    - Mean = theta
    - Variance = sigma^2 + nu * theta^2
    - Skewness = theta * (2*nu + 3*nu^2/sigma^2) / (var^1.5)
    - Kurtosis = 3 * (1 + 2*nu - nu^2*theta^2/(sigma^2*var))

    Uses scipy for accurate CDF computation.
    """
    try:
        from scipy.stats import norm as scipy_norm
        from scipy.integrate import quad
    except ImportError:
        # Fallback to normal approximation if scipy not available
        total_var = sigma**2 + nu * theta**2
        return norm_cdf(x / math.sqrt(total_var)) if total_var > 0 else 0.5

    if nu <= 0:
        # Degenerate to normal
        return norm_cdf(x / sigma) if sigma > 0 else 0.5

    # VG density via subordinated Brownian motion
    # f(x) = integral_0^inf phi((x - theta*g) / (sigma*sqrt(g))) * gamma_pdf(g; 1/nu, nu) dg
    # where gamma_pdf(g; k, theta) = g^(k-1) * exp(-g/theta) / (theta^k * Gamma(k))

    def integrand(g: float) -> float:
        if g <= 1e-10:
            return 0.0
        k = 1.0 / nu  # shape
        scale = nu  # scale
        # Gamma density
        log_gamma_pdf = (k - 1) * math.log(g) - g / scale - k * math.log(scale)
        try:
            from scipy.special import gammaln

            log_gamma_pdf -= gammaln(k)
        except ImportError:
            log_gamma_pdf -= math.lgamma(k)
        gamma_pdf = math.exp(log_gamma_pdf) if log_gamma_pdf > -700 else 0.0

        # Normal CDF at (x - theta*g) / (sigma*sqrt(g))
        z = (x - theta * g) / (sigma * math.sqrt(g))
        return scipy_norm.cdf(z) * gamma_pdf

    # Integrate from 0 to infinity
    # Use adaptive integration with reasonable bounds
    try:
        result, _ = quad(integrand, 1e-8, 50.0, limit=100)
        return float(np.clip(result, 0.0, 1.0))
    except Exception:
        # Fallback to normal approximation
        total_var = sigma**2 + nu * theta**2
        return norm_cdf(x / math.sqrt(total_var)) if total_var > 0 else 0.5


def vg_cdf(x: float, sigma: float = 1.0, theta: float = 0.0, nu: float = 0.5) -> float:
    """
    Variance Gamma distribution CDF.

    The VG process is a pure-jump Levy process that captures both
    skewness (via theta) and excess kurtosis (via nu).

    Args:
        x: Input value
        sigma: Volatility of Brownian motion component
        theta: Drift parameter (controls skewness)
        nu: Variance of gamma subordinator (controls kurtosis)

    Returns:
        Cumulative probability P(X <= x)
    """
    if sigma <= 0:
        sigma = 1.0
    if nu <= 0:
        # Degenerate to normal
        return norm_cdf((x - theta) / sigma)

    return _vg_cdf_numerical(x, sigma, theta, nu)


def vg_params_from_vol(
    sigma: float, T: float, theta: float = 0.0, nu: float = 0.5
) -> Tuple[float, float, float]:
    """
    Scale VG parameters for given volatility and time horizon.

    VG variance = (sigma^2 + nu * theta^2) * T

    For matching BS variance (sigma_bs^2 * T):
    sigma_vg = sqrt(sigma_bs^2 * T - nu * theta^2) / sqrt(T)

    Args:
        sigma: Annualized volatility (BS equivalent)
        T: Time to expiry in years
        theta: Drift parameter
        nu: Variance rate of gamma subordinator

    Returns:
        (sigma_vg, theta, nu) scaled for time T
    """
    target_var = sigma**2 * T
    theta_contrib = nu * theta**2

    if target_var > theta_contrib:
        sigma_vg = math.sqrt(target_var - theta_contrib)
    else:
        sigma_vg = sigma * math.sqrt(T) * 0.5  # Fallback

    return sigma_vg, theta * math.sqrt(T), nu


def vg_binary_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    theta: float = 0.0,
    nu: float = 0.5,
) -> float:
    """
    Binary call option price using Variance Gamma distribution.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        theta: Drift/skewness parameter
        nu: Kurtosis parameter (variance rate of subordinator)

    Returns:
        Binary call price (probability of finishing ITM, discounted)
    """
    if T <= 0:
        return 1.0 if S >= K else 0.0
    if sigma <= 0:
        future_S = S * math.exp(r * T)
        return math.exp(-r * T) if future_S >= K else 0.0
    if S <= 0 or K <= 0:
        return 0.0

    # Log-moneyness adjusted for drift
    log_moneyness = math.log(S / K) + (r - 0.5 * sigma**2) * T

    # Scale VG parameters
    sigma_vg, theta_scaled, nu_scaled = vg_params_from_vol(sigma, T, theta, nu)

    # CDF of -log_moneyness under VG (probability of finishing ITM)
    # P(S_T > K) = P(log(S_T/K) > 0) = P(X > -log_moneyness)
    # where X is VG with scaled params
    prob = 1.0 - vg_cdf(-log_moneyness, sigma_vg, theta_scaled, nu_scaled)

    return math.exp(-r * T) * prob


def vg_binary_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    theta: float = 0.0,
    nu: float = 0.5,
) -> float:
    """
    Binary put option price using Variance Gamma distribution.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        theta: Drift/skewness parameter
        nu: Kurtosis parameter

    Returns:
        Binary put price
    """
    return math.exp(-r * T) - vg_binary_call(S, K, T, r, sigma, theta, nu)


# =============================================================================
# Normal Inverse Gaussian (NIG) Distribution
# =============================================================================


def _nig_cdf_numerical(x: float, alpha: float, beta: float, delta: float) -> float:
    """
    Normal Inverse Gaussian CDF via numerical integration.

    NIG density:
    f(x) = (alpha * delta / pi) * K_1(alpha * sqrt(delta^2 + x^2)) / sqrt(delta^2 + x^2)
           * exp(delta * sqrt(alpha^2 - beta^2) + beta * x)

    where K_1 is the modified Bessel function of the second kind.
    """
    # For very small delta, use normal approximation
    # NIG converges to normal as delta -> 0
    gamma = math.sqrt(max(alpha**2 - beta**2, 1e-10))

    if delta < 0.1:
        # Use normal approximation for small delta
        # NIG variance = delta / gamma, mean = delta * beta / gamma
        var = delta / gamma if gamma > 0 else 1.0
        mean = delta * beta / gamma if gamma > 0 else 0.0
        std = math.sqrt(var) if var > 0 else 1.0
        return norm_cdf((x - mean) / std)

    try:
        from scipy.stats import norminvgauss
    except ImportError:
        # Fallback to normal approximation
        var = delta / gamma if gamma > 0 else 1.0
        mean = delta * beta / gamma if gamma > 0 else 0.0
        return norm_cdf((x - mean) / math.sqrt(var))

    # scipy's norminvgauss uses different parameterization
    # scipy parametrization: a = delta*sqrt(alpha^2-beta^2), b = beta/alpha
    try:
        a = delta * gamma
        b = beta / alpha if alpha > 0 else 0.0

        if a <= 0.01:  # Very small a leads to numerical issues
            var = delta / gamma if gamma > 0 else 1.0
            mean = delta * beta / gamma if gamma > 0 else 0.0
            return norm_cdf((x - mean) / math.sqrt(var))

        result = float(norminvgauss.cdf(x, a, b, loc=0, scale=delta))

        # Check for NaN and fallback
        if math.isnan(result):
            var = delta / gamma if gamma > 0 else 1.0
            mean = delta * beta / gamma if gamma > 0 else 0.0
            return norm_cdf((x - mean) / math.sqrt(var))

        return result
    except Exception:
        # Final fallback to normal
        var = delta / gamma if gamma > 0 else 1.0
        mean = delta * beta / gamma if gamma > 0 else 0.0
        return norm_cdf((x - mean) / math.sqrt(var))


def nig_cdf(
    x: float, alpha: float = 1.0, beta: float = 0.0, delta: float = 1.0
) -> float:
    """
    Normal Inverse Gaussian distribution CDF.

    NIG is a flexible distribution that can capture both skewness and kurtosis.

    Parameters:
        alpha > 0: Tail heaviness (smaller = heavier tails)
        |beta| < alpha: Asymmetry (-beta for left skew, +beta for right skew)
        delta > 0: Scale parameter

    Constraints: alpha > |beta|

    Args:
        x: Input value
        alpha: Tail heaviness parameter
        beta: Asymmetry parameter
        delta: Scale parameter

    Returns:
        Cumulative probability P(X <= x)
    """
    # Enforce constraints
    if alpha <= 0:
        alpha = 1.0
    if abs(beta) >= alpha:
        beta = 0.0  # Symmetric case
    if delta <= 0:
        delta = 1.0

    return _nig_cdf_numerical(x, alpha, beta, delta)


def nig_params_from_vol(
    sigma: float, T: float, alpha: float = 1.0, beta: float = 0.0
) -> Tuple[float, float, float]:
    """
    Scale NIG parameters to match given volatility.

    NIG variance = delta / gamma where gamma = sqrt(alpha^2 - beta^2)
    Target variance = sigma^2 * T

    Solving: delta = sigma^2 * T * gamma

    Args:
        sigma: Annualized volatility
        T: Time to expiry in years
        alpha: Tail heaviness (fixed)
        beta: Asymmetry (fixed)

    Returns:
        (alpha, beta, delta) tuple
    """
    # Ensure constraint
    if abs(beta) >= alpha:
        beta = 0.0

    gamma = math.sqrt(alpha**2 - beta**2)
    target_var = sigma**2 * T
    delta = target_var * gamma

    return alpha, beta, delta


def nig_binary_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> float:
    """
    Binary call option price using Normal Inverse Gaussian distribution.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        alpha: Tail heaviness (larger = lighter tails, closer to normal)
        beta: Asymmetry (positive = right skew)

    Returns:
        Binary call price (probability of finishing ITM, discounted)
    """
    if T <= 0:
        return 1.0 if S >= K else 0.0
    if sigma <= 0:
        future_S = S * math.exp(r * T)
        return math.exp(-r * T) if future_S >= K else 0.0
    if S <= 0 or K <= 0:
        return 0.0

    # Log-moneyness adjusted for drift
    log_moneyness = math.log(S / K) + (r - 0.5 * sigma**2) * T

    # Scale NIG parameters
    alpha_scaled, beta_scaled, delta_scaled = nig_params_from_vol(
        sigma, T, alpha, beta
    )

    # P(S_T > K) = P(X > -log_moneyness) = 1 - CDF(-log_moneyness)
    prob = 1.0 - nig_cdf(-log_moneyness, alpha_scaled, beta_scaled, delta_scaled)

    return math.exp(-r * T) * prob


def nig_binary_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> float:
    """
    Binary put option price using Normal Inverse Gaussian distribution.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        alpha: Tail heaviness
        beta: Asymmetry

    Returns:
        Binary put price
    """
    return math.exp(-r * T) - nig_binary_call(S, K, T, r, sigma, alpha, beta)


# =============================================================================
# Utility Functions
# =============================================================================


def d2_from_prices(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Calculate the d2 parameter used in Black-Scholes.

    d2 = (ln(S/K) + (r - 0.5*sigma^2)*T) / (sigma*sqrt(T))

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate
        sigma: Volatility

    Returns:
        d2 value (standardized log-moneyness)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    sqrt_T = math.sqrt(T)
    return (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * sqrt_T)
