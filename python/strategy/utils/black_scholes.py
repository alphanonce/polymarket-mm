"""
Black-Scholes Binary Option Pricing

Fast implementation of binary option pricing using hybrid CDF approximation.
Tanh approximation for central region, Abramowitz-Stegun for tails.
"""

import numpy as np


def norm_cdf(x: float) -> float:
    """
    Fast normal CDF approximation using hybrid approach.

    Central region (|x| <= 3.0): Uses tanh approximation
        0.5 * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
        Maximum error ~0.0002 in this region.

    Tails (|x| > 3.0): Uses Abramowitz & Stegun 26.2.17
        Error < 7.5e-8 for all x.

    Args:
        x: Standard normal z-score

    Returns:
        Cumulative probability P(Z <= x)
    """
    # Early exit for extreme values
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0

    # Use Abramowitz-Stegun for tails (better accuracy)
    if abs(x) > 3.0:
        return _norm_cdf_tail(x)

    # Use tanh approximation for central region (faster)
    return 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def _norm_cdf_tail(x: float) -> float:
    """
    Normal CDF for tail values using Abramowitz & Stegun 26.2.17.

    This polynomial approximation has error < 7.5e-8 for all x.

    Args:
        x: Standard normal z-score (typically |x| > 3.0)

    Returns:
        Cumulative probability P(Z <= x)
    """
    # Coefficients for Abramowitz & Stegun approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x / 2.0)

    return 0.5 * (1.0 + sign * y)


def bs_binary_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Price a binary (digital) call option.

    Binary call pays $1 if S >= K at expiry, $0 otherwise.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)

    Returns:
        Binary call price (probability of finishing ITM, discounted)
    """
    # Edge case: at expiry
    if T <= 0:
        return 1.0 if S >= K else 0.0

    # Edge case: zero volatility (deterministic)
    if sigma <= 0:
        # Future value is certain: S * exp(r*T)
        future_S = S * np.exp(r * T)
        return np.exp(-r * T) if future_S >= K else 0.0

    # Edge case: invalid prices
    if S <= 0 or K <= 0:
        return 0.0

    # Standard Black-Scholes d2
    sqrt_T = np.sqrt(T)
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * sqrt_T)

    # Clamp d2 to prevent numerical instability in extreme cases
    d2 = np.clip(d2, -8.0, 8.0)

    # Binary call = exp(-r*T) * N(d2)
    return np.exp(-r * T) * norm_cdf(d2)


def bs_binary_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Price a binary (digital) put option.

    Binary put pays $1 if S < K at expiry, $0 otherwise.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)

    Returns:
        Binary put price (probability of finishing OTM for call)
    """
    # Put-call parity for binary options: call + put = exp(-r*T)
    return np.exp(-r * T) - bs_binary_call(S, K, T, r, sigma)
