"""
Cross-Entropy Evaluation of Distribution Models

Evaluates binary option pricing models using log-loss (cross-entropy)
against synthetic 15-minute updown markets from Binance data.
"""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress integration warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*IntegrationWarning.*")

from analysis.distribution_models import (
    BinaryOptionModel,
    DistributionParams,
    LaplaceModel,
    LogisticModel,
    MixtureNormalModel,
    NIGModel,
    NormalModel,
    StudentTModel,
    VarianceGammaModel,
    get_all_models,
)
from strategy.utils.distributions import estimate_kurtosis


@dataclass
class MarketData:
    """Single market evaluation data."""

    asset: str
    start_ts: int
    end_ts: int
    start_price: float
    end_price: float
    outcome: int  # 1 for up, 0 for down
    returns: np.ndarray  # Historical returns for vol estimation


@dataclass
class EvalResult:
    """Evaluation result for a model-parameter combination."""

    model_name: str
    params: Dict[str, float]
    cross_entropy: float
    accuracy: float
    n_markets: int


def load_binance_klines(asset: str, data_dir: str = "data/datalake/global/binance_klines") -> pd.DataFrame:
    """Load all Binance klines for an asset."""
    symbol = f"{asset}usdt"
    files = [f for f in os.listdir(data_dir) if f.startswith(symbol)]

    if not files:
        raise ValueError(f"No klines found for {asset}")

    dfs = []
    for f in sorted(files):
        df = pd.read_parquet(os.path.join(data_dir, f))
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("open_time").drop_duplicates("open_time")
    return df


def generate_synthetic_markets(
    klines: pd.DataFrame,
    market_duration_minutes: int = 15,
    vol_window_minutes: int = 60,
) -> List[MarketData]:
    """Generate synthetic updown markets from klines data."""
    markets = []

    # Resample to 1-minute for easier handling
    klines = klines.copy()
    klines["ts_minute"] = (klines["open_time"] // 60000) * 60000
    minute_df = klines.groupby("ts_minute").agg({
        "open": "first",
        "close": "last",
        "high": "max",
        "low": "min",
        "volume": "sum",
    }).reset_index()

    # Calculate log returns
    minute_df["log_return"] = np.log(minute_df["close"] / minute_df["close"].shift(1))
    minute_df = minute_df.dropna()

    # Extract asset name from first row
    asset = "unknown"

    # Generate markets at regular intervals
    step = market_duration_minutes

    for i in range(vol_window_minutes, len(minute_df) - market_duration_minutes, step):
        start_idx = i
        end_idx = i + market_duration_minutes

        start_ts = int(minute_df.iloc[start_idx]["ts_minute"])
        end_ts = int(minute_df.iloc[end_idx]["ts_minute"])
        start_price = minute_df.iloc[start_idx]["open"]
        end_price = minute_df.iloc[end_idx]["close"]

        # Outcome: 1 if price went up
        outcome = 1 if end_price >= start_price else 0

        # Historical returns for vol estimation (before market start)
        hist_start = max(0, start_idx - vol_window_minutes)
        returns = minute_df.iloc[hist_start:start_idx]["log_return"].values

        if len(returns) >= 10:
            markets.append(MarketData(
                asset=asset,
                start_ts=start_ts,
                end_ts=end_ts,
                start_price=start_price,
                end_price=end_price,
                outcome=outcome,
                returns=returns,
            ))

    return markets


def compute_realized_vol(returns: np.ndarray, annualize: bool = True) -> float:
    """Compute realized volatility from returns."""
    if len(returns) < 2:
        return 0.5  # Default vol

    std = np.std(returns)
    if annualize:
        # Assuming 1-minute returns, annualize
        std *= np.sqrt(365 * 24 * 60)
    return float(std)


def cross_entropy_loss(pred_prob: float, actual: int, eps: float = 1e-10) -> float:
    """Binary cross-entropy loss."""
    pred_prob = np.clip(pred_prob, eps, 1 - eps)
    if actual == 1:
        return -math.log(pred_prob)
    else:
        return -math.log(1 - pred_prob)


def evaluate_model(
    model: BinaryOptionModel,
    markets: List[MarketData],
    params: Optional[DistributionParams] = None,
    time_to_expiry_years: float = 15 / (365 * 24 * 60),  # 15 minutes
    strike_offset_pct: float = 0.0,  # 0 = ATM, positive = OTM call, negative = ITM call
) -> Tuple[float, float]:
    """Evaluate a model on markets, return (cross_entropy, accuracy)."""
    total_loss = 0.0
    correct = 0

    for market in markets:
        vol = compute_realized_vol(market.returns)

        # Strike with offset
        strike = market.start_price * (1 + strike_offset_pct / 100)

        # Price binary call (probability of ending above strike)
        prob_up = model.binary_call_price(
            spot=market.start_price,
            strike=strike,
            time_to_expiry=time_to_expiry_years,
            vol=vol,
            params=params,
        )

        # Actual outcome for this strike
        actual = 1 if market.end_price >= strike else 0

        # Cross-entropy loss
        total_loss += cross_entropy_loss(prob_up, actual)

        # Accuracy
        pred = 1 if prob_up >= 0.5 else 0
        if pred == actual:
            correct += 1

    avg_loss = total_loss / len(markets) if markets else float("inf")
    accuracy = correct / len(markets) if markets else 0.0

    return avg_loss, accuracy


def search_params_logistic(
    markets: List[MarketData],
    kurtosis_range: Tuple[float, float] = (0.0, 6.0),
    n_points: int = 25,
) -> Tuple[Dict[str, float], float]:
    """Search for optimal kurtosis parameter for Logistic model."""
    model = LogisticModel()
    best_params = {"kurtosis": 1.2}
    best_loss = float("inf")

    for k in np.linspace(kurtosis_range[0], kurtosis_range[1], n_points):
        params = DistributionParams(name="logistic", params={"kurtosis": k})
        loss, _ = evaluate_model(model, markets, params)

        if loss < best_loss:
            best_loss = loss
            best_params = {"kurtosis": float(k)}

    return best_params, best_loss


def search_params_laplace(
    markets: List[MarketData],
    kurtosis_range: Tuple[float, float] = (0.0, 6.0),
    n_points: int = 25,
) -> Tuple[Dict[str, float], float]:
    """Search for optimal kurtosis parameter for Laplace/GND model."""
    model = LaplaceModel()
    best_params = {"kurtosis": 3.0}
    best_loss = float("inf")

    for k in np.linspace(kurtosis_range[0], kurtosis_range[1], n_points):
        params = DistributionParams(name="laplace", params={"kurtosis": k})
        loss, _ = evaluate_model(model, markets, params)

        if loss < best_loss:
            best_loss = loss
            best_params = {"kurtosis": float(k)}

    return best_params, best_loss


def search_params_student_t(
    markets: List[MarketData],
    df_range: Tuple[float, float] = (2.5, 50.0),
    n_points: int = 25,
) -> Tuple[Dict[str, float], float]:
    """Search for optimal degrees of freedom for Student's t model."""
    model = StudentTModel()
    best_params = {"df": 5.0}
    best_loss = float("inf")

    for df in np.linspace(df_range[0], df_range[1], n_points):
        params = DistributionParams(name="student_t", params={"df": df})
        loss, _ = evaluate_model(model, markets, params)

        if loss < best_loss:
            best_loss = loss
            best_params = {"df": float(df)}

    return best_params, best_loss


def search_params_mixture(
    markets: List[MarketData],
    w_range: Tuple[float, float] = (0.5, 0.95),
    sigma_ratio_range: Tuple[float, float] = (1.5, 4.0),
    n_points: int = 10,
) -> Tuple[Dict[str, float], float]:
    """Search for optimal mixture parameters."""
    model = MixtureNormalModel()
    best_params = {"w": 0.8, "sigma_ratio": 2.0}
    best_loss = float("inf")

    for w in np.linspace(w_range[0], w_range[1], n_points):
        for sr in np.linspace(sigma_ratio_range[0], sigma_ratio_range[1], n_points):
            params = DistributionParams(name="mixture", params={"w": w, "sigma_ratio": sr})
            loss, _ = evaluate_model(model, markets, params)

            if loss < best_loss:
                best_loss = loss
                best_params = {"w": float(w), "sigma_ratio": float(sr)}

    return best_params, best_loss


def search_params_vg(
    markets: List[MarketData],
    theta_range: Tuple[float, float] = (-0.3, 0.3),
    nu_range: Tuple[float, float] = (0.1, 1.0),
    n_points: int = 5,
) -> Tuple[Dict[str, float], float]:
    """Search for optimal VG parameters (reduced search space for speed)."""
    model = VarianceGammaModel()
    best_params = {"theta": 0.0, "nu": 0.5}
    best_loss = float("inf")

    # Sample subset for faster search
    sample_markets = markets[::3] if len(markets) > 100 else markets

    for theta in np.linspace(theta_range[0], theta_range[1], n_points):
        for nu in np.linspace(nu_range[0], nu_range[1], n_points):
            params = DistributionParams(name="variance_gamma", params={"theta": theta, "nu": nu})
            loss, _ = evaluate_model(model, sample_markets, params)

            if loss < best_loss:
                best_loss = loss
                best_params = {"theta": float(theta), "nu": float(nu)}

    # Re-evaluate on full data with best params
    loss, _ = evaluate_model(model, markets, DistributionParams("variance_gamma", best_params))
    return best_params, loss


def search_params_nig(
    markets: List[MarketData],
    alpha_range: Tuple[float, float] = (0.5, 1.5),
    beta_range: Tuple[float, float] = (-0.3, 0.3),
    n_points: int = 5,
) -> Tuple[Dict[str, float], float]:
    """Search for optimal NIG parameters (reduced search space for speed)."""
    model = NIGModel()
    best_params = {"alpha": 1.0, "beta": 0.0}
    best_loss = float("inf")

    # Sample subset for faster search
    sample_markets = markets[::3] if len(markets) > 100 else markets

    for alpha in np.linspace(alpha_range[0], alpha_range[1], n_points):
        for beta in np.linspace(beta_range[0], beta_range[1], n_points):
            if abs(beta) >= alpha:
                continue
            params = DistributionParams(name="nig", params={"alpha": alpha, "beta": beta})
            loss, _ = evaluate_model(model, sample_markets, params)

            if loss < best_loss:
                best_loss = loss
                best_params = {"alpha": float(alpha), "beta": float(beta)}

    # Re-evaluate on full data with best params
    loss, _ = evaluate_model(model, markets, DistributionParams("nig", best_params))
    return best_params, loss


def evaluate_all_models(
    markets: List[MarketData],
    search_params: bool = True,
) -> Dict[str, EvalResult]:
    """Evaluate all models, optionally searching for optimal parameters."""
    results = {}

    # Normal model (no params)
    model = NormalModel()
    loss, acc = evaluate_model(model, markets)
    results["normal"] = EvalResult(
        model_name="normal",
        params={},
        cross_entropy=loss,
        accuracy=acc,
        n_markets=len(markets),
    )
    print(f"  Normal: CE={loss:.4f}, Acc={acc:.2%}")

    # Student's t
    if search_params:
        params, loss = search_params_student_t(markets)
    else:
        params = {"df": 5.0}
        model = StudentTModel()
        loss, _ = evaluate_model(model, markets, DistributionParams("student_t", params))

    model = StudentTModel()
    _, acc = evaluate_model(model, markets, DistributionParams("student_t", params))
    results["student_t"] = EvalResult(
        model_name="student_t",
        params=params,
        cross_entropy=loss,
        accuracy=acc,
        n_markets=len(markets),
    )
    print(f"  Student's t: CE={loss:.4f}, Acc={acc:.2%}, params={params}")

    # Logistic
    if search_params:
        params, loss = search_params_logistic(markets)
    else:
        params = {"kurtosis": 1.2}
        model = LogisticModel()
        loss, _ = evaluate_model(model, markets, DistributionParams("logistic", params))

    model = LogisticModel()
    _, acc = evaluate_model(model, markets, DistributionParams("logistic", params))
    results["logistic"] = EvalResult(
        model_name="logistic",
        params=params,
        cross_entropy=loss,
        accuracy=acc,
        n_markets=len(markets),
    )
    print(f"  Logistic: CE={loss:.4f}, Acc={acc:.2%}, params={params}")

    # Laplace
    if search_params:
        params, loss = search_params_laplace(markets)
    else:
        params = {"kurtosis": 3.0}
        model = LaplaceModel()
        loss, _ = evaluate_model(model, markets, DistributionParams("laplace", params))

    model = LaplaceModel()
    _, acc = evaluate_model(model, markets, DistributionParams("laplace", params))
    results["laplace"] = EvalResult(
        model_name="laplace",
        params=params,
        cross_entropy=loss,
        accuracy=acc,
        n_markets=len(markets),
    )
    print(f"  Laplace: CE={loss:.4f}, Acc={acc:.2%}, params={params}")

    # Mixture
    if search_params:
        params, loss = search_params_mixture(markets)
    else:
        params = {"w": 0.8, "sigma_ratio": 2.0}
        model = MixtureNormalModel()
        loss, _ = evaluate_model(model, markets, DistributionParams("mixture", params))

    model = MixtureNormalModel()
    _, acc = evaluate_model(model, markets, DistributionParams("mixture", params))
    results["mixture"] = EvalResult(
        model_name="mixture",
        params=params,
        cross_entropy=loss,
        accuracy=acc,
        n_markets=len(markets),
    )
    print(f"  Mixture: CE={loss:.4f}, Acc={acc:.2%}, params={params}")

    # Variance Gamma
    if search_params:
        params, loss = search_params_vg(markets)
    else:
        params = {"theta": 0.0, "nu": 0.5}
        model = VarianceGammaModel()
        loss, _ = evaluate_model(model, markets, DistributionParams("variance_gamma", params))

    model = VarianceGammaModel()
    _, acc = evaluate_model(model, markets, DistributionParams("variance_gamma", params))
    results["variance_gamma"] = EvalResult(
        model_name="variance_gamma",
        params=params,
        cross_entropy=loss,
        accuracy=acc,
        n_markets=len(markets),
    )
    print(f"  Variance Gamma: CE={loss:.4f}, Acc={acc:.2%}, params={params}")

    # NIG
    if search_params:
        params, loss = search_params_nig(markets)
    else:
        params = {"alpha": 1.0, "beta": 0.0}
        model = NIGModel()
        loss, _ = evaluate_model(model, markets, DistributionParams("nig", params))

    model = NIGModel()
    _, acc = evaluate_model(model, markets, DistributionParams("nig", params))
    results["nig"] = EvalResult(
        model_name="nig",
        params=params,
        cross_entropy=loss,
        accuracy=acc,
        n_markets=len(markets),
    )
    print(f"  NIG: CE={loss:.4f}, Acc={acc:.2%}, params={params}")

    return results


def compute_market_mid_stats(
    markets: List[MarketData],
) -> Tuple[float, float]:
    """Compute 'market mid' baseline (0.5 for all ATM markets)."""
    # For ATM binary options, naive baseline is 0.5
    total_loss = 0.0
    correct = 0

    for market in markets:
        prob = 0.5  # Market mid for ATM
        total_loss += cross_entropy_loss(prob, market.outcome)
        # Random guess

    avg_loss = total_loss / len(markets) if markets else float("inf")
    accuracy = sum(1 for m in markets if m.outcome == 1) / len(markets) if markets else 0.5
    # Note: accuracy of 0.5 baseline depends on up/down ratio

    return avg_loss, accuracy


def plot_results_by_asset(
    all_results: Dict[str, Dict[str, EvalResult]],
    market_mid_stats: Dict[str, Tuple[float, float]],
    output_path: str = "data/analysis_output/distribution_eval.png",
):
    """Plot cross-entropy results by asset with optimal params."""
    assets = list(all_results.keys())
    models = ["normal", "student_t", "logistic", "laplace", "mixture", "variance_gamma", "nig"]
    model_labels = ["Normal", "Student-t", "Logistic", "Laplace", "Mixture", "VG", "NIG"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for idx, asset in enumerate(assets[:4]):
        ax = axes[idx]
        results = all_results[asset]
        market_mid_ce, market_mid_acc = market_mid_stats[asset]

        # Cross-entropy values
        ce_values = [results[m].cross_entropy for m in models]

        # Bar plot
        bars = ax.bar(model_labels, ce_values, color=colors, alpha=0.8, edgecolor='black')

        # Add market mid baseline
        ax.axhline(y=market_mid_ce, color='red', linestyle='--', linewidth=2, label=f'Baseline (0.5): {market_mid_ce:.4f}')

        # Add optimal params as text on bars
        for i, (m, bar) in enumerate(zip(models, bars)):
            result = results[m]
            params_str = ", ".join(f"{k}={v:.2f}" for k, v in result.params.items())
            if params_str:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    params_str,
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    rotation=45,
                )

        ax.set_title(f"{asset.upper()} (n={results['normal'].n_markets})", fontsize=12, fontweight='bold')
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_ylim(bottom=min(ce_values) * 0.95, top=max(ce_values + [market_mid_ce]) * 1.15)
        ax.legend(loc='upper right', fontsize=8)
        ax.tick_params(axis='x', rotation=30)

        # Add accuracy annotation
        best_model = min(results.keys(), key=lambda m: results[m].cross_entropy)
        best_acc = results[best_model].accuracy
        ax.text(
            0.02, 0.98,
            f"Best: {best_model} (Acc: {best_acc:.1%})",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )

    plt.suptitle("Distribution Model Evaluation: Cross-Entropy by Asset\n(Optimal Parameters Shown)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_path}")


def evaluate_otm_scenarios(
    markets: List[MarketData],
    strike_offsets: List[float] = [-0.2, -0.1, 0.0, 0.1, 0.2],
) -> Dict[str, Dict[float, Tuple[float, float]]]:
    """Evaluate models across different strike offsets (OTM/ITM scenarios)."""
    models = {
        "normal": NormalModel(),
        "student_t": StudentTModel(),
        "logistic": LogisticModel(),
        "laplace": LaplaceModel(),
        "mixture": MixtureNormalModel(),
    }

    results = {name: {} for name in models}

    for offset in strike_offsets:
        print(f"    Strike offset: {offset:+.1f}%")
        for name, model in models.items():
            loss, acc = evaluate_model(model, markets, strike_offset_pct=offset)
            results[name][offset] = (loss, acc)

    return results


def plot_otm_comparison(
    all_otm_results: Dict[str, Dict[str, Dict[float, Tuple[float, float]]]],
    output_path: str = "data/analysis_output/distribution_otm_eval.png",
):
    """Plot cross-entropy across strike offsets for each asset."""
    assets = list(all_otm_results.keys())
    n_assets = len(assets)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = {
        "normal": "blue",
        "student_t": "orange",
        "logistic": "green",
        "laplace": "red",
        "mixture": "purple",
    }

    for idx, asset in enumerate(assets[:4]):
        ax = axes[idx]
        results = all_otm_results[asset]

        offsets = sorted(list(results["normal"].keys()))

        for model_name, model_results in results.items():
            ce_values = [model_results[o][0] for o in offsets]
            ax.plot(offsets, ce_values, 'o-', label=model_name, color=colors.get(model_name, "gray"), linewidth=2, markersize=6)

        ax.set_xlabel("Strike Offset (%)")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title(f"{asset.upper()}", fontsize=12, fontweight='bold')
        ax.legend(loc='upper center', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    plt.suptitle("Distribution Model Comparison: Cross-Entropy vs Strike Offset\n(Negative = ITM, Positive = OTM)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nOTM comparison plot saved to {output_path}")


def plot_param_sensitivity(
    markets: List[MarketData],
    asset: str,
    output_path: str = "data/analysis_output/param_sensitivity.png",
):
    """Plot parameter sensitivity for Logistic and Laplace kurtosis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Logistic kurtosis sweep
    kurtosis_vals = np.linspace(0, 6, 30)
    logistic_ce = []
    model = LogisticModel()

    for k in kurtosis_vals:
        params = DistributionParams(name="logistic", params={"kurtosis": k})
        loss, _ = evaluate_model(model, markets, params)
        logistic_ce.append(loss)

    axes[0].plot(kurtosis_vals, logistic_ce, 'b-', linewidth=2)
    best_k = kurtosis_vals[np.argmin(logistic_ce)]
    axes[0].axvline(x=best_k, color='r', linestyle='--', label=f'Optimal: {best_k:.2f}')
    axes[0].axvline(x=1.2, color='g', linestyle=':', label='Default: 1.2')
    axes[0].set_xlabel("Kurtosis")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].set_title(f"Logistic Model ({asset.upper()})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Laplace kurtosis sweep
    laplace_ce = []
    model = LaplaceModel()

    for k in kurtosis_vals:
        params = DistributionParams(name="laplace", params={"kurtosis": k})
        loss, _ = evaluate_model(model, markets, params)
        laplace_ce.append(loss)

    axes[1].plot(kurtosis_vals, laplace_ce, 'b-', linewidth=2)
    best_k = kurtosis_vals[np.argmin(laplace_ce)]
    axes[1].axvline(x=best_k, color='r', linestyle='--', label=f'Optimal: {best_k:.2f}')
    axes[1].axvline(x=3.0, color='g', linestyle=':', label='Default: 3.0')
    axes[1].set_xlabel("Kurtosis")
    axes[1].set_ylabel("Cross-Entropy")
    axes[1].set_title(f"Laplace/GND Model ({asset.upper()})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Parameter Sensitivity Analysis", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sensitivity plot saved to {output_path}")


def main():
    """Run full evaluation."""
    print("=" * 70)
    print("Distribution Model Cross-Entropy Evaluation")
    print("=" * 70)

    assets = ["btc", "eth", "sol", "xrp"]
    all_results = {}
    market_mid_stats = {}
    max_markets = 500  # Limit for faster evaluation

    for asset in assets:
        print(f"\n{'='*50}")
        print(f"Loading {asset.upper()} data...")
        print("=" * 50)

        try:
            klines = load_binance_klines(asset)
            print(f"  Loaded {len(klines)} klines")

            markets = generate_synthetic_markets(klines)
            # Set asset name
            for m in markets:
                m.asset = asset

            # Limit markets for speed
            if len(markets) > max_markets:
                markets = markets[:max_markets]

            print(f"  Using {len(markets)} markets for evaluation")

            if len(markets) < 50:
                print(f"  Warning: Low market count for {asset}")
                continue

            # Compute market mid baseline
            mid_ce, mid_acc = compute_market_mid_stats(markets)
            market_mid_stats[asset] = (mid_ce, mid_acc)
            print(f"  Market mid baseline: CE={mid_ce:.4f}")

            # Evaluate all models
            print(f"\n  Evaluating models (with param search)...")
            results = evaluate_all_models(markets, search_params=True)
            all_results[asset] = results

        except Exception as e:
            print(f"  Error processing {asset}: {e}")
            continue

    if all_results:
        # Plot results
        print("\n" + "=" * 50)
        print("Generating plots...")
        print("=" * 50)

        plot_results_by_asset(all_results, market_mid_stats)

        # Plot param sensitivity for first asset
        first_asset = list(all_results.keys())[0]
        klines = load_binance_klines(first_asset)
        markets = generate_synthetic_markets(klines)
        for m in markets:
            m.asset = first_asset
        if len(markets) > max_markets:
            markets = markets[:max_markets]
        plot_param_sensitivity(markets, first_asset)

        # OTM evaluation
        print("\n" + "=" * 50)
        print("Evaluating OTM scenarios...")
        print("=" * 50)

        all_otm_results = {}
        for asset in all_results.keys():
            print(f"\n  {asset.upper()}:")
            klines = load_binance_klines(asset)
            markets = generate_synthetic_markets(klines)
            for m in markets:
                m.asset = asset
            if len(markets) > max_markets:
                markets = markets[:max_markets]

            otm_results = evaluate_otm_scenarios(markets, strike_offsets=[-0.3, -0.15, 0.0, 0.15, 0.3])
            all_otm_results[asset] = otm_results

        plot_otm_comparison(all_otm_results)

        # Summary table
        print("\n" + "=" * 70)
        print("SUMMARY: Best Model by Asset (ATM)")
        print("=" * 70)
        print(f"{'Asset':<8} {'Best Model':<15} {'CE Loss':<10} {'Accuracy':<10} {'Key Params'}")
        print("-" * 70)

        for asset, results in all_results.items():
            best_model = min(results.keys(), key=lambda m: results[m].cross_entropy)
            r = results[best_model]
            params_str = ", ".join(f"{k}={v:.2f}" for k, v in r.params.items())
            print(f"{asset.upper():<8} {best_model:<15} {r.cross_entropy:<10.4f} {r.accuracy:<10.2%} {params_str}")


if __name__ == "__main__":
    main()
