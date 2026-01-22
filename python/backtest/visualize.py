"""
Backtest Visualization

Visualizes backtest results with charts and statistics tables.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from backtest.engine_fast import AssetResult, BacktestReport


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""

    output_dir: str = "data/reports"
    figsize: tuple = (14, 10)
    dpi: int = 100
    style: str = "seaborn-v0_8-whitegrid"


def results_to_dataframe(report: BacktestReport) -> pd.DataFrame:
    """Convert asset results to a DataFrame for analysis."""
    rows = []
    for asset, result in report.asset_results.items():
        rows.append({
            "Asset": asset.upper(),
            "Trades": result.n_trades,
            "Fills": result.n_fills,
            "Fill Rate": result.n_fills / result.n_trades if result.n_trades > 0 else 0,
            "Volume": result.total_volume,
            "PnL": result.total_pnl,
            "Realized PnL": result.realized_pnl,
            "Max Position": result.max_position,
            "Max Drawdown": result.max_drawdown,
            "Sharpe": result.sharpe_ratio,
            "Avg Fill Price": result.avg_fill_price,
        })

    df = pd.DataFrame(rows)
    return df.sort_values("PnL", ascending=False).reset_index(drop=True)


def print_summary_table(report: BacktestReport) -> str:
    """Print a summary table of backtest results."""
    df = results_to_dataframe(report)

    lines = []
    lines.append("=" * 100)
    lines.append(f"{'BACKTEST RESULTS SUMMARY':^100}")
    lines.append("=" * 100)

    # Config summary
    cfg = report.config
    lines.append(f"\nConfig: spread={cfg.base_spread:.1%}, size={cfg.base_size}, "
                 f"refresh={cfg.quote_refresh_sec}s, max_pos={cfg.max_position}")
    lines.append(f"Data: {cfg.start_date or 'all'} ~ {cfg.end_date or 'all'}")
    lines.append("")

    # Per-asset table
    lines.append("-" * 100)
    lines.append(f"{'Asset':<8} {'Trades':>10} {'Fills':>10} {'Fill%':>8} {'Volume':>12} "
                 f"{'PnL':>12} {'MaxPos':>10} {'Sharpe':>8}")
    lines.append("-" * 100)

    for _, row in df.iterrows():
        lines.append(
            f"{row['Asset']:<8} {row['Trades']:>10,} {row['Fills']:>10,} "
            f"{row['Fill Rate']:>7.1%} {row['Volume']:>12,.0f} "
            f"{row['PnL']:>12,.2f} {row['Max Position']:>10,.1f} "
            f"{row['Sharpe']:>8.2f}"
        )

    lines.append("-" * 100)

    # Totals
    lines.append(
        f"{'TOTAL':<8} {report.total_trades:>10,} {report.total_fills:>10,} "
        f"{report.total_fills/report.total_trades if report.total_trades > 0 else 0:>7.1%} "
        f"{report.total_volume:>12,.0f} {report.total_pnl:>12,.2f} "
        f"{df['Max Position'].max():>10,.1f} {report.avg_sharpe:>8.2f}"
    )
    lines.append("=" * 100)

    # Statistics
    lines.append(f"\n{'STATISTICS':^100}")
    lines.append("-" * 100)

    pnls = df["PnL"].values
    lines.append(f"  Mean PnL:     ${np.mean(pnls):>12,.2f}")
    lines.append(f"  Std PnL:      ${np.std(pnls):>12,.2f}")
    lines.append(f"  Min PnL:      ${np.min(pnls):>12,.2f}  ({df.loc[df['PnL'].idxmin(), 'Asset']})")
    lines.append(f"  Max PnL:      ${np.max(pnls):>12,.2f}  ({df.loc[df['PnL'].idxmax(), 'Asset']})")
    lines.append(f"  Total PnL:    ${report.total_pnl:>12,.2f}")
    lines.append("")
    lines.append(f"  Win Rate:     {(pnls > 0).sum()}/{len(pnls)} assets profitable")
    lines.append(f"  Avg Sharpe:   {report.avg_sharpe:>8.2f}")
    lines.append(f"  Total Volume: {report.total_volume:>12,.0f}")
    lines.append("=" * 100)

    output = "\n".join(lines)
    print(output)
    return output


def plot_results(
    report: BacktestReport,
    config: Optional[VisualizationConfig] = None,
    save: bool = True,
) -> Optional[str]:
    """
    Generate visualization charts for backtest results.

    Returns path to saved figure if save=True.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return None

    config = config or VisualizationConfig()
    df = results_to_dataframe(report)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=config.figsize)
    fig.suptitle(
        f"Backtest Results | Spread: {report.config.base_spread:.1%} | "
        f"Size: {report.config.base_size}",
        fontsize=14, fontweight="bold"
    )

    # 1. PnL by Asset (bar chart)
    ax1 = axes[0, 0]
    colors = ["green" if x > 0 else "red" for x in df["PnL"]]
    bars = ax1.bar(df["Asset"], df["PnL"], color=colors, alpha=0.7, edgecolor="black")
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.set_xlabel("Asset")
    ax1.set_ylabel("PnL ($)")
    ax1.set_title("PnL by Asset")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    # Add value labels
    for bar, val in zip(bars, df["PnL"]):
        height = bar.get_height()
        ax1.annotate(
            f"${val:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -12),
            textcoords="offset points",
            ha="center", va="bottom" if height >= 0 else "top",
            fontsize=9
        )

    # 2. Volume by Asset (bar chart)
    ax2 = axes[0, 1]
    ax2.bar(df["Asset"], df["Volume"], color="steelblue", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Asset")
    ax2.set_ylabel("Volume (tokens)")
    ax2.set_title("Trading Volume by Asset")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    # 3. Fill Rate vs Sharpe (scatter)
    ax3 = axes[1, 0]
    scatter = ax3.scatter(
        df["Fill Rate"] * 100, df["Sharpe"],
        s=df["Volume"] / df["Volume"].max() * 500 + 50,
        c=df["PnL"], cmap="RdYlGn", alpha=0.7, edgecolors="black"
    )
    for i, row in df.iterrows():
        ax3.annotate(row["Asset"], (row["Fill Rate"] * 100, row["Sharpe"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax3.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax3.set_xlabel("Fill Rate (%)")
    ax3.set_ylabel("Sharpe Ratio")
    ax3.set_title("Fill Rate vs Sharpe (size = volume, color = PnL)")
    plt.colorbar(scatter, ax=ax3, label="PnL ($)")

    # 4. Summary Statistics Table
    ax4 = axes[1, 1]
    ax4.axis("off")

    stats_data = [
        ["Total Trades", f"{report.total_trades:,}"],
        ["Total Fills", f"{report.total_fills:,}"],
        ["Total Volume", f"{report.total_volume:,.0f}"],
        ["Total PnL", f"${report.total_pnl:,.2f}"],
        ["Mean PnL/Asset", f"${np.mean(df['PnL']):,.2f}"],
        ["Std PnL", f"${np.std(df['PnL']):,.2f}"],
        ["Best Asset", f"{df.loc[df['PnL'].idxmax(), 'Asset']} (${df['PnL'].max():,.2f})"],
        ["Worst Asset", f"{df.loc[df['PnL'].idxmin(), 'Asset']} (${df['PnL'].min():,.2f})"],
        ["Avg Sharpe", f"{report.avg_sharpe:.2f}"],
        ["Win Rate", f"{(df['PnL'] > 0).sum()}/{len(df)} assets"],
    ]

    table = ax4.table(
        cellText=stats_data,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.5, 0.5]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    ax4.set_title("Summary Statistics", pad=20)

    plt.tight_layout()

    # Save figure
    if save:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"backtest_spread{report.config.base_spread:.0%}_size{report.config.base_size:.0f}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=config.dpi, bbox_inches="tight")
        print(f"Chart saved to: {filepath}")
        plt.close()
        return str(filepath)
    else:
        plt.show()
        return None


def plot_pnl_curves(
    report: BacktestReport,
    config: Optional[VisualizationConfig] = None,
    save: bool = True,
) -> Optional[str]:
    """Plot PnL curves over time for each asset."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed.")
        return None

    config = config or VisualizationConfig()

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("PnL Curves by Asset", fontsize=14, fontweight="bold")

    for asset, result in report.asset_results.items():
        if len(result.pnl_history) > 1:
            ax.plot(result.pnl_history, label=f"{asset.upper()} (${result.total_pnl:,.0f})", alpha=0.8)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Time (sampled)")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.legend(loc="best")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    plt.tight_layout()

    if save:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "pnl_curves.png"
        plt.savefig(filepath, dpi=config.dpi, bbox_inches="tight")
        print(f"PnL curves saved to: {filepath}")
        plt.close()
        return str(filepath)
    else:
        plt.show()
        return None


def generate_report(report: BacktestReport, output_dir: str = "data/reports") -> str:
    """Generate a complete report with table and charts."""
    config = VisualizationConfig(output_dir=output_dir)

    # Print summary table
    summary = print_summary_table(report)

    # Generate charts
    chart_path = plot_results(report, config, save=True)
    pnl_path = plot_pnl_curves(report, config, save=True)

    # Save text report
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / "backtest_report.txt"

    with open(report_file, "w") as f:
        f.write(summary)
        f.write(f"\n\nCharts: {chart_path}, {pnl_path}\n")

    print(f"\nReport saved to: {report_file}")
    return str(report_file)
