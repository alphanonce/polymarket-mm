#!/usr/bin/env python3
"""
Parameter Sweep Script for Backtests

Runs backtests with many parameter combinations in parallel,
tracks progress with ETA, and aggregates results.

Usage:
    # Sweep TpBS model
    uv run python backtest/run_param_sweep.py \
        --model tpbs \
        --symbols btc \
        --start-date 2026-01-18 \
        --end-date 2026-01-18 \
        --max-z 2.0,3.0,4.0,5.0 \
        --min-z 1.0,1.5,2.0 \
        --implied-vol 0.3,0.5,0.7 \
        --workers 4 \
        --output-dir data/sweep_results/tpbs_sweep

    # Resume interrupted sweep
    uv run python backtest/run_param_sweep.py \
        --resume data/sweep_results/tpbs_sweep/checkpoint.json

    # Custom linspace for parameters (start:stop:num)
    uv run python backtest/run_param_sweep.py \
        --model inventory \
        --base-spread 0.01:0.05:5 \
        --inventory-skew 0.2:1.0:5
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

# Default parameter grids by model type
DEFAULT_GRIDS: dict[str, dict[str, list[float]]] = {
    "spread": {
        "base-spread": [0.01, 0.02, 0.03, 0.04],
    },
    "inventory": {
        "base-spread": [0.01, 0.02, 0.03, 0.04],
        "inventory-skew": [0.3, 0.5, 0.7, 1.0],
    },
    "tpbs": {
        "max-z": [2.0, 3.0, 4.0, 5.0],
        "min-z": [1.0, 1.5, 2.0],
        "implied-vol": [0.3, 0.5, 0.7, 1.0],
    },
    "tpsl_bs": {
        "z": [0.5, 2.0, 4.0],  # Effect is minimal with small tau
        "tp-ticks": [2, 5, 10],  # Wider range for meaningful variation
        "sl-ticks": [0, 3, 5],  # Wider range
        "implied-vol": [0.3, 0.7, 1.0],  # Affects BS pricing directly
    },
}


@dataclass
class SweepConfig:
    """Configuration for parameter sweep."""

    model: str
    symbols: list[str]
    start_date: str
    end_date: str
    side: str
    workers: int
    output_dir: Path
    data_dir: str
    rtds_price_dir: str
    param_grid: dict[str, list[float]]
    max_position_pct: float = 0.20
    initial_equity: float = 10000.0


@dataclass
class JobResult:
    """Result from a single backtest job."""

    job_id: str
    params: dict[str, Any]
    success: bool
    pnl: float = 0.0
    pnl_after_rebate: float = 0.0
    sharpe_ratio: float = 0.0
    total_fills: int = 0
    total_volume: float = 0.0
    max_drawdown: float = 0.0
    error: str = ""
    duration_seconds: float = 0.0


def parse_param_value(value_str: str) -> list[float]:
    """
    Parse a parameter value string into a list of floats.

    Supports:
    - Comma-separated: "0.1,0.2,0.3"
    - Linspace: "0.1:0.5:5" (start:stop:num)
    - Single value: "0.5"
    """
    if ":" in value_str:
        # Linspace format: start:stop:num
        parts = value_str.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid linspace format: {value_str}. Use start:stop:num")
        start, stop, num = float(parts[0]), float(parts[1]), int(parts[2])
        return list(np.linspace(start, stop, num))
    elif "," in value_str:
        # Comma-separated list
        return [float(x.strip()) for x in value_str.split(",")]
    else:
        # Single value
        return [float(value_str)]


def build_param_grid(
    model: str, cli_overrides: dict[str, str | None]
) -> dict[str, list[float]]:
    """Build parameter grid from defaults and CLI overrides."""
    # Start with defaults for this model
    grid = DEFAULT_GRIDS.get(model, {}).copy()

    # Apply CLI overrides
    for param, value_str in cli_overrides.items():
        if value_str is not None:
            grid[param] = parse_param_value(value_str)

    return grid


def generate_jobs(
    config: SweepConfig,
) -> list[tuple[str, dict[str, Any]]]:
    """Generate all parameter combinations as jobs."""
    param_names = list(config.param_grid.keys())
    param_values = list(config.param_grid.values())

    jobs = []
    for i, combination in enumerate(product(*param_values)):
        params = dict(zip(param_names, combination))
        job_id = f"job_{i:05d}"
        jobs.append((job_id, params))

    return jobs


def run_single_backtest(
    job_id: str,
    params: dict[str, Any],
    config: SweepConfig,
) -> JobResult:
    """Run a single backtest with given parameters."""
    start_time = time.time()

    # Build output path for this job
    output_file = config.output_dir / "jobs" / f"{job_id}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_datalake_backtest.py"),
        "--model",
        config.model,
        "--symbols",
        *config.symbols,
        "--start-date",
        config.start_date,
        "--end-date",
        config.end_date,
        "--side",
        config.side,
        "--data-dir",
        config.data_dir,
        "--rtds-price-dir",
        config.rtds_price_dir,
        "--max-position-pct",
        str(config.max_position_pct),
        "--initial-equity",
        str(config.initial_equity),
        "--output",
        str(output_file),
    ]

    # Add parameter-specific arguments
    # Convert whole floats to ints for params that expect integers (tp-ticks, sl-ticks)
    for param, value in params.items():
        if isinstance(value, float) and value.is_integer():
            value_str = str(int(value))
        else:
            value_str = str(value)
        cmd.extend([f"--{param}", value_str])

    try:
        # Run backtest as subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        duration = time.time() - start_time

        if result.returncode != 0:
            return JobResult(
                job_id=job_id,
                params=params,
                success=False,
                error=result.stderr[:500] if result.stderr else "Unknown error",
                duration_seconds=duration,
            )

        # Parse results from output file
        if output_file.exists():
            with open(output_file) as f:
                data = json.load(f)

            agg = data.get("aggregate", {})
            return JobResult(
                job_id=job_id,
                params=params,
                success=True,
                pnl=agg.get("total_pnl", 0.0),
                pnl_after_rebate=agg.get("pnl_after_rebate", 0.0),
                sharpe_ratio=agg.get("sharpe_ratio", 0.0),
                total_fills=agg.get("total_fills", 0),
                total_volume=agg.get("total_volume", 0.0),
                max_drawdown=agg.get("max_drawdown", 0.0),
                duration_seconds=duration,
            )
        else:
            return JobResult(
                job_id=job_id,
                params=params,
                success=False,
                error="Output file not created",
                duration_seconds=duration,
            )

    except subprocess.TimeoutExpired:
        return JobResult(
            job_id=job_id,
            params=params,
            success=False,
            error="Timeout after 600 seconds",
            duration_seconds=600.0,
        )
    except Exception as e:
        return JobResult(
            job_id=job_id,
            params=params,
            success=False,
            error=str(e),
            duration_seconds=time.time() - start_time,
        )


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load checkpoint file if it exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {"completed_job_ids": [], "timestamp": 0.0}


def save_checkpoint(
    checkpoint_path: Path, completed_job_ids: list[str], config_dict: dict[str, Any]
) -> None:
    """Save checkpoint with completed job IDs."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(
            {
                "completed_job_ids": completed_job_ids,
                "timestamp": time.time(),
                "config": config_dict,
            },
            f,
            indent=2,
        )


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def print_progress(
    completed: int,
    total: int,
    result: JobResult,
    start_time: float,
) -> None:
    """Print progress line with ETA."""
    pct = 100.0 * completed / total
    elapsed = time.time() - start_time

    if completed > 0:
        eta_seconds = (elapsed / completed) * (total - completed)
        eta_str = format_duration(eta_seconds)
    else:
        eta_str = "..."

    status = "OK" if result.success else "FAIL"
    pnl_str = f"${result.pnl_after_rebate:+.2f}" if result.success else result.error[:30]

    print(f"[{completed}/{total}] ({pct:.1f}%) ETA: {eta_str} | {result.job_id}: {status} | {pnl_str}")


def print_summary(results: list[JobResult], config: SweepConfig) -> None:
    """Print summary of sweep results."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print("\n" + "=" * 80)
    print("PARAMETER SWEEP SUMMARY")
    print("=" * 80)

    print(f"Model: {config.model}")
    print(f"Symbols: {', '.join(config.symbols)}")
    print(f"Date Range: {config.start_date} to {config.end_date}")
    print(f"Total Runs: {len(results)} ({len(failed)} failed)")
    print()

    if not successful:
        print("No successful runs!")
        print("=" * 80)
        return

    pnls = [r.pnl_after_rebate for r in successful]

    best_result = max(successful, key=lambda r: r.pnl_after_rebate)
    worst_result = min(successful, key=lambda r: r.pnl_after_rebate)
    best_sharpe = max(successful, key=lambda r: r.sharpe_ratio)

    print(f"Best PnL: ${best_result.pnl_after_rebate:+.2f} ({format_params(best_result.params)})")
    print(f"Worst PnL: ${worst_result.pnl_after_rebate:+.2f}")
    print(f"Mean PnL: ${np.mean(pnls):+.2f} +/- ${np.std(pnls):.2f}")
    print(f"Best Sharpe: {best_sharpe.sharpe_ratio:.2f} ({format_params(best_sharpe.params)})")
    print()

    # Top 5 parameter sets
    print("Top 5 Parameter Sets:")
    top_5 = sorted(successful, key=lambda r: r.pnl_after_rebate, reverse=True)[:5]

    # Get all param names for table header
    param_names = list(top_5[0].params.keys()) if top_5 else []
    header = "  " + "  ".join(f"{p:>12}" for p in param_names) + "  pnl_after_rebate  sharpe_ratio"
    print(header)

    for r in top_5:
        param_vals = "  ".join(f"{r.params[p]:>12.3g}" for p in param_names)
        print(f"  {param_vals}  {r.pnl_after_rebate:>17.2f}  {r.sharpe_ratio:>12.2f}")

    print("=" * 80)


def format_params(params: dict[str, Any]) -> str:
    """Format parameters as a compact string."""
    return ", ".join(f"{k}={v}" for k, v in params.items())


def save_results(
    results: list[JobResult],
    config: SweepConfig,
) -> tuple[Path, Path]:
    """Save results to CSV and JSON files."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    successful = [r for r in results if r.success]

    # Get all param names
    param_names = list(config.param_grid.keys())

    # CSV output
    csv_path = config.output_dir / "results.csv"
    csv_header = ["job_id"] + param_names + [
        "total_pnl",
        "pnl_after_rebate",
        "sharpe_ratio",
        "total_fills",
        "total_volume",
        "max_drawdown",
        "duration_seconds",
    ]

    with open(csv_path, "w") as f:
        f.write(",".join(csv_header) + "\n")
        for r in successful:
            row = [r.job_id]
            row.extend(str(r.params.get(p, "")) for p in param_names)
            row.extend([
                str(r.pnl),
                str(r.pnl_after_rebate),
                str(r.sharpe_ratio),
                str(r.total_fills),
                str(r.total_volume),
                str(r.max_drawdown),
                str(r.duration_seconds),
            ])
            f.write(",".join(row) + "\n")

    # JSON output
    json_path = config.output_dir / "results.json"

    best_result = max(successful, key=lambda r: r.pnl_after_rebate) if successful else None
    best_sharpe = max(successful, key=lambda r: r.sharpe_ratio) if successful else None

    json_data = {
        "config": {
            "model": config.model,
            "symbols": config.symbols,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "side": config.side,
            "param_grid": {k: [float(x) for x in v] for k, v in config.param_grid.items()},
        },
        "summary": {
            "total_runs": len(results),
            "successful_runs": len(successful),
            "failed_runs": len(results) - len(successful),
            "best_pnl": best_result.pnl_after_rebate if best_result else None,
            "best_params": best_result.params if best_result else None,
            "best_sharpe": best_sharpe.sharpe_ratio if best_sharpe else None,
            "best_sharpe_params": best_sharpe.params if best_sharpe else None,
        },
        "results": [
            {
                "job_id": r.job_id,
                "params": r.params,
                "success": r.success,
                "total_pnl": r.pnl,
                "pnl_after_rebate": r.pnl_after_rebate,
                "sharpe_ratio": r.sharpe_ratio,
                "total_fills": r.total_fills,
                "total_volume": r.total_volume,
                "max_drawdown": r.max_drawdown,
                "duration_seconds": r.duration_seconds,
                "error": r.error if not r.success else None,
            }
            for r in results
        ],
        "timestamp": datetime.now().isoformat(),
    }

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    return csv_path, json_path


def run_sweep(config: SweepConfig, resume_checkpoint: Path | None = None) -> list[JobResult]:
    """Run the parameter sweep."""
    # Generate all jobs
    all_jobs = generate_jobs(config)
    total_jobs = len(all_jobs)

    print(f"Generated {total_jobs} parameter combinations")
    print(f"Parameters: {config.param_grid}")
    print(f"Workers: {config.workers}")
    print()

    # Load checkpoint for resume
    checkpoint_path = config.output_dir / "checkpoint.json"
    completed_job_ids: list[str] = []
    results: list[JobResult] = []

    if resume_checkpoint:
        checkpoint_data = load_checkpoint(resume_checkpoint)
        completed_job_ids = checkpoint_data.get("completed_job_ids", [])
        print(f"Resuming from checkpoint: {len(completed_job_ids)} jobs already completed")

        # Load existing results
        jobs_dir = config.output_dir / "jobs"
        for job_id in completed_job_ids:
            job_file = jobs_dir / f"{job_id}.json"
            if job_file.exists():
                # Find the params for this job
                job_params = next((p for jid, p in all_jobs if jid == job_id), {})
                with open(job_file) as f:
                    data = json.load(f)
                agg = data.get("aggregate", {})
                results.append(
                    JobResult(
                        job_id=job_id,
                        params=job_params,
                        success=True,
                        pnl=agg.get("total_pnl", 0.0),
                        pnl_after_rebate=agg.get("pnl_after_rebate", 0.0),
                        sharpe_ratio=agg.get("sharpe_ratio", 0.0),
                        total_fills=agg.get("total_fills", 0),
                        total_volume=agg.get("total_volume", 0.0),
                        max_drawdown=agg.get("max_drawdown", 0.0),
                    )
                )

    # Filter out completed jobs
    pending_jobs = [(jid, params) for jid, params in all_jobs if jid not in completed_job_ids]
    print(f"Pending jobs: {len(pending_jobs)}")
    print()

    if not pending_jobs:
        print("All jobs already completed!")
        return results

    # Prepare config dict for checkpoint
    config_dict = {
        "model": config.model,
        "symbols": config.symbols,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "side": config.side,
        "param_grid": {k: [float(x) for x in v] for k, v in config.param_grid.items()},
    }

    # Run jobs in parallel
    start_time = time.time()
    completed_count = len(completed_job_ids)

    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        # Submit all pending jobs
        future_to_job = {
            executor.submit(run_single_backtest, job_id, params, config): (job_id, params)
            for job_id, params in pending_jobs
        }

        # Process results as they complete
        for future in as_completed(future_to_job):
            job_id, params = future_to_job[future]

            try:
                result = future.result()
            except Exception as e:
                result = JobResult(
                    job_id=job_id,
                    params=params,
                    success=False,
                    error=str(e),
                )

            results.append(result)
            completed_job_ids.append(job_id)
            completed_count += 1

            # Print progress
            print_progress(completed_count, total_jobs, result, start_time)

            # Save checkpoint
            save_checkpoint(checkpoint_path, completed_job_ids, config_dict)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run parameter sweep for backtests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Resume mode
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint file",
    )

    # Model and data selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["spread", "inventory", "tpbs", "tpsl_bs"],
        default="tpbs",
        help="Quote model type (default: tpbs)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        type=str,
        default=["btc"],
        help="Symbols to backtest (default: btc)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["up", "down", "both"],
        default="up",
        help="Which side to trade (default: up)",
    )

    # Execution settings
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() - 1) if os.cpu_count() else 4,
        help="Number of parallel workers (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sweep_results",
        help="Output directory for results",
    )

    # Data directories
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/datalake/processed/15m",
        help="Path to processed datalake data",
    )
    parser.add_argument(
        "--rtds-price-dir",
        type=str,
        default="data/datalake/global/rtds_crypto_prices",
        help="Path to RTDS price data",
    )

    # Common parameters
    parser.add_argument(
        "--max-position-pct",
        type=str,
        default=None,
        help="Max position as %% of equity (comma-list or linspace)",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=10000.0,
        help="Initial equity (default: 10000)",
    )

    # Model-specific parameter overrides (comma-list or linspace)
    parser.add_argument(
        "--base-spread",
        type=str,
        default=None,
        help="Base spread values (comma-list or start:stop:num)",
    )
    parser.add_argument(
        "--inventory-skew",
        type=str,
        default=None,
        help="Inventory skew values (comma-list or start:stop:num)",
    )
    parser.add_argument(
        "--max-z",
        type=str,
        default=None,
        help="Max z-score values for TpBS (comma-list or start:stop:num)",
    )
    parser.add_argument(
        "--min-z",
        type=str,
        default=None,
        help="Min z-score values for TpBS (comma-list or start:stop:num)",
    )
    parser.add_argument(
        "--implied-vol",
        type=str,
        default=None,
        help="Implied volatility values (comma-list or start:stop:num)",
    )
    parser.add_argument(
        "--z",
        type=str,
        default=None,
        help="Z-score values for TpSL-BS (comma-list or start:stop:num)",
    )
    parser.add_argument(
        "--tp-ticks",
        type=str,
        default=None,
        help="Take-profit tick values for TpSL-BS (comma-list or start:stop:num)",
    )
    parser.add_argument(
        "--sl-ticks",
        type=str,
        default=None,
        help="Stop-loss tick values for TpSL-BS (comma-list or start:stop:num)",
    )

    args = parser.parse_args()

    # Handle resume mode
    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            return 1

        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)

        saved_config = checkpoint_data.get("config", {})
        output_dir = checkpoint_path.parent

        config = SweepConfig(
            model=saved_config.get("model", args.model),
            symbols=saved_config.get("symbols", args.symbols),
            start_date=saved_config.get("start_date", args.start_date),
            end_date=saved_config.get("end_date", args.end_date),
            side=saved_config.get("side", args.side),
            workers=args.workers,
            output_dir=output_dir,
            data_dir=args.data_dir,
            rtds_price_dir=args.rtds_price_dir,
            param_grid=saved_config.get("param_grid", {}),
        )

        results = run_sweep(config, resume_checkpoint=checkpoint_path)
    else:
        # Build parameter grid from CLI overrides
        cli_overrides = {
            "base-spread": args.base_spread,
            "inventory-skew": args.inventory_skew,
            "max-z": args.max_z,
            "min-z": args.min_z,
            "implied-vol": args.implied_vol,
            "z": args.z,
            "tp-ticks": args.tp_ticks,
            "sl-ticks": args.sl_ticks,
            "max-position-pct": args.max_position_pct,
        }

        param_grid = build_param_grid(args.model, cli_overrides)

        if not param_grid:
            print(f"Error: No parameters to sweep for model '{args.model}'")
            print("Specify parameters via CLI or use a supported model with defaults")
            return 1

        output_dir = Path(args.output_dir)

        config = SweepConfig(
            model=args.model,
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            side=args.side,
            workers=args.workers,
            output_dir=output_dir,
            data_dir=args.data_dir,
            rtds_price_dir=args.rtds_price_dir,
            param_grid=param_grid,
            max_position_pct=0.20,  # Default, not swept unless specified
            initial_equity=args.initial_equity,
        )

        results = run_sweep(config)

    # Print summary
    print_summary(results, config)

    # Save results
    csv_path, json_path = save_results(results, config)
    print("\nResults saved to:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
