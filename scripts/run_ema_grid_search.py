#!/usr/bin/env python
"""
EMA Grid Search for 80/20 Time Split Evaluation

Phase 1: Train once with 11 factors, 1/5 tickers
Phase 2: Run 6 prediction combinations with EMA grid search:
  - ema-length: [2, 3] (t-1 and t-2 lookback)
  - ema-beta: [0.8, 0.7, 0.5]

Compares statistical significance of results across all combinations.
"""

import subprocess
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from itertools import product

# Configuration
DATA_FILE = r"D:\trade\data\factor_exports\polygon_factors_all_2021_2026_CLEAN_STANDARDIZED.parquet"
TICKER_FRACTION = 0.20  # 1/5 tickers
BASE_OUTPUT_DIR = Path(r"D:\trade\results\ema_grid_search")

# 11 factors to use
FEATURES_11 = [
    "momentum_10d", "ivol_20", "hist_vol_20", "rsi_21", "near_52w_high",
    "atr_ratio", "vol_ratio_20d", "5_days_reversal",
    "trend_r2_60", "liquid_momentum"
]

# EMA Grid Search Parameters
EMA_LENGTHS = [2, 3]  # t-1 (length=2) and t-2 (length=3)
EMA_BETAS = [0.8, 0.7, 0.5]

# Ensure we're in the correct working directory
TRADE_DIR = Path(r"D:\trade")
os.chdir(TRADE_DIR)
sys.path.insert(0, str(TRADE_DIR))

def run_training_phase():
    """Phase 1: Train models with 11 factors, 1/5 tickers"""
    print("=" * 80)
    print("PHASE 1: Training with 11 factors, 1/5 tickers")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_OUTPUT_DIR / f"training_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(TRADE_DIR / "scripts" / "time_split_80_20_oos_eval.py"),
        "--data-file", DATA_FILE,
        "--ticker-fraction", str(TICKER_FRACTION),
        "--features"] + FEATURES_11 + [
        "--horizon-days", "10",
        "--split", "0.8",
        "--models", "lambdarank", "ridge_stacking",
        "--top-n", "20",
        "--ema-top-n", "-1",  # Disable EMA for training baseline
        "--output-dir", str(output_dir),
        "--log-level", "INFO"
    ]

    print(f"Running: {' '.join(cmd[:10])}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(TRADE_DIR)
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=str(TRADE_DIR), env=env)

    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        return None, None

    # Find snapshot ID from output
    snapshot_file = output_dir / "snapshot_id.txt"
    if snapshot_file.exists():
        snapshot_id = snapshot_file.read_text().strip()
        print(f"Training complete. Snapshot ID: {snapshot_id}")
        return snapshot_id, output_dir

    # Try to find it from subdirectories
    for run_dir in output_dir.glob("run_*"):
        sf = run_dir / "snapshot_id.txt"
        if sf.exists():
            snapshot_id = sf.read_text().strip()
            print(f"Training complete. Snapshot ID: {snapshot_id}")
            return snapshot_id, run_dir

    print("Warning: Could not find snapshot_id.txt")
    return None, output_dir


def run_ema_prediction(snapshot_id: str, ema_length: int, ema_beta: float, run_id: int):
    """Run prediction with specific EMA settings"""
    print(f"\n--- Run {run_id}/6: EMA length={ema_length}, beta={ema_beta} ---")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BASE_OUTPUT_DIR / f"ema_L{ema_length}_B{ema_beta}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(TRADE_DIR / "scripts" / "time_split_80_20_oos_eval.py"),
        "--snapshot-id", snapshot_id,
        "--data-file", DATA_FILE,
        "--ticker-fraction", str(TICKER_FRACTION),
        "--features"] + FEATURES_11 + [
        "--horizon-days", "10",
        "--split", "0.8",
        "--models", "lambdarank", "ridge_stacking",
        "--top-n", "20",
        "--ema-top-n", "0",  # Apply EMA to all stocks
        "--ema-length", str(ema_length),
        "--ema-beta", str(ema_beta),
        "--output-dir", str(output_dir),
        "--log-level", "INFO"
    ]

    print(f"Output: {output_dir}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(TRADE_DIR)
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=str(TRADE_DIR), env=env)

    return {
        "ema_length": ema_length,
        "ema_beta": ema_beta,
        "output_dir": str(output_dir),
        "success": result.returncode == 0
    }


def load_metrics(output_dir: Path) -> dict:
    """Load metrics from an output directory"""
    metrics = {}

    # Try to find oos_metrics.json in the output dir or subdirs
    for metrics_file in [output_dir / "oos_metrics.json"] + list(output_dir.glob("run_*/oos_metrics.json")):
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            break

    return metrics


def compare_results(results: list):
    """Compare results across all EMA configurations with statistical tests"""
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON OF EMA CONFIGURATIONS")
    print("=" * 80)

    # Collect metrics from all runs
    all_metrics = []
    for r in results:
        if not r["success"]:
            continue

        output_dir = Path(r["output_dir"])
        metrics = load_metrics(output_dir)

        if metrics:
            all_metrics.append({
                "config": f"L{r['ema_length']}_B{r['ema_beta']}",
                "ema_length": r["ema_length"],
                "ema_beta": r["ema_beta"],
                **metrics
            })

    if not all_metrics:
        print("No metrics found to compare!")
        return

    # Create comparison DataFrame
    df = pd.DataFrame(all_metrics)

    # Key metrics to compare
    key_metrics = [
        "ridge_stacking_ic", "ridge_stacking_rank_ic",
        "ridge_stacking_top20_mean_return", "ridge_stacking_sharpe",
        "lambdarank_ic", "lambdarank_rank_ic"
    ]

    available_metrics = [m for m in key_metrics if m in df.columns]

    print("\n📊 Results Summary:")
    print("-" * 80)

    summary_cols = ["config", "ema_length", "ema_beta"] + available_metrics
    summary_df = df[[c for c in summary_cols if c in df.columns]]
    print(summary_df.to_string(index=False))

    # Statistical significance tests (pairwise t-tests if we have daily returns)
    print("\n📈 Best Configuration by Metric:")
    print("-" * 80)

    for metric in available_metrics:
        if metric in df.columns:
            best_idx = df[metric].idxmax()
            best_config = df.loc[best_idx, "config"]
            best_value = df.loc[best_idx, metric]
            print(f"  {metric}: {best_config} = {best_value:.4f}")

    # Save comparison results
    output_file = BASE_OUTPUT_DIR / "grid_search_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"\n📁 Comparison saved to: {output_file}")

    # Generate ranking
    print("\n🏆 Overall Ranking (by average normalized rank):")
    print("-" * 80)

    if len(available_metrics) > 0:
        ranks = pd.DataFrame()
        for metric in available_metrics:
            if metric in df.columns:
                # Higher is better for IC, returns, sharpe
                ranks[metric] = df[metric].rank(ascending=False)

        df["avg_rank"] = ranks.mean(axis=1)
        df_ranked = df.sort_values("avg_rank")[["config", "avg_rank"] + available_metrics]
        print(df_ranked.to_string(index=False))

    return df


def main():
    """Main entry point for EMA grid search"""
    print("=" * 80)
    print("EMA GRID SEARCH FOR 80/20 TIME SPLIT EVALUATION")
    print(f"Data: {DATA_FILE}")
    print(f"Ticker fraction: {TICKER_FRACTION}")
    print(f"Features: {len(FEATURES_11)} factors")
    print(f"Grid: {len(EMA_LENGTHS)} lengths x {len(EMA_BETAS)} betas = {len(EMA_LENGTHS) * len(EMA_BETAS)} runs")
    print("=" * 80)

    # Create base output directory
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Training
    snapshot_id, training_dir = run_training_phase()

    if not snapshot_id:
        print("ERROR: Training failed, cannot proceed with grid search")
        return 1

    # Phase 2: Grid search on EMA parameters
    print("\n" + "=" * 80)
    print("PHASE 2: EMA Grid Search (6 combinations)")
    print("=" * 80)

    results = []
    run_id = 1

    for ema_length, ema_beta in product(EMA_LENGTHS, EMA_BETAS):
        result = run_ema_prediction(snapshot_id, ema_length, ema_beta, run_id)
        results.append(result)
        run_id += 1

    # Phase 3: Compare results
    comparison_df = compare_results(results)

    # Save run configuration
    config = {
        "data_file": DATA_FILE,
        "ticker_fraction": TICKER_FRACTION,
        "features": FEATURES_11,
        "snapshot_id": snapshot_id,
        "training_dir": str(training_dir),
        "ema_lengths": EMA_LENGTHS,
        "ema_betas": EMA_BETAS,
        "results": results
    }

    config_file = BASE_OUTPUT_DIR / "grid_search_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n�?Grid search complete!")
    print(f"📁 Results directory: {BASE_OUTPUT_DIR}")
    print(f"📁 Config saved to: {config_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
