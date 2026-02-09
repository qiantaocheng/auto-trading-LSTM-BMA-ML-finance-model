#!/usr/bin/env python
"""
9-Run Factor Grid Search (LambdaRank Only) - Systematic Factor Combination Test

S01 �?Base (5) 稳定底座
    rsi_21, ivol_20, vol_ratio_20d, trend_r2_60, near_52w_high

S02 �?Base + 新corr因子 (6)
    S01 + alpha_linreg_corr_10d

S03 �?均值回归分�?(8)
    S02 + 5_days_reversal + 

S04 �?动量分支 (8)
    S02 + momentum_10d + liquid_momentum

S05 �?趋势强度分支 (8)
    S02 + trend_strength_20d + sharpe_momentum_20d

S06 �?量价关系分支 (8)
    S02 + obv_divergence + volume_price_corr_10d

S07 �?均值回�?+ 量价 (9)
    S03 + obv_divergence

S08 �?动量 + 均值回归混�?(10)
    S04 + 5_days_reversal + 

S09 �?综合强化�?(12)
    rsi_21, ivol_20, vol_ratio_20d, trend_r2_60, near_52w_high,
    alpha_linreg_corr_10d, 5_days_reversal, ,
    momentum_10d, liquid_momentum, trend_strength_20d, obv_divergence
"""

import argparse
import subprocess
import sys
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Configuration
DATA_FILE = r"D:\trade\data\factor_exports\polygon_factors_all_2021_2026_CLEAN_STANDARDIZED.parquet"
TRADE_DIR = Path(r"D:\trade")
OUTPUT_BASE = TRADE_DIR / "results" / "9run_factor_grid"
DEFAULT_TICKER_FRACTION = 1.0  # Full data by default
RANDOM_SEED = 42
MAX_PARALLEL_JOBS = 2  # Run two experiments concurrently

# =============================================================================
# FACTOR SET DEFINITIONS
# =============================================================================

# S01: Base (5 factors) - 稳定底座
S01_BASE = [
    "rsi_21",
    "ivol_20",
    "vol_ratio_20d",
    "trend_r2_60",
    "near_52w_high",
]

# S02: Base + corr factor (6 factors)
S02_BASE_CORR = S01_BASE + [
    "alpha_linreg_corr_10d",
]

# S03: 均值回归分�?(8 factors)
S03_MEAN_REVERSION = S02_BASE_CORR + [
    "5_days_reversal",
    ,
]

# S04: 动量分支 (8 factors)
S04_MOMENTUM = S02_BASE_CORR + [
    "momentum_10d",
    "liquid_momentum",
]

# S05: 趋势强度分支 (8 factors)
S05_TREND_STRENGTH = S02_BASE_CORR + [
    "trend_strength_20d",
    "sharpe_momentum_20d",
]

# S06: 量价关系分支 (8 factors)
S06_VOLUME_PRICE = S02_BASE_CORR + [
    "obv_divergence",
    "volume_price_corr_10d",
]

# S07: 均值回�?+ 量价 (9 factors)
S07_MEAN_REV_VOL = S03_MEAN_REVERSION + [
    "obv_divergence",
]

# S08: 动量 + 均值回归混�?(10 factors)
S08_MOM_MEAN_REV = S04_MOMENTUM + [
    "5_days_reversal",
    ,
]

# S09: 综合强化�?(12 factors)
S09_COMPREHENSIVE = [
    "rsi_21",
    "ivol_20",
    "vol_ratio_20d",
    "trend_r2_60",
    "near_52w_high",
    "alpha_linreg_corr_10d",
    "5_days_reversal",
    ,
    "momentum_10d",
    "liquid_momentum",
    "trend_strength_20d",
    "obv_divergence",
]

# =============================================================================
# RUN CONFIGURATIONS - All 9 runs use CUSTOM_FEATURE_RUNS
# =============================================================================

CUSTOM_FEATURE_RUNS = {
    "S01_base_5":           S01_BASE,
    "S02_base_corr_6":      S02_BASE_CORR,
    "S03_mean_rev_8":       S03_MEAN_REVERSION,
    "S04_momentum_8":       S04_MOMENTUM,
    "S05_trend_str_8":      S05_TREND_STRENGTH,
    "S06_vol_price_8":      S06_VOLUME_PRICE,
    "S07_meanrev_vol_9":    S07_MEAN_REV_VOL,
    "S08_mom_meanrev_10":   S08_MOM_MEAN_REV,
    "S09_comprehensive_12": S09_COMPREHENSIVE,
}

# Empty standard runs (all runs are custom)
RUN_CONFIGS = {}

os.chdir(TRADE_DIR)
sys.path.insert(0, str(TRADE_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="9-run LambdaRank-only factor grid search")
    parser.add_argument(
        "--ticker-fraction",
        type=float,
        default=DEFAULT_TICKER_FRACTION,
        help="Fraction of tickers to sample (default 1.0 = full data).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=MAX_PARALLEL_JOBS,
        help="Number of concurrent runs (default 2).",
    )
    parser.add_argument(
        "--single-run",
        choices=list(CUSTOM_FEATURE_RUNS.keys()),
        help="Optional run key (e.g. S01_base_5) to execute only one configuration.",
    )
    return parser.parse_args()


def get_sampled_tickers(data_file: str, fraction: float, seed: int) -> List[str]:
    """Load data and sample tickers"""
    print(f"Loading data: {data_file}")
    df = pd.read_parquet(data_file)
    if isinstance(df.index, pd.MultiIndex):
        all_tickers = df.index.get_level_values('ticker').unique().tolist()
    else:
        all_tickers = df['ticker'].unique().tolist()

    if fraction >= 1.0:
        print(f"Using all {len(all_tickers)} tickers (100%)")
        return sorted(all_tickers)

    np.random.seed(seed)
    n_sample = int(len(all_tickers) * fraction)
    sampled = np.random.choice(all_tickers, size=n_sample, replace=False).tolist()
    print(f"Sampled {len(sampled)}/{len(all_tickers)} tickers ({fraction*100:.0f}%)")
    return sorted(sampled)


def create_subset_data(data_file: str, tickers: List[str], output_path: Path) -> str:
    """Create subset parquet with sampled tickers"""
    print(f"  Creating subset data with {len(tickers)} tickers...")
    df = pd.read_parquet(data_file)

    if isinstance(df.index, pd.MultiIndex):
        mask = df.index.get_level_values('ticker').isin(tickers)
        subset = df.loc[mask].copy()
    else:
        subset = df[df['ticker'].isin(tickers)].copy()

    subset_file = output_path / "subset_data.parquet"
    subset.to_parquet(subset_file)
    print(f"  Saved {len(subset)} rows")
    return str(subset_file)


def run_single_config(
    run_name: str,
    features: List[str],
    data_file: str,
    output_dir: Path
) -> Dict:
    """Run one configuration with LambdaRank only"""

    print(f"\n{'='*70}")
    print(f"RUN: {run_name}")
    print(f"{'='*70}")
    print(f"Features ({len(features)}): {features}")

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save feature list
    (run_dir / "features.json").write_text(json.dumps(features, indent=2))

    # Build command - LambdaRank only (skip training/predicting other models)
    cmd = [
        sys.executable, str(TRADE_DIR / "scripts" / "time_split_80_20_oos_eval.py"),
        "--data-file", data_file,
        "--features"] + features + [
        "--horizon-days", "10",
        "--split", "0.8",
        "--lambdarank-only",  # Skip ElasticNet, XGBoost, CatBoost, RidgeStacking prediction
        "--models", "lambdarank",  # Export only LambdaRank metrics
        "--top-n", "20",
        "--ema-top-n", "300",
        "--ema-length", "2",
        "--ema-beta", "0.7",
        "--ema-min-days", "2",
        "--output-dir", str(run_dir),
        "--log-level", "WARNING"
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(TRADE_DIR)
    env["BMA_TRAIN_ONLY_MODEL"] = "lambdarank"  # Skip training ElasticNet, XGBoost, CatBoost (only train LambdaRank)
    env["BMA_DISABLE_COMPULSORY_FEATURES"] = "1"

    print("Running 80/20 time split (LambdaRank only - training + prediction)...")
    result = subprocess.run(cmd, env=env, cwd=str(TRADE_DIR), capture_output=True, text=True)

    # Collect metrics
    metrics = {
        'run': run_name,
        'n_features': len(features),
        'features': features
    }

    for sub_dir in run_dir.glob("run_*"):
        # Load accumulated returns
        lr_file = sub_dir / "lambdarank_top5_15_rebalance10d_accumulated.csv"
        if lr_file.exists():
            lr_df = pd.read_csv(lr_file)
            metrics['top5_15_mean'] = lr_df['top_gross_return'].mean()
            metrics['top5_15_median'] = lr_df['top_gross_return'].median()
            metrics['acc_return'] = lr_df.iloc[-1]['acc_return']
            metrics['max_drawdown'] = lr_df['drawdown'].min()
            metrics['n_periods'] = len(lr_df)

            # Calculate win rate
            metrics['win_rate'] = (lr_df['top_gross_return'] > 0).mean()

            print(f"  Top5-15 Mean: {metrics['top5_15_mean']*100:.2f}%")
            print(f"  Top5-15 Median: {metrics['top5_15_median']*100:.2f}%")
            print(f"  Accumulated: {metrics['acc_return']*100:.1f}%")
            print(f"  MaxDD: {metrics['max_drawdown']:.1f}%")
            print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")

        # Load bucket returns for spread analysis
        bucket_file = sub_dir / "lambdarank_bucket_returns.csv"
        if bucket_file.exists():
            bucket_df = pd.read_csv(bucket_file)
            # Calculate Top1-10 vs Top11-20 spread
            if 'bucket' in bucket_df.columns and 'mean_return' in bucket_df.columns:
                top1_10 = bucket_df[bucket_df['bucket'] == 'top_1_10']['mean_return'].values
                top11_20 = bucket_df[bucket_df['bucket'] == 'top_11_20']['mean_return'].values
                if len(top1_10) > 0 and len(top11_20) > 0:
                    metrics['top1_10_mean'] = float(top1_10[0])
                    metrics['top11_20_mean'] = float(top11_20[0])
                    metrics['spread'] = float(top1_10[0] - top11_20[0])

        # Load OOS metrics
        oos_file = sub_dir / "oos_metrics.json"
        if oos_file.exists():
            with open(oos_file) as f:
                oos = json.load(f)
                metrics['IC'] = oos.get('IC')
                metrics['Rank_IC'] = oos.get('Rank_IC')

    if result.returncode != 0 and 'acc_return' not in metrics:
        print(f"  Error: {result.stderr[-500:] if result.stderr else 'Unknown'}")

    return metrics


def main():
    args = parse_args()
    ticker_fraction = max(min(float(args.ticker_fraction), 1.0), 0.01)
    max_parallel = max(int(args.max_parallel), 1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure output directories exist
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    output_dir = OUTPUT_BASE / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not output_dir.exists():
        raise RuntimeError(f"Failed to create output directory: {output_dir}")

    print("="*70)
    print("9-RUN FACTOR GRID SEARCH (LAMBDARANK ONLY)")
    print("="*70)
    print(f"Data: {DATA_FILE}")
    print(f"Ticker Fraction: {ticker_fraction*100:.2f}%")
    print(f"Max Parallel: {max_parallel}")
    print(f"Output: {output_dir}")
    print("="*70)
    print("\nRun Configurations:")
    for name, features in CUSTOM_FEATURE_RUNS.items():
        print(f"  {name}: {len(features)} factors")
    print("="*70)

    # Sample tickers
    sampled_tickers = get_sampled_tickers(DATA_FILE, ticker_fraction, RANDOM_SEED)
    (output_dir / "sampled_tickers.txt").write_text("\n".join(sampled_tickers))

    # Create subset data once (or use full data)
    if ticker_fraction >= 1.0:
        subset_file = DATA_FILE
        print(f"Using full data file: {subset_file}")
    else:
        subset_file = create_subset_data(DATA_FILE, sampled_tickers, output_dir)

    # Build all runs
    all_runs = {}
    for run_name, full_features in CUSTOM_FEATURE_RUNS.items():
        all_runs[run_name] = full_features

    if args.single_run:
        if args.single_run in all_runs:
            all_runs = {args.single_run: all_runs[args.single_run]}
        else:
            raise ValueError(f"Unknown run: {args.single_run}")

    # Run all configurations
    all_results = []
    futures = {}

    print(f"\nLaunching {len(all_runs)} runs in parallel (max {max_parallel} concurrent jobs)...")
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        for run_name, features in all_runs.items():
            future = executor.submit(run_single_config, run_name, features, subset_file, output_dir)
            futures[future] = run_name

        for future in as_completed(futures):
            run_name = futures[future]
            try:
                metrics = future.result()
            except Exception as exc:
                print(f"Run {run_name} failed: {exc}")
                continue
            all_results.append(metrics)

    # Summary
    print("\n" + "="*70)
    print("9-RUN GRID SEARCH SUMMARY")
    print("="*70)

    # Preserve original run order for downstream analysis
    ordered_results = []
    metrics_by_run = {m.get('run'): m for m in all_results if m.get('run')}
    for run_name in all_runs.keys():
        if run_name in metrics_by_run:
            ordered_results.append(metrics_by_run[run_name])
    df = pd.DataFrame(ordered_results)

    print("\nResults by Top5-15 Mean (primary metric):")
    print("-"*90)

    # Sort by top5_15_mean descending
    if 'top5_15_mean' in df.columns:
        df_sorted = df.sort_values('top5_15_mean', ascending=False)

        print(f"{'Run':<25} {'#F':>3} {'Mean%':>8} {'Median%':>8} {'Acc%':>10} {'MaxDD%':>8} {'WinRate':>8}")
        print("-"*90)

        for _, row in df_sorted.iterrows():
            n_feat = row.get('n_features', 0)
            mean_pct = f"{row.get('top5_15_mean', 0)*100:.2f}" if pd.notna(row.get('top5_15_mean')) else "N/A"
            median_pct = f"{row.get('top5_15_median', 0)*100:.2f}" if pd.notna(row.get('top5_15_median')) else "N/A"
            acc_pct = f"{row.get('acc_return', 0)*100:.1f}" if pd.notna(row.get('acc_return')) else "N/A"
            dd_pct = f"{row.get('max_drawdown', 0):.1f}" if pd.notna(row.get('max_drawdown')) else "N/A"
            win_rate = f"{row.get('win_rate', 0)*100:.1f}%" if pd.notna(row.get('win_rate')) else "N/A"
            print(f"{row['run']:<25} {n_feat:>3} {mean_pct:>8} {median_pct:>8} {acc_pct:>10} {dd_pct:>8} {win_rate:>8}")

    # Best configuration
    if 'top5_15_mean' in df.columns and len(df) > 0:
        best_idx = df['top5_15_mean'].idxmax()
        best = df.loc[best_idx]
        print(f"\n{'='*70}")
        print(f"BEST RUN: {best['run']}")
        print(f"  Features ({best['n_features']}): {best['features']}")
        print(f"  Top5-15 Mean: {best['top5_15_mean']*100:.2f}%")
        print(f"  Top5-15 Median: {best['top5_15_median']*100:.2f}%")
        print(f"  Accumulated: {best['acc_return']*100:.1f}%")
        print(f"  MaxDD: {best['max_drawdown']:.1f}%")

    # Save results
    df.to_csv(output_dir / "9run_grid_results.csv", index=False)

    config = {
        "data_file": DATA_FILE,
        "run_configs": {k: v for k, v in CUSTOM_FEATURE_RUNS.items()},
        "ticker_fraction": ticker_fraction,
        "n_sampled_tickers": len(sampled_tickers),
        "random_seed": RANDOM_SEED,
        "model": "lambdarank_only",
        "ema_config": {"length": 2, "beta": 0.7, "top_n": 300, "min_days": 2},
        "timestamp": timestamp
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\nResults saved to: {output_dir}")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
