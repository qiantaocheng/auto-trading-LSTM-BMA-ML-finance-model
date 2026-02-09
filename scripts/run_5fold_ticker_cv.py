#!/usr/bin/env python
"""
N-Bucket Ticker Cross-Validation with 80/20 Time Split

Splits tickers into N equal buckets. For each bucket:
- Create subset parquet with that bucket's tickers
- Run 80/20 time split (train on 80% dates, test on 20% dates)
- Retrain and test OOS on that bucket

Default: 3 buckets. Uses default 11 factors and EMA L2 B0.7 with Top300 Gate.
"""

import subprocess
import sys
import os
import json
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import tempfile
from typing import Dict, List, Optional

# Configuration
DATA_FILE = r"D:\trade\data\factor_exports\polygon_factors_all_2021_2026_CLEAN_STANDARDIZED.parquet"
TRADE_DIR = Path(r"D:\trade")
BASE_SEED = 42

# Default 11 factors (TOP_FEATURE_SET)
FEATURES_11 = [
    "momentum_10d", "ivol_20", "hist_vol_20", "rsi_21", "near_52w_high",
    "atr_ratio", "vol_ratio_20d", "5_days_reversal",
    "trend_r2_60", "liquid_momentum"
]

# Old default 9 factors (TOP_9_FEATURES / Stage-A)
FEATURES_9 = [
    "hist_vol_20", "momentum_10d", "trend_r2_60", "near_52w_high",
    "rsi_21", "vol_ratio_20d", "5_days_reversal", "alpha_linreg_corr_20d",
    "liquid_momentum"
]

os.chdir(TRADE_DIR)
sys.path.insert(0, str(TRADE_DIR))


def get_all_tickers(data_file: str) -> list:
    """Load data and get unique tickers"""
    print(f"Loading data to get tickers: {data_file}")
    df = pd.read_parquet(data_file)
    if isinstance(df.index, pd.MultiIndex):
        tickers = df.index.get_level_values('ticker').unique().tolist()
    else:
        tickers = df['ticker'].unique().tolist()
    print(f"Found {len(tickers)} unique tickers")
    return sorted(tickers)


def split_tickers_into_buckets(tickers: list, n_buckets: int, seed: int) -> list:
    """Split tickers into n equal buckets"""
    np.random.seed(seed)
    shuffled = np.random.permutation(tickers).tolist()
    bucket_size = len(shuffled) // n_buckets
    buckets = []
    for i in range(n_buckets):
        start = i * bucket_size
        if i == n_buckets - 1:
            buckets.append(shuffled[start:])
        else:
            buckets.append(shuffled[start:start + bucket_size])
    return buckets


def create_bucket_data(data_file: str, bucket_tickers: list, output_path: Path) -> str:
    """Create a subset parquet file with only the specified tickers"""
    print(f"  Creating bucket data with {len(bucket_tickers)} tickers...")
    df = pd.read_parquet(data_file)

    if isinstance(df.index, pd.MultiIndex):
        ticker_mask = df.index.get_level_values('ticker').isin(bucket_tickers)
        subset = df.loc[ticker_mask].copy()
    else:
        subset = df[df['ticker'].isin(bucket_tickers)].copy()

    subset_file = output_path / "bucket_data.parquet"
    subset.to_parquet(subset_file)
    print(f"  Saved {len(subset)} rows to {subset_file}")
    return str(subset_file)


_TIME_SPLIT_MODULE = None


def load_time_split_module() -> Optional[object]:
    """Lazy load the 80/20 evaluation helpers so we can reuse its utilities."""
    global _TIME_SPLIT_MODULE
    if _TIME_SPLIT_MODULE is None:
        module_path = TRADE_DIR / "scripts" / "time_split_80_20_oos_eval.py"
        if not module_path.exists():
            print(f"Unable to locate helper module: {module_path}")
            return None
        spec = importlib.util.spec_from_file_location("time_split_80_20_oos_eval", module_path)
        if spec is None or spec.loader is None:
            print("Failed to build import spec for time_split_80_20_oos_eval.py")
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        _TIME_SPLIT_MODULE = module
    return _TIME_SPLIT_MODULE


def run_global_full_training(features_list: List[str], output_dir: Path) -> Optional[Path]:
    """Train once on the full ticker universe and return the run directory."""
    global_dir = output_dir / "global_full_run"
    global_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(TRADE_DIR / "scripts" / "time_split_80_20_oos_eval.py"),
        "--train-data", DATA_FILE,
        "--data-file", DATA_FILE,
        "--features",
    ] + features_list + [
        "--horizon-days", "10",
        "--split", "0.8",
        "--models", "lambdarank", "ridge_stacking",
        "--top-n", "20",
        "--ema-top-n", "300",
        "--ema-length", "2",
        "--ema-beta", "0.7",
        "--ema-min-days", "2",
        "--output-dir", str(global_dir),
        "--log-level", "WARNING"
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(TRADE_DIR)

    print("Running single full-universe 80/20 retrain so we can reuse predictions across buckets...")
    result = subprocess.run(cmd, env=env, cwd=str(TRADE_DIR), capture_output=True, text=True)
    if result.returncode != 0:
        print("  Global training run failed. Falling back to per-bucket retrain.")
        stderr_tail = result.stderr[-400:] if result.stderr else "  (no stderr captured)"
        print(stderr_tail)
        return None

    run_dirs = sorted(
        [d for d in global_dir.glob("run_*") if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        print("  No run directories found after global training.")
        return None

    run_dir = run_dirs[0]
    print(f"  Reusing predictions from: {run_dir}")
    return run_dir


def load_global_prediction_frames(run_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load prediction diagnosis files from a time_split run into memory."""
    predictions: Dict[str, pd.DataFrame] = {}
    file_map = {
        'lambdarank': 'lambdarank_predictions_diagnosis.csv',
        'ridge_stacking': 'ridge_stacking_predictions_diagnosis.csv',
    }
    for model_name, filename in file_map.items():
        path = run_dir / filename
        if not path.exists():
            print(f"  Warning: {filename} not found in {run_dir}")
            continue
        df = pd.read_csv(path)
        required_cols = {'date', 'ticker', 'prediction', 'actual'}
        if not required_cols.issubset(df.columns):
            print(f"  Warning: {filename} missing required columns: {required_cols - set(df.columns)}")
            continue
        df = df[list(required_cols)].copy()
        df['date'] = pd.to_datetime(df['date'])
        df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
        df = df.dropna(subset=['prediction', 'actual'])
        predictions[model_name] = df
    return predictions



def run_bucket(
    bucket_idx: int,
    bucket_tickers: list,
    output_dir: Path,
    n_buckets: int,
    features_list: list,
    global_predictions: Optional[Dict[str, pd.DataFrame]] = None,
) -> dict:
    """Dispatch to per-bucket retraining or global prediction reuse."""
    if global_predictions:
        helpers = load_time_split_module()
        if helpers is None:
            print("Unable to load 80/20 helper module; falling back to per-bucket retrain.")
        else:
            return _run_bucket_with_global_predictions(
                bucket_idx,
                bucket_tickers,
                output_dir,
                n_buckets,
                features_list,
                global_predictions,
                helpers,
            )
    return _run_bucket_with_training(bucket_idx, bucket_tickers, output_dir, n_buckets, features_list)


def _run_bucket_with_global_predictions(
    bucket_idx: int,
    bucket_tickers: list,
    output_dir: Path,
    n_buckets: int,
    features_list: list,
    global_predictions: Dict[str, pd.DataFrame],
    helpers,
) -> dict:
    """Use a single global retrain to score each bucket without retraining per fold."""
    print(f"\n{'='*70}")
    print(f"{'='*70}")
    print(f"BUCKET {bucket_idx + 1}/{n_buckets} (global predictions)")
    print(f"{'='*70}")
    print(f"Tickers in this bucket: {len(bucket_tickers)}")

    bucket_dir = output_dir / f"bucket_{bucket_idx + 1}"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    (bucket_dir / "bucket_tickers.txt").write_text("\n".join(bucket_tickers))

    run_dir = bucket_dir / "run_global_eval"
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = {'bucket': bucket_idx + 1, 'n_tickers': len(bucket_tickers)}
    ticker_set = {str(t).upper().strip() for t in bucket_tickers}
    calc_fn = getattr(helpers, 'calc_top10_accumulated_10d_rebalance', None)
    if calc_fn is None:
        print("  Helper module missing calc_top10_accumulated_10d_rebalance; reverting to per-bucket retrain.")
        return _run_bucket_with_training(bucket_idx, bucket_tickers, output_dir, n_buckets, features_list)

    def _safe_corr(a: pd.Series, b: pd.Series, method: str = 'pearson') -> Optional[float]:
        try:
            value = a.corr(b, method=method)
            if value is None or np.isnan(value):
                return None
            return float(value)
        except Exception:
            return None

    for model_name, df in global_predictions.items():
        filtered = df[df['ticker'].isin(ticker_set)].copy()
        if filtered.empty:
            print(f"  {model_name}: no overlapping predictions for this bucket")
            continue
        filtered.sort_values('date', inplace=True)
        ts = calc_fn(
            predictions_df=filtered[['date', 'ticker', 'prediction', 'actual']],
            top_n=15,
            step=10,
            out_dir=None,
            model_name=model_name,
            bucket_range=(5, 15),
        )
        if ts is None or ts.empty:
            print(f"  {model_name}: insufficient data for Top5-15 rebalance results")
            continue
        ts.to_csv(run_dir / f"{model_name}_top5_15_rebalance10d_accumulated.csv", index=False)
        filtered.to_csv(run_dir / f"{model_name}_bucket_predictions.csv", index=False)

        final_return = float(ts.iloc[-1]['acc_return'])
        max_dd = float(ts['drawdown'].min())

        if model_name == 'lambdarank':
            metrics['LR_Return'] = final_return
            metrics['LR_MaxDD'] = max_dd
            ic = _safe_corr(filtered['prediction'], filtered['actual'], method='pearson')
            rank_ic = _safe_corr(filtered['prediction'], filtered['actual'], method='spearman')
            if ic is not None:
                metrics['IC'] = ic
            if rank_ic is not None:
                metrics['Rank_IC'] = rank_ic
            print(f"  LambdaRank (global): {final_return*100:.1f}% return, {max_dd:.1f}% maxDD")
        elif model_name == 'ridge_stacking':
            metrics['RS_Return'] = final_return
            metrics['RS_MaxDD'] = max_dd
            print(f"  Ridge Stacking (global): {final_return*100:.1f}% return, {max_dd:.1f}% maxDD")

    oos_subset = {k: metrics[k] for k in ('IC', 'Rank_IC') if k in metrics}
    if oos_subset:
        (run_dir / "oos_metrics.json").write_text(json.dumps(oos_subset, indent=2))

    return metrics


def _run_bucket_with_training(
    bucket_idx: int,
    bucket_tickers: list,
    output_dir: Path,
    n_buckets: int,
    features_list: list,
) -> dict:
    """Run one bucket: 80/20 time split retrain and OOS test."""
    print(f"\n{'='*70}")
    print(f"BUCKET {bucket_idx + 1}/{n_buckets}")
    print(f"{'='*70}")
    print(f"Tickers in this bucket: {len(bucket_tickers)}")

    bucket_dir = output_dir / f"bucket_{bucket_idx + 1}"
    bucket_dir.mkdir(parents=True, exist_ok=True)

    (bucket_dir / "bucket_tickers.txt").write_text("\n".join(bucket_tickers))

    bucket_data_file = create_bucket_data(DATA_FILE, bucket_tickers, bucket_dir)

    cmd = [
        sys.executable, str(TRADE_DIR / "scripts" / "time_split_80_20_oos_eval.py"),
        "--train-data", bucket_data_file,
        "--data-file", bucket_data_file,
        "--features",
    ] + features_list + [
        "--horizon-days", "10",
        "--split", "0.8",
        "--models", "lambdarank", "ridge_stacking",
        "--top-n", "20",
        "--ema-top-n", "300",
        "--ema-length", "2",
        "--ema-beta", "0.7",
        "--ema-min-days", "2",
        "--output-dir", str(bucket_dir),
        "--log-level", "WARNING"
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(TRADE_DIR)

    print(f"Running 80/20 time split (retrain + OOS test)...")
    result = subprocess.run(cmd, env=env, cwd=str(TRADE_DIR), capture_output=True, text=True)

    metrics = {'bucket': bucket_idx + 1, 'n_tickers': len(bucket_tickers)}

    for run_dir in bucket_dir.glob("run_*"):
        mf = run_dir / "oos_metrics.json"
        if mf.exists():
            with open(mf) as f:
                m = json.load(f)
                metrics['IC'] = m.get('IC')
                metrics['Rank_IC'] = m.get('Rank_IC')

        lr_file = run_dir / "lambdarank_top5_15_rebalance10d_accumulated.csv"
        rs_file = run_dir / "ridge_stacking_top5_15_rebalance10d_accumulated.csv"

        if lr_file.exists():
            lr_df = pd.read_csv(lr_file)
            metrics['LR_Return'] = lr_df.iloc[-1]['acc_return']
            metrics['LR_MaxDD'] = lr_df['drawdown'].min()
            print(f"  LambdaRank: {metrics['LR_Return']*100:.1f}% return, {metrics['LR_MaxDD']:.1f}% maxDD")

        if rs_file.exists():
            rs_df = pd.read_csv(rs_file)
            metrics['RS_Return'] = rs_df.iloc[-1]['acc_return']
            metrics['RS_MaxDD'] = rs_df['drawdown'].min()
            print(f"  Ridge Stacking: {metrics['RS_Return']*100:.1f}% return, {metrics['RS_MaxDD']:.1f}% maxDD")

    if result.returncode != 0 and 'LR_Return' not in metrics:
        print(f"  Error: {result.stderr[-300:] if result.stderr else 'Unknown'}")

    return metrics


def main():
    import argparse
    ap = argparse.ArgumentParser(description="N-bucket ticker CV with 80/20 time split (retrain + OOS per bucket)")
    ap.add_argument("--n-buckets", type=int, default=3, help="Number of ticker buckets (e.g. 5 for 5-fold)")
    ap.add_argument("--use-9-features", action="store_true", help="Use old default 9 features (TOP_9_FEATURES) instead of 11")
    ap.add_argument("--global-train-once", action="store_true", 
                    help="Train once on all available tickers and reuse predictions across buckets")
    ap.add_argument("--force-bucket-retrain", action="store_true", 
                    help="Force per-bucket retraining even if global mode is preferred")
    args = ap.parse_args()

    n_buckets = args.n_buckets
    use_9 = getattr(args, "use_9_features", False)
    features_list = FEATURES_9 if use_9 else FEATURES_11
    prefer_global = not use_9
    if getattr(args, "global_train_once", False):
        prefer_global = True
    if getattr(args, "force_bucket_retrain", False):
        prefer_global = False
    if n_buckets == 5:
        output_base = TRADE_DIR / "results" / "5fold_ticker_cv"
    else:
        output_base = TRADE_DIR / "results" / "3bucket_ticker_cv"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print(f"{n_buckets}-BUCKET TICKER CV WITH 80/20 TIME SPLIT (RETRAIN + OOS)")
    print("="*70)
    print(f"Data: {DATA_FILE}")
    print(f"Features: {len(features_list)} factors {'(old 9)' if use_9 else '(default 11)'}")
    print(f"Buckets: {n_buckets}")
    print(f"Output: {output_dir}")
    print(f"EMA: L2 B0.7 with Top300 Gate (min_days=2)")
    print("="*70)

    global_predictions: Optional[Dict[str, pd.DataFrame]] = None
    global_run_dir: Optional[Path] = None
    if prefer_global:
        global_run_dir = run_global_full_training(features_list, output_dir)
        if global_run_dir:
            global_predictions = load_global_prediction_frames(global_run_dir)
            if not global_predictions:
                print("  Warning: global retrain did not produce prediction frames; defaulting to per-bucket retrain.")
                global_run_dir = None
        else:
            print("  Warning: unable to execute global retrain; continuing with per-bucket retrain.")
    mode_label = "global retrain once + reuse predictions" if global_predictions else "per-bucket retrain"
    print(f"Mode: {mode_label}")


    all_tickers = get_all_tickers(DATA_FILE)

    buckets = split_tickers_into_buckets(all_tickers, n_buckets, BASE_SEED)
    print(f"\nTicker distribution per bucket:")
    for i, b in enumerate(buckets):
        print(f"  Bucket {i+1}: {len(b)} tickers")

    bucket_assignment = {f"bucket_{i+1}": buckets[i] for i in range(n_buckets)}
    (output_dir / "bucket_assignments.json").write_text(json.dumps(bucket_assignment, indent=2))

    all_results = []
    for bucket_idx in range(n_buckets):
        bucket_tickers = buckets[bucket_idx]
        metrics = run_bucket(
            bucket_idx, bucket_tickers, output_dir, n_buckets, features_list, global_predictions=global_predictions
        )
        all_results.append(metrics)

    print("\n" + "="*70)
    print(f"{n_buckets}-BUCKET CV SUMMARY")
    print("="*70)

    df = pd.DataFrame(all_results)

    print("\nPer-Bucket Results:")
    print("-"*70)
    print(f"{'Bucket':>6} {'Tickers':>8} {'LR_Return':>12} {'LR_MaxDD':>10} {'RS_Return':>12} {'RS_MaxDD':>10}")
    print("-"*70)
    for _, row in df.iterrows():
        lr_ret = f"{row.get('LR_Return', 0)*100:.1f}%" if pd.notna(row.get('LR_Return')) else "N/A"
        lr_dd = f"{row.get('LR_MaxDD', 0):.1f}%" if pd.notna(row.get('LR_MaxDD')) else "N/A"
        rs_ret = f"{row.get('RS_Return', 0)*100:.1f}%" if pd.notna(row.get('RS_Return')) else "N/A"
        rs_dd = f"{row.get('RS_MaxDD', 0):.1f}%" if pd.notna(row.get('RS_MaxDD')) else "N/A"
        print(f"{row['bucket']:>6} {row['n_tickers']:>8} {lr_ret:>12} {lr_dd:>10} {rs_ret:>12} {rs_dd:>10}")

    print("\nAggregate Statistics:")
    print("-"*70)
    for col, label in [('LR_Return', 'LambdaRank Return'), ('RS_Return', 'Ridge Stacking Return'),
                       ('LR_MaxDD', 'LambdaRank MaxDD'), ('RS_MaxDD', 'Ridge Stacking MaxDD')]:
        if col in df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"{label:25} Mean: {values.mean()*100:7.2f}%  Std: {values.std()*100:6.2f}%  Min: {values.min()*100:7.2f}%  Max: {values.max()*100:7.2f}%")

    results_csv = output_dir / f"{n_buckets}bucket_cv_results.csv"
    df.to_csv(results_csv, index=False)

    config = {
        "data_file": DATA_FILE,
        "features": features_list,
        "n_factors": len(features_list),
        "use_9_features": use_9,
        "n_buckets": n_buckets,
        "random_seed": BASE_SEED,
        "total_tickers": len(all_tickers),
        "ema_config": {"length": 2, "beta": 0.7, "top_n": 300, "min_days": 2},
        "global_training_mode": "global_full_run" if global_predictions else "per_bucket",
        "global_run_dir": str(global_run_dir) if global_run_dir else None,
        "timestamp": timestamp
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\nResults saved to: {output_dir}")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
