#!/usr/bin/env python3
"""LambdaRank grid search for T+5 micro factors (random 1000 tickers).

This script:
  * Samples 1,000 tickers (deterministic seed) from the T5 micro parquet.
  * Runs a 15-combination LightGBM LambdaRank grid (5 base configs × 3 ranking shapes).
  * Launches up to 5 runs in parallel, each invoking time_split_80_20_oos_eval.py with
    --lambdarank-only and --features overrides for the 10-factor micro set.
  * Uses temporary unified_config overrides per run via BMA_TEMP_CONFIG_PATH so runs do
    not interfere with one another.
  * Extracts IC/top-bucket metrics from results_summary_for_word_doc.json and computes
    daily mean NDCG@10/@20 from lambdarank_predictions_diagnosis.csv.
  * Writes an aggregate CSV summarizing all runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path('bma_models/unified_config.yaml')
DEFAULT_FULL_DATA = Path('data/factor_exports/polygon_factors_all_2021_2026_CLEAN_STANDARDIZED_T5_MICRO.parquet')
DEFAULT_SUBSET_DATA = Path('data/factor_exports/polygon_factors_t5_micro_random1000.parquet')
DEFAULT_OUTPUT_ROOT = Path('results/t5_micro_lambdarank_grid')
FEATURE_COLUMNS = [
    'volume_price_corr_10d',
    'rsi_14',
    'reversal_3d',
    'momentum_10d',
    'liquid_momentum_10d',
    'sharpe_momentum_5d',
    'price_ma20_deviation',
    'avg_trade_size',
    'trend_r2_20',
    'obv_divergence',
    'alpha_linreg_corr_10d',
]

BASE_CAPACITY = {
    'B1': {
        'learning_rate': 0.05,
        'num_leaves': 15,
        'max_depth': 4,
        'min_data_in_leaf': 200,
        'lambda_l2': 50.0,
        'min_gain_to_split': 0.1,
        'feature_fraction': 1.0,
        'bagging_fraction': 0.70,
        'bagging_freq': 1,
    },
    'B2': {
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 150,
        'lambda_l2': 30.0,
        'min_gain_to_split': 0.05,
        'feature_fraction': 0.90,
        'bagging_fraction': 0.75,
        'bagging_freq': 3,
    },
    'B3': {
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': -1,
        'min_data_in_leaf': 80,
        'lambda_l2': 10.0,
        'min_gain_to_split': 0.0,
        'feature_fraction': 0.90,
        'bagging_fraction': 0.80,
        'bagging_freq': 3,
    },
    'B4': {
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 8,
        'min_data_in_leaf': 50,
        'lambda_l2': 20.0,
        'min_gain_to_split': 0.0,
        'feature_fraction': 0.80,
        'bagging_fraction': 0.85,
        'bagging_freq': 2,
    },
    'B5': {
        'learning_rate': 0.02,
        'num_leaves': 63,
        'max_depth': -1,
        'min_data_in_leaf': 30,
        'lambda_l2': 5.0,
        'min_gain_to_split': 0.0,
        'feature_fraction': 0.80,
        'bagging_fraction': 0.90,
        'bagging_freq': 1,
    },
}

SHAPE_VARIANTS = {
    'V1': {
        'lambdarank_truncation_level': 20,
        'sigmoid': 0.9,
        'label_gain_power': 2.0,
    },
    'V2': {
        'lambdarank_truncation_level': 40,
        'sigmoid': 1.1,
        'label_gain_power': 2.1,
    },
    'V3': {
        'lambdarank_truncation_level': 80,
        'sigmoid': 1.3,
        'label_gain_power': 2.3,
    },
}


def sample_random_subset(source: Path, target: Path, n_tickers: int, seed: int) -> List[str]:
    """Sample n_tickers uniformly from the MultiIndex file (date, ticker)."""
    if target.exists():
        df = pd.read_parquet(target)
        if isinstance(df.index, pd.MultiIndex):
            return sorted(df.index.get_level_values('ticker').unique().tolist())
        raise ValueError(f"Existing subset at {target} is not MultiIndex")

    df = pd.read_parquet(source)
    if not isinstance(df.index, pd.MultiIndex):
        if {'date', 'ticker'}.issubset(df.columns):
            df = df.set_index(['date', 'ticker'])
        else:
            raise ValueError('Input parquet must contain MultiIndex or date/ticker columns')

    tickers = df.index.get_level_values('ticker').unique()
    rng = np.random.default_rng(seed)
    if len(tickers) <= n_tickers:
        sampled = tickers
    else:
        sampled = np.sort(rng.choice(tickers, size=n_tickers, replace=False))
    subset = df.loc[(slice(None), sampled), :]
    target.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(target)
    return sampled.tolist()


def build_combo_params(base_name: str, shape_name: str) -> Dict[str, float]:
    params = {
        'metric': 'ndcg',
        'ndcg_eval_at': [10, 20],
        'first_metric_only': True,
        'objective': 'lambdarank',
        'num_boost_round': 1200,
        'lambda_l1': 0.0,
        'n_quantiles': 32,
    }
    params.update(BASE_CAPACITY[base_name])
    params.update(SHAPE_VARIANTS[shape_name])
    return params


def write_temp_config(config_path: Path, overrides: Dict[str, float]) -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    lambdarank_cfg = cfg.setdefault('training', {}).setdefault('base_models', {}).setdefault('lambdarank', {})
    lambdarank_cfg['enable'] = True
    for key, value in overrides.items():
        lambdarank_cfg[key] = value
    lambdarank_cfg.setdefault('fit_params', {})['early_stopping_rounds'] = 100
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')


def run_time_split(combo_id: str, combo_dir: Path, subset_path: Path, feature_list: List[str], env_config: Path,
                   extra_args: List[str]) -> Path:
    if combo_dir.exists():
        for existing in combo_dir.glob('run_*'):
            shutil.rmtree(existing, ignore_errors=True)
    combo_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        'scripts/time_split_80_20_oos_eval.py',
        '--data-file', str(subset_path),
        '--output-dir', str(combo_dir),
        '--models', 'lambdarank',
        '--lambdarank-only',
        '--features', *feature_list,
    ] + extra_args

    env = os.environ.copy()
    env['BMA_TEMP_CONFIG_PATH'] = str(env_config)
    repo_root = str(REPO_ROOT)
    env['PYTHONPATH'] = repo_root + os.pathsep + env.get('PYTHONPATH', '')

    start = time.time()
    subprocess.run(cmd, check=True, env=env)
    elapsed = time.time() - start
    print(f"[{combo_id}] finished in {elapsed/60:.2f} minutes")

    run_dirs = sorted(combo_dir.glob('run_*'), key=lambda p: p.stat().st_mtime)
    if not run_dirs:
        raise RuntimeError(f"No run_* directory created under {combo_dir}")
    return run_dirs[-1]


def extract_summary(run_dir: Path) -> Dict[str, float]:
    summary_path = run_dir / 'results_summary_for_word_doc.json'
    if not summary_path.exists():
        raise FileNotFoundError(f'Missing summary file: {summary_path}')
    payload = json.loads(summary_path.read_text(encoding='utf-8'))
    lambdarank = payload.get('lambdarank', {})
    metrics = lambdarank.get('metrics', {})
    returns = lambdarank.get('returns', {})
    bucket = lambdarank.get('bucket_summary', {})
    return {
        'ic': metrics.get('IC'),
        'ic_pvalue': metrics.get('IC_pvalue'),
        'rank_ic': metrics.get('Rank_IC'),
        'rank_ic_pvalue': metrics.get('Rank_IC_pvalue'),
        'avg_top_bucket_t10': returns.get('avg_top_bucket_t10_return'),
        'median_top_bucket_t10': returns.get('median_top_bucket_t10_return'),
        'avg_top_return': returns.get('avg_top_return'),
        'avg_bottom_return': returns.get('avg_bottom_return'),
        'top_sharpe_net': returns.get('top_sharpe_net'),
        'bucket_top_1_10': bucket.get('avg_top_1_10_return'),
        'bucket_top_11_20': bucket.get('avg_top_11_20_return'),
    }


def ndcg_at_k(pred_df: pd.DataFrame, k: int) -> float:
    scores = pred_df['prediction'].to_numpy()
    actual = pred_df['actual'].to_numpy()
    if len(scores) == 0 or len(actual) == 0:
        return float('nan')
    ranks = actual.argsort()
    relevance = np.zeros_like(actual, dtype=float)
    relevance[ranks] = np.arange(len(actual), dtype=float)
    order = scores.argsort()[::-1]
    k_eff = min(k, len(actual))
    denom = np.log2(np.arange(2, k_eff + 2))
    dcg = np.sum(relevance[order[:k_eff]] / denom)
    ideal = np.sum(np.sort(relevance)[::-1][:k_eff] / denom)
    return float(dcg / ideal) if ideal > 0 else float('nan')


def compute_mean_ndcg(run_dir: Path) -> Dict[str, float]:
    pred_path = run_dir / 'lambdarank_predictions_diagnosis.csv'
    if not pred_path.exists():
        raise FileNotFoundError(f'Missing predictions file: {pred_path}')
    df = pd.read_csv(pred_path, parse_dates=['date'])
    grouped = df.groupby('date')
    ndcg10: List[float] = []
    ndcg20: List[float] = []
    for _, group in grouped:
        nd10 = ndcg_at_k(group, 10)
        if math.isfinite(nd10):
            ndcg10.append(nd10)
        nd20 = ndcg_at_k(group, 20)
        if math.isfinite(nd20):
            ndcg20.append(nd20)
    return {
        'ndcg_at_10': float(np.mean(ndcg10)) if ndcg10 else float('nan'),
        'ndcg_at_20': float(np.mean(ndcg20)) if ndcg20 else float('nan'),
    }


@dataclass
class ComboResult:
    combo_id: str
    run_dir: Path
    metrics: Dict[str, float]


def run_combo(combo_id: str, base_name: str, shape_name: str, subset_path: Path,
              output_root: Path, extra_args: List[str]) -> ComboResult:
    combo_dir = output_root / combo_id
    combo_dir.mkdir(parents=True, exist_ok=True)
    config_path = combo_dir / f'{combo_id}_config.yaml'
    params = build_combo_params(base_name, shape_name)
    write_temp_config(config_path, params)
    run_dir = run_time_split(combo_id, combo_dir, subset_path, FEATURE_COLUMNS, config_path, extra_args)
    summary = extract_summary(run_dir)
    summary.update(compute_mean_ndcg(run_dir))
    summary.update({
        'base_model': base_name,
        'shape_variant': shape_name,
        'combo_id': combo_id,
    })
    summary.update(params)
    return ComboResult(combo_id=combo_id, run_dir=run_dir, metrics=summary)


def main() -> None:
    parser = argparse.ArgumentParser(description='T5 micro LambdaRank grid search (1000 tickers)')
    parser.add_argument('--full-data', type=Path, default=DEFAULT_FULL_DATA)
    parser.add_argument('--subset-data', type=Path, default=DEFAULT_SUBSET_DATA)
    parser.add_argument('--num-tickers', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--summary-csv', type=Path, default=Path('results/t5_micro_lambdarank_grid_summary.csv'))
    parser.add_argument('--max-jobs', type=int, default=5)
    parser.add_argument('--extra-args', nargs=argparse.REMAINDER, default=[],
                        help='Additional args forwarded to time_split_80_20_oos_eval.py')
    args = parser.parse_args()

    sampled = sample_random_subset(args.full_data, args.subset_data, args.num_tickers, args.seed)
    print(f"Subset ready: {len(sampled)} tickers -> {args.subset_data}")

    combos: List[Tuple[str, str, str]] = []
    for base_name in BASE_CAPACITY:
        for shape_name in SHAPE_VARIANTS:
            combos.append((f'{base_name}_{shape_name}', base_name, shape_name))

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    futures = {}
    extra_args = list(args.extra_args)
    with ThreadPoolExecutor(max_workers=max(1, args.max_jobs)) as executor:
        for combo_id, base_name, shape_name in combos:
            futures[executor.submit(run_combo, combo_id, base_name, shape_name,
                                    args.subset_data, args.output_root, extra_args)] = combo_id
        for future in as_completed(futures):
            combo_id = futures[future]
            try:
                result = future.result()
                rows.append(result.metrics)
                pd.DataFrame(rows).to_csv(args.summary_csv, index=False)
                print(f"[{combo_id}] metrics recorded -> {args.summary_csv}")
            except Exception as exc:
                print(f"[{combo_id}] FAILED: {exc}")

    print('Grid search complete. Summary saved to', args.summary_csv)


if __name__ == '__main__':
    main()
