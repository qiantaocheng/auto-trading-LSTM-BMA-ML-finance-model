#!/usr/bin/env python3
"""Run LambdaRank-only grid search across curated feature sets."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

DEFAULT_DATA_FILE = Path('data/factor_exports/polygon_factors_all_2021_2026_T5_final.parquet')
DEFAULT_OUTPUT_BASE = Path('results/feature_grid_lambdarank')
DEFAULT_PIPELINE = Path('scripts/lambdarank_only_pipeline.py')
DEFAULT_RUN_NAMES = ['run9_plus_atr_pct_14_plus_amihud_20']
DEFAULT_SEEDS = [5]

FEATURE_RUNS = [
    {
        "run_id": 1,
        "name": "run2_base_12f",
        "features": [
            "volume_price_corr_3d","rsi_14","reversal_3d","momentum_10d","liquid_momentum_10d",
            "sharpe_momentum_5d","price_ma20_deviation","avg_trade_size","trend_r2_20",
            "dollar_vol_20","ret_skew_20d","reversal_5d"
        ]
    },
    {
        "run_id": 2,
        "name": "run9_base_plus_near_52w_high",
        "features": [
            "volume_price_corr_3d","rsi_14","reversal_3d","momentum_10d","liquid_momentum_10d",
            "sharpe_momentum_5d","price_ma20_deviation","avg_trade_size","trend_r2_20",
            "dollar_vol_20","ret_skew_20d","reversal_5d","near_52w_high"
        ]
    },
    {
        "run_id": 3,
        "name": "run9_plus_atr_pct_14",
        "features": [
            "volume_price_corr_3d","rsi_14","reversal_3d","momentum_10d","liquid_momentum_10d",
            "sharpe_momentum_5d","price_ma20_deviation","avg_trade_size","trend_r2_20",
            "dollar_vol_20","ret_skew_20d","reversal_5d","near_52w_high","atr_pct_14"
        ]
    },
    {
        "run_id": 4,
        "name": "run9_plus_hist_vol_20",
        "features": [
            "volume_price_corr_3d","rsi_14","reversal_3d","momentum_10d","liquid_momentum_10d",
            "sharpe_momentum_5d","price_ma20_deviation","avg_trade_size","trend_r2_20",
            "dollar_vol_20","ret_skew_20d","reversal_5d","near_52w_high","hist_vol_20"
        ]
    },
    {
        "run_id": 5,
        "name": "run9_plus_ivol_20",
        "features": [
            "volume_price_corr_3d","rsi_14","reversal_3d","momentum_10d","liquid_momentum_10d",
            "sharpe_momentum_5d","price_ma20_deviation","avg_trade_size","trend_r2_20",
            "dollar_vol_20","ret_skew_20d","reversal_5d","near_52w_high","ivol_20"
        ]
    },
    {
        "run_id": 6,
        "name": "run9_plus_amihud_20",
        "features": [
            "volume_price_corr_3d","rsi_14","reversal_3d","momentum_10d","liquid_momentum_10d",
            "sharpe_momentum_5d","price_ma20_deviation","avg_trade_size","trend_r2_20",
            "dollar_vol_20","ret_skew_20d","reversal_5d","near_52w_high","amihud_20"
        ]
    },
    {
        "run_id": 7,
        "name": "run9_plus_vol_ratio_20_60",
        "features": [
            "volume_price_corr_3d","rsi_14","reversal_3d","momentum_10d","liquid_momentum_10d",
            "sharpe_momentum_5d","price_ma20_deviation","avg_trade_size","trend_r2_20",
            "dollar_vol_20","ret_skew_20d","reversal_5d","near_52w_high","vol_ratio_20_60"
        ]
    },
    {
        "run_id": 8,
        "name": "run9_plus_obv_divergence_20d",
        "features": [
            "volume_price_corr_3d","rsi_14","reversal_3d","momentum_10d","liquid_momentum_10d",
            "sharpe_momentum_5d","price_ma20_deviation","avg_trade_size","trend_r2_20",
            "dollar_vol_20","ret_skew_20d","reversal_5d","near_52w_high","obv_divergence_20d"
        ]
    },
    {
        "run_id": 9,
        "name": "run9_plus_volume_price_corr_10d",
        "features": [
            "volume_price_corr_3d","rsi_14","reversal_3d","momentum_10d","liquid_momentum_10d",
            "sharpe_momentum_5d","price_ma20_deviation","avg_trade_size","trend_r2_20",
            "dollar_vol_20","ret_skew_20d","reversal_5d","near_52w_high","volume_price_corr_10d"
        ]
    },
    {
        "run_id": 10,
        "name": "run9_plus_atr_pct_14_plus_amihud_20",
        "features": [
            "volume_price_corr_3d","rsi_14","reversal_3d","momentum_10d","liquid_momentum_10d",
            "sharpe_momentum_5d","price_ma20_deviation","avg_trade_size","trend_r2_20",
            "dollar_vol_20","ret_skew_20d","reversal_5d","near_52w_high","atr_pct_14","amihud_20"
        ]
    },
    {
        "run_id": 11,
        "name": "run9_plus_hist_vol_20_plus_vol_ratio_20_60",
        "features": [
            "volume_price_corr_3d","rsi_14","reversal_3d","momentum_10d","liquid_momentum_10d",
            "sharpe_momentum_5d","price_ma20_deviation","avg_trade_size","trend_r2_20",
            "dollar_vol_20","ret_skew_20d","reversal_5d","near_52w_high","hist_vol_20","vol_ratio_20_60"
        ]
    },
    {
        "run_id": 12,
        "name": "run9_plus_obv_divergence_20d_plus_volume_price_corr_10d",
        "features": [
            "volume_price_corr_3d","rsi_14","reversal_3d","momentum_10d","liquid_momentum_10d",
            "sharpe_momentum_5d","price_ma20_deviation","avg_trade_size","trend_r2_20",
            "dollar_vol_20","ret_skew_20d","reversal_5d","near_52w_high","obv_divergence_20d","volume_price_corr_10d"
        ]
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='LambdaRank feature grid search runner')
    parser.add_argument('--data-file', type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument('--time-fraction', type=float, default=0.5)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--horizon-days', type=int, default=5)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--n-boost-round', type=int, default=800)
    parser.add_argument('--max-parallel', type=int, default=3)
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help='Random seeds to sweep for each config (defaults to best-known seed)')
    parser.add_argument('--params-json', type=str, default=None,
                        help='Optional JSON file or string overriding LambdaRank params')
    parser.add_argument('--pipeline', type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument('--output-base', type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument('--runs', nargs='*', default=None,
                        help='Optional subset of run IDs or names; use "all" to sweep every run')
    return parser.parse_args()


def select_runs(requested: List[str] | None) -> List[Dict]:
    if not requested:
        requested = DEFAULT_RUN_NAMES
    elif len(requested) == 1 and requested[0].lower() in {'all', '*', 'full'}:
        requested = [spec['name'] for spec in FEATURE_RUNS]
    selected: List[Dict] = []
    request_lower = {str(r).lower() for r in requested}
    for spec in FEATURE_RUNS:
        if str(spec['run_id']) in request_lower or spec['name'].lower() in request_lower:
            selected.append(spec)
    if not selected:
        raise ValueError('No matching runs for selection: ' + ', '.join(requested))
    return selected


def run_single(spec: Dict, seed: int, args: argparse.Namespace, output_root: Path) -> Dict:
    run_dir = output_root / f'seed{seed:02d}' / f"{spec['run_id']:02d}_{spec['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(args.pipeline),
        '--data-file', str(args.data_file),
        '--time-fraction', str(args.time_fraction),
        '--split', str(args.split),
        '--horizon-days', str(args.horizon_days),
        '--cv-splits', str(args.cv_splits),
        '--n-boost-round', str(args.n_boost_round),
        '--seed', str(seed),
        '--output-dir', str(run_dir),
    ]
    if args.params_json:
        cmd.extend(['--params-json', args.params_json])
    if spec['features']:
        cmd.extend(['--features', *spec['features']])
    print(f"Launching run {spec['run_id']} ({spec['name']}) seed {seed}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    metrics_path = run_dir / 'metrics.json'
    if not metrics_path.exists():
        raise FileNotFoundError(f'Metrics missing for run {spec['name']} (seed {seed})')
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    metrics.update({
        'run_id': spec['run_id'],
        'run_name': spec['name'],
        'seed': seed,
        'features': spec['features'],
        'run_dir': str(run_dir),
    })
    return metrics


def summarize(results: List[Dict], output_root: Path) -> None:
    if not results:
        print('No successful runs to summarize.')
        return
    import pandas as pd
    csv_path = output_root / 'feature_grid_results.csv'
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    keep_cols = [
        'run_id', 'run_name', 'seed', 'top_1_10_mean', 'top_5_15_mean', 'top_10_20_mean',
        'spread', 'NDCG_10', 'NDCG_20', 'IC_mean'
    ]
    cols = [c for c in keep_cols if c in df.columns]
    df_sorted = df.sort_values(['NDCG_10', 'spread'], ascending=[False, False]) if 'NDCG_10' in df else df
    print('\n===== FEATURE GRID SUMMARY =====')
    print(df_sorted[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if not df_sorted.empty:
        best = df_sorted.iloc[0]
        print('\nBEST RUN:')
        print(best[['run_id', 'run_name', 'NDCG_10', 'NDCG_20', 'top_1_10_mean', 'spread']])
    print(f'\nSummary saved to {csv_path}')


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = args.output_base / f'run_{timestamp}'
    output_root.mkdir(parents=True, exist_ok=True)
    runs = select_runs(args.runs)
    run_names = [spec["name"] for spec in runs]
    print('===== FEATURE GRID LAMBDARANK =====')
    print(f'Data file: {args.data_file}')
    if not args.runs:
        print(f'Runs defaulted to {run_names} (use --runs all for full sweep)')
    else:
        print(f'Runs: {run_names}')
    print(f'Seeds: {args.seeds}')
    print(f'Output root: {output_root}')
    results: List[Dict] = []
    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as executor:
        for seed in args.seeds:
            for spec in runs:
                futures[executor.submit(run_single, spec, seed, args, output_root)] = (spec, seed)
        for future in as_completed(futures):
            spec, seed = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"Run {spec['name']} (seed {seed}) failed: {exc}")
    summarize(results, output_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
