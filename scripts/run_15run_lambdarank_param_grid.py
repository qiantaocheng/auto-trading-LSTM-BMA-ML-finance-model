#!/usr/bin/env python3
"""15-run LambdaRank grid search (Lambda-only pipeline, next_grid_15 specs)."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

DATA_FILE = Path('data/factor_exports/polygon_factors_all_2021_2026_T5_final.parquet')
OUTPUT_BASE = Path('results/15run_lambdarank_param_grid')
LAMBDA_PIPELINE = Path('scripts/lambdarank_only_pipeline.py')

FEATURES = [
    'volume_price_corr_10d',
    'rsi_14',
    'reversal_3d',
    'momentum_10d',
    'liquid_momentum_10d',
    'sharpe_momentum_5d',
    'price_ma20_deviation',
    'avg_trade_size',
    'trend_r2_20',
]

NEXT_GRID = [
    {"id": "A11", "style": "smoother", "learning_rate": 0.05, "num_leaves": 39, "max_depth": 6,  "min_data_in_leaf": 150, "lambda_l2": 30, "lambdarank_truncation_level": 80, "sigmoid": 1.30, "label_gain_power": 2.30},
    {"id": "A12", "style": "smoother", "learning_rate": 0.05, "num_leaves": 47, "max_depth": 6,  "min_data_in_leaf": 150, "lambda_l2": 30, "lambdarank_truncation_level": 80, "sigmoid": 1.30, "label_gain_power": 2.30},
    {"id": "A13", "style": "smoother", "learning_rate": 0.05, "num_leaves": 39, "max_depth": 7,  "min_data_in_leaf": 150, "lambda_l2": 30, "lambdarank_truncation_level": 80, "sigmoid": 1.30, "label_gain_power": 2.30},
    {"id": "A14", "style": "smoother", "learning_rate": 0.04, "num_leaves": 39, "max_depth": 6,  "min_data_in_leaf": 150, "lambda_l2": 30, "lambdarank_truncation_level": 80, "sigmoid": 1.25, "label_gain_power": 2.30},
    {"id": "A15", "style": "smoother", "learning_rate": 0.05, "num_leaves": 39, "max_depth": 6,  "min_data_in_leaf": 120, "lambda_l2": 30, "lambdarank_truncation_level": 80, "sigmoid": 1.30, "label_gain_power": 2.30},
    {"id": "A16", "style": "smoother", "learning_rate": 0.05, "num_leaves": 39, "max_depth": 6,  "min_data_in_leaf": 180, "lambda_l2": 25, "lambdarank_truncation_level": 80, "sigmoid": 1.30, "label_gain_power": 2.30},
    {"id": "A17", "style": "smoother", "learning_rate": 0.05, "num_leaves": 39, "max_depth": 6,  "min_data_in_leaf": 200, "lambda_l2": 30, "lambdarank_truncation_level": 80, "sigmoid": 1.35, "label_gain_power": 2.40},
    {"id": "A18", "style": "smoother", "learning_rate": 0.05, "num_leaves": 39, "max_depth": 6,  "min_data_in_leaf": 150, "lambda_l2": 40, "lambdarank_truncation_level": 80, "sigmoid": 1.25, "label_gain_power": 2.20},
    {"id": "B11", "style": "sniper",   "learning_rate": 0.04, "num_leaves": 15, "max_depth": 4,  "min_data_in_leaf": 250, "lambda_l2": 45, "lambdarank_truncation_level": 25, "sigmoid": 1.10, "label_gain_power": 2.10},
    {"id": "B12", "style": "sniper",   "learning_rate": 0.05, "num_leaves": 15, "max_depth": 4,  "min_data_in_leaf": 250, "lambda_l2": 45, "lambdarank_truncation_level": 25, "sigmoid": 1.10, "label_gain_power": 2.10},
    {"id": "B13", "style": "sniper",   "learning_rate": 0.04, "num_leaves": 15, "max_depth": 4,  "min_data_in_leaf": 300, "lambda_l2": 50, "lambdarank_truncation_level": 25, "sigmoid": 1.10, "label_gain_power": 2.10},
    {"id": "B14", "style": "sniper",   "learning_rate": 0.04, "num_leaves": 23, "max_depth": 5,  "min_data_in_leaf": 250, "lambda_l2": 40, "lambdarank_truncation_level": 25, "sigmoid": 1.15, "label_gain_power": 2.00},
    {"id": "C01", "style": "bridge",   "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,  "min_data_in_leaf": 180, "lambda_l2": 35, "lambdarank_truncation_level": 40, "sigmoid": 1.20, "label_gain_power": 2.20}
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='LambdaRank-only parameter grid search (next_grid_15)')
    parser.add_argument('--data-file', type=Path, default=DATA_FILE)
    parser.add_argument('--time-fraction', type=float, default=0.5)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--horizon-days', type=int, default=5)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--n-boost-round', type=int, default=800)
    parser.add_argument('--max-parallel', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                        help='Random seeds to sweep for each config')
    parser.add_argument('--single-run', type=str, default=None)
    parser.add_argument('--features', nargs='+', default=FEATURES)
    return parser.parse_args()


def build_configs() -> Dict[str, Dict[str, Any]]:
    configs: Dict[str, Dict[str, Any]] = {}
    for spec in NEXT_GRID:
        params = spec.copy()
        run_id = params.pop('id')
        configs[run_id] = params
    return configs


def run_single_config(run_name: str, params: Dict[str, Any], args: argparse.Namespace,
                      output_root: Path, seed: int) -> Dict[str, Any]:
    run_dir = output_root / f'seed{seed:02d}' / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    params_path = run_dir / 'params.json'
    params_path.write_text(json.dumps(params, indent=2), encoding='utf-8')
    cmd = [
        sys.executable,
        str(LAMBDA_PIPELINE),
        '--data-file', str(args.data_file),
        '--time-fraction', str(args.time_fraction),
        '--split', str(args.split),
        '--horizon-days', str(args.horizon_days),
        '--cv-splits', str(args.cv_splits),
        '--n-boost-round', str(args.n_boost_round),
        '--seed', str(seed),
        '--output-dir', str(run_dir),
        '--params-json', str(params_path),
        '--features', *args.features,
    ]
    print(f"Launching {run_name} (seed {seed}): {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    metrics_path = run_dir / 'metrics.json'
    if not metrics_path.exists():
        raise FileNotFoundError(f'Missing metrics.json for {run_name}')
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    metrics.update({
        'run': run_name,
        'seed': seed,
        **{k: v for k, v in params.items() if k not in ('note',)},
        'run_dir': str(run_dir),
    })
    return metrics


def summarize(results: List[Dict[str, Any]], csv_path: Path) -> None:
    import pandas as pd
    if not results:
        print('No successful runs.')
        return
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    df_sorted = df.sort_values(['NDCG_10', 'spread'], ascending=[False, False])
    print('\n===== GRID SUMMARY =====')
    cols = ['run', 'learning_rate', 'num_leaves', 'max_depth', 'min_data_in_leaf', 'lambda_l2',
            'lambdarank_truncation_level', 'sigmoid', 'label_gain_power',
            'top_1_10_mean', 'top_5_15_mean', 'top_10_20_mean', 'spread', 'NDCG_10', 'NDCG_20', 'IC_mean']
    present = [c for c in cols if c in df_sorted.columns]
    print(df_sorted[present].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    best = df_sorted.iloc[0]
    print('\nBEST RUN:')
    print(best[['run', 'learning_rate', 'num_leaves', 'max_depth', 'min_data_in_leaf',
                'lambda_l2', 'lambdarank_truncation_level', 'sigmoid', 'label_gain_power',
                'NDCG_10', 'NDCG_20', 'top_1_10_mean', 'spread']])
    print(f"\nResults saved to {csv_path}")


def main() -> int:
    args = parse_args()
    configs = build_configs()
    if args.single_run:
        if args.single_run not in configs:
            raise ValueError(f'Unknown run: {args.single_run}')
        configs = {args.single_run: configs[args.single_run]}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = OUTPUT_BASE / f'run_{timestamp}'
    output_root.mkdir(parents=True, exist_ok=True)
    print('===== LAMBDARANK-ONLY GRID (next_grid_15) =====')
    print(f'Data: {args.data_file}')
    print(f'Time fraction: {args.time_fraction*100:.0f}%  Split: {args.split*100:.0f}/{(1-args.split)*100:.0f}')
    print(f'Horizon: {args.horizon_days}  CV splits: {args.cv_splits}  Max parallel: {args.max_parallel}')
    print(f'Output root: {output_root}')
    print(f'Features ({len(args.features)}): {args.features}')
    print(f'Seeds: {args.seeds}')
    results: List[Dict[str, Any]] = []
    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as executor:
        for seed in args.seeds:
            for run_name, params in configs.items():
                futures[executor.submit(run_single_config, run_name, params, args, output_root, seed)] = (run_name, seed)
        for future in as_completed(futures):
            run_name, seed = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f'Run {run_name} (seed {seed}) failed: {exc}')
    csv_path = output_root / '15run_grid_results.csv'
    summarize([r for r in results if 'NDCG_10' in r], csv_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
