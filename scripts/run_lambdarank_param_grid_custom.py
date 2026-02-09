#!/usr/bin/env python3
"""LambdaRank parameter grid sweep for run9_plus_atr_pct_14_plus_amihud_20 feature set."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_DATA_FILE = Path('data/factor_exports/polygon_full_features_T5.parquet')
DEFAULT_PIPELINE = Path('scripts/lambdarank_only_pipeline.py')
DEFAULT_OUTPUT_BASE = Path('results/param_grid_lambdarank')
DEFAULT_FEATURES = [
    'volume_price_corr_3d', 'rsi_14', 'reversal_3d', 'momentum_10d', 'liquid_momentum_10d',
    'sharpe_momentum_5d', 'price_ma20_deviation', 'avg_trade_size', 'trend_r2_20',
    'dollar_vol_20', 'ret_skew_20d', 'reversal_5d', 'near_52w_high', 'atr_pct_14', 'amihud_20'
]

PARAM_GRID = [
    {
        "run_id": 1,
        "name": "anchor_baseline_r11",
        "params": {
            "learning_rate": 0.04,
            "num_leaves": 11,
            "max_depth": 3,
            "min_data_in_leaf": 350,
            "lambda_l2": 120,
            "feature_fraction": 1.0,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "min_gain_to_split": 0.1,
            "lambdarank_truncation_level": 25,
            "sigmoid": 1.1,
            "label_gain_power": 2.1
        }
    },
    {
        "run_id": 2,
        "name": "bagging_055_ff_090",
        "params": {
            "learning_rate": 0.04,
            "num_leaves": 11,
            "max_depth": 3,
            "min_data_in_leaf": 350,
            "lambda_l2": 120,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.55,
            "bagging_freq": 1,
            "min_gain_to_split": 0.1,
            "lambdarank_truncation_level": 25,
            "sigmoid": 1.1,
            "label_gain_power": 2.1
        }
    },
    {
        "run_id": 3,
        "name": "bagging_060_ff_090",
        "params": {
            "learning_rate": 0.04,
            "num_leaves": 11,
            "max_depth": 3,
            "min_data_in_leaf": 350,
            "lambda_l2": 120,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.6,
            "bagging_freq": 1,
            "min_gain_to_split": 0.1,
            "lambdarank_truncation_level": 25,
            "sigmoid": 1.1,
            "label_gain_power": 2.1
        }
    },
    {
        "run_id": 4,
        "name": "bagging_070_ff_085",
        "params": {
            "learning_rate": 0.04,
            "num_leaves": 11,
            "max_depth": 3,
            "min_data_in_leaf": 350,
            "lambda_l2": 120,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "min_gain_to_split": 0.1,
            "lambdarank_truncation_level": 25,
            "sigmoid": 1.1,
            "label_gain_power": 2.1
        }
    },
    {
        "run_id": 5,
        "name": "bagging_080_ff_085",
        "params": {
            "learning_rate": 0.04,
            "num_leaves": 11,
            "max_depth": 3,
            "min_data_in_leaf": 350,
            "lambda_l2": 120,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_gain_to_split": 0.1,
            "lambdarank_truncation_level": 25,
            "sigmoid": 1.1,
            "label_gain_power": 2.1
        }
    },
    {
        "run_id": 6,
        "name": "splitgain_020",
        "params": {
            "learning_rate": 0.04,
            "num_leaves": 11,
            "max_depth": 3,
            "min_data_in_leaf": 350,
            "lambda_l2": 120,
            "feature_fraction": 1.0,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "min_gain_to_split": 0.2,
            "lambdarank_truncation_level": 25,
            "sigmoid": 1.1,
            "label_gain_power": 2.1
        }
    },
    {
        "run_id": 7,
        "name": "splitgain_030",
        "params": {
            "learning_rate": 0.04,
            "num_leaves": 11,
            "max_depth": 3,
            "min_data_in_leaf": 350,
            "lambda_l2": 120,
            "feature_fraction": 1.0,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "min_gain_to_split": 0.05,
            "lambdarank_truncation_level": 25,
            "sigmoid": 1.1,
            "label_gain_power": 2.1
        }
    },
    {
        "run_id": 8,
        "name": "trunc_15",
        "params": {
            "learning_rate": 0.04,
            "num_leaves": 11,
            "max_depth": 3,
            "min_data_in_leaf": 350,
            "lambda_l2": 120,
            "feature_fraction": 1.0,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "min_gain_to_split": 0.1,
            "lambdarank_truncation_level": 15,
            "sigmoid": 1.1,
            "label_gain_power": 2.1
        }
    },
    {
        "run_id": 9,
        "name": "trunc_35",
        "params": {
            "learning_rate": 0.04,
            "num_leaves": 11,
            "max_depth": 3,
            "min_data_in_leaf": 350,
            "lambda_l2": 120,
            "feature_fraction": 1.0,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "min_gain_to_split": 0.1,
            "lambdarank_truncation_level": 35,
            "sigmoid": 1.1,
            "label_gain_power": 2.1
        }
    },
    {
        "run_id": 10,
        "name": "sigmoid_090",
        "params": {
            "learning_rate": 0.04,
            "num_leaves": 11,
            "max_depth": 3,
            "min_data_in_leaf": 350,
            "lambda_l2": 120,
            "feature_fraction": 1.0,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "min_gain_to_split": 0.1,
            "lambdarank_truncation_level": 25,
            "sigmoid": 0.9,
            "label_gain_power": 2.1
        }
    },
    {
        "run_id": 11,
        "name": "sigmoid_130",
        "params": {
            "learning_rate": 0.04,
            "num_leaves": 11,
            "max_depth": 3,
            "min_data_in_leaf": 350,
            "lambda_l2": 120,
            "feature_fraction": 1.0,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "min_gain_to_split": 0.1,
            "lambdarank_truncation_level": 25,
            "sigmoid": 1.3,
            "label_gain_power": 2.1
        }
    },
    {
        "run_id": 12,
        "name": "combo_best_guess_ff085_bag075_trunc35",
        "params": {
            "learning_rate": 0.04,
            "num_leaves": 11,
            "max_depth": 3,
            "min_data_in_leaf": 350,
            "lambda_l2": 120,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.75,
            "bagging_freq": 1,
            "min_gain_to_split": 0.1,
            "lambdarank_truncation_level": 35,
            "sigmoid": 1.1,
            "label_gain_power": 2.1
        }
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='LambdaRank parameter grid runner')
    parser.add_argument('--data-file', type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument('--time-fraction', type=float, default=0.5)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--horizon-days', type=int, default=5)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--n-boost-round', type=int, default=800)
    parser.add_argument('--seeds', type=int, nargs='+', default=[5])
    parser.add_argument('--max-parallel', type=int, default=3)
    parser.add_argument('--features', nargs='+', default=DEFAULT_FEATURES)
    parser.add_argument('--pipeline', type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument('--output-base', type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument('--runs', nargs='*', default=None,
                        help='Optional subset of run IDs or names (default: entire grid)')
    return parser.parse_args()


def select_configs(requested: List[str] | None) -> List[Dict[str, Any]]:
    if not requested:
        return PARAM_GRID
    wanted = {str(r).lower() for r in requested}
    chosen: List[Dict[str, Any]] = []
    for spec in PARAM_GRID:
        if str(spec['run_id']) in wanted or spec['name'].lower() in wanted:
            chosen.append(spec)
    if not chosen:
        raise ValueError('No matching configs for selection: ' + ', '.join(requested))
    return chosen


def run_single(spec: Dict[str, Any], seed: int, args: argparse.Namespace, output_root: Path) -> Dict[str, Any]:
    run_dir = output_root / f'seed{seed:02d}' / f"{spec['run_id']:02d}_{spec['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    params_path = run_dir / 'params.json'
    params_path.write_text(json.dumps(spec['params'], indent=2), encoding='utf-8')
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
        '--params-json', str(params_path),
        '--features', *args.features,
    ]
    print(f"Launching {spec['name']} (seed {seed}): {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    metrics_path = run_dir / 'metrics.json'
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json for {spec['name']} seed {seed}")
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    metrics.update({
        'run_id': spec['run_id'],
        'run_name': spec['name'],
        'seed': seed,
        **spec['params'],
        'run_dir': str(run_dir),
    })
    return metrics


def summarize(results: List[Dict[str, Any]], output_root: Path) -> None:
    if not results:
        print("No results to summarize.")
        return
    summary_path = output_root / 'summary.json'
    summary = {
        'created_at': datetime.utcnow().isoformat(),
        'num_results': len(results),
        'best_by_ic': sorted(results, key=lambda r: r.get('mean_ic', float('-inf')), reverse=True)[:10],
        'all_results': results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"Wrote summary to {summary_path}")


def main() -> None:
    args = parse_args()
    configs = select_configs(args.runs)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    output_root = args.output_base / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = []
        for spec in configs:
            for seed in args.seeds:
                futures.append(executor.submit(run_single, spec, seed, args, output_root))
        for fut in as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
                print(f"Completed run: {res['run_name']} seed {res['seed']}")
            except Exception as e:
                print(f"Run failed: {e}")

    summarize(results, output_root)


if __name__ == '__main__':
    main()

