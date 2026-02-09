#!/usr/bin/env python3
"""Run LambdaRank grid search on a 1/5 ticker subset and log metrics."""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
import yaml

RESULTS_ROOT = Path('results/t10_time_split_80_20_final')
CONFIG_PATH = Path('bma_models/unified_config.yaml')

FEATURES = [
    'momentum_60d',
    'trend_r2_60',
    'near_52w_high',
    '5_days_reversal',
    'ivol_20',
    'atr_ratio',
    'vol_ratio_20d',
    'liquid_momentum',
    'obv_momentum_60d',
    'obv_divergence',
]

GRID_TRUNCATION = [60, 80, 120, 150]
GRID_SIGMOID = [1.05, 1.15, 1.25, 1.35]
GRID_LABEL_GAIN = [2.0, 2.3, 2.6]


def create_subset(source: Path, target: Path, fraction: float = 0.2) -> None:
    """Create a deterministic ticker subset (~fraction of names)."""
    df = pd.read_parquet(source)
    if isinstance(df.index, pd.MultiIndex) and 'ticker' in df.index.names:
        tickers = df.index.get_level_values('ticker').unique()
    elif 'ticker' in df.columns and 'date' in df.columns:
        df = df.set_index(['date', 'ticker'])
        tickers = df.index.get_level_values('ticker').unique()
    else:
        raise ValueError('Data must contain ticker information')

    tickers = sorted(tickers)
    step = max(1, round(1 / fraction))
    sampled = tickers[::step]
    if not sampled:
        raise ValueError('Sampled ticker list is empty')
    subset = df.loc[(slice(None), sampled), :]
    subset.to_parquet(target)
    print(f'Subset saved to {target} with {len(sampled)} tickers and {len(subset)} rows')


def update_config(ndcg: Sequence[int], truncation: int, sigmoid: float, label_gain: float) -> None:
    """Patch unified_config.yaml with the desired LambdaRank parameters."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    lambdarank_cfg = config['training']['base_models']['lambdarank']
    lambdarank_cfg['ndcg_eval_at'] = list(ndcg)
    lambdarank_cfg['lambdarank_truncation_level'] = truncation
    lambdarank_cfg['sigmoid'] = float(sigmoid)
    lambdarank_cfg['label_gain_power'] = float(label_gain)
    CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')


def latest_run_dir(before: Iterable[Path]) -> Path:
    existing = {p.resolve() for p in before}
    candidates = [p for p in RESULTS_ROOT.glob('run_*') if p.resolve() not in existing]
    if not candidates:
        raise RuntimeError('No new run directory detected')
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_timesplit(data_file: Path, features: List[str]) -> Path:
    """Execute time_split_80_20_oos_eval.py and return its output directory."""
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    before = list(RESULTS_ROOT.glob('run_*'))
    cmd = [
        sys.executable,
        'scripts/time_split_80_20_oos_eval.py',
        '--data-file', str(data_file),
        '--features', *features,
    ]
    print('Running:', ' '.join(cmd))
    start = time.time()
    subprocess.run(cmd, check=True)
    print(f'Run finished in {(time.time() - start)/60:.2f} minutes')
    return latest_run_dir(before)


def extract_metrics(run_dir: Path) -> dict:
    summary_path = run_dir / 'results_summary_for_word_doc.json'
    if not summary_path.exists():
        raise FileNotFoundError(f'Missing summary file: {summary_path}')
    payload = json.loads(summary_path.read_text(encoding='utf-8'))
    lambdarank = payload['lambdarank']
    return {
        'run_dir': run_dir.name,
        'snapshot_id': payload.get('metadata', {}).get('snapshot_id', ''),
        'ic': lambdarank['metrics'].get('IC'),
        'ic_pvalue': lambdarank['metrics'].get('IC_pvalue'),
        'top_bucket_mean': lambdarank['returns'].get('avg_top_bucket_t10_return'),
        'bottom_bucket_mean': lambdarank['returns'].get('avg_bottom_return'),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='LambdaRank grid search on 1/5 subset')
    parser.add_argument('--full-data', type=Path, default=Path('data/factor_exports/polygon_factors_all_2021_2026_CLEAN_STANDARDIZED.parquet'))
    parser.add_argument('--subset-data', type=Path, default=Path('data/factor_exports/polygon_factors_subset_1_5.parquet'))
    parser.add_argument('--output-csv', type=Path, default=Path('results/lambdarank_grid_summary.csv'))
    parser.add_argument('--skip-subset', action='store_true')
    args = parser.parse_args()

    original_config = CONFIG_PATH.read_text(encoding='utf-8')

    try:
        if not args.skip_subset or not args.subset_data.exists():
            create_subset(args.full_data, args.subset_data)

        rows = []
        combos = list(itertools.product(GRID_TRUNCATION, GRID_SIGMOID, GRID_LABEL_GAIN))
        print(f'Starting grid search: {len(combos)} combinations')

        for truncation, sigmoid, label_gain in combos:
            print('\n=== Combination:', truncation, sigmoid, label_gain, '===')
            update_config([10, 20], truncation, sigmoid, label_gain)
            run_dir = run_timesplit(args.subset_data, FEATURES)
            metrics = extract_metrics(run_dir)
            metrics.update({
                'lambdarank_truncation_level': truncation,
                'sigmoid': sigmoid,
                'label_gain_power': label_gain,
            })
            rows.append(metrics)
            pd.DataFrame(rows).to_csv(args.output_csv, index=False)
            print('Results appended to', args.output_csv)

    finally:
        CONFIG_PATH.write_text(original_config, encoding='utf-8')
        print('Config restored')


if __name__ == '__main__':
    main()
