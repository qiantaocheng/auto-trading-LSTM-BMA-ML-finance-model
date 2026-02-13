#!/usr/bin/env python3
"""Bull Market Ranking Improvement — 7-Test Walkforward Battery.

Tests three directions for improving LambdaRank stock ranking in bull markets:
  1. Excess-return labeling (target_excess_qqq instead of raw target)
  2. Universe pre-filtering (top-500 by dollar volume)
  3. Sector-relative features (sector_rel_momentum_10d, etc.)

Tests:
  B0: Baseline (15 stock features, raw target, full universe)
  B1: Excess-return labeling only
  B2: Top-500 universe only
  B3: Excess + top-500
  B4: Sector features + excess (full universe)
  B5: Sector features + excess + top-500
  B6: T5 interaction + excess + top-500
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.lambdarank_only_pipeline import (
    FEATURES as STOCK_FEATURES,
    ensure_multiindex,
    train_lambdarank,
    evaluate,
    _load_params,
)
from scripts.lambdarank_walkforward import (
    iter_time_windows,
    pit_universe_filter,
    slice_by_date,
    to_serializable,
)

# Feature groups
INTERACTION_FEATURES = ['interact_mom_vix', 'interact_mom_bull']
SECTOR_REL_FEATURES = ['sector_rel_momentum_10d', 'sector_rel_reversal_3d', 'sector_rel_rsi_14']

# Walkforward config
TRAIN_DAYS = 504
TEST_DAYS = 63
STEP_DAYS = 63
GAP_DAYS = 5
CV_SPLITS = 5
N_BOOST_ROUND = 800
REBALANCE_DAYS = 5
PIT_MIN_PRICE = 5.0
PIT_MIN_DOLLAR_VOL = 500_000.0

# Test configurations
TEST_CONFIGS = {
    'B0': {
        'description': 'Baseline (15 stock, raw target, full)',
        'features': list(STOCK_FEATURES),
        'target_col': 'target',
        'max_tickers': None,
    },
    'B1': {
        'description': 'Excess-return labeling (15 stock, excess, full)',
        'features': list(STOCK_FEATURES),
        'target_col': 'target_excess_qqq',
        'max_tickers': None,
    },
    'B2': {
        'description': 'Top-500 universe (15 stock, raw, top-500)',
        'features': list(STOCK_FEATURES),
        'target_col': 'target',
        'max_tickers': 500,
    },
    'B3': {
        'description': 'Excess + top-500 (15 stock)',
        'features': list(STOCK_FEATURES),
        'target_col': 'target_excess_qqq',
        'max_tickers': 500,
    },
    'B4': {
        'description': 'Sector-rel + excess (18 features, full)',
        'features': list(STOCK_FEATURES) + SECTOR_REL_FEATURES,
        'target_col': 'target_excess_qqq',
        'max_tickers': None,
    },
    'B5': {
        'description': 'Sector-rel + excess + top-500 (18 features)',
        'features': list(STOCK_FEATURES) + SECTOR_REL_FEATURES,
        'target_col': 'target_excess_qqq',
        'max_tickers': 500,
    },
    'B6': {
        'description': 'T5 interact + excess + top-500 (17 features)',
        'features': list(STOCK_FEATURES) + INTERACTION_FEATURES,
        'target_col': 'target_excess_qqq',
        'max_tickers': 500,
    },
}


def run_walkforward(df: pd.DataFrame, features: List[str], params: dict,
                    target_col: str = 'target',
                    max_tickers: int | None = None,
                    seed: int = 0,
                    max_windows: int | None = None,
                    ) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """Run walkforward with regime-split evaluation.

    Returns (overall_records, bull_records, bear_records, feature_importance).
    """
    dates = df.index.get_level_values('date').unique().sort_values()
    overall_records = []
    bull_records = []
    bear_records = []
    importance_accum: Dict[str, float] = {}

    for win_id, (ts, te, vs, ve) in enumerate(
        iter_time_windows(dates, TRAIN_DAYS, GAP_DAYS, TEST_DAYS, STEP_DAYS)
    ):
        if max_windows is not None and win_id >= max_windows:
            break

        train_range = (dates[ts], dates[te - 1])
        test_range = (dates[vs], dates[ve - 1])
        train_df = slice_by_date(df, train_range[0], train_range[1])
        test_df = slice_by_date(df, test_range[0], test_range[1])
        if train_df.empty or test_df.empty:
            continue

        # PIT filter
        combined = pd.concat([train_df, test_df])
        combined = pit_universe_filter(
            combined, train_range[1],
            min_price=PIT_MIN_PRICE, min_dollar_vol=PIT_MIN_DOLLAR_VOL,
            max_tickers=max_tickers,
        )
        train_df = slice_by_date(combined, train_range[0], train_range[1])
        test_df = slice_by_date(combined, test_range[0], test_range[1])
        if train_df.empty or test_df.empty:
            continue

        # Drop rows where target_col is NaN (important for excess target)
        valid_train = train_df[target_col].notna()
        valid_test = test_df['target'].notna()  # evaluate always uses raw target
        if target_col != 'target':
            valid_train &= train_df['target'].notna()
            valid_test &= test_df[target_col].notna()
        train_df = train_df[valid_train]
        test_df = test_df[valid_test]
        if train_df.empty or test_df.empty:
            continue

        model, best_rounds = train_lambdarank(
            train_df, features, params,
            cv_splits=CV_SPLITS, gap=GAP_DAYS, embargo=GAP_DAYS,
            n_boost_round=N_BOOST_ROUND, seed=seed + win_id,
            target_col=target_col,
        )

        # Feature importance
        imp = dict(zip(features, model.feature_importance(importance_type='gain')))
        for k, v in imp.items():
            importance_accum[k] = importance_accum.get(k, 0) + v

        # Overall evaluation (always uses raw 'target' for return measurement)
        metrics = evaluate(model, test_df, features, rebalance_days=REBALANCE_DAYS)
        metrics['window_id'] = win_id
        metrics['train_start'] = train_range[0]
        metrics['train_end'] = train_range[1]
        metrics['test_start'] = test_range[0]
        metrics['test_end'] = test_range[1]
        metrics['best_rounds'] = int(best_rounds)
        metrics['n_tickers'] = test_df.index.get_level_values('ticker').nunique()
        overall_records.append(metrics)

        # Regime-split evaluation
        if 'regime_spy_above_ma' in test_df.columns:
            for regime_name, regime_val in [('bull', 1.0), ('bear', 0.0)]:
                mask = test_df['regime_spy_above_ma'] == regime_val
                sub_df = test_df[mask]
                if len(sub_df) < 20:
                    continue
                r_metrics = evaluate(model, sub_df, features, rebalance_days=REBALANCE_DAYS)
                r_metrics['window_id'] = win_id
                r_metrics['regime'] = regime_name
                if regime_name == 'bull':
                    bull_records.append(r_metrics)
                else:
                    bear_records.append(r_metrics)

        print(f"  Window {win_id}: train {train_range[0].date()} -> {train_range[1].date()}, "
              f"test {test_range[0].date()} -> {test_range[1].date()}, "
              f"tickers={metrics['n_tickers']}")

    total_imp = sum(importance_accum.values()) or 1.0
    importance = {k: v / total_imp for k, v in sorted(importance_accum.items(), key=lambda x: -x[1])}
    return overall_records, bull_records, bear_records, importance


def aggregate(records: List[Dict]) -> Dict[str, float]:
    if not records:
        return {}
    df = pd.DataFrame(records)
    key_cols = [c for c in df.columns if isinstance(df[c].iloc[0], (int, float))
                and c not in ('window_id', 'best_rounds', 'n_tickers')]
    agg = {}
    for col in key_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            agg[f'{col}_mean'] = float(vals.mean())
    agg['n_windows'] = len(records)
    return agg


def print_comparison(results: Dict[str, Dict]):
    """Print formatted comparison table."""
    print(f"\n{'='*100}")
    print(f" BULL MARKET IMPROVEMENT COMPARISON")
    print(f"{'='*100}")

    key_metrics = [
        ('overlap_top_1_10_mean', 'Top1-10 Mean'),
        ('overlap_top_1_10_sharpe', 'Top1-10 Sharpe'),
        ('overlap_top_1_10_wr', 'Top1-10 WR'),
        ('overlap_top_5_15_mean', 'Top5-15 Mean'),
        ('overlap_top_5_15_sharpe', 'Top5-15 Sharpe'),
        ('spread', 'Spread'),
        ('IC_mean', 'IC'),
        ('NDCG_10', 'NDCG@10'),
    ]

    # Three sections: Overall, Bull, Bear
    for section, suffix in [('OVERALL', 'overall'), ('BULL (SPY > MA200)', 'bull'), ('BEAR (SPY < MA200)', 'bear')]:
        print(f"\n  --- {section} ---")
        tests = sorted(results.keys())
        header = f"  {'Metric':<18}" + "".join(f"{t:>12}" for t in tests)
        print(header)
        print("  " + "-" * (18 + 12 * len(tests)))

        for metric_key, metric_label in key_metrics:
            row = f"  {metric_label:<18}"
            for t in tests:
                agg = results[t].get(f'agg_{suffix}', {})
                val = agg.get(f'{metric_key}_mean', float('nan'))
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    row += f"{'N/A':>12}"
                elif 'WR' in metric_label:
                    row += f"{val:>11.1%}"
                else:
                    row += f"{val:>12.4f}"
            print(row)

    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Bull Market Ranking Improvement Tests')
    parser.add_argument('--data-file', type=Path,
                        default=Path('data/factor_exports/polygon_full_features_T5_v3_bull.parquet'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/bull_improvement'))
    parser.add_argument('--max-windows', type=int, default=None)
    parser.add_argument('--tests', nargs='+', default=None,
                        help='Which tests to run (default: all available)')
    args = parser.parse_args()

    print("=" * 70)
    print("Bull Market Ranking Improvement — 7-Test Battery")
    print("=" * 70)

    # Load data
    print(f"\nLoading data: {args.data_file}")
    df = ensure_multiindex(pd.read_parquet(args.data_file))
    print(f"  Shape: {df.shape}")
    available_cols = set(df.columns)
    dates = df.index.get_level_values('date').unique().sort_values()
    print(f"  Dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")

    params = _load_params(None)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which tests to run
    requested = [t.upper() for t in args.tests] if args.tests else list(TEST_CONFIGS.keys())

    all_results: Dict[str, Dict] = {}

    for test_name in requested:
        if test_name not in TEST_CONFIGS:
            print(f"\nSkipping unknown test: {test_name}")
            continue

        cfg = TEST_CONFIGS[test_name]

        # Check feature availability
        missing = [f for f in cfg['features'] if f not in available_cols]
        if missing:
            print(f"\n{'='*60}")
            print(f"{test_name}: SKIP — missing features: {missing}")
            continue

        # Check target column availability
        if cfg['target_col'] not in available_cols:
            print(f"\n{'='*60}")
            print(f"{test_name}: SKIP — missing target column: {cfg['target_col']}")
            continue

        print(f"\n{'='*60}")
        print(f"{test_name}: {cfg['description']}")
        print(f"  Features: {len(cfg['features'])}, Target: {cfg['target_col']}, "
              f"MaxTickers: {cfg['max_tickers'] or 'all'}")
        print(f"{'='*60}")

        overall, bull, bear, importance = run_walkforward(
            df, cfg['features'], params,
            target_col=cfg['target_col'],
            max_tickers=cfg['max_tickers'],
            max_windows=args.max_windows,
        )

        all_results[test_name] = {
            'description': cfg['description'],
            'features': cfg['features'],
            'target_col': cfg['target_col'],
            'max_tickers': cfg['max_tickers'],
            'overall': overall,
            'bull': bull,
            'bear': bear,
            'agg_overall': aggregate(overall),
            'agg_bull': aggregate(bull),
            'agg_bear': aggregate(bear),
            'importance': importance,
        }

        agg = all_results[test_name]['agg_overall']
        agg_bull = all_results[test_name]['agg_bull']
        agg_bear = all_results[test_name]['agg_bear']
        print(f"  Overall: Top1-10={agg.get('overlap_top_1_10_mean_mean', 'N/A'):.4f}, "
              f"Sharpe={agg.get('overlap_top_1_10_sharpe_mean', 'N/A'):.2f}")
        if agg_bull:
            print(f"  Bull:    Top1-10={agg_bull.get('overlap_top_1_10_mean_mean', 'N/A'):.4f}, "
                  f"Sharpe={agg_bull.get('overlap_top_1_10_sharpe_mean', 'N/A'):.2f}")
        if agg_bear:
            print(f"  Bear:    Top1-10={agg_bear.get('overlap_top_1_10_mean_mean', 'N/A'):.4f}, "
                  f"Sharpe={agg_bear.get('overlap_top_1_10_sharpe_mean', 'N/A'):.2f}")

    # Comparison table
    if all_results:
        print_comparison(all_results)

    # Save results
    json_results = {}
    for test_name, data in all_results.items():
        json_results[test_name] = {
            'description': data['description'],
            'target_col': data['target_col'],
            'max_tickers': data['max_tickers'],
            'n_features': len(data['features']),
            'agg_overall': to_serializable(data['agg_overall']),
            'agg_bull': to_serializable(data['agg_bull']),
            'agg_bear': to_serializable(data['agg_bear']),
            'importance': to_serializable(data['importance']),
        }

    (args.output_dir / 'test_results.json').write_text(
        json.dumps(json_results, indent=2), encoding='utf-8')

    # Per-test window CSVs
    for test_name, data in all_results.items():
        if data['overall']:
            pd.DataFrame(data['overall']).to_csv(
                args.output_dir / f'{test_name}_overall.csv', index=False)
        if data['bull']:
            pd.DataFrame(data['bull']).to_csv(
                args.output_dir / f'{test_name}_bull.csv', index=False)
        if data['bear']:
            pd.DataFrame(data['bear']).to_csv(
                args.output_dir / f'{test_name}_bear.csv', index=False)

    print(f"\nResults saved to {args.output_dir}/")
    print("Done!")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
