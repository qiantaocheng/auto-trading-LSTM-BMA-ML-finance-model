#!/usr/bin/env python3
"""LambdaRank Regime Fusion — 6-Test Walkforward Battery.

Tests whether adding market regime features (VIX, SPY trend) to the
LambdaRank stock ranker improves T+5 ranking quality.

Tests:
  T0: Baseline (15 stock-level features only)
  T1: Full Regime (15 stock + 7 regime = 22 features)
  T2: Regime-split evaluation (same model as T1, split by bull/bear)
  T3: Minimal Regime (15 stock + 2 regime = 17 features)
  T4: Regime-Gated strategy (better of T0/T1, zero return when SPY < MA200)
  T5: Interaction features (15 stock + 2 interaction = 17 features)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
REGIME_FEATURES_FULL = [
    'regime_vix', 'regime_vix_20d_chg', 'regime_spy_ma200_dev',
    'regime_spy_above_ma', 'regime_hvr', 'regime_spy_mom_1m', 'regime_spy_dd_20d',
]
REGIME_FEATURES_MINIMAL = ['regime_vix', 'regime_spy_ma200_dev']
INTERACTION_FEATURES = ['interact_mom_vix', 'interact_mom_bull']

# Walkforward config (matching v2 run)
TRAIN_DAYS = 504
TEST_DAYS = 63
STEP_DAYS = 63
GAP_DAYS = 5
CV_SPLITS = 5
N_BOOST_ROUND = 800
REBALANCE_DAYS = 5
PIT_MIN_PRICE = 5.0
PIT_MIN_DOLLAR_VOL = 500_000.0


def run_walkforward(df: pd.DataFrame, features: List[str], params: dict,
                    seed: int = 0, max_windows: int | None = None,
                    regime_split: bool = False,
                    regime_gated: bool = False,
                    ) -> Tuple[List[Dict], Dict]:
    """Run walkforward evaluation with given features.

    Returns (per-window metrics list, feature importance dict).
    """
    dates = df.index.get_level_values('date').unique().sort_values()
    records = []
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
        )
        train_df = slice_by_date(combined, train_range[0], train_range[1])
        test_df = slice_by_date(combined, test_range[0], test_range[1])
        if train_df.empty or test_df.empty:
            continue

        model, best_rounds = train_lambdarank(
            train_df, features, params,
            cv_splits=CV_SPLITS, gap=GAP_DAYS, embargo=GAP_DAYS,
            n_boost_round=N_BOOST_ROUND, seed=seed + win_id,
        )

        # Accumulate feature importance
        imp = dict(zip(features, model.feature_importance(importance_type='gain')))
        for k, v in imp.items():
            importance_accum[k] = importance_accum.get(k, 0) + v

        if regime_split:
            # T2: split test into bull/bear based on regime_spy_above_ma
            for regime_name, regime_val in [('bull', 1.0), ('bear', 0.0)]:
                if 'regime_spy_above_ma' not in test_df.columns:
                    continue
                mask = test_df['regime_spy_above_ma'] == regime_val
                sub_df = test_df[mask]
                if len(sub_df) < 20:
                    continue
                metrics = evaluate(model, sub_df, features, rebalance_days=REBALANCE_DAYS)
                metrics['window_id'] = win_id
                metrics['regime'] = regime_name
                metrics['train_start'] = train_range[0]
                metrics['train_end'] = train_range[1]
                metrics['test_start'] = test_range[0]
                metrics['test_end'] = test_range[1]
                metrics['best_rounds'] = int(best_rounds)
                records.append(metrics)
        elif regime_gated:
            # T4: zero out returns on bear days
            metrics = evaluate(model, test_df, features, rebalance_days=REBALANCE_DAYS)
            # Also compute gated metrics
            if 'regime_spy_above_ma' in test_df.columns:
                gated_df = test_df.copy()
                bear_mask = gated_df['regime_spy_above_ma'] == 0.0
                gated_df.loc[bear_mask, 'target'] = 0.0
                gated_metrics = evaluate(model, gated_df, features, rebalance_days=REBALANCE_DAYS)
                # Prefix gated metrics
                for k, v in gated_metrics.items():
                    metrics[f'gated_{k}'] = v
            metrics['window_id'] = win_id
            metrics['train_start'] = train_range[0]
            metrics['train_end'] = train_range[1]
            metrics['test_start'] = test_range[0]
            metrics['test_end'] = test_range[1]
            metrics['best_rounds'] = int(best_rounds)
            records.append(metrics)
        else:
            metrics = evaluate(model, test_df, features, rebalance_days=REBALANCE_DAYS)
            metrics['window_id'] = win_id
            metrics['train_start'] = train_range[0]
            metrics['train_end'] = train_range[1]
            metrics['test_start'] = test_range[0]
            metrics['test_end'] = test_range[1]
            metrics['best_rounds'] = int(best_rounds)
            records.append(metrics)

        print(f"  Window {win_id}: train {train_range[0].date()} -> {train_range[1].date()}, "
              f"test {test_range[0].date()} -> {test_range[1].date()}")

    # Normalize importance
    total_imp = sum(importance_accum.values()) or 1.0
    importance = {k: v / total_imp for k, v in sorted(importance_accum.items(), key=lambda x: -x[1])}
    return records, importance


def aggregate_metrics(records: List[Dict]) -> Dict[str, float]:
    """Compute mean/median across windows for key metrics."""
    if not records:
        return {}
    df = pd.DataFrame(records)
    key_cols = [c for c in df.columns if isinstance(df[c].iloc[0], (int, float))
                and c not in ('window_id', 'best_rounds', 'train_rows', 'test_rows')]
    agg = {}
    for col in key_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            agg[f'{col}_mean'] = float(vals.mean())
    agg['n_windows'] = len(records)
    return agg


def print_comparison(results: Dict[str, Dict], label: str = ""):
    """Print formatted comparison table."""
    print(f"\n{'='*80}")
    print(f" REGIME FUSION COMPARISON {label}")
    print(f"{'='*80}")

    # Key metrics to compare
    key_metrics = [
        ('overlap_top_1_10_mean', 'Top1-10 Mean'),
        ('overlap_top_1_10_sharpe', 'Top1-10 Sharpe'),
        ('overlap_top_5_15_mean', 'Top5-15 Mean'),
        ('overlap_top_5_15_sharpe', 'Top5-15 Sharpe'),
        ('overlap_top_10_20_mean', 'Top10-20 Mean'),
        ('overlap_top_10_20_sharpe', 'Top10-20 Sharpe'),
        ('spread', 'Spread'),
        ('IC_mean', 'IC'),
        ('NDCG_10', 'NDCG@10'),
        ('NDCG_20', 'NDCG@20'),
    ]

    # Header
    tests = sorted(results.keys())
    header = f"{'Metric':<22}" + "".join(f"{t:>14}" for t in tests)
    print(header)
    print("-" * len(header))

    for metric_key, metric_label in key_metrics:
        row = f"{metric_label:<22}"
        for t in tests:
            agg = results[t].get('aggregate', {})
            val = agg.get(f'{metric_key}_mean', float('nan'))
            if math.isnan(val) if isinstance(val, float) else False:
                row += f"{'N/A':>14}"
            else:
                row += f"{val:>14.4f}"
        print(row)

    # Win rate
    row = f"{'Top1-10 WR':<22}"
    for t in tests:
        agg = results[t].get('aggregate', {})
        val = agg.get('overlap_top_1_10_wr_mean', float('nan'))
        if not math.isnan(val):
            row += f"{val:>13.1%}"
        else:
            row += f"{'N/A':>14}"
    print(row)

    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='LambdaRank Regime Fusion Tests')
    parser.add_argument('--data-file', type=Path,
                        default=Path('data/factor_exports/polygon_full_features_T5_v2_regime.parquet'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/regime_fusion'))
    parser.add_argument('--max-windows', type=int, default=None)
    parser.add_argument('--tests', nargs='+', default=['T0', 'T1', 'T2', 'T3', 'T4', 'T5'],
                        help='Which tests to run (default: all)')
    args = parser.parse_args()

    print("=" * 70)
    print("LambdaRank Regime Fusion — 6-Test Walkforward Battery")
    print("=" * 70)

    # Load data
    print(f"\nLoading data: {args.data_file}")
    df = ensure_multiindex(pd.read_parquet(args.data_file))
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    dates = df.index.get_level_values('date').unique().sort_values()
    print(f"  Dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")

    params = _load_params(None)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict] = {}
    tests_to_run = [t.upper() for t in args.tests]

    # ── T0: Baseline ──
    if 'T0' in tests_to_run:
        print(f"\n{'─'*60}")
        print("T0: Baseline (15 stock-level features)")
        print(f"{'─'*60}")
        features_t0 = list(STOCK_FEATURES)
        records, importance = run_walkforward(df, features_t0, params,
                                              max_windows=args.max_windows)
        agg = aggregate_metrics(records)
        all_results['T0'] = {
            'description': 'Baseline (15 stock features)',
            'features': features_t0,
            'records': records,
            'aggregate': agg,
            'importance': importance,
        }
        print(f"  Windows: {len(records)}, Top1-10 mean: {agg.get('overlap_top_1_10_mean_mean', 'N/A'):.4f}")

    # ── T1: Full Regime ──
    if 'T1' in tests_to_run:
        print(f"\n{'─'*60}")
        print("T1: Full Regime (15 stock + 7 regime = 22 features)")
        print(f"{'─'*60}")
        features_t1 = list(STOCK_FEATURES) + REGIME_FEATURES_FULL
        missing = [f for f in features_t1 if f not in df.columns]
        if missing:
            print(f"  SKIP: missing features {missing}")
        else:
            records, importance = run_walkforward(df, features_t1, params,
                                                  max_windows=args.max_windows)
            agg = aggregate_metrics(records)
            all_results['T1'] = {
                'description': 'Full Regime (22 features)',
                'features': features_t1,
                'records': records,
                'aggregate': agg,
                'importance': importance,
            }
            print(f"  Windows: {len(records)}, Top1-10 mean: {agg.get('overlap_top_1_10_mean_mean', 'N/A'):.4f}")
            print(f"  Regime feature importance:")
            for f in REGIME_FEATURES_FULL:
                print(f"    {f}: {importance.get(f, 0):.4f}")

    # ── T2: Regime-Split Evaluation ──
    if 'T2' in tests_to_run:
        print(f"\n{'─'*60}")
        print("T2: Regime-Split (same model as T1, split by bull/bear)")
        print(f"{'─'*60}")
        features_t2 = list(STOCK_FEATURES) + REGIME_FEATURES_FULL
        missing = [f for f in features_t2 if f not in df.columns]
        if missing:
            print(f"  SKIP: missing features {missing}")
        else:
            records, importance = run_walkforward(df, features_t2, params,
                                                  max_windows=args.max_windows,
                                                  regime_split=True)
            # Split records by regime
            bull_records = [r for r in records if r.get('regime') == 'bull']
            bear_records = [r for r in records if r.get('regime') == 'bear']
            bull_agg = aggregate_metrics(bull_records)
            bear_agg = aggregate_metrics(bear_records)
            all_results['T2_bull'] = {
                'description': 'Regime-Split: Bull (SPY > MA200)',
                'features': features_t2,
                'records': bull_records,
                'aggregate': bull_agg,
                'importance': importance,
            }
            all_results['T2_bear'] = {
                'description': 'Regime-Split: Bear (SPY < MA200)',
                'features': features_t2,
                'records': bear_records,
                'aggregate': bear_agg,
                'importance': importance,
            }
            print(f"  Bull windows: {len(bull_records)}, Bear windows: {len(bear_records)}")
            if bull_agg:
                print(f"  Bull Top1-10 mean: {bull_agg.get('overlap_top_1_10_mean_mean', 'N/A'):.4f}")
            if bear_agg:
                print(f"  Bear Top1-10 mean: {bear_agg.get('overlap_top_1_10_mean_mean', 'N/A'):.4f}")

    # ── T3: Minimal Regime ──
    if 'T3' in tests_to_run:
        print(f"\n{'─'*60}")
        print("T3: Minimal Regime (15 stock + 2 regime = 17 features)")
        print(f"{'─'*60}")
        features_t3 = list(STOCK_FEATURES) + REGIME_FEATURES_MINIMAL
        missing = [f for f in features_t3 if f not in df.columns]
        if missing:
            print(f"  SKIP: missing features {missing}")
        else:
            records, importance = run_walkforward(df, features_t3, params,
                                                  max_windows=args.max_windows)
            agg = aggregate_metrics(records)
            all_results['T3'] = {
                'description': 'Minimal Regime (17 features)',
                'features': features_t3,
                'records': records,
                'aggregate': agg,
                'importance': importance,
            }
            print(f"  Windows: {len(records)}, Top1-10 mean: {agg.get('overlap_top_1_10_mean_mean', 'N/A'):.4f}")

    # ── T4: Regime-Gated Strategy ──
    if 'T4' in tests_to_run:
        print(f"\n{'─'*60}")
        print("T4: Regime-Gated (best model, zero return on bear days)")
        print(f"{'─'*60}")
        # Use T0 features (baseline) with gating
        features_t4 = list(STOCK_FEATURES)
        records, importance = run_walkforward(df, features_t4, params,
                                              max_windows=args.max_windows,
                                              regime_gated=True)
        agg = aggregate_metrics(records)
        all_results['T4'] = {
            'description': 'Regime-Gated (baseline model, bear=0)',
            'features': features_t4,
            'records': records,
            'aggregate': agg,
            'importance': importance,
        }
        # Report gated vs ungated
        gated_key = 'gated_overlap_top_1_10_mean_mean'
        ungated_key = 'overlap_top_1_10_mean_mean'
        print(f"  Ungated Top1-10: {agg.get(ungated_key, 'N/A'):.4f}")
        print(f"  Gated   Top1-10: {agg.get(gated_key, 'N/A'):.4f}")

    # ── T5: Interaction Features ──
    if 'T5' in tests_to_run:
        print(f"\n{'─'*60}")
        print("T5: Interaction Features (15 stock + 2 interaction = 17 features)")
        print(f"{'─'*60}")
        features_t5 = list(STOCK_FEATURES) + INTERACTION_FEATURES
        missing = [f for f in features_t5 if f not in df.columns]
        if missing:
            print(f"  SKIP: missing features {missing}")
        else:
            records, importance = run_walkforward(df, features_t5, params,
                                                  max_windows=args.max_windows)
            agg = aggregate_metrics(records)
            all_results['T5'] = {
                'description': 'Interaction Features (17 features)',
                'features': features_t5,
                'records': records,
                'aggregate': agg,
                'importance': importance,
            }
            print(f"  Windows: {len(records)}, Top1-10 mean: {agg.get('overlap_top_1_10_mean_mean', 'N/A'):.4f}")
            print(f"  Interaction feature importance:")
            for f in INTERACTION_FEATURES:
                print(f"    {f}: {importance.get(f, 0):.4f}")

    # ── Comparison Table ──
    print_comparison(all_results)

    # ── Save Results ──
    # Serialize for JSON (strip non-serializable items)
    json_results = {}
    for test_name, data in all_results.items():
        json_results[test_name] = {
            'description': data['description'],
            'features': data['features'],
            'aggregate': to_serializable(data['aggregate']),
            'importance': to_serializable(data['importance']),
            'n_windows': len(data['records']),
        }

    (args.output_dir / 'test_results.json').write_text(
        json.dumps(json_results, indent=2), encoding='utf-8')

    # Comparison CSV
    rows = []
    for test_name, data in all_results.items():
        row = {'test': test_name, 'description': data['description']}
        row.update(data['aggregate'])
        rows.append(row)
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(args.output_dir / 'comparison_table.csv', index=False)

    # Per-window metrics CSVs
    for test_name, data in all_results.items():
        if data['records']:
            pd.DataFrame(data['records']).to_csv(
                args.output_dir / f'{test_name}_windows.csv', index=False)

    print(f"\nResults saved to {args.output_dir}/")
    print("Done!")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
