#!/usr/bin/env python3
"""Stock Sleeve Walkforward — Top-10 Equal Weight, 5-Day Rebalance.

Single model (full universe, raw target), always top-10 stocks.
Computes both overlapping (daily) and non-overlapping (5d) stats.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.lambdarank_only_pipeline import (
    FEATURES as STOCK_FEATURES,
    ensure_multiindex,
    train_lambdarank,
    _load_params,
)
from scripts.lambdarank_walkforward import (
    iter_time_windows,
    pit_universe_filter,
    slice_by_date,
    to_serializable,
)

# Walkforward config
TRAIN_DAYS = 504
TEST_DAYS = 63
STEP_DAYS = 63
GAP_DAYS = 5
CV_SPLITS = 5
N_BOOST_ROUND = 800
PIT_MIN_PRICE = 5.0
PIT_MIN_DOLLAR_VOL = 500_000.0
REBALANCE_DAYS = 5
TOP_N = 10


def get_regime(df, date):
    """Get regime for a date. Returns 'bull' or 'bear'."""
    if 'regime_spy_above_ma' not in df.columns:
        return 'bull'
    try:
        day_data = df.loc[(date, slice(None)), :]
        if len(day_data) > 0:
            return 'bull' if day_data['regime_spy_above_ma'].iloc[0] == 1.0 else 'bear'
    except KeyError:
        pass
    return 'bull'


def score_day(model, df, features, date):
    """Score one day, return mean T+5 return of top-N stocks."""
    try:
        day_data = df.loc[(date, slice(None)), :]
    except KeyError:
        return None
    if len(day_data) < 20:
        return None
    X = day_data[features].fillna(0.0).to_numpy()
    preds = model.predict(X)
    targets = day_data['target'].to_numpy()
    order = np.argsort(-preds)
    return float(targets[order[:TOP_N]].mean())


def compute_stats(records: List[Dict], freq_days: int = 1) -> Dict[str, float]:
    """Compute equity stats. freq_days controls annualization."""
    if not records:
        return {}
    df = pd.DataFrame(records).sort_values('date')
    rets = df['return'].values
    cum = np.cumprod(1 + rets)

    n = len(rets)
    periods_per_year = 252.0 / max(1, freq_days)
    mean_ret = float(np.mean(rets))
    std_ret = float(np.std(rets, ddof=1)) if n > 1 else float('nan')
    sharpe = float(mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else float('nan')

    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(dd.min())
    wr = float(np.mean(rets > 0))

    n_years = n / periods_per_year
    cagr = float(cum[-1] ** (1.0 / n_years) - 1) if cum[-1] > 0 and n_years > 0 else float('nan')
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float('nan')

    bull_rets = df[df['regime'] == 'bull']['return'].values
    bear_rets = df[df['regime'] == 'bear']['return'].values

    stats = {
        'sharpe': sharpe, 'cagr': cagr, 'max_dd': max_dd, 'calmar': calmar,
        'win_rate': wr, 'mean_ret': mean_ret, 'n_obs': n,
        'n_bull': len(bull_rets), 'n_bear': len(bear_rets),
    }
    if len(bull_rets) > 1:
        stats['bull_mean'] = float(np.mean(bull_rets))
        bull_std = float(np.std(bull_rets, ddof=1))
        stats['bull_sharpe'] = float(np.mean(bull_rets) / bull_std * np.sqrt(periods_per_year)) \
            if bull_std > 0 else float('nan')
        stats['bull_wr'] = float(np.mean(bull_rets > 0))
    if len(bear_rets) > 1:
        stats['bear_mean'] = float(np.mean(bear_rets))
        bear_std = float(np.std(bear_rets, ddof=1))
        stats['bear_sharpe'] = float(np.mean(bear_rets) / bear_std * np.sqrt(periods_per_year)) \
            if bear_std > 0 else float('nan')
        stats['bear_wr'] = float(np.mean(bear_rets > 0))
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Stock Sleeve Walkforward Test')
    parser.add_argument('--data-file', type=Path,
                        default=Path('data/factor_exports/polygon_full_features_T5_v2_regime.parquet'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/stock_sleeve'))
    parser.add_argument('--max-windows', type=int, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print(f"Stock Sleeve Walkforward — Top-{TOP_N}, {REBALANCE_DAYS}d Rebalance")
    print("=" * 70)

    df = ensure_multiindex(pd.read_parquet(args.data_file))
    print(f"  Shape: {df.shape}")
    dates = df.index.get_level_values('date').unique().sort_values()
    print(f"  Dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")

    params = _load_params(None)
    features = list(STOCK_FEATURES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overlap_records = []
    nonoverlap_records = []

    for win_id, (ts, te, vs, ve) in enumerate(
        iter_time_windows(dates, TRAIN_DAYS, GAP_DAYS, TEST_DAYS, STEP_DAYS)
    ):
        if args.max_windows is not None and win_id >= args.max_windows:
            break
        train_range = (dates[ts], dates[te - 1])
        test_range = (dates[vs], dates[ve - 1])

        # Train
        train_df = slice_by_date(df, train_range[0], train_range[1])
        train_df = pit_universe_filter(train_df, train_range[1],
                                       min_price=PIT_MIN_PRICE, min_dollar_vol=PIT_MIN_DOLLAR_VOL)
        valid = train_df['target'].notna()
        train_df = train_df[valid]
        if train_df.empty:
            continue
        model, _ = train_lambdarank(
            train_df, features, params,
            cv_splits=CV_SPLITS, gap=GAP_DAYS, embargo=GAP_DAYS,
            n_boost_round=N_BOOST_ROUND, seed=win_id,
        )

        # Test with PIT filter
        test_df = slice_by_date(df, test_range[0], test_range[1])
        combined = pd.concat([slice_by_date(df, train_range[0], train_range[1]), test_df])
        combined = pit_universe_filter(combined, train_range[1],
                                       min_price=PIT_MIN_PRICE, min_dollar_vol=PIT_MIN_DOLLAR_VOL)
        test_df = slice_by_date(combined, test_range[0], test_range[1])
        if test_df.empty:
            continue

        test_dates = test_df.index.get_level_values('date').unique().sort_values()
        rebal_dates = set(test_dates[::REBALANCE_DAYS])
        n_bull, n_bear = 0, 0

        for d in test_dates:
            ret = score_day(model, test_df, features, d)
            if ret is None:
                continue
            regime = get_regime(test_df, d)
            record = {'date': d, 'return': ret, 'regime': regime, 'window': win_id}
            overlap_records.append(record)
            if d in rebal_dates:
                nonoverlap_records.append(record)
            if regime == 'bull':
                n_bull += 1
            else:
                n_bear += 1

        print(f"  Window {win_id}: {train_range[0].date()} -> {test_range[1].date()}, "
              f"bull={n_bull}, bear={n_bear}")

    # Stats
    ov_stats = compute_stats(overlap_records, freq_days=1)
    no_stats = compute_stats(nonoverlap_records, freq_days=REBALANCE_DAYS)

    print(f"\n{'='*60}")
    print(f" RESULTS — Top-{TOP_N}, {REBALANCE_DAYS}d Rebalance")
    print(f"{'='*60}")
    for label, s, freq in [('Overlap (daily)', ov_stats, 'daily'), (f'Non-overlap ({REBALANCE_DAYS}d)', no_stats, f'{REBALANCE_DAYS}d')]:
        print(f"\n  --- {label} ---")
        print(f"  Sharpe:      {s.get('sharpe', 0):.2f}")
        print(f"  CAGR:        {s.get('cagr', 0):.2%}")
        print(f"  MaxDD:       {s.get('max_dd', 0):.2%}")
        print(f"  Calmar:      {s.get('calmar', 0):.2f}")
        print(f"  WR:          {s.get('win_rate', 0):.1%}")
        print(f"  Mean({freq}): {s.get('mean_ret', 0):.4%}")
        print(f"  Obs:         {s.get('n_obs', 0)}")
        print(f"  Bull Sharpe: {s.get('bull_sharpe', float('nan')):.2f}  WR: {s.get('bull_wr', float('nan')):.1%}")
        print(f"  Bear Sharpe: {s.get('bear_sharpe', float('nan')):.2f}  WR: {s.get('bear_wr', float('nan')):.1%}")

    # Save
    pd.DataFrame(overlap_records).to_csv(args.output_dir / 'overlap.csv', index=False)
    pd.DataFrame(nonoverlap_records).to_csv(args.output_dir / 'nonoverlap.csv', index=False)
    summary = {'overlap': to_serializable(ov_stats), 'nonoverlap': to_serializable(no_stats)}
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')

    print(f"\nResults saved to {args.output_dir}/")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
