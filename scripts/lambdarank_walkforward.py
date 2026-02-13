#!/usr/bin/env python3
"""Leakage-safe walk-forward runner that reuses the LambdaRank-only stack."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.lambdarank_only_pipeline import (
    FEATURES as DEFAULT_FEATURES,
    ensure_multiindex,
    chronological_subset,
    train_lambdarank,
    evaluate,
    _load_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Walk-forward LambdaRank evaluation with purged windows'
    )
    parser.add_argument(
        '--data-file',
        type=Path,
        default=Path('data/factor_exports/polygon_full_features_T5.parquet'),
        help='Multi-index parquet (date, ticker) holding factors + target',
    )
    parser.add_argument('--time-fraction', type=float, default=1.0,
                        help='Use earliest X% of timeline (default: all)')
    parser.add_argument('--train-window-days', type=int, default=504,
                        help='Trading days per training window (default two years)')
    parser.add_argument('--test-window-days', type=int, default=63,
                        help='Trading days per test window (default one quarter)')
    parser.add_argument('--step-days', type=int, default=63,
                        help='Stride when rolling the window (default = test window)')
    parser.add_argument('--gap-days', type=int, default=5,
                        help='Days skipped between train and test to cover label horizon')
    parser.add_argument('--cv-splits', type=int, default=5,
                        help='Purged CV splits inside each training call')
    parser.add_argument('--n-boost-round', type=int, default=800)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--features', nargs='+', default=None,
                        help='Override feature columns (default uses Lambda list)')
    parser.add_argument('--params-json', type=str, default=None,
                        help='JSON string/path overriding LightGBM params')
    parser.add_argument('--max-windows', type=int, default=None,
                        help='Optional cap for the number of walk-forward windows')
    parser.add_argument('--rebalance-days', type=int, default=5,
                        help='Rebalance spacing passed to evaluate()')
    parser.add_argument('--ema-length', type=int, default=0)
    parser.add_argument('--ema-beta', type=float, default=0.0)
    parser.add_argument('--ema-min-days', type=int, default=1)
    parser.add_argument('--pit-min-price', type=float, default=5.0,
                        help='PIT filter: min median close price (0 disables)')
    parser.add_argument('--pit-min-dollar-vol', type=float, default=500_000.0,
                        help='PIT filter: min median daily dollar volume (0 disables)')
    parser.add_argument('--pit-max-tickers', type=int, default=None,
                        help='PIT filter: keep only top N tickers by dollar volume')
    parser.add_argument('--target-col', type=str, default='target',
                        help='Column to use for labeling (default: target)')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/lambdarank_walkforward'))
    return parser.parse_args()


def iter_time_windows(dates: Sequence[pd.Timestamp], train_days: int,
                      gap_days: int, test_days: int, step_days: int
                      ) -> Iterable[Tuple[int, int, int, int]]:
    total = len(dates)
    start = 0
    while True:
        train_end = start + train_days
        test_start = train_end + gap_days
        test_end = test_start + test_days
        if test_end > total:
            break
        yield start, train_end, test_start, test_end
        start += max(1, step_days)


def slice_by_date(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(slice(start, end), slice(None)), :].copy()


def pit_universe_filter(df: pd.DataFrame, train_end: pd.Timestamp,
                        min_price: float = 5.0,
                        min_dollar_vol: float = 500_000.0,
                        lookback_days: int = 20,
                        max_tickers: int | None = None) -> pd.DataFrame:
    """Point-in-time universe filter to prevent survivorship bias.

    Uses the last ``lookback_days`` trading days of the training period to
    compute median Close and median daily dollar volume per ticker.  Tickers
    that fail the price or liquidity screen are dropped from the *entire*
    DataFrame (train + test windows share the same universe for consistency).

    If ``max_tickers`` is set, keeps only top N tickers by median dollar volume
    after applying minimum thresholds.

    Only filters if ``Close`` and ``Volume`` columns are present; otherwise
    returns the DataFrame unchanged (backward compatible).
    """
    has_close = 'Close' in df.columns
    has_volume = 'Volume' in df.columns
    if not has_close:
        return df

    all_dates = df.index.get_level_values('date').unique().sort_values()
    window_dates = all_dates[all_dates <= train_end][-lookback_days:]
    if len(window_dates) == 0:
        return df

    window_data = df.loc[(slice(window_dates[0], window_dates[-1]), slice(None)), :]

    # Median close per ticker over the lookback window
    med_close = window_data.groupby(level='ticker')['Close'].median()
    valid_tickers = set(med_close[med_close >= min_price].index)

    # Median daily dollar volume filter (Close * Volume)
    med_dvol = None
    if has_volume:
        dollar_vol = (window_data['Close'] * window_data['Volume'])
        med_dvol = dollar_vol.groupby(level='ticker').median()
        valid_tickers &= set(med_dvol[med_dvol >= min_dollar_vol].index)

    # Top-N by dollar volume
    if max_tickers and has_volume and med_dvol is not None and len(valid_tickers) > max_tickers:
        top_n = med_dvol.loc[med_dvol.index.isin(valid_tickers)].nlargest(max_tickers).index
        valid_tickers = set(top_n)

    before = df.index.get_level_values('ticker').nunique()
    keep_mask = df.index.get_level_values('ticker').isin(valid_tickers)
    df = df.loc[keep_mask].sort_index()
    after = df.index.get_level_values('ticker').nunique()
    if before != after:
        msg = f"  PIT filter: {before} -> {after} tickers (dropped {before - after}"
        msg += f", min_price=${min_price}, min_dvol=${min_dollar_vol:,.0f}"
        if max_tickers:
            msg += f", max_tickers={max_tickers}"
        msg += ")"
        print(msg)
    return df


def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    return obj


def summarize_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}
    df = pd.DataFrame(records)
    numeric = df.select_dtypes(include=[float, int])
    summary: Dict[str, float] = {'n_windows': len(records)}
    for col in numeric.columns:
        summary[f'{col}_mean'] = float(numeric[col].mean())
        summary[f'{col}_median'] = float(numeric[col].median())
    return summary


def main() -> int:
    args = parse_args()
    if args.train_window_days <= 0 or args.test_window_days <= 0:
        raise ValueError('Train/test windows must be positive lengths')
    df = ensure_multiindex(pd.read_parquet(args.data_file))
    df = chronological_subset(df, args.time_fraction)
    dates = df.index.get_level_values('date').unique().sort_values()
    if len(dates) < args.train_window_days + args.gap_days + args.test_window_days:
        raise ValueError('Not enough dates for the requested configuration')
    features = args.features or DEFAULT_FEATURES
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise ValueError(f'Missing feature columns: {missing}')
    params = _load_params(args.params_json)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ema_cfg = None
    if args.ema_length > 1 and 0 < args.ema_beta < 1:
        ema_cfg = {
            'length': args.ema_length,
            'beta': args.ema_beta,
            'min_days': max(1, args.ema_min_days),
        }

    records: List[Dict[str, float]] = []
    metrics_rows: List[Dict[str, object]] = []
    for win_id, (train_start, train_end, test_start, test_end) in enumerate(
        iter_time_windows(dates, args.train_window_days, args.gap_days,
                          args.test_window_days, args.step_days)
    ):
        if args.max_windows is not None and win_id >= args.max_windows:
            break
        train_range = (dates[train_start], dates[train_end - 1])
        test_range = (dates[test_start], dates[test_end - 1])
        train_df = slice_by_date(df, train_range[0], train_range[1])
        test_df = slice_by_date(df, test_range[0], test_range[1])
        if train_df.empty or test_df.empty:
            continue

        # Point-in-time universe filter (survivorship bias fix)
        use_pit = (args.pit_min_price > 0 or args.pit_min_dollar_vol > 0)
        if use_pit:
            valid_tickers_before = train_df.index.get_level_values('ticker').nunique()
            combined = pd.concat([train_df, test_df])
            combined = pit_universe_filter(
                combined, train_range[1],
                min_price=args.pit_min_price,
                min_dollar_vol=args.pit_min_dollar_vol,
                max_tickers=args.pit_max_tickers,
            )
            train_df = slice_by_date(combined, train_range[0], train_range[1])
            test_df = slice_by_date(combined, test_range[0], test_range[1])
            if train_df.empty or test_df.empty:
                continue

        model, best_rounds = train_lambdarank(
            train_df,
            features,
            params,
            cv_splits=args.cv_splits,
            gap=args.gap_days,
            embargo=args.gap_days,
            n_boost_round=args.n_boost_round,
            seed=args.seed + win_id,
            target_col=args.target_col,
        )
        metrics = evaluate(
            model,
            test_df,
            features,
            rebalance_days=args.rebalance_days,
            ema_cfg=ema_cfg,
        )
        metrics['window_id'] = win_id
        metrics['train_start'] = train_range[0]
        metrics['train_end'] = train_range[1]
        metrics['test_start'] = test_range[0]
        metrics['test_end'] = test_range[1]
        metrics['train_rows'] = int(len(train_df))
        metrics['test_rows'] = int(len(test_df))
        metrics['best_rounds'] = int(best_rounds)
        numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        records.append(numeric_metrics)
        metrics_rows.append(metrics)
        print(f"Finished window {win_id}: train {train_range[0]} -> {train_range[1]}, "
              f"test {test_range[0]} -> {test_range[1]}")

    if not metrics_rows:
        raise RuntimeError('No completed windows; adjust configuration')

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / 'walkforward_metrics.csv', index=False)
    summary = summarize_metrics(records)
    (output_dir / 'walkforward_metrics.json').write_text(
        json.dumps(to_serializable(metrics_rows), indent=2),
        encoding='utf-8',
    )
    (output_dir / 'walkforward_summary.json').write_text(
        json.dumps(to_serializable(summary), indent=2),
        encoding='utf-8',
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
