#!/usr/bin/env python3
"""Learn actionable thresholds by combining BMA ridge backtest results with HETRS Nasdaq signals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def load_bma_predictions(path: Path, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    grouped = df.groupby('date')

    top_rows: List[Dict] = []
    rank_rows: List[Dict] = []

    for date, group in grouped:
        sorted_df = group.sort_values('prediction', ascending=False).head(top_n).copy()
        if sorted_df.empty:
            continue
        stats = {
            'date': date,
            'top_avg_prediction': float(sorted_df['prediction'].mean()),
            'top_median_prediction': float(sorted_df['prediction'].median()),
            'top_min_prediction': float(sorted_df['prediction'].min()),
            'top_max_prediction': float(sorted_df['prediction'].max()),
            'top_avg_actual': float(sorted_df['actual'].mean()),
            'top_positive_rate': float((sorted_df['actual'] > 0).mean()),
        }
        top_rows.append(stats)

        for rank, row in enumerate(sorted_df.itertuples(index=False), start=1):
            rank_rows.append({
                'date': date,
                'rank': rank,
                'ticker': row.ticker,
                'prediction': float(row.prediction),
                'actual': float(row.actual),
            })

    top_stats = pd.DataFrame(top_rows).sort_values('date').reset_index(drop=True)
    rank_details = pd.DataFrame(rank_rows).sort_values(['date', 'rank']).reset_index(drop=True)
    return top_stats, rank_details


def load_hetrs_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = None
    for candidate in ['date', 'Date', 'Unnamed: 0']:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError('HETRS timeseries must contain a date column (date/Date/Unnamed: 0).')

    df = df.rename(columns={date_col: 'date'})
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    keep_cols = ['date'] + [col for col in df.columns if col != 'date']
    return df[keep_cols].sort_values('date').reset_index(drop=True)


def label_events(position_series: pd.Series) -> pd.Series:
    prev = position_series.shift(1)
    curr = position_series
    events = []
    for p, c in zip(prev, curr):
        if pd.isna(p):
            events.append('init')
        elif p <= 0 and c > 0:
            events.append('enter_long')
        elif p > 0 and c <= 0:
            events.append('exit_long')
        elif p >= 0 and c < 0:
            events.append('enter_short')
        elif p < 0 and c >= 0:
            events.append('exit_short')
        else:
            events.append('hold')
    return pd.Series(events, index=position_series.index)


def compute_rank_allocation(rank_details: pd.DataFrame, positions: pd.DataFrame) -> pd.Series:
    merged = rank_details.merge(positions[['date', 'position']], on='date', how='left')
    merged = merged.dropna(subset=['position'])
    merged = merged.loc[merged['position'] > 0]
    if merged.empty:
        return pd.Series(dtype=float)

    merged['positive_pred'] = merged['prediction'].clip(lower=0)

    def _weights(group: pd.DataFrame) -> pd.Series:
        preds = group['positive_pred'].values
        if np.all(preds == 0):
            w = np.ones_like(preds) / len(preds)
        else:
            w = preds / preds.sum()
        return pd.Series(w, index=group.index)

    merged['implied_weight'] = merged.groupby('date', group_keys=False).apply(_weights)
    allocation = merged.groupby('rank')['implied_weight'].mean()
    return allocation / allocation.sum()


def aggregate_rank_performance(rank_details: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    merged = rank_details.merge(positions[['date', 'position']], on='date', how='left')
    merged = merged.dropna(subset=['position'])
    stats = merged.groupby('rank').agg(
        avg_prediction=('prediction', 'mean'),
        avg_actual=('actual', 'mean'),
        win_rate=('actual', lambda x: float((x > 0).mean())),
    )
    return stats


def cumulative_topk_returns(rank_details: pd.DataFrame, positions: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rank_details = rank_details.copy()
    position_map = positions[['date', 'position']].set_index('date')['position']
    rank_details['position'] = rank_details['date'].map(position_map)
    long_dates = rank_details.loc[rank_details['position'] > 0, 'date'].unique()

    rows = []
    for k in range(1, top_n + 1):
        subset = rank_details.loc[(rank_details['rank'] <= k) & (rank_details['date'].isin(long_dates))]
        if subset.empty:
            avg_ret = np.nan
            win_rate = np.nan
        else:
            daily_avg = subset.groupby('date')['actual'].mean()
            avg_ret = float(daily_avg.mean())
            win_rate = float((daily_avg > 0).mean())
        rows.append({'k': k, 'avg_return': avg_ret, 'win_rate': win_rate})
    return pd.DataFrame(rows)


def summarize_thresholds(top_stats: pd.DataFrame, positions: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    merged = top_stats.merge(positions[['date', 'position']], on='date', how='inner').sort_values('date').reset_index(drop=True)
    merged['event'] = label_events(merged['position'])

    def summarize_event(event: str) -> Dict[str, float]:
        subset = merged.loc[merged['event'] == event]
        if subset.empty:
            return {}
        return {
            'count': int(len(subset)),
            'top_avg_prediction': float(subset['top_avg_prediction'].median()),
            'top_min_prediction': float(subset['top_min_prediction'].median()),
            'hetrs_position_abs': float(subset['position'].abs().median()),
        }

    summary = {
        'enter_long': summarize_event('enter_long'),
        'exit_long': summarize_event('exit_long'),
        'enter_short': summarize_event('enter_short'),
        'exit_short': summarize_event('exit_short'),
    }

    for key, info in list(summary.items()):
        if not info:
            summary.pop(key)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Learn thresholds tying BMA ridge ranks to HETRS Nasdaq signals.')
    parser.add_argument('--bma-predictions', type=Path, required=True)
    parser.add_argument('--hetrs-timeseries', type=Path, required=True)
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--output-dir', type=Path, default=Path('result/hetrs_learning'))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    top_stats, rank_details = load_bma_predictions(args.bma_predictions, args.top_n)
    hetrs_df = load_hetrs_timeseries(args.hetrs_timeseries)
    if 'position' not in hetrs_df.columns:
        raise ValueError("HETRS timeseries must contain a 'position' column")
    hetrs_df['position'] = hetrs_df['position'].abs()

    thresholds = summarize_thresholds(top_stats, hetrs_df)
    allocation = compute_rank_allocation(rank_details, hetrs_df)
    rank_perf = aggregate_rank_performance(rank_details, hetrs_df)
    topk_curve = cumulative_topk_returns(rank_details, hetrs_df, args.top_n)

    recommended_k = None
    if topk_curve['avg_return'].notna().any():
        recommended_k = int(topk_curve.loc[topk_curve['avg_return'].idxmax(), 'k'])

    summary = {
        'bma_prediction_file': str(args.bma_predictions),
        'hetrs_timeseries_file': str(args.hetrs_timeseries),
        'top_n': args.top_n,
        'event_thresholds': thresholds,
        'allocation_per_rank': allocation.to_dict() if not allocation.empty else {},
        'recommended_k_by_avg_return': recommended_k,
    }

    (output_dir / 'learning_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    top_stats.to_csv(output_dir / 'daily_top_stats.csv', index=False)
    rank_details.to_csv(output_dir / 'rank_details.csv', index=False)
    rank_perf.to_csv(output_dir / 'rank_performance.csv')
    topk_curve.to_csv(output_dir / 'topk_curve.csv', index=False)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
