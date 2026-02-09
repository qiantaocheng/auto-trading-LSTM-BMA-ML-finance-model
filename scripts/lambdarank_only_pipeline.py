#!/usr/bin/env python3
"""Standalone LambdaRank-only training/eval pipeline on multi-index factors."""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

FEATURES = [
    'volume_price_corr_3d',
    'rsi_14',
    'reversal_3d',
    'momentum_10d',
    'liquid_momentum_10d',
    'sharpe_momentum_5d',
    'price_ma20_deviation',
    'avg_trade_size',
    'trend_r2_20',
    'dollar_vol_20',
    'ret_skew_20d',
    'reversal_5d',
    'near_52w_high',
    'atr_pct_14',
    'amihud_20',
]

BEST_PARAMS = {
    'learning_rate': 0.04,
    'num_leaves': 11,
    'max_depth': 3,
    'min_data_in_leaf': 350,
    'lambda_l2': 120,
    'feature_fraction': 1.0,
    'bagging_fraction': 0.70,
    'bagging_freq': 1,
    'min_gain_to_split': 0.30,
    'lambdarank_truncation_level': 25,
    'sigmoid': 1.1,
    'label_gain_power': 2.1,
}

COMMON = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [10, 20],
    'n_quantiles': 64,
    'early_stopping_rounds': 50,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pure LambdaRank runner with purged CV')
    parser.add_argument('--data-file', type=Path,
                        default=Path('data/factor_exports/polygon_factors_all_2021_2026_T5_final.parquet'))
    parser.add_argument('--time-fraction', type=float, default=1.0)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--horizon-days', type=int, default=5)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/lambdarank_only_runs'))
    parser.add_argument('--features', nargs='+', default=FEATURES)
    parser.add_argument('--n-boost-round', type=int, default=800)
    parser.add_argument('--seed', type=int, default=0, help='Random seed for LightGBM and bagging')
    parser.add_argument('--params-json', type=str, default=None,
                        help='JSON string or file overriding LambdaRank params')
    parser.add_argument('--ema-length', type=int, default=0,
                        help='EMA window length (0 disables smoothing)')
    parser.add_argument('--ema-beta', type=float, default=0.0,
                        help='EMA decay factor (0<beta<1)')
    parser.add_argument('--ema-min-days', type=int, default=1,
                        help='Minimum observations (current+history) required before applying EMA')
    return parser.parse_args()


def ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex) and {'date', 'ticker'}.issubset(df.index.names):
        return df.sort_index()
    if {'date', 'ticker'}.issubset(df.columns):
        return df.set_index(['date', 'ticker']).sort_index()
    raise ValueError('Data must contain date/ticker columns or a MultiIndex.')


def chronological_subset(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if fraction >= 1.0:
        return df
    dates = df.index.get_level_values('date').unique()
    n_keep = max(1, int(len(dates) * fraction))
    selected = dates[:n_keep]
    return df.loc[(selected, slice(None)), :].copy()


def time_split(df: pd.DataFrame, split: float, purge_gap: int
               ) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.Timestamp]]:
    dates = df.index.get_level_values('date').unique()
    split_idx = int(len(dates) * split)
    train_end_idx = max(0, split_idx - purge_gap)
    train_dates = dates[:train_end_idx]
    test_dates = dates[split_idx:]
    if len(train_dates) == 0 or len(test_dates) == 0:
        raise ValueError('Train/test windows are empty. Adjust split or gap.')
    train_df = df.loc[(train_dates, slice(None)), :].copy()
    test_df = df.loc[(test_dates, slice(None)), :].copy()
    return train_df, test_df, list(train_dates)


def _ema_weights(beta: float, count: int) -> np.ndarray:
    weights = np.array([beta * ((1 - beta) ** i) for i in range(count)], dtype=np.float64)
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        weights = np.zeros(count, dtype=np.float64)
        weights[0] = 1.0
        total = 1.0
    return weights / total


def apply_ema_smoothing(preds: np.ndarray, dates: np.ndarray, tickers: np.ndarray,
                        length: int, beta: float, min_days: int) -> np.ndarray:
    if length <= 1 or not (0 < beta < 1):
        return preds
    smoothed = preds.copy()
    history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=length - 1))
    unique_dates = np.unique(dates)
    for d in unique_dates:
        mask = np.where(dates == d)[0]
        for idx in mask:
            ticker = tickers[idx]
            past_vals = list(history[ticker])
            values = [preds[idx]] + past_vals
            if len(values) >= min_days:
                weights = _ema_weights(beta, len(values))
                smoothed[idx] = float(np.dot(weights, values))
            else:
                smoothed[idx] = preds[idx]
            history[ticker].appendleft(preds[idx])
    return smoothed


def build_quantile_labels(y: np.ndarray, dates: np.ndarray, n_quantiles: int) -> np.ndarray:
    labels = np.zeros(len(y), dtype=np.int32)
    for d in np.unique(dates):
        mask = dates == d
        if np.sum(mask) <= 1:
            continue
        values = y[mask]
        ranks = stats.rankdata(values, method='average')
        quantiles = np.floor(ranks / (len(values) + 1) * n_quantiles).astype(np.int32)
        labels[mask] = np.clip(quantiles, 0, n_quantiles - 1)
    return labels


def group_counts(dates: np.ndarray) -> List[int]:
    return [int(np.sum(dates == d)) for d in np.unique(dates)]


def purged_cv_splits(dates: np.ndarray, n_splits: int, gap: int, embargo: int
                     ) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    unique_dates = np.unique(dates)
    n_dates = len(unique_dates)
    fold_size = max(1, n_dates // n_splits)
    for fold in range(n_splits):
        val_start = fold * fold_size
        val_end = n_dates if fold == n_splits - 1 else (fold + 1) * fold_size
        val_dates = unique_dates[val_start:val_end]
        train_end = max(0, val_start - gap)
        embargo_start = min(n_dates, val_end + embargo)
        train_dates = np.concatenate((unique_dates[:train_end], unique_dates[embargo_start:]))
        train_mask = np.isin(dates, train_dates)
        val_mask = np.isin(dates, val_dates)
        if np.sum(train_mask) < 100 or np.sum(val_mask) < 50:
            continue
        yield np.where(train_mask)[0], np.where(val_mask)[0]


def _load_params(params_json: str) -> Dict[str, float]:
    params = BEST_PARAMS.copy()
    if not params_json:
        return params
    candidate = Path(params_json)
    if candidate.exists():
        overrides = json.loads(candidate.read_text(encoding='utf-8'))
    else:
        overrides = json.loads(params_json)
    params.update(overrides)
    return params


def train_lambdarank(train_df: pd.DataFrame, feature_cols: List[str], params: Dict[str, float],
                     cv_splits: int, gap: int, embargo: int, n_boost_round: int, seed: int
                     ) -> Tuple[lgb.Booster, int]:
    X = train_df[feature_cols].fillna(0.0).to_numpy()
    y = train_df['target'].to_numpy()
    dates = train_df.index.get_level_values('date').to_numpy()
    labels = build_quantile_labels(y, dates, COMMON['n_quantiles'])

    np.random.seed(seed)

    lgb_params = {
        'objective': COMMON['objective'],
        'metric': COMMON['metric'],
        'ndcg_eval_at': COMMON['ndcg_eval_at'],
        'learning_rate': params['learning_rate'],
        'num_leaves': params['num_leaves'],
        'max_depth': params['max_depth'],
        'min_data_in_leaf': params['min_data_in_leaf'],
        'lambda_l1': 0.0,
        'lambda_l2': params['lambda_l2'],
        'feature_fraction': params['feature_fraction'],
        'bagging_fraction': params['bagging_fraction'],
        'bagging_freq': params['bagging_freq'],
        'min_gain_to_split': params['min_gain_to_split'],
        'lambdarank_truncation_level': params['lambdarank_truncation_level'],
        'sigmoid': params['sigmoid'],
        'verbose': -1,
        'force_row_wise': True,
        'seed': seed,
        'bagging_seed': seed,
        'feature_fraction_seed': seed,
        'data_random_seed': seed,
        'deterministic': True,
    }
    label_gain = [(i / (COMMON['n_quantiles'] - 1)) ** params['label_gain_power']
                  * (COMMON['n_quantiles'] - 1)
                  for i in range(COMMON['n_quantiles'])]
    lgb_params['label_gain'] = label_gain

    best_rounds: List[int] = []
    for train_idx, val_idx in purged_cv_splits(dates, cv_splits, gap, embargo):
        train_set = lgb.Dataset(X[train_idx], label=labels[train_idx], group=group_counts(dates[train_idx]))
        val_set = lgb.Dataset(X[val_idx], label=labels[val_idx], group=group_counts(dates[val_idx]))
        booster = lgb.train(lgb_params, train_set, num_boost_round=n_boost_round,
                            valid_sets=[val_set],
                            callbacks=[lgb.early_stopping(COMMON['early_stopping_rounds'], verbose=False)])
        best_rounds.append(booster.best_iteration or n_boost_round)
    final_rounds = int(np.mean(best_rounds)) if best_rounds else n_boost_round
    full_set = lgb.Dataset(X, label=labels, group=group_counts(dates))
    final_model = lgb.train(lgb_params, full_set, num_boost_round=final_rounds)
    return final_model, final_rounds


def evaluate(model: lgb.Booster, test_df: pd.DataFrame, feature_cols: List[str],
             rebalance_days: int, ema_cfg: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    X = test_df[feature_cols].fillna(0.0).to_numpy()
    y = test_df['target'].to_numpy()
    dates = test_df.index.get_level_values('date').to_numpy()
    tickers = test_df.index.get_level_values('ticker').to_numpy()
    preds = model.predict(X)
    if ema_cfg and ema_cfg.get('length', 0) and ema_cfg.get('beta', 0):
        preds = apply_ema_smoothing(
            preds,
            dates,
            tickers,
            int(ema_cfg.get('length', 0)),
            float(ema_cfg.get('beta', 0)),
            max(1, int(ema_cfg.get('min_days', 1)))
        )
    unique_dates = np.unique(dates)

    non_overlap = {'top_1_10': [], 'top_5_15': [], 'top_10_20': []}
    overlap = {'top_1_10': [], 'top_5_15': [], 'top_10_20': []}
    ndcg10: List[float] = []
    ndcg20: List[float] = []
    ic_vals: List[float] = []

    def bucket_values(day_scores: np.ndarray, day_targets: np.ndarray):
        order = np.argsort(-day_scores)
        return (day_targets[order[:10]].mean(),
                day_targets[order[4:15]].mean(),
                day_targets[order[9:20]].mean(),
                order)

    for d in unique_dates:
        mask = dates == d
        day_scores = preds[mask]
        day_targets = y[mask]
        if len(day_targets) < 20:
            continue
        top1, top5, top10, order = bucket_values(day_scores, day_targets)
        overlap['top_1_10'].append(top1)
        overlap['top_5_15'].append(top5)
        overlap['top_10_20'].append(top10)
        if len(day_targets) > 1:
            ic_vals.append(float(np.corrcoef(day_scores, day_targets)[0, 1]))
            ranks = stats.rankdata(day_targets, method='ordinal')
            rel = (ranks - 1) / (len(ranks) - 1)
            for k, store in ((10, ndcg10), (20, ndcg20)):
                topk = min(k, len(day_scores))
                sorted_idx = order[:topk]
                gains = (2 ** rel[sorted_idx] - 1) / np.log2(np.arange(2, topk + 2))
                ideal_idx = np.argsort(-rel)[:topk]
                ideal = (2 ** rel[ideal_idx] - 1) / np.log2(np.arange(2, topk + 2))
                denom = ideal.sum()
                store.append(float(gains.sum() / denom) if denom > 0 else 0.0)

    for d in unique_dates[::max(1, rebalance_days)]:
        mask = dates == d
        day_preds = preds[mask]
        day_tgts = y[mask]
        if len(day_tgts) < 20:
            continue
        top1, top5, top10, _ = bucket_values(day_preds, day_tgts)
        non_overlap['top_1_10'].append(top1)
        non_overlap['top_5_15'].append(top5)
        non_overlap['top_10_20'].append(top10)

    def bucket_stats(values: List[float]) -> Tuple[float, float, float]:
        if not values:
            return float('nan'), float('nan'), float('nan')
        arr = np.array(values)
        return float(arr.mean()), float(np.median(arr)), float(np.mean(arr > 0))

    results = {}
    for prefix, data in (('', non_overlap), ('overlap_', overlap)):
        t1_mean, t1_med, t1_wr = bucket_stats(data['top_1_10'])
        t5_mean, t5_med, t5_wr = bucket_stats(data['top_5_15'])
        t10_mean, t10_med, t10_wr = bucket_stats(data['top_10_20'])
        results[f'{prefix}top_1_10_mean'] = t1_mean
        results[f'{prefix}top_1_10_median'] = t1_med
        results[f'{prefix}top_1_10_wr'] = t1_wr
        results[f'{prefix}top_5_15_mean'] = t5_mean
        results[f'{prefix}top_5_15_median'] = t5_med
        results[f'{prefix}top_5_15_wr'] = t5_wr
        results[f'{prefix}top_10_20_mean'] = t10_mean
        results[f'{prefix}top_10_20_median'] = t10_med
        results[f'{prefix}top_10_20_wr'] = t10_wr

    median_target = float(np.median(y)) if len(y) else float('nan')
    results['median_target'] = median_target
    results['spread'] = (results['top_1_10_mean'] - median_target
                         if np.isfinite(results['top_1_10_mean']) and np.isfinite(median_target)
                         else float('nan'))
    results['IC_mean'] = float(np.nanmean(ic_vals)) if ic_vals else float('nan')
    results['NDCG_10'] = float(np.nanmean(ndcg10)) if ndcg10 else float('nan')
    results['NDCG_20'] = float(np.nanmean(ndcg20)) if ndcg20 else float('nan')
    return results


def save_artifacts(model: lgb.Booster, metrics: Dict[str, float], args: argparse.Namespace,
                   feature_cols: List[str]) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / 'lambdarank_model.txt'))
    (out / 'features.json').write_text(json.dumps(feature_cols, indent=2), encoding='utf-8')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    params = _load_params(args.params_json)
    df = pd.read_parquet(args.data_file)
    df = ensure_multiindex(df)
    df = chronological_subset(df, args.time_fraction)
    train_df, test_df, train_dates = time_split(df, args.split, args.horizon_days)
    model, rounds = train_lambdarank(train_df, args.features, params,
                                     cv_splits=args.cv_splits,
                                     gap=args.horizon_days,
                                     embargo=args.horizon_days,
                                     n_boost_round=args.n_boost_round,
                                     seed=args.seed)
    ema_cfg = None
    if args.ema_length > 1 and 0 < args.ema_beta < 1:
        ema_cfg = {'length': args.ema_length, 'beta': args.ema_beta, 'min_days': max(1, args.ema_min_days)}
    metrics = evaluate(model, test_df, args.features, args.horizon_days, ema_cfg=ema_cfg)
    metrics['best_rounds'] = rounds
    metrics['ema_length'] = args.ema_length
    metrics['ema_beta'] = args.ema_beta
    metrics['ema_min_days'] = args.ema_min_days
    metrics['train_dates'] = [str(train_dates[0]), str(train_dates[-1])]
    test_dates = test_df.index.get_level_values('date')
    metrics['test_dates'] = [str(test_dates[0]), str(test_dates[-1])]
    save_artifacts(model, metrics, args, args.features)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
