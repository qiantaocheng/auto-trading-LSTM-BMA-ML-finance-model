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
    parser.add_argument('--mode', type=str, default='split',
                        choices=['split', 'walkforward', 'both'],
                        help='Test mode: split (80/20), walkforward, or both')
    parser.add_argument('--wf-init-days', type=int, default=252,
                        help='Walk-forward initial training window in trading days (~1 year)')
    parser.add_argument('--wf-step-days', type=int, default=63,
                        help='Walk-forward step size in trading days (~3 months)')
    parser.add_argument('--save-preds', type=Path, default=None,
                        help='Save WF OOS predictions to pickle for external simulation')
    parser.add_argument('--cost-bps', type=float, default=10.0,
                        help='One-way transaction cost in basis points (default 10 bps)')
    return parser.parse_args()


def ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex) and {'date', 'ticker'}.issubset(df.index.names):
        df = df.sort_index()
    elif {'date', 'ticker'}.issubset(df.columns):
        df = df.set_index(['date', 'ticker']).sort_index()
    else:
        raise ValueError('Data must contain date/ticker columns or a MultiIndex.')
    # Clip target to match BMA training pipeline (raw parquet has unclipped values)
    if 'target' in df.columns:
        before_clip = (df['target'].abs() > 0.55).sum()
        df['target'] = df['target'].clip(-0.55, 0.55)
        if before_clip > 0:
            print(f'  [CLIP] Clipped {before_clip} extreme target values to [-0.55, 0.55]')
    return df


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
                     cv_splits: int, gap: int, embargo: int, n_boost_round: int, seed: int,
                     target_col: str = 'target'
                     ) -> Tuple[lgb.Booster, int]:
    X = train_df[feature_cols].fillna(0.0).to_numpy()
    y = train_df[target_col].to_numpy()
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

    def bucket_stats(values: List[float], freq_days: int) -> Tuple[float, float, float, float, float]:
        if not values:
            return (float('nan'),) * 5
        arr = np.array(values, dtype=np.float64)
        mean = float(arr.mean())
        median = float(np.median(arr))
        wr = float(np.mean(arr > 0))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else float('nan')
        sharpe = float('nan')
        if np.isfinite(mean) and np.isfinite(std) and std > 0 and freq_days > 0:
            sharpe = float(mean / std * np.sqrt(252.0 / freq_days))
        return mean, median, wr, std, sharpe

    results = {}
    schedule = (( '', non_overlap, max(1, rebalance_days)), ('overlap_', overlap, 1))
    for prefix, data, freq in schedule:
        t1_mean, t1_med, t1_wr, t1_std, t1_sharpe = bucket_stats(data['top_1_10'], freq)
        t5_mean, t5_med, t5_wr, t5_std, t5_sharpe = bucket_stats(data['top_5_15'], freq)
        t10_mean, t10_med, t10_wr, t10_std, t10_sharpe = bucket_stats(data['top_10_20'], freq)
        results[f'{prefix}top_1_10_mean'] = t1_mean
        results[f'{prefix}top_1_10_median'] = t1_med
        results[f'{prefix}top_1_10_wr'] = t1_wr
        results[f'{prefix}top_1_10_std'] = t1_std
        results[f'{prefix}top_1_10_sharpe'] = t1_sharpe
        results[f'{prefix}top_5_15_mean'] = t5_mean
        results[f'{prefix}top_5_15_median'] = t5_med
        results[f'{prefix}top_5_15_wr'] = t5_wr
        results[f'{prefix}top_5_15_std'] = t5_std
        results[f'{prefix}top_5_15_sharpe'] = t5_sharpe
        results[f'{prefix}top_10_20_mean'] = t10_mean
        results[f'{prefix}top_10_20_median'] = t10_med
        results[f'{prefix}top_10_20_wr'] = t10_wr
        results[f'{prefix}top_10_20_std'] = t10_std
        results[f'{prefix}top_10_20_sharpe'] = t10_sharpe

    median_target = float(np.median(y)) if len(y) else float('nan')
    results['median_target'] = median_target
    results['spread'] = (results['top_1_10_mean'] - median_target
                         if np.isfinite(results['top_1_10_mean']) and np.isfinite(median_target)
                         else float('nan'))
    results['IC_mean'] = float(np.nanmean(ic_vals)) if ic_vals else float('nan')
    results['NDCG_10'] = float(np.nanmean(ndcg10)) if ndcg10 else float('nan')
    results['NDCG_20'] = float(np.nanmean(ndcg20)) if ndcg20 else float('nan')
    return results


def walk_forward(df: pd.DataFrame, feature_cols: List[str], params: Dict[str, float],
                 init_days: int, step_days: int, horizon_days: int,
                 cv_splits: int, n_boost_round: int, seed: int,
                 ema_cfg: Optional[Dict[str, float]] = None,
                 cost_bps: float = 10.0) -> Dict[str, object]:
    """Expanding-window walk-forward backtest."""
    dates = df.index.get_level_values('date').unique().sort_values()
    n_dates = len(dates)
    if init_days >= n_dates:
        raise ValueError(f'init_days ({init_days}) >= total dates ({n_dates})')

    all_preds = []
    all_targets = []
    all_dates_out = []
    all_tickers_out = []
    fold_info = []

    cursor = init_days
    fold_num = 0
    while cursor < n_dates:
        fold_num += 1
        test_end = min(cursor + step_days, n_dates)
        train_end_idx = max(0, cursor - horizon_days)  # purge gap
        train_dates_sel = dates[:train_end_idx]
        test_dates_sel = dates[cursor:test_end]

        if len(train_dates_sel) < 100 or len(test_dates_sel) == 0:
            cursor = test_end
            continue

        train_df = df.loc[(train_dates_sel, slice(None)), :]
        test_df = df.loc[(test_dates_sel, slice(None)), :]

        print(f'  [WF fold {fold_num}] train: {train_dates_sel[0].date()}..{train_dates_sel[-1].date()} '
              f'({len(train_dates_sel)}d, {len(train_df)}rows) | '
              f'test: {test_dates_sel[0].date()}..{test_dates_sel[-1].date()} '
              f'({len(test_dates_sel)}d, {len(test_df)}rows)')

        model, rounds = train_lambdarank(
            train_df, feature_cols, params,
            cv_splits=cv_splits, gap=horizon_days, embargo=horizon_days,
            n_boost_round=n_boost_round, seed=seed)

        X_test = test_df[feature_cols].fillna(0.0).to_numpy()
        preds = model.predict(X_test)
        targets = test_df['target'].to_numpy()
        test_dates_arr = test_df.index.get_level_values('date').to_numpy()
        test_tickers_arr = test_df.index.get_level_values('ticker').to_numpy()

        if ema_cfg and ema_cfg.get('length', 0) and ema_cfg.get('beta', 0):
            preds = apply_ema_smoothing(
                preds, test_dates_arr, test_tickers_arr,
                int(ema_cfg['length']), float(ema_cfg['beta']),
                max(1, int(ema_cfg.get('min_days', 1))))

        all_preds.append(preds)
        all_targets.append(targets)
        all_dates_out.append(test_dates_arr)
        all_tickers_out.append(test_tickers_arr)
        fold_info.append({
            'fold': fold_num, 'train_days': len(train_dates_sel),
            'test_days': len(test_dates_sel), 'best_rounds': rounds,
            'train_start': str(train_dates_sel[0].date()),
            'train_end': str(train_dates_sel[-1].date()),
            'test_start': str(test_dates_sel[0].date()),
            'test_end': str(test_dates_sel[-1].date()),
        })
        cursor = test_end

    # Aggregate all OOS predictions
    all_preds_cat = np.concatenate(all_preds)
    all_targets_cat = np.concatenate(all_targets)
    all_dates_cat = np.concatenate(all_dates_out)
    all_tickers_cat = np.concatenate(all_tickers_out)

    # Save predictions for external simulation comparison
    import pickle as _pkl
    _save_path = Path('data/factor_exports/_pipeline_wf_preds.pkl')
    _save_path.parent.mkdir(parents=True, exist_ok=True)
    _pkl.dump({
        'preds': all_preds_cat,
        'targets': all_targets_cat,
        'dates': all_dates_cat,
        'tickers': all_tickers_cat,
    }, open(_save_path, 'wb'))
    print(f'  [SAVE] Predictions saved to {_save_path} ({len(all_preds_cat)} rows)')

    # Build a synthetic test_df for evaluate()
    oos_df = pd.DataFrame({
        'date': all_dates_cat,
        'ticker': all_tickers_cat,
        'target': all_targets_cat,
        **{f: 0.0 for f in feature_cols},  # placeholder
    })
    oos_df = oos_df.set_index(['date', 'ticker']).sort_index()

    # Compute per-day metrics directly from aggregated predictions
    unique_dates = np.unique(all_dates_cat)
    ic_vals = []
    top10_rets = []      # raw daily bucket returns (no cost)
    top5_rets = []
    ndcg10_vals = []
    bottom10_rets = []
    top10_tickers_per_day = []  # track selected tickers for turnover cost

    for d in unique_dates:
        mask = all_dates_cat == d
        day_preds = all_preds_cat[mask]
        day_tgts = all_targets_cat[mask]
        day_tickers = all_tickers_cat[mask]
        if len(day_tgts) < 20:
            continue
        order = np.argsort(-day_preds)
        top10_rets.append(day_tgts[order[:10]].mean())
        top5_rets.append(day_tgts[order[:5]].mean())
        bottom10_rets.append(day_tgts[order[-10:]].mean())
        top10_tickers_per_day.append(set(day_tickers[order[:10]]))
        if len(day_tgts) > 1:
            ic_vals.append(float(np.corrcoef(day_preds, day_tgts)[0, 1]))
            ranks = stats.rankdata(day_tgts, method='ordinal')
            rel = (ranks - 1) / (len(ranks) - 1)
            topk = min(10, len(day_preds))
            sorted_idx = order[:topk]
            gains = (2 ** rel[sorted_idx] - 1) / np.log2(np.arange(2, topk + 2))
            ideal_idx = np.argsort(-rel)[:topk]
            ideal = (2 ** rel[ideal_idx] - 1) / np.log2(np.arange(2, topk + 2))
            denom = ideal.sum()
            ndcg10_vals.append(float(gains.sum() / denom) if denom > 0 else 0.0)

    top10_arr = np.array(top10_rets)
    top5_arr = np.array(top5_rets)
    bottom10_arr = np.array(bottom10_rets)
    ic_arr = np.array(ic_vals)

    # --- Non-overlapping 5-day rebalance with compounded equity + transaction costs ---
    cost_frac = cost_bps / 10000.0  # e.g. 10 bps = 0.001
    rebalance_freq = max(1, horizon_days)  # rebalance every horizon_days (5)

    # Collect non-overlapping rebalance-period returns and tickers
    rebal_dates_idx = list(range(0, len(unique_dates), rebalance_freq))
    nonoverlap_rets = []       # 5-day period returns (one per rebalance)
    nonoverlap_tickers = []    # tickers selected at each rebalance
    nonoverlap_dates = []      # rebalance date labels

    valid_day_idx = 0  # index into top10_rets (only incremented for days with >=20 stocks)
    date_to_valid_idx = {}
    for d in unique_dates:
        mask = all_dates_cat == d
        if np.sum(mask) >= 20:
            date_to_valid_idx[d] = valid_day_idx
            valid_day_idx += 1

    for rd_idx in rebal_dates_idx:
        d = unique_dates[rd_idx]
        if d not in date_to_valid_idx:
            continue
        vi = date_to_valid_idx[d]
        nonoverlap_rets.append(top10_arr[vi])
        nonoverlap_tickers.append(top10_tickers_per_day[vi])
        nonoverlap_dates.append(d)

    nonoverlap_arr = np.array(nonoverlap_rets)
    n_periods = len(nonoverlap_arr)

    # Build equity curve: one point per rebalance period
    equity = 1.0
    equity_curve = [equity]
    prev_tickers = set()

    for i in range(n_periods):
        # Turnover cost at each rebalance
        curr_tickers = nonoverlap_tickers[i]
        if prev_tickers:
            turnover = len(curr_tickers - prev_tickers) / 10.0  # fraction replaced
        else:
            turnover = 1.0  # initial buy = full turnover
        equity *= (1 - turnover * cost_frac * 2)  # buy + sell = 2x one-way
        prev_tickers = curr_tickers

        # Apply the 5-day period return
        equity *= (1 + nonoverlap_arr[i])
        equity_curve.append(equity)

    equity_curve = np.array(equity_curve)
    n_points = len(equity_curve)

    # Annualized compounded return (CAGR)
    # Each period = rebalance_freq trading days; total trading days = n_periods * rebalance_freq
    total_trading_days = n_periods * rebalance_freq
    if n_periods > 1 and equity_curve[-1] > 0:
        years = total_trading_days / 252.0
        cagr = float(equity_curve[-1] ** (1.0 / years) - 1)
    else:
        cagr = float('nan')

    # Sharpe from 5-day period returns (annualized)
    def _sharpe(arr, freq=1):
        if len(arr) < 2:
            return float('nan')
        m, s = arr.mean(), arr.std(ddof=1)
        return float(m / s * np.sqrt(252 / freq)) if s > 0 else float('nan')

    # Period returns after cost (from equity curve)
    period_rets_after_cost = np.diff(equity_curve) / equity_curve[:-1] if n_points > 1 else np.array([])

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve) if n_points > 0 else np.array([1.0])
    dd = (equity_curve - peak) / peak if n_points > 0 else np.array([0.0])
    max_dd = float(dd.min()) if n_points > 0 else 0.0

    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 0 and np.isfinite(cagr) else float('nan')

    metrics = {
        'mode': 'walkforward',
        'n_folds': fold_num,
        'total_oos_days': len(unique_dates),
        'total_oos_rows': len(all_preds_cat),
        'oos_date_range': [str(unique_dates[0]), str(unique_dates[-1])],
        'cost_bps': cost_bps,
        'rebalance_freq_days': rebalance_freq,
        'n_rebalance_periods': n_periods,
        # Compounded metrics (with transaction costs, 5-day rebalance)
        'top10_CAGR': cagr,
        'top10_sharpe_compounded': _sharpe(period_rets_after_cost, freq=rebalance_freq),
        'top10_max_drawdown': max_dd,
        'top10_calmar': calmar,
        'top10_final_equity': float(equity_curve[-1]) if n_points > 0 else float('nan'),
        # Non-overlapping 5-day period stats (raw, no cost)
        'top10_mean_5d_raw': float(nonoverlap_arr.mean()) if n_periods else float('nan'),
        'top10_median_5d': float(np.median(nonoverlap_arr)) if n_periods else float('nan'),
        'top10_winrate': float((nonoverlap_arr > 0).mean()) if n_periods else float('nan'),
        'top10_sharpe_raw': _sharpe(nonoverlap_arr, freq=rebalance_freq),
        # Overlapping daily stats (for IC/NDCG reference)
        'top10_mean_daily_overlap': float(top10_arr.mean()) if len(top10_arr) else float('nan'),
        'top5_mean_daily_overlap': float(top5_arr.mean()) if len(top5_arr) else float('nan'),
        'bottom10_mean_daily_overlap': float(bottom10_arr.mean()) if len(bottom10_arr) else float('nan'),
        'long_short_spread': float(top10_arr.mean() - bottom10_arr.mean()) if len(top10_arr) and len(bottom10_arr) else float('nan'),
        'IC_mean': float(ic_arr.mean()) if len(ic_arr) else float('nan'),
        'IC_std': float(ic_arr.std(ddof=1)) if len(ic_arr) > 1 else float('nan'),
        'IC_IR': float(ic_arr.mean() / ic_arr.std(ddof=1)) if len(ic_arr) > 1 and ic_arr.std(ddof=1) > 0 else float('nan'),
        'NDCG_10': float(np.nanmean(ndcg10_vals)) if ndcg10_vals else float('nan'),
        'fold_details': fold_info,
    }
    return metrics


def save_artifacts(model: lgb.Booster, metrics: Dict[str, float], args: argparse.Namespace,
                   feature_cols: List[str]) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / 'lambdarank_model.txt'))
    (out / 'features.json').write_text(json.dumps(feature_cols, indent=2), encoding='utf-8')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')


def run_split_test(df: pd.DataFrame, args: argparse.Namespace, params: Dict[str, float],
                   ema_cfg: Optional[Dict[str, float]]) -> Dict[str, object]:
    """80/20 time-split test."""
    print('=' * 80)
    print('  80/20 TIME-SPLIT TEST')
    print('=' * 80)
    train_df, test_df, train_dates = time_split(df, args.split, args.horizon_days)
    print(f'  Train: {train_dates[0].date()}..{train_dates[-1].date()} ({len(train_dates)}d, {len(train_df)}rows)')
    test_dates_idx = test_df.index.get_level_values('date').unique()
    print(f'  Test:  {test_dates_idx[0].date()}..{test_dates_idx[-1].date()} ({len(test_dates_idx)}d, {len(test_df)}rows)')

    model, rounds = train_lambdarank(train_df, args.features, params,
                                     cv_splits=args.cv_splits,
                                     gap=args.horizon_days,
                                     embargo=args.horizon_days,
                                     n_boost_round=args.n_boost_round,
                                     seed=args.seed)
    metrics = evaluate(model, test_df, args.features, args.horizon_days, ema_cfg=ema_cfg)
    metrics['mode'] = 'split_80_20'
    metrics['best_rounds'] = rounds
    metrics['train_dates'] = [str(train_dates[0]), str(train_dates[-1])]
    test_dates = test_df.index.get_level_values('date')
    metrics['test_dates'] = [str(test_dates[0]), str(test_dates[-1])]

    out = args.output_dir / 'split_80_20'
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out / 'lambdarank_model.txt'))
    (out / 'features.json').write_text(json.dumps(args.features, indent=2), encoding='utf-8')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, default=str), encoding='utf-8')
    return metrics


def run_walkforward_test(df: pd.DataFrame, args: argparse.Namespace, params: Dict[str, float],
                         ema_cfg: Optional[Dict[str, float]]) -> Dict[str, object]:
    """Expanding-window walk-forward test."""
    print('=' * 80)
    print('  WALK-FORWARD TEST')
    print(f'  init_days={args.wf_init_days}, step_days={args.wf_step_days}, cost={args.cost_bps}bps')
    print('=' * 80)
    metrics = walk_forward(df, args.features, params,
                           init_days=args.wf_init_days,
                           step_days=args.wf_step_days,
                           horizon_days=args.horizon_days,
                           cv_splits=args.cv_splits,
                           n_boost_round=args.n_boost_round,
                           seed=args.seed,
                           ema_cfg=ema_cfg,
                           cost_bps=args.cost_bps)
    out = args.output_dir / 'walkforward'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, default=str), encoding='utf-8')
    return metrics


def print_summary(label: str, metrics: Dict[str, object]) -> None:
    """Pretty-print key metrics."""
    print(f'\n{"=" * 60}')
    print(f'  {label}')
    print(f'{"=" * 60}')
    mode = metrics.get('mode', '?')
    if mode == 'split_80_20':
        print(f'  Train: {metrics.get("train_dates", "?")}')
        print(f'  Test:  {metrics.get("test_dates", "?")}')
        print(f'  Best rounds: {metrics.get("best_rounds", "?")}')
        print()
        for bucket in ('top_1_10', 'top_5_15', 'top_10_20'):
            mean_k = f'{bucket}_mean'
            wr_k = f'{bucket}_wr'
            sharpe_k = f'{bucket}_sharpe'
            print(f'  {bucket:12s}  mean={metrics.get(mean_k, 0):.4f}  '
                  f'wr={metrics.get(wr_k, 0):.1%}  sharpe={metrics.get(sharpe_k, 0):.3f}')
        print(f'\n  Overlap buckets (daily):')
        for bucket in ('top_1_10', 'top_5_15', 'top_10_20'):
            mean_k = f'overlap_{bucket}_mean'
            wr_k = f'overlap_{bucket}_wr'
            sharpe_k = f'overlap_{bucket}_sharpe'
            print(f'  {bucket:12s}  mean={metrics.get(mean_k, 0):.4f}  '
                  f'wr={metrics.get(wr_k, 0):.1%}  sharpe={metrics.get(sharpe_k, 0):.3f}')
    elif mode == 'walkforward':
        print(f'  OOS range: {metrics.get("oos_date_range", "?")}')
        print(f'  Folds: {metrics.get("n_folds", "?")}, OOS days: {metrics.get("total_oos_days", "?")}')
        print(f'  Rebalance: every {metrics.get("rebalance_freq_days", 5)}d, '
              f'{metrics.get("n_rebalance_periods", "?")} periods, '
              f'cost: {metrics.get("cost_bps", 0):.0f} bps')
        print()
        print(f'  --- Compounded ({metrics.get("rebalance_freq_days", 5)}d rebalance, {metrics.get("cost_bps", 0):.0f}bps cost) ---')
        print(f'  Top-10  CAGR={metrics.get("top10_CAGR", 0):.2%}  '
              f'Sharpe={metrics.get("top10_sharpe_compounded", 0):.3f}  '
              f'MaxDD={metrics.get("top10_max_drawdown", 0):.2%}  '
              f'Calmar={metrics.get("top10_calmar", 0):.3f}')
        print(f'  Final equity: {metrics.get("top10_final_equity", 0):.4f}')
        print()
        print(f'  --- Raw {metrics.get("rebalance_freq_days", 5)}d non-overlap (no cost) ---')
        print(f'  Top-10  mean={metrics.get("top10_mean_5d_raw", 0):.4f}  '
              f'wr={metrics.get("top10_winrate", 0):.1%}  sharpe={metrics.get("top10_sharpe_raw", 0):.3f}')
        print()
        print(f'  --- Overlap daily (reference) ---')
        print(f'  Top-10  mean={metrics.get("top10_mean_daily_overlap", 0):.4f}')
        print(f'  Top-5   mean={metrics.get("top5_mean_daily_overlap", 0):.4f}')
        print(f'  Bot-10  mean={metrics.get("bottom10_mean_daily_overlap", 0):.4f}')
        print(f'  L/S spread={metrics.get("long_short_spread", 0):.4f}')
    print(f'\n  IC mean={metrics.get("IC_mean", 0):.4f}', end='')
    if 'IC_std' in metrics:
        print(f'  IC_std={metrics.get("IC_std", 0):.4f}  IC_IR={metrics.get("IC_IR", 0):.3f}', end='')
    print(f'\n  NDCG@10={metrics.get("NDCG_10", 0):.4f}  NDCG@20={metrics.get("NDCG_20", 0):.4f}')
    print(f'  Median target: {metrics.get("median_target", 0):.4f}  Spread: {metrics.get("spread", 0):.4f}')
    print(f'{"=" * 60}\n')


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    params = _load_params(args.params_json)
    df = pd.read_parquet(args.data_file)
    df = ensure_multiindex(df)
    df = chronological_subset(df, args.time_fraction)

    dates = df.index.get_level_values('date').unique()
    print(f'Data loaded: {len(df)} rows, {len(dates)} dates '
          f'({dates[0].date()} .. {dates[-1].date()})')

    ema_cfg = None
    if args.ema_length > 1 and 0 < args.ema_beta < 1:
        ema_cfg = {'length': args.ema_length, 'beta': args.ema_beta, 'min_days': max(1, args.ema_min_days)}

    all_results = {}

    if args.mode in ('split', 'both'):
        split_metrics = run_split_test(df, args, params, ema_cfg)
        print_summary('80/20 SPLIT RESULTS', split_metrics)
        all_results['split_80_20'] = split_metrics

    if args.mode in ('walkforward', 'both'):
        wf_metrics = run_walkforward_test(df, args, params, ema_cfg)
        print_summary('WALK-FORWARD RESULTS', wf_metrics)
        all_results['walkforward'] = wf_metrics

    # Save combined results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'all_results.json').write_text(
        json.dumps(all_results, indent=2, default=str), encoding='utf-8')

    print(json.dumps(all_results, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
