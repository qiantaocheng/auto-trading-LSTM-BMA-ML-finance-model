#!/usr/bin/env python
"""
Patience Scan Experiment for LambdaRank

Test patience = 50, 100, 200 on best config (B2_V2) to determine:
1. Is best_iter significantly different?
2. Does Test Spread/NDCG improve?
3. Any overfitting signals?
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

TRADE_DIR = Path(r"D:\trade")
DATA_FILE = TRADE_DIR / "data" / "factor_exports" / "polygon_factors_all_2021_2026_T5_final.parquet"

os.chdir(TRADE_DIR)
sys.path.insert(0, str(TRADE_DIR))

import lightgbm as lgb
from scipy import stats

# B2_V2 config (best from grid search)
B2_V2_PARAMS = {
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_data_in_leaf': 150,
    'lambda_l2': 30.0,
    'min_gain_to_split': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.75,
    'bagging_freq': 3,
    'lambdarank_truncation_level': 40,
    'sigmoid': 1.1,
    'label_gain_power': 2.1,
}

FEATURES = [
    "volume_price_corr_10d", "rsi_14", "reversal_3d", "momentum_10d",
    "liquid_momentum_10d", "sharpe_momentum_5d", "price_ma20_deviation",
    "avg_trade_size", "trend_r2_20", "obv_divergence"
]

def load_and_split_data(time_fraction=0.7, split_ratio=0.8, horizon_days=5):
    """Load data with 70% time, 80/20 split, 5-day purge"""
    df = pd.read_parquet(DATA_FILE)

    dates = df.index.get_level_values('date')
    all_unique_dates = sorted(dates.unique())

    n_dates_to_use = int(len(all_unique_dates) * time_fraction)
    selected_dates = all_unique_dates[:n_dates_to_use]

    mask = df.index.get_level_values('date').isin(selected_dates)
    df = df.loc[mask].copy()

    n_dates = len(selected_dates)
    split_idx = int(n_dates * split_ratio)
    purge_gap = horizon_days
    train_end_idx = split_idx - 1 - purge_gap

    train_dates = selected_dates[:train_end_idx + 1]
    test_dates = selected_dates[split_idx:]

    train_mask = df.index.get_level_values('date').isin(train_dates)
    test_mask = df.index.get_level_values('date').isin(test_dates)

    return df.loc[train_mask].copy(), df.loc[test_mask].copy(), train_dates

def train_with_logging(train_df, params, patience, n_quantiles=32):
    """Train LambdaRank with detailed logging of NDCG curve"""

    X_all = train_df[FEATURES].values
    y_all = train_df['target'].values
    X_all = np.nan_to_num(X_all, nan=0.0)
    y_all = np.nan_to_num(y_all, nan=0.0)

    dates = train_df.index.get_level_values('date').values
    unique_dates = sorted(np.unique(dates))
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    sample_date_idx = np.array([date_to_idx[d] for d in dates])

    # Convert to ranking labels
    labels_all = np.zeros(len(y_all), dtype=np.int32)
    for d in unique_dates:
        mask = (dates == d)
        day_targets = y_all[mask]
        if len(day_targets) > 1:
            ranks = stats.rankdata(day_targets, method='average')
            quantile_labels = np.floor(ranks / (len(ranks) + 1) * n_quantiles).astype(np.int32)
            quantile_labels = np.clip(quantile_labels, 0, n_quantiles - 1)
            labels_all[mask] = quantile_labels

    # Label gain
    label_gain_power = params['label_gain_power']
    label_gain = [(i / (n_quantiles - 1)) ** label_gain_power * (n_quantiles - 1)
                  for i in range(n_quantiles)]

    lgb_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [10, 20],
        'label_gain': label_gain,
        'num_leaves': params['num_leaves'],
        'max_depth': params['max_depth'],
        'learning_rate': params['learning_rate'],
        'feature_fraction': params['feature_fraction'],
        'bagging_fraction': params['bagging_fraction'],
        'bagging_freq': params['bagging_freq'],
        'min_data_in_leaf': params['min_data_in_leaf'],
        'lambda_l1': 0.0,
        'lambda_l2': params['lambda_l2'],
        'min_gain_to_split': params.get('min_gain_to_split', 0.0),
        'lambdarank_truncation_level': params['lambdarank_truncation_level'],
        'sigmoid': params['sigmoid'],
        'verbose': -1,
        'random_state': 42,
        'force_col_wise': True,
        'first_metric_only': True,
    }

    # Use fold 1 for validation (simplest split)
    n_dates = len(unique_dates)
    fold_size = n_dates // 5

    val_start_idx = 0
    val_end_idx = fold_size
    train_end_idx = n_dates  # Use dates after validation for training

    # Get indices: train on later dates, validate on early dates
    # Actually let's do proper temporal: train on early, validate on later
    train_end_date_idx = int(n_dates * 0.8) - 5  # 80% minus gap
    val_start_date_idx = int(n_dates * 0.8)

    train_mask = sample_date_idx < train_end_date_idx
    val_mask = sample_date_idx >= val_start_date_idx

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    X_tr, y_tr = X_all[train_idx], labels_all[train_idx]
    X_val, y_val = X_all[val_idx], labels_all[val_idx]
    dates_tr = dates[train_idx]
    dates_val = dates[val_idx]

    # Create groups
    train_unique_dates = sorted(np.unique(dates_tr))
    train_groups = [np.sum(dates_tr == d) for d in train_unique_dates]

    val_unique_dates = sorted(np.unique(dates_val))
    val_groups = [np.sum(dates_val == d) for d in val_unique_dates]

    train_data = lgb.Dataset(X_tr, label=y_tr, group=train_groups)
    val_data = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_data)

    # Train with logging
    evals_result = {}

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=2000,  # High limit
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(patience, verbose=False),
            lgb.record_evaluation(evals_result)
        ]
    )

    return model, evals_result

def evaluate_model(model, test_df, rebalance_days=5):
    """Evaluate on test set"""
    X_test = test_df[FEATURES].values
    X_test = np.nan_to_num(X_test, nan=0.0)
    y_test = test_df['target'].values
    y_test = np.nan_to_num(y_test, nan=0.0)

    preds = model.predict(X_test)
    dates = test_df.index.get_level_values('date')
    unique_dates = sorted(dates.unique())
    rebalance_dates = unique_dates[::rebalance_days]

    bucket_returns = {'top_1_10': [], 'top_5_15': [], 'top_10_20': []}
    all_ic, all_rank_ic, all_ndcg10 = [], [], []

    for d in rebalance_dates:
        mask = (dates == d)
        day_preds = preds[mask]
        day_targets = y_test[mask]

        if len(day_preds) < 20:
            continue

        order = np.argsort(-day_preds)
        bucket_returns['top_1_10'].append(day_targets[order[0:10]].mean())
        bucket_returns['top_5_15'].append(day_targets[order[4:15]].mean())
        bucket_returns['top_10_20'].append(day_targets[order[9:20]].mean())

        ic = np.corrcoef(day_targets, day_preds)[0, 1]
        rank_ic = stats.spearmanr(day_targets, day_preds)[0]
        all_ic.append(ic)
        all_rank_ic.append(rank_ic)

        target_ranks = stats.rankdata(day_targets, method='ordinal')
        relevance = (target_ranks - 1) / (len(target_ranks) - 1)

        # NDCG@10
        k = 10
        pred_order = np.argsort(-day_preds)[:k]
        dcg = np.sum((2**relevance[pred_order] - 1) / np.log2(np.arange(2, k + 2)))
        ideal_order = np.argsort(-relevance)[:k]
        idcg = np.sum((2**relevance[ideal_order] - 1) / np.log2(np.arange(2, k + 2)))
        all_ndcg10.append(dcg / idcg if idcg > 0 else 0)

    # Compute metrics
    top_1_10_mean = np.mean(bucket_returns['top_1_10'])
    top_10_20_mean = np.mean(bucket_returns['top_10_20'])
    spread = top_1_10_mean - top_10_20_mean

    acc = np.prod([1 + r for r in bucket_returns['top_1_10']]) - 1
    cumulative = np.cumprod([1 + r for r in bucket_returns['top_1_10']])
    running_max = np.maximum.accumulate(cumulative)
    max_dd = ((cumulative - running_max) / running_max).min()
    win_rate = np.mean([r > 0 for r in bucket_returns['top_1_10']])

    return {
        'top_1_10_mean': top_1_10_mean,
        'top_1_10_acc': acc,
        'top_1_10_dd': max_dd,
        'top_1_10_wr': win_rate,
        'spread': spread,
        'IC': np.mean(all_ic),
        'Rank_IC': np.mean(all_rank_ic),
        'NDCG_10': np.mean(all_ndcg10),
        'n_periods': len(bucket_returns['top_1_10']),
    }

def main():
    print("="*80)
    print("PATIENCE SCAN EXPERIMENT - B2_V2 Config")
    print("="*80)

    # Load data
    print("\nLoading data...")
    train_df, test_df, train_dates = load_and_split_data()
    print(f"Train: {len(train_df):,} samples, Test: {len(test_df):,} samples")

    patience_values = [50, 100, 200]
    results = []

    for patience in patience_values:
        print(f"\n{'='*80}")
        print(f"PATIENCE = {patience}")
        print("="*80)

        # Train
        print("Training...")
        model, evals_result = train_with_logging(train_df, B2_V2_PARAMS, patience)

        best_iter = model.best_iteration
        print(f"  Best iteration: {best_iter}")

        # Get NDCG curves
        train_ndcg = evals_result['train']['ndcg@10']
        val_ndcg = evals_result['val']['ndcg@10']

        print(f"  Train NDCG@10 curve (last 10 iters before best):")
        start_idx = max(0, best_iter - 10)
        for i in range(start_idx, min(best_iter + 5, len(train_ndcg))):
            marker = " <-- best" if i == best_iter else ""
            print(f"    iter {i:4d}: train={train_ndcg[i]:.5f}, val={val_ndcg[i]:.5f}{marker}")

        # Check if still improving
        if best_iter > 20:
            recent_val = val_ndcg[best_iter-20:best_iter]
            trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0]
            print(f"  Val NDCG trend (last 20 before best): {trend:.6f}/iter {'(improving)' if trend > 0.0001 else '(plateau)'}")

        # Evaluate on test
        print("  Evaluating on test set...")
        metrics = evaluate_model(model, test_df)

        print(f"  Test Results:")
        print(f"    Top 1-10 Mean: {metrics['top_1_10_mean']*100:.3f}%")
        print(f"    Top 1-10 Acc:  {metrics['top_1_10_acc']*100:.1f}%")
        print(f"    Spread:        {metrics['spread']*100:.3f}%")
        print(f"    NDCG@10:       {metrics['NDCG_10']:.4f}")
        print(f"    Win Rate:      {metrics['top_1_10_wr']*100:.1f}%")
        print(f"    Max Drawdown:  {metrics['top_1_10_dd']*100:.1f}%")

        results.append({
            'patience': patience,
            'best_iter': best_iter,
            'final_train_ndcg': train_ndcg[best_iter] if best_iter < len(train_ndcg) else train_ndcg[-1],
            'final_val_ndcg': val_ndcg[best_iter] if best_iter < len(val_ndcg) else val_ndcg[-1],
            **metrics
        })

    # Summary comparison
    print("\n" + "="*80)
    print("PATIENCE SCAN SUMMARY")
    print("="*80)
    print(f"{'Patience':>10} {'BestIter':>10} {'TrainNDCG':>12} {'ValNDCG':>12} {'TestNDCG':>12} {'Top1-10%':>10} {'Spread%':>10} {'WinRate':>10}")
    print("-"*100)

    for r in results:
        print(f"{r['patience']:>10} {r['best_iter']:>10} {r['final_train_ndcg']:>12.5f} {r['final_val_ndcg']:>12.5f} {r['NDCG_10']:>12.4f} {r['top_1_10_mean']*100:>10.2f} {r['spread']*100:>10.2f} {r['top_1_10_wr']*100:>10.1f}")

    # Diagnosis
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)

    iter_50 = results[0]['best_iter']
    iter_100 = results[1]['best_iter']
    iter_200 = results[2]['best_iter']

    if iter_100 > iter_50 * 1.3:
        print("-> best_iter increased significantly with more patience")
        print("   Suggests: patience=50 may be too short (underfitting)")
    else:
        print("-> best_iter similar across patience values")
        print("   Suggests: model converged, patience=50 is sufficient")

    ndcg_50 = results[0]['NDCG_10']
    ndcg_100 = results[1]['NDCG_10']
    ndcg_200 = results[2]['NDCG_10']

    if ndcg_100 > ndcg_50 + 0.005:
        print("-> Test NDCG improved with patience=100")
        print("   Suggests: Use patience=100 for final model")
    elif ndcg_100 < ndcg_50 - 0.005:
        print("-> Test NDCG decreased with more patience")
        print("   Suggests: Overfitting risk, stick with patience=50")
    else:
        print("-> Test NDCG similar across patience values")
        print("   Suggests: patience=50 is fine for this config")

    return 0

if __name__ == "__main__":
    sys.exit(main())
