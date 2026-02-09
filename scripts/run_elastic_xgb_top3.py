#!/usr/bin/env python3
"""Train ElasticNet and XGBoost on 9-factor base using chronological split metrics."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise SystemExit("xgboost is required: pip install xgboost") from exc

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from lambdarank_only_pipeline import (
    ensure_multiindex,
    chronological_subset,
    time_split,
)

FEATURES = [
    'reversal_3d',
    'avg_trade_size',
    'trend_r2_20',
]

DATA_FILE = Path('data/factor_exports/polygon_factors_all_2021_2026_T5_final.parquet')
OUTPUT_ROOT = Path('results/elastic_xgb_nine')


def compute_metrics(preds: np.ndarray, y: np.ndarray, dates: np.ndarray, horizon_days: int) -> dict:
    """Port of lambdarank_only_pipeline.evaluate for arbitrary predictions."""
    import numpy as np
    from scipy import stats

    unique_dates = np.unique(dates)
    non_overlap = {'top_1_10': [], 'top_5_15': [], 'top_10_20': []}
    overlap = {'top_1_10': [], 'top_5_15': [], 'top_10_20': []}
    ndcg10, ndcg20, ic_vals = [], [], []

    def bucket_values(scores, targets):
        order = np.argsort(-scores)
        return (
            targets[order[:10]].mean(),
            targets[order[4:15]].mean(),
            targets[order[9:20]].mean(),
            order,
        )

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

    step = max(1, horizon_days)
    for d in unique_dates[::step]:
        mask = dates == d
        day_scores = preds[mask]
        day_targets = y[mask]
        if len(day_targets) < 20:
            continue
        top1, top5, top10, _ = bucket_values(day_scores, day_targets)
        non_overlap['top_1_10'].append(top1)
        non_overlap['top_5_15'].append(top5)
        non_overlap['top_10_20'].append(top10)

    def bucket_stats(values):
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

    results['spread'] = (
        results['top_1_10_mean'] - results['top_10_20_mean']
        if np.isfinite(results['top_1_10_mean']) and np.isfinite(results['top_10_20_mean'])
        else float('nan')
    )
    results['IC_mean'] = float(np.nanmean(ic_vals)) if ic_vals else float('nan')
    results['NDCG_10'] = float(np.nanmean(ndcg10)) if ndcg10 else float('nan')
    results['NDCG_20'] = float(np.nanmean(ndcg20)) if ndcg20 else float('nan')
    return results


def train_and_eval(model_name: str, model, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols, horizon_days: int) -> dict:
    X_train = train_df[feature_cols].fillna(0.0).to_numpy()
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = train_df['target'].to_numpy()
    X_test = test_df[feature_cols].fillna(0.0).to_numpy()
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = test_df['target'].to_numpy()
    dates_test = test_df.index.get_level_values('date').to_numpy()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = compute_metrics(preds, y_test, dates_test, horizon_days)
    return metrics


def main() -> None:
    df = pd.read_parquet(DATA_FILE)
    df = ensure_multiindex(df)
    df = chronological_subset(df, 1.0)
    train_df, test_df, train_dates = time_split(df, split=0.8, purge_gap=5)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = OUTPUT_ROOT / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    elast_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=0.0005, l1_ratio=0.4, max_iter=10000, random_state=0)),
    ])
    xgb_model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method='hist',
        random_state=0,
    )

    models = {
        'elastic_net': elast_pipeline,
        'xgboost': xgb_model,
    }

    for name, model in models.items():
        metrics = train_and_eval(name, model, train_df, test_df, FEATURES, horizon_days=5)
        metrics['train_dates'] = [str(train_df.index.get_level_values('date')[0]), str(train_df.index.get_level_values('date')[-1])]
        test_dates = test_df.index.get_level_values('date')
        metrics['test_dates'] = [str(test_dates[0]), str(test_dates[-1])]
        model_dir = output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        (model_dir / 'features.json').write_text(json.dumps(FEATURES, indent=2), encoding='utf-8')
        metrics['model'] = name
        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

