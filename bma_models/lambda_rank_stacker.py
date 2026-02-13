#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LambdaRank Stacker - 专门优化排序的二层模型

Aligned with lambdarank_only_pipeline.py:
- Same hyperparameters, label construction, CV, training logic
- No StandardScaler (fillna(0) instead)
- No winsorization (target clipping only)
- rankdata/(n+1) label method (not rank(pct=True))
- Average best_rounds across CV folds, retrain on full data
- Raw OOF predictions (no Gauss-rank normalization)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


def calculate_topk_return_proxy(predictions: np.ndarray, y_true: np.ndarray, dates: pd.Series, k: int = 10) -> Dict[str, float]:
    """
    计算Top-K收益proxy指标 — 最终策略的最直接proxy
    """
    if len(predictions) != len(y_true) or len(predictions) != len(dates):
        raise ValueError("predictions, y_true, dates长度必须一致")

    daily_returns = []
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)
    unique_dates = dates.unique()

    for date in unique_dates:
        date_mask = (dates == date).values if hasattr(dates, 'values') else (dates == date)
        date_preds = predictions[date_mask]
        date_returns = y_true[date_mask]

        if len(date_preds) < k:
            topk_mask = np.ones(len(date_preds), dtype=bool)
        else:
            topk_indices = np.argsort(date_preds)[-k:]
            topk_mask = np.zeros(len(date_preds), dtype=bool)
            topk_mask[topk_indices] = True

        topk_returns = date_returns[topk_mask]
        if len(topk_returns) > 0:
            daily_returns.append(np.mean(topk_returns))

    if len(daily_returns) == 0:
        return {'mean_return': 0.0, 'ir': 0.0, 't_stat': 0.0, 'n_days': 0}

    daily_returns = np.array(daily_returns)
    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns)
    ir = mean_return / (std_return + 1e-10)
    t_stat = mean_return / (std_return / np.sqrt(len(daily_returns)) + 1e-10)

    return {
        'mean_return': float(mean_return),
        'ir': float(ir),
        't_stat': float(t_stat),
        'n_days': len(daily_returns),
        'std_return': float(std_return)
    }


# ─── Aligned with lambdarank_only_pipeline.py ───

def build_quantile_labels(y: np.ndarray, dates: np.ndarray, n_quantiles: int) -> np.ndarray:
    """Exact same label construction as lambdarank_only_pipeline.py"""
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
    """Exact same as lambdarank_only_pipeline.py"""
    return [int(np.sum(dates == d)) for d in np.unique(dates)]


def purged_cv_splits(dates: np.ndarray, n_splits: int, gap: int, embargo: int):
    """Exact same CV implementation as lambdarank_only_pipeline.py"""
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


class LambdaRankStacker:
    """
    LambdaRank排序模型 — aligned with lambdarank_only_pipeline.py

    All hyperparameters, label construction, CV, and training logic
    are identical to the standalone pipeline.
    """

    def __init__(self,
                 base_cols: Tuple[str, ...] = None,
                 n_quantiles: int = 64,
                 label_gain_power: float = 2.1,
                 lgb_params: Optional[Dict[str, Any]] = None,
                 num_boost_round: int = 800,
                 early_stopping_rounds: int = 50,
                 use_internal_cv: bool = True,
                 cv_n_splits: int = 5,
                 cv_gap_days: int = 5,
                 cv_embargo_days: int = 5,
                 random_state: int = 0):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for LambdaRankStacker")

        self.base_cols = base_cols
        self._alpha_factor_cols = None
        self.n_quantiles = n_quantiles
        self.label_gain_power = label_gain_power
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.use_internal_cv = bool(use_internal_cv)
        self.cv_n_splits = cv_n_splits
        self.cv_gap_days = cv_gap_days
        self.cv_embargo_days = cv_embargo_days
        self.random_state = random_state

        # Label gain: same formula as pipeline
        self.label_gain = [(i / (n_quantiles - 1)) ** label_gain_power * (n_quantiles - 1)
                           for i in range(n_quantiles)]

        # Default LGB params: exact match with lambdarank_only_pipeline.py BEST_PARAMS
        default_lgb_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [10, 20],
            'label_gain': self.label_gain,
            'num_leaves': 11,
            'max_depth': 3,
            'learning_rate': 0.04,
            'feature_fraction': 1.0,
            'bagging_fraction': 0.70,
            'bagging_freq': 1,
            'min_data_in_leaf': 350,
            'lambda_l1': 0.0,
            'lambda_l2': 120,
            'min_gain_to_split': 0.30,
            'lambdarank_truncation_level': 25,
            'sigmoid': 1.1,
            'verbose': -1,
            'force_row_wise': True,
            'seed': random_state,
            'bagging_seed': random_state,
            'feature_fraction_seed': random_state,
            'data_random_seed': random_state,
            'deterministic': True,
        }

        self.lgb_params = default_lgb_params.copy()
        if lgb_params:
            self.lgb_params.update(lgb_params)
            # Ensure label_gain stays consistent
            if 'label_gain' not in lgb_params:
                self.lgb_params['label_gain'] = self.label_gain

        # Model state
        self.model = None
        self.fitted_ = False
        self._oof_predictions = None
        self._first_val_date = None

        logger.info("LambdaRank Stacker initialized (aligned with lambdarank_only_pipeline)")
        logger.info(f"  n_quantiles={self.n_quantiles}, label_gain_power={self.label_gain_power}")
        logger.info(f"  lr={self.lgb_params['learning_rate']}, leaves={self.lgb_params['num_leaves']}, "
                     f"depth={self.lgb_params['max_depth']}, min_leaf={self.lgb_params['min_data_in_leaf']}")
        logger.info(f"  L2={self.lgb_params['lambda_l2']}, truncation={self.lgb_params['lambdarank_truncation_level']}")
        logger.info(f"  boost_rounds={self.num_boost_round}, early_stop={self.early_stopping_rounds}")

    def fit(self, df: pd.DataFrame, target_col: str = 'ret_fwd_5d', alpha_factors: pd.DataFrame = None) -> 'LambdaRankStacker':
        """Train LambdaRank model — aligned with pipeline logic."""
        logger.info("Training LambdaRank (pipeline-aligned)...")

        # Validate MultiIndex
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("DataFrame must have MultiIndex(date, ticker)")
        if df.index.nlevels == 2 and df.index.names != ['date', 'ticker']:
            df.index.names = ['date', 'ticker']

        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")

        # Feature selection (same priority: base_cols > alpha_factors > auto-detect)
        if self.base_cols is not None and len(self.base_cols) > 0:
            self._alpha_factor_cols = [col for col in self.base_cols if col in df.columns]
            missing_cols = [col for col in self.base_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing feature columns: {missing_cols}")
        elif alpha_factors is not None:
            self._alpha_factor_cols = [col for col in alpha_factors.columns if col != target_col]
            df = pd.concat([df[[target_col]], alpha_factors[self._alpha_factor_cols]], axis=1)
        else:
            exclude_patterns = [target_col, 'pred_', 'lambda_', 'ridge_', 'final_', 'rank', 'weight']
            self._alpha_factor_cols = [col for col in df.columns
                                       if not any(pattern in col.lower() for pattern in exclude_patterns)]

        self.base_cols = tuple(self._alpha_factor_cols)
        logger.info(f"  Features: {len(self._alpha_factor_cols)} columns")

        # ─── Aligned with pipeline: fillna(0), no scaler ───
        X = df[list(self._alpha_factor_cols)].fillna(0.0).to_numpy()
        y = df[target_col].to_numpy()
        dates = df.index.get_level_values('date').to_numpy()

        # ─── Aligned with pipeline: build_quantile_labels (rankdata/(n+1)) ───
        labels = build_quantile_labels(y, dates, self.n_quantiles)

        logger.info(f"  Samples: {len(X)}, dates: {len(np.unique(dates))}, "
                     f"avg group: {len(X)/len(np.unique(dates)):.0f}")

        if self.use_internal_cv:
            # ─── Aligned with pipeline: purged_cv_splits → avg best_rounds → retrain full ───
            logger.info(f"  Internal CV: {self.cv_n_splits} folds, gap={self.cv_gap_days}, embargo={self.cv_embargo_days}")
            self.model, self._oof_predictions = self._train_with_cv(
                X, labels, y, dates, df
            )
        else:
            # No internal CV: train on full data (same as pipeline final retrain)
            logger.info("  Internal CV disabled: training on full data")
            full_set = lgb.Dataset(X, label=labels, group=group_counts(dates))
            self.model = lgb.train(
                self.lgb_params, full_set,
                num_boost_round=self.num_boost_round,
                callbacks=[lgb.log_evaluation(0)]
            )

        # Post-training metrics
        train_pred = self.model.predict(X)
        ndcg10 = self._calculate_ndcg(labels, train_pred, group_counts(dates), 10)
        ndcg20 = self._calculate_ndcg(labels, train_pred, group_counts(dates), 20)
        logger.info(f"  Train NDCG@10={ndcg10:.4f}, @20={ndcg20:.4f}")

        self.fitted_ = True
        return self

    def _train_with_cv(self, X: np.ndarray, labels: np.ndarray, y_raw: np.ndarray,
                       dates: np.ndarray, df: pd.DataFrame):
        """
        Aligned with pipeline's train_lambdarank():
        1. Run all CV folds to find best_rounds
        2. Collect OOF predictions along the way
        3. Average best_rounds
        4. Retrain on full data with averaged rounds
        """
        oof_predictions = np.full(len(X), np.nan)
        best_rounds = []

        # Record first val date for OOF cold-start filtering
        first_val_date = None

        for fold_idx, (train_idx, val_idx) in enumerate(
            purged_cv_splits(dates, self.cv_n_splits, self.cv_gap_days, self.cv_embargo_days)
        ):
            train_dates = dates[train_idx]
            val_dates = dates[val_idx]

            # Record first val date
            if first_val_date is None:
                first_val_date = pd.Timestamp(np.min(val_dates))
                self._first_val_date = first_val_date

            train_set = lgb.Dataset(
                X[train_idx], label=labels[train_idx],
                group=group_counts(train_dates)
            )
            val_set = lgb.Dataset(
                X[val_idx], label=labels[val_idx],
                group=group_counts(val_dates)
            )

            booster = lgb.train(
                self.lgb_params, train_set,
                num_boost_round=self.num_boost_round,
                valid_sets=[val_set],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ]
            )

            bi = booster.best_iteration or self.num_boost_round
            best_rounds.append(bi)

            # OOF predictions (raw, no normalization — aligned with pipeline)
            val_pred = booster.predict(X[val_idx])
            oof_predictions[val_idx] = val_pred

            # Log fold metrics
            val_dates_series = pd.Series(val_dates)
            topk = calculate_topk_return_proxy(val_pred, y_raw[val_idx], val_dates_series, k=10)
            ndcg10 = self._calculate_ndcg(labels[val_idx], val_pred, group_counts(val_dates), 10)
            logger.info(f"  Fold {fold_idx+1}: best_iter={bi}, NDCG@10={ndcg10:.4f}, "
                        f"Top10 mean={topk['mean_return']:.4f}, IR={topk['ir']:.2f}")

        # ─── Aligned with pipeline: average best_rounds, retrain on full data ───
        final_rounds = int(np.mean(best_rounds)) if best_rounds else self.num_boost_round
        logger.info(f"  CV done: avg best_rounds={final_rounds} from {len(best_rounds)} folds")

        full_set = lgb.Dataset(X, label=labels, group=group_counts(dates))
        final_model = lgb.train(
            self.lgb_params, full_set,
            num_boost_round=final_rounds,
            callbacks=[lgb.log_evaluation(0)]
        )

        return final_model, oof_predictions

    def get_oof_predictions(self, df: pd.DataFrame) -> pd.Series:
        """Get OOF predictions (raw, no Gauss-rank normalization)."""
        if self._oof_predictions is None:
            raise RuntimeError("OOF predictions not available (model not trained with CV)")

        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("df must have MultiIndex(date, ticker)")

        if len(self._oof_predictions) != len(df):
            raise ValueError(
                f"OOF length ({len(self._oof_predictions)}) != df length ({len(df)})"
            )

        oof_series = pd.Series(self._oof_predictions, index=df.index, name='lambda_oof')

        # Cold-start filtering: remove samples before first val date
        if self._first_val_date is not None:
            df_dates = pd.to_datetime(df.index.get_level_values('date')).normalize()
            valid_mask = df_dates >= self._first_val_date
            before_count = (~valid_mask).sum()
            if before_count > 0:
                logger.info(f"  OOF: filtered {before_count} cold-start samples (before {self._first_val_date.date()})")
            oof_series = oof_series[valid_mask]

        logger.info(f"  OOF predictions: {len(oof_series)} valid samples")
        return oof_series

    def predict(self, df: pd.DataFrame, alpha_factors: pd.DataFrame = None) -> pd.DataFrame:
        """Predict rankings — aligned with pipeline (fillna(0), no scaler)."""
        if not self.fitted_:
            raise RuntimeError("Model not fitted")

        df_clean = alpha_factors.copy() if alpha_factors is not None else df.copy()

        if self._alpha_factor_cols is None:
            raise RuntimeError("Feature columns not set")

        missing_cols = [col for col in self._alpha_factor_cols if col not in df_clean.columns]
        if missing_cols:
            available_cols = [col for col in self._alpha_factor_cols if col in df_clean.columns]
            if len(available_cols) < len(self._alpha_factor_cols) * 0.5:
                raise ValueError(f"Too few feature columns available: {len(available_cols)}/{len(self._alpha_factor_cols)}")
            X = df_clean[available_cols].fillna(0.0).values
        else:
            X = df_clean[list(self._alpha_factor_cols)].fillna(0.0).values

        # No scaler — aligned with pipeline
        raw_predictions = self.model.predict(X)

        result = df_clean.copy()
        result['lambda_score'] = raw_predictions

        # Per-day ranking
        ranked_series = result.groupby(level='date')['lambda_score'].rank(method='average', ascending=False)
        result['lambda_rank'] = ranked_series

        pct_series = result.groupby(level='date')['lambda_score'].rank(pct=True)
        result['lambda_pct'] = pct_series

        return result[['lambda_score', 'lambda_rank', 'lambda_pct']]

    def _calculate_ndcg(self, y_true: np.ndarray, y_pred: np.ndarray, group_sizes: list, k: int) -> float:
        """Calculate NDCG@K."""
        try:
            from sklearn.metrics import ndcg_score
            start_idx = 0
            ndcg_scores = []
            for group_size in group_sizes:
                if group_size < k:
                    start_idx += group_size
                    continue
                end_idx = start_idx + group_size
                group_true = y_true[start_idx:end_idx]
                group_pred = y_pred[start_idx:end_idx]
                ndcg = ndcg_score(group_true.reshape(1, -1), group_pred.reshape(1, -1), k=k)
                ndcg_scores.append(ndcg)
                start_idx = end_idx
            return np.mean(ndcg_scores) if ndcg_scores else 0.0
        except ImportError:
            return 0.0

    def get_model_info(self) -> Dict[str, Any]:
        if not self.fitted_:
            return {'fitted': False}
        return {
            'fitted': True,
            'model_type': 'LambdaRank',
            'best_iteration': self.model.best_iteration,
            'feature_importance': dict(zip(self.base_cols, self.model.feature_importance()[:len(self.base_cols)])),
            'n_quantiles': self.n_quantiles,
            'lgb_params': {k: v for k, v in self.lgb_params.items() if k != 'label_gain'},
        }
