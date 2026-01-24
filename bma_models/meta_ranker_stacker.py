#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta Ranker Stacker - LightGBM Ranker for second-layer meta-learning

Uses LightGBM LambdaRank objective to optimize ranking of first-layer predictions.
Designed to replace RidgeStacker with ranking-focused optimization.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Import PurgedCV to prevent data leakage
try:
    from bma_models.unified_purged_cv_factory import create_unified_cv
    PURGED_CV_AVAILABLE = True
except ImportError:
    PURGED_CV_AVAILABLE = False

logger = logging.getLogger(__name__)

class MetaRankerStacker:
    """
    Meta Ranker Stacker - LightGBM Ranker for second-layer stacking
    
    Core advantages:
    - Uses LambdaRank objective to optimize Top-K ranking
    - Takes first-layer predictions as features
    - Converts continuous targets to ranks for ranking optimization
    - Focuses on Top-10/30 performance via NDCG metric
    """

    def __init__(self,
                 base_cols: Tuple[str, ...] = ('pred_catboost', 'pred_xgb', 'pred_lambdarank', 'pred_elastic'),
                 n_quantiles: int = 64,
                 label_gain_power: float = 2.3,  # 🔧 1.7 -> 2.3（★重点：重赏头部准确度）
                 lgb_params: Optional[Dict[str, Any]] = None,
                 num_boost_round: int = 2000,  # 🔧 140 -> 2000（配套大幅增加轮数）
                 early_stopping_rounds: int = 100,  # 🔧 40 -> 100
                 use_purged_cv: bool = True,
                 use_internal_cv: bool = True,
                 cv_n_splits: int = 6,
                 cv_gap_days: int = 5,
                 cv_embargo_days: int = 5,
                 random_state: int = 42):
        """
        Initialize Meta Ranker Stacker
        
        Args:
            base_cols: First-layer prediction column names
            n_quantiles: Number of quantile levels for rank conversion
            label_gain_power: Power for label gain (higher = more focus on top ranks)
            lgb_params: LightGBM parameters (will override defaults)
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping rounds
            use_purged_cv: Use PurgedCV (required for T+5)
            use_internal_cv: Use internal CV (recommended)
            cv_n_splits: Number of CV folds
            cv_gap_days: Gap days between train and test
            cv_embargo_days: Embargo days to prevent leakage
            random_state: Random seed
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for MetaRankerStacker")
        if use_internal_cv and not use_purged_cv:
            raise ValueError("When internal CV is enabled, purged CV must be enabled for T+5 training.")
        if not PURGED_CV_AVAILABLE:
            raise RuntimeError("Unified Purged CV factory is unavailable.")
        if use_internal_cv and (cv_n_splits, cv_gap_days, cv_embargo_days) != (6, 5, 5):
            raise ValueError("MetaRankerStacker enforces T+5 CV settings when internal CV is enabled: splits=6, gap=5, embargo=5.")

        self.base_cols = base_cols
        self.n_quantiles = n_quantiles
        self.label_gain_power = label_gain_power
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.use_internal_cv = bool(use_internal_cv)
        self.use_purged_cv = bool(use_purged_cv)
        self.cv_n_splits = cv_n_splits
        self.cv_gap_days = cv_gap_days
        self.cv_embargo_days = cv_embargo_days
        self.random_state = random_state

        # Generate label_gain sequence (power-enhanced)
        if n_quantiles <= 1:
            raise ValueError(f"n_quantiles must be > 1, got {n_quantiles}")
        if label_gain_power == 1.0:
            self.label_gain = list(range(n_quantiles))
        else:
            self.label_gain = [(i / (n_quantiles - 1)) ** label_gain_power * (n_quantiles - 1)
                              for i in range(n_quantiles)]

        # Default LightGBM parameters (will be overridden by lgb_params)
        default_lgb_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 15],  # 🔧 聚焦更核心的头部
            'label_gain': self.label_gain,
            'num_leaves': 15,  # 🔧 31 -> 15（降低复杂度）
            'max_depth': 3,  # 🔧 4 -> 3
            'learning_rate': 0.005,  # 🔧 大幅降低，更精细的梯度下降
            'min_data_in_leaf': 500,  # 🔧 200 -> 500（提高叶子节点门槛）
            'lambda_l1': 2.0,  # 🔧 0.0 -> 2.0（新增L1正则化）
            'lambda_l2': 20.0,  # 🔧 15.0 -> 20.0（增强L2正则化）
            'feature_fraction': 0.7,  # 🔧 1.0 -> 0.7
            'bagging_fraction': 0.6,  # 🔧 0.8 -> 0.6
            'bagging_freq': 1,
            'lambdarank_truncation_level': 60,  # 🔧 1200 -> 60（★重点：只优化 Top 60）
            'sigmoid': 1.2,
            'verbose': -1,
            'random_state': random_state,
            'force_col_wise': True
        }
        
        # Set defaults first
        self.lgb_params = default_lgb_params.copy()
        
        # Then override with provided lgb_params
        if lgb_params:
            # If label_gain is provided in lgb_params, it will override the power-enhanced one
            # Log warning to make user aware
            if 'label_gain' in lgb_params:
                logger.warning(f"⚠️ label_gain provided in lgb_params will override power-enhanced label_gain (label_gain_power={self.label_gain_power})")
            self.lgb_params.update(lgb_params)

        # Model state
        self.lightgbm_model = None
        self.scaler = None
        self.fitted_ = False
        self._oof_predictions = None
        self._oof_index = None
        self.actual_feature_cols_ = None

        logger.info("✅ Meta Ranker Stacker initialized")
        logger.info(f"   Base features: {self.base_cols}")
        logger.info(f"   Label gain power: {self.label_gain_power}")
        logger.info(f"   NDCG eval at: {self.lgb_params['ndcg_eval_at']}")
        logger.info(f"   训练轮数: {self.num_boost_round}, 早停: {self.early_stopping_rounds}")
        logger.info(f"   模型容量: num_leaves={self.lgb_params.get('num_leaves')}, max_depth={self.lgb_params.get('max_depth')}, min_data_in_leaf={self.lgb_params.get('min_data_in_leaf')}")
        logger.info(f"   采样: feature_fraction={self.lgb_params.get('feature_fraction')}, bagging_fraction={self.lgb_params.get('bagging_fraction')}")
        logger.info(f"   LambdaRank: truncation_level={self.lgb_params.get('lambdarank_truncation_level')}, sigmoid={self.lgb_params.get('sigmoid')}")

    def _convert_to_rank_labels(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Convert continuous target to rank labels for ranking optimization
        
        Args:
            df: DataFrame with target column
            target_col: Target column name
            
        Returns:
            Processed DataFrame with rank labels, conversion report
        """
        logger.info(f"🔄 Converting continuous target to {self.n_quantiles} quantile ranks")

        df_processed = df.copy()

        def _group_rank_transform(group):
            """Group-wise rank transformation"""
            target_values = group[target_col].dropna()
            if len(target_values) <= 1:
                group[f'{target_col}_rank'] = self.n_quantiles // 2
                return group

            # Rank percentile [0, 1]
            rank_pct = target_values.rank(pct=True, method='average')
            
            # Map to quantile levels [0, N-1]
            labels = np.floor(rank_pct * self.n_quantiles).astype(int)
            labels[labels == self.n_quantiles] = self.n_quantiles - 1

            # Map back to full DataFrame
            full_ranks = pd.Series(self.n_quantiles // 2, index=group.index)
            full_ranks.loc[target_values.index] = labels
            group[f'{target_col}_rank'] = full_ranks.astype(int)

            return group

        # Group by date for rank conversion
        df_processed = df_processed.groupby(level='date', group_keys=False).apply(_group_rank_transform)

        rank_col = f'{target_col}_rank'
        unique_ranks = df_processed[rank_col].nunique()
        
        logger.info(f"✅ Rank conversion complete: {unique_ranks} quantile levels used")

        conversion_report = {
            'n_quantiles_configured': self.n_quantiles,
            'n_quantiles_used': unique_ranks,
            'label_gain_power': self.label_gain_power
        }

        return df_processed, conversion_report

    def _create_purged_cv_split(self, dates: pd.Series):
        """Create PurgedCV splits"""
        if not self.use_purged_cv:
            raise RuntimeError('PurgedCV is mandatory for MetaRankerStacker T+5 training.')

        cv = create_unified_cv(
            n_splits=self.cv_n_splits,
            gap=self.cv_gap_days,
            embargo=self.cv_embargo_days
        )

        unique_dates = sorted(dates.unique())
        if not unique_dates:
            raise ValueError('PurgedCV requires non-empty date information.')
        date_to_idx = {date: i for i, date in enumerate(unique_dates)}
        groups = dates.map(date_to_idx).values

        logger.info(f"✅ Using PurgedCV: splits={self.cv_n_splits}, gap={self.cv_gap_days}, embargo={self.cv_embargo_days}")

        return cv.split(X=np.zeros((len(dates), 1)), y=None, groups=groups)

    def fit(self, df: pd.DataFrame, max_train_to_today: bool = True) -> 'MetaRankerStacker':
        """
        Train Meta Ranker Stacker
        
        Args:
            df: Training DataFrame with MultiIndex(date, ticker) and first-layer predictions
            max_train_to_today: Whether to use all available data (for compatibility)
            
        Returns:
            self
        """
        logger.info("🚀 Training Meta Ranker Stacker...")

        # 🔧 统一输入处理：确保使用检测到的MultiIndex
        # Validate input
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("DataFrame must have MultiIndex(date, ticker)")
        
        # 🔧 验证MultiIndex格式正确（date, ticker）
        if df.index.names != ['date', 'ticker']:
            logger.warning(f"MultiIndex名称不匹配: {df.index.names}，期望: ['date', 'ticker']")
            # 尝试修复：如果只有两层，重命名
            if df.index.nlevels == 2:
                df.index.names = ['date', 'ticker']
                logger.info("✅ 已修复MultiIndex名称")
            else:
                raise ValueError(f"MultiIndex格式不正确: names={df.index.names}, levels={df.index.nlevels}")

        # Find target column
        target_cols = [col for col in df.columns if col.startswith('ret_fwd')]
        if not target_cols:
            raise ValueError("No target column found (ret_fwd_*)")
        target_col = target_cols[0]

        # Validate base columns
        missing_cols = [col for col in self.base_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Save actual feature columns
        if self.actual_feature_cols_ is None:
            self.actual_feature_cols_ = list(self.base_cols)
            logger.info(f"🔧 Saved actual feature columns: {self.actual_feature_cols_}")

        # 🔧 OOF冷启动修复：检测并过滤掉未覆盖的日期段
        # 检查第一层预测列，找出最早的有效日期（第一个验证集日期）
        df_dates = pd.to_datetime(df.index.get_level_values('date')).normalize()
        first_val_date = None
        
        # 从第一层预测列中检测最早的有效日期
        # 假设OOF预测为0或NaN的样本是未覆盖的
        for col in self.base_cols:
            if col in df.columns:
                col_data = df[col]
                # 找出第一个非零且非NaN的样本的日期
                valid_mask = (col_data != 0) & (~pd.isna(col_data))
                if valid_mask.any():
                    # 获取第一个有效样本的日期
                    first_valid_mask_idx = valid_mask.idxmax() if hasattr(valid_mask, 'idxmax') else None
                    if first_valid_mask_idx is None:
                        # 使用numpy argmax作为回退
                        valid_positions = np.where(valid_mask.values if hasattr(valid_mask, 'values') else valid_mask)[0]
                        if len(valid_positions) > 0:
                            first_valid_pos = valid_positions[0]
                            first_valid_date = df_dates.iloc[first_valid_pos] if hasattr(df_dates, 'iloc') else df_dates[first_valid_pos]
                    else:
                        # 从MultiIndex中提取日期
                        if isinstance(df.index, pd.MultiIndex):
                            first_valid_date = pd.to_datetime(first_valid_mask_idx[0]).normalize()
                        else:
                            first_valid_date = pd.to_datetime(df_dates[df.index == first_valid_mask_idx].min()).normalize()
                    
                    if first_valid_date is not None:
                        if first_val_date is None or first_valid_date < first_val_date:
                            first_val_date = first_valid_date
        
        # 如果检测到first_val_date，过滤掉之前的样本
        if first_val_date is not None:
            valid_date_mask = df_dates >= first_val_date
            before_count = (~valid_date_mask).sum()
            if before_count > 0:
                logger.warning(
                    f"   ⚠️  OOF冷启动空洞检测: 过滤掉{before_count}个样本 "
                    f"(日期 < {first_val_date.date()})"
                )
                logger.info(
                    f"   🔧 这些样本的第一层OOF预测为0/缺失，会导致MetaRankerStacker学到时间伪信号"
                )
                df = df[valid_date_mask]
                logger.info(f"   ✅ 过滤后样本数: {len(df)} (原始: {len(df.index) + before_count})")
        
        # Convert target to ranks
        df_processed, conversion_report = self._convert_to_rank_labels(df, target_col)
        rank_col = f'{target_col}_rank'

        # Prepare features and labels
        X = df_processed[list(self.base_cols)].values
        y = df_processed[rank_col].values

        # Prepare group information (each trading day is a group)
        date_index = df_processed.index.get_level_values('date')
        unique_dates = date_index.unique()
        group_sizes = [len(df_processed.loc[date]) for date in unique_dates]

        logger.info(f"   Training samples: {len(X)}")
        logger.info(f"   Feature dimension: {X.shape[1]}")
        logger.info(f"   Trading day groups: {len(group_sizes)}")
        logger.info(f"   Average group size: {np.mean(group_sizes):.1f}")

        # Handle missing values
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        # Recalculate group sizes for valid samples
        df_valid = df_processed.iloc[valid_mask]
        valid_date_index = df_valid.index.get_level_values('date')
        valid_unique_dates = valid_date_index.unique()
        valid_group_sizes = [len(df_valid.loc[date]) for date in valid_unique_dates]

        logger.info(f"   Valid samples: {len(X_valid)} ({len(X_valid)/len(X)*100:.1f}%)")

        if len(X_valid) < 30:
            raise ValueError(f"Insufficient valid samples: {len(X_valid)} < 30")

        # Feature standardization
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_valid)

        # Prepare group information for LightGBM
        groups = []
        for date in valid_unique_dates:
            group_size = len(df_valid.loc[date])
            groups.append(group_size)

        # Train with or without internal CV
        if self.use_internal_cv:
            logger.info("📊 Training with internal PurgedCV...")
            dates_series = pd.Series(valid_date_index)
            cv_splits = list(self._create_purged_cv_split(dates_series))
            
            # 🔧 最小训练窗限制：至少2年交易日（约500天）才能计入best_iteration
            try:
                from bma_models.unified_config_loader import get_time_config
                time_config = get_time_config()
                min_train_window_days = getattr(time_config, 'min_train_window_days', 252)
            except:
                min_train_window_days = 252  # 默认1年交易日
            
            logger.info(f"   🔧 最小训练窗限制: {min_train_window_days}天（约{min_train_window_days/252:.1f}年）")
            
            best_iteration = None
            best_score = -np.inf
            successful_folds = 0
            valid_fold_start_idx = None  # 记录第一个有效fold的索引
            
            for fold, (train_idx, val_idx) in enumerate(cv_splits):
                # 计算训练窗天数
                train_dates_fold = valid_date_index[train_idx]
                train_unique_dates_fold = pd.Series(train_dates_fold).unique()
                train_window_days = len(train_unique_dates_fold)
                
                # 🔧 检查训练窗是否满足最小要求
                if train_window_days < min_train_window_days:
                    logger.warning(
                        f"   ⚠️  MetaRanker CV Fold {fold + 1} 训练窗({train_window_days}天) < 最小要求({min_train_window_days}天)，跳过"
                    )
                    logger.info(f"   🔧 此fold的best_iteration将不计入统计（避免噪声污染）")
                    continue
                
                # 记录第一个有效fold
                if valid_fold_start_idx is None:
                    valid_fold_start_idx = fold
                    logger.info(f"   ✅ 从Fold {fold + 1}开始计入best_iteration统计 (训练窗={train_window_days}天)")
                X_train_fold = X_scaled[train_idx]
                y_train_fold = y_valid[train_idx]
                X_val_fold = X_scaled[val_idx]
                y_val_fold = y_valid[val_idx]
                
                # Group sizes for this fold - count samples per date within the fold
                train_dates = valid_date_index[train_idx]
                val_dates = valid_date_index[val_idx]
                # Count samples per unique date in the fold (not from full dataframe)
                train_groups = [np.sum(train_dates == date) for date in train_dates.unique()]
                val_groups = [np.sum(val_dates == date) for date in val_dates.unique()]
                
                train_dataset = lgb.Dataset(
                    X_train_fold,
                    label=y_train_fold,
                    group=train_groups
                )
                val_dataset = lgb.Dataset(
                    X_val_fold,
                    label=y_val_fold,
                    group=val_groups,
                    reference=train_dataset
                )
                
                logger.info(f"🔧 [MetaRanker CV Fold {fold+1}] 使用参数: lambda_l2={self.lgb_params.get('lambda_l2')}, lambdarank_truncation_level={self.lgb_params.get('lambdarank_truncation_level')}, label_gain_power={self.label_gain_power}")
                try:
                    model = lgb.train(
                        self.lgb_params,
                        train_dataset,
                        num_boost_round=self.num_boost_round,
                        valid_sets=[val_dataset],
                        valid_names=['val'],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False),
                            lgb.log_evaluation(period=0)
                        ]
                    )
                    
                    # Get best iteration and score - track maximum best_iteration across folds
                    if hasattr(model, 'best_iteration') and model.best_iteration is not None:
                        successful_folds += 1
                        # Update best_iteration if this fold's is better (or first valid one)
                        if best_iteration is None or model.best_iteration > best_iteration:
                            best_iteration = model.best_iteration
                        # Safely extract best score
                        try:
                            if hasattr(model, 'best_score') and model.best_score:
                                val_scores = model.best_score.get('val', {})
                                if 'ndcg@10' in val_scores:
                                    current_score = val_scores['ndcg@10']
                                    if current_score > best_score:
                                        best_score = current_score
                        except (KeyError, AttributeError, TypeError) as e:
                            logger.debug(f"Could not extract best_score from model: {e}")
                except Exception as e:
                    logger.warning(f"⚠️ CV Fold {fold+1} training failed: {e}")
                    continue
            
            # Validate that at least one fold succeeded
            if successful_folds == 0:
                raise RuntimeError("All CV folds failed to train. Cannot determine best_iteration.")
            
            # Log CV results
            if best_score > -np.inf:
                logger.info(f"✅ CV training complete. Successful folds: {successful_folds}/{len(cv_splits)}, Best iteration: {best_iteration}, Best NDCG@10: {best_score:.4f}")
                if valid_fold_start_idx is not None and valid_fold_start_idx > 0:
                    skipped_folds = valid_fold_start_idx
                    logger.info(f"   🔧 跳过了前{skipped_folds}个fold（训练窗不足{min_train_window_days}天）")
            else:
                logger.warning(f"⚠️ CV training complete but no valid scores found. Successful folds: {successful_folds}/{len(cv_splits)}, Best iteration: {best_iteration}")
            
            # Train final model on all data
            # Use best_iteration if available, otherwise fall back to num_boost_round
            if best_iteration is not None and best_iteration > 0:
                final_rounds = best_iteration
            else:
                final_rounds = self.num_boost_round
                logger.warning(f"⚠️ Using default num_boost_round={final_rounds} (best_iteration={best_iteration})")
            
            logger.info(f"🔧 [MetaRanker最终训练] 使用参数: lambda_l2={self.lgb_params.get('lambda_l2')}, lambdarank_truncation_level={self.lgb_params.get('lambdarank_truncation_level')}, label_gain_power={self.label_gain_power}, min_data_in_leaf={self.lgb_params.get('min_data_in_leaf')}")
            train_dataset = lgb.Dataset(X_scaled, label=y_valid, group=groups)
            self.lightgbm_model = lgb.train(
                self.lgb_params,
                train_dataset,
                num_boost_round=final_rounds,
                callbacks=[lgb.log_evaluation(period=0)]
            )
        else:
            # Direct training without internal CV
            logger.info(f"🔧 [MetaRanker直接训练] 使用参数: lambda_l2={self.lgb_params.get('lambda_l2')}, lambdarank_truncation_level={self.lgb_params.get('lambdarank_truncation_level')}, label_gain_power={self.label_gain_power}")
            train_dataset = lgb.Dataset(X_scaled, label=y_valid, group=groups)
            self.lightgbm_model = lgb.train(
                self.lgb_params,
                train_dataset,
                num_boost_round=self.num_boost_round,
                callbacks=[lgb.log_evaluation(period=0)]
            )
            # Set best_iteration for consistency (direct training uses all rounds)
            if not hasattr(self.lightgbm_model, 'best_iteration') or self.lightgbm_model.best_iteration is None:
                self.lightgbm_model.best_iteration = self.num_boost_round

        self.fitted_ = True
        logger.info("✅ Meta Ranker Stacker training complete")
        
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict using trained Meta Ranker Stacker
        
        Args:
            df: DataFrame with first-layer predictions
            
        Returns:
            DataFrame with 'score' column
        """
        if not self.fitted_ or self.lightgbm_model is None or self.scaler is None:
            raise RuntimeError("MetaRankerStacker not fitted")

        # Validate input
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("Prediction data must have MultiIndex(date, ticker)")

        # Get feature columns
        feature_cols = list(self.actual_feature_cols_ or self.base_cols)
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        # Prepare features
        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)

        # Predict - use best_iteration if available and valid, otherwise use all trees
        num_iteration = None
        if hasattr(self.lightgbm_model, 'best_iteration') and self.lightgbm_model.best_iteration is not None:
            num_iteration = self.lightgbm_model.best_iteration
        y_pred = self.lightgbm_model.predict(X_scaled, num_iteration=num_iteration)

        return pd.DataFrame({'score': y_pred}, index=df.index)

    def replace_ewa_in_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compatibility method for pipeline integration"""
        if not self.fitted_ or self.lightgbm_model is None or self.scaler is None:
            raise RuntimeError('MetaRankerStacker not fitted; call fit before replace_ewa_in_pipeline.')
        if not isinstance(df, pd.DataFrame):
            raise TypeError('replace_ewa_in_pipeline expects a pandas DataFrame.')
        if df.empty:
            raise ValueError('replace_ewa_in_pipeline received an empty DataFrame.')
        if not isinstance(df.index, pd.MultiIndex) or df.index.names != ['date', 'ticker']:
            raise ValueError('Input to replace_ewa_in_pipeline must have MultiIndex(date, ticker).')

        feature_cols = list(self.actual_feature_cols_ or self.base_cols)
        sanitized = df.copy()
        label_cols = [col for col in sanitized.columns if col.startswith('ret_fwd')]
        if label_cols:
            sanitized = sanitized.drop(columns=label_cols)

        missing = [col for col in feature_cols if col not in sanitized.columns]
        if missing:
            raise ValueError(f'replace_ewa_in_pipeline is missing required features: {missing}')

        sanitized = sanitized[feature_cols]
        valid_mask = ~sanitized.isna().any(axis=1)
        if not valid_mask.any():
            raise ValueError('replace_ewa_in_pipeline found no valid rows after dropping NaNs.')

        if not valid_mask.all():
            logger.debug('replace_ewa_in_pipeline dropping %d rows with NaN features', (~valid_mask).sum())

        valid_features = sanitized.loc[valid_mask]
        predictions = self.predict(valid_features)

        if valid_mask.all():
            return predictions

        full_output = pd.DataFrame(index=sanitized.index, columns=predictions.columns, dtype=float)
        full_output.loc[valid_mask] = predictions.values
        return full_output

    def get_model_info(self) -> Dict[str, any]:
        """Get model information"""
        return {
            'model_type': 'MetaRankerStacker',
            'base_cols': list(self.base_cols),
            'n_quantiles': self.n_quantiles,
            'label_gain_power': self.label_gain_power,
            'num_boost_round': self.num_boost_round,
            'lgb_params': self.lgb_params.copy(),
            'fitted': self.fitted_,
            'best_iteration': self.lightgbm_model.best_iteration if hasattr(self.lightgbm_model, 'best_iteration') else None
        }
