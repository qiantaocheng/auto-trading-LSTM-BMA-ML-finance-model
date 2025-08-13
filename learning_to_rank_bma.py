#!/usr/bin/env python3
"""
Learning-to-Rank增强BMA模块
实现排序优化、不确定性感知、Mixture-of-Experts等高级技术
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr, kendalltau, entropy
from scipy.optimize import minimize
import logging

# 尝试导入高级模型
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor, CatBoostRanker
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearningToRankBMA:
    """基于Learning-to-Rank的增强BMA系统"""
    
    def __init__(self, 
                 ranking_objective: str = "rank:pairwise",
                 uncertainty_method: str = "ensemble",
                 temperature: float = 1.2,
                 enable_regime_detection: bool = True):
        """
        初始化Learning-to-Rank BMA
        
        Args:
            ranking_objective: 排序目标函数
            uncertainty_method: 不确定性估计方法
            temperature: BMA温度系数
            enable_regime_detection: 是否启用状态检测
        """
        self.ranking_objective = ranking_objective
        self.uncertainty_method = uncertainty_method
        self.temperature = temperature
        self.enable_regime_detection = enable_regime_detection
        
        self.models = {}
        self.model_uncertainties = {}
        self.regime_weights = {}
        self.expert_gates = {}
        
        # 性能统计
        self.performance_stats = {
            'train_metrics': {},
            'oof_metrics': {},
            'ranking_metrics': {},
            'uncertainty_metrics': {}
        }
        
        logger.info(f"LearningToRankBMA初始化完成，排序目标: {ranking_objective}")
    
    def create_ranking_dataset(self, X: pd.DataFrame, y: pd.Series, 
                              dates: pd.Series, group_col: str = 'date') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        创建排序数据集
        
        Args:
            X: 特征矩阵
            y: 目标变量
            dates: 日期序列
            group_col: 分组列
            
        Returns:
            (特征矩阵, 目标, 分组ID)
        """
        logger.info("创建排序数据集")
        
        # 确保数据对齐
        if len(X) != len(y) or len(X) != len(dates):
            raise ValueError("X, y, dates长度不一致")
        
        # 组装并清洗
        df_temp = pd.DataFrame({'y': y, 'date': dates})
        df_temp = pd.concat([df_temp, X], axis=1)
        df_temp = df_temp.dropna()
        # 按日期排序，确保组内样本连续，便于CatBoost/LightGBM分组
        df_temp = df_temp.sort_values('date').reset_index(drop=True)
        
        if len(df_temp) == 0:
            raise ValueError("数据清洗后为空")
        
        # 创建分组ID（每个交易日为一组，且相同日期样本连续）
        group_ids = pd.factorize(df_temp['date'])[0]
        
        # 提取特征和目标
        feature_cols = [col for col in df_temp.columns if col not in ['y', 'date']]
        X_clean = df_temp[feature_cols].values
        y_clean = df_temp['y'].values
        
        logger.info(f"排序数据集创建完成: {X_clean.shape[0]}样本, {len(np.unique(group_ids))}组")
        
        return X_clean, y_clean, group_ids

    def _discretize_labels_by_group(self, y: np.ndarray, group_ids: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """增强的标签离散化：更细致分箱+标准化处理"""
        y_series = pd.Series(y)
        g_series = pd.Series(group_ids)
        labels = np.zeros_like(y, dtype=int)
        
        for g in np.unique(group_ids):
            mask = (g_series == g)
            if mask.sum() == 0:
                continue
            try:
                y_group = y_series[mask]
                # 方法1：基于组内标准化的分位数分箱
                if y_group.std() > 0:
                    y_norm = (y_group - y_group.mean()) / y_group.std()
                    # 使用标准正态分位点
                    percentiles = np.linspace(0, 100, n_bins + 1)
                    bins = np.percentile(y_norm, percentiles)
                    bins = np.unique(bins)  # 去重
                    if len(bins) > 1:
                        qbins = pd.cut(y_norm, bins=bins, labels=False, duplicates='drop')
                    else:
                        qbins = np.zeros(len(y_group), dtype=int)
                else:
                    qbins = np.zeros(len(y_group), dtype=int)
                
                # 方法2：回退到简单分位数
                if qbins is None or pd.isna(qbins).all():
                    qbins = pd.qcut(y_group, q=min(n_bins, mask.sum()), labels=False, duplicates='drop')
                
                # 归一化并填充NaN
                if qbins is not None:
                    qbins = pd.Series(qbins).fillna(0).astype(int)
                    labels[mask.values] = qbins.values
                    
            except Exception:
                # 最终回退：线性分割
                y_group = y_series[mask]
                ranks = y_group.rank(method='first')
                q = np.floor((ranks - 1) / (max(1, mask.sum() // n_bins)))
                labels[mask.values] = np.clip(q.astype(int), 0, n_bins - 1)
                
        logger.info(f"标签离散化完成：{len(np.unique(labels))}个等级，分布: {np.bincount(labels)}")
        return labels
    
    def train_ranking_models(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series,
                           cv_folds: int = 5, optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """
        训练多个排序模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            dates: 日期序列
            cv_folds: 交叉验证折数
            optimize_hyperparams: 是否优化超参数
            
        Returns:
            训练结果字典
        """
        logger.info("开始训练排序模型")
        
        # 创建排序数据集
        X_rank, y_rank, group_ids = self.create_ranking_dataset(X, y, dates)
        # 为LightGBM准备离散标签
        y_rank_discrete = self._discretize_labels_by_group(y_rank, group_ids, n_bins=5)
        # 创建Purged CV（按周分组示例）
        try:
            from purged_time_series_cv import ValidationConfig, PurgedGroupTimeSeriesSplit
            cv = PurgedGroupTimeSeriesSplit(ValidationConfig(
                n_splits=cv_folds, test_size=63, gap=5, embargo=2, min_train_size=252, group_freq='W'
            ))
            unique_groups = np.unique(group_ids)
            cv_splits = list(cv.split(X_rank, y_rank, group_ids))
        except Exception:
            # 回退到简单TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            unique_groups = np.unique(group_ids)
            cv_splits = list(tscv.split(unique_groups))
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        unique_groups = np.unique(group_ids)
        
        models_results = {}
        
        # 1. XGBoost排序模型
        if XGBOOST_AVAILABLE:
            logger.info("训练XGBoost排序模型")
            xgb_results = self._train_xgboost_ranker(
                X_rank, y_rank, group_ids, cv_splits, unique_groups, optimize_hyperparams
            )
            models_results['xgboost_ranker'] = xgb_results
        
        # 2. LightGBM排序模型
        if LIGHTGBM_AVAILABLE:
            logger.info("训练LightGBM排序模型")
            lgb_results = self._train_lightgbm_ranker(
                X_rank, y_rank_discrete, group_ids, cv_splits, unique_groups, optimize_hyperparams
            )
            models_results['lightgbm_ranker'] = lgb_results
        
        # 3. CatBoost排序模型
        if CATBOOST_AVAILABLE:
            logger.info("训练CatBoost排序模型")
            cat_results = self._train_catboost_ranker(
                X_rank, y_rank, group_ids, cv_splits, unique_groups, optimize_hyperparams
            )
            models_results['catboost_ranker'] = cat_results
        
        # 4. 分位数回归模型
        logger.info("训练分位数回归模型")
        quantile_results = self._train_quantile_models(
            X_rank, y_rank, group_ids, tscv, unique_groups
        )
        models_results['quantile_models'] = quantile_results
        
        # 5. 传统回归模型（作为基准）
        logger.info("训练传统回归模型")
        baseline_results = self._train_baseline_models(
            X_rank, y_rank, group_ids, tscv, unique_groups
        )
        models_results['baseline_models'] = baseline_results
        
        self.models = models_results
        
        # 计算模型性能和不确定性
        self._evaluate_model_performance(X_rank, y_rank, group_ids)
        
        logger.info(f"排序模型训练完成，共{len(models_results)}类模型")
        
        return models_results
    
    def _train_xgboost_ranker(self, X: np.ndarray, y: np.ndarray, group_ids: np.ndarray,
                             cv_splits, unique_groups: np.ndarray,
                             optimize_hyperparams: bool) -> Dict[str, Any]:
        """训练XGBoost排序模型"""
        if not XGBOOST_AVAILABLE:
            return {}
        
        # 计算每组大小
        group_sizes = [np.sum(group_ids == g) for g in unique_groups]
        
        models = []
        oof_predictions = np.full(len(X), np.nan)
        oof_uncertainties = np.full(len(X), np.nan)
        
        # 兼容两种split形式：基于组索引或直接样本索引
        for split in cv_splits:
            if isinstance(split[0][0], (np.integer, int)) and len(split[0].shape) == 1:
                # 样本索引
                train_mask = np.zeros(len(X), dtype=bool)
                train_mask[split[0]] = True
                test_mask = np.zeros(len(X), dtype=bool)
                test_mask[split[1]] = True
                train_groups = np.unique(group_ids[train_mask])
                test_groups = np.unique(group_ids[test_mask])
            else:
                # 组索引
                train_groups_idx, test_groups_idx = split
                train_groups = unique_groups[train_groups_idx]
                test_groups = unique_groups[test_groups_idx]
            
            # 获取训练和测试数据
            train_mask = np.isin(group_ids, train_groups)
            test_mask = np.isin(group_ids, test_groups)
            
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_test = X[test_mask]
            
            # 计算训练集的组大小
            train_group_ids = group_ids[train_mask]
            train_group_sizes = [np.sum(train_group_ids == g) for g in train_groups]
            
            # XGBoost排序参数
            if optimize_hyperparams:
                params = {
                    'objective': self.ranking_objective,
                    'eval_metric': 'ndcg@10',
                    'eta': 0.1,
                    'max_depth': 6,
                    'min_child_weight': 1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'lambda': 1.0,
                    'alpha': 0.0,
                    'silent': 1,
                    'nthread': -1
                }
            else:
                params = {
                    'objective': self.ranking_objective,
                    'eval_metric': 'ndcg@10',
                    'eta': 0.1,
                    'max_depth': 6,
                    'silent': 1
                }
            
            # 创建DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtrain.set_group(train_group_sizes)
            
            dtest = xgb.DMatrix(X_test)
            
            # 训练模型
            try:
                model = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=100,
                    verbose_eval=False
                )
                
                # 预测
                test_pred = model.predict(dtest)
                
                # Bootstrap不确定性估计
                n_bootstrap = 10
                bootstrap_preds = []
                
                for _ in range(n_bootstrap):
                    # 重采样训练数据
                    n_samples = len(X_train)
                    bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
                    X_bootstrap = X_train[bootstrap_idx]
                    y_bootstrap = y_train[bootstrap_idx]
                    
                    dtrain_bootstrap = xgb.DMatrix(X_bootstrap, label=y_bootstrap)
                    # 近似组大小（简化）
                    approx_group_sizes = [len(X_bootstrap) // len(train_groups)] * len(train_groups)
                    dtrain_bootstrap.set_group(approx_group_sizes)
                    
                    try:
                        bootstrap_model = xgb.train(
                            params=params,
                            dtrain=dtrain_bootstrap,
                            num_boost_round=50,
                            verbose_eval=False
                        )
                        bootstrap_pred = bootstrap_model.predict(dtest)
                        bootstrap_preds.append(bootstrap_pred)
                    except:
                        continue
                
                if bootstrap_preds:
                    test_uncertainty = np.std(bootstrap_preds, axis=0)
                else:
                    test_uncertainty = np.ones(len(test_pred)) * 0.1
                
                # 保存OOF预测
                oof_predictions[test_mask] = test_pred
                oof_uncertainties[test_mask] = test_uncertainty
                
                models.append(model)
                
            except Exception as e:
                logger.warning(f"XGBoost训练失败: {e}")
                continue
        
        return {
            'models': models,
            'oof_predictions': oof_predictions,
            'oof_uncertainties': oof_uncertainties,
            'model_type': 'xgboost_ranker'
        }
    
    def _train_lightgbm_ranker(self, X: np.ndarray, y: np.ndarray, group_ids: np.ndarray,
                              cv_splits, unique_groups: np.ndarray,
                              optimize_hyperparams: bool) -> Dict[str, Any]:
        """训练LightGBM排序模型"""
        if not LIGHTGBM_AVAILABLE:
            return {}
        
        models = []
        oof_predictions = np.full(len(X), np.nan)
        oof_uncertainties = np.full(len(X), np.nan)
        
        for split in cv_splits:
            if isinstance(split[0][0], (np.integer, int)) and len(split[0].shape) == 1:
                # 样本索引
                train_mask = np.zeros(len(X), dtype=bool)
                train_mask[split[0]] = True
                test_mask = np.zeros(len(X), dtype=bool)
                test_mask[split[1]] = True
                train_groups = np.unique(group_ids[train_mask])
                test_groups = np.unique(group_ids[test_mask])
            else:
                train_groups_idx, test_groups_idx = split
                train_groups = unique_groups[train_groups_idx]
                test_groups = unique_groups[test_groups_idx]
            
            train_mask = np.isin(group_ids, train_groups)
            test_mask = np.isin(group_ids, test_groups)
            
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_test = X[test_mask]
            
            # 计算组大小
            train_group_ids = group_ids[train_mask]
            train_group_sizes = [np.sum(train_group_ids == g) for g in train_groups]
            
            # LightGBM排序参数
            if optimize_hyperparams:
                params = {
                    'objective': 'lambdarank',
                    'metric': 'ndcg',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1
                }
            else:
                params = {
                    'objective': 'lambdarank',
                    'metric': 'ndcg',
                    'learning_rate': 0.1,
                    'verbose': -1
                }
            
            try:
                # 为LightGBM创建离散化标签和权重
                y_discrete = self._discretize_labels_by_group(y_train, train_group_ids, n_bins=15)  # 增加分箱精度
                
                # 生成label_gain: 基于原始连续y的分位数权重
                label_gains = []
                for g in np.unique(train_group_ids):
                    mask = train_group_ids == g
                    if mask.sum() > 0:
                        y_group = y_train[mask]
                        ranks = y_group.argsort().argsort() + 1  # 1-based ranking
                        gains = np.log2(ranks + 1)  # NDCG-style gains
                        label_gains.extend(gains)
                    
                label_gains = np.array(label_gains)
                
                # 创建数据集（使用离散标签和自定义权重）
                train_data = lgb.Dataset(
                    X_train, 
                    label=y_discrete,
                    group=train_group_sizes,
                    weight=label_gains  # 添加label_gain权重
                )
                
                # 训练模型
                model = lgb.train(
                    params=params,
                    train_set=train_data,
                    num_boost_round=150  # 增加训练轮数
                )
                
                # 预测
                test_pred = model.predict(X_test)
                
                # 简化的不确定性估计
                test_uncertainty = np.ones(len(test_pred)) * 0.1
                
                oof_predictions[test_mask] = test_pred
                oof_uncertainties[test_mask] = test_uncertainty
                
                models.append(model)
                
            except Exception as e:
                logger.warning(f"LightGBM训练失败: {e}")
                continue
        
        return {
            'models': models,
            'oof_predictions': oof_predictions,
            'oof_uncertainties': oof_uncertainties,
            'model_type': 'lightgbm_ranker'
        }
    
    def _train_catboost_ranker(self, X: np.ndarray, y: np.ndarray, group_ids: np.ndarray,
                              cv_splits, unique_groups: np.ndarray,
                              optimize_hyperparams: bool) -> Dict[str, Any]:
        """训练CatBoost排序模型"""
        if not CATBOOST_AVAILABLE:
            return {}
        
        models = []
        oof_predictions = np.full(len(X), np.nan)
        oof_uncertainties = np.full(len(X), np.nan)
        
        for split in cv_splits:
            if isinstance(split[0][0], (np.integer, int)) and len(split[0].shape) == 1:
                # 样本索引
                train_mask = np.zeros(len(X), dtype=bool)
                train_mask[split[0]] = True
                test_mask = np.zeros(len(X), dtype=bool)
                test_mask[split[1]] = True
            else:
                train_groups_idx, test_groups_idx = split
                train_groups = unique_groups[train_groups_idx]
                test_groups = unique_groups[test_groups_idx]
                train_mask = np.isin(group_ids, train_groups)
                test_mask = np.isin(group_ids, test_groups)
            
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_test = X[test_mask]
            
            # 计算组大小
            train_group_ids = group_ids[train_mask]
            # 确保组内样本连续：已在创建数据阶段按日期排序+factorize
            
            # CatBoost排序参数
            if optimize_hyperparams:
                params = {
                    'loss_function': 'YetiRank',
                    'iterations': 100,
                    'learning_rate': 0.1,
                    'depth': 6,
                    'l2_leaf_reg': 3,
                    'verbose': False
                }
            else:
                params = {
                    'loss_function': 'YetiRank',
                    'iterations': 100,
                    'verbose': False
                }
            
            try:
                model = CatBoostRanker(**params)
                # 训练模型（CatBoost 要求 group_id 与训练样本一一对应）
                model.fit(X_train, y_train, group_id=train_group_ids, verbose=False)
                
                # 预测
                test_pred = model.predict(X_test)
                
                # 简化的不确定性估计
                test_uncertainty = np.ones(len(test_pred)) * 0.1
                
                oof_predictions[test_mask] = test_pred
                oof_uncertainties[test_mask] = test_uncertainty
                
                models.append(model)
                
            except Exception as e:
                logger.warning(f"CatBoost训练失败: {e}")
                continue
        
        return {
            'models': models,
            'oof_predictions': oof_predictions,
            'oof_uncertainties': oof_uncertainties,
            'model_type': 'catboost_ranker'
        }
    
    def _train_quantile_models(self, X: np.ndarray, y: np.ndarray, group_ids: np.ndarray,
                              tscv: TimeSeriesSplit, unique_groups: np.ndarray) -> Dict[str, Any]:
        """训练分位数回归模型"""
        models = {}
        oof_predictions = {}
        
        # 训练多个分位数
        quantiles = [0.1, 0.25, 0.5, 0.7, 0.9]
        
        for quantile in quantiles:
            models[f'q{int(quantile*100)}'] = []
            oof_predictions[f'q{int(quantile*100)}'] = np.full(len(X), np.nan)
            
            for train_groups_idx, test_groups_idx in tscv.split(unique_groups):
                train_groups = unique_groups[train_groups_idx]
                test_groups = unique_groups[test_groups_idx]
                
                train_mask = np.isin(group_ids, train_groups)
                test_mask = np.isin(group_ids, test_groups)
                
                if train_mask.sum() == 0 or test_mask.sum() == 0:
                    continue
                
                X_train, y_train = X[train_mask], y[train_mask]
                X_test = X[test_mask]
                
                try:
                    # 限制训练集规模，避免LP求解耗时过长
                    max_train_samples = 5000
                    if len(X_train) > max_train_samples:
                        # 选择最近的训练样本（按组时间顺序）
                        recent_idx = np.arange(len(X_train) - max_train_samples, len(X_train))
                        X_train_small = X_train[recent_idx]
                        y_train_small = y_train[recent_idx]
                    else:
                        X_train_small = X_train
                        y_train_small = y_train

                    # 分位数回归 - 使用更稳健的参数
                    model = QuantileRegressor(
                        quantile=quantile, 
                        alpha=0.1, 
                        solver='highs',
                        solver_options={'presolve': 'off', 'time_limit': 2.0}
                    )
                    model.fit(X_train_small, y_train_small)
                    
                    test_pred = model.predict(X_test)
                    oof_predictions[f'q{int(quantile*100)}'][test_mask] = test_pred
                    
                    models[f'q{int(quantile*100)}'].append(model)
                    
                except Exception as e:
                    logger.warning(f"分位数回归训练失败 (q={quantile}): {e}")
                    continue
        
        # 计算不确定性（使用分位数范围）
        oof_uncertainties = np.full(len(X), np.nan)
        if 'q90' in oof_predictions and 'q10' in oof_predictions:
            q90 = oof_predictions['q90']
            q10 = oof_predictions['q10']
            oof_uncertainties = (q90 - q10) / 2
        
        return {
            'models': models,
            'oof_predictions': oof_predictions['q50'] if 'q50' in oof_predictions else np.full(len(X), np.nan),
            'oof_uncertainties': oof_uncertainties,
            'quantile_predictions': oof_predictions,
            'model_type': 'quantile_regression'
        }
    
    def _train_baseline_models(self, X: np.ndarray, y: np.ndarray, group_ids: np.ndarray,
                              tscv: TimeSeriesSplit, unique_groups: np.ndarray) -> Dict[str, Any]:
        """训练基准回归模型"""
        baseline_models = {
            'linear': LinearRegression(),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for model_name, base_model in baseline_models.items():
            models = []
            oof_predictions = np.full(len(X), np.nan)
            oof_uncertainties = np.full(len(X), np.nan)
            
            for train_groups_idx, test_groups_idx in tscv.split(unique_groups):
                train_groups = unique_groups[train_groups_idx]
                test_groups = unique_groups[test_groups_idx]
                
                train_mask = np.isin(group_ids, train_groups)
                test_mask = np.isin(group_ids, test_groups)
                
                if train_mask.sum() == 0 or test_mask.sum() == 0:
                    continue
                
                X_train, y_train = X[train_mask], y[train_mask]
                X_test = X[test_mask]
                
                try:
                    model = clone(base_model)
                    model.fit(X_train, y_train)
                    
                    test_pred = model.predict(X_test)
                    
                    # 简化的不确定性估计
                    if hasattr(model, 'predict_proba'):
                        # 对于能输出概率的模型，使用概率分布的方差
                        test_uncertainty = np.ones(len(test_pred)) * 0.1
                    else:
                        test_uncertainty = np.ones(len(test_pred)) * 0.1
                    
                    oof_predictions[test_mask] = test_pred
                    oof_uncertainties[test_mask] = test_uncertainty
                    
                    models.append(model)
                    
                except Exception as e:
                    logger.warning(f"{model_name}模型训练失败: {e}")
                    continue
            
            results[model_name] = {
                'models': models,
                'oof_predictions': oof_predictions,
                'oof_uncertainties': oof_uncertainties,
                'model_type': f'baseline_{model_name}'
            }
        
        return results
    
    def _evaluate_model_performance(self, X: np.ndarray, y: np.ndarray, group_ids: np.ndarray):
        """评估模型性能"""
        logger.info("评估模型性能")
        
        for model_category, model_results in self.models.items():
            if isinstance(model_results, dict) and 'oof_predictions' in model_results:
                oof_pred = model_results['oof_predictions']
                oof_unc = model_results.get('oof_uncertainties', np.ones(len(oof_pred)) * 0.1)
                
                # 去除NaN值
                valid_mask = ~(np.isnan(oof_pred) | np.isnan(y))
                if valid_mask.sum() < 10:
                    continue
                
                y_valid = y[valid_mask]
                pred_valid = oof_pred[valid_mask]
                unc_valid = oof_unc[valid_mask]
                group_valid = group_ids[valid_mask]
                
                # 计算各种指标
                metrics = self._calculate_metrics(y_valid, pred_valid, group_valid)
                
                # 不确定性指标
                uncertainty_metrics = self._calculate_uncertainty_metrics(
                    y_valid, pred_valid, unc_valid
                )
                
                self.performance_stats['oof_metrics'][model_category] = metrics
                self.performance_stats['uncertainty_metrics'][model_category] = uncertainty_metrics
                
                logger.info(f"{model_category} - IC: {metrics.get('ic', 0):.4f}, "
                           f"RankIC: {metrics.get('rank_ic', 0):.4f}")
            
            # 处理嵌套模型结果（如baseline_models）
            elif isinstance(model_results, dict):
                for sub_model, sub_results in model_results.items():
                    if isinstance(sub_results, dict) and 'oof_predictions' in sub_results:
                        # 类似处理...
                        pass
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          groups: np.ndarray) -> Dict[str, float]:
        """计算预测指标"""
        metrics = {}
        
        try:
            # 信息系数
            ic = np.corrcoef(y_true, y_pred)[0, 1]
            metrics['ic'] = ic if not np.isnan(ic) else 0.0
            
            # 排序信息系数
            rank_ic = spearmanr(y_true, y_pred)[0]
            metrics['rank_ic'] = rank_ic if not np.isnan(rank_ic) else 0.0
            
            # NDCG (简化版本)
            ndcg = self._calculate_ndcg(y_true, y_pred, groups)
            metrics['ndcg'] = ndcg
            
            # 分组IC平均值
            group_ics = []
            for group_id in np.unique(groups):
                group_mask = groups == group_id
                if group_mask.sum() > 5:  # 至少5个样本
                    y_group = y_true[group_mask]
                    pred_group = y_pred[group_mask]
                    group_ic = np.corrcoef(y_group, pred_group)[0, 1]
                    if not np.isnan(group_ic):
                        group_ics.append(group_ic)
            
            metrics['mean_group_ic'] = np.mean(group_ics) if group_ics else 0.0
            metrics['ic_std'] = np.std(group_ics) if group_ics else 0.0
            metrics['ic_ir'] = metrics['mean_group_ic'] / (metrics['ic_std'] + 1e-12)
            
        except Exception as e:
            logger.warning(f"指标计算失败: {e}")
            metrics = {'ic': 0.0, 'rank_ic': 0.0, 'ndcg': 0.0, 'mean_group_ic': 0.0, 'ic_std': 1.0, 'ic_ir': 0.0}
        
        return metrics
    
    def _calculate_ndcg(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       groups: np.ndarray, k: int = 10) -> float:
        """计算NDCG@k"""
        ndcg_scores = []
        
        for group_id in np.unique(groups):
            group_mask = groups == group_id
            if group_mask.sum() < k:
                continue
            
            y_group = y_true[group_mask]
            pred_group = y_pred[group_mask]
            
            # 按预测值排序
            sorted_indices = np.argsort(pred_group)[::-1][:k]
            
            # DCG
            dcg = 0.0
            for i, idx in enumerate(sorted_indices):
                rel = y_group[idx]
                dcg += rel / np.log2(i + 2)
            
            # IDCG
            ideal_sorted = np.argsort(y_group)[::-1][:k]
            idcg = 0.0
            for i, idx in enumerate(ideal_sorted):
                rel = y_group[idx]
                idcg += rel / np.log2(i + 2)
            
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def _calculate_uncertainty_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     uncertainty: np.ndarray) -> Dict[str, float]:
        """计算不确定性指标"""
        metrics = {}
        
        try:
            # 校准指标：不确定性高的样本误差是否更大
            errors = np.abs(y_true - y_pred)
            calibration_corr = np.corrcoef(uncertainty, errors)[0, 1]
            metrics['uncertainty_calibration'] = calibration_corr if not np.isnan(calibration_corr) else 0.0
            
            # 分位数校准
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_errors = []
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # 找到这个不确定性分位数范围内的样本
                bin_mask = (uncertainty >= np.quantile(uncertainty, bin_lower)) & \
                          (uncertainty <= np.quantile(uncertainty, bin_upper))
                
                if bin_mask.sum() > 0:
                    bin_errors = errors[bin_mask]
                    expected_error = np.mean(bin_errors)
                    predicted_uncertainty = np.mean(uncertainty[bin_mask])
                    calibration_errors.append(abs(expected_error - predicted_uncertainty))
            
            metrics['calibration_error'] = np.mean(calibration_errors) if calibration_errors else 0.0
            
            # 覆盖概率
            confidence_levels = [0.68, 0.95]
            for confidence in confidence_levels:
                threshold = np.quantile(uncertainty, confidence)
                high_conf_mask = uncertainty <= threshold
                
                if high_conf_mask.sum() > 0:
                    high_conf_errors = errors[high_conf_mask]
                    coverage = np.mean(high_conf_errors <= threshold)
                    metrics[f'coverage_{int(confidence*100)}'] = coverage
            
        except Exception as e:
            logger.warning(f"不确定性指标计算失败: {e}")
            metrics = {'uncertainty_calibration': 0.0, 'calibration_error': 1.0}
        
        return metrics
    
    def compute_uncertainty_aware_bma_weights(self, alpha_predictions: Dict[str, np.ndarray],
                                             alpha_uncertainties: Dict[str, np.ndarray],
                                             performance_scores: Dict[str, float]) -> Dict[str, float]:
        """
        计算不确定性感知的BMA权重
        
        Args:
            alpha_predictions: Alpha预测字典
            alpha_uncertainties: Alpha不确定性字典
            performance_scores: 性能评分字典
            
        Returns:
            BMA权重字典
        """
        logger.info("计算不确定性感知的BMA权重")
        
        weights = {}
        
        for alpha_name in alpha_predictions.keys():
            if alpha_name not in performance_scores:
                weights[alpha_name] = 0.0
                continue
            
            # 基础性能分数
            base_score = performance_scores[alpha_name]
            
            # 不确定性调整
            if alpha_name in alpha_uncertainties:
                uncertainty = alpha_uncertainties[alpha_name]
                # 平均不确定性越低，权重加成越大
                avg_uncertainty = np.nanmean(uncertainty)
                uncertainty_factor = 1.0 / (1.0 + avg_uncertainty)
            else:
                uncertainty_factor = 1.0
            
            # 调整后的分数
            adjusted_score = base_score * uncertainty_factor
            weights[alpha_name] = adjusted_score
        
        # 标准化权重（softmax with temperature）
        if weights:
            scores_array = np.array(list(weights.values()))
            
            # 标准化
            scores_std = (scores_array - scores_array.mean()) / (scores_array.std() + 1e-12)
            scores_scaled = scores_std / self.temperature
            
            # Softmax
            exp_scores = np.exp(scores_scaled - scores_scaled.max())
            weights_normalized = exp_scores / exp_scores.sum()
            
            # 更新权重字典
            for i, alpha_name in enumerate(weights.keys()):
                weights[alpha_name] = weights_normalized[i]
        
        logger.info(f"BMA权重计算完成: {weights}")
        return weights
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用训练好的模型进行预测，同时输出不确定性
        
        Args:
            X: 特征矩阵
            
        Returns:
            (预测值, 不确定性)
        """
        if not self.models:
            raise ValueError("模型未训练")
        
        all_predictions = []
        all_uncertainties = []
        
        for model_category, model_results in self.models.items():
            if isinstance(model_results, dict) and 'models' in model_results:
                models = model_results['models']
                if not models:
                    continue
                
                # 集成预测
                category_predictions = []
                for model in models:
                    try:
                        if hasattr(model, 'predict'):
                            # 特殊处理xgboost Booster
                            try:
                                import xgboost as xgb
                                if isinstance(model, xgb.Booster):
                                    dmat = xgb.DMatrix(X.values)
                                    pred = model.predict(dmat)
                                else:
                                    pred = model.predict(X.values)
                            except Exception:
                                pred = model.predict(X.values)
                            category_predictions.append(pred)
                    except Exception as e:
                        logger.warning(f"预测失败: {e}")
                        continue
                
                if category_predictions:
                    # 平均预测
                    mean_pred = np.mean(category_predictions, axis=0)
                    # 预测方差作为不确定性
                    pred_uncertainty = np.std(category_predictions, axis=0)
                    
                    all_predictions.append(mean_pred)
                    all_uncertainties.append(pred_uncertainty)
        
        if not all_predictions:
            raise ValueError("没有有效的预测结果")
        
        # 简单平均所有模型类别的预测
        final_prediction = np.mean(all_predictions, axis=0)
        final_uncertainty = np.sqrt(np.mean(np.array(all_uncertainties)**2, axis=0))
        
        return final_prediction, final_uncertainty
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能总结"""
        return {
            'performance_stats': self.performance_stats,
            'model_types': list(self.models.keys()),
            'total_models': sum(len(v.get('models', [])) if isinstance(v, dict) else 1 
                               for v in self.models.values())
        }
    
    def calibrate_predictions_to_returns(self, scores: np.ndarray, returns: np.ndarray, 
                                       method: str = 'isotonic') -> Tuple[np.ndarray, Any]:
        """
        将模型分数校准为预期收益率 - Enhanced风格核心功能
        
        Args:
            scores: 模型预测分数
            returns: 真实收益率
            method: 校准方法 ('isotonic', 'quantile_bins', 'linear')
            
        Returns:
            Tuple[校准后的收益率预测, 校准器对象]
        """
        try:
            if method == 'isotonic':
                from sklearn.isotonic import IsotonicRegression
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(scores, returns)
                calibrated_returns = calibrator.predict(scores)
                
            elif method == 'quantile_bins':
                # 分位数桶校准 - 更稳健的分段线性映射
                n_bins = 20
                score_quantiles = np.linspace(0, 1, n_bins + 1)
                bin_edges = np.quantile(scores, score_quantiles)
                
                # 创建分位数映射
                bin_means = {}
                for i in range(n_bins):
                    mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
                    if mask.sum() > 0:
                        bin_means[i] = returns[mask].mean()
                    else:
                        bin_means[i] = 0.0
                
                # 应用校准
                calibrated_returns = np.zeros_like(scores)
                for i in range(n_bins):
                    mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
                    calibrated_returns[mask] = bin_means[i]
                
                calibrator = {'bin_edges': bin_edges, 'bin_means': bin_means}
            
            else:
                # 线性校准 - 简单但稳定
                from sklearn.linear_model import LinearRegression
                calibrator = LinearRegression()
                calibrator.fit(scores.reshape(-1, 1), returns)
                calibrated_returns = calibrator.predict(scores.reshape(-1, 1))
            
            # 计算校准质量
            correlation = np.corrcoef(calibrated_returns, returns)[0,1]
            logger.info(f"使用{method}方法完成分数校准，校准后相关性: {correlation:.3f}")
            
            return calibrated_returns, calibrator
            
        except Exception as e:
            logger.warning(f"分数校准失败: {e}")
            # 回退到简单线性映射
            returns_std = np.std(returns) if np.std(returns) > 1e-8 else 0.02
            scores_std = np.std(scores) if np.std(scores) > 1e-8 else 1.0
            slope = returns_std / scores_std
            calibrated_returns = scores * slope
            return calibrated_returns, {'slope': slope, 'method': 'linear_fallback'}
    
    def apply_calibration(self, scores: np.ndarray, calibrator: Any, method: str = 'isotonic') -> np.ndarray:
        """应用已训练的校准器到新分数"""
        try:
            if method == 'isotonic' and hasattr(calibrator, 'predict'):
                return calibrator.predict(scores)
            elif method == 'quantile_bins' and isinstance(calibrator, dict):
                bin_edges = calibrator['bin_edges']
                bin_means = calibrator['bin_means']
                
                calibrated = np.zeros_like(scores)
                for i in range(len(bin_edges) - 1):
                    mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
                    calibrated[mask] = bin_means.get(i, 0.0)
                return calibrated
            elif hasattr(calibrator, 'predict'):
                return calibrator.predict(scores.reshape(-1, 1))
            else:
                # 线性映射回退
                slope = calibrator.get('slope', 1.0)
                return scores * slope
                
        except Exception as e:
            logger.warning(f"校准应用失败: {e}")
            return scores


def clone(estimator):
    """简单的模型克隆函数"""
    from copy import deepcopy
    return deepcopy(estimator)


# ============ 测试代码 ============

def test_learning_to_rank_bma():
    """测试Learning-to-Rank BMA"""
    
    # 生成模拟数据
    np.random.seed(42)
    n_dates = 100
    n_stocks = 200
    n_features = 10
    
    dates = pd.date_range('2020-01-01', periods=n_dates, freq='D')
    
    data = []
    for date in dates:
        for stock_id in range(n_stocks):
            # 生成特征
            features = np.random.randn(n_features)
            
            # 生成目标变量（有一定的信号）
            signal = np.sum(features[:3] * [0.1, -0.05, 0.08])
            noise = np.random.randn() * 0.2
            target = signal + noise
            
            row = {'date': date, 'stock_id': stock_id, 'target': target}
            for i, feat in enumerate(features):
                row[f'feature_{i}'] = feat
            
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # 准备数据
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    X = df[feature_cols]
    y = df['target']
    dates = df['date']
    
    # 初始化Learning-to-Rank BMA
    ltrank_bma = LearningToRankBMA(
        ranking_objective="rank:pairwise",
        uncertainty_method="ensemble",
        temperature=1.2
    )
    
    # 训练模型
    logger.info("开始训练Learning-to-Rank模型")
    training_results = ltrank_bma.train_ranking_models(
        X=X, y=y, dates=dates, cv_folds=3, optimize_hyperparams=False
    )
    
    print(f"训练完成，模型类别: {list(training_results.keys())}")
    
    # 获取性能总结
    performance_summary = ltrank_bma.get_performance_summary()
    print(f"性能总结: {performance_summary}")
    
    # 测试预测
    test_X = X.iloc[:100]  # 使用前100个样本作为测试
    try:
        predictions, uncertainties = ltrank_bma.predict_with_uncertainty(test_X)
        print(f"预测完成，预测形状: {predictions.shape}, 不确定性形状: {uncertainties.shape}")
        print(f"预测统计: {pd.Series(predictions).describe()}")
        print(f"不确定性统计: {pd.Series(uncertainties).describe()}")
    except Exception as e:
        print(f"预测失败: {e}")


if __name__ == "__main__":
    test_learning_to_rank_bma()
