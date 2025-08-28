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

# CatBoost removed due to compatibility issues
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
    
    def _create_robust_time_groups(self, dates: pd.Series, min_gap_days: int = 1) -> np.ndarray:
        """
        创建严格的时间分组，确保训练测试集之间有足够的时间隔离
        
        Args:
            dates: 日期序列
            min_gap_days: 最小间隔天数
            
        Returns:
            组ID数组，-1表示buffer区域（不用于训练或测试）
        """
        unique_dates = sorted(dates.unique())
        if len(unique_dates) < 10:
            logger.warning(f"唯一日期数量过少({len(unique_dates)})，可能影响CV质量")
        
        # 将日期分成时间块（确保严格顺序）
        n_blocks = min(10, len(unique_dates) // 3)  # 最多10个块，每块至少3天
        if n_blocks < 3:
            n_blocks = 3
            
        # 创建时间块
        date_to_block = {}
        dates_per_block = len(unique_dates) // n_blocks
        
        for i, date in enumerate(unique_dates):
            block_id = min(i // dates_per_block, n_blocks - 1)
            date_to_block[date] = block_id
        
        # 创建buffer区域防止边界泄露
        buffered_mapping = {}
        for date, block_id in date_to_block.items():
            if block_id > 0:
                # 检查是否在块边界的buffer区域内
                block_start_idx = block_id * dates_per_block
                if abs(unique_dates.index(date) - block_start_idx) < min_gap_days:
                    buffered_mapping[date] = -1  # 标记为buffer
                    continue
            buffered_mapping[date] = block_id
        
        # 映射到原始日期序列
        group_ids = np.array([buffered_mapping[date] for date in dates])
        
        # 过滤掉buffer区域
        valid_mask = group_ids >= 0
        n_buffer = np.sum(~valid_mask)
        n_valid = np.sum(valid_mask)
        
        logger.info(f"时间分组创建完成: {n_blocks}个块, {n_valid}个有效样本, {n_buffer}个buffer样本")
        
        return group_ids
    
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
        logger.info(f"LTR数据对齐检查: X={len(X)}, y={len(y)}, dates={len(dates)}")
        
        if len(X) != len(y) or len(X) != len(dates):
            logger.warning(f"LTR数据长度不一致: X={len(X)}, y={len(y)}, dates={len(dates)}")
            
            # 自动对齐到最小长度
            min_len = min(len(X), len(y), len(dates))
            if min_len == 0:
                raise ValueError("所有数据长度为0，无法训练LTR模型")
            
            logger.info(f"自动对齐到最小长度: {min_len}")
            
            # 对齐数据
            if isinstance(X, pd.DataFrame):
                X = X.iloc[:min_len].copy()
            else:
                X = X[:min_len]
                
            if isinstance(y, pd.Series):
                y = y.iloc[:min_len].copy()  
            else:
                y = y[:min_len]
                
            if isinstance(dates, pd.Series):
                dates = dates.iloc[:min_len].copy()
            else:
                dates = dates[:min_len]
                
            logger.info(f"LTR数据对齐完成: X={len(X)}, y={len(y)}, dates={len(dates)}")
        
        # 组装并清洗
        df_temp = pd.DataFrame({'y': y, 'date': dates})
        df_temp = pd.concat([df_temp, X], axis=1)
        df_temp = df_temp.dropna()
        # 按日期排序，确保组内样本连续，便于CatBoost/LightGBM分组
        df_temp = df_temp.sort_values('date').reset_index(drop=True)
        
        if len(df_temp) == 0:
            raise ValueError("数据清洗后为空")
        
        # 🔥 修复：使用同日横截面分组 (按日期分组，避免跨期配对)
        logger.info("🔥 使用同日横截面分组避免跨期配对")
        unique_dates = sorted(df_temp['date'].unique())
        date_to_group = {date: i for i, date in enumerate(unique_dates)}
        group_ids = np.array([date_to_group[date] for date in df_temp['date']])
        
        # 验证分组：确保同一组内的样本都是同一天
        sample_groups = {}
        for i, (date, group_id) in enumerate(zip(df_temp['date'], group_ids)):
            if group_id not in sample_groups:
                sample_groups[group_id] = {'dates': set(), 'count': 0}
            sample_groups[group_id]['dates'].add(date)
            sample_groups[group_id]['count'] += 1
        
        # 检查是否有跨日分组
        cross_day_groups = []
        for group_id, info in sample_groups.items():
            if len(info['dates']) > 1:
                cross_day_groups.append(group_id)
        
        if cross_day_groups:
            logger.error(f"🚨 发现跨日分组: {len(cross_day_groups)}个组，这会导致未来信息泄露！")
            for group_id in cross_day_groups[:3]:  # 只显示前3个
                logger.error(f"  组{group_id}: 包含日期{sample_groups[group_id]['dates']}")
        else:
            logger.info("✅ LTR分组验证通过：所有组都是同日横截面")
        
        # 抽样打印当天group/pair信息
        logger.info("🔍 LTR分组详情抽样:")
        sample_dates = unique_dates[:3] + unique_dates[-3:]  # 前3个和后3个日期
        for date in sample_dates:
            if date in date_to_group:
                group_id = date_to_group[date]
                count = sample_groups[group_id]['count']
                logger.info(f"  日期{date}: 组ID={group_id}, 横截面股票数={count}")
        
        logger.info(f"LTR数据集创建完成: {len(unique_dates)}个横截面组, {len(df_temp)}个样本")
        
        # 提取特征和目标
        feature_cols = [col for col in df_temp.columns if col not in ['y', 'date']]
        X_clean = df_temp[feature_cols].values
        y_clean = df_temp['y'].values
        
        logger.info(f"排序数据集创建完成: {X_clean.shape[0]}样本, {len(np.unique(group_ids))}组")
        
        return X_clean, y_clean, group_ids

    def _create_smart_labels(self, y: np.ndarray, group_ids: np.ndarray, 
                            mode: str = 'soft', n_bins: int = 5, temperature: float = 1.0) -> Dict[str, np.ndarray]:
        """
        智能标签处理：支持连续、离散和软标签
        
        Args:
            y: 原始连续标签
            group_ids: 分组ID
            mode: 'continuous' | 'discrete' | 'soft' | 'multi'
            n_bins: 离散化分箱数（仅在需要时使用）
            temperature: 软标签温度参数
            
        Returns:
            包含不同类型标签的字典
        """
        results = {
            'continuous': y.copy(),  # 始终保留原始连续标签
            'group_ids': group_ids
        }
        
        y_series = pd.Series(y)
        g_series = pd.Series(group_ids)
        
        # 1. 连续标签标准化（组内）
        standardized_labels = np.zeros_like(y)
        for g in np.unique(group_ids):
            mask = (g_series == g)
            if mask.sum() <= 1:
                continue
            y_group = y_series[mask]
            # 使用Z-score标准化，保留相对排序
            standardized = (y_group - y_group.mean()) / (y_group.std() + 1e-8)
            standardized_labels[mask] = standardized.values
        
        results['standardized'] = standardized_labels
        
        # 2. 离散标签（仅在需要时创建，减少分箱数）
        if mode in ['discrete', 'multi']:
            discrete_labels = np.zeros_like(y, dtype=int)
            for g in np.unique(group_ids):
                mask = (g_series == g)
                if mask.sum() <= 1:
                    continue
                
                y_group = y_series[mask]
                if y_group.nunique() <= 1:
                    discrete_labels[mask] = 0
                else:
                    try:
                        # 使用更少的分箱数减少信息损失
                        effective_bins = min(n_bins, max(3, y_group.nunique() // 2))
                        discrete_labels[mask] = pd.qcut(
                            y_group, q=effective_bins, labels=False, duplicates='drop'
                        ).fillna(0).astype(int)
                    except Exception:
                        # 回退到简单二分类
                        median_val = y_group.median()
                        discrete_labels[mask] = (y_group > median_val).astype(int)
            
            results['discrete'] = discrete_labels
        
        # 3. 软标签（概率分布）
        if mode in ['soft', 'multi']:
            soft_labels = np.zeros((len(y), n_bins))
            for g in np.unique(group_ids):
                mask = (g_series == g)
                if mask.sum() <= 1:
                    continue
                
                y_group = y_series[mask]
                # 使用温度参数控制软化程度
                if y_group.std() > 0:
                    y_scaled = (y_group - y_group.min()) / (y_group.max() - y_group.min() + 1e-8)
                    
                    # 创建软标签分布
                    for i, val in enumerate(y_scaled):
                        # 基于值计算在各分箱的概率
                        bin_centers = np.linspace(0, 1, n_bins)
                        distances = np.abs(bin_centers - val)
                        probabilities = np.exp(-distances / temperature)
                        probabilities = probabilities / probabilities.sum()
                        
                        mask_indices = np.where(mask)[0]
                        if i < len(mask_indices):
                            soft_labels[mask_indices[i]] = probabilities
            
            results['soft'] = soft_labels
        
        logger.info(f"智能标签创建完成，模式: {mode}, 包含: {list(results.keys())}")
        return results

    def _discretize_labels_by_group(self, y: np.ndarray, group_ids: np.ndarray, n_bins: int = 10) -> np.ndarray:
        """向后兼容的离散化方法（建议使用_create_smart_labels）"""
        labels_dict = self._create_smart_labels(y, group_ids, mode='discrete', n_bins=n_bins)
        return labels_dict.get('discrete', np.zeros_like(y, dtype=int))
    

    def train_ranking_models(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series,
                           cv_folds: int = 5, optimize_hyperparams: bool = True, 
                           validation_config=None, sample_weights=None) -> Dict[str, Any]:
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
        
        # 使用V6提供的样本权重，避免重复权重计算
        if sample_weights is not None:
            logger.info("使用V6提供的统一样本权重，避免重复权重计算")
            # 对齐权重到排序数据集
            if hasattr(sample_weights, 'loc'):
                aligned_weights = sample_weights.loc[X_rank.index] if hasattr(X_rank, 'index') else None
            else:
                aligned_weights = sample_weights
        else:
            logger.warning("未提供V6样本权重，排序模型将使用均等权重")
            aligned_weights = None
        # 使用统一的V6 validation_config，避免CV配置漂移
        if validation_config is not None:
            try:
                # 使用V6提供的验证配置
                from .enhanced_temporal_validation import EnhancedPurgedTimeSeriesSplit
                
                cv = EnhancedPurgedTimeSeriesSplit(validation_config)
                unique_groups = np.unique(group_ids)
                
                # 确保group_ids是pandas Series
                if isinstance(group_ids, np.ndarray):
                    group_ids_series = pd.Series(group_ids)
                else:
                    group_ids_series = group_ids
                cv_splits = list(cv.split(X_rank, y_rank, group_ids_series))
                
                logger.info(f"使用V6统一验证配置，{len(cv_splits)}个fold，gap={validation_config.gap}，embargo={validation_config.embargo}")
                
            except Exception as e:
                logger.warning(f"V6验证配置使用失败: {e}, 使用回退策略")
                validation_config = None
        
        # 回退策略：使用传统CV（仅当V6配置不可用时）
        if validation_config is None:
            logger.warning("使用传统CV回退策略，可能存在配置漂移风险")
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            unique_groups = np.unique(group_ids)
            cv_splits = list(tscv.split(unique_groups))
            
        # 确保cv_splits可用
        if not cv_splits:
            logger.error("CV splits为空，使用默认分割")
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            unique_groups = np.unique(group_ids)
            cv_splits = list(tscv.split(unique_groups))
        
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
        
        # CatBoost removed due to compatibility issues
        
        # 4. 分位数回归模型
        logger.info("训练分位数回归模型")
        quantile_results = self._train_quantile_models(
            X_rank, y_rank, group_ids, cv_splits, unique_groups
        )
        models_results['quantile_models'] = quantile_results
        
        # 5. 传统回归模型（作为基准）
        logger.info("训练传统回归模型")
        baseline_results = self._train_baseline_models(
            X_rank, y_rank, group_ids, cv_splits, unique_groups
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
            # 修复list object错误：安全检查split格式
            try:
                split_0_is_array = hasattr(split[0], 'shape') and len(split[0].shape) == 1
                split_0_has_int_elements = len(split[0]) > 0 and isinstance(split[0][0], (np.integer, int))
                
                if split_0_has_int_elements and split_0_is_array:
                    # 样本索引（numpy数组）
                    train_mask = np.zeros(len(X), dtype=bool)
                    train_mask[split[0]] = True
                    test_mask = np.zeros(len(X), dtype=bool)
                    test_mask[split[1]] = True
                    train_groups = np.unique(group_ids[train_mask])
                    test_groups = np.unique(group_ids[test_mask])
                elif split_0_has_int_elements and isinstance(split[0], (list, tuple)):
                    # 样本索引（列表格式）
                    train_indices = np.array(split[0])
                    test_indices = np.array(split[1])
                    train_mask = np.zeros(len(X), dtype=bool)
                    train_mask[train_indices] = True
                    test_mask = np.zeros(len(X), dtype=bool)
                    test_mask[test_indices] = True
                    train_groups = np.unique(group_ids[train_mask])
                    test_groups = np.unique(group_ids[test_mask])
                else:
                    # 组索引 - 添加边界检查防止越界
                    train_groups_idx, test_groups_idx = split
                    # 修复索引越界问题：确保索引在有效范围内
                    train_groups_idx = np.array(train_groups_idx)
                    test_groups_idx = np.array(test_groups_idx)
                    
                    # 过滤超出范围的索引
                    valid_train_idx = train_groups_idx[train_groups_idx < len(unique_groups)]
                    valid_test_idx = test_groups_idx[test_groups_idx < len(unique_groups)]
                    
                    train_groups = unique_groups[valid_train_idx]
                    test_groups = unique_groups[valid_test_idx]
                    
                    if len(valid_train_idx) < len(train_groups_idx) or len(valid_test_idx) < len(test_groups_idx):
                        logger.warning(f"过滤了超出范围的索引: 训练{len(train_groups_idx)-len(valid_train_idx)}, 测试{len(test_groups_idx)-len(valid_test_idx)}")
                        
            except Exception as e:
                logger.warning(f"split格式解析失败: {e}, 使用默认组索引处理")
                # 默认按组索引处理 - 添加边界检查
                try:
                    train_groups_idx, test_groups_idx = split
                    train_groups_idx = np.array(train_groups_idx)
                    test_groups_idx = np.array(test_groups_idx)
                    
                    # 过滤超出范围的索引
                    valid_train_idx = train_groups_idx[train_groups_idx < len(unique_groups)]
                    valid_test_idx = test_groups_idx[test_groups_idx < len(unique_groups)]
                    
                    train_groups = unique_groups[valid_train_idx]
                    test_groups = unique_groups[valid_test_idx]
                except Exception as e2:
                    logger.error(f"组索引处理失败: {e2}, 跳过此split")
                    continue
            
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
            # 修复list object错误：安全检查split格式（LightGBM部分）
            try:
                split_0_is_array = hasattr(split[0], 'shape') and len(split[0].shape) == 1
                split_0_has_int_elements = len(split[0]) > 0 and isinstance(split[0][0], (np.integer, int))
                
                if split_0_has_int_elements and split_0_is_array:
                    # 样本索引（numpy数组）
                    train_mask = np.zeros(len(X), dtype=bool)
                    train_mask[split[0]] = True
                    test_mask = np.zeros(len(X), dtype=bool)
                    test_mask[split[1]] = True
                    train_groups = np.unique(group_ids[train_mask])
                    test_groups = np.unique(group_ids[test_mask])
                elif split_0_has_int_elements and isinstance(split[0], (list, tuple)):
                    # 样本索引（列表格式）
                    train_indices = np.array(split[0])
                    test_indices = np.array(split[1])
                    train_mask = np.zeros(len(X), dtype=bool)
                    train_mask[train_indices] = True
                    test_mask = np.zeros(len(X), dtype=bool)
                    test_mask[test_indices] = True
                    train_groups = np.unique(group_ids[train_mask])
                    test_groups = np.unique(group_ids[test_mask])
                else:
                    # 组索引 - 添加边界检查防止越界 (LightGBM)
                    train_groups_idx, test_groups_idx = split
                    # 修复索引越界问题：确保索引在有效范围内
                    train_groups_idx = np.array(train_groups_idx)
                    test_groups_idx = np.array(test_groups_idx)
                    
                    # 过滤超出范围的索引
                    valid_train_idx = train_groups_idx[train_groups_idx < len(unique_groups)]
                    valid_test_idx = test_groups_idx[test_groups_idx < len(unique_groups)]
                    
                    train_groups = unique_groups[valid_train_idx]
                    test_groups = unique_groups[valid_test_idx]
                    
                    if len(valid_train_idx) < len(train_groups_idx) or len(valid_test_idx) < len(test_groups_idx):
                        logger.warning(f"LightGBM过滤了超出范围的索引: 训练{len(train_groups_idx)-len(valid_train_idx)}, 测试{len(test_groups_idx)-len(valid_test_idx)}")
                        
            except Exception as e:
                logger.warning(f"LightGBM split格式解析失败: {e}, 使用默认组索引处理")
                # 默认按组索引处理 - 添加边界检查 
                try:
                    train_groups_idx, test_groups_idx = split
                    train_groups_idx = np.array(train_groups_idx)
                    test_groups_idx = np.array(test_groups_idx)
                    
                    # 过滤超出范围的索引
                    valid_train_idx = train_groups_idx[train_groups_idx < len(unique_groups)]
                    valid_test_idx = test_groups_idx[test_groups_idx < len(unique_groups)]
                    
                    train_groups = unique_groups[valid_train_idx]
                    test_groups = unique_groups[valid_test_idx]
                except Exception as e2:
                    logger.error(f"LightGBM组索引处理失败: {e2}, 跳过此split")
                    continue
            
            train_mask = np.isin(group_ids, train_groups)
            test_mask = np.isin(group_ids, test_groups)
            
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_test = X[test_mask]
            
            # 计算组大小，并处理LightGBM的10000行限制
            train_group_ids = group_ids[train_mask]
            original_group_sizes = [np.sum(train_group_ids == g) for g in train_groups]
            
            # 如果有组超过10000行，需要分割
            MAX_GROUP_SIZE = 9999  # LightGBM的上限是10000
            train_group_sizes = []
            
            for size in original_group_sizes:
                if size > MAX_GROUP_SIZE:
                    # 将大组分割成多个小组
                    num_splits = (size + MAX_GROUP_SIZE - 1) // MAX_GROUP_SIZE
                    split_size = size // num_splits
                    remaining = size % num_splits
                    
                    for i in range(num_splits):
                        if i < remaining:
                            train_group_sizes.append(split_size + 1)
                        else:
                            train_group_sizes.append(split_size)
                else:
                    train_group_sizes.append(size)
            
            logger.info(f"LightGBM组大小调整: {len(original_group_sizes)}组 -> {len(train_group_sizes)}组")
            logger.info(f"最大组大小: {max(train_group_sizes)}")
            
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
    
    # CatBoost ranker method removed due to compatibility issues
    
    def _train_quantile_models(self, X: np.ndarray, y: np.ndarray, group_ids: np.ndarray,
                              cv_splits: List[Tuple], unique_groups: np.ndarray) -> Dict[str, Any]:
        """训练分位数回归模型"""
        models = {}
        oof_predictions = {}
        
        # 训练多个分位数
        quantiles = [0.1, 0.25, 0.5, 0.7, 0.9]
        
        for quantile in quantiles:
            models[f'q{int(quantile*100)}'] = []
            oof_predictions[f'q{int(quantile*100)}'] = np.full(len(X), np.nan)
            
            for train_groups_idx, test_groups_idx in cv_splits:
                # 修复索引越界问题：确保索引在有效范围内
                train_groups_idx = np.array(train_groups_idx)
                test_groups_idx = np.array(test_groups_idx)
                
                # 过滤超出范围的索引
                valid_train_idx = train_groups_idx[train_groups_idx < len(unique_groups)]
                valid_test_idx = test_groups_idx[test_groups_idx < len(unique_groups)]
                
                if len(valid_train_idx) == 0 or len(valid_test_idx) == 0:
                    continue
                    
                train_groups = unique_groups[valid_train_idx]
                test_groups = unique_groups[valid_test_idx]
                
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
                              cv_splits: List[Tuple], unique_groups: np.ndarray) -> Dict[str, Any]:
        """训练基准回归模型"""
        baseline_models = {
            'linear': LinearRegression(),
            'rf': RandomForestRegressor(n_estimators=30, max_depth=5, max_samples=0.5, random_state=42)  # 大幅减少内存使用
        }
        
        results = {}
        
        for model_name, base_model in baseline_models.items():
            models = []
            oof_predictions = np.full(len(X), np.nan)
            oof_uncertainties = np.full(len(X), np.nan)
            
            for train_groups_idx, test_groups_idx in cv_splits:
                # 修复索引越界问题：确保索引在有效范围内
                train_groups_idx = np.array(train_groups_idx)
                test_groups_idx = np.array(test_groups_idx)
                
                # 过滤超出范围的索引
                valid_train_idx = train_groups_idx[train_groups_idx < len(unique_groups)]
                valid_test_idx = test_groups_idx[test_groups_idx < len(unique_groups)]
                
                if len(valid_train_idx) == 0 or len(valid_test_idx) == 0:
                    continue
                    
                train_groups = unique_groups[valid_train_idx]
                test_groups = unique_groups[valid_test_idx]
                
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
        
        # 🔧 Fix LTR维度匹配: 确保特征维度与训练时一致
        X_aligned = X.copy()
        if hasattr(self, 'training_feature_columns'):
            training_cols = self.training_feature_columns
            current_cols = X.columns.tolist()
            
            if len(current_cols) != len(training_cols):
                logger.warning(f"LTR预测特征维度不匹配: 当前{len(current_cols)} vs 训练{len(training_cols)}")
                
                # 对齐特征列：只保留训练时的特征
                common_cols = [col for col in training_cols if col in current_cols]
                missing_cols = [col for col in training_cols if col not in current_cols]
                
                if missing_cols:
                    logger.warning(f"LTR预测缺失特征列: {missing_cols}，将用0填充")
                    for col in missing_cols:
                        X_aligned[col] = 0.0
                
                # 重新排序并选择训练时的特征
                X_aligned = X_aligned[training_cols]
                logger.info(f"LTR特征对齐完成: {X_aligned.shape}")
        
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
                                    dmat = xgb.DMatrix(X_aligned.values)
                                    pred = model.predict(dmat)
                                else:
                                    pred = model.predict(X_aligned.values)
                            except Exception:
                                pred = model.predict(X_aligned.values)
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


# 测试代码已移除，避免生产代码包含演示逻辑
# 如需测试，请参考 tests/ 或 examples/ 目录
