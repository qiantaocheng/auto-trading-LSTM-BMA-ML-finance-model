#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purged Group Time Series Cross-Validation with Embargo
金融时间序列专用的交叉验证，防止数据泄漏
"""

import pandas as pd
import numpy as np
import warnings
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Iterator, Union
from dataclasses import dataclass
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import ndcg_score
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """验证配置"""
    n_splits: int = 5
    test_size: int = 63  # 测试集大小（交易日）
    gap: int = 5         # 训练测试间隔（防泄漏）
    embargo: int = 2     # 禁区长度（交易日）
    min_train_size: int = 252  # 最小训练集大小
    group_freq: str = 'W'      # 分组频率：'D'=日, 'W'=周, 'M'=月
    
@dataclass
class CVResults:
    """交叉验证结果"""
    oof_predictions: pd.Series
    oof_ic: float
    oof_ndcg: float
    fold_metrics: List[Dict[str, float]]
    feature_importance: Dict[str, float]
    uncertainty_estimates: pd.Series

class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Purged Group Time Series Split with Embargo
    
    特点：
    1. 按时间分组（交易日/周/月）而非简单时间分割
    2. 训练/验证集之间有gap防止前瞻偏差
    3. 验证集前后有embargo防止数据泄漏
    4. 真实组大小传递给模型
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.config.n_splits
    
    def split(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Union[pd.Series, np.ndarray] = None, 
              groups: Union[pd.Series, np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        生成训练/验证集索引
        
        Args:
            X: 特征矩阵
            y: 目标变量
            groups: 时间分组标识（通常是日期）
        """
        
        if groups is None:
            raise ValueError("groups必须提供（时间分组信息）")
        
        # 转换为DataFrame以便处理
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        if isinstance(groups, np.ndarray):
            groups = pd.Series(groups)
            
        # 按时间分组
        unique_groups = sorted(groups.unique())
        n_groups = len(unique_groups)
        
        if n_groups < self.config.min_train_size // 5:  # 最少需要的组数
            raise ValueError(f"时间组数太少: {n_groups}")
        
        # 计算每次分割的参数
        test_groups = self.config.test_size // 5  # 假设每组约5个交易日
        gap_groups = max(1, self.config.gap // 5)
        embargo_groups = max(1, self.config.embargo // 5)
        
        # 生成分割点
        total_test_span = test_groups + 2 * embargo_groups + gap_groups
        
        for fold in range(self.config.n_splits):
            # 计算验证集结束位置
            test_end_idx = n_groups - fold * (test_groups // 2)  # 重叠验证
            test_start_idx = test_end_idx - test_groups
            
            if test_start_idx < total_test_span:
                continue  # 跳过训练数据不足的分割
            
            # 添加embargo
            embargo_start = test_start_idx - embargo_groups
            embargo_end = test_end_idx + embargo_groups
            
            # 训练集：embargo之前的所有数据
            train_end_idx = embargo_start - gap_groups
            
            if train_end_idx < self.config.min_train_size // 5:
                continue  # 训练集太小
            
            # 获取实际的组
            train_groups = unique_groups[:train_end_idx]
            test_groups_actual = unique_groups[test_start_idx:test_end_idx]
            
            # 转换为索引
            train_mask = groups.isin(train_groups)
            test_mask = groups.isin(test_groups_actual)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            if len(train_indices) == 0 or len(test_indices) == 0:
                continue
                
            logger.debug(f"Fold {fold}: Train groups {len(train_groups)}, "
                        f"Test groups {len(test_groups_actual)}, "
                        f"Train samples {len(train_indices)}, "
                        f"Test samples {len(test_indices)}")
            
            yield train_indices, test_indices

class PurgedTimeSeriesValidator:
    """Purged时间序列验证器"""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.cv_splitter = PurgedGroupTimeSeriesSplit(self.config)
        
    def validate_model(self, 
                      X: pd.DataFrame,
                      y: pd.Series,
                      groups: pd.Series,
                      model_type: str = 'xgboost',
                      model_params: Dict[str, Any] = None) -> CVResults:
        """
        执行Purged时间序列交叉验证
        
        Args:
            X: 特征数据
            y: 目标变量（收益率）
            groups: 时间分组（日期）
            model_type: 模型类型 ('xgboost', 'lightgbm')
            model_params: 模型参数
            
        Returns:
            CVResults: 验证结果
        """
        
        logger.info(f"开始Purged时间序列交叉验证，{self.config.n_splits}折")
        
        # 初始化结果存储
        oof_predictions = pd.Series(index=X.index, dtype=float)
        fold_metrics = []
        feature_importance_sum = {}
        uncertainty_estimates = pd.Series(index=X.index, dtype=float)
        
        valid_folds = 0
        
        for fold, (train_idx, test_idx) in enumerate(self.cv_splitter.split(X, y, groups)):
            try:
                logger.info(f"处理第{fold+1}折，训练样本: {len(train_idx)}, 测试样本: {len(test_idx)}")
                
                # 准备训练/测试数据
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                groups_train = groups.iloc[train_idx]
                groups_test = groups.iloc[test_idx]
                
                # 计算真实组大小用于模型
                train_group_info = self._calculate_group_info(groups_train)
                test_group_info = self._calculate_group_info(groups_test)
                
                # 训练模型
                model, feature_importance = self._train_model(
                    X_train, y_train, groups_train, 
                    model_type, model_params, train_group_info
                )
                
                # 预测和不确定性估计
                predictions, uncertainties = self._predict_with_uncertainty(
                    model, X_test, model_type
                )
                
                # 存储OOF预测
                oof_predictions.iloc[test_idx] = predictions
                uncertainty_estimates.iloc[test_idx] = uncertainties
                
                # 计算本折指标
                fold_ic = self._calculate_ic(y_test, predictions)
                fold_ndcg = self._calculate_ndcg(y_test, predictions)
                
                fold_metrics.append({
                    'fold': fold,
                    'ic': fold_ic,
                    'ndcg': fold_ndcg,
                    'train_samples': len(train_idx),
                    'test_samples': len(test_idx),
                    'train_groups': len(train_group_info),
                    'test_groups': len(test_group_info)
                })
                
                # 累积特征重要性
                for feature, importance in feature_importance.items():
                    feature_importance_sum[feature] = feature_importance_sum.get(feature, 0) + importance
                
                valid_folds += 1
                logger.info(f"第{fold+1}折完成 - IC: {fold_ic:.4f}, NDCG: {fold_ndcg:.4f}")
                
            except Exception as e:
                logger.warning(f"第{fold+1}折处理失败: {e}")
                continue
        
        if valid_folds == 0:
            raise ValueError("所有交叉验证折都失败了")
        
        # 计算整体指标
        valid_mask = ~oof_predictions.isna()
        oof_ic = self._calculate_ic(y[valid_mask], oof_predictions[valid_mask])
        oof_ndcg = self._calculate_ndcg(y[valid_mask], oof_predictions[valid_mask])
        
        # 平均特征重要性
        avg_feature_importance = {
            feature: importance / valid_folds 
            for feature, importance in feature_importance_sum.items()
        }
        
        logger.info(f"交叉验证完成 - 整体IC: {oof_ic:.4f}, NDCG: {oof_ndcg:.4f}")
        
        return CVResults(
            oof_predictions=oof_predictions,
            oof_ic=oof_ic,
            oof_ndcg=oof_ndcg,
            fold_metrics=fold_metrics,
            feature_importance=avg_feature_importance,
            uncertainty_estimates=uncertainty_estimates
        )
    
    def _calculate_group_info(self, groups: pd.Series) -> Dict[str, Any]:
        """计算组信息统计"""
        unique_groups = groups.unique()
        group_sizes = groups.value_counts()
        
        return {
            'n_groups': len(unique_groups),
            'total_samples': len(groups),
            'avg_group_size': group_sizes.mean(),
            'min_group_size': group_sizes.min(),
            'max_group_size': group_sizes.max(),
            'group_names': unique_groups
        }
    
    def _train_model(self, 
                    X_train: pd.DataFrame,
                    y_train: pd.Series, 
                    groups_train: pd.Series,
                    model_type: str,
                    model_params: Dict[str, Any],
                    group_info: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """训练模型并返回特征重要性"""
        
        if model_params is None:
            model_params = self._get_default_params(model_type)
        
        # 确保使用真实的组信息
        if model_type == 'xgboost':
            # XGBoost with group information
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            # 添加组权重（可选）
            if 'sample_weight' not in model_params:
                # 根据组大小调整样本权重
                group_weights = groups_train.map(groups_train.value_counts())
                sample_weights = 1.0 / np.sqrt(group_weights)  # 平衡组大小影响
                dtrain.set_weight(sample_weights)
            
            model = xgb.train(
                model_params,
                dtrain,
                num_boost_round=model_params.get('num_boost_round', 100),
                verbose_eval=False
            )
            
            # 获取特征重要性
            importance_dict = model.get_score(importance_type='weight')
            feature_importance = {
                feature: importance_dict.get(f'f{i}', 0.0) 
                for i, feature in enumerate(X_train.columns)
            }
            
        elif model_type == 'lightgbm':
            # LightGBM with group information
            train_set = lgb.Dataset(X_train, label=y_train)
            
            # 设置组信息
            if group_info['n_groups'] > 1:
                group_boundaries = []
                current_pos = 0
                for group_name in group_info['group_names']:
                    group_size = (groups_train == group_name).sum()
                    current_pos += group_size
                    group_boundaries.append(current_pos)
                
                # LightGBM的group参数
                model_params['group'] = group_boundaries[:-1]  # 除了最后一个边界
            
            model = lgb.train(
                model_params,
                train_set,
                num_boost_round=model_params.get('num_boost_round', 100),
                verbose_eval=False
            )
            
            # 获取特征重要性
            importance_values = model.feature_importance(importance_type='split')
            feature_importance = dict(zip(X_train.columns, importance_values))
            
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return model, feature_importance
    
    def _predict_with_uncertainty(self, 
                                model: Any, 
                                X_test: pd.DataFrame,
                                model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """预测并估计不确定性"""
        
        if model_type == 'xgboost':
            dtest = xgb.DMatrix(X_test)
            predictions = model.predict(dtest)
            
            # XGBoost不确定性估计（使用叶子节点方差）
            leaf_indices = model.predict(dtest, pred_leaf=True)
            # 简化的不确定性估计
            uncertainties = np.std(leaf_indices, axis=1) / 100.0
            
        elif model_type == 'lightgbm':
            predictions = model.predict(X_test)
            
            # LightGBM不确定性估计（基于预测分布）
            # 这里使用简化方法，实际应用中可以用更复杂的方法
            uncertainties = np.abs(predictions) * 0.1  # 预测值的10%作为不确定性
            
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return predictions, uncertainties
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """获取默认模型参数"""
        
        if model_type == 'xgboost':
            return {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'eta': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'seed': 42,
                'num_boost_round': 100
            }
        elif model_type == 'lightgbm':
            return {
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'seed': 42,
                'num_boost_round': 100,
                'verbose': -1
            }
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def _calculate_ic(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """计算信息系数(IC)"""
        try:
            corr = y_true.corr(y_pred)
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def _calculate_ndcg(self, y_true: pd.Series, y_pred: pd.Series, k: int = None) -> float:
        """计算NDCG"""
        try:
            if k is None:
                k = min(20, len(y_true) // 5)  # 默认top 20或20%
            
            # 将连续值转换为相关性分数
            # 使用分位数排名作为相关性分数
            relevance_scores = y_true.rank(pct=True)
            
            # 确保相关性分数在[0,1]范围内
            relevance_scores = np.clip(relevance_scores, 0, 1)
            
            # 计算NDCG
            ndcg = ndcg_score([relevance_scores.values], [y_pred.values], k=k)
            return ndcg if not np.isnan(ndcg) else 0.0
            
        except Exception as e:
            logger.warning(f"NDCG计算失败: {e}")
            return 0.0

def create_time_groups(dates: pd.Series, freq: str = 'W') -> pd.Series:
    """创建时间分组"""
    if freq == 'D':
        return dates.dt.date
    elif freq == 'W':
        return dates.dt.to_period('W')
    elif freq == 'M':
        return dates.dt.to_period('M')
    else:
        raise ValueError(f"不支持的频率: {freq}")

# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 2000
    n_features = 20
    
    # 模拟时间序列数据
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)],
        index=dates
    )
    
    # 模拟目标变量（有时间依赖性）
    y = pd.Series(
        np.random.randn(n_samples) * 0.02 + 
        X['feature_0'] * 0.1 + 
        X['feature_1'].rolling(5).mean() * 0.05,
        index=dates
    ).fillna(0)
    
    # 创建时间组
    groups = create_time_groups(dates, freq='W')
    
    # 配置验证参数
    config = ValidationConfig(
        n_splits=5,
        test_size=63,
        gap=5,
        embargo=2,
        group_freq='W'
    )
    
    # 运行验证
    validator = PurgedTimeSeriesValidator(config)
    results = validator.validate_model(X, y, groups, model_type='xgboost')
    
    print(f"OOF IC: {results.oof_ic:.4f}")
    print(f"OOF NDCG: {results.oof_ndcg:.4f}")
    print(f"每折指标:")
    for fold_metric in results.fold_metrics:
        print(f"  Fold {fold_metric['fold']}: IC={fold_metric['ic']:.4f}, "
              f"NDCG={fold_metric['ndcg']:.4f}")
