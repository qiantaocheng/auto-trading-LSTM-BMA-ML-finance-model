#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML集成学习增强模块 - 高级模型集成系统
包含Voting、Stacking、Boosting和动态BMA权重学习

Features:
- Voting/Stacking集成
- AdaBoost/GradientBoosting
- BMA权重ML动态学习
- 多层Stacking
- 模型多样性分析
- 动态权重调整
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    VotingRegressor, StackingRegressor,
    AdaBoostRegressor, GradientBoostingRegressor,
    RandomForestRegressor, ExtraTreesRegressor,
    BaggingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# 条件导入
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

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """集成学习配置"""
    # 集成方法
    ensemble_methods: List[str] = field(default_factory=lambda: [
        'voting', 'stacking', 'boosting', 'dynamic_bma'
    ])
    
    # Voting配置
    voting_type: str = 'soft'  # 'hard' or 'soft'
    voting_weights: Optional[List[float]] = None  # 自动学习如果为None
    
    # Stacking配置
    stacking_meta_model: str = 'Ridge'  # 元学习器
    stacking_cv_folds: int = 5
    stacking_use_probas: bool = False
    stacking_passthrough: bool = True  # 是否将原始特征传递给元学习器
    multi_level_stacking: bool = True  # 多层stacking
    
    # Boosting配置
    boosting_n_estimators: int = 100
    boosting_learning_rate: float = 0.1
    boosting_loss: str = 'square'  # 'linear', 'square', 'exponential'
    
    # BMA配置
    bma_learning_rate: float = 0.01
    bma_momentum: float = 0.9
    bma_weight_decay: float = 0.001
    bma_update_frequency: int = 10  # 每N个预测更新一次权重
    
    # 模型池配置 - 优化为互补性三类模型
    base_models: List[str] = field(default_factory=lambda: [
        'ElasticNet', 'LightGBM', 'ExtraTrees'  # 线性收缩 + 梯度提升 + 袋装树
    ])
    
    # 多样性配置
    diversity_threshold: float = 0.7  # 相关性阈值
    min_model_performance: float = 0.3  # 最低模型性能要求
    
    # 交叉验证
    cv_strategy: str = 'time_series'  # 'time_series' or 'kfold'
    n_splits: int = 5
    
    # 性能配置
    n_jobs: int = -1
    random_state: int = 42
    verbose: int = 0


class DynamicBMAWeightLearner(BaseEstimator, RegressorMixin):
    """动态BMA权重学习器 - 使用ML学习最优权重"""
    
    def __init__(self, base_models: List, learning_rate: float = 0.01, 
                 momentum: float = 0.9, weight_decay: float = 0.001):
        self.base_models = base_models
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.weights_ = None
        self.weight_history_ = []
        self.velocity_ = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def fit(self, X, y):
        """训练BMA权重"""
        n_models = len(self.base_models)
        
        # 初始化权重（均等权重）
        self.weights_ = np.ones(n_models) / n_models
        self.velocity_ = np.zeros(n_models)
        
        # 获取每个模型的预测
        predictions = self._get_base_predictions(X, y)
        
        # 使用梯度下降优化权重
        n_iterations = 100
        for iteration in range(n_iterations):
            # 计算加权预测
            weighted_pred = np.sum(predictions * self.weights_[:, np.newaxis], axis=0)
            
            # 计算损失和梯度
            loss = np.mean((weighted_pred - y) ** 2)
            gradients = self._compute_gradients(predictions, y, weighted_pred)
            
            # 带动量的梯度下降
            self.velocity_ = self.momentum * self.velocity_ - self.learning_rate * gradients
            self.weights_ += self.velocity_
            
            # L2正则化（权重衰减）
            self.weights_ -= self.weight_decay * self.weights_
            
            # 确保权重非负且和为1
            self.weights_ = np.maximum(self.weights_, 0)
            self.weights_ /= (np.sum(self.weights_) + 1e-8)
            
            # 记录权重历史
            self.weight_history_.append(self.weights_.copy())
            
            if iteration % 20 == 0:
                self.logger.debug(f"迭代 {iteration}, 损失: {loss:.6f}")
        
        # 训练基模型
        for model in self.base_models:
            model.fit(X, y)
        
        self.logger.info(f"BMA权重学习完成: {self.weights_}")
        
        return self
    
    def predict(self, X):
        """使用学习的权重进行预测"""
        predictions = np.array([model.predict(X) for model in self.base_models])
        return np.sum(predictions * self.weights_[:, np.newaxis], axis=0)
    
    def _get_base_predictions(self, X, y):
        """获取基模型的交叉验证预测"""
        # 🚨 禁止内部创建CV，必须使用外部传入的统一CV
        if not hasattr(self, 'cv_factory') or self.cv_factory is None:
            raise NotImplementedError(
                "违反SSOT原则：集成系统不允许内部创建CV！\n"
                "修复指南：必须通过外部传入cv_factory，使用UnifiedCVFactory。\n"
                "禁止TimeSeriesSplit等内部CV创建。"
            )
        
        cv = self.cv_factory
        predictions = []
        
        for model in self.base_models:
            model_clone = clone(model)
            pred = cross_val_predict(model_clone, X, y, cv=cv)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _compute_gradients(self, predictions, y, weighted_pred):
        """计算权重的梯度"""
        error = weighted_pred - y
        gradients = np.array([
            2 * np.mean(error * pred) for pred in predictions
        ])
        return gradients
    
    def update_weights(self, X, y):
        """在线更新权重"""
        predictions = np.array([model.predict(X) for model in self.base_models])
        weighted_pred = np.sum(predictions * self.weights_[:, np.newaxis], axis=0)
        
        # 计算梯度并更新
        gradients = self._compute_gradients(predictions, y, weighted_pred)
        self.velocity_ = self.momentum * self.velocity_ - self.learning_rate * gradients
        self.weights_ += self.velocity_
        
        # 正则化和归一化
        self.weights_ -= self.weight_decay * self.weights_
        self.weights_ = np.maximum(self.weights_, 0)
        self.weights_ /= (np.sum(self.weights_) + 1e-8)
        
        self.weight_history_.append(self.weights_.copy())


class MLEnsembleEnhanced:
    """增强的集成学习系统"""
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.ensemble_models = {}
        self.model_performances = {}
        self.diversity_matrix = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def build_ensemble(self, X: np.ndarray, y: np.ndarray,
                       cv_factory: callable, 
                       feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        构建完整的集成学习系统
        
        Args:
            X: 特征数据
            y: 目标变量
            cv_factory: 统一CV工厂（必须）
            feature_names: 特征名称
            
        Returns:
            集成模型字典和性能指标
        """
        # 保存CV工厂
        self.cv_factory = cv_factory
        self.logger.info("开始构建集成学习系统")
        
        # 1. 创建基础模型池
        base_models = self._create_base_models()
        
        # 2. 评估模型多样性
        self.diversity_matrix = self._evaluate_model_diversity(base_models, X, y)
        
        # 3. 选择多样化的模型子集
        selected_models = self._select_diverse_models(base_models, self.diversity_matrix)
        
        # 4. 构建各种集成
        results = {}
        
        if 'voting' in self.config.ensemble_methods:
            results['voting'] = self._build_voting_ensemble(selected_models, X, y)
        
        if 'stacking' in self.config.ensemble_methods:
            if self.config.multi_level_stacking:
                results['stacking'] = self._build_multi_level_stacking(selected_models, X, y)
            else:
                results['stacking'] = self._build_stacking_ensemble(selected_models, X, y)
        
        if 'boosting' in self.config.ensemble_methods:
            results['boosting'] = self._build_boosting_ensemble(X, y)
        
        if 'dynamic_bma' in self.config.ensemble_methods:
            results['dynamic_bma'] = self._build_dynamic_bma(selected_models, X, y)
        
        # 5. 评估各集成方法性能
        performances = self._evaluate_ensembles(results, X, y)
        
        # 6. 选择最优集成
        best_method = max(performances.keys(), key=lambda k: performances[k]['r2_score'])
        
        self.logger.info(f"最优集成方法: {best_method}, R2: {performances[best_method]['r2_score']:.6f}")
        
        return {
            'models': results,
            'performances': performances,
            'best_method': best_method,
            'best_model': results[best_method],
            'diversity_matrix': self.diversity_matrix
        }
    
    def _create_base_models(self) -> List:
        """创建优化的互补性三类模型池"""
        models = []
        
        for model_name in self.config.base_models:
            if model_name == 'ElasticNet':
                # 线性收缩模型 - 金融弱信号稳健，L1_ratio网格涵盖LASSO到Ridge
                models.append(ElasticNet(
                    alpha=0.1, 
                    l1_ratio=0.5,  # 平衡L1/L2
                    random_state=self.config.random_state,
                    max_iter=2000
                ))
            elif model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
                # LGBM: 浅+强正则+子采样 - 控制过拟合优先
                models.append(lgb.LGBMRegressor(
                    n_estimators=80,         # 减少树数量
                    max_depth=4,             # 浅树
                    learning_rate=0.05,      # 保守学习率
                    num_leaves=20,           # 少叶子节点
                    feature_fraction=0.7,    # 强特征子采样
                    bagging_fraction=0.7,    # 强样本子采样
                    lambda_l1=0.3,           # 强L1正则
                    lambda_l2=0.3,           # 强L2正则
                    min_child_samples=30,    # 大最小叶节点样本
                    random_state=self.config.random_state,
                    verbosity=-1
                ))
            elif model_name == 'ExtraTrees':
                # ET: 深+高随机 - 多样性优先
                models.append(ExtraTreesRegressor(
                    n_estimators=120,        # 更多树
                    max_depth=12,            # 深树
                    min_samples_split=2,     # 激进分裂
                    min_samples_leaf=2,      # 小叶节点
                    max_features=0.5,        # 高特征随机性
                    bootstrap=False,         # ET默认不bootstrap
                    random_state=self.config.random_state
                ))
            elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                # 可选备胎 - 差异化超参增加多样性
                models.append(xgb.XGBRegressor(
                    n_estimators=80,  # 与LGBM不同
                    max_depth=4,      # 稍浅
                    learning_rate=0.08,  # 不同学习率
                    subsample=0.85,   # 不同采样比例
                    colsample_bytree=0.75,
                    random_state=self.config.random_state
                ))
        
        return models
    
    def _evaluate_model_diversity(self, models: List, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """评估模型多样性（通过预测相关性）"""
        cv = self._get_cv_strategy(X)
        predictions = []
        
        for i, model in enumerate(models):
            try:
                model_clone = clone(model)
                pred = cross_val_predict(model_clone, X, y, cv=cv)
                predictions.append(pred)
            except Exception as e:
                self.logger.warning(f"模型 {i} 预测失败: {e}")
                predictions.append(np.zeros_like(y))
        
        # 计算预测之间的相关矩阵
        n_models = len(predictions)
        diversity_matrix = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    corr, _ = spearmanr(predictions[i], predictions[j])
                    diversity_matrix[i, j] = 1 - abs(corr)  # 多样性 = 1 - 相关性
                else:
                    diversity_matrix[i, j] = 0
        
        return diversity_matrix
    
    def _select_diverse_models(self, models: List, diversity_matrix: np.ndarray) -> List:
        """选择多样化的模型子集"""
        n_models = len(models)
        selected_indices = []
        
        # 贪婪选择：每次选择能最大化总体多样性的模型
        for _ in range(min(n_models, 5)):  # 最多选择5个模型
            best_diversity = -1
            best_idx = -1
            
            for i in range(n_models):
                if i not in selected_indices:
                    if not selected_indices:
                        # 第一个模型随机选择
                        diversity_score = np.random.random()
                    else:
                        # 计算与已选模型的平均多样性
                        diversity_score = np.mean([diversity_matrix[i, j] for j in selected_indices])
                    
                    if diversity_score > best_diversity:
                        best_diversity = diversity_score
                        best_idx = i
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
        
        selected_models = [models[i] for i in selected_indices]
        
        self.logger.info(f"选择了 {len(selected_models)} 个多样化模型")
        
        return selected_models
    
    def _build_voting_ensemble(self, models: List, X: np.ndarray, y: np.ndarray):
        """构建投票集成"""
        # 如果没有指定权重，学习最优权重
        if self.config.voting_weights is None:
            weights = self._learn_voting_weights(models, X, y)
        else:
            weights = self.config.voting_weights
        
        # 创建投票集成
        estimators = [(f'model_{i}', model) for i, model in enumerate(models)]
        
        voting = VotingRegressor(
            estimators=estimators,
            weights=weights,
            n_jobs=self.config.n_jobs
        )
        
        voting.fit(X, y)
        
        self.logger.info(f"投票集成构建完成，权重: {weights}")
        
        return voting
    
    def _build_stacking_ensemble(self, models: List, X: np.ndarray, y: np.ndarray):
        """构建Stacking集成"""
        # 获取元学习器
        meta_model = self._get_meta_learner()
        
        # 创建Stacking集成
        estimators = [(f'model_{i}', model) for i, model in enumerate(models)]
        
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_model,
            cv=self._get_cv_strategy(X),
            n_jobs=self.config.n_jobs,
            passthrough=self.config.stacking_passthrough
        )
        
        stacking.fit(X, y)
        
        self.logger.info("Stacking集成构建完成")
        
        return stacking
    
    def _build_multi_level_stacking(self, models: List, X: np.ndarray, y: np.ndarray):
        """构建多层Stacking"""
        # 第一层：原始模型
        level1_models = models[:3] if len(models) > 3 else models
        
        # 第二层：使用第一层的预测作为特征
        level1_estimators = [(f'l1_model_{i}', model) for i, model in enumerate(level1_models)]
        
        level2_model = StackingRegressor(
            estimators=level1_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=self._get_cv_strategy(X),
            n_jobs=self.config.n_jobs
        )
        
        # 第三层：最终元学习器
        remaining_models = models[3:] if len(models) > 3 else []
        if remaining_models:
            level2_estimators = [('level2', level2_model)] + \
                              [(f'extra_{i}', m) for i, m in enumerate(remaining_models)]
        else:
            level2_estimators = [('level2', level2_model)]
        
        final_stacking = StackingRegressor(
            estimators=level2_estimators,
            final_estimator=self._get_meta_learner(),
            cv=self._get_cv_strategy(X),
            n_jobs=self.config.n_jobs,
            passthrough=self.config.stacking_passthrough
        )
        
        final_stacking.fit(X, y)
        
        self.logger.info("多层Stacking集成构建完成")
        
        return final_stacking
    
    def _build_boosting_ensemble(self, X: np.ndarray, y: np.ndarray):
        """构建Boosting集成"""
        # 使用AdaBoost和GradientBoosting的组合
        ada_boost = AdaBoostRegressor(
            n_estimators=self.config.boosting_n_estimators,
            learning_rate=self.config.boosting_learning_rate,
            loss=self.config.boosting_loss,
            random_state=self.config.random_state
        )
        
        gradient_boost = GradientBoostingRegressor(
            n_estimators=self.config.boosting_n_estimators,
            learning_rate=self.config.boosting_learning_rate,
            max_depth=5,
            random_state=self.config.random_state
        )
        
        # 创建Boosting集成的集成
        boosting_ensemble = VotingRegressor(
            estimators=[
                ('ada', ada_boost),
                ('gradient', gradient_boost)
            ],
            n_jobs=self.config.n_jobs
        )
        
        boosting_ensemble.fit(X, y)
        
        self.logger.info("Boosting集成构建完成")
        
        return boosting_ensemble
    
    def _build_dynamic_bma(self, models: List, X: np.ndarray, y: np.ndarray):
        """构建动态BMA权重学习"""
        bma_learner = DynamicBMAWeightLearner(
            base_models=models,
            learning_rate=self.config.bma_learning_rate,
            momentum=self.config.bma_momentum,
            weight_decay=self.config.bma_weight_decay
        )
        
        bma_learner.fit(X, y)
        
        self.logger.info(f"动态BMA构建完成，学习的权重: {bma_learner.weights_}")
        
        return bma_learner
    
    def _learn_voting_weights(self, models: List, X: np.ndarray, y: np.ndarray) -> List[float]:
        """学习投票权重"""
        cv = self._get_cv_strategy(X)
        model_scores = []
        
        for model in models:
            try:
                model_clone = clone(model)
                scores = []
                
                for train_idx, val_idx in cv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    model_clone.fit(X_train, y_train)
                    pred = model_clone.predict(X_val)
                    score = r2_score(y_val, pred)
                    scores.append(max(0, score))  # 确保非负
                
                model_scores.append(np.mean(scores))
                
            except Exception as e:
                self.logger.warning(f"模型评分失败: {e}")
                model_scores.append(0.1)  # 给予最小权重
        
        # 转换分数为权重
        model_scores = np.array(model_scores)
        if model_scores.sum() > 0:
            weights = model_scores / model_scores.sum()
        else:
            weights = np.ones(len(models)) / len(models)
        
        return weights.tolist()
    
    def _get_meta_learner(self):
        """获取元学习器"""
        meta_learners = {
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'Linear': LinearRegression()
        }
        
        return meta_learners.get(self.config.stacking_meta_model, Ridge(alpha=1.0))
    
    def _get_cv_strategy(self, X: np.ndarray):
        """获取交叉验证策略 - 强制使用外部CV工厂"""
        # 🚨 禁止内部创建CV，必须使用外部传入的统一CV
        if not hasattr(self, 'cv_factory') or self.cv_factory is None:
            raise NotImplementedError(
                "违反SSOT原则：集成系统不允许内部创建CV！\n"
                "修复指南：必须通过外部传入cv_factory，使用UnifiedCVFactory。\n"
                "禁止TimeSeriesSplit等内部CV创建。"
            )
        return self.cv_factory
    
    def _evaluate_ensembles(self, models: Dict, X: np.ndarray, y: np.ndarray) -> Dict:
        """评估所有集成方法"""
        performances = {}
        cv = self._get_cv_strategy(X)
        
        for name, model in models.items():
            try:
                # 交叉验证评估
                predictions = cross_val_predict(model, X, y, cv=cv)
                
                mse = mean_squared_error(y, predictions)
                r2 = r2_score(y, predictions)
                
                performances[name] = {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2_score': r2
                }
                
                self.logger.info(f"{name} - MSE: {mse:.6f}, R2: {r2:.6f}")
                
            except Exception as e:
                self.logger.warning(f"评估 {name} 失败: {e}")
                performances[name] = {
                    'mse': float('inf'),
                    'rmse': float('inf'),
                    'r2_score': -float('inf')
                }
        
        return performances
    
    def predict_with_uncertainty(self, models: Dict, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用集成预测并估计不确定性"""
        predictions = []
        
        for name, model in models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                self.logger.warning(f"模型 {name} 预测失败: {e}")
        
        if predictions:
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
        else:
            mean_pred = np.zeros(len(X))
            std_pred = np.ones(len(X))
        
        return mean_pred, std_pred


# 便捷函数
def build_advanced_ensemble(X: np.ndarray, y: np.ndarray, 
                          config: Optional[EnsembleConfig] = None) -> Dict[str, Any]:
    """
    构建高级集成学习系统便捷函数
    
    Args:
        X: 特征数据
        y: 目标变量
        config: 配置（可选）
        
    Returns:
        集成模型和性能指标
    """
    ensemble_builder = MLEnsembleEnhanced(config)
    return ensemble_builder.build_ensemble(X, y)