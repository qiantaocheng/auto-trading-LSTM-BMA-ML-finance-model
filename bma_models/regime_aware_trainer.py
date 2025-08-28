#!/usr/bin/env python3
"""
状态感知训练器 - Regime-Aware Trainer
为BMA Enhanced系统提供基于市场状态的分段建模能力
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from sklearn.model_selection import BaseCrossValidator
import joblib
import os
from pathlib import Path

from market_regime_detector import MarketRegimeDetector, RegimeConfig

logger = logging.getLogger(__name__)

@dataclass
class RegimeTrainingConfig:
    """状态感知训练配置"""
    # 基础配置
    enable_regime_aware: bool = True          # 启用状态感知训练
    regime_config: RegimeConfig = None        # 状态检测配置
    
    # 训练策略
    regime_training_strategy: str = 'separate'  # 'separate', 'weighted', 'mixed'
    min_samples_per_regime: int = 100           # 每个状态最少样本数
    regime_model_cache: bool = True             # 缓存状态特定模型
    
    # 预测策略
    regime_prediction_mode: str = 'adaptive'   # 'adaptive', 'ensemble', 'switching'
    regime_weight_smoothing: float = 0.8       # 状态权重平滑系数
    
    # 性能优化
    parallel_regime_training: bool = True      # 并行训练各状态模型
    regime_feature_selection: bool = True      # 状态特定特征选择

class RegimeAwareTrainer:
    """
    状态感知训练器
    
    核心功能：
    1. 基于市场状态分段训练模型
    2. 状态特定的特征工程和模型优化
    3. 智能的状态切换和预测融合
    4. 与BMA系统的无缝集成
    """
    
    def __init__(self, config: RegimeTrainingConfig = None):
        self.config = config or RegimeTrainingConfig()
        
        # 初始化状态检测器
        regime_config = self.config.regime_config or RegimeConfig()
        self.regime_detector = MarketRegimeDetector(regime_config)
        
        # 状态特定模型存储
        self.regime_models = {}          # {regime_id: model}
        self.regime_features = {}        # {regime_id: feature_names}
        self.regime_performance = {}     # {regime_id: metrics}
        self.regime_thresholds = {}      # {regime_id: thresholds}
        
        # 预测状态管理
        self.current_regime = 0
        self.regime_probabilities = np.array([1.0, 0.0, 0.0])  # 状态概率
        self.regime_history = []
        
        # 缓存目录
        self.cache_dir = Path("cache/regime_models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RegimeAwareTrainer初始化完成，策略: {self.config.regime_training_strategy}")
    
    def fit_regime_aware_models(self, 
                               X: pd.DataFrame, 
                               y: pd.Series,
                               base_model_class,
                               cv_splitter: BaseCrossValidator = None,
                               **model_kwargs) -> Dict[str, Any]:
        """
        训练状态感知模型
        
        Args:
            X: 特征数据
            y: 目标变量
            base_model_class: 基础模型类
            cv_splitter: CV分割器
            **model_kwargs: 模型参数
            
        Returns:
            训练结果字典
        """
        
        if not self.config.enable_regime_aware:
            logger.info("状态感知训练未启用，使用标准训练")
            return self._fit_standard_model(X, y, base_model_class, **model_kwargs)
        
        logger.info("开始状态感知模型训练...")
        
        try:
            # 1. 检测市场状态
            regimes = self.regime_detector.detect_regimes(X)
            if len(regimes) != len(X):
                logger.warning("状态检测结果长度不匹配，使用标准训练")
                return self._fit_standard_model(X, y, base_model_class, **model_kwargs)
            
            # 2. 状态数据分割
            regime_data = self._split_data_by_regime(X, y, regimes)
            
            # 3. 验证每个状态的样本数量
            regime_data = self._validate_regime_samples(regime_data)
            
            if not regime_data:
                logger.warning("没有足够的状态数据，使用标准训练")
                return self._fit_standard_model(X, y, base_model_class, **model_kwargs)
            
            # 4. 训练状态特定模型
            training_results = {}
            
            if self.config.parallel_regime_training and len(regime_data) > 1:
                # 并行训练
                training_results = self._train_regimes_parallel(
                    regime_data, base_model_class, cv_splitter, **model_kwargs
                )
            else:
                # 串行训练
                training_results = self._train_regimes_sequential(
                    regime_data, base_model_class, cv_splitter, **model_kwargs
                )
            
            # 5. 模型性能评估和缓存
            self._evaluate_and_cache_models(training_results, regime_data)
            
            # 6. 构造返回结果
            result = self._construct_training_result(training_results, regime_data, regimes)
            
            logger.info(f"状态感知训练完成，训练了 {len(training_results)} 个状态模型")
            return result
            
        except Exception as e:
            logger.error(f"状态感知训练失败: {e}")
            logger.info("回退到标准训练模式")
            return self._fit_standard_model(X, y, base_model_class, **model_kwargs)
    
    def _split_data_by_regime(self, 
                             X: pd.DataFrame, 
                             y: pd.Series, 
                             regimes: pd.Series) -> Dict[int, Dict[str, Any]]:
        """按状态分割数据"""
        
        regime_data = {}
        
        for regime_id in regimes.unique():
            if pd.isna(regime_id):
                continue
                
            regime_mask = regimes == regime_id
            regime_X = X.loc[regime_mask]
            regime_y = y.loc[regime_mask]
            
            if len(regime_X) > 0:
                regime_data[int(regime_id)] = {
                    'X': regime_X,
                    'y': regime_y,
                    'indices': regime_mask,
                    'sample_count': len(regime_X)
                }
        
        logger.info(f"数据按状态分割完成: {[(r, d['sample_count']) for r, d in regime_data.items()]}")
        return regime_data
    
    def _validate_regime_samples(self, regime_data: Dict) -> Dict:
        """验证每个状态的样本数量"""
        
        valid_regimes = {}
        min_samples = self.config.min_samples_per_regime
        
        for regime_id, data in regime_data.items():
            sample_count = data['sample_count']
            
            if sample_count >= min_samples:
                valid_regimes[regime_id] = data
                logger.info(f"状态 {regime_id}: {sample_count} 样本 (有效)")
            else:
                logger.warning(f"状态 {regime_id}: {sample_count} 样本 (不足，需要至少 {min_samples})")
        
        return valid_regimes
    
    def _train_regimes_sequential(self, 
                                 regime_data: Dict, 
                                 base_model_class,
                                 cv_splitter,
                                 **model_kwargs) -> Dict:
        """串行训练各状态模型"""
        
        results = {}
        
        for regime_id, data in regime_data.items():
            logger.info(f"训练状态 {regime_id} 模型...")
            
            try:
                # 创建模型实例
                model = base_model_class(**model_kwargs)
                
                # 状态特定特征选择
                X_regime = data['X']
                y_regime = data['y']
                
                if self.config.regime_feature_selection:
                    X_regime = self._select_regime_features(X_regime, y_regime, regime_id)
                
                # 训练模型
                model.fit(X_regime, y_regime)
                
                # 存储结果
                results[regime_id] = {
                    'model': model,
                    'features': list(X_regime.columns),
                    'sample_count': len(X_regime),
                    'regime_id': regime_id
                }
                
                logger.info(f"状态 {regime_id} 模型训练完成")
                
            except Exception as e:
                logger.error(f"状态 {regime_id} 模型训练失败: {e}")
                continue
        
        return results
    
    def _train_regimes_parallel(self, 
                               regime_data: Dict, 
                               base_model_class,
                               cv_splitter, 
                               **model_kwargs) -> Dict:
        """并行训练各状态模型"""
        
        # 暂时使用串行实现，避免复杂的并行化问题
        logger.info("使用串行训练（并行训练待优化）")
        return self._train_regimes_sequential(regime_data, base_model_class, cv_splitter, **model_kwargs)
    
    def _select_regime_features(self, X: pd.DataFrame, y: pd.Series, regime_id: int) -> pd.DataFrame:
        """状态特定特征选择"""
        
        if not self.config.regime_feature_selection:
            return X
        
        # 简单的特征选择：基于相关性
        try:
            correlations = X.corrwith(y).abs()
            # 选择前80%的特征
            n_features = max(int(len(correlations) * 0.8), 5)
            top_features = correlations.nlargest(n_features).index
            
            selected_X = X[top_features]
            logger.debug(f"状态 {regime_id}: 从 {len(X.columns)} 选择了 {len(selected_X.columns)} 个特征")
            
            return selected_X
            
        except Exception as e:
            logger.warning(f"状态 {regime_id} 特征选择失败: {e}，使用全部特征")
            return X
    
    def _evaluate_and_cache_models(self, training_results: Dict, regime_data: Dict):
        """评估和缓存模型"""
        
        for regime_id, result in training_results.items():
            # 存储到实例变量
            self.regime_models[regime_id] = result['model']
            self.regime_features[regime_id] = result['features']
            
            # 简单性能评估（使用训练数据）
            model = result['model']
            data = regime_data[regime_id]
            
            try:
                y_pred = model.predict(data['X'][result['features']])
                
                # 计算简单指标
                from sklearn.metrics import mean_squared_error
                mse = mean_squared_error(data['y'], y_pred)
                
                self.regime_performance[regime_id] = {
                    'mse': mse,
                    'sample_count': result['sample_count']
                }
                
                logger.info(f"状态 {regime_id} 性能 - MSE: {mse:.6f}")
                
            except Exception as e:
                logger.warning(f"状态 {regime_id} 性能评估失败: {e}")
                self.regime_performance[regime_id] = {'mse': float('inf'), 'sample_count': result['sample_count']}
            
            # 缓存模型
            if self.config.regime_model_cache:
                self._cache_regime_model(regime_id, result)
    
    def _cache_regime_model(self, regime_id: int, result: Dict):
        """缓存状态模型"""
        
        try:
            cache_file = self.cache_dir / f"regime_{regime_id}_model.pkl"
            joblib.dump({
                'model': result['model'],
                'features': result['features'],
                'regime_id': regime_id,
                'sample_count': result['sample_count']
            }, cache_file)
            
            logger.debug(f"状态 {regime_id} 模型已缓存: {cache_file}")
            
        except Exception as e:
            logger.warning(f"状态 {regime_id} 模型缓存失败: {e}")
    
    def _construct_training_result(self, 
                                  training_results: Dict, 
                                  regime_data: Dict, 
                                  regimes: pd.Series) -> Dict:
        """构造训练结果"""
        
        # 统计信息
        total_samples = sum(data['sample_count'] for data in regime_data.values())
        regime_distribution = {r: data['sample_count']/total_samples for r, data in regime_data.items()}
        
        # 状态统计
        regime_stats = self.regime_detector.get_regime_statistics(regimes)
        
        return {
            'regime_aware': True,
            'regime_models': self.regime_models,
            'regime_features': self.regime_features,
            'regime_performance': self.regime_performance,
            'regime_distribution': regime_distribution,
            'regime_statistics': regime_stats,
            'training_regimes': list(training_results.keys()),
            'total_samples': total_samples,
            'regime_detector': self.regime_detector
        }
    
    def _fit_standard_model(self, X: pd.DataFrame, y: pd.Series, base_model_class, **model_kwargs) -> Dict:
        """标准模型训练（回退方案）"""
        
        logger.info("执行标准模型训练")
        
        try:
            model = base_model_class(**model_kwargs)
            model.fit(X, y)
            
            return {
                'regime_aware': False,
                'model': model,
                'features': list(X.columns),
                'sample_count': len(X)
            }
            
        except Exception as e:
            logger.error(f"标准模型训练也失败: {e}")
            raise
    
    def predict_regime_aware(self, X: pd.DataFrame) -> np.ndarray:
        """
        状态感知预测
        
        Args:
            X: 预测特征数据
            
        Returns:
            预测结果数组
        """
        
        if not hasattr(self, 'regime_models') or not self.regime_models:
            raise ValueError("没有训练好的状态模型")
        
        try:
            # 1. 检测当前市场状态
            current_regimes = self.regime_detector.detect_regimes(X)
            
            # 2. 基于预测模式进行预测
            if self.config.regime_prediction_mode == 'adaptive':
                return self._predict_adaptive(X, current_regimes)
            elif self.config.regime_prediction_mode == 'ensemble':
                return self._predict_ensemble(X, current_regimes)
            elif self.config.regime_prediction_mode == 'switching':
                return self._predict_switching(X, current_regimes)
            else:
                raise ValueError(f"不支持的预测模式: {self.config.regime_prediction_mode}")
                
        except Exception as e:
            logger.error(f"状态感知预测失败: {e}")
            # 回退到第一个可用模型
            if self.regime_models:
                fallback_regime = list(self.regime_models.keys())[0]
                return self._predict_single_regime(X, fallback_regime)
            else:
                raise ValueError("没有可用的状态模型进行预测")
    
    def _predict_adaptive(self, X: pd.DataFrame, regimes: pd.Series) -> np.ndarray:
        """自适应预测：根据实时状态选择模型"""
        
        predictions = np.zeros(len(X))
        
        for regime_id in regimes.unique():
            if pd.isna(regime_id) or int(regime_id) not in self.regime_models:
                continue
                
            regime_mask = regimes == regime_id
            regime_X = X.loc[regime_mask]
            
            if len(regime_X) > 0:
                pred = self._predict_single_regime(regime_X, int(regime_id))
                predictions[regime_mask] = pred
        
        # 处理没有匹配状态的数据点（使用默认模型）
        unmatched_mask = predictions == 0
        if unmatched_mask.sum() > 0:
            default_regime = list(self.regime_models.keys())[0]
            default_pred = self._predict_single_regime(X.loc[unmatched_mask], default_regime)
            predictions[unmatched_mask] = default_pred
        
        return predictions
    
    def _predict_ensemble(self, X: pd.DataFrame, regimes: pd.Series) -> np.ndarray:
        """集成预测：所有状态模型的加权平均"""
        
        all_predictions = []
        model_weights = []
        
        for regime_id, model in self.regime_models.items():
            pred = self._predict_single_regime(X, regime_id)
            all_predictions.append(pred)
            
            # 基于性能的权重
            performance = self.regime_performance.get(regime_id, {'mse': 1.0})
            weight = 1.0 / (1.0 + performance['mse'])
            model_weights.append(weight)
        
        # 归一化权重
        total_weight = sum(model_weights)
        model_weights = [w / total_weight for w in model_weights]
        
        # 加权平均
        ensemble_pred = np.zeros(len(X))
        for pred, weight in zip(all_predictions, model_weights):
            ensemble_pred += pred * weight
        
        return ensemble_pred
    
    def _predict_switching(self, X: pd.DataFrame, regimes: pd.Series) -> np.ndarray:
        """切换预测：基于状态概率的软切换"""
        
        # 简化实现：使用自适应预测
        return self._predict_adaptive(X, regimes)
    
    def _predict_single_regime(self, X: pd.DataFrame, regime_id: int) -> np.ndarray:
        """使用特定状态模型预测"""
        
        if regime_id not in self.regime_models:
            raise ValueError(f"状态 {regime_id} 的模型不存在")
        
        model = self.regime_models[regime_id]
        features = self.regime_features[regime_id]
        
        # 选择相应特征
        X_features = X[features]
        
        return model.predict(X_features)
    
    def get_regime_model_summary(self) -> Dict[str, Any]:
        """获取状态模型摘要"""
        
        if not self.regime_models:
            return {"message": "没有训练好的状态模型"}
        
        summary = {
            "trained_regimes": list(self.regime_models.keys()),
            "regime_descriptions": self.regime_detector.get_regime_descriptions(),
            "regime_performance": self.regime_performance,
            "model_count": len(self.regime_models)
        }
        
        return summary


def create_regime_aware_trainer(config: RegimeTrainingConfig = None) -> RegimeAwareTrainer:
    """工厂函数：创建状态感知训练器"""
    return RegimeAwareTrainer(config)