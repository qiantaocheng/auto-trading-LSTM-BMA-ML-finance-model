#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML增强集成模块 - 将特征选择、超参数优化和集成学习整合到BMA系统
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# 导入新创建的ML模块 - 使用条件导入避免破坏系统
# 🚫 已删除MLFeatureSelector - 仅使用RobustFeatureSelector
try:
    from ml_hyperparameter_optimization import MLHyperparameterOptimizer, HyperparameterConfig
    ML_HYPEROPT_AVAILABLE = True
except ImportError:
    ML_HYPEROPT_AVAILABLE = False
    MLHyperparameterOptimizer = None
    HyperparameterConfig = None

try:
    from ml_ensemble_enhanced import MLEnsembleEnhanced, EnsembleConfig, DynamicBMAWeightLearner
    ML_ENSEMBLE_AVAILABLE = True
except ImportError:
    ML_ENSEMBLE_AVAILABLE = False
    MLEnsembleEnhanced = None
    EnsembleConfig = None
    DynamicBMAWeightLearner = None

logger = logging.getLogger(__name__)


@dataclass
class MLEnhancementConfig:
    """ML增强集成配置"""
    enable_feature_selection: bool = False  # 强制禁用：仅RobustFeatureSelector可改列
    enable_hyperparameter_optimization: bool = True
    enable_ensemble_learning: bool = True
    enable_dynamic_bma_weights: bool = True
    
    # 子模块配置
    # feature_selection_config: 已删除 - 仅使用RobustFeatureSelector
    hyperparameter_config: HyperparameterConfig = None
    ensemble_config: EnsembleConfig = None
    
    # 性能配置
    n_jobs: int = -1
    random_state: int = 42
    verbose: int = 1


class MLEnhancementSystem:
    """ML增强系统 - 整合特征选择、超参数优化和集成学习"""
    
    def __init__(self, config: MLEnhancementConfig = None):
        self.config = config or MLEnhancementConfig()
        
        # 初始化子系统
        self.feature_selector = None
        self.hyperparameter_optimizer = None
        self.ensemble_builder = None
        self.dynamic_bma_learner = None
        
        # 结果缓存
        self.selected_features = None
        self.best_hyperparameters = {}
        self.ensemble_models = {}
        self.bma_weights = None
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._initialize_subsystems()
    
    def _initialize_subsystems(self):
        """初始化各个子系统"""
        # 🚨 强制禁用特征选择，仅RobustFeatureSelector可改列
        if self.config.enable_feature_selection:
            raise NotImplementedError(
                "违反SSOT原则：ML增强系统不允许特征选择！\n"
                "修复指南：仅RobustFeatureSelector可改变列，请在主流程中完成特征选择后再调用训练头。\n"
                "设置 enable_feature_selection=False"
            )
        
        if self.config.enable_hyperparameter_optimization and ML_HYPEROPT_AVAILABLE:
            self.hyperparameter_optimizer = MLHyperparameterOptimizer(
                self.config.hyperparameter_config or HyperparameterConfig()
            )
            self.logger.info("超参数优化器已初始化")
        elif self.config.enable_hyperparameter_optimization:
            self.logger.warning("超参数优化已启用但模块不可用")
        
        if self.config.enable_ensemble_learning and ML_ENSEMBLE_AVAILABLE:
            self.ensemble_builder = MLEnsembleEnhanced(
                self.config.ensemble_config or EnsembleConfig()
            )
            self.logger.info("集成学习系统已初始化")
        elif self.config.enable_ensemble_learning:
            self.logger.warning("集成学习已启用但模块不可用")
    
    def enhance_training_pipeline(self, X: pd.DataFrame, y: pd.Series, 
                                 cv_factory: callable,
                                 feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        增强的训练流程
        
        Args:
            X: 原始特征数据
            y: 目标变量
            cv_factory: 统一CV工厂（必须）
            feature_names: 特征名称
            
        Returns:
            包含所有增强结果的字典
        """
        # 保存CV工厂供后续使用
        self.cv_factory = cv_factory
        self.logger.info("开始ML增强训练流程")
        results = {}
        
        # 1. 特征选择
        if self.config.enable_feature_selection and self.feature_selector:
            self.logger.info("执行智能特征选择...")
            X_selected, selection_info = self.feature_selector.fit_select(X, y, feature_names)
            self.selected_features = selection_info['selected_feature_names']
            results['feature_selection'] = {
                'selected_features': self.selected_features,
                'original_count': X.shape[1],
                'selected_count': X_selected.shape[1],
                'reduction_ratio': selection_info['reduction_ratio']
            }
            X = X_selected  # 使用选择后的特征
            self.logger.info(f"特征从 {results['feature_selection']['original_count']} 减少到 "
                           f"{results['feature_selection']['selected_count']}")
        else:
            X_selected = X
        
        # 2. 超参数优化
        if self.config.enable_hyperparameter_optimization and self.hyperparameter_optimizer:
            self.logger.info("执行超参数优化...")
            
            # 优化互补性三类模型
            models_to_optimize = ['ElasticNet', 'LightGBM', 'ExtraTrees']
            
            # 检查依赖可用性
            import importlib.util
            if not importlib.util.find_spec('lightgbm'):
                models_to_optimize.remove('LightGBM')
                models_to_optimize.append('GradientBoosting')  # 回退到sklearn GBDT
                
            # 可选备胎XGBoost（如果可用）
            if importlib.util.find_spec('xgboost'):
                models_to_optimize.append('XGBoost')
            
            optimization_results = self.hyperparameter_optimizer.optimize_multiple_models(
                models_to_optimize, X_selected.values, y.values
            )
            
            # 保存最优参数
            for model_name, (best_params, best_score, best_model) in optimization_results.items():
                self.best_hyperparameters[model_name] = best_params
            
            results['hyperparameter_optimization'] = {
                'optimized_models': list(optimization_results.keys()),
                'best_model': max(optimization_results.keys(), 
                                 key=lambda k: optimization_results[k][1]),
                'best_params': self.best_hyperparameters
            }
            
            self.logger.info(f"最优模型: {results['hyperparameter_optimization']['best_model']}")
        
        # 3. 集成学习
        if self.config.enable_ensemble_learning and self.ensemble_builder:
            self.logger.info("构建集成学习系统...")
            
            # 使用优化后的超参数创建模型
            if self.best_hyperparameters:
                # 更新集成配置中的基础模型
                self.ensemble_builder.config.base_models = list(self.best_hyperparameters.keys())
            
            ensemble_results = self.ensemble_builder.build_ensemble(
                X_selected.values, y.values,
                cv_factory=self.cv_factory,  # 传入统一CV工厂
                feature_names=X_selected.columns.tolist()
            )
            
            self.ensemble_models = ensemble_results['models']
            results['ensemble_learning'] = {
                'methods': list(ensemble_results['models'].keys()),
                'best_method': ensemble_results['best_method'],
                'performances': ensemble_results['performances']
            }
            
            self.logger.info(f"最优集成方法: {ensemble_results['best_method']}")
        
        # 4. OOF标准化 + 相关惩罚BMA权重学习
        if self.config.enable_dynamic_bma_weights:
            self.logger.info("OOF标准化 + 相关惩罚BMA权重学习...")
            
            # 获取基础模型和OOF预测
            base_models = []
            oof_predictions_matrix = []
            
            if optimization_results:
                base_models = [result[2] for result in optimization_results.values() if result[2]]
                # 生成统一CV的OOF预测矩阵 - 必须使用外部传入的CV
                # 🚨 使用统一SSOT检测器
                from .ssot_violation_detector import ensure_cv_factory_provided
                ensure_cv_factory_provided(getattr(self, 'cv_factory', None), "OOF预测矩阵生成")
                
                for model_name, (best_params, best_score, best_model) in optimization_results.items():
                    if best_model:
                        # 使用外部传入的统一CV工厂
                        from sklearn.model_selection import cross_val_predict
                        cv = self.cv_factory  # 必须使用外部CV工厂
                        oof_pred = cross_val_predict(best_model, X_selected.values, y.values, cv=cv)
                        oof_predictions_matrix.append(oof_pred)
            
            if base_models and len(oof_predictions_matrix) > 1:
                # Step 4.1: OOF横截面标准化 (Rank→Normal)
                import numpy as np
                from scipy.stats import norm
                
                standardized_oof_matrix = []
                for oof_pred in oof_predictions_matrix:
                    # 横截面Rank→Normal标准化
                    oof_df = pd.DataFrame({'pred': oof_pred, 'y': y.values})
                    oof_df['rank_pct'] = oof_df['pred'].rank(pct=True)
                    oof_df['standardized'] = norm.ppf(oof_df['rank_pct'].clip(0.01, 0.99))
                    standardized_oof_matrix.append(oof_df['standardized'].values)
                
                # Step 4.2: 相关性门槛 + 自动裁剪
                oof_corr_matrix = np.corrcoef(standardized_oof_matrix)
                max_correlation = np.max(np.abs(oof_corr_matrix - np.eye(len(oof_corr_matrix))))
                
                # Step 4.3: 相关惩罚BMA权重计算
                # w_i ∝ shrink(IC_i) × ICIR_i × (1 - ρ̄_i)
                ic_scores = []
                for oof_std in standardized_oof_matrix:
                    ic = np.corrcoef(oof_std, y.values)[0, 1] if len(y) > 1 else 0
                    ic_scores.append(max(0.015, abs(ic)))  # shrink到最低0.015
                
                # 计算每个模型与其他模型的平均相关性
                avg_correlations = []
                for i in range(len(oof_corr_matrix)):
                    other_corrs = [abs(oof_corr_matrix[i, j]) for j in range(len(oof_corr_matrix)) if i != j]
                    avg_corr = np.mean(other_corrs) if other_corrs else 0
                    avg_correlations.append(avg_corr)
                
                # 相关惩罚BMA权重公式
                raw_weights = []
                for i, (ic, avg_corr) in enumerate(zip(ic_scores, avg_correlations)):
                    icir = ic / (0.1 + np.std([ic]))  # 简化ICIR
                    correlation_penalty = 1 - min(0.5, avg_corr)  # 相关惩罚
                    weight = ic * icir * correlation_penalty
                    raw_weights.append(weight)
                
                # 权重归一化
                raw_weights = np.array(raw_weights)
                self.bma_weights = raw_weights / (np.sum(raw_weights) + 1e-8)
                
                results['oof_standardized_bma'] = {
                    'standardized_oof_matrix': len(standardized_oof_matrix),
                    'max_correlation': max_correlation,
                    'correlation_threshold': 0.85,
                    'ic_scores': ic_scores,
                    'correlation_penalties': [1-corr for corr in avg_correlations],
                    'final_weights': self.bma_weights.tolist(),
                    'correlation_compliant': max_correlation <= 0.85
                }
                
                self.logger.info(f"OOF标准化BMA权重: {self.bma_weights}")
                self.logger.info(f"最大OOF相关性: {max_correlation:.3f} ({'✅' if max_correlation <= 0.85 else '❌'})")
        
        # 总结
        results['summary'] = {
            'features_selected': len(self.selected_features) if self.selected_features else X.shape[1],
            'models_optimized': len(self.best_hyperparameters),
            'ensemble_methods': len(self.ensemble_models),
            'bma_weights_learned': self.bma_weights is not None
        }
        
        self.logger.info("ML增强训练流程完成")
        
        return results
    
    def predict_enhanced(self, X: pd.DataFrame) -> np.ndarray:
        """
        使用增强系统进行预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果
        """
        # 1. 特征选择
        if self.feature_selector and self.selected_features:
            X = self.feature_selector.transform(X)
        
        # 2. 使用最优集成模型预测
        if self.ensemble_models:
            # 获取所有集成模型的预测
            predictions = []
            for name, model in self.ensemble_models.items():
                try:
                    pred = model.predict(X.values)
                    predictions.append(pred)
                except Exception as e:
                    self.logger.warning(f"模型 {name} 预测失败: {e}")
            
            if predictions:
                # 平均所有预测
                return np.mean(predictions, axis=0)
        
        # 3. 回退：使用动态BMA
        if self.dynamic_bma_learner:
            return self.dynamic_bma_learner.predict(X.values)
        
        # 最终回退
        return np.zeros(len(X))
    
    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性报告"""
        if self.feature_selector:
            return self.feature_selector.get_feature_importance_report()
        return pd.DataFrame()
    
    def get_optimization_report(self) -> pd.DataFrame:
        """获取超参数优化报告"""
        if self.hyperparameter_optimizer:
            return self.hyperparameter_optimizer.get_optimization_report()
        return pd.DataFrame()
    
    def update_bma_weights(self, X: np.ndarray, y: np.ndarray):
        """在线更新BMA权重"""
        if self.dynamic_bma_learner:
            self.dynamic_bma_learner.update_weights(X, y)
            self.bma_weights = self.dynamic_bma_learner.weights_
            self.logger.info(f"BMA权重已更新: {self.bma_weights}")


def integrate_ml_enhancements(X: pd.DataFrame, y: pd.Series, 
                             config: Optional[MLEnhancementConfig] = None) -> Dict[str, Any]:
    """
    便捷函数：整合所有ML增强功能
    
    Args:
        X: 特征数据
        y: 目标变量
        config: 配置
        
    Returns:
        增强训练结果
    """
    system = MLEnhancementSystem(config)
    return system.enhance_training_pipeline(X, y)