#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一训练协调器 - 替代复杂的多训练头系统
单一接口协调所有ML训练，确保SSOT原则
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class UnifiedTrainingCoordinator:
    """
    统一训练协调器 - 单一真相源
    
    职责：
    1. 协调所有ML训练头（Traditional ML, Learning-to-Rank等）
    2. 确保统一使用RobustFeatureSelector和UnifiedCVFactory  
    3. 统一OOF预测生成和对齐
    4. 防止训练头之间的冲突和重复
    """
    
    def __init__(self, cv_factory: Callable):
        """
        初始化统一训练协调器
        
        Args:
            cv_factory: 统一CV工厂
        """
        self.cv_factory = cv_factory
        self.training_results = {}
        self.unified_oof_results = {}
        
        logger.info("✅ 统一训练协调器初始化完成")
    
    def coordinate_all_training(self, 
                               X: pd.DataFrame,
                               y: pd.Series, 
                               dates: pd.Series,
                               tickers: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        协调所有训练头的训练过程
        
        Args:
            X: 已通过RobustFeatureSelector的特征矩阵
            y: 目标变量
            dates: 日期序列
            tickers: 股票代码（可选）
            
        Returns:
            统一的训练结果
        """
        logger.info("🚀 启动统一训练协调...")
        logger.info(f"输入数据: {len(X)} 样本, {len(X.columns)} 特征")
        
        coordinated_results = {
            'training_heads': {},
            'unified_oof_matrix': None,
            'final_ensemble': None,
            'coordination_metadata': {
                'coordinator_version': '1.0',
                'training_timestamp': datetime.now().isoformat(),
                'data_fingerprint': {
                    'samples': len(X),
                    'features': len(X.columns),
                    'feature_names': list(X.columns),
                    'date_range': f"{dates.min()} to {dates.max()}"
                }
            }
        }
        
        # 1. 传统ML训练头
        logger.info("🎯 训练头1: Traditional ML")
        traditional_result = self._train_traditional_ml_head(X, y, dates, tickers)
        coordinated_results['training_heads']['traditional_ml'] = traditional_result
        
        # 2. Learning-to-Rank训练头（简化版）
        logger.info("🎯 训练头2: Learning-to-Rank (简化)")
        ltr_result = self._train_learning_to_rank_head(X, y, dates, tickers)
        coordinated_results['training_heads']['learning_to_rank'] = ltr_result
        
        # 3. 统一OOF矩阵生成
        logger.info("🔄 生成统一OOF矩阵...")
        unified_oof_matrix = self._create_unified_oof_matrix(coordinated_results['training_heads'])
        coordinated_results['unified_oof_matrix'] = unified_oof_matrix
        
        # 4. 最终集成
        logger.info("🏆 生成最终集成预测...")
        final_ensemble = self._create_final_ensemble(unified_oof_matrix, y)
        coordinated_results['final_ensemble'] = final_ensemble
        
        logger.info("✅ 统一训练协调完成")
        
        return coordinated_results
    
    def _train_traditional_ml_head(self, X: pd.DataFrame, y: pd.Series, 
                                 dates: pd.Series, tickers: pd.Series) -> Dict[str, Any]:
        """训练传统ML头"""
        try:
            from .traditional_ml_head import TraditionalMLHead
            
            # 创建传统ML训练器
            trainer = TraditionalMLHead(enable_hyperparam_opt=True)
            
            # 使用统一接口训练
            result = trainer.fit(X, y, dates, tickers or pd.Series(['DUMMY'] * len(X)), self.cv_factory)
            
            logger.info(f"传统ML训练完成: {len(result.get('models', {}))} 个模型")
            
            return {
                'success': True,
                'models': result.get('models', {}),
                'oof_predictions': result.get('oof', pd.Series()),
                'metadata': result.get('metadata', {}),
                'training_head_id': 'traditional_ml'
            }
            
        except Exception as e:
            logger.error(f"传统ML训练失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_head_id': 'traditional_ml'
            }
    
    def _train_learning_to_rank_head(self, X: pd.DataFrame, y: pd.Series,
                                   dates: pd.Series, tickers: pd.Series) -> Dict[str, Any]:
        """训练Learning-to-Rank头（简化版）"""
        try:
            from .unified_oof_generator import generate_unified_oof
            from sklearn.ensemble import RandomForestRegressor
            
            # 创建简化的排序模型
            ranking_models = {
                'rf_ranker': RandomForestRegressor(n_estimators=100, random_state=42),
                'simple_ranker': RandomForestRegressor(n_estimators=50, random_state=42)
            }
            
            # 生成统一OOF预测
            unified_result = generate_unified_oof(
                X=X, y=y, dates=dates,
                models=ranking_models,
                training_head_id='learning_to_rank',
                cv_factory=self.cv_factory
            )
            
            # 选择最佳预测
            oof_results = unified_result['oof_results']
            if oof_results:
                best_model = max(oof_results.keys(), key=lambda k: oof_results[k]['oof_ic'])
                primary_oof = oof_results[best_model]['oof_predictions']
                best_ic = oof_results[best_model]['oof_ic']
            else:
                primary_oof = pd.Series(index=X.index, dtype=float).fillna(0.0)
                best_ic = 0.0
            
            logger.info(f"Learning-to-Rank训练完成: 最佳IC={best_ic:.4f}")
            
            return {
                'success': True,
                'models': ranking_models,
                'oof_predictions': primary_oof,
                'unified_oof_result': unified_result,
                'best_ic': best_ic,
                'training_head_id': 'learning_to_rank'
            }
            
        except Exception as e:
            logger.error(f"Learning-to-Rank训练失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_head_id': 'learning_to_rank'
            }
    
    def _create_unified_oof_matrix(self, training_heads: Dict[str, Any]) -> pd.DataFrame:
        """创建统一的OOF预测矩阵"""
        oof_predictions = []
        
        for head_name, head_result in training_heads.items():
            if head_result.get('success', False) and 'oof_predictions' in head_result:
                oof_pred = head_result['oof_predictions']
                if isinstance(oof_pred, pd.Series):
                    oof_pred.name = f"{head_name}_oof"
                    oof_predictions.append(oof_pred)
        
        if not oof_predictions:
            logger.warning("没有有效的OOF预测")
            return pd.DataFrame()
        
        # 对齐预测矩阵
        unified_matrix = pd.concat(oof_predictions, axis=1)
        
        logger.info(f"统一OOF矩阵: {unified_matrix.shape}")
        
        return unified_matrix
    
    def _create_final_ensemble(self, oof_matrix: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """创建最终集成预测"""
        if oof_matrix.empty:
            logger.warning("OOF矩阵为空，无法创建集成")
            return {
                'ensemble_prediction': pd.Series(index=y.index, dtype=float).fillna(0.0),
                'ensemble_weights': {},
                'ensemble_ic': 0.0
            }
        
        try:
            # 计算各预测的IC权重
            from scipy.stats import spearmanr
            
            weights = {}
            total_ic = 0
            
            for col in oof_matrix.columns:
                valid_mask = ~(oof_matrix[col].isna() | y.isna())
                if valid_mask.sum() > 10:  # 至少10个有效样本
                    ic = spearmanr(y[valid_mask], oof_matrix[col][valid_mask])[0]
                    ic = max(ic, 0.01)  # 最小权重
                    weights[col] = ic
                    total_ic += ic
                else:
                    weights[col] = 0.01
            
            # 归一化权重
            if total_ic > 0:
                weights = {k: v/total_ic for k, v in weights.items()}
            else:
                equal_weight = 1.0 / len(oof_matrix.columns)
                weights = {k: equal_weight for k in oof_matrix.columns}
            
            # 生成加权集成预测
            ensemble_pred = pd.Series(index=oof_matrix.index, dtype=float).fillna(0.0)
            
            for col, weight in weights.items():
                pred_values = oof_matrix[col].fillna(0.0)
                ensemble_pred += weight * pred_values
            
            # 计算集成IC
            valid_mask = ~(ensemble_pred.isna() | y.isna())
            if valid_mask.sum() > 10:
                ensemble_ic = spearmanr(y[valid_mask], ensemble_pred[valid_mask])[0]
            else:
                ensemble_ic = 0.0
            
            logger.info(f"最终集成IC: {ensemble_ic:.4f}")
            
            return {
                'ensemble_prediction': ensemble_pred,
                'ensemble_weights': weights,
                'ensemble_ic': ensemble_ic,
                'component_count': len(weights)
            }
            
        except Exception as e:
            logger.error(f"集成创建失败: {e}")
            return {
                'ensemble_prediction': pd.Series(index=y.index, dtype=float).fillna(0.0),
                'ensemble_weights': {},
                'ensemble_ic': 0.0
            }
    
    def get_coordination_report(self) -> Dict[str, Any]:
        """获取协调报告"""
        return {
            'coordinator_status': 'active',
            'training_results_count': len(self.training_results),
            'unified_oof_count': len(self.unified_oof_results),
            'cv_factory_type': type(self.cv_factory).__name__
        }


# 全局统一训练协调器
_global_coordinator = None

def get_unified_training_coordinator(cv_factory: Callable = None) -> UnifiedTrainingCoordinator:
    """获取全局统一训练协调器"""
    global _global_coordinator
    
    if _global_coordinator is None:
        if cv_factory is None:
            from .unified_cv_factory import get_unified_cv_factory
            factory = get_unified_cv_factory()
            cv_factory = factory.create_cv_factory()
        
        _global_coordinator = UnifiedTrainingCoordinator(cv_factory)
    
    return _global_coordinator

def coordinate_unified_training(X: pd.DataFrame, 
                              y: pd.Series,
                              dates: pd.Series, 
                              tickers: Optional[pd.Series] = None,
                              cv_factory: Callable = None) -> Dict[str, Any]:
    """
    便捷函数：协调统一训练
    """
    coordinator = get_unified_training_coordinator(cv_factory)
    return coordinator.coordinate_all_training(X, y, dates, tickers)


if __name__ == "__main__":
    # 测试统一训练协调器
    logging.basicConfig(level=logging.INFO)
    
    # 创建模拟数据
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    X = pd.DataFrame(np.random.randn(200, 8), columns=[f'feature_{i}' for i in range(8)])
    y = pd.Series(np.random.randn(200), name='target')
    tickers = pd.Series(['AAPL'] * 100 + ['MSFT'] * 100, name='ticker')
    
    print("测试统一训练协调器")
    
    # 协调训练
    result = coordinate_unified_training(X, y, dates, tickers)
    
    print(f"协调结果: {list(result.keys())}")
    print(f"训练头数量: {len(result['training_heads'])}")
    if result['final_ensemble']:
        print(f"最终集成IC: {result['final_ensemble']['ensemble_ic']:.4f}")