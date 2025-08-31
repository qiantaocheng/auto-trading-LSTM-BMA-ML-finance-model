#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一OOF预测生成器 - 单一真相源
所有训练头必须通过此系统生成OOF预测，确保一致性
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class UnifiedOOFGenerator:
    """
    统一OOF预测生成器 - SSOT
    
    职责：
    1. 提供唯一的OOF预测生成接口
    2. 确保所有训练头使用相同的CV策略
    3. 统一OOF预测格式和index对齐
    4. 防止各训练头独立生成不一致的OOF
    """
    
    def __init__(self, cv_factory: Callable):
        """
        初始化统一OOF生成器
        
        Args:
            cv_factory: 统一CV工厂函数
        """
        self.cv_factory = cv_factory
        self.generated_oof = {}  # 缓存已生成的OOF
        self._validate_cv_factory()
        
        logger.info("✅ 统一OOF生成器初始化完成")
    
    def _validate_cv_factory(self):
        """验证CV工厂可用性"""
        if not callable(self.cv_factory):
            raise ValueError("cv_factory必须是可调用对象")
        
        logger.debug("CV工厂验证通过")
    
    def generate_oof_predictions(self, 
                                X: pd.DataFrame, 
                                y: pd.Series,
                                dates: pd.Series,
                                models: Dict[str, Any],
                                training_head_id: str) -> Dict[str, Any]:
        """
        生成统一的OOF预测
        
        Args:
            X: 特征矩阵
            y: 目标变量
            dates: 日期序列
            models: 训练好的模型字典
            training_head_id: 训练头标识
            
        Returns:
            统一格式的OOF预测结果
        """
        logger.info(f"🎯 生成统一OOF预测 - 训练头: {training_head_id}")
        
        # 验证输入数据一致性
        self._validate_input_consistency(X, y, dates)
        
        # 生成缓存键
        cache_key = f"{training_head_id}_{len(X)}_{hash(tuple(X.columns))}"
        
        if cache_key in self.generated_oof:
            logger.info(f"使用缓存的OOF预测: {cache_key}")
            return self.generated_oof[cache_key]
        
        # 创建统一CV分割器
        cv_splitter = self.cv_factory(dates)
        cv_splits = cv_splitter(X, y)
        
        logger.info(f"CV分割: {len(cv_splits)}折")
        
        # 初始化OOF预测矩阵
        oof_results = {}
        
        for model_name, model in models.items():
            logger.info(f"生成 {model_name} 的OOF预测")
            
            # 创建OOF预测向量
            oof_predictions = pd.Series(index=X.index, dtype=float, name=f"{model_name}_oof")
            oof_predictions.fillna(np.nan, inplace=True)
            
            # CV训练和预测
            fold_scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                try:
                    # 获取训练和验证数据
                    X_train = X.iloc[train_idx].fillna(0)
                    y_train = y.iloc[train_idx].fillna(0)
                    X_val = X.iloc[val_idx].fillna(0)
                    y_val = y.iloc[val_idx].fillna(0)
                    
                    # 训练模型
                    model.fit(X_train, y_train)
                    
                    # 生成预测
                    val_pred = model.predict(X_val)
                    
                    # 保存OOF预测
                    oof_predictions.iloc[val_idx] = val_pred
                    
                    # 计算fold得分
                    from scipy.stats import spearmanr
                    fold_ic = spearmanr(y_val, val_pred)[0]
                    fold_scores.append(fold_ic)
                    
                    logger.debug(f"Fold {fold_idx+1}: IC={fold_ic:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Fold {fold_idx+1} 训练失败: {e}")
                    continue
            
            # 计算整体OOF性能
            valid_mask = ~oof_predictions.isna()
            if valid_mask.sum() > 0:
                overall_ic = spearmanr(y[valid_mask], oof_predictions[valid_mask])[0]
                logger.info(f"{model_name} OOF IC: {overall_ic:.4f}")
            else:
                overall_ic = 0.0
                logger.warning(f"{model_name} 没有有效的OOF预测")
            
            # 保存结果
            oof_results[model_name] = {
                'oof_predictions': oof_predictions,
                'oof_ic': overall_ic,
                'fold_scores': fold_scores,
                'coverage': valid_mask.sum() / len(oof_predictions)
            }
        
        # 创建统一结果格式
        unified_result = {
            'training_head_id': training_head_id,
            'oof_results': oof_results,
            'cv_info': {
                'n_splits': len(cv_splits),
                'total_samples': len(X),
                'unique_dates': dates.nunique()
            },
            'data_fingerprint': {
                'features': list(X.columns),
                'date_range': f"{dates.min()} to {dates.max()}",
                'generation_time': datetime.now().isoformat()
            }
        }
        
        # 缓存结果
        self.generated_oof[cache_key] = unified_result
        
        logger.info(f"✅ 统一OOF预测生成完成 - {len(oof_results)}个模型")
        
        return unified_result
    
    def _validate_input_consistency(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series):
        """验证输入数据一致性"""
        if len(X) != len(y) or len(X) != len(dates):
            raise ValueError(f"数据长度不一致: X={len(X)}, y={len(y)}, dates={len(dates)}")
        
        if not X.index.equals(y.index):
            raise ValueError("X和y的index不匹配")
        
        logger.debug(f"输入数据验证通过: {len(X)}样本, {len(X.columns)}特征")
    
    def align_oof_predictions(self, oof_results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        对齐多个训练头的OOF预测
        
        Args:
            oof_results_list: 多个训练头的OOF结果列表
            
        Returns:
            对齐后的OOF预测矩阵
        """
        logger.info(f"🔄 对齐 {len(oof_results_list)} 个训练头的OOF预测")
        
        # 收集所有OOF预测
        all_oof_predictions = []
        
        for oof_result in oof_results_list:
            training_head = oof_result['training_head_id']
            oof_data = oof_result['oof_results']
            
            for model_name, model_result in oof_data.items():
                oof_pred = model_result['oof_predictions']
                oof_pred.name = f"{training_head}_{model_name}"
                all_oof_predictions.append(oof_pred)
        
        if not all_oof_predictions:
            logger.warning("没有有效的OOF预测可对齐")
            return pd.DataFrame()
        
        # 创建对齐的预测矩阵
        aligned_matrix = pd.concat(all_oof_predictions, axis=1)
        
        # 验证对齐质量
        coverage_stats = {}
        for col in aligned_matrix.columns:
            coverage = (~aligned_matrix[col].isna()).sum() / len(aligned_matrix)
            coverage_stats[col] = coverage
        
        logger.info(f"OOF对齐完成: {aligned_matrix.shape}, 平均覆盖率: {np.mean(list(coverage_stats.values())):.2%}")
        
        return aligned_matrix
    
    def get_generation_report(self) -> Dict[str, Any]:
        """获取OOF生成报告"""
        report = {
            'total_generated': len(self.generated_oof),
            'cache_keys': list(self.generated_oof.keys()),
            'generation_stats': {}
        }
        
        for cache_key, result in self.generated_oof.items():
            training_head = result['training_head_id']
            model_count = len(result['oof_results'])
            report['generation_stats'][training_head] = {
                'cache_key': cache_key,
                'model_count': model_count,
                'cv_splits': result['cv_info']['n_splits']
            }
        
        return report


# 全局统一OOF生成器实例
_global_oof_generator = None

def get_unified_oof_generator(cv_factory: Callable = None) -> UnifiedOOFGenerator:
    """获取全局统一OOF生成器"""
    global _global_oof_generator
    
    if _global_oof_generator is None:
        if cv_factory is None:
            # 使用默认的统一CV工厂
            from .unified_cv_factory import get_unified_cv_factory
            factory = get_unified_cv_factory()
            cv_factory = factory.create_cv_factory()
        
        _global_oof_generator = UnifiedOOFGenerator(cv_factory)
    
    return _global_oof_generator

def generate_unified_oof(X: pd.DataFrame, 
                        y: pd.Series,
                        dates: pd.Series, 
                        models: Dict[str, Any],
                        training_head_id: str,
                        cv_factory: Callable = None) -> Dict[str, Any]:
    """
    便捷函数：生成统一OOF预测
    """
    generator = get_unified_oof_generator(cv_factory)
    return generator.generate_oof_predictions(X, y, dates, models, training_head_id)


if __name__ == "__main__":
    # 测试统一OOF生成器
    logging.basicConfig(level=logging.INFO)
    
    # 创建模拟数据
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    X = pd.DataFrame(np.random.randn(500, 10), columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(np.random.randn(500), name='target')
    
    # 模拟模型
    from sklearn.linear_model import LinearRegression
    models = {
        'linear_1': LinearRegression(),
        'linear_2': LinearRegression()
    }
    
    print("测试统一OOF生成器")
    
    # 生成OOF预测
    result = generate_unified_oof(X, y, dates, models, 'test_head')
    
    print(f"生成结果: {result['training_head_id']}")
    print(f"模型数量: {len(result['oof_results'])}")
    
    # 测试报告
    generator = get_unified_oof_generator()
    report = generator.get_generation_report()
    print(f"生成报告: {report}")