#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两段式特征选择集成模块
将Stage-A和Stage-B无缝集成到主BMA系统
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TwoStageFeatureIntegrator:
    """
    两段式特征选择集成器
    负责将Stage-A和Stage-B集成到现有BMA系统
    """
    
    def __init__(self, bma_model, config_mode: str = 'default'):
        """
        初始化集成器
        
        Args:
            bma_model: 现有的BMA模型实例
            config_mode: 配置模式 ('default', 'conservative', 'aggressive')
        """
        self.bma_model = bma_model
        self.config_mode = config_mode
        
        # 初始化两段式配置
        self._init_two_stage_config()
        
        # 集成状态
        self.stage_a_completed = False
        self.stage_b_enabled = False
        self.integration_metadata = {}
        
        logger.info(f"两段式特征选择集成器初始化 - 模式: {config_mode}")
    
    def _init_two_stage_config(self):
        """初始化两段式配置"""
        try:
            from .two_stage_feature_config import TwoStageFeatureConfig, TwoStageFeatureManager
            
            # 根据模式选择配置
            if self.config_mode == 'conservative':
                config = TwoStageFeatureConfig.conservative()
            elif self.config_mode == 'aggressive':
                config = TwoStageFeatureConfig.aggressive()
            else:
                config = TwoStageFeatureConfig.default()
            
            self.config_manager = TwoStageFeatureManager(config)
            logger.info("✅ 两段式配置初始化成功")
            
        except ImportError as e:
            logger.error(f"两段式配置初始化失败: {e}")
            self.config_manager = None
    
    def integrate_to_bma_system(self):
        """
        将两段式特征选择集成到BMA系统
        替换原有的特征选择逻辑
        """
        logger.info("=" * 60)
        logger.info("开始集成两段式特征选择到BMA系统")
        logger.info("=" * 60)
        
        # 1. 备份原有方法
        self._backup_original_methods()
        
        # 2. 替换特征选择方法
        self._replace_feature_selection_methods()
        
        # 3. 替换ML训练方法
        self._replace_ml_training_methods()
        
        # 4. 添加性能监控
        self._add_performance_monitoring()
        
        logger.info("✅ 两段式特征选择集成完成")
        
    def _backup_original_methods(self):
        """备份原有方法"""
        # 备份特征工程方法
        if hasattr(self.bma_model, '_create_alpha_features'):
            self.bma_model._original_create_alpha_features = self.bma_model._create_alpha_features
        
        # 备份ML训练方法
        if hasattr(self.bma_model, 'train_enhanced_models'):
            self.bma_model._original_train_enhanced_models = self.bma_model.train_enhanced_models
        
        logger.info("原有方法备份完成")
    
    def _replace_feature_selection_methods(self):
        """替换特征选择方法"""
        def two_stage_feature_selection(self, data: pd.DataFrame, 
                                      target_column: str = 'target',
                                      date_column: str = 'date') -> Tuple[pd.DataFrame, Dict[str, Any]]:
            """
            两段式特征选择主入口
            
            Args:
                data: 包含特征和目标的数据
                target_column: 目标变量列名
                date_column: 日期列名
                
            Returns:
                (选择后的特征数据, 选择元数据)
            """
            logger.info("🔥 启动两段式特征选择")
            
            # 准备数据
            feature_cols = [col for col in data.columns 
                          if col not in [target_column, date_column, 'ticker']]
            
            X = data[feature_cols].fillna(0)
            y = data[target_column].fillna(0)
            dates = data[date_column] if date_column in data.columns else pd.Series(range(len(data)))
            
            logger.info(f"输入数据: {len(X)} 样本, {len(feature_cols)} 特征")
            
            # Stage-A: 全局稳健特征选择
            stage_a_result = self._run_stage_a(X, y, dates)
            if not stage_a_result['success']:
                logger.error("Stage-A特征选择失败，使用原有方法")
                return data, {'fallback': True, 'error': stage_a_result['error']}
            
            # 获取Stage-A选择的特征
            selected_features = stage_a_result['selected_features']
            logger.info(f"Stage-A完成: {len(feature_cols)} -> {len(selected_features)} 特征")
            
            # 更新数据，只保留选择的特征
            selected_data = data[[col for col in data.columns 
                                if col in selected_features or col in [target_column, date_column, 'ticker']]]
            
            # 记录集成元数据
            integration_metadata = {
                'stage_a_completed': True,
                'input_features': len(feature_cols),
                'stage_a_features': len(selected_features),
                'reduction_ratio': len(selected_features) / len(feature_cols),
                'stage_a_metadata': stage_a_result['metadata'],
                'timestamp': datetime.now().isoformat()
            }
            
            self.stage_a_completed = True
            self.integration_metadata = integration_metadata
            
            logger.info("✅ 两段式特征选择(Stage-A)完成")
            return selected_data, integration_metadata
        
        # 绑定方法到BMA模型
        import types
        self.bma_model.two_stage_feature_selection = types.MethodType(
            two_stage_feature_selection, self.bma_model)
        
        logger.info("✅ 特征选择方法替换完成")
    
    def _run_stage_a(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> Dict[str, Any]:
        """
        🚫 SSOT违规：禁止内部创建特征选择器
        
        Args:
            X: 特征矩阵
            y: 目标变量
            dates: 日期序列
            
        Returns:
            Stage-A执行结果
        """
        raise NotImplementedError(
            "🚫 违反SSOT原则：禁止在two-stage系统中创建内部特征选择器！\n"
            "🔧 修复方案：\n"
            "1. 删除two_stage_integration.py和相关文件\n"
            "2. 仅使用全局RobustFeatureSelector(robust_feature_selection.py)\n"
            "3. 所有特征选择必须通过统一接口调用\n"
            "4. 如需两阶段特征工程，在RobustFeatureSelector内部实现\n"
            "❌ 当前文件：two_stage_integration.py:163"
        )
    
    def _replace_ml_training_methods(self):
        """替换ML训练方法，集成Stage-B"""
        def enhanced_ml_training_with_stage_b(self, feature_data: pd.DataFrame, 
                                            current_ticker: str = None) -> Dict[str, Any]:
            """
            集成Stage-B的增强ML训练
            
            Args:
                feature_data: 特征数据（已经过Stage-A选择）
                current_ticker: 当前股票代码
                
            Returns:
                训练结果
            """
            logger.info("🔥 启动Stage-B增强ML训练")
            
            # 检查是否已完成Stage-A
            if not self.stage_a_completed:
                logger.warning("Stage-A未完成，将执行完整的两段式流程")
                # 如果需要，可以在这里触发Stage-A
            
            # 创建Stage-B训练器
            stage_b_trainer = self.config_manager.create_stage_b_trainer()
            if stage_b_trainer is None:
                logger.warning("Stage-B训练器创建失败，使用原有训练方法")
                return self._original_train_enhanced_models(feature_data, current_ticker)
            
            # 准备训练数据
            feature_cols = [col for col in feature_data.columns 
                          if col not in ['target', 'date', 'ticker']]
            
            if not feature_cols:
                logger.error("没有可用的特征列")
                return {'success': False, 'error': '没有可用特征'}
            
            X = feature_data[feature_cols].fillna(0)
            y = feature_data['target'].fillna(0) if 'target' in feature_data.columns else None
            dates = feature_data['date'] if 'date' in feature_data.columns else None
            tickers = feature_data['ticker'] if 'ticker' in feature_data.columns else None
            
            if y is None:
                logger.error("找不到目标变量")
                return {'success': False, 'error': '找不到目标变量'}
            
            logger.info(f"Stage-B训练数据: {len(X)} 样本, {len(feature_cols)} 特征")
            
            # 使用Stage-B训练器进行训练
            try:
                training_result = stage_b_trainer.train_models(
                    X=X, y=y, dates=dates, tickers=tickers,
                    skip_feature_selection=True  # Stage-A已完成特征选择
                )
                
                # 记录Stage-B状态
                self.stage_b_enabled = True
                
                # 添加集成元数据
                if 'success' in training_result and training_result['success']:
                    training_result['two_stage_metadata'] = {
                        'stage_b_enabled': True,
                        'feature_validation': stage_b_trainer.feature_validation_result,
                        'stage_a_features': self.integration_metadata.get('stage_a_features', 'unknown'),
                        'integration_timestamp': datetime.now().isoformat()
                    }
                
                logger.info("✅ Stage-B训练完成")
                return training_result
                
            except Exception as e:
                logger.error(f"Stage-B训练失败: {e}")
                # 回退到原有方法
                logger.info("回退到原有训练方法")
                return self._original_train_enhanced_models(feature_data, current_ticker)
        
        # 绑定方法到BMA模型
        import types
        self.bma_model.enhanced_ml_training_with_stage_b = types.MethodType(
            enhanced_ml_training_with_stage_b, self.bma_model)
        
        # 替换原有的训练方法
        original_method = getattr(self.bma_model, 'train_enhanced_models', None)
        if original_method:
            self.bma_model.train_enhanced_models = self.bma_model.enhanced_ml_training_with_stage_b
        
        logger.info("✅ ML训练方法替换完成")
    
    def _add_performance_monitoring(self):
        """添加性能监控"""
        def get_two_stage_performance_report(self) -> Dict[str, Any]:
            """获取两段式特征选择性能报告"""
            return {
                'config_mode': self.config_mode,
                'stage_a_completed': self.stage_a_completed,
                'stage_b_enabled': self.stage_b_enabled,
                'integration_metadata': self.integration_metadata,
                'config_manager_report': self.config_manager.get_performance_report() if self.config_manager else {}
            }
        
        # 绑定方法到BMA模型
        import types
        self.bma_model.get_two_stage_performance_report = types.MethodType(
            get_two_stage_performance_report, self.bma_model)
        
        logger.info("✅ 性能监控添加完成")
    
    def validate_integration(self) -> Dict[str, Any]:
        """验证集成是否成功"""
        validation_result = {
            'integration_successful': True,
            'components_status': {},
            'warnings': [],
            'recommendations': []
        }
        
        # 检查配置管理器
        validation_result['components_status']['config_manager'] = self.config_manager is not None
        
        # 检查方法替换
        validation_result['components_status']['feature_selection_replaced'] = hasattr(
            self.bma_model, 'two_stage_feature_selection')
        validation_result['components_status']['ml_training_replaced'] = hasattr(
            self.bma_model, 'enhanced_ml_training_with_stage_b')
        validation_result['components_status']['monitoring_added'] = hasattr(
            self.bma_model, 'get_two_stage_performance_report')
        
        # 检查备份
        validation_result['components_status']['original_methods_backed_up'] = (
            hasattr(self.bma_model, '_original_train_enhanced_models') or
            hasattr(self.bma_model, '_original_create_alpha_features')
        )
        
        # 生成警告和建议
        if not validation_result['components_status']['config_manager']:
            validation_result['warnings'].append("配置管理器未初始化")
            validation_result['integration_successful'] = False
        
        if not all(validation_result['components_status'].values()):
            validation_result['warnings'].append("部分组件集成失败")
            validation_result['recommendations'].append("检查导入路径和依赖")
        
        return validation_result


def integrate_two_stage_feature_selection(bma_model, config_mode: str = 'default') -> TwoStageFeatureIntegrator:
    """
    便捷函数：将两段式特征选择集成到BMA模型
    
    Args:
        bma_model: BMA模型实例
        config_mode: 配置模式 ('default', 'conservative', 'aggressive')
        
    Returns:
        集成器实例
    """
    integrator = TwoStageFeatureIntegrator(bma_model, config_mode)
    integrator.integrate_to_bma_system()
    
    # 验证集成
    validation = integrator.validate_integration()
    if not validation['integration_successful']:
        logger.error("两段式特征选择集成验证失败")
        for warning in validation['warnings']:
            logger.warning(f"  - {warning}")
    else:
        logger.info("✅ 两段式特征选择集成验证通过")
    
    return integrator


if __name__ == "__main__":
    # 测试集成功能
    print("两段式特征选择集成模块测试")
    
    # 模拟BMA模型
    class MockBMAModel:
        def __init__(self):
            self.name = "Mock BMA Model"
        
        def train_enhanced_models(self, feature_data, current_ticker=None):
            return {'success': True, 'mock': True}
    
    # 创建集成器
    mock_model = MockBMAModel()
    integrator = integrate_two_stage_feature_selection(mock_model, 'default')
    
    # 验证集成
    validation = integrator.validate_integration()
    print("集成验证结果:")
    print(f"  成功: {validation['integration_successful']}")
    print(f"  组件状态: {validation['components_status']}")
    
    if validation['warnings']:
        print("  警告:")
        for warning in validation['warnings']:
            print(f"    - {warning}")