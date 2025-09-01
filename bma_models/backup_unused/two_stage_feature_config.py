#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两段式特征选择配置管理
统一管理Stage-A和Stage-B的配置，避免冲突
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class StageAConfig:
    """Stage-A (全局稳健层) 配置"""
    # RobustFeatureSelector 参数
    target_features: int = 16
    ic_window: int = 126
    min_ic_mean: float = 0.01
    min_ic_ir: float = 0.3
    max_correlation: float = 0.6
    
    # 反窥视设置
    max_selection_rounds: int = 1
    lockdown_after_selection: bool = True
    
    # 输出设置
    save_to_registry: bool = True
    registry_path: str = "bma_models/feature_registry"

@dataclass
class StageBConfig:
    """Stage-B (模型内收缩层) 配置"""
    # 基本模式
    mode: str = 'trainer_shrinkage'  # 'global_only' | 'trainer_shrinkage'
    max_features_threshold: int = 32
    
    # 模型内收缩参数 (LightGBM)
    lightgbm_params: Dict = None
    
    # 模型内收缩参数 (Sklearn)
    sklearn_params: Dict = None
    
    # 超参数优化
    enable_hyperopt: bool = True
    hyperopt_trials: int = 50
    
    def __post_init__(self):
        if self.lightgbm_params is None:
            self.lightgbm_params = {
                'feature_fraction_range': [0.6, 1.0],
                'bagging_fraction_range': [0.7, 0.9],
                'lambda_l1_choices': [0.0, 0.1, 0.5],
                'lambda_l2_choices': [0.0, 0.1, 0.5],
                'min_child_samples_choices': [20, 30, 50]
            }
        
        if self.sklearn_params is None:
            self.sklearn_params = {
                'alpha_range': [0.01, 1.0],
                'l1_ratio_range': [0.1, 0.9],
                'max_features_choices': [0.6, 0.8, 1.0]
            }

@dataclass
class TwoStageFeatureConfig:
    """两段式特征选择总配置"""
    stage_a: StageAConfig
    stage_b: StageBConfig
    
    # 全局设置
    anti_snooping_enabled: bool = True
    cross_validation_unified: bool = True  # 是否使用统一的CV策略
    
    # 性能监控
    performance_tracking: bool = True
    ic_target: float = 0.05  # 目标IC
    
    @classmethod
    def default(cls) -> 'TwoStageFeatureConfig':
        """创建默认配置"""
        return cls(
            stage_a=StageAConfig(),
            stage_b=StageBConfig()
        )
    
    @classmethod
    def conservative(cls) -> 'TwoStageFeatureConfig':
        """创建保守配置（更严格的反窥视）"""
        stage_a = StageAConfig(
            target_features=12,  # 更少特征
            min_ic_mean=0.015,   # 更高IC要求
            min_ic_ir=0.4,       # 更高IR要求
            max_correlation=0.5  # 更严格去相关
        )
        
        stage_b = StageBConfig(
            mode='global_only',  # 完全禁用Stage-B裁剪
            enable_hyperopt=False
        )
        
        return cls(
            stage_a=stage_a,
            stage_b=stage_b,
            anti_snooping_enabled=True
        )
    
    @classmethod
    def aggressive(cls) -> 'TwoStageFeatureConfig':
        """创建激进配置（允许更多特征工程）"""
        stage_a = StageAConfig(
            target_features=24,  # 更多特征
            min_ic_mean=0.005,   # 更低IC要求
            min_ic_ir=0.2,       # 更低IR要求
            max_correlation=0.7  # 更宽松去相关
        )
        
        stage_b = StageBConfig(
            mode='trainer_shrinkage',
            max_features_threshold=40,
            enable_hyperopt=True
        )
        
        return cls(
            stage_a=stage_a,
            stage_b=stage_b,
            anti_snooping_enabled=True
        )
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'stage_a': {
                'target_features': self.stage_a.target_features,
                'ic_window': self.stage_a.ic_window,
                'min_ic_mean': self.stage_a.min_ic_mean,
                'min_ic_ir': self.stage_a.min_ic_ir,
                'max_correlation': self.stage_a.max_correlation,
                'max_selection_rounds': self.stage_a.max_selection_rounds,
                'lockdown_after_selection': self.stage_a.lockdown_after_selection,
                'save_to_registry': self.stage_a.save_to_registry,
                'registry_path': self.stage_a.registry_path
            },
            'stage_b': {
                'mode': self.stage_b.mode,
                'max_features_threshold': self.stage_b.max_features_threshold,
                'lightgbm_params': self.stage_b.lightgbm_params,
                'sklearn_params': self.stage_b.sklearn_params,
                'enable_hyperopt': self.stage_b.enable_hyperopt,
                'hyperopt_trials': self.stage_b.hyperopt_trials
            },
            'global': {
                'anti_snooping_enabled': self.anti_snooping_enabled,
                'cross_validation_unified': self.cross_validation_unified,
                'performance_tracking': self.performance_tracking,
                'ic_target': self.ic_target
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TwoStageFeatureConfig':
        """从字典创建配置"""
        stage_a = StageAConfig(**config_dict['stage_a'])
        stage_b = StageBConfig(**config_dict['stage_b'])
        
        return cls(
            stage_a=stage_a,
            stage_b=stage_b,
            anti_snooping_enabled=config_dict['global']['anti_snooping_enabled'],
            cross_validation_unified=config_dict['global']['cross_validation_unified'],
            performance_tracking=config_dict['global']['performance_tracking'],
            ic_target=config_dict['global']['ic_target']
        )
    
    def save(self, filepath: str):
        """保存配置到文件"""
        config_dict = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"两段式特征选择配置已保存: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TwoStageFeatureConfig':
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def validate(self) -> List[str]:
        """验证配置合理性"""
        warnings = []
        
        # Stage-A验证
        if self.stage_a.target_features > 50:
            warnings.append("Stage-A target_features过多，可能导致过拟合")
        
        if self.stage_a.min_ic_mean < 0.001:
            warnings.append("Stage-A min_ic_mean过低，可能选择无效特征")
        
        if self.stage_a.ic_window < 60:
            warnings.append("Stage-A ic_window过短，IC统计可能不稳定")
        
        # Stage-B验证
        if self.stage_b.mode == 'trainer_shrinkage' and self.stage_b.max_features_threshold < 10:
            warnings.append("Stage-B max_features_threshold过小")
        
        # 逻辑一致性验证
        if self.stage_b.max_features_threshold > self.stage_a.target_features:
            if self.stage_b.mode == 'trainer_shrinkage':
                warnings.append("Stage-B阈值大于Stage-A目标数，裁剪逻辑可能失效")
        
        # 反窥视验证
        if not self.anti_snooping_enabled:
            warnings.append("反窥视保护已禁用，存在数据泄露风险")
        
        return warnings


class TwoStageFeatureManager:
    """两段式特征选择管理器"""
    
    def __init__(self, config: Optional[TwoStageFeatureConfig] = None):
        """
        初始化管理器
        
        Args:
            config: 两段式配置，如果为None则使用默认配置
        """
        self.config = config or TwoStageFeatureConfig.default()
        self.stage_a_selector = None
        self.stage_b_trainer = None
        
        # 验证配置
        warnings = self.config.validate()
        for warning in warnings:
            logger.warning(f"配置警告: {warning}")
    
    def get_stage_a_config(self) -> Dict:
        """获取Stage-A配置（用于RobustFeatureSelector）"""
        return {
            'target_features': self.config.stage_a.target_features,
            'ic_window': self.config.stage_a.ic_window,
            'min_ic_mean': self.config.stage_a.min_ic_mean,
            'min_ic_ir': self.config.stage_a.min_ic_ir,
            'max_correlation': self.config.stage_a.max_correlation
        }
    
    def get_stage_b_config(self) -> Dict:
        """获取Stage-B配置（用于EnhancedMLTrainer）"""
        return {
            'stage_b_mode': self.config.stage_b.mode,
            'max_features_threshold': self.config.stage_b.max_features_threshold,
            'enable_hyperparam_opt': self.config.stage_b.enable_hyperopt,
            'lightgbm_params': self.config.stage_b.lightgbm_params,
            'sklearn_params': self.config.stage_b.sklearn_params
        }
    
    def create_stage_a_selector(self):
        """🚫 SSOT违规：禁止创建内部特征选择器"""
        raise NotImplementedError(
            "🚫 违反SSOT原则：禁止在two_stage_feature_config中创建内部RobustFeatureSelector！\n"
            "🔧 修复方案：\n"
            "1. 删除所有two_stage_*文件\n"
            "2. 仅使用全局单例RobustFeatureSelector\n"
            "3. 配置通过robust_feature_selection.py统一管理\n"
            "4. 禁止模块间重复创建特征选择器\n"
            "❌ 当前文件：two_stage_feature_config.py:261"
        )
    
    def create_stage_b_trainer(self):
        """创建Stage-B训练器"""
        try:
            from .enhanced_ml_trainer import EnhancedMLTrainer
            
            stage_b_config = self.get_stage_b_config()
            self.stage_b_trainer = EnhancedMLTrainer(**stage_b_config)
            
            logger.info("✅ Stage-B训练器创建成功")
            return self.stage_b_trainer
            
        except ImportError as e:
            logger.error(f"无法导入EnhancedMLTrainer: {e}")
            return None
    
    def get_performance_report(self) -> Dict:
        """获取两段式特征选择性能报告"""
        report = {
            'config_summary': {
                'stage_a_features': self.config.stage_a.target_features,
                'stage_b_mode': self.config.stage_b.mode,
                'anti_snooping': self.config.anti_snooping_enabled
            },
            'stage_a_status': 'not_initialized',
            'stage_b_status': 'not_initialized'
        }
        
        if self.stage_a_selector:
            report['stage_a_status'] = 'initialized'
            if hasattr(self.stage_a_selector, 'selected_features_'):
                report['stage_a_results'] = {
                    'selected_count': len(self.stage_a_selector.selected_features_),
                    'selected_features': self.stage_a_selector.selected_features_
                }
        
        if self.stage_b_trainer:
            report['stage_b_status'] = 'initialized'
            if hasattr(self.stage_b_trainer, 'feature_validation_result'):
                report['stage_b_results'] = self.stage_b_trainer.feature_validation_result
        
        return report


# 全局配置实例
_global_config_manager = None

def get_two_stage_config() -> TwoStageFeatureManager:
    """获取全局两段式配置管理器"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = TwoStageFeatureManager()
    return _global_config_manager

def set_two_stage_config(config: TwoStageFeatureConfig):
    """设置全局两段式配置"""
    global _global_config_manager
    _global_config_manager = TwoStageFeatureManager(config)


if __name__ == "__main__":
    # 测试配置系统
    
    # 1. 默认配置
    default_config = TwoStageFeatureConfig.default()
    print("默认配置:")
    print(json.dumps(default_config.to_dict(), indent=2, ensure_ascii=False))
    
    # 2. 保守配置
    conservative_config = TwoStageFeatureConfig.conservative()
    print("\n保守配置:")
    print(json.dumps(conservative_config.to_dict(), indent=2, ensure_ascii=False))
    
    # 3. 激进配置
    aggressive_config = TwoStageFeatureConfig.aggressive()
    print("\n激进配置:")
    print(json.dumps(aggressive_config.to_dict(), indent=2, ensure_ascii=False))
    
    # 4. 配置验证
    warnings = default_config.validate()
    print(f"\n配置验证警告: {len(warnings)}")
    for warning in warnings:
        print(f"  - {warning}")
    
    # 5. 配置管理器
    manager = TwoStageFeatureManager(default_config)
    report = manager.get_performance_report()
    print("\n性能报告:")
    print(json.dumps(report, indent=2, ensure_ascii=False))