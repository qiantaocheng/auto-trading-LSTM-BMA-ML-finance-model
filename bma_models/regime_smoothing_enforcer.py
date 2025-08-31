#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regime平滑强制禁用器 - 确保所有组件严格禁用平滑功能
防止未来泄露和时序污染
"""

import logging
from typing import Dict, Any, List
from unified_timing_registry import get_global_timing_registry, TimingEnforcer

logger = logging.getLogger(__name__)


class RegimeSmoothingEnforcer:
    """
    Regime平滑强制禁用器
    
    确保系统中所有涉及Regime的组件都严格禁用任何形式的平滑、滤波或前瞻操作
    """
    
    def __init__(self):
        self.timing_registry = get_global_timing_registry()
        self.enforced_modules = []
        
        logger.info("Regime平滑强制禁用器已初始化")
        logger.info("强制策略: 禁用所有平滑、滤波、前瞻操作")
    
    def enforce_no_smoothing_config(self, config: Dict[str, Any], module_name: str = "unknown") -> Dict[str, Any]:
        """
        强制禁用配置中的所有平滑选项
        
        Args:
            config: 原始配置字典
            module_name: 模块名称（用于记录）
            
        Returns:
            强制禁用平滑后的配置
        """
        logger.info(f"强制禁用 {module_name} 模块的Regime平滑功能")
        
        # 使用TimingEnforcer强制执行
        enforced_config = TimingEnforcer.enforce_regime_no_smoothing(config)
        
        # 额外的安全检查和强制设置
        dangerous_keys = [
            'enable_smoothing', 'prob_smooth_window', 'smooth_transitions',
            'rolling_window', 'ma_window', 'lookforward', 'future_info',
            'smooth_probabilities', 'filter_transitions', 'regime_filtering',
            'transition_smoothing', 'probability_smoothing', 'kalman_filter',
            'ema_smoothing', 'sma_smoothing', 'gaussian_filter'
        ]
        
        violations_found = []
        for key in dangerous_keys:
            if key in enforced_config:
                original_value = enforced_config[key]
                
                if key == 'enable_smoothing':
                    enforced_config[key] = False
                elif isinstance(original_value, bool):
                    enforced_config[key] = False
                elif isinstance(original_value, (int, float)):
                    enforced_config[key] = 0
                elif isinstance(original_value, str):
                    if 'smooth' in original_value.lower() or 'filter' in original_value.lower():
                        enforced_config[key] = 'none'
                
                if original_value != enforced_config[key]:
                    violations_found.append(f"{key}: {original_value} -> {enforced_config[key]}")
        
        if violations_found:
            logger.warning(f"⚠️ {module_name} 发现并修复平滑配置违规:")
            for violation in violations_found:
                logger.warning(f"  修复: {violation}")
        
        # 记录强制执行结果
        self.enforced_modules.append({
            'module_name': module_name,
            'violations_count': len(violations_found),
            'violations': violations_found
        })
        
        logger.info(f"✅ {module_name} Regime平滑已强制禁用")
        return enforced_config
    
    def validate_regime_model_config(self, regime_model: Any, model_name: str = "regime_model") -> bool:
        """
        验证Regime模型配置是否符合禁平滑要求
        
        Args:
            regime_model: Regime模型实例
            model_name: 模型名称
            
        Returns:
            是否通过验证
        """
        logger.info(f"验证 {model_name} 的平滑禁用状态")
        
        validation_errors = []
        
        # 检查常见的平滑相关属性
        smoothing_attributes = [
            'enable_smoothing', 'smooth_probabilities', 'prob_smooth_window',
            'transition_smoothing', 'filter_probabilities'
        ]
        
        for attr in smoothing_attributes:
            if hasattr(regime_model, attr):
                value = getattr(regime_model, attr)
                if attr == 'enable_smoothing' and value is True:
                    validation_errors.append(f"{attr} = {value} (应为False)")
                elif attr in ['prob_smooth_window'] and isinstance(value, (int, float)) and value > 0:
                    validation_errors.append(f"{attr} = {value} (应为0)")
                elif isinstance(value, bool) and value is True and 'smooth' in attr:
                    validation_errors.append(f"{attr} = {value} (应为False)")
        
        # 检查方法存在性（确保不使用平滑方法）
        dangerous_methods = ['smooth_probabilities', 'filter_transitions', 'apply_smoothing']
        for method in dangerous_methods:
            if hasattr(regime_model, method):
                validation_errors.append(f"发现危险方法: {method}")
        
        if validation_errors:
            logger.error(f"❌ {model_name} 平滑验证失败:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            return False
        else:
            logger.info(f"✅ {model_name} 平滑验证通过")
            return True
    
    def get_enforcement_summary(self) -> Dict[str, Any]:
        """获取强制执行摘要"""
        total_modules = len(self.enforced_modules)
        total_violations = sum(module['violations_count'] for module in self.enforced_modules)
        
        return {
            'total_modules_enforced': total_modules,
            'total_violations_fixed': total_violations,
            'enforcement_details': self.enforced_modules,
            'timing_registry_status': {
                'regime_enable_smoothing': self.timing_registry.regime_enable_smoothing,
                'regime_prob_smooth_window': self.timing_registry.regime_prob_smooth_window,
                'regime_transition_buffer': self.timing_registry.regime_transition_buffer
            }
        }
    
    @staticmethod
    def create_safe_regime_config() -> Dict[str, Any]:
        """
        创建安全的Regime配置模板
        
        Returns:
            完全禁用平滑的Regime配置
        """
        registry = get_global_timing_registry()
        
        safe_config = {
            # 核心禁平滑设置
            'enable_smoothing': False,
            'prob_smooth_window': 0,
            'smooth_transitions': False,
            'filter_probabilities': False,
            'transition_smoothing': False,
            
            # 其他安全设置
            'transition_buffer': registry.regime_transition_buffer,
            'use_filtering_only': True,  # 仅允许后验过滤
            'lookforward_days': 0,  # 禁止前瞻
            'future_information': False,  # 禁止使用未来信息
            
            # 滤波器相关（全部禁用）
            'kalman_filter': False,
            'ema_smoothing': False,
            'sma_smoothing': False,
            'gaussian_filter': False,
            'rolling_window': 0,
            'ma_window': 0,
            
            # 时序安全
            'min_regime_duration': 5,  # 最短状态持续期
            'regime_change_threshold': 0.7  # 状态切换阈值
        }
        
        logger.info("创建了安全的Regime配置模板")
        return safe_config


def enforce_regime_no_smoothing_globally(configs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    全局强制禁用所有模块的Regime平滑
    
    Args:
        configs: 模块配置字典 {module_name: config}
        
    Returns:
        强制禁用平滑后的配置字典
    """
    enforcer = RegimeSmoothingEnforcer()
    enforced_configs = {}
    
    logger.info("开始全局Regime平滑禁用强制执行")
    
    for module_name, config in configs.items():
        enforced_configs[module_name] = enforcer.enforce_no_smoothing_config(config, module_name)
    
    # 记录总结
    summary = enforcer.get_enforcement_summary()
    logger.info("=== 全局Regime平滑禁用强制执行完成 ===")
    logger.info(f"强制执行模块数: {summary['total_modules_enforced']}")
    logger.info(f"修复违规配置数: {summary['total_violations_fixed']}")
    
    return enforced_configs


if __name__ == "__main__":
    # 测试Regime平滑强制禁用器
    enforcer = RegimeSmoothingEnforcer()
    
    # 测试配置强制
    test_config = {
        'enable_smoothing': True,  # 应被强制为False
        'prob_smooth_window': 5,   # 应被强制为0
        'other_param': 'value'     # 应保持不变
    }
    
    enforced = enforcer.enforce_no_smoothing_config(test_config, "test_module")
    print("强制后配置:", enforced)
    
    # 测试安全配置创建
    safe_config = RegimeSmoothingEnforcer.create_safe_regime_config()
    print("安全配置模板:", safe_config)
    
    # 获取执行摘要
    summary = enforcer.get_enforcement_summary()
    print("执行摘要:", summary)