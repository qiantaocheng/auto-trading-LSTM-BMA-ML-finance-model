#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
样本权重统一化模块 - 确保所有组件使用一致的样本权重半衰期
防止不同模块使用不同权重衰减导致的不一致性
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta

try:
    from unified_timing_registry import get_global_timing_registry, TimingEnforcer
except ImportError:
    get_global_timing_registry = lambda: None
    TimingEnforcer = None

logger = logging.getLogger(__name__)


class SampleWeightUnificator:
    """
    样本权重统一化器
    
    确保所有模块使用相同的样本权重半衰期参数，
    防止训练过程中权重不一致导致的模型偏差
    """
    
    def __init__(self):
        self.timing_registry = get_global_timing_registry()
        self.canonical_half_life = self.timing_registry.sample_weight_half_life
        self.enforced_modules = []
        
        logger.info("样本权重统一化器已初始化")
        logger.info(f"标准半衰期: {self.canonical_half_life}天")
    
    def enforce_weight_consistency(self, module_name: str, proposed_half_life: int) -> int:
        """
        强制权重半衰期一致性
        
        Args:
            module_name: 模块名称
            proposed_half_life: 模块提议的半衰期
            
        Returns:
            强制统一后的半衰期
        """
        return TimingEnforcer.enforce_sample_weight_consistency(module_name, proposed_half_life)
    
    def create_unified_sample_weights(self, 
                                    dates: pd.DatetimeIndex, 
                                    reference_date: Optional[datetime] = None,
                                    weight_type: str = 'exponential') -> np.ndarray:
        """
        创建统一的样本权重
        
        Args:
            dates: 样本日期索引
            reference_date: 参考日期（最新日期），默认使用最后一个日期
            weight_type: 权重类型 ('exponential', 'linear')
            
        Returns:
            样本权重数组
        """
        if reference_date is None:
            reference_date = dates.max()
        
        # 计算时间差（天数）
        time_diffs = (reference_date - dates).days
        
        if weight_type == 'exponential':
            # 指数衰减权重: w = exp(-ln(2) * t / half_life)
            weights = np.exp(-np.log(2) * time_diffs / self.canonical_half_life)
        elif weight_type == 'linear':
            # 线性衰减权重: w = max(0, 1 - t / (2 * half_life))
            weights = np.maximum(0, 1 - time_diffs / (2 * self.canonical_half_life))
        else:
            raise ValueError(f"不支持的权重类型: {weight_type}")
        
        # 归一化权重
        weights = weights / weights.sum()
        
        logger.info(f"创建统一样本权重: {len(weights)}个样本，半衰期{self.canonical_half_life}天")
        logger.info(f"权重范围: [{weights.min():.6f}, {weights.max():.6f}]")
        
        return weights
    
    def validate_module_weights(self, 
                               module_name: str,
                               module_weights: np.ndarray,
                               dates: pd.DatetimeIndex,
                               tolerance: float = 0.1) -> bool:
        """
        验证模块权重是否与标准权重一致
        
        Args:
            module_name: 模块名称
            module_weights: 模块计算的权重
            dates: 对应的日期索引
            tolerance: 允许的相对误差
            
        Returns:
            是否通过验证
        """
        # 创建标准权重
        standard_weights = self.create_unified_sample_weights(dates)
        
        # 检查长度一致性
        if len(module_weights) != len(standard_weights):
            logger.error(f"❌ {module_name} 权重长度不一致: {len(module_weights)} vs {len(standard_weights)}")
            return False
        
        # 计算相对误差
        rel_errors = np.abs(module_weights - standard_weights) / (standard_weights + 1e-10)
        max_error = rel_errors.max()
        mean_error = rel_errors.mean()
        
        if max_error > tolerance:
            logger.error(f"❌ {module_name} 权重不一致:")
            logger.error(f"  最大相对误差: {max_error:.4f} > {tolerance:.4f}")
            logger.error(f"  平均相对误差: {mean_error:.4f}")
            
            # 找到最大误差的位置
            max_error_idx = np.argmax(rel_errors)
            logger.error(f"  最大误差位置: {dates[max_error_idx]}")
            logger.error(f"  模块权重: {module_weights[max_error_idx]:.6f}")
            logger.error(f"  标准权重: {standard_weights[max_error_idx]:.6f}")
            
            return False
        else:
            logger.info(f"✅ {module_name} 权重验证通过")
            logger.info(f"  最大相对误差: {max_error:.4f}")
            logger.info(f"  平均相对误差: {mean_error:.4f}")
            return True
    
    def fix_module_weight_config(self, config: Dict[str, Any], module_name: str) -> Dict[str, Any]:
        """
        修复模块配置中的权重参数
        
        Args:
            config: 原始配置
            module_name: 模块名称
            
        Returns:
            修复后的配置
        """
        logger.info(f"修复 {module_name} 模块的权重配置")
        
        fixed_config = config.copy()
        
        # 权重半衰期相关的键名变体
        half_life_keys = [
            'sample_weight_half_life', 'half_life_days', 'weight_half_life',
            'decay_half_life', 'sample_decay_days', 'weight_decay_half_life'
        ]
        
        violations_found = []
        for key in half_life_keys:
            if key in fixed_config:
                original_value = fixed_config[key]
                if original_value != self.canonical_half_life:
                    fixed_config[key] = self.canonical_half_life
                    violations_found.append(f"{key}: {original_value} -> {self.canonical_half_life}")
        
        # 权重衰减类型统一
        decay_type_keys = ['decay_type', 'weight_type', 'sample_weight_type']
        for key in decay_type_keys:
            if key in fixed_config:
                original_value = fixed_config[key]
                if original_value != 'exponential':
                    fixed_config[key] = 'exponential'
                    violations_found.append(f"{key}: {original_value} -> exponential")
        
        # 权重相关的其他参数
        other_weight_keys = {
            'weight_min_threshold': 1e-6,  # 最小权重阈值
            'weight_normalize': True,      # 归一化权重
            'weight_clip_quantile': 0.01   # 权重截断分位数
        }
        
        for key, standard_value in other_weight_keys.items():
            if key in fixed_config and fixed_config[key] != standard_value:
                original_value = fixed_config[key]
                fixed_config[key] = standard_value
                violations_found.append(f"{key}: {original_value} -> {standard_value}")
        
        if violations_found:
            logger.warning(f"⚠️ {module_name} 发现并修复权重配置违规:")
            for violation in violations_found:
                logger.warning(f"  修复: {violation}")
        
        # 记录强制执行结果
        self.enforced_modules.append({
            'module_name': module_name,
            'violations_count': len(violations_found),
            'violations': violations_found
        })
        
        logger.info(f"✅ {module_name} 权重配置已统一")
        return fixed_config
    
    def create_weight_decay_function(self) -> callable:
        """
        创建标准的权重衰减函数
        
        Returns:
            权重衰减函数
        """
        def weight_decay_func(days_back: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
            """
            标准权重衰减函数
            
            Args:
                days_back: 距离参考日期的天数
                
            Returns:
                权重值
            """
            return np.exp(-np.log(2) * days_back / self.canonical_half_life)
        
        logger.info(f"创建标准权重衰减函数，半衰期 {self.canonical_half_life}天")
        return weight_decay_func
    
    def get_weight_params(self) -> Dict[str, Any]:
        """获取统一的权重参数"""
        return self.timing_registry.get_sample_weight_params()
    
    def get_unification_summary(self) -> Dict[str, Any]:
        """获取统一化摘要"""
        total_modules = len(self.enforced_modules)
        total_violations = sum(module['violations_count'] for module in self.enforced_modules)
        
        return {
            'canonical_half_life': self.canonical_half_life,
            'total_modules_unified': total_modules,
            'total_violations_fixed': total_violations,
            'unification_details': self.enforced_modules,
            'weight_params': self.get_weight_params()
        }


def unify_sample_weights_globally(module_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    全局统一所有模块的样本权重配置
    
    Args:
        module_configs: 模块配置字典 {module_name: config}
        
    Returns:
        统一后的配置字典
    """
    unifier = SampleWeightUnificator()
    unified_configs = {}
    
    logger.info("开始全局样本权重统一化")
    
    for module_name, config in module_configs.items():
        unified_configs[module_name] = unifier.fix_module_weight_config(config, module_name)
    
    # 记录总结
    summary = unifier.get_unification_summary()
    logger.info("=== 全局样本权重统一化完成 ===")
    logger.info(f"标准半衰期: {summary['canonical_half_life']}天")
    logger.info(f"统一模块数: {summary['total_modules_unified']}")
    logger.info(f"修复违规配置数: {summary['total_violations_fixed']}")
    
    return unified_configs


def create_weight_validation_report(dates: pd.DatetimeIndex, 
                                   module_weights_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    创建权重验证报告
    
    Args:
        dates: 日期索引
        module_weights_dict: 各模块的权重 {module_name: weights}
        
    Returns:
        权重验证报告DataFrame
    """
    unifier = SampleWeightUnificator()
    standard_weights = unifier.create_unified_sample_weights(dates)
    
    report_data = []
    
    for module_name, module_weights in module_weights_dict.items():
        # 计算统计指标
        if len(module_weights) == len(standard_weights):
            rel_errors = np.abs(module_weights - standard_weights) / (standard_weights + 1e-10)
            max_error = rel_errors.max()
            mean_error = rel_errors.mean()
            
            # 计算相关性
            correlation = np.corrcoef(module_weights, standard_weights)[0, 1]
            
            # 检查是否通过验证
            passed = max_error <= 0.1
            
            report_data.append({
                'module_name': module_name,
                'max_relative_error': max_error,
                'mean_relative_error': mean_error,
                'correlation_with_standard': correlation,
                'validation_passed': passed,
                'weight_sum': module_weights.sum(),
                'weight_range': f"[{module_weights.min():.6f}, {module_weights.max():.6f}]"
            })
        else:
            report_data.append({
                'module_name': module_name,
                'max_relative_error': np.nan,
                'mean_relative_error': np.nan,
                'correlation_with_standard': np.nan,
                'validation_passed': False,
                'weight_sum': module_weights.sum(),
                'weight_range': f"长度不匹配: {len(module_weights)} vs {len(standard_weights)}"
            })
    
    return pd.DataFrame(report_data)


if __name__ == "__main__":
    # 测试样本权重统一化
    unifier = SampleWeightUnificator()
    
    # 创建测试日期
    test_dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    
    # 创建统一权重
    unified_weights = unifier.create_unified_sample_weights(test_dates)
    print(f"统一权重: 长度{len(unified_weights)}, 和{unified_weights.sum():.6f}")
    
    # 测试配置修复
    test_config = {
        'sample_weight_half_life': 60,  # 应被修正为75
        'decay_type': 'linear',         # 应被修正为exponential
        'other_param': 'value'          # 应保持不变
    }
    
    fixed_config = unifier.fix_module_weight_config(test_config, "test_module")
    print("修复后配置:", fixed_config)
    
    # 获取统一化摘要
    summary = unifier.get_unification_summary()
    print("统一化摘要:", summary)