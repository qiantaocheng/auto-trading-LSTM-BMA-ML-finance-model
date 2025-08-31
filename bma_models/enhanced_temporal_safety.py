#!/usr/bin/env python3
"""
增强时间安全验证器
防止时间序列数据泄露，强化隔离机制
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class TemporalSafetyConfig:
    """时间安全配置"""
    min_absolute_gap_days: int = 12  # 绝对最小间隔，无论数据大小
    standard_gap_days: int = 10      # 标准间隔
    min_train_size: int = 100        # 最小训练样本数
    strict_mode: bool = True         # 严格模式，禁用所有适应性减少
    emergency_min_gap: int = 5       # 紧急情况下的最小间隔
    
class EnhancedTemporalSafety:
    """增强的时间安全验证器"""
    
    def __init__(self, config: TemporalSafetyConfig = None):
        self.config = config or TemporalSafetyConfig()
        logger.info(f"初始化增强时间安全验证器 - 最小绝对间隔: {self.config.min_absolute_gap_days}天")
    
    def validate_temporal_split(self, train_dates: List, test_dates: List, 
                              dataset_name: str = "dataset") -> Dict:
        """
        验证时间分割的安全性
        
        Args:
            train_dates: 训练集日期列表
            test_dates: 测试集日期列表
            dataset_name: 数据集名称(用于日志)
            
        Returns:
            验证结果字典
        """
        try:
            # 检查空数据
            if train_dates is None or test_dates is None:
                return {
                    'is_safe': False,
                    'error': '训练或测试日期为空',
                    'actual_gap': 0,
                    'required_gap': self.config.min_absolute_gap_days
                }
            
            # 转换为pandas Series以避免DatetimeIndex模糊性
            train_dates = pd.to_datetime(pd.Series(train_dates))
            test_dates = pd.to_datetime(pd.Series(test_dates))
            
            # 检查是否为空
            if len(train_dates) == 0 or len(test_dates) == 0:
                return {
                    'is_safe': False,
                    'error': '训练或测试日期序列为空',
                    'actual_gap': 0,
                    'required_gap': self.config.min_absolute_gap_days
                }
            
            max_train_date = train_dates.max()
            min_test_date = test_dates.min()
            
            actual_gap_days = (min_test_date - max_train_date).days
            
            # 获取所需的最小间隔
            required_gap = self._get_required_gap(len(train_dates), len(test_dates))
            
            is_safe = actual_gap_days >= required_gap
            
            result = {
                'is_safe': is_safe,
                'actual_gap': actual_gap_days,
                'required_gap': required_gap,
                'max_train_date': max_train_date,
                'min_test_date': min_test_date,
                'train_size': len(train_dates),
                'test_size': len(test_dates),
                'dataset_name': dataset_name
            }
            
            if is_safe:
                logger.info(f"✅ {dataset_name} 时间安全验证通过: "
                           f"间隔{actual_gap_days}天 >= 要求{required_gap}天")
            else:
                logger.error(f"❌ {dataset_name} 时间安全验证失败: "
                           f"间隔{actual_gap_days}天 < 要求{required_gap}天")
                result['error'] = f"时间间隔不足: {actual_gap_days} < {required_gap}天"
            
            return result
            
        except Exception as e:
            logger.error(f"时间安全验证异常: {e}")
            return {
                'is_safe': False,
                'error': f"验证异常: {str(e)}",
                'actual_gap': 0,
                'required_gap': self.config.min_absolute_gap_days
            }
    
    def _get_required_gap(self, train_size: int, test_size: int) -> int:
        """
        获取所需的最小时间间隔
        
        严格模式下不允许适应性减少
        """
        if self.config.strict_mode:
            # 严格模式：始终使用绝对最小间隔
            logger.debug(f"严格模式: 使用绝对最小间隔 {self.config.min_absolute_gap_days}天")
            return self.config.min_absolute_gap_days
        
        # 非严格模式下的适应性逻辑(仅用于开发测试)
        total_samples = train_size + test_size
        
        if total_samples >= 1000:
            return self.config.min_absolute_gap_days
        elif total_samples >= 500:
            # 轻微减少但不低于标准间隔
            return max(self.config.standard_gap_days, self.config.min_absolute_gap_days - 2)
        elif total_samples >= 200:
            # 显著减少但不低于紧急最小值
            return max(self.config.emergency_min_gap, self.config.min_absolute_gap_days - 5)
        else:
            # 极小数据集仍保持紧急最小间隔
            logger.warning(f"数据集过小({total_samples}样本)，使用紧急最小间隔")
            return self.config.emergency_min_gap
    
    def fix_cv_split_safety(self, cv_splitter, X, y, groups):
        """
        修复CV分割的安全性问题
        为现有CV分割器添加时间安全检查
        """
        safe_splits = []
        total_splits = 0
        rejected_splits = 0
        
        for train_idx, test_idx in cv_splitter.split(X, y, groups):
            total_splits += 1
            
            # 获取时间信息
            if hasattr(groups, 'iloc'):
                train_times = groups.iloc[train_idx]
                test_times = groups.iloc[test_idx]
            else:
                train_times = groups[train_idx] if hasattr(groups, '__getitem__') else []
                test_times = groups[test_idx] if hasattr(groups, '__getitem__') else []
            
            # 验证时间安全性
            validation_result = self.validate_temporal_split(
                train_times, test_times, f"CV fold {total_splits}"
            )
            
            if validation_result['is_safe']:
                safe_splits.append((train_idx, test_idx))
            else:
                rejected_splits += 1
                logger.warning(f"拒绝不安全的CV分割 #{total_splits}: "
                             f"{validation_result.get('error', 'Unknown error')}")
        
        logger.info(f"CV分割安全检查完成: {len(safe_splits)}/{total_splits} 通过, "
                   f"{rejected_splits} 被拒绝")
        
        if not safe_splits:
            logger.error("❌ 所有CV分割都不安全，无法进行交叉验证")
            raise ValueError("无安全的CV分割可用，考虑增加数据量或调整参数")
        
        return safe_splits

# 增强现有CV分割器
def enhance_cv_splitter_safety(cv_splitter, safety_config: TemporalSafetyConfig = None):
    """
    为现有CV分割器添加时间安全增强
    """
    safety_validator = EnhancedTemporalSafety(safety_config)
    
    class SafeEnhancedCV:
        def __init__(self, original_splitter):
            self.original_splitter = original_splitter
            self.safety_validator = safety_validator
        
        def split(self, X, y=None, groups=None):
            """安全的分割方法"""
            if groups is None:
                logger.error("groups参数是时间安全验证的必需参数")
                raise ValueError("时间安全验证需要groups参数")
            
            return self.safety_validator.fix_cv_split_safety(
                self.original_splitter, X, y, groups
            )
        
        def __getattr__(self, name):
            # 代理其他方法到原始分割器
            return getattr(self.original_splitter, name)
    
    return SafeEnhancedCV(cv_splitter)