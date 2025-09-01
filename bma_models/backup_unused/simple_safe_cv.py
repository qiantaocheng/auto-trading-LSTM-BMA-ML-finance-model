#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMA Enhanced - 安全时序交叉验证系统
解决预测性能问题：修复V6 purged CV失败，提供稳定的时序CV策略
专为BMA Enhanced系统设计，兼容sklearn的cross_val_score
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
import logging

logger = logging.getLogger(__name__)


class SimpleSafeTimeSeriesSplit(BaseCrossValidator):
    """
    简单安全的时间序列分割器
    
    特点：
    1. 兼容sklearn的cross_val_score
    2. 不需要groups参数  
    3. 有适当的gap防止数据泄露
    4. 简单可靠的实现
    """
    
    def __init__(self, n_splits=5, gap_days=11, test_size=0.2):  # CRITICAL FIX: gap_days=11
        self.n_splits = n_splits
        self.gap_days = gap_days
        self.test_size = test_size
        logger.info(f"SimpleSafeTimeSeriesSplit初始化: n_splits={n_splits}, gap={gap_days}天")
    
    def split(self, X, y=None, groups=None):
        """
        生成安全的时间序列分割
        
        Args:
            X: 特征数据
            y: 目标变量（可选）
            groups: 分组（忽略，保持兼容性）
            
        Yields:
            (train_indices, test_indices): 训练和测试索引
        """
        n_samples = len(X)
        test_size = int(n_samples * self.test_size)
        
        # 确保有足够的数据进行分割
        if n_samples < (self.n_splits + 1) * test_size:
            logger.warning(f"数据量{n_samples}不足，减少分割数")
            effective_splits = max(1, n_samples // (test_size * 2))
            self.n_splits = min(self.n_splits, effective_splits)
        
        for i in range(self.n_splits):
            # 计算测试集的结束位置
            test_end = n_samples - i * (test_size // 2)  # 重叠的测试集
            test_start = test_end - test_size
            
            if test_start < self.gap_days:
                continue  # 跳过数据不足的分割
                
            # 训练集结束位置（考虑gap）
            train_end = test_start - self.gap_days
            
            if train_end <= test_size:  # 确保有足够的训练数据
                continue
                
            # 生成索引
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            # 确保索引有效
            train_indices = train_indices[train_indices < n_samples]
            test_indices = test_indices[test_indices < n_samples]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                logger.debug(f"分割 {i+1}: 训练[0:{train_end}], gap={self.gap_days}, 测试[{test_start}:{test_end}]")
                yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """返回分割数"""
        return self.n_splits


def create_simple_safe_cv(n_splits=5, gap_days=11, test_size=0.2):  # CRITICAL FIX: gap_days=11
    """创建简单安全CV分割器"""
    return SimpleSafeTimeSeriesSplit(n_splits=n_splits, gap_days=gap_days, test_size=test_size)


if __name__ == "__main__":
    # 测试简单安全CV
    import numpy as np
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    
    # 创建测试数据
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)
    
    # 创建安全CV
    safe_cv = create_simple_safe_cv(n_splits=3, gap_days=11)  # CRITICAL FIX: gap_days=11
    
    # 测试与sklearn的兼容性
    model = RandomForestRegressor(n_estimators=10)
    scores = cross_val_score(model, X, y, cv=safe_cv, scoring='r2')
    
    print(f"简单安全CV测试成功: 得分={scores}, 平均={scores.mean():.3f}")