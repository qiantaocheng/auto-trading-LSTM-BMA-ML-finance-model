"""
Purged Time Series Cross-Validation
清洗时序交叉验证，防止数据泄漏
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit

class PurgedTimeSeriesCV:
    """
    时序交叉验证，带清洗和禁运期
    """
    
    def __init__(self, n_splits: int = 5, purge_days: int = 10, embargo_days: int = 10):
        """
        初始化
        
        Args:
            n_splits: CV折数
            purge_days: 清洗天数（训练集和验证集之间的间隔）
            embargo_days: 禁运天数（防止信息泄漏）
        """
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        生成训练集和验证集索引
        """
        n_samples = len(X)
        
        for train_idx, test_idx in self.tscv.split(X):
            # 应用清洗：移除训练集末尾的样本
            if len(train_idx) > self.purge_days:
                train_idx = train_idx[:-self.purge_days]
            
            # 应用禁运：移除测试集开始的样本  
            if len(test_idx) > self.embargo_days:
                test_idx = test_idx[self.embargo_days:]
            
            # 确保索引有效
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """获取折数"""
        return self.n_splits
    
    def get_purged_cv_params(self) -> dict:
        """获取CV参数"""
        return {
            'n_splits': self.n_splits,
            'purge_days': self.purge_days,
            'embargo_days': self.embargo_days
        }

# 创建默认实例
purged_cv = PurgedTimeSeriesCV(n_splits=5, purge_days=10, embargo_days=10)
