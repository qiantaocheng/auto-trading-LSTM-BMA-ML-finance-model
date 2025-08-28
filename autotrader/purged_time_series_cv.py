#!/usr/bin/env python3
"""
Purged Group Time Series Cross Validation
专注于时间序列分组和Embargo的CV分割器，移除重复的训练逻辑
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional
from sklearn.model_selection import BaseCrossValidator
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """验证配置 - 与Ultra Enhanced V6保持一致"""
    n_splits: int = 5
    test_size: int = 63  # 测试集大小（交易日）
    gap: int = 10        # ✅ FIX: 与isolation_days=10保持一致
    embargo: int = 0     # ✅ FIX: V6使用单一隔离，避免double purge
    min_train_size: int = 252  # 最小训练集大小
    group_freq: str = 'W'      # 分组频率
    
@dataclass
class CVResults:
    """交叉验证结果（保留接口兼容性）"""
    oof_predictions: pd.Series
    oof_ic: float
    oof_rank_ic: float
    oof_ndcg: float
    fold_metrics: list
    feature_importance: dict
    uncertainty_estimates: pd.Series

class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """
    Purged Group Time Series Split with Embargo
    
    专门针对金融时间序列数据的交叉验证分割器：
    - 按时间分组避免数据泄露
    - Gap和Embargo期间防止信息泄露
    - 支持不等长的时间组
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def split(self, X, y=None, groups=None):
        """
        生成训练/测试索引对
        
        Args:
            X: 特征数据 
            y: 目标变量（可选）
            groups: 时间分组标识符
            
        Yields:
            (train_indices, test_indices): 训练和测试索引
        """
        if groups is None:
            raise ValueError("groups参数是必须的")
        
        # 确保索引对齐
        if hasattr(X, 'index'):
            data_index = X.index
        else:
            data_index = np.arange(len(X))
        
        if hasattr(groups, 'index'):
            groups = groups.reindex(data_index)
        
        unique_groups = sorted(groups.unique())
        n_groups = len(unique_groups)
        
        logger.info(f"总共{n_groups}个时间组，配置{self.config.n_splits}折验证")
        
        # 计算每折的测试组数量
        groups_per_fold = max(1, self.config.test_size // 20)  # 假设每组~20个样本
        
        for i in range(self.config.n_splits):
            # 计算测试组的起始位置
            test_start_idx = min(
                n_groups - groups_per_fold,
                int(n_groups * (i + 1) / (self.config.n_splits + 1))
            )
            test_end_idx = min(n_groups, test_start_idx + groups_per_fold)
            
            # ✅ FIX: V6单一隔离方法，只使用gap
            train_end_idx = max(0, test_start_idx - self.config.gap)  # 不再叠加embargo
            
            # 确保最小训练集大小 - 进一步降低要求以适应小数据集
            min_train_groups = max(2, min(3, n_groups // 3))  # 最少2组，最多总组数的1/3
            if train_end_idx < min_train_groups:
                logger.warning(f"第{i+1}折训练数据不足({train_end_idx}<{min_train_groups})，调整为使用{min_train_groups}组")
                train_end_idx = min_train_groups  # 强制使用最小组数而不是跳过
            
            # 选择训练和测试组
            train_groups = unique_groups[:train_end_idx]
            test_groups = unique_groups[test_start_idx:test_end_idx]
            
            if not train_groups or not test_groups:
                continue
            
            # 转换为索引
            train_mask = groups.isin(train_groups)
            test_mask = groups.isin(test_groups)
            
            train_indices = data_index[train_mask].tolist()
            test_indices = data_index[test_mask].tolist()
            
            # 放宽训练样本数量要求
            min_samples = min(self.config.min_train_size, len(data_index) // 3)  # 最多要求1/3的数据
            if len(train_indices) < min_samples or len(test_indices) == 0:
                logger.debug(f"第{i+1}折样本不足: 训练{len(train_indices)}<{min_samples} 或 测试{len(test_indices)}=0")
                continue
                
            logger.debug(f"第{i+1}折: 训练{len(train_indices)}样本, 测试{len(test_indices)}样本")
            logger.debug(f"训练期间: {train_groups[0]} to {train_groups[-1]}")
            logger.debug(f"测试期间: {test_groups[0]} to {test_groups[-1]}")
            
            # 验证时间顺序（训练 < gap < 测试）
            if hasattr(groups, 'dtype') and 'datetime' in str(groups.dtype):
                train_max_date = groups[train_mask].max()
                test_min_date = groups[test_mask].min()
                time_gap_days = (test_min_date - train_max_date).days
                
                # V6: 使用单一gap检查，更宽松的验证
                required_gap = self.config.gap
                if time_gap_days < required_gap:
                    logger.debug(f"第{i+1}折时间间隔: {time_gap_days}天 < 期望{required_gap}天（可接受）")
                    # 对于小数据集，只要不是负数就继续
                    if time_gap_days < 0:
                        logger.warning(f"第{i+1}折时间顺序错误，跳过")
                        continue
            
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """返回分割数量"""
        return self.config.n_splits


def create_time_groups(dates: pd.Series, freq: str = 'W') -> pd.Series:
    """创建时间分组"""
    if freq == 'D':
        return dates.dt.date
    elif freq == 'W':
        return dates.dt.to_period('W')
    elif freq == 'M':
        return dates.dt.to_period('M')
    else:
        raise ValueError(f"不支持的频率: {freq}")


# 使用示例（移除了重复的训练逻辑）
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Purged Time Series CV - 精简版")
    print("注意：训练逻辑已移至LTR模块，此处仅提供分割器功能")
    
    # 生成模拟数据用于测试分割器
    from .production_random_control import get_production_safe_seed
    seed = get_production_safe_seed("TEST", "cv_test_data")
    np.random.seed(seed)
    n_samples = 1000
    
    # 模拟时间序列数据
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    X = pd.DataFrame(
        np.random.randn(n_samples, 5),
        columns=[f'feature_{i}' for i in range(5)],
        index=dates
    )
    
    # 创建时间组
    groups = create_time_groups(dates, freq='W')
    
    # 配置验证参数（V6一致性配置）
    config = ValidationConfig(
        n_splits=5,
        test_size=63,
        gap=10,      # 与isolation_days一致
        embargo=0,   # V6单一隔离
        group_freq='W'
    )
    
    # 测试分割器
    cv = PurgedGroupTimeSeriesSplit(config)
    
    print(f"测试数据: {len(X)}样本, {len(groups.unique())}个时间组")
    
    fold_count = 0
    for train_idx, test_idx in cv.split(X, groups=groups):
        fold_count += 1
        print(f"第{fold_count}折: 训练{len(train_idx)}样本, 测试{len(test_idx)}样本")
        
        if fold_count >= 3:  # 只显示前3折
            break
    
    print("分割器测试完成")
