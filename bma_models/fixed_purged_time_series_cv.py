#!/usr/bin/env python3
"""
修复的 Purged Group Time Series Cross Validation
解决数据泄露、样本不足、时间间隔等问题
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
    """验证配置 - 与Ultra Enhanced模型保持一致"""
    n_splits: int = 5
    test_size: int = 63  # 测试集大小（交易日）
    gap: int = 10        # ✅ FIX: 统一为10天，与isolation_days一致
    embargo: int = 0     # ✅ FIX: 避免双重隔离，V6使用单一隔离方法
    min_train_size: int = 252  # 最小训练集大小
    group_freq: str = 'W'      # 分组频率
    strict_validation: bool = True  # 严格验证模式
    
@dataclass
class CVResults:
    """交叉验证结果"""
    oof_predictions: pd.Series
    oof_ic: float
    oof_rank_ic: float
    oof_ndcg: float
    fold_metrics: list
    feature_importance: dict
    uncertainty_estimates: pd.Series

class FixedPurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """
    修复的 Purged Group Time Series Split with Embargo
    
    主要修复:
    1. 严格的时间顺序验证
    2. 数据泄露防护
    3. 合理的样本数要求
    4. 清晰的Gap和Embargo逻辑
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def split(self, X, y=None, groups=None):
        """
        生成训练/测试索引对（修复版）
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
        
        # ✅ FIX: 更新数据充足性检查（V6单一隔离）
        min_required_groups = self.config.n_splits + self.config.gap + 2  # 移除embargo
        if n_groups < min_required_groups:
            logger.warning(f"数据较少: 推荐至少{min_required_groups}组，实际只有{n_groups}组")
            # V6: 对小数据集更宽松，不直接返回空
            if n_groups < (self.config.n_splits + 2):  # 最低要求
                logger.error(f"数据极少: 至少需要{self.config.n_splits + 2}组进行CV")
                if self.config.strict_validation:
                    return  # 严格模式下直接返回空
        
        # 计算每折的测试组数量
        groups_per_fold = max(1, self.config.test_size // 20)  # 假设每组~20个样本
        
        valid_folds = 0
        for i in range(self.config.n_splits):
            # 计算测试组的起始位置
            test_start_idx = min(
                n_groups - groups_per_fold,
                int(n_groups * (i + 1) / (self.config.n_splits + 1))
            )
            test_end_idx = min(n_groups, test_start_idx + groups_per_fold)
            
            # ✅ FIX: V6单一隔离方法 - 只使用gap，避免双重隔离
            # Gap已经包含了所有需要的隔离期间
            # Embargo设为0以避免与Enhanced Temporal Validation重复
            total_buffer = self.config.gap  # V6: 使用单一gap，不再叠加embargo
            train_end_idx = max(0, test_start_idx - total_buffer)
            
            logger.debug(f"第{i+1}折: 使用单一隔离gap={self.config.gap}天，避免双重隔离")
            
            # 【修复3】严格检查最小训练集要求
            if self.config.strict_validation:
                if train_end_idx < (self.config.min_train_size // 20):  # 转换为组数
                    logger.warning(f"第{i+1}折训练数据不足，跳过")
                    continue
            else:
                # 非严格模式下的最低要求
                min_train_groups = max(2, n_groups // 10)
                if train_end_idx < min_train_groups:
                    logger.warning(f"第{i+1}折训练数据过少，跳过")
                    continue
            
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
            
            if len(train_indices) == 0 or len(test_indices) == 0:
                continue
            
            # 【修复4】严格验证时间顺序
            if hasattr(groups, 'dtype') and 'datetime' in str(groups.dtype):
                train_max_date = groups[train_mask].max()
                test_min_date = groups[test_mask].min()
                time_gap_days = int((test_min_date - train_max_date) / pd.Timedelta(days=1))
                
                required_gap = self.config.gap  # V6: 只需要gap，不需要embargo
                if time_gap_days < required_gap:
                    if self.config.strict_validation:
                        logger.error(f"第{i+1}折时间间隔不足: {time_gap_days}天 < {required_gap}天，跳过")
                        continue
                    else:
                        logger.warning(f"第{i+1}折时间间隔不足: {time_gap_days}天 < {required_gap}天")
                
                logger.info(f"第{i+1}折时间验证通过: Gap={time_gap_days}天")
            
            # 【修复5】额外的重叠检查
            train_date_range = set(groups[train_mask])
            test_date_range = set(groups[test_mask])
            overlap = train_date_range & test_date_range
            
            if overlap:
                logger.error(f"第{i+1}折发现数据重叠: {len(overlap)}个组")
                if self.config.strict_validation:
                    continue
            
            valid_folds += 1
            
            # 🔥 详细的训练/验证时间范围日志
            train_start_date = train_groups[0]
            train_end_date = train_groups[-1]
            valid_start_date = test_groups[0]  
            valid_end_date = test_groups[-1]
            
            # 计算实际时间间隔
            if hasattr(train_groups[0], 'strftime'):
                # datetime类型
                train_start_str = train_start_date.strftime('%Y-%m-%d')
                train_end_str = train_end_date.strftime('%Y-%m-%d')
                valid_start_str = valid_start_date.strftime('%Y-%m-%d')
                valid_end_str = valid_end_date.strftime('%Y-%m-%d')
            else:
                # 其他类型
                train_start_str = str(train_start_date)
                train_end_str = str(train_end_date)
                valid_start_str = str(valid_start_date)
                valid_end_str = str(valid_end_date)
            
            logger.info(f"第{valid_folds}折: 训练{len(train_indices)}样本, 测试{len(test_indices)}样本")
            logger.info(f"训练期间: {train_start_str} to {train_end_str}")
            logger.info(f"测试期间: {valid_start_str} to {valid_end_str}")
            logger.info(f"时间缓冲: {total_buffer}组 (Gap:{self.config.gap} + Embargo:{self.config.embargo})")
            
            # 🔥 验证无重叠：确保 train_end < valid_start
            if hasattr(train_end_date, 'strftime') and hasattr(valid_start_date, 'strftime'):
                gap_days = int((valid_start_date - train_end_date) / pd.Timedelta(days=1))
                if gap_days < 0:
                    logger.error(f"🚨 第{valid_folds}折发现重叠: train_end({train_end_str}) >= valid_start({valid_start_str})")
                else:
                    logger.info(f"✅ 第{valid_folds}折时间无重叠: 间隔{gap_days}天")
            
            # 🔥 验证embargo>=持有期+1 (假设持有期H=1天)
            H = 1  # 持有期
            required_embargo = H + 1
            if self.config.embargo >= required_embargo:
                logger.info(f"✅ Embargo验证通过: {self.config.embargo}天 >= {required_embargo}天(H+1)")
            else:
                logger.warning(f"⚠️ Embargo不足: {self.config.embargo}天 < {required_embargo}天(H+1)")
            
            yield train_indices, test_indices
        
        if valid_folds == 0:
            logger.warning("没有生成有效的交叉验证折")
        else:
            logger.info(f"成功生成{valid_folds}个有效交叉验证折")

    def get_n_splits(self, X=None, y=None, groups=None):
        """返回分割数量"""
        return self.config.n_splits


def create_time_groups(dates: pd.Series, freq: str = 'W') -> pd.Series:
    """创建时间分组（修复版）"""
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)
    
    # 确保是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(dates):
        try:
            dates = pd.to_datetime(dates)
        except Exception as e:
            logger.error(f"日期转换失败: {e}")
            raise
    
    if freq == 'D':
        return dates.dt.date
    elif freq == 'W':
        # 修复：使用周的开始日期而非Period对象，避免与Timedelta的运算错误
        return dates.dt.to_period('W').dt.start_time
    elif freq == '7D':
        # 7天分组：使用每周起始日期作为分组标识
        # 修复：使用周的开始日期而非Period对象，避免与Timedelta的运算错误
        return dates.dt.to_period('W').dt.start_time
    elif freq == 'M':
        return dates.dt.to_period('M')
    else:
        raise ValueError(f"不支持的频率: {freq}")


def validate_timesplit_integrity(cv_splitter, X, groups):
    """验证时间分割的完整性"""
    logger.info("开始时间分割完整性验证...")
    
    issues_found = []
    fold_count = 0
    
    for train_idx, test_idx in cv_splitter.split(X, groups=groups):
        fold_count += 1
        
        # 检查重叠
        overlap = set(train_idx) & set(test_idx)
        if overlap:
            issues_found.append(f"第{fold_count}折: 发现{len(overlap)}个重叠样本")
        
        # 检查时间顺序
        if hasattr(groups, 'iloc'):
            train_dates = groups.iloc[train_idx]
            test_dates = groups.iloc[test_idx]
            
            train_max = train_dates.max()
            test_min = test_dates.min()
            
            if train_max >= test_min:
                issues_found.append(f"第{fold_count}折: 时间顺序错误 (训练最大={train_max}, 测试最小={test_min})")
    
    if issues_found:
        logger.error(f"发现{len(issues_found)}个问题:")
        for issue in issues_found:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info(f"时间分割验证通过 ({fold_count}折)")
        return True


# 测试代码
if __name__ == "__main__":
    import sys
    sys.path.append('.')
    
    logging.basicConfig(level=logging.INFO)
    
    print("=== 修复的 Purged Time Series CV 测试 ===")
    
    # 生成模拟数据
    # np.random.seed removed
    n_samples = 1000
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    X = pd.DataFrame(
        np.zeros(n_samples),
        columns=[f'feature_{i}' for i in range(5)],
        index=dates
    )
    
    # 创建时间组
    groups = create_time_groups(dates, freq='7D')
    
    # 测试严格模式
    config_strict = ValidationConfig(
        n_splits=5,
        test_size=63,
        gap=5,
        embargo=2,
        strict_validation=True
    )
    
    cv_strict = FixedPurgedGroupTimeSeriesSplit(config_strict)
    print(f"\n=== 严格模式测试 ===")
    print(f"数据: {len(X)}样本, {len(groups.unique())}个时间组")
    
    # 验证完整性
    integrity_ok = validate_timesplit_integrity(cv_strict, X, groups)
    print(f"完整性验证: {'通过' if integrity_ok else '失败'}")
    
    # 显示分割结果
    fold_count = 0
    for train_idx, test_idx in cv_strict.split(X, groups=groups):
        fold_count += 1
        print(f"第{fold_count}折: 训练{len(train_idx)}样本, 测试{len(test_idx)}样本")
        if fold_count >= 3:  # 只显示前3折
            break
    
    print("修复的时间分割测试完成")