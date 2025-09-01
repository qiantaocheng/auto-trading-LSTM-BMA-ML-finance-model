"""
Multi-Asset Safe Cross-Validation Module
========================================
防止多股票时间序列交叉验证中的信息泄露
确保不同股票的同期数据不会出现在同一个训练/验证分割中
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Iterator
from sklearn.model_selection import BaseCrossValidator
import logging

logger = logging.getLogger(__name__)


class MultiAssetTimeSeriesSplit(BaseCrossValidator):
    """
    多资产时间序列安全分割器
    
    防止信息泄露的原则:
    1. 严格按时间顺序分割，不允许训练数据的时间晚于验证数据
    2. 同一时间段内的不同股票不能同时出现在训练集和验证集
    3. 添加embargo period防止前瞻偏差
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 test_size_days: int = 21,
                 gap_days: int = 1,
                 embargo_days: int = 1):
        """
        初始化多资产时间序列分割器
        
        Parameters:
        -----------
        n_splits : int
            分割数量
        test_size_days : int 
            验证集时间长度（交易日）
        gap_days : int
            训练集和验证集之间的间隔天数
        embargo_days : int
            禁止期天数，防止前瞻偏差
        """
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.gap_days = gap_days
        self.embargo_days = embargo_days
    
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        生成时间序列安全的训练/验证分割
        
        Parameters:
        -----------
        X : pd.DataFrame
            特征数据，必须包含 'date' 和 'ticker' 列
        y : pd.Series, optional
            目标变量
        groups : array-like, optional
            分组信息（此处不使用，日期从X中提取）
            
        Yields:
        -------
        train_idx, val_idx : tuple of arrays
            训练集和验证集的索引
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame with 'date' and 'ticker' columns")
        
        if 'date' not in X.columns:
            raise ValueError("X must contain 'date' column")
            
        if 'ticker' not in X.columns:
            raise ValueError("X must contain 'ticker' column")
        
        # 确保date列是datetime类型
        X = X.copy()
        X['date'] = pd.to_datetime(X['date'])
        
        # 获取所有唯一日期并排序
        unique_dates = sorted(X['date'].unique())
        total_days = len(unique_dates)
        
        logger.info(f"多资产CV: 总共{total_days}个交易日, {X['ticker'].nunique()}只股票")
        
        if total_days < self.n_splits * (self.test_size_days + self.gap_days + self.embargo_days):
            logger.warning(f"数据长度不足，减少分割数量: {total_days} < 需要的最小长度")
            actual_splits = max(1, total_days // (self.test_size_days + self.gap_days + self.embargo_days))
            actual_splits = min(actual_splits, self.n_splits)
        else:
            actual_splits = self.n_splits
        
        # 计算每个分割的起始位置
        available_days = total_days - self.test_size_days - self.embargo_days
        step_size = available_days // actual_splits
        
        for i in range(actual_splits):
            # 计算验证集的时间范围
            val_end_idx = total_days - self.embargo_days - i * step_size
            val_start_idx = val_end_idx - self.test_size_days
            
            if val_start_idx < self.gap_days:
                logger.warning(f"分割 {i+1} 训练数据不足，跳过")
                continue
            
            # 训练集结束时间（留出gap）
            train_end_idx = val_start_idx - self.gap_days
            
            if train_end_idx <= 0:
                logger.warning(f"分割 {i+1} 训练数据不足，跳过")
                continue
            
            # 获取对应的日期范围
            train_end_date = unique_dates[train_end_idx]
            val_start_date = unique_dates[val_start_idx]
            val_end_date = unique_dates[val_end_idx]
            
            # 生成索引
            train_idx = X[X['date'] <= train_end_date].index.values
            val_idx = X[(X['date'] >= val_start_date) & (X['date'] <= val_end_date)].index.values
            
            if len(train_idx) == 0 or len(val_idx) == 0:
                logger.warning(f"分割 {i+1} 数据为空，跳过")
                continue
            
            # 验证没有时间泄露
            train_max_date = X.loc[train_idx, 'date'].max()
            val_min_date = X.loc[val_idx, 'date'].min()
            
            if train_max_date >= val_min_date:
                logger.error(f"发现时间泄露: 训练集最大日期 {train_max_date} >= 验证集最小日期 {val_min_date}")
                continue
            
            logger.info(f"分割 {i+1}: 训练集 {len(train_idx)} 样本 (至 {train_max_date.strftime('%Y-%m-%d')}), "
                       f"验证集 {len(val_idx)} 样本 ({val_min_date.strftime('%Y-%m-%d')} - {X.loc[val_idx, 'date'].max().strftime('%Y-%m-%d')})")
            
            yield train_idx, val_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """返回分割数量"""
        return self.n_splits


class SafeMultiAssetValidator:
    """
    多资产安全验证器
    提供额外的安全检查和验证功能
    """
    
    @staticmethod
    def validate_no_leakage(X: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray) -> bool:
        """
        验证训练集和验证集之间没有时间泄露
        
        Returns:
        --------
        bool : 如果没有泄露返回True
        """
        train_dates = X.loc[train_idx, 'date']
        val_dates = X.loc[val_idx, 'date']
        
        train_max = train_dates.max()
        val_min = val_dates.min()
        
        if train_max >= val_min:
            logger.error(f"时间泄露检测: 训练集最大日期 {train_max} >= 验证集最小日期 {val_min}")
            return False
        
        return True
    
    @staticmethod
    def validate_sufficient_data(train_idx: np.ndarray, val_idx: np.ndarray, 
                                min_train: int = 100, min_val: int = 20) -> bool:
        """
        验证训练集和验证集有足够的数据
        
        Returns:
        --------
        bool : 如果数据足够返回True
        """
        if len(train_idx) < min_train:
            logger.warning(f"训练集样本不足: {len(train_idx)} < {min_train}")
            return False
        
        if len(val_idx) < min_val:
            logger.warning(f"验证集样本不足: {len(val_idx)} < {min_val}")
            return False
        
        return True
    
    @staticmethod
    def check_asset_distribution(X: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray) -> dict:
        """
        检查资产在训练集和验证集中的分布
        
        Returns:
        --------
        dict : 分布统计信息
        """
        train_tickers = set(X.loc[train_idx, 'ticker'].unique())
        val_tickers = set(X.loc[val_idx, 'ticker'].unique())
        
        common_tickers = train_tickers.intersection(val_tickers)
        
        return {
            'train_assets': len(train_tickers),
            'val_assets': len(val_tickers),
            'common_assets': len(common_tickers),
            'coverage_ratio': len(common_tickers) / len(val_tickers) if val_tickers else 0
        }


def create_safe_multi_asset_cv(n_splits: int = 5, 
                              test_size_days: int = 21,
                              gap_days: int = 1,
                              embargo_days: int = 1) -> MultiAssetTimeSeriesSplit:
    """
    创建安全的多资产交叉验证器
    
    Parameters:
    -----------
    n_splits : int
        分割数量
    test_size_days : int
        验证集天数
    gap_days : int
        间隔天数
    embargo_days : int
        禁止期天数
        
    Returns:
    --------
    MultiAssetTimeSeriesSplit : 配置好的CV分割器
    """
    return MultiAssetTimeSeriesSplit(
        n_splits=n_splits,
        test_size_days=test_size_days,
        gap_days=gap_days,
        embargo_days=embargo_days
    )


# 使用示例
if __name__ == "__main__":
    # 创建测试数据
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    test_data = []
    for date in dates:
        for ticker in tickers:
            test_data.append({
                'date': date,
                'ticker': ticker,
                'feature1': np.random.randn(),
                'feature2': np.random.randn(),
                'target': np.random.randn()
            })
    
    df = pd.DataFrame(test_data)
    
    # 测试CV分割器
    cv = create_safe_multi_asset_cv(n_splits=3, test_size_days=30)
    validator = SafeMultiAssetValidator()
    
    for i, (train_idx, val_idx) in enumerate(cv.split(df)):
        print(f"\n分割 {i+1}:")
        print(f"训练集: {len(train_idx)} 样本")
        print(f"验证集: {len(val_idx)} 样本")
        
        # 验证安全性
        is_safe = validator.validate_no_leakage(df, train_idx, val_idx)
        has_sufficient_data = validator.validate_sufficient_data(train_idx, val_idx)
        distribution = validator.check_asset_distribution(df, train_idx, val_idx)
        
        print(f"时间安全: {is_safe}")
        print(f"数据充足: {has_sufficient_data}")
        print(f"资产分布: {distribution}")