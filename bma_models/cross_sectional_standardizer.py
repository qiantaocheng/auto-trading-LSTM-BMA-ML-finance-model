#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
横截面因子标准化模块
Cross-Sectional Factor Standardization

实现量化投资中的标准做法：每个时间点对所有股票的每个因子进行标准化
确保每个因子在每个时间点都是均值0方差1的标准正态分布
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List, Union
import warnings

logger = logging.getLogger(__name__)

class CrossSectionalStandardizer:
    """
    横截面因子标准化器

    特性：
    1. 按时间点对每个因子进行横截面标准化
    2. 保持时间顺序，避免前视偏误
    3. 处理缺失值和异常值
    4. 支持分组标准化（如行业中性）
    """

    def __init__(self,
                 min_valid_ratio: float = 0.5,
                 outlier_method: str = 'iqr',
                 outlier_threshold: float = 3.0,
                 fill_method: str = 'cross_median',
                 industry_neutral: bool = False):
        """
        初始化横截面标准化器

        Args:
            min_valid_ratio: 最小有效样本比例（每个时间点）
            outlier_method: 异常值处理方法 ['iqr', 'zscore', 'quantile']
            outlier_threshold: 异常值阈值
            fill_method: 缺失值填充方法 ['cross_median', 'zero', 'forward_fill']
            industry_neutral: 是否进行行业中性化
        """
        self.min_valid_ratio = min_valid_ratio
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.fill_method = fill_method
        self.industry_neutral = industry_neutral

        # 记录标准化统计信息
        self.standardization_stats = {}
        self.fitted = False

    def _handle_outliers(self, series: pd.Series) -> pd.Series:
        """处理异常值"""
        if len(series.dropna()) < 3:
            return series

        if self.outlier_method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR

        elif self.outlier_method == 'zscore':
            mean = series.mean()
            std = series.std()
            lower_bound = mean - self.outlier_threshold * std
            upper_bound = mean + self.outlier_threshold * std

        elif self.outlier_method == 'quantile':
            lower_bound = series.quantile(0.01)
            upper_bound = series.quantile(0.99)

        else:
            return series

        # Winsorize异常值
        series_winsorized = series.clip(lower=lower_bound, upper=upper_bound)

        outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()
        if outlier_count > 0:
            logger.debug(f"处理了 {outlier_count} 个异常值")

        return series_winsorized

    def _fill_missing_values(self, df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        """填充缺失值"""
        if self.fill_method == 'cross_median':
            # 使用当日横截面中位数填充
            for col in df.columns:
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)

        elif self.fill_method == 'zero':
            df = df.fillna(0)

        elif self.fill_method == 'forward_fill':
            # 这里需要历史数据，暂时用0填充
            df = df.fillna(0)

        return df

    def _standardize_cross_section(self, df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        """
        对单个时间点的横截面数据进行标准化

        Args:
            df: 单个时间点的因子数据 (stocks x factors)
            date: 时间点

        Returns:
            标准化后的DataFrame
        """
        df_standardized = df.copy()
        date_stats = {}

        for factor in df.columns:
            factor_data = df[factor].copy()

            # 检查有效样本数
            valid_count = factor_data.notna().sum()
            total_count = len(factor_data)

            if valid_count < max(3, int(total_count * self.min_valid_ratio)):
                logger.warning(f"日期 {date}, 因子 {factor}: 有效样本不足 ({valid_count}/{total_count})")
                df_standardized[factor] = 0  # 样本不足时填充0
                continue

            # 处理异常值
            factor_data_clean = self._handle_outliers(factor_data)

            # 计算标准化统计量（只使用有效数据）
            valid_data = factor_data_clean.dropna()

            if len(valid_data) < 3:
                df_standardized[factor] = 0
                continue

            mean_val = valid_data.mean()
            std_val = valid_data.std()

            # 避免除以0 - 优化处理
            if std_val < 1e-10:
                # 不再打印警告，改为静默处理
                # 使用去均值但不标准化的方式，保持因子原有信息
                df_standardized[factor] = factor_data_clean - mean_val
                continue

            # 执行标准化
            df_standardized[factor] = (factor_data_clean - mean_val) / std_val

            # 记录统计信息
            date_stats[factor] = {
                'mean': mean_val,
                'std': std_val,
                'valid_count': valid_count,
                'outlier_handled': True
            }

        # 填充剩余的NaN
        df_standardized = self._fill_missing_values(df_standardized, date)

        # 保存统计信息
        self.standardization_stats[date] = date_stats

        return df_standardized

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        拟合并转换数据

        Args:
            data: MultiIndex(date, ticker) DataFrame with factor columns

        Returns:
            标准化后的DataFrame，保持相同的索引结构
        """
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("数据必须具有MultiIndex(date, ticker)结构")

        if data.index.names != ['date', 'ticker']:
            logger.warning(f"Index names: {data.index.names}, 期望: ['date', 'ticker']")

        logger.info(f"开始横截面标准化: {data.shape}, 因子数: {len(data.columns)}")

        # 按日期分组处理
        standardized_dfs = []
        dates = data.index.get_level_values('date').unique().sort_values()

        for date in dates:
            try:
                # 获取当日横截面数据
                date_data = data.loc[date].copy()

                if date_data.empty:
                    logger.warning(f"日期 {date} 没有数据")
                    continue

                # 标准化当日数据
                date_standardized = self._standardize_cross_section(date_data, date)

                # 恢复MultiIndex
                date_standardized.index = pd.MultiIndex.from_tuples(
                    [(date, ticker) for ticker in date_standardized.index],
                    names=['date', 'ticker']
                )

                standardized_dfs.append(date_standardized)

            except Exception as e:
                logger.error(f"处理日期 {date} 时出错: {e}")
                continue

        if not standardized_dfs:
            raise ValueError("没有成功处理任何数据")

        # 合并所有日期的数据
        result = pd.concat(standardized_dfs, axis=0).sort_index()

        self.fitted = True

        logger.info(f"横截面标准化完成: {result.shape}")
        logger.info(f"处理了 {len(dates)} 个交易日")

        # 验证结果
        self._validate_standardization(result)

        return result

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        使用已拟合的参数转换新数据
        注意：这个方法主要用于在线预测，使用历史统计量
        """
        if not self.fitted:
            raise ValueError("请先调用fit_transform方法")

        return self.fit_transform(data)  # 对于横截面标准化，每次都重新计算

    def _validate_standardization(self, data: pd.DataFrame):
        """验证标准化效果"""
        try:
            # 随机选择几个日期验证
            dates = data.index.get_level_values('date').unique()
            sample_dates = np.random.choice(dates, min(5, len(dates)), replace=False)

            for date in sample_dates:
                date_data = data.loc[date]

                for factor in data.columns:
                    factor_values = date_data[factor].dropna()

                    if len(factor_values) > 3:
                        mean_val = factor_values.mean()
                        std_val = factor_values.std()

                        # 检查是否接近标准正态分布
                        if abs(mean_val) > 0.1:
                            logger.warning(f"日期 {date}, 因子 {factor}: 均值偏离0 ({mean_val:.4f})")

                        if abs(std_val - 1.0) > 0.2:
                            logger.warning(f"日期 {date}, 因子 {factor}: 标准差偏离1 ({std_val:.4f})")

        except Exception as e:
            logger.warning(f"标准化验证失败: {e}")

    def get_standardization_summary(self) -> Dict[str, Any]:
        """获取标准化统计摘要"""
        if not self.fitted:
            return {"error": "未拟合"}

        summary = {
            "总处理日期数": len(self.standardization_stats),
            "因子数": len(self.standardization_stats[list(self.standardization_stats.keys())[0]]) if self.standardization_stats else 0,
            "配置": {
                "min_valid_ratio": self.min_valid_ratio,
                "outlier_method": self.outlier_method,
                "outlier_threshold": self.outlier_threshold,
                "fill_method": self.fill_method
            }
        }

        return summary

def standardize_factors_cross_sectionally(data: pd.DataFrame,
                                        **kwargs) -> Tuple[pd.DataFrame, CrossSectionalStandardizer]:
    """
    便捷函数：对因子进行横截面标准化

    Args:
        data: MultiIndex(date, ticker) DataFrame
        **kwargs: CrossSectionalStandardizer的参数

    Returns:
        (standardized_data, standardizer)
    """
    standardizer = CrossSectionalStandardizer(**kwargs)
    standardized_data = standardizer.fit_transform(data)

    return standardized_data, standardizer

# 快速测试函数
def test_cross_sectional_standardization():
    """测试横截面标准化"""
    import datetime

    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

    np.random.seed(42)
    data_list = []

    for date in dates:
        for ticker in tickers:
            data_list.append({
                'date': date,
                'ticker': ticker,
                'factor1': np.random.normal(0, 1) * 100,  # 大尺度因子
                'factor2': np.random.normal(0, 1) * 0.01,  # 小尺度因子
                'factor3': np.random.normal(0, 1),          # 正常尺度因子
            })

    df = pd.DataFrame(data_list)
    df = df.set_index(['date', 'ticker'])

    print("原始数据统计:")
    print(df.describe())

    # 执行标准化
    standardized_df, standardizer = standardize_factors_cross_sectionally(df)

    print("\n标准化后数据统计:")
    print(standardized_df.describe())

    print("\n标准化摘要:")
    print(standardizer.get_standardization_summary())

    return standardized_df, standardizer

if __name__ == "__main__":
    test_cross_sectional_standardization()