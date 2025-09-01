#!/usr/bin/env python3
"""
全局NaN处理配置 - CRITICAL FIX
统一整个BMA系统的NaN处理策略，确保一致性
"""

import pandas as pd
import numpy as np
import logging
from typing import Union, Optional, Literal
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GlobalNaNConfig:
    """全局NaN处理配置 - 单一真相源"""
    
    # 主要填充策略
    primary_method: Literal['cross_sectional_median', 'forward_fill', 'zero_fill'] = 'cross_sectional_median'
    
    # 备用填充策略（当主策略失效时）
    fallback_method: Literal['zero_fill', 'forward_fill', 'mean_fill'] = 'zero_fill'
    
    # 数值限制
    max_fill_ratio: float = 0.5  # 单列NaN比例超过50%时警告
    fill_limit: int = 5  # 前向填充最大次数
    
    # 异常值处理
    winsorize_percentiles: tuple = (0.01, 0.99)  # 1%和99%分位数
    enable_winsorization: bool = True
    
    # 最小样本数要求
    min_samples_per_date: int = 3  # 每日最少3个非NaN样本才进行横截面填充


# 全局实例 - 所有模块使用此配置
GLOBAL_NAN_CONFIG = GlobalNaNConfig()


def unified_nan_handler(data: Union[pd.Series, pd.DataFrame], 
                       df: Optional[pd.DataFrame] = None,
                       date_col: str = 'date',
                       method: Optional[str] = None) -> Union[pd.Series, pd.DataFrame]:
    """
    CRITICAL FIX: 统一NaN处理函数
    所有BMA组件必须使用此函数处理NaN值
    
    Args:
        data: 要处理的数据（Series或DataFrame）
        df: 包含日期信息的DataFrame（用于横截面填充）
        date_col: 日期列名
        method: 覆盖默认的处理方法
        
    Returns:
        处理后的数据
    """
    config = GLOBAL_NAN_CONFIG
    method = method or config.primary_method
    
    if isinstance(data, pd.Series):
        return _handle_series_nan(data, df, date_col, method, config)
    elif isinstance(data, pd.DataFrame):
        return _handle_dataframe_nan(data, date_col, method, config)
    else:
        raise ValueError(f"不支持的数据类型: {type(data)}")


def _handle_series_nan(series: pd.Series, 
                      df: Optional[pd.DataFrame],
                      date_col: str,
                      method: str,
                      config: GlobalNaNConfig) -> pd.Series:
    """处理Series的NaN值"""
    if series.empty or series.isna().sum() == 0:
        return series
    
    nan_ratio = series.isna().sum() / len(series)
    if nan_ratio > config.max_fill_ratio:
        logger.warning(f"Series NaN比例过高: {nan_ratio:.2%}，建议检查数据质量")
    
    if method == 'cross_sectional_median' and df is not None and date_col in df.columns:
        return _cross_sectional_fill_series(series, df, date_col, config)
    elif method == 'forward_fill':
        filled = series.fillna(method='ffill', limit=config.fill_limit)
        remaining_nan = filled.fillna(0)  # 剩余NaN用0填充
        return remaining_nan
    elif method == 'zero_fill':
        return series.fillna(0)
    else:
        logger.warning(f"未知方法 {method}，使用零填充")
        return series.fillna(0)


def _cross_sectional_fill_series(series: pd.Series,
                                df: pd.DataFrame,
                                date_col: str,
                                config: GlobalNaNConfig) -> pd.Series:
    """横截面中位数填充Series"""
    try:
        temp_df = pd.DataFrame({
            'data': series,
            'date': df[date_col].reindex(series.index),
            'original_index': series.index
        })
        
        def fill_cross_section(group):
            if len(group.dropna()) >= config.min_samples_per_date:
                daily_median = group['data'].median()
                group['data'] = group['data'].fillna(daily_median)
            else:
                # 样本数不足，使用历史中位数
                historical_median = temp_df['data'].median()
                group['data'] = group['data'].fillna(historical_median if not pd.isna(historical_median) else 0)
            return group
        
        filled_df = temp_df.groupby('date').apply(fill_cross_section).reset_index(drop=True)
        result = filled_df.set_index('original_index')['data']
        
        # 最终清理
        return result.fillna(0)
        
    except Exception as e:
        logger.warning(f"横截面填充失败: {e}，使用fallback方法")
        return series.fillna(0)


def _handle_dataframe_nan(df: pd.DataFrame,
                         date_col: str,
                         method: str,
                         config: GlobalNaNConfig) -> pd.DataFrame:
    """处理DataFrame的NaN值"""
    if df.empty:
        return df
        
    df_clean = df.copy()
    feature_cols = [col for col in df.columns if col != date_col and df[col].dtype in ['float64', 'int64']]
    
    if not feature_cols:
        return df_clean
    
    for col in feature_cols:
        df_clean[col] = unified_nan_handler(df_clean[col], df_clean, date_col, method)
    
    return df_clean


def validate_data_quality(data: Union[pd.Series, pd.DataFrame],
                         name: str = "unnamed") -> dict:
    """
    数据质量验证
    返回数据质量报告
    """
    if isinstance(data, pd.Series):
        total_count = len(data)
        nan_count = data.isna().sum()
        inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum() if not data.empty else 0
    else:
        total_count = data.shape[0] * data.shape[1]
        nan_count = data.isna().sum().sum()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(data[numeric_cols]).sum().sum() if len(numeric_cols) > 0 else 0
    
    report = {
        'name': name,
        'total_count': total_count,
        'nan_count': int(nan_count),
        'inf_count': int(inf_count),
        'nan_ratio': nan_count / total_count if total_count > 0 else 0,
        'quality_score': 1 - (nan_count + inf_count) / total_count if total_count > 0 else 0
    }
    
    if report['nan_ratio'] > 0.1:
        logger.warning(f"数据质量警告 {name}: NaN比例 {report['nan_ratio']:.2%}")
    
    return report


# 兼容性函数 - 确保现有代码正常工作
def safe_fillna(data: pd.Series, df: pd.DataFrame = None, date_col: str = 'date') -> pd.Series:
    """兼容性函数 - 重定向到统一处理器"""
    return unified_nan_handler(data, df, date_col)


def clean_nan_predictive_safe(df: pd.DataFrame, 
                             feature_cols: list = None,
                             date_col: str = 'date') -> pd.DataFrame:
    """兼容性函数 - 重定向到统一处理器"""
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != date_col and df[col].dtype in ['float64', 'int64']]
    
    return unified_nan_handler(df, None, date_col)


if __name__ == "__main__":
    # 测试统一NaN处理
    test_data = pd.Series([1, np.nan, 3, np.nan, 5], name='test_series')
    test_df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=5),
        'ticker': ['A', 'B', 'A', 'B', 'A']
    })
    
    print("原始数据:", test_data.tolist())
    
    # 测试零填充
    zero_filled = unified_nan_handler(test_data, method='zero_fill')
    print("零填充:", zero_filled.tolist())
    
    # 测试横截面填充
    cross_filled = unified_nan_handler(test_data, test_df, 'date', 'cross_sectional_median')
    print("横截面填充:", cross_filled.tolist())
    
    # 数据质量报告
    quality_report = validate_data_quality(test_data, "测试数据")
    print("数据质量报告:", quality_report)