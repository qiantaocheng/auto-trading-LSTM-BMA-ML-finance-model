"""
统一NaN处理策略 - 提升预测性能
避免不一致的NaN处理导致的虚假信号和噪音
"""

import pandas as pd
import numpy as np
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

class UnifiedNaNHandler:
    """统一的NaN处理器，避免预测性能损失"""
    
    def __init__(self, method: str = "cross_sectional_median"):
        """
        初始化NaN处理器
        
        Args:
            method: 处理方法
                - cross_sectional_median: 横截面中位数填充（推荐）
                - cross_sectional_mean: 横截面均值填充
                - industry_median: 行业中位数填充
                - forward_fill_safe: 安全前向填充（限制天数）
        """
        self.method = method
        self.fill_limit = 5  # 前向填充最大天数
        
    def handle_nan(self, df: pd.DataFrame, 
                   feature_cols: list = None,
                   date_col: str = 'date',
                   group_col: str = 'ticker',
                   industry_col: str = 'SECTOR') -> pd.DataFrame:
        """
        统一处理NaN值，避免引入虚假信号
        
        Args:
            df: 输入数据
            feature_cols: 需要处理的特征列，如果None则处理所有数值列
            date_col: 日期列名
            group_col: 分组列名（如ticker）
            industry_col: 行业分类列名
            
        Returns:
            处理后的DataFrame
        """
        if df is None or df.empty:
            return df
            
        df_clean = df.copy()
        
        # 确定需要处理的列
        if feature_cols is None:
            feature_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            # 排除日期和ID列
            feature_cols = [col for col in feature_cols 
                          if col not in [date_col, group_col] and not col.endswith('_id')]
        
        logger.debug(f"使用{self.method}方法处理{len(feature_cols)}个特征的NaN值")
        
        if self.method == "cross_sectional_median":
            df_clean = self._cross_sectional_fill(df_clean, feature_cols, date_col, 'median')
        elif self.method == "cross_sectional_mean":
            df_clean = self._cross_sectional_fill(df_clean, feature_cols, date_col, 'mean')
        elif self.method == "industry_median":
            df_clean = self._industry_fill(df_clean, feature_cols, date_col, industry_col)
        elif self.method == "forward_fill_safe":
            df_clean = self._forward_fill_safe(df_clean, feature_cols, group_col)
        else:
            logger.warning(f"未知的NaN处理方法: {self.method}，使用默认横截面中位数")
            df_clean = self._cross_sectional_fill(df_clean, feature_cols, date_col, 'median')
            
        # 最终清理：剩余NaN填充为0（极少数情况）
        remaining_nan = df_clean[feature_cols].isna().sum().sum()
        if remaining_nan > 0:
            logger.warning(f"剩余{remaining_nan}个NaN值，使用0填充")
            df_clean[feature_cols] = df_clean[feature_cols].fillna(0)
            
        return df_clean
    
    def _cross_sectional_fill(self, df: pd.DataFrame, 
                             feature_cols: list, 
                             date_col: str, 
                             stat_method: str = 'median') -> pd.DataFrame:
        """横截面统计量填充 - 避免引入虚假信号"""
        def fill_cross_section(group):
            for col in feature_cols:
                if col in group.columns:
                    if stat_method == 'median':
                        fill_value = group[col].median()
                    else:
                        fill_value = group[col].mean()
                    
                    if not pd.isna(fill_value):
                        group[col] = group[col].fillna(fill_value)
            return group
        
        return df.groupby(date_col).apply(fill_cross_section).reset_index(level=0, drop=True)
    
    def _industry_fill(self, df: pd.DataFrame, 
                      feature_cols: list, 
                      date_col: str, 
                      industry_col: str) -> pd.DataFrame:
        """行业中位数填充"""
        def fill_industry_section(group):
            for col in feature_cols:
                if col in group.columns:
                    # 先尝试行业内填充
                    if industry_col in group.columns:
                        for industry in group[industry_col].unique():
                            industry_mask = group[industry_col] == industry
                            industry_median = group[industry_mask][col].median()
                            if not pd.isna(industry_median):
                                group.loc[industry_mask, col] = group.loc[industry_mask, col].fillna(industry_median)
                    
                    # 行业填充后仍有NaN，使用全截面中位数
                    overall_median = group[col].median()
                    if not pd.isna(overall_median):
                        group[col] = group[col].fillna(overall_median)
            return group
        
        return df.groupby(date_col).apply(fill_industry_section).reset_index(level=0, drop=True)
    
    def _forward_fill_safe(self, df: pd.DataFrame, 
                          feature_cols: list, 
                          group_col: str) -> pd.DataFrame:
        """安全的前向填充 - 限制填充天数避免使用过期信息"""
        def safe_ffill_group(group):
            for col in feature_cols:
                if col in group.columns:
                    # 限制前向填充天数
                    group[col] = group[col].fillna(method='ffill', limit=self.fill_limit)
            return group
        
        return df.groupby(group_col).apply(safe_ffill_group).reset_index(level=0, drop=True)
    
    def handle_nans(self, df: pd.DataFrame, 
                    feature_cols: list = None,
                    date_col: str = 'date',
                    group_col: str = 'ticker',
                    industry_col: str = 'SECTOR') -> pd.DataFrame:
        """别名方法 - 调用handle_nan以保持兼容性"""
        return self.handle_nan(df, feature_cols, date_col, group_col, industry_col)
    
    def get_nan_summary(self, df: pd.DataFrame, feature_cols: list = None) -> dict:
        """获取NaN统计信息"""
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        nan_counts = df[feature_cols].isna().sum()
        total_values = len(df) * len(feature_cols)
        total_nan = nan_counts.sum()
        
        return {
            'total_nan_values': int(total_nan),
            'total_values': int(total_values),
            'nan_percentage': float(total_nan / total_values * 100),
            'columns_with_nan': int((nan_counts > 0).sum()),
            'worst_columns': nan_counts.nlargest(5).to_dict()
        }

# 全局实例
unified_nan_handler = UnifiedNaNHandler(method="cross_sectional_median")

def clean_nan_predictive_safe(df: pd.DataFrame, 
                             feature_cols: list = None,
                             method: str = "cross_sectional_median") -> pd.DataFrame:
    """
    预测性能安全的NaN清理函数
    
    Args:
        df: 输入数据
        feature_cols: 特征列列表
        method: 处理方法
        
    Returns:
        清理后的数据
    """
    handler = UnifiedNaNHandler(method=method)
    return handler.handle_nan(df, feature_cols=feature_cols)

if __name__ == "__main__":
    # 测试NaN处理器
    import pandas as pd
    import numpy as np
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=20),
        'ticker': ['A'] * 10 + ['B'] * 10,
        'SECTOR': ['Tech'] * 5 + ['Finance'] * 5 + ['Tech'] * 5 + ['Finance'] * 5,
        'feature1': [1, 2, np.nan, 4, 5] * 4,
        'feature2': [np.nan, 2, 3, np.nan, 5] * 4,
        'feature3': [1, np.nan, np.nan, np.nan, 5] * 4
    })
    
    handler = UnifiedNaNHandler()
    
    print("原始数据NaN统计:")
    print(handler.get_nan_summary(test_data))
    
    cleaned = handler.handle_nan(test_data)
    
    print("\n清理后NaN统计:")
    print(handler.get_nan_summary(cleaned))
    
    print("\n清理效果:")
    print(f"原始NaN数量: {test_data.select_dtypes(include=[np.number]).isna().sum().sum()}")
    print(f"清理后NaN数量: {cleaned.select_dtypes(include=[np.number]).isna().sum().sum()}")