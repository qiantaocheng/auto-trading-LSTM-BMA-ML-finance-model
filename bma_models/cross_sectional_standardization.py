"""
横截面标准化系统 - 每个时间点横截面标准化，消除时间漂移
确保因子在不同时间点具有可比性，提升预测稳定性
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class CrossSectionalStandardizer:
    """横截面标准化器 - 时序一致性保证"""
    
    def __init__(self, method: str = "robust_zscore", 
                 winsorize_quantiles: tuple = (0.01, 0.99),
                 min_observations: int = 10):
        """
        初始化横截面标准化器
        
        Args:
            method: 标准化方法
                - robust_zscore: 鲁棒Z-score（中位数+MAD）
                - zscore: 传统Z-score（均值+标准差）
                - rank: 排名标准化 [-1, 1]
                - quantile: 分位数标准化
            winsorize_quantiles: Winsorize分位数
            min_observations: 最小观测数量
        """
        self.method = method
        self.winsorize_quantiles = winsorize_quantiles
        self.min_observations = min_observations
        self.standardization_stats = {}
        
    def fit_transform(self, data: pd.DataFrame,
                     feature_cols: Optional[List[str]] = None,
                     date_col: str = 'date',
                     group_col: str = 'ticker') -> pd.DataFrame:
        """
        按时间截面标准化数据
        
        Args:
            data: 输入数据
            feature_cols: 需要标准化的特征列
            date_col: 日期列名
            group_col: 分组列名
            
        Returns:
            标准化后的数据
        """
        if data is None or data.empty:
            return data
            
        if feature_cols is None:
            feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols 
                          if col not in [date_col, group_col]]
        
        if len(feature_cols) == 0:
            logger.warning("没有数值特征需要标准化")
            return data
            
        logger.info(f"开始横截面标准化{len(feature_cols)}个特征，使用方法: {self.method}")
        
        result_data = data.copy()
        standardized_sections = []
        stats_by_date = {}
        
        # 按日期分组进行横截面标准化
        # 处理date既在index又在columns的情况
        try:
            # 尝试按columns中的date分组
            if date_col in data.columns:
                grouped_data = data.groupby(data[date_col])
            else:
                # 如果date不在columns中，尝试从index获取
                grouped_data = data.groupby(date_col)
        except ValueError as e:
            if "ambiguous" in str(e).lower():
                logger.warning(f"date列存在歧义，使用columns中的date: {e}")
                # 强制使用columns中的date
                grouped_data = data.groupby(data[date_col])
            else:
                raise e
        
        for date, group in grouped_data:
            if len(group) < self.min_observations:
                # 观测数量太少，跳过标准化
                standardized_sections.append(group)
                continue
            
            group_standardized = group.copy()
            date_stats = {}
            
            for col in feature_cols:
                if col not in group.columns:
                    continue
                    
                values = group[col].dropna()
                if len(values) < self.min_observations:
                    continue
                    
                # 应用Winsorization
                if self.winsorize_quantiles:
                    lower_q, upper_q = self.winsorize_quantiles
                    lower_bound = values.quantile(lower_q)
                    upper_bound = values.quantile(upper_q)
                    values = values.clip(lower_bound, upper_bound)
                
                # 标准化
                if self.method == "robust_zscore":
                    standardized_values, col_stats = self._robust_zscore_standardize(values)
                elif self.method == "zscore":
                    standardized_values, col_stats = self._zscore_standardize(values)
                elif self.method == "rank":
                    standardized_values, col_stats = self._rank_standardize(values)
                elif self.method == "quantile":
                    standardized_values, col_stats = self._quantile_standardize(values)
                else:
                    logger.warning(f"未知标准化方法: {self.method}，使用鲁棒Z-score")
                    standardized_values, col_stats = self._robust_zscore_standardize(values)
                
                # 更新数据
                group_standardized.loc[values.index, col] = standardized_values
                date_stats[col] = col_stats
            
            standardized_sections.append(group_standardized)
            stats_by_date[date] = date_stats
        
        # 合并结果
        if standardized_sections:
            result_data = pd.concat(standardized_sections, ignore_index=True)
            self.standardization_stats = stats_by_date
            logger.info(f"✅ 横截面标准化完成，处理{len(stats_by_date)}个时间截面")
            
        return result_data
    
    def _robust_zscore_standardize(self, values: pd.Series) -> tuple:
        """鲁棒Z-score标准化（使用中位数和MAD）"""
        median = values.median()
        mad = stats.median_abs_deviation(values, nan_policy='omit')
        
        if mad == 0:
            # MAD为0时，使用标准差
            std = values.std()
            if std == 0:
                return values * 0, {'method': 'robust_zscore', 'center': median, 'scale': 0}
            standardized = (values - median) / std
        else:
            # 标准的鲁棒标准化，MAD * 1.4826 ≈ std for normal distribution
            standardized = (values - median) / (mad * 1.4826)
        
        return standardized, {
            'method': 'robust_zscore',
            'center': median,
            'scale': mad * 1.4826,
            'mad': mad
        }
    
    def _zscore_standardize(self, values: pd.Series) -> tuple:
        """传统Z-score标准化"""
        mean = values.mean()
        std = values.std()
        
        if std == 0:
            return values * 0, {'method': 'zscore', 'center': mean, 'scale': 0}
        
        standardized = (values - mean) / std
        
        return standardized, {
            'method': 'zscore',
            'center': mean,
            'scale': std
        }
    
    def _rank_standardize(self, values: pd.Series) -> tuple:
        """排名标准化到 [-1, 1] 区间"""
        ranks = values.rank(method='average')
        n = len(ranks)
        
        # 转换到 [-1, 1] 区间
        standardized = 2 * (ranks - 1) / (n - 1) - 1
        
        return standardized, {
            'method': 'rank',
            'n_observations': n,
            'min_rank': ranks.min(),
            'max_rank': ranks.max()
        }
    
    def _quantile_standardize(self, values: pd.Series) -> tuple:
        """分位数标准化"""
        # 将数据转换为分位数 [0, 1]，然后标准正态化
        n = len(values)
        ranks = values.rank(method='average')
        quantiles = (ranks - 0.5) / n  # 避免边界问题
        
        # 逆标准正态分布
        standardized = stats.norm.ppf(quantiles)
        
        return standardized, {
            'method': 'quantile',
            'n_observations': n,
            'quantile_range': (quantiles.min(), quantiles.max())
        }
    
    def get_standardization_summary(self) -> Dict:
        """获取标准化统计摘要"""
        if not self.standardization_stats:
            return {}
        
        summary = {
            'n_dates': len(self.standardization_stats),
            'method': self.method,
            'winsorize_quantiles': self.winsorize_quantiles
        }
        
        # 统计各特征的标准化效果
        all_features = set()
        for date_stats in self.standardization_stats.values():
            all_features.update(date_stats.keys())
        
        feature_summary = {}
        for feature in all_features:
            centers = []
            scales = []
            
            for date_stats in self.standardization_stats.values():
                if feature in date_stats:
                    centers.append(date_stats[feature].get('center', 0))
                    scales.append(date_stats[feature].get('scale', 1))
            
            if centers and scales:
                feature_summary[feature] = {
                    'avg_center': np.mean(centers),
                    'avg_scale': np.mean(scales),
                    'center_stability': np.std(centers),
                    'scale_stability': np.std(scales)
                }
        
        summary['features'] = feature_summary
        return summary
    
    def validate_standardization(self, standardized_data: pd.DataFrame,
                               feature_cols: List[str],
                               date_col: str = 'date') -> Dict:
        """验证标准化效果"""
        validation_results = {}
        
        for date, group in standardized_data.groupby(date_col):
            date_validation = {}
            
            for col in feature_cols:
                if col in group.columns:
                    values = group[col].dropna()
                    if len(values) > 0:
                        date_validation[col] = {
                            'mean': values.mean(),
                            'std': values.std(),
                            'median': values.median(),
                            'mad': stats.median_abs_deviation(values, nan_policy='omit'),
                            'min': values.min(),
                            'max': values.max(),
                            'n_obs': len(values)
                        }
            
            validation_results[date] = date_validation
        
        return validation_results


# 全局实例
cross_sectional_standardizer = CrossSectionalStandardizer(method="robust_zscore")

def standardize_cross_sectional_predictive_safe(data: pd.DataFrame,
                                              feature_cols: Optional[List[str]] = None,
                                              method: str = "robust_zscore",
                                              winsorize_quantiles: tuple = (0.01, 0.99)) -> pd.DataFrame:
    """
    预测性能安全的横截面标准化
    
    Args:
        data: 输入数据
        feature_cols: 特征列
        method: 标准化方法
        winsorize_quantiles: Winsorize分位数
        
    Returns:
        标准化后的数据
    """
    standardizer = CrossSectionalStandardizer(
        method=method,
        winsorize_quantiles=winsorize_quantiles
    )
    return standardizer.fit_transform(data, feature_cols=feature_cols)


if __name__ == "__main__":
    # 测试横截面标准化
    import pandas as pd
    import numpy as np
    
    # 创建测试数据（模拟时间漂移）
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=10)
    tickers = ['A', 'B', 'C', 'D', 'E']
    
    test_data = []
    for i, date in enumerate(dates):
        # 模拟时间漂移：均值随时间变化
        drift_factor = i * 0.1
        for ticker in tickers:
            test_data.append({
                'date': date,
                'ticker': ticker,
                'factor1': np.random.randn() + drift_factor,  # 有漂移
                'factor2': np.random.randn() * 2 + drift_factor * 0.5,  # 有漂移
                'factor3': np.random.randn() * 0.5  # 无漂移
            })
    
    test_df = pd.DataFrame(test_data)
    
    print("原始数据统计（按日期）:")
    print(test_df.groupby('date')[['factor1', 'factor2', 'factor3']].mean().round(3))
    
    # 应用横截面标准化
    standardizer = CrossSectionalStandardizer(method="robust_zscore")
    standardized_df = standardizer.fit_transform(test_df)
    
    print("\n标准化后数据统计（按日期）:")
    print(standardized_df.groupby('date')[['factor1', 'factor2', 'factor3']].mean().round(3))
    
    print("\n标准化后数据标准差（按日期）:")
    print(standardized_df.groupby('date')[['factor1', 'factor2', 'factor3']].std().round(3))
    
    # 验证标准化效果
    validation = standardizer.validate_standardization(
        standardized_df, ['factor1', 'factor2', 'factor3'])
    
    print("\n标准化摘要:")
    summary = standardizer.get_standardization_summary()
    for feature, stats in summary.get('features', {}).items():
        print(f"{feature}: 中心稳定性={stats['center_stability']:.4f}, "
              f"尺度稳定性={stats['scale_stability']:.4f}")