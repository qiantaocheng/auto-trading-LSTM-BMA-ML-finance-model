"""特征滞后优化器"""
import pandas as pd
import numpy as np

class FeatureLagOptimizer:
    """特征滞后优化器"""
    
    def __init__(self, lag_days=5):
        self.lag_days = lag_days
    
    def apply_lags(self, data):
        """应用特征滞后"""
        if data is None or data.empty:
            return data
        
        # 对每个特征创建滞后版本
        lagged_features = []
        for col in data.columns:
            if col not in ['target', 'date', 'ticker']:
                for lag in range(1, self.lag_days + 1):
                    lagged_col = f"{col}_lag{lag}"
                    if isinstance(data.index, pd.MultiIndex):
                        # MultiIndex处理
                        data[lagged_col] = data.groupby(level='ticker')[col].shift(lag)
                    else:
                        data[lagged_col] = data[col].shift(lag)
        
        return data
    
    def optimize_lags(self, data):
        """优化特征滞后（兼容旧API）"""
        return self.apply_lags(data)
