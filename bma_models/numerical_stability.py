"""
Numerical Stability Protection Module
====================================
数值稳定性保护模块，防止除零、log计算异常等数值计算错误
"""

import numpy as np
import pandas as pd
import warnings
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NumericalStabilityProtector:
    """数值稳定性保护器"""
    
    def __init__(self, 
                 epsilon: float = 1e-8,
                 log_min_value: float = 1e-10,
                 max_value: float = 1e6,
                 inf_replacement: float = 0.0):
        """
        Args:
            epsilon: 除零保护的最小值
            log_min_value: log计算的最小输入值
            max_value: 数值截断的最大值
            inf_replacement: 无穷值的替换值
        """
        self.epsilon = epsilon
        self.log_min_value = log_min_value
        self.max_value = max_value
        self.inf_replacement = inf_replacement
        
        logger.info(f"NumericalStabilityProtector initialized: "
                   f"epsilon={epsilon}, log_min={log_min_value}, max={max_value}")
    
    def safe_divide(self, numerator: Union[float, np.ndarray, pd.Series], 
                   denominator: Union[float, np.ndarray, pd.Series],
                   fill_value: float = 0.0) -> Union[float, np.ndarray, pd.Series]:
        """
        安全除法，防止除零错误
        
        Args:
            numerator: 分子
            denominator: 分母
            fill_value: 除零时的填充值
            
        Returns:
            安全的除法结果
        """
        if isinstance(denominator, (pd.Series, np.ndarray)):
            # 向量化操作
            safe_denominator = np.where(
                np.abs(denominator) < self.epsilon, 
                self.epsilon * np.sign(denominator), 
                denominator
            )
            result = numerator / safe_denominator
            
            # 处理除零情况
            zero_mask = np.abs(denominator) < self.epsilon
            if isinstance(result, pd.Series):
                result.loc[zero_mask] = fill_value
            else:
                result[zero_mask] = fill_value
                
        else:
            # 标量操作
            if abs(denominator) < self.epsilon:
                result = fill_value
            else:
                result = numerator / denominator
        
        return self._clip_extreme_values(result)
    
    def safe_log(self, values: Union[float, np.ndarray, pd.Series],
                base: Optional[float] = None) -> Union[float, np.ndarray, pd.Series]:
        """
        安全对数计算，防止log(0)和log(负数)错误
        
        Args:
            values: 输入值
            base: 对数底数，None为自然对数
            
        Returns:
            安全的对数结果
        """
        # 确保输入值为正数
        safe_values = np.maximum(values, self.log_min_value)
        
        # 计算对数
        if base is None:
            result = np.log(safe_values)
        else:
            result = np.log(safe_values) / np.log(base)
        
        # 记录警告信息
        if isinstance(values, (pd.Series, np.ndarray)):
            negative_count = np.sum(values <= 0)
            if negative_count > 0:
                logger.debug(f"安全log计算: {negative_count} 个非正值被调整")
        elif values <= 0:
            logger.debug(f"安全log计算: 输入值 {values} 被调整为 {self.log_min_value}")
        
        return self._clip_extreme_values(result)
    
    def safe_sqrt(self, values: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """
        安全开方计算，防止负数开方
        
        Args:
            values: 输入值
            
        Returns:
            安全的开方结果
        """
        safe_values = np.maximum(values, 0.0)
        result = np.sqrt(safe_values)
        
        # 记录负值调整
        if isinstance(values, (pd.Series, np.ndarray)):
            negative_count = np.sum(values < 0)
            if negative_count > 0:
                logger.debug(f"安全sqrt计算: {negative_count} 个负值被调整为0")
        elif values < 0:
            logger.debug(f"安全sqrt计算: 负值 {values} 被调整为0")
        
        return result
    
    def safe_power(self, base: Union[float, np.ndarray, pd.Series],
                  exponent: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """
        安全幂运算，防止0^0、负数的分数次幂等问题
        
        Args:
            base: 底数
            exponent: 指数
            
        Returns:
            安全的幂运算结果
        """
        # 处理特殊情况
        if isinstance(base, (pd.Series, np.ndarray)):
            # 向量化处理
            result = np.ones_like(base, dtype=float)
            
            # 正常情况
            normal_mask = (base > 0) | ((base < 0) & (np.floor(exponent) == exponent))
            result[normal_mask] = np.power(base[normal_mask], exponent[normal_mask] if hasattr(exponent, '__getitem__') else exponent)
            
            # 0^0 = 1 (按照惯例)
            zero_zero_mask = (np.abs(base) < self.epsilon) & (np.abs(exponent) < self.epsilon if hasattr(exponent, '__getitem__') else abs(exponent) < self.epsilon)
            result[zero_zero_mask] = 1.0
            
            # 0^正数 = 0
            zero_pos_mask = (np.abs(base) < self.epsilon) & (exponent > self.epsilon if hasattr(exponent, '__getitem__') else exponent > self.epsilon)
            result[zero_pos_mask] = 0.0
            
        else:
            # 标量处理
            if abs(base) < self.epsilon and abs(exponent) < self.epsilon:
                result = 1.0  # 0^0 = 1
            elif abs(base) < self.epsilon and exponent > self.epsilon:
                result = 0.0  # 0^正数 = 0
            elif base > 0 or (base < 0 and int(exponent) == exponent):
                result = base ** exponent
            else:
                # 负数的分数次幂，返回NaN后处理
                result = np.nan
        
        return self._handle_nan_inf(result)
    
    def safe_momentum(self, prices: pd.Series, window: int) -> pd.Series:
        """
        安全动量计算，防止数值异常
        
        Args:
            prices: 价格序列
            window: 动量窗口
            
        Returns:
            安全的动量值
        """
        if len(prices) <= window + 2:
            logger.warning(f"价格序列长度 {len(prices)} 不足以计算 {window} 期动量")
            return pd.Series(index=prices.index, dtype=float)
        
        # 获取当前价格和历史价格
        current_prices = prices.shift(2)
        past_prices = prices.shift(window + 2)
        
        # 安全的对数动量计算
        momentum = self.safe_log(current_prices) - self.safe_log(past_prices)
        
        return momentum
    
    def safe_returns(self, prices: pd.Series, periods: int = 1) -> pd.Series:
        """
        安全收益率计算
        
        Args:
            prices: 价格序列
            periods: 计算周期
            
        Returns:
            安全的收益率序列
        """
        if len(prices) <= periods:
            return pd.Series(index=prices.index, dtype=float)
        
        # 使用safe_divide计算收益率
        price_current = prices
        price_past = prices.shift(periods)
        
        returns = self.safe_divide(price_current, price_past, fill_value=0.0) - 1.0
        
        return returns
    
    def safe_volatility(self, returns: pd.Series, window: int) -> pd.Series:
        """
        安全波动率计算
        
        Args:
            returns: 收益率序列
            window: 计算窗口
            
        Returns:
            安全的波动率序列
        """
        if len(returns) < window:
            return pd.Series(index=returns.index, dtype=float)
        
        # 使用滚动标准差，并进行数值保护
        volatility = returns.rolling(window=window, min_periods=max(1, window//2)).std()
        
        # 填充缺失值并限制极值
        volatility = volatility.fillna(0.0)
        volatility = self._clip_extreme_values(volatility)
        
        return volatility
    
    def _clip_extreme_values(self, values: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """截断极值"""
        if isinstance(values, (pd.Series, np.ndarray)):
            clipped = np.clip(values, -self.max_value, self.max_value)
        else:
            clipped = np.clip(values, -self.max_value, self.max_value)
        
        return clipped
    
    def _handle_nan_inf(self, values: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
        """处理NaN和无穷值"""
        if isinstance(values, pd.Series):
            # 处理无穷值
            values = values.replace([np.inf, -np.inf], self.inf_replacement)
            # 处理NaN值
            values = values.fillna(self.inf_replacement)
        elif isinstance(values, np.ndarray):
            # 处理无穷值
            values = np.where(np.isinf(values), self.inf_replacement, values)
            # 处理NaN值
            values = np.where(np.isnan(values), self.inf_replacement, values)
        else:
            # 标量处理
            if np.isnan(values) or np.isinf(values):
                values = self.inf_replacement
        
        return values
    
    def validate_dataframe(self, df: pd.DataFrame, name: str = "DataFrame") -> Tuple[bool, dict]:
        """
        验证DataFrame的数值稳定性
        
        Args:
            df: 待验证的DataFrame
            name: DataFrame名称用于日志
            
        Returns:
            (is_valid, stats): 验证结果和统计信息
        """
        stats = {
            'total_values': df.size,
            'nan_count': df.isna().sum().sum(),
            'inf_count': np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
            'zero_count': (df == 0).sum().sum(),
            'negative_count': (df < 0).sum().sum(),
            'extreme_values_count': 0
        }
        
        # 检查极值
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            extreme_mask = (np.abs(numeric_df) > self.max_value)
            stats['extreme_values_count'] = extreme_mask.sum().sum()
        
        # 判断是否有效
        is_valid = (
            stats['nan_count'] == 0 and 
            stats['inf_count'] == 0 and
            stats['extreme_values_count'] == 0
        )
        
        if not is_valid:
            logger.warning(f"{name} 数值稳定性问题: "
                          f"NaN={stats['nan_count']}, "
                          f"Inf={stats['inf_count']}, "
                          f"Extreme={stats['extreme_values_count']}")
        
        return is_valid, stats
    
    def clean_dataframe(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        清理DataFrame中的数值稳定性问题
        
        Args:
            df: 待清理的DataFrame
            inplace: 是否原地修改
            
        Returns:
            清理后的DataFrame
        """
        if not inplace:
            df = df.copy()
        
        # 获取数值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # 处理无穷值
            df[col] = df[col].replace([np.inf, -np.inf], self.inf_replacement)
            
            # 截断极值
            df[col] = np.clip(df[col], -self.max_value, self.max_value)
            
            # 填充NaN值
            df[col] = df[col].fillna(self.inf_replacement)
        
        return df


# 创建全局实例
_global_protector = NumericalStabilityProtector()


# 便捷函数
def safe_divide(numerator, denominator, fill_value=0.0):
    """全局安全除法"""
    return _global_protector.safe_divide(numerator, denominator, fill_value)


def safe_log(values, base=None):
    """全局安全对数"""
    return _global_protector.safe_log(values, base)


def safe_sqrt(values):
    """全局安全开方"""
    return _global_protector.safe_sqrt(values)


def safe_momentum(prices, window):
    """全局安全动量计算"""
    return _global_protector.safe_momentum(prices, window)


def safe_returns(prices, periods=1):
    """全局安全收益率计算"""
    return _global_protector.safe_returns(prices, periods)


def validate_numerical_stability(df, name="DataFrame"):
    """全局数值稳定性验证"""
    return _global_protector.validate_dataframe(df, name)


def clean_numerical_issues(df, inplace=False):
    """全局数值问题清理"""
    return _global_protector.clean_dataframe(df, inplace)