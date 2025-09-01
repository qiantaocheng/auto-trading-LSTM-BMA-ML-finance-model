#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的错误处理系统
提供统一的错误处理、日志记录和恢复机制
"""

import logging
import traceback
import functools
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass
import warnings

# 配置logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ErrorHandlingConfig:
    """错误处理配置类"""
    max_retries: int = 3
    retry_delay: float = 1.0
    log_level: str = "INFO"
    enable_fallback: bool = True
    fallback_strategy: str = "zero"  # zero, nan, last_valid
    max_memory_mb: float = 1000.0

class BMAErrorHandler:
    """BMA Enhanced系统的统一错误处理器"""
    
    def __init__(self, log_level: str = "INFO"):
        self.log_level = log_level
        self.error_counts = {}
        self.error_history = []
        
    def handle_error(
        self, 
        error: Exception, 
        context: str = "",
        fallback_value: Any = None,
        reraise: bool = False
    ) -> Any:
        """
        统一错误处理
        
        Args:
            error: 异常对象
            context: 错误上下文描述
            fallback_value: 错误时返回的默认值
            reraise: 是否重新抛出异常
            
        Returns:
            fallback_value 或重新抛出异常
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        # 记录错误统计
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # 记录错误历史
        self.error_history.append({
            'timestamp': datetime.now(),
            'error_type': error_type,
            'message': error_msg,
            'context': context,
            'traceback': traceback.format_exc()
        })
        
        # 限制历史记录长度
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]
        
        # 根据错误类型决定日志级别
        if error_type in ['KeyError', 'ValueError', 'TypeError']:
            logger.error(f"{context} - {error_type}: {error_msg}")
        elif error_type in ['RuntimeWarning', 'UserWarning']:
            logger.warning(f"{context} - {error_type}: {error_msg}")
        else:
            logger.critical(f"{context} - {error_type}: {error_msg}")
            if self.log_level == "DEBUG":
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
        
        if reraise:
            raise error
        
        return fallback_value
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误统计摘要"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_types': dict(self.error_counts),
            'recent_errors': self.error_history[-10:] if self.error_history else [],
            'error_rate': len(self.error_history) / max(1, len(self.error_history))
        }

# 全局错误处理器实例
global_error_handler = BMAErrorHandler()

def robust_calculation(
    fallback_value: Any = None,
    log_context: str = "",
    suppress_warnings: bool = True
):
    """
    稳健计算装饰器
    自动处理数值计算中的异常
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if suppress_warnings:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        context = log_context or f"Function: {func.__name__}"
                        return global_error_handler.handle_error(
                            e, context, fallback_value
                        )
            else:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = log_context or f"Function: {func.__name__}"
                    return global_error_handler.handle_error(
                        e, context, fallback_value
                    )
        return wrapper
    return decorator

def safe_divide(a: float, b: float, fallback: float = 0.0) -> float:
    """安全除法，避免除零错误"""
    try:
        if abs(b) < 1e-10:
            return fallback
        result = a / b
        return result if np.isfinite(result) else fallback
    except:
        return fallback

def safe_log(x: float, fallback: float = 0.0) -> float:
    """安全对数，处理负数和零"""
    try:
        if x <= 0:
            return fallback
        result = np.log(x)
        return result if np.isfinite(result) else fallback
    except:
        return fallback

def safe_sqrt(x: float, fallback: float = 0.0) -> float:
    """安全平方根，处理负数"""
    try:
        if x < 0:
            return fallback
        result = np.sqrt(x)
        return result if np.isfinite(result) else fallback
    except:
        return fallback

def clean_numeric_series(series: pd.Series, method: str = "zero") -> pd.Series:
    """
    清理数值序列中的异常值
    
    Args:
        series: 输入序列
        method: 清理方法 ('zero', 'mean', 'median', 'drop')
    """
    try:
        cleaned = series.copy()
        
        # 处理无穷大和NaN
        if method == "zero":
            cleaned = cleaned.replace([np.inf, -np.inf], 0).fillna(0)
        elif method == "mean":
            finite_mean = cleaned[np.isfinite(cleaned)].mean()
            cleaned = cleaned.replace([np.inf, -np.inf], finite_mean).fillna(finite_mean)
        elif method == "median":
            finite_median = cleaned[np.isfinite(cleaned)].median()
            cleaned = cleaned.replace([np.inf, -np.inf], finite_median).fillna(finite_median)
        elif method == "drop":
            cleaned = cleaned[np.isfinite(cleaned)]
        
        return cleaned
        
    except Exception as e:
        global_error_handler.handle_error(e, "clean_numeric_series")
        return pd.Series(dtype=float)

def validate_dataframe(df: pd.DataFrame, required_columns: Optional[list] = None) -> bool:
    """
    验证DataFrame完整性
    
    Args:
        df: 待验证的DataFrame
        required_columns: 必需的列名列表
        
    Returns:
        验证是否通过
    """
    try:
        if df is None or df.empty:
            return False
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}")
                return False
        
        # 检查是否有足够的非空数据
        if df.dropna().empty:
            logger.warning("DataFrame contains only NaN values")
            return False
        
        return True
        
    except Exception as e:
        global_error_handler.handle_error(e, "validate_dataframe")
        return False

def memory_efficient_operation(max_memory_mb: float = 1000):
    """
    内存高效操作装饰器
    监控内存使用并在必要时进行清理
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import psutil
            import gc
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                
                current_memory = process.memory_info().rss / 1024 / 1024
                
                if current_memory > max_memory_mb:
                    logger.warning(
                        f"Memory usage high: {current_memory:.1f}MB, triggering cleanup"
                    )
                    gc.collect()
                
                return result
                
            except MemoryError as e:
                logger.error(f"Memory error in {func.__name__}: {e}")
                gc.collect()
                raise
            except Exception as e:
                global_error_handler.handle_error(e, f"memory_efficient_operation: {func.__name__}")
                raise
                
        return wrapper
    return decorator

class RobustDataProcessor:
    """稳健的数据处理器"""
    
    @staticmethod
    @robust_calculation(fallback_value=pd.DataFrame(), log_context="RobustDataProcessor.process")
    def safe_merge(left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """安全的数据合并"""
        if not validate_dataframe(left) or not validate_dataframe(right):
            return pd.DataFrame()
        
        return pd.merge(left, right, **kwargs)
    
    @staticmethod
    @robust_calculation(fallback_value=pd.Series(dtype=float), log_context="RobustDataProcessor.rolling")
    def safe_rolling_apply(series: pd.Series, window: int, func: Callable) -> pd.Series:
        """安全的滚动计算"""
        cleaned_series = clean_numeric_series(series)
        if len(cleaned_series) < window:
            return pd.Series(dtype=float)
        
        return cleaned_series.rolling(window=window, min_periods=1).apply(func)
    
    @staticmethod  
    @robust_calculation(fallback_value=0.0, log_context="RobustDataProcessor.correlation")
    def safe_correlation(x: pd.Series, y: pd.Series, method: str = 'pearson') -> float:
        """安全的相关系数计算"""
        # 清理数据
        x_clean = clean_numeric_series(x)
        y_clean = clean_numeric_series(y)
        
        # 对齐数据
        aligned_data = pd.DataFrame({'x': x_clean, 'y': y_clean}).dropna()
        
        if len(aligned_data) < 10:  # 最小样本要求
            return 0.0
        
        return aligned_data['x'].corr(aligned_data['y'], method=method)


def create_error_handler(log_level: str = "INFO") -> BMAErrorHandler:
    """创建错误处理器的工厂函数"""
    return BMAErrorHandler(log_level=log_level)

# 导出主要组件
__all__ = [
    'BMAErrorHandler',
    'global_error_handler', 
    'robust_calculation',
    'safe_divide',
    'safe_log', 
    'safe_sqrt',
    'clean_numeric_series',
    'validate_dataframe',
    'memory_efficient_operation',
    'RobustDataProcessor',
    'create_error_handler'
]