#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一异常处理框架
集中管理所有异常处理逻辑
"""

import logging
import traceback
import functools
from typing import Any, Callable, Optional, Dict
from contextlib import contextmanager
from dataclasses import dataclass
import time
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ErrorHandlingConfig:
    """异常处理配置"""
    enable_retry: bool = True
    max_retries: int = 3
    enable_strict_mode: bool = True
    log_errors: bool = True
    raise_on_critical: bool = True
    retry_delay: float = 1.0
    exponential_backoff: bool = True


class UnifiedExceptionHandler:
    """统一异常处理器"""
    
    def __init__(self, config: Optional[ErrorHandlingConfig] = None):
        """
        初始化异常处理器
        
        Args:
            config: 异常处理配置
        """
        self.config = config or ErrorHandlingConfig()
        self.error_count = 0
        self.error_history = []
        self.retry_count = {}
        
    @contextmanager
    def safe_execution(self, operation_name: str, default_value: Any = None):
        """
        安全执行上下文管理器
        
        Args:
            operation_name: 操作名称
            default_value: 失败时的默认值
        """
        try:
            if self.config.log_errors:
                logger.debug(f"开始执行: {operation_name}")
            yield
            if self.config.log_errors:
                logger.debug(f"成功完成: {operation_name}")
        except Exception as e:
            self.error_count += 1
            error_info = {
                'operation': operation_name,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': time.time()
            }
            self.error_history.append(error_info)
            
            if self.config.log_errors:
                logger.error(f"{operation_name} 失败: {e}")
                logger.debug(traceback.format_exc())
            
            if self.config.raise_on_critical and self._is_critical_error(e):
                raise
            
            if not self.config.enable_strict_mode and default_value is not None:
                logger.info(f"使用默认值: {default_value}")
                return default_value
    
    def with_retry(self, func: Callable) -> Callable:
        """
        重试装饰器
        
        Args:
            func: 要装饰的函数
            
        Returns:
            装饰后的函数
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = func.__name__
            
            if not self.config.enable_retry:
                return func(*args, **kwargs)
            
            for attempt in range(self.config.max_retries):
                try:
                    if attempt > 0:
                        delay = self._calculate_delay(attempt)
                        logger.info(f"重试 {operation_name} (尝试 {attempt + 1}/{self.config.max_retries})，等待 {delay:.1f}秒")
                        time.sleep(delay)
                    
                    result = func(*args, **kwargs)
                    
                    if attempt > 0:
                        logger.info(f"{operation_name} 在第 {attempt + 1} 次尝试成功")
                    
                    return result
                    
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        logger.error(f"{operation_name} 在 {self.config.max_retries} 次尝试后失败")
                        raise
                    
                    if self._is_non_retryable_error(e):
                        logger.error(f"{operation_name} 遇到不可重试错误: {e}")
                        raise
                    
                    logger.warning(f"{operation_name} 失败 (尝试 {attempt + 1}): {e}")
            
            return None  # Should never reach here
        
        return wrapper
    
    def handle_data_error(self, data: Any, operation: str) -> Any:
        """
        处理数据错误
        
        Args:
            data: 输入数据
            operation: 操作名称
            
        Returns:
            清洗后的数据或None
        """
        try:
            if data is None:
                logger.warning(f"{operation}: 数据为None")
                return None
            
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    logger.warning(f"{operation}: DataFrame为空")
                    return data
                
                # 检查并修复常见数据问题
                if data.isnull().all().all():
                    logger.warning(f"{operation}: DataFrame全为NaN")
                    return pd.DataFrame()
                
                # 移除全NaN列
                data = data.dropna(axis=1, how='all')
                
                # 填充部分NaN
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    data[numeric_cols] = data[numeric_cols].fillna(method='ffill').fillna(0)
                
                return data
                
            elif isinstance(data, pd.Series):
                if data.empty:
                    logger.warning(f"{operation}: Series为空")
                    return data
                
                if data.isnull().all():
                    logger.warning(f"{operation}: Series全为NaN")
                    return pd.Series(dtype=float)
                
                return data.fillna(method='ffill').fillna(0)
                
            elif isinstance(data, np.ndarray):
                if data.size == 0:
                    logger.warning(f"{operation}: 数组为空")
                    return data
                
                if np.all(np.isnan(data)):
                    logger.warning(f"{operation}: 数组全为NaN")
                    return np.array([])
                
                return np.nan_to_num(data, nan=0.0)
                
            else:
                return data
                
        except Exception as e:
            logger.error(f"{operation} 数据处理失败: {e}")
            return None
    
    def _calculate_delay(self, attempt: int) -> float:
        """计算重试延迟"""
        if self.config.exponential_backoff:
            return self.config.retry_delay * (2 ** attempt)
        else:
            return self.config.retry_delay
    
    def _is_critical_error(self, error: Exception) -> bool:
        """判断是否为关键错误"""
        critical_errors = [
            MemoryError,
            SystemError,
            KeyboardInterrupt,
            SystemExit
        ]
        return any(isinstance(error, err_type) for err_type in critical_errors)
    
    def _is_non_retryable_error(self, error: Exception) -> bool:
        """判断是否为不可重试错误"""
        non_retryable = [
            ValueError,  # 参数错误
            TypeError,   # 类型错误
            AttributeError,  # 属性错误
            ImportError,  # 导入错误
        ]
        return any(isinstance(error, err_type) for err_type in non_retryable)
    
    def get_error_summary(self) -> Dict:
        """获取错误摘要"""
        if not self.error_history:
            return {
                'total_errors': 0,
                'error_rate': 0.0,
                'most_common': None
            }
        
        operations = [err['operation'] for err in self.error_history]
        from collections import Counter
        operation_counts = Counter(operations)
        
        return {
            'total_errors': self.error_count,
            'unique_operations': len(set(operations)),
            'most_common': operation_counts.most_common(5),
            'recent_errors': self.error_history[-5:] if len(self.error_history) > 5 else self.error_history
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.error_count = 0
        self.error_history = []
        self.retry_count = {}
        logger.info("异常处理统计已重置")


# 全局异常处理器实例
_global_handler = None


def get_global_exception_handler(config: Optional[ErrorHandlingConfig] = None) -> UnifiedExceptionHandler:
    """获取全局异常处理器实例"""
    global _global_handler
    if _global_handler is None:
        _global_handler = UnifiedExceptionHandler(config)
    return _global_handler


def safe_execute(operation_name: str, default_value: Any = None):
    """
    装饰器：安全执行函数
    
    Args:
        operation_name: 操作名称
        default_value: 失败时的默认值
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = get_global_exception_handler()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler.error_count += 1
                logger.error(f"{operation_name} 失败: {e}")
                if default_value is not None:
                    return default_value
                raise
        return wrapper
    return decorator


def with_automatic_retry(max_retries: int = 3, delay: float = 1.0):
    """
    装饰器：自动重试
    
    Args:
        max_retries: 最大重试次数
        delay: 重试延迟
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator