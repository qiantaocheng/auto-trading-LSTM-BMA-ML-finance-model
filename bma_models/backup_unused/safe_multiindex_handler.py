#!/usr/bin/env python3
"""
安全的MultiIndex操作处理器
防止数据破坏和丢失
"""

import pandas as pd
import numpy as np
import logging
from typing import Callable, Any, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataIntegrityCheck:
    """数据完整性检查结果"""
    original_shape: Tuple[int, int]
    result_shape: Tuple[int, int]
    original_tickers: int
    result_tickers: int
    data_loss_ratio: float
    ticker_loss_ratio: float
    is_safe: bool
    warnings: list

class SafeMultiIndexHandler:
    """安全的MultiIndex数据处理器"""
    
    def __init__(self, 
                 max_data_loss_ratio: float = 0.05,
                 max_ticker_loss_ratio: float = 0.1,
                 min_tickers: int = 2):
        self.max_data_loss_ratio = max_data_loss_ratio
        self.max_ticker_loss_ratio = max_ticker_loss_ratio
        self.min_tickers = min_tickers
    
    def safe_operation(self, df: pd.DataFrame, operation_func: Callable, 
                      operation_name: str = "MultiIndex operation",
                      **kwargs) -> pd.DataFrame:
        """
        安全执行MultiIndex操作
        
        Args:
            df: 输入DataFrame
            operation_func: 要执行的操作函数
            operation_name: 操作名称(用于日志)
            **kwargs: 传递给操作函数的参数
            
        Returns:
            处理后的DataFrame，失败时返回原始数据
        """
        # 保存原始状态
        checkpoint = self._create_checkpoint(df)
        
        try:
            logger.info(f"执行安全MultiIndex操作: {operation_name}")
            logger.debug(f"原始数据: {checkpoint.original_shape}, "
                        f"股票数: {checkpoint.original_tickers}")
            
            # 执行操作
            result = operation_func(df, **kwargs)
            
            # 验证结果
            integrity_check = self._check_data_integrity(checkpoint, result)
            
            if integrity_check.is_safe:
                logger.info(f"✅ {operation_name} 成功完成")
                if integrity_check.warnings:
                    for warning in integrity_check.warnings:
                        logger.warning(f"⚠️ {warning}")
                return result
            else:
                logger.error(f"❌ {operation_name} 数据完整性检查失败")
                logger.error(f"数据丢失: {integrity_check.data_loss_ratio:.2%}, "
                           f"股票丢失: {integrity_check.ticker_loss_ratio:.2%}")
                return df  # 返回原始数据
                
        except Exception as e:
            logger.error(f"❌ {operation_name} 执行异常: {e}")
            logger.error("返回原始数据以避免系统崩溃")
            return df
    
    def _create_checkpoint(self, df: pd.DataFrame) -> DataIntegrityCheck:
        """创建数据检查点"""
        original_shape = df.shape
        
        if isinstance(df.index, pd.MultiIndex):
            # 获取股票数量
            if 'ticker' in df.index.names:
                original_tickers = len(df.index.get_level_values('ticker').unique())
            elif df.index.nlevels >= 2:
                original_tickers = len(df.index.get_level_values(1).unique())
            else:
                original_tickers = 1
        else:
            original_tickers = 1
        
        return DataIntegrityCheck(
            original_shape=original_shape,
            result_shape=(0, 0),  # 将在检查时更新
            original_tickers=original_tickers,
            result_tickers=0,  # 将在检查时更新
            data_loss_ratio=0.0,
            ticker_loss_ratio=0.0,
            is_safe=True,
            warnings=[]
        )
    
    def _check_data_integrity(self, checkpoint: DataIntegrityCheck, 
                            result: pd.DataFrame) -> DataIntegrityCheck:
        """检查数据完整性"""
        result_shape = result.shape
        
        # 计算股票数量
        if isinstance(result.index, pd.MultiIndex):
            if 'ticker' in result.index.names:
                result_tickers = len(result.index.get_level_values('ticker').unique())
            elif result.index.nlevels >= 2:
                result_tickers = len(result.index.get_level_values(1).unique())
            else:
                result_tickers = 1
        else:
            result_tickers = 1
        
        # 计算损失比例
        data_loss_ratio = 1.0 - (result_shape[0] / max(checkpoint.original_shape[0], 1))
        ticker_loss_ratio = 1.0 - (result_tickers / max(checkpoint.original_tickers, 1))
        
        # 更新检查点
        checkpoint.result_shape = result_shape
        checkpoint.result_tickers = result_tickers
        checkpoint.data_loss_ratio = data_loss_ratio
        checkpoint.ticker_loss_ratio = ticker_loss_ratio
        checkpoint.warnings = []
        
        # 安全性检查
        is_safe = True
        
        if data_loss_ratio > self.max_data_loss_ratio:
            checkpoint.warnings.append(
                f"数据丢失超过阈值: {data_loss_ratio:.2%} > {self.max_data_loss_ratio:.2%}"
            )
            is_safe = False
        
        if ticker_loss_ratio > self.max_ticker_loss_ratio:
            checkpoint.warnings.append(
                f"股票丢失超过阈值: {ticker_loss_ratio:.2%} > {self.max_ticker_loss_ratio:.2%}"
            )
            is_safe = False
        
        if result_tickers < self.min_tickers:
            checkpoint.warnings.append(
                f"股票数不足: {result_tickers} < {self.min_tickers}"
            )
            is_safe = False
        
        # 轻微损失的警告
        if 0 < data_loss_ratio <= self.max_data_loss_ratio:
            checkpoint.warnings.append(
                f"轻微数据丢失: {data_loss_ratio:.2%}"
            )
        
        if 0 < ticker_loss_ratio <= self.max_ticker_loss_ratio:
            checkpoint.warnings.append(
                f"轻微股票减少: {ticker_loss_ratio:.2%}"
            )
        
        checkpoint.is_safe = is_safe
        return checkpoint

def safe_datetime_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    安全的MultiIndex日期转换
    """
    handler = SafeMultiIndexHandler()
    
    def _convert_dates(data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data.index, pd.MultiIndex):
            return data
        
        if 'date' not in data.index.names:
            return data
        
        # 检查date层级是否已经是datetime类型
        date_level_values = data.index.get_level_values('date')
        if pd.api.types.is_datetime64_any_dtype(date_level_values):
            logger.debug("date层级已经是datetime类型，跳过转换")
            return data
        
        logger.info("转换date层级为datetime...")
        
        # 安全的重建方式：使用set_levels而不是重建整个index
        date_level = data.index.names.index('date')
        new_levels = list(data.index.levels)
        new_levels[date_level] = pd.to_datetime(data.index.levels[date_level])
        
        # 使用set_levels方法，保持codes不变
        new_index = data.index.set_levels(new_levels)
        
        result = data.copy()
        result.index = new_index
        
        return result.sort_index()
    
    return handler.safe_operation(
        df, _convert_dates, 
        "MultiIndex日期转换"
    )

# 全局安全处理器实例
DEFAULT_HANDLER = SafeMultiIndexHandler()

# 便捷函数
def safe_multiindex_operation(df: pd.DataFrame, operation_func: Callable,
                             operation_name: str = "operation", **kwargs) -> pd.DataFrame:
    """全局便捷函数"""
    return DEFAULT_HANDLER.safe_operation(df, operation_func, operation_name, **kwargs)