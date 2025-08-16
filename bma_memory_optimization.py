#!/usr/bin/env python3
"""
BMA Enhanced 内存优化模块
解决量化模型中的内存泄漏和内存增长问题
"""

import pandas as pd
import numpy as np
import gc
import logging
from typing import Dict, List, Optional, Any
import weakref
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MemoryOptimizedBMA:
    """内存优化的BMA增强模型辅助类"""
    
    def __init__(self):
        self.memory_tracker = {}
        self.temp_objects = weakref.WeakSet()
        
    @contextmanager
    def memory_context(self, context_name: str):
        """内存上下文管理器，自动清理临时对象"""
        initial_memory = self._get_memory_usage()
        temp_vars = []
        
        try:
            yield temp_vars
        finally:
            # 清理临时变量
            for var in temp_vars:
                if hasattr(var, '__del__'):
                    del var
            
            # 强制垃圾回收
            gc.collect()
            
            final_memory = self._get_memory_usage()
            memory_delta = final_memory - initial_memory
            
            if memory_delta > 50:  # 50MB增长
                logger.warning(f"{context_name} 内存增长: {memory_delta:.1f}MB")
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量 (MB)"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def optimize_dataframe_memory(self, df: pd.DataFrame, 
                                optimize_dtypes: bool = True) -> pd.DataFrame:
        """优化DataFrame内存使用"""
        if df.empty:
            return df
        
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        if optimize_dtypes:
            # 优化数据类型
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        # 尝试转换为数值类型
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except:
                        pass
                elif df[col].dtype == 'float64':
                    # 降精度到float32
                    if df[col].abs().max() < 1e6:  # 小数值可以用float32
                        df[col] = df[col].astype('float32')
                elif df[col].dtype == 'int64':
                    # 降精度到合适的int类型
                    col_min, col_max = df[col].min(), df[col].max()
                    if col_min >= -128 and col_max <= 127:
                        df[col] = df[col].astype('int8')
                    elif col_min >= -32768 and col_max <= 32767:
                        df[col] = df[col].astype('int16')
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        df[col] = df[col].astype('int32')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        memory_saved = original_memory - final_memory
        
        if memory_saved > 1:  # 节省超过1MB
            logger.info(f"DataFrame内存优化: {original_memory:.1f}MB -> {final_memory:.1f}MB "
                       f"(节省{memory_saved:.1f}MB)")
        
        return df
    
    def chunk_feature_engineering(self, raw_data: Dict[str, pd.DataFrame], 
                                chunk_size: int = 50) -> pd.DataFrame:
        """分块特征工程，避免内存峰值"""
        logger.info(f"开始分块特征工程，块大小: {chunk_size}")
        
        tickers = list(raw_data.keys())
        all_features = []
        
        # 按块处理
        for i in range(0, len(tickers), chunk_size):
            chunk_tickers = tickers[i:i + chunk_size]
            logger.info(f"处理第{i//chunk_size + 1}块: {len(chunk_tickers)}只股票")
            
            with self.memory_context(f"chunk_{i//chunk_size}"):
                chunk_features = []
                
                for ticker in chunk_tickers:
                    df = raw_data[ticker]
                    
                    # 就地特征工程（避免复制）
                    df_features = self._inplace_feature_engineering(df, ticker)
                    
                    # 立即优化内存
                    df_features = self.optimize_dataframe_memory(df_features)
                    
                    chunk_features.append(df_features)
                
                # 合并当前块
                if chunk_features:
                    chunk_combined = pd.concat(chunk_features, ignore_index=True)
                    chunk_combined = self.optimize_dataframe_memory(chunk_combined)
                    all_features.append(chunk_combined)
                    
                    # 清理临时变量
                    del chunk_features
                    gc.collect()
        
        # 最终合并
        if all_features:
            logger.info("合并所有特征块...")
            final_features = pd.concat(all_features, ignore_index=True)
            final_features = self.optimize_dataframe_memory(final_features)
            
            # 清理
            del all_features
            gc.collect()
            
            return final_features
        
        return pd.DataFrame()
    
    def _inplace_feature_engineering(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """就地特征工程，最小化内存复制"""
        # 直接在原DataFrame上操作，避免copy()
        df = df.sort_values('date')
        
        # 基础特征
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 移动平均（只保留关键窗口）
        key_windows = [20, 50]  # 减少窗口数量
        for window in key_windows:
            df[f'ma_{window}'] = df['close'].rolling(window, min_periods=window//2).mean()
            df[f'ma_ratio_{window}'] = df['close'] / df[f'ma_{window}']
        
        # 波动率（减少计算）
        df['vol_20'] = df['log_returns'].rolling(20, min_periods=10).std()
        
        # RSI（简化计算）
        try:
            df['rsi_14'] = self._fast_rsi(df['close'], 14)
        except:
            df['rsi_14'] = 50.0  # 默认值
        
        # 成交量特征（如果存在）
        if 'volume' in df.columns:
            df['volume_ma_20'] = df['volume'].rolling(20, min_periods=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # 价格位置
        df['price_position_20'] = self._price_position(df, 20)
        
        # 动量（只保留关键期）
        for period in [10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # 目标变量（简化）
        PRED_START, PRED_END = 1, 5
        df['target'] = (df['close'].shift(-PRED_END) / df['close'].shift(-PRED_START + 1) - 1)
        
        # 添加元数据
        df['ticker'] = ticker
        df['date'] = df.index
        df['COUNTRY'] = 'US'
        df['SECTOR'] = ticker[:2] if len(ticker) >= 2 else 'TECH'
        df['SUBINDUSTRY'] = ticker[:3] if len(ticker) >= 3 else 'SOFTWARE'
        
        return df
    
    def _fast_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """快速RSI计算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=window//2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=window//2).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def _price_position(self, df: pd.DataFrame, window: int) -> pd.Series:
        """价格位置计算"""
        high_roll = df['high'].rolling(window, min_periods=window//2).max()
        low_roll = df['low'].rolling(window, min_periods=window//2).min()
        return (df['close'] - low_roll) / (high_roll - low_roll + 1e-8)
    
    def memory_efficient_concat(self, dataframes: List[pd.DataFrame], 
                              batch_size: int = 10) -> pd.DataFrame:
        """内存高效的DataFrame合并"""
        if not dataframes:
            return pd.DataFrame()
        
        if len(dataframes) <= batch_size:
            return pd.concat(dataframes, ignore_index=True)
        
        # 分批合并
        batches = []
        for i in range(0, len(dataframes), batch_size):
            batch = dataframes[i:i + batch_size]
            batch_combined = pd.concat(batch, ignore_index=True)
            batch_combined = self.optimize_dataframe_memory(batch_combined)
            batches.append(batch_combined)
            
            # 清理临时对象
            del batch
            gc.collect()
        
        # 最终合并
        final_result = pd.concat(batches, ignore_index=True)
        final_result = self.optimize_dataframe_memory(final_result)
        
        del batches
        gc.collect()
        
        return final_result
    
    def cleanup_dataframe_columns(self, df: pd.DataFrame, 
                                keep_patterns: List[str] = None) -> pd.DataFrame:
        """清理不必要的DataFrame列"""
        if keep_patterns is None:
            keep_patterns = ['target', 'ticker', 'date', 'close', 'returns', 
                           'ma_', 'vol_', 'rsi', 'momentum_', 'COUNTRY', 'SECTOR']
        
        # 找出要保留的列
        keep_cols = []
        for col in df.columns:
            if any(pattern in col for pattern in keep_patterns):
                keep_cols.append(col)
        
        # 删除不必要的列
        dropped_cols = set(df.columns) - set(keep_cols)
        if dropped_cols:
            logger.info(f"删除不必要的列: {len(dropped_cols)}个")
            df = df[keep_cols].copy()
            gc.collect()
        
        return df
    
    def force_memory_cleanup(self):
        """强制内存清理"""
        # 清理弱引用集合
        self.temp_objects.clear()
        
        # 多轮垃圾回收
        for _ in range(3):
            collected = gc.collect()
            if collected == 0:
                break
        
        logger.info("强制内存清理完成")


def create_memory_optimized_feature_engineering():
    """创建内存优化的特征工程函数"""
    optimizer = MemoryOptimizedBMA()
    
    def optimized_feature_engineering(raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """内存优化的特征工程主函数"""
        logger.info(f"开始内存优化特征工程，股票数量: {len(raw_data)}")
        
        # 使用分块处理
        result = optimizer.chunk_feature_engineering(raw_data, chunk_size=30)
        
        # 清理不必要的列
        result = optimizer.cleanup_dataframe_columns(result)
        
        # 最终内存优化
        result = optimizer.optimize_dataframe_memory(result)
        
        # 强制清理
        optimizer.force_memory_cleanup()
        
        logger.info(f"特征工程完成，最终大小: {result.shape}")
        return result
    
    return optimized_feature_engineering


# 使用示例和测试
if __name__ == "__main__":
    # 测试内存优化功能
    optimizer = MemoryOptimizedBMA()
    
    # 创建测试数据
    import datetime
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    
    test_data = {}
    for i, ticker in enumerate(['AAPL', 'MSFT', 'GOOGL']):
        df = pd.DataFrame({
            'date': dates,
            'open': 100 + np.random.randn(len(dates)).cumsum(),
            'high': 100 + np.random.randn(len(dates)).cumsum(),
            'low': 100 + np.random.randn(len(dates)).cumsum(),
            'close': 100 + np.random.randn(len(dates)).cumsum(),
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'amount': np.random.randint(10000000, 100000000, len(dates))
        })
        df.set_index('date', inplace=True)
        test_data[ticker] = df
    
    print("测试内存优化特征工程...")
    
    # 创建优化的特征工程函数
    optimized_fe = create_memory_optimized_feature_engineering()
    
    # 运行测试
    result = optimized_fe(test_data)
    
    print(f"结果形状: {result.shape}")
    print(f"内存使用: {result.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB")
    print("内存优化测试完成")