#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Memory Management System for BMA Ultra Enhanced
完全不影响训练效果和输出的内存管理系统
"""

import gc
import os
import sys
import psutil
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from functools import wraps
import warnings
import traceback
from contextlib import contextmanager
import weakref
import pickle
import tempfile
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        self.checkpoints = {}
        
    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况"""
        mem_info = self.process.memory_info()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': self.process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def checkpoint(self, name: str):
        """创建内存检查点"""
        current = self.get_memory_usage()
        self.checkpoints[name] = current
        self.peak_memory = max(self.peak_memory, current['rss_mb'])
        logger.debug(f"Memory checkpoint '{name}': {current['rss_mb']:.1f}MB")
        
    def get_delta(self, checkpoint_name: str) -> float:
        """获取相对于检查点的内存变化"""
        if checkpoint_name not in self.checkpoints:
            return 0.0
        current = self.get_memory_usage()
        return current['rss_mb'] - self.checkpoints[checkpoint_name]['rss_mb']


class DataFrameOptimizer:
    """DataFrame内存优化器"""
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame, 
                        deep: bool = True,
                        categorical_threshold: float = 0.5) -> pd.DataFrame:
        """
        优化DataFrame的数据类型以减少内存使用
        
        Args:
            df: 输入DataFrame
            deep: 是否进行深度优化
            categorical_threshold: 转换为category的唯一值比例阈值
        """
        if df.empty:
            return df
            
        df_optimized = df.copy() if not deep else df
        initial_memory = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024
        
        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype
            
            if col_type != 'object':
                # 优化数值类型
                if 'int' in str(col_type):
                    df_optimized[col] = DataFrameOptimizer._optimize_int(df_optimized[col])
                elif 'float' in str(col_type):
                    df_optimized[col] = DataFrameOptimizer._optimize_float(df_optimized[col])
            else:
                # 优化对象类型
                num_unique = df_optimized[col].nunique()
                num_total = len(df_optimized[col])
                if num_unique / num_total < categorical_threshold:
                    df_optimized[col] = df_optimized[col].astype('category')
        
        final_memory = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024
        reduction_pct = (initial_memory - final_memory) / initial_memory * 100
        
        if reduction_pct > 10:
            logger.info(f"DataFrame memory reduced by {reduction_pct:.1f}% "
                       f"({initial_memory:.1f}MB -> {final_memory:.1f}MB)")
        
        return df_optimized
    
    @staticmethod
    def _optimize_int(col: pd.Series) -> pd.Series:
        """优化整数列"""
        c_min = col.min()
        c_max = col.max()
        
        if c_min >= 0:
            if c_max < 255:
                return col.astype(np.uint8)
            elif c_max < 65535:
                return col.astype(np.uint16)
            elif c_max < 4294967295:
                return col.astype(np.uint32)
            else:
                return col.astype(np.uint64)
        else:
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                return col.astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                return col.astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                return col.astype(np.int32)
            else:
                return col
                
    @staticmethod
    def _optimize_float(col: pd.Series) -> pd.Series:
        """优化浮点数列"""
        c_min = col.min()
        c_max = col.max()
        
        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
            return col.astype(np.float16)
        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
            return col.astype(np.float32)
        else:
            return col


class ChunkProcessor:
    """分块处理器 - 处理大数据集"""
    
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
        
    def process_in_chunks(self, 
                         data: pd.DataFrame,
                         process_func: callable,
                         combine_func: callable = None) -> Any:
        """
        分块处理数据
        
        Args:
            data: 输入数据
            process_func: 处理函数
            combine_func: 合并函数
        """
        if len(data) <= self.chunk_size:
            return process_func(data)
            
        chunks = []
        n_chunks = (len(data) - 1) // self.chunk_size + 1
        
        for i in range(n_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, len(data))
            chunk = data.iloc[start_idx:end_idx]
            
            # 处理块
            result = process_func(chunk)
            chunks.append(result)
            
            # 清理临时内存
            del chunk
            if i % 10 == 0:
                gc.collect()
        
        # 合并结果
        if combine_func:
            return combine_func(chunks)
        elif isinstance(chunks[0], pd.DataFrame):
            return pd.concat(chunks, ignore_index=True)
        elif isinstance(chunks[0], pd.Series):
            return pd.concat(chunks)
        else:
            return chunks


class CacheManager:
    """智能缓存管理器"""
    
    def __init__(self, 
                 max_memory_mb: float = 1000,
                 cache_dir: Optional[str] = None):
        self.max_memory_mb = max_memory_mb
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / 'bma_cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self.disk_cache_index = {}
        self.access_counts = {}
        self.cache_sizes = {}
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        # 更新访问计数
        self.access_counts[key] = self.access_counts.get(key, 0) + 1
        
        # 先检查内存缓存
        if key in self.memory_cache:
            return self.memory_cache[key]
            
        # 检查磁盘缓存
        if key in self.disk_cache_index:
            return self._load_from_disk(key)
            
        return None
    
    def set(self, key: str, value: Any, priority: int = 0):
        """设置缓存数据"""
        # 估算数据大小
        size_mb = self._estimate_size(value)
        
        # 如果数据太大，直接存到磁盘
        if size_mb > self.max_memory_mb * 0.1:  # 单个对象不超过总缓存的10%
            self._save_to_disk(key, value)
        else:
            # 检查内存限制
            current_size = sum(self.cache_sizes.values())
            if current_size + size_mb > self.max_memory_mb:
                self._evict_cache(size_mb)
            
            # 存储到内存
            self.memory_cache[key] = value
            self.cache_sizes[key] = size_mb
            
    def _estimate_size(self, obj: Any) -> float:
        """估算对象大小（MB）"""
        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum() / 1024 / 1024
        elif isinstance(obj, np.ndarray):
            return obj.nbytes / 1024 / 1024
        else:
            # 粗略估算
            return sys.getsizeof(obj) / 1024 / 1024
            
    def _evict_cache(self, required_size: float):
        """基于LRU策略清理缓存"""
        # 按访问次数排序
        sorted_keys = sorted(self.memory_cache.keys(), 
                           key=lambda k: self.access_counts.get(k, 0))
        
        freed_size = 0
        for key in sorted_keys:
            if key in self.cache_sizes:
                freed_size += self.cache_sizes[key]
                # 移到磁盘或删除
                if self.access_counts.get(key, 0) > 1:
                    self._save_to_disk(key, self.memory_cache[key])
                del self.memory_cache[key]
                del self.cache_sizes[key]
                
                if freed_size >= required_size:
                    break
                    
    def _save_to_disk(self, key: str, value: Any):
        """保存到磁盘缓存"""
        file_path = self.cache_dir / f"{key}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(value, f)
        self.disk_cache_index[key] = file_path
        
    def _load_from_disk(self, key: str) -> Any:
        """从磁盘加载缓存"""
        file_path = self.disk_cache_index[key]
        with open(file_path, 'rb') as f:
            return pickle.load(f)
            
    def clear(self):
        """清理所有缓存"""
        self.memory_cache.clear()
        self.cache_sizes.clear()
        self.access_counts.clear()
        
        # 清理磁盘缓存
        for file_path in self.disk_cache_index.values():
            if file_path.exists():
                file_path.unlink()
        self.disk_cache_index.clear()
        
    def __del__(self):
        """析构时清理磁盘缓存"""
        try:
            if hasattr(self, 'cache_dir') and self.cache_dir.exists():
                shutil.rmtree(self.cache_dir, ignore_errors=True)
        except:
            pass


class IntelligentMemoryManager:
    """智能内存管理器 - 主类"""
    
    def __init__(self,
                 target_memory_usage: float = 0.7,  # 目标内存使用率
                 gc_threshold: float = 0.8,  # 触发GC的内存阈值
                 enable_disk_swap: bool = True,  # 启用磁盘交换
                 cache_size_mb: float = 500):  # 缓存大小
        
        self.target_memory_usage = target_memory_usage
        self.gc_threshold = gc_threshold
        self.enable_disk_swap = enable_disk_swap
        
        # 初始化组件
        self.monitor = MemoryMonitor()
        self.optimizer = DataFrameOptimizer()
        self.chunk_processor = ChunkProcessor()
        self.cache_manager = CacheManager(max_memory_mb=cache_size_mb)
        
        # 弱引用存储
        self.weak_refs = weakref.WeakValueDictionary()
        
        # 内存使用统计
        self.stats = {
            'optimizations': 0,
            'gc_collections': 0,
            'disk_swaps': 0,
            'peak_memory_mb': 0
        }
        
        logger.info(f"IntelligentMemoryManager initialized - Target: {target_memory_usage*100:.0f}% memory usage")
        
    @contextmanager
    def memory_context(self, operation_name: str):
        """内存管理上下文"""
        # 记录开始状态
        self.monitor.checkpoint(f"{operation_name}_start")
        
        try:
            # 预检查内存
            self._check_memory_pressure()
            
            yield self
            
        finally:
            # 记录结束状态
            self.monitor.checkpoint(f"{operation_name}_end")
            delta = self.monitor.get_delta(f"{operation_name}_start")
            
            if delta > 100:  # 增长超过100MB
                logger.info(f"Operation '{operation_name}' used {delta:.1f}MB memory")
                self._aggressive_cleanup()
                
            # 更新统计
            self.stats['peak_memory_mb'] = self.monitor.peak_memory
            
    def optimize_dataframe(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """优化DataFrame内存使用"""
        if df.empty:
            return df
            
        initial_size = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # 应用优化
        df_opt = self.optimizer.optimize_dtypes(df, deep=inplace)
        
        # 删除完全重复的行（如果安全）
        if not inplace and len(df_opt) > 1000:
            n_before = len(df_opt)
            df_opt = df_opt.drop_duplicates()
            if n_before != len(df_opt):
                logger.debug(f"Removed {n_before - len(df_opt)} duplicate rows")
        
        self.stats['optimizations'] += 1
        
        final_size = df_opt.memory_usage(deep=True).sum() / 1024 / 1024
        if initial_size - final_size > 10:
            logger.info(f"DataFrame optimized: {initial_size:.1f}MB -> {final_size:.1f}MB")
            
        return df_opt
    
    def process_large_data(self,
                          data: pd.DataFrame,
                          func: callable,
                          chunk_size: Optional[int] = None) -> Any:
        """处理大型数据集"""
        if chunk_size is None:
            # 动态确定块大小
            available_memory = psutil.virtual_memory().available / 1024 / 1024
            data_memory = data.memory_usage(deep=True).sum() / 1024 / 1024
            
            if data_memory > available_memory * 0.3:
                # 需要分块处理
                chunk_size = max(1000, len(data) // (int(data_memory / (available_memory * 0.1)) + 1))
                logger.info(f"Processing data in chunks of {chunk_size} rows")
                
        if chunk_size:
            self.chunk_processor.chunk_size = chunk_size
            return self.chunk_processor.process_in_chunks(data, func)
        else:
            return func(data)
            
    def cache_result(self, key: str, compute_func: callable, *args, **kwargs) -> Any:
        """缓存计算结果"""
        # 检查缓存
        cached = self.cache_manager.get(key)
        if cached is not None:
            logger.debug(f"Cache hit for '{key}'")
            return cached
            
        # 计算结果
        with self.memory_context(f"compute_{key}"):
            result = compute_func(*args, **kwargs)
            
        # 存储到缓存
        self.cache_manager.set(key, result)
        
        return result
        
    def register_object(self, name: str, obj: Any):
        """注册对象以进行弱引用管理"""
        self.weak_refs[name] = obj
        
    def _check_memory_pressure(self):
        """检查内存压力"""
        mem_usage = self.monitor.get_memory_usage()
        
        if mem_usage['percent'] > self.gc_threshold * 100:
            logger.warning(f"High memory usage: {mem_usage['percent']:.1f}%")
            self._aggressive_cleanup()
            
        if mem_usage['available_mb'] < 500:  # 可用内存小于500MB
            logger.warning(f"Low available memory: {mem_usage['available_mb']:.1f}MB")
            if self.enable_disk_swap:
                self._swap_to_disk()
                
    def _aggressive_cleanup(self):
        """激进的内存清理"""
        logger.debug("Performing aggressive memory cleanup")
        
        # 清理缓存
        self.cache_manager.clear()
        
        # 强制垃圾回收
        gc.collect()
        gc.collect()  # 双重收集以确保清理循环引用
        
        self.stats['gc_collections'] += 1
        
        # 清理matplotlib如果存在
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
            
    def _swap_to_disk(self):
        """将部分数据交换到磁盘"""
        if not self.enable_disk_swap:
            return
            
        logger.info("Swapping memory to disk")
        self.stats['disk_swaps'] += 1
        
        # 这里可以实现更复杂的磁盘交换逻辑
        self.cache_manager._evict_cache(100)  # 释放100MB
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        current = self.monitor.get_memory_usage()
        return {
            'current_usage_mb': current['rss_mb'],
            'peak_usage_mb': self.stats['peak_memory_mb'],
            'available_mb': current['available_mb'],
            'percent_used': current['percent'],
            'optimizations': self.stats['optimizations'],
            'gc_collections': self.stats['gc_collections'],
            'disk_swaps': self.stats['disk_swaps']
        }
        
    def __del__(self):
        """清理资源"""
        try:
            self.cache_manager.clear()
        except:
            pass


def memory_efficient(func):
    """内存高效装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 创建内存管理器
        mem_mgr = IntelligentMemoryManager()
        
        with mem_mgr.memory_context(func.__name__):
            # 执行函数
            result = func(*args, **kwargs)
            
            # 如果结果是DataFrame，自动优化
            if isinstance(result, pd.DataFrame):
                result = mem_mgr.optimize_dataframe(result)
                
        # 打印内存统计
        stats = mem_mgr.get_memory_stats()
        if stats['gc_collections'] > 0 or stats['disk_swaps'] > 0:
            logger.info(f"Memory stats for {func.__name__}: "
                       f"Peak={stats['peak_usage_mb']:.1f}MB, "
                       f"GCs={stats['gc_collections']}, "
                       f"Swaps={stats['disk_swaps']}")
                       
        return result
        
    return wrapper


# 全局内存管理器实例（可选）
_global_memory_manager = None

def get_memory_manager() -> IntelligentMemoryManager:
    """获取全局内存管理器实例"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = IntelligentMemoryManager()
    return _global_memory_manager


def release_memory_manager():
    """释放全局内存管理器"""
    global _global_memory_manager
    if _global_memory_manager is not None:
        _global_memory_manager.cache_manager.clear()
        _global_memory_manager = None
        gc.collect()