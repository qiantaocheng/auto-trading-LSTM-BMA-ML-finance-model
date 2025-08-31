#!/usr/bin/env python3
"""
性能优化器
消除重复计算、优化缓存策略、提高系统执行效率
"""

import time
import pandas as pd
import numpy as np
import logging
import functools
import hashlib
import pickle
from typing import Dict, Any, Optional, Callable, List, Union
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref

logger = logging.getLogger(__name__)

class SmartCache:
    """智能缓存系统"""
    
    def __init__(self, 
                 cache_dir: str = "./cache/performance",
                 max_memory_items: int = 100,
                 ttl_seconds: int = 3600):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 内存缓存
        self.memory_cache = {}
        self.cache_times = {}
        self.access_counts = {}
        
        self.max_memory_items = max_memory_items
        self.ttl_seconds = ttl_seconds
        
        logger.info(f"智能缓存初始化 - 目录: {cache_dir}, TTL: {ttl_seconds}s")
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        # 创建参数的哈希值
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Any:
        """获取缓存值"""
        # 检查内存缓存
        if key in self.memory_cache:
            # 检查TTL
            if time.time() - self.cache_times[key] < self.ttl_seconds:
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                logger.debug(f"缓存命中(内存): {key}")
                return self.memory_cache[key]
            else:
                # 过期，删除
                del self.memory_cache[key]
                del self.cache_times[key]
                del self.access_counts[key]
        
        # 检查磁盘缓存
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # 检查TTL
                if time.time() - cache_data['timestamp'] < self.ttl_seconds:
                    # 加载到内存缓存
                    self.set(key, cache_data['value'], use_disk=False)
                    logger.debug(f"缓存命中(磁盘): {key}")
                    return cache_data['value']
                else:
                    # 过期，删除文件
                    cache_file.unlink()
            except Exception as e:
                logger.warning(f"磁盘缓存读取失败: {e}")
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def set(self, key: str, value: Any, use_disk: bool = True) -> None:
        """设置缓存值"""
        current_time = time.time()
        
        # 内存缓存
        if len(self.memory_cache) >= self.max_memory_items:
            # 使用LRU策略清理
            self._evict_lru()
        
        self.memory_cache[key] = value
        self.cache_times[key] = current_time
        self.access_counts[key] = 1
        
        # 磁盘缓存
        if use_disk:
            try:
                cache_file = self.cache_dir / f"{key}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'value': value,
                        'timestamp': current_time
                    }, f)
            except Exception as e:
                logger.warning(f"磁盘缓存写入失败: {e}")
    
    def _evict_lru(self) -> None:
        """清理最少使用的缓存项"""
        if not self.access_counts:
            return
        
        # 找到访问次数最少的键
        lru_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        
        del self.memory_cache[lru_key]
        del self.cache_times[lru_key]
        del self.access_counts[lru_key]
        
        logger.debug(f"LRU清理: {lru_key}")

# 全局缓存实例
_global_cache = SmartCache()

def cached(ttl_seconds: int = 3600, use_disk: bool = True):
    """缓存装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = _global_cache._generate_key(func.__name__, args, kwargs)
            
            # 尝试获取缓存
            cached_result = _global_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 计算结果
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # 缓存结果
            _global_cache.set(cache_key, result, use_disk=use_disk)
            
            logger.debug(f"函数执行并缓存: {func.__name__} ({duration:.3f}s)")
            return result
        
        return wrapper
    return decorator

class BatchProcessor:
    """批处理优化器"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def batch_process(self, 
                     func: Callable,
                     data_list: List[Any],
                     batch_size: int = 50) -> List[Any]:
        """批处理函数调用"""
        results = []
        
        # 分批处理
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            
            # 并行处理批次
            future_to_data = {
                self.executor.submit(func, data): data 
                for data in batch
            }
            
            # 收集结果
            for future in as_completed(future_to_data):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"批处理任务失败: {e}")
                    results.append(None)
        
        return results
    
    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.profiles = {}
        self.call_counts = {}
        self.total_times = {}
    
    def profile(self, func_name: str = None):
        """性能分析装饰器"""
        def decorator(func: Callable) -> Callable:
            name = func_name or func.__name__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    
                    # 记录性能数据
                    if name not in self.profiles:
                        self.profiles[name] = []
                        self.call_counts[name] = 0
                        self.total_times[name] = 0
                    
                    self.profiles[name].append(duration)
                    self.call_counts[name] += 1
                    self.total_times[name] += duration
            
            return wrapper
        return decorator
    
    def get_report(self) -> Dict[str, Dict[str, float]]:
        """获取性能报告"""
        report = {}
        
        for func_name in self.profiles:
            times = self.profiles[func_name]
            report[func_name] = {
                'call_count': self.call_counts[func_name],
                'total_time': self.total_times[func_name],
                'avg_time': np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'std_time': np.std(times)
            }
        
        return report
    
    def log_report(self):
        """输出性能报告"""
        report = self.get_report()
        
        logger.info("=== 性能分析报告 ===")
        for func_name, stats in sorted(report.items(), 
                                     key=lambda x: x[1]['total_time'], 
                                     reverse=True):
            logger.info(f"{func_name}:")
            logger.info(f"  调用次数: {stats['call_count']}")
            logger.info(f"  总时间: {stats['total_time']:.3f}s")
            logger.info(f"  平均时间: {stats['avg_time']:.3f}s")
            logger.info(f"  最大时间: {stats['max_time']:.3f}s")

# 全局性能分析器
_global_profiler = PerformanceProfiler()

def optimized_sample_weights(half_lives: List[int], 
                           dates: pd.DatetimeIndex,
                           current_date: datetime) -> Dict[int, pd.Series]:
    """
    优化的样本权重计算 - 消除重复计算
    """
    @cached(ttl_seconds=1800)  # 30分钟缓存
    def _compute_weights_for_halflife(half_life: int, 
                                    dates_hash: str,
                                    current_date_str: str) -> pd.Series:
        # 重建日期索引 (从缓存键恢复)
        current_dt = pd.to_datetime(current_date_str)
        
        # 计算时间差 (天)
        time_diffs = (current_dt - dates).days
        
        # 计算指数衰减权重
        weights = np.exp(-np.log(2) * time_diffs / half_life)
        
        # 标准化权重
        weights = weights / weights.sum()
        
        return pd.Series(weights, index=dates)
    
    # 计算日期的哈希值用于缓存
    dates_hash = hashlib.md5(str(dates.tolist()).encode()).hexdigest()
    current_date_str = current_date.isoformat()
    
    # 并行计算不同半衰期的权重
    batch_processor = BatchProcessor(max_workers=len(half_lives))
    
    def compute_single_weight(half_life):
        return half_life, _compute_weights_for_halflife(
            half_life, dates_hash, current_date_str
        )
    
    results = batch_processor.batch_process(compute_single_weight, half_lives)
    
    return dict(results)

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    优化DataFrame内存使用
    """
    logger.debug(f"内存优化前: {df.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
    
    optimized_df = df.copy()
    
    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype
        
        if col_type == 'object':
            # 尝试转换为category
            if optimized_df[col].nunique() / len(optimized_df) < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
        
        elif np.issubdtype(col_type, np.integer):
            # 优化整数类型
            c_min = optimized_df[col].min()
            c_max = optimized_df[col].max()
            
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                optimized_df[col] = optimized_df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                optimized_df[col] = optimized_df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                optimized_df[col] = optimized_df[col].astype(np.int32)
        
        elif np.issubdtype(col_type, np.floating):
            # 优化浮点类型
            if optimized_df[col].between(np.finfo(np.float32).min, 
                                       np.finfo(np.float32).max).all():
                optimized_df[col] = optimized_df[col].astype(np.float32)
    
    logger.debug(f"内存优化后: {optimized_df.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
    
    return optimized_df

# 获取全局对象的便捷函数
def get_cache() -> SmartCache:
    return _global_cache

def get_profiler() -> PerformanceProfiler:
    return _global_profiler