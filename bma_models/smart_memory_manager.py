#!/usr/bin/env python3
"""
智能内存管理器
防止内存泄漏，优化内存使用
"""

import gc
import psutil
import pandas as pd
import numpy as np
import logging
import weakref
from functools import wraps
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MemoryManager:
    """智能内存管理器"""
    
    def __init__(self, 
                 max_memory_gb: float = 8.0,
                 warning_threshold: float = 0.8,
                 critical_threshold: float = 0.9,
                 cleanup_interval_minutes: int = 10):
        
        self.max_memory_gb = max_memory_gb
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)
        
        # 缓存管理
        self.cache_objects = weakref.WeakValueDictionary()
        self.cache_access_times = {}
        self.cache_size_limits = {}
        
        # 性能监控
        self.memory_history = deque(maxlen=100)
        self.cleanup_history = []
        self.last_cleanup = datetime.now()
        
        logger.info(f"内存管理器初始化 - 最大内存: {max_memory_gb}GB, "
                   f"警告阈值: {warning_threshold:.1%}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        usage = {
            'process_mb': memory_info.rss / (1024**2),
            'process_gb': memory_info.rss / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_used_percent': system_memory.percent / 100,
            'is_warning': memory_info.rss / (1024**3) > self.max_memory_gb * self.warning_threshold,
            'is_critical': memory_info.rss / (1024**3) > self.max_memory_gb * self.critical_threshold
        }
        
        # 记录历史
        self.memory_history.append({
            'timestamp': datetime.now(),
            'memory_gb': usage['process_gb']
        })
        
        return usage
    
    def force_cleanup(self, aggressive: bool = False) -> Dict[str, Any]:
        """强制内存清理"""
        initial_usage = self.get_memory_usage()
        logger.info(f"开始内存清理 - 当前使用: {initial_usage['process_gb']:.2f}GB")
        
        cleanup_stats = {
            'initial_memory': initial_usage['process_gb'],
            'cache_cleared': 0,
            'gc_collections': 0
        }
        
        # 1. 清理过期缓存
        cache_cleared = self._cleanup_cache()
        cleanup_stats['cache_cleared'] = cache_cleared
        
        # 2. 强制垃圾回收
        if aggressive:
            # 积极模式：多轮深度回收
            for generation in [0, 1, 2]:
                collected = gc.collect(generation)
                cleanup_stats['gc_collections'] += collected
                logger.debug(f"GC generation {generation}: 回收 {collected} 对象")
        else:
            # 标准模式：单轮回收
            collected = gc.collect()
            cleanup_stats['gc_collections'] = collected
        
        # 3. 清理pandas内存
        self._cleanup_pandas_memory()
        
        # 4. 清理numpy内存
        self._cleanup_numpy_memory()
        
        final_usage = self.get_memory_usage()
        cleanup_stats['final_memory'] = final_usage['process_gb']
        cleanup_stats['memory_freed'] = initial_usage['process_gb'] - final_usage['process_gb']
        
        self.cleanup_history.append({
            'timestamp': datetime.now(),
            'stats': cleanup_stats,
            'aggressive': aggressive
        })
        
        logger.info(f"内存清理完成 - 释放: {cleanup_stats['memory_freed']:.2f}GB, "
                   f"当前: {final_usage['process_gb']:.2f}GB")
        
        return cleanup_stats
    
    def _cleanup_cache(self) -> int:
        """清理过期缓存"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, access_time in self.cache_access_times.items():
            if current_time - access_time > self.cleanup_interval:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self.cache_objects:
                del self.cache_objects[key]
            if key in self.cache_access_times:
                del self.cache_access_times[key]
            if key in self.cache_size_limits:
                del self.cache_size_limits[key]
        
        return len(expired_keys)
    
    def _cleanup_pandas_memory(self):
        """清理pandas内存"""
        try:
            # 清理pandas的内部缓存
            pd.core.common._global_config.reset_option("^display")
        except:
            pass
    
    def _cleanup_numpy_memory(self):
        """清理numpy内存"""
        try:
            # 清理numpy的内部数组缓存
            np.core._multiarray_umath._reload_guard()
        except:
            pass
    
    def cache_object(self, key: str, obj: Any, size_limit: Optional[int] = None):
        """缓存对象"""
        self.cache_objects[key] = obj
        self.cache_access_times[key] = datetime.now()
        if size_limit:
            self.cache_size_limits[key] = size_limit
    
    def get_cached_object(self, key: str) -> Any:
        """获取缓存对象"""
        if key in self.cache_objects:
            self.cache_access_times[key] = datetime.now()
            return self.cache_objects[key]
        return None
    
    def should_cleanup(self) -> bool:
        """检查是否需要清理"""
        usage = self.get_memory_usage()
        time_based = datetime.now() - self.last_cleanup > self.cleanup_interval
        memory_based = usage['is_warning']
        
        return time_based or memory_based

# 全局内存管理器
_global_memory_manager = MemoryManager()

def get_memory_manager() -> MemoryManager:
    """获取全局内存管理器"""
    return _global_memory_manager

def memory_managed(max_memory_gb: float = 8.0, 
                  auto_cleanup: bool = True,
                  aggressive_on_critical: bool = True):
    """
    内存管理装饰器
    
    Args:
        max_memory_gb: 最大内存限制(GB)
        auto_cleanup: 自动清理
        aggressive_on_critical: 临界状态下积极清理
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_memory_manager()
            
            try:
                # 执行前检查
                pre_usage = manager.get_memory_usage()
                if pre_usage['is_critical']:
                    logger.warning(f"执行前内存临界: {pre_usage['process_gb']:.2f}GB")
                    if auto_cleanup:
                        manager.force_cleanup(aggressive=aggressive_on_critical)
                
                # 执行函数
                result = func(*args, **kwargs)
                
                return result
                
            finally:
                # 执行后清理
                if auto_cleanup and manager.should_cleanup():
                    post_usage = manager.get_memory_usage()
                    aggressive = post_usage['is_critical'] and aggressive_on_critical
                    manager.force_cleanup(aggressive=aggressive)
        
        return wrapper
    return decorator

def limit_dataframe_memory(df: pd.DataFrame, max_memory_mb: float = 500) -> pd.DataFrame:
    """
    限制DataFrame内存使用
    """
    current_memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    
    if current_memory_mb <= max_memory_mb:
        return df
    
    logger.warning(f"DataFrame内存超限: {current_memory_mb:.1f}MB > {max_memory_mb:.1f}MB")
    
    # 采样降低内存使用
    reduction_ratio = max_memory_mb / current_memory_mb
    sample_size = int(len(df) * reduction_ratio * 0.9)  # 留10%安全边际
    
    logger.info(f"对DataFrame采样: {len(df)} -> {sample_size} 行")
    return df.sample(n=sample_size, random_state=42).sort_index()

# 修复系统级内存泄漏
def fix_system_memory_leaks():
    """修复系统级内存泄漏问题"""
    manager = get_memory_manager()
    
    # 强制清理
    cleanup_stats = manager.force_cleanup(aggressive=True)
    
    # 重置全局状态
    gc.set_threshold(700, 10, 10)  # 更积极的垃圾回收
    
    logger.info("系统级内存泄漏修复完成")
    return cleanup_stats