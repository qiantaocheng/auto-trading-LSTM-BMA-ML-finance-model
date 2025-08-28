#!/usr/bin/env python3
"""
智能内存管理器
实时监控内存使用，自动释放和优化
"""

import gc
import os
import sys
import psutil
import threading
import time
import logging
import weakref
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class MemoryThreshold:
    """内存阈值配置"""
    warning_percent: float = 75.0    # 警告阈值
    critical_percent: float = 85.0   # 临界阈值
    emergency_percent: float = 95.0  # 紧急阈值
    max_process_gb: float = 4.0      # 进程最大内存

class SmartMemoryManager:
    """智能内存管理器"""
    
    def __init__(self, 
                 thresholds: Optional[MemoryThreshold] = None,
                 monitor_interval: float = 5.0,
                 auto_cleanup: bool = True):
        """
        初始化智能内存管理器
        
        Args:
            thresholds: 内存阈值配置
            monitor_interval: 监控间隔(秒)
            auto_cleanup: 自动清理
        """
        self.thresholds = thresholds or MemoryThreshold()
        self.monitor_interval = monitor_interval
        self.auto_cleanup = auto_cleanup
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        self._stop_event = threading.Event()
        
        # 内存统计
        self.memory_history = []
        self.cleanup_count = 0
        self.warning_count = 0
        self.emergency_count = 0
        
        # 注册的对象引用（用于清理）
        self.registered_objects = weakref.WeakSet()
        self.cleanup_callbacks = []
        
        # 缓存管理
        self.cache_objects = {}
        self.cache_access_times = {}
        self.max_cache_age_minutes = 30
        
    def get_memory_info(self) -> Dict[str, Any]:
        """获取详细内存信息"""
        # 系统内存
        system_memory = psutil.virtual_memory()
        
        # 进程内存
        process = psutil.Process()
        process_memory = process.memory_info()
        process_percent = process.memory_percent()
        
        # 计算各种指标
        memory_info = {
            # 系统内存
            'system_total_gb': system_memory.total / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_used_gb': (system_memory.total - system_memory.available) / (1024**3),
            'system_percent': system_memory.percent,
            
            # 进程内存
            'process_rss_gb': process_memory.rss / (1024**3),
            'process_vms_gb': process_memory.vms / (1024**3),
            'process_percent': process_percent,
            
            # 阈值状态
            'warning_threshold': self.thresholds.warning_percent,
            'critical_threshold': self.thresholds.critical_percent,
            'emergency_threshold': self.thresholds.emergency_percent,
            
            # 状态评估
            'status': self._assess_memory_status(system_memory.percent, process_percent),
            'timestamp': datetime.now()
        }
        
        return memory_info
    
    def _assess_memory_status(self, system_percent: float, process_percent: float) -> str:
        """评估内存状态"""
        process_gb = psutil.Process().memory_info().rss / (1024**3)
        
        if (system_percent >= self.thresholds.emergency_percent or 
            process_gb >= self.thresholds.max_process_gb):
            return "EMERGENCY"
        elif (system_percent >= self.thresholds.critical_percent or 
              process_gb >= self.thresholds.max_process_gb * 0.8):
            return "CRITICAL"
        elif system_percent >= self.thresholds.warning_percent:
            return "WARNING"
        else:
            return "NORMAL"
    
    def register_object(self, obj: Any, cleanup_func: Optional[Callable] = None):
        """注册需要内存管理的对象"""
        self.registered_objects.add(obj)
        if cleanup_func:
            self.cleanup_callbacks.append((weakref.ref(obj), cleanup_func))
    
    def add_to_cache(self, key: str, obj: Any):
        """添加对象到缓存"""
        self.cache_objects[key] = obj
        self.cache_access_times[key] = datetime.now()
    
    def get_from_cache(self, key: str) -> Any:
        """从缓存获取对象"""
        if key in self.cache_objects:
            self.cache_access_times[key] = datetime.now()
            return self.cache_objects[key]
        return None
    
    def cleanup_expired_cache(self):
        """清理过期缓存"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, access_time in self.cache_access_times.items():
            if current_time - access_time > timedelta(minutes=self.max_cache_age_minutes):
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self.cache_objects:
                del self.cache_objects[key]
            if key in self.cache_access_times:
                del self.cache_access_times[key]
        
        if expired_keys:
            logger.debug(f"清理了 {len(expired_keys)} 个过期缓存对象")
    
    def force_garbage_collection(self) -> int:
        """强制垃圾回收"""
        collected = 0
        
        # 多轮垃圾回收
        for generation in range(3):
            collected += gc.collect(generation)
        
        # 额外的全面回收
        collected += gc.collect()
        
        # 尝试释放numpy内存
        try:
            # 清理numpy临时数组
            import numpy as np
            # 强制刷新numpy缓存
            if hasattr(np, '_clear_cache'):
                np._clear_cache()
        except (ImportError, AttributeError):
            pass
        
        # 清理pandas缓存
        try:
            import pandas as pd
            # 清理pandas的一些内部缓存
            if hasattr(pd.core.common, '_values_from_object'):
                pd.core.common._values_from_object.cache_clear()
        except (ImportError, AttributeError):
            pass
        
        self.cleanup_count += 1
        return collected
    
    def emergency_cleanup(self):
        """紧急内存清理"""
        logger.warning("执行紧急内存清理...")
        
        # 1. 强制垃圾回收
        collected = self.force_garbage_collection()
        
        # 2. 清理所有缓存
        cache_cleared = len(self.cache_objects)
        self.cache_objects.clear()
        self.cache_access_times.clear()
        
        # 3. 调用注册的清理回调
        callbacks_called = 0
        for obj_ref, cleanup_func in self.cleanup_callbacks:
            try:
                obj = obj_ref()
                if obj is not None:
                    cleanup_func(obj)
                    callbacks_called += 1
            except Exception as e:
                logger.warning(f"Cleanup callback failed: {e}")
                continue
        
        # 4. 尝试释放大型数组
        freed_arrays = self._cleanup_large_arrays()
        
        logger.warning(f"紧急清理完成: 回收对象={collected}, 清理缓存={cache_cleared}, "
                      f"回调函数={callbacks_called}, 释放数组={freed_arrays}")
        
        self.emergency_count += 1
    
    def _cleanup_large_arrays(self) -> int:
        """清理大型数组对象"""
        freed_count = 0
        
        # 遍历当前命名空间中的大型numpy数组和DataFrame
        current_frame = sys._getframe(1)
        while current_frame:
            local_vars = current_frame.f_locals
            for name, obj in list(local_vars.items()):
                try:
                    # 检查numpy数组
                    if isinstance(obj, np.ndarray) and obj.nbytes > 100 * 1024 * 1024:  # >100MB
                        logger.debug(f"释放大型数组 {name}: {obj.nbytes / (1024**2):.1f}MB")
                        del local_vars[name]
                        freed_count += 1
                    
                    # 检查DataFrame
                    elif isinstance(obj, pd.DataFrame) and obj.memory_usage(deep=True).sum() > 100 * 1024 * 1024:
                        logger.debug(f"释放大型DataFrame {name}: {obj.memory_usage(deep=True).sum() / (1024**2):.1f}MB")
                        del local_vars[name]
                        freed_count += 1
                        
                except (AttributeError, KeyError, ValueError):
                    continue
            
            current_frame = current_frame.f_back
        
        return freed_count
    
    def _monitor_loop(self):
        """内存监控循环"""
        logger.info("内存监控已启动")
        
        while not self._stop_event.wait(self.monitor_interval):
            try:
                memory_info = self.get_memory_info()
                self.memory_history.append(memory_info)
                
                # 限制历史记录长度
                if len(self.memory_history) > 1000:
                    self.memory_history = self.memory_history[-500:]
                
                status = memory_info['status']
                
                if status == "EMERGENCY":
                    logger.error(f"内存紧急状态! 系统使用率: {memory_info['system_percent']:.1f}%, "
                               f"进程使用: {memory_info['process_rss_gb']:.2f}GB")
                    if self.auto_cleanup:
                        self.emergency_cleanup()
                
                elif status == "CRITICAL":
                    logger.warning(f"内存临界状态! 系统使用率: {memory_info['system_percent']:.1f}%, "
                                 f"进程使用: {memory_info['process_rss_gb']:.2f}GB")
                    if self.auto_cleanup:
                        self.force_garbage_collection()
                        self.cleanup_expired_cache()
                
                elif status == "WARNING":
                    self.warning_count += 1
                    if self.warning_count % 10 == 0:  # 每10次警告记录一次
                        logger.info(f"内存警告状态: 系统使用率: {memory_info['system_percent']:.1f}%, "
                                  f"进程使用: {memory_info['process_rss_gb']:.2f}GB")
                    
                    if self.auto_cleanup:
                        self.cleanup_expired_cache()
                
            except Exception as e:
                logger.error(f"内存监控错误: {e}")
    
    def start_monitoring(self):
        """启动内存监控"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self._stop_event.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("内存监控已启动")
    
    def stop_monitoring(self):
        """停止内存监控"""
        if self.is_monitoring:
            self.is_monitoring = False
            self._stop_event.set()
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            logger.info("内存监控已停止")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取内存管理统计信息"""
        current_memory = self.get_memory_info()
        
        return {
            'current_memory': current_memory,
            'monitoring_active': self.is_monitoring,
            'cleanup_count': self.cleanup_count,
            'warning_count': self.warning_count,
            'emergency_count': self.emergency_count,
            'registered_objects': len(self.registered_objects),
            'cache_objects': len(self.cache_objects),
            'memory_history_length': len(self.memory_history)
        }
    
    def optimize_memory_settings(self):
        """优化内存相关设置"""
        # 设置numpy线程数（避免过度并行）
        try:
            import numpy as np
            os.environ['OMP_NUM_THREADS'] = '4'
            os.environ['MKL_NUM_THREADS'] = '4'
            os.environ['NUMEXPR_NUM_THREADS'] = '4'
            logger.debug("已优化numpy线程设置")
        except (ImportError, AttributeError):
            pass
        
        # 设置pandas显示选项
        try:
            import pandas as pd
            pd.set_option('display.max_rows', 100)
            pd.set_option('display.max_columns', 50)
            logger.debug("已优化pandas显示设置")
        except (ImportError, AttributeError):
            pass
        
        # 设置matplotlib后端（如果使用）
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端，节省内存
            logger.debug("已设置matplotlib非交互式后端")
        except (ImportError, AttributeError):
            pass


# 全局内存管理器实例
_global_memory_manager = None

def get_memory_manager() -> SmartMemoryManager:
    """获取全局内存管理器"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = SmartMemoryManager()
        _global_memory_manager.optimize_memory_settings()
        _global_memory_manager.start_monitoring()
    return _global_memory_manager

def memory_optimize(func):
    """内存优化装饰器"""
    def wrapper(*args, **kwargs):
        memory_manager = get_memory_manager()
        
        # 执行前检查内存
        memory_info = memory_manager.get_memory_info()
        if memory_info['status'] in ['CRITICAL', 'EMERGENCY']:
            logger.warning(f"函数 {func.__name__} 执行前内存紧张，执行清理")
            memory_manager.force_garbage_collection()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # 执行后清理
            memory_manager.force_garbage_collection()
    
    return wrapper