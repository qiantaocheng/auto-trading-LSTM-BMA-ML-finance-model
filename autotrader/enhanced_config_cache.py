#!/usr/bin/env python3
"""
增强的配置缓存系统
提供智能配置缓存、部分失效、预加载等优化功能
"""

import json
import time
import hashlib
import threading
import logging
from typing import Dict, Any, Optional, Set, List, Callable
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from copy import deepcopy
import weakref

@dataclass
class CacheEntry:
    """缓存条目"""
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    dependencies: Set[str] = field(default_factory=set)  # 依赖的其他键
    version: int = 1
    
    def update_access(self):
        """更新访问统计"""
        self.access_count += 1
        self.last_access = time.time()

class EnhancedConfigCache:
    """增强的配置缓存系统"""
    
    def __init__(self,
                 max_entries: int = 1000,
                 default_ttl: float = 3600.0,  # 1小时
                 enable_dependency_tracking: bool = True,
                 enable_preloading: bool = True):
        
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.enable_dependency_tracking = enable_dependency_tracking
        self.enable_preloading = enable_preloading
        
        # 多级缓存存储
        self._hot_cache: Dict[str, CacheEntry] = {}      # 热数据缓存
        self._warm_cache: Dict[str, CacheEntry] = {}     # 温数据缓存
        
        # 配置源缓存（按文件分组）
        self._source_cache: Dict[str, Dict[str, Any]] = {}
        self._source_timestamps: Dict[str, float] = {}
        
        # 依赖关系图
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # 访问模式分析
        self._access_patterns: Dict[str, List[float]] = defaultdict(list)
        self._prediction_cache: Dict[str, float] = {}  # 预测下次访问时间
        
        # 变更监听
        self._change_listeners: List[Callable[[str, Any, Any], None]] = []
        
        # 统计信息
        self._stats = {
            'hot_hits': 0, 'warm_hits': 0, 'misses': 0,
            'evictions': 0, 'invalidations': 0,
            'dependency_invalidations': 0, 'preloads': 0
        }
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 日志
        self.logger = logging.getLogger("EnhancedConfigCache")
        
        # 预加载队列
        self._preload_queue: List[str] = []
        
        # 后台清理线程
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        self._start_background_tasks()
    
    def get(self, key: str, default: Any = None, ttl: Optional[float] = None) -> Any:
        """获取配置值（智能缓存）"""
        with self._lock:
            # 热缓存检查
            if key in self._hot_cache:
                entry = self._hot_cache[key]
                if self._is_entry_valid(entry, ttl):
                    self._stats['hot_hits'] += 1
                    entry.update_access()
                    self._update_access_pattern(key)
                    self._move_to_front(key, self._hot_cache)
                    return deepcopy(entry.value)
                else:
                    del self._hot_cache[key]
            
            # 温缓存检查
            if key in self._warm_cache:
                entry = self._warm_cache[key]
                if self._is_entry_valid(entry, ttl):
                    self._stats['warm_hits'] += 1
                    entry.update_access()
                    self._update_access_pattern(key)
                    # 提升到热缓存
                    self._promote_to_hot(key, entry)
                    return deepcopy(entry.value)
                else:
                    del self._warm_cache[key]
            
            # 缓存未命中
            self._stats['misses'] += 1
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None, 
            dependencies: Optional[Set[str]] = None) -> None:
        """设置配置值"""
        with self._lock:
            # 检查是否有变化
            old_value = self.get(key)
            if old_value == value:
                return  # 值未变化，无需更新
            
            # 创建缓存条目
            entry = CacheEntry(
                value=deepcopy(value),
                timestamp=time.time(),
                dependencies=dependencies or set()
            )
            
            # 智能存储位置选择
            if self._should_store_in_hot(key):
                self._store_in_hot(key, entry)
            else:
                self._store_in_warm(key, entry)
            
            # 更新依赖关系
            if self.enable_dependency_tracking and dependencies:
                self._update_dependencies(key, dependencies)
            
            # 通知变更监听器
            self._notify_change_listeners(key, old_value, value)
            
            # 预测相关键的访问
            if self.enable_preloading:
                self._schedule_preloading(key)
    
    def invalidate(self, key: str, cascade: bool = True) -> None:
        """失效指定键"""
        with self._lock:
            # 移除缓存
            removed = False
            if key in self._hot_cache:
                del self._hot_cache[key]
                removed = True
            if key in self._warm_cache:
                del self._warm_cache[key]
                removed = True
            
            if removed:
                self._stats['invalidations'] += 1
            
            # 级联失效依赖项
            if cascade and self.enable_dependency_tracking:
                dependents = self._reverse_dependencies.get(key, set())
                for dependent in dependents:
                    self.invalidate(dependent, cascade=True)
                    self._stats['dependency_invalidations'] += 1
    
    def invalidate_pattern(self, pattern: str) -> None:
        """按模式失效缓存"""
        import fnmatch
        with self._lock:
            keys_to_invalidate = []
            
            # 检查热缓存
            for key in self._hot_cache:
                if fnmatch.fnmatch(key, pattern):
                    keys_to_invalidate.append(key)
            
            # 检查温缓存
            for key in self._warm_cache:
                if fnmatch.fnmatch(key, pattern):
                    keys_to_invalidate.append(key)
            
            # 批量失效
            for key in keys_to_invalidate:
                self.invalidate(key, cascade=False)
    
    def set_source(self, source_name: str, config_dict: Dict[str, Any], 
                   file_path: Optional[str] = None) -> None:
        """设置配置源"""
        with self._lock:
            # 检查文件时间戳
            if file_path and Path(file_path).exists():
                file_timestamp = Path(file_path).stat().st_mtime
                cached_timestamp = self._source_timestamps.get(source_name, 0)
                
                if file_timestamp <= cached_timestamp:
                    return  # 文件未变化
                
                self._source_timestamps[source_name] = file_timestamp
            
            # 更新源缓存
            old_config = self._source_cache.get(source_name, {})
            self._source_cache[source_name] = deepcopy(config_dict)
            
            # 找出变化的键
            changed_keys = self._find_changed_keys(old_config, config_dict, source_name)
            
            # 失效相关缓存
            for key in changed_keys:
                self.invalidate(key, cascade=True)
    
    def get_source(self, source_name: str) -> Optional[Dict[str, Any]]:
        """获取配置源"""
        with self._lock:
            return deepcopy(self._source_cache.get(source_name))
    
    def _should_store_in_hot(self, key: str) -> bool:
        """判断是否应该存储在热缓存"""
        # 基于访问频率和模式决定
        recent_accesses = self._access_patterns.get(key, [])
        
        if len(recent_accesses) > 5:  # 有足够的访问历史
            # 计算访问频率
            now = time.time()
            recent_accesses_count = len([t for t in recent_accesses if now - t < 300])  # 5分钟内
            
            if recent_accesses_count > 3:  # 高频访问
                return True
        
        # 检查是否是关键配置
        critical_patterns = ['connection.', 'capital.', 'risk.', 'orders.']
        if any(key.startswith(pattern) for pattern in critical_patterns):
            return True
        
        return len(self._hot_cache) < self.max_entries // 4  # 热缓存不超过总容量25%
    
    def _store_in_hot(self, key: str, entry: CacheEntry) -> None:
        """存储到热缓存"""
        # 检查容量
        if len(self._hot_cache) >= self.max_entries // 4:
            self._evict_from_hot()
        
        self._hot_cache[key] = entry
    
    def _store_in_warm(self, key: str, entry: CacheEntry) -> None:
        """存储到温缓存"""
        # 检查容量
        if len(self._warm_cache) >= self.max_entries * 3 // 4:
            self._evict_from_warm()
        
        self._warm_cache[key] = entry
    
    def _promote_to_hot(self, key: str, entry: CacheEntry) -> None:
        """提升到热缓存"""
        if key in self._warm_cache:
            del self._warm_cache[key]
        self._store_in_hot(key, entry)
    
    def _evict_from_hot(self) -> None:
        """从热缓存淘汰"""
        if not self._hot_cache:
            return
        
        # 使用LRU策略，但考虑访问频率
        candidates = list(self._hot_cache.items())
        candidates.sort(key=lambda x: (x[1].access_count, x[1].last_access))
        
        # 淘汰访问频率最低的
        key, entry = candidates[0]
        del self._hot_cache[key]
        
        # 降级到温缓存
        self._store_in_warm(key, entry)
        self._stats['evictions'] += 1
    
    def _evict_from_warm(self) -> None:
        """从温缓存淘汰"""
        if not self._warm_cache:
            return
        
        # 简单LRU
        oldest_key = min(self._warm_cache.keys(), 
                        key=lambda k: self._warm_cache[k].last_access)
        del self._warm_cache[oldest_key]
        self._stats['evictions'] += 1
    
    def _move_to_front(self, key: str, cache: Dict[str, CacheEntry]) -> None:
        """移动到前面（LRU更新）"""
        # Python 3.7+ 字典保持插入顺序，重新插入即可移到最后
        if key in cache:
            entry = cache.pop(key)
            cache[key] = entry
    
    def _is_entry_valid(self, entry: CacheEntry, ttl: Optional[float] = None) -> bool:
        """检查缓存条目是否有效"""
        effective_ttl = ttl or self.default_ttl
        return (time.time() - entry.timestamp) < effective_ttl
    
    def _update_dependencies(self, key: str, dependencies: Set[str]) -> None:
        """更新依赖关系"""
        # 清理旧依赖
        old_deps = self._dependency_graph.get(key, set())
        for dep in old_deps:
            self._reverse_dependencies[dep].discard(key)
        
        # 设置新依赖
        self._dependency_graph[key] = dependencies
        for dep in dependencies:
            self._reverse_dependencies[dep].add(key)
    
    def _update_access_pattern(self, key: str) -> None:
        """更新访问模式"""
        now = time.time()
        pattern = self._access_patterns[key]
        
        # 保留最近100次访问
        pattern.append(now)
        if len(pattern) > 100:
            pattern.pop(0)
        
        # 预测下次访问时间
        if len(pattern) > 3:
            intervals = [pattern[i] - pattern[i-1] for i in range(1, len(pattern))]
            avg_interval = sum(intervals) / len(intervals)
            self._prediction_cache[key] = now + avg_interval
    
    def _schedule_preloading(self, key: str) -> None:
        """调度预加载"""
        if not self.enable_preloading:
            return
        
        # 基于依赖关系预加载相关配置
        related_keys = self._dependency_graph.get(key, set())
        related_keys.update(self._reverse_dependencies.get(key, set()))
        
        for related_key in related_keys:
            if (related_key not in self._hot_cache and 
                related_key not in self._warm_cache and
                related_key not in self._preload_queue):
                self._preload_queue.append(related_key)
    
    def _find_changed_keys(self, old_config: Dict[str, Any], 
                          new_config: Dict[str, Any], prefix: str = "") -> Set[str]:
        """找出变化的配置键"""
        changed_keys = set()
        
        # 递归比较嵌套字典
        def compare_dict(old_dict: Dict, new_dict: Dict, path: str = ""):
            all_keys = set(old_dict.keys()) | set(new_dict.keys())
            
            for key in all_keys:
                full_key = f"{path}.{key}" if path else key
                
                if key not in old_dict or key not in new_dict:
                    changed_keys.add(full_key)
                elif isinstance(old_dict[key], dict) and isinstance(new_dict[key], dict):
                    compare_dict(old_dict[key], new_dict[key], full_key)
                elif old_dict[key] != new_dict[key]:
                    changed_keys.add(full_key)
        
        compare_dict(old_config, new_config, prefix)
        return changed_keys
    
    def _notify_change_listeners(self, key: str, old_value: Any, new_value: Any) -> None:
        """通知变更监听器"""
        for listener in self._change_listeners:
            try:
                listener(key, old_value, new_value)
            except Exception as e:
                self.logger.warning(f"配置变更监听器异常: {e}")
    
    def add_change_listener(self, listener: Callable[[str, Any, Any], None]) -> None:
        """添加配置变更监听器"""
        self._change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[[str, Any, Any], None]) -> None:
        """移除配置变更监听器"""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
    
    def _start_background_tasks(self) -> None:
        """启动后台任务"""
        self._cleanup_thread = threading.Thread(
            target=self._background_cleanup,
            name="ConfigCacheCleanup",
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _background_cleanup(self) -> None:
        """后台清理任务"""
        while not self._shutdown_event.wait(60):  # 每分钟清理一次
            try:
                self._cleanup_expired_entries()
                self._process_preload_queue()
                self._optimize_cache_layout()
            except Exception as e:
                self.logger.error(f"后台清理异常: {e}")
    
    def _cleanup_expired_entries(self) -> None:
        """清理过期条目"""
        with self._lock:
            now = time.time()
            
            # 清理热缓存过期条目
            expired_hot = [k for k, v in self._hot_cache.items() 
                          if (now - v.timestamp) > self.default_ttl]
            for key in expired_hot:
                del self._hot_cache[key]
            
            # 清理温缓存过期条目
            expired_warm = [k for k, v in self._warm_cache.items() 
                           if (now - v.timestamp) > self.default_ttl * 2]  # 温缓存TTL更长
            for key in expired_warm:
                del self._warm_cache[key]
            
            if expired_hot or expired_warm:
                self.logger.debug(f"清理过期缓存: 热={len(expired_hot)}, 温={len(expired_warm)}")
    
    def _process_preload_queue(self) -> None:
        """处理预加载队列"""
        if not self.enable_preloading or not self._preload_queue:
            return
        
        with self._lock:
            # 处理前5个预加载请求
            for _ in range(min(5, len(self._preload_queue))):
                key = self._preload_queue.pop(0)
                # 这里应该从实际数据源加载配置
                # 为了示例，我们跳过实际加载
                self._stats['preloads'] += 1
    
    def _optimize_cache_layout(self) -> None:
        """优化缓存布局"""
        with self._lock:
            # 基于访问模式调整热缓存
            if len(self._warm_cache) > 0:
                # 找出温缓存中访问频率高的条目
                candidates = [(k, v) for k, v in self._warm_cache.items() 
                             if v.access_count > 5]
                
                # 提升到热缓存
                for key, entry in candidates[:3]:  # 最多提升3个
                    if len(self._hot_cache) < self.max_entries // 4:
                        self._promote_to_hot(key, entry)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            total_hits = self._stats['hot_hits'] + self._stats['warm_hits']
            total_requests = total_hits + self._stats['misses']
            
            return {
                'cache_stats': self._stats.copy(),
                'hit_rate': total_hits / max(total_requests, 1),
                'hot_cache_size': len(self._hot_cache),
                'warm_cache_size': len(self._warm_cache),
                'source_cache_size': len(self._source_cache),
                'dependency_count': len(self._dependency_graph),
                'preload_queue_size': len(self._preload_queue),
                'access_patterns_tracked': len(self._access_patterns),
                'change_listeners': len(self._change_listeners),
                'memory_efficiency': {
                    'hot_cache_ratio': len(self._hot_cache) / max(self.max_entries // 4, 1),
                    'warm_cache_ratio': len(self._warm_cache) / max(self.max_entries * 3 // 4, 1)
                }
            }
    
    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            self._hot_cache.clear()
            self._warm_cache.clear()
            self._source_cache.clear()
            self._source_timestamps.clear()
            self._dependency_graph.clear()
            self._reverse_dependencies.clear()
            self._access_patterns.clear()
            self._prediction_cache.clear()
            self._preload_queue.clear()
            
            # 重置统计
            for key in self._stats:
                self._stats[key] = 0
    
    def shutdown(self) -> None:
        """关闭缓存系统"""
        self.logger.info("关闭增强配置缓存")
        
        # 停止后台线程
        if self._cleanup_thread:
            self._shutdown_event.set()
            self._cleanup_thread.join(timeout=5)
        
        # 清空缓存
        self.clear()


# 全局增强配置缓存实例
_global_enhanced_config_cache: Optional[EnhancedConfigCache] = None

def get_enhanced_config_cache() -> EnhancedConfigCache:
    """获取全局增强配置缓存"""
    global _global_enhanced_config_cache
    if _global_enhanced_config_cache is None:
        _global_enhanced_config_cache = EnhancedConfigCache()
    return _global_enhanced_config_cache
