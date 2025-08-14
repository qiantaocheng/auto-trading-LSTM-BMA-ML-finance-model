#!/usr/bin/env python3
"""
增强技术指标缓存系统
提供更高效缓存策略、内存管理and预热机制
"""

import numpy as np
import pandas as pd
import logging
import time
import hashlib
import weakref
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from collections import OrderedDict, defaultdict, deque
from threading import RLock
from dataclasses import dataclass, field
import pickle
import psutil
import gc
from enum import Enum

class CachePolicy(Enum):
    """缓存策略"""
    LRU = "lru"          # 最近最少使use
    LFU = "lfu"          # 最少频率使use  
    TTL = "ttl"          # when间过期
    ADAPTIVE = "adaptive" # 自适应

@dataclass
class IndicatorResult:
    """指标计算结果"""
    value: Any
    timestamp: float
    data_hash: str
    computation_time: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_access(self):
        """updates访问统计"""
        self.access_count += 1
        self.last_access = time.time()

class EnhancedIndicatorCache:
    """增强技术指标缓存系统"""
    
    def __init__(self, 
                 max_cache_size: int = 2000,
                 max_memory_mb: int = 100,
                 ttl_seconds: float = 600,
                 cache_policy: CachePolicy = CachePolicy.ADAPTIVE,
                 enable_prewarming: bool = True):
        
        self.logger = logging.getLogger("EnhancedIndicatorCache")
        self.max_cache_size = max_cache_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.cache_policy = cache_policy
        self.enable_prewarming = enable_prewarming
        
        # 多层缓存存储
        self._l1_cache: OrderedDict[str, IndicatorResult] = OrderedDict()  # 热数据
        self._l2_cache: Dict[str, IndicatorResult] = {}  # 温数据
        self._cold_storage: Dict[str, bytes] = {}  # 冷数据（序列化）
        
        # 缓存统计and管理
        self._cache_stats = {
            'l1_hits': 0, 'l2_hits': 0, 'cold_hits': 0,
            'misses': 0, 'evictions': 0, 'prewarmed': 0
        }
        
        # 访问频率统计（useatLFU）
        self._access_frequency: Dict[str, int] = defaultdict(int)
        self._frequency_buckets: Dict[int, set] = defaultdict(set)
        
        # 内存监控
        self._current_memory = 0
        self._memory_threshold = 0.8  # 80%内存使use率触发清理
        
        # 哈希缓存（避免重复计算）
        self._hash_cache: Dict[tuple, str] = {}
        
        # 预热队列
        self._prewarming_queue: deque = deque(maxlen=100)
        
        # 线程锁
        self.lock = RLock()
        
        # 预定义指标函数
        self._indicator_functions = {
            'sma': self._compute_sma,
            'ema': self._compute_ema,
            'rsi': self._compute_rsi,
            'atr': self._compute_atr,
            'bollinger': self._compute_bollinger,
            'macd': self._compute_macd,
            'stoch': self._compute_stochastic,
            'williams_r': self._compute_williams_r
        }
        
        self.logger.info(f"初始化增强缓存: 最大{max_cache_size} items, {max_memory_mb}MB, 策略={cache_policy.value}")
    
    def get_indicator(self, indicator_name: str, symbol: str, data: List[float], 
                     period: int = 14, **kwargs) -> Any:
        """retrieval技术指标值（多层缓存）"""
        with self.lock:
            # 生成缓存键（优化版）
            cache_key = self._generate_cache_key_optimized(indicator_name, symbol, data, period, **kwargs)
            
            # L1 缓存check（热数据）
            if cache_key in self._l1_cache:
                result = self._l1_cache[cache_key]
                if self._is_cache_valid(result):
                    self._cache_stats['l1_hits'] += 1
                    result.update_access()
                    self._update_frequency(cache_key)
                    # 移动to最before面（LRU）
                    self._l1_cache.move_to_end(cache_key)
                    self.logger.debug(f"L1缓存命in: {indicator_name}({symbol}, {period})")
                    return result.value
                else:
                    del self._l1_cache[cache_key]
            
            # L2 缓存check（温数据）
            if cache_key in self._l2_cache:
                result = self._l2_cache[cache_key]
                if self._is_cache_valid(result):
                    self._cache_stats['l2_hits'] += 1
                    result.update_access()
                    self._update_frequency(cache_key)
                    # 提升toL1缓存
                    self._promote_to_l1(cache_key, result)
                    self.logger.debug(f"L2缓存命in: {indicator_name}({symbol}, {period})")
                    return result.value
                else:
                    del self._l2_cache[cache_key]
            
            # 冷存储check
            if cache_key in self._cold_storage:
                try:
                    result = pickle.loads(self._cold_storage[cache_key])
                    if self._is_cache_valid(result):
                        self._cache_stats['cold_hits'] += 1
                        result.update_access()
                        self._update_frequency(cache_key)
                        # 提升toL2缓存
                        self._promote_to_l2(cache_key, result)
                        if cache_key in self._cold_storage:
                            del self._cold_storage[cache_key]
                        self.logger.debug(f"冷存储命in: {indicator_name}({symbol}, {period})")
                        return result.value
                    else:
                        del self._cold_storage[cache_key]
                except Exception as e:
                    self.logger.warning(f"冷存储反序列化failed: {e}")
                    del self._cold_storage[cache_key]
            
            # 缓存未命in，计算指标
            self._cache_stats['misses'] += 1
            return self._compute_and_cache_enhanced(cache_key, indicator_name, symbol, data, period, **kwargs)
    
    def _generate_cache_key_optimized(self, indicator_name: str, symbol: str, data: List[float], 
                                    period: int, **kwargs) -> str:
        """优化缓存键生成（使use哈希缓存）"""
        # 创建键元组
        key_tuple = (indicator_name, symbol, period, tuple(sorted(kwargs.items())))
        
        # check哈希缓存
        if key_tuple in self._hash_cache:
            base_key = self._hash_cache[key_tuple]
        else:
            base_str = f"{indicator_name}_{symbol}_{period}"
            for key, value in sorted(kwargs.items()):
                base_str += f"_{key}_{value}"
            base_key = hashlib.md5(base_str.encode()).hexdigest()[:16]
            self._hash_cache[key_tuple] = base_key
        
        # 数据哈希（使use滑动窗口）
        if len(data) > period * 2:
            # 只使use相关数据窗口
            relevant_data = data[-(period * 2):]
        else:
            relevant_data = data
        
        data_hash = hashlib.md5(str(relevant_data).encode()).hexdigest()[:8]
        return f"{base_key}_{data_hash}"
    
    def _compute_and_cache_enhanced(self, cache_key: str, indicator_name: str, 
                                  symbol: str, data: List[float], period: int, **kwargs) -> Any:
        """计算并缓存指标（增强版）"""
        start_time = time.time()
        
        # check数据足够性
        if len(data) < period:
            self.logger.warning(f"数据not足计算{indicator_name}: {len(data)} < {period}")
            return None
        
        # 计算指标
        if indicator_name not in self._indicator_functions:
            self.logger.error(f"未知指标: {indicator_name}")
            return None
        
        try:
            value = self._indicator_functions[indicator_name](data, period, **kwargs)
            computation_time = time.time() - start_time
            
            # 创建结果for象
            result = IndicatorResult(
                value=value,
                timestamp=time.time(),
                data_hash=hashlib.md5(str(data[-period:]).encode()).hexdigest()[:8],
                computation_time=computation_time,
                metadata={'symbol': symbol, 'indicator': indicator_name, 'period': period}
            )
            
            # 智能缓存存储
            self._store_with_policy(cache_key, result)
            
            # updates统计
            self._update_frequency(cache_key)
            
            # 预热相关指标
            if self.enable_prewarming:
                self._schedule_prewarming(symbol, data, period)
            
            self.logger.debug(f"计算并缓存: {indicator_name}({symbol}, {period}) = {value} ({computation_time:.4f}s)")
            return value
            
        except Exception as e:
            self.logger.error(f"指标计算failed {indicator_name}: {e}")
            return None
    
    def _store_with_policy(self, cache_key: str, result: IndicatorResult):
        """根据策略存储缓存"""
        # check内存使use
        self._check_memory_pressure()
        
        if self.cache_policy == CachePolicy.ADAPTIVE:
            # 自适应策略：根据访问频率and计算成本决定存储层级
            if result.computation_time > 0.01:  # 计算成本高存L1
                self._store_in_l1(cache_key, result)
            elif result.computation_time > 0.001:  # in等成本存L2
                self._store_in_l2(cache_key, result)
            else:  # 低成本can以冷存储
                self._store_in_cold(cache_key, result)
        elif self.cache_policy == CachePolicy.LRU:
            self._store_in_l1(cache_key, result)
        elif self.cache_policy == CachePolicy.LFU:
            # 根据访问频率决定
            freq = self._access_frequency[cache_key]
            if freq > 10:
                self._store_in_l1(cache_key, result)
            elif freq > 3:
                self._store_in_l2(cache_key, result)
            else:
                self._store_in_cold(cache_key, result)
        else:  # TTL
            self._store_in_l1(cache_key, result)
    
    def _store_in_l1(self, cache_key: str, result: IndicatorResult):
        """存储toL1缓存"""
        # check容量
        if len(self._l1_cache) >= self.max_cache_size // 4:  # L1占总容量25%
            self._evict_from_l1()
        
        self._l1_cache[cache_key] = result
        self._update_memory_usage()
    
    def _store_in_l2(self, cache_key: str, result: IndicatorResult):
        """存储toL2缓存"""
        if len(self._l2_cache) >= self.max_cache_size // 2:  # L2占总容量50%
            self._evict_from_l2()
        
        self._l2_cache[cache_key] = result
        self._update_memory_usage()
    
    def _store_in_cold(self, cache_key: str, result: IndicatorResult):
        """存储to冷存储"""
        try:
            serialized = pickle.dumps(result)
            if len(self._cold_storage) >= self.max_cache_size:  # 冷存储占剩余容量
                self._evict_from_cold()
            
            self._cold_storage[cache_key] = serialized
            self._update_memory_usage()
        except Exception as e:
            self.logger.warning(f"冷存储序列化failed: {e}")
    
    def _evict_from_l1(self):
        """fromL1缓存淘汰"""
        if not self._l1_cache:
            return
        
        if self.cache_policy == CachePolicy.LRU or self.cache_policy == CachePolicy.ADAPTIVE:
            # 淘汰最旧
            key, result = self._l1_cache.popitem(last=False)
        elif self.cache_policy == CachePolicy.LFU:
            # 淘汰访问频率最低
            key = min(self._l1_cache.keys(), key=lambda k: self._access_frequency[k])
            result = self._l1_cache.pop(key)
        else:  # TTL
            # 淘汰最早
            key = min(self._l1_cache.keys(), key=lambda k: self._l1_cache[k].timestamp)
            result = self._l1_cache.pop(key)
        
        # 降级toL2
        self._store_in_l2(key, result)
        self._cache_stats['evictions'] += 1
    
    def _evict_from_l2(self):
        """fromL2缓存淘汰"""
        if not self._l2_cache:
            return
        
        # 选择淘汰策略
        if self.cache_policy == CachePolicy.LFU:
            key = min(self._l2_cache.keys(), key=lambda k: self._access_frequency[k])
        else:
            key = min(self._l2_cache.keys(), key=lambda k: self._l2_cache[k].last_access)
        
        result = self._l2_cache.pop(key)
        
        # if果访问频率还can以，降级to冷存储
        if self._access_frequency[key] > 1:
            self._store_in_cold(key, result)
        
        self._cache_stats['evictions'] += 1
    
    def _evict_from_cold(self):
        """from冷存储淘汰"""
        if not self._cold_storage:
            return
        
        # 随机淘汰or基atwhen间戳
        import random
        key = random.choice(list(self._cold_storage.keys()))
        del self._cold_storage[key]
        self._cache_stats['evictions'] += 1
    
    def _promote_to_l1(self, cache_key: str, result: IndicatorResult):
        """提升toL1缓存"""
        if cache_key in self._l2_cache:
            del self._l2_cache[cache_key]
        self._store_in_l1(cache_key, result)
    
    def _promote_to_l2(self, cache_key: str, result: IndicatorResult):
        """提升toL2缓存"""
        # notin这里删除冷存储，by调use方处理
        self._store_in_l2(cache_key, result)
    
    def _update_frequency(self, cache_key: str):
        """updates访问频率"""
        old_freq = self._access_frequency[cache_key]
        new_freq = old_freq + 1
        self._access_frequency[cache_key] = new_freq
        
        # updates频率桶
        if old_freq > 0:
            self._frequency_buckets[old_freq].discard(cache_key)
        self._frequency_buckets[new_freq].add(cache_key)
    
    def _is_cache_valid(self, result: IndicatorResult) -> bool:
        """check缓存is否has效"""
        if self.cache_policy == CachePolicy.TTL:
            return (time.time() - result.timestamp) < self.ttl_seconds
        return True  # 其他策略not基atwhen间过期
    
    def _check_memory_pressure(self):
        """check内存压力"""
        if self._current_memory > self.max_memory_bytes * self._memory_threshold:
            self.logger.info("内存压力大，starting清理缓存")
            self._cleanup_memory()
    
    def _update_memory_usage(self):
        """updates内存使use统计"""
        try:
            import sys
            self._current_memory = (
                sum(sys.getsizeof(v) for v in self._l1_cache.values()) +
                sum(sys.getsizeof(v) for v in self._l2_cache.values()) +
                sum(len(v) for v in self._cold_storage.values())
            )
        except Exception:
            pass  # 内存统计failednot影响功能
    
    def _cleanup_memory(self):
        """内存清理"""
        # 清理访问频率最低一半缓存
        total_items = len(self._l1_cache) + len(self._l2_cache) + len(self._cold_storage)
        target_cleanup = total_items // 2
        
        cleaned = 0
        # 优先清理冷存储
        while cleaned < target_cleanup and self._cold_storage:
            self._evict_from_cold()
            cleaned += 1
        
        # 清理L2
        while cleaned < target_cleanup and self._l2_cache:
            key = min(self._l2_cache.keys(), key=lambda k: self._access_frequency[k])
            del self._l2_cache[key]
            cleaned += 1
        
        self.logger.info(f"内存清理completed，清理了{cleaned}个缓存 items")
        self._update_memory_usage()
    
    def _schedule_prewarming(self, symbol: str, data: List[float], period: int):
        """调度预热"""
        if not self.enable_prewarming:
            return
        
        # 预热相关周期指标
        for related_period in [period // 2, period * 2]:
            if related_period >= 5:  # 最小周期限制
                self._prewarming_queue.append((symbol, data, related_period))
    
    def prewarm_indicators(self, symbols: List[str], indicators: List[str], periods: List[int]):
        """批量预热指标"""
        if not self.enable_prewarming:
            return
        
        prewarmed = 0
        for symbol in symbols:
            for indicator in indicators:
                for period in periods:
                    # 模拟数据（实际应useinfrom数据源retrieval）
                    mock_data = [100 + i * 0.1 for i in range(period * 3)]
                    try:
                        self.get_indicator(indicator, symbol, mock_data, period)
                        prewarmed += 1
                    except Exception as e:
                        self.logger.warning(f"预热failed {indicator}({symbol}, {period}): {e}")
        
        self._cache_stats['prewarmed'] = prewarmed
        self.logger.info(f"预热completed: {prewarmed} 个指标")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """retrieval缓存统计"""
        total_hits = self._cache_stats['l1_hits'] + self._cache_stats['l2_hits'] + self._cache_stats['cold_hits']
        total_requests = total_hits + self._cache_stats['misses']
        
        return {
            'cache_stats': self._cache_stats.copy(),
            'hit_rate': total_hits / total_requests if total_requests > 0 else 0,
            'l1_size': len(self._l1_cache),
            'l2_size': len(self._l2_cache),
            'cold_size': len(self._cold_storage),
            'total_size': len(self._l1_cache) + len(self._l2_cache) + len(self._cold_storage),
            'memory_usage_mb': self._current_memory / (1024 * 1024),
            'hash_cache_size': len(self._hash_cache),
            'top_indicators': self._get_top_indicators()
        }
    
    def _get_top_indicators(self) -> List[Tuple[str, int]]:
        """retrieval最常use指标"""
        return sorted(self._access_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # 指标计算函数（简化实现）
    def _compute_sma(self, data: List[float], period: int, **kwargs) -> float:
        """简单移动平均"""
        return np.mean(data[-period:]) if len(data) >= period else None
    
    def _compute_ema(self, data: List[float], period: int, **kwargs) -> float:
        """指数移动平均"""
        alpha = 2.0 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def _compute_rsi(self, data: List[float], period: int, **kwargs) -> float:
        """相for强弱指数"""
        if len(data) < period + 1:
            return None
        
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _compute_atr(self, data: List[float], period: int, **kwargs) -> float:
        """平均真实波动范围（简化版）"""
        if len(data) < period + 1:
            return None
        
        # 简化版ATR：使useprice变化平均绝for值
        price_changes = np.abs(np.diff(data))
        return np.mean(price_changes[-period:])
    
    def _compute_bollinger(self, data: List[float], period: int, **kwargs) -> Dict[str, float]:
        """布林带"""
        std_dev = kwargs.get('std_dev', 2.0)
        
        if len(data) < period:
            return None
        
        sma = np.mean(data[-period:])
        std = np.std(data[-period:])
        
        return {
            'middle': sma,
            'upper': sma + (std_dev * std),
            'lower': sma - (std_dev * std)
        }
    
    def _compute_macd(self, data: List[float], period: int, **kwargs) -> Dict[str, float]:
        """MACD指标"""
        fast_period = kwargs.get('fast_period', 12)
        slow_period = kwargs.get('slow_period', 26)
        signal_period = kwargs.get('signal_period', 9)
        
        if len(data) < slow_period:
            return None
        
        ema_fast = self._compute_ema(data, fast_period)
        ema_slow = self._compute_ema(data, slow_period)
        macd_line = ema_fast - ema_slow
        
        # 简化信号线计算
        signal_line = macd_line  # 实际应该isMACD线EMA
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line
        }
    
    def _compute_stochastic(self, data: List[float], period: int, **kwargs) -> Dict[str, float]:
        """随机指标（简化版）"""
        if len(data) < period:
            return None
        
        recent_data = data[-period:]
        highest = max(recent_data)
        lowest = min(recent_data)
        current = data[-1]
        
        if highest == lowest:
            k_percent = 50
        else:
            k_percent = 100 * (current - lowest) / (highest - lowest)
        
        return {
            'k_percent': k_percent,
            'd_percent': k_percent  # 简化版
        }
    
    def _compute_williams_r(self, data: List[float], period: int, **kwargs) -> float:
        """威廉指标"""
        if len(data) < period:
            return None
        
        recent_data = data[-period:]
        highest = max(recent_data)
        lowest = min(recent_data)
        current = data[-1]
        
        if highest == lowest:
            return -50
        
        return -100 * (highest - current) / (highest - lowest)


# 全局增强缓存实例
_global_enhanced_cache: Optional[EnhancedIndicatorCache] = None

def get_enhanced_indicator_cache() -> EnhancedIndicatorCache:
    """retrieval全局增强指标缓存"""
    global _global_enhanced_cache
    if _global_enhanced_cache is None:
        _global_enhanced_cache = EnhancedIndicatorCache()
    return _global_enhanced_cache

def cached_indicator(indicator_name: str, symbol: str, data: List[float], 
                    period: int = 14, **kwargs) -> Any:
    """便捷缓存指标计算函数"""
    cache = get_enhanced_indicator_cache()
    return cache.get_indicator(indicator_name, symbol, data, period, **kwargs)
