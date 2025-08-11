#!/usr/bin/env python3
"""
技术指标缓存系统
优化指标计算效率，避免重复计算
"""

import numpy as np
import pandas as pd
import logging
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict, deque
from threading import RLock
from dataclasses import dataclass, field
import pickle

@dataclass
class IndicatorResult:
    """指标计算结果"""
    value: Any
    timestamp: float
    data_hash: str
    computation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class IndicatorCache:
    """技术指标缓存系统"""
    
    def __init__(self, max_cache_size: int = 1000, ttl_seconds: float = 300):
        self.logger = logging.getLogger("IndicatorCache")
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        
        # 缓存存储
        self._cache: Dict[str, IndicatorResult] = {}
        self._access_count: Dict[str, int] = defaultdict(int)
        self._access_time: Dict[str, float] = {}
        
        # 历史数据缓存
        self._price_history: Dict[str, deque] = {}
        self._max_history_size = 500
        
        # 线程锁
        self.lock = RLock()
        
        # 统计
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_computations = 0
        self.total_computation_time = 0.0
        
        # 预定义的指标函数
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
    
    def get_indicator(self, indicator_name: str, symbol: str, data: List[float], 
                     period: int = 14, **kwargs) -> Any:
        """获取技术指标值（带缓存）"""
        with self.lock:
            # 生成缓存键
            cache_key = self._generate_cache_key(indicator_name, symbol, data, period, **kwargs)
            
            # 检查缓存
            if self._is_cache_valid(cache_key):
                self.cache_hits += 1
                self._access_count[cache_key] += 1
                self._access_time[cache_key] = time.time()
                result = self._cache[cache_key]
                
                self.logger.debug(f"缓存命中: {indicator_name}({symbol}, {period})")
                return result.value
            
            # 缓存未命中，计算指标
            self.cache_misses += 1
            return self._compute_and_cache(cache_key, indicator_name, symbol, data, period, **kwargs)
    
    def _generate_cache_key(self, indicator_name: str, symbol: str, data: List[float], 
                           period: int, **kwargs) -> str:
        """生成缓存键"""
        # 创建数据哈希
        data_str = f"{symbol}_{indicator_name}_{period}"
        
        # 包含kwargs参数
        for key, value in sorted(kwargs.items()):
            data_str += f"_{key}_{value}"
        
        # 添加数据哈希（只使用最后几个值）
        if len(data) > 10:
            # 只使用最后10个数据点来生成哈希，提高效率
            recent_data = data[-10:]
        else:
            recent_data = data
        
        data_hash = hashlib.md5(str(recent_data).encode()).hexdigest()[:8]
        data_str += f"_{data_hash}"
        
        return data_str
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self._cache:
            return False
        
        result = self._cache[cache_key]
        
        # 检查TTL
        if time.time() - result.timestamp > self.ttl_seconds:
            self._remove_from_cache(cache_key)
            return False
        
        return True
    
    def _compute_and_cache(self, cache_key: str, indicator_name: str, symbol: str, 
                          data: List[float], period: int, **kwargs) -> Any:
        """计算指标并缓存"""
        start_time = time.time()
        
        try:
            # 获取计算函数
            if indicator_name not in self._indicator_functions:
                raise ValueError(f"不支持的指标: {indicator_name}")
            
            compute_func = self._indicator_functions[indicator_name]
            
            # 计算指标
            value = compute_func(data, period, **kwargs)
            
            computation_time = time.time() - start_time
            self.total_computations += 1
            self.total_computation_time += computation_time
            
            # 创建结果对象
            result = IndicatorResult(
                value=value,
                timestamp=time.time(),
                data_hash=hashlib.md5(str(data[-10:]).encode()).hexdigest()[:8],
                computation_time=computation_time,
                metadata={
                    'symbol': symbol,
                    'period': period,
                    'data_length': len(data),
                    **kwargs
                }
            )
            
            # 存储到缓存
            self._add_to_cache(cache_key, result)
            
            self.logger.debug(f"计算并缓存: {indicator_name}({symbol}, {period}) = {value}")
            
            return value
            
        except Exception as e:
            self.logger.error(f"指标计算失败 {indicator_name}: {e}")
            return None
    
    def _add_to_cache(self, cache_key: str, result: IndicatorResult):
        """添加到缓存"""
        # 检查缓存大小限制
        if len(self._cache) >= self.max_cache_size:
            self._evict_oldest()
        
        self._cache[cache_key] = result
        self._access_count[cache_key] = 1
        self._access_time[cache_key] = time.time()
    
    def _remove_from_cache(self, cache_key: str):
        """从缓存中移除"""
        self._cache.pop(cache_key, None)
        self._access_count.pop(cache_key, None)
        self._access_time.pop(cache_key, None)
    
    def _evict_oldest(self):
        """驱逐最老的缓存项"""
        if not self._cache:
            return
        
        # 按访问时间排序，移除最老的
        oldest_key = min(self._access_time.keys(), key=lambda k: self._access_time[k])
        self._remove_from_cache(oldest_key)
        
        self.logger.debug(f"驱逐缓存项: {oldest_key}")
    
    # ==================== 指标计算函数 ====================
    
    def _compute_sma(self, data: List[float], period: int, **kwargs) -> float:
        """简单移动平均"""
        if len(data) < period:
            return None
        
        return np.mean(data[-period:])
    
    def _compute_ema(self, data: List[float], period: int, **kwargs) -> float:
        """指数移动平均"""
        if len(data) < period:
            return None
        
        # 使用pandas计算EMA
        series = pd.Series(data)
        ema = series.ewm(span=period).mean()
        return ema.iloc[-1]
    
    def _compute_rsi(self, data: List[float], period: int = 14, **kwargs) -> float:
        """相对强弱指数"""
        if len(data) < period + 1:
            return None
        
        # 计算价格变化
        deltas = np.diff(data)
        
        # 分离收益和损失
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # 计算平均收益和损失
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def _compute_atr(self, highs: List[float], lows: List[float], closes: List[float], 
                    period: int = 14, **kwargs) -> float:
        """平均真实范围"""
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return None
        
        # 计算真实范围
        true_ranges = []
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        if len(true_ranges) < period:
            return None
        
        # 计算ATR
        return np.mean(true_ranges[-period:])
    
    def _compute_bollinger(self, data: List[float], period: int = 20, std_dev: float = 2.0, **kwargs) -> Dict[str, float]:
        """布林带"""
        if len(data) < period:
            return None
        
        recent_data = data[-period:]
        sma = np.mean(recent_data)
        std = np.std(recent_data)
        
        return {
            'middle': sma,
            'upper': sma + (std_dev * std),
            'lower': sma - (std_dev * std),
            'bandwidth': (2 * std_dev * std) / sma if sma != 0 else 0
        }
    
    def _compute_macd(self, data: List[float], fast_period: int = 12, slow_period: int = 26, 
                     signal_period: int = 9, **kwargs) -> Dict[str, float]:
        """MACD指标"""
        if len(data) < slow_period:
            return None
        
        # 计算快慢EMA
        series = pd.Series(data)
        ema_fast = series.ewm(span=fast_period).mean()
        ema_slow = series.ewm(span=slow_period).mean()
        
        # MACD线
        macd_line = ema_fast - ema_slow
        
        # 信号线
        signal_line = macd_line.ewm(span=signal_period).mean()
        
        # 直方图
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
    
    def _compute_stochastic(self, highs: List[float], lows: List[float], closes: List[float],
                           k_period: int = 14, d_period: int = 3, **kwargs) -> Dict[str, float]:
        """随机振荡器"""
        if len(highs) < k_period or len(lows) < k_period or len(closes) < k_period:
            return None
        
        # 计算%K
        recent_highs = highs[-k_period:]
        recent_lows = lows[-k_period:]
        current_close = closes[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            k_percent = 0
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # 计算%D（%K的移动平均）
        # 简化版本，实际应该维护%K的历史
        d_percent = k_percent  # 简化处理
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    def _compute_williams_r(self, highs: List[float], lows: List[float], closes: List[float],
                           period: int = 14, **kwargs) -> float:
        """威廉指标"""
        if len(highs) < period or len(lows) < period or len(closes) < period:
            return None
        
        recent_highs = highs[-period:]
        recent_lows = lows[-period:]
        current_close = closes[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            return 0
        
        williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
        
        return williams_r
    
    # ==================== 缓存管理 ====================
    
    def clear_cache(self):
        """清空缓存"""
        with self.lock:
            self._cache.clear()
            self._access_count.clear()
            self._access_time.clear()
            self.logger.info("指标缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
            avg_computation_time = (self.total_computation_time / self.total_computations * 1000) if self.total_computations > 0 else 0
            
            return {
                'cache_size': len(self._cache),
                'max_size': self.max_cache_size,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate_percent': hit_rate,
                'total_computations': self.total_computations,
                'avg_computation_time_ms': avg_computation_time,
                'total_computation_time_ms': self.total_computation_time * 1000
            }
    
    def optimize_cache(self):
        """优化缓存（移除过期项）"""
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, result in self._cache.items():
                if current_time - result.timestamp > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_from_cache(key)
            
            if expired_keys:
                self.logger.info(f"清理了 {len(expired_keys)} 个过期缓存项")
    
    def precompute_indicators(self, symbol: str, data: List[float], indicators: List[str]):
        """预计算指标（提高后续访问速度）"""
        with self.lock:
            for indicator in indicators:
                if indicator in self._indicator_functions:
                    try:
                        self.get_indicator(indicator, symbol, data)
                    except Exception as e:
                        self.logger.error(f"预计算指标失败 {indicator}: {e}")


# 全局指标缓存实例
_global_indicator_cache: Optional[IndicatorCache] = None

def get_indicator_cache() -> IndicatorCache:
    """获取全局指标缓存"""
    global _global_indicator_cache
    if _global_indicator_cache is None:
        _global_indicator_cache = IndicatorCache()
    return _global_indicator_cache

def cached_sma(symbol: str, data: List[float], period: int = 20) -> float:
    """缓存的SMA计算"""
    cache = get_indicator_cache()
    return cache.get_indicator('sma', symbol, data, period)

def cached_ema(symbol: str, data: List[float], period: int = 20) -> float:
    """缓存的EMA计算"""
    cache = get_indicator_cache()
    return cache.get_indicator('ema', symbol, data, period)

def cached_rsi(symbol: str, data: List[float], period: int = 14) -> float:
    """缓存的RSI计算"""
    cache = get_indicator_cache()
    return cache.get_indicator('rsi', symbol, data, period)

def cached_atr(symbol: str, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """缓存的ATR计算"""
    cache = get_indicator_cache()
    return cache.get_indicator('atr', symbol, closes, period, highs=highs, lows=lows, closes=closes)
