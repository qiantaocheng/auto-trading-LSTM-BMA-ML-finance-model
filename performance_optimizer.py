#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化模块
实现IBKR数据请求的并发处理、缓存机制和数据管道优化
"""

import time
import logging
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import queue
import json
import os
import hashlib
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import pickle
import sqlite3
from dataclasses import dataclass, field

try:
    from ib_insync import *
    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    data: Any
    timestamp: datetime
    expiry: datetime
    access_count: int = 0
    last_access: Optional[datetime] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        return datetime.now() > self.expiry
    
    def touch(self):
        self.access_count += 1
        self.last_access = datetime.now()


class DataCache:
    """智能数据缓存系统"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 缓存配置
        self.max_memory_mb = config.get('max_memory_mb', 512)
        self.default_ttl_minutes = config.get('default_ttl_minutes', 30)
        self.cleanup_interval_minutes = config.get('cleanup_interval_minutes', 5)
        self.persistence_enabled = config.get('persistence_enabled', True)
        self.persistence_file = config.get('persistence_file', 'cache/data_cache.db')
        
        # 内存缓存
        self.memory_cache = {}  # key -> CacheEntry
        self.current_memory_usage = 0
        
        # 访问统计
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 清理线程
        self.cleanup_thread = None
        self.is_running = False
        
        # 持久化数据库
        if self.persistence_enabled:
            self._init_persistence()
        
        # 启动清理线程
        self._start_cleanup_thread()
    
    def put(self, key: str, data: Any, ttl_minutes: Optional[int] = None) -> bool:
        """存储数据到缓存"""
        try:
            with self._lock:
                ttl = ttl_minutes or self.default_ttl_minutes
                expiry = datetime.now() + timedelta(minutes=ttl)
                
                # 计算数据大小
                data_size = self._calculate_size(data)
                
                # 检查是否需要清理空间
                if self._needs_eviction(data_size):
                    self._evict_lru()
                
                # 创建缓存条目
                entry = CacheEntry(
                    key=key,
                    data=data,
                    timestamp=datetime.now(),
                    expiry=expiry,
                    size_bytes=data_size
                )
                
                # 如果key已存在，先移除旧条目
                if key in self.memory_cache:
                    old_entry = self.memory_cache[key]
                    self.current_memory_usage -= old_entry.size_bytes
                
                # 添加新条目
                self.memory_cache[key] = entry
                self.current_memory_usage += data_size
                
                # 持久化
                if self.persistence_enabled:
                    self._persist_entry(entry)
                
                self.logger.debug(f"Cached data for key: {key} (size: {data_size} bytes, TTL: {ttl} minutes)")
                return True
                
        except Exception as e:
            self.logger.error(f"Error caching data for key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        try:
            with self._lock:
                # 检查内存缓存
                if key in self.memory_cache:
                    entry = self.memory_cache[key]
                    
                    if entry.is_expired():
                        # 过期数据，移除
                        self._remove_entry(key)
                        self.miss_count += 1
                        return None
                    
                    # 更新访问信息
                    entry.touch()
                    self.hit_count += 1
                    return entry.data
                
                # 检查持久化存储
                if self.persistence_enabled:
                    data = self._load_from_persistence(key)
                    if data is not None:
                        # 重新放入内存缓存
                        self.put(key, data)
                        self.hit_count += 1
                        return data
                
                self.miss_count += 1
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting data for key {key}: {e}")
            self.miss_count += 1
            return None
    
    def has(self, key: str) -> bool:
        """检查缓存中是否存在指定key"""
        with self._lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not entry.is_expired():
                    return True
                else:
                    self._remove_entry(key)
            
            if self.persistence_enabled:
                return self._exists_in_persistence(key)
            
            return False
    
    def invalidate(self, key: str) -> bool:
        """使指定key的缓存失效"""
        try:
            with self._lock:
                if key in self.memory_cache:
                    self._remove_entry(key)
                
                if self.persistence_enabled:
                    self._remove_from_persistence(key)
                
                self.logger.debug(f"Invalidated cache for key: {key}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error invalidating cache for key {key}: {e}")
            return False
    
    def clear(self):
        """清空所有缓存"""
        try:
            with self._lock:
                self.memory_cache.clear()
                self.current_memory_usage = 0
                
                if self.persistence_enabled:
                    self._clear_persistence()
                
                self.logger.info("Cache cleared")
                
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate_pct': hit_rate,
                'eviction_count': self.eviction_count,
                'memory_usage_mb': self.current_memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_mb,
                'memory_utilization_pct': (self.current_memory_usage / (self.max_memory_mb * 1024 * 1024)) * 100,
                'entry_count': len(self.memory_cache)
            }
    
    def _calculate_size(self, data: Any) -> int:
        """计算数据大小（字节）"""
        try:
            if isinstance(data, (str, bytes)):
                return len(data)
            elif isinstance(data, (pd.DataFrame, pd.Series)):
                return data.memory_usage(deep=True).sum()
            else:
                # 使用pickle序列化来估算大小
                return len(pickle.dumps(data))
        except:
            return 1024  # 默认1KB
    
    def _needs_eviction(self, new_data_size: int) -> bool:
        """检查是否需要清理空间"""
        max_bytes = self.max_memory_mb * 1024 * 1024
        return (self.current_memory_usage + new_data_size) > max_bytes
    
    def _evict_lru(self):
        """最近最少使用算法清理缓存"""
        try:
            # 按最后访问时间排序
            entries = list(self.memory_cache.values())
            entries.sort(key=lambda x: x.last_access or x.timestamp)
            
            # 清理最老的条目，直到有足够空间
            max_bytes = self.max_memory_mb * 1024 * 1024
            target_size = max_bytes * 0.8  # 清理到80%使用率
            
            for entry in entries:
                if self.current_memory_usage <= target_size:
                    break
                
                self._remove_entry(entry.key)
                self.eviction_count += 1
                
        except Exception as e:
            self.logger.error(f"Error in LRU eviction: {e}")
    
    def _remove_entry(self, key: str):
        """移除缓存条目"""
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            self.current_memory_usage -= entry.size_bytes
            del self.memory_cache[key]
    
    def _start_cleanup_thread(self):
        """启动清理线程"""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return
        
        self.is_running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                with self._lock:
                    expired_keys = []
                    for key, entry in self.memory_cache.items():
                        if entry.is_expired():
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        self._remove_entry(key)
                    
                    if expired_keys:
                        self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                time.sleep(self.cleanup_interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                time.sleep(60)
    
    def _init_persistence(self):
        """初始化持久化存储"""
        try:
            os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)
            
            conn = sqlite3.connect(self.persistence_file)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_data (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    timestamp TEXT,
                    expiry TEXT
                )
            ''')
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing persistence: {e}")
            self.persistence_enabled = False
    
    def _persist_entry(self, entry: CacheEntry):
        """持久化缓存条目"""
        if not self.persistence_enabled:
            return
        
        try:
            conn = sqlite3.connect(self.persistence_file)
            data_blob = pickle.dumps(entry.data)
            
            conn.execute('''
                INSERT OR REPLACE INTO cache_data (key, data, timestamp, expiry)
                VALUES (?, ?, ?, ?)
            ''', (entry.key, data_blob, entry.timestamp.isoformat(), entry.expiry.isoformat()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error persisting entry {entry.key}: {e}")
    
    def _load_from_persistence(self, key: str) -> Optional[Any]:
        """从持久化存储加载数据"""
        if not self.persistence_enabled:
            return None
        
        try:
            conn = sqlite3.connect(self.persistence_file)
            cursor = conn.execute(
                'SELECT data, expiry FROM cache_data WHERE key = ?', (key,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                data_blob, expiry_str = row
                expiry = datetime.fromisoformat(expiry_str)
                
                if datetime.now() > expiry:
                    # 过期数据，删除
                    self._remove_from_persistence(key)
                    return None
                
                return pickle.loads(data_blob)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading from persistence for key {key}: {e}")
            return None
    
    def _exists_in_persistence(self, key: str) -> bool:
        """检查持久化存储中是否存在key"""
        if not self.persistence_enabled:
            return False
        
        try:
            conn = sqlite3.connect(self.persistence_file)
            cursor = conn.execute(
                'SELECT expiry FROM cache_data WHERE key = ?', (key,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                expiry = datetime.fromisoformat(row[0])
                return datetime.now() <= expiry
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking persistence for key {key}: {e}")
            return False
    
    def _remove_from_persistence(self, key: str):
        """从持久化存储中移除数据"""
        if not self.persistence_enabled:
            return
        
        try:
            conn = sqlite3.connect(self.persistence_file)
            conn.execute('DELETE FROM cache_data WHERE key = ?', (key,))
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error removing from persistence for key {key}: {e}")
    
    def _clear_persistence(self):
        """清空持久化存储"""
        if not self.persistence_enabled:
            return
        
        try:
            conn = sqlite3.connect(self.persistence_file)
            conn.execute('DELETE FROM cache_data')
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error clearing persistence: {e}")
    
    def stop(self):
        """停止缓存系统"""
        self.is_running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)


class ConcurrentDataDownloader:
    """并发数据下载器"""
    
    def __init__(self, ib_connection, cache: DataCache, config: Dict[str, Any] = None):
        self.ib = ib_connection
        self.cache = cache
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 并发配置
        self.max_concurrent_requests = config.get('max_concurrent_requests', 5)
        self.request_timeout = config.get('request_timeout_seconds', 30)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay_seconds', 1)
        
        # 限流配置
        self.rate_limit_per_second = config.get('rate_limit_per_second', 50)
        self.rate_limit_window = timedelta(seconds=1)
        
        # 请求队列和限流
        self.request_queue = queue.Queue()
        self.rate_limiter = deque()
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_requests)
        
        # 统计信息
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cache_hits = 0
        
        # 请求历史（用于调试）
        self.request_history = deque(maxlen=1000)
    
    def download_historical_data_batch(self, symbols: List[str], duration: str = "1 M", 
                                     bar_size: str = "1 day", what_to_show: str = "TRADES",
                                     use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """批量下载历史数据"""
        results = {}
        futures = {}
        
        # 检查缓存
        if use_cache:
            cached_results = self._check_cache_batch(symbols, duration, bar_size, what_to_show)
            results.update(cached_results)
            
            # 移除已缓存的符号
            symbols = [s for s in symbols if s not in cached_results]
            self.cache_hits += len(cached_results)
        
        if not symbols:
            return results
        
        # 提交并发请求
        for symbol in symbols:
            future = self.executor.submit(
                self._download_single_historical,
                symbol, duration, bar_size, what_to_show, use_cache
            )
            futures[future] = symbol
        
        # 收集结果
        for future in as_completed(futures, timeout=self.request_timeout * 2):
            symbol = futures[future]
            try:
                data = future.result()
                if data is not None:
                    results[symbol] = data
                    self.successful_requests += 1
                else:
                    self.failed_requests += 1
                    self.logger.warning(f"Failed to download data for {symbol}")
                    
            except Exception as e:
                self.failed_requests += 1
                self.logger.error(f"Error downloading data for {symbol}: {e}")
        
        self.total_requests += len(symbols)
        return results
    
    def _download_single_historical(self, symbol: str, duration: str, bar_size: str, 
                                  what_to_show: str, use_cache: bool) -> Optional[pd.DataFrame]:
        """下载单个股票的历史数据"""
        cache_key = self._generate_cache_key(symbol, duration, bar_size, what_to_show)
        
        # 检查缓存
        if use_cache and self.cache.has(cache_key):
            data = self.cache.get(cache_key)
            if data is not None:
                return data
        
        # 限流
        self._apply_rate_limit()
        
        # 重试机制
        for attempt in range(self.retry_attempts):
            try:
                # 创建合约
                contract = self._create_contract(symbol)
                if not contract:
                    return None
                
                # 请求历史数据
                bars = self.ib.reqHistoricalData(
                    contract=contract,
                    endDateTime='',
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=True,
                    formatDate=1,
                    timeout=self.request_timeout
                )
                
                if bars:
                    # 转换为DataFrame
                    df = util.df(bars)
                    df['symbol'] = symbol
                    
                    # 缓存结果
                    if use_cache:
                        ttl_minutes = self._get_cache_ttl(bar_size)
                        self.cache.put(cache_key, df, ttl_minutes)
                    
                    # 记录请求历史
                    self._record_request(symbol, duration, bar_size, True, attempt + 1)
                    
                    return df
                else:
                    self.logger.warning(f"No data returned for {symbol} (attempt {attempt + 1})")
                    
            except Exception as e:
                self.logger.error(f"Error downloading {symbol} (attempt {attempt + 1}): {e}")
                
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
        
        # 记录失败的请求
        self._record_request(symbol, duration, bar_size, False, self.retry_attempts)
        return None
    
    def download_real_time_bars_batch(self, symbols: List[str], 
                                    callback: Optional[Callable] = None) -> Dict[str, Any]:
        """批量订阅实时数据"""
        results = {}
        
        for symbol in symbols:
            try:
                contract = self._create_contract(symbol)
                if contract:
                    # 订阅实时5秒bar数据
                    bars = self.ib.reqRealTimeBars(contract, 5, 'TRADES', False)
                    if bars:
                        results[symbol] = bars
                        
                        # 设置回调
                        if callback:
                            bars.updateEvent += lambda bars, hasNewBar: callback(symbol, bars, hasNewBar)
                        
                        self.successful_requests += 1
                    else:
                        self.failed_requests += 1
                        
            except Exception as e:
                self.logger.error(f"Error subscribing to real-time bars for {symbol}: {e}")
                self.failed_requests += 1
        
        self.total_requests += len(symbols)
        return results
    
    def _check_cache_batch(self, symbols: List[str], duration: str, bar_size: str, 
                          what_to_show: str) -> Dict[str, pd.DataFrame]:
        """批量检查缓存"""
        results = {}
        
        for symbol in symbols:
            cache_key = self._generate_cache_key(symbol, duration, bar_size, what_to_show)
            data = self.cache.get(cache_key)
            if data is not None:
                results[symbol] = data
        
        return results
    
    def _generate_cache_key(self, symbol: str, duration: str, bar_size: str, what_to_show: str) -> str:
        """生成缓存键"""
        key_parts = [symbol, duration, bar_size, what_to_show]
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_ttl(self, bar_size: str) -> int:
        """根据bar大小确定缓存TTL（分钟）"""
        ttl_mapping = {
            "1 min": 1,
            "5 mins": 5,
            "15 mins": 15,
            "30 mins": 30,
            "1 hour": 60,
            "1 day": 24 * 60,
            "1 week": 7 * 24 * 60,
            "1 month": 30 * 24 * 60
        }
        return ttl_mapping.get(bar_size, 30)  # 默认30分钟
    
    def _create_contract(self, symbol: str) -> Optional[Contract]:
        """创建交易合约"""
        try:
            if '.' in symbol:
                base_symbol = symbol.split('.')[0]
                if symbol.endswith('.HK'):
                    return Stock(base_symbol, 'SEHK', 'HKD')
                else:
                    return Stock(symbol, 'SMART', 'USD')
            else:
                return Stock(symbol, 'SMART', 'USD')
        except Exception as e:
            self.logger.error(f"Error creating contract for {symbol}: {e}")
            return None
    
    def _apply_rate_limit(self):
        """应用速率限制"""
        now = datetime.now()
        
        # 清理过期的请求记录
        while self.rate_limiter and (now - self.rate_limiter[0]) > self.rate_limit_window:
            self.rate_limiter.popleft()
        
        # 检查是否超过速率限制
        if len(self.rate_limiter) >= self.rate_limit_per_second:
            sleep_time = (self.rate_limiter[0] + self.rate_limit_window - now).total_seconds()
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # 记录当前请求
        self.rate_limiter.append(now)
    
    def _record_request(self, symbol: str, duration: str, bar_size: str, 
                       success: bool, attempts: int):
        """记录请求历史"""
        record = {
            'symbol': symbol,
            'duration': duration,
            'bar_size': bar_size,
            'success': success,
            'attempts': attempts,
            'timestamp': datetime.now()
        }
        self.request_history.append(record)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate_pct': success_rate,
            'cache_hits': self.cache_hits,
            'cache_hit_rate_pct': (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0,
            'avg_requests_per_second': len(self.rate_limiter),
            'rate_limit_per_second': self.rate_limit_per_second
        }
    
    def cleanup(self):
        """清理资源"""
        self.executor.shutdown(wait=True)


class DataPipeline:
    """数据处理管道"""
    
    def __init__(self, downloader: ConcurrentDataDownloader, config: Dict[str, Any] = None):
        self.downloader = downloader
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 处理管道
        self.processors = []
        self.processing_stats = defaultdict(int)
        
        # 异步处理队列
        self.processing_queue = queue.Queue()
        self.processing_threads = []
        self.is_processing = False
        
        # 数据质量检查
        self.quality_checks_enabled = config.get('quality_checks_enabled', True)
        self.min_data_points = config.get('min_data_points', 20)
        self.max_gap_days = config.get('max_gap_days', 7)
    
    def add_processor(self, processor: Callable[[pd.DataFrame], pd.DataFrame]):
        """添加数据处理器"""
        self.processors.append(processor)
    
    def process_data_batch(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """批量处理数据"""
        processed_data = {}
        
        for symbol, df in data_dict.items():
            try:
                processed_df = self._process_single(symbol, df)
                if processed_df is not None:
                    processed_data[symbol] = processed_df
                    self.processing_stats['successful'] += 1
                else:
                    self.processing_stats['failed'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing data for {symbol}: {e}")
                self.processing_stats['failed'] += 1
        
        return processed_data
    
    def _process_single(self, symbol: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """处理单个数据集"""
        # 数据质量检查
        if self.quality_checks_enabled:
            if not self._quality_check(symbol, df):
                return None
        
        # 应用处理器
        processed_df = df.copy()
        for processor in self.processors:
            try:
                processed_df = processor(processed_df)
            except Exception as e:
                self.logger.error(f"Error in processor for {symbol}: {e}")
                return None
        
        return processed_df
    
    def _quality_check(self, symbol: str, df: pd.DataFrame) -> bool:
        """数据质量检查"""
        try:
            # 检查数据点数量
            if len(df) < self.min_data_points:
                self.logger.warning(f"Insufficient data points for {symbol}: {len(df)} < {self.min_data_points}")
                return False
            
            # 检查关键列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"Missing columns for {symbol}: {missing_columns}")
                return False
            
            # 检查数据完整性
            if df[required_columns].isnull().any().any():
                self.logger.warning(f"Null values found in {symbol} data")
                return False
            
            # 检查价格合理性
            if (df['high'] < df['low']).any():
                self.logger.warning(f"Invalid price data for {symbol}: high < low")
                return False
            
            if (df['close'] > df['high']).any() or (df['close'] < df['low']).any():
                self.logger.warning(f"Invalid price data for {symbol}: close outside high-low range")
                return False
            
            # 检查时间间隔
            if 'date' in df.columns or df.index.name == 'date':
                date_col = df.index if df.index.name == 'date' else df['date']
                if len(date_col) > 1:
                    max_gap = (date_col.max() - date_col.min()).days
                    if max_gap > len(date_col) + self.max_gap_days:
                        self.logger.warning(f"Large data gaps detected for {symbol}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in quality check for {symbol}: {e}")
            return False


# 整合性能优化器
class PerformanceOptimizer:
    """性能优化器 - 整合所有优化组件"""
    
    def __init__(self, ib_connection, config: Dict[str, Any] = None):
        self.ib = ib_connection
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        cache_config = config.get('cache', {})
        self.cache = DataCache(cache_config)
        
        downloader_config = config.get('downloader', {})
        self.downloader = ConcurrentDataDownloader(ib_connection, self.cache, downloader_config)
        
        pipeline_config = config.get('pipeline', {})
        self.pipeline = DataPipeline(self.downloader, pipeline_config)
        
        # 添加默认数据处理器
        self._setup_default_processors()
    
    def _setup_default_processors(self):
        """设置默认数据处理器"""
        def technical_indicators_processor(df: pd.DataFrame) -> pd.DataFrame:
            """添加技术指标"""
            try:
                # RSI
                if len(df) >= 14:
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                
                # 移动平均
                if len(df) >= 20:
                    df['ma_20'] = df['close'].rolling(window=20).mean()
                if len(df) >= 50:
                    df['ma_50'] = df['close'].rolling(window=50).mean()
                
                # 布林带
                if len(df) >= 20:
                    ma_20 = df['close'].rolling(window=20).mean()
                    std_20 = df['close'].rolling(window=20).std()
                    df['bb_upper'] = ma_20 + (std_20 * 2)
                    df['bb_lower'] = ma_20 - (std_20 * 2)
                
                return df
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Error in technical indicators processor: {e}")
                return df
        
        self.pipeline.add_processor(technical_indicators_processor)
    
    def download_and_process_batch(self, symbols: List[str], duration: str = "1 M", 
                                 bar_size: str = "1 day", use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """下载并处理批量数据"""
        # 下载数据
        raw_data = self.downloader.download_historical_data_batch(
            symbols, duration, bar_size, use_cache=use_cache
        )
        
        # 处理数据
        processed_data = self.pipeline.process_data_batch(raw_data)
        
        return processed_data
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'cache_stats': self.cache.get_stats(),
            'downloader_stats': self.downloader.get_performance_stats(),
            'processing_stats': dict(self.pipeline.processing_stats),
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup(self):
        """清理资源"""
        self.cache.stop()
        self.downloader.cleanup()
        self.logger.info("Performance optimizer cleaned up")


# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 配置
    config = {
        'cache': {
            'max_memory_mb': 512,
            'default_ttl_minutes': 30,
            'persistence_enabled': True,
            'persistence_file': 'cache/data_cache.db'
        },
        'downloader': {
            'max_concurrent_requests': 5,
            'request_timeout_seconds': 30,
            'retry_attempts': 3,
            'rate_limit_per_second': 50
        },
        'pipeline': {
            'quality_checks_enabled': True,
            'min_data_points': 20,
            'max_gap_days': 7
        }
    }
    
    # 模拟IB连接
    class MockIB:
        def reqHistoricalData(self, *args, **kwargs):
            # 模拟返回空数据
            return []
    
    mock_ib = MockIB()
    
    # 创建性能优化器
    optimizer = PerformanceOptimizer(mock_ib, config)
    
    # 测试
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data = optimizer.download_and_process_batch(symbols)
    
    # 性能报告
    report = optimizer.get_performance_report()
    print("Performance Report:", json.dumps(report, indent=2))
    
    print("Performance optimizer initialized successfully")