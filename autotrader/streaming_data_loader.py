#!/usr/bin/env python3
"""
流式数据加载器
解决大规模股票数据内存占用问题
"""

import gc
import logging
import numpy as np
import pandas as pd
import sys
from typing import List, Dict, Any, Optional, Iterator, Tuple
from datetime import datetime, timedelta
import time
from pathlib import Path
import pickle
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class StreamingDataLoader:
    """流式数据加载器"""
    
    def __init__(self,
                 chunk_size: int = 50,
                 cache_dir: str = "cache/streaming",
                 enable_disk_cache: bool = True,
                 memory_limit_mb: int = 512):
        """
        初始化流式数据加载器
        
        Args:
            chunk_size: 每次加载的股票数量
            cache_dir: 磁盘缓存目录
            enable_disk_cache: 启用磁盘缓存
            memory_limit_mb: 内存限制(MB)
        """
        self.chunk_size = chunk_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_disk_cache = enable_disk_cache
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        
        # 数据缓存
        self.memory_cache = {}
        self.cache_access_times = {}
        self.disk_cache_db = self.cache_dir / "disk_cache.db"
        
        # 统计信息
        self.cache_hits = 0
        self.cache_misses = 0
        self.disk_reads = 0
        self.memory_cleanups = 0
        
        # 初始化磁盘缓存数据库
        if self.enable_disk_cache:
            self._init_disk_cache()
    
    def _init_disk_cache(self):
        """初始化磁盘缓存数据库"""
        with sqlite3.connect(self.disk_cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_cache (
                    key TEXT PRIMARY KEY,
                    data BLOB,
                    timestamp REAL,
                    size_bytes INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON data_cache(timestamp)")
    
    def _get_cache_key(self, ticker: str, data_type: str, start_date: str, end_date: str) -> str:
        """生成缓存键"""
        return f"{ticker}_{data_type}_{start_date}_{end_date}"
    
    def _check_memory_usage(self):
        """检查内存使用并清理"""
        total_size = sum(
            sys.getsizeof(data) for data in self.memory_cache.values()
        )
        
        if total_size > self.memory_limit_bytes:
            self._cleanup_memory_cache()
    
    def _cleanup_memory_cache(self):
        """清理内存缓存"""
        # 按访问时间排序，删除最旧的缓存
        current_time = time.time()
        sorted_items = sorted(
            self.cache_access_times.items(),
            key=lambda x: x[1]
        )
        
        # 删除一半最旧的缓存
        items_to_remove = len(sorted_items) // 2
        for i in range(items_to_remove):
            key, _ = sorted_items[i]
            if key in self.memory_cache:
                del self.memory_cache[key]
            if key in self.cache_access_times:
                del self.cache_access_times[key]
        
        gc.collect()
        self.memory_cleanups += 1
        logger.debug(f"清理了 {items_to_remove} 个内存缓存项")
    
    def _save_to_disk_cache(self, key: str, data: Any):
        """保存数据到磁盘缓存"""
        if not self.enable_disk_cache:
            return
        
        try:
            serialized_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            size_bytes = len(serialized_data)
            timestamp = time.time()
            
            with sqlite3.connect(self.disk_cache_db) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO data_cache (key, data, timestamp, size_bytes) VALUES (?, ?, ?, ?)",
                    (key, serialized_data, timestamp, size_bytes)
                )
            
            logger.debug(f"数据已保存到磁盘缓存: {key} ({size_bytes / 1024:.1f}KB)")
            
        except Exception as e:
            logger.warning(f"保存磁盘缓存失败: {e}")
    
    def _load_from_disk_cache(self, key: str) -> Optional[Any]:
        """从磁盘缓存加载数据"""
        if not self.enable_disk_cache:
            return None
        
        try:
            with sqlite3.connect(self.disk_cache_db) as conn:
                cursor = conn.execute(
                    "SELECT data FROM data_cache WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    data = pickle.loads(row[0])
                    self.disk_reads += 1
                    logger.debug(f"从磁盘缓存加载: {key}")
                    return data
                    
        except Exception as e:
            logger.warning(f"加载磁盘缓存失败: {e}")
        
        return None
    
    def _cleanup_old_disk_cache(self, max_age_days: int = 7):
        """清理旧的磁盘缓存"""
        if not self.enable_disk_cache:
            return
        
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        try:
            with sqlite3.connect(self.disk_cache_db) as conn:
                cursor = conn.execute(
                    "DELETE FROM data_cache WHERE timestamp < ?",
                    (cutoff_time,)
                )
                deleted_count = cursor.rowcount
                
            if deleted_count > 0:
                logger.info(f"清理了 {deleted_count} 个过期磁盘缓存项")
                
        except Exception as e:
            logger.warning(f"清理磁盘缓存失败: {e}")
    
    def get_data(self, 
                 ticker: str,
                 data_type: str,
                 start_date: str,
                 end_date: str,
                 data_loader_func: callable) -> Optional[pd.DataFrame]:
        """获取单个股票数据（支持缓存）"""
        cache_key = self._get_cache_key(ticker, data_type, start_date, end_date)
        
        # 1. 检查内存缓存
        if cache_key in self.memory_cache:
            self.cache_access_times[cache_key] = time.time()
            self.cache_hits += 1
            logger.debug(f"内存缓存命中: {ticker}")
            return self.memory_cache[cache_key]
        
        # 2. 检查磁盘缓存
        cached_data = self._load_from_disk_cache(cache_key)
        if cached_data is not None:
            # 加载到内存缓存
            self.memory_cache[cache_key] = cached_data
            self.cache_access_times[cache_key] = time.time()
            self.cache_hits += 1
            self._check_memory_usage()
            return cached_data
        
        # 3. 缓存未命中，加载数据
        self.cache_misses += 1
        try:
            data = data_loader_func(ticker, start_date, end_date)
            
            if data is not None and not data.empty:
                # 保存到缓存
                self.memory_cache[cache_key] = data
                self.cache_access_times[cache_key] = time.time()
                self._save_to_disk_cache(cache_key, data)
                self._check_memory_usage()
                
                logger.debug(f"数据加载完成: {ticker} ({len(data)} 行)")
                return data
            else:
                logger.warning(f"数据加载失败或为空: {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"数据加载异常: {ticker}, {e}")
            return None
    
    def stream_data_chunks(self,
                          tickers: List[str],
                          data_type: str,
                          start_date: str,
                          end_date: str,
                          data_loader_func: callable) -> Iterator[Tuple[List[str], Dict[str, pd.DataFrame]]]:
        """流式加载数据块"""
        total_tickers = len(tickers)
        processed = 0
        
        logger.info(f"开始流式加载 {total_tickers} 只股票，块大小 {self.chunk_size}")
        
        for i in range(0, total_tickers, self.chunk_size):
            chunk_tickers = tickers[i:i + self.chunk_size]
            chunk_data = {}
            
            chunk_start_time = time.time()
            
            # 加载当前块的所有数据
            for ticker in chunk_tickers:
                data = self.get_data(ticker, data_type, start_date, end_date, data_loader_func)
                if data is not None:
                    chunk_data[ticker] = data
            
            chunk_time = time.time() - chunk_start_time
            processed += len(chunk_tickers)
            progress = processed / total_tickers * 100
            
            logger.info(f"块 {i//self.chunk_size + 1} 加载完成: {len(chunk_data)}/{len(chunk_tickers)} 股票, "
                       f"用时 {chunk_time:.1f}秒, 进度 {progress:.1f}%")
            
            yield chunk_tickers, chunk_data
            
            # 强制垃圾回收
            del chunk_data
            gc.collect()
    
    def preload_data_async(self,
                          tickers: List[str],
                          data_type: str,
                          start_date: str,
                          end_date: str,
                          data_loader_func: callable,
                          max_workers: int = 4):
        """异步预加载数据"""
        import concurrent.futures
        import threading
        
        logger.info(f"开始异步预加载 {len(tickers)} 只股票")
        
        def load_ticker_data(ticker):
            return self.get_data(ticker, data_type, start_date, end_date, data_loader_func)
        
        # 使用线程池并行加载
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_ticker = {
                executor.submit(load_ticker_data, ticker): ticker 
                for ticker in tickers
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    future.result()
                    completed += 1
                    
                    if completed % 50 == 0:
                        progress = completed / len(tickers) * 100
                        logger.info(f"预加载进度: {progress:.1f}% ({completed}/{len(tickers)})")
                        
                except Exception as e:
                    logger.error(f"预加载失败: {ticker}, {e}")
        
        logger.info(f"预加载完成: {completed}/{len(tickers)} 股票")
    
    def get_feature_data_streaming(self,
                                  tickers: List[str],
                                  feature_calculator_func: callable,
                                  start_date: str,
                                  end_date: str,
                                  data_loader_func: callable) -> Iterator[Tuple[str, pd.DataFrame]]:
        """流式计算特征数据"""
        logger.info(f"开始流式特征计算: {len(tickers)} 股票")
        
        for ticker in tickers:
            try:
                # 流式加载原始数据
                raw_data = self.get_data(ticker, "raw", start_date, end_date, data_loader_func)
                
                if raw_data is not None and not raw_data.empty:
                    # 计算特征
                    features = feature_calculator_func(raw_data, ticker)
                    
                    if features is not None and not features.empty:
                        yield ticker, features
                    
                    # 清理原始数据
                    del raw_data
                    gc.collect()
                else:
                    logger.warning(f"跳过特征计算: {ticker} (数据为空)")
                    
            except Exception as e:
                logger.error(f"特征计算失败: {ticker}, {e}")
                continue
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取加载器统计信息"""
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) * 100 if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            'memory_cache_size': len(self.memory_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate_percent': cache_hit_rate,
            'disk_reads': self.disk_reads,
            'memory_cleanups': self.memory_cleanups,
            'chunk_size': self.chunk_size,
            'memory_limit_mb': self.memory_limit_bytes / (1024 * 1024)
        }
    
    def clear_cache(self):
        """清理所有缓存"""
        # 清理内存缓存
        self.memory_cache.clear()
        self.cache_access_times.clear()
        
        # 清理磁盘缓存
        if self.enable_disk_cache:
            try:
                with sqlite3.connect(self.disk_cache_db) as conn:
                    conn.execute("DELETE FROM data_cache")
                logger.info("磁盘缓存已清理")
            except Exception as e:
                logger.warning(f"清理磁盘缓存失败: {e}")
        
        # 重置统计
        self.cache_hits = 0
        self.cache_misses = 0
        self.disk_reads = 0
        self.memory_cleanups = 0
        
        gc.collect()
        logger.info("所有缓存已清理")


class ChunkedDataProcessor:
    """分块数据处理器"""
    
    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size
    
    def process_dataframe_chunks(self, 
                                df: pd.DataFrame,
                                process_func: callable) -> Iterator[pd.DataFrame]:
        """分块处理DataFrame"""
        total_rows = len(df)
        
        for start_idx in range(0, total_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx].copy()
            
            try:
                processed_chunk = process_func(chunk)
                yield processed_chunk
            finally:
                del chunk
                gc.collect()
    
    def aggregate_chunks(self, 
                        chunks: Iterator[pd.DataFrame],
                        aggregation_func: callable = None) -> pd.DataFrame:
        """聚合处理后的数据块"""
        if aggregation_func is None:
            aggregation_func = pd.concat
        
        chunk_list = list(chunks)
        try:
            result = aggregation_func(chunk_list, ignore_index=True)
            return result
        finally:
            for chunk in chunk_list:
                del chunk
            gc.collect()


def create_streaming_loader(**kwargs) -> StreamingDataLoader:
    """创建流式数据加载器的工厂函数"""
    return StreamingDataLoader(**kwargs)