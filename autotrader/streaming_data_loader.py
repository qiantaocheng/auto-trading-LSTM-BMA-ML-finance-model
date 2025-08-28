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
import psutil
import threading
from typing import List, Dict, Any, Optional, Iterator, Tuple
from datetime import datetime, timedelta
import time
from pathlib import Path
import pickle
import sqlite3
from contextlib import contextmanager
from collections import deque, defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_delta: float
    ticker: Optional[str] = None
    data_size: Optional[int] = None
    success: bool = True
    error_msg: Optional[str] = None

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_stats: Dict[str, Dict] = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'memory_usage': [],
            'error_count': 0
        })
        self._lock = threading.Lock()
        
    def record_metric(self, metric: PerformanceMetrics):
        """记录性能指标"""
        with self._lock:
            self.metrics_history.append(metric)
            
            # 更新操作统计
            stats = self.operation_stats[metric.operation]
            stats['count'] += 1
            stats['total_time'] += metric.duration
            stats['avg_time'] = stats['total_time'] / stats['count']
            stats['min_time'] = min(stats['min_time'], metric.duration)
            stats['max_time'] = max(stats['max_time'], metric.duration)
            stats['memory_usage'].append(metric.memory_after)
            
            if not metric.success:
                stats['error_count'] += 1
    
    def get_bottlenecks(self, top_n: int = 5) -> List[Dict]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        with self._lock:
            for operation, stats in self.operation_stats.items():
                if stats['count'] > 0:
                    bottleneck_score = (
                        stats['avg_time'] * stats['count'] +  # 总时间影响
                        stats['max_time'] * 2 +              # 最大时间影响  
                        stats['error_count'] * 10            # 错误次数影响
                    )
                    
                    bottlenecks.append({
                        'operation': operation,
                        'bottleneck_score': bottleneck_score,
                        'avg_time': stats['avg_time'],
                        'max_time': stats['max_time'],
                        'total_calls': stats['count'],
                        'error_rate': stats['error_count'] / stats['count'],
                        'avg_memory_mb': np.mean(stats['memory_usage']) if stats['memory_usage'] else 0
                    })
        
        # 按瓶颈评分排序
        bottlenecks.sort(key=lambda x: x['bottleneck_score'], reverse=True)
        return bottlenecks[:top_n]
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        with self._lock:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            recent_metrics = list(self.metrics_history)[-100:]  # 最近100次操作
            
            total_operations = sum(stats['count'] for stats in self.operation_stats.values())
            total_errors = sum(stats['error_count'] for stats in self.operation_stats.values())
            
            return {
                'current_memory_mb': current_memory,
                'total_operations': total_operations,
                'total_errors': total_errors,
                'error_rate': total_errors / max(1, total_operations),
                'operations_per_type': {op: stats['count'] for op, stats in self.operation_stats.items()},
                'avg_times_per_type': {op: stats['avg_time'] for op, stats in self.operation_stats.items()},
                'recent_avg_time': np.mean([m.duration for m in recent_metrics]) if recent_metrics else 0,
                'bottlenecks': self.get_bottlenecks(3)
            }

@contextmanager
def performance_tracker(monitor: PerformanceMonitor, operation: str, ticker: str = None):
    """性能跟踪上下文管理器"""
    start_time = time.time()
    memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    success = True
    error_msg = None
    data_size = None
    
    try:
        yield lambda size: setattr(locals(), 'data_size', size)  # 允许设置数据大小
    except Exception as e:
        success = False
        error_msg = str(e)
        raise
    finally:
        end_time = time.time()
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        metric = PerformanceMetrics(
            operation=operation,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_delta=memory_after - memory_before,
            ticker=ticker,
            data_size=locals().get('data_size'),
            success=success,
            error_msg=error_msg
        )
        
        monitor.record_metric(metric)

class StreamingDataLoader:
    """流式数据加载器"""
    
    def __init__(self,
                 chunk_size: int = 50,
                 cache_dir: str = "cache/streaming",
                 enable_disk_cache: bool = True,
                 memory_limit_mb: int = 512,
                 enable_performance_monitoring: bool = True):
        """
        初始化流式数据加载器
        
        Args:
            chunk_size: 每次加载的股票数量
            cache_dir: 磁盘缓存目录
            enable_disk_cache: 启用磁盘缓存
            memory_limit_mb: 内存限制(MB)
            enable_performance_monitoring: 启用性能监控
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
        
        # 性能监控
        self.enable_performance_monitoring = enable_performance_monitoring
        self.performance_monitor = PerformanceMonitor() if enable_performance_monitoring else None
        
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
        """获取单个股票数据（支持缓存和性能监控）"""
        cache_key = self._get_cache_key(ticker, data_type, start_date, end_date)
        
        # 性能监控包装器
        if self.enable_performance_monitoring:
            with performance_tracker(self.performance_monitor, "get_data", ticker) as track:
                return self._get_data_internal(cache_key, ticker, data_type, start_date, end_date, data_loader_func, track)
        else:
            return self._get_data_internal(cache_key, ticker, data_type, start_date, end_date, data_loader_func)
    
    def _get_data_internal(self, cache_key: str, ticker: str, data_type: str, 
                          start_date: str, end_date: str, data_loader_func: callable, track=None) -> Optional[pd.DataFrame]:
        """内部数据获取方法"""
        
        # 1. 检查内存缓存
        if cache_key in self.memory_cache:
            self.cache_access_times[cache_key] = time.time()
            self.cache_hits += 1
            logger.debug(f"内存缓存命中: {ticker}")
            data = self.memory_cache[cache_key]
            if track:
                track(len(data) if data is not None else 0)
            return data
        
        # 2. 检查磁盘缓存
        if self.enable_performance_monitoring and self.performance_monitor:
            with performance_tracker(self.performance_monitor, "disk_cache_load", ticker):
                cached_data = self._load_from_disk_cache(cache_key)
        else:
            cached_data = self._load_from_disk_cache(cache_key)
            
        if cached_data is not None:
            # 加载到内存缓存
            self.memory_cache[cache_key] = cached_data
            self.cache_access_times[cache_key] = time.time()
            self.cache_hits += 1
            self._check_memory_usage()
            if track:
                track(len(cached_data))
            return cached_data
        
        # 3. 缓存未命中，加载数据
        self.cache_misses += 1
        try:
            if self.enable_performance_monitoring and self.performance_monitor:
                with performance_tracker(self.performance_monitor, "data_loader_func", ticker):
                    data = data_loader_func(ticker, start_date, end_date)
            else:
                data = data_loader_func(ticker, start_date, end_date)
            
            if data is not None and not data.empty:
                # 保存到缓存
                self.memory_cache[cache_key] = data
                self.cache_access_times[cache_key] = time.time()
                
                if self.enable_performance_monitoring and self.performance_monitor:
                    with performance_tracker(self.performance_monitor, "disk_cache_save", ticker):
                        self._save_to_disk_cache(cache_key, data)
                else:
                    self._save_to_disk_cache(cache_key, data)
                    
                self._check_memory_usage()
                
                logger.debug(f"数据加载完成: {ticker} ({len(data)} 行)")
                if track:
                    track(len(data))
                return data
            else:
                logger.warning(f"数据加载失败或为空: {ticker}")
                if track:
                    track(0)
                return None
                
        except Exception as e:
            logger.error(f"数据加载异常: {ticker}, {e}")
            if track:
                track(0)
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
    
    async def preload_data_async(self,
                               tickers: List[str],
                               data_type: str,
                               start_date: str,
                               end_date: str,
                               data_loader_func: callable,
                               max_concurrent: int = 4):
        """真正的异步预加载数据"""
        import asyncio
        
        logger.info(f"开始异步预加载 {len(tickers)} 只股票")
        
        async def load_ticker_data_async(ticker):
            """异步加载单个股票数据"""
            try:
                # 在线程池中运行同步数据加载函数
                loop = asyncio.get_event_loop()
                data = await loop.run_in_executor(
                    None, 
                    self.get_data, 
                    ticker, data_type, start_date, end_date, data_loader_func
                )
                return ticker, True, data
            except Exception as e:
                logger.error(f"异步预加载失败: {ticker}, {e}")
                return ticker, False, None
        
        # 使用信号量限制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def load_with_semaphore(ticker):
            async with semaphore:
                return await load_ticker_data_async(ticker)
        
        # 创建所有任务
        tasks = [load_with_semaphore(ticker) for ticker in tickers]
        
        # 执行所有任务并收集结果
        completed = 0
        successful = 0
        
        for coro in asyncio.as_completed(tasks):
            ticker, success, data = await coro
            completed += 1
            if success:
                successful += 1
            
            if completed % 50 == 0:
                progress = completed / len(tickers) * 100
                logger.info(f"预加载进度: {progress:.1f}% ({completed}/{len(tickers)})")
        
        logger.info(f"异步预加载完成: {successful}/{len(tickers)} 股票成功")
    
    def validate_data_quality(self, data: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """数据质量检查"""
        if data is None or data.empty:
            return {
                'is_valid': False,
                'issues': ['empty_data'],
                'ticker': ticker,
                'metrics': {}
            }
        
        issues = []
        metrics = {}
        
        try:
            # 1. 基本统计
            metrics['rows'] = len(data)
            metrics['columns'] = len(data.columns)
            
            # 2. 缺失值检查
            missing_ratio = data.isnull().sum().sum() / data.size
            metrics['missing_ratio'] = missing_ratio
            if missing_ratio > 0.3:  # 超过30%缺失
                issues.append('high_missing_data')
            
            # 3. 异常值检测（使用IQR方法）
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            outlier_counts = {}
            
            for col in numeric_cols:
                if col in data.columns:
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                    outlier_ratio = outliers / len(data[col]) if len(data[col]) > 0 else 0
                    outlier_counts[col] = outlier_ratio
                    
                    if outlier_ratio > 0.1:  # 超过10%异常值
                        issues.append(f'high_outliers_{col}')
            
            metrics['outlier_ratios'] = outlier_counts
            
            # 4. 零值检查
            zero_counts = {}
            for col in numeric_cols:
                if col in data.columns:
                    zero_ratio = (data[col] == 0).sum() / len(data[col]) if len(data[col]) > 0 else 0
                    zero_counts[col] = zero_ratio
                    
                    if zero_ratio > 0.5:  # 超过50%为零
                        issues.append(f'high_zero_values_{col}')
            
            metrics['zero_ratios'] = zero_counts
            
            # 5. 重复值检查
            duplicate_ratio = data.duplicated().sum() / len(data) if len(data) > 0 else 0
            metrics['duplicate_ratio'] = duplicate_ratio
            if duplicate_ratio > 0.2:  # 超过20%重复
                issues.append('high_duplicate_data')
            
            # 6. 数据类型一致性检查
            expected_numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume']
            type_issues = []
            for col in data.columns:
                if any(expected in col.lower() for expected in ['price', 'volume', 'return']):
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        type_issues.append(col)
            
            if type_issues:
                issues.append('wrong_data_types')
                metrics['type_issues'] = type_issues
            
            # 7. 时间序列连续性检查（如果有时间索引）
            if isinstance(data.index, pd.DatetimeIndex):
                expected_freq = pd.infer_freq(data.index)
                if expected_freq is None:
                    issues.append('irregular_time_series')
                    
                # 检查时间跳跃
                time_diffs = data.index.to_series().diff().dropna()
                if len(time_diffs) > 0:
                    std_diff = time_diffs.std()
                    mean_diff = time_diffs.mean()
                    if std_diff > mean_diff * 2:  # 时间间隔变化过大
                        issues.append('irregular_time_intervals')
                        
                metrics['time_series_freq'] = expected_freq
                metrics['time_std_ratio'] = (std_diff / mean_diff).total_seconds() if mean_diff.total_seconds() > 0 else 0
            
            # 8. 价格数据合理性检查
            price_cols = [col for col in data.columns if any(p in col.lower() for p in ['open', 'high', 'low', 'close', 'price'])]
            for col in price_cols:
                if col in data.columns:
                    negative_prices = (data[col] < 0).sum()
                    if negative_prices > 0:
                        issues.append(f'negative_prices_{col}')
                    
                    # 检查OHLC逻辑关系
                    if 'high' in col.lower() and 'low' in data.columns:
                        low_col = [c for c in data.columns if 'low' in c.lower()]
                        if low_col:
                            invalid_hl = (data[col] < data[low_col[0]]).sum()
                            if invalid_hl > 0:
                                issues.append('invalid_high_low_relationship')
            
        except Exception as e:
            issues.append(f'quality_check_error: {str(e)}')
            logger.warning(f"数据质量检查异常 {ticker}: {e}")
        
        is_valid = len(issues) == 0 or all(not issue.startswith('high_') for issue in issues)
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'ticker': ticker,
            'metrics': metrics,
            'quality_score': max(0, 1 - len(issues) * 0.1)  # 每个问题扣0.1分
        }
    
    def get_data_with_quality_check(self, 
                                  ticker: str,
                                  data_type: str,
                                  start_date: str,
                                  end_date: str,
                                  data_loader_func: callable,
                                  min_quality_score: float = 0.7) -> Tuple[Optional[pd.DataFrame], Dict]:
        """获取数据并进行质量检查"""
        
        # 获取原始数据
        data = self.get_data(ticker, data_type, start_date, end_date, data_loader_func)
        
        # 质量检查
        quality_result = self.validate_data_quality(data, ticker)
        
        # 根据质量评分决定是否返回数据
        if quality_result['quality_score'] < min_quality_score:
            logger.warning(f"数据质量不合格: {ticker}, 评分: {quality_result['quality_score']:.2f}, 问题: {quality_result['issues']}")
            return None, quality_result
        
        if not quality_result['is_valid']:
            logger.info(f"数据有质量问题但可接受: {ticker}, 评分: {quality_result['quality_score']:.2f}")
        
        return data, quality_result
    
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
        
        basic_stats = {
            'memory_cache_size': len(self.memory_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate_percent': cache_hit_rate,
            'disk_reads': self.disk_reads,
            'memory_cleanups': self.memory_cleanups,
            'chunk_size': self.chunk_size,
            'memory_limit_mb': self.memory_limit_bytes / (1024 * 1024)
        }
        
        # 添加性能监控统计
        if self.enable_performance_monitoring and self.performance_monitor:
            performance_stats = self.performance_monitor.get_performance_summary()
            return {**basic_stats, 'performance': performance_stats}
        
        return basic_stats
    
    def get_performance_report(self, include_bottlenecks: bool = True) -> str:
        """生成性能报告"""
        if not self.enable_performance_monitoring or not self.performance_monitor:
            return "性能监控未启用"
        
        stats = self.get_statistics()
        perf = stats.get('performance', {})
        
        report = [
            "=== StreamingDataLoader 性能报告 ===",
            f"当前内存使用: {perf.get('current_memory_mb', 0):.1f} MB",
            f"总操作次数: {perf.get('total_operations', 0)}",
            f"总错误次数: {perf.get('total_errors', 0)}",
            f"错误率: {perf.get('error_rate', 0)*100:.2f}%",
            f"最近操作平均时间: {perf.get('recent_avg_time', 0)*1000:.1f} ms",
            "",
            "各操作平均耗时:",
        ]
        
        for op, avg_time in perf.get('avg_times_per_type', {}).items():
            count = perf.get('operations_per_type', {}).get(op, 0)
            report.append(f"  {op}: {avg_time*1000:.1f} ms (调用 {count} 次)")
        
        if include_bottlenecks:
            bottlenecks = perf.get('bottlenecks', [])
            if bottlenecks:
                report.extend([
                    "",
                    "性能瓶颈 (按影响排序):",
                ])
                
                for i, bottleneck in enumerate(bottlenecks, 1):
                    report.append(
                        f"  {i}. {bottleneck['operation']}: "
                        f"平均{bottleneck['avg_time']*1000:.1f}ms, "
                        f"最大{bottleneck['max_time']*1000:.1f}ms, "
                        f"错误率{bottleneck['error_rate']*100:.1f}%, "
                        f"瓶颈评分{bottleneck['bottleneck_score']:.1f}"
                    )
        
        return "\n".join(report)
    
    def print_performance_report(self):
        """打印性能报告"""
        print(self.get_performance_report())
    
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