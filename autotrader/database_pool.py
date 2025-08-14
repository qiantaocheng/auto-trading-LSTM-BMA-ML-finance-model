#!/usr/bin/env python3
"""
数据库connection池管理
提供高效SQLiteconnection复useand管理
"""

import sqlite3
import threading
import time
import logging
import queue
import os
from typing import Optional, Dict, Any, ContextManager, List, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
import weakref

@dataclass
class ConnectionStats:
    """connection统计信息"""
    total_created: int = 0
    total_closed: int = 0
    current_active: int = 0
    current_idle: int = 0
    total_operations: int = 0
    total_time: float = 0.0
    peak_active: int = 0
    errors: int = 0

class PooledConnection:
    """池化connection包装器"""
    
    def __init__(self, connection: sqlite3.Connection, pool: 'DatabasePool'):
        self.connection = connection
        self.pool = weakref.ref(pool)
        self.created_at = time.time()
        self.last_used = time.time()
        self.operations_count = 0
        self.is_active = False
        self.connection_id = id(connection)
        
        # 配置connection
        self._configure_connection()
    
    def _configure_connection(self):
        """配置connection参数"""
        # 优化SQLitesettings
        pragmas = [
            "PRAGMA journal_mode=WAL",           # WAL模式，支持并发读
            "PRAGMA synchronous=NORMAL",        # 平衡安全性and性能
            "PRAGMA temp_store=MEMORY",         # 临when表使use内存
            "PRAGMA cache_size=10000",          # 10MB缓存
            "PRAGMA busy_timeout=30000",        # 30 seconds超when
            "PRAGMA foreign_keys=ON",           # 启use外键约束
            "PRAGMA optimize"                   # 优化查询计划
        ]
        
        for pragma in pragmas:
            try:
                self.connection.execute(pragma)
            except sqlite3.Error as e:
                logging.warning(f"Failed to execute pragma '{pragma}': {e}")
    
    def execute(self, query: str, params: tuple = (), fetch_result: bool = True):
        """执行SQL查询"""
        self.last_used = time.time()
        self.operations_count += 1
        self.is_active = True
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            
            # 自动提交修改操作
            if not fetch_result or query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER')):
                self.connection.commit()
                return cursor
            else:
                return cursor.fetchall()
                
        finally:
            self.is_active = False
    
    def executemany(self, query: str, params_list: List[tuple]):
        """批量执行SQL"""
        self.last_used = time.time()
        self.operations_count += len(params_list)
        self.is_active = True
        
        try:
            cursor = self.connection.cursor()
            cursor.executemany(query, params_list)
            self.connection.commit()
            return cursor
        finally:
            self.is_active = False
    
    def begin_transaction(self):
        """Start transaction"""
        self.connection.execute("BEGIN")
    
    def commit(self):
        """提交事务"""
        self.connection.commit()
    
    def rollback(self):
        """回滚事务"""
        self.connection.rollback()
    
    def close(self):
        """关闭connection"""
        try:
            self.connection.close()
        except Exception:
            pass
    
    def is_valid(self) -> bool:
        """checkconnectionis否has效"""
        try:
            self.connection.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False
    
    def get_age(self) -> float:
        """retrievalconnection年龄（ seconds）"""
        return time.time() - self.created_at
    
    def get_idle_time(self) -> float:
        """retrieval空闲when间（ seconds）"""
        return time.time() - self.last_used

class DatabasePool:
    """数据库connection池"""
    
    def __init__(self, 
                 db_path: str,
                 min_connections: int = 2,
                 max_connections: int = 10,
                 max_idle_time: float = 300.0,  # 5分钟
                 max_connection_age: float = 3600.0,  # 1小when
                 connection_timeout: float = 30.0,
                 enable_monitoring: bool = True):
        
        self.db_path = self._prepare_db_path(db_path)
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        self.max_connection_age = max_connection_age
        self.connection_timeout = connection_timeout
        self.enable_monitoring = enable_monitoring
        
        # connection池
        self._available_connections: queue.Queue[PooledConnection] = queue.Queue()
        self._all_connections: Dict[int, PooledConnection] = {}
        
        # 同步锁
        self._pool_lock = threading.RLock()
        
        # 统计信息
        self.stats = ConnectionStats()
        
        # 监控线程
        self._monitor_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # 日志
        self.logger = logging.getLogger("DatabasePool")
        
        # 初始化connection池
        self._initialize_pool()
        
        # start监控
        if self.enable_monitoring:
            self._start_monitor()
    
    def _prepare_db_path(self, db_path: str) -> str:
        """准备数据库路径"""
        if not os.path.isabs(db_path):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            return os.path.join(data_dir, db_path)
        else:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            return db_path
    
    def _initialize_pool(self):
        """初始化connection池"""
        self.logger.info(f"初始化数据库connection池: {self.db_path}")
        
        # 创建最小connection数
        for _ in range(self.min_connections):
            try:
                connection = self._create_connection()
                self._available_connections.put(connection)
                self.logger.debug(f"创建初始connection: {connection.connection_id}")
            except Exception as e:
                self.logger.error(f"创建初始connectionfailed: {e}")
                self.stats.errors += 1
        
        self.logger.info(f"connection池初始化completed: {self._available_connections.qsize()} 个connection")
    
    def _create_connection(self) -> PooledConnection:
        """创建新connection"""
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.connection_timeout,
                isolation_level="DEFERRED",
                check_same_thread=False
            )
            
            pooled_conn = PooledConnection(conn, self)
            
            with self._pool_lock:
                self._all_connections[pooled_conn.connection_id] = pooled_conn
                self.stats.total_created += 1
                self.stats.current_idle += 1
                
            self.logger.debug(f"创建新connection: {pooled_conn.connection_id}")
            return pooled_conn
            
        except sqlite3.Error as e:
            self.logger.error(f"创建数据库connectionfailed: {e}")
            self.stats.errors += 1
            raise
    
    @contextmanager
    def get_connection(self) -> ContextManager[PooledConnection]:
        """retrievalconnection（上下文管理器）"""
        connection = None
        start_time = time.time()
        
        try:
            # retrievalconnection
            connection = self._acquire_connection()
            
            with self._pool_lock:
                self.stats.current_active += 1
                self.stats.current_idle -= 1
                self.stats.peak_active = max(self.stats.peak_active, self.stats.current_active)
            
            yield connection
            
        finally:
            # 归还connection
            if connection:
                self._release_connection(connection)
                
                with self._pool_lock:
                    self.stats.current_active -= 1
                    self.stats.current_idle += 1
                    self.stats.total_operations += 1
                    self.stats.total_time += time.time() - start_time
    
    def _acquire_connection(self) -> PooledConnection:
        """retrievalconnection"""
        try:
            # 尝试from池inretrievalconnection
            connection = self._available_connections.get(timeout=self.connection_timeout)
            
            # 验证connectionhas效性
            if not connection.is_valid():
                self.logger.warning(f"connectionno效，重新创建: {connection.connection_id}")
                self._close_connection(connection)
                connection = self._create_connection()
            
            return connection
            
        except queue.Empty:
            # 池innocanuseconnection
            with self._pool_lock:
                if len(self._all_connections) < self.max_connections:
                    # can以创建新connection
                    return self._create_connection()
                else:
                    # 达最大connection数，等待or抛出异常
                    self.logger.warning("connection池满，等待canuseconnection")
                    raise Exception(f"connection池满 ({self.max_connections} 个connection)")
    
    def _release_connection(self, connection: PooledConnection):
        """释放connection回池"""
        if connection and connection.is_valid():
            # checkconnectionis否应该be关闭
            if (connection.get_age() > self.max_connection_age or 
                connection.get_idle_time() > self.max_idle_time):
                self.logger.debug(f"connection过期，关闭: {connection.connection_id}")
                self._close_connection(connection)
            else:
                # 归还to池in
                self._available_connections.put(connection)
        else:
            self.logger.warning("尝试释放no效connection")
            if connection:
                self._close_connection(connection)
    
    def _close_connection(self, connection: PooledConnection):
        """关闭connection"""
        with self._pool_lock:
            if connection.connection_id in self._all_connections:
                del self._all_connections[connection.connection_id]
                self.stats.total_closed += 1
        
        connection.close()
        self.logger.debug(f"connection关闭: {connection.connection_id}")
    
    def _start_monitor(self):
        """start监控线程"""
        self._monitor_thread = threading.Thread(
            target=self._monitor_connections,
            name="DatabasePoolMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("数据库connection池监控start")
    
    def _monitor_connections(self):
        """监控connection状态"""
        while not self._shutdown_event.wait(60):  # 每分钟check一次
            try:
                self._cleanup_expired_connections()
                self._ensure_minimum_connections()
                self._log_pool_status()
            except Exception as e:
                self.logger.error(f"connection池监控异常: {e}")
    
    def _cleanup_expired_connections(self):
        """清理过期connection"""
        current_time = time.time()
        expired_connections = []
        
        with self._pool_lock:
            for conn_id, connection in list(self._all_connections.items()):
                if (not connection.is_active and 
                    (connection.get_age() > self.max_connection_age or 
                     connection.get_idle_time() > self.max_idle_time)):
                    expired_connections.append(connection)
        
        for connection in expired_connections:
            self.logger.debug(f"清理过期connection: {connection.connection_id}")
            self._close_connection(connection)
    
    def _ensure_minimum_connections(self):
        """确保最小connection数"""
        with self._pool_lock:
            current_connections = len(self._all_connections)
            if current_connections < self.min_connections:
                needed = self.min_connections - current_connections
                for _ in range(needed):
                    try:
                        connection = self._create_connection()
                        self._available_connections.put(connection)
                        self.logger.debug("补充最小connection数")
                    except Exception as e:
                        self.logger.error(f"补充connectionfailed: {e}")
                        break
    
    def _log_pool_status(self):
        """记录connection池状态"""
        stats = self.get_stats()
        self.logger.debug(f"connection池状态: 活跃={stats['active']}, 空闲={stats['idle']}, "
                         f"总操作={stats['total_operations']}, 命in率={stats['hit_rate']:.2%}")
    
    def execute_with_retry(self, query: str, params: tuple = (), max_retries: int = 3, fetch_result: bool = True):
        """执行SQL查询（带重试）"""
        for attempt in range(max_retries):
            try:
                with self.get_connection() as conn:
                    return conn.execute(query, params, fetch_result)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    self.logger.warning(f"数据库锁定，重试 {attempt + 1}/{max_retries}")
                    time.sleep(0.1 * (attempt + 1))
                    continue
                raise
            except Exception as e:
                self.logger.error(f"SQL执行failed: {e}")
                raise
    
    def execute_transaction(self, operations: List[Tuple[str, tuple]]):
        """执行事务"""
        with self.get_connection() as conn:
            try:
                conn.begin_transaction()
                results = []
                
                for query, params in operations:
                    result = conn.execute(query, params, fetch_result=False)
                    results.append(result)
                
                conn.commit()
                return results
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"事务执行failed，回滚: {e}")
                raise
    
    def get_stats(self) -> Dict[str, Any]:
        """retrievalconnection池统计"""
        with self._pool_lock:
            available = self._available_connections.qsize()
            total_connections = len(self._all_connections)
            
            return {
                'total_connections': total_connections,
                'active': self.stats.current_active,
                'idle': self.stats.current_idle,
                'available': available,
                'created': self.stats.total_created,
                'closed': self.stats.total_closed,
                'operations': self.stats.total_operations,
                'avg_operation_time': self.stats.total_time / max(self.stats.total_operations, 1),
                'peak_active': self.stats.peak_active,
                'errors': self.stats.errors,
                'hit_rate': available / max(total_connections, 1),
                'pool_config': {
                    'min_connections': self.min_connections,
                    'max_connections': self.max_connections,
                    'max_idle_time': self.max_idle_time,
                    'max_connection_age': self.max_connection_age
                }
            }
    
    def close_all(self):
        """关闭所hasconnection"""
        self.logger.info("关闭数据库connection池")
        
        # 停止监控
        if self._monitor_thread:
            self._shutdown_event.set()
            self._monitor_thread.join(timeout=5)
        
        # 关闭所hasconnection
        with self._pool_lock:
            connections_to_close = list(self._all_connections.values())
            
        for connection in connections_to_close:
            self._close_connection(connection)
        
        # 清空队列
        while not self._available_connections.empty():
            try:
                self._available_connections.get_nowait()
            except queue.Empty:
                break
        
        self.logger.info(f"connection池关闭，共关闭 {len(connections_to_close)} 个connection")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()


# 全局connection池实例
_global_db_pools: Dict[str, DatabasePool] = {}
_pool_lock = threading.Lock()

def get_database_pool(db_path: str = "autotrader_stocks.db", **kwargs) -> DatabasePool:
    """retrieval数据库connection池实例"""
    with _pool_lock:
        if db_path not in _global_db_pools:
            _global_db_pools[db_path] = DatabasePool(db_path, **kwargs)
        return _global_db_pools[db_path]

def close_all_pools():
    """关闭所hasconnection池"""
    with _pool_lock:
        for pool in _global_db_pools.values():
            pool.close_all()
        _global_db_pools.clear()


# 兼容性接口：增强数据库类
class EnhancedStockDatabase:
    """增强股票数据库（使useconnection池）"""
    
    def __init__(self, db_path: str = "autotrader_stocks.db", **pool_kwargs):
        self.db_path = db_path
        self.pool = get_database_pool(db_path, **pool_kwargs)
        self.logger = logging.getLogger("EnhancedStockDatabase")
        
        # 初始化数据库结构
        self._init_database()
    
    def _init_database(self):
        """Initialize database table structure"""
        try:
            # 使use原has初始化逻辑，但通过connection池执行
            # 这里简化实现，实际应该导入完整表结构
            init_queries = [
                """
                CREATE TABLE IF NOT EXISTS stock_lists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS stocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    sector TEXT,
                    exchange TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            ]
            
            for query in init_queries:
                self.pool.execute_with_retry(query, fetch_result=False)
            
            self.logger.info("数据库结构初始化completed")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def execute_with_retry(self, query: str, params: tuple = (), max_retries: int = 3, fetch_result: bool = True):
        """执行SQL查询（通过connection池）"""
        return self.pool.execute_with_retry(query, params, max_retries, fetch_result)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """retrievalconnection池统计"""
        return self.pool.get_stats()
    
    def close(self):
        """关闭connection（实际上is关闭整个池）"""
        # 注意：这会关闭整个池，影响其他使use者
        # in实际使usein，can能需要引use计数
        pass  # not实际关闭池，让池管理自己生命周期
