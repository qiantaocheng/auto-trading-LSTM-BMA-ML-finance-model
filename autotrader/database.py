#!/usr/bin/env python3
"""
SQLite database module - Manages stock lists and trading configurations
"""

from __future__ import annotations

import sqlite3
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging
import threading

from .system_paths import DEFAULT_DB_FILENAME, resolve_data_path

class StockDatabase:
    def __init__(self, db_path: str | os.PathLike[str] = DEFAULT_DB_FILENAME):
        self._conn_lock = threading.RLock()
        self.db_path = self._resolve_db_path(db_path)
        self.logger = logging.getLogger("StockDatabase")
        self._connection = None
        self._connection_count = 0  # 跟踪连接数量
        self._init_database()

    def _resolve_db_path(self, db_path: str | os.PathLike[str]) -> str:
        """Return an absolute path for the SQLite database file."""
        path = Path(db_path)
        if not path.is_absolute():
            path = resolve_data_path(str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure connection is closed"""
        self.close()
        # Check for and report connection leaks
        if self.check_connection_leaks():
            self.logger.warning("Connection leaks detected during exit")
            self._force_cleanup()
    
    def _get_connection(self):
        """Get database connection with proper timeout and concurrency settings"""
        self._conn_lock.acquire()
        try:
            self._connection_count += 1
            self.logger.debug(f"Creating database connection #{self._connection_count}")
            
            conn = sqlite3.connect(
                self.db_path, 
                timeout=30.0,  # 30 second timeout
                isolation_level="DEFERRED",  # Allow concurrent reads
                check_same_thread=False  # Allow cross-thread usage
            )
            
            # Configure SQLite for better concurrency and performance
            conn.execute("PRAGMA journal_mode=WAL;")  # Write-Ahead Logging for concurrent reads/writes
            conn.execute("PRAGMA synchronous=NORMAL;")  # Balance between safety and speed
            conn.execute("PRAGMA temp_store=MEMORY;")  # Use memory for temp tables
            conn.execute("PRAGMA cache_size=10000;")  # Larger cache for better performance
            conn.execute("PRAGMA busy_timeout=30000;")  # 30 second busy timeout
            conn.execute("PRAGMA foreign_keys=ON;")  # Enable foreign key constraints
            
            # 使用装饰器模式跟踪连接关闭
            class TrackedConnection:
                def __init__(self, conn, tracker):
                    self._conn = conn
                    self._tracker = tracker
                    
                def __getattr__(self, name):
                    return getattr(self._conn, name)
                    
                def __enter__(self):
                    return self
                    
                def __exit__(self, exc_type, exc_val, exc_tb):
                    self.close()
                    return False
                    
                def close(self):
                    self._tracker._connection_count -= 1
                    self._tracker.logger.debug(f"Closing database connection, remaining: {self._tracker._connection_count}")
                    return self._conn.close()
                    
            conn = TrackedConnection(conn, self)
            
            return conn
        except sqlite3.Error as e:
            self._connection_count -= 1  # 回滚计数
            self.logger.error(f"Database connection failed: {e}")
            raise
        finally:
            self._conn_lock.release()
    
    def close(self):
        """Close database connection"""
        if hasattr(self, '_connection') and self._connection:
            try:
                self._connection.close()
                self._connection = None
            except Exception as e:
                self.logger.warning(f"Error closing database connection: {e}")
    
    def get_connection_info(self) -> Dict[str, Any]:
        """获取数据库连接信息"""
        return {
            'db_path': self.db_path,
            'active_connections': self._connection_count,
            'connection_exists': self._connection is not None
        }
    
    def check_connection_leaks(self) -> bool:
        """检查连接泄漏"""
        if self._connection_count > 0:
            self.logger.warning(f"Potential connection leak detected: {self._connection_count} connections still active")
            return True
        return False
    
    def _force_cleanup(self):
        """强制清理所有连接"""
        try:
            # Reset connection count
            self._connection_count = 0
            if hasattr(self, '_connection') and self._connection:
                self._connection.close()
                self._connection = None
            self.logger.info("Forced database connection cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during forced cleanup: {e}")
    
    def _validate_query_safety(self, query: str, params: tuple = ()) -> bool:
        """
        验证SQL查询安全性，防止SQL注入
        
        Args:
            query: SQL查询字符串
            params: 查询参数
            
        Returns:
            bool: 如果查询安全返回True，否则返回False
        """
        # 检查查询是否使用参数化查询
        if params is None:
            params = ()
        
        # 统计查询中的参数占位符数量
        placeholder_count = query.count('?')
        param_count = len(params)
        
        # 参数数量不匹配是潜在的安全问题
        if placeholder_count != param_count:
            self.logger.warning(f"参数数量不匹配: 查询需要{placeholder_count}个参数，提供了{param_count}个")
            return False
        
        # 检查是否有潜在的SQL注入模式（仅在没有使用参数化查询时）
        if placeholder_count == 0 and param_count == 0:
            dangerous_patterns = [
                '; DROP ',
                '; DELETE ',
                '; UPDATE ',
                '; INSERT ',
                '-- ',
                '/*',
                '*/',
                'UNION SELECT',
                'OR 1=1',
                'OR TRUE',
                "' OR '",
                '" OR "',
            ]
            
            query_upper = query.upper()
            for pattern in dangerous_patterns:
                if pattern in query_upper:
                    self.logger.error(f"检测到潜在SQL注入模式: {pattern}")
                    return False
        
        return True
    
    def _execute_with_retry(self, query: str, params: tuple = (), max_retries: int = 3, fetch_result: bool = True):
        """Execute query with retry mechanism for handling database locks
        
        Args:
            query: SQL query to execute
            params: Query parameters
            max_retries: Maximum retry attempts
            fetch_result: Whether to fetch and return results (SELECT) or just execute (INSERT/UPDATE/DELETE)
        """
        import time
        
        # 安全验证：检查SQL注入风险
        if not self._validate_query_safety(query, params):
            raise ValueError("检测到不安全的SQL查询，可能存在SQL注入风险")
        
        for attempt in range(max_retries):
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    
                    # Auto-commit for modification operations
                    if not fetch_result or query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER')):
                        conn.commit()
                        return cursor  # Return cursor for rowcount, lastrowid etc.
                    else:
                        return cursor.fetchall()  # Return results for SELECT
                        
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    self.logger.warning(f"Database locked, retrying in {0.1 * (attempt + 1)}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                raise
            except Exception as e:
                self.logger.error(f"Database operation failed: {e}")
                raise

    def create_tables(self):
        """Public method to create database tables"""
        return self._init_database()
    
    def init_database(self):
        """向后兼容的数据库初始化方法"""
        return self._init_database()
    
    def _init_database(self):
        """Initialize database table structure（using retry mechanism）"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Start transaction
                cursor.execute("BEGIN IMMEDIATE;")
                
                try:
                    # Stock list table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS stock_lists (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL UNIQUE,
                            description TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Stock table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS stocks (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            list_id INTEGER NOT NULL,
                            symbol TEXT NOT NULL,
                            name TEXT,
                            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (list_id) REFERENCES stock_lists (id) ON DELETE CASCADE,
                            UNIQUE(list_id, symbol)
                        )
                    """)

                    # Simplified mode：Global tickers table（only save stock codes，satisfy"only store string codes" requirements）
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS tickers (
                            symbol TEXT PRIMARY KEY,
                            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # Trading configuration table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS trading_configs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL UNIQUE,
                            alloc REAL NOT NULL DEFAULT 0.03,
                            poll_sec REAL NOT NULL DEFAULT 10.0,
                            auto_sell_removed BOOLEAN NOT NULL DEFAULT 1,
                            fixed_qty INTEGER NOT NULL DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Trading audit table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS trade_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT NOT NULL,
                            side TEXT NOT NULL,
                            quantity INTEGER NOT NULL,
                            avg_fill_price REAL,
                            order_id INTEGER,
                            status TEXT,
                            net_liq REAL,
                            cash_balance REAL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Risk management configuration table
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS risk_configs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL UNIQUE,
                            config_json TEXT NOT NULL,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    conn.commit()
                    
                    # Create default stock list
                    self._create_default_data()
                    
                    self.logger.info(f"Database initialization completed: {self.db_path}")
                    
                except Exception as inner_e:
                    conn.rollback()
                    self.logger.error(f"Database table creation failed: {inner_e}")
                    raise
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def _create_default_data(self):
        """Create default data"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if data already exists
                cursor.execute("SELECT COUNT(*) FROM stock_lists")
                if cursor.fetchone()[0] > 0:
                    return  # has数据，notCreate default data
                
                # Create default stock list（idempotent）
                cursor.execute("""
                    INSERT INTO stock_lists (name, description)
                    VALUES ('Tech Stocks', 'Major US tech companies')
                    ON CONFLICT(name) DO NOTHING
                """)
                # Get or create list ID
                if cursor.lastrowid:
                    list_id = cursor.lastrowid
                else:
                    cursor.execute("SELECT id FROM stock_lists WHERE name=?", ("Tech Stocks",))
                    row = cursor.fetchone()
                    list_id = row[0] if row else None
                
                # 添加默认股票
                default_stocks = [
                    
                ]
                
                if list_id is not None:
                    for symbol, name in default_stocks:
                        cursor.execute(
                            """
                            INSERT INTO stocks (list_id, symbol, name)
                            VALUES (?, ?, ?)
                            ON CONFLICT(list_id, symbol) DO NOTHING
                            """,
                            (list_id, symbol, name),
                        )
                
                # 创建默认交易配置（idempotent）
                cursor.execute("""
                    INSERT INTO trading_configs (name, alloc, poll_sec, auto_sell_removed, fixed_qty)
                    VALUES ('默认配置', 0.03, 10.0, 1, 0)
                    ON CONFLICT(name) DO NOTHING
                """)
                
                # 创建默认风险配置（idempotent）
                import json as _json
                default_risk_config = {
                    "default_stop_pct": 0.02,  # 2% 止损
                    "default_target_pct": 0.05,  # 5% 止盈
                    "max_single_position_pct": 0.1,  # 单笔最大10%
                    "max_daily_orders": 5,
                    "min_order_value_usd": 100,
                    "strategy_configs": {
                        "scalping": {"stop_pct": 0.005, "target_pct": 0.01},
                        "swing": {"stop_pct": 0.03, "target_pct": 0.08},
                        "position": {"stop_pct": 0.05, "target_pct": 0.15},
                    }
                }
                cursor.execute("""
                    INSERT INTO risk_configs (name, config_json)
                    VALUES ('默认风险配置', ?)
                    ON CONFLICT(name) DO NOTHING
                """, (_json.dumps(default_risk_config, ensure_ascii=False),))
                
                conn.commit()
                self.logger.info("默认数据创建completed")
                
        except Exception as e:
            self.logger.warning(f"Create default datafailed: {e}")

    # ===== 交易审计 API =====
    def record_trade(self, symbol: str, side: str, quantity: int, avg_fill_price: float,
                      order_id: int, status: str, net_liq: float, cash_balance: float) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO trade_history (symbol, side, quantity, avg_fill_price, order_id, status, net_liq, cash_balance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (symbol.upper(), side.upper(), int(quantity), float(avg_fill_price or 0.0), int(order_id or 0), status, float(net_liq or 0.0), float(cash_balance or 0.0))
                )
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"记录交易failed: {e}")
            return False

    def get_recent_trades(self, limit: int = 100) -> List[Dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, symbol, side, quantity, avg_fill_price, order_id, status, net_liq, cash_balance, created_at
                    FROM trade_history
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (int(limit),)
                )
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"读取交易记录failed: {e}")
            return []

    # ===== 风险配置 API =====
    def save_risk_config(self, config: Dict, name: str = "default") -> bool:
        try:
            import json as _json
            payload = _json.dumps(config, ensure_ascii=False)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO risk_configs (name, config_json, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(name) DO UPDATE SET config_json=excluded.config_json, updated_at=CURRENT_TIMESTAMP
                    """,
                    (name, payload)
                )
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"保存风险配置failed: {e}")
            return False

    def get_risk_config(self, name: str = "default") -> Optional[Dict]:
        try:
            import json as _json
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT config_json FROM risk_configs WHERE name=?", (name,))
                row = cursor.fetchone()
                if row and row[0]:
                    return _json.loads(row[0])
                return None
        except Exception as e:
            self.logger.error(f"读取风险配置failed: {e}")
            return None

    # ===== Simplified mode API：全局 tickers 表 =====
    def get_all_tickers(self) -> List[str]:
        """retrieval全局 tickers 表in所has股票代码（大写，按字母排序）。"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT symbol FROM tickers ORDER BY symbol")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"retrieval全局tickersfailed: {e}")
            return []
    
    def get_stock_universe(self) -> List[str]:
        """retrieval股票池（兼容方法，等同atget_all_tickers）"""
        return self.get_all_tickers()
    
    def clear_tickers(self) -> bool:
        """清空tickers表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tickers")
                conn.commit()
                self.logger.info("清空tickers表")
                return True
        except Exception as e:
            self.logger.error(f"清空tickers表failed: {e}")
            return False
    
    def batch_add_tickers(self, symbols: List[str]) -> bool:
        """批量添加股票代码totickers表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 批量插入
                for symbol in symbols:
                    symbol = symbol.upper().strip()
                    if symbol:
                        try:
                            cursor.execute("""
                                INSERT OR IGNORE INTO tickers (symbol, added_at) 
                                VALUES (?, ?)
                            """, (symbol, datetime.now().isoformat()))
                        except sqlite3.OperationalError:
                            # if果没hasadded_at列，只插入symbol
                            cursor.execute("""
                                INSERT OR IGNORE INTO tickers (symbol) 
                                VALUES (?)
                            """, (symbol,))
                
                conn.commit()
                self.logger.info(f"批量添加 {len(symbols)} 只股票totickers表")
                return True
        except Exception as e:
            self.logger.error(f"批量添加tickersfailed: {e}")
            return False

    def get_all_tickers_with_meta(self) -> List[Dict]:
        """retrieval全局 tickers 及其元数据（symbol, added_at）。"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT symbol, added_at FROM tickers ORDER BY symbol")
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"retrieval全局tickers(含元数据)failed: {e}")
            return []

    def clear_all_tickers(self) -> List[str]:
        """清空全局 tickers 表，返回之before存in股票代码。"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT symbol FROM tickers")
                prev = [row[0] for row in cursor.fetchall()]
                cursor.execute("DELETE FROM tickers")
                conn.commit()
                return prev
        except Exception as e:
            self.logger.error(f"清空全局tickersfailed: {e}")
            return []

    def add_ticker(self, symbol: str) -> bool:
        """to全局 tickers 表添加一个股票代码。"""
        try:
            sym = (symbol or "").strip().upper()
            if not sym:
                return False
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT OR IGNORE INTO tickers (symbol) VALUES (?)", (sym,))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"添加tickerfailed: {symbol} -> {e}")
            return False

    def remove_ticker(self, symbol: str) -> bool:
        """from全局 tickers 表移除一个股票代码。"""
        try:
            sym = (symbol or "").strip().upper()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tickers WHERE symbol = ?", (sym,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"移除tickerfailed: {symbol} -> {e}")
            return False

    def replace_all_tickers(self, symbols: List[str]) -> Tuple[List[str], int, int]:
        """use给定symbols替换全局 tickers。返回：(be删除旧代码, success添加数, failed数)。"""
        try:
            prev = self.clear_all_tickers()
            success, fail = 0, 0
            uniq = []
            seen = set()
            for s in symbols:
                s_up = (s or "").strip().upper()
                if not s_up or s_up in seen:
                    continue
                seen.add(s_up)
                if self.add_ticker(s_up):
                    success += 1
                else:
                    fail += 1
            return prev, success, fail
        except Exception as e:
            self.logger.error(f"替换全局tickersfailed: {e}")
            return [], 0, 0
    
    def get_stock_lists(self) -> List[Dict]:
        """retrieval所has股票列表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT sl.id, sl.name, sl.description, sl.created_at, sl.updated_at,
                           COUNT(s.id) as stock_count
                    FROM stock_lists sl
                    LEFT JOIN stocks s ON sl.id = s.list_id
                    GROUP BY sl.id, sl.name, sl.description, sl.created_at, sl.updated_at
                    ORDER BY sl.name
                """)
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"retrieval股票列表failed: {e}")
            return []
    
    def get_stocks_in_list(self, list_id: int) -> List[Dict]:
        """retrieval指定列表in股票"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, symbol, name, added_at
                    FROM stocks 
                    WHERE list_id = ?
                    ORDER BY symbol
                """, (list_id,))
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"retrieval股票failed: {e}")
            return []
    
    def get_stocks_as_csv(self, list_id: int) -> str:
        """retrieval股票列表asCSV格式"""
        stocks = self.get_stocks_in_list(list_id)
        return ",".join([stock["symbol"] for stock in stocks])
    
    def create_stock_list(self, name: str, symbols_or_description=None, description: str = "") -> int:
        """创建新股票列表
        
        Args:
            name: 列表名称
            symbols_or_description: 可以是符号列表(list)或描述(str)
            description: 描述(当symbols_or_description是list时使用)
        """
        # 处理参数兼容性
        if isinstance(symbols_or_description, str):
            # 旧的调用方式: create_stock_list(name, description)
            actual_description = symbols_or_description
            symbols = None
        elif isinstance(symbols_or_description, list):
            # 新的调用方式: create_stock_list(name, symbols, description)
            symbols = symbols_or_description
            actual_description = description
        else:
            # 只有name参数
            symbols = None
            actual_description = description
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO stock_lists (name, description) 
                    VALUES (?, ?)
                """, (name, actual_description))
                
                list_id = cursor.lastrowid
                
                # 如果提供了symbols，添加到列表中
                if symbols:
                    for symbol in symbols:
                        cursor.execute("""
                            INSERT INTO stocks (list_id, symbol, name) 
                            VALUES (?, ?, ?)
                        """, (list_id, symbol.upper(), ""))
                
                conn.commit()
                return list_id
                
        except sqlite3.IntegrityError:
            raise ValueError(f"股票列表 '{name}' 存in")
        except Exception as e:
            self.logger.error(f"创建股票列表failed: {e}")
            raise
    
    def add_stock(self, list_id: int, symbol: str, name: str = "") -> bool:
        """添加股票to列表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO stocks (list_id, symbol, name) 
                    VALUES (?, ?, ?)
                """, (list_id, symbol.upper(), name))
                
                # updates列表修改when间
                cursor.execute("""
                    UPDATE stock_lists 
                    SET updated_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (list_id,))
                
                conn.commit()
                return True
                
        except sqlite3.IntegrityError:
            self.logger.warning(f"股票 {symbol} in列表in")
            return False
        except Exception as e:
            self.logger.error(f"添加股票failed: {e}")
            return False
    
    def remove_stock(self, list_id: int, symbol: str) -> bool:
        """from列表in移除股票"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM stocks 
                    WHERE list_id = ? AND symbol = ?
                """, (list_id, symbol.upper()))
                
                if cursor.rowcount > 0:
                    # updates列表修改when间
                    cursor.execute("""
                        UPDATE stock_lists 
                        SET updated_at = CURRENT_TIMESTAMP 
                        WHERE id = ?
                    """, (list_id,))
                    conn.commit()
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"移除股票failed: {e}")
            return False
    
    def delete_stock_list(self, list_id: int) -> bool:
        """删除股票列表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM stock_lists WHERE id = ?", (list_id,))
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            self.logger.error(f"删除股票列表failed: {e}")
            return False
    
    def get_trading_configs(self) -> List[Dict]:
        """retrieval所has交易配置"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, alloc, poll_sec, auto_sell_removed, fixed_qty, created_at
                    FROM trading_configs
                    ORDER BY name
                """)
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"retrieval交易配置failed: {e}")
            return []
    
    def save_trading_config(self, name: str, alloc: float, poll_sec: float, 
                           auto_sell_removed: bool, fixed_qty: int) -> bool:
        """保存交易配置"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO trading_configs 
                    (name, alloc, poll_sec, auto_sell_removed, fixed_qty)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, alloc, poll_sec, auto_sell_removed, fixed_qty))
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"保存交易配置failed: {e}")
            return False
    
    def load_trading_config(self, name: str) -> Optional[Dict]:
        """加载交易配置"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT alloc, poll_sec, auto_sell_removed, fixed_qty
                    FROM trading_configs
                    WHERE name = ?
                """, (name,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        "alloc": row[0],
                        "poll_sec": row[1], 
                        "auto_sell_removed": bool(row[2]),
                        "fixed_qty": row[3]
                    }
                return None
                
        except Exception as e:
            self.logger.error(f"加载交易配置failed: {e}")
            return None
    
    def import_from_csv(self, list_id: int, csv_symbols: str) -> Tuple[int, int]:
        """fromCSV导入股票（返回：success数量，failed数量）"""
        success_count = 0
        fail_count = 0
        
        symbols = [s.strip().upper() for s in csv_symbols.split(",") if s.strip()]
        
        for symbol in symbols:
            if self.add_stock(list_id, symbol):
                success_count += 1
            else:
                fail_count += 1
        
        return success_count, fail_count
    
    def clear_stock_list(self, list_id: int) -> List[str]:
        """清空股票列表，返回be删除股票代码列表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 先retrieval要删除股票代码
                cursor.execute("SELECT symbol FROM stocks WHERE list_id = ?", (list_id,))
                symbols = [row[0] for row in cursor.fetchall()]
                
                # 删除所has股票
                cursor.execute("DELETE FROM stocks WHERE list_id = ?", (list_id,))
                
                # updates列表修改when间
                cursor.execute("""
                    UPDATE stock_lists 
                    SET updated_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (list_id,))
                
                conn.commit()
                return symbols
                
        except Exception as e:
            self.logger.error(f"清空股票列表failed: {e}")
            return []
    
    def import_from_file_with_clear(self, list_id: int, csv_symbols: str) -> Tuple[List[str], int, int]:
        """from文件导入股票，先清空原has股票（返回：be删除股票，success数量，failed数量）"""
        try:
            # 先清空原has股票
            removed_symbols = self.clear_stock_list(list_id)
            
            # 导入新股票
            success_count, fail_count = self.import_from_csv(list_id, csv_symbols)
            
            return removed_symbols, success_count, fail_count
            
        except Exception as e:
            self.logger.error(f"文件导入failed: {e}")
            return [], 0, 0