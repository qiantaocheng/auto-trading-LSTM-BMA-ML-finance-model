#!/usr/bin/env python3
"""
SQLite数据库模块 - 管理股票列表和交易配置
"""

import sqlite3
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

class StockDatabase:
    def __init__(self, db_path: str = "autotrader_stocks.db"):
        # Persist DB under project data directory to avoid CWD-dependent paths
        try:
            if not os.path.isabs(db_path):
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
                data_dir = os.path.join(base_dir, "data")
                os.makedirs(data_dir, exist_ok=True)
                self.db_path = os.path.join(data_dir, db_path)
            else:
                # Ensure parent exists
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                self.db_path = db_path
        except Exception:
            # Fallback to current directory if path prep fails
            self.db_path = db_path
        self.logger = logging.getLogger("StockDatabase")
        self._connection = None
        self._init_database()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure connection is closed"""
        self.close()
    
    def _get_connection(self):
        """Get database connection with proper timeout and concurrency settings"""
        try:
            conn = sqlite3.connect(
                self.db_path, 
                timeout=30.0,  # 30 second timeout
                isolation_level="DEFERRED",  # Allow concurrent reads
                check_same_thread=False  # Allow cross-thread usage
            )
            # Configure SQLite for better concurrency
            conn.execute("PRAGMA journal_mode=WAL;")  # Write-Ahead Logging for concurrent reads/writes
            conn.execute("PRAGMA synchronous=NORMAL;")  # Balance between safety and speed
            conn.execute("PRAGMA temp_store=MEMORY;")  # Use memory for temp tables
            conn.execute("PRAGMA cache_size=10000;")  # Larger cache for better performance
            conn.execute("PRAGMA busy_timeout=30000;")  # 30 second busy timeout
            return conn
        except sqlite3.Error as e:
            self.logger.error(f"Database connection failed: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if hasattr(self, '_connection') and self._connection:
            try:
                self._connection.close()
                self._connection = None
            except Exception as e:
                self.logger.warning(f"Error closing database connection: {e}")
    
    def _execute_with_retry(self, query: str, params: tuple = (), max_retries: int = 3):
        """Execute query with retry mechanism for handling database locks"""
        import time
        for attempt in range(max_retries):
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    return cursor.fetchall()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    self.logger.warning(f"Database locked, retrying in {0.1 * (attempt + 1)}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                raise
            except Exception as e:
                self.logger.error(f"Database operation failed: {e}")
                raise

    def _init_database(self):
        """初始化数据库表结构（使用重试机制）"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # 开始事务
                cursor.execute("BEGIN IMMEDIATE;")
                
                try:
                    # 股票列表表
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS stock_lists (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL UNIQUE,
                            description TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # 股票表
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

                    # 简化模式：全局tickers表（仅保存股票代码，满足"只存字符串代号"的需求）
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS tickers (
                            symbol TEXT PRIMARY KEY,
                            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    # 交易配置表
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

                    # 交易审计表
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

                    # 风险管理配置表
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS risk_configs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL UNIQUE,
                            config_json TEXT NOT NULL,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    conn.commit()
                    
                    # 创建默认股票列表
                    self._create_default_data()
                    
                    self.logger.info(f"数据库初始化完成: {self.db_path}")
                    
                except Exception as inner_e:
                    conn.rollback()
                    self.logger.error(f"数据库表创建失败: {inner_e}")
                    raise
                
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
            raise
    
    def _create_default_data(self):
        """创建默认数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 检查是否已有数据
                cursor.execute("SELECT COUNT(*) FROM stock_lists")
                if cursor.fetchone()[0] > 0:
                    return  # 已有数据，不创建默认数据
                
                # 创建默认股票列表（幂等）
                cursor.execute("""
                    INSERT INTO stock_lists (name, description)
                    VALUES ('科技股', '美股主要科技公司')
                    ON CONFLICT(name) DO NOTHING
                """)
                # 获取或创建后的列表ID
                if cursor.lastrowid:
                    list_id = cursor.lastrowid
                else:
                    cursor.execute("SELECT id FROM stock_lists WHERE name=?", ("科技股",))
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
                
                # 创建默认交易配置（幂等）
                cursor.execute("""
                    INSERT INTO trading_configs (name, alloc, poll_sec, auto_sell_removed, fixed_qty)
                    VALUES ('默认配置', 0.03, 10.0, 1, 0)
                    ON CONFLICT(name) DO NOTHING
                """)
                
                # 创建默认风险配置（幂等）
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
                self.logger.info("默认数据创建完成")
                
        except Exception as e:
            self.logger.warning(f"创建默认数据失败: {e}")

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
            self.logger.error(f"记录交易失败: {e}")
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
            self.logger.error(f"读取交易记录失败: {e}")
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
            self.logger.error(f"保存风险配置失败: {e}")
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
            self.logger.error(f"读取风险配置失败: {e}")
            return None

    # ===== 简化模式 API：全局 tickers 表 =====
    def get_all_tickers(self) -> List[str]:
        """获取全局 tickers 表中的所有股票代码（大写，按字母排序）。"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT symbol FROM tickers ORDER BY symbol")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"获取全局tickers失败: {e}")
            return []

    def get_all_tickers_with_meta(self) -> List[Dict]:
        """获取全局 tickers 及其元数据（symbol, added_at）。"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT symbol, added_at FROM tickers ORDER BY symbol")
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"获取全局tickers(含元数据)失败: {e}")
            return []

    def clear_all_tickers(self) -> List[str]:
        """清空全局 tickers 表，返回之前存在的股票代码。"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT symbol FROM tickers")
                prev = [row[0] for row in cursor.fetchall()]
                cursor.execute("DELETE FROM tickers")
                conn.commit()
                return prev
        except Exception as e:
            self.logger.error(f"清空全局tickers失败: {e}")
            return []

    def add_ticker(self, symbol: str) -> bool:
        """向全局 tickers 表添加一个股票代码。"""
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
            self.logger.error(f"添加ticker失败: {symbol} -> {e}")
            return False

    def remove_ticker(self, symbol: str) -> bool:
        """从全局 tickers 表移除一个股票代码。"""
        try:
            sym = (symbol or "").strip().upper()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tickers WHERE symbol = ?", (sym,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"移除ticker失败: {symbol} -> {e}")
            return False

    def replace_all_tickers(self, symbols: List[str]) -> Tuple[List[str], int, int]:
        """用给定symbols替换全局 tickers。返回：(被删除的旧代码, 成功添加数, 失败数)。"""
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
            self.logger.error(f"替换全局tickers失败: {e}")
            return [], 0, 0
    
    def get_stock_lists(self) -> List[Dict]:
        """获取所有股票列表"""
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
            self.logger.error(f"获取股票列表失败: {e}")
            return []
    
    def get_stocks_in_list(self, list_id: int) -> List[Dict]:
        """获取指定列表中的股票"""
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
            self.logger.error(f"获取股票失败: {e}")
            return []
    
    def get_stocks_as_csv(self, list_id: int) -> str:
        """获取股票列表为CSV格式"""
        stocks = self.get_stocks_in_list(list_id)
        return ",".join([stock["symbol"] for stock in stocks])
    
    def create_stock_list(self, name: str, description: str = "") -> int:
        """创建新的股票列表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO stock_lists (name, description) 
                    VALUES (?, ?)
                """, (name, description))
                conn.commit()
                return cursor.lastrowid
                
        except sqlite3.IntegrityError:
            raise ValueError(f"股票列表 '{name}' 已存在")
        except Exception as e:
            self.logger.error(f"创建股票列表失败: {e}")
            raise
    
    def add_stock(self, list_id: int, symbol: str, name: str = "") -> bool:
        """添加股票到列表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO stocks (list_id, symbol, name) 
                    VALUES (?, ?, ?)
                """, (list_id, symbol.upper(), name))
                
                # 更新列表的修改时间
                cursor.execute("""
                    UPDATE stock_lists 
                    SET updated_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (list_id,))
                
                conn.commit()
                return True
                
        except sqlite3.IntegrityError:
            self.logger.warning(f"股票 {symbol} 已在列表中")
            return False
        except Exception as e:
            self.logger.error(f"添加股票失败: {e}")
            return False
    
    def remove_stock(self, list_id: int, symbol: str) -> bool:
        """从列表中移除股票"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM stocks 
                    WHERE list_id = ? AND symbol = ?
                """, (list_id, symbol.upper()))
                
                if cursor.rowcount > 0:
                    # 更新列表的修改时间
                    cursor.execute("""
                        UPDATE stock_lists 
                        SET updated_at = CURRENT_TIMESTAMP 
                        WHERE id = ?
                    """, (list_id,))
                    conn.commit()
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"移除股票失败: {e}")
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
            self.logger.error(f"删除股票列表失败: {e}")
            return False
    
    def get_trading_configs(self) -> List[Dict]:
        """获取所有交易配置"""
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
            self.logger.error(f"获取交易配置失败: {e}")
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
            self.logger.error(f"保存交易配置失败: {e}")
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
            self.logger.error(f"加载交易配置失败: {e}")
            return None
    
    def import_from_csv(self, list_id: int, csv_symbols: str) -> Tuple[int, int]:
        """从CSV导入股票（返回：成功数量，失败数量）"""
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
        """清空股票列表，返回被删除的股票代码列表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 先获取要删除的股票代码
                cursor.execute("SELECT symbol FROM stocks WHERE list_id = ?", (list_id,))
                symbols = [row[0] for row in cursor.fetchall()]
                
                # 删除所有股票
                cursor.execute("DELETE FROM stocks WHERE list_id = ?", (list_id,))
                
                # 更新列表的修改时间
                cursor.execute("""
                    UPDATE stock_lists 
                    SET updated_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (list_id,))
                
                conn.commit()
                return symbols
                
        except Exception as e:
            self.logger.error(f"清空股票列表失败: {e}")
            return []
    
    def import_from_file_with_clear(self, list_id: int, csv_symbols: str) -> Tuple[List[str], int, int]:
        """从文件导入股票，先清空原有股票（返回：被删除的股票，成功数量，失败数量）"""
        try:
            # 先清空原有股票
            removed_symbols = self.clear_stock_list(list_id)
            
            # 导入新股票
            success_count, fail_count = self.import_from_csv(list_id, csv_symbols)
            
            return removed_symbols, success_count, fail_count
            
        except Exception as e:
            self.logger.error(f"文件导入失败: {e}")
            return [], 0, 0