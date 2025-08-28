#!/usr/bin/env python3
"""
🔥 P0级别修复：订单链路追踪系统
=======================================

实现完整的signal_id → order_id → fill_id → execution_id链路追踪，
确保每个交易决策都有完整的审计轨迹，满足量化交易合规要求。
"""

import uuid
import time
import logging
import sqlite3
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import threading

# 统一使用unified_trading_core中的OrderStatus定义
from .unified_trading_core import OrderStatus

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class SignalRecord:
    """信号记录"""
    signal_id: str
    symbol: str
    signal_type: SignalType
    confidence: float
    signal_value: float
    features_hash: str
    model_version: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderRecord:
    """订单记录"""
    order_id: str
    signal_id: str  # 关联的信号ID
    symbol: str
    side: str  # BUY/SELL
    order_type: str  # MKT/LMT/STP等
    quantity: float
    price: Optional[float]
    status: OrderStatus
    broker_order_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class FillRecord:
    """成交记录"""
    fill_id: str
    order_id: str  # 关联的订单ID
    execution_id: str  # 券商执行ID
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    fill_time: datetime
    broker_exec_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionRecord:
    """执行记录（汇总）"""
    execution_id: str
    signal_id: str
    order_ids: List[str]
    fill_ids: List[str] 
    symbol: str
    total_quantity: float
    average_price: float
    total_commission: float
    execution_start: datetime
    execution_end: datetime
    pnl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderChainTracker:
    """订单链路追踪器"""
    
    def __init__(self, db_path: str = "data/order_chain.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.RLock()
        self._init_database()
        
        # 内存缓存（用于快速查询）
        self._signal_cache: Dict[str, SignalRecord] = {}
        self._order_cache: Dict[str, OrderRecord] = {}
        self._chain_cache: Dict[str, List[str]] = {}  # signal_id -> order_ids
        
        logger.info(f"Order chain tracker initialized: {self.db_path}")
    
    def _init_database(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 信号表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    signal_value REAL NOT NULL,
                    features_hash TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at)")
            
            # 订单表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    signal_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    status TEXT NOT NULL,
                    broker_order_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY(signal_id) REFERENCES signals(signal_id)
                );
                CREATE INDEX IF NOT EXISTS idx_orders_signal_id ON orders(signal_id);
                CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
                CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)
                )
            """)
            
            # 成交表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fills (
                    fill_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    execution_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    commission REAL NOT NULL,
                    fill_time TEXT NOT NULL,
                    broker_exec_id TEXT,
                    metadata TEXT,
                    FOREIGN KEY(order_id) REFERENCES orders(order_id)
                );
                CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id);
                CREATE INDEX IF NOT EXISTS idx_fills_execution_id ON fills(execution_id);
                CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol)
                )
            """)
            
            # 执行汇总表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY,
                    signal_id TEXT NOT NULL,
                    order_ids TEXT NOT NULL,  -- JSON array
                    fill_ids TEXT NOT NULL,   -- JSON array
                    symbol TEXT NOT NULL,
                    total_quantity REAL NOT NULL,
                    average_price REAL NOT NULL,
                    total_commission REAL NOT NULL,
                    execution_start TEXT NOT NULL,
                    execution_end TEXT NOT NULL,
                    pnl REAL,
                    metadata TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_executions_signal_id ON executions(signal_id);
                CREATE INDEX IF NOT EXISTS idx_executions_symbol ON executions(symbol)
                )
            """)
            
            conn.commit()
    
    def generate_signal_id(self, symbol: str) -> str:
        """生成信号ID"""
        timestamp = int(time.time() * 1000)  # 毫秒时间戳
        unique_id = str(uuid.uuid4())[:8]
        return f"SIG_{symbol}_{timestamp}_{unique_id}"
    
    def generate_order_id(self, signal_id: str) -> str:
        """生成订单ID"""
        timestamp = int(time.time() * 1000)
        unique_id = str(uuid.uuid4())[:8]
        return f"ORD_{signal_id.split('_')[1]}_{timestamp}_{unique_id}"
    
    def generate_fill_id(self, order_id: str) -> str:
        """生成成交ID"""
        timestamp = int(time.time() * 1000)
        unique_id = str(uuid.uuid4())[:8]
        return f"FILL_{order_id.split('_')[1]}_{timestamp}_{unique_id}"
    
    def generate_execution_id(self, signal_id: str) -> str:
        """生成执行ID"""
        timestamp = int(time.time() * 1000)
        unique_id = str(uuid.uuid4())[:8]
        return f"EXEC_{signal_id.split('_')[1]}_{timestamp}_{unique_id}"
    
    def record_signal(self, signal: SignalRecord) -> str:
        """记录交易信号"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO signals 
                        (signal_id, symbol, signal_type, confidence, signal_value, 
                         features_hash, model_version, created_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        signal.signal_id, signal.symbol, signal.signal_type.value,
                        signal.confidence, signal.signal_value, signal.features_hash,
                        signal.model_version, signal.created_at.isoformat(),
                        json.dumps(signal.metadata)
                    ))
                    conn.commit()
                
                # 更新缓存
                self._signal_cache[signal.signal_id] = signal
                
                logger.info(f"Signal recorded: {signal.signal_id} - {signal.symbol} {signal.signal_type.value}")
                return signal.signal_id
                
            except Exception as e:
                logger.error(f"Failed to record signal: {e}")
                raise
    
    def record_order(self, order: OrderRecord) -> str:
        """记录订单"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO orders 
                        (order_id, signal_id, symbol, side, order_type, quantity, 
                         price, status, broker_order_id, created_at, updated_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        order.order_id, order.signal_id, order.symbol, order.side,
                        order.order_type, order.quantity, order.price, order.status.value,
                        order.broker_order_id, order.created_at.isoformat(),
                        order.updated_at.isoformat(), json.dumps(order.metadata)
                    ))
                    conn.commit()
                
                # 更新缓存
                self._order_cache[order.order_id] = order
                
                # 更新链路缓存
                if order.signal_id not in self._chain_cache:
                    self._chain_cache[order.signal_id] = []
                self._chain_cache[order.signal_id].append(order.order_id)
                
                logger.info(f"Order recorded: {order.order_id} - {order.symbol} {order.side} {order.quantity}")
                return order.order_id
                
            except Exception as e:
                logger.error(f"Failed to record order: {e}")
                raise
    
    def record_fill(self, fill: FillRecord) -> str:
        """记录成交"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO fills 
                        (fill_id, order_id, execution_id, symbol, side, quantity, 
                         price, commission, fill_time, broker_exec_id, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        fill.fill_id, fill.order_id, fill.execution_id, fill.symbol,
                        fill.side, fill.quantity, fill.price, fill.commission,
                        fill.fill_time.isoformat(), fill.broker_exec_id,
                        json.dumps(fill.metadata)
                    ))
                    conn.commit()
                
                logger.info(f"Fill recorded: {fill.fill_id} - {fill.symbol} {fill.quantity}@{fill.price}")
                return fill.fill_id
                
            except Exception as e:
                logger.error(f"Failed to record fill: {e}")
                raise
    
    def update_order_status(self, order_id: str, status: OrderStatus, 
                           broker_order_id: Optional[str] = None):
        """更新订单状态"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE orders 
                        SET status = ?, broker_order_id = ?, updated_at = ?
                        WHERE order_id = ?
                    """, (status.value, broker_order_id, datetime.now(timezone.utc).isoformat(), order_id))
                    conn.commit()
                
                # 更新缓存
                if order_id in self._order_cache:
                    self._order_cache[order_id].status = status
                    self._order_cache[order_id].updated_at = datetime.now(timezone.utc)
                    if broker_order_id:
                        self._order_cache[order_id].broker_order_id = broker_order_id
                
                logger.info(f"Order status updated: {order_id} -> {status.value}")
                
            except Exception as e:
                logger.error(f"Failed to update order status: {e}")
                raise
    
    def get_order_chain(self, signal_id: str) -> Dict[str, Any]:
        """获取完整的订单链路"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 获取信号
                    cursor.execute("SELECT * FROM signals WHERE signal_id = ?", (signal_id,))
                    signal_row = cursor.fetchone()
                    if not signal_row:
                        return {}
                    
                    # 获取关联订单
                    cursor.execute("SELECT * FROM orders WHERE signal_id = ?", (signal_id,))
                    order_rows = cursor.fetchall()
                    
                    order_ids = [row[0] for row in order_rows]  # order_id是第一列
                    
                    # 获取关联成交
                    fills = []
                    if order_ids:
                        placeholders = ','.join('?' * len(order_ids))
                        cursor.execute(f"SELECT * FROM fills WHERE order_id IN ({placeholders})", order_ids)
                        fill_rows = cursor.fetchall()
                        fills = [dict(zip([col[0] for col in cursor.description], row)) for row in fill_rows]
                    
                    # 构建完整链路
                    chain = {
                        'signal': dict(zip([col[0] for col in cursor.description], signal_row)) if signal_row else None,
                        'orders': [dict(zip([col[0] for col in cursor.description], row)) for row in order_rows],
                        'fills': fills,
                        'chain_summary': {
                            'signal_id': signal_id,
                            'total_orders': len(order_rows),
                            'total_fills': len(fills),
                            'total_quantity': sum(fill.get('quantity', 0) for fill in fills),
                            'total_commission': sum(fill.get('commission', 0) for fill in fills)
                        }
                    }
                    
                    return chain
                    
            except Exception as e:
                logger.error(f"Failed to get order chain: {e}")
                return {}
    
    def get_chain_analytics(self, symbol: str = None, 
                           start_date: datetime = None,
                           end_date: datetime = None) -> Dict[str, Any]:
        """获取链路分析数据"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 构建查询条件
                    where_conditions = []
                    params = []
                    
                    if symbol:
                        where_conditions.append("s.symbol = ?")
                        params.append(symbol)
                    
                    if start_date:
                        where_conditions.append("s.created_at >= ?")
                        params.append(start_date.isoformat())
                    
                    if end_date:
                        where_conditions.append("s.created_at <= ?")
                        params.append(end_date.isoformat())
                    
                    where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
                    
                    # 链路统计查询
                    query = f"""
                        SELECT 
                            COUNT(DISTINCT s.signal_id) as total_signals,
                            COUNT(DISTINCT o.order_id) as total_orders,
                            COUNT(DISTINCT f.fill_id) as total_fills,
                            AVG(CASE WHEN o.status = 'FILLED' THEN 1.0 ELSE 0.0 END) as fill_rate,
                            AVG(f.commission) as avg_commission,
                            SUM(f.quantity * f.price) as total_notional,
                            COUNT(DISTINCT s.symbol) as symbols_traded
                        FROM signals s
                        LEFT JOIN orders o ON s.signal_id = o.signal_id
                        LEFT JOIN fills f ON o.order_id = f.order_id
                        {where_clause}
                    """
                    
                    cursor.execute(query, params)
                    result = cursor.fetchone()
                    
                    analytics = dict(zip([col[0] for col in cursor.description], result)) if result else {}
                    
                    return analytics
                    
            except Exception as e:
                logger.error(f"Failed to get chain analytics: {e}")
                return {}


# 全局实例
_global_chain_tracker: Optional[OrderChainTracker] = None


def get_order_chain_tracker() -> OrderChainTracker:
    """获取全局订单链路追踪器"""
    global _global_chain_tracker
    if _global_chain_tracker is None:
        _global_chain_tracker = OrderChainTracker()
    return _global_chain_tracker


if __name__ == "__main__":
    # 测试订单链路追踪
    logging.basicConfig(level=logging.INFO)
    
    tracker = OrderChainTracker("test_chain.db")
    
    # 创建测试信号
    signal = SignalRecord(
        signal_id=tracker.generate_signal_id("AAPL"),
        symbol="AAPL", 
        signal_type=SignalType.BUY,
        confidence=0.85,
        signal_value=0.12,
        features_hash="abc123",
        model_version="v1.0.0",
        created_at=datetime.now(timezone.utc)
    )
    
    tracker.record_signal(signal)
    
    # 创建测试订单
    order = OrderRecord(
        order_id=tracker.generate_order_id(signal.signal_id),
        signal_id=signal.signal_id,
        symbol="AAPL",
        side="BUY", 
        order_type="MKT",
        quantity=100,
        price=None,
        status=OrderStatus.SUBMITTED
    )
    
    tracker.record_order(order)
    
    # 获取链路
    chain = tracker.get_order_chain(signal.signal_id)
    print(f"Order chain: {json.dumps(chain, indent=2, default=str)}")