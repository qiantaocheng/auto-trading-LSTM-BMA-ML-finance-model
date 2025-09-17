#!/usr/bin/env python3
"""
📊 P1级别修复：幂等下单控制系统
=======================================

实现幂等下单控制，防止重复订单、确保订单唯一性，
支持订单去重、重试机制、状态一致性检查等功能。
"""

import time
import hashlib
import logging
import sqlite3
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid

logger = logging.getLogger(__name__)


class OrderDedupStrategy(Enum):
    """订单去重策略"""
    SYMBOL_SIDE_QUANTITY = "symbol_side_quantity"    # 基于标的+方向+数量
    SIGNAL_BASED = "signal_based"                    # 基于信号ID
    TIME_WINDOW = "time_window"                      # 基于时间窗口
    HASH_BASED = "hash_based"                        # 基于完整哈希


class OrderRetryStatus(Enum):
    """订单重试状态"""
    PENDING = "PENDING"
    RETRYING = "RETRYING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ABANDONED = "ABANDONED"


@dataclass
class OrderKey:
    """订单唯一键"""
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    price: Optional[float]
    order_type: str
    signal_id: Optional[str] = None
    strategy_id: Optional[str] = None
    timestamp_window: Optional[int] = None  # 时间窗口（秒）


@dataclass
class OrderDedupRecord:
    """订单去重记录"""
    dedup_key: str
    original_order_id: str
    symbol: str
    side: str
    quantity: float
    price: Optional[float]
    order_type: str
    signal_id: Optional[str]
    created_at: datetime
    expires_at: Optional[datetime]
    status: str
    retry_count: int = 0
    last_retry: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class IdempotentOrderController:
    """幂等订单控制器"""
    
    def __init__(self, 
                 db_path: str = "data/order_dedup.db",
                 dedup_strategy: OrderDedupStrategy = OrderDedupStrategy.SYMBOL_SIDE_QUANTITY,
                 default_ttl_seconds: int = 3600,  # 1小时过期
                 max_retry_attempts: int = 3,
                 retry_backoff_base: float = 2.0):
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.dedup_strategy = dedup_strategy
        self.default_ttl_seconds = default_ttl_seconds
        self.max_retry_attempts = max_retry_attempts
        self.retry_backoff_base = retry_backoff_base
        
        self._lock = threading.RLock()
        self._order_cache: Dict[str, OrderDedupRecord] = {}
        self._retry_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="OrderRetry")
        
        self._init_database()
        self._load_active_orders()
        
        logger.info(f"Idempotent order controller initialized - Strategy: {dedup_strategy.value}")
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 订单去重表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS order_dedup (
                    dedup_key TEXT PRIMARY KEY,
                    original_order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL,
                    order_type TEXT NOT NULL,
                    signal_id TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    status TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    last_retry TEXT,
                    metadata TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_dedup_symbol ON order_dedup(symbol);
                CREATE INDEX IF NOT EXISTS idx_dedup_signal_id ON order_dedup(signal_id);
                CREATE INDEX IF NOT EXISTS idx_dedup_created_at ON order_dedup(created_at);
                CREATE INDEX IF NOT EXISTS idx_dedup_status ON order_dedup(status)
                )
            """)
            
            # 重试记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS order_retry_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dedup_key TEXT NOT NULL,
                    retry_attempt INTEGER NOT NULL,
                    retry_time TEXT NOT NULL,
                    error_message TEXT,
                    success BOOLEAN NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY(dedup_key) REFERENCES order_dedup(dedup_key)
                )
            """)
            
            conn.commit()
    
    def _load_active_orders(self):
        """加载活跃的订单到缓存"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM order_dedup 
                WHERE status IN ('PENDING', 'RETRYING') 
                AND (expires_at IS NULL OR expires_at > ?)
            """, (datetime.now(timezone.utc).isoformat(),))
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            for row in rows:
                record_dict = dict(zip(columns, row))
                record = self._dict_to_record(record_dict)
                self._order_cache[record.dedup_key] = record
        
        logger.info(f"Loaded {len(self._order_cache)} active orders to cache")
    
    def _dict_to_record(self, record_dict: Dict[str, Any]) -> OrderDedupRecord:
        """将字典转换为订单记录"""
        return OrderDedupRecord(
            dedup_key=record_dict['dedup_key'],
            original_order_id=record_dict['original_order_id'],
            symbol=record_dict['symbol'],
            side=record_dict['side'],
            quantity=record_dict['quantity'],
            price=record_dict['price'],
            order_type=record_dict['order_type'],
            signal_id=record_dict['signal_id'],
            created_at=datetime.fromisoformat(record_dict['created_at']),
            expires_at=datetime.fromisoformat(record_dict['expires_at']) if record_dict['expires_at'] else None,
            status=record_dict['status'],
            retry_count=record_dict['retry_count'] or 0,
            last_retry=datetime.fromisoformat(record_dict['last_retry']) if record_dict['last_retry'] else None,
            metadata=json.loads(record_dict['metadata']) if record_dict['metadata'] else {}
        )
    
    def generate_dedup_key(self, order_key: OrderKey) -> str:
        """生成订单去重键"""
        if self.dedup_strategy == OrderDedupStrategy.SYMBOL_SIDE_QUANTITY:
            # 基于标的+方向+数量
            key_parts = [
                order_key.symbol,
                order_key.side,
                f"{order_key.quantity:.6f}",
                order_key.order_type
            ]
            
        elif self.dedup_strategy == OrderDedupStrategy.SIGNAL_BASED:
            # 基于信号ID
            if not order_key.signal_id:
                raise ValueError("Signal ID required for signal-based deduplication")
            key_parts = [order_key.signal_id, order_key.symbol]
            
        elif self.dedup_strategy == OrderDedupStrategy.TIME_WINDOW:
            # 基于时间窗口（5分钟窗口）
            window_size = order_key.timestamp_window or 300  # 5分钟
            current_window = int(time.time() // window_size)
            key_parts = [
                order_key.symbol,
                order_key.side,
                f"{order_key.quantity:.6f}",
                str(current_window)
            ]
            
        elif self.dedup_strategy == OrderDedupStrategy.HASH_BASED:
            # 基于完整哈希
            hash_input = json.dumps({
                'symbol': order_key.symbol,
                'side': order_key.side,
                'quantity': order_key.quantity,
                'price': order_key.price,
                'order_type': order_key.order_type,
                'signal_id': order_key.signal_id,
                'strategy_id': order_key.strategy_id
            }, sort_keys=True)
            return hashlib.sha256(hash_input.encode()).hexdigest()[:32]
        
        else:
            raise ValueError(f"Unsupported dedup strategy: {self.dedup_strategy}")
        
        # 生成哈希键
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def check_order_duplicate(self, order_key: OrderKey) -> Tuple[bool, Optional[str]]:
        """检查订单是否重复（增强竞态条件防护）"""
        dedup_key = self.generate_dedup_key(order_key)

        with self._lock:
            # 使用数据库级别的原子操作确保一致性
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # 启用WAL模式以提高并发性能
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA busy_timeout=5000")  # 5秒超时

                    cursor = conn.cursor()

                    # 使用SELECT FOR UPDATE模拟的原子检查和更新
                    cursor.execute("BEGIN IMMEDIATE")  # 立即获取写锁

                    try:
                        # 原子性检查存在性
                        cursor.execute("""
                            SELECT original_order_id, status, expires_at, created_at
                            FROM order_dedup
                            WHERE dedup_key = ?
                        """, (dedup_key,))

                        result = cursor.fetchone()
                        current_time = datetime.now(timezone.utc)

                        if result:
                            original_order_id, status, expires_at, created_at = result

                            # 检查过期（在事务内原子性清理）
                            if expires_at:
                                expires_dt = datetime.fromisoformat(expires_at)
                                if expires_dt < current_time:
                                    # 原子性删除过期记录
                                    cursor.execute("DELETE FROM order_dedup WHERE dedup_key = ?", (dedup_key,))
                                    conn.commit()

                                    # 从缓存中移除
                                    if dedup_key in self._order_cache:
                                        del self._order_cache[dedup_key]

                                    return False, dedup_key

                            # 检查活跃状态
                            if status in ['PENDING', 'RETRYING', 'SUCCESS']:
                                conn.commit()

                                # 更新缓存（双重保险）
                                if dedup_key not in self._order_cache:
                                    record = OrderDedupRecord(
                                        dedup_key=dedup_key,
                                        original_order_id=original_order_id,
                                        symbol=order_key.symbol,
                                        side=order_key.side,
                                        quantity=order_key.quantity,
                                        price=order_key.price,
                                        order_type=order_key.order_type,
                                        signal_id=order_key.signal_id,
                                        created_at=datetime.fromisoformat(created_at),
                                        expires_at=datetime.fromisoformat(expires_at) if expires_at else None,
                                        status=status
                                    )
                                    self._order_cache[dedup_key] = record

                                return True, original_order_id

                        # 不存在重复记录
                        conn.commit()
                        return False, dedup_key

                    except Exception as e:
                        conn.rollback()
                        logger.error(f"数据库事务失败: {e}")
                        raise

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower():
                    logger.warning(f"数据库锁定，重试检查: {dedup_key}")
                    time.sleep(0.01)  # 短暂等待后重试
                    return self.check_order_duplicate(order_key)
                else:
                    logger.error(f"数据库操作错误: {e}")
                    # 降级到缓存检查
                    return self._fallback_cache_check(dedup_key)
            except Exception as e:
                logger.error(f"订单重复检查异常: {e}")
                return self._fallback_cache_check(dedup_key)

    def _fallback_cache_check(self, dedup_key: str) -> Tuple[bool, Optional[str]]:
        """降级到缓存检查（数据库不可用时）"""
        if dedup_key in self._order_cache:
            cached_record = self._order_cache[dedup_key]

            # 检查是否过期
            if cached_record.expires_at and cached_record.expires_at < datetime.now(timezone.utc):
                del self._order_cache[dedup_key]
                return False, dedup_key

            if cached_record.status in ['PENDING', 'RETRYING', 'SUCCESS']:
                return True, cached_record.original_order_id

        return False, dedup_key
    
    def register_order(self, order_key: OrderKey, order_id: str,
                      ttl_seconds: Optional[int] = None) -> str:
        """注册订单（防重复）- 增强原子性保护"""
        dedup_key = self.generate_dedup_key(order_key)

        with self._lock:
            # 使用原子性的检查和插入操作
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # 启用WAL模式和设置超时
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA busy_timeout=5000")

                    cursor = conn.cursor()
                    cursor.execute("BEGIN IMMEDIATE")  # 立即获取写锁

                    try:
                        # 再次检查重复（双重检查模式）
                        cursor.execute("""
                            SELECT original_order_id, status, expires_at
                            FROM order_dedup
                            WHERE dedup_key = ?
                        """, (dedup_key,))

                        result = cursor.fetchone()
                        current_time = datetime.now(timezone.utc)

                        if result:
                            original_order_id, status, expires_at = result

                            # 检查过期状态
                            if expires_at:
                                expires_dt = datetime.fromisoformat(expires_at)
                                if expires_dt < current_time:
                                    # 删除过期记录，继续注册新订单
                                    cursor.execute("DELETE FROM order_dedup WHERE dedup_key = ?", (dedup_key,))
                                else:
                                    # 未过期且活跃，返回重复
                                    if status in ['PENDING', 'RETRYING', 'SUCCESS']:
                                        conn.commit()
                                        logger.warning(f"Duplicate order detected (atomic): {order_key.symbol} {order_key.side} {order_key.quantity}")
                                        return original_order_id

                        # 原子性插入新记录
                        ttl_seconds = ttl_seconds or self.default_ttl_seconds
                        expires_at = current_time + timedelta(seconds=ttl_seconds)

                        cursor.execute("""
                            INSERT INTO order_dedup
                            (dedup_key, original_order_id, symbol, side, quantity, price,
                             order_type, signal_id, created_at, expires_at, status,
                             retry_count, last_retry, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            dedup_key, order_id, order_key.symbol, order_key.side,
                            order_key.quantity, order_key.price, order_key.order_type,
                            order_key.signal_id, current_time.isoformat(),
                            expires_at.isoformat(), 'PENDING', 0, None, "{}"
                        ))

                        conn.commit()

                        # 更新缓存
                        record = OrderDedupRecord(
                            dedup_key=dedup_key,
                            original_order_id=order_id,
                            symbol=order_key.symbol,
                            side=order_key.side,
                            quantity=order_key.quantity,
                            price=order_key.price,
                            order_type=order_key.order_type,
                            signal_id=order_key.signal_id,
                            created_at=current_time,
                            expires_at=expires_at,
                            status='PENDING'
                        )
                        self._order_cache[dedup_key] = record

                        logger.info(f"Order registered atomically: {order_id} -> {dedup_key}")
                        return order_id

                    except Exception as e:
                        conn.rollback()
                        logger.error(f"订单注册事务失败: {e}")
                        raise

            except sqlite3.IntegrityError as e:
                # 可能是并发插入导致的唯一键冲突
                logger.warning(f"订单注册冲突，可能已存在: {dedup_key}")
                # 重新检查是否存在
                is_duplicate, existing_id = self.check_order_duplicate(order_key)
                return existing_id if is_duplicate else order_id

            except Exception as e:
                logger.error(f"订单注册异常: {e}")
                # 降级处理：仅使用缓存
                if dedup_key not in self._order_cache:
                    record = OrderDedupRecord(
                        dedup_key=dedup_key,
                        original_order_id=order_id,
                        symbol=order_key.symbol,
                        side=order_key.side,
                        quantity=order_key.quantity,
                        price=order_key.price,
                        order_type=order_key.order_type,
                        signal_id=order_key.signal_id,
                        created_at=datetime.now(timezone.utc),
                        expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds or self.default_ttl_seconds),
                        status='PENDING'
                    )
                    self._order_cache[dedup_key] = record
                    logger.warning(f"Order registered in cache only: {order_id}")

                return order_id
    
    def update_order_status(self, order_id: str, status: str, 
                           error_message: Optional[str] = None):
        """更新订单状态"""
        with self._lock:
            # 查找订单记录
            dedup_key = None
            for key, record in self._order_cache.items():
                if record.original_order_id == order_id:
                    dedup_key = key
                    break
            
            if not dedup_key:
                # 从数据库查找
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT dedup_key FROM order_dedup WHERE original_order_id = ?
                    """, (order_id,))
                    result = cursor.fetchone()
                    if result:
                        dedup_key = result[0]
            
            if dedup_key:
                # 更新数据库
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE order_dedup 
                        SET status = ?, last_retry = ?
                        WHERE dedup_key = ?
                    """, (status, datetime.now(timezone.utc).isoformat(), dedup_key))
                    conn.commit()
                
                # 更新缓存
                if dedup_key in self._order_cache:
                    self._order_cache[dedup_key].status = status
                    self._order_cache[dedup_key].last_retry = datetime.now(timezone.utc)
                
                logger.info(f"Order status updated: {order_id} -> {status}")
                
                # 如果是失败状态，考虑重试
                if status in ['REJECTED', 'FAILED'] and error_message:
                    self._schedule_retry(dedup_key, error_message)
    
    def _schedule_retry(self, dedup_key: str, error_message: str):
        """安排订单重试"""
        if dedup_key not in self._order_cache:
            return
        
        record = self._order_cache[dedup_key]
        if record.retry_count >= self.max_retry_attempts:
            logger.warning(f"Order {record.original_order_id} exceeded max retry attempts")
            self.update_order_status(record.original_order_id, 'ABANDONED')
            return
        
        # 计算退避延迟
        delay_seconds = self.retry_backoff_base ** record.retry_count
        
        logger.info(f"Scheduling retry for order {record.original_order_id} in {delay_seconds:.1f}s")
        
        # 提交重试任务
        self._retry_executor.submit(self._retry_order, dedup_key, error_message, delay_seconds)
    
    def _retry_order(self, dedup_key: str, error_message: str, delay_seconds: float):
        """重试订单"""
        time.sleep(delay_seconds)
        
        with self._lock:
            if dedup_key not in self._order_cache:
                return
            
            record = self._order_cache[dedup_key]
            record.retry_count += 1
            record.last_retry = datetime.now(timezone.utc)
            record.status = 'RETRYING'
            
            # 记录重试日志
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO order_retry_log 
                    (dedup_key, retry_attempt, retry_time, error_message, success, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    dedup_key, record.retry_count, record.last_retry.isoformat(),
                    error_message, False, json.dumps({'delay_seconds': delay_seconds})
                ))
                
                # 更新主记录
                cursor.execute("""
                    UPDATE order_dedup 
                    SET retry_count = ?, last_retry = ?, status = ?
                    WHERE dedup_key = ?
                """, (record.retry_count, record.last_retry.isoformat(), record.status, dedup_key))
                
                conn.commit()
            
            logger.info(f"Order retry #{record.retry_count}: {record.original_order_id}")
    
    def cleanup_expired_orders(self):
        """清理过期订单"""
        current_time = datetime.now(timezone.utc)
        expired_keys = []
        
        with self._lock:
            # 检查缓存中的过期订单
            for key, record in list(self._order_cache.items()):
                if record.expires_at and record.expires_at < current_time:
                    expired_keys.append(key)
                    del self._order_cache[key]
            
            # 清理数据库中的过期订单
            if expired_keys:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    placeholders = ','.join('?' * len(expired_keys))
                    cursor.execute(f"""
                        DELETE FROM order_dedup 
                        WHERE dedup_key IN ({placeholders}) AND expires_at < ?
                    """, expired_keys + [current_time.isoformat()])
                    conn.commit()
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired orders")
    
    def get_dedup_statistics(self) -> Dict[str, Any]:
        """获取去重统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 基础统计
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_orders,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    AVG(retry_count) as avg_retries,
                    COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END) as successful_orders,
                    COUNT(CASE WHEN status = 'FAILED' THEN 1 END) as failed_orders,
                    COUNT(CASE WHEN status = 'ABANDONED' THEN 1 END) as abandoned_orders
                FROM order_dedup
            """)
            
            basic_stats = dict(zip([col[0] for col in cursor.description], cursor.fetchone()))
            
            # 按状态分组统计
            cursor.execute("""
                SELECT status, COUNT(*) as count 
                FROM order_dedup 
                GROUP BY status
            """)
            
            status_stats = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'basic_stats': basic_stats,
                'status_distribution': status_stats,
                'active_cache_size': len(self._order_cache),
                'dedup_strategy': self.dedup_strategy.value,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }


# 全局实例
_global_idempotent_controller: Optional[IdempotentOrderController] = None


def get_idempotent_controller() -> IdempotentOrderController:
    """获取全局幂等订单控制器"""
    global _global_idempotent_controller
    if _global_idempotent_controller is None:
        _global_idempotent_controller = IdempotentOrderController()
    return _global_idempotent_controller


if __name__ == "__main__":
    # 测试幂等订单控制
    logging.basicConfig(level=logging.INFO)
    
    controller = IdempotentOrderController()
    
    # 创建测试订单键
    order_key = OrderKey(
        symbol="AAPL",
        side="BUY",
        quantity=100,
        price=150.0,
        order_type="LMT",
        signal_id="SIG_TEST_001"
    )
    
    # 测试重复检查
    is_dup1, key1 = controller.check_order_duplicate(order_key)
    print(f"First check - Duplicate: {is_dup1}, Key: {key1}")
    
    # 注册订单
    order_id = "ORD_TEST_001"
    registered_id = controller.register_order(order_key, order_id)
    print(f"Registered order: {registered_id}")
    
    # 再次检查（应该是重复）
    is_dup2, key2 = controller.check_order_duplicate(order_key)
    print(f"Second check - Duplicate: {is_dup2}, Key: {key2}")
    
    # 获取统计信息
    stats = controller.get_dedup_statistics()
    print(f"Statistics: {json.dumps(stats, indent=2)}")