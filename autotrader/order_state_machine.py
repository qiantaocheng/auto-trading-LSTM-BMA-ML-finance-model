#!/usr/bin/env python3
"""
订单状态机模块 - 专业级订单状态管理
"""

import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from threading import Lock
import asyncio


class OrderState(Enum):
    """订单状态枚举"""
    PENDING = "PENDING"           # 待提交
    SUBMITTED = "SUBMITTED"       # 已提交
    ACKNOWLEDGED = "ACKNOWLEDGED" # 已确认
    PARTIAL = "PARTIAL"          # 部分成交
    FILLED = "FILLED"            # 完全成交
    CANCELLED = "CANCELLED"      # 已取消
    REJECTED = "REJECTED"        # 已拒绝
    EXPIRED = "EXPIRED"          # 已过期
    FAILED = "FAILED"            # 执行失败


class OrderType(Enum):
    """订单类型"""
    MARKET = "MKT"
    LIMIT = "LMT" 
    STOP = "STP"
    STOP_LIMIT = "STP LMT"
    BRACKET = "BRACKET"


@dataclass
class OrderTransition:
    """订单状态转换记录"""
    from_state: str
    to_state: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


@dataclass
class OrderSnapshot:
    """订单快照"""
    order_id: int
    symbol: str
    side: str
    quantity: int
    order_type: str
    price: Optional[float]
    filled_quantity: int
    avg_fill_price: Optional[float]
    remaining_quantity: int
    state: str
    submit_time: float
    last_update: float
    parent_id: Optional[int] = None
    strategy: Optional[str] = None
    
    
class OrderStateMachine:
    """订单状态机 - 管理单个订单的完整生命周期"""
    
    # 定义有效的状态转换
    VALID_TRANSITIONS = {
        OrderState.PENDING: [OrderState.SUBMITTED, OrderState.FAILED, OrderState.REJECTED],
        OrderState.SUBMITTED: [OrderState.ACKNOWLEDGED, OrderState.REJECTED, OrderState.CANCELLED, OrderState.FAILED],
        OrderState.ACKNOWLEDGED: [OrderState.PARTIAL, OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED],
        OrderState.PARTIAL: [OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED],
        OrderState.FILLED: [],  # 终态
        OrderState.CANCELLED: [], # 终态
        OrderState.REJECTED: [],  # 终态
        OrderState.EXPIRED: [],   # 终态
        OrderState.FAILED: []     # 终态
    }
    
    TERMINAL_STATES = {OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED, OrderState.FAILED}
    
    def __init__(self, order_id: int, symbol: str, side: str, quantity: int, 
                 order_type: OrderType, price: Optional[float] = None, 
                 strategy: Optional[str] = None, parent_id: Optional[int] = None,
                 state_change_callback: Optional[Callable] = None):
        self.order_id = order_id
        self.symbol = symbol.upper()
        self.side = side.upper()
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.strategy = strategy
        self.parent_id = parent_id
        
        # 状态管理
        self.state = OrderState.PENDING
        self.state_history: List[OrderTransition] = []
        self.timestamps: Dict[str, float] = {}
        
        # 成交信息
        self.filled_quantity = 0
        self.avg_fill_price: Optional[float] = None
        self.remaining_quantity = quantity
        
        # 同步锁
        self._lock = Lock()
        
        # 回调函数
        self.state_change_callback = state_change_callback
        
        # 日志
        self.logger = logging.getLogger(f"OrderSM.{order_id}")
        
        # 初始化
        self.submit_time = time.time()
        self.last_update = self.submit_time
        self.timestamps[OrderState.PENDING.value] = self.submit_time
        
        self.logger.info(f"订单状态机创建: {self._basic_info()}")
        
        # 触发创建回调
        if self.state_change_callback:
            try:
                self.state_change_callback(self, None, OrderState.PENDING, metadata={'action': 'created'})
            except Exception as e:
                self.logger.warning(f"状态回调失败: {e}")
    
    def _basic_info(self) -> str:
        """基本订单信息"""
        return f"{self.symbol} {self.side} {self.quantity}@{self.price or 'MKT'}"
    
    def transition(self, new_state: OrderState, metadata: Optional[Dict[str, Any]] = None, 
                  reason: Optional[str] = None) -> bool:
        """状态转换"""
        with self._lock:
            if not self._is_valid_transition(self.state, new_state):
                self.logger.warning(f"无效状态转换: {self.state.value} -> {new_state.value}")
                return False
            
            # 记录转换
            transition = OrderTransition(
                from_state=self.state.value,
                to_state=new_state.value,
                timestamp=time.time(),
                metadata=metadata,
                reason=reason
            )
            
            self.state_history.append(transition)
            
            # 更新状态
            old_state = self.state
            self.state = new_state
            self.last_update = transition.timestamp
            self.timestamps[new_state.value] = transition.timestamp
            
            self.logger.info(f"状态转换: {old_state.value} -> {new_state.value} ({reason or 'N/A'})")
            
            # 触发状态变化回调
            if self.state_change_callback:
                try:
                    self.state_change_callback(self, old_state, new_state, metadata=metadata, reason=reason)
                except Exception as e:
                    self.logger.warning(f"状态变化回调失败: {e}")
            
            # 处理特定状态的逻辑
            self._handle_state_entry(new_state, metadata)
            
            return True
    
    def _is_valid_transition(self, from_state: OrderState, to_state: OrderState) -> bool:
        """检查状态转换是否有效"""
        return to_state in self.VALID_TRANSITIONS.get(from_state, [])
    
    def _handle_state_entry(self, state: OrderState, metadata: Optional[Dict[str, Any]]):
        """处理进入新状态时的逻辑"""
        if state == OrderState.PARTIAL and metadata:
            self._update_fill_info(metadata)
        elif state == OrderState.FILLED and metadata:
            self._update_fill_info(metadata)
        elif state in self.TERMINAL_STATES:
            self.logger.info(f"订单进入终态: {state.value} - {self._basic_info()}")
    
    def _update_fill_info(self, fill_data: Dict[str, Any]):
        """更新成交信息"""
        if 'filled_quantity' in fill_data:
            self.filled_quantity = fill_data['filled_quantity']
            self.remaining_quantity = self.quantity - self.filled_quantity
        
        if 'avg_fill_price' in fill_data:
            self.avg_fill_price = fill_data['avg_fill_price']
        
        self.logger.info(f"成交更新: {self.filled_quantity}/{self.quantity} @ {self.avg_fill_price}")
    
    def update_fill(self, filled_qty: int, avg_price: float) -> bool:
        """更新成交信息并转换状态"""
        with self._lock:
            fill_data = {
                'filled_quantity': filled_qty,
                'avg_fill_price': avg_price
            }
            
            if filled_qty >= self.quantity:
                # 完全成交
                return self.transition(OrderState.FILLED, fill_data, "完全成交")
            elif filled_qty > self.filled_quantity:
                # 部分成交
                return self.transition(OrderState.PARTIAL, fill_data, "部分成交")
            
            return False
    
    def cancel(self, reason: str = "用户取消") -> bool:
        """取消订单"""
        return self.transition(OrderState.CANCELLED, reason=reason)
    
    def reject(self, reason: str = "系统拒绝") -> bool:
        """拒绝订单"""
        return self.transition(OrderState.REJECTED, reason=reason)
    
    def fail(self, reason: str = "执行失败") -> bool:
        """订单失败"""
        return self.transition(OrderState.FAILED, reason=reason)
    
    def is_terminal(self) -> bool:
        """是否处于终态"""
        return self.state in self.TERMINAL_STATES
    
    def is_active(self) -> bool:
        """是否为活跃订单"""
        return not self.is_terminal()
    
    def get_snapshot(self) -> OrderSnapshot:
        """获取订单快照"""
        with self._lock:
            return OrderSnapshot(
                order_id=self.order_id,
                symbol=self.symbol,
                side=self.side,
                quantity=self.quantity,
                order_type=self.order_type.value,
                price=self.price,
                filled_quantity=self.filled_quantity,
                avg_fill_price=self.avg_fill_price,
                remaining_quantity=self.remaining_quantity,
                state=self.state.value,
                submit_time=self.submit_time,
                last_update=self.last_update,
                parent_id=self.parent_id,
                strategy=self.strategy
            )
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取状态历史"""
        with self._lock:
            return [asdict(transition) for transition in self.state_history]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        with self._lock:
            now = time.time()
            duration = now - self.submit_time
            
            metrics = {
                'total_duration': duration,
                'current_state': self.state.value,
                'fill_rate': self.filled_quantity / self.quantity if self.quantity > 0 else 0,
                'is_terminal': self.is_terminal(),
                'state_changes': len(self.state_history),
            }
            
            # 计算各状态持续时间
            state_durations = {}
            for i, transition in enumerate(self.state_history):
                if i == 0:
                    state_durations[OrderState.PENDING.value] = transition.timestamp - self.submit_time
                else:
                    prev_transition = self.state_history[i-1]
                    state_durations[prev_transition.to_state] = transition.timestamp - prev_transition.timestamp
            
            # 当前状态持续时间
            if self.state_history:
                last_transition = self.state_history[-1]
                state_durations[last_transition.to_state] = now - last_transition.timestamp
            else:
                state_durations[OrderState.PENDING.value] = duration
            
            metrics['state_durations'] = state_durations
            
            return metrics


class OrderManager:
    """订单管理器 - 管理所有订单的状态机"""
    
    def __init__(self, auditor=None):
        self.orders: Dict[int, OrderStateMachine] = {}
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger("OrderManager")
        self.auditor = auditor  # TradingAuditor实例
        
        # 统计信息
        self.stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'failed_orders': 0,
        }
    
    def _audit_state_change(self, order_sm: 'OrderStateMachine', old_state: Optional[OrderState], 
                           new_state: OrderState, metadata=None, reason=None):
        """审计回调 - 记录订单状态变化"""
        if not self.auditor:
            return
        
        try:
            audit_data = {
                'order_id': order_sm.order_id,
                'symbol': order_sm.symbol,
                'side': order_sm.side,
                'quantity': order_sm.quantity,
                'order_type': order_sm.order_type.value,
                'price': order_sm.price,
                'old_state': old_state.value if old_state else None,
                'new_state': new_state.value,
                'timestamp': time.time(),
                'filled_quantity': order_sm.filled_quantity,
                'avg_fill_price': order_sm.avg_fill_price,
                'strategy': order_sm.strategy,
                'parent_id': order_sm.parent_id,
                'reason': reason,
                'metadata': metadata
            }
            
            self.auditor.log_order(audit_data)
            
        except Exception as e:
            self.logger.warning(f"审计回调失败: {e}")
    
    async def create_order(self, order_id: int, symbol: str, side: str, quantity: int,
                          order_type: OrderType, price: Optional[float] = None,
                          strategy: Optional[str] = None, parent_id: Optional[int] = None) -> OrderStateMachine:
        """创建新订单状态机"""
        async with self._lock:
            if order_id in self.orders:
                raise ValueError(f"订单ID {order_id} 已存在")
            
            order_sm = OrderStateMachine(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                strategy=strategy,
                parent_id=parent_id,
                state_change_callback=self._audit_state_change  # 设置审计回调
            )
            
            self.orders[order_id] = order_sm
            self.stats['total_orders'] += 1
            
            self.logger.info(f"创建订单状态机: {order_id} - {symbol} {side} {quantity}")
            return order_sm
    
    async def get_order(self, order_id: int) -> Optional[OrderStateMachine]:
        """获取订单状态机"""
        async with self._lock:
            return self.orders.get(order_id)
    
    async def update_order_state(self, order_id: int, new_state: OrderState, 
                               metadata: Optional[Dict[str, Any]] = None, 
                               reason: Optional[str] = None) -> bool:
        """更新订单状态"""
        order_sm = await self.get_order(order_id)
        if not order_sm:
            self.logger.warning(f"未找到订单: {order_id}")
            return False
        
        success = order_sm.transition(new_state, metadata, reason)
        
        # 更新统计
        if success and order_sm.is_terminal():
            await self._update_stats(order_sm)
        
        return success
    
    async def _update_stats(self, order_sm: OrderStateMachine):
        """更新统计信息"""
        state = order_sm.state
        if state == OrderState.FILLED:
            self.stats['filled_orders'] += 1
        elif state == OrderState.CANCELLED:
            self.stats['cancelled_orders'] += 1
        elif state == OrderState.REJECTED:
            self.stats['rejected_orders'] += 1
        elif state == OrderState.FAILED:
            self.stats['failed_orders'] += 1
    
    async def get_active_orders(self) -> List[OrderStateMachine]:
        """获取所有活跃订单"""
        async with self._lock:
            return [order for order in self.orders.values() if order.is_active()]
    
    async def get_terminal_orders(self) -> List[OrderStateMachine]:
        """获取所有终态订单"""
        async with self._lock:
            return [order for order in self.orders.values() if order.is_terminal()]
    
    async def get_orders_by_symbol(self, symbol: str) -> List[OrderStateMachine]:
        """获取指定标的的所有订单"""
        async with self._lock:
            return [order for order in self.orders.values() if order.symbol == symbol.upper()]
    
    async def get_orders_by_strategy(self, strategy: str) -> List[OrderStateMachine]:
        """获取指定策略的所有订单"""
        async with self._lock:
            return [order for order in self.orders.values() if order.strategy == strategy]
    
    async def cleanup_terminal_orders(self, max_age_hours: int = 24):
        """清理老旧的终态订单"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        async with self._lock:
            to_remove = []
            for order_id, order_sm in self.orders.items():
                if order_sm.is_terminal() and order_sm.last_update < cutoff_time:
                    to_remove.append(order_id)
            
            for order_id in to_remove:
                del self.orders[order_id]
                self.logger.info(f"清理终态订单: {order_id}")
            
            return len(to_remove)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        async with self._lock:
            active_orders = len([o for o in self.orders.values() if o.is_active()])
            terminal_orders = len([o for o in self.orders.values() if o.is_terminal()])
            
            stats = self.stats.copy()
            stats.update({
                'active_orders': active_orders,
                'terminal_orders': terminal_orders,
                'total_managed': len(self.orders),
                'fill_rate': self.stats['filled_orders'] / max(self.stats['total_orders'], 1),
                'success_rate': (self.stats['filled_orders'] + self.stats['cancelled_orders']) / max(self.stats['total_orders'], 1)
            })
            
            return stats
    
    async def export_order_history(self, start_time: Optional[float] = None, 
                                 end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """导出订单历史（用于审计）"""
        async with self._lock:
            history = []
            
            for order_sm in self.orders.values():
                if start_time and order_sm.submit_time < start_time:
                    continue
                if end_time and order_sm.submit_time > end_time:
                    continue
                
                order_data = {
                    'snapshot': asdict(order_sm.get_snapshot()),
                    'history': order_sm.get_history(),
                    'metrics': order_sm.get_performance_metrics()
                }
                history.append(order_data)
            
            return history