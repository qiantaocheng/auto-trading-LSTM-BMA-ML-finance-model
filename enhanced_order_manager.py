#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的订单管理系统
实现trade.filled监控、订单状态跟踪和Bracket Orders
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
import uuid
import json
import os

try:
    from ib_insync import *
    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"
    ERROR = "error"


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    BRACKET = "bracket"


@dataclass
class OrderRecord:
    """订单记录"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    action: str = ""  # BUY/SELL
    quantity: int = 0
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Bracket order parameters
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    ib_order_id: Optional[int] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    
    # Execution details
    filled_quantity: int = 0
    remaining_quantity: int = 0
    avg_fill_price: Optional[float] = None
    commission: Optional[float] = None
    
    # Error tracking
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Metadata
    strategy_name: Optional[str] = None
    reason: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'action': self.action,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'take_profit_price': self.take_profit_price,
            'stop_loss_price': self.stop_loss_price,
            'status': self.status.value,
            'ib_order_id': self.ib_order_id,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'cancelled_at': self.cancelled_at.isoformat() if self.cancelled_at else None,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'avg_fill_price': self.avg_fill_price,
            'commission': self.commission,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'strategy_name': self.strategy_name,
            'reason': self.reason,
            'created_at': self.created_at.isoformat()
        }


class EnhancedOrderManager:
    """增强的订单管理器"""
    
    def __init__(self, ib_connection, config: Dict[str, Any] = None, logger=None):
        self.ib = ib_connection
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # 订单存储
        self.orders = {}  # order_id -> OrderRecord
        self.ib_order_mapping = {}  # ib_order_id -> order_id
        self.symbol_orders = {}  # symbol -> list of order_ids
        
        # 状态跟踪
        self.active_orders = set()  # 活跃订单ID
        self.filled_orders = set()  # 已成交订单ID
        
        # 配置
        self.auto_retry = config.get('auto_retry', True)
        self.retry_delay = config.get('retry_delay_seconds', 5)
        self.order_timeout = config.get('order_timeout_seconds', 300)  # 5分钟
        self.save_orders = config.get('save_orders', True)
        self.orders_file = config.get('orders_file', 'orders/order_history.json')
        
        # 回调函数
        self.order_callbacks = {
            OrderStatus.SUBMITTED: [],
            OrderStatus.FILLED: [],
            OrderStatus.CANCELLED: [],
            OrderStatus.REJECTED: [],
            OrderStatus.PARTIALLY_FILLED: [],
            OrderStatus.ERROR: []
        }
        
        # 监控线程
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # 设置事件处理
        self._setup_event_handlers()
        
        # 加载历史订单
        if self.save_orders:
            self._load_order_history()
        
        # 启动监控
        self._start_monitoring()
    
    def _setup_event_handlers(self):
        """设置IB事件处理器"""
        if not self.ib:
            return
        
        # 订单状态事件
        self.ib.orderStatusEvent += self._on_order_status
        
        # 执行事件
        self.ib.execDetailsEvent += self._on_execution
        
        # 错误事件
        self.ib.errorEvent += self._on_error
    
    def submit_market_order(self, symbol: str, action: str, quantity: int, 
                           strategy_name: str = None, reason: str = None) -> str:
        """提交市价单"""
        order_record = OrderRecord(
            symbol=symbol,
            action=action.upper(),
            quantity=quantity,
            order_type=OrderType.MARKET,
            strategy_name=strategy_name,
            reason=reason
        )
        
        return self._submit_order(order_record)
    
    def submit_limit_order(self, symbol: str, action: str, quantity: int, limit_price: float,
                          strategy_name: str = None, reason: str = None) -> str:
        """提交限价单"""
        order_record = OrderRecord(
            symbol=symbol,
            action=action.upper(),
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            strategy_name=strategy_name,
            reason=reason
        )
        
        return self._submit_order(order_record)
    
    def submit_bracket_order(self, symbol: str, action: str, quantity: int, 
                           limit_price: Optional[float] = None,
                           take_profit_price: Optional[float] = None,
                           stop_loss_price: Optional[float] = None,
                           strategy_name: str = None, reason: str = None) -> str:
        """提交括号单（包含止盈止损）"""
        order_record = OrderRecord(
            symbol=symbol,
            action=action.upper(),
            quantity=quantity,
            order_type=OrderType.BRACKET,
            limit_price=limit_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            strategy_name=strategy_name,
            reason=reason
        )
        
        return self._submit_order(order_record)
    
    def _submit_order(self, order_record: OrderRecord) -> str:
        """提交订单的内部方法"""
        try:
            # 创建合约
            contract = self._create_contract(order_record.symbol)
            if not contract:
                order_record.status = OrderStatus.ERROR
                order_record.error_message = f"Failed to create contract for {order_record.symbol}"
                self._update_order(order_record)
                return order_record.order_id
            
            # 创建订单
            if order_record.order_type == OrderType.BRACKET:
                ib_order = self._create_bracket_order(order_record)
            else:
                ib_order = self._create_simple_order(order_record)
            
            if not ib_order:
                order_record.status = OrderStatus.ERROR
                order_record.error_message = "Failed to create IB order"
                self._update_order(order_record)
                return order_record.order_id
            
            # 提交订单
            trade = self.ib.placeOrder(contract, ib_order)
            
            if trade:
                order_record.ib_order_id = trade.order.orderId
                order_record.status = OrderStatus.SUBMITTED
                order_record.submitted_at = datetime.now()
                order_record.remaining_quantity = order_record.quantity
                
                # 建立映射
                self.ib_order_mapping[trade.order.orderId] = order_record.order_id
                self.active_orders.add(order_record.order_id)
                
                # 按股票分组
                if order_record.symbol not in self.symbol_orders:
                    self.symbol_orders[order_record.symbol] = []
                self.symbol_orders[order_record.symbol].append(order_record.order_id)
                
                self.logger.info(f"Order submitted: {order_record.order_id} - {order_record.symbol} {order_record.action} {order_record.quantity}")
                
            else:
                order_record.status = OrderStatus.ERROR
                order_record.error_message = "Failed to place order with IB"
            
            self._update_order(order_record)
            return order_record.order_id
            
        except Exception as e:
            order_record.status = OrderStatus.ERROR
            order_record.error_message = str(e)
            self._update_order(order_record)
            self.logger.error(f"Error submitting order: {e}")
            return order_record.order_id
    
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
    
    def _create_simple_order(self, order_record: OrderRecord) -> Optional[Order]:
        """创建简单订单"""
        try:
            if order_record.order_type == OrderType.MARKET:
                return MarketOrder(order_record.action, order_record.quantity)
            elif order_record.order_type == OrderType.LIMIT:
                return LimitOrder(order_record.action, order_record.quantity, order_record.limit_price)
            elif order_record.order_type == OrderType.STOP:
                return StopOrder(order_record.action, order_record.quantity, order_record.stop_price)
            elif order_record.order_type == OrderType.STOP_LIMIT:
                return StopLimitOrder(order_record.action, order_record.quantity, 
                                    order_record.limit_price, order_record.stop_price)
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error creating simple order: {e}")
            return None
    
    def _create_bracket_order(self, order_record: OrderRecord) -> Optional[Order]:
        """创建括号单"""
        try:
            # 主订单
            if order_record.limit_price:
                parent_order = LimitOrder(order_record.action, order_record.quantity, order_record.limit_price)
            else:
                parent_order = MarketOrder(order_record.action, order_record.quantity)
            
            # 设置括号单参数
            parent_order.transmit = False  # 不立即传输，等待子订单
            
            # 获取唯一的订单ID
            parent_id = self.ib.client.getReqId()
            parent_order.orderId = parent_id
            
            # 子订单列表
            child_orders = []
            
            # 止盈单
            if order_record.take_profit_price:
                profit_action = 'SELL' if order_record.action == 'BUY' else 'BUY'
                profit_order = LimitOrder(profit_action, order_record.quantity, order_record.take_profit_price)
                profit_order.parentId = parent_id
                profit_order.orderId = self.ib.client.getReqId()
                profit_order.transmit = False
                child_orders.append(profit_order)
            
            # 止损单
            if order_record.stop_loss_price:
                stop_action = 'SELL' if order_record.action == 'BUY' else 'BUY'
                stop_order = StopOrder(stop_action, order_record.quantity, order_record.stop_loss_price)
                stop_order.parentId = parent_id
                stop_order.orderId = self.ib.client.getReqId()
                stop_order.transmit = len(child_orders) == 0  # 最后一个订单设置transmit=True
                child_orders.append(stop_order)
            
            # 设置最后一个子订单的transmit=True
            if child_orders:
                child_orders[-1].transmit = True
            
            # 返回主订单（子订单会自动跟随）
            return parent_order
            
        except Exception as e:
            self.logger.error(f"Error creating bracket order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id not in self.orders:
            self.logger.warning(f"Order {order_id} not found")
            return False
        
        order_record = self.orders[order_id]
        
        if order_record.status not in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
            self.logger.warning(f"Order {order_id} cannot be cancelled (status: {order_record.status})")
            return False
        
        try:
            if order_record.ib_order_id:
                # 找到对应的trade对象
                trades = [t for t in self.ib.trades() if t.order.orderId == order_record.ib_order_id]
                if trades:
                    self.ib.cancelOrder(trades[0].order)
                    self.logger.info(f"Cancel request sent for order {order_id}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def _on_order_status(self, trade):
        """处理订单状态更新"""
        try:
            ib_order_id = trade.order.orderId
            
            if ib_order_id not in self.ib_order_mapping:
                return  # 不是我们管理的订单
            
            order_id = self.ib_order_mapping[ib_order_id]
            order_record = self.orders[order_id]
            
            # 更新订单状态
            old_status = order_record.status
            
            if trade.orderStatus.status == 'Filled':
                order_record.status = OrderStatus.FILLED
                order_record.filled_at = datetime.now()
                order_record.filled_quantity = trade.orderStatus.filled
                order_record.remaining_quantity = trade.orderStatus.remaining
                
                # 移到已成交集合
                self.active_orders.discard(order_id)
                self.filled_orders.add(order_id)
                
            elif trade.orderStatus.status == 'Cancelled':
                order_record.status = OrderStatus.CANCELLED
                order_record.cancelled_at = datetime.now()
                order_record.remaining_quantity = trade.orderStatus.remaining
                
                self.active_orders.discard(order_id)
                
            elif trade.orderStatus.status == 'PartiallyFilled':
                order_record.status = OrderStatus.PARTIALLY_FILLED
                order_record.filled_quantity = trade.orderStatus.filled
                order_record.remaining_quantity = trade.orderStatus.remaining
                
            elif trade.orderStatus.status in ['Rejected', 'ApiCancelled']:
                order_record.status = OrderStatus.REJECTED
                order_record.error_message = f"Order rejected: {trade.orderStatus.status}"
                
                self.active_orders.discard(order_id)
            
            # 更新平均成交价
            if hasattr(trade.orderStatus, 'avgFillPrice') and trade.orderStatus.avgFillPrice:
                order_record.avg_fill_price = trade.orderStatus.avgFillPrice
            
            # 记录状态变化
            if old_status != order_record.status:
                self.logger.info(f"Order {order_id} status changed: {old_status.value} -> {order_record.status.value}")
                self._trigger_callbacks(order_record.status, order_record)
            
            self._update_order(order_record)
            
        except Exception as e:
            self.logger.error(f"Error handling order status: {e}")
    
    def _on_execution(self, trade, fill):
        """处理执行回报"""
        try:
            ib_order_id = trade.order.orderId
            
            if ib_order_id not in self.ib_order_mapping:
                return
            
            order_id = self.ib_order_mapping[ib_order_id]
            order_record = self.orders[order_id]
            
            # 更新成交信息
            if fill.execution.cumQty > order_record.filled_quantity:
                order_record.filled_quantity = fill.execution.cumQty
                order_record.avg_fill_price = fill.execution.avgPrice
                
                # 更新佣金
                if hasattr(fill, 'commissionReport') and fill.commissionReport:
                    order_record.commission = fill.commissionReport.commission
                
                self.logger.info(f"Order {order_id} execution: {fill.execution.shares} @ {fill.execution.price}")
                
            self._update_order(order_record)
            
        except Exception as e:
            self.logger.error(f"Error handling execution: {e}")
    
    def _on_error(self, reqId, errorCode, errorString, contract):
        """处理错误事件"""
        try:
            # 查找相关订单
            relevant_orders = []
            for order_id, order_record in self.orders.items():
                if order_record.ib_order_id == reqId:
                    relevant_orders.append(order_record)
            
            if relevant_orders:
                for order_record in relevant_orders:
                    order_record.error_message = f"Error {errorCode}: {errorString}"
                    
                    # 某些错误代码表示订单被拒绝
                    if errorCode in [201, 202, 203, 400, 401, 402]:
                        order_record.status = OrderStatus.REJECTED
                        self.active_orders.discard(order_record.order_id)
                        self._trigger_callbacks(OrderStatus.REJECTED, order_record)
                    
                    self._update_order(order_record)
                    
                    self.logger.error(f"Order {order_record.order_id} error: {errorCode} - {errorString}")
            
        except Exception as e:
            self.logger.error(f"Error handling order error event: {e}")
    
    def _update_order(self, order_record: OrderRecord):
        """更新订单记录"""
        self.orders[order_record.order_id] = order_record
        
        # 保存到文件
        if self.save_orders:
            self._save_order_history()
    
    def _trigger_callbacks(self, status: OrderStatus, order_record: OrderRecord):
        """触发状态回调"""
        callbacks = self.order_callbacks.get(status, [])
        for callback in callbacks:
            try:
                callback(order_record)
            except Exception as e:
                self.logger.error(f"Error in order callback: {e}")
    
    def add_callback(self, status: OrderStatus, callback: Callable[[OrderRecord], None]):
        """添加订单状态回调"""
        if status not in self.order_callbacks:
            self.order_callbacks[status] = []
        self.order_callbacks[status].append(callback)
    
    def remove_callback(self, status: OrderStatus, callback: Callable[[OrderRecord], None]):
        """移除订单状态回调"""
        if status in self.order_callbacks and callback in self.order_callbacks[status]:
            self.order_callbacks[status].remove(callback)
    
    def get_order(self, order_id: str) -> Optional[OrderRecord]:
        """获取订单记录"""
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[OrderRecord]:
        """获取指定股票的所有订单"""
        if symbol not in self.symbol_orders:
            return []
        
        return [self.orders[order_id] for order_id in self.symbol_orders[symbol] 
                if order_id in self.orders]
    
    def get_active_orders(self) -> List[OrderRecord]:
        """获取活跃订单"""
        return [self.orders[order_id] for order_id in self.active_orders 
                if order_id in self.orders]
    
    def get_filled_orders(self, since: datetime = None) -> List[OrderRecord]:
        """获取已成交订单"""
        filled = [self.orders[order_id] for order_id in self.filled_orders 
                  if order_id in self.orders]
        
        if since:
            filled = [order for order in filled 
                     if order.filled_at and order.filled_at >= since]
        
        return filled
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """获取订单统计信息"""
        total_orders = len(self.orders)
        active_count = len(self.active_orders)
        filled_count = len(self.filled_orders)
        
        status_counts = {}
        for order in self.orders.values():
            status = order.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_orders': total_orders,
            'active_orders': active_count,
            'filled_orders': filled_count,
            'status_distribution': status_counts,
            'symbols_traded': len(self.symbol_orders)
        }
    
    def _start_monitoring(self):
        """启动订单监控线程"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Order monitoring started")
    
    def _stop_monitoring(self):
        """停止订单监控"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """监控循环 - 检查超时订单等"""
        while self.is_monitoring:
            try:
                now = datetime.now()
                timeout_orders = []
                
                # 检查超时订单
                for order_id in list(self.active_orders):
                    if order_id in self.orders:
                        order_record = self.orders[order_id]
                        if (order_record.submitted_at and 
                            (now - order_record.submitted_at).total_seconds() > self.order_timeout):
                            timeout_orders.append(order_record)
                
                # 处理超时订单
                for order_record in timeout_orders:
                    self.logger.warning(f"Order {order_record.order_id} timeout, attempting to cancel")
                    self.cancel_order(order_record.order_id)
                
                time.sleep(30)  # 每30秒检查一次
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # 发生错误时等待更长时间
    
    def _save_order_history(self):
        """保存订单历史到文件"""
        try:
            os.makedirs(os.path.dirname(self.orders_file), exist_ok=True)
            
            orders_data = {
                order_id: order_record.to_dict() 
                for order_id, order_record in self.orders.items()
            }
            
            with open(self.orders_file, 'w', encoding='utf-8') as f:
                json.dump(orders_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving order history: {e}")
    
    def _load_order_history(self):
        """从文件加载订单历史"""
        try:
            if os.path.exists(self.orders_file):
                with open(self.orders_file, 'r', encoding='utf-8') as f:
                    orders_data = json.load(f)
                
                for order_id, order_dict in orders_data.items():
                    # 重建OrderRecord对象
                    order_record = OrderRecord(
                        order_id=order_dict['order_id'],
                        symbol=order_dict['symbol'],
                        action=order_dict['action'],
                        quantity=order_dict['quantity'],
                        order_type=OrderType(order_dict['order_type']),
                        limit_price=order_dict.get('limit_price'),
                        stop_price=order_dict.get('stop_price'),
                        take_profit_price=order_dict.get('take_profit_price'),
                        stop_loss_price=order_dict.get('stop_loss_price'),
                        status=OrderStatus(order_dict['status']),
                        ib_order_id=order_dict.get('ib_order_id'),
                        filled_quantity=order_dict.get('filled_quantity', 0),
                        remaining_quantity=order_dict.get('remaining_quantity', 0),
                        avg_fill_price=order_dict.get('avg_fill_price'),
                        commission=order_dict.get('commission'),
                        error_message=order_dict.get('error_message'),
                        retry_count=order_dict.get('retry_count', 0),
                        strategy_name=order_dict.get('strategy_name'),
                        reason=order_dict.get('reason')
                    )
                    
                    # 解析时间戳
                    if order_dict.get('submitted_at'):
                        order_record.submitted_at = datetime.fromisoformat(order_dict['submitted_at'])
                    if order_dict.get('filled_at'):
                        order_record.filled_at = datetime.fromisoformat(order_dict['filled_at'])
                    if order_dict.get('cancelled_at'):
                        order_record.cancelled_at = datetime.fromisoformat(order_dict['cancelled_at'])
                    if order_dict.get('created_at'):
                        order_record.created_at = datetime.fromisoformat(order_dict['created_at'])
                    
                    self.orders[order_id] = order_record
                    
                    # 重建映射关系
                    if order_record.ib_order_id:
                        self.ib_order_mapping[order_record.ib_order_id] = order_id
                    
                    # 重建符号映射
                    if order_record.symbol not in self.symbol_orders:
                        self.symbol_orders[order_record.symbol] = []
                    self.symbol_orders[order_record.symbol].append(order_id)
                    
                    # 重建状态集合
                    if order_record.status == OrderStatus.FILLED:
                        self.filled_orders.add(order_id)
                    elif order_record.status in [OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                        self.active_orders.add(order_id)
                
                self.logger.info(f"Loaded {len(self.orders)} orders from history")
                
        except Exception as e:
            self.logger.error(f"Error loading order history: {e}")
    
    def cleanup(self):
        """清理资源"""
        self._stop_monitoring()
        
        if self.save_orders:
            self._save_order_history()
        
        self.logger.info("Order manager cleaned up")


# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建配置
    config = {
        'auto_retry': True,
        'retry_delay_seconds': 5,
        'order_timeout_seconds': 300,
        'save_orders': True,
        'orders_file': 'orders/order_history.json'
    }
    
    # 模拟IB连接
    class MockIB:
        def isConnected(self):
            return True
        
        def placeOrder(self, contract, order):
            return None
    
    mock_ib = MockIB()
    
    # 创建订单管理器
    order_manager = EnhancedOrderManager(mock_ib, config)
    
    # 添加回调示例
    def on_order_filled(order_record: OrderRecord):
        print(f"Order filled: {order_record.symbol} {order_record.action} {order_record.filled_quantity} @ {order_record.avg_fill_price}")
    
    order_manager.add_callback(OrderStatus.FILLED, on_order_filled)
    
    print("Enhanced order manager initialized successfully")