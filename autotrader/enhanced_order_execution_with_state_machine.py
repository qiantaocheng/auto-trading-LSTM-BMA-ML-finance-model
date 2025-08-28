#!/usr/bin/env python3
"""
=============================================================================
增强订单执行系统 - 整合版本
=============================================================================
整合以下订单管理功能:
- 增强订单执行 (原 enhanced_order_execution.py)
- 订单状态机管理 (整合自 order_state_machine.py)
整合时间: 2025-08-20
=============================================================================

提供智能订单路由、执行算法和完整的订单状态生命周期管理
"""

import logging
import time
import json
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)

# =============================================================================
# 订单状态机 (从 order_state_machine.py 整合)
# =============================================================================

class OrderState(Enum):
    """订单状态枚举"""
    PENDING = "PENDING"           # 待提交
    SUBMITTED = "SUBMITTED"       # 提交释放
    ACKNOWLEDGED = "ACKNOWLEDGED" # 确认
    PARTIAL = "PARTIAL"          # 部分execution
    FILLED = "FILLED"            # 完全execution
    CANCELLED = "CANCELLED"      # 取消
    REJECTED = "REJECTED"        # 拒绝
    EXPIRED = "EXPIRED"          # 过期
    FAILED = "FAILED"            # 执行failed

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
    """订单状态机 - 管理单个订单完整生命周期"""
    
    # 定义有效状态转换
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
        
        # 执行信息
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
            
            # 处理特定状态逻辑
            self._handle_state_entry(new_state, metadata)
            
            return True
    
    def _is_valid_transition(self, from_state: OrderState, to_state: OrderState) -> bool:
        """检查状态转换是否有效"""
        return to_state in self.VALID_TRANSITIONS.get(from_state, [])
    
    def _handle_state_entry(self, state: OrderState, metadata: Optional[Dict[str, Any]]):
        """处理进入新状态时逻辑"""
        if state == OrderState.PARTIAL and metadata:
            self._update_fill_info(metadata)
        elif state == OrderState.FILLED and metadata:
            self._update_fill_info(metadata)
        elif state in self.TERMINAL_STATES:
            self.logger.info(f"订单进入终态: {state.value} - {self._basic_info()}")
    
    def _update_fill_info(self, fill_data: Dict[str, Any]):
        """更新执行信息"""
        if 'filled_quantity' in fill_data:
            self.filled_quantity = fill_data['filled_quantity']
            self.remaining_quantity = self.quantity - self.filled_quantity
        
        if 'avg_fill_price' in fill_data:
            self.avg_fill_price = fill_data['avg_fill_price']
        
        self.logger.info(f"执行更新: {self.filled_quantity}/{self.quantity} @ {self.avg_fill_price}")
    
    def update_fill(self, filled_qty: int, avg_price: float) -> bool:
        """更新执行信息并转换状态"""
        with self._lock:
            fill_data = {
                'filled_quantity': filled_qty,
                'avg_fill_price': avg_price
            }
            
            if filled_qty >= self.quantity:
                # 完全执行
                return self.transition(OrderState.FILLED, fill_data, "完全执行")
            elif filled_qty > self.filled_quantity:
                # 部分执行
                return self.transition(OrderState.PARTIAL, fill_data, "部分执行")
            
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

# =============================================================================
# 订单执行系统 (从 enhanced_order_execution.py 整合并增强)
# =============================================================================

class OrderExecutionStrategy(Enum):
    """订单执行策略"""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    ADAPTIVE = "adaptive"

@dataclass
class OrderExecutionConfig:
    """增强订单执行配置"""
    strategy: OrderExecutionStrategy = OrderExecutionStrategy.ADAPTIVE
    max_participation_rate: float = 0.20  # 最大20%成交量参与
    time_horizon_minutes: int = 30         # 执行时间窗口
    price_improvement_threshold: float = 0.001  # 0.1%价格改善阈值
    min_fill_size: int = 100              # 最小成交数量
    urgency_factor: float = 0.5           # 0=耐心, 1=紧急

@dataclass
class ExecutionResult:
    """订单执行结果"""
    order_id: str
    symbol: str
    side: str
    requested_quantity: int
    filled_quantity: int
    average_price: float
    execution_time_seconds: float
    strategy_used: OrderExecutionStrategy
    total_cost: float
    market_impact: float
    success: bool = True
    error_message: Optional[str] = None
    state_machine: Optional[OrderStateMachine] = None

class OrderManager:
    """订单管理器 - 管理所有订单状态机"""
    
    def __init__(self, auditor=None):
        self.orders: Dict[int, OrderStateMachine] = {}
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger("OrderManager")
        self.auditor = auditor
        
        # 统计信息
        self.stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'failed_orders': 0,
        }
    
    def _audit_state_change(self, order_sm: OrderStateMachine, old_state: Optional[OrderState], 
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
                state_change_callback=self._audit_state_change
            )
            
            self.orders[order_id] = order_sm
            self.stats['total_orders'] += 1
            
            self.logger.info(f"创建订单状态机: {order_id} - {symbol} {side} {quantity}")
            return order_sm
    
    async def get_order(self, order_id: int) -> Optional[OrderStateMachine]:
        """获取订单状态机"""
        async with self._lock:
            return self.orders.get(order_id)
    
    async def get_active_orders(self) -> List[OrderStateMachine]:
        """获取所有活跃订单"""
        async with self._lock:
            return [order for order in self.orders.values() if order.is_active()]

class EnhancedOrderExecutor:
    """
    增强订单执行引擎 - 整合状态机管理
    提供智能订单路由和执行算法
    """
    
    def __init__(self, ib_client=None, order_manager=None, config: OrderExecutionConfig = None):
        self.ib_client = ib_client
        self.order_manager = order_manager or OrderManager()
        self.config = config or OrderExecutionConfig()
        self.logger = logger
        
        # 执行统计
        self.stats = {
            'total_orders': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'average_market_impact': 0.0,
            'total_volume_executed': 0
        }
        
        logger.info("增强订单执行器初始化完成")
    
    async def execute_order(self, symbol: str, side: str, quantity: int, 
                          limit_price: Optional[float] = None,
                          strategy: Optional[OrderExecutionStrategy] = None) -> ExecutionResult:
        """
        执行订单使用增强算法
        """
        start_time = time.time()
        order_id = f"{symbol}_{side}_{int(start_time * 1000)}"
        
        try:
            self.stats['total_orders'] += 1
            
            # 确定执行策略
            exec_strategy = strategy or self.config.strategy
            
            # 创建订单状态机
            order_type = OrderType.LIMIT if limit_price else OrderType.MARKET
            order_sm = await self.order_manager.create_order(
                order_id=int(start_time * 1000),
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=limit_price,
                strategy=exec_strategy.value
            )
            
            # 转换到已提交状态
            order_sm.transition(OrderState.SUBMITTED, reason="订单已提交到执行引擎")
            order_sm.transition(OrderState.ACKNOWLEDGED, reason="订单已确认")
            
            # 模拟执行
            result = await self._simulate_execution_with_state_machine(
                order_id, symbol, side, quantity, limit_price, exec_strategy, start_time, order_sm
            )
            
            if result.success:
                self.stats['successful_executions'] += 1
                self.stats['total_volume_executed'] += result.filled_quantity
                self._update_execution_stats(result)
            else:
                self.stats['failed_executions'] += 1
            
            return result
            
        except Exception as e:
            self.stats['failed_executions'] += 1
            error_msg = f"订单执行失败: {str(e)}"
            self.logger.error(error_msg)
            
            return ExecutionResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                requested_quantity=quantity,
                filled_quantity=0,
                average_price=0.0,
                execution_time_seconds=time.time() - start_time,
                strategy_used=exec_strategy,
                total_cost=0.0,
                market_impact=0.0,
                success=False,
                error_message=error_msg
            )
    
    async def execute_market_order(self, symbol: str, action: str, quantity: int, 
                                 config: OrderExecutionConfig = None) -> OrderStateMachine:
        """执行市价单并返回状态机"""
        start_time = time.time()
        order_id = int(start_time * 1000)
        
        # 创建订单状态机
        order_sm = await self.order_manager.create_order(
            order_id=order_id,
            symbol=symbol,
            side=action,
            quantity=quantity,
            order_type=OrderType.MARKET,
            strategy="market_execution"
        )
        
        # 状态转换
        order_sm.transition(OrderState.SUBMITTED, reason="市价单已提交")
        order_sm.transition(OrderState.ACKNOWLEDGED, reason="市价单已确认")
        
        # 模拟执行
        await asyncio.sleep(0.1)  # 模拟网络延迟
        
        # 模拟成功执行
        simulated_price = 150.0 + (hash(symbol) % 100) / 10.0  # 模拟价格
        order_sm.update_fill(quantity, simulated_price)
        
        return order_sm
    
    async def _simulate_execution_with_state_machine(self, order_id: str, symbol: str, side: str, 
                                                   quantity: int, limit_price: Optional[float], 
                                                   strategy: OrderExecutionStrategy, start_time: float,
                                                   order_sm: OrderStateMachine) -> ExecutionResult:
        """模拟执行并更新状态机"""
        
        # 模拟执行延迟
        execution_delay = 0.5 + (hash(symbol) % 1000) / 1000.0
        await asyncio.sleep(execution_delay)
        
        # 计算模拟执行价格
        base_price = 150.0 + (hash(symbol) % 100)
        spread = 0.01 * (1 + hash(order_id) % 5)
        
        if side.upper() == "BUY":
            execution_price = base_price + spread/2
        else:
            execution_price = base_price - spread/2
        
        # 价格检查（如果是限价单）
        if limit_price:
            if side.upper() == "BUY" and execution_price > limit_price:
                execution_price = limit_price
            elif side.upper() == "SELL" and execution_price < limit_price:
                execution_price = limit_price
        
        # 模拟填充
        success_rate = 0.95  # 95%成功率
        if hash(order_id) % 100 < success_rate * 100:
            # 成功执行
            filled_quantity = quantity
            
            # 可能部分填充
            if strategy == OrderExecutionStrategy.VWAP and quantity > 1000:
                # 大单可能分批执行
                if hash(order_id) % 3 == 0:
                    # 先部分填充
                    partial_qty = quantity // 2
                    order_sm.update_fill(partial_qty, execution_price)
                    
                    # 稍后完全填充
                    await asyncio.sleep(0.2)
                    filled_quantity = quantity
            
            # 完全填充
            order_sm.update_fill(filled_quantity, execution_price)
            
        else:
            # 执行失败
            filled_quantity = 0
            execution_price = 0.0
            order_sm.fail("模拟执行失败")
        
        # 计算成本和市场影响
        total_cost = filled_quantity * execution_price
        market_impact = abs(execution_price - base_price) / base_price
        actual_execution_time = time.time() - start_time
        
        return ExecutionResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            requested_quantity=quantity,
            filled_quantity=filled_quantity,
            average_price=execution_price,
            execution_time_seconds=actual_execution_time,
            strategy_used=strategy,
            total_cost=total_cost,
            market_impact=market_impact,
            success=filled_quantity > 0,
            state_machine=order_sm
        )
    
    def _update_execution_stats(self, result: ExecutionResult):
        """更新执行统计"""
        current_avg_time = self.stats['average_execution_time']
        current_avg_impact = self.stats['average_market_impact']
        successful_count = self.stats['successful_executions']
        
        # 更新平均执行时间
        if successful_count == 1:
            self.stats['average_execution_time'] = result.execution_time_seconds
            self.stats['average_market_impact'] = result.market_impact
        else:
            # 加权平均
            self.stats['average_execution_time'] = (
                (current_avg_time * (successful_count - 1) + result.execution_time_seconds) / successful_count
            )
            self.stats['average_market_impact'] = (
                (current_avg_impact * (successful_count - 1) + result.market_impact) / successful_count
            )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        return self.stats.copy()
    
    def update_config(self, config: OrderExecutionConfig):
        """更新执行配置"""
        self.config = config
        logger.info("增强订单执行配置已更新")

# =============================================================================
# 全局实例和工厂函数
# =============================================================================

# 全局执行器实例
_enhanced_executor = None
_global_order_manager = None

def get_enhanced_executor(ib_client=None, order_manager=None, config: OrderExecutionConfig = None) -> EnhancedOrderExecutor:
    """获取全局增强订单执行器"""
    global _enhanced_executor
    if _enhanced_executor is None:
        _enhanced_executor = EnhancedOrderExecutor(ib_client, order_manager, config)
    return _enhanced_executor

def get_order_manager(auditor=None) -> OrderManager:
    """获取全局订单管理器"""
    global _global_order_manager
    if _global_order_manager is None:
        _global_order_manager = OrderManager(auditor)
    return _global_order_manager

def create_enhanced_executor(ib_client=None, order_manager=None, config: OrderExecutionConfig = None) -> EnhancedOrderExecutor:
    """创建新的增强订单执行器"""
    return EnhancedOrderExecutor(ib_client, order_manager, config)

# 向后兼容的执行配置类
ExecutionConfig = OrderExecutionConfig