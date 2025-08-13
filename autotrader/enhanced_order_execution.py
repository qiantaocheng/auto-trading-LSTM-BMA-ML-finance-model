#!/usr/bin/env python3
"""
增强订单执行模块 - 专业级订单执行算法
"""

import asyncio
import time
import logging
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass
from enum import Enum

from ib_insync import IB, Contract, Trade, Order, MarketOrder, LimitOrder

from .order_state_machine import OrderManager, OrderState, OrderType, OrderStateMachine

# OrderRef 在 ibkr_auto_trader.py 中定义，避免循环导入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .ibkr_auto_trader import OrderRef


class ExecutionAlgorithm(Enum):
    """执行算法类型"""
    MARKET = "MARKET"           # 市价单
    LIMIT = "LIMIT"             # 限价单
    ADAPTIVE = "ADAPTIVE"       # 自适应限价
    TWAP = "TWAP"              # 时间加权平均价格
    VWAP = "VWAP"              # 成交量加权平均价格
    ICEBERG = "ICEBERG"        # 冰山订单


@dataclass
class ExecutionConfig:
    """执行配置"""
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.MARKET
    timeout_base: float = 30.0      # 基础超时时间
    timeout_max: float = 120.0      # 最大超时时间
    liquidity_factor: float = 1.0   # 流动性因子 (0-1)
    urgency_factor: float = 0.5     # 紧急程度 (0-1)
    max_retries: int = 3           # 最大重试次数
    partial_fill_min: float = 0.1  # 最小部分成交比例
    price_tolerance: float = 0.02  # 价格容忍度 (2%)
    
    # TWAP配置
    twap_duration_minutes: int = 10
    twap_slice_count: int = 5
    
    # 冰山单配置
    iceberg_slice_size: float = 0.2  # 每次显示20%


class LiquidityEstimator:
    """流动性估算器"""
    
    def __init__(self):
        self.logger = logging.getLogger("LiquidityEstimator")
        self._liquidity_cache: Dict[str, float] = {}
        self._cache_ttl = 300  # 5分钟缓存
        self._cache_timestamps: Dict[str, float] = {}
    
    async def estimate_liquidity(self, ib: IB, symbol: str) -> float:
        """估算股票流动性 (0-1, 1为最高流动性)"""
        # 检查缓存
        current_time = time.time()
        if symbol in self._liquidity_cache:
            if current_time - self._cache_timestamps.get(symbol, 0) < self._cache_ttl:
                return self._liquidity_cache[symbol]
        
        try:
            # 获取ticker信息
            from ib_insync import Stock
            contract = Stock(symbol, exchange='SMART', currency='USD')
            ticker = ib.ticker(contract)
            
            if not ticker:
                # 请求ticker数据
                ib.reqMktData(contract)
                await asyncio.sleep(2)  # 等待数据
                ticker = ib.ticker(contract)
            
            # 基于买卖价差估算流动性
            bid = getattr(ticker, 'bid', 0)
            ask = getattr(ticker, 'ask', 0)
            last = getattr(ticker, 'last', 0)
            
            if bid > 0 and ask > 0 and last > 0:
                spread = ask - bid
                spread_pct = spread / last
                
                # 价差越小，流动性越好
                if spread_pct < 0.001:  # 0.1%
                    liquidity = 1.0
                elif spread_pct < 0.005:  # 0.5%
                    liquidity = 0.8
                elif spread_pct < 0.01:   # 1%
                    liquidity = 0.6
                elif spread_pct < 0.02:   # 2%
                    liquidity = 0.4
                else:
                    liquidity = 0.2
            else:
                # 没有买卖价，流动性较低
                liquidity = 0.3
            
            # 缓存结果
            self._liquidity_cache[symbol] = liquidity
            self._cache_timestamps[symbol] = current_time
            
            self.logger.debug(f"{symbol} 流动性估算: {liquidity:.2f}")
            return liquidity
            
        except Exception as e:
            self.logger.warning(f"流动性估算失败 {symbol}: {e}")
            return 0.5  # 默认中等流动性


class EnhancedOrderExecutor:
    """增强订单执行器"""
    
    def __init__(self, ib: IB, order_manager: OrderManager):
        self.ib = ib
        self.order_manager = order_manager
        self.liquidity_estimator = LiquidityEstimator()
        self.logger = logging.getLogger("EnhancedOrderExecutor")
        
        # 执行统计
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'timeout_orders': 0,
            'rejected_orders': 0,
            'avg_execution_time': 0.0
        }
        
        # 使用任务生命周期管理器
        from .task_lifecycle_manager import get_task_manager
        self.task_manager = get_task_manager()
    
    def calculate_dynamic_timeout(self, config: ExecutionConfig, liquidity: float) -> float:
        """计算动态超时时间"""
        # 基础超时 + 流动性调整 + 紧急程度调整
        base_timeout = config.timeout_base
        liquidity_adjustment = (1.0 - liquidity) * 30  # 低流动性增加最多30秒
        urgency_adjustment = (1.0 - config.urgency_factor) * 20  # 低紧急度增加最多20秒
        
        dynamic_timeout = base_timeout + liquidity_adjustment + urgency_adjustment
        return min(dynamic_timeout, config.timeout_max)
    
    async def execute_market_order(self, symbol: str, action: str, quantity: int, 
                                 config: ExecutionConfig) -> OrderStateMachine:
        """执行市价单 - 带动态超时和状态轮询"""
        start_time = time.time()
        
        # 估算流动性
        liquidity = await self.liquidity_estimator.estimate_liquidity(self.ib, symbol)
        
        # 计算动态超时
        timeout = self.calculate_dynamic_timeout(config, liquidity)
        
        # 创建合约和订单
        from ib_insync import Stock
        contract = Stock(symbol, exchange='SMART', currency='USD')
        order = MarketOrder(action, quantity)
        
        # 下单
        trade = self.ib.placeOrder(contract, order)
        order_id = trade.order.orderId
        
        # 创建订单状态机
        order_sm = await self.order_manager.create_order(
            order_id=order_id,
            symbol=symbol,
            side=action,
            quantity=quantity,
            order_type=OrderType.MARKET
        )
        
        self.logger.info(f"提交市价单: {symbol} {action} {quantity} (ID: {order_id}, 超时: {timeout:.1f}s)")
        
        # 更新为已提交状态
        await self.order_manager.update_order_state(
            order_id, OrderState.SUBMITTED, 
            {'trade': trade, 'liquidity': liquidity}, 
            "订单已提交"
        )
        
        # 使用任务生命周期管理器创建任务
        execution_task = self.task_manager.create_task(
            self._monitor_order_execution(order_sm, trade, timeout, config),
            task_id=f"market_order_monitor_{order_id}",
            creator="enhanced_order_execution",
            description=f"监控市价单执行: {symbol} {action} {quantity}",
            group="order_execution",
            max_lifetime=timeout + 30  # 超时时间 + 缓冲
        )
        
        return order_sm
    
    async def _monitor_order_execution(self, order_sm: OrderStateMachine, trade: Trade, 
                                     timeout: float, config: ExecutionConfig):
        """监控订单执行过程"""
        start_time = time.time()
        check_interval = 0.5  # 500ms检查间隔
        
        try:
            while time.time() - start_time < timeout:
                # 检查订单状态
                status = getattr(trade.orderStatus, 'status', 'Unknown')
                filled = getattr(trade.orderStatus, 'filled', 0)
                remaining = getattr(trade.orderStatus, 'remaining', order_sm.quantity)
                avg_fill_price = getattr(trade.orderStatus, 'avgFillPrice', 0.0)
                
                # 状态更新
                if status == 'Submitted' and order_sm.state == OrderState.SUBMITTED:
                    await self.order_manager.update_order_state(
                        order_sm.order_id, OrderState.ACKNOWLEDGED, 
                        {'ib_status': status}, "交易所确认"
                    )
                
                elif filled > 0:
                    # 有成交
                    fill_data = {
                        'filled_quantity': filled,
                        'avg_fill_price': avg_fill_price,
                        'remaining': remaining,
                        'ib_status': status
                    }
                    
                    if filled >= order_sm.quantity:
                        # 完全成交
                        order_sm.update_fill(filled, avg_fill_price)
                        self.logger.info(f"订单完全成交: {order_sm.order_id} - {filled}股 @ ${avg_fill_price:.2f}")
                        break
                    elif filled > order_sm.filled_quantity:
                        # 部分成交
                        order_sm.update_fill(filled, avg_fill_price)
                        self.logger.info(f"订单部分成交: {order_sm.order_id} - {filled}/{order_sm.quantity}股")
                
                elif status in ['Cancelled', 'Rejected']:
                    # 订单被取消或拒绝
                    if status == 'Cancelled':
                        await self.order_manager.update_order_state(
                            order_sm.order_id, OrderState.CANCELLED, 
                            {'ib_status': status}, "订单被取消"
                        )
                    else:
                        await self.order_manager.update_order_state(
                            order_sm.order_id, OrderState.REJECTED, 
                            {'ib_status': status}, "订单被拒绝"
                        )
                    break
                
                # 等待下次检查
                await asyncio.sleep(check_interval)
            
            # 超时处理
            if not order_sm.is_terminal():
                execution_time = time.time() - start_time
                
                # 检查是否有部分成交
                if order_sm.filled_quantity > 0:
                    fill_rate = order_sm.filled_quantity / order_sm.quantity
                    if fill_rate >= config.partial_fill_min:
                        # 部分成交可接受，取消剩余订单
                        self.ib.cancelOrder(trade.order)
                        self.logger.info(f"订单部分完成，取消剩余: {order_sm.order_id} ({fill_rate*100:.1f}%成交)")
                        
                        await self.order_manager.update_order_state(
                            order_sm.order_id, OrderState.CANCELLED, 
                            {'execution_time': execution_time, 'partial_fill': True}, 
                            f"超时后部分成交取消 ({fill_rate*100:.1f}%)"
                        )
                    else:
                        # 成交太少，标记为失败
                        await self.order_manager.update_order_state(
                            order_sm.order_id, OrderState.FAILED, 
                            {'execution_time': execution_time, 'timeout': True}, 
                            f"执行超时 ({execution_time:.1f}s)"
                        )
                else:
                    # 完全没有成交
                    self.ib.cancelOrder(trade.order)
                    await self.order_manager.update_order_state(
                        order_sm.order_id, OrderState.FAILED, 
                        {'execution_time': execution_time, 'timeout': True}, 
                        f"执行超时，无成交 ({execution_time:.1f}s)"
                    )
            
            # 更新统计
            self._update_execution_stats(order_sm, time.time() - start_time)
            
        except Exception as e:
            self.logger.error(f"订单监控异常 {order_sm.order_id}: {e}")
            await self.order_manager.update_order_state(
                order_sm.order_id, OrderState.FAILED, 
                {'error': str(e)}, f"监控异常: {e}"
            )
        finally:
            # 任务生命周期管理器会自动清理
            pass
    
    async def execute_adaptive_limit_order(self, symbol: str, action: str, quantity: int,
                                         reference_price: float, config: ExecutionConfig) -> OrderStateMachine:
        """执行自适应限价单"""
        # 基于市场条件和紧急度调整限价
        liquidity = await self.liquidity_estimator.estimate_liquidity(self.ib, symbol)
        
        # 价格调整逻辑
        if action.upper() == 'BUY':
            # 买单：根据紧急度和流动性确定限价
            aggressive_factor = config.urgency_factor * (1.0 - liquidity)
            limit_price = reference_price * (1 + config.price_tolerance * aggressive_factor)
        else:
            # 卖单
            aggressive_factor = config.urgency_factor * (1.0 - liquidity)
            limit_price = reference_price * (1 - config.price_tolerance * aggressive_factor)
        
        # 创建限价单
        from ib_insync import Stock
        contract = Stock(symbol, exchange='SMART', currency='USD')
        order = LimitOrder(action, quantity, limit_price)
        
        trade = self.ib.placeOrder(contract, order)
        order_id = trade.order.orderId
        
        # 创建订单状态机
        order_sm = await self.order_manager.create_order(
            order_id=order_id,
            symbol=symbol,
            side=action,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            price=limit_price
        )
        
        self.logger.info(f"提交自适应限价单: {symbol} {action} {quantity} @ ${limit_price:.2f}")
        
        # 启动监控（使用任务生命周期管理器）
        timeout = self.calculate_dynamic_timeout(config, liquidity)
        execution_task = self.task_manager.create_task(
            self._monitor_adaptive_limit_execution(order_sm, trade, timeout, config, reference_price),
            task_id=f"adaptive_limit_monitor_{order_id}",
            creator="enhanced_order_execution",
            description=f"监控自适应限价单: {symbol} {action} {quantity}",
            group="order_execution",
            max_lifetime=timeout + 30
        )
        
        return order_sm
    
    async def _monitor_adaptive_limit_execution(self, order_sm: OrderStateMachine, trade: Trade,
                                              timeout: float, config: ExecutionConfig, reference_price: float):
        """监控自适应限价单执行"""
        start_time = time.time()
        price_update_interval = 10.0  # 10秒更新一次价格
        last_price_update = start_time
        
        try:
            while time.time() - start_time < timeout and not order_sm.is_terminal():
                current_time = time.time()
                
                # 检查订单状态
                status = getattr(trade.orderStatus, 'status', 'Unknown')
                filled = getattr(trade.orderStatus, 'filled', 0)
                
                if filled > 0:
                    # 有成交，更新状态
                    avg_fill_price = getattr(trade.orderStatus, 'avgFillPrice', 0.0)
                    order_sm.update_fill(filled, avg_fill_price)
                    
                    if filled >= order_sm.quantity:
                        self.logger.info(f"自适应限价单完成: {order_sm.order_id}")
                        break
                
                # 动态价格调整
                if current_time - last_price_update > price_update_interval:
                    try:
                        # 获取当前市价
                        ticker = self.ib.ticker(trade.contract)
                        if ticker and hasattr(ticker, 'last') and ticker.last > 0:
                            current_market_price = ticker.last
                            
                            # 判断是否需要调整价格
                            if order_sm.side == 'BUY':
                                # 买单：如果市价上涨太多，提高限价
                                if current_market_price > reference_price * (1 + config.price_tolerance):
                                    new_limit_price = current_market_price * (1 + config.price_tolerance * 0.5)
                                    # 修改订单价格
                                    trade.order.lmtPrice = new_limit_price
                                    self.ib.placeOrder(trade.contract, trade.order)
                                    self.logger.info(f"调整买单限价: {order_sm.order_id} ${order_sm.price:.2f} -> ${new_limit_price:.2f}")
                            else:
                                # 卖单：如果市价下跌太多，降低限价
                                if current_market_price < reference_price * (1 - config.price_tolerance):
                                    new_limit_price = current_market_price * (1 - config.price_tolerance * 0.5)
                                    trade.order.lmtPrice = new_limit_price
                                    self.ib.placeOrder(trade.contract, trade.order)
                                    self.logger.info(f"调整卖单限价: {order_sm.order_id} ${order_sm.price:.2f} -> ${new_limit_price:.2f}")
                        
                        last_price_update = current_time
                        
                    except Exception as e:
                        self.logger.warning(f"价格调整失败 {order_sm.order_id}: {e}")
                
                await asyncio.sleep(1.0)  # 1秒检查间隔
            
            # 超时处理
            if not order_sm.is_terminal():
                self.ib.cancelOrder(trade.order)
                await self.order_manager.update_order_state(
                    order_sm.order_id, OrderState.CANCELLED, 
                    {'timeout': True}, "自适应限价单超时取消"
                )
            
        except Exception as e:
            self.logger.error(f"自适应限价单监控异常 {order_sm.order_id}: {e}")
            await self.order_manager.update_order_state(
                order_sm.order_id, OrderState.FAILED, 
                {'error': str(e)}, f"监控异常: {e}"
            )
        finally:
            # 任务生命周期管理器会自动清理
            pass
    
    def _update_execution_stats(self, order_sm: OrderStateMachine, execution_time: float):
        """更新执行统计"""
        self.execution_stats['total_orders'] += 1
        
        if order_sm.state == OrderState.FILLED:
            self.execution_stats['successful_orders'] += 1
        elif order_sm.state == OrderState.REJECTED:
            self.execution_stats['rejected_orders'] += 1
        else:
            self.execution_stats['timeout_orders'] += 1
        
        # 更新平均执行时间
        total = self.execution_stats['total_orders']
        old_avg = self.execution_stats['avg_execution_time']
        self.execution_stats['avg_execution_time'] = ((old_avg * (total - 1)) + execution_time) / total
    
    async def cancel_order(self, order_id: int, reason: str = "用户取消") -> bool:
        """取消订单"""
        order_sm = await self.order_manager.get_order(order_id)
        if not order_sm:
            return False
        
        if order_sm.is_terminal():
            self.logger.warning(f"订单已处于终态，无法取消: {order_id}")
            return False
        
        try:
            # 取消IB订单
            # 这里需要根据实际情况获取trade对象
            # 暂时使用通用取消方法
            self.ib.cancelOrder(order_id)
            
            # 更新状态
            await self.order_manager.update_order_state(order_id, OrderState.CANCELLED, reason=reason)
            
            # 通过任务管理器取消任务
            self.task_manager.cancel_task(f"market_order_monitor_{order_id}", "手动取消")
            self.task_manager.cancel_task(f"adaptive_limit_monitor_{order_id}", "手动取消")
            
            return True
            
        except Exception as e:
            self.logger.error(f"取消订单失败 {order_id}: {e}")
            return False
    
    async def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        stats = self.execution_stats.copy()
        
        # 计算成功率
        if stats['total_orders'] > 0:
            stats['success_rate'] = stats['successful_orders'] / stats['total_orders']
        else:
            stats['success_rate'] = 0.0
        
        # 活跃订单数（从任务管理器获取）
        order_tasks = self.task_manager.list_tasks(group="order_execution")
        stats['active_executions'] = len(order_tasks)
        
        return stats
    
    async def cleanup(self):
        """清理资源"""
        # 通过任务管理器取消所有订单执行任务
        cancelled_count = self.task_manager.cancel_group("order_execution", "系统清理")
        self.logger.info(f"订单执行器已清理，取消了 {cancelled_count} 个任务")

    # ==================== 高级执行算法 ====================
    
    async def execute_twap_order(self, contract, total_quantity: int, 
                                duration_minutes: int = 30, slice_count: int = 10):
        """TWAP (时间加权平均价格) 执行算法"""
        symbol = contract.symbol
        action = "BUY" if total_quantity > 0 else "SELL"
        abs_quantity = abs(total_quantity)
        
        self.logger.info(f"开始TWAP执行: {symbol} {action} {abs_quantity}股, {duration_minutes}分钟, {slice_count}片")
        
        # 计算每片参数
        slice_quantity = abs_quantity // slice_count
        remainder = abs_quantity % slice_count
        slice_interval = (duration_minutes * 60) / slice_count  # 秒
        
        executed_orders = []
        
        try:
            for i in range(slice_count):
                # 计算本片数量
                current_slice = slice_quantity
                if i < remainder:  # 余数分配给前几片
                    current_slice += 1
                
                if current_slice == 0:
                    continue
                
                self.logger.info(f"TWAP片段 {i+1}/{slice_count}: {current_slice}股")
                
                # 执行本片订单
                try:
                    from ib_insync import MarketOrder
                    order = MarketOrder(action, current_slice)
                    trade = self.ib.placeOrder(contract, order)
                    
                    # 等待成交
                    await asyncio.wait_for(trade.doneEvent.wait(), timeout=30.0)
                    
                    executed_orders.append({
                        'slice': i+1,
                        'quantity': current_slice,
                        'filled': getattr(trade.orderStatus, 'filled', 0),
                        'avg_price': getattr(trade.orderStatus, 'avgFillPrice', 0.0),
                        'status': trade.orderStatus.status
                    })
                    
                    # 等待片段间隔（除了最后一片）
                    if i < slice_count - 1:
                        await asyncio.sleep(slice_interval)
                        
                except Exception as e:
                    self.logger.error(f"TWAP片段 {i+1} 执行失败: {e}")
                    continue
            
            # 统计执行结果
            total_filled = sum(order['filled'] for order in executed_orders)
            total_value = sum(order['filled'] * order['avg_price'] for order in executed_orders)
            avg_price = total_value / total_filled if total_filled > 0 else 0
            
            self.logger.info(f"TWAP执行完成: {symbol} 总成交{total_filled}/{abs_quantity}股, 均价${avg_price:.4f}")
            
            return executed_orders
            
        except Exception as e:
            self.logger.error(f"TWAP执行异常: {symbol}: {e}")
            return executed_orders
    
    async def execute_vwap_order(self, contract, total_quantity: int,
                                participation_rate: float = 0.1):
        """VWAP (成交量加权平均价格) 执行算法"""
        symbol = contract.symbol
        action = "BUY" if total_quantity > 0 else "SELL"
        abs_quantity = abs(total_quantity)
        
        self.logger.info(f"开始VWAP执行: {symbol} {action} {abs_quantity}股, 参与率{participation_rate:.1%}")
        
        executed_orders = []
        remaining_qty = abs_quantity
        
        try:
            while remaining_qty > 0:
                # 获取当前成交量
                ticker = self.ib.ticker(contract)
                await self.ib.sleep(1)  # 等待数据更新
                
                # 计算当前时段的目标执行量
                if hasattr(ticker, 'volume') and ticker.volume > 0:
                    # 基于实际成交量计算
                    recent_volume = ticker.volume  # 简化版，实际应该是近期N分钟成交量
                    target_volume = int(recent_volume * participation_rate)
                else:
                    # 降级到固定比例
                    target_volume = max(1, int(remaining_qty * 0.1))
                
                # 限制单次执行量
                current_qty = min(target_volume, remaining_qty, 1000)  # 最大1000股
                
                if current_qty <= 0:
                    self.logger.warning("VWAP计算量为0，等待市场活跃")
                    await asyncio.sleep(30)  # 等待30秒
                    continue
                
                self.logger.info(f"VWAP执行: {current_qty}股 (剩余{remaining_qty})")
                
                try:
                    from ib_insync import MarketOrder
                    order = MarketOrder(action, current_qty)
                    trade = self.ib.placeOrder(contract, order)
                    
                    # 等待成交
                    await asyncio.wait_for(trade.doneEvent.wait(), timeout=30.0)
                    
                    filled = getattr(trade.orderStatus, 'filled', 0)
                    executed_orders.append({
                        'quantity': current_qty,
                        'filled': filled,
                        'avg_price': getattr(trade.orderStatus, 'avgFillPrice', 0.0),
                        'status': trade.orderStatus.status
                    })
                    
                    remaining_qty -= filled
                    
                    # 动态调整等待时间
                    if remaining_qty > 0:
                        wait_time = max(5, min(60, 30 * (1 - participation_rate)))
                        await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    self.logger.error(f"VWAP片段执行失败: {e}")
                    await asyncio.sleep(10)
                    continue
            
            # 统计执行结果
            total_filled = sum(order['filled'] for order in executed_orders)
            total_value = sum(order['filled'] * order['avg_price'] for order in executed_orders)
            avg_price = total_value / total_filled if total_filled > 0 else 0
            
            self.logger.info(f"VWAP执行完成: {symbol} 总成交{total_filled}/{abs_quantity}股, 均价${avg_price:.4f}")
            
            return executed_orders
            
        except Exception as e:
            self.logger.error(f"VWAP执行异常: {symbol}: {e}")
            return executed_orders