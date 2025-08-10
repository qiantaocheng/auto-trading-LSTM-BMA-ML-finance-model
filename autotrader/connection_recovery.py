#!/usr/bin/env python3
"""
连接恢复模块 - 处理断线重连和状态恢复
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from ib_insync import IB

from .order_state_machine import OrderManager, OrderState


class ConnectionState(Enum):
    """连接状态"""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING" 
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"


@dataclass
class RecoveryConfig:
    """恢复配置"""
    max_reconnect_attempts: int = 10
    reconnect_interval: float = 5.0     # 重连间隔
    reconnect_backoff: float = 1.5      # 退避倍数
    max_reconnect_interval: float = 60.0 # 最大重连间隔
    connection_timeout: float = 30.0    # 连接超时
    recovery_timeout: float = 120.0     # 状态恢复超时
    
    # 状态同步配置
    sync_account_data: bool = True
    sync_positions: bool = True
    sync_open_orders: bool = True
    sync_market_data: bool = True


@dataclass
class ConnectionSnapshot:
    """连接快照 - 用于恢复"""
    timestamp: float = field(default_factory=time.time)
    subscribed_symbols: Set[str] = field(default_factory=set)
    account_ready: bool = False
    last_account_update: float = 0.0
    positions: Dict[str, int] = field(default_factory=dict)
    open_order_ids: Set[int] = field(default_factory=set)
    cash_balance: float = 0.0
    net_liq: float = 0.0
    
    # 策略状态
    active_strategies: Set[str] = field(default_factory=set)
    stop_tasks: Set[str] = field(default_factory=set)  # 有止损任务的symbols


class ConnectionRecoveryManager:
    """连接恢复管理器 - 简化版，重试委托给TaskManager"""
    
    def __init__(self, trader_instance, task_manager=None, config: RecoveryConfig = None):
        self.trader = trader_instance  # IbkrAutoTrader实例
        self.task_manager = task_manager  # TaskManager实例
        self.config = config or RecoveryConfig()
        self.logger = logging.getLogger("ConnectionRecovery")
        
        # 连接状态
        self.connection_state = ConnectionState.DISCONNECTED
        self.last_disconnect_time: Optional[float] = None
        
        # 状态快照
        self.last_snapshot: Optional[ConnectionSnapshot] = None
        self.recovery_in_progress = False
        
        # 恢复任务
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # 回调函数
        self.on_disconnect_callbacks: List[Callable] = []
        self.on_reconnect_callbacks: List[Callable] = []
        self.on_recovery_complete_callbacks: List[Callable] = []
        
        # 设置IB事件监听
        self._setup_ib_event_handlers()
    
    def _setup_ib_event_handlers(self):
        """设置IB事件处理器"""
        if hasattr(self.trader, 'ib') and self.trader.ib:
            # 连接事件
            self.trader.ib.connectedEvent += self._on_ib_connected
            self.trader.ib.disconnectedEvent += self._on_ib_disconnected
            self.trader.ib.errorEvent += self._on_ib_error
    
    def _on_ib_connected(self):
        """IB连接成功回调"""
        self.logger.info("IBKR连接已建立")
        self.connection_state = ConnectionState.CONNECTED
        
        # 使用TaskManager启动心跳监控
        if self.task_manager:
            self.task_manager.ensure_task_running(
                "heartbeat_monitor", 
                self._heartbeat_monitor,
                max_restarts=100,  # 心跳任务需要持续运行
                restart_delay=1.0
            )
        else:
            # 回退方案
            if not self._heartbeat_task or self._heartbeat_task.done():
                self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        
        # 如果是重连，启动恢复流程
        if self.last_disconnect_time is not None:
            if self.task_manager:
                self.task_manager.ensure_task_running(
                    "state_recovery", 
                    self._start_recovery,
                    max_restarts=1,  # 恢复任务只需要运行一次
                    restart_delay=0.0
                )
            else:
                asyncio.create_task(self._start_recovery())
    
    def _on_ib_disconnected(self):
        """IB断开连接回调"""
        self.logger.warning("IBKR连接已断开")
        self.connection_state = ConnectionState.DISCONNECTED
        self.last_disconnect_time = time.time()
        
        # 保存当前状态快照
        self._create_snapshot()
        
        # 停止心跳监控任务
        if self.task_manager:
            self.task_manager.cancel_task("heartbeat_monitor")
        elif self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        
        # 通知断开连接
        for callback in self.on_disconnect_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.warning(f"断开连接回调失败: {e}")
        
        # 使用TaskManager启动重连任务（带重试机制）
        if self.task_manager:
            self.task_manager.ensure_task_running(
                "connection_recovery", 
                self._reconnect_with_recovery,
                max_restarts=self.config.max_reconnect_attempts,
                restart_delay=self.config.reconnect_interval
            )
        else:
            # 回退方案：直接创建重连任务
            self._recovery_task = asyncio.create_task(self._reconnect_loop())
    
    def _on_ib_error(self, reqId, errorCode, errorString, contract):
        """IB错误回调"""
        # 检查是否是连接相关错误
        connection_errors = [502, 504, 1100, 1101, 1102, 2104, 2106, 2108]
        if errorCode in connection_errors:
            self.logger.warning(f"检测到连接错误: {errorCode} - {errorString}")
            if self.connection_state == ConnectionState.CONNECTED:
                # 标记为连接问题，但不立即断开（等待disconnectedEvent）
                self.connection_state = ConnectionState.RECONNECTING
    
    def _create_snapshot(self):
        """创建当前状态快照"""
        try:
            snapshot = ConnectionSnapshot()
            
            # 基础状态
            snapshot.account_ready = getattr(self.trader, 'account_ready', False)
            snapshot.last_account_update = getattr(self.trader, '_last_account_update', 0.0)
            snapshot.cash_balance = getattr(self.trader, 'cash_balance', 0.0)
            snapshot.net_liq = getattr(self.trader, 'net_liq', 0.0)
            
            # 持仓
            if hasattr(self.trader, 'positions') and self.trader.positions:
                snapshot.positions = self.trader.positions.copy()
            
            # 订阅的标的
            if hasattr(self.trader, 'subscriptions') and self.trader.subscriptions:
                snapshot.subscribed_symbols = set(self.trader.subscriptions.keys())
            
            # 未完成订单
            if hasattr(self.trader, 'open_orders') and self.trader.open_orders:
                snapshot.open_order_ids = set(self.trader.open_orders.keys())
            
            # 止损任务
            if hasattr(self.trader, '_stop_tasks') and self.trader._stop_tasks:
                snapshot.stop_tasks = set(
                    symbol for symbol, task in self.trader._stop_tasks.items() 
                    if not task.done()
                )
            
            self.last_snapshot = snapshot
            self.logger.info(f"状态快照已创建: {len(snapshot.subscribed_symbols)}个订阅, {len(snapshot.positions)}个持仓")
            
        except Exception as e:
            self.logger.error(f"创建状态快照失败: {e}")
    
    async def _reconnect_with_recovery(self):
        """简化的重连方法 - 供TaskManager调用，单次重连尝试"""
        if self.connection_state == ConnectionState.CONNECTED:
            # 已经连接，无需重连
            return
        
        self.connection_state = ConnectionState.RECONNECTING
        self.logger.info("尝试重新连接...")
        
        try:
            # 尝试重新连接
            await asyncio.wait_for(
                self.trader.connect(), 
                timeout=self.config.connection_timeout
            )
            
            # 等待连接确认
            await asyncio.sleep(1.0)
            
            if self.connection_state == ConnectionState.CONNECTED:
                self.logger.info("重连成功")
                # TaskManager会停止重试
                return
            else:
                # 连接未成功确认，抛出异常让TaskManager重试
                raise Exception("连接未确认成功")
                
        except asyncio.TimeoutError:
            self.logger.warning("重连超时")
            raise
        except Exception as e:
            self.logger.warning(f"重连失败: {e}")
            raise
    
    async def _reconnect_loop(self):
        """保留的传统重连循环（回退方案）"""
        current_interval = self.config.reconnect_interval
        reconnect_attempts = 0
        
        while reconnect_attempts < self.config.max_reconnect_attempts:
            if self.connection_state == ConnectionState.CONNECTED:
                break
            
            reconnect_attempts += 1
            self.connection_state = ConnectionState.RECONNECTING
            
            self.logger.info(f"尝试重连 ({reconnect_attempts}/{self.config.max_reconnect_attempts})")
            
            try:
                await asyncio.wait_for(
                    self.trader.connect(), 
                    timeout=self.config.connection_timeout
                )
                
                await asyncio.sleep(1.0)
                
                if self.connection_state == ConnectionState.CONNECTED:
                    self.logger.info("重连成功")
                    break
                
            except asyncio.TimeoutError:
                self.logger.warning(f"重连超时 (尝试 {reconnect_attempts})")
            except Exception as e:
                self.logger.warning(f"重连失败: {e} (尝试 {reconnect_attempts})")
            
            # 等待下次重连
            self.logger.info(f"等待 {current_interval:.1f}s 后重试...")
            await asyncio.sleep(current_interval)
            
            # 指数退避
            current_interval = min(
                current_interval * self.config.reconnect_backoff, 
                self.config.max_reconnect_interval
            )
        
        if reconnect_attempts >= self.config.max_reconnect_attempts:
            self.logger.error("达到最大重连次数，停止重连")
            self.connection_state = ConnectionState.FAILED
    
    async def _start_recovery(self):
        """开始状态恢复流程"""
        if self.recovery_in_progress or not self.last_snapshot:
            return
        
        self.recovery_in_progress = True
        self.logger.info("开始状态恢复...")
        
        try:
            await asyncio.wait_for(
                self._recover_state(), 
                timeout=self.config.recovery_timeout
            )
            
            # 通知恢复完成
            for callback in self.on_recovery_complete_callbacks:
                try:
                    callback()
                except Exception as e:
                    self.logger.warning(f"恢复完成回调失败: {e}")
            
            self.logger.info("状态恢复完成")
            
        except asyncio.TimeoutError:
            self.logger.error("状态恢复超时")
        except Exception as e:
            self.logger.error(f"状态恢复失败: {e}")
        finally:
            self.recovery_in_progress = False
            self.last_disconnect_time = None
    
    async def _recover_state(self):
        """恢复状态"""
        snapshot = self.last_snapshot
        if not snapshot:
            return
        
        recovery_tasks = []
        
        # 1. 恢复账户数据
        if self.config.sync_account_data:
            recovery_tasks.append(self._recover_account_data())
        
        # 2. 恢复持仓数据
        if self.config.sync_positions:
            recovery_tasks.append(self._recover_positions())
        
        # 3. 恢复订单追踪
        if self.config.sync_open_orders:
            recovery_tasks.append(self._recover_open_orders())
        
        # 4. 恢复市场数据订阅
        if self.config.sync_market_data:
            recovery_tasks.append(self._recover_market_data_subscriptions())
        
        # 5. 恢复策略状态
        recovery_tasks.append(self._recover_strategy_state())
        
        # 并发执行所有恢复任务
        await asyncio.gather(*recovery_tasks, return_exceptions=True)
    
    async def _recover_account_data(self):
        """恢复账户数据"""
        try:
            self.logger.info("恢复账户数据...")
            await self.trader.refresh_account_balances_and_positions()
            self.logger.info("账户数据恢复完成")
        except Exception as e:
            self.logger.error(f"恢复账户数据失败: {e}")
    
    async def _recover_positions(self):
        """恢复持仓数据"""
        try:
            self.logger.info("验证持仓数据...")
            # 账户刷新已经包含了持仓数据，这里做验证
            if hasattr(self.trader, 'positions') and self.last_snapshot.positions:
                current_positions = self.trader.positions
                snapshot_positions = self.last_snapshot.positions
                
                # 比较持仓差异
                for symbol, old_qty in snapshot_positions.items():
                    new_qty = current_positions.get(symbol, 0)
                    if new_qty != old_qty:
                        self.logger.info(f"{symbol} 持仓变化: {old_qty} -> {new_qty}")
            
            self.logger.info("持仓数据验证完成")
        except Exception as e:
            self.logger.error(f"恢复持仓数据失败: {e}")
    
    async def _recover_open_orders(self):
        """恢复订单追踪"""
        try:
            self.logger.info("恢复订单追踪...")
            
            # 获取当前未完成订单
            open_trades = self.trader.ib.openTrades()
            current_order_ids = {trade.order.orderId for trade in open_trades}
            
            # 与快照对比
            snapshot_order_ids = self.last_snapshot.open_order_ids
            
            # 新订单（快照中没有的）
            new_orders = current_order_ids - snapshot_order_ids
            # 已完成订单（快照中有但现在没有的）
            completed_orders = snapshot_order_ids - current_order_ids
            
            if new_orders:
                self.logger.info(f"发现新订单: {new_orders}")
            
            if completed_orders:
                self.logger.info(f"订单已完成: {completed_orders}")
                # 更新订单状态机
                for order_id in completed_orders:
                    order_sm = await self.trader.order_manager.get_order(order_id)
                    if order_sm and order_sm.is_active():
                        # 标记为已完成（假设是成交，实际需要查询具体状态）
                        await self.trader.order_manager.update_order_state(
                            order_id, OrderState.FILLED, 
                            {'recovery': True}, "断线期间完成"
                        )
            
            # 恢复订单状态机追踪
            for trade in open_trades:
                order_id = trade.order.orderId
                if order_id not in self.trader.open_orders:
                    # 重新建立订单追踪
                    from .ibkr_auto_trader import OrderRef
                    ref = OrderRef(
                        order_id=order_id,
                        symbol=trade.contract.symbol,
                        side=trade.order.action,
                        qty=int(trade.order.totalQuantity),
                        order_type=trade.order.orderType
                    )
                    self.trader.open_orders[order_id] = ref
                    self.logger.info(f"恢复订单追踪: {order_id}")
            
            self.logger.info("订单追踪恢复完成")
            
        except Exception as e:
            self.logger.error(f"恢复订单追踪失败: {e}")
    
    async def _recover_market_data_subscriptions(self):
        """恢复市场数据订阅"""
        try:
            self.logger.info("恢复市场数据订阅...")
            
            if self.last_snapshot.subscribed_symbols:
                for symbol in self.last_snapshot.subscribed_symbols:
                    try:
                        await self.trader.subscribe(symbol)
                        self.logger.debug(f"恢复订阅: {symbol}")
                        await asyncio.sleep(0.1)  # 避免过快请求
                    except Exception as e:
                        self.logger.warning(f"恢复订阅失败 {symbol}: {e}")
                
                self.logger.info(f"市场数据订阅恢复完成: {len(self.last_snapshot.subscribed_symbols)}个标的")
            
        except Exception as e:
            self.logger.error(f"恢复市场数据订阅失败: {e}")
    
    async def _recover_strategy_state(self):
        """恢复策略状态"""
        try:
            self.logger.info("恢复策略状态...")
            
            # 恢复止损任务
            if (hasattr(self.trader, '_stop_tasks') and 
                hasattr(self.trader, '_start_dynamic_stop_manager') and
                self.last_snapshot.stop_tasks):
                
                for symbol in self.last_snapshot.stop_tasks:
                    if symbol not in self.trader._stop_tasks or self.trader._stop_tasks[symbol].done():
                        try:
                            # 重新启动止损任务
                            self.trader._stop_tasks[symbol] = asyncio.create_task(
                                self.trader._dynamic_stop_manager(symbol)
                            )
                            self.logger.debug(f"恢复止损任务: {symbol}")
                        except Exception as e:
                            self.logger.warning(f"恢复止损任务失败 {symbol}: {e}")
            
            self.logger.info("策略状态恢复完成")
            
        except Exception as e:
            self.logger.error(f"恢复策略状态失败: {e}")
    
    async def _heartbeat_monitor(self):
        """心跳监控"""
        check_interval = 30.0  # 30秒检查一次
        
        while self.connection_state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(check_interval)
                
                if self.connection_state != ConnectionState.CONNECTED:
                    break
                
                # 简单的心跳检查 - 请求账户摘要
                try:
                    await asyncio.wait_for(
                        self.trader.ib.accountSummaryAsync(), 
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("心跳检查超时，可能连接异常")
                    # 不立即断开，等待IB的disconnectedEvent
                except Exception as e:
                    self.logger.warning(f"心跳检查失败: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"心跳监控异常: {e}")
                break
        
        self.logger.info("心跳监控已停止")
    
    def add_disconnect_callback(self, callback: Callable):
        """添加断开连接回调"""
        self.on_disconnect_callbacks.append(callback)
    
    def add_reconnect_callback(self, callback: Callable):
        """添加重连回调"""
        self.on_reconnect_callbacks.append(callback)
    
    def add_recovery_complete_callback(self, callback: Callable):
        """添加恢复完成回调"""
        self.on_recovery_complete_callbacks.append(callback)
    
    def get_connection_status(self) -> Dict[str, Any]:
        """获取连接状态"""
        return {
            'state': self.connection_state.value,
            'reconnect_attempts': self.reconnect_attempts,
            'last_disconnect': self.last_disconnect_time,
            'recovery_in_progress': self.recovery_in_progress,
            'has_snapshot': self.last_snapshot is not None,
            'snapshot_age': time.time() - self.last_snapshot.timestamp if self.last_snapshot else None
        }
    
    async def force_reconnect(self) -> bool:
        """强制重连"""
        try:
            self.logger.info("执行强制重连...")
            
            # 断开当前连接
            if self.trader.ib.isConnected():
                self.trader.ib.disconnect()
            
            # 等待断开
            await asyncio.sleep(1.0)
            
            # 重新连接
            await self.trader.connect()
            
            return True
            
        except Exception as e:
            self.logger.error(f"强制重连失败: {e}")
            return False
    
    async def cleanup(self):
        """清理资源"""
        # 取消任务
        if self._recovery_task and not self._recovery_task.done():
            self._recovery_task.cancel()
        
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        
        # 等待任务完成
        tasks = [t for t in [self._recovery_task, self._heartbeat_task] if t and not t.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info("连接恢复管理器已清理")