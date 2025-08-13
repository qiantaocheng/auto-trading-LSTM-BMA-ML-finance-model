#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一连接管理器 - 智能断线重连和连接状态管理
提供稳健的连接恢复机制和连接监控功能
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from enum import Enum
from threading import Lock
import math

class ConnectionState(Enum):
    """连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"

class ReconnectStrategy(Enum):
    """重连策略"""
    FIXED_DELAY = "fixed_delay"          # 固定延迟
    EXPONENTIAL_BACKOFF = "exponential"  # 指数退避
    LINEAR_BACKOFF = "linear"            # 线性退避

@dataclass
class ConnectionConfig:
    """连接配置"""
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 3130
    timeout: float = 20.0
    
    # 重连配置
    max_reconnect_attempts: int = 10
    reconnect_strategy: ReconnectStrategy = ReconnectStrategy.EXPONENTIAL_BACKOFF
    base_reconnect_delay: float = 5.0
    max_reconnect_delay: float = 300.0  # 5分钟
    
    # 监控配置
    health_check_interval: float = 30.0
    connection_timeout: float = 15.0
    
    # 账户配置
    account_id: Optional[str] = None
    use_delayed_if_no_realtime: bool = True

@dataclass
class ConnectionEvent:
    """连接事件"""
    timestamp: float
    event_type: str
    state: ConnectionState
    details: Dict[str, Any]

class UnifiedConnectionManager:
    """统一连接管理器"""
    
    def __init__(self, ib_client, config: ConnectionConfig, 
                 logger: Optional[logging.Logger] = None):
        self.ib = ib_client
        self.config = config
        self.logger = logger or logging.getLogger("UnifiedConnectionManager")
        
        # 连接状态
        self.state = ConnectionState.DISCONNECTED
        self.last_connection_time = 0.0
        self.reconnect_attempts = 0
        self.consecutive_failures = 0
        
        # 监控和回调
        self.connection_callbacks: List[Callable] = []
        self.disconnection_callbacks: List[Callable] = []
        self.health_check_task: Optional[asyncio.Task] = None
        
        # 事件历史
        self.connection_events: List[ConnectionEvent] = []
        self.max_event_history = 100
        
        # 同步锁
        self.connection_lock = asyncio.Lock()
        self.state_lock = Lock()
        
        # 统计
        self.stats = {
            'total_connections': 0,
            'total_disconnections': 0,
            'total_reconnect_attempts': 0,
            'successful_reconnects': 0,
            'connection_uptime': 0.0,
            'average_connection_duration': 0.0
        }
    
    async def connect(self, force: bool = False) -> bool:
        """建立连接"""
        async with self.connection_lock:
            if self.state == ConnectionState.CONNECTED and not force:
                self.logger.debug("已连接，跳过重复连接")
                return True
            
            if self.state == ConnectionState.CONNECTING:
                self.logger.debug("正在连接中，等待完成")
                return await self._wait_for_connection()
            
            self._set_state(ConnectionState.CONNECTING)
            self._record_event("connect_start", {"force": force})
            
            try:
                # 如果强制重连，先断开现有连接
                if force and self.ib.isConnected():
                    await self._disconnect_internal()
                
                # 执行连接
                connect_start = time.time()
                await asyncio.wait_for(
                    self.ib.connectAsync(
                        self.config.host, 
                        self.config.port, 
                        clientId=self.config.client_id
                    ),
                    timeout=self.config.connection_timeout
                )
                
                # 验证连接
                if not self.ib.isConnected():
                    raise ConnectionError("连接建立后验证失败")
                
                # 连接成功处理
                connect_duration = time.time() - connect_start
                self._set_state(ConnectionState.CONNECTED)
                self.last_connection_time = time.time()
                self.reconnect_attempts = 0
                self.consecutive_failures = 0
                
                self.stats['total_connections'] += 1
                self._record_event("connect_success", {
                    "duration": connect_duration,
                    "client_id": self.config.client_id
                })
                
                self.logger.info(f"连接成功: {self.config.host}:{self.config.port} "
                               f"(ClientID: {self.config.client_id}, 耗时: {connect_duration:.2f}s)")
                
                # 启动健康检查
                await self._start_health_check()
                
                # 执行连接回调
                await self._execute_callbacks(self.connection_callbacks)
                
                return True
                
            except asyncio.TimeoutError:
                self._set_state(ConnectionState.FAILED)
                self.consecutive_failures += 1
                self._record_event("connect_timeout", {
                    "timeout": self.config.connection_timeout
                })
                self.logger.error(f"连接超时: {self.config.connection_timeout}秒")
                return False
                
            except Exception as e:
                self._set_state(ConnectionState.FAILED)
                self.consecutive_failures += 1
                self._record_event("connect_error", {"error": str(e)})
                self.logger.error(f"连接失败: {e}")
                return False
    
    async def disconnect(self) -> bool:
        """断开连接"""
        async with self.connection_lock:
            if self.state == ConnectionState.DISCONNECTED:
                return True
            
            return await self._disconnect_internal()
    
    async def _disconnect_internal(self) -> bool:
        """内部断开连接方法"""
        try:
            # 停止健康检查
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
                self.health_check_task = None
            
            # 断开连接
            if self.ib.isConnected():
                self.ib.disconnect()
                self.logger.info("连接已断开")
            
            # 更新状态和统计
            self._set_state(ConnectionState.DISCONNECTED)
            self.stats['total_disconnections'] += 1
            
            if self.last_connection_time > 0:
                uptime = time.time() - self.last_connection_time
                self.stats['connection_uptime'] += uptime
                self._record_event("disconnect", {"uptime": uptime})
            
            # 执行断开回调
            await self._execute_callbacks(self.disconnection_callbacks)
            
            return True
            
        except Exception as e:
            self.logger.error(f"断开连接失败: {e}")
            return False
    
    async def reconnect(self, strategy: Optional[ReconnectStrategy] = None) -> bool:
        """智能重连"""
        if strategy is None:
            strategy = self.config.reconnect_strategy
        
        self._set_state(ConnectionState.RECONNECTING)
        self.logger.info(f"开始重连 (策略: {strategy.value}, 尝试次数: {self.reconnect_attempts + 1})")
        
        while self.reconnect_attempts < self.config.max_reconnect_attempts:
            self.reconnect_attempts += 1
            self.stats['total_reconnect_attempts'] += 1
            
            # 计算延迟时间
            delay = self._calculate_reconnect_delay(strategy)
            
            self.logger.info(f"重连尝试 {self.reconnect_attempts}/{self.config.max_reconnect_attempts}, "
                           f"等待 {delay:.1f}秒...")
            
            self._record_event("reconnect_attempt", {
                "attempt": self.reconnect_attempts,
                "strategy": strategy.value,
                "delay": delay
            })
            
            # 等待延迟
            await asyncio.sleep(delay)
            
            # 尝试连接
            if await self.connect(force=True):
                self.stats['successful_reconnects'] += 1
                self.logger.info(f"重连成功 (尝试次数: {self.reconnect_attempts})")
                return True
            
            self.logger.warning(f"重连失败 (尝试 {self.reconnect_attempts}/{self.config.max_reconnect_attempts})")
        
        # 重连失败
        self._set_state(ConnectionState.FAILED)
        self.logger.error(f"重连最终失败，已达最大尝试次数: {self.config.max_reconnect_attempts}")
        self._record_event("reconnect_failed", {
            "total_attempts": self.reconnect_attempts,
            "reason": "max_attempts_reached"
        })
        
        return False
    
    def _calculate_reconnect_delay(self, strategy: ReconnectStrategy) -> float:
        """计算重连延迟时间"""
        base_delay = self.config.base_reconnect_delay
        
        if strategy == ReconnectStrategy.FIXED_DELAY:
            return base_delay
        
        elif strategy == ReconnectStrategy.EXPONENTIAL_BACKOFF:
            # 指数退避: base * (2 ^ attempt) + jitter
            exponential_delay = base_delay * (2 ** min(self.reconnect_attempts - 1, 10))
            # 添加抖动以避免多客户端同时重连
            jitter = base_delay * 0.1 * (time.time() % 1)
            delay = exponential_delay + jitter
            
        elif strategy == ReconnectStrategy.LINEAR_BACKOFF:
            # 线性退避: base + (attempt * increment)
            increment = base_delay * 0.5
            delay = base_delay + (self.reconnect_attempts - 1) * increment
        
        else:
            delay = base_delay
        
        # 限制最大延迟
        return min(delay, self.config.max_reconnect_delay)
    
    async def _start_health_check(self):
        """启动连接健康检查"""
        if self.health_check_task:
            return
        
        async def health_monitor():
            while self.state == ConnectionState.CONNECTED:
                try:
                    await asyncio.sleep(self.config.health_check_interval)
                    
                    # 检查连接状态
                    if not self.ib.isConnected():
                        self.logger.warning("健康检查发现连接断开")
                        self._set_state(ConnectionState.DISCONNECTED)
                        
                        # 触发自动重连
                        if await self.reconnect():
                            self.logger.info("自动重连成功")
                        else:
                            self.logger.error("自动重连失败")
                        break
                    
                    # 可选：发送心跳包验证连接活性
                    # await self._send_heartbeat()
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.warning(f"健康检查异常: {e}")
        
        self.health_check_task = asyncio.create_task(health_monitor())
    
    async def _wait_for_connection(self, timeout: float = 30.0) -> bool:
        """等待连接完成"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.state == ConnectionState.CONNECTED:
                return True
            elif self.state in [ConnectionState.FAILED, ConnectionState.DISCONNECTED]:
                return False
            await asyncio.sleep(0.1)
        return False
    
    def _set_state(self, new_state: ConnectionState):
        """更新连接状态"""
        with self.state_lock:
            old_state = self.state
            self.state = new_state
            
            if old_state != new_state:
                self.logger.debug(f"连接状态变化: {old_state.value} -> {new_state.value}")
    
    def _record_event(self, event_type: str, details: Dict[str, Any] = None):
        """记录连接事件"""
        event = ConnectionEvent(
            timestamp=time.time(),
            event_type=event_type,
            state=self.state,
            details=details or {}
        )
        
        self.connection_events.append(event)
        
        # 限制事件历史长度
        if len(self.connection_events) > self.max_event_history:
            self.connection_events = self.connection_events[-self.max_event_history:]
    
    async def _execute_callbacks(self, callbacks: List[Callable]):
        """执行回调函数"""
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                self.logger.error(f"回调执行失败: {e}")
    
    def add_connection_callback(self, callback: Callable):
        """添加连接成功回调"""
        self.connection_callbacks.append(callback)
    
    def add_disconnection_callback(self, callback: Callable):
        """添加断开连接回调"""
        self.disconnection_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """移除回调"""
        self.connection_callbacks = [cb for cb in self.connection_callbacks if cb != callback]
        self.disconnection_callbacks = [cb for cb in self.disconnection_callbacks if cb != callback]
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.state == ConnectionState.CONNECTED and self.ib.isConnected()
    
    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        with self.state_lock:
            uptime = time.time() - self.last_connection_time if self.last_connection_time > 0 else 0
            
            return {
                'state': self.state.value,
                'connected': self.is_connected(),
                'host': self.config.host,
                'port': self.config.port,
                'client_id': self.config.client_id,
                'uptime': uptime,
                'reconnect_attempts': self.reconnect_attempts,
                'consecutive_failures': self.consecutive_failures,
                'last_connection_time': self.last_connection_time
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        with self.state_lock:
            total_uptime = self.stats['connection_uptime']
            if self.last_connection_time > 0 and self.state == ConnectionState.CONNECTED:
                total_uptime += time.time() - self.last_connection_time
            
            avg_duration = (total_uptime / self.stats['total_connections'] 
                           if self.stats['total_connections'] > 0 else 0)
            
            reconnect_success_rate = (self.stats['successful_reconnects'] / 
                                    max(self.stats['total_reconnect_attempts'], 1))
            
            recent_events = [
                e for e in self.connection_events 
                if time.time() - e.timestamp < 3600  # 最近1小时
            ]
            
            return {
                'connections': self.stats['total_connections'],
                'disconnections': self.stats['total_disconnections'],
                'reconnect_attempts': self.stats['total_reconnect_attempts'],
                'successful_reconnects': self.stats['successful_reconnects'],
                'reconnect_success_rate': reconnect_success_rate,
                'total_uptime': total_uptime,
                'average_connection_duration': avg_duration,
                'recent_events': len(recent_events),
                'current_state': self.state.value,
                'health_check_active': self.health_check_task is not None
            }
    
    def get_recent_events(self, limit: int = 20) -> List[ConnectionEvent]:
        """获取最近的连接事件"""
        return self.connection_events[-limit:] if limit > 0 else self.connection_events
    
    async def force_health_check(self) -> Dict[str, Any]:
        """强制执行健康检查"""
        check_result = {
            'timestamp': time.time(),
            'ib_connected': self.ib.isConnected(),
            'manager_state': self.state.value,
            'uptime': time.time() - self.last_connection_time if self.last_connection_time > 0 else 0
        }
        
        # 检查状态一致性
        ib_connected = self.ib.isConnected()
        manager_connected = self.state == ConnectionState.CONNECTED
        
        if ib_connected != manager_connected:
            self.logger.warning(f"连接状态不一致: IB={ib_connected}, Manager={manager_connected}")
            check_result['status_mismatch'] = True
            
            # 修正状态
            if ib_connected and not manager_connected:
                self._set_state(ConnectionState.CONNECTED)
                check_result['corrected_to'] = 'connected'
            elif not ib_connected and manager_connected:
                self._set_state(ConnectionState.DISCONNECTED)
                check_result['corrected_to'] = 'disconnected'
        else:
            check_result['status_mismatch'] = False
        
        return check_result
    
    async def shutdown(self):
        """关闭连接管理器"""
        self.logger.info("关闭连接管理器...")
        
        # 停止健康检查
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # 断开连接
        await self.disconnect()
        
        # 清理回调
        self.connection_callbacks.clear()
        self.disconnection_callbacks.clear()
        
        self.logger.info("连接管理器已关闭")

def create_connection_manager(ib_client, config_manager, 
                            logger: Optional[logging.Logger] = None) -> UnifiedConnectionManager:
    """创建连接管理器的便捷函数"""
    # 从配置管理器构建连接配置
    conn_config = ConnectionConfig(
        host=config_manager.get('connection.host', '127.0.0.1'),
        port=config_manager.get('connection.port', 4002),
        client_id=config_manager.get('connection.client_id', 3130),
        timeout=config_manager.get('connection.timeout', 20.0),
        max_reconnect_attempts=config_manager.get('connection.max_reconnect_attempts', 10),
        base_reconnect_delay=config_manager.get('connection.reconnect_delay', 5.0),
        account_id=config_manager.get('connection.account_id'),
        use_delayed_if_no_realtime=config_manager.get('connection.use_delayed_if_no_realtime', True)
    )
    
    return UnifiedConnectionManager(ib_client, conn_config, logger)
