#!/usr/bin/env python3
"""
=============================================================================
统一事件管理器 - 整合版本
=============================================================================
整合以下事件功能:
- 事件循环管理 (原 event_loop_manager.py)
- 事件总线系统 (整合自 event_system.py)
整合时间: 2025-08-20
=============================================================================

支持GUI应用的事件循环管理和事件发布订阅系统
"""

import asyncio
import threading
import logging
import queue
import time
import weakref
from typing import Optional, Any, Callable, Dict, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# =============================================================================
# 事件数据结构
# =============================================================================

@dataclass
class Event:
    """事件数据结构"""
    type: str
    data: Any
    timestamp: datetime
    source: Optional[str] = None

# =============================================================================
# 事件循环管理器 (从 event_loop_manager.py 整合)
# =============================================================================

class EventLoopManager:
    """事件循环管理器"""
    
    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.is_running = False
        self.task_queue = queue.Queue()
        self._startup_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()
        
    def start(self) -> bool:
        """启动事件循环"""
        with self._lock:
            if self.is_running:
                logger.warning("事件循环已经在运行")
                return True
                
            try:
                # 重置事件
                self._startup_event.clear()
                self._shutdown_event.clear()
                
                # 创建新的事件循环
                self.loop = asyncio.new_event_loop()
                
                # 在新线程中运行事件循环
                self.thread = threading.Thread(target=self._run_loop, daemon=True)
                self.thread.start()
                
                # 等待线程实际启动（最多等待5秒）
                if self._startup_event.wait(timeout=5.0):
                    self.is_running = True
                    logger.info("事件循环管理器启动成功")
                    return True
                else:
                    logger.error("事件循环启动超时")
                    self._cleanup_failed_start()
                    return False
                    
            except Exception as e:
                logger.error(f"启动事件循环失败: {e}")
                self._cleanup_failed_start()
                return False
    
    def _cleanup_failed_start(self):
        """清理失败的启动状态"""
        self.is_running = False
        if self.loop and not self.loop.is_closed():
            self.loop.close()
        self.loop = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None
            
    def _run_loop(self):
        """在线程中运行事件循环"""
        try:
            asyncio.set_event_loop(self.loop)
            # 通知启动成功
            self._startup_event.set()
            logger.debug("事件循环线程启动完成")
            
            # 运行事件循环
            self.loop.run_forever()
            
        except Exception as e:
            logger.error(f"事件循环运行错误: {e}")
        finally:
            # 清理状态
            with self._lock:
                self.is_running = False
            self._shutdown_event.set()
            logger.debug("事件循环线程结束")
            
    def stop(self):
        """停止事件循环"""
        with self._lock:
            if not self.is_running or not self.loop:
                return
                
            try:
                # 停止事件循环
                self.loop.call_soon_threadsafe(self.loop.stop)
                
                # 等待线程结束
                if self.thread and self.thread.is_alive():
                    if self._shutdown_event.wait(timeout=10.0):
                        logger.info("事件循环管理器停止成功")
                    else:
                        logger.warning("事件循环停止超时")
                        
                # 关闭事件循环
                if not self.loop.is_closed():
                    self.loop.close()
                    
                self.loop = None
                self.thread = None
                
            except Exception as e:
                logger.error(f"停止事件循环失败: {e}")
                
    def run_coroutine_threadsafe(self, coro) -> asyncio.Future:
        """在事件循环中线程安全地运行协程"""
        if not self.is_running or not self.loop:
            raise RuntimeError("事件循环未运行")
        
        return asyncio.run_coroutine_threadsafe(coro, self.loop)
    
    def call_soon_threadsafe(self, callback, *args):
        """在事件循环中线程安全地调用函数"""
        if not self.is_running or not self.loop:
            raise RuntimeError("事件循环未运行")
        
        return self.loop.call_soon_threadsafe(callback, *args)
    
    def submit_coroutine_nowait(self, coro) -> str:
        """立即提交协程到事件循环，返回任务ID"""
        import uuid
        
        if not self.is_running or not self.loop:
            raise RuntimeError("事件循环未运行")
        
        task_id = str(uuid.uuid4())
        
        def create_task():
            task = self.loop.create_task(coro)
            task.set_name(f"nowait_task_{task_id}")
            return task
        
        # 使用call_soon_threadsafe立即提交任务
        self.loop.call_soon_threadsafe(create_task)
        return task_id
    
    def submit_coroutine(self, coro) -> asyncio.Future:
        """提交协程到事件循环并返回Future"""
        if not self.is_running or not self.loop:
            raise RuntimeError("事件循环未运行")
        
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

# =============================================================================
# 事件总线系统 (从 event_system.py 整合)
# =============================================================================

class EventBus:
    """事件总线 - 管理事件的发布和订阅"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._event_queue = queue.Queue()
        self._running = False
        self._worker_thread = None
        
    def subscribe(self, event_type: str, callback: Callable):
        """订阅事件"""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
            logger.debug(f"订阅事件: {event_type}")
            
    def unsubscribe(self, event_type: str, callback: Callable):
        """取消订阅"""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    logger.debug(f"取消订阅事件: {event_type}")
                except ValueError:
                    logger.warning(f"尝试取消不存在的订阅: {event_type}")
                    
    def publish(self, event_type: str, data: Any, source: Optional[str] = None):
        """发布事件"""
        event = Event(
            type=event_type,
            data=data,
            timestamp=datetime.now(),
            source=source
        )
        
        try:
            self._event_queue.put(event, block=False)
            logger.debug(f"发布事件: {event_type}")
        except queue.Full:
            logger.warning(f"事件队列已满，丢弃事件: {event_type}")
        
    def start(self):
        """启动事件处理"""
        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(target=self._process_events, daemon=True)
            self._worker_thread.start()
            logger.info("事件总线已启动")
            
    def stop(self):
        """停止事件处理"""
        if self._running:
            self._running = False
            try:
                self._event_queue.put(None, timeout=1.0)  # 停止信号
            except queue.Full:
                logger.warning("无法发送停止信号，队列已满")
                
            if self._worker_thread:
                self._worker_thread.join(timeout=5.0)
                if self._worker_thread.is_alive():
                    logger.warning("事件处理线程未能正常停止")
                    
            logger.info("事件总线已停止")
            
    def _process_events(self):
        """处理事件的工作线程"""
        while self._running:
            try:
                event = self._event_queue.get(timeout=1.0)
                if event is None:  # 停止信号
                    break
                    
                # 获取订阅者副本以避免长时间锁定
                with self._lock:
                    subscribers = self._subscribers.get(event.type, []).copy()
                
                # 调用订阅者回调
                for callback in subscribers:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"事件回调执行失败 {event.type}: {e}")
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"事件处理错误: {e}")
        
        logger.debug("事件处理线程退出")

class GUIEventAdapter:
    """GUI事件适配器 - 将业务事件转换为GUI更新"""
    
    def __init__(self, gui_app, event_bus: EventBus):
        self.gui_app = weakref.ref(gui_app)  # 使用弱引用避免循环引用
        self.event_bus = event_bus
        self._setup_subscriptions()
        
    def _setup_subscriptions(self):
        """设置事件订阅"""
        # 交易相关事件
        self.event_bus.subscribe("trade_executed", self._on_trade_executed)
        self.event_bus.subscribe("order_status_changed", self._on_order_status_changed)
        self.event_bus.subscribe("position_updated", self._on_position_updated)
        
        # 系统状态事件
        self.event_bus.subscribe("connection_status", self._on_connection_status)
        self.event_bus.subscribe("error_occurred", self._on_error_occurred)
        self.event_bus.subscribe("log_message", self._on_log_message)
        
        # 数据更新事件
        self.event_bus.subscribe("market_data_updated", self._on_market_data_updated)
        self.event_bus.subscribe("portfolio_updated", self._on_portfolio_updated)
        
    def _on_trade_executed(self, event: Event):
        """处理交易执行事件"""
        gui = self.gui_app()
        if gui:
            try:
                gui.after(0, lambda: self._update_trade_status(event.data))
            except Exception as e:
                logger.error(f"更新交易状态失败: {e}")
                
    def _on_order_status_changed(self, event: Event):
        """处理订单状态变化事件"""
        gui = self.gui_app()
        if gui:
            try:
                gui.after(0, lambda: self._update_order_status(event.data))
            except Exception as e:
                logger.error(f"更新订单状态失败: {e}")
                
    def _on_position_updated(self, event: Event):
        """处理持仓更新事件"""
        gui = self.gui_app()
        if gui:
            try:
                gui.after(0, lambda: self._update_positions(event.data))
            except Exception as e:
                logger.error(f"更新持仓失败: {e}")
                
    def _on_connection_status(self, event: Event):
        """处理连接状态事件"""
        gui = self.gui_app()
        if gui:
            try:
                gui.after(0, lambda: self._update_connection_status(event.data))
            except Exception as e:
                logger.error(f"更新连接状态失败: {e}")
                
    def _on_error_occurred(self, event: Event):
        """处理错误事件"""
        gui = self.gui_app()
        if gui:
            try:
                gui.after(0, lambda: self._show_error(event.data))
            except Exception as e:
                logger.error(f"显示错误失败: {e}")
                
    def _on_log_message(self, event: Event):
        """处理日志消息事件"""
        gui = self.gui_app()
        if gui:
            try:
                gui.after(0, lambda: self._append_log(event.data))
            except Exception as e:
                logger.error(f"添加日志失败: {e}")
                
    def _on_market_data_updated(self, event: Event):
        """处理市场数据更新事件"""
        gui = self.gui_app()
        if gui:
            try:
                gui.after(0, lambda: self._update_market_data(event.data))
            except Exception as e:
                logger.error(f"更新市场数据失败: {e}")
                
    def _on_portfolio_updated(self, event: Event):
        """处理投资组合更新事件"""
        gui = self.gui_app()
        if gui:
            try:
                gui.after(0, lambda: self._update_portfolio(event.data))
            except Exception as e:
                logger.error(f"更新投资组合失败: {e}")
    
    # GUI更新方法的安全实现
    def _update_trade_status(self, data):
        """更新交易状态显示"""
        gui = self.gui_app()
        if gui and hasattr(gui, 'log'):
            gui.log(f"交易执行: {data}")
            
    def _update_order_status(self, data):
        """更新订单状态显示"""
        gui = self.gui_app()
        if gui and hasattr(gui, 'log'):
            gui.log(f"订单状态: {data}")
            
    def _update_positions(self, data):
        """更新持仓显示"""
        gui = self.gui_app()
        if gui and hasattr(gui, 'log'):
            gui.log(f"持仓更新: {data}")
            
    def _update_connection_status(self, data):
        """更新连接状态显示"""
        gui = self.gui_app()
        if gui and hasattr(gui, 'log'):
            status = "已连接" if data.get('connected', False) else "未连接"
            gui.log(f"IBKR连接状态: {status}")
            
    def _show_error(self, data):
        """显示错误信息"""
        gui = self.gui_app()
        if gui and hasattr(gui, 'log'):
            gui.log(f"错误: {data}")
            
    def _append_log(self, data):
        """添加日志消息"""
        gui = self.gui_app()
        if gui and hasattr(gui, 'log'):
            gui.log(str(data))
            
    def _update_market_data(self, data):
        """更新市场数据显示"""
        gui = self.gui_app()
        if gui and hasattr(gui, 'log'):
            gui.log(f"市场数据更新: {data.get('symbol', 'Unknown')}")
            
    def _update_portfolio(self, data):
        """更新投资组合显示"""
        gui = self.gui_app()
        if gui and hasattr(gui, 'log'):
            gui.log(f"投资组合更新: {len(data.get('positions', []))} 个持仓")

# =============================================================================
# 统一事件管理器主类
# =============================================================================

class UnifiedEventManager:
    """统一事件管理器 - 整合事件循环和事件总线"""
    
    def __init__(self):
        self.loop_manager = EventLoopManager()
        self.event_bus = EventBus()
        self.gui_adapter: Optional[GUIEventAdapter] = None
        
        logger.info("统一事件管理器初始化完成")
    
    def start(self):
        """启动所有事件组件"""
        logger.info("启动统一事件管理器...")
        
        # 启动事件循环管理器
        loop_started = self.loop_manager.start()
        
        # 启动事件总线
        self.event_bus.start()
        
        if loop_started:
            logger.info("统一事件管理器启动完成")
        else:
            logger.warning("事件循环启动失败，但事件总线已启动")
        
        return loop_started
    
    def stop(self):
        """停止所有事件组件"""
        logger.info("停止统一事件管理器...")
        
        # 停止事件总线
        self.event_bus.stop()
        
        # 停止事件循环管理器
        self.loop_manager.stop()
        
        logger.info("统一事件管理器停止完成")
    
    def setup_gui_adapter(self, gui_app):
        """设置GUI适配器"""
        self.gui_adapter = GUIEventAdapter(gui_app, self.event_bus)
        logger.info("GUI事件适配器设置完成")
    
    def publish_event(self, event_type: str, data: Any, source: Optional[str] = None):
        """发布事件"""
        self.event_bus.publish(event_type, data, source)
    
    def subscribe_event(self, event_type: str, callback: Callable):
        """订阅事件"""
        self.event_bus.subscribe(event_type, callback)
    
    def unsubscribe_event(self, event_type: str, callback: Callable):
        """取消订阅事件"""
        self.event_bus.unsubscribe(event_type, callback)
    
    def run_coroutine_threadsafe(self, coro) -> asyncio.Future:
        """在事件循环中线程安全地运行协程"""
        return self.loop_manager.run_coroutine_threadsafe(coro)
    
    def call_soon_threadsafe(self, callback, *args):
        """在事件循环中线程安全地调用函数"""
        return self.loop_manager.call_soon_threadsafe(callback, *args)
    
    def get_event_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """获取事件循环"""
        return self.loop_manager.loop
    
    def is_running(self) -> bool:
        """检查事件管理器是否运行中"""
        return self.loop_manager.is_running and self.event_bus._running

# =============================================================================
# 全局实例和工厂函数
# =============================================================================

# 全局统一事件管理器实例
_unified_event_manager: Optional[UnifiedEventManager] = None
_global_event_bus: Optional[EventBus] = None
_global_event_loop_manager: Optional[EventLoopManager] = None

def get_unified_event_manager() -> UnifiedEventManager:
    """获取统一事件管理器单例"""
    global _unified_event_manager
    if _unified_event_manager is None:
        _unified_event_manager = UnifiedEventManager()
    return _unified_event_manager

def get_event_bus() -> EventBus:
    """获取事件总线单例 - 向后兼容"""
    global _global_event_bus
    if _global_event_bus is None:
        event_manager = get_unified_event_manager()
        _global_event_bus = event_manager.event_bus
    return _global_event_bus

def get_event_loop_manager() -> EventLoopManager:
    """获取事件循环管理器单例 - 向后兼容"""
    global _global_event_loop_manager
    if _global_event_loop_manager is None:
        event_manager = get_unified_event_manager()
        _global_event_loop_manager = event_manager.loop_manager
    return _global_event_loop_manager

def init_unified_event_system() -> UnifiedEventManager:
    """初始化并启动统一事件系统"""
    event_manager = get_unified_event_manager()
    event_manager.start()
    return event_manager

def shutdown_unified_event_system():
    """关闭统一事件系统"""
    global _unified_event_manager, _global_event_bus, _global_event_loop_manager
    
    if _unified_event_manager:
        _unified_event_manager.stop()
        _unified_event_manager = None
        _global_event_bus = None
        _global_event_loop_manager = None
        logger.info("统一事件系统已关闭")

# 向后兼容的便捷函数
def publish_event(event_type: str, data: Any, source: Optional[str] = None):
    """发布事件的便捷函数 - 向后兼容"""
    bus = get_event_bus()
    bus.publish(event_type, data, source)

def subscribe_event(event_type: str, callback: Callable):
    """订阅事件的便捷函数 - 向后兼容"""
    bus = get_event_bus()
    bus.subscribe(event_type, callback)

def shutdown_event_bus():
    """关闭事件总线 - 向后兼容"""
    # 统一事件系统会处理关闭
    pass