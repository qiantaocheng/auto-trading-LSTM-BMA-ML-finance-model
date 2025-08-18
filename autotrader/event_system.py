#!/usr/bin/env python3
"""
事件系统模块 - GUI和交易核心的事件通信
"""

import logging
import threading
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import queue
import weakref

logger = logging.getLogger(__name__)

@dataclass
class Event:
    """事件数据结构"""
    type: str
    data: Any
    timestamp: datetime
    source: Optional[str] = None

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
            
    def unsubscribe(self, event_type: str, callback: Callable):
        """取消订阅"""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                except ValueError:
                    pass
                    
    def publish(self, event_type: str, data: Any, source: Optional[str] = None):
        """发布事件"""
        event = Event(
            type=event_type,
            data=data,
            timestamp=datetime.now(),
            source=source
        )
        self._event_queue.put(event)
        
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
            self._event_queue.put(None)  # 停止信号
            if self._worker_thread:
                self._worker_thread.join(timeout=5.0)
            logger.info("事件总线已停止")
            
    def _process_events(self):
        """处理事件的工作线程"""
        while self._running:
            try:
                event = self._event_queue.get(timeout=1.0)
                if event is None:  # 停止信号
                    break
                    
                # 获取订阅者副本以避免锁定
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

# 全局事件总线实例
_global_event_bus: Optional[EventBus] = None

def get_event_bus() -> EventBus:
    """获取全局事件总线实例"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
        _global_event_bus.start()
    return _global_event_bus

def shutdown_event_bus():
    """关闭全局事件总线"""
    global _global_event_bus
    if _global_event_bus:
        _global_event_bus.stop()
        _global_event_bus = None
        logger.info("全局事件总线已关闭")

# 便捷函数
def publish_event(event_type: str, data: Any, source: Optional[str] = None):
    """发布事件的便捷函数"""
    bus = get_event_bus()
    bus.publish(event_type, data, source)

def subscribe_event(event_type: str, callback: Callable):
    """订阅事件的便捷函数"""
    bus = get_event_bus()
    bus.subscribe(event_type, callback)