#!/usr/bin/env python3
"""
事件系统 - 解耦GUI和Engine
通过事件发布订阅模式避免跨线程直接调用
"""

import threading
import logging
import weakref
import uuid
from typing import Dict, List, Any, Callable, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import time
import queue

class EventType(Enum):
    """事件类型枚举"""
    # 系统事件
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    
    # 连接事件
    CONNECTION_STATUS = "connection.status"
    CONNECTION_ERROR = "connection.error"
    
    # 交易事件
    TRADE_SIGNAL = "trade.signal"
    ORDER_STATUS = "order.status"
    POSITION_UPDATE = "position.update"
    ACCOUNT_UPDATE = "account.update"
    
    # 引擎事件
    ENGINE_STATUS = "engine.status"
    ENGINE_LOG = "engine.log"
    ENGINE_PROGRESS = "engine.progress"
    
    # GUI事件
    GUI_UPDATE = "gui.update"
    GUI_LOG = "gui.log"
    GUI_NOTIFICATION = "gui.notification"
    
    # 数据事件
    DATA_UPDATE = "data.update"
    MARKET_DATA = "market.data"
    
    # 风险事件
    RISK_WARNING = "risk.warning"
    RISK_LIMIT = "risk.limit"

@dataclass
class Event:
    """事件数据结构"""
    type: EventType
    data: Dict[str, Any]
    source: str
    timestamp: float
    event_id: str
    priority: int = 1  # 1=低, 2=中, 3=高, 4=紧急
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()
    
    def __lt__(self, other):
        """比较方法，用于优先队列排序"""
        if isinstance(other, Event):
            # 首先按优先级排序（高优先级在前）
            if self.priority != other.priority:
                return self.priority > other.priority
            # 然后按时间戳排序（早的在前）
            return self.timestamp < other.timestamp
        return NotImplemented
    
    def __eq__(self, other):
        """相等比较"""
        if isinstance(other, Event):
            return self.event_id == other.event_id
        return NotImplemented
    
    def __hash__(self):
        """哈希方法"""
        return hash(self.event_id)

class EventBus:
    """线程安全的事件总线"""
    
    def __init__(self, max_history: int = 1000):
        self.logger = logging.getLogger("EventBus")
        
        # 线程安全
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        
        # 订阅者管理
        self._subscribers: Dict[EventType, Set[weakref.WeakMethod]] = defaultdict(set)
        self._sync_subscribers: Dict[EventType, Set[weakref.WeakMethod]] = defaultdict(set)
        
        # 事件队列（异步处理）
        self._event_queue: queue.PriorityQueue = queue.PriorityQueue()
        
        # 事件历史
        self._event_history: deque = deque(maxlen=max_history)
        
        # 统计信息
        self._stats = {
            'events_published': 0,
            'events_processed': 0,
            'subscribers_count': 0,
            'processing_errors': 0
        }
        
        # 事件处理线程
        self._processor_thread: Optional[threading.Thread] = None
        self._running = False
        
        # 性能监控
        self._performance_stats = defaultdict(list)
        
    def start(self):
        """启动事件总线"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._stop_event.clear()
            
            # 启动事件处理线程
            self._processor_thread = threading.Thread(
                target=self._process_events,
                name="EventBus-Processor",
                daemon=True
            )
            self._processor_thread.start()
            
            self.logger.info("事件总线已启动")
    
    def stop(self, timeout: float = 5.0):
        """停止事件总线"""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            self._stop_event.set()
            
            # 等待处理线程结束
            if self._processor_thread and self._processor_thread.is_alive():
                self._processor_thread.join(timeout)
            
            self.logger.info("事件总线已停止")
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None], sync: bool = False):
        """订阅事件
        
        Args:
            event_type: 事件类型
            callback: 回调函数
            sync: 是否同步处理（True=立即处理，False=异步队列处理）
        """
        with self._lock:
            try:
                # 尝试创建弱引用
                try:
                    weak_callback = weakref.WeakMethod(callback)
                except TypeError:
                    # 对于函数（非方法），直接存储
                    weak_callback = callback
                
                if sync:
                    self._sync_subscribers[event_type].add(weak_callback)
                else:
                    self._subscribers[event_type].add(weak_callback)
                
                self._stats['subscribers_count'] += 1
                
                self.logger.debug(f"订阅事件: {event_type.value} ({'同步' if sync else '异步'})")
                
            except Exception as e:
                self.logger.error(f"订阅失败: {e}")
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """取消订阅事件"""
        with self._lock:
            try:
                weak_callback = weakref.WeakMethod(callback)
                
                # 从两个集合中移除
                removed = False
                if weak_callback in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(weak_callback)
                    removed = True
                
                if weak_callback in self._sync_subscribers[event_type]:
                    self._sync_subscribers[event_type].remove(weak_callback)
                    removed = True
                
                if removed:
                    self._stats['subscribers_count'] -= 1
                    self.logger.debug(f"取消订阅: {event_type.value}")
                
            except (TypeError, KeyError):
                pass
    
    def publish(self, event_type: EventType, data: Dict[str, Any], source: str = "unknown", priority: int = 1):
        """发布事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
            source: 事件源
            priority: 优先级（1-4，数字越大优先级越高）
        """
        if not self._running:
            return
        
        # 创建事件对象
        event = Event(
            type=event_type,
            data=data,
            source=source,
            timestamp=time.time(),
            event_id=str(uuid.uuid4()),
            priority=priority
        )
        
        with self._lock:
            # 记录到历史
            self._event_history.append(event)
            self._stats['events_published'] += 1
            
            # 同步处理
            self._process_sync_subscribers(event)
            
            # 异步处理（添加到队列）
            try:
                # 直接放入事件对象，因为Event已经实现了比较方法
                self._event_queue.put(event)
            except Exception as e:
                self.logger.error(f"添加事件到队列失败: {e}")
                self._stats['processing_errors'] += 1
    
    def _process_sync_subscribers(self, event: Event):
        """同步处理订阅者（在发布线程中立即执行）"""
        subscribers = self._sync_subscribers.get(event.type, set()).copy()
        
        # 清理失效的弱引用
        invalid_refs = set()
        
        for weak_callback in subscribers:
            try:
                # 检查是否是弱引用
                if hasattr(weak_callback, '__call__') and not hasattr(weak_callback, '__func__'):
                    # 直接的函数对象
                    callback = weak_callback
                else:
                    # 弱引用对象
                    callback = weak_callback()
                    if callback is None:
                        invalid_refs.add(weak_callback)
                        continue
                
                start_time = time.time()
                callback(event)
                
                # 记录性能
                execution_time = time.time() - start_time
                self._performance_stats[f"sync_{event.type.value}"].append(execution_time)
                
                # 限制性能统计历史
                if len(self._performance_stats[f"sync_{event.type.value}"]) > 100:
                    self._performance_stats[f"sync_{event.type.value}"].pop(0)
                
            except Exception as e:
                self.logger.error(f"同步事件处理失败 {event.type.value}: {e}")
                self._stats['processing_errors'] += 1
        
        # 清理失效的弱引用
        if invalid_refs:
            self._sync_subscribers[event.type] -= invalid_refs
    
    def _process_events(self):
        """事件处理线程主循环"""
        self.logger.info("事件处理线程启动")
        
        while self._running and not self._stop_event.is_set():
            try:
                # 获取事件（带超时）
                try:
                    event = self._event_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # 处理事件
                self._dispatch_event(event)
                self._stats['events_processed'] += 1
                
            except Exception as e:
                self.logger.error(f"事件处理线程异常: {e}")
                self._stats['processing_errors'] += 1
        
        self.logger.info("事件处理线程结束")
    
    def _dispatch_event(self, event: Event):
        """分发事件给异步订阅者"""
        subscribers = self._subscribers.get(event.type, set()).copy()
        
        # 清理失效的弱引用
        invalid_refs = set()
        
        for weak_callback in subscribers:
            try:
                # 检查是否是弱引用
                if hasattr(weak_callback, '__call__') and not hasattr(weak_callback, '__func__'):
                    # 直接的函数对象
                    callback = weak_callback
                else:
                    # 弱引用对象
                    callback = weak_callback()
                    if callback is None:
                        invalid_refs.add(weak_callback)
                        continue
                
                start_time = time.time()
                callback(event)
                
                # 记录性能
                execution_time = time.time() - start_time
                self._performance_stats[f"async_{event.type.value}"].append(execution_time)
                
                # 限制性能统计历史
                if len(self._performance_stats[f"async_{event.type.value}"]) > 100:
                    self._performance_stats[f"async_{event.type.value}"].pop(0)
                
            except Exception as e:
                self.logger.error(f"异步事件处理失败 {event.type.value}: {e}")
                self._stats['processing_errors'] += 1
        
        # 清理失效的弱引用
        if invalid_refs:
            with self._lock:
                self._subscribers[event.type] -= invalid_refs
    
    def get_event_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """获取事件历史"""
        with self._lock:
            if event_type:
                filtered = [e for e in self._event_history if e.type == event_type]
                return list(filtered)[-limit:]
            else:
                return list(self._event_history)[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            # 计算性能统计
            performance_summary = {}
            for key, times in self._performance_stats.items():
                if times:
                    performance_summary[key] = {
                        'count': len(times),
                        'avg_ms': sum(times) / len(times) * 1000,
                        'max_ms': max(times) * 1000,
                        'min_ms': min(times) * 1000
                    }
            
            return {
                'running': self._running,
                'queue_size': self._event_queue.qsize(),
                'history_size': len(self._event_history),
                'performance': performance_summary,
                **self._stats
            }
    
    def clear_history(self):
        """清空事件历史"""
        with self._lock:
            self._event_history.clear()
            self.logger.info("事件历史已清空")


# GUI事件适配器
class GUIEventAdapter:
    """GUI事件适配器，将Engine事件转换为GUI更新"""
    
    def __init__(self, gui_instance, event_bus: EventBus):
        self.gui = weakref.ref(gui_instance)
        self.event_bus = event_bus
        self.logger = logging.getLogger("GUIEventAdapter")
        
        # 订阅相关事件
        self._subscribe_events()
    
    def _subscribe_events(self):
        """订阅GUI相关事件"""
        # 日志事件（同步处理，确保及时显示）
        self.event_bus.subscribe(EventType.ENGINE_LOG, self._handle_engine_log, sync=True)
        self.event_bus.subscribe(EventType.GUI_LOG, self._handle_gui_log, sync=True)
        
        # 状态更新（异步处理）
        self.event_bus.subscribe(EventType.ENGINE_STATUS, self._handle_engine_status)
        self.event_bus.subscribe(EventType.CONNECTION_STATUS, self._handle_connection_status)
        self.event_bus.subscribe(EventType.ORDER_STATUS, self._handle_order_status)
        
        # 进度更新
        self.event_bus.subscribe(EventType.ENGINE_PROGRESS, self._handle_progress)
        
        # 通知
        self.event_bus.subscribe(EventType.GUI_NOTIFICATION, self._handle_notification)
    
    def _handle_engine_log(self, event: Event):
        """处理引擎日志事件"""
        gui = self.gui()
        if gui is None:
            return
        
        message = event.data.get('message', '')
        level = event.data.get('level', 'info')
        
        try:
            # 使用after确保在主线程中执行
            gui.after(0, lambda msg=message: gui.log(msg))
        except Exception as e:
            self.logger.error(f"处理引擎日志失败: {e}")
    
    def _handle_gui_log(self, event: Event):
        """处理GUI日志事件"""
        gui = self.gui()
        if gui is None:
            return
        
        message = event.data.get('message', '')
        
        try:
            gui.after(0, lambda msg=message: gui.log(msg))
        except Exception as e:
            self.logger.error(f"处理GUI日志失败: {e}")
    
    def _handle_engine_status(self, event: Event):
        """处理引擎状态事件"""
        gui = self.gui()
        if gui is None:
            return
        
        status = event.data.get('status', '')
        color = event.data.get('color', 'black')
        
        try:
            gui.after(0, lambda s=status, c=color: gui._update_signal_status(s, c))
        except Exception as e:
            self.logger.error(f"处理引擎状态失败: {e}")
    
    def _handle_connection_status(self, event: Event):
        """处理连接状态事件"""
        gui = self.gui()
        if gui is None:
            return
        
        connected = event.data.get('connected', False)
        message = event.data.get('message', '')
        
        try:
            if message:
                gui.after(0, lambda msg=message: gui.log(msg))
        except Exception as e:
            self.logger.error(f"处理连接状态失败: {e}")
    
    def _handle_order_status(self, event: Event):
        """处理订单状态事件"""
        gui = self.gui()
        if gui is None:
            return
        
        order_info = event.data.get('order_info', {})
        status = event.data.get('status', '')
        
        try:
            message = f"订单状态更新: {status}"
            gui.after(0, lambda msg=message: gui.log(msg))
        except Exception as e:
            self.logger.error(f"处理订单状态失败: {e}")
    
    def _handle_progress(self, event: Event):
        """处理进度事件"""
        gui = self.gui()
        if gui is None:
            return
        
        progress = event.data.get('progress', 0)
        message = event.data.get('message', '')
        
        try:
            if message:
                gui.after(0, lambda msg=f"进度: {message} ({progress}%)": gui.log(msg))
        except Exception as e:
            self.logger.error(f"处理进度失败: {e}")
    
    def _handle_notification(self, event: Event):
        """处理通知事件"""
        gui = self.gui()
        if gui is None:
            return
        
        message = event.data.get('message', '')
        notification_type = event.data.get('type', 'info')
        
        try:
            gui.after(0, lambda msg=f"[{notification_type.upper()}] {message}": gui.log(msg))
        except Exception as e:
            self.logger.error(f"处理通知失败: {e}")


# 全局事件总线
_global_event_bus: Optional[EventBus] = None

def get_event_bus() -> EventBus:
    """获取全局事件总线"""
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

# 便捷函数
def log_to_gui(message: str, level: str = 'info', source: str = 'system'):
    """发送日志到GUI"""
    event_bus = get_event_bus()
    event_bus.publish(
        EventType.GUI_LOG,
        {'message': message, 'level': level},
        source=source,
        priority=2
    )

def log_from_engine(message: str, level: str = 'info', source: str = 'engine'):
    """从Engine发送日志"""
    event_bus = get_event_bus()
    event_bus.publish(
        EventType.ENGINE_LOG,
        {'message': message, 'level': level},
        source=source,
        priority=2
    )

def update_engine_status(status: str, color: str = 'black', source: str = 'engine'):
    """更新引擎状态"""
    event_bus = get_event_bus()
    event_bus.publish(
        EventType.ENGINE_STATUS,
        {'status': status, 'color': color},
        source=source,
        priority=3
    )
