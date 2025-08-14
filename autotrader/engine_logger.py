#!/usr/bin/env python3
"""
Engine日志适配器
willEngine日志调use重定toto事件系统，避免跨线程GUI调use
"""

import logging
from typing import Optional, Any

class EngineLoggerAdapter(logging.LoggerAdapter):
    """Engine日志适配器，will日志通过事件系统发送toGUI"""
    
    def __init__(self, logger: logging.Logger, source: str = "engine"):
        super().__init__(logger, {})
        self.source = source
        self._event_bus = None
    
    def _get_event_bus(self):
        """延迟retrieval事件总线（避免循环导入）"""
        if self._event_bus is None:
            try:
                from .event_system import get_event_bus
                self._event_bus = get_event_bus()
            except ImportError:
                pass
        return self._event_bus
    
    def _emit_to_gui(self, level: str, message: str):
        """发送日志toGUI"""
        event_bus = self._get_event_bus()
        if event_bus:
            try:
                from .event_system import EventType
                
                event_bus.publish(
                    EventType.ENGINE_LOG,
                    {
                        'message': message,
                        'level': level
                    },
                    source=self.source,
                    priority=2
                )
            except Exception:
                # 降级to标准日志
                pass
    
    def debug(self, msg: Any, *args, **kwargs) -> None:
        """调试日志"""
        message = str(msg) % args if args else str(msg)
        super().debug(message, **kwargs)
        self._emit_to_gui('debug', message)
    
    def info(self, msg: Any, *args, **kwargs) -> None:
        """信息日志"""
        message = str(msg) % args if args else str(msg)
        super().info(message, **kwargs)
        self._emit_to_gui('info', message)
    
    def warning(self, msg: Any, *args, **kwargs) -> None:
        """警告日志"""
        message = str(msg) % args if args else str(msg)
        super().warning(message, **kwargs)
        self._emit_to_gui('warning', message)
    
    def error(self, msg: Any, *args, **kwargs) -> None:
        """错误日志"""
        message = str(msg) % args if args else str(msg)
        super().error(message, **kwargs)
        self._emit_to_gui('error', message)
    
    def critical(self, msg: Any, *args, **kwargs) -> None:
        """严重错误日志"""
        message = str(msg) % args if args else str(msg)
        super().critical(message, **kwargs)
        self._emit_to_gui('critical', message)

def create_engine_logger(name: str, source: str = "engine") -> EngineLoggerAdapter:
    """创建Engine日志器"""
    base_logger = logging.getLogger(name)
    return EngineLoggerAdapter(base_logger, source)

def update_status(status: str, color: str = 'black', source: str = 'engine'):
    """updates状态toGUI"""
    try:
        from .event_system import get_event_bus, EventType
        
        event_bus = get_event_bus()
        event_bus.publish(
            EventType.ENGINE_STATUS,
            {
                'status': status,
                'color': color
            },
            source=source,
            priority=3
        )
    except Exception:
        # 静默failed，避免影响核心功能
        pass
