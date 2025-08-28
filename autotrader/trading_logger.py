#!/usr/bin/env python3
"""
Engine Logger - 统一引擎日志系统
为Engine组件提供统一的日志记录功能
"""

import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime
import threading


class EngineLogger:
    """引擎专用日志记录器"""
    
    def __init__(self, name: str, component: str = "engine"):
        self.name = name
        self.component = component
        self.logger = logging.getLogger(f"{component}.{name}")
        self._setup_logger()
        
    def _setup_logger(self):
        """设置日志记录器"""
        if not self.logger.handlers:
            # 创建控制台处理器
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # 创建格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            
            # 添加处理器
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """记录错误日志"""
        self.logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        self.logger.debug(message, **kwargs)


# 全局日志记录器缓存
_logger_cache: Dict[str, EngineLogger] = {}
_cache_lock = threading.Lock()


def create_engine_logger(name: str, component: str = "engine") -> EngineLogger:
    """
    创建引擎日志记录器
    
    Args:
        name: 日志记录器名称
        component: 组件名称
        
    Returns:
        EngineLogger实例
    """
    cache_key = f"{component}.{name}"
    
    with _cache_lock:
        if cache_key not in _logger_cache:
            _logger_cache[cache_key] = EngineLogger(name, component)
        return _logger_cache[cache_key]


def get_engine_logger(name: str = "Engine") -> EngineLogger:
    """获取默认引擎日志记录器"""
    return create_engine_logger(name, "engine")


# 兼容性函数
def setup_engine_logging():
    """设置引擎日志系统（兼容性函数）"""
    return get_engine_logger()