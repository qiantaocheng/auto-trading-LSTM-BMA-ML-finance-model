#!/usr/bin/env python3
"""
Compatibility Layer - 兼容性层
确保新的组件架构与旧代码完全兼容
"""

import logging
import warnings
from typing import Optional, Any, Dict
import sys
import os

logger = logging.getLogger(__name__)

# 导入新的组件
try:
    from .master_config_manager import get_master_config_manager
    from .lazy_component_loader import get_component_loader
    from .unified_monitoring_interface import get_monitoring_interface
    from .real_risk_balancer import get_risk_balancer_adapter
except ImportError as e:
    logger.warning(f"Failed to import new components: {e}")

class CompatibilityWrapper:
    """兼容性包装器，确保旧代码正常工作"""
    
    def __init__(self):
        self.logger = logging.getLogger("CompatibilityWrapper")
        self.component_loader = None
        self._old_components = {}
        
        try:
            self.component_loader = get_component_loader()
        except Exception as e:
            self.logger.warning(f"Failed to initialize component loader: {e}")
    
    def get_config_manager(self):
        """获取配置管理器 - 向后兼容"""
        try:
            if self.component_loader:
                return self.component_loader.get_component("config_manager")
            else:
                # Fallback to old config manager
                from .config_manager import get_config_manager as old_get_config_manager
                return old_get_config_manager()
        except Exception as e:
            self.logger.warning(f"Config manager fallback: {e}")
            # Create minimal config manager
            return self._create_minimal_config_manager()
    
    def get_event_loop_manager(self):
        """获取事件循环管理器 - 向后兼容"""
        try:
            if self.component_loader:
                manager = self.component_loader.get_component("event_manager")
                if manager:
                    return manager
            
            # Fallback to old event manager
            from .unified_event_manager import get_event_loop_manager
            return get_event_loop_manager()
        except ImportError:
            self.logger.warning("Event loop manager not available, creating mock")
            return self._create_mock_event_manager()
        except Exception as e:
            self.logger.warning(f"Event loop manager error: {e}")
            return self._create_mock_event_manager()
    
    def get_resource_monitor(self):
        """获取资源监控器 - 向后兼容"""
        try:
            if self.component_loader:
                monitor = self.component_loader.get_component("resource_monitor")
                if monitor:
                    return monitor
            
            # Fallback to old resource monitor
            from .resource_monitor import get_resource_monitor
            return get_resource_monitor()
        except ImportError:
            self.logger.warning("Resource monitor not available, creating mock")
            return self._create_mock_resource_monitor()
        except Exception as e:
            self.logger.warning(f"Resource monitor error: {e}")
            return self._create_mock_resource_monitor()
    
    def _create_minimal_config_manager(self):
        """创建最小配置管理器"""
        class MinimalConfigManager:
            def __init__(self):
                self.defaults = {
                    'ibkr': {
                        'host': '127.0.0.1',
                        'port': 7497,
                        'client_id': 3130,
                        'account_id': None,
                        'use_delayed_if_no_realtime': True
                    },
                    'trading': {
                        'alloc': 0.03,
                        'poll_sec': 10.0,
                        'auto_sell_removed': True,
                        'fixed_quantity': 0
                    },
                    'risk_management': {
                        'max_single_position_pct': 0.15,
                        'max_sector_exposure_pct': 0.30,
                        'daily_loss_limit_pct': 0.05
                    }
                }
            
            def get(self, key: str, default=None):
                keys = key.split('.')
                current = self.defaults
                try:
                    for k in keys:
                        current = current[k]
                    return current
                except (KeyError, TypeError):
                    return default
            
            def set(self, key: str, value: Any):
                # 最小实现，不持久化
                pass
            
            def update_runtime_config(self, config: Dict):
                # 最小实现
                pass
        
        return MinimalConfigManager()
    
    def _create_mock_event_manager(self):
        """创建模拟事件管理器"""
        class MockEventManager:
            def __init__(self):
                self.started = False
            
            def start(self):
                self.started = True
                return True
            
            def stop(self):
                self.started = False
            
            def submit_coroutine_nowait(self, coro):
                import asyncio
                import uuid
                task_id = str(uuid.uuid4())
                # 简单调度到事件循环
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(coro)
                except RuntimeError:
                    # 没有事件循环，创建新的
                    asyncio.create_task(coro)
                return task_id
        
        return MockEventManager()
    
    def _create_mock_resource_monitor(self):
        """创建模拟资源监控器"""
        class MockResourceMonitor:
            def __init__(self):
                self.monitoring = False
            
            def start_monitoring(self):
                self.monitoring = True
            
            def stop_monitoring(self):
                self.monitoring = False
            
            def register_connection(self, connection):
                # Mock implementation
                pass
            
            def get_stats(self):
                return {
                    'cpu_usage': 0.0,
                    'memory_usage': 0.0,
                    'connections': 0
                }
        
        return MockResourceMonitor()


# 全局兼容性包装器实例
_compatibility_wrapper = None

def get_compatibility_wrapper():
    """获取兼容性包装器"""
    global _compatibility_wrapper
    if _compatibility_wrapper is None:
        _compatibility_wrapper = CompatibilityWrapper()
    return _compatibility_wrapper


# 向后兼容的函数导出
def get_config_manager():
    """向后兼容的配置管理器获取函数"""
    return get_compatibility_wrapper().get_config_manager()

def get_event_loop_manager():
    """向后兼容的事件循环管理器获取函数"""
    return get_compatibility_wrapper().get_event_loop_manager()

def get_resource_monitor():
    """向后兼容的资源监控器获取函数"""
    return get_compatibility_wrapper().get_resource_monitor()


# 确保旧的导入路径仍然有效
def ensure_backward_compatibility():
    """确保向后兼容性"""
    # 将兼容性函数注入到模块中
    current_module = sys.modules[__name__]
    
    # 如果旧模块不存在，创建兼容性版本
    compatibility_modules = [
        'autotrader.config_manager',
        'autotrader.unified_event_manager', 
        'autotrader.resource_monitor'
    ]
    
    for module_name in compatibility_modules:
        if module_name not in sys.modules:
            try:
                # 尝试导入原始模块
                __import__(module_name)
            except ImportError:
                # 如果导入失败，创建兼容性模块
                sys.modules[module_name] = current_module


# 自动执行兼容性确保
ensure_backward_compatibility()


if __name__ == "__main__":
    # 测试兼容性层
    logging.basicConfig(level=logging.INFO)
    
    wrapper = CompatibilityWrapper()
    
    # 测试配置管理器
    config_manager = wrapper.get_config_manager()
    print(f"IBKR Host: {config_manager.get('ibkr.host')}")
    
    # 测试事件管理器
    event_manager = wrapper.get_event_loop_manager()
    print(f"Event manager started: {event_manager.start()}")
    
    # 测试资源监控器
    resource_monitor = wrapper.get_resource_monitor()
    resource_monitor.start_monitoring()
    print(f"Resource monitor stats: {resource_monitor.get_stats()}")
    
    print("Compatibility layer test completed")