#!/usr/bin/env python3
"""
Lazy Component Loader - 懒加载组件管理器
减少app.py的直接依赖，提高启动速度和稳定性
"""

import logging
import sys
from typing import Dict, Any, Optional, Callable, Type
from pathlib import Path
import importlib
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)

@dataclass
class ComponentSpec:
    """组件规格定义"""
    module_path: str
    class_name: str
    init_args: tuple = field(default_factory=tuple)
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    required: bool = True
    description: str = ""

class LazyComponentLoader:
    """
    懒加载组件管理器
    只有在实际使用时才加载组件，减少启动时间和内存使用
    """
    
    def __init__(self):
        self.logger = logging.getLogger("LazyComponentLoader")
        self._components: Dict[str, Any] = {}
        self._component_specs: Dict[str, ComponentSpec] = {}
        self._loading_lock = threading.RLock()
        self._failed_components: set = set()
        
        # 注册核心组件
        self._register_core_components()
    
    def _register_core_components(self):
        """注册核心组件规格"""
        
        # 交易引擎 (需要参数，将在使用时手动创建)
        self.register_component(
            "engine",
            ComponentSpec(
                module_path="autotrader.engine",
                class_name="Engine", 
                required=False,  # 标记为非必需，因为需要手动创建
                description="Core trading engine (requires manual initialization)"
            )
        )
        
        # 数据库管理
        self.register_component(
            "database",
            ComponentSpec(
                module_path="autotrader.database",
                class_name="StockDatabase",
                required=True,
                description="Stock database manager"
            )
        )
        
        # IBKR交易器
        self.register_component(
            "ibkr_trader",
            ComponentSpec(
                module_path="autotrader.ibkr_auto_trader",
                class_name="IbkrAutoTrader",
                required=False,
                description="IBKR automated trader"
            )
        )
        
        # 统一交易核心
        self.register_component(
            "trading_core",
            ComponentSpec(
                module_path="autotrader.unified_trading_core",
                class_name="create_unified_trading_core",
                required=False,
                description="Unified trading core factory"
            )
        )
        
        # Alpha策略引擎已废弃 - 现在使用Simple 25策略
        # self.register_component(
        #     "alpha_engine",
        #     ComponentSpec(
        #         module_path="bma_models.enhanced_alpha_strategies",
        #         class_name="AlphaStrategiesEngine",
        #         required=False,
        #         description="Enhanced alpha strategies engine (DEPRECATED)"
        #     )
        # )
        
        # Polygon因子
        self.register_component(
            "polygon_factors",
            ComponentSpec(
                module_path="autotrader.unified_polygon_factors",
                class_name="UnifiedPolygonFactors",
                required=False,
                description="Unified Polygon factors"
            )
        )
        
        # 统一配置管理器
        self.register_component(
            "config_manager", 
            ComponentSpec(
                module_path="bma_models.unified_config_loader",
                class_name="get_config_manager",
                required=True,
                description="Unified configuration manager"
            )
        )
        
        # 风险管理器
        self.register_component(
            "risk_manager",
            ComponentSpec(
                module_path="autotrader.unified_risk_manager",
                class_name="get_risk_manager",
                required=True,
                description="Unified risk manager"
            )
        )
        
        # 事件循环管理器
        self.register_component(
            "event_manager",
            ComponentSpec(
                module_path="autotrader.unified_event_manager",
                class_name="get_event_loop_manager",
                required=False,
                description="Event loop manager"
            )
        )
        
        # 资源监控器
        self.register_component(
            "resource_monitor",
            ComponentSpec(
                module_path="autotrader.resource_monitor",
                class_name="get_resource_monitor",
                required=False,
                description="Resource monitor"
            )
        )
        
        # 错误处理器
        self.register_component(
            "error_handler",
            ComponentSpec(
                module_path="autotrader.error_handling_system",
                class_name="get_error_handler",
                required=True,
                description="Enhanced error handler"
            )
        )
        
        # 统一监控接口
        self.register_component(
            "monitoring_interface",
            ComponentSpec(
                module_path="autotrader.unified_monitoring_interface",
                class_name="get_monitoring_interface",
                required=False,
                description="Unified monitoring interface"
            )
        )
        
        # 原有监控系统（向后兼容）
        self.register_component(
            "monitoring_system",
            ComponentSpec(
                module_path="autotrader.unified_monitoring_system",
                class_name="UnifiedMonitoringSystem",
                required=False,
                description="Legacy unified monitoring system"
            )
        )
        
        self.logger.info(f"Registered {len(self._component_specs)} component specifications")
    
    def register_component(self, name: str, spec: ComponentSpec):
        """注册组件规格"""
        self._component_specs[name] = spec
        self.logger.debug(f"Registered component: {name} -> {spec.module_path}.{spec.class_name}")
    
    def get_component(self, name: str, **override_kwargs) -> Optional[Any]:
        """
        获取组件实例（懒加载）
        
        Args:
            name: 组件名称
            **override_kwargs: 覆盖初始化参数
            
        Returns:
            组件实例，如果加载失败则返回None
        """
        with self._loading_lock:
            # 检查是否已加载
            if name in self._components:
                return self._components[name]
            
            # 检查是否已失败
            if name in self._failed_components:
                self.logger.debug(f"Component {name} previously failed to load")
                return None
            
            # 检查规格是否存在
            if name not in self._component_specs:
                self.logger.error(f"Unknown component: {name}")
                return None
            
            # 加载组件
            spec = self._component_specs[name]
            try:
                instance = self._load_component(spec, **override_kwargs)
                if instance is not None:
                    self._components[name] = instance
                    self.logger.info(f"Successfully loaded component: {name}")
                    return instance
                else:
                    self._failed_components.add(name)
                    if spec.required:
                        self.logger.error(f"Failed to load required component: {name}")
                    else:
                        self.logger.warning(f"Failed to load optional component: {name}")
                    return None
                    
            except Exception as e:
                self._failed_components.add(name)
                if spec.required:
                    self.logger.error(f"Critical component {name} failed to load: {e}")
                else:
                    self.logger.warning(f"Optional component {name} failed to load: {e}")
                return None
    
    def _load_component(self, spec: ComponentSpec, **override_kwargs) -> Optional[Any]:
        """加载单个组件"""
        try:
            # 动态导入模块
            module = importlib.import_module(spec.module_path)
            
            # 获取类或函数
            if hasattr(module, spec.class_name):
                component_class = getattr(module, spec.class_name)
            else:
                self.logger.error(f"Class {spec.class_name} not found in module {spec.module_path}")
                return None
            
            # 合并初始化参数
            init_kwargs = {**spec.init_kwargs, **override_kwargs}
            
            # 创建实例
            if callable(component_class):
                if spec.init_args:
                    instance = component_class(*spec.init_args, **init_kwargs)
                else:
                    instance = component_class(**init_kwargs)
            else:
                # 如果是单例函数（如get_xxx_manager），直接调用
                instance = component_class
            
            self.logger.debug(f"Created component instance: {spec.class_name}")
            return instance
            
        except ImportError as e:
            self.logger.warning(f"Module import failed for {spec.module_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Component creation failed for {spec.class_name}: {e}")
            return None
    
    def preload_critical_components(self) -> bool:
        """预加载关键组件"""
        critical_components = [
            name for name, spec in self._component_specs.items() 
            if spec.required
        ]
        
        success_count = 0
        for name in critical_components:
            if self.get_component(name) is not None:
                success_count += 1
        
        success_rate = success_count / len(critical_components) if critical_components else 1.0
        
        self.logger.info(
            f"Preloaded {success_count}/{len(critical_components)} critical components "
            f"({success_rate:.1%} success rate)"
        )
        
        return success_rate >= 0.8  # 80%成功率认为可用
    
    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """获取组件状态"""
        status = {}
        
        for name, spec in self._component_specs.items():
            is_loaded = name in self._components
            has_failed = name in self._failed_components
            
            status[name] = {
                'loaded': is_loaded,
                'failed': has_failed,
                'required': spec.required,
                'description': spec.description,
                'module_path': spec.module_path,
                'class_name': spec.class_name
            }
        
        return status
    
    def reload_component(self, name: str, **override_kwargs) -> Optional[Any]:
        """重新加载组件"""
        with self._loading_lock:
            # 清除缓存
            if name in self._components:
                del self._components[name]
            if name in self._failed_components:
                self._failed_components.remove(name)
            
            # 重新加载
            return self.get_component(name, **override_kwargs)
    
    def clear_failed_components(self):
        """清除失败组件标记，允许重试"""
        with self._loading_lock:
            self._failed_components.clear()
            self.logger.info("Cleared failed component markers")
    
    def get_loaded_components(self) -> Dict[str, Any]:
        """获取已加载的组件"""
        return self._components.copy()
    
    def unload_component(self, name: str):
        """卸载组件"""
        with self._loading_lock:
            if name in self._components:
                component = self._components[name]
                # 如果组件有cleanup方法，调用它
                if hasattr(component, 'cleanup'):
                    try:
                        component.cleanup()
                    except Exception as e:
                        self.logger.warning(f"Component {name} cleanup failed: {e}")
                
                del self._components[name]
                self.logger.info(f"Unloaded component: {name}")
    
    def shutdown(self):
        """关闭组件加载器，清理所有组件"""
        with self._loading_lock:
            for name in list(self._components.keys()):
                self.unload_component(name)
            
            self._components.clear()
            self._failed_components.clear()
            
            self.logger.info("Component loader shutdown completed")


# 全局单例
_global_loader: Optional[LazyComponentLoader] = None
_loader_lock = threading.RLock()

def get_component_loader() -> LazyComponentLoader:
    """获取全局组件加载器"""
    global _global_loader
    
    with _loader_lock:
        if _global_loader is None:
            _global_loader = LazyComponentLoader()
        return _global_loader

def reset_component_loader():
    """重置全局组件加载器（用于测试）"""
    global _global_loader
    with _loader_lock:
        if _global_loader:
            _global_loader.shutdown()
        _global_loader = None


if __name__ == "__main__":
    # 测试组件加载器
    logging.basicConfig(level=logging.INFO)
    
    loader = LazyComponentLoader()
    
    # 测试组件状态
    status = loader.get_component_status()
    print(f"Component status: {len(status)} components registered")
    
    # 测试预加载
    success = loader.preload_critical_components()
    print(f"Critical components preload: {'SUCCESS' if success else 'FAILED'}")
    
    # 显示加载状态
    loaded = loader.get_loaded_components()
    print(f"Loaded components: {list(loaded.keys())}")
    
    print("Component loader test completed")