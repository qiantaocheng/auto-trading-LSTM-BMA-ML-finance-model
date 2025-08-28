#!/usr/bin/env python3
"""
Master Configuration Manager - 主配置管理器
整合所有配置管理功能，提供统一的配置访问接口
"""

import os
import json
import logging
import sqlite3
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from copy import deepcopy
import yaml
import configparser

logger = logging.getLogger(__name__)

@dataclass
class ConfigSource:
    """配置源定义"""
    name: str
    path: str
    format: str  # json, yaml, ini, env, db
    priority: int  # 数值越大优先级越高
    required: bool = False
    description: str = ""

class MasterConfigManager:
    """
    主配置管理器
    整合所有配置源，提供统一的配置访问接口
    """
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger("MasterConfigManager")
        self._lock = threading.RLock()
        
        # 配置数据
        self.config_data: Dict[str, Any] = {}
        self.config_sources: List[ConfigSource] = []
        self.last_load_time = 0
        self.auto_reload = True
        
        # 变更监听器
        self.change_listeners: List[Callable[[str, Any, Any], None]] = []
        
        # 缓存
        self._cache_enabled = True
        self._cache: Dict[str, Any] = {}
        self._cache_dirty = True
        
        # 初始化配置源
        self._initialize_config_sources()
        
        # 加载所有配置
        self.reload_all_configs()
        
        self.logger.info("Master configuration manager initialized")
    
    def _initialize_config_sources(self):
        """初始化配置源"""
        # 默认配置源（按优先级从低到高）
        default_sources = [
            ConfigSource(
                name="defaults",
                path="",  # 内置默认值
                format="dict",
                priority=0,
                required=True,
                description="Built-in default configuration"
            ),
            ConfigSource(
                name="main_config",
                path=str(self.base_path / "config.json"),
                format="json",
                priority=10,
                required=False,
                description="Main configuration file"
            ),
            ConfigSource(
                name="risk_config",
                path=str(self.base_path / "data" / "risk_config.json"),
                format="json",
                priority=15,
                required=False,
                description="Risk management configuration"
            ),
            ConfigSource(
                name="trading_config",
                path=str(self.base_path / "data" / "trading_config.yaml"),
                format="yaml",
                priority=15,
                required=False,
                description="Trading configuration"
            ),
            ConfigSource(
                name="connection_config",
                path=str(self.base_path / "data" / "connection.json"),
                format="json",
                priority=20,
                required=False,
                description="Connection configuration"
            ),
            ConfigSource(
                name="environment",
                path="",  # 环境变量
                format="env",
                priority=30,
                required=False,
                description="Environment variables"
            ),
            ConfigSource(
                name="database_config",
                path=str(self.base_path / "data" / "autotrader_stocks.db"),
                format="db",
                priority=25,
                required=False,
                description="Database stored configuration"
            ),
            ConfigSource(
                name="runtime_overrides",
                path="",  # 运行时覆盖
                format="dict",
                priority=40,
                required=False,
                description="Runtime configuration overrides"
            )
        ]
        
        self.config_sources = default_sources
        self.logger.info(f"Initialized {len(self.config_sources)} configuration sources")
    
    def reload_all_configs(self) -> bool:
        """重新加载所有配置"""
        success = True
        merged_config = {}
        
        # 按优先级排序
        sorted_sources = sorted(self.config_sources, key=lambda x: x.priority)
        
        for source in sorted_sources:
            try:
                config_data = self._load_config_source(source)
                if config_data:
                    # 深度合并配置
                    merged_config = self._deep_merge(merged_config, config_data)
                    self.logger.debug(f"Loaded config from {source.name}")
                elif source.required:
                    self.logger.error(f"Required config source {source.name} failed to load")
                    success = False
            except Exception as e:
                if source.required:
                    self.logger.error(f"Failed to load required config {source.name}: {e}")
                    success = False
                else:
                    self.logger.warning(f"Failed to load optional config {source.name}: {e}")
        
        with self._lock:
            self.config_data = merged_config
            self.last_load_time = datetime.now().timestamp()
            self._cache_dirty = True
            self._cache.clear()
        
        self.logger.info(f"Configuration reload completed (success: {success})")
        return success
    
    def _load_config_source(self, source: ConfigSource) -> Optional[Dict[str, Any]]:
        """加载单个配置源"""
        if source.format == "dict" and source.name == "defaults":
            return self._get_default_config()
        elif source.format == "dict" and source.name == "runtime_overrides":
            return getattr(self, '_runtime_overrides', {})
        elif source.format == "env":
            return self._load_environment_config()
        elif source.format == "json":
            return self._load_json_config(source.path)
        elif source.format == "yaml":
            return self._load_yaml_config(source.path)
        elif source.format == "ini":
            return self._load_ini_config(source.path)
        elif source.format == "db":
            return self._load_database_config(source.path)
        else:
            self.logger.warning(f"Unsupported config format: {source.format}")
            return None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'ibkr': {
                'host': '127.0.0.1',
                'port': 7497,
                'client_id': 3130,
                'account_id': None,
                'use_delayed_if_no_realtime': True,
                'min_client_id': 1000,
                'max_client_id': 9999,
                'reserved_ports': [7496, 7497],
                'connection_timeout': 30,
                'retry_attempts': 3
            },
            'trading': {
                'alloc': 0.03,
                'poll_sec': 10.0,
                'default_stop_loss_pct': 0.02,
                'default_take_profit_pct': 0.05,
                'acceptance_threshold': 0.6,
                'max_positions': 20,
                'auto_sell_removed': True,
                'fixed_quantity': 0
            },
            'risk_management': {
                'max_single_position_pct': 0.15,
                'max_sector_exposure_pct': 0.30,
                'max_correlation': 0.70,
                'daily_loss_limit_pct': 0.05,
                'max_daily_orders': 20,
                'min_order_value': 1000.0,
                'max_order_value': 50000.0,
                'concentration_warning_pct': 0.25,
                'volatility_warning_pct': 0.05
            },
            'sizing': {
                'method': 'equal_weight',
                'base_position_size': 1000,
                'max_position_pct_of_equity': 0.10,
                'min_position_usd': 500,
                'volatility_adjustment': True
            },
            'data': {
                'polygon_api_key': None,
                'data_refresh_interval': 300,
                'cache_enabled': True,
                'cache_duration_hours': 24,
                'data_quality_threshold': 0.8
            },
            'logging': {
                'level': 'INFO',
                'file_enabled': True,
                'console_enabled': True,
                'max_file_size_mb': 100,
                'backup_count': 5
            },
            'monitoring': {
                'enabled': True,
                'level': 'standard',
                'performance_tracking': True,
                'alert_enabled': True,
                'export_interval_hours': 24
            },
            'system': {
                'max_memory_mb': 2048,
                'cpu_limit_percent': 80,
                'temp_dir': 'temp',
                'backup_enabled': True,
                'backup_interval_hours': 6
            }
        }
    
    def _load_environment_config(self) -> Dict[str, Any]:
        """从环境变量加载配置"""
        env_config = {}
        
        # 定义环境变量映射
        env_mappings = {
            'IBKR_HOST': 'ibkr.host',
            'IBKR_PORT': 'ibkr.port',
            'IBKR_CLIENT_ID': 'ibkr.client_id',
            'IBKR_ACCOUNT_ID': 'ibkr.account_id',
            'POLYGON_API_KEY': 'data.polygon_api_key',
            'TRADING_ALLOC': 'trading.alloc',
            'TRADING_POLL_SEC': 'trading.poll_sec',
            'LOG_LEVEL': 'logging.level',
            'MAX_POSITIONS': 'trading.max_positions'
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                # 类型转换
                if config_path.endswith('.port') or config_path.endswith('.client_id'):
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                elif config_path.endswith('.alloc') or config_path.endswith('.poll_sec'):
                    try:
                        value = float(value)
                    except ValueError:
                        continue
                
                # 设置嵌套配置
                self._set_nested_config(env_config, config_path, value)
        
        return env_config
    
    def _load_json_config(self, filepath: str) -> Optional[Dict[str, Any]]:
        """加载JSON配置文件"""
        try:
            path = Path(filepath)
            if not path.exists():
                return None
            
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load JSON config {filepath}: {e}")
            return None
    
    def _load_yaml_config(self, filepath: str) -> Optional[Dict[str, Any]]:
        """加载YAML配置文件"""
        try:
            path = Path(filepath)
            if not path.exists():
                return None
            
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load YAML config {filepath}: {e}")
            return None
    
    def _load_ini_config(self, filepath: str) -> Optional[Dict[str, Any]]:
        """加载INI配置文件"""
        try:
            path = Path(filepath)
            if not path.exists():
                return None
            
            config = configparser.ConfigParser()
            config.read(path, encoding='utf-8')
            
            # 转换为字典
            result = {}
            for section in config.sections():
                result[section] = dict(config.items(section))
            
            return result
        except Exception as e:
            self.logger.warning(f"Failed to load INI config {filepath}: {e}")
            return None
    
    def _load_database_config(self, db_path: str) -> Optional[Dict[str, Any]]:
        """从数据库加载配置"""
        try:
            path = Path(db_path)
            if not path.exists():
                return None
            
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            
            # 检查配置表是否存在
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='app_config'
            """)
            
            if not cursor.fetchone():
                conn.close()
                return None
            
            # 加载配置
            cursor.execute("SELECT key, value FROM app_config")
            rows = cursor.fetchall()
            conn.close()
            
            config = {}
            for key, value in rows:
                try:
                    # 尝试解析JSON值
                    parsed_value = json.loads(value)
                    self._set_nested_config(config, key, parsed_value)
                except json.JSONDecodeError:
                    # 如果不是JSON，直接设置字符串值
                    self._set_nested_config(config, key, value)
            
            return config
            
        except Exception as e:
            self.logger.warning(f"Failed to load database config {db_path}: {e}")
            return None
    
    def _set_nested_config(self, config: Dict, path: str, value: Any):
        """设置嵌套配置值"""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """深度合并字典"""
        result = deepcopy(base)
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点分隔的嵌套键 (e.g., "ibkr.host")
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        if self._cache_enabled and not self._cache_dirty:
            cached_value = self._cache.get(key)
            if cached_value is not None:
                return cached_value
        
        with self._lock:
            current = self.config_data
            
            try:
                for part in key.split('.'):
                    current = current[part]
                
                # 缓存结果
                if self._cache_enabled:
                    self._cache[key] = current
                
                return current
                
            except (KeyError, TypeError):
                return default
    
    def set(self, key: str, value: Any, persist: bool = False) -> bool:
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
            persist: 是否持久化到文件
            
        Returns:
            设置是否成功
        """
        old_value = self.get(key)
        
        with self._lock:
            # 设置运行时覆盖
            if not hasattr(self, '_runtime_overrides'):
                self._runtime_overrides = {}
            
            self._set_nested_config(self._runtime_overrides, key, value)
            
            # 清除缓存
            self._cache_dirty = True
            self._cache.clear()
            
            # 重新加载配置以应用覆盖
            self.reload_all_configs()
        
        # 通知监听器
        for listener in self.change_listeners:
            try:
                listener(key, old_value, value)
            except Exception as e:
                self.logger.warning(f"Config change listener failed: {e}")
        
        # 持久化（如果需要）
        if persist:
            return self._persist_config(key, value)
        
        return True
    
    def _persist_config(self, key: str, value: Any) -> bool:
        """持久化配置到文件"""
        try:
            # 根据配置键决定持久化到哪个文件
            if key.startswith('ibkr.'):
                config_file = self.base_path / "data" / "connection.json"
            elif key.startswith('risk_management.'):
                config_file = self.base_path / "data" / "risk_config.json"
            elif key.startswith('trading.'):
                config_file = self.base_path / "data" / "trading_config.yaml"
            else:
                config_file = self.base_path / "config.json"
            
            # 确保目录存在
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 加载现有配置
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    if config_file.suffix == '.yaml':
                        existing_config = yaml.safe_load(f) or {}
                    else:
                        existing_config = json.load(f)
            else:
                existing_config = {}
            
            # 更新配置
            self._set_nested_config(existing_config, key, value)
            
            # 写入文件
            with open(config_file, 'w', encoding='utf-8') as f:
                if config_file.suffix == '.yaml':
                    yaml.dump(existing_config, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(existing_config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration {key} persisted to {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to persist configuration {key}: {e}")
            return False
    
    def add_change_listener(self, listener: Callable[[str, Any, Any], None]):
        """添加配置变更监听器"""
        self.change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[[str, Any, Any], None]):
        """移除配置变更监听器"""
        if listener in self.change_listeners:
            self.change_listeners.remove(listener)
    
    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        with self._lock:
            return deepcopy(self.config_data)
    
    def get_connection_params(self, auto_allocate_client_id: bool = True) -> Dict[str, Any]:
        """获取IBKR连接参数"""
        params = {
            'host': self.get('ibkr.host', '127.0.0.1'),
            'port': self.get('ibkr.port', 7497),
            'client_id': self.get('ibkr.client_id', 3130)
        }
        
        # 如果需要自动分配client_id，可以在这里实现逻辑
        if auto_allocate_client_id:
            # 简单的自动分配逻辑，基于当前时间生成
            import time
            base_id = params['client_id']
            params['client_id'] = base_id + int(time.time()) % 1000
        
        return params
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            'last_load_time': self.last_load_time,
            'config_sources': [
                {
                    'name': source.name,
                    'path': source.path,
                    'format': source.format,
                    'priority': source.priority,
                    'required': source.required,
                    'exists': Path(source.path).exists() if source.path else True
                }
                for source in self.config_sources
            ],
            'total_config_keys': self._count_config_keys(self.config_data),
            'cache_enabled': self._cache_enabled,
            'cache_size': len(self._cache),
            'change_listeners': len(self.change_listeners)
        }
    
    def _count_config_keys(self, config: Dict, prefix: str = "") -> int:
        """递归计算配置键数量"""
        count = 0
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                count += self._count_config_keys(value, full_key)
            else:
                count += 1
        return count


# 全局单例
_master_config_manager: Optional[MasterConfigManager] = None
_config_lock = threading.RLock()

def get_master_config_manager(base_path: str = ".") -> MasterConfigManager:
    """获取主配置管理器单例"""
    global _master_config_manager
    
    with _config_lock:
        if _master_config_manager is None:
            _master_config_manager = MasterConfigManager(base_path)
        return _master_config_manager

def reset_master_config_manager():
    """重置主配置管理器（用于测试）"""
    global _master_config_manager
    with _config_lock:
        _master_config_manager = None


# 向后兼容的便捷函数
def get_config_manager():
    """向后兼容的配置管理器获取函数"""
    return get_master_config_manager()


def get_default_config():
    """获取默认配置管理器实例"""
    return get_master_config_manager()


if __name__ == "__main__":
    # 测试主配置管理器
    logging.basicConfig(level=logging.INFO)
    
    config_manager = MasterConfigManager()
    
    # 测试配置获取
    print(f"IBKR Host: {config_manager.get('ibkr.host')}")
    print(f"Trading Alloc: {config_manager.get('trading.alloc')}")
    print(f"Unknown Key: {config_manager.get('unknown.key', 'default_value')}")
    
    # 测试配置设置
    config_manager.set('trading.custom_setting', 'test_value')
    print(f"Custom Setting: {config_manager.get('trading.custom_setting')}")
    
    # 获取配置摘要
    summary = config_manager.get_config_summary()
    print(f"Config Summary: {json.dumps(summary, indent=2, default=str)}")
    
    print("Master configuration manager test completed")