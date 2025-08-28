#!/usr/bin/env python3
"""
统一配置管理中心 - 提供集中化、层级化的配置管理
支持多层配置覆盖：默认配置 < 文件配置 < 数据库配置 < 环境变量
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """交易配置数据结构"""
    # 连接配置
    connection: Dict[str, Any] = None
    
    # 风险管理配置
    risk_management: Dict[str, Any] = None
    
    # 价格验证配置
    price_validation: Dict[str, Any] = None
    
    # 监控配置
    monitoring: Dict[str, Any] = None
    
    # 执行配置
    execution: Dict[str, Any] = None
    
    # 数据配置
    data_sources: Dict[str, Any] = None
    
    def __post_init__(self):
        # 设置默认值
        if self.connection is None:
            self.connection = self._get_default_connection_config()
        if self.risk_management is None:
            self.risk_management = self._get_default_risk_config()
        if self.price_validation is None:
            self.price_validation = self._get_default_price_validation_config()
        if self.monitoring is None:
            self.monitoring = self._get_default_monitoring_config()
        if self.execution is None:
            self.execution = self._get_default_execution_config()
        if self.data_sources is None:
            self.data_sources = self._get_default_data_sources_config()
    
    @staticmethod
    def _get_default_connection_config() -> Dict[str, Any]:
        return {
            "host": "127.0.0.1",
            "port": 7497,
            "client_id": 1,
            "account_id": "",
            "use_delayed_if_no_realtime": True,
            "reconnect_attempts": 5,
            "reconnect_delay_seconds": 5,
            "connection_timeout": 30
        }
    
    @staticmethod 
    def _get_default_risk_config() -> Dict[str, Any]:
        return {
            "max_single_position_pct": 0.20,  # 最大单仓20%
            "max_sector_exposure_pct": 0.30,  # 最大行业敞口30%
            "cash_reserve_pct": 0.10,  # 现金保留10%
            "max_daily_orders": 50,  # 日内最大订单数
            "max_orders_per_5min": 10,  # 5分钟最大订单数
            "max_orders_per_hour": 25,  # 小时最大订单数
            "stop_loss_pct": 0.02,  # 默认止损2%
            "take_profit_pct": 0.05,  # 默认止盈5%
            "max_drawdown_pct": 0.05,  # 最大回撤5%
            "verify_tolerance_usd": 100.0,  # 验证容差$100
            "enable_sector_limits": True,
            "enable_correlation_check": True,
            "max_correlation": 0.7
        }
    
    @staticmethod
    def _get_default_price_validation_config() -> Dict[str, Any]:
        return {
            "min_price": 0.01,
            "max_price": 50000.0,
            "max_daily_change_pct": 0.30,
            "max_tick_change_pct": 0.05,
            "max_data_age_seconds": 180.0,
            "stale_warning_seconds": 30.0,
            "outlier_std_multiplier": 3.0,
            "allow_avgcost_fallback": True,
            "allow_last_known_fallback": True,
            "fallback_max_age_hours": 24.0
        }
    
    @staticmethod
    def _get_default_monitoring_config() -> Dict[str, Any]:
        return {
            "account_update_interval": 60.0,  # 账户更新间隔60秒
            "position_monitor_interval": 30.0,  # 头寸监控间隔30秒
            "risk_monitor_interval": 15.0,  # 风险监控间隔15秒
            "enable_real_time_alerts": True,
            "alert_cooldown_seconds": 300,  # 告警冷却5分钟
            "log_level": "INFO",
            "enable_performance_metrics": True,
            "metrics_export_interval": 300  # 指标导出间隔5分钟
        }
    
    @staticmethod
    def _get_default_execution_config() -> Dict[str, Any]:
        return {
            "default_order_type": "LIMIT",
            "execution_timeout_seconds": 30,
            "max_slippage_pct": 0.01,  # 最大滑点1%
            "enable_smart_routing": True,
            "min_order_value": 100.0,  # 最小订单价值$100
            "enable_fractional_shares": False,
            "default_time_in_force": "DAY",
            "enable_pre_market": False,
            "enable_after_hours": False
        }
    
    @staticmethod
    def _get_default_data_sources_config() -> Dict[str, Any]:
        return {
            "primary_source": "IBKR",
            "fallback_source": "POLYGON", 
            "enable_delayed_data": True,
            "data_cache_ttl_seconds": 5,
            "enable_data_validation": True,
            "max_quote_age_seconds": 60,
            "enable_tick_data": False,
            "enable_level2_data": False
        }

class CentralizedConfigManager:
    """统一配置管理器"""
    
    def __init__(self, config_dir: str = "config", db_path: Optional[str] = None):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.db_path = db_path
        self._config_cache: Dict[str, Any] = {}
        self._last_reload_time: float = 0.0
        self._config_lock = threading.RLock()
        
        # 配置文件路径
        self.main_config_file = self.config_dir / "trading_config.json"
        self.risk_config_file = self.config_dir / "risk_config.json"
        self.monitoring_config_file = self.config_dir / "monitoring_config.json"
        
        # 加载配置
        self.reload_config()
    
    def reload_config(self) -> None:
        """重新加载所有配置"""
        with self._config_lock:
            try:
                # 1. 加载默认配置
                self._config_cache = asdict(TradingConfig())
                
                # 2. 加载文件配置
                self._load_file_configs()
                
                # 3. 加载数据库配置
                self._load_database_config()
                
                # 4. 加载环境变量配置
                self._load_environment_config()
                
                self._last_reload_time = datetime.now().timestamp()
                logger.info("配置重新加载完成")
                
            except Exception as e:
                logger.error(f"配置加载失败: {e}")
                # 使用默认配置作为回退
                if not self._config_cache:
                    self._config_cache = asdict(TradingConfig())
    
    def _load_file_configs(self) -> None:
        """加载文件配置"""
        config_files = {
            'main': self.main_config_file,
            'risk': self.risk_config_file, 
            'monitoring': self.monitoring_config_file
        }
        
        for config_type, config_file in config_files.items():
            try:
                if config_file.exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        file_config = json.load(f)
                        self._merge_config(self._config_cache, file_config)
                        logger.debug(f"已加载{config_type}配置文件: {config_file}")
                else:
                    logger.info(f"配置文件不存在，创建默认配置: {config_file}")
                    self._create_default_config_file(config_file, config_type)
                    
            except json.JSONDecodeError as e:
                logger.error(f"配置文件格式错误 {config_file}: {e}")
            except Exception as e:
                logger.error(f"加载配置文件失败 {config_file}: {e}")
    
    def _create_default_config_file(self, config_file: Path, config_type: str) -> None:
        """创建默认配置文件"""
        try:
            if config_type == 'main':
                default_config = asdict(TradingConfig())
            elif config_type == 'risk':
                default_config = {'risk_management': TradingConfig._get_default_risk_config()}
            elif config_type == 'monitoring':
                default_config = {'monitoring': TradingConfig._get_default_monitoring_config()}
            else:
                default_config = {}
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"创建默认配置文件失败 {config_file}: {e}")
    
    def _load_database_config(self) -> None:
        """从数据库加载配置"""
        if not self.db_path:
            return
            
        try:
            # 这里集成现有的数据库配置加载逻辑
            from .database import StockDatabase
            db = StockDatabase()
            
            # 加载风险配置
            risk_config = db.get_risk_config("默认风险配置")
            if risk_config and isinstance(risk_config, dict):
                if 'risk_management' not in self._config_cache:
                    self._config_cache['risk_management'] = {}
                self._merge_config(self._config_cache['risk_management'], risk_config)
                logger.debug("已加载数据库风险配置")
                
        except Exception as e:
            logger.warning(f"加载数据库配置失败: {e}")
    
    def _load_environment_config(self) -> None:
        """从环境变量加载配置"""
        env_mappings = {
            'IBKR_HOST': ('connection', 'host'),
            'IBKR_PORT': ('connection', 'port'),
            'IBKR_CLIENT_ID': ('connection', 'client_id'),
            'IBKR_ACCOUNT_ID': ('connection', 'account_id'),
            'TRADING_LOG_LEVEL': ('monitoring', 'log_level'),
            'MAX_POSITION_PCT': ('risk_management', 'max_single_position_pct'),
            'CASH_RESERVE_PCT': ('risk_management', 'cash_reserve_pct'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                try:
                    # 类型转换
                    if key in ['port', 'client_id']:
                        value = int(value)
                    elif key.endswith('_pct'):
                        value = float(value)
                    elif key == 'use_delayed_if_no_realtime':
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    
                    if section not in self._config_cache:
                        self._config_cache[section] = {}
                    self._config_cache[section][key] = value
                    logger.debug(f"已加载环境变量配置: {env_var}={value}")
                    
                except ValueError as e:
                    logger.error(f"环境变量类型转换失败 {env_var}={value}: {e}")
    
    def _merge_config(self, base_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """递归合并配置字典"""
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值，支持点分路径
        
        Examples:
            get('risk_management.max_single_position_pct')
            get('connection.host')
        """
        with self._config_lock:
            keys = key_path.split('.')
            current = self._config_cache
            
            try:
                for key in keys:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        return default
                return current
            except (KeyError, TypeError):
                return default
    
    def set(self, key_path: str, value: Any, persist: bool = False) -> None:
        """
        设置配置值
        
        Args:
            key_path: 配置路径
            value: 配置值
            persist: 是否持久化到文件
        """
        with self._config_lock:
            keys = key_path.split('.')
            current = self._config_cache
            
            # 导航到目标位置
            for key in keys[:-1]:
                if key not in current or not isinstance(current[key], dict):
                    current[key] = {}
                current = current[key]
            
            # 设置值
            current[keys[-1]] = value
            
            if persist:
                self._persist_config(keys[0])  # 持久化到对应的配置文件
            
            logger.debug(f"配置已更新: {key_path}={value}")
    
    def _persist_config(self, section: str) -> None:
        """持久化配置到文件"""
        try:
            if section == 'risk_management':
                config_file = self.risk_config_file
                config_data = {'risk_management': self._config_cache.get('risk_management', {})}
            elif section == 'monitoring':
                config_file = self.monitoring_config_file
                config_data = {'monitoring': self._config_cache.get('monitoring', {})}
            else:
                config_file = self.main_config_file
                config_data = self._config_cache
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"配置已持久化: {config_file}")
            
        except Exception as e:
            logger.error(f"持久化配置失败 {section}: {e}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取整个配置段"""
        with self._config_lock:
            return self._config_cache.get(section, {}).copy()
    
    def get_full_config(self) -> Dict[str, Any]:
        """获取完整配置副本"""
        with self._config_lock:
            return self._config_cache.copy()
    
    def validate_config(self) -> List[str]:
        """验证配置有效性，返回错误列表"""
        errors = []
        
        try:
            # 验证连接配置
            conn = self.get_section('connection')
            if not conn.get('host'):
                errors.append("连接主机地址未配置")
            if not isinstance(conn.get('port'), int) or conn.get('port') <= 0:
                errors.append("连接端口配置无效")
            
            # 验证风险配置
            risk = self.get_section('risk_management')
            max_pos = risk.get('max_single_position_pct', 0)
            if not 0 < max_pos <= 1.0:
                errors.append(f"最大单仓比例配置无效: {max_pos}")
                
            cash_reserve = risk.get('cash_reserve_pct', 0)
            if not 0 <= cash_reserve < 1.0:
                errors.append(f"现金保留比例配置无效: {cash_reserve}")
            
            # 验证监控配置
            monitor = self.get_section('monitoring')
            update_interval = monitor.get('account_update_interval', 0)
            if update_interval <= 0:
                errors.append(f"账户更新间隔配置无效: {update_interval}")
            
        except Exception as e:
            errors.append(f"配置验证异常: {e}")
        
        return errors
    
    def export_config(self, file_path: str) -> None:
        """导出当前配置到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.get_full_config(), f, indent=2, ensure_ascii=False)
            logger.info(f"配置已导出到: {file_path}")
        except Exception as e:
            logger.error(f"导出配置失败: {e}")

# 全局配置管理器实例
_global_config_manager: Optional[CentralizedConfigManager] = None

def get_centralized_config_manager(config_dir: str = "config", 
                                 db_path: Optional[str] = None) -> CentralizedConfigManager:
    """获取全局配置管理器实例"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = CentralizedConfigManager(config_dir, db_path)
    return _global_config_manager

def create_config_manager(config_dir: str = "config", 
                        db_path: Optional[str] = None) -> CentralizedConfigManager:
    """创建新的配置管理器实例"""
    return CentralizedConfigManager(config_dir, db_path)