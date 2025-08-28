#!/usr/bin/env python3
"""
Unified Configuration Manager - Single source of truth for all configuration
Replaces: UnifiedConfigManager, EnvironmentConfig, HotConfig inconsistencies
"""

import os
import json
import logging
import sqlite3
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from threading import RLock
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import threading


logger = logging.getLogger(__name__)


@dataclass 
class ConfigValidationResult:
    """Configuration validation result"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ConfigManager:
    """Unified Configuration Manager - Single source of truth"""
    
    # Configuration layer priorities (higher = higher priority)
    LAYER_PRIORITY = {
        'defaults': 0,
        'file_config': 10, 
        'database': 20,
        'environment': 30,
        'runtime': 40
    }
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger("ConfigManager")
        self._lock = RLock()
        
        # Configuration layers
        self._layers: Dict[str, Dict[str, Any]] = {
            'defaults': {},
            'file_config': {},
            'database': {},
            'environment': {},
            'runtime': {}
        }
        
        # Merged configuration cache
        self._merged_cache: Optional[Dict[str, Any]] = None
        self._cache_dirty = True
        self._last_merge_time = 0
        
        # File paths
        self.config_files = {
            'main': self.base_path / 'config.json',
            'risk': self.base_path / 'data' / 'risk_config.json',
            'connection': self.base_path / 'data' / 'connection.json',
            'database': self.base_path / 'data' / 'autotrader_stocks.db'
        }
        
        # Change listeners
        self._change_listeners: List[Callable[[str, Any, Any], None]] = []
        
        # Initialize configuration layers
        self._initialize_defaults()
        self._load_environment()
        self._load_file_configs()
        self._load_database_config()
        
        self.logger.info("ConfigManager initialized successfully")
    
    def _initialize_defaults(self):
        """Initialize default configuration values"""
        self._layers['defaults'] = {
            # Connection settings
            'ibkr': {
                'host': '127.0.0.1',
                'port': 7497,
                'client_id': 3130,
                'account_id': None,  # Must be set via environment
                'use_delayed_if_no_realtime': True,  # Allow delayed data fallback
                'min_client_id': 1000,
                'max_client_id': 9999,
                'reserved_ports': [7496, 7497]
            },
            
            # Trading parameters
            'trading': {
                'default_stop_loss_pct': 0.02,
                'default_take_profit_pct': 0.05,
                'acceptance_threshold': 0.6,
                'max_position_pct': 0.15,
                'cash_reserve_pct': 0.15,
                'max_daily_orders': 20,
                'per_trade_risk_pct': 0.02
            },
            
            # Risk management  
            'risk': {
                'sector_exposure_limit': 0.30,
                'portfolio_exposure_limit': 0.85,
                'max_single_position_pct': 0.15,
                'require_production_validation': True
            },
            
            # Pricing fallbacks
            'pricing': {
                'fallback': {
                    'default': 100.0,
                    'AAPL': 180.0,
                    'MSFT': 350.0,
                    'GOOGL': 140.0,
                    'TSLA': 200.0,
                    'AMZN': 130.0
                }
            },
            
            # System settings
            'system': {
                'environment': 'development',
                'demo_mode': False,
                'log_level': 'INFO',
                'data_source': 'polygon',
                'allow_fallback_data': False,
                'signal_mode': 'production'
            },
            
            # Data settings
            'data': {
                'polygon_api_key': None,
                'enable_delayed_data': True,
                'data_delay_minutes': 15,
                'cache_timeout_seconds': 300
            },
            
            # Scanner/Universe settings
            'scanner': {
                'universe': ['SPY', 'QQQ', 'IWM'],  # Safe default
                'max_universe_size': 100,
                'min_market_cap': 1000000000,  # $1B minimum
                'min_volume': 1000000  # 1M shares minimum
            },
            
            # Signals configuration
            'signals': {
                'acceptance_threshold': 0.6,
                'min_confidence': 0.5,
                'max_signal_age_minutes': 30,
                'require_multiple_confirmations': True
            },
            
            # Order execution
            'orders': {
                'smart_price_mode': 'midpoint',
                'max_slippage_pct': 0.001,
                'order_timeout_seconds': 60,
                'retry_attempts': 3
            },
            
            # Capital management
            'capital': {
                'require_account_ready': False,  # Allow trading without full account validation
                'cash_reserve_pct': 0.05,        # Reduced from 0.15 to allow more trading
                'max_single_position_pct': 0.20,  # Increased from 0.15
                'max_new_positions_per_day': 50,  # Increased from 20
                'min_account_value': 1000        # Minimum account value required
            },
            
            # Position sizing
            'sizing': {
                'min_position_usd': 500,         # Reduced from 1000
                'min_shares': 1,
                'max_position_usd': 10000,
                'position_size_method': 'risk_based'
            },
            
            # Timeout configurations
            'timeouts': {
                'account_refresh': 10.0,
                'position_request': 15.0,
                'order_placement': 30.0,
                'market_data': 5.0,
                'connection_timeout_seconds': 10
            }
        }
    
    def _load_environment(self):
        """Load configuration from environment variables"""
        try:
            # Load .env file if available
            try:
                import sys
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                sys.path.insert(0, parent_dir)
                from load_env import ensure_env_loaded
                ensure_env_loaded()
            except ImportError:
                self.logger.debug("load_env module not available, using system environment only")
            
            env_config = {}
            
            # IBKR Connection
            if os.getenv('TRADING_ACCOUNT_ID'):
                env_config.setdefault('ibkr', {})['account_id'] = os.getenv('TRADING_ACCOUNT_ID')
            if os.getenv('IBKR_HOST'):
                env_config.setdefault('ibkr', {})['host'] = os.getenv('IBKR_HOST')
            if os.getenv('IBKR_PORT'):
                env_config.setdefault('ibkr', {})['port'] = int(os.getenv('IBKR_PORT'))
            if os.getenv('CLIENT_ID'):
                env_config.setdefault('ibkr', {})['client_id'] = int(os.getenv('CLIENT_ID'))
            
            # Trading parameters
            trading_env_vars = {
                'DEFAULT_STOP_LOSS': 'default_stop_loss_pct',
                'DEFAULT_TAKE_PROFIT': 'default_take_profit_pct', 
                'ACCEPTANCE_THRESHOLD': 'acceptance_threshold',
                'MAX_POSITION_PCT': 'max_position_pct',
                'CASH_RESERVE_PCT': 'cash_reserve_pct',
                'MAX_DAILY_ORDERS': 'max_daily_orders',
                'PER_TRADE_RISK_PCT': 'per_trade_risk_pct'
            }
            
            for env_key, config_key in trading_env_vars.items():
                value = os.getenv(env_key)
                if value is not None:
                    env_config.setdefault('trading', {})
                    try:
                        if env_key in ['MAX_DAILY_ORDERS']:
                            env_config['trading'][config_key] = int(value)
                        else:
                            env_config['trading'][config_key] = float(value)
                    except ValueError:
                        self.logger.warning(f"Invalid value for {env_key}: {value}")
            
            # System settings
            if os.getenv('ENVIRONMENT'):
                env_config.setdefault('system', {})['environment'] = os.getenv('ENVIRONMENT')
            if os.getenv('DEMO_MODE'):
                env_config.setdefault('system', {})['demo_mode'] = os.getenv('DEMO_MODE').lower() in ('true', '1', 'yes')
            if os.getenv('LOG_LEVEL'):
                env_config.setdefault('system', {})['log_level'] = os.getenv('LOG_LEVEL')
            if os.getenv('DATA_SOURCE'):
                env_config.setdefault('system', {})['data_source'] = os.getenv('DATA_SOURCE')
            if os.getenv('ALLOW_FALLBACK_DATA'):
                env_config.setdefault('system', {})['allow_fallback_data'] = os.getenv('ALLOW_FALLBACK_DATA').lower() in ('true', '1', 'yes')
            
            # 🔐 Secure API key handling
            if os.getenv('POLYGON_API_KEY'):
                try:
                    from .secure_config import get_secure_config_manager
                    secure_config = get_secure_config_manager()
                    
                    api_key = os.getenv('POLYGON_API_KEY')
                    encrypted_key = secure_config.store_sensitive_config('data.polygon_api_key', api_key)
                    
                    env_config.setdefault('data', {})
                    env_config['data']['polygon_api_key_encrypted'] = encrypted_key
                    
                    self.logger.info("API key encrypted and stored securely")
                except Exception as e:
                    self.logger.error(f"Failed to encrypt API key: {e}")
                    # Fallback to storing in memory (still cleared from env)
                    env_config.setdefault('data', {})['polygon_api_key'] = os.getenv('POLYGON_API_KEY')
                    if 'POLYGON_API_KEY' in os.environ:
                        del os.environ['POLYGON_API_KEY']
            
            # Trading universe from file
            universe_file = os.getenv('TRADING_UNIVERSE_FILE')
            if universe_file and os.path.exists(universe_file):
                try:
                    with open(universe_file, 'r') as f:
                        universe = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
                    env_config.setdefault('scanner', {})['universe'] = universe
                    self.logger.info(f"Loaded trading universe from {universe_file}: {len(universe)} symbols")
                except Exception as e:
                    self.logger.error(f"Failed to load trading universe from {universe_file}: {e}")
            
            self._layers['environment'] = env_config
            
        except Exception as e:
            self.logger.error(f"Failed to load environment configuration: {e}")
            self._layers['environment'] = {}
    
    def _load_file_configs(self):
        """Load configuration from JSON files"""
        file_config = {}
        
        for config_name, config_path in self.config_files.items():
            if config_name == 'database':  # Skip database file here
                continue
                
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    
                    if config_name == 'main':
                        # Main config file - merge directly
                        file_config.update(config_data)
                    else:
                        # Scoped config files
                        file_config[config_name] = config_data
                    
                    self.logger.info(f"Loaded {config_name} configuration from {config_path}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load {config_name} config from {config_path}: {e}")
        
        self._layers['file_config'] = file_config
    
    def _load_database_config(self):
        """Load configuration from database"""
        db_config = {}
        
        db_path = self.config_files['database']
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Try to load from config table if it exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='config'")
                if cursor.fetchone():
                    cursor.execute("SELECT key, value, type FROM config WHERE enabled = 1")
                    for key, value, value_type in cursor.fetchall():
                        try:
                            if value_type == 'json':
                                parsed_value = json.loads(value)
                            elif value_type == 'int':
                                parsed_value = int(value)
                            elif value_type == 'float':
                                parsed_value = float(value)
                            elif value_type == 'bool':
                                parsed_value = value.lower() in ('true', '1', 'yes')
                            else:
                                parsed_value = value
                            
                            # Set nested keys using dot notation
                            self._set_nested_key(db_config, key, parsed_value)
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to parse database config {key}={value}: {e}")
                
                conn.close()
                self.logger.info("Loaded database configuration")
                
            except Exception as e:
                self.logger.error(f"Failed to load database configuration: {e}")
        
        self._layers['database'] = db_config
    
    def _set_nested_key(self, config_dict: dict, key_path: str, value: Any):
        """Set a nested configuration key using dot notation"""
        keys = key_path.split('.')
        current = config_dict
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _get_nested_key(self, config_dict: dict, key_path: str, default: Any = None):
        """Get a nested configuration key using dot notation"""
        keys = key_path.split('.')
        current = config_dict
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def _merge_configurations(self) -> Dict[str, Any]:
        """Merge all configuration layers according to priority"""
        if not self._cache_dirty and self._merged_cache is not None:
            return self._merged_cache
        
        with self._lock:
            merged = {}
            
            # Merge layers in priority order (lowest to highest)
            for layer_name in sorted(self._layers.keys(), key=lambda x: self.LAYER_PRIORITY.get(x, 0)):
                layer_config = self._layers[layer_name]
                if layer_config:
                    merged = self._deep_merge(merged, layer_config)
            
            self._merged_cache = merged
            self._cache_dirty = False
            self._last_merge_time = datetime.now().timestamp()
            
            self.logger.debug("Configuration layers merged successfully")
            return merged
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries"""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'ibkr.host')"""
        config = self._merge_configurations()
        return self._get_nested_key(config, key_path, default)
    
    def get_secure_api_key(self, service: str) -> Optional[str]:
        """🔐 Securely retrieve API key for service"""
        try:
            # Try encrypted key first
            encrypted_key = self.get(f'data.{service}_api_key_encrypted')
            if encrypted_key:
                from .secure_config import get_secure_config_manager
                secure_config = get_secure_config_manager()
                return secure_config.get_sensitive_config(encrypted_key)
            
            # Fallback to plain key (if still exists)
            plain_key = self.get(f'data.{service}_api_key')
            if plain_key:
                self.logger.warning(f"Using unencrypted API key for {service}")
                return plain_key
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve API key for {service}: {e}")
            return None
    
    def set_runtime(self, key_path: str, value: Any):
        """Set runtime configuration value"""
        with self._lock:
            old_value = self.get(key_path)
            self._set_nested_key(self._layers['runtime'], key_path, value)
            self._invalidate_cache()
            
            # Notify listeners
            self._notify_change_listeners(key_path, old_value, value)
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get the complete merged configuration"""
        return deepcopy(self._merge_configurations())
    
    def is_demo_mode(self) -> bool:
        """Check if running in demo mode"""
        return self.get('system.demo_mode', False)
    
    def is_production_mode(self) -> bool:
        """Check if running in production mode"""
        return not self.is_demo_mode() and self.get('system.signal_mode', 'testing') == 'production'
    
    def validate_config(self) -> ConfigValidationResult:
        """Validate the current configuration"""
        result = ConfigValidationResult(is_valid=True)
        config = self._merge_configurations()
        
        # Critical validations
        critical_checks = [
            ('ibkr.account_id', 'Trading account ID must be set'),
            ('ibkr.host', 'IBKR host must be set'),
            ('ibkr.port', 'IBKR port must be set'),
            ('system.environment', 'Environment must be set')
        ]
        
        for key_path, error_msg in critical_checks:
            value = self.get(key_path)
            if value is None or (isinstance(value, str) and not value.strip()):
                result.errors.append(f"{error_msg} ({key_path})")
                result.is_valid = False
        
        # Production environment validations
        if self.get('system.environment') == 'production':
            if not self.get('ibkr.account_id'):
                result.errors.append("Production environment requires trading account ID")
                result.is_valid = False
            
            if self.get('system.allow_fallback_data', False):
                result.warnings.append("Production environment allows fallback data - consider disabling")
        
        # Risk parameter validations
        risk_params = {
            'trading.max_position_pct': (0.0, 1.0),
            'trading.cash_reserve_pct': (0.0, 0.5),
            'risk.portfolio_exposure_limit': (0.0, 1.0),
            'trading.per_trade_risk_pct': (0.0, 0.1)
        }
        
        for key_path, (min_val, max_val) in risk_params.items():
            value = self.get(key_path)
            if value is not None and not (min_val <= value <= max_val):
                result.warnings.append(f"{key_path} = {value} is outside recommended range [{min_val}, {max_val}]")
        
        return result
    
    def add_change_listener(self, listener: Callable[[str, Any, Any], None]):
        """Add configuration change listener"""
        with self._lock:
            if listener not in self._change_listeners:
                self._change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[[str, Any, Any], None]):
        """Remove configuration change listener"""
        with self._lock:
            if listener in self._change_listeners:
                self._change_listeners.remove(listener)
    
    def _notify_change_listeners(self, key_path: str, old_value: Any, new_value: Any):
        """Notify all change listeners"""
        for listener in self._change_listeners:
            try:
                listener(key_path, old_value, new_value)
            except Exception as e:
                self.logger.error(f"Error in config change listener: {e}")
    
    def _invalidate_cache(self):
        """Mark configuration cache as dirty"""
        self._cache_dirty = True
    
    def reload(self):
        """Reload configuration from all sources"""
        with self._lock:
            self.logger.info("Reloading configuration...")
            self._load_environment()
            self._load_file_configs() 
            self._load_database_config()
            self._invalidate_cache()
    
    def save_runtime_to_file(self, file_path: Optional[Path] = None) -> bool:
        """Save runtime configuration changes to file"""
        try:
            save_path = file_path or self.config_files['main']
            runtime_config = self._layers.get('runtime', {})
            
            if not runtime_config:
                self.logger.info("No runtime configuration changes to save")
                return True
            
            # Load existing file config if it exists
            existing_config = {}
            if save_path.exists():
                with open(save_path, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)
            
            # Merge runtime changes into existing config
            merged_config = self._deep_merge(existing_config, runtime_config)
            
            # Save back to file
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(merged_config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Runtime configuration saved to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save runtime configuration: {e}")
            return False


# Global instance management
_global_config_manager: Optional[ConfigManager] = None
_config_lock = threading.RLock()


def get_config_manager(base_path: str = ".") -> ConfigManager:
    """Get the global configuration manager instance"""
    global _global_config_manager
    
    with _config_lock:
        if _global_config_manager is None:
            _global_config_manager = ConfigManager(base_path)
        return _global_config_manager


def reset_config_manager():
    """Reset the global configuration manager (for testing)"""
    global _global_config_manager
    with _config_lock:
        _global_config_manager = None


# Backward compatibility functions
def get_merged_config() -> Dict[str, Any]:
    """Get merged configuration (backward compatibility)"""
    return get_config_manager().get_full_config()


def get_connection_params() -> Dict[str, Any]:
    """Get IBKR connection parameters (backward compatibility)"""
    config = get_config_manager()
    return {
        'host': config.get('ibkr.host', '127.0.0.1'),
        'port': config.get('ibkr.port', 7497),
        'clientId': config.get('ibkr.client_id', 3130),
        'account': config.get('ibkr.account_id')
    }


def get_trading_universe() -> List[str]:
    """Get trading universe (backward compatibility)"""
    return get_config_manager().get('scanner.universe', ['SPY'])


def get_risk_params() -> Dict[str, Any]:
    """Get risk parameters (backward compatibility)"""
    config = get_config_manager()
    return {
        'max_position_pct': config.get('trading.max_position_pct', 0.15),
        'cash_reserve_pct': config.get('trading.cash_reserve_pct', 0.15), 
        'per_trade_risk_pct': config.get('trading.per_trade_risk_pct', 0.02),
        'portfolio_exposure_limit': config.get('risk.portfolio_exposure_limit', 0.85),
        'sector_exposure_limit': config.get('risk.sector_exposure_limit', 0.30)
    }


def get_signal_params() -> Dict[str, Any]:
    """Get signal parameters (backward compatibility)"""
    config = get_config_manager()
    return {
        'acceptance_threshold': config.get('signals.acceptance_threshold', 0.6),
        'min_confidence': config.get('signals.min_confidence', 0.5),
        'max_signal_age_minutes': config.get('signals.max_signal_age_minutes', 30)
    }


if __name__ == "__main__":
    # Testing
    logging.basicConfig(level=logging.INFO)
    
    config_manager = ConfigManager()
    
    print("=== Configuration Manager Test ===")
    print(f"IBKR Host: {config_manager.get('ibkr.host')}")
    print(f"Default Stop Loss: {config_manager.get('trading.default_stop_loss_pct')}")
    print(f"Environment: {config_manager.get('system.environment')}")
    print(f"Trading Universe: {config_manager.get('scanner.universe')}")
    
    # Test validation
    validation = config_manager.validate_config()
    print(f"\nValidation Result: {validation.is_valid}")
    if validation.errors:
        print(f"Errors: {validation.errors}")
    if validation.warnings:
        print(f"Warnings: {validation.warnings}")
    
    print("Configuration Manager test completed")