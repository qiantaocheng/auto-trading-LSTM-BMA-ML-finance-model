#!/usr/bin/env python3
"""
Environment Configuration Manager
Handles environment variables and production settings with proper fallbacks
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Load .env file if available
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from load_env import ensure_env_loaded
    ensure_env_loaded()
except ImportError:
    logger.debug("load_env module not available - using system environment only")

@dataclass
class EnvironmentConfig:
    """Environment-based configuration with validation"""
    
    # Critical trading settings
    trading_account_id: Optional[str] = None
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    client_id: int = 3130
    
    # Trading parameters (from environment or defaults)
    default_stop_loss: float = 0.02
    default_take_profit: float = 0.05
    max_position_pct: float = 0.15
    cash_reserve_pct: float = 0.15
    acceptance_threshold: float = 0.6
    
    # Risk management
    max_daily_orders: int = 20
    sector_exposure_limit: float = 0.30
    portfolio_exposure_limit: float = 0.85
    per_trade_risk_pct: float = 0.02
    
    # System settings
    signal_mode: str = "production"
    demo_mode: bool = False
    log_level: str = "INFO"
    data_source: str = "polygon"
    
    # Safety settings
    allow_random_signals: bool = False
    require_production_validation: bool = True
    
    # Data settings
    polygon_api_key: Optional[str] = None
    enable_delayed_data: bool = True
    data_delay_minutes: int = 15

class EnvironmentManager:
    """Manages environment variables and configuration validation"""
    
    def __init__(self):
        self.config = self._load_from_environment()
        self._validate_critical_settings()
    
    def _load_from_environment(self) -> EnvironmentConfig:
        """Load configuration from environment variables with fallbacks"""
        
        def get_env_float(key: str, default: float) -> float:
            try:
                return float(os.getenv(key, default))
            except (ValueError, TypeError):
                logger.warning(f"Invalid float value for {key}, using default: {default}")
                return default
        
        def get_env_int(key: str, default: int) -> int:
            try:
                return int(os.getenv(key, default))
            except (ValueError, TypeError):
                logger.warning(f"Invalid int value for {key}, using default: {default}")
                return default
        
        def get_env_bool(key: str, default: bool) -> bool:
            value = os.getenv(key, str(default)).lower()
            return value in ('true', '1', 'yes', 'on')
        
        return EnvironmentConfig(
            # Critical settings
            trading_account_id=os.getenv('TRADING_ACCOUNT_ID'),
            ibkr_host=os.getenv('IBKR_HOST', '127.0.0.1'),
            ibkr_port=get_env_int('IBKR_PORT', 7497),
            client_id=get_env_int('CLIENT_ID', 3130),
            
            # Trading parameters
            default_stop_loss=get_env_float('DEFAULT_STOP_LOSS', 0.02),
            default_take_profit=get_env_float('DEFAULT_TAKE_PROFIT', 0.05),
            max_position_pct=get_env_float('MAX_POSITION_PCT', 0.15),
            cash_reserve_pct=get_env_float('CASH_RESERVE_PCT', 0.15),
            acceptance_threshold=get_env_float('ACCEPTANCE_THRESHOLD', 0.6),
            
            # Risk management
            max_daily_orders=get_env_int('MAX_DAILY_ORDERS', 20),
            sector_exposure_limit=get_env_float('SECTOR_EXPOSURE_LIMIT', 0.30),
            portfolio_exposure_limit=get_env_float('PORTFOLIO_EXPOSURE_LIMIT', 0.85),
            per_trade_risk_pct=get_env_float('PER_TRADE_RISK_PCT', 0.02),
            
            # System settings
            signal_mode=os.getenv('SIGNAL_MODE', 'production'),
            demo_mode=get_env_bool('DEMO_MODE', False),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            data_source=os.getenv('DATA_SOURCE', 'polygon'),
            
            # Safety settings
            allow_random_signals=get_env_bool('ALLOW_RANDOM_SIGNALS', False),
            require_production_validation=get_env_bool('REQUIRE_PRODUCTION_VALIDATION', True),
            
            # Data settings
            polygon_api_key=os.getenv('POLYGON_API_KEY'),
            enable_delayed_data=get_env_bool('ENABLE_DELAYED_DATA', True),
            data_delay_minutes=get_env_int('DATA_DELAY_MINUTES', 15),
        )
    
    def _validate_critical_settings(self):
        """Validate critical settings required for production"""
        issues = []
        
        # Check trading account ID
        if not self.config.trading_account_id or self.config.trading_account_id in ['your_account_id_here', '${TRADING_ACCOUNT_ID}']:
            issues.append("TRADING_ACCOUNT_ID environment variable not set or contains placeholder")
        
        # Validate numeric ranges
        if not (0.001 <= self.config.default_stop_loss <= 0.1):
            issues.append(f"DEFAULT_STOP_LOSS ({self.config.default_stop_loss}) outside valid range (0.1% - 10%)")
            
        if not (0.01 <= self.config.max_position_pct <= 0.5):
            issues.append(f"MAX_POSITION_PCT ({self.config.max_position_pct}) outside valid range (1% - 50%)")
        
        # Check system settings
        if self.config.signal_mode not in ['production', 'testing', 'demo']:
            issues.append(f"Invalid SIGNAL_MODE: {self.config.signal_mode}")
        
        if issues:
            logger.warning("Configuration issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            
            # Only raise error in strict production mode with proper account setup
            if (self.config.require_production_validation and 
                self.config.signal_mode == 'production' and 
                not self.config.demo_mode and
                self.config.trading_account_id and 
                self.config.trading_account_id not in ['your_account_id_here', '${TRADING_ACCOUNT_ID}']):
                logger.error("Strict production validation enabled - stopping due to configuration issues")
                raise ValueError(f"Production validation failed: {len(issues)} critical issues found")
            else:
                logger.warning("Continuing despite configuration issues (validation relaxed for development/testing)")
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for IBKR"""
        return {
            'host': self.config.ibkr_host,
            'port': self.config.ibkr_port,
            'client_id': self.config.client_id,
            'account_id': self.config.trading_account_id,
            'timeout': 20.0,  # Keep existing timeout
        }
    
    def get_risk_params(self) -> Dict[str, Any]:
        """Get risk management parameters"""
        return {
            'per_trade_risk_pct': self.config.per_trade_risk_pct,
            'max_position_pct': self.config.max_position_pct,
            'cash_reserve_pct': self.config.cash_reserve_pct,
            'max_daily_orders': self.config.max_daily_orders,
            'sector_exposure_limit': self.config.sector_exposure_limit,
            'portfolio_exposure_limit': self.config.portfolio_exposure_limit,
            'default_stop_loss_pct': self.config.default_stop_loss,
            'default_take_profit_pct': self.config.default_take_profit,
        }
    
    def get_signal_params(self) -> Dict[str, Any]:
        """Get signal generation parameters"""
        return {
            'acceptance_threshold': self.config.acceptance_threshold,
            'signal_mode': self.config.signal_mode,
            'allow_random_signals': self.config.allow_random_signals,
            'enable_delayed_data': self.config.enable_delayed_data,
            'data_delay_minutes': self.config.data_delay_minutes,
        }
    
    def is_production_mode(self) -> bool:
        """Check if running in production mode"""
        return (self.config.signal_mode == 'production' and 
                not self.config.demo_mode and 
                not self.config.allow_random_signals)
    
    def is_demo_mode(self) -> bool:
        """Check if running in demo mode"""
        return self.config.demo_mode
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging"""
        return {
            'trading_mode': 'PRODUCTION' if self.is_production_mode() else 'DEMO',
            'account_configured': bool(self.config.trading_account_id),
            'host': self.config.ibkr_host,
            'port': self.config.ibkr_port,
            'data_source': self.config.data_source,
            'validation_enabled': self.config.require_production_validation,
        }

# Global instance
_env_manager = None

def get_environment_manager() -> EnvironmentManager:
    """Get global environment manager instance"""
    global _env_manager
    if _env_manager is None:
        _env_manager = EnvironmentManager()
    return _env_manager

def get_env_config() -> EnvironmentConfig:
    """Get environment configuration"""
    return get_environment_manager().config

# Convenience functions for backward compatibility
def get_connection_params() -> Dict[str, Any]:
    """Get connection parameters"""
    return get_environment_manager().get_connection_params()

def get_risk_params() -> Dict[str, Any]:
    """Get risk management parameters"""  
    return get_environment_manager().get_risk_params()

def get_signal_params() -> Dict[str, Any]:
    """Get signal parameters"""
    return get_environment_manager().get_signal_params()

def is_production_mode() -> bool:
    """Check if in production mode"""
    return get_environment_manager().is_production_mode()