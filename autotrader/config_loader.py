#!/usr/bin/env python3
"""
Environment-based configuration loader
Replaces hardcoded values with environment variables
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Trading system configuration from environment"""
    
    # Connection settings
    account_id: str
    host: str
    port: int
    client_id: int
    
    # Trading parameters
    max_order_size: float
    default_stop_loss: float
    default_take_profit: float
    acceptance_threshold: float
    min_confidence_threshold: float
    
    # System settings
    signal_mode: str
    log_level: str
    data_source: str
    
    # Safety settings
    demo_mode: bool
    allow_random_signals: bool
    require_production_validation: bool
    
    # Risk management
    max_daily_orders: int
    max_position_pct: float
    sector_exposure_limit: float
    portfolio_exposure_limit: float

def load_config() -> TradingConfig:
    """Load configuration from environment variables"""
    
    # Validate critical settings
    account_id = os.getenv('TRADING_ACCOUNT_ID')
    if not account_id:
        raise ValueError("TRADING_ACCOUNT_ID environment variable is required")
    
    if account_id == 'your_account_id_here':
        raise ValueError("TRADING_ACCOUNT_ID must be set to a real account ID")
    
    return TradingConfig(
        # Connection settings
        account_id=account_id,
        host=os.getenv('IBKR_HOST', '127.0.0.1'),
        port=int(os.getenv('IBKR_PORT', '7497')),
        client_id=int(os.getenv('CLIENT_ID', '3130')),
        
        # Trading parameters
        max_order_size=float(os.getenv('MAX_ORDER_SIZE', '1000000')),
        default_stop_loss=float(os.getenv('DEFAULT_STOP_LOSS', '0.02')),
        default_take_profit=float(os.getenv('DEFAULT_TAKE_PROFIT', '0.05')),
        acceptance_threshold=float(os.getenv('ACCEPTANCE_THRESHOLD', '0.6')),
        min_confidence_threshold=float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.8')),
        
        # System settings
        signal_mode=os.getenv('SIGNAL_MODE', 'production'),
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        data_source=os.getenv('DATA_SOURCE', 'polygon'),
        
        # Safety settings
        demo_mode=os.getenv('DEMO_MODE', 'false').lower() == 'true',
        allow_random_signals=os.getenv('ALLOW_RANDOM_SIGNALS', 'false').lower() == 'true',
        require_production_validation=os.getenv('REQUIRE_PRODUCTION_VALIDATION', 'true').lower() == 'true',
        
        # Risk management
        max_daily_orders=int(os.getenv('MAX_DAILY_ORDERS', '20')),
        max_position_pct=float(os.getenv('MAX_POSITION_PCT', '0.15')),
        sector_exposure_limit=float(os.getenv('SECTOR_EXPOSURE_LIMIT', '0.30')),
        portfolio_exposure_limit=float(os.getenv('PORTFOLIO_EXPOSURE_LIMIT', '0.85'))
    )

def validate_config(config: TradingConfig) -> None:
    """Validate configuration for production use"""
    issues = []
    
    # Check for demo/test values
    if config.account_id in ['demo', 'test', 'your_account_id_here']:
        issues.append("Account ID appears to be a placeholder")
    
    if config.demo_mode and config.signal_mode == 'production':
        issues.append("Demo mode enabled with production signals")
    
    if config.allow_random_signals and config.signal_mode == 'production':
        issues.append("Random signals allowed in production mode")
    
    # Check risk limits
    if config.max_position_pct > 0.20:
        issues.append(f"Max position percentage too high: {config.max_position_pct}")
    
    if config.portfolio_exposure_limit > 1.0:
        issues.append(f"Portfolio exposure limit too high: {config.portfolio_exposure_limit}")
    
    if issues:
        raise ValueError(f"Configuration validation failed: {issues}")
    
    logger.info("Configuration validation passed")

# Global config instance
_global_config: Optional[TradingConfig] = None

def get_config() -> TradingConfig:
    """Get global configuration instance"""
    global _global_config
    
    if _global_config is None:
        _global_config = load_config()
        validate_config(_global_config)
        logger.info(f"Loaded configuration for account: {_global_config.account_id}")
    
    return _global_config
