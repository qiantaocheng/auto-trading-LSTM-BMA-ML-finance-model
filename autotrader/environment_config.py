"""
Environment configuration management for autotrader
"""
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Manages environment configuration settings"""
    
    def __init__(self):
        self.config = self._load_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default environment configuration"""
        return {
            'trading_environment': os.getenv('TRADING_ENV', 'paper'),
            'data_provider': os.getenv('DATA_PROVIDER', 'polygon'),
            'risk_limits': {
                'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '0.05')),
                'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '0.02')),
            },
            'logging_level': os.getenv('LOG_LEVEL', 'INFO'),
        }
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value
        logger.info(f"Environment config updated: {key} = {value}")


# Global environment manager instance
_environment_manager = None


def get_environment_manager() -> EnvironmentManager:
    """Get global environment manager instance"""
    global _environment_manager
    if _environment_manager is None:
        _environment_manager = EnvironmentManager()
    return _environment_manager