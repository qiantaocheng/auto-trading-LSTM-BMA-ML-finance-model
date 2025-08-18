#!/usr/bin/env python3
"""
Simple .env file loader for the trading system
Loads environment variables from .env file if present
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_dotenv(env_file: str = ".env") -> bool:
    """
    Load environment variables from .env file
    Returns True if file was loaded successfully
    """
    try:
        env_path = Path(env_file)
        if not env_path.exists():
            # Try relative to current file
            env_path = Path(__file__).parent / env_file
            if not env_path.exists():
                logger.debug(f"No .env file found at {env_file}")
                return False
        
        loaded_vars = 0
        with open(env_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value pairs
                if '=' not in line:
                    logger.warning(f"Invalid line {line_num} in {env_file}: {line}")
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                # Only set if not already in environment (environment takes precedence)
                if key not in os.environ:
                    os.environ[key] = value
                    loaded_vars += 1
                else:
                    logger.debug(f"Environment variable {key} already set, skipping .env value")
        
        logger.info(f"Loaded {loaded_vars} variables from {env_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading .env file {env_file}: {e}")
        return False

def ensure_env_loaded():
    """Ensure .env is loaded - call this early in application startup"""
    if not load_dotenv():
        logger.info("No .env file loaded - using system environment variables only")

if __name__ == "__main__":
    # Test the loader
    ensure_env_loaded()
    
    # Show some loaded variables
    test_vars = ['TRADING_ACCOUNT_ID', 'IBKR_HOST', 'IBKR_PORT', 'SIGNAL_MODE']
    print("Environment variables after loading:")
    for var in test_vars:
        value = os.getenv(var, 'NOT_SET')
        print(f"  {var} = {value}")