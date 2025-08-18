#!/usr/bin/env python3
"""
Engine Logger Compatibility Module
Provides logging functionality for the trading engine
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

def create_engine_logger(name: str, log_prefix: str = "engine") -> logging.Logger:
    """
    Create a logger for the trading engine
    
    Args:
        name: Logger name
        log_prefix: Prefix for log files
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    try:
        # Ensure logs directory exists
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamp for log file
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"{log_prefix}_{timestamp}.log")
        
        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info(f"Engine logger '{name}' initialized with log file: {log_file}")
        
    except Exception as e:
        # Fallback to basic console logging if file logging fails
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.warning(f"Failed to setup file logging, using console only: {e}")
    
    return logger

def get_engine_logger(name: str = "Engine") -> logging.Logger:
    """Get or create an engine logger"""
    return create_engine_logger(name, "engine")

def setup_trading_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """
    Setup comprehensive logging for the trading system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
    """
    
    try:
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Convert log level string to logging constant
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create main trading log file
        timestamp = datetime.now().strftime("%Y%m%d")
        main_log_file = os.path.join(log_dir, f"trading_{timestamp}.log")
        
        # Add file handler to root logger
        root_logger = logging.getLogger()
        file_handler = logging.FileHandler(main_log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Trading system logging initialized: {main_log_file}")
        
    except Exception as e:
        print(f"Warning: Failed to setup enhanced logging: {e}")
        # Fall back to basic logging
        logging.basicConfig(level=logging.INFO)

# Initialize basic logging when module is imported
if not logging.getLogger().handlers:
    setup_trading_logging()