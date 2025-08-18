#!/usr/bin/env python3
"""
Delayed Data Configuration
Handles configuration for trading with delayed market data
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Tuple

logger = logging.getLogger(__name__)

@dataclass
class DelayedDataConfig:
    """Configuration for delayed data trading"""
    enabled: bool = True
    data_delay_minutes: int = 15
    min_confidence_threshold: float = 0.8
    position_size_reduction: float = 0.4
    trading_start_time: time = time(9, 30)  # 9:30 AM
    trading_end_time: time = time(16, 0)    # 4:00 PM
    avoid_first_minutes: int = 30           # Avoid first 30 minutes
    avoid_last_minutes: int = 30            # Avoid last 30 minutes
    min_alpha_multiplier: float = 1.0       # Alpha adjustment for delayed data

DEFAULT_DELAYED_CONFIG = DelayedDataConfig()

def should_trade_with_delayed_data(config: DelayedDataConfig) -> Tuple[bool, str]:
    """
    Determine if trading should occur with delayed data
    
    Returns:
        Tuple[bool, str]: (can_trade, reason)
    """
    if not config.enabled:
        return False, "Delayed data trading disabled"
    
    now = datetime.now().time()
    
    # Check if within trading hours
    if now < config.trading_start_time or now > config.trading_end_time:
        return False, "Outside trading hours"
    
    # Avoid first minutes of trading
    start_buffer = time(
        config.trading_start_time.hour,
        config.trading_start_time.minute + config.avoid_first_minutes
    )
    if now < start_buffer:
        return False, f"Within first {config.avoid_first_minutes} minutes of trading"
    
    # Avoid last minutes of trading
    end_buffer = time(
        config.trading_end_time.hour,
        config.trading_end_time.minute - config.avoid_last_minutes
    )
    if now > end_buffer:
        return False, f"Within last {config.avoid_last_minutes} minutes of trading"
    
    return True, "Delayed data trading allowed"

def get_position_size_multiplier(config: DelayedDataConfig) -> float:
    """Get position size multiplier for delayed data"""
    if not config.enabled:
        return 1.0
    
    return 1.0 - config.position_size_reduction
