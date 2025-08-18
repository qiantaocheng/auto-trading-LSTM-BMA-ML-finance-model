#!/usr/bin/env python3
"""
Unified Trading Engine
Single source of truth for all trading operations
Replaces conflicting logic across multiple files
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .config_loader import get_config, TradingConfig
from .unified_signal_processor import get_unified_signal_processor, SignalMode, SignalResult
from .delayed_data_config import should_trade_with_delayed_data, DEFAULT_DELAYED_CONFIG
from .input_validator import InputValidator

logger = logging.getLogger(__name__)

@dataclass
class TradingDecision:
    """Unified trading decision"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: int
    price: Optional[float]
    confidence: float
    reason: str
    risk_approved: bool
    timestamp: datetime

class UnifiedTradingEngine:
    """Unified trading engine - single source of trading decisions"""
    
    def __init__(self):
        self.config = get_config()
        self.signal_processor = get_unified_signal_processor(
            SignalMode(self.config.signal_mode)
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Safety checks
        if self.config.demo_mode:
            self.logger.warning("DEMO MODE ENABLED - No real trading")
        
        if self.config.require_production_validation:
            self._validate_production_readiness()
    
    def _validate_production_readiness(self):
        """Validate system is ready for production trading"""
        issues = []
        
        # Check signal mode
        if self.config.signal_mode not in ['production', 'testing']:
            issues.append(f"Invalid signal mode: {self.config.signal_mode}")
        
        # Check for safety flags
        if self.config.allow_random_signals and self.config.signal_mode == 'production':
            issues.append("Random signals enabled in production")
        
        if issues:
            raise RuntimeError(f"Production validation failed: {issues}")
        
        self.logger.info("Production validation passed")
    
    async def get_trading_decisions(self, symbols: List[str]) -> List[TradingDecision]:
        """
        Get trading decisions for a list of symbols
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            List of trading decisions
        """
        decisions = []
        
        for symbol in symbols:
            try:
                decision = await self._get_single_trading_decision(symbol)
                if decision:
                    decisions.append(decision)
            except Exception as e:
                self.logger.error(f"Error getting decision for {symbol}: {e}")
        
        return decisions
    
    async def _get_single_trading_decision(self, symbol: str) -> Optional[TradingDecision]:
        """Get trading decision for a single symbol"""
        
        # Step 1: Validate symbol
        try:
            validated_symbol = InputValidator.validate_symbol(symbol)
        except Exception as e:
            self.logger.warning(f"Invalid symbol {symbol}: {e}")
            return None
        
        # Step 2: Check delayed data trading window
        can_trade_delayed, delay_reason = should_trade_with_delayed_data(DEFAULT_DELAYED_CONFIG)
        if not can_trade_delayed:
            self.logger.debug(f"Delayed trading not allowed for {symbol}: {delay_reason}")
            return None
        
        # Step 3: Get trading signal
        signal_result = self.signal_processor.get_trading_signal(
            validated_symbol, 
            self.config.acceptance_threshold
        )
        
        # Step 4: Risk assessment
        if not signal_result.can_trade:
            return TradingDecision(
                symbol=validated_symbol,
                action="HOLD",
                quantity=0,
                price=None,
                confidence=signal_result.confidence,
                reason=f"Signal not tradeable: {signal_result.reason}",
                risk_approved=False,
                timestamp=datetime.now()
            )
        
        # Step 5: Determine action and quantity
        action = signal_result.side
        if action not in ["BUY", "SELL"]:
            action = "HOLD"
        
        # Calculate position size (simplified)
        if action in ["BUY", "SELL"]:
            quantity = self._calculate_position_size(signal_result)
        else:
            quantity = 0
        
        return TradingDecision(
            symbol=validated_symbol,
            action=action,
            quantity=quantity,
            price=None,  # Market price
            confidence=signal_result.confidence,
            reason=f"Signal: {signal_result.signal_strength:.3f}, Source: {signal_result.source}",
            risk_approved=True,
            timestamp=datetime.now()
        )
    
    def _calculate_position_size(self, signal: SignalResult) -> int:
        """Calculate position size based on signal and risk parameters"""
        # Simplified position sizing - should be enhanced
        base_size = 100  # Base position size
        
        # Adjust for signal strength
        size_multiplier = min(signal.signal_strength * 2, 1.0)
        
        # Adjust for confidence
        confidence_multiplier = signal.confidence
        
        # Apply delayed data reduction if applicable
        if DEFAULT_DELAYED_CONFIG.enabled:
            from .delayed_data_config import get_position_size_multiplier
            size_multiplier *= get_position_size_multiplier(DEFAULT_DELAYED_CONFIG)
        
        final_size = int(base_size * size_multiplier * confidence_multiplier)
        
        # Ensure minimum position
        return max(final_size, 1) if final_size > 0 else 0

# Global engine instance
_global_engine: Optional[UnifiedTradingEngine] = None

def get_unified_trading_engine() -> UnifiedTradingEngine:
    """Get global unified trading engine"""
    global _global_engine
    
    if _global_engine is None:
        _global_engine = UnifiedTradingEngine()
        logger.info("Created unified trading engine")
    
    return _global_engine

async def get_trading_decisions(symbols: List[str]) -> List[TradingDecision]:
    """Convenience function for getting trading decisions"""
    engine = get_unified_trading_engine()
    return await engine.get_trading_decisions(symbols)
