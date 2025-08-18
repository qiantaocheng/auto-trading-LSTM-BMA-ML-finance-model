#!/usr/bin/env python3
"""
Unified Signal Processor - Single source of truth for trading signals
Replaces multiple conflicting signal generation functions
"""

import logging
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SignalMode(Enum):
    PRODUCTION = "production"
    TESTING = "testing"
    DEMO = "demo"

@dataclass
class SignalResult:
    symbol: str
    signal_value: float
    signal_strength: float
    confidence: float
    side: str
    can_trade: bool
    reason: str = ""
    source: str = ""
    timestamp: float = 0.0

class UnifiedSignalProcessor:
    """Unified signal processor - consolidates signal generation from multiple sources"""
    
    def __init__(self, mode: SignalMode = SignalMode.PRODUCTION):
        self.mode = mode
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Integration with environment config and data manager
        try:
            from .environment_config import get_environment_manager
            from .unified_data_manager import get_unified_data_manager
            from .unified_polygon_factors import get_unified_polygon_factors
            
            self.env_manager = get_environment_manager()
            self.data_manager = get_unified_data_manager()
            self.polygon_factors = get_unified_polygon_factors()
            
            self.logger.info(f"UnifiedSignalProcessor initialized in {mode.value} mode")
            
        except ImportError as e:
            self.logger.warning(f"Some signal components not available: {e}")
            self.env_manager = None
            self.data_manager = None
            self.polygon_factors = None
    
    def get_trading_signal(self, symbol: str, threshold: float = 0.3) -> SignalResult:
        """
        Get unified trading signal for a symbol - consolidates all signal sources
        
        This method replaces duplicate signal generation in:
        - autotrader/app.py (GUI signal generation)
        - autotrader/ibkr_auto_trader.py (trader signal generation) 
        - autotrader/unified_polygon_factors.py (factor signals)
        """
        import time
        timestamp = time.time()
        
        try:
            # Get environment-aware signal parameters
            if self.env_manager:
                signal_params = self.env_manager.get_signal_params()
                threshold = signal_params.get('acceptance_threshold', threshold)
                
                # Check if we're in demo mode
                if self.env_manager.is_demo_mode():
                    return self._get_demo_signal(symbol, timestamp)
            
            # Production signal generation using Polygon factors
            if self.polygon_factors:
                polygon_signal = self.polygon_factors.get_trading_signal(symbol, threshold=threshold)
                
                return SignalResult(
                    symbol=symbol,
                    signal_value=polygon_signal.get('signal_value', 0.0),
                    signal_strength=polygon_signal.get('signal_strength', 0.0),
                    confidence=polygon_signal.get('confidence', 0.0),
                    side=polygon_signal.get('side', 'HOLD'),
                    can_trade=polygon_signal.get('can_trade', False),
                    reason=polygon_signal.get('delay_reason', 'Signal calculated'),
                    source="UnifiedPolygonFactors",
                    timestamp=timestamp
                )
            
            # Fallback to basic signal calculation if Polygon factors not available
            return self._get_fallback_signal(symbol, threshold, timestamp)
            
        except Exception as e:
            self.logger.error(f"Signal generation error for {symbol}: {e}")
            return SignalResult(
                symbol=symbol,
                signal_value=0.0,
                signal_strength=0.0,
                confidence=0.0,
                side='HOLD',
                can_trade=False,
                reason=f"Error: {str(e)}",
                source="Error",
                timestamp=timestamp
            )
    
    def _get_demo_signal(self, symbol: str, timestamp: float) -> SignalResult:
        """Generate demo signal for testing"""
        return SignalResult(
            symbol=symbol,
            signal_value=0.0,
            signal_strength=0.0,
            confidence=0.0,
            side='HOLD',
            can_trade=False,
            reason="Demo mode - no real trading",
            source="Demo",
            timestamp=timestamp
        )
    
    def _get_fallback_signal(self, symbol: str, threshold: float, timestamp: float) -> SignalResult:
        """Fallback signal calculation when main systems unavailable"""
        try:
            if self.data_manager:
                # Get recent price data
                prices = self.data_manager.get_historical_prices(symbol, days=20)
                
                if len(prices) >= 10:
                    # Simple momentum signal
                    recent_avg = np.mean(prices[-5:])
                    older_avg = np.mean(prices[-10:-5]) if len(prices) >= 10 else recent_avg
                    
                    if older_avg > 0:
                        momentum = (recent_avg - older_avg) / older_avg
                        signal_strength = abs(momentum)
                        
                        if signal_strength >= threshold:
                            return SignalResult(
                                symbol=symbol,
                                signal_value=momentum,
                                signal_strength=signal_strength,
                                confidence=min(signal_strength * 2, 1.0),
                                side='BUY' if momentum > 0 else 'SELL',
                                can_trade=True,
                                reason="Fallback momentum signal",
                                source="Fallback",
                                timestamp=timestamp
                            )
            
            # No signal
            return SignalResult(
                symbol=symbol,
                signal_value=0.0,
                signal_strength=0.0,
                confidence=0.0,
                side='HOLD',
                can_trade=False,
                reason="Insufficient data for fallback signal",
                source="Fallback",
                timestamp=timestamp
            )
            
        except Exception as e:
            self.logger.error(f"Fallback signal error for {symbol}: {e}")
            return SignalResult(
                symbol=symbol,
                signal_value=0.0,
                signal_strength=0.0,
                confidence=0.0,
                side='HOLD',
                can_trade=False,
                reason=f"Fallback error: {str(e)}",
                source="FallbackError",
                timestamp=timestamp
            )

# Global unified signal processor
_signal_processor = None

def get_unified_signal_processor(mode: SignalMode = SignalMode.PRODUCTION) -> UnifiedSignalProcessor:
    """Get global unified signal processor"""
    global _signal_processor
    if _signal_processor is None:
        _signal_processor = UnifiedSignalProcessor(mode)
    return _signal_processor

def create_signal_processor(mode: SignalMode = SignalMode.PRODUCTION) -> UnifiedSignalProcessor:
    """Create new signal processor instance"""
    return UnifiedSignalProcessor(mode)

# Convenience function for backward compatibility
def get_trading_signal(symbol: str, threshold: float = 0.3) -> Dict[str, Any]:
    """Get trading signal (backward compatible)"""
    processor = get_unified_signal_processor()
    result = processor.get_trading_signal(symbol, threshold)
    
    # Convert to dictionary for backward compatibility
    return {
        'symbol': result.symbol,
        'signal_value': result.signal_value,
        'signal_strength': result.signal_strength,
        'confidence': result.confidence,
        'side': result.side,
        'can_trade': result.can_trade,
        'delay_reason': result.reason,
        'source': result.source,
        'timestamp': result.timestamp
    }