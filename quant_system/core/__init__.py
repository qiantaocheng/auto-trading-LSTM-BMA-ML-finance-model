"""Core module containing data types and signal aggregation."""

from .data_types import (
    SignalStrength,
    SignalResult,
    StockData,
    CompositeSignal,
    BaseSignalGenerator,
    BacktestResult,
)
from .signal_aggregator import (
    MarketRegimeDetector,
    SignalAggregator,
)

__all__ = [
    'SignalStrength',
    'SignalResult',
    'StockData',
    'CompositeSignal',
    'BaseSignalGenerator',
    'BacktestResult',
    'MarketRegimeDetector',
    'SignalAggregator',
]
