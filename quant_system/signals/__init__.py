"""Signal generators module for technical analysis."""

from .trend_signals import (
    TrendTemplateSignal,
    ADXSignal,
    LinearRegressionSlopeSignal,
)
from .stage_analysis import StageAnalysisSignal
from .vcp_detector import VCPSignal
from .relative_strength import (
    RelativeStrengthSignal,
    UniverseRSCalculator,
)

__all__ = [
    'TrendTemplateSignal',
    'ADXSignal',
    'LinearRegressionSlopeSignal',
    'StageAnalysisSignal',
    'VCPSignal',
    'RelativeStrengthSignal',
    'UniverseRSCalculator',
]
