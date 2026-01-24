"""
Signal Aggregator
================

Combines multiple signal sources into a unified composite score
with market regime awareness and dynamic weighting.

Features:
1. Weighted combination of technical, pattern, and fundamental signals
2. Market regime detection and adaptive weights
3. Signal confidence integration
4. Rank ordering across universe
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_types import (
    SignalResult, StockData, CompositeSignal, BaseSignalGenerator
)
from config.settings import (
    SystemConfig, SignalAggregationConfig, MarketRegime
)
from signals.trend_signals import TrendTemplateSignal, ADXSignal
from signals.stage_analysis import StageAnalysisSignal
from signals.vcp_detector import VCPSignal
from signals.relative_strength import RelativeStrengthSignal


@dataclass
class RegimeIndicators:
    """Market regime diagnostic indicators."""
    pct_above_200ma: float
    market_trend: str  # 'up', 'down', 'sideways'
    volatility_regime: str  # 'low', 'normal', 'high'
    breadth_score: float  # Advance/decline ratio
    regime: MarketRegime


class MarketRegimeDetector:
    """
    Detect current market regime for adaptive signal weighting.
    
    Uses breadth indicators and index analysis.
    """
    
    def __init__(self, index_data: Optional[pd.DataFrame] = None):
        self.index_data = index_data
        self._cache: Dict[str, RegimeIndicators] = {}
    
    def set_index_data(self, index_data: pd.DataFrame):
        """Set market index data."""
        self.index_data = index_data
    
    def detect_regime(self, 
                     universe_data: Optional[Dict[str, pd.DataFrame]] = None
                     ) -> RegimeIndicators:
        """
        Detect current market regime.
        
        Args:
            universe_data: Dict of ticker -> OHLCV for breadth calculation
            
        Returns:
            RegimeIndicators with current regime assessment
        """
        # Default values
        pct_above_200ma = 0.5
        market_trend = 'sideways'
        volatility_regime = 'normal'
        breadth_score = 0.5
        
        # Calculate breadth if universe data available
        if universe_data:
            pct_above_200ma = self._calc_pct_above_200ma(universe_data)
            breadth_score = pct_above_200ma
        
        # Analyze index trend if available
        if self.index_data is not None:
            market_trend = self._analyze_index_trend()
            volatility_regime = self._analyze_volatility()
        
        # Determine regime
        regime = self._classify_regime(pct_above_200ma, market_trend, volatility_regime)
        
        return RegimeIndicators(
            pct_above_200ma=pct_above_200ma,
            market_trend=market_trend,
            volatility_regime=volatility_regime,
            breadth_score=breadth_score,
            regime=regime
        )
    
    def _calc_pct_above_200ma(self, universe_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate percentage of stocks above their 200-day MA."""
        above_count = 0
        total_count = 0
        
        for ticker, df in universe_data.items():
            if len(df) < 200:
                continue
            
            close = df['Close']
            ma200 = close.rolling(200).mean()
            
            if not pd.isna(ma200.iloc[-1]):
                total_count += 1
                if close.iloc[-1] > ma200.iloc[-1]:
                    above_count += 1
        
        return above_count / total_count if total_count > 0 else 0.5
    
    def _analyze_index_trend(self) -> str:
        """Analyze market index trend."""
        if self.index_data is None or len(self.index_data) < 200:
            return 'sideways'
        
        close = self.index_data['Close']
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        
        current = close.iloc[-1]
        
        # Check MA relationships
        if current > ma50.iloc[-1] > ma200.iloc[-1]:
            # Check slope of 200MA
            ma200_slope = (ma200.iloc[-1] - ma200.iloc[-20]) / ma200.iloc[-20]
            if ma200_slope > 0.01:
                return 'up'
            else:
                return 'sideways'
        elif current < ma50.iloc[-1] < ma200.iloc[-1]:
            return 'down'
        else:
            return 'sideways'
    
    def _analyze_volatility(self) -> str:
        """Analyze market volatility regime."""
        if self.index_data is None or len(self.index_data) < 252:
            return 'normal'
        
        close = self.index_data['Close']
        returns = close.pct_change()
        
        # Calculate 20-day realized volatility
        current_vol = returns.iloc[-20:].std() * np.sqrt(252)
        
        # Calculate historical volatility distribution
        hist_vol = returns.rolling(20).std() * np.sqrt(252)
        hist_vol = hist_vol.dropna()
        
        if len(hist_vol) < 100:
            return 'normal'
        
        percentile = (hist_vol < current_vol).mean()
        
        if percentile < 0.25:
            return 'low'
        elif percentile > 0.75:
            return 'high'
        else:
            return 'normal'
    
    def _classify_regime(self, pct_above_200ma: float, 
                        market_trend: str,
                        volatility_regime: str) -> MarketRegime:
        """Classify into one of five market regimes."""
        
        # Primary classification based on breadth
        if pct_above_200ma >= 0.80:
            if market_trend == 'up':
                return MarketRegime.BULL_STRONG
            else:
                return MarketRegime.BULL_WEAK
        elif pct_above_200ma >= 0.50:
            if market_trend == 'up':
                return MarketRegime.BULL_WEAK
            else:
                return MarketRegime.NEUTRAL
        elif pct_above_200ma >= 0.30:
            if market_trend == 'down':
                return MarketRegime.BEAR_WEAK
            else:
                return MarketRegime.NEUTRAL
        elif pct_above_200ma >= 0.20:
            return MarketRegime.BEAR_WEAK
        else:
            return MarketRegime.BEAR_STRONG


class SignalAggregator:
    """
    Aggregates multiple signals into composite scores.
    
    Handles:
    1. Signal generation from all sources
    2. Regime-adaptive weighting
    3. Confidence-weighted combination
    4. Universe ranking
    """
    
    def __init__(self, config: Optional[SystemConfig] = None,
                 benchmark_data: Optional[pd.DataFrame] = None):
        self.config = config or SystemConfig()
        self.benchmark_data = benchmark_data
        
        # Initialize signal generators
        self._init_signal_generators()
        
        # Regime detector
        self.regime_detector = MarketRegimeDetector(benchmark_data)
        
        # Cache
        self._signal_cache: Dict[str, Dict[str, SignalResult]] = {}
    
    def _init_signal_generators(self):
        """Initialize all signal generator instances."""
        cfg = self.config
        
        self.trend_signal = TrendTemplateSignal(cfg.trend_template)
        self.adx_signal = ADXSignal(cfg.adx)
        self.stage_signal = StageAnalysisSignal(cfg.stage_analysis, self.benchmark_data)
        self.vcp_signal = VCPSignal(cfg.vcp)
        self.rs_signal = RelativeStrengthSignal(cfg.relative_strength, self.benchmark_data)
    
    def set_benchmark(self, benchmark_data: pd.DataFrame):
        """Update benchmark data for all signals."""
        self.benchmark_data = benchmark_data
        self.stage_signal.set_benchmark(benchmark_data)
        self.rs_signal.set_benchmark(benchmark_data)
        self.regime_detector.set_index_data(benchmark_data)
    
    def generate_composite_signal(self, data: StockData,
                                  regime: Optional[MarketRegime] = None
                                  ) -> CompositeSignal:
        """
        Generate composite signal for a single stock.
        
        Args:
            data: StockData object
            regime: Optional pre-computed market regime
            
        Returns:
            CompositeSignal with all scores
        """
        # Generate individual signals
        signals: Dict[str, SignalResult] = {}
        
        signals['trend_template'] = self.trend_signal.generate(data)
        signals['adx'] = self.adx_signal.generate(data)
        signals['stage_analysis'] = self.stage_signal.generate(data)
        signals['vcp'] = self.vcp_signal.generate(data)
        signals['relative_strength'] = self.rs_signal.generate(data)
        
        # Get regime adjustments
        if regime is None:
            regime = MarketRegime.NEUTRAL  # Default if not provided
        
        regime_adj = self.config.aggregation.regime_adjustments.get(
            regime.value,
            {'technical': 1.0, 'pattern': 1.0, 'fundamental': 1.0}
        )
        
        # Calculate sub-scores
        technical_score = self._calc_technical_score(signals, regime_adj['technical'])
        pattern_score = self._calc_pattern_score(signals, regime_adj['pattern'])
        fundamental_score = self._calc_fundamental_score(signals, regime_adj['fundamental'])
        
        # Composite score
        agg_cfg = self.config.aggregation
        
        # Normalize regime-adjusted weights
        total_weight = (
            agg_cfg.technical_weight * regime_adj['technical'] +
            agg_cfg.pattern_weight * regime_adj['pattern'] +
            agg_cfg.fundamental_weight * regime_adj['fundamental']
        )
        
        composite_score = (
            agg_cfg.technical_weight * regime_adj['technical'] * technical_score +
            agg_cfg.pattern_weight * regime_adj['pattern'] * pattern_score +
            agg_cfg.fundamental_weight * regime_adj['fundamental'] * fundamental_score
        ) / total_weight if total_weight > 0 else 0.5
        
        return CompositeSignal(
            ticker=data.ticker,
            composite_score=composite_score,
            technical_score=technical_score,
            pattern_score=pattern_score,
            fundamental_score=fundamental_score,
            signals=signals,
            regime=regime.value if regime else None,
            timestamp=datetime.now()
        )
    
    def _calc_technical_score(self, signals: Dict[str, SignalResult],
                              regime_multiplier: float) -> float:
        """Calculate technical sub-score."""
        cfg = self.config.aggregation
        
        trend_score = signals['trend_template'].score
        adx_score = signals['adx'].score
        rs_score = signals['relative_strength'].score
        
        # Confidence weighting
        trend_conf = signals['trend_template'].confidence
        adx_conf = signals['adx'].confidence
        rs_conf = signals['relative_strength'].confidence
        
        # Weighted average with confidence
        weighted_sum = (
            cfg.trend_template_weight * trend_score * trend_conf +
            cfg.adx_weight * adx_score * adx_conf +
            cfg.relative_strength_weight * rs_score * rs_conf
        )
        
        weight_sum = (
            cfg.trend_template_weight * trend_conf +
            cfg.adx_weight * adx_conf +
            cfg.relative_strength_weight * rs_conf
        )
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.5
    
    def _calc_pattern_score(self, signals: Dict[str, SignalResult],
                           regime_multiplier: float) -> float:
        """Calculate pattern recognition sub-score."""
        cfg = self.config.aggregation
        
        vcp_score = signals['vcp'].score
        stage_score = signals['stage_analysis'].score
        
        vcp_conf = signals['vcp'].confidence
        stage_conf = signals['stage_analysis'].confidence
        
        weighted_sum = (
            cfg.vcp_weight * vcp_score * vcp_conf +
            cfg.stage_weight * stage_score * stage_conf
        )
        
        weight_sum = (
            cfg.vcp_weight * vcp_conf +
            cfg.stage_weight * stage_conf
        )
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.5
    
    def _calc_fundamental_score(self, signals: Dict[str, SignalResult],
                               regime_multiplier: float) -> float:
        """
        Calculate fundamental sub-score.
        
        Note: Fundamental data integration would require external data sources.
        For this implementation, we use RS as a proxy (strong stocks tend to 
        have strong fundamentals).
        """
        # Use RS as fundamental proxy + stage analysis fundamentals
        rs_score = signals['relative_strength'].score
        stage_score = signals['stage_analysis'].score
        
        # RS captures institutional buying, which correlates with fundamentals
        return 0.6 * rs_score + 0.4 * stage_score
    
    def scan_universe(self, 
                     universe_data: Dict[str, pd.DataFrame],
                     min_score: Optional[float] = None
                     ) -> List[CompositeSignal]:
        """
        Scan entire universe and return ranked signals.
        
        Args:
            universe_data: Dict mapping ticker -> OHLCV DataFrame
            min_score: Optional minimum composite score filter
            
        Returns:
            List of CompositeSignal objects, sorted by score descending
        """
        if min_score is None:
            min_score = self.config.aggregation.min_composite_score
        
        # Detect market regime
        regime_indicators = self.regime_detector.detect_regime(universe_data)
        regime = regime_indicators.regime
        
        results: List[CompositeSignal] = []
        
        for ticker, df in universe_data.items():
            try:
                stock_data = StockData(ticker=ticker, df=df)
                
                # Check minimum data requirement
                if len(df) < 252:  # Need 1 year minimum
                    continue
                
                signal = self.generate_composite_signal(stock_data, regime)
                
                if signal.composite_score >= min_score:
                    results.append(signal)
                    
            except Exception as e:
                # Log error but continue scanning
                print(f"Error processing {ticker}: {str(e)}")
                continue
        
        # Sort by composite score
        results.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Add rank
        for i, signal in enumerate(results):
            signal.rank = i + 1
        
        return results
    
    def generate_signal_report(self, signal: CompositeSignal) -> str:
        """Generate human-readable signal report."""
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SIGNAL REPORT: {signal.ticker:<10}                                            ║
║  Generated: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  COMPOSITE SCORE:  {signal.composite_score:>6.2f}  {'★' * int(signal.composite_score * 5):<5}                                 ║
║  Rank:             #{signal.rank if signal.rank else 'N/A':<5}                                             ║
║  Market Regime:    {signal.regime or 'Unknown':<15}                                    ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SUB-SCORES                                                                  ║
║  ├─ Technical:     {signal.technical_score:>6.2f}                                             ║
║  ├─ Pattern:       {signal.pattern_score:>6.2f}                                             ║
║  └─ Fundamental:   {signal.fundamental_score:>6.2f}                                             ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SIGNAL BREAKDOWN                                                            ║
"""
        
        for name, sig in signal.signals.items():
            strength_bar = '█' * int(sig.score * 10) + '░' * (10 - int(sig.score * 10))
            report += f"║  {name:<20} {sig.score:>5.2f}  [{strength_bar}]  ({sig.strength.name})     ║\n"
        
        report += """║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        
        return report


def create_default_aggregator(benchmark_data: Optional[pd.DataFrame] = None
                              ) -> SignalAggregator:
    """Factory function to create aggregator with default settings."""
    return SignalAggregator(SystemConfig(), benchmark_data)
