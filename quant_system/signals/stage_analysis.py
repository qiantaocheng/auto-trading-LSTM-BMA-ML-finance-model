"""
Weinstein Stage Analysis Implementation
=======================================

Algorithmic detection of market stages:
- Stage 1: Basing/Accumulation
- Stage 2: Advancing (Primary Uptrend)
- Stage 3: Topping/Distribution  
- Stage 4: Declining (Primary Downtrend)

The key signal is detecting Stage 2 breakouts - the transition from
Stage 1 to Stage 2 which marks the beginning of major advances.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from enum import Enum
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/claude/quant_system')

from core.data_types import BaseSignalGenerator, SignalResult, StockData
from config.settings import StageAnalysisConfig


class Stage(Enum):
    """Weinstein's four market stages."""
    STAGE_1 = 1  # Basing/Accumulation
    STAGE_2 = 2  # Advancing
    STAGE_3 = 3  # Topping/Distribution
    STAGE_4 = 4  # Declining
    UNKNOWN = 0


@dataclass
class StageMetrics:
    """Metrics used for stage determination."""
    stage: Stage
    confidence: float
    ma_slope: float
    price_vs_ma: float  # % above/below MA
    volume_pattern: str  # 'accumulation', 'distribution', 'neutral'
    mansfield_rs: float
    weeks_in_stage: int
    breakout_detected: bool
    breakout_volume_ratio: Optional[float] = None


class StageAnalysisSignal(BaseSignalGenerator):
    """
    Weinstein Stage Analysis Signal Generator
    
    Focuses on identifying:
    1. Current market stage
    2. Stage 2 breakout signals
    3. Mansfield Relative Strength
    4. Volume confirmation patterns
    """
    
    def __init__(self, config: Optional[StageAnalysisConfig] = None,
                 benchmark_data: Optional[pd.DataFrame] = None):
        """
        Args:
            config: Stage analysis configuration
            benchmark_data: Index data for RS calculation (e.g., SPY)
        """
        self.config = config or StageAnalysisConfig()
        self.benchmark_data = benchmark_data
        self._cache = {}
    
    def set_benchmark(self, benchmark_data: pd.DataFrame):
        """Set benchmark data for relative strength calculation."""
        self.benchmark_data = benchmark_data
    
    def get_required_lookback(self) -> int:
        """Need ~2 years of weekly data for RS calculation."""
        return self.config.rs_ma_period * 5 + 20  # Weekly periods converted to daily
    
    def generate(self, data: StockData) -> SignalResult:
        """Generate stage analysis signal."""
        if not self.validate_data(data):
            return SignalResult.from_score(
                name='stage_analysis',
                score=0.5,
                confidence=0.1,
                metadata={'error': 'Insufficient data'}
            )
        
        # Convert to weekly data for stage analysis
        weekly_df = self._to_weekly(data.df)
        
        # Calculate 30-week MA (Weinstein's standard)
        ma_30w = weekly_df['Close'].rolling(window=self.config.ma_period_weeks).mean()
        
        # Determine current stage
        stage_metrics = self._determine_stage(weekly_df, ma_30w)
        
        # Calculate stage-based score
        stage_score = self._score_stage(stage_metrics)
        
        # Check for Stage 2 breakout
        breakout_score = self._check_stage2_breakout(weekly_df, ma_30w)
        
        # Calculate Mansfield RS if benchmark available
        rs_score = self._calculate_rs_score(data.df)
        
        # Composite score
        composite_score = (
            0.40 * stage_score +
            0.35 * breakout_score +
            0.25 * rs_score
        )
        
        components = {
            'stage_score': stage_score,
            'breakout_score': breakout_score,
            'rs_score': rs_score
        }
        
        metadata = {
            'current_stage': stage_metrics.stage.name,
            'stage_confidence': stage_metrics.confidence,
            'ma_slope': stage_metrics.ma_slope,
            'price_vs_ma_pct': stage_metrics.price_vs_ma,
            'volume_pattern': stage_metrics.volume_pattern,
            'mansfield_rs': stage_metrics.mansfield_rs,
            'weeks_in_stage': stage_metrics.weeks_in_stage,
            'breakout_detected': stage_metrics.breakout_detected
        }
        
        if stage_metrics.breakout_volume_ratio is not None:
            metadata['breakout_volume_ratio'] = stage_metrics.breakout_volume_ratio
        
        return SignalResult.from_score(
            name='stage_analysis',
            score=composite_score,
            confidence=stage_metrics.confidence,
            components=components,
            metadata=metadata
        )
    
    def _to_weekly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Convert daily OHLCV to weekly."""
        weekly = daily_df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        return weekly
    
    def _determine_stage(self, weekly_df: pd.DataFrame, 
                         ma_30w: pd.Series) -> StageMetrics:
        """
        Determine current Weinstein stage based on:
        1. Price position relative to 30-week MA
        2. MA slope direction
        3. Volume patterns
        """
        if len(weekly_df) < self.config.ma_period_weeks + 10:
            return StageMetrics(
                stage=Stage.UNKNOWN,
                confidence=0.0,
                ma_slope=0.0,
                price_vs_ma=0.0,
                volume_pattern='unknown',
                mansfield_rs=0.0,
                weeks_in_stage=0,
                breakout_detected=False
            )
        
        close = weekly_df['Close']
        volume = weekly_df['Volume']
        
        latest_close = close.iloc[-1]
        latest_ma = ma_30w.iloc[-1]
        
        # Calculate MA slope (over past 4 weeks)
        ma_slope = (ma_30w.iloc[-1] - ma_30w.iloc[-5]) / ma_30w.iloc[-5] if ma_30w.iloc[-5] > 0 else 0
        
        # Price position relative to MA
        price_vs_ma = (latest_close - latest_ma) / latest_ma if latest_ma > 0 else 0
        
        # Volume pattern analysis
        volume_pattern = self._analyze_volume_pattern(weekly_df)
        
        # Mansfield RS calculation
        mansfield_rs = self._calculate_mansfield_rs(weekly_df)
        
        # Stage determination logic
        stage, confidence = self._classify_stage(
            price_vs_ma, ma_slope, volume_pattern, close, ma_30w
        )
        
        # Count weeks in current stage
        weeks_in_stage = self._count_stage_duration(close, ma_30w, stage)
        
        # Check for breakout
        breakout_detected, volume_ratio = self._detect_breakout(weekly_df, ma_30w)
        
        return StageMetrics(
            stage=stage,
            confidence=confidence,
            ma_slope=ma_slope,
            price_vs_ma=price_vs_ma,
            volume_pattern=volume_pattern,
            mansfield_rs=mansfield_rs,
            weeks_in_stage=weeks_in_stage,
            breakout_detected=breakout_detected,
            breakout_volume_ratio=volume_ratio
        )
    
    def _classify_stage(self, price_vs_ma: float, ma_slope: float,
                        volume_pattern: str, close: pd.Series,
                        ma_30w: pd.Series) -> Tuple[Stage, float]:
        """
        Classify into one of four stages based on multiple factors.
        
        Returns: (Stage, confidence)
        """
        # Stage classification rules
        
        # STAGE 2: Advancing
        # - Price above MA
        # - MA slope positive
        # - Accumulation volume pattern preferred
        if price_vs_ma > 0 and ma_slope > 0.005:
            if volume_pattern in ['accumulation', 'neutral']:
                return Stage.STAGE_2, min(0.9, 0.5 + price_vs_ma + ma_slope * 10)
            else:
                return Stage.STAGE_2, 0.6
        
        # STAGE 4: Declining
        # - Price below MA
        # - MA slope negative
        elif price_vs_ma < 0 and ma_slope < -0.005:
            return Stage.STAGE_4, min(0.9, 0.5 + abs(price_vs_ma) + abs(ma_slope) * 10)
        
        # STAGE 1: Basing (MA flat, price oscillating around it)
        elif abs(ma_slope) < 0.01 and abs(price_vs_ma) < 0.05:
            return Stage.STAGE_1, 0.6
        
        # STAGE 3: Topping (MA flattening from uptrend)
        elif price_vs_ma > 0 and ma_slope < 0.01 and ma_slope >= 0:
            if volume_pattern == 'distribution':
                return Stage.STAGE_3, 0.7
            else:
                return Stage.STAGE_3, 0.5
        
        # Default: try to infer
        else:
            if price_vs_ma > 0:
                return Stage.STAGE_2, 0.4
            elif price_vs_ma < 0:
                return Stage.STAGE_4, 0.4
            else:
                return Stage.STAGE_1, 0.3
    
    def _analyze_volume_pattern(self, weekly_df: pd.DataFrame) -> str:
        """
        Analyze volume pattern to detect accumulation vs distribution.
        
        Accumulation: Up weeks have higher volume than down weeks
        Distribution: Down weeks have higher volume than up weeks
        """
        recent = weekly_df.iloc[-13:]  # Last quarter
        
        if len(recent) < 4:
            return 'neutral'
        
        # Classify up vs down weeks
        returns = recent['Close'].pct_change()
        
        up_weeks = recent[returns > 0]
        down_weeks = recent[returns < 0]
        
        if len(up_weeks) == 0 or len(down_weeks) == 0:
            return 'neutral'
        
        avg_up_volume = up_weeks['Volume'].mean()
        avg_down_volume = down_weeks['Volume'].mean()
        
        ratio = avg_up_volume / avg_down_volume if avg_down_volume > 0 else 1
        
        if ratio > 1.2:
            return 'accumulation'
        elif ratio < 0.8:
            return 'distribution'
        else:
            return 'neutral'
    
    def _calculate_mansfield_rs(self, weekly_df: pd.DataFrame) -> float:
        """
        Calculate Mansfield Relative Strength.
        
        RS = (Stock Price / Index Price) / 52-week MA of ratio - 1
        
        Returns 0 if no benchmark data available.
        """
        if self.benchmark_data is None:
            return 0.0
        
        try:
            # Align dates
            stock_close = weekly_df['Close']
            
            # Convert benchmark to weekly if needed
            if isinstance(self.benchmark_data.index, pd.DatetimeIndex):
                benchmark_weekly = self.benchmark_data.resample('W')['Close'].last()
            else:
                benchmark_weekly = self.benchmark_data['Close']
            
            # Calculate RS ratio
            common_idx = stock_close.index.intersection(benchmark_weekly.index)
            if len(common_idx) < self.config.rs_ma_period:
                return 0.0
            
            stock_aligned = stock_close.loc[common_idx]
            bench_aligned = benchmark_weekly.loc[common_idx]
            
            rs_ratio = stock_aligned / bench_aligned
            rs_ma = rs_ratio.rolling(window=self.config.rs_ma_period).mean()
            
            # Mansfield RS = (current ratio / MA of ratio) - 1
            if rs_ma.iloc[-1] > 0:
                mansfield_rs = (rs_ratio.iloc[-1] / rs_ma.iloc[-1]) - 1
                return mansfield_rs
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _count_stage_duration(self, close: pd.Series, ma_30w: pd.Series,
                              current_stage: Stage) -> int:
        """Count how many weeks stock has been in current stage."""
        count = 0
        
        for i in range(len(close) - 1, -1, -1):
            if pd.isna(ma_30w.iloc[i]):
                break
            
            price_vs_ma = (close.iloc[i] - ma_30w.iloc[i]) / ma_30w.iloc[i]
            
            # Simplified stage check
            if current_stage == Stage.STAGE_2:
                if price_vs_ma > 0:
                    count += 1
                else:
                    break
            elif current_stage == Stage.STAGE_4:
                if price_vs_ma < 0:
                    count += 1
                else:
                    break
            else:
                # For stages 1 and 3, just count while close to MA
                if abs(price_vs_ma) < 0.10:
                    count += 1
                else:
                    break
        
        return count
    
    def _detect_breakout(self, weekly_df: pd.DataFrame,
                        ma_30w: pd.Series) -> Tuple[bool, Optional[float]]:
        """
        Detect Stage 2 breakout conditions:
        1. Price breaks above resistance (recent high)
        2. Volume surge (2x average)
        3. MA turning up
        """
        if len(weekly_df) < self.config.min_base_weeks:
            return False, None
        
        close = weekly_df['Close']
        volume = weekly_df['Volume']
        
        # Current values
        current_close = close.iloc[-1]
        current_volume = volume.iloc[-1]
        
        # Calculate average volume
        avg_volume = volume.iloc[-self.config.volume_lookback_weeks:-1].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Check for resistance breakout
        # Resistance = highest close in past N weeks
        lookback = min(self.config.max_base_weeks, len(close) - 1)
        resistance = close.iloc[-lookback:-1].max()
        
        # Breakout conditions
        price_breakout = current_close > resistance
        volume_confirmation = volume_ratio >= self.config.volume_breakout_multiplier
        ma_turning_up = (ma_30w.iloc[-1] > ma_30w.iloc[-2]) if len(ma_30w) >= 2 else False
        
        breakout_detected = price_breakout and volume_confirmation and ma_turning_up
        
        return breakout_detected, volume_ratio if breakout_detected else None
    
    def _check_stage2_breakout(self, weekly_df: pd.DataFrame,
                               ma_30w: pd.Series) -> float:
        """Score the strength of a potential Stage 2 breakout."""
        breakout_detected, volume_ratio = self._detect_breakout(weekly_df, ma_30w)
        
        if not breakout_detected:
            # Check for near-breakout conditions
            close = weekly_df['Close']
            lookback = min(self.config.max_base_weeks, len(close) - 1)
            resistance = close.iloc[-lookback:-1].max()
            
            current_close = close.iloc[-1]
            distance_to_resistance = (resistance - current_close) / resistance
            
            if distance_to_resistance < 0.02:  # Within 2%
                return 0.6
            elif distance_to_resistance < 0.05:  # Within 5%
                return 0.4
            else:
                return 0.2
        
        # Score the breakout
        base_score = 0.8
        
        # Volume quality bonus
        if volume_ratio and volume_ratio >= 2.5:
            base_score += 0.1
        
        # Price extension bonus (breaking out to new high)
        if len(weekly_df) >= 52:
            high_52w = weekly_df['High'].iloc[-52:].max()
            if weekly_df['Close'].iloc[-1] >= high_52w:
                base_score += 0.1
        
        return min(1.0, base_score)
    
    def _score_stage(self, metrics: StageMetrics) -> float:
        """Convert stage metrics to a buyability score."""
        stage = metrics.stage
        
        if stage == Stage.STAGE_2:
            base = 0.75
            # Adjust for how far into stage 2
            if metrics.weeks_in_stage < 4:
                base += 0.15  # Early stage 2 is best
            elif metrics.weeks_in_stage > 40:
                base -= 0.15  # Late stage 2, might be topping
        elif stage == Stage.STAGE_1:
            base = 0.5  # Waiting for breakout
            if metrics.breakout_detected:
                base = 0.85  # Just broke out!
        elif stage == Stage.STAGE_3:
            base = 0.3  # Avoid
        elif stage == Stage.STAGE_4:
            base = 0.1  # Strong avoid
        else:
            base = 0.4
        
        # RS adjustment
        if metrics.mansfield_rs > 0:
            base += min(0.1, metrics.mansfield_rs * 0.5)
        else:
            base -= min(0.1, abs(metrics.mansfield_rs) * 0.5)
        
        return max(0, min(1, base * metrics.confidence + (1 - metrics.confidence) * 0.5))
    
    def _calculate_rs_score(self, daily_df: pd.DataFrame) -> float:
        """Calculate relative strength score."""
        if self.benchmark_data is None:
            return 0.5  # Neutral if no benchmark
        
        try:
            stock_return = (daily_df['Close'].iloc[-1] / 
                          daily_df['Close'].iloc[-252] - 1)
            
            benchmark_return = (self.benchmark_data['Close'].iloc[-1] / 
                              self.benchmark_data['Close'].iloc[-252] - 1)
            
            excess_return = stock_return - benchmark_return
            
            if excess_return > 0.20:
                return 0.95
            elif excess_return > 0.10:
                return 0.8
            elif excess_return > 0:
                return 0.6
            elif excess_return > -0.10:
                return 0.4
            else:
                return 0.2
                
        except Exception:
            return 0.5
