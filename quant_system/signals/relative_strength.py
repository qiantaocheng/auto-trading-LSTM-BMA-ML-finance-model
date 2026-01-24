"""
Relative Strength Rating Calculator
===================================

Implements IBD-style Relative Strength (RS) Rating:
- Measures stock's price performance vs. all other stocks
- Time-weighted to favor recent momentum
- Returns percentile rank (0-99)

Also includes:
- RS divergence detection (stock strength during market weakness)
- Sector-relative strength
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/claude/quant_system')

from core.data_types import BaseSignalGenerator, SignalResult, StockData
from config.settings import RelativeStrengthConfig


@dataclass
class RSRankResult:
    """Container for RS ranking results."""
    rs_rating: int  # 0-99 percentile
    raw_weighted_return: float
    quarter_returns: Dict[str, float]
    vs_benchmark_return: float  # Excess return vs benchmark
    divergence_detected: bool
    divergence_strength: float


class RelativeStrengthSignal(BaseSignalGenerator):
    """
    IBD-Style Relative Strength Signal Generator
    
    Features:
    1. Time-weighted return calculation (recent performance weighted more)
    2. Percentile ranking across universe
    3. Divergence detection during market corrections
    """
    
    def __init__(self, config: Optional[RelativeStrengthConfig] = None,
                 benchmark_data: Optional[pd.DataFrame] = None,
                 universe_returns: Optional[pd.DataFrame] = None):
        """
        Args:
            config: RS calculation configuration
            benchmark_data: Index data (e.g., SPY) for divergence detection
            universe_returns: DataFrame of returns for all stocks in universe
                             Used for percentile ranking
        """
        self.config = config or RelativeStrengthConfig()
        self.benchmark_data = benchmark_data
        self.universe_returns = universe_returns
        self._cache = {}
    
    def set_benchmark(self, benchmark_data: pd.DataFrame):
        """Set benchmark data."""
        self.benchmark_data = benchmark_data
    
    def set_universe(self, universe_returns: pd.DataFrame):
        """Set universe returns for ranking."""
        self.universe_returns = universe_returns
    
    def get_required_lookback(self) -> int:
        """Need 1 year of data for RS calculation."""
        return self.config.total_lookback_days + 20  # Buffer
    
    def generate(self, data: StockData) -> SignalResult:
        """Generate RS signal for a single stock."""
        if not self.validate_data(data):
            return SignalResult.from_score(
                name='relative_strength',
                score=0.5,
                confidence=0.1,
                metadata={'error': 'Insufficient data'}
            )
        
        # Calculate RS metrics
        rs_result = self._calculate_rs(data.df)
        
        # Convert to signal score
        rs_score = self._rs_to_score(rs_result)
        
        # Divergence score
        divergence_score = self._calculate_divergence_score(data.df, rs_result)
        
        # Momentum consistency score
        momentum_score = self._calculate_momentum_consistency(rs_result)
        
        # Composite score
        composite_score = (
            0.50 * rs_score +
            0.30 * divergence_score +
            0.20 * momentum_score
        )
        
        components = {
            'rs_rating_score': rs_score,
            'divergence_score': divergence_score,
            'momentum_consistency': momentum_score
        }
        
        metadata = {
            'rs_rating': rs_result.rs_rating,
            'weighted_return': rs_result.raw_weighted_return,
            'q1_return': rs_result.quarter_returns.get('quarter_1', 0),
            'q2_return': rs_result.quarter_returns.get('quarter_2', 0),
            'q3_return': rs_result.quarter_returns.get('quarter_3', 0),
            'q4_return': rs_result.quarter_returns.get('quarter_4', 0),
            'vs_benchmark': rs_result.vs_benchmark_return,
            'divergence_detected': rs_result.divergence_detected,
            'divergence_strength': rs_result.divergence_strength
        }
        
        # Confidence based on data quality and RS consistency
        confidence = 0.8 if rs_result.rs_rating >= 70 else 0.6
        
        return SignalResult.from_score(
            name='relative_strength',
            score=composite_score,
            confidence=confidence,
            components=components,
            metadata=metadata
        )
    
    def _calculate_rs(self, df: pd.DataFrame) -> RSRankResult:
        """
        Calculate IBD-style RS rating.
        
        Methodology:
        1. Calculate returns for each quarter (most recent 4 quarters)
        2. Apply time weights (most recent weighted 2x)
        3. Calculate percentile rank vs universe (if available)
        """
        close = df['Close']
        cfg = self.config
        
        # Ensure we have enough data
        if len(close) < cfg.total_lookback_days:
            return RSRankResult(
                rs_rating=50,
                raw_weighted_return=0.0,
                quarter_returns={},
                vs_benchmark_return=0.0,
                divergence_detected=False,
                divergence_strength=0.0
            )
        
        # Calculate quarterly returns (63 trading days per quarter)
        quarter_days = 63
        quarter_returns = {}
        
        # Q1: Most recent quarter (days -63 to 0)
        if len(close) >= quarter_days:
            q1_return = (close.iloc[-1] / close.iloc[-quarter_days] - 1)
            quarter_returns['quarter_1'] = q1_return
        else:
            q1_return = 0
            quarter_returns['quarter_1'] = 0
        
        # Q2: Second quarter (days -126 to -63)
        if len(close) >= 2 * quarter_days:
            q2_return = (close.iloc[-quarter_days] / close.iloc[-2*quarter_days] - 1)
            quarter_returns['quarter_2'] = q2_return
        else:
            q2_return = 0
            quarter_returns['quarter_2'] = 0
        
        # Q3: Third quarter (days -189 to -126)
        if len(close) >= 3 * quarter_days:
            q3_return = (close.iloc[-2*quarter_days] / close.iloc[-3*quarter_days] - 1)
            quarter_returns['quarter_3'] = q3_return
        else:
            q3_return = 0
            quarter_returns['quarter_3'] = 0
        
        # Q4: Fourth quarter (days -252 to -189)
        if len(close) >= 4 * quarter_days:
            q4_return = (close.iloc[-3*quarter_days] / close.iloc[-4*quarter_days] - 1)
            quarter_returns['quarter_4'] = q4_return
        else:
            q4_return = 0
            quarter_returns['quarter_4'] = 0
        
        # Apply time weights
        weights = cfg.period_weights
        weighted_return = (
            weights['quarter_1'] * q1_return +
            weights['quarter_2'] * q2_return +
            weights['quarter_3'] * q3_return +
            weights['quarter_4'] * q4_return
        )
        
        # Calculate RS rating (percentile rank)
        if self.universe_returns is not None and len(self.universe_returns) > 0:
            # Rank against universe
            universe_weighted = self._calculate_universe_weighted_returns()
            if universe_weighted is not None:
                rs_rating = int((universe_weighted < weighted_return).mean() * 99)
            else:
                rs_rating = self._estimate_rs_from_return(weighted_return)
        else:
            # Estimate based on absolute return
            rs_rating = self._estimate_rs_from_return(weighted_return)
        
        # Calculate vs benchmark
        vs_benchmark = self._calc_vs_benchmark(df)
        
        # Check for divergence
        divergence, strength = self._detect_divergence(df)
        
        return RSRankResult(
            rs_rating=rs_rating,
            raw_weighted_return=weighted_return,
            quarter_returns=quarter_returns,
            vs_benchmark_return=vs_benchmark,
            divergence_detected=divergence,
            divergence_strength=strength
        )
    
    def _calculate_universe_weighted_returns(self) -> Optional[pd.Series]:
        """Calculate weighted returns for entire universe."""
        if self.universe_returns is None:
            return None
        
        cfg = self.config
        weights = cfg.period_weights
        quarter_days = 63
        
        # This assumes universe_returns has stock returns
        # In practice, you'd calculate weighted returns for each stock
        
        # Simplified: return the pre-calculated weighted returns if available
        if 'weighted_return' in self.universe_returns.columns:
            return self.universe_returns['weighted_return']
        
        return None
    
    def _estimate_rs_from_return(self, weighted_return: float) -> int:
        """
        Estimate RS rating from weighted return without universe data.
        
        Based on historical distributions of stock returns.
        """
        # Approximate mapping based on typical market distributions
        # These are rough estimates; real RS requires universe comparison
        
        if weighted_return >= 0.60:
            return 99
        elif weighted_return >= 0.40:
            return 95 + int((weighted_return - 0.40) / 0.20 * 4)
        elif weighted_return >= 0.25:
            return 85 + int((weighted_return - 0.25) / 0.15 * 10)
        elif weighted_return >= 0.15:
            return 70 + int((weighted_return - 0.15) / 0.10 * 15)
        elif weighted_return >= 0.05:
            return 50 + int((weighted_return - 0.05) / 0.10 * 20)
        elif weighted_return >= 0:
            return 40 + int(weighted_return / 0.05 * 10)
        elif weighted_return >= -0.10:
            return 25 + int((weighted_return + 0.10) / 0.10 * 15)
        elif weighted_return >= -0.25:
            return 10 + int((weighted_return + 0.25) / 0.15 * 15)
        else:
            return max(1, int(10 + weighted_return * 20))
    
    def _calc_vs_benchmark(self, df: pd.DataFrame) -> float:
        """Calculate excess return vs benchmark."""
        if self.benchmark_data is None:
            return 0.0
        
        try:
            stock_return = df['Close'].iloc[-1] / df['Close'].iloc[-252] - 1
            
            # Align dates
            bench_close = self.benchmark_data['Close']
            benchmark_return = bench_close.iloc[-1] / bench_close.iloc[-252] - 1
            
            return stock_return - benchmark_return
            
        except Exception:
            return 0.0
    
    def _detect_divergence(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Detect positive divergence: stock holds up while market falls.
        
        This is a bullish signal indicating institutional accumulation.
        """
        if self.benchmark_data is None:
            return False, 0.0
        
        cfg = self.config
        lookback = cfg.divergence_lookback_days
        
        try:
            stock_close = df['Close']
            bench_close = self.benchmark_data['Close']
            
            # Calculate recent returns
            stock_return = stock_close.iloc[-1] / stock_close.iloc[-lookback] - 1
            bench_return = bench_close.iloc[-1] / bench_close.iloc[-lookback] - 1
            
            # Divergence: benchmark down, stock flat or up
            if bench_return < cfg.divergence_index_threshold:
                if stock_return > 0:
                    # Strong positive divergence
                    strength = abs(stock_return - bench_return)
                    return True, min(1.0, strength * 5)
                elif stock_return > bench_return * 0.5:
                    # Moderate positive divergence (held up better)
                    strength = abs(stock_return - bench_return) / abs(bench_return)
                    return True, min(0.7, strength)
            
            return False, 0.0
            
        except Exception:
            return False, 0.0
    
    def _rs_to_score(self, rs_result: RSRankResult) -> float:
        """Convert RS rating to normalized score."""
        rs = rs_result.rs_rating
        cfg = self.config
        
        if rs >= cfg.elite_rs_rating:
            return 0.95
        elif rs >= cfg.min_rs_rating:
            # Scale from 0.7 to 0.9 for RS 80-90
            return 0.7 + (rs - cfg.min_rs_rating) / (cfg.elite_rs_rating - cfg.min_rs_rating) * 0.2
        elif rs >= 60:
            return 0.5 + (rs - 60) / 20 * 0.2
        elif rs >= 40:
            return 0.3 + (rs - 40) / 20 * 0.2
        else:
            return max(0.1, rs / 40 * 0.3)
    
    def _calculate_divergence_score(self, df: pd.DataFrame,
                                    rs_result: RSRankResult) -> float:
        """Score based on divergence characteristics."""
        if rs_result.divergence_detected:
            return 0.6 + rs_result.divergence_strength * 0.4
        
        # Even without divergence, positive vs benchmark is good
        vs_bench = rs_result.vs_benchmark_return
        
        if vs_bench >= 0.20:
            return 0.8
        elif vs_bench >= 0.10:
            return 0.6
        elif vs_bench >= 0:
            return 0.5
        elif vs_bench >= -0.10:
            return 0.3
        else:
            return 0.1
    
    def _calculate_momentum_consistency(self, rs_result: RSRankResult) -> float:
        """
        Score based on momentum consistency across quarters.
        
        Prefer stocks with consistent positive returns over all quarters.
        """
        qr = rs_result.quarter_returns
        
        if not qr:
            return 0.5
        
        # Count positive quarters
        positive_count = sum(1 for v in qr.values() if v > 0)
        
        # Check for acceleration (Q1 > Q2 > Q3 > Q4)
        q_values = [qr.get(f'quarter_{i}', 0) for i in range(1, 5)]
        accelerating = all(q_values[i] >= q_values[i+1] for i in range(len(q_values)-1))
        
        base_score = positive_count / 4
        
        if accelerating and positive_count >= 3:
            return min(1.0, base_score + 0.2)
        elif positive_count >= 3:
            return base_score + 0.1
        else:
            return base_score


class UniverseRSCalculator:
    """
    Calculate RS ratings for entire universe of stocks.
    
    This is needed for proper percentile ranking.
    """
    
    def __init__(self, config: Optional[RelativeStrengthConfig] = None):
        self.config = config or RelativeStrengthConfig()
    
    def calculate_universe_rs(self, 
                             universe_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate RS ratings for all stocks in universe.
        
        Args:
            universe_data: Dict mapping ticker -> OHLCV DataFrame
            
        Returns:
            DataFrame with columns: ticker, weighted_return, rs_rating
        """
        results = []
        cfg = self.config
        quarter_days = 63
        
        for ticker, df in universe_data.items():
            if len(df) < cfg.total_lookback_days:
                continue
            
            close = df['Close']
            
            # Calculate quarterly returns
            try:
                q1 = close.iloc[-1] / close.iloc[-quarter_days] - 1 if len(close) >= quarter_days else 0
                q2 = close.iloc[-quarter_days] / close.iloc[-2*quarter_days] - 1 if len(close) >= 2*quarter_days else 0
                q3 = close.iloc[-2*quarter_days] / close.iloc[-3*quarter_days] - 1 if len(close) >= 3*quarter_days else 0
                q4 = close.iloc[-3*quarter_days] / close.iloc[-4*quarter_days] - 1 if len(close) >= 4*quarter_days else 0
                
                weights = cfg.period_weights
                weighted_return = (
                    weights['quarter_1'] * q1 +
                    weights['quarter_2'] * q2 +
                    weights['quarter_3'] * q3 +
                    weights['quarter_4'] * q4
                )
                
                results.append({
                    'ticker': ticker,
                    'weighted_return': weighted_return,
                    'q1_return': q1,
                    'q2_return': q2,
                    'q3_return': q3,
                    'q4_return': q4
                })
            except Exception:
                continue
        
        if not results:
            return pd.DataFrame()
        
        df_results = pd.DataFrame(results)
        
        # Calculate percentile rank
        df_results['rs_rating'] = (
            df_results['weighted_return'].rank(pct=True) * 99
        ).astype(int)
        
        return df_results.sort_values('rs_rating', ascending=False)
