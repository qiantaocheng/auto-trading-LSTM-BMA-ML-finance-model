"""
Trend Signal Generators
=======================

Implements:
1. Minervini Trend Template
2. Moving Average System Analysis
3. ADX/DMI Trend Strength
4. Linear Regression Slope Analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional

import sys
sys.path.insert(0, '/home/claude/quant_system')

from core.data_types import BaseSignalGenerator, SignalResult, StockData
from config.settings import TrendTemplateConfig, ADXConfig


class TrendTemplateSignal(BaseSignalGenerator):
    """
    Minervini Trend Template Implementation
    
    Evaluates:
    1. MA Alignment: Price > SMA50 > SMA150 > SMA200
    2. 200-day MA Slope: Must be positive for at least 1 month
    3. Price Position: Above 30% from 52-week low, within 25% of 52-week high
    4. Acceleration: SMA50 slope > SMA200 slope
    """
    
    def __init__(self, config: Optional[TrendTemplateConfig] = None):
        self.config = config or TrendTemplateConfig()
        self._cache = {}
    
    def get_required_lookback(self) -> int:
        """Need 252 days for 52-week calculations."""
        return max(252, self.config.ma_long + 50)
    
    def generate(self, data: StockData) -> SignalResult:
        """Generate trend template signal."""
        if not self.validate_data(data):
            return SignalResult.from_score(
                name='trend_template',
                score=0.0,
                confidence=0.1,
                metadata={'error': 'Insufficient data'}
            )
        
        df = data.df.copy()
        close = df['Close']
        
        # Calculate moving averages
        sma50 = close.rolling(window=self.config.ma_short).mean()
        sma150 = close.rolling(window=self.config.ma_medium).mean()
        sma200 = close.rolling(window=self.config.ma_long).mean()
        
        # Get latest values
        latest_close = close.iloc[-1]
        latest_sma50 = sma50.iloc[-1]
        latest_sma150 = sma150.iloc[-1]
        latest_sma200 = sma200.iloc[-1]
        
        # 1. MA Alignment Score
        ma_alignment_score = self._calculate_ma_alignment(
            latest_close, latest_sma50, latest_sma150, latest_sma200
        )
        
        # 2. Price Position Score (52-week range)
        price_position_score = self._calculate_price_position(close)
        
        # 3. 200-day MA Slope Score
        slope_score, slope_value = self._calculate_slope_score(sma200)
        
        # 4. Acceleration Score (SMA50 slope vs SMA200 slope)
        acceleration_score = self._calculate_acceleration(sma50, sma200)
        
        # 5. MA Distance Score (how far above MAs - reward stronger trends)
        ma_distance_score = self._calculate_ma_distance(
            latest_close, latest_sma50, latest_sma150, latest_sma200
        )
        
        # Weighted composite
        cfg = self.config
        composite_score = (
            cfg.weight_ma_alignment * ma_alignment_score +
            cfg.weight_price_position * price_position_score +
            cfg.weight_slope * slope_score +
            cfg.weight_acceleration * acceleration_score +
            cfg.weight_ma_distance * ma_distance_score
        )
        
        # Calculate confidence based on data quality and consistency
        confidence = self._calculate_confidence(
            ma_alignment_score, price_position_score, slope_score
        )
        
        components = {
            'ma_alignment': ma_alignment_score,
            'price_position': price_position_score,
            'slope': slope_score,
            'acceleration': acceleration_score,
            'ma_distance': ma_distance_score
        }
        
        metadata = {
            'sma50': latest_sma50,
            'sma150': latest_sma150,
            'sma200': latest_sma200,
            'sma200_slope': slope_value,
            'price': latest_close,
            'pct_from_52w_high': self._pct_from_high(close),
            'pct_from_52w_low': self._pct_from_low(close)
        }
        
        return SignalResult.from_score(
            name='trend_template',
            score=composite_score,
            confidence=confidence,
            components=components,
            metadata=metadata
        )
    
    def _calculate_ma_alignment(self, price: float, sma50: float,
                                sma150: float, sma200: float) -> float:
        """
        Score MA alignment condition.
        Perfect alignment: Price > SMA50 > SMA150 > SMA200
        """
        if np.isnan(sma200):
            return 0.0
        
        score = 0.0
        
        # Each correct relationship adds to score
        if price > sma50:
            score += 0.25
        if price > sma150:
            score += 0.25
        if price > sma200:
            score += 0.25
        if sma50 > sma150 > sma200:
            score += 0.25  # Perfect stacking bonus
        
        return score
    
    def _calculate_price_position(self, close: pd.Series) -> float:
        """Score based on position within 52-week range."""
        high_52w = close.rolling(window=252).max().iloc[-1]
        low_52w = close.rolling(window=252).min().iloc[-1]
        current = close.iloc[-1]
        
        if np.isnan(high_52w) or np.isnan(low_52w):
            return 0.0
        
        # Calculate percentage from low and high
        pct_from_low = (current - low_52w) / low_52w if low_52w > 0 else 0
        pct_from_high = (current / high_52w) if high_52w > 0 else 0
        
        score = 0.0
        
        # Must be at least 30% above 52-week low
        if pct_from_low >= self.config.min_above_52w_low:
            score += 0.5
            # Extra credit for being much higher
            if pct_from_low >= 0.5:
                score += 0.1
        
        # Must be within 25% of 52-week high
        if pct_from_high >= self.config.min_near_52w_high:
            score += 0.4
            # Bonus for being very close to high
            if pct_from_high >= 0.90:
                score += 0.1
        
        return min(1.0, score)
    
    def _calculate_slope_score(self, sma200: pd.Series) -> Tuple[float, float]:
        """Calculate 200-day MA slope using linear regression."""
        lookback = self.config.slope_lookback
        recent_sma = sma200.iloc[-lookback:].dropna()
        
        if len(recent_sma) < lookback * 0.8:
            return 0.0, 0.0
        
        # Linear regression for slope
        x = np.arange(len(recent_sma))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_sma.values)
        
        # Normalize slope by price level (percentage change per day)
        normalized_slope = slope / recent_sma.mean() * 100  # Daily % change
        
        # Score based on slope magnitude
        if normalized_slope <= 0:
            return 0.0, normalized_slope
        elif normalized_slope < 0.02:  # Very shallow positive
            score = 0.3
        elif normalized_slope < 0.05:  # Moderate positive
            score = 0.6
        elif normalized_slope < 0.10:  # Strong positive
            score = 0.85
        else:  # Very strong (potentially overextended)
            score = 0.95
        
        # Adjust for consistency (R-squared)
        r_squared = r_value ** 2
        score *= (0.5 + 0.5 * r_squared)  # Reward consistent trends
        
        return min(1.0, score), normalized_slope
    
    def _calculate_acceleration(self, sma50: pd.Series, sma200: pd.Series) -> float:
        """Compare SMA50 slope to SMA200 slope - looking for acceleration."""
        lookback = self.config.slope_lookback
        
        sma50_recent = sma50.iloc[-lookback:].dropna()
        sma200_recent = sma200.iloc[-lookback:].dropna()
        
        if len(sma50_recent) < 10 or len(sma200_recent) < 10:
            return 0.5  # Neutral if insufficient data
        
        # Calculate slopes
        x = np.arange(len(sma50_recent))
        slope_50, *_ = stats.linregress(x, sma50_recent.values)
        
        x = np.arange(len(sma200_recent))
        slope_200, *_ = stats.linregress(x, sma200_recent.values)
        
        # Normalize slopes
        norm_slope_50 = slope_50 / sma50_recent.mean()
        norm_slope_200 = slope_200 / sma200_recent.mean()
        
        # Acceleration = SMA50 rising faster than SMA200
        if norm_slope_50 > norm_slope_200 > 0:
            return min(1.0, 0.7 + 0.3 * (norm_slope_50 / max(0.001, norm_slope_200) - 1))
        elif norm_slope_50 > 0 and norm_slope_200 > 0:
            return 0.5
        elif norm_slope_50 > 0:
            return 0.3
        else:
            return 0.0
    
    def _calculate_ma_distance(self, price: float, sma50: float,
                               sma150: float, sma200: float) -> float:
        """Score based on distance above moving averages."""
        if np.isnan(sma200):
            return 0.0
        
        # Calculate percentage above each MA
        pct_above_50 = (price - sma50) / sma50 if sma50 > 0 else 0
        pct_above_150 = (price - sma150) / sma150 if sma150 > 0 else 0
        pct_above_200 = (price - sma200) / sma200 if sma200 > 0 else 0
        
        # Score: want to be above, but not TOO far (overextended)
        def distance_score(pct: float) -> float:
            if pct < 0:
                return 0.0
            elif pct < 0.05:  # 0-5% above
                return 0.5 + pct * 10  # 0.5 to 1.0
            elif pct < 0.15:  # 5-15% above (ideal zone)
                return 1.0
            elif pct < 0.30:  # 15-30% (getting extended)
                return 1.0 - (pct - 0.15) * 2  # 1.0 to 0.7
            else:  # Very extended
                return 0.5
        
        avg_score = (
            distance_score(pct_above_50) * 0.4 +
            distance_score(pct_above_150) * 0.3 +
            distance_score(pct_above_200) * 0.3
        )
        
        return avg_score
    
    def _calculate_confidence(self, ma_align: float, price_pos: float,
                             slope: float) -> float:
        """Calculate confidence in the signal."""
        # High confidence if all components agree
        if ma_align >= 0.8 and price_pos >= 0.8 and slope >= 0.6:
            return 0.9
        elif ma_align >= 0.6 and price_pos >= 0.6 and slope >= 0.4:
            return 0.7
        elif min(ma_align, price_pos, slope) >= 0.3:
            return 0.5
        else:
            return 0.3
    
    def _pct_from_high(self, close: pd.Series) -> float:
        """Calculate percentage below 52-week high."""
        high_52w = close.rolling(window=252).max().iloc[-1]
        return (close.iloc[-1] - high_52w) / high_52w if high_52w > 0 else 0
    
    def _pct_from_low(self, close: pd.Series) -> float:
        """Calculate percentage above 52-week low."""
        low_52w = close.rolling(window=252).min().iloc[-1]
        return (close.iloc[-1] - low_52w) / low_52w if low_52w > 0 else 0


class ADXSignal(BaseSignalGenerator):
    """
    Average Directional Index (ADX) Signal Generator
    
    Measures trend strength without direction, combined with
    DMI (+DI/-DI) for directional bias.
    """
    
    def __init__(self, config: Optional[ADXConfig] = None):
        self.config = config or ADXConfig()
        self._cache = {}
    
    def get_required_lookback(self) -> int:
        """Need extra buffer for ATR smoothing."""
        return self.config.period * 3
    
    def generate(self, data: StockData) -> SignalResult:
        """Generate ADX-based signal."""
        if not self.validate_data(data):
            return SignalResult.from_score(
                name='adx',
                score=0.5,  # Neutral on insufficient data
                confidence=0.1,
                metadata={'error': 'Insufficient data'}
            )
        
        df = data.df.copy()
        
        # Calculate ADX and DMI
        adx, plus_di, minus_di = self._calculate_adx_dmi(df)
        
        latest_adx = adx.iloc[-1]
        latest_plus_di = plus_di.iloc[-1]
        latest_minus_di = minus_di.iloc[-1]
        
        # Score based on ADX level
        trend_strength_score = self._score_adx(latest_adx)
        
        # Direction score from DMI
        direction_score = self._score_direction(latest_plus_di, latest_minus_di)
        
        # ADX momentum (is trend getting stronger?)
        adx_momentum_score = self._score_adx_momentum(adx)
        
        # Combined score
        composite_score = (
            0.40 * trend_strength_score +
            0.35 * direction_score +
            0.25 * adx_momentum_score
        )
        
        # Penalty if ADX shows no trend
        if latest_adx < self.config.trend_threshold:
            composite_score *= 0.5  # Halve score in non-trending market
        
        components = {
            'trend_strength': trend_strength_score,
            'direction': direction_score,
            'adx_momentum': adx_momentum_score
        }
        
        metadata = {
            'adx': latest_adx,
            'plus_di': latest_plus_di,
            'minus_di': latest_minus_di,
            'di_spread': latest_plus_di - latest_minus_di,
            'is_trending': latest_adx >= self.config.trend_threshold
        }
        
        confidence = 0.8 if latest_adx >= self.config.trend_threshold else 0.4
        
        return SignalResult.from_score(
            name='adx',
            score=composite_score,
            confidence=confidence,
            components=components,
            metadata=metadata
        )
    
    def _calculate_adx_dmi(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate ADX and DMI indicators.
        
        Returns: (ADX, +DI, -DI) series
        """
        period = self.config.period
        
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)
        
        # Smoothed DM
        plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx, plus_di, minus_di
    
    def _score_adx(self, adx: float) -> float:
        """Score based on ADX level."""
        if np.isnan(adx):
            return 0.5
        
        cfg = self.config
        
        if adx < 20:
            return 0.2  # No trend
        elif adx < cfg.trend_threshold:
            return 0.4  # Weak trend
        elif adx < cfg.strong_trend_threshold:
            return 0.7  # Moderate trend
        elif adx < cfg.extreme_trend_threshold:
            return 0.9  # Strong trend
        else:
            return 0.75  # Very strong but overheated (slight penalty)
    
    def _score_direction(self, plus_di: float, minus_di: float) -> float:
        """Score based on DMI direction."""
        if np.isnan(plus_di) or np.isnan(minus_di):
            return 0.5
        
        if self.config.require_di_positive:
            if plus_di > minus_di:
                # Score based on spread
                spread = plus_di - minus_di
                if spread > 20:
                    return 1.0
                elif spread > 10:
                    return 0.8
                elif spread > 5:
                    return 0.6
                else:
                    return 0.55
            else:
                return 0.2  # Downtrend
        else:
            return 0.5  # Direction not required
    
    def _score_adx_momentum(self, adx: pd.Series) -> float:
        """Score ADX momentum (is trend strengthening?)."""
        if len(adx) < 5:
            return 0.5
        
        recent_adx = adx.iloc[-5:]
        
        # Simple slope check
        if recent_adx.iloc[-1] > recent_adx.iloc[0]:
            # ADX rising
            pct_change = (recent_adx.iloc[-1] - recent_adx.iloc[0]) / max(1, recent_adx.iloc[0])
            return min(1.0, 0.6 + pct_change)
        else:
            # ADX falling
            return 0.4


class LinearRegressionSlopeSignal(BaseSignalGenerator):
    """
    Linear Regression Slope Analysis
    
    Measures the quality and consistency of price trends using
    linear regression on price data.
    """
    
    def __init__(self, lookback: int = 50, ma_period: int = 20):
        self.lookback = lookback
        self.ma_period = ma_period
        self._cache = {}
    
    def get_required_lookback(self) -> int:
        return self.lookback + self.ma_period
    
    def generate(self, data: StockData) -> SignalResult:
        """Generate linear regression slope signal."""
        if not self.validate_data(data):
            return SignalResult.from_score(
                name='lr_slope',
                score=0.5,
                confidence=0.1,
                metadata={'error': 'Insufficient data'}
            )
        
        close = data.df['Close']
        
        # Calculate regression on price
        price_slope, price_r2 = self._calc_regression(close.iloc[-self.lookback:])
        
        # Calculate regression on moving average (smoother)
        ma = close.rolling(window=self.ma_period).mean()
        ma_slope, ma_r2 = self._calc_regression(ma.iloc[-self.lookback:].dropna())
        
        # Normalize slopes by price level
        avg_price = close.iloc[-self.lookback:].mean()
        norm_price_slope = price_slope / avg_price * 252  # Annualized
        norm_ma_slope = ma_slope / avg_price * 252 if not np.isnan(ma_slope) else 0
        
        # Score based on slope direction and magnitude
        if norm_ma_slope <= 0:
            base_score = 0.2 + 0.3 * max(0, 1 + norm_ma_slope)  # 0.2-0.5 for negative
        elif norm_ma_slope < 0.10:
            base_score = 0.5 + norm_ma_slope * 3  # 0.5-0.8
        elif norm_ma_slope < 0.30:
            base_score = 0.8 + (norm_ma_slope - 0.10) * 0.5  # 0.8-0.9
        else:
            base_score = 0.9  # Cap at 0.9 for very steep slopes
        
        # Adjust for R-squared (trend consistency)
        consistency_multiplier = 0.7 + 0.3 * ma_r2
        final_score = base_score * consistency_multiplier
        
        components = {
            'price_slope_annualized': norm_price_slope,
            'ma_slope_annualized': norm_ma_slope,
            'price_r2': price_r2,
            'ma_r2': ma_r2
        }
        
        metadata = {
            'lookback_days': self.lookback,
            'ma_period': self.ma_period
        }
        
        return SignalResult.from_score(
            name='lr_slope',
            score=min(1.0, max(0.0, final_score)),
            confidence=ma_r2,  # R2 as confidence
            components=components,
            metadata=metadata
        )
    
    def _calc_regression(self, series: pd.Series) -> Tuple[float, float]:
        """Calculate linear regression slope and R-squared."""
        series = series.dropna()
        if len(series) < 10:
            return 0.0, 0.0
        
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
        
        return slope, r_value ** 2
