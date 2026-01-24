"""
Volatility Contraction Pattern (VCP) Detector
=============================================

Implements Minervini's VCP pattern recognition:

A VCP is characterized by:
1. Series of price contractions (2-6), each smaller than the previous
2. Volume dry-up during final contraction (supply exhaustion)
3. Tight price range ("the squat") before breakout
4. Pivot point breakout with volume surge

This is the most challenging pattern to detect programmatically.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, argrelextrema
from scipy.ndimage import uniform_filter1d
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/claude/quant_system')

from core.data_types import BaseSignalGenerator, SignalResult, StockData
from config.settings import VCPConfig


@dataclass
class Contraction:
    """Represents a single contraction within a VCP pattern."""
    start_idx: int
    end_idx: int
    high_price: float
    low_price: float
    depth_pct: float  # Percentage decline from high to low
    duration_days: int
    avg_volume: float
    volume_trend: str  # 'increasing', 'decreasing', 'flat'


@dataclass
class VCPPattern:
    """Complete VCP pattern detection result."""
    is_valid: bool
    contractions: List[Contraction]
    pattern_quality: float  # 0-1 score
    
    # Current state
    in_final_squeeze: bool
    squeeze_tightness: float  # How tight the current range is
    volume_dry_up_confirmed: bool
    
    # Pivot detection
    pivot_price: Optional[float]
    pivot_breakout_triggered: bool
    
    # Pattern metadata
    total_duration_days: int
    total_depth_pct: float  # First contraction depth
    depth_decay_ratio: float  # How well contractions are shrinking


class VCPSignal(BaseSignalGenerator):
    """
    VCP Pattern Signal Generator
    
    Uses multiple techniques to detect VCP:
    1. Peak/trough detection for identifying contractions
    2. Volatility measurement (ATR, Bollinger Bandwidth)
    3. Volume trend analysis
    4. Pattern quality scoring
    """
    
    def __init__(self, config: Optional[VCPConfig] = None):
        self.config = config or VCPConfig()
        self._cache = {}
    
    def get_required_lookback(self) -> int:
        """Need sufficient history to detect base patterns."""
        return 200  # ~10 months of daily data
    
    def generate(self, data: StockData) -> SignalResult:
        """Generate VCP signal."""
        if not self.validate_data(data):
            return SignalResult.from_score(
                name='vcp',
                score=0.3,
                confidence=0.1,
                metadata={'error': 'Insufficient data'}
            )
        
        df = data.df.copy()
        
        # Step 1: Detect VCP pattern
        vcp_pattern = self._detect_vcp(df)
        
        # Step 2: Analyze Bollinger Band squeeze
        bb_squeeze_score = self._analyze_bb_squeeze(df)
        
        # Step 3: Check volume dry-up
        volume_dryup_score = self._check_volume_dryup(df)
        
        # Step 4: Detect pivot point
        pivot_score = self._detect_pivot_point(df, vcp_pattern)
        
        # Step 5: Check for pocket pivot
        pocket_pivot_score = self._check_pocket_pivot(df)
        
        # Composite score
        if vcp_pattern.is_valid:
            pattern_weight = 0.35
            squeeze_weight = 0.25
            volume_weight = 0.20
            pivot_weight = 0.20
        else:
            # If no clear VCP, weight squeeze and volume more
            pattern_weight = 0.15
            squeeze_weight = 0.35
            volume_weight = 0.30
            pivot_weight = 0.20
        
        composite_score = (
            pattern_weight * vcp_pattern.pattern_quality +
            squeeze_weight * bb_squeeze_score +
            volume_weight * volume_dryup_score +
            pivot_weight * max(pivot_score, pocket_pivot_score)
        )
        
        components = {
            'pattern_quality': vcp_pattern.pattern_quality,
            'bb_squeeze': bb_squeeze_score,
            'volume_dryup': volume_dryup_score,
            'pivot': pivot_score,
            'pocket_pivot': pocket_pivot_score
        }
        
        metadata = {
            'vcp_valid': vcp_pattern.is_valid,
            'num_contractions': len(vcp_pattern.contractions),
            'in_final_squeeze': vcp_pattern.in_final_squeeze,
            'squeeze_tightness': vcp_pattern.squeeze_tightness,
            'volume_dryup_confirmed': vcp_pattern.volume_dry_up_confirmed,
            'pivot_price': vcp_pattern.pivot_price,
            'breakout_triggered': vcp_pattern.pivot_breakout_triggered,
            'total_pattern_duration': vcp_pattern.total_duration_days,
            'depth_decay_ratio': vcp_pattern.depth_decay_ratio
        }
        
        # Confidence based on pattern clarity
        if vcp_pattern.is_valid and vcp_pattern.in_final_squeeze:
            confidence = 0.85
        elif vcp_pattern.is_valid:
            confidence = 0.7
        elif bb_squeeze_score > 0.7:
            confidence = 0.5
        else:
            confidence = 0.3
        
        return SignalResult.from_score(
            name='vcp',
            score=composite_score,
            confidence=confidence,
            components=components,
            metadata=metadata
        )
    
    def _detect_vcp(self, df: pd.DataFrame) -> VCPPattern:
        """
        Main VCP detection algorithm.
        
        Uses peak/trough analysis to identify contractions.
        """
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        # Find local peaks and troughs
        # Use smoothed data to reduce noise
        smoothed = uniform_filter1d(close, size=5)
        
        # Detect peaks (local highs)
        peak_indices = argrelextrema(smoothed, np.greater, order=10)[0]
        # Detect troughs (local lows)
        trough_indices = argrelextrema(smoothed, np.less, order=10)[0]
        
        if len(peak_indices) < 2 or len(trough_indices) < 2:
            return self._empty_vcp_pattern()
        
        # Extract contractions (swing high to swing low sequences)
        contractions = self._extract_contractions(
            df, peak_indices, trough_indices
        )
        
        # Validate VCP pattern
        if len(contractions) < self.config.min_contractions:
            return self._empty_vcp_pattern()
        
        # Check for proper contraction decay
        valid_decay = self._validate_contraction_decay(contractions)
        
        if not valid_decay:
            pattern_quality = 0.3
        else:
            pattern_quality = self._calculate_pattern_quality(contractions)
        
        # Check if in final squeeze
        in_squeeze, tightness = self._check_final_squeeze(df, contractions)
        
        # Check volume dry-up
        volume_dryup = self._is_volume_drying_up(df)
        
        # Determine pivot price
        if contractions:
            pivot_price = max(c.high_price for c in contractions[-2:])
        else:
            pivot_price = high[-20:].max()
        
        # Check if breakout triggered
        breakout = close[-1] > pivot_price and volume[-1] > volume[-20:].mean() * 1.5
        
        # Calculate depth decay ratio
        if len(contractions) >= 2:
            depths = [c.depth_pct for c in contractions]
            decay_ratios = [depths[i+1] / depths[i] for i in range(len(depths)-1) if depths[i] > 0]
            avg_decay = np.mean(decay_ratios) if decay_ratios else 1.0
        else:
            avg_decay = 1.0
        
        return VCPPattern(
            is_valid=valid_decay and len(contractions) >= self.config.min_contractions,
            contractions=contractions,
            pattern_quality=pattern_quality,
            in_final_squeeze=in_squeeze,
            squeeze_tightness=tightness,
            volume_dry_up_confirmed=volume_dryup,
            pivot_price=pivot_price,
            pivot_breakout_triggered=breakout,
            total_duration_days=sum(c.duration_days for c in contractions),
            total_depth_pct=contractions[0].depth_pct if contractions else 0,
            depth_decay_ratio=avg_decay
        )
    
    def _extract_contractions(self, df: pd.DataFrame,
                              peak_indices: np.ndarray,
                              trough_indices: np.ndarray) -> List[Contraction]:
        """Extract contraction sequences from peak/trough data."""
        contractions = []
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        # Only look at recent history (last 100 days or so)
        lookback = min(150, len(df) - 1)
        min_idx = len(df) - lookback
        
        # Filter to recent peaks/troughs
        recent_peaks = peak_indices[peak_indices >= min_idx]
        recent_troughs = trough_indices[trough_indices >= min_idx]
        
        # Build contractions from peak-to-trough sequences
        for i, peak_idx in enumerate(recent_peaks):
            # Find next trough after this peak
            following_troughs = recent_troughs[recent_troughs > peak_idx]
            if len(following_troughs) == 0:
                continue
            
            trough_idx = following_troughs[0]
            
            # Calculate contraction metrics
            high_price = high[peak_idx]
            low_price = low[trough_idx]
            depth_pct = (high_price - low_price) / high_price if high_price > 0 else 0
            duration = trough_idx - peak_idx
            
            # Volume analysis during contraction
            contraction_volume = volume[peak_idx:trough_idx+1]
            avg_vol = np.mean(contraction_volume)
            
            # Volume trend
            if len(contraction_volume) > 5:
                first_half = np.mean(contraction_volume[:len(contraction_volume)//2])
                second_half = np.mean(contraction_volume[len(contraction_volume)//2:])
                if second_half < first_half * 0.8:
                    vol_trend = 'decreasing'
                elif second_half > first_half * 1.2:
                    vol_trend = 'increasing'
                else:
                    vol_trend = 'flat'
            else:
                vol_trend = 'flat'
            
            contraction = Contraction(
                start_idx=peak_idx,
                end_idx=trough_idx,
                high_price=high_price,
                low_price=low_price,
                depth_pct=depth_pct,
                duration_days=duration,
                avg_volume=avg_vol,
                volume_trend=vol_trend
            )
            
            contractions.append(contraction)
        
        return contractions
    
    def _validate_contraction_decay(self, contractions: List[Contraction]) -> bool:
        """
        Validate that contractions are getting tighter.
        
        Each contraction should be <= decay_ratio * previous contraction.
        """
        if len(contractions) < 2:
            return False
        
        # Check depth decay
        depths = [c.depth_pct for c in contractions]
        
        # First contraction shouldn't be too deep
        if depths[0] > self.config.max_first_contraction:
            return False
        
        # Subsequent contractions should decay
        valid_count = 0
        for i in range(1, len(depths)):
            if depths[i] <= depths[i-1] * self.config.contraction_decay_ratio:
                valid_count += 1
        
        # At least 50% should show proper decay
        return valid_count >= (len(depths) - 1) * 0.5
    
    def _calculate_pattern_quality(self, contractions: List[Contraction]) -> float:
        """Calculate overall pattern quality score."""
        if not contractions:
            return 0.0
        
        scores = []
        
        # 1. Number of contractions (2-4 is ideal)
        n = len(contractions)
        if self.config.min_contractions <= n <= 4:
            scores.append(0.9)
        elif n <= self.config.max_contractions:
            scores.append(0.7)
        else:
            scores.append(0.4)
        
        # 2. Depth decay quality
        depths = [c.depth_pct for c in contractions]
        decay_rates = []
        for i in range(1, len(depths)):
            if depths[i-1] > 0:
                decay_rates.append(depths[i] / depths[i-1])
        
        if decay_rates:
            avg_decay = np.mean(decay_rates)
            if 0.3 <= avg_decay <= 0.7:
                scores.append(0.9)  # Ideal decay
            elif avg_decay < 0.3:
                scores.append(0.6)  # Too fast
            elif avg_decay < 1.0:
                scores.append(0.7)  # Some decay
            else:
                scores.append(0.3)  # No decay (bad)
        else:
            scores.append(0.5)
        
        # 3. Final contraction tightness
        if contractions:
            final_depth = contractions[-1].depth_pct
            if final_depth <= self.config.final_contraction_max:
                scores.append(0.9)
            elif final_depth <= 0.15:
                scores.append(0.7)
            else:
                scores.append(0.4)
        
        # 4. Volume characteristics
        vol_scores = []
        for c in contractions:
            if c.volume_trend == 'decreasing':
                vol_scores.append(0.9)
            elif c.volume_trend == 'flat':
                vol_scores.append(0.6)
            else:
                vol_scores.append(0.3)
        
        scores.append(np.mean(vol_scores) if vol_scores else 0.5)
        
        return np.mean(scores)
    
    def _check_final_squeeze(self, df: pd.DataFrame,
                            contractions: List[Contraction]) -> Tuple[bool, float]:
        """
        Check if price is in the final tight squeeze phase.
        
        Returns: (in_squeeze, tightness_score)
        """
        # Look at last 10 days
        recent = df.iloc[-10:]
        
        # Calculate range
        high_range = recent['High'].max() - recent['Low'].min()
        avg_price = recent['Close'].mean()
        
        pct_range = high_range / avg_price if avg_price > 0 else 0
        
        # Compare to recent history (last 60 days)
        hist = df.iloc[-60:-10]
        if len(hist) > 0:
            hist_range = (hist['High'].max() - hist['Low'].min()) / hist['Close'].mean()
        else:
            hist_range = pct_range
        
        # In squeeze if current range is much smaller than historical
        range_ratio = pct_range / hist_range if hist_range > 0 else 1
        
        in_squeeze = range_ratio < 0.5 and pct_range < 0.08
        tightness = 1 - min(1, range_ratio)
        
        return in_squeeze, tightness
    
    def _is_volume_drying_up(self, df: pd.DataFrame) -> bool:
        """Check for volume dry-up condition."""
        cfg = self.config
        
        # Recent volume
        recent_vol = df['Volume'].iloc[-cfg.volume_dry_up_days:]
        
        # 50-day average volume
        avg_vol = df['Volume'].iloc[-50:].mean()
        
        # Check if recent volume is below threshold
        dry_up_threshold = avg_vol * cfg.volume_dry_up_threshold
        
        below_threshold = (recent_vol < dry_up_threshold).sum()
        
        return below_threshold >= cfg.volume_dry_up_days
    
    def _analyze_bb_squeeze(self, df: pd.DataFrame) -> float:
        """
        Analyze Bollinger Band squeeze.
        
        A squeeze indicates low volatility before potential breakout.
        """
        cfg = self.config
        
        close = df['Close']
        
        # Calculate Bollinger Bands
        ma = close.rolling(window=cfg.bb_period).mean()
        std = close.rolling(window=cfg.bb_period).std()
        
        upper = ma + cfg.bb_std * std
        lower = ma - cfg.bb_std * std
        
        # Bandwidth = (Upper - Lower) / Middle
        bandwidth = (upper - lower) / ma
        bandwidth = bandwidth.dropna()
        
        if len(bandwidth) < 120:
            return 0.5
        
        current_bw = bandwidth.iloc[-1]
        
        # Calculate percentile of current bandwidth over 6 months
        bw_6m = bandwidth.iloc[-120:]
        percentile = (bw_6m < current_bw).mean()
        
        # Score: lower bandwidth = higher score
        if percentile <= cfg.bb_squeeze_percentile:
            return 0.95  # Extreme squeeze
        elif percentile <= 0.20:
            return 0.8
        elif percentile <= 0.35:
            return 0.6
        elif percentile <= 0.50:
            return 0.4
        else:
            return 0.2
    
    def _check_volume_dryup(self, df: pd.DataFrame) -> float:
        """Score the volume dry-up condition."""
        recent_5 = df['Volume'].iloc[-5:]
        avg_50 = df['Volume'].iloc[-50:].mean()
        
        # Calculate average recent volume as % of 50-day avg
        vol_ratio = recent_5.mean() / avg_50 if avg_50 > 0 else 1
        
        if vol_ratio < 0.5:
            return 0.95  # Extreme dry-up
        elif vol_ratio < 0.7:
            return 0.8
        elif vol_ratio < 0.9:
            return 0.6
        elif vol_ratio < 1.1:
            return 0.4
        else:
            return 0.2  # Volume still elevated
    
    def _detect_pivot_point(self, df: pd.DataFrame,
                           vcp_pattern: VCPPattern) -> float:
        """
        Score pivot point breakout conditions.
        """
        close = df['Close'].values
        high = df['High'].values
        volume = df['Volume'].values
        
        if vcp_pattern.pivot_price is None:
            # Estimate pivot from recent highs
            pivot = high[-20:].max()
        else:
            pivot = vcp_pattern.pivot_price
        
        current_close = close[-1]
        current_volume = volume[-1]
        avg_volume = volume[-10:].mean()
        
        # Distance to pivot
        dist_to_pivot = (pivot - current_close) / pivot if pivot > 0 else 0
        
        if dist_to_pivot < 0:  # Above pivot = breakout
            # Score based on volume confirmation
            vol_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            if vol_ratio >= self.config.pivot_breakout_volume_multiplier:
                return 0.95  # Strong breakout
            elif vol_ratio >= 1.0:
                return 0.75  # Breakout without volume
            else:
                return 0.5  # Weak breakout
        elif dist_to_pivot < 0.02:  # Within 2% of pivot
            return 0.7
        elif dist_to_pivot < 0.05:  # Within 5%
            return 0.5
        else:
            return 0.3
    
    def _check_pocket_pivot(self, df: pd.DataFrame) -> float:
        """
        Detect pocket pivot signal.
        
        A pocket pivot occurs when:
        1. Stock is near rising MA (10 or 50 day)
        2. Volume > max down volume of past 10 days
        3. Price closes in upper half of range
        """
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Check if today qualifies
        today_close = close.iloc[-1]
        today_vol = volume.iloc[-1]
        today_range = high.iloc[-1] - low.iloc[-1]
        today_mid = (high.iloc[-1] + low.iloc[-1]) / 2
        
        # Near rising 50-day MA?
        ma50 = close.rolling(50).mean()
        ma50_slope = ma50.iloc[-1] - ma50.iloc[-5]
        near_ma = abs(today_close - ma50.iloc[-1]) / ma50.iloc[-1] < 0.03
        ma_rising = ma50_slope > 0
        
        # Down volume check (max of down days in past 10)
        returns = close.pct_change()
        down_days_mask = returns.iloc[-10:] < 0
        down_volumes = volume.iloc[-10:][down_days_mask]
        max_down_vol = down_volumes.max() if len(down_volumes) > 0 else 0
        
        vol_condition = today_vol > max_down_vol
        
        # Close in upper half of range
        price_position = today_close > today_mid if today_range > 0 else True
        
        # Today was an up day
        today_up = close.iloc[-1] > close.iloc[-2]
        
        # Score
        if all([near_ma, ma_rising, vol_condition, price_position, today_up]):
            return 0.9  # Strong pocket pivot
        elif sum([near_ma, ma_rising, vol_condition, price_position, today_up]) >= 3:
            return 0.6  # Partial pocket pivot
        else:
            return 0.3
    
    def _empty_vcp_pattern(self) -> VCPPattern:
        """Return empty VCP pattern for insufficient data cases."""
        return VCPPattern(
            is_valid=False,
            contractions=[],
            pattern_quality=0.0,
            in_final_squeeze=False,
            squeeze_tightness=0.0,
            volume_dry_up_confirmed=False,
            pivot_price=None,
            pivot_breakout_triggered=False,
            total_duration_days=0,
            total_depth_pct=0.0,
            depth_decay_ratio=1.0
        )
