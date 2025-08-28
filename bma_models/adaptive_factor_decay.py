#!/usr/bin/env python3
"""
Adaptive Factor Decay System
Implements factor-family specific decay half-lives instead of uniform decay
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import datetime

logger = logging.getLogger(__name__)

class FactorFamily(Enum):
    """Factor family classifications"""
    MOMENTUM = "momentum"
    REVERSAL = "reversal"  
    VOLATILITY = "volatility"
    QUALITY = "quality"
    VALUE = "value"
    SENTIMENT = "sentiment"
    VOLUME = "volume"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MICROSTRUCTURE = "microstructure"

@dataclass
class FactorDecayConfig:
    """Configuration for factor-family specific decay"""
    # Family-specific half-lives (in days)
    family_half_lives: Dict[FactorFamily, int] = field(default_factory=lambda: {
        FactorFamily.MOMENTUM: 8,      # Medium decay for momentum
        FactorFamily.REVERSAL: 5,      # Fast decay for mean reversion
        FactorFamily.VOLATILITY: 6,    # Fast decay for volatility signals
        FactorFamily.QUALITY: 60,      # Slow decay for quality factors
        FactorFamily.VALUE: 90,        # Very slow decay for value factors
        FactorFamily.SENTIMENT: 10,    # Medium-fast decay for sentiment
        FactorFamily.VOLUME: 7,        # Fast decay for volume/liquidity
        FactorFamily.TECHNICAL: 12,    # Medium decay for technical
        FactorFamily.FUNDAMENTAL: 120, # Very slow decay for fundamentals
        FactorFamily.MICROSTRUCTURE: 3 # Very fast decay for microstructure
    })
    
    # Regime-adaptive adjustments
    enable_regime_adaptation: bool = True
    high_vol_multiplier: float = 0.7  # Shorten half-life in high vol
    low_vol_multiplier: float = 1.3   # Lengthen half-life in low vol
    
    # Default fallback
    default_half_life: int = 15  # For unclassified factors
    
    # Optimization parameters
    enable_factor_optimization: bool = True
    optimization_lookback: int = 252  # Days for IC optimization
    min_ic_threshold: float = 0.01    # Minimum IC to consider factor

@dataclass
class FactorClassification:
    """Factor classification with metadata"""
    factor_name: str
    family: FactorFamily
    base_half_life: int
    effective_half_life: int
    confidence: float  # Confidence in classification (0-1)
    
class AdaptiveFactorDecay:
    """
    Adaptive Factor Decay System
    
    Implements intelligent decay based on:
    1. Factor family characteristics
    2. Market regime state
    3. Historical IC optimization
    """
    
    def __init__(self, config: FactorDecayConfig = None):
        self.config = config or FactorDecayConfig()
        
        # Factor classification database
        self.factor_classifications = {}
        self.factor_patterns = self._initialize_factor_patterns()
        
        # Optimization results
        self.optimization_results = {}
        
        logger.info("AdaptiveFactorDecay initialized")
        logger.info(f"Family half-lives: {dict(self.config.family_half_lives)}")
    
    def _initialize_factor_patterns(self) -> Dict[str, FactorFamily]:
        """Initialize factor name patterns for classification"""
        return {
            # Momentum patterns
            'momentum': FactorFamily.MOMENTUM,
            'trend': FactorFamily.MOMENTUM,
            'price_momentum': FactorFamily.MOMENTUM,
            'return': FactorFamily.MOMENTUM,
            'ma_': FactorFamily.MOMENTUM,
            'ema_': FactorFamily.MOMENTUM,
            'rsi': FactorFamily.MOMENTUM,
            'macd': FactorFamily.MOMENTUM,
            
            # Reversal patterns
            'reversal': FactorFamily.REVERSAL,
            'mean_reversion': FactorFamily.REVERSAL,
            'contrarian': FactorFamily.REVERSAL,
            'short_term_rev': FactorFamily.REVERSAL,
            'overnight': FactorFamily.REVERSAL,
            
            # Volatility patterns
            'volatility': FactorFamily.VOLATILITY,
            'vol_': FactorFamily.VOLATILITY,
            'realized_vol': FactorFamily.VOLATILITY,
            'garch': FactorFamily.VOLATILITY,
            'vix': FactorFamily.VOLATILITY,
            'beta': FactorFamily.VOLATILITY,
            'idiosyncratic': FactorFamily.VOLATILITY,
            
            # Quality patterns
            'quality': FactorFamily.QUALITY,
            'stability': FactorFamily.QUALITY,
            'earnings_quality': FactorFamily.QUALITY,
            'accruals': FactorFamily.QUALITY,
            'profitability': FactorFamily.QUALITY,
            
            # Value patterns
            'value': FactorFamily.VALUE,
            'pe_': FactorFamily.VALUE,
            'pb_': FactorFamily.VALUE,
            'pcf': FactorFamily.VALUE,
            'ev_': FactorFamily.VALUE,
            'book_to_market': FactorFamily.VALUE,
            
            # Sentiment patterns
            'sentiment': FactorFamily.SENTIMENT,
            'news_': FactorFamily.SENTIMENT,
            'analyst': FactorFamily.SENTIMENT,
            'fear_greed': FactorFamily.SENTIMENT,
            'put_call': FactorFamily.SENTIMENT,
            'short_interest': FactorFamily.SENTIMENT,
            
            # Volume patterns
            'volume': FactorFamily.VOLUME,
            'turnover': FactorFamily.VOLUME,
            'liquidity': FactorFamily.VOLUME,
            'bid_ask': FactorFamily.VOLUME,
            'market_cap': FactorFamily.VOLUME,
            'dollar_volume': FactorFamily.VOLUME,
            
            # Technical patterns
            'technical': FactorFamily.TECHNICAL,
            'bollinger': FactorFamily.TECHNICAL,
            'stochastic': FactorFamily.TECHNICAL,
            'williams': FactorFamily.TECHNICAL,
            'cci': FactorFamily.TECHNICAL,
            
            # Fundamental patterns
            'fundamental': FactorFamily.FUNDAMENTAL,
            'earnings': FactorFamily.FUNDAMENTAL,
            'revenue': FactorFamily.FUNDAMENTAL,
            'debt': FactorFamily.FUNDAMENTAL,
            'cash_flow': FactorFamily.FUNDAMENTAL,
            'roe': FactorFamily.FUNDAMENTAL,
            'roa': FactorFamily.FUNDAMENTAL,
            
            # Microstructure patterns
            'microstructure': FactorFamily.MICROSTRUCTURE,
            'order_flow': FactorFamily.MICROSTRUCTURE,
            'trade_size': FactorFamily.MICROSTRUCTURE,
            'quote': FactorFamily.MICROSTRUCTURE,
            'tick': FactorFamily.MICROSTRUCTURE,
        }
    
    def classify_factors(self, factor_names: List[str]) -> Dict[str, FactorClassification]:
        """Classify factors into families and assign decay parameters"""
        classifications = {}
        
        for factor_name in factor_names:
            family, confidence = self._classify_single_factor(factor_name)
            base_half_life = self.config.family_half_lives.get(family, self.config.default_half_life)
            
            classification = FactorClassification(
                factor_name=factor_name,
                family=family,
                base_half_life=base_half_life,
                effective_half_life=base_half_life,  # Will be adjusted by regime
                confidence=confidence
            )
            
            classifications[factor_name] = classification
            
        self.factor_classifications.update(classifications)
        
        logger.info(f"Classified {len(factor_names)} factors:")
        family_counts = {}
        for classification in classifications.values():
            family = classification.family
            family_counts[family] = family_counts.get(family, 0) + 1
        
        for family, count in family_counts.items():
            logger.info(f"  {family.value}: {count} factors")
        
        return classifications
    
    def _classify_single_factor(self, factor_name: str) -> Tuple[FactorFamily, float]:
        """Classify a single factor with confidence score"""
        factor_lower = factor_name.lower()
        
        # Pattern matching with confidence
        matches = []
        for pattern, family in self.factor_patterns.items():
            if pattern in factor_lower:
                # Longer matches get higher confidence
                confidence = len(pattern) / len(factor_lower)
                matches.append((family, confidence))
        
        if matches:
            # Return highest confidence match
            best_family, best_confidence = max(matches, key=lambda x: x[1])
            return best_family, min(best_confidence * 2, 1.0)  # Scale confidence
        
        # No clear pattern - try alpha prefix classification
        if factor_name.startswith('alpha_'):
            alpha_num = factor_name.split('_')[1] if '_' in factor_name else '0'
            try:
                num = int(alpha_num)
                # Use heuristic based on alpha number
                if num <= 10:
                    return FactorFamily.MOMENTUM, 0.3
                elif num <= 20:
                    return FactorFamily.REVERSAL, 0.3
                elif num <= 30:
                    return FactorFamily.VOLATILITY, 0.3
                else:
                    return FactorFamily.TECHNICAL, 0.3
            except ValueError:
                pass
        
        # Default classification
        return FactorFamily.TECHNICAL, 0.1
    
    def optimize_factor_half_lives(self, factor_data: pd.DataFrame, 
                                 target: pd.Series, 
                                 factor_names: List[str] = None) -> Dict[str, Dict]:
        """Optimize factor half-lives based on IC performance"""
        if not self.config.enable_factor_optimization:
            logger.info("Factor optimization disabled")
            return {}
        
        if factor_names is None:
            factor_names = list(self.factor_classifications.keys())
        
        optimization_results = {}
        
        logger.info(f"Optimizing half-lives for {len(factor_names)} factors")
        
        for factor_name in factor_names:
            if factor_name not in factor_data.columns:
                continue
            
            classification = self.factor_classifications.get(factor_name)
            if not classification:
                continue
            
            # Test range around base half-life
            base_hl = classification.base_half_life
            test_half_lives = self._get_test_half_lives(base_hl, classification.family)
            
            ic_results = {}
            for test_hl in test_half_lives:
                ic = self._calculate_factor_ic_with_decay(
                    factor_data[factor_name], target, test_hl
                )
                ic_results[test_hl] = ic
            
            # Find optimal half-life
            if ic_results:
                optimal_hl = max(ic_results.items(), key=lambda x: abs(x[1]))[0]
                optimal_ic = ic_results[optimal_hl]
                
                optimization_results[factor_name] = {
                    'base_half_life': base_hl,
                    'optimal_half_life': optimal_hl,
                    'base_ic': ic_results.get(base_hl, 0.0),
                    'optimal_ic': optimal_ic,
                    'improvement': abs(optimal_ic) - abs(ic_results.get(base_hl, 0.0)),
                    'all_results': ic_results
                }
                
                # Update classification
                if abs(optimal_ic) > self.config.min_ic_threshold:
                    classification.effective_half_life = optimal_hl
        
        self.optimization_results.update(optimization_results)
        
        # Log optimization summary
        improved = sum(1 for r in optimization_results.values() if r['improvement'] > 0.001)
        logger.info(f"Optimization complete: {improved}/{len(optimization_results)} factors improved")
        
        return optimization_results
    
    def _get_test_half_lives(self, base_hl: int, family: FactorFamily) -> List[int]:
        """Get test half-lives around base value"""
        # Family-specific test ranges
        if family in [FactorFamily.MICROSTRUCTURE, FactorFamily.REVERSAL]:
            # Fast factors: test narrow range
            return [max(1, base_hl - 2), base_hl, base_hl + 2, base_hl + 5]
        elif family in [FactorFamily.QUALITY, FactorFamily.FUNDAMENTAL, FactorFamily.VALUE]:
            # Slow factors: test wider range
            return [base_hl // 2, base_hl, int(base_hl * 1.5), base_hl * 2]
        else:
            # Medium factors: balanced range
            return [max(1, base_hl - 3), base_hl, base_hl + 3, base_hl + 8]
    
    def _calculate_factor_ic_with_decay(self, factor_data: pd.Series, 
                                      target: pd.Series, half_life: int) -> float:
        """Calculate IC with exponential decay weighting"""
        from scipy.stats import pearsonr
        
        # Align data
        common_idx = factor_data.index.intersection(target.index)
        if len(common_idx) < 50:  # Minimum samples
            return 0.0
        
        factor_aligned = factor_data.loc[common_idx]
        target_aligned = target.loc[common_idx]
        
        # Remove NaN pairs
        valid_mask = ~(factor_aligned.isna() | target_aligned.isna())
        factor_clean = factor_aligned[valid_mask]
        target_clean = target_aligned[valid_mask]
        
        if len(factor_clean) < 20:
            return 0.0
        
        # Apply exponential decay weights
        n_samples = len(factor_clean)
        decay_factor = np.exp(-np.log(2) / half_life)  # Convert half-life to decay factor
        
        # Create weights (more recent = higher weight)
        weights = np.array([decay_factor ** (n_samples - 1 - i) for i in range(n_samples)])
        weights = weights / weights.sum()  # Normalize
        
        # Calculate weighted correlation
        try:
            # Use weighted Pearson correlation
            weighted_ic = self._weighted_correlation(factor_clean.values, target_clean.values, weights)
            return weighted_ic if not np.isnan(weighted_ic) else 0.0
        except:
            return 0.0
    
    def _weighted_correlation(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Calculate weighted Pearson correlation"""
        # Weighted means
        x_mean = np.average(x, weights=weights)
        y_mean = np.average(y, weights=weights)
        
        # Weighted covariance and variances
        cov = np.average((x - x_mean) * (y - y_mean), weights=weights)
        var_x = np.average((x - x_mean) ** 2, weights=weights)
        var_y = np.average((y - y_mean) ** 2, weights=weights)
        
        # Correlation
        if var_x * var_y > 0:
            return cov / np.sqrt(var_x * var_y)
        else:
            return 0.0
    
    def adjust_for_regime(self, regime_state: Dict[str, Any]) -> None:
        """Adjust factor half-lives based on market regime"""
        if not self.config.enable_regime_adaptation:
            return
        
        # Determine regime characteristics
        regime_type = regime_state.get('regime', 0)
        volatility_regime = regime_state.get('volatility', 'normal')  # high/normal/low
        
        # Adjustment multipliers based on regime
        if volatility_regime == 'high' or regime_type == 2:  # High vol regime
            multiplier = self.config.high_vol_multiplier
            logger.info(f"High volatility regime detected, shortening half-lives by {1-multiplier:.1%}")
        elif volatility_regime == 'low' or regime_type == 0:  # Low vol regime
            multiplier = self.config.low_vol_multiplier
            logger.info(f"Low volatility regime detected, lengthening half-lives by {multiplier-1:.1%}")
        else:
            multiplier = 1.0  # Normal regime
        
        # Apply adjustments
        adjusted_count = 0
        for factor_name, classification in self.factor_classifications.items():
            old_hl = classification.effective_half_life
            new_hl = max(1, int(classification.base_half_life * multiplier))
            
            if new_hl != old_hl:
                classification.effective_half_life = new_hl
                adjusted_count += 1
        
        if adjusted_count > 0:
            logger.info(f"Adjusted half-lives for {adjusted_count} factors based on regime")
    
    def get_factor_weights(self, factor_data: pd.DataFrame, 
                          current_date: pd.Timestamp = None) -> Dict[str, np.ndarray]:
        """Get exponential decay weights for each factor"""
        if current_date is None:
            current_date = factor_data.index[-1]
        
        weights = {}
        
        for factor_name, classification in self.factor_classifications.items():
            if factor_name not in factor_data.columns:
                continue
            
            # Get factor data up to current date
            factor_series = factor_data[factor_name]
            
            # Handle type mismatch between index and current_date
            if isinstance(current_date, (pd.Timestamp, datetime.datetime)):
                # If index is datetime-like, compare directly
                if pd.api.types.is_datetime64_any_dtype(factor_series.index):
                    factor_series = factor_series[factor_series.index <= current_date]
                else:
                    # If index is not datetime, use all available data
                    # This assumes factor_series is already chronologically ordered
                    pass  # Use all available factor_series data
            else:
                # If current_date is not datetime, try to convert or use position-based filtering
                try:
                    factor_series = factor_series[factor_series.index <= current_date]
                except TypeError:
                    # If comparison fails, use all data (assume chronological order)
                    pass
            
            if factor_series.empty:
                weights[factor_name] = np.array([])
                continue
            
            # Calculate exponential weights
            half_life = classification.effective_half_life
            n_samples = len(factor_series)
            
            # Decay factor from half-life
            decay_factor = np.exp(-np.log(2) / half_life)
            
            # Create weights (more recent = higher weight)
            factor_weights = np.array([
                decay_factor ** (n_samples - 1 - i) for i in range(n_samples)
            ])
            
            # Normalize weights
            factor_weights = factor_weights / factor_weights.sum()
            
            weights[factor_name] = factor_weights
        
        return weights
    
    def get_decay_summary(self) -> Dict[str, Any]:
        """Get summary of factor decay configuration"""
        if not self.factor_classifications:
            return {'status': 'no_factors_classified'}
        
        # Group by family
        family_summary = {}
        for classification in self.factor_classifications.values():
            family = classification.family.value
            if family not in family_summary:
                family_summary[family] = {
                    'count': 0,
                    'base_half_lives': [],
                    'effective_half_lives': [],
                    'avg_confidence': []
                }
            
            family_summary[family]['count'] += 1
            family_summary[family]['base_half_lives'].append(classification.base_half_life)
            family_summary[family]['effective_half_lives'].append(classification.effective_half_life)
            family_summary[family]['avg_confidence'].append(classification.confidence)
        
        # Calculate averages
        for family_data in family_summary.values():
            family_data['avg_base_half_life'] = np.mean(family_data['base_half_lives'])
            family_data['avg_effective_half_life'] = np.mean(family_data['effective_half_lives'])
            family_data['avg_confidence'] = np.mean(family_data['avg_confidence'])
            
            # Remove raw lists for cleaner output
            del family_data['base_half_lives']
            del family_data['effective_half_lives']
        
        return {
            'total_factors': len(self.factor_classifications),
            'family_breakdown': family_summary,
            'optimization_enabled': self.config.enable_factor_optimization,
            'regime_adaptation_enabled': self.config.enable_regime_adaptation,
            'optimized_factors': len(self.optimization_results)
        }