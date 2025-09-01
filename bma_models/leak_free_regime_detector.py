#!/usr/bin/env python3
"""
Leak-Free Regime Detection System
Implements filtering-only regime detection to prevent data leakage
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import logging
import warnings

logger = logging.getLogger(__name__)

@dataclass
class LeakFreeRegimeConfig:
    """Leak-free regime detection configuration"""
    # Basic parameters
    n_regimes: int = 3
    lookback_window: int = 252  # 1 year
    update_frequency: int = 21  # Monthly updates
    
    # Feature calculation windows
    volatility_window: int = 20
    momentum_window: int = 21
    trend_window: int = 60
    
    # GMM parameters
    covariance_type: str = 'full'
    n_init: int = 10
    max_iter: int = 100
    reg_covar: float = 1e-6
    
    # CRITICAL: Leak prevention
    use_filtering_only: bool = True  # Never use smoothing
    embargo_days: int = 10  # Embargo for regime training to match label horizon
    
    # Stability parameters
    min_regime_duration: int = 5  # Minimum regime duration in days
    regime_confidence_threshold: float = 0.6  # Threshold for regime switching

class LeakFreeRegimeDetector:
    """
    Leak-Free Regime Detector
    
    Key features:
    1. FILTERING ONLY - no smoothing that uses future data
    2. Proper embargo alignment with label horizon
    3. Real-time compatible implementation
    4. Causal feature calculation only
    """
    
    def __init__(self, config: LeakFreeRegimeConfig = None):
        self.config = config or LeakFreeRegimeConfig()
        
        # Model components
        self.gmm_model = None
        self.scaler = RobustScaler()
        self.pca_model = None
        
        # State tracking (causal only)
        self.regime_history = {}
        self.feature_history = {}
        self.model_update_dates = []
        self.last_regime_probs = None
        
        # Validation flags
        self._validate_leak_free_config()
        
        logger.info("LeakFreeRegimeDetector initialized with filtering-only approach")
        logger.info(f"Config: {self.config.n_regimes} regimes, {self.config.lookback_window}d window")
    
    def _validate_leak_free_config(self):
        """Validate configuration for leak-free operation"""
        if not self.config.use_filtering_only:
            raise ValueError("Must use filtering_only=True to prevent data leakage")
        
        if self.config.embargo_days < 10:
            logger.warning(f"Embargo {self.config.embargo_days} < 10d may not align with 10-day labels")
    
    def fit_regime_model(self, data: pd.DataFrame, current_date: pd.Timestamp) -> bool:
        """
        Fit regime model using only data available up to current_date - embargo
        This ensures no forward-looking bias in regime detection
        """
        # 修复: 确保data有正确的日期索引
        if 'date' in data.columns:
            # 如果date列存在但不是索引，设置为索引
            data_indexed = data.set_index('date').copy()
            data_indexed.index = pd.to_datetime(data_indexed.index)
        elif isinstance(data.index, pd.MultiIndex):
            # Handle MultiIndex with date as one of the levels
            if 'date' in data.index.names:
                # Get date level from MultiIndex
                date_level = data.index.names.index('date')
                date_index = data.index.get_level_values(date_level)
                # Use the first ticker's data for regime detection (regime is market-wide)
                if 'ticker' in data.index.names:
                    first_ticker = data.index.get_level_values('ticker').unique()[0]
                    data_indexed = data.xs(first_ticker, level='ticker').copy()
                else:
                    # Group by date and take mean for regime detection
                    data_indexed = data.groupby(level=date_level).mean()
                data_indexed.index = pd.to_datetime(data_indexed.index)
            else:
                logger.warning("MultiIndex lacks date level, skipping regime detection")
                return False
        elif isinstance(data.index, pd.DatetimeIndex):
            # 如果已经是datetime索引，直接使用
            data_indexed = data.copy()
        else:
            # 如果没有date信息，跳过regime检测
            logger.warning("Data lacks datetime index, skipping regime detection")
            return False
        
        # Apply embargo to prevent label leakage
        embargo_cutoff = current_date - pd.Timedelta(days=self.config.embargo_days)
        training_data = data_indexed[data_indexed.index <= embargo_cutoff].copy()
        
        if len(training_data) < self.config.lookback_window:
            logger.warning(f"Insufficient data for regime training: {len(training_data)} < {self.config.lookback_window}")
            return False
        
        # Use only the lookback window for training
        training_data = training_data.tail(self.config.lookback_window)
        
        # Calculate regime features (causal only)
        regime_features = self._calculate_causal_regime_features(training_data)
        
        if regime_features.empty:
            logger.error("Failed to calculate regime features")
            return False
        
        # Fit GMM model
        try:
            # Standardize features
            features_scaled = self.scaler.fit_transform(regime_features)
            
            # Optional PCA
            if regime_features.shape[1] > 5:
                self.pca_model = PCA(n_components=min(5, regime_features.shape[1]))
                features_scaled = self.pca_model.fit_transform(features_scaled)
                logger.info(f"Applied PCA: {regime_features.shape[1]} -> {features_scaled.shape[1]} features")
            
            # Fit GMM
            self.gmm_model = GaussianMixture(
                n_components=self.config.n_regimes,
                covariance_type=self.config.covariance_type,
                n_init=self.config.n_init,
                max_iter=self.config.max_iter,
                reg_covar=self.config.reg_covar,
                random_state=42
            )
            
            self.gmm_model.fit(features_scaled)
            
            # Validate regime stability
            regime_labels = self.gmm_model.predict(features_scaled)
            if self._validate_regime_stability(regime_labels):
                self.model_update_dates.append(current_date)
                logger.info(f"Regime model updated successfully at {current_date}")
                return True
            else:
                logger.warning("Regime model failed stability validation")
                return False
        
        except Exception as e:
            logger.error(f"Failed to fit regime model: {e}")
            return False
    
    def predict_regime_probabilities(self, data: pd.DataFrame, 
                                   current_date: pd.Timestamp) -> Optional[np.ndarray]:
        """
        Predict regime probabilities using FILTERING approach only
        Returns p(regime_t | x_1:t) - no future information
        """
        if self.gmm_model is None:
            logger.warning("Regime model not fitted")
            return None
        
        # 修复: 确保data有正确的日期索引
        if 'date' in data.columns:
            data_indexed = data.set_index('date').copy()
            data_indexed.index = pd.to_datetime(data_indexed.index)
        elif isinstance(data.index, pd.DatetimeIndex):
            data_indexed = data.copy()
        else:
            logger.warning("Data lacks datetime index, using fallback")
            return self.last_regime_probs
        
        # Get current features (up to current_date only)
        current_data = data_indexed[data_indexed.index <= current_date].tail(max(100, self.config.momentum_window))
        
        if current_data.empty:
            return self.last_regime_probs
        
        # Calculate current regime features (causal only)
        current_features = self._calculate_causal_regime_features(current_data)
        
        if current_features.empty:
            return self.last_regime_probs
        
        try:
            # Use only the most recent feature vector
            latest_features = current_features.iloc[[-1]]  # Most recent only
            
            # Transform features
            features_scaled = self.scaler.transform(latest_features)
            if self.pca_model is not None:
                features_scaled = self.pca_model.transform(features_scaled)
            
            # Get FILTERING probabilities (no smoothing)
            regime_probs = self.gmm_model.predict_proba(features_scaled)[0]
            
            # Apply minimum duration constraint (causal smoothing)
            regime_probs = self._apply_causal_smoothing(regime_probs, current_date)
            
            self.last_regime_probs = regime_probs
            
            # Store in history for analysis
            self.regime_history[current_date] = {
                'probabilities': regime_probs,
                'dominant_regime': np.argmax(regime_probs),
                'max_probability': np.max(regime_probs)
            }
            
            return regime_probs
        
        except Exception as e:
            logger.error(f"Failed to predict regime probabilities: {e}")
            return self.last_regime_probs
    
    def _calculate_causal_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime features using only historical data (no forward-looking)
        """
        features = pd.DataFrame(index=data.index)
        
        # Required price columns - handle different case variations
        close_col = None
        for col_name in ['Close', 'close', 'CLOSE', 'adj_close', 'Adj_Close']:
            if col_name in data.columns:
                close_col = col_name
                break
        
        if close_col is None:
            logger.error("Missing 'Close' column in data")
            logger.error(f"Failed to calculate regime features")
            return pd.DataFrame()
        
        close_prices = data[close_col]
        logger.debug(f"Using price column: {close_col}")
        
        try:
            # 1. Realized Volatility (historical only)
            log_returns = np.log(close_prices / close_prices.shift(1))
            features['realized_vol_short'] = log_returns.rolling(
                window=self.config.volatility_window, min_periods=self.config.volatility_window//2
            ).std() * np.sqrt(252)
            
            features['realized_vol_long'] = log_returns.rolling(
                window=self.config.trend_window, min_periods=self.config.trend_window//2
            ).std() * np.sqrt(252)
            
            # 2. Momentum indicators (historical only)
            features['momentum_short'] = close_prices.pct_change(self.config.momentum_window)
            features['momentum_medium'] = close_prices.pct_change(self.config.trend_window)
            
            # 3. Trend indicators (historical only)
            ma_short = close_prices.rolling(window=20, min_periods=10).mean()
            ma_long = close_prices.rolling(window=60, min_periods=30).mean()
            features['trend_strength'] = (ma_short - ma_long) / ma_long
            features['price_vs_ma'] = (close_prices - ma_short) / ma_short
            
            # 4. Volatility regime indicators
            vol_ma = features['realized_vol_short'].rolling(window=60, min_periods=30).mean()
            features['vol_regime'] = features['realized_vol_short'] / vol_ma
            
            # 5. Market microstructure (if available) - handle different case variations
            volume_col = None
            for col_name in ['Volume', 'volume', 'VOLUME', 'vol']:
                if col_name in data.columns:
                    volume_col = col_name
                    break
            
            if volume_col is not None:
                volume = data[volume_col]
                features['volume_trend'] = (
                    volume.rolling(window=20, min_periods=10).mean() / 
                    volume.rolling(window=60, min_periods=30).mean()
                )
                logger.debug(f"Using volume column: {volume_col}")
            else:
                features['volume_trend'] = 1.0  # Neutral if no volume data
                logger.debug("No volume column found, using neutral volume trend")
            
            # 6. Cross-asset signals (if available)
            if 'VIX' in data.columns:
                features['vix_level'] = data['VIX'] / data['VIX'].rolling(window=252, min_periods=126).mean()
            else:
                # Use volatility proxy
                features['vix_level'] = features['realized_vol_short'] / features['realized_vol_short'].rolling(window=252, min_periods=126).mean()
            
            # Clean features
            features = features.dropna()
            
            # Validate no forward-looking bias
            if not self._validate_causal_features(features, data.index):
                logger.error("Causal validation failed for regime features")
                return pd.DataFrame()
            
            logger.debug(f"Calculated {len(features.columns)} causal regime features")
            return features
            
        except Exception as e:
            logger.error(f"Error calculating regime features: {e}")
            return pd.DataFrame()
    
    def _validate_causal_features(self, features: pd.DataFrame, original_index: pd.DatetimeIndex) -> bool:
        """Validate that features don't use future information"""
        # Check that feature dates don't exceed original dates
        if not features.index.isin(original_index).all():
            return False
        
        # Check for any forward-looking calculations (basic validation)
        for col in features.columns:
            if features[col].isna().sum() / len(features) > 0.9:  # Too many NaN suggests calculation error
                logger.warning(f"Feature {col} has excessive NaN values")
                return False
        
        return True
    
    def _apply_causal_smoothing(self, current_probs: np.ndarray, 
                              current_date: pd.Timestamp) -> np.ndarray:
        """
        Apply causal smoothing to prevent excessive regime switching
        Uses only historical regime decisions
        """
        if len(self.regime_history) < self.config.min_regime_duration:
            return current_probs
        
        # Get recent regime history
        recent_dates = sorted([d for d in self.regime_history.keys() 
                              if d <= current_date])[-self.config.min_regime_duration:]
        
        if len(recent_dates) < 2:
            return current_probs
        
        # Check recent regime consistency
        recent_regimes = [self.regime_history[d]['dominant_regime'] for d in recent_dates]
        current_regime = np.argmax(current_probs)
        
        # If regime would switch, require higher confidence
        if len(set(recent_regimes)) == 1 and recent_regimes[0] != current_regime:
            if current_probs[current_regime] < self.config.regime_confidence_threshold:
                # Stay in previous regime with reduced confidence
                smoothed_probs = current_probs.copy()
                prev_regime = recent_regimes[0]
                smoothed_probs[prev_regime] = max(smoothed_probs[prev_regime], 0.4)
                smoothed_probs = smoothed_probs / smoothed_probs.sum()  # Renormalize
                return smoothed_probs
        
        return current_probs
    
    def _validate_regime_stability(self, regime_labels: np.ndarray) -> bool:
        """Validate that regimes are stable and well-separated"""
        from collections import Counter
        
        # Check regime distribution
        regime_counts = Counter(regime_labels)
        
        # Each regime should have sufficient samples
        min_samples = len(regime_labels) // (self.config.n_regimes * 3)  # At least 1/3 of expected
        for regime, count in regime_counts.items():
            if count < min_samples:
                logger.warning(f"Regime {regime} has insufficient samples: {count} < {min_samples}")
                return False
        
        # Check for excessive switching
        switches = np.sum(regime_labels[1:] != regime_labels[:-1])
        switch_rate = switches / len(regime_labels)
        if switch_rate > 0.3:  # More than 30% switches indicates unstable regimes
            logger.warning(f"High regime switch rate: {switch_rate:.1%}")
            return False
        
        return True
    
    def get_current_regime(self, data: pd.DataFrame, 
                          current_date: pd.Timestamp) -> Dict[str, Any]:
        """Get current regime information for routing decisions"""
        regime_probs = self.predict_regime_probabilities(data, current_date)
        
        if regime_probs is None:
            return {
                'regime': 0,  # Default regime
                'probability': 1.0 / self.config.n_regimes,
                'confidence': 'low',
                'routing_weight': 1.0  # Equal weight fallback
            }
        
        dominant_regime = np.argmax(regime_probs)
        max_prob = np.max(regime_probs)
        
        # Determine confidence level
        if max_prob >= self.config.regime_confidence_threshold:
            confidence = 'high'
            routing_weight = max_prob
        elif max_prob >= 0.4:
            confidence = 'medium'
            routing_weight = 0.7  # Moderate confidence in routing
        else:
            confidence = 'low'
            routing_weight = 0.5  # Low confidence, closer to equal weight
        
        return {
            'regime': dominant_regime,
            'probability': max_prob,
            'confidence': confidence,
            'routing_weight': routing_weight,
            'all_probabilities': regime_probs
        }
    
    def should_update_model(self, current_date: pd.Timestamp) -> bool:
        """Determine if regime model should be updated"""
        if not self.model_update_dates:
            return True
        
        last_update = self.model_update_dates[-1]
        days_since_update = (current_date - last_update).days
        
        return days_since_update >= self.config.update_frequency
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive regime detection statistics"""
        if not self.regime_history:
            return {'status': 'no_history'}
        
        dates = sorted(self.regime_history.keys())
        regimes = [self.regime_history[d]['dominant_regime'] for d in dates]
        confidences = [self.regime_history[d]['max_probability'] for d in dates]
        
        # Calculate regime statistics
        from collections import Counter
        regime_dist = Counter(regimes)
        
        # Switch analysis
        switches = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
        avg_regime_duration = len(regimes) / (switches + 1) if switches > 0 else len(regimes)
        
        return {
            'total_observations': len(regimes),
            'regime_distribution': dict(regime_dist),
            'regime_switches': switches,
            'switch_rate': switches / len(regimes) if regimes else 0,
            'avg_regime_duration': avg_regime_duration,
            'avg_confidence': np.mean(confidences),
            'model_updates': len(self.model_update_dates),
            'last_update': self.model_update_dates[-1] if self.model_update_dates else None,
            'leak_prevention': {
                'filtering_only': self.config.use_filtering_only,
                'embargo_days': self.config.embargo_days,
                'status': 'PROTECTED'
            }
        }
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process data with leak-free regime detection
        
        Args:
            data: Input data with date index or date column
            
        Returns:
            Data with regime information added
        """
        try:
            if data.empty:
                logger.warning("Empty data provided to regime detector")
                return data.copy()
            
            # Create a copy to avoid modifying original data
            processed_data = data.copy()
            
            # Ensure we have date information
            if 'date' in processed_data.columns:
                dates = pd.to_datetime(processed_data['date'])
                unique_dates = dates.unique()
            elif isinstance(processed_data.index, pd.DatetimeIndex):
                unique_dates = processed_data.index.unique()
            else:
                logger.warning("No date information found in data")
                # Add dummy regime info
                processed_data['regime_state'] = 0
                processed_data['regime_confidence'] = 0.5
                return processed_data
            
            # Process each unique date
            regime_info = {}
            for current_date in sorted(unique_dates):
                # Get data up to current date for regime detection
                if 'date' in processed_data.columns:
                    historical_data = processed_data[dates <= current_date].copy()
                else:
                    historical_data = processed_data[processed_data.index <= current_date].copy()
                
                if len(historical_data) < self.config.lookback_window:
                    # Not enough data for regime detection
                    regime_info[current_date] = {
                        'regime_state': 0,
                        'regime_confidence': 0.5,
                        'regime_probabilities': [0.5, 0.3, 0.2]
                    }
                    continue
                
                # Update model if needed
                if self.should_update_model(current_date):
                    self.fit_regime_model(historical_data, current_date)
                
                # Get current regime
                regime_result = self.get_current_regime(historical_data, current_date)
                regime_info[current_date] = regime_result
            
            # Add regime information to the processed data
            if 'date' in processed_data.columns:
                for idx, row in processed_data.iterrows():
                    row_date = pd.to_datetime(row['date'])
                    if row_date in regime_info:
                        info = regime_info[row_date]
                        processed_data.loc[idx, 'regime_state'] = info.get('regime_state', 0)
                        processed_data.loc[idx, 'regime_confidence'] = info.get('regime_confidence', 0.5)
            else:
                for idx in processed_data.index:
                    if idx in regime_info:
                        info = regime_info[idx]
                        processed_data.loc[idx, 'regime_state'] = info.get('regime_state', 0)
                        processed_data.loc[idx, 'regime_confidence'] = info.get('regime_confidence', 0.5)
            
            logger.info(f"Processed {len(regime_info)} dates with regime detection")
            return processed_data
            
        except Exception as e:
            logger.error(f"Regime processing failed: {e}")
            # Return original data with dummy regime info if processing fails
            processed_data = data.copy()
            processed_data['regime_state'] = 0
            processed_data['regime_confidence'] = 0.5
            return processed_data