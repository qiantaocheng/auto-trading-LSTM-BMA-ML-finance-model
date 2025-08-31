"""
Enhanced Alpha Configuration Module
====================================
Optimized alpha factor configuration with increased weight (55-70%)
and reduced dimensionality (40+ -> 10 features)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EnhancedAlphaConfig:
    """Enhanced configuration for alpha factor processing"""
    
    # =========================================================================
    # Alpha Weight Configuration (55-70% of total features)
    # =========================================================================
    ALPHA_TARGET_WEIGHT_MIN: float = 0.55  # Minimum 55% weight
    ALPHA_TARGET_WEIGHT_MAX: float = 0.70  # Maximum 70% weight
    
    # =========================================================================
    # Dimensionality Reduction Settings
    # =========================================================================
    MAX_ALPHA_FEATURES: int = 10  # Reduce from 40+ to 10
    MIN_ALPHA_FEATURES: int = 6   # Minimum required features
    
    # PCA Configuration
    PCA_N_COMPONENTS: int = 5     # First 5 principal components
    PCA_VARIANCE_THRESHOLD: float = 0.85  # Explain 85% variance
    
    # Clustering Configuration  
    N_CLUSTERS: int = 3           # 3 clusters for alpha grouping
    
    # Composite Feature Settings
    COMPOSITE_N_FEATURES: int = 2  # 2 composite features
    
    # =========================================================================
    # Data Processing Configuration
    # =========================================================================
    # Time alignment settings to fix violations
    DEFAULT_LAG: int = 5          # T-5 lag for all alpha factors
    SAFETY_BUFFER: int = 2        # Additional safety buffer
    STRICT_TIME_CHECK: bool = True  # Enable strict time validation
    
    # MultiIndex handling
    HANDLE_MULTIINDEX: bool = True  # Properly handle MultiIndex
    RESET_INDEX_FOR_PROCESSING: bool = True  # Reset index before processing
    
    # Missing data handling
    FILL_MISSING_AMOUNT: bool = True  # Generate amount column if missing
    AMOUNT_PROXY_METHOD: str = 'volume_close'  # amount = volume * close
    
    # Winsorization settings
    WINSORIZE_LOWER: float = 0.01
    WINSORIZE_UPPER: float = 0.99
    USE_MAD_WINSORIZE: bool = True  # Use more robust MAD method
    
    # Neutralization settings
    NEUTRALIZE_BY_INDUSTRY: bool = False  # Disable to avoid MultiIndex issues
    NEUTRALIZE_BY_MARKET: bool = False    # Disable to avoid errors
    
    # =========================================================================
    # Feature Selection Priority (Top 10 factors by importance)
    # =========================================================================
    TOP_ALPHA_FACTORS = [
        'momentum_d22_d66',          # Top momentum signal
        'reversal_d5_d22',           # Mean reversion
        'volatility_d22_d66',        # Risk signal
        'news_sentiment_factor',     # Sentiment signal
        'qmj_quality_score',         # Quality metric
        'earnings_yield_ep',         # Value signal
        'amihud_illiquidity',        # Liquidity
        'low_beta_anomaly_factor',   # Low risk anomaly
        'residual_momentum_factor',  # Idiosyncratic momentum
        'fear_greed_sentiment_factor' # Market sentiment
    ]
    
    # Backup factors if primary ones fail
    BACKUP_ALPHA_FACTORS = [
        'altman_z_score',
        'piotroski_f_score',
        'momentum_6_1_enhanced',
        'bid_ask_spread',
        'market_sentiment_factor'
    ]
    
    # =========================================================================
    # Alpha Summary Feature Names
    # =========================================================================
    ALPHA_SUMMARY_FEATURES = [
        'alpha_pc1',              # First principal component
        'alpha_pc2',              # Second principal component
        'alpha_pc3',              # Third principal component
        'alpha_pc4',              # Fourth principal component
        'alpha_pc5',              # Fifth principal component
        'alpha_cluster_0',        # Cluster 0 score
        'alpha_cluster_1',        # Cluster 1 score
        'alpha_cluster_2',        # Cluster 2 score
        'alpha_composite_orth1', # Composite feature 1
        'alpha_composite_orth2'  # Composite feature 2
    ]
    
    # =========================================================================
    # Validation Methods
    # =========================================================================
    
    def validate_config(self) -> bool:
        """Validate configuration consistency"""
        # Check feature counts
        if self.MAX_ALPHA_FEATURES > 15:
            raise ValueError(f"MAX_ALPHA_FEATURES {self.MAX_ALPHA_FEATURES} > 15, too many dimensions")
        
        if self.MIN_ALPHA_FEATURES < 5:
            raise ValueError(f"MIN_ALPHA_FEATURES {self.MIN_ALPHA_FEATURES} < 5, too few dimensions")
        
        # Check weight range
        if not (0 <= self.ALPHA_TARGET_WEIGHT_MIN <= self.ALPHA_TARGET_WEIGHT_MAX <= 1):
            raise ValueError(f"Invalid weight range: {self.ALPHA_TARGET_WEIGHT_MIN}-{self.ALPHA_TARGET_WEIGHT_MAX}")
        
        # Check PCA components
        total_features = (self.PCA_N_COMPONENTS + 
                         self.N_CLUSTERS + 
                         self.COMPOSITE_N_FEATURES)
        
        if total_features != self.MAX_ALPHA_FEATURES:
            print(f"Warning: Total features {total_features} != MAX_ALPHA_FEATURES {self.MAX_ALPHA_FEATURES}")
        
        return True
    
    def get_feature_weight_distribution(self, n_traditional_features: int) -> dict:
        """Calculate target feature weight distribution"""
        # Calculate target alpha features based on weight
        target_alpha_weight = (self.ALPHA_TARGET_WEIGHT_MIN + self.ALPHA_TARGET_WEIGHT_MAX) / 2
        target_traditional_weight = 1 - target_alpha_weight
        
        # If we have n_traditional features and want alpha to be 60% of total
        # then n_alpha / (n_alpha + n_traditional) = 0.6
        # so n_alpha = 0.6 * n_traditional / 0.4
        n_alpha_target = int(target_alpha_weight * n_traditional_features / target_traditional_weight)
        
        # But we limit to MAX_ALPHA_FEATURES
        n_alpha_actual = min(n_alpha_target, self.MAX_ALPHA_FEATURES)
        
        # Calculate actual weights
        total_features = n_alpha_actual + n_traditional_features
        alpha_weight_actual = n_alpha_actual / total_features
        traditional_weight_actual = n_traditional_features / total_features
        
        return {
            'n_alpha_target': n_alpha_target,
            'n_alpha_actual': n_alpha_actual,
            'n_traditional': n_traditional_features,
            'total_features': total_features,
            'alpha_weight_target': target_alpha_weight,
            'alpha_weight_actual': alpha_weight_actual,
            'traditional_weight_actual': traditional_weight_actual
        }


# Create singleton instance
ENHANCED_ALPHA_CONFIG = EnhancedAlphaConfig()

# Validate on load
ENHANCED_ALPHA_CONFIG.validate_config()


def get_alpha_config() -> EnhancedAlphaConfig:
    """Get the singleton alpha configuration"""
    return ENHANCED_ALPHA_CONFIG