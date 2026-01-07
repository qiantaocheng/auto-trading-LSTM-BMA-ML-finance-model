#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple 20 Factor Engine - No Dependencies
Direct implementation of all 20 optimized factors for BMA pipeline
No external dependencies, works out of the box
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import os
try:
    from bma_models.alpha_factor_quality_monitor import AlphaFactorQualityMonitor
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False

logger = logging.getLogger(__name__)

# 🔥 CORE ALPHA FACTORS (horizon-aware)
# - T+10 default: updated bi-weekly set (adds liquidity/OBV divergence/IVOL; removes fast/noisy/lagging factors)
# - T+5 legacy: preserved for backward compatibility when explicitly requested
#
# NOTE: Downstream training expects a stable set of feature names. We keep both sets and switch
#       by engine.horizon. (horizon=5 -> T5 set, horizon=10 -> T10 set)
T5_ALPHA_FACTORS = [
    'momentum_60d',          # 60-day price momentum (T+5 horizon)
    'rsi_21',                # 21-period RSI tuned for smoother signals
    'bollinger_squeeze',     # Bollinger Band volatility squeeze
    'obv_momentum_60d',      # 60-day OBV momentum
    'atr_ratio',             # ATR intensity ratio
    'blowoff_ratio',         # Recent blowoff jump normalized by 14d vol
    'hist_vol_40d',          # 40-day historical volatility level
    'vol_ratio_20d',         # 20-day volume spike ratio
    'near_52w_high',         # Distance to 52-week high (252-day window)
    'price_ma60_deviation',  # Deviation from 60-day moving average
    'mom_accel_20_5',        # Momentum acceleration (20d vs 5d)
    'streak_reversal',       # Streak reversal signal
    'ma30_ma60_cross',       # 30vs60-day moving-average cross trend signal
    'ret_skew_20d',          # 20-day return skewness
    'trend_r2_60',           # 60-day trend R-squared
]

# 🔥 T+10 CORE ALPHA FACTORS (bi-weekly optimized)
# REMOVED: streak_reversal, ma30_ma60_cross, mom_accel_20_5
# ADDED: liquid_momentum, obv_divergence, ivol_20
T10_ALPHA_FACTORS = [
    'liquid_momentum',
    'obv_divergence',
    'ivol_20',
    'rsi_21',
    'trend_r2_60',
    'near_52w_high',
    'ret_skew_20d',
    'blowoff_ratio',
    'hist_vol_40d',
    'atr_ratio',
    'bollinger_squeeze',
    'vol_ratio_20d',
    'price_ma60_deviation',
]

# Backward compatibility aliases - default to the T+10 set and fall back to T+5 only when explicitly requested.
DEFAULT_REQUIRED_FACTORS = T10_ALPHA_FACTORS
REQUIRED_14_FACTORS = DEFAULT_REQUIRED_FACTORS
REQUIRED_16_FACTORS = DEFAULT_REQUIRED_FACTORS
REQUIRED_17_FACTORS = DEFAULT_REQUIRED_FACTORS
REQUIRED_20_FACTORS = DEFAULT_REQUIRED_FACTORS
REQUIRED_22_FACTORS = DEFAULT_REQUIRED_FACTORS
REQUIRED_24_FACTORS = DEFAULT_REQUIRED_FACTORS

class Simple17FactorEngine:
    """
    Simple 17 Factor Engine - Complete High-Quality Factor Suite
    Directly computes all 17 high-quality factors: 15 alpha factors + sentiment_score + Close
    (Removed redundant and unstable factors: momentum_20d, momentum_reversal_short,
     price_to_ma20, cci, growth_proxy, profitability_momentum, growth_acceleration)

    支持训练/预测模式分离:
    - 训练模式 (mode='train'): 计算target并dropna
    - 预测模式 (mode='predict'): 不计算target，保留所有数据
    """

    def __init__(self,
                 lookback_days: int = 252,
                 enable_sentiment: Optional[bool] = None,
                 polygon_api_key: Optional[str] = None,
                 sentiment_max_workers: int = 4,
                 sentiment_batch_size: int = 32,
                 skip_cross_sectional_standardization: bool = False,
                 mode: str = 'train',
                 horizon: int = 10):
        self.lookback_days = lookback_days
        self.skip_cross_sectional_standardization = skip_cross_sectional_standardization

        # 🔥 NEW: 训练/预测模式支持
        self.mode = mode.lower() if mode else 'train'
        if self.mode == 'inference':
            self.mode = 'predict'  # 统一命名

        self.horizon = horizon
        # Horizon-aware feature set (default T+10)
        try:
            horizon_value = int(self.horizon)
        except Exception:
            horizon_value = 10
        self.alpha_factors = T10_ALPHA_FACTORS if horizon_value >= 10 else T5_ALPHA_FACTORS
        logger.info(
            f"Using {'T+10' if horizon_value >= 10 else 'T+5'} alpha factor set ({len(self.alpha_factors)} factors)"
        )

        if self.mode not in ['train', 'predict']:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'train' or 'predict'.")

        logger.info(f"✅ Simple17FactorEngine initialized in {self.mode.upper()} mode (horizon={self.horizon})")

        # Default API key (can be overridden by environment or parameter)
        DEFAULT_POLYGON_KEY = "FExbaO1xdmrV6f6p3zHCxk8IArjeowQ1"

        # Check for API key from various sources (in order of priority)
        env_key = os.environ.get('POLYGON_API_KEY')
        client_key = None
        try:
            from polygon_client import polygon_client as _global_polygon_client
            client_key = getattr(_global_polygon_client, 'api_key', None)
        except Exception:
            client_key = None

        # Use provided key, then env key, then client key, then default
        self.polygon_api_key = polygon_api_key or env_key or client_key or DEFAULT_POLYGON_KEY

        if self.polygon_api_key:
            logger.info(f"✓ Polygon API key configured (length: {len(self.polygon_api_key)} chars)")
        else:
            logger.warning("⚠️ No Polygon API key available - sentiment features will be disabled")
            logger.warning("  Set POLYGON_API_KEY environment variable to enable sentiment analysis")

        if enable_sentiment is None:
            # Default: DISABLE sentiment integration
            self.enable_sentiment = False
            logger.info("Sentiment features disabled by default (enable_sentiment=False)")
        else:
            self.enable_sentiment = bool(enable_sentiment)
        self.sentiment_max_workers = sentiment_max_workers
        self.sentiment_batch_size = sentiment_batch_size
        self._sentiment_analyzer = None

        if MONITOR_AVAILABLE:
            self.factor_monitor = AlphaFactorQualityMonitor(save_reports=True)
        else:
            self.factor_monitor = None

        # For IVOL calculation
        self.spy_data = None
        
    def fetch_market_data(self, symbols: List[str], use_optimized_downloader: bool = True, 
                         start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch market data from Polygon API with optimized downloader option
        
        Args:
            symbols: List of stock symbols
            use_optimized_downloader: Use optimized downloader for better performance
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
        """
        if use_optimized_downloader:
            try:
                # Use optimized downloader for better performance
                from optimized_25_factor_data_downloader import download_for_25_factors
                
                logger.info(f"🚀 Using optimized downloader for {len(symbols)} symbols")
                
                # Get optimized data directly in MultiIndex format
                optimized_data = download_for_25_factors(
                    symbols=symbols,
                    lookback_days=self.lookback_days,
                    enable_validation=True,
                    enable_progress_log=True,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not optimized_data.empty:
                    # Reset index to convert MultiIndex back to columns for compatibility
                    data_with_cols = optimized_data.reset_index()
                    logger.info(f"✅ Optimized data retrieved: {data_with_cols.shape}")
                    return data_with_cols
                else:
                    logger.warning("⚠️ Optimized downloader returned empty data, falling back to legacy method")
                    
            except ImportError:
                logger.warning("⚠️ Optimized downloader not available, using legacy method")
            except Exception as e:
                logger.error(f"⚠️ Optimized downloader failed: {e}, falling back to legacy method")
        
        # Legacy method (fallback)
        try:
            from polygon_client import polygon_client
            
            # Use provided dates or fall back to default
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
            
            logger.info(f"Fetching real data for {len(symbols)} symbols from {start_date} to {end_date}")
            
            all_data = []
            for symbol in symbols:
                try:
                    df = polygon_client.get_historical_bars(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        timespan='day',
                        multiplier=1
                    )
                    
                    if not df.empty:
                        # Don't reset index here - preserve DatetimeIndex for concatenation
                        df['ticker'] = symbol
                        all_data.append(df)
                        logger.info(f"Retrieved {len(df)} rows for {symbol}")
                
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {e}")
            
            if all_data:
                # Don't ignore index to preserve DatetimeIndex, then reset to create 'Date' column
                combined = pd.concat(all_data, ignore_index=False)
                combined = combined.reset_index()  # This creates 'Date' column from DatetimeIndex
                logger.info(f"After reset_index - columns: {list(combined.columns)}")
                if 'Date' in combined.columns:
                    combined = combined.rename(columns={'Date': 'date'})
                    logger.info("✅ Renamed 'Date' to 'date'")
                else:
                    logger.error(f"❌ 'Date' column not found after reset_index. Columns: {list(combined.columns)}")
                combined = combined.sort_values(['date', 'ticker'])
                logger.info(f"Real data retrieved: {combined.shape}")
                return combined
        
        except Exception as e:
            logger.error(f"Error fetching real data: {e}")
            
        return pd.DataFrame()
    
    def compute_all_17_factors(self, market_data: pd.DataFrame, mode: str = None) -> pd.DataFrame:
        """
        🔥 Compute all core horizon-aware alpha factors + Close for target calculation

        Args:
            market_data: Market data DataFrame
            mode: 'train' or 'predict' (overrides self.mode if provided)

        Returns:
            DataFrame with factors (and target if mode='train')
        """
        import time

        # 使用传入的mode或默认使用实例mode
        actual_mode = mode.lower() if mode else self.mode
        if actual_mode == 'inference':
            actual_mode = 'predict'

        if market_data.empty:
            logger.error("No market data provided")
            return pd.DataFrame()

        logger.info("=" * 80)
        horizon_label = getattr(self, 'horizon', '?')
        logger.info(f"🔥 COMPUTING ALL CORE T+{horizon_label} ALPHA FACTORS + CLOSE")
        logger.info("=" * 80)
        logger.info(f"📊 Market data input: shape={market_data.shape}")
        if 'Close' in market_data.columns:
            logger.info(f"📈 Price range: ${market_data['Close'].min():.2f} - ${market_data['Close'].max():.2f}")
        logger.info(f"📅 Data points: {len(market_data)} rows")
        
        # Track timing for each factor group
        factor_timings = {}
        
        # Prepare date column if not present
        market_data_clean = market_data.copy()
        
        # Debug: Check what data we received
        logger.info(f"compute_all_25_factors input - shape: {market_data.shape}, columns: {list(market_data.columns)}")
        logger.info(f"compute_all_25_factors input - index type: {type(market_data.index)}")
        
        # Handle date column - check if it exists, if not create from index
        if 'date' not in market_data_clean.columns:
            if isinstance(market_data_clean.index, pd.MultiIndex):
                # Handle MultiIndex (date, ticker) format
                if 'date' in market_data_clean.index.names:
                    market_data_clean = market_data_clean.reset_index()
                    logger.info("✅ Extracted 'date' and 'ticker' columns from MultiIndex")
                else:
                    logger.error("❌ MultiIndex does not contain 'date' level")
                    logger.error(f"Index names: {market_data_clean.index.names}")
                    return pd.DataFrame()
            elif isinstance(market_data_clean.index, pd.DatetimeIndex):
                market_data_clean['date'] = market_data_clean.index
                logger.info("✅ Created 'date' column from DatetimeIndex")
            elif 'timestamp' in market_data_clean.columns:
                market_data_clean['date'] = pd.to_datetime(market_data_clean['timestamp'])
                logger.info("✅ Created 'date' column from 'timestamp' column")
            else:
                logger.error("❌ Cannot create date column - no valid date information found")
                logger.error(f"Available columns: {list(market_data_clean.columns)}")
                logger.error(f"Index type: {type(market_data_clean.index)}")
                return pd.DataFrame()
        else:
            # Ensure existing date column is properly formatted
            market_data_clean['date'] = pd.to_datetime(market_data_clean['date'])
            logger.info("✅ Using existing 'date' column")

        # Ensure ticker column exists
        if 'ticker' not in market_data_clean.columns:
            logger.error("❌ Missing 'ticker' column in market data")
            logger.info(f"Available columns: {list(market_data_clean.columns)}")
            return pd.DataFrame()
        
        # Group data by ticker for efficient computation
        compute_data = market_data_clean
        grouped = compute_data.groupby('ticker')
        
        # Collect all factor results, ensuring consistent indexing
        all_factors = []
        
        # 1: Momentum Factors (REDUCED: only momentum_10d)
        logger.info("="*60)
        logger.info("🎯 [ALPHA FACTOR 1] MOMENTUM FACTORS")
        logger.info("="*60)
        start_t = time.time()
        momentum_results = self._compute_momentum_factors(compute_data, grouped)
        factor_timings['momentum'] = time.time() - start_t

        # Monitor each momentum factor if monitor available
        if self.factor_monitor:
            for factor_name in ['momentum_60d']:  # 🔥 Updated for T+5
                if factor_name in momentum_results.columns:
                    self.factor_monitor.monitor_factor_computation(
                        factor_name, momentum_results[factor_name],
                        computation_time=factor_timings['momentum']
                    )
        
        logger.info(f"⏱️ Momentum factors computed in {factor_timings['momentum']:.3f}s")
        logger.info("="*60)
        all_factors.append(momentum_results)
        
        # 2-4: Mean Reversion Factors (REDUCED: removed price_to_ma20, cci)
        logger.info("Computing mean reversion factors (2/14)...")
        start_t = time.time()
        meanrev_results = self._compute_mean_reversion_factors(compute_data, grouped)
        factor_timings['mean_reversion'] = time.time() - start_t
        logger.info(f"   Mean reversion factors computed in {factor_timings['mean_reversion']:.3f}s")
        all_factors.append(meanrev_results)

        # 5-6: Volume Factors
        logger.info("Computing volume factors (1/14)...")
        start_t = time.time()
        volume_results = self._compute_volume_factors(compute_data, grouped)
        factor_timings['volume'] = time.time() - start_t
        logger.info(f"   Volume factors computed in {factor_timings['volume']:.3f}s")
        all_factors.append(volume_results)

        # T+10: RSRS beta (rolling regression slope of Low -> High)
        if 'rsrs_beta_18' in getattr(self, 'alpha_factors', []):
            try:
                logger.info("🔥 Computing rsrs_beta_18 (RSRS rolling beta: High~Low, window=18)...")
                start_t = time.time()
                rsrs_df = self._compute_rsrs_beta(data=compute_data, grouped=grouped, window=18)
                factor_timings['rsrs_beta_18'] = time.time() - start_t
                logger.info(f"   rsrs_beta_18 computed in {factor_timings['rsrs_beta_18']:.3f}s")
                all_factors.append(rsrs_df)
            except Exception as e:
                logger.warning(f"rsrs_beta_18 computation failed (continue without): {e}")
        
        # 7: Volatility Factors (1 factor: atr_ratio)
        logger.info("Computing volatility factors (1/14)...")
        start_t = time.time()
        vol_results = self._compute_volatility_factors(compute_data, grouped)
        factor_timings['volatility'] = time.time() - start_t
        logger.info(f"   Volatility factors computed in {factor_timings['volatility']:.3f}s")
        all_factors.append(vol_results)

        # REMOVED: IVOL Factor (multicollinearity with hist_vol_40d, r=-0.95, VIF=10.4)

        # 10-13: Fundamental Proxy Factors (REDUCED: removed growth_proxy, profitability_momentum, growth_acceleration)
        logger.info("Computing fundamental proxy factors (1/14)...")
        start_t = time.time()
        fundamental_results = self._compute_fundamental_factors(compute_data, grouped)
        factor_timings['fundamental'] = time.time() - start_t
        logger.info(f"   Fundamental factors computed in {factor_timings['fundamental']:.3f}s")
        all_factors.append(fundamental_results)

        # High-Alpha Factors
        logger.info("🔥 Computing 3 high-alpha factors...")
        start_t = time.time()
        new_alpha_results = self._compute_new_alpha_factors(compute_data, grouped)
        factor_timings['new_alpha'] = time.time() - start_t
        logger.info(f"   High-alpha factors computed in {factor_timings['new_alpha']:.3f}s")
        all_factors.append(new_alpha_results)

        # 18: Behavioral Factors (only if requested by active factor set)
        if 'streak_reversal' in getattr(self, 'alpha_factors', []):
            logger.info("🔥 Computing behavioral factor: streak_reversal ...")
            start_t = time.time()
            behavioral_results = self._compute_behavioral_factors(compute_data, grouped)
            factor_timings['behavioral'] = time.time() - start_t
            logger.info(f"   Behavioral factors computed in {factor_timings['behavioral']:.3f}s")
            all_factors.append(behavioral_results)

        # 🔥 NEW LOW-FREQUENCY FACTORS FOR T+5
        logger.info("🔥 Computing low-frequency factors for T+5...")

        # Return skewness 20d
        logger.info("   Computing ret_skew_20d (1/3)...")
        start_t = time.time()
        skew_results = self._compute_ret_skew_20d(compute_data, grouped)
        factor_timings['ret_skew_20d'] = time.time() - start_t
        logger.info(f"   Ret_skew_20d computed in {factor_timings['ret_skew_20d']:.3f}s")
        all_factors.append(skew_results)

        # Trend R² 60d
        logger.info("   Computing trend_r2_60 (2/3)...")
        start_t = time.time()
        trend_r2_results = self._compute_trend_r2_60(compute_data, grouped)
        factor_timings['trend_r2_60'] = time.time() - start_t
        logger.info(f"   Trend_r2_60 computed in {factor_timings['trend_r2_60']:.3f}s")
        all_factors.append(trend_r2_results)

        # MA30/MA60 cross (only if requested)
        if 'ma30_ma60_cross' in getattr(self, 'alpha_factors', []):
            logger.info("   Computing ma30_ma60_cross (3/3)...")
            start_t = time.time()
            ma_cross_results = self._compute_ma_cross_30_60(compute_data, grouped)
            factor_timings['ma_cross'] = time.time() - start_t
            logger.info(f"   MA30/MA60 cross computed in {factor_timings['ma_cross']:.3f}s")
            all_factors.append(ma_cross_results)

        # T+10: IVOL 20 (idiosyncratic vol) if requested
        if 'ivol_20' in getattr(self, 'alpha_factors', []):
            try:
                logger.info("🔥 Computing ivol_20 (idiosyncratic volatility vs SPY)...")
                start_t = time.time()
                ivol_df = self._compute_ivol_20(compute_data, grouped)
                factor_timings['ivol_20'] = time.time() - start_t
                logger.info(f"   ivol_20 computed in {factor_timings['ivol_20']:.3f}s")
                all_factors.append(ivol_df)
            except Exception as e:
                logger.warning(f"ivol_20 computation failed (continue without): {e}")

        # Blowoff jump ratio (max jump over last 5d normalized by 14d vol) and medium-term volatility level
        logger.info("   Computing blowoff_ratio and hist_vol_40d...")
        start_t = time.time()
        blowoff_vol_results = self._compute_blowoff_and_volatility(compute_data, grouped)
        factor_timings['blowoff_volatility'] = time.time() - start_t
        logger.info(f"   Blowoff/Volatility computed in {factor_timings['blowoff_volatility']:.3f}s")
        all_factors.append(blowoff_vol_results)

        # ==============================
        # Falling-knife risk features (engineering-grade)
        # ==============================
        try:
            logger.info("🔥 Computing falling-knife risk features: making_new_low_5d...")

            # Per-ticker rolling constructs with strict non-leakage (shifted windows)
            ret_1d = grouped['Close'].transform(lambda s: s.pct_change())
            avg_vol_20d = grouped['Volume'].transform(
                lambda v: v.rolling(20, min_periods=10).mean().shift(1)
            )
            vol_ratio_20d = (compute_data['Volume'] / avg_vol_20d).replace([np.inf, -np.inf], np.nan)

            min_close_prev5 = grouped['Close'].transform(
                lambda s: s.rolling(5, min_periods=5).min().shift(1)
            )
            making_new_low_5d = (compute_data['Close'] <= min_close_prev5).astype(float)

            knife_df = pd.DataFrame({
                'making_new_low_5d': making_new_low_5d,
            }, index=compute_data.index)

            all_factors.append(knife_df)
            logger.info("   ✅ Falling-knife risk features computed")
        except Exception as e:
            logger.warning(f"Falling-knife feature computation failed (continue without): {e}")

        # Combine all factor DataFrames
        factors_df = pd.concat(all_factors, axis=1)
        
        # Add Close prices BEFORE setting MultiIndex to preserve alignment
        factors_df['Close'] = compute_data['Close']
        
        # Set MultiIndex using the prepared date and ticker columns
        factors_df.index = pd.MultiIndex.from_arrays(
            [compute_data['date'], compute_data['ticker']], 
            names=['date', 'ticker']
        )

        # Optionally integrate sentiment features (FinBERT via Polygon)
        try:
            if self.enable_sentiment and self.polygon_api_key:
                logger.info("Integrating FinBERT sentiment features into Simple25 engine output...")
                start_t = time.time()
                sentiment_df = self._compute_sentiment_for_market_data(compute_data)
                elapsed = time.time() - start_t
                if sentiment_df is not None and not sentiment_df.empty:
                    logger.info(f"   Sentiment features computed: {sentiment_df.shape} in {elapsed:.3f}s")

                    # Debug: Check indices before join
                    logger.debug(f"   factors_df index: {factors_df.index.names}, shape: {factors_df.shape}")
                    logger.debug(f"   sentiment_df index: {sentiment_df.index.names}, shape: {sentiment_df.shape}")

                    # Check for any non-zero sentiment values before join
                    if 'sentiment_score' in sentiment_df.columns:
                        non_zero_count = (sentiment_df['sentiment_score'] != 0).sum()
                        logger.info(f"   Non-zero sentiment scores before join: {non_zero_count}/{len(sentiment_df)}")

                    # Join on MultiIndex (date, ticker)
                    factors_df = factors_df.join(sentiment_df, how='left')

                    # Check sentiment values after join
                    if 'sentiment_score' in factors_df.columns:
                        non_zero_after = (factors_df['sentiment_score'] != 0).sum()
                        logger.info(f"   Non-zero sentiment scores after join: {non_zero_after}/{len(factors_df)}")

                    # 智能填充sentiment缺失值
                    if self._sentiment_analyzer is not None:
                        for col in getattr(self._sentiment_analyzer, 'sentiment_features', []):
                            if col in factors_df.columns:
                                before_fill = (factors_df[col].notna()).sum()
                                total_values = len(factors_df[col])

                                # ✅ PIT-safe fill (NO future leakage):
                                # - Never use full-sample mean (leaks future dates)
                                # - Fill missing sentiment with 0.0 (neutral) which is safe and stable
                                coverage_rate = before_fill / total_values if total_values else 0.0
                                fill_value = 0.0
                                logger.info(f"   Using zero fill for {col} (coverage: {coverage_rate:.1%})")
                                factors_df[col] = factors_df[col].fillna(0.0)
                                filled_count = total_values - before_fill
                                logger.debug(f"   Filled {filled_count} NaN values in {col}")
                else:
                    logger.warning("   Sentiment features empty or unavailable; skipping integration")
            elif self.enable_sentiment and not self.polygon_api_key:
                logger.warning("⚠️ Sentiment enabled but no API key available - skipping sentiment features")
                logger.info("   Set POLYGON_API_KEY environment variable to enable sentiment analysis")
            elif not self.enable_sentiment:
                logger.debug("Sentiment features disabled by configuration")
        except Exception as e:
            logger.warning(f"Sentiment integration failed: {e}")
        
        # Clean data for factors only (preserve Close prices)
        # Replace infinities with NaN for robust cross-sectional processing later
        for col in factors_df.columns:
            if col != 'Close':
                factors_df[col] = factors_df[col].replace([np.inf, -np.inf], np.nan)
        
        # Verify all required factors are present (horizon-aware)
        base = list(getattr(self, "alpha_factors", REQUIRED_14_FACTORS))
        missing = set(base) - set(factors_df.columns)
        if missing:
            logger.error(f"Missing factors: {missing}")
            for factor in missing:
                factors_df[factor] = 0.0

        # Reorder columns: base factors first, then any extras (e.g., sentiment_*), finally Close
        extras = [c for c in factors_df.columns if c not in base + ['Close']]
        column_order = base + extras + ['Close']
        factors_df = factors_df[column_order]

        # ==============================
        # 🔥 NEW: 根据模式计算target
        # ==============================
        # 无论哪种模式，都先计算target（有未来数据的会有值，没有的会是NaN）
        logger.info("=" * 80)
        logger.info(f"📊 计算T+{self.horizon} target")
        logger.info("=" * 80)

        # 计算forward returns as target
        if 'Close' not in factors_df.columns:
            logger.error("❌ 缺少Close列，无法计算target")
        else:
            # 使用shift(-horizon)计算未来收益
            target_series = (
                factors_df.groupby(level='ticker')['Close']
                .pct_change(self.horizon)
                .shift(-self.horizon)
            )
            factors_df['target'] = target_series

            # 统计target质量
            total_samples = len(factors_df)
            valid_targets = target_series.notna().sum()
            valid_ratio = valid_targets / total_samples if total_samples > 0 else 0

            logger.info(f"✅ Target计算完成:")
            logger.info(f"   总样本: {total_samples}")
            logger.info(f"   有效target: {valid_targets} ({valid_ratio:.1%})")
            logger.info(f"   缺失target: {total_samples - valid_targets} (最近{self.horizon}天无未来数据)")

            # 显示日期范围
            if 'date' in factors_df.index.names:
                all_dates = factors_df.index.get_level_values('date')
                min_date = all_dates.min()
                max_date = all_dates.max()
                logger.info(f"📅 数据日期范围: {min_date} 到 {max_date}")

                # 找出有target和无target的日期范围
                if valid_targets > 0:
                    has_target_dates = factors_df[factors_df['target'].notna()].index.get_level_values('date')
                    max_target_date = has_target_dates.max()
                    logger.info(f"   有target日期: {min_date} 到 {max_target_date}")

                if valid_targets < total_samples:
                    no_target_dates = factors_df[factors_df['target'].isna()].index.get_level_values('date')
                    if len(no_target_dates) > 0:
                        min_no_target = no_target_dates.min()
                        max_no_target = no_target_dates.max()
                        logger.info(f"   无target日期: {min_no_target} 到 {max_no_target} (最新数据)")
                        predict_target_date = pd.to_datetime(max_no_target) + pd.Timedelta(days=self.horizon)
                        logger.info(f"🎯 预测目标日期: {predict_target_date.date()} (T+{self.horizon})")

        # 🔥 根据模式决定是否dropna
        if actual_mode == 'train':
            logger.info("=" * 80)
            logger.info("📚 训练模式: 执行dropna，删除无target样本")
            logger.info("=" * 80)

            samples_before = len(factors_df)
            factors_df = factors_df.dropna(subset=['target'])
            samples_after = len(factors_df)
            samples_removed = samples_before - samples_after

            logger.info(f"✅ Dropna执行完成:")
            logger.info(f"   删除前: {samples_before} 样本")
            logger.info(f"   删除后: {samples_after} 样本")
            removal_pct = (samples_removed/samples_before*100) if samples_before > 0 else 0
            logger.info(f"   移除: {samples_removed} 样本 ({removal_pct:.1f}%)")

            # 显示最后训练日期
            if 'date' in factors_df.index.names:
                max_train_date = factors_df.index.get_level_values('date').max()
                logger.info(f"📅 训练数据截至日期: {max_train_date}")
        else:
            logger.info("=" * 80)
            logger.info("🔮 预测模式: 不执行dropna，保留所有样本")
            logger.info("=" * 80)
            logger.info(f"   ✅ 保留所有 {len(factors_df)} 个样本")
            logger.info(f"   - 有target样本: {factors_df['target'].notna().sum()} (用于训练)")
            logger.info(f"   - 无target样本: {factors_df['target'].isna().sum()} (用于预测)")
            logger.info(f"   💡 系统会自动分离：训练用有target数据，预测用无target数据")

        # ==============================
        # Cross-sectional robust winsorization and standardization (per date)
        # ==============================
        # 🔥 FIX: Skip cross-sectional standardization for single-stock prediction
        if self.skip_cross_sectional_standardization:
            logger.info("⏭️ Skipping cross-sectional standardization (prediction mode)")
        else:
            try:
                if isinstance(factors_df.index, pd.MultiIndex) and 'date' in factors_df.index.names:
                    # Check minimum stocks per date for cross-sectional standardization
                    stocks_per_date = factors_df.groupby(level='date').size()
                    min_stocks = stocks_per_date.min()

                    if min_stocks < 3:
                        logger.warning(f"⚠️ Insufficient stocks for cross-sectional standardization (min={min_stocks} < 3)")
                        logger.warning("   Skipping cross-sectional standardization - using raw factor values")
                    else:
                        # ✅ CRITICAL PIT FIX:
                        # - Never winsorize/fill/standardize the *label* (`target`)
                        #   (it is a future return; transforming it breaks backtests and can create
                        #    misleadingly high “returns” when users treat target as real PnL)
                        # - Standardize only feature columns, excluding Close and target.
                        cs_cols = [c for c in factors_df.columns if c not in ('Close', 'target')]

                        # Winsorize by IQR per date
                        def _winsorize_group(g: pd.DataFrame) -> pd.DataFrame:
                            n_stocks = len(g)
                            for c in cs_cols:
                                s = g[c].astype(float)
                                valid = s.dropna()
                                # Lower threshold from 5 to 3 for small universes
                                if len(valid) >= max(3, min(5, n_stocks)):
                                    q1 = valid.quantile(0.25)
                                    q3 = valid.quantile(0.75)
                                    iqr = q3 - q1
                                    lo = q1 - 3.0 * iqr
                                    hi = q3 + 3.0 * iqr
                                    g[c] = s.clip(lower=lo, upper=hi)
                                else:
                                    g[c] = s
                            return g

                        factors_df = factors_df.groupby(level='date', group_keys=False).apply(_winsorize_group)

                        # Fill remaining NaNs with cross-sectional median
                        def _median_fill_group(g: pd.DataFrame) -> pd.DataFrame:
                            for c in cs_cols:
                                if g[c].isna().any():
                                    med = g[c].median()
                                    g[c] = g[c].fillna(0.0 if not np.isfinite(med) else med)
                            return g

                        factors_df = factors_df.groupby(level='date', group_keys=False).apply(_median_fill_group)

                        # Standardize per date (z-score)
                        def _standardize_group(g: pd.DataFrame) -> pd.DataFrame:
                            n_stocks = len(g)
                            for c in cs_cols:
                                s = g[c].astype(float)
                                mean = s.mean()
                                std = s.std(ddof=0)
                                # 🔥 FIX: Only standardize if we have enough stocks
                                if n_stocks >= 3 and np.isfinite(std) and std > 1e-10:
                                    g[c] = (s - mean) / (std + 1e-10)
                                else:
                                    # Keep raw values for single/few stocks
                                    g[c] = s
                            return g

                        factors_df = factors_df.groupby(level='date', group_keys=False).apply(_standardize_group)
                        logger.info("✅ Applied cross-sectional IQR winsorization and z-score standardization per date")
            except Exception as e:
                logger.warning(f"Cross-sectional standardization failed: {e}")

        logger.info("=" * 60)
        logger.info(f"ALL 14 HIGH-QUALITY FACTORS + CLOSE COMPUTED: {factors_df.shape}")
        logger.info("Factor Computation Timing:")
        total_time = sum(factor_timings.values())
        for name, duration in factor_timings.items():
            pct = 100 * duration / total_time if total_time > 0 else 0
            logger.info(f"   {name:<15}: {duration:.3f}s ({pct:.1f}%)")
        logger.info(f"   {'TOTAL':<15}: {total_time:.3f}s")
        logger.info("=" * 60)

        # 🔥 NEW: Apply smart forward-fill to long-lookback factors
        # This prevents massive sample loss from dropna() in downstream processing
        factors_df = self._apply_smart_forward_fill(factors_df)

        return factors_df

    def _compute_rsrs_beta(self, data: pd.DataFrame, grouped, window: int = 18) -> pd.DataFrame:
        """
        RSRS beta: rolling regression slope of High ~ Low over `window` days (per ticker).

        We avoid sklearn in rolling loops by using:
          beta = Cov(Low, High) / Var(Low)
        """
        try:
            if 'Low' not in data.columns or 'High' not in data.columns:
                raise ValueError("rsrs_beta requires 'Low' and 'High' columns")

            eps = 1e-10
            x = data['Low'].astype(float)
            y = data['High'].astype(float)

            ex = grouped['Low'].transform(lambda s: s.astype(float).rolling(window, min_periods=window).mean())
            ey = grouped['High'].transform(lambda s: s.astype(float).rolling(window, min_periods=window).mean())

            exy = (x * y).groupby(data['ticker']).transform(lambda s: s.rolling(window, min_periods=window).mean())
            ex2 = (x * x).groupby(data['ticker']).transform(lambda s: s.rolling(window, min_periods=window).mean())

            cov_xy = (exy - ex * ey)
            var_x = (ex2 - ex * ex)

            beta = (cov_xy / (var_x + eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return pd.DataFrame({'rsrs_beta_18': beta}, index=data.index)
        except Exception as e:
            logger.warning(f"⚠️ rsrs_beta_18 failed, using zeros: {e}")
            return pd.DataFrame({'rsrs_beta_18': np.zeros(len(data))}, index=data.index)

    def _apply_smart_forward_fill(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        智能前向填充：仅对长周期因子的warm-up期进行填充
        防止dropna()造成大量样本损失

        策略：
        1. 识别长周期因子（lookback > 60天）
        2. 对每个ticker，找到第一个有效值
        3. 用第一个有效值填充之前的NaN（warm-up期）
        4. 中间的NaN用前向填充（限制30天）

        Args:
            factors_df: 因子DataFrame，必须有(date, ticker)的MultiIndex

        Returns:
            填充后的DataFrame
        """
        # 🔥 定义需要填充的长周期因子 (T+5 optimized: removed max_lottery_factor, overnight_intraday_gap)
        LONG_LOOKBACK_FACTORS = {
            'momentum_60d': 60,
            'obv_momentum_60d': 60,
            'price_ma60_deviation': 60,
            'ma30_ma60_cross': 60,
            'near_52w_high': 252,
        }

        if not isinstance(factors_df.index, pd.MultiIndex):
            logger.warning("factors_df must have MultiIndex for smart forward-fill, skipping")
            return factors_df

        nan_before = factors_df.isna().sum().sum()
        filled_df = factors_df.copy()

        # 简化版：使用groupby forward-fill，避免复杂的索引操作
        for factor_name in LONG_LOOKBACK_FACTORS.keys():
            if factor_name in filled_df.columns:
                # Group by ticker and apply forward-fill with limit
                filled_df[factor_name] = filled_df.groupby(level='ticker')[factor_name].ffill(limit=30)

        nan_after = filled_df.isna().sum().sum()
        nan_filled = nan_before - nan_after

        if nan_filled > 0:
            logger.info(f"✅ Smart forward-fill applied:")
            logger.info(f"   NaN before: {nan_before:,}")
            logger.info(f"   NaN after:  {nan_after:,}")
            logger.info(f"   NaN filled: {nan_filled:,} ({nan_filled/nan_before*100:.1f}%)")
            logger.info(f"   Affected factors: {list(LONG_LOOKBACK_FACTORS.keys())}")

        return filled_df

    def _get_sentiment_analyzer(self):
        """Lazily create Ultra Fast sentiment analyzer (统一使用优化版本)."""
        if self._sentiment_analyzer is None:
            try:
                logger.info(f"Creating UltraFastSentimentFactor with API key: {self.polygon_api_key[:8] if self.polygon_api_key else 'None'}...")
                from bma_models.ultra_fast_sentiment_factor import UltraFastSentimentFactor
                self._sentiment_analyzer = UltraFastSentimentFactor(
                    polygon_api_key=self.polygon_api_key
                )
                logger.info("使用UltraFastSentimentFactor: 45词/新闻, 3条新闻/天, 真实API数据")
            except Exception as e:
                logger.warning(f"Failed to initialize UltraFastSentimentFactor: {e}")
                self._sentiment_analyzer = None
        return self._sentiment_analyzer

    def _compute_sentiment_for_market_data(self, data: pd.DataFrame, limit_dates: Optional[List[pd.Timestamp]] = None) -> pd.DataFrame:
        """Compute sentiment features aligned with the market_data date/ticker grid."""
        try:
            analyzer = self._get_sentiment_analyzer()
            if analyzer is None:
                return pd.DataFrame()

            # Extract tickers and trading dates from data
            tickers = sorted(pd.Series(data['ticker']).dropna().unique().tolist())
            trading_dates = sorted(pd.to_datetime(data['date']).dt.normalize().unique())
            if limit_dates is not None:
                limit_set = set(pd.to_datetime(limit_dates))
                trading_dates = [d for d in trading_dates if d in limit_set]
            trading_dates_dt = [pd.Timestamp(d).to_pydatetime() for d in trading_dates]
            if not tickers or not trading_dates_dt:
                return pd.DataFrame()

            start_date = trading_dates_dt[0]
            end_date = trading_dates_dt[-1]

            logger.info(f"   Computing sentiment for {len(tickers)} tickers across {len(trading_dates_dt)} trading days")
            logger.info(f"   Date range: {start_date} to {end_date}")
            logger.info(f"   Tickers: {tickers}")

            sent_df = analyzer.process_universe_sentiment(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                trading_dates=trading_dates_dt
            )

            # 注意：情感数据质量监控已在UltraFastSentimentFactor内部完成
            # 这里不需要重复监控，避免双重报告
            logger.info(f"   Sentiment computation completed: {sent_df.shape if sent_df is not None else 'No data'}")

            # Debug: Check if sentiment_df is empty
            if sent_df is None or sent_df.empty:
                logger.warning("   Sentiment DataFrame is empty or None")
            else:
                logger.info(f"   Sentiment DataFrame shape: {sent_df.shape}")
                logger.info(f"   Sentiment DataFrame columns: {list(sent_df.columns)}")
                if 'sentiment_score' in sent_df.columns:
                    non_zero = (sent_df['sentiment_score'] != 0).sum()
                    logger.info(f"   Non-zero sentiment scores: {non_zero}/{len(sent_df)}")

            return sent_df
        except Exception as e:
            logger.warning(f"Sentiment computation error: {e}")
            return pd.DataFrame()

    def _compute_ret_skew_20d(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """🔥 Compute 20-day return skewness - T+5 low-frequency factor
        Uses log returns for scale-invariance and sample skewness
        """
        # Initialize result array
        skew_values = []

        # Process each ticker
        for ticker, ticker_data in grouped:
            # Calculate log returns
            log_ret = np.log(ticker_data['Close'] / ticker_data['Close'].shift(1))

            # Calculate rolling skewness
            skew = log_ret.rolling(20, min_periods=20).apply(
                lambda x: pd.Series(x).skew() if len(x) >= 20 else 0.0,
                raw=False
            )

            # Clean inf/nan values
            skew = skew.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            # Extend the values array
            skew_values.extend(skew.values)

        return pd.DataFrame({'ret_skew_20d': skew_values}, index=data.index)

    def _compute_trend_r2_60(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """🔥 Compute 60-day trend R² (linear regression goodness of fit) - T+5 low-frequency factor
        Uses log prices for scale-invariance and proper index alignment
        """
        window = 60
        x_base = np.arange(window)

        # Initialize result array
        r2_values = []

        # Process each ticker
        for ticker, ticker_data in grouped:
            # Use log prices for scale-invariance
            logp = np.log(ticker_data['Close'].values)
            r2 = np.zeros(len(ticker_data), dtype=float)

            # Rolling window linear regression
            for i in range(window - 1, len(ticker_data)):
                y = logp[i - window + 1:i + 1]

                # Check for sufficient valid data
                if np.isfinite(y).sum() < window:
                    r2[i] = 0.0
                    continue

                # Linear regression: y = β0 + β1*x
                X = np.column_stack([np.ones(window), x_base])
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    y_hat = X @ beta
                    ss_res = np.sum((y - y_hat) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
                    r2_val = 1.0 - ss_res / ss_tot
                    # Clamp to [0, 1]
                    r2[i] = max(0.0, min(1.0, r2_val))
                except Exception:
                    r2[i] = 0.0

            # Extend the values array
            r2_values.extend(r2)

        return pd.DataFrame({'trend_r2_60': r2_values}, index=data.index)

    def _compute_ma_cross_30_60(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """🔥 Compute 30/60-day moving-average cross signal (1 if MA30 above MA60, else -1)."""

        ma30 = grouped['Close'].transform(lambda x: x.rolling(30, min_periods=15).mean())
        ma60 = grouped['Close'].transform(lambda x: x.rolling(60, min_periods=30).mean())
        diff = (ma30 - ma60).fillna(0.0)
        cross_signal = pd.Series(np.sign(diff.values), index=diff.index).fillna(0.0).astype(float)
        cross_signal = cross_signal.reindex(data.index).fillna(0.0)
        return pd.DataFrame({'ma30_ma60_cross': cross_signal}, index=data.index)

    def _compute_blowoff_and_volatility(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute blowoff_ratio and hist_vol_40d following Polygon OHLC pipeline.

        Definitions:
        - blowoff_ratio = max_{t-4..t} log_return / (std_14(log_return) + eps)
        - hist_vol_40d = rolling 40-day standard deviation of log returns
        """
        try:
            eps = 1e-8
            # Log returns per ticker for stability and scale-invariance
            log_ret = grouped['Close'].transform(lambda s: np.log(s / s.shift(1)))

            # σ14: rolling std of log returns (min_periods tuned for robustness)
            sigma14 = log_ret.groupby(data['ticker']).transform(lambda s: s.rolling(14, min_periods=10).std())
            # max jump over past 5 days (inclusive)
            max_jump_5d = log_ret.groupby(data['ticker']).transform(lambda s: s.rolling(5, min_periods=2).max())
            blowoff_ratio = (max_jump_5d / (sigma14 + eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # σ40: rolling std of log returns for medium-term volatility regime
            sigma40 = log_ret.groupby(data['ticker']).transform(lambda s: s.rolling(40, min_periods=15).std())
            hist_vol_40d = sigma40.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            return pd.DataFrame({
                'blowoff_ratio': blowoff_ratio,
                'hist_vol_40d': hist_vol_40d,
            }, index=data.index)
        except Exception as e:
            logger.warning(f"Blowoff/Volatility computation failed: {e}")
            return pd.DataFrame({
                'blowoff_ratio': np.zeros(len(data)),
                'hist_vol_40d': np.zeros(len(data)),
            }, index=data.index)

    def _compute_momentum_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """
        Momentum factors.
        - Always compute momentum_60d (backward compatibility)
        - If horizon==10 (or liquid_momentum requested), also compute liquid_momentum
          = momentum_60d * (Volume / AvgVolume_126d)
        """

        logger.info("?? [FACTOR COMPUTATION] Starting momentum factors calculation")
        factor_quality = {}


        logger.info("   ?? Computing momentum_60d (60-day price momentum)...")
        momentum_60d = grouped['Close'].pct_change(60).fillna(0)
        factor_quality['momentum_60d'] = {
            'non_zero': (momentum_60d != 0).sum(),
            'nan_count': momentum_60d.isna().sum(),
            'mean': momentum_60d.mean(),
            'std': momentum_60d.std(),
            'coverage': (momentum_60d != 0).sum() / len(momentum_60d) * 100
        }
        logger.info(f"   ? momentum_60d: coverage={factor_quality['momentum_60d']['coverage']:.1f}%, mean={factor_quality['momentum_60d']['mean']:.4f}")

        # Data quality warning
        for factor_name, quality in factor_quality.items():
            if quality['coverage'] < 50:
                logger.warning(f"   ?? {factor_name}: Low coverage {quality['coverage']:.1f}%")
            if quality['std'] == 0:
                logger.warning(f"   ?? {factor_name}: Zero variance detected")

        logger.info("   ? Momentum factors computation completed")

        out = {'momentum_60d': momentum_60d}

        # T+10: Liquid Momentum (momentum * turnover validation)
        if 'liquid_momentum' in getattr(self, 'alpha_factors', []):
            try:
                logger.info("   💧 Computing liquid_momentum (momentum * turnover validation)...")
                avg_vol_126 = grouped['Volume'].transform(lambda x: x.rolling(126, min_periods=30).mean().shift(1))
                turnover_ratio = (data['Volume'] / (avg_vol_126 + 1e-10)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                liquid_momentum = (momentum_60d * turnover_ratio).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                out['liquid_momentum'] = liquid_momentum
            except Exception as e:
                logger.warning(f"   ⚠️ liquid_momentum failed, using 0: {e}")
                out['liquid_momentum'] = np.zeros(len(data))

        return pd.DataFrame(out, index=data.index)

    def _compute_new_alpha_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """
        🔥 Compute high-alpha factors (T+5 optimized):
        - near_52w_high: 52-week high momentum (252-day window)
        - mom_accel_20_5: Momentum deceleration detector (daily slope comparison: recent 5d vs prior 15d)
        """
        logger.info("📊 [NEW FACTORS] Computing high-alpha factors")

        # Compute factors using grouped operations and transform to preserve index
        logger.info("   🔄 Computing near_52w_high (252-day high momentum)...")
        high_252_hist = data.groupby('ticker')['High'].transform(lambda x: x.rolling(252, min_periods=20).max().shift(1))
        near_52w_high = ((data['Close'] / high_252_hist) - 1).fillna(0)
        logger.info(f"   ✅ near_52w_high: mean={near_52w_high.mean():.4f}, std={near_52w_high.std():.4f}")

        logger.info("   🔄 Computing mom_accel_20_5 (momentum deceleration: daily slope recent 5d vs prior 15d)...")
        # 🔥 FIX: Use daily slope comparison instead of total return difference
        # This correctly detects deceleration/reversal in "spike-then-drop" scenarios
        close_now = data['Close']
        close_5d = data.groupby('ticker')['Close'].transform(lambda x: x.shift(5))
        close_20d = data.groupby('ticker')['Close'].transform(lambda x: x.shift(20))

        # Daily average return rate for recent 5 days
        mom_recent_5d = ((close_now / close_5d) ** (1.0 / 5.0) - 1).fillna(0)

        # Daily average return rate for prior 15 days (day 5-20)
        mom_prior_15d = ((close_5d / close_20d) ** (1.0 / 15.0) - 1).fillna(0)

        # Acceleration = recent slope - prior slope
        # Negative value = deceleration/reversal (recent momentum slower than before)
        mom_accel_20_5 = (mom_recent_5d - mom_prior_15d).fillna(0)
        logger.info(f"   ✅ mom_accel_20_5: mean={mom_accel_20_5.mean():.4f}, std={mom_accel_20_5.std():.4f}")

        logger.info("   ✅ New alpha factors computation completed")

        out = {'near_52w_high': near_52w_high}
        # Only compute mom_accel_20_5 if it's part of the active factor set (T+5)
        if 'mom_accel_20_5' in getattr(self, 'alpha_factors', []):
            out['mom_accel_20_5'] = mom_accel_20_5
        return pd.DataFrame(out, index=data.index)

    def _compute_mean_reversion_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """🔥 Compute mean reversion factors: rsi_21 (smoother RSI), price_ma60_deviation, bollinger_squeeze"""

        def _rsi21(x: pd.Series) -> pd.Series:
            ret = x.diff()
            gain = ret.clip(lower=0).rolling(21, min_periods=1).mean()
            loss = (-ret).clip(lower=0).rolling(21, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            # Regime context for T+10: invert RSI in bearish regime (price below MA200)
            # Keep output as standardized [-1, 1] to match existing pipeline expectations.
            if int(getattr(self, "horizon", 5) or 5) == 10:
                ma200 = x.rolling(200, min_periods=60).mean()
                bull = (x > ma200).astype(float)
                rsi = (bull * rsi) + ((1.0 - bull) * (100.0 - rsi))
            return (rsi - 50) / 50

        # 🔥 FIX: Use transform consistently and ensure Series output
        rsi = grouped['Close'].transform(_rsi21)
        ma20 = grouped['Close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        std20 = grouped['Close'].transform(lambda x: x.rolling(20, min_periods=1).std().fillna(0))
        bb_bandwidth = std20 / (ma20 + 1e-10)
        # T+10: convert squeeze to a directional breakout hint by flagging low-bandwidth regimes
        # and multiplying by medium-term return direction.
        if int(getattr(self, "horizon", 5) or 5) == 10:
            bw_q20 = bb_bandwidth.groupby(data['ticker']).transform(lambda s: s.rolling(126, min_periods=30).quantile(0.20))
            squeeze_flag = (bb_bandwidth < bw_q20).astype(float)
            dir20 = grouped['Close'].transform(lambda s: s.pct_change(20)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            bb_squeeze = squeeze_flag * np.sign(dir20)
        else:
            bb_squeeze = bb_bandwidth

        # 🔥 FIX: Use data['Close'] instead of grouped['Close'] for element-wise division
        ma60 = grouped['Close'].transform(lambda x: x.rolling(60, min_periods=10).mean())
        price_ma60_dev = (data['Close'] / (ma60 + 1e-10) - 1).fillna(0)

        return pd.DataFrame({
            'rsi_21': rsi,
            'bollinger_squeeze': bb_squeeze,
            'price_ma60_deviation': price_ma60_dev
        }, index=data.index)

    def _compute_volume_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """🔥 Compute volume factors: obv_momentum_60d, vol_ratio_20d, and (T+10) obv_divergence"""

        # 🔥 FIX: Ensure proper Series handling
        dir_ = grouped['Close'].transform(lambda s: s.pct_change()).fillna(0.0)
        dir_ = np.sign(dir_)

        # Calculate OBV and momentum
        obv = (dir_ * data['Volume']).groupby(data['ticker']).cumsum()
        obv_momentum_60d = obv.groupby(data['ticker']).pct_change(60).fillna(0)

        # Calculate volume ratio
        vol_ma20 = grouped['Volume'].transform(lambda v: v.rolling(20, min_periods=10).mean().shift(1))
        vol_ratio_20d = (data['Volume'] / (vol_ma20 + 1e-10) - 1).replace([np.inf, -np.inf], 0.0).fillna(0)

        out = {'obv_momentum_60d': obv_momentum_60d, 'vol_ratio_20d': vol_ratio_20d}

        if 'obv_divergence' in getattr(self, 'alpha_factors', []):
            try:
                def fast_norm(x: pd.Series) -> float:
                    if len(x) < 2:
                        return 0.0
                    mn = float(x.min())
                    mx = float(x.max())
                    return float((x.iloc[-1] - mn) / (mx - mn + 1e-10))

                price_norm = grouped['Close'].rolling(60, min_periods=30).apply(fast_norm, raw=False).reset_index(0, drop=True)
                obv_norm = obv.groupby(data['ticker']).rolling(60, min_periods=30).apply(fast_norm, raw=False).reset_index(0, drop=True)
                out['obv_divergence'] = (price_norm - obv_norm).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            except Exception as e:
                logger.warning(f"⚠️ obv_divergence failed, using 0: {e}")
                out['obv_divergence'] = np.zeros(len(data))

        return pd.DataFrame(out, index=data.index)

    def _compute_volatility_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute volatility factors: atr_ratio using transform/apply to preserve index"""

        prev_close = grouped['Close'].transform(lambda s: s.shift(1))
        high_low = data['High'] - data['Low']
        high_prev_close = (data['High'] - prev_close).abs()
        low_prev_close = (data['Low'] - prev_close).abs()

        tr_components = pd.concat([high_low, high_prev_close, low_prev_close], axis=1)
        true_range = tr_components.max(axis=1)

        atr_20d = true_range.groupby(data['ticker']).transform(lambda x: x.rolling(20, min_periods=1).mean())
        atr_5d = true_range.groupby(data['ticker']).transform(lambda x: x.rolling(5, min_periods=1).mean())
        atr_ratio = (atr_5d / (atr_20d + 1e-10) - 1).fillna(0)

        return pd.DataFrame({'atr_ratio': atr_ratio}, index=data.index)

    def _compute_ivol_20(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """
        T+10: IVOL 20 (Idiosyncratic Volatility proxy).
        Uses SPY as benchmark if present in the same dataset:
          ivol_20 = rolling_20_std( stock_ret_1d - spy_ret_1d ) per ticker
        If SPY is missing, returns zeros (still keeps feature name stable).
        """
        try:
            if 'date' not in data.columns:
                raise ValueError("ivol_20 requires 'date' column in compute_data")

            # Build SPY daily return series by date if SPY exists in the universe
            spy = data[data['ticker'].astype(str).str.upper().str.strip() == 'SPY'].copy()
            if spy.empty:
                logger.warning("⚠️ ivol_20: SPY not found in data; using zeros")
                return pd.DataFrame({'ivol_20': np.zeros(len(data))}, index=data.index)

            spy = spy.sort_values('date')
            spy_ret = spy['Close'].pct_change().replace([np.inf, -np.inf], np.nan)
            spy_ret_by_date = pd.Series(spy_ret.values, index=pd.to_datetime(spy['date']).dt.normalize())

            # Map SPY return to each row by date (same calendar)
            dates = pd.to_datetime(data['date']).dt.normalize()
            mkt_ret = dates.map(spy_ret_by_date).astype(float)

            # Stock daily return per ticker
            stock_ret = grouped['Close'].transform(lambda s: s.pct_change()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            diff = (stock_ret - mkt_ret).replace([np.inf, -np.inf], np.nan)

            ivol = diff.groupby(data['ticker']).transform(lambda s: s.rolling(20, min_periods=10).std()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return pd.DataFrame({'ivol_20': ivol}, index=data.index)
        except Exception as e:
            logger.warning(f"⚠️ ivol_20 failed, using zeros: {e}")
            return pd.DataFrame({'ivol_20': np.zeros(len(data))}, index=data.index)

    def _compute_fundamental_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """🔥 Placeholder for fundamental factors
        Note: ma30_ma60_cross is computed in _compute_ma_cross_30_60
        Note: hist_vol_40d is computed in _compute_blowoff_and_volatility
        """
        # Return empty DataFrame - all factors moved to specialized methods
        return pd.DataFrame(index=data.index)

    # REMOVED: _compute_ivol_factor (multicollinearity with hist_vol_40d, r=-0.95, VIF=10.4)

    def _compute_behavioral_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """🔥 Compute behavioral factors (T+5 optimized): only streak_reversal (removed overnight_intraday_gap, max_lottery_factor)"""
        try:
            # Initialize result arrays to match the input data length and preserve index alignment
            streak_reversal_values = []

            for ticker, ticker_data in grouped:
                ticker_data = ticker_data.sort_values('date')

                # Required columns
                if not all(col in ticker_data.columns for col in ['Open', 'Close', 'High', 'Low']):
                    logger.warning(f"Missing OHLC data for {ticker}")
                    n_obs = len(ticker_data)
                    # Append zeros for missing data
                    streak_reversal_values.extend([0.0] * n_obs)
                    continue

                # Return Streak Reversal (用户提供的精确版本)
                def streak_reversal_series(close: pd.Series,
                                         mkt_close: pd.Series | None = None,
                                         thr: float = 0.0005,  # 5bp 阈值
                                         cap: int = 5) -> pd.Series:
                    """
                    连续收益反转（streak reversal）——相对市场、带阈值、带长度上限
                    返回值已取负号：连涨越久→值越负；连跌越久→值越正。
                    """
                    # 1) 相对市场超额收益（不做 shift，信号在 t 收盘生成、预测 t→t+5）
                    r = close.pct_change()
                    if mkt_close is not None:
                        rm = mkt_close.pct_change().reindex_like(close)
                        r = r - rm
                    r = r.fillna(0.0).to_numpy()

                    # 2) 微动阈值（避免极小波动翻转）
                    s = np.where(r >  thr,  1,
                        np.where(r < -thr, -1, 0)).astype(np.int8)

                    # 3) 连续天数累计（符号一致则累加，否则重置符号为 ±1）
                    run = 0
                    out = np.zeros_like(s, dtype=np.int32)
                    for i, v in enumerate(s):
                        if v > 0:
                            run = run + 1 if run >= 0 else 1
                        elif v < 0:
                            run = run - 1 if run <= 0 else -1
                        else:
                            run = 0
                        out[i] = run

                    # 4) 长度上限（信息递减；3~5 常见）
                    out = np.clip(out, -cap, cap)

                    # 取负号="反转"定义：连涨越久→越想回落；连跌越久→越想反弹
                    return -pd.Series(out, index=close.index, name="streak_reversal")

                # 使用报告版精确实现
                # 注：暂时使用股票自身作为市场代理，后续可加入真实市场指数
                streak_reversal = streak_reversal_series(
                    close=ticker_data['Close'],
                    mkt_close=None,  # 暂无市场数据，可后续优化
                    thr=0.0005,      # 5bp 微动阈值
                    cap=5            # 连续天数上限
                )

                # Clean and handle edge cases
                streak_reversal = streak_reversal.replace([np.inf, -np.inf], 0).fillna(0)

                # Extend the values arrays
                streak_reversal_values.extend(streak_reversal.values)

            return pd.DataFrame({
                'streak_reversal': streak_reversal_values
            }, index=data.index)

        except Exception as e:
            logger.error(f"Behavioral factors computation failed: {e}")
            return pd.DataFrame({
                'streak_reversal': np.zeros(len(data))
            }, index=data.index)


class Simple21FactorEngine(Simple17FactorEngine):
    """Compatibility wrapper for the main model (T+1 optimized path).

    Provides the interface expected by `量化模型_bma_ultra_enhanced.py`,
    delegating computations to the existing 20-factor engine implementation.
    """

    def compute_all_21_factors(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute factors via the 20-factor engine.

        Returns a DataFrame with 20 factor columns plus 'Close', matching
        the expected 21-column layout in the caller logs.
        """
        return super().compute_all_20_factors(market_data)


def test_simple_20_factor_engine():
    """Test the simple 20 factor engine"""
    logger.info("Testing Simple 20 Factor Engine...")

    try:
        # Initialize
        engine = Simple20FactorEngine(lookback_days=120)
        
        # Test symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        
        # Fetch data
        market_data = engine.fetch_market_data(symbols)
        
        if market_data.empty:
            logger.error("No market data available")
            return None
        
        # Compute factors
        factors = engine.compute_all_20_factors(market_data)
        
        # Results
        logger.info("=" * 60)
        logger.info("SIMPLE 20 FACTOR ENGINE TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Market data shape: {market_data.shape}")
        logger.info(f"Factors shape: {factors.shape}")
        logger.info(f"Required factors: {len(REQUIRED_20_FACTORS)}")
        logger.info(f"Computed factors: {len(factors.columns)}")

        # Check all factors present
        missing = set(REQUIRED_14_FACTORS) - set(factors.columns)
        extra = set(factors.columns) - set(REQUIRED_20_FACTORS)
        
        logger.info(f"Missing factors: {missing if missing else 'None'}")
        logger.info(f"Extra factors: {extra if extra else 'None'}")
        
        # Factor statistics
        logger.info("\nFactor statistics:")
        for i, factor in enumerate(factors.columns):
            non_zero = (factors[factor] != 0).sum()
            mean_val = factors[factor].mean()
            std_val = factors[factor].std()
            logger.info(f"  {i+1:2d}. {factor:<25}: {non_zero:4d} non-zero, mean={mean_val:8.4f}, std={std_val:8.4f}")
        
        success = len(factors.columns) == 21 and len(missing) == 0  # 20 factors + Close

        if success:
            logger.info("\nSUCCESS: All 20 factors computed correctly!")
        else:
            logger.error("\nFAILED: Missing or incorrect factors")
        
        return factors if success else None
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# Backward compatibility aliases
Simple20FactorEngine = Simple17FactorEngine  # Alias for backward compatibility
Simple22FactorEngine = Simple17FactorEngine  # Alias for backward compatibility
Simple24FactorEngine = Simple17FactorEngine  # Alias for backward compatibility
Simple25FactorEngine = Simple17FactorEngine  # Alias for name consistency

# Add backward compatibility method to the class
def compute_all_20_factors(self, market_data: pd.DataFrame) -> pd.DataFrame:
    """Backward compatibility: compute all 17 factors"""
    all_17_factors = self.compute_all_17_factors(market_data)
    return all_17_factors

# Add backward compatibility method to the class
def compute_all_21_factors(self, market_data: pd.DataFrame) -> pd.DataFrame:
    """Backward compatibility: alias for compute_all_17_factors"""
    return self.compute_all_17_factors(market_data)

# Add backward compatibility method to the class
def compute_all_22_factors(self, market_data: pd.DataFrame) -> pd.DataFrame:
    """Backward compatibility: alias for compute_all_17_factors"""
    return self.compute_all_17_factors(market_data)

# Add backward compatibility method to the class
def compute_all_24_factors(self, market_data: pd.DataFrame) -> pd.DataFrame:
    """Backward compatibility: alias for compute_all_17_factors"""
    return self.compute_all_17_factors(market_data)

# Monkey-patch the methods onto the class
Simple17FactorEngine.compute_all_20_factors = compute_all_20_factors
Simple17FactorEngine.compute_all_21_factors = compute_all_21_factors
Simple17FactorEngine.compute_all_22_factors = compute_all_22_factors
Simple17FactorEngine.compute_all_24_factors = compute_all_24_factors

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    result = test_simple_20_factor_engine()

    if result is not None:
        print("\n" + "="*60)
        print("SUCCESS: SIMPLE 20 FACTOR ENGINE WORKING!")
        print("="*60)
        print(f"All 20 factors computed: {result.shape}")
        print("Ready for BMA integration!")
    else:
        print("\n" + "="*60)
        print("FAILED: SIMPLE 20 FACTOR ENGINE NOT WORKING")
        print("="*60)
