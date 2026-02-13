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
from typing import Any, Dict, List, Optional
import logging
import os
try:
    import requests
except ImportError:
    requests = None
try:
    from bma_models.alpha_factor_quality_monitor import AlphaFactorQualityMonitor
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False

logger = logging.getLogger(__name__)

# 🔥 CORE ALPHA FACTORS (horizon-aware)
# - Stage-A T+5 factor set used across training/prediction
# - Legacy T+10 alias is preserved for backward compatibility
#
# NOTE: Downstream training expects a stable set of feature names. We keep both names but both map
#       to this Stage-A configuration.
CANONICAL_ALPHA_FACTORS = [
    'volume_price_corr_3d',
    'rsi_14',
    'reversal_3d',
    'momentum_10d',
    'liquid_momentum_10d',
    'sharpe_momentum_5d',
    'price_ma20_deviation',
    'avg_trade_size',
    'trend_r2_20',
    'dollar_vol_20',
    'ret_skew_20d',
    'reversal_5d',
    'near_52w_high',
    'atr_pct_14',
    'amihud_20',
]

# Canonical factor set used for both horizons
T5_ALPHA_FACTORS = [

    'volume_price_corr_3d',

    'rsi_14',

    'reversal_3d',

    'momentum_10d',

    'liquid_momentum_10d',

    'sharpe_momentum_5d',

    'price_ma20_deviation',

    'avg_trade_size',

    'trend_r2_20',

    'dollar_vol_20',

    'ret_skew_20d',

    'reversal_5d',

    'near_52w_high',

    'atr_pct_14',

    'amihud_20'

]

T10_ALPHA_FACTORS = T5_ALPHA_FACTORS.copy()


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
        # 🔥 ALWAYS USE STAGE-A T+5 FACTORS
        self.alpha_factors = T10_ALPHA_FACTORS
        logger.info(
            f"Using T+5 Stage-A factor set ({len(self.alpha_factors)} factors)"
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
        self._fundamental_cache: Dict[str, pd.DataFrame] = {}
        self._fundamental_fetch_disabled = False

        # Benchmark cache for beta-style factors (e.g., downside beta vs QQQ)
        self._benchmark_cache: Dict[str, pd.Series] = {}
        
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
        # CRITICAL: Ensure (ticker, date) is sorted for all rolling/shift ops.
        compute_data = market_data_clean.sort_values(['ticker', 'date']).reset_index(drop=True)
        grouped = compute_data.groupby('ticker', sort=False)
        
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
            for factor_name in ['momentum_10d', 'reversal_3d', 'reversal_5d', 'liquid_momentum_10d', 'sharpe_momentum_5d']:
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
        
        # 7: Volatility Factors (1 factor: atr_pct_14)
        logger.info("Computing volatility factors (1/14)...")
        start_t = time.time()
        vol_results = self._compute_volatility_factors(compute_data, grouped)
        factor_timings['volatility'] = time.time() - start_t
        logger.info(f"   Volatility factors computed in {factor_timings['volatility']:.3f}s")
        all_factors.append(vol_results)

        # REMOVED: IVOL Factor (multicollinearity with hist_vol_40d, r=-0.95, VIF=10.4)

        # 10-13: Fundamental Proxy Factors (skip if not in active factor set — very slow due to Polygon API)
        if {'roa', 'ebit'} & set(getattr(self, 'alpha_factors', [])):
            logger.info("Computing fundamental proxy factors (1/14)...")
            start_t = time.time()
            fundamental_results = self._compute_fundamental_factors(compute_data, grouped)
            factor_timings['fundamental'] = time.time() - start_t
            logger.info(f"   Fundamental factors computed in {factor_timings['fundamental']:.3f}s")
            all_factors.append(fundamental_results)
        else:
            logger.info("Skipping fundamental proxy factors (roa/ebit not in active factor set)")

        # Downside beta vs benchmark (QQQ)
        if 'downside_beta_ewm_21' in getattr(self, 'alpha_factors', []):
            logger.info("🔥 Computing downside_beta_ewm_21 vs QQQ (21-day EWMA)...")
            start_t = time.time()
            beta_results = self._compute_downside_beta_ewm_21(compute_data, grouped, benchmark='QQQ')
            factor_timings['downside_beta'] = time.time() - start_t
            logger.info(f"   Downside beta EWM 21 computed in {factor_timings['downside_beta']:.3f}s")
            all_factors.append(beta_results)

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

        # Trend R² 20d
        logger.info("   Computing trend_r2_20 (2/3)...")
        start_t = time.time()
        trend_r2_results = self._compute_trend_r2_20(compute_data, grouped)
        factor_timings['trend_r2_20'] = time.time() - start_t
        logger.info(f"   Trend_r2_20 computed in {factor_timings['trend_r2_20']:.3f}s")
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
        allowed_features = set(self.alpha_factors)

        factors_df = factors_df[[col for col in factors_df.columns if col in allowed_features]]



        
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
            # 🔥 FIX: T+1 execution lag — target = Close[T+1+horizon] / Close[T+1] - 1
            # Rationale: factors are computed at Close[T], so earliest executable
            # price is Close[T+1] (or Open[T+1]).  Old target used Close[T]→Close[T+h]
            # which assumed you could trade at the same close used for signal calc.
            next_close = factors_df.groupby(level='ticker')['Close'].shift(-1)
            future_close = factors_df.groupby(level='ticker')['Close'].shift(-(1 + self.horizon))
            target_series = (future_close / next_close - 1).replace([np.inf, -np.inf], np.nan)
            factors_df['target'] = target_series

            # 统计target质量
            total_samples = len(factors_df)
            valid_targets = target_series.notna().sum()
            valid_ratio = valid_targets / total_samples if total_samples > 0 else 0

            logger.info(f"✅ Target计算完成 (T+1 execution lag applied):")
            logger.info(f"   target = Close[T+1+{self.horizon}] / Close[T+1] - 1")
            logger.info(f"   总样本: {total_samples}")
            logger.info(f"   有效target: {valid_targets} ({valid_ratio:.1%})")
            logger.info(f"   缺失target: {total_samples - valid_targets} (最近{self.horizon + 1}天无未来数据)")

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
                        predict_target_date = pd.to_datetime(max_no_target) + pd.Timedelta(days=self.horizon + 1)
                        logger.info(f"🎯 预测目标日期: {predict_target_date.date()} (T+1+{self.horizon})")

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
            'near_52w_high': 252,
            'trend_r2_20': 20,
            'dollar_vol_20': 20,
            'amihud_20': 20,
            'ret_skew_20d': 20,
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
        """Compute 20-day return skewness with .shift(1) to avoid lookahead."""
        ret_skew = grouped['Close'].transform(
            lambda s: np.log(s / s.shift(1)).rolling(20, min_periods=20).skew().shift(1)
        )
        ret_skew = ret_skew.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({'ret_skew_20d': ret_skew}, index=data.index)

    def _compute_trend_r2_20(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute 20-day trend R² (corr² method) with .shift(1) to avoid lookahead."""
        window = 20

        def _r2_corr(arr: np.ndarray) -> float:
            if arr is None or len(arr) < window or not np.all(np.isfinite(arr)):
                return np.nan
            x = np.arange(len(arr))
            try:
                corr = np.corrcoef(x, arr)[0, 1]
                return corr ** 2
            except Exception:
                return np.nan

        r2 = grouped['Close'].transform(
            lambda s: s.rolling(window, min_periods=window).apply(_r2_corr, raw=True).shift(1)
        )
        r2 = r2.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({'trend_r2_20': r2}, index=data.index)

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
        """Compute T5 momentum factors: momentum_10d, reversal_3d, reversal_5d,
        liquid_momentum_10d, sharpe_momentum_5d. All shifted by 1 to avoid lookahead."""
        logger.info("[FACTOR] Computing T5 momentum factors (5 features)")

        # momentum_10d: 10-day return, shifted
        momentum_10d = grouped['Close'].transform(
            lambda s: s.pct_change(10).shift(1)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # reversal_3d: negative 3-day return, shifted
        reversal_3d = grouped['Close'].transform(
            lambda s: (-s.pct_change(3)).shift(1)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # reversal_5d: negative 5-day return, shifted
        reversal_5d = grouped['Close'].transform(
            lambda s: (-s.pct_change(5)).shift(1)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # liquid_momentum_10d: dollar-volume-weighted 10-day momentum, shifted
        ret_1d = grouped['Close'].transform(lambda s: s.pct_change())
        dollar_vol_raw = data['Close'] * data['Volume']
        weighted_ret_sum = (ret_1d * dollar_vol_raw).groupby(data['ticker']).transform(
            lambda s: s.rolling(10, min_periods=10).sum()
        )
        total_vol_sum = dollar_vol_raw.groupby(data['ticker']).transform(
            lambda s: s.rolling(10, min_periods=10).sum()
        )
        liquid_momentum_10d_raw = weighted_ret_sum / total_vol_sum.replace(0, np.nan)
        liquid_momentum_10d = liquid_momentum_10d_raw.groupby(data['ticker']).shift(1)
        liquid_momentum_10d = liquid_momentum_10d.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # sharpe_momentum_5d: 5-day mean return / std return, shifted
        def _sharpe_mom(s):
            ret = s.pct_change()
            mean_ret = ret.rolling(5, min_periods=5).mean()
            std_ret = ret.rolling(5, min_periods=5).std()
            sharpe = mean_ret / std_ret.replace(0, np.nan)
            return sharpe.shift(1)

        sharpe_momentum_5d = grouped['Close'].transform(_sharpe_mom)
        sharpe_momentum_5d = sharpe_momentum_5d.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        logger.info("[FACTOR] Momentum factors done")
        return pd.DataFrame({
            'momentum_10d': momentum_10d,
            'reversal_3d': reversal_3d,
            'reversal_5d': reversal_5d,
            'liquid_momentum_10d': liquid_momentum_10d,
            'sharpe_momentum_5d': sharpe_momentum_5d,
        }, index=data.index)

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
        """Compute T5 mean-reversion factors: rsi_14, price_ma20_deviation.
        All shifted by 1 to avoid lookahead."""
        logger.info("[FACTOR] Computing T5 mean-reversion factors (2 features)")

        # rsi_14: standard RSI(14), shifted
        def _rsi14(x: pd.Series) -> pd.Series:
            delta = x.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.rolling(window=14, min_periods=14).mean()
            avg_loss = loss.rolling(window=14, min_periods=14).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi.shift(1)

        rsi_14 = grouped['Close'].transform(_rsi14)
        rsi_14 = rsi_14.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # price_ma20_deviation: (Close - MA20) / MA20, shifted
        def _ma20_dev(s: pd.Series) -> pd.Series:
            ma = s.rolling(window=20, min_periods=20).mean()
            deviation = (s - ma) / ma.replace(0, np.nan)
            return deviation.shift(1)

        price_ma20_dev = grouped['Close'].transform(_ma20_dev)
        price_ma20_dev = price_ma20_dev.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        logger.info("[FACTOR] Mean-reversion factors done")
        return pd.DataFrame({
            'rsi_14': rsi_14,
            'price_ma20_deviation': price_ma20_dev,
        }, index=data.index)

    def _compute_volume_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute T5 volume factors: volume_price_corr_3d, dollar_vol_20, avg_trade_size, amihud_20.
        All shifted by 1 to avoid lookahead."""
        logger.info("[FACTOR] Computing T5 volume factors (4 features)")

        # volume_price_corr_3d: rolling 3-day correlation between returns and volume, shifted
        ret_for_corr = grouped['Close'].transform(lambda s: s.pct_change())
        # rolling corr needs per-ticker; use groupby transform workaround
        def _corr_transform(group):
            ret = group['Close'].pct_change()
            corr = ret.rolling(window=3, min_periods=3).corr(group['Volume'])
            return corr.shift(1)

        _corr_parts = []
        for ticker, grp in data.groupby('ticker'):
            _corr_parts.append(_corr_transform(grp))
        volume_price_corr_3d = pd.concat(_corr_parts).sort_index()
        volume_price_corr_3d = volume_price_corr_3d.reindex(data.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # dollar_vol_20: 20-day average dollar volume (Close * Volume), shifted
        dv_raw = data['Close'] * data['Volume']
        dollar_vol_20 = dv_raw.groupby(data['ticker']).transform(
            lambda s: s.rolling(20, min_periods=20).mean().shift(1)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # avg_trade_size: (Volume / Transactions) rolling 20-day mean, shifted
        if 'Transactions' in data.columns:
            ats_raw = data['Volume'] / data['Transactions'].replace(0, np.nan)
            avg_trade_size = ats_raw.groupby(data['ticker']).transform(
                lambda s: s.rolling(window=20, min_periods=20).mean().shift(1)
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            logger.warning("[FACTOR] 'Transactions' column missing; avg_trade_size will be 0")
            avg_trade_size = pd.Series(0.0, index=data.index)

        # amihud_20: 20-day average of |return| / dollar_volume (illiquidity), shifted
        abs_ret = grouped['Close'].transform(lambda s: s.pct_change().abs())
        dv_for_amihud = (data['Close'] * data['Volume']).replace(0, np.nan)
        illiq_raw = abs_ret / dv_for_amihud
        amihud_20 = illiq_raw.groupby(data['ticker']).transform(
            lambda s: s.rolling(20, min_periods=20).mean().shift(1)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        logger.info("[FACTOR] Volume factors done")
        return pd.DataFrame({
            'volume_price_corr_3d': volume_price_corr_3d,
            'dollar_vol_20': dollar_vol_20,
            'avg_trade_size': avg_trade_size,
            'amihud_20': amihud_20,
        }, index=data.index)

    def _compute_volatility_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute T5 volatility factor: atr_pct_14 = ATR(14) / Close, shifted by 1."""
        logger.info("[FACTOR] Computing T5 volatility factors (1 feature)")

        prev_close = grouped['Close'].transform(lambda s: s.shift(1))
        high_low = data['High'] - data['Low']
        high_prev_close = (data['High'] - prev_close).abs()
        low_prev_close = (data['Low'] - prev_close).abs()

        tr_components = pd.concat([high_low, high_prev_close, low_prev_close], axis=1)
        true_range = tr_components.max(axis=1)

        atr_14 = true_range.groupby(data['ticker']).transform(
            lambda x: x.rolling(14, min_periods=14).mean()
        )
        atr_pct_14_raw = atr_14 / data['Close'].replace(0, np.nan)
        atr_pct_14 = atr_pct_14_raw.groupby(data['ticker']).shift(1)
        atr_pct_14 = atr_pct_14.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        logger.info("[FACTOR] Volatility factors done")
        return pd.DataFrame({'atr_pct_14': atr_pct_14}, index=data.index)

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

    @staticmethod
    def _coerce_polygon_value(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, dict):
            for key in ('value', 'amount', 'raw', 'reported_value'):
                if key in value and value[key] is not None:
                    return Simple17FactorEngine._coerce_polygon_value(value[key])
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_polygon_metric(record: Dict[str, Any], candidates: List[str]) -> Optional[float]:
        if not isinstance(record, dict):
            return None
        search_spaces: List[Dict[str, Any]] = [record]
        financials = record.get('financials')
        if isinstance(financials, dict):
            for section in ('income_statement', 'balance_sheet', 'cash_flow_statement', 'comprehensive_income', 'ratios'):
                block = financials.get(section)
                if isinstance(block, dict):
                    search_spaces.append(block)
        for space in search_spaces:
            for name in candidates:
                if name in space:
                    val = Simple17FactorEngine._coerce_polygon_value(space.get(name))
                    if val is not None:
                        return val
        return None

    def _fetch_polygon_financial_history(self, ticker: str) -> pd.DataFrame:
        ticker_norm = str(ticker).upper().strip()
        cached = self._fundamental_cache.get(ticker_norm)
        if cached is not None:
            return cached
        if not self.polygon_api_key or requests is None:
            if not self._fundamental_fetch_disabled:
                logger.warning("?? Polygon fundamentals unavailable - using zeros for roa/ebit")
                self._fundamental_fetch_disabled = True
            self._fundamental_cache[ticker_norm] = pd.DataFrame()
            return self._fundamental_cache[ticker_norm]
        url = "https://api.polygon.io/vX/reference/financials"
        params = {
            'ticker': ticker_norm,
            'timeframe': 'quarterly',
            'limit': 8,
            # Polygon uses 'asc' / 'desc'
            'order': 'desc',
            'apiKey': self.polygon_api_key,
        }
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json() or {}
        except Exception as exc:
            logger.warning(f"?? Polygon financials fetch failed for {ticker_norm}: {exc}")
            self._fundamental_cache[ticker_norm] = pd.DataFrame()
            return self._fundamental_cache[ticker_norm]
        rows = []
        for entry in payload.get('results') or []:
            eff_date = entry.get('filing_date') or entry.get('report_date') or entry.get('end_date') or entry.get('calendar_date')
            eff_ts = pd.to_datetime(eff_date, errors='coerce')
            if pd.isna(eff_ts):
                continue

            # Raw fundamentals / ratios
            roa_ratio = self._extract_polygon_metric(entry, ['return_on_assets', 'roa'])
            ebit_q = self._extract_polygon_metric(entry, [
                'ebit',
                'earnings_before_interest_and_taxes',
                'earnings_before_interest_and_tax',
                'operating_income',
                'operating_income_loss',
            ])
            net_income_q = self._extract_polygon_metric(entry, [
                'net_income',
                'net_income_loss',
                'net_income_common_stockholders',
            ])
            total_assets = self._extract_polygon_metric(entry, [
                'total_assets',
                'assets',
            ])
            market_cap = self._extract_polygon_metric(entry, ['market_cap', 'marketCapitalization', 'market_capitalization'])
            enterprise_value = self._extract_polygon_metric(entry, ['enterprise_value', 'enterpriseValue'])

            rows.append({
                'effective_date': eff_ts.normalize(),
                'roa_ratio': roa_ratio,
                'ebit_q': ebit_q,
                'net_income_q': net_income_q,
                'total_assets': total_assets,
                'market_cap': market_cap,
                'enterprise_value': enterprise_value,
            })
        history = pd.DataFrame(rows)
        if not history.empty:
            history = history.sort_values('effective_date')

            # Compute TTM (rolling sum of last 4 quarters) for scale-stable yields
            # These are aligned to effective_date and will be merged "as of" trading dates (backward only).
            for col in ('ebit_q', 'net_income_q'):
                if col in history.columns:
                    history[col] = pd.to_numeric(history[col], errors='coerce')
            history['ebit_ttm'] = history['ebit_q'].rolling(4, min_periods=1).sum()
            history['net_income_ttm'] = history['net_income_q'].rolling(4, min_periods=1).sum()

            # Compute ROA if not provided: NetIncome_TTM / TotalAssets
            roa_calc = None
            try:
                roa_calc = history['net_income_ttm'] / history['total_assets']
            except Exception:
                roa_calc = None
            history['roa'] = history['roa_ratio']
            if roa_calc is not None:
                history['roa'] = history['roa'].where(history['roa'].notna(), roa_calc)

            # Compute scaled EBIT and CFO yields if denominators are available.
            # - ebit: EBIT/EV when enterprise_value exists; fallback to EBIT_TTM
            history['ebit'] = history['ebit_ttm']
            if 'enterprise_value' in history.columns:
                ev = pd.to_numeric(history['enterprise_value'], errors='coerce')
                denom = ev.replace({0.0: np.nan})
                history['ebit'] = (history['ebit_ttm'] / denom).where(denom.notna(), history['ebit_ttm'])

            # Keep only the final factor columns + effective_date for merge_asof
            history = history[['effective_date', 'roa', 'ebit']].copy()
        self._fundamental_cache[ticker_norm] = history
        return history

    def _compute_fundamental_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute Polygon-based fundamental factors (roa, ebit)."""
        logger.info("Computing Polygon fundamentals (roa/ebit)...")
        fundamentals = pd.DataFrame({
            'roa': np.nan,
            'ebit': np.nan,
        }, index=data.index)
        if 'date' not in data.columns or 'ticker' not in data.columns:
            logger.warning("Fundamental computation requires 'date' and 'ticker' columns; returning zeros")
            return fundamentals.fillna(0.0)
        unique_tickers = data['ticker'].dropna().astype(str).unique()
        for ticker in unique_tickers:
            history = self._fetch_polygon_financial_history(ticker)
            if history.empty:
                continue
            ticker_mask = data['ticker'] == ticker
            row_idx = data.index[ticker_mask]
            ticker_dates = pd.DataFrame({
                'row_idx': row_idx,
                'date': pd.to_datetime(data.loc[row_idx, 'date']).dt.normalize(),
            }).sort_values('date').reset_index(drop=True)
            hist = history.dropna(subset=['effective_date']).copy()
            if hist.empty:
                continue
            merged = pd.merge_asof(
                ticker_dates[['date']],
                hist.rename(columns={'effective_date': 'date'}),
                on='date',
                direction='backward'
            )
            merged['row_idx'] = ticker_dates['row_idx'].values
            fundamentals.loc[merged['row_idx'], ['roa', 'ebit']] = merged[['roa', 'ebit']].values
        total = len(fundamentals) or 1
        coverage = {col: (~fundamentals[col].isna()).sum() / total * 100 for col in ['roa', 'ebit']}
        logger.info("   Fundamental coverage: " + ', '.join(f"{col}={pct:.1f}%" for col, pct in coverage.items()))
        return fundamentals

    def _get_benchmark_returns_by_date(self, benchmark: str, dates: pd.Series) -> Optional[pd.Series]:
        """
        Return daily % returns for benchmark indexed by normalized date.
        Uses cache; tries local data first if present; otherwise falls back to Polygon historical bars.
        """
        try:
            bench = str(benchmark).upper().strip()
        except Exception:
            bench = 'QQQ'

        # Cache key includes benchmark only; values are date-indexed.
        cached = self._benchmark_cache.get(bench)
        if cached is not None and not cached.empty:
            return cached

        # Try to fetch via polygon_client (preferred) if available.
        try:
            from polygon_client import polygon_client
        except Exception:
            polygon_client = None

        # Compute date window
        try:
            dmin = pd.to_datetime(dates).dt.normalize().min()
            dmax = pd.to_datetime(dates).dt.normalize().max()
        except Exception:
            dmin, dmax = None, None
        if dmin is None or dmax is None or pd.isna(dmin) or pd.isna(dmax):
            return None
        start = pd.Timestamp(dmin).strftime('%Y-%m-%d')
        end = pd.Timestamp(dmax).strftime('%Y-%m-%d')

        # Fallback to raw Polygon REST if polygon_client is unavailable.
        if polygon_client is None:
            if not self.polygon_api_key or requests is None:
                return None
            try:
                url = f"https://api.polygon.io/v2/aggs/ticker/{bench}/range/1/day/{start}/{end}"
                params = {
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": 50000,
                    "apiKey": self.polygon_api_key,
                }
                resp = requests.get(url, params=params, timeout=20)
                resp.raise_for_status()
                payload = resp.json() or {}
                results = payload.get("results") or []
                if not results:
                    return None
                idx = pd.to_datetime([r.get("t") for r in results], unit="ms", errors="coerce").normalize()
                close = pd.to_numeric([r.get("c") for r in results], errors="coerce")
                s = pd.Series(close, index=idx).sort_index()
                ret = s.pct_change().replace([np.inf, -np.inf], np.nan)
                self._benchmark_cache[bench] = ret
                return ret
            except Exception:
                return None

        try:
            df = polygon_client.get_historical_bars(bench, start, end, 'day', 1)
            if df is None or df.empty:
                return None
            df = df.sort_index()
            # Use Close return (aligned to trading date)
            ret = df['Close'].pct_change().replace([np.inf, -np.inf], np.nan)
            ret_by_date = pd.Series(ret.values, index=pd.to_datetime(df.index).normalize())
            self._benchmark_cache[bench] = ret_by_date
            return ret_by_date
        except Exception:
            return None

    def _compute_downside_beta_ewm_21(self, data: pd.DataFrame, grouped, benchmark: str = 'QQQ') -> pd.DataFrame:
        """
        Downside beta using EWMA with 21-day span (~1 month):
        Uses only days where benchmark return < 0, weighted exponentially (recent data more important).
        
        Formula: beta_down_ewm = Cov_ewm(R_s, R_m | R_m < 0) / Var_ewm(R_m | R_m < 0)
        
        Returns:
          - downside_beta_ewm_21
        """
        if 'date' not in data.columns or 'Close' not in data.columns or 'ticker' not in data.columns:
            return pd.DataFrame(
                {'downside_beta_ewm_21': np.zeros(len(data))},
                index=data.index,
            )

        dates = pd.to_datetime(data['date']).dt.normalize()
        bench_ret_by_date = self._get_benchmark_returns_by_date(benchmark, dates)
        if bench_ret_by_date is None or bench_ret_by_date.empty:
            # No benchmark -> return zeros
            return pd.DataFrame(
                {'downside_beta_ewm_21': np.zeros(len(data))},
                index=data.index,
            )

        bench_ret = dates.map(bench_ret_by_date).astype(float)
        stock_ret = grouped['Close'].transform(lambda s: s.pct_change()).replace([np.inf, -np.inf], np.nan)

        # Downside-only series (NaN on non-down days -> EWMA ignores them)
        is_down = bench_ret < 0
        stock_down = stock_ret.where(is_down, np.nan)
        bench_down = pd.Series(bench_ret, index=data.index).where(is_down, np.nan)

        def _downside_beta_ewm_for_group(g: pd.DataFrame) -> pd.DataFrame:
            """
            Compute EWMA downside beta for a single ticker group.
            Uses span=21 for ~1 month window with exponential decay.
            """
            sd = g['stock_down']
            bd = g['bench_down']
            
            # EWMA parameters: span=21 days (~1 month)
            span = 21
            min_periods = max(5, span // 4)  # Require at least 5 down days
            
            # Compute EWMA means
            ewm_sd = sd.ewm(span=span, min_periods=min_periods, adjust=False).mean()
            ewm_bd = bd.ewm(span=span, min_periods=min_periods, adjust=False).mean()
            
            # Compute EWMA of product for covariance: E[XY]
            product = (sd * bd).ewm(span=span, min_periods=min_periods, adjust=False).mean()
            
            # Covariance: Cov(X,Y) = E[XY] - E[X]E[Y]
            cov_ewm = product - (ewm_sd * ewm_bd)
            
            # Variance: Var(Y) = E[Y^2] - E[Y]^2
            bd_squared = (bd * bd).ewm(span=span, min_periods=min_periods, adjust=False).mean()
            var_ewm = bd_squared - (ewm_bd * ewm_bd)
            
            # Beta: beta = Cov(X,Y) / Var(Y)
            beta_down_ewm = cov_ewm / var_ewm.replace({0.0: np.nan})
            
            return pd.DataFrame({'downside_beta_ewm_21': beta_down_ewm}, index=g.index)

        tmp = pd.DataFrame(
            {
                'stock_ret': stock_ret,
                'bench_ret': bench_ret,
                'stock_down': stock_down,
                'bench_down': bench_down,
            },
            index=data.index,
        )

        betas = tmp.groupby(data['ticker'], sort=False, group_keys=False).apply(_downside_beta_ewm_for_group)
        betas = betas.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return betas.reindex(data.index)
    # REMOVED: _compute_ivol_factor (multicollinearity with hist_vol_40d, r=-0.95, VIF=10.4)

    def _compute_behavioral_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """🔥 Compute behavioral factors (T+5 optimized): only streak_reversal (removed overnight_intraday_gap, max_lottery_factor)"""
        try:
            # IMPORTANT: preserve index alignment with `data` using groupby-apply + reset_index.
            def streak_reversal_series(close: pd.Series,
                                      mkt_close: pd.Series | None = None,
                                      thr: float = 0.0005,
                                      cap: int = 5) -> pd.Series:
                """
                Streak reversal (mean-reversion pressure).
                Output is negated: longer up-streak -> more negative; longer down-streak -> more positive.
                """
                r = close.pct_change()
                if mkt_close is not None:
                    rm = mkt_close.pct_change().reindex_like(close)
                    r = r - rm
                r = r.fillna(0.0).to_numpy()
                s = np.where(r > thr, 1, np.where(r < -thr, -1, 0)).astype(np.int8)

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

                out = np.clip(out, -cap, cap)
                return -pd.Series(out, index=close.index, name="streak_reversal")

            sr = grouped['Close'].apply(lambda s: streak_reversal_series(s, None, 0.0005, 5))
            sr = sr.reset_index(level=0, drop=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            sr = sr.reindex(data.index).fillna(0.0)
            return pd.DataFrame({'streak_reversal': sr}, index=data.index)

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
