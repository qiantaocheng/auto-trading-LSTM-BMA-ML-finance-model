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

# THE EXACT 20 FACTORS REQUIRED BY BMA (Optimized)
# Removed: macd_histogram (redundant with momentum), stoch_k (redundant with RSI), market_cap_proxy (weak size effect)
# Removed: atr_20d (redundant with atr_ratio), ad_line (redundant with obv_momentum/MFI), quality_consistency (redundant with quality_proxy)
# Updated: Added 7 new factors for better T+5 prediction, removed redundant factors
# Total: 14 factors (streamlined from original set)
REQUIRED_14_FACTORS = [
    # Original momentum factors - REMOVED: momentum_20d, momentum_reversal_short
    'momentum_10d_ex1',
    # Technical indicators - REMOVED: price_to_ma20, cci (redundant with bollinger_position/RSI)
    'rsi_7', 'bollinger_squeeze',
    'obv_momentum',  # Removed ad_line (redundant)
    'atr_ratio',     # Removed atr_20d (redundant)
    'ivol_60d',      # Idiosyncratic volatility factor
    # Fundamental factors - REMOVED: growth_proxy, profitability_momentum, growth_acceleration, value_proxy, profitability_proxy, quality_proxy, mfi (redundant/unstable)
    'liquidity_factor',
    # NEW HIGH-ALPHA FACTORS (4 additions)
    'near_52w_high',      # 52-week high momentum
    'reversal_1d',        # 1-day reversal (T+1 friendly)
    'rel_volume_spike',   # Volume anomaly (20d z-score)
    'mom_accel_5_2',      # Momentum acceleration (5d vs 2d)
    # NEW BEHAVIORAL FACTORS (3 factors - need 180+ days)
    'overnight_intraday_gap',  # Overnight vs intraday return gap (83.7% @ 180d)
    'max_lottery_factor',      # Maximum return in recent window (88.6% @ 365d)
    'streak_reversal',         # Consecutive streak reversal (86.4% @ 30d, 83.3% @ 180d)
    # NEW CUSTOM FACTOR (user-requested, T+1 optimized)
    'price_efficiency_5d',     # Directional efficiency over 5d (T+1 optimized)
    # New triggered compression breakout bias (NR7 gate × close location)
    # Added for T+1; low-collinearity with sustained compression (e.g., bollinger_squeeze)
    # Name kept concise for downstream pipelines
    'nr7_breakout_bias'
]

# Keep backward compatibility
REQUIRED_16_FACTORS = REQUIRED_14_FACTORS       # Alias for backward compatibility
REQUIRED_17_FACTORS = REQUIRED_14_FACTORS       # Alias for backward compatibility
REQUIRED_20_FACTORS = REQUIRED_14_FACTORS       # Alias for backward compatibility
REQUIRED_22_FACTORS = REQUIRED_14_FACTORS       # Alias for backward compatibility
REQUIRED_24_FACTORS = REQUIRED_14_FACTORS       # Alias for backward compatibility

class Simple17FactorEngine:
    """
    Simple 17 Factor Engine - Complete High-Quality Factor Suite
    Directly computes all 17 high-quality factors: 15 alpha factors + sentiment_score + Close
    (Removed redundant and unstable factors: momentum_20d, momentum_reversal_short,
     price_to_ma20, cci, growth_proxy, profitability_momentum, growth_acceleration)
    """
    
    def __init__(self,
                 lookback_days: int = 252,
                 enable_sentiment: Optional[bool] = None,
                 polygon_api_key: Optional[str] = None,
                 sentiment_max_workers: int = 4,
                 sentiment_batch_size: int = 32):
        self.lookback_days = lookback_days

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
    
    def compute_all_17_factors(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute all 17 high-quality factors (15 alpha factors + sentiment_score + Close for target calculation)"""
        import time

        if market_data.empty:
            logger.error("No market data provided")
            return pd.DataFrame()

        logger.info("=" * 80)
        logger.info("COMPUTING ALL 17 HIGH-QUALITY FACTORS (15 ALPHA + SENTIMENT + CLOSE)")
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
            for factor_name in ['momentum_10d_ex1']:
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
        
        # 7: Volatility Factors (1 factor: atr_ratio)
        logger.info("Computing volatility factors (1/14)...")
        start_t = time.time()
        vol_results = self._compute_volatility_factors(compute_data, grouped)
        factor_timings['volatility'] = time.time() - start_t
        logger.info(f"   Volatility factors computed in {factor_timings['volatility']:.3f}s")
        all_factors.append(vol_results)

        # 8: IVOL Factor
        logger.info("Computing IVOL factor (1/14)...")
        start_t = time.time()
        ivol_result = self._compute_ivol_factor(compute_data)
        factor_timings['ivol'] = time.time() - start_t
        logger.info(f"   IVOL factor computed in {factor_timings['ivol']:.3f}s")
        all_factors.append(ivol_result)

        # 10-13: Fundamental Proxy Factors (REDUCED: removed growth_proxy, profitability_momentum, growth_acceleration)
        logger.info("Computing fundamental proxy factors (1/14)...")
        start_t = time.time()
        fundamental_results = self._compute_fundamental_factors(compute_data, grouped)
        factor_timings['fundamental'] = time.time() - start_t
        logger.info(f"   Fundamental factors computed in {factor_timings['fundamental']:.3f}s")
        all_factors.append(fundamental_results)

        # 14-17: High-Alpha Factors
        logger.info("Computing 4 high-alpha factors...")
        start_t = time.time()
        new_alpha_results = self._compute_new_alpha_factors(compute_data, grouped)
        factor_timings['new_alpha'] = time.time() - start_t
        logger.info(f"   High-alpha factors computed in {factor_timings['new_alpha']:.3f}s")
        all_factors.append(new_alpha_results)

        # 18-20: Behavioral Factors (TESTING coverage with more days)
        logger.info("Computing 3 behavioral factors...")
        start_t = time.time()
        behavioral_results = self._compute_behavioral_factors(compute_data, grouped)
        factor_timings['behavioral'] = time.time() - start_t
        logger.info(f"   Behavioral factors computed in {factor_timings['behavioral']:.3f}s")
        all_factors.append(behavioral_results)

        # 20+: NR7 breakout bias (triggered compression + close location)
        logger.info("Computing custom factor: nr7_breakout_bias (NR7 gate × close location)...")
        start_t = time.time()
        nr7_results = self._compute_nr7_breakout_bias(compute_data, grouped)
        factor_timings['nr7_breakout_bias'] = time.time() - start_t
        logger.info(f"   nr7_breakout_bias computed in {factor_timings['nr7_breakout_bias']:.3f}s")
        all_factors.append(nr7_results)

        # 21: New custom factor - price_efficiency_5d (T+1 optimized)
        logger.info("Computing custom factor: price_efficiency_5d (1/1)...")
        start_t = time.time()
        efficiency_results = self._compute_price_efficiency_5d(compute_data, grouped)
        factor_timings['price_efficiency_5d'] = time.time() - start_t
        logger.info(f"   price_efficiency_5d computed in {factor_timings['price_efficiency_5d']:.3f}s")
        all_factors.append(efficiency_results)

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
                sentiment_df = self._compute_sentiment_for_market_data(compute_data, limit_dates=limit_dates_for_sentiment)
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

                                # 如果有足够的真实数据(>20%)，用均值填充；否则用0填充
                                coverage_rate = before_fill / total_values
                                if coverage_rate > 0.2:  # 超过20%的数据有sentiment值
                                    fill_value = factors_df[col].mean()
                                    logger.info(f"   Using mean fill ({fill_value:.4f}) for {col} (coverage: {coverage_rate:.1%})")
                                else:
                                    fill_value = 0.0
                                    logger.info(f"   Using zero fill for {col} (low coverage: {coverage_rate:.1%})")

                                factors_df[col] = factors_df[col].fillna(fill_value)
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
        factor_columns = [col for col in factors_df.columns if col != 'Close']
        # Replace infinities with NaN for robust cross-sectional processing later
        factors_df[factor_columns] = factors_df[factor_columns].replace([np.inf, -np.inf], np.nan)
        
        # Verify all required factors are present
        missing = set(REQUIRED_14_FACTORS) - set(factors_df.columns)
        if missing:
            logger.error(f"Missing factors: {missing}")
            for factor in missing:
                factors_df[factor] = 0.0

        # Reorder columns: base factors first, then any extras (e.g., sentiment_*), finally Close
        base = REQUIRED_14_FACTORS
        extras = [c for c in factors_df.columns if c not in base + ['Close']]
        column_order = base + extras + ['Close']
        factors_df = factors_df[column_order]

        # ==============================
        # Cross-sectional robust winsorization and standardization (per date)
        # ==============================
        try:
            if isinstance(factors_df.index, pd.MultiIndex) and 'date' in factors_df.index.names:
                cs_cols = [c for c in factors_df.columns if c != 'Close']

                # Winsorize by IQR per date
                def _winsorize_group(g: pd.DataFrame) -> pd.DataFrame:
                    for c in cs_cols:
                        s = g[c].astype(float)
                        valid = s.dropna()
                        if len(valid) >= 5:
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
                    for c in cs_cols:
                        s = g[c].astype(float)
                        mean = s.mean()
                        std = s.std(ddof=0)
                        if not np.isfinite(std) or std == 0:
                            g[c] = 0.0
                        else:
                            g[c] = (s - mean) / (std + 1e-10)
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
        # 定义需要填充的长周期因子
        LONG_LOOKBACK_FACTORS = {
            'max_lottery_factor': 365,
            'near_52w_high': 252,
            'overnight_intraday_gap': 180,
        }

        if not isinstance(factors_df.index, pd.MultiIndex):
            logger.warning("factors_df must have MultiIndex for smart forward-fill, skipping")
            return factors_df

        nan_before = factors_df.isna().sum().sum()
        filled_df = factors_df.copy()
        fill_stats = {}

        # 按ticker分组处理
        for ticker in filled_df.index.get_level_values('ticker').unique():
            ticker_mask = filled_df.index.get_level_values('ticker') == ticker

            # 对每个长周期因子进行填充
            for factor_name, lookback_days in LONG_LOOKBACK_FACTORS.items():
                if factor_name not in filled_df.columns:
                    continue

                factor_series = filled_df.loc[ticker_mask, factor_name]
                nan_count_before = factor_series.isna().sum()

                if nan_count_before == 0:
                    continue

                # 找到第一个有效值
                first_valid_idx = factor_series.first_valid_index()

                if first_valid_idx is None:
                    # 全是NaN，用0填充
                    filled_df.loc[ticker_mask, factor_name] = 0
                    fill_stats[f'{ticker}_{factor_name}'] = 'all_nan_filled_zero'
                    continue

                # 获取第一个有效值
                first_valid_value = factor_series.loc[first_valid_idx]

                # 填充warm-up期（第一个有效值之前的NaN）
                warmup_mask = ticker_mask.copy()
                warmup_mask.loc[:] = False
                for idx in factor_series.index:
                    if idx < first_valid_idx and pd.isna(factor_series.loc[idx]):
                        warmup_mask.loc[idx] = True

                if warmup_mask.sum() > 0:
                    filled_df.loc[warmup_mask, factor_name] = first_valid_value
                    fill_stats[f'{ticker}_{factor_name}'] = f'warmup_filled_{warmup_mask.sum()}_samples'

                # 对中间的NaN用前向填充（限制30天）
                # 使用 ffill() 代替 fillna(method='ffill')
                filled_df.loc[ticker_mask, factor_name] = filled_df.loc[ticker_mask, factor_name].ffill(limit=30)

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

    def _compute_price_efficiency_5d(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute price_efficiency_5d: net move vs total path over 5 days (T+1 optimized).

        Definition:
        efficiency = (Close_t / Close_{t-5} - 1) / \\sum_{k=1..5} |Close_{t-k+1}/Close_{t-k} - 1|
        Range in [-1, 1]; higher means more directional and efficient trend.
        """
        try:
            # Net move over 5d for T+1 responsiveness
            close = data['Close']
            close_5d_ago = grouped['Close'].transform(lambda x: x.shift(5))
            net_move = (close / close_5d_ago - 1.0)

            # Sum absolute daily moves over last 5 days
            abs_roll_sum_5 = grouped.apply(lambda g: g['Close'].pct_change().abs().rolling(5, min_periods=1).sum()).reset_index(level=0, drop=True)

            denom = abs_roll_sum_5.replace(0, np.nan)
            efficiency = (net_move / denom).fillna(0.0)

            # Clean
            efficiency = efficiency.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            return pd.DataFrame({
                'price_efficiency_5d': efficiency
            }, index=data.index)
        except Exception as e:
            logger.warning(f"price_efficiency_5d computation failed: {e}")
            return pd.DataFrame({'price_efficiency_5d': np.zeros(len(data))}, index=data.index)

    def compute_all_20_factors(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute all 20 factors including behavioral factors"""
        return self.compute_all_17_factors(market_data)  # Use the updated method

    def _compute_momentum_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute momentum factors: momentum_10d_ex1 (exclude last 1d)"""

        logger.info("📊 [FACTOR COMPUTATION] Starting momentum factors calculation")
        factor_quality = {}

        # Momentum 10d - pure 10-day momentum
        logger.info("   🔄 Computing momentum_10d_ex1...")
        momentum_10d_ex1 = grouped['Close'].pct_change(10).fillna(0)
        factor_quality['momentum_10d_ex1'] = {
            'non_zero': (momentum_10d_ex1 != 0).sum(),
            'nan_count': momentum_10d_ex1.isna().sum(),
            'mean': momentum_10d_ex1.mean(),
            'std': momentum_10d_ex1.std(),
            'coverage': (momentum_10d_ex1 != 0).sum() / len(momentum_10d_ex1) * 100
        }
        logger.info(f"   ✅ momentum_10d_ex1: coverage={factor_quality['momentum_10d_ex1']['coverage']:.1f}%, mean={factor_quality['momentum_10d_ex1']['mean']:.4f}")

        # Data quality warning
        for factor_name, quality in factor_quality.items():
            if quality['coverage'] < 50:
                logger.warning(f"   ⚠️ {factor_name}: Low coverage {quality['coverage']:.1f}%")
            if quality['std'] == 0:
                logger.warning(f"   ⚠️ {factor_name}: Zero variance detected")

        logger.info("   ✅ Momentum factors computation completed")

        return pd.DataFrame({
            'momentum_10d_ex1': momentum_10d_ex1
        }, index=data.index)
    
    def _compute_new_alpha_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """
        Compute 4 new high-alpha factors:
        - near_52w_high: 52-week high momentum
        - reversal_1d: 1-day reversal
        - rel_volume_spike: Volume spike relative to 20-day max
        - mom_accel_5_2: Momentum acceleration (5d vs 2d for T+1)
        """
        logger.info("📊 [NEW FACTORS] Computing 4 high-alpha factors")

        # Compute factors using grouped operations and transform to preserve index
        logger.info("   🔄 Computing near_52w_high (52-week high momentum)...")
        # 使用更短窗口适配T+1：126天（半年），不含当日
        high_126_hist = data.groupby('ticker')['High'].transform(lambda x: x.rolling(126, min_periods=10).max().shift(1))
        near_52w_high = ((data['Close'] / high_126_hist) - 1).fillna(0)
        logger.info(f"   ✅ near_52w_high: mean={near_52w_high.mean():.4f}, std={near_52w_high.std():.4f}")

        logger.info("   🔄 Computing reversal_1d (1-day mean reversion)...")
        close_1d_ago = data.groupby('ticker')['Close'].transform(lambda x: x.shift(1))
        reversal_1d = -(((data['Close'] - close_1d_ago) / close_1d_ago)).fillna(0)
        logger.info(f"   ✅ reversal_1d: mean={reversal_1d.mean():.4f}, std={reversal_1d.std():.4f}")

        logger.info("   🔄 Computing rel_volume_spike (20d z-score)...")
        vol_ma20 = data.groupby('ticker')['Volume'].transform(lambda v: v.rolling(20, min_periods=1).mean())
        vol_std20 = data.groupby('ticker')['Volume'].transform(lambda v: v.rolling(20, min_periods=1).std())
        rel_volume_spike = ((data['Volume'] - vol_ma20) / (vol_std20 + 1e-10)).fillna(0)
        logger.info(f"   ✅ rel_volume_spike(z): mean={rel_volume_spike.mean():.4f}, std={rel_volume_spike.std():.4f}")

        logger.info("   🔄 Computing mom_accel_5_2 (momentum acceleration for T+1: 5d vs 2d)...")
        close_5d_ago = data.groupby('ticker')['Close'].transform(lambda x: x.shift(5))
        close_2d_ago = data.groupby('ticker')['Close'].transform(lambda x: x.shift(2))
        mom_5 = (data['Close'] - close_5d_ago) / close_5d_ago
        mom_2 = (data['Close'] - close_2d_ago) / close_2d_ago
        mom_accel_5_2 = (mom_2 - mom_5).fillna(0)
        logger.info(f"   ✅ mom_accel_5_2: mean={mom_accel_5_2.mean():.4f}, std={mom_accel_5_2.std():.4f}")

        logger.info("   ✅ New alpha factors computation completed")

        return pd.DataFrame({
            'near_52w_high': near_52w_high,
            'reversal_1d': reversal_1d,
            'rel_volume_spike': rel_volume_spike,
            'mom_accel_5_2': mom_accel_5_2
        }, index=data.index)

    def _compute_mean_reversion_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute mean reversion factors: rsi(7), bollinger_squeeze using transform/apply to preserve index"""

        def _rsi7(x: pd.Series) -> pd.Series:
            ret = x.diff()
            gain = ret.clip(lower=0).rolling(7, min_periods=1).mean()
            loss = (-ret).clip(lower=0).rolling(7, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return (rsi - 50) / 50

        rsi = grouped['Close'].transform(_rsi7)
        ma20 = grouped['Close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        std20 = grouped['Close'].transform(lambda x: x.rolling(20, min_periods=1).std().fillna(0))
        bb_squeeze = std20 / (ma20 + 1e-10)

        return pd.DataFrame({'rsi_7': rsi, 'bollinger_squeeze': bb_squeeze}, index=data.index)

    def _compute_nr7_breakout_bias(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """
        NR7 窄幅触发 + 收盘位置偏置:
        - 若当日为 NR7 (True Range 为过去7日最小)，则取 (C-L)/(H-L) - 0.5；否则 0。
        - 对 H==L 的边界返回 0（避免除零）；仅使用当日 OHLC，时间安全。
        - 输出名: nr7_breakout_bias
        """
        # True Range 组件
        prev_close = grouped['Close'].transform(lambda s: s.shift(1))
        tr_high_low = (data['High'] - data['Low']).abs()
        tr_high_pc = (data['High'] - prev_close).abs()
        tr_low_pc = (data['Low'] - prev_close).abs()
        tr = pd.concat([tr_high_low, tr_high_pc, tr_low_pc], axis=1).max(axis=1)

        # 7日窗口最小TR（不含当日以避免“与自己比”引入噪声，使用shift(1)）
        tr_rolling_min = tr.groupby(data['ticker']).transform(lambda x: x.rolling(7, min_periods=3).min().shift(1))
        # 当日是否为NR7（与过去7日最小TR相等，考虑浮点容差）
        eps = 1e-12
        is_nr7 = (tr <= (tr_rolling_min + eps)).fillna(False)

        # 收盘位置偏置 (C-L)/(H-L) - 0.5
        hl_range = (data['High'] - data['Low']).replace(0, np.nan)
        pos = ((data['Close'] - data['Low']) / hl_range) - 0.5
        pos = pos.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 触发门控：非NR7日置0
        nr7_breakout_bias = pos.where(is_nr7, 0.0)

        return pd.DataFrame({'nr7_breakout_bias': nr7_breakout_bias}, index=data.index)
    
    def _compute_volume_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute volume factors: obv_momentum using transform to preserve index"""

        dir_ = grouped['Close'].transform(lambda s: s.pct_change()).fillna(0.0).pipe(np.sign)
        obv = (dir_ * data['Volume']).groupby(data['ticker']).cumsum()
        obv_momentum = obv.groupby(data['ticker']).pct_change(10).fillna(0)

        return pd.DataFrame({'obv_momentum': obv_momentum}, index=data.index)
    
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
    
    def _compute_fundamental_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute fundamental proxy factors using transform to preserve index"""

        vol_ma20_hist = grouped['Volume'].transform(lambda v: v.rolling(20, min_periods=1).mean())
        liquidity_factor = (data['Volume'] / (vol_ma20_hist + 1e-10) - 1).fillna(0)

        return pd.DataFrame({'liquidity_factor': liquidity_factor}, index=data.index)

    def _compute_ivol_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute IVOL (Idiosyncratic Volatility) factor using adaptive rolling window

        IVOL 特质波动率 (T+10 适用):
        使用自适应滚动窗口计算相对于市场的特异性波动率

        改进版本：
        - 使用较短的20天窗口计算快速IVOL
        - 使用60天窗口计算稳定IVOL（如有足够数据）
        - 加权组合两个IVOL值，提高覆盖率
        """
        try:
            # 使用两个窗口：快速(10天)和稳定(30天) — 为T+1加速
            window_fast = 10
            window_stable = 30
            min_periods_fast = 5
            min_periods_stable = 10

            # Group by ticker for processing
            grouped = data.groupby('ticker')
            ivol_values = []

            # 计算独立市场基准收益 (使用等权重组合，排除目标股票)
            # 为每只股票创建独立的基准，防止自相关偏误

            for ticker, ticker_data in grouped:
                ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
                n_obs = len(ticker_data)

                # 创建独立市场基准 (排除当前股票)
                other_stocks_data = data[data['ticker'] != ticker]
                if other_stocks_data.empty:
                    # 如果只有一只股票，使用自身作为基准
                    market_close = ticker_data['Close']
                else:
                    market_close = other_stocks_data.groupby('date')['Close'].mean()
                market_returns = market_close.pct_change().fillna(0)

                # 计算股票收益
                close_prices = ticker_data['Close']
                log_returns = close_prices.pct_change().fillna(0).values

                # 获取对应日期的独立市场收益
                ticker_dates = ticker_data['date'].values
                market_log_returns = []
                for date in ticker_dates:
                    if date in market_returns.index:
                        market_log_returns.append(market_returns[date])
                    else:
                        market_log_returns.append(0.0)
                market_log_returns = np.array(market_log_returns)

                # 初始化结果数组
                ivol_result = np.full(n_obs, 0.0)

                # 计算快速IVOL (20天窗口，覆盖率更高)
                for i in range(window_fast, n_obs):
                    start_idx = i - window_fast
                    y = log_returns[start_idx:i]
                    x = market_log_returns[start_idx:i]

                    # 去除NaN和无限值
                    valid_mask = ~(np.isnan(y) | np.isnan(x) | np.isinf(y) | np.isinf(x))
                    if valid_mask.sum() < min_periods_fast:
                        continue

                    y_clean = y[valid_mask]
                    x_clean = x[valid_mask]

                    try:
                        # 简化回归或使用简单标准差
                        if len(np.unique(x_clean)) > 1:
                            # CAPM回归
                            X = np.column_stack([np.ones(len(x_clean)), x_clean])
                            beta_coef = np.linalg.lstsq(X, y_clean, rcond=None)[0]
                            predicted = X @ beta_coef
                            residuals = y_clean - predicted
                            ivol_fast = np.std(residuals, ddof=1)
                        else:
                            # 市场收益无变化，使用简单标准差
                            ivol_fast = np.std(y_clean, ddof=1)

                        ivol_result[i] = ivol_fast

                    except (np.linalg.LinAlgError, ValueError):
                        ivol_result[i] = np.std(y_clean, ddof=1) if len(y_clean) > 1 else 0.0

                # 如果有足够数据，计算稳定IVOL并混合
                if n_obs >= window_stable:
                    ivol_stable = np.full(n_obs, 0.0)

                    for i in range(window_stable, n_obs):
                        start_idx = i - window_stable
                        y = log_returns[start_idx:i]
                        x = market_log_returns[start_idx:i]

                        valid_mask = ~(np.isnan(y) | np.isnan(x) | np.isinf(y) | np.isinf(x))
                        if valid_mask.sum() < min_periods_stable:
                            continue

                        y_clean = y[valid_mask]
                        x_clean = x[valid_mask]

                        try:
                            if len(np.unique(x_clean)) > 1:
                                X = np.column_stack([np.ones(len(x_clean)), x_clean])
                                beta_coef = np.linalg.lstsq(X, y_clean, rcond=None)[0]
                                predicted = X @ beta_coef
                                residuals = y_clean - predicted
                                ivol_stable[i] = np.std(residuals, ddof=1)
                            else:
                                ivol_stable[i] = np.std(y_clean, ddof=1)

                        except (np.linalg.LinAlgError, ValueError):
                            ivol_stable[i] = 0.0

                    # 混合快速和稳定IVOL (优先使用稳定值，否则使用快速值)
                    for i in range(n_obs):
                        if ivol_stable[i] > 0:
                            # 有稳定IVOL时，加权平均 (70%稳定 + 30%快速)
                            if ivol_result[i] > 0:
                                ivol_result[i] = 0.7 * ivol_stable[i] + 0.3 * ivol_result[i]
                            else:
                                ivol_result[i] = ivol_stable[i]
                        # 否则保持快速IVOL值

                ivol_values.extend(ivol_result)

            return pd.DataFrame({'ivol_60d': ivol_values}, index=data.index)

        except Exception as e:
            logger.warning(f"IVOL computation failed: {e}")
            return pd.DataFrame({'ivol_60d': np.zeros(len(data))}, index=data.index)

    def _compute_behavioral_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute behavioral factors capturing market microstructure effects"""
        try:
            # Initialize result arrays to match the input data length and preserve index alignment
            overnight_intraday_gap_values = []
            max_lottery_factor_values = []
            streak_reversal_values = []

            for ticker, ticker_data in grouped:
                ticker_data = ticker_data.sort_values('date')

                # Required columns
                if not all(col in ticker_data.columns for col in ['Open', 'Close', 'High', 'Low']):
                    logger.warning(f"Missing OHLC data for {ticker}")
                    n_obs = len(ticker_data)
                    # Append zeros for missing data
                    overnight_intraday_gap_values.extend([0.0] * n_obs)
                    max_lottery_factor_values.extend([0.0] * n_obs)
                    streak_reversal_values.extend([0.0] * n_obs)
                    continue

                # 1) Overnight-Intraday Return Gap
                # Overnight return: Open[t] / Close[t-1] - 1
                r_on = ticker_data['Open'] / ticker_data['Close'].shift(1) - 1.0
                # Intraday return: Close[t] / Open[t] - 1
                r_day = ticker_data['Close'] / ticker_data['Open'] - 1.0
                # 20-day cumulative gap
                K = 20
                gap = (r_on - r_day).rolling(K, min_periods=K).sum().fillna(0)

                # 2) MAX Lottery Factor (maximum return in recent window)
                r_close = ticker_data['Close'] / ticker_data['Close'].shift(1) - 1.0
                max_factor = r_close.rolling(K, min_periods=K).max().fillna(0)

                # 3) Return Streak Reversal (用户提供的精确版本)
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
                gap = gap.replace([np.inf, -np.inf], 0).fillna(0)
                max_factor = max_factor.replace([np.inf, -np.inf], 0).fillna(0)
                streak_reversal = streak_reversal.replace([np.inf, -np.inf], 0).fillna(0)

                # Extend the values arrays
                overnight_intraday_gap_values.extend(gap.values)
                max_lottery_factor_values.extend(max_factor.values)
                streak_reversal_values.extend(streak_reversal.values)

            return pd.DataFrame({
                'overnight_intraday_gap': overnight_intraday_gap_values,
                'max_lottery_factor': max_lottery_factor_values,
                'streak_reversal': streak_reversal_values
            }, index=data.index)

        except Exception as e:
            logger.error(f"Behavioral factors computation failed: {e}")
            return pd.DataFrame({
                'overnight_intraday_gap': np.zeros(len(data)),
                'max_lottery_factor': np.zeros(len(data)),
                'streak_reversal': np.zeros(len(data))
            }, index=data.index)


class Simple21FactorEngine(Simple17FactorEngine):
    """Compatibility wrapper for the main model (T+5 optimized path).

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