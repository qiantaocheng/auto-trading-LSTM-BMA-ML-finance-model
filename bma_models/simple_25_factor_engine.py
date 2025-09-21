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
    'momentum_10d',
    # Technical indicators - REMOVED: price_to_ma20, cci (redundant with bollinger_position/RSI)
    'rsi', 'bollinger_squeeze',
    'obv_momentum',  # Removed ad_line (redundant)
    'atr_ratio',     # Removed atr_20d (redundant)
    'ivol_60d',      # Idiosyncratic volatility factor
    # Fundamental factors - REMOVED: growth_proxy, profitability_momentum, growth_acceleration, value_proxy, profitability_proxy, quality_proxy, mfi (redundant/unstable)
    'liquidity_factor',
    # NEW HIGH-ALPHA FACTORS (4 additions)
    'near_52w_high',      # 52-week high momentum
    'reversal_5d',        # 5-day reversal
    'rel_volume_spike',   # Volume spike relative to 20-day max
    'mom_accel_10_5',     # Momentum acceleration (5d vs 10d)
    # NEW BEHAVIORAL FACTORS (3 microstructure additions)
    'overnight_intraday_gap',  # Overnight vs intraday return gap
    'max_lottery_factor',      # Maximum return in recent window (lottery effect)
    'streak_reversal'          # Consecutive return streak reversal signal
]

# Keep backward compatibility
REQUIRED_16_FACTORS = REQUIRED_14_FACTORS       # Alias for backward compatibility
REQUIRED_17_FACTORS = REQUIRED_14_FACTORS       # Alias for backward compatibility
REQUIRED_20_FACTORS = REQUIRED_14_FACTORS       # Alias for backward compatibility
REQUIRED_22_FACTORS = REQUIRED_14_FACTORS       # Alias for backward compatibility
REQUIRED_24_FACTORS = REQUIRED_14_FACTORS       # Alias for backward compatibility

class Simple17FactorEngine:
    """
    Simple 14 Factor Engine - No Dependencies (Backward Compatible Name)
    Directly computes all 17 high-quality factors with robust implementation
    (Removed redundant and unstable factors: momentum_20d, momentum_reversal_short,
     price_to_ma20, cci, growth_proxy, profitability_momentum, growth_acceleration)
    """
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
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
        """Compute all 20 high-quality factors (17 original + 3 behavioral factors)"""
        import time

        if market_data.empty:
            logger.error("No market data provided")
            return pd.DataFrame()

        logger.info("=" * 80)
        logger.info("COMPUTING ALL 14 HIGH-QUALITY ALPHA FACTORS WITH BEHAVIORAL FACTORS")
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
            if isinstance(market_data_clean.index, pd.DatetimeIndex):
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
        grouped = market_data_clean.groupby('ticker')
        
        # Collect all factor results, ensuring consistent indexing
        all_factors = []
        
        # 1: Momentum Factors (REDUCED: only momentum_10d)
        logger.info("="*60)
        logger.info("🎯 [ALPHA FACTOR 1] MOMENTUM FACTORS")
        logger.info("="*60)
        start_t = time.time()
        momentum_results = self._compute_momentum_factors(market_data_clean, grouped)
        factor_timings['momentum'] = time.time() - start_t

        # Monitor each momentum factor if monitor available
        if self.factor_monitor:
            for factor_name in ['momentum_10d']:
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
        meanrev_results = self._compute_mean_reversion_factors(market_data_clean, grouped)
        factor_timings['mean_reversion'] = time.time() - start_t
        logger.info(f"   Mean reversion factors computed in {factor_timings['mean_reversion']:.3f}s")
        all_factors.append(meanrev_results)

        # 5-6: Volume Factors
        logger.info("Computing volume factors (1/14)...")
        start_t = time.time()
        volume_results = self._compute_volume_factors(market_data_clean, grouped)
        factor_timings['volume'] = time.time() - start_t
        logger.info(f"   Volume factors computed in {factor_timings['volume']:.3f}s")
        all_factors.append(volume_results)
        
        # 7: Volatility Factors (1 factor: atr_ratio)
        logger.info("Computing volatility factors (1/14)...")
        start_t = time.time()
        vol_results = self._compute_volatility_factors(market_data_clean, grouped)
        factor_timings['volatility'] = time.time() - start_t
        logger.info(f"   Volatility factors computed in {factor_timings['volatility']:.3f}s")
        all_factors.append(vol_results)

        # 8: IVOL Factor
        logger.info("Computing IVOL factor (1/14)...")
        start_t = time.time()
        ivol_result = self._compute_ivol_factor(market_data_clean)
        factor_timings['ivol'] = time.time() - start_t
        logger.info(f"   IVOL factor computed in {factor_timings['ivol']:.3f}s")
        all_factors.append(ivol_result)

        # 10-13: Fundamental Proxy Factors (REDUCED: removed growth_proxy, profitability_momentum, growth_acceleration)
        logger.info("Computing fundamental proxy factors (1/14)...")
        start_t = time.time()
        fundamental_results = self._compute_fundamental_factors(market_data_clean, grouped)
        factor_timings['fundamental'] = time.time() - start_t
        logger.info(f"   Fundamental factors computed in {factor_timings['fundamental']:.3f}s")
        all_factors.append(fundamental_results)

        # 14-17: High-Alpha Factors
        logger.info("Computing 4 high-alpha factors...")
        start_t = time.time()
        new_alpha_results = self._compute_new_alpha_factors(market_data_clean, grouped)
        factor_timings['new_alpha'] = time.time() - start_t
        logger.info(f"   High-alpha factors computed in {factor_timings['new_alpha']:.3f}s")
        all_factors.append(new_alpha_results)

        # 18-20: Behavioral Factors (NEW)
        logger.info("Computing 3 behavioral factors...")
        start_t = time.time()
        behavioral_results = self._compute_behavioral_factors(market_data_clean, grouped)
        factor_timings['behavioral'] = time.time() - start_t
        logger.info(f"   Behavioral factors computed in {factor_timings['behavioral']:.3f}s")
        all_factors.append(behavioral_results)

        # Combine all factor DataFrames
        factors_df = pd.concat(all_factors, axis=1)
        
        # Add Close prices BEFORE setting MultiIndex to preserve alignment
        factors_df['Close'] = market_data_clean['Close']
        
        # Set MultiIndex using the prepared date and ticker columns
        factors_df.index = pd.MultiIndex.from_arrays(
            [market_data_clean['date'], market_data_clean['ticker']], 
            names=['date', 'ticker']
        )
        
        # Clean data for factors only (preserve Close prices)
        factor_columns = [col for col in factors_df.columns if col != 'Close']
        factors_df[factor_columns] = factors_df[factor_columns].replace([np.inf, -np.inf], 0)
        factors_df[factor_columns] = factors_df[factor_columns].fillna(0)
        
        # Verify all 20 factors are present
        missing = set(REQUIRED_14_FACTORS) - set(factors_df.columns)
        if missing:
            logger.error(f"Missing factors: {missing}")
            for factor in missing:
                factors_df[factor] = 0.0

        # Reorder columns: 14 factors first, then Close for target generation
        column_order = REQUIRED_14_FACTORS + ['Close']
        factors_df = factors_df[column_order]

        logger.info("=" * 60)
        logger.info(f"ALL 14 HIGH-QUALITY FACTORS + CLOSE COMPUTED: {factors_df.shape}")
        logger.info("Factor Computation Timing:")
        total_time = sum(factor_timings.values())
        for name, duration in factor_timings.items():
            pct = 100 * duration / total_time if total_time > 0 else 0
            logger.info(f"   {name:<15}: {duration:.3f}s ({pct:.1f}%)")
        logger.info(f"   {'TOTAL':<15}: {total_time:.3f}s")
        logger.info("=" * 60)

        return factors_df

    def compute_all_20_factors(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Compute all 20 factors including behavioral factors"""
        return self.compute_all_17_factors(market_data)  # Use the updated method

    def _compute_momentum_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute momentum factors: momentum_10d only (REMOVED: momentum_20d, momentum_reversal_short)"""

        logger.info("📊 [FACTOR COMPUTATION] Starting momentum factors calculation")
        factor_quality = {}

        # Momentum 10d
        logger.info("   🔄 Computing momentum_10d...")
        momentum_10d = grouped['Close'].pct_change(10).fillna(0)
        factor_quality['momentum_10d'] = {
            'non_zero': (momentum_10d != 0).sum(),
            'nan_count': momentum_10d.isna().sum(),
            'mean': momentum_10d.mean(),
            'std': momentum_10d.std(),
            'coverage': (momentum_10d != 0).sum() / len(momentum_10d) * 100
        }
        logger.info(f"   ✅ momentum_10d: coverage={factor_quality['momentum_10d']['coverage']:.1f}%, mean={factor_quality['momentum_10d']['mean']:.4f}")

        # Data quality warning
        for factor_name, quality in factor_quality.items():
            if quality['coverage'] < 50:
                logger.warning(f"   ⚠️ {factor_name}: Low coverage {quality['coverage']:.1f}%")
            if quality['std'] == 0:
                logger.warning(f"   ⚠️ {factor_name}: Zero variance detected")

        logger.info("   ✅ Momentum factors computation completed")

        return pd.DataFrame({
            'momentum_10d': momentum_10d
        }, index=data.index)
    
    def _compute_new_alpha_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """
        Compute 4 new high-alpha factors:
        - near_52w_high: 52-week high momentum
        - reversal_5d: 5-day reversal
        - rel_volume_spike: Volume spike relative to 20-day max
        - mom_accel_10_5: Momentum acceleration (5d vs 10d)
        """
        logger.info("📊 [NEW FACTORS] Computing 4 high-alpha factors")

        # Compute factors using grouped operations and transform to preserve index
        logger.info("   🔄 Computing near_52w_high (52-week high momentum)...")
        high_252 = data.groupby('ticker')['High'].transform(lambda x: x.rolling(252, min_periods=20).max())
        near_52w_high = ((data['Close'] / high_252) - 1).fillna(0)
        logger.info(f"   ✅ near_52w_high: mean={near_52w_high.mean():.4f}, std={near_52w_high.std():.4f}")

        logger.info("   🔄 Computing reversal_5d (5-day mean reversion)...")
        close_5d_ago = data.groupby('ticker')['Close'].transform(lambda x: x.shift(5))
        reversal_5d = -(((data['Close'] - close_5d_ago) / close_5d_ago)).fillna(0)
        logger.info(f"   ✅ reversal_5d: mean={reversal_5d.mean():.4f}, std={reversal_5d.std():.4f}")

        logger.info("   🔄 Computing rel_volume_spike (volume anomaly)...")
        volume_max_20 = data.groupby('ticker')['Volume'].transform(lambda x: x.rolling(20, min_periods=1).max())
        rel_volume_spike = (data['Volume'] / volume_max_20.clip(lower=1)).fillna(0)
        logger.info(f"   ✅ rel_volume_spike: mean={rel_volume_spike.mean():.4f}, std={rel_volume_spike.std():.4f}")

        logger.info("   🔄 Computing mom_accel_10_5 (momentum acceleration)...")
        close_10d_ago = data.groupby('ticker')['Close'].transform(lambda x: x.shift(10))
        close_5d_ago = data.groupby('ticker')['Close'].transform(lambda x: x.shift(5))
        mom_10 = (data['Close'] - close_10d_ago) / close_10d_ago
        mom_5 = (data['Close'] - close_5d_ago) / close_5d_ago
        mom_accel_10_5 = (mom_5 - mom_10).fillna(0)
        logger.info(f"   ✅ mom_accel_10_5: mean={mom_accel_10_5.mean():.4f}, std={mom_accel_10_5.std():.4f}")

        logger.info("   ✅ New alpha factors computation completed")

        return pd.DataFrame({
            'near_52w_high': near_52w_high,
            'reversal_5d': reversal_5d,
            'rel_volume_spike': rel_volume_spike,
            'mom_accel_10_5': mom_accel_10_5
        }, index=data.index)

    def _compute_mean_reversion_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute mean reversion factors: rsi, bollinger_squeeze (REMOVED: price_to_ma20, bollinger_position)"""

        # RSI - collect results as arrays to avoid index issues
        rsi_values = []
        bollinger_squeeze_values = []

        for ticker, group in data.groupby('ticker'):
            closes = group['Close'].values

            # RSI computation
            deltas = np.diff(np.concatenate([[closes[0]], closes]))
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            gain_avg = pd.Series(gains).rolling(14, min_periods=1).mean().values
            loss_avg = pd.Series(losses).rolling(14, min_periods=1).mean().values

            rs = gain_avg / (loss_avg + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            rsi_normalized = (rsi - 50) / 50
            rsi_values.extend(rsi_normalized)

            # Bollinger Squeeze computation
            ma20 = pd.Series(closes).rolling(20, min_periods=1).mean().values
            std20 = pd.Series(closes).rolling(20, min_periods=1).std().fillna(0).values
            bb_squeeze = std20 / (ma20 + 1e-10)
            bollinger_squeeze_values.extend(bb_squeeze)

        return pd.DataFrame({
            'rsi': rsi_values,
            'bollinger_squeeze': bollinger_squeeze_values
        }, index=data.index)
    
    def _compute_volume_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute volume factors: obv_momentum (removed ad_line - redundant)"""

        obv_momentum_values = []

        for ticker, group in data.groupby('ticker'):
            closes = group['Close'].values
            volumes = group['Volume'].values

            # OBV momentum computation
            price_changes = np.diff(np.concatenate([[closes[0]], closes]))
            directions = np.sign(price_changes)
            obv = np.cumsum(directions * volumes)
            obv_series = pd.Series(obv)
            obv_momentum = obv_series.pct_change(10).fillna(0).values
            obv_momentum_values.extend(obv_momentum)

        return pd.DataFrame({
            'obv_momentum': obv_momentum_values
        }, index=data.index)
    
    def _compute_volatility_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute volatility factors: atr_ratio (removed atr_20d - redundant)"""

        atr_ratio_values = []

        for ticker, group in data.groupby('ticker'):
            closes = group['Close'].values
            highs = group['High'].values
            lows = group['Low'].values

            # True Range computation
            high_low = highs - lows
            high_close = np.abs(highs[1:] - closes[:-1])
            low_close = np.abs(lows[1:] - closes[:-1])

            # Add first value (no previous close)
            true_range = np.concatenate([[high_low[0]], np.maximum(high_low[1:], np.maximum(high_close, low_close))])

            # ATR calculations
            tr_series = pd.Series(true_range)
            atr_20d = tr_series.rolling(20, min_periods=1).mean().values
            atr_5d = tr_series.rolling(5, min_periods=1).mean().values

            atr_ratio = (atr_5d / (atr_20d + 1e-10) - 1)

            atr_ratio_values.extend(atr_ratio)

        return pd.DataFrame({
            'atr_ratio': atr_ratio_values
        }, index=data.index)
    
    def _compute_fundamental_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute fundamental proxy factors (REDUCED: removed growth_proxy, profitability_momentum, growth_acceleration, value_proxy, profitability_proxy, ivol_60d, quality_proxy, mfi, financial_resilience)"""

        liquidity_factor_values = []

        for ticker, group in data.groupby('ticker'):
            closes = group['Close'].values
            volumes = group['Volume'].values

            # Liquidity factor
            close_series = pd.Series(closes)
            vol_series = pd.Series(volumes)
            vol_ma20 = vol_series.rolling(20, min_periods=1).mean().values
            liquidity_factor = (volumes / (vol_ma20 + 1e-10) - 1)
            liquidity_factor_values.extend(liquidity_factor)

        return pd.DataFrame({
            'liquidity_factor': liquidity_factor_values
        }, index=data.index)

    def _compute_ivol_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute IVOL (Idiosyncratic Volatility) factor using 60-day rolling regression

        IVOL 特质波动率 (T+10 适用):
        使用60日滚动回归计算相对于SPY的特异性波动率

        步骤:
        1. 计算对数收益: r_i,t = ln(c_t/c_t-1), r_m,t 使用SPY
        2. 60日滚动回归: r_i,t = α + β*r_m,t + ε_i,t
        3. IVOL_60d = sqrt(1/(N-1) * Σ(ε_i,t-k)^2) for k=1 to 60
        4. 每日横截面 winsorize → z-score
        """
        try:
            window = 60
            min_periods = 30

            # Group by ticker for processing
            grouped = data.groupby('ticker')
            ivol_values = []

            # 计算市场基准收益 (使用市场平均作为SPY proxy)
            market_close = data.groupby('date')['Close'].mean()
            market_returns = np.log(market_close / market_close.shift(1)).fillna(0)

            for ticker, ticker_data in grouped:
                ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
                n_obs = len(ticker_data)

                if n_obs < min_periods:
                    ivol_values.extend([0.0] * n_obs)
                    continue

                # 计算股票对数收益
                close_prices = ticker_data['Close'].values
                log_returns = np.log(close_prices[1:] / close_prices[:-1])
                log_returns = np.concatenate([[0], log_returns])  # 第一天设为0

                # 获取对应日期的市场收益
                ticker_dates = ticker_data['date'].values
                market_log_returns = []
                for date in ticker_dates:
                    if date in market_returns.index:
                        market_log_returns.append(market_returns[date])
                    else:
                        market_log_returns.append(0.0)
                market_log_returns = np.array(market_log_returns)

                # 计算滚动IVOL
                ivol_result = np.full(n_obs, 0.0)

                for i in range(window, n_obs):
                    start_idx = i - window

                    # 获取窗口数据
                    y = log_returns[start_idx:i]
                    x = market_log_returns[start_idx:i]

                    # 去除NaN和无限值
                    valid_mask = ~(np.isnan(y) | np.isnan(x) | np.isinf(y) | np.isinf(x))
                    if valid_mask.sum() < min_periods:
                        continue

                    y_clean = y[valid_mask]
                    x_clean = x[valid_mask]

                    try:
                        # 简化CAPM回归: r_i = α + β*r_m + ε
                        X = np.column_stack([np.ones(len(x_clean)), x_clean])
                        beta_coef = np.linalg.lstsq(X, y_clean, rcond=None)[0]

                        # 计算残差
                        predicted = X @ beta_coef
                        residuals = y_clean - predicted

                        # IVOL = 残差标准差
                        if len(residuals) >= min_periods:
                            ivol_std = np.std(residuals, ddof=1)
                            # 负号：低波动率更好
                            ivol_result[i] = -ivol_std

                    except (np.linalg.LinAlgError, ValueError):
                        # 回归失败时使用简单收益标准差
                        ivol_result[i] = -np.std(y_clean, ddof=1) if len(y_clean) > 1 else 0.0

                # 应用指数移动平均平滑
                alpha = 2.0 / (20 + 1)  # 20日EMA
                for i in range(1, len(ivol_result)):
                    if ivol_result[i] != 0:
                        if ivol_result[i-1] != 0:
                            ivol_result[i] = alpha * ivol_result[i] + (1 - alpha) * ivol_result[i-1]

                ivol_values.extend(ivol_result)

            return pd.DataFrame({'ivol_60d': ivol_values}, index=data.index)

        except Exception as e:
            logger.warning(f"IVOL computation failed: {e}")
            return pd.DataFrame({'ivol_60d': np.zeros(len(data))}, index=data.index)

    def _compute_behavioral_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
        """Compute behavioral factors capturing market microstructure effects"""
        try:
            all_results = []

            for ticker, ticker_data in grouped:
                ticker_data = ticker_data.sort_values('date').reset_index(drop=True)

                # Required columns
                if not all(col in ticker_data.columns for col in ['Open', 'Close', 'High', 'Low']):
                    logger.warning(f"Missing OHLC data for {ticker}")
                    n_obs = len(ticker_data)
                    result_df = pd.DataFrame({
                        'overnight_intraday_gap': np.zeros(n_obs),
                        'max_lottery_factor': np.zeros(n_obs),
                        'streak_reversal': np.zeros(n_obs)
                    }, index=ticker_data.index)
                    all_results.append(result_df)
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

                # 3) Return Streak Reversal (relative to market)
                # Simple market proxy: use overall market average return if available
                # For now, use absolute returns (can enhance with market data later)
                excess = r_close.fillna(0)  # Simplified - can add market adjustment

                # Compute consecutive streak lengths
                streak = np.zeros(len(excess))
                run = 0
                for i, val in enumerate(excess.values):
                    if val > 0:
                        run = run + 1 if run >= 0 else 1
                    elif val < 0:
                        run = run - 1 if run <= 0 else -1
                    else:
                        run = 0
                    streak[i] = run

                # Reversal signal: negative of streak (longer streaks more likely to reverse)
                streak_reversal = -pd.Series(streak, index=ticker_data.index)

                # Clean and handle edge cases
                gap = gap.replace([np.inf, -np.inf], 0).fillna(0)
                max_factor = max_factor.replace([np.inf, -np.inf], 0).fillna(0)
                streak_reversal = streak_reversal.replace([np.inf, -np.inf], 0).fillna(0)

                result_df = pd.DataFrame({
                    'overnight_intraday_gap': gap,
                    'max_lottery_factor': max_factor,
                    'streak_reversal': streak_reversal
                }, index=ticker_data.index)

                all_results.append(result_df)

            return pd.concat(all_results, ignore_index=True)

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