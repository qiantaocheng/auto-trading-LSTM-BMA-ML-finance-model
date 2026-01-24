# Unified Features Calculation Code

## ✅ Verified Unified Feature Set (14 Features)

All models (Training, Direct Predict, 80/20 OOS) use the same 14 features:

1. `ivol_30` - Idiosyncratic Volatility (30-day)
2. `hist_vol_40d` - Historical Volatility (40-day)
3. `near_52w_high` - Distance to 52-week High
4. `rsi_21` - Relative Strength Index (21-period)
5. `vol_ratio_30d` - Volume Ratio (30-day)
6. `trend_r2_60` - Trend R² (60-day)
7. `liquid_momentum` - Liquidity-adjusted Momentum
8. `obv_momentum_40d` - OBV Momentum (40-day) ⭐ NEW
9. `atr_ratio` - ATR Ratio
10. `ret_skew_30d` - Return Skewness (30-day) ⭐ UPDATED
11. `price_ma60_deviation` - Price Deviation from MA60
12. `blowoff_ratio_30d` - Blowoff Ratio (30-day std window) ⭐ UPDATED
13. `bollinger_squeeze` - Bollinger Band Squeeze
14. `feat_vol_price_div_30d` - Volume-Price Divergence (30-day) ⭐ NEW

---

## 📝 Updated Calculation Code

### Location: `bma_models/simple_25_factor_engine.py`

### 1. `ivol_30` - Idiosyncratic Volatility (30-day)

```python
def _compute_ivol_30(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
    """
    T+10: IVOL 30 (Idiosyncratic Volatility proxy) - Updated window size.
    Uses SPY as benchmark if present in the same dataset:
      ivol_30 = rolling_30_std( stock_ret_1d - spy_ret_1d ) per ticker
    If SPY is missing, returns zeros (still keeps feature name stable).
    """
    try:
        if 'date' not in data.columns:
            raise ValueError("ivol_30 requires 'date' column in compute_data")

        # Build SPY daily return series by date if SPY exists in the universe
        spy = data[data['ticker'].astype(str).str.upper().str.strip() == 'SPY'].copy()
        if spy.empty:
            logger.warning("⚠️ ivol_30: SPY not found in data; using zeros")
            return pd.DataFrame({'ivol_30': np.zeros(len(data))}, index=data.index)

        spy = spy.sort_values('date')
        spy_ret = spy['Close'].pct_change().replace([np.inf, -np.inf], np.nan)
        spy_ret_by_date = pd.Series(spy_ret.values, index=pd.to_datetime(spy['date']).dt.normalize())

        # Map SPY return to each row by date (same calendar)
        dates = pd.to_datetime(data['date']).dt.normalize()
        mkt_ret = dates.map(spy_ret_by_date).astype(float)

        # Stock daily return per ticker
        stock_ret = grouped['Close'].transform(lambda s: s.pct_change()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        diff = (stock_ret - mkt_ret).replace([np.inf, -np.inf], np.nan)

        ivol = diff.groupby(data['ticker']).transform(lambda s: s.rolling(30, min_periods=15).std()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame({'ivol_30': ivol}, index=data.index)
    except Exception as e:
        logger.warning(f"⚠️ ivol_30 failed, using zeros: {e}")
        return pd.DataFrame({'ivol_30': np.zeros(len(data))}, index=data.index)
```

**Key Changes**: Window updated from 20 to 30 days, min_periods=15

---

### 2. `ret_skew_30d` - Return Skewness (30-day)

```python
def _compute_ret_skew_30d(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
    """
    🔥 Compute 30-day return skewness - T+10 updated factor
    Uses log returns for scale-invariance and sample skewness
    Updated window: 20d → 30d for better signal stability
    """
    # IMPORTANT: must preserve index alignment with `data`.
    ret_skew = grouped['Close'].transform(
        lambda s: np.log(s / s.shift(1)).rolling(30, min_periods=15).skew()
    )
    ret_skew = ret_skew.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.DataFrame({'ret_skew_30d': ret_skew}, index=data.index)
```

**Key Changes**: Window updated from 20 to 30 days, min_periods=15

---

### 3. `vol_ratio_30d` - Volume Ratio (30-day)

```python
# Inside _compute_volume_factors method:
# Calculate volume ratio [Updated: 20d → 30d]
vol_ma30 = grouped['Volume'].transform(lambda v: v.rolling(30, min_periods=15).mean().shift(1))
vol_ratio_30d = (data['Volume'] / (vol_ma30 + 1e-10) - 1).replace([np.inf, -np.inf], 0.0).fillna(0)
```

**Key Changes**: Window updated from 20 to 30 days, min_periods=15

---

### 4. `blowoff_ratio_30d` - Blowoff Ratio (30-day std window)

```python
def _compute_blowoff_and_volatility(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
    """Compute blowoff_ratio_30d and hist_vol_40d following Polygon OHLC pipeline.

    Definitions:
    - blowoff_ratio_30d = max_{t-4..t} log_return / (std_30(log_return) + eps)  [Updated: 14d → 30d]
    - hist_vol_40d = rolling 40-day standard deviation of log returns
    """
    try:
        eps = 1e-8
        # Log returns per ticker for stability and scale-invariance
        log_ret = grouped['Close'].transform(lambda s: np.log(s / s.shift(1)))

        # σ30: rolling std of log returns (min_periods tuned for robustness) [Updated: 14d → 30d]
        sigma30 = log_ret.groupby(data['ticker']).transform(lambda s: s.rolling(30, min_periods=15).std())
        # max jump over past 5 days (inclusive)
        max_jump_5d = log_ret.groupby(data['ticker']).transform(lambda s: s.rolling(5, min_periods=2).max())
        blowoff_ratio_30d = (max_jump_5d / (sigma30 + eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # σ40: rolling std of log returns for medium-term volatility regime
        sigma40 = log_ret.groupby(data['ticker']).transform(lambda s: s.rolling(40, min_periods=15).std())
        hist_vol_40d = sigma40.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return pd.DataFrame({
            'blowoff_ratio_30d': blowoff_ratio_30d,
            'hist_vol_40d': hist_vol_40d,
        }, index=data.index)
    except Exception as e:
        logger.warning(f"Blowoff/Volatility computation failed: {e}")
        return pd.DataFrame({
            'blowoff_ratio_30d': np.zeros(len(data)),
            'hist_vol_40d': np.zeros(len(data)),
        }, index=data.index)
```

**Key Changes**: Standard deviation window updated from 14d to 30d, min_periods=15

---

### 5. `obv_momentum_40d` - OBV Momentum (40-day) ⭐ NEW

```python
# Inside _compute_volume_factors method:
# NEW: Compute OBV Momentum 40d (replaces OBV Divergence for T+10 strategy)
if 'obv_momentum_40d' in getattr(self, 'alpha_factors', []):
    try:
        # Calculate OBV Momentum 40d for each ticker
        def _calc_obv_momentum_40d_per_ticker(ticker_group):
            """
            Calculate OBV Momentum (40d) for a single ticker.
            
            Logic:
            1. OBV is already calculated above (cumsum of signed volume)
            2. Calculate OBV MA10 (short-term trend) and MA40 (long-term trend)
            3. Spread = MA10 - MA40 (positive = accelerating accumulation)
            4. Normalize by 40-day average volume (makes it comparable across stocks)
            
            Returns: obv_momentum_40d (normalized momentum score)
            """
            ticker_obv = ticker_group
            
            # Calculate OBV moving averages
            obv_ma10 = ticker_obv.rolling(window=10, min_periods=5).mean()
            obv_ma40 = ticker_obv.rolling(window=40, min_periods=20).mean()
            
            # Calculate spread (short-term vs long-term trend)
            obv_spread = obv_ma10 - obv_ma40
            
            return obv_spread
        
        # Calculate OBV spread per ticker
        obv_spread = obv.groupby(data['ticker']).transform(_calc_obv_momentum_40d_per_ticker)
        
        # Calculate 40-day average volume per ticker for normalization
        avg_volume_40 = grouped['Volume'].transform(lambda v: v.rolling(window=40, min_periods=20).mean())
        
        # Normalize by 40-day average volume (critical for LambdaRank!)
        # This makes the factor comparable across stocks of different sizes
        obv_momentum_40d = (obv_spread / (avg_volume_40 + 1e-6)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        out['obv_momentum_40d'] = obv_momentum_40d
        
    except Exception as e:
        logger.warning(f"⚠️ obv_momentum_40d failed, using 0: {e}")
        out['obv_momentum_40d'] = np.zeros(len(data))
```

**Key Features**:
- Replaces `obv_divergence` (which was too early for T+10 strategy)
- Uses OBV MA10-MA40 spread normalized by 40-day average volume
- Makes factor comparable across stocks of different sizes

---

### 6. `feat_vol_price_div_30d` - Volume-Price Divergence (30-day) ⭐ NEW

```python
def _compute_vol_price_div_30d(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
    """
    🔥 Compute Volume-Price Divergence Factor (30-day)
    
    Logic:
    1. Calculate price momentum (30d Return)
    2. Calculate volume trend (MA10 vs MA30)
    3. Convert both to Rank (0~1) on cross-sectional basis
    4. Divergence = Price_Rank - Volume_Rank
    
    Returns:
        DataFrame with column:
        - feat_vol_price_div_30d: Volume-Price Divergence Factor (30-day)
    """
    try:
        # Ensure data has required columns
        if 'Close' not in data.columns or 'Volume' not in data.columns:
            logger.warning("⚠️ Missing Close or Volume for vol_price_div_30d, using zeros")
            return pd.DataFrame({'feat_vol_price_div_30d': np.zeros(len(data))}, index=data.index)
        
        # 1. Calculate price momentum (30-day return)
        raw_price_chg = grouped['Close'].transform(
            lambda x: x.pct_change(periods=30)
        )
        
        # 2. Calculate volume trend (MA10 vs MA30)
        def calc_vol_trend(x):
            ma10 = x.rolling(window=10, min_periods=5).mean()
            ma30 = x.rolling(window=30, min_periods=15).mean()
            # Avoid division by zero
            return (ma10 - ma30) / (ma30 + 1e-6)
        
        raw_vol_chg = grouped['Volume'].transform(calc_vol_trend)
        
        # 3. Cross-sectional Rank normalization
        # Determine grouping column (date)
        if 'date' in data.columns:
            groupby_col = 'date'
        else:
            # Assume date is in index first level
            groupby_col = data.index.get_level_values('date') if isinstance(data.index, pd.MultiIndex) else data['date']
        
        # Calculate ranks (0.0 to 1.0)
        rank_price = raw_price_chg.groupby(groupby_col).rank(pct=True)
        rank_vol = raw_vol_chg.groupby(groupby_col).rank(pct=True)
        
        # 4. Calculate divergence
        # > 0: Price rank higher than volume rank (low volume rally -> top divergence risk)
        # < 0: Volume rank higher than price rank (high volume stagnation/decline -> accumulation or panic)
        # ≈ 0: Price-volume alignment (healthy trend)
        feat_vol_price_div_30d = (rank_price - rank_vol).fillna(0.0)
        
        return pd.DataFrame({'feat_vol_price_div_30d': feat_vol_price_div_30d}, index=data.index)
        
    except Exception as e:
        logger.error(f"Volume-Price Divergence computation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame({
            'feat_vol_price_div_30d': np.zeros(len(data))
        }, index=data.index)
```

**Key Features**:
- Replaces Sato factors (`feat_sato_momentum_10d`, `feat_sato_divergence_10d`)
- Uses cross-sectional rank normalization for comparability
- Captures price-volume divergence signals

---

## ✅ Verification Status

All features verified:
- ✅ `t10_selected` in `量化模型_bma_ultra_enhanced.py` matches unified features
- ✅ All unified features present in `polygon_factors_all_filtered_clean.parquet`
- ✅ All old factors removed from multiindex file
- ✅ Calculation methods updated with correct window sizes

---

## 📋 Usage Locations

### Training (`train_full_dataset.py`):
- Uses `t10_selected` from `量化模型_bma_ultra_enhanced.py`
- All first-layer models (ElasticNet, XGBoost, CatBoost, LambdaRank) use same features

### Direct Predict (`app.py`):
- Uses `t10_selected` from `量化模型_bma_ultra_enhanced.py`
- Features computed via `Simple17FactorEngine`

### 80/20 Time Split (`time_split_80_20_oos_eval.py`):
- Uses `t10_selected` from `量化模型_bma_ultra_enhanced.py`
- Features loaded from `polygon_factors_all_filtered_clean.parquet`

---

## 🔄 Factor Calculation Flow

1. **Data Source**: `polygon_factors_all_filtered_clean.parquet` (MultiIndex: date, ticker)
2. **Engine**: `Simple17FactorEngine.compute_all_17_factors()`
3. **Output**: DataFrame with all 14 unified features + other factors
4. **Usage**: All models use the same 14-feature subset

---

## ✅ Summary

- **14 unified features** used across all stages
- **All calculation methods updated** with correct window sizes
- **Old factors removed** from multiindex file
- **New factors added**: `obv_momentum_40d`, `feat_vol_price_div_30d`
- **Updated factors**: `ivol_30`, `ret_skew_30d`, `vol_ratio_30d`, `blowoff_ratio_30d`
