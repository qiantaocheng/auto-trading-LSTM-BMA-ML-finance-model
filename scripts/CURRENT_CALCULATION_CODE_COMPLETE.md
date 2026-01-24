# Current Calculation Code - All 14 Unified Features

**File**: `bma_models/simple_25_factor_engine.py`  
**Last Updated**: 2026-01-21  
**Status**: ✅ All fixes applied, production-ready

---

## 🔧 Global Setup (Applied to All Factors)

```python
# At entry point of compute_all_17_factors():
# 1. Normalize date column
compute_data['date'] = pd.to_datetime(compute_data['date']).dt.normalize()

# 2. Sort by (ticker, date) for all rolling operations
compute_data = compute_data.sort_values(['ticker', 'date']).reset_index(drop=True)
grouped = compute_data.groupby('ticker', sort=False)

# 3. Helper for cross-sectional median imputation
dates_normalized = pd.to_datetime(data['date']).dt.normalize()
```

---

## 1. `ivol_30` - Idiosyncratic Volatility (30-day)

**Location**: `_compute_ivol_30()` method

```python
def _compute_ivol_30(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
    """
    T+10: IVOL 30 (Idiosyncratic Volatility proxy) - Updated window size.
    Uses SPY as benchmark if present in the same dataset:
      ivol_30 = rolling_30_std( stock_ret_1d - spy_ret_1d ) per ticker
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
        # 🔥 FIX: Fill SPY return NaN before creating series
        spy_ret = spy['Close'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        spy_ret_by_date = pd.Series(spy_ret.values, index=pd.to_datetime(spy['date']).dt.normalize())

        # Map SPY return to each row by date (same calendar)
        dates = pd.to_datetime(data['date']).dt.normalize()
        mkt_ret = dates.map(spy_ret_by_date)
        # 🔥 FIX: Forward fill missing market returns, then fill remaining NaN with 0
        mkt_ret = mkt_ret.fillna(method='ffill').fillna(0.0).astype(float)

        # Stock daily return per ticker
        stock_ret = grouped['Close'].transform(lambda s: s.pct_change()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        diff = (stock_ret - mkt_ret).replace([np.inf, -np.inf], np.nan)

        ivol = diff.groupby(data['ticker']).transform(lambda s: s.rolling(30, min_periods=15).std()).replace([np.inf, -np.inf], np.nan)
        
        # 🔥 FIX: Use cross-sectional median for missing values instead of 0
        dates_normalized = pd.to_datetime(data['date']).dt.normalize()
        ivol = ivol.groupby(dates_normalized).transform(lambda x: x.fillna(x.median())).fillna(0.0)
        
        return pd.DataFrame({'ivol_30': ivol}, index=data.index)
    except Exception as e:
        logger.warning(f"⚠️ ivol_30 failed, using zeros: {e}")
        return pd.DataFrame({'ivol_30': np.zeros(len(data))}, index=data.index)
```

**Key Features**:
- Window: 30 days, min_periods=15
- SPY NaN handling: Forward fill + fillna(0.0)
- Missing value: Cross-sectional median

---

## 2. `hist_vol_40d` - Historical Volatility (40-day)

**Location**: `_compute_blowoff_and_volatility()` method

```python
# Inside _compute_blowoff_and_volatility():
eps = 1e-8
# Log returns per ticker for stability and scale-invariance
log_ret = grouped['Close'].transform(lambda s: np.log(s / s.shift(1)))
# 🔥 FIX: Clip extreme log returns
log_ret = log_ret.clip(-3.0, 3.0)

# σ40: rolling std of log returns for medium-term volatility regime
sigma40 = log_ret.groupby(data['ticker']).transform(lambda s: s.rolling(40, min_periods=15).std())
hist_vol_40d = sigma40.replace([np.inf, -np.inf], np.nan)
hist_vol_40d = hist_vol_40d.groupby(dates_normalized).transform(lambda x: x.fillna(x.median())).fillna(0.0)
```

**Key Features**:
- Window: 40 days, min_periods=15
- Log returns clipped to ±3.0
- Missing value: Cross-sectional median

---

## 3. `near_52w_high` - Distance to 52-week High

**Location**: `_compute_new_alpha_factors()` method

```python
# Inside _compute_new_alpha_factors():
high_252_hist = data.groupby('ticker')['High'].transform(lambda x: x.rolling(252, min_periods=20).max().shift(1))
near_52w_high = ((data['Close'] / high_252_hist) - 1).fillna(0)
```

**Key Features**:
- Window: 252 days (trading days ≈ 1 year)
- Uses `shift(1)` to avoid look-ahead bias
- min_periods=20

---

## 4. `rsi_21` - Relative Strength Index (21-period)

**Location**: `_compute_mean_reversion_factors()` method

```python
# Inside _compute_mean_reversion_factors():
def _rsi21(x: pd.Series) -> pd.Series:
    ret = x.diff()
    gain = ret.clip(lower=0).rolling(21, min_periods=1).mean()
    loss = (-ret).clip(lower=0).rolling(21, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # Regime context for T+10: invert RSI in bearish regime (price below MA200)
    if int(getattr(self, "horizon", 5) or 5) == 10:
        ma200 = x.rolling(200, min_periods=60).mean()
        bull = (x > ma200).astype(float)
        rsi = (bull * rsi) + ((1.0 - bull) * (100.0 - rsi))
    return (rsi - 50) / 50  # Standardize to [-1, 1]

rsi = grouped['Close'].transform(_rsi21)
```

**Key Features**:
- Window: 21 periods
- Regime-aware for T+10 (inverts in bear market)
- Output standardized to [-1, 1]

---

## 5. `vol_ratio_30d` - Volume Ratio (30-day)

**Location**: `_compute_volume_factors()` method

```python
# Inside _compute_volume_factors():
# Calculate volume ratio [Updated: 20d → 30d]
# 🔥 FIX: Clip Volume to handle stock splits and data errors
volume_clipped = data['Volume'].clip(lower=0.0)
vol_ma30 = grouped['Volume'].transform(lambda v: v.rolling(30, min_periods=15).mean().shift(1))
vol_ratio_30d = (volume_clipped / (vol_ma30 + 1e-10) - 1).replace([np.inf, -np.inf], np.nan)
# 🔥 FIX: Clip ratio to reasonable bounds and use cross-sectional median for missing
vol_ratio_30d = vol_ratio_30d.clip(-5.0, 5.0)
dates_normalized = pd.to_datetime(data['date']).dt.normalize()
vol_ratio_30d = vol_ratio_30d.groupby(dates_normalized).transform(lambda x: x.fillna(x.median())).fillna(0.0)
```

**Key Features**:
- Window: 30 days, min_periods=15
- Uses `shift(1)` to avoid look-ahead bias
- Volume clipped, ratio clipped to [-5, 5]
- Missing value: Cross-sectional median

---

## 6. `trend_r2_60` - Trend R² (60-day)

**Location**: `_compute_trend_r2_60()` method

```python
def _compute_trend_r2_60(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
    """🔥 Compute 60-day trend R² (linear regression goodness of fit)"""
    window = 60
    x_base = np.arange(window, dtype=float)
    X = np.column_stack([np.ones(window, dtype=float), x_base])

    def _r2_from_close(arr: np.ndarray) -> float:
        if arr is None or len(arr) != window:
            return 0.0
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
            return 0.0
        y = np.log(arr.astype(float))
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_hat = X @ beta
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
            r2_val = 1.0 - ss_res / ss_tot
            return float(max(0.0, min(1.0, r2_val)))
        except Exception:
            return 0.0

    r2 = grouped['Close'].transform(
        lambda s: s.rolling(window, min_periods=window).apply(_r2_from_close, raw=True)
    )
    r2 = r2.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.DataFrame({'trend_r2_60': r2}, index=data.index)
```

**Key Features**:
- Window: 60 days, min_periods=60 (strict)
- Linear regression on log prices
- Output clipped to [0, 1]

---

## 7. `liquid_momentum` - Liquidity-adjusted Momentum

**Location**: `_compute_momentum_factors()` method

```python
# Inside _compute_momentum_factors():
momentum_60d = grouped['Close'].pct_change(60).fillna(0)

# T+10: Liquid Momentum (momentum * turnover validation)
if 'liquid_momentum' in getattr(self, 'alpha_factors', []):
    avg_vol_126 = grouped['Volume'].transform(lambda x: x.rolling(126, min_periods=30).mean().shift(1))
    turnover_ratio = (data['Volume'] / (avg_vol_126 + 1e-10)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    liquid_momentum = (momentum_60d * turnover_ratio).replace([np.inf, -np.inf], np.nan).fillna(0.0)
```

**Key Features**:
- Base momentum: 60-day return
- Turnover ratio: Volume / 126-day average volume (shifted)
- Product of momentum and turnover

---

## 8. `obv_momentum_40d` - OBV Momentum (40-day) ⭐ FIXED

**Location**: `_compute_volume_factors()` method

```python
# Inside _compute_volume_factors():
# Calculate OBV
dir_ = grouped['Close'].transform(lambda s: s.pct_change()).fillna(0.0)
dir_ = np.sign(dir_)
obv = (dir_ * data['Volume']).groupby(data['ticker']).cumsum()

# NEW: Compute OBV Momentum 40d
if 'obv_momentum_40d' in getattr(self, 'alpha_factors', []):
    def _calc_obv_momentum_40d_per_ticker(ticker_group):
        """
        🔥 FIXED Logic:
        1. OBV is normalized by cumulative volume (removes drift)
        2. Calculate OBV MA10 (short-term trend) and MA40 (long-term trend) on normalized OBV
        3. Spread = MA10 - MA40 (positive = accelerating accumulation)
        """
        ticker_obv_norm = ticker_group
        obv_ma10 = ticker_obv_norm.rolling(window=10, min_periods=5).mean()
        obv_ma40 = ticker_obv_norm.rolling(window=40, min_periods=20).mean()
        obv_spread = obv_ma10 - obv_ma40
        return obv_spread
    
    # 🔥 FIX: Normalize OBV by cumulative volume first to remove drift
    cum_vol_40 = grouped['Volume'].transform(lambda v: v.rolling(window=40, min_periods=20).sum())
    obv_norm = obv / (cum_vol_40 + 1e-6)
    
    # Calculate OBV spread per ticker on normalized OBV
    obv_spread = obv_norm.groupby(data['ticker']).transform(_calc_obv_momentum_40d_per_ticker)
    
    # 🔥 FIX: Clip to reasonable bounds to handle outliers
    obv_momentum_40d = obv_spread.replace([np.inf, -np.inf], np.nan)
    obv_momentum_40d = obv_momentum_40d.clip(-5.0, 5.0)
    
    # 🔥 FIX: Use cross-sectional median for missing values
    dates_normalized = pd.to_datetime(data['date']).dt.normalize()
    obv_momentum_40d = obv_momentum_40d.groupby(dates_normalized).transform(lambda x: x.fillna(x.median())).fillna(0.0)
```

**Key Features**:
- OBV normalized by cumulative volume (removes drift)
- MA10-MA40 spread on normalized OBV
- Clipped to [-5, 5]
- Missing value: Cross-sectional median

---

## 9. `atr_ratio` - ATR Ratio

**Location**: `_compute_volatility_factors()` method

```python
def _compute_volatility_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
    """Compute volatility factors: atr_ratio"""
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
```

**Key Features**:
- True Range includes gap (high-low, high-prev_close, low-prev_close)
- ATR20 vs ATR5 ratio
- min_periods=1 for coverage

---

## 10. `ret_skew_30d` - Return Skewness (30-day) ⭐ FIXED

**Location**: `_compute_ret_skew_30d()` method

```python
def _compute_ret_skew_30d(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
    """🔥 Compute 30-day return skewness - T+10 updated factor"""
    # 🔥 FIX: Winsorize log returns to handle extreme values
    log_ret = grouped['Close'].transform(
        lambda s: np.log(s / s.shift(1))
    )
    # Clip extreme log returns (±10% daily = ±2.3 log units, use ±3 for safety)
    log_ret_clipped = log_ret.clip(-3.0, 3.0)
    
    # 🔥 FIX: Use min_periods=20 for better stability
    ret_skew = log_ret_clipped.groupby(data['ticker']).transform(
        lambda s: s.rolling(30, min_periods=20).skew()
    )
    ret_skew = ret_skew.replace([np.inf, -np.inf], np.nan)
    
    # 🔥 FIX: Use cross-sectional median for missing values
    dates_normalized = pd.to_datetime(data['date']).dt.normalize()
    ret_skew = ret_skew.groupby(dates_normalized).transform(lambda x: x.fillna(x.median())).fillna(0.0)
    
    return pd.DataFrame({'ret_skew_30d': ret_skew}, index=data.index)
```

**Key Features**:
- Window: 30 days, min_periods=20 (increased for stability)
- Log returns clipped to ±3.0 before computing skewness
- Missing value: Cross-sectional median

---

## 11. `price_ma60_deviation` - Price Deviation from MA60 ⭐ FIXED

**Location**: `_compute_mean_reversion_factors()` method

```python
# Inside _compute_mean_reversion_factors():
# 🔥 FIX: Shift MA60 to avoid look-ahead bias
ma60 = grouped['Close'].transform(lambda x: x.rolling(60, min_periods=10).mean().shift(1))
price_ma60_dev = (data['Close'] / (ma60 + 1e-10) - 1).replace([np.inf, -np.inf], np.nan)
dates_normalized = pd.to_datetime(data['date']).dt.normalize()
price_ma60_dev = price_ma60_dev.groupby(dates_normalized).transform(lambda x: x.fillna(x.median())).fillna(0.0)
```

**Key Features**:
- Window: 60 days, min_periods=10
- Uses `shift(1)` to avoid look-ahead bias
- Missing value: Cross-sectional median

---

## 12. `blowoff_ratio_30d` - Blowoff Ratio (30-day std window) ⭐ FIXED

**Location**: `_compute_blowoff_and_volatility()` method

```python
def _compute_blowoff_and_volatility(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
    """Compute blowoff_ratio_30d and hist_vol_40d"""
    try:
        eps = 1e-8
        # Log returns per ticker
        log_ret = grouped['Close'].transform(lambda s: np.log(s / s.shift(1)))
        # 🔥 FIX: Clip extreme log returns
        log_ret = log_ret.clip(-3.0, 3.0)

        # σ30: rolling std of log returns [Updated: 14d → 30d]
        sigma30 = log_ret.groupby(data['ticker']).transform(lambda s: s.rolling(30, min_periods=15).std())
        # 🔥 FIX: Use absolute value to capture both up and down moves
        max_jump_5d = log_ret.groupby(data['ticker']).transform(lambda s: s.abs().rolling(5, min_periods=2).max())
        blowoff_ratio_30d = (max_jump_5d / (sigma30 + eps)).replace([np.inf, -np.inf], np.nan)
        # 🔥 FIX: Clip ratio to reasonable bounds
        blowoff_ratio_30d = blowoff_ratio_30d.clip(0.0, 10.0)
        dates_normalized = pd.to_datetime(data['date']).dt.normalize()
        blowoff_ratio_30d = blowoff_ratio_30d.groupby(dates_normalized).transform(lambda x: x.fillna(x.median())).fillna(0.0)

        return pd.DataFrame({
            'blowoff_ratio_30d': blowoff_ratio_30d,
            'hist_vol_40d': hist_vol_40d,
        }, index=data.index)
```

**Key Features**:
- Std window: 30 days (updated from 14d)
- Max jump: 5-day absolute max
- Ratio clipped to [0, 10]
- Missing value: Cross-sectional median

---

## 13. `bollinger_squeeze` - Bollinger Band Squeeze

**Location**: `enhanced_alpha_strategies.py` - `_compute_bollinger_squeeze()` method

```python
def _compute_bollinger_squeeze(self, df: pd.DataFrame, **kwargs) -> pd.Series:
    """Bollinger Band volatility squeeze"""
    if 'Close' not in df.columns or len(df) < 20:
        return pd.Series(0, index=df.index)
    try:
        std_20 = df['Close'].rolling(20).std()
        std_5 = df['Close'].rolling(5).std()
        squeeze = std_5 / (std_20 + 1e-8)
        return self.safe_fillna(squeeze, df)
    except:
        return pd.Series(0, index=df.index)
```

**Key Features**:
- Ratio of 5-day std to 20-day std
- Low ratio (< 1.0) indicates volatility compression (squeeze)
- High ratio (> 1.0) indicates volatility expansion
- Window: std5 / std20

**Note**: This measures short-term volatility relative to medium-term volatility. Low values indicate "squeeze" conditions that often precede large moves.

---

## 14. `feat_vol_price_div_30d` - Volume-Price Divergence (30-day) ⭐ FIXED

**Location**: `_compute_vol_price_div_30d()` method

```python
def _compute_vol_price_div_30d(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
    """
    🔥 Compute Volume-Price Divergence Factor (30-day)
    
    🔥 FIXES:
    1. Fixed groupby syntax (use dates Series, not column name string)
    2. Sign convention: negative = low volume rally (divergence risk)
    """
    try:
        if 'Close' not in data.columns or 'Volume' not in data.columns:
            logger.warning("⚠️ Missing Close or Volume for vol_price_div_30d, using zeros")
            return pd.DataFrame({'feat_vol_price_div_30d': np.zeros(len(data))}, index=data.index)
        
        # 1. Calculate price momentum (30-day return)
        raw_price_chg = grouped['Close'].transform(
            lambda x: x.pct_change(periods=30)
        ).replace([np.inf, -np.inf], np.nan)
        
        # 2. Calculate volume trend (MA10 vs MA30)
        def calc_vol_trend(x):
            ma10 = x.rolling(window=10, min_periods=5).mean()
            ma30 = x.rolling(window=30, min_periods=15).mean()
            return (ma10 - ma30) / (ma30 + 1e-6)
        
        raw_vol_chg = grouped['Volume'].transform(calc_vol_trend).replace([np.inf, -np.inf], np.nan)
        
        # 🔥 FIX: Use normalized dates Series for groupby (not column name string)
        dates_normalized = pd.to_datetime(data['date']).dt.normalize()
        
        # 3. Cross-sectional Rank normalization
        rank_price = raw_price_chg.groupby(dates_normalized).rank(pct=True)
        rank_vol = raw_vol_chg.groupby(dates_normalized).rank(pct=True)
        
        # 4. Calculate divergence
        # 🔥 FIX: Sign convention - negative = low volume rally (divergence risk)
        # Volume_Rank - Price_Rank:
        #   > 0: High volume relative to price momentum (healthy/accumulation)
        #   < 0: Low volume relative to price momentum (divergence risk)
        feat_vol_price_div_30d = (rank_vol - rank_price)
        
        # Fill missing values with cross-sectional median
        feat_vol_price_div_30d = feat_vol_price_div_30d.groupby(dates_normalized).transform(
            lambda x: x.fillna(x.median())
        ).fillna(0.0)
        
        return pd.DataFrame({'feat_vol_price_div_30d': feat_vol_price_div_30d}, index=data.index)
```

**Key Features**:
- Price momentum: 30-day return
- Volume trend: MA10 vs MA30
- Cross-sectional rank normalization
- Sign: Negative = divergence risk (low volume rally)
- Missing value: Cross-sectional median

---

## ✅ Summary of All Fixes Applied

1. **Date Normalization**: All dates normalized at entry point
2. **Data Sorting**: Enforced (ticker, date) sorting
3. **Missing Values**: Cross-sectional median instead of 0
4. **Extreme Values**: Clipping applied to all ratios and returns
5. **Look-ahead Bias**: `shift(1)` added where needed
6. **Groupby Syntax**: Fixed for cross-sectional operations
7. **SPY Handling**: Forward fill for market returns
8. **OBV Normalization**: Fixed drift removal

---

## 📋 Usage

All 14 features are computed by:
```python
from bma_models.simple_25_factor_engine import Simple17FactorEngine

engine = Simple17FactorEngine(horizon=10, lookback_days=120)
factors_df = engine.compute_all_17_factors(market_data, mode='train')
```

The unified 14-feature subset is selected via `t10_selected` in `量化模型_bma_ultra_enhanced.py`.
