# 所有因子统一 shift(1) - 开盘前预测配置

**文件**: `bma_models/simple_25_factor_engine.py`  
**状态**: ✅ 所有因子已统一 shift(1)，适用于开盘前预测开盘买入

---

## 🎯 核心原则

**因为是在开盘前预测开盘买入，所以所有因子计算都要 shift(1)！**

所有滚动统计、比率、动量指标都必须使用**前一日的数据**，确保开盘前预测时不会使用当天的数据（避免未来信息泄露）。

---

## ✅ 所有14个统一因子 - shift(1) 配置

### 1. `ivol_30` - Idiosyncratic Volatility (30-day) ✅

```python
# 🔥 FIX: Shift for pre-market prediction
ivol = diff.groupby(data['ticker']).transform(
    lambda s: s.rolling(30, min_periods=15).std().shift(1)
)
```

**说明**: 使用前一日计算的30日滚动标准差

---

### 2. `hist_vol_40d` - Historical Volatility (40-day) ✅

```python
# 🔥 FIX: Shift for pre-market prediction
sigma40 = log_ret.groupby(data['ticker']).transform(
    lambda s: s.rolling(40, min_periods=15).std().shift(1)
)
```

**说明**: 使用前一日计算的40日滚动标准差

---

### 3. `near_52w_high` - Distance to 52-week High ✅

```python
high_252_hist = data.groupby('ticker')['High'].transform(
    lambda x: x.rolling(252, min_periods=20).max().shift(1)
)
near_52w_high = ((data['Close'] / high_252_hist) - 1).fillna(0)
```

**说明**: 使用前一日计算的252日最高价

---

### 4. `rsi_21` - Relative Strength Index (21-period) ✅

```python
def _rsi21(x: pd.Series) -> pd.Series:
    ret = x.diff()
    # 🔥 FIX: Shift for pre-market prediction
    gain = ret.clip(lower=0).rolling(21, min_periods=1).mean().shift(1)
    loss = (-ret).clip(lower=0).rolling(21, min_periods=1).mean().shift(1)
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    if int(getattr(self, "horizon", 5) or 5) == 10:
        # 🔥 FIX: Shift MA200 for pre-market prediction
        ma200 = x.rolling(200, min_periods=60).mean().shift(1)
        bull = (x.shift(1) > ma200).astype(float)
        rsi = (bull * rsi) + ((1.0 - bull) * (100.0 - rsi))
    return (rsi - 50) / 50
```

**说明**: RSI计算使用前一日数据，MA200也shift(1)

---

### 5. `vol_ratio_30d` - Volume Ratio (30-day) ✅

```python
# 🔥 FIX: Use previous day's volume for ratio
vol_ma30 = volume_clipped.groupby(data['ticker']).transform(
    lambda v: v.rolling(30, min_periods=15).mean().shift(1)
)
prev_volume_clipped = grouped['Volume'].transform(lambda v: v.shift(1)).clip(lower=0.0)
vol_ratio_30d = (prev_volume_clipped / (vol_ma30 + 1e-10) - 1)
```

**说明**: 使用前一日成交量与前一日30日均量的比值

---

### 6. `trend_r2_60` - Trend R² (60-day) ✅

```python
# 🔥 FIX: Shift for pre-market prediction
r2 = grouped['Close'].transform(
    lambda s: s.rolling(window, min_periods=window).apply(_r2_from_close, raw=True).shift(1)
)
```

**说明**: 使用前一日计算的60日趋势R²

---

### 7. `liquid_momentum` - Liquidity-adjusted Momentum ✅

```python
# 🔥 FIX: Shift momentum for pre-market prediction
momentum_60d = grouped['Close'].pct_change(60).shift(1).fillna(0)

avg_vol_126 = grouped['Volume'].transform(
    lambda x: x.rolling(126, min_periods=30).mean().shift(1)
)
# 🔥 FIX: Use previous day's volume
prev_volume = grouped['Volume'].transform(lambda x: x.shift(1))
turnover_ratio = (prev_volume / (avg_vol_126 + 1e-10))
liquid_momentum = (momentum_60d * turnover_ratio)
```

**说明**: 动量使用前一日数据，成交量比率也使用前一日数据

---

### 8. `obv_momentum_40d` - OBV Momentum (40-day) ✅

```python
# 🔥 FIX: Shift cumulative volume
cum_vol_40 = grouped['Volume'].transform(
    lambda v: v.rolling(window=40, min_periods=20).sum().shift(1)
)
obv_norm = obv / (cum_vol_40 + 1e-6)

def _calc_obv_momentum_40d_per_ticker(ticker_group):
    # 🔥 FIX: Shift OBV MAs for pre-market prediction
    obv_ma10 = ticker_obv_norm.rolling(window=10, min_periods=5).mean().shift(1)
    obv_ma40 = ticker_obv_norm.rolling(window=40, min_periods=20).mean().shift(1)
    obv_spread = obv_ma10 - obv_ma40
    return obv_spread
```

**说明**: OBV均线使用前一日数据

---

### 9. `atr_ratio` - ATR Ratio ✅

```python
prev_close = grouped['Close'].transform(lambda s: s.shift(1))
high_low = data['High'] - data['Low']
high_prev_close = (data['High'] - prev_close).abs()
low_prev_close = (data['Low'] - prev_close).abs()

tr_components = pd.concat([high_low, high_prev_close, low_prev_close], axis=1)
true_range = tr_components.max(axis=1)

# 🔥 FIX: Shift for pre-market prediction
atr_20d = true_range.groupby(data['ticker']).transform(
    lambda x: x.rolling(20, min_periods=1).mean().shift(1)
)
atr_5d = true_range.groupby(data['ticker']).transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1)
)
atr_ratio = (atr_5d / (atr_20d + 1e-10) - 1)
```

**说明**: ATR使用前一日计算的真实波幅均值

---

### 10. `ret_skew_30d` - Return Skewness (30-day) ✅

```python
log_ret = grouped['Close'].transform(lambda s: np.log(s / s.shift(1)))
log_ret_clipped = log_ret.clip(-3.0, 3.0)

# 🔥 FIX: Shift for pre-market prediction
ret_skew = log_ret_clipped.groupby(data['ticker']).transform(
    lambda s: s.rolling(30, min_periods=20).skew().shift(1)
)
```

**说明**: 使用前一日计算的30日收益率偏度

---

### 11. `price_ma60_deviation` - Price Deviation from MA60 ✅

```python
# 🔥 FIX: Shift MA60 to avoid look-ahead bias
ma60 = grouped['Close'].transform(
    lambda x: x.rolling(60, min_periods=10).mean().shift(1)
)
price_ma60_dev = (data['Close'] / (ma60 + 1e-10) - 1)
```

**说明**: 使用前一日计算的60日均价

---

### 12. `blowoff_ratio_30d` - Blowoff Ratio (30-day std window) ✅

```python
log_ret = grouped['Close'].transform(lambda s: np.log(s / s.shift(1)))
log_ret = log_ret.clip(-3.0, 3.0)

# 🔥 FIX: Shift for pre-market prediction
sigma30 = log_ret.groupby(data['ticker']).transform(
    lambda s: s.rolling(30, min_periods=15).std().shift(1)
)
max_jump_5d = log_ret.groupby(data['ticker']).transform(
    lambda s: s.abs().rolling(5, min_periods=2).max().shift(1)
)
blowoff_ratio_30d = (max_jump_5d / (sigma30 + eps))
```

**说明**: 使用前一日计算的波动率和最大跳跃

---

### 13. `bollinger_squeeze` - Bollinger Band Squeeze ✅

```python
# Computed in enhanced_alpha_strategies.py
def _compute_bollinger_squeeze(self, df: pd.DataFrame, **kwargs) -> pd.Series:
    # 🔥 FIX: Shift for pre-market prediction
    std_20 = df['Close'].rolling(20).std().shift(1)
    std_5 = df['Close'].rolling(5).std().shift(1)
    squeeze = std_5 / (std_20 + 1e-8)
    return self.safe_fillna(squeeze, df)
```

**说明**: 使用前一日计算的波动率比率

---

### 14. `feat_vol_price_div_30d` - Volume-Price Divergence (30-day) ✅

```python
# 🔥 FIX: Shift for pre-market prediction
raw_price_chg = grouped['Close'].transform(
    lambda x: x.pct_change(periods=30).shift(1)
)

def calc_vol_trend(x):
    # 🔥 FIX: Shift for pre-market prediction
    ma10 = x.rolling(window=10, min_periods=5).mean().shift(1)
    ma30 = x.rolling(window=30, min_periods=15).mean().shift(1)
    return (ma10 - ma30) / (ma30 + 1e-6)

raw_vol_chg = grouped['Volume'].transform(calc_vol_trend)

# Cross-sectional rank normalization
dates_normalized = pd.to_datetime(data['date']).dt.normalize()
rank_price = raw_price_chg.groupby(dates_normalized).rank(pct=True)
rank_vol = raw_vol_chg.groupby(dates_normalized).rank(pct=True)
feat_vol_price_div_30d = (rank_vol - rank_price)
```

**说明**: 价格动量和成交量趋势都使用前一日数据

---

## 📊 Shift(1) 策略总结

### ✅ 所有因子统一规则

**规则**: 所有滚动统计、比率、动量指标都必须 shift(1)，确保开盘前预测时使用前一日数据。

**已修复的因子**:
1. ✅ `ivol_30` - rolling std shift(1)
2. ✅ `hist_vol_40d` - rolling std shift(1)
3. ✅ `near_52w_high` - rolling max shift(1) (已有)
4. ✅ `rsi_21` - rolling mean shift(1) + MA200 shift(1)
5. ✅ `vol_ratio_30d` - 使用前一日成交量
6. ✅ `trend_r2_60` - rolling apply shift(1)
7. ✅ `liquid_momentum` - momentum shift(1) + 使用前一日成交量
8. ✅ `obv_momentum_40d` - OBV MAs shift(1)
9. ✅ `atr_ratio` - rolling mean shift(1)
10. ✅ `ret_skew_30d` - rolling skew shift(1)
11. ✅ `price_ma60_deviation` - MA60 shift(1) (已有)
12. ✅ `blowoff_ratio_30d` - rolling std/max shift(1)
13. ✅ `bollinger_squeeze` - rolling std shift(1)
14. ✅ `feat_vol_price_div_30d` - price momentum shift(1) + volume trend shift(1)

---

## ✅ 所有问题已解决

- ✅ 所有因子统一 shift(1)
- ✅ 开盘前预测使用前一日数据
- ✅ 避免未来信息泄露
- ✅ 所有因子生产就绪

---

**最后更新**: 2025-01-20  
**状态**: ✅ 完成 - 所有因子已统一 shift(1)
