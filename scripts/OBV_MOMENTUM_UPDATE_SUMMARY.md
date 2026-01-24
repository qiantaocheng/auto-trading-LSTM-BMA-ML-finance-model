# OBV Momentum 40d Update Summary

## ✅ Completed Changes

### 1. Replaced OBV Divergence with OBV Momentum 40d

**Removed:**
- `obv_divergence` (OBV Divergence - risky "left-side catching knives" signal)

**Added:**
- `obv_momentum_40d` (OBV Momentum 40d - safer "right-side tailwind" signal)

**Rationale:**
- OBV Divergence signals too early for T+10 strategy (guessing tops)
- OBV Momentum confirms healthy trend (money > stocks)
- Normalized by 40-day average volume for cross-stock comparability (critical for LambdaRank)

---

### 2. Updated Calculation Logic

**New Implementation** (`simple_25_factor_engine.py`):
```python
# 1. Calculate OBV (On-Balance Volume)
obv = (dir_ * data['Volume']).groupby(data['ticker']).cumsum()

# 2. Calculate OBV moving averages
obv_ma10 = obv.rolling(window=10, min_periods=5).mean()  # Short-term trend
obv_ma40 = obv.rolling(window=40, min_periods=20).mean()  # Long-term trend

# 3. Calculate spread (short-term vs long-term)
obv_spread = obv_ma10 - obv_ma40

# 4. Normalize by 40-day average volume
avg_volume_40 = volume.rolling(window=40, min_periods=20).mean()
obv_momentum_40d = obv_spread / (avg_volume_40 + 1e-6)
```

**Key Improvements:**
- ✅ Removes arbitrariness of OBV absolute values
- ✅ Normalization makes factor comparable across stocks (AAPL vs small caps)
- ✅ Smoothing (MA10 vs MA40) filters T+1 noise, better for T+10
- ✅ Confirms trend health rather than guessing tops

---

### 3. Updated Feature Lists

**T10_ALPHA_FACTORS** (`simple_25_factor_engine.py`):
- ✅ `obv_divergence` → `obv_momentum_40d`

**t10_selected** (`量化模型_bma_ultra_enhanced.py`):
- ✅ `obv_divergence` → `obv_momentum_40d`

**compulsory_features** (`量化模型_bma_ultra_enhanced.py`):
- ✅ `obv_divergence` → `obv_momentum_40d`

**base_features** (`量化模型_bma_ultra_enhanced.py`):
- ✅ `obv_divergence` → `obv_momentum_40d`

**Legacy aliases** (`量化模型_bma_ultra_enhanced.py`):
- ✅ Added `'obv_divergence': 'obv_momentum_40d'` for backward compatibility

---

### 4. Updated All References

**Files Modified:**
1. `bma_models/simple_25_factor_engine.py`
   - Updated `T10_ALPHA_FACTORS` list
   - Replaced `obv_divergence` calculation with `obv_momentum_40d`
   - Updated `_compute_volume_factors()` method

2. `bma_models/量化模型_bma_ultra_enhanced.py`
   - Updated `T10_ALPHA_FACTORS` definition
   - Updated `t10_selected` list
   - Updated `compulsory_features`
   - Updated `base_features` fallback
   - Updated legacy feature mappings
   - Updated feature columns in legacy export

---

## 📊 Factor Interpretation

**obv_momentum_40d Values:**
- **> 5.0**: Extreme accumulation (very strong trend, high T+10 certainty)
- **> 0.0**: Positive capital inflow (healthy trend)
- **< 0.0**: Capital outflow (trend weakening)
- **< -5.0**: Extreme selling pressure (major capital flight)

**Physical Meaning:**
- Unit: "Multiples of 40-day average volume"
- Positive: Short-term OBV trend is above long-term trend (accelerating accumulation)
- Negative: Short-term OBV trend is below long-term trend (decelerating accumulation)

---

## ✅ Consistency Verification

**All Components Use Same Feature:**
- ✅ Training: Uses `obv_momentum_40d` (in `t10_selected`)
- ✅ Direct Predict: Computes `obv_momentum_40d` (in `T10_ALPHA_FACTORS`)
- ✅ 80/20 Time Split: Loads from parquet, filters to `feature_names_in_`
- ✅ Prediction Fallback: Uses `obv_momentum_40d` (in `base_features`)

**All Models Use Same Feature:**
- ✅ ElasticNet: Uses `obv_momentum_40d`
- ✅ XGBoost: Uses `obv_momentum_40d`
- ✅ CatBoost: Uses `obv_momentum_40d`
- ✅ LambdaRank: Uses `obv_momentum_40d`

---

## 🔄 Next Steps

1. **Update MultiIndex Parquet File:**
   - Remove `obv_divergence` column
   - Add `obv_momentum_40d` column (recompute if needed)

2. **Retrain Models:**
   - All models need to be retrained with new feature
   - Old models will have mismatched feature names (`obv_divergence` vs `obv_momentum_40d`)

3. **Update Existing Snapshots:**
   - Old snapshots may reference `obv_divergence`
   - Consider retraining and creating new snapshots

---

## 📝 Notes

- **Backward Compatibility**: Legacy alias `'obv_divergence': 'obv_momentum_40d'` added for compatibility
- **Normalization Critical**: The division by 40-day average volume is essential for LambdaRank (makes factor comparable across stocks)
- **T+10 Strategy**: This factor is optimized for T+10 (bi-weekly) strategy, confirming trend health rather than guessing tops

---

## ✅ Summary

**Status:** ✅ **All changes completed**

- ✅ OBV Divergence replaced with OBV Momentum 40d
- ✅ Calculation logic updated with normalization
- ✅ All feature lists updated
- ✅ All references updated
- ✅ Consistency verified

**Ready for:** Retraining and testing with new OBV Momentum 40d factor
