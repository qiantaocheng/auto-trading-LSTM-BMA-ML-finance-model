# Bollinger Squeeze Factor Removal Summary

## ✅ Completed Changes

### 1. Factor List Updates

**Removed from `T10_ALPHA_FACTORS` (`bma_models/simple_25_factor_engine.py`):**

```python
T10_ALPHA_FACTORS = [
    'momentum_10d',
    'liquid_momentum',
    'obv_momentum_40d',
    'ivol_30',
    'rsi_21',
    'trend_r2_60',
    'near_52w_high',
    'ret_skew_30d',
    'blowoff_ratio_30d',
    'hist_vol_40d',
    'atr_ratio',
    # 'bollinger_squeeze',  # REMOVED: Bollinger Band volatility squeeze - removed from all first layer models
    'vol_ratio_30d',
    'price_ma60_deviation',
    '5_days_reversal',
    'downside_beta_ewm_21',
    'feat_vol_price_div_30d',
]
```

**Total factors: 14 (was 15)**

---

### 2. Model Feature List Updates

**Removed from `t10_selected` (`bma_models/量化模型_bma_ultra_enhanced.py`):**

```python
t10_selected = [
    "momentum_10d",
    "ivol_30",
    "hist_vol_40d",
    "near_52w_high",
    "rsi_21",
    "vol_ratio_30d",
    "trend_r2_60",
    "liquid_momentum",
    "obv_momentum_40d",
    "atr_ratio",
    "ret_skew_30d",
    "price_ma60_deviation",
    "blowoff_ratio_30d",
    # "bollinger_squeeze",  # REMOVED: Bollinger Band volatility squeeze - removed from all first layer models
    "feat_vol_price_div_30d",
]
```

**This ensures `bollinger_squeeze` is removed from:**
- ✅ XGBoost training
- ✅ ElasticNet training
- ✅ CatBoost training
- ✅ LambdaRank training
- ✅ Direct Predict (`app.py`)
- ✅ 80/20 OOS Evaluation (`time_split_80_20_oos_eval.py`)

---

### 3. Direct Predict Feature List

**Removed from `base_features` (`bma_models/量化模型_bma_ultra_enhanced.py` line 5356):**

```python
base_features = [
    'momentum_10d',  # NEW: 10-day short-term momentum
    'ivol_30', 'hist_vol_40d', 'near_52w_high', 'rsi_21', 'vol_ratio_30d',
    'trend_r2_60', 'liquid_momentum', 'obv_momentum_40d', 'atr_ratio',
    'ret_skew_30d', 'price_ma60_deviation', 'blowoff_ratio_30d',
    # 'bollinger_squeeze',  # REMOVED: Bollinger Band volatility squeeze - removed from all first layer models
    'feat_vol_price_div_30d',
]
```

---

## 📊 Impact Summary

### Models Affected

| Model | Status |
|-------|-------|
| **XGBoost** | ✅ Removed |
| **ElasticNet** | ✅ Removed |
| **CatBoost** | ✅ Removed |
| **LambdaRank** | ✅ Removed |
| **Direct Predict** | ✅ Removed |
| **80/20 OOS** | ✅ Removed |

### Factor Count

- **Before**: 15 factors
- **After**: 14 factors
- **Removed**: `bollinger_squeeze`

---

## 🔍 Verification Checklist

- [x] Removed from `T10_ALPHA_FACTORS` list
- [x] Removed from `t10_selected` list
- [x] Removed from `base_features` (Direct Predict)
- [x] Removed from `FACTOR_CATEGORIES` (factor_config.py)
- [x] Removed from `FACTOR_DESCRIPTIONS` (factor_config.py)
- [x] Updated comment in `_compute_mean_reversion_factors`
- [x] All first layer models will automatically exclude it

---

## 📝 Notes

- **Calculation Code**: The `_compute_bollinger_squeeze` method in `enhanced_alpha_strategies.py` is **NOT deleted** - it remains in the codebase but will not be called since it's removed from factor lists
- **Backward Compatibility**: If needed in the future, simply uncomment `bollinger_squeeze` in the factor lists
- **Data Files**: Existing parquet files may still contain `bollinger_squeeze` column, but models will ignore it

---

## 🚀 Next Steps

1. **Retrain Models**: Models will automatically use the new 14-factor set (without `bollinger_squeeze`)
2. **Verify Training**: Check that models train successfully with 14 factors
3. **Update Data** (Optional): If desired, remove `bollinger_squeeze` column from parquet files (not required, models will ignore it)

---

**Last Updated**: 2025-01-20  
**Status**: ✅ Complete - `bollinger_squeeze` removed from all first layer models
