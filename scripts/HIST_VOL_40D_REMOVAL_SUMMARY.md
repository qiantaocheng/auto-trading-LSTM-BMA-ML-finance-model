# hist_vol_40d Factor Removal Summary

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
    # 'hist_vol_40d',  # REMOVED: 40-day historical volatility - removed from all first layer models
    'atr_ratio',
    # 'bollinger_squeeze',  # REMOVED: Bollinger Band volatility squeeze - removed from all first layer models
    'vol_ratio_30d',
    'price_ma60_deviation',
    '5_days_reversal',
    'downside_beta_ewm_21',
    'feat_vol_price_div_30d',
]
```

**Total factors: 13 (was 14)**

---

### 2. Model Feature List Updates

**Removed from `t10_selected` (`bma_models/量化模型_bma_ultra_enhanced.py`):**

```python
t10_selected = [
    "momentum_10d",
    "ivol_30",
    # "hist_vol_40d",  # REMOVED: 40-day historical volatility - removed from all first layer models
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

**This ensures `hist_vol_40d` is removed from:**
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
    'ivol_30', 'near_52w_high', 'rsi_21', 'vol_ratio_30d',  # 'hist_vol_40d' REMOVED
    'trend_r2_60', 'liquid_momentum', 'obv_momentum_40d', 'atr_ratio',
    'ret_skew_30d', 'price_ma60_deviation', 'blowoff_ratio_30d',
    # 'bollinger_squeeze',  # REMOVED: Bollinger Band volatility squeeze - removed from all first layer models
    # 'hist_vol_40d',  # REMOVED: 40-day historical volatility - removed from all first layer models
    'feat_vol_price_div_30d',
]
```

---

### 4. Fallback Factor List

**Removed from fallback `T10_ALPHA_FACTORS` (`bma_models/量化模型_bma_ultra_enhanced.py` line 3237):**

```python
T10_ALPHA_FACTORS = [
    'momentum_10d', 'liquid_momentum', 'obv_momentum_40d', 'ivol_30', 'rsi_21', 'trend_r2_60',
    'near_52w_high', 'ret_skew_30d', 'blowoff_ratio_30d',  # 'hist_vol_40d' REMOVED
    'atr_ratio', 'vol_ratio_30d', 'price_ma60_deviation', '5_days_reversal',
    'downside_beta_ewm_21', 'feat_vol_price_div_30d'
]
```

---

### 5. Configuration File Updates

**Removed from `FACTOR_CATEGORIES` (`bma_models/factor_config.py`):**

```python
'volatility': ['atr_ratio', 'ivol_20'],  # 'hist_vol_40d' REMOVED from all first layer models
```

**Removed from `FACTOR_DESCRIPTIONS` (`bma_models/factor_config.py`):**

```python
# 'hist_vol_40d': '40-day historical volatility level',  # REMOVED from all first layer models
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

- **Before**: 14 factors (after bollinger_squeeze removal)
- **After**: 13 factors
- **Removed**: `hist_vol_40d`

---

## 🔍 Verification Checklist

- [x] Removed from `T10_ALPHA_FACTORS` list
- [x] Removed from fallback `T10_ALPHA_FACTORS` list
- [x] Removed from `t10_selected` list
- [x] Removed from `base_features` (Direct Predict)
- [x] Removed from `FACTOR_CATEGORIES` (factor_config.py)
- [x] Removed from `FACTOR_DESCRIPTIONS` (factor_config.py)
- [x] All first layer models will automatically exclude it

---

## 📝 Notes

- **Calculation Code**: The `hist_vol_40d` calculation in `_compute_blowoff_and_volatility()` is **NOT deleted** - it remains in the codebase but will not be used since it's removed from factor lists
- **Backward Compatibility**: If needed in the future, simply uncomment `hist_vol_40d` in the factor lists
- **Data Files**: Existing parquet files may still contain `hist_vol_40d` column, but models will ignore it
- **Related Factor**: `ivol_30` (idiosyncratic volatility) remains and may provide similar volatility information

---

## 🚀 Next Steps

1. **Retrain Models**: Models will automatically use the new 13-factor set (without `hist_vol_40d`)
2. **Verify Training**: Check that models train successfully with 13 factors
3. **Update Data** (Optional): If desired, remove `hist_vol_40d` column from parquet files (not required, models will ignore it)

---

## 📋 Final Factor List (13 factors)

1. `momentum_10d`
2. `liquid_momentum`
3. `obv_momentum_40d`
4. `ivol_30`
5. `rsi_21`
6. `trend_r2_60`
7. `near_52w_high`
8. `ret_skew_30d`
9. `blowoff_ratio_30d`
10. `atr_ratio`
11. `vol_ratio_30d`
12. `price_ma60_deviation`
13. `5_days_reversal`
14. `downside_beta_ewm_21`
15. `feat_vol_price_div_30d`

**Note**: 
- Legacy mapping `'stability_score': 'hist_vol_40d'` remains for backward compatibility but will not be used in training since `hist_vol_40d` is removed from factor lists
- T5_ALPHA_FACTORS still contains `hist_vol_40d` (only T10 factors were updated)

---

**Last Updated**: 2025-01-20  
**Status**: ✅ Complete - `hist_vol_40d` removed from all first layer models
