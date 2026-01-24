# Unified Feature Input Confirmation

## ✅ All Models Use the Same Feature Input

### Unified Feature List (15 features)

**Used by**: Training, Direct Predict, 80/20 Time Split (all first-layer models)

| # | Feature Name | Category | Status |
|---|--------------|----------|--------|
| 1 | ivol_20 | Volatility | ✅ |
| 2 | hist_vol_40d | Volatility | ✅ |
| 3 | near_52w_high | Price | ✅ |
| 4 | rsi_21 | Technical | ✅ |
| 5 | vol_ratio_20d | Momentum | ✅ |
| 6 | trend_r2_60 | Technical | ✅ |
| 7 | liquid_momentum | Momentum | ✅ |
| 8 | obv_divergence | Momentum | ✅ |
| 9 | atr_ratio | Volatility | ✅ |
| 10 | ret_skew_20d | Technical | ✅ |
| 11 | price_ma60_deviation | Price | ✅ |
| 12 | blowoff_ratio | Volatility | ✅ |
| 13 | **bollinger_squeeze** | Volatility | ✅ |
| 14 | **feat_sato_momentum_10d** | Sato Factor | ✅ |
| 15 | **feat_sato_divergence_10d** | Sato Factor | ✅ |

**Total**: 15 features

---

## 📊 Feature Usage by Component

### 1. Training Phase
**Location**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 3283-3298)

**Feature List**: `t10_selected` (15 features)

**Used by**: 
- ElasticNet
- XGBoost
- CatBoost
- LambdaRank

**Code**:
```python
t10_selected = [
    "ivol_20", "hist_vol_40d", "near_52w_high", "rsi_21", "vol_ratio_20d",
    "trend_r2_60", "liquid_momentum", "obv_divergence", "atr_ratio",
    "ret_skew_20d", "price_ma60_deviation", "blowoff_ratio",
    "bollinger_squeeze",
    "feat_sato_momentum_10d", "feat_sato_divergence_10d",
]
base_overrides = {
    'elastic_net': list(t10_selected),
    'catboost': list(t10_selected),
    'xgboost': list(t10_selected),
    'lambdarank': list(t10_selected),
}
```

---

### 2. Direct Predict Phase
**Location**: `autotrader/app.py` (line 1647-1723)

**Feature Computation**: `Simple17FactorEngine.compute_all_17_factors()`

**Feature Universe**: `T10_ALPHA_FACTORS` (17 features - includes all 15 training features + 2 extra)

**Feature Filtering**: Models filter to `t10_selected` (15 features) via `feature_names_in_`

**Process**:
1. Compute all features from `T10_ALPHA_FACTORS` (17 features)
2. Models automatically filter to their `feature_names_in_` (15 features, same as training)

**Code**:
```python
engine = Simple17FactorEngine(lookback_days=total_lookback_days, mode='predict', horizon=prediction_horizon)
all_feature_data = engine.compute_all_17_factors(market_data, mode='predict')
# Models filter to feature_names_in_ (15 features, same as training)
```

---

### 3. 80/20 Time Split Evaluation
**Location**: `scripts/time_split_80_20_oos_eval.py` (line 1777-1814)

**Feature Source**: Loaded from parquet file

**Feature Filtering**: Models filter to their `feature_names_in_` (15 features, same as training)

**Process**:
1. Load features from parquet file (includes all 15 training features)
2. Models automatically filter to their `feature_names_in_` (15 features, same as training)

**Code**:
```python
# Load data from parquet
df = pd.read_parquet(data_file)

# Models filter to feature_names_in_ (15 features, same as training)
X_aligned = align_test_features_with_model(X, models_dict['elastic_net'], ...)
```

---

### 4. Prediction Fallback (when model detection fails)
**Location**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 5352-5378)

**Feature List**: `base_features` (15 features, same as `t10_selected`)

**Used when**: Model's `feature_names_in_` cannot be detected

**Code**:
```python
base_features = [
    'ivol_20', 'hist_vol_40d', 'near_52w_high', 'rsi_21', 'vol_ratio_20d',
    'trend_r2_60', 'liquid_momentum', 'obv_divergence', 'atr_ratio',
    'ret_skew_20d', 'price_ma60_deviation', 'blowoff_ratio',
    'bollinger_squeeze',
    'feat_sato_momentum_10d', 'feat_sato_divergence_10d',
]
```

---

## ✅ Consistency Guarantees

### 1. Training → Prediction Alignment
- ✅ Models store `feature_names_in_` during training (15 features)
- ✅ Prediction uses `feature_names_in_` to filter features (15 features)
- ✅ **Result**: Prediction uses exactly the same features as training

### 2. Direct Predict → Training Alignment
- ✅ Direct Predict computes all `T10_ALPHA_FACTORS` (17 features)
- ✅ Models filter to `feature_names_in_` (15 features, same as training)
- ✅ **Result**: Direct Predict uses the same 15 features as training

### 3. 80/20 Time Split → Training Alignment
- ✅ 80/20 loads features from parquet (includes all 15 training features)
- ✅ Models filter to `feature_names_in_` (15 features, same as training)
- ✅ **Result**: 80/20 uses the same 15 features as training

### 4. All First-Layer Models
- ✅ ElasticNet: Uses `t10_selected` (15 features)
- ✅ XGBoost: Uses `t10_selected` (15 features)
- ✅ CatBoost: Uses `t10_selected` (15 features)
- ✅ LambdaRank: Uses `t10_selected` (15 features)
- ✅ **Result**: All models use the same 15 features

---

## 📋 Complete Feature List

### All 15 Features (Unified Input)

1. **ivol_20** - Implied volatility (20-day)
2. **hist_vol_40d** - Historical volatility (40-day)
3. **near_52w_high** - Distance to 52-week high
4. **rsi_21** - RSI (21-period)
5. **vol_ratio_20d** - Volume spike ratio (20-day)
6. **trend_r2_60** - Trend R-squared (60-day)
7. **liquid_momentum** - Liquidity momentum
8. **obv_divergence** - OBV divergence signal
9. **atr_ratio** - ATR intensity ratio
10. **ret_skew_20d** - Return skewness (20-day)
11. **price_ma60_deviation** - Price deviation from 60-day MA
12. **blowoff_ratio** - Blowoff ratio
13. **bollinger_squeeze** - Bollinger Band volatility squeeze ✅ (restored)
14. **feat_sato_momentum_10d** - Sato Square Root Factor (Momentum) ✅
15. **feat_sato_divergence_10d** - Sato Square Root Factor (Divergence) ✅

---

## 🔍 Feature Categories

### Volatility Factors (5)
- ivol_20
- hist_vol_40d
- atr_ratio
- blowoff_ratio
- bollinger_squeeze

### Momentum Factors (3)
- vol_ratio_20d
- liquid_momentum
- obv_divergence

### Price Factors (2)
- near_52w_high
- price_ma60_deviation

### Technical Indicators (3)
- rsi_21
- trend_r2_60
- ret_skew_20d

### Sato Factors (2)
- feat_sato_momentum_10d
- feat_sato_divergence_10d

---

## ✅ Verification Checklist

- [x] **Training**: Uses `t10_selected` (15 features)
- [x] **Direct Predict**: Computes `T10_ALPHA_FACTORS`, filters to `t10_selected` (15 features)
- [x] **80/20 Time Split**: Loads from parquet, filters to `feature_names_in_` (15 features)
- [x] **Prediction Fallback**: Uses `base_features` = `t10_selected` (15 features)
- [x] **All First-Layer Models**: Use the same `t10_selected` (15 features)
- [x] **Feature Consistency**: All components use the same 15 features

---

## 📝 Notes

1. **T10_ALPHA_FACTORS** (17 features) is the **universe** of computable features
2. **t10_selected** (15 features) is the **actual** feature set used by all models
3. **Direct Predict** computes all 17 features, but models filter to 15 features
4. **80/20 Time Split** loads all features from parquet, but models filter to 15 features
5. **All models** (ElasticNet, XGBoost, CatBoost, LambdaRank) use the same 15 features

---

## 🎯 Summary

**Unified Feature Input**: **15 features** (t10_selected)

**Used by**:
- ✅ Training (all first-layer models)
- ✅ Direct Predict (all first-layer models)
- ✅ 80/20 Time Split (all first-layer models)
- ✅ Prediction fallback (when model detection fails)

**Consistency**: ✅ **Guaranteed** - All models use the same 15 features across all phases
