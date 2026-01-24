# Current Training and Prediction Features - Final Confirmation

## ✅ Verification Results

### 1. T10_ALPHA_FACTORS (Base Universe)
**Location**: `bma_models/simple_25_factor_engine.py` (line 58-78)

**Total**: **16 features**

| # | Feature Name | Type |
|---|--------------|------|
| 1 | liquid_momentum | Alpha Factor |
| 2 | obv_divergence | Alpha Factor |
| 3 | ivol_20 | Volatility Factor |
| 4 | rsi_21 | Momentum Factor |
| 5 | trend_r2_60 | Trend Factor |
| 6 | near_52w_high | Price Factor |
| 7 | ret_skew_20d | Distribution Factor |
| 8 | blowoff_ratio | Volatility Factor |
| 9 | hist_vol_40d | Volatility Factor |
| 10 | atr_ratio | Volatility Factor |
| 11 | vol_ratio_20d | Volume Factor |
| 12 | price_ma60_deviation | Price Factor |
| 13 | 5_days_reversal | Reversal Factor |
| 14 | downside_beta_ewm_21 | Risk Factor |
| 15 | **feat_sato_momentum_10d** | ✅ **Sato Factor** |
| 16 | **feat_sato_divergence_10d** | ✅ **Sato Factor** |

**Status**: ✅ **Both Sato factors confirmed**

---

### 2. Training Features (t10_selected - Fallback List)
**Location**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 3283-3298)

**Total**: **14 features** (optimized subset for T+10)

| # | Feature Name | Type |
|---|--------------|------|
| 1 | ivol_20 | Volatility Factor |
| 2 | hist_vol_40d | Volatility Factor |
| 3 | near_52w_high | Price Factor |
| 4 | rsi_21 | Momentum Factor |
| 5 | vol_ratio_20d | Volume Factor |
| 6 | trend_r2_60 | Trend Factor |
| 7 | liquid_momentum | Alpha Factor |
| 8 | obv_divergence | Alpha Factor |
| 9 | atr_ratio | Volatility Factor |
| 10 | ret_skew_20d | Distribution Factor |
| 11 | price_ma60_deviation | Price Factor |
| 12 | blowoff_ratio | Volatility Factor |
| 13 | **feat_sato_momentum_10d** | ✅ **Sato Factor** |
| 14 | **feat_sato_divergence_10d** | ✅ **Sato Factor** |

**Used for**: ElasticNet, CatBoost, XGBoost, LambdaRank (all first-layer models)

**Status**: ✅ **Both Sato factors confirmed**

**Note**: This is the fallback list used when `best_features_per_model.json` is not available.

---

### 3. Compulsory Features (T+10)
**Location**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 3250-3256)

**Total**: **5 features** (must be included in all models)

1. `obv_divergence`
2. `ivol_20`
3. `rsi_21`
4. `near_52w_high`
5. `trend_r2_60`

**Note**: These are automatically added if missing from model-specific feature lists.

---

### 4. Data File Status
**File**: `data/factor_exports/polygon_factors_all_filtered_clean.parquet`

- **Shape**: (4,180,394 rows, 24 columns)
- **feat_sato_momentum_10d**: ✅ Present (4,141,162 non-zero values, 99.1%)
- **feat_sato_divergence_10d**: ✅ Present (4,141,185 non-zero values, 99.1%)

**Status**: ✅ **Both Sato factors pre-computed in data file**

---

## 📊 Feature Selection Logic

### Training Phase

1. **Primary Source**: `results/t10_optimized_all_models/best_features_per_model.json` (if exists)
2. **Fallback**: `t10_selected` list (14 features including Sato)
3. **Enforcement**: Add `compulsory_features` if missing
4. **Per-Model**: All first-layer models use the same feature list

**Current Status**: Using fallback `t10_selected` list (best_features_per_model.json does not exist)

### Prediction Phase

1. **Primary**: Detect from trained model (`model.feature_names_in_`)
2. **Fallback**: Use `base_features` list (16 features including Sato)
3. **Union**: Combine detected features with base_features
4. **Alignment**: Ensure features match training-time features

---

## ✅ Confirmation Checklist

- [x] **T10_ALPHA_FACTORS**: ✅ Includes both Sato factors (16 features total)
- [x] **t10_selected (Training)**: ✅ Includes both Sato factors (14 features total)
- [x] **base_features (Prediction)**: ✅ Includes both Sato factors (16 features total)
- [x] **Data file**: ✅ Includes both Sato factors (pre-computed, 99.1% coverage)
- [x] **Feature consistency**: ✅ Training and prediction use same features (when model detection works)

---

## 🎯 Summary

### Current Feature Configuration

**Training Features (14 features)**:
- 12 Alpha/Technical factors
- 2 Sato factors (momentum + divergence)

**Prediction Features**:
- Same as training features (detected from model)
- Fallback to 16-feature list if detection fails

### Sato Factor Integration

✅ **Confirmed in**:
1. T10_ALPHA_FACTORS (base universe)
2. t10_selected (training fallback)
3. base_features (prediction fallback)
4. Data file (pre-computed)

✅ **Status**: Fully integrated and ready for training and prediction

---

## 📝 Notes

1. **Feature Count**: 
   - Base universe: 16 features
   - Training (optimized): 14 features
   - Prediction fallback: 16 features

2. **Sato Factors**:
   - Always included in all feature lists
   - Pre-computed in data file (99.1% coverage)
   - Automatically computed if missing during data loading

3. **Consistency**:
   - Training and prediction use the same feature set
   - Sato factors are always included in both phases

4. **Best Features File**:
   - `best_features_per_model.json` does not exist
   - System uses fallback `t10_selected` list (includes Sato)
