# Bollinger Squeeze Feature Restored

## ✅ Changes Made

### 1. T10_ALPHA_FACTORS (Base Universe)
**File**: `bma_models/simple_25_factor_engine.py` (line 58-78)

**Change**: Re-added `bollinger_squeeze` to the feature list
- **Before**: Commented out (line 69)
- **After**: Active feature (line 11 in list)

**Total Features**: Now **17 features** (was 16)

---

### 2. Training Features (t10_selected)
**File**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 3283-3298)

**Change**: Added `bollinger_squeeze` to training feature list
- **Before**: 14 features (without bollinger_squeeze)
- **After**: **15 features** (with bollinger_squeeze)

**Used for**: ElasticNet, CatBoost, XGBoost, LambdaRank (all first-layer models)

---

### 3. Prediction Features (base_features)
**File**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 5352-5360)

**Change**: Added `bollinger_squeeze` to prediction fallback list
- **Before**: 16 features (without bollinger_squeeze)
- **After**: **17 features** (with bollinger_squeeze)

**Used for**: Prediction when model feature detection fails

---

### 4. Data File Status
**File**: `data/factor_exports/polygon_factors_all_filtered_clean.parquet`

**Status**: ✅ **Already contains `bollinger_squeeze`**
- Feature exists in the multiindex file
- No data file update needed

---

## 📊 Updated Feature Lists

### T10_ALPHA_FACTORS (17 features)
1. liquid_momentum
2. obv_divergence
3. ivol_20
4. rsi_21
5. trend_r2_60
6. near_52w_high
7. ret_skew_20d
8. blowoff_ratio
9. hist_vol_40d
10. atr_ratio
11. **bollinger_squeeze** ✅ (restored)
12. vol_ratio_20d
13. price_ma60_deviation
14. 5_days_reversal
15. downside_beta_ewm_21
16. feat_sato_momentum_10d
17. feat_sato_divergence_10d

### t10_selected (15 features - Training)
1. ivol_20
2. hist_vol_40d
3. near_52w_high
4. rsi_21
5. vol_ratio_20d
6. trend_r2_60
7. liquid_momentum
8. obv_divergence
9. atr_ratio
10. ret_skew_20d
11. price_ma60_deviation
12. blowoff_ratio
13. **bollinger_squeeze** ✅ (restored)
14. feat_sato_momentum_10d
15. feat_sato_divergence_10d

### base_features (17 features - Prediction Fallback)
1. liquid_momentum
2. obv_divergence
3. ivol_20
4. rsrs_beta_18
5. rsi_21
6. trend_r2_60
7. near_52w_high
8. ret_skew_20d
9. blowoff_ratio
10. hist_vol_40d
11. atr_ratio
12. **bollinger_squeeze** ✅ (restored)
13. vol_ratio_20d
14. price_ma60_deviation
15. feat_sato_momentum_10d
16. feat_sato_divergence_10d
17. making_new_low_5d

---

## ✅ Confirmation Checklist

- [x] **T10_ALPHA_FACTORS**: ✅ Includes bollinger_squeeze (17 features total)
- [x] **t10_selected (Training)**: ✅ Includes bollinger_squeeze (15 features total)
- [x] **base_features (Prediction)**: ✅ Includes bollinger_squeeze (17 features total)
- [x] **Data file**: ✅ Contains bollinger_squeeze (pre-computed)
- [x] **All first-layer models**: ✅ Will use bollinger_squeeze in training and prediction

---

## 🎯 Impact

### Training
- All first-layer models (ElasticNet, XGBoost, CatBoost, LambdaRank) will now use `bollinger_squeeze` as input feature
- Training feature count: 15 features (was 14)

### Direct Predict
- Will compute `bollinger_squeeze` from live market data
- Will include it in predictions

### 80/20 Time Split OOS
- Will load `bollinger_squeeze` from parquet file
- All first-layer models will use it for predictions

---

## 📝 Notes

1. **Feature Availability**: `bollinger_squeeze` is already present in the multiindex data file, so no data file update is needed
2. **Backward Compatibility**: Models trained before this change will still work (they just won't use bollinger_squeeze)
3. **Next Training**: New models trained after this change will include bollinger_squeeze
4. **Comment Updated**: Removed the "REMOVED" comment and updated to indicate it's been re-added

---

## 🔄 Next Steps

1. **Re-train models** to include bollinger_squeeze:
   ```bash
   python scripts/train_full_dataset.py \
     --train-data "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
     --top-n 50
   ```

2. **Re-run 80/20 evaluation**:
   ```bash
   python scripts/time_split_80_20_oos_eval.py \
     --data-file "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean.parquet" \
     --horizon-days 10 \
     --split 0.8 \
     --models catboost lambdarank ridge_stacking
   ```

3. **Direct Predict**: Will automatically use bollinger_squeeze (computed from live data)
