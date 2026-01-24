# Retraining and 80/20 Evaluation Status

## ✅ File Update Completed

**Original file backed up**: `polygon_factors_all_filtered_clean_backup_YYYYMMDD_HHMMSS.parquet`
**File replaced**: `polygon_factors_all_filtered_clean.parquet` (now contains updated factors)

### Updated Factors in File:
- ✅ `obv_momentum_40d` (replaces obv_divergence)
- ✅ `feat_vol_price_div_30d` (replaces Sato factors)
- ✅ `vol_ratio_30d` (replaces vol_ratio_20d)
- ✅ `ret_skew_30d` (replaces ret_skew_20d)
- ✅ `ivol_30` (replaces ivol_20)
- ✅ `blowoff_ratio_30d` (replaces blowoff_ratio)

### Old Factors Removed:
- ✅ `obv_divergence`
- ✅ `feat_sato_momentum_10d`
- ✅ `feat_sato_divergence_10d`
- ✅ `vol_ratio_20d`
- ✅ `ret_skew_20d`
- ✅ `ivol_20`
- ✅ `blowoff_ratio`

---

## 🚀 Processes Started

### 1. Full Dataset Training
**Command**: 
```bash
python scripts/train_full_dataset.py \
  --train-data "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
  --top-n 50 \
  --log-level INFO
```

**Status**: ⏳ **Running in background**

**Expected Output**:
- `results/full_dataset_training/run_YYYYMMDD_HHMMSS/snapshot_id.txt`
- Training logs
- Model files

**Estimated Time**: 30-60 minutes

---

### 2. 80/20 Time Split Evaluation
**Command**:
```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean.parquet" \
  --horizon-days 10 \
  --split 0.8 \
  --models catboost lambdarank ridge_stacking \
  --top-n 20 \
  --disable-ema
```

**Status**: ⏳ **Running in background**

**Expected Output**:
- `results/t10_time_split_80_20_*/run_YYYYMMDD_HHMMSS/report_df.csv`
- `ridge_top20_timeseries.csv`
- `top20_vs_qqq.png`
- Model-specific bucket return files

**Estimated Time**: 20-40 minutes

---

## 📊 What's Being Trained/Evaluated

### Models:
1. **ElasticNet** (first layer)
2. **XGBoost** (first layer)
3. **CatBoost** (first layer)
4. **LambdaRank** (first layer)
5. **MetaRankerStacker** (second layer)

### Features Used (14 unified features):
1. `ivol_30`
2. `hist_vol_40d`
3. `near_52w_high`
4. `rsi_21`
5. `vol_ratio_30d`
6. `trend_r2_60`
7. `liquid_momentum`
8. `obv_momentum_40d` ⭐ NEW
9. `atr_ratio`
10. `ret_skew_30d` ⭐ UPDATED
11. `price_ma60_deviation`
12. `blowoff_ratio_30d` ⭐ UPDATED
13. `bollinger_squeeze`
14. `feat_vol_price_div_30d` ⭐ NEW

---

## ⏱️ Next Steps

1. **Monitor progress**: Check output directories for completion
2. **Review results**: Once complete, review training logs and evaluation metrics
3. **Update snapshot**: New snapshot ID will be saved for Direct Predict

---

## ✅ Summary

- ✅ File updated with new factors
- ✅ Full dataset training started
- ✅ 80/20 time split evaluation started
- ✅ EMA disabled by default in 80/20 evaluation
- ✅ All models using unified 14-feature set
