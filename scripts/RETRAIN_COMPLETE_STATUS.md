# Retraining and 80/20 Evaluation - Status

## ✅ File Update Completed

**Main File**: `D:/trade/data/factor_exports/polygon_factors_all_filtered_clean.parquet`
- ✅ **Backup created**: `polygon_factors_all_filtered_clean_backup_YYYYMMDD_HHMMSS.parquet`
- ✅ **File replaced** with recalculated version

### Verification Results:
- ✅ **Shape**: (4,180,394 rows, 24 columns) - Preserved
- ✅ **All 6 new factors present**:
  - `obv_momentum_40d`
  - `feat_vol_price_div_30d`
  - `vol_ratio_30d`
  - `ret_skew_30d`
  - `ivol_30`
  - `blowoff_ratio_30d`
- ✅ **All 7 old factors removed**
- ✅ **All 16 T10_ALPHA_FACTORS present**

---

## 🚀 Processes Running

### 1. Full Dataset Training
**Status**: ⏳ **Running** (Started: 2026-01-21 14:03:19)

**Command**:
```bash
python scripts/train_full_dataset.py \
  --train-data "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
  --top-n 50 \
  --log-level INFO
```

**Output Directory**: `results/full_dataset_training/run_20260121_140319`

**Expected Output**:
- `snapshot_id.txt` - New snapshot ID for Direct Predict
- Training logs
- Model files

**Estimated Time**: 30-60 minutes

**Progress**: Training started, loading data and initializing models...

---

### 2. 80/20 Time Split Evaluation
**Status**: ⏳ **Running** (Started: 2026-01-21 14:03:XX)

**Command**:
```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean.parquet" \
  --horizon-days 10 \
  --split 0.8 \
  --models catboost lambdarank ridge_stacking \
  --top-n 20 \
  --ema-top-n -1
```

**Output Directory**: `results/t10_time_split_80_20_*/run_YYYYMMDD_HHMMSS`

**Expected Output**:
- `report_df.csv` - Performance metrics
- `ridge_top20_timeseries.csv` - Time series data
- `top20_vs_qqq.png` - Comparison plots
- Model-specific bucket return files

**Estimated Time**: 20-40 minutes

**EMA**: Disabled (`--ema-top-n -1`)

---

## 📊 Features Being Used

### Unified 14-Feature Set:
1. `ivol_30` ⭐ (updated from ivol_20)
2. `hist_vol_40d`
3. `near_52w_high`
4. `rsi_21`
5. `vol_ratio_30d` ⭐ (updated from vol_ratio_20d)
6. `trend_r2_60`
7. `liquid_momentum`
8. `obv_momentum_40d` ⭐ NEW (replaces obv_divergence)
9. `atr_ratio`
10. `ret_skew_30d` ⭐ (updated from ret_skew_20d)
11. `price_ma60_deviation`
12. `blowoff_ratio_30d` ⭐ (updated from blowoff_ratio)
13. `bollinger_squeeze`
14. `feat_vol_price_div_30d` ⭐ NEW (replaces Sato factors)

### Models Training:
- **ElasticNet** (first layer)
- **XGBoost** (first layer)
- **CatBoost** (first layer)
- **LambdaRank** (first layer)
- **MetaRankerStacker** (second layer)

---

## ⏱️ Next Steps

1. **Monitor progress**: Check terminal outputs for completion
2. **Review results**: Once complete, review:
   - Training snapshot ID
   - 80/20 evaluation metrics
   - Performance comparisons
3. **Update Direct Predict**: New snapshot will be available for Direct Predict feature

---

## ✅ Summary

- ✅ **File updated** with all new factors
- ✅ **Old factors removed** completely
- ✅ **Full dataset training** running
- ✅ **80/20 time split evaluation** running
- ✅ **EMA disabled** in 80/20 evaluation
- ✅ **All models using unified 14-feature set**

**Status**: Both processes running successfully! 🚀
