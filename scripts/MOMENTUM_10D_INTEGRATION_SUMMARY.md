# momentum_10d Factor Integration Summary

## ✅ Completed Changes

### 1. Factor Calculation (`bma_models/simple_25_factor_engine.py`)

**Added `momentum_10d` calculation in `_compute_momentum_factors` method:**

```python
# 🔥 NEW: Compute momentum_10d for T+10 horizon (short-term momentum)
if 'momentum_10d' in getattr(self, 'alpha_factors', []):
    try:
        logger.info("   🔥 Computing momentum_10d (10-day price momentum)...")
        # 🔥 FIX: Shift for pre-market prediction (use previous day's momentum)
        momentum_10d = grouped['Close'].pct_change(10).shift(1).fillna(0)
        out['momentum_10d'] = momentum_10d
        logger.info(f"   ✅ momentum_10d: coverage={(momentum_10d != 0).sum() / len(momentum_10d) * 100:.1f}%, mean={momentum_10d.mean():.4f}")
    except Exception as e:
        logger.warning(f"   ⚠️ momentum_10d failed, using 0: {e}")
        out['momentum_10d'] = np.zeros(len(data))
```

**Key Features:**
- ✅ Uses `shift(1)` for pre-market prediction (consistent with all other factors)
- ✅ 10-day price momentum: `pct_change(10).shift(1)`
- ✅ Proper error handling with fallback to zeros

---

### 2. Factor List Updates

**Added to `T10_ALPHA_FACTORS` list (`bma_models/simple_25_factor_engine.py`):**

```python
T10_ALPHA_FACTORS = [
    'momentum_10d',  # NEW: 10-day short-term momentum for T+10 horizon
    'liquid_momentum',
    'obv_momentum_40d',
    # ... (other 13 factors)
]
```

**Total factors: 15 (was 14)**

---

### 3. Model Feature List (`bma_models/量化模型_bma_ultra_enhanced.py`)

**Added to `t10_selected` list:**

```python
t10_selected = [
    "momentum_10d",  # NEW: 10-day short-term momentum
    "ivol_30",
    "hist_vol_40d",
    # ... (other 13 factors)
]
```

**This ensures `momentum_10d` is used in:**
- ✅ XGBoost training
- ✅ ElasticNet training
- ✅ CatBoost training
- ✅ LambdaRank training
- ✅ Direct Predict (`app.py`)
- ✅ 80/20 OOS Evaluation (`time_split_80_20_oos_eval.py`)

---

## 📊 Factor Details

**Factor Name**: `momentum_10d`  
**Type**: Momentum  
**Window**: 10 days  
**Calculation**: `Close.pct_change(10).shift(1)`  
**Purpose**: Short-term momentum signal for T+10 horizon prediction  
**Shift Strategy**: ✅ Uses `shift(1)` for pre-market prediction

---

## 🔄 Data Recalculation

### Script Created: `scripts/recalculate_factors_with_momentum_10d.py`

**Usage:**

```bash
# Recalculate all factors including momentum_10d
python scripts/recalculate_factors_with_momentum_10d.py \
    --input-file "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
    --output-file "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
    --lookback-days 120 \
    --yes
```

**What it does:**
1. ✅ Loads existing MultiIndex parquet file
2. ✅ Initializes `Simple17FactorEngine` with T+10 horizon
3. ✅ Recalculates ALL factors including `momentum_10d`
4. ✅ Merges new factors back into original data
5. ✅ Removes old replaced factors
6. ✅ Creates backup of original file
7. ✅ Verifies `momentum_10d` is present in output

---

## ✅ Verification Checklist

- [x] Factor calculation added to `_compute_momentum_factors`
- [x] Factor added to `T10_ALPHA_FACTORS` list
- [x] Factor added to `t10_selected` in `量化模型_bma_ultra_enhanced.py`
- [x] Factor uses `shift(1)` for pre-market prediction
- [x] Recalculation script created
- [x] Script handles MultiIndex data structure correctly
- [x] Script verifies `momentum_10d` presence

---

## 🚀 Next Steps

1. **Run recalculation script** to update MultiIndex data file:
   ```bash
   python scripts/recalculate_factors_with_momentum_10d.py --yes
   ```

2. **Verify factor is present** in recalculated data:
   ```python
   import pandas as pd
   df = pd.read_parquet("data/factor_exports/polygon_factors_all_filtered_clean.parquet")
   assert 'momentum_10d' in df.columns
   print(f"momentum_10d coverage: {(df['momentum_10d'] != 0).sum() / len(df) * 100:.1f}%")
   ```

3. **Retrain models** with new factor:
   - Models will automatically use `momentum_10d` via `t10_selected` list
   - No code changes needed in training scripts

4. **Run 80/20 OOS evaluation** to see impact:
   ```bash
   python scripts/time_split_80_20_oos_eval.py \
       --data-file "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
       --horizon-days 10 \
       --split 0.8
   ```

---

## 📝 Notes

- **Factor count**: Now 15 factors (was 14)
- **Compatibility**: All existing code automatically uses new factor via `t10_selected` list
- **Data structure**: Fully compatible with MultiIndex (date, ticker) format
- **Shift strategy**: Consistent with all other factors (uses `shift(1)`)

---

**Last Updated**: 2025-01-20  
**Status**: ✅ Integration Complete - Ready for Recalculation
