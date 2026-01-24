# Snapshot Data File Comparison

## Current Snapshot

**Snapshot ID**: `7d5893d2-c8b0-43c1-b38e-9195424f8581`  
**Tag**: `auto_20260121_005021`  
**Created**: 2026-01-21 00:50:21

## Previous Training Snapshot

**Snapshot ID**: `f7e666ca-7b60-4a61-aa48-69b1bc9db8f0`  
**Tag**: `auto_20260120_124936`  
**Created**: 2026-01-20 12:49:36  
**Used in**: 90/10 time split evaluation

## Data Files Used

### Previous Training (90/10 Split)
**Script**: `scripts/time_split_80_20_oos_eval.py`  
**Default Data File**: `D:\trade\data\factor_exports\polygon_factors_all_filtered.parquet`

### Current Training Script (Full Dataset)
**Script**: `scripts/train_full_dataset.py`  
**Default Data File**: `data/factor_exports/polygon_factors_all_filtered_clean.parquet`

## ⚠️ Important Finding

**These are DIFFERENT data files!**

1. **`polygon_factors_all_filtered.parquet`**
   - Used by time split evaluation script
   - Previous training runs

2. **`polygon_factors_all_filtered_clean.parquet`**
   - Used by full dataset training script
   - Current training script default

## Verification Needed

Since the current snapshot has tag `auto_20260121_005021` (not `full_dataset_...`), it was likely created automatically during training, not by the `train_full_dataset.py` script.

**To verify which data file was used:**

1. **Check training logs** (if available)
2. **Compare file contents** (date ranges, sample counts)
3. **Check snapshot metadata** (if training data path is stored)

## Recommendation

### Option 1: Verify Current Snapshot
Check if the current snapshot (`7d5893d2...`) was trained with:
- `polygon_factors_all_filtered.parquet` (same as previous)
- OR `polygon_factors_all_filtered_clean.parquet` (different file)

### Option 2: Retrain with Explicit Data File
To ensure using the correct data file:

```bash
# Use the same file as previous training
python scripts/train_full_dataset.py \
    --train-data data/factor_exports/polygon_factors_all_filtered.parquet \
    --top-n 50

# OR use the clean file (if that's what you want)
python scripts/train_full_dataset.py \
    --train-data data/factor_exports/polygon_factors_all_filtered_clean.parquet \
    --top-n 50
```

## Feature Comparison

Both snapshots have:
- **Same number of features**: 20
- **Same feature names**: Identical feature sets

This suggests they might be using similar data sources, but the actual data files are different.

## Next Steps

1. **Determine which data file to use**:
   - `polygon_factors_all_filtered.parquet` (previous standard)
   - `polygon_factors_all_filtered_clean.parquet` (clean version)

2. **Retrain with explicit data file** to ensure consistency

3. **Update `train_full_dataset.py`** default to match your preference
