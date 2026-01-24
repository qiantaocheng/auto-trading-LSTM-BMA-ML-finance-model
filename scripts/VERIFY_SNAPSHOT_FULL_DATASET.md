# Verify Snapshot Full Dataset Training

## Current Snapshot Status

**Snapshot ID**: `7d5893d2-c8b0-43c1-b38e-9195424f8581`  
**Tag**: `auto_20260121_005021`  
**Created**: 2026-01-21 00:50:21

## ⚠️ Issue Identified

The snapshot tag is `auto_20260121_005021`, which suggests it was created automatically during training, **not** by the `train_full_dataset.py` script.

### Expected vs Actual

**Expected** (from `train_full_dataset.py`):
- Tag format: `full_dataset_YYYYMMDD_HHMMSS`
- Example: `full_dataset_20260121_000932`

**Actual**:
- Tag format: `auto_YYYYMMDD_HHMMSS`
- Example: `auto_20260121_005021`

## Possible Scenarios

### Scenario 1: Snapshot Created by Model Auto-Save
- The `UltraEnhancedQuantitativeModel.train_from_document()` method automatically creates snapshots
- Tag format: `auto_YYYYMMDD_HHMMSS`
- This happens during normal training, not specifically from `train_full_dataset.py`

### Scenario 2: Training Script Didn't Force-Save
- The `train_full_dataset.py` script may have used an existing snapshot
- Or the training completed but snapshot was created before the script's force-save logic

## How to Verify

### Method 1: Check Training Logs
Look for training logs in:
```
results/full_dataset_training/run_20260121_000932/
```

Check if logs show:
- "Training BMA Ultra models with COMPLETE DATASET (no time split)"
- "✅ Training complete! Snapshot ready for production use."

### Method 2: Check Snapshot Metadata
The manifest should contain training metadata:
```python
from bma_models.model_registry import load_manifest
m = load_manifest('7d5893d2-c8b0-43c1-b38e-9195424f8581')
print(m.get('training_date_range'))  # Should show full range
print(m.get('sample_count'))  # Should show all samples
```

### Method 3: Verify Training Data Range
Check if the training used ALL available data:
- Full dataset: All dates from data file
- Time-split: Only 90% of dates (for example)

## Recommendation

### Option 1: Verify Current Snapshot
If the current snapshot was trained with full dataset (no time split), it's fine to use.

**Check**:
1. Review training logs
2. Verify no time filtering was applied
3. Confirm all available data was used

### Option 2: Retrain with Explicit Tag
To ensure the snapshot is from full dataset training:

```bash
python scripts/train_full_dataset.py \
    --train-data data/factor_exports/polygon_factors_all_filtered_clean.parquet \
    --top-n 50 \
    --log-level INFO
```

This will:
- Train with complete dataset (no time split)
- Create snapshot with tag `full_dataset_YYYYMMDD_HHMMSS`
- Update `latest_snapshot_id.txt`

## Current Status

**Snapshot in use**: `7d5893d2-c8b0-43c1-b38e-9195424f8581`  
**Tag**: `auto_20260121_005021`  
**Likely source**: Auto-created during `train_from_document()` call  
**Full dataset**: ⚠️ **Unverified** - Need to check training logs or retrain

## Next Steps

1. **Check training logs** to verify full dataset was used
2. **Or retrain** with `train_full_dataset.py` to get explicit `full_dataset_` tag
3. **Verify** the snapshot contains MetaRankerStacker (✅ Confirmed)
