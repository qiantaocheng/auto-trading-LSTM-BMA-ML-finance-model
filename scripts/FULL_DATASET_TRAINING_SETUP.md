# Full Dataset Training Setup

## Overview

This document describes the setup for training models with the complete dataset (no time split) and using the resulting snapshot as the default for production predictions in `app.py`.

## Changes Made

### 1. Created Training Script: `scripts/train_full_dataset.py`

A new script that:
- Trains all BMA Ultra models using the **complete dataset** (no time filtering)
- Creates a snapshot automatically during training
- Updates `latest_snapshot_id.txt` with the new snapshot ID
- Saves training logs and snapshot ID to `results/full_dataset_training/run_YYYYMMDD_HHMMSS/`

**Usage:**
```bash
python scripts/train_full_dataset.py \
    --train-data data/factor_exports/polygon_factors_all_filtered_clean.parquet \
    --top-n 50 \
    --log-level INFO
```

**Key Features:**
- No time split: Uses all available data for training
- Automatic snapshot creation: Snapshot is saved during training
- Updates latest snapshot: Automatically updates `latest_snapshot_id.txt`
- Production ready: Snapshot is ready for use in `app.py`

### 2. Updated `app.py` to Use Latest Snapshot

Modified `_direct_predict_snapshot()` method in `app.py` to:
- Read snapshot ID from `latest_snapshot_id.txt` if available
- Fallback to latest snapshot from database if file doesn't exist
- Use the snapshot for all predictions in the Direct Predict section

**Code Changes:**
```python
# Try to load snapshot ID from latest_snapshot_id.txt, fallback to None (latest from DB)
snapshot_id_to_use = None
try:
    latest_snapshot_file = Path(__file__).resolve().parent.parent / "latest_snapshot_id.txt"
    if latest_snapshot_file.exists():
        snapshot_id_to_use = latest_snapshot_file.read_text(encoding="utf-8").strip()
        if not snapshot_id_to_use:
            snapshot_id_to_use = None
except Exception:
    pass  # Fallback to None (latest from DB)

results = model.predict_with_snapshot(
    feature_data=date_feature_data,
    snapshot_id=snapshot_id_to_use,  # Use snapshot from latest_snapshot_id.txt or latest from DB
    universe_tickers=tickers,
    as_of_date=pred_date,
    prediction_days=1
)
```

## Training Process

### Current Status

Training is running in the background. The process will:

1. **Load Training Data**: Loads from `data/factor_exports/polygon_factors_all_filtered_clean.parquet`
2. **Train Models**: Trains all base models (ElasticNet, XGBoost, CatBoost, LambdaRank) and MetaRankerStacker
3. **Create Snapshot**: Automatically saves snapshot to model registry
4. **Update Files**: Updates `latest_snapshot_id.txt` with the new snapshot ID

### Expected Output

After training completes, you will find:

- **Snapshot ID**: Saved to `results/full_dataset_training/run_YYYYMMDD_HHMMSS/snapshot_id.txt`
- **Latest Snapshot**: Updated in `latest_snapshot_id.txt` (project root)
- **Training Logs**: Detailed logs in the run directory

### Verification

To verify the training completed successfully:

1. Check `latest_snapshot_id.txt` contains the new snapshot ID
2. Check `results/full_dataset_training/run_*/snapshot_id.txt` matches
3. Check training logs for "✅ Training complete! Snapshot ready for production use."

## Usage in app.py

Once training completes:

1. **Direct Predict Section**: Will automatically use the snapshot from `latest_snapshot_id.txt`
2. **No Manual Configuration**: No need to specify snapshot ID manually
3. **Automatic Updates**: When you retrain with full dataset, `latest_snapshot_id.txt` is updated automatically

## Retraining

To retrain with updated data:

```bash
python scripts/train_full_dataset.py \
    --train-data data/factor_exports/polygon_factors_all_filtered_clean.parquet \
    --top-n 50
```

The new snapshot will automatically become the default for `app.py` predictions.

## Notes

- **Full Dataset**: This training uses ALL available data (no time split), maximizing model performance
- **Production Ready**: The snapshot is immediately usable for predictions in `app.py`
- **Automatic**: No manual snapshot ID configuration needed in `app.py`
- **Backward Compatible**: Falls back to latest snapshot from database if `latest_snapshot_id.txt` doesn't exist
