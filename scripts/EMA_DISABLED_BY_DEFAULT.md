# EMA Disabled by Default in 80/20 Time Split OOS Evaluation

## 🔧 Change Summary

The default EMA smoothing setting in 80/20 time split OOS evaluation has been changed to **disabled** (`--ema-top-n -1`).

## ✅ Before

- **Default**: `--ema-top-n 300` (EMA enabled, applied to Top 300 stocks)
- EMA smoothing was applied by default to stocks in the top 300 for at least 3 consecutive days

## ✅ After

- **Default**: `--ema-top-n -1` (EMA disabled)
- Raw predictions are used by default (no EMA smoothing)
- Users can explicitly enable EMA by setting `--ema-top-n` to a positive value

## 📊 Code Changes

**File**: `scripts/time_split_80_20_oos_eval.py`

**Line ~361**:
```python
# Before
p.add_argument("--ema-top-n", type=int, default=300, 
               help="Only apply EMA to stocks in top N (default: 300, set to 0 to apply to all, set to -1 to disable EMA)")

# After
p.add_argument("--ema-top-n", type=int, default=-1, 
               help="Only apply EMA to stocks in top N (default: -1 to disable EMA, set to 0 to apply to all, set to >0 for top N filter)")
```

## 🎯 Usage

### Default Behavior (EMA Disabled)
```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
  --horizon-days 10 \
  --split 0.8 \
  --models catboost lambdarank ridge_stacking \
  --top-n 20
```
**Result**: Uses raw predictions (no EMA smoothing)

### Enable EMA for Top 300
```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
  --horizon-days 10 \
  --split 0.8 \
  --models catboost lambdarank ridge_stacking \
  --top-n 20 \
  --ema-top-n 300
```
**Result**: EMA smoothing applied to Top 300 stocks (min 3 consecutive days)

### Enable EMA for All Stocks
```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
  --horizon-days 10 \
  --split 0.8 \
  --models catboost lambdarank ridge_stacking \
  --top-n 20 \
  --ema-top-n 0
```
**Result**: EMA smoothing applied to all stocks

## 📝 Impact

### Scripts Using Default Settings

1. **`scripts/run_sato_evaluation.py`**:
   - ✅ Now uses EMA disabled by default (no change needed)
   - Uses raw predictions for evaluation

2. **Direct calls to `time_split_80_20_oos_eval.py`**:
   - ✅ Now default to EMA disabled
   - Users must explicitly enable EMA if desired

### Log Output

When EMA is disabled (default), you'll see:
```
📊 EMA smoothing DISABLED for catboost (using raw predictions)...
✅ Using raw predictions for catboost (EMA disabled)
```

When EMA is enabled (explicitly):
```
📊 Applying EMA smoothing to catboost predictions (Top300 filter, min 3 days)...
   EMA coverage: 45.23% of predictions applied EMA
✅ EMA smoothing applied to catboost
```

## 🎯 Rationale

1. **Consistency**: Direct Predict already uses raw predictions (no EMA)
2. **Transparency**: Raw predictions show model performance without smoothing artifacts
3. **Flexibility**: Users can still enable EMA when needed
4. **Baseline**: Provides a clear baseline for comparison

## ⚠️ Notes

- This change only affects the **default** behavior
- Existing scripts that explicitly set `--ema-top-n` will continue to work as before
- The `--ema-min-days` parameter is only used when `--ema-top-n > 0`
