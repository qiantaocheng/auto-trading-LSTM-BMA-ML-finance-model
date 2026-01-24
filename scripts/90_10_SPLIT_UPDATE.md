# 90/10 Time Split Update Summary

## Changes Made

The time split has been updated from **80/20** (80% training, 20% prediction) to **90/10** (90% training, 10% prediction).

### Script Updates (`time_split_80_20_oos_eval.py`)

1. **Default Split Ratio**
   - Changed from `0.8` to `0.9` (line 346)
   - Updated help text: "Train split fraction by time (default 0.9)"

2. **Default Output Directory**
   - Changed from `results/t10_time_split_80_20` to `results/t10_time_split_90_10` (line 359)

3. **Documentation Updates**
   - Updated docstring: "90/10 time-split train/test" (line 4)
   - Updated comments: "Train on first 90% dates" and "Evaluate on last 10% dates" (lines 8, 10)

4. **Log Messages**
   - Updated: "Loading data to compute 90/10 time split..." (line 1307)
   - Updated: "90/10 split may be noisy" (line 1431)

5. **Logger Name**
   - Updated from `time_split_80_20` to `time_split_90_10` (line 1285)

6. **Comments**
   - Updated EMA smoothing comments to reflect current implementation (lines 1658, 1881)

### Documentation Updates

1. **EWMA_IMPLEMENTATION_SUMMARY.md**
   - Updated example command to include `--split 0.9`
   - Updated output directory to `results/t10_time_split_90_10_ewma`
   - Added note about default split being 0.9

2. **COMPLETE_METRICS_AND_GRAPHS_SUMMARY.md**
   - Updated example command to include `--split 0.9`
   - Updated output directory structure to `results/t10_time_split_90_10_ewma`
   - Added note that `--split 0.9` is optional (now default)

## Impact

- **Training Data**: Now uses 90% of available dates (increased from 80%)
- **Test Data**: Now uses 10% of available dates (decreased from 20%)
- **More Training Data**: Better model training with more historical data
- **Smaller Test Set**: More conservative out-of-sample evaluation

## Usage

The script now defaults to 90/10 split. You can still override it:

```bash
# Use default 90/10 split
python scripts/time_split_80_20_oos_eval.py \
  --horizon-days 10 \
  --top-n 20 \
  --cost-bps 10 \
  --output-dir results/t10_time_split_90_10_ewma \
  --snapshot-id <snapshot-id> \
  --models catboost lambdarank ridge_stacking

# Override to use different split (e.g., 85/15)
python scripts/time_split_80_20_oos_eval.py \
  --split 0.85 \
  --horizon-days 10 \
  --top-n 20 \
  --cost-bps 10 \
  --output-dir results/t10_time_split_85_15_ewma \
  --snapshot-id <snapshot-id> \
  --models catboost lambdarank ridge_stacking
```

## Validation

- ✅ No linter errors
- ✅ All references to 80/20 updated to 90/10
- ✅ Default values updated correctly
- ✅ Documentation updated
- ✅ Backward compatible (can still specify `--split 0.8` if needed)
