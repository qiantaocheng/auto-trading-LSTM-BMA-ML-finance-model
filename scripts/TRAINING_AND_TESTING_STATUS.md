# Training and Testing Status

## Current Status: Training in Progress

### Training Configuration
- **Time Split**: 90/10 (90% training, 10% prediction)
- **Training Period**: 2021-01-19 to 2025-06-17
- **Test Period**: 2025-07-03 to 2025-12-30
- **Horizon Days**: 10
- **Models**: catboost, lambdarank, ridge_stacking (MetaRankerStacker)
- **Output Directory**: `results/t10_time_split_90_10_ewma_train`

### What's Happening

1. **Training Phase** (Currently Running)
   - Loading data from `polygon_factors_all_filtered.parquet`
   - Training all models on 90% of data (2021-01-19 to 2025-06-17)
   - Creating MetaRankerStacker (replacing old RidgeStacker)
   - Generating new snapshot with MetaRankerStacker
   - This may take 30-60 minutes depending on data size

2. **After Training Completes**
   - A new snapshot ID will be generated
   - The snapshot will contain MetaRankerStacker (not RidgeStacker)
   - Testing will automatically run on the 10% test set

3. **Expected Outputs**
   - New snapshot ID saved to `snapshot_id.txt`
   - All metrics and graphs generated automatically
   - Complete metrics report in `complete_metrics_report.txt`

### Monitoring Progress

Check the training log:
```powershell
Get-Content training_output.log -Tail 50 -Wait
```

Or check the output directory:
```powershell
ls results/t10_time_split_90_10_ewma_train/
```

### Next Steps After Training

Once training completes:
1. The script will automatically run testing on the 10% test set
2. All metrics will be calculated (Overlap and Non-Overlap)
3. All graphs will be generated
4. Complete metrics report will be saved

### Expected Files Generated

For each model (catboost, lambdarank, ridge_stacking):
- `{model}_bucket_returns.csv` - Overlap daily returns
- `{model}_top5_15_rebalance10d_accumulated.csv` - Non-overlap periods with drawdown
- `{model}_top5_15_rebalance10d_accumulated.png` - Cumulative return curve
- `{model}_bucket_returns_period.png` - Per-period returns
- `{model}_bucket_returns_cumulative.png` - Cumulative returns
- `complete_metrics_report.txt` - Complete metrics report

### Key Differences from Previous Run

1. **New Snapshot**: Will contain MetaRankerStacker instead of deprecated RidgeStacker
2. **90/10 Split**: More training data (90% vs 80%), less test data (10% vs 20%)
3. **EWMA Smoothing**: All predictions use 3-day EMA smoothing
4. **Complete Metrics**: All Overlap and Non-Overlap metrics automatically calculated

### Troubleshooting

If training fails:
- Check `training_output.log` for error messages
- Verify data file exists: `D:\trade\data\factor_exports\polygon_factors_all_filtered.parquet`
- Ensure sufficient disk space for cache/model_snapshots

If you need to stop and restart:
- The training can be interrupted (Ctrl+C)
- You can resume by providing the snapshot_id if one was created
- Or restart training from scratch
