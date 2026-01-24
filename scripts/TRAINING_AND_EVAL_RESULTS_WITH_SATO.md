# Training and Evaluation Results with Sato Factors

## ✅ Training Completed Successfully

### Snapshot Information
- **Snapshot ID**: `57232316-d42d-46f3-9bc3-90c005528337`
- **Training Run**: `results/full_dataset_training/run_20260121_034255/`
- **Status**: ✅ Completed
- **Sato Factors**: ✅ Included (confirmed in snapshot manifest)

### Snapshot Details
- **Total Features**: 22 features
- **Sato Factors Present**: ✅ `feat_sato_momentum_10d` and `feat_sato_divergence_10d` confirmed
- **Models Trained**: All models (ElasticNet, XGBoost, CatBoost, LambdaRank, MetaRankerStacker)

### Latest Snapshot ID
The snapshot ID has been saved to:
- `latest_snapshot_id.txt`: `57232316-d42d-46f3-9bc3-90c005528337`
- `results/full_dataset_training/run_20260121_034255/snapshot_id.txt`

---

## 🔄 80/20 Time Split Evaluation

### Status
- **Previous Run**: May have failed or is still running (no results in `t10_time_split_90_10`)
- **New Run Started**: ✅ Running with Sato factors
- **Output Directory**: `results/t10_time_split_80_20_sato/`
- **Command**:
  ```bash
  python scripts/time_split_80_20_oos_eval.py \
    --data-file "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
    --horizon-days 10 \
    --split 0.8 \
    --models catboost lambdarank ridge_stacking \
    --top-n 20 \
    --output-dir "results/t10_time_split_80_20_sato" \
    --log-level INFO
  ```

### Expected Output (when complete)
Results will be saved to `results/t10_time_split_80_20_sato/run_<timestamp>/`:
- `snapshot_id.txt` - Snapshot ID from training
- `report_df.csv` - Performance metrics (IC, Rank IC, Sharpe, returns, etc.)
- `ridge_top20_timeseries.csv` - Time series of Top 20 returns
- `top20_vs_qqq.png` - Top 20 vs QQQ comparison plot
- `top20_vs_qqq_cumulative.png` - Cumulative returns plot
- Model-specific bucket return plots

---

## 📊 Reference: Previous 80/20 Results (Without Sato)

For comparison, here are results from a previous 80/20 evaluation (run_20260120_041850):

### Model Performance Summary

| Model | IC | Rank IC | Top Sharpe (Net) | Win Rate | Avg Top Return |
|-------|----|---------|------------------|----------|----------------|
| **CatBoost** | 0.024 | 0.016 | 1.54 | 64% | 2.12% |
| **LambdaRank** | 0.033 | 0.011 | 0.69 | 56% | 1.78% |
| **Ridge Stacking** | 0.013 | -0.012 | 0.24 | 48% | 0.93% |
| **XGBoost** | 0.008 | 0.0003 | 0.36 | 60% | 0.54% |
| **ElasticNet** | -0.011 | 0.010 | 0.25 | 52% | -0.20% |

### Key Metrics
- **Best IC**: LambdaRank (0.033)
- **Best Rank IC**: CatBoost (0.016)
- **Best Sharpe**: CatBoost (1.54)
- **Best Win Rate**: CatBoost (64%)

---

## 🔍 What to Look For in New Results

### Expected Improvements with Sato Factors
1. **IC Improvement**: Sato factors showed IC = 0.0208 in validation, so models should benefit
2. **Feature Importance**: Sato factors should appear in feature importance rankings
3. **Model Performance**: Potentially improved Sharpe ratios and win rates
4. **Top 20 Returns**: Better cumulative returns vs QQQ benchmark

### Key Metrics to Compare
- **IC (Information Coefficient)**: Should be positive and significant
- **Rank IC**: Should show predictive power
- **Sharpe Ratio**: Top 20 Sharpe should be > 1.0
- **Win Rate**: Should be > 50%
- **Cumulative Returns**: Top 20 should outperform QQQ

---

## 📝 Next Steps

1. **Wait for 80/20 evaluation to complete** (~20-40 minutes)
2. **Review results**:
   - Check `results/t10_time_split_80_20_sato/run_<timestamp>/report_df.csv`
   - Compare metrics with previous results (above)
   - Check if Sato factors improved performance
3. **Analyze feature importance**:
   - Check which models use Sato factors
   - Verify Sato factors appear in top features
4. **Update production snapshot**:
   - The new snapshot (`57232316-d42d-46f3-9bc3-90c005528337`) is already set as default
   - Direct Predict will use this snapshot with Sato factors

---

## ✅ Confirmation

- ✅ Training completed with Sato factors
- ✅ Snapshot saved and set as default
- ✅ 80/20 evaluation running with Sato factors
- ✅ Data file confirmed to have Sato factors (24 columns total, 2 Sato columns)

---

## 📊 Data File Status

- **File**: `data/factor_exports/polygon_factors_all_filtered_clean.parquet`
- **Shape**: (4,180,394 rows, 24 columns)
- **Sato Factors**: ✅ Present
  - `feat_sato_momentum_10d`: 4,141,162 non-zero values (99.1%)
  - `feat_sato_divergence_10d`: 4,141,185 non-zero values (99.1%)
- **Date Range**: 2021-01-19 to 2025-12-30
- **Tickers**: 3,921
