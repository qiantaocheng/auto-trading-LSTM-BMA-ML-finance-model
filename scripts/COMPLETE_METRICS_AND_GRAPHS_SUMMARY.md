# Complete Metrics and Graphs Summary

## Integration Status

✅ **All metrics from `generate_complete_metrics_report.py` have been integrated into `time_split_80_20_oos_eval.py`**

✅ **All requirements from `EWMA_IMPLEMENTATION_SUMMARY.md` are implemented:**

### 1. Filter Functionality Removal ✅
- ✅ Deleted `filter_top15_by_volatility_volume` function
- ✅ Removed all `filter_top15` parameters
- ✅ Removed all `factor_data` parameters (filtering-related parts)
- ✅ Cleaned up all filtering-related calls

### 2. EWMA Smoothing ✅
- ✅ `apply_ema_smoothing` function is enabled and active
- ✅ Applied to all models: catboost, lambdarank, ridge_stacking
- ✅ EWMA parameters: 3-day EMA with weights (0.6, 0.3, 0.1)
- ✅ Smoothing formula: `S_smooth_t = 0.6*S_t + 0.3*S_{t-1} + 0.1*S_{t-2}`
- ✅ All predictions use smoothed scores for ranking

### 3. Max Drawdown Calculation ✅
- ✅ Added to `calc_top10_accumulated_10d_rebalance` function
- ✅ Saved to CSV file (drawdown column)
- ✅ Logged in console output
- ✅ Included in metrics report

### 4. Default Models ✅
- ✅ Default model list: `["catboost", "lambdarank", "ridge_stacking"]`

---

## Generated Files Per Model

For each model (catboost, lambdarank, ridge_stacking), the following files are generated:

### CSV Files

1. **`{model}_bucket_returns.csv`**
   - Overlap daily returns (249 trading days)
   - Columns include:
     - `date`: Trading date
     - `top_1_10_return`: Top 1-10 bucket return (%)
     - `top_5_15_return`: Top 5-15 bucket return (%) ⭐
     - `top_11_20_return`: Top 11-20 bucket return (%)
     - `top_21_30_return`: Top 21-30 bucket return (%)
     - `bottom_1_10_return`: Bottom 1-10 bucket return (%)
     - `bottom_11_20_return`: Bottom 11-20 bucket return (%)
     - `bottom_21_30_return`: Bottom 21-30 bucket return (%)
     - `benchmark_return`: Benchmark (QQQ) return (%)
     - Cumulative return columns (`cum_top_*`, `cum_bottom_*`, `cum_benchmark_return`)
     - Net return columns (with transaction costs)

2. **`{model}_top5_15_rebalance10d_accumulated.csv`**
   - Non-Overlap 10-day rebalance periods (25 periods)
   - Columns include:
     - `date`: Rebalance date
     - `top_gross_return`: Period gross return (decimal)
     - `acc_value`: Cumulative value (1 + returns)
     - `acc_return`: Cumulative return (decimal)
     - `drawdown`: Drawdown percentage (%) ⭐

3. **`{model}_bucket_summary.csv`**
   - Summary statistics for bucket returns

4. **`{model}_top30_nonoverlap_timeseries.csv`**
   - Non-overlapping timeseries for Top 30 stocks

### PNG Graph Files

1. **`{model}_top5_15_rebalance10d_accumulated.png`**
   - Cumulative return curve for Top 5-15 bucket
   - Shows accumulated return over 25 periods (10-day rebalance)

2. **`{model}_bucket_returns_period.png`**
   - Two-panel plot:
     - Top panel: Top bucket returns (per-period) vs benchmark
     - Bottom panel: Bottom bucket returns (per-period) vs benchmark

3. **`{model}_bucket_returns_cumulative.png`**
   - Two-panel plot:
     - Top panel: Cumulative top bucket returns vs benchmark
     - Bottom panel: Cumulative bottom bucket returns vs benchmark

### Report Files

1. **`complete_metrics_report.txt`**
   - Complete metrics report for all models
   - Includes both Overlap and Non-Overlap metrics
   - Generated automatically at the end of processing

2. **`results_summary_for_word_doc.json`**
   - JSON summary with all metrics (HAC-corrected)

---

## Calculated Metrics

### Overlap Metrics (Daily Observations, ~249 Trading Days)

For **Top 5-15** bucket, calculated from `{model}_bucket_returns.csv`:

1. **平均收益 (Average Return)**: Mean of daily returns
2. **中位数收益 (Median Return)**: Median of daily returns
3. **标准差 (Standard Deviation)**: Std dev of daily returns
4. **Overlap 胜率 (Win Rate)**: Percentage of days with positive returns
5. **Sharpe Ratio (年化)**: Annualized Sharpe ratio
   - Formula: `(mean / std) * sqrt(252)`

### Non-Overlap Metrics (10-Day Periods, 25 Periods)

For **Top 5-15** bucket, calculated from `{model}_top5_15_rebalance10d_accumulated.csv`:

1. **平均期间收益 (Average Period Return)**: Mean of period returns
2. **中位数期间收益 (Median Period Return)**: Median of period returns
3. **标准差 (Standard Deviation)**: Std dev of period returns
4. **Non-Overlap 胜率 (Win Rate)**: Percentage of periods with positive returns
5. **累积收益 (Cumulative Return)**: Final accumulated return
6. **最大回撤 (Max Drawdown)**: Maximum drawdown percentage ⭐
   - Formula: `min((acc_value / running_max - 1) * 100)`
7. **年化收益 (Annualized Return)**: Annualized return
   - Formula: `((1 + final_return) ^ (252 / total_days) - 1) * 100`
8. **Sharpe Ratio (基于期间)**: Sharpe ratio based on periods
   - Formula: `(mean / std) * sqrt(25)`

---

## Console Output

The script outputs:

1. **EWMA Smoothing Status**
   - Confirmation that EMA smoothing is applied to each model
   - Shows smoothing weights: (0.6, 0.3, 0.1)

2. **Max Drawdown Logging**
   - Logged for each model's Top 5-15 accumulated return calculation
   - Format: `最大回撤: {max_dd:.4f}%`

3. **Complete Metrics Report**
   - Printed to console at the end
   - Includes all Overlap and Non-Overlap metrics for all models
   - Includes EWMA smoothing disclosure note

---

## Example Run Command

```bash
python scripts/time_split_80_20_oos_eval.py \
  --horizon-days 10 \
  --top-n 20 \
  --cost-bps 10 \
  --split 0.9 \
  --output-dir results/t10_time_split_90_10_ewma \
  --snapshot-id 9de0b13d-647d-4c8d-bf3d-86d3ab8a738f \
  --models catboost lambdarank ridge_stacking
```

**Note:** Default split is now 0.9 (90% training, 10% prediction). The `--split 0.9` parameter is optional as it's now the default.

---

## Output Directory Structure

```
results/t10_time_split_90_10_ewma/
└── run_YYYYMMDD_HHMMSS/
    ├── snapshot_id.txt
    ├── report_df.csv
    ├── complete_metrics_report.txt ⭐ NEW
    ├── results_summary_for_word_doc.json
    │
    ├── catboost_bucket_returns.csv
    ├── catboost_top5_15_rebalance10d_accumulated.csv
    ├── catboost_top5_15_rebalance10d_accumulated.png
    ├── catboost_bucket_returns_period.png
    ├── catboost_bucket_returns_cumulative.png
    ├── catboost_bucket_summary.csv
    ├── catboost_top30_nonoverlap_timeseries.csv
    │
    ├── lambdarank_bucket_returns.csv
    ├── lambdarank_top5_15_rebalance10d_accumulated.csv
    ├── lambdarank_top5_15_rebalance10d_accumulated.png
    ├── lambdarank_bucket_returns_period.png
    ├── lambdarank_bucket_returns_cumulative.png
    ├── lambdarank_bucket_summary.csv
    ├── lambdarank_top30_nonoverlap_timeseries.csv
    │
    ├── ridge_stacking_bucket_returns.csv
    ├── ridge_stacking_top5_15_rebalance10d_accumulated.csv
    ├── ridge_stacking_top5_15_rebalance10d_accumulated.png
    ├── ridge_stacking_bucket_returns_period.png
    ├── ridge_stacking_bucket_returns_cumulative.png
    ├── ridge_stacking_bucket_summary.csv
    └── ridge_stacking_top30_nonoverlap_timeseries.csv
```

---

## Notes

1. **EWMA Smoothing**: Uses 3-day history, so the first 2 days may not have full smoothing effect
2. **All Predictions**: Use smoothed scores for ranking
3. **Max Drawdown**: Based on cumulative return curve (non-overlapping periods)
4. **All Metrics**: Calculated based on EWMA-smoothed predictions
5. **Report Generation**: Automatic at the end of processing, no separate script needed

---

## Verification Checklist

- ✅ Filter functionality removed
- ✅ EWMA smoothing enabled and applied
- ✅ Max drawdown calculation added
- ✅ Default models updated
- ✅ Complete metrics report integrated
- ✅ All CSV files generated with correct columns
- ✅ All PNG graphs generated
- ✅ Console output includes all metrics
- ✅ Report file saved automatically
