# Retrain and 80/20 Time Split Status

## 🚀 Started: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

### 1. Full Dataset Training
**Status**: ⏳ **Running in background**

**Command**:
```bash
python scripts/train_full_dataset.py \
  --train-data "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
  --top-n 50 \
  --log-level INFO
```

**What it does**:
- Trains all models (ElasticNet, XGBoost, CatBoost, LambdaRank, MetaRankerStacker)
- Uses complete dataset (no time split)
- Includes Sato factors from `polygon_factors_all_filtered_clean.parquet`
- Creates snapshot for production use
- Updates `latest_snapshot_id.txt`

**Expected Output**:
- Snapshot ID saved to `results/full_dataset_training/run_<timestamp>/snapshot_id.txt`
- Snapshot ID also saved to `latest_snapshot_id.txt` in project root

**Expected Duration**: ~30-60 minutes

---

### 2. 80/20 Time Split Evaluation
**Status**: ⏳ **Running in background**

**Command** (will run after training):
```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean.parquet" \
  --horizon-days 10 \
  --split 0.8 \
  --models catboost lambdarank ridge_stacking \
  --top-n 20 \
  --output-dir "results/t10_time_split_80_20_sato" \
  --log-level INFO
```

**What it does**:
- Splits data: 80% train, 20% test (by time)
- Trains models on 80% of dates
- Evaluates on 20% of dates (out-of-sample)
- Includes Sato factors from `polygon_factors_all_filtered_clean.parquet`
- Generates performance metrics and plots
- EMA disabled by default (--ema-top-n -1)

**Expected Output** (in `results/t10_time_split_80_20_sato/run_<timestamp>/`):
- `snapshot_id.txt` - Snapshot ID from training
- `report_df.csv` - Performance metrics (IC, Rank IC, Sharpe, returns, etc.)
- `ridge_top20_timeseries.csv` - Time series of Top 20 returns
- `top20_vs_qqq.png` - Top 20 vs QQQ comparison plot
- `top20_vs_qqq_cumulative.png` - Cumulative returns plot
- Model-specific bucket return plots

**Expected Duration**: ~20-40 minutes

---

## ✅ Confirmation

- ✅ Training data: `polygon_factors_all_filtered_clean.parquet` (includes Sato factors)
- ✅ Evaluation data: `polygon_factors_all_filtered_clean.parquet` (includes Sato factors)
- ✅ Sato factors: Pre-computed in data file (99.1% coverage)
- ✅ All first-layer models will use Sato factors

---

## 🔍 How to Monitor

### Check Training Progress
```powershell
# Check if process is running
Get-Process python | Where-Object {$_.StartTime -gt (Get-Date).AddHours(-2)}

# Check latest training log
Get-ChildItem "results\full_dataset_training\run_*" -Directory | Sort-Object Name -Descending | Select-Object -First 1 | Get-ChildItem -Filter "*.log" | Get-Content -Tail 50
```

### Check for Snapshot ID
```powershell
# After training completes, check snapshot ID
Get-Content "latest_snapshot_id.txt"
Get-Content "results\full_dataset_training\run_*\snapshot_id.txt" | Select-Object -Last 1
```

### Check 80/20 Evaluation Progress
```powershell
# Check if process is running
Get-Process python | Where-Object {$_.StartTime -gt (Get-Date).AddHours(-2)}

# Check for result files
Get-ChildItem "results\t10_time_split_80_20_sato\run_*" -Recurse -File | Sort-Object LastWriteTime -Descending | Select-Object -First 20
```

---

## 📝 Next Steps

1. **Wait for training to complete** (~30-60 minutes)
2. **Run 80/20 evaluation** (will start automatically or manually)
3. **Review results**:
   - Check `results/t10_time_split_80_20_sato/run_<timestamp>/report_df.csv`
   - Compare metrics with previous results
   - Check if Sato factors improved performance

---

## 📊 Expected Results

### Training Metrics
- Model performance metrics (IC, R2, etc.)
- Feature importance (Sato factors should appear)
- Cross-validation scores

### 80/20 Evaluation Metrics
- Out-of-sample IC (Information Coefficient)
- Top 20 cumulative returns
- Sharpe ratio
- Max drawdown
- Comparison vs QQQ benchmark
- HAC-corrected standard errors
