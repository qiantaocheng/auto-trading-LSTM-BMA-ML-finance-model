# Retrain with Sato Factors - Status

## ✅ Verification Complete

### Data File Status
- **File**: `data/factor_exports/polygon_factors_all_filtered_clean.parquet`
- **Shape**: (4,180,394 rows, 24 columns)
- **Date Range**: 2021-01-19 to 2025-12-30
- **Tickers**: 3,921

### Sato Factors Status
- ✅ `feat_sato_momentum_10d`: **PRESENT**
  - Min: -35.23, Max: 40.85, Mean: 0.094
  - Non-zero values: 4,141,162 / 4,180,394 (99.1%)
  
- ✅ `feat_sato_divergence_10d`: **PRESENT**
  - Min: -3.86, Max: 1.18, Mean: 0.009
  - Non-zero values: 4,141,185 / 4,180,394 (99.1%)

## 🚀 Training & Evaluation Started

### 1. Full Dataset Training
**Command**: 
```bash
python scripts/train_full_dataset.py --train-data "data/factor_exports/polygon_factors_all_filtered_clean.parquet" --top-n 50 --log-level INFO
```

**Status**: ✅ Running in background

**What it does**:
- Trains all models (ElasticNet, XGBoost, CatBoost, LambdaRank, MetaRankerStacker)
- Uses complete dataset (no time split)
- Includes Sato factors in feature list
- Creates snapshot for production use
- Updates `latest_snapshot_id.txt`

**Expected Output**:
- Snapshot ID saved to `results/full_dataset_training/run_<timestamp>/snapshot_id.txt`
- Snapshot ID also saved to `latest_snapshot_id.txt` in project root

### 2. 80/20 Time Split Evaluation
**Command**:
```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
  --horizon-days 10 \
  --split 0.8 \
  --models catboost lambdarank ridge_stacking \
  --top-n 20 \
  --log-level INFO
```

**Status**: ✅ Running in background

**What it does**:
- Splits data: 80% train, 20% test (by time)
- Trains models on 80% of dates
- Evaluates on 20% of dates (out-of-sample)
- Includes Sato factors in training and evaluation
- Generates performance metrics and plots

**Expected Output** (in `results/t10_time_split_90_10/run_<timestamp>/`):
- `snapshot_id.txt` - Snapshot ID from training
- `report_df.csv` - Performance metrics
- `ridge_top20_timeseries.csv` - Time series of Top 20 returns
- `top20_vs_qqq.png` - Top 20 vs QQQ comparison plot
- `top20_vs_qqq_cumulative.png` - Cumulative returns plot
- Model-specific bucket return plots

## ✅ Sato Factor Integration Confirmed

### Feature Lists Include Sato Factors
1. **`T10_ALPHA_FACTORS`** (line 3241 in `量化模型_bma_ultra_enhanced.py`):
   - ✅ `feat_sato_momentum_10d`
   - ✅ `feat_sato_divergence_10d`

2. **`t10_selected`** (lines 3296-3297):
   - ✅ `feat_sato_momentum_10d`
   - ✅ `feat_sato_divergence_10d`

3. **`base_features`** (line 5357):
   - ✅ `feat_sato_momentum_10d`
   - ✅ `feat_sato_divergence_10d`

### Data Loading Includes Sato Factors
- `_standardize_loaded_data()` (line 8115): Computes Sato if missing
- `_ensure_standard_feature_index()` (line 8260): Computes Sato if missing
- `time_split_80_20_oos_eval.py` (line 1429): Computes Sato if missing

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

## 🔍 Monitoring

### Check Training Progress
```bash
# Check latest training log
tail -f results/full_dataset_training/run_*/training.log

# Or check terminal output
# Training is running in background (Shell ID: 97879)
```

### Check 80/20 Evaluation Progress
```bash
# Check latest evaluation log
tail -f results/t10_time_split_90_10/run_*/evaluation.log

# Or check terminal output
# Evaluation is running in background (Shell ID: 635921)
```

### Verify Sato Factors in Training
The training process will:
1. Load data from `polygon_factors_all_filtered_clean.parquet`
2. Verify Sato factors are present (or compute if missing)
3. Include Sato factors in feature selection
4. Train models with Sato factors included

## ✅ Next Steps

1. **Wait for training to complete** (~30-60 minutes depending on data size)
2. **Wait for 80/20 evaluation to complete** (~20-40 minutes)
3. **Review results**:
   - Check snapshot ID in `latest_snapshot_id.txt`
   - Review evaluation metrics in `results/t10_time_split_90_10/run_<timestamp>/report_df.csv`
   - Check plots in the same directory
4. **Compare with previous results** (if available) to see impact of Sato factors

## 📝 Notes

- Both processes are running in parallel (training and evaluation)
- Sato factors are pre-computed in the cleaned parquet file (no on-the-fly computation needed)
- All models (ElasticNet, XGBoost, CatBoost, LambdaRank, MetaRankerStacker) will use Sato factors
- The 80/20 split ensures proper out-of-sample evaluation
