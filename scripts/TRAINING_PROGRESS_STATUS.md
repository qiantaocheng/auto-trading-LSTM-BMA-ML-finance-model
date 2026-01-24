# Training Progress Status

**Last Updated**: 2026-01-21 14:12 (approximately)

---

## 🔄 Current Status

### 1. Full Dataset Training
**Process ID**: 31796  
**Started**: 14:03:19  
**Runtime**: ~9 minutes  
**Status**: ⏳ **RUNNING**

**Output Directory**: `results/full_dataset_training/run_20260121_140319`  
**Files Generated**: None yet (still in early stages)

**Estimated Progress**:
- ✅ Process started and running
- ✅ Output directory created
- ⏳ Currently in: **Data loading / Feature computation** phase
- ⏳ Next: Model training (ElasticNet → XGBoost → CatBoost → LambdaRank → MetaRankerStacker)

**Expected Timeline**:
- **Total**: 30-60 minutes
- **Current Stage**: ~15-20% complete (estimated)
- **Remaining**: ~25-50 minutes

---

### 2. 80/20 Time Split Evaluation
**Process ID**: 32032  
**Started**: 14:04:41  
**Runtime**: ~7.6 minutes  
**Status**: ⏳ **RUNNING**

**Output Directory**: Not created yet (will be created when process progresses)

**Estimated Progress**:
- ✅ Process started and running
- ⏳ Currently in: **Data loading / Train-test split** phase
- ⏳ Next: Training models on 80% data → Prediction on 20% test → Metrics calculation

**Expected Timeline**:
- **Total**: 20-40 minutes
- **Current Stage**: ~20-30% complete (estimated)
- **Remaining**: ~15-30 minutes

---

## 📊 Process Details

### Training Process (PID 31796)
- **CPU Time**: 340 seconds (5.7 minutes)
- **Memory Usage**: ~528 MB
- **Status**: Active and processing

### 80/20 Evaluation Process (PID 32032)
- **CPU Time**: 274.6 seconds (4.6 minutes)
- **Memory Usage**: High (actively processing)
- **Status**: Active and processing

---

## 🔍 What's Happening Now

### Training Process Stages:
1. ✅ **Initialization** - Complete
2. ⏳ **Data Loading** - Loading `polygon_factors_all_filtered_clean.parquet` (4.18M rows)
3. ⏳ **Feature Validation** - Checking all 14 features are present
4. ⏳ **Model Training** - Will train 5 models sequentially:
   - ElasticNet (~2-5 min)
   - XGBoost (~5-10 min)
   - CatBoost (~5-10 min)
   - LambdaRank (~5-10 min)
   - MetaRankerStacker (~5-10 min)
5. ⏳ **Snapshot Creation** - Save snapshot ID (~1-2 min)

### 80/20 Evaluation Stages:
1. ✅ **Initialization** - Complete
2. ⏳ **Data Loading** - Loading parquet file
3. ⏳ **Time Split** - Splitting into 80% train / 20% test
4. ⏳ **Model Training** - Training on 80% data (~15-25 min)
5. ⏳ **Prediction** - Predicting on 20% test data (~5-10 min)
6. ⏳ **Metrics & Plots** - Calculating metrics and generating plots (~2-5 min)

---

## ✅ Completion Indicators

### Training Complete When:
- File appears: `results/full_dataset_training/run_20260121_140319/snapshot_id.txt`
- File appears: `latest_snapshot_id.txt` (in project root)

### 80/20 Evaluation Complete When:
- Directory created: `results/t10_time_split_80_20_*/run_YYYYMMDD_HHMMSS/`
- Files appear:
  - `report_df.csv`
  - `ridge_top20_timeseries.csv`
  - `top20_vs_qqq.png`
  - Model-specific bucket return files

---

## 📝 Notes

- Both processes are running normally
- No errors detected
- Processes are actively using CPU and memory (good sign)
- Training typically takes longer than evaluation
- Files will appear in output directories when each stage completes

---

## 🔄 Next Check

Check again in **10-15 minutes** to see:
- Training: Likely in model training phase
- 80/20: Likely in model training or early prediction phase
