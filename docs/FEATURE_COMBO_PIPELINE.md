# Feature Combination Grid Search Pipeline - Complete Process Documentation

## Overview
This pipeline tests different feature combinations across 4 models (elastic_net, xgboost, catboost, lambdarank) to find the optimal feature subset for T+10 prediction.

---

## 📋 Quick Reference: Python Files & Locations

| Script | Location | Purpose | Inputs | Outputs |
|--------|----------|---------|--------|---------|
| `t10_feature_selection.py` | `scripts/` | RankIC-based feature selection | `factors_all.parquet` | `feature_ic_summary.csv`, `feature_corr.csv`, `selected_features.txt` |
| `t10_feature_combo_grid_search_all_models.py` | `scripts/` | Stage-1: Fast ranking of all 8192 subsets | IC summary, correlation matrix | `stage1_ranked_combos.csv` |
| `t10_feature_combo_grid_search_parallel_models.py` | `scripts/` | **Stage-2: Current active process** - Train+backtest top 500 combos | `stage1_ranked_combos.csv` | `combo_results.csv`, `combo_summary.csv` |
| `train_single_model.py` | `scripts/` | Train one model with feature subset | Training data, feature list | Snapshot ID |
| `comprehensive_model_backtest.py` | `scripts/` | Backtest a trained snapshot | Snapshot ID, factor data | `performance_report_*.csv` |
| `量化模型_bma_ultra_enhanced.py` | `bma_models/` | Core model implementation | Config YAML, factor data | Trained model snapshots |
| `simple_25_factor_engine.py` | `bma_models/` | T+10 factor computation | Raw OHLCV data | T+10 factors (MultiIndex) |

---

## 📁 File Locations & Roles

### 1. **Feature Selection (Stage 0)**
**File**: `scripts/t10_feature_selection.py`  
**Purpose**: Pre-selects candidate features based on RankIC and de-correlation

**Inputs**:
- `data/factor_exports/factors/factors_all.parquet` (or shard directory)
- T+10 factors: `liquid_momentum`, `obv_divergence`, `ivol_20`, `rsi_21`, `trend_r2_60`, `near_52w_high`, `ret_skew_20d`, `blowoff_ratio`, `hist_vol_40d`, `atr_ratio`, `bollinger_squeeze`, `vol_ratio_20d`, `price_ma60_deviation`

**Outputs** (saved to `results/t10_feature_selection/`):
- `feature_ic_summary.csv`: Per-feature RankIC statistics (mean, std, t-stat, win-rate)
- `feature_corr.csv`: Feature-feature Pearson correlation matrix
- `selected_features.txt`: Greedy-selected feature list (de-correlated, one per line)

**Process**:
1. Loads factor dataset (MultiIndex or flat)
2. Samples up to `--max-dates` dates uniformly
3. For each date, computes Spearman RankIC between each feature and `target` (T+10 return)
4. Summarizes IC across dates (mean, std, t-stat, win-rate)
5. Computes correlation matrix (sampled `--corr-sample-rows` rows)
6. Greedy selection: starts with highest |RankIC|, adds features if correlation < `--corr-max` (default 0.90)

---

### 2. **Stage-1: Fast Ranking (Optional)**
**File**: `scripts/t10_feature_combo_grid_search_all_models.py` (with `--stage1-only`)  
**Purpose**: Scores ALL possible feature subsets (2^13 = 8192) without training

**Inputs**:
- `results/t10_feature_selection/selected_features.txt` (candidates)
- `results/t10_feature_selection/feature_ic_summary.csv` (RankIC weights)
- `results/t10_feature_selection/feature_corr.csv` (correlation matrix)

**Outputs** (saved to `results/t10_feature_combo_grid_all_ranked_full/`):
- `stage1_ranked_combos.csv`: All subsets ranked by score (descending)
  - Columns: `score`, `features` (JSON array), `k` (subset size)

**Process**:
1. Loads candidates from `selected_features.txt`
2. Enumerates all subsets of size `k_min` to `k_max` (default 8-13)
3. For each subset:
   - Score = sum(|RankIC|) - penalty × (sum of excess correlations above threshold)
   - Penalty weight: `--score-corr-penalty` (default 0.25)
4. Ranks all subsets by score
5. Saves to CSV (no training/backtest)

**Compulsory Features** (always included unless `--no-compulsory`):
- `obv_divergence`, `ivol_20`, `rsi_21`, `near_52w_high`, `trend_r2_60`

---

### 3. **Stage-2: Training + Backtesting (Current Active Process)**
**File**: `scripts/t10_feature_combo_grid_search_parallel_models.py`  
**Purpose**: For each feature combo, trains all 4 models in parallel, then backtests each

**Inputs**:
- `results/t10_feature_combo_grid_all_ranked_full/stage1_ranked_combos.csv` (ranked subsets)
- `data/factor_exports/factors/factors_all.parquet` (backtest data)
- `data/factor_exports/factors/` (training data directory with MultiIndex shards)
- `bma_models/unified_config.yaml` (base model config)

**Outputs** (saved to `results/t10_feature_combo_grid_parallel_run500_no_overlap/`):
- `combo_results.csv`: One row per (combo, model) with `avg_top_return`
- `combo_summary.csv`: One row per combo with per-model metrics
- `runs/combo_XXXX/model_name/`: Per-combo per-model training snapshots and backtest reports

**Process** (for each feature combo):
1. **Parallel Training** (4 processes):
   - Calls `scripts/train_single_model.py` for each model (elastic_net, xgboost, catboost, lambdarank)
   - Each model trains with the feature subset via `BMA_FEATURE_OVERRIDES` env var
   - Saves snapshot ID to `runs/combo_XXXX/model_name/snapshot_id.txt`

2. **Parallel Backtesting** (4 processes):
   - Calls `scripts/comprehensive_model_backtest.py` for each snapshot
   - Uses `--rebalance-mode horizon --target-horizon-days 10` (non-overlapping: rebalance every 10 trading days)
   - Extracts `avg_top_return` from `performance_report_*.csv`

3. **Aggregation**:
   - Collects `avg_top_return` for each (combo, model)
   - Incrementally writes `combo_results.csv` and `combo_summary.csv`

---

## 🔄 Complete Process Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 0: Feature Selection                                       │
│ File: scripts/t10_feature_selection.py                          │
│ Input: data/factor_exports/factors/factors_all.parquet          │
│ Output: results/t10_feature_selection/                          │
│   - feature_ic_summary.csv (RankIC stats)                       │
│   - feature_corr.csv (correlation matrix)                       │
│   - selected_features.txt (candidate list)                      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Stage-1 Fast Ranking (Optional)                         │
│ File: scripts/t10_feature_combo_grid_search_all_models.py      │
│   with --stage1-only                                            │
│ Input: results/t10_feature_selection/*                          │
│ Output: results/t10_feature_combo_grid_all_ranked_full/        │
│   - stage1_ranked_combos.csv (all 8192 subsets ranked)          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Stage-2 Training + Backtesting (CURRENT RUN)            │
│ File: scripts/t10_feature_combo_grid_search_parallel_models.py │
│ Input: stage1_ranked_combos.csv (top 500)                       │
│ Process:                                                        │
│   For each combo (top 500):                                     │
│     ├─ Train elastic_net  ─┐                                   │
│     ├─ Train xgboost       ├─→ Parallel (4 processes)           │
│     ├─ Train catboost      │                                   │
│     └─ Train lambdarank   ─┘                                   │
│     ├─ Backtest elastic_net  ─┐                               │
│     ├─ Backtest xgboost        ├─→ Parallel (4 processes)       │
│     ├─ Backtest catboost       │                               │
│     └─ Backtest lambdarank   ─┘                               │
│ Output: results/t10_feature_combo_grid_parallel_run500_no_overlap/│
│   - combo_results.csv (one row per combo×model)                │
│   - combo_summary.csv (one row per combo)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Key Metrics Explained

### `avg_top_return`
- **Definition**: Average T+10 return (%) of Top 30 predicted stocks
- **Calculation**: 
  1. For each rebalance date (every 10 trading days), rank stocks by model prediction
  2. Take Top 30 stocks
  3. Compute actual T+10 return (`target` column) for those 30
  4. Average across all rebalance dates
- **Format**: Decimal (0.0676 = 6.76% per T+10 period)
- **Annualized** (non-overlapping): `(1 + avg_top_return)^(252/10) - 1`

---

## 🔧 Supporting Scripts

### `scripts/train_single_model.py`
**Purpose**: Train a single model with specific feature subset  
**Called by**: `t10_feature_combo_grid_search_parallel_models.py`  
**Process**:
1. Creates temporary config with parameter overrides
2. Sets `BMA_FEATURE_OVERRIDES` env var (JSON dict: `{model_name: [feature_list]}`)
3. Instantiates `UltraEnhancedQuantitativeModel`
4. Calls `model.train_from_document(training_data_path)`
5. Writes snapshot ID to `--output-file`

**Key Environment Variables**:
- `BMA_FEATURE_OVERRIDES`: JSON dict of per-model feature lists
- `BMA_TRAIN_ONLY_MODEL`: Limits training to one first-layer model (for speed)
- `BMA_TEMP_CONFIG_PATH`: Path to temporary YAML config

---

### `scripts/comprehensive_model_backtest.py`
**Purpose**: Backtest a trained snapshot on historical data  
**Called by**: `t10_feature_combo_grid_search_parallel_models.py`  
**Process**:
1. Loads snapshot (model + config)
2. Gets rebalance dates:
   - `--rebalance-mode weekly`: Every Monday (overlapping with T+10 target)
   - `--rebalance-mode horizon`: Every 10 trading days (non-overlapping)
3. For each rebalance date:
   - Extracts features for that date
   - Runs model prediction
   - Records `actual` = `target` (T+10 forward return)
4. Computes metrics:
   - `avg_top_return`: Mean of Top 30 `actual` returns
   - IC, Rank IC, MSE, MAE, R²
   - Bucket returns (Top 1-10, 11-20, ..., Bottom 1-10, ...)
5. Saves `performance_report_*.csv`

---

## 🎯 Current Running Configuration

**Active Process**: PID 37336  
**Script**: `scripts/t10_feature_combo_grid_search_parallel_models.py`  
**Arguments**:
```bash
--stage1-file results/t10_feature_combo_grid_all_ranked_full/stage1_ranked_combos.csv
--top-combos 500
--models elastic_net xgboost catboost lambdarank
--data-file data/factor_exports/factors/factors_all.parquet
--train-data data/factor_exports/factors
--max-weeks 260
--max-parallel-models 4
--rebalance-mode horizon
--target-horizon-days 10
--output-dir results/t10_feature_combo_grid_parallel_run500_no_overlap
```

**Status**: Processing combo_0003 (as of last log check)  
**Progress**: 
- ✅ combo_0001: All 4 models complete
- ✅ combo_0002: All 4 models complete  
- 🔄 combo_0003: All 4 models complete (processing combo_0004)
- ⏳ combo_0004-0500: Pending

**Output Files** (incremental updates):
- `results/t10_feature_combo_grid_parallel_run500_no_overlap/combo_results.csv` (one row per combo×model)
- `results/t10_feature_combo_grid_parallel_run500_no_overlap/combo_summary.csv` (one row per combo)

**Example Results** (from combo_0001):
- `elastic_net`: `avg_top_return=0.0221` (2.21% per T+10)
- `xgboost`: `avg_top_return=0.0594` (5.94% per T+10) ⭐ **Best so far**
- `catboost`: `avg_top_return=0.0299` (2.99% per T+10)
- `lambdarank`: `avg_top_return=0.0260` (2.60% per T+10)

---

## 📈 Data Flow Summary

1. **Factor Data**: `data/factor_exports/factors/factors_all.parquet` (3M+ rows, 17 cols)
   - MultiIndex: `(date, ticker)`
   - Columns: 13 T+10 factors + `Close` + `target` (T+10 forward return)

2. **Feature Selection**: Computes RankIC per date → selects de-correlated candidates

3. **Stage-1 Ranking**: Scores all 8192 subsets → ranks by (RankIC sum - correlation penalty)

4. **Stage-2 Training**: Top 500 combos → 4 models × 500 combos = 2000 training jobs (parallelized per combo)

5. **Stage-2 Backtesting**: 2000 snapshots → 2000 backtests (non-overlapping, 260 weeks each)

6. **Results**: `combo_results.csv` with `avg_top_return` for each (combo, model) pair

---

## 🔍 How to Interpret Results

**Best Combo Per Model** (from `combo_results.csv`):
```python
import pandas as pd
df = pd.read_csv('results/.../combo_results.csv')
best = df.sort_values('avg_top_return', ascending=False).groupby('model').first()
```

**Example Output**:
- `xgboost combo_0060`: `avg_top_return=0.0676` → **6.76% per T+10 period**
- Annualized (non-overlapping): `(1.0676)^(252/10) - 1 ≈ 419.58% / year`

**Note**: These are **non-overlapping** returns (rebalance every 10 trading days), so annualization is valid.

---

## 🛠️ Key Fixes Applied

1. **Overlapping Problem Fixed**: Changed from weekly rebalance to horizon-based (every 10 trading days)
2. **Memory Optimization**: Training reads from MultiIndex shard directory instead of flat parquet
3. **Parallel Per Combo**: All 4 models train+backtest simultaneously for each feature combo
4. **Feature Override Fix**: `BMA_FEATURE_OVERRIDES` now correctly applied after base config (prevents overwrite)

---

## 📝 Next Steps After Completion

1. **Analyze Results**: Find best combo per model from `combo_summary.csv`
2. **Apply Best Features**: Update `bma_models/unified_config.yaml` and `量化模型_bma_ultra_enhanced.py` with winning feature lists
3. **Re-train Production Model**: Train final model with best feature combo
4. **Full Backtest**: Run comprehensive backtest on full 260 weeks with best combo

