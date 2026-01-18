# BMA Ultra System: Complete Train-Test-Backtest Pipeline Documentation

## Overview
This document describes all Python code involved in the complete training, testing, and backtesting pipeline for the BMA Ultra Enhanced Quantitative Model system.

---

## 1. DATA PREPARATION & FEATURE ENGINEERING

### 1.1 Factor Data Generation
**File**: `bma_models/simple_25_factor_engine.py`
- **Responsibility**: 
  - Computes 25 alpha factors (momentum, volatility, quality, value, sentiment, etc.)
  - Generates forward return targets (`ret_fwd_10d` for T+10 prediction)
  - Applies cross-sectional standardization and winsorization
  - Ensures point-in-time (PIT) data integrity (no future leakage)
- **Key Functions**:
  - `get_data_and_features()`: Main entry point for factor computation
  - Cross-sectional normalization per date
  - Target column exclusion from transformations

**File**: `autotrader/factor_export_service.py`
- **Responsibility**:
  - Orchestrates factor data export using Polygon.io API
  - Calls `UltraEnhancedQuantitativeModel.get_data_and_features()`
  - Saves MultiIndex(date, ticker) factor datasets to parquet
- **Output**: `data/factor_exports/factors/factors_all.parquet`

---

## 2. MODEL TRAINING PIPELINE

### 2.1 Main Training Entry Point
**File**: `bma_models/量化模型_bma_ultra_enhanced.py`
- **Class**: `UltraEnhancedQuantitativeModel`
- **Responsibility**:
  - Orchestrates the entire training pipeline
  - Manages configuration from `unified_config.yaml`
  - Coordinates first-layer models (ElasticNet, XGBoost, CatBoost, LightGBM Ranker, LambdaRank)
  - Trains second-layer Meta Ranker Stacker (LightGBM Ranker with LambdaRank objective)
  - Saves model snapshots via `model_registry`
- **Key Methods**:
  - `train_from_document()`: Main training entry point
    - Loads MultiIndex factor data
    - Splits into train/validation with PurgedCV
    - Trains all first-layer models
    - Generates OOF (Out-Of-Fold) predictions
    - Trains Ridge Stacker on OOF predictions
    - Saves snapshot via `model_registry.save_model_snapshot()`
  - `get_data_and_features()`: Delegates to `Simple25FactorEngine`
  - `_unified_model_training()`: Core training orchestration

### 2.2 First-Layer Models

#### 2.2.1 ElasticNet
**File**: `bma_models/量化模型_bma_ultra_enhanced.py` (embedded)
- **Responsibility**:
  - L1+L2 regularized linear regression
  - Feature selection via L1 penalty
  - Trained with PurgedCV (6-fold, gap=10, embargo=10 for T+10)
- **Output**: `pred_elastic` (OOF predictions)

#### 2.2.2 XGBoost
**File**: `bma_models/量化模型_bma_ultra_enhanced.py` (embedded)
- **Responsibility**:
  - Gradient boosting with tree-based learners
  - Handles non-linear feature interactions
  - Trained with PurgedCV
- **Output**: `pred_xgb` (OOF predictions)

#### 2.2.3 CatBoost
**File**: `bma_models/量化模型_bma_ultra_enhanced.py` (embedded)
- **Responsibility**:
  - Gradient boosting optimized for categorical features
  - Robust to overfitting
  - Trained with PurgedCV
- **Output**: `pred_catboost` (OOF predictions)

#### 2.2.4 LightGBM Ranker
**File**: `bma_models/量化模型_bma_ultra_enhanced.py` (embedded)
- **Responsibility**:
  - Pointwise LightGBM ensemble with bagging-style sampling
  - Complements LambdaRank by providing a regression-style gradient booster
  - Shares CV/OOF tracking with other first-layer models
  - Uses deterministic LightGBM configuration (gbdt + bagging)
- **Output**: `pred_lightgbm_ranker` (OOF predictions)

#### 2.2.5 LambdaRank
**File**: `bma_models/lambdarank_stacker.py`
- **Class**: `LambdaRankStacker`
- **Responsibility**:
  - Learning-to-rank algorithm (LightGBM-based)
  - Optimizes pairwise ranking loss (LambdaLoss)
  - Better suited for ranking tasks than pointwise regression
  - Trained with PurgedCV
- **Output**: `pred_lambdarank` (OOF predictions)

### 2.3 Second-Layer Meta-Model

**File**: `bma_models/meta_ranker_stacker.py`
- **Class**: `MetaRankerStacker`
- **Responsibility**:
  - Combines first-layer predictions with a LightGBM Ranker meta-model using LambdaRank objective
  - Optimizes Top-K ranking metrics directly (NDCG@10, NDCG@30)
  - Converts continuous targets to quantile ranks for ranking optimization
  - Base inputs: `pred_catboost`, `pred_elastic`, `pred_xgb`, `pred_lightgbm_ranker`, `pred_lambdarank`
  - Uses PurgedCV for T+5 prediction horizon (6-fold, gap=5, embargo=5)
  - Trained on OOF predictions to prevent leakage
  - Focuses on Top-10/30 performance via NDCG metric
- **Key Methods**:
  - `fit()`: Trains Meta Ranker Stacker on OOF stacker_data with ranking optimization
  - `predict()`: Generates final stock rankings
  - `_convert_to_rank_labels()`: Converts continuous targets to quantile ranks
  - `replace_ewa_in_pipeline()`: Compatibility method for pipeline integration
- **Configuration** (from `unified_config.yaml`):
  - `objective`: "lambdarank"
  - `metric`: "ndcg"
  - `ndcg_eval_at`: [10, 30]
  - `num_boost_round`: 180
  - `learning_rate`: 0.03
  - `num_leaves`: 63
  - `max_depth`: 6
  - `label_gain_power`: 2.2 (emphasizes Top-10 performance)
  - `lambda_l2`: 10.0
- **Output**: `score` (final ranking predictions)

### 2.4 Model Registry & Snapshot Management

**File**: `bma_models/model_registry.py`
- **Responsibility**:
  - Saves/loads model snapshots (all models + metadata)
  - Manages snapshot database (SQLite)
  - Persists Meta Ranker Stacker model and scaler
  - Handles backward compatibility (e.g., loading snapshots with/without CatBoost)
- **Key Functions**:
  - `save_model_snapshot()`: Saves all models, RidgeStacker, LambdaRank, metadata
  - `load_models_from_snapshot()`: Loads models from snapshot ID
  - `load_manifest()`: Loads snapshot metadata (paths, config, training dates)
- **Snapshot Structure**:
  ```
  cache/model_snapshots/YYYYMMDD/<snapshot_id>/
    - manifest.json (metadata)
    - elastic_net.pkl
    - xgboost.json
    - catboost.cbm
    - lambdarank_lgb.txt
    - meta_ranker_txt (LightGBM model file)
    - meta_ranker_scaler.pkl
    - ridge_model.pkl (backward compatibility - old RidgeStacker format)
    - ridge_scaler.pkl (backward compatibility)
    - ridge_meta.json (base_cols, metadata - backward compatibility)
  ```

### 2.5 Configuration Management

**File**: `bma_models/unified_config.yaml`
- **Responsibility**:
  - Central configuration for all models and training parameters
  - Defines Meta Ranker Stacker base_cols and hyperparameters
  - Temporal safety parameters (gap, embargo, horizon)
  - Feature engineering settings
- **Key Sections**:
  - `temporal`: Time-based safety (horizon=10, cv_gap=10, embargo=10)
  - `training.meta_ranker`: Base columns, LambdaRank parameters, and hyperparameters (replaces ridge_stacker)
  - `training.base_models`: Hyperparameters for ElasticNet, XGBoost, CatBoost, LightGBM Ranker, LambdaRank

**File**: `bma_models/unified_config_loader.py`
- **Responsibility**:
  - Loads and validates `unified_config.yaml`
  - Provides `UnifiedTrainingConfig` class
  - Handles environment variable overrides (`BMA_TEMP_CONFIG_PATH`)

---

## 3. TIME-SPLIT OUT-OF-SAMPLE EVALUATION

### 3.0 Time Split Configuration

The system uses a **strict 80/20 time-split** for out-of-sample evaluation:
- **Training Period**: First 80% of available dates (sorted chronologically)
- **Test Period**: Last 20% of available dates
- **Purge Gap**: Equal to `horizon_days` (default 10 days for T+10) to prevent label leakage
- **Configuration**: Default split ratio is `0.8` (configurable via `--split` argument in `time_split_80_20_oos_eval.py`)

This ensures:
- No temporal data leakage between train and test periods
- Realistic evaluation of model performance on unseen future data
- Proper handling of overlapping observations with HAC corrections

### 3.1 80/20 Time-Split Evaluation Script
**File**: `scripts/time_split_80_20_oos_eval.py`
- **Responsibility**:
  - Performs strict 80/20 time-split evaluation (train on first 80% dates, test on last 20%)
  - Default split ratio: `--split 0.8` (configurable via command-line argument)
  - Applies purge gap (= horizon_days) to prevent label leakage
  - Trains models on training window only
  - Runs comprehensive backtest on test window
  - Reports predictive metrics (IC, Rank IC, MSE, MAE, R²) and backtest metrics (Sharpe, turnover, costs)
  - Generates bucket return time series and plots (top/middle/bottom buckets)
  - Computes benchmark returns aligned to the test window (QQQ by default; uses yfinance fallback when needed)
- **Key Functions**:
  - `main()`: Orchestrates train → snapshot → backtest → metrics
  - `_write_model_topn_vs_benchmark()`: Generates per-model time series and plots
  - `_compute_benchmark_tplus_from_yfinance()`: Fetches QQQ benchmark returns
- **Workflow**:
  1. Load factor data (via `ComprehensiveModelBacktest.load_factor_data()`)
  2. Compute date split (80% train, 20% test) with purge gap
  3. Train `UltraEnhancedQuantitativeModel` on train window (uses MetaRankerStacker as second layer)
  4. Save snapshot
  5. Run `ComprehensiveModelBacktest` on test window
  6. Export metrics (IC, Sharpe, net returns) and plots
- **Time Split Configuration**:
  - Default train split: 80% (`--split 0.8`)
  - Default test split: 20% (1 - 0.8)
  - Purge gap: Equal to `horizon_days` (default 10 days for T+10)
  - Ensures no data leakage between train and test periods

#### 3.2 OOS Run Directory (Artifacts)
Each execution writes a self-contained run folder under the chosen `--output-dir`, e.g.:

- `results/extreme_filter_evaluation/run_20260113_054433/`

Typical outputs (per run):
- **Core summaries**
  - `snapshot_id.txt`
  - `oos_metrics.csv` / `oos_metrics.json`
  - `report_df.csv`
  - `results_summary_for_word_doc.json` (single source of truth for Word tables/claims)
- **Per-model predictions**
  - `<model>_predictions_<timestamp>.parquet`
- **Per-model bucket analysis**
  - `<model>_bucket_returns.csv` (includes `benchmark_return` and cumulative columns `cum_*`)
  - `<model>_bucket_summary.csv`
  - `<model>_bucket_returns_period.png`
  - `<model>_bucket_returns_cumulative.png`
- **Per-model Top-N vs benchmark plots**
  - `<model>_top20_vs_qqq.png`
  - `<model>_top20_vs_qqq_cumulative.png`

---

## 4. COMPREHENSIVE BACKTESTING

### 4.1 Main Backtest Engine
**File**: `scripts/comprehensive_model_backtest.py`
- **Class**: `ComprehensiveModelBacktest`
- **Responsibility**:
  - Loads models from snapshot
  - Runs walk-forward backtest with rebalancing
  - Supports multiple models (elastic_net, xgboost, catboost, lambdarank, lightgbm_ranker, ridge_stacking/meta_ranker)
  - Applies transaction costs (bps per turnover)
  - Calculates long-only performance metrics (Top-N Sharpe, win rate, turnover)
  - Generates group returns (Top-10/20/30, Bottom-10/20/30)
- **Key Methods**:
  - `__init__()`: Loads snapshot, initializes models
  - `_load_models()`: Loads models from snapshot (skips if snapshot_id=None for data-only mode)
  - `load_factor_data()`: Loads and standardizes MultiIndex factor data
  - `run_backtest()`: Main backtest loop
    - Iterates through rebalance dates
    - Generates predictions for all models
    - Calculates returns, turnover, costs
    - Aggregates into time series
  - `calculate_group_returns()`: Computes Top-N/Bottom-N returns and metrics
- **Output**:
  - `report_df.csv`: Per-model performance summary
  - `predictions/`: Per-model prediction time series
  - Metrics: IC, Rank IC, MSE, MAE, R², Top-N Sharpe (gross/net), turnover, costs

### 4.2 Data Loading & Standardization

**File**: `scripts/comprehensive_model_backtest.py`
- **Method**: `load_factor_data()`
- **Responsibility**:
  - Loads parquet factor files
  - Calls `_standardize_multiindex()` to convert RangeIndex → MultiIndex(date, ticker)
  - Applies universe filters (if configured)
  - Filters date window (if start_date/end_date specified)
- **Method**: `_standardize_multiindex()`
- **Responsibility**:
  - Converts data with `date`/`ticker` columns to MultiIndex
  - Handles various input formats (RangeIndex, MultiIndex, columns)
  - Ensures proper date/ticker normalization

---

## 5. WALK-FORWARD BACKTESTING (Alternative)

### 5.1 Walk-Forward Top-N Backtest
**File**: `scripts/walkforward_top10.py`
- **Responsibility**:
  - Implements walk-forward backtesting (train on past, test on future)
  - Supports annual retraining mode
  - Applies purge gap to prevent leakage
  - Generates equity curves, top picks, trades, summary metrics
- **Key Features**:
  - `--annual-retrain`: Retrain once per year instead of each rebalance
  - `--min-train-years`: Minimum training window (e.g., 4 years)
  - `--test-years`: Test window size (e.g., 1 year)
  - `--cost-bps`: Transaction cost modeling
  - Outputs `leakage_audit.csv` to verify causality

---

## 6. ARTIFACT GENERATION FOR RESEARCH PAPER

### 6.1 Paper Revision Artifacts Generator
**File**: `scripts/generate_paper_revision_artifacts.py`
- **Responsibility**:
  - Extracts Ridge meta-learner weights from snapshot
  - Generates feature importance lists per model
  - Computes yearly performance statistics
  - Generates return distribution statistics (mean, std, skewness, kurtosis)
  - Creates visualization plots (Ridge weights bar chart)
- **Output**:
  - `ridge_meta_weights.csv`: Ridge stacking coefficients
  - `ridge_meta_weights_top20.png`: Visualization
  - `yearly_stats_all_models.csv`: Year-by-year performance
  - `dist_stats_all_models.csv`: Distribution statistics

### 6.2 Word Document Updater
**File**: `scripts/update_equity_ranking_docx_reviewer.py`
- **Responsibility**:
  - Updates Word document (`Equity Ranking With Ridge Stacking_FINAL.docx`)
  - Embeds latest backtest results (tables, plots)
  - Adds reviewer-requested sections:
    - Theoretical Formalization (LambdaRank Loss, Meta Ranker Stacking equation)
    - Feature Taxonomy
    - SHAP analysis (placeholder)
    - Risk Attribution (Style Exposure, Skew/Kurtosis)
    - Microstructure (Turnover, Capacity, Slippage)
    - Robustness Checks (Yearly Performance, Hyperparameter Sensitivity)
- **Input**: Artifacts directory from `generate_paper_revision_artifacts.py`

### 6.3 Post-processing Plots (Bucket/Benchmark) + Word Insertion Helpers
These scripts are used to create additional publication-ready figures from an existing OOS run folder and insert them into the paper DOCX without re-running training:

- **Bucket vs benchmark (per model, benchmark-visible)**
  - `scripts/plot_time_split_bucket_cumulative_vs_benchmark.py`
    - Produces: `<model>_bucket_accum_compare_vs_QQQ.png`
    - Design: wealth-index on log scale + cumulative excess return vs benchmark
  - `scripts/insert_bucket_cumulative_vs_benchmark_into_doc.py`
    - Inserts the per-model bucket-vs-benchmark figures into the DOCX at the results section (before “Table 3”)
- **$1,000,000 equity curves by bucket (all models vs QQQ)**
  - `scripts/plot_equity_curves_all_models_by_bucket.py`
    - Produces:
      - `all_models_equity_by_bucket_top_net_vs_QQQ.png`
      - `all_models_equity_by_bucket_bottom_gross_vs_QQQ.png`
  - `scripts/insert_all_models_equity_curve_into_doc.py`
    - Inserts both top+bottom equity-curve figures into the DOCX (and uses a short output filename to avoid Windows MAX_PATH issues)
- **DOCX hygiene and labeling**
  - `scripts/remove_duplicate_tables_graphs.py` (deduplicate repeated tables/paragraphs/images after multiple runs)
  - `scripts/verify_and_label_oos_oof.py` (OOS/OOF labels + disclosure + model-name standardization)

---

## 7. SUPPORTING UTILITIES

### 7.1 Purged Cross-Validation
**File**: `bma_models/unified_purged_cv_factory.py`
- **Responsibility**:
  - Creates PurgedCV splits for time-series data
  - Prevents data leakage via gap and embargo periods
  - Used by all first-layer models and Ridge Stacker
- **Key Function**: `create_unified_cv(n_splits, gap, embargo)`

### 7.2 Polygon.io Data Client
**File**: `bma_models/polygon_client.py` / `polygon_client.py`
- **Responsibility**:
  - Fetches historical stock data from Polygon.io API
  - Uses `ffill()` only (no `bfill()` to prevent future leakage)
  - Handles API rate limiting and errors

### 7.3 Results Pack Builder
**File**: `scripts/build_results_pack_artifacts.py`
- **Responsibility**:
  - Generates `results_pack_inventory.csv` and `results_pack_summary.json`
  - Tracks all backtest runs and their metrics
  - Used for research paper reproducibility

---

## 8. COMPLETE WORKFLOW SUMMARY

### Phase 1: Data Preparation
1. **Factor Generation**: `simple_25_factor_engine.py` → `factors_all.parquet`
2. **Data Export**: `factor_export_service.py` orchestrates export

### Phase 2: Model Training
1. **Main Training**: `量化模型_bma_ultra_enhanced.py.train_from_document()`
   - Loads factor data
   - Trains ElasticNet, XGBoost, CatBoost, LightGBM Ranker, LambdaRank (first layer)
   - Generates OOF predictions
   - Trains Ridge Stacker (second layer) on OOF predictions
2. **Snapshot Save**: `model_registry.save_model_snapshot()` saves all models + metadata

### Phase 3: Out-of-Sample Evaluation
1. **Time-Split Eval**: `time_split_80_20_oos_eval.py`
   - Trains on first 80% dates (with purge gap)
   - Tests on last 20% dates
   - Runs `ComprehensiveModelBacktest` on test window
   - Exports metrics and plots

### Phase 4: Comprehensive Backtesting
1. **Backtest Engine**: `comprehensive_model_backtest.py`
   - Loads snapshot
   - Iterates through rebalance dates
   - Generates predictions for all models
   - Calculates returns, turnover, costs
   - Aggregates metrics (IC, Sharpe, turnover, etc.)

### Phase 5: Artifact Generation
1. **Paper Artifacts**: `generate_paper_revision_artifacts.py`
   - Extracts Meta Ranker Stacker model info, feature lists, yearly stats
2. **Word Update**: `update_equity_ranking_docx_reviewer.py`
   - Embeds results into Word document

---

## 9. KEY CONFIGURATION FILES

- **`bma_models/unified_config.yaml`**: Central configuration
  - Meta Ranker Stacker: `base_cols: [pred_catboost, pred_elastic, pred_xgb, pred_lightgbm_ranker, pred_lambdarank]`, `objective: lambdarank`, `ndcg_eval_at: [10, 30]`, `label_gain_power: 2.2`
  - Temporal: `horizon=10`, `cv_gap=10`, `embargo=10`
  - Time Split: 80/20 train/test split (default `--split 0.8` in `time_split_80_20_oos_eval.py`)
- **`data/factor_exports/factors/factors_all.parquet`**: Factor dataset (MultiIndex format)

---

## 10. COMMAND-LINE INTERFACES

### Training
```bash
python -m bma_models.量化模型_bma_ultra_enhanced --start-date 2020-01-01 --end-date 2024-12-31
```

### Time-Split Evaluation

**Standard Command (Recommended)**:
```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "data/factor_exports/polygon_factors_all_filtered.parquet" \
  --horizon-days 10 \
  --split 0.8 \
  --models elastic_net xgboost catboost lightgbm_ranker lambdarank ridge_stacking \
  --model ridge_stacking \
  --top-n 20 \
  --cost-bps 10 \
  --benchmark QQQ \
  --output-dir "results/t10_time_split_80_20_new_params" \
  --log-level INFO
```

**Alternative Data File** (legacy path):
```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file data/factor_exports/factors/factors_all.parquet \
  --horizon-days 10 --split 0.8 \
  --models elastic_net xgboost catboost lambdarank ridge_stacking \
  --model ridge_stacking --top-n 30 \
  --cost-bps 10 --benchmark QQQ \
  --output-dir results/t10_time_split_test20
```

### Plot “Bucket vs QQQ” (benchmark-visible) from an existing run folder
```bash
python scripts/plot_time_split_bucket_cumulative_vs_benchmark.py \
  --run-dir results/extreme_filter_evaluation/run_20260113_054433 \
  --benchmark QQQ --horizon-days 10
```

### Plot $1,000,000 Equity Curves (all models) from an existing run folder
```bash
python scripts/plot_equity_curves_all_models_by_bucket.py \
  --run-dir results/extreme_filter_evaluation/run_20260113_054433 \
  --benchmark QQQ --start-capital 1000000 --log-scale
```

### Insert figures into the paper DOCX (post-processing)
```bash
# 1) Insert per-model bucket-vs-QQQ figures
python scripts/insert_bucket_cumulative_vs_benchmark_into_doc.py \
  --input-doc "D:\trade\<your_latest_doc>.docx" \
  --run-dir results/extreme_filter_evaluation/run_20260113_054433 \
  --benchmark QQQ

# 2) Insert all-models $1,000,000 equity curves (top+bottom buckets)
python scripts/insert_all_models_equity_curve_into_doc.py \
  --input-doc "D:\trade\<your_latest_doc>.docx" \
  --top-figure-path results/extreme_filter_evaluation/run_20260113_054433/all_models_equity_by_bucket_top_net_vs_QQQ.png \
  --bottom-figure-path results/extreme_filter_evaluation/run_20260113_054433/all_models_equity_by_bucket_bottom_gross_vs_QQQ.png \
  --benchmark QQQ
```

### Comprehensive Backtest
```bash
python scripts/comprehensive_model_backtest.py \
  --snapshot-id <snapshot_id> \
  --data-file data/factor_exports/factors/factors_all.parquet \
  --start-date 2024-01-01 --end-date 2024-12-31 \
  --cost-bps 10
```

---

## 11. DATA FLOW DIAGRAM

```
factors_all.parquet (MultiIndex)
    ↓
UltraEnhancedQuantitativeModel.train_from_document()
    ↓
Simple25FactorEngine.get_data_and_features()
    ↓
PurgedCV splits (6-fold, gap=10, embargo=10)
    ↓
First-Layer Models (ElasticNet, XGBoost, CatBoost, LightGBM Ranker, LambdaRank)
    ↓
OOF Predictions (pred_elastic, pred_xgb, pred_catboost, pred_lightgbm_ranker, pred_lambdarank)
    ↓
MetaRankerStacker.fit() on OOF predictions (LightGBM Ranker with LambdaRank objective)
    ↓
model_registry.save_model_snapshot()
    ↓
Snapshot (all models + metadata)
    ↓
ComprehensiveModelBacktest.run_backtest()
    ↓
Predictions → Returns → Metrics (IC, Sharpe, turnover, costs)
    ↓
generate_paper_revision_artifacts.py
    ↓
update_equity_ranking_docx_reviewer.py
    ↓
Word Document (with embedded results)
```

---

## 12. CRITICAL DESIGN DECISIONS

1. **Point-in-Time (PIT) Data**: All data transformations exclude forward-looking information
2. **PurgedCV**: Gap + embargo periods prevent label leakage
3. **OOF Predictions**: Meta Ranker Stacker trained on out-of-fold predictions to prevent leakage
4. **Ranking Optimization**: Meta Ranker Stacker optimizes Top-K ranking (NDCG@10, NDCG@30) instead of continuous regression
5. **Long-Only Strategy**: All backtests focus on Top-N long positions (no shorting)
6. **Transaction Costs**: Applied as `turnover * cost_bps / 1e4` per rebalance
7. **Time-Split Evaluation**: Strict 80/20 split (`--split 0.8` default) with purge gap for robust OOS testing

---

## 13. FILE REFERENCE TABLE

| File | Responsibility | Key Classes/Functions |
|------|----------------|---------------------|
| `bma_models/simple_25_factor_engine.py` | Factor computation | `Simple25FactorEngine.get_data_and_features()` |
| `bma_models/量化模型_bma_ultra_enhanced.py` | Main training orchestration | `UltraEnhancedQuantitativeModel.train_from_document()` |
| `bma_models/meta_ranker_stacker.py` | Meta Ranker Stacker (LightGBM Ranker) | `MetaRankerStacker.fit()`, `MetaRankerStacker.predict()` |
| `bma_models/ridge_stacker.py` | Ridge meta-model (deprecated, backward compatibility) | `RidgeStacker.fit()`, `RidgeStacker.predict()` |
| `bma_models/lambdarank_stacker.py` | LambdaRank model | `LambdaRankStacker.fit()`, `LambdaRankStacker.predict()` |
| `bma_models/model_registry.py` | Snapshot management | `save_model_snapshot()`, `load_models_from_snapshot()` |
| `bma_models/unified_config.yaml` | Central configuration | Ridge base_cols, alpha, temporal params |
| `scripts/time_split_80_20_oos_eval.py` | 80/20 time-split evaluation | `main()`, `_write_model_topn_vs_benchmark()` |
| `scripts/comprehensive_model_backtest.py` | Comprehensive backtest engine | `ComprehensiveModelBacktest.run_backtest()` |
| `scripts/walkforward_top10.py` | Walk-forward backtest | Annual retraining, leakage audit |
| `scripts/generate_paper_revision_artifacts.py` | Paper artifact generation | Ridge weights, yearly stats, distributions |
| `scripts/update_equity_ranking_docx_reviewer.py` | Word document updater | Embeds tables, plots, sections |
| `scripts/plot_time_split_bucket_cumulative_vs_benchmark.py` | Bucket vs benchmark plots | per-model benchmark-visible bucket figures |
| `scripts/insert_bucket_cumulative_vs_benchmark_into_doc.py` | Insert bucket vs benchmark into DOCX | inserts before “Table 3” in results section |
| `scripts/plot_equity_curves_all_models_by_bucket.py` | $1,000,000 equity curves (all models) | top/bottom bucket equity curves vs QQQ |
| `scripts/insert_all_models_equity_curve_into_doc.py` | Insert equity curves into DOCX | inserts top+bottom equity figures; avoids MAX_PATH |
| `scripts/remove_duplicate_tables_graphs.py` | DOCX deduplication | removes duplicated tables/paragraphs/images |
| `scripts/verify_and_label_oos_oof.py` | OOS/OOF verification + labeling | adds disclosure and standardizes model names |

---

## END OF DOCUMENTATION
