# Pipeline Overview

## Entry / Ticker Ingestion
- `load_tickers_from_file()` (`scripts/test_simple25_polygon.py:35`) loads tickers from a txt file (`bma_models/default_tickers.txt`)
- `determine_tickers()` (`scripts/test_simple25_polygon.py:52`) determines tickers from CLI args or file
- `fetch_and_compute_factors()` (`scripts/test_simple25_polygon.py:64`) fetches market data and computes factors
- Running with `--run-analysis` hands control to `UltraEnhancedQuantitativeModel.run_complete_analysis()` (`bma_models/量化模型_bma_ultra_enhanced.py:12285`) so the full pipeline executes automatically once tickers are selected

## Data Fetch + Feature Generation
- `Simple17FactorEngine.fetch_market_data()` (`bma_models/simple_25_factor_engine.py:178`) fetches Polygon market data (optimized downloader, legacy fallback) for the ticker list and returns a row-per-(date,ticker) table
- `Simple17FactorEngine.compute_all_17_factors()` (`bma_models/simple_25_factor_engine.py:268`) computes the **T10 alpha features** (14 factors), adds `Close`, and enforces a strict `MultiIndex(['date','ticker'])` before optional sentiment enrichment. This is the "get data, calculate features, into MultiIndex" stage
- **Factor Set**: Always uses `T10_ALPHA_FACTORS` (14 factors) regardless of prediction horizon. T5 factors have been completely removed (2026-01-24 update).
- **T10 Alpha Factors**: `liquid_momentum`, `momentum_10d`, `momentum_60d`, `obv_divergence`, `obv_momentum_60d`, `ivol_20`, `hist_vol_40d`, `atr_ratio`, `rsi_21`, `trend_r2_60`, `near_52w_high`, `vol_ratio_20d`, `price_ma60_deviation`, `5_days_reversal`
- **MultiIndex Alignment Fix**: Failed factor computations now use `pd.Series(0.0, index=data.index, name='factor_name')` instead of `np.zeros(len(data))` to ensure proper MultiIndex alignment (2026-01-24 fix)

## MultiIndex Normalization
- `UnifiedFeaturePipeline._ensure_multiindex_format()` (`bma_models/unified_feature_pipeline.py:266`) converts any `(date,ticker)` columns into a proper MultiIndex for feature processing
- `_load_training_data_from_file()` (`bma_models/量化模型_bma_ultra_enhanced.py:7969`) reads cached parquet/pickle shards
- `_standardize_loaded_data()` (`bma_models/量化模型_bma_ultra_enhanced.py:8072`) standardizes the loaded data format
- `_ensure_standard_feature_index()` (`bma_models/量化模型_bma_ultra_enhanced.py:8160`) normalizes them to the same MultiIndex format

## Training Data Location (MultiIndex Format)

**Primary Training Data File** (Standard Path):
```
data/factor_exports/polygon_factors_all_filtered.parquet
```

**Alternative Data Files**:
- `data/factor_exports/factors/factors_all.parquet` (legacy, still supported)
- Any parquet file with MultiIndex(date, ticker) format

**Format Requirements**:
- **MultiIndex**: `(date, ticker)` - Required format for all training and prediction
- **Date Index**: First level must be datetime (date)
- **Ticker Index**: Second level must be string (ticker/symbol)
- **Data Quality**: Filtered dataset with:
  - Extreme volatility filter (hist_vol_40d > P99.9 removed)
  - Low price filter (Close < 5 USD removed)
  - Recalculated `downside_beta_ewm_21`

## Parameter Configuration Location

**All model parameters are centralized in**:
```
bma_models/unified_config.yaml
```

**Key Parameter Sections**:

### First Layer: LambdaRankStacker
```yaml
training:
  base_models:
    lambdarank:
      num_boost_round: 500
      early_stopping_rounds: 70
      min_data_in_leaf: 800
      feature_fraction: 0.7
      lambdarank_truncation_level: 300
      label_gain_power: 2.3
      # ... (see unified_config.yaml lines 263-282)
```

### Second Layer: MetaRankerStacker
```yaml
training:
  meta_ranker:
    num_boost_round: 300
    early_stopping_rounds: 50
    num_leaves: 31
    max_depth: 4
    min_data_in_leaf: 1300
    lambda_l2: 50.0
    lambdarank_truncation_level: 300
    label_gain_power: 2.2
    # ... (see unified_config.yaml lines 286-322)
```

**Parameter Loading**:
- Parameters are loaded in `量化模型_bma_ultra_enhanced.py:676-723` (`__init__` method)
- Fallback defaults are synchronized in model classes:
  - `lambda_rank_stacker.py` (defaults)
  - `meta_ranker_stacker.py` (defaults)

## Training (BMA Ultra Mode)

### Standard Training Flow
- `run_complete_analysis()` (`bma_models/量化模型_bma_ultra_enhanced.py:12285`) orchestrates `train/predict/full` modes and defaults to the MultiIndex export for training if no path is provided.
- `train_from_document()` (`bma_models/量化模型_bma_ultra_enhanced.py:8327`) loads the MultiIndex dataset, optionally filters dates/universe, and hands the frame to `_run_training_phase()` (`bma_models/量化模型_bma_ultra_enhanced.py:8240`).
- `_run_training_phase()` invokes `train_enhanced_models()` (`bma_models/量化模型_bma_ultra_enhanced.py:9131`), which calls `_execute_modular_training()` (`bma_models/量化模型_bma_ultra_enhanced.py:9135`) to train first-layer models (XGB, CatBoost, ElasticNet, LightGBM Ranker, LambdaRank) and then calls `_train_ridge_stacker()` (`bma_models/量化模型_bma_ultra_enhanced.py:10207`) which **actually trains MetaRankerStacker** (function name is legacy), storing the results for prediction.

**Important**: `_train_ridge_stacker()` is a legacy function name. It actually instantiates and trains **MetaRankerStacker** (see `bma_models/量化模型_bma_ultra_enhanced.py:10454-10484`). No RidgeStacker is used in training or prediction.

### Two-Layer Architecture

**First Layer Models** (Purged Cross-Validation):
1. **ElasticNet**: Linear baseline with L1/L2 regularization
2. **XGBoost**: Gradient boosting with optimized hyperparameters
3. **CatBoost**: Categorical boosting for robust feature handling
4. **LightGBM Ranker**: Ranking-focused gradient boosting
5. **LambdaRankStacker**: LambdaRank with ranking optimization
   - Input: **T10 alpha factors** (14 features: `T10_ALPHA_FACTORS`)
   - Output: `pred_lambdarank` (first-layer prediction)
   - Parameters: `num_boost_round=500`, `early_stopping_rounds=70`, `min_data_in_leaf=800`, `feature_fraction=0.7`, `lambdarank_truncation_level=300`, `label_gain_power=2.3`
   - **Note**: Always uses T+10 factors regardless of prediction horizon (T5 factors removed, 2026-01-24)

**Second Layer Meta-Learner**:
- **MetaRankerStacker** (LightGBM Ranker with LambdaRank objective)
  - Input: First-layer predictions (`pred_catboost`, `pred_elastic`, `pred_xgb`, `pred_lightgbm_ranker`, `pred_lambdarank`)
  - Output: Final ranking scores
  - Parameters: `num_boost_round=300`, `early_stopping_rounds=50`, `num_leaves=31`, `max_depth=4`, `min_data_in_leaf=1300`, `lambda_l2=50.0`, `lambdarank_truncation_level=300`, `label_gain_power=2.2`
  - Optimization: NDCG@10 and NDCG@30 (ranking-focused)

## 80/20 Time Split Out-of-Sample Evaluation

### Script Location
```
scripts/time_split_80_20_oos_eval.py
```

### Purpose
Strict 80/20 time-split evaluation for out-of-sample testing:
- **Training**: First 80% of dates (chronologically sorted)
- **Testing**: Last 20% of dates (OOS evaluation)
- **Purge Gap**: 10 days (equal to horizon_days) to prevent label leakage

### Command

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

**Alternative Data File Path** (if using filtered dataset):
```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "D:\trade\data\factor_exports\polygon_factors_all_filtered.parquet" \
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

**Note**: `ridge_stacking` is a legacy identifier for backward compatibility. Internally, it uses **MetaRankerStacker** (LightGBM Ranker with LambdaRank objective).

### Process Flow

1. **Data Loading**:
   - Loads MultiIndex parquet file: `polygon_factors_all_filtered.parquet` or `polygon_factors_all_filtered_clean_final_v2.parquet`
   - Validates MultiIndex format: `(date, ticker)`
   - Sorts by date for chronological split
   - **Factor Set**: Uses `T10_ALPHA_FACTORS` (14 factors) - T5 factors removed (2026-01-24)

2. **Time Split**:
   - Computes unique dates and sorts chronologically
   - Split index: `int(n_dates * 0.8)` (80% training)
   - Purge gap: `split_idx - 1 - horizon_days` (prevents label leakage)
   - Training period: `dates[0]` to `dates[train_end_idx]`
   - Testing period: `dates[split_idx]` to `dates[-1]`

3. **Training Phase** (`scripts/time_split_80_20_oos_eval.py:788-902`):
   - Trains all first-layer models on training period via `UltraEnhancedQuantitativeModel.train_enhanced_models()`:
     - ElasticNet
     - XGBoost
     - CatBoost
     - LightGBM Ranker
     - **LambdaRankStacker** (with new parameters: 500 rounds, 70 early stopping)
   - **Factor Input**: All first-layer models use `T10_ALPHA_FACTORS` (14 factors) - T5 factors removed (2026-01-24)
   - Trains **MetaRankerStacker** via `MetaRankerStacker.fit()` (`scripts/time_split_80_20_oos_eval.py:863-891`) with new parameters: 300 rounds, 50 early stopping, 31 leaves
   - Uses PurgedCV (6-fold, gap=5, embargo=5) for temporal safety
   - Saves model snapshot via `save_model_snapshot()` (`scripts/time_split_80_20_oos_eval.py:894-901`)

**Important**: The training code instantiates `MetaRankerStacker` (`scripts/time_split_80_20_oos_eval.py:863`) and saves it. No RidgeStacker is instantiated or trained.

4. **Testing Phase** (`scripts/time_split_80_20_oos_eval.py:933-1171`):
   - Loads snapshot via `load_models_from_snapshot()` (`scripts/time_split_80_20_oos_eval.py:933`) - loads MetaRankerStacker (checks for `lightgbm_model` attribute, `scripts/time_split_80_20_oos_eval.py:940-941`)
   - Runs predictions on test window (last 20% dates) via prediction loop (`scripts/time_split_80_20_oos_eval.py:1095-1171`)
   - MetaRankerStacker prediction: `ridge_stacker.predict()` (`scripts/time_split_80_20_oos_eval.py:1144`) - returns DataFrame with 'score' column (`scripts/time_split_80_20_oos_eval.py:1148-1152`)
   - Evaluates all models including MetaRankerStacker
   - Calculates metrics:
     - IC (Information Coefficient)
     - Rank IC
     - MSE, MAE, R²
     - NDCG@10, NDCG@30 (for ranking models)
   - Generates bucket returns:
     - Top-20 (highest predicted returns)
     - Middle buckets
     - Bottom buckets
   - Compares against QQQ benchmark
   - Applies transaction costs (10 bps)

**Important**: The prediction code checks for `lightgbm_model` attribute (`scripts/time_split_80_20_oos_eval.py:940`) to identify MetaRankerStacker. If `ridge_model` exists instead, it would be old RidgeStacker (which is no longer supported). The code only uses MetaRankerStacker for predictions.

5. **Outputs** (in `results/t10_time_split_80_20_new_params/run_YYYYMMDD_HHMMSS/`):
   - `snapshot_id.txt` - Model snapshot ID
   - `oos_metrics.json` / `oos_metrics.csv` - Out-of-sample metrics
   - `report_df.csv` - Performance summary for all models
   - `results_summary_for_word_doc.json` - Summary for Word document
   - Per-model predictions and bucket analysis
   - Top-20 vs QQQ plots (MetaRankerStacker outputs):
     - `ridge_stacking_top20_vs_qqq.png` (legacy filename, contains MetaRankerStacker results)
     - `ridge_stacking_top20_vs_qqq_cumulative.png` (legacy filename, contains MetaRankerStacker results)

### Key Features
- **Temporal Safety**: Purge gap prevents label leakage
- **HAC Corrections**: Newey-West (lag≥10) or Hansen-Hodrick standard errors for overlapping observations
- **Daily Rebalancing**: One prediction per trading day
- **Explicit OOS**: All test data is strictly out-of-sample

## Prediction

### Direct Prediction (Using Existing Snapshot)

**Prediction Script**:
```bash
python scripts/time_split_80_20_oos_eval.py \
  --snapshot-id <snapshot_id> \
  --data-file "D:\trade\data\factor_exports\polygon_factors_all_filtered.parquet" \
  --horizon-days 10 \
  --split 0.8 \
  --models elastic_net xgboost catboost lightgbm_ranker lambdarank ridge_stacking \
  --model ridge_stacking \
  --top-n 20 \
  --output-dir "results/prediction_<timestamp>"
```

**Alternative Direct Predict Script** (Excel Output):
```bash
python scripts/direct_predict_ewma_excel.py \
  --snapshot-id <snapshot_id> \
  --tickers-file bma_models/default_tickers.txt \
  --output-dir results/direct_predict
```

**Note**: `ridge_stacking` identifier uses MetaRankerStacker internally.

**Process**:
1. Loads snapshot from `snapshots/<snapshot_id>/`
2. Loads prediction data (MultiIndex format)
3. Generates predictions using trained models:
   - First-layer models predict on **T10 alpha factors** (14 factors)
   - MetaRankerStacker blends first-layer predictions
4. Outputs ranked scores and top-N recommendations
5. **Factor Set**: Always uses `T10_ALPHA_FACTORS` regardless of horizon (T5 removed, 2026-01-24)

### Live Prediction Flow
- `_download_stock_data_for_25factors()` (`bma_models/量化模型_bma_ultra_enhanced.py:7645`) + `Simple17FactorEngine.compute_all_17_factors()` (`bma_models/simple_25_factor_engine.py:270-552`) fetch and compute fresh Polygon features for the requested tickers/dates during prediction.
- **Factor Computation**: Always computes `T10_ALPHA_FACTORS` (14 factors) regardless of prediction horizon. T5 factors removed (2026-01-24).
- `_run_prediction_phase()` (`bma_models/量化模型_bma_ultra_enhanced.py:8272`) ensures features are guarded/validated and then calls `_generate_stacked_predictions()` (`bma_models/量化模型_bma_ultra_enhanced.py:5265`) to blend the base-model outputs via the **MetaRankerStacker** meta learner (see `bma_models/量化模型_bma_ultra_enhanced.py:5717-5722`), yielding ranked scores.
- `predict_with_live_data()` (`bma_models/量化模型_bma_ultra_enhanced.py:8390`) wraps this prediction path and requires prior training state (from `train_from_document()` or persisted snapshot).

**Important**: 
- Prediction uses `self.meta_ranker_stacker.predict()` (`bma_models/量化模型_bma_ultra_enhanced.py:5717-5722`). No RidgeStacker is used in prediction.
- Factor selection always uses `T10_ALPHA_FACTORS` - no conditional logic based on horizon (2026-01-24 update).

### Prediction Data Format
- **Input**: MultiIndex DataFrame with `(date, ticker)` index
- **Required Columns**: **T10 alpha factors** (14 factors: `T10_ALPHA_FACTORS`) - same as training features
- **Factor Set**: Always uses T+10 factors regardless of prediction horizon (T5 removed, 2026-01-24)
- **Output**: DataFrame with predictions and ranked scores
- **Top-N Selection**: Based on MetaRankerStacker final scores

## Full Command Examples

### Complete Training + 80/20 Evaluation

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

**Alternative Data File Path** (if using filtered dataset):
```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "D:\trade\data\factor_exports\polygon_factors_all_filtered.parquet" \
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

**Note**: The `ridge_stacking` model identifier uses MetaRankerStacker internally (LightGBM Ranker with LambdaRank objective).

### Prediction Only (Using Existing Snapshot)
```bash
python scripts/time_split_80_20_oos_eval.py \
  --snapshot-id <snapshot_id_from_previous_training> \
  --data-file "data/factor_exports/polygon_factors_all_filtered.parquet" \
  --horizon-days 10 \
  --split 0.8 \
  --models elastic_net xgboost catboost lightgbm_ranker lambdarank ridge_stacking \
  --model ridge_stacking \
  --top-n 20 \
  --output-dir "results/prediction_<timestamp>"
```

**Note**: The `ridge_stacking` model identifier uses MetaRankerStacker internally.

### Standard Analysis Pipeline
```bash
python scripts/test_simple25_polygon.py \
  --tickers-file bma_models/default_tickers.txt \
  --start-date YYYY-MM-DD --end-date YYYY-MM-DD \
  --run-analysis --analysis-top-n 20 \
  --config bma_models/unified_config.yaml
```

## Key Files and Locations

### Configuration
- **Parameters**: `bma_models/unified_config.yaml`
- **Model Defaults**: 
  - `bma_models/lambda_rank_stacker.py` (first layer)
  - `bma_models/meta_ranker_stacker.py` (second layer)

### Training Data
- **Primary**: `D:\trade\data\factor_exports\polygon_factors_all_filtered.parquet`
- **Format**: MultiIndex `(date, ticker)`
- **Quality**: Filtered (extreme volatility, low price removed)

### Evaluation Scripts
- **80/20 Time Split**: `scripts/time_split_80_20_oos_eval.py`
- **Standard Training**: `bma_models/量化模型_bma_ultra_enhanced.py`

### Model Snapshots
- **Location**: `snapshots/<snapshot_id>/`
- **Contents**: 
  - Trained model files (`.pkl`, `.txt`)
  - Metadata JSON files
  - Configuration backups

### Results
- **80/20 Evaluation**: `results/t10_time_split_80_20_new_params/run_YYYYMMDD_HHMMSS/`
- **Standard Analysis**: `result/` directory

## Architecture Summary

```
Data Loading (MultiIndex)
    ↓
Feature Engineering (T10 Alpha Factors - 14 factors)
    ├── Always uses T10_ALPHA_FACTORS (T5 removed, 2026-01-24)
    ├── liquid_momentum, momentum_10d, momentum_60d, obv_divergence, obv_momentum_60d
    ├── ivol_20, hist_vol_40d, atr_ratio, rsi_21, trend_r2_60
    └── near_52w_high, vol_ratio_20d, price_ma60_deviation, 5_days_reversal
    ↓
First Layer Training (PurgedCV)
    ├── ElasticNet (T10 factors)
    ├── XGBoost (T10 factors)
    ├── CatBoost (T10 factors)
    ├── LightGBM Ranker (T10 factors)
    └── LambdaRankStacker (500 rounds, 70 early stopping, T10 factors)
    ↓
First-Layer Predictions (OOF)
    ↓
Second Layer Training
    └── MetaRankerStacker (300 rounds, 50 early stopping, 31 leaves)
    ↓
Final Ranking Scores
    ↓
80/20 Time Split Evaluation (OOS)
    ├── Training: First 80% dates
    ├── Testing: Last 20% dates (with purge gap)
    └── Metrics: IC, Rank IC, NDCG@10, NDCG@30, Bucket Returns
```

## Recent Updates (2026-01-24)

### Factor Set Unification
- **T5 factors completely removed**: All code paths now exclusively use `T10_ALPHA_FACTORS` (14 factors)
- **Factor selection logic**: Removed conditional logic based on prediction horizon - always uses T+10 factors
- **Compulsory features**: Updated to match `T10_ALPHA_FACTORS` exactly
- **Files updated**:
  - `bma_models/量化模型_bma_ultra_enhanced.py`: Lines 3229-3301 (factor selection)
  - `bma_models/simple_25_factor_engine.py`: Factor engine always uses T10
  - `bma_models/corrected_prediction_exporter.py`: Uses T10 factors

### MultiIndex Alignment Fix
- **Issue**: Failed factor computations used `np.zeros(len(data))` which could cause MultiIndex misalignment
- **Fix**: Changed to `pd.Series(0.0, index=data.index, name='factor_name')` for proper alignment
- **Files updated**: `bma_models/simple_25_factor_engine.py` (multiple locations)

### Direct Predict Scripts
- **Main script**: `scripts/direct_predict_ewma_excel.py` - Generates Excel ranking reports
- **Factor consistency**: Direct predict uses same T10 factors as training
- **80/20 comparison**: `scripts/compare_obv_divergence_8020_split.py` - Compares with/without obv_divergence

This pipeline represents a complete institutional-grade quantitative trading solution with strict temporal safety, ranking optimization, and comprehensive out-of-sample evaluation.

## Important: No RidgeStacker in Training or Prediction

**Critical Clarification**: Despite legacy function names and variable names containing "ridge", **NO RidgeStacker is used in actual training or prediction**:

1. **Training**: 
   - Function `_train_ridge_stacker()` (`bma_models/量化模型_bma_ultra_enhanced.py:10207`) has a legacy name but actually instantiates and trains **MetaRankerStacker** (`bma_models/量化模型_bma_ultra_enhanced.py:10454-10484`)
   - `self.ridge_stacker` is set to `None` (`bma_models/量化模型_bma_ultra_enhanced.py:2995`) and never used
   - Only `self.meta_ranker_stacker` is instantiated and trained

2. **Prediction**:
   - Code checks `self.meta_ranker_stacker` (`bma_models/量化模型_bma_ultra_enhanced.py:5278`)
   - Uses `self.meta_ranker_stacker.predict()` (`bma_models/量化模型_bma_ultra_enhanced.py:5717-5722`)
   - In time_split script, checks for `lightgbm_model` attribute to identify MetaRankerStacker (`scripts/time_split_80_20_oos_eval.py:940`)

3. **Legacy Identifiers**:
   - `ridge_stacking` model identifier in command-line: Legacy name, uses MetaRankerStacker internally
   - `ridge_stacker` parameter name in `save_model_snapshot()`: Legacy parameter name, accepts MetaRankerStacker
   - Function/variable names with "ridge": Legacy naming, all use MetaRankerStacker

**All actual model instances are MetaRankerStacker. RidgeStacker class exists only for backward compatibility checks and is never instantiated or used.**

## Factor Configuration Summary

### T10 Alpha Factors (Always Used)
The following 14 factors are always used in training, prediction, and evaluation (T5 factors removed, 2026-01-24):

1. `liquid_momentum` - Liquidity-adjusted momentum
2. `momentum_10d` - 10-day short-term momentum
3. `momentum_60d` - 60-day momentum
4. `obv_divergence` - OBV divergence (volume-price divergence)
5. `obv_momentum_60d` - 60-day OBV momentum
6. `ivol_20` - 20-day implied volatility
7. `hist_vol_40d` - 40-day historical volatility
8. `atr_ratio` - ATR ratio
9. `rsi_21` - 21-period RSI
10. `trend_r2_60` - 60-day trend R²
11. `near_52w_high` - Distance to 52-week high
12. `vol_ratio_20d` - 20-day volume ratio
13. `price_ma60_deviation` - Price deviation from 60-day MA
14. `5_days_reversal` - 5-day reversal factor

**Configuration Location**: `bma_models/simple_25_factor_engine.py` - `T10_ALPHA_FACTORS` (Line 52-68)

**Usage**: All three main processes (training, direct predict, 80/20 OOS) use this exact factor set.
