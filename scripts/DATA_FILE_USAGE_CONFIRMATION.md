# Data File Usage Confirmation

## 📊 File: `polygon_factors_all_filtered_clean.parquet`

### ✅ 1. Direct Predict Usage

**Location**: `autotrader/app.py` (line 1540, 1657-1723)

**Usage**:
- ✅ **Used for**: Extracting stock ticker list (default tickers)
- ❌ **NOT used for**: Loading pre-computed factors

**How it works**:
1. **Ticker List Extraction** (line 1540-1571):
   ```python
   default_tickers_file = Path(r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean.parquet")
   df_tickers = pd.read_parquet(default_tickers_file)
   default_tickers = sorted(df_tickers.index.get_level_values('ticker').unique().tolist())
   ```
   - Reads parquet file to extract unique tickers
   - Uses these tickers as default input for Direct Predict

2. **Factor Computation** (line 1657-1723):
   ```python
   engine = Simple17FactorEngine(lookback_days=total_lookback_days, mode='predict', horizon=prediction_horizon)
   market_data = engine.fetch_market_data(symbols=tickers, ...)  # Fetches from API
   all_feature_data = engine.compute_all_17_factors(market_data, mode='predict')  # Computes factors
   ```
   - **Does NOT load factors from parquet file**
   - Fetches market data from Polygon API
   - Computes all factors on-the-fly (including Sato factors if missing)

**Conclusion**: 
- ✅ Uses parquet file for **ticker list only**
- ❌ Does **NOT** use parquet file for factor data
- ✅ Computes factors from live market data

---

### ✅ 2. 80/20 Time Split Evaluation Usage

**Location**: `scripts/time_split_80_20_oos_eval.py` (line 344, 1312-1484)

**Usage**:
- ✅ **Used for**: Loading pre-computed factors (including Sato factors)
- ✅ **Used by**: All first-layer models (ElasticNet, XGBoost, CatBoost, LambdaRank)

**How it works**:

1. **Data File Path** (line 344):
   ```python
   p.add_argument("--data-file", type=str, 
                  default=r"D:\trade\data\factor_exports\polygon_factors_all_filtered.parquet")
   ```
   - Default: `polygon_factors_all_filtered.parquet`
   - Can be overridden: `--data-file "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean.parquet"`

2. **Data Loading** (line 1312-1380):
   ```python
   data_path = Path(args.data_file)
   if use_path.is_file():
       logger.info(f"Loading single parquet file: {use_path}")
       df = pd.read_parquet(str(use_path))  # Loads factor data from parquet
   ```
   - Loads factor data directly from parquet file
   - Expects MultiIndex (date, ticker) format

3. **Sato Factor Check** (line 1429-1484):
   ```python
   if 'feat_sato_momentum_10d' not in df.columns or 'feat_sato_divergence_10d' not in df.columns:
       # Computes Sato factors if missing
       sato_factors_df = calculate_sato_factors(...)
       df['feat_sato_momentum_10d'] = sato_factors_df['feat_sato_momentum_10d']
       df['feat_sato_divergence_10d'] = sato_factors_df['feat_sato_divergence_10d']
   ```
   - Checks if Sato factors exist in loaded data
   - Computes them if missing

4. **Model Usage** (line 1777-1814):
   ```python
   # ElasticNet
   if 'elastic_net' in models_dict:
       X_aligned = align_test_features_with_model(X, models_dict['elastic_net'], ...)
       pred = models_dict['elastic_net'].predict(X_aligned)
   
   # XGBoost
   if 'xgboost' in models_dict:
       X_aligned = align_test_features_with_model(X, models_dict['xgboost'], ...)
       pred = models_dict['xgboost'].predict(X_aligned)
   
   # CatBoost
   if 'catboost' in models_dict:
       X_aligned = align_test_features_with_model(X, models_dict['catboost'], ...)
       pred = models_dict['catboost'].predict(X_aligned)
   
   # LambdaRank
   if lambda_rank_stacker is not None:
       # Uses X (features from parquet file)
   ```
   - All first-layer models use the same `X` (features loaded from parquet file)
   - Features are aligned with each model's expected feature list

**Conclusion**:
- ✅ Uses parquet file for **factor data** (not just tickers)
- ✅ All first-layer models (ElasticNet, XGBoost, CatBoost, LambdaRank) use this data
- ✅ Automatically computes Sato factors if missing from file

---

## 📋 Summary Table

| Feature | Direct Predict | 80/20 Time Split |
|---------|---------------|------------------|
| **Uses parquet file** | ✅ Yes (tickers only) | ✅ Yes (factors) |
| **Uses `polygon_factors_all_filtered_clean.parquet`** | ✅ Yes (ticker list) | ⚠️ Can be specified via `--data-file` |
| **Default parquet file** | `polygon_factors_all_filtered_clean.parquet` | `polygon_factors_all_filtered.parquet` |
| **Factor source** | Live API + computation | Pre-computed in parquet |
| **Sato factors** | Computed on-the-fly | Loaded from parquet (or computed if missing) |
| **First-layer models** | ✅ All models use computed factors | ✅ All models use parquet factors |

---

## ✅ Confirmation Checklist

### Direct Predict
- [x] Uses `polygon_factors_all_filtered_clean.parquet` for ticker list extraction
- [x] Does NOT load factors from parquet (computes from live data)
- [x] All first-layer models use computed factors
- [x] Sato factors computed on-the-fly if needed

### 80/20 Time Split
- [x] Can use `polygon_factors_all_filtered_clean.parquet` via `--data-file` parameter
- [x] Loads factor data directly from parquet file
- [x] All first-layer models (ElasticNet, XGBoost, CatBoost, LambdaRank) use parquet factors
- [x] Automatically computes Sato factors if missing from parquet

---

## 🔍 How to Verify

### Check Direct Predict
```python
# In app.py _direct_predict_snapshot():
# Line 1540: Uses parquet for tickers
# Line 1657: Fetches market data from API
# Line 1723: Computes factors from market data
```

### Check 80/20 Time Split
```bash
# Default (uses polygon_factors_all_filtered.parquet):
python scripts/time_split_80_20_oos_eval.py --horizon-days 10 --split 0.8

# Use cleaned file explicitly:
python scripts/time_split_80_20_oos_eval.py \
  --data-file "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean.parquet" \
  --horizon-days 10 \
  --split 0.8 \
  --models catboost lambdarank ridge_stacking
```

---

## 📝 Notes

1. **Direct Predict**:
   - Does NOT use pre-computed factors from parquet
   - Always computes factors from live market data
   - This ensures predictions use the most up-to-date data

2. **80/20 Time Split**:
   - Uses pre-computed factors from parquet for efficiency
   - Can use `polygon_factors_all_filtered_clean.parquet` if specified
   - All first-layer models use the same factor data from parquet

3. **Sato Factors**:
   - Direct Predict: Computed on-the-fly during factor computation
   - 80/20 Time Split: Loaded from parquet (or computed if missing)

4. **Model Consistency**:
   - Both Direct Predict and 80/20 Time Split use the same first-layer models
   - All models (ElasticNet, XGBoost, CatBoost, LambdaRank) use the same feature set
