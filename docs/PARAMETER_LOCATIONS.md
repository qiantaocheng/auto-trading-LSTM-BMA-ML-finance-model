# Parameter locations

Where key parameters for 80/20 time-split training and prediction are defined.

---

## 1. Nine default factors (TOP_FEATURE_SET)

| Location | What |
|----------|------|
| **`bma_models/simple_25_factor_engine.py`** | `TOP_FEATURE_SET` (lines ~165–185), `DEFAULT_FEATURE_SET = TOP_FEATURE_SET` |
| **`bma_models/unified_config.yaml`** | `features.default_feature_set` (lines 67–76) — same 9 names |

Used when 80/20 script trains with default features (if `--features` / overrides not set).  
**Note:** Prediction-with-snapshot path uses hardcoded `allowed_feature_cols` in the script (see below), not TOP_FEATURE_SET.

---

## 2. 80/20 script defaults (split, horizon, data, models, EMA, HAC)

| Parameter | Location | Default |
|-----------|----------|---------|
| **--data-file** | `scripts/time_split_80_20_oos_eval.py` `_parse_args()` | `polygon_factors_all_2021_2026_T5_final.parquet` |
| **--train-data** | same | `polygon_factors_all_2021_2026_T5_final.parquet` |
| **--horizon-days** | same | 10 |
| **--split** | same | 0.8 |
| **--top-n** | same | 20 |
| **--models** | same | catboost, lambdarank, ridge_stacking |
| **--benchmark** | same | QQQ |
| **--cost-bps** | same | 0 |
| **--hac-method** | same | newey-west |
| **--hac-lag** | same | None → `max(10, 2*horizon_days)` |
| **--output-dir** | same | results/t10_time_split_80_20_final |
| **--snapshot-id** | same | None |
| **--ema-top-n** | same | -1 (EMA off) |
| **--ema-min-days** | same | 3 |
| **--ema-length** | same | 3 |
| **--ema-beta** | same | None |
| **--ema-models** | same | None |
| **--exclude-tickers** | same | None |

**File:** `scripts/time_split_80_20_oos_eval.py` — function `_parse_args()` (around lines 342–376).

---

## 3. Prediction-path feature list (snapshot eval)

| Location | What |
|----------|------|
| **`scripts/time_split_80_20_oos_eval.py`** | Hardcoded `allowed_feature_cols` (~lines 1821–1836) — 14 factors; then aligned to snapshot’s `feature_names` via `align_test_features_with_model`. |

Actual model input = snapshot’s `feature_names` (from manifest); `allowed_feature_cols` only filters which columns are read from data before alignment.

---

## 4. LambdaRank parameters

| Parameter | Primary location | Fallback / code default |
|-----------|------------------|--------------------------|
| **n_quantiles** | `bma_models/unified_config.yaml` → `training.lambdarank.n_quantiles` (32) | `bma_models/lambda_rank_stacker.py` `__init__` default 32 |
| **label_gain_power** | `unified_config.yaml` → `lambdarank.label_gain_power` (2.6) | `lambda_rank_stacker.py` default 2.6 |
| **winsorize_quantiles** | Not in YAML | `lambda_rank_stacker.py` (0.01, 0.99) |
| **cv_n_splits, cv_gap_days, cv_embargo_days** | Not in YAML lambdarank section | `lambda_rank_stacker.py` (6, 5, 5) |
| **lambdarank_truncation_level, sigmoid** | `unified_config.yaml` → lambdarank (60, 1.15) | `lambda_rank_stacker.py` `default_lgb_params` (60, 1.15) |
| **num_boost_round, early_stopping_rounds** | `unified_config.yaml` → lambdarank | `lambda_rank_stacker.py` (1200, 100) |
| **lgb_params** (num_leaves, max_depth, learning_rate, etc.) | `unified_config.yaml` → `training.lambdarank` (lines 188–210) | `lambda_rank_stacker.py` `default_lgb_params` |

**Files:**
- **`bma_models/unified_config.yaml`** — `training.lambdarank` (lines 188–210).
- **`bma_models/lambda_rank_stacker.py`** — `LambdaRankStacker.__init__` and `default_lgb_params`.
- **`bma_models/量化模型_bma_ultra_enhanced.py`** — builds lambda config from YAML and passes to `LambdaRankStacker`; fallbacks when config missing.

---

## 5. Temporal / CV (training pipeline)

| Parameter | Location |
|-----------|----------|
| **prediction_horizon_days, holding_period_days** | `unified_config.yaml` → `temporal` (10, 10) |
| **cv_gap_days, cv_embargo_days, cv_n_splits** (top-level) | `unified_config.yaml` → `temporal` (10, 10, 6) |
| **cv_gap_days, cv_embargo_days, cv_splits** (training) | `unified_config.yaml` → `training` (5, 5, 6) |

**File:** `bma_models/unified_config.yaml` — `temporal` (lines 1–19), `training` (lines 113–122).

---

## 6. Meta-ranker (second layer)

| Parameter | Location |
|-----------|----------|
| **base_cols, n_quantiles, label_gain_power, lgb_params, etc.** | `unified_config.yaml` → `training.meta_ranker` (lines 211–239) |
| **Code fallbacks** | `bma_models/量化模型_bma_ultra_enhanced.py` (e.g. `_META_RANKER_CONFIG`) |

---

## 7. Snapshot / manifest (saved run config)

| What | Location |
|------|----------|
| **feature_names, feature_names_by_model** | Each snapshot dir: `cache/model_snapshots/<date>/<snapshot_id>/manifest.json` |
| **LambdaRank meta** (n_quantiles, label_gain_power, cv_*, base_cols) | Same snapshot dir: `lambdarank_meta.json` |
| **Meta-ranker meta** | Same snapshot dir: `meta_ranker_meta.json` |

---

## 8. Quick reference

| Want to change… | Edit |
|------------------|------|
| **9 default factors** | `simple_25_factor_engine.py` `TOP_FEATURE_SET` and/or `unified_config.yaml` `features.default_feature_set` |
| **80/20 split, horizon, data file, output dir** | `scripts/time_split_80_20_oos_eval.py` `_parse_args()` |
| **LambdaRank (n_quantiles, label_gain_power, lgb_params)** | `unified_config.yaml` `training.lambdarank` and/or `lambda_rank_stacker.py` defaults |
| **Prediction feature whitelist (snapshot eval)** | `time_split_80_20_oos_eval.py` `allowed_feature_cols` (~line 1821) |
| **Temporal/CV** | `unified_config.yaml` `temporal` and `training` |
| **What a past run used** | Snapshot `manifest.json` and `lambdarank_meta.json` (and run’s `oos_metrics.json` if present) |
