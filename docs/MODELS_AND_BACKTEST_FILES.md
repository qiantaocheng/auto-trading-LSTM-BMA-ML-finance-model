## Models & Backtests – Files Used (HETRS‑NASDAQ + BMA Ultra)

This document lists the **primary files** used by the two modeling stacks in this repo:

- **HETRS‑NASDAQ**: single‑asset QQQ system (data → features → TFT → meta‑labeling → CPCV backtests).
- **BMA Ultra**: multi‑asset factor + stacking system (training → snapshot → rolling prediction/backtests).

> Note: “every file used” can be interpreted as *all transitive imports*, which is huge (stdlib + 3rd‑party libs).
> Here we list the **project files** that are directly part of the pipelines and are the ones you edit/execute.

---

## 1) HETRS‑NASDAQ (QQQ single asset) – Code Files

Location: `hetrs_nasdaq/`

- **`hetrs_nasdaq/__init__.py`**
  - Package marker + module overview.

- **`hetrs_nasdaq/repro.py`**
  - `set_global_seed()` – fixes RNG sources for reproducibility.

- **`hetrs_nasdaq/data_loader.py`**
  - Downloads `QQQ`, `^TNX`, `^VIX`, `DX-Y.NYB` via `yfinance`.
  - Aligns macro to QQQ trading days (ffill only).
  - Computes `vol_gk` (Garman‑Klass volatility).
  - CLI: `python -m hetrs_nasdaq.data_loader ...`

- **`hetrs_nasdaq/features.py`**
  - Fixed‑width window **FFD** with ADF‑based `d` search (first 60% only).
  - Expanding **GMM regime probabilities** (causal refit; no lookahead).
  - Technical indicators: RSI, MACD, Bollinger width.
  - Output includes aliases: `regime_p0/p1/p2`, `rsi`, `tnx`.
  - CLI: `python -m hetrs_nasdaq.features ...`

- **`hetrs_nasdaq/tft_model.py`**
  - Trains **TemporalFusionTransformer** (`p10/p50/p90` quantiles).
  - Generates rolling holdout predictions to `tft_p10/tft_p50/tft_p90`.
  - Handles Torch 2.6+ checkpoint loading (`weights_only=False`).
  - CLI: `python -m hetrs_nasdaq.tft_model ...`

- **`hetrs_nasdaq/meta_model.py`**
  - Meta‑labeling:
    - Primary signal from `tft_p50` threshold.
    - Meta features: uncertainty `(p90-p10)`, `vol_gk`, regimes, `rsi`, `macd`.
    - Meta labels: trade success vs future 5‑day return direction.
  - RandomForest meta model: `MetaLabelModel`.

- **`hetrs_nasdaq/backtest.py`**
  - CPCV splitter (purge + embargo).
  - Threshold policy (hysteresis) with optional objectives:
    - `--threshold-objective sharpe|return|drawdown`
  - Outputs:
    - `cpcv_equity.png` (Strategy vs QQQ buy&hold)
    - `cpcv_timeseries.csv`, `learned_thresholds.csv`, `metrics.json`
  - CLI: `python -m hetrs_nasdaq.backtest ...`

- **`hetrs_nasdaq/backtest_v2.py`**
  - v3.1 execution layer:
    - CPCV‑safe meta‑labeling (train on fold train, infer on fold test).
    - Dynamic transaction costs: `bp = 5 + 0.1 * vol_gk`.
    - Metrics include PSR (Probabilistic Sharpe Ratio).
    - Tear‑sheet: Buy&Hold vs Naive TFT vs Meta‑TFT.
  - CLI: `python -m hetrs_nasdaq.backtest_v2 ...`

- **`hetrs_nasdaq/explainability.py`**
  - Best‑effort TFT interpretability extraction:
    - Variable importance (VSN)
    - Attention weights
  - Saves heatmaps + a JSON meta dump.

- **`hetrs_nasdaq/rl_agent.py`**
  - Optional RL path (SB3 PPO/A2C/DDPG) with Differential Sharpe reward.
  - Ensemble weighting by recent Sharpe.

- **`hetrs_nasdaq/sweep_meta_params.py`**
  - Grid sweep for v3.1 meta parameters:
    - `primary_threshold × meta_prob_threshold`
  - Outputs:
    - `sweep_results.csv`, `best_configs.json`
    - Heatmaps for meta Sharpe/CumReturn/MaxDD.

- **`hetrs_nasdaq/trade_stats.py`**
  - Computes estimated buy/sell frequency from a timeseries CSV (`position` column).
  - CLI: `python -m hetrs_nasdaq.trade_stats --csv <path>`

- **`hetrs_nasdaq/market_signals.py`**
  - Generates daily market gating signals from QQQ features + TFT:
    - `tft_p50`
    - causal `meta_prob_success` (expanding meta model)
    - `market_regime` + `exposure_scalar`
  - CLI: `python -m hetrs_nasdaq.market_signals ...`

---

## 2) HETRS‑NASDAQ – Data / Results Artifacts (examples)

- **Data**
  - `data/hetrs_nasdaq/qqq_macro.parquet`
  - `data/hetrs_nasdaq/qqq_features.parquet`
  - `data/hetrs_nasdaq/tft_preds.parquet`
  - `data/hetrs_nasdaq/qqq_features_with_tft.parquet`

- **Results (example runs)**
  - `results/hetrs_nasdaq/backtest_threshold_20251221_235104/`
  - `results/hetrs_nasdaq/backtest_v2_20251222_002930/`
  - `results/hetrs_nasdaq/meta_param_sweep_20251222_003244/`

---

## 3) BMA Ultra – Core Model / Training / Snapshot Files

Location: `bma_models/`

- **`bma_models/量化模型_bma_ultra_enhanced.py`**
  - The main BMA Ultra model implementation (training + prediction + stacking orchestration).
  - Feature subset configuration / per‑model overrides.
  - Unified CV training loop and LambdaRank target handling (`ret_fwd_*`).

- **`bma_models/unified_config.yaml`**
  - Central configuration for the unified pipeline.
  - Includes ridge stacker base columns (e.g., `pred_elastic`, `pred_xgb`, `pred_lambdarank`).

- **`bma_models/model_registry.py`**
  - Snapshot save/load for reproducibility (models + metadata + features).

- **`bma_models/ridge_stacker.py`**
  - Second‑layer meta learner (Ridge regression).

- **`bma_models/lambda_rank_stacker.py`**
  - LambdaRank model implementation requiring `target_col` like `ret_fwd_10d`.

- **`bma_models/simple_25_factor_engine.py`**
  - Factor computation pipeline (17 factors etc.) and forward return targets (shifted).

- **Purged CV / leakage prevention helpers**
  - `bma_models/unified_purged_cv_factory.py`
  - `bma_models/cv_leakage_prevention.py`

---

## 4) BMA Ultra – Main Scripts Used for Training / Backtesting / Sweeps

Location: `scripts/`

- **`scripts/train_single_model.py`**
  - Train one model with optional feature list and env overrides.

- **`scripts/t10_feature_combo_grid_search_parallel_models.py`**
  - Grid search over feature combos for T+10 horizon across models.

- **`scripts/extract_best_features_from_grid.py`**
  - Parse grid results and export best feature list per model.

- **`scripts/test_ridge_stacking_best_features.py`**
  - Train a full BMA Ultra snapshot with “best per model” features and run backtests.

- **`scripts/sweep_ridge_input_combinations.py`**
  - Sweep combinations of base prediction columns fed into Ridge stacker.

- **`scripts/comprehensive_model_backtest.py`**
  - Comprehensive backtest runner:
    - loads snapshots
    - rolling prediction
    - bucket returns, reports

- **Visualization**
  - `scripts/plot_t10_buckets_vs_nasdaq.py`
  - `scripts/plot_curve_vs_benchmark.py`

- **Time split OOS evaluation**
  - `scripts/time_split_80_20_oos_eval.py`

- **Ridge portfolio decision layer (Top‑10 bi‑weekly)**
  - `scripts/ridge_stacking_portfolio_manager.py`
    - Consumes `ridge_stacking_predictions_*.parquet` with `date,ticker,prediction,actual`
    - Uses HETRS market signals to gate exposure (`market_signals.parquet`)
    - Outputs weights/trades/equity/metrics.

---

## 5) BMA Ultra – Key Result Artifacts Used by the Pipeline

- **Best per‑model feature lists**
  - `results/t10_optimized_all_models/best_features_per_model.json`
  - `results/t10_optimized_all_models/best_features_per_model.csv`

- **Example sweep/backtest outputs**
  - `results/t10_optimized_all_models/ridge_input_sweep3/...`
  - `results/t10_optimized_all_models/ridge_stacking_test*/...`


