## US T+10 Top‑K — Results Pack + Repro Guide (repo-derived)

Hard rules applied:
- **No fabricated numbers**: every metric cites a **file path** + **column name** (and where possible a row key/date).
- Missing result artifacts are marked **NOT FOUND**, plus what file/output should exist and how to produce it.

### Repo context (filled from repo outputs; otherwise NOT FOUND)

- **Research question**: Rank US stocks cross-sectionally and test whether Top‑K selections outperform over **T+10 trading days**, using single models and **Ridge stacking**.
- **Universe**: all tickers present in the factor dataset.
  - Time-split run uses `data/factor_exports/factors/factors_all.parquet` (no tickers filter arg in this script).
- **Horizon**: 10 trading days (T+10)
  - Source: `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/run_config.json` (`target_horizon_days`)
- **Rebalance frequency / constraints**: non-overlapping rebalance every 10 trading days (`rebalance_mode=horizon`).
  - Source: `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/run_config.json` (`rebalance_mode`)
- **Primary objective metric (TEST / last 20%, complete backtest)**: `avg_top_return_net` (Top‑30, equal-weight, net of costs, per rebalance period; fraction units).
  - Source: `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/performance_report_20260107_030838.csv` (`avg_top_return_net`)
- **Evaluation period (TEST / last 20%, complete backtest)**: 2024‑11‑08 → 2025‑11‑06 (25 rebalances, non-overlapping).
  - Source: `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/run_config.json` (`start_date`, `end_date`)
  - Rebalance count evidence: `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/xgboost_weekly_returns_20260107_030838.csv` (25 data rows)

---

## A) Repo Map (1 page)

### Entry scripts / pipeline stages

- **Dataset audit**: `scripts/audit_factor_dataset_pit.py`
  - Example output: `result/factor_audit_current/audit_report.json`
- **Backtest engine (multi-model, Top‑N)**: `scripts/comprehensive_model_backtest.py`
  - Outputs (per run): `performance_report_*.csv`, `*_weekly_returns_*.csv`, `*_predictions_*.parquet`
- **Bucket plots vs QQQ**: `scripts/plot_t10_buckets_vs_nasdaq.py`
  - Outputs: `buckets_vs_nasdaq.csv`, `buckets_vs_nasdaq*.png`
- **Single-model plots vs QQQ**: `scripts/plot_single_models_vs_benchmark.py`
  - Outputs: `per_model_topN_vs_benchmark.csv`, `per_model_topN_vs_benchmark*.png`
- **Walk-forward sanity / no-leakage**: `scripts/walkforward_top10.py`
  - Outputs: `summary.json`, `leakage_audit.csv`, plus equity/trade outputs (per run directory)
- **Results Pack artifact builder (added)**: `scripts/build_results_pack_artifacts.py`
  - Outputs: `docs/results_pack_inventory.csv`, `<run>/results_pack_summary.json`

### Configs / snapshot system

- **Main config**: `bma_models/unified_config.yaml`
- **Snapshot manifest (example for key run’s snapshot)**:
  - `cache/model_snapshots/20251221/7de6f766-da32-43a5-b5a0-4d69d2426f18/manifest.json`

### Experiments organization

- **Primary experiments**: `results/<run_dir>/...`
- **Walk-forward outputs**: `result/walkforward_top10_*/...`
- **Generated index**: `docs/results_pack_inventory.csv` (inventory of `results/` runs with key artifacts)

### Minimal commands to reproduce key outputs

The commands below reproduce the key outputs used in sections D/E.

```powershell
cd D:\trade

# (0) Optional: regenerate inventory+summary artifacts
.\trading_env\Scripts\python.exe scripts\build_results_pack_artifacts.py `
  --out-run-dir results\paper_costs_nocat_20260106_135011 `
  --results-dir results `
  --inventory-out docs\results_pack_inventory.csv `
  --data-file data\factor_exports\factors\factors_all.parquet `
  --rebalance-mode horizon `
  --target-horizon-days 10 `
  --primary-objective-col avg_top_return_net

# (1) Run cost-aware backtest from snapshot
$out = "results\paper_costs_nocat_REPRO"
New-Item -ItemType Directory -Force $out | Out-Null
$snap = "7de6f766-da32-43a5-b5a0-4d69d2426f18"
.\trading_env\Scripts\python.exe scripts\comprehensive_model_backtest.py `
  --data-dir data\factor_exports\factors `
  --data-file data\factor_exports\factors\factors_all.parquet `
  --snapshot-id $snap `
  --rebalance-mode horizon `
  --target-horizon-days 10 `
  --max-weeks 260 `
  --cost-bps 10 `
  --output-dir $out

# (2) Bucket plots (ridge_stacking) vs QQQ
.\trading_env\Scripts\python.exe scripts\plot_t10_buckets_vs_nasdaq.py `
  --snapshot-id $snap `
  --model ridge_stacking `
  --benchmark QQQ `
  --data-dir data\factor_exports\factors `
  --data-file data\factor_exports\factors\factors_all.parquet `
  --rebalance-mode horizon `
  --target-horizon-days 10 `
  --max-weeks 260 `
  --cost-bps 10 `
  --output-dir $out

# (3) Per-model plot vs QQQ (gross vs net)
.\trading_env\Scripts\python.exe scripts\plot_single_models_vs_benchmark.py `
  --backtest-outdir $out `
  --models elastic_net xgboost lambdarank ridge_stacking `
  --benchmark QQQ `
  --output-dir $out
```

Expected outputs:
- Backtest: `$out/performance_report_*.csv`, `$out/*_weekly_returns_*.csv`, `$out/*_predictions_*.parquet`
- Bucket plots: `$out/buckets_vs_nasdaq.csv`, `$out/buckets_vs_nasdaq*.png`
- Single model plots: `$out/per_model_topN_vs_benchmark.csv`, `$out/per_model_topN_vs_benchmark*.png`

---

## B) Data & Feature Summary

### Data sources and sample filters

- **Main dataset**: `data/factor_exports/factors/factors_all.parquet`
  - Audit output: `result/factor_audit_current/audit_report.json`
- **Universe filter applied in key run**: NONE
  - Source: `results/paper_costs_nocat_20260106_135011/results_pack_summary.json` (`tickers_file_applied=null`)
- **Universe filter list exists (not applied here)**: `filtered_stocks_20250817_002928.txt`

### Feature groups (high level)

- **Alpha factors + OHLCV-derived features** (examples): `ivol_20`, `hist_vol_40d`, `near_52w_high`, `rsi_21`, `trend_r2_60`, `vol_ratio_20d`, `price_ma60_deviation`, etc.
  - Source: `result/factor_audit_current/audit_report.json` (`cols`)
- **Best-per-model feature lists (artifact)**:
  - Source: `results/t10_optimized_all_models/best_features_per_model.json`
  - Note: file includes a `catboost` key, but CatBoost is **not evaluated** in the key run (see D/E sources).

### Leakage prevention steps found in code / outputs

- **Non-overlapping evaluation** via horizon rebalance:
  - Source: `results/paper_costs_nocat_20260106_135011/results_pack_summary.json` (`rebalance_mode=horizon`, `target_horizon_days=10`)
- **Walk-forward purge gap rule**:
  - Code: `scripts/walkforward_top10.py` (`train_end = rebalance_i - horizon_days - purge_days`)
  - Evidence: `result/walkforward_top10_yearly_20251223_131840/leakage_audit.csv` (column `ok=True`)
- **Dataset PIT audit**:
  - Output: `result/factor_audit_current/audit_report.json` (`target_looks_standardized_per_date=false`)

---

## C) Experiment Inventory

### Inventory index (authoritative list)

- **File**: `docs/results_pack_inventory.csv`
  - Generated by: `scripts/build_results_pack_artifacts.py`
  - This file lists **29** `results/` subdirectories with key artifacts.

### Key run used in D/E (cost-aware, CatBoost excluded)

- `results/paper_costs_nocat_20260106_135011/`
  - Contains: `performance_report_20260106_135130.csv`, `buckets_vs_nasdaq.csv`, `per_model_topN_vs_benchmark.csv`, plus plots and predictions parquet.

NOT FOUND:
- A repo-wide “run registry” with explicit CLI + git hash per run: NOT FOUND
  - What should exist: `results/<run>/run_metadata.json`
  - How: write a metadata file at run start in `scripts/comprehensive_model_backtest.py`

---

## D) Executive Results Snapshot

### Best performing run on the PRIMARY objective metric (test set)

**Time-split TEST window (last 20%) — complete backtest — with costs — all models** (primary test result):

- **Run folder**: `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/`
- **Config / window / costs**: `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/run_config.json`
  - `start_date=2024-11-08`, `end_date=2025-11-06`, `target_horizon_days=10`, `cost_bps=10.0`
- **Report (all models)**: `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/performance_report_20260107_030838.csv`
- **Benchmark comparison (all models)**:
  - `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/per_model_topN_vs_benchmark.csv`
  - `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/per_model_topN_vs_benchmark_summary.csv`
  - `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/per_model_topN_vs_benchmark.png`
  - `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/per_model_topN_vs_benchmark_cum.png`

### Top 5 headline metrics (best model, best run)

Source:
- File: `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/performance_report_20260107_030838.csv`
- Best model by primary objective `avg_top_return_net`: row `Model == lambdarank`

| Metric | Value | Source column |
|---|---:|---|
| avg_top_return_net | 0.041402332191512584 | `avg_top_return_net` |
| avg_top_return | 0.04297566552484592 | `avg_top_return` |
| IC | 0.017642754884820287 | `IC` |
| Rank_IC | -0.005404305812764513 | `Rank_IC` |
| avg_top_turnover | 1.5733333333333333 | `avg_top_turnover` |

### 3 key observations (repo-derived)

- Best `avg_top_return_net` on the test window is `lambdarank` (Top‑30) among the evaluated models.
  - Source: `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/performance_report_20260107_030838.csv` (`avg_top_return_net`)
- Ridge stacking is materially positive on `avg_top_return_net` but has very poor scale metrics (`R2`, `MSE`) on the same window.
  - Source: same file (`avg_top_return_net`, `R2`, `MSE`) for row `ridge_stacking`
- Benchmark comparison (Top‑30) shows large dispersion across models on this window (see net cumulative % series).
  - Source: `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/per_model_topN_vs_benchmark_summary.csv` (`end_cum_pct`)

### 3 caveats

- **Units/scale**: bucket returns appear “percent-like” (e.g., `top_1_10_return` ≈ 5.27 on 2020‑11‑30), so cumulative series can become extremely large.
  - Source: `results/paper_costs_nocat_20260106_135011/buckets_vs_nasdaq.csv`
- **Regression R²/MSE**: `ridge_stacking` has very negative `R2` and very large `MSE`; treat scale-based regression metrics cautiously for rank objectives.
  - Source: `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/performance_report_20260107_030838.csv` (`R2`, `MSE`)
- **Snapshot ID not recorded in the run folder**: `results/paper_costs_nocat_20260106_135011` has no `snapshot_id.txt`.
  - Workaround source: `cache/model_snapshots/20251221/7de6f766-da32-43a5-b5a0-4d69d2426f18/manifest.json`

---

## E) Core Tables

### E1) Predictive metrics table

Time-split TEST window predictive metrics (all models).

Source: `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/performance_report_20260107_030838.csv`

| Model | IC | Rank_IC | MSE | MAE | R2 |
|---|---:|---:|---:|---:|---:|
| elastic_net | -0.034771579710549984 | -0.0005170273434417982 | 0.07465991696355985 | 0.07648231880329785 | -0.0016590770803230903 |
| xgboost | -0.01067801890655071 | 0.006536196878077974 | 0.07470452117044112 | 0.07666478491581369 | -0.0022575000429470027 |
| catboost | -0.04418289168885258 | 0.0028485597929391025 | 0.07463674935692195 | 0.07641514234919282 | -0.0013482537573559217 |
| lambdarank | 0.017642754884820287 | -0.005404305812764513 | 0.07528550427737016 | 0.0812315073352395 | -0.010052137732804667 |
| ridge_stacking | -0.015224732977128489 | 0.0007593220120601699 | 1.207564214060249 | 0.6576350188268877 | -15.201031361462437 |

### E2) Backtest performance table (Top‑30 basket, gross vs net)

Time-split TEST window backtest summary (Top‑30, gross vs net; fraction units).

Source: `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/performance_report_20260107_030838.csv`

| Model | avg_top_return | avg_top_return_net | avg_top_turnover | avg_top_cost | cost_bps |
|---|---:|---:|---:|---:|---:|
| elastic_net | 0.007855362684538228 | 0.006007362684538225 | 1.848 | 0.0018480000000000003 | 10.0 |
| xgboost | 0.027513482952439747 | 0.02576148295243975 | 1.7520000000000002 | 0.0017520000000000003 | 10.0 |
| catboost | 0.011943416426986164 | 0.010066083093652834 | 1.8773333333333333 | 0.0018773333333333337 | 10.0 |
| lambdarank | 0.04297566552484592 | 0.041402332191512584 | 1.5733333333333333 | 0.0015733333333333333 | 10.0 |
| ridge_stacking | 0.03035890554232758 | 0.02854023887566091 | 1.8186666666666664 | 0.001818666666666667 | 10.0 |

### E3) Benchmark comparison table (bucket net cumulative vs QQQ cumulative)

Time-split TEST window benchmark comparison (Top‑30 vs QQQ, % units; includes gross and net series).

Source: `results/t10_time_split_test20_completebacktest_cost10_allmodels_20260107_030817/per_model_topN_vs_benchmark_summary.csv`

| series | mean_pct | end_cum_pct |
|---|---:|---:|
| benchmark_return | 0.8829144973344378 | 21.217948946243137 |
| elastic_net_top_return_net | 0.6007362684538217 | 13.248487959619082 |
| xgboost_top_return_net | 2.576148295243974 | 70.42422185713902 |
| catboost_top_return_net | 1.0066083093652818 | 25.272525299762382 |
| lambdarank_top_return_net | 4.1402332191512565 | 143.6771084005041 |
| ridge_stacking_top_return_net | 2.85402388756609 | 82.80185537850289 |

---

## F) Robustness & Sanity Checks

### Already done (with paths)

- **Dataset audit** (target distribution + correlations): `result/factor_audit_current/audit_report.json`
- **Leakage audit (walk-forward annual retrain)**: `result/walkforward_top10_yearly_20251223_131840/leakage_audit.csv` (column `ok`)

### Missing / NOT FOUND (recommended)

- **Split-aware (train/val/test) metrics** for the key run: NOT FOUND
- **Snapshot ID written into the run folder**: NOT FOUND
- **A single consolidated metrics index across runs** (beyond `docs/results_pack_inventory.csv`): NOT FOUND
  - What should exist: `docs/results_pack_metrics_index.csv` (one row per run/model)
  - How: extend `scripts/build_results_pack_artifacts.py` to parse `performance_report_*.csv` from each run directory

### Next 8 robustness checks (repo-specific, prioritized)

1) Time-split OOS report (80/20) with costs (test segment only).
2) Cost sensitivity: `cost_bps ∈ {0,5,10,20}`.
3) K sensitivity: K ∈ {10,20,30,50}.
4) Universe sensitivity: run with `--tickers-file filtered_stocks_20250817_002928.txt` (documented list) vs none.
5) Horizon sensitivity: T+5 and T+20 (requires corresponding labels).
6) Purge sensitivity in walk-forward: vary `purge_days` and verify `leakage_audit.csv` stays `ok=True`.
7) Target integrity check: re-run `scripts/audit_factor_dataset_pit.py` after any factor pipeline changes.
8) Stability by sub-universe (liquidity/volatility strata) if suitable columns exist.

---

## G) Repro Guide (Cursor-friendly)

### Commands with expected outputs

1) Inventory + summary for an existing run:

```powershell
cd D:\trade
.\trading_env\Scripts\python.exe scripts\build_results_pack_artifacts.py `
  --out-run-dir results\paper_costs_nocat_20260106_135011 `
  --results-dir results `
  --inventory-out docs\results_pack_inventory.csv `
  --data-file data\factor_exports\factors\factors_all.parquet `
  --rebalance-mode horizon `
  --target-horizon-days 10 `
  --primary-objective-col avg_top_return_net
```

Expected:
- `docs/results_pack_inventory.csv`
- `results/paper_costs_nocat_20260106_135011/results_pack_summary.json`

2) Key run backtest (from snapshot, cost-aware):

```powershell
cd D:\trade
$out="results\paper_costs_nocat_REPRO"
New-Item -ItemType Directory -Force $out | Out-Null
.\trading_env\Scripts\python.exe scripts\comprehensive_model_backtest.py `
  --data-dir data\factor_exports\factors `
  --data-file data\factor_exports\factors\factors_all.parquet `
  --snapshot-id 7de6f766-da32-43a5-b5a0-4d69d2426f18 `
  --rebalance-mode horizon `
  --target-horizon-days 10 `
  --max-weeks 260 `
  --cost-bps 10 `
  --output-dir $out
```

Expected (within `$out`):
- `performance_report_*.csv`
- `*_weekly_returns_*.csv`
- `*_predictions_*.parquet`

2b) Time-aware 80/20 split (purged) + backtest on last 20% with costs:

```powershell
cd D:\trade
$ts=Get-Date -Format "yyyyMMdd_HHmmss"
$out=("results\t10_time_split_80_20_costs_" + $ts)
New-Item -ItemType Directory -Force $out | Out-Null
.\trading_env\Scripts\python.exe scripts\time_split_80_20_oos_eval.py `
  --train-data data\factor_exports\factors `
  --data-dir data\factor_exports\factors `
  --data-file data\factor_exports\factors\factors_all.parquet `
  --horizon-days 10 `
  --split 0.8 `
  --model ridge_stacking `
  --top-n 20 `
  --rebalance-mode horizon `
  --max-weeks 260 `
  --cost-bps 10 `
  --benchmark QQQ `
  --output-dir $out `
  --log-level INFO
Write-Output $out
```

Expected (within `$out/run_<ts>/`):
- `oos_metrics.csv`, `oos_metrics.json`
- `ridge_top20_timeseries.csv`, `report_df.csv`, `snapshot_id.txt`
- `top20_vs_qqq.png`, `top20_vs_qqq_cumulative.png`

3) Plot buckets vs QQQ (ridge_stacking):

```powershell
.\trading_env\Scripts\python.exe scripts\plot_t10_buckets_vs_nasdaq.py `
  --snapshot-id 7de6f766-da32-43a5-b5a0-4d69d2426f18 `
  --model ridge_stacking `
  --benchmark QQQ `
  --data-dir data\factor_exports\factors `
  --data-file data\factor_exports\factors\factors_all.parquet `
  --rebalance-mode horizon `
  --target-horizon-days 10 `
  --max-weeks 260 `
  --cost-bps 10 `
  --output-dir $out
```

Expected:
- `buckets_vs_nasdaq.csv`
- `buckets_vs_nasdaq.png`, `buckets_vs_nasdaq_cumulative.png`

4) Plot per-model vs QQQ:

```powershell
.\trading_env\Scripts\python.exe scripts\plot_single_models_vs_benchmark.py `
  --backtest-outdir $out `
  --models elastic_net xgboost lambdarank ridge_stacking `
  --benchmark QQQ `
  --output-dir $out
```

Expected:
- `per_model_topN_vs_benchmark.csv`
- `per_model_topN_vs_benchmark.png`, `per_model_topN_vs_benchmark_cum.png`

### Where each number came from (metric → file → column)

- avg_top_return_net → `results/paper_costs_nocat_20260106_135011/performance_report_20260106_135130.csv` → `avg_top_return_net`
- avg_top_return → same file → `avg_top_return`
- Rank_IC → same file → `Rank_IC`
- IC → same file → `IC`
- long_short_sharpe → same file → `long_short_sharpe`
- avg_top_turnover → same file → `avg_top_turnover`
- avg_top_cost → same file → `avg_top_cost`
- cost_bps → same file → `cost_bps`
- bucket cumulative net → `results/paper_costs_nocat_20260106_135011/buckets_vs_nasdaq.csv` → `cum_top_1_10_return_net`, `cum_top_11_20_return_net`, `cum_top_21_30_return_net`
- QQQ cumulative → same file → `cum_benchmark_return`

- avg_top_return_net_pct (OOS) → `results/t10_time_split_80_20_costs_20260106_212931/run_20260106_212934/oos_metrics.csv` → `avg_top_return_net_pct`
- avg_top_return_pct (OOS) → same file → `avg_top_return_pct`
- avg_benchmark_return_pct (OOS) → same file → `avg_benchmark_return_pct`
- end_cum_top_return_net_pct (OOS) → same file → `end_cum_top_return_net_pct`
- end_cum_benchmark_return_pct (OOS) → same file → `end_cum_benchmark_return_pct`

