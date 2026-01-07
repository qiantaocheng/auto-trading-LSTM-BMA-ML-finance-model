**Autotrader Workflow (Top‑10 + HETRS)**

---

### 1. Data → Model → Threshold Learning
1. **Historical data (12+ years)**  
   `python -m hetrs_nasdaq.data_loader --start 2012-01-01 --out data/hetrs_nasdaq/qqq_macro.parquet`  
   Downloads QQQ + macro series (US10Y, VIX, DXY) via yfinance.
2. **Feature engineering**  
   `python -m hetrs_nasdaq.features --in data/hetrs_nasdaq/qqq_macro.parquet --out data/hetrs_nasdaq/qqq_features.parquet`  
   Runs FFD, regime detection, technical indicators (see `hetrs_nasdaq/features.py`).
3. **TFT forecasting**  
   `python -m hetrs_nasdaq.tft_model --in data/hetrs_nasdaq/qqq_features.parquet --outdir results/hetrs_nasdaq/tft --predict-out data/hetrs_nasdaq/tft_preds.parquet`  
   Trains a Temporal Fusion Transformer (`hetrs_nasdaq/tft_model.py`), creates quantile predictions (`tft_p10/p50/p90`).
4. **Meta-label backtest**  
   `python -m hetrs_nasdaq.backtest_v2 --in data/hetrs_nasdaq/qqq_features_with_tft.parquet --outdir results/hetrs_nasdaq/backtest_v2_latest`  
   Performs CPCV with purge/embargo (`hetrs_nasdaq/backtest_v2.py`), outputs `meta_timeseries.csv`.
5. **Threshold learner**  
   `python scripts/learn_hetrs_thresholds.py --bma-predictions result/model_backtest/ridge_stacking_predictions_*.parquet --hetrs-timeseries results/hetrs_nasdaq/backtest_v2_latest/meta_timeseries.csv --top-n 10`  
   Merges ridge Top‑10 predictions with HETRS positions (`scripts/learn_hetrs_thresholds.py`). Produces:
   - `learning_summary.json` (entry/exit exposure thresholds, recommended Top‑K, rank weights)
   - CSVs for Top‑10 daily stats, per-rank performance, and Top‑K returns.

---

### 2. GUI / Universe Management (`autotrader/app.py`)
- **Initialization:** AutoTraderGUI bootstraps config, event loop, resource monitor, and DB (`__init__` ~lines 97–205).
- **Top‑10 state:** Maintains `cache/hetrs_top10_state.json` (`_top10_state_path` props). `_load_top10_refresh_state/_save_top10_refresh_state` persist refresh timestamps and tickers.
- **Biweekly refresh (`_maybe_refresh_top10_pool`)**  
  - Triggers at startup and again every even-week Monday (uses `_is_biweekly_monday`).  
  - Reads latest `ridge_stacking_predictions_*.parquet` (or `result/bma_top10.txt` fallback).  
  - Calls `_apply_top10_to_stock_pool`: `StockDatabase.clear_tickers()` → `batch_add_tickers()` → `_refresh_global_tickers_table()`; defaults to `['QQQ']` if no list.  
  - Logs the refresh and, if trading, schedules `_auto_sell_stocks` for removed names via `_run_async_safe`.
- **Autotrade start (`_start_autotrade`)**  
  1. Refresh Top‑10 pool again.  
  2. Connects to IBKR (`IbkrAutoTrader`).  
  3. Loads tickers via `_get_current_stock_symbols()` (now the Top‑10 list).  
  4. Writes the universe to `scanner.universe` in the config manager; kicks off the Engine loop.

---

### 3. Engine & Signals
- **Universe:** `Engine.on_signal_and_trade` now only iterates the Top‑10 tickers supplied by the DB.
- **Signal processor (`autotrader/unified_signal_processor.py`)**  
  - Primary signals still derived from Polygon/alpha factors.  
  - HETRS adapter sits in `_get_trading_signal` as a gating layer: for each ticker it checks HETRS `can_trade` before allowing buys.
- **HETRS integration (`autotrader/hetrs_signal_adapter.py`)**  
  - Loads `result/hetrs_learning/learning_summary.json` to set global entry/exit exposure thresholds (`HetrsConfig.entry_position_abs` and `.exit_position_abs`).
  - `get_signal(..)` converts raw HETRS positions to absolute exposures, enforces entry/exposure thresholds, and tags metadata (`raw_position`, `entry_threshold`, etc.).
  - Works as a “go/no-go” filter: signals only pass to the engine when HETRS exposure exceeds the learned entry threshold; sells are triggered when the exit threshold is hit.

---

### 4. Database & Stock Pool
- **SQLite storage (`autotrader/database.py`)**:  
  - `StockDatabase.get_all_tickers()` returns the current pool (now Top‑10).  
  - `clear_tickers()` wipes existing entries; `batch_add_tickers()` inserts fresh symbols.  
  - GUI tables and config combos are refreshed via `_refresh_global_tickers_table`, `_refresh_stock_lists`, etc.

---

### 5. Risk / Execution
- Position sizing, volatility gating, Kronos, and stop/TP modules remain as before:  
  - `autotrader/position_size_calculator.py`  
  - `autotrader/volatility_adaptive_gating.py`  
  - `autotrader/real_risk_balancer.py`
- `_auto_sell_stocks` liquidates removed tickers via `IbkrAutoTrader.place_market_order`.

---

### 6. Runbook Summary
1. Regenerate features/TFT/backtest thresholds as needed (steps 1–5).  
2. Launch GUI; `_maybe_refresh_top10_pool` ensures the stock pool matches the latest Top‑10.  
3. Start autotrading: Universe is restricted to Top‑10, per-stock signals come from Polygon factors, and HETRS exposure acts as a trade filter.  
4. Key artifacts:
   - `result/model_backtest/ridge_stacking_predictions_*.parquet`
   - `results/hetrs_nasdaq/backtest_v2_latest/meta_timeseries.csv`
   - `result/hetrs_learning/learning_summary.json`
   - `autotrader/system_paths.py` → `DEFAULT_DB_FILENAME` (stock pool DB)
