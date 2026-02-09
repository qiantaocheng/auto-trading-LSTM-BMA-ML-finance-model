# `autotrader/app.py` Deep Dive

This file implements the Tkinter-based front end that orchestrates IBKR auto-trading, Polygon factor ingestion, advanced modelling (BMA, LambdaRank, CatBoost, etc.), and operational tooling. Below is a section-by-section breakdown of the module.

## 1. Global configuration & shared state (`app.py:19-228`)
- Defines default universes (e.g., `DEFAULT_AUTO_TRAIN_TICKERS`) and time horizons used across auto-training, timesplit experiments, and direct predictions.
- Hard-codes factor export locations (`STAGE_A_DATA_PATH`, `DIRECT_PREDICT_TICKER_DATA_PATH`) and smoothing constants (`DIRECT_PREDICT_EMA_WEIGHTS`, guardrail `DIRECT_PREDICT_MAX_CLOSE`).
- `_attach_tooltip` adds lightweight help bubbles without extra dependencies.
- `AppState` (`app.py:190`) captures UI selections (file paths, ticker lists) plus IBKR connection parameters and trading defaults, acting as the single mutable snapshot that `_capture_ui` writes back to disk/config.

## 2. Application bootstrap (`app.py:229-390`)
- `AutoTraderGUI` inherits `tk.Tk`, boots configuration/event loop/resource monitoring via the `bma_models` helpers, and wires up an async event bus (`GUIEventAdapter`).
- Initialises database (`StockDatabase`), log buffers, Tk widgets, resource monitor alerts, and strategy helpers.
- `_init_enhanced_trading_components` (`app.py:391`) lazily imports sizing/gating modules and configures high/low allocation thresholds, ATR-driven volatility gating, etc.
- `_init_strategy_components` (`app.py:451`) loads unified Polygon factor service, risk balancer adapter and sets initial health flags inside `self.strategy_status` so later tabs can show readiness.

## 3. Polygon integration (`app.py:543-932`)
- `_init_polygon_factors`, `_ensure_polygon_factors`, and `get_dynamic_price` provide multi-method fallbacks to fetch intraday data from `polygon_client` (current price, snapshot, last trade, intraday bars).
- `_build_polygon_tab` (`app.py:1344`) defines UI for connection state, return comparison, Excel-driven Top20 T+5 backtests, risk balancer toggles, cache controls, and stats.
- `_compare_polygon_returns` (`app.py:1572`) orchestrates yfinance market-cap filters, downloads daily bars via Polygon, compares vs QQQ, and reports average/excess returns.
- `_compare_returns_from_excel` (`app.py:2034`) processes user Excel sheets, normalises target tickers, downloads Polygon histories, computes realised vs SPY returns, writes results to `backtest_results`, and updates GUI.
- `_enable_polygon_factors`, `_clear_polygon_cache`, `_toggle_polygon_balancer`, `_open_balancer_config`, `_update_polygon_status`, `_schedule_polygon_update` keep the Polygon pipeline healthy, including showing cache hit stats and balancer status.

## 4. Logging and core UI (`app.py:651-1040`)
- `_build_ui` creates a scrollable Notebook with tabs for Data Services, File Imports, Risk Engine, Polygon API, Strategy Engine, Direct Trading, Time Split, Backtesting, Prediction, Kronos, and Temporal Stacking.
- Adds top-level action buttons (start/stop trading, clear log, account view, DB maintenance) and a status/log panel.
- `log` mirrors everything to stdout and the Tk text widget with threading-safe buffering.

## 5. Risk Engine (`app.py:1020-1342`)
- `_build_risk_tab` exposes stop/target, allocation, price bounds, cash reserve, per-trade limits, ATR settings, webhook URL, and toggles for shorting/bracket removal.
- `_risk_load`/`_risk_save` pull/persist JSON blobs via `StockDatabase`, convert percentages, and push selections to the runtime `config_manager` (ensuring live Engine honours limits).

## 6. Strategy Engine tab (`app.py:2932-4872`)
- `_build_engine_tab` offers buttons for starting/stopping the async engine, running one-off signal cycles, generating signals, toggling risk balancer, and launching diagnostics (BMA model, α-factor tests, etc.).
- `self.strategy_status_text` is populated by `_update_strategy_status`, summarising alpha, polygon, risk balancer and model readiness.
- `_run_bma_model` (`app.py:9758`) pipes selected tickers (prompting via `_show_stock_selection_dialog`) through optional auto-training (`_auto_build_multiindex_training_file`) and kicks off `UltraEnhancedQuantitativeModel` training in a background thread with Tk-friendly logging.

### 6.1 Direct Predict snapshot pipeline (`app.py:3034-4804`)
- Loads tickers from multi-index parquet, selected pools, or manual input; auto-detects horizon via `get_time_config`.
- Uses `Simple17FactorEngine` to fetch >280 trading days of Polygon data, building all 17 factors plus optional Sato features, and standardises indices to match training snapshots.
- Feeds pre-computed features to `UltraEnhancedQuantitativeModel.predict_with_snapshot` for a fixed snapshot (`DIRECT_PREDICT_SNAPSHOT_ID`), capturing raw/base predictions, deduplicating MultiIndex rows, and logging per-date coverage.
- `_apply_direct_predict_ema` ensures UI + Excel outputs match CLI smoothing, applying a weighted EMA over recent predictions.
- Generates Excel ranking reports via `scripts/direct_predict_ewma_excel.generate_excel_ranking_report`, logs top-N per model (MetaRanker, CatBoost, LambdaRank, ElasticNet, XGB), and archives to SQLite for audit.

### 6.2 Manual/diagnostic helpers
- `_test_connection`, `_test_market_data`, `_test_order_placement`, `_test_strategy_components`, `_test_risk_controls`, `_manual_signal_entry`, `_execute_alpha_signals`, `_portfolio_rebalance`, `_view_factor_analysis`, `_toggle_risk_balancer`, `_view_risk_stats`, `_update_system_status` provide quick health checks and manual overrides.

## 7. Direct trading tab (`app.py:4874-5440`)
- `_build_direct_tab` surfaces market/limit/bracket/algo order widgets; `_direct_market`, `_direct_limit`, `_direct_bracket`, `_direct_algo` borrow `_ensure_loop`/`loop_manager` to submit asynchronous IBKR orders with logging + message dialogs for invalid inputs.
- `_delete_database`/`_print_database` allow clearing/inspecting the SQLite-based symbol store straight from the tab.

## 8. Time Split (80/20) tab (`app.py:5021-4859/15248+`)
- `_build_timesplit_tab` lets users choose the Stage-A parquet, adjust train/test split slider, pick feature subset, and toggle EMA smoothing before running out-of-sample evaluation.
- `_run_timesplit_eval` spawns `scripts/time_split_80_20_oos_eval.py` as a subprocess, streams stdout into the Tk log, and `_open_timesplit_results` opens `results/timesplit_gui`. (The same three methods are duplicated near the file end after `main()`.)

## 9. Data Services tab (`app.py:5618-5996`)
- `_build_database_tab` visualises tickers stored in `StockDatabase`, supports single/batch add/remove, and exposes config save/load combos.
- Integrates with the stock pool manager (`_open_stock_pool_manager`) and factor export service; `_export_factor_dataset` spawns a worker thread that calls `factor_export_service.export_polygon_factors` with the current pool and writes progress to the UI.

## 10. File Imports tab (`app.py:5998-6120, 6072-9386`)
- `_build_file_tab` allows selecting JSON, Excel sheets/columns, or inline CSV to derive tickers. `_pick_json`/`_pick_excel` update `AppState` and labels.
- `_import_file_to_database` replaces the DB with parsed tickers (showing preview + confirmation, optionally auto-selling removed positions via `_auto_sell_stocks`).
- `_append_file_to_database` incrementally adds tickers.
- `_extract_symbols_from_files` consolidates JSON arrays, Excel columns, and CSV input, deduping while uppercasing as needed.

## 11. Event loop + async infrastructure (`app.py:6120-6522`)
- `_ensure_loop` spins a dedicated asyncio loop inside a daemon thread, synchronises via `_loop_ready_event`, cancels leftover tasks on shutdown, and logs issues through `self.after` to avoid Tk thread issues.
- `_run_async_safe` centralises coroutine submission, preferring the shared `loop_manager` when available, else launching isolated threads with their own event loop.

## 12. Direct Predict helpers (`app.py:6248-6392`)
- `_get_direct_predict_features` reads the specific snapshot manifest (falling back to `TOP_FEATURE_SET`) and caches the list keyed by snapshot ID.
- `_apply_direct_predict_ema` enforces MultiIndex, normalises EMA weights, reuses raw score columns, and logs the smoothing coverage.
- `_capture_ui` and `_normalize_ticker_input/_normalize_ticker_list/_batch_import_global` glue UI text fields to `AppState` and to DB operations.

## 13. Connection lifecycle & trading automation (`app.py:6524-7086`)
- `_test_connection` reuses `_capture_ui`, builds a fresh `IbkrAutoTrader`, and validates connectivity asynchronously.
- `_start_autotrade` closes stale sessions, reinitialises trader + engine, refreshes top10 pool, and launches a perpetual `_engine_loop` calling `engine.on_signal_and_trade()` with the configured polling interval.
- `_stop` / `_disconnect_api` stop scheduled tasks, cancel loops, close IBKR sessions, and reset UI indicators.
- `_show_stock_selection_dialog`, `_compute_prediction_window`, `_auto_build_multiindex_training_file` provide the selection/autotrain building blocks for BMA/backtests.

## 14. Stock pools & Top10 automation (`app.py:7938-9074`)
- `_refresh_stock_lists`, `_refresh_configs`, `_save_config`, `_load_config`, `_refresh_global_tickers_table`, `_add_ticker_global`, `_delete_selected_ticker_global`, `_batch_import_global`, `_sync_*` keep the pool + DB in sync with UI selection boxes.
- `_get_current_stock_symbols` surfaces the DB universe for engine use.
- `_load_top10_refresh_state`, `_save_top10_refresh_state`, `_is_biweekly_monday`, `_load_top10_from_predictions/_text`, `_apply_top10_to_stock_pool`, `_maybe_refresh_top10_pool` drive fortnightly replacements of the stock pool using BMA predictions or manual text files, optionally auto-selling removed positions by calling `_auto_sell_stocks` via async tasks.

## 15. File import back-end (`app.py:9114-9496`)
- `_import_file_to_database` and `_append_file_to_database` handle confirmation prompts, update logs, and optionally trigger `_auto_sell_stocks` when tickers are removed.
- `_extract_symbols_from_files` supports JSON arrays, Excel sheet/column references, and inline CSV, producing an ordered unique ticker list.
- `_on_resource_warning` routes background monitoring alerts into the log; `_on_closing` contains comprehensive shutdown logic (cancels engine loop tasks, sets trader stop events, closes event loops, persists config, and finally destroys Tk window).

## 16. Backtesting & advanced modelling (`app.py:9758-12380`)
- `_run_bma_model` covers per-section logging, data generation, inference, and workbook output.
- `_build_backtest_tab` (`app.py:10180`) wraps manual ticker lists, multiple backtest types (professional, autotrader, weekly), configurable capital windows, output directories, and buttons for quick/comprehensive runs.
- `_build_prediction_tab` (`app.py:10704`) adds a dedicated panel for running prediction-only jobs, including pool integration, horizon selection, indeterminate progress bars, and status console.
- `_build_kronos_tab` (`app.py:10892`) embeds the Kronos predictor UI when dependencies are installed, else surfaces the commands needed to install them.
- `_build_temporal_stacking_tab` (`app.py:10974-11836`) is a full UI for LambdaRank temporal stacking: selecting the parquet dataset, target column, lookback, LightGBM-like hyperparameters, IPO handling, run controls, integrity checks, and state viewing.
- `_run_temporal_stacking_training`, `_validate_temporal_integrity`, `_view_temporal_state` implement the associated jobs via subprocesses or direct pandas/LightGBM calls (see file for specifics).

## 17. Status & monitoring (`app.py:12428-12930`)
- `_build_status_panel`, `_start_status_monitor`, `_update_status` update labels for connection, engine, alpha model, account net-liq, account/client IDs, position counts, daily trade totals, watchlist size, timestamps, and aggregate signal lights.
- `_update_signal_status`, `_set_connection_error_state`, `_update_daily_trades`, `_update_strategy_status` drive finer-grained messaging.
- `_test_alpha_factors`, `_run_bma_model_demo`, `_generate_trading_signals`, `_load_polygon_data`, `_compute_t5_factors`, `_view_factor_analysis`, `_toggle_risk_balancer`, `_view_risk_stats`, `_reset_risk_limits`, `_update_system_status`, `_test_connection/_market_data/_order_placement`, `_run_full_system_test`, `_test_strategy_components`, `_test_risk_controls`, `_manual_signal_entry`, `_execute_alpha_signals`, `_apply_enhanced_signal_processing`, `_portfolio_rebalance`, `_add_backtest_stock`, `_import_stocks_from_db`, `_clear_backtest_stocks`, `_remove_selected_stocks`, `_run_professional_backtest` flesh out diagnostics and workflows tied to that panel.

## 18. Menu/toolbar/auxiliary windows (`app.py:14826-14900`)
- `_ensure_top_menu`, `_ensure_toolbar`, `_open_return_comparison_window` add quick access menus for functions like the Polygon return comparison window even outside the Polygon tab.

## 19. Entry points & duplication (`app.py:15040-15316`)
- `init_batch` initialises config, event loop, resource monitor, and database without Tk; returns a dict of running services for headless use.
- `main` inspects `--batch` flag, supports console-only runs (`init_batch`) or launches the GUI, ensuring coroutine loop + resource monitor shut down gracefully when the window closes.
- After `if __name__ == "__main__"`, the file currently repeats the three timesplit methods; this duplication likely arose from a copy/paste merge and should be cleaned up.

## 20. Key architectural themes
- **Event-driven layering:** UI triggers call into async coroutines via `loop_manager`, which handles IBKR I/O without blocking Tk.
- **Persistent config/state:** Most UI controls either read from or write to `StockDatabase` or `config_manager`, meaning manual changes survive restarts and feed the trading engine.
- **Thread safety:** Long-running operations (Polygon comparisons, factor exports, BMA training) are offloaded to background threads with `self.after` callbacks updating widgets.
- **Data validation:** Major workflows include guardrails (market-cap filters, duplicate-index checks, invalid price removal) and fallback logic to keep outputs sane even when upstream data is messy.

Use this document as a map when modifying `app.py`. Each section references the relevant method names and line numbers for quick navigation.
