## BMA Ultra Enhanced Quant Trading System

### Overview

This repository contains a **production-grade** equity research and auto-trading stack for US stocks, featuring a streamlined codebase with institutional-quality components. The system is centered on the BMA Ultra Enhanced model with Polygon.io market data and IBKR trading integration.

**Recent Major Update (September 2025):** The codebase has undergone comprehensive cleanup and reorganization, removing over 100,000 lines of deprecated code while adding enhanced monitoring and quality control systems.

### Core Components

- **Enhanced BMA Model**: Institutional-grade factor engine with strict temporal controls
- **Risk Management**: T-1 Size factor model with robust portfolio optimization
- **Market Data**: Unified Polygon.io integration with comprehensive factor library
- **Trading Engine**: IBKR auto-trader with advanced execution and monitoring
- **Quality Control**: Production-ready validation and monitoring systems
- **Configuration**: Unified config management with hot-reload capabilities

### Key Features

- **Temporal Safety**: Strict time-leakage controls with purged cross-validation
- **Institutional Monitoring**: Alpha quality monitoring and evaluation integrity systems
- **Production Ready**: Enhanced exception handling and robust numerical methods
- **Risk Management**: Professional risk model with factor loadings and covariance estimation
- **Advanced Execution**: IBKR integration with SMART routing and bracket orders
- **Data Infrastructure**: Reliable Polygon API client with pagination and backoff
- **Quality Assurance**: Comprehensive validation and monitoring dashboard

## Quick Start

### Prerequisites

- Windows 10+ (tested), Python 3.9–3.11 recommended
- IBKR TWS or IB Gateway running, paper/live account ready
- Polygon.io API key (Starter or above recommended)

### Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you do not have a requirements file yet, install the primary dependencies:

```bash
pip install pandas numpy requests matplotlib seaborn scipy statsmodels scikit-learn xgboost lightgbm ib-insync pyyaml psutil
```

### Configure

- IBKR connection: `autotrader/data/connection.json` (created automatically when saved) or use GUI fields
- Risk config: `autotrader/data/risk_config.json` (also saved from GUI)
- Stock list: `stocks.txt` (root) or DB watchlist
- Polygon API key: edit `polygon_client.py` global instance near the bottom if needed (search for `polygon_client = PolygonClient("...")`)

### Run the GUI

```bash
python -m autotrader.app
```

In the app:

- Test connection → Start Auto Trading
- “一键运行BMA模型” to run the BMA Ultra pipeline and save recommendations (Excel/CSV under `result/`)
- Use tabs for Polygon, risk settings, database, engine status, logs

### Run the BMA Model (CLI)

```bash
python 量化模型_bma_ultra_enhanced.py --start-date 2023-01-01 --end-date 2025-12-31 --top-n 200 --config alphas_config.yaml
```

Environment overrides (optional): `BMA_START_DATE`, `BMA_END_DATE`, `BMA_EXTRA_ARGS`. The model internally clamps end_date to T‑1.

## Architecture

### Data and Alphas

- Unified Polygon factor library: `polygon_factors.py`
- Alpha engine: `enhanced_alpha_strategies.py`
- Strict time alignment: feature lags, safety gaps, and T+1~T+5 targets
- PurgedGroupTimeSeriesSplit with gaps and embargo

### BMA Ultra Enhanced Model

- **File**: `bma_models/量化模型_bma_ultra_enhanced.py`
- **Architecture**: Learning-to-rank models with traditional ML baselines and regime-aware fusion
- **Risk Model**: Professional-grade factor loadings, covariance matrices, and specific risk estimation
- **Quality Control**: Integrated alpha quality monitoring and evaluation integrity systems
- **Robust Methods**: Enhanced numerical stability and exception handling
- **Output**: Excel/CSV recommendations with comprehensive analysis reports

## Recent Improvements (September 2025)

### Codebase Modernization
- **Massive Cleanup**: Removed over 100,000 lines of deprecated code and temporary files
- **Streamlined Architecture**: Consolidated core functionality into `autotrader/` and `bma_models/` packages
- **Enhanced .gitignore**: Comprehensive exclusions for virtual environments and temporary data

### New Production-Grade Components
- **Alpha Quality Monitoring**: `alpha_factor_quality_monitor.py` and `enhanced_alpha_quality_monitor.py`
- **Institutional Dashboard**: `institutional_monitoring_dashboard.py` for comprehensive system oversight
- **Evaluation Integrity**: `evaluation_integrity_monitor.py` for validation and quality assurance
- **Robust Numerical Methods**: `robust_numerical_methods.py` for enhanced stability
- **Optimized Factor Engine**: `optimized_factor_engine.py` with improved performance

### Enhanced Infrastructure
- **Unified Configuration**: Streamlined config loading with `unified_config_loader.py`
- **Exception Handling**: Enhanced error management with `enhanced_exception_handler.py`
- **Training Monitoring**: `unified_training_monitor.py` for ML pipeline oversight
- **Cross-sectional Standardization**: Improved normalization with `cross_sectional_standardizer.py`

## Algorithm Details

### Data Pipeline and Leakage Controls

- Quote/Bar data via Polygon; historical aggregation uses `next_url` pagination and adjusted prices.
- Feature engineering enforces temporal isolation:
  - Features lagged by at least T-4 (base lag + safety gap)
  - Targets predict T+1 to T+5 windowed returns only from information available at or before T-4
  - Validation uses PurgedGroupTimeSeriesSplit with gap and embargo; stacking meta‑learner uses even larger gap/embargo to prevent cross‑fold leakage
- Explicit checks log alignment issues (warnings only) for shallow histories.

### Alpha Layer

- Core technical features: moving averages/ratios, volatility windows, RSI, price position in rolling high‑low bands, momentum windows (5/10/20), volume trend (20D), and derived microstructure factors consolidated via `polygon_factors.py`.
- Neutralization: within‑date cross‑section winsorization (1–99%), z‑scoring, and optional industry de‑meaning if metadata available.
- Outlier handling: clip extreme values; NaNs imputed (median for numeric) before ML.

### Learning‑to‑Rank and Traditional Models

- Traditional baselines: Ridge, ElasticNet, RandomForest; optional XGBoost/LightGBM if installed.
- OOF predictions using PGTS split; indices are re‑aligned to ensure strictly out‑of‑fold.
- Stacking: meta‑learner trained on first‑layer OOF with larger gap/embargo; time sanity checks (train_max + gap + embargo < test_min) enforced.

### Regime‑Aware Fusion

- Market regime detection from composite market index:
  - Rolling 21D volatility and trend; quantile thresholds define states (Bull/Bear, Low/High Vol, Normal)
- Per‑regime alpha weights emphasize momentum in Bull, quality/reversion in Bear, reversion in Volatile, balanced in Normal.
- Fusion combines ML predictions and weighted alpha signals; indices aligned before blending.

### Risk Model

- T‑1 Size factor: group by prior‑day free‑float market cap to form Small‑minus‑Big on T returns (no look‑ahead). Fallback: 60D average dollar volume proxy when cap unavailable.
- Factor loadings: robust regression (HuberRegressor) of each asset’s returns on factor set.
- Factor covariance: Ledoit‑Wolf shrinkage, projected to positive‑definite if needed.
- Specific risk: residual variance square‑root from fitted factor model.

### Portfolio Construction

- Inputs: expected_returns (from fused signals), covariance matrix assembled as B·F·Bᵀ + diag(S²).
- Constraints (configurable):
  - Position caps (e.g., max 3%), country/sector exposure guards, cash reserve, daily order cap
  - Optional liquidity rank (e.g., 20D volume) for soft constraints
- Objective (conceptual): maximize expected return minus risk_aversion · variance and turnover_penalty.
- Signal processing before optimization: cross‑section standardization, small amplification (e.g., ×0.02), micro‑jitter to avoid ties/flat solutions.
- Fallbacks: stratified or equal‑weight if the solver fails or covariance ill‑conditioned; metrics still computed.

### Risk‑Reward Controller (Execution Planning)

- For delayed‑data environments, an optional planner screens orders based on:
  - Liquidity (ADV USD), spreads (bps), volatility buckets (ATR), expected alpha bps
  - Dynamic limit pricing around last/prev close with tickSize and confidence
  - Throttling and per‑cycle order caps
  - Outputs symbol, side, quantity, limit price for submission

### Execution and Routing

- IBKR SMART routing by default: `Stock(symbol, exchange="SMART")`; optional primaryExchange hint for contract qualification.
- Orders: Market, Limit, and Bracket (parent market + take‑profit limit + stop). Optional local dynamic stops (ATR‑based, time‑decay) can replace server‑side stops if enabled.
- Real‑time vs Delayed data: auto fallback to delayed if permissions missing; price retrieval attempts subscription first, then Polygon close fallback.

### Monitoring and Health

- Health metrics track initialization and failure counts (alpha computation failures, optimization fallbacks, risk model failures, etc.).
- Unified config debounced reload avoids repeated noisy log lines.

## Implementation Reference (deeper details)

### Temporal Alignment (no look‑ahead)

The system enforces strict temporal alignment to prevent look-ahead bias. Features are lagged by a minimum base lag plus an additional safety gap, ensuring that predictions at time T only use information available before T. The target variable represents a windowed forward return, calculated as the ratio of future prices minus one. All feature columns are shifted by the combined lag amount within each ticker group, and the system validates that training data ends before test data begins with sufficient gaps and embargo periods.

### Purged CV and Stacking

Cross-validation uses PurgedGroupTimeSeriesSplit with configurable gaps and embargo periods to prevent data leakage between training and validation sets. The stacking meta-learner employs even stricter temporal constraints with larger gaps and embargo periods to ensure the second-layer model doesn't see future information from the first-layer predictions. The system validates that maximum training dates plus gaps and embargo periods are strictly less than minimum test dates.

### Industry/Date Cross‑Section Neutralization

Within each trading date, the system applies winsorization to the 1st and 99th percentiles, followed by z-score standardization. When industry metadata is available, the system performs cross-sectional neutralization by demeaning factors within industry groups, removing industry-specific biases from the alpha signals.

### Risk Model (T‑1 Size, Loadings, Covariance, Specific Risk)

The risk model constructs a Size factor using market capitalization data from the previous period to predict current returns, implementing the small-minus-big (SMB) factor. Factor loadings are estimated using robust regression techniques that are less sensitive to outliers. The factor covariance matrix is estimated using Ledoit-Wolf shrinkage and ensures positive semi-definiteness through eigenvalue adjustment. Specific risk is calculated as the square root of the variance of residuals from the factor model fit.

### Regime Detection and Fusion

Market regime detection analyzes 21-day rolling volatility and trend characteristics of a market index, classifying regimes using 33rd and 67th percentile thresholds. The system maintains separate alpha weights for each regime and combines predictions through a weighted fusion mechanism that balances machine learning predictions with regime-adjusted alpha signals.

### Portfolio Optimization

The optimization process aligns all inputs to common assets and constructs the covariance matrix using factor loadings, factor covariance, and specific risk. The objective function maximizes expected returns while penalizing portfolio risk and turnover, subject to various constraints including position limits, sector exposure caps, and country concentration limits. The system includes signal preprocessing to prevent flat solutions and implements multiple fallback mechanisms when the primary optimizer fails.

### Risk‑Reward Controller (optional)

This optional module screens symbols based on liquidity metrics, average daily volume in USD, bid-ask spreads in basis points, and average true range buckets. It computes conservative limit prices near the last or previous close with proper tick size rounding, implements per-cycle and daily throttling mechanisms, and returns structured order specifications including symbol, side, quantity, and limit price.

### Polygon Client: Pagination and Backoff

The Polygon client implements robust error handling with exponential backoff for rate limit errors, processes Retry-After headers, and supports cursor-based pagination using next_url tokens to fetch complete historical datasets. The system logs detailed error information including status codes, error messages, and request identifiers for debugging purposes.

### IBKR Contracts and Routing

Contract qualification uses SMART routing by default, with fallback mechanisms to specific exchanges when qualification fails. The system attempts qualification with the primary exchange first, then falls back to a predefined list of major exchanges including NASDAQ, NYSE, ARCA, and AMEX.

### Order Submission and Stops

The system implements bracket orders with a market order parent and limit order take-profit and stop-loss children, using transmit chaining for atomic execution. An optional local dynamic stop manager can cancel and replace server-side stops based on ATR-based trailing mechanisms with time decay, allowing for more sophisticated risk management.

### Unified Config Reload (debounced)

The configuration system implements debounced reloading that only logs and reloads when configuration files actually change. The get method checks for updates every 60 seconds and only calls the full reload process when file modification times indicate actual changes, preventing excessive logging and redundant operations.

### Polygon Client

- File: `polygon_client.py`
- Full error logging with status, message, request_id
- `next_url` cursor pagination for aggs endpoints
- 429/503 exponential backoff with Retry‑After support and inter‑request delay

### Auto‑Trader (IBKR)

- File: `autotrader/ibkr_auto_trader.py`
- IB (ib_insync) connectivity, account/position sync, market data subscription
- SMART routing via `Stock(symbol, exchange="SMART")`; optional `primaryExchange` hint
- Market/limit/bracket orders; optional local dynamic stops
- Unified risk check before order submission (exposure, caps, min notional)

### GUI

- File: `autotrader/app.py` (Tkinter)
- Tabs for database, import, risk, Polygon, engine, backtest
- Global scroll and log pane scroll
- Performance optimizer routes model execution without subprocess overhead

### Unified Configuration

- File: `autotrader/unified_config.py`
- Sources: defaults, files (risk/connection), database, hotconfig (`config.json`), runtime
- Automatic file‑change detection with debounced reload
- While reading (`get()`), only reloads if files actually changed

## Configuration Details

### Connection

- `connection.host`, `connection.port`, `connection.client_id`, `connection.account_id`
- `connection.use_delayed_if_no_realtime`: fallback to delayed quotes if real‑time unavailable

### Capital and Orders

- `capital.cash_reserve_pct`, `capital.max_single_position_pct`, `capital.max_portfolio_exposure`
- `orders.min_order_value_usd`, `orders.daily_order_limit`

### Signals and Sizing

- `signals.acceptance_threshold` (e.g., 0.6)
- `sizing.per_trade_risk_pct`, `sizing.max_position_pct_of_equity`, `sizing.notional_round_lots`

### Risk Controls

- `risk_controls.sector_exposure_limit`, `risk_controls.max_correlation`, dynamic stops toggle

## Operating Notes

### Trading Hours and Filters

- Auto‑trader runs during US market hours 9:30–16:00 ET by default
- Filters include price band [$2, $800], min order value, cash reserve, daily order caps

### Routing

- Default SMART routing; manual exchange is possible by passing `primary_exchange` to `qualify_stock()` or changing `exchange` (not recommended)

### Data Quality and T‑1 Size Factor

- Risk model Size factor uses T‑1 market cap for T returns (no look‑ahead)
- Fallback to liquidity proxy if full market cap data unavailable

## Troubleshooting

- “Starting to load all configuration sources…” repeated: addressed via debounced file‑change reload
- No orders placed:
  - Not in trading hours; price not available; filters (min order value, price band) block; risk validation failed; hit daily/per‑cycle order caps
  - Check GUI risk settings; try lowering `min_order_value_usd`, `cash_reserve_pct`, raising `max_single_position_pct`
- Polygon errors: check logs for `status`, `request_id`, message; ensure API key; pagination enabled by default

## Project Layout (selected)

- `autotrader/app.py`: GUI
- `autotrader/ibkr_auto_trader.py`: trading engine
- `autotrader/unified_config.py`: config manager
- `autotrader/unified_*`: connection/position/risk managers
- `autotrader/enhanced_order_execution.py`, `order_state_machine.py`, `trading_auditor_v2.py`: execution & auditing
- `autotrader/data_source_manager.py`, `autotrader/polygon_unified_factors.py`: data/factors integration
- `polygon_client.py`: Polygon API client
- `polygon_factors.py`: consolidated factor library
- `量化模型_bma_ultra_enhanced.py`: BMA model

### Legacy/Optional (not required for main flow)

- `autotrader/backtest_engine.py`, `autotrader/backtest_analyzer.py`
- `autotrader/database_pool.py`, `autotrader/engine_logger.py`
- `autotrader/enhanced_config_cache.py`, `autotrader/indicator_cache.py`, `autotrader/enhanced_indicator_cache.py`
- `autotrader/client_id_manager.py`
- `autotrader/launcher.py` (needed only for CLI launching alternative to GUI)

## Security & Disclaimers

- Paper trade first. Real trading involves risk. Ensure market data permissions and exchange routing are correct.
- Manage credentials securely. Avoid committing API keys.

## Contributing

- Use clear commit messages and keep edits scoped
- Follow strict time‑leakage rules for any factor/label work
- Keep logging informative but rate‑limited where needed

## License

Proprietary usage within your environment unless otherwise specified.


