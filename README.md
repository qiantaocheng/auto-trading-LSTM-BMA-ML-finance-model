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
- Strict time alignment: feature lags, safety gaps, and T+1 targets
- PurgedGroupTimeSeriesSplit with gaps and embargo

### BMA Ultra Enhanced Model

- **File**: `bma_models/量化模型_bma_ultra_enhanced.py`
- **Architecture**: Ridge regression models with traditional ML baselines and regime-aware fusion
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

### Modernized Architecture (September 2025)

The algorithm has been completely redesigned with a simplified, production-ready architecture that eliminates previous complexity and failure modes:

**Core Principles:**
- Single DataFrame MultiIndex format (date, ticker) - no complex branching
- Fail-fast validation - no fallback cascades that mask errors
- Unified configuration system - all parameters centralized
- Direct ML pipeline - clean first-layer models with Ridge stacking
- Quality monitoring at every stage

### Enhanced Data Pipeline

**Data Sources & Quality Control:**
- **Polygon.io Integration**: Robust API client with cursor pagination and backoff
- **Quality Monitoring**: Real-time data validation with `alpha_factor_quality_monitor.py`
- **Temporal Safety**: Strict T-4 lag enforcement with safety gaps
- **Outlier Detection**: Multi-stage filtering with institutional-grade controls

**Factor Engineering (25 High-Quality Factors):**
- **Momentum Factors**: 5D/10D/20D momentum with quality filters and reversal signals
- **Technical Indicators**: RSI, Bollinger position/squeeze, MA ratios, trend strength
- **Volume Factors**: OBV momentum, Money Flow Index, volume-price correlation
- **Volatility Factors**: Parkinson estimator, GARCH(1,1), volatility clustering
- **Microstructure**: Bid-ask dynamics, trade imbalance, liquidity scoring

### Machine Learning Pipeline

**Simplified First-Layer Models:**
- **XGBoost**: Gradient boosting with optimized hyperparameters
- **CatBoost**: Categorical boosting for robust feature handling
- **ElasticNet**: Linear baseline with L1/L2 regularization

**Ridge Regression Stacking:**
- **Ridge Regression**: Linear meta-learner optimizing continuous returns
- **Isotonic Calibration**: Monotonic probability calibration
- **Temporal Validation**: Purged time series cross-validation with strict gaps

### Cross-Sectional Processing

**Robust Standardization:**
- **Cross-sectional Z-scoring**: Within-date normalization using `cross_sectional_standardizer.py`
- **Winsorization**: 1st/99th percentile clipping with outlier detection
- **Industry Neutralization**: Sector-adjusted signals when metadata available
- **Missing Value Handling**: Forward-fill with decay and median imputation

### Risk Model Enhancement

**Professional-Grade Risk Estimation:**
- **T-1 Size Factor**: Market cap based SMB factor with no look-ahead bias
- **Robust Factor Loadings**: Huber regression resistant to outliers
- **Covariance Estimation**: Ledoit-Wolf shrinkage with PSD projection
- **Specific Risk**: Residual variance with robust estimation methods

### Portfolio Optimization

**Institutional-Quality Construction:**
- **Objective Function**: Mean-variance optimization with turnover penalties
- **Risk Constraints**: Position limits, sector exposure caps, concentration limits
- **Execution Constraints**: Liquidity filters, order size limits, cash reserves
- **Robust Optimization**: Multiple fallback mechanisms with quality validation

### Quality Assurance Framework

**Comprehensive Monitoring:**
- **Evaluation Integrity**: `evaluation_integrity_monitor.py` for validation oversight
- **Production Gate**: `enhanced_production_gate.py` for deployment readiness
- **Exception Handling**: `enhanced_exception_handler.py` for robust error management
- **Performance Tracking**: Real-time monitoring with alerting capabilities

### Advanced Execution Engine

**Smart Order Management:**
- **Pre-execution Screening**: Liquidity, spread, and volatility analysis
- **Dynamic Pricing**: ATR-based limit pricing with tick size optimization
- **Risk Controls**: Real-time position monitoring and exposure limits
- **Order Throttling**: Per-cycle and daily order caps with queue management

**IBKR Integration:**
- **SMART Routing**: Intelligent order routing across exchanges
- **Order Types**: Market, limit, and bracket orders with advanced stops
- **Real-time Data**: Live market data with delayed fallback
- **Contract Qualification**: Automatic exchange selection and validation

### Production Monitoring

**Comprehensive Health Tracking:**
- **System Metrics**: Initialization status, failure rates, performance KPIs
- **Model Health**: Prediction quality, feature stability, regime detection
- **Execution Quality**: Fill rates, slippage analysis, latency monitoring
- **Risk Oversight**: Real-time exposure tracking and limit violations

**Configuration Management:**
- **Hot Reload**: Dynamic configuration updates without restart
- **Version Control**: Configuration change tracking and rollback
- **Environment Isolation**: Separate configs for dev/staging/production

## Implementation Reference

### Architectural Simplification (2025 Update)

The system has moved from complex multi-path architectures to a streamlined, fail-fast design:

**Single Data Format:** All data flows through DataFrame MultiIndex(date, ticker) - eliminating complex branching logic that previously caused alignment errors.

**Unified Configuration:** The `UnifiedConfig` class centralizes all parameters, replacing scattered hardcoded constants and complex inheritance hierarchies.

**Direct ML Pipeline:** First-layer models (XGBoost, CatBoost, ElasticNet) feed directly into Ridge Regression Stacking - no complex ensemble cascades.

### Enhanced Temporal Safety

**Strict Lag Enforcement:** Features use T-4 base lag plus safety gaps. The system validates alignment at every stage rather than relying on complex fallback mechanisms.

**Purged Cross-Validation:** Uses `unified_purged_cv_factory.py` with gap and embargo periods. The stacking meta-learner employs even stricter temporal constraints to prevent information leakage.

**Validation Gates:** Each stage includes temporal sanity checks that fail fast rather than masking errors with fallbacks.

### Production-Grade Processing

**Cross-Sectional Standardization:** The `cross_sectional_standardizer.py` handles within-date normalization, winsorization, and industry neutralization with robust statistical methods.

**Quality Monitoring:** Real-time factor quality assessment using `alpha_factor_quality_monitor.py` and `enhanced_alpha_quality_monitor.py` catches degradation before it affects predictions.

**Robust Numerics:** Enhanced numerical stability through `robust_numerical_methods.py` prevents common optimization failures and matrix conditioning issues.

### Advanced Risk Modeling

**Professional Risk Estimation:** Factor loadings use Huber regression, covariance estimation employs Ledoit-Wolf shrinkage with PSD projection, and specific risk calculation includes robust variance estimation.

**Institutional Controls:** Multiple validation layers ensure risk model quality and catch degenerate scenarios before they propagate to portfolio construction.

### Modern Execution Framework

**Smart Order Management:** Pre-execution screening covers liquidity, spreads, and volatility. Dynamic pricing uses ATR-based methods with proper tick size handling.

**Robust IBKR Integration:** SMART routing with automatic contract qualification, comprehensive error handling, and real-time vs delayed data fallback mechanisms.

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

## Project Structure (Modernized)

### Core Trading System
- `autotrader/app.py`: Main GUI application
- `autotrader/ibkr_auto_trader.py`: IBKR trading engine
- `autotrader/unified_trading_core.py`: Unified trading orchestration
- `autotrader/unified_error_handler.py`: Centralized error management
- `autotrader/enhanced_order_execution_with_state_machine.py`: Advanced order execution

### BMA Model Architecture
- `bma_models/量化模型_bma_ultra_enhanced.py`: Main BMA model
- `bma_models/optimized_factor_engine.py`: High-quality factor calculations (25 factors)
- `bma_models/ridge_stacker.py`: Ridge regression meta-learner
- `bma_models/cross_sectional_standardizer.py`: Robust cross-sectional processing
- `bma_models/unified_purged_cv_factory.py`: Temporal validation framework

### Quality & Monitoring
- `bma_models/alpha_factor_quality_monitor.py`: Factor quality assessment
- `bma_models/enhanced_alpha_quality_monitor.py`: Advanced quality monitoring
- `bma_models/evaluation_integrity_monitor.py`: Validation oversight
- `bma_models/institutional_monitoring_dashboard.py`: Production monitoring
- `bma_models/enhanced_production_gate.py`: Deployment readiness validation

### Infrastructure
- `bma_models/robust_numerical_methods.py`: Enhanced numerical stability
- `bma_models/enhanced_exception_handler.py`: Robust error handling
- `bma_models/unified_config_loader.py`: Configuration management
- `bma_models/unified_constants.py`: System constants and parameters
- `polygon_client.py`: Polygon API client with pagination and backoff

### Configuration & Data
- `bma_models/unified_config.yaml`: Main configuration file
- `autotrader/config/autotrader_unified_config.json`: Trading system config
- `bma_models/default_tickers.txt`: Default stock universe

## Security & Disclaimers

- Paper trade first. Real trading involves risk. Ensure market data permissions and exchange routing are correct.
- Manage credentials securely. Avoid committing API keys.

## Contributing

- Use clear commit messages and keep edits scoped
- Follow strict time‑leakage rules for any factor/label work
- Keep logging informative but rate‑limited where needed

## License

Proprietary usage within your environment unless otherwise specified.


