# MultiIndex Factor Backtest Design

## Objective
Evaluate the Ridge-stacking Top-10 universe against a Nasdaq benchmark by building a repeatable backtest that consumes the MultiIndex factor snapshot `data/factors/factors_all.parquet`, generates model predictions, and compares cumulative returns to a Nasdaq proxy (e.g., QQQ) retrieved via yfinance. The goal is to have a self-contained process that can be re-run after each refresh of the factor file.

## Data Sources
1. **Factor inputs**: `data/factors/factors_all.parquet` (MultiIndex: `(date, ticker)`), produced by the polygon factor exporter. This file contains all features required by the BMA model.
2. **Model snapshot**: Latest ridge-stacking snapshot (e.g., `cache/model_snapshots/<id>/ridge_model.pkl`), used by `scripts/comprehensive_model_backtest.py`.
3. **Nasdaq benchmark**: QQQ (or ^IXIC) fetched via yfinance for the evaluation window.

## Workflow
### 1. Load Factors
- Use `pd.read_parquet('data/factors/factors_all.parquet')` and ensure the index is a MultiIndex with levels `['date','ticker']`.
- Restrict to the desired evaluation window via `_filter_date_window` in `ComprehensiveModelBacktest`.

### 2. Model Predictions (Comprehensive Backtest)
- Run `python scripts/comprehensive_model_backtest.py --data-file data/factors/factors_all.parquet --output-dir result/model_backtest --snapshot-id <optional>`.
- This produces per-model prediction parquet files (e.g., `ridge_stacking_predictions_<timestamp>.parquet`) with columns `date, ticker, prediction, actual`.
- From that output, extract Top-N predictions per date to represent the trading universe; keep actual forward returns for PnL calculation.

### 3. Portfolio Construction
- For each rebalance date (10 trading days apart when using `--rebalance-mode horizon`):
  1. Select Top-N tickers by `prediction`.
  2. Assign weights (equal weight or based on rank weights from `learning_summary.json`).
  3. Compute forward returns using the corresponding `actual` values.
- Accumulate returns by multiplying `(1 + weighted_return)` sequentially to obtain the strategy equity curve.

### 4. Nasdaq Benchmark via yfinance
- Fetch QQQ daily closes over the same date range:
  ```python
  import yfinance as yf
  qqq = yf.download('QQQ', start=start_date, end=end_date, auto_adjust=True)
  qqq['benchmark_ret'] = qqq['Close'].pct_change().fillna(0)
  qqq['benchmark_equity'] = (1 + qqq['benchmark_ret']).cumprod()
  ```
- Align benchmark dates with the strategy rebalance dates (e.g., forward-fill to the rebalance schedule) for apples-to-apples comparison.

### 5. Cumulative Return Comparison
- Combine the strategy equity curve with the benchmark equity series.
- Compute key metrics: total return, annualized return, max drawdown, Sharpe, and outperformance vs. QQQ.
- Plot cumulative curves (strategy vs. QQQ) and summarize in a CSV/Markdown report.

## Implementation Notes
- Reuse `ComprehensiveModelBacktest` to get consistent predictions and weekly summaries; no need to rebuild the prediction engine manually.
- Use the `result/model_backtest/ridge_stacking_weekly_returns_*.csv` files for precomputed weekly returns if desired.
- When comparing to QQQ, make sure to convert the weekly returns to the same frequency (e.g., 10-day horizon) to avoid mismatched periods.
- Store the final cumulative return analysis (strategy vs. Nasdaq) under `result/model_backtest/nasdaq_comparison/` with CSV plots for audit.

## Sample Command Sequence
```bash
# 1) Generate BMA predictions on the MultiIndex factor set
python scripts/comprehensive_model_backtest.py \
  --data-file data/factors/factors_all.parquet \
  --output-dir result/model_backtest \
  --rebalance-mode horizon --target-horizon-days 10

# 2) Derive Top10 returns and compare with QQQ (custom script)
python scripts/compare_top10_vs_nasdaq.py \
  --predictions result/model_backtest/ridge_stacking_predictions_*.parquet \
  --benchmark QQQ --output result/model_backtest/nasdaq_comparison
```

This design ensures the entire workflow is reproducible: read the MultiIndex factor file, run the comprehensive backtest to get predictions, construct the Top-N strategy, pull QQQ via yfinance, and compare cumulative returns. Update `learning_summary.json` if rank weights change so the same weighting scheme used live is reflected in the backtest.
