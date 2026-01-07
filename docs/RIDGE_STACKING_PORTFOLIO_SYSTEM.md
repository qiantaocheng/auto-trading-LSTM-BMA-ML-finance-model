## Ridge Stacking Portfolio System (BMA Ultra + HETRS Market Gating)

This describes the **decision + position sizing layer** built on top of:
- **BMA Ultra Ridge Stacking predictions** (per stock: `prediction`, optional `actual`)
- **HETRS-NASDAQ market signals** (QQQ: `tft_p50`, `meta_prob_success`, `market_regime`, `exposure_scalar`)

Goal: every ~2 weeks (T+10 trading days), select **Top‑10** stocks, produce **buy/hold/sell** deltas and **target weights**, and evaluate performance.

---

## Inputs

### 1) Ridge Stacking predictions (stock-level)
File: `results/**/ridge_stacking_predictions_*.parquet`

Expected columns:
- `date`: rebalance date
- `ticker`: stock symbol
- `prediction`: Ridge stacking score (higher = better rank)
- `actual` (optional): realized forward return for the holding period

### 2) Market signals (QQQ-level)
File: `data/hetrs_nasdaq/market_signals.parquet`

Generated from `hetrs_nasdaq/market_signals.py`, includes:
- `tft_p50`
- `meta_prob_success` (causal expanding RF)
- `market_regime` in {`favorable`,`neutral`,`unfavorable`}
- `exposure_scalar` in [0..1]

---

## Outputs

From the portfolio manager:
- `weights.csv`: target weights per rebalance date
- `trades.csv`: BUY/ADD vs SELL/REDUCE deltas per rebalance date
- `equity.csv`: equity curve (if `actual` exists)
- `metrics.json`: CAGR / Sharpe / MaxDD / turnover

---

## How decisions are made

Implemented in `scripts/ridge_stacking_portfolio_manager.py`:

1) For each rebalance date `date`:
   - rank stocks by `prediction` descending
   - select `Top‑N` (default 10)

2) Market gating (from HETRS signals):
   - **favorable** if (`tft_p50 > 0` and `meta_prob_success >= 0.70`) → `exposure_scalar = 1.0`
   - **unfavorable** if (`tft_p50 < 0` or `meta_prob_success <= 0.55`) → `exposure_scalar = 0.2`
   - else **neutral** → `exposure_scalar = 0.6`

3) Position sizing:
   - Base weights decay by rank
   - ranks 1–3 get a boost
   - ranks 7–10 are cut
   - weights are scaled by `exposure_scalar`
   - per‑name cap (default 15%)

4) Buy/Hold/Sell:
   - trades are the delta between current target weights and previous target weights
   - delta > 0 → BUY/ADD, delta < 0 → SELL/REDUCE, delta = 0 → HOLD

---

## Run steps (PowerShell)

### A) Build market signals (once you have `qqq_features_with_tft.parquet`)

```powershell
cd D:\trade
.\trading_env\Scripts\python.exe -m hetrs_nasdaq.market_signals --in data\hetrs_nasdaq\qqq_features_with_tft.parquet --out data\hetrs_nasdaq\market_signals.parquet --seed 42
```

### B) Run portfolio manager on a ridge predictions file

```powershell
cd D:\trade
$pred = "results\...\ridge_stacking_predictions_YYYYMMDD_HHMMSS.parquet"
$out = "results\ridge_portfolio_manager_run_" + (Get-Date -Format "yyyyMMdd_HHmmss")
New-Item -ItemType Directory -Force $out | Out-Null

.\trading_env\Scripts\python.exe scripts\ridge_stacking_portfolio_manager.py `
  --ridge-preds $pred `
  --market-signals data\hetrs_nasdaq\market_signals.parquet `
  --outdir $out `
  --top-n 10 `
  --rebalance-step 1 `
  --horizon-days 10 `
  --max-weight 0.15
```

---

## Notes / Next upgrades
- Add transaction costs & slippage (e.g. bp by volatility and turnover).
- Add risk constraints (sector caps, max single-name exposure, volatility targeting).
- Make the market gating thresholds and sizing policy fully learnable (meta-policy).



