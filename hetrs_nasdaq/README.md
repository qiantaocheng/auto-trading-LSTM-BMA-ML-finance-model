## HETRS-NASDAQ (QQQ) — research pipeline

This folder is **self-contained** and does not modify your existing `bma_models` stack.

### What it includes
- **Module 1**: `data_loader.py` — download QQQ + macro (US10Y, VIX, DXY), align, compute returns + Garman-Klass vol
- **Module 2**: `features.py` — Fixed-Width Window Fractional Differentiation (FFD) + ADF-driven `d` search, GMM regimes, RSI/MACD/BB width
- **Module 3**: `tft_model.py` — Temporal Fusion Transformer training/inference (PyTorch Forecasting)
- **Module 4**: `rl_agent.py` — custom trading env + differential Sharpe reward + PPO/A2C/DDPG + weighted ensemble
- **Module 5**: `backtest.py` — CPCV with purge/embargo + triple barrier evaluation + metrics + plots

### Install (recommended: separate venv)

Create a fresh venv and install:

```bash
pip install -r requirements-hetrs_nasdaq.txt
```

### Run order (minimal)

1) Download and build base dataset:

```bash
python -m hetrs_nasdaq.data_loader --start 2010-01-01 --out data/hetrs_nasdaq/qqq_macro.parquet
```

2) Feature engineering:

```bash
python -m hetrs_nasdaq.features --in data/hetrs_nasdaq/qqq_macro.parquet --out data/hetrs_nasdaq/qqq_features.parquet
```

3) Train TFT (optional / heavy):

```bash
python -m hetrs_nasdaq.tft_model --in data/hetrs_nasdaq/qqq_features.parquet --outdir results/hetrs_nasdaq/tft --seed 42 --predict-out data/hetrs_nasdaq/tft_preds.parquet
```

4) Train RL agents (optional / heavy):

```bash
python -m hetrs_nasdaq.rl_agent --in data/hetrs_nasdaq/qqq_features.parquet --outdir results/hetrs_nasdaq/rl --seed 42
```

5) CPCV backtest:

```bash
python -m hetrs_nasdaq.backtest --in data/hetrs_nasdaq/qqq_features.parquet --rl_dir results/hetrs_nasdaq/rl --outdir results/hetrs_nasdaq/backtest --seed 42
```

### Threshold policy (recommended if you want explicit buy/sell thresholds)

This mode **learns buy/sell thresholds on each training fold** (deterministic grid over quantiles),
then applies those thresholds to that fold’s test segment (CPCV, with purge/embargo).

It saves:
- `cpcv_equity.png`: Strategy vs QQQ buy&hold
- `learned_thresholds.csv`: per-fold learned thresholds and in-fold Sharpe

Example:

```bash
python -m hetrs_nasdaq.backtest --in data/hetrs_nasdaq/qqq_features.parquet --outdir results/hetrs_nasdaq/backtest_threshold --policy threshold --signal-col tft_p50 --seed 42
```


