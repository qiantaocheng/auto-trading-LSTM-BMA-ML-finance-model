"""
HETRS-NASDAQ: A self-contained research pipeline for QQQ forecasting + RL execution + CPCV backtesting.

Modules:
- data_loader.py: download and clean market + macro data
- features.py: feature engineering (FFD, regime detection, indicators)
- tft_model.py: Temporal Fusion Transformer training/inference
- rl_agent.py: RL trading environment + ensemble agent
- backtest.py: CPCV + triple barrier evaluation + plots
"""


