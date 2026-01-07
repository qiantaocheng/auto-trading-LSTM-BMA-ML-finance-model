# Kronos K-Line Prediction Model

## Overview
Financial time series prediction using original Kronos model integrated with AutoTrader system.

## Features
- 🔮 OHLCV predictions
- 📊 Base model by default (recommended); large model optional
- 🌐 yfinance data source (Yahoo Finance)
- 🖥️ AutoTrader GUI integration

## Usage

### GUI
AutoTrader → "Kronos预测" tab → Enter symbol → Generate predictions

### Code
```python
from kronos import KronosService

service = KronosService()
result = service.predict_stock("AAPL", model_size="base", pred_len=30)
```

## Architecture
```
kronos/
├── kronos_model.py          # Model wrapper
├── kronos_service.py        # Service layer
├── kronos_tkinter_ui.py     # GUI component
├── polygon_data_adapter.py  # Data source
└── utils.py                 # Utilities
```

## Requirements
- `yfinance` installed and network access to Yahoo Finance
- If original Kronos model cannot be loaded, a statistical fallback predictor is used automatically