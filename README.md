# Quantitative Trading Management Software

A powerful platform for quantitative trading analysis, integrating advanced machine learning models, factor analysis, and backtesting to support data-driven investment decisions.

## Table of Contents

* [Features](#features)
* [Requirements](#requirements)
* [Quick Start](#quick-start)
* [Installation](#installation)
* [How to Use](#how-to-use)
* [Project Layout](#project-layout)
* [Technical Overview](#technical-overview)
* [Troubleshooting](#troubleshooting)
* [License](#license)

---

## Features

### Core Capabilities

* **Quantitative Analysis**: Machine-learning-based stock scoring and selection
* **Multi-Factor Models**: Combines technical, fundamental, and sentiment factors
* **Backtesting**: Historical data validation and strategy evaluation
* **Real-Time Monitoring**: Live data updates and alerts
* **Result Management**: Automatic saving of reports and summaries

### Machine Learning

* **Ensemble Models**: XGBoost, LightGBM, CatBoost fusion
* **Factor Selection**: Information Coefficient filtering and multicollinearity checks
* **Outlier Handling**: Winsorization of extreme values
* **Factor Neutralization**: Removes size and sector biases
* **Hyperparameter Tuning**: Automated parameter search

### User Interface

* **Modern GUI**: Built with Tkinter
* **Animations**: Hover effects and load indicators
* **Custom Background**: Image support
* **Status Display**: Progress bar and status messages
* **Notifications**: Desktop alerts on task completion

### Data Analysis

* **Multi-Stock Support**: Customizable stock pools
* **Flexible Date Ranges**: User-defined analysis periods
* **Export Options**: Excel and CSV output
* **Charts**: Automatic plot generation
* **History**: SQLite storage of past analyses

---

## Requirements

### Minimum

* **OS**: Windows 10/11, Linux, or macOS
* **Python**: 3.8 or higher
* **Memory**: 4 GB RAM
* **Disk**: 2 GB free
* **Network**: Internet access

### Recommended

* **OS**: Windows 11 or Ubuntu 20.04+
* **Python**: 3.10 or higher
* **Memory**: 8 GB RAM or more
* **Disk**: 5 GB free
* **Network**: High-speed connection

---

## Quick Start

### One-Click Launch

```bash
# Windows
run “start_trading_software.bat”

# Linux/Mac
python3 quantitative_trading_manager.py
```

### Command-Line

```bash
# Activate your virtual environment
source trading_env/bin/activate   # Linux/Mac
trading_env\Scripts\activate.bat  # Windows

# Run the program
python quantitative_trading_manager.py
```

---

## Installation

1. **Create a Virtual Environment (optional)**

   ```bash
   python -m venv trading_env
   ```

2. **Install Core Dependencies**

   ```bash
   pip install -r requirements_portable.txt
   ```

3. **Verify Installation**

   ```bash
   python -c "import pandas, numpy, yfinance; print('OK')"
   ```

---

## How to Use

1. **Launch the Software** by running the script or batch file.
2. **Choose an Analysis**:

   * Quantitative model
   * Backtest
   * Machine learning backtest
3. **Set Parameters**: Pick date range, stock pool, and model settings.
4. **Run Analysis**: Click the button to start.
5. **View Results**: Check the “result” folder for Excel, CSV, and charts.



## Technical Overview

* **Language**: Python 3.8+
* **GUI**: Tkinter
* **Database**: SQLite
* **Data Handling**: pandas, NumPy
* **ML**: scikit-learn, XGBoost, LightGBM, CatBoost
* **Visualization**: matplotlib
* **Data Source**: yfinance

### Layers

1. **User Interface** (Tkinter)
2. **Business Logic** (Analysis engines, models, backtester)
3. **Data Access** (SQLite, local files, web API)
4. **External Data** (yfinance)

---

## Troubleshooting

* **Startup Errors**: Check Python version and dependencies.
* **Dependency Issues**: Upgrade pip, try a different mirror, reinstall.
* **Database Errors**: Use provided scripts to inspect or reset the SQLite file.
* **Data Fetch Problems**: Test yfinance connectivity with a simple script.

---

## License

This project is released under the MIT License.
