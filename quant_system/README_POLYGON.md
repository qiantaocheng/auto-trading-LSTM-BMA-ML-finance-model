# Polygon.io Integration Guide

This document describes the Polygon.io data integration for the Quantitative Signal System.

## Overview

The system has been completely adapted to use Polygon.io API for data fetching, replacing any Yahoo Finance dependencies. Data is stored in MultiIndex format (Symbol, Date) for efficient multi-stock operations and backtesting.

## Key Components

### 1. Polygon Data Fetcher (`data/polygon_data_fetcher.py`)

The `PolygonDataFetcher` class handles all data fetching from Polygon.io:

- **MultiIndex Format**: Stores data as (Symbol, Date) MultiIndex DataFrame
- **Caching**: Automatic caching to reduce API calls
- **Data Validation**: Cleans and validates OHLCV data
- **Progress Tracking**: Shows progress for large universes

### 2. Backtest Runner (`run_backtest.py`)

Comprehensive backtest runner that:
- Fetches 3 years of historical data from Polygon
- Runs backtest on specified universe
- Generates performance reports
- Compares against benchmark (SPY)

## Usage

### Basic Data Fetching

```python
from quant_system.data import PolygonDataFetcher

fetcher = PolygonDataFetcher()

# Fetch single symbol
df = fetcher.fetch_symbol('AAPL', '2021-01-01', '2024-01-01')

# Fetch universe (returns MultiIndex DataFrame)
universe_df = fetcher.fetch_universe(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    years=3
)

# Convert to dict format for signal system
universe_dict = fetcher.get_universe_dict(universe_df)
```

### Running Backtest

```bash
# Basic backtest with default symbols
python quant_system/run_backtest.py

# Custom universe
python quant_system/run_backtest.py --symbols AAPL MSFT NVDA --years 3

# Full options
python quant_system/run_backtest.py \
    --symbols AAPL MSFT GOOGL AMZN NVDA \
    --benchmark SPY \
    --years 3 \
    --capital 100000 \
    --max-positions 10 \
    --min-score 0.60
```

### Analyzing Single Stock

```python
python quant_system/main.py --analyze AAPL
```

## Data Format

### MultiIndex Structure

The data fetcher returns DataFrames with MultiIndex:

```
Index: (Symbol, Date)
Columns: Open, High, Low, Close, Volume
```

Example:
```
                    Open    High     Low    Close    Volume
Symbol Date                                                
AAPL   2021-01-04  133.52  133.61  126.76  129.41  143301900
       2021-01-05  128.89  131.74  128.43  131.01  97664900
MSFT   2021-01-04  222.53  224.30  221.20  222.42  23818000
       2021-01-05  222.60  223.49  221.00  223.00  19638700
```

### Dictionary Format (for Signal System)

The signal system expects `Dict[str, pd.DataFrame]` format:

```python
{
    'AAPL': DataFrame with Date index and OHLCV columns,
    'MSFT': DataFrame with Date index and OHLCV columns,
    ...
}
```

Use `fetcher.get_universe_dict(multiindex_df)` to convert.

## Configuration

### API Key Setup

The system uses the Polygon API key from:
1. Environment variable: `POLYGON_API_KEY` or `POLYGON_API_TOKEN`
2. `api_config.py`: `POLYGON_API_KEY` variable
3. Global `polygon_client` instance

### Cache Directory

Default cache directory: `data_cache/`

You can specify custom cache directory:
```python
fetcher = PolygonDataFetcher(cache_dir='custom_cache')
```

## Parameters

### Data Fetching Parameters

- `years`: Number of years of history (default: 3)
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD), default: today
- `use_cache`: Whether to use cached data (default: True)

### Backtest Parameters

- `initial_capital`: Starting capital (default: $100,000)
- `max_positions`: Maximum concurrent positions (default: 10)
- `min_signal_score`: Minimum signal score for entry (default: 0.60)
- `rebalance_frequency`: 'daily', 'weekly', or 'monthly' (default: 'weekly')

## Performance Considerations

1. **API Rate Limits**: The fetcher includes automatic rate limiting (0.2s delay)
2. **Caching**: Data is cached to avoid redundant API calls
3. **Batch Processing**: Universe fetching processes symbols sequentially with progress tracking
4. **Data Validation**: Invalid data points are automatically cleaned

## Troubleshooting

### No Data Returned

- Check API key is valid
- Verify symbol exists on Polygon
- Check date range (some symbols may not have data for full period)
- Review logs for specific error messages

### Import Errors

Make sure parent directory is in Python path:
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### Cache Issues

Clear cache directory if data seems stale:
```bash
rm -rf data_cache/
```

## Example: Complete Backtest Workflow

```python
from quant_system.data import PolygonDataFetcher
from quant_system.backtest.engine import BacktestEngine, BacktestConfig
from quant_system.core.signal_aggregator import SignalAggregator
from quant_system.config.settings import SystemConfig

# 1. Fetch data
fetcher = PolygonDataFetcher()
universe_df = fetcher.fetch_universe(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
    years=3
)
universe_dict = fetcher.get_universe_dict(universe_df)

benchmark = fetcher.fetch_symbol('SPY', '2021-01-01', '2024-01-01')

# 2. Initialize signal system
config = SystemConfig()
aggregator = SignalAggregator(config=config, benchmark_data=benchmark)

# 3. Configure backtest
backtest_config = BacktestConfig(
    initial_capital=100000,
    max_positions=10,
    min_signal_score=0.60
)

# 4. Run backtest
engine = BacktestEngine(
    signal_aggregator=aggregator,
    config=backtest_config,
    risk_config=config.risk
)

result = engine.run(
    universe_data=universe_dict,
    benchmark_data=benchmark,
    start_date='2021-01-01',
    end_date='2024-01-01'
)

# 5. View results
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
```

## Next Steps

1. **Expand Universe**: Add more symbols to test
2. **Parameter Optimization**: Adjust signal thresholds and risk parameters
3. **Extended Analysis**: Add more performance metrics
4. **Visualization**: Create equity curve and drawdown charts

## References

- Polygon.io API Documentation: https://polygon.io/docs
- Quantitative Signal System: See main README.md
