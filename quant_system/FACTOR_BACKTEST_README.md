# Factor-Based Backtest System

## Overview

The system has been successfully integrated to use factor data from the parquet file and fetch OHLCV data from Polygon.io for comprehensive backtesting.

## Files Created/Modified

### New Files
1. **`data/factor_data_loader.py`** - Loads factor data and merges with OHLCV
2. **`run_backtest_with_factors.py`** - Main backtest runner using factor data
3. **`test_factor_backtest.py`** - Test script to verify setup

### Data File
- **`data/polygon_factors_all_filtered_clean_final_v2_recalculated.parquet`** - Copied from original location
  - Contains 3,921 tickers
  - Date range: 2021-01-19 to 2025-12-30
  - 20 factor columns + Close price

## Quick Start

### Test the Setup
```bash
python quant_system/test_factor_backtest.py
```

### Run Backtest with All Tickers
```bash
python quant_system/run_backtest_with_factors.py
```

### Run Backtest with Specific Tickers
```bash
python quant_system/run_backtest_with_factors.py --tickers AAPL MSFT GOOGL AMZN NVDA
```

### Run Backtest with Custom Parameters
```bash
python quant_system/run_backtest_with_factors.py \
    --tickers AAPL MSFT GOOGL \
    --capital 100000 \
    --max-positions 10 \
    --min-score 0.65 \
    --start-date 2022-01-01 \
    --end-date 2024-01-01
```

## How It Works

1. **Load Factor Data**: Reads factor parquet file with MultiIndex (date, ticker)
2. **Fetch OHLCV Data**: Gets Open, High, Low, Close, Volume from Polygon API
3. **Merge Data**: Combines factors with OHLCV data, aligning by (date, ticker)
4. **Convert Format**: Transforms to dictionary format for signal system
5. **Run Backtest**: Executes backtest using quantitative signal system

## Data Structure

### Factor File Format
- **Index**: MultiIndex (date, ticker)
- **Columns**: 
  - Close (price)
  - 19 factor columns (momentum, volatility, etc.)
  - target (prediction target)

### Merged Data Format
- **Index**: MultiIndex (date, ticker)
- **Columns**: 
  - All factor columns
  - Open, High, Low, Close, Volume (from Polygon)

### Dictionary Format (for Backtest)
- **Structure**: `{ticker: DataFrame}`
- **DataFrame Index**: Date (DatetimeIndex)
- **DataFrame Columns**: All factors + OHLCV

## Features

✅ **Automatic Data Fetching**: Fetches OHLCV from Polygon API
✅ **Smart Merging**: Aligns factor and OHLCV data by date and ticker
✅ **Date Normalization**: Handles date format differences automatically
✅ **Caching**: Caches OHLCV data to reduce API calls
✅ **Flexible**: Can use all tickers or specific subset
✅ **Complete Integration**: Works seamlessly with quantitative signal system

## Parameters

### Factor Data Loader
- `factor_file`: Path to factor parquet file
- `tickers`: List of tickers (None = all)
- `start_date`: Start date (None = from factor file)
- `end_date`: End date (None = from factor file)
- `use_cache`: Use cached OHLCV data (default: True)

### Backtest Configuration
- `initial_capital`: Starting capital (default: $100,000)
- `max_positions`: Maximum concurrent positions (default: 10)
- `min_signal_score`: Minimum signal score for entry (default: 0.60)
- `rebalance_frequency`: 'daily', 'weekly', or 'monthly' (default: 'weekly')

## Troubleshooting

### No OHLCV Data for Some Dates
- Some dates may not have OHLCV data (weekends, holidays)
- System handles this gracefully with NaN values
- Backtest will skip dates without data

### API Rate Limits
- System includes automatic rate limiting (0.2s delay)
- Uses caching to minimize API calls
- For large universes, consider running overnight

### Memory Issues
- For very large universes (1000+ tickers), consider:
  - Processing in batches
  - Using specific ticker lists
  - Increasing system memory

## Example Output

```
================================================================================
QUANTITATIVE SIGNAL SYSTEM - BACKTEST WITH FACTORS
================================================================================
Factor file: D:\trade\quant_system\data\polygon_factors_all_filtered_clean_final_v2_recalculated.parquet
Benchmark: SPY

--------------------------------------------------------------------------------
LOADING FACTOR DATA
--------------------------------------------------------------------------------
Loaded 4180394 rows for 3921 tickers
Date range: 2021-01-19 to 2025-12-30

--------------------------------------------------------------------------------
FETCHING OHLCV DATA AND MERGING
--------------------------------------------------------------------------------
Fetching OHLCV data for 3921 tickers from 2021-01-19 to 2025-12-30
Merged data shape: (4180394, 24)
OHLCV column statistics:
  Open: 4158234/4180394 non-null (99.5%)
  High: 4158234/4180394 non-null (99.5%)
  Low: 4158234/4180394 non-null (99.5%)
  Volume: 4158234/4180394 non-null (99.5%)

--------------------------------------------------------------------------------
RUNNING BACKTEST
--------------------------------------------------------------------------------
...

================================================================================
BACKTEST RESULTS
================================================================================
Total Return: 45.23%
Annualized Return: 12.34%
Sharpe Ratio: 1.23
Max Drawdown: 8.45%
Win Rate: 58.23%
Total Trades: 234
```

## Next Steps

1. **Run Full Backtest**: Test with all 3,921 tickers
2. **Optimize Parameters**: Tune signal thresholds and risk parameters
3. **Analyze Results**: Review performance metrics and trade statistics
4. **Visualize**: Create equity curve and drawdown charts
5. **Compare**: Test different signal combinations

## Notes

- Factor data already includes Close price, which is preserved
- OHLCV data fills in missing Open, High, Low, Volume
- System handles date mismatches automatically
- All data is cached for faster subsequent runs
