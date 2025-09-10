# BMA Model Data Requirements

## Critical Data Structure Requirements

The BMA Ultra Enhanced model requires specific data columns to function properly. This document outlines the mandatory data structure.

## Required Columns in feature_data

When calling `train_enhanced_models(feature_data)`, the `feature_data` DataFrame MUST contain:

### 1. Market Data Columns (MANDATORY)
At least one of these price columns is REQUIRED:
- `Close` or `close` - Closing price
- `Open` or `open` - Opening price  
- `High` or `high` - High price
- `Low` or `low` - Low price
- `Volume` or `volume` - Trading volume

### 2. Identification Columns (MANDATORY)
- `ticker` - Stock symbol/identifier
- `date` - Date of the observation
- `target` - Target variable for prediction (e.g., future returns)

### 3. Feature Columns
- Any additional features for model training (technical indicators, fundamental data, etc.)

## Example Data Structure

```python
feature_data = pd.DataFrame({
    # Market data (REQUIRED)
    'Close': [...],
    'Open': [...],
    'High': [...],
    'Low': [...],
    'Volume': [...],
    
    # Identifiers (REQUIRED)
    'ticker': [...],
    'date': [...],
    'target': [...],  # e.g., 10-day forward returns
    
    # Features (model-specific)
    'rsi': [...],
    'macd': [...],
    'momentum_5d': [...],
    # ... other features
})
```

## Why Market Data is Required

The regime detection module needs raw market data (especially Close prices) to:
1. Calculate market regimes (bull/bear/neutral)
2. Compute volatility and other market features
3. Properly weight model predictions based on market conditions

## Common Errors and Solutions

### Error: "Missing 'Close' column in data"
**Cause**: The feature_data passed to the model doesn't contain price columns.
**Solution**: Ensure your feature_data includes at least the Close price column.

### Error: "No market features found for regime detection"
**Cause**: The data only contains processed features without raw market data.
**Solution**: Include original OHLCV (Open, High, Low, Close, Volume) data in your feature_data.

## Data Preparation Example

```python
# Correct way to prepare data
def prepare_training_data(tickers, start_date, end_date):
    all_data = []
    
    for ticker in tickers:
        # Get market data
        market_data = get_market_data(ticker, start_date, end_date)
        
        # Calculate features
        features = calculate_features(market_data)
        
        # IMPORTANT: Combine market data WITH features
        combined_data = pd.concat([
            market_data[['Open', 'High', 'Low', 'Close', 'Volume']],  # Keep market columns
            features  # Add calculated features
        ], axis=1)
        
        combined_data['ticker'] = ticker
        combined_data['date'] = market_data.index
        combined_data['target'] = calculate_target(market_data['Close'])
        
        all_data.append(combined_data)
    
    feature_data = pd.concat(all_data, ignore_index=True)
    return feature_data
```

## Validation

Before training, validate your data:

```python
def validate_feature_data(feature_data):
    required_market_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
    required_id_cols = ['ticker', 'date', 'target']
    
    # Check for at least one market column
    has_market_data = any(col in feature_data.columns for col in required_market_cols)
    if not has_market_data:
        raise ValueError("feature_data must contain at least Close price column")
    
    # Check for required ID columns
    for col in required_id_cols:
        if col not in feature_data.columns:
            raise ValueError(f"feature_data must contain '{col}' column")
    
    return True
```

## Important Notes

1. **Don't remove market columns**: Even if you've calculated features from them, keep the original OHLCV data.
2. **Column names are case-sensitive**: The model looks for both 'Close' and 'close', but it's better to use standard names.
3. **MultiIndex support**: The data can also be in MultiIndex format with (date, ticker) as index levels.

## Contact

For issues or questions about data structure, check the model logs which will indicate exactly which columns are missing.