# Prediction-Only System Guide

## Overview

The Prediction-Only system allows you to use trained BMA models to generate predictions **without retraining**. It loads saved model snapshots and runs fast predictions on new stocks.

## Features

✅ **No Training Required** - Uses saved model snapshots
✅ **Fast Predictions** - Generates predictions in seconds
✅ **All 5 Models** - ElasticNet, XGBoost, CatBoost, LambdaRank, Ridge
✅ **17 Alpha Factors** - Uses Simple17FactorEngine for feature calculation
✅ **GUI Integration** - Built into the stock pool management system

## Architecture

```
User Input Stocks
       ↓
Load Model Snapshot (ElasticNet + XGBoost + CatBoost + LambdaRank + Ridge)
       ↓
Fetch Market Data (Polygon API)
       ↓
Calculate 17 Alpha Factors (Simple17FactorEngine)
       ↓
First Layer Predictions (ElasticNet, XGBoost, CatBoost, LambdaRank)
       ↓
Ridge Stacker Fusion
       ↓
Top N Recommendations
```

## Usage

### 1. Through GUI (Recommended)

1. Launch the AutoTrader app:
   ```bash
   python autotrader/app.py
   ```

2. Go to **BMA回测** tab

3. Add stocks to the stock pool:
   - Type ticker symbol (e.g., AAPL)
   - Click "添加" to add
   - Or click "从数据库导入" to import from database

4. Click **"快速预测 (无训练)"** button (pink button)

5. View results in the status panel

### 2. Through Command Line

```bash
python scripts/test_prediction_only.py
```

### 3. Programmatically

```python
from bma_models.prediction_only_engine import create_prediction_engine

# Create engine (loads latest snapshot automatically)
engine = create_prediction_engine(snapshot_id=None)

# Run predictions
results = engine.predict(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    top_n=10
)

# View recommendations
for rec in results['recommendations']:
    print(f"{rec['rank']}. {rec['ticker']}: {rec['score']:.6f}")
```

## Model Snapshots

### Snapshot Location
All model snapshots are saved in: `cache/model_snapshots/YYYYMMDD/<snapshot_id>/`

### Snapshot Contents
Each snapshot contains:
- `elastic_net.pkl` - ElasticNet model
- `xgboost.json` - XGBoost model
- `catboost.cbm` - CatBoost model
- `lambdarank_lgb.txt` - LambdaRank LightGBM booster
- `ridge_model.pkl` - Ridge stacker model
- `manifest.json` - Snapshot metadata
- Weight files (`weights_*.json`) - Feature importances

### Using Specific Snapshot

```python
# Use specific snapshot by ID
engine = create_prediction_engine(snapshot_id='abc-123-def-456')
```

### List All Snapshots

```python
from bma_models.model_registry import list_snapshots

snapshots = list_snapshots()
for snapshot_id, created_at, tag in snapshots:
    print(f"{tag}: {snapshot_id} (created: {created_at})")
```

## Requirements

### Training Phase (One-time)
Before using prediction-only mode, you must **train models once**:

1. Run a full training cycle through the GUI or:
   ```python
   from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

   model = UltraEnhancedQuantitativeModel()
   results = model.run_complete_analysis(
       tickers=['AAPL', 'MSFT', ...],
       start_date='2023-01-01',
       end_date='2024-12-31'
   )
   ```

2. Models are automatically saved to snapshot after training

### Prediction Phase (Repeated Use)
Once models are trained and saved:
- No retraining needed
- Just load snapshot and predict
- Fast (< 1 minute for 10 stocks)

## API Reference

### PredictionOnlyEngine

```python
class PredictionOnlyEngine:
    def __init__(self, snapshot_id: Optional[str] = None):
        """Load models from snapshot"""

    def predict(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Generate predictions

        Args:
            tickers: List of stock symbols
            start_date: Start date (YYYY-MM-DD), default: 365 days ago
            end_date: End date (YYYY-MM-DD), default: today
            top_n: Number of top recommendations

        Returns:
            {
                'success': True/False,
                'snapshot_id': str,
                'predictions': pd.Series,
                'recommendations': List[Dict],
                'tickers': List[str],
                'n_stocks': int,
                'date_range': str
            }
        """
```

### Model Loading

```python
from bma_models.model_registry import (
    load_models_from_snapshot,
    load_manifest,
    list_snapshots
)

# Load all models from snapshot
loaded = load_models_from_snapshot(snapshot_id)
# Returns: {'models': {...}, 'ridge_stacker': ..., 'lambda_rank_stacker': ...}

# Load manifest only
manifest = load_manifest(snapshot_id)

# List all available snapshots
snapshots = list_snapshots()
```

## Performance

- **Training**: 20-60 minutes (depends on # stocks and data size)
- **Prediction-Only**: < 1 minute for 10 stocks
- **Speedup**: **20-60x faster** than retraining

## Workflow Comparison

### With Training (Old Way)
```
Input Stocks → Download Data → Calculate Factors → Train Models → Predict
[20-60 minutes for 10 stocks]
```

### Without Training (New Way)
```
Input Stocks → Load Snapshot → Download Data → Calculate Factors → Predict
[< 1 minute for 10 stocks]
```

## Troubleshooting

### Error: "未找到可用的模型快照"
**Solution**: You need to train models at least once first. Run a full training cycle.

### Error: "无法获取特征数据"
**Solution**: Check your Polygon API connection and stock symbols.

### Error: Model loading failed
**Solution**: Ensure the snapshot directory exists and contains all required files.

### Predictions look wrong
**Solution**:
1. Check if you're using the correct snapshot
2. Verify input stocks are valid US tickers
3. Ensure date range has sufficient data (recommend 365+ days)

## Best Practices

1. **Train Periodically**: Retrain models monthly with fresh data
2. **Use Latest Snapshot**: Unless testing, always use the latest snapshot
3. **Sufficient Data**: Provide at least 365 days of historical data for accurate predictions
4. **Valid Tickers**: Use valid US stock tickers from major exchanges
5. **Backup Snapshots**: Keep important snapshots backed up

## Files

### Core Files
- `bma_models/prediction_only_engine.py` - Prediction engine
- `bma_models/model_registry.py` - Snapshot save/load utilities
- `autotrader/app.py` - GUI integration (line ~3419-3495)
- `scripts/test_prediction_only.py` - CLI test script

### Model Files
- `bma_models/simple_25_factor_engine.py` - Factor calculation
- `bma_models/lambda_rank_stacker.py` - LambdaRank model
- `bma_models/ridge_stacker.py` - Ridge stacker
- `bma_models/lambda_percentile_transformer.py` - Percentile transformer

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log output in terminal
3. Check `data/model_registry.db` for snapshot records
4. Verify snapshot files exist in `cache/model_snapshots/`
