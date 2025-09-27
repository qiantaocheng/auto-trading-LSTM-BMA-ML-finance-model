import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from .kronos_model import KronosModelWrapper, KronosConfig
from .utils import (
    prepare_kline_data,
    format_prediction_results,
    calculate_prediction_metrics
)

logger = logging.getLogger(__name__)

class KronosService:
    """Service layer for Kronos predictions without UI dependencies"""

    def __init__(self):
        self.model_wrapper = None
        self.last_predictions = None
        self.last_symbol = None

    def initialize(self, model_size: str = "large") -> bool:
        """Initialize the Kronos model wrapper"""
        try:
            config = KronosConfig(model_size=model_size)
            self.model_wrapper = KronosModelWrapper(config)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Kronos: {str(e)}")
            return False

    def predict_stock(self,
                     symbol: str,
                     period: str = "3mo",
                     interval: str = "1d",
                     pred_len: int = 30,
                     model_size: str = "large",
                     temperature: float = 0.7) -> Dict[str, Any]:
        """Generate predictions for a stock"""
        try:
            # Initialize or reconfigure model if needed
            if self.model_wrapper is None or self.model_wrapper.config.model_size != model_size:
                if not self.initialize(model_size):
                    return {"status": "error", "error": "Failed to initialize model"}

            # Update temperature
            self.model_wrapper.config.temperature = temperature

            # Fetch historical data (Polygon-only, enforce US equities)
            logger.info(f"Fetching data for {symbol} via Polygon (US equities only, interval={interval})...")
            df = prepare_kline_data(symbol, period, interval)
            if df is None or df.empty:
                return {"status": "error", "error": f"Failed to fetch Polygon data for {symbol} (US equities only)"}

            # Load model if not loaded
            if not self.model_wrapper.is_loaded:
                logger.info("Loading Kronos model...")
                if not self.model_wrapper.load_model():
                    logger.warning("Kronos model not available; using fallback predictor")
            # Convert data to array
            data_array = df.values

            # Generate predictions
            logger.info(f"Generating {pred_len} predictions...")
            result = self.model_wrapper.predict(
                data=data_array,
                pred_len=pred_len
            )

            if result['status'] != 'success':
                return {"status": "error", "error": result.get('error', 'Prediction failed')}

            # Format predictions
            predictions = result['predictions']
            pred_df = format_prediction_results(
                predictions,
                base_timestamp=df.index[-1],
                interval=interval
            )

            # Store last results
            self.last_predictions = pred_df
            self.last_symbol = symbol

            return {
                "status": "success",
                "symbol": symbol,
                "historical_data": df,
                "predictions": pred_df,
                "config": result['config']
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def get_statistics(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for predictions"""
        try:
            stats = {
                "price_stats": {
                    "mean_close": predictions['close'].mean(),
                    "std_dev": predictions['close'].std(),
                    "min_close": predictions['close'].min(),
                    "max_close": predictions['close'].max(),
                    "price_range": predictions['close'].max() - predictions['close'].min()
                },
                "volume_stats": {
                    "mean_volume": predictions['volume'].mean(),
                    "max_volume": predictions['volume'].max(),
                    "min_volume": predictions['volume'].min(),
                    "volume_volatility": (predictions['volume'].std() / predictions['volume'].mean() * 100)
                },
                "change_stats": {
                    "total_change": (predictions['close'].iloc[-1] - predictions['close'].iloc[0]) / predictions['close'].iloc[0] * 100,
                    "first_close": predictions['close'].iloc[0],
                    "last_close": predictions['close'].iloc[-1]
                }
            }
            return stats
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return {}

    def export_predictions(self, predictions: pd.DataFrame, symbol: str, filepath: Optional[str] = None) -> str:
        """Export predictions to CSV file"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"kronos_predictions_{symbol}_{timestamp}.csv"

            predictions.to_csv(filepath)
            logger.info(f"Predictions exported to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            raise