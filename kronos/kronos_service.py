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

    def initialize(self, model_size: str = "base") -> bool:
        """Initialize the Kronos model wrapper"""
        try:
            config = KronosConfig(
                model_size=model_size,
                allow_fallback=False,
                allow_model_downgrade=False,
                attempt_dependency_install_on_failure=True
            )
            self.model_wrapper = KronosModelWrapper(config)
            # Eagerly load the model so failures are surfaced now
            loaded = self.model_wrapper.load_model()
            if not loaded:
                raise RuntimeError("Kronos load_model() returned False")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Kronos: {str(e)}")
            return False

    def predict_stock(self,
                     symbol: str,
                     period: str = "3mo",
                     interval: str = "1d",
                     pred_len: int = 30,
                     model_size: str = "base",
                     temperature: float = 0.7,
                     end_date: Optional[datetime] = None,
                     historical_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate predictions for a stock

        Args:
            symbol: Stock symbol
            period: Historical data period
            interval: Data interval
            pred_len: Prediction length
            model_size: Kronos model size
            temperature: Sampling temperature
            end_date: End date for historical data (None=now for GUI, training date for training)
            historical_df: Optional pre-fetched OHLCV dataframe (index datetime, cols open/high/low/close/volume).
                          If provided, KronosService will NOT fetch from yfinance, enabling efficient backtests.
        """
        try:
            # Initialize or reconfigure model if needed (defensive reload when not loaded)
            needs_init = (
                self.model_wrapper is None or
                getattr(self.model_wrapper, 'config', None) is None or
                getattr(self.model_wrapper.config, 'model_size', None) != model_size or
                not getattr(self.model_wrapper, 'is_loaded', False)
            )
            if needs_init:
                if not self.initialize(model_size):
                    return {"status": "error", "error": "Failed to initialize model"}

            # Update temperature
            self.model_wrapper.config.temperature = temperature

            # Use provided history (fast path) or fetch via yfinance
            if historical_df is not None:
                df = historical_df.copy()
                # Standardize expected columns
                df.columns = [str(c).lower() for c in df.columns]
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    return {"status": "error", "error": f"historical_df missing columns: {missing}"}
                df = df[required_cols]
                try:
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                except Exception:
                    df.index = pd.to_datetime(df.index)
                df = df.sort_index()
            else:
                # Fetch historical data (yfinance)
                logger.info(f"Fetching data for {symbol} via yfinance (interval={interval})...")
                df = prepare_kline_data(symbol, period, interval, end_date=end_date)
            if df is None or df.empty:
                return {"status": "error", "error": f"Failed to fetch yfinance data for {symbol}"}

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