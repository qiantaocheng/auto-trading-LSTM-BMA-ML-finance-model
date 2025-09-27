import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def prepare_kline_data(symbol: str,
                       period: str = "1mo",
                       interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Prepare K-line data using Polygon API only.

    - Enforces US stock symbols and uppercase normalization
    - Disables any yfinance fallback to guarantee a single data source
    """
    try:
        from .polygon_data_adapter import polygon_adapter

        # Normalize and validate symbol
        symbol = (symbol or "").strip().upper()
        if not symbol:
            logger.error("Empty symbol provided")
            return None

        if not polygon_adapter.is_us_equity(symbol):
            logger.error(f"Symbol {symbol} is not a US stock or not supported")
            return None

        logger.info(f"Fetching {symbol} data via Polygon API (Polygon-only mode)...")
        df = polygon_adapter.get_stock_data(symbol, period, interval)

        if df is None or df.empty:
            logger.error(f"Polygon API returned no data for {symbol}")
            return None

        logger.info(f"Successfully retrieved {len(df)} records from Polygon API for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error in prepare_kline_data for {symbol}: {str(e)}")
        return None

# Note: yfinance fallback intentionally removed to enforce Polygon-only data source

def format_prediction_results(predictions: np.ndarray,
                             base_timestamp: Optional[datetime] = None,
                             interval: str = "1d") -> pd.DataFrame:

    if base_timestamp is None:
        base_timestamp = datetime.now()

    timestamps = []
    current_time = base_timestamp

    interval_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1),
        "1wk": timedelta(weeks=1)
    }

    delta = interval_map.get(interval, timedelta(days=1))

    for _ in range(len(predictions)):
        current_time += delta
        timestamps.append(current_time)

    result_df = pd.DataFrame(
        predictions,
        columns=['open', 'high', 'low', 'close', 'volume'],
        index=timestamps
    )

    result_df.index.name = 'timestamp'

    return result_df

def calculate_prediction_metrics(actual: pd.DataFrame,
                                predicted: pd.DataFrame) -> Dict[str, float]:

    if len(actual) != len(predicted):
        min_len = min(len(actual), len(predicted))
        actual = actual.iloc[:min_len]
        predicted = predicted.iloc[:min_len]

    metrics = {}

    for col in ['open', 'high', 'low', 'close']:
        if col in actual.columns and col in predicted.columns:
            actual_vals = actual[col].values
            pred_vals = predicted[col].values

            mse = np.mean((actual_vals - pred_vals) ** 2)
            mae = np.mean(np.abs(actual_vals - pred_vals))
            mape = np.mean(np.abs((actual_vals - pred_vals) / actual_vals)) * 100

            actual_returns = np.diff(actual_vals) / actual_vals[:-1]
            pred_returns = np.diff(pred_vals) / pred_vals[:-1]

            direction_accuracy = np.mean(
                np.sign(actual_returns) == np.sign(pred_returns)
            ) * 100

            metrics[f"{col}_mse"] = mse
            metrics[f"{col}_mae"] = mae
            metrics[f"{col}_mape"] = mape
            metrics[f"{col}_direction_accuracy"] = direction_accuracy

    close_actual = actual['close'].values
    close_pred = predicted['close'].values
    correlation = np.corrcoef(close_actual, close_pred)[0, 1]
    metrics['correlation'] = correlation

    return metrics

def visualize_predictions(actual: Optional[pd.DataFrame],
                         predicted: pd.DataFrame,
                         title: str = "Kronos K-Line Predictions") -> Dict[str, Any]:

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(title, 'Volume'),
        row_heights=[0.7, 0.3]
    )

    if actual is not None:
        fig.add_trace(go.Candlestick(
            x=actual.index,
            open=actual['open'],
            high=actual['high'],
            low=actual['low'],
            close=actual['close'],
            name='Actual',
            increasing_line_color='green',
            decreasing_line_color='red'
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=actual.index,
            y=actual['volume'],
            name='Actual Volume',
            marker_color='gray',
            opacity=0.5
        ), row=2, col=1)

    fig.add_trace(go.Candlestick(
        x=predicted.index,
        open=predicted['open'],
        high=predicted['high'],
        low=predicted['low'],
        close=predicted['close'],
        name='Predicted',
        increasing_line_color='lightgreen',
        decreasing_line_color='lightcoral',
        opacity=0.7
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=predicted.index,
        y=predicted['volume'],
        name='Predicted Volume',
        marker_color='lightblue',
        opacity=0.7
    ), row=2, col=1)

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified'
    )

    return {
        'figure': fig,
        'data': {
            'actual': actual.to_dict() if actual is not None else None,
            'predicted': predicted.to_dict()
        }
    }

def normalize_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    for col in ['open', 'high', 'low', 'close']:
        if col in normalized.columns:
            mean_val = normalized[col].mean()
            std_val = normalized[col].std()
            if std_val > 0:
                normalized[col] = (normalized[col] - mean_val) / std_val

    if 'volume' in normalized.columns:
        vol_mean = normalized['volume'].mean()
        vol_std = normalized['volume'].std()
        if vol_std > 0:
            normalized['volume'] = (normalized['volume'] - vol_mean) / vol_std

    return normalized

def denormalize_predictions(predictions: np.ndarray,
                           original_stats: Dict[str, Dict[str, float]]) -> np.ndarray:

    denormalized = predictions.copy()

    for i, col in enumerate(['open', 'high', 'low', 'close', 'volume']):
        if col in original_stats:
            mean_val = original_stats[col]['mean']
            std_val = original_stats[col]['std']
            denormalized[:, i] = (predictions[:, i] * std_val) + mean_val

    return denormalized