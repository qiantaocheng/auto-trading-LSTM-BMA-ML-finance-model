import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def prepare_kline_data(symbol: str,
                       period: str = "1mo",
                       interval: str = "1d",
                       end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """
    Prepare K-line data using yfinance.

    - Uses Yahoo Finance data (yfinance) as the single source of truth
    - Supports an optional end_date (to prevent look-ahead during backtests / training)

    Args:
        symbol: Stock symbol
        period: Time period for historical data
        interval: Data interval
        end_date: End date for data (None = now for GUI predictions, training date for training)
    """
    try:
        try:
            import yfinance as yf
        except Exception as e_imp:
            logger.error(f"yfinance is required for Kronos data fetch but is not available: {e_imp}")
            return None

        # Normalize symbol
        symbol = (symbol or "").strip().upper()
        if not symbol:
            logger.error("Empty symbol provided")
            return None

        # Map common interval names to yfinance equivalents
        interval_map = {
            "1h": "60m",
        }
        yf_interval = interval_map.get(interval, interval)

        # Convert period to a start date so we can respect end_date (yf.download period+end is inconsistent across intervals)
        period_days_map = {
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
        }
        days = period_days_map.get(period, 90)

        if end_date is None:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # yfinance end is exclusive; add a small buffer to include end_date bars
        end_plus = end_date + timedelta(days=1)

        logger.info(f"Fetching data for {symbol} via yfinance (interval={yf_interval}, start={start_date.date()}, end={end_date.date()})...")
        raw = yf.download(
            tickers=symbol,
            start=start_date,
            end=end_plus,
            interval=yf_interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )

        if raw is None or raw.empty:
            logger.error(f"yfinance returned no data for {symbol}")
            return None

        # Standardize columns to lower-case OHLCV
        col_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        df = raw.rename(columns=col_map).copy()

        # Some yfinance versions return multi-index columns for multiple tickers; ensure flat
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
            df = df.rename(columns=col_map)

        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error(f"yfinance data missing required columns for {symbol}: {missing}")
            return None

        df = df[required_cols]

        # Normalize index to timezone-naive datetimes
        try:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        except Exception:
            df.index = pd.to_datetime(df.index)

        # Respect end_date cut (avoid any future bars beyond the requested end_date)
        try:
            df = df[df.index <= pd.to_datetime(end_date)]
        except Exception:
            pass

        # Basic cleaning
        df = df.sort_index()
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        df = df[df["volume"].fillna(0) >= 0]

        if df.empty:
            logger.error(f"yfinance cleaned data is empty for {symbol}")
            return None

        logger.info(f"Successfully retrieved {len(df)} records from yfinance for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error in prepare_kline_data for {symbol}: {str(e)}")
        return None

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