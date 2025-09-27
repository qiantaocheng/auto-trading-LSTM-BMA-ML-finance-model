"""
Generate graphs for Kronos predictions
"""
import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add kronos to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kronos.utils import (
    prepare_kline_data,
    format_prediction_results,
    calculate_prediction_metrics,
    visualize_predictions
)
from kronos.kronos_model import KronosModelWrapper, KronosConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_kronos_graphs(symbol='AAPL', period='3mo', interval='1d', prediction_length=30):
    """
    Generate comprehensive graphs for Kronos model predictions
    """
    results = {}

    try:
        # 1. Fetch historical data
        logger.info(f"Fetching historical data for {symbol}...")
        historical_data = prepare_kline_data(symbol, period=period, interval=interval)

        if historical_data is None or historical_data.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return None

        logger.info(f"Fetched {len(historical_data)} historical records")
        results['historical_data'] = historical_data

        # 2. Initialize Kronos model
        logger.info("Initializing Kronos model...")
        config = KronosConfig()
        config.seq_len = 60
        config.pred_len = prediction_length
        model = KronosModelWrapper(config)

        # Load the model
        if not model.load_model():
            logger.error("Failed to load Kronos model")
            return None

        # Prepare training data
        train_data = historical_data[['open', 'high', 'low', 'close', 'volume']].values

        # 3. Generate predictions using the model
        logger.info("Generating predictions...")

        # Use the last 60 data points for prediction
        if len(train_data) >= 60:
            input_sequence = train_data[-60:]
        else:
            # Pad with zeros if we have less than 60 data points
            padding = np.zeros((60 - len(train_data), 5))
            input_sequence = np.vstack([padding, train_data])

        # Call the predict method with the correct parameters
        prediction_result = model.predict(
            data=input_sequence,
            timestamps=historical_data.index.to_list(),
            pred_len=prediction_length
        )

        if prediction_result is None or 'predictions' not in prediction_result:
            logger.error("Failed to generate predictions")
            return None

        # Extract predictions from the result
        predictions_raw = prediction_result['predictions']

        # Ensure predictions have the right shape
        if len(predictions_raw.shape) == 1:
            # If it's a 1D array (just prices), expand to include all OHLCV
            close_predictions = predictions_raw
            predictions = np.zeros((len(close_predictions), 5))

            # Use close price as base for other prices with small variations
            for i in range(len(close_predictions)):
                predictions[i, 3] = close_predictions[i]  # close
                predictions[i, 0] = close_predictions[i] * 0.999  # open (slightly lower)
                predictions[i, 1] = close_predictions[i] * 1.002  # high (slightly higher)
                predictions[i, 2] = close_predictions[i] * 0.998  # low (slightly lower)
                predictions[i, 4] = historical_data['volume'].mean()  # Use average volume
        else:
            predictions = predictions_raw

        # Format predictions
        last_timestamp = historical_data.index[-1]
        predicted_df = format_prediction_results(
            predictions,
            base_timestamp=last_timestamp,
            interval=interval
        )
        results['predictions'] = predicted_df

        # 4. Create comprehensive visualizations
        logger.info("Creating visualizations...")

        # Main prediction vs historical chart
        fig_main = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'{symbol} - Historical vs Predicted Prices',
                'Volume',
                'Price Movement Indicators'
            ),
            row_heights=[0.5, 0.25, 0.25]
        )

        # Historical candlestick
        fig_main.add_trace(go.Candlestick(
            x=historical_data.index,
            open=historical_data['open'],
            high=historical_data['high'],
            low=historical_data['low'],
            close=historical_data['close'],
            name='Historical',
            increasing_line_color='green',
            decreasing_line_color='red'
        ), row=1, col=1)

        # Predicted candlestick
        fig_main.add_trace(go.Candlestick(
            x=predicted_df.index,
            open=predicted_df['open'],
            high=predicted_df['high'],
            low=predicted_df['low'],
            close=predicted_df['close'],
            name='Predicted',
            increasing_line_color='lightgreen',
            decreasing_line_color='pink',
            opacity=0.7
        ), row=1, col=1)

        # Volume bars
        fig_main.add_trace(go.Bar(
            x=historical_data.index,
            y=historical_data['volume'],
            name='Historical Volume',
            marker_color='gray',
            opacity=0.5
        ), row=2, col=1)

        fig_main.add_trace(go.Bar(
            x=predicted_df.index,
            y=predicted_df['volume'],
            name='Predicted Volume',
            marker_color='lightblue',
            opacity=0.7
        ), row=2, col=1)

        # Add moving averages
        historical_data['MA20'] = historical_data['close'].rolling(window=20).mean()
        historical_data['MA50'] = historical_data['close'].rolling(window=50).mean()

        fig_main.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['MA20'],
            name='MA20',
            line=dict(color='orange', width=1)
        ), row=1, col=1)

        fig_main.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['MA50'],
            name='MA50',
            line=dict(color='blue', width=1)
        ), row=1, col=1)

        # Calculate RSI
        delta = historical_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        historical_data['RSI'] = 100 - (100 / (1 + rs))

        fig_main.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['RSI'],
            name='RSI',
            line=dict(color='purple', width=2)
        ), row=3, col=1)

        # Add RSI reference lines
        fig_main.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig_main.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

        fig_main.update_xaxes(rangeslider_visible=False)
        fig_main.update_layout(
            height=800,
            title_text=f"Kronos Model Analysis - {symbol}",
            showlegend=True,
            hovermode='x unified'
        )

        # Save main figure into graph/ folder
        project_root = os.path.dirname(os.path.abspath(__file__))
        graph_dir = os.path.join(project_root, 'graph')
        os.makedirs(graph_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        interval_hint = interval
        main_path = os.path.join(graph_dir, f'kronos_{symbol}_{interval_hint}_{timestamp}_analysis.html')
        fig_main.write_html(main_path)
        logger.info(f"Saved main analysis to {main_path}")

        # 5. Create comparison chart (closing prices only)
        fig_compare = go.Figure()

        fig_compare.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['close'],
            name='Historical Close',
            line=dict(color='blue', width=2)
        ))

        fig_compare.add_trace(go.Scatter(
            x=predicted_df.index,
            y=predicted_df['close'],
            name='Predicted Close',
            line=dict(color='red', width=2, dash='dash')
        ))

        # Add confidence bands for predictions
        std_dev = predicted_df['close'].std() * 0.1  # 10% confidence band
        upper_band = predicted_df['close'] + std_dev
        lower_band = predicted_df['close'] - std_dev

        fig_compare.add_trace(go.Scatter(
            x=predicted_df.index,
            y=upper_band,
            fill=None,
            mode='lines',
            line_color='rgba(255,0,0,0)',
            showlegend=False
        ))

        fig_compare.add_trace(go.Scatter(
            x=predicted_df.index,
            y=lower_band,
            fill='tonexty',
            mode='lines',
            line_color='rgba(255,0,0,0)',
            name='Confidence Band',
            fillcolor='rgba(255,0,0,0.2)'
        ))

        fig_compare.update_layout(
            title=f'{symbol} - Historical vs Predicted Closing Prices',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            hovermode='x unified'
        )

        compare_path = os.path.join(graph_dir, f'kronos_{symbol}_{interval_hint}_{timestamp}_comparison.html')
        fig_compare.write_html(compare_path)
        logger.info(f"Saved comparison chart to {compare_path}")

        # 6. Create performance metrics chart
        # Calculate daily returns
        historical_returns = historical_data['close'].pct_change().dropna()
        predicted_returns = predicted_df['close'].pct_change().dropna()

        fig_metrics = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Daily Returns Distribution',
                'Cumulative Returns',
                'Volatility (5-day rolling)',
                'Price Momentum'
            )
        )

        # Returns distribution
        fig_metrics.add_trace(go.Histogram(
            x=historical_returns,
            name='Historical Returns',
            opacity=0.7,
            nbinsx=30
        ), row=1, col=1)

        # Cumulative returns
        cumulative_hist = (1 + historical_returns).cumprod()
        fig_metrics.add_trace(go.Scatter(
            x=cumulative_hist.index,
            y=cumulative_hist,
            name='Historical Cumulative',
            line=dict(color='blue')
        ), row=1, col=2)

        # Volatility
        volatility = historical_returns.rolling(window=5).std()
        fig_metrics.add_trace(go.Scatter(
            x=volatility.index,
            y=volatility,
            name='5-day Volatility',
            line=dict(color='orange')
        ), row=2, col=1)

        # Momentum
        momentum = historical_data['close'].diff(5)
        fig_metrics.add_trace(go.Scatter(
            x=momentum.index,
            y=momentum,
            name='5-day Momentum',
            line=dict(color='green')
        ), row=2, col=2)

        fig_metrics.update_layout(
            height=600,
            title_text=f"Performance Metrics - {symbol}",
            showlegend=True
        )

        metrics_path = os.path.join(graph_dir, f'kronos_{symbol}_{interval_hint}_{timestamp}_metrics.html')
        fig_metrics.write_html(metrics_path)
        logger.info(f"Saved metrics chart to {metrics_path}")

        # 7. Calculate and display prediction metrics (simplified for now)
        # Since we're predicting future values, we can't validate against actual future data
        # Instead, let's show the prediction statistics
        print("\n" + "="*50)
        print(f"KRONOS MODEL PREDICTIONS - {symbol}")
        print("="*50)
        print(f"Prediction Period: {prediction_length} {interval}")
        print(f"Starting from: {last_timestamp}")
        print(f"Ending at: {predicted_df.index[-1]}")
        print("\nPredicted Price Range:")
        print(f"  Close: ${predicted_df['close'].min():.2f} - ${predicted_df['close'].max():.2f}")
        print(f"  Average: ${predicted_df['close'].mean():.2f}")

        # Calculate trend
        price_change = predicted_df['close'].iloc[-1] - historical_data['close'].iloc[-1]
        price_change_pct = (price_change / historical_data['close'].iloc[-1]) * 100
        print(f"\nExpected Price Change: ${price_change:.2f} ({price_change_pct:+.2f}%)")

        # Instead of validation metrics, show prediction confidence based on volatility
        historical_volatility = historical_data['close'].pct_change().std()
        print(f"\nHistorical Volatility: {historical_volatility:.4f}")

        if False:  # Disable validation for now since we're predicting future
            metrics = calculate_prediction_metrics(actual_validation, validation_predicted_df)


        print(f"\nGraphs generated successfully!")
        print(f"1. Main analysis: {main_path}")
        print(f"2. Price comparison: {compare_path}")
        print(f"3. Performance metrics: {metrics_path}")

        return results

    except Exception as e:
        logger.error(f"Error generating graphs: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Kronos model prediction graphs')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol (default: AAPL)')
    parser.add_argument('--period', type=str, default='3mo', help='Historical data period (default: 3mo)')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval (default: 1d)')
    parser.add_argument('--prediction-length', type=int, default=30, help='Number of periods to predict (default: 30)')

    args = parser.parse_args()

    print(f"\nGenerating Kronos graphs for {args.symbol}...")
    print(f"Period: {args.period}, Interval: {args.interval}, Prediction Length: {args.prediction_length}")

    generate_kronos_graphs(
        symbol=args.symbol,
        period=args.period,
        interval=args.interval,
        prediction_length=args.prediction_length
    )