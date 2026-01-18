@echo off
REM Example usage of exponential weighted prediction script

REM Example 1: Basic usage with fixed weights
python scripts/exponential_weighted_prediction.py --tickers AAPL,MSFT,GOOGL,AMZN,TSLA --output results/ewm_predictions_basic.xlsx

REM Example 2: Using half-life (3 days)
python scripts/exponential_weighted_prediction.py --tickers AAPL,MSFT,GOOGL,AMZN,TSLA --use-half-life --half-life-days 3.0 --output results/ewm_predictions_half_life.xlsx

REM Example 3: Custom weights
python scripts/exponential_weighted_prediction.py --tickers AAPL,MSFT,GOOGL,AMZN,TSLA --weights 0.6,0.25,0.15 --output results/ewm_predictions_custom.xlsx

REM Example 4: With specific date and snapshot
python scripts/exponential_weighted_prediction.py --tickers AAPL,MSFT,GOOGL,AMZN,TSLA --as-of-date 2026-01-15 --snapshot-id latest --output results/ewm_predictions_20260115.xlsx

pause
