"""Simple test of sentiment feature"""

import os
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Set API key
os.environ['POLYGON_API_KEY'] = 'FExbaO1xdmrV6f6p3zHCxk8IArjeowQ1'

from bma_models.simple_25_factor_engine import Simple25FactorEngine

print("Testing sentiment with real API key...")

engine = Simple25FactorEngine(enable_sentiment=True)

# Single stock, recent data
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

print(f"Fetching AAPL data from {start_date} to {end_date}")

market_data = engine.fetch_market_data(
    symbols=['AAPL'],
    start_date=start_date,
    end_date=end_date
)

print(f"Market data shape: {market_data.shape}")

# Compute factors
factors = engine.compute_all_17_factors(market_data)

print(f"\nFactors shape: {factors.shape}")
print(f"Columns: {list(factors.columns)}")

if 'sentiment_score' in factors.columns:
    sentiment = factors['sentiment_score']
    print(f"\nSentiment column exists!")
    print(f"Non-NaN values: {sentiment.notna().sum()}/{len(sentiment)}")
    print(f"Non-zero values: {(sentiment != 0).sum()}/{len(sentiment)}")
    print(f"Unique values: {sentiment.nunique()}")

    if sentiment.notna().any():
        print(f"Mean: {sentiment.mean():.4f}")
        print(f"Std: {sentiment.std():.4f}")
        print(f"Min: {sentiment.min():.4f}")
        print(f"Max: {sentiment.max():.4f}")
else:
    print("\nNo sentiment_score column found!")