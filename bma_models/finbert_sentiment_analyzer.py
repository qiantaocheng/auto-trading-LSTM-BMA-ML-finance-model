#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINBERT SENTIMENT ANALYSIS MODULE FOR BMA ULTRA ENHANCED
=========================================================
Professional-grade sentiment analysis using ProsusAI/finbert model
Integrates with Polygon API for news data and aligns with BMA data structure
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests
import os
from tqdm import tqdm
import pickle
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial news with Polygon API integration
    Generates sentiment features aligned with BMA Ultra Enhanced data structure
    """

    def __init__(self,
                 polygon_api_key: Optional[str] = None,
                 cache_dir: str = "cache/sentiment",
                 max_workers: int = 4,
                 batch_size: int = 32):
        """
        Initialize FinBERT sentiment analyzer

        Args:
            polygon_api_key: Polygon.io API key
            cache_dir: Directory for caching sentiment results
            max_workers: Number of parallel workers for API calls
            batch_size: Batch size for BERT inference
        """
        self.polygon_api_key = polygon_api_key or os.environ.get('POLYGON_API_KEY')
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.batch_size = batch_size

        # Create cache directory if not exists
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize FinBERT model and tokenizer
        logger.info("Loading FinBERT model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"FinBERT loaded successfully on {self.device}")

        # Sentiment feature configuration aligned with BMA structure
        self.sentiment_features = [
            'sentiment_score',           # Raw sentiment score [-1, 1]
            'sentiment_positive',         # Positive probability
            'sentiment_negative',         # Negative probability
            'sentiment_neutral',          # Neutral probability
            'sentiment_momentum_5d',      # 5-day sentiment momentum
            'sentiment_momentum_20d',     # 20-day sentiment momentum
            'sentiment_volatility',       # Sentiment volatility
            'news_volume',               # Number of news articles
            'news_volume_ratio',         # Volume ratio vs 20d average
            'sentiment_skew',            # Sentiment distribution skew
            'sentiment_dispersion',      # Sentiment dispersion across articles
            'sentiment_trend',           # Linear trend of sentiment
            'sentiment_reversal',        # Mean reversion signal
            'sentiment_acceleration',    # Rate of sentiment change
            'sentiment_consistency'      # Consistency of sentiment signal
        ]

    def fetch_news_from_polygon(self,
                               ticker: str,
                               start_date: datetime,
                               end_date: datetime) -> pd.DataFrame:
        """
        Fetch news articles from Polygon API for a specific ticker

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for news retrieval
            end_date: End date for news retrieval

        Returns:
            DataFrame with news articles
        """
        if not self.polygon_api_key:
            raise ValueError("Polygon API key not provided")

        # Check cache first
        cache_key = f"{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        cache_path = os.path.join(self.cache_dir, f"news_{cache_key}.pkl")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if (datetime.now() - cached_data['timestamp']).days < 1:
                        logger.debug(f"Using cached news for {ticker}")
                        return cached_data['data']
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

        # Fetch from API
        url = "https://api.polygon.io/v2/reference/news"
        params = {
            'ticker': ticker,
            'published_utc.gte': start_date.strftime('%Y-%m-%d'),
            'published_utc.lte': end_date.strftime('%Y-%m-%d'),
            'sort': 'published_utc',
            'order': 'asc',
            'limit': 100,
            'apiKey': self.polygon_api_key
        }

        all_articles = []

        try:
            while True:
                response = requests.get(url, params=params)
                if response.status_code != 200:
                    logger.error(f"Polygon API error: {response.status_code}")
                    break

                data = response.json()
                articles = data.get('results', [])
                all_articles.extend(articles)

                # Check for next page
                next_url = data.get('next_url')
                if not next_url:
                    break

                # Add API key to next URL if not present
                if 'apiKey' not in next_url:
                    next_url += f"&apiKey={self.polygon_api_key}"
                url = next_url
                params = {}  # Clear params for next request

                time.sleep(0.12)  # Rate limiting

        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")

        # Convert to DataFrame
        if all_articles:
            df = pd.DataFrame(all_articles)
            df['published_utc'] = pd.to_datetime(df['published_utc'])
            df['ticker'] = ticker

            # Cache the result
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump({'timestamp': datetime.now(), 'data': df}, f)
            except Exception as e:
                logger.warning(f"Cache write error: {e}")

            return df
        else:
            return pd.DataFrame()

    def analyze_sentiment_batch(self, texts: List[str]) -> np.ndarray:
        """
        Analyze sentiment for a batch of texts using FinBERT

        Args:
            texts: List of text strings to analyze

        Returns:
            Array of sentiment scores and probabilities
        """
        if not texts:
            return np.array([])

        # Tokenize texts
        inputs = self.tokenizer(texts,
                               padding=True,
                               truncation=True,
                               max_length=512,
                               return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = predictions.cpu().numpy()

        # Convert to sentiment scores
        # FinBERT outputs: [positive, negative, neutral]
        sentiment_scores = predictions[:, 0] - predictions[:, 1]  # Positive - Negative

        results = np.column_stack([
            sentiment_scores,     # Overall sentiment score
            predictions[:, 0],    # Positive probability
            predictions[:, 1],    # Negative probability
            predictions[:, 2]     # Neutral probability
        ])

        return results

    def calculate_sentiment_features(self,
                                    news_df: pd.DataFrame,
                                    target_date: datetime) -> Dict[str, float]:
        """
        Calculate comprehensive sentiment features for a specific date

        Args:
            news_df: DataFrame with news articles
            target_date: Target date for feature calculation

        Returns:
            Dictionary of sentiment features
        """
        features = {}

        if news_df.empty:
            # Return neutral/zero features if no news
            for feat in self.sentiment_features:
                features[feat] = 0.0
            return features

        # Filter news for different time windows
        end_date = target_date
        windows = {
            '1d': 1,
            '5d': 5,
            '20d': 20,
            '60d': 60
        }

        window_sentiments = {}

        for window_name, days in windows.items():
            start_date = end_date - timedelta(days=days)
            window_news = news_df[
                (news_df['published_utc'] >= start_date) &
                (news_df['published_utc'] <= end_date)
            ]

            if not window_news.empty:
                # Combine title and description for sentiment analysis
                texts = (window_news['title'].fillna('') + ' ' +
                        window_news.get('description', '').fillna('')).tolist()

                # Analyze sentiment in batches
                all_sentiments = []
                for i in range(0, len(texts), self.batch_size):
                    batch_texts = texts[i:i+self.batch_size]
                    batch_sentiments = self.analyze_sentiment_batch(batch_texts)
                    all_sentiments.append(batch_sentiments)

                if all_sentiments:
                    sentiments = np.vstack(all_sentiments)
                    window_sentiments[window_name] = sentiments
                else:
                    window_sentiments[window_name] = np.array([[0, 0.33, 0.33, 0.34]])
            else:
                window_sentiments[window_name] = np.array([[0, 0.33, 0.33, 0.34]])

        # Calculate base sentiment features
        current_sentiments = window_sentiments.get('1d', np.array([[0, 0.33, 0.33, 0.34]]))

        features['sentiment_score'] = np.mean(current_sentiments[:, 0])
        features['sentiment_positive'] = np.mean(current_sentiments[:, 1])
        features['sentiment_negative'] = np.mean(current_sentiments[:, 2])
        features['sentiment_neutral'] = np.mean(current_sentiments[:, 3])

        # News volume features
        features['news_volume'] = len(window_sentiments.get('1d', []))
        avg_volume_20d = len(window_sentiments.get('20d', [])) / 20.0
        features['news_volume_ratio'] = (features['news_volume'] / max(avg_volume_20d, 1.0)) - 1.0

        # Sentiment momentum
        sent_5d = np.mean(window_sentiments.get('5d', [[0]])[:, 0])
        sent_20d = np.mean(window_sentiments.get('20d', [[0]])[:, 0])
        features['sentiment_momentum_5d'] = features['sentiment_score'] - sent_5d
        features['sentiment_momentum_20d'] = features['sentiment_score'] - sent_20d

        # Sentiment volatility and dispersion
        if len(current_sentiments) > 1:
            features['sentiment_volatility'] = np.std(current_sentiments[:, 0])
            features['sentiment_dispersion'] = np.mean(np.abs(current_sentiments[:, 0] - features['sentiment_score']))
            features['sentiment_skew'] = np.sign(np.sum(current_sentiments[:, 0] > 0) - np.sum(current_sentiments[:, 0] < 0))
        else:
            features['sentiment_volatility'] = 0.0
            features['sentiment_dispersion'] = 0.0
            features['sentiment_skew'] = 0.0

        # Sentiment trend (linear regression slope over 20d)
        if len(window_sentiments.get('20d', [])) > 5:
            daily_sentiments = window_sentiments['20d'][:, 0]
            x = np.arange(len(daily_sentiments))
            if len(x) > 1:
                slope = np.polyfit(x, daily_sentiments, 1)[0]
                features['sentiment_trend'] = slope
            else:
                features['sentiment_trend'] = 0.0
        else:
            features['sentiment_trend'] = 0.0

        # Sentiment reversal (mean reversion signal)
        features['sentiment_reversal'] = -features['sentiment_momentum_20d'] if abs(sent_20d) > 0.1 else 0.0

        # Sentiment acceleration
        features['sentiment_acceleration'] = features['sentiment_momentum_5d'] - features['sentiment_momentum_20d']

        # Sentiment consistency (correlation between short and long term)
        if len(window_sentiments.get('5d', [])) > 1 and len(window_sentiments.get('20d', [])) > 1:
            short_term = window_sentiments['5d'][:, 0]
            long_term = window_sentiments['20d'][:, 0]
            if len(short_term) > 1 and len(long_term) > 1:
                features['sentiment_consistency'] = np.corrcoef(short_term[:len(long_term)], long_term[:len(short_term)])[0, 1]
                if np.isnan(features['sentiment_consistency']):
                    features['sentiment_consistency'] = 0.0
            else:
                features['sentiment_consistency'] = 0.0
        else:
            features['sentiment_consistency'] = 0.0

        return features

    def process_stock_sentiment(self,
                               ticker: str,
                               start_date: datetime,
                               end_date: datetime,
                               trading_dates: List[datetime]) -> pd.DataFrame:
        """
        Process sentiment for a single stock over specified trading dates

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for analysis
            end_date: End date for analysis
            trading_dates: List of trading dates to calculate features for

        Returns:
            DataFrame with sentiment features aligned to trading dates
        """
        # Fetch news with buffer for historical calculations
        buffer_start = start_date - timedelta(days=90)
        news_df = self.fetch_news_from_polygon(ticker, buffer_start, end_date)

        if news_df.empty:
            logger.warning(f"No news found for {ticker}")
            # Return DataFrame with zero features
            zero_features = {feat: 0.0 for feat in self.sentiment_features}
            df = pd.DataFrame([zero_features] * len(trading_dates))
            df['date'] = trading_dates
            df['ticker'] = ticker
            return df.set_index(['date', 'ticker'])

        # Calculate features for each trading date
        features_list = []
        for date in tqdm(trading_dates, desc=f"Processing {ticker}", leave=False):
            features = self.calculate_sentiment_features(news_df, date)
            features['date'] = date
            features['ticker'] = ticker
            features_list.append(features)

        df = pd.DataFrame(features_list)
        return df.set_index(['date', 'ticker'])

    def process_universe_sentiment(self,
                                  tickers: List[str],
                                  start_date: datetime,
                                  end_date: datetime,
                                  trading_dates: Optional[List[datetime]] = None) -> pd.DataFrame:
        """
        Process sentiment for entire stock universe with parallel processing

        Args:
            tickers: List of stock tickers
            start_date: Start date for analysis
            end_date: End date for analysis
            trading_dates: Optional list of trading dates

        Returns:
            MultiIndex DataFrame (date, ticker) with sentiment features
        """
        # Generate trading dates if not provided
        if trading_dates is None:
            date_range = pd.date_range(start_date, end_date, freq='B')
            trading_dates = [d.to_pydatetime() for d in date_range]

        logger.info(f"Processing sentiment for {len(tickers)} stocks from {start_date} to {end_date}")

        # Process stocks in parallel
        all_results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_stock_sentiment, ticker, start_date, end_date, trading_dates): ticker
                for ticker in tickers
            }

            for future in tqdm(as_completed(futures), total=len(tickers), desc="Processing stocks"):
                ticker = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                    # Add zero features for failed ticker
                    zero_features = {feat: 0.0 for feat in self.sentiment_features}
                    df = pd.DataFrame([zero_features] * len(trading_dates))
                    df['date'] = trading_dates
                    df['ticker'] = ticker
                    all_results.append(df.set_index(['date', 'ticker']))

        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, axis=0)
            combined_df = combined_df.sort_index()

            # Apply cross-sectional standardization (aligned with BMA)
            logger.info("Applying cross-sectional standardization...")
            combined_df = self.cross_sectional_standardize(combined_df)

            return combined_df
        else:
            return pd.DataFrame()

    def cross_sectional_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cross-sectional standardization to sentiment features
        Aligns with BMA Ultra Enhanced standardization approach

        Args:
            df: MultiIndex DataFrame with sentiment features

        Returns:
            Standardized DataFrame
        """
        standardized_dfs = []

        # Group by date for cross-sectional standardization
        for date, date_group in df.groupby(level='date'):
            # Reset index for easier manipulation
            date_df = date_group.reset_index(level='ticker')

            # Standardize each feature
            for col in self.sentiment_features:
                if col in date_df.columns:
                    values = date_df[col].values

                    # Remove outliers using IQR
                    q1 = np.percentile(values, 25)
                    q3 = np.percentile(values, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    # Winsorize outliers
                    values = np.clip(values, lower_bound, upper_bound)

                    # Z-score standardization
                    mean = np.mean(values)
                    std = np.std(values)
                    if std > 1e-8:
                        date_df[col] = (values - mean) / std
                    else:
                        date_df[col] = 0.0

            # Restore MultiIndex
            date_df['date'] = date
            date_df = date_df.set_index(['date', 'ticker'])
            standardized_dfs.append(date_df)

        return pd.concat(standardized_dfs) if standardized_dfs else df

    def align_with_bma_data(self,
                           sentiment_df: pd.DataFrame,
                           bma_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align sentiment features with BMA Ultra Enhanced data structure

        Args:
            sentiment_df: DataFrame with sentiment features
            bma_df: BMA Ultra Enhanced DataFrame

        Returns:
            Merged DataFrame with sentiment features added
        """
        # Ensure both DataFrames have MultiIndex (date, ticker)
        if not isinstance(sentiment_df.index, pd.MultiIndex):
            raise ValueError("Sentiment DataFrame must have MultiIndex (date, ticker)")
        if not isinstance(bma_df.index, pd.MultiIndex):
            raise ValueError("BMA DataFrame must have MultiIndex (date, ticker)")

        # Merge on index
        merged_df = bma_df.join(sentiment_df, how='left')

        # Fill missing sentiment features with 0 (neutral sentiment)
        for col in self.sentiment_features:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(0)

        logger.info(f"Successfully merged sentiment features. Shape: {merged_df.shape}")
        return merged_df

    def save_sentiment_data(self, df: pd.DataFrame, filepath: str):
        """
        Save sentiment data with proper formatting for BMA pipeline

        Args:
            df: DataFrame with sentiment features
            filepath: Path to save the data
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save with compression
        df.to_pickle(filepath)
        logger.info(f"Sentiment data saved to {filepath}")

        # Also save as CSV for inspection
        csv_path = filepath.replace('.pkl', '.csv')
        df.to_csv(csv_path)
        logger.info(f"CSV backup saved to {csv_path}")


def create_sentiment_enhanced_pipeline(
    stocks_file: str = "filtered_stocks_20250817_002928.txt",
    polygon_api_key: Optional[str] = None,
    lookback_years: int = 3,
    output_dir: str = "result/sentiment_enhanced"
) -> pd.DataFrame:
    """
    Create complete sentiment-enhanced dataset for BMA Ultra Enhanced pipeline

    Args:
        stocks_file: Path to file containing stock tickers
        polygon_api_key: Polygon API key
        lookback_years: Number of years of historical data
        output_dir: Directory to save output files

    Returns:
        DataFrame with sentiment features ready for ML training
    """
    # Load stock tickers
    with open(stocks_file, 'r') as f:
        content = f.read()
        # Parse the comma-separated quoted tickers
        import re
        tickers = re.findall(r'"([^"]+)"', content)

    logger.info(f"Loaded {len(tickers)} tickers from {stocks_file}")

    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * lookback_years)

    # Initialize sentiment analyzer
    analyzer = FinBERTSentimentAnalyzer(polygon_api_key=polygon_api_key)

    # Process sentiment for all stocks
    sentiment_df = analyzer.process_universe_sentiment(
        tickers=tickers[:50],  # Start with first 50 for testing
        start_date=start_date,
        end_date=end_date
    )

    # Save sentiment data
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'sentiment_features_{timestamp}.pkl')
    analyzer.save_sentiment_data(sentiment_df, output_path)

    return sentiment_df


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Generate sentiment features for BMA Ultra Enhanced')
    parser.add_argument('--stocks-file', default='filtered_stocks_20250817_002928.txt',
                       help='File containing stock tickers')
    parser.add_argument('--api-key', default=None,
                       help='Polygon API key (or set POLYGON_API_KEY env variable)')
    parser.add_argument('--years', type=int, default=3,
                       help='Number of years of historical data')
    parser.add_argument('--output-dir', default='result/sentiment_enhanced',
                       help='Output directory for results')

    args = parser.parse_args()

    # Run sentiment analysis pipeline
    sentiment_df = create_sentiment_enhanced_pipeline(
        stocks_file=args.stocks_file,
        polygon_api_key=args.api_key,
        lookback_years=args.years,
        output_dir=args.output_dir
    )

    print(f"Sentiment analysis complete. Shape: {sentiment_df.shape}")
    print(f"Features generated: {list(sentiment_df.columns)}")