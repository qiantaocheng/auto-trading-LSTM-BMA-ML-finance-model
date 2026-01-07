#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POLYGON CLIENT SENTIMENT ANALYZER
==================================
Updated version using official polygon-api-client library
Integrates with BMA Ultra Enhanced for sentiment-based trading
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
import os
from tqdm import tqdm
import pickle
import json
from polygon import RESTClient

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PolygonClientSentimentAnalyzer:
    """
    Sentiment analyzer using official Polygon API client and FinBERT
    """

    def __init__(self,
                 polygon_api_key: Optional[str] = None,
                 cache_dir: str = "cache/sentiment_client",
                 max_workers: int = 2,
                 batch_size: int = 16):
        """
        Initialize analyzer with Polygon API client

        Args:
            polygon_api_key: Polygon.io API key
            cache_dir: Directory for caching
            max_workers: Number of parallel workers
            batch_size: Batch size for BERT inference
        """
        self.api_key = polygon_api_key or os.environ.get('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon API key required. Set POLYGON_API_KEY environment variable.")

        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.batch_size = batch_size

        # Initialize Polygon client
        self.client = RESTClient(api_key=self.api_key)

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize FinBERT
        logger.info("Loading FinBERT model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"FinBERT loaded on {self.device}")

        # Sentiment features
        self.sentiment_features = [
            'sentiment_score', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
            'sentiment_momentum_5d', 'sentiment_momentum_20d', 'sentiment_volatility',
            'news_volume', 'news_volume_ratio', 'sentiment_skew', 'sentiment_dispersion',
            'sentiment_trend', 'sentiment_reversal', 'sentiment_acceleration', 'sentiment_consistency'
        ]

    def fetch_news_for_ticker(self,
                             ticker: str,
                             start_date: datetime,
                             end_date: datetime) -> pd.DataFrame:
        """
        Fetch news using Polygon API client

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with news articles
        """
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

        # Fetch from API using client
        try:
            logger.info(f"Fetching news for {ticker} from {start_date.date()} to {end_date.date()}")

            # Use polygon client to get news
            news_iterator = self.client.list_ticker_news(
                ticker=ticker,
                published_utc_gte=start_date.strftime('%Y-%m-%d'),
                published_utc_lte=end_date.strftime('%Y-%m-%d'),
                limit=1000,
                sort='published_utc',
                order='asc'
            )

            # Collect all articles
            articles = []
            for article in news_iterator:
                try:
                    articles.append({
                        'id': getattr(article, 'id', ''),
                        'title': getattr(article, 'title', ''),
                        'description': getattr(article, 'description', ''),
                        'published_utc': getattr(article, 'published_utc', datetime.now()),
                        'author': getattr(article, 'author', ''),
                        'article_url': getattr(article, 'article_url', ''),
                        'amp_url': getattr(article, 'amp_url', ''),
                        'image_url': getattr(article, 'image_url', ''),
                        'keywords': getattr(article, 'keywords', [])
                    })
                except Exception as e:
                    logger.warning(f"Error processing article: {e}")
                    continue

                # Rate limiting
                time.sleep(0.01)

            # Convert to DataFrame
            if articles:
                df = pd.DataFrame(articles)
                df['published_utc'] = pd.to_datetime(df['published_utc'])
                df['ticker'] = ticker

                # Cache the result
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump({'timestamp': datetime.now(), 'data': df}, f)
                except Exception as e:
                    logger.warning(f"Cache write error: {e}")

                logger.info(f"Fetched {len(df)} articles for {ticker}")
                return df
            else:
                logger.warning(f"No news found for {ticker}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_market_data(self,
                         ticker: str,
                         start_date: datetime,
                         end_date: datetime) -> pd.DataFrame:
        """
        Fetch market data using Polygon client

        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache
        cache_key = f"{ticker}_market_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        cache_path = os.path.join(self.cache_dir, f"market_{cache_key}.pkl")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if (datetime.now() - cached_data['timestamp']).days < 1:
                        return cached_data['data']
            except Exception:
                pass

        try:
            logger.info(f"Fetching market data for {ticker}")

            # Get daily bars
            bars = self.client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan='day',
                from_=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                adjusted=True,
                sort='asc',
                limit=5000
            )

            if bars and hasattr(bars, 'results') and bars.results:
                data = []
                for bar in bars.results:
                    data.append({
                        'date': datetime.fromtimestamp(bar.timestamp / 1000).date(),
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume,
                        'vwap': getattr(bar, 'vwap', bar.close)
                    })

                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df['ticker'] = ticker
                df = df.set_index(['date', 'ticker'])

                # Cache result
                with open(cache_path, 'wb') as f:
                    pickle.dump({'timestamp': datetime.now(), 'data': df}, f)

                logger.info(f"Fetched {len(df)} days of market data for {ticker}")
                return df

        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {e}")

        return pd.DataFrame()

    def analyze_sentiment_batch(self, texts: List[str]) -> np.ndarray:
        """Analyze sentiment using FinBERT"""
        if not texts:
            return np.array([])

        inputs = self.tokenizer(texts,
                               padding=True,
                               truncation=True,
                               max_length=512,
                               return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = predictions.cpu().numpy()

        # Convert to sentiment scores
        sentiment_scores = predictions[:, 0] - predictions[:, 1]

        results = np.column_stack([
            sentiment_scores,     # Overall sentiment
            predictions[:, 0],    # Positive probability
            predictions[:, 1],    # Negative probability
            predictions[:, 2]     # Neutral probability
        ])

        return results

    def calculate_sentiment_features(self,
                                   news_df: pd.DataFrame,
                                   target_date: datetime) -> Dict[str, float]:
        """Calculate sentiment features for a date"""
        features = {}

        if news_df.empty:
            return {feat: 0.0 for feat in self.sentiment_features}

        # Filter news by time windows
        end_date = target_date
        windows = {'1d': 1, '5d': 5, '20d': 20, '60d': 60}
        window_sentiments = {}

        for window_name, days in windows.items():
            start_date = end_date - timedelta(days=days)
            window_news = news_df[
                (news_df['published_utc'] >= start_date) &
                (news_df['published_utc'] <= end_date)
            ]

            if not window_news.empty:
                texts = (window_news['title'].fillna('') + ' ' +
                        window_news['description'].fillna('')).tolist()

                all_sentiments = []
                for i in range(0, len(texts), self.batch_size):
                    batch_texts = texts[i:i+self.batch_size]
                    batch_sentiments = self.analyze_sentiment_batch(batch_texts)
                    all_sentiments.append(batch_sentiments)

                if all_sentiments:
                    sentiments = np.vstack(all_sentiments)
                    window_sentiments[window_name] = sentiments

        # Calculate features
        current_sentiments = window_sentiments.get('1d', np.array([[0, 0.33, 0.33, 0.34]]))

        features['sentiment_score'] = np.mean(current_sentiments[:, 0])
        features['sentiment_positive'] = np.mean(current_sentiments[:, 1])
        features['sentiment_negative'] = np.mean(current_sentiments[:, 2])
        features['sentiment_neutral'] = np.mean(current_sentiments[:, 3])

        # Volume features
        features['news_volume'] = len(current_sentiments)
        avg_volume_20d = len(window_sentiments.get('20d', [])) / 20.0
        features['news_volume_ratio'] = (features['news_volume'] / max(avg_volume_20d, 1.0)) - 1.0

        # Momentum
        sent_5d = np.mean(window_sentiments.get('5d', [[0]])[:, 0])
        sent_20d = np.mean(window_sentiments.get('20d', [[0]])[:, 0])
        features['sentiment_momentum_5d'] = features['sentiment_score'] - sent_5d
        features['sentiment_momentum_20d'] = features['sentiment_score'] - sent_20d

        # Volatility and dispersion
        if len(current_sentiments) > 1:
            features['sentiment_volatility'] = np.std(current_sentiments[:, 0])
            features['sentiment_dispersion'] = np.mean(np.abs(current_sentiments[:, 0] - features['sentiment_score']))
            features['sentiment_skew'] = np.sign(np.sum(current_sentiments[:, 0] > 0) - np.sum(current_sentiments[:, 0] < 0))
        else:
            features['sentiment_volatility'] = 0.0
            features['sentiment_dispersion'] = 0.0
            features['sentiment_skew'] = 0.0

        # Trend
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

        # Reversal and acceleration
        features['sentiment_reversal'] = -features['sentiment_momentum_20d'] if abs(sent_20d) > 0.1 else 0.0
        features['sentiment_acceleration'] = features['sentiment_momentum_5d'] - features['sentiment_momentum_20d']

        # Consistency
        if len(window_sentiments.get('5d', [])) > 1 and len(window_sentiments.get('20d', [])) > 1:
            short_term = window_sentiments['5d'][:, 0]
            long_term = window_sentiments['20d'][:, 0]
            if len(short_term) > 1 and len(long_term) > 1:
                try:
                    features['sentiment_consistency'] = np.corrcoef(
                        short_term[:len(long_term)],
                        long_term[:len(short_term)]
                    )[0, 1]
                    if np.isnan(features['sentiment_consistency']):
                        features['sentiment_consistency'] = 0.0
                except:
                    features['sentiment_consistency'] = 0.0
            else:
                features['sentiment_consistency'] = 0.0
        else:
            features['sentiment_consistency'] = 0.0

        return features

    def process_stock_universe(self,
                              tickers: List[str],
                              start_date: datetime,
                              end_date: datetime) -> pd.DataFrame:
        """
        Process sentiment and market data for stock universe

        Args:
            tickers: List of stock tickers
            start_date: Start date
            end_date: End date

        Returns:
            Combined DataFrame with market and sentiment features
        """
        logger.info(f"Processing {len(tickers)} stocks from {start_date.date()} to {end_date.date()}")

        # Generate business days
        business_days = pd.date_range(start_date, end_date, freq='B')
        trading_dates = [d.to_pydatetime() for d in business_days[-60:]]  # Last 60 trading days

        all_data = []

        for ticker in tqdm(tickers, desc="Processing stocks"):
            try:
                # Fetch market data
                market_df = self.fetch_market_data(ticker, start_date, end_date)

                if market_df.empty:
                    logger.warning(f"No market data for {ticker}")
                    continue

                # Fetch news data with buffer
                buffer_start = start_date - timedelta(days=90)
                news_df = self.fetch_news_for_ticker(ticker, buffer_start, end_date)

                # Calculate sentiment for each trading date
                stock_features = []
                for date in trading_dates:
                    # Get market features for this date
                    try:
                        market_row = market_df.loc[(pd.Timestamp(date).normalize(), ticker)]
                        market_features = market_row.to_dict()
                    except (KeyError, IndexError):
                        # If no market data for this date, skip
                        continue

                    # Calculate sentiment features
                    sentiment_features = self.calculate_sentiment_features(news_df, date)

                    # Combine features
                    combined = {**market_features, **sentiment_features}
                    combined['date'] = date
                    combined['ticker'] = ticker

                    stock_features.append(combined)

                if stock_features:
                    stock_df = pd.DataFrame(stock_features)
                    all_data.append(stock_df)

                # Rate limiting between stocks
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue

        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.set_index(['date', 'ticker'])

            # Calculate additional technical indicators
            combined_df = self._calculate_technical_indicators(combined_df)

            # Calculate target (10-day forward returns)
            combined_df['target'] = combined_df.groupby(level='ticker')['close'].pct_change(10).shift(-10)

            # Apply cross-sectional standardization
            combined_df = self._cross_sectional_standardize(combined_df)

            logger.info(f"Final combined dataset shape: {combined_df.shape}")
            return combined_df

        return pd.DataFrame()

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Group by ticker to calculate indicators
        result_dfs = []

        for ticker in df.index.get_level_values('ticker').unique():
            ticker_df = df.xs(ticker, level='ticker').sort_index()

            # Returns
            ticker_df['returns_1d'] = ticker_df['close'].pct_change(1)
            ticker_df['returns_5d'] = ticker_df['close'].pct_change(5)
            ticker_df['returns_20d'] = ticker_df['close'].pct_change(20)

            # Volatility
            ticker_df['volatility_5d'] = ticker_df['returns_1d'].rolling(5).std()
            ticker_df['volatility_20d'] = ticker_df['returns_1d'].rolling(20).std()

            # Volume
            ticker_df['volume_ratio'] = ticker_df['volume'] / ticker_df['volume'].rolling(20).mean()

            # RSI
            delta = ticker_df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            ticker_df['rsi_14'] = 100 - (100 / (1 + rs))

            # Momentum
            ticker_df['momentum_5d'] = ticker_df['close'] / ticker_df['close'].shift(5) - 1
            ticker_df['momentum_20d'] = ticker_df['close'] / ticker_df['close'].shift(20) - 1

            # Add ticker back to index
            ticker_df['ticker'] = ticker
            ticker_df = ticker_df.set_index('ticker', append=True)

            result_dfs.append(ticker_df)

        return pd.concat(result_dfs).sort_index()

    def _cross_sectional_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cross-sectional standardization"""
        feature_columns = [col for col in df.columns if col not in ['target', 'close', 'open', 'high', 'low']]
        standardized_dfs = []

        for date, date_group in df.groupby(level='date'):
            date_df = date_group.copy()

            for col in feature_columns:
                if col in date_df.columns:
                    values = date_df[col].values
                    values = np.where(np.isfinite(values), values, np.nan)

                    if not np.all(np.isnan(values)):
                        median_val = np.nanmedian(values)
                        values = np.where(np.isnan(values), median_val, values)

                        # Winsorize
                        lower = np.percentile(values, 1)
                        upper = np.percentile(values, 99)
                        values = np.clip(values, lower, upper)

                        # Standardize
                        mean = np.mean(values)
                        std = np.std(values)
                        if std > 1e-8:
                            date_df[col] = (values - mean) / std
                        else:
                            date_df[col] = 0.0
                    else:
                        date_df[col] = 0.0

            standardized_dfs.append(date_df)

        return pd.concat(standardized_dfs) if standardized_dfs else df


def run_polygon_client_demo(api_key: str = None):
    """Run demo with Polygon API client"""
    if not api_key:
        api_key = os.environ.get('POLYGON_API_KEY')
        if not api_key:
            print("Please provide Polygon API key or set POLYGON_API_KEY environment variable")
            return

    # Initialize analyzer
    analyzer = PolygonClientSentimentAnalyzer(polygon_api_key=api_key)

    # Test with small universe
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months

    print("Starting sentiment analysis with Polygon client...")
    print(f"Tickers: {test_tickers}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")

    # Process data
    combined_df = analyzer.process_stock_universe(test_tickers, start_date, end_date)

    if not combined_df.empty:
        print(f"\nResults:")
        print(f"Final shape: {combined_df.shape}")
        print(f"Features: {len([c for c in combined_df.columns if c != 'target'])}")
        print(f"Sentiment features: {[c for c in combined_df.columns if 'sentiment' in c]}")

        # Save results
        os.makedirs('result', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'result/polygon_client_sentiment_{timestamp}.pkl'
        combined_df.to_pickle(output_path)
        print(f"\nResults saved to: {output_path}")

        # Show sample data
        print(f"\nSample data (last 5 rows):")
        print(combined_df.tail())

        return combined_df
    else:
        print("No data returned")
        return None


if __name__ == "__main__":
    run_polygon_client_demo()
