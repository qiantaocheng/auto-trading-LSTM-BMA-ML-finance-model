#!/usr/bin/env python3
"""
Enhanced Sentiment Factors - 增强情绪因子模块
统一使用Polygon API获取：
- Polygon新闻情绪分析
- SP500市场情绪指数 (通过SPY ETF)
- Fear & Greed指数情绪因子
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import requests
import json
from dataclasses import dataclass
from textblob import TextBlob
# import yfinance as yf  # 替换为Polygon API

logger = logging.getLogger(__name__)

@dataclass
class SentimentConfig:
    """情绪因子配置"""
    # Polygon新闻配置
    polygon_api_key: str = ""
    news_lookback_days: int = 5
    sentiment_decay_days: float = 2.0
    
    # SP500指数配置
    sp500_lookback_days: int = 30
    market_regime_windows: List[int] = None
    
    # Fear & Greed指数配置
    fear_greed_cache_minutes: int = 60
    
    def __post_init__(self):
        if self.market_regime_windows is None:
            self.market_regime_windows = [5, 20, 60]

class EnhancedSentimentFactors:
    """
    增强情绪因子计算器
    整合多种情绪数据源构建情绪类量化因子
    """
    
    def __init__(self, config: SentimentConfig = None):
        """初始化情绪因子计算器"""
        self.config = config or SentimentConfig()
        
        # 缓存设置
        self.cache = {}
        self.cache_timestamps = {}
        
        # 移除硬编码权重 - 让机器学习模型自动学习最优权重
        # 每个情绪因子将作为独立特征参与训练
        
        # 统计信息
        self.stats = {
            'news_processed': 0,
            'sentiment_calculations': 0,
            'cache_hits': 0,
            'api_calls': 0
        }
        
        logger.info("Enhanced Sentiment Factors initialized for independent ML feature training")
    
    def compute_all_sentiment_factors(self, tickers: List[str], 
                                    end_date: datetime = None) -> Dict[str, pd.DataFrame]:
        """
        计算所有情绪因子
        
        Args:
            tickers: 股票代码列表
            end_date: 结束日期，默认为今天
            
        Returns:
            包含所有情绪因子的字典
        """
        if end_date is None:
            end_date = datetime.now()
            
        logger.info(f"Computing sentiment factors for {len(tickers)} tickers")
        
        results = {}
        
        try:
            # 1. Polygon新闻情绪因子
            logger.info("Computing Polygon news sentiment factors...")
            news_factors = self._compute_news_sentiment_factors(tickers, end_date)
            if news_factors is not None and not news_factors.empty:
                results['news_sentiment'] = news_factors
                logger.info(f"✅ News sentiment factors: {news_factors.shape}")
            
            # 2. SP500大盘情绪因子
            logger.info("Computing SP500 market sentiment factors...")
            market_factors = self._compute_sp500_sentiment_factors(end_date)
            if market_factors is not None and not market_factors.empty:
                results['market_sentiment'] = market_factors
                logger.info(f"✅ SP500 market factors: {market_factors.shape}")
            
            # 3. Fear & Greed指数因子
            logger.info("Computing Fear & Greed sentiment factors...")
            fear_greed_factors = self._compute_fear_greed_factors(end_date)
            if fear_greed_factors is not None and not fear_greed_factors.empty:
                results['fear_greed'] = fear_greed_factors
                logger.info(f"✅ Fear & Greed factors: {fear_greed_factors.shape}")
            
            # 4. 返回原始因子而非综合因子 - 让ML模型自动学习权重
            # 移除综合情绪因子计算，保持所有原始因子独立
            
            logger.info(f"Successfully computed {len(results)} sentiment factor groups")
            return results
            
        except Exception as e:
            logger.error(f"Error computing sentiment factors: {e}")
            return {}
    
    def _compute_news_sentiment_factors(self, tickers: List[str], 
                                       end_date: datetime) -> Optional[pd.DataFrame]:
        """
        计算Polygon新闻情绪因子
        
        使用Polygon API获取新闻，通过NLP分析情绪并量化为因子
        """
        if not self.config.polygon_api_key:
            logger.warning("Polygon API key not configured, skipping news sentiment")
            return None
        
        try:
            start_date = end_date - timedelta(days=self.config.news_lookback_days)
            sentiment_data = []
            
            for ticker in tickers:
                logger.debug(f"Fetching news sentiment for {ticker}")
                
                # 获取新闻数据
                news_items = self._fetch_polygon_news(ticker, start_date, end_date)
                
                if news_items:
                    # 计算情绪分数
                    sentiment_scores = self._analyze_news_sentiment(news_items)
                    
                    # 构建时间序列情绪因子
                    sentiment_ts = self._build_sentiment_timeseries(
                        sentiment_scores, start_date, end_date, ticker
                    )
                    
                    sentiment_data.append(sentiment_ts)
                    self.stats['news_processed'] += len(news_items)
            
            if sentiment_data:
                # 合并所有股票的情绪数据
                all_sentiment = pd.concat(sentiment_data, ignore_index=True)
                
                # 计算衍生情绪因子
                enhanced_sentiment = self._enhance_news_sentiment_factors(all_sentiment)
                
                return enhanced_sentiment
            
            return None
            
        except Exception as e:
            logger.error(f"Error computing news sentiment factors: {e}")
            return None
    
    def _compute_sp500_sentiment_factors(self, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        计算SP500大盘情绪因子
        
        基于SP500指数的技术指标和市场行为构建大盘情绪因子
        """
        try:
            # 获取SP500数据 - 使用Polygon API统一数据源
            start_date = end_date - timedelta(days=self.config.sp500_lookback_days * 2)  # 更多历史数据用于计算
            
            sp500_data = self._fetch_polygon_sp500_data(start_date, end_date)
            
            if sp500_data is None or sp500_data.empty:
                logger.warning("Failed to fetch SP500 data from Polygon API")
                return None
            
            logger.debug(f"Fetched SP500 data: {sp500_data.shape}")
            
            # 计算多种市场情绪指标
            factors_data = []
            
            for window in self.config.market_regime_windows:
                if len(sp500_data) >= window:
                    # 动量情绪
                    momentum = sp500_data['Close'].pct_change(window)
                    
                    # 波动率情绪（VIX代理）
                    volatility = sp500_data['Close'].pct_change().rolling(window).std() * np.sqrt(252)
                    
                    # 成交量情绪
                    volume_ma = sp500_data['Volume'].rolling(window).mean()
                    volume_ratio = sp500_data['Volume'] / volume_ma
                    
                    # RSI类似指标
                    price_change = sp500_data['Close'].diff()
                    gain = price_change.where(price_change > 0, 0).rolling(window).mean()
                    loss = (-price_change.where(price_change < 0, 0)).rolling(window).mean()
                    rs = gain / (loss + 1e-8)
                    rsi = 100 - (100 / (1 + rs))
                    
                    # 构建因子DataFrame
                    factor_df = pd.DataFrame({
                        'date': sp500_data.index,
                        f'sp500_momentum_{window}d': momentum,
                        f'sp500_volatility_{window}d': volatility,
                        f'sp500_volume_sentiment_{window}d': volume_ratio,
                        f'sp500_rsi_{window}d': rsi,
                        f'sp500_fear_level_{window}d': volatility * (-momentum),  # 恐慌水平
                    })
                    
                    factors_data.append(factor_df)
            
            if factors_data:
                # 合并所有窗口的因子
                combined_factors = factors_data[0]
                for df in factors_data[1:]:
                    combined_factors = combined_factors.merge(df, on='date', how='outer')
                
                # 添加复合情绪指标
                combined_factors = self._add_composite_market_sentiment(combined_factors)
                
                # 过滤最近的数据
                recent_data = combined_factors[
                    combined_factors['date'] >= (end_date - timedelta(days=self.config.sp500_lookback_days))
                ]
                
                return recent_data.dropna()
            
            return None
            
        except Exception as e:
            logger.error(f"Error computing SP500 sentiment factors: {e}")
            return None
    
    def _compute_fear_greed_factors(self, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        计算Fear & Greed指数因子
        
        使用fear-and-greed包获取CNN恐惧贪婪指数
        """
        try:
            # 检查缓存
            cache_key = 'fear_greed_index'
            if self._is_cache_valid(cache_key, self.config.fear_greed_cache_minutes):
                logger.debug("Using cached Fear & Greed data")
                self.stats['cache_hits'] += 1
                return self.cache[cache_key]
            
            # 尝试获取Fear & Greed指数
            try:
                import fear_and_greed
                fg_data = fear_and_greed.get()
                self.stats['api_calls'] += 1
                
                logger.debug(f"Fear & Greed Index: {fg_data.value} ({fg_data.description})")
                
                # 构建因子DataFrame
                factor_df = pd.DataFrame({
                    'date': [end_date],
                    'fear_greed_value': [fg_data.value],
                    'fear_greed_category': [fg_data.description],
                    'fear_greed_normalized': [(fg_data.value - 50) / 50],  # 归一化到[-1, 1]
                    'is_extreme_fear': [fg_data.value < 25],
                    'is_extreme_greed': [fg_data.value > 75],
                    'market_fear_level': [max(0, 50 - fg_data.value) / 50],  # 恐惧水平[0, 1]
                    'market_greed_level': [max(0, fg_data.value - 50) / 50],  # 贪婪水平[0, 1]
                })
                
                # 缓存结果
                self.cache[cache_key] = factor_df
                self.cache_timestamps[cache_key] = datetime.now()
                
                return factor_df
                
            except ImportError:
                logger.warning("fear-and-greed package not installed, installing...")
                # 这里可以添加自动安装或提示用户安装
                return None
                
        except Exception as e:
            logger.error(f"Error computing Fear & Greed factors: {e}")
            return None
    
    def _fetch_polygon_news(self, ticker: str, start_date: datetime, 
                           end_date: datetime) -> List[Dict]:
        """获取Polygon新闻数据"""
        try:
            # 这里是Polygon新闻API调用的示例实现
            # 实际使用时需要有效的API密钥
            
            url = f"https://api.polygon.io/v2/reference/news"
            params = {
                'ticker': ticker,
                'published_utc.gte': start_date.strftime('%Y-%m-%d'),
                'published_utc.lte': end_date.strftime('%Y-%m-%d'),
                'limit': 50,
                'apikey': self.config.polygon_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            else:
                logger.warning(f"Polygon news API returned status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching Polygon news for {ticker}: {e}")
            return []
    
    def _fetch_polygon_sp500_data(self, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """使用Polygon API获取SP500指数数据"""
        if not self.config.polygon_api_key:
            logger.warning("Polygon API key not configured, cannot fetch SP500 data")
            return None
            
        try:
            # Polygon股票聚合API
            # SPX是SP500的标准symbol，也可以用SPY（SPDR S&P 500 ETF）
            symbol = "SPY"  # 使用SPY ETF作为SP500代理
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'apikey': self.config.polygon_api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and data['results']:
                    # 转换为DataFrame
                    results = data['results']
                    df = pd.DataFrame(results)
                    
                    # 重命名列以匹配标准格式
                    column_mapping = {
                        't': 'timestamp',
                        'o': 'Open',
                        'h': 'High', 
                        'l': 'Low',
                        'c': 'Close',
                        'v': 'Volume'
                    }
                    
                    df = df.rename(columns=column_mapping)
                    
                    # 转换时间戳
                    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('date')
                    
                    # 选择需要的列
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    
                    logger.debug(f"Successfully fetched {len(df)} days of SP500 data from Polygon")
                    self.stats['api_calls'] += 1
                    return df
                else:
                    logger.warning("No SP500 data returned from Polygon API")
                    return None
            else:
                logger.warning(f"Polygon SP500 API returned status {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching SP500 data from Polygon: {e}")
            return None
    
    def _analyze_news_sentiment(self, news_items: List[Dict]) -> List[Dict]:
        """分析新闻情绪"""
        sentiments = []
        
        for item in news_items:
            try:
                # 获取标题和描述
                title = item.get('title', '')
                description = item.get('description', '')
                
                # 合并文本
                text = f"{title}. {description}"
                
                # 使用TextBlob进行情绪分析
                blob = TextBlob(text)
                
                # 计算情绪分数
                polarity = blob.sentiment.polarity  # [-1, 1]
                subjectivity = blob.sentiment.subjectivity  # [0, 1]
                
                # 构建情绪记录
                sentiment_record = {
                    'published_utc': item.get('published_utc'),
                    'title': title,
                    'sentiment_polarity': polarity,
                    'sentiment_subjectivity': subjectivity,
                    'sentiment_score': polarity * (1 - subjectivity * 0.3),  # 调整主观性影响
                    'text_length': len(text),
                    'relevance_score': self._calculate_relevance_score(item)
                }
                
                sentiments.append(sentiment_record)
                
            except Exception as e:
                logger.debug(f"Error analyzing sentiment for news item: {e}")
                continue
        
        return sentiments
    
    def _calculate_relevance_score(self, news_item: Dict) -> float:
        """计算新闻相关性分数"""
        # 基于新闻来源、关键词等计算相关性
        relevance = 0.5  # 基础分数
        
        # 根据新闻来源调整
        publisher = news_item.get('publisher', {}).get('name', '').lower()
        if any(source in publisher for source in ['reuters', 'bloomberg', 'cnbc', 'marketwatch']):
            relevance += 0.3
        
        # 根据关键词调整
        title = news_item.get('title', '').lower()
        if any(keyword in title for keyword in ['earnings', 'revenue', 'profit', 'loss']):
            relevance += 0.2
        
        return min(1.0, relevance)
    
    def _build_sentiment_timeseries(self, sentiment_scores: List[Dict], 
                                   start_date: datetime, end_date: datetime, 
                                   ticker: str) -> pd.DataFrame:
        """构建情绪时间序列"""
        if not sentiment_scores:
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(sentiment_scores)
        df['published_utc'] = pd.to_datetime(df['published_utc'])
        df['date'] = df['published_utc'].dt.date
        
        # 按日期聚合情绪分数
        daily_sentiment = df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'sentiment_polarity': ['mean', 'min', 'max'],
            'relevance_score': 'mean'
        }).reset_index()
        
        # 扁平化列名
        daily_sentiment.columns = [
            'date', 'sentiment_mean', 'sentiment_std', 'news_count',
            'polarity_mean', 'polarity_min', 'polarity_max', 'relevance_mean'
        ]
        
        # 添加ticker信息
        daily_sentiment['ticker'] = ticker
        
        # 应用时间衰减
        daily_sentiment = self._apply_sentiment_decay(daily_sentiment, end_date)
        
        return daily_sentiment
    
    def _apply_sentiment_decay(self, sentiment_df: pd.DataFrame, 
                              end_date: datetime) -> pd.DataFrame:
        """应用情绪时间衰减"""
        sentiment_df = sentiment_df.copy()
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # 计算距离当前时间的天数
        sentiment_df['days_ago'] = (end_date - sentiment_df['date']).dt.days
        
        # 计算衰减权重
        decay_factor = self.config.sentiment_decay_days
        sentiment_df['decay_weight'] = np.exp(-sentiment_df['days_ago'] / decay_factor)
        
        # 应用衰减权重到情绪分数
        for col in ['sentiment_mean', 'polarity_mean']:
            if col in sentiment_df.columns:
                sentiment_df[f'{col}_decayed'] = sentiment_df[col] * sentiment_df['decay_weight']
        
        return sentiment_df
    
    def _enhance_news_sentiment_factors(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """增强新闻情绪因子"""
        enhanced_df = sentiment_df.copy()
        
        # 计算情绪动量
        enhanced_df = enhanced_df.sort_values(['ticker', 'date'])
        enhanced_df['sentiment_momentum_1d'] = enhanced_df.groupby('ticker')['sentiment_mean'].diff(1)
        enhanced_df['sentiment_momentum_3d'] = enhanced_df.groupby('ticker')['sentiment_mean'].diff(3)
        
        # 计算情绪波动率
        enhanced_df['sentiment_volatility_5d'] = enhanced_df.groupby('ticker')['sentiment_mean'].rolling(5).std().reset_index(0, drop=True)
        
        # 计算情绪极值
        enhanced_df['sentiment_extremity'] = np.abs(enhanced_df['sentiment_mean'])
        
        # 计算新闻密度
        enhanced_df['news_density'] = enhanced_df['news_count'] / enhanced_df['news_count'].rolling(5).mean()
        
        return enhanced_df
    
    def _add_composite_market_sentiment(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """添加复合市场情绪指标"""
        enhanced_df = market_df.copy()
        
        # 获取主要指标列
        momentum_cols = [col for col in enhanced_df.columns if 'momentum' in col]
        volatility_cols = [col for col in enhanced_df.columns if 'volatility' in col]
        
        if momentum_cols and volatility_cols:
            # 计算复合动量情绪
            enhanced_df['composite_momentum_sentiment'] = enhanced_df[momentum_cols].mean(axis=1)
            
            # 计算复合恐慌指数
            enhanced_df['composite_fear_index'] = enhanced_df[volatility_cols].mean(axis=1)
            
            # 移除硬编码权重的复合计算 - 保持所有因子独立
            # enhanced_df['market_sentiment_bias'] = (
            #     enhanced_df['composite_momentum_sentiment'] * 0.6 - 
            #     enhanced_df['composite_fear_index'] * 0.4
            # )
            # 让机器学习模型自动学习这些因子的最优权重组合
        
        return enhanced_df
    
    # 移除综合情绪因子计算方法
    # 所有情绪因子将作为独立特征参与机器学习训练
    
    def _is_cache_valid(self, cache_key: str, cache_minutes: int) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self.cache or cache_key not in self.cache_timestamps:
            return False
        
        cache_age = datetime.now() - self.cache_timestamps[cache_key]
        return cache_age.total_seconds() < cache_minutes * 60
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'cache_size': len(self.cache),
            'note': 'Individual sentiment factors for ML training (no preset weights)'
        }


# 工厂函数
def create_sentiment_factors(config: SentimentConfig = None) -> EnhancedSentimentFactors:
    """创建情绪因子计算器"""
    return EnhancedSentimentFactors(config)


# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建情绪因子计算器 - 统一使用Polygon API
    config = SentimentConfig(
        polygon_api_key="your_polygon_api_key_here",  # 统一API密钥：新闻+SP500数据
        news_lookback_days=7,
        sp500_lookback_days=30
    )
    
    sentiment_engine = create_sentiment_factors(config)
    
    # 计算情绪因子
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    factors = sentiment_engine.compute_all_sentiment_factors(tickers)
    
    print(f"Computed factors for {len(factors)} factor groups")
    for name, df in factors.items():
        print(f"  {name}: {df.shape}")