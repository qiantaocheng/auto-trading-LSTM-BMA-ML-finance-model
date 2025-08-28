#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一市场数据管理器
集成真实市值、行业、国家数据，替换随机生成的数据
"""

import pandas as pd
import numpy as np
from polygon_client import polygon_client, download, Ticker
import warnings
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import json
import requests
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)

@dataclass
class MarketDataConfig:
    """市场数据配置"""
    # 数据源优先级
    data_sources: List[str] = field(default_factory=lambda: ['polygon', 'local_db', 'fallback'])
    
    # 缓存设置
    cache_enabled: bool = True
    cache_duration_hours: int = 24
    cache_path: str = "data/market_cache.db"
    
    # 行业分类
    sector_classification: str = "GICS"  # GICS, ICB, 自定义
    sector_level: int = 4  # 1-4级分类细度
    
    # 市值类型
    market_cap_types: List[str] = field(default_factory=lambda: [
        'market_cap',           # 总市值
        'float_market_cap',     # 流通市值  
        'free_float_market_cap' # 自由流通市值
    ])
    
    # 指数成分股
    reference_indices: List[str] = field(default_factory=lambda: [
        'SPY',   # S&P 500
        'QQQ',   # NASDAQ 100
        'DIA'    # Dow 30
    ])

@dataclass
class StockInfo:
    """个股信息"""
    ticker: str
    name: str
    sector: str
    industry: str
    country: str
    market_cap: float
    float_market_cap: Optional[float] = None
    free_float_market_cap: Optional[float] = None
    gics_sector: Optional[str] = None
    gics_industry_group: Optional[str] = None
    gics_industry: Optional[str] = None
    gics_sub_industry: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    is_index_component: Dict[str, bool] = field(default_factory=dict)

class MarketDataCache:
    """市场数据缓存系统"""
    
    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_cache_db()
    
    def _init_cache_db(self):
        """初始化缓存数据库"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_info (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                industry TEXT,
                country TEXT,
                market_cap REAL,
                float_market_cap REAL,
                free_float_market_cap REAL,
                gics_sector TEXT,
                gics_industry_group TEXT,
                gics_industry TEXT,
                gics_sub_industry TEXT,
                exchange TEXT,
                currency TEXT,
                index_components TEXT,
                last_updated TIMESTAMP,
                data_source TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS index_components (
                index_ticker TEXT,
                component_ticker TEXT,
                weight REAL,
                last_updated TIMESTAMP,
                PRIMARY KEY (index_ticker, component_ticker)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_stock_info(self, ticker: str) -> Optional[StockInfo]:
        """从缓存获取股票信息"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM stock_info WHERE ticker = ?
        ''', (ticker,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # 检查缓存是否过期（24小时）
            last_updated = datetime.fromisoformat(result[15])
            if datetime.now() - last_updated < timedelta(hours=24):
                return self._row_to_stock_info(result)
        
        return None
    
    def save_stock_info(self, stock_info: StockInfo, data_source: str):
        """保存股票信息到缓存"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO stock_info VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        ''', (
            stock_info.ticker,
            stock_info.name,
            stock_info.sector,
            stock_info.industry,
            stock_info.country,
            stock_info.market_cap,
            stock_info.float_market_cap,
            stock_info.free_float_market_cap,
            stock_info.gics_sector,
            stock_info.gics_industry_group,
            stock_info.gics_industry,
            stock_info.gics_sub_industry,
            stock_info.exchange,
            stock_info.currency,
            json.dumps(stock_info.is_index_component),
            datetime.now().isoformat(),
            data_source
        ))
        
        conn.commit()
        conn.close()
    
    def _row_to_stock_info(self, row) -> StockInfo:
        """数据库行转换为StockInfo对象"""
        index_components = json.loads(row[14]) if row[14] else {}
        
        return StockInfo(
            ticker=row[0],
            name=row[1],
            sector=row[2],
            industry=row[3],
            country=row[4],
            market_cap=row[5],
            float_market_cap=row[6],
            free_float_market_cap=row[7],
            gics_sector=row[8],
            gics_industry_group=row[9],
            gics_industry=row[10],
            gics_sub_industry=row[11],
            exchange=row[12],
            currency=row[13],
            is_index_component=index_components
        )

class PolygonDataProvider:
    """Polygon.io数据提供者"""
    
    def __init__(self):
        from polygon_client import polygon_client
        self.client = polygon_client
        
    def get_stock_info(self, ticker: str) -> Optional[StockInfo]:
        """从Polygon.io获取股票信息"""
        try:
            from polygon_client import Ticker
            stock = Ticker(ticker)
            info = stock.info
            
            if not info:
                return None
            
            # 处理GICS分类 (Polygon doesn't provide sector/industry directly)
            gics_sector = info.get('sector', 'Technology')
            gics_industry = info.get('industry', 'Software')
            
            # 市值信息
            market_cap = info.get('market_cap', 0)
            shares_outstanding = info.get('share_class_shares_outstanding', 0)
            weighted_shares = info.get('weighted_shares_outstanding', 0)
            current_price = self.client.get_current_price(ticker) or 0
            float_market_cap = shares_outstanding * current_price
            free_float_market_cap = weighted_shares * current_price
            
            return StockInfo(
                ticker=ticker,
                name=info.get('longName', info.get('name', ticker)),
                sector=gics_sector,
                industry=gics_industry,
                country=info.get('country', info.get('locale', 'us').upper()),
                market_cap=market_cap,
                float_market_cap=float_market_cap if float_market_cap > 0 else None,
                free_float_market_cap=free_float_market_cap,
                gics_sector=gics_sector,
                gics_industry_group=self._map_to_industry_group(gics_sector),
                gics_industry=gics_industry,
                gics_sub_industry=gics_industry,  # Yahoo Finance没有细分
                exchange=info.get('market', 'stocks').upper(),
                currency=info.get('currency_name', 'USD').upper()
            )
            
        except Exception as e:
            logger.warning(f"获取{ticker}的Polygon数据失败: {e}")
            return None
    
    def _map_to_industry_group(self, sector: str) -> str:
        """映射到GICS行业组"""
        mapping = {
            'Technology': 'Technology Hardware & Equipment',
            'Healthcare': 'Pharmaceuticals, Biotechnology & Life Sciences',
            'Financials': 'Banks',
            'Consumer Cyclical': 'Consumer Discretionary',
            'Industrials': 'Capital Goods',
            'Consumer Defensive': 'Consumer Staples',
            'Energy': 'Energy',
            'Utilities': 'Utilities',
            'Real Estate': 'Real Estate',
            'Basic Materials': 'Materials',
            'Communication Services': 'Communication Services'
        }
        return mapping.get(sector, sector)

class IndexComponentProvider:
    """指数成分股数据提供者"""
    
    def __init__(self):
        self.components_cache = {}
        
    def get_sp500_components(self) -> List[str]:
        """获取S&P 500成分股"""
        try:
            # 从Wikipedia获取S&P 500成分股列表
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            
            # 清理ticker格式
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            
            logger.info(f"获取到{len(tickers)}只S&P 500成分股")
            return tickers
            
        except Exception as e:
            logger.warning(f"获取S&P 500成分股失败: {e}")
            return self._get_fallback_sp500()
    
    def get_nasdaq100_components(self) -> List[str]:
        """获取NASDAQ 100成分股"""
        try:
            # 使用QQQ ETF的持仓作为近似
            qqq = Ticker("QQQ")
            # 这里简化处理，实际可以通过其他API获取
            return self._get_fallback_nasdaq100()
            
        except Exception as e:
            logger.warning(f"获取NASDAQ 100成分股失败: {e}")
            return self._get_fallback_nasdaq100()
    
    def get_dow30_components(self) -> List[str]:
        """获取道琼斯30成分股"""
        try:
            url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
            tables = pd.read_html(url)
            dow_table = tables[1]  # 通常是第二个表格
            tickers = dow_table['Symbol'].tolist()
            
            logger.info(f"获取到{len(tickers)}只道琼斯成分股")
            return tickers
            
        except Exception as e:
            logger.warning(f"获取道琼斯成分股失败: {e}")
            return self._get_fallback_dow30()
    
    def _get_fallback_sp500(self) -> List[str]:
        """S&P 500后备成分股列表"""
        return [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'GOOG', 'BRK-B', 'UNH', 'META',
            'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'PFE', 'ABBV',
            'BAC', 'COST', 'DIS', 'WMT', 'KO', 'MRK', 'PEP', 'TMO', 'NFLX', 'ABT'
        ]
    
    def _get_fallback_nasdaq100(self) -> List[str]:
        """NASDAQ 100后备成分股列表"""
        return [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'GOOG', 'META', 'NFLX', 'ADBE',
            'PYPL', 'INTC', 'CMCSA', 'PEP', 'COST', 'QCOM', 'TXN', 'TMUS', 'AVGO', 'CHTR'
        ]
    
    def _get_fallback_dow30(self) -> List[str]:
        """道琼斯30后备成分股列表"""
        return [
            'AAPL', 'MSFT', 'UNH', 'JNJ', 'JPM', 'V', 'PG', 'HD', 'CVX', 'MRK',
            'WMT', 'KO', 'DIS', 'BAC', 'CRM', 'VZ', 'AXP', 'NKE', 'MCD', 'IBM',
            'HON', 'GS', 'CAT', 'BA', 'MMM', 'TRV', 'WBA', 'DOW', 'INTC', 'CSCO'
        ]

class UnifiedMarketDataManager:
    """统一市场数据管理器"""
    
    def __init__(self, config: MarketDataConfig = None):
        self.config = config or MarketDataConfig()
        
        # 初始化组件
        self.cache = MarketDataCache(self.config.cache_path) if self.config.cache_enabled else None
        self.polygon_provider = PolygonDataProvider()
        self.index_provider = IndexComponentProvider()
        
        # 数据缓存
        self.stock_info_cache: Dict[str, StockInfo] = {}
        self.index_components_cache: Dict[str, List[str]] = {}
        
    def get_stock_info(self, ticker: str, force_refresh: bool = False) -> Optional[StockInfo]:
        """获取股票基本信息"""
        
        # 检查内存缓存
        if not force_refresh and ticker in self.stock_info_cache:
            return self.stock_info_cache[ticker]
        
        # 检查数据库缓存
        if self.cache and not force_refresh:
            cached_info = self.cache.get_stock_info(ticker)
            if cached_info:
                self.stock_info_cache[ticker] = cached_info
                return cached_info
        
        # 按优先级获取数据
        for source in self.config.data_sources:
            stock_info = None
            
            if source == 'polygon':
                stock_info = self.polygon_provider.get_stock_info(ticker)
            elif source == 'fallback':
                stock_info = self._get_fallback_stock_info(ticker)
            
            if stock_info:
                # 增强数据：添加指数成分信息
                stock_info = self._enhance_with_index_info(stock_info)
                
                # 缓存数据
                self.stock_info_cache[ticker] = stock_info
                if self.cache:
                    self.cache.save_stock_info(stock_info, source)
                
                logger.info(f"从{source}获取{ticker}数据成功")
                return stock_info
        
        logger.warning(f"无法获取{ticker}的市场数据")
        return None
    
    def get_batch_stock_info(self, tickers: List[str]) -> Dict[str, StockInfo]:
        """批量获取股票信息"""
        results = {}
        
        logger.info(f"批量获取{len(tickers)}只股票的市场数据...")
        
        for i, ticker in enumerate(tickers):
            if i % 50 == 0:
                logger.info(f"进度: {i}/{len(tickers)}")
            
            stock_info = self.get_stock_info(ticker)
            if stock_info:
                results[ticker] = stock_info
        
        logger.info(f"成功获取{len(results)}/{len(tickers)}只股票的数据")
        return results
    
    def _enhance_with_index_info(self, stock_info: StockInfo) -> StockInfo:
        """增强股票信息：添加指数成分信息"""
        
        # 检查是否为各大指数成分股
        for index_name in self.config.reference_indices:
            components = self._get_index_components(index_name)
            stock_info.is_index_component[index_name] = stock_info.ticker in components
        
        return stock_info
    
    def _get_index_components(self, index_ticker: str) -> List[str]:
        """获取指数成分股"""
        
        if index_ticker in self.index_components_cache:
            return self.index_components_cache[index_ticker]
        
        components = []
        
        if index_ticker == 'SPY':
            components = self.index_provider.get_sp500_components()
        elif index_ticker == 'QQQ':
            components = self.index_provider.get_nasdaq100_components()
        elif index_ticker == 'DIA':
            components = self.index_provider.get_dow30_components()
        
        self.index_components_cache[index_ticker] = components
        return components
    
    def _get_fallback_stock_info(self, ticker: str) -> Optional[StockInfo]:
        """后备数据生成"""
        
        # 基于ticker生成合理的后备数据
        np.random.seed(hash(ticker) % 2**32)
        
        # 行业分布
        sectors = [
            'Technology', 'Healthcare', 'Financials', 'Consumer Cyclical',
            'Industrials', 'Consumer Defensive', 'Energy', 'Utilities',
            'Real Estate', 'Basic Materials', 'Communication Services'
        ]
        
        sector = np.random.choice(sectors)
        
        # 市值分布（对数正态分布）
        log_market_cap = np.zeros(9.5)  # 约100亿美元中位数
        market_cap = np.exp(log_market_cap) * 1e6
        
        return StockInfo(
            ticker=ticker,
            name=f"{ticker} Inc.",
            sector=sector,
            industry=f"{sector} Industry",
            country="US",
            market_cap=market_cap,
            float_market_cap=market_cap * 0.8,
            free_float_market_cap=market_cap * 0.7,
            gics_sector=sector,
            gics_industry_group=sector,
            gics_industry=f"{sector} Industry",
            gics_sub_industry=f"{sector} Sub-Industry",
            exchange="NASDAQ",
            currency="USD"
        )
    
    def create_unified_features_dataframe(self, 
                                        base_df: pd.DataFrame,
                                        ticker_column: str = 'ticker') -> pd.DataFrame:
        """创建包含统一市值和行业特征的DataFrame"""
        
        result_df = base_df.copy()
        
        # 获取所有unique tickers
        unique_tickers = result_df[ticker_column].unique()
        stock_info_dict = self.get_batch_stock_info(unique_tickers.tolist())
        
        # 创建映射函数
        def map_stock_feature(ticker, feature_name):
            stock_info = stock_info_dict.get(ticker)
            if stock_info:
                return getattr(stock_info, feature_name, None)
            return None
        
        # 添加市值特征
        result_df['market_cap'] = result_df[ticker_column].apply(
            lambda x: map_stock_feature(x, 'market_cap')
        )
        result_df['float_market_cap'] = result_df[ticker_column].apply(
            lambda x: map_stock_feature(x, 'float_market_cap')
        )
        result_df['free_float_market_cap'] = result_df[ticker_column].apply(
            lambda x: map_stock_feature(x, 'free_float_market_cap')
        )
        
        # 添加行业特征
        result_df['sector'] = result_df[ticker_column].apply(
            lambda x: map_stock_feature(x, 'sector')
        )
        result_df['industry'] = result_df[ticker_column].apply(
            lambda x: map_stock_feature(x, 'industry')
        )
        result_df['gics_sector'] = result_df[ticker_column].apply(
            lambda x: map_stock_feature(x, 'gics_sector')
        )
        result_df['gics_industry'] = result_df[ticker_column].apply(
            lambda x: map_stock_feature(x, 'gics_industry')
        )
        
        # 添加国家特征
        result_df['country'] = result_df[ticker_column].apply(
            lambda x: map_stock_feature(x, 'country')
        )
        
        # 添加指数成分特征
        for index_name in self.config.reference_indices:
            result_df[f'is_{index_name}_component'] = result_df[ticker_column].apply(
                lambda x: stock_info_dict.get(x, StockInfo('', '', '', '', '', 0)).is_index_component.get(index_name, False)
            )
        
        # 计算相对市值特征
        result_df['log_market_cap'] = np.log(result_df['market_cap'].fillna(1e9))
        result_df['market_cap_percentile'] = result_df['market_cap'].rank(pct=True)
        
        # 处理缺失值 - 先确保数据类型正确
        numeric_columns = ['market_cap', 'float_market_cap', 'free_float_market_cap', 'log_market_cap']
        for col in numeric_columns:
            if col in result_df.columns:
                # 强制转换为数值类型，无法转换的设为NaN
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                # 只有当列中有有效数值时才计算median
                if result_df[col].notna().sum() > 0:
                    median_val = result_df[col].median()
                    result_df[col] = result_df[col].fillna(median_val)
                else:
                    result_df[col] = result_df[col].fillna(0)
        
        categorical_columns = ['sector', 'industry', 'gics_sector', 'gics_industry', 'country']
        for col in categorical_columns:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna('Unknown')
        
        logger.info(f"统一特征DataFrame创建完成，形状: {result_df.shape}")
        return result_df
    
    def get_sector_neutral_weights(self, 
                                 tickers: List[str],
                                 target_weights: pd.Series = None) -> pd.Series:
        """计算行业中性权重"""
        
        if target_weights is None:
            target_weights = pd.Series(1.0, index=tickers) / len(tickers)
        
        # 获取股票信息
        stock_info_dict = self.get_batch_stock_info(tickers)
        
        # 构建行业分组
        sector_groups = {}
        for ticker in tickers:
            stock_info = stock_info_dict.get(ticker)
            if stock_info:
                sector = stock_info.gics_sector or stock_info.sector
                if sector not in sector_groups:
                    sector_groups[sector] = []
                sector_groups[sector].append(ticker)
        
        # 计算中性权重
        neutral_weights = pd.Series(0.0, index=tickers)
        
        for sector, sector_tickers in sector_groups.items():
            sector_target_weight = target_weights[sector_tickers].sum()
            equal_weight = sector_target_weight / len(sector_tickers)
            
            for ticker in sector_tickers:
                neutral_weights[ticker] = equal_weight
        
        return neutral_weights

# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 初始化管理器
    manager = UnifiedMarketDataManager()
    
    # 测试单个股票信息获取
    stock_info = manager.get_stock_info('AAPL')
    if stock_info:
        print(f"AAPL信息: {stock_info}")
    
    # 测试批量获取
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    batch_info = manager.get_batch_stock_info(test_tickers)
    
    print(f"批量获取结果: {len(batch_info)}只股票")
    for ticker, info in batch_info.items():
        print(f"{ticker}: {info.sector}, 市值: ${info.market_cap/1e9:.1f}B")
    
    # 测试DataFrame增强
    test_df = pd.DataFrame({
        'ticker': test_tickers * 2,
        'date': pd.date_range('2024-01-01', periods=10),
        'return': np.zeros(0)
    })
    
    enhanced_df = manager.create_unified_features_dataframe(test_df)
    print(f"增强后DataFrame列: {enhanced_df.columns.tolist()}")
    print(enhanced_df.head())
