#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美股股票池管理系统
功能：
1. 爬取NYSE+NASDAQ所有上市股票
2. 根据流动性、价格、市值等标准筛选
3. 排除低质量股票（penny stocks、低流动性等）
4. 提供手动添加/修改股票池功能
5. 数据库存储和管理
"""

import os
import sqlite3
import requests
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from pathlib import Path

try:
    from ib_insync import *
    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False


@dataclass
class StockInfo:
    """股票信息数据类"""
    symbol: str
    name: str
    exchange: str
    sector: str
    industry: str
    market_cap: float
    price: float
    volume: float
    avg_volume_30d: float
    beta: float
    pe_ratio: float
    dividend_yield: float
    volatility: float
    bid_ask_spread_pct: float
    days_since_ipo: int
    is_tradeable: bool
    quality_score: float
    exclusion_reasons: List[str]
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'exchange': self.exchange,
            'sector': self.sector,
            'industry': self.industry,
            'market_cap': self.market_cap,
            'price': self.price,
            'volume': self.volume,
            'avg_volume_30d': self.avg_volume_30d,
            'beta': self.beta,
            'pe_ratio': self.pe_ratio,
            'dividend_yield': self.dividend_yield,
            'volatility': self.volatility,
            'bid_ask_spread_pct': self.bid_ask_spread_pct,
            'days_since_ipo': self.days_since_ipo,
            'is_tradeable': self.is_tradeable,
            'quality_score': self.quality_score,
            'exclusion_reasons': json.dumps(self.exclusion_reasons),
            'last_updated': self.last_updated.isoformat()
        }


class QualityFilter:
    """股票质量筛选器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 筛选标准
        self.min_price = config.get('min_price', 2.0)  # 最低价格 $2
        self.max_price = config.get('max_price', 10000.0)  # 最高价格 $10000
        self.min_market_cap = config.get('min_market_cap', 200_000_000)  # 最小市值 $200M
        self.min_avg_volume = config.get('min_avg_volume', 100_000)  # 最小日均成交量 10万股
        self.max_bid_ask_spread = config.get('max_bid_ask_spread_pct', 1.0)  # 最大买卖价差 1%
        self.max_volatility = config.get('max_volatility', 100.0)  # 最大年化波动率 100%
        self.max_beta = config.get('max_beta', 3.0)  # 最大Beta值
        self.min_days_since_ipo = config.get('min_days_since_ipo', 365)  # 上市至少1年
        
        # 排除的股票类型
        self.excluded_suffixes = config.get('excluded_suffixes', [
            '.WS', '.WT', '.U', '.UN', '-U', '-WT', '-WS',  # 权证
            '.PR', '-PR',  # 优先股
            '.RT', '-RT'   # 权利
        ])
        
        self.excluded_sectors = config.get('excluded_sectors', [])
        
        # 质量评分权重
        self.scoring_weights = config.get('scoring_weights', {
            'market_cap': 0.25,
            'volume': 0.25,
            'price_stability': 0.20,
            'spread': 0.15,
            'beta': 0.10,
            'age': 0.05
        })
    
    def evaluate_stock(self, stock_info: StockInfo) -> Tuple[bool, float, List[str]]:
        """评估股票质量
        
        Returns:
            Tuple[是否可交易, 质量评分, 排除原因列表]
        """
        exclusion_reasons = []
        quality_score = 0.0
        
        try:
            # 1. 价格筛选
            if stock_info.price < self.min_price:
                exclusion_reasons.append(f"Price too low: ${stock_info.price:.2f} < ${self.min_price}")
            elif stock_info.price > self.max_price:
                exclusion_reasons.append(f"Price too high: ${stock_info.price:.2f} > ${self.max_price}")
            else:
                # 价格稳定性评分 (5-50为好，1-5和>100为差)
                if 5 <= stock_info.price <= 50:
                    price_score = 1.0
                elif 1 <= stock_info.price < 5 or stock_info.price > 100:
                    price_score = 0.3
                else:
                    price_score = 0.7
                quality_score += price_score * self.scoring_weights['price_stability']
            
            # 2. 市值筛选
            if stock_info.market_cap < self.min_market_cap:
                exclusion_reasons.append(f"Market cap too small: ${stock_info.market_cap:,.0f} < ${self.min_market_cap:,.0f}")
            else:
                # 市值评分 (对数标准化)
                cap_score = min(1.0, np.log10(stock_info.market_cap / self.min_market_cap) / 3)
                quality_score += cap_score * self.scoring_weights['market_cap']
            
            # 3. 流动性筛选
            if stock_info.avg_volume_30d < self.min_avg_volume:
                exclusion_reasons.append(f"Volume too low: {stock_info.avg_volume_30d:,.0f} < {self.min_avg_volume:,.0f}")
            else:
                # 成交量评分
                vol_score = min(1.0, np.log10(stock_info.avg_volume_30d / self.min_avg_volume) / 2)
                quality_score += vol_score * self.scoring_weights['volume']
            
            # 4. 买卖价差筛选
            if stock_info.bid_ask_spread_pct > self.max_bid_ask_spread:
                exclusion_reasons.append(f"Bid-ask spread too wide: {stock_info.bid_ask_spread_pct:.2f}% > {self.max_bid_ask_spread:.2f}%")
            else:
                # 价差评分 (越小越好)
                spread_score = max(0, 1 - stock_info.bid_ask_spread_pct / self.max_bid_ask_spread)
                quality_score += spread_score * self.scoring_weights['spread']
            
            # 5. 波动率筛选
            if stock_info.volatility > self.max_volatility:
                exclusion_reasons.append(f"Volatility too high: {stock_info.volatility:.1f}% > {self.max_volatility:.1f}%")
            
            # 6. Beta筛选
            if abs(stock_info.beta) > self.max_beta:
                exclusion_reasons.append(f"Beta too high: {abs(stock_info.beta):.2f} > {self.max_beta:.2f}")
            else:
                # Beta评分 (1附近最好)
                beta_score = max(0, 1 - abs(stock_info.beta - 1) / 2)
                quality_score += beta_score * self.scoring_weights['beta']
            
            # 7. 上市时间筛选
            if stock_info.days_since_ipo < self.min_days_since_ipo:
                exclusion_reasons.append(f"Too new: {stock_info.days_since_ipo} days < {self.min_days_since_ipo} days")
            else:
                # 上市时间评分
                age_score = min(1.0, stock_info.days_since_ipo / (self.min_days_since_ipo * 5))
                quality_score += age_score * self.scoring_weights['age']
            
            # 8. 股票类型筛选
            for suffix in self.excluded_suffixes:
                if stock_info.symbol.endswith(suffix):
                    exclusion_reasons.append(f"Excluded stock type: {suffix}")
                    break
            
            # 9. 行业筛选
            if stock_info.sector in self.excluded_sectors:
                exclusion_reasons.append(f"Excluded sector: {stock_info.sector}")
            
            # 是否可交易
            is_tradeable = len(exclusion_reasons) == 0
            
            return is_tradeable, quality_score, exclusion_reasons
            
        except Exception as e:
            self.logger.error(f"Error evaluating stock {stock_info.symbol}: {e}")
            return False, 0.0, [f"Evaluation error: {str(e)}"]


class StockDataCrawler:
    """股票数据爬虫"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 爬虫配置
        self.max_workers = config.get('max_workers', 10)
        self.request_delay = config.get('request_delay_seconds', 0.1)
        self.timeout = config.get('timeout_seconds', 30)
        self.retry_attempts = config.get('retry_attempts', 3)
        
        # 数据源URL
        self.nasdaq_url = "https://api.nasdaq.com/api/screener/stocks"
        self.fmp_base_url = "https://financialmodelingprep.com/api/v3"
        
        # API密钥（如果需要）
        self.fmp_api_key = config.get('fmp_api_key', '')
        
        # 统计信息
        self.stats = {
            'total_symbols': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'excluded_symbols': 0
        }
    
    def get_all_nasdaq_symbols(self) -> List[Dict[str, str]]:
        """获取NASDAQ所有股票列表"""
        try:
            self.logger.info("Fetching NASDAQ stock list...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            params = {
                'tableonly': 'true',
                'limit': '25000',
                'offset': '0',
                'download': 'true'
            }
            
            response = requests.get(self.nasdaq_url, headers=headers, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            stocks = []
            if 'data' in data and 'table' in data['data'] and 'rows' in data['data']['table']:
                for row in data['data']['table']['rows']:
                    stocks.append({
                        'symbol': row.get('symbol', ''),
                        'name': row.get('name', ''),
                        'exchange': row.get('exchange', ''),
                        'sector': row.get('sector', ''),
                        'industry': row.get('industry', '')
                    })
            
            self.logger.info(f"Fetched {len(stocks)} NASDAQ stocks")
            return stocks
            
        except Exception as e:
            self.logger.error(f"Error fetching NASDAQ symbols: {e}")
            return []
    
    def get_nyse_symbols_from_yfinance(self) -> List[Dict[str, str]]:
        """从yfinance获取NYSE股票列表（备用方法）"""
        try:
            self.logger.info("Fetching NYSE stock list from alternative source...")
            
            # 使用一些已知的大盘股作为起点，然后通过相关推荐扩展
            # 这里简化处理，实际可以使用更完整的数据源
            
            # 备用：从文件读取或使用预定义列表
            nyse_symbols = [
                # 可以在这里添加已知的NYSE股票
                # 或者从其他数据源获取
            ]
            
            return [{'symbol': s, 'name': '', 'exchange': 'NYSE', 'sector': '', 'industry': ''} 
                   for s in nyse_symbols]
            
        except Exception as e:
            self.logger.error(f"Error fetching NYSE symbols: {e}")
            return []
    
    def fetch_stock_data(self, symbol: str) -> Optional[StockInfo]:
        """获取单个股票的详细信息"""
        try:
            ticker = yf.Ticker(symbol)
            
            # 获取基本信息
            info = ticker.info
            
            # 获取历史数据用于计算指标
            hist = ticker.history(period="1y")
            if hist.empty:
                return None
            
            # 基本信息
            name = info.get('longName', info.get('shortName', symbol))
            exchange = info.get('exchange', 'UNKNOWN')
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # 财务指标
            market_cap = info.get('marketCap', 0)
            price = info.get('currentPrice', hist['Close'][-1] if not hist.empty else 0)
            volume = info.get('volume', hist['Volume'][-30:].mean() if len(hist) >= 30 else 0)
            avg_volume_30d = hist['Volume'][-30:].mean() if len(hist) >= 30 else volume
            beta = info.get('beta', 1.0)
            pe_ratio = info.get('trailingPE', 0)
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            
            # 计算波动率
            if len(hist) >= 30:
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100  # 年化波动率%
            else:
                volatility = 0
            
            # 计算买卖价差
            bid = info.get('bid', price)
            ask = info.get('ask', price)
            if bid > 0 and ask > 0:
                bid_ask_spread_pct = ((ask - bid) / ((ask + bid) / 2)) * 100
            else:
                bid_ask_spread_pct = 0
            
            # 计算上市时间
            first_trade_date = info.get('firstTradeDateEpochUtc')
            if first_trade_date:
                ipo_date = datetime.fromtimestamp(first_trade_date)
                days_since_ipo = (datetime.now() - ipo_date).days
            else:
                days_since_ipo = 365  # 默认假设已上市1年
            
            stock_info = StockInfo(
                symbol=symbol,
                name=name,
                exchange=exchange,
                sector=sector,
                industry=industry,
                market_cap=market_cap,
                price=price,
                volume=volume,
                avg_volume_30d=avg_volume_30d,
                beta=beta,
                pe_ratio=pe_ratio,
                dividend_yield=dividend_yield,
                volatility=volatility,
                bid_ask_spread_pct=bid_ask_spread_pct,
                days_since_ipo=days_since_ipo,
                is_tradeable=True,
                quality_score=0.0,
                exclusion_reasons=[],
                last_updated=datetime.now()
            )
            
            self.stats['successful_fetches'] += 1
            return stock_info
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            self.stats['failed_fetches'] += 1
            return None
    
    def crawl_all_stocks(self, symbols: List[str]) -> List[StockInfo]:
        """并发爬取所有股票数据"""
        self.logger.info(f"Starting to crawl {len(symbols)} stocks...")
        self.stats['total_symbols'] = len(symbols)
        
        stocks_data = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_symbol = {
                executor.submit(self.fetch_stock_data, symbol): symbol 
                for symbol in symbols
            }
            
            # 收集结果
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    stock_info = future.result()
                    if stock_info:
                        stocks_data.append(stock_info)
                    
                    # 延迟以避免被限制
                    time.sleep(self.request_delay)
                    
                    # 进度报告
                    completed = self.stats['successful_fetches'] + self.stats['failed_fetches']
                    if completed % 100 == 0:
                        self.logger.info(f"Progress: {completed}/{len(symbols)} stocks processed")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    self.stats['failed_fetches'] += 1
        
        self.logger.info(f"Crawling completed. Success: {self.stats['successful_fetches']}, Failed: {self.stats['failed_fetches']}")
        return stocks_data


class StockUniverseDatabase:
    """股票池数据库管理"""
    
    def __init__(self, db_file: str = "stock_universe.db"):
        self.db_file = db_file
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        try:
            os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
            
            conn = sqlite3.connect(self.db_file)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS stocks (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    exchange TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    price REAL,
                    volume REAL,
                    avg_volume_30d REAL,
                    beta REAL,
                    pe_ratio REAL,
                    dividend_yield REAL,
                    volatility REAL,
                    bid_ask_spread_pct REAL,
                    days_since_ipo INTEGER,
                    is_tradeable BOOLEAN,
                    quality_score REAL,
                    exclusion_reasons TEXT,
                    last_updated TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引
            conn.execute('CREATE INDEX IF NOT EXISTS idx_tradeable ON stocks(is_tradeable)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_quality ON stocks(quality_score)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_exchange ON stocks(exchange)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sector ON stocks(sector)')
            
            # 创建用户自定义股票池表
            conn.execute('''
                CREATE TABLE IF NOT EXISTS custom_portfolios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    description TEXT,
                    symbols TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Database initialized: {self.db_file}")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def save_stocks(self, stocks: List[StockInfo]):
        """批量保存股票数据"""
        try:
            conn = sqlite3.connect(self.db_file)
            
            for stock in stocks:
                data = stock.to_dict()
                
                conn.execute('''
                    INSERT OR REPLACE INTO stocks (
                        symbol, name, exchange, sector, industry, market_cap, price,
                        volume, avg_volume_30d, beta, pe_ratio, dividend_yield,
                        volatility, bid_ask_spread_pct, days_since_ipo, is_tradeable,
                        quality_score, exclusion_reasons, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['symbol'], data['name'], data['exchange'], data['sector'],
                    data['industry'], data['market_cap'], data['price'], data['volume'],
                    data['avg_volume_30d'], data['beta'], data['pe_ratio'],
                    data['dividend_yield'], data['volatility'], data['bid_ask_spread_pct'],
                    data['days_since_ipo'], data['is_tradeable'], data['quality_score'],
                    data['exclusion_reasons'], data['last_updated']
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Saved {len(stocks)} stocks to database")
            
        except Exception as e:
            self.logger.error(f"Error saving stocks to database: {e}")
            raise
    
    def get_tradeable_stocks(self, min_quality_score: float = 0.5, 
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取可交易的股票"""
        try:
            conn = sqlite3.connect(self.db_file)
            conn.row_factory = sqlite3.Row
            
            query = '''
                SELECT * FROM stocks 
                WHERE is_tradeable = 1 AND quality_score >= ?
                ORDER BY quality_score DESC
            '''
            
            if limit:
                query += f' LIMIT {limit}'
            
            cursor = conn.execute(query, (min_quality_score,))
            stocks = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            
            return stocks
            
        except Exception as e:
            self.logger.error(f"Error getting tradeable stocks: {e}")
            return []
    
    def get_stocks_by_sector(self, sector: str) -> List[Dict[str, Any]]:
        """按行业获取股票"""
        try:
            conn = sqlite3.connect(self.db_file)
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute(
                'SELECT * FROM stocks WHERE sector = ? AND is_tradeable = 1 ORDER BY quality_score DESC',
                (sector,)
            )
            stocks = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return stocks
            
        except Exception as e:
            self.logger.error(f"Error getting stocks by sector: {e}")
            return []
    
    def search_stocks(self, keyword: str) -> List[Dict[str, Any]]:
        """搜索股票"""
        try:
            conn = sqlite3.connect(self.db_file)
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute('''
                SELECT * FROM stocks 
                WHERE (symbol LIKE ? OR name LIKE ?) AND is_tradeable = 1
                ORDER BY quality_score DESC
            ''', (f'%{keyword}%', f'%{keyword}%'))
            
            stocks = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return stocks
            
        except Exception as e:
            self.logger.error(f"Error searching stocks: {e}")
            return []
    
    def save_custom_portfolio(self, name: str, symbols: List[str], description: str = ""):
        """保存自定义股票池"""
        try:
            conn = sqlite3.connect(self.db_file)
            
            symbols_json = json.dumps(symbols)
            
            conn.execute('''
                INSERT OR REPLACE INTO custom_portfolios (name, symbols, description, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (name, symbols_json, description))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Saved custom portfolio '{name}' with {len(symbols)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error saving custom portfolio: {e}")
            raise
    
    def get_custom_portfolio(self, name: str) -> Optional[List[str]]:
        """获取自定义股票池"""
        try:
            conn = sqlite3.connect(self.db_file)
            
            cursor = conn.execute(
                'SELECT symbols FROM custom_portfolios WHERE name = ?',
                (name,)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return json.loads(row[0])
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting custom portfolio: {e}")
            return None
    
    def list_custom_portfolios(self) -> List[Dict[str, Any]]:
        """列出所有自定义股票池"""
        try:
            conn = sqlite3.connect(self.db_file)
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute('''
                SELECT name, description, 
                       LENGTH(symbols) - LENGTH(REPLACE(symbols, ',', '')) + 1 as stock_count,
                       created_at, updated_at
                FROM custom_portfolios
                ORDER BY updated_at DESC
            ''')
            
            portfolios = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return portfolios
            
        except Exception as e:
            self.logger.error(f"Error listing custom portfolios: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            conn = sqlite3.connect(self.db_file)
            
            stats = {}
            
            # 总股票数
            cursor = conn.execute('SELECT COUNT(*) FROM stocks')
            stats['total_stocks'] = cursor.fetchone()[0]
            
            # 可交易股票数
            cursor = conn.execute('SELECT COUNT(*) FROM stocks WHERE is_tradeable = 1')
            stats['tradeable_stocks'] = cursor.fetchone()[0]
            
            # 按交易所分布
            cursor = conn.execute('''
                SELECT exchange, COUNT(*) as count 
                FROM stocks WHERE is_tradeable = 1 
                GROUP BY exchange
            ''')
            stats['by_exchange'] = dict(cursor.fetchall())
            
            # 按行业分布
            cursor = conn.execute('''
                SELECT sector, COUNT(*) as count 
                FROM stocks WHERE is_tradeable = 1 
                GROUP BY sector 
                ORDER BY count DESC 
                LIMIT 10
            ''')
            stats['top_sectors'] = dict(cursor.fetchall())
            
            # 质量分布
            cursor = conn.execute('''
                SELECT 
                    CASE 
                        WHEN quality_score >= 0.8 THEN 'High (0.8+)'
                        WHEN quality_score >= 0.6 THEN 'Good (0.6-0.8)'
                        WHEN quality_score >= 0.4 THEN 'Medium (0.4-0.6)'
                        ELSE 'Low (<0.4)'
                    END as quality_tier,
                    COUNT(*) as count
                FROM stocks WHERE is_tradeable = 1
                GROUP BY quality_tier
            ''')
            stats['quality_distribution'] = dict(cursor.fetchall())
            
            conn.close()
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}


class StockUniverseManager:
    """股票池管理器主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.crawler = StockDataCrawler(config.get('crawler', {}))
        self.quality_filter = QualityFilter(config.get('quality_filter', {}))
        self.database = StockUniverseDatabase(config.get('database_file', 'data/stock_universe.db'))
        
        # 更新配置
        self.auto_update_days = config.get('auto_update_days', 7)  # 每周更新
        self.force_update = config.get('force_update', False)
    
    def crawl_and_update_universe(self, force_update: bool = False) -> bool:
        """爬取并更新股票池"""
        try:
            self.logger.info("Starting stock universe update...")
            
            # 检查是否需要更新
            if not force_update and not self._needs_update():
                self.logger.info("Stock universe is up to date, skipping crawl")
                return True
            
            # 1. 获取所有股票列表
            self.logger.info("Fetching stock symbols...")
            nasdaq_stocks = self.crawler.get_all_nasdaq_symbols()
            nyse_stocks = self.crawler.get_nyse_symbols_from_yfinance()
            
            all_symbols = set()
            
            # 合并NASDAQ股票
            for stock in nasdaq_stocks:
                symbol = stock['symbol'].strip()
                if symbol and len(symbol) <= 5:  # 基本验证
                    all_symbols.add(symbol)
            
            # 合并NYSE股票
            for stock in nyse_stocks:
                symbol = stock['symbol'].strip()
                if symbol and len(symbol) <= 5:
                    all_symbols.add(symbol)
            
            all_symbols = list(all_symbols)
            self.logger.info(f"Found {len(all_symbols)} unique symbols")
            
            # 2. 爬取股票数据
            self.logger.info("Crawling stock data...")
            stocks_data = self.crawler.crawl_all_stocks(all_symbols)
            self.logger.info(f"Successfully crawled {len(stocks_data)} stocks")
            
            # 3. 质量筛选
            self.logger.info("Applying quality filters...")
            tradeable_count = 0
            
            for stock in stocks_data:
                is_tradeable, quality_score, exclusion_reasons = self.quality_filter.evaluate_stock(stock)
                
                stock.is_tradeable = is_tradeable
                stock.quality_score = quality_score
                stock.exclusion_reasons = exclusion_reasons
                
                if is_tradeable:
                    tradeable_count += 1
            
            self.logger.info(f"Quality filter results: {tradeable_count}/{len(stocks_data)} stocks are tradeable")
            
            # 4. 保存到数据库
            self.logger.info("Saving to database...")
            self.database.save_stocks(stocks_data)
            
            # 5. 显示统计信息
            stats = self.database.get_statistics()
            self.logger.info("Stock universe update completed!")
            self.logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating stock universe: {e}")
            return False
    
    def _needs_update(self) -> bool:
        """检查是否需要更新"""
        try:
            # 检查数据库是否为空或过期
            stats = self.database.get_statistics()
            
            if stats.get('total_stocks', 0) == 0:
                return True
            
            # 检查最后更新时间（简化实现）
            # 实际应该从数据库检查最后更新时间
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking update necessity: {e}")
            return True
    
    def get_trading_universe(self, min_quality_score: float = 0.6, 
                           max_stocks: int = 1000) -> List[str]:
        """获取交易股票池"""
        stocks = self.database.get_tradeable_stocks(min_quality_score, max_stocks)
        return [stock['symbol'] for stock in stocks]
    
    def create_custom_portfolio(self, name: str, symbols: List[str], 
                              description: str = "") -> bool:
        """创建自定义股票池"""
        try:
            # 验证股票是否存在
            valid_symbols = []
            for symbol in symbols:
                stocks = self.database.search_stocks(symbol)
                if stocks:
                    valid_symbols.append(symbol.upper())
                else:
                    self.logger.warning(f"Symbol {symbol} not found in database")
            
            if not valid_symbols:
                self.logger.error("No valid symbols provided")
                return False
            
            self.database.save_custom_portfolio(name, valid_symbols, description)
            self.logger.info(f"Created custom portfolio '{name}' with {len(valid_symbols)} symbols")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating custom portfolio: {e}")
            return False
    
    def interactive_portfolio_builder(self):
        """交互式股票池构建器"""
        print("\n=== 股票池构建器 ===")
        print("1. 使用默认高质量股票池")
        print("2. 创建自定义股票池")
        print("3. 查看现有股票池")
        print("4. 搜索股票")
        
        choice = input("请选择 (1-4): ").strip()
        
        if choice == '1':
            self._show_default_universe()
        elif choice == '2':
            self._create_custom_portfolio_interactive()
        elif choice == '3':
            self._show_existing_portfolios()
        elif choice == '4':
            self._search_stocks_interactive()
        else:
            print("无效选择")
    
    def _show_default_universe(self):
        """显示默认股票池"""
        print("\n=== 默认股票池 ===")
        
        quality_levels = [0.8, 0.6, 0.4]
        
        for quality in quality_levels:
            stocks = self.database.get_tradeable_stocks(quality, 20)
            print(f"\n质量评分 >= {quality} 的前20只股票:")
            
            for i, stock in enumerate(stocks, 1):
                print(f"{i:2d}. {stock['symbol']:6s} - {stock['name'][:30]:30s} "
                      f"({stock['sector']:15s}) Score: {stock['quality_score']:.2f}")
        
        use_default = input("\n是否使用默认股票池? (y/n): ").strip().lower()
        if use_default == 'y':
            min_quality = float(input("请输入最低质量评分 (0.0-1.0): ") or "0.6")
            max_stocks = int(input("请输入最大股票数量: ") or "500")
            
            symbols = self.get_trading_universe(min_quality, max_stocks)
            print(f"已选择 {len(symbols)} 只股票用于交易")
            
            # 保存为自定义股票池
            name = input("请输入股票池名称 (回车跳过): ").strip()
            if name:
                description = f"Default universe with min quality {min_quality}"
                self.create_custom_portfolio(name, symbols, description)
    
    def _create_custom_portfolio_interactive(self):
        """交互式创建自定义股票池"""
        print("\n=== 创建自定义股票池 ===")
        
        name = input("股票池名称: ").strip()
        if not name:
            print("名称不能为空")
            return
        
        description = input("描述 (可选): ").strip()
        
        symbols = []
        print("请输入股票代码 (输入空行结束):")
        
        while True:
            symbol = input("股票代码: ").strip().upper()
            if not symbol:
                break
            
            # 验证股票
            stocks = self.database.search_stocks(symbol)
            if stocks:
                symbols.append(symbol)
                stock = stocks[0]
                print(f"  已添加: {symbol} - {stock['name']} (质量评分: {stock['quality_score']:.2f})")
            else:
                print(f"  警告: 未找到股票 {symbol}")
                add_anyway = input("  是否仍要添加? (y/n): ").strip().lower()
                if add_anyway == 'y':
                    symbols.append(symbol)
        
        if symbols:
            success = self.create_custom_portfolio(name, symbols, description)
            if success:
                print(f"成功创建股票池 '{name}'，包含 {len(symbols)} 只股票")
            else:
                print("创建股票池失败")
        else:
            print("未添加任何股票")
    
    def _show_existing_portfolios(self):
        """显示现有股票池"""
        print("\n=== 现有股票池 ===")
        
        portfolios = self.database.list_custom_portfolios()
        
        if not portfolios:
            print("暂无自定义股票池")
            return
        
        for i, portfolio in enumerate(portfolios, 1):
            print(f"{i}. {portfolio['name']}")
            print(f"   描述: {portfolio['description']}")
            print(f"   股票数量: {portfolio['stock_count']}")
            print(f"   更新时间: {portfolio['updated_at']}")
            print()
        
        choice = input("选择股票池查看详情 (输入编号): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(portfolios):
                portfolio_name = portfolios[idx]['name']
                symbols = self.database.get_custom_portfolio(portfolio_name)
                if symbols:
                    print(f"\n{portfolio_name} 包含的股票:")
                    for i, symbol in enumerate(symbols, 1):
                        print(f"{i:3d}. {symbol}")
        except (ValueError, IndexError):
            print("无效选择")
    
    def _search_stocks_interactive(self):
        """交互式股票搜索"""
        print("\n=== 股票搜索 ===")
        
        keyword = input("请输入搜索关键字 (股票代码或公司名称): ").strip()
        if not keyword:
            return
        
        stocks = self.database.search_stocks(keyword)
        
        if not stocks:
            print("未找到匹配的股票")
            return
        
        print(f"\n找到 {len(stocks)} 只匹配的股票:")
        for i, stock in enumerate(stocks[:20], 1):  # 只显示前20个
            print(f"{i:2d}. {stock['symbol']:6s} - {stock['name'][:40]:40s}")
            print(f"     行业: {stock['sector']:15s} 质量评分: {stock['quality_score']:.2f}")
            
            if stock['exclusion_reasons']:
                reasons = json.loads(stock['exclusion_reasons'])
                if reasons:
                    print(f"     排除原因: {', '.join(reasons[:2])}")
            print()


def main():
    """主函数 - 演示用法"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 配置
    config = {
        'database_file': 'data/stock_universe.db',
        'crawler': {
            'max_workers': 10,
            'request_delay_seconds': 0.1,
            'timeout_seconds': 30
        },
        'quality_filter': {
            'min_price': 5.0,
            'min_market_cap': 200_000_000,
            'min_avg_volume': 100_000,
            'max_bid_ask_spread_pct': 1.0,
            'max_volatility': 100.0,
            'max_beta': 3.0,
            'min_days_since_ipo': 365
        }
    }
    
    # 创建管理器
    manager = StockUniverseManager(config)
    
    import argparse
    parser = argparse.ArgumentParser(description='Stock Universe Manager')
    parser.add_argument('--update', action='store_true', help='Update stock universe')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    
    args = parser.parse_args()
    
    if args.update:
        print("正在更新股票池...")
        success = manager.crawl_and_update_universe(force_update=True)
        if success:
            print("股票池更新完成!")
        else:
            print("股票池更新失败!")
    
    elif args.interactive:
        manager.interactive_portfolio_builder()
    
    elif args.stats:
        stats = manager.database.get_statistics()
        print("=== 股票池统计信息 ===")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    else:
        print("Stock Universe Manager")
        print("使用 --help 查看选项")
        print("使用 --update 更新股票池")
        print("使用 --interactive 进入交互模式")
        print("使用 --stats 查看统计信息")


if __name__ == "__main__":
    main()