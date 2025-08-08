#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
美股全市场爬虫
使用yfinance获取所有美股股票数据，替代NASDAQ和NYSE的API
新增功能：从Wikipedia爬取NASDAQ和NYSE股票列表

主要功能:
- 获取所有美股符号列表
- 从Wikipedia爬取NASDAQ和NYSE股票列表
- 按市值、交易量等筛选股票
- 支持实时数据更新
- 集成到量化交易管理器

Author: AI Assistant
Version: 2.0
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
import concurrent.futures
from pathlib import Path
import warnings
import re
from bs4 import BeautifulSoup

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class USStockCrawler:
    """美股全市场爬虫类"""
    
    def __init__(self, cache_dir: str = "stock_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        
        # 设置请求头，模拟浏览器
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 常用的美股交易所代码
        self.exchanges = {
            'NASDAQ': 'NASDAQ',
            'NYSE': 'NYSE',
            'AMEX': 'AMEX',
            'OTC': 'OTC'
        }
        
        # 缓存文件路径
        self.all_stocks_cache = self.cache_dir / "all_us_stocks.json"
        self.filtered_stocks_cache = self.cache_dir / "filtered_us_stocks.json"
        self.stock_info_cache = self.cache_dir / "stock_info_cache.json"
        self.nasdaq_wiki_cache = self.cache_dir / "nasdaq_wiki_stocks.json"
        self.nyse_wiki_cache = self.cache_dir / "nyse_wiki_stocks.json"
        
        logger.info(f"[US爬虫] 初始化完成，缓存目录: {self.cache_dir}")
    
    def crawl_nasdaq_wikipedia(self) -> List[str]:
        """从Wikipedia爬取NASDAQ股票列表"""
        try:
            logger.info("[US爬虫] 开始从Wikipedia爬取NASDAQ股票列表...")
            
            # 检查缓存
            if self.nasdaq_wiki_cache.exists():
                cache_time = datetime.fromtimestamp(self.nasdaq_wiki_cache.stat().st_mtime)
                if datetime.now() - cache_time < timedelta(hours=24):
                    logger.info("[US爬虫] 使用缓存的NASDAQ Wikipedia数据")
                    with open(self.nasdaq_wiki_cache, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        return cached_data.get('tickers', [])
            
            # Wikipedia NASDAQ页面URL
            urls = [
                'https://en.wikipedia.org/wiki/List_of_Nasdaq-100_companies',
                'https://en.wikipedia.org/wiki/Nasdaq-100',
                'https://en.wikipedia.org/wiki/List_of_companies_listed_on_the_Nasdaq'
            ]
            
            nasdaq_stocks = set()
            
            for url in urls:
                try:
                    logger.info(f"[US爬虫] 爬取URL: {url}")
                    response = self.session.get(url, timeout=30)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # 查找表格
                    tables = soup.find_all('table', {'class': 'wikitable'})
                    
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows[1:]:  # 跳过表头
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 2:
                                # 尝试提取股票代码
                                ticker_cell = cells[0] if cells else None
                                if ticker_cell:
                                    ticker = ticker_cell.get_text(strip=True)
                                    # 清理股票代码
                                    ticker = self._clean_ticker(ticker)
                                    if ticker and len(ticker) <= 5:  # 股票代码通常不超过5个字符
                                        nasdaq_stocks.add(ticker)
                    
                    # 避免请求过于频繁
                    time.sleep(2)
                    
                except Exception as e:
                    logger.warning(f"[US爬虫] 爬取 {url} 失败: {e}")
                    continue
            
            # 转换为列表并排序
            nasdaq_list = sorted(list(nasdaq_stocks))
            
            # 保存到缓存
            cache_data = {
                'tickers': nasdaq_list,
                'timestamp': datetime.now().isoformat(),
                'source': 'Wikipedia',
                'total_count': len(nasdaq_list)
            }
            
            with open(self.nasdaq_wiki_cache, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[US爬虫] 从Wikipedia获取到 {len(nasdaq_list)} 只NASDAQ股票")
            return nasdaq_list
            
        except Exception as e:
            logger.error(f"[US爬虫] 爬取NASDAQ Wikipedia失败: {e}")
            return []
    
    def crawl_nyse_wikipedia(self) -> List[str]:
        """从Wikipedia爬取NYSE股票列表"""
        try:
            logger.info("[US爬虫] 开始从Wikipedia爬取NYSE股票列表...")
            
            # 检查缓存
            if self.nyse_wiki_cache.exists():
                cache_time = datetime.fromtimestamp(self.nyse_wiki_cache.stat().st_mtime)
                if datetime.now() - cache_time < timedelta(hours=24):
                    logger.info("[US爬虫] 使用缓存的NYSE Wikipedia数据")
                    with open(self.nyse_wiki_cache, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        return cached_data.get('tickers', [])
            
            # Wikipedia NYSE页面URL
            urls = [
                'https://en.wikipedia.org/wiki/List_of_NYSE_American_stocks',
                'https://en.wikipedia.org/wiki/List_of_companies_listed_on_the_New_York_Stock_Exchange',
                'https://en.wikipedia.org/wiki/New_York_Stock_Exchange'
            ]
            
            nyse_stocks = set()
            
            for url in urls:
                try:
                    logger.info(f"[US爬虫] 爬取URL: {url}")
                    response = self.session.get(url, timeout=30)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # 查找表格
                    tables = soup.find_all('table', {'class': 'wikitable'})
                    
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows[1:]:  # 跳过表头
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 2:
                                # 尝试提取股票代码
                                ticker_cell = cells[0] if cells else None
                                if ticker_cell:
                                    ticker = ticker_cell.get_text(strip=True)
                                    # 清理股票代码
                                    ticker = self._clean_ticker(ticker)
                                    if ticker and len(ticker) <= 5:  # 股票代码通常不超过5个字符
                                        nyse_stocks.add(ticker)
                    
                    # 避免请求过于频繁
                    time.sleep(2)
                    
                except Exception as e:
                    logger.warning(f"[US爬虫] 爬取 {url} 失败: {e}")
                    continue
            
            # 转换为列表并排序
            nyse_list = sorted(list(nyse_stocks))
            
            # 保存到缓存
            cache_data = {
                'tickers': nyse_list,
                'timestamp': datetime.now().isoformat(),
                'source': 'Wikipedia',
                'total_count': len(nyse_list)
            }
            
            with open(self.nyse_wiki_cache, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[US爬虫] 从Wikipedia获取到 {len(nyse_list)} 只NYSE股票")
            return nyse_list
            
        except Exception as e:
            logger.error(f"[US爬虫] 爬取NYSE Wikipedia失败: {e}")
            return []
    
    def _clean_ticker(self, ticker: str) -> str:
        """清理股票代码"""
        if not ticker:
            return ""
        
        # 移除特殊字符和空格
        ticker = re.sub(r'[^\w\-\.]', '', ticker.strip())
        
        # 处理BRK.B这种格式
        if '.' in ticker:
            ticker = ticker.replace('.', '-')
        
        # 确保只包含字母、数字和连字符
        ticker = re.sub(r'[^A-Za-z0-9\-]', '', ticker)
        
        return ticker.upper()
    
    def get_sp500_stocks(self) -> List[str]:
        """获取S&P 500股票列表"""
        try:
            logger.info("[US爬虫] 获取S&P 500股票列表...")
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            df = pd.read_html(url)[0]
            tickers = df['Symbol'].tolist()
            
            # 清理股票代码（移除特殊字符）
            cleaned_tickers = []
            for ticker in tickers:
                # 处理BRK.B这种格式
                if '.' in ticker:
                    ticker = ticker.replace('.', '-')
                cleaned_tickers.append(ticker)
            
            logger.info(f"[US爬虫] 获取到 {len(cleaned_tickers)} 只S&P 500股票")
            return cleaned_tickers
            
        except Exception as e:
            logger.error(f"[US爬虫] 获取S&P 500股票失败: {e}")
            return []
    
    def get_nasdaq_stocks(self) -> List[str]:
        """获取NASDAQ股票列表（增强版，包含Wikipedia数据）"""
        try:
            logger.info("[US爬虫] 获取NASDAQ股票列表...")
            
            # 首先尝试从Wikipedia获取
            wiki_nasdaq = self.crawl_nasdaq_wikipedia()
            
            # 使用NASDAQ官方API作为补充
            api_nasdaq = self._get_nasdaq_api_stocks()
            
            # 合并两个来源的股票
            all_nasdaq = set(wiki_nasdaq + api_nasdaq)
            
            # 如果API获取失败，使用备用列表
            if not api_nasdaq:
                fallback_nasdaq = self._get_fallback_nasdaq_stocks()
                all_nasdaq.update(fallback_nasdaq)
            
            nasdaq_list = sorted(list(all_nasdaq))
            logger.info(f"[US爬虫] 获取到 {len(nasdaq_list)} 只NASDAQ股票")
            return nasdaq_list
            
        except Exception as e:
            logger.error(f"[US爬虫] 获取NASDAQ股票失败: {e}")
            return self._get_fallback_nasdaq_stocks()
    
    def _get_nasdaq_api_stocks(self) -> List[str]:
        """从NASDAQ API获取股票列表"""
        try:
            url = "https://api.nasdaq.com/api/screener/stocks"
            params = {
                'tableonly': 'true',
                'limit': '10000',
                'offset': '0',
                'download': 'true'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'table' in data['data']:
                    stocks = data['data']['table']['rows']
                    tickers = [stock['symbol'] for stock in stocks if stock.get('symbol')]
                    return tickers
            
            return []
            
        except Exception as e:
            logger.warning(f"[US爬虫] NASDAQ API获取失败: {e}")
            return []
    
    def get_nyse_stocks(self) -> List[str]:
        """获取NYSE股票列表（增强版，包含Wikipedia数据）"""
        try:
            logger.info("[US爬虫] 获取NYSE股票列表...")
            
            # 首先尝试从Wikipedia获取
            wiki_nyse = self.crawl_nyse_wikipedia()
            
            # 合并静态列表
            static_nyse = self._get_static_nyse_stocks()
            
            # 合并两个来源的股票
            all_nyse = set(wiki_nyse + static_nyse)
            
            nyse_list = sorted(list(all_nyse))
            logger.info(f"[US爬虫] 获取到 {len(nyse_list)} 只NYSE股票")
            return nyse_list
            
        except Exception as e:
            logger.error(f"[US爬虫] 获取NYSE股票失败: {e}")
            return self._get_static_nyse_stocks()
    
    def _get_static_nyse_stocks(self) -> List[str]:
        """获取静态NYSE股票列表"""
        return [
            # 主要蓝筹股
            'JPM', 'JNJ', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'V', 'PFE',
            'KO', 'MRK', 'ABBV', 'PEP', 'TMO', 'ACN', 'COST', 'DHR', 'ABT', 'MCD',
            'WFC', 'BMY', 'HON', 'NEE', 'PM', 'TXN', 'UPS', 'LOW', 'C', 'QCOM',
            'LIN', 'RTX', 'AMGN', 'SPGI', 'AMT', 'BLK', 'SYK', 'MDLZ', 'CAT', 'GE',
            'IBM', 'DE', 'ADP', 'TJX', 'ISRG', 'CVS', 'MMM', 'TMUS', 'ZTS', 'SO',
            'BSX', 'MO', 'NOW', 'PLD', 'SHW', 'CB', 'DUK', 'CSX', 'ITW', 'WM',
            'COP', 'EMR', 'NSC', 'AON', 'BSX', 'PNC', 'EL', 'MCO', 'ECL', 'APD',
            # 金融股
            'GS', 'MS', 'AXP', 'USB', 'TFC', 'COF', 'PNC', 'SCHW', 'BK', 'STT',
            # 能源股
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL',
            # 消费品股
            'WMT', 'PG', 'KO', 'PEP', 'MCD', 'COST', 'HD', 'TGT', 'LOW', 'SBUX',
            # 医疗保健股
            'JNJ', 'PFE', 'UNH', 'MRK', 'ABT', 'TMO', 'DHR', 'AMGN', 'SYK', 'BSX',
            # 工业股
            'GE', 'CAT', 'BA', 'MMM', 'HON', 'UPS', 'LMT', 'RTX', 'DE', 'EMR'
        ]
    
    def _get_fallback_nasdaq_stocks(self) -> List[str]:
        """备用NASDAQ股票列表"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE',
            'CRM', 'ORCL', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX',
            'KLAC', 'MRVL', 'SNPS', 'CDNS', 'FTNT', 'TEAM', 'DOCU', 'ZM', 'SPOT', 'ROKU',
            'PYPL', 'SQ', 'SHOP', 'ZS', 'CRWD', 'OKTA', 'NET', 'DDOG', 'SNOW', 'PLTR'
        ]
    
    def get_all_us_stocks(self, use_cache: bool = True, max_age_hours: int = 24) -> List[str]:
        """获取所有美股股票列表（增强版，包含Wikipedia数据）"""
        try:
            # 检查缓存
            if use_cache and self.all_stocks_cache.exists():
                cache_time = datetime.fromtimestamp(self.all_stocks_cache.stat().st_mtime)
                if datetime.now() - cache_time < timedelta(hours=max_age_hours):
                    logger.info("[US爬虫] 使用缓存的股票列表")
                    with open(self.all_stocks_cache, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        return cached_data.get('tickers', [])
            
            logger.info("[US爬虫] 开始获取所有美股股票列表...")
            
            all_tickers = set()
            
            # 获取S&P 500股票
            sp500_stocks = self.get_sp500_stocks()
            all_tickers.update(sp500_stocks)
            
            # 获取NASDAQ股票（包含Wikipedia数据）
            nasdaq_stocks = self.get_nasdaq_stocks()
            all_tickers.update(nasdaq_stocks)
            
            # 获取NYSE股票（包含Wikipedia数据）
            nyse_stocks = self.get_nyse_stocks()
            all_tickers.update(nyse_stocks)
            
            # 添加一些其他知名股票
            other_stocks = [
                # ETFs
                'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'BND',
                # REITs
                'O', 'REIT', 'VNQ', 'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB',
                # 其他行业代表股票
                'BRK-B', 'BRK-A', 'TSM', 'ASML', 'NVO', 'NTES', 'BABA', 'JD', 'BIDU', 'NIO'
            ]
            all_tickers.update(other_stocks)
            
            # 转换为列表并排序
            final_tickers = sorted(list(all_tickers))
            
            # 保存到缓存
            cache_data = {
                'tickers': final_tickers,
                'timestamp': datetime.now().isoformat(),
                'total_count': len(final_tickers),
                'sources': ['S&P 500', 'NASDAQ Wikipedia', 'NYSE Wikipedia', 'Static Lists']
            }
            
            with open(self.all_stocks_cache, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[US爬虫] 获取完成，总计 {len(final_tickers)} 只股票")
            return final_tickers
            
        except Exception as e:
            logger.error(f"[US爬虫] 获取所有股票失败: {e}")
            return []
    
    def get_stock_info_batch(self, tickers: List[str], batch_size: int = 2000) -> Dict[str, Dict]:
        """批量获取股票信息"""
        try:
            logger.info(f"[US爬虫] 开始批量获取 {len(tickers)} 只股票信息...")
            
            stock_info = {}
            failed_tickers = []
            
            # 分批处理
            for i in range(0, len(tickers), batch_size):
                batch_tickers = tickers[i:i + batch_size]
                logger.info(f"[US爬虫] 处理第 {i//batch_size + 1} 批，共 {len(batch_tickers)} 只股票")
                
                # 并行获取股票信息
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_ticker = {
                        executor.submit(self._get_single_stock_info, ticker): ticker 
                        for ticker in batch_tickers
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_ticker):
                        ticker = future_to_ticker[future]
                        try:
                            info = future.result(timeout=30)
                            if info:
                                stock_info[ticker] = info
                            else:
                                failed_tickers.append(ticker)
                        except Exception as e:
                            logger.warning(f"[US爬虫] 获取 {ticker} 信息失败: {e}")
                            failed_tickers.append(ticker)
                
                # 避免请求过于频繁
                time.sleep(1)
            
            logger.info(f"[US爬虫] 批量获取完成，成功 {len(stock_info)} 只，失败 {len(failed_tickers)} 只")
            
            if failed_tickers:
                logger.warning(f"[US爬虫] 失败的股票: {failed_tickers[:20]}...")
            
            return stock_info
            
        except Exception as e:
            logger.error(f"[US爬虫] 批量获取股票信息失败: {e}")
            return {}
    
    def _get_single_stock_info(self, ticker: str) -> Optional[Dict]:
        """获取单只股票信息"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # 基本验证
            if not info or 'symbol' not in info:
                return None
            
            # 提取关键信息
            result = {
                'symbol': ticker,
                'longName': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'marketCap': info.get('marketCap', 0),
                'volume': info.get('volume', 0),
                'avgVolume': info.get('averageVolume', 0),
                'price': info.get('regularMarketPrice', 0),
                'exchange': info.get('exchange', 'Unknown'),
                'country': info.get('country', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', '')[:500] if info.get('longBusinessSummary') else '',
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"[US爬虫] 获取 {ticker} 详细信息失败: {e}")
            return None
    
    def filter_stocks(self, 
                     tickers: List[str],
                     min_market_cap: int = 1000000,  # 10亿美元
                     min_volume: int = 10000,         # 100万股
                     max_price: float = 1000.0,         # 最高价格
                     min_price: float = 1.0,            # 最低价格
                     exclude_sectors: List[str] = None,
                     include_sectors: List[str] = None) -> List[str]:
        """根据条件筛选股票"""
        try:
            logger.info(f"[US爬虫] 开始筛选股票，原始数量: {len(tickers)}")
            
            if exclude_sectors is None:
                exclude_sectors = []
            
            # 获取股票信息
            stock_info = self.get_stock_info_batch(tickers)
            
            filtered_tickers = []
            
            for ticker, info in stock_info.items():
                try:
                    # 市值筛选
                    market_cap = info.get('marketCap', 0)
                    if market_cap < min_market_cap:
                        continue
                    
                    # 成交量筛选
                    volume = info.get('avgVolume', 0)
                    if volume < min_volume:
                        continue
                    
                    # 价格筛选
                    price = info.get('price', 0)
                    if price < min_price or price > max_price:
                        continue
                    
                    # 行业筛选
                    sector = info.get('sector', '')
                    if exclude_sectors and sector in exclude_sectors:
                        continue
                    
                    if include_sectors and sector not in include_sectors:
                        continue
                    
                    # 排除一些特殊类型的股票
                    name = info.get('longName', '').upper()
                    if any(keyword in name for keyword in ['ETF', 'FUND', 'TRUST', 'WARRANT']):
                        continue
                    
                    filtered_tickers.append(ticker)
                    
                except Exception as e:
                    logger.warning(f"[US爬虫] 筛选 {ticker} 时出错: {e}")
                    continue
            
            # 保存筛选结果
            filter_data = {
                'tickers': filtered_tickers,
                'timestamp': datetime.now().isoformat(),
                'original_count': len(tickers),
                'filtered_count': len(filtered_tickers),
                'filters': {
                    'min_market_cap': min_market_cap,
                    'min_volume': min_volume,
                    'max_price': max_price,
                    'min_price': min_price,
                    'exclude_sectors': exclude_sectors,
                    'include_sectors': include_sectors
                }
            }
            
            with open(self.filtered_stocks_cache, 'w', encoding='utf-8') as f:
                json.dump(filter_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[US爬虫] 筛选完成，筛选后数量: {len(filtered_tickers)}")
            return filtered_tickers
            
        except Exception as e:
            logger.error(f"[US爬虫] 股票筛选失败: {e}")
            return []
    
    def get_trading_pool_stocks(self, 
                               pool_size: int = 2000,
                               use_cache: bool = True) -> List[str]:
        """获取适合交易的股票池"""
        try:
            logger.info(f"[US爬虫] 生成交易股票池，目标数量: {pool_size}")
            
            # 获取所有股票
            all_stocks = self.get_all_us_stocks(use_cache=use_cache)
            if not all_stocks:
                return []
            
            # 筛选条件
            filtered_stocks = self.filter_stocks(
                all_stocks[:4000],  # 限制处理数量以提高效率
                min_market_cap=5000000,    # 5亿美元市值
                min_volume=50000,           # 50万股日均成交量
                max_price=500.0,             # 最高500美元
                min_price=2,              # 最低2美元
                exclude_sectors=['Real Estate Investment Trusts']  # 排除REITs
            )
            
            # 如果筛选后的股票数量超过目标，按市值排序取前N只
            if len(filtered_stocks) > pool_size:
                # 获取股票信息并按市值排序
                stock_info = {}
                for ticker in filtered_stocks[:pool_size * 2]:  # 获取2倍数量以便排序
                    info = self._get_single_stock_info(ticker)
                    if info and info.get('marketCap', 0) > 0:
                        stock_info[ticker] = info
                
                # 按市值降序排序
                sorted_stocks = sorted(
                    stock_info.items(),
                    key=lambda x: x[1].get('marketCap', 0),
                    reverse=True
                )
                
                final_stocks = [ticker for ticker, _ in sorted_stocks[:pool_size]]
            else:
                final_stocks = filtered_stocks
            
            logger.info(f"[US爬虫] 交易股票池生成完成，最终数量: {len(final_stocks)}")
            return final_stocks
            
        except Exception as e:
            logger.error(f"[US爬虫] 生成交易股票池失败: {e}")
            return []
    
    def update_stock_cache(self, force_update: bool = False):
        """更新股票缓存"""
        try:
            logger.info("[US爬虫] 开始更新股票缓存...")
            
            # 获取最新股票列表
            all_stocks = self.get_all_us_stocks(use_cache=not force_update)
            
            # 更新股票信息缓存
            if all_stocks:
                stock_info = self.get_stock_info_batch(all_stocks[:1000])  # 限制数量
                
                with open(self.stock_info_cache, 'w', encoding='utf-8') as f:
                    json.dump(stock_info, f, indent=2, ensure_ascii=False)
                
                logger.info(f"[US爬虫] 缓存更新完成，股票信息: {len(stock_info)} 只")
            
        except Exception as e:
            logger.error(f"[US爬虫] 更新缓存失败: {e}")


def main():
    """主函数 - 测试用"""
    try:
        crawler = USStockCrawler()
        
        print("=== 美股爬虫测试 ===")
        
        # 获取所有股票
        all_stocks = crawler.get_all_us_stocks()
        print(f"获取到股票总数: {len(all_stocks)}")
      
        # 生成交易股票池
        trading_pool = crawler.get_trading_pool_stocks(pool_size=100)
        print(f"交易股票池数量: {len(trading_pool)}")
    
        
        # 获取部分股票详细信息
        sample_stocks = trading_pool[:5]
        stock_info = crawler.get_stock_info_batch(sample_stocks)
        
        print("\n=== 股票详细信息示例 ===")
        for ticker, info in stock_info.items():
            print(f"{ticker}: {info.get('longName', 'N/A')}")
        
    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    main()