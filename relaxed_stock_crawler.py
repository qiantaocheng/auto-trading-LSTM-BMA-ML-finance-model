#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
放宽条件股票爬虫 - 使用新的宽松筛选标准获取更多股票
筛选条件:
- 最低股价: ≥$2.00
- 最小市值: ≥$70M  
- 最小日均成交量: ≥10K股
- Beta值范围: -4.0到+4.0
- 不考虑波动率和质量分层
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RelaxedStockCrawler:
    """放宽条件的股票爬虫"""
    
    def __init__(self):
        self.all_symbols = set()
        self.stock_data = {}
        self.failed_symbols = set()
        
        # 新的宽松筛选标准
        self.filters = {
            'min_price': 2.0,           # 最低股价 $2.00
            'min_market_cap': 70_000_000,   # 最小市值 $70M
            'min_avg_volume': 10_000,   # 最小日均成交量 10K股
            'min_beta': -4.0,           # 最小Beta
            'max_beta': 4.0             # 最大Beta
            # 删除: max_volatility (波动率不再考虑)
            # 删除: 质量分层标准
        }
    
    def get_comprehensive_symbol_list(self):
        """获取comprehensive股票符号列表"""
        logger.info("正在获取comprehensive股票符号列表...")
        
        symbols = set()
        
        # 1. S&P 500
        try:
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            sp500_table = pd.read_html(sp500_url)[0]
            sp500_symbols = sp500_table['Symbol'].str.replace('.', '-').tolist()
            symbols.update(sp500_symbols)
            logger.info(f"获取S&P 500: {len(sp500_symbols)}只")
        except Exception as e:
            logger.warning(f"获取S&P 500失败: {e}")
        
        # 2. NASDAQ列表 (使用更全面的方法)
        nasdaq_symbols = self.get_nasdaq_comprehensive()
        symbols.update(nasdaq_symbols)
        logger.info(f"获取NASDAQ: {len(nasdaq_symbols)}只")
        
        # 3. 热门股票补充
        popular_stocks = [
            # 大型科技股
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'ORCL', 'CRM',
            'ADBE', 'NFLX', 'PYPL', 'INTC', 'CSCO', 'AMD', 'UBER', 'LYFT', 'SNOW', 'PLTR',
            
            # 金融股
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
            
            # 医疗股
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
            
            # 消费股
            'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW',
            
            # 工业股
            'GE', 'BA', 'CAT', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC',
            
            # 能源股
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'BKR', 'HAL',
            
            # 中概股
            'BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'XPEV', 'LI', 'BILI', 'IQ', 'TME',
            
            # 生物科技
            'MRNA', 'BNTX', 'NVAX', 'REGN', 'VRTX', 'BIIB', 'GILD', 'ILMN',
            
            # 新兴股票
            'RBLX', 'COIN', 'HOOD', 'SQ', 'AFRM', 'UPST', 'SOFI', 'PATH', 'DDOG', 'CRWD',
            
            # 热门ETF和其他
            'SPY', 'QQQ', 'IWM', 'VTI', 'ARKK', 'GME', 'AMC', 'BB', 'NOK'
        ]
        
        symbols.update(popular_stocks)
        logger.info(f"添加热门股票: {len(popular_stocks)}只")
        
        # 4. 小盘股补充 (Russell 2000样本)
        small_cap_samples = [
            'ABCB', 'ACIW', 'ACLS', 'ADTN', 'AEYE', 'AGIO', 'AGNC', 'ALRM', 'ALRS', 'AMWD',
            'ANET', 'APOG', 'ARCT', 'ARDX', 'ARKR', 'ARRY', 'ARTL', 'ASML', 'ASTR', 'ATEC',
            'ATRO', 'AUPH', 'AVIR', 'AXDX', 'BAND', 'BCPC', 'BDTX', 'BEAM', 'BGCP', 'BILI',
            'BLFS', 'BMBL', 'BMRN', 'BOOT', 'BPMC', 'BRMK', 'BTAI', 'BURL', 'BYND', 'CACC',
            'CAKE', 'CALM', 'CARA', 'CARG', 'CARS', 'CASY', 'CBRL', 'CCMP', 'CDAY', 'CDMO',
            'CELH', 'CERS', 'CGEM', 'CHGG', 'CHWY', 'CLOV', 'CNMD', 'CODX', 'COHU', 'COLB',
            'CONN', 'CORT', 'COTY', 'COUR', 'CREE', 'CRSR', 'CRUS', 'CSOD', 'CTLT', 'CTRA',
            'CUTR', 'CVBF', 'CVCO', 'CVGW', 'CWST', 'CYBE', 'CYTH', 'DARE', 'DASH', 'DBVT',
            'DCBO', 'DCOM', 'DFIN', 'DGII', 'DISH', 'DNLI', 'DOCU', 'DOMO', 'DRNA', 'DSGX',
            'DVAX', 'DXCM', 'DYNT', 'EBON', 'ECHO', 'EDIT', 'EGOV', 'EHTH', 'EIGI', 'ELLI'
        ]
        
        symbols.update(small_cap_samples)
        logger.info(f"添加小盘股样本: {len(small_cap_samples)}只")
        
        total_symbols = list(symbols)
        logger.info(f"总计收集到: {len(total_symbols)}只独特股票符号")
        
        return total_symbols
    
    def get_nasdaq_comprehensive(self):
        """获取更全面的NASDAQ股票列表"""
        symbols = set()
        
        try:
            # 方法1: 使用NASDAQ API (多个页面)
            url = "https://api.nasdaq.com/api/screener/stocks"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            for offset in range(0, 6000, 25):  # 获取更多页面
                params = {
                    'tableonly': 'true',
                    'limit': '25',
                    'offset': str(offset)
                }
                
                try:
                    response = requests.get(url, headers=headers, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'data' in data and 'table' in data['data']:
                            rows = data['data']['table']['rows']
                            if not rows:
                                break
                                
                            for row in rows:
                                symbol = row.get('symbol', '').strip()
                                if symbol and len(symbol) <= 5:
                                    symbols.add(symbol)
                            
                            if offset % 500 == 0:
                                logger.info(f"NASDAQ API: 已获取 {len(symbols)} 只股票 (offset: {offset})")
                        else:
                            break
                    else:
                        break
                        
                except Exception as e:
                    logger.debug(f"NASDAQ API请求失败 offset {offset}: {e}")
                    break
                
                time.sleep(0.1)
                
        except Exception as e:
            logger.warning(f"NASDAQ API获取失败: {e}")
        
        return symbols
    
    def download_stock_data(self, symbol):
        """下载单只股票数据 - 使用新的宽松筛选标准"""
        try:
            ticker = yf.Ticker(symbol)
            
            # 获取基本信息
            info = ticker.info
            if not info or len(info) < 3:
                return None
            
            # 获取历史数据 (减少到3个月提高速度)
            hist = ticker.history(period="3mo")
            if hist.empty or len(hist) < 10:
                return None
            
            # 提取基本信息
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not price:
                price = hist['Close'].iloc[-1]
            
            if not price or price <= 0:
                return None
            
            # 应用新的宽松筛选标准
            
            # 1. 股价筛选 (≥$2.00)
            if price < self.filters['min_price']:
                return None
            
            # 2. 市值筛选 (≥$70M)
            market_cap = info.get('marketCap', 0)
            if market_cap < self.filters['min_market_cap']:
                return None
            
            # 3. 成交量筛选 (≥10K)
            volume = hist['Volume'].tail(10).mean()
            if volume < self.filters['min_avg_volume']:
                return None
            
            # 4. Beta值筛选 (-4.0 到 +4.0)
            beta = info.get('beta')
            if beta is not None:
                if beta < self.filters['min_beta'] or beta > self.filters['max_beta']:
                    return None
            else:
                beta = 1.0  # 默认值
            
            # 不再计算波动率 (删除了波动率筛选)
            
            stock_data = {
                'symbol': symbol,
                'name': str(info.get('longName', info.get('shortName', symbol)))[:50],
                'sector': str(info.get('sector', 'Unknown'))[:30],
                'industry': str(info.get('industry', 'Unknown'))[:50],
                'market_cap': float(market_cap) if market_cap else 0,
                'price': float(price),
                'volume': float(volume) if volume > 0 else 0,
                'beta': float(beta),
                'exchange': str(info.get('exchange', 'Unknown')),
                'currency': str(info.get('currency', 'USD')),
                'country': str(info.get('country', 'US')),
                'updated_at': datetime.now().isoformat(),
                'meets_criteria': True  # 所有通过筛选的都标记为符合标准
            }
            
            return stock_data
            
        except Exception as e:
            logger.debug(f"获取 {symbol} 数据失败: {e}")
            return None
    
    def download_all_data(self, symbols):
        """并发下载所有股票数据"""
        logger.info(f"开始下载 {len(symbols)} 只股票的数据...")
        
        successful_count = 0
        failed_count = 0
        
        # 使用更多线程加速
        with ThreadPoolExecutor(max_workers=15) as executor:
            future_to_symbol = {executor.submit(self.download_stock_data, symbol): symbol 
                              for symbol in symbols}
            
            for i, future in enumerate(as_completed(future_to_symbol), 1):
                symbol = future_to_symbol[future]
                
                try:
                    data = future.result(timeout=20)
                    if data:
                        self.stock_data[symbol] = data
                        successful_count += 1
                    else:
                        self.failed_symbols.add(symbol)
                        failed_count += 1
                        
                except Exception as e:
                    self.failed_symbols.add(symbol)
                    failed_count += 1
                    logger.debug(f"{symbol} 处理异常: {e}")
                
                if i % 200 == 0:
                    logger.info(f"进度: {i}/{len(symbols)}, 成功: {successful_count}, 失败: {failed_count}")
        
        logger.info(f"数据下载完成! 成功: {successful_count}, 失败: {failed_count}")
        return successful_count
    
    def save_results(self):
        """保存结果 - 不再分层，所有股票统一处理"""
        logger.info("正在保存结果...")
        
        os.makedirs("relaxed_stock_data", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 按行业分类但不分质量层级
        stocks_by_sector = {}
        for symbol, data in self.stock_data.items():
            sector = data.get('sector', 'Unknown')
            if sector not in stocks_by_sector:
                stocks_by_sector[sector] = []
            stocks_by_sector[sector].append(data)
        
        # 按市值排序每个行业的股票
        for sector in stocks_by_sector:
            stocks_by_sector[sector].sort(key=lambda x: x['market_cap'], reverse=True)
        
        # 保存所有符合条件的股票
        all_qualified_stocks = list(self.stock_data.values())
        all_qualified_stocks.sort(key=lambda x: x['market_cap'], reverse=True)
        
        # 保存股票列表文件
        txt_file = f"relaxed_stock_data/all_qualified_stocks_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"# 放宽条件股票列表\n")
            f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 筛选标准: 股价≥$2.00, 市值≥$70M, 成交量≥10K, -4≤Beta≤4\n")
            f.write(f"# 股票数量: {len(all_qualified_stocks)} 只\n\n")
            
            for stock in all_qualified_stocks:
                f.write(f"{stock['symbol']}\n")
        
        # 保存详细数据
        json_file = f"relaxed_stock_data/all_qualified_details_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_qualified_stocks, f, indent=2, ensure_ascii=False, default=str)
        
        # 按行业保存
        sector_file = f"relaxed_stock_data/stocks_by_sector_{timestamp}.json"
        with open(sector_file, 'w', encoding='utf-8') as f:
            json.dump(stocks_by_sector, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成报告
        self.generate_report(all_qualified_stocks, stocks_by_sector, timestamp)
        
        logger.info(f"保存完成: {len(all_qualified_stocks)} 只股票")
        return all_qualified_stocks, txt_file, json_file
    
    def generate_report(self, stocks, sectors, timestamp):
        """生成报告"""
        report = f"""# 放宽条件股票爬虫报告

## 爬取时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 新筛选标准 (放宽条件)
- **最低股价**: ≥$2.00 (避免仙股)
- **最小市值**: ≥$70M (大幅降低，包含更多小盘股)
- **最小日均成交量**: ≥10K股 (大幅降低，确保基本流动性)
- **Beta值范围**: -4.0到+4.0 (合理的市场敏感度)
- **已删除**: 不再考虑波动率筛选
- **已删除**: 不再进行质量分层，统一处理

## 📊 爬取结果
- **总爬取股票数**: {len(self.stock_data):,}
- **符合条件股票数**: {len(stocks):,}
- **筛选通过率**: {len(stocks)/len(self.stock_data)*100:.1f}%
- **失败股票数**: {len(self.failed_symbols):,}

## 🏢 行业分布
"""
        
        for sector, sector_stocks in sorted(sectors.items(), key=lambda x: len(x[1]), reverse=True):
            report += f"- **{sector}**: {len(sector_stocks)} 只股票\n"
        
        report += f"""
## 💰 市值分布
"""
        
        # 市值统计
        market_caps = [s['market_cap'] for s in stocks if s['market_cap'] > 0]
        if market_caps:
            report += f"- **最大市值**: ${max(market_caps)/1e9:.1f}B\n"
            report += f"- **最小市值**: ${min(market_caps)/1e6:.1f}M\n"
            report += f"- **平均市值**: ${sum(market_caps)/len(market_caps)/1e6:.1f}M\n"
        
        # 市值分段统计
        mega_cap = len([s for s in stocks if s['market_cap'] >= 200e9])  # ≥$200B
        large_cap = len([s for s in stocks if 10e9 <= s['market_cap'] < 200e9])  # $10B-$200B
        mid_cap = len([s for s in stocks if 2e9 <= s['market_cap'] < 10e9])    # $2B-$10B
        small_cap = len([s for s in stocks if 300e6 <= s['market_cap'] < 2e9])  # $300M-$2B
        micro_cap = len([s for s in stocks if 70e6 <= s['market_cap'] < 300e6]) # $70M-$300M
        
        report += f"""
### 市值分段统计
- **巨型股** (≥$200B): {mega_cap} 只
- **大盘股** ($10B-$200B): {large_cap} 只  
- **中盘股** ($2B-$10B): {mid_cap} 只
- **小盘股** ($300M-$2B): {small_cap} 只
- **微盘股** ($70M-$300M): {micro_cap} 只

## 📈 股价分布
"""
        
        prices = [s['price'] for s in stocks]
        if prices:
            under_5 = len([p for p in prices if p < 5])
            under_10 = len([p for p in prices if 5 <= p < 10])
            under_50 = len([p for p in prices if 10 <= p < 50])
            under_100 = len([p for p in prices if 50 <= p < 100])
            over_100 = len([p for p in prices if p >= 100])
            
            report += f"- **$2-$5**: {under_5} 只\n"
            report += f"- **$5-$10**: {under_10} 只\n"
            report += f"- **$10-$50**: {under_50} 只\n"
            report += f"- **$50-$100**: {under_100} 只\n"
            report += f"- **≥$100**: {over_100} 只\n"
        
        report += f"""
## 🔝 前20只股票 (按市值)
"""
        
        top_20 = sorted(stocks, key=lambda x: x['market_cap'], reverse=True)[:20]
        for i, stock in enumerate(top_20, 1):
            report += f"{i:2d}. {stock['symbol']:5s} - {stock['name'][:30]:30s} - ${stock['market_cap']/1e9:.1f}B\n"
        
        report += f"""
## 📁 生成文件
- `relaxed_stock_data/all_qualified_stocks_{timestamp}.txt` - 所有符合条件的股票列表
- `relaxed_stock_data/all_qualified_details_{timestamp}.json` - 详细股票数据
- `relaxed_stock_data/stocks_by_sector_{timestamp}.json` - 按行业分类数据

---
生成时间: {datetime.now().isoformat()}
筛选标准: 大幅放宽，涵盖更多股票
"""
        
        report_file = f"relaxed_stock_data/RELAXED_CRAWLER_REPORT_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"报告已生成: {report_file}")

def main():
    """主函数"""
    print("=" * 70)
    print("放宽条件股票爬虫 - 获取更多股票")
    print("=" * 70)
    print("新筛选标准:")
    print("- 最低股价: ≥$2.00")
    print("- 最小市值: ≥$70M")  
    print("- 最小日均成交量: ≥10K股")
    print("- Beta值范围: -4.0到+4.0")
    print("- 不考虑波动率和质量分层")
    print("=" * 70)
    
    crawler = RelaxedStockCrawler()
    
    try:
        # 1. 收集股票符号
        logger.info("第一步: 收集股票符号...")
        all_symbols = crawler.get_comprehensive_symbol_list()
        
        if len(all_symbols) < 100:
            logger.error("收集到的股票符号太少，请检查网络连接")
            return False
        
        # 2. 下载数据
        logger.info("第二步: 下载股票数据...")
        success_count = crawler.download_all_data(all_symbols)
        
        if success_count == 0:
            logger.error("没有成功下载任何股票数据")
            return False
        
        # 3. 保存结果
        logger.info("第三步: 保存结果...")
        qualified_stocks, txt_file, json_file = crawler.save_results()
        
        # 4. 显示总结
        print("\n" + "=" * 70)
        print("放宽条件爬虫完成!")
        print("=" * 70)
        print(f"总爬取: {len(crawler.stock_data):,} 只股票")
        print(f"符合条件: {len(qualified_stocks):,} 只股票")
        print(f"失败: {len(crawler.failed_symbols):,} 只股票")
        
        # 显示市值分布
        if qualified_stocks:
            market_caps = [s['market_cap'] for s in qualified_stocks if s['market_cap'] > 0]
            if market_caps:
                print(f"\n市值范围: ${min(market_caps)/1e6:.0f}M - ${max(market_caps)/1e9:.1f}B")
                
                mega_cap = len([s for s in qualified_stocks if s['market_cap'] >= 200e9])
                large_cap = len([s for s in qualified_stocks if 10e9 <= s['market_cap'] < 200e9])
                mid_cap = len([s for s in qualified_stocks if 2e9 <= s['market_cap'] < 10e9])
                small_cap = len([s for s in qualified_stocks if 300e6 <= s['market_cap'] < 2e9])
                micro_cap = len([s for s in qualified_stocks if s['market_cap'] < 300e6])
                
                print(f"巨型股: {mega_cap}, 大盘股: {large_cap}, 中盘股: {mid_cap}")
                print(f"小盘股: {small_cap}, 微盘股: {micro_cap}")
        
        print(f"\n结果文件:")
        print(f"- {txt_file}")
        print(f"- {json_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"爬虫执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 放宽条件爬虫成功! 获得更多股票用于模型训练。")
    else:
        print("\n❌ 爬虫失败!")