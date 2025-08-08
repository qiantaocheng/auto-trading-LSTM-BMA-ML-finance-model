#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高效美股爬虫 - 分批获取和处理，避免超时
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
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_sp500_symbols():
    """获取S&P 500股票列表"""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_df = tables[0]
        symbols = sp500_df['Symbol'].str.replace('.', '-').tolist()
        logger.info(f"获取到 {len(symbols)} 只S&P 500股票")
        return symbols
    except Exception as e:
        logger.error(f"获取S&P 500失败: {e}")
        return []

def get_popular_stocks():
    """获取热门股票列表"""
    stocks = [
        # 大型科技股
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'ORCL', 'CRM',
        'ADBE', 'NFLX', 'PYPL', 'INTC', 'CSCO', 'AMD', 'UBER', 'LYFT', 'SNOW', 'PLTR',
        'DOCU', 'ZM', 'SHOP', 'SQ', 'ROKU', 'TWLO', 'OKTA', 'DDOG', 'CRWD', 'ZS',
        'MDB', 'NET', 'FSLY', 'WORK', 'TEAM', 'NOW', 'WDAY', 'VEEV', 'SPLK', 'PANW',
        
        # 金融股
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
        'AXP', 'BLK', 'SCHW', 'CB', 'MMC', 'AON', 'PGR', 'TRV', 'ALL', 'AFL',
        'MA', 'V', 'PYPL', 'SQ', 'FIS', 'FISV', 'ADP', 'PAYX', 'BR', 'ICE',
        
        # 医疗保健
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
        'GILD', 'REGN', 'VRTX', 'BIIB', 'ILMN', 'MRNA', 'BNTX', 'ZTS', 'ELV', 'CVS',
        'MDT', 'ISRG', 'BSX', 'SYK', 'BDX', 'BAX', 'EW', 'IDXX', 'A', 'IQV',
        
        # 消费品和零售
        'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW',
        'COST', 'CMG', 'LULU', 'YUM', 'QSR', 'DPZ', 'BKNG', 'MAR', 'HLT', 'MGM',
        'DIS', 'NFLX', 'T', 'VZ', 'TMUS', 'CHTR', 'CMCSA', 'FOXA', 'FOX', 'PARA',
        
        # 工业股
        'GE', 'BA', 'CAT', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC',
        'GD', 'DE', 'EMR', 'ETN', 'PH', 'ROK', 'DOV', 'XYL', 'FTV', 'ITW',
        'LUV', 'DAL', 'UAL', 'AAL', 'JBLU', 'SAVE', 'ALK', 'HA', 'MESA', 'SKYW',
        
        # 能源股
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'BKR', 'HAL',
        'OXY', 'DVN', 'FANG', 'MRO', 'APA', 'CNX', 'EQT', 'AR', 'CHK', 'CTRA',
        
        # 材料和基础工业
        'LIN', 'APD', 'ECL', 'SHW', 'DD', 'DOW', 'NEM', 'FCX', 'VMC', 'MLM',
        'NUE', 'STLD', 'RS', 'X', 'CLF', 'MT', 'PKG', 'CCK', 'IP', 'WRK',
        
        # 公用事业
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'EIX',
        'AWK', 'ATO', 'CMS', 'DTE', 'ED', 'ES', 'ETR', 'EVRG', 'FE', 'NI',
        
        # 房地产
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'SBAC', 'SPG', 'AVB', 'EQR',
        'WELL', 'DLR', 'BXP', 'ARE', 'VTR', 'ESS', 'MAA', 'UDR', 'CPT', 'FRT',
        
        # 中概股和国际股
        'BABA', 'JD', 'PDD', 'BIDU', 'NTES', 'TME', 'NIO', 'XPEV', 'LI', 'BILI',
        'IQ', 'WB', 'DOYU', 'YY', 'MOMO', 'GDS', 'VIPS', 'BZUN', 'KC', 'TAL',
        
        # 生物科技和制药
        'BIIB', 'GILD', 'REGN', 'VRTX', 'BMRN', 'SGEN', 'MYGN', 'IONS', 'EXAS', 'ILMN',
        'CRISPR', 'EDIT', 'NTLA', 'CRSP', 'BEAM', 'PRIME', 'VERV', 'SGMO', 'BLUE', 'FOLD',
        
        # 新兴公司和热门股
        'RBLX', 'U', 'PATH', 'DDOG', 'CRWD', 'ZS', 'OKTA', 'MDB', 'NET', 'FSLY',
        'COIN', 'HOOD', 'SQ', 'AFRM', 'UPST', 'LC', 'SOFI', 'OPEN', 'PTON', 'BYND',
        
        # 游戏娱乐
        'EA', 'ATVI', 'TTWO', 'ZNGA', 'RBLX', 'U', 'HUYA', 'DOYU', 'SE', 'GLUU',
        
        # 半导体
        'NVDA', 'AMD', 'INTC', 'TSM', 'ASML', 'AVGO', 'TXN', 'QCOM', 'AMAT', 'LRCX',
        'ADI', 'MXIM', 'XLNX', 'MCHP', 'MU', 'WDC', 'STX', 'NXPI', 'MRVL', 'ON',
        
        # 其他热门
        'GME', 'AMC', 'BB', 'NOK', 'SNDL', 'TLRY', 'CGC', 'ACB', 'RIOT', 'MARA',
        'DKNG', 'PENN', 'CZR', 'WYNN', 'LVS', 'NCLH', 'CCL', 'RCL', 'EXPE', 'TRIP'
    ]
    
    return list(set(stocks))

def download_stock_data_batch(symbols_batch, batch_num):
    """下载一批股票数据"""
    logger.info(f"正在处理第 {batch_num} 批股票 ({len(symbols_batch)} 只)")
    
    stock_data = []
    failed_symbols = []
    
    for i, symbol in enumerate(symbols_batch, 1):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or len(info) < 5:
                failed_symbols.append(symbol)
                continue
            
            # 获取历史数据
            hist = ticker.history(period="6mo")  # 减少到6个月提高速度
            if hist.empty or len(hist) < 20:
                failed_symbols.append(symbol)
                continue
            
            # 提取基本信息
            price = info.get('currentPrice') or info.get('regularMarketPrice') or hist['Close'].iloc[-1]
            if not price or price <= 0:
                failed_symbols.append(symbol)
                continue
            
            market_cap = info.get('marketCap', 0)
            volume = hist['Volume'].tail(10).mean()  # 只用最近10天
            
            # 简化技术指标计算
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 5 else 0
            
            beta = info.get('beta', 1.0)
            if not isinstance(beta, (int, float)):
                beta = 1.0
            
            stock_info = {
                'symbol': symbol,
                'name': str(info.get('longName', info.get('shortName', symbol)))[:50],
                'sector': str(info.get('sector', 'Unknown'))[:30],
                'industry': str(info.get('industry', 'Unknown'))[:50],
                'market_cap': float(market_cap) if market_cap else 0,
                'price': float(price),
                'volume': float(volume) if volume > 0 else 0,
                'volatility': float(volatility),
                'beta': float(beta),
                'exchange': str(info.get('exchange', 'Unknown')),
                'currency': str(info.get('currency', 'USD')),
                'country': str(info.get('country', 'US'))
            }
            
            stock_data.append(stock_info)
            
            if i % 10 == 0:
                logger.info(f"  批次 {batch_num}: {i}/{len(symbols_batch)} 完成")
            
        except Exception as e:
            failed_symbols.append(symbol)
            logger.debug(f"获取 {symbol} 失败: {e}")
        
        time.sleep(0.1)  # 避免被限流
    
    logger.info(f"第 {batch_num} 批完成: 成功 {len(stock_data)}, 失败 {len(failed_symbols)}")
    return stock_data, failed_symbols

def apply_filters(stocks):
    """应用质量筛选"""
    filters = {
        'min_price': 2.0,
        'min_market_cap': 200_000_000,
        'min_avg_volume': 50_000,  # 降低门槛
        'max_volatility': 200.0,   # 提高门槛
        'max_beta': 5.0
    }
    
    high_quality = []
    medium_quality = []
    low_quality = []
    
    for stock in stocks:
        # 基本筛选
        if (stock['price'] < filters['min_price'] or 
            stock['market_cap'] < filters['min_market_cap'] or
            stock['volume'] < filters['min_avg_volume'] or
            stock['volatility'] > filters['max_volatility'] or
            abs(stock['beta']) > filters['max_beta']):
            continue
        
        # 分层
        if (stock['market_cap'] > 10_000_000_000 and 
            stock['volatility'] < 30 and 
            abs(stock['beta']) < 1.5):
            high_quality.append(stock)
        elif (stock['market_cap'] > 1_000_000_000 and 
              stock['volatility'] < 60):
            medium_quality.append(stock)
        else:
            low_quality.append(stock)
    
    # 排序
    high_quality.sort(key=lambda x: x['market_cap'], reverse=True)
    medium_quality.sort(key=lambda x: x['market_cap'], reverse=True)
    low_quality.sort(key=lambda x: x['market_cap'], reverse=True)
    
    return high_quality, medium_quality, low_quality

def save_results(high_quality, medium_quality, low_quality):
    """保存结果"""
    os.makedirs("efficient_stock_data", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    categories = {
        'high_quality': high_quality,
        'medium_quality': medium_quality,
        'low_quality': low_quality
    }
    
    saved_files = {}
    
    for category, stocks in categories.items():
        if not stocks:
            continue
            
        # 保存股票列表文件
        txt_file = f"efficient_stock_data/{category}_stocks_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"# {category.replace('_', ' ').title()} 股票列表\n")
            f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 股票数量: {len(stocks)}\n\n")
            
            for stock in stocks:
                f.write(f"{stock['symbol']}\n")
        
        # 保存详细数据
        json_file = f"efficient_stock_data/{category}_details_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(stocks, f, indent=2, ensure_ascii=False, default=str)
        
        saved_files[category] = {
            'txt_file': txt_file,
            'json_file': json_file,
            'count': len(stocks)
        }
        
        logger.info(f"保存 {category}: {len(stocks)} 只股票")
    
    # 生成汇总报告
    total_stocks = sum(len(stocks) for stocks in categories.values())
    
    report = f"""# 高效美股爬虫报告

## 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 股票统计
- **总股票数**: {total_stocks:,}
- **高质量股票**: {len(high_quality):,} 只
- **中等质量股票**: {len(medium_quality):,} 只  
- **低质量股票**: {len(low_quality):,} 只

## 质量标准
- 最低价格: $2.0
- 最小市值: $200M
- 最小日均成交量: 50K股
- 最大波动率: 200%
- 最大Beta: 5.0

## 高质量股票前20名
"""
    
    if high_quality:
        for i, stock in enumerate(high_quality[:20], 1):
            report += f"{i:2d}. {stock['symbol']:5s} - {stock['name'][:30]:30s} - ${stock['market_cap']/1e9:.1f}B\n"
    
    report += f"""
## 生成文件
- `{saved_files.get('high_quality', {}).get('txt_file', 'N/A')}` - 高质量股票列表
- `{saved_files.get('medium_quality', {}).get('txt_file', 'N/A')}` - 中等质量股票列表
- `{saved_files.get('low_quality', {}).get('txt_file', 'N/A')}` - 低质量股票列表

---
生成时间: {datetime.now().isoformat()}
"""
    
    with open(f"efficient_stock_data/EFFICIENT_CRAWLER_REPORT_{timestamp}.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    return saved_files

def main():
    """主函数"""
    print("=" * 70)
    print("高效美股爬虫 - 快速获取优质股票数据")
    print("=" * 70)
    
    try:
        # 1. 收集股票符号
        logger.info("收集股票符号...")
        
        sp500_symbols = get_sp500_symbols()
        popular_stocks = get_popular_stocks()
        
        all_symbols = list(set(sp500_symbols + popular_stocks))
        logger.info(f"总共收集到 {len(all_symbols)} 只独特股票符号")
        
        # 2. 分批下载数据
        logger.info("开始分批下载股票数据...")
        
        batch_size = 50  # 每批50只股票
        all_stock_data = []
        all_failed = []
        
        for i in range(0, len(all_symbols), batch_size):
            batch = all_symbols[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            stock_data, failed = download_stock_data_batch(batch, batch_num)
            all_stock_data.extend(stock_data)
            all_failed.extend(failed)
            
            # 保存中间结果
            if batch_num % 5 == 0:  # 每5批保存一次
                logger.info(f"中间保存: 已处理 {len(all_stock_data)} 只股票")
        
        logger.info(f"数据下载完成: 成功 {len(all_stock_data)}, 失败 {len(all_failed)}")
        
        if not all_stock_data:
            logger.error("没有获取到任何股票数据!")
            return False
        
        # 3. 应用筛选
        logger.info("应用质量筛选...")
        high_quality, medium_quality, low_quality = apply_filters(all_stock_data)
        
        # 4. 保存结果
        logger.info("保存结果...")
        saved_files = save_results(high_quality, medium_quality, low_quality)
        
        # 5. 显示总结
        print("\n" + "=" * 70)
        print("爬虫完成!")
        print("=" * 70)
        print(f"总爬取股票: {len(all_stock_data):,}")
        print(f"高质量股票: {len(high_quality):,}")
        print(f"中等质量股票: {len(medium_quality):,}")
        print(f"低质量股票: {len(low_quality):,}")
        print(f"失败股票: {len(all_failed):,}")
        
        if high_quality:
            print(f"\n前10只高质量股票:")
            for i, stock in enumerate(high_quality[:10], 1):
                print(f"{i:2d}. {stock['symbol']:5s} - {stock['name'][:30]:30s} - 市值: ${stock['market_cap']/1e9:.1f}B")
        
        print(f"\n结果文件保存在: efficient_stock_data/ 目录")
        
        return True
        
    except Exception as e:
        logger.error(f"爬虫执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n程序结束，成功: {'是' if success else '否'}")