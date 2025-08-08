#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速股票池测试脚本
用于测试股票数据获取和筛选功能
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import requests


def test_yfinance_connection():
    """测试yfinance连接"""
    print("🔧 测试yfinance连接...")
    
    try:
        # 测试获取AAPL数据
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        hist = ticker.history(period="5d")
        
        print(f"✅ yfinance连接正常")
        print(f"   AAPL当前价格: ${info.get('currentPrice', 'N/A')}")
        print(f"   历史数据行数: {len(hist)}")
        return True
        
    except Exception as e:
        print(f"❌ yfinance连接失败: {e}")
        return False


def test_nasdaq_api():
    """测试NASDAQ API"""
    print("\n🔧 测试NASDAQ API连接...")
    
    try:
        url = "https://api.nasdaq.com/api/screener/stocks"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        params = {
            'tableonly': 'true',
            'limit': '10',  # 只测试10只股票
            'offset': '0'
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'data' in data and 'table' in data['data']:
            rows = data['data']['table']['rows']
            print(f"✅ NASDAQ API连接正常")
            print(f"   获取到 {len(rows)} 只股票数据")
            
            # 显示前5只股票
            print("   前5只股票:")
            for i, row in enumerate(rows[:5], 1):
                symbol = row.get('symbol', 'N/A')
                name = row.get('name', 'N/A')[:30]
                print(f"   {i}. {symbol} - {name}")
            
            return True
        else:
            print("❌ NASDAQ API返回数据格式异常")
            return False
            
    except Exception as e:
        print(f"❌ NASDAQ API连接失败: {e}")
        return False


def get_sample_stocks():
    """获取样本股票进行测试"""
    print("\n📊 获取样本股票数据...")
    
    # 使用一些知名股票进行测试
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'UNH']
    
    results = []
    
    for symbol in test_symbols:
        try:
            print(f"   处理 {symbol}...", end=' ')
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1y")
            
            if hist.empty:
                print("❌ 无历史数据")
                continue
            
            # 基本信息
            name = info.get('longName', info.get('shortName', symbol))[:30]
            sector = info.get('sector', 'Unknown')
            market_cap = info.get('marketCap', 0)
            price = info.get('currentPrice', hist['Close'][-1])
            volume = hist['Volume'][-30:].mean()
            
            # 计算波动率
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
            
            # 计算Beta (简化)
            beta = info.get('beta', 1.0)
            
            stock_data = {
                'symbol': symbol,
                'name': name,
                'sector': sector,
                'market_cap': market_cap,
                'price': price,
                'avg_volume': volume,
                'volatility': volatility,
                'beta': beta,
                'tradeable': True
            }
            
            results.append(stock_data)
            print("✅")
            
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    return results


def apply_quality_filters(stocks):
    """应用质量筛选"""
    print(f"\n🔍 应用质量筛选 (共{len(stocks)}只股票)...")
    
    # 筛选标准
    filters = {
        'min_price': 2.0,
        'min_market_cap': 200_000_000,  # 2亿美元
        'min_avg_volume': 100_000,      # 10万股
        'max_volatility': 100.0,        # 100%
        'max_beta': 3.0
    }
    
    print("筛选标准:")
    for key, value in filters.items():
        print(f"   {key}: {value:,}")
    
    filtered_stocks = []
    
    for stock in stocks:
        exclusion_reasons = []
        
        # 价格筛选
        if stock['price'] < filters['min_price']:
            exclusion_reasons.append(f"价格过低: ${stock['price']:.2f}")
        
        # 市值筛选
        if stock['market_cap'] < filters['min_market_cap']:
            exclusion_reasons.append(f"市值过小: ${stock['market_cap']:,.0f}")
        
        # 成交量筛选
        if stock['avg_volume'] < filters['min_avg_volume']:
            exclusion_reasons.append(f"成交量过低: {stock['avg_volume']:,.0f}")
        
        # 波动率筛选
        if stock['volatility'] > filters['max_volatility']:
            exclusion_reasons.append(f"波动率过高: {stock['volatility']:.1f}%")
        
        # Beta筛选
        if abs(stock['beta']) > filters['max_beta']:
            exclusion_reasons.append(f"Beta过高: {stock['beta']:.2f}")
        
        stock['exclusion_reasons'] = exclusion_reasons
        stock['tradeable'] = len(exclusion_reasons) == 0
        
        if stock['tradeable']:
            filtered_stocks.append(stock)
    
    print(f"\n筛选结果: {len(filtered_stocks)}/{len(stocks)} 只股票通过筛选")
    
    return filtered_stocks


def display_results(stocks):
    """显示结果"""
    print("\n📋 股票列表:")
    print("-" * 80)
    print(f"{'序号':>3} {'股票':>6} {'公司名称':>25} {'价格':>8} {'市值(亿)':>10} {'行业':>15}")
    print("-" * 80)
    
    for i, stock in enumerate(stocks, 1):
        status = "✅" if stock['tradeable'] else "❌"
        market_cap_b = stock['market_cap'] / 1_000_000_000  # 转换为亿
        
        print(f"{i:>3} {stock['symbol']:>6} {stock['name']:>25} "
              f"${stock['price']:>7.2f} {market_cap_b:>9.1f} {stock['sector']:>15}")
        
        if not stock['tradeable'] and stock['exclusion_reasons']:
            print(f"     ⚠️  {stock['exclusion_reasons'][0]}")
    
    print("-" * 80)
    
    # 统计信息
    tradeable_count = sum(1 for s in stocks if s['tradeable'])
    print(f"\n📊 统计信息:")
    print(f"   总股票数: {len(stocks)}")
    print(f"   可交易股票: {tradeable_count}")
    print(f"   通过率: {tradeable_count/len(stocks)*100:.1f}%")
    
    # 按行业分组
    sectors = {}
    for stock in stocks:
        if stock['tradeable']:
            sector = stock['sector']
            sectors[sector] = sectors.get(sector, 0) + 1
    
    if sectors:
        print(f"\n🏭 行业分布:")
        for sector, count in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
            print(f"   {sector}: {count}")


def save_results(stocks, filename="test_stock_results.json"):
    """保存结果到文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(stocks, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 结果已保存到 {filename}")
        
        # 同时保存可交易股票列表
        tradeable_symbols = [s['symbol'] for s in stocks if s['tradeable']]
        with open("tradeable_stocks.txt", 'w') as f:
            for symbol in tradeable_symbols:
                f.write(f"{symbol}\n")
        
        print(f"📄 可交易股票列表已保存到 tradeable_stocks.txt")
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")


def main():
    """主函数"""
    print("🚀 美股股票池快速测试")
    print("=" * 40)
    
    # 1. 测试连接
    yf_ok = test_yfinance_connection()
    nasdaq_ok = test_nasdaq_api()
    
    if not yf_ok:
        print("\n❌ yfinance连接失败，无法继续测试")
        return False
    
    # 2. 获取样本数据
    stocks = get_sample_stocks()
    
    if not stocks:
        print("\n❌ 未获取到任何股票数据")
        return False
    
    # 3. 应用筛选
    filtered_stocks = apply_quality_filters(stocks)
    
    # 4. 显示结果
    display_results(stocks)
    
    # 5. 保存结果
    save_results(stocks)
    
    print(f"\n✅ 测试完成!")
    print(f"   API连接: yfinance={'✅' if yf_ok else '❌'}, NASDAQ={'✅' if nasdaq_ok else '❌'}")
    print(f"   样本股票: {len(stocks)} 只")
    print(f"   可交易股票: {len(filtered_stocks)} 只")
    
    # 如果NASDAQ API正常，询问是否要进行完整更新
    if nasdaq_ok:
        print(f"\n🎯 NASDAQ API连接正常，可以进行完整的股票池更新")
        print(f"   运行 'python setup_stock_universe.py' 开始完整更新")
        print(f"   或运行 'run_stock_setup.bat' 使用图形界面")
    
    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
    
    input("\n按回车键退出...")