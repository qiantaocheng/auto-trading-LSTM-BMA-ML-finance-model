#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将新的宽松筛选标准应用到现有的扩展股票池
并更新到trading manager软件中
"""

import json
import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def load_existing_stock_pool():
    """读取现有的扩展股票池"""
    try:
        with open('expanded_stock_universe/stock_universe_20250806_200255.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] 读取现有股票池失败: {e}")
        return None

def apply_new_filters_to_stock(symbol):
    """对单只股票应用新的筛选标准"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info or len(info) < 3:
            return None, f"信息不足"
        
        # 获取历史数据
        hist = ticker.history(period="1mo")  # 只获取1个月数据加速
        if hist.empty or len(hist) < 5:
            return None, f"历史数据不足"
        
        # 提取价格
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        if not price:
            price = hist['Close'].iloc[-1]
        
        # 应用新的宽松筛选标准
        
        # 1. 最低股价: ≥$2.00
        if price < 2.0:
            return None, f"股价过低: ${price:.2f}"
        
        # 2. 最小市值: ≥$70M
        market_cap = info.get('marketCap', 0)
        if market_cap < 70_000_000:
            return None, f"市值过小: ${market_cap:,.0f}"
        
        # 3. 最小日均成交量: ≥10K股
        volume = hist['Volume'].tail(5).mean()
        if volume < 10_000:
            return None, f"成交量过低: {volume:,.0f}"
        
        # 4. Beta值范围: -4.0到+4.0
        beta = info.get('beta', 1.0)
        if beta is not None and (beta < -4.0 or beta > 4.0):
            return None, f"Beta超范围: {beta:.2f}"
        
        # 通过所有筛选，保留股票
        stock_data = {
            'symbol': symbol,
            'name': str(info.get('longName', info.get('shortName', symbol)))[:50],
            'sector': str(info.get('sector', 'Unknown'))[:30],
            'industry': str(info.get('industry', 'Unknown'))[:50],
            'market_cap': float(market_cap) if market_cap else 0,
            'price': float(price),
            'volume': float(volume),
            'beta': float(beta) if beta is not None else 1.0,
            'exchange': str(info.get('exchange', 'Unknown')),
            'passes_new_criteria': True
        }
        
        return stock_data, "通过"
        
    except Exception as e:
        return None, f"处理异常: {str(e)[:50]}"

def process_existing_stocks():
    """处理现有股票池的股票"""
    print("=== 对现有股票池应用新筛选标准 ===")
    
    # 读取现有股票池
    existing_pool = load_existing_stock_pool()
    if not existing_pool:
        return None
    
    all_stocks = existing_pool['all_stocks']
    print(f"[INFO] 现有股票池包含: {len(all_stocks)} 只股票")
    
    # 应用新标准筛选
    qualified_stocks = []
    failed_stocks = []
    
    print("正在应用新筛选标准...")
    
    # 使用多线程处理
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {executor.submit(apply_new_filters_to_stock, symbol): symbol 
                          for symbol in all_stocks}
        
        for i, future in enumerate(as_completed(future_to_symbol), 1):
            symbol = future_to_symbol[future]
            
            try:
                stock_data, reason = future.result(timeout=15)
                if stock_data:
                    qualified_stocks.append(stock_data)
                else:
                    failed_stocks.append((symbol, reason))
                    
            except Exception as e:
                failed_stocks.append((symbol, f"处理超时: {e}"))
            
            if i % 50 == 0:
                print(f"  进度: {i}/{len(all_stocks)}, 通过: {len(qualified_stocks)}, 失败: {len(failed_stocks)}")
    
    print(f"\n筛选完成:")
    print(f"  原始股票: {len(all_stocks)} 只")
    print(f"  通过新标准: {len(qualified_stocks)} 只")
    print(f"  筛选率: {len(qualified_stocks)/len(all_stocks)*100:.1f}%")
    
    return qualified_stocks, failed_stocks

def create_trading_manager_format(qualified_stocks):
    """创建Trading Manager格式的股票池"""
    
    # 按行业分类
    stocks_by_sector = {}
    for stock in qualified_stocks:
        sector = stock['sector']
        if sector not in stocks_by_sector:
            stocks_by_sector[sector] = []
        stocks_by_sector[sector].append(stock)
    
    # 映射到Trading Manager的类别
    trading_manager_pool = {}
    
    # 技术股
    tech_sectors = ['Technology', 'Communication Services', 'Consumer Discretionary']
    tech_stocks = []
    for sector in tech_sectors:
        if sector in stocks_by_sector:
            tech_stocks.extend([s['symbol'] for s in stocks_by_sector[sector]])
    trading_manager_pool['科技股'] = tech_stocks[:150]  # 最多150只
    
    # 金融保险
    finance_sectors = ['Financial Services', 'Financials']
    finance_stocks = []
    for sector in finance_sectors:
        if sector in stocks_by_sector:
            finance_stocks.extend([s['symbol'] for s in stocks_by_sector[sector]])
    trading_manager_pool['金融保险'] = finance_stocks[:100]
    
    # 医疗健康
    health_sectors = ['Healthcare']
    health_stocks = []
    for sector in health_sectors:
        if sector in stocks_by_sector:
            health_stocks.extend([s['symbol'] for s in stocks_by_sector[sector]])
    trading_manager_pool['医疗健康'] = health_stocks[:100]
    
    # 消费零售
    consumer_sectors = ['Consumer Cyclical', 'Consumer Staples']
    consumer_stocks = []
    for sector in consumer_sectors:
        if sector in stocks_by_sector:
            consumer_stocks.extend([s['symbol'] for s in stocks_by_sector[sector]])
    trading_manager_pool['消费零售'] = consumer_stocks[:80]
    
    # 工业制造
    industrial_sectors = ['Industrials']
    industrial_stocks = []
    for sector in industrial_sectors:
        if sector in stocks_by_sector:
            industrial_stocks.extend([s['symbol'] for s in stocks_by_sector[sector]])
    trading_manager_pool['工业制造'] = industrial_stocks[:80]
    
    # 能源化工
    energy_sectors = ['Energy']
    energy_stocks = []
    for sector in energy_sectors:
        if sector in stocks_by_sector:
            energy_stocks.extend([s['symbol'] for s in stocks_by_sector[sector]])
    trading_manager_pool['能源化工'] = energy_stocks[:60]
    
    # 基础材料
    materials_sectors = ['Basic Materials']
    materials_stocks = []
    for sector in materials_sectors:
        if sector in stocks_by_sector:
            materials_stocks.extend([s['symbol'] for s in stocks_by_sector[sector]])
    trading_manager_pool['基础材料'] = materials_stocks[:50]
    
    # 公用事业
    utilities_sectors = ['Utilities']
    utilities_stocks = []
    for sector in utilities_sectors:
        if sector in stocks_by_sector:
            utilities_stocks.extend([s['symbol'] for s in stocks_by_sector[sector]])
    trading_manager_pool['公用事业'] = utilities_stocks[:40]
    
    # 房地产
    real_estate_sectors = ['Real Estate']
    real_estate_stocks = []
    for sector in real_estate_sectors:
        if sector in stocks_by_sector:
            real_estate_stocks.extend([s['symbol'] for s in stocks_by_sector[sector]])
    trading_manager_pool['房地产'] = real_estate_stocks[:40]
    
    # 其他股票
    used_symbols = set()
    for stocks in trading_manager_pool.values():
        used_symbols.update(stocks)
    
    other_stocks = [s['symbol'] for s in qualified_stocks if s['symbol'] not in used_symbols]
    if other_stocks:
        trading_manager_pool['其他股票'] = other_stocks[:50]
    
    return trading_manager_pool

def update_trading_manager(trading_manager_pool):
    """更新Trading Manager配置"""
    
    # 备份现有配置
    backup_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    if os.path.exists('default_stocks.json'):
        os.rename('default_stocks.json', f'default_stocks_backup_relaxed_{backup_time}.json')
        print(f"[BACKUP] 已备份现有配置")
    
    # 保存新配置
    with open('default_stocks.json', 'w', encoding='utf-8') as f:
        json.dump(trading_manager_pool, f, ensure_ascii=False, indent=2)
    
    total_stocks = sum(len(stocks) for stocks in trading_manager_pool.values())
    print(f"[OK] 已更新Trading Manager配置: {total_stocks} 只股票")
    
    for category, stocks in trading_manager_pool.items():
        print(f"  - {category}: {len(stocks)} 只")
    
    return total_stocks

def update_model_configs(qualified_stocks):
    """更新模型配置文件"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 为BMA和LSTM创建股票列表 (使用所有通过新标准的股票)
    all_symbols = [s['symbol'] for s in qualified_stocks]
    
    # BMA配置 (使用所有股票)
    bma_file = f'bma_training_stocks_relaxed_{timestamp}.txt'
    with open(bma_file, 'w', encoding='utf-8') as f:
        f.write(f"# BMA训练股票列表 (新宽松标准)\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 筛选标准: 股价≥$2.00, 市值≥$70M, 成交量≥10K, -4≤Beta≤4\n")
        f.write(f"# 股票数量: {len(all_symbols)} 只\n\n")
        
        for symbol in all_symbols:
            f.write(f"{symbol}\n")
    
    # LSTM配置 (使用所有股票)
    lstm_file = f'lstm_training_stocks_relaxed_{timestamp}.txt'
    with open(lstm_file, 'w', encoding='utf-8') as f:
        f.write(f"# LSTM训练股票列表 (新宽松标准)\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 筛选标准: 股价≥$2.00, 市值≥$70M, 成交量≥10K, -4≤Beta≤4\n")
        f.write(f"# 股票数量: {len(all_symbols)} 只\n\n")
        
        for symbol in all_symbols:
            f.write(f"{symbol}\n")
    
    # 更新BMA配置
    bma_config = {
        "model_name": "BMA_Quantitative_Analysis_Relaxed",
        "version": "3.0_relaxed",
        "updated_at": datetime.now().isoformat(),
        "stock_pool": {
            "source": "relaxed_criteria",
            "symbols": all_symbols,
            "count": len(all_symbols),
            "description": "使用新宽松标准的所有股票"
        },
        "analysis_parameters": {
            "min_price": 2.0,
            "min_market_cap": 70_000_000,
            "min_volume": 10_000,
            "beta_range": [-4.0, 4.0],
            "max_stocks_per_analysis": len(all_symbols),
            "confidence_threshold": 0.6,
            "lookback_days": 252
        }
    }
    
    with open(f'bma_stock_config_relaxed_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(bma_config, f, indent=2, ensure_ascii=False)
    
    # 更新LSTM配置
    lstm_config = {
        "model_name": "LSTM_Multi_Day_Analysis_Relaxed",
        "version": "3.0_relaxed",
        "updated_at": datetime.now().isoformat(),
        "training_data": {
            "source_file": lstm_file,
            "total_stocks": len(all_symbols),
            "criteria": "relaxed_standards",
            "symbols": all_symbols
        },
        "model_parameters": {
            "sequence_length": 60,
            "prediction_days": 5,
            "batch_size": 32,
            "epochs": 100,
            "validation_split": 0.2
        }
    }
    
    with open(f'lstm_stock_config_relaxed_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(lstm_config, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] 已更新模型配置:")
    print(f"  - BMA: {bma_file} ({len(all_symbols)} 只股票)")
    print(f"  - LSTM: {lstm_file} ({len(all_symbols)} 只股票)")
    
    return bma_file, lstm_file

def main():
    """主函数"""
    print("=" * 70)
    print("应用新宽松筛选标准到现有股票池")
    print("=" * 70)
    print("新筛选标准:")
    print("- 最低股价: ≥$2.00")
    print("- 最小市值: ≥$70M") 
    print("- 最小日均成交量: ≥10K股")
    print("- Beta值范围: -4.0到+4.0")
    print("- 删除波动率限制")
    print("- 删除质量分层")
    print("=" * 70)
    
    try:
        # 1. 处理现有股票
        qualified_stocks, failed_stocks = process_existing_stocks()
        if not qualified_stocks:
            print("[ERROR] 没有股票通过新标准")
            return False
        
        # 2. 创建Trading Manager格式
        trading_manager_pool = create_trading_manager_format(qualified_stocks)
        
        # 3. 更新Trading Manager
        total_stocks = update_trading_manager(trading_manager_pool)
        
        # 4. 更新模型配置
        bma_file, lstm_file = update_model_configs(qualified_stocks)
        
        # 5. 显示总结
        print("\n" + "=" * 70)
        print("新宽松标准应用完成!")
        print("=" * 70)
        print(f"通过新标准: {len(qualified_stocks)} 只股票")
        print(f"Trading Manager: {total_stocks} 只股票")
        print(f"BMA模型: {len(qualified_stocks)} 只股票")
        print(f"LSTM模型: {len(qualified_stocks)} 只股票")
        
        print("\n失败股票样本:")
        for symbol, reason in failed_stocks[:10]:
            print(f"  {symbol}: {reason}")
        
        if len(failed_stocks) > 10:
            print(f"  ... 还有 {len(failed_stocks) - 10} 只股票未通过")
        
        print("\n立即可用:")
        print("1. python quantitative_trading_manager.py  # 启动主软件")
        print(f"2. python 量化模型_bma_enhanced.py --stock-file {bma_file}")
        print(f"3. python lstm_multi_day_enhanced.py --stock-file {lstm_file}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 应用新标准失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 新宽松标准应用成功! 获得更多股票用于模型训练。")
    else:
        print("\n❌ 应用失败!")