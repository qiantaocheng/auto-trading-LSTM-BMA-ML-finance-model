#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票池更新脚本
从爬虫获取的数据更新LSTM/BMA/Trading Manager的股票池

Author: AI Assistant
Version: 1.0
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict

def load_crawler_stocks() -> List[str]:
    """从爬虫缓存中加载股票列表"""
    cache_file = Path("stock_cache/filtered_us_stocks.json")
    
    if not cache_file.exists():
        print("❌ 未找到爬虫数据文件，请先运行us_stock_crawler.py")
        return []
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            stocks = data.get('tickers', [])
            print(f"✅ 从爬虫缓存加载了 {len(stocks)} 只筛选后的股票")
            return stocks
    except Exception as e:
        print(f"❌ 加载爬虫数据失败: {e}")
        return []

def categorize_stocks(stocks: List[str]) -> Dict[str, List[str]]:
    """将股票按行业分类（基于股票代码模式）"""
    categories = {
        '大型科技股': [],
        '金融服务': [],
        '医疗保健': [],
        '工业制造': [],
        '消费品牌': [],
        '能源公用': [],
        '通信媒体': [],
        '其他优质股': []
    }
    
    # 基于股票代码的简单分类（可以后续改进为基于实际行业数据）
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 
                   'ADBE', 'CRM', 'ORCL', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'MU', 
                   'AMAT', 'LRCX', 'KLAC', 'MRVL', 'SNPS', 'CDNS', 'ADSK', 'ANET', 'NOW']
    
    finance_stocks = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'COF', 'SCHW', 
                      'AXP', 'BLK', 'SPGI', 'MCO', 'AON', 'MMC', 'AJG', 'CB', 'PGR', 'TRV']
    
    healthcare_stocks = ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'DHR', 'MRK', 'ABBV', 'AMGN', 
                         'SYK', 'BSX', 'MDT', 'ISRG', 'DXCM', 'ZTS', 'ELV', 'CVS', 'CI', 'HUM']
    
    industrial_stocks = ['GE', 'CAT', 'BA', 'MMM', 'HON', 'UPS', 'LMT', 'RTX', 'DE', 'EMR',
                         'ETN', 'PH', 'ITW', 'CSX', 'UNP', 'NSC', 'FDX', 'WM', 'RSG']
    
    consumer_stocks = ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'COST', 'LOW', 'SBUX', 'TGT',
                       'NKE', 'LULU', 'EL', 'CL', 'KMB', 'GIS', 'K', 'SJM', 'CPB']
    
    energy_utility_stocks = ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY',
                             'NEE', 'DUK', 'SO', 'D', 'EXC', 'SRE', 'AEP', 'ES']
    
    comm_media_stocks = ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 
                         'CHTR', 'EA', 'ATVI', 'TTWO']
    
    for stock in stocks:
        if stock in tech_stocks:
            categories['大型科技股'].append(stock)
        elif stock in finance_stocks:
            categories['金融服务'].append(stock)
        elif stock in healthcare_stocks:
            categories['医疗保健'].append(stock)
        elif stock in industrial_stocks:
            categories['工业制造'].append(stock)
        elif stock in consumer_stocks:
            categories['消费品牌'].append(stock)
        elif stock in energy_utility_stocks:
            categories['能源公用'].append(stock)
        elif stock in comm_media_stocks:
            categories['通信媒体'].append(stock)
        else:
            categories['其他优质股'].append(stock)
    
    # 移除空分类
    categories = {k: v for k, v in categories.items() if v}
    
    return categories

def update_trading_manager_stock_pool(stocks: List[str]) -> bool:
    """更新Trading Manager的股票池"""
    print("\n=== 更新Trading Manager股票池 ===")
    
    try:
        # 按行业分类股票
        categorized_stocks = categorize_stocks(stocks)
        
        # 保存为JSON格式（Trading Manager读取的格式）
        pool_file = "default_stocks.json"
        
        with open(pool_file, 'w', encoding='utf-8') as f:
            json.dump(categorized_stocks, f, indent=2, ensure_ascii=False)
        
        total_stocks = sum(len(stocks) for stocks in categorized_stocks.values())
        print(f"✅ Trading Manager股票池更新完成")
        print(f"   - 总股票数: {total_stocks}")
        print(f"   - 分类数: {len(categorized_stocks)}")
        print(f"   - 保存文件: {pool_file}")
        
        for category, stocks in categorized_stocks.items():
            print(f"   - {category}: {len(stocks)}只")
        
        return True
        
    except Exception as e:
        print(f"❌ 更新Trading Manager股票池失败: {e}")
        return False

def update_lstm_model_stock_pool(stocks: List[str]) -> bool:
    """更新LSTM模型的股票池"""
    print("\n=== 更新LSTM模型股票池 ===")
    
    try:
        # 读取LSTM文件
        lstm_file = "lstm_multi_day_enhanced.py"
        
        if not os.path.exists(lstm_file):
            print(f"❌ LSTM文件不存在: {lstm_file}")
            return False
        
        with open(lstm_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 找到默认股票列表的定义
        # 查找ticker_list或类似的股票列表定义
        pattern = r'(ticker_list\s*=\s*\[)(.*?)(\])'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            # 构建新的股票列表
            stock_list_str = "[\n"
            for i, stock in enumerate(stocks[:200]):  # 限制为前200只避免过多
                stock_list_str += f"    '{stock}'"
                if i < min(len(stocks), 200) - 1:
                    stock_list_str += ","
                stock_list_str += "\n"
            stock_list_str += "]"
            
            # 替换内容
            new_content = content[:match.start(1)] + "ticker_list = " + stock_list_str + content[match.end(3):]
            
            # 备份原文件
            backup_file = f"{lstm_file}.backup"
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 写入新内容
            with open(lstm_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✅ LSTM模型股票池更新完成")
            print(f"   - 更新股票数: {min(len(stocks), 200)}")
            print(f"   - 备份文件: {backup_file}")
            return True
        else:
            print("⚠️ 未找到LSTM模型中的ticker_list定义")
            return False
            
    except Exception as e:
        print(f"❌ 更新LSTM模型股票池失败: {e}")
        return False

def update_bma_model_stock_pool(stocks: List[str]) -> bool:
    """更新BMA模型的股票池"""
    print("\n=== 更新BMA模型股票池 ===")
    
    try:
        # 读取BMA文件
        bma_file = "量化模型_bma_enhanced.py"
        
        if not os.path.exists(bma_file):
            print(f"❌ BMA文件不存在: {bma_file}")
            return False
        
        with open(bma_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 找到默认股票列表的定义
        pattern = r'(ticker_list\s*=\s*\[)(.*?)(\])'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            # 构建新的股票列表
            stock_list_str = "[\n"
            for i, stock in enumerate(stocks[:300]):  # BMA可以处理更多股票
                stock_list_str += f"    '{stock}'"
                if i < min(len(stocks), 300) - 1:
                    stock_list_str += ","
                stock_list_str += "\n"
            stock_list_str += "]"
            
            # 替换内容
            new_content = content[:match.start(1)] + "ticker_list = " + stock_list_str + content[match.end(3):]
            
            # 备份原文件
            backup_file = f"{bma_file}.backup"
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 写入新内容
            with open(bma_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✅ BMA模型股票池更新完成")
            print(f"   - 更新股票数: {min(len(stocks), 300)}")
            print(f"   - 备份文件: {backup_file}")
            return True
        else:
            print("⚠️ 未找到BMA模型中的ticker_list定义")
            return False
            
    except Exception as e:
        print(f"❌ 更新BMA模型股票池失败: {e}")
        return False

def main():
    """主函数"""
    print("[START] 开始更新所有模型的股票池...")
    
    # 1. 加载爬虫获取的股票数据
    stocks = load_crawler_stocks()
    if not stocks:
        print("[ERROR] 无法获取爬虫数据，退出")
        return
    
    print(f"[INFO] 准备更新 {len(stocks)} 只股票到各个模型")
    
    # 2. 更新Trading Manager
    success_tm = update_trading_manager_stock_pool(stocks)
    
    # 3. 更新LSTM模型
    success_lstm = update_lstm_model_stock_pool(stocks)
    
    # 4. 更新BMA模型  
    success_bma = update_bma_model_stock_pool(stocks)
    
    # 5. 总结
    print("\n" + "="*50)
    print("[SUMMARY] 更新结果总结:")
    print(f"   Trading Manager: {'[OK] 成功' if success_tm else '[FAIL] 失败'}")
    print(f"   LSTM模型: {'[OK] 成功' if success_lstm else '[FAIL] 失败'}")
    print(f"   BMA模型: {'[OK] 成功' if success_bma else '[FAIL] 失败'}")
    
    successful_updates = sum([success_tm, success_lstm, success_bma])
    print(f"\n[RESULT] 总体结果: {successful_updates}/3 个模型更新成功")
    
    if successful_updates == 3:
        print("\n[SUCCESS] 所有模型股票池更新完成！可以开始训练了。")
        print("[NEXT] 建议下一步:")
        print("   1. 运行LSTM模型: python lstm_multi_day_enhanced.py")
        print("   2. 运行BMA模型: python 量化模型_bma_enhanced.py")
        print("   3. 在Trading Manager中查看更新的股票池")
    else:
        print(f"\n[WARNING] 部分模型更新失败，请检查错误信息")

if __name__ == "__main__":
    main()