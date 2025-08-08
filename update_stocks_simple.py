#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版股票池更新脚本
"""

import json
import os
import re
from pathlib import Path

def load_crawler_stocks():
    """从爬虫缓存中加载股票列表"""
    cache_file = Path("stock_cache/filtered_us_stocks.json")
    
    if not cache_file.exists():
        print("[ERROR] 未找到爬虫数据文件")
        return []
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            stocks = data.get('tickers', [])
            print(f"[OK] 加载了 {len(stocks)} 只股票")
            return stocks
    except Exception as e:
        print(f"[ERROR] 加载数据失败: {e}")
        return []

def categorize_stocks(stocks):
    """将股票分类"""
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 
                   'ADBE', 'CRM', 'ORCL', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'MU', 
                   'AMAT', 'LRCX', 'KLAC', 'MRVL', 'SNPS', 'CDNS', 'ADSK', 'ANET', 'NOW']
    
    finance_stocks = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'COF', 'SCHW', 
                      'AXP', 'BLK', 'SPGI', 'MCO', 'AON', 'MMC', 'AJG', 'CB', 'PGR', 'TRV']
    
    healthcare_stocks = ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'DHR', 'MRK', 'ABBV', 'AMGN', 
                         'SYK', 'BSX', 'MDT', 'ISRG', 'DXCM', 'ZTS', 'ELV', 'CVS', 'CI', 'HUM']
    
    categories = {
        '科技股': [],
        '金融股': [], 
        '医疗保健': [],
        '其他股票': []
    }
    
    for stock in stocks:
        if stock in tech_stocks:
            categories['科技股'].append(stock)
        elif stock in finance_stocks:
            categories['金融股'].append(stock)
        elif stock in healthcare_stocks:
            categories['医疗保健'].append(stock)
        else:
            categories['其他股票'].append(stock)
    
    # 移除空分类
    categories = {k: v for k, v in categories.items() if v}
    return categories

def update_trading_manager():
    """更新Trading Manager"""
    print("\n[UPDATE] Trading Manager...")
    
    stocks = load_crawler_stocks()
    if not stocks:
        return False
    
    try:
        categorized_stocks = categorize_stocks(stocks)
        
        with open("default_stocks.json", 'w', encoding='utf-8') as f:
            json.dump(categorized_stocks, f, indent=2, ensure_ascii=False)
        
        total = sum(len(v) for v in categorized_stocks.values())
        print(f"[OK] Trading Manager 更新完成: {total} 只股票, {len(categorized_stocks)} 个分类")
        return True
        
    except Exception as e:
        print(f"[ERROR] Trading Manager 更新失败: {e}")
        return False

def update_lstm_model():
    """更新LSTM模型"""
    print("\n[UPDATE] LSTM模型...")
    
    stocks = load_crawler_stocks()
    if not stocks:
        return False
    
    try:
        lstm_file = "lstm_multi_day_enhanced.py"
        if not os.path.exists(lstm_file):
            print(f"[ERROR] LSTM文件不存在: {lstm_file}")
            return False
        
        with open(lstm_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 找到ticker_list定义
        pattern = r'(ticker_list\s*=\s*\[)(.*?)(\])'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            # 构建新股票列表 - 限制200只
            selected_stocks = stocks[:200]
            stock_list_str = "[\n"
            for i, stock in enumerate(selected_stocks):
                stock_list_str += f"    '{stock}'"
                if i < len(selected_stocks) - 1:
                    stock_list_str += ","
                stock_list_str += "\n"
            stock_list_str += "]"
            
            # 替换内容
            new_content = content[:match.start(1)] + "ticker_list = " + stock_list_str + content[match.end(3):]
            
            # 备份
            with open(f"{lstm_file}.backup", 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 写入新内容
            with open(lstm_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"[OK] LSTM模型 更新完成: {len(selected_stocks)} 只股票")
            return True
        else:
            print("[WARNING] 未找到LSTM中的ticker_list")
            return False
            
    except Exception as e:
        print(f"[ERROR] LSTM模型 更新失败: {e}")
        return False

def update_bma_model():
    """更新BMA模型"""
    print("\n[UPDATE] BMA模型...")
    
    stocks = load_crawler_stocks()
    if not stocks:
        return False
    
    try:
        bma_file = "量化模型_bma_enhanced.py"
        if not os.path.exists(bma_file):
            print(f"[ERROR] BMA文件不存在: {bma_file}")
            return False
        
        with open(bma_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 找到ticker_list定义
        pattern = r'(ticker_list\s*=\s*\[)(.*?)(\])'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            # BMA可以处理更多股票 - 限制300只
            selected_stocks = stocks[:300]
            stock_list_str = "[\n"
            for i, stock in enumerate(selected_stocks):
                stock_list_str += f"    '{stock}'"
                if i < len(selected_stocks) - 1:
                    stock_list_str += ","
                stock_list_str += "\n"
            stock_list_str += "]"
            
            # 替换内容
            new_content = content[:match.start(1)] + "ticker_list = " + stock_list_str + content[match.end(3):]
            
            # 备份
            with open(f"{bma_file}.backup", 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 写入新内容
            with open(bma_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"[OK] BMA模型 更新完成: {len(selected_stocks)} 只股票")
            return True
        else:
            print("[WARNING] 未找到BMA中的ticker_list")
            return False
            
    except Exception as e:
        print(f"[ERROR] BMA模型 更新失败: {e}")
        return False

def main():
    print("[START] 开始更新股票池...")
    
    success_tm = update_trading_manager()
    success_lstm = update_lstm_model()
    success_bma = update_bma_model()
    
    print("\n" + "="*40)
    print("[SUMMARY] 更新结果:")
    print(f"  Trading Manager: {'成功' if success_tm else '失败'}")
    print(f"  LSTM模型: {'成功' if success_lstm else '失败'}")
    print(f"  BMA模型: {'成功' if success_bma else '失败'}")
    
    successful = sum([success_tm, success_lstm, success_bma])
    print(f"\n[RESULT] {successful}/3 个模型更新成功")
    
    if successful == 3:
        print("\n[SUCCESS] 所有股票池更新完成！")
        print("[NEXT] 现在可以运行模型训练:")
        print("  1. LSTM: python lstm_multi_day_enhanced.py")
        print("  2. BMA: python 量化模型_bma_enhanced.py")

if __name__ == "__main__":
    main()