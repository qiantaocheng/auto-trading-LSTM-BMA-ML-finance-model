#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test Direct Predict functionality"""

import sys
import os
from pathlib import Path
import pandas as pd
from pandas.tseries.offsets import BDay

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_load_default_tickers():
    """Test loading default tickers from data file"""
    print("=" * 80)
    print("Test 1: Load Default Tickers from Data File")
    print("=" * 80)
    
    default_tickers_file = Path(r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean.parquet")
    
    if not default_tickers_file.exists():
        print(f"[ERROR] 数据文件不存在: {default_tickers_file}")
        return None
    
    try:
        print(f"[INFO] 加载数据文件: {default_tickers_file}")
        df_tickers = pd.read_parquet(default_tickers_file)
        
        if isinstance(df_tickers.index, pd.MultiIndex):
            default_tickers = sorted(df_tickers.index.get_level_values('ticker').unique().tolist())
        elif 'ticker' in df_tickers.columns:
            default_tickers = sorted(df_tickers['ticker'].unique().tolist())
        else:
            print("❌ 无法找到 ticker 列或 MultiIndex")
            return None
        
        print(f"[OK] 成功加载 {len(default_tickers)} 只股票")
        print(f"[INFO] 数据形状: {df_tickers.shape}")
        print(f"[INFO] 日期范围:")
        if isinstance(df_tickers.index, pd.MultiIndex):
            dates = df_tickers.index.get_level_values('date').unique()
            print(f"   从 {pd.Timestamp(min(dates)).strftime('%Y-%m-%d')} 至 {pd.Timestamp(max(dates)).strftime('%Y-%m-%d')}")
        print(f"[INFO] 前10只股票: {', '.join(default_tickers[:10])}")
        
        return default_tickers
        
    except Exception as e:
        print(f"[ERROR] 加载失败: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def test_date_calculation():
    """Test date calculation using previous trading day"""
    print("\n" + "=" * 80)
    print("Test 2: Date Calculation (Previous Trading Day)")
    print("=" * 80)
    
    try:
        today = pd.Timestamp.today()
        end_date = today - BDay(1)  # Previous trading day
        
        MIN_REQUIRED_LOOKBACK_DAYS = 280
        prediction_days = 1
        total_lookback_days = MIN_REQUIRED_LOOKBACK_DAYS + prediction_days
        start_date = end_date - pd.Timedelta(days=total_lookback_days + 30)
        
        print(f"[INFO] 今天: {today.strftime('%Y-%m-%d')} ({today.strftime('%A')})")
        print(f"[INFO] 上一个交易日: {end_date.strftime('%Y-%m-%d')} ({end_date.strftime('%A')})")
        print(f"[INFO] 开始日期: {start_date.strftime('%Y-%m-%d')}")
        print(f"[INFO] 数据范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
        print(f"[INFO] 历史数据天数: {MIN_REQUIRED_LOOKBACK_DAYS} 天")
        print(f"[INFO] 预测天数: {prediction_days} 天")
        print(f"[INFO] 总回看天数: {total_lookback_days} 天")
        
        # Test prediction date calculation
        print(f"\n[INFO] 预测日期计算:")
        for day_offset in range(prediction_days - 1, -1, -1):
            if day_offset == 0:
                pred_date = end_date
            else:
                pred_date = end_date - BDay(day_offset)
            print(f"   day_offset={day_offset}: {pred_date.strftime('%Y-%m-%d')} ({pred_date.strftime('%A')})")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 日期计算失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def test_ticker_priority():
    """Test ticker priority logic"""
    print("\n" + "=" * 80)
    print("Test 3: Ticker Priority Logic")
    print("=" * 80)
    
    # Load default tickers
    default_tickers = test_load_default_tickers()
    
    if not default_tickers:
        print("[ERROR] 无法加载默认股票列表，跳过优先级测试")
        return False
    
    # Test priority 1: Pool selection
    print(f"\n[TEST] 优先级1: 股票池选择")
    selected_pool_tickers = ['AAPL', 'MSFT', 'GOOGL']
    tickers = list(set([t.strip().upper() for t in selected_pool_tickers if isinstance(t, str) and t.strip()]))
    print(f"   股票池: {tickers}")
    print(f"   [OK] 使用股票池中的 {len(tickers)} 只股票")
    
    # Test priority 2: Default from file
    print(f"\n[TEST] 优先级2: 数据文件默认值")
    tickers = []
    if not tickers and default_tickers:
        tickers = default_tickers.copy()
        print(f"   [OK] 使用数据文件中的默认股票列表: {len(tickers)} 只股票")
    
    # Test priority 3: User input (simulated)
    print(f"\n[TEST] 优先级3: 用户输入")
    tickers = []
    user_input = "AAPL,MSFT,TSLA"
    if not tickers:
        tickers = list({s.strip().upper() for s in user_input.split(',') if s.strip()})
        print(f"   用户输入: {user_input}")
        print(f"   [OK] 使用用户输入的 {len(tickers)} 只股票: {tickers}")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("Direct Predict 功能测试")
    print("=" * 80)
    
    # Test 1: Load default tickers
    default_tickers = test_load_default_tickers()
    
    # Test 2: Date calculation
    date_test_passed = test_date_calculation()
    
    # Test 3: Ticker priority
    priority_test_passed = test_ticker_priority()
    
    # Summary
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"[RESULT] 默认股票加载: {'通过' if default_tickers else '失败'}")
    print(f"[RESULT] 日期计算: {'通过' if date_test_passed else '失败'}")
    print(f"[RESULT] 优先级逻辑: {'通过' if priority_test_passed else '失败'}")
    
    if default_tickers and date_test_passed and priority_test_passed:
        print("\n[SUCCESS] 所有测试通过！Direct Predict 功能已就绪。")
    else:
        print("\n[WARNING] 部分测试失败，请检查错误信息。")


if __name__ == "__main__":
    main()
