#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查数据泄露问题：分析模型是否在大涨之前就预测到了异常收益
验证异常值是否是真实的市场表现
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def check_data_leakage_and_timing():
    """检查数据泄露和时机问题"""
    print("=" * 80)
    print("数据泄露和时机分析 - 验证异常值是否真实及模型预测时机")
    print("=" * 80)
    
    data_file = "data/factor_exports/polygon_factors_all_filtered_clean_final_v2.parquet"
    
    print(f"\n[1] 加载数据文件...")
    df = pd.read_parquet(data_file)
    print(f"   [OK] 数据形状: {df.shape}")
    
    # 提取测试期数据（2025年）
    test_start = pd.Timestamp('2025-01-02')
    test_end = pd.Timestamp('2025-12-30')
    
    test_dates = df.index.get_level_values('date')
    test_mask = (test_dates >= test_start) & (test_dates <= test_end)
    test_df = df[test_mask].copy()
    
    print(f"\n[2] 测试期数据统计:")
    print(f"   测试期样本数: {len(test_df):,}")
    
    if 'target' not in test_df.columns or 'Close' not in test_df.columns:
        print(f"\n[ERROR] 缺少必要的列")
        return
    
    # 找出异常target值（>50%）
    extreme_df = test_df[test_df['target'] > 0.5].copy()
    ticker_counts = extreme_df.index.get_level_values('ticker').value_counts()
    
    print(f"\n[3] 分析Top 10异常值ticker的预测时机和实际表现:")
    print("=" * 80)
    
    # 检查每个异常值ticker的价格走势
    for ticker, count in ticker_counts.head(10).items():
        print(f"\n[Ticker: {ticker}] 异常值样本数: {count}")
        print("-" * 80)
        
        ticker_data = extreme_df.loc[extreme_df.index.get_level_values('ticker') == ticker].copy()
        ticker_data = ticker_data.sort_index(level='date')
        
        # 获取该ticker的所有数据（包括价格走势）
        ticker_all_data = test_df.loc[test_df.index.get_level_values('ticker') == ticker].copy()
        ticker_all_data = ticker_all_data.sort_index(level='date')
        
        if len(ticker_all_data) == 0:
            print(f"   [WARN] 没有找到该ticker的完整数据")
            continue
        
        # 分析前3个异常值样本
        sample_count = min(3, len(ticker_data))
        
        for idx in range(sample_count):
            row = ticker_data.iloc[idx]
            date = row.name[0]  # 预测日期
            target_val = row['target']
            close_t = row['Close']
            
            # 找到t+10的价格
            date_idx = ticker_all_data.index.get_level_values('date').get_loc(date)
            
            if date_idx + 10 < len(ticker_all_data):
                close_t10_row = ticker_all_data.iloc[date_idx + 10]
                close_t10 = close_t10_row['Close']
                date_t10 = close_t10_row.name[0]
                
                # 手动计算收益率
                if close_t > 0 and not pd.isna(close_t10) and close_t10 > 0:
                    manual_return = (close_t10 - close_t) / close_t
                    
                    # 检查预测日期之前的价格走势（检查是否有数据泄露）
                    # 查看预测日期前5天、10天、20天的价格
                    lookback_periods = [5, 10, 20]
                    price_before = {}
                    
                    for lookback in lookback_periods:
                        if date_idx >= lookback:
                            price_before[lookback] = ticker_all_data.iloc[date_idx - lookback]['Close']
                        else:
                            price_before[lookback] = None
                    
                    # 计算预测日期之前的收益率（如果模型使用了未来信息，这些值会异常高）
                    returns_before = {}
                    for lookback in lookback_periods:
                        if price_before[lookback] is not None and price_before[lookback] > 0:
                            returns_before[lookback] = (close_t - price_before[lookback]) / price_before[lookback]
                        else:
                            returns_before[lookback] = None
                    
                    # 检查预测日期之后的价格走势（验证是否真的涨了）
                    # 查看预测日期后5天、10天、20天的价格
                    price_after = {}
                    for lookforward in [5, 10, 20]:
                        if date_idx + lookforward < len(ticker_all_data):
                            price_after[lookforward] = ticker_all_data.iloc[date_idx + lookforward]['Close']
                        else:
                            price_after[lookforward] = None
                    
                    # 计算预测日期之后的收益率
                    returns_after = {}
                    for lookforward in [5, 10, 20]:
                        if price_after[lookforward] is not None and close_t > 0:
                            returns_after[lookforward] = (price_after[lookforward] - close_t) / close_t
                        else:
                            returns_after[lookforward] = None
                    
                    print(f"\n  样本 {idx+1}: 预测日期={date.date()}, 目标日期={date_t10.date()}")
                    print(f"    Close[t] = ${close_t:.2f}, Close[t+10] = ${close_t10:.2f}")
                    print(f"    Target = {target_val:.4f} ({target_val*100:.2f}%)")
                    print(f"    手动计算 = {manual_return:.4f} ({manual_return*100:.2f}%)")
                    
                    # 分析预测时机
                    print(f"\n    [预测时机分析]")
                    print(f"      预测日期之前的价格走势:")
                    for lookback in lookback_periods:
                        if price_before[lookback] is not None:
                            ret = returns_before[lookback]
                            print(f"        T-{lookback}天: ${price_before[lookback]:.2f} (从T-{lookback}到T的收益={ret:.4f} ({ret*100:.2f}%))")
                        else:
                            print(f"        T-{lookback}天: 无数据")
                    
                    print(f"\n      预测日期之后的价格走势:")
                    for lookforward in [5, 10, 20]:
                        if price_after[lookforward] is not None:
                            ret = returns_after[lookforward]
                            print(f"        T+{lookforward}天: ${price_after[lookforward]:.2f} (从T到T+{lookforward}的收益={ret:.4f} ({ret*100:.2f}%))")
                        else:
                            print(f"        T+{lookforward}天: 无数据")
                    
                    # 检查是否有数据泄露
                    has_leakage = False
                    leakage_reasons = []
                    
                    # 检查1: 预测日期之前是否已经大涨
                    for lookback in lookback_periods:
                        if returns_before[lookback] is not None and returns_before[lookback] > 0.5:  # >50%
                            has_leakage = True
                            leakage_reasons.append(f"T-{lookback}天已经大涨{returns_before[lookback]*100:.2f}%")
                    
                    # 检查2: 预测日期之后是否真的涨了（验证预测是否正确）
                    if returns_after[10] is not None and returns_after[10] > 0.5:  # T+10收益>50%
                        print(f"\n    [验证] T+10确实大涨{returns_after[10]*100:.2f}%，预测可能是正确的")
                    elif returns_after[10] is not None:
                        print(f"\n    [WARN] T+10收益只有{returns_after[10]*100:.2f}%，但target={target_val*100:.2f}%，可能有问题")
                    
                    # 检查3: 价格是否在预测日期附近有异常跳变
                    if date_idx > 0 and date_idx < len(ticker_all_data) - 1:
                        price_change_today = (close_t - ticker_all_data.iloc[date_idx - 1]['Close']) / ticker_all_data.iloc[date_idx - 1]['Close']
                        if abs(price_change_today) > 0.5:  # 当日涨跌>50%
                            print(f"\n    [WARN] 预测日期当日价格跳变{price_change_today*100:.2f}%")
                    
                    if has_leakage:
                        print(f"\n    [数据泄露警告] 预测日期之前已经大涨:")
                        for reason in leakage_reasons:
                            print(f"      - {reason}")
                        print(f"    这可能表明模型使用了未来信息（look-ahead bias）")
                    else:
                        print(f"\n    [OK] 预测日期之前价格正常，没有明显的数据泄露")
                    
                    # 检查是否是真实的市场表现
                    if manual_return > 1.0:  # >100%
                        print(f"\n    [市场表现] 10天内收益{manual_return*100:.2f}%，这是真实的市场表现")
                        print(f"    但10天内>100%的收益非常罕见，可能是:")
                        print(f"      1. 重大新闻/事件驱动")
                        print(f"      2. 股票拆分/合并")
                        print(f"      3. 数据质量问题")
                    elif manual_return > 0.5:  # >50%
                        print(f"\n    [市场表现] 10天内收益{manual_return*100:.2f}%，这是真实的市场表现")
                        print(f"    10天内50-100%的收益虽然罕见，但在小盘股/特殊情况下可能发生")
    
    # 汇总分析
    print(f"\n" + "=" * 80)
    print("[4] 数据泄露和时机问题汇总分析")
    print("=" * 80)
    
    # 统计有多少异常值是在大涨之前预测的
    leakage_count = 0
    valid_prediction_count = 0
    total_checked = 0
    
    for ticker, count in ticker_counts.head(10).items():
        ticker_data = extreme_df.loc[extreme_df.index.get_level_values('ticker') == ticker].copy()
        ticker_data = ticker_data.sort_index(level='date')
        
        ticker_all_data = test_df.loc[test_df.index.get_level_values('ticker') == ticker].copy()
        ticker_all_data = ticker_all_data.sort_index(level='date')
        
        if len(ticker_all_data) == 0:
            continue
        
        sample_count = min(3, len(ticker_data))
        
        for idx in range(sample_count):
            row = ticker_data.iloc[idx]
            date = row.name[0]
            date_idx = ticker_all_data.index.get_level_values('date').get_loc(date)
            
            if date_idx >= 10:  # 有足够的历史数据
                close_t = row['Close']
                # 检查T-10的收益
                price_t_minus_10 = ticker_all_data.iloc[date_idx - 10]['Close']
                if price_t_minus_10 > 0:
                    ret_before = (close_t - price_t_minus_10) / price_t_minus_10
                    total_checked += 1
                    
                    if ret_before > 0.5:  # T-10已经大涨>50%
                        leakage_count += 1
                    else:
                        valid_prediction_count += 1
    
    if total_checked > 0:
        print(f"\n检查的异常值样本数: {total_checked}")
        print(f"预测日期之前已经大涨（可能数据泄露）: {leakage_count} ({leakage_count/total_checked*100:.1f}%)")
        print(f"预测日期之前价格正常（可能是真实预测）: {valid_prediction_count} ({valid_prediction_count/total_checked*100:.1f}%)")
        
        if leakage_count / total_checked > 0.3:
            print(f"\n[WARN] {leakage_count/total_checked*100:.1f}%的异常值在预测日期之前已经大涨")
            print(f"这可能表明存在数据泄露问题（look-ahead bias）")
        else:
            print(f"\n[OK] 大部分异常值在预测日期之前价格正常")
            print(f"模型可能确实在大涨之前就预测到了这些异常收益")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    check_data_leakage_and_timing()
