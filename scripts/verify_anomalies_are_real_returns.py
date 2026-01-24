#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证异常值是否是真实收益还是数据错误
通过检查实际价格数据来验证
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def verify_anomalies_are_real():
    """验证异常值是否是真实收益"""
    print("=" * 80)
    print("验证异常值是否是真实收益")
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
    target_valid = test_df['target'].dropna()
    extreme_targets = target_valid[target_valid > 0.5]  # > 50%
    
    print(f"\n[3] 异常target值统计:")
    print(f"   异常值数量（>50%）: {len(extreme_targets):,}")
    print(f"   最高target: {extreme_targets.max():.4f} ({extreme_targets.max()*100:.2f}%)")
    
    # 按ticker分组，找出异常值最多的ticker
    extreme_df = test_df[test_df['target'] > 0.5].copy()
    ticker_counts = extreme_df.index.get_level_values('ticker').value_counts()
    
    print(f"\n[4] 验证Top 20异常值ticker的真实收益:")
    print("=" * 80)
    
    verified_results = []
    
    for ticker, count in ticker_counts.head(20).items():
        ticker_data = extreme_df.loc[extreme_df.index.get_level_values('ticker') == ticker].copy()
        
        # 按日期排序
        ticker_data = ticker_data.sort_index(level='date')
        
        print(f"\n[Ticker: {ticker}] 异常值样本数: {count}")
        
        # 检查前5个异常值样本
        sample_count = min(5, len(ticker_data))
        
        for idx in range(sample_count):
            row = ticker_data.iloc[idx]
            date = row.name[0]  # MultiIndex的第一层是date
            target_val = row['target']
            close_t = row['Close']
            
            # 找到t+10的价格（10个交易日）
            # 需要找到该ticker在date之后10个交易日的价格
            ticker_all_data = test_df.loc[test_df.index.get_level_values('ticker') == ticker].copy()
            ticker_all_data = ticker_all_data.sort_index(level='date')
            
            # 找到date在ticker_all_data中的位置
            date_idx = ticker_all_data.index.get_level_values('date').get_loc(date)
            
            # 计算t+10的位置
            if date_idx + 10 < len(ticker_all_data):
                close_t10_row = ticker_all_data.iloc[date_idx + 10]
                close_t10 = close_t10_row['Close']
                date_t10 = close_t10_row.name[0]
                
                # 手动计算收益率
                if close_t > 0 and not pd.isna(close_t10) and close_t10 > 0:
                    manual_return = (close_t10 - close_t) / close_t
                    
                    # 检查价格是否异常
                    is_price_anomaly = False
                    price_anomaly_reason = ""
                    
                    if close_t == 0:
                        is_price_anomaly = True
                        price_anomaly_reason = "Close[t] = $0.00"
                    elif close_t < 0.01:
                        is_price_anomaly = True
                        price_anomaly_reason = f"Close[t] = ${close_t:.4f} (异常低)"
                    elif close_t > 10000:
                        is_price_anomaly = True
                        price_anomaly_reason = f"Close[t] = ${close_t:.2f} (异常高)"
                    elif close_t10 > 10000:
                        is_price_anomaly = True
                        price_anomaly_reason = f"Close[t+10] = ${close_t10:.2f} (异常高)"
                    
                    # 检查收益率是否合理
                    is_return_anomaly = False
                    return_anomaly_reason = ""
                    
                    if abs(manual_return - target_val) > 0.1:  # 差异>10%
                        is_return_anomaly = True
                        return_anomaly_reason = f"手动计算={manual_return:.4f}, target={target_val:.4f}, 差异={abs(manual_return-target_val):.4f}"
                    
                    # 判断是否是真实收益
                    is_real_return = True
                    if is_price_anomaly:
                        is_real_return = False
                    elif abs(manual_return - target_val) > 0.01:  # 差异>1%
                        is_real_return = False
                    elif manual_return > 1.0:  # >100%在10天内不太可能
                        is_real_return = False
                    
                    verified_results.append({
                        'ticker': ticker,
                        'date': date,
                        'date_t10': date_t10,
                        'close_t': close_t,
                        'close_t10': close_t10,
                        'target': target_val,
                        'manual_return': manual_return,
                        'diff': abs(manual_return - target_val),
                        'is_price_anomaly': is_price_anomaly,
                        'price_anomaly_reason': price_anomaly_reason,
                        'is_return_anomaly': is_return_anomaly,
                        'return_anomaly_reason': return_anomaly_reason,
                        'is_real_return': is_real_return
                    })
                    
                    status = "[REAL]" if is_real_return else "[ANOMALY]"
                    print(f"  样本 {idx+1}: {date.date()} -> {date_t10.date()}")
                    print(f"    {status} Close[t]=${close_t:.2f}, Close[t+10]=${close_t10:.2f}")
                    print(f"    Target={target_val:.4f} ({target_val*100:.2f}%), "
                          f"手动计算={manual_return:.4f} ({manual_return*100:.2f}%)")
                    if is_price_anomaly:
                        print(f"    [价格异常] {price_anomaly_reason}")
                    if is_return_anomaly:
                        print(f"    [计算异常] {return_anomaly_reason}")
                    if not is_real_return:
                        print(f"    [结论] 这不是真实收益，是数据异常")
                    else:
                        print(f"    [结论] 可能是真实收益（但10天内{manual_return*100:.2f}%非常罕见）")
                else:
                    print(f"  样本 {idx+1}: {date.date()} - 无法计算（价格数据缺失或为0）")
            else:
                print(f"  样本 {idx+1}: {date.date()} - 无法计算（没有t+10的数据）")
    
    # 汇总统计
    print(f"\n" + "=" * 80)
    print("[5] 验证结果汇总")
    print("=" * 80)
    
    if verified_results:
        results_df = pd.DataFrame(verified_results)
        
        real_count = results_df['is_real_return'].sum()
        anomaly_count = (~results_df['is_real_return']).sum()
        price_anomaly_count = results_df['is_price_anomaly'].sum()
        return_anomaly_count = results_df['is_return_anomaly'].sum()
        
        print(f"\n验证的样本数: {len(results_df)}")
        print(f"真实收益: {real_count} ({real_count/len(results_df)*100:.1f}%)")
        print(f"数据异常: {anomaly_count} ({anomaly_count/len(results_df)*100:.1f}%)")
        print(f"价格异常: {price_anomaly_count} ({price_anomaly_count/len(results_df)*100:.1f}%)")
        print(f"计算异常: {return_anomaly_count} ({return_anomaly_count/len(results_df)*100:.1f}%)")
        
        # 分析价格异常的原因
        if price_anomaly_count > 0:
            print(f"\n价格异常详情:")
            price_anomalies = results_df[results_df['is_price_anomaly'] == True]
            for reason, count in price_anomalies['price_anomaly_reason'].value_counts().items():
                print(f"   {reason}: {count}个样本")
        
        # 分析计算异常的原因
        if return_anomaly_count > 0:
            print(f"\n计算异常详情:")
            return_anomalies = results_df[results_df['is_return_anomaly'] == True]
            print(f"   平均差异: {return_anomalies['diff'].mean():.4f} ({return_anomalies['diff'].mean()*100:.2f}%)")
            print(f"   最大差异: {return_anomalies['diff'].max():.4f} ({return_anomalies['diff'].max()*100:.2f}%)")
        
        # 检查真实收益的合理性
        if real_count > 0:
            real_returns = results_df[results_df['is_real_return'] == True]
            print(f"\n真实收益统计:")
            print(f"   平均收益: {real_returns['manual_return'].mean():.4f} ({real_returns['manual_return'].mean()*100:.2f}%)")
            print(f"   最高收益: {real_returns['manual_return'].max():.4f} ({real_returns['manual_return'].max()*100:.2f}%)")
            print(f"   最低收益: {real_returns['manual_return'].min():.4f} ({real_returns['manual_return'].min()*100:.2f}%)")
            
            # 检查是否有>100%的收益（10天内几乎不可能）
            extreme_real = real_returns[real_returns['manual_return'] > 1.0]
            if len(extreme_real) > 0:
                print(f"\n   [WARN] {len(extreme_real)}个样本的收益>100%（10天内几乎不可能）")
                print(f"   这些可能是数据错误，即使价格数据看起来正常")
        
        # 结论
        print(f"\n" + "=" * 80)
        print("[6] 结论")
        print("=" * 80)
        
        if anomaly_count / len(results_df) > 0.8:
            print(f"\n[结论] 异常值主要是数据错误（{anomaly_count/len(results_df)*100:.1f}%）")
            print(f"   这些异常值不是真实收益，应该被清洗")
        elif anomaly_count / len(results_df) > 0.5:
            print(f"\n[结论] 异常值大部分是数据错误（{anomaly_count/len(results_df)*100:.1f}%）")
            print(f"   建议清洗这些异常值")
        else:
            print(f"\n[结论] 异常值中有一部分可能是真实收益（{real_count/len(results_df)*100:.1f}%）")
            print(f"   但即使是真实收益，10天内{real_returns['manual_return'].mean()*100:.2f}%的收益也异常高")
            print(f"   建议对所有异常值进行winsorization处理")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    verify_anomalies_are_real()
