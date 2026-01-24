#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析异常值对四个模型胜率的影响
找出所有异常值并分析其严重程度
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def analyze_anomalies_impact():
    """分析异常值对模型胜率的影响"""
    print("=" * 80)
    print("异常值对四个模型胜率影响分析")
    print("=" * 80)
    
    data_file = "data/factor_exports/polygon_factors_all_filtered_clean_final_v2.parquet"
    
    print(f"\n[1] 加载数据文件...")
    df = pd.read_parquet(data_file)
    print(f"   [OK] 数据形状: {df.shape}")
    print(f"   日期范围: {df.index.get_level_values('date').min()} 至 {df.index.get_level_values('date').max()}")
    
    # 提取测试期数据（2025年）
    test_start = pd.Timestamp('2025-01-02')
    test_end = pd.Timestamp('2025-12-30')
    
    test_dates = df.index.get_level_values('date')
    test_mask = (test_dates >= test_start) & (test_dates <= test_end)
    test_df = df[test_mask].copy()
    
    print(f"\n[2] 测试期数据统计:")
    print(f"   测试期样本数: {len(test_df):,}")
    print(f"   唯一日期: {test_df.index.get_level_values('date').nunique()}")
    print(f"   唯一ticker: {test_df.index.get_level_values('ticker').nunique()}")
    
    if 'target' not in test_df.columns:
        print(f"\n[ERROR] 测试数据中没有target列")
        return
    
    if 'Close' not in test_df.columns:
        print(f"\n[ERROR] 测试数据中没有Close列")
        return
    
    # ========== 1. 分析价格异常值 ==========
    print(f"\n" + "=" * 80)
    print("[3] 价格异常值分析")
    print("=" * 80)
    
    close = test_df['Close']
    target = test_df['target']
    
    # 1.1 异常高价格
    high_price_threshold = 10000  # $10,000
    high_prices = close[close > high_price_threshold]
    print(f"\n[3.1] 异常高价格（> ${high_price_threshold:,}）:")
    print(f"   数量: {len(high_prices):,} ({len(high_prices)/len(close)*100:.2f}%)")
    
    if len(high_prices) > 0:
        print(f"   最高价格: ${high_prices.max():.2f}")
        print(f"   平均价格: ${high_prices.mean():.2f}")
        print(f"   涉及的ticker数量: {test_df[test_df['Close'] > high_price_threshold].index.get_level_values('ticker').nunique()}")
        
        # 找出异常高价格的ticker
        high_price_tickers = test_df[test_df['Close'] > high_price_threshold].index.get_level_values('ticker').unique()
        print(f"\n   Top 10 异常高价格ticker:")
        for ticker in high_price_tickers[:10]:
            ticker_prices = test_df.loc[test_df.index.get_level_values('ticker') == ticker, 'Close']
            ticker_targets = test_df.loc[test_df.index.get_level_values('ticker') == ticker, 'target']
            print(f"     {ticker}: 最高价格=${ticker_prices.max():.2f}, 平均价格=${ticker_prices.mean():.2f}, "
                  f"最高target={ticker_targets.max():.4f} ({ticker_targets.max()*100:.2f}%)")
    
    # 1.2 异常低价格（$0.00或接近$0）
    zero_prices = close[close == 0]
    very_low_prices = close[(close > 0) & (close < 0.01)]
    print(f"\n[3.2] 异常低价格:")
    print(f"   $0.00价格: {len(zero_prices):,} ({len(zero_prices)/len(close)*100:.2f}%)")
    print(f"   < $0.01价格: {len(very_low_prices):,} ({len(very_low_prices)/len(close)*100:.2f}%)")
    
    if len(zero_prices) > 0:
        zero_price_tickers = test_df[test_df['Close'] == 0].index.get_level_values('ticker').unique()
        print(f"   涉及的ticker数量: {len(zero_price_tickers)}")
        print(f"\n   Top 10 $0.00价格ticker:")
        for ticker in zero_price_tickers[:10]:
            ticker_data = test_df.loc[test_df.index.get_level_values('ticker') == ticker]
            zero_dates = ticker_data[ticker_data['Close'] == 0].index.get_level_values('date')
            print(f"     {ticker}: {len(zero_dates)}个$0.00价格日期")
            # 检查这些日期之后的target值
            for date in zero_dates[:3]:  # 只显示前3个
                date_data = ticker_data.loc[ticker_data.index.get_level_values('date') == date]
                if len(date_data) > 0:
                    target_val = date_data['target'].iloc[0]
                    print(f"       {date.date()}: target={target_val:.4f} ({target_val*100:.2f}%)")
    
    # ========== 2. 分析Target异常值 ==========
    print(f"\n" + "=" * 80)
    print("[4] Target异常值分析")
    print("=" * 80)
    
    target_valid = target.dropna()
    
    # 2.1 异常高target值
    extreme_targets_50 = target_valid[target_valid > 0.5]  # > 50%
    extreme_targets_100 = target_valid[target_valid > 1.0]  # > 100%
    extreme_targets_30 = target_valid[target_valid > 0.3]   # > 30%
    
    print(f"\n[4.1] 异常高target值统计:")
    print(f"   > 30%: {len(extreme_targets_30):,} ({len(extreme_targets_30)/len(target_valid)*100:.2f}%)")
    print(f"   > 50%: {len(extreme_targets_50):,} ({len(extreme_targets_50)/len(target_valid)*100:.2f}%)")
    print(f"   > 100%: {len(extreme_targets_100):,} ({len(extreme_targets_100)/len(target_valid)*100:.2f}%)")
    
    if len(extreme_targets_50) > 0:
        print(f"\n   异常target值统计（>50%）:")
        print(f"     均值: {extreme_targets_50.mean():.4f} ({extreme_targets_50.mean()*100:.2f}%)")
        print(f"     中位数: {extreme_targets_50.median():.4f} ({extreme_targets_50.median()*100:.2f}%)")
        print(f"     最大值: {extreme_targets_50.max():.4f} ({extreme_targets_50.max()*100:.2f}%)")
        print(f"     99%分位数: {extreme_targets_50.quantile(0.99):.4f} ({extreme_targets_50.quantile(0.99)*100:.2f}%)")
    
    # 2.2 找出异常target值的ticker和日期
    if len(extreme_targets_50) > 0:
        extreme_targets_df = test_df[test_df['target'] > 0.5].copy()
        extreme_tickers = extreme_targets_df.index.get_level_values('ticker').unique()
        extreme_dates = extreme_targets_df.index.get_level_values('date').unique()
        
        print(f"\n[4.2] 异常target值分布:")
        print(f"   涉及的ticker数量: {len(extreme_tickers)}")
        print(f"   涉及的日期数量: {len(extreme_dates)}")
        
        # 按ticker统计
        ticker_counts = extreme_targets_df.index.get_level_values('ticker').value_counts()
        print(f"\n   Top 20 异常target值ticker（按样本数）:")
        for ticker, count in ticker_counts.head(20).items():
            ticker_data = extreme_targets_df.loc[extreme_targets_df.index.get_level_values('ticker') == ticker]
            max_target = ticker_data['target'].max()
            mean_target = ticker_data['target'].mean()
            print(f"     {ticker}: {count}个样本, 最高target={max_target:.4f} ({max_target*100:.2f}%), "
                  f"平均target={mean_target:.4f} ({mean_target*100:.2f}%)")
        
        # 按日期统计
        date_counts = extreme_targets_df.index.get_level_values('date').value_counts()
        print(f"\n   Top 20 异常target值日期（按样本数）:")
        for date, count in date_counts.head(20).items():
            date_data = extreme_targets_df.loc[extreme_targets_df.index.get_level_values('date') == date]
            max_target = date_data['target'].max()
            mean_target = date_data['target'].mean()
            print(f"     {date.date()}: {count}个样本, 最高target={max_target:.4f} ({max_target*100:.2f}%), "
                  f"平均target={mean_target:.4f} ({mean_target*100:.2f}%)")
    
    # ========== 3. 分析异常值对Top N选择的影响 ==========
    print(f"\n" + "=" * 80)
    print("[5] 异常值对Top N选择的影响分析")
    print("=" * 80)
    
    # 模拟Top N选择（假设模型会选择target值最高的股票）
    dates = test_df.index.get_level_values('date').unique()
    dates_sorted = sorted(dates)
    
    # 计算每期Top N的平均收益（包含异常值）
    top5_returns_with_anomalies = []
    top10_returns_with_anomalies = []
    top20_returns_with_anomalies = []
    
    # 计算每期Top N的平均收益（排除异常值）
    top5_returns_without_anomalies = []
    top10_returns_without_anomalies = []
    top20_returns_without_anomalies = []
    
    # 统计每期异常值被选中的情况
    anomaly_selected_counts = {'top5': 0, 'top10': 0, 'top20': 0}
    total_periods = 0
    
    for date in dates_sorted:
        date_data = test_df.loc[test_df.index.get_level_values('date') == date]
        date_target = date_data['target'].dropna()
        
        if len(date_target) < 20:
            continue
        
        total_periods += 1
        
        # 识别异常值（target > 50%）
        is_anomaly = date_target > 0.5
        
        # 排序（降序）
        sorted_target = date_target.sort_values(ascending=False)
        
        # Top N（包含异常值）
        top5_with = sorted_target.head(5)
        top10_with = sorted_target.head(10)
        top20_with = sorted_target.head(20)
        
        top5_returns_with_anomalies.append(top5_with.mean())
        top10_returns_with_anomalies.append(top10_with.mean())
        top20_returns_with_anomalies.append(top20_with.mean())
        
        # 统计异常值被选中的情况
        top5_anomalies = is_anomaly.loc[top5_with.index].sum()
        top10_anomalies = is_anomaly.loc[top10_with.index].sum()
        top20_anomalies = is_anomaly.loc[top20_with.index].sum()
        
        if top5_anomalies > 0:
            anomaly_selected_counts['top5'] += 1
        if top10_anomalies > 0:
            anomaly_selected_counts['top10'] += 1
        if top20_anomalies > 0:
            anomaly_selected_counts['top20'] += 1
        
        # Top N（排除异常值）
        sorted_target_no_anomaly = date_target[~is_anomaly].sort_values(ascending=False)
        
        if len(sorted_target_no_anomaly) >= 5:
            top5_without = sorted_target_no_anomaly.head(5)
            top5_returns_without_anomalies.append(top5_without.mean())
        if len(sorted_target_no_anomaly) >= 10:
            top10_without = sorted_target_no_anomaly.head(10)
            top10_returns_without_anomalies.append(top10_without.mean())
        if len(sorted_target_no_anomaly) >= 20:
            top20_without = sorted_target_no_anomaly.head(20)
            top20_returns_without_anomalies.append(top20_without.mean())
    
    print(f"\n[5.1] 异常值被Top N选中的频率:")
    print(f"   总期数: {total_periods}")
    print(f"   Top 5包含异常值的期数: {anomaly_selected_counts['top5']} ({anomaly_selected_counts['top5']/total_periods*100:.1f}%)")
    print(f"   Top 10包含异常值的期数: {anomaly_selected_counts['top10']} ({anomaly_selected_counts['top10']/total_periods*100:.1f}%)")
    print(f"   Top 20包含异常值的期数: {anomaly_selected_counts['top20']} ({anomaly_selected_counts['top20']/total_periods*100:.1f}%)")
    
    print(f"\n[5.2] Top N平均收益对比（包含vs排除异常值）:")
    if top20_returns_with_anomalies:
        print(f"\n   Top 5:")
        print(f"     包含异常值: 均值={np.mean(top5_returns_with_anomalies):.4f} ({np.mean(top5_returns_with_anomalies)*100:.2f}%), "
              f"胜率={(np.array(top5_returns_with_anomalies) > 0).mean()*100:.1f}%")
        if top5_returns_without_anomalies:
            print(f"     排除异常值: 均值={np.mean(top5_returns_without_anomalies):.4f} ({np.mean(top5_returns_without_anomalies)*100:.2f}%), "
                  f"胜率={(np.array(top5_returns_without_anomalies) > 0).mean()*100:.1f}%")
        
        print(f"\n   Top 10:")
        print(f"     包含异常值: 均值={np.mean(top10_returns_with_anomalies):.4f} ({np.mean(top10_returns_with_anomalies)*100:.2f}%), "
              f"胜率={(np.array(top10_returns_with_anomalies) > 0).mean()*100:.1f}%")
        if top10_returns_without_anomalies:
            print(f"     排除异常值: 均值={np.mean(top10_returns_without_anomalies):.4f} ({np.mean(top10_returns_without_anomalies)*100:.2f}%), "
                  f"胜率={(np.array(top10_returns_without_anomalies) > 0).mean()*100:.1f}%")
        
        print(f"\n   Top 20:")
        print(f"     包含异常值: 均值={np.mean(top20_returns_with_anomalies):.4f} ({np.mean(top20_returns_with_anomalies)*100:.2f}%), "
              f"胜率={(np.array(top20_returns_with_anomalies) > 0).mean()*100:.1f}%")
        if top20_returns_without_anomalies:
            print(f"     排除异常值: 均值={np.mean(top20_returns_without_anomalies):.4f} ({np.mean(top20_returns_without_anomalies)*100:.2f}%), "
                  f"胜率={(np.array(top20_returns_without_anomalies) > 0).mean()*100:.1f}%")
    
    # ========== 4. 分析异常值的严重程度 ==========
    print(f"\n" + "=" * 80)
    print("[6] 异常值严重程度评估")
    print("=" * 80)
    
    print(f"\n[6.1] 异常值数量评估:")
    print(f"   总样本数: {len(target_valid):,}")
    print(f"   异常target值（>50%）: {len(extreme_targets_50):,} ({len(extreme_targets_50)/len(target_valid)*100:.2f}%)")
    print(f"   异常target值（>30%）: {len(extreme_targets_30):,} ({len(extreme_targets_30)/len(target_valid)*100:.2f}%)")
    
    # 计算如果每期Top 20选择，异常值被选中的概率
    avg_stocks_per_date = len(target_valid) / test_df.index.get_level_values('date').nunique()
    avg_anomalies_per_date = len(extreme_targets_50) / test_df.index.get_level_values('date').nunique()
    
    print(f"\n[6.2] 异常值被选中的概率分析:")
    print(f"   平均每期股票数: {avg_stocks_per_date:.0f}")
    print(f"   平均每期异常值数（>50%）: {avg_anomalies_per_date:.2f}")
    print(f"   如果选择Top 20，异常值被选中的概率: {min(1.0, avg_anomalies_per_date / 20) * 100:.1f}%")
    
    # 分析异常值对收益的影响
    if len(extreme_targets_50) > 0:
        avg_anomaly_return = extreme_targets_50.mean()
        avg_normal_return = target_valid[target_valid <= 0.5].mean()
        
        print(f"\n[6.3] 异常值对收益的影响:")
        print(f"   异常值平均收益（>50%）: {avg_anomaly_return:.4f} ({avg_anomaly_return*100:.2f}%)")
        print(f"   正常值平均收益（≤50%）: {avg_normal_return:.4f} ({avg_normal_return*100:.2f}%)")
        print(f"   差异: {(avg_anomaly_return - avg_normal_return):.4f} ({(avg_anomaly_return - avg_normal_return)*100:.2f}%)")
        
        # 计算如果Top 20中有1个异常值，对平均收益的影响
        impact_per_anomaly = (avg_anomaly_return - avg_normal_return) / 20
        print(f"   如果Top 20中有1个异常值，对平均收益的影响: {impact_per_anomaly:.4f} ({impact_per_anomaly*100:.2f}%)")
    
    # ========== 5. 结论 ==========
    print(f"\n" + "=" * 80)
    print("[7] 结论")
    print("=" * 80)
    
    severity_score = 0
    severity_reasons = []
    
    # 评估严重程度
    if len(extreme_targets_50) > 1000:
        severity_score += 3
        severity_reasons.append(f"异常值数量多（{len(extreme_targets_50):,}个，{len(extreme_targets_50)/len(target_valid)*100:.2f}%）")
    
    if len(extreme_targets_50) > 0 and extreme_targets_50.max() > 10.0:  # > 1000%
        severity_score += 3
        severity_reasons.append(f"异常值极端高（最高{extreme_targets_50.max()*100:.2f}%）")
    
    if anomaly_selected_counts['top20'] / total_periods > 0.5:  # > 50%的期数包含异常值
        severity_score += 2
        severity_reasons.append(f"异常值被频繁选中（{anomaly_selected_counts['top20']/total_periods*100:.1f}%的期数）")
    
    if top20_returns_with_anomalies and np.mean(top20_returns_with_anomalies) > 0.3:  # > 30%
        severity_score += 2
        severity_reasons.append(f"Top N收益异常高（{np.mean(top20_returns_with_anomalies)*100:.2f}%）")
    
    print(f"\n严重程度评分: {severity_score}/10")
    print(f"\n严重程度原因:")
    for reason in severity_reasons:
        print(f"   - {reason}")
    
    if severity_score >= 7:
        print(f"\n[结论] [WARN] 异常值严重程度: **非常严重**")
        print(f"   这些异常值足以导致所有模型的高胜率（96-100%）")
        print(f"   建议立即进行数据清洗")
    elif severity_score >= 4:
        print(f"\n[结论] [WARN] 异常值严重程度: **严重**")
        print(f"   这些异常值很可能导致模型的高胜率")
        print(f"   建议进行数据清洗")
    else:
        print(f"\n[结论] [WARN] 异常值严重程度: **中等**")
        print(f"   异常值可能对模型胜率有影响，但可能不是唯一原因")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_anomalies_impact()
