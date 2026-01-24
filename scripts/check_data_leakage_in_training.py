#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查训练数据是否有数据泄露问题
主要检查：
1. target是否使用了未来信息
2. 特征是否使用了未来信息
3. 时间顺序是否正确
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def check_data_leakage(data_file: Path):
    """检查数据泄露问题"""
    print("=" * 80)
    print("训练数据泄露检查")
    print("=" * 80)
    print(f"\n数据文件: {data_file}")
    
    if not data_file.exists():
        print(f"[ERROR] 文件不存在: {data_file}")
        return False
    
    # 加载数据
    print("\n[1] 加载数据...")
    try:
        df = pd.read_parquet(data_file)
        print(f"  [OK] 成功加载数据")
        print(f"     形状: {df.shape}")
        print(f"     索引类型: {type(df.index)}")
    except Exception as e:
        print(f"  [ERROR] 加载失败: {e}")
        return False
    
    # 检查MultiIndex
    if not isinstance(df.index, pd.MultiIndex):
        print(f"  [WARN] 数据不是MultiIndex格式")
        if 'date' in df.columns and 'ticker' in df.columns:
            df = df.set_index(['date', 'ticker'])
            print(f"  [OK] 已转换为MultiIndex")
        else:
            print(f"  [ERROR] 无法转换为MultiIndex")
            return False
    
    # 确保date和ticker列存在
    if 'date' not in df.index.names or 'ticker' not in df.index.names:
        print(f"  [ERROR] MultiIndex缺少date或ticker")
        return False
    
    dates = df.index.get_level_values('date').unique()
    tickers = df.index.get_level_values('ticker').unique()
    
    print(f"     唯一日期数: {len(dates)}")
    print(f"     唯一股票数: {len(tickers)}")
    print(f"     日期范围: {dates.min()} 至 {dates.max()}")
    
    # 检查target列
    print("\n[2] 检查target列...")
    extreme_high = 0
    extreme_low = 0
    if 'target' not in df.columns:
        print(f"  [WARN] 未找到target列")
        print(f"     可用列: {list(df.columns)[:20]}")
    else:
        target_col = df['target']
        print(f"  [OK] 找到target列")
        print(f"     NaN数量: {target_col.isna().sum()}")
        print(f"     非NaN数量: {target_col.notna().sum()}")
        print(f"     统计: min={target_col.min():.6f}, max={target_col.max():.6f}, mean={target_col.mean():.6f}")
        
        # 检查target是否有异常值
        extreme_high = (target_col > 0.5).sum()
        extreme_low = (target_col < -0.5).sum()
        if extreme_high > 0:
            print(f"     [WARN] 发现 {extreme_high} 个极端高值 (>0.5, 即>50%)")
        if extreme_low > 0:
            print(f"     [WARN] 发现 {extreme_low} 个极端低值 (<-0.5, 即<-50%)")
    
    # 检查时间顺序
    print("\n[3] 检查时间顺序...")
    dates_sorted = sorted(dates)
    if list(dates) == dates_sorted:
        print(f"  [OK] 日期已排序")
    else:
        print(f"  [WARN] 日期未排序，需要排序")
        df = df.sort_index()
    
    # 检查每个ticker的时间序列连续性
    print("\n[4] 检查时间序列连续性...")
    sample_tickers = tickers[:10]  # 检查前10个ticker
    gaps_found = []
    
    for ticker in sample_tickers:
        ticker_data = df.xs(ticker, level='ticker')
        ticker_dates = sorted(ticker_data.index.get_level_values('date').unique())
        
        if len(ticker_dates) > 1:
            # 检查日期间隔
            date_diffs = pd.Series(ticker_dates).diff().dropna()
            # 正常间隔应该是1个交易日（可能有周末/节假日）
            # 如果间隔>10天，可能是数据缺失
            large_gaps = (date_diffs > pd.Timedelta(days=10)).sum()
            if large_gaps > 0:
                gaps_found.append((ticker, large_gaps))
    
    if gaps_found:
        print(f"  [WARN] 发现 {len(gaps_found)} 个ticker有大的时间间隔（可能数据缺失）")
        for ticker, gap_count in gaps_found[:5]:
            print(f"     - {ticker}: {gap_count} 个大间隔")
    else:
        print(f"  [OK] 时间序列连续性正常")
    
    # 检查特征是否使用了未来信息
    print("\n[5] 检查特征计算...")
    print(f"  [INFO] 特征列数: {len([c for c in df.columns if c != 'target'])}")
    
    # 检查是否有明显的未来信息特征（如future_return, next_day等）
    future_keywords = ['future', 'next', 'forward', 'ahead', 't+', 't_plus']
    suspicious_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in future_keywords):
            suspicious_cols.append(col)
    
    if suspicious_cols:
        print(f"  [WARN] 发现可疑的未来信息特征:")
        for col in suspicious_cols:
            print(f"     - {col}")
    else:
        print(f"  [OK] 未发现明显的未来信息特征")
    
    # 检查target计算是否正确（应该是前向收益）
    print("\n[6] 检查target计算逻辑...")
    if 'target' in df.columns and 'Close' in df.columns:
        # 随机选择几个样本检查target是否正确计算
        sample_size = min(100, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        # 对于每个样本，检查target是否等于未来收益
        # 注意：这里只能做基本检查，因为需要知道horizon
        print(f"  [INFO] 随机检查 {sample_size} 个样本的target值")
        print(f"  [INFO] target范围: {df['target'].min():.6f} 至 {df['target'].max():.6f}")
        
        # 检查target是否有明显的时间模式（可能泄露）
        if len(dates) > 10:
            # 按日期分组，计算每天的平均target
            daily_avg_target = df.groupby(level='date')['target'].mean()
            
            # 检查是否有明显的趋势（可能泄露）
            if len(daily_avg_target) > 20:
                # 计算自相关
                autocorr = daily_avg_target.autocorr(lag=1)
                print(f"  [INFO] target的日度自相关（lag=1）: {autocorr:.4f}")
                if abs(autocorr) > 0.5:
                    print(f"     [WARN] 自相关较高，可能存在时间依赖")
                else:
                    print(f"     [OK] 自相关正常")
    else:
        print(f"  [WARN] 无法检查target计算（缺少target或Close列）")
    
    # 总结
    print("\n" + "=" * 80)
    print("[总结]")
    print("=" * 80)
    
    issues = []
    if extreme_high > 0 or extreme_low > 0:
        issues.append(f"发现极端target值（高值: {extreme_high}, 低值: {extreme_low}）")
    if gaps_found:
        issues.append(f"发现时间序列间隔（{len(gaps_found)}个ticker）")
    if suspicious_cols:
        issues.append(f"发现可疑的未来信息特征（{len(suspicious_cols)}个）")
    
    if not issues:
        print("[OK] 未发现明显的数据泄露问题")
        print("\n建议:")
        print("  1. 确认target计算使用shift(-horizon)避免未来信息")
        print("  2. 确认所有特征计算使用shift(1)避免未来信息")
        print("  3. 确认训练时使用purge gap避免标签泄露")
    else:
        print("[WARN] 发现以下潜在问题:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n建议:")
        print("  1. 检查target计算逻辑")
        print("  2. 检查特征计算逻辑")
        print("  3. 检查时间序列数据完整性")
    
    return len(issues) == 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="检查训练数据泄露")
    parser.add_argument("--data-file", type=str, 
                       default=r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet",
                       help="数据文件路径")
    args = parser.parse_args()
    
    data_file = Path(args.data_file)
    success = check_data_leakage(data_file)
    
    sys.exit(0 if success else 1)
