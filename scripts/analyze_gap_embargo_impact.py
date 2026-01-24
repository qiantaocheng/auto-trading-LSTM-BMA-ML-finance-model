#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析PurgedCV中gap=10和embargo=10对训练的影响
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def simulate_purged_cv_splits(n_groups: int, n_splits: int = 6, gap: int = 10, 
                               embargo: int = 10, test_size: int = None) -> List[Dict]:
    """
    模拟PurgedCV的split过程，分析gap和embargo的影响
    
    Args:
        n_groups: 总的时间组数（日期数）
        n_splits: CV折数
        gap: Gap天数（训练集结束到测试集开始之间的间隔）
        embargo: Embargo天数（测试集结束后禁止用于训练的天数）
        test_size: 每个测试集的大小（如果为None则自动计算）
    """
    if test_size is None:
        test_size = n_groups // (n_splits + 1)
    
    splits_info = []
    
    for i in range(n_splits):
        # 计算测试集位置
        test_start = (i + 1) * test_size
        test_end = min(test_start + test_size, n_groups - embargo)
        
        if test_start >= test_end:
            break
        
        # 计算训练集位置（考虑gap）
        train_end = test_start - gap
        train_start = 0
        
        if train_start >= train_end:
            continue
        
        # 计算被排除的区域
        excluded_before_test = max(0, test_start - gap)  # gap区域
        excluded_after_test = min(n_groups, test_end + embargo)  # embargo区域
        
        # 统计信息
        train_size = train_end - train_start
        test_size_actual = test_end - test_start
        gap_size = gap
        embargo_size = embargo
        total_excluded = gap_size + embargo_size
        
        # 计算可用数据比例
        total_available = n_groups
        usable_for_train = train_size
        usable_pct = (usable_for_train / total_available) * 100
        
        splits_info.append({
            'fold': i + 1,
            'train_start': train_start,
            'train_end': train_end,
            'train_size': train_size,
            'gap_start': train_end,
            'gap_end': test_start,
            'gap_size': gap_size,
            'test_start': test_start,
            'test_end': test_end,
            'test_size': test_size_actual,
            'embargo_start': test_end,
            'embargo_end': excluded_after_test,
            'embargo_size': embargo_size,
            'total_excluded': total_excluded,
            'usable_pct': usable_pct,
            'total_available': total_available
        })
    
    return splits_info


def analyze_gap_embargo_impact(n_dates: int = 985, n_splits: int = 6, 
                                gap: int = 10, embargo: int = 10):
    """
    分析gap和embargo对训练的影响
    
    Args:
        n_dates: 训练期的总日期数（例如985天）
        n_splits: CV折数
        gap: Gap天数
        embargo: Embargo天数
    """
    print("=" * 80)
    print("PURGEDCV GAP和EMBARGO影响分析")
    print("=" * 80)
    
    print(f"\n配置参数:")
    print(f"  总日期数: {n_dates}")
    print(f"  CV折数: {n_splits}")
    print(f"  Gap天数: {gap}")
    print(f"  Embargo天数: {embargo}")
    print(f"  总隔离天数: {gap + embargo}")
    
    # 模拟CV splits
    splits_info = simulate_purged_cv_splits(n_dates, n_splits, gap, embargo)
    
    print(f"\n" + "=" * 80)
    print("各折详细信息")
    print("=" * 80)
    
    total_train_samples = 0
    total_test_samples = 0
    total_excluded_samples = 0
    
    for split in splits_info:
        print(f"\n折 {split['fold']}:")
        print(f"  训练集: 日期 {split['train_start']} 到 {split['train_end']} ({split['train_size']} 天)")
        print(f"  Gap区域: 日期 {split['gap_start']} 到 {split['gap_end']} ({split['gap_size']} 天) [排除]")
        print(f"  测试集: 日期 {split['test_start']} 到 {split['test_end']} ({split['test_size']} 天)")
        print(f"  Embargo区域: 日期 {split['embargo_start']} 到 {split['embargo_end']} ({split['embargo_size']} 天) [排除]")
        print(f"  总排除: {split['total_excluded']} 天")
        print(f"  可用训练数据比例: {split['usable_pct']:.1f}%")
        
        total_train_samples += split['train_size']
        total_test_samples += split['test_size']
        total_excluded_samples += split['total_excluded']
    
    print(f"\n" + "=" * 80)
    print("总体影响分析")
    print("=" * 80)
    
    avg_train_size = total_train_samples / len(splits_info) if splits_info else 0
    avg_test_size = total_test_samples / len(splits_info) if splits_info else 0
    avg_excluded = total_excluded_samples / len(splits_info) if splits_info else 0
    
    print(f"\n平均每折:")
    print(f"  训练集大小: {avg_train_size:.0f} 天")
    print(f"  测试集大小: {avg_test_size:.0f} 天")
    print(f"  排除区域: {avg_excluded:.0f} 天 (gap={gap} + embargo={embargo})")
    
    # 计算数据利用率
    # 在PurgedCV中，每个样本可能被用于训练多次（不同折），但每次训练时都会排除gap+embargo区域
    max_possible_train = n_dates  # 如果没有gap和embargo，最多可用n_dates天
    actual_avg_train = avg_train_size
    
    data_utilization = (actual_avg_train / max_possible_train) * 100
    data_loss = 100 - data_utilization
    
    print(f"\n数据利用率分析:")
    print(f"  最大可能训练天数: {max_possible_train} 天")
    print(f"  实际平均训练天数: {actual_avg_train:.0f} 天")
    print(f"  数据利用率: {data_utilization:.1f}%")
    print(f"  数据损失: {data_loss:.1f}% (由于gap和embargo)")
    
    # 分析不同gap/embargo组合的影响
    print(f"\n" + "=" * 80)
    print("不同Gap/Embargo组合对比")
    print("=" * 80)
    
    scenarios = [
        (0, 0, "无隔离"),
        (5, 5, "Gap=5, Embargo=5"),
        (10, 5, "Gap=10, Embargo=5"),
        (10, 10, "Gap=10, Embargo=10 (当前)"),
        (15, 10, "Gap=15, Embargo=10"),
        (10, 15, "Gap=10, Embargo=15"),
    ]
    
    print(f"\n{'场景':<25} {'平均训练天数':<15} {'数据利用率':<15} {'隔离天数':<15}")
    print("-" * 70)
    
    for g, e, name in scenarios:
        splits = simulate_purged_cv_splits(n_dates, n_splits, g, e)
        if splits:
            avg_train = sum(s['train_size'] for s in splits) / len(splits)
            utilization = (avg_train / n_dates) * 100
            isolation = g + e
            print(f"{name:<25} {avg_train:<15.0f} {utilization:<15.1f}% {isolation:<15}")
    
    # 分析对模型性能的潜在影响
    print(f"\n" + "=" * 80)
    print("对训练的影响分析")
    print("=" * 80)
    
    print(f"\n1. 数据量影响:")
    print(f"   - Gap={gap}, Embargo={embargo} 导致每折平均损失 {avg_excluded:.0f} 天训练数据")
    print(f"   - 数据利用率从100%降至{data_utilization:.1f}%")
    print(f"   - 这意味着模型看到的训练样本减少了 {data_loss:.1f}%")
    
    print(f"\n2. 时间安全性:")
    print(f"   - Gap防止前向泄漏: 训练集结束到测试集开始之间有{gap}天间隔")
    print(f"   - Embargo防止后向泄漏: 测试集结束后{embargo}天内不能用于训练")
    print(f"   - 总隔离期: {gap + embargo}天，确保时间安全")
    
    print(f"\n3. 模型性能影响:")
    print(f"   - 优点: 完全防止数据泄漏，确保OOS性能真实可靠")
    print(f"   - 缺点: 训练数据减少可能导致:")
    print(f"     * 模型容量受限（需要更简单的模型）")
    print(f"     * 过拟合风险降低（但可能欠拟合）")
    print(f"     * 特征重要性估计更保守")
    
    print(f"\n4. 与Horizon的关系:")
    print(f"   - 当前horizon=10天，gap=10天，embargo=10天")
    print(f"   - Gap应该 >= horizon (当前: {gap} >= 10) [满足]")
    print(f"   - Embargo通常 = horizon (当前: {embargo} == 10) [满足]")
    print(f"   - 这是标准配置，确保完全的时间隔离")
    
    # 计算实际样本数影响（假设每天约3000个样本）
    samples_per_day = 3270  # 基于之前的数据: 3,220,836 / 985 ≈ 3270
    avg_train_samples = avg_train_size * samples_per_day
    avg_excluded_samples = avg_excluded * samples_per_day
    
    print(f"\n5. 样本数影响 (假设每天{samples_per_day}个样本):")
    print(f"   - 平均每折训练样本: {avg_train_samples:,.0f}")
    print(f"   - 平均每折排除样本: {avg_excluded_samples:,.0f}")
    print(f"   - 样本损失比例: {(avg_excluded_samples / (avg_train_samples + avg_excluded_samples)) * 100:.1f}%")
    
    print(f"\n" + "=" * 80)
    print("建议")
    print("=" * 80)
    
    if gap == 10 and embargo == 10:
        print(f"\n当前配置 (Gap=10, Embargo=10) 是合理的:")
        print(f"  [优点]")
        print(f"  - 完全防止数据泄漏")
        print(f"  - Gap = Horizon，符合最佳实践")
        print(f"  - Embargo = Horizon，确保后向安全")
        print(f"  - 数据利用率 {data_utilization:.1f}% 仍然可接受")
        print(f"  ")
        print(f"  [潜在改进]")
        if data_utilization < 80:
            print(f"  - 如果数据充足，可以考虑减少到 Gap=5, Embargo=5")
            print(f"  - 这将提高数据利用率，但仍保持时间安全")
        else:
            print(f"  - 当前配置已经平衡了时间安全和数据利用")
    
    print(f"\n" + "=" * 80)


if __name__ == "__main__":
    # 使用实际的训练期日期数（985天，从之前的分析）
    analyze_gap_embargo_impact(n_dates=985, n_splits=6, gap=10, embargo=10)
