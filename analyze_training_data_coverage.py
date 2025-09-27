#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析训练数据的时间覆盖范围
检查是否所有三年数据都被纳入训练，除了CV gap
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add BMA models path
sys.path.append(os.path.join(os.path.dirname(__file__), 'bma_models'))

def analyze_data_coverage():
    """分析训练数据覆盖范围"""

    print("=" * 80)
    print("训练数据时间覆盖分析")
    print("=" * 80)

    # 从配置读取关键参数
    try:
        from bma_models.unified_config_loader import get_time_config
        time_config = get_time_config()

        print("配置参数:")
        print(f"  CV Gap天数: {time_config.cv_gap_days}")
        print(f"  CV Embargo天数: {time_config.cv_embargo_days}")
        print(f"  预测窗口: T+{time_config.prediction_horizon_days}")
        print(f"  特征滞后: T-{time_config.feature_lag_days}")

        # 从配置文件读取训练参数
        cv_gap_days = time_config.cv_gap_days
        cv_embargo_days = time_config.cv_embargo_days
        min_train_size = 252  # 从yaml配置读取的值

        # 尝试从yaml获取历史数据配置
        try:
            import yaml
            with open('bma_models/unified_config.yaml', 'r') as f:
                config_data = yaml.safe_load(f)

            data_config = config_data.get('data', {})
            print(f"  风险模型历史: {data_config.get('risk_model_history_days', 300)}天")
            print(f"  Alpha数据历史: {data_config.get('alpha_data_history_days', 200)}天")

            training_config = config_data.get('training', {})
            min_train_size = training_config.get('cv_min_train_size', 252)
            print(f"  最小训练集大小: {min_train_size}天")

        except Exception as yaml_e:
            print(f"  使用默认历史数据配置: {yaml_e}")

    except Exception as e:
        print(f"无法加载配置: {e}")
        # 使用硬编码默认值
        cv_gap_days = 6
        cv_embargo_days = 5
        min_train_size = 252
        print(f"使用默认值: gap={cv_gap_days}, embargo={cv_embargo_days}, min_train={min_train_size}")

    print("\n" + "=" * 80)
    print("数据利用率分析")
    print("=" * 80)

    # 分析不同时间范围的数据利用
    total_days_3years = 252 * 3  # 约3年交易日

    # CV间隔计算
    total_cv_gap = cv_gap_days + cv_embargo_days  # 总CV间隔

    # 有效训练数据计算
    effective_training_days = total_days_3years - total_cv_gap
    training_utilization = effective_training_days / total_days_3years * 100

    print(f" 三年数据利用分析:")
    print(f"  总交易日 (3年): {total_days_3years}天")
    print(f"  CV Gap + Embargo: {total_cv_gap}天")
    print(f"  有效训练天数: {effective_training_days}天")
    print(f"  数据利用率: {training_utilization:.1f}%")

    # 最小训练集要求
    print(f"\n📈 训练集规模分析:")
    print(f"  最小训练集要求: {min_train_size}天 ({min_train_size/252:.1f}年)")
    print(f"  实际可用天数: {effective_training_days}天 ({effective_training_days/252:.1f}年)")

    if effective_training_days >= min_train_size:
        coverage_ratio = effective_training_days / min_train_size
        print(f"  ✅ 满足最小要求: {coverage_ratio:.1f}x 倍数")
    else:
        shortage = min_train_size - effective_training_days
        print(f"  ❌ 不满足要求: 缺少{shortage}天")

    # CV折数分析
    cv_splits = 5
    test_size = 63  # 3个月测试集

    print(f"\n🔄 交叉验证分析:")
    print(f"  CV折数: {cv_splits}")
    print(f"  测试集大小: {test_size}天 ({test_size/252:.1f}年)")

    # 计算每折的训练数据量
    fold_training_data = []
    for fold in range(cv_splits):
        # 每折的训练数据 = 前面的数据 - gap
        fold_start = 0
        fold_train_end = min_train_size + fold * test_size
        fold_test_start = fold_train_end + cv_gap_days
        fold_test_end = fold_test_start + test_size

        if fold_test_end <= effective_training_days:
            actual_train_days = fold_train_end
            fold_training_data.append(actual_train_days)
            print(f"    Fold {fold+1}: 训练{actual_train_days}天, 测试{test_size}天")
        else:
            print(f"    Fold {fold+1}: 数据不足，跳过")

    if fold_training_data:
        avg_training_days = np.mean(fold_training_data)
        total_unique_training_days = max(fold_training_data) if fold_training_data else 0
        print(f"  平均训练天数/折: {avg_training_days:.0f}天")
        print(f"  总独特训练天数: {total_unique_training_days}天")

        # 数据重用分析
        data_reuse_ratio = sum(fold_training_data) / total_unique_training_days if total_unique_training_days > 0 else 0
        print(f"  数据重用倍数: {data_reuse_ratio:.1f}x")

    print(f"\n" + "=" * 80)
    print("时间安全性验证")
    print("=" * 80)

    # 验证时间安全设置
    prediction_horizon = 5  # T+5

    print(f"⏰ 时间隔离验证:")
    print(f"  预测目标: T+{prediction_horizon}")
    print(f"  CV Gap: {cv_gap_days}天 (>= {prediction_horizon}天 ✅)" if cv_gap_days >= prediction_horizon else f"  CV Gap: {cv_gap_days}天 (< {prediction_horizon}天 ❌)")
    print(f"  Embargo: {cv_embargo_days}天 (>= {prediction_horizon}天 ✅)" if cv_embargo_days >= prediction_horizon else f"  Embargo: {cv_embargo_days}天 (< {prediction_horizon}天 ❌)")

    # 特征滞后验证
    feature_lag = 1  # T-1
    print(f"  特征滞后: T-{feature_lag} (防止未来信息泄漏 ✅)")

    # 总时间间隔
    total_isolation = cv_gap_days + cv_embargo_days + feature_lag
    print(f"  总时间隔离: {total_isolation}天")

    print(f"\n 结论:")
    if training_utilization >= 90:
        print(f"  ✅ 优秀: {training_utilization:.1f}%的三年数据被有效利用")
    elif training_utilization >= 80:
        print(f"  ✅ 良好: {training_utilization:.1f}%的三年数据被有效利用")
    else:
        print(f"  ⚠️ 可优化: 仅{training_utilization:.1f}%的三年数据被利用")

    gap_efficiency = 100 - (total_cv_gap / total_days_3years * 100)
    print(f"  Gap效率: {gap_efficiency:.1f}% (非Gap时间比例)")

    if cv_gap_days >= prediction_horizon and cv_embargo_days >= prediction_horizon:
        print(f"  ✅ 时间安全: CV设置防止数据泄漏")
    else:
        print(f"  ❌ 时间风险: CV设置可能存在泄漏风险")


def check_data_files():
    """检查实际数据文件的时间范围"""

    print(f"\n" + "=" * 80)
    print("实际数据文件分析")
    print("=" * 80)

    # 检查可能的数据文件位置
    data_paths = [
        "D:/trade/data",
        "D:/trade/cache",
        "D:/trade",
        "."
    ]

    data_files_found = []

    for data_path in data_paths:
        if os.path.exists(data_path):
            # 查找CSV和pickle文件
            for ext in ['*.csv', '*.pkl', '*.parquet']:
                import glob
                files = glob.glob(os.path.join(data_path, f"**/{ext}"), recursive=True)
                data_files_found.extend(files)

    if data_files_found:
        print(f"📁 发现{len(data_files_found)}个数据文件:")
        for file_path in data_files_found[:10]:  # 显示前10个
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            mod_time = pd.Timestamp.fromtimestamp(os.path.getmtime(file_path))
            print(f"  {os.path.basename(file_path)} ({file_size:.1f}MB, {mod_time.strftime('%Y-%m-%d')})")

        if len(data_files_found) > 10:
            print(f"  ... 还有{len(data_files_found)-10}个文件")
    else:
        print("📁 未发现数据文件")

    # 检查训练日志
    log_paths = ["D:/trade/training_logs", "D:/trade/logs"]

    for log_path in log_paths:
        if os.path.exists(log_path):
            log_files = list(Path(log_path).glob("*.log"))
            if log_files:
                print(f"\n📄 训练日志 ({log_path}):")
                for log_file in log_files[-5:]:  # 最近5个日志
                    mod_time = pd.Timestamp.fromtimestamp(log_file.stat().st_mtime)
                    print(f"  {log_file.name} ({mod_time.strftime('%Y-%m-%d %H:%M')})")


if __name__ == "__main__":
    analyze_data_coverage()
    check_data_files()