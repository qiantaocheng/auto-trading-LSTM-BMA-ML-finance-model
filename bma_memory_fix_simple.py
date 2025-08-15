#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMA增强版快速内存修复
解决stock数量多时的memory error问题
"""

import gc
import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)

def fix_bma_memory_issues():
    """快速修复BMA内存问题的建议"""
    
    print("=" * 60)
    print("BMA增强版内存优化快速修复指南")
    print("=" * 60)
    
    fixes = [
        {
            "问题": "股票池过大(80只)",
            "位置": "ENHANCED_STOCK_POOL",
            "修复": "限制为20-30只股票",
            "代码": """
# 在bma_walkforward_enhanced.py第1052行附近
test_tickers = ENHANCED_STOCK_POOL[:20]  # 从30改为20
"""
        },
        {
            "问题": "RandomForest参数过大", 
            "位置": "create_enhanced_bma_model()",
            "修复": "减少n_estimators和max_depth",
            "代码": """
# 第206-209行
base_models['RandomForest'] = RandomForestRegressor(
    n_estimators=50,     # 从200改为50
    max_depth=5,         # 从8改为5
    min_samples_split=10,
    min_samples_leaf=5, 
    random_state=42, 
    n_jobs=1             # 限制并行度
)
"""
        },
        {
            "问题": "数据类型使用float64",
            "位置": "calculate_enhanced_features()",
            "修复": "转换为float32",
            "代码": """
# 在第195行return之前添加:
for col in features.columns:
    if features[col].dtype == 'float64':
        features[col] = features[col].astype('float32')
return features.fillna(method='ffill').fillna(0)
"""
        },
        {
            "问题": "无内存清理机制",
            "位置": "run_enhanced_walkforward_backtest()",
            "修复": "添加定期内存清理",
            "代码": """
# 在主循环中(第499行附近)每10次迭代后添加:
if i % 10 == 0:
    gc.collect()
    logger.info(f"第{i}次迭代，执行内存清理")
"""
        },
        {
            "问题": "历史记录无限累积",
            "位置": "portfolio_values等列表",
            "修复": "限制历史记录长度",
            "代码": """
# 在记录结果后添加(第604行附近):
if len(portfolio_values) > 500:  # 限制最大长度
    portfolio_values = portfolio_values[-400:]
if len(signal_history) > 5000:
    signal_history = signal_history[-4000:]
"""
        }
    ]
    
    for i, fix in enumerate(fixes, 1):
        print(f"\n{i}. {fix['问题']}")
        print(f"   位置: {fix['位置']}")
        print(f"   修复: {fix['修复']}")
        print(f"   代码修改:")
        print(fix['代码'])
    
    print("\n" + "=" * 60)
    print("快速临时解决方案:")
    print("=" * 60)
    print("""
1. 立即减少股票数量:
   - 打开 bma_walkforward_enhanced.py
   - 找到第1052行: test_tickers = ENHANCED_STOCK_POOL[:30]
   - 改为: test_tickers = ENHANCED_STOCK_POOL[:15]

2. 减少模型复杂度:
   - 找到第206行RandomForestRegressor
   - 将n_estimators=200改为n_estimators=50
   - 将max_depth=8改为max_depth=5

3. 添加内存监控:
   - 在主循环中每10次迭代执行gc.collect()
""")
    
    return fixes

def create_optimized_bma_runner():
    """创建内存优化的BMA运行器"""
    
    optimized_code = '''
# 内存优化版BMA运行器
def run_memory_optimized_bma():
    """内存优化版BMA回测"""
    import gc
    from bma_walkforward_enhanced import EnhancedBMAWalkForward, ENHANCED_STOCK_POOL
    
    # 1. 限制股票数量
    limited_stocks = ENHANCED_STOCK_POOL[:15]  # 只用15只股票
    
    # 2. 创建优化配置的回测器
    backtest = EnhancedBMAWalkForward(
        initial_capital=100000,    # 减少初始资金
        max_positions=10,          # 减少最大持仓
        training_window_months=3,  # 减少训练窗口
        min_training_samples=60    # 减少最小样本
    )
    
    print(f"内存优化设置:")
    print(f"- 股票数量: {len(limited_stocks)}")
    print(f"- 最大持仓: 10")
    print(f"- 训练窗口: 3个月")
    
    try:
        # 3. 运行回测
        results = backtest.run_enhanced_walkforward_backtest(
            tickers=limited_stocks,
            start_date="2023-01-01",  # 缩短时间范围
            end_date="2024-06-01"
        )
        
        print("回测完成!")
        if results and 'performance_metrics' in results:
            metrics = results['performance_metrics']
            print(f"总收益率: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"Sharpe比率: {metrics.get('sharpe_ratio', 0):.3f}")
        
        return results
        
    except Exception as e:
        print(f"回测失败: {e}")
        return None
    finally:
        # 4. 强制清理内存
        gc.collect()

if __name__ == "__main__":
    run_memory_optimized_bma()
'''
    
    with open("D:/trade/run_optimized_bma.py", "w", encoding="utf-8") as f:
        f.write(optimized_code)
    
    print("已创建内存优化运行器: run_optimized_bma.py")

def get_memory_usage():
    """获取当前内存使用情况"""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        print(f"当前进程内存使用:")
        print(f"- 物理内存: {memory_info.rss / 1024 / 1024:.1f} MB")
        print(f"- 虚拟内存: {memory_info.vms / 1024 / 1024:.1f} MB")
        print(f"- 内存占比: {process.memory_percent():.1f}%")
        
        system_memory = psutil.virtual_memory()
        print(f"系统内存:")
        print(f"- 总内存: {system_memory.total / 1024 / 1024 / 1024:.1f} GB")
        print(f"- 可用内存: {system_memory.available / 1024 / 1024 / 1024:.1f} GB")
        print(f"- 使用率: {system_memory.percent:.1f}%")
        
    except ImportError:
        print("需要安装psutil: pip install psutil")
    except Exception as e:
        print(f"获取内存信息失败: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--fix":
            fix_bma_memory_issues()
        elif sys.argv[1] == "--create":
            create_optimized_bma_runner()
        elif sys.argv[1] == "--memory":
            get_memory_usage()
        else:
            print("用法: python bma_memory_fix_simple.py [--fix|--create|--memory]")
    else:
        fix_bma_memory_issues()
        create_optimized_bma_runner()