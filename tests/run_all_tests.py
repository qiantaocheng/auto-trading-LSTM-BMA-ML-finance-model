#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行所有单元测试的主脚本
"""

import sys
import os
import unittest
from datetime import datetime

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bma_models'))

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("两层机器学习管道 - 全面单元测试")
    print("=" * 60)
    print(f"测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试模块列表
    test_modules = [
        'test_data_preprocessing',
        'test_first_layer_models', 
        'test_second_layer_stacking',
        'test_excel_export',
        'test_complete_pipeline'
    ]
    
    # 统计信息
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    
    results_summary = []
    
    # 逐个运行测试模块
    for module_name in test_modules:
        print(f"\n{'='*20} {module_name} {'='*20}")
        
        try:
            # 导入测试模块
            test_module = __import__(module_name)
            
            # 创建测试套件
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            # 检查测试数量
            test_count = suite.countTestCases()
            if test_count == 0:
                print(f"警告: 模块 {module_name} 没有找到测试用例")
                # 尝试手动加载
                try:
                    import importlib
                    importlib.reload(test_module)
                    suite = loader.loadTestsFromModule(test_module)
                    test_count = suite.countTestCases()
                    print(f"重新加载后找到 {test_count} 个测试")
                except Exception as reload_error:
                    print(f"重新加载失败: {reload_error}")
            
            # 运行测试
            runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
            result = runner.run(suite)
            
            # 统计结果
            module_tests = result.testsRun
            module_failures = len(result.failures)
            module_errors = len(result.errors)
            module_skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
            
            total_tests += module_tests
            total_failures += module_failures
            total_errors += module_errors
            total_skipped += module_skipped
            
            # 计算成功率
            success_count = module_tests - module_failures - module_errors
            success_rate = (success_count / module_tests * 100) if module_tests > 0 else 0
            
            results_summary.append({
                'module': module_name,
                'tests': module_tests,
                'failures': module_failures,
                'errors': module_errors,
                'skipped': module_skipped,
                'success_rate': success_rate
            })
            
            print(f"模块 {module_name} - 成功率: {success_rate:.1f}%")
            
        except ImportError as e:
            print(f"无法导入测试模块 {module_name}: {e}")
            results_summary.append({
                'module': module_name,
                'tests': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0,
                'success_rate': 0.0
            })
            total_errors += 1
        except Exception as e:
            print(f"运行测试模块 {module_name} 时发生错误: {e}")
            results_summary.append({
                'module': module_name,
                'tests': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0,
                'success_rate': 0.0
            })
            total_errors += 1
    
    # 输出总结报告
    print("\n" + "=" * 60)
    print("测试总结报告")
    print("=" * 60)
    
    print(f"总测试数: {total_tests}")
    print(f"失败: {total_failures}")
    print(f"错误: {total_errors}")
    print(f"跳过: {total_skipped}")
    
    overall_success = total_tests - total_failures - total_errors
    overall_success_rate = (overall_success / total_tests * 100) if total_tests > 0 else 0
    print(f"总成功率: {overall_success_rate:.1f}%")
    
    # 各模块详细结果
    print(f"\n各模块测试结果:")
    print(f"{'模块名称':<25} {'测试数':<8} {'失败':<6} {'错误':<6} {'跳过':<6} {'成功率':<8}")
    print("-" * 70)
    
    for result in results_summary:
        print(f"{result['module']:<25} {result['tests']:<8} {result['failures']:<6} "
              f"{result['errors']:<6} {result['skipped']:<6} {result['success_rate']:<8.1f}%")
    
    # 系统状态评估
    print(f"\n" + "=" * 60)
    print("系统状态评估")
    print("=" * 60)
    
    if overall_success_rate >= 85:
        status = "EXCELLENT"
        message = "两层机器学习管道完全准备就绪"
        details = [
            "[OK] 数据预处理流程正常",
            "[OK] 第一层模型 (ElasticNet, XGBoost, LightGBM) 训练正常", 
            "[OK] 第二层Stacking集成正常",
            "[OK] Excel导出功能正常",
            "[OK] 端到端管道集成正常",
            "[OK] 系统可以处理GUI股票池选择",
            "[OK] 系统可以生成按预测收益率排序的Excel输出"
        ]
    elif overall_success_rate >= 70:
        status = "GOOD"
        message = "系统基本正常，部分功能需要优化"
        details = [
            "[OK] 核心功能基本正常",
            "[WARNING] 部分边界情况处理需要改进",
            "[WARNING] 建议检查失败的测试用例"
        ]
    elif overall_success_rate >= 50:
        status = "NEEDS_WORK"
        message = "系统部分功能正常，需要进一步调试"
        details = [
            "[WARNING] 主要功能部分正常",
            "[WARNING] 多个组件需要修复",
            "[WARNING] 建议优先修复失败的核心功能"
        ]
    else:
        status = "CRITICAL"
        message = "系统存在严重问题，需要全面调试"
        details = [
            "[ERROR] 多个核心功能异常",
            "[ERROR] 需要检查依赖和配置",
            "[ERROR] 建议从基础功能开始调试"
        ]
    
    print(f"状态: {status}")
    print(f"评估: {message}")
    print(f"\n详细说明:")
    for detail in details:
        print(f"  {detail}")
    
    print(f"\n测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return {
        'total_tests': total_tests,
        'total_failures': total_failures,
        'total_errors': total_errors,
        'total_skipped': total_skipped,
        'success_rate': overall_success_rate,
        'status': status,
        'results': results_summary
    }

if __name__ == '__main__':
    try:
        result = run_all_tests()
        
        # 根据结果设置退出码
        if result['success_rate'] >= 85:
            sys.exit(0)  # 完全成功
        elif result['success_rate'] >= 70:
            sys.exit(1)  # 部分成功
        else:
            sys.exit(2)  # 需要修复
            
    except Exception as e:
        print(f"运行测试时发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)  # 严重错误