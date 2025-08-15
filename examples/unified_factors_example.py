#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一因子管理器使用示例
演示如何使用新的统一因子管理系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime
from unified_factor_manager import (
    UnifiedFactorManager, 
    FactorCategory,
    calculate_factor,
    calculate_factors, 
    get_available_factors
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数演示"""
    print("=== 统一因子管理器使用示例 ===\n")
    
    # 1. 创建因子管理器实例
    config_path = "../config/unified_factors_config.json"
    manager = UnifiedFactorManager(config_path)
    
    # 2. 检查引擎状态
    print("1. 引擎状态检查")
    engine_status = manager.get_engine_status()
    for engine, status in engine_status.items():
        if engine != 'cache':
            print(f"   {engine}: 可用={status['available']}, 优先级={status['priority']}")
    print()
    
    # 3. 获取可用因子
    print("2. 可用因子列表")
    all_factors = manager.get_available_factors()
    print(f"   总计因子数: {len(all_factors)}")
    
    # 按类别显示因子
    for category in FactorCategory:
        category_factors = manager.get_available_factors(category)
        if category_factors:
            print(f"   {category.value}: {len(category_factors)}个")
    print()
    
    # 4. 测试因子计算
    test_symbol = "AAPL"
    print(f"3. 测试因子计算 (股票: {test_symbol})")
    
    # 单个因子计算
    if all_factors:
        test_factor = all_factors[0]
        print(f"   计算因子: {test_factor}")
        
        result = manager.calculate_factor(test_factor, test_symbol)
        if result:
            print(f"   结果: {result.value:.4f}")
            print(f"   数据源: {result.data_source.value}")
            print(f"   计算时间: {result.computation_time:.3f}s")
            print(f"   数据质量: {result.data_quality:.2f}")
        else:
            print("   计算失败")
    print()
    
    # 5. 批量因子计算
    print("4. 批量因子计算")
    momentum_factors = manager.get_available_factors(FactorCategory.MOMENTUM)
    if momentum_factors:
        # 选择前3个动量因子
        test_factors = momentum_factors[:3]
        print(f"   测试因子: {test_factors}")
        
        results = manager.calculate_factor_set(test_factors, test_symbol)
        for factor_name, result in results.items():
            print(f"   {factor_name}: {result.value:.4f} (来源: {result.data_source.value})")
    print()
    
    # 6. 使用便捷函数
    print("5. 使用便捷函数")
    
    # 获取可用因子
    available_momentum = get_available_factors("momentum")
    print(f"   动量因子数: {len(available_momentum)}")
    
    # 计算单个因子
    if available_momentum:
        factor_result = calculate_factor(available_momentum[0], test_symbol)
        if factor_result:
            print(f"   {available_momentum[0]}: {factor_result.value:.4f}")
    
    # 批量计算
    if len(available_momentum) >= 2:
        batch_results = calculate_factors(available_momentum[:2], test_symbol)
        print(f"   批量计算了 {len(batch_results)} 个因子")
    print()
    
    # 7. 缓存统计
    print("6. 缓存统计")
    cache_stats = manager.cache_manager.get_stats()
    print(f"   命中次数: {cache_stats['hits']}")
    print(f"   未命中次数: {cache_stats['misses']}")
    print(f"   命中率: {cache_stats['hit_rate']:.2%}")
    print(f"   内存缓存大小: {cache_stats['memory_cache_size']}")
    print()
    
    # 8. 演示因子对比
    print("7. 因子对比演示")
    test_symbols = ["AAPL", "MSFT", "GOOGL"]
    
    if momentum_factors:
        factor_name = momentum_factors[0]
        print(f"   对比因子: {factor_name}")
        
        factor_values = {}
        for symbol in test_symbols:
            result = manager.calculate_factor(factor_name, symbol)
            if result:
                factor_values[symbol] = result.value
        
        for symbol, value in factor_values.items():
            print(f"   {symbol}: {value:.4f}")
    print()
    
    # 9. 清理缓存
    print("8. 清理缓存")
    initial_cache_size = cache_stats['memory_cache_size']
    manager.cleanup_cache()
    final_stats = manager.cache_manager.get_stats()
    print(f"   清理前缓存大小: {initial_cache_size}")
    print(f"   清理后缓存大小: {final_stats['memory_cache_size']}")
    
    print("\n=== 示例完成 ===")

def demonstrate_factor_categories():
    """演示不同类别因子的使用"""
    print("\n=== 因子分类演示 ===")
    
    manager = UnifiedFactorManager()
    test_symbol = "AAPL"
    
    # 演示各类别因子
    categories_to_demo = [
        (FactorCategory.MOMENTUM, "动量因子 - 反映价格趋势和动量效应"),
        (FactorCategory.VALUE, "价值因子 - 反映股票估值水平"),
        (FactorCategory.QUALITY, "质量因子 - 反映公司基本面质量"),
        (FactorCategory.VOLATILITY, "波动率因子 - 反映风险特征"),
        (FactorCategory.LIQUIDITY, "流动性因子 - 反映交易活跃度")
    ]
    
    for category, description in categories_to_demo:
        factors = manager.get_available_factors(category)
        print(f"\n{description}")
        print(f"可用因子数: {len(factors)}")
        
        if factors:
            # 计算第一个因子作为示例
            result = manager.calculate_factor(factors[0], test_symbol)
            if result:
                print(f"示例 - {factors[0]}: {result.value:.4f}")

def demonstrate_engine_comparison():
    """演示不同引擎的对比"""
    print("\n=== 引擎对比演示 ===")
    
    manager = UnifiedFactorManager()
    test_symbol = "AAPL"
    
    # 比较不同引擎计算相同类型的因子
    engines = ['barra', 'polygon', 'autotrader']
    
    for engine in engines:
        if engine in manager.engines:
            print(f"\n{engine.upper()}引擎:")
            engine_factors = manager.get_available_factors(engine=engine)
            print(f"  因子数量: {len(engine_factors)}")
            
            # 计算几个示例因子
            for factor in engine_factors[:3]:
                result = manager.calculate_factor(factor, test_symbol, engine=engine)
                if result:
                    print(f"  {factor}: {result.value:.4f}")

if __name__ == "__main__":
    try:
        main()
        demonstrate_factor_categories()
        demonstrate_engine_comparison()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()