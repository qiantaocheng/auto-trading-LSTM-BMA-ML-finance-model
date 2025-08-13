#!/usr/bin/env python3
"""
验证所有Alpha因子整合到enhanced_alpha_strategies.py的简化测试
"""

import pandas as pd
import numpy as np
import yaml
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_alpha_config():
    """测试配置文件中的因子数量"""
    
    print("🔍 检查Alpha配置文件...")
    
    try:
        with open('alphas_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        alphas = config.get('alphas', [])
        total_factors = len(alphas)
        
        print(f"📊 配置文件统计:")
        print(f"  总配置因子数: {total_factors}")
        
        # 按类型分类统计
        factor_types = {}
        for alpha in alphas:
            kind = alpha.get('kind', 'unknown')
            factor_types[kind] = factor_types.get(kind, 0) + 1
        
        print(f"  因子类型分布:")
        for kind, count in sorted(factor_types.items()):
            print(f"    {kind}: {count}个")
        
        # 检查weight_hint
        factors_with_hints = [alpha for alpha in alphas if 'weight_hint' in alpha]
        print(f"  包含weight_hint的因子: {len(factors_with_hints)}/{total_factors}")
        
        return True, total_factors
        
    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        return False, 0

def test_alpha_engine_import():
    """测试Alpha引擎导入"""
    
    print("\n🔧 测试Alpha引擎导入...")
    
    try:
        from enhanced_alpha_strategies import AlphaStrategiesEngine
        print("✅ AlphaStrategiesEngine导入成功")
        
        # 尝试初始化
        engine = AlphaStrategiesEngine("alphas_config.yaml")
        print("✅ Alpha引擎初始化成功")
        
        # 检查注册的因子函数
        alpha_functions = engine.alpha_functions
        print(f"📈 已注册的因子函数: {len(alpha_functions)}个")
        
        # 显示所有注册的因子类型
        print("注册的因子类型:")
        for kind in sorted(alpha_functions.keys()):
            if alpha_functions[kind] is not None:
                print(f"  ✓ {kind}")
            else:
                print(f"  ⚠ {kind} (特殊处理)")
        
        return True, engine, len(alpha_functions)
        
    except Exception as e:
        print(f"❌ Alpha引擎测试失败: {e}")
        return False, None, 0

def test_factor_methods():
    """测试因子计算方法是否存在"""
    
    print("\n🧪 检查因子计算方法...")
    
    success, engine, _ = test_alpha_engine_import()
    if not success:
        return False
    
    # 测试的关键因子方法
    test_methods = [
        '_compute_momentum',
        '_compute_reversal', 
        '_compute_volatility',
        '_compute_gross_margin',
        '_compute_operating_profitability',
        '_compute_total_accruals',
        '_compute_piotroski_score',
        '_compute_qmj_score'
    ]
    
    existing_methods = []
    missing_methods = []
    
    for method in test_methods:
        if hasattr(engine, method):
            existing_methods.append(method)
            print(f"  ✅ {method}")
        else:
            missing_methods.append(method)
            print(f"  ❌ {method}")
    
    print(f"\n📊 方法检查结果:")
    print(f"  存在的方法: {len(existing_methods)}/{len(test_methods)}")
    print(f"  缺失的方法: {len(missing_methods)}")
    
    return len(missing_methods) == 0

def create_simple_test_data():
    """创建简单的测试数据"""
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    tickers = ['TEST_001', 'TEST_002', 'TEST_003']
    
    data = []
    for ticker in tickers:
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, len(dates))))
        volumes = np.random.lognormal(10, 0.5, len(dates))
        
        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'ticker': ticker,
                'Close': prices[i],
                'High': prices[i] * 1.02,
                'Low': prices[i] * 0.98,
                'volume': volumes[i],
                'amount': prices[i] * volumes[i],
                'COUNTRY': 'US',
                'SECTOR': 'TECH',
                'SUBINDUSTRY': 'SOFTWARE'
            })
    
    return pd.DataFrame(data)

def test_basic_computation():
    """测试基础计算功能"""
    
    print("\n⚡ 测试基础因子计算...")
    
    success, engine, _ = test_alpha_engine_import()
    if not success:
        return False
    
    # 创建测试数据
    df = create_simple_test_data()
    print(f"  测试数据: {df.shape[0]}行, {df['ticker'].nunique()}只股票")
    
    try:
        # 尝试计算Alpha因子
        result_df = engine.compute_all_alphas(df)
        
        # 统计结果
        original_columns = ['date', 'ticker', 'Close', 'High', 'Low', 'volume', 'amount', 'COUNTRY', 'SECTOR', 'SUBINDUSTRY']
        alpha_columns = [col for col in result_df.columns if col not in original_columns]
        
        print(f"  ✅ 因子计算完成")
        print(f"  原始列数: {len(original_columns)}")
        print(f"  新增Alpha列: {len(alpha_columns)}")
        print(f"  总列数: {result_df.shape[1]}")
        
        # 检查计算成功的因子
        successful_factors = []
        failed_factors = []
        
        for col in alpha_columns:
            if not result_df[col].isna().all():
                successful_factors.append(col)
            else:
                failed_factors.append(col)
        
        print(f"  成功计算: {len(successful_factors)}个因子")
        print(f"  计算失败: {len(failed_factors)}个因子")
        
        if len(successful_factors) > 0:
            print(f"  成功率: {len(successful_factors)/len(alpha_columns)*100:.1f}%")
            
            # 显示部分成功的因子
            print("  成功计算的因子示例:")
            for factor in successful_factors[:5]:
                non_null_count = result_df[factor].notna().sum()
                print(f"    {factor}: {non_null_count}个有效值")
        
        return len(successful_factors) > 0
        
    except Exception as e:
        print(f"  ❌ 计算失败: {e}")
        return False

def main():
    """主测试函数"""
    
    print("🚀 开始验证Alpha因子整合")
    print("=" * 50)
    
    # 1. 测试配置文件
    config_ok, total_factors = test_alpha_config()
    
    # 2. 测试引擎导入
    import_ok, engine, registered_functions = test_alpha_engine_import()
    
    # 3. 测试方法存在性
    methods_ok = test_factor_methods()
    
    # 4. 测试基础计算
    computation_ok = test_basic_computation()
    
    # 总结结果
    print("\n" + "=" * 50)
    print("📋 整合验证结果:")
    print(f"  配置文件读取: {'✅' if config_ok else '❌'}")
    print(f"  引擎导入初始化: {'✅' if import_ok else '❌'}")
    print(f"  因子方法检查: {'✅' if methods_ok else '❌'}")
    print(f"  基础计算测试: {'✅' if computation_ok else '❌'}")
    
    all_passed = config_ok and import_ok and methods_ok and computation_ok
    
    if all_passed:
        print(f"\n🎉 所有测试通过！")
        print(f"  配置因子总数: {total_factors}")
        print(f"  注册函数数量: {registered_functions}")
        print(f"  系统已准备就绪，支持机器学习动态权重分配")
    else:
        print(f"\n⚠️ 部分测试失败，需要进一步检查")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
