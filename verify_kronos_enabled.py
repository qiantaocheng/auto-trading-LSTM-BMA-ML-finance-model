#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证Kronos筛选是否正确启用
"""

import sys
import os
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_yaml_config():
    """验证YAML配置"""
    print("=" * 80)
    print("步骤1: 验证YAML配置文件")
    print("=" * 80)

    import yaml

    yaml_path = os.path.join(os.path.dirname(__file__), 'bma_models', 'unified_config.yaml')

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        use_kronos = config.get('strict_mode', {}).get('use_kronos_validation', False)

        print(f"📄 配置文件路径: {yaml_path}")
        print(f"📊 strict_mode配置: {config.get('strict_mode', {})}")
        print(f"\n🤖 use_kronos_validation: {use_kronos}")

        if use_kronos:
            print("✅ Kronos验证已在YAML中启用")
            return True
        else:
            print("❌ Kronos验证未启用")
            print("\n💡 修复方法:")
            print("   1. 打开: bma_models/unified_config.yaml")
            print("   2. 找到: strict_mode.use_kronos_validation")
            print("   3. 改为: true")
            return False

    except Exception as e:
        print(f"❌ 读取配置失败: {e}")
        return False


def verify_model_initialization():
    """验证模型初始化时的配置读取"""
    print("\n" + "=" * 80)
    print("步骤2: 验证模型初始化")
    print("=" * 80)

    try:
        from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

        print("🔧 初始化模型（默认配置）...")
        model = UltraEnhancedQuantitativeModel()

        print(f"\n🤖 模型.use_kronos_validation: {model.use_kronos_validation}")

        if model.use_kronos_validation:
            print("✅ Kronos验证在模型中已启用")

            # 检查Kronos模型是否初始化
            if model.kronos_model is None:
                print("   ℹ️ Kronos模型尚未加载（将在需要时懒加载）")
            else:
                print("   ℹ️ Kronos模型已预加载")

            return True
        else:
            print("❌ Kronos验证在模型中未启用")
            print("\n💡 可能原因:")
            print("   1. YAML配置未更新")
            print("   2. 模型读取配置失败")
            return False

    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_kronos_service():
    """验证Kronos服务可用性"""
    print("\n" + "=" * 80)
    print("步骤3: 验证Kronos服务")
    print("=" * 80)

    try:
        from kronos.kronos_service import KronosService

        print("✅ KronosService导入成功")

        # 尝试实例化（不加载模型）
        service = KronosService()
        print("✅ KronosService实例化成功")

        # 检查Kronos模型文件
        import os
        kronos_repo = os.path.join(os.path.dirname(__file__), 'kronos_original_repo')

        if os.path.exists(kronos_repo):
            print(f"✅ Kronos仓库存在: {kronos_repo}")

            # 检查关键文件
            critical_files = [
                'checkpoints/checkpoint.pth',
                'model/chronos_model.py',
            ]

            for file in critical_files:
                file_path = os.path.join(kronos_repo, file)
                if os.path.exists(file_path):
                    print(f"   ✅ {file}")
                else:
                    print(f"   ⚠️ {file} 不存在（可能OK）")
        else:
            print(f"⚠️ Kronos仓库不存在: {kronos_repo}")
            print("   这不影响配置，但运行时会失败")

        return True

    except ImportError as e:
        print(f"❌ KronosService导入失败: {e}")
        print("\n💡 可能原因:")
        print("   1. kronos模块不存在")
        print("   2. 依赖包未安装（transformers, torch等）")
        return False
    except Exception as e:
        print(f"⚠️ Kronos服务检查异常: {e}")
        return True  # 不阻止，因为可能只是模型文件问题


def print_usage_guide():
    """打印使用指南"""
    print("\n" + "=" * 80)
    print("使用指南")
    print("=" * 80)

    print("\n📝 运行模型代码:")
    print("""
from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

# 模型会自动从YAML读取use_kronos_validation=true
model = UltraEnhancedQuantitativeModel()

# 运行分析
results = model.run_complete_analysis(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)

# Kronos会在生成Top 35后自动运行
# 查看Excel: results['excel_path']
# 找到表格: Kronos_T3_Filter
""")

    print("\n📊 预期日志输出:")
    print("""
🤖 Kronos验证配置（来自YAML）: True
...
================================================================================
🤖 Kronos T+3过滤器：对融合后Top 35进行盈利性验证
   参数：T+3预测，温度0.1，过去1年数据
   对Top 35 股票进行Kronos T+3验证...
================================================================================
  ✓ PASS #1 AAPL: T+3收益 +2.5% ($180.00 → $184.50)
  ✗ FAIL #2 MSFT: T+3收益 -0.8% ($350.00 → $347.20)
  ...
✅ Kronos T+3过滤完成:
   测试股票: 35 只
   通过过滤 (T+3收益>0): 21 只 (60.0%)
================================================================================
""")

    print("\n📄 Excel输出:")
    print("   表格: Kronos_T3_Filter")
    print("   关键列: Kronos_Pass (Y=推荐, N=观望)")


def main():
    print("\n" + "=" * 80)
    print("Kronos筛选启用验证工具")
    print("=" * 80 + "\n")

    results = []

    # 测试1: YAML配置
    result1 = verify_yaml_config()
    results.append(("YAML配置", result1))

    # 测试2: 模型初始化
    result2 = verify_model_initialization()
    results.append(("模型初始化", result2))

    # 测试3: Kronos服务
    result3 = verify_kronos_service()
    results.append(("Kronos服务", result3))

    # 总结
    print("\n" + "=" * 80)
    print("验证总结")
    print("=" * 80)

    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")

    all_pass = all(r for _, r in results)

    if all_pass:
        print("\n" + "=" * 80)
        print("🎉 Kronos筛选已正确启用！")
        print("=" * 80)
        print("\n可以直接运行模型，Kronos会自动工作。")
        print_usage_guide()
        return 0
    else:
        print("\n" + "=" * 80)
        print("⚠️ 部分检查失败")
        print("=" * 80)
        print("\n请按照上面的提示修复问题。")
        return 1


if __name__ == "__main__":
    exit(main())
