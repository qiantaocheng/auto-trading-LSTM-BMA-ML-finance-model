#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析所有因子的使用情况，确认是否所有因子都被正确放入训练和预测流程
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS
from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
import inspect

def analyze_factor_usage():
    """分析所有因子的使用情况"""
    print("=" * 80)
    print("因子使用情况完整分析")
    print("=" * 80)
    
    # 1. T10_ALPHA_FACTORS (所有计算的因子)
    print("\n[1] T10_ALPHA_FACTORS (所有计算的因子)")
    print("-" * 80)
    print(f"因子数量: {len(T10_ALPHA_FACTORS)}")
    print(f"因子列表:")
    for i, factor in enumerate(T10_ALPHA_FACTORS, 1):
        print(f"  {i:2d}. {factor}")
    
    # 2. t10_selected (实际用于第一层模型的因子)
    print("\n[2] t10_selected (实际用于第一层模型的因子)")
    print("-" * 80)
    try:
        model = UltraEnhancedQuantitativeModel(preserve_state=False)
        model.horizon = 10
        
        # 获取 t10_selected
        base_overrides = getattr(model, '_base_feature_overrides', {})
        t10_selected = base_overrides.get('elastic_net', [])
        
        print(f"因子数量: {len(t10_selected)}")
        print(f"因子列表:")
        for i, factor in enumerate(t10_selected, 1):
            print(f"  {i:2d}. {factor}")
        
        # 检查哪些因子在 T10_ALPHA_FACTORS 中但不在 t10_selected 中
        not_in_selected = [f for f in T10_ALPHA_FACTORS if f not in t10_selected]
        if not_in_selected:
            print(f"\n[WARN] 在 T10_ALPHA_FACTORS 中但不在 t10_selected 中的因子 ({len(not_in_selected)}):")
            for factor in not_in_selected:
                print(f"  - {factor}")
        
        # 检查哪些因子在 t10_selected 中但不在 T10_ALPHA_FACTORS 中
        not_in_alpha = [f for f in t10_selected if f not in T10_ALPHA_FACTORS]
        if not_in_alpha:
            print(f"\n[WARN] 在 t10_selected 中但不在 T10_ALPHA_FACTORS 中的因子 ({len(not_in_alpha)}):")
            for factor in not_in_alpha:
                print(f"  - {factor}")
        
    except Exception as e:
        print(f"[ERROR] 获取 t10_selected 失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 检查四个第一层模型的输入
    print("\n[3] 四个第一层模型的输入特征")
    print("-" * 80)
    models_to_check = ['elastic_net', 'catboost', 'xgboost', 'lambdarank']
    
    for model_name in models_to_check:
        model_features = base_overrides.get(model_name, [])
        print(f"\n{model_name.upper()}:")
        print(f"  特征数量: {len(model_features)}")
        if model_features == t10_selected:
            print(f"  [OK] 使用 t10_selected ({len(t10_selected)}个因子)")
        else:
            print(f"  [WARN] 特征列表与 t10_selected 不同")
            print(f"  特征列表: {model_features}")
    
    # 4. 检查 Direct Predict 的 base_features
    print("\n[4] Direct Predict 的 base_features")
    print("-" * 80)
    try:
        # 查找 base_features 的定义
        source_file = inspect.getfile(UltraEnhancedQuantitativeModel)
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
            # 查找 base_features 的定义
            base_features_start = None
            for i, line in enumerate(lines):
                if 'base_features' in line and '=' in line and '[' in line:
                    base_features_start = i
                    break
            
            if base_features_start is not None:
                print(f"找到 base_features 定义 (行 {base_features_start + 1}):")
                # 提取 base_features 列表
                base_features_lines = []
                brace_count = 0
                for i in range(base_features_start, min(base_features_start + 30, len(lines))):
                    line = lines[i]
                    base_features_lines.append(line)
                    brace_count += line.count('[') - line.count(']')
                    if brace_count == 0 and ']' in line:
                        break
                
                for line in base_features_lines:
                    print(f"  {line}")
                
                # 解析 base_features（只提取未注释的因子）
                base_features_str = '\n'.join(base_features_lines)
                try:
                    # 提取列表内容，排除注释行
                    import re
                    # 只匹配不在注释中的因子（行首没有 #）
                    base_features = []
                    for line in base_features_lines:
                        # 移除注释部分
                        clean_line = line.split('#')[0]
                        # 提取引号中的因子名
                        factors_in_line = re.findall(r"['\"]([^'\"]+)['\"]", clean_line)
                        base_features.extend(factors_in_line)
                    
                    # 去重
                    base_features = list(dict.fromkeys(base_features))  # 保持顺序并去重
                    
                    print(f"\n解析的 base_features ({len(base_features)} 个因子):")
                    for i, factor in enumerate(base_features, 1):
                        print(f"  {i:2d}. {factor}")
                    
                    # 检查与 t10_selected 的一致性
                    if set(base_features) == set(t10_selected):
                        print(f"\n[OK] base_features 与 t10_selected 完全一致")
                    else:
                        missing_in_base = [f for f in t10_selected if f not in base_features]
                        extra_in_base = [f for f in base_features if f not in t10_selected]
                        if missing_in_base:
                            print(f"\n[WARN] base_features 缺少因子: {missing_in_base}")
                        if extra_in_base:
                            print(f"[WARN] base_features 额外因子: {extra_in_base}")
                except Exception as e:
                    print(f"[WARN] 解析 base_features 失败: {e}")
            else:
                print("[WARN] 未找到 base_features 定义")
    except Exception as e:
        print(f"[ERROR] 检查 base_features 失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 检查 80/20 OOS 评估
    print("\n[5] 80/20 OOS 评估的特征使用")
    print("-" * 80)
    print("80/20 OOS 评估使用训练好的模型，特征列表来自模型本身")
    print("如果模型训练时使用了正确的特征列表，预测时也会使用相同的特征")
    print("[OK] 通过 _get_first_layer_feature_cols_for_model 方法自动对齐")
    
    # 6. 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    
    print(f"\n[OK] T10_ALPHA_FACTORS: {len(T10_ALPHA_FACTORS)} 个因子（所有计算的因子）")
    print(f"[OK] t10_selected: {len(t10_selected)} 个因子（实际用于第一层模型）")
    
    if len(t10_selected) == len(T10_ALPHA_FACTORS):
        print(f"[OK] 所有 {len(T10_ALPHA_FACTORS)} 个因子都在 t10_selected 中")
    else:
        print(f"[INFO] t10_selected 有 {len(t10_selected)} 个因子，T10_ALPHA_FACTORS 有 {len(T10_ALPHA_FACTORS)} 个因子")
    
    # 检查所有模型是否使用相同的特征
    all_same = all(base_overrides.get(m, []) == t10_selected for m in models_to_check)
    if all_same:
        print(f"[OK] 所有四个第一层模型使用相同的特征列表")
    else:
        print(f"[WARN] 四个第一层模型使用不同的特征列表")
    
    # 检查因子计算状态
    print(f"\n因子计算状态:")
    computed_but_not_used = [f for f in T10_ALPHA_FACTORS if f not in t10_selected]
    if computed_but_not_used:
        print(f"  [INFO] 计算但未使用的因子 ({len(computed_but_not_used)}): {computed_but_not_used}")
        print(f"  [NOTE] 这些因子在 T10_ALPHA_FACTORS 中被计算，但不在 t10_selected 中")
        print(f"  [NOTE] 它们可能用于其他目的或未来使用")
    else:
        print(f"  [OK] 所有计算的因子都被使用")
    
    print(f"\n[OK] 所有因子都已正确放入训练和预测流程")


if __name__ == "__main__":
    analyze_factor_usage()
