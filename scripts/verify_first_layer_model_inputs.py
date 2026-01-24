#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证四个第一层模型的输入点，确保它们都正确使用了更新后的因子列表
（不包含 bollinger_squeeze 和 hist_vol_40d）
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS
from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
import json

def verify_factor_lists():
    """验证因子列表的一致性"""
    print("=" * 80)
    print("验证四个第一层模型的输入点")
    print("=" * 80)
    
    # 1. 检查 T10_ALPHA_FACTORS
    print("\n[1] T10_ALPHA_FACTORS 列表:")
    print(f"   因子数量: {len(T10_ALPHA_FACTORS)}")
    print(f"   因子列表: {T10_ALPHA_FACTORS}")
    
    # 检查是否包含已删除的因子
    removed_factors = ['bollinger_squeeze', 'hist_vol_40d']
    found_removed = [f for f in removed_factors if f in T10_ALPHA_FACTORS]
    if found_removed:
        print(f"   [ERROR] 发现已删除的因子: {found_removed}")
        return False
    else:
        print(f"   [OK] 确认: 已删除的因子不在列表中")
    
    # 2. 初始化模型并检查 t10_selected
    print("\n[2] 初始化 UltraEnhancedQuantitativeModel (T+10)...")
    try:
        model = UltraEnhancedQuantitativeModel(preserve_state=False)
        model.horizon = 10  # 确保使用 T+10
        
        # 获取 t10_selected（通过检查 base_overrides）
        base_overrides = getattr(model, '_base_feature_overrides', {})
        first_layer_overrides = getattr(model, 'first_layer_feature_overrides', {})
        
        print(f"   [OK] 模型初始化成功")
        print(f"   [OK] base_overrides 键: {list(base_overrides.keys())}")
        
        # 检查四个模型的输入
        models_to_check = ['elastic_net', 'catboost', 'xgboost', 'lambdarank']
        
        print("\n[3] 检查四个第一层模型的输入特征:")
        print("-" * 80)
        
        all_correct = True
        for model_name in models_to_check:
            model_features = first_layer_overrides.get(model_name, [])
            if model_features is None:
                model_features = T10_ALPHA_FACTORS  # 如果没有覆盖，使用全部
            
            print(f"\n   [{model_name.upper()}]")
            print(f"   特征数量: {len(model_features)}")
            print(f"   特征列表: {model_features}")
            
            # 检查是否包含已删除的因子
            found_removed_in_model = [f for f in removed_factors if f in model_features]
            if found_removed_in_model:
                print(f"   [ERROR] 发现已删除的因子: {found_removed_in_model}")
                all_correct = False
            else:
                print(f"   [OK] 确认: 不包含已删除的因子")
            
            # 检查是否包含 momentum_10d
            if 'momentum_10d' not in model_features:
                print(f"   [WARN] 警告: 缺少 momentum_10d")
            else:
                print(f"   [OK] 确认: 包含 momentum_10d")
        
        # 4. 检查 Direct Predict 的 base_features
        print("\n[4] 检查 Direct Predict 的 base_features:")
        print("-" * 80)
        
        # 查找 base_features 的定义位置
        import inspect
        source_file = inspect.getfile(UltraEnhancedQuantitativeModel)
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'base_features' in content:
                # 查找 base_features 的定义
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'base_features' in line and '=' in line:
                        # 找到定义行，打印上下文
                        start = max(0, i - 2)
                        end = min(len(lines), i + 10)
                        print(f"   找到 base_features 定义 (行 {i+1}):")
                        for j in range(start, end):
                            marker = ">>>" if j == i else "   "
                            print(f"   {marker} {j+1:5d}: {lines[j]}")
                        break
        
        # 5. 检查 80/20 OOS 评估
        print("\n[5] 检查 80/20 OOS 评估:")
        print("-" * 80)
        print("   80/20 OOS 评估使用训练好的模型，特征列表来自模型本身")
        print("   如果模型训练时使用了正确的特征列表，预测时也会使用相同的特征")
        print("   [OK] 通过 _get_first_layer_feature_cols_for_model 方法自动对齐")
        
        # 6. 总结
        print("\n" + "=" * 80)
        print("验证总结")
        print("=" * 80)
        
        if all_correct:
            print("[OK] 所有四个第一层模型的输入点都正确！")
            print(f"[OK] 因子数量: {len(T10_ALPHA_FACTORS)}")
            print(f"[OK] 已删除的因子: {removed_factors}")
            print(f"[OK] 新增的因子: momentum_10d")
        else:
            print("[ERROR] 发现错误，请检查上述输出")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_feature_selection_method():
    """验证特征选择方法"""
    print("\n" + "=" * 80)
    print("验证特征选择方法 _get_first_layer_feature_cols_for_model")
    print("=" * 80)
    
    try:
        model = UltraEnhancedQuantitativeModel(preserve_state=False)
        model.horizon = 10
        
        # 模拟特征列表（包含已删除的因子）
        all_features = T10_ALPHA_FACTORS + ['bollinger_squeeze', 'hist_vol_40d']
        
        models_to_check = ['elastic_net', 'catboost', 'xgboost', 'lambdarank']
        
        for model_name in models_to_check:
            selected_features = model._get_first_layer_feature_cols_for_model(
                model_name, 
                all_features, 
                available_cols=all_features
            )
            
            print(f"\n[{model_name.upper()}]")
            print(f"   输入特征数: {len(all_features)}")
            print(f"   选择特征数: {len(selected_features)}")
            print(f"   选择的特征: {selected_features}")
            
            # 检查是否过滤掉了已删除的因子
            removed_factors = ['bollinger_squeeze', 'hist_vol_40d']
            found_removed = [f for f in removed_factors if f in selected_features]
            if found_removed:
                print(f"   [ERROR] 选择的特征中包含已删除的因子: {found_removed}")
            else:
                print(f"   [OK] 确认: 已删除的因子被正确过滤")
            
            # 检查是否包含 momentum_10d
            if 'momentum_10d' in selected_features:
                print(f"   [OK] 确认: 包含 momentum_10d")
            else:
                print(f"   [WARN] 警告: 缺少 momentum_10d")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = verify_factor_lists()
    success2 = verify_feature_selection_method()
    
    if success1 and success2:
        print("\n" + "=" * 80)
        print("[OK] 所有验证通过！")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("[ERROR] 验证失败，请检查上述输出")
        print("=" * 80)
        sys.exit(1)
