#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析Direct Predict和训练时使用的特征是否一致
检查需要的feature能否被正确计算调用
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def analyze_direct_predict_vs_training_features():
    """分析Direct Predict和训练时的特征一致性"""
    print("=" * 80)
    print("Direct Predict vs 训练特征一致性分析")
    print("=" * 80)
    
    # 1. 获取训练时使用的特征
    print("\n[1] 获取训练时使用的特征...")
    try:
        from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
        
        model = UltraEnhancedQuantitativeModel(preserve_state=False)
        model.horizon = 10
        
        # 获取t10_selected（训练时使用的特征）
        t10_selected = model._base_feature_overrides.get('elastic_net', [])
        
        print(f"  训练时使用的特征 (t10_selected): {len(t10_selected)} 个")
        print(f"  {t10_selected}")
        
    except Exception as e:
        print(f"  [ERROR] 无法获取训练特征: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 获取Direct Predict使用的特征
    print("\n[2] 获取Direct Predict使用的特征...")
    try:
        # 从app.py的_direct_predict_snapshot方法中提取base_features定义
        # 这是Direct Predict在predict_with_snapshot中使用的特征
        direct_predict_base_features = [
            'momentum_10d',
            'ivol_30', 'near_52w_high', 'rsi_21', 'vol_ratio_30d',
            'trend_r2_60', 'liquid_momentum', 'obv_momentum_40d', 'atr_ratio',
            'ret_skew_30d', 'price_ma60_deviation', 'blowoff_ratio_30d',
            'feat_vol_price_div_30d',
            '5_days_reversal',
            'downside_beta_ewm_21',
        ]
        
        print(f"  Direct Predict使用的特征 (base_features): {len(direct_predict_base_features)} 个")
        print(f"  {direct_predict_base_features}")
        
    except Exception as e:
        print(f"  [ERROR] 无法获取Direct Predict特征: {e}")
        return
    
    # 3. 获取T10_ALPHA_FACTORS（Simple17FactorEngine计算的所有因子）
    print("\n[3] 获取T10_ALPHA_FACTORS（Simple17FactorEngine计算的所有因子）...")
    try:
        from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS
        
        print(f"  T10_ALPHA_FACTORS: {len(T10_ALPHA_FACTORS)} 个")
        print(f"  {list(T10_ALPHA_FACTORS)}")
        
    except Exception as e:
        print(f"  [ERROR] 无法获取T10_ALPHA_FACTORS: {e}")
        return
    
    # 4. 对比特征列表
    print("\n" + "=" * 80)
    print("[4] 特征一致性对比")
    print("=" * 80)
    
    t10_selected_set = set(t10_selected)
    direct_predict_set = set(direct_predict_base_features)
    t10_alpha_factors_set = set(T10_ALPHA_FACTORS)
    
    # 4.1 训练 vs Direct Predict
    print(f"\n[4.1] 训练特征 vs Direct Predict特征:")
    if t10_selected_set == direct_predict_set:
        print(f"  [OK] 完全一致！")
        print(f"  共同特征: {len(t10_selected_set)} 个")
    else:
        missing_in_direct = t10_selected_set - direct_predict_set
        extra_in_direct = direct_predict_set - t10_selected_set
        
        if missing_in_direct:
            print(f"  [ERROR] Direct Predict缺少的特征: {missing_in_direct}")
        if extra_in_direct:
            print(f"  [WARN] Direct Predict额外的特征: {extra_in_direct}")
        if not missing_in_direct and not extra_in_direct:
            print(f"  [OK] 特征一致（顺序可能不同）")
    
    # 4.2 训练特征是否都在T10_ALPHA_FACTORS中
    print(f"\n[4.2] 训练特征是否都能被Simple17FactorEngine计算:")
    missing_in_t10_alpha = t10_selected_set - t10_alpha_factors_set
    
    if not missing_in_t10_alpha:
        print(f"  [OK] 所有训练特征都在T10_ALPHA_FACTORS中，可以被计算")
    else:
        print(f"  [ERROR] 以下训练特征不在T10_ALPHA_FACTORS中，无法被计算:")
        for feat in missing_in_t10_alpha:
            print(f"    - {feat}")
    
    # 4.3 Direct Predict特征是否都在T10_ALPHA_FACTORS中
    print(f"\n[4.3] Direct Predict特征是否都能被Simple17FactorEngine计算:")
    missing_in_t10_alpha_dp = direct_predict_set - t10_alpha_factors_set
    
    if not missing_in_t10_alpha_dp:
        print(f"  [OK] 所有Direct Predict特征都在T10_ALPHA_FACTORS中，可以被计算")
    else:
        print(f"  [ERROR] 以下Direct Predict特征不在T10_ALPHA_FACTORS中，无法被计算:")
        for feat in missing_in_t10_alpha_dp:
            print(f"    - {feat}")
    
    # 5. 检查_get_first_layer_feature_cols_for_model方法
    print("\n" + "=" * 80)
    print("[5] 检查_get_first_layer_feature_cols_for_model方法")
    print("=" * 80)
    
    print(f"\n[5.1] 测试每个模型的特征选择:")
    test_features = list(T10_ALPHA_FACTORS)  # 假设所有T10_ALPHA_FACTORS都可用
    
    for model_name in ['elastic_net', 'xgboost', 'catboost', 'lambdarank']:
        try:
            selected_cols = model._get_first_layer_feature_cols_for_model(
                model_name, 
                test_features, 
                available_cols=test_features
            )
            print(f"\n  {model_name}:")
            print(f"    选择的特征数: {len(selected_cols)}")
            print(f"    特征列表: {selected_cols}")
            
            # 检查是否与t10_selected一致
            selected_set = set(selected_cols)
            if selected_set == t10_selected_set:
                print(f"    [OK] 与训练特征一致")
            else:
                missing = t10_selected_set - selected_set
                extra = selected_set - t10_selected_set
                if missing:
                    print(f"    [WARN] 缺少: {missing}")
                if extra:
                    print(f"    [WARN] 额外: {extra}")
                    
        except Exception as e:
            print(f"  [ERROR] {model_name} 特征选择失败: {e}")
    
    # 6. 验证Simple17FactorEngine能否计算所有需要的特征
    print("\n" + "=" * 80)
    print("[6] 验证Simple17FactorEngine能否计算所有需要的特征")
    print("=" * 80)
    
    print(f"\n[6.1] 检查Simple17FactorEngine的计算方法:")
    try:
        from bma_models.simple_25_factor_engine import Simple17FactorEngine
        
        # 检查每个因子是否有对应的计算方法
        engine = Simple17FactorEngine(lookback_days=280, mode='predict', horizon=10)
        
        missing_methods = []
        for factor in t10_selected:
            # 检查是否有对应的计算方法
            method_name = f"_calculate_{factor}"
            if not hasattr(engine, method_name):
                # 有些因子可能使用通用方法，检查是否在alpha_factors中
                if factor not in T10_ALPHA_FACTORS:
                    missing_methods.append(factor)
        
        if not missing_methods:
            print(f"  [OK] 所有训练特征都有对应的计算方法")
        else:
            print(f"  [WARN] 以下特征可能没有对应的计算方法:")
            for feat in missing_methods:
                print(f"    - {feat}")
                # 检查是否在T10_ALPHA_FACTORS中
                if feat in T10_ALPHA_FACTORS:
                    print(f"      (但在T10_ALPHA_FACTORS中，可能使用通用方法)")
        
    except Exception as e:
        print(f"  [ERROR] 无法验证计算方法: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. 总结
    print("\n" + "=" * 80)
    print("[7] 总结")
    print("=" * 80)
    
    all_consistent = True
    issues = []
    
    # 检查1: 训练 vs Direct Predict
    if t10_selected_set != direct_predict_set:
        all_consistent = False
        issues.append("训练特征和Direct Predict特征不一致")
    
    # 检查2: 训练特征是否都能被计算
    if missing_in_t10_alpha:
        all_consistent = False
        issues.append(f"训练特征中有{len(missing_in_t10_alpha)}个无法被Simple17FactorEngine计算")
    
    # 检查3: Direct Predict特征是否都能被计算
    if missing_in_t10_alpha_dp:
        all_consistent = False
        issues.append(f"Direct Predict特征中有{len(missing_in_t10_alpha_dp)}个无法被Simple17FactorEngine计算")
    
    if all_consistent:
        print(f"\n[结论] [OK] 所有特征一致且可以被正确计算")
        print(f"  - 训练特征: {len(t10_selected)} 个")
        print(f"  - Direct Predict特征: {len(direct_predict_base_features)} 个")
        print(f"  - T10_ALPHA_FACTORS: {len(T10_ALPHA_FACTORS)} 个")
        print(f"  - 所有需要的特征都能被Simple17FactorEngine计算")
    else:
        print(f"\n[结论] [WARN] 发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
        print(f"\n建议:")
        print(f"  1. 检查训练特征和Direct Predict特征的定义")
        print(f"  2. 确保所有特征都在T10_ALPHA_FACTORS中")
        print(f"  3. 验证Simple17FactorEngine能计算所有需要的特征")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_direct_predict_vs_training_features()
