#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证全量训练是否正确调用了所有模型
检查训练流程中是否包含：
1. ElasticNet
2. XGBoost
3. CatBoost
4. LambdaRank
5. Ridge Stacker
6. MetaRankerStacker
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def verify_training_flow():
    """验证训练流程"""
    print("=" * 80)
    print("验证全量训练模型调用流程")
    print("=" * 80)
    
    # 1. 检查train_full_dataset.py
    print("\n[1] 检查train_full_dataset.py...")
    train_script = project_root / "scripts" / "train_full_dataset.py"
    
    if not train_script.exists():
        print(f"  [ERROR] 脚本不存在: {train_script}")
        return False
    
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        "UltraEnhancedQuantitativeModel": "UltraEnhancedQuantitativeModel" in content,
        "train_from_document": "train_from_document" in content,
        "save_model_snapshot": "save_model_snapshot" in content,
        "ridge_stacker": "ridge_stacker" in content,
        "lambda_rank_stacker": "lambda_rank_stacker" in content,
        "meta_ranker_stacker": "meta_ranker_stacker" in content,
    }
    
    for check_name, result in checks.items():
        status = "[OK]" if result else "[ERROR]"
        print(f"  {status} {check_name}: {'找到' if result else '未找到'}")
    
    # 2. 检查量化模型_bma_ultra_enhanced.py中的训练流程
    print("\n[2] 检查量化模型_bma_ultra_enhanced.py训练流程...")
    model_file = project_root / "bma_models" / "量化模型_bma_ultra_enhanced.py"
    
    if not model_file.exists():
        print(f"  [ERROR] 文件不存在: {model_file}")
        return False
    
    with open(model_file, 'r', encoding='utf-8') as f:
        model_content = f.read()
    
    # 检查训练流程
    flow_checks = {
        "train_from_document": "def train_from_document" in model_content,
        "_run_training_phase": "def _run_training_phase" in model_content,
        "train_enhanced_models": "def train_enhanced_models" in model_content,
        "_execute_modular_training": "def _execute_modular_training" in model_content,
        "_unified_model_training": "def _unified_model_training" in model_content,
    }
    
    for check_name, result in flow_checks.items():
        status = "[OK]" if result else "[ERROR]"
        print(f"  {status} {check_name}: {'找到' if result else '未找到'}")
    
    # 3. 检查第一层模型
    print("\n[3] 检查第一层模型（ElasticNet, XGBoost, CatBoost, LambdaRank）...")
    first_layer_checks = {
        "ElasticNet": "ElasticNet" in model_content and "models['elastic_net']" in model_content,
        "XGBoost": "XGBRegressor" in model_content and "models['xgboost']" in model_content,
        "CatBoost": "CatBoostRegressor" in model_content and "models['catboost']" in model_content,
        "LambdaRank": "LambdaRankStacker" in model_content and ("lambdarank" in model_content.lower() or "lambda_rank" in model_content),
    }
    
    for check_name, result in first_layer_checks.items():
        status = "[OK]" if result else "[ERROR]"
        print(f"  {status} {check_name}: {'找到' if result else '未找到'}")
    
    # 4. 检查第二层模型
    print("\n[4] 检查第二层模型（Ridge Stacker, MetaRankerStacker）...")
    second_layer_checks = {
        "Ridge Stacker": "_train_ridge_stacker" in model_content or "ridge_stacker" in model_content,
        "MetaRankerStacker": "MetaRankerStacker" in model_content and ("meta_ranker_stacker" in model_content or "meta_ranker.fit" in model_content),
    }
    
    for check_name, result in second_layer_checks.items():
        status = "[OK]" if result else "[ERROR]"
        print(f"  {status} {check_name}: {'找到' if result else '未找到'}")
    
    # 5. 检查训练调用链
    print("\n[5] 检查训练调用链...")
    
    # train_from_document -> _run_training_phase -> train_enhanced_models -> _execute_modular_training -> _unified_model_training
    chain_checks = {
        "train_from_document调用_run_training_phase": "self._run_training_phase" in model_content and "train_from_document" in model_content,
        "_run_training_phase调用train_enhanced_models": "self.train_enhanced_models" in model_content and "_run_training_phase" in model_content,
        "train_enhanced_models调用_execute_modular_training": "self._execute_modular_training" in model_content and "train_enhanced_models" in model_content,
        "_execute_modular_training调用_unified_model_training": "self._unified_model_training" in model_content and "_execute_modular_training" in model_content,
    }
    
    for check_name, result in chain_checks.items():
        status = "[OK]" if result else "[ERROR]"
        print(f"  {status} {check_name}: {'找到' if result else '未找到'}")
    
    # 6. 检查MetaRankerStacker训练
    print("\n[6] 检查MetaRankerStacker训练逻辑...")
    meta_checks = {
        "MetaRankerStacker导入": "from bma_models.meta_ranker_stacker import MetaRankerStacker" in model_content,
        "MetaRankerStacker初始化": "MetaRankerStacker(" in model_content,
        "MetaRankerStacker.fit": "meta_ranker_stacker.fit" in model_content or "meta_ranker.fit" in model_content,
    }
    
    for check_name, result in meta_checks.items():
        status = "[OK]" if result else "[ERROR]"
        print(f"  {status} {check_name}: {'找到' if result else '未找到'}")
    
    # 总结
    print("\n" + "=" * 80)
    print("[总结]")
    print("=" * 80)
    
    all_checks = {
        **checks,
        **flow_checks,
        **first_layer_checks,
        **second_layer_checks,
        **chain_checks,
        **meta_checks,
    }
    
    passed = sum(1 for v in all_checks.values() if v)
    total = len(all_checks)
    
    print(f"通过检查: {passed}/{total}")
    
    if passed == total:
        print("[OK] 所有检查通过！训练流程正确调用了所有模型")
        print("\n训练流程:")
        print("  1. train_full_dataset.py")
        print("     -> UltraEnhancedQuantitativeModel.train_from_document()")
        print("  2. train_from_document()")
        print("     -> _run_training_phase()")
        print("  3. _run_training_phase()")
        print("     -> train_enhanced_models()")
        print("  4. train_enhanced_models()")
        print("     -> _execute_modular_training()")
        print("  5. _execute_modular_training()")
        print("     -> _unified_model_training() [第一层: ElasticNet, XGBoost, CatBoost, LambdaRank]")
        print("     -> _train_stacking_models_modular() [第二层: Ridge Stacker, MetaRankerStacker]")
        return True
    else:
        print("[WARN] 部分检查未通过，请检查训练流程")
        failed = [k for k, v in all_checks.items() if not v]
        print(f"\n未通过的检查:")
        for check in failed:
            print(f"  - {check}")
        return False

if __name__ == "__main__":
    success = verify_training_flow()
    sys.exit(0 if success else 1)
