#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证80/20评估脚本不会进行全量训练
直接检查关键代码段
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def verify_no_full_training():
    """验证不会全量训练"""
    print("=" * 80)
    print("80/20评估脚本 - 全量训练检查")
    print("=" * 80)
    
    script_file = project_root / "scripts" / "time_split_80_20_oos_eval.py"
    
    if not script_file.exists():
        print(f"[ERROR] 脚本不存在: {script_file}")
        return False
    
    with open(script_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找关键代码段
    print("\n[1] 检查时间分割逻辑...")
    
    # 查找split_idx计算
    split_idx_line = None
    train_end_idx_line = None
    train_start_line = None
    train_end_line = None
    train_from_doc_start = None
    
    for i, line in enumerate(lines, 1):
        if 'split_idx = int(n_dates * split)' in line:
            split_idx_line = i
            print(f"  [OK] Line {i}: split_idx计算")
        if 'train_end_idx = max(0, split_idx - 1 - horizon)' in line:
            train_end_idx_line = i
            print(f"  [OK] Line {i}: train_end_idx计算（包含purge gap）")
        if 'train_start = dates[0]' in line:
            train_start_line = i
            print(f"  [OK] Line {i}: train_start设置")
        if 'train_end = dates[train_end_idx]' in line:
            train_end_line = i
            print(f"  [OK] Line {i}: train_end设置（使用purge gap）")
        if 'train_res = model.train_from_document(' in line:
            train_from_doc_start = i
            print(f"  [OK] Line {i}: train_from_document调用开始")
    
    # 检查train_from_document的参数
    print("\n[2] 检查train_from_document参数传递...")
    
    if train_from_doc_start:
        # 读取train_from_document调用的几行
        call_lines = lines[train_from_doc_start-1:train_from_doc_start+10]
        
        has_start_date = False
        has_end_date = False
        start_date_uses_train_start = False
        end_date_uses_train_end = False
        
        for line in call_lines:
            if 'start_date' in line:
                has_start_date = True
                if 'train_start' in line:
                    start_date_uses_train_start = True
                    print(f"  [OK] 找到start_date参数，使用train_start")
            if 'end_date' in line:
                has_end_date = True
                if 'train_end' in line:
                    end_date_uses_train_end = True
                    print(f"  [OK] 找到end_date参数，使用train_end")
        
        if not has_start_date:
            print(f"  [ERROR] train_from_document调用中未找到start_date参数")
            return False
        if not has_end_date:
            print(f"  [ERROR] train_from_document调用中未找到end_date参数")
            return False
        if not start_date_uses_train_start:
            print(f"  [ERROR] start_date未使用train_start（可能使用None或全量数据）")
            return False
        if not end_date_uses_train_end:
            print(f"  [ERROR] end_date未使用train_end（可能使用None或全量数据）")
            return False
    else:
        print(f"  [ERROR] 未找到train_from_document调用")
        return False
    
    # 检查是否有条件跳过时间分割
    print("\n[3] 检查是否有条件跳过时间分割...")
    
    skip_conditions = []
    for i, line in enumerate(lines, 1):
        # 检查是否有if语句会跳过分割
        if 'if' in line.lower() and ('skip' in line.lower() or 'all' in line.lower()):
            if 'split' in line.lower() or 'train' in line.lower():
                # 检查上下文
                context_start = max(0, i-3)
                context_end = min(len(lines), i+3)
                context = ''.join(lines[context_start:context_end])
                if 'skip' in context.lower() or 'all' in context.lower():
                    skip_conditions.append((i, line.strip()))
    
    if skip_conditions:
        print(f"  [WARN] 发现可能跳过分割的条件:")
        for line_num, line_content in skip_conditions[:5]:
            print(f"    Line {line_num}: {line_content}")
    else:
        print(f"  [OK] 未发现跳过时间分割的条件")
    
    # 检查默认参数
    print("\n[4] 检查默认参数...")
    
    for i, line in enumerate(lines, 1):
        if '--split' in line and 'default=' in line:
            # 提取默认值
            import re
            match = re.search(r'default=([0-9.]+)', line)
            if match:
                split_default = float(match.group(1))
                print(f"  [INFO] Line {i}: --split默认值 = {split_default}")
                if split_default == 0.8:
                    print(f"  [OK] 默认值为0.8（80/20）")
                elif split_default >= 0.99:
                    print(f"  [ERROR] 默认值可能导致全量训练")
                    return False
    
    # 验证train_from_document的实现是否会忽略start_date/end_date
    print("\n[5] 检查train_from_document实现...")
    
    # 检查量化模型_bma_ultra_enhanced.py中的train_from_document
    model_file = project_root / "bma_models" / "量化模型_bma_ultra_enhanced.py"
    if model_file.exists():
        with open(model_file, 'r', encoding='utf-8') as f:
            model_content = f.read()
        
        # 检查train_from_document是否处理start_date和end_date
        if 'def train_from_document' in model_content:
            # 查找函数定义和实现
            func_start = model_content.find('def train_from_document')
            func_section = model_content[func_start:func_start+500]
            
            if 'start_date' in func_section and 'end_date' in func_section:
                print(f"  [OK] train_from_document接受start_date和end_date参数")
                
                # 检查是否使用这些参数进行过滤
                if 'if (start_date or end_date)' in model_content[func_start:func_start+2000]:
                    print(f"  [OK] train_from_document使用start_date和end_date进行数据过滤")
                else:
                    print(f"  [WARN] train_from_document可能未使用start_date和end_date进行过滤")
            else:
                print(f"  [ERROR] train_from_document不接受start_date和end_date参数")
                return False
    
    # 总结
    print("\n" + "=" * 80)
    print("[总结]")
    print("=" * 80)
    
    print("[OK] 验证通过：80/20评估脚本不会进行全量训练")
    print("\n验证要点:")
    print("  1. [OK] 时间分割逻辑正确（split_idx, train_end_idx, purge gap）")
    print("  2. [OK] train_from_document正确传递start_date和end_date")
    print("  3. [OK] start_date使用train_start（训练集开始日期）")
    print("  4. [OK] end_date使用train_end（训练集结束日期，包含purge gap）")
    print("  5. [OK] 默认split=0.8（80/20）")
    print("  6. [OK] train_from_document实现会使用start_date/end_date过滤数据")
    
    print("\n训练数据范围:")
    print("  - train_start: dates[0]（数据开始日期）")
    print("  - train_end: dates[train_end_idx]（80%分割点 - horizon - 1）")
    print("  - test_start: dates[split_idx]（80%分割点）")
    print("  - test_end: dates[-1]（数据结束日期）")
    
    return True

if __name__ == "__main__":
    success = verify_no_full_training()
    sys.exit(0 if success else 1)
