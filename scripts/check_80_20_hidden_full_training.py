#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查80/20评估脚本是否有隐藏的全量训练"地雷"
检查：
1. train_from_document是否传递了start_date和end_date
2. 是否有条件会跳过时间分割
3. 是否有默认值导致全量训练
4. 是否有逻辑错误导致使用全部数据
"""

import sys
from pathlib import Path
import re

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def check_hidden_full_training():
    """检查隐藏的全量训练问题"""
    print("=" * 80)
    print("80/20评估脚本隐藏全量训练检查")
    print("=" * 80)
    
    script_file = project_root / "scripts" / "time_split_80_20_oos_eval.py"
    
    if not script_file.exists():
        print(f"[ERROR] 脚本不存在: {script_file}")
        return False
    
    with open(script_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    warnings = []
    
    # 1. 检查train_from_document调用
    print("\n[1] 检查train_from_document调用...")
    
    # 查找train_from_document调用（支持多行）
    train_call_pattern = r'train_from_document\s*\([^)]*(?:\([^)]*\)[^)]*)*\)'
    train_calls = re.findall(train_call_pattern, content, re.MULTILINE | re.DOTALL)
    
    # 如果没找到，尝试更宽松的匹配
    if not train_calls:
        # 查找train_from_document所在的行范围
        train_from_doc_pos = content.find('train_from_document')
        if train_from_doc_pos >= 0:
            # 提取从train_from_document到下一个函数调用或代码块结束的内容
            section_start = train_from_doc_pos
            section_end = min(
                content.find('\n    ', section_start + 100),
                content.find('\n        ', section_start + 100),
                section_start + 500
            )
            if section_end > section_start:
                call_section = content[section_start:section_end]
                train_calls = [call_section]
    
    if train_calls:
        for i, call in enumerate(train_calls, 1):
            print(f"\n  调用 {i}:")
            # 显示关键部分
            call_preview = call.replace('\n', ' ')[:300]
            print(f"    {call_preview}...")
            
            # 检查是否包含start_date和end_date
            has_start_date = 'start_date' in call
            has_end_date = 'end_date' in call
            
            if has_start_date and has_end_date:
                print(f"    [OK] 包含start_date和end_date参数")
                
                # 检查参数值
                start_date_match = re.search(r'start_date\s*=\s*([^,\n)]+)', call)
                end_date_match = re.search(r'end_date\s*=\s*([^,\n)]+)', call)
                
                if start_date_match:
                    start_val = start_date_match.group(1).strip()
                    print(f"    [INFO] start_date = {start_val}")
                    if 'None' in start_val or 'null' in start_val.lower():
                        warnings.append(f"train_from_document调用 {i}: start_date可能为None")
                    elif 'train_start' in start_val:
                        print(f"    [OK] start_date使用train_start（正确）")
                
                if end_date_match:
                    end_val = end_date_match.group(1).strip()
                    print(f"    [INFO] end_date = {end_val}")
                    if 'None' in end_val or 'null' in end_val.lower():
                        warnings.append(f"train_from_document调用 {i}: end_date可能为None")
                    elif 'train_end' in end_val:
                        print(f"    [OK] end_date使用train_end（正确）")
            else:
                if not has_start_date:
                    issues.append(f"train_from_document调用 {i}: 缺少start_date参数（可能导致全量训练）")
                    print(f"    [ERROR] 缺少start_date参数")
                if not has_end_date:
                    issues.append(f"train_from_document调用 {i}: 缺少end_date参数（可能导致全量训练）")
                    print(f"    [ERROR] 缺少end_date参数")
    else:
        print(f"  [WARN] 未找到train_from_document调用")
    
    # 2. 检查时间分割逻辑
    print("\n[2] 检查时间分割逻辑...")
    
    # 检查split_idx计算
    if 'split_idx = int(n_dates * split)' in content:
        print(f"  [OK] 找到split_idx计算")
    else:
        issues.append("未找到split_idx计算逻辑")
        print(f"  [ERROR] 未找到split_idx计算逻辑")
    
    # 检查train_end_idx计算（purge gap）
    if 'train_end_idx = max(0, split_idx - 1 - horizon)' in content:
        print(f"  [OK] 找到train_end_idx计算（包含purge gap）")
    else:
        warnings.append("未找到train_end_idx计算（可能缺少purge gap）")
        print(f"  [WARN] 未找到train_end_idx计算")
    
    # 检查train_start和train_end设置
    if 'train_start = dates[0]' in content:
        print(f"  [OK] train_start设置为dates[0]")
    else:
        warnings.append("train_start可能未正确设置")
        print(f"  [WARN] train_start可能未正确设置")
    
    if 'train_end = dates[train_end_idx]' in content:
        print(f"  [OK] train_end设置为dates[train_end_idx]（使用purge gap后的索引）")
    else:
        issues.append("train_end可能未正确使用purge gap")
        print(f"  [ERROR] train_end可能未正确使用purge gap")
    
    # 3. 检查是否有条件跳过时间分割
    print("\n[3] 检查是否有条件跳过时间分割...")
    
    # 检查是否有if语句跳过分割
    skip_patterns = [
        r'if\s+.*split.*skip',
        r'if\s+.*split.*None',
        r'if\s+.*split.*0',
        r'if\s+.*train.*all',
        r'if\s+.*full.*data',
    ]
    
    for pattern in skip_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            warnings.append(f"发现可能跳过分割的条件: {pattern}")
            print(f"  [WARN] 发现可能跳过分割的条件: {pattern}")
            for match in matches[:3]:
                print(f"    - {match}")
    
    # 4. 检查默认参数
    print("\n[4] 检查默认参数...")
    
    # 检查split默认值
    split_default_match = re.search(r'--split.*?default=([0-9.]+)', content)
    if split_default_match:
        split_default = float(split_default_match.group(1))
        print(f"  [INFO] --split默认值: {split_default}")
        if split_default == 1.0 or split_default >= 0.99:
            issues.append(f"--split默认值为{split_default}，可能导致全量训练")
            print(f"  [ERROR] --split默认值可能导致全量训练")
        elif split_default == 0.8:
            print(f"  [OK] --split默认值为0.8（80/20）")
        else:
            print(f"  [INFO] --split默认值为{split_default}")
    
    # 5. 检查train_from_document的参数传递
    print("\n[5] 检查train_from_document参数传递...")
    
    # 查找train_from_document调用附近的代码
    train_from_doc_section = content[content.find('train_from_document'):content.find('train_from_document') + 2000]
    
    # 检查是否使用了train_start和train_end
    if 'train_start' in train_from_doc_section and 'train_end' in train_from_doc_section:
        print(f"  [OK] train_from_document调用附近使用了train_start和train_end")
        
        # 检查是否作为参数传递
        if 'start_date=str(train_start' in train_from_doc_section:
            print(f"  [OK] start_date参数使用train_start")
        else:
            issues.append("start_date参数可能未使用train_start")
            print(f"  [ERROR] start_date参数可能未使用train_start")
        
        if 'end_date=str(train_end' in train_from_doc_section:
            print(f"  [OK] end_date参数使用train_end")
        else:
            issues.append("end_date参数可能未使用train_end")
            print(f"  [ERROR] end_date参数可能未使用train_end")
    else:
        issues.append("train_from_document调用附近未找到train_start和train_end")
        print(f"  [ERROR] train_from_document调用附近未找到train_start和train_end")
    
    # 6. 检查数据过滤逻辑
    print("\n[6] 检查数据过滤逻辑...")
    
    # 检查是否有数据过滤
    filter_patterns = [
        r'df\s*=\s*df\.loc\[.*train',
        r'df\s*=\s*df\[.*train',
        r'train_data\s*=\s*df\[.*train',
    ]
    
    found_filter = False
    for pattern in filter_patterns:
        if re.search(pattern, content):
            found_filter = True
            print(f"  [OK] 找到数据过滤逻辑: {pattern}")
            break
    
    if not found_filter:
        warnings.append("未找到明确的数据过滤逻辑（可能通过train_from_document的start_date/end_date过滤）")
        print(f"  [INFO] 未找到明确的数据过滤逻辑（可能通过train_from_document的start_date/end_date过滤）")
    
    # 总结
    print("\n" + "=" * 80)
    print("[总结]")
    print("=" * 80)
    
    if issues:
        print(f"[ERROR] 发现 {len(issues)} 个严重问题（可能导致全量训练）:")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print(f"\n[WARN] 发现 {len(warnings)} 个警告:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues and not warnings:
        print("[OK] 未发现隐藏的全量训练问题")
        print("\n验证要点:")
        print("  1. train_from_document正确传递了start_date和end_date")
        print("  2. 时间分割逻辑正确（包含purge gap）")
        print("  3. 没有条件跳过时间分割")
        print("  4. 默认参数正确（split=0.8）")
        return True
    else:
        if issues:
            print(f"\n[ERROR] 发现严重问题，可能导致全量训练而非80/20分割！")
            return False
        else:
            print(f"\n[WARN] 发现警告，但可能不影响80/20分割")
            return True

if __name__ == "__main__":
    success = check_hidden_full_training()
    sys.exit(0 if success else 1)
