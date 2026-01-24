#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查当前80/20 split脚本的默认配置
"""

import sys
from pathlib import Path
import inspect

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def check_current_split_config():
    """检查当前split配置"""
    print("=" * 80)
    print("当前80/20 Split配置分析")
    print("=" * 80)
    
    script_path = project_root / "scripts" / "time_split_80_20_oos_eval.py"
    
    if not script_path.exists():
        print(f"❌ 脚本不存在: {script_path}")
        return
    
    # 读取脚本文件
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找关键配置
    print("\n[1] 关键配置参数:")
    
    # 查找 --split 默认值
    import re
    split_match = re.search(r'--split.*?default=([0-9.]+)', content)
    if split_match:
        split_default = float(split_match.group(1))
        print(f"\n  --split 默认值: {split_default}")
        if split_default == 0.8:
            print(f"    [OK] 已设置为80/20 (0.8)")
        elif split_default == 0.9:
            print(f"    [WARN] 当前是90/10 (0.9)，不是80/20")
        else:
            print(f"    [WARN] 当前是 {split_default*100:.0f}/{100-split_default*100:.0f}")
    else:
        print(f"    [ERROR] 未找到 --split 默认值")
    
    # 查找 --output-dir 默认值
    output_match = re.search(r'--output-dir.*?default=["\']([^"\']+)["\']', content)
    if output_match:
        output_default = output_match.group(1)
        print(f"\n  --output-dir 默认值: {output_default}")
        if "80_20" in output_default or "80-20" in output_default:
            print(f"    [OK] 输出目录包含80_20")
        elif "90_10" in output_default or "90-10" in output_default:
            print(f"    [WARN] 输出目录包含90_10，不是80_20")
        else:
            print(f"    [WARN] 输出目录: {output_default}")
    else:
        print(f"    [ERROR] 未找到 --output-dir 默认值")
    
    # 检查脚本名称
    print(f"\n  脚本名称: {script_path.name}")
    if "80_20" in script_path.name or "80-20" in script_path.name:
        print(f"    [OK] 脚本名称包含80_20")
    else:
        print(f"    [WARN] 脚本名称不包含80_20")
    
    # 总结
    print("\n" + "=" * 80)
    print("[总结]")
    print("=" * 80)
    
    if split_match and output_match:
        split_val = float(split_match.group(1))
        output_val = output_match.group(1)
        
        is_80_20 = (split_val == 0.8) and ("80_20" in output_val or "80-20" in output_val)
        
        if is_80_20:
            print("[OK] 当前配置是80/20")
        else:
            print("[WARN] 当前配置不是80/20:")
            print(f"   - split默认值: {split_val} ({'90/10' if split_val == 0.9 else f'{int(split_val*100)}/{int((1-split_val)*100)}'})")
            print(f"   - output-dir: {output_val}")
            print("\n建议修改:")
            print("   1. 将 --split 默认值改为 0.8")
            print("   2. 将 --output-dir 默认值改为 'results/t10_time_split_80_20_final'")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    check_current_split_config()
