#!/usr/bin/env python3
"""
Fix broken method chains in the BMA file
"""

import re

def fix_method_chains():
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Fixing broken method chains...")
    
    # Fix patterns like .meanaxis = 1 -> .mean(axis=1)
    content = re.sub(r'\.meanaxis\s*=\s*1', '.mean(axis=1)', content)
    content = re.sub(r'\.stdaxis\s*=\s*1', '.std(axis=1)', content)
    content = re.sub(r'\.sumaxis\s*=\s*1', '.sum(axis=1)', content)
    content = re.sub(r'\.maxaxis\s*=\s*1', '.max(axis=1)', content)
    content = re.sub(r'\.minaxis\s*=\s*1', '.min(axis=1)', content)
    
    # Fix broken chaining like .mean(axis=1) / .std(axis=1).fillna(0)
    content = re.sub(r'\.mean\(axis=1\)\s*/\s*([^.]+)\.std\(axis=1\)\.', r'.mean(axis=1) / \1.std(axis=1).', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Method chain fixes applied!")
    
    try:
        compile(content, file_path, 'exec')
        print("SUCCESS: File compiles correctly!")
        return True
    except SyntaxError as e:
        print(f"SYNTAX ERROR at line {e.lineno}: {e.msg}")
        return False

if __name__ == "__main__":
    fix_method_chains()