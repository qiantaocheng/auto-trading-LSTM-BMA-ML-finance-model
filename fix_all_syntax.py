#!/usr/bin/env python3
"""
Comprehensive syntax fixer for BMA Ultra Enhanced
Fixes common patterns systematically
"""

import re
import os

def fix_all_syntax_errors():
    """Fix all syntax errors in the file"""
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Fixing syntax errors...")
    
    # Fix 1: Remove broken comment patterns in code
    # Pattern: variable[other_var  # COMMENT'key'] = value
    pattern1 = r"(\w+)\[([^[\]]*?)\s+#[^']*?'([^']+)'\]"
    content = re.sub(pattern1, r"\1['\3']", content)
    
    # Fix 2: Fix invalid assignments like (VAR = value)
    pattern2 = r"\((\w+)\s*=\s*([^)]+)\)"
    content = re.sub(pattern2, r"\1 = \2", content)
    
    # Fix 3: Fix broken matrix operations
    content = content.replace(".pipe(df_optimizer.efficient_fillna)  # OPTIMIZED", ".fillna(0)")
    content = content.replace("# OPTIMIZED @ equal_weights.reindex(expected_returns.index))", "@ equal_weights.reindex(expected_returns.index))")
    
    # Fix 4: Fix Chinese characters in variable assignments
    content = re.sub(r"(\w+)\s*=\s*(\d+)期", r"\1 = \2  # \2期", content)
    
    # Fix 5: Fix broken string patterns
    content = re.sub(r"(\w+)\s+#[^']*?'([^']+)'\]\s*=", r"\1['\2'] =", content)
    
    print("Fixed common syntax patterns")
    
    # Write the fixed content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Syntax fixes applied successfully!")
    
    # Test the syntax
    try:
        compile(content, file_path, 'exec')
        print("SUCCESS: File compiles without syntax errors!")
        return True
    except SyntaxError as e:
        print(f"REMAINING SYNTAX ERROR at line {e.lineno}: {e.msg}")
        return False

if __name__ == "__main__":
    fix_all_syntax_errors()