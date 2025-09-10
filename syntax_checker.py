#!/usr/bin/env python3
"""
Syntax Error Checker for BMA Ultra Enhanced
Find and report all syntax errors systematically
"""

import ast
import sys

def check_syntax_errors(file_path):
    """Check for syntax errors in the file"""
    print("Checking syntax errors in BMA Ultra Enhanced...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse the file
        try:
            ast.parse(content)
            print("SUCCESS: No syntax errors found!")
            return True
        except SyntaxError as e:
            print(f"SYNTAX ERROR found at line {e.lineno}:")
            print(f"  Error: {e.msg}")
            print(f"  Text: {e.text.strip() if e.text else 'N/A'}")
            
            # Show context around the error
            lines = content.split('\n')
            start = max(0, e.lineno - 3)
            end = min(len(lines), e.lineno + 2)
            
            print(f"\nContext around line {e.lineno}:")
            for i in range(start, end):
                marker = " >>> " if i == e.lineno - 1 else "     "
                print(f"{marker}Line {i+1}: {lines[i]}")
            
            return False
            
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

def main():
    file_path = "bma_models/量化模型_bma_ultra_enhanced.py"
    
    while True:
        if check_syntax_errors(file_path):
            print("\nAll syntax errors fixed!")
            break
        else:
            print("\nPress Enter to check again after fixing...")
            input()

if __name__ == "__main__":
    main()