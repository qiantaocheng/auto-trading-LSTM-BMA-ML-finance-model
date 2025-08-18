#!/usr/bin/env python3
"""
Test Runner - Run all system tests
"""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run all system tests"""
    project_root = Path(__file__).parent
    test_file = project_root / "tests" / "test_system_integration.py"
    
    print("Running system integration tests...")
    print("=" * 50)
    
    result = subprocess.run([
        sys.executable, str(test_file)
    ], cwd=project_root)
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
        return True
    else:
        print("\n❌ Tests failed!")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
