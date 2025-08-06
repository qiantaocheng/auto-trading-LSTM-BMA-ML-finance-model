#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安装量化模型缺失的依赖包
"""

import subprocess
import sys

def install_package(package):
    """安装Python包"""
    try:
        print(f"📦 正在安装 {package}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 成功安装: {package}")
            return True
        else:
            print(f"❌ 安装失败: {package}")
            print(f"错误信息: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 安装异常: {package} - {e}")
        return False

def main():
    """主函数"""
    print("🔧 安装量化模型缺失的依赖包...")
    print("=" * 50)
    
    # 量化模型需要的依赖包
    required_packages = [
        "statsmodels>=0.13.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.6.0",
        "lightgbm>=3.3.0",
        "catboost>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "yfinance>=0.1.87"
    ]
    
    success_count = 0
    total_count = len(required_packages)
    
    for package in required_packages:
        if install_package(package):
            success_count += 1
        print()
    
    print("=" * 50)
    print(f"📊 安装完成: {success_count}/{total_count} 个包安装成功")
    
    if success_count == total_count:
        print("✅ 所有依赖包安装成功！")
    else:
        print("⚠️ 部分依赖包安装失败，请检查网络连接或手动安装")
    
    print("\n🎯 现在可以运行量化模型了！")

if __name__ == "__main__":
    main() 