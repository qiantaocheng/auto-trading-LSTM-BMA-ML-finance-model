#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安装高级LSTM模型所需的依赖包

Dependencies for Advanced LSTM Multi-Day Model:
- optuna: 超参数优化
- scikit-optimize: Bayesian优化
- tensorflow: 深度学习框架
- statsmodels: 统计分析
- factor_analyzer: 因子分析（可选）

Author: AI Assistant
Version: 1.0
"""

import subprocess
import sys
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_package(package_name, pip_name=None):
    """安装Python包"""
    install_name = pip_name or package_name
    
    try:
        logger.info(f"安装 {install_name}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", install_name
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"✅ {install_name} 安装成功")
            return True
        else:
            logger.error(f"❌ {install_name} 安装失败:")
            logger.error(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {install_name} 安装超时")
        return False
    except Exception as e:
        logger.error(f"❌ {install_name} 安装出错: {e}")
        return False

def check_package_installed(package_name):
    """检查包是否已安装"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    logger.info("=" * 60)
    logger.info("高级LSTM模型依赖包安装脚本")
    logger.info("=" * 60)
    
    # 定义需要安装的包
    packages = [
        # 基础科学计算（通常已安装）
        ("numpy", "numpy>=1.21.0"),
        ("pandas", "pandas>=1.3.0"),
        ("scipy", "scipy>=1.7.0"),
        ("matplotlib", "matplotlib>=3.4.0"),
        ("seaborn", "seaborn>=0.11.0"),
        
        # 机器学习核心
        ("sklearn", "scikit-learn>=1.0.0"),
        ("statsmodels", "statsmodels>=0.13.0"),
        
        # 深度学习
        ("tensorflow", "tensorflow>=2.8.0"),
        
        # 超参数优化
        ("optuna", "optuna>=3.0.0"),
        
        # Bayesian优化（可选）
        ("skopt", "scikit-optimize>=0.9.0"),
        
        # 因子分析（可选）
        ("factor_analyzer", "factor_analyzer>=0.4.0"),
        
        # 金融数据
        ("yfinance", "yfinance>=0.2.0"),
        
        # 其他有用的包
        ("tqdm", "tqdm>=4.64.0"),
        ("joblib", "joblib>=1.1.0"),
    ]
    
    # 检查已安装的包
    logger.info("\n检查当前已安装的包...")
    installed_packages = []
    missing_packages = []
    
    for import_name, pip_name in packages:
        if check_package_installed(import_name):
            logger.info(f"✅ {import_name} 已安装")
            installed_packages.append(import_name)
        else:
            logger.info(f"❌ {import_name} 未安装")
            missing_packages.append((import_name, pip_name))
    
    if not missing_packages:
        logger.info("\n🎉 所有依赖包都已安装！")
        return True
    
    # 安装缺失的包
    logger.info(f"\n发现 {len(missing_packages)} 个缺失的包，开始安装...")
    
    successful_installs = 0
    failed_installs = []
    
    for import_name, pip_name in missing_packages:
        if install_package(import_name, pip_name):
            successful_installs += 1
        else:
            failed_installs.append(import_name)
    
    # 报告结果
    logger.info("\n" + "=" * 60)
    logger.info("安装结果汇总")
    logger.info("=" * 60)
    
    logger.info(f"✅ 成功安装: {successful_installs} 个包")
    logger.info(f"❌ 安装失败: {len(failed_installs)} 个包")
    
    if failed_installs:
        logger.info(f"\n失败的包: {', '.join(failed_installs)}")
        logger.info("\n请手动安装失败的包:")
        for pkg in failed_installs:
            logger.info(f"  pip install {pkg}")
    
    # 特殊说明
    logger.info("\n" + "=" * 60)
    logger.info("重要说明")
    logger.info("=" * 60)
    
    logger.info("""
1. TensorFlow安装说明:
   - 如果有GPU，建议安装 tensorflow-gpu
   - 确保CUDA和cuDNN版本兼容
   - CPU版本也可以正常工作，只是速度较慢

2. Optuna安装说明:
   - Optuna需要Python 3.7+
   - 如果安装失败，可以尝试: pip install --upgrade optuna

3. 可选包说明:
   - scikit-optimize: 用于Bayesian优化，可选
   - factor_analyzer: 用于因子分析，可选
   - 即使这些包安装失败，主要功能仍可使用

4. 如果遇到权限问题:
   - 尝试添加 --user 参数: pip install --user package_name
   - 或在虚拟环境中运行

5. 国内用户建议使用镜像源加速:
   - pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name
    """)
    
    return len(failed_installs) == 0

def install_with_mirror():
    """使用国内镜像源安装（中国用户）"""
    logger.info("使用清华镜像源安装依赖包...")
    
    mirror_url = "https://pypi.tuna.tsinghua.edu.cn/simple"
    
    packages_to_install = [
        "tensorflow>=2.8.0",
        "optuna>=3.0.0", 
        "scikit-optimize>=0.9.0",
        "statsmodels>=0.13.0",
        "factor_analyzer>=0.4.0",
        "yfinance>=0.2.0"
    ]
    
    for package in packages_to_install:
        try:
            logger.info(f"安装 {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "-i", mirror_url, package
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✅ {package} 安装成功")
            else:
                logger.error(f"❌ {package} 安装失败: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ {package} 安装出错: {e}")

def check_installation():
    """检查关键包的安装情况"""
    logger.info("\n" + "=" * 60)
    logger.info("验证关键包安装")
    logger.info("=" * 60)
    
    critical_packages = [
        ("tensorflow", "TensorFlow深度学习框架"),
        ("optuna", "超参数优化"),
        ("sklearn", "机器学习算法"),
        ("statsmodels", "统计分析"),
        ("pandas", "数据处理"),
        ("numpy", "数值计算")
    ]
    
    all_installed = True
    
    for package, description in critical_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} ({description}) - 可用")
        except ImportError:
            logger.error(f"❌ {package} ({description}) - 未安装")
            all_installed = False
    
    if all_installed:
        logger.info("\n🎉 所有关键包都已正确安装！")
        logger.info("现在可以运行高级LSTM模型了。")
    else:
        logger.warning("\n⚠️  某些关键包未安装，部分功能可能不可用。")
    
    return all_installed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="安装高级LSTM模型依赖包")
    parser.add_argument("--mirror", action="store_true", 
                       help="使用国内镜像源（推荐中国用户）")
    parser.add_argument("--check-only", action="store_true",
                       help="仅检查安装情况，不安装新包")
    
    args = parser.parse_args()
    
    if args.check_only:
        check_installation()
    elif args.mirror:
        install_with_mirror()
        check_installation()
    else:
        success = main()
        if success:
            check_installation()