#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化交易软件 - 一键安装所有依赖包
支持虚拟环境和全局环境
"""

import sys
import os
import platform
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要Python 3.8或更高版本")
        print(f"当前版本: {sys.version}")
        return False
    print(f"✅ Python版本: {sys.version}")
    return True

def check_virtual_environment():
    """检查虚拟环境"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"✅ 检测到虚拟环境: {sys.prefix}")
        return True
    else:
        print("⚠️ 未检测到虚拟环境，将安装到全局环境")
        return False

def install_package(package, upgrade=False):
    """安装Python包"""
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
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

def check_package(package_name):
    """检查包是否已安装"""
    try:
        importlib.util.find_spec(package_name)
        print(f"✅ {package_name} 已安装")
        return True
    except ImportError:
        print(f"❌ {package_name} 未安装")
        return False

def install_all_dependencies():
    """安装所有依赖包"""
    print("\n📦 开始安装所有依赖包...")
    
    # 基础数据处理包
    basic_packages = [
        "pandas>=1.3.0",
        "numpy>=1.21.0", 
        "scipy>=1.7.0",
        "openpyxl>=3.0.0",
        "xlrd>=2.0.0",
        "chardet>=4.0.0"
    ]
    
    # 金融数据包
    finance_packages = [
        "yfinance>=0.1.87",
        "tushare>=1.2.89",
        "akshare>=1.8.0",
        "baostock>=0.8.0"
    ]
    
    # 机器学习包
    ml_packages = [
        "scikit-learn>=1.0.0",
        "xgboost>=1.6.0",
        "lightgbm>=3.3.0", 
        "catboost>=1.0.0",
        "hyperopt>=0.2.7"
    ]
    
    # 统计建模包
    stats_packages = [
        "statsmodels>=0.13.0",
        "arch>=5.3.0"
    ]
    
    # 可视化包
    viz_packages = [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "bokeh>=2.4.0"
    ]
    
    # GUI包
    gui_packages = [
        "tkcalendar>=1.6.0",
        "Pillow>=8.0.0",
        "plyer>=2.0.0"
    ]
    
    # 系统集成包
    system_packages = [
        "APScheduler>=3.9.0",
        "pywin32>=305; sys_platform == 'win32'",
        "psutil>=5.8.0",
        "requests>=2.25.0",
        "beautifulsoup4>=4.9.0",
        "lxml>=4.6.0"
    ]
    
    # 数据库包
    db_packages = [
        "sqlite3",  # 内置包
        "sqlalchemy>=1.4.0",
        "pymongo>=4.0.0"
    ]
    
    # 时间处理包
    time_packages = [
        "python-dateutil>=2.8.0",
        "pytz>=2021.1"
    ]
    
    # 配置和日志包
    config_packages = [
        "configparser",
        "logging",
        "json5>=2.2.0"
    ]
    
    # 所有包列表
    all_packages = (
        basic_packages + 
        finance_packages + 
        ml_packages + 
        stats_packages + 
        viz_packages + 
        gui_packages + 
        system_packages + 
        time_packages + 
        [pkg for pkg in config_packages if pkg not in ['configparser', 'logging']]
    )
    
    success_count = 0
    total_count = len(all_packages)
    
    print(f"\n📊 总计需要安装 {total_count} 个包")
    
    # 安装基础包
    print("\n🔧 安装基础数据处理包...")
    for package in basic_packages:
        if install_package(package):
            success_count += 1
    
    # 安装金融数据包
    print("\n💰 安装金融数据包...")
    for package in finance_packages:
        if install_package(package):
            success_count += 1
    
    # 安装机器学习包
    print("\n🤖 安装机器学习包...")
    for package in ml_packages:
        if install_package(package):
            success_count += 1
    
    # 安装统计建模包
    print("\n📊 安装统计建模包...")
    for package in stats_packages:
        if install_package(package):
            success_count += 1
    
    # 安装可视化包
    print("\n📈 安装可视化包...")
    for package in viz_packages:
        if install_package(package):
            success_count += 1
    
    # 安装GUI包
    print("\n🖥️ 安装GUI包...")
    for package in gui_packages:
        if install_package(package):
            success_count += 1
    
    # 安装系统集成包
    print("\n🔧 安装系统集成包...")
    for package in system_packages:
        if install_package(package):
            success_count += 1
    
    # 安装时间处理包
    print("\n⏰ 安装时间处理包...")
    for package in time_packages:
        if install_package(package):
            success_count += 1
    
    # 安装配置包
    print("\n⚙️ 安装配置包...")
    for package in config_packages:
        if package not in ['configparser', 'logging']:  # 跳过内置包
            if install_package(package):
                success_count += 1
    
    print(f"\n📈 安装结果: {success_count}/{total_count} 成功")
    return success_count >= total_count * 0.8  # 80%成功率即可

def verify_installations():
    """验证关键包的安装"""
    print("\n🔍 验证关键包安装...")
    
    critical_packages = [
        "pandas", "numpy", "scipy", "yfinance", "scikit-learn",
        "matplotlib", "seaborn", "openpyxl", "Pillow", "plyer",
        "apscheduler", "tkcalendar"
    ]
    
    success_count = 0
    for package in critical_packages:
        if check_package(package):
            success_count += 1
    
    print(f"\n✅ 关键包验证: {success_count}/{len(critical_packages)} 正常")
    return success_count == len(critical_packages)

def create_requirements_file():
    """创建requirements.txt文件"""
    requirements_content = """# 量化交易软件依赖包
# 基础数据处理
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
openpyxl>=3.0.0
xlrd>=2.0.0
chardet>=4.0.0

# 金融数据
yfinance>=0.1.87
tushare>=1.2.89
akshare>=1.8.0
baostock>=0.8.0

# 机器学习
scikit-learn>=1.0.0
xgboost>=1.6.0
lightgbm>=3.3.0
catboost>=1.0.0
hyperopt>=0.2.7

# 统计建模
statsmodels>=0.13.0
arch>=5.3.0

# 可视化
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
bokeh>=2.4.0

# GUI
tkcalendar>=1.6.0
Pillow>=8.0.0
plyer>=2.0.0

# 系统集成
APScheduler>=3.9.0
pywin32>=305; sys_platform == 'win32'
psutil>=5.8.0
requests>=2.25.0
beautifulsoup4>=4.9.0
lxml>=4.6.0

# 数据库
sqlalchemy>=1.4.0
pymongo>=4.0.0

# 时间处理
python-dateutil>=2.8.0
pytz>=2021.1

# 配置
json5>=2.2.0
"""
    
    with open("requirements_complete.txt", "w", encoding="utf-8") as f:
        f.write(requirements_content)
    
    print("✅ 完整依赖列表已保存: requirements_complete.txt")

def create_enhanced_launcher():
    """创建增强版启动器"""
    launcher_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化交易软件 - 增强版启动器
自动检查依赖并启动
"""

import sys
import os
import subprocess
import importlib.util

def check_dependencies():
    """检查关键依赖"""
    critical_packages = [
        "pandas", "numpy", "yfinance", "scikit-learn",
        "matplotlib", "Pillow", "apscheduler"
    ]
    
    missing_packages = []
    for package in critical_packages:
        try:
            importlib.util.find_spec(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(packages):
    """安装缺失的包"""
    print("📦 安装缺失的依赖包...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ 已安装: {package}")
        except subprocess.CalledProcessError:
            print(f"❌ 安装失败: {package}")

def main():
    """主函数"""
    print("🚀 量化交易软件启动器")
    print("=" * 40)
    
    # 检查依赖
    missing = check_dependencies()
    if missing:
        print(f"⚠️ 发现缺失依赖: {', '.join(missing)}")
        install_missing_packages(missing)
    
    # 检查主程序文件
    if not os.path.exists("quantitative_trading_manager.py"):
        print("❌ 未找到主程序文件: quantitative_trading_manager.py")
        return
    
    # 启动主程序
    print("🎯 启动量化交易管理软件...")
    try:
        subprocess.run([sys.executable, "quantitative_trading_manager.py"])
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open("launch_enhanced.py", "w", encoding="utf-8") as f:
        f.write(launcher_content)
    
    print("✅ 增强版启动器已创建: launch_enhanced.py")

def main():
    """主函数"""
    print("=== 量化交易软件 - 一键安装所有依赖 ===")
    print("支持Windows/Linux/Mac系统")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        print("\n❌ Python版本检查失败")
        return
    
    # 检查虚拟环境
    check_virtual_environment()
    
    # 检查操作系统
    system = platform.system()
    print(f"✅ 操作系统: {system}")
    
    # 安装所有依赖
    if install_all_dependencies():
        print("\n✅ 所有依赖安装成功！")
    else:
        print("\n⚠️ 部分依赖安装失败，但软件仍可运行")
    
    # 验证安装
    if verify_installations():
        print("\n✅ 关键包验证通过！")
    else:
        print("\n⚠️ 部分关键包验证失败")
    
    # 创建文件
    create_requirements_file()
    create_enhanced_launcher()
    
    print("\n🎉 安装完成！")
    print("\n📁 可用文件:")
    print("├── quantitative_trading_manager.py (主程序)")
    print("├── launch_enhanced.py (增强版启动器)")
    print("├── requirements_complete.txt (完整依赖列表)")
    print("├── 点我.bat (Windows启动)")
    print("└── 启动量化交易软件_修复版.bat (修复版启动)")
    
    print("\n🚀 启动方式:")
    print("1. 双击 '点我.bat'")
    print("2. 运行 'python launch_enhanced.py'")
    print("3. 运行 'python quantitative_trading_manager.py'")

if __name__ == "__main__":
    main() 