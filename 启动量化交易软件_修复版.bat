@echo off
chcp 65001 > nul
title 量化交易管理软件 - 修复版启动包

echo.
echo ==========================================
echo        量化交易管理软件启动包 v1.1
echo ==========================================
echo.
echo 🚀 正在启动软件...
echo.

cd /d "%~dp0"

REM 激活虚拟环境
echo 📦 激活虚拟环境...
call "trading_env\Scripts\activate.bat"

REM 检查并安装依赖
echo 🔧 检查依赖包...
python -c "import apscheduler" 2>nul
if errorlevel 1 (
    echo 📥 安装 apscheduler...
    pip install apscheduler
)

python -c "import pywin32" 2>nul
if errorlevel 1 (
    echo 📥 安装 pywin32...
    pip install pywin32
)

python -c "import plyer" 2>nul
if errorlevel 1 (
    echo 📥 安装 plyer...
    pip install plyer
)

python -c "import tkcalendar" 2>nul
if errorlevel 1 (
    echo 📥 安装 tkcalendar...
    pip install tkcalendar
)

python -c "import PIL" 2>nul
if errorlevel 1 (
    echo 📥 安装 pillow...
    pip install pillow
)

echo ✅ 依赖检查完成
echo.

REM 启动主程序
echo 🎯 启动量化交易管理软件...
python quantitative_trading_manager.py

echo.
echo 程序已启动，启动窗口将自动关闭
timeout /t 3 /nobreak > nul 