@echo off
chcp 65001 >nul

echo ===============================================
echo     IBKR 自动交易 APP - 快速启动
echo ===============================================

cd /d "%~dp0"

:: 直接启动GUI界面
echo [快速启动] IBKR专业交易界面...
echo [提示] 启动后可在界面中选择所有功能
echo.

trading_env\Scripts\python.exe autotrader\launcher.py

echo.
echo [结束] 交易APP已关闭
pause