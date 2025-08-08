@echo off
chcp 65001 >nul
title Quantitative Trading Software

echo.
echo ========================================
echo    Quantitative Trading Software
echo ========================================
echo.

:: Check Python
echo [1/3] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found
    echo Please install Python 3.8+ and add to PATH
    pause
    exit /b 1
)
echo OK: Python ready

:: Check virtual environment
echo.
echo [2/3] Checking virtual environment...
if not exist "trading_env\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv trading_env
    echo OK: Virtual environment created
) else (
    echo OK: Virtual environment exists
)

:: Activate and launch
echo.
echo [3/3] Starting software...
call trading_env\Scripts\activate.bat

:: Set environment
set PYTHONPATH=%cd%
set PYTHONIOENCODING=utf-8

:: Launch
python quantitative_trading_manager.py

:: Cleanup
call trading_env\Scripts\deactivate.bat >nul 2>&1
pause 