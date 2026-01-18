@echo off
REM GUI Launcher Batch File for IBKR Trading System

cd /d "%~dp0"
python launch_gui.py

if errorlevel 1 (
    echo.
    echo [ERROR] GUI launch failed
    pause
)
