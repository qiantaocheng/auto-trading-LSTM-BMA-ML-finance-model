@echo off
chcp 65001 >nul
setlocal

echo.
echo ===============================================
echo     IBKR 专业交易系统 - 统一界面版本
echo     策略引擎 ^| 直接交易 ^| 风险管理 ^| 系统监控
echo ===============================================
echo.

cd /d "%~dp0"

:: 检查Python环境
where py >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ 未找到Python环境，请安装Python
    pause
    exit /b 1
)

:: 检查依赖文件
if not exist "autotrader" (
    echo ❌ 未找到autotrader目录
    pause
    exit /b 1
)

if not exist "autotrader\app.py" (
    echo ❌ 未找到主程序文件
    pause
    exit /b 1
)

echo ✅ 环境检查通过
echo.

:: 启动统一GUI界面
echo [启动] 专业交易界面 (包含所有功能)...
echo.
trading_env\Scripts\python.exe autotrader/launcher.py
goto end

:end
echo.
echo 程序已结束
pause

endlocal
exit /b 0