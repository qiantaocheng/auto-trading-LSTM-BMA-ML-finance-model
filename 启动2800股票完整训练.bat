@echo off
chcp 65001 >nul
setlocal

echo ===============================================
echo     2800股票完整训练启动器
echo     内存优化版 - 预计1-3小时完成
echo ===============================================

cd /d "%~dp0"

:: 检查系统资源
echo [检查] 系统资源状态...
trading_env\Scripts\python.exe -c "
import psutil
memory = psutil.virtual_memory()
if memory.available < 2.5 * 1024**3:
    print('[WARNING] 可用内存不足2.5GB，建议释放内存')
    exit(1)
else:
    print(f'[OK] 内存充足: {memory.available/(1024**3):.1f}GB 可用')
"

if %errorlevel% neq 0 (
    echo [ERROR] 内存不足，请关闭其他程序后重试
    pause
    exit /b 1
)

echo.
echo [警告] 即将开始2800股票完整训练！
echo [预估] 训练时间: 1-3小时
echo [预估] 内存使用: 2-4GB
echo [预估] 磁盘使用: 3GB (缓存+结果)
echo.
echo [提示] 训练过程中请勿关闭此窗口
echo [提示] 可以随时按Ctrl+C安全中断训练
echo.

set /p confirm="确认开始训练？(y/N): "
if /i not "%confirm%"=="y" (
    echo [取消] 训练已取消
    pause
    exit /b 0
)

echo.
echo [启动] 2800股票完整训练开始...
echo [时间] %date% %time%
echo.

:: 启动优化版训练
trading_env\Scripts\python.exe "bma_models\量化模型_bma_ultra_enhanced.py" --tickers-limit 0 --start-date 2022-01-01 --end-date 2024-12-31 --top-n 50

echo.
echo [完成] 训练结束于 %date% %time%
echo [结果] 请查看 result\ 目录中的Excel文件
echo.

pause
endlocal