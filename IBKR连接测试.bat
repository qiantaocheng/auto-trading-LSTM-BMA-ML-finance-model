@echo off
chcp 65001 >nul

echo ===============================================
echo     IBKR 连接测试工具
echo ===============================================

cd /d "%~dp0"

echo [检查] IBKR TWS/Gateway 连接状态...
echo.
echo [提示] 请确保：
echo   1. TWS (Trader Workstation) 或 Gateway 已启动
echo   2. API连接已启用 (配置 -^> API -^> 启用ActiveX和Socket客户端)
echo   3. 端口设置正确 (默认7497 for TWS, 4001 for Gateway)
echo.

echo [测试] 开始连接测试...
trading_env\Scripts\python.exe -c "
import asyncio
from autotrader.launcher import TradingSystemLauncher

async def test_connection():
    launcher = TradingSystemLauncher()
    print('[测试] 启动连接测试模式...')
    result = await launcher.launch_test_mode()
    
    if result:
        print('[成功] IBKR连接测试通过')
        print('[建议] 可以启动完整交易系统')
    else:
        print('[失败] IBKR连接测试失败')
        print('[建议] 请检查TWS/Gateway是否运行且API已启用')
    
    return result

# 运行测试
success = asyncio.run(test_connection())
"

echo.
echo [完成] 连接测试完成
echo [提示] 如果测试通过，可运行 启动自动交易_最终版.bat
echo.
pause