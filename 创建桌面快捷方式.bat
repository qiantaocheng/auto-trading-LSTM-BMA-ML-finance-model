@echo off
chcp 65001 >nul

echo ===============================================
echo     创建桌面快捷方式
echo ===============================================

set "DESKTOP=%USERPROFILE%\Desktop"
set "TRADE_DIR=%~dp0"

echo [创建] 桌面快捷方式...

:: 创建主要启动快捷方式
echo [1/4] IBKR自动交易APP...
powershell -Command "
$WshShell = New-Object -comObject WScript.Shell;
$Shortcut = $WshShell.CreateShortcut('%DESKTOP%\IBKR自动交易APP.lnk');
$Shortcut.TargetPath = '%TRADE_DIR%启动自动交易_最终版.bat';
$Shortcut.WorkingDirectory = '%TRADE_DIR%';
$Shortcut.Description = 'IBKR专业交易系统';
$Shortcut.Save()
"

:: 创建快速启动快捷方式
echo [2/4] 快速启动交易...
powershell -Command "
$WshShell = New-Object -comObject WScript.Shell;
$Shortcut = $WshShell.CreateShortcut('%DESKTOP%\快速启动交易.lnk');
$Shortcut.TargetPath = '%TRADE_DIR%快速启动交易APP.bat';
$Shortcut.WorkingDirectory = '%TRADE_DIR%';
$Shortcut.Description = '快速启动IBKR交易界面';
$Shortcut.Save()
"

:: 创建连接测试快捷方式
echo [3/4] IBKR连接测试...
powershell -Command "
$WshShell = New-Object -comObject WScript.Shell;
$Shortcut = $WshShell.CreateShortcut('%DESKTOP%\IBKR连接测试.lnk');
$Shortcut.TargetPath = '%TRADE_DIR%IBKR连接测试.bat';
$Shortcut.WorkingDirectory = '%TRADE_DIR%';
$Shortcut.Description = '测试IBKR连接状态';
$Shortcut.Save()
"

:: 创建量化模型快捷方式
echo [4/4] 量化模型训练...
powershell -Command "
$WshShell = New-Object -comObject WScript.Shell;
$Shortcut = $WshShell.CreateShortcut('%DESKTOP%\量化模型训练.lnk');
$Shortcut.TargetPath = '%TRADE_DIR%启动量化模型_一键运行.bat';
$Shortcut.WorkingDirectory = '%TRADE_DIR%';
$Shortcut.Description = 'BMA量化模型训练系统';
$Shortcut.Save()
"

echo.
echo [完成] 桌面快捷方式创建完成！
echo.
echo [已创建的快捷方式:]
echo   • IBKR自动交易APP.lnk - 主要交易系统
echo   • 快速启动交易.lnk - 快速启动界面  
echo   • IBKR连接测试.lnk - 连接状态测试
echo   • 量化模型训练.lnk - 量化策略训练
echo.
echo [使用建议:]
echo   1. 首次使用先运行"IBKR连接测试"
echo   2. 测试通过后使用"快速启动交易"
echo   3. 需要量化策略时使用"量化模型训练"
echo.
pause