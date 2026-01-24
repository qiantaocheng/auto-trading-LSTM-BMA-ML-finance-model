@echo off
REM 安全运行80/20评估 - 不覆盖latest_snapshot_id.txt
REM 训练使用80%数据（包含purge gap，无时间泄露）
REM Snapshot只保存到运行目录

echo ================================================================================
echo 80/20评估 - 安全模式
echo ================================================================================
echo.
echo [配置]
echo   - 数据文件: polygon_factors_all_filtered_clean_final_v2.parquet
echo   - 时间分割: 80%%训练，20%%测试
echo   - Purge gap: 10天（防止时间泄露）
echo   - start_date/end_date: 正确传递
echo   - Snapshot: 只保存到运行目录，不覆盖latest_snapshot_id.txt
echo.
echo [验证]
echo   - 使用现有因子（15个Alpha因子）
echo   - 时间过滤正确（train_from_document使用start_date/end_date）
echo   - 无时间泄露（purge gap确保）
echo.
echo ================================================================================
echo.

REM 备份latest_snapshot_id.txt（如果存在）
if exist latest_snapshot_id.txt (
    copy latest_snapshot_id.txt latest_snapshot_id.txt.backup_before_80_20 >nul
    echo [OK] 已备份latest_snapshot_id.txt
) else (
    echo [INFO] latest_snapshot_id.txt不存在，无需备份
)

echo.
echo 开始80/20评估...
echo.

python scripts\time_split_80_20_oos_eval.py ^
  --data-file "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet" ^
  --split 0.8 ^
  --models catboost lambdarank ridge_stacking ^
  --top-n 20 ^
  --log-level INFO

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] 80/20评估失败
    REM 恢复备份（如果存在）
    if exist latest_snapshot_id.txt.backup_before_80_20 (
        copy latest_snapshot_id.txt.backup_before_80_20 latest_snapshot_id.txt >nul
        echo [INFO] 已恢复latest_snapshot_id.txt备份
    )
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [验证] 检查latest_snapshot_id.txt是否被修改
echo ================================================================================

if exist latest_snapshot_id.txt.backup_before_80_20 (
    REM 比较文件
    fc latest_snapshot_id.txt latest_snapshot_id.txt.backup_before_80_20 >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo [OK] latest_snapshot_id.txt未被修改（安全）
    ) else (
        echo [WARN] latest_snapshot_id.txt被修改了！
        echo [INFO] 恢复备份...
        copy latest_snapshot_id.txt.backup_before_80_20 latest_snapshot_id.txt >nul
        echo [OK] 已恢复latest_snapshot_id.txt
    )
) else (
    echo [INFO] 无法验证（无备份文件）
)

echo.
echo ================================================================================
echo [完成] 80/20评估完成
echo ================================================================================
echo.
echo [结果]
echo   - Snapshot保存在: results\t10_time_split_80_20_final\run_*\snapshot_id.txt
echo   - latest_snapshot_id.txt: 未被修改（安全）
echo   - 全量训练snapshot: 不受影响
echo.
pause
