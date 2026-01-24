@echo off
REM 后台运行全量训练脚本
REM 训练完成后自动更新snapshot到latest_snapshot_id.txt
REM Direct Predict会自动使用这个snapshot

echo ================================================================================
echo 全量训练 - 后台运行
echo ================================================================================
echo.
echo 训练数据: D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet
echo Snapshot Tag: FINAL_V2_FULL_DATASET_YYYYMMDD_HHMMSS
echo.
echo 训练完成后，snapshot会自动保存到:
echo   - latest_snapshot_id.txt (Direct Predict默认使用)
echo   - results/full_dataset_training/run_*/snapshot_id.txt
echo.
echo 开始训练...
echo ================================================================================
echo.

start /B python scripts\train_full_dataset.py --train-data "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet" --top-n 50 --log-level INFO > results\full_dataset_training\training_log.txt 2>&1

echo 训练已在后台启动
echo 日志文件: results\full_dataset_training\training_log.txt
echo.
echo 可以使用以下命令查看训练进度:
echo   type results\full_dataset_training\training_log.txt
echo.
echo 训练完成后，检查latest_snapshot_id.txt确认snapshot已更新
echo.
