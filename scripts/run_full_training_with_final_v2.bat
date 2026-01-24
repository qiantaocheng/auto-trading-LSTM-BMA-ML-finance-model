@echo off
REM 全量训练脚本 - 使用final_v2数据文件
REM 生成显眼的snapshot: FINAL_V2_FULL_DATASET_YYYYMMDD_HHMMSS

echo ================================================================================
echo 全量训练 - 使用final_v2数据文件
echo ================================================================================
echo.

python scripts\train_full_dataset.py ^
  --train-data "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet" ^
  --top-n 50 ^
  --log-level INFO

echo.
echo ================================================================================
echo 训练完成！
echo Snapshot ID已保存到: latest_snapshot_id.txt
echo Snapshot Tag格式: FINAL_V2_FULL_DATASET_YYYYMMDD_HHMMSS
echo Direct Predict将自动使用这个snapshot
echo ================================================================================
pause
