# 子集训练和评估重启状态

## ✅ 操作完成

**操作时间**: 2026-01-22 21:36

### 1. 终止旧进程

- ✅ 已终止所有旧的Python进程
- ✅ 确认没有遗留进程

### 2. 重新启动

- ✅ 已重新启动子集训练和评估脚本
- ✅ 脚本: `scripts/train_and_eval_subset.py`
- ✅ 运行模式: 后台运行

### 3. 新进程状态

**Python进程**:
- **进程1**: ID 22348, 启动时间: 2026-01-22 21:36:18
- **进程2**: ID 27020, 启动时间: 2026-01-22 21:36:18

## 📋 执行流程

脚本将按顺序执行：

### 步骤1: 训练（当前进行中）

```bash
python scripts/train_full_dataset.py \
    --train-data "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet" \
    --top-n 50 \
    --log-level INFO
```

**预计时间**: 1.5-2小时

### 步骤2: 80/20评估（训练完成后自动开始）

```bash
python scripts/time_split_80_20_oos_eval.py \
    --data-file "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet" \
    --horizon-days 10 \
    --split 0.8 \
    --top-n 20 \
    --log-level INFO
```

**预计时间**: 30-60分钟

## 🔍 如何监控

### 检查训练状态

```bash
# 检查最新的训练运行
ls results/full_dataset_training/run_*/

# 检查训练是否完成
cat results/full_dataset_training/run_*/snapshot_id.txt
```

### 检查评估状态

```bash
# 检查最新的评估运行
ls results/t10_time_split_80_20_final/run_*/

# 检查评估是否完成
cat results/t10_time_split_80_20_final/run_*/report_df.csv
```

### 检查进程状态

```bash
# Windows PowerShell
Get-Process python | Select-Object Id, ProcessName, StartTime, @{Name="Runtime";Expression={(Get-Date) - $_.StartTime}}
```

### 使用状态检查脚本

```bash
cd d:\trade
python scripts\check_subset_training_status.py
```

## 📊 预期输出

### 训练完成后

- **输出目录**: `results/full_dataset_training/run_YYYYMMDD_HHMMSS/`
- **Snapshot ID**: `snapshot_id.txt`
- **训练日志**: 详细的训练过程

### 80/20评估完成后

- **输出目录**: `results/t10_time_split_80_20_final/run_YYYYMMDD_HHMMSS/`
- **报告文件**: `report_df.csv`
- **Top20时间序列**: `*_top20_timeseries.csv`
- **图表**: `*_top20_vs_qqq.png`, `*_top20_vs_qqq_cumulative.png`

## ⏱️ 预计完成时间

- **训练**: 1.5-2小时（从21:36开始）
- **评估**: 30-60分钟（训练完成后）
- **总计**: 约2-3小时

**预计完成时间**: 约23:36-00:36

## 🎯 下一步

1. **等待训练完成**: 预计1.5-2小时
2. **自动开始评估**: 训练完成后自动开始
3. **查看结果**: 完成后查看报告文件

---

**状态**: ✅ **已重新启动，训练进行中**

**启动时间**: 2026-01-22 21:36:18
