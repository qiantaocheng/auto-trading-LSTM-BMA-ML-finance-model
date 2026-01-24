# 1/5 Ticker子集训练和80/20评估指南

## 📊 子集信息

**原始文件**: `D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet`
- 总ticker数: 3,921
- 总行数: 4,180,394

**子集文件**: `D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet`
- Ticker数: 784 (20%)
- 行数: 827,900 (约20%)
- 文件大小: ~130 MB

**Ticker列表**: `D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers_tickers.txt`

## 🚀 快速开始

### 方法1: 使用自动化脚本（推荐）

```bash
cd d:\trade
python scripts\train_and_eval_subset.py
```

这个脚本会自动：
1. 使用子集进行训练
2. 使用子集进行80/20 OOS评估

### 方法2: 手动执行

#### 步骤1: 训练

```bash
cd d:\trade
python scripts\train_full_dataset.py \
    --train-data "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet" \
    --top-n 50 \
    --log-level INFO
```

#### 步骤2: 80/20 OOS评估

```bash
cd d:\trade
python scripts\time_split_80_20_oos_eval.py \
    --data-file "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet" \
    --horizon-days 10 \
    --split 0.8 \
    --top-n 20 \
    --log-level INFO
```

## 📋 预期输出

### 训练输出

- **输出目录**: `results/full_dataset_training/run_YYYYMMDD_HHMMSS/`
- **Snapshot ID**: 保存在 `snapshot_id.txt`
- **训练日志**: 详细的训练过程日志

### 80/20评估输出

- **输出目录**: `output-dir/run_YYYYMMDD_HHMMSS/`
- **报告文件**: `report_df.csv`
- **Top20时间序列**: `ridge_top20_timeseries.csv`
- **图表**: `top20_vs_qqq.png`, `top20_vs_qqq_cumulative.png`

## 🔍 验证

### 验证子集格式

子集文件应该：
- ✅ MultiIndex格式: `['date', 'ticker']`
- ✅ 日期类型: `datetime64[ns]` (normalized)
- ✅ Ticker类型: `object/string` (UPPERCASE)
- ✅ 包含所有必需的因子列
- ✅ 格式与原始训练文件完全一致

### 验证训练结果

1. 检查 `results/full_dataset_training/run_*/snapshot_id.txt` 是否存在
2. 检查训练日志中是否有 "Training complete" 消息
3. 验证snapshot ID已保存到数据库

### 验证80/20评估结果

1. 检查 `output-dir/run_*/report_df.csv` 是否存在
2. 检查评估指标（Sharpe ratio, 累计收益等）
3. 查看图表文件

## 📊 子集特点

### 优势

- **快速训练**: 数据量减少80%，训练时间大幅缩短
- **快速评估**: 80/20评估速度更快
- **格式一致**: 与原始数据格式完全一致
- **可重复**: 使用固定随机种子(42)，结果可重复

### 注意事项

- **代表性**: 子集是随机选择的，可能不完全代表整个市场
- **性能差异**: 使用子集训练的模型性能可能与全量数据训练的模型有差异
- **仅用于测试**: 建议仅用于快速测试和验证，生产环境应使用全量数据

## 🎯 使用场景

1. **快速原型验证**: 快速验证新的特征或模型架构
2. **参数调优**: 快速测试不同的超参数组合
3. **代码调试**: 快速定位和修复问题
4. **格式验证**: 验证数据格式一致性

## 📝 注意事项

1. **不要覆盖生产snapshot**: 子集训练的snapshot不应用于生产环境
2. **保持格式一致**: 确保子集格式与原始数据完全一致
3. **记录ticker列表**: 保存选择的ticker列表以便后续分析
4. **比较结果**: 将子集结果与全量数据结果进行比较

---

**创建时间**: 2025-01-20

**子集文件**: `D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet`
