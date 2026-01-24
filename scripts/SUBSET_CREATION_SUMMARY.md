# 1/5 Ticker子集创建和训练总结

## ✅ 已完成

### 1. 子集创建

**原始文件**: `D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet`
- 总ticker数: **3,921**
- 总行数: **4,180,394**
- 文件大小: ~650 MB

**子集文件**: `D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet`
- Ticker数: **784** (20%)
- 行数: **827,900** (约20%)
- 文件大小: **~130 MB**
- 数据减少: **80.2%**

**Ticker列表**: `D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers_tickers.txt`

### 2. 格式验证

✅ **MultiIndex格式**: `['date', 'ticker']`
✅ **日期类型**: `datetime64[ns]` (normalized)
✅ **Ticker类型**: `object/string`
✅ **唯一日期数**: 1,244
✅ **唯一ticker数**: 784
✅ **列数**: 28 (与原始文件一致)

### 3. 训练和评估

**状态**: 🟢 **已启动** (后台运行)

**脚本**: `scripts/train_and_eval_subset.py`

**执行步骤**:
1. ✅ 使用子集进行训练 (`train_full_dataset.py`)
2. 🔄 使用子集进行80/20 OOS评估 (`time_split_80_20_oos_eval.py`)

## 📊 子集特点

### 随机选择

- **方法**: 随机选择20%的ticker
- **随机种子**: 42 (确保可重复)
- **选择数量**: 784个ticker

### 数据分布

- **日期覆盖**: 完整日期范围 (1,244个交易日)
- **Ticker分布**: 随机分布，覆盖不同行业和市值
- **数据完整性**: 每个ticker的完整历史数据

## 🎯 使用场景

1. **快速原型验证**: 快速验证新特征或模型架构
2. **参数调优**: 快速测试不同超参数
3. **代码调试**: 快速定位和修复问题
4. **格式验证**: 验证数据格式一致性

## 📋 预期输出

### 训练输出

- **输出目录**: `results/full_dataset_training/run_YYYYMMDD_HHMMSS/`
- **Snapshot ID**: `snapshot_id.txt`
- **训练日志**: 详细训练过程

### 80/20评估输出

- **输出目录**: `output-dir/run_YYYYMMDD_HHMMSS/`
- **报告文件**: `report_df.csv`
- **Top20时间序列**: `ridge_top20_timeseries.csv`
- **图表**: `top20_vs_qqq.png`, `top20_vs_qqq_cumulative.png`

## 🔍 验证命令

### 检查子集文件

```python
import pandas as pd
df = pd.read_parquet(r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet")
print(f"Shape: {df.shape}")
print(f"Index: {df.index.names}")
print(f"Unique tickers: {df.index.get_level_values('ticker').nunique()}")
print(f"Unique dates: {df.index.get_level_values('date').nunique()}")
```

### 检查训练状态

```bash
# 检查最新的训练输出目录
ls -lt results/full_dataset_training/run_*/

# 检查snapshot ID
cat results/full_dataset_training/run_*/snapshot_id.txt
```

### 检查80/20评估状态

```bash
# 检查最新的评估输出目录
ls -lt output-dir/run_*/

# 查看评估报告
cat output-dir/run_*/report_df.csv
```

## 📝 注意事项

1. **格式一致性**: ✅ 子集格式与原始文件完全一致
2. **代表性**: ⚠️ 子集是随机选择的，可能不完全代表整个市场
3. **性能差异**: ⚠️ 使用子集训练的模型性能可能与全量数据有差异
4. **仅用于测试**: ⚠️ 建议仅用于快速测试，生产环境应使用全量数据

## 🚀 下一步

1. **等待训练完成**: 检查 `results/full_dataset_training/run_*/` 目录
2. **等待评估完成**: 检查 `output-dir/run_*/` 目录
3. **分析结果**: 比较子集结果与全量数据结果
4. **验证格式**: 确保所有数据格式一致

---

**创建时间**: 2025-01-20

**状态**: ✅ **子集已创建，训练和评估已启动**
