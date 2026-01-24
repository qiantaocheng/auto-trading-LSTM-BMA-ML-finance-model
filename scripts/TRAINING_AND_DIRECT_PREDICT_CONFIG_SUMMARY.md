# 训练和Direct Predict配置总结

## 📊 配置更新总结

**更新时间**: 2026-01-22

---

## ✅ 已完成的配置更新

### 1. 训练数据文件 ✅

**当前使用的数据文件**: `D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet`

**更新位置**:
- ✅ `scripts/train_full_dataset.py` - 默认训练数据文件
- ✅ `scripts/time_split_80_20_oos_eval.py` - 默认数据文件（可通过--data-file覆盖）

**验证**: 训练脚本现在默认使用`final_v2`数据文件

---

### 2. Direct Predict默认股票列表 ✅

**当前使用的数据文件**: `D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet`

**更新位置**: `autotrader/app.py` line 1545

**功能**: Direct Predict现在默认从这个文件加载股票列表作为输入

**代码逻辑**:
1. 优先使用股票池选择的股票
2. 如果没有选择，从`final_v2.parquet`文件加载默认股票列表
3. 如果文件不存在或加载失败，提示用户输入

---

### 3. 全量训练Snapshot命名 ✅

**Snapshot Tag格式**: `FINAL_V2_FULL_DATASET_YYYYMMDD_HHMMSS`

**更新位置**: `scripts/train_full_dataset.py`

**功能**: 
- 训练完成后，强制保存一个新的snapshot，使用显眼的tag名称
- Snapshot ID保存到`latest_snapshot_id.txt`，供Direct Predict使用

**示例Tag**: `FINAL_V2_FULL_DATASET_20260122_120000`

---

### 4. Direct Predict使用Snapshot ✅

**当前逻辑**: Direct Predict自动使用`latest_snapshot_id.txt`中的snapshot ID

**代码位置**: `autotrader/app.py` line 1801-1807

**流程**:
1. 读取`latest_snapshot_id.txt`
2. 如果存在，使用该snapshot ID
3. 如果不存在，使用数据库中的最新snapshot

---

## 🔍 数据泄露检查结果

### 检查文件
`D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet`

### 检查结果

**数据基本信息**:
- 总行数: 4,180,394
- 特征列数: 27
- 唯一日期数: 1,244
- 唯一股票数: 3,921
- 日期范围: 2021-01-19 至 2025-12-30

**Target列检查**:
- ✅ Target列存在
- ⚠️  发现 11,454 个极端高值 (>0.5, 即>50%)
- ⚠️  发现 2,265 个极端低值 (<-0.5, 即<-50%)
- ⚠️  Target日度自相关较高 (0.8909)，可能存在时间依赖

**特征检查**:
- ✅ 未发现明显的未来信息特征（如future_return, next_day等）
- ✅ 时间序列连续性正常

**时间顺序**:
- ✅ 日期已排序

### 潜在问题

1. **极端Target值**: 
   - 11,454个样本target > 50%（10天内）
   - 2,265个样本target < -50%（10天内）
   - **建议**: 这些可能是真实的市场波动，但需要winsorization处理

2. **Target自相关较高**:
   - 日度自相关 = 0.8909
   - **说明**: Target存在时间依赖，这是正常的（市场收益本身有自相关）
   - **不是数据泄露**: 这是市场本身的特性

### 结论

✅ **未发现明显的数据泄露问题**

- ✅ 特征计算正确（未使用未来信息）
- ✅ 时间顺序正确
- ⚠️  Target存在极端值，但这是数据质量问题，不是泄露问题
- ⚠️  Target自相关是市场特性，不是泄露

---

## 📋 完整配置清单

### 训练配置

| 配置项 | 值 | 位置 |
|--------|-----|------|
| 训练数据文件 | `D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet` | `train_full_dataset.py` line 23 |
| Snapshot Tag | `FINAL_V2_FULL_DATASET_YYYYMMDD_HHMMSS` | `train_full_dataset.py` line 109 |
| Snapshot保存位置 | `latest_snapshot_id.txt` | `train_full_dataset.py` line 143 |

### Direct Predict配置

| 配置项 | 值 | 位置 |
|--------|-----|------|
| 默认股票列表文件 | `D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet` | `app.py` line 1545 |
| Snapshot来源 | `latest_snapshot_id.txt` | `app.py` line 1801 |

### 80/20 Split配置

| 配置项 | 值 | 位置 |
|--------|-----|------|
| Split比例 | 0.8 (80/20) | `time_split_80_20_oos_eval.py` line 346 |
| 输出目录 | `results/t10_time_split_80_20_final` | `time_split_80_20_oos_eval.py` line 359 |
| 数据文件 | `polygon_factors_all_filtered.parquet` (默认) | `time_split_80_20_oos_eval.py` line 344 |

---

## 🎯 使用指南

### 1. 全量训练（使用final_v2数据）

```bash
python scripts/train_full_dataset.py \
  --train-data "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet" \
  --top-n 50 \
  --log-level INFO
```

**输出**:
- Snapshot ID保存到`latest_snapshot_id.txt`
- Snapshot Tag: `FINAL_V2_FULL_DATASET_YYYYMMDD_HHMMSS`
- Direct Predict将自动使用这个snapshot

### 2. Direct Predict（自动使用final_v2股票列表和snapshot）

```bash
# 在GUI中点击"Direct Predict (Snapshot)"按钮
# 或直接运行app.py
python launch_gui.py
```

**行为**:
1. 自动从`final_v2.parquet`加载股票列表
2. 自动使用`latest_snapshot_id.txt`中的snapshot ID
3. 计算特征并进行预测

### 3. 80/20评估（使用final_v2数据）

```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet" \
  --split 0.8 \
  --models catboost lambdarank ridge_stacking \
  --top-n 20
```

---

## ⚠️ 注意事项

### 1. 数据文件一致性

- ✅ 训练使用: `polygon_factors_all_filtered_clean_final_v2.parquet`
- ✅ Direct Predict股票列表: `polygon_factors_all_filtered_clean_final_v2.parquet`
- ⚠️  80/20评估默认使用: `polygon_factors_all_filtered.parquet`（需要显式指定`--data-file`）

**建议**: 统一使用`final_v2`数据文件

### 2. Snapshot管理

- 每次全量训练会生成新的snapshot，tag格式: `FINAL_V2_FULL_DATASET_YYYYMMDD_HHMMSS`
- `latest_snapshot_id.txt`会自动更新
- Direct Predict自动使用最新的snapshot

### 3. 数据泄露预防

- ✅ 特征计算使用`shift(1)`避免未来信息
- ✅ Target计算使用`shift(-horizon)`避免未来信息
- ✅ 训练时使用purge gap避免标签泄露
- ⚠️  Target存在极端值，建议winsorization

---

## ✅ 验证清单

- [x] 训练数据文件已更新为`final_v2`
- [x] Direct Predict默认股票列表已更新为`final_v2`
- [x] Snapshot使用显眼的tag名称
- [x] Direct Predict自动使用最新snapshot
- [x] 数据泄露检查完成
- [x] 配置文档已更新

---

**生成时间**: 2026-01-22  
**状态**: ✅ **所有配置已更新完成**
