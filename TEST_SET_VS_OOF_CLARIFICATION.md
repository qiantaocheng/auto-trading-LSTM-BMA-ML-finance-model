# 测试集 vs OOF 数据澄清

## 问题
`D:\trade\results\t10_time_split_80_20_final` 目录下保存的是**测试集预测**还是**OOF预测**？

## 答案
**这是测试集预测，不是OOF预测。**

---

## 详细说明

### 1. `time_split_80_20_oos_eval.py` 脚本的工作流程

```
1. 时间分割（80/20）:
   - 训练期: 前80%的时间（例如：2021-01-19 到 2024-12-16）
   - 测试期: 后20%的时间（例如：2024-12-17 到 2025-01-23）

2. 训练模型:
   - 在训练期数据上训练模型
   - 使用5-fold CV进行训练（生成OOF预测，但不保存到结果目录）

3. 测试集预测:
   - 加载训练好的模型
   - 在测试期数据上进行预测
   - 保存预测结果到 results/t10_time_split_80_20_final/ 目录
```

### 2. 结果目录中的文件

`D:\trade\results\t10_time_split_80_20_final\run_20260123_005132\` 包含：

- ✅ **`report_df.csv`**: 测试集预测的IC、Rank IC等指标
- ✅ **`*_top30_nonoverlap_timeseries.csv`**: 测试集回测时间序列
- ✅ **`*_bucket_returns.csv`**: 测试集分桶收益
- ✅ **`oos_metrics.csv`**: 测试集（Out-of-Sample）指标

**这些都是测试集预测的结果，不是OOF预测。**

---

## OOF预测在哪里？

### OOF预测的生成位置

OOF预测是在**训练过程中**生成的，保存在：

1. **训练结果对象中** (`training_results`):
   ```python
   training_results = {
       'traditional_models': {
           'oof_predictions': {
               'elastic_net': pd.Series(...),
               'xgboost': pd.Series(...),
               'catboost': pd.Series(...),
               'lambdarank': pd.Series(...)
           }
       }
   }
   ```

2. **模型快照中**（如果保存了）:
   - 快照ID保存在 `snapshot_id.txt`
   - 但OOF预测通常不直接保存到文件系统

3. **训练日志中**:
   - 训练过程中的OOF IC和Rank IC会打印到日志
   - 但不保存完整的OOF预测数据

---

## 关键区别总结

| 特征 | 测试集预测 | OOF预测 |
|------|-----------|---------|
| **保存位置** | `results/t10_time_split_80_20_final/` | 训练结果对象中 |
| **时间范围** | 训练期外（后20%） | 训练期内（通过CV） |
| **用途** | 评估真实泛化能力 | 训练第二层模型 |
| **文件** | `report_df.csv`, `oos_metrics.csv` | 不直接保存到文件 |

---

## 如何获取OOF预测？

### 方法1: 从训练结果中提取（实验5完整版）

```python
# 运行完整训练
train_res = model.train_from_document(...)

# 提取OOF预测
training_results = train_res.get('training_results', {})
oof_predictions = training_results['traditional_models']['oof_predictions']
```

### 方法2: 从快照中加载（如果保存了）

```python
# 加载快照
loaded = load_models_from_snapshot(snapshot_id)

# OOF预测通常不保存在快照中
# 需要重新训练或从训练结果中提取
```

---

## 实验5的当前状态

### 简化版（已运行）
- 使用单fold CV + 简单模型
- 估算OOF IC（不是真实的OOF预测）

### 完整版（正在运行）
- 运行完整的5-fold CV训练
- 提取真实的OOF预测
- 比较OOF IC vs 测试集IC

---

## 总结

**`D:\trade\results\t10_time_split_80_20_final` 目录下保存的是测试集预测，不是OOF预测。**

- ✅ 测试集预测：保存在结果目录中
- ❌ OOF预测：保存在训练结果对象中，不直接保存到文件

要获取OOF预测，需要：
1. 运行完整的训练流程
2. 从 `training_results` 中提取
3. 或者修改代码保存OOF预测到文件
