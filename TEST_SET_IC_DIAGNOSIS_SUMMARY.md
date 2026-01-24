# 测试集IC异常高问题诊断总结

## 问题

- **OOF IC正常**（0.01-0.09）
- **但测试集IC异常高**（XGBoost: 0.9387, LambdaRank: 0.8272）

---

## 已检查的代码

### 1. 特征对齐 ✅
- `align_test_features_with_model` 函数正确对齐特征
- 没有标准化（如果训练时已标准化，这是正确的）

### 2. IC计算逻辑 ✅
- `calculate_newey_west_hac_ic` 函数逻辑正确
- 先按日计算横截面correlation，再对日度IC序列做HAC

### 3. 目标变量获取 ✅
- `actual_target = date_data['target']` 正确获取未来收益
- 这是正确的，因为我们要预测未来收益

---

## 可能的问题源

### 1. 测试集特征可能包含未来信息 ⚠️

**检查点**：
- 测试集特征是否在某个地方被错误处理？
- 特征是否包含了未来信息？

**需要检查**：
- 测试集数据文件中的特征是否包含未来信息
- 测试集特征处理代码是否有问题

### 2. 测试集预测和实际值的时间对齐问题 ⚠️

**检查点**：
- 预测和实际值的时间对齐是否正确？
- 是否有时间错位？

**需要检查**：
- `predictions_df` 中的 `date`, `ticker`, `prediction`, `actual` 是否正确对齐
- 是否有时间错位

### 3. 测试集数据本身的问题 ⚠️

**检查点**：
- 测试集数据文件中的特征或目标变量是否有问题？
- 数据是否被错误处理？

**需要检查**：
- 测试集数据文件的内容
- 特征和目标变量的分布

---

## 建议的下一步

### 1. 修改评估脚本保存predictions DataFrame

修改 `time_split_80_20_oos_eval.py`，在计算IC之前保存predictions DataFrame：

```python
# 保存predictions DataFrame用于诊断
predictions_file = run_dir / f"{model_name}_predictions.csv"
predictions.to_csv(predictions_file, index=False)
logger.info(f"Saved predictions for diagnosis: {predictions_file}")
```

### 2. 运行诊断脚本

使用保存的predictions DataFrame运行诊断脚本：

```bash
python scripts/diagnose_test_set_ic_issue.py \
    --results-dir "results/t10_time_split_80_20_final/run_20260123_005132" \
    --model-name "xgboost"
```

### 3. 检查测试集数据文件

检查测试集数据文件中的特征和目标变量：

```python
# 加载测试集数据
test_data = df.loc[(df.index.get_level_values('date') >= test_start) & 
                   (df.index.get_level_values('date') <= test_end)]

# 检查特征分布
# 检查目标变量分布
# 检查是否有未来信息泄露
```

---

## 关键问题

**为什么OOF IC正常，但测试集IC异常高？**

可能的原因：
1. **测试集特征可能包含未来信息**（最可能）
2. **测试集预测和实际值的时间对齐问题**
3. **测试集数据本身的问题**

---

**创建时间**: 2026-01-23
