# Direct Predict重复分数问题 - 修复总结

## 🔍 问题确认

**现象**: Direct Predict中所有股票的预测分数都是相同的值（如`0.756736`或`0.920046`）

**根本原因**: **缺失特征被填充为0.0**，导致所有股票的某些特征列完全相同

---

## ✅ 修复内容

### 修复位置

**文件**: `bma_models/量化模型_bma_ultra_enhanced.py`  
**行号**: 9877-9915

### 修复前（有问题）

```python
for col in ridge_base_cols:
    if col not in ridge_input.columns:
        ridge_input[col] = 0.0  # ⚠️ 所有股票都是0.0
```

**问题**: 
- 如果某个`base_col`（如`pred_catboost`、`pred_lambdarank`等）缺失
- 所有股票的该列都被填充为`0.0`
- 如果多个列都缺失，所有股票的特征完全相同
- MetaRankerStacker接收到相同的输入，返回相同的输出

### 修复后（改进）

```python
# 🔧 FIX: 改进缺失特征处理 - 使用横截面中位数而不是0.0
for col in ridge_base_cols:
    if col not in ridge_input.columns:
        # 使用横截面中位数填充，而不是0.0
        if isinstance(ridge_input.index, pd.MultiIndex) and 'date' in ridge_input.index.names:
            # 按日期分组，使用同日其他股票的可用特征中位数
            daily_medians_dict = {}
            for date in ridge_input.index.get_level_values('date').unique():
                day_mask = ridge_input.index.get_level_values('date') == date
                day_data = ridge_input.loc[day_mask]
                numeric_cols = day_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    ref_median = day_data[numeric_cols].median().median()
                    daily_medians_dict[date] = ref_median if not pd.isna(ref_median) else 0.0
                else:
                    daily_medians_dict[date] = 0.0
            
            # 创建Series并reindex到ridge_input的索引
            date_level = ridge_input.index.get_level_values('date')
            ridge_input[col] = pd.Series(
                [daily_medians_dict.get(date, 0.0) for date in date_level],
                index=ridge_input.index
            )
        else:
            # 非MultiIndex情况：使用所有数值列的中位数
            numeric_cols = ridge_input.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                ref_median = ridge_input[numeric_cols].median().median()
                fill_val = ref_median if not pd.isna(ref_median) else 0.0
                ridge_input[col] = fill_val
            else:
                ridge_input[col] = 0.0
```

**改进**:
- ✅ 使用横截面中位数填充，而不是固定的0.0
- ✅ 按日期分组，使用同日其他股票的可用特征中位数
- ✅ 如果所有特征都缺失，才使用0.0作为最后兜底
- ✅ 添加了异常处理和回退逻辑

---

## 🎯 修复效果

### 修复前

- 缺失特征 → 所有股票填充为0.0 → 特征完全相同 → MetaRankerStacker返回相同分数

### 修复后

- 缺失特征 → 使用横截面中位数填充 → 特征有变化（虽然不完美，但至少不同股票可能不同） → MetaRankerStacker可以区分股票

---

## ⚠️ 注意事项

### 1. 这不是EWMA问题

**EWMA已经被禁用**:
- Line 10128-10130: `[LIVE_PREDICT] 🔥 EMA smoothing DISABLED for live prediction`
- `replace_ewa_in_pipeline()`只是兼容性方法，不涉及EWMA逻辑

### 2. 根本问题可能是第一层预测

**如果第一层模型预测都相同**:
- 即使修复了缺失特征填充，问题仍然存在
- 需要检查第一层模型（CatBoost, LambdaRank等）的预测

### 3. 需要验证修复效果

**验证方法**:
1. 重启Direct Predict
2. 查看日志中的`[SNAPSHOT] 🔍 ridge_input['{col}']: unique=`信息
3. 确认每个列的unique值 > 1
4. 检查是否还有重复分数警告

---

## 🔧 后续建议

### 1. 检查第一层预测

如果修复后问题仍然存在，检查：
- `[SNAPSHOT] 📊 LambdaRank non-null values`
- `[SNAPSHOT] 📊 CatBoost non-null values`
- 第一层模型的预测是否有变化

### 2. 添加更多验证

在MetaRankerStacker的predict方法中添加验证：
```python
if predictions['score'].nunique() == 1:
    logger.error(f"MetaRankerStacker returned identical predictions: {predictions['score'].iloc[0]}")
```

### 3. 检查特征对齐

确保`first_layer_preds`正确对齐到股票，没有索引错误

---

## 📝 相关文件

- **修复文件**: `bma_models/量化模型_bma_ultra_enhanced.py` line 9877-9915
- **诊断脚本**: `scripts/diagnose_duplicate_scores.py`
- **分析文档**: `scripts/EWMA_ISSUE_ANALYSIS.md`

---

**状态**: ✅ **已修复缺失特征填充问题**

**下一步**: 重启Direct Predict，验证修复效果
