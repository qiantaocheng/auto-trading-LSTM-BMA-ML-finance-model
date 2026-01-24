# 🚨 关键Bug发现：target_new被错误地用作特征！

## 问题

**测试集IC异常高**（XGBoost: 0.9387, LambdaRank: 0.8272）的根本原因找到了！

---

## Bug详情

### 问题代码

```python
# scripts/time_split_80_20_oos_eval.py:1742
exclude_cols = {'target', 'Close', 'ret_fwd_5d', 'sector'}
all_feature_cols = [col for col in test_data.columns if col not in exclude_cols]
```

**问题**：
- ❌ 只排除了 `target` 和 `Close`
- ❌ **没有排除 `target_new` 和 `Close_new`**
- ⚠️ **`target_new` 被错误地当作特征使用！**

### 证据

从快照元数据可以看到，模型训练时使用的特征包括：
- `target_new` ⚠️ **这是目标变量，不应该作为特征！**
- `Close_new` ⚠️ **这可能包含未来信息**

### 影响

如果 `target_new` 被当作特征使用：
1. **模型直接看到了目标变量**
2. **IC会异常高**（接近完美预测）
3. **这是严重的数据泄露！**

---

## 修复

### 修复代码

```python
# 🔥 CRITICAL FIX: Exclude target_new and Close_new to prevent data leakage!
exclude_cols = {'target', 'Close', 'ret_fwd_5d', 'sector', 'target_new', 'Close_new'}
all_feature_cols = [col for col in test_data.columns if col not in exclude_cols]
```

---

## 验证

### 检查测试数据中的列

运行以下命令检查：
```python
import pandas as pd
df = pd.read_parquet('data/factor_exports/polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet')
test_data = df.loc[(df.index.get_level_values('date') >= pd.to_datetime('2024-12-17')) & 
                   (df.index.get_level_values('date') <= pd.to_datetime('2025-01-23'))]
exclude_cols = {'target', 'Close', 'ret_fwd_5d', 'sector'}
all_feature_cols = [col for col in test_data.columns if col not in exclude_cols]
print('Features that might contain target:', [c for c in all_feature_cols if 'target' in c.lower()])
```

---

## 下一步

1. **修复代码**：排除 `target_new` 和 `Close_new`
2. **重新运行评估**：使用修复后的代码
3. **验证IC**：IC应该会大幅下降

---

**发现时间**: 2026-01-23
**严重程度**: 🔴 **严重** - 数据泄露导致IC异常高
