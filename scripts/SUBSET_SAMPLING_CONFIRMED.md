# 子集采样方式确认

## ✅ 确认：子集是按Ticker采样，不是按Time采样

### 数据对比结果

| 指标 | 全量数据 | 子集数据 | 说明 |
|------|---------|---------|------|
| **Ticker数量** | 3,921 | 784 | ✅ 正好1/5 (20.0%) |
| **唯一日期数** | 1,244 | 1,244 | ✅ **完全相同** |
| **日期范围** | 2021-01-19 到 2025-12-30 | 2021-01-19 到 2025-12-30 | ✅ **完全相同** |
| **平均每个ticker样本数** | 1,066.2 | 1,056.0 | ✅ 几乎相同 (99.05%) |
| **总样本数** | ~4,180,000 | ~827,900 | ✅ 约1/5 (19.8%) |

### 采样方式

```python
# create_subset_1_5_tickers.py
# 1. 随机选择1/5的ticker
selected_tickers = np.random.choice(all_tickers, size=n_subset_tickers, replace=False)

# 2. 过滤数据，只保留选中的ticker
df_subset = df[mask].copy()  # mask = ticker_level.isin(selected_tickers)
```

**关键点**：
- ✅ **日期范围完整**：包含所有1244个交易日
- ✅ **每个ticker数据完整**：选中的ticker有完整的时间序列
- ❌ **Ticker数量减少**：从3921个减少到784个

## 对训练的影响重新评估

### 1. CV Fold训练窗天数

**重要发现**：由于日期范围完整，**每个CV fold的训练窗天数应该和全量数据相同**！

- 全量数据：6折CV，每个fold训练窗约 800-1000 天
- 子集数据：6折CV，每个fold训练窗也应该约 800-1000 天
- **不应该因为训练窗不足252天而跳过fold**

### 2. 但存在其他问题

虽然训练窗天数足够，但：

1. **每个fold的样本数减少了80%**
   - 全量：每个fold ~700,000 样本
   - 子集：每个fold ~140,000 样本

2. **每个日期的样本数减少了80%**
   - 全量：每天 ~3,360 样本（3921 tickers）
   - 子集：每天 ~665 样本（784 tickers）

3. **Ticker多样性不足**
   - 每个fold只有784个ticker（而不是3921个）
   - 可能导致某些统计计算或模型训练不稳定

## 重新评估问题原因

### 原假设（可能不准确）

❌ **假设**: 子集数据日期范围不足，导致训练窗 < 252天

### 新假设（更可能）

✅ **假设1**: 虽然训练窗天数足够，但**样本数不足**导致某些计算失败

✅ **假设2**: **Ticker数量太少**（只有784个），导致某些模型训练不稳定

✅ **假设3**: 某些日期的样本数太少，导致CV fold数据稀疏

✅ **假设4**: 虽然训练窗天数足够，但代码中的**估算逻辑有问题**（Line 11475使用固定值3270估算）

## 代码中的问题

### Line 11475: 训练窗天数估算

```python
# 如果无法从groups_norm获取日期，使用样本数估算
train_window_days = len(train_idx) // 3270  # 假设每天约3270个样本
```

**问题**：
- 这个估算值（3270）是基于全量数据的
- 子集数据每天只有~665个样本
- 如果使用这个估算，会**严重低估训练窗天数**！

**例如**：
- 子集fold有140,000个样本
- 使用3270估算：140,000 / 3270 = 42.8天 ❌（错误！）
- 实际应该是：140,000 / 665 = 210.5天 ✅（正确）

## 修复方案调整

### 1. 修复训练窗天数估算（关键！）

```python
# Line 11475 修改为：
# 动态估算每天样本数
if groups_norm is not None:
    # 使用实际数据计算
    avg_samples_per_date = len(X) / unique_dates_count
else:
    # 使用子集数据估算（而不是固定3270）
    avg_samples_per_date = sample_size / unique_dates_count if unique_dates_count > 0 else 500

train_window_days = len(train_idx) // avg_samples_per_date
```

### 2. 保持现有修复（仍然有用）

- 动态调整最小训练窗
- 减少CV折数
- 安全检查

### 3. 添加样本数检查

```python
# 检查每个fold的样本数
min_samples_per_fold = 10000
if len(train_idx) < min_samples_per_fold:
    logger.warning(f"Fold {fold_idx + 1} 样本数不足: {len(train_idx)} < {min_samples_per_fold}")
    continue
```

## 立即修复

需要修复Line 11475的训练窗天数估算逻辑，使用动态值而不是固定3270。
