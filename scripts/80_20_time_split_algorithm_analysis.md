# 80/20 Time Split 测试平均算法分析

## 算法流程概述

您的理解基本正确，但有一些细节需要澄清。以下是完整的算法流程：

### 1. 数据准备阶段

**关键点：`actual`列已经是T+10的收益率**

- 测试数据（`test_data`）在加载时已经包含`target`列
- `target`列的计算公式（在数据预处理阶段完成）：
  ```python
  target = groupby('ticker')['Close'].pct_change(horizon).shift(-horizon)
  # 其中 horizon = 10（T+10）
  ```
- 这意味着：对于日期`pred_date`，`target`列存储的是从`pred_date`到`pred_date+10`的**前向收益率**

### 2. 预测循环（逐日预测）

**代码位置：** `time_split_80_20_oos_eval.py` 行1045-1235

对每个预测日期`pred_date`（T+0）：

1. **提取当日数据**：
   ```python
   date_data = test_data.xs(pred_date, level='date', drop_level=True)
   ```

2. **获取特征和实际收益率**：
   ```python
   X = date_data[all_feature_cols]  # T+0的特征
   actual_target = date_data['target']  # 已经是T+10的收益率
   ```

3. **生成预测**：
   - 使用T+0的特征进行预测，得到`prediction`列

4. **保存预测结果**：
   ```python
   pred_df = pd.DataFrame({
       'date': pred_date,           # T+0预测日期
       'ticker': tickers,
       'prediction': pred_values,   # T+0的预测分数
       'actual': actual_target      # T+10的实际收益率（已预先计算）
   })
   ```

### 3. 计算平均收益（Top10示例）

**代码位置：** `calculate_group_returns_standalone()` 行290-352

#### 步骤1：按日期分组，计算每日Top10收益

```python
for date, date_group in predictions.groupby('date'):
    # 1. 按prediction排序
    sorted_group = valid.sort_values('prediction', ascending=False)
    
    # 2. 取Top10
    top_n_group = sorted_group.head(10)
    
    # 3. 计算Top10的actual中位数（actual已经是T+10收益率）
    top_return = float(top_n_group['actual'].median())
    
    rows.append({
        'date': date,
        'top_return': top_return  # 该日期的Top10中位数收益
    })
```

**关键理解：**
- 对于每个预测日期（T+0），我们：
  1. 使用T+0的`prediction`排序，选出Top10股票
  2. 使用这些股票的`actual`（T+10收益率）计算中位数
  3. 得到该日期的Top10中位数收益

#### 步骤2：对所有日期求中位数

```python
ts_df = pd.DataFrame(rows).sort_values('date')

summary = {
    'avg_top_return': float(ts_df['top_return'].median()),  # 所有日期的中位数
    ...
}
```

**关键理解：**
- 不是简单的平均（mean），而是**中位数（median）**
- 对所有预测日期的Top10中位数收益再求一次中位数

### 4. 预测日期数量

**代码位置：** 行1000-1006

```python
rebalance_dates = pd.to_datetime(rebalance_dates).tz_localize(None)
# 测试期的所有交易日
```

- 预测日期数量 = 测试期的交易日数量
- 如果测试期是1年（约252个交易日），则大约有252次预测
- 实际数量取决于测试期的长度

## 算法总结

### 您的理解 vs 实际实现

| 您的理解 | 实际实现 | 说明 |
|---------|---------|------|
| T+0那天预测Top10 | ✅ 正确 | 在T+0使用特征预测，选出Top10 |
| 对照T+10的真实结果 | ✅ 正确 | `actual`列已经是T+10收益率 |
| 计算平均/中位数收益 | ⚠️ 部分正确 | **使用中位数（median），不是平均（mean）** |
| 对约252次预测做平均 | ⚠️ 部分正确 | **对约252次预测的Top10中位数收益再求中位数** |

### 完整算法流程

```
对于每个预测日期 pred_date (T+0):
  ├─ 1. 使用T+0的特征生成预测分数 (prediction)
  ├─ 2. 获取预先计算的T+10收益率 (actual)
  └─ 3. 保存: (date=pred_date, ticker, prediction, actual)

对所有预测结果:
  ├─ 按日期分组
  ├─ 对每个日期:
  │   ├─ 按prediction排序
  │   ├─ 选出Top10股票
  │   └─ 计算Top10的actual中位数 → top_return[date]
  └─ 对所有日期的top_return求中位数 → avg_top_return
```

### 关键代码位置

1. **预测循环**：`time_split_80_20_oos_eval.py:1045-1235`
2. **Top10收益计算**：`calculate_group_returns_standalone()` 行290-352
3. **Bucket收益计算**：`calculate_bucket_returns_standalone()` 行531-578

### 注意事项

1. **使用中位数而非均值**：
   - 每日Top10收益：`top_return = top_n_group['actual'].median()`
   - 最终平均收益：`avg_top_return = ts_df['top_return'].median()`
   - 这是为了减少异常值的影响

2. **actual列已预先计算**：
   - `actual`列在数据加载时就已经是T+10的收益率
   - 不需要在预测时重新计算

3. **重叠观测（Overlapping Observations）**：
   - 由于每日都做预测，观测是重叠的（overlapping）
   - 因此需要使用HAC（Heteroskedasticity and Autocorrelation Consistent）标准误进行统计推断
   - 代码中使用Newey-West或Hansen-Hodrick方法

4. **MultiIndex数据结构**：
   - `test_data`使用MultiIndex: `(date, ticker)`
   - 这确保了每个日期-股票组合的唯一性

## 示例

假设测试期有3个交易日：

```
日期1 (T+0):
  - 预测Top10: [A, B, C, D, E, F, G, H, I, J]
  - Top10的T+10收益率: [0.05, 0.03, 0.04, 0.02, 0.06, 0.01, 0.04, 0.03, 0.05, 0.02]
  - 中位数: 0.035

日期2 (T+0):
  - 预测Top10: [K, L, M, N, O, P, Q, R, S, T]
  - Top10的T+10收益率: [0.08, 0.06, 0.07, 0.05, 0.09, 0.04, 0.06, 0.07, 0.08, 0.05]
  - 中位数: 0.065

日期3 (T+0):
  - 预测Top10: [U, V, W, X, Y, Z, AA, AB, AC, AD]
  - Top10的T+10收益率: [0.02, 0.01, 0.03, 0.00, 0.02, -0.01, 0.01, 0.02, 0.03, 0.00]
  - 中位数: 0.015

最终结果:
  - avg_top_return = median([0.035, 0.065, 0.015]) = 0.035
```

## 结论

您的理解基本正确，主要区别在于：
1. **使用中位数而非均值**（减少异常值影响）
2. **actual列已预先计算**（不需要在预测时计算T+10收益）
3. **预测日期数量取决于测试期长度**（不一定是252，取决于实际测试期）
