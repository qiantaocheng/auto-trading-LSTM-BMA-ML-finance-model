# 收益率计算验证 - 检查是否异常高

## ⚠️ 问题描述

用户反馈：**avg_top_return 37-39% 高的逆天**

每10天35-45%的收益率确实异常高，需要验证计算是否正确。

## 🔍 当前数据

### LambdaRank Top20收益率（每10天）

| 日期 | Top20收益率 (%) | 累计收益率 (%) |
|------|----------------|---------------|
| 2025-01-02 | 35.61 | 35.61 |
| 2025-01-17 | 44.31 | 95.69 |
| 2025-02-03 | 42.44 | 178.74 |
| ... | ... | ... |

**平均**: 37.39% (每10天)

### 如果转换为年化收益率

假设每10天收益率是37.39%：
- 一年约25个10天周期（250个交易日）
- 年化收益率 = (1 + 0.3739)^25 - 1 ≈ **1,000,000%+**

这显然不合理！

## 🔎 需要检查的问题

### 1. actual列的单位问题

**代码位置**: `scripts/time_split_80_20_oos_eval.py` line ~630

```python
top_return_mean = float(top_n_group['actual'].mean())
```

**问题**: `actual`列的值是什么单位？
- 如果是**小数形式**（0.3561 = 35.61%），那么计算正确
- 如果已经是**百分比形式**（35.61），那么被当作小数处理，会导致错误

### 2. 收益率计算方式

**需要确认**:
- `actual`列是如何计算的？
- 是T+10的简单收益率 `(P_t+10 - P_t) / P_t`？
- 还是其他计算方式？

### 3. 数据来源

**需要检查**:
- `actual`列来自哪个数据源？
- 是否使用了正确的价格数据？
- 是否有数据错误或异常值？

## 📋 验证步骤

### 步骤1: 检查actual列的计算

查找代码中`actual`列的创建位置：
```bash
grep -r "actual.*=" scripts/time_split_80_20_oos_eval.py
grep -r "forward.*return" scripts/time_split_80_20_oos_eval.py
```

### 步骤2: 检查数据文件

检查训练数据文件中的target列：
```python
import pandas as pd
df = pd.read_parquet("data/factor_exports/polygon_factors_all_filtered_clean_final_v2.parquet")
# 检查target列的统计信息
print(df['target'].describe())
print(df['target'].head(20))
```

### 步骤3: 手动验证

选择一个具体的日期和ticker，手动计算收益率：
```python
# 例如：2025-01-02的某个ticker
# 检查该ticker在2025-01-02的价格
# 检查该ticker在2025-01-17（10天后）的价格
# 计算收益率 = (P_2025-01-17 - P_2025-01-02) / P_2025-01-02
```

## 🎯 可能的问题

### 问题1: 单位错误

如果`actual`列已经是百分比形式（35.61），但代码当作小数处理（0.3561），那么：
- 实际收益率应该是 35.61% / 100 = 0.3561
- 但代码可能直接使用了35.61，导致收益率被放大100倍

### 问题2: 计算方式错误

如果收益率计算使用了错误的价格：
- 例如使用了调整后的价格，但计算方式不正确
- 或者使用了错误的日期偏移

### 问题3: 数据异常

如果数据本身有问题：
- 某些ticker的价格数据异常
- 或者有极端值没有被过滤

## 📝 建议的修复

### 如果确认是单位问题

修改代码，确保单位一致：
```python
# 如果actual是百分比形式，需要除以100
if actual_is_percentage:
    top_return_mean = float(top_n_group['actual'].mean()) / 100.0
else:
    top_return_mean = float(top_n_group['actual'].mean())
```

### 如果确认是计算问题

检查并修复收益率计算逻辑：
```python
# 确保使用正确的价格和日期
forward_price = get_price_at_date(ticker, date + pd.Timedelta(days=10))
current_price = get_price_at_date(ticker, date)
return = (forward_price - current_price) / current_price
```

## 🔍 下一步行动

1. **立即检查**: `actual`列的计算方式和单位
2. **验证数据**: 手动计算几个样本的收益率
3. **对比基准**: 与QQQ的收益率对比（报告中QQQ的收益率是2-4%，这是合理的）
4. **修复问题**: 如果发现问题，修复计算逻辑

---

**创建时间**: 2026-01-22  
**状态**: ⚠️ **需要验证**
