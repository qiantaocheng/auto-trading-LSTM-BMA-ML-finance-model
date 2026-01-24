# Sato 平方根因子计算公式

## 📐 核心公式

### Sato Impact（单日冲击）

```
Sato_Impact(t) = sign(r_t) * |r_t / σ_t| * sqrt(V_rel_t)
```

其中：
- `r_t`: 第t日的对数收益 = `ln(P_t / P_{t-1})`
- `σ_t`: 第t日的波动率 = `std(r_{t-19:t})` (20日滚动标准差)
- `V_rel_t`: 第t日的相对成交量 = `Volume_t / Volume_MA20_t`
- `sign(r_t)`: 收益的符号（+1或-1）
- `sqrt(V_rel_t)`: 相对成交量的平方根

### Sato Factor（因子值）

```
Sato_Factor(t) = sum(Sato_Impact(t-9:t))  # 10日滚动求和
```

---

## 💻 Python 实现代码

### 完整计算流程

```python
import pandas as pd
import numpy as np

def calculate_sato_factor(df, price_col='adj_close', volume_col='Volume', 
                          lookback_days=10, vol_window=20):
    """
    计算Sato平方根因子
    
    Args:
        df: DataFrame with MultiIndex (date, ticker) or single ticker
        price_col: 价格列名（复权后收盘价）
        volume_col: 成交量列名
        lookback_days: Sato因子滚动窗口（默认10天）
        vol_window: 波动率计算窗口（默认20天）
    
    Returns:
        Sato因子 Series
    """
    
    # Step 1: 计算对数收益
    log_ret = np.log(df[price_col] / df[price_col].shift(1))
    
    # Step 2: 计算波动率（20日滚动标准差）
    vol = log_ret.rolling(vol_window).std()
    
    # Step 3: 计算相对成交量
    adv20 = df[volume_col].rolling(vol_window).mean()  # 20日平均成交量
    rel_vol = df[volume_col] / (adv20 + 1e-8)  # 相对成交量
    
    # Step 4: 构建Sato Impact
    normalized_ret = log_ret / (vol + 1e-8)  # 标准化收益
    sato_impact = np.sign(normalized_ret) * np.abs(normalized_ret) * np.sqrt(rel_vol + 1e-8)
    
    # Step 5: 计算Sato Factor（10日滚动求和）
    sato_factor = sato_impact.rolling(lookback_days).sum()
    
    return sato_factor
```

---

## 🔢 详细步骤说明

### Step 1: 对数收益计算

```python
log_ret = np.log(df['adj_close'] / df['adj_close'].shift(1))
```

**为什么使用对数收益？**
- 对数收益具有时间可加性
- 更适合金融建模
- 处理价格变化更稳定

### Step 2: 波动率计算

```python
vol = log_ret.rolling(20).std()
```

**为什么使用20日窗口？**
- 平衡了稳定性和敏感性
- 捕捉短期波动特征
- 与相对成交量的窗口一致

### Step 3: 相对成交量计算

```python
adv20 = df['Volume'].rolling(20).mean()  # 20日平均成交量
rel_vol = df['Volume'] / adv20  # 相对成交量
```

**为什么使用相对成交量？**
- 标准化不同股票的成交量水平
- 捕捉成交量的异常变化
- 横截面可比性

### Step 4: Sato Impact 构建

```python
normalized_ret = log_ret / vol  # 标准化收益（Sharpe-like）
sato_impact = sign(normalized_ret) * |normalized_ret| * sqrt(rel_vol)
```

**公式解释**：
- `normalized_ret`: 收益/波动率（类似Sharpe比率）
- `sign(normalized_ret)`: 保持收益方向
- `sqrt(rel_vol)`: **平方根定律** - 这是Sato理论的核心

**平方根定律的意义**：
- 价格冲击与成交量的平方根成正比
- 大单的影响不是线性的，而是平方根关系
- 这反映了市场深度的非线性特征

### Step 5: Sato Factor 计算

```python
sato_factor = sato_impact.rolling(10).sum()
```

**为什么使用10日滚动求和？**
- 平滑单日噪声
- 捕捉持续的价格冲击
- 与T+10预测horizon对齐

---

## 📊 MultiIndex 数据处理

对于MultiIndex (date, ticker) 格式的数据：

```python
# 按ticker分组计算，确保时间序列正确
def calc_log_ret(group):
    return np.log(group / group.shift(1))

log_ret = df.groupby(level=1)[price_col].apply(calc_log_ret)
# 处理MultiIndex结果，确保索引对齐
if isinstance(log_ret.index, pd.MultiIndex) and len(log_ret.index.names) > 2:
    log_ret = log_ret.droplevel(0)
log_ret.index = df.index
```

---

## 🎯 对照组因子（用于正交化测试）

### 传统动量因子

```python
factor_mom_raw = log_ret.rolling(10).sum()  # 10日累计收益
```

### 波动率因子

```python
factor_vol = log_ret.rolling(20).std()  # 20日波动率
```

---

## 🔍 正交化测试

为了测试Sato因子是否提供增量信息，需要对每一天做横截面回归：

```python
from statsmodels.api import sm

def get_residual(group):
    """对每一天做横截面回归，提取残差"""
    X = group[['factor_mom_raw', 'factor_vol']].values
    X = sm.add_constant(X)  # 添加常数项
    y = group['factor_sato'].values
    
    model = sm.OLS(y, X).fit()
    return pd.Series(model.resid, index=group.index)

# 对每一天做回归
pure_sato_residual = df.groupby(level=0).apply(get_residual)
```

**Pure Sato IC** = IC(pure_sato_residual, fwd_ret_10d)

如果 Pure IC > 0.02，说明Sato因子提供了增量信息。

---

## 📝 完整代码文件

完整实现代码请参考：
- `scripts/sato_factor_calculation.py` - 独立的因子计算模块
- `scripts/test_sato_factor_validation.py` - 完整的验证框架

---

## ⚠️ 注意事项

1. **数据要求**：
   - 必须使用复权后价格（adj_close）
   - 成交量不需要复权
   - 需要足够的历史数据（至少20天）

2. **边界处理**：
   - 使用 `+ 1e-8` 避免除零
   - 使用 `fillna(1.0)` 处理缺失值
   - 使用 `clip()` 限制极端值

3. **MultiIndex处理**：
   - 必须按ticker分组计算
   - 确保索引正确对齐
   - 处理groupby产生的额外索引层级

---

## 🔬 理论背景

Sato平方根定律（Square Root Law）：
- 价格冲击与成交量的平方根成正比
- 反映了市场深度的非线性特征
- 大单的影响不是线性的，而是平方根关系

**数学表达**：
```
Price Impact ∝ sqrt(Volume)
```

**Sato因子**：
```
Sato = sum( sign(return) * |return/volatility| * sqrt(relative_volume) )
```

---

## 📚 参考文献

- Sato, K. (2009). "Square Root Law of Price Impact"
- Almgren, R., & Chriss, N. (2000). "Optimal execution of portfolio transactions"
