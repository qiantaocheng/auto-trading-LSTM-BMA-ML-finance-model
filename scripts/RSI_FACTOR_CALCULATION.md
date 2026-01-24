# RSI 因子计算详解

**因子名称**: `rsi_21`  
**文件位置**: `bma_models/simple_25_factor_engine.py`  
**方法**: `_compute_mean_reversion_factors()` → `_rsi21()`

---

## 📊 计算逻辑

### 1. 基础 RSI 计算

```python
def _rsi21(x: pd.Series) -> pd.Series:
    # 1. 计算价格变化
    ret = x.diff()
    
    # 2. 分离上涨和下跌
    # 🔥 FIX: Shift for pre-market prediction (use previous day's RSI)
    gain = ret.clip(lower=0).rolling(21, min_periods=1).mean().shift(1)
    loss = (-ret).clip(lower=0).rolling(21, min_periods=1).mean().shift(1)
    
    # 3. 计算相对强度 (RS)
    rs = gain / (loss + 1e-10)
    
    # 4. 转换为 RSI (0-100)
    rsi = 100 - (100 / (1 + rs))
```

**关键点**:
- ✅ **周期**: 21 天（平滑信号）
- ✅ **Shift(1)**: gain 和 loss 都使用 `shift(1)`，确保开盘前预测使用前一日数据
- ✅ **最小周期**: `min_periods=1` 允许早期数据计算
- ✅ **数值稳定性**: `loss + 1e-10` 防止除零

---

### 2. T+10 策略的 Regime 调整

```python
# Regime context for T+10: invert RSI in bearish regime (price below MA200)
if int(getattr(self, "horizon", 5) or 5) == 10:
    # 🔥 FIX: Shift MA200 for pre-market prediction
    ma200 = x.rolling(200, min_periods=60).mean().shift(1)
    bull = (x.shift(1) > ma200).astype(float)  # Use previous day's price vs MA200
    
    # 牛市: 使用原始 RSI
    # 熊市: 反转 RSI (100 - rsi)
    rsi = (bull * rsi) + ((1.0 - bull) * (100.0 - rsi))
```

**逻辑说明**:
- **牛市** (价格 > MA200): 使用原始 RSI
- **熊市** (价格 < MA200): 反转 RSI (`100 - rsi`)
- **原因**: T+10 策略在熊市中，低 RSI（超卖）可能意味着反弹机会

---

### 3. 标准化输出

```python
return (rsi - 50) / 50  # Standardize to [-1, 1]
```

**转换**:
- RSI 原始范围: `[0, 100]`
- 标准化后: `[-1, 1]`
- 中心点: `0` (对应 RSI=50)
- 正值: RSI > 50 (相对强势)
- 负值: RSI < 50 (相对弱势)

---

## 🔧 Shift(1) 策略详解

### 为什么需要 Shift(1)?

**场景**: 开盘前预测开盘买入

**问题**: 如果使用当天数据计算 RSI，会泄露未来信息

**解决方案**: 所有滚动统计都使用 `shift(1)`

```python
# ❌ 错误 (泄露未来信息)
gain = ret.clip(lower=0).rolling(21).mean()  # 包含当天数据

# ✅ 正确 (使用前一日数据)
gain = ret.clip(lower=0).rolling(21).mean().shift(1)  # 使用前一日统计
```

---

## 📈 RSI 值解读

### 标准化后的 RSI_21 值

| RSI 原始值 | 标准化值 | 含义 |
|-----------|---------|------|
| 0 | -1.0 | 极度超卖 |
| 30 | -0.4 | 超卖 |
| 50 | 0.0 | 中性 |
| 70 | 0.4 | 超买 |
| 100 | 1.0 | 极度超买 |

### T+10 策略的特殊处理

**熊市 Regime** (价格 < MA200):
- 原始 RSI = 30 → 标准化 = -0.4
- 反转后 RSI = 70 → 标准化 = 0.4
- **含义**: 在熊市中，超卖信号被转换为超买信号（可能反弹）

---

## 🔍 计算流程总结

```
1. 价格序列 (Close)
   ↓
2. 计算价格变化 (diff)
   ↓
3. 分离上涨/下跌 (clip)
   ↓
4. 21日滚动平均 (rolling mean)
   ↓
5. Shift(1) - 使用前一日统计 ⚠️
   ↓
6. 计算 RS = gain / loss
   ↓
7. 转换为 RSI = 100 - (100 / (1 + RS))
   ↓
8. T+10 Regime 调整 (如果适用)
   ↓
9. 标准化到 [-1, 1]
   ↓
10. 输出 rsi_21
```

---

## ✅ 验证要点

### 1. Shift 正确性
```python
# 验证: gain 和 loss 都使用了 shift(1)
assert gain.index[0] == loss.index[0]  # 索引对齐
# gain[date] 应该基于 date-1 及之前的数据
```

### 2. Regime 调整
```python
# 验证: T+10 时，熊市 RSI 被反转
if horizon == 10 and price < ma200:
    assert rsi_adjusted > rsi_original  # 反转后值更大
```

### 3. 数值范围
```python
# 验证: 标准化后的值在 [-1, 1] 范围内
assert -1.0 <= rsi_standardized <= 1.0
```

---

## 📝 代码位置

**文件**: `bma_models/simple_25_factor_engine.py`  
**行号**: 1193-1213

```python
def _compute_mean_reversion_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
    """🔥 Compute mean reversion factors: rsi_21 (smoother RSI), price_ma60_deviation"""
    
    def _rsi21(x: pd.Series) -> pd.Series:
        ret = x.diff()
        # 🔥 FIX: Shift for pre-market prediction (use previous day's RSI)
        gain = ret.clip(lower=0).rolling(21, min_periods=1).mean().shift(1)
        loss = (-ret).clip(lower=0).rolling(21, min_periods=1).mean().shift(1)
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        # Regime context for T+10: invert RSI in bearish regime
        if int(getattr(self, "horizon", 5) or 5) == 10:
            ma200 = x.rolling(200, min_periods=60).mean().shift(1)
            bull = (x.shift(1) > ma200).astype(float)
            rsi = (bull * rsi) + ((1.0 - bull) * (100.0 - rsi))
        return (rsi - 50) / 50
    
    rsi = grouped['Close'].transform(_rsi21)
    
    return pd.DataFrame({
        'rsi_21': rsi,
        'price_ma60_deviation': price_ma60_dev
    }, index=data.index)
```

---

## 🎯 关键特性总结

| 特性 | 值/说明 |
|------|---------|
| **周期** | 21 天 |
| **Shift** | ✅ Yes (gain, loss, MA200) |
| **标准化** | [-1, 1] |
| **Regime 调整** | ✅ Yes (T+10 熊市反转) |
| **最小周期** | 1 天 |
| **数值稳定性** | ✅ Yes (1e-10 防除零) |

---

**最后更新**: 2025-01-20  
**状态**: ✅ 已实现并验证 - 适用于开盘前预测
