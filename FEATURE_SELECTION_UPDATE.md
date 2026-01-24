# 特征选择更新说明

## 更新日期
2026-01-23

## 更新内容

### 只使用指定的13个因子作为输入特征

根据要求，系统现在只使用以下13个因子作为输入：

1. `momentum_10d` - 10日动量
2. `obv_momentum_40d` - 40日OBV动量
3. `ivol_30` - 30日隐含波动率
4. `rsi_21` - 21日RSI
5. `trend_r2_60` - 60日趋势R²
6. `near_52w_high` - 距离52周高点
7. `ret_skew_30d` - 30日收益偏度
8. `atr_ratio` - ATR比率
9. `vol_ratio_30d` - 30日成交量比率
10. `price_ma60_deviation` - 价格偏离60日均线
11. `5_days_reversal` - 5日反转
12. `feat_vol_price_div_30d` - 30日量价背离因子
13. `downside_beta_ewm_21` - 21日下行Beta（EWMA）

### 已移除的因子

以下因子已从特征集中移除：
- `liquid_momentum` - 流动性动量
- `blowoff_ratio_30d` - 30日爆发行情比率
- `making_new_low_5d` - 5日创新低指标（用户明确排除）

---

## 修改的文件

### 1. `bma_models/simple_25_factor_engine.py`

**修改位置**: `T10_ALPHA_FACTORS` 列表

**修改内容**:
- 更新 `T10_ALPHA_FACTORS` 列表，只包含指定的13个因子
- 移除 `liquid_momentum`, `blowoff_ratio_30d`

### 2. `bma_models/量化模型_bma_ultra_enhanced.py`

**修改位置**: `_configure_feature_subsets` 方法

**修改内容**:
- 更新 `T10_ALPHA_FACTORS` 列表（fallback定义）
- 更新 `compulsory_features` 列表，只包含指定的13个因子
- 移除重复的 `trend_r2_60`

### 3. `scripts/time_split_80_20_oos_eval.py`

**修改位置**: 特征选择逻辑（第1740-1745行）

**修改内容**:
- 添加 `allowed_feature_cols` 列表，明确指定只使用这13个因子
- 修改特征选择逻辑，只从 `allowed_feature_cols` 中选择特征

---

## 影响范围

### 训练阶段
- `Simple17FactorEngine` 只会计算和返回这13个因子
- 模型训练时只会使用这13个因子作为输入特征

### 测试/评估阶段
- `time_split_80_20_oos_eval.py` 只会使用这13个因子进行预测
- 确保训练和测试使用相同的特征集

---

## 验证

运行以下命令验证特征选择是否正确：

```python
# 检查因子列表
from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS
print("T10_ALPHA_FACTORS:", T10_ALPHA_FACTORS)
print("Count:", len(T10_ALPHA_FACTORS))

# 应该输出13个因子
```

---

## 注意事项

1. **数据文件**: 确保数据文件中包含这13个因子的列
2. **模型快照**: 如果使用旧的模型快照，可能需要重新训练
3. **特征对齐**: 训练和测试必须使用相同的特征集

---

**更新完成时间**: 2026-01-23
