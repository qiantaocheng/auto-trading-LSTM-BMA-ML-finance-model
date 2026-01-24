# 四个第一层模型输入点完整验证报告

## 📊 当前状态

### 1. T10_ALPHA_FACTORS (定义列表) - 15个因子

**位置**: `bma_models/simple_25_factor_engine.py:58-78`

```python
T10_ALPHA_FACTORS = [
    'momentum_10d',              # ✅ NEW
    'liquid_momentum',           # ✅
    'obv_momentum_40d',          # ✅
    'ivol_30',                   # ✅
    'rsi_21',                    # ✅
    'trend_r2_60',               # ✅
    'near_52w_high',             # ✅
    'ret_skew_30d',              # ✅
    'blowoff_ratio_30d',         # ✅
    # 'hist_vol_40d',            # ❌ REMOVED
    'atr_ratio',                 # ✅
    # 'bollinger_squeeze',       # ❌ REMOVED
    'vol_ratio_30d',             # ✅
    'price_ma60_deviation',      # ✅
    '5_days_reversal',           # ⚠️ 在定义中，但不在 t10_selected 中
    'downside_beta_ewm_21',      # ⚠️ 在定义中，但不在 t10_selected 中
    'feat_vol_price_div_30d',    # ✅
]
```

---

### 2. t10_selected (实际使用的列表) - 13个因子

**位置**: `bma_models/量化模型_bma_ultra_enhanced.py:3283-3299`

```python
t10_selected = [
    "momentum_10d",              # ✅
    "ivol_30",                   # ✅
    # "hist_vol_40d",            # ❌ REMOVED
    "near_52w_high",             # ✅
    "rsi_21",                    # ✅
    "vol_ratio_30d",             # ✅
    "trend_r2_60",               # ✅
    "liquid_momentum",          # ✅
    "obv_momentum_40d",          # ✅
    "atr_ratio",                 # ✅
    "ret_skew_30d",              # ✅
    "price_ma60_deviation",      # ✅
    "blowoff_ratio_30d",         # ✅
    # "bollinger_squeeze",       # ❌ REMOVED
    "feat_vol_price_div_30d",    # ✅
]
```

**缺失的因子**:
- ⚠️ `5_days_reversal` - 在 T10_ALPHA_FACTORS 中定义，但不在 t10_selected 中
- ⚠️ `downside_beta_ewm_21` - 在 T10_ALPHA_FACTORS 中定义，但不在 t10_selected 中

---

### 3. 四个第一层模型的输入点

**位置**: `bma_models/量化模型_bma_ultra_enhanced.py:3301-3306`

```python
base_overrides = {
    'elastic_net': list(t10_selected),   # 13个因子
    'catboost': list(t10_selected),      # 13个因子
    'xgboost': list(t10_selected),       # 13个因子
    'lambdarank': list(t10_selected),    # 13个因子
}
```

**特征选择方法**: `_get_first_layer_feature_cols_for_model()` (line 6792)
- 从 `first_layer_feature_overrides` 获取每个模型的因子列表
- 如果因子列表为 None，使用所有可用因子
- 如果因子列表存在，只使用列表中的因子

**训练时使用** (line 11299):
```python
use_cols = self._get_first_layer_feature_cols_for_model(
    name, 
    list(X_train.columns), 
    available_cols=X_train.columns
)
X_train_use = X_train[use_cols].copy()
X_val_use = X_val[use_cols].copy()
```

**预测时使用** (line 5500):
```python
cols = self._get_first_layer_feature_cols_for_model(
    model_name, 
    list(X.columns), 
    available_cols=X.columns
)
```

---

### 4. Direct Predict 输入点

**位置**: `bma_models/量化模型_bma_ultra_enhanced.py:5356-5364`

```python
base_features = [
    'momentum_10d',
    'ivol_30', 'near_52w_high', 'rsi_21', 'vol_ratio_30d',
    'trend_r2_60', 'liquid_momentum', 'obv_momentum_40d', 'atr_ratio',
    'ret_skew_30d', 'price_ma60_deviation', 'blowoff_ratio_30d',
    # 'bollinger_squeeze',  # ❌ REMOVED
    # 'hist_vol_40d',       # ❌ REMOVED
    'feat_vol_price_div_30d',
]
```

**状态**: ✅ 与 `t10_selected` 一致 (13个因子)

---

### 5. 80/20 OOS 评估输入点

**位置**: `scripts/time_split_80_20_oos_eval.py:48-100`

**特征对齐方法**: `align_test_features_with_model()`
- 从训练好的模型中获取特征列表 (`model.feature_names_in_` 等)
- 自动对齐测试数据的特征与训练时的特征

**状态**: ✅ 自动使用训练时的特征列表

---

## ✅ 验证结果

### 四个第一层模型的输入点

| 模型 | 输入来源 | 因子数量 | 包含 momentum_10d | 不包含已删除因子 | 状态 |
|------|---------|---------|-------------------|-----------------|------|
| **ElasticNet** | `base_overrides['elastic_net']` = `t10_selected` | 13 | ✅ | ✅ | ✅ 正确 |
| **CatBoost** | `base_overrides['catboost']` = `t10_selected` | 13 | ✅ | ✅ | ✅ 正确 |
| **XGBoost** | `base_overrides['xgboost']` = `t10_selected` | 13 | ✅ | ✅ | ✅ 正确 |
| **LambdaRank** | `base_overrides['lambdarank']` = `t10_selected` | 13 | ✅ | ✅ | ✅ 正确 |

### Direct Predict 输入点

| 功能 | 输入来源 | 因子数量 | 状态 |
|------|---------|---------|------|
| **Direct Predict** | `base_features` | 13 | ✅ 正确 |

### 80/20 OOS 评估输入点

| 功能 | 输入来源 | 状态 |
|------|---------|------|
| **80/20 OOS** | 模型训练时的特征列表 (自动对齐) | ✅ 正确 |

---

## 📋 最终确认的因子列表 (13个)

**所有四个第一层模型、Direct Predict 和 80/20 OOS 都使用以下 13个因子**:

1. ✅ `momentum_10d` - NEW: 10-day short-term momentum
2. ✅ `ivol_30` - Idiosyncratic Volatility (30-day)
3. ✅ `near_52w_high` - Distance to 52-week High
4. ✅ `rsi_21` - Relative Strength Index (21-period)
5. ✅ `vol_ratio_30d` - Volume Ratio (30-day)
6. ✅ `trend_r2_60` - Trend R² (60-day)
7. ✅ `liquid_momentum` - Liquidity-adjusted Momentum
8. ✅ `obv_momentum_40d` - OBV Momentum (40-day)
9. ✅ `atr_ratio` - ATR Ratio
10. ✅ `ret_skew_30d` - Return Skewness (30-day)
11. ✅ `price_ma60_deviation` - Price Deviation from MA60
12. ✅ `blowoff_ratio_30d` - Blowoff Ratio (30-day)
13. ✅ `feat_vol_price_div_30d` - Volume-Price Divergence (30-day)

**已删除的因子** (确认不在任何输入点中):
- ❌ `bollinger_squeeze` - 已从所有输入点删除
- ❌ `hist_vol_40d` - 已从所有输入点删除

**未使用的因子** (在 T10_ALPHA_FACTORS 中定义，但不在实际使用的列表中):
- ⚠️ `5_days_reversal` - 在 T10_ALPHA_FACTORS 中，但不在 t10_selected 中
- ⚠️ `downside_beta_ewm_21` - 在 T10_ALPHA_FACTORS 中，但不在 t10_selected 中

---

## 🎯 结论

✅ **所有四个第一层模型的输入点都正确！**

- ✅ 四个模型都使用相同的 `t10_selected` 列表 (13个因子)
- ✅ Direct Predict 使用相同的因子列表 (13个因子)
- ✅ 80/20 OOS 自动使用训练时的特征列表
- ✅ 已删除的因子 (`bollinger_squeeze`, `hist_vol_40d`) 不在任何输入点中
- ✅ 新增的因子 (`momentum_10d`) 在所有输入点中
- ✅ 特征选择方法 `_get_first_layer_feature_cols_for_model()` 正确工作

**注意**: 
- `5_days_reversal` 和 `downside_beta_ewm_21` 在 T10_ALPHA_FACTORS 中定义，但不在实际使用的 t10_selected 中
- 如果需要使用它们，需要添加到 `t10_selected` 和 `base_features` 中
- 如果不需要使用它们，可以考虑从 T10_ALPHA_FACTORS 中删除，以保持一致性

---

**最后更新**: 2025-01-20  
**状态**: ✅ 验证完成 - 所有输入点正确
