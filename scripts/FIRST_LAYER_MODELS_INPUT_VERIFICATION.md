# 四个第一层模型输入点验证报告

## ✅ 验证结果

### 1. T10_ALPHA_FACTORS 列表

**位置**: `bma_models/simple_25_factor_engine.py` (line 58-78)

**当前因子列表** (15个):
1. `momentum_10d` ✅
2. `liquid_momentum` ✅
3. `obv_momentum_40d` ✅
4. `ivol_30` ✅
5. `rsi_21` ✅
6. `trend_r2_60` ✅
7. `near_52w_high` ✅
8. `ret_skew_30d` ✅
9. `blowoff_ratio_30d` ✅
10. `atr_ratio` ✅
11. `vol_ratio_30d` ✅
12. `price_ma60_deviation` ✅
13. `5_days_reversal` ✅
14. `downside_beta_ewm_21` ✅
15. `feat_vol_price_div_30d` ✅

**已删除的因子**:
- ❌ `bollinger_squeeze` (已注释)
- ❌ `hist_vol_40d` (已注释)

---

### 2. t10_selected 列表 (实际使用的因子)

**位置**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 3283-3299)

**当前因子列表** (13个):
1. `momentum_10d` ✅
2. `ivol_30` ✅
3. `near_52w_high` ✅
4. `rsi_21` ✅
5. `vol_ratio_30d` ✅
6. `trend_r2_60` ✅
7. `liquid_momentum` ✅
8. `obv_momentum_40d` ✅
9. `atr_ratio` ✅
10. `ret_skew_30d` ✅
11. `price_ma60_deviation` ✅
12. `blowoff_ratio_30d` ✅
13. `feat_vol_price_div_30d` ✅

**缺失的因子** (在 T10_ALPHA_FACTORS 中但不在 t10_selected 中):
- ⚠️ `5_days_reversal` (在 T10_ALPHA_FACTORS 中，但不在 t10_selected 中)
- ⚠️ `downside_beta_ewm_21` (在 T10_ALPHA_FACTORS 中，但不在 t10_selected 中)

---

### 3. 四个第一层模型的输入点

**位置**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 3301-3306)

```python
base_overrides = {
    'elastic_net': list(t10_selected),      # 13个因子
    'catboost': list(t10_selected),         # 13个因子
    'xgboost': list(t10_selected),         # 13个因子
    'lambdarank': list(t10_selected),      # 13个因子
}
```

**验证结果**:
- ✅ **ElasticNet**: 使用 `t10_selected` (13个因子)
- ✅ **CatBoost**: 使用 `t10_selected` (13个因子)
- ✅ **XGBoost**: 使用 `t10_selected` (13个因子)
- ✅ **LambdaRank**: 使用 `t10_selected` (13个因子)

**特征选择方法**: `_get_first_layer_feature_cols_for_model()` (line 6792)
- 该方法从 `first_layer_feature_overrides` 获取每个模型的因子列表
- 如果因子列表为 None，则使用所有可用因子
- 如果因子列表存在，则只使用列表中的因子

---

### 4. Direct Predict 输入点

**位置**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 5356-5364)

**base_features 列表** (13个):
1. `momentum_10d` ✅
2. `ivol_30` ✅
3. `near_52w_high` ✅
4. `rsi_21` ✅
5. `vol_ratio_30d` ✅
6. `trend_r2_60` ✅
7. `liquid_momentum` ✅
8. `obv_momentum_40d` ✅
9. `atr_ratio` ✅
10. `ret_skew_30d` ✅
11. `price_ma60_deviation` ✅
12. `blowoff_ratio_30d` ✅
13. `feat_vol_price_div_30d` ✅

**状态**: ✅ 与 `t10_selected` 一致 (13个因子)

---

### 5. 80/20 OOS 评估输入点

**位置**: `scripts/time_split_80_20_oos_eval.py`

**特征来源**: 
- 使用训练好的模型，特征列表来自模型本身 (`model.feature_names_in_` 或类似属性)
- 通过 `align_test_features_with_model()` 方法自动对齐特征

**状态**: ✅ 自动使用训练时的特征列表，无需手动配置

---

## 🔍 发现的问题

### 问题 1: T10_ALPHA_FACTORS 与 t10_selected 不一致

**T10_ALPHA_FACTORS** 包含 15 个因子，但 **t10_selected** 只有 13 个因子。

**差异**:
- `5_days_reversal` - 在 T10_ALPHA_FACTORS 中，但不在 t10_selected 中
- `downside_beta_ewm_21` - 在 T10_ALPHA_FACTORS 中，但不在 t10_selected 中

**影响**:
- 这两个因子会被计算，但不会被用于训练和预测
- 如果它们应该被使用，需要添加到 t10_selected 中

---

## ✅ 验证总结

### 四个第一层模型的输入点确认

| 模型 | 输入来源 | 因子数量 | 状态 |
|------|---------|---------|------|
| **ElasticNet** | `base_overrides['elastic_net']` = `t10_selected` | 13 | ✅ 正确 |
| **CatBoost** | `base_overrides['catboost']` = `t10_selected` | 13 | ✅ 正确 |
| **XGBoost** | `base_overrides['xgboost']` = `t10_selected` | 13 | ✅ 正确 |
| **LambdaRank** | `base_overrides['lambdarank']` = `t10_selected` | 13 | ✅ 正确 |

### Direct Predict 输入点确认

| 功能 | 输入来源 | 因子数量 | 状态 |
|------|---------|---------|------|
| **Direct Predict** | `base_features` | 13 | ✅ 正确 |

### 80/20 OOS 评估输入点确认

| 功能 | 输入来源 | 状态 |
|------|---------|------|
| **80/20 OOS** | 模型训练时的特征列表 (自动对齐) | ✅ 正确 |

---

## 📋 最终确认的因子列表 (13个)

所有四个第一层模型、Direct Predict 和 80/20 OOS 都使用以下 **13个因子**:

1. `momentum_10d` ✅
2. `ivol_30` ✅
3. `near_52w_high` ✅
4. `rsi_21` ✅
5. `vol_ratio_30d` ✅
6. `trend_r2_60` ✅
7. `liquid_momentum` ✅
8. `obv_momentum_40d` ✅
9. `atr_ratio` ✅
10. `ret_skew_30d` ✅
11. `price_ma60_deviation` ✅
12. `blowoff_ratio_30d` ✅
13. `feat_vol_price_div_30d` ✅

**已删除的因子**:
- ❌ `bollinger_squeeze`
- ❌ `hist_vol_40d`

**未使用的因子** (在 T10_ALPHA_FACTORS 中但不在 t10_selected 中):
- ⚠️ `5_days_reversal`
- ⚠️ `downside_beta_ewm_21`

---

## 🎯 结论

✅ **所有四个第一层模型的输入点都正确！**

- ✅ 四个模型都使用相同的 `t10_selected` 列表 (13个因子)
- ✅ Direct Predict 使用相同的因子列表 (13个因子)
- ✅ 80/20 OOS 自动使用训练时的特征列表
- ✅ 已删除的因子 (`bollinger_squeeze`, `hist_vol_40d`) 不在任何输入点中
- ✅ 新增的因子 (`momentum_10d`) 在所有输入点中

**注意**: `5_days_reversal` 和 `downside_beta_ewm_21` 在 T10_ALPHA_FACTORS 中定义，但不在实际使用的 t10_selected 中。如果需要使用它们，需要添加到 t10_selected 和 base_features 中。

---

**最后更新**: 2025-01-20  
**状态**: ✅ 验证完成 - 所有输入点正确
