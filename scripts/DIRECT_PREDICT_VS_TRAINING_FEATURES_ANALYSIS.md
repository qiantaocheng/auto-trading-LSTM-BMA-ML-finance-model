# Direct Predict vs 训练特征一致性分析报告

## 📊 执行时间
2026-01-22

## 🎯 分析目标
1. 验证Direct Predict使用的特征与训练时使用的特征是否一致
2. 确认所有需要的特征都能被Simple17FactorEngine正确计算
3. 检查特征选择逻辑（`_get_first_layer_feature_cols_for_model`）是否正确

---

## ✅ 分析结果总结

### [结论] ✅ 所有特征一致且可以被正确计算

- **训练特征**: 15 个
- **Direct Predict特征**: 15 个
- **T10_ALPHA_FACTORS**: 15 个
- **所有需要的特征都能被Simple17FactorEngine计算**

---

## 📋 详细分析

### 1. 训练时使用的特征 (t10_selected)

**来源**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 3283-3301)

**特征列表** (15个):
1. `momentum_10d` - 10天短期动量
2. `ivol_30` - 30天隐含波动率
3. `near_52w_high` - 接近52周高点
4. `rsi_21` - 21天RSI
5. `vol_ratio_30d` - 30天成交量比率
6. `trend_r2_60` - 60天趋势R²
7. `liquid_momentum` - 流动性动量
8. `obv_momentum_40d` - 40天OBV动量
9. `atr_ratio` - ATR比率
10. `ret_skew_30d` - 30天收益偏度
11. `price_ma60_deviation` - 价格相对MA60偏离度
12. `blowoff_ratio_30d` - 30天爆发比率
13. `feat_vol_price_div_30d` - 30天量价背离因子
14. `5_days_reversal` - 5天反转因子
15. `downside_beta_ewm_21` - 21天EWMA下行beta（相对QQQ）

**配置位置**: 
- `_base_feature_overrides['elastic_net']` = `t10_selected`
- `_base_feature_overrides['catboost']` = `t10_selected`
- `_base_feature_overrides['xgboost']` = `t10_selected`
- `_base_feature_overrides['lambdarank']` = `t10_selected`

---

### 2. Direct Predict使用的特征 (base_features)

**来源**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 5358-5368)  
**调用位置**: `autotrader/app.py` → `_direct_predict_snapshot()` → `predict_with_snapshot()`

**特征列表** (15个):
1. `momentum_10d`
2. `ivol_30`
3. `near_52w_high`
4. `rsi_21`
5. `vol_ratio_30d`
6. `trend_r2_60`
7. `liquid_momentum`
8. `obv_momentum_40d`
9. `atr_ratio`
10. `ret_skew_30d`
11. `price_ma60_deviation`
12. `blowoff_ratio_30d`
13. `feat_vol_price_div_30d`
14. `5_days_reversal`
15. `downside_beta_ewm_21`

**对比结果**: ✅ **与训练特征完全一致**

---

### 3. T10_ALPHA_FACTORS (Simple17FactorEngine计算的所有因子)

**来源**: `bma_models/simple_25_factor_engine.py` (line 58-78)

**特征列表** (15个):
1. `momentum_10d`
2. `liquid_momentum`
3. `obv_momentum_40d`
4. `ivol_30`
5. `rsi_21`
6. `trend_r2_60`
7. `near_52w_high`
8. `ret_skew_30d`
9. `blowoff_ratio_30d`
10. `atr_ratio`
11. `vol_ratio_30d`
12. `price_ma60_deviation`
13. `5_days_reversal`
14. `downside_beta_ewm_21`
15. `feat_vol_price_div_30d`

**验证结果**:
- ✅ 所有训练特征都在T10_ALPHA_FACTORS中
- ✅ 所有Direct Predict特征都在T10_ALPHA_FACTORS中
- ✅ Simple17FactorEngine可以计算所有需要的特征

---

### 4. 特征一致性对比

#### 4.1 训练特征 vs Direct Predict特征

| 对比项 | 结果 |
|--------|------|
| 特征数量 | ✅ 一致 (15个) |
| 特征列表 | ✅ 完全一致 |
| 特征顺序 | ⚠️ 顺序不同（不影响功能） |

**结论**: ✅ **完全一致**

#### 4.2 训练特征是否都能被计算

| 检查项 | 结果 |
|--------|------|
| 所有特征在T10_ALPHA_FACTORS中 | ✅ 是 |
| Simple17FactorEngine能计算 | ✅ 是 |

**结论**: ✅ **所有训练特征都能被Simple17FactorEngine计算**

#### 4.3 Direct Predict特征是否都能被计算

| 检查项 | 结果 |
|--------|------|
| 所有特征在T10_ALPHA_FACTORS中 | ✅ 是 |
| Simple17FactorEngine能计算 | ✅ 是 |

**结论**: ✅ **所有Direct Predict特征都能被Simple17FactorEngine计算**

---

### 5. 特征选择逻辑验证

**方法**: `_get_first_layer_feature_cols_for_model()`

**测试结果**:

#### 5.1 ElasticNet
- **选择的特征数**: 15个
- **特征列表**: 与`t10_selected`一致
- **状态**: ✅ 与训练特征一致

#### 5.2 XGBoost
- **选择的特征数**: 15个
- **特征列表**: 与`t10_selected`一致
- **状态**: ✅ 与训练特征一致

#### 5.3 CatBoost
- **选择的特征数**: 15个
- **特征列表**: 与`t10_selected`一致
- **状态**: ✅ 与训练特征一致

#### 5.4 LambdaRank
- **选择的特征数**: 15个
- **特征列表**: 与`t10_selected`一致
- **状态**: ✅ 与训练特征一致

**结论**: ✅ **所有模型的特征选择逻辑都正确，与训练时一致**

---

### 6. Simple17FactorEngine计算方法验证

**方法**: `compute_all_17_factors()`

**计算流程**:
1. ✅ Momentum Factors (`_compute_momentum_factors`)
   - 计算 `momentum_10d`, `liquid_momentum`
2. ✅ Mean Reversion Factors (`_compute_mean_reversion_factors`)
   - 计算 `near_52w_high`, `rsi_21`, `price_ma60_deviation`, `5_days_reversal`
3. ✅ Volume Factors (`_compute_volume_factors`)
   - 计算 `obv_momentum_40d`, `vol_ratio_30d`, `feat_vol_price_div_30d`
4. ✅ Volatility Factors (`_compute_volatility_factors`)
   - 计算 `atr_ratio`, `ivol_30`
5. ✅ Downside Beta (`_compute_downside_beta_ewm_21`)
   - 计算 `downside_beta_ewm_21`
6. ✅ High-Alpha Factors (`_compute_new_alpha_factors`)
   - 计算 `trend_r2_60`, `ret_skew_30d`, `blowoff_ratio_30d`

**验证结果**: ✅ **所有训练特征都有对应的计算方法**

---

## 🔍 Direct Predict特征计算流程

### 流程概述

1. **数据获取** (`app.py` line 1662-1667)
   - 使用`Simple17FactorEngine.fetch_market_data()`获取市场数据
   - 获取280+天的历史数据（用于因子计算）

2. **因子计算** (`app.py` line 1728)
   - 调用`engine.compute_all_17_factors(market_data, mode='predict')`
   - 计算所有T10_ALPHA_FACTORS（15个因子）

3. **特征选择** (`量化模型_bma_ultra_enhanced.py` line 5504)
   - 在`predict_with_snapshot()`中调用`_get_first_layer_feature_cols_for_model()`
   - 根据模型类型选择对应的特征子集（15个）

4. **预测** (`量化模型_bma_ultra_enhanced.py` line 5527)
   - 使用选定的特征进行预测

### 关键代码位置

- **Direct Predict入口**: `autotrader/app.py` → `_direct_predict_snapshot()` (line 1527)
- **因子计算**: `autotrader/app.py` → `engine.compute_all_17_factors()` (line 1728)
- **特征选择**: `bma_models/量化模型_bma_ultra_enhanced.py` → `_get_first_layer_feature_cols_for_model()` (line 6796)
- **预测**: `bma_models/量化模型_bma_ultra_enhanced.py` → `predict_with_snapshot()` (line 5350)

---

## ✅ 最终结论

### 1. 特征一致性 ✅
- **训练特征**和**Direct Predict特征**完全一致（15个因子）
- 所有特征都在`T10_ALPHA_FACTORS`中定义
- 特征顺序不同但不影响功能（使用特征名称匹配）

### 2. 特征计算能力 ✅
- 所有需要的特征都能被`Simple17FactorEngine`计算
- `compute_all_17_factors()`方法包含所有因子的计算方法
- 每个因子都有对应的计算方法（如`_compute_momentum_factors`, `_compute_volume_factors`等）

### 3. 特征选择逻辑 ✅
- `_get_first_layer_feature_cols_for_model()`方法正确工作
- 所有模型（ElasticNet, XGBoost, CatBoost, LambdaRank）都选择相同的15个特征
- 特征选择与训练时完全一致

### 4. 数据流一致性 ✅
- Direct Predict使用与训练相同的特征集
- 特征计算使用相同的引擎（Simple17FactorEngine）
- 特征选择使用相同的逻辑（`_get_first_layer_feature_cols_for_model`）

---

## 📝 建议

### ✅ 当前状态良好
- 所有特征一致且可计算
- 无需修改

### 🔍 监控建议
1. **定期验证**: 当添加新因子时，确保同时更新训练和Direct Predict的特征列表
2. **测试验证**: 在修改特征列表后，运行此分析脚本验证一致性
3. **文档维护**: 保持特征列表的文档同步更新

---

## 📊 特征列表对比表

| # | 特征名称 | 训练 | Direct Predict | T10_ALPHA_FACTORS | 可计算 |
|---|---------|------|----------------|-------------------|--------|
| 1 | momentum_10d | ✅ | ✅ | ✅ | ✅ |
| 2 | ivol_30 | ✅ | ✅ | ✅ | ✅ |
| 3 | near_52w_high | ✅ | ✅ | ✅ | ✅ |
| 4 | rsi_21 | ✅ | ✅ | ✅ | ✅ |
| 5 | vol_ratio_30d | ✅ | ✅ | ✅ | ✅ |
| 6 | trend_r2_60 | ✅ | ✅ | ✅ | ✅ |
| 7 | liquid_momentum | ✅ | ✅ | ✅ | ✅ |
| 8 | obv_momentum_40d | ✅ | ✅ | ✅ | ✅ |
| 9 | atr_ratio | ✅ | ✅ | ✅ | ✅ |
| 10 | ret_skew_30d | ✅ | ✅ | ✅ | ✅ |
| 11 | price_ma60_deviation | ✅ | ✅ | ✅ | ✅ |
| 12 | blowoff_ratio_30d | ✅ | ✅ | ✅ | ✅ |
| 13 | feat_vol_price_div_30d | ✅ | ✅ | ✅ | ✅ |
| 14 | 5_days_reversal | ✅ | ✅ | ✅ | ✅ |
| 15 | downside_beta_ewm_21 | ✅ | ✅ | ✅ | ✅ |

**总计**: 15/15 ✅

---

**生成时间**: 2026-01-22  
**分析脚本**: `scripts/analyze_direct_predict_vs_training_features.py`  
**状态**: ✅ **所有检查通过**
