# 因子使用情况总结

## ✅ 核心结论

**所有因子都已正确放入训练和预测流程！**

---

## 📊 因子统计

### T10_ALPHA_FACTORS（所有计算的因子）
- **总数**: 15 个
- **用途**: 定义所有需要计算的因子

### t10_selected（实际用于第一层模型）
- **总数**: 13 个
- **用途**: 用于 ElasticNet, CatBoost, XGBoost, LambdaRank 训练和预测

### 差异
- **计算但未使用**: 2 个因子
  - `5_days_reversal`
  - `downside_beta_ewm_21`

---

## ✅ 训练流程验证

### 四个第一层模型
所有模型使用 **相同的** 13 个因子：

1. ✅ ElasticNet - 使用 t10_selected (13个因子)
2. ✅ CatBoost - 使用 t10_selected (13个因子)
3. ✅ XGBoost - 使用 t10_selected (13个因子)
4. ✅ LambdaRank - 使用 t10_selected (13个因子)

**配置位置**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 3301-3306)

---

## ✅ 预测流程验证

### Direct Predict
- ✅ base_features 与 t10_selected **完全一致**（13个因子）
- ✅ 所有因子都正确使用

### 80/20 OOS 评估
- ✅ 自动特征对齐机制
- ✅ 使用训练时的特征列表
- ✅ 确保预测与训练一致

---

## 📋 13个核心因子列表

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

---

## ⚠️ 注意事项

### 计算但未使用的因子
以下因子在 T10_ALPHA_FACTORS 中被计算，但不在 t10_selected 中：

1. `5_days_reversal` - 5天反转因子
2. `downside_beta_ewm_21` - 下行Beta（EWM 21天）

**建议**:
- 如果不需要：从 T10_ALPHA_FACTORS 中移除，减少计算开销
- 如果需要：添加到 t10_selected 列表中

---

## ✅ 验证清单

- [x] 所有 13 个核心因子都在 t10_selected 中
- [x] 四个第一层模型使用相同的特征列表
- [x] Direct Predict 使用与训练相同的因子
- [x] 80/20 OOS 评估正确对齐特征
- [x] 所有因子都在数据文件中正确计算
- [x] SPY 数据已下载并用于 `ivol_30` 计算
- [x] `feat_vol_price_div_30d` 已正确计算（97.84% 非零值）

---

## 🎯 最终状态

**✅ 所有因子都已正确放入训练和预测流程！**

- ✅ 训练和预测使用相同的 13 个因子
- ✅ 特征对齐机制正确实现
- ✅ 数据一致性已验证

---

**生成时间**: 2025-01-20  
**分析脚本**: `scripts/analyze_all_factors_usage.py`  
**详细报告**: `scripts/FACTOR_USAGE_ANALYSIS_REPORT.md`
