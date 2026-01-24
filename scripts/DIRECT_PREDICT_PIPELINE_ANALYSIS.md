# Direct Prediction Pipeline 分析报告

## 分析日期
2026-01-24

## 分析目标
全面检查 direct prediction pipeline 的潜在问题，确保：
1. 因子一致性
2. 模型加载正确性
3. 特征对齐
4. 预测结果处理
5. 列顺序一致性

## 发现的问题

### ✅ 已修复的问题

#### 1. **列顺序不一致问题** (已修复)
- **问题描述**: `pred_lambdarank` 在 `ridge_input` 按 `base_cols` 排序之后添加，导致列顺序不匹配训练时
- **影响**: Meta Stacker 可能因为列顺序不一致而产生错误预测
- **修复位置**: `bma_models/量化模型_bma_ultra_enhanced.py` 第10141-10148行
- **修复方案**: 在添加 `pred_lambdarank` 后，重新按 `base_cols` 排序

```python
# 🔥 Ensure pred_lambdarank is in ridge_input (for MetaRankerStacker)
if 'pred_lambdarank' not in ridge_input.columns and 'pred_lambdarank' in first_layer_preds.columns:
    ridge_input['pred_lambdarank'] = first_layer_preds['pred_lambdarank'].reindex(ridge_input.index)
    logger.info("[SNAPSHOT] Added pred_lambdarank to ridge_input")

# 🔧 FIX: Re-order columns to match base_cols after adding pred_lambdarank
if 'pred_lambdarank' in ridge_base_cols and 'pred_lambdarank' in ridge_input.columns:
    available_base_cols = [col for col in ridge_base_cols if col in ridge_input.columns]
    ridge_input = ridge_input[available_base_cols].copy()
    logger.info(f"[SNAPSHOT] Re-ordered ridge_input columns to match base_cols: {list(ridge_input.columns)}")
```

### ✅ 已验证正常的部分

#### 1. **因子一致性** ✅
- **状态**: PASS
- **验证**: Training script 和 Direct Prediction 使用相同的 14 个因子
- **因子列表**: `liquid_momentum`, `momentum_10d`, `momentum_60d`, `obv_divergence`, `obv_momentum_60d`, `ivol_20`, `hist_vol_40d`, `atr_ratio`, `rsi_21`, `trend_r2_60`, `near_52w_high`, `vol_ratio_20d`, `price_ma60_deviation`, `5_days_reversal`

#### 2. **Meta Stacker base_cols** ✅
- **状态**: PASS
- **配置**: `['pred_catboost', 'pred_xgb', 'pred_lambdarank', 'pred_elastic']`
- **验证**: 所有必需的列都在配置中，包括 `pred_catboost`

#### 3. **CatBoost 模型加载** ✅
- **状态**: PASS
- **验证**: 
  - CatBoost 模型加载代码存在
  - 检查 `CatBoostRegressor is not None`
  - `pred_catboost` 正确添加到 `first_layer_preds`

#### 4. **特征对齐** ✅
- **状态**: PASS
- **验证**:
  - `fill_missing_features_with_median` 函数存在
  - 使用横截面中位数填充缺失特征（而不是 0.0）
  - `feature_names_by_model` 用于每个模型的特征选择

#### 5. **预测结果处理** ✅
- **状态**: PASS
- **验证**:
  - Raw 和 smoothed scores 都正确处理
  - 所有 base model scores (`score_lambdarank`, `score_catboost`, `score_elastic`, `score_xgb`) 都被提取

## Pipeline 流程验证

### 1. 数据获取和因子计算
```
✅ Auto-fetch from Polygon API
✅ Compute factors using Simple17FactorEngine
✅ Use T10_ALPHA_FACTORS (14 factors)
✅ Filter to prediction period AFTER factor calculation
```

### 2. 第一层模型预测
```
✅ ElasticNet: 加载模型 → 选择特征 → 预测 → 添加到 first_layer_preds
✅ XGBoost: 加载模型 → 选择特征 → 预测 → 添加到 first_layer_preds
✅ CatBoost: 加载模型 → 选择特征 → 预测 → 添加到 first_layer_preds
✅ LambdaRank: 加载模型 → 选择特征 → 预测 → 转换为百分位 → 添加到 first_layer_preds
```

### 3. Meta Stacker 输入准备
```
✅ 从 first_layer_preds 创建 ridge_input
✅ 移除 pred_lightgbm_ranker (向后兼容)
✅ 填充缺失的 base_cols (使用横截面中位数)
✅ 按 base_cols 排序
✅ 确保 pred_lambdarank 存在 (如果缺失则添加)
✅ 重新排序以匹配 base_cols (修复后)
```

### 4. Meta Stacker 预测
```
✅ 加载 MetaRankerStacker 模型
✅ 验证模型状态 (fitted_, has_model)
✅ 预测并生成 scores
✅ 验证预测结果 (检查唯一值数量)
```

### 5. Rank-Aware Blending (可选)
```
✅ 如果 LambdaRank 预测可用，执行 Rank-Aware Blending
✅ 否则使用 Meta Stacker 预测
```

### 6. 结果输出
```
✅ 生成 final_df 包含 blended_score
✅ 提取所有 base model scores
✅ 生成 Excel 报告
```

## 潜在风险点

### 1. **LambdaRank 预测失败**
- **风险**: 如果 LambdaRank 预测失败，`pred_lambdarank` 可能缺失
- **当前处理**: 代码会尝试添加 `pred_lambdarank`，但如果 `first_layer_preds` 中也没有，则可能缺失
- **建议**: 确保 LambdaRank 预测失败时，至少使用默认值或报错

### 2. **特征缺失处理**
- **当前处理**: 使用横截面中位数填充
- **风险**: 如果所有特征都缺失，可能使用 0.0 填充
- **建议**: 确保至少有一些特征可用

### 3. **日期过滤时机**
- **当前处理**: 在因子计算后过滤到最近 N 天
- **风险**: 如果 `prediction_days` 设置不当，可能影响因子计算所需的历史数据
- **建议**: 确保 `MIN_LOOKBACK_DAYS = 280` 足够

## 建议改进

### 1. **增强错误处理**
- 如果 LambdaRank 预测失败，应该明确报错或使用默认值
- 如果 Meta Stacker 输入列缺失，应该明确报错

### 2. **增强日志**
- 添加更详细的日志记录每个步骤的状态
- 记录预测结果的统计信息（唯一值数量、范围等）

### 3. **验证检查**
- 在预测前验证所有必需的列都存在
- 验证列顺序与训练时一致
- 验证预测结果的合理性（不是所有值都相同）

## 总结

### ✅ 总体状态: GOOD
- 因子一致性: ✅ PASS
- 模型加载: ✅ PASS
- 特征对齐: ✅ PASS
- 列顺序: ✅ FIXED
- 预测结果处理: ✅ PASS

### 修复的问题
1. ✅ 列顺序不一致问题（pred_lambdarank 添加后重新排序）

### 建议
1. 增强错误处理（LambdaRank 失败情况）
2. 增强日志记录
3. 添加预测前验证检查

## 验证脚本
运行 `scripts/analyze_direct_predict_pipeline.py` 可以重新验证这些问题。
