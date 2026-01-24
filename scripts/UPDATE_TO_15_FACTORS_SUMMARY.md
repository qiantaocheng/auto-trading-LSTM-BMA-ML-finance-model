# 更新到15个因子 - 完整总结

## ✅ 完成的更新

### 1. 添加两个新因子

**新增因子**:
- `5_days_reversal` - 5天反转因子
- `downside_beta_ewm_21` - 下行Beta（EWM 21天）

### 2. 更新位置

#### 2.1 t10_selected (训练用)
**文件**: `bma_models/量化模型_bma_ultra_enhanced.py`  
**位置**: Line 3283-3299  
**更新**: 添加 `5_days_reversal` 和 `downside_beta_ewm_21`

#### 2.2 base_features (Direct Predict用)
**文件**: `bma_models/量化模型_bma_ultra_enhanced.py`  
**位置**: Line 5356-5364  
**更新**: 添加 `5_days_reversal` 和 `downside_beta_ewm_21`

#### 2.3 SPY/QQQ 数据自动获取
**文件**: `bma_models/simple_25_factor_engine.py`  
**更新**:
- `_compute_ivol_30()`: 自动下载 SPY（如果数据中没有）
- `_compute_downside_beta_ewm_21()`: 自动下载 QQQ

---

## 📊 最终因子列表（15个）

1. `momentum_10d`
2. `ivol_30` (需要 SPY/QQQ)
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
14. `5_days_reversal` ✅ **新增**
15. `downside_beta_ewm_21` ✅ **新增** (需要 QQQ)

---

## ✅ 验证结果

### 因子一致性
- ✅ T10_ALPHA_FACTORS: 15 个因子
- ✅ t10_selected: 15 个因子（完全匹配）
- ✅ base_features: 15 个因子（完全匹配）
- ✅ 所有四个第一层模型: 15 个因子（完全匹配）

### SPY/QQQ 数据获取
- ✅ `ivol_30`: 自动从 Polygon 下载 SPY
- ✅ `downside_beta_ewm_21`: 自动从 Polygon 下载 QQQ
- ✅ 缓存机制: 避免重复下载
- ✅ Fallback 机制: SPY 失败时使用 QQQ

---

## 🚀 下一步

运行以下命令更新数据文件：

```bash
python scripts/verify_and_update_all_factors.py \
    --yes \
    --input-file data/factor_exports/polygon_factors_all_filtered_clean_recalculated.parquet \
    --output-file data/factor_exports/polygon_factors_all_filtered_clean_15factors.parquet \
    --lookback-days 120
```

---

**状态**: ✅ **完成** - 所有15个因子已添加到训练和预测流程
