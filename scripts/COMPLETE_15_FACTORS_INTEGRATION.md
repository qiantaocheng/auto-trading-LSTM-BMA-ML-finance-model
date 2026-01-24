# 15个因子完整集成总结

## ✅ 完成状态

**所有15个因子已成功添加到训练和预测流程！**

---

## 📊 因子列表（15个）

### 完整因子列表（T10_ALPHA_FACTORS = t10_selected）

1. `momentum_10d` - 10天短期动量
2. `ivol_30` - 特质波动率（30天，需要SPY/QQQ）
3. `near_52w_high` - 接近52周高点
4. `rsi_21` - RSI指标（21天）
5. `vol_ratio_30d` - 成交量比率（30天）
6. `trend_r2_60` - 趋势R²（60天）
7. `liquid_momentum` - 流动性动量
8. `obv_momentum_40d` - OBV动量（40天）
9. `atr_ratio` - ATR比率
10. `ret_skew_30d` - 收益偏度（30天）
11. `price_ma60_deviation` - 价格MA60偏离度
12. `blowoff_ratio_30d` - 爆量比率（30天）
13. `feat_vol_price_div_30d` - 量价背离因子（30天）
14. `5_days_reversal` ✅ **新增** - 5天反转因子
15. `downside_beta_ewm_21` ✅ **新增** - 下行Beta（EWM 21天，需要QQQ）

---

## ✅ 训练流程

### 四个第一层模型
所有模型使用 **相同的** 15个因子：

- ✅ **ElasticNet**: t10_selected (15个因子)
- ✅ **CatBoost**: t10_selected (15个因子)
- ✅ **XGBoost**: t10_selected (15个因子)
- ✅ **LambdaRank**: t10_selected (15个因子)

**配置位置**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 3301-3306)

---

## ✅ 预测流程

### Direct Predict
- ✅ **base_features**: 15个因子（与 t10_selected 完全一致）

**配置位置**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 5356-5364)

### 80/20 OOS 评估
- ✅ **自动特征对齐**: 使用训练时的特征列表
- ✅ **机制**: `align_test_features_with_model()` + `_get_first_layer_feature_cols_for_model()`

---

## 🔧 SPY/QQQ 数据自动获取

### `ivol_30` 因子
- ✅ **自动下载 SPY**: 如果数据中没有 SPY，自动从 Polygon 下载
- ✅ **QQQ Fallback**: 如果 SPY 不可用，使用 QQQ 作为备选
- ✅ **实现位置**: `bma_models/simple_25_factor_engine.py` `_compute_ivol_30()`

### `downside_beta_ewm_21` 因子
- ✅ **自动下载 QQQ**: 自动从 Polygon 下载 QQQ 数据
- ✅ **缓存机制**: 使用 `_benchmark_cache` 避免重复下载
- ✅ **实现位置**: `bma_models/simple_25_factor_engine.py` `_compute_downside_beta_ewm_21()`

### `_get_benchmark_returns_by_date` 方法
- ✅ **Polygon Client**: 优先使用 `polygon_client.get_historical_bars()`
- ✅ **REST API Fallback**: 如果 client 不可用，使用 REST API
- ✅ **缓存**: 避免重复下载相同数据

---

## ✅ 因子计算验证

### 所有15个因子都有对应的计算方法

| 因子 | 计算方法 | 状态 |
|------|---------|------|
| `momentum_10d` | `_compute_momentum_factors` | ✅ |
| `liquid_momentum` | `_compute_momentum_factors` | ✅ |
| `obv_momentum_40d` | `_compute_volume_factors` | ✅ |
| `ivol_30` | `_compute_ivol_30` | ✅ (自动下载SPY) |
| `rsi_21` | `_compute_mean_reversion_factors` | ✅ |
| `trend_r2_60` | `_compute_trend_r2_60` | ✅ |
| `near_52w_high` | `_compute_new_alpha_factors` | ✅ |
| `ret_skew_30d` | `_compute_ret_skew_30d` | ✅ |
| `blowoff_ratio_30d` | `_compute_blowoff_and_volatility` | ✅ |
| `atr_ratio` | `_compute_volatility_factors` | ✅ |
| `vol_ratio_30d` | `_compute_volume_factors` | ✅ |
| `price_ma60_deviation` | `_compute_mean_reversion_factors` | ✅ |
| `5_days_reversal` | `_compute_momentum_factors` | ✅ |
| `downside_beta_ewm_21` | `_compute_downside_beta_ewm_21` | ✅ (自动下载QQQ) |
| `feat_vol_price_div_30d` | `_compute_vol_price_div_30d` | ✅ |

---

## 📋 代码修改清单

### 1. `bma_models/量化模型_bma_ultra_enhanced.py`

#### Line 3283-3299: 更新 t10_selected
```python
t10_selected = [
    "momentum_10d",
    "ivol_30",
    "near_52w_high",
    "rsi_21",
    "vol_ratio_30d",
    "trend_r2_60",
    "liquid_momentum",
    "obv_momentum_40d",
    "atr_ratio",
    "ret_skew_30d",
    "price_ma60_deviation",
    "blowoff_ratio_30d",
    "feat_vol_price_div_30d",
    "5_days_reversal",  # ADDED
    "downside_beta_ewm_21",  # ADDED
]
```

#### Line 5356-5364: 更新 base_features
```python
base_features = [
    'momentum_10d',
    'ivol_30', 'near_52w_high', 'rsi_21', 'vol_ratio_30d',
    'trend_r2_60', 'liquid_momentum', 'obv_momentum_40d', 'atr_ratio',
    'ret_skew_30d', 'price_ma60_deviation', 'blowoff_ratio_30d',
    'feat_vol_price_div_30d',
    '5_days_reversal',  # ADDED
    'downside_beta_ewm_21',  # ADDED
]
```

### 2. `bma_models/simple_25_factor_engine.py`

#### `_compute_ivol_30()`: 改进 SPY 数据获取
- 如果数据中没有 SPY，自动从 Polygon 下载
- 如果 SPY 下载失败，使用 QQQ 作为备选

#### `_compute_downside_beta_ewm_21()`: 改进 QQQ 数据获取
- 自动从 Polygon 下载 QQQ 数据（如果缓存中没有）
- 改进错误处理和日志记录

#### `compute_all_17_factors()`: 确保所有因子被调用
- `_compute_vol_price_div_30d()` 已添加显式调用
- `_compute_downside_beta_ewm_21()` 已包含在流程中
- `5_days_reversal` 在 `_compute_momentum_factors()` 中计算

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

### 因子计算
- ✅ 所有15个因子都有对应的计算方法
- ✅ 所有因子都使用 `shift(1)` 用于开盘前预测
- ✅ 所有因子都正确处理 MultiIndex 数据

---

## 🚀 下一步操作

### 1. 更新数据文件（推荐）
运行以下命令重新计算所有15个因子并更新数据文件：

```bash
python scripts/verify_and_update_all_factors.py \
    --yes \
    --input-file data/factor_exports/polygon_factors_all_filtered_clean_recalculated.parquet \
    --output-file data/factor_exports/polygon_factors_all_filtered_clean_15factors.parquet \
    --lookback-days 120
```

这将：
- 自动下载 SPY 数据（如果数据中没有）
- 重新计算所有15个因子
- 确保所有因子都正确计算

### 2. 验证训练
使用更新后的数据文件训练模型，确认所有15个因子都被使用。

### 3. 验证预测
确认 Direct Predict 和 80/20 OOS 评估都正确使用15个因子。

---

## 📝 重要说明

### SPY/QQQ 数据获取
- **自动机制**: 因子计算时会自动尝试从 Polygon 下载所需数据
- **缓存**: 使用 `_benchmark_cache` 避免重复下载
- **Fallback**: `ivol_30` 在 SPY 不可用时使用 QQQ
- **错误处理**: 如果下载失败，返回零值但记录警告

### 因子计算顺序
1. 动量因子（包括 `momentum_10d`, `5_days_reversal`）
2. 均值回归因子（`rsi_21`, `price_ma60_deviation`）
3. 成交量因子（`obv_momentum_40d`, `vol_ratio_30d`）
4. 波动率因子（`atr_ratio`）
5. 特质波动率（`ivol_30` - 需要 SPY）
6. 趋势因子（`trend_r2_60`）
7. 新Alpha因子（`near_52w_high`）
8. 收益偏度（`ret_skew_30d`）
9. 爆量比率（`blowoff_ratio_30d`）
10. 下行Beta（`downside_beta_ewm_21` - 需要 QQQ）
11. 量价背离（`feat_vol_price_div_30d`）

---

## 🎯 最终状态

**✅ 所有15个因子已成功集成！**

- ✅ 训练流程: 所有四个第一层模型使用15个因子
- ✅ 预测流程: Direct Predict 和 80/20 OOS 使用15个因子
- ✅ 因子计算: 所有15个因子都能正确计算
- ✅ SPY/QQQ 数据: 自动获取机制已实现

---

**最后更新**: 2025-01-20  
**状态**: ✅ **完成** - 所有15个因子已添加到训练和预测流程，SPY/QQQ 数据自动获取机制已实现
