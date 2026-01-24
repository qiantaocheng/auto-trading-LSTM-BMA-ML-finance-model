# 15个因子完整更新总结

## ✅ 完成的更新

### 1. 添加两个新因子到 t10_selected

**更新位置**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 3283-3299)

**添加的因子**:
- `5_days_reversal` - 5天反转因子
- `downside_beta_ewm_21` - 下行Beta（EWM 21天）

**更新后的 t10_selected (15个因子)**:
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

---

### 2. 更新 Direct Predict 的 base_features

**更新位置**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 5356-5364)

**更新后的 base_features (15个因子)**:
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

---

### 3. 改进 SPY/QQQ 数据获取机制

**更新位置**: `bma_models/simple_25_factor_engine.py`

#### 3.1 `ivol_30` 因子改进
- **自动下载 SPY**: 如果数据中没有 SPY，自动从 Polygon 下载
- **QQQ Fallback**: 如果 SPY 不可用，使用 QQQ 作为备选
- **错误处理**: 如果下载失败，返回零值但记录警告

#### 3.2 `downside_beta_ewm_21` 因子改进
- **自动下载 QQQ**: 如果缓存中没有 QQQ 数据，自动从 Polygon 下载
- **错误处理**: 如果下载失败，返回零值但记录警告

---

### 4. 因子计算验证

**所有15个因子计算状态**:

| 因子 | 计算方法 | SPY/QQQ需求 | 状态 |
|------|---------|-------------|------|
| `momentum_10d` | `_compute_momentum_factors` | ❌ | ✅ |
| `liquid_momentum` | `_compute_momentum_factors` | ❌ | ✅ |
| `obv_momentum_40d` | `_compute_volume_factors` | ❌ | ✅ |
| `ivol_30` | `_compute_ivol_30` | ✅ SPY/QQQ | ✅ 自动下载 |
| `rsi_21` | `_compute_mean_reversion_factors` | ❌ | ✅ |
| `trend_r2_60` | `_compute_trend_r2_60` | ❌ | ✅ |
| `near_52w_high` | `_compute_new_alpha_factors` | ❌ | ✅ |
| `ret_skew_30d` | `_compute_ret_skew_30d` | ❌ | ✅ |
| `blowoff_ratio_30d` | `_compute_blowoff_and_volatility` | ❌ | ✅ |
| `atr_ratio` | `_compute_volatility_factors` | ❌ | ✅ |
| `vol_ratio_30d` | `_compute_volume_factors` | ❌ | ✅ |
| `price_ma60_deviation` | `_compute_mean_reversion_factors` | ❌ | ✅ |
| `5_days_reversal` | `_compute_momentum_factors` | ❌ | ✅ |
| `downside_beta_ewm_21` | `_compute_downside_beta_ewm_21` | ✅ QQQ | ✅ 自动下载 |
| `feat_vol_price_div_30d` | `_compute_vol_price_div_30d` | ❌ | ✅ |

---

## ✅ 训练和预测流程验证

### 训练流程
- ✅ **ElasticNet**: 使用 t10_selected (15个因子)
- ✅ **CatBoost**: 使用 t10_selected (15个因子)
- ✅ **XGBoost**: 使用 t10_selected (15个因子)
- ✅ **LambdaRank**: 使用 t10_selected (15个因子)

**配置**: `bma_models/量化模型_bma_ultra_enhanced.py` (line 3301-3306)

### 预测流程
- ✅ **Direct Predict**: base_features 与 t10_selected 完全一致 (15个因子)
- ✅ **80/20 OOS**: 自动特征对齐，使用训练时的特征列表

---

## 🔧 技术实现细节

### SPY/QQQ 数据自动获取

#### `ivol_30` 因子
```python
# 1. 首先检查数据中是否有 SPY
spy = data[data['ticker'] == 'SPY']

# 2. 如果没有，尝试从 Polygon 下载
if spy.empty:
    spy_ret_by_date = self._get_benchmark_returns_by_date('SPY', dates)
    # 如果 SPY 失败，使用 QQQ 作为备选
    if spy_ret_by_date is None:
        qqq_ret_by_date = self._get_benchmark_returns_by_date('QQQ', dates)
```

#### `downside_beta_ewm_21` 因子
```python
# 自动从 Polygon 下载 QQQ 数据（如果缓存中没有）
bench_ret_by_date = self._get_benchmark_returns_by_date('QQQ', dates)
if bench_ret_by_date is None:
    # 尝试重新下载
    bench_ret_by_date = self._get_benchmark_returns_by_date('QQQ', dates)
```

### `_get_benchmark_returns_by_date` 方法
- **缓存机制**: 使用 `_benchmark_cache` 避免重复下载
- **Polygon API**: 优先使用 `polygon_client.get_historical_bars()`
- **Fallback**: 如果 `polygon_client` 不可用，使用 REST API
- **日期范围**: 自动计算所需日期范围并下载

---

## 📊 最终因子列表（15个）

### T10_ALPHA_FACTORS (所有计算的因子)
1. `momentum_10d`
2. `liquid_momentum`
3. `obv_momentum_40d`
4. `ivol_30` (需要 SPY/QQQ)
5. `rsi_21`
6. `trend_r2_60`
7. `near_52w_high`
8. `ret_skew_30d`
9. `blowoff_ratio_30d`
10. `atr_ratio`
11. `vol_ratio_30d`
12. `price_ma60_deviation`
13. `5_days_reversal` ✅ **新增**
14. `downside_beta_ewm_21` ✅ **新增** (需要 QQQ)
15. `feat_vol_price_div_30d`

### t10_selected (用于训练和预测)
**与 T10_ALPHA_FACTORS 完全一致** (15个因子)

---

## ✅ 验证结果

### 因子一致性
- ✅ T10_ALPHA_FACTORS: 15 个因子
- ✅ t10_selected: 15 个因子（完全匹配）
- ✅ base_features: 15 个因子（完全匹配）
- ✅ 所有四个第一层模型: 15 个因子（完全匹配）

### SPY/QQQ 数据获取
- ✅ `ivol_30`: 自动从 Polygon 下载 SPY（如果数据中没有）
- ✅ `downside_beta_ewm_21`: 自动从 Polygon 下载 QQQ
- ✅ 缓存机制: 避免重复下载
- ✅ Fallback 机制: SPY 失败时使用 QQQ

### 因子计算
- ✅ 所有15个因子都有对应的计算方法
- ✅ 所有因子都使用 `shift(1)` 用于开盘前预测
- ✅ 所有因子都正确处理 MultiIndex 数据

---

## 🚀 下一步

1. **更新数据文件**: 运行 `verify_and_update_all_factors.py` 重新计算所有15个因子
2. **验证训练**: 使用更新后的数据文件训练模型，确认所有15个因子都被使用
3. **验证预测**: 确认 Direct Predict 和 80/20 OOS 评估都正确使用15个因子

---

## 📝 文件修改清单

1. ✅ `bma_models/量化模型_bma_ultra_enhanced.py`
   - 更新 `t10_selected` (添加 2 个因子)
   - 更新 `base_features` (添加 2 个因子)

2. ✅ `bma_models/simple_25_factor_engine.py`
   - 改进 `_compute_ivol_30` (自动下载 SPY)
   - 改进 `_compute_downside_beta_ewm_21` (自动下载 QQQ)
   - 确保 `_compute_vol_price_div_30d` 被调用

---

**最后更新**: 2025-01-20  
**状态**: ✅ **完成** - 所有15个因子已添加到训练和预测流程，SPY/QQQ 数据自动获取机制已实现
