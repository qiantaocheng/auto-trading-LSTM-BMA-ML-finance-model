# Prediction Data Lookback Analysis
## 预测数据获取的历史数据窗口分析

## 问题分析

当使用 `_direct_predict_snapshot` 获取多天预测数据（默认3天）时，某些因子需要超过3天的历史数据来计算。本文档分析数据获取机制如何确保有足够的历史数据。

## 因子所需的历史数据窗口

### 最长窗口因子

| 因子名称 | 所需窗口 | 说明 |
|---------|---------|------|
| `near_52w_high` | **252天** | 52周最高价（252个交易日） |
| `ma200` | **200天** | 200日移动平均线 |
| `trend_r2_60` | **60天** | 60日趋势R² |
| `ma60` | **60天** | 60日移动平均线 |
| `ma30` | **30天** | 30日移动平均线 |
| `hist_vol_40d` | **40天** | 40日历史波动率 |
| `rsi_21` | **21天** | 21日RSI |
| `downside_beta_ewm_21` | **21天** | 21日EWMA下行Beta |

### 关键发现

**最长窗口因子**: `near_52w_high` 需要 **252个交易日** 的历史数据

**日历天数转换**: 
- 252个交易日 ≈ 280-300个日历天（考虑周末和节假日）
- 系统使用 **MIN_REQUIRED_LOOKBACK_DAYS = 280天** 作为最小要求

## 数据获取机制分析

### 1. `predict_with_snapshot` 中的 Lookback 计算

**代码位置**: `bma_models/量化模型_bma_ultra_enhanced.py:9571-9590`

```python
# Calculate required lookback days
# Maximum rolling window: 252 days (near_52w_high)
# Add buffer for weekends/holidays: 252 trading days ≈ 280-300 calendar days
MIN_REQUIRED_LOOKBACK_DAYS = 280  # 252 trading days + buffer
lookback_days = max(prediction_days + 50, MIN_REQUIRED_LOOKBACK_DAYS)

# Determine date range
if as_of_date is None:
    as_of_date = pd.Timestamp.today()
end_date = pd.to_datetime(as_of_date).strftime('%Y-%m-%d')
start_date = (pd.to_datetime(as_of_date) - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
```

**计算逻辑**:
- **最小要求**: 280天（确保252个交易日）
- **动态计算**: `max(prediction_days + 50, 280)`
- **实际效果**: 
  - 预测3天 → lookback = max(3+50, 280) = **280天**
  - 预测10天 → lookback = max(10+50, 280) = **280天**
  - 预测100天 → lookback = max(100+50, 280) = **150天**（但实际仍需要280天）

### 2. `_direct_predict_snapshot` 中的多天预测

**代码位置**: `autotrader/app.py:1594-1653`

```python
# Get predictions for multiple days (for EMA smoothing)
all_predictions = []
end_date = pd.Timestamp.today()

# Fetch predictions for each day (from oldest to newest)
for day_offset in range(prediction_days - 1, -1, -1):  # From oldest to newest
    pred_date = end_date - pd.Timedelta(days=day_offset)
    
    results = model.predict_with_snapshot(
        feature_data=None,  # Auto-fetch from Polygon API
        snapshot_id=None,
        universe_tickers=tickers,
        as_of_date=pred_date,  # Use specific date
        prediction_days=1  # Get prediction for single day
    )
```

**关键点**:
- 每次调用 `predict_with_snapshot` 时，`as_of_date` 设置为特定日期
- `predict_with_snapshot` 会自动计算从 `as_of_date` 往前推280天的数据
- **每次预测都独立获取足够的历史数据**

### 3. 数据获取时间线示例

假设今天是 **2025-01-20**，需要获取最近3天的预测：

#### Day 1 (2025-01-18)
```
预测日期: 2025-01-18
数据获取范围: 2025-01-18 - 280天 = 2024-04-13 至 2025-01-18
实际获取: ~280个日历天的数据（确保有252个交易日）
```

#### Day 2 (2025-01-19)
```
预测日期: 2025-01-19
数据获取范围: 2025-01-19 - 280天 = 2024-04-14 至 2025-01-19
实际获取: ~280个日历天的数据
```

#### Day 3 (2025-01-20)
```
预测日期: 2025-01-20
数据获取范围: 2025-01-20 - 280天 = 2024-04-15 至 2025-01-20
实际获取: ~280个日历天的数据
```

## 数据获取优化

### 当前实现的问题

1. **重复数据获取**: 每次预测都独立获取280天的数据，存在大量重叠
2. **效率问题**: 3天预测需要获取3次280天的数据（共840天，但实际只需要约282天）

### 优化建议

#### 方案1: 批量获取（推荐）

```python
# 计算总体数据范围
end_date = pd.Timestamp.today()
start_date = end_date - pd.Timedelta(days=prediction_days + MIN_REQUIRED_LOOKBACK_DAYS)
# 例如: 3天预测 → start_date = today - (3 + 280) = today - 283天

# 一次性获取所有需要的数据
market_data = engine.fetch_market_data(
    symbols=tickers,
    start_date=start_date,
    end_date=end_date
)

# 计算所有因子
feature_data = engine.compute_all_17_factors(market_data, mode='predict')

# 然后按日期分割进行预测
for day_offset in range(prediction_days - 1, -1, -1):
    pred_date = end_date - pd.Timedelta(days=day_offset)
    # 从feature_data中提取该日期的数据
    day_features = feature_data[feature_data.index.get_level_values('date') == pred_date]
    # 进行预测
```

#### 方案2: 缓存机制

```python
# 缓存最近获取的数据
if not hasattr(self, '_cached_market_data') or \
   self._cached_market_data_end_date < end_date:
    # 重新获取数据
    self._cached_market_data = fetch_data(...)
    self._cached_market_data_end_date = end_date
```

## 因子计算的数据需求总结

### 按窗口大小分类

| 窗口大小 | 因子数量 | 因子示例 |
|---------|---------|---------|
| 252天 | 1 | `near_52w_high` |
| 200天 | 1 | `ma200` |
| 60天 | 2 | `trend_r2_60`, `ma60` |
| 40天 | 1 | `hist_vol_40d` |
| 30天 | 1 | `ma30` |
| 21天 | 2 | `rsi_21`, `downside_beta_ewm_21` |
| 20天 | 3 | `ret_skew_20d`, `vol_ratio_20d`, `atr_ratio` |
| 14天 | 1 | `bollinger_squeeze` |
| 5天 | 2 | `5_days_reversal`, `blowoff_ratio` |

### 关键约束

**最长窗口**: 252个交易日（`near_52w_high`）

**系统保证**: 
- 每次预测都获取至少280个日历天的数据
- 确保有足够的交易日来计算所有因子
- 即使只预测1天，也会获取280天的历史数据

## 实际数据获取流程

### 当前实现流程

```
_direct_predict_snapshot (预测3天)
  ↓
循环3次:
  ├─ Day 1: predict_with_snapshot(as_of_date=day1)
  │    └─ 获取: day1 - 280天 至 day1 的数据
  ├─ Day 2: predict_with_snapshot(as_of_date=day2)
  │    └─ 获取: day2 - 280天 至 day2 的数据
  └─ Day 3: predict_with_snapshot(as_of_date=day3)
       └─ 获取: day3 - 280天 至 day3 的数据
```

### 优化后的流程

```
_direct_predict_snapshot (预测3天)
  ↓
计算总体范围: today - (3 + 280) = today - 283天
  ↓
一次性获取: start_date 至 end_date 的所有数据
  ↓
计算所有因子
  ↓
循环3次:
  ├─ Day 1: 从feature_data提取day1的数据 → 预测
  ├─ Day 2: 从feature_data提取day2的数据 → 预测
  └─ Day 3: 从feature_data提取day3的数据 → 预测
```

## 结论

1. **数据充足性**: 系统通过 `MIN_REQUIRED_LOOKBACK_DAYS = 280天` 确保有足够的历史数据计算所有因子，包括需要252个交易日的 `near_52w_high`。

2. **当前实现**: 每次预测都独立获取280天的数据，虽然存在重复，但保证了数据的完整性和独立性。

3. **优化空间**: 可以通过批量获取和缓存机制减少重复数据获取，提高效率。

4. **关键保证**: 无论预测多少天，系统都会确保有至少280个日历天的历史数据，足以计算所有因子。

## 相关代码位置

- **Lookback计算**: `bma_models/量化模型_bma_ultra_enhanced.py:9571-9590`
- **多天预测循环**: `autotrader/app.py:1594-1653`
- **因子窗口定义**: `bma_models/simple_25_factor_engine.py:827, 1069-1077`
- **数据获取**: `bma_models/simple_25_factor_engine.py:178-268`
