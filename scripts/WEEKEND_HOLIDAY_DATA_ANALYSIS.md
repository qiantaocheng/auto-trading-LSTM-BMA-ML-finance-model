# 分析：是否因为获取了周末或假期数据导致重复？

## 🔍 用户的问题

**用户问**: "would that be because fetch weekend day's data or holiday's data?"

**问题**: 是否因为获取了周末或假期的数据导致重复的(date, ticker)组合？

---

## 📊 Polygon API行为分析

### Polygon API的`timespan='day'`行为

**代码位置**: `polygon_client.py` line ~433-506

```python
df = polygon_client.get_historical_bars(
    symbol=symbol,
    timespan='day',  # 每天一条数据
    multiplier=1
)
```

**Polygon API文档说明**:
- `timespan='day'`应该**只返回交易日的数据**
- 周末和假期**不应该**有数据
- 每天每个ticker应该**只有一条记录**

**但是**: API可能在某些情况下返回非交易日数据，或者返回同一天的多条记录。

---

## 🎯 可能的情况

### 情况1: API返回了周末/假期数据

**如果API返回了周末数据**:
```python
# API返回:
#   2024-01-13 (Saturday), AAPL, Close=150.0
#   2024-01-15 (Monday), AAPL, Close=150.5

# normalize()后:
#   2024-01-13, AAPL, Close=150.0
#   2024-01-15, AAPL, Close=150.5

# 结果: 不会产生重复！因为日期不同
```

**结论**: 周末/假期数据**不会**导致重复，因为它们是不同的日期。

### 情况2: API返回了同一天的多条记录

**如果API返回了同一天的多条记录**:
```python
# API返回:
#   2024-01-15 09:30:00, AAPL, Close=150.0
#   2024-01-15 16:00:00, AAPL, Close=150.5  # 同一天，不同时间

# normalize()后:
#   2024-01-15, AAPL, Close=150.0
#   2024-01-15, AAPL, Close=150.5  # 重复！

# 结果: 会产生重复的(date, ticker)组合
```

**结论**: 如果API返回了**同一天的多条记录**（不同时间戳），标准化后会变成重复。

### 情况3: 数据合并时产生重复

**如果多个DataFrame有重叠的索引**:
```python
# 假设有两个DataFrame都包含AAPL的数据
df1 = pd.DataFrame({'Close': [150.0]}, index=[pd.Timestamp('2024-01-15')])
df2 = pd.DataFrame({'Close': [150.5]}, index=[pd.Timestamp('2024-01-15')])

# 合并后
combined = pd.concat([df1, df2], ignore_index=False)
# combined.index = [2024-01-15, 2024-01-15]  # 重复！
```

**结论**: 如果数据合并时没有正确处理，可能产生重复。

---

## 🔍 验证方法

### 检查1: 检查API返回的数据是否包含周末/假期

```python
# 在fetch_market_data返回后
if 'date' in combined.columns:
    combined['date'] = pd.to_datetime(combined['date'])
    combined['weekday'] = combined['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    # 检查周末数据
    weekend_data = combined[combined['weekday'].isin([5, 6])]  # Saturday, Sunday
    if len(weekend_data) > 0:
        logger.warning(f"⚠️ API returned {len(weekend_data)} weekend records!")
        logger.warning(f"⚠️ Weekend dates: {weekend_data['date'].unique()[:10]}")
    
    # 检查是否有重复的(date, ticker)组合
    duplicates = combined.duplicated(subset=['date', 'ticker'])
    if duplicates.any():
        logger.error(f"❌ API returned {duplicates.sum()} duplicate (date, ticker) combinations!")
        # 检查这些重复是否在周末
        dup_data = combined[combined.duplicated(subset=['date', 'ticker'], keep=False)]
        weekend_dups = dup_data[dup_data['weekday'].isin([5, 6])]
        if len(weekend_dups) > 0:
            logger.error(f"❌ {len(weekend_dups)} duplicates are on weekends!")
```

### 检查2: 检查日期标准化后是否产生重复

```python
# 在compute_data创建后
compute_data['date'] = pd.to_datetime(compute_data['date']).dt.normalize()
compute_data['weekday'] = compute_data['date'].dt.dayofweek

# 检查重复
duplicates = compute_data.duplicated(subset=['date', 'ticker'])
if duplicates.any():
    dup_data = compute_data[compute_data.duplicated(subset=['date', 'ticker'], keep=False)]
    
    # 检查重复是否在周末
    weekend_dups = dup_data[dup_data['weekday'].isin([5, 6])]
    if len(weekend_dups) > 0:
        logger.warning(f"⚠️ {len(weekend_dups)} duplicates are on weekends!")
        logger.warning(f"⚠️ This suggests API returned weekend data!")
    
    # 检查重复是否在同一天（不同时间戳）
    for (date, ticker), group in dup_data.groupby(['date', 'ticker']):
        if len(group) > 1:
            logger.warning(f"⚠️ ({date}, {ticker}): {len(group)} rows")
            logger.warning(f"⚠️   Weekday: {group['weekday'].iloc[0]} ({'Weekend' if group['weekday'].iloc[0] in [5, 6] else 'Weekday'})")
```

---

## ✅ 解决方案

### 修复1: 过滤掉周末/假期数据

**位置**: `bma_models/simple_25_factor_engine.py` line ~350

**修改**:
```python
compute_data = market_data_clean.copy()
compute_data['date'] = pd.to_datetime(compute_data['date']).dt.normalize()
compute_data = compute_data.sort_values(['ticker', 'date']).reset_index(drop=True)

# 🔧 FIX: Filter out weekend data (Saturday=5, Sunday=6)
if 'date' in compute_data.columns:
    compute_data['weekday'] = compute_data['date'].dt.dayofweek
    weekend_count = (compute_data['weekday'].isin([5, 6])).sum()
    if weekend_count > 0:
        logger.warning(f"⚠️ Filtering out {weekend_count} weekend records (should not exist)")
        compute_data = compute_data[~compute_data['weekday'].isin([5, 6])].reset_index(drop=True)
    compute_data = compute_data.drop(columns=['weekday'])

# 🔧 FIX: Remove duplicate (date, ticker) combinations immediately
if 'date' in compute_data.columns and 'ticker' in compute_data.columns:
    duplicates = compute_data.duplicated(subset=['date', 'ticker'], keep='last')
    if duplicates.any():
        dup_count = duplicates.sum()
        logger.warning(f"⚠️ compute_data: Removing {dup_count} duplicate (date, ticker) combinations")
        # Log examples for debugging
        dup_data = compute_data[compute_data.duplicated(subset=['date', 'ticker'], keep=False)]
        if len(dup_data) > 0:
            logger.warning(f"⚠️ Duplicate examples (first 3):")
            for (date, ticker), group in list(dup_data.groupby(['date', 'ticker']))[:3]:
                logger.warning(f"  ({date}, {ticker}): {len(group)} rows")
                if 'Close' in group.columns:
                    logger.warning(f"    Close values: {group['Close'].tolist()}")
        
        compute_data = compute_data[~duplicates].reset_index(drop=True)
    
    unique_pairs = compute_data.groupby(['date', 'ticker']).size().shape[0]
    logger.info(f"✅ compute_data after deduplication: {len(compute_data)} rows, {unique_pairs} unique (date, ticker) pairs")
```

### 修复2: 在`fetch_market_data`返回前过滤

**位置**: `bma_models/simple_25_factor_engine.py` line ~260

**修改**:
```python
if all_data:
    combined = pd.concat(all_data, ignore_index=False)
    combined = combined.reset_index()
    
    if 'Date' in combined.columns:
        combined = combined.rename(columns={'Date': 'date'})
    
    # 🔧 FIX: Filter out weekend data
    if 'date' in combined.columns:
        combined['date'] = pd.to_datetime(combined['date']).dt.normalize()
        combined['weekday'] = combined['date'].dt.dayofweek
        weekend_count = (combined['weekday'].isin([5, 6])).sum()
        if weekend_count > 0:
            logger.warning(f"⚠️ fetch_market_data: Filtering out {weekend_count} weekend records")
            combined = combined[~combined['weekday'].isin([5, 6])].reset_index(drop=True)
        combined = combined.drop(columns=['weekday'])
    
    # 🔧 FIX: Remove duplicate (date, ticker) combinations
    if 'date' in combined.columns and 'ticker' in combined.columns:
        duplicates = combined.duplicated(subset=['date', 'ticker'], keep='last')
        if duplicates.any():
            logger.warning(f"⚠️ fetch_market_data: Removing {duplicates.sum()} duplicate (date, ticker) combinations")
            combined = combined[~duplicates].reset_index(drop=True)
    
    return combined
```

---

## 🎯 总结

**是否因为周末/假期数据导致重复？**

**答案**: **不太可能**，因为：
1. 周末和假期是**不同的日期**，不会产生重复的(date, ticker)组合
2. Polygon API的`timespan='day'`应该**只返回交易日数据**

**更可能的原因**:
1. **API返回了同一天的多条记录**（不同时间戳）→ 标准化后变成重复
2. **数据合并时产生了重复** → `pd.concat`保留了重叠的索引
3. **API bug返回了重复数据** → 虽然不应该，但可能发生

**解决方案**:
- 过滤掉周末数据（虽然不应该存在）
- 在数据流的每个关键节点去重
- 添加诊断日志，定位重复来源

---

**状态**: ⚠️ **需要添加周末过滤和去重逻辑**

**下一步**: 实施修复，过滤周末数据并确保每个(date, ticker)组合只出现一次
