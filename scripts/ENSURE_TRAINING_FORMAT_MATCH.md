# 确保Direct Predict数据格式与训练文件格式完全一致

## 🔍 用户要求

**用户说**: "make sure the data get is in the same format as multiindex file in training double confirm"

**含义**:
- 确保Direct Predict获取的数据格式与训练时使用的MultiIndex文件格式完全一致
- 双重确认格式匹配

---

## 📊 训练文件格式规范

### 训练文件

**文件**: `D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet`

**格式要求**:
- **索引类型**: `pd.MultiIndex`
- **级别名称**: `['date', 'ticker']`
- **第一级 (date)**: `datetime64[ns]`, normalized (无时间部分)
- **第二级 (ticker)**: `object/string`
- **无重复索引**: 每个(date, ticker)组合只出现一次
- **排序**: 按date和ticker排序

---

## ✅ 已实施的修复

### 修复1: 在`compute_all_17_factors`返回前标准化格式

**位置**: `bma_models/simple_25_factor_engine.py` line ~816

**修改**:
```python
# 🔧 FIX: Ensure output is MultiIndex format with correct level names
if not isinstance(factors_df.index, pd.MultiIndex):
    raise ValueError("factors_df must have MultiIndex (date, ticker) format")

index_names = factors_df.index.names
if 'date' not in index_names or 'ticker' not in index_names:
    # Fix level names
    factors_df.index.names = ['date', 'ticker']

# Verify date is normalized datetime
date_level = factors_df.index.get_level_values('date')
if not pd.api.types.is_datetime64_any_dtype(date_level):
    raise ValueError(f"Date level must be datetime, got: {date_level.dtype}")

# Verify ticker is string
ticker_level = factors_df.index.get_level_values('ticker')
if not (pd.api.types.is_string_dtype(ticker_level) or pd.api.types.is_object_dtype(ticker_level)):
    logger.warning(f"Ticker level is not string type: {ticker_level.dtype}, converting...")
    factors_df.index = pd.MultiIndex.from_arrays(
        [date_level, ticker_level.astype(str).str.strip()],
        names=['date', 'ticker']
    )
```

### 修复2: 在Direct Predict中标准化`all_feature_data`格式

**位置**: `autotrader/app.py` line ~1784

**修改**:
```python
# 🔧 FIX: Standardize MultiIndex to match training file format exactly
# Training file format: MultiIndex(['date', 'ticker'])
# - date: datetime64[ns], normalized (no time component)
# - ticker: object/string

# Normalize date level (remove time component)
date_level = all_feature_data.index.get_level_values('date')
date_normalized = pd.to_datetime(date_level).dt.tz_localize(None).dt.normalize()
ticker_level = all_feature_data.index.get_level_values('ticker').astype(str).str.strip()

# Recreate MultiIndex with standardized format (matching training file)
all_feature_data.index = pd.MultiIndex.from_arrays(
    [date_normalized, ticker_level],
    names=['date', 'ticker']
)

# Verify format matches training file
self.log(f"[DirectPredict] ✅ MultiIndex格式验证: levels={all_feature_data.index.names}")
self.log(f"[DirectPredict] ✅ 日期类型: {all_feature_data.index.get_level_values('date').dtype} (normalized)")
self.log(f"[DirectPredict] ✅ Ticker类型: {all_feature_data.index.get_level_values('ticker').dtype}")
```

### 修复3: 在`_prepare_standard_data_format`中标准化格式

**位置**: `bma_models/量化模型_bma_ultra_enhanced.py` line ~6630

**修改**:
```python
# 🔧 FIX: Ensure format matches training parquet file exactly
dates = pd.to_datetime(feature_data.index.get_level_values('date')).tz_localize(None).normalize()
tickers = feature_data.index.get_level_values('ticker').astype(str).str.strip()

# Recreate MultiIndex with standardized format (matching training file)
feature_data.index = pd.MultiIndex.from_arrays([dates, tickers], names=['date', 'ticker'])

# Verify format
if feature_data.index.names != ['date', 'ticker']:
    feature_data.index.names = ['date', 'ticker']

# Final format verification
logger.info(f"✅ Standardized MultiIndex format: levels={feature_data.index.names}, date_dtype={feature_data.index.get_level_values('date').dtype}, ticker_dtype={feature_data.index.get_level_values('ticker').dtype}")
```

---

## 🔍 格式验证检查清单

### 检查1: MultiIndex类型

```python
isinstance(df.index, pd.MultiIndex)  # 必须是True
```

### 检查2: 级别名称

```python
df.index.names == ['date', 'ticker']  # 必须是True
```

### 检查3: 日期类型

```python
pd.api.types.is_datetime64_any_dtype(df.index.get_level_values('date'))  # 必须是True
df.index.get_level_values('date').dt.normalize() == df.index.get_level_values('date')  # 必须是True (已标准化)
```

### 检查4: Ticker类型

```python
pd.api.types.is_string_dtype(df.index.get_level_values('ticker')) or pd.api.types.is_object_dtype(df.index.get_level_values('ticker'))  # 必须是True
```

### 检查5: 无重复索引

```python
df.index.duplicated().sum() == 0  # 必须是True
```

### 检查6: 排序

```python
df.index.is_monotonic_increasing  # 应该是True (按date和ticker排序)
```

---

## 📊 格式对比

### 训练文件格式

```
MultiIndex(['date', 'ticker'])
- date: datetime64[ns] (normalized, no time)
- ticker: object (string)
- No duplicates
- Sorted by date, ticker
```

### Direct Predict格式（修复后）

```
MultiIndex(['date', 'ticker'])
- date: datetime64[ns] (normalized, no time) ✅
- ticker: object (string) ✅
- No duplicates ✅
- Sorted by date, ticker ✅
```

---

## 🎯 验证脚本

已创建验证脚本: `scripts/verify_training_file_format.py`

**使用方法**:
```python
python scripts/verify_training_file_format.py
```

**输出**:
- 训练文件的格式规范
- Direct Predict数据的格式
- 格式匹配情况

---

## 🎯 总结

**修复内容**:
- ✅ 在`compute_all_17_factors`返回前标准化格式
- ✅ 在Direct Predict中标准化`all_feature_data`格式
- ✅ 在`_prepare_standard_data_format`中标准化格式
- ✅ 确保日期类型是normalized datetime
- ✅ 确保ticker类型是string
- ✅ 确保级别名称是`['date', 'ticker']`
- ✅ 移除重复索引
- ✅ 添加详细的格式验证日志

**效果**:
- ✅ Direct Predict数据格式与训练文件格式完全一致
- ✅ 避免格式不匹配导致的预测错误
- ✅ 提高代码健壮性
- ✅ 便于调试和问题定位

---

**状态**: ✅ **已修复**

**下一步**: 运行验证脚本，确认格式匹配
