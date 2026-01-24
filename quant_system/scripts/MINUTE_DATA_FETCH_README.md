# 分钟级别数据获取说明

## 概述

此脚本用于从Polygon API获取2025年全年的分钟级别股票数据，并存储为MultiIndex格式。

## 数据格式

### MultiIndex结构

```
Index: (Symbol, DateTime)
Columns: Open, High, Low, Close, Volume
```

**示例：**
```
Symbol  DateTime            Open    High    Low     Close   Volume
AAPL    2025-01-02 09:00:00 185.50  185.75  185.45  185.60  1234567
AAPL    2025-01-02 09:01:00 185.60  185.80  185.55  185.70  1234568
...
MSFT    2025-01-02 09:00:00 380.20  380.50  380.10  380.30  2345678
...
```

## 使用方法

### 1. 获取所有ticker的数据（2025年全年）

```bash
cd d:\trade\quant_system\scripts
python fetch_minute_data_2025.py \
    --start-date 2025-01-01 \
    --end-date 2025-12-31 \
    --output ../data/minute_data_2025.parquet \
    --batch-size 5
```

### 2. 获取指定ticker的数据

```bash
python fetch_minute_data_2025.py \
    --tickers AAPL MSFT GOOGL \
    --start-date 2025-01-01 \
    --end-date 2025-01-31 \
    --output test_minute_data.parquet
```

### 3. 参数说明

- `--tickers`: 指定ticker列表（如果不提供，使用所有240个ticker）
- `--start-date`: 开始日期（YYYY-MM-DD，默认：2025-01-01）
- `--end-date`: 结束日期（YYYY-MM-DD，默认：2025-12-31）
- `--output`: 输出文件路径（默认：minute_data_2025.parquet）
- `--cache-dir`: 缓存目录（默认：data_cache/minute_data_2025）
- `--no-cache`: 禁用缓存
- `--batch-size`: 检查点保存的批次大小（默认：10）

## 功能特性

### 1. 自动检查点保存

- 每处理`batch_size`个ticker后自动保存检查点
- 如果中断，可以从检查点恢复
- 检查点文件：`cache_dir/checkpoint_{start_date}_{end_date}.pkl`

### 2. 缓存机制

- 每个ticker的数据单独缓存
- 缓存文件：`cache_dir/{symbol}_{start_date}_{end_date}_minute.pkl`
- 避免重复获取相同数据

### 3. 数据验证

- 自动验证OHLC关系（High >= max(Open, Close), Low <= min(Open, Close)）
- 移除无效价格（<= 0或NaN）
- 前向填充缺失值
- 移除重复行

### 4. 错误处理

- 自动重试失败的请求
- 记录失败的ticker列表
- 继续处理其他ticker

## 数据量估算

### 单个ticker（2025年）

- 交易日：约252天
- 交易时间：6.5小时/天 = 390分钟/天
- 总分钟数：252 × 390 ≈ 98,280分钟

### 全部240个ticker

- 总行数：240 × 98,280 ≈ 23,587,200行
- 数据大小（估算）：约5-10GB（取决于压缩）

## 输出文件

### Parquet格式

- 高效压缩
- 支持快速查询
- 兼容pandas/polars

### 读取数据示例

```python
import pandas as pd

# 读取全部数据
df = pd.read_parquet('minute_data_2025.parquet')

# 查询特定ticker
aapl_data = df.loc['AAPL']

# 查询特定日期范围
date_range = df.loc[(slice(None), '2025-01-01':'2025-01-31'), :]

# 查询特定ticker和日期
aapl_jan = df.loc[('AAPL', '2025-01-01':'2025-01-31'), :]
```

## 进度监控

### 查看日志

```bash
# 实时查看日志
tail -f minute_fetch_log.txt

# 查看最后50行
tail -n 50 minute_fetch_log.txt
```

### 检查进度

```python
import pickle
from pathlib import Path

checkpoint_file = Path('data_cache/minute_data_2025/checkpoint_2025-01-01_2025-12-31.pkl')
if checkpoint_file.exists():
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
        print(f"Processed: {len(checkpoint['processed'])} symbols")
        print(f"Timestamp: {checkpoint['timestamp']}")
```

## 注意事项

1. **API限制**：
   - Polygon API有速率限制
   - 脚本已内置延迟（0.3秒/请求）
   - 240个ticker × 365天 ≈ 需要较长时间

2. **数据完整性**：
   - 某些ticker可能没有完整数据
   - 检查日志中的"failed symbols"列表

3. **存储空间**：
   - 确保有足够的磁盘空间（建议10GB+）
   - Parquet格式会自动压缩

4. **网络稳定性**：
   - 建议在稳定的网络环境下运行
   - 检查点机制可以从中断处恢复

## 因子计算

如果需要计算因子，可以：

1. **读取分钟数据**：
```python
df = pd.read_parquet('minute_data_2025.parquet')
```

2. **转换为日线数据**（如果需要）：
```python
daily_df = df.groupby(level=0).resample('D', level=1).agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})
```

3. **计算因子**：
```python
# 示例：计算移动平均
df['MA20'] = df.groupby(level=0)['Close'].rolling(20).mean()
```

4. **保存因子数据**：
```python
df.to_parquet('minute_data_2025_with_factors.parquet')
```

## 故障排除

### 问题1：API错误

**症状**：`HTTP 429`或`HTTP 503`错误

**解决**：
- 增加延迟时间（修改脚本中的`time.sleep(0.3)`）
- 检查API配额
- 使用`--no-cache`避免重复请求

### 问题2：内存不足

**症状**：`MemoryError`

**解决**：
- 减少`batch_size`（例如：`--batch-size 3`）
- 分批处理ticker（使用`--tickers`参数）

### 问题3：数据不完整

**症状**：某些ticker没有数据

**解决**：
- 检查日志中的"failed symbols"
- 单独重新获取失败的ticker
- 某些ticker可能在2025年不存在或已退市

## 相关文件

- `fetch_minute_data_2025.py`: 主脚本
- `extract_tickers_from_list.py`: Ticker列表提取脚本
- `minute_data_2025.parquet`: 输出数据文件（MultiIndex格式）
