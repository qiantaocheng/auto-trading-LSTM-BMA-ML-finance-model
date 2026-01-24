# OBV_DIVERGENCE 80/20 对比评估 - 完整任务指南

## 📋 任务概述

**目标**: 使用 80/20 时间分割评估对比 `obv_divergence` 因子的影响，使用 1/5 股票子集

**状态**: 
- ✅ 脚本已创建
- ✅ 代码错误已修复
- ✅ 已有部分结果（但指标提取需要改进）
- ⏳ 需要改进指标提取逻辑

## 📁 文件位置详细说明

### 1. 主对比脚本
**路径**: `D:\trade\scripts\compare_obv_divergence_8020_split.py`
**行数**: 480 行
**关键函数位置**:
- `sample_tickers()`: **Line 16-32** - 采样 1/5 tickers
- `filter_dataframe_by_tickers()`: **Line 34-41** - 过滤数据
- `run_8020_eval_with_obv_divergence()`: **Line 43-138** - 实验1（包含 obv_divergence）
- `run_8020_eval_without_obv_divergence()`: **Line 140-280** - 实验2（不包含 obv_divergence）
- `extract_metrics_from_output()`: **Line 282-310** - 从 stdout 提取指标 ⚠️ **需要改进**
- `compare_results()`: **Line 312-380** - 对比结果
- `main()`: **Line 382-480** - 主函数

### 2. 80/20 评估脚本
**路径**: `D:\trade\scripts\time_split_80_20_oos_eval.py`
**行数**: 2984 行
**关键位置**:
- 参数解析: **Line 340-367**
- 主函数: **Line 1285**
- 指标输出: **Line ~2800-2900** (需要检查实际输出格式)

### 3. 因子引擎
**路径**: `D:\trade\bma_models\simple_25_factor_engine.py`
**关键位置**:
- T10_ALPHA_FACTORS 定义: **Line 52-68**
- obv_divergence 位置: **Line 56** ⭐
- 因子计算: **Line 1334-1357** (`_compute_volume_factors`)
- 临时修改: 脚本会注释 Line 56

### 4. 主模型文件
**路径**: `D:\trade\bma_models\量化模型_bma_ultra_enhanced.py`
**关键位置**:
- compulsory_features: **Line 3245-3250**
- 因子选择: **Line 3239-3301** (已修复，始终 T+10)

### 5. 数据文件
**主文件**: `D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet`
**子集文件**: `D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet`
- **格式**: Parquet
- **索引**: MultiIndex(date, ticker)
- **形状**: (827900, 21) - 子集文件
- **唯一 tickers**: 784 个 → 采样后 156 个 (1/5)

### 6. 输出目录
**路径**: `D:\trade\results\obv_divergence_8020_comparison\`
**文件**:
- `comparison_YYYYMMDD_HHMMSS.json` - 对比结果
- `temp_data_with_obv.parquet` - 临时文件（自动清理）
- `temp_data_without_obv.parquet` - 临时文件（自动清理）

## 🔄 详细执行流程

### 阶段 1: 初始化 (main(), Line 382-400)

```python
# 1. 检测数据文件
subset_file = "data/factor_exports/...subset_1_5_tickers.parquet"
if exists: use subset_file
else: use full data file

# 2. 创建输出目录
output_dir = "results/obv_divergence_8020_comparison"
```

### 阶段 2: 数据采样 (main(), Line 402-410)

```python
# 1. 加载数据
df = pd.read_parquet(data_file)  # Shape: (827900, 21)

# 2. 采样 1/5 tickers
tickers = sample_tickers(df, fraction=0.2, random_seed=42)
# 结果: 156 tickers from 784 total

# 3. 显示采样信息
print(f"采样股票数: {len(tickers)} / {total_tickers}")
```

### 阶段 3: 实验1 - WITH obv_divergence (Line 43-138)

```python
# 1. 加载并过滤数据
df = pd.read_parquet(data_file)  # (827900, 21)
df_filtered = filter_dataframe_by_tickers(df, tickers)  # (166346, 21)

# 2. 验证数据格式
- MultiIndex levels: ['date', 'ticker']
- Unique dates: 1244
- Unique tickers: 156

# 3. 保存临时文件
temp_file = "temp_data_with_obv.parquet"  # ~25 MB

# 4. 运行评估
subprocess.run([
    "python", "scripts/time_split_80_20_oos_eval.py",
    "--data-file", temp_file,
    "--horizon-days", "10",
    "--split", "0.8",
    "--top-n", "20",
    "--log-level", "INFO"
])

# 5. 提取指标（从 stdout）
metrics = extract_metrics_from_output(stdout)

# 6. 清理临时文件
temp_file.unlink()
```

### 阶段 4: 实验2 - WITHOUT obv_divergence (Line 140-280)

```python
# 1. 备份因子文件
backup_file = "simple_25_factor_engine.py.backup_obv_8020"
copy(factor_engine_file, backup_file)

# 2. 修改因子文件
# Line 56: 'obv_divergence' → # 'obv_divergence' (注释掉)

# 3. 加载并过滤数据
df = pd.read_parquet(data_file)
df_filtered = filter_dataframe_by_tickers(df, tickers)

# 4. 移除 obv_divergence 列（如果存在）
if 'obv_divergence' in df_filtered.columns:
    df_filtered = df_filtered.drop(columns=['obv_divergence'])

# 5. 保存临时文件
temp_file = "temp_data_without_obv.parquet"

# 6. 运行评估（相同参数）

# 7. 恢复因子文件
copy(backup_file, factor_engine_file)

# 8. 清理临时文件
```

### 阶段 5: 结果对比 (Line 312-380)

```python
# 1. 提取指标
with_metrics = extract_metrics_from_output(with_stdout)
without_metrics = extract_metrics_from_output(without_stdout)

# 2. 计算差异
difference = {
    'ic': with_ic - without_ic,
    'rank_ic': with_rank_ic - without_rank_ic,
    'win_rate': with_win_rate - without_win_rate,
    'avg_return': with_avg_return - without_avg_return
}

# 3. 保存 JSON
comparison_file = "comparison_YYYYMMDD_HHMMSS.json"
```

## 📥 输入格式详细说明

### 数据文件要求

**文件格式**: Parquet
**索引结构**: MultiIndex
```python
index = pd.MultiIndex.from_arrays(
    [dates, tickers],  # dates: datetime64[ns], tickers: str
    names=['date', 'ticker']
)
```

**必需列**:
```python
必需列 = {
    'target': float64,           # T+10 收益率（必需）
    'Close': float64,            # 收盘价（必需）
    'liquid_momentum': float64,  # T10 因子
    'momentum_10d': float64,     # T10 因子
    'momentum_60d': float64,     # T10 因子
    'obv_divergence': float64,   # ⭐ 对比因子（实验1需要）
    'obv_momentum_60d': float64, # T10 因子
    'ivol_20': float64,          # T10 因子
    'hist_vol_40d': float64,     # T10 因子
    'atr_ratio': float64,        # T10 因子
    'rsi_21': float64,           # T10 因子
    'trend_r2_60': float64,      # T10 因子
    'near_52w_high': float64,    # T10 因子
    'vol_ratio_20d': float64,    # T10 因子
    'price_ma60_deviation': float64,  # T10 因子
    '5_days_reversal': float64,  # T10 因子
}
```

**数据示例**:
```
date       ticker  target    Close  liquid_momentum  obv_divergence  ...
2020-01-02 AAPL    0.0234    150.0  0.0123          0.0045          ...
2020-01-02 MSFT    -0.0102   180.0  0.0089          -0.0023         ...
2020-01-03 AAPL    0.0156    152.0  0.0134          0.0056          ...
```

### 脚本调用

**无参数调用**（推荐）:
```bash
cd D:\trade
python scripts\compare_obv_divergence_8020_split.py
```

**自动检测**:
- ✅ 优先使用子集文件
- ✅ 自动采样 1/5 tickers
- ✅ 自动运行两个实验
- ✅ 自动对比结果

## 📤 输出格式详细说明

### JSON 结果格式

```json
{
  "timestamp": "2026-01-24T06:22:59.380782",
  "tickers_used": 156,
  "ticker_sample": ["ACEL", "ACR", "AHL", ...],
  "with_obv_divergence": {
    "success": true,
    "elapsed_time_minutes": 2.03,
    "metrics": {
      "ic": 0.023,           // 信息系数
      "rank_ic": 0.031,      // 排序信息系数
      "win_rate": 52.5,      // 胜率 (%)
      "avg_return": 0.45     // 平均收益率 (%)
    }
  },
  "without_obv_divergence": {
    "success": true,
    "elapsed_time_minutes": 1.99,
    "metrics": {
      "ic": 0.019,
      "rank_ic": 0.028,
      "win_rate": 51.2,
      "avg_return": 0.42
    }
  },
  "difference": {
    "ic": 0.004,            // 差异: +0.004 (obv_divergence 提升 IC)
    "rank_ic": 0.003,       // 差异: +0.003
    "win_rate": 1.3,        // 差异: +1.3% (提升胜率)
    "avg_return": 0.03      // 差异: +0.03% (提升收益)
  }
}
```

### 指标提取逻辑（需要改进）

**当前实现** (`extract_metrics_from_output`, Line 282-310):
```python
# 使用正则表达式提取
ic_match = re.search(r'IC[:\s]+([-]?\d+\.?\d*)', stdout)
rank_ic_match = re.search(r'Rank[_\s]?IC[:\s]+([-]?\d+\.?\d*)', stdout, re.IGNORECASE)
win_rate_match = re.search(r'Win[_\s]?Rate[:\s]+(\d+\.?\d*)%?', stdout, re.IGNORECASE)
avg_return_match = re.search(r'Avg[_\s]?Return[:\s]+([-]?\d+\.?\d*)%?', stdout, re.IGNORECASE)
```

**问题**: 当前结果中 `metrics: {}` 为空，说明正则表达式没有匹配到输出格式。

**需要**: 检查 `time_split_80_20_oos_eval.py` 的实际输出格式，更新正则表达式。

## 🔍 当前结果分析

### 已有结果文件

1. **comparison_20260124_062259.json** (最新)
   - ✅ 两个实验都成功 (`success: true`)
   - ⚠️ 指标为空 (`metrics: {}`)
   - ⏱️ 执行时间: ~2 分钟

2. **comparison_20260124_061633.json**
   - ❌ 两个实验都失败 (`success: false`)
   - ⏱️ 执行时间: <0.1 分钟（快速失败）

3. **comparison_20260124_061544.json**
   - ❌ 两个实验都失败 (`success: false`)
   - ⏱️ 执行时间: <0.1 分钟（快速失败）

### 问题诊断

**最新结果 (062259)**:
- ✅ 实验执行成功
- ❌ 指标提取失败（`metrics: {}`）
- **原因**: 正则表达式没有匹配到实际输出格式

**需要改进**:
1. 检查 `time_split_80_20_oos_eval.py` 的实际输出格式
2. 更新 `extract_metrics_from_output()` 的正则表达式
3. 或者从评估脚本的输出文件中读取指标

## 🛠️ 需要完成的任务

### 任务 1: 改进指标提取 ⚠️ **关键**

**文件**: `scripts/compare_obv_divergence_8020_split.py`
**函数**: `extract_metrics_from_output()` (Line 282-310)

**需要**:
1. 检查 `time_split_80_20_oos_eval.py` 的实际输出格式
2. 查看评估脚本生成的报告文件（CSV/JSON）
3. 更新正则表达式或添加文件读取逻辑

**可能的输出位置**:
- `results/t10_time_split_80_20_final/run_*/report_df.csv`
- `results/t10_time_split_80_20_final/run_*/ridge_top20_timeseries.csv`

### 任务 2: 验证评估脚本输出

**检查点**:
1. 评估脚本是否正常完成？
2. 输出文件是否生成？
3. 指标在哪个文件中？

### 任务 3: 重新运行对比（如果需要）

**如果指标提取修复后**:
```bash
python scripts\compare_obv_divergence_8020_split.py
```

## 📊 80/20 评估脚本内部流程

### 评估脚本执行流程 (`time_split_80_20_oos_eval.py`)

1. **数据加载** (Line ~1300)
   ```python
   df = pd.read_parquet(data_file)
   # 验证 MultiIndex
   # 检查必需列
   ```

2. **时间分割** (Line ~1400)
   ```python
   unique_dates = sorted(df.index.get_level_values('date').unique())
   split_idx = int(len(unique_dates) * 0.8)
   train_dates = unique_dates[:split_idx]
   test_dates = unique_dates[split_idx:]
   ```

3. **训练阶段** (Line ~1500)
   ```python
   # 使用训练集训练模型
   # 生成 OOF 预测
   # 训练第一层模型 + Meta Stacker
   ```

4. **测试阶段** (Line ~2000)
   ```python
   # 对测试集每日预测
   # Top 20 重新平衡
   # 计算指标
   ```

5. **结果输出** (Line ~2800)
   ```python
   # 输出到 stdout
   # 保存 CSV 文件
   # 生成图表
   ```

## 🎯 下一步行动

1. **检查评估脚本输出格式**
   - 查看 `time_split_80_20_oos_eval.py` 的实际输出
   - 确认指标在 stdout 中的格式

2. **改进指标提取**
   - 更新正则表达式
   - 或从输出文件读取指标

3. **重新运行对比**
   - 使用改进后的指标提取
   - 生成完整的对比结果

4. **分析结果**
   - 评估 `obv_divergence` 的影响
   - 做出是否保留该因子的决定

## 📝 关键代码位置总结

| 功能 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 采样 tickers | `compare_obv_divergence_8020_split.py` | 16-32 | 随机采样 1/5 |
| 过滤数据 | `compare_obv_divergence_8020_split.py` | 34-41 | MultiIndex 过滤 |
| 实验1 | `compare_obv_divergence_8020_split.py` | 43-138 | WITH obv_divergence |
| 实验2 | `compare_obv_divergence_8020_split.py` | 140-280 | WITHOUT obv_divergence |
| 指标提取 | `compare_obv_divergence_8020_split.py` | 282-310 | ⚠️ 需要改进 |
| 结果对比 | `compare_obv_divergence_8020_split.py` | 312-380 | 生成对比报告 |
| obv_divergence | `simple_25_factor_engine.py` | 56 | T10 因子定义 |
| 因子计算 | `simple_25_factor_engine.py` | 1334-1357 | 计算逻辑 |
| 评估脚本 | `time_split_80_20_oos_eval.py` | 1285 | 主函数 |

## ✅ 完成状态

- ✅ 脚本创建完成
- ✅ 编码问题修复
- ✅ 缩进错误修复
- ✅ T5 因子移除完成
- ✅ T10 因子统一完成
- ⚠️ 指标提取需要改进（当前为空）
- ⏳ 完整对比结果待生成
