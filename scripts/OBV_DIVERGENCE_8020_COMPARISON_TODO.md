# OBV_DIVERGENCE 80/20 对比评估 - 详细任务说明

## 任务概述

使用 80/20 时间分割评估对比 `obv_divergence` 因子的影响，使用 1/5 股票子集进行对比。

## 文件位置

### 1. 主脚本
**文件**: `scripts/compare_obv_divergence_8020_split.py`
- **功能**: 对比有无 `obv_divergence` 的 80/20 评估
- **输入**: 数据文件路径（自动检测）
- **输出**: `results/obv_divergence_8020_comparison/comparison_YYYYMMDD_HHMMSS.json`

### 2. 评估脚本
**文件**: `scripts/time_split_80_20_oos_eval.py`
- **功能**: 执行 80/20 时间分割评估
- **调用方式**: 通过 subprocess 调用
- **参数**:
  - `--data-file`: 数据文件路径（Parquet 格式）
  - `--horizon-days`: 预测天数（默认 10）
  - `--split`: 训练/测试分割比例（默认 0.8）
  - `--top-n`: Top N 股票数量（默认 20）
  - `--log-level`: 日志级别（INFO）

### 3. 数据文件
**主数据文件**: `data/factor_exports/polygon_factors_all_filtered_clean_final_v2.parquet`
**子集文件**: `data/factor_exports/polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet`
- **格式**: Parquet
- **索引**: MultiIndex(date, ticker)
- **必需列**: 
  - `target`: 目标变量（T+10 收益率）
  - `Close`: 收盘价
  - 所有 T10 因子列（包括 `obv_divergence`）

### 4. 因子引擎文件
**文件**: `bma_models/simple_25_factor_engine.py`
- **Line 56**: `'obv_divergence'` 在 `T10_ALPHA_FACTORS` 中
- **临时修改**: 脚本会临时注释掉 `obv_divergence` 进行对比
- **备份文件**: `.py.backup_obv_8020`

### 5. 输出目录
**目录**: `results/obv_divergence_8020_comparison/`
- **对比结果**: `comparison_YYYYMMDD_HHMMSS.json`
- **临时文件**: `temp_data_with_obv.parquet`, `temp_data_without_obv.parquet`（自动清理）

## 详细流程

### 阶段 1: 数据准备

1. **加载数据**
   - 优先使用子集文件（如果存在）
   - 否则使用完整数据文件
   - 验证文件存在性

2. **采样 1/5 股票**
   - 从 MultiIndex 中提取所有唯一 tickers
   - 随机采样 20% (1/5) 的 tickers
   - 随机种子: 42（确保可重复）

3. **过滤数据**
   - 根据采样的 tickers 过滤 DataFrame
   - 验证 MultiIndex 格式
   - 检查必需列（target, Close）

### 阶段 2: 实验 1 - 包含 obv_divergence

1. **数据过滤**
   - 使用采样的 tickers 过滤数据
   - 保存到临时文件: `temp_data_with_obv.parquet`

2. **运行 80/20 评估**
   ```bash
   python scripts/time_split_80_20_oos_eval.py \
     --data-file temp_data_with_obv.parquet \
     --horizon-days 10 \
     --split 0.8 \
     --top-n 20 \
     --log-level INFO
   ```

3. **收集结果**
   - 捕获 stdout/stderr
   - 记录执行时间
   - 提取关键指标（IC, Rank IC, Win Rate, Avg Return）

### 阶段 3: 实验 2 - 不包含 obv_divergence

1. **备份因子文件**
   - 备份: `bma_models/simple_25_factor_engine.py` → `.py.backup_obv_8020`

2. **修改因子文件**
   - 注释掉 `T10_ALPHA_FACTORS` 中的 `'obv_divergence'`
   - 修改位置: Line 56

3. **数据过滤和清理**
   - 使用采样的 tickers 过滤数据
   - 如果数据中有 `obv_divergence` 列，移除它
   - 保存到临时文件: `temp_data_without_obv.parquet`

4. **运行 80/20 评估**
   - 使用相同的参数运行评估脚本

5. **恢复因子文件**
   - 从备份恢复原文件
   - 确保文件恢复成功

### 阶段 4: 结果对比

1. **提取指标**
   - 从 stdout 中提取:
     - IC (Information Coefficient)
     - Rank IC
     - Win Rate
     - Avg Return

2. **计算差异**
   - 对比两个实验的指标
   - 计算差异

3. **保存结果**
   - JSON 格式保存对比结果
   - 包含时间戳、使用的股票、指标对比、差异分析

## 输入格式

### 数据文件格式要求

**Parquet 文件**，包含以下结构：

```python
# MultiIndex
index: MultiIndex(['date', 'ticker'])
  - date: datetime64[ns]
  - ticker: str

# 必需列
columns: [
    'target',           # T+10 收益率（必需）
    'Close',            # 收盘价（必需）
    'liquid_momentum',  # T10 因子
    'momentum_10d',     # T10 因子
    'momentum_60d',     # T10 因子
    'obv_divergence',   # T10 因子（实验1需要）
    'obv_momentum_60d', # T10 因子
    'ivol_20',          # T10 因子
    'hist_vol_40d',     # T10 因子
    'atr_ratio',        # T10 因子
    'rsi_21',           # T10 因子
    'trend_r2_60',      # T10 因子
    'near_52w_high',    # T10 因子
    'vol_ratio_20d',    # T10 因子
    'price_ma60_deviation', # T10 因子
    '5_days_reversal',  # T10 因子
    # ... 其他因子
]
```

### 脚本调用格式

```bash
# 直接运行（无参数，自动检测数据文件）
python scripts/compare_obv_divergence_8020_split.py
```

**自动行为**:
- 自动检测子集文件或完整数据文件
- 自动采样 1/5 tickers
- 自动运行两个实验
- 自动对比结果

## 输出格式

### JSON 对比结果

```json
{
  "timestamp": "2026-01-24T06:17:34.452626",
  "tickers_used": 156,
  "ticker_sample": ["ACEL", "ACR", "AHL", ...],
  "with_obv_divergence": {
    "success": true,
    "elapsed_time_minutes": 15.5,
    "metrics": {
      "ic": 0.023,
      "rank_ic": 0.031,
      "win_rate": 52.5,
      "avg_return": 0.45
    }
  },
  "without_obv_divergence": {
    "success": true,
    "elapsed_time_minutes": 14.8,
    "metrics": {
      "ic": 0.019,
      "rank_ic": 0.028,
      "win_rate": 51.2,
      "avg_return": 0.42
    }
  },
  "difference": {
    "ic": 0.004,
    "rank_ic": 0.003,
    "win_rate": 1.3,
    "avg_return": 0.03
  }
}
```

## 关键指标说明

### IC (Information Coefficient)
- **定义**: 预测值与实际值的 Pearson 相关系数
- **范围**: -1 到 1
- **解释**: 正值表示正相关，值越大越好

### Rank IC
- **定义**: 预测值与实际值的 Spearman 秩相关系数
- **范围**: -1 到 1
- **解释**: 基于排序的相关性，更稳健

### Win Rate
- **定义**: Top N 股票的胜率（正收益比例）
- **范围**: 0% 到 100%
- **解释**: 值越高越好，>50% 表示有效

### Avg Return
- **定义**: Top N 股票的平均收益率
- **范围**: 无限制
- **解释**: 正值表示盈利，值越大越好

## 执行步骤

### 步骤 1: 准备数据
- ✅ 确保数据文件存在
- ✅ 验证数据格式（MultiIndex, 必需列）

### 步骤 2: 运行对比脚本
```bash
cd D:\trade
python scripts\compare_obv_divergence_8020_split.py
```

### 步骤 3: 监控执行
- 观察两个实验的执行进度
- 检查是否有错误
- 验证文件恢复

### 步骤 4: 查看结果
- 检查 `results/obv_divergence_8020_comparison/comparison_*.json`
- 分析指标差异
- 评估 `obv_divergence` 的影响

## 注意事项

1. **文件修改**: 脚本会临时修改 `simple_25_factor_engine.py`，确保自动恢复
2. **数据采样**: 使用固定随机种子（42）确保可重复
3. **临时文件**: 自动清理临时数据文件
4. **错误处理**: 包含完整的错误处理和日志输出
5. **编码问题**: 已处理 Windows GBK 编码问题

## 当前状态

- ✅ 脚本已创建并修复编码问题
- ✅ 缩进错误已修复
- ⏳ 等待执行对比评估
- ⏳ 结果分析待完成

## 下一步

1. 运行对比脚本
2. 分析结果差异
3. 评估 `obv_divergence` 的实际影响
4. 根据结果决定是否保留该因子
