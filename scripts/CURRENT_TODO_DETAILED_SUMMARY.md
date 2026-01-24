# 当前任务详细总结

## 任务目标

使用 80/20 时间分割评估对比 `obv_divergence` 因子的影响，使用 1/5 股票子集进行对比。

## 文件位置和流程

### 📁 主要文件

#### 1. 对比脚本
**路径**: `scripts/compare_obv_divergence_8020_split.py`
- **功能**: 自动化对比有无 `obv_divergence` 的 80/20 评估
- **行数**: 480 行
- **主要函数**:
  - `sample_tickers()`: 采样 1/5 tickers (Line 16-32)
  - `filter_dataframe_by_tickers()`: 过滤数据 (Line 34-41)
  - `run_8020_eval_with_obv_divergence()`: 实验1 (Line 43-138)
  - `run_8020_eval_without_obv_divergence()`: 实验2 (Line 140-280)
  - `extract_metrics_from_output()`: 提取指标 (Line 282-310)
  - `compare_results()`: 对比结果 (Line 312-380)
  - `main()`: 主函数 (Line 382-480)

#### 2. 评估脚本
**路径**: `scripts/time_split_80_20_oos_eval.py`
- **功能**: 执行 80/20 时间分割评估
- **调用方式**: 通过 subprocess 调用
- **参数解析**: Line 340-367
- **主函数**: Line 1285

#### 3. 因子引擎
**路径**: `bma_models/simple_25_factor_engine.py`
- **T10 因子定义**: Line 52-68
- **obv_divergence 位置**: Line 56
- **因子计算**: Line 1334-1357 (`_compute_volume_factors`)
- **临时修改**: 脚本会注释 Line 56 的 `'obv_divergence'`

#### 4. 主模型文件
**路径**: `bma_models/量化模型_bma_ultra_enhanced.py`
- **compulsory_features**: Line 3245-3250
- **因子选择逻辑**: Line 3239-3301
- **已修复**: 移除 T5 因子，始终使用 T10

### 📊 数据文件

#### 输入数据
**主文件**: `data/factor_exports/polygon_factors_all_filtered_clean_final_v2.parquet`
**子集文件**: `data/factor_exports/polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet`
- **格式**: Parquet
- **索引**: MultiIndex(date, ticker)
- **形状**: (827900, 21) - 子集文件
- **唯一 tickers**: 784 个（子集文件）
- **采样后**: 约 156 个 tickers (1/5)

#### 必需列
```python
必需列 = [
    'target',              # T+10 收益率（必需）
    'Close',               # 收盘价（必需）
    # T10 因子（14个）
    'liquid_momentum',
    'momentum_10d',
    'momentum_60d',
    'obv_divergence',      # ⭐ 对比因子
    'obv_momentum_60d',
    'ivol_20',
    'hist_vol_40d',
    'atr_ratio',
    'rsi_21',
    'trend_r2_60',
    'near_52w_high',
    'vol_ratio_20d',
    'price_ma60_deviation',
    '5_days_reversal',
]
```

### 📁 输出文件

#### 结果目录
**路径**: `results/obv_divergence_8020_comparison/`
- **对比结果**: `comparison_YYYYMMDD_HHMMSS.json`
- **临时文件**: `temp_data_with_obv.parquet`, `temp_data_without_obv.parquet`（自动清理）

#### 已有结果文件
1. `comparison_20260124_061544.json`
2. `comparison_20260124_061633.json`
3. `comparison_20260124_062259.json`

## 详细流程

### 🔄 完整执行流程

```
开始
  ↓
[1] 数据准备
  ├─ 检测数据文件（优先子集文件）
  ├─ 加载 Parquet 数据
  ├─ 验证 MultiIndex 格式
  └─ 采样 1/5 tickers (随机种子=42)
  ↓
[2] 实验1: WITH obv_divergence
  ├─ 过滤数据（使用采样的 tickers）
  ├─ 验证数据格式
  ├─ 保存临时文件: temp_data_with_obv.parquet
  ├─ 调用 80/20 评估脚本
  │   └─ scripts/time_split_80_20_oos_eval.py
  │       ├─ --data-file: temp_data_with_obv.parquet
  │       ├─ --horizon-days: 10
  │       ├─ --split: 0.8
  │       ├─ --top-n: 20
  │       └─ --log-level: INFO
  ├─ 捕获 stdout/stderr
  ├─ 提取指标（IC, Rank IC, Win Rate, Avg Return）
  └─ 清理临时文件
  ↓
[3] 实验2: WITHOUT obv_divergence
  ├─ 备份因子文件: simple_25_factor_engine.py → .backup_obv_8020
  ├─ 修改因子文件: 注释 'obv_divergence' (Line 56)
  ├─ 过滤数据（使用采样的 tickers）
  ├─ 移除 obv_divergence 列（如果存在）
  ├─ 保存临时文件: temp_data_without_obv.parquet
  ├─ 调用 80/20 评估脚本（相同参数）
  ├─ 捕获 stdout/stderr
  ├─ 提取指标
  ├─ 恢复因子文件（从备份）
  └─ 清理临时文件
  ↓
[4] 结果对比
  ├─ 提取两个实验的指标
  ├─ 计算差异
  ├─ 生成对比报告
  └─ 保存 JSON 结果
  ↓
结束
```

### 📝 80/20 时间分割逻辑

**评估脚本内部流程** (`time_split_80_20_oos_eval.py`):

1. **加载数据** (Line ~1300)
   - 读取 Parquet 文件
   - 验证 MultiIndex 格式
   - 检查必需列

2. **时间分割** (Line ~1400)
   - 按日期排序
   - 前 80% 日期 → 训练集
   - 后 20% 日期 → 测试集
   - Gap = horizon_days (10天) 防止数据泄漏

3. **训练阶段** (Line ~1500)
   - 使用训练集训练模型
   - 生成 OOF 预测
   - 训练第一层模型 + Meta Stacker

4. **测试阶段** (Line ~2000)
   - 使用测试集进行预测
   - 每日重新平衡（Top 20）
   - 计算 IC, Rank IC, Win Rate, Avg Return

5. **结果输出** (Line ~2800)
   - 生成报告
   - 保存 CSV 和图表
   - 输出指标到 stdout

## 输入格式详细说明

### 数据文件格式

**Parquet 文件结构**:
```python
# 索引
index: pd.MultiIndex(
    levels=[
        [datetime64[ns], ...],  # date level
        [str, ...]               # ticker level
    ],
    names=['date', 'ticker']
)

# 列
columns: [
    # 必需列
    'target': float64,      # T+10 收益率
    'Close': float64,       # 收盘价
    
    # T10 因子（14个）
    'liquid_momentum': float64,
    'momentum_10d': float64,
    'momentum_60d': float64,
    'obv_divergence': float64,      # ⭐ 对比因子
    'obv_momentum_60d': float64,
    'ivol_20': float64,
    'hist_vol_40d': float64,
    'atr_ratio': float64,
    'rsi_21': float64,
    'trend_r2_60': float64,
    'near_52w_high': float64,
    'vol_ratio_20d': float64,
    'price_ma60_deviation': float64,
    '5_days_reversal': float64,
    
    # 其他可能的列
    ...
]

# 数据示例
# date       ticker  target    Close  liquid_momentum  obv_divergence  ...
# 2020-01-02 AAPL    0.0234    150.0  0.0123          0.0045          ...
# 2020-01-02 MSFT    -0.0102   180.0  0.0089          -0.0023         ...
# 2020-01-03 AAPL    0.0156    152.0  0.0134          0.0056          ...
```

### 脚本调用格式

**无参数调用**（推荐）:
```bash
python scripts/compare_obv_divergence_8020_split.py
```

**自动行为**:
- ✅ 自动检测数据文件（优先子集文件）
- ✅ 自动采样 1/5 tickers
- ✅ 自动运行两个实验
- ✅ 自动对比结果
- ✅ 自动清理临时文件

## 输出格式详细说明

### JSON 对比结果格式

```json
{
  "timestamp": "2026-01-24T06:22:59.123456",
  "tickers_used": 156,
  "ticker_sample": [
    "ACEL", "ACR", "AHL", "AIT", "ALHC",
    "ALLY", "AMRZ", "AMTB", "ANF", "APTV"
  ],
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

### 指标提取逻辑

**从 stdout 提取** (`extract_metrics_from_output`, Line 282-310):

1. **IC**: 正则 `r'IC[:\s]+([-]?\d+\.?\d*)'`
2. **Rank IC**: 正则 `r'Rank[_\s]?IC[:\s]+([-]?\d+\.?\d*)'`
3. **Win Rate**: 正则 `r'Win[_\s]?Rate[:\s]+(\d+\.?\d*)%?'`
4. **Avg Return**: 正则 `r'Avg[_\s]?Return[:\s]+([-]?\d+\.?\d*)%?'`

## 关键代码位置

### 因子移除逻辑

**文件**: `bma_models/simple_25_factor_engine.py`
**修改位置**: Line 56
```python
# 原始代码
'obv_divergence',  # OBV divergence

# 临时修改后
# 'obv_divergence',  # OBV divergence - TEMPORARILY REMOVED FOR TESTING
```

**备份位置**: `bma_models/simple_25_factor_engine.py.backup_obv_8020`

### 数据过滤逻辑

**函数**: `filter_dataframe_by_tickers()` (Line 34-41)
```python
if isinstance(df.index, pd.MultiIndex):
    return df[df.index.get_level_values('ticker').isin(tickers)]
```

### 指标提取逻辑

**函数**: `extract_metrics_from_output()` (Line 282-310)
- 使用正则表达式从 stdout 提取指标
- 支持多种格式变体

## 执行状态

### ✅ 已完成

1. ✅ 创建对比脚本
2. ✅ 修复编码问题
3. ✅ 修复缩进错误
4. ✅ 添加详细错误处理
5. ✅ 移除所有 T5 因子引用
6. ✅ 统一使用 T10 因子

### ⏳ 进行中/待完成

1. ⏳ 运行完整的对比评估
2. ⏳ 分析已有结果文件
3. ⏳ 生成最终对比报告

### 📊 已有结果

**结果文件位置**: `results/obv_divergence_8020_comparison/`
- `comparison_20260124_061544.json`
- `comparison_20260124_061633.json`
- `comparison_20260124_062259.json`

**需要检查**: 这些结果文件的内容，确认实验是否成功完成。

## 下一步行动

1. **检查已有结果**
   ```bash
   # 查看最新的对比结果
   cat results/obv_divergence_8020_comparison/comparison_20260124_062259.json
   ```

2. **如果结果不完整，重新运行**
   ```bash
   python scripts/compare_obv_divergence_8020_split.py
   ```

3. **分析结果差异**
   - 对比 IC 差异
   - 对比 Rank IC 差异
   - 对比 Win Rate 差异
   - 对比 Avg Return 差异

4. **得出结论**
   - `obv_divergence` 是否显著提升性能？
   - 差异是否统计显著？
   - 是否应该保留该因子？

## 注意事项

1. **文件修改**: 脚本会临时修改因子文件，确保自动恢复
2. **数据采样**: 使用固定随机种子（42）确保可重复
3. **临时文件**: 自动清理，但异常退出时可能需要手动清理
4. **编码问题**: 已处理 Windows GBK 编码，但某些输出可能仍有问题
5. **执行时间**: 每个实验可能需要 10-30 分钟（取决于数据量）

## 故障排除

### 如果评估失败

1. **检查数据格式**
   - 验证 MultiIndex 格式
   - 检查必需列是否存在

2. **检查因子文件**
   - 确认 `obv_divergence` 在 T10_ALPHA_FACTORS 中
   - 确认文件恢复成功

3. **查看详细错误**
   - 脚本会输出 stdout/stderr 的最后 100 行
   - 检查 IndentationError 或其他语法错误

4. **验证 Python 环境**
   - 确认所有依赖已安装
   - 确认路径正确
