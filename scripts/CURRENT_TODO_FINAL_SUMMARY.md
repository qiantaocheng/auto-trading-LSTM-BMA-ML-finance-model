# OBV_DIVERGENCE 80/20 对比评估 - 最终任务总结

## 📋 任务概述

**目标**: 使用 80/20 时间分割评估对比 `obv_divergence` 因子的影响，使用 1/5 股票子集

**当前状态**: 
- ✅ 脚本已创建并修复
- ✅ 指标提取逻辑已改进（支持从报告文件读取）
- ⏳ 等待重新运行获取完整结果

## 📁 文件位置和行号

### 1. 主对比脚本
**文件**: `scripts/compare_obv_divergence_8020_split.py` (480 行)

| 功能 | 行号 | 说明 |
|------|------|------|
| 采样 tickers | 16-32 | 随机采样 1/5 (20%) |
| 过滤数据 | 34-41 | MultiIndex 过滤 |
| 实验1 (WITH obv) | 43-138 | 包含 obv_divergence |
| 实验2 (WITHOUT obv) | 140-280 | 临时移除 obv_divergence |
| 指标提取 | 282-330 | 从 stdout 和报告文件提取 |
| 结果对比 | 312-380 | 生成对比报告 |
| 主函数 | 382-480 | 执行流程 |

### 2. 评估脚本
**文件**: `scripts/time_split_80_20_oos_eval.py` (2984 行)

| 功能 | 行号 | 说明 |
|------|------|------|
| 参数解析 | 340-367 | 命令行参数 |
| IC 计算 | 105-217 | Newey-West HAC |
| Rank IC 计算 | 105-217 | Spearman correlation |
| IC 输出 | 2187-2188 | `IC: 0.0234 (t-stat=...)` |
| Rank IC 输出 | 2187-2188 | `Rank IC: 0.0312 (t-stat=...)` |
| Win Rate 计算 | 2138 | 从 non-overlapping 回测 |
| Avg Return 输出 | 1242 | `OOS Top20 avg return: 0.45%` |
| 报告文件 | ~2600 | `report_df.csv` |

### 3. 因子引擎
**文件**: `bma_models/simple_25_factor_engine.py`

| 位置 | 行号 | 说明 |
|------|------|------|
| T10 因子定义 | 52-68 | 包含 obv_divergence |
| obv_divergence | 56 | ⭐ 对比因子 |
| 因子计算 | 1334-1357 | `_compute_volume_factors` |

### 4. 主模型
**文件**: `bma_models/量化模型_bma_ultra_enhanced.py`

| 位置 | 行号 | 说明 |
|------|------|------|
| compulsory_features | 3245-3250 | T10 因子列表 |
| 因子选择 | 3239-3301 | 始终使用 T+10 |

## 🔄 详细执行流程

### 阶段 1: 初始化
```python
# main() Line 382-400
1. 检测数据文件
   - 优先: subset_1_5_tickers.parquet
   - 备用: polygon_factors_all_filtered_clean_final_v2.parquet
2. 创建输出目录
   - results/obv_divergence_8020_comparison/
```

### 阶段 2: 数据采样
```python
# main() Line 402-410
1. 加载数据: pd.read_parquet(data_file)
2. 采样 tickers: sample_tickers(df, fraction=0.2, random_seed=42)
   - 从 784 tickers → 156 tickers (1/5)
3. 显示采样信息
```

### 阶段 3: 实验1 - WITH obv_divergence
```python
# run_8020_eval_with_obv_divergence() Line 43-138
1. 加载数据: pd.read_parquet(data_file)  # (827900, 21)
2. 过滤 tickers: filter_dataframe_by_tickers(df, tickers)  # (166346, 21)
3. 验证格式:
   - MultiIndex: ['date', 'ticker']
   - Unique dates: 1244
   - Unique tickers: 156
4. 保存临时文件: temp_data_with_obv.parquet (~25 MB)
5. 运行评估:
   python scripts/time_split_80_20_oos_eval.py \
     --data-file temp_data_with_obv.parquet \
     --horizon-days 10 \
     --split 0.8 \
     --top-n 20 \
     --log-level INFO \
     --output-dir run_with_obv_YYYYMMDD_HHMMSS
6. 提取指标:
   - 从 stdout: extract_metrics_from_output()
   - 从报告文件: report_df.csv (如果 stdout 失败)
7. 清理临时文件
```

### 阶段 4: 实验2 - WITHOUT obv_divergence
```python
# run_8020_eval_without_obv_divergence() Line 140-280
1. 备份因子文件: simple_25_factor_engine.py → .backup_obv_8020
2. 修改因子文件: 注释 Line 56 的 'obv_divergence'
3. 加载并过滤数据（同实验1）
4. 移除 obv_divergence 列（如果存在）
5. 保存临时文件: temp_data_without_obv.parquet
6. 运行评估（相同参数，不同输出目录）
7. 恢复因子文件（从备份）
8. 提取指标（同实验1）
9. 清理临时文件
```

### 阶段 5: 结果对比
```python
# compare_results() Line 312-380
1. 提取指标（优先从文件，否则从 stdout）
2. 计算差异:
   - ic_diff = with_ic - without_ic
   - rank_ic_diff = with_rank_ic - without_rank_ic
   - win_rate_diff = with_win_rate - without_win_rate
   - avg_return_diff = with_avg_return - without_avg_return
3. 保存 JSON: comparison_YYYYMMDD_HHMMSS.json
```

## 📥 输入格式

### 数据文件
**格式**: Parquet
**索引**: MultiIndex(date, ticker)
**必需列**:
```python
必需列 = [
    'target',              # T+10 收益率 (float64)
    'Close',               # 收盘价 (float64)
    # T10 因子 (14个)
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

### 脚本调用
```bash
# 无参数调用（推荐）
python scripts\compare_obv_divergence_8020_split.py
```

## 📤 输出格式

### JSON 结果
```json
{
  "timestamp": "2026-01-24T06:22:59.380782",
  "tickers_used": 156,
  "ticker_sample": ["ACEL", "ACR", ...],
  "with_obv_divergence": {
    "success": true,
    "elapsed_time_minutes": 2.03,
    "metrics": {
      "ic": 0.0234,
      "rank_ic": 0.0312,
      "win_rate": 52.5,
      "avg_return": 0.45
    },
    "output_dir": "results/.../run_with_obv_..."
  },
  "without_obv_divergence": {
    "success": true,
    "elapsed_time_minutes": 1.99,
    "metrics": {
      "ic": 0.0190,
      "rank_ic": 0.0280,
      "win_rate": 51.2,
      "avg_return": 0.42
    },
    "output_dir": "results/.../run_without_obv_..."
  },
  "difference": {
    "ic": 0.0044,
    "rank_ic": 0.0032,
    "win_rate": 1.3,
    "avg_return": 0.03
  }
}
```

### 评估脚本输出文件
**目录**: `results/t10_time_split_80_20_final/run_YYYYMMDD_HHMMSS/`
**文件**:
- `report_df.csv` - 包含所有模型的指标（IC, Rank IC, win_rate, avg_top_return）
- `ridge_top20_timeseries.csv` - Top 20 时间序列
- `complete_metrics_report.txt` - 完整指标报告

## 🔍 指标提取逻辑

### 从 stdout 提取
**格式**:
- IC: `IC: 0.0234 (t-stat=2.34, SE=0.001234)` (Line 2187)
- Rank IC: `Rank IC: 0.0312 (t-stat=3.12, SE=0.001456)` (Line 2188)
- Avg Return: `[ridge_stacking] OOS Top20 avg return gross (mean, %): 0.450000` (Line 1242)

**正则表达式**:
```python
ic_match = re.search(r'IC:\s+([-]?\d+\.?\d*)\s*\(', stdout)
rank_ic_match = re.search(r'Rank\s+IC:\s+([-]?\d+\.?\d*)\s*\(', stdout, re.IGNORECASE)
avg_return_match = re.search(r'OOS\s+Top\d+\s+avg\s+return\s+gross.*?:\s*([-]?\d+\.?\d*)', stdout, re.IGNORECASE)
```

### 从报告文件提取（备用）
**文件**: `report_df.csv`
**列名**:
- `IC` - 信息系数
- `Rank_IC` - 排序信息系数
- `win_rate` - 胜率（小数形式，需转换为百分比）
- `avg_top_return` - 平均收益率（小数形式，需转换为百分比）

**读取逻辑**:
```python
report_df = pd.read_csv(report_file)
ridge_row = report_df[report_df['Model'] == 'ridge_stacking'].iloc[0]
metrics['ic'] = float(ridge_row['IC'])
metrics['rank_ic'] = float(ridge_row['Rank_IC'])
metrics['win_rate'] = float(ridge_row['win_rate']) * 100.0  # 转换为百分比
metrics['avg_return'] = float(ridge_row['avg_top_return']) * 100.0  # 转换为百分比
```

## ✅ 已完成

1. ✅ 创建对比脚本
2. ✅ 修复编码问题
3. ✅ 修复缩进错误
4. ✅ 移除所有 T5 因子引用
5. ✅ 统一使用 T10 因子
6. ✅ 改进指标提取逻辑（支持从报告文件读取）

## ⏳ 待完成

1. ⏳ 重新运行对比评估（使用改进后的指标提取）
2. ⏳ 分析完整结果
3. ⏳ 评估 obv_divergence 的实际影响

## 🚀 下一步

运行改进后的脚本：
```bash
python scripts\compare_obv_divergence_8020_split.py
```

脚本将：
1. 自动采样 1/5 tickers
2. 运行两个实验
3. 从 stdout 和报告文件提取指标
4. 生成完整的对比结果
