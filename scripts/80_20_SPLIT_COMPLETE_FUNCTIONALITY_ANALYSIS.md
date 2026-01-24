# 80/20 Time Split OOS评估完整功能分析

## 📊 概述

`time_split_80_20_oos_eval.py` 是一个完整的模型训练和样本外（Out-of-Sample, OOS）评估脚本，用于量化交易模型的回测和性能评估。

---

## 🎯 核心功能

### 1. 时间分割（Time Split）

**功能**: 按时间顺序将数据分割为训练集和测试集

**实现逻辑**:
- 按唯一日期排序
- 使用`--split`参数（默认0.9，即90/10，但可设置为0.8实现80/20）
- **Purge Gap**: 在训练集和测试集之间留出`horizon_days`天的间隔，避免标签泄露
  - 因为target在日期t使用前向收益到t+horizon_days
  - 训练集结束日期 = `split_idx - 1 - horizon`

**代码位置**: `main()` line 1494-1506

```python
split_idx = int(n_dates * split)
train_end_idx = max(0, split_idx - 1 - horizon)  # Purge gap
train_start = dates[0]
train_end = dates[train_end_idx]
test_start = dates[split_idx]
test_end = dates[-1]
```

---

### 2. 模型训练

**功能**: 在训练集上训练所有模型

**支持的模型**:
- `elastic_net` - ElasticNet回归
- `xgboost` - XGBoost回归
- `catboost` - CatBoost回归
- `lambdarank` - LambdaRank排序模型
- `ridge_stacking` - Ridge堆叠模型（MetaRankerStacker）

**训练流程**:
1. 初始化`UltraEnhancedQuantitativeModel`
2. 调用`train_from_document()`在训练窗口上训练
3. 生成snapshot ID用于后续预测

**代码位置**: `main()` line 1540-1560

**可选功能**:
- `--snapshot-id`: 使用已有snapshot，跳过训练
- `--ridge-base-cols`: 覆盖RidgeStacker的base_cols配置

---

### 3. 样本外预测（OOS Prediction）

**功能**: 在测试集上进行逐日预测

**预测流程**:
1. 加载训练好的模型（从snapshot）
2. 对测试集的每一天：
   - 获取当天的特征数据
   - 使用模型进行预测
   - 记录预测值和实际值（target）
3. 生成预测结果DataFrame

**代码位置**: `main()` line 1600-2000

**关键特性**:
- **特征对齐**: 使用`align_test_features_with_model()`确保测试特征与训练特征一致
- **EMA平滑**: 可选应用EWMA平滑（通过`--ema-top-n`和`--ema-min-days`控制）
- **多模型支持**: 同时评估多个模型

---

### 4. 性能指标计算

#### 4.1 预测质量指标（Predictive Metrics）

**IC (Information Coefficient)**:
- **定义**: 预测值与实际值的Pearson相关系数
- **计算**: 按日期分组，计算每天的横截面correlation，然后对日度IC序列做HAC修正
- **HAC修正**: Newey-West或Hansen-Hodrick标准误

**Rank IC**:
- **定义**: 预测值与实际值的Spearman秩相关系数
- **计算**: 类似IC，但使用rank correlation

**回归指标**:
- `MSE` - 均方误差
- `MAE` - 平均绝对误差
- `R2` - R²得分

**代码位置**: 
- `calculate_newey_west_hac_ic()` (line 105-219)
- `calculate_hansen_hodrick_se_ic()` (line 220-339)
- `main()` line 2026-2058

#### 4.2 回测指标（Backtest Metrics）

**分组收益（Group Returns）**:
- **Top N**: 选择预测值最高的N只股票
- **Bottom N**: 选择预测值最低的N只股票
- **计算方式**:
  - **Daily**: 每日计算平均/中位数收益（用于预测质量评估）
  - **Non-Overlapping**: 每`horizon_days`天再平衡一次（用于回测）

**分桶收益（Bucket Returns）**:
- **Top Buckets**: Top 1-10, Top 5-15, Top 11-20, Top 21-30
- **Bottom Buckets**: Bottom 1-10, Bottom 11-20, Bottom 21-30
- **计算方式**: 每日计算（用于预测质量评估）

**累计收益（Accumulated Returns）**:
- 基于非重叠持有期的累计收益
- 每`horizon_days`天再平衡一次
- 按期复利计算

**风险指标**:
- **Sharpe Ratio**: 年化Sharpe比率
- **Max Drawdown**: 最大回撤（基于非重叠回测）
- **Win Rate**: 胜率（正收益期数占比）

**代码位置**:
- `calculate_group_returns_standalone()` (line 608-692) - Daily计算
- `calculate_group_returns_hold10d_nonoverlap()` (line 693-770) - Non-overlapping计算
- `calculate_bucket_returns_standalone()` (line 947-1023) - Bucket计算
- `calculate_bucket_returns_hold_horizon_nonoverlap()` (line 1024-1124) - Bucket non-overlapping计算

---

### 5. HAC标准误修正

**功能**: 对重叠观测（overlapping observations）进行统计推断修正

**方法**:
1. **Newey-West HAC** (默认):
   - 适用于自相关和异方差
   - Lag参数: `max(10, 2*horizon_days)`
   - 使用`statsmodels`的`cov_type='HAC'`

2. **Hansen-Hodrick SE**:
   - 适用于固定horizon的重叠观测
   - 使用horizon作为lag

**实现逻辑**:
1. 按日期分组，计算每天的IC（横截面correlation）
2. 得到日度IC序列
3. 对日度IC序列做HAC修正
4. 计算IC均值的标准误和t统计量

**代码位置**:
- `calculate_newey_west_hac_ic()` (line 105-219)
- `calculate_hansen_hodrick_se_ic()` (line 220-339)

---

### 6. 基准对比（Benchmark Comparison）

**功能**: 与基准（如QQQ）进行收益对比

**实现**:
- 使用`yfinance`获取基准数据
- 计算基准的T+`horizon_days`收益
- 与Top N组合收益对比

**代码位置**:
- `_compute_benchmark_tplus_from_yfinance()` (line 368-416)
- `_write_model_topn_vs_benchmark()` (line 1125-1196)

**输出**:
- Top N vs基准的时间序列CSV
- Top N vs基准的对比图（PNG）
- 累计收益对比图（PNG）

---

### 7. 交易成本（Transaction Costs）

**功能**: 在回测中考虑交易成本

**实现**:
- `--cost-bps`: 每次再平衡的交易成本（基点）
- 计算方式: `turnover * cost_bps / 1e4`
- 应用于净收益（net return）计算

**代码位置**: 所有收益计算函数都支持`cost_bps`参数

---

### 8. EMA平滑（Exponential Moving Average Smoothing）

**功能**: 对预测分数应用EWMA平滑

**参数**:
- `--ema-top-n`: 仅对Top N股票应用EMA（-1禁用，0全部，>0 Top N）
- `--ema-min-days`: 连续N天在Top N才应用EMA（默认3）

**公式**: `S_t = 0.6*S_t + 0.3*S_{t-1} + 0.1*S_{t-2}`

**代码位置**: `apply_ema_smoothing()` (line 771-835)

---

### 9. 可视化输出

#### 9.1 时间序列图

**Top N vs基准对比图**:
- 每期收益对比
- 累计收益对比

**代码位置**: `_write_model_topn_vs_benchmark()` (line 1125-1196)

#### 9.2 分桶收益图

**功能**: 显示不同分桶的收益表现

**包含**:
- Top buckets (1-10, 5-15, 11-20, 21-30)
- Bottom buckets (1-10, 11-20, 21-30)
- 基准收益

**代码位置**: `_plot_bucket_returns()` (line 2829-2941)

---

### 10. 报告生成

#### 10.1 核心报告文件

**`report_df.csv`**:
- 每个模型的性能指标
- IC, Rank IC, MSE, MAE, R2
- HAC修正的统计量
- 平均/中位数收益
- Sharpe, Win Rate

**`results_summary_for_word_doc.json`**:
- 完整的JSON格式结果摘要
- 包含所有指标和元数据

**`complete_metrics_report.txt`**:
- 文本格式的完整指标报告
- 包含Overlap和Non-Overlap指标

**代码位置**: 
- `main()` line 2168-2170 (report_df.csv)
- `main()` line 2652-2654 (results_summary_for_word_doc.json)
- `_generate_complete_metrics_report()` (line 2672-2827)

#### 10.2 模型特定文件

**每个模型生成**:
- `{model}_top20_timeseries.csv` - Top 20时间序列
- `{model}_top30_nonoverlap_timeseries.csv` - Top 30非重叠时间序列
- `{model}_top5_15_rebalance10d_accumulated.csv` - Top 5-15累计收益
- `{model}_bucket_returns.csv` - 分桶收益数据
- `{model}_bucket_summary.csv` - 分桶摘要
- `{model}_top20_vs_qqq.png` - Top 20 vs QQQ对比图
- `{model}_top20_vs_qqq_cumulative.png` - 累计收益对比图
- `{model}_bucket_returns_period.png` - 分桶收益期间图
- `{model}_bucket_returns_cumulative.png` - 分桶累计收益图
- `{model}_top5_15_rebalance10d_accumulated.png` - Top 5-15累计收益图

---

## 📋 命令行参数

### 数据参数

- `--data-file`: 数据文件路径（Parquet格式）
- `--train-data`: 训练数据路径（向后兼容）
- `--data-dir`: 数据目录（向后兼容）

### 时间分割参数

- `--split`: 训练集比例（默认0.9，可设置为0.8实现80/20）
- `--horizon-days`: 预测horizon天数（默认10）

### 模型参数

- `--model`: 主模型名称（用于legacy单模型图）
- `--models`: 要评估的模型列表（默认: catboost lambdarank ridge_stacking）
- `--snapshot-id`: 使用已有snapshot ID（跳过训练）
- `--ridge-base-cols`: 覆盖RidgeStacker的base_cols

### 回测参数

- `--top-n`: Top N股票数量（默认20）
- `--cost-bps`: 交易成本（基点，默认0.0）
- `--benchmark`: 基准名称（默认QQQ）
- `--max-weeks`: 最大周数限制（默认260）

### HAC参数

- `--hac-method`: HAC方法（newey-west或hansen-hodrick，默认newey-west）
- `--hac-lag`: HAC lag阶数（默认: max(10, 2*horizon_days)）

### EMA参数

- `--ema-top-n`: EMA应用的Top N（-1禁用，0全部，>0 Top N，默认-1）
- `--ema-min-days`: 连续N天在Top N才应用EMA（默认3）

### 输出参数

- `--output-dir`: 输出目录（默认results/t10_time_split_90_10）
- `--log-level`: 日志级别（默认INFO）

---

## 🔄 完整工作流程

### 阶段1: 数据加载和预处理

1. 加载Parquet数据文件（支持单文件或目录）
2. 确保MultiIndex格式（date, ticker）
3. 计算Sato因子（如果缺失）
4. 排序和标准化索引

### 阶段2: 时间分割

1. 获取唯一日期并排序
2. 计算分割点（考虑purge gap）
3. 确定训练集和测试集日期范围

### 阶段3: 模型训练（如果未提供snapshot）

1. 初始化模型
2. 在训练集上训练
3. 生成snapshot ID
4. 可选：重新拟合RidgeStacker（如果指定了--ridge-base-cols）

### 阶段4: 样本外预测

1. 加载模型（从snapshot）
2. 对测试集的每一天：
   - 获取特征数据
   - 特征对齐
   - 应用EMA平滑（如果启用）
   - 模型预测
   - 记录结果

### 阶段5: 指标计算

1. **预测质量指标**:
   - IC和Rank IC（带HAC修正）
   - MSE, MAE, R2

2. **回测指标**:
   - Daily平均/中位数收益（预测质量）
   - Non-overlapping累计收益（回测）
   - Sharpe, Win Rate, Max Drawdown

3. **分桶收益**:
   - Top/Bottom buckets
   - Daily和Non-overlapping

### 阶段6: 基准对比

1. 获取基准数据（yfinance）
2. 计算基准收益
3. 生成对比图和CSV

### 阶段7: 报告生成

1. 生成`report_df.csv`
2. 生成`results_summary_for_word_doc.json`
3. 生成`complete_metrics_report.txt`
4. 生成所有模型特定的CSV和PNG文件

---

## 📊 关键指标说明

### Overlap vs Non-Overlap

**Overlap（重叠观测）**:
- 每日再平衡
- 用于预测质量评估（IC, Rank IC）
- 需要HAC修正进行统计推断

**Non-Overlap（非重叠观测）**:
- 每`horizon_days`天再平衡一次
- 用于回测指标（累计收益、回撤、Sharpe）
- 避免重叠导致的统计偏差

### 收益计算方式

**Daily（每日）**:
- 每天选择Top N，计算当天收益
- 用于平均/中位数收益计算
- 反映预测质量

**Non-Overlapping（非重叠）**:
- 每10天选择Top N，持有10天
- 计算10天持有期收益
- 用于累计收益和风险指标

---

## 🔍 关键函数详解

### `align_test_features_with_model()`

**功能**: 确保测试特征与训练特征一致

**逻辑**:
1. 从模型获取训练特征名（`feature_names_in_`）
2. 检查缺失特征（填充0）
3. 选择并重排序特征

**代码位置**: line 48-103

### `calculate_newey_west_hac_ic()`

**功能**: 计算带Newey-West HAC修正的IC

**流程**:
1. 按日期分组
2. 计算每天的横截面correlation
3. 得到日度IC序列
4. 对日度IC序列做Newey-West HAC
5. 计算IC均值的标准误和t统计量

**代码位置**: line 105-219

### `calculate_group_returns_standalone()`

**功能**: 计算分组收益（Daily模式）

**输出**:
- `avg_top_return`: Top N平均收益
- `median_top_return`: Top N中位数收益
- `avg_top_return_net`: Top N平均净收益（扣除成本）
- `avg_top_turnover`: 平均换手率

**代码位置**: line 608-692

### `calculate_group_returns_hold10d_nonoverlap()`

**功能**: 计算非重叠持有期收益

**逻辑**:
- 每`horizon_days`天再平衡一次
- 持有`horizon_days`天
- 计算持有期收益

**输出**: 时间序列DataFrame，包含每期的收益

**代码位置**: line 693-770

### `_write_model_topn_vs_benchmark()`

**功能**: 生成Top N vs基准的对比图和CSV

**输出**:
- CSV文件：时间序列数据
- PNG文件：每期收益对比图
- PNG文件：累计收益对比图

**代码位置**: line 1125-1196

### `_generate_complete_metrics_report()`

**功能**: 生成完整指标报告

**包含**:
- Overlap指标（每日）
- Non-Overlap指标（每期）
- 累计收益、回撤、年化收益

**代码位置**: line 2672-2827

---

## 📁 输出文件结构

```
results/t10_time_split_80_20_final/run_YYYYMMDD_HHMMSS/
├── snapshot_id.txt                          # Snapshot ID
├── report_df.csv                            # 核心报告（所有模型指标）
├── results_summary_for_word_doc.json        # JSON格式结果摘要
├── complete_metrics_report.txt              # 完整指标报告（文本）
├── oos_metrics.csv                          # OOS指标（CSV）
├── oos_metrics.json                         # OOS指标（JSON）
├── oos_topn_vs_benchmark_all_models.csv     # 所有模型OOS Top N vs基准
│
├── {model}_top20_timeseries.csv              # Top 20时间序列
├── {model}_top30_nonoverlap_timeseries.csv   # Top 30非重叠时间序列
├── {model}_top5_15_rebalance10d_accumulated.csv  # Top 5-15累计收益
├── {model}_bucket_returns.csv                # 分桶收益数据
├── {model}_bucket_summary.csv                # 分桶摘要
│
├── {model}_top20_vs_qqq.png                 # Top 20 vs QQQ对比图
├── {model}_top20_vs_qqq_cumulative.png      # 累计收益对比图
├── {model}_bucket_returns_period.png        # 分桶收益期间图
├── {model}_bucket_returns_cumulative.png     # 分桶累计收益图
└── {model}_top5_15_rebalance10d_accumulated.png  # Top 5-15累计收益图
```

---

## 🎯 使用示例

### 基本用法（80/20分割）

```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
  --horizon-days 10 \
  --split 0.8 \
  --models catboost lambdarank ridge_stacking \
  --top-n 20 \
  --output-dir "results/t10_time_split_80_20_final" \
  --log-level INFO
```

### 使用已有snapshot（跳过训练）

```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
  --snapshot-id "snapshot_20260122_123456" \
  --split 0.8 \
  --models catboost lambdarank \
  --top-n 20
```

### 启用EMA平滑

```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
  --split 0.8 \
  --ema-top-n 20 \
  --ema-min-days 3 \
  --models catboost lambdarank
```

### 指定交易成本

```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
  --split 0.8 \
  --cost-bps 10.0 \
  --models catboost lambdarank
```

---

## ⚠️ 重要注意事项

### 1. Purge Gap

- 训练集和测试集之间必须留出`horizon_days`天的间隔
- 避免标签泄露（target使用未来收益）

### 2. 特征对齐

- 测试特征必须与训练特征完全一致
- 缺失特征会被填充0
- 额外特征会被忽略

### 3. HAC修正

- IC和Rank IC使用HAC修正的标准误
- 因为观测是重叠的（每日预测，但target是T+10收益）

### 4. Overlap vs Non-Overlap

- **Overlap**: 用于预测质量评估（IC, 平均收益）
- **Non-Overlap**: 用于回测指标（累计收益、回撤、Sharpe）

### 5. 数据格式要求

- 必须是MultiIndex格式（date, ticker）
- 必须包含所有需要的特征列
- 必须包含target列（用于计算IC）

---

## 🔧 技术细节

### 内存优化

- 使用`pyarrow`的内存映射读取大文件
- 支持分块处理大数据集

### 错误处理

- 特征对齐失败时回退到原始特征
- Sato因子计算失败时填充0
- 基准数据获取失败时继续执行

### 日志记录

- 详细的日志记录每个步骤
- 可配置日志级别
- 记录关键指标和警告

---

## 📈 性能指标解读

### IC和Rank IC

- **IC > 0.05**: 较强的预测能力
- **IC > 0.1**: 非常强的预测能力
- **t-stat > 2**: 统计显著

### Sharpe Ratio

- **Sharpe > 1**: 较好的风险调整收益
- **Sharpe > 2**: 优秀的风险调整收益

### Win Rate

- **Win Rate > 50%**: 正收益期数超过负收益期数
- **Win Rate > 60%**: 较强的稳定性

### Max Drawdown

- **Max DD < -20%**: 可接受的回撤
- **Max DD < -10%**: 较低的回撤

---

**生成时间**: 2026-01-22  
**脚本位置**: `scripts/time_split_80_20_oos_eval.py`  
**状态**: ✅ **完整功能分析**
