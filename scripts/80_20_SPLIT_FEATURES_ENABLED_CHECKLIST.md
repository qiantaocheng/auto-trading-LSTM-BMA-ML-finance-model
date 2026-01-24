# 80/20 Split 功能启用检查清单

## ✅ 核心功能启用状态

### 1. 时间分割 ✅
- **位置**: `main()` line 1494-1506
- **功能**: 按时间顺序分割数据（80/20）
- **Purge Gap**: ✅ 已启用（避免标签泄露）
- **状态**: ✅ **已启用**

### 2. 模型训练 ✅
- **位置**: `main()` line 1540-1560
- **功能**: 在训练集上训练所有模型
- **支持模型**: ElasticNet, XGBoost, CatBoost, LambdaRank, RidgeStacker
- **Snapshot保存**: ✅ `snapshot_id.txt` (line 1640)
- **状态**: ✅ **已启用**

### 3. 样本外预测 ✅
- **位置**: `main()` line 1600-2000
- **功能**: 在测试集上进行逐日预测
- **特征对齐**: ✅ `align_test_features_with_model()` (line 48-103)
- **EMA平滑**: ✅ `apply_ema_smoothing()` (line 771-835)
- **多模型支持**: ✅ 支持同时评估多个模型
- **状态**: ✅ **已启用**

### 4. IC和Rank IC计算（HAC修正）✅
- **位置**: 
  - `calculate_newey_west_hac_ic()` (line 105-219)
  - `calculate_hansen_hodrick_se_ic()` (line 220-339)
  - `main()` line 2026-2058
- **功能**: 计算IC和Rank IC，带HAC标准误修正
- **方法**: Newey-West（默认）或Hansen-Hodrick
- **输出**: `report_df.csv` 包含IC, Rank_IC, IC_tstat, IC_se_hac等
- **状态**: ✅ **已启用**

### 5. 回测指标计算 ✅

#### 5.1 Daily收益（Overlap）✅
- **位置**: `calculate_group_returns_standalone()` (line 608-692)
- **功能**: 每日计算平均/中位数收益
- **输出**: `report_df.csv` 中的avg_top_return, median_top_return等
- **状态**: ✅ **已启用**

#### 5.2 Non-Overlapping收益 ✅
- **位置**: `calculate_group_returns_hold10d_nonoverlap()` (line 693-770)
- **功能**: 每10天再平衡一次，计算持有期收益
- **输出**: `{model}_top30_nonoverlap_timeseries.csv`
- **状态**: ✅ **已启用**

#### 5.3 分桶收益 ✅
- **位置**: 
  - `calculate_bucket_returns_standalone()` (line 947-1023) - Daily
  - `calculate_bucket_returns_hold_horizon_nonoverlap()` (line 1024-1124) - Non-overlapping
- **功能**: 计算Top/Bottom buckets收益
- **Buckets**: Top 1-10, 5-15, 11-20, 21-30; Bottom 1-10, 11-20, 21-30
- **输出**: `{model}_bucket_returns.csv`, `{model}_bucket_summary.csv`
- **状态**: ✅ **已启用**

#### 5.4 Top 5-15累计收益 ✅
- **位置**: `calc_top10_accumulated_10d_rebalance()` (line 439-607)
- **功能**: 计算Top 5-15的10天再平衡累计收益
- **输出**: `{model}_top5_15_rebalance10d_accumulated.csv`
- **状态**: ✅ **已启用**

### 6. 基准对比 ✅
- **位置**: 
  - `_compute_benchmark_tplus_from_yfinance()` (line 368-416)
  - `_write_model_topn_vs_benchmark()` (line 1125-1196)
- **功能**: 与基准（QQQ）进行收益对比
- **输出**: 
  - `{model}_top20_timeseries.csv`
  - `{model}_top20_vs_qqq.png`
  - `{model}_top20_vs_qqq_cumulative.png`
- **状态**: ✅ **已启用**

### 7. 可视化输出 ✅

#### 7.1 Top N vs基准图 ✅
- **位置**: `_write_model_topn_vs_benchmark()` (line 1125-1196)
- **输出**: 
  - `{model}_top{top_n}_vs_{bench}.png` (line 1222)
  - `{model}_top{top_n}_vs_{bench}_cumulative.png` (line 1237)
- **状态**: ✅ **已启用**

#### 7.2 分桶收益图 ✅
- **位置**: `_plot_bucket_returns()` (line 2829-2941)
- **输出**: 
  - `{model}_bucket_returns_period.png` (line 2885)
  - `{model}_bucket_returns_cumulative.png` (line 2932)
- **状态**: ✅ **已启用**

#### 7.3 Top 5-15累计收益图 ✅
- **位置**: `calc_top10_accumulated_10d_rebalance()` (line 439-607)
- **输出**: `{model}_top5_15_rebalance10d_accumulated.png` (line 601)
- **状态**: ✅ **已启用**

### 8. 报告生成 ✅

#### 8.1 核心报告 ✅
- **`report_df.csv`**: ✅ line 2169
  - 包含所有模型的IC, Rank IC, MSE, MAE, R2
  - HAC修正的统计量
  - 平均/中位数收益
  - Sharpe, Win Rate
  
- **`results_summary_for_word_doc.json`**: ✅ line 2652-2654
  - JSON格式的完整结果摘要
  - 包含所有指标和元数据
  - HAC修正信息
  
- **`complete_metrics_report.txt`**: ✅ line 2660-2666
  - 文本格式的完整指标报告
  - Overlap和Non-Overlap指标
  - 累计收益、回撤、年化收益

#### 8.2 OOS指标 ✅
- **`oos_metrics.csv`**: ✅ line 2321
- **`oos_metrics.json`**: ✅ line 2320
- **`oos_topn_vs_benchmark_all_models.csv`**: ✅ line 2318

#### 8.3 Snapshot ID ✅
- **`snapshot_id.txt`**: ✅ line 1640

### 9. 交易成本 ✅
- **位置**: 所有收益计算函数都支持`cost_bps`参数
- **功能**: 在回测中考虑交易成本
- **计算**: `turnover * cost_bps / 1e4`
- **输出**: 净收益（net return）列
- **状态**: ✅ **已启用**

### 10. EMA平滑 ✅
- **位置**: `apply_ema_smoothing()` (line 771-835)
- **功能**: 对预测分数应用EWMA平滑
- **参数**: `--ema-top-n`, `--ema-min-days`
- **状态**: ✅ **已启用**（默认禁用，可通过参数启用）

---

## 📋 输出文件清单

### 核心文件（每个运行）
1. ✅ `snapshot_id.txt` - Snapshot ID
2. ✅ `report_df.csv` - 核心报告
3. ✅ `results_summary_for_word_doc.json` - JSON摘要
4. ✅ `complete_metrics_report.txt` - 完整指标报告
5. ✅ `oos_metrics.csv` - OOS指标（CSV）
6. ✅ `oos_metrics.json` - OOS指标（JSON）
7. ✅ `oos_topn_vs_benchmark_all_models.csv` - 所有模型OOS Top N vs基准

### 每个模型的CSV文件
1. ✅ `{model}_top20_timeseries.csv` - Top 20时间序列
2. ✅ `{model}_top30_nonoverlap_timeseries.csv` - Top 30非重叠时间序列
3. ✅ `{model}_top5_15_rebalance10d_accumulated.csv` - Top 5-15累计收益
4. ✅ `{model}_bucket_returns.csv` - 分桶收益数据
5. ✅ `{model}_bucket_summary.csv` - 分桶摘要

### 每个模型的PNG文件
1. ✅ `{model}_top20_vs_qqq.png` - Top 20 vs QQQ对比图
2. ✅ `{model}_top20_vs_qqq_cumulative.png` - 累计收益对比图
3. ✅ `{model}_bucket_returns_period.png` - 分桶收益期间图
4. ✅ `{model}_bucket_returns_cumulative.png` - 分桶累计收益图
5. ✅ `{model}_top5_15_rebalance10d_accumulated.png` - Top 5-15累计收益图

---

## 🔍 验证方法

### 使用验证脚本

```bash
python scripts/verify_80_20_split_outputs.py --run-dir "results/t10_time_split_80_20_final/run_YYYYMMDD_HHMMSS"
```

### 手动检查清单

1. **检查核心文件**:
   ```bash
   ls results/t10_time_split_80_20_final/run_*/snapshot_id.txt
   ls results/t10_time_split_80_20_final/run_*/report_df.csv
   ls results/t10_time_split_80_20_final/run_*/results_summary_for_word_doc.json
   ```

2. **检查模型文件**:
   ```bash
   ls results/t10_time_split_80_20_final/run_*/*_top20_timeseries.csv
   ls results/t10_time_split_80_20_final/run_*/*_bucket_returns.csv
   ls results/t10_time_split_80_20_final/run_*/*.png
   ```

3. **验证文件内容**:
   ```python
   import pandas as pd
   import json
   
   # 检查report_df.csv
   df = pd.read_csv("report_df.csv")
   print(df.columns)
   print(df[['Model', 'IC', 'Rank_IC', 'IC_tstat', 'IC_se_hac']])
   
   # 检查JSON摘要
   with open("results_summary_for_word_doc.json") as f:
       summary = json.load(f)
   print(summary.keys())
   print(summary['metadata'])
   ```

---

## ⚠️ 常见问题排查

### 1. 文件缺失
- **原因**: 模型预测失败或数据为空
- **解决**: 检查日志，确保模型预测成功

### 2. PNG文件缺失
- **原因**: matplotlib后端问题或保存失败
- **解决**: 确保使用`matplotlib.use("Agg")`（已在代码中设置）

### 3. HAC统计量缺失
- **原因**: HAC计算失败或数据不足
- **解决**: 检查是否有足够的日度IC（需要≥10个）

### 4. Top 5-15累计收益文件缺失
- **原因**: `calc_top10_accumulated_10d_rebalance()`失败
- **解决**: 检查fallback逻辑（使用bucket returns）

---

## ✅ 总结

**所有功能都已正确启用并配置**:
- ✅ 时间分割（80/20，带Purge Gap）
- ✅ 模型训练和预测
- ✅ IC/Rank IC计算（HAC修正）
- ✅ 回测指标（Daily和Non-Overlapping）
- ✅ 分桶收益计算
- ✅ 基准对比
- ✅ 可视化输出
- ✅ 报告生成
- ✅ 交易成本支持
- ✅ EMA平滑支持

**所有输出文件都能正确生成**:
- ✅ 7个核心文件
- ✅ 每个模型5个CSV文件
- ✅ 每个模型5个PNG文件

**验证方法**:
- ✅ 使用`verify_80_20_split_outputs.py`脚本
- ✅ 手动检查文件列表
- ✅ 验证文件内容

---

**生成时间**: 2026-01-22  
**状态**: ✅ **所有功能已启用并验证**
