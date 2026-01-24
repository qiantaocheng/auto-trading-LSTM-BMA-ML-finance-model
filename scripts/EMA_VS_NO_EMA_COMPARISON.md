# EMA vs 无EMA 结果对比指南

## 📊 对比方法

### 方法1：使用对比脚本（推荐）

运行对比脚本，自动运行两次评估并生成对比报告：

```bash
python scripts/compare_ema_vs_no_ema.py \
  --ema-top-n 300 \
  --models catboost lambdarank ridge_stacking \
  --base-args \
    --horizon-days 10 \
    --top-n 20 \
    --cost-bps 10 \
    --split 0.9 \
    --output-dir results/ema_comparison \
    --snapshot-id <your-snapshot-id> \
    --models catboost lambdarank ridge_stacking
```

**输出：**
- `results/ema_comparison_with_ema/run_*/` - 使用EMA的结果
- `results/ema_comparison_no_ema/run_*/` - 不使用EMA的结果
- `results/ema_comparison/ema_comparison_report.txt` - 对比报告
- `results/ema_comparison/ema_comparison_report.json` - 对比数据（JSON）

### 方法2：手动运行两次

#### 1. 运行使用EMA的版本

```bash
python scripts/time_split_80_20_oos_eval.py \
  --horizon-days 10 \
  --top-n 20 \
  --cost-bps 10 \
  --split 0.9 \
  --output-dir results/with_ema \
  --snapshot-id <snapshot-id> \
  --models catboost lambdarank ridge_stacking \
  --ema-top-n 300 \
  --ema-min-days 3
```

#### 2. 运行不使用EMA的版本

```bash
python scripts/time_split_80_20_oos_eval.py \
  --horizon-days 10 \
  --top-n 20 \
  --cost-bps 10 \
  --split 0.9 \
  --output-dir results/no_ema \
  --snapshot-id <snapshot-id> \
  --models catboost lambdarank ridge_stacking \
  --ema-top-n -1
```

**注意：** `--ema-top-n -1` 表示完全禁用EMA

#### 3. 手动对比结果

对比两个输出目录中的：
- `complete_metrics_report.txt`
- `{model}_bucket_returns.csv`
- `{model}_top5_15_rebalance10d_accumulated.csv`

## 📈 对比指标

### Overlap指标（每日观测）

| 指标 | 说明 | 改善方向 |
|------|------|---------|
| 平均收益 | 每日收益的平均值 | ↑ 越大越好 |
| 中位数收益 | 每日收益的中位数 | ↑ 越大越好 |
| 标准差 | 收益的波动性 | ↓ 越小越好 |
| 胜率 | 正收益日的比例 | ↑ 越大越好 |
| Sharpe Ratio | 风险调整后收益 | ↑ 越大越好 |

### Non-Overlap指标（10天期间）

| 指标 | 说明 | 改善方向 |
|------|------|---------|
| 平均期间收益 | 每期收益的平均值 | ↑ 越大越好 |
| 中位数期间收益 | 每期收益的中位数 | ↑ 越大越好 |
| 标准差 | 期间收益的波动性 | ↓ 越小越好 |
| 胜率 | 正收益期的比例 | ↑ 越大越好 |
| 累积收益 | 总累积收益 | ↑ 越大越好 |
| 最大回撤 | 最大回撤幅度 | ↓ 越小越好（绝对值） |
| 年化收益 | 年化收益率 | ↑ 越大越好 |
| Sharpe Ratio | 风险调整后收益 | ↑ 越大越好 |

## 📝 对比报告示例

```
================================================================================
EMA vs 无EMA 结果对比报告
================================================================================
生成时间: 2026-01-20 12:00:00

================================================================================
【CATBOOST】
================================================================================

【Overlap指标对比（每日观测）】
--------------------------------------------------------------------------------
指标                   使用EMA        不使用EMA           差异      改善
--------------------------------------------------------------------------------
平均收益 (%)           0.1234         0.1156         +0.0078        ✅
中位数收益 (%)         0.0987         0.0923         +0.0064        ✅
标准差 (%)             0.3456         0.3521         -0.0065        ✅
胜率 (%)               52.34          51.23          +1.11          ✅
Sharpe Ratio           0.5678         0.5234         +0.0444        ✅

【Non-Overlap指标对比（10天期间）】
--------------------------------------------------------------------------------
指标                      使用EMA        不使用EMA           差异      改善
--------------------------------------------------------------------------------
平均期间收益 (%)           1.2345         1.1567         +0.0778        ✅
中位数期间收益 (%)         0.9876         0.9234         +0.0642        ✅
标准差 (%)                 3.4567         3.5212         -0.0645        ✅
胜率 (%)                   56.00          54.00          +2.00          ✅
累积收益 (%)              12.3456        11.5678         +0.7778        ✅
最大回撤 (%)              -5.6789        -6.1234         +0.4445        ✅
年化收益 (%)              15.6789        14.5678         +1.1111        ✅
Sharpe Ratio               0.7123         0.6567         +0.0556        ✅
```

## 🎯 预期结果

### EMA的优势

1. **更稳定的排名**：
   - 减少单日异常导致的排名变化
   - 提高选股的稳定性

2. **更好的风险调整收益**：
   - 可能提高Sharpe Ratio
   - 可能降低波动性（标准差）

3. **更平滑的收益曲线**：
   - 减少极端收益
   - 更稳定的累积收益

### 无EMA的优势

1. **更快的响应**：
   - 对市场变化反应更快
   - 可能捕捉短期机会

2. **更真实的预测**：
   - 使用原始模型预测
   - 不引入额外的平滑

## 📊 分析建议

### 1. 关注关键指标

- **Sharpe Ratio**：最重要的风险调整收益指标
- **最大回撤**：风险控制的关键指标
- **累积收益**：最终收益表现

### 2. 分模型分析

不同模型可能对EMA的响应不同：
- catboost：可能受益于EMA平滑
- lambdarank：可能已经比较稳定，EMA影响较小
- ridge_stacking：作为集成模型，EMA影响可能更明显

### 3. 分时期分析

检查不同市场环境下的表现：
- 牛市：EMA可能平滑收益
- 熊市：EMA可能减少损失
- 震荡市：EMA可能提高稳定性

## 🔧 快速对比命令

```bash
# 使用对比脚本（最简单）
python scripts/compare_ema_vs_no_ema.py \
  --ema-top-n 300 \
  --models catboost lambdarank \
  --base-args --horizon-days 10 --top-n 20 --cost-bps 10 --split 0.9 \
    --output-dir results/ema_comparison --snapshot-id <snapshot-id> \
    --models catboost lambdarank
```

## 📚 相关文档

- `scripts/EWMA_IMPLEMENTATION_SUMMARY.md` - EMA实现说明
- `scripts/TOP300_EMA_STRATEGY.md` - Top300 EMA策略
- `scripts/EMA_COMPUTATION_REDUCTION_ANALYSIS.md` - 运算量分析
