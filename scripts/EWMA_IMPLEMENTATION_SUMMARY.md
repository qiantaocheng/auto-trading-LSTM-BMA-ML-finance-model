# EWMA实现总结

## 修改内容

### 1. 删除过滤功能
- ✅ 删除了 `filter_top15_by_volatility_volume` 函数
- ✅ 删除了所有 `filter_top15` 参数
- ✅ 删除了所有 `factor_data` 参数（仅用于过滤的部分）
- ✅ 清理了所有过滤相关的调用

### 2. 添加EWMA平滑
- ✅ 启用了 `apply_ema_smoothing` 函数
- ✅ 对所有模型应用EWMA平滑（catboost, lambdarank, ridge_stacking）
- ✅ EWMA参数：3天EMA，权重 (0.6, 0.3, 0.1)
- ✅ 使用平滑后的预测进行排名

### 3. 添加Max Drawdown计算
- ✅ 在 `calc_top10_accumulated_10d_rebalance` 中添加了最大回撤计算
- ✅ 保存到CSV文件（drawdown列）
- ✅ 在日志中输出最大回撤

### 4. 默认模型更新
- ✅ 默认模型列表更新为：`["catboost", "lambdarank", "ridge_stacking"]`

## 计算的指标

### Overlap指标（每日重叠观测，249个交易日）
- 平均收益
- 中位数收益
- 标准差
- Overlap胜率
- Sharpe Ratio（年化）

### Non-Overlap指标（每10天一期，共25期）
- 平均期间收益
- 中位数期间收益
- 标准差
- Non-Overlap胜率
- 累积收益
- **最大回撤**（新增）
- 年化收益
- Sharpe Ratio（基于期间）

## 输出文件

每个模型会生成：
1. `{model}_bucket_returns.csv` - Overlap每日收益
2. `{model}_top5_15_rebalance10d_accumulated.csv` - Non-Overlap累积收益（包含drawdown列）
3. `{model}_top5_15_rebalance10d_accumulated.png` - 累积收益曲线图

## 运行命令

```bash
python scripts/time_split_80_20_oos_eval.py \
  --horizon-days 10 \
  --top-n 20 \
  --cost-bps 10 \
  --split 0.9 \
  --output-dir results/t10_time_split_90_10_ewma \
  --snapshot-id 9de0b13d-647d-4c8d-bf3d-86d3ab8a738f \
  --models catboost lambdarank ridge_stacking
```

注意：默认split已更新为0.9（90%训练，10%预测）

## 注意事项

1. EWMA平滑使用3天历史数据，前2天可能没有平滑效果
2. 所有预测都使用平滑后的分数进行排名
3. Max drawdown基于累积收益曲线计算
4. 所有指标都基于EWMA平滑后的预测计算
