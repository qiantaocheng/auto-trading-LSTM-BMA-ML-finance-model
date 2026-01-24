# 独立股票回测模式说明

## 概述

系统现在采用**独立股票回测模式**：每个股票独立运行回测，每个股票拥有自己独立的资金池（portfolio），没有持仓上限。

## 工作原理

### 1. 独立回测循环

```python
for ticker in all_tickers:
    # 每个ticker独立运行回测
    universe = {ticker: data[ticker]}  # 只包含这一个股票
    result = engine.run(universe, ...)  # 独立回测
    all_results.append(result)
```

### 2. 独立资金池

- **每个股票**：拥有独立的 `initial_capital`（例如 $100,000）
- **没有持仓上限**：每个股票只持有自己，所以 `max_positions=1`
- **独立计算**：每个股票的回测结果完全独立，互不影响

### 3. 结果聚合

所有股票回测完成后，系统会：

1. **收集所有交易**：从所有股票的回测结果中收集所有交易
2. **计算平均收益率**：计算所有交易的平均收益率
3. **统计胜率**：计算所有交易的胜率
4. **汇总退出原因**：统计所有交易的退出原因分布

## 关键特性

### ✅ 独立资金池

每个股票都有自己独立的资金，互不干扰：

```
Stock A: $100,000 独立资金池
Stock B: $100,000 独立资金池
Stock C: $100,000 独立资金池
...
```

### ✅ 无持仓上限

- 每个股票只持有自己（`max_positions=1`）
- 没有跨股票的持仓限制
- 每个股票可以随时买入/卖出自己

### ✅ 独立信号生成

每个股票独立生成信号：

- 信号基于该股票自己的因子数据
- 信号基于该股票自己的价格数据
- 不受其他股票影响

### ✅ 独立退出逻辑

每个股票独立执行退出逻辑：

- 止损、止盈、时间限制等规则独立应用
- 场景转换（FAILED → SWING → ZOMBIE）独立进行
- 退出原因独立记录

## 输出结果

### 主要指标

```
Total Stocks Tested: 3921
Successful Runs: 3500
Failed Runs: 421
Total Trades Across All Stocks: 15000

Average Return Per Trade (All Stocks): 2.35%
Win Rate (All Stocks): 58.5%
```

### 退出原因统计

```
Exit Reasons Breakdown (All Stocks):
  scenario_B_stop_loss: 4500 (30.0%)
  scenario_B_time_limit_6weeks: 3000 (20.0%)
  scenario_E_zombie_180days: 2500 (16.7%)
  ...
```

## 使用示例

### 运行所有股票

```bash
python run_backtest_with_factors.py \
    --capital 100000 \
    --min-score 0.50 \
    --start-date 2021-01-19 \
    --end-date 2024-12-31
```

### 运行指定股票

```bash
python run_backtest_with_factors.py \
    --tickers AAPL MSFT GOOGL \
    --capital 100000 \
    --min-score 0.50 \
    --start-date 2021-01-19 \
    --end-date 2024-12-31
```

## 与之前模式的区别

### 之前模式（组合回测）

- 所有股票共享一个资金池
- 有 `max_positions` 限制（例如最多同时持有10个股票）
- 信号竞争：只有前N个信号会被执行
- 资金分配：资金在多个股票之间分配

### 现在模式（独立回测）

- 每个股票独立资金池
- 无持仓上限（每个股票只持有自己）
- 信号独立：每个股票独立生成和执行信号
- 资金独立：每个股票使用自己的全部资金

## 优势

1. **更真实的测试**：每个股票独立测试，更接近实际交易场景
2. **无资金竞争**：不会因为资金限制而错过信号
3. **更清晰的统计**：可以清楚地看到每个股票的表现
4. **更公平的比较**：所有股票使用相同的资金规模

## 注意事项

1. **计算时间**：独立回测需要更多时间（每个股票都要运行一次）
2. **内存使用**：需要加载所有股票的数据到内存
3. **结果解释**：平均收益率是所有股票所有交易的平均，不是组合收益率

## 技术实现

### 关键代码修改

1. **`run_backtest_with_factors()`**：
   - 循环每个ticker
   - 为每个ticker创建独立的universe
   - 独立运行回测

2. **`BacktestConfig`**：
   - `max_positions=1`（每个股票只持有自己）

3. **结果聚合**：
   - 收集所有交易的 `pnl_pct`
   - 计算平均值和胜率
   - 汇总退出原因

## 总结

独立股票回测模式提供了更真实、更公平的回测环境，每个股票都有独立的机会展示其表现，最终通过平均收益率来评估整体策略的有效性。
