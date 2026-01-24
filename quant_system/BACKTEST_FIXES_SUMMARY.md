# 回测系统关键风险修复总结

## 修复的关键问题

### 1. 数据泄露问题（Look-Ahead Bias）✅

**问题：**
- 使用当日收盘价生成信号，然后用当日收盘价成交
- 等于用未来已知信息下单（收盘前不可能知道收盘价）

**修复：**
- **信号生成**：使用 t-1 的收盘价（`prev_close_prices`）
- **成交执行**：使用 t 的开盘价（`current_open_prices`）
- **退出触发**：使用 t-1 的收盘价检测退出条件
- **退出执行**：使用 t 的开盘价执行退出

**实现：**
```python
# 主循环中
prev_close_prices = self._get_prev_close_prices(universe_data, date)  # t-1 close
current_open_prices = self._get_current_open_prices(universe_data, date)  # t open

# 信号生成使用 t-1 数据
signals = scan_universe(universe_subset)  # subset 只包含 t-1 及之前的数据

# 成交使用 t open
entry_price = current_open_prices[symbol]

# 退出检测使用 t-1 close
if prev_close <= stop_loss_price:
    # 退出执行使用 t open
    exit_price = current_open_prices[symbol]
```

### 2. 退出规则优先级 ✅

**问题：**
- 多个退出条件可能同时满足
- 执行顺序影响结果，但不明确

**修复：**
- 明确优先级顺序：
  1. **优先级1：风险类退出**（止损、破位）
  2. **优先级2：时间类退出**（超期）
  3. **优先级3：盈利类退出**（目标、阻力）

**实现：**
```python
# 场景B示例
# PRIORITY 1: Risk exits (stop loss)
if prev_close <= stop_loss_price:
    exit(...)

# PRIORITY 2: Time-based exits
if holding_days > 42:
    exit(...)

# PRIORITY 3: Profit-based exits
if unrealized_pnl_pct >= 0.20:
    if check_resistance(...):
        exit(...)
```

### 3. 退出原因统计 ✅

**问题：**
- 不知道系统真正靠什么赚钱/亏钱
- 无法分析哪种退出规则最有效

**修复：**
- 统计每种退出原因的数量和占比
- 在最终报告中输出退出原因分布

**实现：**
```python
exit_reasons = {}
for trade in portfolio.closed_trades:
    reason = trade.exit_reason or "unknown"
    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

# 输出示例：
# Exit Reasons Breakdown:
#   scenario_B_stop_loss: 45 (32.1%)
#   scenario_B_resistance_20pct: 38 (27.1%)
#   scenario_A_7pct_stop: 25 (17.9%)
```

### 4. 场景转换的时间问题 ✅

**问题：**
- 场景转换可能使用未来信息
- 例如：持仓5天后检测波动<2%，但检测时可能用了未来5天的信息

**修复：**
- 确保场景转换只使用当前时刻（t-1）已知的信息
- 僵尸股检测：使用 t-1 收盘价计算的波动率
- 场景判断：使用 t-1 及之前的数据

**实现：**
```python
# 僵尸股检测（只使用 t-1 数据）
if holding_days >= 5 and trade.scenario != TradeScenario.ZOMBIE:
    # unrealized_pnl_pct 基于 prev_close vs entry_price
    price_change = abs(unrealized_pnl_pct)
    if price_change < 0.02:  # 只使用已知数据
        trade.scenario = TradeScenario.ZOMBIE
```

### 5. 市场环境检测 ✅

**问题：**
- 市场环境检测可能使用当日数据

**修复：**
- 市场环境检测使用 t-1 的数据
- 确保没有未来信息泄露

**实现：**
```python
def _detect_market_regime(self, benchmark_data, date):
    # 使用 t-1 数据
    bench_available_dates = benchmark_data.index[benchmark_data.index < date]
    bench_signal_date = bench_available_dates[-1]
    bench_subset = benchmark_data.loc[lookback_start:bench_signal_date]
```

## 修改的文件

1. **`backtest/engine.py`**
   - 添加 `_get_prev_close_prices()` 方法
   - 添加 `_get_current_open_prices()` 方法
   - 修改主循环使用 t-1 close 和 t open
   - 修改 `_check_exits()` 使用分离的触发和执行价格
   - 修改 `_generate_entries()` 使用 t-1 数据生成信号，t open 执行
   - 明确退出规则优先级
   - 添加退出原因统计

2. **`core/data_types.py`**
   - 在 `BacktestResult` 中添加 `exit_reasons` 字段

3. **`run_backtest_with_factors.py`**
   - 添加退出原因统计输出

## 关键改进点

### 时间线清晰化

```
t-1 收盘后：
  ├─ 生成信号（基于 t-1 close）
  ├─ 检测退出条件（基于 t-1 close）
  └─ 决定第二天（t）的操作

t 开盘：
  ├─ 执行买入（使用 t open）
  └─ 执行卖出（使用 t open）
```

### 数据使用规范

| 用途 | 使用的价格 | 说明 |
|------|-----------|------|
| 信号生成 | t-1 close | 避免未来信息 |
| 退出触发检测 | t-1 close | 避免未来信息 |
| 买入执行 | t open | 实际可执行价格 |
| 卖出执行 | t open | 实际可执行价格 |
| 权益计算 | t open | 每日估值 |

### 退出优先级

所有场景统一优先级：
1. **风险退出**（止损、破位）→ 最高优先级
2. **时间退出**（超期）→ 中等优先级
3. **盈利退出**（目标、阻力）→ 最低优先级

## 注意事项

1. **数据要求**：确保数据包含 `Open` 价格，否则回退到 `Close`
2. **价格获取**：如果当日无数据，使用最近可用日期的价格
3. **止损执行**：如果开盘价已经低于止损价，使用止损价执行（保护性止损）
4. **独立测试**：每个股票独立运行，最后统计平均收益率

## 验证建议

运行回测后检查：
1. 退出原因分布是否合理
2. 平均收益率是否在合理范围
3. 风险退出（止损）占比是否过高/过低
4. 盈利退出占比是否符合预期

## 未来改进

1. **日内执行模型**：可以添加更精细的日内执行（VWAP、TWAP）
2. **部分退出**：支持部分仓位退出（当前全部卖出）
3. **滑点模型**：可以添加更复杂的滑点模型
4. **独立回测**：每个股票独立资金池回测，最后平均（如用户要求）
