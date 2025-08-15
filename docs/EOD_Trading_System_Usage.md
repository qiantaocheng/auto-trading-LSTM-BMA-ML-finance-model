# EOD Trading System Usage Guide

## 概述

EOD (End-of-Day) 交易系统是为解决延迟行情环境下的交易挑战而设计的完整解决方案。系统采用"T日收盘后分析 → T+1开盘下单"的模式，避免延迟数据对实时交易的影响。

## 核心组件

### 1. 增强订单执行器 (EnhancedOrderExecutor)

**新增方法：**

- `place_open_order()` - LOO/MOO开盘单
- `place_limit_rth()` - 仅RTH限价单+自动撤单
- `update_server_stop()` - 服务器端止损管理
- `eod_update_trailing_stop()` - EOD移动止损更新

### 2. EOD调度器 (EODScheduler)

**核心任务：**

- EOD信号生成（收盘后+20分钟）
- EOD移动止损更新（收盘后+20分钟）
- 次日开盘订单执行

## 快速开始

### 1. 配置设置

```yaml
# config/eod_trading_config.yaml
strategy:
  double_trend:
    eod_mode: true
    run_after_close_min: 20
  
  execution:
    open_order_type: "LOO"        # LOO 或 MOO
    limit_band_by_atr_mult: 0.5   # 价格带宽
    cancel_if_not_filled_minutes: 30

risk_management:
  default_stop_pct: 0.01          # 1%初始止损
  atr_trailing:
    enabled: true
    multiplier: 2.0
    activate_after_R: 1.0
```

### 2. 基本用法

```python
import asyncio
from autotrader.eod_scheduler import EODConfig, create_eod_scheduler

# 创建配置
config = EODConfig(
    enabled=True,
    open_order_type="LOO",
    atr_trailing_enabled=True
)

# 创建并启动EOD调度器
scheduler = create_eod_scheduler(config)
scheduler.set_dependencies(data_feed, signal_engine, order_executor, position_manager)

# 启动EOD任务调度
await scheduler.start_eod_tasks()
```

## 订单类型详解

### 开盘单类型

#### LOO (Limit-on-Open)
```python
await executor.place_open_order(
    symbol="AAPL", 
    side="BUY", 
    quantity=100, 
    limit_price=192.30, 
    order_type="LOO"
)
```

#### MOO (Market-on-Open)
```python
await executor.place_open_order(
    symbol="AAPL", 
    side="BUY", 
    quantity=100, 
    order_type="MOO"
)
```

#### RTH限价单（30分钟自动撤单）
```python
await executor.place_limit_rth(
    symbol="AAPL", 
    side="BUY", 
    quantity=100, 
    limit_price=193.20, 
    cancel_after_min=30
)
```

## 移动止损系统

### EOD移动止损更新

```python
await executor.eod_update_trailing_stop(
    symbol="AAPL",
    side="BUY",
    entry_price=190.00,
    current_close=195.40,
    atr_value=2.10,
    initial_stop=188.10,
    activate_after_R=1.0,  # 浮盈>=1R时激活
    atr_mult=2.0           # ATR倍数
)
```

### 服务器端止损管理

```python
# 更新或新建止损单
await executor.update_server_stop("AAPL", new_stop=185.50)

# 指定数量
await executor.update_server_stop("AAPL", new_stop=185.50, quantity=100)
```

## 价格保护机制

### 价格带计算

系统使用多层价格保护：

```python
# 价格带 = min(max(ATR_mult*ATR, floor_pct*close), cap_pct*close)
band = min(
    max(0.5 * atr_value, 0.003 * reference_price),  # 最小0.3%
    0.015 * reference_price                          # 最大1.5%
)

# 限价计算
if side == 'BUY':
    limit_price = reference_price + band
else:
    limit_price = reference_price - band
```

## 工作流程

### 1. EOD信号生成（收盘后+20分钟）

1. 获取交易股票池
2. 拉取日线数据（200天）
3. 计算技术指标（DMI/ADX/MA等）
4. 生成交易信号
5. 创建次日开盘计划

### 2. EOD移动止损更新（收盘后+20分钟）

1. 遍历当前持仓
2. 计算当日ATR和收盘价
3. 检查浮盈是否达到激活阈值
4. 计算新的移动止损位
5. 更新服务器端止损单

### 3. 次日开盘执行（开盘时）

1. 处理平仓队列
2. 执行开仓计划
3. 挂设初始止损单
4. 记录入场信息

## 风控集成

### 订单级风控

```python
# 自动调用（如果有risk_engine）
if hasattr(executor, "risk_engine"):
    validation = await executor.risk_engine.validate_order(
        symbol=symbol, 
        side=action, 
        quantity=qty, 
        price=price, 
        account_value=account_value
    )
    if not validation.is_valid:
        raise RuntimeError("风控拒单")
```

### 组合级风控

- 单仓位最大占比限制
- 总敞口控制
- 单笔风险控制

## 审计日志

系统自动记录关键事件：

```python
# 订单提交
auditor.emit("order_submitted", symbol=symbol, side=action, qty=quantity)

# 止损修改
auditor.emit("stop_modified", symbol=symbol, old_stop=old, new_stop=new)

# 移动止损更新
auditor.emit("trailing_stop_eod", symbol=symbol, old_stop=cur_stop, new_stop=new_stop)
```

## 最佳实践

### 1. 延迟环境优化

- 使用EOD模式避免实时信号
- 设置合理的价格保护带
- 30分钟未成交自动撤单

### 2. 风险控制

- 单笔风险不超过0.5%
- 设置1%初始止损
- 浮盈1R后启动移动止损

### 3. 执行效率

- 优先使用LOO获得更好成交价
- 避开开盘竞价如果流动性差
- 使用服务器端止损减少延迟

## 故障处理

### 常见问题

1. **订单被拒**：检查风控设置和账户余额
2. **合约确认失败**：检查股票代码和市场状态
3. **止损更新失败**：检查持仓状态和订单格式

### 监控指标

- 订单成功率
- 平均执行时间
- 滑点统计
- 风控拒单率

## 扩展开发

### 自定义信号引擎

```python
class MySignalEngine:
    async def on_daily_bar_close(self, symbol, bars):
        # 实现自定义信号逻辑
        if signal_condition:
            return {
                'action': 'BUY',
                'confidence': 0.8,
                'reason': 'Custom signal'
            }
        return None
```

### 自定义数据源

```python
class MyDataFeed:
    async def fetch_daily_bars(self, symbol, lookback):
        # 实现自定义数据获取
        return pd.DataFrame(...)
```

## 性能优化

- 合约缓存减少重复确认
- 异步任务并发执行
- 批量处理多个标的
- 智能任务生命周期管理

## 支持与反馈

如有问题或建议，请通过以下方式联系：

- 查看日志文件获取详细错误信息
- 检查配置文件格式和参数设置
- 确认依赖组件正确初始化

---

*本文档基于EOD Trading System v1.0编写*