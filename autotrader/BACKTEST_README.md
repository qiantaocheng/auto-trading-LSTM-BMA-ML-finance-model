# AutoTrader 回测系统

专业级美股量化策略回测框架，集成 BMA 模型、风险管理和执行引擎。

## 🎯 系统特性

### 核心功能
- **BMA 模型集成**: 复用现有的贝叶斯模型平均算法
- **多频率调仓**: 支持日频和周频策略
- **专业风控**: 集成止损止盈、仓位控制、相关性监控
- **真实成本**: 考虑手续费、滑点等交易成本
- **全面分析**: 30+ 绩效指标和专业图表

### 数据源
- 复用现有 SQLite 数据库 (`autotrader_stocks.db`)
- 自动加载股票池和历史价格数据
- 集成技术因子计算模块

## 🚀 快速开始

### 1. 基础回测
```bash
# 使用默认参数运行回测
python autotrader/run_backtest.py

# 自定义回测期间
python autotrader/run_backtest.py \
    --start-date 2022-01-01 \
    --end-date 2023-12-31 \
    --initial-capital 100000
```

### 2. 高级配置
```bash
# 周频BMA策略回测
python autotrader/run_backtest.py \
    --start-date 2022-01-01 \
    --end-date 2023-12-31 \
    --rebalance-freq weekly \
    --max-positions 20 \
    --use-bma-model \
    --model-retrain-freq 4 \
    --max-position-weight 0.15 \
    --stop-loss-pct 0.08 \
    --output-dir ./my_backtest_results
```

### 3. 预设策略对比
```bash
# 运行预设的保守/激进策略对比
python autotrader/run_backtest.py --preset
```

## 📊 回测配置

### 基础参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--start-date` | 回测开始日期 | 2022-01-01 |
| `--end-date` | 回测结束日期 | 2023-12-31 |
| `--initial-capital` | 初始资金 | 100,000 |
| `--rebalance-freq` | 调仓频率 (daily/weekly) | weekly |
| `--max-positions` | 最大持仓数量 | 20 |

### 交易成本
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--commission-rate` | 手续费率 | 0.1% |
| `--slippage-rate` | 滑点率 | 0.2% |

### BMA 模型
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--use-bma-model` | 启用BMA模型 | True |
| `--model-retrain-freq` | 重训频率(周) | 4 |
| `--prediction-horizon` | 预测周期(天) | 5 |

### 风险控制
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--max-position-weight` | 单仓最大权重 | 15% |
| `--stop-loss-pct` | 止损比例 | 8% |
| `--take-profit-pct` | 止盈比例 | 20% |

## 📈 输出结果

### 1. 控制台输出
```
=================================================
回测结果摘要
=================================================
回测期间: 2022-01-01 -> 2023-12-31
总收益率: 15.67%
年化收益率: 8.34%
年化波动率: 12.45%
夏普比率: 0.670
最大回撤: -6.78%
胜率: 54.32%
总交易次数: 156
最终资产: $115,670.00
=================================================
```

### 2. 可视化报告
生成 8 个专业图表:
- **净值曲线**: 策略资产增长轨迹
- **回撤图**: 历史回撤分析
- **月度收益热力图**: 按年月展示收益分布
- **收益分布**: 日收益率直方图和正态性检验
- **滚动夏普比率**: 60日滚动风险调整收益
- **持仓权重**: 主要持仓权重变化
- **交易统计**: 月度交易频率分析
- **绩效汇总**: 30+ 关键指标表格

### 3. Excel 报告
包含以下工作表:
- **回测摘要**: 关键指标和配置信息
- **每日绩效**: 完整的日频净值数据
- **交易记录**: 每笔买卖交易详情
- **详细指标**: 风险收益全面分析

## 🔧 系统架构

### 核心模块

#### 1. 数据管理 (`BacktestDataManager`)
- 从 SQLite 数据库加载历史价格
- 计算技术因子和特征工程
- 管理股票池和数据缓存

#### 2. 信号生成 (`BMASignalGenerator`)
- 集成 `QuantitativeModel` BMA 算法
- 滚动模型训练和预测
- 简化信号生成（BMA 不可用时）

#### 3. 组合管理 (`Portfolio`)
- 等权重 Top-N 选股策略
- 实时持仓跟踪和风险监控
- 现金和持仓价值管理

#### 4. 风险控制
- 止损止盈自动触发
- 单仓权重限制
- 集中度风险控制

#### 5. 回测引擎 (`AutoTraderBacktestEngine`)
- 时间序列主循环
- 模块协调和状态管理
- 绩效数据收集

#### 6. 分析器 (`BacktestAnalyzer`)
- 30+ 绩效指标计算
- 专业图表生成
- Excel 报告导出

## 📋 绩效指标

### 收益指标
- 总收益率、年化收益率
- 月度收益率分布
- 滚动收益分析

### 风险指标
- 年化波动率、最大回撤
- VaR (95%, 99%)
- 下行波动率

### 风险调整收益
- 夏普比率 (Sharpe Ratio)
- 索提诺比率 (Sortino Ratio)
- 卡尔马比率 (Calmar Ratio)

### 交易统计
- 胜率、盈利因子
- 平均盈利/亏损
- 最大连续盈亏次数

### 相对基准 (可选)
- Beta、Alpha
- 信息比率
- 跟踪误差

## 🔍 使用案例

### 案例 1: BMA 周频策略
```python
from autotrader.backtest_engine import AutoTraderBacktestEngine, BacktestConfig

config = BacktestConfig(
    start_date="2022-01-01",
    end_date="2023-12-31",
    initial_capital=100000.0,
    rebalance_freq="weekly",
    max_positions=20,
    use_bma_model=True,
    model_retrain_freq=4,
    max_position_weight=0.15
)

engine = AutoTraderBacktestEngine(config)
results = engine.run_backtest()
```

### 案例 2: 快速分析
```python
from autotrader.backtest_analyzer import analyze_backtest_results

# 直接分析回测结果
analyzer = analyze_backtest_results(results, "./analysis_output")
```

### 案例 3: 自定义因子
修改 `BMASignalGenerator._calculate_ml_factors()` 方法添加自定义因子:

```python
def _calculate_ml_factors(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    factor_df = df.copy()
    
    # 现有因子...
    
    # 添加自定义因子
    factor_df['custom_momentum'] = factor_df['close'].pct_change(10)
    factor_df['custom_volatility'] = factor_df['close'].pct_change().rolling(20).std()
    
    return factor_df
```

## ⚠️ 注意事项

### 数据要求
1. **数据库完整性**: 确保 `autotrader_stocks.db` 包含足够的历史数据
2. **股票池**: 系统从 `stock_lists` 表加载活跃股票
3. **价格数据**: 需要 OHLCV 完整的日频数据

### 性能考虑
1. **内存使用**: 大股票池和长回测期间会消耗较多内存
2. **计算时间**: BMA 模型训练较为耗时，建议合理设置重训频率
3. **数据量**: 推荐单次回测不超过 3 年历史数据

### 模型限制
1. **前瞻偏差**: 系统已内置滞后处理，但需验证因子计算逻辑
2. **生存偏差**: 当前未处理退市股票，可能高估收益
3. **流动性**: 未考虑市场流动性对大额交易的影响

## 🛠️ 定制开发

### 添加新因子
在 `_calculate_ml_factors()` 中添加因子计算逻辑

### 修改选股策略
在 `_execute_rebalance()` 中调整选股和权重分配

### 扩展风控规则
在 `_check_risk_exits()` 中添加自定义风控逻辑

### 自定义分析
继承 `BacktestAnalyzer` 类添加特定分析功能

## 📞 支持

如有问题或建议，请查看:
1. 日志文件: `logs/backtest_YYYYMMDD_HHMMSS.log`
2. 代码注释和文档字符串
3. 现有的测试用例和示例
