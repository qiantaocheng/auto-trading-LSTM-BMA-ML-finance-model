# 统一因子管理系统

## 概述

统一因子管理系统整合了项目中的所有因子计算库，提供统一的接口来访问Barra风格因子、Polygon因子和AutoTrader因子。该系统解决了代码重复、接口不一致和缓存管理等问题。

## 主要特性

### 🎯 统一接口
- 标准化的因子计算接口
- 统一的结果格式
- 一致的错误处理机制

### 🚀 智能引擎选择
- 自动选择最佳因子引擎
- 基于优先级的fallback机制
- 引擎健康状态监控

### 💾 高效缓存
- 内存 + 磁盘双重缓存
- 智能缓存失效策略
- 缓存命中率统计

### 📊 多因子支持
- **Barra风格因子**: 31个标准风格因子（动量、价值、质量、波动率、流动性、成长）
- **Polygon完整因子**: 40+专业量化因子（基本面、微观结构、盈利能力）
- **AutoTrader因子**: 8个优化的交易信号因子

## 文件结构

```
D:\trade\
├── unified_factor_manager.py          # 主要管理器
├── config/
│   └── unified_factors_config.json    # 配置文件
├── examples/
│   └── unified_factors_example.py     # 使用示例
├── barra_style_factors.py             # Barra因子库（保留）
├── polygon_complete_factors.py        # Polygon因子库（保留）
└── autotrader/
    └── unified_polygon_factors.py     # AutoTrader因子库（保留）
```

## 安装和使用

### 基本使用

```python
from unified_factor_manager import UnifiedFactorManager

# 创建管理器
manager = UnifiedFactorManager()

# 计算单个因子
result = manager.calculate_factor("momentum_12_1", "AAPL")
if result:
    print(f"动量因子值: {result.value:.4f}")

# 批量计算因子
factors = ["momentum_12_1", "book_to_price", "roe"]
results = manager.calculate_factor_set(factors, "AAPL")
```

### 便捷函数

```python
from unified_factor_manager import calculate_factor, get_available_factors

# 快速计算因子
result = calculate_factor("rsi", "AAPL")

# 获取可用因子列表
momentum_factors = get_available_factors("momentum")
```

## 因子分类

### 动量因子 (Momentum)
- `momentum_12_1`: 12-1月动量
- `momentum_6_1`: 6-1月动量  
- `price_trend`: 价格趋势
- `mean_reversion`: 均值回归信号

### 价值因子 (Value)
- `book_to_price`: 账面市值比
- `earnings_to_price`: 市盈率倒数
- `sales_to_price`: 市销率倒数
- `earnings_yield`: 盈利收益率

### 质量因子 (Quality)
- `roe`: 净资产收益率
- `gross_profitability`: 毛利率
- `accruals`: 应计项目
- `earnings_quality`: 盈利质量

### 波动率因子 (Volatility)
- `volatility_90d`: 90日波动率
- `residual_volatility`: 残差波动率
- `idiosyncratic_volatility`: 特异波动率
- `downside_volatility`: 下行波动率

### 流动性因子 (Liquidity)
- `amihud_illiquidity`: Amihud非流动性指标
- `turnover_rate`: 换手率
- `trading_volume_ratio`: 成交量比率
- `volume`: 成交量信号

### 技术因子 (Technical)
- `rsi`: 相对强弱指数
- `bollinger`: 布林带指标
- `trend`: 趋势信号
- `composite`: 综合技术信号

## 引擎优先级

系统按以下优先级选择引擎：

1. **Barra引擎 (优先级: 3)** - 适合长期价值投资和风险模型
2. **Polygon引擎 (优先级: 2)** - 适合专业量化研究和多因子模型
3. **AutoTrader引擎 (优先级: 1)** - 适合短期自动交易

## 配置说明

### 引擎配置

```json
{
    "engines": {
        "barra": {
            "enabled": true,
            "priority": 3,
            "description": "Barra风格因子引擎"
        }
    }
}
```

### 缓存配置

```json
{
    "cache": {
        "default_ttl": 300,
        "max_memory_items": 1000,
        "disk_cache_enabled": true
    }
}
```

## 数据质量控制

### 质量指标
- **数据完整性**: 检查必需字段是否存在
- **数据新鲜度**: 检查数据时效性
- **异常值检测**: 自动识别和处理异常值

### 质量阈值

```json
{
    "data_quality": {
        "min_quality_threshold": 0.7,
        "max_staleness_hours": 24,
        "required_data_points": 20
    }
}
```

## 性能优化

### 缓存策略
- **L1缓存**: 内存缓存，快速访问
- **L2缓存**: 磁盘缓存，持久存储
- **智能失效**: 基于TTL和数据变化的缓存失效

### 批量计算
```python
# 批量计算多个股票的同一因子
symbols = ["AAPL", "MSFT", "GOOGL"]
factor_results = {}

for symbol in symbols:
    result = manager.calculate_factor("momentum_12_1", symbol)
    if result:
        factor_results[symbol] = result.value
```

## 错误处理和日志

### 日志级别
- **INFO**: 正常操作记录
- **WARNING**: 警告信息（如缓存失效）
- **ERROR**: 错误信息（如计算失败）
- **DEBUG**: 详细调试信息

### 常见错误

1. **引擎不可用**: 检查依赖库是否正确安装
2. **数据源错误**: 检查API配置和网络连接
3. **计算失败**: 检查输入参数和数据质量

## 迁移指南

### 从现有系统迁移

#### 旧代码:
```python
from barra_style_factors import BarraStyleFactors

barra = BarraStyleFactors()
result = barra.momentum_12_1("AAPL")
```

#### 新代码:
```python
from unified_factor_manager import calculate_factor

result = calculate_factor("barra_momentum_12_1", "AAPL")
# 或自动选择最佳引擎
result = calculate_factor("momentum_12_1", "AAPL")
```

## 扩展开发

### 添加新因子引擎

```python
class CustomFactorEngine:
    def __init__(self):
        self.name = "custom"
    
    def calculate_custom_factor(self, symbol: str) -> float:
        # 自定义因子计算逻辑
        pass

# 注册到管理器
manager.engines['custom'] = CustomFactorEngine()
```

### 添加新因子

```python
# 在对应引擎中添加新因子方法
def new_momentum_factor(self, symbol: str) -> float:
    # 计算逻辑
    pass

# 在因子注册表中添加映射
manager.factor_registry['custom_new_momentum'] = {
    'engine': 'custom',
    'factor_name': 'new_momentum_factor',
    'category': FactorCategory.MOMENTUM,
    'priority': 1
}
```

## API 参考

### UnifiedFactorManager

#### 方法

- `calculate_factor(factor_name, symbol, engine='auto', **kwargs)` - 计算单个因子
- `calculate_factor_set(factor_names, symbol, **kwargs)` - 批量计算因子
- `get_available_factors(category=None, engine=None)` - 获取可用因子列表
- `get_engine_status()` - 获取引擎状态
- `cleanup_cache()` - 清理过期缓存

### FactorResult

#### 属性

- `factor_name: str` - 因子名称
- `category: FactorCategory` - 因子分类
- `value: float` - 因子值
- `confidence: float` - 置信度
- `timestamp: datetime` - 计算时间戳
- `symbol: str` - 股票代码
- `data_source: DataSource` - 数据源
- `computation_time: float` - 计算耗时
- `data_quality: float` - 数据质量评分

## 监控和调试

### 性能监控

```python
# 获取引擎状态
status = manager.get_engine_status()
print(f"Barra引擎可用: {status['barra']['available']}")

# 获取缓存统计
cache_stats = manager.cache_manager.get_stats()
print(f"缓存命中率: {cache_stats['hit_rate']:.2%}")
```

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细日志
manager = UnifiedFactorManager()
result = manager.calculate_factor("momentum_12_1", "AAPL")
```

## 常见问题 (FAQ)

### Q: 如何选择最适合的因子引擎？

A: 系统会自动根据以下规则选择：
- 长期投资策略：优先使用Barra引擎
- 短期交易策略：优先使用AutoTrader引擎  
- 研究和回测：优先使用Polygon引擎

### Q: 因子计算失败怎么办？

A: 系统会自动尝试fallback引擎，并记录详细错误日志。检查：
1. 网络连接是否正常
2. API配置是否正确
3. 股票代码是否有效

### Q: 如何提高计算性能？

A: 建议：
1. 启用缓存（默认启用）
2. 使用批量计算接口
3. 合理设置缓存TTL
4. 定期清理过期缓存

### Q: 如何扩展支持新的数据源？

A: 实现新的引擎类，并在配置文件中注册。参考扩展开发章节。

## 支持和贡献

如有问题或建议，请联系开发团队或提交Issue。欢迎贡献代码和改进建议。