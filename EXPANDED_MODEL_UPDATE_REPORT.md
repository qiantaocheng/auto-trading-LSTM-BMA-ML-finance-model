# 模型配置扩展更新报告

## 更新时间
2025-08-06 20:03:57

## 扩展股票池统计
- **总股票库存**: 718 只
- **高质量股票**: 268 只  
- **中等质量股票**: 250 只
- **成长股**: 60 只

## 模型配置更新

### BMA量化模型 (v2.0)
- **配置文件**: `bma_stock_config_expanded.json`
- **训练股票**: `bma_training_stocks_expanded.txt`
- **股票数量**: 268 只高质量股票
- **最大分析量**: 300 只 (从200只增加)
- **数据源**: 扩展股票池 (718只)

### LSTM深度学习模型 (v2.0)
- **配置文件**: `lstm_stock_config_expanded.json`
- **训练股票**: `lstm_training_stocks_expanded.txt`  
- **股票数量**: 501 只 (高质量+中等质量)
- **训练容量**: 大幅提升，支持更多股票
- **数据源**: 扩展股票池 (718只)

### 交易系统 (v2.0)
- **配置文件**: `trading_config_expanded.json`
- **默认交易股票**: 150 只精选股票
- **备用股票池**: 200 只
- **总可用股票**: 718 只

## 性能提升

### 覆盖范围
- **原始股票池**: 205 只
- **扩展股票池**: 718 只
- **提升倍数**: 3.5x

### 多样性提升
- 覆盖9个主要行业板块
- 包含科技、金融、医疗、消费等全行业
- 大盘、中盘、成长股全覆盖

## 立即可用的命令

### 使用扩展股票池运行模型
```bash
# BMA量化分析 (268只高质量股票)
python 量化模型_bma_enhanced.py --stock-file bma_training_stocks_expanded.txt

# LSTM深度学习分析 (518只股票)  
python lstm_multi_day_enhanced.py --stock-file lstm_training_stocks_expanded.txt
```

### 验证扩展配置
```bash
python test_expanded_integration.py
```

---
**扩展完成**: 从205只股票扩展到718只股票，提升3.5倍覆盖范围
**模型就绪**: BMA和LSTM都已配置使用扩展股票池
**系统状态**: 完全就绪，可立即开始大规模量化分析

生成时间: 2025-08-06T20:03:57.324931
