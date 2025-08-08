# 股票池创建报告

## 创建时间
2025-08-06 20:46:10

## 数据来源
精心挑选的优质美股列表

## 股票池详情

### High Quality
- 股票数量: 154
- 文本文件: exports/high_quality_stocks.txt
- 详细信息: exports/high_quality_details.json

### Medium Quality
- 股票数量: 33
- 文本文件: exports/medium_quality_stocks.txt
- 详细信息: exports/medium_quality_details.json

### Growth Stocks
- 股票数量: 20
- 文本文件: exports/growth_stocks_stocks.txt
- 详细信息: exports/growth_stocks_details.json

### All Tradeable
- 股票数量: 205
- 文本文件: exports/all_tradeable_stocks.txt
- 详细信息: exports/all_tradeable_details.json

## 使用建议

1. **模型训练**: 使用 `high_quality_stocks.txt` (推荐)
   - 包含大盘蓝筹股，数据质量高，适合模型学习
   
2. **稳健交易**: 使用 `high_quality_stocks.txt`
   - 风险较低，流动性好，适合稳健策略
   
3. **平衡策略**: 使用 `medium_quality_stocks.txt`
   - 风险和收益平衡，适合多元化投资
   
4. **成长策略**: 使用 `growth_stocks.txt`
   - 成长潜力大，但波动也较大

## 文件说明

- `*_stocks.txt`: 纯股票代码列表，每行一个
- `*_details.json`: 包含完整股票池信息
- `training_universe.json`: 专用训练配置文件
- `README.md`: 本报告文件

## 股票特点

### High Quality (高质量)
- 大盘股，市值通常 > $50B
- 流动性极高，日交易量 > 1M股
- 财务稳健，盈利稳定
- 包含: AAPL, MSFT, GOOGL, AMZN等

### Medium Quality (中等质量)  
- 中盘股，市值 $1B - $50B
- 有成长潜力但风险略高
- 部分周期性行业股票
- 包含: F, GM, AAL, SNAP等

### Growth Stocks (成长股)
- 新兴科技公司
- 高成长率但波动大
- 适合看好未来趋势的投资
- 包含: RBLX, DDOG, CRWD等

---
自动生成 - 股票池管理系统
