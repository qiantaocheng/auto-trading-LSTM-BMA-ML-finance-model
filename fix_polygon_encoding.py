#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to fix Chinese text in polygon_factors.py
"""

import re

# Translation dictionary for polygon_factors.py specific terms
polygon_translations = {
    # Factor categories
    "动量因子": "Momentum factors",
    "基本面因子": "Fundamental factors", 
    "盈利能力因子": "Profitability factors",
    "质量因子": "Quality factors",
    "低风险因子": "Low risk factors", 
    "技术因子": "Technical factors",
    "微观结构因子": "Microstructure factors",
    "事件驱动因子": "Event-driven factors",
    
    # Factor names
    "月动量": "month momentum",
    "复权": "adjusted",
    "周新高接近度": "week high proximity", 
    "残差动量": "residual momentum",
    "剔除市场": "exclude market",
    "行业": "sector",
    "日反转": "day reversal", 
    "短期反转": "short-term reversal",
    "价格动量一致性": "price momentum consistency",
    "动量加速度": "momentum acceleration",
    "截面动量": "cross-sectional momentum",
    "财报意外": "earnings surprise",
    "标准化盈余惊喜": "standardized earnings surprise", 
    "分析师": "analyst",
    "上调修正": "upward revision",
    "月": "month",
    "收益率": "yield",
    "自由现金流": "free cash flow",
    "毛利率": "gross margin",
    "总应计": "total accruals",
    "取负": "take negative",
    "资产增长": "asset growth", 
    "净股本发行": "net equity issuance",
    "净回购为正": "net buyback positive",
    "投资因子": "investment factor",
    "资产": "assets",
    "经营盈利能力": "operating profitability",
    "现金收益率": "cash yield",
    "现金流": "cash flow",
    "股东收益率": "shareholder yield", 
    "股息": "dividend",
    "回购": "buyback",
    "市值": "market cap",
    "盈利收益率": "earnings yield",
    "销售收益率": "sales yield",
    "中性化后": "after neutralization",
    "净利率": "net margin",
    "盈利稳定性": "earnings stability",
    "波动低": "low volatility",
    "销售增长稳定性": "sales growth stability",
    "毛利率扩张": "gross margin expansion",
    "质量评分": "quality score",
    "财务实力": "financial strength", 
    "经营效率": "operating efficiency",
    "会计质量": "accounting quality",
    "低风险异常": "low risk anomaly",
    "最大回撤": "max drawdown",
    "低波动": "low volatility",
    "价值": "value",
    "质量": "quality",
    "多元化": "diversification",
    "风险平价": "risk parity",
    "特异波动率": "idiosyncratic volatility",
    "偏度": "skewness",
    "峰度": "kurtosis",
    "短期反转": "short reversal",
    "动量": "momentum", 
    "相对强弱": "relative strength",
    "成交量": "volume",
    "异常": "anomaly",
    "价差": "spread",
    "买卖价差": "bid-ask spread",
    "有效价差": "effective spread",
    "订单流毒性": "order flow toxicity", 
    "价格冲击": "price impact",
    "资金流": "money flow",
    "机构持仓": "institutional holdings",
    "内部人交易": "insider trading",
    "分析师覆盖": "analyst coverage",
    "新闻情绪": "news sentiment",
    "社交媒体情绪": "social media sentiment",
    "卖空利息": "short interest",
    "期权": "options",
    "隐含波动率": "implied volatility",
    "波动率": "volatility",
    "偏斜": "skew",
    "期限结构": "term structure",
    
    # Common terms
    "个": "",
    "因子": "factor",
    "计算": "calculation",
    "数据": "data", 
    "结果": "result",
    "方法": "method",
    "函数": "function", 
    "参数": "parameter",
    "配置": "configuration",
    "模块": "module",
    "引擎": "engine", 
    "策略": "strategy",
    "优化": "optimization",
    "预测": "prediction",
    "模型": "model",
    "特征": "feature",
    "目标": "target",
    "训练": "training",
    "测试": "test",
    "验证": "validation",
    "错误": "error",
    "警告": "warning",
    "信息": "info",
    "开始": "start",
    "完成": "completed", 
    "成功": "success",
    "失败": "failed"
}

def fix_polygon_file():
    """Fix Chinese text in polygon_factors.py"""
    try:
        with open('polygon_factors.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Original file length: {len(content)} characters")
        
        # Apply translations
        for chinese, english in polygon_translations.items():
            content = content.replace(chinese, english)
        
        # Remove any remaining emojis
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0000FE0F]+'
        content = re.sub(emoji_pattern, '', content)
        
        print(f"Fixed file length: {len(content)} characters")
        
        # Write back
        with open('polygon_factors.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("Successfully fixed polygon_factors.py encoding")
        return True
        
    except Exception as e:
        print(f"Error fixing polygon_factors.py: {e}")
        return False

if __name__ == "__main__":
    success = fix_polygon_file()
    
    if success:
        print("Polygon factors encoding fix completed!")
    else:
        print("Polygon factors encoding fix failed!")