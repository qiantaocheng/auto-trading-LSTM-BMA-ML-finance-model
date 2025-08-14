#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to replace all Chinese text and emojis with English equivalents
"""

import re

# Dictionary of Chinese to English translations for the Alpha strategy file
translations = {
    # Main section headers
    "增强Alpha策略模块": "Enhanced Alpha Strategy Module", 
    "集成delay/decay、hump+rank、中性化、winsorize等高级技术": "Integrates advanced techniques: delay/decay, hump+rank, neutralization, winsorize",
    "移除外部高级因子模块依赖，所有因子已整合到本模块": "Removed external advanced factor dependencies, all factors integrated into this module",
    "配置日志": "Configure logging",
    "Alpha策略引擎：统一计算、中性化、排序、门控": "Alpha Strategy Engine: Unified computation, neutralization, ranking, gating",
    "初始化Alpha策略引擎": "Initialize Alpha Strategy Engine",
    "配置文件路径": "Configuration file path",
    "缓存计算结果": "Cache computation results",
    "所有因子已整合到本模块，无需外部依赖": "All factors integrated into this module, no external dependencies needed",
    "所有Alpha因子已整合到本模块": "All Alpha factors integrated into this module",
    "统计信息": "Statistics",
    "Alpha策略引擎初始化完成，加载": "Alpha Strategy Engine initialized, loaded",
    "个因子": " factors",
    "加载配置文件": "Load configuration file",
    "配置文件": "Config file", 
    "未找到，使用默认配置": " not found, using default config",
    "获取默认配置": "Get default configuration",
    "注册Alpha计算函数 - 所有因子已整合": "Register Alpha computation functions - All factors integrated",
    "技术因子": "Technical factors",
    "动量扩展因子": "Extended momentum factors", 
    "基本面因子": "Fundamental factors",
    "盈利能力因子": "Profitability factors",
    "应计项目因子": "Accrual factors",
    "质量评分因子": "Quality score factors",
    "特殊处理": "Special handling",
    "基础工具函数": "Basic Utility Functions",
    
    # Function docstrings and comments
    "Winsorize序列：去极值": "Winsorize series: Remove outliers",
    "按组标准化": "Group standardization", 
    "时间安全的线性回归中性化 - 防止使用未来数据": "Time-safe linear regression neutralization - Prevents use of future data",
    "关键修复：使用expanding window确保只使用历史数据": "KEY FIX: Use expanding window to ensure only historical data is used",
    "在实时交易中，T时刻不应该知道同一天其他股票的未来表现": "In real-time trading, at time T should not know future performance of other stocks on same day",
    "使用时间递进的方式计算中性化参数": "Use time-progressive approach to calculate neutralization parameters",
    "只使用到当前时点的历史数据": "Only use historical data up to current time point",
    "构建历史虚拟变量矩阵": "Build historical dummy variable matrix",
    "使用历史数据拟合回归模型": "Use historical data to fit regression model", 
    "对当前点进行中性化": "Neutralize current point",
    "点": "Point",
    "中性化失败": " neutralization failed",
    "门控变换：小信号置零": "Gating transformation: Set small signals to zero",
    "排序变换": "Ranking transformation",
    "时间安全的指数移动平均衰减 - 只使用历史数据": "Time-safe exponential moving average decay - Only use historical data",
    "使用expanding window确保每个时点只使用历史数据": "Use expanding window to ensure each time point only uses historical data",
    "增加一期延迟确保不使用当期数据": "Add one period lag to ensure current period data is not used",
    
    # Alpha factor function names and descriptions
    "Alpha因子计算函数": "Alpha Factor Computation Functions",
    "时间安全的动量因子：多窗口价格动量": "Time-safe momentum factor: Multi-window price momentum",
    "计算对数收益率动量，增加安全边际": "Calculate log return momentum with safety margin",
    "时间安全的指数衰减 - 使用expanding计算确保只用历史数据": "Time-safe exponential decay - Use expanding computation to ensure only historical data",
    "多窗口平均": "Multi-window average",
    "反转因子：短期价格反转": "Reversal factor: Short-term price reversal",
    "短期收益率，取负值表示反转": "Short-term returns, take negative to indicate reversal",
    "波动率因子：已实现波动率的倒数": "Volatility factor: Reciprocal of realized volatility",
    "滚动波动率（对每个ticker独立计算）": "Rolling volatility (calculated independently for each ticker)",
    "波动率倒数（低波动率异常）": "Volatility reciprocal (low volatility anomaly)",
    "成交量换手率因子": "Volume turnover factor",
    "成交量相对强度": "Volume relative strength",
    "如果没有成交量数据，用成交额替代": "If no volume data, use amount as substitute",
    "Amihud流动性指标：价格冲击的倒数": "Amihud liquidity indicator: Reciprocal of price impact",
    "计算日收益率": "Calculate daily returns",
    "流动性指标": "liquidity",
    "替代方案：使用价格*成交量": "Alternative: use price * volume",
    "滚动平均": "Rolling average",
    "流动性 = 1 / Amihud（高流动性更好）": "Liquidity = 1 / Amihud (higher liquidity is better)",
    "买卖价差因子（模拟）": "Bid-ask spread factor (simulated)",
    "如果有高低价数据，用": "If high-low price data available, use",
    "作为价差代理": " as spread proxy",
    "替代方案：用价格波动作为价差代理": "Alternative: use price volatility as spread proxy",
    "滚动平均价差": "Rolling average spread",
    "窄价差因子（价差越小越好）": "Narrow spread factor (smaller spread is better)",
    "残差动量：去除市场beta后的特异动量": "Residual momentum: Idiosyncratic momentum after removing market beta",
    "计算个股收益率": "Calculate individual stock returns",
    "计算市场收益率（等权平均或市值加权）": "Calculate market returns (equal weight or market cap weighted)",
    "滚动回归计算beta和残差": "Rolling regression to calculate beta and residuals",
    "从外部预计算的Series按索引切片，避免.name依赖": "Slice from externally pre-computed Series by index to avoid .name dependency",
    "简单线性回归：个股收益 = alpha + beta * 市场收益 + 残差": "Simple linear regression: stock return = alpha + beta * market return + residual",
    
    # v2 新增因子
    "新增因子：统一进入类方法并在注册表登记": "New factors: Unified entry into class methods and registered",
    "时间安全的短期反转": "Time-safe short-term reversal",
    "增加安全边际": "with safety margin",
    "使用": "Using",
    "的数据，增加安全边际": " data with safety margin",
    "短期反转计算失败": "Short-term reversal computation failed",
    "改进Amihud非流动性：更稳健的滚动中位数与EMA衰减": "Improved Amihud illiquidity: More robust rolling median with EMA decay",
    "Amihud非流动性计算失败": "Amihud illiquidity computation failed",
    "事件驱动代理": "event-driven proxy",
    "计算失败": " computation failed",
    
    # 动量类因子
    "新增动量类因子": "New momentum factors",
    "动量": "momentum",
    "的价格动量，排除最近": " price momentum, excluding recent",
    "个月前到": " months ago to",
    "个月前的收益率": " months ago returns",
    "动量计算失败": "momentum computation failed",
    "新高接近度：当前价格占": "proximity to new high: Current price as percentage of",
    "最高价的比例": " high price",
    "新高接近度计算失败": "proximity to new high computation failed",
    "低β异象：使用滚动闭式估计或": "Low beta anomaly: Using rolling closed-form estimation or",
    "实现": "implementation",
    "近似，取负值（低β更优）": "approximation, take negative (low beta is better)",
    "低β异象计算失败": "Low beta anomaly computation failed",
    "特异波动率：使用": "Idiosyncratic volatility: Using",
    "快速估计残差方差，取负值（低波动更优）": "fast estimation of residual variance, take negative (low volatility is better)",
    "特异波动率计算失败": "Idiosyncratic volatility computation failed",
    
    # 基本面因子
    "基本面因子（使用代理数据）": "Fundamental factors (using proxy data)",
    "财报意外": "Earnings surprise",
    "标准化盈余惊喜（使用价格反应作为代理）": "Standardized earnings surprise (using price reaction as proxy)",
    "季度": "Quarter",
    "使用价格在财报期间的异常反应作为": "Use abnormal price reaction during earnings period as",
    "代理": " proxy",
    "季度超额收益率作为": "Quarterly excess return as",
    "标准化": "Standardization",
    "财报意外": "Earnings surprise",
    "分析师": "Analyst",
    "上调修正（使用动量变化率作为代理）": "upward revision (using momentum change rate as proxy)",
    "使用动量变化作为分析师预期修正的代理": "Use momentum change as proxy for analyst expectation revision",
    "月": "month",
    "分析师修正计算失败": "Analyst revision computation failed",
    "收益率（使用收益率代理）": "yield (using return proxy)",
    "使用基于价格的收益率代理": "Use price-based return proxy for",
    "简化的": "Simplified",
    "代理": " proxy",
    "年化收益率作为": "Annualized return as",
    "计算失败": " computation failed",
    "自由现金流收益率": "Free cash flow yield",
    "使用现金流代理": "using cash flow proxy",
    "使用基于成交量和价格的现金流代理": "Use volume and price based cash flow proxy",
    "成交额": "amount",
    "价格作为现金流代理": "price as cash flow proxy",
    "盈利收益率": "Earnings yield",
    "市盈率倒数的代理": "proxy for P/E ratio reciprocal",
    "使用收益率历史数据作为": "Use historical return data as",
    "代理": " proxy",
    "销售收益率": "Sales yield",
    "市销率倒数的代理": "proxy for P/S ratio reciprocal",
    "使用成交量作为销售额代理": "Use volume as sales proxy",
    
    # 高级Alpha因子
    "高级Alpha因子（暂时移除复杂实现，保持基础功能）": "Advanced Alpha factors (temporarily removed complex implementation, maintain basic functionality)",
    "毛利率": "Gross margin",
    "简化实现": "Simplified implementation",
    "经营盈利能力": "Operating profitability",
    "为所有其他高级因子添加简化实现": "Add simplified implementation for all other advanced factors",
    "中性化": "neutralization",
    "净利率": "Net margin", 
    "现金收益率": "Cash yield",
    "股东收益率": "Shareholder yield",
    
    # 应计项目因子
    "总应计": "Total accruals",
    "取负值": "Take negative",
    "营运资本应计": "Working capital accruals",
    "净经营资产": "Net operating assets",
    "资产增长": "Asset growth",
    "净股本发行": "Net equity issuance",
    "投资因子": "Investment factor",
    
    # 质量评分因子
    "评分": "Score",
    "低风险更优": "lower risk is better",
    "稳定性": "Stability",
    "质量评分": "Quality score",
    "盈利稳定性": "Earnings stability", 
    "低波动更优": "lower volatility is better",
    
    # 主要计算流程
    "主要计算流程": "Main Computation Pipeline",
    "计算所有Alpha因子": "Compute all Alpha factors",
    "包含价格数据的DataFrame，必须有columns": "DataFrame containing price data, must have columns",
    "包含所有Alpha因子的DataFrame": "DataFrame containing all Alpha factors",
    "开始计算": "Starting computation of",
    "个Alpha因子": " Alpha factors",
    "确保必需的列存在": "Ensure required columns exist",
    "缺少必需的列": "Missing required columns",
    "添加元数据列（如果不存在）": "Add metadata columns (if not exist)",
    "获取参数": "Get parameters",
    "特殊处理hump变换": "Special handling for hump transformation",
    "Hump因子": "Hump factor",
    "的基础因子": "'s base factor",
    "未找到": " not found",
    "常规因子计算 - 所有因子已整合到本模块": "Regular factor computation - All factors integrated into this module",
    "未知的Alpha类型": "Unknown Alpha type",
    "数据处理流水线": "Data processing pipeline",
    "全局特征滞后以防止任何潜在的数据泄露": "Global feature lag to prevent any potential data leakage",
    "使用配置项": "Using config item",
    "默认": "default",
    "表示预测时仅使用至少": "indicates prediction only uses at least",
    "的信息": " information",
    "计算完成": " computation completed",
    "计算失败": " computation failed",
    "更新统计信息": "Update statistics",
    "构建结果DataFrame，保留原始列": "Build result DataFrame, preserve original columns",
    "Alpha计算完成，共": "Alpha computation completed, total",
    "所有Alpha因子计算失败": "All Alpha factor computation failed",
    
    # Alpha因子处理流水线
    "Alpha因子处理流水线": "Alpha factor processing pipeline",
    "去极值": "remove outliers",
    "中性化": "neutralize",
    "截面标准化": "cross-sectional standardization",
    "构建临时DataFrame进行中性化": "Build temporary DataFrame for neutralization",
    "中性化（默认关闭，避免与全局Pipeline重复；仅研究使用时打开）": "Neutralization (default off to avoid duplication with global pipeline; only enable for research)",
    "变换（rank或保持原样）": "Transform (rank or keep original)",
    
    # OOF评分
    "计算Out-of-Fold评分": "Compute Out-of-Fold scoring",
    "Alpha因子DataFrame": "Alpha factor DataFrame",
    "目标变量": "Target variable",
    "日期序列": "Date sequence",
    "评分指标": "Scoring metric",
    "每个Alpha的OOF评分": "OOF score for each Alpha",
    "开始计算OOF评分，指标": "Starting OOF scoring computation, metric",
    "统一索引以避免布尔索引不对齐": "Unify indices to avoid boolean index misalignment",
    "OOF评分跳过：alpha/target/dates无共同索引": "OOF scoring skipped: no common index between alpha/target/dates",
    "索引对齐失败，尝试继续": "Index alignment failed, trying to continue",
    "只评估数值型的因子列，排除标识/价格/元数据列": "Only evaluate numerical factor columns, exclude ID/price/metadata columns",
    "使用TimeSeriesSplit进行时间序列交叉验证": "Use TimeSeriesSplit for time series cross-validation",
    "获取测试期间的数据": "Get test period data",
    "使用numpy布尔数组，避免索引不一致": "Use numpy boolean array to avoid index inconsistency",
    "使用iloc配合布尔数组，确保位置索引对齐": "Use iloc with boolean array to ensure positional index alignment",
    "重置索引以确保对齐": "Reset index to ensure alignment",
    "去除NaN值": "Remove NaN values",
    "最少需要": "Minimum required",
    "个有效样本": " valid samples",
    "直接使用布尔索引，因为索引已重置": "Use boolean indexing directly since indices are reset",
    "计算评分": "Calculate score",
    "信息系数": "Information Coefficient",
    "样本数": "sample size",
    "更新统计信息": "Update statistics",
    "OOF评分完成，平均": "OOF scoring completed, average",
    
    # BMA权重计算
    "基于评分计算BMA权重，支持": "Calculate BMA weights based on scores, supports",
    "先验": " prior",
    "温度系数，控制权重集中度": "Temperature coefficient, controls weight concentration",
    "是否使用": "Whether to use",
    "作为先验权重": " as prior weights",
    "BMA权重": "BMA weights",
    "获取": "Get",
    "先验权重": " prior weights",
    "标准化评分": "Standardize scores",
    "数值稳定": "numerically stable",
    "结合": "Combine",
    "先验": " prior",
    "标准化": "Standardize",
    "贝叶斯更新：先验": "Bayesian update: prior",
    "似然": "likelihood", 
    "使用": "Using",
    "先验进行贝叶斯权重更新": " prior for Bayesian weight update",
    "普通softmax": "Regular softmax",
    "BMA权重计算完成，权重分布": "BMA weight calculation completed, weight distribution",
    "主要因子权重": "Main factor weights",
    
    # Alpha组合
    "使用BMA权重组合Alpha因子": "Combine Alpha factors using BMA weights",
    "组合后的Alpha信号": "Combined Alpha signal",
    "仅使用数值型因子列，排除元数据": "Only use numerical factor columns, exclude metadata",
    "确保权重对齐（列方向）": "Ensure weight alignment (column direction)",
    "列方向相乘，避免与行索引对齐导致的类型错误": "Column-wise multiplication to avoid type errors from row index alignment",
    "Alpha组合完成，信号范围": "Alpha combination completed, signal range",
    
    # 交易过滤器
    "应用交易过滤器": "Apply trading filters",
    "门控、截断、仓位限制": "gating, truncation, position limits",
    "原始信号": "Raw signal",
    "包含日期信息的DataFrame": "DataFrame containing date information",
    "过滤后的交易信号": "Filtered trading signal",
    "门控": "gating",
    "截断控制集中度": "Truncation controls concentration",
    "仅保留顶部和底部信号": "Only keep top and bottom signals",
    "交易过滤完成，非零信号比例": "Trading filter completed, non-zero signal ratio",
    "获取计算统计信息": "Get computation statistics",
    
    # Common words and phrases
    "失败": "failed",
    "成功": "success", 
    "错误": "error",
    "警告": "warning",
    "信息": "info",
    "开始": "Starting",
    "完成": "completed",
    "计算": "computation",
    "数据": "data",
    "因子": "factor",
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
    "交叉": "cross",
    "时间": "time",
    "序列": "series",
    "窗口": "window",
    "滚动": "rolling",
    "移动": "moving",
    "平均": "average",
    "标准": "standard",
    "偏差": "deviation",
    "方差": "variance",
    "协方差": "covariance",
    "相关": "correlation",
    "回归": "regression",
    "线性": "linear",
    "非线性": "nonlinear",
    "对数": "log",
    "指数": "exponential",
    "收益": "return",
    "收益率": "return rate",
    "价格": "price",
    "成交量": "volume",
    "成交额": "amount",
    "市值": "market cap",
    "流动性": "liquidity",
    "波动": "volatility",
    "波动率": "volatility",
    "风险": "risk",
    "权重": "weight",
    "组合": "portfolio",
    "资产": "asset",
    "股票": "stock",
    "市场": "market",
    "行业": "sector",
    "分组": "group",
    "排序": "rank",
    "排名": "ranking",
    "分位数": "quantile",
    "百分位": "percentile"
}

def replace_chinese_text(text):
    """Replace Chinese text with English equivalents"""
    result = text
    
    # Apply translations
    for chinese, english in translations.items():
        result = result.replace(chinese, english)
    
    # Remove remaining emojis
    emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0000FE0F]+'
    result = re.sub(emoji_pattern, '', result)
    
    return result

def fix_file_encoding(file_path):
    """Fix encoding issues in the specified file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Processing file: {file_path}")
        print(f"Original length: {len(content)} characters")
        
        # Replace Chinese text
        fixed_content = replace_chinese_text(content)
        
        print(f"Fixed length: {len(fixed_content)} characters")
        
        # Write back the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"Successfully fixed encoding in {file_path}")
        return True
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

if __name__ == "__main__":
    # Fix the enhanced_alpha_strategies.py file
    file_path = "enhanced_alpha_strategies.py"
    success = fix_file_encoding(file_path)
    
    if success:
        print("Encoding fix completed successfully!")
    else:
        print("Encoding fix failed!")