#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to fix all Chinese text in autotrader directory
"""

import os
import re
import glob

# Comprehensive translation dictionary for autotrader modules
autotrader_translations = {
    # Module descriptions
    "SQLite数据库模块 - 管理股票列表和交易配置": "SQLite database module - Manages stock lists and trading configurations",
    "IBKR 自动交易最小闭环脚本": "IBKR automated trading minimal closed-loop script",
    "基于 ib_insync 封装 TWS API": "Based on ib_insync wrapper for TWS API",
    "覆盖自动交易常见用法": "Covers common automated trading use cases",
    "连接/重连、行情类型切换": "Connection/reconnection, market data type switching",
    "合约资格校验、主交易所设置": "Contract qualification verification, primary exchange settings",
    "行情订阅与价格获取": "Market data subscription and price retrieval",
    "账户摘要/账户更新、持仓、PnL": "Account summary/account updates, positions, PnL",
    "下单": "Place orders",
    "市价/限价/括号单": "market/limit/bracket orders",
    "撤单、订单/成交/佣金回报": "cancel orders, order/execution/commission reports",
    "简易风险控制": "Simple risk control",
    "资金占比/持仓检查/订单去重": "fund allocation ratio/position checks/order deduplication",
    
    # Database related
    "初始化数据库表结构": "Initialize database table structure",
    "使用重试机制": "using retry mechanism",
    "开始事务": "Start transaction",
    "股票列表表": "Stock list table",
    "股票表": "Stock table", 
    "简化模式": "Simplified mode",
    "全局tickers表": "Global tickers table",
    "仅保存股票代码": "only save stock codes",
    "满足": "satisfy",
    "只存字符串代号": "only store string codes",
    "的需求": " requirements",
    "交易配置表": "Trading configuration table",
    "交易审计表": "Trading audit table",
    "风险管理配置表": "Risk management configuration table",
    "创建默认股票列表": "Create default stock list",
    "数据库初始化完成": "Database initialization completed",
    "数据库表创建失败": "Database table creation failed", 
    "数据库初始化失败": "Database initialization failed",
    "创建默认数据": "Create default data",
    "检查是否已有数据": "Check if data already exists",
    "已有数据，不创建默认数据": "Data exists, not creating default data",
    "创建默认股票列表": "Create default stock list",
    "幂等": "idempotent",
    "科技股": "Tech Stocks",
    "美股主要科技公司": "Major US tech companies",
    "获取或创建后的列表ID": "Get or create list ID",
    
    # Configuration and logging
    "开始加载所有配置源": "Starting to load all configuration sources",
    "已加载数据库风险配置": "Loaded database risk configuration",
    "项": " items",
    "已加载数据库tickers": "Loaded database tickers",
    "条": " records",
    "配置加载完成": "Configuration loading completed",
    "启动事件循环管理器": "Starting event loop manager",
    "事件循环已在线程": "Event loop started in thread",
    "中启动": " started",
    "事件循环管理器启动成功": "Event loop manager started successfully",
    "资源监控已启动，间隔": "Resource monitoring started, interval",
    "秒": " seconds",
    "分配ClientID": "Assigned ClientID",
    "分配动态ClientID": "Assigned dynamic ClientID",
    "数据库初始化完成": "Database initialization completed",
    "已加载风险配置": "Risk configuration loaded",
    "事件处理线程启动": "Event processing thread started",
    "事件总线已启动": "Event bus started",
    
    # Trading related
    "导入现有模块失败": "Failed to import existing modules",
    "连接": "connection",
    "行情": "market data",
    "下单": "order placement", 
    "撤单": "order cancellation",
    "回报": "reports",
    "账户": "account",
    "持仓": "positions",
    "风控": "risk control",
    "工具": "tools",
    "实时": "real-time",
    "延时": "delayed",
    "合约": "contract",
    "资格校验": "qualification verification",
    "主交易所": "primary exchange",
    "设置": "settings",
    "订阅": "subscription",
    "价格": "price",
    "获取": "retrieval",
    "深度": "depth",
    "可按需扩展": "expandable as needed",
    "摘要": "summary",
    "更新": "updates",
    "市价": "market",
    "限价": "limit", 
    "括号单": "bracket order",
    "成交": "execution",
    "佣金": "commission",
    "简易": "simple",
    "占比": "ratio",
    "检查": "check",
    "去重": "deduplication",
    
    # Common terms
    "失败": "failed",
    "成功": "success",
    "完成": "completed",
    "开始": "starting",
    "启动": "start",
    "已": "",
    "的": "",
    "和": "and",
    "或": "or",
    "与": "and",
    "中": "in",
    "为": "as",
    "是": "is",
    "有": "has",
    "无": "no",
    "不": "not",
    "可": "can",
    "将": "will",
    "被": "be",
    "用": "use",
    "由": "by",
    "对": "for",
    "在": "in",
    "到": "to",
    "从": "from",
    "向": "to",
    "于": "at",
    "后": "after",
    "前": "before",
    "时": "when",
    "如": "if"
}

def fix_file_encoding(file_path):
    """Fix Chinese text in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_length = len(content)
        
        # Apply translations
        for chinese, english in autotrader_translations.items():
            content = content.replace(chinese, english)
        
        # Remove any remaining emojis
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0000FE0F]+'
        content = re.sub(emoji_pattern, '', content)
        
        # Write back if changed
        if len(content) != original_length:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path} ({original_length} -> {len(content)} chars)")
            return True
        else:
            print(f"No changes: {file_path}")
            return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def fix_autotrader_directory():
    """Fix all Python files in autotrader directory"""
    autotrader_dir = "autotrader"
    
    if not os.path.exists(autotrader_dir):
        print(f"Directory {autotrader_dir} not found")
        return
    
    # Find all Python files in autotrader directory
    python_files = glob.glob(os.path.join(autotrader_dir, "*.py"))
    
    total_fixed = 0
    total_files = 0
    
    for file_path in python_files:
        total_files += 1
        if fix_file_encoding(file_path):
            total_fixed += 1
    
    print(f"\n=== Summary ===")
    print(f"Total files processed: {total_files}")
    print(f"Files modified: {total_fixed}")
    print(f"Files unchanged: {total_files - total_fixed}")

if __name__ == "__main__":
    fix_autotrader_directory()