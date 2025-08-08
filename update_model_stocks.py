#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新模型股票池脚本
将新创建的精选股票池应用到BMA和LSTM模型的配置中
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path

def read_stock_list(file_path):
    """读取股票列表文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            stocks = [line.strip() for line in f if line.strip()]
        return stocks
    except Exception as e:
        print(f"[ERROR] 读取股票列表失败: {e}")
        return []

def update_trading_manager_config():
    """更新量化交易管理器的股票配置"""
    print("=== 更新量化交易管理器配置 ===")
    
    # 读取高质量股票列表
    high_quality_stocks = read_stock_list("exports/high_quality_stocks.txt")
    if not high_quality_stocks:
        print("[ERROR] 无法读取高质量股票列表")
        return False
    
    print(f"[OK] 读取到 {len(high_quality_stocks)} 只高质量股票")
    
    # 更新integrated_trading_system配置
    try:
        config_files = ["config_template.json", "config.json"]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                # 备份原配置
                backup_file = f"{config_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(config_file, backup_file)
                print(f"[BACKUP] {config_file} -> {backup_file}")
                
                # 读取现有配置
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 更新股票列表
                if 'trading' not in config:
                    config['trading'] = {}
                
                config['trading']['default_symbols'] = high_quality_stocks[:100]  # 前100只作为默认
                
                # 保存更新后的配置
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                print(f"[OK] 已更新 {config_file}: {len(high_quality_stocks[:100])} 只股票")
    
    except Exception as e:
        print(f"[ERROR] 更新交易系统配置失败: {e}")
        return False
    
    return True

def create_bma_stock_config():
    """为BMA模型创建股票配置文件"""
    print("\n=== 创建BMA模型股票配置 ===")
    
    # 读取所有股票池
    high_quality = read_stock_list("exports/high_quality_stocks.txt")
    medium_quality = read_stock_list("exports/medium_quality_stocks.txt")
    growth_stocks = read_stock_list("exports/growth_stocks_stocks.txt")
    
    if not high_quality:
        print("[ERROR] 无法读取股票列表")
        return False
    
    # 创建BMA配置文件
    bma_config = {
        "model_name": "BMA_Quantitative_Analysis",
        "updated_at": datetime.now().isoformat(),
        "stock_pools": {
            "high_quality": {
                "description": "大盘蓝筹股，适合稳健策略",
                "count": len(high_quality),
                "symbols": high_quality
            },
            "medium_quality": {
                "description": "中盘成长股，平衡风险收益",
                "count": len(medium_quality),
                "symbols": medium_quality
            },
            "growth_stocks": {
                "description": "高成长潜力股",
                "count": len(growth_stocks),
                "symbols": growth_stocks
            }
        },
        "default_pool": "high_quality",
        "analysis_parameters": {
            "min_price": 2.0,
            "max_stocks_per_analysis": 200,
            "confidence_threshold": 0.6,
            "lookback_days": 252
        }
    }
    
    # 保存BMA配置
    bma_config_file = "bma_stock_config.json"
    with open(bma_config_file, 'w', encoding='utf-8') as f:
        json.dump(bma_config, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] BMA配置已保存: {bma_config_file}")
    print(f"   - 高质量股票: {len(high_quality)} 只")
    print(f"   - 中等质量股票: {len(medium_quality)} 只")
    print(f"   - 成长股票: {len(growth_stocks)} 只")
    
    return True

def create_lstm_stock_config():
    """为LSTM模型创建股票配置文件"""
    print("\n=== 创建LSTM模型股票配置 ===")
    
    # 读取训练配置
    try:
        with open("exports/training_universe.json", 'r', encoding='utf-8') as f:
            training_config = json.load(f)
        
        symbols = training_config['training_universe']['symbols']
    except Exception as e:
        print(f"[ERROR] 读取训练配置失败: {e}")
        return False
    
    # 创建LSTM配置文件
    lstm_config = {
        "model_name": "LSTM_Multi_Day_Analysis",
        "updated_at": datetime.now().isoformat(),
        "training_data": {
            "source": "curated_high_quality_stocks",
            "total_stocks": len(symbols),
            "symbols": symbols,
            "data_source": training_config['training_universe']['data_source'],
            "note": training_config['training_universe']['note']
        },
        "model_parameters": {
            "sequence_length": 60,
            "prediction_days": 5,
            "batch_size": 32,
            "epochs": 100,
            "validation_split": 0.2
        },
        "analysis_parameters": {
            "min_confidence": 0.7,
            "top_n_predictions": 10,
            "risk_threshold": 0.15
        }
    }
    
    # 保存LSTM配置
    lstm_config_file = "lstm_stock_config.json"
    with open(lstm_config_file, 'w', encoding='utf-8') as f:
        json.dump(lstm_config, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] LSTM配置已保存: {lstm_config_file}")
    print(f"   - 训练股票数量: {len(symbols)} 只")
    print(f"   - 数据源: {training_config['training_universe']['data_source']}")
    
    return True

def create_model_stock_lists():
    """创建模型专用的股票列表文件"""
    print("\n=== 创建模型专用股票列表 ===")
    
    # 读取股票池
    high_quality = read_stock_list("exports/high_quality_stocks.txt")
    
    if not high_quality:
        return False
    
    # 创建BMA训练股票列表（前150只）
    bma_stocks = high_quality[:150]
    with open("bma_training_stocks.txt", 'w', encoding='utf-8') as f:
        f.write("# BMA模型训练股票列表\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 股票数量: {len(bma_stocks)}\n\n")
        for stock in bma_stocks:
            f.write(f"{stock}\n")
    
    print(f"[OK] BMA训练股票列表: bma_training_stocks.txt ({len(bma_stocks)} 只)")
    
    # 创建LSTM训练股票列表（所有高质量股票）
    lstm_stocks = high_quality
    with open("lstm_training_stocks.txt", 'w', encoding='utf-8') as f:
        f.write("# LSTM模型训练股票列表\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 股票数量: {len(lstm_stocks)}\n\n")
        for stock in lstm_stocks:
            f.write(f"{stock}\n")
    
    print(f"[OK] LSTM训练股票列表: lstm_training_stocks.txt ({len(lstm_stocks)} 只)")
    
    return True

def create_summary_report():
    """创建股票池更新总结报告"""
    print("\n=== 生成股票池更新报告 ===")
    
    # 收集统计信息
    high_quality = read_stock_list("exports/high_quality_stocks.txt")
    medium_quality = read_stock_list("exports/medium_quality_stocks.txt")
    growth_stocks = read_stock_list("exports/growth_stocks_stocks.txt")
    
    total_stocks = len(set(high_quality + medium_quality + growth_stocks))
    
    report_content = f"""# 量化模型股票池更新报告

## 更新时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 股票池统计
- **高质量股票**: {len(high_quality)} 只 (大盘蓝筹)
- **中等质量股票**: {len(medium_quality)} 只 (中盘成长)
- **成长股票**: {len(growth_stocks)} 只 (高增长潜力)
- **总计不重复股票**: {total_stocks} 只

## 模型配置更新

### BMA模型
- 配置文件: `bma_stock_config.json`
- 训练股票列表: `bma_training_stocks.txt`
- 默认使用: 高质量股票池 (前150只)
- 分析参数: 最低价格$2, 最大200只股票

### LSTM模型  
- 配置文件: `lstm_stock_config.json`
- 训练股票列表: `lstm_training_stocks.txt`
- 默认使用: 全部高质量股票 ({len(high_quality)} 只)
- 训练参数: 60天序列长度, 5天预测期

### 交易系统集成
- 更新文件: `config_template.json`, `config.json`
- 默认交易股票: 前100只高质量股票
- 实时数据订阅: 支持所有股票池

## 使用说明

### 运行BMA分析
```bash
python 量化模型_bma_enhanced.py --stock-file bma_training_stocks.txt
```

### 运行LSTM分析  
```bash
python lstm_multi_day_enhanced.py --stock-file lstm_training_stocks.txt
```

### 运行模型验证
```bash
python model_validator.py
```

## 股票池质量标准
- 最低股价: $2.00
- 最小市值: $300M
- 最小日均成交量: 150K股
- 最大年化波动率: 80%
- 最大Beta值: 2.5
- 至少上市1年

## 文件清单
- `exports/high_quality_stocks.txt` - 高质量股票池
- `exports/training_universe.json` - 训练配置文件
- `bma_stock_config.json` - BMA模型配置
- `lstm_stock_config.json` - LSTM模型配置
- `bma_training_stocks.txt` - BMA训练股票列表
- `lstm_training_stocks.txt` - LSTM训练股票列表

---
*报告生成时间: {datetime.now().isoformat()}*
*系统版本: 量化交易系统 v2.0*
"""
    
    # 保存报告
    with open("MODEL_STOCK_UPDATE_REPORT.md", 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("[OK] 更新报告已生成: MODEL_STOCK_UPDATE_REPORT.md")
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("量化模型股票池更新工具")
    print("=" * 60)
    print()
    
    # 检查必要文件
    required_files = [
        "exports/high_quality_stocks.txt",
        "exports/training_universe.json"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"[ERROR] 缺少必要文件: {file_path}")
            print("请先运行 quick_stock_update.py 创建股票池")
            return False
    
    try:
        # 1. 更新交易管理器配置
        if not update_trading_manager_config():
            return False
        
        # 2. 创建BMA模型配置
        if not create_bma_stock_config():
            return False
        
        # 3. 创建LSTM模型配置
        if not create_lstm_stock_config():
            return False
        
        # 4. 创建模型专用股票列表
        if not create_model_stock_lists():
            return False
        
        # 5. 生成总结报告
        if not create_summary_report():
            return False
        
        print("\n" + "=" * 60)
        print("[SUCCESS] 所有模型股票池更新完成!")
        print("=" * 60)
        print()
        print("更新内容:")
        print("1. [OK] 交易系统配置已更新")
        print("2. [OK] BMA模型配置已创建")
        print("3. [OK] LSTM模型配置已创建")
        print("4. [OK] 模型训练股票列表已创建")
        print("5. [OK] 更新报告已生成")
        print()
        print("现在可以使用新的股票池运行模型训练:")
        print("- BMA模型: python 量化模型_bma_enhanced.py --stock-file bma_training_stocks.txt")
        print("- LSTM模型: python lstm_multi_day_enhanced.py --stock-file lstm_training_stocks.txt")
        print("- 模型验证: python model_validator.py")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 更新过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    print(f"\n程序结束，退出代码: {exit_code}")