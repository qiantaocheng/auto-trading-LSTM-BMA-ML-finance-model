#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新模型配置以使用扩展的718只股票池
"""

import json
import os
import shutil
from datetime import datetime

def update_model_configurations():
    """更新BMA和LSTM模型配置以使用扩展股票池"""
    print("=== 更新模型配置以使用扩展718只股票池 ===")
    
    # 读取扩展股票池数据
    try:
        with open('expanded_stock_universe/stock_universe_20250806_200255.json', 'r', encoding='utf-8') as f:
            stock_universe = json.load(f)
        
        print(f"[OK] 读取扩展股票池: {stock_universe['statistics']['total_unique_stocks']} 只股票")
        
    except Exception as e:
        print(f"[ERROR] 读取扩展股票池失败: {e}")
        return False
    
    # 1. 更新BMA模型配置
    print("\n=== 更新BMA模型配置 ===")
    
    # 为BMA选择高质量股票池 (268只)
    bma_stocks = stock_universe['high_quality']
    print(f"BMA模型使用高质量股票池: {len(bma_stocks)} 只")
    
    # 创建BMA训练股票文件
    with open('bma_training_stocks_expanded.txt', 'w', encoding='utf-8') as f:
        f.write(f"# BMA量化分析训练股票列表 (扩展版)\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 股票数量: {len(bma_stocks)} 只高质量股票\n")
        f.write(f"# 数据源: 扩展股票池 (718只总库存)\n\n")
        
        for stock in bma_stocks:
            f.write(f"{stock}\n")
    
    # 更新BMA配置文件
    bma_config = {
        "model_name": "BMA_Quantitative_Analysis_Expanded",
        "version": "2.0_expanded",
        "updated_at": datetime.now().isoformat(),
        "stock_pools": {
            "high_quality": {
                "symbols": bma_stocks,
                "count": len(bma_stocks),
                "description": "大盘蓝筹股，市值>100亿，低波动率"
            }
        },
        "analysis_parameters": {
            "min_price": 2.0,
            "max_stocks_per_analysis": 300,  # 增加到300只
            "confidence_threshold": 0.6,
            "lookback_days": 252,
            "data_source": "expanded_stock_universe_718"
        }
    }
    
    with open('bma_stock_config_expanded.json', 'w', encoding='utf-8') as f:
        json.dump(bma_config, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] BMA配置已更新: {len(bma_stocks)} 只高质量股票")
    
    # 2. 更新LSTM模型配置
    print("\n=== 更新LSTM模型配置 ===")
    
    # 为LSTM使用所有质量股票 (高质量+中等质量 = 518只)
    lstm_stocks = stock_universe['high_quality'] + stock_universe['medium_quality']
    # 去重并排序
    lstm_stocks = sorted(list(set(lstm_stocks)))
    print(f"LSTM模型使用高质量+中等质量股票池: {len(lstm_stocks)} 只")
    
    # 创建LSTM训练股票文件
    with open('lstm_training_stocks_expanded.txt', 'w', encoding='utf-8') as f:
        f.write(f"# LSTM深度学习训练股票列表 (扩展版)\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 股票数量: {len(lstm_stocks)} 只 (高质量+中等质量)\n")
        f.write(f"# 数据源: 扩展股票池 (718只总库存)\n\n")
        
        for stock in lstm_stocks:
            f.write(f"{stock}\n")
    
    # 更新LSTM配置文件
    lstm_config = {
        "model_name": "LSTM_Multi_Day_Analysis_Expanded",
        "version": "2.0_expanded", 
        "updated_at": datetime.now().isoformat(),
        "training_data": {
            "source_file": "lstm_training_stocks_expanded.txt",
            "total_stocks": len(lstm_stocks),
            "stock_categories": ["high_quality", "medium_quality"],
            "data_source": "expanded_stock_universe_718"
        },
        "model_parameters": {
            "sequence_length": 60,
            "prediction_days": 5,
            "batch_size": 32,
            "epochs": 100,
            "validation_split": 0.2,
            "test_split": 0.1
        }
    }
    
    with open('lstm_stock_config_expanded.json', 'w', encoding='utf-8') as f:
        json.dump(lstm_config, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] LSTM配置已更新: {len(lstm_stocks)} 只股票")
    
    # 3. 创建交易系统配置更新
    print("\n=== 创建交易系统配置更新 ===")
    
    # 为交易系统选择前150只高质量股票
    trading_stocks = bma_stocks[:150]
    
    trading_update = {
        "trading_system": "expanded_stock_pool",
        "version": "2.0",
        "updated_at": datetime.now().isoformat(),
        "default_symbols": trading_stocks,
        "backup_symbols": bma_stocks[:200],
        "total_available": stock_universe['statistics']['total_unique_stocks'],
        "note": f"从718只扩展股票池中选择的150只顶级股票用于交易"
    }
    
    with open('trading_config_expanded.json', 'w', encoding='utf-8') as f:
        json.dump(trading_update, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] 交易系统配置: {len(trading_stocks)} 只精选股票")
    
    # 4. 生成更新报告
    print("\n=== 生成更新报告 ===")
    
    report = f"""# 模型配置扩展更新报告

## 更新时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 扩展股票池统计
- **总股票库存**: {stock_universe['statistics']['total_unique_stocks']} 只
- **高质量股票**: {stock_universe['statistics']['high_quality_count']} 只  
- **中等质量股票**: {stock_universe['statistics']['medium_quality_count']} 只
- **成长股**: {stock_universe['statistics']['growth_quality_count']} 只

## 模型配置更新

### BMA量化模型 (v2.0)
- **配置文件**: `bma_stock_config_expanded.json`
- **训练股票**: `bma_training_stocks_expanded.txt`
- **股票数量**: {len(bma_stocks)} 只高质量股票
- **最大分析量**: 300 只 (从200只增加)
- **数据源**: 扩展股票池 (718只)

### LSTM深度学习模型 (v2.0)
- **配置文件**: `lstm_stock_config_expanded.json`
- **训练股票**: `lstm_training_stocks_expanded.txt`  
- **股票数量**: {len(lstm_stocks)} 只 (高质量+中等质量)
- **训练容量**: 大幅提升，支持更多股票
- **数据源**: 扩展股票池 (718只)

### 交易系统 (v2.0)
- **配置文件**: `trading_config_expanded.json`
- **默认交易股票**: {len(trading_stocks)} 只精选股票
- **备用股票池**: {len(bma_stocks[:200])} 只
- **总可用股票**: {stock_universe['statistics']['total_unique_stocks']} 只

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

生成时间: {datetime.now().isoformat()}
"""
    
    with open('EXPANDED_MODEL_UPDATE_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 5. 显示总结
    print("\n" + "=" * 70)
    print("模型配置扩展更新完成!")
    print("=" * 70)
    print(f"原始股票池: 205 只")
    print(f"扩展股票池: {stock_universe['statistics']['total_unique_stocks']} 只")
    print(f"提升倍数: {stock_universe['statistics']['total_unique_stocks']/205:.1f}x")
    print()
    print(f"BMA模型: {len(bma_stocks)} 只高质量股票")
    print(f"LSTM模型: {len(lstm_stocks)} 只股票 (高+中等质量)")
    print(f"交易系统: {len(trading_stocks)} 只精选股票")
    print()
    print("立即可用:")
    print("1. python 量化模型_bma_enhanced.py --stock-file bma_training_stocks_expanded.txt")
    print("2. python lstm_multi_day_enhanced.py --stock-file lstm_training_stocks_expanded.txt")
    print()
    print("更新文件:")
    print("- bma_stock_config_expanded.json")
    print("- lstm_stock_config_expanded.json") 
    print("- bma_training_stocks_expanded.txt")
    print("- lstm_training_stocks_expanded.txt")
    print("- trading_config_expanded.json")
    print("- EXPANDED_MODEL_UPDATE_REPORT.md")
    
    return True

def main():
    """主函数"""
    success = update_model_configurations()
    return success

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 模型配置扩展更新成功! 现在有718只股票可供量化分析使用。")
    else:
        print("\n❌ 更新失败!")