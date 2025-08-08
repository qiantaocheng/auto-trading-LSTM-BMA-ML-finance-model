#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动更新股票池并导入默认股票池
运行完成后直接将筛选后的股票作为训练参数
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_universe_manager import StockUniverseManager


def setup_logging():
    """设置日志"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"auto_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_config():
    """加载配置"""
    config_file = "stock_config.json"
    
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # 使用默认配置
        return {
            'database_file': 'data/stock_universe.db',
            'quality_filter': {
                'min_price': 2.0,
                'min_market_cap': 300_000_000,
                'min_avg_volume': 150_000,
                'max_bid_ask_spread_pct': 0.8,
                'max_volatility': 80.0,
                'max_beta': 2.5,
                'min_days_since_ipo': 365
            },
            'crawler': {
                'max_workers': 8,
                'request_delay_seconds': 0.2,
                'timeout_seconds': 30
            }
        }


def update_and_export_stock_universe(logger):
    """更新股票池并导出默认股票"""
    try:
        logger.info("🚀 开始自动更新股票池...")
        
        # 创建必要目录
        Path("data").mkdir(exist_ok=True)
        Path("exports").mkdir(exist_ok=True)
        
        # 加载配置
        config = load_config()
        logger.info(f"✅ 配置加载完成")
        
        # 创建管理器
        manager = StockUniverseManager(config)
        logger.info("✅ 股票池管理器初始化完成")
        
        # 更新股票池
        logger.info("📊 开始爬取和更新股票数据...")
        success = manager.crawl_and_update_universe(force_update=True)
        
        if not success:
            logger.error("❌ 股票池更新失败")
            return False
        
        logger.info("✅ 股票池更新完成")
        
        # 获取统计信息
        stats = manager.database.get_statistics()
        logger.info(f"📊 统计信息:")
        logger.info(f"   总股票数: {stats.get('total_stocks', 0):,}")
        logger.info(f"   可交易股票: {stats.get('tradeable_stocks', 0):,}")
        
        # 导出不同质量等级的股票池
        quality_levels = [
            (0.8, 'high_quality'),      # 高质量
            (0.6, 'good_quality'),      # 良好质量  
            (0.4, 'medium_quality'),    # 中等质量
            (0.0, 'all_tradeable')      # 所有可交易
        ]
        
        exported_pools = {}
        
        for min_quality, pool_name in quality_levels:
            # 获取股票列表
            symbols = manager.get_trading_universe(min_quality_score=min_quality, max_stocks=5000)
            
            if symbols:
                # 保存为自定义股票池
                description = f"Auto-generated pool with quality >= {min_quality}"
                manager.create_custom_portfolio(pool_name, symbols, description)
                
                # 导出到文件
                export_file = f"exports/{pool_name}_stocks.txt"
                with open(export_file, 'w', encoding='utf-8') as f:
                    for symbol in symbols:
                        f.write(f"{symbol}\n")
                
                # 导出详细信息
                detail_file = f"exports/{pool_name}_details.json"
                stocks_detail = manager.database.get_tradeable_stocks(min_quality, len(symbols))
                with open(detail_file, 'w', encoding='utf-8') as f:
                    json.dump(stocks_detail, f, indent=2, default=str, ensure_ascii=False)
                
                exported_pools[pool_name] = {
                    'count': len(symbols),
                    'min_quality': min_quality,
                    'file': export_file,
                    'detail_file': detail_file
                }
                
                logger.info(f"✅ 导出 {pool_name}: {len(symbols)} 只股票 -> {export_file}")
        
        # 设置默认训练股票池 (使用good_quality作为默认)
        default_symbols = exported_pools.get('good_quality', {}).get('count', 0)
        if default_symbols > 0:
            # 更新integrated_trading_system的默认配置
            update_default_trading_symbols(exported_pools['good_quality'], logger)
        
        # 生成汇总报告
        generate_summary_report(exported_pools, stats, logger)
        
        logger.info("🎉 自动更新和导出完成!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 自动更新失败: {e}")
        return False


def update_default_trading_symbols(pool_info, logger):
    """更新默认交易股票配置"""
    try:
        # 读取good_quality股票列表
        with open(pool_info['file'], 'r', encoding='utf-8') as f:
            symbols = [line.strip() for line in f if line.strip()]
        
        # 更新config_template.json
        config_template_file = "config_template.json"
        if os.path.exists(config_template_file):
            with open(config_template_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 更新默认股票列表 (取前100只作为默认)
            config['trading']['default_symbols'] = symbols[:100]
            
            with open(config_template_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 已更新默认交易股票配置: {len(symbols[:100])} 只股票")
        
        # 创建专门的训练数据配置文件
        training_config = {
            'training_universe': {
                'total_stocks': len(symbols),
                'quality_threshold': pool_info['min_quality'],
                'symbols': symbols,
                'updated_at': datetime.now().isoformat(),
                'data_source': 'auto_crawl_nyse_nasdaq'
            }
        }
        
        with open('exports/training_universe.json', 'w', encoding='utf-8') as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 已生成训练数据配置: exports/training_universe.json")
        
    except Exception as e:
        logger.error(f"❌ 更新默认配置失败: {e}")


def generate_summary_report(exported_pools, stats, logger):
    """生成汇总报告"""
    try:
        report = {
            'update_time': datetime.now().isoformat(),
            'database_stats': stats,
            'exported_pools': exported_pools,
            'recommendations': {
                'for_training': 'good_quality (质量评分 >= 0.6)',
                'for_conservative_trading': 'high_quality (质量评分 >= 0.8)',
                'for_aggressive_trading': 'medium_quality (质量评分 >= 0.4)'
            }
        }
        
        # 保存JSON报告
        with open('exports/stock_universe_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        # 生成人类可读的报告
        readme_content = f"""# 股票池更新报告

## 更新时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据库统计
- 总股票数: {stats.get('total_stocks', 0):,}
- 可交易股票: {stats.get('tradeable_stocks', 0):,}

## 导出的股票池

"""
        
        for pool_name, info in exported_pools.items():
            readme_content += f"### {pool_name.replace('_', ' ').title()}\n"
            readme_content += f"- 股票数量: {info['count']:,}\n"
            readme_content += f"- 质量阈值: {info['min_quality']}\n"
            readme_content += f"- 文件位置: {info['file']}\n\n"
        
        readme_content += """## 使用建议

1. **训练模型**: 使用 `good_quality_stocks.txt` (推荐)
2. **保守交易**: 使用 `high_quality_stocks.txt`
3. **激进交易**: 使用 `medium_quality_stocks.txt`

## 文件说明

- `*_stocks.txt`: 纯股票代码列表
- `*_details.json`: 包含完整股票信息
- `training_universe.json`: 专用训练配置
- `stock_universe_report.json`: 完整报告数据

---
自动生成于股票池管理系统
"""
        
        with open('exports/README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info("✅ 汇总报告已生成: exports/README.md")
        
    except Exception as e:
        logger.error(f"❌ 生成报告失败: {e}")


def main():
    """主函数"""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("美股股票池自动更新程序")
    logger.info("=" * 60)
    
    try:
        success = update_and_export_stock_universe(logger)
        
        if success:
            logger.info("🎉 程序执行成功!")
            logger.info("📁 结果文件位置:")
            logger.info("   - exports/good_quality_stocks.txt (推荐用于训练)")
            logger.info("   - exports/high_quality_stocks.txt (高质量股票)")
            logger.info("   - exports/training_universe.json (训练配置)")
            logger.info("   - exports/README.md (详细报告)")
            
            return 0
        else:
            logger.error("❌ 程序执行失败!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("👋 用户中断程序")
        return 1
    except Exception as e:
        logger.error(f"❌ 程序异常: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    
    print(f"\n程序结束，退出代码: {exit_code}")
    
    if exit_code == 0:
        print("✅ 股票池更新成功! 检查 exports/ 文件夹获取结果")
    else:
        print("❌ 股票池更新失败! 检查日志文件获取详细信息")
    
    input("按回车键退出...")
    sys.exit(exit_code)