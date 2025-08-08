#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票池设置脚本
简化的股票池管理界面
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_universe_manager import StockUniverseManager


def setup_logging():
    """设置日志"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'stock_universe.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config():
    """加载配置"""
    config_file = "stock_config.json"
    
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"配置文件 {config_file} 不存在，使用默认配置")
        return {}


def show_menu():
    """显示主菜单"""
    print("\n" + "="*50)
    print("           美股股票池管理系统")
    print("="*50)
    print("1. 🔄 更新股票池数据 (爬取NYSE+NASDAQ)")
    print("2. 📊 查看股票池统计")
    print("3. 🎯 获取交易股票列表")
    print("4. 🔍 搜索股票")
    print("5. 📝 创建自定义股票池")
    print("6. 📋 管理现有股票池")
    print("7. ⚙️  系统设置")
    print("0. 🚪 退出")
    print("="*50)


def update_stock_universe(manager):
    """更新股票池"""
    print("\n正在更新股票池...")
    print("⚠️  注意：首次运行可能需要20-30分钟")
    print("将爬取NYSE和NASDAQ所有股票数据")
    
    confirm = input("确认开始更新? (y/N): ").strip().lower()
    if confirm != 'y':
        print("更新已取消")
        return
    
    print("\n开始更新股票池...")
    try:
        success = manager.crawl_and_update_universe(force_update=True)
        if success:
            print("\n✅ 股票池更新完成!")
            
            # 显示统计信息
            stats = manager.database.get_statistics()
            print(f"\n📊 统计信息:")
            print(f"   总股票数: {stats.get('total_stocks', 0):,}")
            print(f"   可交易股票: {stats.get('tradeable_stocks', 0):,}")
            
            if stats.get('by_exchange'):
                print(f"   交易所分布:")
                for exchange, count in stats['by_exchange'].items():
                    print(f"     {exchange}: {count:,}")
        else:
            print("\n❌ 股票池更新失败!")
            
    except Exception as e:
        print(f"\n❌ 更新过程中出错: {e}")


def show_statistics(manager):
    """显示统计信息"""
    print("\n📊 股票池统计信息")
    print("-" * 40)
    
    try:
        stats = manager.database.get_statistics()
        
        if stats.get('total_stocks', 0) == 0:
            print("⚠️  股票池为空，请先运行更新功能")
            return
        
        print(f"总股票数量: {stats.get('total_stocks', 0):,}")
        print(f"可交易股票: {stats.get('tradeable_stocks', 0):,}")
        
        print(f"\n📈 交易所分布:")
        for exchange, count in stats.get('by_exchange', {}).items():
            print(f"  {exchange}: {count:,}")
        
        print(f"\n🏭 主要行业分布:")
        for sector, count in list(stats.get('top_sectors', {}).items())[:8]:
            print(f"  {sector}: {count:,}")
        
        print(f"\n⭐ 质量分布:")
        for tier, count in stats.get('quality_distribution', {}).items():
            print(f"  {tier}: {count:,}")
            
    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}")


def get_trading_stocks(manager):
    """获取交易股票列表"""
    print("\n🎯 获取交易股票列表")
    print("-" * 30)
    
    try:
        print("请设置筛选条件:")
        
        # 质量评分
        min_quality = input("最低质量评分 (0.0-1.0, 默认0.6): ").strip()
        min_quality = float(min_quality) if min_quality else 0.6
        
        # 最大股票数
        max_stocks = input("最大股票数量 (默认500): ").strip()
        max_stocks = int(max_stocks) if max_stocks else 500
        
        # 获取股票列表
        symbols = manager.get_trading_universe(min_quality, max_stocks)
        
        if not symbols:
            print("⚠️  没有符合条件的股票")
            return
        
        print(f"\n找到 {len(symbols)} 只符合条件的股票:")
        
        # 显示前20只
        print("前20只股票:")
        for i, symbol in enumerate(symbols[:20], 1):
            print(f"{i:2d}. {symbol}")
        
        if len(symbols) > 20:
            print(f"... 还有 {len(symbols) - 20} 只股票")
        
        # 询问是否保存
        save_choice = input(f"\n是否保存为自定义股票池? (y/N): ").strip().lower()
        if save_choice == 'y':
            name = input("股票池名称: ").strip()
            if name:
                description = f"Quality >= {min_quality}, Max {max_stocks} stocks"
                success = manager.create_custom_portfolio(name, symbols, description)
                if success:
                    print(f"✅ 已保存为股票池 '{name}'")
                else:
                    print("❌ 保存失败")
        
        # 显示到文件
        export_choice = input(f"是否导出到文件? (y/N): ").strip().lower()
        if export_choice == 'y':
            filename = f"trading_stocks_{min_quality}_{max_stocks}.txt"
            with open(filename, 'w') as f:
                for symbol in symbols:
                    f.write(f"{symbol}\n")
            print(f"✅ 已导出到 {filename}")
            
    except Exception as e:
        print(f"❌ 获取交易股票失败: {e}")


def search_stocks(manager):
    """搜索股票"""
    print("\n🔍 股票搜索")
    print("-" * 20)
    
    keyword = input("请输入搜索关键字 (股票代码或公司名称): ").strip()
    if not keyword:
        return
    
    try:
        stocks = manager.database.search_stocks(keyword)
        
        if not stocks:
            print("❌ 未找到匹配的股票")
            return
        
        print(f"\n找到 {len(stocks)} 只匹配的股票:")
        print("-" * 80)
        
        for i, stock in enumerate(stocks[:15], 1):  # 显示前15个
            tradeable = "✅" if stock['is_tradeable'] else "❌"
            print(f"{i:2d}. {tradeable} {stock['symbol']:6s} - {stock['name'][:35]:35s}")
            print(f"     💰 ${stock['price']:6.2f}  📊 {stock['sector']:15s}  "
                  f"⭐ {stock['quality_score']:.2f}")
            
            if not stock['is_tradeable'] and stock['exclusion_reasons']:
                reasons = json.loads(stock['exclusion_reasons'])
                if reasons:
                    print(f"     ⚠️  {reasons[0]}")
            print()
            
    except Exception as e:
        print(f"❌ 搜索失败: {e}")


def create_custom_portfolio(manager):
    """创建自定义股票池"""
    print("\n📝 创建自定义股票池")
    print("-" * 25)
    
    name = input("股票池名称: ").strip()
    if not name:
        print("❌ 名称不能为空")
        return
    
    description = input("描述 (可选): ").strip()
    
    print("\n请选择添加方式:")
    print("1. 手动输入股票代码")
    print("2. 从文件导入")
    
    choice = input("选择 (1-2): ").strip()
    
    symbols = []
    
    if choice == '1':
        print("\n请输入股票代码 (每行一个，输入空行结束):")
        while True:
            symbol = input("股票代码: ").strip().upper()
            if not symbol:
                break
            symbols.append(symbol)
            print(f"  已添加: {symbol}")
    
    elif choice == '2':
        filename = input("文件路径: ").strip()
        try:
            with open(filename, 'r') as f:
                for line in f:
                    symbol = line.strip().upper()
                    if symbol:
                        symbols.append(symbol)
            print(f"从文件导入了 {len(symbols)} 个股票代码")
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            return
    
    if not symbols:
        print("⚠️  未添加任何股票")
        return
    
    try:
        success = manager.create_custom_portfolio(name, symbols, description)
        if success:
            print(f"✅ 成功创建股票池 '{name}'，包含 {len(symbols)} 只股票")
        else:
            print("❌ 创建失败")
    except Exception as e:
        print(f"❌ 创建失败: {e}")


def manage_portfolios(manager):
    """管理现有股票池"""
    print("\n📋 管理现有股票池")
    print("-" * 20)
    
    try:
        portfolios = manager.database.list_custom_portfolios()
        
        if not portfolios:
            print("⚠️  暂无自定义股票池")
            return
        
        print("现有股票池:")
        for i, portfolio in enumerate(portfolios, 1):
            print(f"{i}. {portfolio['name']}")
            print(f"   📊 {portfolio['stock_count']} 只股票")
            print(f"   📝 {portfolio['description']}")
            print(f"   🕐 {portfolio['updated_at']}")
            print()
        
        choice = input("选择股票池查看详情 (输入编号): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(portfolios):
                portfolio_name = portfolios[idx]['name']
                symbols = manager.database.get_custom_portfolio(portfolio_name)
                
                if symbols:
                    print(f"\n📋 '{portfolio_name}' 包含的股票:")
                    print("-" * 40)
                    
                    # 分列显示
                    for i in range(0, len(symbols), 5):
                        row_symbols = symbols[i:i+5]
                        print("  ".join(f"{sym:6s}" for sym in row_symbols))
                    
                    print(f"\n总计: {len(symbols)} 只股票")
                    
                    # 导出选项
                    export = input("\n是否导出到文件? (y/N): ").strip().lower()
                    if export == 'y':
                        filename = f"{portfolio_name.replace(' ', '_')}_stocks.txt"
                        with open(filename, 'w') as f:
                            for symbol in symbols:
                                f.write(f"{symbol}\n")
                        print(f"✅ 已导出到 {filename}")
                        
        except (ValueError, IndexError):
            print("❌ 无效选择")
            
    except Exception as e:
        print(f"❌ 管理股票池失败: {e}")


def show_settings():
    """显示系统设置"""
    print("\n⚙️  系统设置")
    print("-" * 15)
    
    config_file = "stock_config.json"
    
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("当前配置:")
        print(f"  数据库文件: {config.get('database_file', 'N/A')}")
        
        quality_config = config.get('quality_filter', {})
        print(f"  最低价格: ${quality_config.get('min_price', 'N/A')}")
        print(f"  最小市值: ${quality_config.get('min_market_cap', 0):,}")
        print(f"  最小成交量: {quality_config.get('min_avg_volume', 0):,}")
        print(f"  最大价差: {quality_config.get('max_bid_ask_spread_pct', 0)}%")
        print(f"  最大波动率: {quality_config.get('max_volatility', 0)}%")
        
    else:
        print("⚠️  配置文件不存在")
    
    print(f"\n配置文件位置: {os.path.abspath(config_file)}")
    print("可以直接编辑该文件来修改设置")


def main():
    """主函数"""
    setup_logging()
    
    print("正在初始化股票池管理系统...")
    
    try:
        # 创建必要的目录
        Path("data").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # 加载配置
        config = load_config()
        
        # 创建管理器
        manager = StockUniverseManager(config)
        
        print("✅ 初始化完成!")
        
        while True:
            show_menu()
            choice = input("\n请选择操作: ").strip()
            
            if choice == '0':
                print("👋 再见!")
                break
            elif choice == '1':
                update_stock_universe(manager)
            elif choice == '2':
                show_statistics(manager)
            elif choice == '3':
                get_trading_stocks(manager)
            elif choice == '4':
                search_stocks(manager)
            elif choice == '5':
                create_custom_portfolio(manager)
            elif choice == '6':
                manage_portfolios(manager)
            elif choice == '7':
                show_settings()
            else:
                print("❌ 无效选择，请重试")
            
            input("\n按回车键继续...")
    
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，再见!")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        logging.error(f"System error: {e}")


if __name__ == "__main__":
    main()