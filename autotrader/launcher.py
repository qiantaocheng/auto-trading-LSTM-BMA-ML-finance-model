#!/usr/bin/env python3
"""
IBKR自动交易系统 - 统一启动器
直接启动专业交易GUI，所有功能已集成
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path


class TradingSystemLauncher:
    """统一交易系统启动器"""
    
    def __init__(self):
        self.logger = logging.getLogger("Launcher")
        self._setup_imports()
    
    def _setup_imports(self):
        """设置导入路径，解决相对导入问题"""
        import sys
        import os
        
        # 确保能找到autotrader包
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
    @staticmethod
    def print_banner():
        """显示系统横幅"""
        print("\n" + "="*60)
        print("    IBKR Professional Trading System v1.0")
        print("    Unified Professional Trading Interface")
        print("    策略引擎 | 直接交易 | 风险管理 | 系统监控")
        print("="*60)
        print()
        

    def launch_gui_mode(self, **kwargs) -> Any:
        """启动图形界面模式"""
        try:
            print("Starting GUI mode...")
            
            from autotrader.app import AutoTraderGUI
            
            # 创建GUI应用
            gui = AutoTraderGUI()
            
            print("GUI mode started successfully")
            print("Tip: Use the interface for trading configuration and monitoring")
            
            # 在主线程中启动GUI主循环（Tkinter必须在主线程运行）
            gui.mainloop()
            return gui
            
        except Exception as e:
            print(f"❌ GUI模式启动失败: {e}")
            self.logger.error(f"GUI mode failed: {e}")
            return None
            
    def launch_strategy_mode(self, config_path: Optional[str] = None, **kwargs) -> Any:
        """启动策略引擎模式"""
        try:
            print("🧠 启动策略引擎模式...")
            
            from autotrader.config import HotConfig
            from autotrader.engine import Engine
            from autotrader.ibkr_auto_trader import IbkrAutoTrader
            
            # 加载配置
            cfg = HotConfig(config_path) if config_path else HotConfig()
            
            # 创建底层交易器
            trader = IbkrAutoTrader("127.0.0.1", 7497, 1)
            
            # 创建策略引擎
            engine = Engine(cfg, trader)
            
            print("✅ 策略引擎已启动")
            print("💡 提示: 使用 engine.start() 开始策略执行")
            
            return engine
            
        except Exception as e:
            print(f"❌ 策略模式启动失败: {e}")
            self.logger.error(f"Strategy mode failed: {e}")
            return None
            
    def launch_direct_mode(self, host: str = "127.0.0.1", port: int = 7497, 
                          client_id: int = 1, **kwargs) -> Any:
        """启动直接交易模式"""
        try:
            print("⚡ 启动直接交易模式...")
            from autotrader.ibkr_auto_trader import IbkrAutoTrader
            
            # 创建直接交易器
            trader = IbkrAutoTrader(host, port, client_id)
            
            print("✅ 直接交易器已启动")
            print("💡 提示: 使用 await trader.connect() 连接到IBKR")
            
            return trader
            
        except Exception as e:
            print(f"❌ 直接模式启动失败: {e}")
            self.logger.error(f"Direct mode failed: {e}")
            return None
            
    async def launch_test_mode(self, **kwargs) -> bool:
        """启动系统测试模式"""
        try:
            print("🔍 启动系统测试模式...")
            
            # 导入测试
            test_results = {}
            
            print("📋 模块导入测试...")
            try:
                from autotrader.ibkr_auto_trader import IbkrAutoTrader
                from autotrader.app import AutoTraderGUI
                from autotrader.engine import Engine
                from autotrader.config import HotConfig
                test_results['imports'] = True
                print("  ✅ 核心模块导入成功")
            except Exception as e:
                test_results['imports'] = False
                print(f"  ❌ 模块导入失败: {e}")
                
            print("🔧 功能组件测试...")
            try:
                # 测试数据库
                from autotrader.database import StockDatabase
                db = StockDatabase()
                configs = db.get_trading_configs()
                test_results['database'] = True
                print(f"  ✅ 数据库连接成功，配置数量: {len(configs)}")
            except Exception as e:
                test_results['database'] = False
                print(f"  ❌ 数据库测试失败: {e}")
                
            print("🎯 组件创建测试...")
            try:
                trader = IbkrAutoTrader("127.0.0.1", 7497, 1)
                test_results['trader'] = True
                print("  ✅ 交易器创建成功")
            except Exception as e:
                test_results['trader'] = False
                print(f"  ❌ 交易器创建失败: {e}")
                
            # 汇总结果
            passed = sum(test_results.values())
            total = len(test_results)
            success_rate = (passed / total) * 100
            
            print()
            print("📊 测试结果汇总:")
            for test, result in test_results.items():
                status = "✅ 通过" if result else "❌ 失败"
                print(f"  {test:<15} - {status}")
                
            print(f"\n🎯 总成功率: {passed}/{total} ({success_rate:.1f}%)")
            
            if success_rate >= 80:
                print("🎉 系统测试通过！")
                return True
            else:
                print("⚠️ 系统测试发现问题，请检查配置")
                return False
                
        except Exception as e:
            print(f"❌ 测试模式失败: {e}")
            self.logger.error(f"Test mode failed: {e}")
            return False
    
    def auto_launcher(self):
        """自动启动器 - 直接启动GUI模式"""
        self.print_banner()
        print("🚀 正在启动专业交易界面...")
        print("💡 提示: 所有功能（策略引擎、直接交易、系统测试）已集成到GUI中")
        print()
        
        try:
            gui = self.launch_gui_mode()
            if gui:
                return gui
            else:
                print("❌ GUI启动失败，请检查系统环境")
                return None
                
        except KeyboardInterrupt:
            print("\n👋 用户取消，再见！")
            return None
        except Exception as e:
            print(f"❌ 系统启动失败: {e}")
            return None


def main():
    """主入口函数 - 直接启动GUI"""
    launcher = TradingSystemLauncher()
    return launcher.auto_launcher()


if __name__ == "__main__":
    main()