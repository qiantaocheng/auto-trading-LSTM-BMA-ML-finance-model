#!/usr/bin/env python3
"""
IBKR自动交易系统 - 统一start器
直接start专业交易GUI，所has功能集成
"""

import asyncio
import logging
from typing import Optional, Any
# 清理：移除未使use导入
# from typing import Dict
# from pathlib import Path


class AutoTraderLauncher:
    """统一交易系统启动器别名"""
    pass

class TradingSystemLauncher:
    """统一交易系统start器"""
    
    def __init__(self):
        self.logger = logging.getLogger("Launcher")
        self._setup_imports()
    
    def _setup_imports(self):
        """settings导入路径，解决相for导入问题"""
        import sys
        import os
        
        # 确保能找toautotrader包
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
        """start图形界面模式"""
        try:
            print("Starting GUI mode...")
            
            from autotrader.app import AutoTraderGUI
            
            # 创建GUI应use
            gui = AutoTraderGUI()
            
            print("GUI mode started successfully")
            print("Tip: Use the interface for trading configuration and monitoring")
            
            # in主线程 startedGUI主循环（Tkinter必须in主线程运行）
            gui.mainloop()
            return gui
            
        except Exception as e:
            print(f"[ERROR] GUI模式startfailed: {e}")
            self.logger.error(f"GUI mode failed: {e}")
            return None
            
    def launch_strategy_mode(self, config_path: Optional[str] = None, **kwargs) -> Any:
        """start策略引擎模式"""
        try:
            print("Starting strategy engine mode...")
            
            from autotrader.config_manager import get_config_manager
            from autotrader.engine import Engine
            from autotrader.ibkr_auto_trader import IbkrAutoTrader
            
            # 使use统一配置管理器
            cfg = get_config_manager()
            
            # 创建底层交易器
            trader = IbkrAutoTrader(config_manager=cfg)
            
            # 创建策略引擎
            engine = Engine(cfg, trader)
            
            print("[OK] 策略引擎start")
            print("[提示] 使use engine.start() starting策略执行")
            
            return engine
            
        except Exception as e:
            print(f"[ERROR] 策略模式startfailed: {e}")
            self.logger.error(f"Strategy mode failed: {e}")
            return None
            
    def launch_direct_mode(self, host: str = "127.0.0.1", port: int = 7497, 
                          client_id: int = 1, **kwargs) -> Any:
        """start直接交易模式"""
        try:
            print(" start直接交易模式...")
            from autotrader.ibkr_auto_trader import IbkrAutoTrader
            from autotrader.config_manager import get_config_manager
            
            # 修复：使use统一配置管理器创建交易器
            config_manager = get_config_manager()
            trader = IbkrAutoTrader(config_manager=config_manager)
            
            print("[OK] 直接交易器start")
            print("[提示] 使use await trader.connect() connectiontoIBKR")
            
            return trader
            
        except Exception as e:
            print(f"[ERROR] 直接模式startfailed: {e}")
            self.logger.error(f"Direct mode failed: {e}")
            return None
            
    async def launch_test_mode(self, **kwargs) -> bool:
        """start系统测试模式"""
        try:
            print(" start系统测试模式...")
            
            # 导入测试
            test_results = {}
            
            print(" 模块导入测试...")
            try:
                from autotrader.ibkr_auto_trader import IbkrAutoTrader
                from autotrader.app import AutoTraderGUI
                from autotrader.engine import Engine
                from autotrader.config_manager import get_config_manager
                test_results['imports'] = True
                print("  [OK] 核心模块导入success")
            except Exception as e:
                test_results['imports'] = False
                print(f"  [ERROR] 模块导入failed: {e}")
                
            print(" 功能组件测试...")
            try:
                # 测试数据库
                from autotrader.database import StockDatabase
                db = StockDatabase()
                configs = db.get_trading_configs()
                test_results['database'] = True
                print(f"  [OK] 数据库connectionsuccess，配置数量: {len(configs)}")
            except Exception as e:
                test_results['database'] = False
                print(f"  [ERROR] 数据库测试failed: {e}")
                
            print(" 组件创建测试...")
            try:
                # 修复：使use正确初始化参数（config_manager）
                from autotrader.config_manager import get_config_manager
                config_manager = get_config_manager()
                trader = IbkrAutoTrader(config_manager=config_manager)
                test_results['trader'] = True
                print("  [OK] 交易器创建success")
            except Exception as e:
                test_results['trader'] = False
                print(f"  [ERROR] 交易器创建failed: {e}")
                
            # 汇总结果
            passed = sum(test_results.values())
            total = len(test_results)
            success_rate = (passed / total) * 100
            
            print()
            print(" 测试结果汇总:")
            for test, result in test_results.items():
                status = "[OK] 通过" if result else "[ERROR] failed"
                print(f"  {test:<15} - {status}")
                
            print(f"\n 总success率: {passed}/{total} ({success_rate:.1f}%)")
            
            if success_rate >= 80:
                print(" 系统测试通过！")
                return True
            else:
                print(" 系统测试发现问题，请check配置")
                return False
                
        except Exception as e:
            print(f"[ERROR] 测试模式failed: {e}")
            self.logger.error(f"Test mode failed: {e}")
            return False
    
    def auto_launcher(self):
        """自动start器 - 直接startGUI模式"""
        self.print_banner()
        print("[start] 正instart专业交易界面...")
        print("[提示] 所has功能（策略引擎、直接交易、系统测试）集成toGUIin")
        print()
        
        try:
            gui = self.launch_gui_mode()
            if gui:
                return gui
            else:
                print("[ERROR] GUIstartfailed，请check系统环境")
                return None
                
        except KeyboardInterrupt:
            print("\n[EXIT] use户取消，再见！")
            return None
        except Exception as e:
            print(f"[ERROR] 系统startfailed: {e}")
            return None


def main():
    """主入口函数 - 直接startGUI"""
    launcher = TradingSystemLauncher()
    return launcher.auto_launcher()


if __name__ == "__main__":
    main()