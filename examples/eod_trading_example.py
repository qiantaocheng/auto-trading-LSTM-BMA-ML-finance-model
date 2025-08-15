#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EOD交易系统集成示例
展示如何使用新的EOD调度器和增强订单执行器
"""

import asyncio
import logging
import yaml
from datetime import datetime
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """主函数 - EOD交易系统示例"""
    print("=== EOD Trading System Example ===")
    
    try:
        # 1. 加载配置
        config_path = Path("D:/trade/config/eod_trading_config.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"[OK] 配置加载完成: {config['strategy']['execution']['open_order_type']} 模式")
        
        # 2. 创建EOD配置对象
        import sys
        sys.path.append('D:/trade')
        from autotrader.eod_scheduler import EODConfig, create_eod_scheduler
        
        eod_config = EODConfig(
            enabled=config['strategy']['double_trend']['eod_mode'],
            run_after_close_min=config['strategy']['double_trend']['run_after_close_min'],
            open_order_type=config['strategy']['execution']['open_order_type'],
            avoid_open_auction=config['strategy']['execution']['avoid_open_auction'],
            limit_band_by_atr_mult=config['strategy']['execution']['limit_band_by_atr_mult'],
            limit_band_floor_pct=config['strategy']['execution']['limit_band_floor_pct'],
            limit_band_cap_pct=config['strategy']['execution']['limit_band_cap_pct'],
            cancel_if_not_filled_minutes=config['strategy']['execution']['cancel_if_not_filled_minutes'],
            atr_trailing_enabled=config['risk_management']['atr_trailing']['enabled'],
            atr_trailing_period=config['risk_management']['atr_trailing']['period'],
            atr_trailing_multiplier=config['risk_management']['atr_trailing']['multiplier'],
            atr_trailing_activate_after_R=config['risk_management']['atr_trailing']['activate_after_R']
        )
        
        # 3. 创建EOD调度器
        scheduler = create_eod_scheduler(eod_config)
        print(f"[OK] EOD调度器创建完成")
        
        # 4. 模拟组件设置（实际使用中应该是真实组件）
        print("[OK] 设置模拟组件...")
        
        class MockDataFeed:
            async def fetch_daily_bars(self, symbol, lookback):
                # 返回模拟数据
                import pandas as pd
                import numpy as np
                dates = pd.date_range('2024-01-01', periods=lookback, freq='D')
                return pd.DataFrame({
                    'high': np.random.uniform(100, 110, lookback),
                    'low': np.random.uniform(90, 100, lookback),
                    'close': np.random.uniform(95, 105, lookback),
                    'volume': np.random.uniform(1e6, 5e6, lookback)
                }, index=dates)
        
        class MockSignalEngine:
            async def on_daily_bar_close(self, symbol, bars):
                # 返回模拟信号
                import random
                if random.random() > 0.7:  # 30%概率生成信号
                    return {
                        'action': random.choice(['BUY', 'SELL']),
                        'confidence': random.uniform(0.6, 1.0),
                        'reason': f'Mock signal for {symbol}'
                    }
                return None
        
        class MockOrderExecutor:
            def __init__(self):
                self.logger = logging.getLogger("MockOrderExecutor")
            
            async def place_open_order(self, symbol, side, quantity, limit_price=None, order_type="LOO"):
                self.logger.info(f"[MOCK] 开盘单: {symbol} {side} {quantity} @ {limit_price} ({order_type})")
                return type('Trade', (), {'order': type('Order', (), {'orderId': 12345})()})()
            
            async def place_limit_rth(self, symbol, side, quantity, limit_price, cancel_after_min=30):
                self.logger.info(f"[MOCK] RTH限价单: {symbol} {side} {quantity} @ {limit_price}")
                return type('Trade', (), {'order': type('Order', (), {'orderId': 12346})()})()
            
            async def update_server_stop(self, symbol, new_stop, quantity=None):
                self.logger.info(f"[MOCK] 更新止损: {symbol} @ {new_stop}")
            
            async def eod_update_trailing_stop(self, symbol, side, entry_price, current_close, 
                                             atr_value, initial_stop, activate_after_R=1.0, atr_mult=2.0):
                self.logger.info(f"[MOCK] EOD移动止损: {symbol} {side} entry={entry_price} close={current_close}")
        
        class MockPositionManager:
            async def get_all_positions(self):
                return {
                    'AAPL': {'quantity': 100, 'side': 'BUY', 'avg_price': 150.0},
                    'MSFT': {'quantity': 50, 'side': 'BUY', 'avg_price': 300.0}
                }
        
        # 5. 设置依赖组件
        data_feed = MockDataFeed()
        signal_engine = MockSignalEngine()
        order_executor = MockOrderExecutor()
        position_manager = MockPositionManager()
        
        scheduler.set_dependencies(data_feed, signal_engine, order_executor, position_manager)
        
        # 6. 测试EOD信号生成（模拟收盘后执行）
        print("\\n--- 测试EOD信号生成 ---")
        await scheduler._execute_eod_signal_generation()
        
        print(f"生成的开盘计划数量: {len(scheduler.pending_opening_plans)}")
        for plan in scheduler.pending_opening_plans:
            print(f"  计划: {plan.symbol} {plan.side} {plan.quantity}股 @ ${plan.limit_price:.2f}")
        
        # 7. 测试EOD移动止损更新
        print("\\n--- 测试EOD移动止损更新 ---")
        # 设置一些模拟持仓状态
        scheduler.position_states = {
            'AAPL': {
                'entry_price': 145.0,
                'initial_stop': 143.55,  # 1%止损
                'side': 'BUY',
                'atr_value': 2.5
            },
            'MSFT': {
                'entry_price': 295.0,
                'initial_stop': 292.05,  # 1%止损  
                'side': 'BUY',
                'atr_value': 4.0
            }
        }
        
        await scheduler._execute_eod_trailing_stops()
        
        # 8. 测试开盘订单执行
        print("\\n--- 测试开盘订单执行 ---")
        await scheduler._execute_opening_orders()
        
        print(f"\\n[SUCCESS] EOD交易系统示例完成")
        print(f"  - 处理了{len(scheduler._get_trading_universe())}只股票")
        print(f"  - 生成了{len(scheduler.pending_opening_plans)}个开盘计划")
        print(f"  - 配置模式: {eod_config.open_order_type}")
        
        # 9. 展示如何启动完整的EOD任务调度（实际运行时使用）
        print("\\n--- EOD任务调度示例 ---")
        print("在实际运行中，可以这样启动EOD任务调度:")
        print("await scheduler.start_eod_tasks()  # 这将按配置的时间自动执行")
        
    except Exception as e:
        print(f"[ERROR] 示例执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())