#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成交易系统 - 整合所有优化组件
将连接管理、实时数据、订单管理、风险控制、性能优化和监控仪表板整合在一起
"""

import os
import sys
import time
import logging
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import argparse

# 导入自定义模块
try:
    from connection_manager import ConnectionManager, AlertManager
    from realtime_market_data import RealTimeDataProcessor, EventDrivenStrategyEngine
    from enhanced_order_manager import EnhancedOrderManager, OrderStatus
    from enhanced_risk_manager import EnhancedRiskManager, RiskCheckResult
    from performance_optimizer import PerformanceOptimizer
    from monitoring_dashboard import MonitoringSystem
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all module files are in the same directory")
    sys.exit(1)

try:
    from ib_insync import *
    IB_INSYNC_AVAILABLE = True
except ImportError:
    IB_INSYNC_AVAILABLE = False
    print("Warning: ib_insync not available, running in simulation mode")


class IntegratedTradingSystem:
    """集成交易系统"""
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
        # 系统状态
        self.is_running = False
        self.system_start_time = None
        
        # 核心组件
        self.ib = None
        self.alert_manager = None
        self.connection_manager = None
        self.data_processor = None
        self.strategy_engine = None
        self.order_manager = None
        self.risk_manager = None
        self.performance_optimizer = None
        self.monitoring_system = None
        
        # 交易策略
        self.active_strategies = {}
        self.subscribed_symbols = set()
        
        # 统计信息
        self.system_stats = {
            'start_time': None,
            'total_signals': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'risk_rejections': 0,
            'connection_events': 0
        }
    
    def _load_config(self, config_file: str = None) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            # IBKR连接配置
            'ibkr': {
                'host': '127.0.0.1',
                'port': 4002,
                'client_id': 1,
                'auto_reconnect': True,
                'max_reconnect_attempts': 10,
                'reconnect_delay_seconds': 30,
                'heartbeat_interval_seconds': 30
            },
            
            # 告警配置
            'alerts': {
                'email': {
                    'enabled': False,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'from_email': '',
                    'to_email': ''
                },
                'alert_cooldown_minutes': 5
            },
            
            # 风险管理配置
            'risk': {
                'max_portfolio_risk': 0.02,
                'max_position_size': 0.05,
                'max_sector_exposure': 0.25,
                'max_daily_loss': 0.05,
                'max_drawdown': 0.10,
                'max_new_positions_per_day': 10,
                'max_trades_per_symbol_per_day': 3,
                'loss_cooldown_days': 3,
                'min_time_between_trades_minutes': 15
            },
            
            # 订单管理配置
            'orders': {
                'auto_retry': True,
                'retry_delay_seconds': 5,
                'order_timeout_seconds': 300,
                'save_orders': True,
                'orders_file': 'orders/order_history.json'
            },
            
            # 性能优化配置
            'performance': {
                'cache': {
                    'max_memory_mb': 512,
                    'default_ttl_minutes': 30,
                    'persistence_enabled': True
                },
                'downloader': {
                    'max_concurrent_requests': 5,
                    'request_timeout_seconds': 30,
                    'retry_attempts': 3,
                    'rate_limit_per_second': 50
                }
            },
            
            # 监控配置
            'monitoring': {
                'collector': {
                    'update_interval_seconds': 5
                },
                'dashboard': {
                    'host': '127.0.0.1',
                    'port': 5000,
                    'debug': False
                }
            },
            
            # 交易配置
            'trading': {
                'default_symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                'auto_start_trading': False,
                'simulation_mode': False
            },
            
            # 日志配置
            'logging': {
                'level': 'INFO',
                'file': 'logs/integrated_system.log',
                'max_size_mb': 100,
                'backup_count': 5
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # 深度合并配置
                def deep_merge(default, user):
                    for key, value in user.items():
                        if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                            deep_merge(default[key], value)
                        else:
                            default[key] = value
                
                deep_merge(default_config, user_config)
                print(f"Loaded configuration from {config_file}")
                
            except Exception as e:
                print(f"Error loading config file {config_file}: {e}")
                print("Using default configuration")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        log_config = self.config['logging']
        
        # 创建日志目录
        log_file = log_config['file']
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 配置日志
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        logger = logging.getLogger('IntegratedTradingSystem')
        logger.info("Logging system initialized")
        
        return logger
    
    def initialize_system(self) -> bool:
        """初始化系统组件"""
        try:
            self.logger.info("Initializing integrated trading system...")
            
            # 1. 初始化告警管理器
            self.alert_manager = AlertManager(self.config['alerts'])
            self.logger.info("Alert manager initialized")
            
            # 2. 初始化连接管理器
            if IB_INSYNC_AVAILABLE and not self.config['trading']['simulation_mode']:
                self.connection_manager = ConnectionManager(self.config['ibkr'], self.alert_manager)
                self._setup_connection_callbacks()
                self.logger.info("Connection manager initialized")
            else:
                self.logger.warning("Running in simulation mode - no IBKR connection")
            
            # 3. 初始化风险管理器
            self.risk_manager = EnhancedRiskManager(self.config['risk'], self.logger)
            self.logger.info("Risk manager initialized")
            
            # 4. 初始化订单管理器
            if self.connection_manager:
                self.order_manager = EnhancedOrderManager(
                    self.connection_manager.ib if self.connection_manager else None,
                    self.config['orders'],
                    self.logger
                )
                self._setup_order_callbacks()
                self.logger.info("Order manager initialized")
            
            # 5. 初始化实时数据处理器
            if self.connection_manager:
                self.data_processor = RealTimeDataProcessor(
                    self.connection_manager.ib if self.connection_manager else None,
                    self.logger
                )
                self.strategy_engine = EventDrivenStrategyEngine(self.data_processor, self.logger)
                self.logger.info("Real-time data processor initialized")
            
            # 6. 初始化性能优化器
            if self.connection_manager:
                self.performance_optimizer = PerformanceOptimizer(
                    self.connection_manager.ib if self.connection_manager else None,
                    self.config['performance']
                )
                self.logger.info("Performance optimizer initialized")
            
            # 7. 初始化监控系统
            self.monitoring_system = MonitoringSystem(self.config['monitoring'])
            self.monitoring_system.set_trading_components(
                risk_manager=self.risk_manager,
                order_manager=self.order_manager,
                connection_manager=self.connection_manager,
                data_processor=self.data_processor
            )
            self.logger.info("Monitoring system initialized")
            
            self.logger.info("All system components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing system: {e}")
            return False
    
    def _setup_connection_callbacks(self):
        """设置连接回调"""
        if not self.connection_manager:
            return
        
        def on_connected(ib):
            self.logger.info("Connected to IBKR")
            self.system_stats['connection_events'] += 1
            
            # 订阅默认股票数据
            default_symbols = self.config['trading']['default_symbols']
            if default_symbols and self.data_processor:
                self.data_processor.subscribe_market_data(default_symbols)
                self.subscribed_symbols.update(default_symbols)
        
        def on_disconnected():
            self.logger.warning("Disconnected from IBKR")
            self.system_stats['connection_events'] += 1
        
        def on_reconnected(ib):
            self.logger.info("Reconnected to IBKR")
            self.system_stats['connection_events'] += 1
            
            # 重新订阅数据
            if self.subscribed_symbols and self.data_processor:
                self.data_processor.subscribe_market_data(list(self.subscribed_symbols))
        
        def on_connection_failed():
            self.logger.error("Connection to IBKR failed permanently")
            self.system_stats['connection_events'] += 1
        
        self.connection_manager.set_callbacks(
            on_connected=on_connected,
            on_disconnected=on_disconnected,
            on_reconnected=on_reconnected,
            on_connection_failed=on_connection_failed
        )
    
    def _setup_order_callbacks(self):
        """设置订单回调"""
        if not self.order_manager:
            return
        
        def on_order_filled(order_record):
            self.logger.info(f"Order filled: {order_record.symbol} {order_record.action} {order_record.filled_quantity}")
            self.system_stats['successful_trades'] += 1
            
            # 记录交易到风险管理器
            if self.risk_manager:
                self.risk_manager.record_trade(
                    order_record.symbol,
                    order_record.action,
                    order_record.filled_quantity,
                    order_record.avg_fill_price or 0,
                    order_record.strategy_name
                )
        
        def on_order_rejected(order_record):
            self.logger.warning(f"Order rejected: {order_record.symbol} - {order_record.error_message}")
            self.system_stats['failed_trades'] += 1
        
        self.order_manager.add_callback(OrderStatus.FILLED, on_order_filled)
        self.order_manager.add_callback(OrderStatus.REJECTED, on_order_rejected)
    
    def start_system(self) -> bool:
        """启动系统"""
        if self.is_running:
            self.logger.warning("System is already running")
            return True
        
        try:
            self.logger.info("Starting integrated trading system...")
            self.system_start_time = datetime.now()
            self.system_stats['start_time'] = self.system_start_time
            
            # 1. 启动连接
            if self.connection_manager:
                if not self.connection_manager.connect():
                    self.logger.error("Failed to connect to IBKR")
                    if not self.config['trading']['simulation_mode']:
                        return False
            
            # 2. 启动监控系统
            if self.monitoring_system:
                self.monitoring_system.start()
                dashboard_url = self.monitoring_system.get_dashboard_url()
                self.logger.info(f"Monitoring dashboard available at: {dashboard_url}")
            
            # 3. 注册示例策略
            if self.strategy_engine:
                self._register_sample_strategies()
            
            # 4. 自动开始交易（如果配置了）
            if self.config['trading']['auto_start_trading']:
                self._start_trading()
            
            self.is_running = True
            self.logger.info("Integrated trading system started successfully")
            
            # 发送启动通知
            if self.alert_manager:
                self.alert_manager.send_alert(
                    'system_start',
                    '交易系统启动',
                    f'集成交易系统在 {self.system_start_time.strftime("%Y-%m-%d %H:%M:%S")} 成功启动\n'
                    f'监控仪表板: {dashboard_url if self.monitoring_system else "N/A"}',
                    'info'
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting system: {e}")
            return False
    
    def stop_system(self):
        """停止系统"""
        if not self.is_running:
            return
        
        try:
            self.logger.info("Stopping integrated trading system...")
            
            # 1. 停止交易
            self._stop_trading()
            
            # 2. 停止监控
            if self.monitoring_system:
                self.monitoring_system.stop()
            
            # 3. 清理组件
            if self.data_processor:
                self.data_processor.cleanup()
            
            if self.order_manager:
                self.order_manager.cleanup()
            
            if self.risk_manager:
                self.risk_manager.cleanup()
            
            if self.performance_optimizer:
                self.performance_optimizer.cleanup()
            
            # 4. 断开连接
            if self.connection_manager:
                self.connection_manager.disconnect()
            
            self.is_running = False
            uptime = datetime.now() - self.system_start_time if self.system_start_time else timedelta(0)
            
            self.logger.info(f"System stopped. Uptime: {uptime}")
            
            # 发送停止通知
            if self.alert_manager:
                self.alert_manager.send_alert(
                    'system_stop',
                    '交易系统停止',
                    f'集成交易系统已停止\n'
                    f'运行时间: {uptime}\n'
                    f'统计信息: {self.system_stats}',
                    'info'
                )
            
        except Exception as e:
            self.logger.error(f"Error stopping system: {e}")
    
    def _register_sample_strategies(self):
        """注册示例策略"""
        def mean_reversion_strategy(event_type: str, symbol: str, data: Dict):
            """均值回归策略示例"""
            try:
                if event_type == 'indicator':
                    indicators = data['indicators']
                    
                    # 获取技术指标
                    rsi = indicators.get('rsi')
                    bb_position = indicators.get('bb_position')
                    zscore = indicators.get('zscore')
                    
                    if not all([rsi, bb_position, zscore]):
                        return
                    
                    # 获取当前价格
                    latest_data = self.data_processor.get_latest_data(symbol)
                    current_price = latest_data.get('last')
                    
                    if not current_price:
                        return
                    
                    # 生成交易信号
                    signal = None
                    quantity = 100  # 基础数量
                    
                    # 买入信号：超卖
                    if rsi < 30 and bb_position < 0.2 and zscore < -2:
                        signal = ('BUY', quantity, current_price, 'Mean reversion - oversold')
                    
                    # 卖出信号：超买
                    elif rsi > 70 and bb_position > 0.8 and zscore > 2:
                        signal = ('SELL', quantity, current_price, 'Mean reversion - overbought')
                    
                    if signal:
                        self._execute_strategy_signal(symbol, *signal)
                        self.system_stats['total_signals'] += 1
                        
            except Exception as e:
                self.logger.error(f"Error in mean reversion strategy for {symbol}: {e}")
        
        # 注册策略
        default_symbols = self.config['trading']['default_symbols']
        self.strategy_engine.register_strategy(
            'mean_reversion',
            default_symbols,
            mean_reversion_strategy
        )
        
        self.logger.info("Sample strategies registered")
    
    def _execute_strategy_signal(self, symbol: str, action: str, quantity: int, 
                                price: float, reason: str):
        """执行策略信号"""
        try:
            # 风险检查
            if self.risk_manager:
                risk_check = self.risk_manager.pre_trade_check(
                    symbol, action, quantity, price, 'integrated_system'
                )
                
                if risk_check.result == RiskCheckResult.REJECTED:
                    self.logger.warning(f"Trade rejected by risk manager: {symbol} {action} {quantity} - {risk_check.reasons}")
                    self.system_stats['risk_rejections'] += 1
                    return
                
                elif risk_check.result == RiskCheckResult.SCALED_DOWN:
                    self.logger.info(f"Trade scaled down: {quantity} -> {risk_check.approved_quantity}")
                    quantity = risk_check.approved_quantity
                
                if risk_check.warnings:
                    self.logger.warning(f"Risk warnings for {symbol}: {risk_check.warnings}")
            
            # 提交订单
            if self.order_manager and quantity > 0:
                order_id = self.order_manager.submit_market_order(
                    symbol, action, quantity, 'integrated_system', reason
                )
                
                self.logger.info(f"Strategy signal executed: {order_id} - {symbol} {action} {quantity} @ {price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error executing strategy signal: {e}")
    
    def _start_trading(self):
        """开始交易"""
        self.logger.info("Starting automated trading")
    
    def _stop_trading(self):
        """停止交易"""
        self.logger.info("Stopping automated trading")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'running': self.is_running,
            'start_time': self.system_start_time.isoformat() if self.system_start_time else None,
            'uptime_seconds': (datetime.now() - self.system_start_time).total_seconds() if self.system_start_time else 0,
            'stats': self.system_stats.copy(),
            'components': {}
        }
        
        # 组件状态
        if self.connection_manager:
            status['components']['connection'] = self.connection_manager.get_status()
        
        if self.risk_manager:
            status['components']['risk'] = self.risk_manager.get_risk_status()
        
        if self.order_manager:
            status['components']['orders'] = self.order_manager.get_order_statistics()
        
        if self.data_processor:
            status['components']['data_quality'] = self.data_processor.get_data_quality_report()
        
        if self.performance_optimizer:
            status['components']['performance'] = self.performance_optimizer.get_performance_report()
        
        return status
    
    def run(self):
        """运行系统主循环"""
        if not self.initialize_system():
            self.logger.error("Failed to initialize system")
            return False
        
        if not self.start_system():
            self.logger.error("Failed to start system")
            return False
        
        try:
            self.logger.info("System is running. Press Ctrl+C to stop.")
            
            # 主循环
            while self.is_running:
                time.sleep(1)
                
                # 定期状态检查（每分钟）
                if int(time.time()) % 60 == 0:
                    status = self.get_system_status()
                    self.logger.debug(f"System status: Running={status['running']}, Uptime={status['uptime_seconds']:.0f}s")
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.stop_system()
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Integrated Trading System')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Log level')
    parser.add_argument('--simulation', action='store_true', 
                       help='Run in simulation mode (no real IBKR connection)')
    
    args = parser.parse_args()
    
    # 创建系统实例
    system = IntegratedTradingSystem(args.config)
    
    # 覆盖配置
    if args.simulation:
        system.config['trading']['simulation_mode'] = True
    
    if args.log_level:
        system.config['logging']['level'] = args.log_level
        # 重新设置日志级别
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 运行系统
    success = system.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())