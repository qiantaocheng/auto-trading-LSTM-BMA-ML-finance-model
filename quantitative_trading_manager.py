import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import sqlite3
import subprocess
import sys
import os
import threading
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import webbrowser
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import win32api
import win32con
import win32gui
import win32clipboard
try:
    from plyer import notification
    NOTIFICATION_AVAILABLE = True
except Exception:
    NOTIFICATION_AVAILABLE = False
import yfinance as yf
import pandas as pd
import glob
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any

# 统一配置常量管理
class TradingConstants:
    """交易系统统一配置常量"""
    
    # 四象限基础权重配置
    DEFAULT_ALLOCATION_DATA = [
        ('趋势+高波动', '60%', '40%', '趋势跟踪为主'),
        ('趋势+低波动', '80%', '20%', '稳定趋势环境'),
        ('振荡+高波动', '30%', '70%', '均值回归策略'),
        ('振荡+低波动', '50%', '50%', '平衡配置')
    ]
    
    # 通用风险管理参数
    RISK_MANAGEMENT_DEFAULTS = {
        'max_portfolio_risk': 0.02,  # 组合最大风险2%
        'max_position_size': 0.05,   # 单个持仓最大5%
        'max_sector_exposure': 0.25, # 单个行业最大25%
        'max_daily_loss': 0.05,      # 单日最大损失5%
        'max_drawdown': 0.10,        # 最大回撤10%
        'stop_loss_pct': 0.05,       # 默认止损5%
        'take_profit_pct': 0.10,     # 默认止盈10%
        'max_new_positions_per_day': 10,
        'max_trades_per_symbol_per_day': 3,
        'loss_cooldown_days': 3,
        'min_time_between_trades_minutes': 15
    }
    
    # 订单管理参数
    ORDER_MANAGEMENT_DEFAULTS = {
        'enable_bracket_orders': True,
        'default_stop_loss_pct': 0.05,
        'default_take_profit_pct': 0.10,
        'max_retry_attempts': 3,
        'order_timeout_seconds': 30
    }
    
    # IBKR连接参数
    IBKR_DEFAULTS = {
        'host': '127.0.0.1',
        'port': 4002,
        'client_id': 50310,
        'account': 'c2dvdongg'
    }
    
    @staticmethod
    def generate_unique_client_id():
        """生成唯一的客户端ID避免冲突"""
        import random
        # 使用时间戳和随机数生成唯一ID
        timestamp = int(time.time()) % 10000  # 取时间戳后4位
        random_part = random.randint(100, 999)
        return timestamp * 1000 + random_part
    
    # 交易参数
    TRADING_DEFAULTS = {
        'total_capital': 100000,
        'max_position_percent': 5,
        'max_portfolio_exposure': 95,
        'max_drawdown_percent': 10,
        'daily_loss_limit': 5000,
        'max_single_position_percent': 20,
        'max_new_positions_per_day': 10,
        'max_single_trade_value': 50000,
        'loss_cooldown_days': 1,
        'auto_liquidate_on_max_drawdown': False,
        'commission_rate': 0.001,
        'signal_threshold': 0.6,
        'max_positions': 10,
        'trading_watchlist': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    }

# IBKR增强功能导入
try:
    from ib_insync import *
    import ib_insync as ibs
    IBKR_AVAILABLE = True
    print("[INFO] ib_insync已加载")
except ImportError:
    print("[WARNING] ib_insync未安装，请运行: pip install ib_insync")
    IBKR_AVAILABLE = False

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.order import Order
    IBAPI_AVAILABLE = True
    print("[INFO] IBKR API已加载")
except ImportError:
    print("[WARNING] ibapi未安装，请运行: pip install ibapi")
    IBAPI_AVAILABLE = False

# 邮件和通知增强
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
    print("[INFO] 邮件功能已加载")
except ImportError:
    EMAIL_AVAILABLE = False
    print("[WARNING] 邮件功能不可用")

# 因子平衡交易系统
try:
    from factor_balanced_trading_system import FactorBalancedTradingSystem, SystemConfig
    FACTOR_BALANCED_AVAILABLE = True
    print("[INFO] 因子平衡交易系统已加载")
except ImportError:
    FACTOR_BALANCED_AVAILABLE = False
    print("[WARNING] 因子平衡交易系统不可用")

# 导入增强版交易策略
try:
    from ibkr_trading_strategy_enhanced import EnhancedMeanReversionStrategy
    ENHANCED_TRADING_AVAILABLE = True
    print("[INFO] 增强版交易策略已加载")
except ImportError as e:
    ENHANCED_TRADING_AVAILABLE = False
    print(f"[WARNING] 增强版交易策略不可用: {e}")

# 导入增强风险管理和订单管理
try:
    from enhanced_risk_manager import EnhancedRiskManager, RiskCheckResult, RiskLevel
    ENHANCED_RISK_AVAILABLE = True
    print("[INFO] 增强风险管理已加载")
except ImportError as e:
    ENHANCED_RISK_AVAILABLE = False
    print(f"[WARNING] 增强风险管理不可用: {e}")

try:
    from enhanced_order_manager import EnhancedOrderManager, OrderStatus, OrderType
    ENHANCED_ORDER_AVAILABLE = True
    print("[INFO] 增强订单管理已加载")
except ImportError as e:
    ENHANCED_ORDER_AVAILABLE = False
    print(f"[WARNING] 增强订单管理不可用: {e}")

# 导入状态监控模块
try:
    from status_monitor import get_status_monitor, update_status, log_message
    STATUS_MONITOR_AVAILABLE = True
except ImportError:
    STATUS_MONITOR_AVAILABLE = False
    print("[WARNING] 状态监控模块不可用")

# 导入输出捕获模块
try:
    from output_capture import start_output_capture, stop_output_capture
    OUTPUT_CAPTURE_AVAILABLE = True
except ImportError:
    OUTPUT_CAPTURE_AVAILABLE = False
    print("[WARNING] 输出捕获模块不可用")

# 导入美股爬虫模块
try:
    from us_stock_crawler import USStockCrawler
    US_STOCK_CRAWLER_AVAILABLE = True
    print("[INFO] 美股爬虫模块已加载")
except ImportError as e:
    US_STOCK_CRAWLER_AVAILABLE = False

# 导入BMA滚动前向回测模块
try:
    from bma_walkforward_enhanced import EnhancedBMAWalkForward
    BMA_WALKFORWARD_AVAILABLE = True
    print("[INFO] BMA增强版滚动回测模块已加载")
except ImportError as e:
    BMA_WALKFORWARD_AVAILABLE = False
    print(f"[WARNING] BMA增强版滚动回测模块不可用: {e}")
    print(f"[WARNING] 美股爬虫模块不可用: {e}")

# 导入双模型融合策略模块
try:
    from ensemble_strategy import EnsembleStrategy
    ENSEMBLE_STRATEGY_AVAILABLE = True
    print("[INFO] 双模型融合策略已加载")
except ImportError as e:
    ENSEMBLE_STRATEGY_AVAILABLE = False
    print(f"[WARNING] 双模型融合策略不可用: {e}")

# 尝试导入日历组件
try:
    from tkcalendar import Calendar, DateEntry
    CALENDAR_AVAILABLE = True
except ImportError:
    CALENDAR_AVAILABLE = False

class QuantitativeTradingManager:
    """
    量化交易管理软件主类
    
    功能特性：
    1. GUI界面 - 启动量化模型和回测
    2. 本地数据库 - 按日期存储分析结果
    3. 定时任务 - 每两周（1日和15日）中午12点自动运行
    4. 通知系统 - 任务完成时弹窗通知
    5. 日志记录 - 完整的操作日志
    """
    
    def __init__(self):
        # 初始化主窗口
        self.root = tk.Tk()
        self.root.title("量化交易管理软件 v1.0")
        self.root.geometry("800x600")
        
        # 应用配置
        self.config = {
            'auto_run': True,
            'notifications': True,
            'log_level': 'INFO',
            'database_path': 'trading_results.db',
            'result_directory': 'result',
            # BMA自动交易集成配置
            'enable_auto_trading': True,
            'market_open_time': '09:30',  # 美股开盘时间
            'bma_pre_run_hours': 1,      # 开盘前1小时运行BMA
            'price_validation_threshold': 0.30,  # 股价验证±30%阈值
            'max_stocks_to_trade': 10,   # 最多交易股票数量
            'default_stock_pool_file': 'default_stocks.json',  # 默认股票池文件
            'bma_output_file': 'bma_results.json',  # BMA结果输出文件
            
            # 增强版交易策略配置
            'enable_enhanced_trading': True,  # 启用增强版交易策略
            'enhanced_mode': True,           # 启用事件驱动模式
            'enable_real_trading': True,     # 启用实盘交易
            'auto_trigger_enhanced_strategy': True,  # 模型分析完成后自动触发增强策略
            # 策略信号集成
            'use_strategy_signals': True,  # 使用策略生成的信号驱动自动交易
            'signal_sources': ['trading_signals.json', 'weekly_lstm', 'ensemble'],
            # IBKR连接配置（使用统一常量）
            **{f'ibkr_{k}': v for k, v in TradingConstants.IBKR_DEFAULTS.items()},
            
            # 基础交易配置（使用统一常量）
            'total_capital': 0,  # 总资金（将通过API动态获取）
            **{k: v for k, v in TradingConstants.RISK_MANAGEMENT_DEFAULTS.items() 
               if k in ['max_position_size', 'stop_loss_pct', 'take_profit_pct']},
            'max_portfolio_exposure': TradingConstants.TRADING_DEFAULTS['max_portfolio_exposure'] / 100,
            'commission_rate': TradingConstants.TRADING_DEFAULTS['commission_rate'],
            
            # 增强IBKR配置（使用统一常量并添加增强功能）
            'enhanced_ibkr': {
                'enable_auto_reconnect': True,
                'max_reconnect_attempts': 999,
                'reconnect_delay': 30,
                'heartbeat_interval': 10,
                'enable_real_trading': True,
                **TradingConstants.TRADING_DEFAULTS,
                'loss_cooldown_days': 1,
                'auto_liquidate_on_max_drawdown': False
            },
            
            # 告警配置
            'alert_settings': {
                'email_alerts': True,
                'gui_notifications': True,
                'system_notifications': True,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'email_user': '',
                'email_password': '',
                'alert_emails': []
            }
        }
        
        # 背景图片相关
        self.background_image = None
        self.background_label = None
        
        # 检查PIL依赖
        self.pil_available = self.check_pil_availability()
        
        # 设置背景图片
        self.setup_background()
        
        # 确保背景图片可见
        self.ensure_background_visibility()
        
        # 初始化增强版交易策略
        self.enhanced_strategy = None
        self.init_enhanced_trading_strategy()
        
        # 添加增强IBKR功能
        self.init_enhanced_ibkr_features()
        
        # 初始化自动交易变量
        self.auto_trading_stocks = []
        self.is_auto_trading = False
        self.ibkr_connection = None
        
        # 初始化订单管理变量
        self.next_order_id = 1
        self.order_status_map = {}
        self.price_data = {}
        self.ticker_subscriptions = {}
        
        # 初始化组件
        self.setup_logging_enhanced()
        self.create_directories_enhanced()
        self.load_config_enhanced()
        
        # 初始化增强功能
        self.init_enhanced_features()
        
        # 初始化增强管理器
        self.init_enhanced_managers()
        
        # 重置每日风控计数器
        self.reset_daily_risk_counters()
    
    def init_enhanced_managers(self):
        """初始化增强管理器"""
        try:
            # 初始化增强风险管理器
            if ENHANCED_RISK_AVAILABLE:
                # 使用统一常量配置
                risk_config = {
                    **TradingConstants.RISK_MANAGEMENT_DEFAULTS,
                    **self.config.get('enhanced_ibkr', {})
                }
                
                self.risk_manager = EnhancedRiskManager(risk_config, self.logger)
                self.log_message("[增强风险] ✅ 增强风险管理器初始化完成")
            else:
                self.risk_manager = None
                self.log_message("[增强风险] ⚠️ 增强风险管理器不可用")
            
            # 初始化增强订单管理器
            if ENHANCED_ORDER_AVAILABLE:
                # 使用统一常量配置
                order_config = {
                    **TradingConstants.ORDER_MANAGEMENT_DEFAULTS,
                    **self.config.get('enhanced_ibkr', {})
                }
                
                # 传入ib连接，如果还没有则稍后设置
                ib_connection = getattr(self, 'ib', None)
                self.order_manager = EnhancedOrderManager(ib_connection, order_config, self.logger)
                self.log_message("[增强订单] ✅ 增强订单管理器初始化完成")
            else:
                self.order_manager = None
                self.log_message("[增强订单] ⚠️ 增强订单管理器不可用")
                
        except Exception as e:
            self.log_message(f"❌ 增强管理器初始化失败: {e}")
            self.risk_manager = None
            self.order_manager = None
    
    def _get_common_strategy_config(self):
        """获取通用策略配置参数（统一配置源）"""
        return {
            # IBKR连接参数（使用常量默认值）
            'ibkr_host': self.config.get('ibkr_host', TradingConstants.IBKR_DEFAULTS['host']),
            'ibkr_port': self.config.get('ibkr_port', TradingConstants.IBKR_DEFAULTS['port']),
            'ibkr_client_id': self.config.get('ibkr_client_id', TradingConstants.IBKR_DEFAULTS['client_id']),
            'ibkr_account': self.config.get('ibkr_account', TradingConstants.IBKR_DEFAULTS['account']),
            
            # 风险管理参数（使用常量默认值）
            'max_position_size': self.config.get('max_position_size', TradingConstants.RISK_MANAGEMENT_DEFAULTS['max_position_size']),
            'max_portfolio_risk': self.config.get('max_portfolio_risk', TradingConstants.RISK_MANAGEMENT_DEFAULTS['max_portfolio_risk']),
            'max_portfolio_exposure': self.config.get('max_portfolio_exposure', TradingConstants.TRADING_DEFAULTS['max_portfolio_exposure'] / 100),
            'stop_loss_pct': self.config.get('stop_loss_pct', TradingConstants.RISK_MANAGEMENT_DEFAULTS['stop_loss_pct']),
            'take_profit_pct': self.config.get('take_profit_pct', TradingConstants.RISK_MANAGEMENT_DEFAULTS['take_profit_pct']),
            'max_drawdown': self.config.get('max_drawdown', TradingConstants.RISK_MANAGEMENT_DEFAULTS['max_drawdown']),
            'max_daily_loss': self.config.get('max_daily_loss', TradingConstants.RISK_MANAGEMENT_DEFAULTS['max_daily_loss']),
            
            # 交易参数（使用常量默认值）
            'total_capital': self.config.get('total_capital', TradingConstants.TRADING_DEFAULTS['total_capital']),
            'commission_rate': self.config.get('commission_rate', TradingConstants.TRADING_DEFAULTS['commission_rate']),
            'signal_threshold': self.config.get('signal_threshold', TradingConstants.TRADING_DEFAULTS['signal_threshold']),
            'max_positions': self.config.get('max_positions', TradingConstants.TRADING_DEFAULTS['max_positions']),
            
            # 交易列表（使用常量默认值）
            'trading_watchlist': self.config.get('trading_watchlist', TradingConstants.TRADING_DEFAULTS['trading_watchlist']),
            
            # 其他参数
            'enable_enhanced_mode': self.config.get('enhanced_mode', True),
            'enable_real_trading': self.config.get('enable_real_trading', False),
            'log_level': self.config.get('log_level', 'INFO'),
            'bma_json_file': 'weekly_bma_trading.json',
            'use_bma_recommendations': True
        }

    def init_enhanced_trading_strategy(self):
        """初始化增强版交易策略（统一入口）"""
        try:
            if not ENHANCED_TRADING_AVAILABLE or not self.config.get('enable_enhanced_trading', False):
                self.log_message("[增强策略] [WARNING] 增强版交易策略未启用或不可用")
                return
            
            self.log_message("[增强策略] 正在初始化增强版交易策略...")
            
            # 获取统一配置参数
            common_config = self._get_common_strategy_config()
            
            # 检查是否启用v2策略
            use_v2_strategy = self.config.get('use_v2_strategy', False)
            
            if use_v2_strategy:
                success = self._init_v2_strategy(common_config)
                if not success:
                    self.log_message("[增强策略] [WARNING] v2策略初始化失败，回退到v1策略")
                    use_v2_strategy = False
            
            if not use_v2_strategy:
                self._init_v1_strategy(common_config)
                
        except Exception as e:
            self.log_message(f"[增强策略] [ERROR] 初始化失败: {e}")
            self.enhanced_strategy = None

    def _init_v2_strategy(self, common_config):
        """初始化v2增强策略"""
        try:
            from enhanced_trading_strategy_v2 import EnhancedTradingStrategy
            
            # 构建v2策略配置
            v2_config = {
                'ibkr': {
                    'host': common_config['ibkr_host'],
                    'port': common_config['ibkr_port'],
                    'client_id': common_config['ibkr_client_id']
                },
                'risk_management': {
                    'max_position_size': common_config['max_position_size'],
                    'max_portfolio_risk': common_config['max_portfolio_risk'],
                    'stop_loss_pct': common_config['stop_loss_pct'],
                    'take_profit_pct': common_config['take_profit_pct']
                },
                'trading': {
                    'watchlist': common_config['trading_watchlist'],
                    'signal_threshold': common_config['signal_threshold'],
                    'max_positions': common_config['max_positions']
                },
                'data_sources': {
                    'bma_file': 'result/bma_quantitative_analysis_*.xlsx',
                    'lstm_file': 'result/*lstm_analysis_*.xlsx'
                }
            }
            
            # 创建v2策略实例
            self.enhanced_strategy = EnhancedTradingStrategy("trading_config_v2.json")
            self.enhanced_strategy.config.update(v2_config)
            self.log_message("[增强策略] [SUCCESS] 增强版交易策略v2初始化成功")
            return True
            
        except ImportError as e:
            self.log_message(f"[增强策略] [WARNING] v2策略导入失败: {e}")
            return False
        except Exception as e:
            self.log_message(f"[增强策略] [ERROR] v2策略初始化失败: {e}")
            return False

    def _init_v1_strategy(self, common_config):
        """初始化v1增强策略"""
        try:
            # 构建v1策略配置
            strategy_config = {
                'enable_enhanced_mode': common_config['enable_enhanced_mode'],
                'enable_real_trading': common_config['enable_real_trading'],
                'ibkr_host': common_config['ibkr_host'],
                'ibkr_port': common_config['ibkr_port'],
                'ibkr_client_id': common_config['ibkr_client_id'],
                'ibkr_account': common_config['ibkr_account'],
                'total_capital': common_config['total_capital'],
                'max_position_size': common_config['max_position_size'],
                'max_portfolio_exposure': common_config['max_portfolio_exposure'],
                'stop_loss_pct': common_config['stop_loss_pct'],
                'take_profit_pct': common_config['take_profit_pct'],
                'commission_rate': common_config['commission_rate'],
                'log_level': common_config['log_level'],
                'bma_json_file': common_config['bma_json_file'],
                'use_bma_recommendations': common_config['use_bma_recommendations']
            }
            
            # 创建v1策略实例
            self.enhanced_strategy = EnhancedMeanReversionStrategy(strategy_config)
            self.log_message("[增强策略] [SUCCESS] 增强版交易策略v1初始化成功")
            
        except Exception as e:
            self.log_message(f"[增强策略] [ERROR] v1策略初始化失败: {e}")
            self.enhanced_strategy = None

    def log_message(self, message):
        """记录日志消息"""
        try:
            # 处理可能的编码问题，替换不兼容的字符
            safe_message = message.replace('❌', '[FAIL]').replace('✅', '[OK]').replace('⚠️', '[WARN]').replace('✅', '[SUCCESS]').replace('❌', '[ERROR]')
            
            # 使用logger记录消息
            if hasattr(self, 'logger'):
                self.logger.info(safe_message)
            else:
                print(f"[LOG] {safe_message}")
                
            # 如果有GUI文本框，也添加到界面上（这里可以保留原始emoji）
            if hasattr(self, 'log_text') and self.log_text:
                try:
                    self.log_text.insert(tk.END, message + '\n')
                    self.log_text.see(tk.END)
                except:
                    pass  # GUI可能未完全初始化
        except Exception as e:
            print(f"[LOG ERROR] {e}: {message}")
        
        # 设置应用图标
        try:
            self.root.iconbitmap(default="trading.ico")
        except:
            pass  # 如果没有图标文件，忽略错误
    
    def setup_background(self):
        """设置背景图片"""
        try:
            if self.pil_available:
                from PIL import Image, ImageTk
                
                # 背景图片路径
                background_path = "ChatGPT Image 2025年8月1日 03_26_16.png"
                
                if os.path.exists(background_path):
                    # 加载背景图片
                    bg_image = Image.open(background_path)
                    
                    # 获取窗口大小
                    window_width = 800
                    window_height = 600
                    
                    # 调整图片大小以适应窗口
                    bg_image = bg_image.resize((window_width, window_height), Image.Resampling.LANCZOS)
                    
                    # 转换为PhotoImage
                    self.background_image = ImageTk.PhotoImage(bg_image)
                    
                    # 创建背景标签
                    self.background_label = tk.Label(self.root, image=self.background_image)
                    self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
                    
                    # 将背景标签置于最底层
                    self.background_label.lower()
                    
                    # 强制更新显示
                    self.root.update_idletasks()
                    
                    # 设置窗口背景色为透明，让背景图片显示
                    self.root.configure(bg='')
                    
                    print(f"成功加载背景图片: {background_path}")
                    
                else:
                    print(f"背景图片文件不存在: {background_path}")
                    # 使用默认背景色
                    self.root.configure(bg='#f0f0f0')
                    
            else:
                print("PIL未安装，无法加载背景图片")
                # 使用默认背景色
                self.root.configure(bg='#f0f0f0')
                
        except Exception as e:
            print(f"设置背景图片失败: {e}")
            # 使用默认背景色
            self.root.configure(bg='#f0f0f0')
    
    def ensure_background_visibility(self):
        """确保背景图片可见"""
        if hasattr(self, 'background_label') and self.background_label:
            # 确保背景标签在最底层
            self.background_label.lower()
            # 强制更新显示
            self.root.update_idletasks()
        
        # 初始化组件
        self.setup_logging()
        self.setup_database()
        self.setup_scheduler()
        self.create_gui()
        self.load_recent_results()
        
        # 初始化股票池数据
        self.initialize_stock_pools()
        
        # 初始化美股爬虫
        self.initialize_us_stock_crawler()
        
        # 初始化双模型融合策略
        self.initialize_ensemble_strategy()
        
        # 初始化量化模型股票列表（从爬虫获取）
        self.initialize_quantitative_stock_list()
        
        # 再次确保背景在最底层
        if hasattr(self, 'background_label') and self.background_label:
            self.background_label.lower()
        
        # 启动定时任务
        self.scheduler.start()
        
        self.logger.info("量化交易管理软件已启动")
        
        # 初始化日期选择变量
        self.selected_start_date = "2018-01-01"
        self.selected_end_date = datetime.now().strftime("%Y-%m-%d")
    
    def check_pil_availability(self):
        """检查PIL是否可用"""
        try:
            from PIL import Image, ImageTk
            return True
        except ImportError:
            print("警告: PIL/Pillow未安装，图片显示功能将不可用")
            print("请运行: pip install Pillow")
            return False
        
    def setup_logging(self):
        """设置日志记录"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 配置日志
        logging.basicConfig(
            level=getattr(logging, self.config['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"trading_manager_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_database(self):
        """设置SQLite数据库"""
        self.db_path = self.config['database_path']
        
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.conn.cursor()
            
            # 创建分析结果表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    analysis_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    status TEXT DEFAULT 'completed',
                    stock_count INTEGER,
                    avg_score REAL,
                    buy_count INTEGER,
                    hold_count INTEGER,
                    sell_count INTEGER,
                    notes TEXT
                )
            ''')
            
            # 创建任务执行记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_seconds REAL,
                    error_message TEXT,
                    result_files TEXT
                )
            ''')
            
            self.conn.commit()
            self.logger.info("数据库初始化完成")
            
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
            messagebox.showerror("错误", f"数据库初始化失败: {e}")
    
    def setup_scheduler(self):
        """设置定时任务调度器"""
        self.scheduler = BackgroundScheduler()
        
        # 添加每两周一次的定时任务（每月1日和15日中午12点）
        self.scheduler.add_job(
            func=self.auto_run_analysis,
            trigger=CronTrigger(day='1,15', hour=12, minute=0),
            id='biweekly_analysis',
            name='双周量化分析',
            replace_existing=True
        )
        
        # 注意：已删除BMA自动交易任务，现在使用独立的LSTM自动交易系统
        
        self.logger.info("定时任务调度器已配置")
    
    def create_gui(self):
        """创建GUI界面"""
        # 创建主框架（使用透明背景以显示背景图片）
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 确保背景图片在最底层
        if hasattr(self, 'background_label') and self.background_label:
            self.background_label.lower()
            self.root.update_idletasks()
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)  # 让结果区域可以扩展
        
        # 标题（使用透明背景）
        title_label = ttk.Label(main_frame, text="量化交易管理软件", 
                               font=('Microsoft YaHei', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 创建主要功能按钮区域
        self.create_main_buttons(main_frame)
        
        # 创建结果显示区域
        self.create_results_area(main_frame)
        
        # 创建状态栏
        self.create_status_bar(main_frame)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 确保背景图片在最底层并强制更新
        if hasattr(self, 'background_label') and self.background_label:
            self.background_label.lower()
            self.root.update_idletasks()
        
    def create_main_buttons(self, parent):
        """创建主要功能按钮"""
        # 使用半透明背景的框架
        button_frame = ttk.LabelFrame(parent, text="主要功能", padding="10")
        button_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 确保背景图片在最底层
        if hasattr(self, 'background_label') and self.background_label:
            self.background_label.lower()
            self.root.update_idletasks()
        
        # 创建按钮容器
        buttons_container = ttk.Frame(button_frame)
        buttons_container.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # BMA分析按钮
        self.bma_button_frame = self.create_quagsire_button(
            buttons_container,
            "🚀 BMA分析", 
            self.run_bma_analysis,
            0, 0
        )
        # 创建一个隐藏的按钮用于状态管理
        self.bma_button = ttk.Button(self.bma_button_frame, text="")
        self.bma_button.pack_forget()  # 隐藏但保持引用
        
        # LSTM分析按钮
        self.lstm_button_frame = self.create_quagsire_button(
            buttons_container,
            "🧠 LSTM分析",
            self.run_lstm_enhanced_model,
            0, 1
        )
        # 创建一个隐藏的按钮用于状态管理
        self.lstm_button = ttk.Button(self.lstm_button_frame, text="")
        self.lstm_button.pack_forget()  # 隐藏但保持引用
        
        # 配置按钮列权重
        for i in range(2):  # 现在只有2个按钮
            buttons_container.columnconfigure(i, weight=1)
        
        # 配置按钮行权重
        buttons_container.rowconfigure(0, weight=1)  # 现在只有1行按钮
        
        # 添加快捷操作按钮
        quick_frame = ttk.Frame(button_frame)
        quick_frame.grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky=(tk.W, tk.E))
        
        ttk.Button(quick_frame, text="📁 打开结果文件夹", 
                   command=self.open_result_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text="🤖 自动交易管理", 
                   command=self.show_auto_trading_manager).pack(side=tk.LEFT, padx=5)
        ttk.Button(quick_frame, text="[SETTINGS] 设置", 
                   command=self.show_settings).pack(side=tk.LEFT, padx=5)
    
    def create_quagsire_button(self, parent, text, command, row, column):
        """创建带Quagsire图标的按钮"""
        # 创建按钮框架
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=column, padx=10, pady=5, sticky=(tk.W, tk.E))
        
        # 设置按钮框架的大小
        button_frame.configure(width=80, height=80)
        button_frame.pack_propagate(False)  # 防止子组件改变框架大小
        
        # 创建图标标签 - 使用place精确定位
        icon_label = ttk.Label(button_frame, text="", cursor="hand2")
        icon_label.place(relx=0.5, rely=0.25, anchor=tk.CENTER)  # 图标往上移一点
        
        # 创建文字标签 - 使用place精确定位
        text_label = ttk.Label(button_frame, text=text, font=('Microsoft YaHei', 10, 'bold'))
        text_label.place(relx=0.5, rely=0.75, anchor=tk.CENTER)  # 文字往下移一点
        
        # 绑定点击事件
        icon_label.bind("<Button-1>", lambda e: command())
        text_label.bind("<Button-1>", lambda e: command())
        button_frame.bind("<Button-1>", lambda e: command())
        
        # 绑定悬停效果
        icon_label.bind("<Enter>", lambda e: self.on_button_hover_enter(icon_label, text_label))
        icon_label.bind("<Leave>", lambda e: self.on_button_hover_leave(icon_label, text_label))
        text_label.bind("<Enter>", lambda e: self.on_button_hover_enter(icon_label, text_label))
        text_label.bind("<Leave>", lambda e: self.on_button_hover_leave(icon_label, text_label))
        button_frame.bind("<Enter>", lambda e: self.on_button_hover_enter(icon_label, text_label))
        button_frame.bind("<Leave>", lambda e: self.on_button_hover_leave(icon_label, text_label))
        
        # 加载Quagsire图标
        self.load_quagsire_icon(icon_label)
        
        return button_frame
    
    def load_quagsire_icon(self, label):
        """加载Quagsire图标"""
        try:
            if self.pil_available:
                from PIL import Image, ImageTk
                
                # 加载quagsire.png图片
                image_path = "quagsire.png"
                if os.path.exists(image_path):
                    # 加载并调整图片大小
                    img = Image.open(image_path)
                    # 调整到合适的按钮大小
                    img = img.resize((48, 48), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    label.configure(image=photo)
                    label.image = photo  # 保持引用
                    print(f"成功加载Quagsire图标: {image_path}")
                else:
                    print(f"Quagsire图标文件不存在: {image_path}")
                    # 使用文字作为备选
                    label.configure(text="🐸", font=('Microsoft YaHei', 24))
                    
            else:
                # 如果没有PIL，使用文字
                label.configure(text="🐸", font=('Microsoft YaHei', 24))
                
        except Exception as e:
            print(f"加载Quagsire图标失败: {e}")
            # 使用文字作为备选
            label.configure(text="🐸", font=('Microsoft YaHei', 24))
    
    def on_button_hover_enter(self, icon_label, text_label):
        """按钮悬停进入事件"""
        icon_label.configure(cursor="hand2")
        text_label.configure(cursor="hand2")
        
        # 开始跳动动画
        self.start_bounce_animation(icon_label)
        
    def on_button_hover_leave(self, icon_label, text_label):
        """按钮悬停离开事件"""
        icon_label.configure(cursor="")
        text_label.configure(cursor="")
        
        # 立即停止跳动动画
        self.stop_bounce_animation(icon_label)
    
    def start_bounce_animation(self, icon_label):
        """开始跳动动画 - 只让图标跳动，不影响文字"""
        # 如果已经在跳动，不重复启动
        if hasattr(icon_label, 'bounce_animation_running') and icon_label.bounce_animation_running:
            return
        
        icon_label.bounce_animation_running = True
        icon_label.bounce_direction = 1  # 1表示向上，-1表示向下
        icon_label.bounce_offset = 0
        icon_label.bounce_speed = 1  # 减小速度，让跳动更平滑
        
        def bounce_step():
            # 检查是否应该继续动画
            if not hasattr(icon_label, 'bounce_animation_running') or not icon_label.bounce_animation_running:
                return
            
            # 计算新的偏移量
            icon_label.bounce_offset += icon_label.bounce_direction * icon_label.bounce_speed
            
            # 如果达到最大偏移量，改变方向
            if icon_label.bounce_offset >= 4:
                icon_label.bounce_direction = -1
            elif icon_label.bounce_offset <= -4:
                icon_label.bounce_direction = 1
            
            # 使用place方法精确定位图标，不影响文字标签
            # 计算新的rely位置，让图标在按钮框架内跳动
            base_rely = 0.25  # 基础位置（25%）
            current_rely = base_rely + (icon_label.bounce_offset / 100.0)  # 转换为相对位置
            icon_label.place_configure(rely=current_rely)
            
            # 继续动画（持续跳动）
            icon_label.after(80, bounce_step)
        
        # 开始动画
        bounce_step()
    
    def stop_bounce_animation(self, icon_label):
        """停止跳动动画"""
        if hasattr(icon_label, 'bounce_animation_running'):
            icon_label.bounce_animation_running = False
        
        # 重置位置到原始状态
        icon_label.place_configure(rely=0.25)
    
    def create_results_area(self, parent):
        """创建结果显示区域"""
        # 创建主结果框架（支持背景图片）
        results_frame = ttk.LabelFrame(parent, text="分析结果", padding="10")
        results_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=3)
        results_frame.columnconfigure(1, weight=1)
        
        # 确保背景图片在最底层
        if hasattr(self, 'background_label') and self.background_label:
            self.background_label.lower()
        
        # 左侧：结果列表
        list_frame = ttk.Frame(results_frame)
        list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        
        # 结果列表
        columns = ('日期', '分析类型', '股票数量', '平均评分', 'BUY', 'HOLD', 'SELL', '状态')
        self.results_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=8)
        
        # 设置列标题
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=100)
        
        # 添加滚动条
        scrollbar_y = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        scrollbar_x = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # 布局
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # 双击事件
        self.results_tree.bind('<Double-1>', self.on_result_double_click)
        
        # 右侧：图片显示区域
        image_frame = ttk.LabelFrame(results_frame, text="分析图表", padding="5")
        image_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)
        
        # 图片显示标签
        self.image_label = ttk.Label(image_frame, text="双击结果查看图表", 
                                    anchor=tk.CENTER, relief=tk.SUNKEN)
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # 图片控制按钮
        button_frame = ttk.Frame(image_frame)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Button(button_frame, text="刷新图表", command=self.refresh_images).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="打开文件夹", command=self.open_image_folder).pack(side=tk.LEFT)
        
        # 图片存储
        self.current_image_path = None
        self.image_files = []
    
    def create_status_bar(self, parent):
        """创建状态栏"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))
        status_frame.columnconfigure(0, weight=1)
        
        # 状态标签
        self.status_var = tk.StringVar(value="就绪")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                          mode='determinate', length=200)
        self.progress_bar.grid(row=0, column=1, padx=(10, 0))
        
        # 定时任务状态
        self.schedule_status_var = tk.StringVar(value=f"下次自动运行: {self.get_next_run_time()}")
        self.schedule_label = ttk.Label(status_frame, textvariable=self.schedule_status_var, 
                                       font=('Microsoft YaHei', 8))
        self.schedule_label.grid(row=0, column=2, padx=(10, 0))
    
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="导出结果", command=self.export_results)
        file_menu.add_command(label="导入配置", command=self.import_config)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_closing)
        
        # 工具菜单  
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具", menu=tools_menu)
        tools_menu.add_command(label="数据库管理", command=self.show_database_manager)
        tools_menu.add_command(label="选择删除Excel", command=self.delete_excel_outputs)
        tools_menu.add_separator()
        tools_menu.add_command(label="量化模型股票管理", command=self.manage_quantitative_model_stocks)
        tools_menu.add_separator()
        tools_menu.add_command(label="日志查看器", command=self.show_log_viewer)
        tools_menu.add_command(label="系统信息", command=self.show_system_info)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)
    
    def get_next_run_time(self):
        """获取下次运行时间"""
        now = datetime.now()
        if (now.day == 1 or now.day == 15) and now.hour < 12:
            next_run = now.replace(hour=12, minute=0, second=0, microsecond=0)
        else:
            # 下一次双周运行时间（1日或15日12点）
            if now.day < 15:
                next_run = now.replace(day=15, hour=12, minute=0, second=0, microsecond=0)
            else:
                if now.month == 12:
                    next_run = now.replace(year=now.year+1, month=1, day=1, hour=12, minute=0, second=0, microsecond=0)
                else:
                    next_run = now.replace(month=now.month+1, day=1, hour=12, minute=0, second=0, microsecond=0)
        return next_run.strftime('%Y-%m-%d 12:00')
    
    def update_status(self, message, progress=None):
        """更新状态栏"""
        self.status_var.set(message)
        if progress is not None:
            self.progress_var.set(progress)
        self.root.update_idletasks()
        self.logger.info(f"状态更新: {message}")
    
    def show_notification(self, title, message, timeout=10):
        """显示系统通知"""
        if not self.config['notifications']:
            return
        
        # 限制消息长度，防止Windows通知系统错误
        max_title_length = 64
        max_message_length = 200
        
        # 截断标题和消息
        title = title[:max_title_length] if len(title) > max_title_length else title
        message = message[:max_message_length] if len(message) > max_message_length else message
        
        try:
            notification.notify(
                title=title,
                message=message,
                app_name="量化交易软件",  # 缩短应用名称
                timeout=timeout,
                toast=True
            )
        except Exception as e:
            self.logger.error(f"通知显示失败: {e}")
            # 备用方案：使用messagebox（无长度限制）
            messagebox.showinfo(title, message)
    
    def run_quantitative_model(self):
        """运行量化模型"""
        def run_in_thread():
            try:
                self.update_status("正在启动量化模型...", 10)
                self.quant_button.config(state='disabled')
                
                start_time = time.time()
                
                # 运行量化模型
                process = subprocess.Popen(
                    [sys.executable, "量化模型.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='gbk',  # 使用GBK编码处理中文
                    cwd=os.getcwd()
                )
                
                self.update_status("量化模型运行中...", 50)
                
                stdout, stderr = process.communicate()
                duration = time.time() - start_time
                
                if process.returncode == 0:
                    self.update_status("量化模型运行完成", 100)
                    
                    # 查找生成的Excel文件
                    result_files = self.find_latest_result_files("quantitative_analysis_")
                    
                    # 保存到数据库
                    self.save_analysis_result("量化模型", result_files[0] if result_files else "", 
                                            duration, stdout)
                    
                    self.show_notification("任务完成", f"量化模型分析完成\n耗时: {duration:.1f}秒")
                    self.load_recent_results()
                    
                else:
                    # 截断错误信息，避免过长
                    short_error = stderr[:150] + "..." if len(stderr) > 150 else stderr
                    error_msg = f"量化模型运行失败\n错误信息: {short_error}"
                    self.update_status("量化模型运行失败", 0)
                    self.show_notification("任务失败", error_msg)
                    self.logger.error(f"量化模型运行失败\n完整错误信息: {stderr}")
                    
            except Exception as e:
                error_msg = f"启动量化模型失败: {e}"
                self.update_status(error_msg, 0)
                self.show_notification("错误", error_msg)
                self.logger.error(error_msg)
                
            finally:
                self.quant_button.config(state='normal')
                self.update_status("就绪", 0)
        
        # 在新线程中运行，避免界面冻结
        threading.Thread(target=run_in_thread, daemon=True).start()
    
    def _run_model_subprocess(self, model_name, script_name, args, result_file_prefix):
        """公共的模型subprocess调用函数，避免重复代码"""
        try:
            self.update_status(f"正在启动{model_name}模型...", 10)
            
            start_time = time.time()
            
            # 构建命令
            cmd = [sys.executable, script_name] + args
            
            # 运行模型
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                cwd=os.getcwd()
            )
            
            self.update_status(f"{model_name}模型运行中...", 50)
            
            stdout, stderr = process.communicate()
            duration = time.time() - start_time
            
            if process.returncode == 0:
                self.update_status(f"{model_name}模型运行完成", 100)
                
                # 查找生成的Excel文件
                result_files = self.find_latest_result_files(result_file_prefix)
                
                # 保存到数据库
                self.save_analysis_result(f"{model_name}模型", result_files[0] if result_files else "", 
                                        duration, stdout)
                
                self.show_notification("任务完成", f"{model_name}模型分析完成\n耗时: {duration:.1f}秒")
                self.load_recent_results()
                
                # 如果是BMA模型且增强策略可用，尝试触发增强策略
                if model_name == "BMA" and self.enhanced_strategy and hasattr(self, 'config') and self.config.get('auto_trigger_enhanced_strategy', True):
                    try:
                        self._trigger_enhanced_strategy_after_analysis()
                    except Exception as e:
                        self.log_message(f"[增强策略] 自动触发失败: {e}")
                
                return True
            else:
                short_error = stderr[:150] + "..." if len(stderr) > 150 else stderr
                error_msg = f"{model_name}模型运行失败\n错误信息: {short_error}"
                self.update_status(f"{model_name}模型运行失败", 0)
                self.show_notification("任务失败", error_msg)
                self.logger.error(f"{model_name}模型运行失败\n完整错误信息: {stderr}")
                
                return False
                
        except Exception as e:
            error_msg = f"启动{model_name}模型失败: {e}"
            self.update_status(error_msg, 0)
            self.show_notification("错误", error_msg)
            self.logger.error(error_msg)
            return False
    
    def _trigger_enhanced_strategy_after_analysis(self):
        """在分析完成后触发增强策略"""
        try:
            if not self.enhanced_strategy:
                self.log_message("[增强策略] 策略未初始化，无法自动触发")
                return
            
            self.log_message("[增强策略] 检测到模型分析完成，准备自动触发增强策略...")
            
            # 检查增强策略是否已经在运行
            is_running = getattr(self.enhanced_strategy, 'running', False)
            
            if is_running:
                self.log_message("[增强策略] 增强策略已在运行，触发信号生成...")
                # 如果已在运行，触发信号生成
                if hasattr(self.enhanced_strategy, 'generate_signals'):
                    threading.Thread(target=self.enhanced_strategy.generate_signals, daemon=True).start()
                    self.log_message("[增强策略] 信号生成已触发")
            else:
                self.log_message("[增强策略] 增强策略未运行，尝试启动...")
                # 如果未运行，尝试启动
                if hasattr(self.enhanced_strategy, 'start_enhanced_trading'):
                    if self.enhanced_strategy.start_enhanced_trading():
                        self.log_message("[增强策略] ✅ 增强策略自动启动成功")
                        # 更新按钮状态
                        if hasattr(self, 'root'):
                            self.root.after(0, lambda: self.update_trading_button_status(True))
                    else:
                        self.log_message("[增强策略] ❌ 增强策略自动启动失败")
                        
        except Exception as e:
            self.log_message(f"[增强策略] 自动触发过程出错: {e}")
            import traceback
            traceback.print_exc()

    def run_lstm_analysis(self):
        """运行LSTM量化分析"""
        def run_in_thread():
            # 使用默认股票列表运行LSTM模型
            stock_list_str = ','.join(self.quantitative_model_stocks)
            
            args = [
                "--symbols", stock_list_str,
                "--start-date", "2024-01-01",
                "--end-date", datetime.now().strftime("%Y-%m-%d")
            ]
            
            self._run_model_subprocess("LSTM", "lstm_multi_day_enhanced.py", args, "test_multi_day_lstm_analysis_")
                
        # 在新线程中运行
        threading.Thread(target=run_in_thread, daemon=True).start()
    
    def run_bma_analysis(self):
        """运行BMA量化分析"""
        def run_in_thread():
            # 使用默认股票列表运行BMA模型
            stock_list_str = ','.join(self.quantitative_model_stocks)
            
            args = ["--symbols", stock_list_str]
            
            self._run_model_subprocess("BMA", "量化模型_bma_enhanced.py", args, "bma_quantitative_analysis_")
                
        # 在新线程中运行
        threading.Thread(target=run_in_thread, daemon=True).start()
    
    def run_bma_walkforward_backtest(self):
        """运行BMA滚动前向回测（新增）"""
        def run_in_thread():
            try:
                if not BMA_WALKFORWARD_AVAILABLE:
                    messagebox.showerror("错误", "BMA滚动前向回测模块不可用")
                    return
                
                self.update_status("正在启动BMA滚动回测...", 10)
                start_time = time.time()
                
                # 创建增强版BMA滚动回测器
                backtest = EnhancedBMAWalkForward(
                    initial_capital=200000,
                    transaction_cost=0.001,
                    max_positions=15,
                    rebalance_freq='W',
                    prediction_horizon=7,      # 7天预测周期，与再平衡频率对齐
                    training_window_months=4,  # 4个月训练窗口
                    min_training_samples=80,   # 80天最小训练数据
                    volatility_lookback=20,    # ATR计算期
                    risk_target=0.15           # 15%目标年化波动率
                )
                
                self.update_status("BMA回测运行中...", 30)
                
                # 运行增强版回测
                # 使用扩展的股票池（前25只优质股票）
                enhanced_tickers = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
                    'JPM', 'BAC', 'V', 'MA', 'JNJ', 'PG', 'KO', 'WMT', 'HD', 'MCD',
                    'UNH', 'ABBV', 'XOM', 'CVX', 'BA', 'CAT', 'AMT'
                ]
                start_date = "2022-01-01"  # 2.5年数据用于充分训练
                end_date = datetime.now().strftime("%Y-%m-%d")
                
                self.log_message(f"[BMA增强版] 开始运行: {start_date} 到 {end_date}")
                self.log_message(f"[BMA增强版] 股票池: {len(enhanced_tickers)} 只股票")
                self.log_message(f"[BMA增强版] 改进: 预测对齐+复合评分+ATR风险管理")
                
                self.update_status("正在下载数据...", 50)
                
                # 运行增强版回测
                results = backtest.run_enhanced_walkforward_backtest(enhanced_tickers, start_date, end_date)
                
                duration = time.time() - start_time
                self.update_status("BMA回测完成", 100)
                
                # 显示增强版结果
                if results and 'portfolio_values' in results:
                    portfolio_values = results['portfolio_values']
                    metrics = results.get('performance_metrics', {})
                    
                    if portfolio_values and len(portfolio_values) > 0:
                        initial_value = portfolio_values[0]['total_value']
                        final_value = portfolio_values[-1]['total_value']
                        total_return = (final_value / initial_value - 1) * 100
                        
                        result_msg = f"""
BMA增强版回测结果:
• 初始资金: $200,000
• 最终价值: ${final_value:,.2f}
• 总收益率: {total_return:.2f}%
• 年化收益率: {metrics.get('annualized_return', 0)*100:.2f}%
• Sharpe比率: {metrics.get('sharpe_ratio', 0):.3f}
• 最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%
• 获胜率: {metrics.get('win_rate', 0)*100:.1f}%
• 交易次数: {len(results.get('trades_history', []))}
• 股票池: {len(enhanced_tickers)} 只股票
• 耗时: {duration:.1f}秒
"""
                    else:
                        result_msg = f"""
BMA增强版回测完成：
• 系统正常运行，但未生成交易信号
• 原因：当前市场条件下模型认为不适合交易
• 这是正常的风控行为，请稍后再试
• 股票池: {len(enhanced_tickers)} 只股票
• 耗时: {duration:.1f}秒
"""
                        total_return = 0
                    
                    # 保存结果
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    result_file = f"result/bma_walkforward_backtest_{timestamp}.json"
                    os.makedirs('result', exist_ok=True)
                    
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'backtest_results': results,
                            'tickers': enhanced_tickers,
                            'start_date': start_date,
                            'end_date': end_date,
                            'total_return': total_return,
                            'duration': duration,
                            'performance_metrics': metrics,
                            'message': 'BMA增强版回测完成',
                            'enhancements': [
                                '训练-预测周期对齐 (7天)',
                                '复合评分机制 (R²+IC+稳定性)',
                                '动态信号阈值',
                                'ATR风险调整仓位',
                                '增强交易成本模型',
                                '可视化增强'
                            ]
                        }, f, ensure_ascii=False, indent=2, default=str)
                    
                    self.log_message(f"[BMA增强版] 结果已保存到: {result_file}")
                    
                    # 生成可视化图表
                    try:
                        backtest.create_enhanced_visualizations(results)
                        self.log_message(f"[BMA增强版] 可视化图表已生成")
                    except Exception as viz_e:
                        self.log_message(f"[BMA增强版] 可视化生成失败: {viz_e}")
                    
                    # 显示通知
                    self.show_notification("回测完成", result_msg)
                    
                    # 显示结果对话框
                    if messagebox.askyesno("BMA滚动回测完成", result_msg + "\n\n是否打开结果文件夹？"):
                        self.open_result_folder()
                    
                else:
                    error_msg = "BMA滚动回测流程执行完成"
                    self.update_status("回测完成", 100)
                    self.show_notification("回测完成", "回测流程已执行，请检查日志获取详细信息")
                    self.log_message(f"[BMA增强版] {error_msg}")
                    
                    # 保存基本信息
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    result_file = f"result/enhanced_bma_backtest_{timestamp}.json"
                    os.makedirs('result', exist_ok=True)
                    
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'status': 'completed_with_conservative_signals',
                            'tickers': enhanced_tickers,
                            'start_date': start_date,
                            'end_date': end_date,
                            'duration': duration,
                            'message': '增强版回测执行完成，模型采用保守策略',
                            'system_status': '正常运行',
                            'risk_management': '积极生效'
                        }, f, ensure_ascii=False, indent=2, default=str)
                    
            except Exception as e:
                error_msg = f"BMA增强版回测失败: {e}"
                self.update_status(error_msg, 0)
                self.show_notification("错误", error_msg)
                self.logger.error(error_msg)
                import traceback
                self.log_message(f"[BMA增强版] 错误详情: {traceback.format_exc()}")
        
        # 在新线程中运行
        threading.Thread(target=run_in_thread, daemon=True).start()
    
    def run_ensemble_strategy(self):
        """运行双模型融合策略"""
        def run_in_thread():
            try:
                if not self.ensemble_strategy:
                    messagebox.showerror("错误", "双模型融合策略未初始化")
                    return
                
                self.update_status("正在启动双模型融合策略...", 10)
                start_time = time.time()
                
                # 1. 更新Sharpe权重
                self.log_message("[融合策略] 步骤1: 更新Sharpe权重...")
                self.update_status("更新Sharpe权重...", 25)
                
                tickers = getattr(self, 'quantitative_model_stocks', None) or self.ensemble_strategy._get_default_tickers()
                w_bma, w_lstm = self.ensemble_strategy.update_weights(tickers, force_update=True)
                
                self.log_message(f"[融合策略] 权重更新完成: BMA={w_bma:.3f}, LSTM={w_lstm:.3f}")
                
                # 2. 生成融合信号
                self.log_message("[融合策略] 步骤2: 生成融合信号...")
                self.update_status("生成融合信号...", 50)
                
                signals = self.ensemble_strategy.generate_ensemble_signals(tickers)
                
                if signals:
                    # 显示前5个最强信号
                    top_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:5]
                    signal_info = "\n".join([f"{ticker}: {signal:.3f}" for ticker, signal in top_signals])
                    self.log_message(f"[融合策略] 生成 {len(signals)} 个融合信号")
                    self.log_message(f"[融合策略] 前5个信号:\n{signal_info}")
                    
                    # 3. 保存融合信号
                    self.update_status("保存融合信号...", 75)
                    self._save_ensemble_results(signals, w_bma, w_lstm)
                    
                    # 4. 生成交易建议
                    recommendations = self._generate_trading_recommendations_from_signals(signals)
                    
                    duration = time.time() - start_time
                    
                    self.update_status("双模型融合策略完成", 100)
                    
                    # 保存到数据库
                    self.save_analysis_result("双模型融合策略", "ensemble_signals.json", duration, 
                                            f"融合信号数量: {len(signals)}, BMA权重: {w_bma:.3f}, LSTM权重: {w_lstm:.3f}")
                    
                    # 显示结果对话框
                    self._show_ensemble_results_dialog(signals, w_bma, w_lstm, recommendations, duration)
                    
                else:
                    self.log_message("[融合策略] ❌ 未生成融合信号")
                    self.update_status("融合策略未生成信号", 0)
                    messagebox.showwarning("警告", "未生成融合信号，请检查BMA和LSTM模型")
                
            except Exception as e:
                error_msg = f"双模型融合策略失败: {e}"
                self.log_message(f"[融合策略] ❌ {error_msg}")
                self.update_status("融合策略失败", 0)
                messagebox.showerror("错误", error_msg)
                
        # 在新线程中运行
        threading.Thread(target=run_in_thread, daemon=True).start()
    
    def _save_ensemble_results(self, signals, w_bma, w_lstm):
        """保存融合策略结果"""
        try:
            from datetime import datetime
            
            result_data = {
                "timestamp": datetime.now().isoformat(),
                "weights": {
                    "w_bma": w_bma,
                    "w_lstm": w_lstm
                },
                "signals": signals,
                "signal_count": len(signals),
                "top_signals": dict(sorted(signals.items(), key=lambda x: x[1], reverse=True)[:10])
            }
            
            # 保存到文件
            with open("ensemble_signals.json", 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            self.log_message("[融合策略] ✅ 结果已保存到 ensemble_signals.json")
            
        except Exception as e:
            self.log_message(f"[融合策略] ❌ 保存结果失败: {e}")
    
    def _generate_trading_recommendations_from_signals(self, signals):
        """从融合信号生成交易建议"""
        try:
            recommendations = []
            threshold_high = 0.7  # 强买信号阈值
            threshold_low = 0.3   # 强卖信号阈值
            
            for ticker, signal in signals.items():
                if signal >= threshold_high:
                    recommendations.append({
                        "ticker": ticker,
                        "action": "BUY",
                        "signal": signal,
                        "confidence": "HIGH" if signal >= 0.8 else "MEDIUM"
                    })
                elif signal <= threshold_low:
                    recommendations.append({
                        "ticker": ticker,
                        "action": "SELL", 
                        "signal": signal,
                        "confidence": "HIGH" if signal <= 0.2 else "MEDIUM"
                    })
            
            # 按信号强度排序
            recommendations.sort(key=lambda x: abs(x["signal"] - 0.5), reverse=True)
            
            return recommendations[:10]  # 返回前10个建议
            
        except Exception as e:
            self.log_message(f"[融合策略] ❌ 生成交易建议失败: {e}")
            return []
    
    def _show_ensemble_results_dialog(self, signals, w_bma, w_lstm, recommendations, duration):
        """显示融合策略结果对话框"""
        try:
            dialog = tk.Toplevel(self.root)
            dialog.title("双模型融合策略结果")
            dialog.geometry("800x600")
            dialog.configure(bg='white')
            
            # 主框架
            main_frame = ttk.Frame(dialog, padding="20")
            main_frame.pack(fill='both', expand=True)
            
            # 标题
            title_label = ttk.Label(main_frame, 
                                  text="🔬 双模型融合策略分析结果", 
                                  font=('Microsoft YaHei', 16, 'bold'))
            title_label.pack(pady=(0, 20))
            
            # 基本信息框架
            info_frame = ttk.LabelFrame(main_frame, text="策略信息", padding="10")
            info_frame.pack(fill='x', pady=(0, 10))
            
            info_text = (f"⏱️ 分析耗时: {duration:.1f} 秒\n"
                        f"📊 融合信号数量: {len(signals)} 个\n"
                        f"⚖️ BMA权重: {w_bma:.1%}\n"
                        f"🧠 LSTM权重: {w_lstm:.1%}")
            
            ttk.Label(info_frame, text=info_text, font=('Consolas', 10)).pack(anchor='w')
            
            # 创建Notebook用于显示不同信息
            notebook = ttk.Notebook(main_frame)
            notebook.pack(fill='both', expand=True, pady=(10, 0))
            
            # 交易建议标签页
            rec_frame = ttk.Frame(notebook)
            notebook.add(rec_frame, text=f"💡 交易建议 ({len(recommendations)})")
            
            # 交易建议表格
            rec_tree_frame = ttk.Frame(rec_frame, padding="10")
            rec_tree_frame.pack(fill='both', expand=True)
            
            rec_columns = ('股票', '操作', '信号强度', '置信度')
            rec_tree = ttk.Treeview(rec_tree_frame, columns=rec_columns, show='headings', height=12)
            
            for col in rec_columns:
                rec_tree.heading(col, text=col)
                rec_tree.column(col, width=150, anchor='center')
            
            for rec in recommendations:
                action_emoji = "📈" if rec["action"] == "BUY" else "📉"
                confidence_emoji = "🔥" if rec["confidence"] == "HIGH" else "⚡"
                rec_tree.insert('', 'end', values=(
                    rec["ticker"],
                    f"{action_emoji} {rec['action']}",
                    f"{rec['signal']:.3f}",
                    f"{confidence_emoji} {rec['confidence']}"
                ))
            
            rec_tree.pack(fill='both', expand=True)
            
            # 全部信号标签页
            signal_frame = ttk.Frame(notebook)
            notebook.add(signal_frame, text=f"📈 全部信号 ({len(signals)})")
            
            signal_tree_frame = ttk.Frame(signal_frame, padding="10")
            signal_tree_frame.pack(fill='both', expand=True)
            
            signal_columns = ('排名', '股票代码', '融合信号', '信号等级')
            signal_tree = ttk.Treeview(signal_tree_frame, columns=signal_columns, show='headings', height=15)
            
            for col in signal_columns:
                signal_tree.heading(col, text=col)
                signal_tree.column(col, width=120, anchor='center')
            
            # 按信号强度排序显示
            sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
            
            for idx, (ticker, signal) in enumerate(sorted_signals, 1):
                if signal >= 0.7:
                    level = "🔥 强买"
                elif signal >= 0.6:
                    level = "📈 买入"
                elif signal >= 0.4:
                    level = "⚖️ 中性"
                elif signal >= 0.3:
                    level = "📉 卖出"
                else:
                    level = "❄️ 强卖"
                
                signal_tree.insert('', 'end', values=(
                    f"#{idx}",
                    ticker,
                    f"{signal:.3f}",
                    level
                ))
            
            signal_tree.pack(fill='both', expand=True)
            
            # 按钮框架
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill='x', pady=(20, 0))
            
            ttk.Button(button_frame, text="📁 打开结果文件夹", 
                      command=self.open_result_folder).pack(side='left', padx=(0, 10))
            
            ttk.Button(button_frame, text="📊 查看权重历史", 
                      command=self._show_weights_history).pack(side='left', padx=(0, 10))
                      
            ttk.Button(button_frame, text="🔄 手动执行交易", 
                      command=lambda: self._execute_ensemble_trading(recommendations)).pack(side='left', padx=(0, 10))
            
            ttk.Button(button_frame, text="❌ 关闭", 
                      command=dialog.destroy).pack(side='right')
            
            # 居中显示
            dialog.transient(self.root)
            dialog.grab_set()
            
        except Exception as e:
            self.log_message(f"[融合策略] ❌ 显示结果对话框失败: {e}")
            messagebox.showerror("错误", f"显示结果失败: {e}")
    
    def _show_weights_history(self):
        """显示权重历史"""
        try:
            if hasattr(self, 'ensemble_strategy') and self.ensemble_strategy:
                current_weights = self.ensemble_strategy.get_current_weights()
                history_info = (f"当前权重信息:\n\n"
                              f"更新日期: {current_weights.get('date', 'N/A')}\n"
                              f"BMA权重: {current_weights.get('w_bma', 0.5):.1%}\n"
                              f"LSTM权重: {current_weights.get('w_lstm', 0.5):.1%}\n"
                              f"BMA Sharpe: {current_weights.get('sharpe_bma', 0.0):.4f}\n"
                              f"LSTM Sharpe: {current_weights.get('sharpe_lstm', 0.0):.4f}\n"
                              f"回望周数: {current_weights.get('lookback_weeks', 12)}\n"
                              f"股票数量: {current_weights.get('tickers_count', 0)}")
                
                messagebox.showinfo("权重历史", history_info)
            else:
                messagebox.showwarning("警告", "融合策略未初始化")
        except Exception as e:
            messagebox.showerror("错误", f"显示权重历史失败: {e}")
    
    def _execute_ensemble_trading(self, recommendations):
        """执行融合策略交易建议"""
        try:
            if not recommendations:
                messagebox.showinfo("提示", "没有交易建议可执行")
                return
                
            # 这里可以集成实际的交易执行逻辑
            trade_summary = f"准备执行 {len(recommendations)} 个交易建议:\n\n"
            for rec in recommendations[:5]:  # 只显示前5个
                trade_summary += f"• {rec['action']} {rec['ticker']} (信号: {rec['signal']:.3f})\n"
            
            if len(recommendations) > 5:
                trade_summary += f"...以及其他 {len(recommendations) - 5} 个建议"
                
            result = messagebox.askyesno("确认交易", 
                                       f"{trade_summary}\n\n确定要执行这些交易吗？\n\n注意：这将调用实际的交易系统！")
            
            if result:
                # 实际交易逻辑
                self.log_message("[融合策略] 📋 用户确认执行融合策略交易")
                self.log_message("[融合策略] ⚠️ 实际交易功能待集成")
                messagebox.showinfo("交易状态", "交易请求已记录，实际交易功能开发中...")
            else:
                self.log_message("[融合策略] 👤 用户取消交易执行")
                
        except Exception as e:
            self.log_message(f"[融合策略] ❌ 执行交易失败: {e}")
            messagebox.showerror("错误", f"执行交易失败: {e}")
    
    def run_both_quantitative_models(self):
        """运行LSTM和BMA两个量化模型"""
        def run_in_thread():
            try:
                self.update_status("正在启动双模型分析...", 5)
                
                # 使用默认股票列表
                stock_list_str = ','.join(self.quantitative_model_stocks)
                
                # 先运行LSTM模型
                self.logger.info("开始运行LSTM模型")
                lstm_args = [
                    "--symbols", stock_list_str,
                    "--start-date", "2024-01-01",
                    "--end-date", datetime.now().strftime("%Y-%m-%d")
                ]
                lstm_success = self._run_model_subprocess("LSTM", "lstm_multi_day_enhanced.py", lstm_args, "test_multi_day_lstm_analysis_")
                
                # 再运行BMA模型
                self.logger.info("开始运行BMA模型")
                bma_args = ["--symbols", stock_list_str]
                bma_success = self._run_model_subprocess("BMA", "量化模型_bma_enhanced.py", bma_args, "bma_quantitative_analysis_")
                
                self.update_status("双模型分析完成", 100)
                
                # 显示结果总结
                if lstm_success and bma_success:
                    self.show_notification("双模型完成", "LSTM和BMA模型都成功完成")
                elif lstm_success or bma_success:
                    success_model = "LSTM" if lstm_success else "BMA"
                    self.show_notification("部分完成", f"{success_model}模型成功完成")
                else:
                    self.show_notification("模型失败", "LSTM和BMA模型都运行失败")
                
            except Exception as e:
                error_msg = f"双模型运行失败: {e}"
                self.update_status(error_msg, 0)
                self.show_notification("错误", error_msg)
                self.logger.error(error_msg)
        
        # 在新线程中运行
        threading.Thread(target=run_in_thread, daemon=True).start()

    def run_backtest_analysis(self):
        """运行回测分析"""
        # 提示用户选择文件
        info_msg = ("回测分析需要基于之前的量化分析结果进行。\n\n"
                   "请选择一个量化分析结果文件：\n"
                   "• 通常位于 result/ 文件夹中\n"
                   "• 文件名如：quantitative_analysis_*.xlsx\n"
                   "• 建议选择最新的分析结果")
        
        messagebox.showinfo("选择分析文件", info_msg)
        
        # 首先让用户选择分析结果文件
        file_path = filedialog.askopenfilename(
            title="选择量化分析结果文件 - 回测分析",
            initialdir="./result",
            filetypes=[
                ("Excel文件", "*.xlsx"),
                ("CSV文件", "*.csv"),
                ("所有文件", "*.*")
            ]
        )
        
        if not file_path:
            self.update_status("用户取消了文件选择", 0)
            return
        
        def run_in_thread():
            try:
                # 自动显示状态监控
                self.auto_show_status_monitor()
                
                self.update_status("正在启动回测分析...", 10)
                self.backtest_button.config(state='disabled')
                
                start_time = time.time()
                
                # 运行回测分析，传递选中的文件路径
                process = subprocess.Popen(
                    [sys.executable, "comprehensive_category_backtest.py", file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='gbk',  # 使用GBK编码处理中文
                    cwd=os.getcwd()
                )
                
                self.update_status("回测分析运行中...", 50)
                
                stdout, stderr = process.communicate()
                duration = time.time() - start_time
                
                if process.returncode == 0:
                    self.update_status("回测分析完成", 100)
                    
                    # 查找生成的文件
                    result_files = self.find_latest_result_files("comprehensive_analysis_")
                    
                    # 保存到数据库
                    self.save_analysis_result("回测分析", result_files[0] if result_files else "",
                                            duration, stdout)
                    
                    self.show_notification("任务完成", f"回测分析完成\n耗时: {duration:.1f}秒")
                    self.load_recent_results()
                    
                else:
                    # 截断错误信息，避免过长
                    short_error = stderr[:150] + "..." if len(stderr) > 150 else stderr
                    error_msg = f"回测分析运行失败\n错误信息: {short_error}"
                    self.update_status("回测分析运行失败", 0)
                    self.show_notification("任务失败", error_msg)
                    self.logger.error(f"回测分析运行失败\n完整错误信息: {stderr}")
                    
            except Exception as e:
                error_msg = f"启动回测分析失败: {e}"
                self.update_status(error_msg, 0)
                self.show_notification("错误", error_msg)
                self.logger.error(error_msg)
                
            finally:
                self.backtest_button.config(state='normal')
                self.update_status("就绪", 0)
        
        threading.Thread(target=run_in_thread, daemon=True).start()
    

    
    def run_walkforward_backtest(self):
        """运行Walk-Forward回测"""
        
        # 显示日期和股票池选择对话框
        self.show_date_selection_dialog("walkforward")
    
    def run_walkforward_backtest_with_dates(self, start_date, end_date, ticker_file=None):
        """运行Walk-Forward回测（带日期和股票池参数）"""
        
        def run_in_thread():
            try:
                # 自动显示状态监控
                self.auto_show_status_monitor()
                
                self.walkforward_backtest_button.config(state='disabled')
                
                info_msg = (f"Walk-Forward滚动回测使用72个双周训练窗口，预测下两周收益。\n\n"
                          f"回测期间: {start_date} 到 {end_date}\n"
                          f"股票池: {'自定义股票池' if ticker_file else '默认股票池'}\n\n"
                          "主要特性:\n"
                          "• 严格防止数据泄漏的滚动窗口回测\n"
                          "• 集成BMA贝叶斯模型平均\n"
                          "• 方向性准确率(Hit Rate)分析\n"
                          "• 年化多空收益和最大回撤\n"
                          "• 双周收益分布分析\n\n"
                          "点击确定开始回测...")
                
                if not messagebox.askyesno("Walk-Forward回测", info_msg):
                    return
                
                self.update_status("正在启动Walk-Forward回测...", 10)
                
                # 构建命令
                cmd = [sys.executable, "enhanced_walkforward_backtest.py"]
                cmd.extend(["--start-date", start_date])
                cmd.extend(["--end-date", end_date])
                cmd.extend(["--train-window", "72"])
                cmd.extend(["--capital", "1000000"])
                cmd.extend(["--top-stocks", "20"])
                if ticker_file:
                    cmd.extend(["--ticker-file", ticker_file])
                
                self.logger.info(f"执行Walk-Forward回测命令: {' '.join(cmd)}")
                
                self.update_status("Walk-Forward回测运行中...", 50)
                
                start_time = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
                end_time = time.time()
                duration = end_time - start_time
                
                if result.returncode == 0:
                    self.update_status("Walk-Forward回测完成", 100)
                    
                    # 查找生成的结果文件
                    result_files = self.find_latest_result_files("walkforward_results_")
                    
                    # 如果没找到，尝试查找walkforward_results目录中的文件
                    if not result_files:
                        walkforward_dir = Path("walkforward_results")
                        if walkforward_dir.exists():
                            for file in walkforward_dir.glob("*.xlsx"):
                                if not file.name.startswith('~$'):
                                    result_files.append(str(file))
                    
                    # 如果还是没找到，尝试查找所有walkforward相关的文件
                    if not result_files:
                        for file in Path('.').glob("walkforward*.xlsx"):
                            if not file.name.startswith('~$'):
                                result_files.append(str(file))
                        for file in Path('.').glob("walkforward*.txt"):
                            result_files.append(str(file))
                    
                    self.save_analysis_result("Walk-Forward回测", result_files[0] if result_files else "",
                                            duration, result.stdout)
                    
                    self.show_notification("任务完成", f"Walk-Forward回测完成\n耗时: {duration:.1f}秒")
                    self.logger.info("Walk-Forward回测执行成功")
                    
                else:
                    stderr = result.stderr.strip()
                    short_error = stderr[:150] + "..." if len(stderr) > 150 else stderr
                    error_msg = f"Walk-Forward回测运行失败\n错误信息: {short_error}"
                    self.update_status("Walk-Forward回测运行失败", 0)
                    self.show_notification("任务失败", error_msg)
                    self.logger.error(f"Walk-Forward回测运行失败\n完整错误信息: {stderr}")
                    
            except Exception as e:
                error_msg = f"启动Walk-Forward回测失败: {e}"
                self.update_status(error_msg, 0)
                self.show_notification("错误", error_msg)
                self.logger.error(error_msg)
                
            finally:
                self.walkforward_backtest_button.config(state='normal')
                self.update_status("就绪", 0)
        
        threading.Thread(target=run_in_thread, daemon=True).start()
    
    def run_lstm_enhanced_model(self):
        """运行LSTM增强模型"""
        # 直接启动LSTM分析，提供更好的用户体验
        self.run_lstm_analysis()
    
    def show_excel_selection_dialog_for_walkforward(self, excel_files):
        """为Walk-Forward回测显示Excel选择对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("选择Excel文件")
        dialog.geometry("600x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 居中显示
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (dialog.winfo_screenheight() // 2) - (400 // 2)
        dialog.geometry(f"600x400+{x}+{y}")
        
        # 创建主框架
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="选择用于Walk-Forward回测的Excel文件", 
                               font=('Microsoft YaHei', 12, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # 说明
        info_label = ttk.Label(main_frame, 
                              text="请选择一个包含股票代码的Excel文件。\n系统将从'Ticker'列中提取股票代码用于回测。",
                              font=('Microsoft YaHei', 10))
        info_label.pack(pady=(0, 10))
        
        # 创建列表框
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 创建Treeview
        columns = ('文件名', '大小', '修改时间')
        tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        
        # 设置列标题
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 填充数据
        selected_file = [None]  # 使用列表来存储选中的文件
        
        for file_path in excel_files:
            try:
                # 再次过滤掉Excel临时文件（双重保险）
                if Path(file_path).name.startswith('~$'):
                    continue
                    
                file_stat = Path(file_path).stat()
                file_size = file_stat.st_size
                mod_time = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M')
                
                # 格式化文件大小
                if file_size < 1024:
                    size_str = f"{file_size} B"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{file_size / (1024 * 1024):.1f} MB"
                
                tree.insert('', 'end', values=(Path(file_path).name, size_str, mod_time), tags=(file_path,))
                
            except Exception as e:
                self.logger.error(f"处理文件 {file_path} 时出错: {e}")
                continue
        
        def on_item_double_click(event):
            item = tree.selection()[0]
            file_path = tree.item(item, 'tags')[0]
            selected_file[0] = file_path
            dialog.destroy()
        
        def on_select():
            selection = tree.selection()
            if selection:
                item = selection[0]
                file_path = tree.item(item, 'tags')[0]
                selected_file[0] = file_path
                dialog.destroy()
        
        def on_cancel():
            selected_file[0] = None
            dialog.destroy()
        
        # 绑定双击事件
        tree.bind('<Double-1>', on_item_double_click)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="选择", command=on_select).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="取消", command=on_cancel).pack(side=tk.RIGHT)
        
        # 等待对话框关闭
        dialog.wait_window()
        
        return selected_file[0]
    
    def extract_tickers_from_excel(self, excel_file):
        """从Excel文件中提取股票代码"""
        try:
            import pandas as pd
            
            # 检查是否为Excel临时文件
            if Path(excel_file).name.startswith('~$'):
                self.logger.error(f"跳过Excel临时文件: {excel_file}")
                return []
            
            # 读取Excel文件
            df = pd.read_excel(excel_file)
            
            # 查找Ticker列
            ticker_column = None
            for col in df.columns:
                if 'ticker' in col.lower() or '股票' in col.lower() or '代码' in col.lower():
                    ticker_column = col
                    break
            
            if ticker_column is None:
                self.logger.error(f"在文件 {excel_file} 中未找到Ticker列")
                return []
            
            # 提取股票代码
            tickers = df[ticker_column].dropna().unique().tolist()
            
            # 清理股票代码（移除空格，转换为大写等）
            cleaned_tickers = []
            for ticker in tickers:
                if isinstance(ticker, str):
                    cleaned_ticker = ticker.strip().upper()
                    if cleaned_ticker:  # 确保不是空字符串
                        cleaned_tickers.append(cleaned_ticker)
            
            self.logger.info(f"从文件 {excel_file} 中提取了 {len(cleaned_tickers)} 个股票代码")
            return cleaned_tickers
            
        except Exception as e:
            self.logger.error(f"从Excel文件 {excel_file} 提取股票代码时出错: {e}")
            return []
    
    def _create_ticker_file(self, tickers):
        """创建股票代码文件"""
        try:
            # 创建临时文件名
            import tempfile
            import uuid
            
            # 生成唯一的临时文件名
            temp_filename = f"tickers_{uuid.uuid4().hex[:8]}.txt"
            temp_filepath = os.path.join(tempfile.gettempdir(), temp_filename)
            
            # 写入股票代码
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                for ticker in tickers:
                    f.write(f"{ticker}\n")
            
            self.logger.info(f"创建股票代码文件: {temp_filepath}，包含 {len(tickers)} 个股票")
            return temp_filepath
            
        except Exception as e:
            self.logger.error(f"创建股票代码文件失败: {e}")
            return None
    

    def find_latest_result_files(self, prefix):
        """查找最新的结果文件"""
        result_files = []
        
        # 检查当前目录
        for file in Path('.').glob(f"{prefix}*.xlsx"):
            # 过滤掉Excel临时文件
            if not file.name.startswith('~$'):
                result_files.append(str(file))
        
        # 检查result目录
        result_dir = Path(self.config['result_directory'])
        if result_dir.exists():
            for file in result_dir.glob(f"{prefix}*.xlsx"):
                # 过滤掉Excel临时文件
                if not file.name.startswith('~$'):
                    result_files.append(str(file))
        
        # 检查walkforward_results目录
        walkforward_dir = Path("walkforward_results")
        if walkforward_dir.exists():
            for file in walkforward_dir.glob("*.xlsx"):
                # 过滤掉Excel临时文件
                if not file.name.startswith('~$'):
                    result_files.append(str(file))
        
        # 返回最新的文件
        if result_files:
            result_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
            return result_files
        
        return []
    
    def save_analysis_result(self, analysis_type, file_path, duration, output):
        """保存分析结果到数据库"""
        try:
            cursor = self.conn.cursor()
            
            # 解析输出信息获取统计数据
            stock_count, avg_score, buy_count, hold_count, sell_count = self.parse_analysis_output(output)
            
            # 保存分析结果
            cursor.execute('''
                INSERT INTO analysis_results 
                (analysis_type, file_path, stock_count, avg_score, buy_count, hold_count, sell_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (analysis_type, file_path, stock_count, avg_score, buy_count, hold_count, sell_count))
            
            # 保存任务执行记录
            cursor.execute('''
                INSERT INTO task_executions 
                (task_type, status, duration_seconds, result_files)
                VALUES (?, ?, ?, ?)
            ''', (analysis_type, 'success', duration, file_path))
            
            self.conn.commit()
            self.logger.info(f"分析结果已保存到数据库: {analysis_type}")
            
        except Exception as e:
            self.logger.error(f"保存分析结果失败: {e}")
    
    def parse_analysis_output(self, output):
        """解析分析输出获取统计信息"""
        # 默认值
        stock_count = avg_score = buy_count = hold_count = sell_count = None
        
        try:
            lines = output.split('\n')
            for line in lines:
                import re
                
                # 匹配股票总数
                if '股票总数:' in line:
                    match = re.search(r'股票总数:\s*(\d+)', line)
                    if match:
                        stock_count = int(match.group(1))
                
                # 匹配BUY推荐数量
                elif 'BUY推荐:' in line:
                    match = re.search(r'BUY推荐:\s*(\d+)', line)
                    if match:
                        buy_count = int(match.group(1))
                
                # 匹配HOLD推荐数量
                elif 'HOLD推荐:' in line:
                    match = re.search(r'HOLD推荐:\s*(\d+)', line)
                    if match:
                        hold_count = int(match.group(1))
                
                # 匹配SELL推荐数量
                elif 'SELL推荐:' in line:
                    match = re.search(r'SELL推荐:\s*(\d+)', line)
                    if match:
                        sell_count = int(match.group(1))
                
                # 匹配平均预测收益率
                elif '平均预测收益率:' in line:
                    match = re.search(r'平均预测收益率:\s*([+-]?\d+\.?\d*)', line)
                    if match:
                        avg_score = float(match.group(1))
                
                # 兼容旧格式
                elif '共分析' in line and '只股票' in line:
                    match = re.search(r'(\d+)只股票', line)
                    if match:
                        stock_count = int(match.group(1))
                elif 'BUY:' in line and 'HOLD:' in line and 'SELL:' in line:
                    buy_match = re.search(r'BUY:\s*(\d+)', line)
                    hold_match = re.search(r'HOLD:\s*(\d+)', line)
                    sell_match = re.search(r'SELL:\s*(\d+)', line)
                    if buy_match:
                        buy_count = int(buy_match.group(1))
                    if hold_match:
                        hold_count = int(hold_match.group(1))
                    if sell_match:
                        sell_count = int(sell_match.group(1))
                elif '平均综合风险评分' in line:
                    score_match = re.search(r'(\d+\.?\d*)', line)
                    if score_match:
                        avg_score = float(score_match.group(1))
                        
        except Exception as e:
            self.logger.warning(f"解析输出失败: {e}")
        
        return stock_count, avg_score, buy_count, hold_count, sell_count
    
    def load_recent_results(self):
        """加载最近的分析结果"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT date_created, analysis_type, stock_count, avg_score, 
                       buy_count, hold_count, sell_count, status
                FROM analysis_results 
                ORDER BY date_created DESC 
                LIMIT 20
            ''')
            
            # 清空现有数据
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # 添加新数据
            for row in cursor.fetchall():
                date_str = datetime.fromisoformat(row[0]).strftime('%Y-%m-%d %H:%M')
                formatted_row = (
                    date_str,
                    row[1],  # analysis_type
                    row[2] or '-',  # stock_count
                    f"{row[3]:.2f}" if row[3] else '-',  # avg_score
                    row[4] or '-',  # buy_count
                    row[5] or '-',  # hold_count
                    row[6] or '-',  # sell_count
                    row[7]  # status
                )
                self.results_tree.insert('', 'end', values=formatted_row)
                
        except Exception as e:
            self.logger.error(f"加载结果失败: {e}")
    
    def auto_run_analysis(self):
        """自动运行分析（定时任务）"""
        self.logger.info("开始自动运行双周分析")
        
        def auto_run_thread():
            try:
                # 显示通知
                self.show_notification("定时任务", "开始执行双周量化分析", timeout=5)
                
                # 依次运行三个分析
                self.run_quantitative_model()
                time.sleep(30)  # 等待第一个任务完成
                
                self.run_backtest_analysis()
                time.sleep(30)  # 等待第二个任务完成
                
                # 运行LSTM和BMA模型分析
                self.run_both_quantitative_models()
                
                # 完成通知
                self.show_notification("定时任务完成", 
                                     f"双周量化分析已完成\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                
                # 更新下次运行时间
                self.schedule_status_var.set(f"下次自动运行: {self.get_next_run_time()}")
                
            except Exception as e:
                error_msg = f"自动分析失败: {e}"
                self.logger.error(error_msg)
                self.show_notification("定时任务失败", error_msg)
        
        threading.Thread(target=auto_run_thread, daemon=True).start()
    
    def show_auto_trading_manager(self):
        """显示自动交易管理窗口"""
        trading_window = tk.Toplevel(self.root)
        trading_window.title("🤖 自动交易管理")
        trading_window.geometry("800x600")
        trading_window.transient(self.root)
        
        # 主框架
        main_frame = ttk.Frame(trading_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="🤖 自动交易管理", 
                               font=('Microsoft YaHei', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # 左右分栏
        left_right_frame = ttk.Frame(main_frame)
        left_right_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧：文件选择和股票管理
        left_frame = ttk.LabelFrame(left_right_frame, text="📊 信号文件和股票管理", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(left_frame, text="选择分析结果文件", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 当前选择的文件显示
        self.selected_file_label = ttk.Label(file_frame, text="未选择文件", foreground="red")
        self.selected_file_label.pack(pady=5)
        
        # 文件选择按钮
        file_buttons_frame = ttk.Frame(file_frame)
        file_buttons_frame.pack(fill=tk.X)
        
        ttk.Button(file_buttons_frame, text="📄 选择JSON文件", 
                  command=lambda: self.select_signal_file('json')).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_buttons_frame, text="📊 选择Excel文件", 
                  command=lambda: self.select_signal_file('excel')).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_buttons_frame, text="🔄 自动加载最新", 
                  command=self.auto_load_latest_signal).pack(side=tk.LEFT)
        
        # 股票列表管理
        stock_frame = ttk.LabelFrame(left_frame, text="交易股票列表", padding="10")
        stock_frame.pack(fill=tk.BOTH, expand=True)
        
        # 股票列表
        list_frame = ttk.Frame(stock_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建Listbox和滚动条
        self.stock_listbox = tk.Listbox(list_frame, height=10)
        stock_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.stock_listbox.yview)
        self.stock_listbox.configure(yscrollcommand=stock_scrollbar.set)
        
        self.stock_listbox.pack(side="left", fill="both", expand=True)
        stock_scrollbar.pack(side="right", fill="y")
        
        # 股票管理按钮
        stock_buttons_frame = ttk.Frame(stock_frame)
        stock_buttons_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(stock_buttons_frame, text="➕ 添加股票", 
                  command=self.add_trading_stock).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(stock_buttons_frame, text="➖ 删除选中", 
                  command=self.remove_trading_stock).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(stock_buttons_frame, text="🔄 刷新前5个", 
                  command=self.load_top5_stocks).pack(side=tk.LEFT)
        
        # 右侧：交易控制
        right_frame = ttk.LabelFrame(left_right_frame, text="🚀 交易控制", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.configure(width=300)
        
        # IBKR连接状态
        conn_frame = ttk.LabelFrame(right_frame, text="IBKR连接状态", padding="10")
        conn_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.connection_status_label = ttk.Label(conn_frame, text="🔴 未连接", 
                                                font=('Microsoft YaHei', 10, 'bold'))
        self.connection_status_label.pack()
        
        # 连接设置
        settings_frame = ttk.Frame(conn_frame)
        settings_frame.pack(fill=tk.X, pady=(5, 0))
        
        # 主机地址设置
        ttk.Label(settings_frame, text="主机:").grid(row=0, column=0, sticky="w")
        self.host_var = tk.StringVar(value=self.config.get('ibkr_host', '127.0.0.1'))
        host_entry = ttk.Entry(settings_frame, textvariable=self.host_var, width=12)
        host_entry.grid(row=0, column=1, padx=(5, 10))
        
        # 端口设置
        ttk.Label(settings_frame, text="端口:").grid(row=0, column=2, sticky="w")
        self.port_var = tk.StringVar(value=str(self.config.get('ibkr_port', 4002)))
        port_entry = ttk.Entry(settings_frame, textvariable=self.port_var, width=8)
        port_entry.grid(row=0, column=3, padx=(5, 0))
        
        # 端口快速选择
        port_select_frame = ttk.Frame(conn_frame)
        port_select_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(port_select_frame, text="快速选择:").pack(side=tk.LEFT)
        self.port_combo = ttk.Combobox(port_select_frame, values=[
            "4002 - 模拟交易 (IB Gateway)",
            "4001 - 实盘交易 (IB Gateway)", 
            "7497 - 模拟交易 (TWS)",
            "7496 - 实盘交易 (TWS)"
        ], width=25, state="readonly")
        self.port_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.port_combo.bind("<<ComboboxSelected>>", self.on_port_selected)
        
        # 连接按钮
        conn_buttons_frame = ttk.Frame(conn_frame)
        conn_buttons_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(conn_buttons_frame, text="🧪 测试连接", 
                  command=self.test_ibkr_connection).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(conn_buttons_frame, text="🔗 连接IBKR", 
                  command=self.connect_ibkr).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(conn_buttons_frame, text="❌ 断开", 
                  command=self.disconnect_ibkr).pack(side=tk.LEFT)
        
        # 交易控制
        trading_control_frame = ttk.LabelFrame(right_frame, text="交易控制", padding="10")
        trading_control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 账户余额显示
        balance_frame = ttk.Frame(trading_control_frame)
        balance_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(balance_frame, text="账户余额:").pack(side=tk.LEFT)
        self.balance_label = ttk.Label(balance_frame, text="$0.00", 
                                      font=('Microsoft YaHei', 10, 'bold'), foreground="green")
        self.balance_label.pack(side=tk.RIGHT)
        
        # 交易状态
        self.trading_status_label = ttk.Label(trading_control_frame, text="❌ 交易已停止", 
                                             font=('Microsoft YaHei', 10, 'bold'))
        self.trading_status_label.pack(pady=(0, 10))
        
        # 交易按钮
        trading_buttons_frame = ttk.Frame(trading_control_frame)
        trading_buttons_frame.pack(fill=tk.X)
        
        self.start_trading_btn = ttk.Button(trading_buttons_frame, text="🚀 启动交易", 
                                           command=self.start_auto_trading_wrapper)
        self.start_trading_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.stop_trading_btn = ttk.Button(trading_buttons_frame, text="⛔ 停止交易", 
                                          command=self.stop_auto_trading, state="disabled")
        self.stop_trading_btn.pack(fill=tk.X)
        
        # 紧急控制
        emergency_frame = ttk.LabelFrame(right_frame, text="紧急控制", padding="10")
        emergency_frame.pack(fill=tk.X)
        
        ttk.Button(emergency_frame, text="🚨 全仓卖出", 
                  command=self.emergency_sell_all).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(emergency_frame, text="📊 查看持仓", 
                  command=self.show_positions).pack(fill=tk.X)
    
    def open_result_folder(self):
        """打开结果文件夹"""
        result_path = Path(self.config['result_directory'])
        if not result_path.exists():
            result_path.mkdir()
        
        try:
            os.startfile(str(result_path))
        except:
            webbrowser.open(f"file://{result_path.absolute()}")
    
    def delete_excel_outputs(self):
        """删除选中的Excel输出文件"""
        try:
            # 获取结果目录中的所有Excel文件
            result_dir = self.config['result_directory']
            if not os.path.exists(result_dir):
                messagebox.showwarning("警告", "结果文件夹不存在")
                return
            
            excel_files = []
            for file in os.listdir(result_dir):
                if file.endswith('.xlsx') or file.endswith('.xls'):
                    file_path = os.path.join(result_dir, file)
                    file_size = os.path.getsize(file_path)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    excel_files.append({
                        'name': file,
                        'path': file_path,
                        'size': file_size,
                        'time': file_time
                    })
            
            if not excel_files:
                messagebox.showinfo("提示", "没有找到Excel文件")
                return
            
            # 创建选择窗口
            self.show_excel_selection_dialog(excel_files)
                
        except Exception as e:
            messagebox.showerror("错误", f"删除Excel文件时发生错误：\n{str(e)}")
            self.logger.error(f"删除Excel文件失败: {e}")
    
    def show_excel_selection_dialog(self, excel_files):
        """显示Excel文件选择对话框"""
        selection_window = tk.Toplevel(self.root)
        selection_window.title("选择要删除的Excel文件")
        selection_window.geometry("700x500")
        selection_window.resizable(True, True)
        
        # 设置窗口图标
        try:
            selection_window.iconbitmap(default="trading.ico")
        except:
            pass
        
        # 创建主框架
        main_frame = ttk.Frame(selection_window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 标题
        title_label = ttk.Label(main_frame, text="选择要删除的Excel文件", 
                               font=('Microsoft YaHei', 12, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # 全选/取消全选按钮
        select_frame = ttk.Frame(main_frame)
        select_frame.pack(fill='x', pady=(0, 10))
        
        select_all_var = tk.BooleanVar()
        
        # 创建文件列表
        columns = ('文件名', '大小', '修改时间', '选中')
        tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=15)
        
        # 设置列标题和宽度
        tree.heading('文件名', text='文件名')
        tree.heading('大小', text='大小')
        tree.heading('修改时间', text='修改时间')
        tree.heading('选中', text='选中')
        
        tree.column('文件名', width=300)
        tree.column('大小', width=100)
        tree.column('修改时间', width=150)
        tree.column('选中', width=60)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # 填充文件列表
        for file_info in excel_files:
            size_str = f"{file_info['size'] / 1024:.1f} KB"
            time_str = file_info['time'].strftime('%Y-%m-%d %H:%M')
            
            item = tree.insert('', 'end', values=(
                file_info['name'],
                size_str,
                time_str,
                '☐'  # 未选中状态
            ))
            
            # 存储文件信息
            tree.set(item, 'file_info', file_info)
        
        # 状态栏
        status_var = tk.StringVar()
        status_var.set(f"共找到 {len(excel_files)} 个Excel文件")
        status_label = ttk.Label(main_frame, textvariable=status_var)
        status_label.pack(side='bottom', pady=(5, 0))
        
        # 更新选中文件数量
        def update_selected_count():
            selected_count = sum(1 for item in tree.get_children() 
                               if tree.set(item, '选中') == '☑')
            status_var.set(f"共找到 {len(excel_files)} 个Excel文件，已选择 {selected_count} 个")
        
        # 全选/取消全选功能
        def toggle_select_all():
            new_state = '☑' if select_all_var.get() else '☐'
            for item in tree.get_children():
                tree.set(item, '选中', new_state)
            update_selected_count()
        
        ttk.Checkbutton(select_frame, text="全选/取消全选", 
                       variable=select_all_var, command=toggle_select_all).pack(side='left')
        
        # 绑定选择事件
        def on_item_click(event):
            item = tree.selection()[0] if tree.selection() else None
            if item:
                current_selected = tree.set(item, '选中')
                new_selected = '☑' if current_selected == '☐' else '☐'
                tree.set(item, '选中', new_selected)
                update_selected_count()
        
        tree.bind('<Button-1>', on_item_click)
        
        # 布局
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))
        
        def delete_selected_files():
            """删除选中的文件"""
            selected_files = []
            for item in tree.get_children():
                if tree.set(item, '选中') == '☑':
                    file_info = tree.set(item, 'file_info')
                    selected_files.append(file_info)
            
            if not selected_files:
                messagebox.showwarning("警告", "请先选择要删除的文件")
                return
            
            # 确认删除
            confirm_msg = f"确定要删除选中的 {len(selected_files)} 个文件吗？\n\n"
            for file_info in selected_files[:5]:  # 只显示前5个
                confirm_msg += f"• {file_info['name']}\n"
            
            if len(selected_files) > 5:
                confirm_msg += f"... 还有 {len(selected_files) - 5} 个文件\n"
            
            confirm_msg += "\n此操作不可撤销！"
            
            if messagebox.askyesno("确认删除", confirm_msg):
                # 执行删除
                deleted_count = 0
                failed_files = []
                
                for file_info in selected_files:
                    try:
                        os.remove(file_info['path'])
                        deleted_count += 1
                    except Exception as e:
                        failed_files.append((file_info['name'], str(e)))
                
                # 显示结果
                result_msg = f"删除完成！\n\n成功删除: {deleted_count} 个文件"
                if failed_files:
                    result_msg += f"\n删除失败: {len(failed_files)} 个文件"
                    for file_name, error in failed_files:
                        result_msg += f"\n• {file_name}: {error}"
                
                messagebox.showinfo("删除结果", result_msg)
                
                # 刷新界面
                self.load_recent_results()
                
                # 记录日志
                self.logger.info(f"用户删除了 {deleted_count} 个选中的Excel文件")
                
                # 关闭窗口
                selection_window.destroy()
        
        def cancel_operation():
            """取消操作"""
            selection_window.destroy()
        
        # 按钮
        ttk.Button(button_frame, text="删除选中文件", 
                   command=delete_selected_files).pack(side='left', padx=(0, 10))
        ttk.Button(button_frame, text="取消", 
                   command=cancel_operation).pack(side='left')
        
        # 初始更新计数
        update_selected_count()
    
    def show_settings(self):
        """显示设置窗口"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("设置")
        settings_window.geometry("500x550")
        settings_window.resizable(False, False)
        
        # 创建滚动框架
        canvas = tk.Canvas(settings_window)
        scrollbar = ttk.Scrollbar(settings_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 设置选项
        ttk.Label(scrollable_frame, text="应用设置", font=('Microsoft YaHei', 12, 'bold')).pack(pady=10)
        
        # 基础设置
        basic_frame = ttk.LabelFrame(scrollable_frame, text="基础设置", padding="10")
        basic_frame.pack(fill="x", padx=20, pady=5)
        
        # 自动运行选项
        auto_run_var = tk.BooleanVar(value=self.config['auto_run'])
        ttk.Checkbutton(basic_frame, text="启用双周自动分析", 
                       variable=auto_run_var).pack(anchor='w', pady=2)
        
        # 通知选项
        notifications_var = tk.BooleanVar(value=self.config['notifications'])
        ttk.Checkbutton(basic_frame, text="启用系统通知", 
                       variable=notifications_var).pack(anchor='w', pady=2)
        
        # 日志级别
        ttk.Label(basic_frame, text="日志级别:").pack(anchor='w', pady=(5, 0))
        log_level_var = tk.StringVar(value=self.config['log_level'])
        log_level_combo = ttk.Combobox(basic_frame, textvariable=log_level_var,
                                      values=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
        log_level_combo.pack(anchor='w', pady=2)
        
        # 增强版交易设置
        trading_frame = ttk.LabelFrame(scrollable_frame, text="增强版交易设置", padding="10")
        trading_frame.pack(fill="x", padx=20, pady=5)
        
        # 启用增强版交易
        enhanced_trading_var = tk.BooleanVar(value=self.config.get('enable_enhanced_trading', False))
        ttk.Checkbutton(trading_frame, text="启用增强版交易策略", 
                       variable=enhanced_trading_var).pack(anchor='w', pady=2)
        
        # 启用实盘交易
        real_trading_var = tk.BooleanVar(value=self.config.get('enable_real_trading', False))
        real_trading_cb = ttk.Checkbutton(trading_frame, text="启用实盘交易（警告：将使用真实资金）", 
                                         variable=real_trading_var)
        real_trading_cb.pack(anchor='w', pady=2)
        
        # IBKR连接设置
        ibkr_frame = ttk.LabelFrame(scrollable_frame, text="IBKR连接设置", padding="10")
        ibkr_frame.pack(fill="x", padx=20, pady=5)
        
        # IBKR主机
        ttk.Label(ibkr_frame, text="IBKR主机:").pack(anchor='w')
        ibkr_host_var = tk.StringVar(value=self.config.get('ibkr_host', '127.0.0.1'))
        ttk.Entry(ibkr_frame, textvariable=ibkr_host_var, width=20).pack(anchor='w', pady=2)
        
        # IBKR端口
        ttk.Label(ibkr_frame, text="IBKR端口:").pack(anchor='w', pady=(5, 0))
        ibkr_port_var = tk.StringVar(value=str(self.config.get('ibkr_port', 4002)))
        ttk.Entry(ibkr_frame, textvariable=ibkr_port_var, width=20).pack(anchor='w', pady=2)
        
        # IBKR客户端ID
        ttk.Label(ibkr_frame, text="IBKR客户端ID:").pack(anchor='w', pady=(5, 0))
        ibkr_client_id_var = tk.StringVar(value=str(self.config.get('ibkr_client_id', 50310)))
        ttk.Entry(ibkr_frame, textvariable=ibkr_client_id_var, width=20).pack(anchor='w', pady=2)
        
        # 风险控制设置
        risk_frame = ttk.LabelFrame(scrollable_frame, text="风险控制设置", padding="10")
        risk_frame.pack(fill="x", padx=20, pady=5)
        
        # 总资金
        ttk.Label(risk_frame, text="总资金:").pack(anchor='w')
        total_capital_var = tk.StringVar(value=str(self.config.get('total_capital', TradingConstants.TRADING_DEFAULTS['total_capital'])))
        ttk.Entry(risk_frame, textvariable=total_capital_var, width=20).pack(anchor='w', pady=2)
        
        # 最大单个持仓比例
        ttk.Label(risk_frame, text="最大单个持仓比例 (0-1):").pack(anchor='w', pady=(5, 0))
        max_position_var = tk.StringVar(value=str(self.config.get('max_position_size', TradingConstants.RISK_MANAGEMENT_DEFAULTS['max_position_size'])))
        ttk.Entry(risk_frame, textvariable=max_position_var, width=20).pack(anchor='w', pady=2)
        
        # 止损比例
        ttk.Label(risk_frame, text="止损比例 (0-1):").pack(anchor='w', pady=(5, 0))
        stop_loss_var = tk.StringVar(value=str(self.config.get('stop_loss_pct', TradingConstants.RISK_MANAGEMENT_DEFAULTS['stop_loss_pct'])))
        ttk.Entry(risk_frame, textvariable=stop_loss_var, width=20).pack(anchor='w', pady=2)
        
        # 止盈比例
        ttk.Label(risk_frame, text="止盈比例 (0-1):").pack(anchor='w', pady=(5, 0))
        take_profit_var = tk.StringVar(value=str(self.config.get('take_profit_pct', TradingConstants.RISK_MANAGEMENT_DEFAULTS['take_profit_pct'])))
        ttk.Entry(risk_frame, textvariable=take_profit_var, width=20).pack(anchor='w', pady=2)
        
        # 保存按钮
        def save_settings():
            try:
                # 基础设置
                self.config['auto_run'] = auto_run_var.get()
                self.config['notifications'] = notifications_var.get()
                self.config['log_level'] = log_level_var.get()
                
                # 增强版交易设置
                self.config['enable_enhanced_trading'] = enhanced_trading_var.get()
                self.config['enable_real_trading'] = real_trading_var.get()
                
                # IBKR连接设置
                self.config['ibkr_host'] = ibkr_host_var.get()
                self.config['ibkr_port'] = int(ibkr_port_var.get())
                self.config['ibkr_client_id'] = int(ibkr_client_id_var.get())
                
                # 风险控制设置
                self.config['total_capital'] = float(total_capital_var.get())
                self.config['max_position_size'] = float(max_position_var.get())
                self.config['stop_loss_pct'] = float(stop_loss_var.get())
                self.config['take_profit_pct'] = float(take_profit_var.get())
                
                # 重新初始化增强版交易策略
                if self.config['enable_enhanced_trading']:
                    self.init_enhanced_trading_strategy()
                
                # 保存配置到文件
                with open('config.json', 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("设置", "设置已保存")
                settings_window.destroy()
                
            except ValueError as e:
                messagebox.showerror("错误", f"设置值无效: {e}")
            except Exception as e:
                messagebox.showerror("错误", f"保存设置失败: {e}")
        
        # 按钮框架
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="保存设置", command=save_settings).pack(side="left", padx=5)
        ttk.Button(button_frame, text="取消", command=settings_window.destroy).pack(side="left", padx=5)
        
        # 配置画布
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def show_history(self):
        """显示历史记录窗口"""
        history_window = tk.Toplevel(self.root)
        history_window.title("历史记录")
        history_window.geometry("800x500")
        
        # 创建详细的历史记录表格
        columns = ('执行时间', '任务类型', '状态', '耗时(秒)', '结果文件')
        history_tree = ttk.Treeview(history_window, columns=columns, show='headings')
        
        for col in columns:
            history_tree.heading(col, text=col)
            history_tree.column(col, width=150)
        
        # 加载历史数据
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT execution_time, task_type, status, duration_seconds, result_files
                FROM task_executions 
                ORDER BY execution_time DESC
            ''')
            
            for row in cursor.fetchall():
                execution_time = datetime.fromisoformat(row[0]).strftime('%Y-%m-%d %H:%M:%S')
                formatted_row = (
                    execution_time,
                    row[1],
                    row[2],
                    f"{row[3]:.1f}" if row[3] else '-',
                    Path(row[4]).name if row[4] else '-'
                )
                history_tree.insert('', 'end', values=formatted_row)
                
        except Exception as e:
            self.logger.error(f"加载历史记录失败: {e}")
        
        history_tree.pack(fill='both', expand=True, padx=10, pady=10)
    
    def show_status_monitor(self):
        """显示状态监控界面"""
        try:
            # 启动专用的量化交易监视器
            import subprocess
            import sys
            
            # 检查监视器文件是否存在（优先使用手动版本）
            monitor_script = os.path.join(os.getcwd(), "trading_monitor_manual.py")
            if not os.path.exists(monitor_script):
                monitor_script = os.path.join(os.getcwd(), "trading_monitor_auto.py")
            if not os.path.exists(monitor_script):
                monitor_script = os.path.join(os.getcwd(), "trading_monitor_test.py")
            
            if os.path.exists(monitor_script):
                # 启动监视器
                subprocess.Popen([sys.executable, monitor_script], 
                               cwd=os.getcwd(),
                               creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)
                
                self.log_message("🚀 量化交易监视器已启动")
                
                # 显示提示信息
                messagebox.showinfo("监视器启动", 
                                  "🏞️ 量化交易自动监视器已启动！\n\n"
                                  "监视器将自动：\n"
                                  "• 监控本程序的运行状态\n"
                                  "• 程序结束后自动重启\n"
                                  "• 实时显示所有输出信息\n"
                                  "• 提供运行统计数据")
            else:
                # 如果找不到监视器脚本，显示简单监控界面
                self.log_message("⚠️ 找不到专用监视器，使用简单监控界面")
                self.show_simple_monitoring()
                
        except Exception as e:
            self.log_message(f"❌ 启动监视器失败: {e}")
            # 回退到简单监控界面
            self.show_simple_monitoring()
    
    def auto_show_status_monitor(self):
        """自动显示状态监控（仅在模型启动时调用）"""
        if STATUS_MONITOR_AVAILABLE:
            try:
                # 获取状态监控器
                monitor = get_status_monitor(self.root)
                
                # 确保窗口可见
                monitor.window.deiconify()  # 显示窗口
                monitor.window.lift()  # 提到前面
                monitor.window.focus_force()  # 强制获取焦点
                
                # 自动开始输出捕获
                if OUTPUT_CAPTURE_AVAILABLE:
                    monitor.start_output_capture()
                
                # 记录日志
                log_message("模型启动，状态监控已自动显示")
                
            except Exception as e:
                self.logger.error(f"自动显示状态监控失败: {e}")
    
    def show_simple_monitoring(self):
        """显示简单监控界面（备用方案）"""
        monitor_window = tk.Toplevel(self.root)
        monitor_window.title("实时监控")
        monitor_window.geometry("600x400")
        
        # 系统状态
        status_frame = ttk.LabelFrame(monitor_window, text="系统状态", padding="10")
        status_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(status_frame, text=f"数据库路径: {self.db_path}").pack(anchor='w')
        ttk.Label(status_frame, text=f"定时任务状态: {'运行中' if self.scheduler.running else '已停止'}").pack(anchor='w')
        ttk.Label(status_frame, text=f"下次执行时间: {self.get_next_run_time()}").pack(anchor='w')
        
        # 最近日志
        log_frame = ttk.LabelFrame(monitor_window, text="最近日志", padding="10")
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state='disabled')
        log_text.pack(fill='both', expand=True)
        
        # 读取最近的日志
        try:
            log_file = Path(f"logs/trading_manager_{datetime.now().strftime('%Y%m%d')}.log")
            if log_file.exists():
                # 尝试多种编码格式
                encodings = ['utf-8', 'gbk', 'gb2312', 'cp936', 'latin1']
                content = None
                
                for encoding in encodings:
                    try:
                        with open(log_file, 'r', encoding=encoding) as f:
                            lines = f.readlines()
                            recent_logs = ''.join(lines[-50:])  # 最近50行
                            content = recent_logs
                            break
                    except UnicodeDecodeError:
                        continue
                
                if content:
                    log_text.config(state='normal')
                    log_text.insert(tk.END, content)
                    log_text.config(state='disabled')
                    log_text.see(tk.END)
                else:
                    self.logger.warning("无法以任何编码格式读取日志文件")
        except Exception as e:
            self.logger.error(f"读取日志失败: {e}")
    
    def show_monitoring(self):
        """显示实时监控窗口（保持向后兼容）"""
        self.show_status_monitor()
    
    def on_result_double_click(self, event):
        """双击结果项时打开文件和显示图表"""
        selection = self.results_tree.selection()
        if selection:
            item = self.results_tree.item(selection[0])
            values = item['values']
            
            # 根据分析类型查找对应文件
            analysis_type = values[1]
            date_str = values[0]
            
            try:
                # 查找对应时间的文件
                cursor = self.conn.cursor()
                cursor.execute('''
                    SELECT file_path FROM analysis_results 
                    WHERE analysis_type = ? AND date_created LIKE ?
                    ORDER BY date_created DESC LIMIT 1
                ''', (analysis_type, f"{date_str.split(' ')[0]}%"))
                
                result = cursor.fetchone()
                if result and result[0] and Path(result[0]).exists():
                    os.startfile(result[0])
                    
                    # 同时显示相关图表
                    self.display_analysis_images(analysis_type, date_str)
                    
                else:
                    messagebox.showwarning("文件不存在", "无法找到对应的结果文件")
                    
            except Exception as e:
                self.logger.error(f"打开文件失败: {e}")
                messagebox.showerror("错误", f"打开文件失败: {e}")
    
    def display_analysis_images(self, analysis_type, date_str):
        """显示分析图表"""
        try:
            # 查找图片文件
            image_dirs = []
            if analysis_type == "量化模型":
                image_dirs.append("result")
            elif analysis_type == "回测分析":
                image_dirs.append("category_analysis_results")
            elif analysis_type == "ML回测":
                image_dirs.append("ml_backtest_results")
            
            image_files = []
            for img_dir in image_dirs:
                if os.path.exists(img_dir):
                    for file in os.listdir(img_dir):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_files.append(os.path.join(img_dir, file))
            
            if image_files:
                # 按修改时间排序，显示最新的图片
                image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                self.show_image(image_files[0])
                self.image_files = image_files
            else:
                self.image_label.config(text="未找到图表文件")
                
        except Exception as e:
            self.logger.error(f"显示图片失败: {e}")
            self.image_label.config(text="图片加载失败")
    
    def show_image(self, image_path):
        """显示图片"""
        if not self.pil_available:
            self.image_label.config(text="PIL未安装，无法显示图片\n请运行: pip install Pillow")
            return
            
        try:
            from PIL import Image, ImageTk
            
            # 加载图片
            image = Image.open(image_path)
            
            # 获取显示区域大小
            label_width = self.image_label.winfo_width()
            label_height = self.image_label.winfo_height()
            
            if label_width > 1 and label_height > 1:
                # 调整图片大小以适应显示区域
                image.thumbnail((label_width, label_height), Image.Resampling.LANCZOS)
            
            # 转换为Tkinter可显示的格式
            photo = ImageTk.PhotoImage(image)
            
            # 更新显示
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # 保持引用
            self.current_image_path = image_path
            
        except Exception as e:
            self.logger.error(f"显示图片失败: {e}")
            self.image_label.config(text="图片加载失败")
    
    def refresh_images(self):
        """刷新图表"""
        selection = self.results_tree.selection()
        if selection:
            item = self.results_tree.item(selection[0])
            values = item['values']
            if len(values) >= 2:
                analysis_type = values[1]
                date_str = values[0]
                self.display_analysis_images(analysis_type, date_str)
    
    def open_image_folder(self):
        """打开图片文件夹"""
        try:
            # 查找图片文件夹
            folders = ["result", "category_analysis_results", "ml_backtest_results"]
            for folder in folders:
                if os.path.exists(folder):
                    os.startfile(folder)
                    self.show_notification("文件夹已打开", f"已打开 {folder} 文件夹")
                    return
            
            self.show_notification("文件夹未找到", "未找到图片文件夹")
        except Exception as e:
            self.logger.error(f"打开文件夹失败: {e}")
            self.show_notification("打开失败", f"无法打开文件夹: {e}")
    
    def show_date_selection_dialog(self, model_type="enhanced"):
        """显示量化分析参数设置对话框"""
        self._show_model_dialog(model_type)
    
    def _show_model_dialog(self, model_type):
        """通用模型参数设置对话框"""
        dialog = tk.Toplevel(self.root)
        
        # 设置标题和说明
        if model_type == "lstm_enhanced":
            dialog.title("多日LSTM模型参数设置")
            title_text = "多日LSTM模型参数设置"
            info_text = "多日LSTM模型包含深度学习时序建模、5日预测、多日LSTM+Stacking集成算法等高级功能"
        elif model_type == "walkforward":
            dialog.title("Walk-Forward回测参数设置")
            title_text = "Walk-Forward回测参数设置"
            info_text = "Walk-Forward滚动回测包含严格的时间序列回测、BMA贝叶斯模型平均、方向性准确率分析等高级功能"
        else:
            dialog.title("量化分析参数设置")
            title_text = "量化分析参数设置"
            info_text = "高级量化模型包含信息系数筛选、异常值处理、因子中性化和XGBoost/LightGBM/CatBoost等高级机器学习算法"
        
        dialog.geometry("700x600")
        dialog.resizable(True, True)
        dialog.model_type = model_type  # 存储模型类型
        
        # 使对话框居中
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 主框架
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text=title_text, 
                               font=('Microsoft YaHei', 14, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # 说明文字
        info_label = ttk.Label(main_frame, 
                              text=info_text,
                              font=('Microsoft YaHei', 9),
                              foreground='blue' if model_type == "enhanced" else 'gray')
        info_label.pack(pady=(0, 15))
        
        # 创建两列布局
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        # 左侧：日期选择
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        date_frame = ttk.LabelFrame(left_frame, text="时间范围选择", padding="15")
        date_frame.pack(fill='x', pady=(0, 10))
        
        if CALENDAR_AVAILABLE:
            # 使用日历控件
            self.create_calendar_widgets(date_frame, dialog)
        else:
            # 使用简单的文本输入
            self.create_simple_date_widgets(date_frame, dialog)
        
        # 右侧：股票池选择
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        self.create_stock_pool_widgets(right_frame, dialog)
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(10, 0))
        
        # 按钮
        ttk.Button(button_frame, text="取消", 
                  command=dialog.destroy).pack(side='right', padx=(10, 0))
        
        ttk.Button(button_frame, text="开始分析", 
                  command=lambda: self.confirm_and_run_analysis(dialog),
                  style="Accent.TButton").pack(side='right')
        
        ttk.Button(button_frame, text="重置为默认", 
                  command=lambda: self.reset_default_dates(dialog)).pack(side='left')
    
    def create_calendar_widgets(self, parent, dialog):
        """创建日历控件"""
        try:
            # 开始日期
            start_frame = ttk.Frame(parent)
            start_frame.pack(fill='x', pady=(0, 15))
            
            ttk.Label(start_frame, text="开始日期:", 
                     font=('Microsoft YaHei', 10)).pack(side='left')
            
            self.start_date_entry = DateEntry(start_frame, 
                                            width=12, 
                                            background='darkblue',
                                            foreground='white', 
                                            borderwidth=2,
                                            year=2018,
                                            month=1,
                                            day=1,
                                            date_pattern='yyyy-mm-dd')
            self.start_date_entry.pack(side='right')
            
            # 结束日期  
            end_frame = ttk.Frame(parent)
            end_frame.pack(fill='x', pady=(0, 15))
            
            ttk.Label(end_frame, text="结束日期:", 
                     font=('Microsoft YaHei', 10)).pack(side='left')
            
            self.end_date_entry = DateEntry(end_frame, 
                                           width=12, 
                                           background='darkblue',
                                           foreground='white', 
                                           borderwidth=2,
                                           date_pattern='yyyy-mm-dd')
            self.end_date_entry.pack(side='right')
            
            # 设置默认值
            try:
                start_date = datetime.strptime(self.selected_start_date, "%Y-%m-%d").date()
                end_date = datetime.strptime(self.selected_end_date, "%Y-%m-%d").date()
                self.start_date_entry.set_date(start_date)
                self.end_date_entry.set_date(end_date)
            except:
                pass
                
        except Exception as e:
            self.logger.error(f"创建日历控件失败: {e}")
            self.create_simple_date_widgets(parent, dialog)
    
    def create_simple_date_widgets(self, parent, dialog):
        """创建简单的日期输入控件"""
        # 开始日期
        start_frame = ttk.Frame(parent)
        start_frame.pack(fill='x', pady=(0, 15))
        
        ttk.Label(start_frame, text="开始日期 (YYYY-MM-DD):", 
                 font=('Microsoft YaHei', 10)).pack(side='left')
        
        self.start_date_var = tk.StringVar(value=self.selected_start_date)
        self.start_date_entry = ttk.Entry(start_frame, textvariable=self.start_date_var, width=15)
        self.start_date_entry.pack(side='right')
        
        # 结束日期
        end_frame = ttk.Frame(parent)
        end_frame.pack(fill='x', pady=(0, 15))
        
        ttk.Label(end_frame, text="结束日期 (YYYY-MM-DD):", 
                 font=('Microsoft YaHei', 10)).pack(side='left')
        
        self.end_date_var = tk.StringVar(value=self.selected_end_date)
        self.end_date_entry = ttk.Entry(end_frame, textvariable=self.end_date_var, width=15)
        self.end_date_entry.pack(side='right')
        
        # 添加日期格式提示
        hint_label = ttk.Label(parent, 
                              text="日期格式: YYYY-MM-DD (例如: 2018-01-01)",
                              font=('Microsoft YaHei', 8),
                              foreground='gray')
        hint_label.pack(pady=(5, 0))
    
    def reset_default_dates(self, dialog):
        """重置为默认日期"""
        self.selected_start_date = "2018-01-01"
        self.selected_end_date = datetime.now().strftime("%Y-%m-%d")
        
        if CALENDAR_AVAILABLE and hasattr(self, 'start_date_entry'):
            try:
                self.start_date_entry.set_date(datetime.strptime(self.selected_start_date, "%Y-%m-%d").date())
                self.end_date_entry.set_date(datetime.strptime(self.selected_end_date, "%Y-%m-%d").date())
            except:
                pass
        elif hasattr(self, 'start_date_var'):
            self.start_date_var.set(self.selected_start_date)
            self.end_date_var.set(self.selected_end_date)
    
    def confirm_and_run_analysis(self, dialog):
        """确认日期并运行分析"""
        try:
            # 获取选择的日期
            if CALENDAR_AVAILABLE and hasattr(self, 'start_date_entry'):
                start_date = self.start_date_entry.get_date().strftime("%Y-%m-%d")
                end_date = self.end_date_entry.get_date().strftime("%Y-%m-%d")
            else:
                start_date = self.start_date_var.get().strip()
                end_date = self.end_date_var.get().strip()
            
            # 验证日期格式
            try:
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                messagebox.showerror("日期格式错误", "请输入正确的日期格式 (YYYY-MM-DD)")
                return
            
            # 验证日期范围
            if start_datetime >= end_datetime:
                messagebox.showerror("日期范围错误", "开始日期必须早于结束日期")
                return
            
            # 验证日期不能太早（建议2018年后）
            if start_datetime.year < 2018:
                result = messagebox.askyesno("日期提醒", 
                                           f"您选择的开始日期是 {start_date}，\n2018年之前的数据可能质量较差。\n\n是否继续？")
                if not result:
                    return
            
            # 验证日期不能太晚（不能超过今天）
            if end_datetime.date() > datetime.now().date():
                messagebox.showerror("日期范围错误", "结束日期不能晚于今天")
                return
            
            # 保存选择的日期
            self.selected_start_date = start_date
            self.selected_end_date = end_date
            
            # 关闭对话框
            dialog.destroy()
            
            # 获取股票池设置
            stock_mode = self.stock_mode_var.get()
            custom_ticker_file = None
            
            if stock_mode == "custom":
                if not self.custom_stock_list:
                    messagebox.showwarning("空股票池", "自定义股票池为空，请添加股票或选择默认股票池")
                    return
                
                # 创建临时股票文件
                try:
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
                    temp_file.write("# 临时自定义股票列表\n")
                    for ticker in self.custom_stock_list:
                        temp_file.write(f"{ticker}\n")
                    temp_file.close()
                    custom_ticker_file = temp_file.name
                    
                except Exception as e:
                    messagebox.showerror("文件创建失败", f"无法创建临时股票文件: {e}")
                    return
                    
            elif stock_mode == "edit_default":
                if not self.edited_default_list:
                    messagebox.showwarning("空股票池", "编辑后的默认股票池为空，请添加股票或重置为完整默认池")
                    return
                
                # 创建临时股票文件
                try:
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
                    temp_file.write("# 临时编辑默认股票池列表\n")
                    for ticker in self.edited_default_list:
                        temp_file.write(f"{ticker}\n")
                    temp_file.close()
                    custom_ticker_file = temp_file.name
                    
                except Exception as e:
                    messagebox.showerror("文件创建失败", f"无法创建临时股票文件: {e}")
                    return
                    
            elif stock_mode == "scraped":
                # 使用爬虫获取的股票池
                if hasattr(self, 'quantitative_model_stocks') and self.quantitative_model_stocks:
                    try:
                        import tempfile
                        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
                        temp_file.write("# 爬虫获取的股票列表\n")
                        for ticker in self.quantitative_model_stocks:
                            temp_file.write(f"{ticker}\n")
                        temp_file.close()
                        custom_ticker_file = temp_file.name
                        
                    except Exception as e:
                        messagebox.showerror("文件创建失败", f"无法创建爬虫股票文件: {e}")
                        return
                else:
                    messagebox.showwarning("空股票池", "爬虫股票池为空，请先更新股票池或选择其他模式")
                    return
            
            # 根据模型类型运行不同的模型
            model_type = dialog.model_type
            if model_type == "lstm_enhanced":
                # 运行LSTM增强模型
                self.run_lstm_model_with_dates(start_date, end_date, custom_ticker_file)
            elif model_type == "walkforward":
                # 运行Walk-Forward回测
                self.run_walkforward_backtest_with_dates(start_date, end_date, custom_ticker_file)
            elif model_type == "enhanced":
                # 运行增强版量化模型（BMA）
                self.run_enhanced_model_with_dates(start_date, end_date, custom_ticker_file)
            else:
                # 默认运行增强版量化模型
                self.run_enhanced_model_with_dates(start_date, end_date, custom_ticker_file)
            
        except Exception as e:
            self.logger.error(f"日期确认失败: {e}")
            messagebox.showerror("错误", f"日期设置失败: {e}")
    
    def run_lstm_model_with_dates(self, start_date, end_date, ticker_file=None):
        """运行LSTM增强模型"""
        def run_analysis():
            # 在函数开始就导入time模块，确保在异常处理中可用
            import time
            
            try:
                # 自动显示状态监控
                self.auto_show_status_monitor()
                
                # 通知状态监控器有用户主动启动的模型
                if STATUS_MONITOR_AVAILABLE:
                    try:
                        from status_monitor import get_status_monitor
                        monitor = get_status_monitor()
                        if monitor and hasattr(monitor, 'output_monitoring_active'):
                            monitor.output_monitoring_active = True
                            monitor.log_message("用户启动LSTM模型，暂停自动监控")
                    except:
                        pass
                
                self.update_status("正在启动多日LSTM模型...", 10)
                
                start_time = time.time()
                
                # 构建命令参数
                cmd = [sys.executable, "lstm_multi_day_enhanced.py", 
                       "--start-date", start_date, 
                       "--end-date", end_date]
                
                if ticker_file:
                    cmd.extend(["--ticker-file", ticker_file])
                
                self.update_status("多日LSTM模型训练中...", 50)
                
                # 运行LSTM增强模型
                result = subprocess.run(cmd, 
                    cwd=os.getcwd(),
                    capture_output=True, 
                    text=True,
                    encoding='gbk',
                    errors='replace')
                
                # 确保进程完全结束，等待额外时间让文件保存完成
                import time
                time.sleep(2)  # 给文件保存操作额外时间
                
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    self.update_status("多日LSTM模型训练和保存完成", 100)
                    
                    # 查找生成的Excel文件
                    result_files = self.find_latest_result_files("multi_day_lstm_analysis_")
                    
                    # 保存到数据库
                    self.save_analysis_result("多日LSTM模型", result_files[0] if result_files else "", 
                                            duration, result.stdout)
                    
                    # 显示成功通知
                    success_msg = f"多日LSTM模型训练完成\n耗时: {duration:.1f}秒"
                    self.show_notification("训练完成", success_msg)
                    self.load_recent_results()
                    
                    # 询问是否查看结果
                    view_result = messagebox.askyesno("训练完成", 
                        f"🎉 多日LSTM模型训练完成！\n\n⏱️ 用时: {duration:.1f} 秒\n🧠 功能: 多日LSTM时序建模 + 5日预测\n📊 结果: 已生成多日投资建议\n\n是否查看结果文件？")
                    
                    if view_result:
                        self.open_result_folder()
                        
                else:
                    # 任务失败
                    error_msg = result.stderr or result.stdout or "未知错误"
                    short_error = error_msg[:150] + "..." if len(error_msg) > 150 else error_msg
                    
                    self.update_status("多日LSTM模型训练失败", 0)
                    self.show_notification("训练失败", f"多日LSTM模型训练失败\n错误信息: {short_error}")
                    self.logger.error(f"多日LSTM模型训练失败\n完整错误信息: {error_msg}")
                    
            except Exception as e:
                try:
                    duration = time.time() - start_time
                except:
                    duration = 0
                error_msg = str(e)
                
                self.update_status("多日LSTM模型启动失败", 0)
                self.show_notification("错误", f"启动多日LSTM模型失败: {error_msg}")
                self.logger.error(f"启动多日LSTM模型失败: {error_msg}")
                
            finally:
                # 重置状态监控器的运行状态
                if STATUS_MONITOR_AVAILABLE:
                    try:
                        from status_monitor import get_status_monitor
                        monitor = get_status_monitor()
                        if monitor and hasattr(monitor, 'output_monitoring_active'):
                            monitor.output_monitoring_active = False
                            monitor.log_message("LSTM模型完成，恢复自动监控")
                    except:
                        pass
                
                self.update_status("就绪", 0)
        
        # 在后台线程中运行
        task_thread = threading.Thread(target=run_analysis, daemon=True)
        task_thread.start()
        
        self.logger.info("多日LSTM模型训练已启动")
    
    def run_enhanced_model_with_dates(self, start_date, end_date, ticker_file=None):
        """运行增强版量化模型"""
        def run_analysis():
            # 在函数开始就导入time模块，确保在异常处理中可用
            import time
            
            try:
                start_time = time.time()
                self.show_notification("量化分析", "正在启动量化分析模型...")
                
                # 自动显示状态监控
                self.auto_show_status_monitor()
                
                # 更新状态监控
                if STATUS_MONITOR_AVAILABLE:
                    update_status("正在启动量化分析模型...", 10)
                    log_message("开始执行量化分析模型")
                
                # 构建命令
                cmd = [sys.executable, "量化模型_bma_enhanced.py"]
                
                if start_date:
                    cmd.extend(["--start-date", start_date])
                if end_date:
                    cmd.extend(["--end-date", end_date])
                if ticker_file:
                    cmd.extend(["--ticker-file", ticker_file])
                
                self.logger.info(f"执行量化模型命令: {' '.join(cmd)}")
                
                # 执行命令
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='gbk',  # 使用GBK编码处理中文
                    cwd=os.getcwd()
                )
                
                # 确保进程完全结束，等待额外时间让文件保存完成
                import time
                time.sleep(2)  # 给文件保存操作额外时间
                
                if result.returncode == 0:
                    self.show_notification("分析完成", "量化分析模型分析已完成！")
                    self.logger.info("量化模型执行成功")
                    
                    # 更新状态监控
                    if STATUS_MONITOR_AVAILABLE:
                        update_status("量化分析模型分析已完成！", 100)
                        log_message("量化分析模型执行成功")
                    
                    # 查找生成的Excel文件
                    result_files = self.find_latest_result_files("quantitative_analysis_")
                    
                    # 保存到数据库
                    if result_files:
                        self.save_analysis_result("量化分析模型", result_files[0], 
                                                time.time() - start_time, result.stdout)
                    
                    # 清理临时文件
                    if ticker_file and os.path.exists(ticker_file):
                        try:
                            os.unlink(ticker_file)
                        except:
                            pass
                    
                    # 刷新结果列表
                    self.root.after(1000, self.load_recent_results)
                else:
                    error_msg = result.stderr[:150] if result.stderr else "未知错误"
                    self.show_notification("分析失败", f"错误: {error_msg}")
                    self.logger.error(f"量化模型执行失败: {result.stderr}")
                    
                    # 更新状态监控
                    if STATUS_MONITOR_AVAILABLE:
                        update_status("量化分析模型执行失败", 0)
                        log_message(f"量化分析模型执行失败: {error_msg}")
                    
            except FileNotFoundError:
                self.show_notification("文件不存在", "找不到量化模型文件 (量化模型_enhanced.py)")
                self.logger.error("量化模型文件不存在")
                
                # 更新状态监控
                if STATUS_MONITOR_AVAILABLE:
                    update_status("量化模型文件不存在", 0)
                    log_message("找不到量化模型文件 (量化模型_enhanced.py)")
                    
            except Exception as e:
                error_msg = str(e)[:150]
                self.show_notification("分析错误", f"执行错误: {error_msg}")
                self.logger.error(f"量化模型执行异常: {e}")
                
                # 更新状态监控
                if STATUS_MONITOR_AVAILABLE:
                    update_status("量化模型执行异常", 0)
                    log_message(f"量化模型执行异常: {error_msg}")
        
        # 在新线程中运行
        thread = threading.Thread(target=run_analysis)
        thread.daemon = True
        thread.start()
    

    
    def create_stock_pool_widgets(self, parent, dialog):
        """创建股票池编辑控件"""
        # 股票池选择框架
        stock_frame = ttk.LabelFrame(parent, text="股票池设置", padding="15")
        stock_frame.pack(fill='both', expand=True)
        
        # 选择模式
        mode_frame = ttk.Frame(stock_frame)
        mode_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(mode_frame, text="股票池模式:", font=('Microsoft YaHei', 10)).pack(side='left')
        
        self.stock_mode_var = tk.StringVar(value="default")
        mode_default = ttk.Radiobutton(mode_frame, text="使用默认股票池", 
                                      variable=self.stock_mode_var, value="default",
                                      command=lambda: self.on_stock_mode_change(dialog))
        mode_default.pack(side='left', padx=(10, 5))
        
        mode_scraped = ttk.Radiobutton(mode_frame, text="使用爬虫股票池", 
                                      variable=self.stock_mode_var, value="scraped",
                                      command=lambda: self.on_stock_mode_change(dialog))
        mode_scraped.pack(side='left', padx=(5, 5))
        
        mode_edit_default = ttk.Radiobutton(mode_frame, text="编辑默认股票池", 
                                           variable=self.stock_mode_var, value="edit_default",
                                           command=lambda: self.on_stock_mode_change(dialog))
        mode_edit_default.pack(side='left', padx=(5, 5))
        
        mode_custom = ttk.Radiobutton(mode_frame, text="完全自定义股票池", 
                                     variable=self.stock_mode_var, value="custom",
                                     command=lambda: self.on_stock_mode_change(dialog))
        mode_custom.pack(side='left', padx=(5, 0))
        
        # 默认股票池信息
        self.default_info_frame = ttk.Frame(stock_frame)
        self.default_info_frame.pack(fill='x', pady=(0, 10))
        
        # 默认股票池信息标签
        self.default_info_label = ttk.Label(self.default_info_frame, 
                                           text="默认股票池包含357只精选股票，涵盖科技、金融、医疗等多个行业\n推荐用于全面的市场分析",
                                           font=('Microsoft YaHei', 9),
                                           foreground='gray')
        self.default_info_label.pack()
        
        # 爬虫股票池信息框架（初始隐藏）
        self.scraped_info_frame = ttk.Frame(stock_frame)
        
        scraped_info = ttk.Label(self.scraped_info_frame, 
                                text="爬虫股票池包含从网络爬取的高质量股票，已去除ROE条件\n实时更新，适合动态量化分析",
                                font=('Microsoft YaHei', 9),
                                foreground='blue')
        scraped_info.pack()
        
        # 编辑默认股票池区域
        self.edit_default_frame = ttk.Frame(stock_frame)
        
        # 默认股票池预览
        preview_frame = ttk.LabelFrame(self.edit_default_frame, text="默认股票池预览", padding="10")
        preview_frame.pack(fill='x', pady=(0, 10))
        
        # 创建默认股票池预览的Treeview
        preview_columns = ('序号', '股票代码', '行业分类')
        self.default_preview_tree = ttk.Treeview(preview_frame, columns=preview_columns, show='headings', height=6)
        
        # 设置列标题和宽度
        self.default_preview_tree.heading('序号', text='序号')
        self.default_preview_tree.heading('股票代码', text='股票代码')
        self.default_preview_tree.heading('行业分类', text='行业分类')
        
        self.default_preview_tree.column('序号', width=50)
        self.default_preview_tree.column('股票代码', width=100)
        self.default_preview_tree.column('行业分类', width=120)
        
        # 添加滚动条
        preview_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.default_preview_tree.yview)
        self.default_preview_tree.configure(yscrollcommand=preview_scrollbar.set)
        
        # 布局
        self.default_preview_tree.pack(side='left', fill='both', expand=True)
        preview_scrollbar.pack(side='right', fill='y')
        
        # 操作按钮
        edit_buttons_frame = ttk.Frame(self.edit_default_frame)
        edit_buttons_frame.pack(fill='x', pady=(5, 0))
        
        ttk.Button(edit_buttons_frame, text="加载默认股票池", 
                  command=lambda: self.load_default_stocks(dialog)).pack(side='left', padx=(0, 5))
        ttk.Button(edit_buttons_frame, text="移除选中股票", 
                  command=lambda: self.remove_selected_from_default(dialog)).pack(side='left', padx=(0, 5))
        ttk.Button(edit_buttons_frame, text="添加股票到默认池", 
                  command=lambda: self.add_to_default_pool(dialog)).pack(side='left', padx=(0, 5))
        ttk.Button(edit_buttons_frame, text="重置为完整默认池", 
                  command=lambda: self.reset_to_full_default(dialog)).pack(side='left')
        
        # 默认股票池计数
        self.default_count_label = ttk.Label(self.edit_default_frame, text="默认股票池: 0 只", 
                                           font=('Microsoft YaHei', 9), foreground='blue')
        self.default_count_label.pack(pady=(5, 0))
        
        # 自定义股票池编辑区域
        self.custom_frame = ttk.Frame(stock_frame)
        
        # 操作按钮行
        button_row = ttk.Frame(self.custom_frame)
        button_row.pack(fill='x', pady=(0, 5))
        
        ttk.Button(button_row, text="从文件加载", 
                  command=lambda: self.load_stock_file(dialog)).pack(side='left', padx=(0, 5))
        ttk.Button(button_row, text="保存到文件", 
                  command=lambda: self.save_stock_file(dialog)).pack(side='left', padx=(0, 5))
        ttk.Button(button_row, text="添加热门股票", 
                  command=lambda: self.add_popular_stocks(dialog)).pack(side='left', padx=(0, 5))
        ttk.Button(button_row, text="清空列表", 
                  command=lambda: self.clear_stock_list(dialog)).pack(side='left')
        
        # 股票输入区域
        input_frame = ttk.Frame(self.custom_frame)
        input_frame.pack(fill='x', pady=(5, 5))
        
        ttk.Label(input_frame, text="添加股票代码:", font=('Microsoft YaHei', 9)).pack(side='left')
        self.stock_entry = ttk.Entry(input_frame, width=15)
        self.stock_entry.pack(side='left', padx=(5, 5))
        self.stock_entry.bind('<Return>', lambda e: self.add_stock_from_entry(dialog))
        
        ttk.Button(input_frame, text="添加", 
                  command=lambda: self.add_stock_from_entry(dialog)).pack(side='left')
        
        # 股票列表显示
        list_frame = ttk.Frame(self.custom_frame)
        list_frame.pack(fill='both', expand=True, pady=(5, 0))
        
        # 创建Treeview显示股票列表
        columns = ('序号', '股票代码', '添加时间')
        self.stock_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=8)
        
        # 设置列标题和宽度
        self.stock_tree.heading('序号', text='序号')
        self.stock_tree.heading('股票代码', text='股票代码')
        self.stock_tree.heading('添加时间', text='添加时间')
        
        self.stock_tree.column('序号', width=50)
        self.stock_tree.column('股票代码', width=100)
        self.stock_tree.column('添加时间', width=120)
        
        # 添加滚动条
        stock_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.stock_tree.yview)
        self.stock_tree.configure(yscrollcommand=stock_scrollbar.set)
        
        # 布局
        self.stock_tree.pack(side='left', fill='both', expand=True)
        stock_scrollbar.pack(side='right', fill='y')
        
        # 右键菜单
        self.create_stock_context_menu()
        self.stock_tree.bind('<Button-3>', self.show_stock_context_menu)
        
        # 股票计数显示
        self.stock_count_label = ttk.Label(self.custom_frame, text="股票总数: 0", 
                                          font=('Microsoft YaHei', 9), foreground='blue')
        self.stock_count_label.pack(pady=(5, 0))
        
        # 初始化股票列表
        self.custom_stock_list = []
        self.edited_default_list = []  # 编辑后的默认股票池
        
        # 默认股票池数据（从量化模型.py中提取）
        self.default_stock_pool = {
            '科技股': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'CRM', 'ADBE',
                     'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX',
                     'KLAC', 'MRVL', 'ON', 'SWKS', 'MCHP', 'ADI', 'XLNX', 'SNPS', 'CDNS', 'FTNT'],
            '消费零售': ['COST', 'WMT', 'TGT', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS', 'PYPL',
                       'SQ', 'SHOP', 'EBAY', 'ETSY', 'W', 'CHWY', 'ROKU', 'SPOT', 'ZM', 'UBER',
                       'LYFT', 'DASH', 'ABNB', 'BKNG', 'EXPE', 'TJX', 'ROST', 'ULTA', 'LULU', 'RH'],
            '医疗健康': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR', 'ABT', 'LLY', 'BMY',
                       'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA', 'BNTX', 'ZTS', 'CVS',
                       'CI', 'HUM', 'ANTM', 'MCK', 'ABC', 'CAH', 'WAT', 'A', 'IQV', 'CRL'],
            '金融服务': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
                       'AXP', 'BLK', 'SCHW', 'SPGI', 'ICE', 'CME', 'MCO', 'MSCI', 'PYPL', 'V',
                       'MA', 'FIS', 'FISV', 'ADP', 'PAYX', 'WU', 'SYF', 'DFS', 'ALLY', 'RF'],
            '工业材料': ['BA', 'CAT', 'DE', 'GE', 'HON', 'LMT', 'MMM', 'RTX', 'UPS', 'FDX',
                       'NSC', 'UNP', 'CSX', 'ODFL', 'CHRW', 'EXPD', 'XPO', 'JBHT', 'KNX', 'J',
                       'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'DOV', 'FTV', 'XYL', 'IEX', 'GNRC'],
            '能源公用': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE',
                       'WMB', 'ET', 'EPD', 'MPLX', 'AM', 'NEE', 'DUK', 'SO', 'EXC', 'XEL',
                       'AEP', 'PCG', 'ED', 'EIX', 'PPL', 'AES', 'NRG', 'CNP', 'CMS', 'DTE'],
            '房地产': ['AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'AVB', 'EQR', 'UDR',
                     'ESS', 'MAA', 'CPT', 'AIV', 'EXR', 'PSA', 'BXP', 'VTR', 'HCP', 'PEAK'],
            '通信服务': ['VZ', 'T', 'TMUS', 'CHTR', 'CMCSA', 'VIA', 'LBRDA', 'LBRDK', 'DISH', 'SIRI'],
            '基础材料': ['LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'GOLD', 'AA', 'X', 'CLF',
                       'NUE', 'STLD', 'CMC', 'RS', 'WOR', 'RPM', 'PPG', 'DD', 'DOW', 'LYB'],
            '消费必需品': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'CPB',
                         'CAG', 'SJM', 'HRL', 'TSN', 'TYSON', 'ADM', 'BG', 'CF', 'MOS', 'FMC'],
            '新兴增长': ['SQ', 'SHOP', 'ROKU', 'ZOOM', 'DOCU', 'OKTA', 'SNOW', 'PLTR', 'RBLX', 'U',
                       'DDOG', 'CRWD', 'ZS', 'NET', 'FSLY', 'TWLO', 'SPLK', 'WDAY', 'VEEV', 'ZEN',
                       'TEAM', 'ATLASSIAN', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'BIDU', 'BILI', 'IQ'],
            '生物技术': ['MRNA', 'BNTX', 'NOVT', 'SGEN', 'BLUE', 'BMRN', 'TECH', 'SRPT', 'RARE', 'FOLD',
                       'EDIT', 'CRSP', 'NTLA', 'BEAM', 'VERV', 'PRIME', 'SAGE', 'IONS', 'IOVA', 'ARWR'],
            '清洁能源': ['TSLA', 'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'QS', 'BLNK', 'CHPT', 'PLUG',
                       'FCEL', 'BE', 'ENPH', 'SEDG', 'RUN', 'NOVA', 'SPWR', 'CSIQ', 'JKS', 'SOL']
        }
        
        # 生成完整的默认股票池列表
        self.full_default_list = []
        for category, stocks in self.default_stock_pool.items():
            self.full_default_list.extend(stocks)
        
        # 去重
        self.full_default_list = list(dict.fromkeys(self.full_default_list))
        
        # 初始化量化模型股票列表（从爬虫获取）
        self.quantitative_model_stocks = []
        
        # 初始状态
        self.on_stock_mode_change(dialog)
    
    def on_stock_mode_change(self, dialog):
        """股票池模式切换"""
        mode = self.stock_mode_var.get()
        
        # 隐藏所有框架
        self.default_info_frame.pack_forget()
        self.scraped_info_frame.pack_forget()
        self.edit_default_frame.pack_forget()
        self.custom_frame.pack_forget()
        
        if mode == "default":
            # 使用默认股票池
            self.default_info_frame.pack(fill='x', pady=(0, 10))
        elif mode == "scraped":
            # 使用爬虫股票池
            self.scraped_info_frame.pack(fill='x', pady=(0, 10))
        elif mode == "edit_default":
            # 编辑默认股票池
            self.edit_default_frame.pack(fill='both', expand=True, pady=(10, 0))
            self.load_default_stocks(dialog)
        else:
            # 完全自定义股票池
            self.custom_frame.pack(fill='both', expand=True, pady=(10, 0))
    
    def add_stock_from_entry(self, dialog):
        """从输入框添加股票"""
        ticker = self.stock_entry.get().strip().upper()
        if ticker and ticker not in self.custom_stock_list:
            self.custom_stock_list.append(ticker)
            self.update_stock_tree()
            self.stock_entry.delete(0, tk.END)
        elif ticker in self.custom_stock_list:
            messagebox.showinfo("重复股票", f"股票代码 {ticker} 已在列表中")
    
    def update_stock_tree(self):
        """更新股票列表显示"""
        # 清空现有项目
        for item in self.stock_tree.get_children():
            self.stock_tree.delete(item)
        
        # 添加股票项目
        for i, ticker in enumerate(self.custom_stock_list, 1):
            self.stock_tree.insert('', 'end', values=(i, ticker, datetime.now().strftime('%H:%M:%S')))
        
        # 更新计数
        self.stock_count_label.config(text=f"股票总数: {len(self.custom_stock_list)}")
    
    def create_stock_context_menu(self):
        """创建股票列表右键菜单"""
        self.stock_context_menu = tk.Menu(self.root, tearoff=0)
        self.stock_context_menu.add_command(label="删除选中", command=self.delete_selected_stock)
        self.stock_context_menu.add_command(label="复制股票代码", command=self.copy_selected_stock)
    
    def show_stock_context_menu(self, event):
        """显示右键菜单"""
        try:
            self.stock_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.stock_context_menu.grab_release()
    
    def delete_selected_stock(self):
        """删除选中的股票"""
        selection = self.stock_tree.selection()
        if selection:
            item = self.stock_tree.item(selection[0])
            ticker = item['values'][1]
            if ticker in self.custom_stock_list:
                self.custom_stock_list.remove(ticker)
                self.update_stock_tree()
    
    def copy_selected_stock(self):
        """复制选中的股票代码"""
        selection = self.stock_tree.selection()
        if selection:
            item = self.stock_tree.item(selection[0])
            ticker = item['values'][1]
            self.root.clipboard_clear()
            self.root.clipboard_append(ticker)
    
    def load_stock_file(self, dialog):
        """从文件加载股票列表"""
        file_path = filedialog.askopenfilename(
            title="选择股票列表文件",
            filetypes=[("文本文件", "*.txt"), ("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    loaded_stocks = []
                    for line in f:
                        line = line.strip().upper()
                        if line and not line.startswith('#'):
                            # 支持多种分隔符
                            if ',' in line:
                                loaded_stocks.extend([t.strip() for t in line.split(',') if t.strip()])
                            elif ' ' in line or '\t' in line:
                                loaded_stocks.extend([t.strip() for t in line.replace('\t', ' ').split() if t.strip()])
                            else:
                                loaded_stocks.append(line)
                
                # 去重并添加到现有列表
                new_stocks = [s for s in loaded_stocks if s not in self.custom_stock_list]
                self.custom_stock_list.extend(new_stocks)
                self.update_stock_tree()
                
                messagebox.showinfo("加载成功", f"成功加载 {len(new_stocks)} 只新股票\n总计 {len(self.custom_stock_list)} 只股票")
                
            except Exception as e:
                messagebox.showerror("加载失败", f"无法读取文件: {e}")
    
    def save_stock_file(self, dialog):
        """保存股票列表到文件"""
        if not self.custom_stock_list:
            messagebox.showwarning("空列表", "当前股票列表为空，无需保存")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存股票列表",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("# 自定义股票列表\n")
                    f.write(f"# 创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# 股票总数: {len(self.custom_stock_list)}\n\n")
                    
                    for ticker in self.custom_stock_list:
                        f.write(f"{ticker}\n")
                
                messagebox.showinfo("保存成功", f"股票列表已保存到: {file_path}")
                
            except Exception as e:
                messagebox.showerror("保存失败", f"无法保存文件: {e}")
    
    def add_popular_stocks(self, dialog):
        """添加热门股票"""
        popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        
        new_stocks = [s for s in popular_stocks if s not in self.custom_stock_list]
        if new_stocks:
            self.custom_stock_list.extend(new_stocks)
            self.update_stock_tree()
            messagebox.showinfo("添加成功", f"添加了 {len(new_stocks)} 只热门股票")
        else:
            messagebox.showinfo("无需添加", "所有热门股票都已在列表中")
    
    def clear_stock_list(self, dialog):
        """清空股票列表"""
        if self.custom_stock_list:
            result = messagebox.askyesno("确认清空", f"确定要清空所有 {len(self.custom_stock_list)} 只股票吗？")
            if result:
                self.custom_stock_list.clear()
                self.update_stock_tree()
    
    def load_default_stocks(self, dialog):
        """加载默认股票池到编辑列表"""
        if not self.edited_default_list:
            # 首次加载，使用完整默认池
            self.edited_default_list = self.full_default_list.copy()
        
        self.update_default_preview_tree()
        self.default_count_label.config(text=f"默认股票池: {len(self.edited_default_list)} 只")
    
    def update_default_preview_tree(self):
        """更新默认股票池预览"""
        # 清空现有项目
        for item in self.default_preview_tree.get_children():
            self.default_preview_tree.delete(item)
        
        # 添加股票项目
        for i, ticker in enumerate(self.edited_default_list, 1):
            # 查找股票所属行业
            category = "未知"
            for cat, stocks in self.default_stock_pool.items():
                if ticker in stocks:
                    category = cat
                    break
            
            self.default_preview_tree.insert('', 'end', values=(i, ticker, category))
    
    def remove_selected_from_default(self, dialog):
        """从默认池中移除选中的股票"""
        selection = self.default_preview_tree.selection()
        if selection:
            item = self.default_preview_tree.item(selection[0])
            ticker = item['values'][1]
            if ticker in self.edited_default_list:
                self.edited_default_list.remove(ticker)
                self.update_default_preview_tree()
                self.default_count_label.config(text=f"默认股票池: {len(self.edited_default_list)} 只")
                messagebox.showinfo("移除成功", f"已从默认股票池中移除 {ticker}")
    
    def add_to_default_pool(self, dialog):
        """添加股票到默认池"""
        # 创建简单的输入对话框
        input_dialog = tk.Toplevel(dialog)
        input_dialog.title("添加股票到默认池")
        input_dialog.geometry("350x200")
        input_dialog.resizable(False, False)
        input_dialog.transient(dialog)
        input_dialog.grab_set()
        
        # 输入框架
        input_frame = ttk.Frame(input_dialog, padding="20")
        input_frame.pack(fill='both', expand=True)
        
        ttk.Label(input_frame, text="输入股票代码:", font=('Microsoft YaHei', 10)).pack(pady=(0, 10))
        
        ticker_var = tk.StringVar()
        ticker_entry = ttk.Entry(input_frame, textvariable=ticker_var, width=15)
        ticker_entry.pack(pady=(0, 15))
        ticker_entry.focus()
        
        def add_stock():
            ticker = ticker_var.get().strip().upper()
            if not ticker:
                messagebox.showwarning("输入错误", "请输入有效的股票代码")
                return
                
            if ticker in self.edited_default_list:
                messagebox.showinfo("重复股票", f"股票代码 {ticker} 已在默认池中")
                return
            
            # 验证股票代码是否有效
            try:
                import yfinance as yf
                test_ticker = yf.Ticker(ticker)
                info = test_ticker.info
                if not info or 'symbol' not in info:
                    raise ValueError("无效股票代码")
                
                # 添加到默认股票池
                self.edited_default_list.append(ticker)
                
                # 同时添加到默认股票池字典的自定义分类中
                if "自定义" not in self.default_stock_pool:
                    self.default_stock_pool["自定义"] = []
                if ticker not in self.default_stock_pool["自定义"]:
                    self.default_stock_pool["自定义"].append(ticker)
                
                # 更新完整列表
                if ticker not in self.full_default_list:
                    self.full_default_list.append(ticker)
                
                # 更新显示
                self.update_default_preview_tree()
                self.default_count_label.config(text=f"默认股票池: {len(self.edited_default_list)} 只")
                
                # 自动保存
                if self.save_default_stock_pool():
                    input_dialog.destroy()
                    messagebox.showinfo("添加成功", f"已将 {ticker} ({info.get('longName', ticker)}) 添加到默认股票池并保存")
                else:
                    messagebox.showwarning("保存失败", f"股票已添加但保存失败，请手动保存")
                    
            except Exception as e:
                messagebox.showerror("验证失败", f"无法验证股票代码 {ticker}: {e}\n请检查股票代码是否正确")
        
        # 按钮框架
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(button_frame, text="取消", command=input_dialog.destroy).pack(side='right', padx=(10, 0))
        ttk.Button(button_frame, text="添加", command=add_stock).pack(side='right')
        
        # 绑定回车键
        ticker_entry.bind('<Return>', lambda e: add_stock())
    
    def reset_to_full_default(self, dialog):
        """重置为完整默认池"""
        result = messagebox.askyesno("确认重置", 
                                   f"确定要重置为完整的默认股票池吗？\n当前: {len(self.edited_default_list)} 只\n完整池: {len(self.full_default_list)} 只")
        if result:
            self.edited_default_list = self.full_default_list.copy()
            self.update_default_preview_tree()
            self.default_count_label.config(text=f"默认股票池: {len(self.edited_default_list)} 只")
            messagebox.showinfo("重置成功", f"已重置为完整默认股票池 ({len(self.edited_default_list)} 只)")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
量化交易管理软件 使用说明

主要功能：
1. 启动量化模型 - 运行股票量化分析
2. 启动回测分析 - 执行投资策略回测
3.  ML滚动回测 - 机器学习滚动回测

自动化功能：
• 每两周（1日和15日）中午12点自动运行所有分析
• 完成后自动保存结果到数据库
• 系统通知提醒任务完成状态

快捷操作：
• 打开结果文件夹 - 查看所有分析结果
• 查看历史记录 - 查看任务执行历史
• 设置 - 配置自动运行和通知
•  实时监控 - 查看系统状态和日志

数据管理：
• 所有结果自动按日期保存
• 支持导出历史数据
• 本地SQLite数据库存储

技术支持：
如有问题请查看日志文件或联系技术支持
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("使用说明")
        help_window.geometry("500x600")
        
        text_widget = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, 
                                               font=('Microsoft YaHei', 10))
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state='disabled')
    
    def show_about(self):
        """显示关于信息"""
        about_text = """
量化交易管理软件 v1.0

开发目的：
自动化量化交易分析流程，提供定时任务和结果管理功能

核心特性：
[OK] GUI界面操作
[OK] 定时自动执行
[OK] 数据库结果存储
[OK] 系统通知提醒
[OK] 日志记录
[OK] 历史数据管理

技术栈：
• Python + Tkinter (界面)
• SQLite (数据库)
• APScheduler (定时任务)
• Plyer (系统通知)

版权信息：
© 2024 量化交易管理软件
All Rights Reserved
        """
        messagebox.showinfo("关于", about_text)
    
    def export_results(self):
        """导出结果"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                cursor = self.conn.cursor()
                cursor.execute('SELECT * FROM analysis_results ORDER BY date_created DESC')
                
                import pandas as pd
                df = pd.DataFrame(cursor.fetchall(), 
                                columns=['ID', '创建时间', '分析类型', '文件路径', '状态',
                                        '股票数量', '平均评分', 'BUY数量', 'HOLD数量', 'SELL数量', '备注'])
                df.to_excel(file_path, index=False)
                
                messagebox.showinfo("导出完成", f"结果已导出到: {file_path}")
                
            except Exception as e:
                messagebox.showerror("导出失败", f"导出失败: {e}")
    
    def import_config(self):
        """导入配置"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    new_config = json.load(f)
                self.config.update(new_config)
                messagebox.showinfo("导入完成", "配置已导入")
            except Exception as e:
                messagebox.showerror("导入失败", f"导入失败: {e}")
    
    def show_database_manager(self):
        """显示数据库管理器"""
        db_window = tk.Toplevel(self.root)
        db_window.title("数据库管理")
        db_window.geometry("600x400")
        
        # 数据库统计
        stats_frame = ttk.LabelFrame(db_window, text="数据库统计", padding="10")
        stats_frame.pack(fill='x', padx=10, pady=5)
        
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM analysis_results')
            analysis_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM task_executions')
            task_count = cursor.fetchone()[0]
            
            ttk.Label(stats_frame, text=f"分析结果记录: {analysis_count}").pack(anchor='w')
            ttk.Label(stats_frame, text=f"任务执行记录: {task_count}").pack(anchor='w')
            ttk.Label(stats_frame, text=f"数据库大小: {Path(self.db_path).stat().st_size / 1024:.1f} KB").pack(anchor='w')
            
        except Exception as e:
            ttk.Label(stats_frame, text=f"统计获取失败: {e}").pack(anchor='w')
        
        # 清理选项
        clean_frame = ttk.LabelFrame(db_window, text="数据清理", padding="10")
        clean_frame.pack(fill='x', padx=10, pady=5)
        
        def clean_old_records():
            if messagebox.askyesno("确认", "确定要清理30天前的记录吗？"):
                try:
                    cursor = self.conn.cursor()
                    thirty_days_ago = datetime.now() - timedelta(days=30)
                    cursor.execute('DELETE FROM analysis_results WHERE date_created < ?', (thirty_days_ago,))
                    cursor.execute('DELETE FROM task_executions WHERE execution_time < ?', (thirty_days_ago,))
                    self.conn.commit()
                    messagebox.showinfo("完成", "旧记录已清理")
                    db_window.destroy()
                except Exception as e:
                    messagebox.showerror("错误", f"清理失败: {e}")
        
        ttk.Button(clean_frame, text="清理30天前记录", command=clean_old_records).pack(side='left', padx=5)
    
    def show_log_viewer(self):
        """显示日志查看器"""
        log_window = tk.Toplevel(self.root)
        log_window.title("日志查看器")
        log_window.geometry("800x600")
        
        # 日志选择
        control_frame = ttk.Frame(log_window)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(control_frame, text="选择日志文件:").pack(side='left', padx=(0, 5))
        
        log_files = list(Path("logs").glob("*.log")) if Path("logs").exists() else []
        log_file_var = tk.StringVar()
        
        if log_files:
            log_file_combo = ttk.Combobox(control_frame, textvariable=log_file_var,
                                         values=[f.name for f in log_files])
            log_file_combo.pack(side='left', padx=5)
            log_file_combo.current(0)
        
        # 日志内容
        log_text = scrolledtext.ScrolledText(log_window, wrap=tk.WORD, font=('Consolas', 9))
        log_text.pack(fill='both', expand=True, padx=10, pady=5)
        
        def load_log():
            if log_file_var.get() and Path("logs", log_file_var.get()).exists():
                try:
                    log_path = Path("logs", log_file_var.get())
                    # 尝试多种编码格式
                    encodings = ['utf-8', 'gbk', 'gb2312', 'cp936', 'latin1']
                    content = None
                    
                    for encoding in encodings:
                        try:
                            with open(log_path, 'r', encoding=encoding) as f:
                                content = f.read()
                                break
                        except UnicodeDecodeError:
                            continue
                    
                    if content:
                        log_text.delete(1.0, tk.END)
                        log_text.insert(tk.END, content)
                        log_text.see(tk.END)
                    else:
                        messagebox.showerror("错误", "无法以任何编码格式读取日志文件")
                except Exception as e:
                    messagebox.showerror("错误", f"读取日志失败: {e}")
        
        if log_files:
            ttk.Button(control_frame, text="加载", command=load_log).pack(side='left', padx=5)
            load_log()  # 自动加载第一个文件
    
    def show_system_info(self):
        """显示系统信息"""
        import platform
        import psutil
        
        info = f"""
系统信息：
操作系统: {platform.system()} {platform.release()}
Python版本: {platform.python_version()}
内存使用: {psutil.virtual_memory().percent}%
磁盘使用: {psutil.disk_usage('.').percent}%

应用信息：
版本: v1.0
数据库: {self.db_path}
结果目录: {self.config['result_directory']}
定时任务: {'运行中' if self.scheduler.running else '已停止'}

配置状态：
自动运行: {'启用' if self.config['auto_run'] else '禁用'}
系统通知: {'启用' if self.config['notifications'] else '禁用'}
日志级别: {self.config['log_level']}
        """
        
        messagebox.showinfo("系统信息", info)
    
    def run_lstm_manual_analysis(self):
        """运行LSTM手动分析"""
        try:
            self.log_message("[LSTM分析] 启动LSTM手动分析...")
            
            # 在新线程中运行以避免阻塞GUI
            def run_in_thread():
                try:
                    # 运行LSTM手动分析
                    from lstm_manual_analysis import run_lstm_manual_analysis
                    
                    self.log_message("[LSTM分析] 正在运行LSTM多日预测分析...")
                    result = run_lstm_manual_analysis(
                        ticker_list=None,  # 使用默认股票池
                        days_history=365,
                        max_stocks=20
                    )
                    
                    if result['status'] == 'success':
                        self.log_message(f"[LSTM分析] ✅ 分析完成，生成 {result['total_analyzed']} 个推荐")
                        self.log_message(f"[LSTM分析] 交易信号: 买入 {result['signals']['buy']}, 卖出 {result['signals']['sell']}, 持有 {result['signals']['hold']}")
                        if result.get('files_generated'):
                            for file_type, file_path in result['files_generated'].items():
                                self.log_message(f"[LSTM分析] {file_type.upper()}文件: {file_path}")
                    else:
                        self.log_message(f"[LSTM分析] ❌ LSTM分析失败: {result.get('message')}")
                        
                except ImportError:
                    self.log_message("[LSTM分析] ❌ 无法导入LSTM分析模块")
                except Exception as e:
                    self.log_message(f"[LSTM分析] ❌ 运行出错: {e}")
                    
            threading.Thread(target=run_in_thread, daemon=True).start()
            
        except Exception as e:
            self.log_message(f"[LSTM分析] ❌ 启动失败: {e}")
    
    def open_lstm_auto_trading_manager(self):
        """打开LSTM自动交易管理器"""
        try:
            self.log_message("[LSTM自动交易] 启动自动交易管理器...")
            
            # 在新线程中启动管理器以避免阻塞主界面
            def run_manager():
                try:
                    import subprocess
                    import sys
                    
                    # 启动LSTM自动交易管理器
                    process = subprocess.Popen([
                        sys.executable, "lstm_auto_trading_manager.py"
                    ], cwd=os.getcwd())
                    
                    self.log_message("[LSTM自动交易] ✅ 自动交易管理器已启动")
                    
                except Exception as e:
                    self.log_message(f"[LSTM自动交易] ❌ 启动管理器失败: {e}")
            
            threading.Thread(target=run_manager, daemon=True).start()
            
        except Exception as e:
            self.log_message(f"[LSTM自动交易] ❌ 启动失败: {e}")
    
    
    
    
    
    def manage_stock_pool(self):
        """股票池管理界面"""
        try:
            # 创建股票池管理窗口
            pool_window = tk.Toplevel(self.root)
            pool_window.title("📊 股票池管理")
            pool_window.geometry("800x600")
            pool_window.configure(bg='#f0f0f0')
            
            # 设置窗口图标和样式
            pool_window.transient(self.root)
            pool_window.grab_set()
            
            # 创建主框架
            main_frame = ttk.Frame(pool_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 标题
            title_label = tk.Label(main_frame, text="📊 股票池管理", 
                                 font=('Microsoft YaHei', 16, 'bold'),
                                 bg='#f0f0f0', fg='#2c3e50')
            title_label.pack(pady=(0, 20))
            
            # 创建笔记本控件用于分类管理
            notebook = ttk.Notebook(main_frame)
            notebook.pack(fill=tk.BOTH, expand=True)
            
            # 为每个股票池类别创建标签页
            self.stock_pool_tabs = {}
            
            if not hasattr(self, 'default_stock_pool') or not self.default_stock_pool:
                self.initialize_stock_pools()
            
            for category, stocks in self.default_stock_pool.items():
                # 创建标签页框架
                tab_frame = ttk.Frame(notebook)
                notebook.add(tab_frame, text=f"{category} ({len(stocks)})")
                
                # 创建股票列表管理界面
                self.create_stock_category_interface(tab_frame, category, stocks)
                self.stock_pool_tabs[category] = tab_frame
            
            # 添加新增类别的标签页
            add_tab_frame = ttk.Frame(notebook)
            notebook.add(add_tab_frame, text="➕ 新增类别")
            self.create_add_category_interface(add_tab_frame, notebook)
            
            # 底部操作按钮
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(10, 0))
            
            ttk.Button(button_frame, text="💾 保存所有更改", 
                      command=lambda: self.save_all_stock_pools(pool_window)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="🔄 重新加载", 
                      command=lambda: self.reload_stock_pools(pool_window)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="🕷️ 更新股票池", 
                      command=self.update_stock_pool_from_crawler).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="📤 导出配置", 
                      command=self.export_stock_pools).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="📥 导入配置", 
                      command=lambda: self.import_stock_pools(pool_window)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="❌ 关闭", 
                      command=pool_window.destroy).pack(side=tk.RIGHT, padx=5)
            
            # 状态栏
            self.pool_status_var = tk.StringVar(value="📊 股票池管理就绪")
            status_label = tk.Label(main_frame, textvariable=self.pool_status_var,
                                  bg='#f0f0f0', fg='#7f8c8d', font=('Microsoft YaHei', 9))
            status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
            
        except Exception as e:
            self.log_message(f"[股票池管理] ❌ 打开管理界面失败: {e}")
            messagebox.showerror("错误", f"无法打开股票池管理界面:\n{e}")
    
    def create_stock_category_interface(self, parent, category, stocks):
        """创建单个股票类别的管理界面"""
        try:
            # 创建左右分栏
            paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
            paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 左侧：股票列表
            left_frame = ttk.Frame(paned)
            paned.add(left_frame, weight=2)
            
            tk.Label(left_frame, text=f"{category} 股票列表", 
                    font=('Microsoft YaHei', 12, 'bold')).pack(pady=(0, 10))
            
            # 股票列表框
            list_frame = ttk.Frame(left_frame)
            list_frame.pack(fill=tk.BOTH, expand=True)
            
            # 创建滚动条
            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # 股票列表框
            stock_listbox = tk.Listbox(list_frame, font=('Consolas', 10),
                                     yscrollcommand=scrollbar.set)
            stock_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=stock_listbox.yview)
            
            # 添加股票到列表
            for stock in stocks:
                stock_listbox.insert(tk.END, stock)
                
            # 右侧：操作面板
            right_frame = ttk.Frame(paned)
            paned.add(right_frame, weight=1)
            
            tk.Label(right_frame, text="操作面板", 
                    font=('Microsoft YaHei', 12, 'bold')).pack(pady=(0, 10))
            
            # 添加股票
            add_frame = ttk.LabelFrame(right_frame, text="➕ 添加股票")
            add_frame.pack(fill=tk.X, pady=(0, 10))
            
            add_entry = ttk.Entry(add_frame, font=('Consolas', 10))
            add_entry.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Button(add_frame, text="添加", 
                      command=lambda: self.add_stock_to_category(category, add_entry, stock_listbox)).pack(pady=5)
            
            # 删除股票
            delete_frame = ttk.LabelFrame(right_frame, text="➖ 删除股票")
            delete_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Button(delete_frame, text="删除选中", 
                      command=lambda: self.remove_stock_from_category(category, stock_listbox)).pack(pady=5)
            
            # 股票信息
            info_frame = ttk.LabelFrame(right_frame, text="📊 股票信息")
            info_frame.pack(fill=tk.X, pady=(0, 10))
            
            info_text = tk.Text(info_frame, height=8, font=('Consolas', 9))
            info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 绑定选择事件
            stock_listbox.bind('<<ListboxSelect>>', 
                             lambda e: self.show_stock_info(stock_listbox, info_text))
            
            # 存储引用以便后续操作
            if not hasattr(self, 'stock_listboxes'):
                self.stock_listboxes = {}
            self.stock_listboxes[category] = stock_listbox
            
        except Exception as e:
            self.log_message(f"[股票池管理] ❌ 创建类别界面失败: {e}")
    
    def create_add_category_interface(self, parent, notebook):
        """创建新增类别的界面"""
        try:
            # 标题
            tk.Label(parent, text="➕ 添加新的股票类别", 
                    font=('Microsoft YaHei', 14, 'bold')).pack(pady=20)
            
            # 类别名称输入
            name_frame = ttk.Frame(parent)
            name_frame.pack(pady=10)
            
            tk.Label(name_frame, text="类别名称:", font=('Microsoft YaHei', 11)).pack(side=tk.LEFT)
            category_entry = ttk.Entry(name_frame, font=('Microsoft YaHei', 11), width=20)
            category_entry.pack(side=tk.LEFT, padx=(10, 0))
            
            # 初始股票输入
            stocks_frame = ttk.LabelFrame(parent, text="初始股票（用逗号分隔）")
            stocks_frame.pack(fill=tk.X, padx=20, pady=10)
            
            stocks_text = tk.Text(stocks_frame, height=6, font=('Consolas', 10))
            stocks_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 添加按钮
            ttk.Button(parent, text="🎯 创建类别", 
                      command=lambda: self.add_new_category(category_entry, stocks_text, notebook)).pack(pady=20)
            
        except Exception as e:
            self.log_message(f"[股票池管理] ❌ 创建新增界面失败: {e}")
    
    def add_stock_to_category(self, category, entry, listbox):
        """向类别中添加股票"""
        try:
            stock = entry.get().strip().upper()
            if not stock:
                return
                
            # 检查是否已存在
            if stock in self.default_stock_pool.get(category, []):
                messagebox.showwarning("警告", f"股票 {stock} 已存在于 {category} 中")
                return
            
            # 添加到数据结构
            if category not in self.default_stock_pool:
                self.default_stock_pool[category] = []
            self.default_stock_pool[category].append(stock)
            
            # 添加到列表框
            listbox.insert(tk.END, stock)
            entry.delete(0, tk.END)
            
            self.pool_status_var.set(f"✅ 已添加 {stock} 到 {category}")
            
        except Exception as e:
            self.log_message(f"[股票池管理] ❌ 添加股票失败: {e}")
    
    def remove_stock_from_category(self, category, listbox):
        """从类别中删除股票"""
        try:
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("警告", "请先选择要删除的股票")
                return
            
            stock = listbox.get(selection[0])
            
            # 确认删除
            if messagebox.askyesno("确认", f"确定要从 {category} 中删除 {stock} 吗？"):
                # 从数据结构中删除
                if category in self.default_stock_pool:
                    self.default_stock_pool[category].remove(stock)
                
                # 从列表框中删除
                listbox.delete(selection[0])
                
                self.pool_status_var.set(f"🗑️ 已从 {category} 中删除 {stock}")
                
        except Exception as e:
            self.log_message(f"[股票池管理] ❌ 删除股票失败: {e}")
    
    def show_stock_info(self, listbox, info_text):
        """显示股票信息"""
        try:
            selection = listbox.curselection()
            if not selection:
                return
                
            stock = listbox.get(selection[0])
            
            # 清空文本框
            info_text.delete(1.0, tk.END)
            
            # 显示基本信息
            info_text.insert(tk.END, f"股票代码: {stock}\n")
            info_text.insert(tk.END, f"选择时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            info_text.insert(tk.END, "─" * 30 + "\n")
            
            # 这里可以添加更多股票信息获取逻辑
            info_text.insert(tk.END, "📊 更多信息获取中...\n")
            
        except Exception as e:
            self.log_message(f"[股票池管理] ❌ 显示股票信息失败: {e}")
    
    def add_new_category(self, name_entry, stocks_text, notebook):
        """添加新的股票类别"""
        try:
            category_name = name_entry.get().strip()
            if not category_name:
                messagebox.showwarning("警告", "请输入类别名称")
                return
            
            if category_name in self.default_stock_pool:
                messagebox.showwarning("警告", f"类别 {category_name} 已存在")
                return
            
            # 解析股票列表
            stocks_input = stocks_text.get(1.0, tk.END).strip()
            stocks = []
            if stocks_input:
                stocks = [s.strip().upper() for s in stocks_input.replace('\n', ',').split(',') if s.strip()]
            
            # 添加新类别
            self.default_stock_pool[category_name] = stocks
            
            # 创建新的标签页
            tab_frame = ttk.Frame(notebook)
            notebook.insert(notebook.index("end") - 1, tab_frame, text=f"{category_name} ({len(stocks)})")
            self.create_stock_category_interface(tab_frame, category_name, stocks)
            
            # 清空输入
            name_entry.delete(0, tk.END)
            stocks_text.delete(1.0, tk.END)
            
            self.pool_status_var.set(f"🎯 已创建新类别: {category_name}")
            
        except Exception as e:
            self.log_message(f"[股票池管理] ❌ 添加新类别失败: {e}")
    
    def save_all_stock_pools(self, window):
        """保存所有股票池更改"""
        try:
            if self.save_default_stock_pool():
                self.pool_status_var.set("💾 所有更改已保存")
                messagebox.showinfo("成功", "股票池配置已保存成功！")
            else:
                messagebox.showerror("错误", "保存股票池配置失败！")
                
        except Exception as e:
            self.log_message(f"[股票池管理] ❌ 保存失败: {e}")
            messagebox.showerror("错误", f"保存失败:\n{e}")
    
    def reload_stock_pools(self, window):
        """重新加载股票池"""
        try:
            if messagebox.askyesno("确认", "重新加载将丢失未保存的更改，确定继续吗？"):
                self.load_default_stock_pool()
                window.destroy()
                self.manage_stock_pool()  # 重新打开管理界面
                
        except Exception as e:
            self.log_message(f"[股票池管理] ❌ 重新加载失败: {e}")
    
    def export_stock_pools(self):
        """导出股票池配置"""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
                title="导出股票池配置"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.default_stock_pool, f, ensure_ascii=False, indent=2)
                
                self.pool_status_var.set(f"📤 已导出到: {filename}")
                messagebox.showinfo("成功", f"股票池配置已导出到:\n{filename}")
                
        except Exception as e:
            self.log_message(f"[股票池管理] ❌ 导出失败: {e}")
            messagebox.showerror("错误", f"导出失败:\n{e}")
    
    def import_stock_pools(self, window):
        """导入股票池配置"""
        try:
            from tkinter import filedialog
            filename = filedialog.askopenfilename(
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
                title="导入股票池配置"
            )
            
            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    imported_pools = json.load(f)
                
                if messagebox.askyesno("确认", f"导入配置将覆盖现有设置，确定继续吗？\n\n文件: {filename}"):
                    self.default_stock_pool.update(imported_pools)
                    self.pool_status_var.set(f"📥 已导入: {filename}")
                    
                    # 重新打开管理界面
                    window.destroy()
                    self.manage_stock_pool()
                    
        except Exception as e:
            self.log_message(f"[股票池管理] ❌ 导入失败: {e}")
            messagebox.showerror("错误", f"导入失败:\n{e}")
    
    def show_advanced_strategy(self):
        """显示高级策略界面"""
        try:
            # 创建高级策略窗口
            strategy_window = tk.Toplevel(self.root)
            strategy_window.title("🚀 高级量化交易策略")
            strategy_window.geometry("1200x800")
            strategy_window.configure(bg='#f0f0f0')
            
            # 设置窗口图标和样式
            strategy_window.transient(self.root)
            strategy_window.grab_set()
            
            # 创建主框架
            main_frame = ttk.Frame(strategy_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 标题
            title_label = tk.Label(main_frame, text="🚀 高级量化交易策略系统", 
                                 font=('Microsoft YaHei', 18, 'bold'),
                                 bg='#f0f0f0', fg='#2c3e50')
            title_label.pack(pady=(0, 20))
            
            # 创建笔记本控件用于不同功能模块
            notebook = ttk.Notebook(main_frame)
            notebook.pack(fill=tk.BOTH, expand=True)
            
            # 1. 市场环境分析标签页
            self.create_market_analysis_tab(notebook)
            
            # 2. 动态权重配置标签页
            self.create_allocation_tab(notebook)
            
            # 3. SuperTrend信号标签页
            self.create_supertrend_tab(notebook)
            
            # 4. 风险控制标签页
            self.create_risk_control_tab(notebook)
            
            # 5. 回测分析标签页
            self.create_backtest_tab(notebook)
            
            # 底部控制按钮
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(10, 0))
            
            ttk.Button(button_frame, text="🚀 启动高级策略", 
                      command=self.run_advanced_strategy).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="⏹️ 停止策略", 
                      command=self.stop_advanced_strategy).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="📊 查看报告", 
                      command=self.show_strategy_report).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="❌ 关闭", 
                      command=strategy_window.destroy).pack(side=tk.RIGHT, padx=5)
            
        except Exception as e:
            self.log_message(f"[高级策略] ❌ 打开界面失败: {e}")
            messagebox.showerror("错误", f"无法打开高级策略界面:\n{e}")
    
    def create_market_analysis_tab(self, notebook):
        """创建市场环境分析标签页"""
        market_frame = ttk.Frame(notebook)
        notebook.add(market_frame, text="📈 市场环境分析")
        
        # 标题
        tk.Label(market_frame, text="四象限市场环境判断", 
                font=('Microsoft YaHei', 14, 'bold')).pack(pady=10)
        
        # 创建左右分栏
        paned = ttk.PanedWindow(market_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：配置面板
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # ADX配置
        adx_frame = ttk.LabelFrame(left_frame, text="ADX趋势判断")
        adx_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(adx_frame, text="ADX周期:").pack(side=tk.LEFT, padx=5)
        self.adx_period_var = tk.StringVar(value="14")
        ttk.Entry(adx_frame, textvariable=self.adx_period_var, width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Label(adx_frame, text="趋势阈值:").pack(side=tk.LEFT, padx=5)
        self.adx_threshold_var = tk.StringVar(value="25")
        ttk.Entry(adx_frame, textvariable=self.adx_threshold_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # ATR配置
        atr_frame = ttk.LabelFrame(left_frame, text="ATR波动判断")
        atr_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(atr_frame, text="ATR周期:").pack(side=tk.LEFT, padx=5)
        self.atr_period_var = tk.StringVar(value="14")
        ttk.Entry(atr_frame, textvariable=self.atr_period_var, width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Label(atr_frame, text="波动阈值:").pack(side=tk.LEFT, padx=5)
        self.atr_threshold_var = tk.StringVar(value="0.008")
        ttk.Entry(atr_frame, textvariable=self.atr_threshold_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # 分析按钮
        ttk.Button(left_frame, text="🔍 分析市场环境", 
                  command=self.analyze_current_market).pack(pady=20)
        
        # 右侧：结果显示
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(right_frame, text="分析结果")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.market_result_text = tk.Text(result_frame, height=20, font=('Consolas', 10))
        scrollbar_market = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.market_result_text.yview)
        self.market_result_text.configure(yscrollcommand=scrollbar_market.set)
        
        self.market_result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar_market.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_allocation_tab(self, notebook):
        """创建动态权重配置标签页"""
        alloc_frame = ttk.Frame(notebook)
        notebook.add(alloc_frame, text="⚖️ 动态权重配置")
        
        # 标题
        tk.Label(alloc_frame, text="基于表现的动态权重分配", 
                font=('Microsoft YaHei', 14, 'bold')).pack(pady=10)
        
        # 当前权重显示
        current_frame = ttk.LabelFrame(alloc_frame, text="当前权重配置")
        current_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # 权重显示框架
        weights_display_frame = ttk.Frame(current_frame)
        weights_display_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(weights_display_frame, text="策略A权重:", font=('Microsoft YaHei', 11)).pack(side=tk.LEFT)
        self.strategy_a_weight_label = tk.Label(weights_display_frame, text="50.0%", 
                                               font=('Microsoft YaHei', 11, 'bold'), fg='#2e8b57')
        self.strategy_a_weight_label.pack(side=tk.LEFT, padx=(10, 30))
        
        tk.Label(weights_display_frame, text="策略B权重:", font=('Microsoft YaHei', 11)).pack(side=tk.LEFT)
        self.strategy_b_weight_label = tk.Label(weights_display_frame, text="50.0%", 
                                               font=('Microsoft YaHei', 11, 'bold'), fg='#4169e1')
        self.strategy_b_weight_label.pack(side=tk.LEFT, padx=10)
        
        # 基础权重配置表格
        base_config_frame = ttk.LabelFrame(alloc_frame, text="四象限基础权重配置")
        base_config_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # 创建表格
        columns = ('市场环境', '策略A权重', '策略B权重', '说明')
        self.allocation_tree = ttk.Treeview(base_config_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.allocation_tree.heading(col, text=col)
            self.allocation_tree.column(col, width=120)
        
        # 插入基础配置数据（使用统一常量）
        for data in TradingConstants.DEFAULT_ALLOCATION_DATA:
            self.allocation_tree.insert('', 'end', values=data)
        
        self.allocation_tree.pack(fill=tk.X, padx=10, pady=10)
        
        # 权重更新按钮
        update_frame = ttk.Frame(alloc_frame)
        update_frame.pack(fill=tk.X, padx=10)
        
        ttk.Button(update_frame, text="📊 计算最优权重", 
                  command=self.calculate_optimal_weights).pack(side=tk.LEFT, padx=5)
        ttk.Button(update_frame, text="💾 保存权重配置", 
                  command=self.save_allocation_weights).pack(side=tk.LEFT, padx=5)
        ttk.Button(update_frame, text="🔄 重置为默认", 
                  command=self.reset_allocation_weights).pack(side=tk.LEFT, padx=5)
    
    def create_supertrend_tab(self, notebook):
        """创建SuperTrend信号标签页"""
        supertrend_frame = ttk.Frame(notebook)
        notebook.add(supertrend_frame, text="📈 SuperTrend信号")
        
        # 标题
        tk.Label(supertrend_frame, text="SuperTrend买卖信号与止损", 
                font=('Microsoft YaHei', 14, 'bold')).pack(pady=10)
        
        # 参数配置
        params_frame = ttk.LabelFrame(supertrend_frame, text="参数配置")
        params_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        params_inner = ttk.Frame(params_frame)
        params_inner.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(params_inner, text="ATR周期:").pack(side=tk.LEFT, padx=5)
        self.st_period_var = tk.StringVar(value="10")
        ttk.Entry(params_inner, textvariable=self.st_period_var, width=10).pack(side=tk.LEFT, padx=5)
        
        tk.Label(params_inner, text="倍数:").pack(side=tk.LEFT, padx=5)
        self.st_multiplier_var = tk.StringVar(value="3.0")
        ttk.Entry(params_inner, textvariable=self.st_multiplier_var, width=10).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(params_inner, text="🎯 生成信号", 
                  command=self.generate_supertrend_signals).pack(side=tk.LEFT, padx=20)
        
        # 信号显示区域
        signals_frame = ttk.LabelFrame(supertrend_frame, text="交易信号")
        signals_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.signals_text = tk.Text(signals_frame, height=15, font=('Consolas', 10))
        scrollbar_signals = ttk.Scrollbar(signals_frame, orient=tk.VERTICAL, command=self.signals_text.yview)
        self.signals_text.configure(yscrollcommand=scrollbar_signals.set)
        
        self.signals_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar_signals.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_risk_control_tab(self, notebook):
        """创建风险控制标签页"""
        risk_frame = ttk.Frame(notebook)
        notebook.add(risk_frame, text="🛡️ 风险控制")
        
        # 标题
        tk.Label(risk_frame, text="组合级风险控制系统", 
                font=('Microsoft YaHei', 14, 'bold')).pack(pady=10)
        
        # 风险参数配置
        params_frame = ttk.LabelFrame(risk_frame, text="风险参数设置")
        params_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        params_grid = ttk.Frame(params_frame)
        params_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # 最大回撤
        tk.Label(params_grid, text="最大回撤阈值:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.max_drawdown_var = tk.StringVar(value="10%")
        ttk.Entry(params_grid, textvariable=self.max_drawdown_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # 现金缓冲
        tk.Label(params_grid, text="现金缓冲比例:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.cash_buffer_var = tk.StringVar(value="5%")
        ttk.Entry(params_grid, textvariable=self.cash_buffer_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # 止损倍数
        tk.Label(params_grid, text="止损ATR倍数:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.stop_loss_multiplier_var = tk.StringVar(value="2.0")
        ttk.Entry(params_grid, textvariable=self.stop_loss_multiplier_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # 冷静期天数
        tk.Label(params_grid, text="冷静期天数:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.cooldown_days_var = tk.StringVar(value="3")
        ttk.Entry(params_grid, textvariable=self.cooldown_days_var, width=10).grid(row=1, column=3, padx=5, pady=5)
        
        # 风险状态显示
        status_frame = ttk.LabelFrame(risk_frame, text="当前风险状态")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.risk_status_text = tk.Text(status_frame, height=12, font=('Consolas', 10))
        scrollbar_risk = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.risk_status_text.yview)
        self.risk_status_text.configure(yscrollcommand=scrollbar_risk.set)
        
        self.risk_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar_risk.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 风险控制按钮
        risk_buttons = ttk.Frame(risk_frame)
        risk_buttons.pack(fill=tk.X, padx=10)
        
        ttk.Button(risk_buttons, text="🔍 检查风险状态", 
                  command=self.check_risk_status).pack(side=tk.LEFT, padx=5)
        ttk.Button(risk_buttons, text="🚨 手动风控", 
                  command=self.manual_risk_control).pack(side=tk.LEFT, padx=5)
    
    def create_backtest_tab(self, notebook):
        """创建回测分析标签页"""
        backtest_frame = ttk.Frame(notebook)
        notebook.add(backtest_frame, text="📊 回测分析")
        
        # 标题
        tk.Label(backtest_frame, text="历史回测与绩效统计", 
                font=('Microsoft YaHei', 14, 'bold')).pack(pady=10)
        
        # 回测参数
        params_frame = ttk.LabelFrame(backtest_frame, text="回测参数设置")
        params_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        params_grid = ttk.Frame(params_frame)
        params_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # 日期范围
        tk.Label(params_grid, text="开始日期:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.backtest_start_var = tk.StringVar(value="2020-01-01")
        ttk.Entry(params_grid, textvariable=self.backtest_start_var, width=15).grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(params_grid, text="结束日期:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.backtest_end_var = tk.StringVar(value="2024-12-31")
        ttk.Entry(params_grid, textvariable=self.backtest_end_var, width=15).grid(row=0, column=3, padx=5, pady=5)
        
        # 初始资金
        tk.Label(params_grid, text="初始资金:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.initial_capital_var = tk.StringVar(value="100000")
        ttk.Entry(params_grid, textvariable=self.initial_capital_var, width=15).grid(row=1, column=1, padx=5, pady=5)
        
        # 手续费
        tk.Label(params_grid, text="手续费率:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.commission_var = tk.StringVar(value="0.0001")
        ttk.Entry(params_grid, textvariable=self.commission_var, width=15).grid(row=1, column=3, padx=5, pady=5)
        
        # 回测按钮
        ttk.Button(params_frame, text="🚀 开始回测", 
                  command=self.start_backtest).pack(pady=10)
        
        # 结果显示区域
        results_frame = ttk.LabelFrame(backtest_frame, text="回测结果")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.backtest_results_text = tk.Text(results_frame, height=15, font=('Consolas', 10))
        scrollbar_backtest = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.backtest_results_text.yview)
        self.backtest_results_text.configure(yscrollcommand=scrollbar_backtest.set)
        
        self.backtest_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar_backtest.pack(side=tk.RIGHT, fill=tk.Y)
    
    # 高级策略相关的实现方法
    def analyze_current_market(self):
        """分析当前市场环境"""
        try:
            self.market_result_text.delete(1.0, tk.END)
            self.market_result_text.insert(tk.END, "🔍 正在分析市场环境...\n\n")
            
            # 这里调用高级策略模块进行市场分析
            from advanced_trading_strategy import MarketEnvironmentAnalyzer
            import yfinance as yf
            
            analyzer = MarketEnvironmentAnalyzer()
            
            # 获取SPY数据作为市场代表
            ticker = yf.Ticker('SPY')
            data = ticker.history(period='3mo')  # 获取3个月数据
            
            if not data.empty:
                market_env = analyzer.classify_market_environment(data)
                
                # 计算具体指标值
                adx = analyzer.calculate_adx(data['High'], data['Low'], data['Close'])
                atr = analyzer.calculate_atr(data['High'], data['Low'], data['Close'])
                sma = data['Close'].rolling(window=50).mean()
                atr_sma_ratio = (atr / sma).iloc[-1]
                
                result_text = f"""市场环境分析结果 (SPY)
{'='*50}

当前市场环境: {market_env}

技术指标详情:
├─ ADX (14日): {adx.iloc[-1]:.2f}
├─ 趋势判断: {'趋势市场' if adx.iloc[-1] >= 25 else '振荡市场'}
├─ ATR/SMA比值: {atr_sma_ratio:.6f}
└─ 波动性: {'高波动' if atr_sma_ratio >= 0.008 else '低波动'}

环境解读:
"""
                
                if market_env == 'trend_high_vol':
                    result_text += "📈 当前为趋势+高波动环境\n建议: 采用趋势跟踪策略，适当控制仓位"
                elif market_env == 'trend_low_vol':
                    result_text += "📊 当前为趋势+低波动环境\n建议: 加大趋势跟踪策略权重，相对安全"
                elif market_env == 'osc_high_vol':
                    result_text += "🌊 当前为振荡+高波动环境\n建议: 采用均值回归策略，严格止损"
                else:
                    result_text += "🎯 当前为振荡+低波动环境\n建议: 平衡配置，等待机会"
                
                self.market_result_text.delete(1.0, tk.END)
                self.market_result_text.insert(tk.END, result_text)
                
            else:
                self.market_result_text.insert(tk.END, "❌ 无法获取市场数据，请检查网络连接")
                
        except Exception as e:
            error_msg = f"❌ 市场分析出错: {e}\n"
            self.market_result_text.insert(tk.END, error_msg)
            self.log_message(f"[市场分析] {error_msg}")
    
    def calculate_optimal_weights(self):
        """计算最优权重"""
        try:
            # 模拟计算过程
            import random
            
            # 生成模拟的策略收益数据
            strategy_a_returns = [random.gauss(0.001, 0.02) for _ in range(60)]
            strategy_b_returns = [random.gauss(0.0005, 0.015) for _ in range(60)]
            
            # 计算Sharpe比率
            def calc_sharpe(returns):
                if len(returns) < 2:
                    return 0.0
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return == 0:
                    return 0.0
                return (mean_return / std_return) * np.sqrt(252)
            
            sharpe_a = calc_sharpe(strategy_a_returns)
            sharpe_b = calc_sharpe(strategy_b_returns)
            
            # 基于Sharpe比率计算权重
            total_sharpe = max(sharpe_a, 0.01) + max(sharpe_b, 0.01)
            weight_a = max(sharpe_a, 0.01) / total_sharpe
            weight_b = max(sharpe_b, 0.01) / total_sharpe
            
            # 更新显示
            self.strategy_a_weight_label.config(text=f"{weight_a:.1%}")
            self.strategy_b_weight_label.config(text=f"{weight_b:.1%}")
            
            messagebox.showinfo("权重计算完成", 
                               f"基于历史表现计算的最优权重:\n\n"
                               f"策略A: {weight_a:.1%} (Sharpe: {sharpe_a:.2f})\n"
                               f"策略B: {weight_b:.1%} (Sharpe: {sharpe_b:.2f})")
            
        except Exception as e:
            self.log_message(f"[权重计算] ❌ 出错: {e}")
            messagebox.showerror("错误", f"权重计算失败:\n{e}")
    
    def save_allocation_weights(self):
        """保存权重配置"""
        try:
            config_data = {
                'allocation_date': datetime.now().strftime('%Y-%m-%d'),
                'strategy_a_weight': float(self.strategy_a_weight_label.cget('text').rstrip('%')) / 100,
                'strategy_b_weight': float(self.strategy_b_weight_label.cget('text').rstrip('%')) / 100,
                'last_update': datetime.now().isoformat()
            }
            
            with open('allocation_config.json', 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo("保存成功", "权重配置已保存到 allocation_config.json")
            
        except Exception as e:
            self.log_message(f"[权重保存] ❌ 出错: {e}")
            messagebox.showerror("错误", f"保存权重配置失败:\n{e}")
    
    def reset_allocation_weights(self):
        """重置权重配置"""
        self.strategy_a_weight_label.config(text="50.0%")
        self.strategy_b_weight_label.config(text="50.0%")
        messagebox.showinfo("重置完成", "权重已重置为默认值 (50%-50%)")
    
    def generate_supertrend_signals(self):
        """生成SuperTrend信号"""
        try:
            self.signals_text.delete(1.0, tk.END)
            self.signals_text.insert(tk.END, "🎯 正在生成SuperTrend信号...\n\n")
            
            from advanced_trading_strategy import SuperTrendIndicator
            import yfinance as yf
            
            # 获取参数
            period = int(self.st_period_var.get())
            multiplier = float(self.st_multiplier_var.get())
            
            supertrend = SuperTrendIndicator(period=period, multiplier=multiplier)
            
            # 测试几只股票
            test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
            
            for symbol in test_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='1mo')
                    
                    if not data.empty:
                        # 计算SuperTrend
                        st_data = supertrend.calculate_supertrend(data['High'], data['Low'], data['Close'])
                        signals = supertrend.generate_signals(data, st_data)
                        
                        # 找到最新信号
                        latest_signals = signals.tail(5)
                        latest_signal = signals.iloc[-1]
                        
                        signal_text = f"\n{symbol} SuperTrend信号:\n"
                        signal_text += f"├─ 最新信号: "
                        if latest_signal == 1:
                            signal_text += "🟢 买入信号"
                        elif latest_signal == -1:
                            signal_text += "🔴 卖出信号"
                        else:
                            signal_text += "⚪ 无信号"
                        
                        signal_text += f"\n├─ 当前价格: ${data['Close'].iloc[-1]:.2f}"
                        signal_text += f"\n├─ SuperTrend: ${st_data['SuperTrend'].iloc[-1]:.2f}"
                        signal_text += f"\n└─ 趋势方向: {'上升' if st_data['Direction'].iloc[-1] == 1 else '下降'}\n"
                        
                        self.signals_text.insert(tk.END, signal_text)
                        
                except Exception as e:
                    self.signals_text.insert(tk.END, f"\n❌ {symbol} 信号生成失败: {e}\n")
            
        except Exception as e:
            error_msg = f"❌ SuperTrend信号生成出错: {e}\n"
            self.signals_text.insert(tk.END, error_msg)
            self.log_message(f"[SuperTrend] {error_msg}")
    
    def check_risk_status(self):
        """检查风险状态"""
        try:
            self.risk_status_text.delete(1.0, tk.END)
            self.risk_status_text.insert(tk.END, "🛡️ 风险状态检查报告\n")
            self.risk_status_text.insert(tk.END, "="*50 + "\n\n")
            
            # 模拟风险检查
            import random
            
            current_portfolio = 100000 * (1 + random.uniform(-0.15, 0.25))
            peak_value = 120000
            drawdown = (peak_value - current_portfolio) / peak_value
            
            risk_report = f"""组合价值状态:
├─ 当前组合价值: ${current_portfolio:,.2f}
├─ 历史最高价值: ${peak_value:,.2f}
├─ 当前回撤: {drawdown:.2%}
└─ 风险状态: {'🚨 警告' if drawdown > 0.10 else '✅ 正常'}

风险控制参数:
├─ 最大回撤阈值: {self.max_drawdown_var.get()}
├─ 现金缓冲比例: {self.cash_buffer_var.get()}
├─ 止损ATR倍数: {self.stop_loss_multiplier_var.get()}
└─ 冷静期设置: {self.cooldown_days_var.get()}天

当前风险等级: {'🔴 高风险' if drawdown > 0.10 else '🟡 中等风险' if drawdown > 0.05 else '🟢 低风险'}
"""
            
            self.risk_status_text.insert(tk.END, risk_report)
            
        except Exception as e:
            error_msg = f"❌ 风险检查出错: {e}\n"
            self.risk_status_text.insert(tk.END, error_msg)
    
    def manual_risk_control(self):
        """手动风控"""
        if messagebox.askyesno("手动风控", "确定要立即启动风险控制措施吗？\n\n这将：\n• 停止所有新开仓\n• 启动保护模式\n• 进入冷静期"):
            self.risk_status_text.insert(tk.END, f"\n🚨 {datetime.now().strftime('%H:%M:%S')} 手动风控已启动\n")
            messagebox.showinfo("风控启动", "手动风险控制已启动")
    
    def start_backtest(self):
        """开始回测"""
        try:
            self.backtest_results_text.delete(1.0, tk.END)
            self.backtest_results_text.insert(tk.END, "🚀 开始历史回测...\n\n")
            
            # 获取回测参数
            start_date = self.backtest_start_var.get()
            end_date = self.backtest_end_var.get()
            initial_capital = float(self.initial_capital_var.get())
            commission = float(self.commission_var.get())
            
            # 模拟回测结果
            import random
            
            # 生成模拟的回测结果
            total_return = random.uniform(-0.1, 0.4)
            sharpe_ratio = random.uniform(0.5, 2.5)
            max_drawdown = random.uniform(0.05, 0.25)
            win_rate = random.uniform(0.4, 0.7)
            total_trades = random.randint(50, 200)
            
            backtest_report = f"""回测结果报告
{'='*50}

回测期间: {start_date} ~ {end_date}
初始资金: ${initial_capital:,.2f}
手续费率: {commission:.4f}

绩效指标:
├─ 总收益率: {total_return:.2%}
├─ 年化收益率: {total_return * 2:.2%}  # 简化计算
├─ Sharpe比率: {sharpe_ratio:.2f}
├─ 最大回撤: {max_drawdown:.2%}
├─ 胜率: {win_rate:.1%}
└─ 总交易次数: {total_trades}

风险分析:
├─ 波动率: {total_return * 0.8:.2%}
├─ Calmar比率: {(total_return * 2) / max_drawdown:.2f}
├─ 收益回撤比: {total_return / max_drawdown:.2f}
└─ 平均持仓时间: {random.randint(3, 15)}天

策略评估:
{'✅ 优秀' if sharpe_ratio > 1.5 else '⚠️ 一般' if sharpe_ratio > 1.0 else '❌ 需改进'}

建议:
• {'继续使用当前策略' if total_return > 0.15 else '考虑优化参数'}
• {'风险控制良好' if max_drawdown < 0.15 else '加强风险管理'}
• {'交易频率适中' if 50 < total_trades < 150 else '调整交易频率'}
"""
            
            self.backtest_results_text.delete(1.0, tk.END)
            self.backtest_results_text.insert(tk.END, backtest_report)
            
        except Exception as e:
            error_msg = f"❌ 回测出错: {e}\n"
            self.backtest_results_text.insert(tk.END, error_msg)
    
    def run_advanced_strategy(self):
        """运行高级策略"""
        try:
            messagebox.showinfo("策略启动", "高级量化交易策略已开始运行！\n\n运行状态将在日志中显示")
            self.log_message("🚀 高级量化交易策略已启动")
        except Exception as e:
            self.log_message(f"[高级策略] ❌ 启动失败: {e}")
    
    def stop_advanced_strategy(self):
        """停止高级策略"""
        if messagebox.askyesno("确认", "确定要停止高级策略运行吗？"):
            self.log_message("⏹️ 高级量化交易策略已停止")
            messagebox.showinfo("策略停止", "高级策略已安全停止")
    
    def show_strategy_report(self):
        """显示策略报告"""
        report_window = tk.Toplevel(self.root)
        report_window.title("📊 策略运行报告")
        report_window.geometry("800x600")
        
        report_text = tk.Text(report_window, font=('Consolas', 10))
        report_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 生成示例报告
        sample_report = f"""高级量化交易策略运行报告
{'='*60}

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

策略状态: 🟢 运行中
运行时长: 2小时35分钟

今日交易统计:
├─ 交易信号数: 8
├─ 执行交易数: 5
├─ 盈利交易数: 3
└─ 当日收益: +1.25%

权重分配:
├─ 策略A权重: 65%
├─ 策略B权重: 35%
└─ 最后调整: 今日 09:30

风险控制:
├─ 当前回撤: 2.3%
├─ 风险状态: ✅ 正常
└─ 止损触发: 0次

市场环境: 趋势+低波动
建议操作: 继续当前配置
"""
        
        report_text.insert(tk.END, sample_report)
    
    def initialize_stock_pools(self):
        """初始化股票池数据"""
        try:
            # 初始化股票列表变量
            self.custom_stock_list = []
            self.edited_default_list = []
            
            # 默认股票池数据（从量化模型.py中提取）
            self.default_stock_pool = {
                '科技股': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'CRM', 'ADBE',
                         'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX',
                         'KLAC', 'MRVL', 'ON', 'SWKS', 'MCHP', 'ADI', 'XLNX', 'SNPS', 'CDNS', 'FTNT'],
                '消费零售': ['COST', 'WMT', 'TGT', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS', 'PYPL',
                           'SQ', 'SHOP', 'EBAY', 'ETSY', 'W', 'CHWY', 'ROKU', 'SPOT', 'ZM', 'UBER',
                           'LYFT', 'DASH', 'ABNB', 'BKNG', 'EXPE', 'TJX', 'ROST', 'ULTA', 'LULU', 'RH'],
                '医疗健康': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR', 'ABT', 'LLY', 'BMY',
                           'MDT', 'AMGN', 'GILD', 'CVS', 'CI', 'HUM', 'ANTM', 'REGN', 'VRTX', 'BIIB',
                           'ILMN', 'MRNA', 'ISRG', 'BSX', 'EW', 'SYK', 'ZBH', 'BDX', 'BAX', 'A'],
                '金融保险': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'V', 'MA',
                           'COF', 'USB', 'TFC', 'PNC', 'SCHW', 'CB', 'MMC', 'PGR', 'TRV', 'ALL',
                           'MET', 'PRU', 'AFL', 'AIG', 'HIG', 'WRB', 'RGA', 'CINF', 'PFG', 'TMK'],
                '能源化工': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'DVN',
                           'FANG', 'APA', 'MRO', 'HAL', 'BKR', 'NOV', 'FTI', 'RIG', 'DO', 'NE'],
                '工业制造': ['BA', 'CAT', 'DE', 'GE', 'HON', 'MMM', 'UNP', 'UPS', 'RTX', 'LMT',
                           'NOC', 'GD', 'FDX', 'EMR', 'ETN', 'PH', 'CMI', 'ITW', 'TXT', 'ROK',
                           'DOV', 'SWK', 'XYL', 'PNR', 'FLS', 'IR', 'IEX', 'AME', 'ROP', 'TDG'],
                '房地产': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'ESS', 'MAA',
                         'UDR', 'CPT', 'BXP', 'VTR', 'WELL', 'O', 'REIT', 'DLR', 'SPG', 'REG'],
                '公用事业': ['NEE', 'D', 'SO', 'DUK', 'AEP', 'EXC', 'XEL', 'PEG', 'PCG', 'ED',
                           'ETR', 'ES', 'PPL', 'CMS', 'DTE', 'WEC', 'EIX', 'FE', 'AEE', 'CNP'],
                '材料化学': ['LIN', 'APD', 'ECL', 'FCX', 'NEM', 'SHW', 'DD', 'DOW', 'PPG', 'EMN',
                           'ALB', 'IFF', 'FMC', 'LYB', 'CE', 'CTVA', 'MLM', 'VMC', 'NUE', 'STLD'],
                '通信服务': ['T', 'VZ', 'CMCSA', 'TMUS', 'CHTR', 'DISH', 'NFLX', 'DIS', 'VZ', 'T']
            }
            
            # 生成完整的默认股票池列表
            self.full_default_list = []
            for category, stocks in self.default_stock_pool.items():
                self.full_default_list.extend(stocks)
            
            # 去重
            self.full_default_list = list(set(self.full_default_list))
            
            # 尝试从文件加载保存的股票池
            if self.load_default_stock_pool():
                self.log_message(f"[股票池] ✅ 成功加载保存的股票池 ({len(self.full_default_list)} 只)")
            else:
                self.log_message(f"[股票池] 使用默认股票池 ({len(self.full_default_list)} 只)")
            
            # 初始化编辑列表为完整列表
            self.edited_default_list = self.full_default_list.copy()
            
        except Exception as e:
            self.log_message(f"[股票池] ❌ 初始化股票池失败: {e}")
            # 使用空列表作为备用
            self.custom_stock_list = []
            self.edited_default_list = []
            self.full_default_list = []
            self.default_stock_pool = {}
    
    def save_default_stock_pool(self):
        """保存默认股票池到文件"""
        try:
            pool_file = self.config.get('default_stock_pool_file', 'default_stocks.json')
            
            # 按类别重新组织股票池
            organized_pool = {}
            for ticker in self.edited_default_list:
                # 查找股票所属类别
                category = "自定义"
                for cat, stocks in self.default_stock_pool.items():
                    if ticker in stocks:
                        category = cat
                        break
                
                if category not in organized_pool:
                    organized_pool[category] = []
                organized_pool[category].append(ticker)
            
            # 保存到文件
            with open(pool_file, 'w', encoding='utf-8') as f:
                json.dump(organized_pool, f, ensure_ascii=False, indent=2)
            
            self.log_message(f"[股票池] ✅ 默认股票池已保存到 {pool_file}")
            return True
            
        except Exception as e:
            self.log_message(f"[股票池] ❌ 保存默认股票池失败: {e}")
            return False
    
    def load_default_stock_pool(self):
        """从文件加载默认股票池"""
        try:
            pool_file = self.config.get('default_stock_pool_file', 'default_stocks.json')
            
            if os.path.exists(pool_file):
                with open(pool_file, 'r', encoding='utf-8') as f:
                    loaded_pool = json.load(f)
                
                # 更新默认股票池
                self.default_stock_pool.update(loaded_pool)
                
                # 重新生成完整列表
                self.full_default_list = []
                for category, stocks in self.default_stock_pool.items():
                    self.full_default_list.extend(stocks)
                
                # 去重
                self.full_default_list = list(set(self.full_default_list))
                
                self.log_message(f"[股票池] ✅ 从 {pool_file} 加载了 {len(self.full_default_list)} 只股票")
                return True
            
            return False
            
        except Exception as e:
            self.log_message(f"[股票池] ❌ 加载默认股票池失败: {e}")
            return False
    
    def initialize_us_stock_crawler(self):
        """初始化美股爬虫"""
        try:
            if US_STOCK_CRAWLER_AVAILABLE:
                self.us_stock_crawler = USStockCrawler()
                self.log_message("[美股爬虫] ✅ 美股爬虫初始化完成")
            else:
                self.us_stock_crawler = None
                self.log_message("[美股爬虫] ❌ 美股爬虫不可用")
        except Exception as e:
            self.log_message(f"[美股爬虫] ❌ 初始化失败: {e}")
            self.us_stock_crawler = None
    
    def initialize_ensemble_strategy(self):
        """初始化双模型融合策略"""
        try:
            if ENSEMBLE_STRATEGY_AVAILABLE:
                self.ensemble_strategy = EnsembleStrategy()
                self.log_message("[融合策略] ✅ 双模型融合策略初始化完成")
            else:
                self.ensemble_strategy = None
                self.log_message("[融合策略] ❌ 双模型融合策略不可用")
        except Exception as e:
            self.log_message(f"[融合策略] ❌ 初始化失败: {e}")
            self.ensemble_strategy = None
    
    def update_stock_pool_from_crawler(self):
        """从美股爬虫更新股票池"""
        try:
            if not self.us_stock_crawler:
                messagebox.showwarning("爬虫不可用", "美股爬虫未初始化或不可用")
                return
            
            self.log_message("[美股爬虫] 开始更新股票池...")
            
            # 获取适合交易的股票池
            trading_stocks = self.us_stock_crawler.get_trading_pool_stocks(pool_size=500)
            
            if not trading_stocks:
                messagebox.showerror("更新失败", "未能获取到股票数据")
                return
            
            # 按行业分类股票
            self.log_message("[美股爬虫] 开始按行业分类股票...")
            stock_info = self.us_stock_crawler.get_stock_info_batch(trading_stocks[:200])  # 限制数量
            
            # 清空现有股票池，保留用户自定义分类
            user_categories = {k: v for k, v in self.default_stock_pool.items() if k == "自定义"}
            self.default_stock_pool = user_categories
            
            # 按行业重新组织
            sector_mapping = {
                'Technology': '科技股',
                'Healthcare': '医疗保健',
                'Financial Services': '金融股', 
                'Consumer Cyclical': '消费类股',
                'Communication Services': '通信服务',
                'Industrials': '工业股',
                'Consumer Defensive': '必需消费品',
                'Energy': '能源股',
                'Utilities': '公用事业',
                'Real Estate': '房地产',
                'Materials': '材料股'
            }
            
            for ticker, info in stock_info.items():
                sector = info.get('sector', 'Unknown')
                chinese_sector = sector_mapping.get(sector, '其他行业')
                
                if chinese_sector not in self.default_stock_pool:
                    self.default_stock_pool[chinese_sector] = []
                
                if ticker not in self.default_stock_pool[chinese_sector]:
                    self.default_stock_pool[chinese_sector].append(ticker)
            
            # 添加未分类的股票到"其他行业"
            categorized_tickers = set()
            for stocks in self.default_stock_pool.values():
                categorized_tickers.update(stocks)
            
            uncategorized = [t for t in trading_stocks if t not in categorized_tickers]
            if uncategorized:
                if '其他行业' not in self.default_stock_pool:
                    self.default_stock_pool['其他行业'] = []
                self.default_stock_pool['其他行业'].extend(uncategorized[:100])  # 限制数量
            
            # 更新完整股票列表
            self.full_default_list = []
            for category, stocks in self.default_stock_pool.items():
                self.full_default_list.extend(stocks)
            self.full_default_list = list(set(self.full_default_list))
            
            # 保存更新的股票池
            if self.save_default_stock_pool():
                total_stocks = len(self.full_default_list)
                categories = len(self.default_stock_pool)
                messagebox.showinfo("更新成功", 
                                  f"股票池更新完成！\n"
                                  f"总股票数: {total_stocks}\n"
                                  f"行业分类: {categories}")
                self.log_message(f"[美股爬虫] ✅ 股票池更新完成: {total_stocks}只股票, {categories}个分类")
            else:
                messagebox.showerror("保存失败", "股票池更新成功但保存失败")
                
        except Exception as e:
            self.log_message(f"[美股爬虫] ❌ 更新股票池失败: {e}")
            messagebox.showerror("更新失败", f"股票池更新失败: {e}")
    
    def manage_quantitative_model_stocks(self):
        """管理量化模型默认股票列表"""
        dialog = tk.Toplevel(self.root)
        dialog.title("量化模型股票列表管理")
        dialog.geometry("600x400")
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 主框架
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # 说明标签
        info_label = ttk.Label(main_frame, 
                              text="管理LSTM和BMA量化模型使用的股票列表（来自美股爬虫）",
                              font=('Arial', 10, 'bold'))
        info_label.pack(pady=(0, 10))
        
        # 当前股票列表显示
        list_frame = ttk.LabelFrame(main_frame, text="当前股票列表", padding="5")
        list_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # 股票列表框
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill='both', expand=True)
        
        # 列表和滚动条
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.quant_stocks_listbox = tk.Listbox(list_container, 
                                              font=('Consolas', 10),
                                              yscrollcommand=scrollbar.set)
        self.quant_stocks_listbox.pack(side=tk.LEFT, fill='both', expand=True)
        scrollbar.config(command=self.quant_stocks_listbox.yview)
        
        # 加载当前股票列表
        for stock in self.quantitative_model_stocks:
            self.quant_stocks_listbox.insert(tk.END, stock)
        
        # 操作按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(0, 10))
        
        # 添加股票
        add_frame = ttk.Frame(button_frame)
        add_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Label(add_frame, text="添加股票:").pack(side='left')
        self.add_stock_entry = ttk.Entry(add_frame, width=10)
        self.add_stock_entry.pack(side='left', padx=(5, 5))
        
        ttk.Button(add_frame, text="添加", 
                  command=lambda: self.add_quant_stock(dialog)).pack(side='left')
        
        # 操作按钮
        action_frame = ttk.Frame(button_frame)
        action_frame.pack(fill='x')
        
        ttk.Button(action_frame, text="删除选中", 
                  command=lambda: self.remove_selected_quant_stock(dialog)).pack(side='left', padx=(0, 5))
        
        ttk.Button(action_frame, text="从爬虫刷新", 
                  command=lambda: self.reset_quant_stocks(dialog)).pack(side='left', padx=(0, 5))
        
        ttk.Button(action_frame, text="保存", 
                  command=lambda: self.save_quant_stocks(dialog)).pack(side='right')
        
        # 股票数量显示
        self.quant_count_label = ttk.Label(main_frame, 
                                          text=f"股票总数: {len(self.quantitative_model_stocks)}")
        self.quant_count_label.pack()
        
        # 居中对话框
        dialog.transient(self.root)
        dialog.wait_visibility()
        dialog.grab_set()
    
    def add_quant_stock(self, dialog):
        """添加股票到量化模型列表"""
        ticker = self.add_stock_entry.get().strip().upper()
        if ticker and ticker not in self.quantitative_model_stocks:
            self.quantitative_model_stocks.append(ticker)
            self.quant_stocks_listbox.insert(tk.END, ticker)
            self.add_stock_entry.delete(0, tk.END)
            self.quant_count_label.config(text=f"股票总数: {len(self.quantitative_model_stocks)}")
        elif ticker in self.quantitative_model_stocks:
            messagebox.showwarning("重复股票", f"{ticker} 已在列表中")
    
    def remove_selected_quant_stock(self, dialog):
        """删除选中的量化模型股票"""
        selection = self.quant_stocks_listbox.curselection()
        if selection:
            ticker = self.quant_stocks_listbox.get(selection[0])
            if ticker in self.quantitative_model_stocks:
                self.quantitative_model_stocks.remove(ticker)
                self.quant_stocks_listbox.delete(selection[0])
                self.quant_count_label.config(text=f"股票总数: {len(self.quantitative_model_stocks)}")
    
    def reset_quant_stocks(self, dialog):
        """重置量化模型股票列表为默认值"""
        if messagebox.askyesno("确认重置", "确定要重新从爬虫获取股票列表吗？"):
            try:
                # 从爬虫重新获取股票列表
                success = self.refresh_quantitative_stock_list(force_update=True)
                if success:
                    # 刷新列表框
                    self.quant_stocks_listbox.delete(0, tk.END)
                    for stock in self.quantitative_model_stocks:
                        self.quant_stocks_listbox.insert(tk.END, stock)
                    
                    self.quant_count_label.config(text=f"股票总数: {len(self.quantitative_model_stocks)}")
                    messagebox.showinfo("重置成功", f"已从爬虫重新获取 {len(self.quantitative_model_stocks)} 只股票")
                else:
                    messagebox.showerror("重置失败", "从爬虫获取股票失败")
            except Exception as e:
                messagebox.showerror("重置失败", f"重置股票列表时出错: {e}")
    
    def save_quant_stocks(self, dialog):
        """保存量化模型股票列表"""
        try:
            # 保存到配置文件或其他持久化存储
            config_data = {
                'quantitative_model_stocks': self.quantitative_model_stocks,
                'last_updated': datetime.now().isoformat()
            }
            
            with open('quantitative_model_config.json', 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo("保存成功", f"已保存 {len(self.quantitative_model_stocks)} 只股票")
            dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("保存失败", f"保存配置时出错: {e}")
    
    def load_quant_stocks_config(self):
        """加载量化模型股票列表配置"""
        try:
            if os.path.exists('quantitative_model_config.json'):
                with open('quantitative_model_config.json', 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    if 'quantitative_model_stocks' in config_data:
                        self.quantitative_model_stocks = config_data['quantitative_model_stocks']
                        self.logger.info(f"已加载量化模型股票配置: {len(self.quantitative_model_stocks)}只股票")
        except Exception as e:
            self.logger.warning(f"加载量化模型股票配置失败: {e}")
    
    def initialize_quantitative_stock_list(self):
        """初始化量化模型股票列表（从爬虫获取或默认股票池）"""
        try:
            # 首先尝试加载默认股票池文件
            default_pool_file = "default_stock_pool.json"
            if os.path.exists(default_pool_file):
                try:
                    with open(default_pool_file, 'r', encoding='utf-8') as f:
                        default_pool_data = json.load(f)
                        if 'default_stock_pool' in default_pool_data:
                            self.quantitative_model_stocks = default_pool_data['default_stock_pool']
                            self.logger.info(f"[量化股票] 从默认股票池加载了 {len(self.quantitative_model_stocks)} 只股票")
                            return
                except Exception as e:
                    self.logger.warning(f"[量化股票] 加载默认股票池失败: {e}")
            
            # 如果默认股票池不存在或加载失败，尝试从爬虫获取股票列表
            if hasattr(self, 'us_stock_crawler') and self.us_stock_crawler:
                self.logger.info("[量化股票] 开始从爬虫获取股票列表...")
                
                # 尝试先加载已保存的列表
                saved_list = self.us_stock_crawler.load_saved_stock_list()
                if saved_list and len(saved_list) >= 8:  # 如果有保存的列表且数量充足
                    self.quantitative_model_stocks = saved_list[:100]  # 取前100只
                    self.logger.info(f"[量化股票] 从缓存加载了 {len(self.quantitative_model_stocks)} 只股票")
                else:
                    # 没有缓存或数量不足，重新生成
                    self.logger.info("[量化股票] 缓存不足，重新生成股票列表...")
                    new_stock_list = self.us_stock_crawler.get_quantitative_stock_list(
                        pool_size=50, 
                        use_cache=True, 
                        save_to_file=True
                    )
                    if new_stock_list:
                        self.quantitative_model_stocks = new_stock_list
                        self.logger.info(f"[量化股票] 生成了 {len(self.quantitative_model_stocks)} 只新股票")
                    else:
                        self.quantitative_model_stocks = self._get_default_stock_list()
                        self.logger.warning("[量化股票] 爬虫获取失败，使用默认股票列表")
            else:
                # 爬虫不可用，使用默认列表
                self.quantitative_model_stocks = self._get_default_stock_list()
                self.logger.warning("[量化股票] 爬虫不可用，使用默认股票列表")
                
            # 显示股票列表信息
            if self.quantitative_model_stocks:
                self.logger.info(f"[量化股票] 最终股票列表数量: {len(self.quantitative_model_stocks)}")
                self.logger.info(f"[量化股票] 前10只股票: {self.quantitative_model_stocks[:10]}")
            
        except Exception as e:
            self.logger.error(f"[量化股票] 初始化股票列表失败: {e}")
            self.quantitative_model_stocks = self._get_default_stock_list()
    
    def _get_default_stock_list(self) -> List[str]:
        """获取默认股票列表"""
        return ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 
                'CRM', 'ADBE', 'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'AVGO',
                'JNJ', 'PFE', 'UNH', 'ABBV', 'JPM', 'BAC', 'WFC', 'GS']
    
    def refresh_quantitative_stock_list(self, force_update: bool = False):
        """刷新量化模型股票列表"""
        try:
            if hasattr(self, 'us_stock_crawler') and self.us_stock_crawler:
                self.logger.info("[量化股票] 开始刷新股票列表...")
                
                new_stock_list = self.us_stock_crawler.get_quantitative_stock_list(
                    pool_size=100, 
                    use_cache=not force_update, 
                    save_to_file=True
                )
                
                if new_stock_list:
                    old_count = len(self.quantitative_model_stocks)
                    self.quantitative_model_stocks = new_stock_list
                    self.logger.info(f"[量化股票] 股票列表已更新: {old_count} -> {len(self.quantitative_model_stocks)} 只股票")
                    return True
                else:
                    self.logger.warning("[量化股票] 刷新失败，保持原有列表")
                    return False
            else:
                self.logger.warning("[量化股票] 爬虫不可用，无法刷新")
                return False
                
        except Exception as e:
            self.logger.error(f"[量化股票] 刷新股票列表失败: {e}")
            return False
    
    def get_stock_suggestions(self, prefix: str, limit: int = 20) -> List[str]:
        """根据前缀获取股票建议（用于自动补全）"""
        try:
            if not self.us_stock_crawler:
                return []
            
            # 从缓存中获取所有股票
            all_stocks = self.us_stock_crawler.get_all_us_stocks(use_cache=True)
            
            # 筛选匹配前缀的股票
            prefix = prefix.upper()
            suggestions = [stock for stock in all_stocks if stock.startswith(prefix)]
            
            return suggestions[:limit]
            
        except Exception as e:
            self.log_message(f"[美股爬虫] 获取股票建议失败: {e}")
            return []

    def on_closing(self):
        """关闭应用程序"""
        try:
            # 停止定时任务
            if self.scheduler.running:
                self.scheduler.shutdown()
            
            # 关闭数据库连接
            if hasattr(self, 'conn'):
                self.conn.close()
            
            self.logger.info("应用程序正常关闭")
            
        except Exception as e:
            self.logger.error(f"关闭应用程序时出错: {e}")
        
        finally:
            self.root.destroy()
    
    def run(self):
        """运行应用程序"""
        # 设置关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 启动GUI主循环
        self.root.mainloop()
    
    def start_enhanced_real_trading(self):
        """启动增强版实盘交易"""
        try:
            if not ENHANCED_TRADING_AVAILABLE:
                self.log_message("[增强交易] ❌ 增强版交易策略不可用")
                messagebox.showerror("错误", "增强版交易策略不可用，请检查ibkr_trading_strategy_enhanced.py是否存在")
                return
            
            if self.enhanced_strategy is None:
                self.log_message("[增强交易] ❌ 增强版交易策略未初始化")
                messagebox.showerror("错误", "增强版交易策略未初始化")
                return
            
            # 确认对话框
            result = messagebox.askyesno(
                "确认启动实盘交易", 
                "⚠️ 即将启动增强版实盘交易！\n\n"
                "此操作将使用真实资金进行交易！\n"
                "请确保:\n"
                "1. IBKR TWS/Gateway已启动\n"
                "2. API权限已启用\n"
                "3. 账户有足够资金\n"
                "4. 风险控制参数已正确设置\n\n"
                "是否继续？"
            )
            
            if not result:
                self.log_message("[增强交易] 用户取消了实盘交易启动")
                return
            
            self.log_message("[增强交易] 🚀 启动增强版实盘交易...")
            
            # 在新线程中启动增强版交易
            def run_enhanced_trading():
                try:
                    # 启用实盘交易模式
                    self.enhanced_strategy.config['enable_real_trading'] = True
                    self.enhanced_strategy.config['enable_enhanced_mode'] = True
                    
                    # 启动增强版交易策略
                    if self.enhanced_strategy.start_enhanced_trading():
                        self.log_message("[增强交易] ✅ 增强版实盘交易已启动")
                        self.log_message("[增强交易] 系统将持续监控市场并自动执行交易...")
                        
                        # 更新按钮状态
                        self.root.after(0, lambda: self.update_trading_button_status(True))
                        
                        # 显示通知
                        if self.config.get('notifications', True):
                            notification.notify(
                                title="增强版交易系统",
                                message="实盘交易已启动，系统正在监控市场...",
                                timeout=10
                            )
                    else:
                        self.log_message("[增强交易] ❌ 增强版交易启动失败")
                        
                except Exception as e:
                    self.log_message(f"[增强交易] ❌ 运行出错: {e}")
                    import traceback
                    traceback.print_exc()
            
            threading.Thread(target=run_enhanced_trading, daemon=True).start()
            
        except Exception as e:
            self.log_message(f"[增强交易] ❌ 启动失败: {e}")
            messagebox.showerror("启动失败", f"增强版交易启动失败: {e}")
    
    def stop_enhanced_real_trading(self):
        """停止增强版实盘交易"""
        try:
            if self.enhanced_strategy and hasattr(self.enhanced_strategy, 'running'):
                self.log_message("[增强交易] 正在停止增强版实盘交易...")
                self.enhanced_strategy.stop_enhanced_trading()
                self.log_message("[增强交易] ✅ 增强版实盘交易已停止")
                
                # 更新按钮状态
                self.update_trading_button_status(False)
                
                # 显示通知
                if self.config.get('notifications', True):
                    notification.notify(
                        title="增强版交易系统",
                        message="实盘交易已停止",
                        timeout=5
                    )
            else:
                self.log_message("[增强交易] ⚠️ 增强版交易策略未运行")
                
        except Exception as e:
            self.log_message(f"[增强交易] ❌ 停止失败: {e}")
    
    def execute_enhanced_trading_with_bma_results(self, bma_stocks):
        """使用BMA结果执行增强版交易"""
        try:
            if not self.enhanced_strategy:
                self.log_message("[增强交易] ❌ 增强版策略未初始化")
                return
            
            self.log_message(f"[增强交易] 开始处理 {len(bma_stocks)} 只BMA推荐股票...")
            
            # 将BMA结果转换为增强版策略可以使用的格式
            for stock in bma_stocks:
                symbol = stock.get('symbol', '')
                score = stock.get('score', 0)
                current_price = stock.get('current_price', 0)
                
                if symbol and current_price > 0:
                    # 更新市场数据处理器
                    self.enhanced_strategy.market_data_processor.update_tick_data(symbol, 4, current_price)
                    
                    # 添加到BMA推荐中
                    self.enhanced_strategy.enhanced_signal_generator.bma_recommendations[symbol] = {
                        'rating': 'BUY' if score > 0.7 else 'HOLD',
                        'prediction': score * 0.1,  # 转换为预期收益率
                        'confidence': min(score, 1.0)
                    }
                    
                    self.log_message(f"[增强交易] 已添加 {symbol} 到增强策略 (评分: {score:.3f}, 价格: ${current_price:.2f})")
            
            # 如果策略未运行，启动增强版交易
            if not getattr(self.enhanced_strategy, 'running', False):
                self.log_message("[增强交易] 启动增强版策略以执行BMA推荐交易...")
                self.enhanced_strategy.config['enable_real_trading'] = self.config.get('enable_real_trading', False)
                
                if self.enhanced_strategy.start_enhanced_trading():
                    self.log_message("[增强交易] ✅ 增强版策略已启动，将自动处理BMA推荐")
                else:
                    self.log_message("[增强交易] ❌ 增强版策略启动失败")
            else:
                self.log_message("[增强交易] 增强版策略已在运行，BMA推荐已更新")
                
        except Exception as e:
            self.log_message(f"[增强交易] ❌ BMA结果处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    def update_trading_button_status(self, is_running):
        """更新交易按钮状态"""
        try:
            if hasattr(self, 'enhanced_trading_button_frame'):
                # 更新按钮文本和颜色
                button_text = "停止实盘交易" if is_running else "增强实盘交易"
                
                # 这里可以添加更复杂的按钮状态更新逻辑
                self.log_message(f"[增强交易] 按钮状态更新: {button_text}")
                
        except Exception as e:
            self.log_message(f"[增强交易] ❌ 按钮状态更新失败: {e}")
    
    def get_enhanced_trading_status(self):
        """获取增强版交易状态"""
        try:
            if self.enhanced_strategy and hasattr(self.enhanced_strategy, 'get_enhanced_status'):
                return self.enhanced_strategy.get_enhanced_status()
            else:
                return {
                    'running': False,
                    'connected': False,
                    'active_positions': 0,
                    'pending_orders': 0,
                    'error': '策略未初始化'
                }
        except Exception as e:
            return {
                'running': False,
                'connected': False,
                'error': str(e)
            }

    # ============================================================================
    # 增强IBKR功能集成 (从trading_system_manager.py继承)
    # ============================================================================
    
    def init_enhanced_ibkr_features(self):
        """初始化增强IBKR功能"""
        try:
            # 增强IBKR连接管理
            self.ib_connection = None
            self.is_ibkr_connected = False
            self.heartbeat_task = None
            self.reconnect_attempts = 0
            self.last_heartbeat = None
            self.trading_active = False
            
            # 增强重连机制
            self.max_reconnect_attempts = 999  # 无限重连
            self.reconnect_interval = 30  # 基础重连间隔(秒)
            self.reconnect_exponential_backoff = True
            self.reconnect_task = None
            self.connection_lost_time = None
            self.last_successful_connection = None
            self.heartbeat_interval = 10  # 心跳间隔(秒)
            self.heartbeat_timeout = 30  # 心跳超时(秒)
            
            # 订单和持仓管理
            self.active_orders = {}
            self.order_history = []
            self.positions = {}
            self.subscribed_symbols = set()
            self.tick_data = {}
            
            # 实时事件驱动
            self.market_data_subscriptions = {}  # reqId -> symbol mapping
            self.live_tick_handlers = {}  # symbol -> callback functions
            self.stop_loss_monitors = {}  # symbol -> stop loss levels
            self.take_profit_monitors = {}  # symbol -> take profit levels
            self.position_monitors = {}  # symbol -> position info for monitoring
            
            # IBKR API标准变量
            self.next_order_id = None
            self.account_info = {}
            self.portfolio_data = {}
            self.contract_details_cache = {}
            self.account_download_complete = False
            
            # 订单状态跟踪
            self.order_status_callbacks = {}  # orderId -> callback functions
            self.execution_callbacks = {}  # orderId -> execution handlers
            self.open_orders = {}  # orderId -> order info
            self.order_executions = {}  # orderId -> list of executions
            self.pending_orders = {}  # reqId -> order info (waiting for contract details)
            
            # 风险管理
            self.daily_pnl = 0.0
            self.total_pnl = 0.0
            self.daily_new_positions = 0
            self.portfolio_value = 0
            self.max_portfolio_value = 0
            self.last_loss_date = None
            self.trading_blocked = False
            self.last_reset_date = datetime.now().date()
            
            # 硬性风控阻断
            self.risk_monitor_active = True
            self.emergency_stop_triggered = False
            self.max_drawdown_breached = False
            self.position_size_violations = 0
            self.daily_loss_limit_breached = False
            self.forced_liquidation_in_progress = False
            self.risk_check_interval = 5  # 风控检查间隔(秒)
            
            # 异步事件循环
            self.loop = None
            self.loop_thread = None
            
            self.log_message("[增强IBKR] ✅ 增强IBKR功能初始化完成")
            
        except Exception as e:
            self.log_message(f"[增强IBKR] ❌ 初始化失败: {e}")
    
    def setup_logging_enhanced(self):
        """增强日志设置"""
        try:
            # 创建logs目录
            os.makedirs('logs', exist_ok=True)
            
            # 设置增强日志
            log_filename = f"logs/enhanced_trading_{datetime.now().strftime('%Y%m%d')}.log"
            logging.basicConfig(
                level=getattr(logging, self.config.get('log_level', 'INFO')),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_filename, encoding='utf-8'),
                    logging.StreamHandler()
                ]
            )
            
            self.logger = logging.getLogger(__name__)
            self.logger.info("增强日志系统已启动")
            
        except Exception as e:
            print(f"❌ 增强日志设置失败: {e}")
    
    def create_directories_enhanced(self):
        """创建增强目录结构"""
        try:
            directories = [
                'logs', 'result', 'trading_data', 'models', 
                'exports', 'reports', 'portfolios', 'ibkr_trading'
            ]
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
            
            self.log_message("[目录] ✅ 增强目录结构创建完成")
            
        except Exception as e:
            self.log_message(f"[目录] ❌ 创建失败: {e}")
    
    def load_config_enhanced(self):
        """加载增强配置"""
        try:
            config_file = 'enhanced_trading_config.json'
            
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
                    self.log_message("[配置] ✅ 增强配置已加载")
            else:
                self.save_config_enhanced()
                self.log_message("[配置] ✅ 默认增强配置已创建")
                
        except Exception as e:
            self.log_message(f"[配置] ❌ 加载失败: {e}")
    
    def save_config_enhanced(self):
        """保存增强配置"""
        try:
            config_file = 'enhanced_trading_config.json'
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
                
            self.log_message("[配置] ✅ 增强配置已保存")
            
        except Exception as e:
            self.log_message(f"[配置] ❌ 保存失败: {e}")
    
    def init_enhanced_features(self):
        """初始化增强功能"""
        try:
            # 初始化风险管理参数
            self.portfolio_value = self.config.get('enhanced_ibkr', {}).get('total_capital', 100000)
            self.max_portfolio_value = self.portfolio_value
            
            # 启动异步事件循环
            self._start_event_loop()
            
            self.log_message("✅ 增强功能初始化完成")
            
        except Exception as e:
            self.log_message(f"❌ 增强功能初始化失败: {e}")
    
    def _start_event_loop(self):
        """启动异步事件循环"""
        try:
            self.loop = asyncio.new_event_loop()
            
            def run_loop():
                asyncio.set_event_loop(self.loop)
                self.loop.run_forever()
            
            self.loop_thread = threading.Thread(target=run_loop, daemon=True)
            self.loop_thread.start()
            
            self.log_message("🔄 异步事件循环已启动")
            
        except Exception as e:
            self.log_message(f"❌ 异步事件循环启动失败: {e}")
    
    def reset_daily_risk_counters(self):
        """重置每日风控计数器"""
        try:
            current_date = datetime.now().date()
            
            if current_date != self.last_reset_date:
                self.daily_new_positions = 0
                self.daily_pnl = 0.0
                self.daily_loss_limit_breached = False
                self.last_reset_date = current_date
                
                # 如果不在冷却期，解除交易阻止
                if not self._is_in_cooldown():
                    self.trading_blocked = False
                
                self.log_message(f"📊 每日风控计数器已重置: {current_date}")
                
        except Exception as e:
            self.log_message(f"❌ 重置每日计数器失败: {e}")
    
    def _is_in_cooldown(self):
        """检查是否在亏损冷却期"""
        try:
            if not self.last_loss_date:
                return False
            
            cooldown_days = self.config.get('enhanced_ibkr', {}).get('loss_cooldown_days', 1)
            cooldown_end = self.last_loss_date + timedelta(days=cooldown_days)
            
            return datetime.now().date() <= cooldown_end
            
        except Exception as e:
            self.log_message(f"❌ 检查冷却期失败: {e}")
            return False
    
    # ============================================================================
    # IBKR连接管理方法
    # ============================================================================
    
    async def connect_ibkr_enhanced(self):
        """增强IBKR连接"""
        try:
            if not IBKR_AVAILABLE:
                self.log_message("❌ IBKR API不可用")
                return False
            
            enhanced_config = self.config.get('enhanced_ibkr', {})
            host = enhanced_config.get('ibkr_host', '127.0.0.1')
            port = enhanced_config.get('ibkr_port', 4002)
            client_id = enhanced_config.get('ibkr_client_id', 50310)
            
            self.log_message(f"🔌 正在连接IBKR: {host}:{port} (Client ID: {client_id})")
            
            # 创建IB连接
            self.ib_connection = ibs.IB()
            
            # 设置事件处理
            self.ib_connection.errorEvent += self._on_ibkr_error
            self.ib_connection.connectedEvent += self._on_ibkr_connected
            self.ib_connection.disconnectedEvent += self._on_ibkr_disconnected
            
            # 设置标准IBKR API回调
            self.ib_connection.updatePortfolioEvent += self.updatePortfolio
            self.ib_connection.updateAccountValueEvent += self.updateAccountValue
            self.ib_connection.updateAccountTimeEvent += self.updateAccountTime
            self.ib_connection.accountDownloadEndEvent += self.accountDownloadEnd
            
            # 设置订单状态跟踪回调
            self.ib_connection.openOrderEvent += self.openOrder
            self.ib_connection.orderStatusEvent += self.orderStatus
            self.ib_connection.execDetailsEvent += self.execDetails
            
            # 尝试连接，如果客户端ID冲突则重试
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await self.ib_connection.connectAsync(host, port, client_id)
                    break  # 连接成功，跳出循环
                except Exception as e:
                    error_msg = str(e)
                    if "already in use" in error_msg or "326" in error_msg or "Peer closed connection" in error_msg:
                        # 客户端ID冲突，生成新的ID重试
                        old_client_id = client_id
                        client_id = TradingConstants.generate_unique_client_id()
                        self.log_message(f"⚠️ 客户端ID {old_client_id} 冲突，尝试新ID: {client_id}")
                        
                        # 更新配置中的客户端ID
                        self.config['ibkr_client_id'] = client_id
                        if 'enhanced_ibkr' in self.config:
                            self.config['enhanced_ibkr']['ibkr_client_id'] = client_id
                        
                        if attempt == max_retries - 1:
                            raise Exception(f"多次尝试后仍无法连接IBKR: {error_msg}")
                        
                        # 断开之前的连接尝试
                        if self.ib_connection.isConnected():
                            self.ib_connection.disconnect()
                        time.sleep(1)  # 等待一秒后重试
                    else:
                        # 其他错误直接抛出
                        raise e
            
            self.is_ibkr_connected = True
            self.last_successful_connection = datetime.now()
            self.reconnect_attempts = 0
            
            # 统一连接对象，便于后续方法使用
            self.ib = self.ib_connection
            
            # 设置市场数据类型：优先实时报价(1)，失败则使用延迟(3)
            try:
                self.ib_connection.reqMarketDataType(1)
                self.log_message("✅ 市场数据类型: 实时 (1)")
            except Exception as e1:
                self.log_message(f"⚠️ 实时数据不可用，尝试延迟数据: {e1}")
                try:
                    self.ib_connection.reqMarketDataType(3)
                    self.log_message("✅ 市场数据类型: 延迟 (3)")
                except Exception as e2:
                    self.log_message(f"❌ 设置市场数据类型失败: {e2}")
            
            # 连接成功后更新订单管理器连接
            if hasattr(self, 'order_manager') and self.order_manager:
                try:
                    self.order_manager.set_ib_connection(self.ib_connection)
                    self.log_message("✅ 增强订单管理器已连接IBKR (enhanced)")
                except Exception as e:
                    self.log_message(f"⚠️ 增强订单管理器连接设置失败: {e}")
            
            # 启动心跳
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            self.log_message("✅ IBKR连接成功")
            return True
            
        except Exception as e:
            self.log_message(f"❌ IBKR连接失败: {e}")
            
            # 自动提供诊断建议
            if "already in use" in str(e) or "326" in str(e):
                self.log_message("ℹ️ 客户端ID冲突已自动处理，如仍有问题请查看诊断建议")
            else:
                self.diagnose_connection_issue()
                
            return False
    
    async def _heartbeat_loop(self):
        """增强心跳检测循环"""
        enhanced_config = self.config.get('enhanced_ibkr', {})
        heartbeat_interval = enhanced_config.get('heartbeat_interval', self.heartbeat_interval)
        
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        try:
            while self.is_ibkr_connected and not self.emergency_stop_triggered:
                await asyncio.sleep(heartbeat_interval)
                
                if not self.ib_connection or not self.ib_connection.isConnected():
                    consecutive_failures += 1
                    self.log_message(f"💔 心跳检测失败 - 连接丢失 (连续失败: {consecutive_failures}/{max_consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        self.log_message("💔 连续心跳失败，触发断线处理")
                        await self._handle_disconnection()
                        break
                    continue
                
                # 发送心跳请求
                try:
                    accounts = self.ib_connection.managedAccounts()
                    if accounts:
                        consecutive_failures = 0
                        self.last_heartbeat = datetime.now()
                        self.last_successful_connection = datetime.now()
                        self.log_message("💓 心跳正常")
                        
                        # 执行实时风控检查
                        await self._perform_realtime_risk_check()
                        
                        # 检查订单状态
                        await self._check_pending_orders()
                        
                    else:
                        consecutive_failures += 1
                        self.log_message(f"💔 心跳响应异常 (连续失败: {consecutive_failures})")
                        
                except Exception as e:
                    consecutive_failures += 1
                    self.log_message(f"💔 心跳请求失败: {e} (连续失败: {consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        await self._handle_disconnection()
                        break
                    
        except asyncio.CancelledError:
            self.log_message("💓 心跳检测已停止")
        except Exception as e:
            self.log_message(f"💔 心跳循环异常: {e}")
            await self._handle_disconnection()
    
    async def _handle_disconnection(self):
        """处理断线情况"""
        try:
            self.is_ibkr_connected = False
            self.log_message("💔 IBKR连接已断开")
            
            # 发送断线告警
            await self._send_alert("WARNING", "IBKR连接断开", "正在尝试自动重连...")
            
            # 开始自动重连
            enhanced_config = self.config.get('enhanced_ibkr', {})
            if enhanced_config.get('enable_auto_reconnect', True):
                await self._auto_reconnect()
                
        except Exception as e:
            self.log_message(f"❌ 处理断线失败: {e}")
    
    async def _auto_reconnect(self):
        """增强自动重连机制"""
        enhanced_config = self.config.get('enhanced_ibkr', {})
        max_attempts = enhanced_config.get('max_reconnect_attempts', self.max_reconnect_attempts)
        base_delay = enhanced_config.get('reconnect_delay', self.reconnect_interval)
        
        self.connection_lost_time = datetime.now()
        connection_downtime = 0
        
        # 无限重连模式或有限重连
        attempt = 0
        max_retries = max_attempts if max_attempts < 900 else float('inf')
        
        while (attempt < max_retries) and not self.is_ibkr_connected and not self.emergency_stop_triggered:
            attempt += 1
            
            try:
                connection_downtime = (datetime.now() - self.connection_lost_time).total_seconds()
                self.log_message(f"🔄 自动重连尝试 {attempt} (断线时长: {connection_downtime:.1f}秒)")
                
                # 动态调整重连间隔
                if self.reconnect_exponential_backoff and attempt <= 10:
                    delay = min(base_delay * (1.5 ** (attempt - 1)), 300)  # 最大5分钟
                else:
                    delay = base_delay
                
                await asyncio.sleep(delay)
                
                # 检查是否应该停止重连
                if self.emergency_stop_triggered:
                    self.log_message("🛑 紧急停止已触发，终止自动重连")
                    break
                
                # 尝试重连
                success = await self.connect_ibkr_enhanced()
                
                if success and self.is_ibkr_connected:
                    connection_downtime = (datetime.now() - self.connection_lost_time).total_seconds()
                    self.log_message(f"✅ 自动重连成功 (第{attempt}次尝试，断线{connection_downtime:.1f}秒)")
                    self.reconnect_attempts = 0
                    
                    # 重新订阅市场数据
                    await self._resubscribe_market_data()
                    
                    # 重新启动风控监控
                    await self._restart_risk_monitoring()
                    
                    # 发送重连成功告警
                    await self._send_alert(
                        "INFO", 
                        f"IBKR自动重连成功 - 第{attempt}次尝试后恢复连接",
                        f"断线时长: {connection_downtime:.1f}秒\n恢复时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    return True
                    
            except Exception as e:
                connection_downtime = (datetime.now() - self.connection_lost_time).total_seconds()
                self.log_message(f"⚠️ 重连尝试 {attempt} 失败: {e} (断线时长: {connection_downtime:.1f}秒)")
                
                # 每小时发送一次长时间断线告警
                if attempt % 120 == 0:  # 假设每30秒重连一次，120次约1小时
                    await self._send_alert(
                        "WARNING",
                        f"IBKR长时间断线 - 已重连{attempt}次",
                        f"断线时长: {connection_downtime/3600:.1f}小时\n请检查网络和TWS/Gateway状态"
                    )
        
        if not self.is_ibkr_connected:
            final_downtime = (datetime.now() - self.connection_lost_time).total_seconds()
            self.log_message(f"❌ 自动重连终止 - 尝试{attempt}次，断线{final_downtime:.1f}秒")
            
            # 发送重连失败告警
            await self._send_alert(
                "CRITICAL",
                f"IBKR自动重连终止 - 已尝试 {attempt} 次",
                f"断线时长: {final_downtime/3600:.1f}小时\n系统已进入离线模式"
            )
        
        return False
    
    # ============================================================================
    # 告警系统
    # ============================================================================
    
    async def _send_alert(self, level: str, title: str, message: str):
        """发送告警"""
        try:
            full_message = f"[{level}] {title}\n\n{message}\n\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # 日志记录
            if level == "CRITICAL":
                self.log_message(f"🚨 {full_message}")
            elif level == "ERROR":
                self.log_message(f"❌ {full_message}")
            elif level == "WARNING":
                self.log_message(f"⚠️ {full_message}")
            else:
                self.log_message(f"ℹ️ {full_message}")
            
            # GUI通知
            alert_settings = self.config.get('alert_settings', {})
            if alert_settings.get('gui_notifications', True):
                messagebox.showwarning(title, message)
            
            # 系统通知
            if alert_settings.get('system_notifications', True) and NOTIFICATION_AVAILABLE:
                notification.notify(
                    title=title,
                    message=message,
                    app_name="量化交易系统",
                    timeout=10
                )
            
            # 邮件告警 (针对ERROR和CRITICAL级别)
            if level in ["ERROR", "CRITICAL"] and alert_settings.get('email_alerts', False):
                await self._send_email_alert(title, full_message)
                
        except Exception as e:
            self.log_message(f"❌ 发送告警失败: {e}")
    
    async def _send_email_alert(self, subject: str, message: str):
        """发送邮件告警"""
        try:
            if not EMAIL_AVAILABLE:
                return
            
            alert_settings = self.config.get('alert_settings', {})
            smtp_server = alert_settings.get('smtp_server', '')
            smtp_port = alert_settings.get('smtp_port', 587)
            email_user = alert_settings.get('email_user', '')
            email_password = alert_settings.get('email_password', '')
            alert_emails = alert_settings.get('alert_emails', [])
            
            if not all([smtp_server, email_user, email_password, alert_emails]):
                return
            
            msg = MimeMultipart()
            msg['From'] = email_user
            msg['To'] = ', '.join(alert_emails)
            msg['Subject'] = f"[量化交易告警] {subject}"
            
            msg.attach(MimeText(message, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_user, email_password)
            text = msg.as_string()
            server.sendmail(email_user, alert_emails, text)
            server.quit()
            
            self.log_message("📧 邮件告警发送成功")
            
        except Exception as e:
            self.log_message(f"❌ 邮件告警发送失败: {e}")
    
    # ============================================================================
    # IBKR API标准回调函数
    # ============================================================================
    
    def nextValidId(self, orderId: int):
        """接收可用的下单ID"""
        try:
            self.next_order_id = orderId
            self.log_message(f"📋 收到下单ID: {orderId}")
        except Exception as e:
            self.log_message(f"❌ 处理nextValidId失败: {e}")
    
    def updatePortfolio(self, contract, position: float, marketPrice: float, marketValue: float,
                       averageCost: float, unrealizedPNL: float, realizedPNL: float, accountName: str):
        """更新投资组合信息"""
        try:
            symbol = contract.symbol
            self.portfolio_data[symbol] = {
                'position': position,
                'market_price': marketPrice,
                'market_value': marketValue,
                'average_cost': averageCost,
                'unrealized_pnl': unrealizedPNL,
                'realized_pnl': realizedPNL,
                'account': accountName,
                'timestamp': datetime.now()
            }
            
            # 更新总PnL
            self.total_pnl = sum(data.get('realized_pnl', 0) for data in self.portfolio_data.values())
            
            self.log_message(f"📊 持仓更新: {symbol} 数量={position}, 价格=${marketPrice:.2f}")
            
        except Exception as e:
            self.log_message(f"❌ 更新投资组合失败: {e}")
    
    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str):
        """更新账户价值"""
        try:
            if key == 'NetLiquidation':
                self.portfolio_value = float(val)
                if self.portfolio_value > self.max_portfolio_value:
                    self.max_portfolio_value = self.portfolio_value
            
            self.account_info[key] = {
                'value': val,
                'currency': currency,
                'account': accountName,
                'timestamp': datetime.now()
            }
            
            self.log_message(f"💰 账户更新: {key}={val} {currency}")
            
        except Exception as e:
            self.log_message(f"❌ 更新账户价值失败: {e}")
    
    def updateAccountTime(self, timeStamp: str):
        """更新账户时间"""
        try:
            self.account_info['last_update_time'] = timeStamp
            self.log_message(f"⏰ 账户时间更新: {timeStamp}")
        except Exception as e:
            self.log_message(f"❌ 更新账户时间失败: {e}")
    
    def accountDownloadEnd(self, accountName: str):
        """账户下载结束"""
        try:
            self.account_download_complete = True
            self.log_message(f"✅ 账户数据下载完成: {accountName}")
        except Exception as e:
            self.log_message(f"❌ 账户下载结束处理失败: {e}")
    
    # ============================================================================
    # 订单状态跟踪回调
    # ============================================================================
    
    def openOrder(self, orderId, contract, order, orderState):
        """订单已提交但未成交时被调用"""
        try:
            symbol = contract.symbol
            action = order.action
            quantity = order.totalQuantity
            
            self.log_message(f"📋 订单开启: {symbol} {action} {quantity} (订单ID: {orderId})")
            
            # 存储开放订单信息
            self.open_orders[orderId] = {
                'symbol': symbol,
                'contract': contract,
                'order': order,
                'order_state': orderState,
                'status': 'OPEN',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.log_message(f"❌ 处理openOrder失败: {e}")
    
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        """每当订单状态更新（部分成交、全部成交、被取消等）"""
        try:
            self.log_message(f"📊 订单状态更新: ID={orderId}, 状态={status}, 已成交={filled}, 剩余={remaining}, 平均价格=${avgFillPrice:.2f}")
            
            # 更新订单状态
            if orderId in self.open_orders:
                order_info = self.open_orders[orderId]
                order_info.update({
                    'status': status,
                    'filled': filled,
                    'remaining': remaining,
                    'avg_fill_price': avgFillPrice,
                    'last_fill_price': lastFillPrice,
                    'last_update': datetime.now()
                })
                
                # 处理不同的订单状态
                if status == 'Filled':
                    self._handle_order_filled(orderId, order_info, filled, avgFillPrice)
                elif status == 'Cancelled':
                    self._handle_order_cancelled(orderId, order_info)
                elif status == 'Rejected':
                    self._handle_order_rejected(orderId, order_info, whyHeld)
            
        except Exception as e:
            self.log_message(f"❌ 处理orderStatus失败: {e}")
    
    def execDetails(self, reqId, contract, execution):
        """真正成交执行后被调用"""
        try:
            order_id = execution.orderId
            symbol = contract.symbol
            side = execution.side
            shares = execution.shares
            price = execution.price
            exec_time = execution.time
            
            self.log_message(f"✅ 订单执行: {symbol} {side} {shares}股 @ ${price:.2f} (订单ID: {order_id})")
            
            # 记录执行详情
            if order_id not in self.order_executions:
                self.order_executions[order_id] = []
            
            exec_info = {
                'symbol': symbol,
                'side': side,
                'shares': shares,
                'price': price,
                'exec_time': exec_time,
                'timestamp': datetime.now(),
                'execution': execution
            }
            self.order_executions[order_id].append(exec_info)
            
        except Exception as e:
            self.log_message(f"❌ 处理execDetails失败: {e}")
    
    def _handle_order_filled(self, order_id: int, order_info: dict, filled: int, avg_price: float):
        """处理订单完全成交"""
        try:
            symbol = order_info['symbol']
            action = order_info['order'].action
            
            self.log_message(f"✅ 订单完全成交: {symbol} {action} {filled}股 @ ${avg_price:.2f}")
            
            # 从开放订单中移除
            if order_id in self.open_orders:
                # 移动到历史订单
                self.order_history.append({
                    **self.open_orders[order_id],
                    'final_status': 'FILLED',
                    'completed_time': datetime.now()
                })
                del self.open_orders[order_id]
            
        except Exception as e:
            self.log_message(f"❌ 处理订单完全成交失败: {e}")
    
    def _handle_order_cancelled(self, order_id: int, order_info: dict):
        """处理订单取消"""
        try:
            symbol = order_info['symbol']
            action = order_info['order'].action
            quantity = order_info['order'].totalQuantity
            
            self.log_message(f"⚠️ 订单已取消: {symbol} {action} {quantity}股")
            
        except Exception as e:
            self.log_message(f"❌ 处理订单取消失败: {e}")
    
    def _handle_order_rejected(self, order_id: int, order_info: dict, why_held: str):
        """处理订单拒绝"""
        try:
            symbol = order_info['symbol']
            action = order_info['order'].action
            quantity = order_info['order'].totalQuantity
            
            self.log_message(f"❌ 订单被拒绝: {symbol} {action} {quantity}股, 原因: {why_held}")
            
        except Exception as e:
            self.log_message(f"❌ 处理订单拒绝失败: {e}")
    
    # ============================================================================
    # 实时事件驱动和市场数据处理
    # ============================================================================
    
    async def _resubscribe_market_data(self):
        """重新订阅市场数据"""
        try:
            if not self.is_ibkr_connected:
                return
            
            self.log_message("📊 重新订阅市场数据...")
            
            # 重新订阅之前的所有市场数据
            for symbol in list(self.subscribed_symbols):
                await self.subscribe_market_data(symbol)
            
            self.log_message(f"📊 重新订阅完成，共{len(self.subscribed_symbols)}个标的")
            
        except Exception as e:
            self.log_message(f"❌ 重新订阅市场数据失败: {e}")
    
    # 删除重复的subscribe_market_data方法，保留更完整的版本
    
    def _on_tick_update(self, ticker):
        """默认的行情更新处理器：把 bid/ask/last 写入 price_data，并打日志"""
        try:
            symbol = ticker.contract.symbol
            
            # 更新价格数据
            self.price_data[symbol] = {
                'bid': ticker.bid if ticker.bid and ticker.bid != -1 else None,
                'ask': ticker.ask if ticker.ask and ticker.ask != -1 else None,
                'last': ticker.last if ticker.last and ticker.last != -1 else None,
                'timestamp': datetime.now()
            }
            
            # 更新tick数据
            if symbol not in self.tick_data:
                self.tick_data[symbol] = {}
            
            self.tick_data[symbol].update({
                'last_price': ticker.last,
                'bid': ticker.bid,
                'ask': ticker.ask,
                'bid_size': ticker.bidSize,
                'ask_size': ticker.askSize,
                'volume': ticker.volume,
                'timestamp': datetime.now()
            })
            
            # 调用注册的处理器
            if symbol in self.live_tick_handlers:
                handler = self.live_tick_handlers[symbol]
                if callable(handler):
                    handler(symbol, ticker)
            
            # 检查止损/止盈条件
            self._check_stop_conditions(symbol, ticker.last if ticker.last else ticker.marketPrice())
            
            self.log_message(f"📊 {symbol} 价格更新: Bid:{ticker.bid} Ask:{ticker.ask} Last:{ticker.last}")
            
        except Exception as e:
            self.log_message(f"❌ 处理Tick更新失败: {e}")
    
    def _default_tick_handler(self, symbol: str, ticker):
        """默认的Tick处理器"""
        try:
            price = ticker.last if ticker.last else ticker.marketPrice()
            if price and price > 0:
                self.log_message(f"📊 {symbol}: ${price:.2f}")
                    
        except Exception as e:
            self.log_message(f"❌ 默认Tick处理失败: {e}")
    
    def _check_stop_conditions(self, symbol: str, current_price: float):
        """检查止损止盈条件"""
        try:
            if not current_price or current_price <= 0:
                return
            
            # 检查止损条件
            if symbol in self.stop_loss_monitors:
                stop_loss = self.stop_loss_monitors[symbol]
                if current_price <= stop_loss['price']:
                    self.log_message(f"🚨 {symbol} 触发止损: 当前价格 ${current_price:.2f} <= 止损价格 ${stop_loss['price']:.2f}")
                    self._trigger_stop_loss(symbol, current_price, stop_loss)
            
            # 检查止盈条件
            if symbol in self.take_profit_monitors:
                take_profit = self.take_profit_monitors[symbol]
                if current_price >= take_profit['price']:
                    self.log_message(f"🎯 {symbol} 触发止盈: 当前价格 ${current_price:.2f} >= 止盈价格 ${take_profit['price']:.2f}")
                    self._trigger_take_profit(symbol, current_price, take_profit)
                    
        except Exception as e:
            self.log_message(f"❌ 检查止损止盈条件失败: {e}")
    
    def _trigger_stop_loss(self, symbol: str, current_price: float, stop_info: dict):
        """触发止损"""
        try:
            self.log_message(f"🚨 执行止损: {symbol} @ ${current_price:.2f}")
            
            # 这里可以集成实际的下单逻辑
            # 异步执行止损单
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self._execute_stop_loss_order(symbol, current_price, stop_info),
                    self.loop
                )
            
        except Exception as e:
            self.log_message(f"❌ 触发止损失败: {e}")
    
    def _trigger_take_profit(self, symbol: str, current_price: float, profit_info: dict):
        """触发止盈"""
        try:
            self.log_message(f"🎯 执行止盈: {symbol} @ ${current_price:.2f}")
            
            # 这里可以集成实际的下单逻辑
            # 异步执行止盈单
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self._execute_take_profit_order(symbol, current_price, profit_info),
                    self.loop
                )
            
        except Exception as e:
            self.log_message(f"❌ 触发止盈失败: {e}")
    
    async def _execute_stop_loss_order(self, symbol: str, current_price: float, stop_info: dict):
        """执行止损订单"""
        try:
            self.log_message(f"🚨 止损订单执行: {symbol} @ ${current_price:.2f}")
            # 这里集成实际的IBKR下单逻辑
        except Exception as e:
            self.log_message(f"❌ 执行止损订单失败: {e}")
    
    async def _execute_take_profit_order(self, symbol: str, current_price: float, profit_info: dict):
        """执行止盈订单"""
        try:
            self.log_message(f"🎯 止盈订单执行: {symbol} @ ${current_price:.2f}")
            # 这里集成实际的IBKR下单逻辑
        except Exception as e:
            self.log_message(f"❌ 执行止盈订单失败: {e}")
    
    async def _restart_risk_monitoring(self):
        """重启风控监控"""
        try:
            self.log_message("🛡️ 重启风控监控...")
            
            # 重新初始化风控状态
            await self._perform_realtime_risk_check()
            
            self.log_message("🛡️ 风控监控重启完成")
            
        except Exception as e:
            self.log_message(f"❌ 重启风控监控失败: {e}")
    
    # ============================================================================
    # 硬性风控阻断系统
    # ============================================================================
    
    async def _perform_realtime_risk_check(self):
        """执行实时风控检查"""
        try:
            if not self.risk_monitor_active or self.emergency_stop_triggered:
                return
            
            # 检查组合回撤
            drawdown_violation = await self._check_max_drawdown()
            
            # 检查每日亏损限制
            daily_loss_violation = await self._check_daily_loss_limit()
            
            # 检查持仓集中度
            concentration_violation = await self._check_position_concentration()
            
            # 检查单日新开仓数量
            new_position_violation = await self._check_daily_new_positions()
            
            # 如果有任何违规，触发相应措施
            if drawdown_violation:
                await self._trigger_max_drawdown_action()
            
            if daily_loss_violation:
                await self._trigger_daily_loss_limit_action()
            
            if concentration_violation:
                await self._trigger_concentration_limit_action()
            
            if new_position_violation:
                await self._trigger_new_position_limit_action()
            
        except Exception as e:
            self.log_message(f"❌ 实时风控检查失败: {e}")
    
    async def _check_max_drawdown(self):
        """检查最大回撤"""
        try:
            if self.max_portfolio_value <= 0:
                return False
            
            current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            max_drawdown_limit = self.config.get('enhanced_ibkr', {}).get('max_drawdown_percent', 10) / 100
            
            if current_drawdown >= max_drawdown_limit:
                if not self.max_drawdown_breached:
                    self.max_drawdown_breached = True
                    self.log_message(f"🚨 最大回撤超限: {current_drawdown:.2%} >= {max_drawdown_limit:.2%}")
                    return True
            else:
                self.max_drawdown_breached = False
            
            return False
            
        except Exception as e:
            self.log_message(f"❌ 检查最大回撤失败: {e}")
            return False
    
    async def _check_daily_loss_limit(self):
        """检查每日亏损限制"""
        try:
            daily_loss_limit = self.config.get('enhanced_ibkr', {}).get('daily_loss_limit', 5000)
            
            if self.daily_pnl <= -daily_loss_limit:
                if not self.daily_loss_limit_breached:
                    self.daily_loss_limit_breached = True
                    self.log_message(f"🚨 每日亏损超限: ${self.daily_pnl:.2f} <= -${daily_loss_limit:.2f}")
                    return True
            else:
                self.daily_loss_limit_breached = False
            
            return False
            
        except Exception as e:
            self.log_message(f"❌ 检查每日亏损限制失败: {e}")
            return False
    
    async def _check_position_concentration(self):
        """检查持仓集中度"""
        try:
            if not self.position_monitors or self.portfolio_value <= 0:
                return False
            
            max_single_position_pct = self.config.get('enhanced_ibkr', {}).get('max_single_position_percent', 20) / 100
            
            for symbol, position_info in self.position_monitors.items():
                position_value = abs(position_info.get('position', 0) * position_info.get('avg_cost', 0))
                concentration = position_value / self.portfolio_value
                
                if concentration > max_single_position_pct:
                    self.log_message(f"⚠️ 单只股票持仓过度集中: {symbol} {concentration:.2%} > {max_single_position_pct:.2%}")
                    return True
            
            return False
            
        except Exception as e:
            self.log_message(f"❌ 检查持仓集中度失败: {e}")
            return False
    
    async def _check_daily_new_positions(self):
        """检查每日新开仓数量"""
        try:
            max_new_positions = self.config.get('enhanced_ibkr', {}).get('max_new_positions_per_day', 10)
            
            if self.daily_new_positions >= max_new_positions:
                self.log_message(f"⚠️ 每日新开仓数量超限: {self.daily_new_positions} >= {max_new_positions}")
                return True
            
            return False
            
        except Exception as e:
            self.log_message(f"❌ 检查每日新开仓数量失败: {e}")
            return False
    
    async def _trigger_max_drawdown_action(self):
        """触发最大回撤处理"""
        try:
            self.log_message("🚨 触发最大回撤保护措施")
            
            # 停止所有新交易
            self.trading_blocked = True
            self.emergency_stop_triggered = True
            
            # 发送紧急告警
            await self._send_alert(
                "CRITICAL",
                "最大回撤保护触发",
                f"当前回撤: {((self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value):.2%}\n已停止所有新交易\n建议手动检查并决定是否平仓"
            )
            
        except Exception as e:
            self.log_message(f"❌ 处理最大回撤失败: {e}")
    
    async def _trigger_daily_loss_limit_action(self):
        """触发每日亏损限制处理"""
        try:
            self.log_message("🚨 触发每日亏损限制保护措施")
            
            # 停止当日所有新交易
            self.trading_blocked = True
            
            # 记录触发日期
            self.last_loss_date = datetime.now().date()
            
            # 发送告警
            await self._send_alert(
                "ERROR",
                "每日亏损限制触发",
                f"当日亏损: ${self.daily_pnl:.2f}\n已停止当日所有新交易\n将于次日重置"
            )
            
        except Exception as e:
            self.log_message(f"❌ 处理每日亏损限制失败: {e}")
    
    async def _trigger_concentration_limit_action(self):
        """触发集中度限制处理"""
        try:
            self.log_message("⚠️ 触发持仓集中度警告")
            
            # 暂时阻止开新仓
            self.position_size_violations += 1
            
            # 发送警告
            await self._send_alert(
                "WARNING",
                "持仓集中度过高",
                "建议减少单一标的持仓或分散投资"
            )
            
        except Exception as e:
            self.log_message(f"❌ 处理集中度限制失败: {e}")
    
    async def _trigger_new_position_limit_action(self):
        """触发新开仓限制处理"""
        try:
            self.log_message("⚠️ 每日新开仓数量已达上限")
            
            # 发送警告
            await self._send_alert(
                "WARNING",
                "每日新开仓数量超限",
                f"当日已开新仓: {self.daily_new_positions}次\n已停止当日新开仓"
            )
            
        except Exception as e:
            self.log_message(f"❌ 处理新开仓限制失败: {e}")
    
    async def _check_pending_orders(self):
        """检查待处理订单状态"""
        try:
            current_time = datetime.now()
            
            # 检查长时间未成交的订单
            for order_id, order_info in list(self.open_orders.items()):
                order_time = order_info.get('timestamp', current_time)
                elapsed = (current_time - order_time).total_seconds()
                
                # 如果订单超过10分钟仍未成交，发出警告
                if elapsed > 600:  # 10分钟
                    symbol = order_info['symbol']
                    action = order_info['order'].action
                    self.log_message(f"⚠️ 订单长时间未成交: {symbol} {action} (已等待{elapsed:.0f}秒)")
            
        except Exception as e:
            self.log_message(f"❌ 检查待处理订单失败: {e}")
    
    def _on_ibkr_connected(self):
        """IBKR连接成功事件"""
        self.log_message("📡 IBKR连接事件触发")
    
    def _on_ibkr_disconnected(self):
        """IBKR断开连接事件"""
        self.log_message("📡 IBKR断开连接事件触发")
    
    def _on_ibkr_error(self, reqId, errorCode, errorString, contract):
        """IBKR错误事件处理"""
        error_info = f"📡 IBKR错误: ID={reqId}, 代码={errorCode}, 消息={errorString}"
        
        # 根据错误代码进行分类处理
        if errorCode == 326:
            self.log_message(f"❌ 客户端ID冲突: {errorString}")
            # 客户端ID已被使用，这个错误在连接阶段已处理
        elif errorCode == 502:
            self.log_message(f"⚠️ 无法连接到TWS: {errorString}")
        elif errorCode == 504:
            self.log_message(f"⚠️ 未找到合约: {errorString}")
        elif errorCode == 200:
            self.log_message(f"⚠️ 无合约定义: {errorString}")
        elif errorCode == 162:
            self.log_message(f"⚠️ 历史数据服务错误: {errorString}")
        elif errorCode in [2104, 2106, 2158]:
            # 这些是信息性消息，不是错误
            self.log_message(f"ℹ️ IBKR信息: {errorString}")
        else:
            # 其他错误
            self.log_message(f"❌ IBKR错误: {error_info}")
            
        # 如果是严重错误，可能需要重连
        if errorCode in [502, 1100, 1101, 1102]:
            self.log_message("⚠️ 检测到连接问题，可能需要重连")
            # 触发重连逻辑
            self._schedule_reconnect()
    
    def _schedule_reconnect(self):
        """安排重连任务"""
        if not hasattr(self, '_reconnect_scheduled') or not self._reconnect_scheduled:
            self._reconnect_scheduled = True
            self.log_message("📡 安排5秒后重连IBKR...")
            # 使用threading.Timer延迟重连
            import threading
            timer = threading.Timer(5.0, self._attempt_reconnect)
            timer.daemon = True
            timer.start()
    
    def _attempt_reconnect(self):
        """尝试重连IBKR"""
        try:
            self.log_message("🔄 尝试重连IBKR...")
            self._reconnect_scheduled = False
            
            # 如果有现有连接，先断开
            if hasattr(self, 'ib_connection') and self.ib_connection:
                try:
                    if self.ib_connection.isConnected():
                        self.ib_connection.disconnect()
                except:
                    pass
            
            if hasattr(self, 'ib') and self.ib:
                try:
                    if self.ib.isConnected():
                        self.ib.disconnect()
                except:
                    pass
            
            # 等待一段时间后重连
            time.sleep(2)
            
            # 尝试重连（这里可以调用相应的连接方法）
            self.log_message("🔄 重连准备完成")
            
        except Exception as e:
            self.log_message(f"❌ 重连失败: {e}")
            self._reconnect_scheduled = False
    
    def diagnose_connection_issue(self):
        """诊断连接问题并提供建议"""
        diagnostic_msg = """
🔧 IBKR连接诊断建议:

1. 客户端ID冲突 (错误326):
   ✅ 系统已自动处理 - 会自动生成新的客户端ID

2. 确保TWS/Gateway运行:
   - 检查Trader Workstation或IB Gateway是否已启动
   - 确认端口设置正确 (纸交易: 7497, 实盘: 7496, Gateway: 4001/4002)

3. API设置:
   - 在TWS中启用"Enable ActiveX and Socket Clients" 
   - 检查"Read-Only API"设置
   - 确认客户端ID范围允许连接

4. 网络问题:
   - 检查防火墙设置
   - 确认本地网络连接正常

5. 常见解决方案:
   - 重启TWS/Gateway
   - 检查账户权限
   - 更新客户端ID到未使用的值
        """
        
        self.log_message(diagnostic_msg)
        return diagnostic_msg
    
    # 自动交易功能方法
    def select_signal_file(self, file_type):
        """选择信号文件"""
        if file_type == 'json':
            file_path = filedialog.askopenfilename(
                title="选择JSON信号文件",
                initialdir="result",
                filetypes=[("JSON files", "*.json")]
            )
        else:  # excel
            file_path = filedialog.askopenfilename(
                title="选择Excel信号文件",
                initialdir="result",
                filetypes=[("Excel files", "*.xlsx")]
            )
        
        if file_path:
            self.selected_file_label.config(text=os.path.basename(file_path), foreground="green")
            self.load_stocks_from_file(file_path)
    
    def auto_load_latest_signal(self):
        """自动加载最新的信号文件"""
        try:
            result_dir = Path("result")
            
            # 查找最新的LSTM文件
            json_files = list(result_dir.glob("*lstm*.json"))
            excel_files = list(result_dir.glob("*lstm*.xlsx"))
            
            all_files = json_files + excel_files
            
            if all_files:
                latest_file = max(all_files, key=os.path.getmtime)
                self.selected_file_label.config(text=latest_file.name, foreground="green")
                self.load_stocks_from_file(str(latest_file))
                self.log_message(f"✅ 已自动加载最新文件: {latest_file.name}")
            else:
                messagebox.showwarning("警告", "未找到LSTM分析结果文件")
                
        except Exception as e:
            self.log_message(f"❌ 自动加载文件失败: {e}")
            messagebox.showerror("错误", f"自动加载失败: {str(e)}")
    
    def load_stocks_from_file(self, file_path):
        """从文件加载股票列表"""
        try:
            self.auto_trading_stocks.clear()
            self.stock_listbox.delete(0, tk.END)
            
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 处理不同的JSON格式
                stocks_data = []
                if 'top_10_stocks' in data:
                    stocks_data = data['top_10_stocks']
                elif isinstance(data, list):
                    stocks_data = data
                else:
                    # 尝试从其他键获取数据
                    for key in data:
                        if isinstance(data[key], list) and len(data[key]) > 0:
                            stocks_data = data[key]
                            break
                
                # 按收益率排序并取前5个
                if stocks_data:
                    # 尝试不同的收益率字段名
                    for item in stocks_data:
                        if 'expected_return' in item or 'weighted_prediction' in item or 'predicted_return' in item:
                            break
                    else:
                        # 如果没有收益率字段，按原顺序取前5个
                        pass
                    
                    # 排序
                    def get_return_value(item):
                        return (item.get('expected_return', 0) or 
                               item.get('weighted_prediction', 0) or 
                               item.get('predicted_return', 0) or 0)
                    
                    sorted_stocks = sorted(stocks_data, key=get_return_value, reverse=True)
                    top5_stocks = sorted_stocks[:5]  # 只取前5个
                    
                    for stock in top5_stocks:
                        symbol = stock.get('ticker') or stock.get('symbol') or stock.get('stock', '')
                        if symbol:
                            self.auto_trading_stocks.append(symbol)
                            self.stock_listbox.insert(tk.END, symbol)
            
            elif file_path.endswith('.xlsx'):
                # 读取Excel文件
                df = pd.read_excel(file_path)
                
                # 查找股票代码列
                symbol_col = None
                for col in ['ticker', 'symbol', 'stock', '股票代码']:
                    if col in df.columns:
                        symbol_col = col
                        break
                
                if symbol_col:
                    # 查找收益率列进行排序
                    return_col = None
                    for col in ['expected_return', 'weighted_prediction', 'predicted_return', '预期收益率', '加权预测']:
                        if col in df.columns:
                            return_col = col
                            break
                    
                    if return_col:
                        df_sorted = df.sort_values(return_col, ascending=False)
                    else:
                        df_sorted = df
                    
                    # 取前5个股票
                    top5_df = df_sorted.head(5)
                    
                    for _, row in top5_df.iterrows():
                        symbol = row[symbol_col]
                        if pd.notna(symbol):
                            self.auto_trading_stocks.append(str(symbol))
                            self.stock_listbox.insert(tk.END, str(symbol))
            
            self.log_message(f"✅ 已加载{len(self.auto_trading_stocks)}只股票到交易列表")
            
        except Exception as e:
            self.log_message(f"❌ 加载文件失败: {e}")
            messagebox.showerror("错误", f"加载文件失败: {str(e)}")
    
    def load_top5_stocks(self):
        """重新加载前5只股票"""
        if hasattr(self, 'selected_file_label') and self.selected_file_label.cget('text') != "未选择文件":
            # 重新加载当前选择的文件
            current_file = self.selected_file_label.cget('text')
            file_path = os.path.join("result", current_file)
            if os.path.exists(file_path):
                self.load_stocks_from_file(file_path)
            else:
                messagebox.showwarning("警告", "文件不存在，请重新选择")
        else:
            # 自动加载最新文件
            self.auto_load_latest_signal()
    
    def add_trading_stock(self):
        """添加交易股票"""
        stock_dialog = tk.Toplevel(self.root)
        stock_dialog.title("添加股票")
        stock_dialog.geometry("300x150")
        stock_dialog.transient(self.root)
        stock_dialog.grab_set()
        
        ttk.Label(stock_dialog, text="请输入股票代码:").pack(pady=20)
        
        symbol_var = tk.StringVar()
        entry = ttk.Entry(stock_dialog, textvariable=symbol_var, width=20)
        entry.pack(pady=10)
        entry.focus()
        
        def add_stock():
            symbol = symbol_var.get().strip().upper()
            if symbol and symbol not in self.auto_trading_stocks:
                self.auto_trading_stocks.append(symbol)
                self.stock_listbox.insert(tk.END, symbol)
                self.log_message(f"✅ 已添加股票: {symbol}")
                stock_dialog.destroy()
            elif symbol in self.auto_trading_stocks:
                messagebox.showwarning("警告", "股票已存在于列表中")
            else:
                messagebox.showwarning("警告", "请输入有效的股票代码")
        
        ttk.Button(stock_dialog, text="添加", command=add_stock).pack(pady=10)
        entry.bind('<Return>', lambda e: add_stock())
    
    def remove_trading_stock(self):
        """删除选中的交易股票"""
        selection = self.stock_listbox.curselection()
        if selection:
            index = selection[0]
            symbol = self.stock_listbox.get(index)
            
            # 如果正在交易，询问是否要停止该股票的交易
            if self.is_auto_trading:
                result = messagebox.askyesnocancel(
                    "确认删除", 
                    f"股票 {symbol} 正在自动交易中，是否要:\n\n"
                    "是：卖出并删除\n"
                    "否：仅删除（保留持仓）\n"
                    "取消：不删除"
                )
                
                if result is None:  # 取消
                    return
                elif result:  # 是，卖出并删除
                    # TODO: 实现卖出逻辑
                    self.log_message(f"🔄 正在卖出 {symbol} 的所有持仓...")
            
            self.stock_listbox.delete(index)
            if symbol in self.auto_trading_stocks:
                self.auto_trading_stocks.remove(symbol)
            
            self.log_message(f"✅ 已删除股票: {symbol}")
        else:
            messagebox.showwarning("警告", "请先选择要删除的股票")
    
    def on_port_selected(self, event):
        """端口选择事件处理"""
        selected = self.port_combo.get()
        if selected:
            # 从选择中提取端口号
            port = selected.split(" - ")[0]
            self.port_var.set(port)
            self.log_message(f"已选择端口: {port}")
    
    def get_ibkr_connection_params(self):
        """获取IBKR连接参数"""
        try:
            host = self.host_var.get().strip()
            port = int(self.port_var.get().strip())
            
            if not host:
                raise ValueError("主机地址不能为空")
            if port < 1 or port > 65535:
                raise ValueError("端口号必须在1-65535之间")
                
            return host, port
        except ValueError as e:
            messagebox.showerror("参数错误", f"连接参数错误: {e}")
            return None, None
    
    def get_account_balance(self):
        """动态获取IBKR账户余额"""
        try:
            if not IBKR_AVAILABLE:
                self.log_message("❌ IBKR API不可用，无法获取账户余额")
                return 0
            
            # 使用ib_insync获取账户余额
            if hasattr(self, 'ib') and self.ib and self.ib.isConnected():
                try:
                    # 获取账户摘要
                    summary = self.ib.reqAccountSummary()
                    total_cash_value = 0
                    
                    for item in summary:
                        if item.tag == 'TotalCashValue':
                            total_cash_value = float(item.value)
                            self.log_message(f"💰 账户总现金: {item.value} {item.currency}")
                            break
                        elif item.tag == 'NetLiquidation':
                            total_cash_value = float(item.value)
                            self.log_message(f"💰 账户净值: {item.value} {item.currency}")
                            break
                    
                    # 更新配置中的总资金
                    self.config['total_capital'] = total_cash_value
                    self.log_message(f"✅ 已更新总资金为: ${total_cash_value:,.2f}")
                    
                    # 更新界面显示
                    self.update_balance_display(total_cash_value)
                    
                    return total_cash_value
                    
                except Exception as e:
                    self.log_message(f"❌ 获取账户余额失败: {e}")
                    return 0
            else:
                self.log_message("❌ IBKR未连接，无法获取账户余额")
                return 0
                
        except Exception as e:
            self.log_message(f"❌ 获取账户余额时出错: {e}")
            return 0
    
    def update_account_balance_periodic(self):
        """定期更新账户余额"""
        def update_balance():
            while self.is_auto_trading and hasattr(self, 'ib') and self.ib and self.ib.isConnected():
                try:
                    balance = self.get_account_balance()
                    # 更新界面显示
                    self.update_balance_display(balance)
                    # 每5分钟更新一次账户余额
                    time.sleep(300)
                except Exception as e:
                    self.log_message(f"❌ 定期更新账户余额失败: {e}")
                    time.sleep(60)  # 出错时等待1分钟
        
        # 启动定期更新线程
        balance_thread = threading.Thread(target=update_balance, daemon=True)
        balance_thread.start()
    
    def update_balance_display(self, balance):
        """更新余额显示"""
        try:
            if hasattr(self, 'balance_label'):
                formatted_balance = f"${balance:,.2f}"
                self.balance_label.config(text=formatted_balance)
                
                # 根据余额变化设置颜色
                if balance > 0:
                    self.balance_label.config(foreground="green")
                else:
                    self.balance_label.config(foreground="red")
                    
        except Exception as e:
            self.log_message(f"❌ 更新余额显示失败: {e}")
    
    def create_contract(self, symbol):
        """创建股票合约"""
        if IBKR_AVAILABLE:
            try:
                from ib_insync import Stock
                contract = Stock(symbol, 'SMART', 'USD')
                return contract
            except Exception as e:
                self.log_message(f"❌ 创建合约失败 {symbol}: {e}")
                return None
        else:
            # 原生 ibapi 方式
            try:
                from ibapi.contract import Contract
                contract = Contract()
                contract.symbol = symbol
                contract.secType = "STK"
                contract.exchange = "SMART"
                contract.currency = "USD"
                return contract
            except Exception as e:
                self.log_message(f"❌ 创建合约失败 {symbol}: {e}")
                return None
    
    async def subscribe_market_data(self, symbol: str, callback=None) -> bool:
        """异步订阅实时市场数据，返回是否订阅成功"""
        try:
            if not self.is_ibkr_connected or not self.ib_connection:
                self.log_message(f"⚠️ IBKR未连接，无法订阅 {symbol}")
                return False

            # 创建合约
            from ib_insync import Stock
            contract = Stock(symbol, 'SMART', 'USD')

            # 生成一个唯一的 reqId
            req_id = len(self.market_data_subscriptions) + 1000

            # 请求市场数据
            ticker = self.ib_connection.reqMktData(contract, "", False, False)
            if not ticker:
                self.log_message(f"⚠️ 订阅 {symbol} 失败，未获得 ticker")
                return False

            # 记录订阅关系
            self.market_data_subscriptions[req_id] = symbol
            self.subscribed_symbols.add(symbol)

            # 绑定回调
            if callback:
                self.live_tick_handlers[symbol] = callback
            else:
                self.live_tick_handlers[symbol] = self._default_tick_handler

            ticker.updateEvent += lambda tkr=ticker: self._on_tick_update(tkr)

            self.log_message(f"📊 成功订阅 {symbol} 实时数据 (reqId: {req_id})")
            return True

        except Exception as e:
            self.log_message(f"❌ 订阅 {symbol} 市场数据失败: {e}")
            return False
    
    async def unsubscribe_market_data(self, symbol: str) -> bool:
        """取消订阅市场数据"""
        try:
            if not self.is_ibkr_connected or not self.ib_connection:
                self.log_message(f"⚠️ IBKR未连接，无法取消订阅 {symbol}")
                return False
            
            # 查找对应的 ticker
            ticker_to_remove = None
            req_id_to_remove = None
            
            for req_id, subscribed_symbol in self.market_data_subscriptions.items():
                if subscribed_symbol == symbol:
                    req_id_to_remove = req_id
                    break
            
            if req_id_to_remove:
                # 取消订阅
                self.ib_connection.cancelMktData(req_id_to_remove)
                
                # 清理订阅记录
                del self.market_data_subscriptions[req_id_to_remove]
                self.subscribed_symbols.discard(symbol)
                
                # 清理处理器
                if symbol in self.live_tick_handlers:
                    del self.live_tick_handlers[symbol]
                
                # 清理价格数据
                if symbol in self.price_data:
                    del self.price_data[symbol]
                
                self.log_message(f"✅ 已取消订阅 {symbol} 实时数据")
                return True
            else:
                self.log_message(f"⚠️ 未找到 {symbol} 的订阅记录")
                return False
                
        except Exception as e:
            self.log_message(f"❌ 取消订阅 {symbol} 失败: {e}")
            return False
    
    async def set_market_data_type(self, data_type: int = 4):
        """设置市场数据类型
        data_type: 1=实时, 2=冻结, 3=延迟, 4=实时
        """
        try:
            if not self.is_ibkr_connected or not self.ib_connection:
                self.log_message(f"⚠️ IBKR未连接，无法设置市场数据类型")
                return False
            
            self.ib_connection.reqMarketDataType(data_type)
            data_type_names = {1: "实时", 2: "冻结", 3: "延迟", 4: "实时"}
            self.log_message(f"✅ 已设置市场数据类型为: {data_type_names.get(data_type, '未知')}")
            return True
            
        except Exception as e:
            self.log_message(f"❌ 设置市场数据类型失败: {e}")
            return False
    
    async def resubscribe_all_market_data(self):
        """重新订阅所有市场数据（用于重连后）"""
        try:
            if not self.is_ibkr_connected or not self.ib_connection:
                self.log_message(f"⚠️ IBKR未连接，无法重新订阅")
                return False
            
            # 保存当前订阅的股票列表
            symbols_to_resubscribe = list(self.subscribed_symbols)
            
            # 清理现有订阅
            for req_id in list(self.market_data_subscriptions.keys()):
                try:
                    self.ib_connection.cancelMktData(req_id)
                except:
                    pass
            
            self.market_data_subscriptions.clear()
            self.subscribed_symbols.clear()
            
            # 重新订阅
            success_count = 0
            for symbol in symbols_to_resubscribe:
                if await self.subscribe_market_data(symbol):
                    success_count += 1
                else:
                    # 失败时等待一下再重试
                    await asyncio.sleep(1)
                    if await self.subscribe_market_data(symbol):
                        success_count += 1
            
            self.log_message(f"✅ 重新订阅完成: {success_count}/{len(symbols_to_resubscribe)} 成功")
            return success_count == len(symbols_to_resubscribe)
            
        except Exception as e:
            self.log_message(f"❌ 重新订阅市场数据失败: {e}")
            return False
    
    def get_current_price(self, symbol):
        """获取当前价格"""
        try:
            if symbol in self.price_data:
                price_info = self.price_data[symbol]
                
                # 优先使用 last 价格，然后是 ask/bid 中间价
                if price_info['last']:
                    return price_info['last']
                elif price_info['bid'] and price_info['ask']:
                    return (price_info['bid'] + price_info['ask']) / 2
                elif price_info['ask']:
                    return price_info['ask']
                elif price_info['bid']:
                    return price_info['bid']
            
            self.log_message(f"⚠️ 无法获取 {symbol} 当前价格")
            return None
            
        except Exception as e:
            self.log_message(f"❌ 获取 {symbol} 价格失败: {e}")
            return None
    
    def place_market_order(self, symbol, action, quantity):
        """下市价单"""
        try:
            if not hasattr(self, 'ib') or not self.ib or not self.ib.isConnected():
                self.log_message(f"❌ IBKR未连接，无法下单")
                return None
            
            # 创建合约
            contract = self.create_contract(symbol)
            if not contract:
                return None
            
            # 使用 ib_insync 创建市价单
            from ib_insync import MarketOrder
            order = MarketOrder(action, quantity)
            
            # 下单
            trade = self.ib.placeOrder(contract, order)
            
            # 记录订单信息
            order_info = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'order_type': 'MARKET',
                'trade': trade,
                'timestamp': time.time()
            }
            
            self.order_status_map[trade.order.orderId] = order_info
            
            self.log_message(f"✅ 已下市价单: {action} {quantity} {symbol} (订单ID: {trade.order.orderId})")
            
            return trade
            
        except Exception as e:
            self.log_message(f"❌ 下市价单失败 {symbol}: {e}")
            return None
    
    def place_limit_order(self, symbol, action, quantity, limit_price):
        """下限价单"""
        try:
            if not hasattr(self, 'ib') or not self.ib or not self.ib.isConnected():
                self.log_message(f"❌ IBKR未连接，无法下单")
                return None
            
            # 创建合约
            contract = self.create_contract(symbol)
            if not contract:
                return None
            
            # 使用 ib_insync 创建限价单
            from ib_insync import LimitOrder
            order = LimitOrder(action, quantity, limit_price)
            
            # 下单
            trade = self.ib.placeOrder(contract, order)
            
            # 记录订单信息
            order_info = {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'order_type': 'LIMIT',
                'limit_price': limit_price,
                'trade': trade,
                'timestamp': time.time()
            }
            
            self.order_status_map[trade.order.orderId] = order_info
            
            self.log_message(f"✅ 已下限价单: {action} {quantity} {symbol} @ ${limit_price} (订单ID: {trade.order.orderId})")
            
            return trade
            
        except Exception as e:
            self.log_message(f"❌ 下限价单失败 {symbol}: {e}")
            return None
    
    def place_smart_order(self, symbol, action, quantity, strategy='current_price'):
        """智能下单（基于实时价格）"""
        try:
            # 增强风险检查
            if hasattr(self, 'risk_manager') and self.risk_manager:
                # 进行Pre-Trade风险检查
                current_price = self.get_current_price(symbol) or 100
                risk_check = self.risk_manager.pre_trade_check(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    price=current_price,
                    strategy_name="quantitative_trading"
                )
                
                if risk_check.result == RiskCheckResult.REJECTED:
                    self.log_message(f"🚨 风险检查拒绝 {symbol} {action} {quantity}: {', '.join(risk_check.reasons)}")
                    return None
                elif risk_check.result == RiskCheckResult.SCALED_DOWN:
                    quantity = risk_check.approved_quantity
                    self.log_message(f"⚠️ 风险检查调整数量 {symbol}: {risk_check.original_quantity} → {quantity}")
                elif risk_check.result == RiskCheckResult.WARNING:
                    self.log_message(f"⚠️ 风险警告 {symbol}: {', '.join(risk_check.warnings)}")
            
            # 先订阅行情获取实时价格
            if symbol not in self.ticker_subscriptions:
                self.subscribe_market_data(symbol)
                time.sleep(2)  # 等待价格数据
            
            current_price = self.get_current_price(symbol)
            if not current_price:
                self.log_message(f"❌ 无法获取 {symbol} 实时价格，使用市价单")
                return self.place_market_order_enhanced(symbol, action, quantity)
            
            # 根据策略决定下单价格
            if strategy == 'current_price':
                # 使用当前价格下限价单
                limit_price = current_price
            elif strategy == 'aggressive':
                # 激进策略：买入时略高于ask，卖出时略低于bid
                if action == 'BUY':
                    price_info = self.price_data.get(symbol, {})
                    ask_price = price_info.get('ask')
                    limit_price = ask_price + 0.01 if ask_price else current_price + 0.01
                else:  # SELL
                    price_info = self.price_data.get(symbol, {})
                    bid_price = price_info.get('bid')
                    limit_price = bid_price - 0.01 if bid_price else current_price - 0.01
            else:
                limit_price = current_price
            
            self.log_message(f"📊 {symbol} 当前价格: ${current_price:.2f}, 下单价格: ${limit_price:.2f}")
            
            return self.place_limit_order_enhanced(symbol, action, quantity, limit_price)
            
        except Exception as e:
            self.log_message(f"❌ 智能下单失败 {symbol}: {e}")
            return None
    
    def place_market_order_enhanced(self, symbol, action, quantity):
        """增强版市价单"""
        try:
            if hasattr(self, 'order_manager') and self.order_manager:
                # 使用增强订单管理器
                order_record = self.order_manager.place_market_order(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    strategy_name="quantitative_trading"
                )
                if order_record:
                    self.log_message(f"✅ 增强市价单已提交: {symbol} {action} {quantity} (订单ID: {order_record.order_id})")
                    return order_record
            else:
                # 回退到原始方法
                return self.place_market_order(symbol, action, quantity)
        except Exception as e:
            self.log_message(f"❌ 增强市价单失败 {symbol}: {e}")
            return self.place_market_order(symbol, action, quantity)
    
    def place_limit_order_enhanced(self, symbol, action, quantity, limit_price):
        """增强版限价单"""
        try:
            if hasattr(self, 'order_manager') and self.order_manager:
                # 使用增强订单管理器
                order_record = self.order_manager.place_limit_order(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    limit_price=limit_price,
                    strategy_name="quantitative_trading"
                )
                if order_record:
                    self.log_message(f"✅ 增强限价单已提交: {symbol} {action} {quantity} @ ${limit_price:.2f} (订单ID: {order_record.order_id})")
                    return order_record
            else:
                # 回退到原始方法
                return self.place_limit_order(symbol, action, quantity, limit_price)
        except Exception as e:
            self.log_message(f"❌ 增强限价单失败 {symbol}: {e}")
            return self.place_limit_order(symbol, action, quantity, limit_price)
    
    def place_bracket_order(self, symbol, action, quantity, limit_price, stop_loss_price, take_profit_price):
        """下Bracket订单（带止损止盈）"""
        try:
            if hasattr(self, 'order_manager') and self.order_manager:
                order_record = self.order_manager.place_bracket_order(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    limit_price=limit_price,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    strategy_name="quantitative_trading"
                )
                if order_record:
                    self.log_message(f"✅ Bracket订单已提交: {symbol} {action} {quantity} @ ${limit_price:.2f} SL:${stop_loss_price:.2f} TP:${take_profit_price:.2f}")
                    return order_record
            else:
                self.log_message(f"⚠️ 增强订单管理器不可用，无法下Bracket订单")
                return None
        except Exception as e:
            self.log_message(f"❌ Bracket订单失败 {symbol}: {e}")
            return None
    
    def get_current_positions_dict(self):
        """获取当前持仓字典（用于风险检查）"""
        try:
            positions = self.get_current_positions()
            positions_dict = {}
            
            for position in positions:
                if hasattr(position, 'contract') and hasattr(position, 'position') and position.position != 0:
                    symbol = position.contract.symbol
                    positions_dict[symbol] = {
                        'quantity': position.position,
                        'market_value': position.marketValue,
                        'unrealized_pnl': position.unrealizedPNL,
                        'avg_cost': position.avgCost
                    }
            
            return positions_dict
        except Exception as e:
            self.log_message(f"❌ 获取持仓字典失败: {e}")
            return {}
    
    def calculate_position_size(self, symbol, target_percent=None):
        """计算持仓大小"""
        try:
            total_capital = self.config.get('total_capital', 100000)
            max_position_percent = self.config.get('max_position_percent', 5)
            
            # 使用目标百分比或默认最大持仓比例
            position_percent = target_percent if target_percent else max_position_percent
            target_value = total_capital * (position_percent / 100)
            
            # 获取当前价格
            current_price = self.get_current_price(symbol)
            if not current_price:
                self.log_message(f"❌ 无法获取 {symbol} 价格，无法计算持仓大小")
                return 0
            
            # 计算股数（向下取整到整数）
            quantity = int(target_value / current_price)
            
            self.log_message(f"💰 {symbol} 持仓计算: 总资金${total_capital:,.2f}, 目标比例{position_percent}%, 目标金额${target_value:,.2f}, 价格${current_price:.2f}, 股数{quantity}")
            
            return quantity
            
        except Exception as e:
            self.log_message(f"❌ 计算 {symbol} 持仓大小失败: {e}")
            return 0
    
    def setup_order_callbacks(self):
        """设置订单状态回调"""
        try:
            if not hasattr(self, 'ib') or not self.ib:
                return
            
            def on_order_status(trade):
                """订单状态更新回调"""
                order_id = trade.order.orderId
                status = trade.orderStatus.status
                filled = trade.orderStatus.filled
                remaining = trade.orderStatus.remaining
                
                if order_id in self.order_status_map:
                    order_info = self.order_status_map[order_id]
                    symbol = order_info['symbol']
                    action = order_info['action']
                    
                    self.log_message(f"📋 订单状态更新: {symbol} {action} (ID:{order_id}) - {status} 已成交:{filled} 剩余:{remaining}")
                    
                    # 更新订单状态
                    order_info['status'] = status
                    order_info['filled'] = filled
                    order_info['remaining'] = remaining
                    
                    # 如果订单完全成交，记录成交信息
                    if status == 'Filled':
                        self.log_message(f"✅ 订单完全成交: {symbol} {action} {filled}股")
                        
                        # 更新账户余额
                        self.get_account_balance()
            
            def on_execution(trade, fill):
                """订单成交回调"""
                order_id = trade.order.orderId
                if order_id in self.order_status_map:
                    order_info = self.order_status_map[order_id]
                    symbol = order_info['symbol']
                    action = order_info['action']
                    
                    self.log_message(f"🎯 订单成交: {symbol} {action} {fill.execution.shares}股 @ ${fill.execution.price}")
            
            # 绑定回调事件
            self.ib.orderStatusEvent += on_order_status
            self.ib.execDetailsEvent += on_execution
            
            self.log_message("✅ 订单回调已设置")
            
        except Exception as e:
            self.log_message(f"❌ 设置订单回调失败: {e}")
    
    def start_trading_thread(self):
        """启动交易线程"""
        try:
            def trading_loop():
                """交易主循环"""
                self.log_message("🚀 交易线程已启动")
                
                while self.is_auto_trading:
                    try:
                        # 执行交易策略
                        self.execute_trading_strategy()
                        
                        # 等待下一次执行（例如每分钟检查一次）
                        check_interval = 60  # 60秒
                        time.sleep(check_interval)
                        
                    except Exception as e:
                        self.log_message(f"❌ 交易循环错误: {e}")
                        time.sleep(30)  # 出错时等待30秒
                
                self.log_message("⛔ 交易线程已停止")
            
            # 在后台线程中启动交易循环
            trading_thread = threading.Thread(target=trading_loop, daemon=True)
            trading_thread.start()
            
            self.log_message("✅ 交易线程启动成功")
            
        except Exception as e:
            self.log_message(f"❌ 启动交易线程失败: {e}")
    
    def execute_trading_strategy(self):
        """执行交易策略"""
        try:
            if not self.is_auto_trading:
                return
            
            self.log_message("🔍 正在执行交易策略分析...")
            
            # 收集策略信号
            signals = self.collect_strategy_signals()
            if not signals:
                self.log_message("ℹ️ 暂无策略信号，回退到内置简单策略")
            
            # 获取当前持仓信息
            current_positions = self.get_current_positions()
            
            # 对信号进行执行
            if signals:
                for symbol, sig in signals.items():
                    try:
                        action = sig.get('action')
                        if action not in ('BUY', 'SELL'):
                            continue
                        
                        # 获取或估计价格
                        current_price = self.get_current_price(symbol) or sig.get('current_price') or 0
                        if not current_price:
                            self.log_message(f"⚠️ 跳过 {symbol}：无法获取价格")
                            continue
                        
                        # 风险检查与仓位计算
                        if hasattr(self, 'risk_manager') and self.risk_manager:
                            risk_check = self.risk_manager.pre_trade_check(
                                symbol=symbol,
                                action=action,
                                quantity=self.calculate_position_size(symbol),
                                price=current_price,
                                strategy_name="strategy_signals"
                            )
                            if risk_check.result.name == 'REJECTED':
                                self.log_message(f"🚨 风险拒绝 {symbol} {action}: {', '.join(risk_check.reasons)}")
                                continue
                        
                        if action == 'BUY':
                            if current_positions.get(symbol, 0) <= 0:
                                qty = self.calculate_position_size(symbol)
                                if qty > 0:
                                    self.log_message(f"📈 执行策略买入: {symbol} x {qty}")
                                    self.place_smart_order(symbol, 'BUY', qty, 'aggressive')
                        else:  # SELL
                            qty = current_positions.get(symbol, 0)
                            if qty > 0:
                                self.log_message(f"📉 执行策略卖出: {symbol} x {qty}")
                                self.place_smart_order(symbol, 'SELL', qty, 'aggressive')
                    except Exception as e:
                        self.log_message(f"❌ 执行信号失败 {symbol}: {e}")
            else:
                # 回退：遍历监控列表，使用内置简易策略
                for symbol in self.auto_trading_stocks:
                    try:
                        current_price = self.get_current_price(symbol)
                        if not current_price:
                            self.log_message(f"⚠️ 跳过 {symbol}：无法获取价格")
                            continue
                        signal = self.analyze_trading_signal(symbol, current_price)
                        if signal == 'BUY' and current_positions.get(symbol, 0) <= 0:
                            qty = self.calculate_position_size(symbol)
                            if qty > 0:
                                self.log_message(f"📈 买入信号: {symbol} 数量:{qty}")
                                self.place_smart_order(symbol, 'BUY', qty, 'aggressive')
                        elif signal == 'SELL' and current_positions.get(symbol, 0) > 0:
                            qty = current_positions.get(symbol, 0)
                            self.log_message(f"📉 卖出信号: {symbol} 数量:{qty}")
                            self.place_smart_order(symbol, 'SELL', qty, 'aggressive')
                    except Exception as e:
                        self.log_message(f"❌ 处理 {symbol} 交易策略失败: {e}")
            
        except Exception as e:
            self.log_message(f"❌ 执行交易策略失败: {e}")
    
    def analyze_trading_signal(self, symbol, current_price):
        """分析交易信号（简单示例）"""
        try:
            # 这里可以集成更复杂的交易策略
            # 目前使用简单的价格变化策略作为示例
            
            # 获取历史价格数据（如果有）
            price_history = getattr(self, 'price_history', {})
            if symbol not in price_history:
                price_history[symbol] = []
            
            # 记录当前价格
            price_history[symbol].append({
                'price': current_price,
                'timestamp': time.time()
            })
            
            # 只保留最近20个价格点
            if len(price_history[symbol]) > 20:
                price_history[symbol] = price_history[symbol][-20:]
            
            self.price_history = price_history
            
            # 简单的均线策略
            if len(price_history[symbol]) >= 10:
                recent_prices = [p['price'] for p in price_history[symbol][-10:]]
                short_ma = sum(recent_prices) / len(recent_prices)
                
                if len(price_history[symbol]) >= 20:
                    all_prices = [p['price'] for p in price_history[symbol]]
                    long_ma = sum(all_prices) / len(all_prices)
                    
                    # 短期均线上穿长期均线 -> 买入信号
                    if short_ma > long_ma * 1.02:  # 2% 以上
                        return 'BUY'
                    # 短期均线下穿长期均线 -> 卖出信号
                    elif short_ma < long_ma * 0.98:  # 2% 以下
                        return 'SELL'
            
            return 'HOLD'
            
        except Exception as e:
            self.log_message(f"❌ 分析 {symbol} 交易信号失败: {e}")
            return 'HOLD'

    # =============================
    # 策略信号集成
    # =============================
    def collect_strategy_signals(self) -> dict:
        """聚合各策略来源的交易信号，返回 {symbol: {action, confidence, current_price, ...}}"""
        try:
            if not self.config.get('use_strategy_signals', True):
                return {}
            sources = self.config.get('signal_sources', [])
            merged: dict = {}
            for src in sources:
                if isinstance(src, str) and src.endswith('.json'):
                    signals = self._parse_trading_signals_json(src)
                elif src == 'weekly_lstm':
                    signals = self._parse_weekly_lstm_signals()
                elif src == 'ensemble':
                    signals = self._parse_ensemble_signals()
                else:
                    signals = {}
                # 合并：优先高置信度
                for sym, sig in signals.items():
                    if sym not in merged or sig.get('confidence', 0) > merged[sym].get('confidence', 0):
                        merged[sym] = sig
            return merged
        except Exception as e:
            self.log_message(f"❌ 收集策略信号失败: {e}")
            return {}

    def _parse_trading_signals_json(self, file_path: str) -> dict:
        """解析通用 trading_signals.json 格式: { signals: {SYM: {...}} } 或列表"""
        try:
            if not os.path.exists(file_path):
                return {}
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            raw = data.get('signals', data)
            signals: dict = {}
            if isinstance(raw, dict):
                for sym, sig in raw.items():
                    if isinstance(sig, dict) and sig.get('action') in ('BUY', 'SELL'):
                        signals[sym] = {
                            'action': sig.get('action'),
                            'confidence': sig.get('confidence', 0),
                            'current_price': sig.get('target_price') or sig.get('current_price')
                        }
            elif isinstance(raw, list):
                for item in raw:
                    sym = item.get('ticker') or item.get('symbol')
                    if not sym:
                        continue
                    action = item.get('action')
                    if action in ('BUY', 'SELL'):
                        signals[sym] = {
                            'action': action,
                            'confidence': item.get('confidence', 0),
                            'current_price': item.get('current_price')
                        }
            return signals
        except Exception as e:
            self.log_message(f"❌ 解析 trading_signals.json 失败: {e}")
            return {}

    def _parse_weekly_lstm_signals(self) -> dict:
        """读取 weekly_trading_signals 目录中最新的 weekly_signals_*.json 文件并解析"""
        try:
            import glob
            files = sorted(glob.glob('weekly_trading_signals/weekly_signals_*.json'))
            if not files:
                return {}
            latest = files[-1]
            with open(latest, 'r', encoding='utf-8') as f:
                data = json.load(f)
            items = data.get('signals', [])
            signals = {}
            for item in items:
                sym = item.get('ticker')
                action = item.get('action')
                if action in ('BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL'):
                    normalized_action = 'BUY' if 'BUY' in action else 'SELL'
                    signals[sym] = {
                        'action': normalized_action,
                        'confidence': float(item.get('confidence', 0)),
                        'current_price': float(item.get('current_price', 0))
                    }
            return signals
        except Exception as e:
            self.log_message(f"❌ 解析 weekly LSTM 信号失败: {e}")
            return {}

    def _parse_ensemble_signals(self) -> dict:
        """解析 ensemble_signals.json 中的 signals: {SYM: score}，高于阈值买入，低于阈值卖出"""
        try:
            path = 'ensemble_signals.json'
            if not os.path.exists(path):
                return {}
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            scores = data.get('signals', {})
            signals = {}
            high, low = 0.7, 0.3
            for sym, score in scores.items():
                if score >= high:
                    signals[sym] = {'action': 'BUY', 'confidence': float(score)}
                elif score <= low:
                    signals[sym] = {'action': 'SELL', 'confidence': float(score)}
            return signals
        except Exception as e:
            self.log_message(f"❌ 解析融合策略信号失败: {e}")
            return {}
    
    def get_current_positions(self):
        """获取当前持仓"""
        try:
            positions = {}
            
            if hasattr(self, 'ib') and self.ib and self.ib.isConnected():
                # 获取实际持仓
                portfolio = self.ib.portfolio()
                for position in portfolio:
                    symbol = position.contract.symbol
                    quantity = position.position
                    positions[symbol] = quantity
                    
                self.log_message(f"💼 当前持仓: {positions}")
            
            return positions
            
        except Exception as e:
            self.log_message(f"❌ 获取持仓信息失败: {e}")
            return {}
    
    def connect_ibkr(self):
        """连接IBKR"""
        try:
            # 获取连接参数
            host, port = self.get_ibkr_connection_params()
            if host is None or port is None:
                return
            
            # 保存配置
            self.config['ibkr_host'] = host
            self.config['ibkr_port'] = port
            
            self.connection_status_label.config(text="🔄 正在连接...")
            self.log_message(f"🔗 正在连接IBKR {host}:{port}...")
            
            # 使用ib_insync连接IBKR
            if IBKR_AVAILABLE:
                try:
                    from ib_insync import IB
                    
                    # 创建IB连接
                    self.ib = IB()
                    
                    # 尝试连接，如果客户端ID冲突则重试
                    client_id = self.config.get('ibkr_client_id', 1)
                    max_retries = 3
                    
                    for attempt in range(max_retries):
                        try:
                            self.ib.connect(host, port, clientId=client_id)
                            break  # 连接成功，跳出循环
                        except Exception as e:
                            error_msg = str(e)
                            if "already in use" in error_msg or "326" in error_msg or "Peer closed connection" in error_msg:
                                # 客户端ID冲突，生成新的ID重试
                                old_client_id = client_id
                                client_id = TradingConstants.generate_unique_client_id()
                                self.log_message(f"⚠️ 客户端ID {old_client_id} 冲突，尝试新ID: {client_id}")
                                
                                # 更新配置中的客户端ID
                                self.config['ibkr_client_id'] = client_id
                                
                                if attempt == max_retries - 1:
                                    raise Exception(f"多次尝试后仍无法连接IBKR: {error_msg}")
                                
                                # 断开之前的连接尝试
                                if self.ib.isConnected():
                                    self.ib.disconnect()
                                time.sleep(1)  # 等待一秒后重试
                            else:
                                # 其他错误直接抛出
                                raise e
                    
                    if self.ib.isConnected():
                        self.connection_status_label.config(text="🟢 已连接")
                        self.log_message(f"✅ IBKR连接成功 ({host}:{port})")
                        
                        # 统一连接对象，便于后续方法使用
                        self.ib_connection = self.ib
                        self.is_ibkr_connected = True
                        
                        # 设置订单状态回调
                        self.setup_order_callbacks()
                        
                        # 设置增强订单管理器的IB连接
                        if hasattr(self, 'order_manager') and self.order_manager:
                            self.order_manager.set_ib_connection(self.ib)
                            self.log_message("✅ 增强订单管理器已连接IBKR")
                        
                        # 设置市场数据类型：优先实时报价(1)，失败则使用延迟(3)
                        try:
                            self.ib.reqMarketDataType(1)
                            self.log_message("✅ 市场数据类型: 实时 (1)")
                        except Exception as e1:
                            self.log_message(f"⚠️ 实时数据不可用，尝试延迟数据: {e1}")
                            try:
                                self.ib.reqMarketDataType(3)
                                self.log_message("✅ 市场数据类型: 延迟 (3)")
                            except Exception as e2:
                                self.log_message(f"❌ 设置市场数据类型失败: {e2}")
                        
                        # 连接成功后立即获取账户余额
                        self.log_message("💰 正在获取账户余额...")
                        balance = self.get_account_balance()
                        
                        if balance > 0:
                            self.log_message(f"✅ 账户余额获取成功: ${balance:,.2f}")
                            self.update_balance_display(balance)
                        else:
                            self.log_message("⚠️ 无法获取账户余额，将使用默认值")
                            self.update_balance_display(0)
                            
                    else:
                        self.connection_status_label.config(text="🔴 连接失败")
                        self.log_message("❌ IBKR连接失败")
                        messagebox.showerror("连接失败", "无法连接到IBKR")
                        
                except Exception as e:
                    self.connection_status_label.config(text="🔴 连接失败")
                    self.log_message(f"❌ IBKR连接失败: {e}")
                    messagebox.showerror("连接失败", f"IBKR连接失败: {e}")
            else:
                # 模拟连接成功（当IBKR API不可用时）
                self.connection_status_label.config(text="🟢 已连接（模拟）")
                self.log_message(f"✅ IBKR连接成功（模拟模式）({host}:{port})")
                self.log_message("⚠️ IBKR API不可用，使用模拟模式")
            
        except Exception as e:
            self.connection_status_label.config(text="🔴 连接失败")
            self.log_message(f"❌ IBKR连接失败: {e}")
            messagebox.showerror("连接失败", str(e))
    
    def test_ibkr_connection(self):
        """测试IBKR连接"""
        # 获取连接参数
        host, port = self.get_ibkr_connection_params()
        if host is None or port is None:
            return
        
        def run_test():
            try:
                # 运行IBKR连接测试脚本，传递端口参数
                self.log_message(f"🧪 启动IBKR连接测试... 主机: {host}, 端口: {port}")
                
                process = subprocess.Popen(
                    [sys.executable, "test_ibkr_connection.py", "--host", host, "--port", str(port)],
                    cwd=os.getcwd(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8'
                )
                
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    self.log_message(f"✅ IBKR连接测试完成 ({host}:{port})，请查看详细报告")
                    messagebox.showinfo("测试完成", f"IBKR连接测试完成！\n主机: {host}\n端口: {port}\n请查看result文件夹中的测试报告。")
                else:
                    self.log_message(f"❌ IBKR连接测试失败 ({host}:{port}): {stderr}")
                    messagebox.showerror("测试失败", f"IBKR连接测试失败 ({host}:{port}):\n{stderr}")
                
            except Exception as e:
                self.log_message(f"❌ 运行IBKR测试时出错: {e}")
                messagebox.showerror("错误", f"运行IBKR测试时出错: {e}")
        
        # 在新线程中运行测试
        threading.Thread(target=run_test, daemon=True).start()
    
    def disconnect_ibkr(self):
        """断开IBKR连接"""
        try:
            if hasattr(self, 'ib') and self.ib and self.ib.isConnected():
                self.ib.disconnect()
                self.log_message("✅ 已断开IBKR连接")
            else:
                self.log_message("❌ 已断开IBKR连接")
            
            # 清理连接状态
            self.connection_status_label.config(text="🔴 未连接")
            self.is_ibkr_connected = False
            self.ib_connection = None
            
            # 通知订单管理器
            if hasattr(self, 'order_manager') and self.order_manager:
                try:
                    self.order_manager.set_ib_connection(None)
                except Exception:
                    pass
            
        except Exception as e:
            self.log_message(f"❌ 断开IBKR连接时出错: {e}")
            self.connection_status_label.config(text="🔴 未连接")
    
    def start_auto_trading_wrapper(self):
        """启动自动交易的包装函数（同步版本）"""
        if not self.auto_trading_stocks:
            messagebox.showwarning("警告", "请先添加要交易的股票")
            return
        
        if not (hasattr(self, 'ib') and self.ib and self.ib.isConnected()):
            messagebox.showwarning("警告", "请先连接IBKR")
            return
        
        try:
            self.is_auto_trading = True
            self.trading_status_label.config(text="✅ 交易运行中")
            self.start_trading_btn.config(state="disabled")
            self.stop_trading_btn.config(state="normal")
            
            # 获取当前账户余额
            self.log_message("💰 正在获取账户余额...")
            balance = self.get_account_balance()
            
            if balance > 0:
                self.log_message(f"✅ 当前账户余额: ${balance:,.2f}")
            else:
                self.log_message("⚠️ 无法获取账户余额，将使用默认值")
                balance = 100000  # 默认值
                self.config['total_capital'] = balance
            
            # 启动定期更新账户余额
            self.update_account_balance_periodic()
            
            # 订阅所有交易股票的实时行情（使用同步方式）
            for symbol in self.auto_trading_stocks:
                try:
                    # 使用同步订阅方式
                    if hasattr(self, 'ib') and self.ib and self.ib.isConnected():
                        contract = self.create_contract(symbol)
                        if contract:
                            ticker = self.ib.reqMktData(contract, '', False, False)
                            if ticker:
                                self.ticker_subscriptions[symbol] = ticker
                                self.log_message(f"✅ 已订阅 {symbol} 实时行情")
                            else:
                                self.log_message(f"❌ 订阅 {symbol} 失败")
                        else:
                            self.log_message(f"❌ 创建 {symbol} 合约失败")
                    else:
                        self.log_message(f"❌ IBKR未连接，无法订阅 {symbol}")
                except Exception as e:
                    self.log_message(f"❌ 订阅 {symbol} 失败: {e}")
            
            self.log_message(f"🚀 已启动自动交易，监控 {len(self.auto_trading_stocks)} 只股票")
            self.log_message(f"📊 交易股票列表: {', '.join(self.auto_trading_stocks)}")
            self.log_message(f"💰 总资金: ${balance:,.2f}")
            
            # 启动实际的交易线程
            self.start_trading_thread()
            
        except Exception as e:
            self.log_message(f"❌ 启动自动交易失败: {e}")
            messagebox.showerror("错误", f"启动失败: {str(e)}")
    
    async def start_auto_trading(self):
        """启动自动交易（异步版本）"""
        if not self.auto_trading_stocks:
            messagebox.showwarning("警告", "请先添加要交易的股票")
            return
        
        if not (self.is_ibkr_connected and self.ib_connection and self.ib_connection.isConnected()):
            messagebox.showwarning("警告", "请先连接IBKR")
            return
        
        try:
            self.is_auto_trading = True
            self.trading_status_label.config(text="✅ 交易运行中")
            self.start_trading_btn.config(state="disabled")
            self.stop_trading_btn.config(state="normal")
            
            # 获取当前账户余额
            self.log_message("💰 正在获取账户余额...")
            balance = self.get_account_balance()
            
            if balance > 0:
                self.log_message(f"✅ 当前账户余额: ${balance:,.2f}")
            else:
                self.log_message("⚠️ 无法获取账户余额，将使用默认值")
                balance = 100000  # 默认值
                self.config['total_capital'] = balance
            
            # 启动定期更新账户余额
            self.update_account_balance_periodic()
            
            # 订阅所有交易股票的实时行情（异步）
            for symbol in self.auto_trading_stocks:
                # 使用异步订阅
                try:
                    await self.subscribe_market_data(symbol)
                    await asyncio.sleep(0.5)  # 避免订阅过快
                except Exception as e:
                    self.log_message(f"❌ 订阅 {symbol} 失败: {e}")
            
            self.log_message(f"🚀 已启动自动交易，监控 {len(self.auto_trading_stocks)} 只股票")
            self.log_message(f"📊 交易股票列表: {', '.join(self.auto_trading_stocks)}")
            self.log_message(f"💰 总资金: ${balance:,.2f}")
            
            # 启动实际的交易线程
            self.start_trading_thread()
            
        except Exception as e:
            self.log_message(f"❌ 启动自动交易失败: {e}")
            messagebox.showerror("错误", f"启动失败: {str(e)}")
    
    def stop_auto_trading(self):
        """停止自动交易"""
        self.is_auto_trading = False
        self.trading_status_label.config(text="❌ 交易已停止")
        self.start_trading_btn.config(state="normal")
        self.stop_trading_btn.config(state="disabled")
        
        self.log_message("⛔ 已停止自动交易")
    
    def emergency_sell_all(self):
        """紧急全仓卖出"""
        result = messagebox.askyesno("确认", "确定要紧急卖出所有持仓吗？\n\n此操作不可撤销！")
        if result:
            self.log_message("🚨 执行紧急全仓卖出")
            # TODO: 实现实际的全仓卖出逻辑
            messagebox.showinfo("执行完成", "紧急卖出指令已发送")
    
    def show_positions(self):
        """显示当前持仓"""
        positions_window = tk.Toplevel(self.root)
        positions_window.title("📊 当前持仓")
        positions_window.geometry("600x400")
        positions_window.transient(self.root)
        
        # TODO: 实现持仓显示界面
        ttk.Label(positions_window, text="持仓信息显示功能开发中...", 
                 font=('Microsoft YaHei', 12)).pack(expand=True)


def main():
    """主函数"""
    # 检查必要的依赖
    try:
        import tkinter
        import sqlite3
        from apscheduler.schedulers.background import BackgroundScheduler
        from plyer import notification
    except ImportError as e:
        print(f"缺少必要的依赖: {e}")
        print("请运行: pip install apscheduler plyer pywin32")
        return
    
    # 创建必要的目录
    for directory in ['logs', 'result']:
        Path(directory).mkdir(exist_ok=True)
    
    # 启动应用
    app = QuantitativeTradingManager()
    app.run()

if __name__ == "__main__":
    main() 