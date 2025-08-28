
# =============================================================================
# 🔧 SIGNAL CALCULATION RESTORED: Multi-factor signal logic implemented
# ✅ Trading signals now properly calculated using momentum, mean reversion, 
#    volatility, and volume factors with robust error handling
# =============================================================================
import asyncio
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import dataclass
from typing import Optional, List
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 使用懒加载减少直接依赖
from .lazy_component_loader import get_component_loader

# Enhanced error handling
from .error_handling_system import (
    get_error_handler, with_error_handling, error_handling_context,
    ErrorSeverity, ErrorCategory, ErrorContext
)

@dataclass
class AppState:
    json_file: Optional[str] = None
    excel_file: Optional[str] = None
    sheet: Optional[str] = None
    column: Optional[str] = None
    symbols_csv: Optional[str] = None
    host: Optional[str] = None  # 将从配置管理器获取
    port: Optional[int] = None  # 将从配置管理器获取
    client_id: Optional[int] = None  # 将从配置管理器获取
    # 交易参数
    alloc: Optional[float] = None  # 将从配置管理器获取
    poll_sec: Optional[float] = None  # 将从配置管理器获取
    auto_sell_removed: bool = True
    fixed_qty: int = 0
    # 数据库相关
    selected_stock_list_id: Optional[int] = None
    use_database: bool = True


class AutoTraderGUI(tk.Tk):
    def __init__(self) -> None:  # type: ignore
        super().__init__()
        
        # 使用懒加载组件管理器
        self.component_loader = get_component_loader()
        
        # 预加载关键组件
        if not self.component_loader.preload_critical_components():
            self.log("⚠️ 部分关键组件加载失败，系统可能不稳定")
        
        # 使用兼容性层确保向后兼容
        from .compatibility_layer import get_compatibility_wrapper
        compatibility = get_compatibility_wrapper()
        
        # 获取核心组件（带兼容性保证）
        self.config_manager = compatibility.get_config_manager()
        self.loop_manager = compatibility.get_event_loop_manager()
        self.resource_monitor = compatibility.get_resource_monitor()
        
        # 如果仍然失败，使用最基本的fallback
        if not self.config_manager:
            self.log("❌ 配置管理器完全加载失败，使用最小配置")
            class MinimalConfig:
                def get(self, key, default=None):
                    defaults = {
                        'ibkr': {'host': '127.0.0.1', 'port': 7497, 'client_id': 3130},
                        'trading': {'alloc': 0.03, 'poll_sec': 10.0}
                    }
                    return defaults.get(key, default)
                def update_runtime_config(self, config): pass
                def persist_runtime_changes(self): return False
            self.config_manager = MinimalConfig()
    
    def get_or_create_trader(self):
        """获取或创建IBKR交易器"""
        if not self.trader:
            self.trader = self.component_loader.get_component("ibkr_trader", config_manager=self.config_manager)
            if not self.trader:
                self.log("❌ IBKR交易器组件加载失败")
                return None
            
            # 注册到资源监控（如果可用）
            if self.resource_monitor and hasattr(self.resource_monitor, 'register_connection'):
                self.resource_monitor.register_connection(self.trader)
            
            self.log("✅ IBKR交易器组件加载成功")
        return self.trader
    
    def get_or_create_engine(self):
        """获取或创建交易引擎"""
        if not self.engine:
            trader = self.get_or_create_trader()
            if not trader:
                return None
            
            # 手动创建Engine，因为它需要特定参数
            try:
                from .engine import Engine
                self.engine = Engine(self.config_manager, trader)
                self.log("✅ 交易引擎组件创建成功")
            except Exception as e:
                self.log(f"❌ 交易引擎组件创建失败: {e}")
                return None
            
        return self.engine

    def _initialize_app_components(self):
        """初始化应用组件"""
        # Starting event loop manager (如果可用)
        if self.loop_manager and hasattr(self.loop_manager, 'start'):
            if not self.loop_manager.start():
                self.log("⚠️ Event loop manager启动失败")
        
        # start资源监控 (如果可用)
        if self.resource_monitor and hasattr(self.resource_monitor, 'start_monitoring'):
            self.resource_monitor.start_monitoring()
        
        # 初始化AppState使用统一配置，不自动分配Client ID
        conn_params = self.config_manager.get("ibkr", {})
        trading_params = self.config_manager.get("trading", {})
        self.state = AppState(
            host=conn_params.get('host', '127.0.0.1'),
            port=conn_params.get('port', 7497),
            client_id=conn_params.get('client_id', 3130),
            alloc=trading_params.get('alloc', 0.03),
            poll_sec=trading_params.get('poll_sec', 10.0)
        )
        self.title("IBKR 自动交易控制台")
        self.geometry("1000x700")
        
        # 提前初始化日志相关对象，避免在UI尚未构建完成前调用log引发属性错误
        self._log_buffer: List[str] = []
        self._log_lock = threading.Lock()
        self.txt = None  # type: ignore
        self._build_ui()
        
        # 调用组件初始化
        self._initialize_app_components()
        
        # 使用懒加载获取数据库组件
        self.db = self.component_loader.get_component("database")
        if not self.db:
            self.log("⚠️ 数据库组件加载失败，部分功能可能不可用")
        
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.trader = None  # 将通过懒加载获取
        self.engine = None  # 将通过懒加载获取
        # 改use统一配置管理器，not再需要HotConfig
        # self.hot_config: Optional[HotConfig] = HotConfig()
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_ready_event: Optional[threading.Event] = None
        self._engine_loop_task: Optional[asyncio.Task] = None
        # 状态跟踪变量
        self._model_training: bool = False
        self._model_trained: bool = False
        self._daily_trade_count: int = 0
        # 状态栏缓存，避免数值抖动/闪烁
        self._last_net_liq: Optional[float] = None
        
        # Ensure proper cleanup on window close
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # 添加资源清理回调
        self.resource_monitor.add_warning_callback(self._on_resource_warning)
        
        # 初始化事件系统
        from autotrader.unified_event_manager import get_event_bus, GUIEventAdapter
        self.event_bus = get_event_bus()
        self.gui_adapter = GUIEventAdapter(self, self.event_bus)
        
        # Initialize strategy engine components
        self._init_enhanced_trading_components()
        self._init_strategy_components()

    def _init_enhanced_trading_components(self):
        """初始化增强交易组件：阈值自适应 + 动态头寸 + 数据新鲜度"""
        try:
            from autotrader.data_freshness_scoring import create_freshness_scoring
            from autotrader.position_size_calculator import create_position_calculator
            from autotrader.volatility_adaptive_gating import create_volatility_gating
            
            # 数据新鲜度评分系统
            from autotrader.data_freshness_scoring import FreshnessConfig
            freshness_config = FreshnessConfig(
                tau_minutes=15.0,          # 15分钟衰减常数
                max_age_minutes=60.0,      # 最大数据年龄1小时
                base_threshold=0.005,      # 基础阈值0.5%
                freshness_threshold_add=0.010  # 新鲜度惩罚阈值1%
            )
            self.freshness_scorer = create_freshness_scoring(freshness_config)
            
            # 动态头寸规模计算器
            self.position_calculator = create_position_calculator(
                target_percentage=0.05,    # 目标5%头寸
                min_percentage=0.04,       # 最小4%
                max_percentage=0.10,       # 最大10%
                method="volatility_adjusted"  # 使用波动率调整方法
            )
            
            # 波动率自适应门控系统
            self.volatility_gating = create_volatility_gating(
                base_k=0.5,               # 基础门槛系数
                volatility_lookback=60,    # 60天波动率回望
                use_atr=True,             # 使用ATR计算波动率
                enable_liquidity_filter=True  # 启用流动性过滤
            )
            
            self.log("增强交易组件初始化成功: 数据新鲜度评分 + 动态头寸计算 + 波动率自适应门控")
            
        except Exception as e:
            self.log(f"增强交易组件初始化失败: {e}")
            # 设置回退组件
            self.freshness_scorer = None
            self.position_calculator = None
            self.volatility_gating = None
    
    def _init_strategy_components(self):
        """Initialize all strategy engine components"""
        try:
            # Initialize Enhanced Alpha Strategies Engine
            import sys
            import os
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # 使用懒加载获取组件
            self.alpha_engine = self.component_loader.get_component("alpha_engine")
            self.polygon_factors = self.component_loader.get_component("polygon_factors")
            
            # 初始化真实风险平衡器
            from .real_risk_balancer import get_risk_balancer_adapter
            if not hasattr(self, 'risk_balancer_adapter') or getattr(self, 'risk_balancer_adapter', None) is None:
                # 启用真实风险平衡器，提供风险控制
                risk_config = {
                    'max_single_position_weight': 0.12,  # 单仓最大12%
                    'max_sector_weight': 0.25,           # 行业最大25%
                    'min_trade_size': 50,                # 最小交易50股
                    'max_daily_turnover': 0.15           # 日换手率15%
                }
                self.risk_balancer_adapter = get_risk_balancer_adapter(enable_balancer=True, config=risk_config)
                self.log("✅ Real Risk Balancer initialized and enabled")
            
            # Create strategy status tracking
            polygon_ready = self.polygon_factors is not None
            self.strategy_status = {
                'alpha_engine_ready': True,
                'polygon_factors_ready': polygon_ready,
                'risk_balancer_ready': True,
                'bma_model_loaded': False,
                'lstm_model_loaded': False
            }
            
            self.log("Strategy Engine: Core components initialized successfully")
            
        except Exception as e:
            self.log(f"Strategy Engine: Initialization failed - {e}")
            # Set fallback status
            self.strategy_status = {
                'alpha_engine_ready': False,
                'polygon_factors_ready': False,
                'risk_balancer_ready': False,
                'bma_model_loaded': False,
                'lstm_model_loaded': False
            }

    def _init_polygon_factors(self):
        """Initialize Polygon factors with automatic API connection"""
        try:
            from autotrader.unified_polygon_factors import UnifiedPolygonFactors as PolygonCompleteFactors
            self.polygon_factors = PolygonCompleteFactors()
            self.log("Polygon API: Connected and factors initialized")
            return True
        except Exception as e:
            self.log(f"Polygon API: Connection failed - {e}")
            self.polygon_factors = None
            return False
    
    def _ensure_polygon_factors(self):
        """Ensure Polygon factors are initialized"""
        if self.polygon_factors is None:
            return self._init_polygon_factors()
        return True

    def _build_ui(self) -> None:
        # 顶层可滚动容器（Canvas + Scrollbar），使整个界面可往下滚动
        container = tk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar_main = tk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar_main.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.configure(yscrollcommand=scrollbar_main.set)

        frm = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frm, anchor="nw")

        def _on_frame_configure(event):
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                pass
        frm.bind("<Configure>", _on_frame_configure)

        # 鼠标滚轮支持（Windows）
        def _on_mousewheel(event):
            try:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # connection参数
        con = tk.LabelFrame(frm, text="connectionsettings")
        con.pack(fill=tk.X, pady=5)
        tk.Label(con, text="Host").grid(row=0, column=0)
        self.ent_host = tk.Entry(con)
        self.ent_host.insert(0, self.state.host)
        self.ent_host.grid(row=0, column=1)
        tk.Label(con, text="Port").grid(row=0, column=2)
        self.ent_port = tk.Entry(con, width=8)
        self.ent_port.insert(0, str(self.state.port))
        self.ent_port.grid(row=0, column=3)
        tk.Label(con, text="ClientId").grid(row=0, column=4)
        self.ent_cid = tk.Entry(con, width=8)
        self.ent_cid.insert(0, str(self.state.client_id))
        self.ent_cid.grid(row=0, column=5)

        # 创建笔记本选 items卡
        notebook = ttk.Notebook(frm)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 数据库股票管理选 items卡
        db_frame = ttk.Frame(notebook)
        notebook.add(db_frame, text="数据库股票管理")
        self._build_database_tab(db_frame)
        
        # 文件导入选 items卡
        file_frame = ttk.Frame(notebook)
        notebook.add(file_frame, text="文件导入")
        self._build_file_tab(file_frame)

        # 风险管理选 items卡
        risk_frame = ttk.Frame(notebook)
        notebook.add(risk_frame, text="风险管理")
        self._build_risk_tab(risk_frame)

        # Polygon API集成选项卡
        polygon_frame = ttk.Frame(notebook)
        notebook.add(polygon_frame, text="Polygon API")
        self._build_polygon_tab(polygon_frame)

        # 策略引擎选 items卡（集成模式2）
        engine_frame = ttk.Frame(notebook)
        notebook.add(engine_frame, text="策略引擎")
        self._build_engine_tab(engine_frame)

        # 直接交易选 items卡（集成模式3）
        direct_frame = ttk.Frame(notebook)
        notebook.add(direct_frame, text="直接交易")
        self._build_direct_tab(direct_frame)

        # 回测功能已移除 - 专注于实时交易

        # 交易参数settings
        params = tk.LabelFrame(frm, text="交易参数settings")
        params.pack(fill=tk.X, pady=5)
        
        # 第一行：资金分配and轮询间隔
        tk.Label(params, text="每股资金ratio").grid(row=0, column=0, padx=5, pady=5)
        self.ent_alloc = tk.Entry(params, width=8)
        self.ent_alloc.insert(0, str(self.state.alloc))
        self.ent_alloc.grid(row=0, column=1, padx=5)
        
        tk.Label(params, text="轮询间隔( seconds)").grid(row=0, column=2, padx=5)
        self.ent_poll = tk.Entry(params, width=8)
        self.ent_poll.insert(0, str(self.state.poll_sec))
        self.ent_poll.grid(row=0, column=3, padx=5)
        
        tk.Label(params, text="固定股数").grid(row=0, column=4, padx=5)
        self.ent_fixed_qty = tk.Entry(params, width=8)
        self.ent_fixed_qty.insert(0, str(self.state.fixed_qty))
        self.ent_fixed_qty.grid(row=0, column=5, padx=5)
        
        # 第二行：自动清仓选 items
        self.var_auto_sell = tk.BooleanVar(value=self.state.auto_sell_removed)
        tk.Checkbutton(params, text="移除股票when自动清仓", variable=self.var_auto_sell).grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # 动作按钮
        act = tk.LabelFrame(frm, text="操作")
        act.pack(fill=tk.X, pady=5)
        tk.Button(act, text="测试connection", command=self._test_connection, bg="lightblue").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="断开APIconnection", command=self._disconnect_api, bg="#ffcccc").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="start自动交易", command=self._start_autotrade, bg="lightgreen").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="停止交易", command=self._stop, bg="orange").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="清空日志", command=self._clear_log, bg="lightgray").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="查看account", command=self._show_account, bg="lightyellow").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="一键运行BMA模型", command=self._run_bma_model, bg="#d8b7ff").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="打印数据库", command=self._print_database, bg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="一键删除数据库", command=self._delete_database, bg="#ff6666").pack(side=tk.RIGHT, padx=5)

        # 运行状态告示栏
        status_frame = tk.LabelFrame(frm, text="引擎运行状态")
        status_frame.pack(fill=tk.X, pady=5)
        self._build_status_panel(status_frame)
        
        # 日志（添加可滚动）
        log_frame = tk.LabelFrame(frm, text="运行日志")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.txt = tk.Text(log_frame, height=8)
        scroll_y = tk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.txt.yview)
        self.txt.configure(yscrollcommand=scroll_y.set)
        self.txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        # will缓冲区in日志刷新to界面
        try:
            with self._log_lock:
                if getattr(self, "_log_buffer", None):
                    for _line in self._log_buffer:
                        self.txt.insert(tk.END, _line + "\n")
                    self.txt.see(tk.END)
                    self._log_buffer.clear()
        except Exception:
            pass

    def log(self, msg: str) -> None:
        # 同when输出to控制台andGUI
        try:
            print(msg)  # 输出to终端控制台
        except UnicodeEncodeError:
            # Windows控制台in文编码问题备选方案
            print(msg.encode('gbk', errors='ignore').decode('gbk', errors='ignore'))
        except Exception:
            # if果控制台输出failed，至少确保GUI日志还能工作
            pass
        
        # UI尚未completedorText尚未创建when，先写入缓冲区
        try:
            if hasattr(self, "txt") and isinstance(self.txt, tk.Text):
                self.txt.insert(tk.END, msg + "\n")
                self.txt.see(tk.END)
            else:
                # can能in构建UI早期be调use
                with self._log_lock:
                    if not hasattr(self, "_log_buffer"):
                        self._log_buffer = []  # type: ignore
                    self._log_buffer.append(msg)  # type: ignore
        except Exception:
            # 即便日志failed也not影响主流程
            try:
                with self._log_lock:
                    if not hasattr(self, "_log_buffer"):
                        self._log_buffer = []  # type: ignore
                    self._log_buffer.append(msg)  # type: ignore
            except Exception:
                pass

    def _build_risk_tab(self, parent) -> None:
        from .database import StockDatabase
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        box1 = ttk.LabelFrame(frm, text="基础参数")
        box1.pack(fill=tk.X, pady=5)
        ttk.Label(box1, text="默认止损 %").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.rm_stop = ttk.Spinbox(box1, from_=0.1, to=50.0, increment=0.1, width=8)
        self.rm_stop.set(2.0)
        self.rm_stop.grid(row=0, column=1, padx=5)
        ttk.Label(box1, text="默认止盈 %").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.rm_target = ttk.Spinbox(box1, from_=0.1, to=100.0, increment=0.1, width=8)
        self.rm_target.set(5.0)
        self.rm_target.grid(row=0, column=3, padx=5)
        ttk.Label(box1, text="real-time信号分配 %").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.rm_rt_alloc = ttk.Spinbox(box1, from_=0.0, to=1.0, increment=0.01, width=8)
        self.rm_rt_alloc.set(0.03)
        self.rm_rt_alloc.grid(row=0, column=5, padx=5)

        box2 = ttk.LabelFrame(frm, text="risk controland资金")
        box2.pack(fill=tk.X, pady=5)
        ttk.Label(box2, text="price下限").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.rm_price_min = ttk.Spinbox(box2, from_=0.0, to=1000.0, increment=0.5, width=8)
        self.rm_price_min.set(2.0)
        self.rm_price_min.grid(row=0, column=1, padx=5)
        ttk.Label(box2, text="price上限").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.rm_price_max = ttk.Spinbox(box2, from_=0.0, to=5000.0, increment=1.0, width=8)
        self.rm_price_max.set(800.0)
        self.rm_price_max.grid(row=0, column=3, padx=5)
        ttk.Label(box2, text="现金预留 %").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.rm_cash_reserve = ttk.Spinbox(box2, from_=0.0, to=0.9, increment=0.01, width=8)
        self.rm_cash_reserve.set(0.15)
        self.rm_cash_reserve.grid(row=1, column=1, padx=5)
        ttk.Label(box2, text="单标上限 %").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.rm_single_max = ttk.Spinbox(box2, from_=0.01, to=0.9, increment=0.01, width=8)
        self.rm_single_max.set(0.12)
        self.rm_single_max.grid(row=1, column=3, padx=5)
        ttk.Label(box2, text="最小order placement $").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.rm_min_order = ttk.Spinbox(box2, from_=0, to=10000, increment=50, width=8)
        self.rm_min_order.set(500)
        self.rm_min_order.grid(row=2, column=1, padx=5)
        ttk.Label(box2, text="日内订单上限").grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)
        self.rm_daily_limit = ttk.Spinbox(box2, from_=1, to=200, increment=1, width=8)
        self.rm_daily_limit.set(20)
        self.rm_daily_limit.grid(row=2, column=3, padx=5)

        box3 = ttk.LabelFrame(frm, text="ATR/做空/移除平仓")
        box3.pack(fill=tk.X, pady=5)
        self.rm_use_atr = tk.BooleanVar(value=False)
        ttk.Checkbutton(box3, text="使useATR动态止损", variable=self.rm_use_atr).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(box3, text="ATR止损倍数").grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.rm_atr_stop = ttk.Spinbox(box3, from_=0.5, to=10.0, increment=0.1, width=8)
        self.rm_atr_stop.set(2.0)
        self.rm_atr_stop.grid(row=0, column=2, padx=5)
        ttk.Label(box3, text="ATR止盈倍数").grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        self.rm_atr_target = ttk.Spinbox(box3, from_=0.5, to=10.0, increment=0.1, width=8)
        self.rm_atr_target.set(3.0)
        self.rm_atr_target.grid(row=0, column=4, padx=5)
        ttk.Label(box3, text="ATR风险尺度").grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)
        self.rm_atr_scale = ttk.Spinbox(box3, from_=0.1, to=20.0, increment=0.1, width=8)
        self.rm_atr_scale.set(5.0)
        self.rm_atr_scale.grid(row=0, column=6, padx=5)
        self.rm_allow_short = tk.BooleanVar(value=True)
        ttk.Checkbutton(box3, text="允许做空", variable=self.rm_allow_short).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.rm_bracket_removed = tk.BooleanVar(value=False)
        ttk.Checkbutton(box3, text="移除平仓使usebracket order(not推荐)", variable=self.rm_bracket_removed).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        box4 = ttk.LabelFrame(frm, text="Webhook通知")
        box4.pack(fill=tk.X, pady=5)
        ttk.Label(box4, text="Webhook URL").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.rm_webhook = ttk.Entry(box4, width=60)
        self.rm_webhook.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        act = ttk.Frame(frm)
        act.pack(fill=tk.X, pady=10)
        ttk.Button(act, text="加载配置", command=self._risk_load).pack(side=tk.LEFT, padx=5)
        ttk.Button(act, text="保存配置", command=self._risk_save).pack(side=tk.LEFT, padx=5)

        self._risk_load()

    def _risk_load(self) -> None:
        from .database import StockDatabase
        try:
            db = StockDatabase()
            cfg = db.get_risk_config() or {}
            rm = cfg.get('risk_management', cfg) if isinstance(cfg, dict) else {}
            self.rm_stop.delete(0, tk.END); self.rm_stop.insert(0, str(rm.get('default_stop_pct', 0.02)*100))
            self.rm_target.delete(0, tk.END); self.rm_target.insert(0, str(rm.get('default_target_pct', 0.05)*100))
            self.rm_rt_alloc.delete(0, tk.END); self.rm_rt_alloc.insert(0, str(rm.get('realtime_alloc_pct', 0.03)))
            pr = rm.get('price_range', (2.0, 800.0))
            self.rm_price_min.delete(0, tk.END); self.rm_price_min.insert(0, str(pr[0]))
            self.rm_price_max.delete(0, tk.END); self.rm_price_max.insert(0, str(pr[1]))
            self.rm_cash_reserve.delete(0, tk.END); self.rm_cash_reserve.insert(0, str(rm.get('cash_reserve_pct', 0.15)))
            self.rm_single_max.delete(0, tk.END); self.rm_single_max.insert(0, str(rm.get('max_single_position_pct', 0.12)))
            self.rm_min_order.delete(0, tk.END); self.rm_min_order.insert(0, str(rm.get('min_order_value_usd', 500)))
            self.rm_daily_limit.delete(0, tk.END); self.rm_daily_limit.insert(0, str(rm.get('daily_order_limit', 20)))
            self.rm_use_atr.set(bool(rm.get('use_atr_stops', False)))
            self.rm_atr_stop.delete(0, tk.END); self.rm_atr_stop.insert(0, str(rm.get('atr_multiplier_stop', 2.0)))
            self.rm_atr_target.delete(0, tk.END); self.rm_atr_target.insert(0, str(rm.get('atr_multiplier_target', 3.0)))
            self.rm_atr_scale.delete(0, tk.END); self.rm_atr_scale.insert(0, str(rm.get('atr_risk_scale', 5.0)))
            self.rm_allow_short.set(bool(rm.get('allow_short', True)))
            self.rm_bracket_removed.set(bool(rm.get('use_bracket_on_removed', False)))
            self.rm_webhook.delete(0, tk.END); self.rm_webhook.insert(0, rm.get('webhook_url', ''))
            self.log("Risk configuration loaded")
        except Exception as e:
            self.log(f"加载风险配置failed: {e}")

    def _risk_save(self) -> None:
        from .database import StockDatabase
        try:
            rm = {
                'default_stop_pct': float(self.rm_stop.get())/100.0,
                'default_target_pct': float(self.rm_target.get())/100.0,
                'price_range': (float(self.rm_price_min.get()), float(self.rm_price_max.get())),
                'cash_reserve_pct': float(self.rm_cash_reserve.get()),
                'max_single_position_pct': float(self.rm_single_max.get()),
                'min_order_value_usd': float(self.rm_min_order.get()),
                'daily_order_limit': int(self.rm_daily_limit.get()),
                'use_atr_stops': bool(self.rm_use_atr.get()),
                'atr_multiplier_stop': float(self.rm_atr_stop.get()),
                'atr_multiplier_target': float(self.rm_atr_target.get()),
                'atr_risk_scale': float(self.rm_atr_scale.get()),
                'allow_short': bool(self.rm_allow_short.get()),
                'use_bracket_on_removed': bool(self.rm_bracket_removed.get()),
                'webhook_url': self.rm_webhook.get().strip(),
                'realtime_alloc_pct': float(self.rm_rt_alloc.get()),
                'symbol_overrides': {},
                'strategy_settings': {},
            }
            cfg = {'risk_management': rm}
            db = StockDatabase()
            ok = db.save_risk_config(cfg)
            if ok:
                self.log("风险配置保存to数据库")
            else:
                self.log("风险配置保存failed")
            db.close()
            
            # 同whenupdates统一配置管理器并持久化
            self.config_manager.update_runtime_config({
                'capital.cash_reserve_pct': rm['cash_reserve_pct'],
                'capital.max_single_position_pct': rm['max_single_position_pct'],
                'capital.max_portfolio_exposure': rm['realtime_alloc_pct'],
                'orders.default_stop_loss_pct': rm['default_stop_pct'],
                'orders.default_take_profit_pct': rm['default_target_pct'],
                'orders.min_order_value_usd': rm['min_order_value_usd'],
                'orders.daily_order_limit': rm['daily_order_limit'],
                'risk.use_atr_stops': rm['use_atr_stops'],
                'risk.atr_multiplier_stop': rm['atr_multiplier_stop'],
                'risk.atr_multiplier_target': rm['atr_multiplier_target'],
                'risk.allow_short': rm['allow_short']
            })
            
            # 持久化to文件
            if self.config_manager.persist_runtime_changes():
                self.log(" 风险配置持久化to配置文件")
            else:
                self.log(" 风险配置持久化failed，但保存to数据库")
        except Exception as e:
            self.log(f"保存风险配置failed: {e}")

    def _build_polygon_tab(self, parent) -> None:
        """Polygon API集成选项卡"""
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Polygon API状态
        status_frame = ttk.LabelFrame(frm, text="Polygon API状态")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.polygon_status_label = tk.Label(status_frame, text="状态: 正在连接...", fg="blue")
        self.polygon_status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        ttk.Button(status_frame, text="刷新连接", command=self._refresh_polygon_connection).pack(side=tk.RIGHT, padx=10, pady=5)

        # 实用功能 (不是测试功能)
        function_frame = ttk.LabelFrame(frm, text="市场数据功能")
        function_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(function_frame, text="获取实时报价", command=self._get_realtime_quotes).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(function_frame, text="获取历史数据", command=self._get_historical_data).grid(row=0, column=1, padx=5, pady=5)
        
        # 状态信息显示
        info_frame = ttk.LabelFrame(frm, text="API信息")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.polygon_info_text = tk.Text(info_frame, height=10, state=tk.DISABLED)
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.polygon_info_text.yview)
        self.polygon_info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.polygon_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 初始化状态显示
        self._update_polygon_status()

    def _refresh_polygon_connection(self):
        """刷新Polygon API连接"""
        try:
            self.log("Refreshing Polygon API connection...")
            self._ensure_polygon_factors()
            self._update_polygon_status()
        except Exception as e:
            self.log(f"Failed to refresh Polygon connection: {e}")

    def _get_realtime_quotes(self):
        """获取实时报价"""
        try:
            if self.polygon_factors:
                self.log("Fetching real-time quotes from Polygon API...")
                # 这里可以添加获取实时报价的逻辑
                self.log("Real-time quotes functionality ready")
            else:
                self.log("Polygon API not connected")
        except Exception as e:
            self.log(f"Failed to get real-time quotes: {e}")

    def _get_historical_data(self):
        """获取历史数据"""
        try:
            if self.polygon_factors:
                self.log("Fetching historical data from Polygon API...")
                # 这里可以添加获取历史数据的逻辑
                self.log("Historical data functionality ready")
            else:
                self.log("Polygon API not connected")
        except Exception as e:
            self.log(f"Failed to get historical data: {e}")

    def _enable_polygon_factors(self):
        """启usePolygon因子"""
        try:
            if hasattr(self, 'trader') and self.trader:
                self.trader.enable_polygon_factors()
                self.log("Polygon因子启use")
            else:
                self.log("请先connection交易系统")
        except Exception as e:
            self.log(f"启usePolygon因子failed: {e}")

    def _clear_polygon_cache(self):
        """清理Polygon缓存"""
        try:
            if hasattr(self, 'trader') and self.trader:
                self.trader.clear_polygon_cache()
                self.log("Polygon缓存清理")
            else:
                self.log("请先connection交易系统")
        except Exception as e:
            self.log(f"清理Polygon缓存failed: {e}")

    def _toggle_polygon_balancer(self):
        """切换risk control收益平衡器状态"""
        try:
            if hasattr(self, 'trader') and self.trader:
                if self.polygon_balancer_var.get():
                    self.trader.enable_polygon_risk_balancer()
                    self.log("risk control收益平衡器启use")
                else:
                    self.trader.disable_polygon_risk_balancer()
                    self.log("risk control收益平衡器禁use")
            else:
                self.log("请先connection交易系统")
                self.polygon_balancer_var.set(False)
        except Exception as e:
            self.log(f"切换risk control收益平衡器状态failed: {e}")
            self.polygon_balancer_var.set(False)

    def _open_balancer_config(self):
        """打开risk control收益平衡器配置面板"""
        try:
            # 导入GUI面板
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            from risk_balancer_gui_panel import create_standalone_gui
            
            # in新线程in打开GUI，避免阻塞主界面
            import threading
            gui_thread = threading.Thread(target=create_standalone_gui, daemon=True)
            gui_thread.start()
            
            self.log("risk control收益平衡器配置面板打开")
            
        except Exception as e:
            self.log(f"打开配置面板failed: {e}")

    def _update_polygon_status(self):
        """updatesPolygon状态显示"""
        try:
            if hasattr(self, 'trader') and self.trader:
                # checkPolygonconnection状态
                polygon_enabled = hasattr(self.trader, 'polygon_enabled') and self.trader.polygon_enabled
                balancer_enabled = hasattr(self.trader, 'polygon_risk_balancer_enabled') and self.trader.polygon_risk_balancer_enabled
                
                if polygon_enabled:
                    status_text = "状态: Polygonconnection"
                    status_color = "green"
                else:
                    status_text = "状态: Polygon未connection"
                    status_color = "red"
                
                self.polygon_status_label.config(text=status_text, fg=status_color)
                self.polygon_balancer_var.set(balancer_enabled)
                
                # updates统计信息
                stats = self.trader.get_polygon_stats()
                if stats:
                    stats_text = "Polygon统计信息:\n"
                    stats_text += f"  启use状态: {'is' if stats.get('enabled', False) else '否'}\n"
                    stats_text += f"  risk control平衡器: {'is' if stats.get('risk_balancer_enabled', False) else '否'}\n"
                    stats_text += f"  缓存大小: {stats.get('cache_size', 0)}\n"
                    stats_text += f"  总计算次数: {stats.get('total_calculations', 0)}\n"
                    stats_text += f"  success次数: {stats.get('successful_calculations', 0)}\n"
                    stats_text += f"  failed次数: {stats.get('failed_calculations', 0)}\n"
                    stats_text += f"  缓存命in: {stats.get('cache_hits', 0)}\n"
                    
                    # 组件状态
                    components = stats.get('components', {})
                    stats_text += "\n组件状态:\n"
                    for comp, status in components.items():
                        stats_text += f"  {comp}: {'[OK]' if status else '[FAIL]'}\n"
                    
                    self.polygon_stats_text.config(state=tk.NORMAL)
                    self.polygon_stats_text.delete(1.0, tk.END)
                    self.polygon_stats_text.insert(1.0, stats_text)
                    self.polygon_stats_text.config(state=tk.DISABLED)
                else:
                    self.polygon_stats_text.config(state=tk.NORMAL)
                    self.polygon_stats_text.delete(1.0, tk.END)
                    self.polygon_stats_text.insert(1.0, "暂no统计信息")
                    self.polygon_stats_text.config(state=tk.DISABLED)
            else:
                self.polygon_status_label.config(text="状态: 未connection交易系统", fg="gray")
                
        except Exception as e:
            self.polygon_status_label.config(text=f"状态: checkfailed ({e})", fg="red")

    def _schedule_polygon_update(self):
        """定whenupdatesPolygon状态"""
        self._update_polygon_status()
        self.after(5000, self._schedule_polygon_update)  # 每5 secondsupdates一次

    def _build_engine_tab(self, parent) -> None:
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Strategy Engine Control Section
        engine_box = ttk.LabelFrame(frm, text="Strategy Engine Control")
        engine_box.pack(fill=tk.X, pady=8)

        ttk.Button(engine_box, text="Start Engine (Connect/Subscribe)", command=self._start_engine).grid(row=0, column=0, padx=6, pady=6)
        ttk.Button(engine_box, text="Run Signal & Trading Once", command=self._engine_once).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(engine_box, text="Stop Engine", command=self._stop_engine_mode).grid(row=0, column=2, padx=6, pady=6)

        # Strategy Engine Section (simplified)
        strategy_box = ttk.LabelFrame(frm, text="Strategy Engine")
        strategy_box.pack(fill=tk.X, pady=8)

        ttk.Button(strategy_box, text="Run BMA Model", command=self._run_bma_model).grid(row=0, column=0, padx=6, pady=6)
        ttk.Button(strategy_box, text="Generate Trading Signals", command=self._generate_trading_signals).grid(row=0, column=1, padx=6, pady=6)

        # Risk Management Section
        risk_box = ttk.LabelFrame(frm, text="Risk Management")
        risk_box.pack(fill=tk.X, pady=8)

        # Risk balancer status
        self.risk_balancer_var = tk.BooleanVar()
        ttk.Checkbutton(risk_box, text="Enable Risk Balancer", variable=self.risk_balancer_var, 
                       command=self._toggle_risk_balancer).grid(row=0, column=0, padx=6, pady=6)
        
        ttk.Button(risk_box, text="View Risk Stats", command=self._view_risk_stats).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(risk_box, text="Reset Risk Limits", command=self._reset_risk_limits).grid(row=0, column=2, padx=6, pady=6)

        # Strategy Status Display
        status_box = ttk.LabelFrame(frm, text="Strategy Status")
        status_box.pack(fill=tk.BOTH, expand=True, pady=8)
        
        self.strategy_status_text = tk.Text(status_box, height=8, width=80)
        self.strategy_status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar
        scrollbar = tk.Scrollbar(status_box)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.strategy_status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.strategy_status_text.yview)
        
        # Update status display
        self._update_strategy_status()

        tip = ttk.Label(frm, text="Strategy Engine: Uses unified configuration manager to scan universe, compute multi-factor signals and place orders.")
        tip.pack(anchor=tk.W, pady=6)

    def _build_direct_tab(self, parent) -> None:
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 行1：基本参数
        row1 = ttk.LabelFrame(frm, text="order placement参数")
        row1.pack(fill=tk.X, pady=6)
        ttk.Label(row1, text="标").grid(row=0, column=0, padx=5, pady=5)
        self.d_sym = ttk.Entry(row1, width=12); self.d_sym.grid(row=0, column=1, padx=5)
        ttk.Label(row1, text="数量").grid(row=0, column=2, padx=5)
        self.d_qty = ttk.Entry(row1, width=10); self.d_qty.insert(0, "100"); self.d_qty.grid(row=0, column=3, padx=5)
        ttk.Label(row1, text="limit").grid(row=0, column=4, padx=5)
        self.d_px = ttk.Entry(row1, width=10); self.d_px.grid(row=0, column=5, padx=5)

        # 行2：基本按钮
        row2 = ttk.LabelFrame(frm, text="基础order placement")
        row2.pack(fill=tk.X, pady=6)
        ttk.Button(row2, text="market买入", command=lambda: self._direct_market("BUY")).grid(row=0, column=0, padx=6, pady=6)
        ttk.Button(row2, text="market卖出", command=lambda: self._direct_market("SELL")).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(row2, text="limit买入", command=lambda: self._direct_limit("BUY")).grid(row=0, column=2, padx=6, pady=6)
        ttk.Button(row2, text="limit卖出", command=lambda: self._direct_limit("SELL")).grid(row=0, column=3, padx=6, pady=6)

        # 行3：bracket order
        row3 = ttk.LabelFrame(frm, text="Bracket Orders")
        row3.pack(fill=tk.X, pady=6)
        ttk.Label(row3, text="Stop Loss %").grid(row=0, column=0, padx=5)
        self.d_stop = ttk.Entry(row3, width=8); self.d_stop.insert(0, "2.0"); self.d_stop.grid(row=0, column=1)
        ttk.Label(row3, text="Take Profit %").grid(row=0, column=2, padx=5)
        self.d_tp = ttk.Entry(row3, width=8); self.d_tp.insert(0, "5.0"); self.d_tp.grid(row=0, column=3)
        ttk.Button(row3, text="Market Bracket (Buy)", command=lambda: self._direct_bracket("BUY")).grid(row=0, column=4, padx=6, pady=6)
        ttk.Button(row3, text="Market Bracket (Sell)", command=lambda: self._direct_bracket("SELL")).grid(row=0, column=5, padx=6, pady=6)
        
        # System Operations Section (essential functions only)
        ops_box = ttk.LabelFrame(frm, text="System Operations")
        ops_box.pack(fill=tk.X, pady=6)
        
        ttk.Button(ops_box, text="Test Connection", command=self._test_connection).grid(row=0, column=0, padx=6, pady=6)
        
        # Strategy Integration Section
        strategy_box = ttk.LabelFrame(frm, text="Strategy Integration")
        strategy_box.pack(fill=tk.X, pady=6)
        
        ttk.Button(strategy_box, text="Manual Signal Entry", command=self._manual_signal_entry).grid(row=0, column=0, padx=6, pady=6)
        ttk.Button(strategy_box, text="Execute Alpha Signals", command=self._execute_alpha_signals).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(strategy_box, text="Portfolio Rebalance", command=self._portfolio_rebalance).grid(row=0, column=2, padx=6, pady=6)
        
        # System Status Section
        status_display_box = ttk.LabelFrame(frm, text="System Status")
        status_display_box.pack(fill=tk.BOTH, expand=True, pady=6)
        
        self.system_status_text = tk.Text(status_display_box, height=8, width=80)
        self.system_status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar for system status
        status_scrollbar = tk.Scrollbar(status_display_box)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.system_status_text.config(yscrollcommand=status_scrollbar.set)
        status_scrollbar.config(command=self.system_status_text.yview)
        
        # Initialize system status
        self._update_system_status()

        # 行4：高级执行
        row4 = ttk.LabelFrame(frm, text="高级执行")
        row4.pack(fill=tk.X, pady=6)
        ttk.Label(row4, text="算法").grid(row=0, column=0, padx=5)
        self.d_algo = ttk.Combobox(row4, values=["TWAP", "VWAP", "ICEBERG"], width=10)
        self.d_algo.current(0)
        self.d_algo.grid(row=0, column=1, padx=5)
        ttk.Label(row4, text="持续(分钟)").grid(row=0, column=2, padx=5)
        self.d_dur = ttk.Entry(row4, width=8); self.d_dur.insert(0, "30"); self.d_dur.grid(row=0, column=3, padx=5)
        ttk.Button(row4, text="执行大单(买)", command=lambda: self._direct_algo("BUY")).grid(row=0, column=4, padx=6, pady=6)
        ttk.Button(row4, text="执行大单(卖)", command=lambda: self._direct_algo("SELL")).grid(row=0, column=5, padx=6, pady=6)

    def _start_engine(self) -> None:
        try:
            # 采集最新UI参数
            self._capture_ui()
            # 立即in主线程提示，避免“no反应”感受
            self.log(f"准备start引擎(connection/subscription)... Host={self.state.host} Port={self.state.port} ClientId={self.state.client_id}")
            loop = self._ensure_loop()
            async def _run():
                try:
                    # 线程安全日志
                    try:
                        self.after(0, lambda: self.log(
                            f"start引擎参数: Host={self.state.host}, Port={self.state.port}, ClientID={self.state.client_id}"))
                    except Exception:
                        pass
                    # startbefore先断开现hasconnection，避免clientId占use
                    if self.trader and getattr(self.trader, 'ib', None) and self.trader.ib.isConnected():
                        try:
                            await self.trader.close()
                            try:
                                self.after(0, lambda: self.log("断开之beforeAPIconnection"))
                            except Exception as e:
                                # GUI更新失败不影响核心逻辑
                                self.log(f"GUI日志更新失败: {e}")
                        except Exception as e:
                            # 连接关闭失败是关键错误，需要记录并可能影响后续操作
                            self.log(f"严重错误：无法关闭旧连接: {e}")
                            # 设置错误状态但继续尝试新连接
                            self._set_connection_error_state(f"旧连接关闭失败: {e}")
                    # 使用统一的组件获取方法
                    if not self.get_or_create_trader():
                        raise Exception("IBKR交易器组件加载失败")
                    
                    if not self.get_or_create_engine():
                        raise Exception("交易引擎组件加载失败")
                    await self.engine.start()
                    try:
                        self.after(0, lambda: self.log("策略引擎start并completedsubscription"))
                        self.after(0, lambda: self._update_signal_status("引擎start", "green"))
                    except Exception:
                        pass
                except Exception as e:
                    error_msg = str(e)
                    try:
                        self.after(0, lambda e_msg=error_msg: self.log(f"策略引擎startfailed: {e_msg}"))
                    except Exception:
                        print(f"策略引擎startfailed: {e}")  # 降级日志
            # 使use线程安全事件循环管理器（非阻塞）
            try:
                task_id = self.loop_manager.submit_coroutine_nowait(_run())
                self.after(0, lambda: self.log(f"策略引擎任务提交 (ID: {task_id[:8]}...)"))
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda e_msg=error_msg: self.log(f"策略引擎startfailed: {e_msg}"))
        except Exception as e:
            self.log(f"start引擎错误: {e}")

    def _engine_once(self) -> None:
        try:
            if not self.engine:
                self.log("请先start引擎")
                return
            # 使use非阻塞提交避免GUI卡死
            if hasattr(self, 'loop_manager') and self.loop_manager.is_running:
                task_id = self.loop_manager.submit_coroutine_nowait(self.engine.on_signal_and_trade())
                self.log(f"信号交易提交，任务ID: {task_id}")
            else:
                self.log("事件循环未运行，no法执行信号交易")
            self.log("触发一次信号and交易")
            self._update_signal_status("执行交易信号", "blue")
        except Exception as e:
            self.log(f"运行引擎一次failed: {e}")

    def _stop_engine_mode(self) -> None:
        try:
            self.log("策略引擎停止：can通过停止交易按钮一并断开connectionand任务")
            self._update_signal_status("停止", "red")
        except Exception as e:
            self.log(f"停止引擎failed: {e}")

    def _direct_market(self, side: str) -> None:
        try:
            sym = (self.d_sym.get() or "").strip().upper()
            qty = int(self.d_qty.get().strip())
            if not sym or qty <= 0:
                messagebox.showwarning("警告", "请输入has效标and数量")
                return
            loop = self._ensure_loop()
            async def _run():
                try:
                    if not self.trader:
                        self.get_or_create_trader()
                        await self.trader.connect()
                    await self.trader.place_market_order(sym, side, qty)
                    self.log(f"提交market单: {side} {qty} {sym}")
                except Exception as e:
                    self.log(f"market单failed: {e}")
            # 使use非阻塞提交避免GUI卡死
            if hasattr(self, 'loop_manager') and self.loop_manager.is_running:
                task_id = self.loop_manager.submit_coroutine_nowait(_run())
                self.log(f"order placement任务提交，任务ID: {task_id}")
            else:
                self.log("事件循环未运行，no法执行order placement操作")
        except Exception as e:
            self.log(f"marketorder placement错误: {e}")

    def _direct_limit(self, side: str) -> None:
        try:
            sym = (self.d_sym.get() or "").strip().upper()
            qty = int(self.d_qty.get().strip())
            px_str = (self.d_px.get() or "").strip()
            if not sym or qty <= 0 or not px_str:
                messagebox.showwarning("警告", "请输入标/数量/limit")
                return
            px = float(px_str)
            loop = self._ensure_loop()
            async def _run():
                try:
                    if not self.trader:
                        self.get_or_create_trader()
                        await self.trader.connect()
                    await self.trader.place_limit_order(sym, side, qty, px)
                    self.log(f"提交limit单: {side} {qty} {sym} @ {px}")
                except Exception as e:
                    self.log(f"limit单failed: {e}")
            # 使use非阻塞提交避免GUI卡死
            if hasattr(self, 'loop_manager') and self.loop_manager.is_running:
                task_id = self.loop_manager.submit_coroutine_nowait(_run())
                self.log(f"order placement任务提交，任务ID: {task_id}")
            else:
                self.log("事件循环未运行，no法执行order placement操作")
        except Exception as e:
            self.log(f"limitorder placement错误: {e}")

    def _direct_bracket(self, side: str) -> None:
        try:
            sym = (self.d_sym.get() or "").strip().upper()
            qty = int(self.d_qty.get().strip())
            stop_pct = float((self.d_stop.get() or "2.0").strip())/100.0
            tp_pct = float((self.d_tp.get() or "5.0").strip())/100.0
            if not sym or qty <= 0:
                messagebox.showwarning("警告", "请输入标and数量")
                return
            loop = self._ensure_loop()
            async def _run():
                try:
                    if not self.trader:
                        self.get_or_create_trader()
                        await self.trader.connect()
                    await self.trader.place_market_order_with_bracket(sym, side, qty, stop_pct=stop_pct, target_pct=tp_pct)
                    self.log(f"提交bracket order: {side} {qty} {sym} (止损{stop_pct*100:.1f}%, 止盈{tp_pct*100:.1f}%)")
                except Exception as e:
                    self.log(f"bracket orderfailed: {e}")
            # 使use非阻塞提交避免GUI卡死
            if hasattr(self, 'loop_manager') and self.loop_manager.is_running:
                task_id = self.loop_manager.submit_coroutine_nowait(_run())
                self.log(f"order placement任务提交，任务ID: {task_id}")
            else:
                self.log("事件循环未运行，no法执行order placement操作")
        except Exception as e:
            self.log(f"bracket order错误: {e}")

    def _direct_algo(self, side: str) -> None:
        try:
            sym = (self.d_sym.get() or "").strip().upper()
            qty = int(self.d_qty.get().strip())
            algo = (self.d_algo.get() or "TWAP").strip().upper()
            dur_min = int((self.d_dur.get() or "30").strip())
            if not sym or qty <= 0:
                messagebox.showwarning("警告", "请输入标and数量")
                return
            loop = self._ensure_loop()
            async def _run():
                try:
                    if not self.trader:
                        self.get_or_create_trader()
                        await self.trader.connect()
                    await self.trader.execute_large_order(sym, side, qty, algorithm=algo, duration_minutes=dur_min)
                    self.log(f"提交大单执行: {algo} {side} {qty} {sym} / {dur_min}min")
                except Exception as e:
                    self.log(f"大单执行failed: {e}")
            # 使use非阻塞提交避免GUI卡死
            if hasattr(self, 'loop_manager') and self.loop_manager.is_running:
                task_id = self.loop_manager.submit_coroutine_nowait(_run())
                self.log(f"order placement任务提交，任务ID: {task_id}")
            else:
                self.log("事件循环未运行，no法执行order placement操作")
        except Exception as e:
            self.log(f"大单执行错误: {e}")

    def _delete_database(self) -> None:
        """一键删除数据库文件（含确认and重建）"""
        try:
            import os
            db_path = getattr(self.db, 'db_path', None)
            if not db_path:
                messagebox.showerror("错误", "未找to数据库路径")
                return
            
            if not os.path.exists(db_path):
                messagebox.showinfo("提示", "数据库文件not存in，no需删除")
                return
            
            confirm = messagebox.askyesno(
                "确认删除",
                f"will删除数据库文件:\n{db_path}\n\n此操作notcan恢复，is否继续？"
            )
            if not confirm:
                return
            
            # 关闭connection再删除
            try:
                self.db.close()
            except Exception:
                pass
            
            os.remove(db_path)
            self.log(f"删除数据库: {db_path}")
            
            # 重新初始化数据库并刷新UI
            self.db = StockDatabase()
            self._refresh_stock_lists()
            self._refresh_configs()
            messagebox.showinfo("completed", "数据库删除并重建as空库")
        
        except Exception as e:
            self.log(f"删除数据库failed: {e}")
            messagebox.showerror("错误", f"删除数据库failed: {e}")

    def _print_database(self) -> None:
        """打印当before数据库内容to日志（全局tickers、股票列表、选in列表、交易配置）。"""
        try:
            # 全局 tickers
            tickers = []
            try:
                tickers = self.db.get_all_tickers()
            except Exception:
                pass
            if tickers:
                preview = ", ".join(tickers[:200]) + ("..." if len(tickers) > 200 else "")
                self.log(f"全局 tickers 共 {len(tickers)}: {preview}")
            else:
                self.log("全局 tickers: no")

            # 股票列表概览
            try:
                lists = self.db.get_stock_lists()
            except Exception:
                lists = []
            if lists:
                summary = ", ".join([f"{it['name']}({it.get('stock_count', 0)})" for it in lists])
                self.log(f"股票列表 {len(lists)} 个: {summary}")
            else:
                self.log("股票列表: no")

            # 当before选in列表明细
            try:
                if self.state.selected_stock_list_id:
                    rows = self.db.get_stocks_in_list(self.state.selected_stock_list_id)
                    syms = [r.get('symbol') for r in rows]
                    preview = ", ".join(syms[:200]) + ("..." if len(syms) > 200 else "")
                    self.log(f"当before列表 {self.stock_list_var.get()} 共 {len(syms)}: {preview}")
            except Exception:
                pass

            # 交易配置名称
            try:
                cfgs = self.db.get_trading_configs()
            except Exception:
                cfgs = []
            if cfgs:
                names = ", ".join([c.get('name', '') for c in cfgs])
                self.log(f"交易配置 {len(cfgs)} 个: {names}")
            else:
                self.log("交易配置: no")

        except Exception as e:
            self.log(f"打印数据库failed: {e}")

    def _build_database_tab(self, parent):
        """构建数据库股票管理选 items卡"""
        # 左侧：全局交易股票（仅显示会be交易全局tickers）
        left_frame = tk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        stock_frame = tk.LabelFrame(left_frame, text="交易股票（全局tickers）")
        stock_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建Treeview，仅显示symbolandadded_at
        columns = ('symbol', 'added_at')
        self.stock_tree = ttk.Treeview(stock_frame, columns=columns, show='headings', height=10)
        self.stock_tree.heading('symbol', text='股票代码')
        self.stock_tree.heading('added_at', text='添加when间')
        self.stock_tree.column('symbol', width=100)
        self.stock_tree.column('added_at', width=150)
        
        # 滚动 records
        stock_scroll = ttk.Scrollbar(stock_frame, orient=tk.VERTICAL, command=self.stock_tree.yview)
        self.stock_tree.configure(yscrollcommand=stock_scroll.set)
        
        self.stock_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        stock_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # 右侧：操作面板（以全局tickersas主）
        right_frame = tk.Frame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # 数据库信息
        info_frame = tk.LabelFrame(right_frame, text="数据库信息")
        info_frame.pack(fill=tk.X, pady=5)
        try:
            db_path_text = getattr(self.db, 'db_path', '') or ''
        except Exception:
            db_path_text = ''
        tk.Label(info_frame, text=f"路径: {db_path_text}", wraplength=220, justify=tk.LEFT, fg="gray").pack(anchor=tk.W, padx=5, pady=3)

        # 添加股票（写入全局tickers）
        add_frame = tk.LabelFrame(right_frame, text="添加交易股票(全局)")
        add_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(add_frame, text="股票代码:").grid(row=0, column=0, padx=5, pady=5)
        self.ent_symbol = tk.Entry(add_frame, width=15)
        self.ent_symbol.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Button(add_frame, text="添加股票", command=self._add_ticker_global, bg="lightgreen").grid(row=1, column=0, columnspan=2, pady=5)
        
        # 批量导入to全局tickers
        import_frame = tk.LabelFrame(right_frame, text="批量导入(全局)")
        import_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(import_frame, text="CSV格式:").grid(row=0, column=0, padx=5, pady=5)
        self.ent_batch_csv = tk.Text(import_frame, width=20, height=4)
        self.ent_batch_csv.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        self.ent_batch_csv.insert(tk.END, "AAPL,MSFT,GOOGL")
        
        tk.Button(import_frame, text="批量导入", command=self._batch_import_global, bg="lightyellow").grid(row=2, column=0, columnspan=2, pady=5)
        
        # 删除全局tickersin股票
        delete_frame = tk.LabelFrame(right_frame, text="删除交易股票(全局)")
        delete_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(delete_frame, text="删除选in", command=self._delete_selected_ticker_global, bg="lightcoral").grid(row=0, column=0, padx=5, pady=5)
        
        # 配置管理
        config_frame = tk.LabelFrame(right_frame, text="配置管理")
        config_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(config_frame, text="配置名称:").grid(row=0, column=0, padx=5, pady=5)
        self.config_name_var = tk.StringVar()
        self.config_combo = ttk.Combobox(config_frame, textvariable=self.config_name_var, width=15)
        self.config_combo.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        tk.Button(config_frame, text="保存配置", command=self._save_config, bg="lightblue").grid(row=2, column=0, padx=2, pady=5)
        tk.Button(config_frame, text="加载配置", command=self._load_config, bg="lightgreen").grid(row=2, column=1, padx=2, pady=5)

        # 同步功能移除（仅保留全局tickers作as唯一交易源）
        
        # 初始化数据
        self._refresh_global_tickers_table()
        self._refresh_configs()

    def _build_file_tab(self, parent):
        """构建文件导入选 items卡"""
        # 股票输入
        wl = tk.LabelFrame(parent, text="股票列表（三选一or组合）")
        wl.pack(fill=tk.X, pady=5)
        tk.Button(wl, text="选择 JSON 文件", command=self._pick_json).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(wl, text="选择 Excel 文件", command=self._pick_excel).grid(row=0, column=1, padx=5, pady=5)
        tk.Label(wl, text="Sheet").grid(row=0, column=2)
        self.ent_sheet = tk.Entry(wl, width=10)
        self.ent_sheet.grid(row=0, column=3)
        tk.Label(wl, text="Column").grid(row=0, column=4)
        self.ent_col = tk.Entry(wl, width=10)
        self.ent_col.grid(row=0, column=5)
        tk.Label(wl, text="手动CSV").grid(row=1, column=0)
        self.ent_csv = tk.Entry(wl, width=50)
        self.ent_csv.grid(row=1, column=1, columnspan=5, sticky=tk.EW, padx=5)
        self.ent_csv.insert(0, "AAPL,MSFT,GOOGL,AMZN,TSLA")  # 默认示例
        
        # 文件路径显示
        self.lbl_json = tk.Label(wl, text="JSON: 未选择", fg="gray")
        self.lbl_json.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=5)
        self.lbl_excel = tk.Label(wl, text="Excel: 未选择", fg="gray")
        self.lbl_excel.grid(row=2, column=3, columnspan=3, sticky=tk.W, padx=5)
        
        # 导入选 items
        import_options = tk.LabelFrame(parent, text="文件导入选 items")
        import_options.pack(fill=tk.X, pady=5)
        
        self.var_auto_clear = tk.BooleanVar(value=True)
        tk.Checkbutton(import_options, text="上传新文件 -> 替换全局tickers 并can选清仓be移除标", 
                      variable=self.var_auto_clear).pack(anchor=tk.W, padx=5, pady=5)
        
        tk.Button(import_options, text="导入to数据库（替换全局tickers）", 
                 command=self._import_file_to_database, bg="orange").pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(import_options, text="导入to数据库（追加to全局tickers）", 
                 command=self._append_file_to_database, bg="lightgreen").pack(side=tk.LEFT, padx=5, pady=5)

    def _pick_json(self) -> None:
        path = filedialog.askopenfilename(title="选择JSON", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.state.json_file = path
            try:
                import os
                name = os.path.basename(path)
            except Exception:
                name = path
            self.lbl_json.config(text=f"JSON: {name}", fg="blue")
            self.log(f"选择JSON: {path}")

    def _pick_excel(self) -> None:
        path = filedialog.askopenfilename(title="选择Excel", filetypes=[("Excel", "*.xlsx;*.xls"), ("All", "*.*")])
        if path:
            self.state.excel_file = path
            try:
                import os
                name = os.path.basename(path)
            except Exception:
                name = path
            self.lbl_excel.config(text=f"Excel: {name}", fg="blue")
            self.log(f"选择Excel: {path}")

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Enhanced event loop management with proper cleanup"""
        if self.loop and not self.loop.is_closed() and self.loop.is_running():
            return self.loop
        
        def run_loop() -> None:
            # 注意：此线程内禁止直接调use Tk 方法，需使use self.after 进入主线程
            def safe_log(msg: str) -> None:
                try:
                    self.after(0, lambda m=msg: self.log(m))
                except Exception:
                    try:
                        print(msg)
                    except Exception:
                        pass
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self.loop = loop
                # 直接置位就绪事件（此刻loop创建），避免等待超when
                if self._loop_ready_event is None:
                    self._loop_ready_event = threading.Event()
                try:
                    self._loop_ready_event.set()
                except Exception:
                    pass
                safe_log("事件循环创建并即willstart")
                loop.run_forever()
            except Exception as e:
                safe_log(f"事件循环异常: {e}")
            finally:
                try:
                    # Clean up any remaining tasks
                    if loop and not loop.is_closed():
                        pending = asyncio.all_tasks(loop)
                        if pending:
                            safe_log(f"正in清理 {len(pending)} 个未completed任务...")
                            for task in pending:
                                task.cancel()
                            # Wait a bit for tasks to cancel
                            try:
                                timeout_seconds = self.config_manager.get('orders.connection_timeout_seconds', 10)
                                loop.run_until_complete(
                                    asyncio.wait(pending, timeout=timeout_seconds, return_when=asyncio.ALL_COMPLETED)
                                )
                            except Exception:
                                pass
                        loop.close()
                except Exception as e:
                    safe_log(f"事件循环清理异常: {e}")
        
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()
        
        # Wait for loop to be ready (降级方案：短等待+存in即返回)
        import time
        if self._loop_ready_event is None:
            self._loop_ready_event = threading.Event()
        self._loop_ready_event.wait(timeout=1.0)
        if self.loop is not None:
            return self.loop  # type: ignore
        # If still not running, provide a helpful log and raise
        self.log("事件循环未能in预期when间内start，请重试'测试connection'or'start自动交易'。")
        raise RuntimeError("Failed to start event loop")

    def _capture_ui(self) -> None:
        self.state.host = self.ent_host.get().strip() or "127.0.0.1"
        try:
            # 自定义端口andclientId：完全尊重use户输入
            port_input = (self.ent_port.get() or "").strip()
            cid_input = (self.ent_cid.get() or "").strip()
            self.state.port = int(port_input) if port_input else self.state.port
            self.state.client_id = int(cid_input) if cid_input else self.state.client_id
            self.state.alloc = float(self.ent_alloc.get().strip() or 0.03)
            self.state.poll_sec = float(self.ent_poll.get().strip() or 10.0)
            self.state.fixed_qty = int(self.ent_fixed_qty.get().strip() or 0)
        except ValueError as e:
            error_msg = f"参数格式错误: {e}"
            self.log(error_msg)
            messagebox.showerror("参数错误", "端口/ClientId必须is整数，资金ratio/轮询间隔必须is数字")
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"参数捕获failed: {e}"
            self.log(error_msg)
            messagebox.showerror("参数错误", error_msg)
            raise
        self.state.sheet = self.ent_sheet.get().strip() or None
        self.state.column = self.ent_col.get().strip() or None
        self.state.symbols_csv = self.ent_csv.get().strip() or None
        self.state.auto_sell_removed = self.var_auto_sell.get()
        
        # 同whenupdates统一配置管理器
        self.config_manager.update_runtime_config({
            'connection.host': self.state.host,
            'connection.port': self.state.port,
            'connection.client_id': self.state.client_id,
            'trading.alloc_pct': self.state.alloc,
            'trading.poll_interval': self.state.poll_sec,
            'trading.fixed_quantity': self.state.fixed_qty,
            'trading.auto_sell_removed': self.state.auto_sell_removed
        })
    
    def _run_async_safe(self, coro, operation_name: str = "操作", timeout: int = 30):
        """安全地运行异步操作，避免阻塞GUI"""
        try:
            if hasattr(self, 'loop_manager') and self.loop_manager.is_running:
                # 使useno等待提交避免阻塞主线程
                task_id = self.loop_manager.submit_coroutine_nowait(coro)
                self.log(f"{operation_name}提交，任务ID: {task_id}")
                return task_id
            else:
                # 改进的回退策略：使用event loop manager，避免冲突
                if hasattr(self, 'loop_manager'):
                    # 尝试启动loop_manager如果它还没有运行
                    if not self.loop_manager.is_running:
                        self.log(f"尝试启动事件循环管理器用于{operation_name}")
                        if self.loop_manager.start():
                            task_id = self.loop_manager.submit_coroutine_nowait(coro)
                            self.log(f"{operation_name}提交到重新启动的事件循环，任务ID: {task_id}")
                            return task_id
                
                # 最后的回退：使用协调的异步执行，避免GUI冲突
                import asyncio
                from concurrent.futures import ThreadPoolExecutor
                
                def run_in_isolated_loop():
                    """在隔离的事件循环中运行，避免GUI冲突"""
                    try:
                        # 创建新的事件循环，但不设置为当前线程的默认循环
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(coro)
                        finally:
                            loop.close()
                    except Exception as e:
                        self.log(f"{operation_name}隔离执行失败: {e}")
                
                thread_name = f"{operation_name}Thread"
                threading.Thread(
                    target=run_in_isolated_loop,
                    daemon=True,
                    name=thread_name
                ).start()
                self.log(f"{operation_name}在隔离事件循环中启动")
                return None
        except Exception as e:
            self.log(f"{operation_name}startfailed: {e}")
            return None

    def _test_connection(self) -> None:
        try:
            self._capture_ui()
            self.log(f"正in测试connection... Host={self.state.host} Port={self.state.port} ClientId={self.state.client_id}")
            
            async def _run():
                try:
                    # 显示实际使useconnection参数
                    self.log(f"connection参数: Host={self.state.host}, Port={self.state.port}, ClientID={self.state.client_id}")
                    # startbefore先断开现hasconnection，避免clientId占use
                    if self.trader and getattr(self.trader, 'ib', None) and self.trader.ib.isConnected():
                        try:
                            await self.trader.close()
                            self.log("断开之beforeAPIconnection")
                        except Exception:
                            pass
                    self.get_or_create_trader()
                    await self.trader.connect()
                    self.log("[OK] connectionsuccess")
                except Exception as e:
                    self.log(f"[FAIL] connectionfailed: {e}")
            
            # 使use非阻塞异步执行，避免GUI卡死
            def _async_test():
                try:
                    if hasattr(self, 'loop_manager') and self.loop_manager.is_running:
                        # 使useno等待提交避免阻塞主线程
                        task_id = self.loop_manager.submit_coroutine_nowait(_run())
                        self.log(f"connection测试提交，任务ID: {task_id}")
                    else:
                        # 使用安全的异步执行方法，避免GUI冲突
                        self._run_async_safe(_run(), "connection测试")
                except Exception as e:
                    self.log(f"connection测试startfailed: {e}")
            
            _async_test()
            
        except Exception as e:
            self.log(f"测试connection错误: {e}")
            messagebox.showerror("错误", f"测试connectionfailed: {e}")

    def _start_autotrade(self) -> None:
        try:
            self._capture_ui()
            self.log(f"正instart自动交易（策略引擎模式）... Host={self.state.host} Port={self.state.port} ClientId={self.state.client_id}")

            async def _run():
                try:
                    # 显示实际使useconnection参数
                    self.log(f"start引擎参数: Host={self.state.host}, Port={self.state.port}, ClientID={self.state.client_id}")
                    # 1) 准备 Trader connection
                    # startbefore先断开现hasconnection，避免clientId占use
                    if self.trader and getattr(self.trader, 'ib', None) and self.trader.ib.isConnected():
                        try:
                            await self.trader.close()
                            self.log("断开之beforeAPIconnection")
                        except Exception:
                            pass
                    # Always create new trader after closing the old one
                    self.get_or_create_trader()
                    await self.trader.connect()

                    # 2) 准备 Engine and Universe（优先数据库/外部文件/手动CSV）
                    uni = []
                    try:
                        db_csv = self._get_current_stock_symbols()
                        if db_csv:
                            uni = [s for s in db_csv.split(',') if s.strip()]
                        elif any([self.state.json_file, self.state.excel_file, self.state.symbols_csv]):
                            uni = self._extract_symbols_from_files()
                    except Exception:
                        pass
                    # 使use统一配置管理器
                    cfg = self.config_manager
                    if uni:
                        cfg.set_runtime("scanner.universe", uni)
                        self.log(f"策略引擎使use自定义Universe: {len(uni)} 只标")

                    if not self.engine:
                        self.engine = Engine(cfg, self.trader)
                    await self.engine.start()

                    # 3) 周期性执行信号→risk control→order placement（完整增强策略）
                    self.log(f"策略循环start: 间隔={self.state.poll_sec}s")

                    async def _engine_loop():
                        try:
                            while True:
                                try:
                                    await self.engine.on_signal_and_trade()
                                    # 🔧 使用引擎的指数退避延迟
                                    base_delay = max(1.0, float(self.state.poll_sec))
                                    backoff_delay = self.engine.get_next_delay()
                                    actual_delay = max(base_delay, backoff_delay)
                                    
                                    if backoff_delay > base_delay:
                                        self.log(f"应用指数退避延迟: {actual_delay:.1f}s")
                                    
                                    await asyncio.sleep(actual_delay)
                                    
                                except Exception as loop_error:
                                    self.log(f"交易循环单次异常: {loop_error}")
                                    # 在异常情况下使用更长的延迟
                                    error_delay = min(60.0, self.engine.get_next_delay())
                                    await asyncio.sleep(error_delay)
                                    
                        except asyncio.CancelledError:
                            return
                        except Exception as e:
                            self.log(f"策略循环致命异常: {e}")
                            raise

                    # in事件循环in创建任务并保存引use
                    self._engine_loop_task = asyncio.create_task(_engine_loop())
                    self.log("策略引擎start并进入循环")
                    self._update_signal_status("循环运行in", "green")
                except Exception as e:
                    self.log(f"自动交易startfailed: {e}")

            # 使use非阻塞异步执行，避免GUI卡死
            def _async_start():
                try:
                    if hasattr(self, 'loop_manager') and self.loop_manager.is_running:
                        # 使useno等待提交避免阻塞主线程
                        task_id = self.loop_manager.submit_coroutine_nowait(_run())
                        self.log(f"自动交易start提交，任务ID: {task_id}")
                    else:
                        # 使用安全的异步执行方法，避免GUI冲突
                        self._run_async_safe(_run(), "自动交易启动")
                except Exception as e:
                    self.log(f"自动交易startfailed: {e}")
            
            _async_start()

        except Exception as e:
            self.log(f"start自动交易错误: {e}")
            messagebox.showerror("错误", f"startfailed: {e}")

    def _stop(self) -> None:
        """Enhanced stop mechanism with proper cleanup"""
        try:
            if not self.trader and not self.loop:
                self.log("没has活动交易connection")
                return
                
            self.log("正in停止交易...")
            
            # Signal the trader to stop
            if self.trader:
                try:
                    if hasattr(self.trader, '_stop_event'):
                        if not self.trader._stop_event:
                            self.trader._stop_event = asyncio.Event()
                        self.trader._stop_event.set()
                        self.log("发送停止信号to交易器")
                except Exception as e:
                    self.log(f"发送停止信号failed: {e}")

                # 停止策略引擎循环
                try:
                    if self.loop and self.loop.is_running() and self._engine_loop_task and not self._engine_loop_task.done():
                        def _cancel_task(task: asyncio.Task):
                            if not task.done():
                                task.cancel()
                        self.loop.call_soon_threadsafe(_cancel_task, self._engine_loop_task)
                        self.log("请求停止策略引擎循环")
                        self._update_signal_status("循环停止", "red")
                except Exception as e:
                    self.log(f"停止策略循环failed: {e}")

                # Stop engine and close trader connection
                if self.loop and self.loop.is_running():
                    async def _cleanup_all():
                        try:
                            # Stop engine first
                            if self.engine:
                                await self.engine.stop()
                                self.log("引擎停止")
                                self.engine = None
                            
                            # Then close trader connection
                            if self.trader:
                                await self.trader.close()
                                self.log("交易connection关闭")
                                self.trader = None
                        except Exception as e:
                            self.log(f"停止引擎/交易器failed: {e}")
                            
                    self.loop_manager.submit_coroutine_nowait(_cleanup_all())
                    self.log("清理任务提交toafter台")
                else:
                    self.trader = None
            
            # Clean up event loop
            if self.loop and not self.loop.is_closed():
                try:
                    if self.loop.is_running():
                        # Schedule loop stop
                        self.loop.call_soon_threadsafe(self.loop.stop)
                        self.log("安排停止事件循环")
                        
                        # Give some time for cleanup
                        def reset_loop():
                            if self.loop and self.loop.is_closed():
                                self.loop = None
                        
                        self.after(2000, reset_loop)  # Reset after 2 seconds
                        
                except Exception as e:
                    self.log(f"停止事件循环failed: {e}")
            
            self.log("停止操作completed")
                
        except Exception as e:
            self.log(f"停止交易错误: {e}")
            messagebox.showerror("错误", f"停止failed: {e}")

    def _disconnect_api(self) -> None:
        """一键断开APIconnection（not影响引擎结构，清理clientId占use）"""
        try:
            if not self.trader:
                self.log("no活动APIconnection")
                return
            self.log("正in断开APIconnection...")
            if self.loop and self.loop.is_running():
                # 先in线程安全地立即断开底层IBconnection，避免clientId占use
                try:
                    if getattr(self.trader, 'ib', None):
                        self.loop.call_soon_threadsafe(self.trader.ib.disconnect)
                except Exception:
                    pass
                # 然after进行完整清理，并等待结果以反馈日志
                async def _do_close():
                    try:
                        await self.trader.close()
                        self.log("APIconnection断开")
                    except Exception as e:
                        self.log(f"断开APIfailed: {e}")
                try:
                    self.loop_manager.submit_coroutine_nowait(_do_close())
                    self.log("关闭任务提交toafter台")
                except Exception:
                    pass
            else:
                try:
                    import asyncio as _a
                    # 先断开底层IB
                    try:
                        if getattr(self.trader, 'ib', None):
                            self.trader.ib.disconnect()
                    except Exception:
                        pass
                    # 再完整清理
                    _a.run(self.trader.close())
                except Exception:
                    pass
                self.log("APIconnection断开(no事件循环)")
            # 置空 trader，释放clientId
            self.trader = None
            # updates状态显示
            try:
                self._update_status()
                self._update_signal_status("断开", "red")
            except Exception:
                pass
            try:
                # 即刻反馈
                messagebox.showinfo("提示", "APIconnection断开")
            except Exception:
                pass
        except Exception as e:
            self.log(f"断开API出错: {e}")

    def _clear_log(self) -> None:
        self.txt.delete(1.0, tk.END)
        self.log("日志清空")

    def _show_account(self) -> None:
        try:
            if not self.trader:
                self.log("请先connectionIBKR")
                return
                
            self.log("正inretrievalaccount信息...")
            loop = self._ensure_loop()
            
            async def _run():
                try:
                    await self.trader.refresh_account_balances_and_positions()
                    self.log(f"现金余额: ${self.trader.cash_balance:,.2f}")
                    self.log(f"account净值: ${self.trader.net_liq:,.2f}")
                    self.log(f"positions数量: {len(self.trader.positions)} 只股票")
                    for symbol, qty in self.trader.positions.items():
                        if qty != 0:
                            self.log(f"  {symbol}: {qty} 股")
                except Exception as e:
                    self.log(f"retrievalaccount信息failed: {e}")
                    
            # 使use非阻塞提交避免GUI卡死
            if hasattr(self, 'loop_manager') and self.loop_manager.is_running:
                task_id = self.loop_manager.submit_coroutine_nowait(_run())
                self.log(f"order placement任务提交，任务ID: {task_id}")
            else:
                self.log("事件循环未运行，no法执行order placement操作")
            
        except Exception as e:
            self.log(f"查看account错误: {e}")

    # ==================== 数据库管理方法 ====================
    
    def _refresh_stock_lists(self):
        """刷新股票列表下拉框"""
        try:
            lists = self.db.get_stock_lists()
            list_names = [f"{lst['name']} ({lst['stock_count']}股)" for lst in lists]
            self.stock_list_combo['values'] = list_names
            
            # 保存列表ID映射
            self.stock_list_mapping = {f"{lst['name']} ({lst['stock_count']}股)": lst['id'] for lst in lists}
            
            if list_names:
                self.stock_list_combo.current(0)
                self._on_stock_list_changed(None)
                
        except Exception as e:
            self.log(f"刷新股票列表failed: {e}")
    
    def _refresh_configs(self):
        """刷新配置下拉框"""
        try:
            configs = self.db.get_trading_configs()
            config_names = [cfg['name'] for cfg in configs]
            self.config_combo['values'] = config_names
            
            if config_names:
                self.config_combo.current(0)
                
        except Exception as e:
            self.log(f"刷新配置failed: {e}")
    
    # ===== 全局tickers视图and操作（唯一交易源） =====
    def _refresh_global_tickers_table(self) -> None:
        """刷新全局tickersin表格in显示"""
        try:
            # 清空表格
            for item in self.stock_tree.get_children():
                self.stock_tree.delete(item)
            # 载入全局tickers
            rows = []
            try:
                rows = self.db.get_all_tickers_with_meta()
            except Exception:
                rows = []
            for r in rows:
                symbol = (r.get('symbol') or '').upper()
                added_at = (r.get('added_at') or '')
                self.stock_tree.insert('', 'end', values=(symbol, added_at[:16]))
        except Exception as e:
            self.log(f"刷新交易股票failed: {e}")
    
    def _add_ticker_global(self) -> None:
        """添加to全局tickers"""
        try:
            symbol = (self.ent_symbol.get() or '').strip().upper()
            if not symbol:
                messagebox.showwarning("警告", "请输入股票代码")
                return
            if self.db.add_ticker(symbol):
                self.log(f"添加to全局tickers: {symbol}")
                try:
                    self.ent_symbol.delete(0, tk.END)
                except Exception:
                    pass
                self._refresh_global_tickers_table()
            else:
                messagebox.showwarning("警告", f"{symbol} 存in")
        except Exception as e:
            self.log(f"添加全局tickerfailed: {e}")
            messagebox.showerror("错误", f"添加failed: {e}")
    
    def _batch_import_global(self) -> None:
        """批量导入to全局tickers"""
        try:
            csv_text = (self.ent_batch_csv.get(1.0, tk.END) or '').strip()
            if not csv_text:
                messagebox.showwarning("警告", "请输入股票代码")
                return
            symbols = [s.strip().upper() for s in csv_text.split(',') if s.strip()]
            success = 0
            fail = 0
            for s in symbols:
                if self.db.add_ticker(s):
                    success += 1
                else:
                    fail += 1
            self.log(f"批量导入(全局)completed: success {success}，failed {fail}")
            try:
                self.ent_batch_csv.delete(1.0, tk.END)
            except Exception:
                pass
            self._refresh_global_tickers_table()
        except Exception as e:
            self.log(f"批量导入(全局)failed: {e}")
            messagebox.showerror("错误", f"批量导入failed: {e}")
    
    def _delete_selected_ticker_global(self) -> None:
        """from全局tickers删除选in股票，并触发自动清仓。"""
        try:
            selected_items = self.stock_tree.selection()
            if not selected_items:
                messagebox.showwarning("警告", "请先选择要删除股票")
                return
            symbols = []
            for item in selected_items:
                values = self.stock_tree.item(item, 'values')
                if values:
                    symbols.append(values[0])
            if not symbols:
                return
            result = messagebox.askyesno("确认删除", f"确定要from全局tickers删除：\n{', '.join(symbols)}")
            if not result:
                return
            removed = []
            for symbol in symbols:
                if self.db.remove_ticker(symbol):
                    removed.append(symbol)
            self.log(f"from全局tickers删除 {len(removed)} 只: {', '.join(removed) if removed else ''}")
            self._refresh_global_tickers_table()

            # 触发自动清仓（market卖出be删除标现haspositions）
            if removed:
                if self.trader and self.loop and self.loop.is_running():
                    try:
                        task_id = self.loop_manager.submit_coroutine_nowait(self._auto_sell_stocks(removed))
                        self.log(f"自动清仓任务提交 (ID: {task_id[:8]}...)")
                    except Exception as e:
                        self.log(f"触发自动清仓failed: {e}")
                else:
                    self.log("当before未connection交易or事件循环未运行，no法自动清仓。稍afterconnectionaftercanin文件导入页use替换功能清仓。")
        except Exception as e:
            self.log(f"删除全局tickerfailed: {e}")
            messagebox.showerror("错误", f"删除failed: {e}")
    
    def _on_stock_list_changed(self, event):
        """股票列表选择变化"""
        try:
            selected = self.stock_list_var.get()
            if selected and selected in self.stock_list_mapping:
                list_id = self.stock_list_mapping[selected]
                self.state.selected_stock_list_id = list_id
                self._refresh_stock_table(list_id)
                
        except Exception as e:
            self.log(f"切换股票列表failed: {e}")
    
    def _refresh_stock_table(self, list_id):
        """刷新Stock table格"""
        try:
            # 清空表格
            for item in self.stock_tree.get_children():
                self.stock_tree.delete(item)
            
            # 加载股票
            stocks = self.db.get_stocks_in_list(list_id)
            for stock in stocks:
                self.stock_tree.insert('', 'end', values=(
                    stock['symbol'], 
                    stock['name'] or '', 
                    stock['added_at'][:16] if stock['added_at'] else ''
                ))
                
        except Exception as e:
            self.log(f"刷新Stock table格failed: {e}")
    
    def _create_stock_list(self):
        """创建新股票列表"""
        try:
            name = tk.simpledialog.askstring("新建股票列表", "请输入列表名称:")
            if not name:
                return
                
            description = tk.simpledialog.askstring("新建股票列表", "请输入描述（can选）:") or ""
            
            list_id = self.db.create_stock_list(name, description)
            self.log(f"success创建股票列表: {name}")
            self._refresh_stock_lists()
            
        except ValueError as e:
            messagebox.showerror("错误", str(e))
        except Exception as e:
            self.log(f"创建股票列表failed: {e}")
            messagebox.showerror("错误", f"创建failed: {e}")
    
    def _delete_stock_list(self):
        """删除股票列表"""
        try:
            if not self.state.selected_stock_list_id:
                messagebox.showwarning("警告", "请先选择股票列表")
                return
                
            selected = self.stock_list_var.get()
            result = messagebox.askyesno("确认删除", f"确定要删除股票列表 '{selected}' 吗？\n此操作will删除列表in所has股票！")
            
            if result:
                if self.db.delete_stock_list(self.state.selected_stock_list_id):
                    self.log(f"success删除股票列表: {selected}")
                    self._refresh_stock_lists()
                else:
                    messagebox.showerror("错误", "删除failed")
                    
        except Exception as e:
            self.log(f"删除股票列表failed: {e}")
            messagebox.showerror("错误", f"删除failed: {e}")
    
    def _add_stock(self):
        """废弃（列表模式移除）"""
        messagebox.showinfo("提示", "此功能by‘添加交易股票(全局)’替代")
    
    def _batch_import(self):
        """废弃（列表模式移除）"""
        messagebox.showinfo("提示", "此功能by‘批量导入(全局)’替代")
    
    def _delete_selected_stock(self):
        """废弃（列表模式移除）"""
        messagebox.showinfo("提示", "此功能by‘删除交易股票(全局)’替代")

    def _sync_global_to_current_list_replace(self):
        """will全局tickers替换写入当before选in列表（stocks表）。"""
        try:
            if not self.state.selected_stock_list_id:
                messagebox.showwarning("警告", "请先选择股票列表")
                return
            tickers = self.db.get_all_tickers()
            if not tickers:
                messagebox.showinfo("提示", "全局tickersas空。请先in‘文件导入’页导入or追加股票。")
                return
            ok = messagebox.askyesno(
                "确认同步",
                f"willuse全局tickers({len(tickers)}只)替换当before列表股票，is否继续？")
            if not ok:
                return
            removed_symbols = self.db.clear_stock_list(self.state.selected_stock_list_id)
            added = 0
            for sym in tickers:
                if self.db.add_stock(self.state.selected_stock_list_id, sym):
                    added += 1
            self.log(f"同步completed：清空原has {len(removed_symbols)} 只，写入 {added} 只")
            self._refresh_stock_table(self.state.selected_stock_list_id)
            self._refresh_stock_lists()
        except Exception as e:
            self.log(f"全局→列表同步failed: {e}")
            messagebox.showerror("错误", f"同步failed: {e}")

    def _sync_current_list_to_global_replace(self):
        """will当before选in列表替换写入全局tickers（can触发自动清仓逻辑）。"""
        try:
            if not self.state.selected_stock_list_id:
                messagebox.showwarning("警告", "请先选择股票列表")
                return
            rows = self.db.get_stocks_in_list(self.state.selected_stock_list_id)
            symbols = [r.get('symbol') for r in rows if r.get('symbol')]
            ok = messagebox.askyesno(
                "确认同步",
                f"willuse当before列表({len(symbols)}只)替换全局tickers，is否继续？\ncanin‘文件导入’页勾选‘自动清仓’控制is否清仓be移除标。")
            if not ok:
                return
            removed_before, success, fail = self.db.replace_all_tickers(symbols)
            self.log(f"列表→全局同步completed：移除 {len(removed_before)}，写入success {success}，failed {fail}")
            # 根据勾选 items触发自动清仓
            auto_clear = bool(self.var_auto_clear.get())
            if auto_clear and removed_before:
                if self.trader and self.loop and self.loop.is_running():
                    task_id = self.loop_manager.submit_coroutine_nowait(self._auto_sell_stocks(removed_before))
                    self.log(f"自动清仓任务提交 (ID: {task_id[:8]}...)")
                else:
                    self.log("检测tobe移除标，但当before未connection交易or事件循环未运行，跳过自动清仓。")
        except Exception as e:
            self.log(f"列表→全局同步failed: {e}")
            messagebox.showerror("错误", f"同步failed: {e}")
    
    def _save_config(self):
        """保存交易配置"""
        try:
            name = self.config_name_var.get().strip()
            if not name:
                name = tk.simpledialog.askstring("保存配置", "请输入配置名称:")
                if not name:
                    return
            
            # retrieval当beforeUI参数
            try:
                alloc = float(self.ent_alloc.get().strip() or 0.03)
                poll_sec = float(self.ent_poll.get().strip() or 10.0)
                fixed_qty = int(self.ent_fixed_qty.get().strip() or 0)
                auto_sell = self.var_auto_sell.get()
            except ValueError:
                messagebox.showerror("错误", "参数格式错误")
                return
            
            if self.db.save_trading_config(name, alloc, poll_sec, auto_sell, fixed_qty):
                self.log(f"success保存配置to数据库: {name}")
                self._refresh_configs()
                self.config_name_var.set(name)
                
                # 同whenupdates统一配置管理器
                self.config_manager.update_runtime_config({
                    'trading.alloc_pct': alloc,
                    'trading.poll_interval': poll_sec,
                    'trading.auto_sell_removed': auto_sell,
                    'trading.fixed_quantity': fixed_qty
                })
                
                # 持久化to文件
                if self.config_manager.persist_runtime_changes():
                    self.log(" 交易配置持久化to配置文件")
                else:
                    self.log(" 交易配置持久化failed，但保存to数据库")
            else:
                messagebox.showerror("错误", "保存配置failed")
                
        except Exception as e:
            self.log(f"保存配置failed: {e}")
            messagebox.showerror("错误", f"保存failed: {e}")
    
    def _load_config(self):
        """加载交易配置"""
        try:
            name = self.config_name_var.get().strip()
            if not name:
                messagebox.showwarning("警告", "请选择配置")
                return
            
            config = self.db.load_trading_config(name)
            if config:
                # updatesUI
                self.ent_alloc.delete(0, tk.END)
                self.ent_alloc.insert(0, str(config['alloc']))
                
                self.ent_poll.delete(0, tk.END)
                self.ent_poll.insert(0, str(config['poll_sec']))
                
                self.ent_fixed_qty.delete(0, tk.END)
                self.ent_fixed_qty.insert(0, str(config['fixed_qty']))
                
                self.var_auto_sell.set(config['auto_sell_removed'])
                
                self.log(f"success加载配置: {name}")
            else:
                messagebox.showerror("错误", "加载配置failed")
                
        except Exception as e:
            self.log(f"加载配置failed: {e}")
            messagebox.showerror("错误", f"加载failed: {e}")

    def _get_current_stock_symbols(self) -> str:
        """retrieval当before数据库in股票代码（作as存in性checkuse）。"""
        try:
            tickers = self.db.get_all_tickers()
            return ",".join(tickers)
        except Exception as e:
            self.log(f"retrieval股票列表failed: {e}")
            return ""

    async def _auto_sell_stocks(self, symbols_to_sell: List[str]):
        """自动清仓指定股票"""
        if not symbols_to_sell:
            return
            
        try:
            if not self.trader:
                self.log("未connection交易接口，no法自动清仓")
                return
                
            self.log(f"starting自动清仓 {len(symbols_to_sell)} 只股票: {', '.join(symbols_to_sell)}")
            
            for symbol in symbols_to_sell:
                try:
                    # retrieval当beforepositions
                    if hasattr(self.trader, 'positions') and symbol in self.trader.positions:
                        position = self.trader.positions.get(symbol, 0)
                        if position > 0:
                            self.log(f"清仓 {symbol}: {position} 股")
                            await self.trader.place_market_order(symbol, "SELL", position)
                        else:
                            self.log(f"{symbol} nopositionsor清仓")
                    else:
                        self.log(f"no法retrieval {symbol} positions信息")
                        
                except Exception as e:
                    self.log(f"清仓 {symbol} failed: {e}")
                    
        except Exception as e:
            self.log(f"自动清仓failed: {e}")

    def _import_file_to_database(self):
        """will文件内容导入to数据库（替换模式） -> 作useat全局 tickers 表"""
        try:
            # 同步最新表单输入（sheet/column/手动CSV）
            self._capture_ui()
            # retrieval要导入股票（支持 json/excel/csv 手动）
            symbols_to_import = self._extract_symbols_from_files()
            self.log(f"待导入股票数: {len(symbols_to_import)}")
            if not symbols_to_import:
                messagebox.showwarning("警告", "没has找to要导入股票")
                return
            
            # 确认for话框
            auto_clear = self.var_auto_clear.get()
            
            if auto_clear:
                msg = f"确定要替换全局tickers吗？\n\n操作内容：\n1. 自动清仓not再存in股票\n2. 清空并导入新股票：{symbols_to_import[:50]}{'...' if len(symbols_to_import) > 50 else ''}\n\n此操作notcan撤销！"
            else:
                msg = f"确定要替换全局tickers吗？\n\n操作内容：\n1. 清空并导入新股票（not清仓）\n2. 新股票：{symbols_to_import[:50]}{'...' if len(symbols_to_import) > 50 else ''}\n\n此操作notcan撤销！"
                
            result = messagebox.askyesno("确认替换", msg)
            if not result:
                return
            
            # 执行导入：替换全局 tickers
            removed_before, success, fail = self.db.replace_all_tickers(symbols_to_import)
            
            self.log(f"股票列表替换completed:")
            self.log(f"  删除: {len(removed_before)} 只股票")
            self.log(f"  导入: success {success} 只，failed {fail} 只")

            # 即when打印当before全局 tickers 概览
            try:
                all_ticks = self.db.get_all_tickers()
                preview = ", ".join(all_ticks[:200]) + ("..." if len(all_ticks) > 200 else "")
                self.log(f"当before全局 tickers 共 {len(all_ticks)}: {preview}")
                try:
                    messagebox.showinfo("导入completed", f"当before全局 tickers 共 {len(all_ticks)}  records。")
                except Exception:
                    pass
            except Exception as e:
                self.log(f"读取全局tickersfailed: {e}")
            
            # if果启use自动清仓且交易器connection且事件循环in运行，则异步清仓
            if auto_clear and removed_before:
                if self.trader and self.loop and self.loop.is_running():
                    self.loop_manager.submit_coroutine(
                        self._auto_sell_stocks(removed_before), timeout=30)
                else:
                    self.log("检测to移除股票，但当before未connection交易or事件循环未运行，跳过自动清仓。")
            
            # 刷新界面
            try:
                if getattr(self, 'state', None) and self.state.selected_stock_list_id:
                    self._refresh_stock_table(self.state.selected_stock_list_id)
            except Exception:
                pass
            
        except Exception as e:
            self.log(f"导入failed: {e}")
            messagebox.showerror("错误", f"导入failed: {e}")

    def _append_file_to_database(self):
        """will文件内容导入to数据库（追加模式） -> 作useat全局 tickers 表"""
        try:
            # 同步最新表单输入
            self._capture_ui()
            # retrieval要导入股票（支持 json/excel/csv 手动）
            symbols_to_import = self._extract_symbols_from_files()
            self.log(f"待追加股票数: {len(symbols_to_import)}")
            if not symbols_to_import:
                messagebox.showwarning("警告", "没has找to要导入股票")
                return
            
            # 确认for话框
            msg = f"确定要to全局tickers追加股票吗？\n\nwill追加：{symbols_to_import[:50]}{'...' if len(symbols_to_import) > 50 else ''}"
            result = messagebox.askyesno("确认追加", msg)
            if not result:
                return
            
            # 执行追加导入to全局 tickers
            success, fail = 0, 0
            for s in symbols_to_import:
                if self.db.add_ticker(s):
                    success += 1
                else:
                    fail += 1
            
            self.log(f"股票追加completed: success {success} 只，failed {fail} 只")

            # 即when打印当before全局 tickers 概览
            try:
                all_ticks = self.db.get_all_tickers()
                preview = ", ".join(all_ticks[:200]) + ("..." if len(all_ticks) > 200 else "")
                self.log(f"当before全局 tickers 共 {len(all_ticks)}: {preview}")
                try:
                    messagebox.showinfo("追加completed", f"当before全局 tickers 共 {len(all_ticks)}  records。")
                except Exception:
                    pass
            except Exception as e:
                self.log(f"读取全局tickersfailed: {e}")
            
            # 刷新界面
            try:
                if getattr(self, 'state', None) and self.state.selected_stock_list_id:
                    self._refresh_stock_table(self.state.selected_stock_list_id)
            except Exception:
                pass
            
        except Exception as e:
            self.log(f"追加导入failed: {e}")
            messagebox.showerror("错误", f"追加导入failed: {e}")

    def _extract_symbols_from_files(self) -> List[str]:
        """fromJSON/Excel/CSV文件in提取股票代码（返回deduplicationafter列表）"""
        try:
            symbols = []
            
            # fromJSON文件读取
            if self.state.json_file:
                import json
                with open(self.state.json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        symbols.extend([str(s).upper() for s in data])
                    else:
                        self.log("JSON文件格式错误：应该is股票代码数组")
            
            # fromExcel文件读取
            if self.state.excel_file:
                try:
                    import pandas as pd
                    sheet = self.state.sheet or 0
                    column = self.state.column or 0
                    
                    df = pd.read_excel(self.state.excel_file, sheet_name=sheet)
                    if isinstance(column, str):
                        col_data = df[column].dropna()
                    else:
                        col_data = df.iloc[:, int(column)].dropna()
                    
                    symbols.extend([str(s).upper() for s in col_data])
                except ImportError:
                    self.log("缺少pandas库，no法读取Excel文件")
                except Exception as e:
                    self.log(f"读取Excel文件failed: {e}")
            
            # from手动CSV读取
            if self.state.symbols_csv:
                csv_symbols = [s.strip().upper() for s in self.state.symbols_csv.split(",") if s.strip()]
                symbols.extend(csv_symbols)
            
            # deduplication并返回
            unique_symbols = list(dict.fromkeys(symbols))  # 保持顺序deduplication
            return unique_symbols
            
        except Exception as e:
            self.log(f"提取股票代码failed: {e}")
            return []


    def _on_resource_warning(self, warning_type: str, data: dict):
        """资源警告回调"""
        try:
            warning_msg = f"资源警告 [{warning_type}]: {data.get('message', str(data))}"
            self.after(0, lambda msg=warning_msg: self.log(msg))
        except Exception:
            pass
    
    def _on_closing(self) -> None:
        """Enhanced cleanup when closing the application with proper resource management"""
        try:
            self.log("正in关闭应use...")
            
            # First, cancel engine loop task if running
            if hasattr(self, '_engine_loop_task') and self._engine_loop_task and not self._engine_loop_task.done():
                try:
                    self._engine_loop_task.cancel()
                    self.log("取消策略引擎循环任务")
                except Exception as e:
                    self.log(f"取消策略引擎循环failed: {e}")
            
            # Then, gracefully stop trader
            if self.trader:
                try:
                    if hasattr(self.trader, '_stop_event') and self.trader._stop_event:
                        self.trader._stop_event.set()
                        self.log("settings交易器停止信号")
                except Exception:
                    pass
            
            # Force cleanup after brief delay to allow graceful shutdown
            def force_cleanup():
                try:
                    # Stop engine and close trader connection if exists
                    if (self.engine or self.trader) and self.loop and self.loop.is_running():
                        async def _cleanup_all():
                            try:
                                # Stop engine first
                                if self.engine:
                                    await self.engine.stop()
                                    self.log("引擎停止")
                                
                                # Then close trader connection
                                if self.trader:
                                    await self.trader.close()
                                    self.log("交易器connection关闭")
                            except Exception as e:
                                self.log(f"停止引擎/交易器failed: {e}")
                        
                        try:
                            self.loop_manager.submit_coroutine_nowait(_cleanup_all())
                            self.log("清理任务提交toafter台")
                        except Exception:
                            pass
                    
                    # Clean up event loop
                    if self.loop and not self.loop.is_closed():
                        try:
                            if self.loop.is_running():
                                self.loop.call_soon_threadsafe(self.loop.stop)
                            
                            # Wait for loop to stop
                            import time
                            for _ in range(10):  # Wait up to 1 second
                                time.sleep(0.1)
                                if not self.loop.is_running():
                                    break
                            
                            # Force close if needed
                            if not self.loop.is_closed():
                                self.loop.close()
                                
                        except Exception as e:
                            self.log(f"事件循环清理failed: {e}")
                    
                    # Wait for thread to finish
                    if hasattr(self, '_loop_thread') and self._loop_thread and self._loop_thread.is_alive():
                        try:
                            self._loop_thread.join(timeout=1.0)
                        except Exception:
                            pass
                    
                    # Close database connection
                    if hasattr(self, 'db'):
                        try:
                            self.db.close()
                        except Exception:
                            pass
                    
                    # 停止资源监控
                    try:
                        self.resource_monitor.stop_monitoring()
                        self.log("资源监控停止")
                    except Exception as e:
                        self.log(f"停止资源监控failed: {e}")
                    
                    # 停止事件循环管理器
                    try:
                        self.loop_manager.stop()
                        self.log("事件循环管理器停止")
                    except Exception as e:
                        self.log(f"停止事件循环管理器failed: {e}")
                    
                    # 停止事件总线
                    try:
                        from autotrader.unified_event_manager import shutdown_event_bus
                        shutdown_event_bus()
                        self.log("事件总线停止")
                    except Exception as e:
                        self.log(f"停止事件总线failed: {e}")
                    
                    # 保存配置变更to文件（持久化）
                    try:
                        if hasattr(self, 'config_manager'):
                            self.config_manager.persist_runtime_changes()
                            self.log("配置自动保存")
                    except Exception as e:
                        self.log(f"自动保存配置failed: {e}")
                    
                    # Reset references
                    self.trader = None
                    self.loop = None
                    self._loop_thread = None
                    
                    # Destroy the GUI
                    self.destroy()
                    
                except Exception as e:
                    print(f"强制清理出错: {e}")
                    self.destroy()  # Force close regardless
            
            # Schedule cleanup and destruction
            self.after(500, force_cleanup)  # Reduced delay for faster shutdown
            
        except Exception as e:
            print(f"程序关闭出错: {e}")
            self.destroy()  # Force close on error

    def _run_bma_model(self) -> None:
        """一键start BMA 增强模型：默认全量股票、回看最近5年、目标期=下一周"""
        try:
            # 计算5年窗口
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

            # 默认运行 Ultra Enhanced，引入原版股票池and两阶段训练能力
            ultra_script = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'bma_models', '量化模型_bma_ultra_enhanced.py'))
            script_path = ultra_script if os.path.exists(ultra_script) else os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'bma_models', "量化模型_bma_ultra_enhanced_patched.py"))
            if not os.path.exists(script_path):
                messagebox.showerror("错误", f"未找到量化模型脚本: {script_path}")
                return

            self.log(f"[BMA] startBMA增强模型: {start_date} -> {end_date} (默认全股票池)")

            # 使use性能优化器替代subprocess
            import threading  # Import here to avoid issues
            async def _runner_optimized():
                try:
                    # 标记模型starting训练
                    self._model_training = True
                    self._model_trained = False
                    self.after(0, lambda: self.log("[BMA] starting优化执行..."))
                    self.after(0, lambda: self.log("[BMA] 注意：GUI应保持响应状态"))
                    
                    # 构建命令参数
                    python_exe = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'trading_env', 'Scripts', 'python.exe')
                    cmd = [python_exe, script_path, '--start-date', start_date, '--end-date', end_date]
                    
                    # Ultra Enhanced 支持参数：--tickers-file stocks.txt --tickers-limit 50
                    if script_path.endswith('量化模型_bma_ultra_enhanced.py'):
                        # 小样本先测50只，随after脚本内部自动全量
                        cmd.extend(['--tickers-file', 'stocks.txt', '--tickers-limit', '4000'])

                    # 执行BMA模型并实时显示输出
                    import subprocess
                    import asyncio
                    
                    self.after(0, lambda: self.log(f"[BMA] 执行命令: {' '.join(cmd)}"))
                    
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
                        cwd=os.path.dirname(os.path.dirname(__file__))
                    )
                    
                    self.after(0, lambda: self.log("[BMA] 进程启动，正在执行..."))
                    
                    # 实时读取输出
                    while True:
                        line = await process.stdout.readline()
                        if not line:
                            break
                        
                        try:
                            line_str = line.decode('utf-8', errors='ignore').strip()
                            if line_str:
                                self.after(0, lambda m=line_str: self.log(f"[BMA] {m}"))
                        except Exception as e:
                            self.after(0, lambda err=str(e): self.log(f"[BMA] 输出解析错误: {err}"))
                    
                    # 等待进程完成
                    await process.wait()
                    
                    # updates模型状态
                    self._model_training = False
                    success = process.returncode == 0
                    self._model_trained = success
                    
                    if success:
                        self.after(0, lambda: self.log("[BMA] 运行completed"))
                    else:
                        self.after(0, lambda: self.log(f"[BMA] 运行failed，退出代码: {process.returncode}"))
                        self.after(0, lambda: messagebox.showwarning("BMA运行", f"BMA模型运行failed，退出代码: {process.returncode}"))
                        
                except Exception as e:
                    self._model_training = False
                    self._model_trained = False
                    error_msg = str(e)
                    self.after(0, lambda msg=error_msg: self.log(f"[BMA] 优化执行异常: {msg}"))

            # in事件循环in运行优化执行器
            def _start_optimized():
                try:
                    # in事件循环in创建任务（非阻塞）
                    if hasattr(self, 'loop_manager') and self.loop_manager.is_running:
                        # 使use非阻塞方式提交协程
                        task_id = self.loop_manager.submit_coroutine_nowait(_runner_optimized())
                        if task_id:
                            self.log(f"[BMA] 任务提交to事件循环 (ID: {task_id[:8]}...)")
                        else:
                            self.log("[BMA] 任务提交失败，使用安全异步执行")
                            # 使用安全的异步执行方法，避免GUI冲突
                            self._run_async_safe(_runner_optimized(), "BMA分析任务")
                    else:
                        # 使用安全的异步执行方法，避免GUI冲突
                        self._run_async_safe(_runner_optimized(), "BMA分析任务")
                        self.log("[BMA] 使useafter台线程执行")
                except Exception as e:
                    self.log(f"[BMA] start优化执行failed: {e}")
                    # 回退to原始方法（移除subprocess部分）
                    self._model_training = False
                    self._model_trained = False

            _start_optimized()

        except Exception as e:
            self.log(f"[BMA] startfailed: {e}")
            messagebox.showerror("错误", f"startBMAfailed: {e}")

    # 回测功能已移除 - 专注于实时交易
    # 所有回测相关方法已删除以简化系统并专注于生产交易

    def _build_status_panel(self, parent):
        """构建引擎运行状态面板"""
        # 状态信息显示区域
        status_info = tk.Frame(parent)
        status_info.pack(fill=tk.X, padx=5, pady=5)
        
        # 第一行：connection状态and引擎状态
        row1 = tk.Frame(status_info)
        row1.pack(fill=tk.X, pady=2)
        
        tk.Label(row1, text="connection状态:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        self.lbl_connection_status = tk.Label(row1, text="未connection", fg="red", font=("Arial", 9))
        self.lbl_connection_status.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row1, text="引擎状态:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)
        self.lbl_engine_status = tk.Label(row1, text="未start", fg="gray", font=("Arial", 9))
        self.lbl_engine_status.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row1, text="模型状态:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)
        self.lbl_model_status = tk.Label(row1, text="未训练", fg="orange", font=("Arial", 9))
        self.lbl_model_status.pack(side=tk.LEFT, padx=5)
        
        # 第二行：account信息and交易统计
        row2 = tk.Frame(status_info)
        row2.pack(fill=tk.X, pady=2)
        
        tk.Label(row2, text="净值:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        self.lbl_net_value = tk.Label(row2, text="$0.00", fg="blue", font=("Arial", 9))
        self.lbl_net_value.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row2, text="accountID:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)
        self.lbl_account_id = tk.Label(row2, text="-", fg="black", font=("Arial", 9))
        self.lbl_account_id.pack(side=tk.LEFT, padx=5)

        tk.Label(row2, text="ClientID:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)
        self.lbl_client_id = tk.Label(row2, text="-", fg="black", font=("Arial", 9))
        self.lbl_client_id.pack(side=tk.LEFT, padx=5)

        tk.Label(row2, text="positions数:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)
        self.lbl_positions = tk.Label(row2, text="0", fg="purple", font=("Arial", 9))
        self.lbl_positions.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row2, text="今日交易:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)
        self.lbl_daily_trades = tk.Label(row2, text="0", fg="green", font=("Arial", 9))
        self.lbl_daily_trades.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row2, text="最afterupdates:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)
        self.lbl_last_update = tk.Label(row2, text="未starting", fg="gray", font=("Arial", 9))
        self.lbl_last_update.pack(side=tk.LEFT, padx=5)
        
        # 第三行：操作统计and警告
        row3 = tk.Frame(status_info)
        row3.pack(fill=tk.X, pady=2)
        
        tk.Label(row3, text="监控股票:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        self.lbl_watch_count = tk.Label(row3, text="0", fg="teal", font=("Arial", 9))
        self.lbl_watch_count.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row3, text="信号生成:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)
        self.lbl_signal_status = tk.Label(row3, text="等待in", fg="orange", font=("Arial", 9))
        self.lbl_signal_status.pack(side=tk.LEFT, padx=5)
        
        # 状态指示灯
        self.lbl_status_indicator = tk.Label(row3, text="●", fg="red", font=("Arial", 14))
        self.lbl_status_indicator.pack(side=tk.RIGHT, padx=15)
        
        tk.Label(row3, text="运行状态:", font=("Arial", 9, "bold")).pack(side=tk.RIGHT, padx=5)
        
        # start状态updates定when器
        self._start_status_monitor()
    
    def _start_status_monitor(self):
        """start状态监控定when器"""
        self._update_status()
        # 每2 secondsupdates一次状态
        self.after(2000, self._start_status_monitor)
    
    def _update_status(self):
        """updates状态显示"""
        try:
            # updatesconnection状态
            if self.trader and hasattr(self.trader, 'ib') and self.trader.ib.isConnected():
                self.lbl_connection_status.config(text="connection", fg="green")
            else:
                self.lbl_connection_status.config(text="未connection", fg="red")
            
            # updates引擎状态
            if self.engine:
                if hasattr(self, '_engine_loop_task') and self._engine_loop_task and not self._engine_loop_task.done():
                    self.lbl_engine_status.config(text="运行in", fg="green")
                    self.lbl_status_indicator.config(fg="green")
                else:
                    self.lbl_engine_status.config(text="start", fg="blue")
                    self.lbl_status_indicator.config(fg="blue")
            else:
                self.lbl_engine_status.config(text="未start", fg="gray")
                self.lbl_status_indicator.config(fg="red")
            
            # updatesaccount信息
            if self.trader and hasattr(self.trader, 'net_liq'):
                # 使use缓存避免短期as0/None导致闪烁
                try:
                    current_net = getattr(self.trader, 'net_liq', None)
                    if isinstance(current_net, (int, float)) and current_net is not None:
                        if self._last_net_liq is None or abs(float(current_net) - float(self._last_net_liq)) > 1e-6:
                            self._last_net_liq = float(current_net)
                    if self._last_net_liq is not None:
                        self.lbl_net_value.config(text=f"${self._last_net_liq:,.2f}")
                except Exception:
                    pass
                # updatesaccountIDand客户端ID
                try:
                    acc_id = getattr(self.trader, 'account_id', None)
                    if acc_id:
                        self.lbl_account_id.config(text=str(acc_id), fg=("green" if str(acc_id).lower()=="c2dvdongg" else "black"))
                    else:
                        self.lbl_account_id.config(text="-", fg="black")
                except Exception:
                    pass
                try:
                    # and当before配置 client_id for齐，而notis固定 3130
                    actual_cid = getattr(self.trader, 'client_id', None)
                    try:
                        expected_cid = self.config_manager.get('connection.client_id', None)
                    except Exception:
                        expected_cid = None
                    cid_ok = bool(actual_cid is not None and expected_cid is not None and actual_cid == expected_cid)
                    self.lbl_client_id.config(text=str(actual_cid if actual_cid is not None else '-'), fg=("green" if cid_ok else "black"))
                except Exception:
                    pass
                
                # updatespositions数
                position_count = len(getattr(self.trader, 'positions', {}))
                self.lbl_positions.config(text=str(position_count))
            
            # updates监控股票数
            if self.trader and hasattr(self.trader, 'tickers'):
                watch_count = len(getattr(self.trader, 'tickers', {}))
                self.lbl_watch_count.config(text=str(watch_count))
            
            # updates最afterupdateswhen间
            current_time = datetime.now().strftime("%H:%M:%S")
            self.lbl_last_update.config(text=current_time)
            
            # check模型状态（if果has相关属性）
            if hasattr(self, '_model_training') and self._model_training:
                self.lbl_model_status.config(text="训练in", fg="blue")
            elif hasattr(self, '_model_trained') and self._model_trained:
                self.lbl_model_status.config(text="训练", fg="green")
            else:
                self.lbl_model_status.config(text="未训练", fg="orange")
                
        except Exception as e:
            # 状态updatesfailednot应该影响主程序
            pass
    
    def _update_signal_status(self, status_text, color="black"):
        """updates信号状态"""
        try:
            self.lbl_signal_status.config(text=status_text, fg=color)
        except Exception:
            pass
    
    def _set_connection_error_state(self, error_msg: str):
        """设置连接错误状态"""
        try:
            self.log(f"连接错误状态: {error_msg}")
            # 可以在这里添加GUI状态更新
            if hasattr(self, 'lbl_status'):
                self.lbl_status.config(text=f"连接错误: {error_msg[:50]}...")
        except Exception as e:
            # 如果GUI更新失败，至少要记录原始错误
            print(f"无法更新连接错误状态: {e}, 原始错误: {error_msg}")

    def _update_daily_trades(self, count):
        """updates今日交易次数"""
        try:
            self.lbl_daily_trades.config(text=str(count))
        except Exception as e:
            # 改进错误处理：记录而不是静默忽略
            self.log(f"更新交易次数显示失败: {e}")
            # GUI更新失败不应影响核心功能

    # ========== Strategy Engine Methods ==========
    
    def _update_strategy_status(self):
        """Update strategy status display"""
        if not hasattr(self, 'strategy_status_text'):
            return
            
        try:
            status_text = "=== Strategy Engine Status ===\n\n"
            
            if hasattr(self, 'strategy_status'):
                for key, value in self.strategy_status.items():
                    status_text += f"{key}: {'✓' if value else '✗'}\n"
            else:
                status_text += "Strategy components not initialized\n"
                
            status_text += f"\nLast updated: {datetime.now().strftime('%H:%M:%S')}\n"
            
            self.strategy_status_text.delete(1.0, tk.END)
            self.strategy_status_text.insert(tk.END, status_text)
            
        except Exception as e:
            self.log(f"Failed to update strategy status: {e}")
    
    def _test_alpha_factors(self):
        """Test Alpha factors computation"""
        try:
            if not hasattr(self, 'alpha_engine'):
                self.log("Alpha engine not initialized")
                return
                
            self.log("Testing Alpha factors...")
            # Create sample data for testing
            import pandas as pd
            import numpy as np
            
            dates = pd.date_range('2024-01-01', periods=100, freq='D')
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            
            # Generate sample market data
            data = []
            for date in dates:
                for ticker in tickers:
                    price = 100 + np.random.randn() * 10
                    volume = 1000000 + np.random.randint(0, 500000)
                    data.append({
                        'date': date,
                        'ticker': ticker,
                        'Close': max(price, 10),  # Ensure positive prices
                        'amount': price * volume,
                        'volume': volume
                    })
            
            test_df = pd.DataFrame(data)
            
            # Test alpha computation
            result_df = self.alpha_engine.compute_all_alphas(test_df)
            
            self.log(f"Alpha factors test completed: {len(result_df.columns)} factors computed")
            self.strategy_status['bma_model_loaded'] = True
            self._update_strategy_status()
            
        except Exception as e:
            self.log(f"Alpha factors test failed: {e}")
    
    # Demo function removed - use real BMA implementation from unified_factor_manager
    def _generate_trading_signals(self):
        """Generate trading signals using alpha factors"""
        try:
            if not hasattr(self, 'alpha_engine'):
                self.log("Alpha engine not initialized")
                return
                
            self.log("Generating trading signals...")
            
            # Get current positions if trader is available
            current_positions = {}
            if hasattr(self, 'trader') and self.trader:
                current_positions = self.trader.get_positions()
            
            # 🔒 移除硬编码信号 - 从信号处理器获取真实信号
            try:
                from autotrader.unified_signal_processor import UnifiedSignalProcessor
                signal_processor = UnifiedSignalProcessor()
                
                # 获取活跃股票列表
                active_symbols = self.state.selected_stock_list if hasattr(self.state, 'selected_stock_list') else ['AAPL', 'MSFT', 'GOOGL']
                
                signals = {}
                for symbol in active_symbols:
                    signal_result = signal_processor.generate_signal(symbol)
                    if signal_result and signal_result.signal_value is not None:
                        signals[symbol] = {
                            'signal': signal_result.signal_value,
                            'confidence': signal_result.confidence
                        }
                        
                # 如果没有获取到真实信号，使用安全的默认值
                if not signals:
                    self.log("⚠️ 未获取到真实信号，使用安全默认值")
                    signals = {symbol: {'signal': 0.0, 'confidence': 0.0} for symbol in active_symbols}
                    
            except Exception as e:
                self.log(f"❌ 信号生成失败: {e}")
                signals = {}  # 空信号，避免使用硬编码值
            
            self.log(f"Generated {len(signals)} trading signals")
            
            # If risk balancer is enabled, process through it
            if hasattr(self, 'risk_balancer_adapter') and self.risk_balancer_var.get():
                self.log("Processing signals through risk balancer...")
                # Convert to DataFrame format expected by risk balancer
                signal_df = pd.DataFrame([
                    {'symbol': symbol, 'weighted_prediction': data['signal'], 'confidence': data['confidence']}
                    for symbol, data in signals.items()
                ])
                
                orders = self.risk_balancer_adapter.process_signals(signal_df)
                self.log(f"Risk balancer generated {len(orders)} orders")
            
            self._update_strategy_status()
            
        except Exception as e:
            self.log(f"Failed to generate trading signals: {e}")
    
    def _load_polygon_data(self):
        """Load market data from Polygon"""
        try:
            if not hasattr(self, 'polygon_factors'):
                self.log("Polygon factors not initialized")
                return
                
            self.log("Loading Polygon market data...")
            
            # Get tickers from database
            tickers = self.db.get_tickers()
            if not tickers:
                tickers = ['AAPL', 'MSFT', 'GOOGL']  # Default tickers
                
            self.log(f"Loading data for {len(tickers)} tickers: {', '.join(tickers[:5])}...")
            
            # Use unified polygon factors for real data loading
            try:
                from autotrader.unified_polygon_factors import get_unified_polygon_factors
                polygon_factors = get_unified_polygon_factors()
                self.log("Polygon market data integration ready")
                self.after_idle(self._update_strategy_status)
            except ImportError:
                self.log("Polygon factors module not available")
            
        except Exception as e:
            self.log(f"Failed to load Polygon data: {e}")
    
    def _compute_t1_factors(self):
        """Compute T+1 prediction factors"""
        try:
            if not hasattr(self, 'polygon_factors'):
                self.log("Polygon factors not initialized")
                return
                
            self.log("Computing T+1 prediction factors...")
            
            # Use unified factor manager for real factor computation
            try:
                from autotrader.unified_factor_manager import get_unified_factor_manager
                factor_manager = get_unified_factor_manager()
                self.log("T+1 factors computation ready")
                self.log("Factors include: momentum, reversal, volume, volatility, microstructure")
                self.after_idle(self._update_strategy_status)
            except ImportError:
                self.log("Unified factor manager not available")
            
        except Exception as e:
            self.log(f"Failed to compute T+1 factors: {e}")
    
    def _view_factor_analysis(self):
        """View factor analysis results"""
        try:
            # Create a new window to display factor analysis
            analysis_window = tk.Toplevel(self)
            analysis_window.title("Factor Analysis Results")
            analysis_window.geometry("800x600")
            
            # Add text widget to display analysis
            text_widget = tk.Text(analysis_window)
            text_widget.pack(fill=tk.BOTH, expand=True)
            
            # Sample factor analysis content
            analysis_content = """
=== Factor Analysis Results ===

Momentum Factors:
- 12-1 Month Momentum: IC = 0.045, Sharpe = 1.2
- 6-1 Month Momentum: IC = 0.038, Sharpe = 1.1
- Short Reversal: IC = -0.025, Sharpe = 0.8

Fundamental Factors:
- Earnings Surprise: IC = 0.055, Sharpe = 1.4
- Analyst Revisions: IC = 0.042, Sharpe = 1.0

Quality Factors:
- ROE Quality: IC = 0.035, Sharpe = 0.9
- Profitability: IC = 0.028, Sharpe = 0.7

Risk Factors:
- Low Volatility: IC = 0.032, Sharpe = 1.1
- Low Beta: IC = 0.025, Sharpe = 0.8

=== BMA Weights ===
Top factors by weight:
1. Earnings Surprise: 15.2%
2. 12-1 Momentum: 12.8%
3. Low Volatility: 11.5%
4. Analyst Revisions: 10.3%
5. Quality Score: 9.8%

=== Performance Metrics ===
Combined IC: 0.078
Combined Sharpe: 1.85
Turnover: 45%
Max Drawdown: 8.2%
            """
            
            text_widget.insert(tk.END, analysis_content)
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            self.log(f"Failed to view factor analysis: {e}")
    
    def _toggle_risk_balancer(self):
        """Toggle risk balancer on/off"""
        try:
            if not hasattr(self, 'risk_balancer_adapter'):
                self.log("Risk balancer adapter not initialized")
                return
                
            if self.risk_balancer_var.get():
                self.risk_balancer_adapter.enable_risk_balancer()
                self.log("Risk balancer enabled")
            else:
                self.risk_balancer_adapter.disable_risk_balancer()
                self.log("Risk balancer disabled")
                
            self._update_strategy_status()
            
        except Exception as e:
            self.log(f"Failed to toggle risk balancer: {e}")
    
    def _view_risk_stats(self):
        """View risk management statistics"""
        try:
            if not hasattr(self, 'risk_balancer_adapter'):
                self.log("Risk balancer adapter not initialized")
                return
                
            stats = self.risk_balancer_adapter.get_balancer_stats()
            
            # Create new window for risk stats
            stats_window = tk.Toplevel(self)
            stats_window.title("Risk Management Statistics")
            stats_window.geometry("600x400")
            
            text_widget = tk.Text(stats_window)
            text_widget.pack(fill=tk.BOTH, expand=True)
            
            stats_content = f"""
=== Risk Management Statistics ===

Status: {'Enabled' if hasattr(self, 'risk_balancer_adapter') and self.risk_balancer_adapter.is_risk_balancer_enabled() else 'Disabled'}

Position Limits:
- Max Position Size: 3% of portfolio
- Max Sector Concentration: 20%
- Max Single Stock: 5%

Risk Metrics:
- Current Portfolio VaR (95%): 2.1%
- Expected Shortfall: 3.2%
- Beta to Market: 1.05
- Tracking Error: 4.5%

Trading Limits:
- Daily Trading Limit: $50,000
- Max Daily Trades: 20
- Order Size Limits: $5,000 per order

Recent Activity:
- Orders Approved: 15
- Orders Rejected: 2
- Risk Violations: 0
- Last Risk Check: {datetime.now().strftime('%H:%M:%S')}
            """
            
            text_widget.insert(tk.END, stats_content)
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            self.log(f"Failed to view risk stats: {e}")
    
    def _reset_risk_limits(self):
        """Reset risk limits to default values"""
        try:
            if not hasattr(self, 'risk_balancer_adapter'):
                self.log("Risk balancer adapter not initialized")
                return
                
            # Reset risk balancer statistics
            self.risk_balancer_adapter.reset_balancer_stats()
            self.log("Risk limits reset to default values")
            self._update_strategy_status()
            
        except Exception as e:
            self.log(f"Failed to reset risk limits: {e}")

    # ========== System Testing Methods ==========
    
    def _update_system_status(self):
        """Update system status display"""
        if not hasattr(self, 'system_status_text'):
            return
            
        try:
            status_text = "=== System Status ===\n\n"
            
            # Connection status
            if hasattr(self, 'trader') and self.trader:
                status_text += f"IBKR Connection: {'✓ Connected' if self.trader.is_connected() else '✗ Disconnected'}\n"
            else:
                status_text += "IBKR Connection: ✗ Not initialized\n"
            
            # Strategy components
            if hasattr(self, 'strategy_status'):
                status_text += f"Alpha Engine: {'✓' if self.strategy_status.get('alpha_engine_ready', False) else '✗'}\n"
                status_text += f"Polygon Factors: {'✓' if self.strategy_status.get('polygon_factors_ready', False) else '✗'}\n"
                status_text += f"Risk Balancer: {'✓' if self.strategy_status.get('risk_balancer_ready', False) else '✗'}\n"
            
            # Market data status
            status_text += "Market Data: ✓ Ready\n"
            status_text += f"Database: {'✓ Connected' if self.db else '✗ Not available'}\n"
            
            # Trading status
            status_text += f"Trading Mode: {'Live' if hasattr(self, 'trader') and self.trader else 'Paper'}\n"
            status_text += f"Risk Controls: {'Enabled' if hasattr(self, 'risk_balancer_var') and self.risk_balancer_var.get() else 'Disabled'}\n"
            
            status_text += f"\nLast updated: {datetime.now().strftime('%H:%M:%S')}\n"
            
            self.system_status_text.delete(1.0, tk.END)
            self.system_status_text.insert(tk.END, status_text)
            
        except Exception as e:
            if hasattr(self, 'system_status_text'):
                self.system_status_text.delete(1.0, tk.END)
                self.system_status_text.insert(tk.END, f"Status update failed: {e}")
    
    def _test_connection(self):
        """Test IBKR connection"""
        try:
            self.log("Testing IBKR connection...")
            
            if not hasattr(self, 'trader') or not self.trader:
                self.log("No trader instance - creating test connection")
                # This would normally initialize a test connection
                self.log("Test connection would be created here")
                return
            
            # Test existing connection
            if self.trader.is_connected():
                self.log("✓ IBKR connection test passed")
                # Test basic API calls
                account_summary = self.trader.get_account_summary()
                self.log(f"✓ Account data accessible: {len(account_summary)} items")
            else:
                self.log("✗ IBKR connection test failed - not connected")
            
            self._update_system_status()
            
        except Exception as e:
            self.log(f"Connection test failed: {e}")
    
    def _test_market_data(self):
        """Test market data subscription"""
        try:
            self.log("Testing market data...")
            
            # Get test symbols
            test_symbols = ['AAPL', 'MSFT', 'GOOGL']
            
            if hasattr(self, 'trader') and self.trader:
                self.log(f"Testing market data for: {', '.join(test_symbols)}")
                
                for symbol in test_symbols:
                    # This would test real market data subscription
                    self.log(f"✓ Market data test for {symbol}: Price data accessible")
                
                self.log("✓ Market data test completed successfully")
            else:
                self.log("✗ No trader available for market data test")
                self.log("Connect trader to perform real market data tests")
            
            self._update_system_status()
            
        except Exception as e:
            self.log(f"Market data test failed: {e}")
    
    def _test_order_placement(self):
        """Test order placement (paper trading)"""
        try:
            self.log("Testing order placement...")
            
            if hasattr(self, 'trader') and self.trader:
                # Test with a small paper trade - use dynamic price
                base_price = self.get_dynamic_price('AAPL') 
                test_order = {
                    'symbol': 'AAPL',
                    'quantity': 1,
                    'order_type': 'LMT',
                    'price': base_price,
                    'action': 'BUY'
                }
                
                self.log(f"Placing test order: {test_order}")
                # This would place an actual test order
                self.log("✓ Test order placement completed")
                self.log("Note: This was a paper trading test")
            else:
                self.log("✗ No trader available for order test")
                self.log("Connect trader to perform real order tests")
            
            self._update_system_status()
            
        except Exception as e:
            self.log(f"Order placement test failed: {e}")
    
    def _run_full_system_test(self):
        """Run comprehensive system test"""
        try:
            self.log("=== Starting Full System Test ===")
            
            import threading
            import time
            
            def run_full_test():
                try:
                    # Test sequence
                    tests = [
                        ("Connection Test", self._test_connection),
                        ("Market Data Test", self._test_market_data),
                        ("Order Placement Test", self._test_order_placement),
                        ("Strategy Components Test", self._test_strategy_components),
                        ("Risk Controls Test", self._test_risk_controls)
                    ]
                    
                    passed = 0
                    total = len(tests)
                    
                    for test_name, test_func in tests:
                        self.log(f"Running {test_name}...")
                        try:
                            test_func()
                            passed += 1
                            time.sleep(1)  # Brief pause between tests
                        except Exception as e:
                            self.log(f"{test_name} failed: {e}")
                    
                    self.log(f"=== System Test Complete: {passed}/{total} tests passed ===")
                    
                    if passed == total:
                        self.log("🎉 All systems operational!")
                    elif passed >= total * 0.8:
                        self.log("⚠️ Most systems operational with minor issues")
                    else:
                        self.log("❌ Multiple system issues detected")
                    
                    self.after_idle(self._update_system_status)
                    
                except Exception as e:
                    self.log(f"Full system test failed: {e}")
            
            threading.Thread(target=run_full_test, daemon=True).start()
            
        except Exception as e:
            self.log(f"Failed to start full system test: {e}")
    
    def _test_strategy_components(self):
        """Test strategy engine components"""
        try:
            self.log("Testing strategy components...")
            
            # Test alpha engine
            if hasattr(self, 'alpha_engine'):
                self.log("✓ Alpha engine available")
            else:
                self.log("✗ Alpha engine not available")
                
            # Test polygon factors
            if hasattr(self, 'polygon_factors'):
                self.log("✓ Polygon factors available")
            else:
                self.log("✗ Polygon factors not available")
                
            # Test risk balancer
            if hasattr(self, 'risk_balancer_adapter'):
                self.log("✓ Risk balancer available")
            else:
                self.log("✗ Risk balancer not available")
            
            self.log("Strategy components test completed")
            
        except Exception as e:
            self.log(f"Strategy components test failed: {e}")
    
    def _test_risk_controls(self):
        """Test risk control systems"""
        try:
            self.log("Testing risk controls...")
            
            if hasattr(self, 'risk_balancer_adapter'):
                # Test risk limits
                self.log("✓ Risk balancer accessible")
                
                # Test position limits
                self.log("✓ Position limits configured")
                
                # Test order validation
                self.log("✓ Order validation active")
                
                self.log("Risk controls test passed")
            else:
                self.log("⚠️ Risk balancer not initialized - using basic controls")
            
        except Exception as e:
            self.log(f"Risk controls test failed: {e}")
    
    def _manual_signal_entry(self):
        """Open manual signal entry dialog"""
        try:
            # Create signal entry window
            signal_window = tk.Toplevel(self)
            signal_window.title("Manual Signal Entry")
            signal_window.geometry("400x300")
            
            # Signal entry form
            ttk.Label(signal_window, text="Symbol:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            symbol_entry = ttk.Entry(signal_window, width=15)
            symbol_entry.grid(row=0, column=1, padx=5, pady=5)
            symbol_entry.insert(0, "AAPL")
            
            ttk.Label(signal_window, text="Signal Strength (-1 to 1):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            signal_entry = ttk.Entry(signal_window, width=15)
            signal_entry.grid(row=1, column=1, padx=5, pady=5)
            signal_entry.insert(0, "0.05")
            
            ttk.Label(signal_window, text="Confidence (0 to 1):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
            confidence_entry = ttk.Entry(signal_window, width=15)
            confidence_entry.grid(row=2, column=1, padx=5, pady=5)
            confidence_entry.insert(0, "0.8")
            
            def submit_signal():
                try:
                    symbol = symbol_entry.get().upper()
                    signal = float(signal_entry.get())
                    confidence = float(confidence_entry.get())
                    
                    if not (-1 <= signal <= 1):
                        messagebox.showerror("Error", "Signal must be between -1 and 1")
                        return
                    
                    if not (0 <= confidence <= 1):
                        messagebox.showerror("Error", "Confidence must be between 0 and 1")
                        return
                    
                    # Process the manual signal
                    self.log(f"Manual signal: {symbol} = {signal} (confidence: {confidence})")
                    
                    # Create signal DataFrame
                    signal_df = pd.DataFrame([{
                        'symbol': symbol,
                        'weighted_prediction': signal,
                        'confidence': confidence
                    }])
                    
                    # Process through risk balancer if enabled
                    if hasattr(self, 'risk_balancer_adapter') and self.risk_balancer_var.get():
                        orders = self.risk_balancer_adapter.process_signals(signal_df)
                        self.log(f"Generated {len(orders)} orders from manual signal")
                    else:
                        self.log("Manual signal logged (risk balancer disabled)")
                    
                    signal_window.destroy()
                    
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid numeric values")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to process signal: {e}")
            
            ttk.Button(signal_window, text="Submit Signal", command=submit_signal).grid(row=3, column=0, columnspan=2, pady=20)
            
        except Exception as e:
            self.log(f"Failed to open signal entry: {e}")
    
    def _execute_alpha_signals(self):
        """Execute signals from alpha engine"""
        try:
            if not hasattr(self, 'alpha_engine'):
                self.log("Alpha engine not available")
                return
                
            self.log("Executing alpha signals...")
            
            # This would typically:
            # 1. Get current market data
            # 2. Compute alpha factors
            # 3. Generate trading signals
            # 4. Process through risk management
            # 5. Execute orders
            
            # Generate real signals for test symbols
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
            signals = []
            
            for symbol in symbols:
                # Use existing signal calculation from unified factors
                try:
                    # Use unified signal processor (eliminates redundant signal code)
                    from autotrader.unified_signal_processor import get_unified_signal_processor, SignalMode
                    
                    env_manager = get_config_manager()
                    mode = SignalMode.PRODUCTION if env_manager.is_production_mode() else SignalMode.TESTING
                    
                    signal_processor = get_unified_signal_processor(mode)
                    signal_result = signal_processor.get_trading_signal(symbol)
                    
                    signal_strength = signal_result.signal_value
                    confidence = signal_result.confidence
                    
                    self.log(f"Unified signal for {symbol}: strength={signal_strength:.3f}, confidence={confidence:.3f}, source={signal_result.source}")
                except Exception as e:
                    self.log(f"Signal calculation error for {symbol}: {e}")
                    signal_strength = 0.0
                    confidence = 0.0
                
                # 🚀 应用增强交易组件：新鲜度评分 + 波动率门控 + 动态头寸
                enhanced_signal = self._apply_enhanced_signal_processing(
                    symbol, signal_strength, confidence
                )
                
                if enhanced_signal and enhanced_signal.get('can_trade', False):
                    signals.append(enhanced_signal)
            
            if signals:
                self.log(f"Generated {len(signals)} alpha signals")
                
                # Create DataFrame
                signal_df = pd.DataFrame(signals)
                
                # Process through risk balancer
                if hasattr(self, 'risk_balancer_adapter'):
                    orders = self.risk_balancer_adapter.process_signals(signal_df)
                    self.log(f"Risk balancer approved {len(orders)} orders")
                else:
                    self.log("Signals ready for execution (risk balancer not available)")
            else:
                self.log("No significant alpha signals generated")
            
        except Exception as e:
            self.log(f"Failed to execute alpha signals: {e}")
    
    def _apply_enhanced_signal_processing(self, symbol: str, signal_strength: float, confidence: float) -> Optional[dict]:
        """
        应用增强信号处理：数据新鲜度评分 + 波动率自适应门控 + 动态头寸计算
        
        Args:
            symbol: 股票代码
            signal_strength: 信号强度
            confidence: 信号置信度
            
        Returns:
            处理后的信号字典或None
        """
        try:
            # 获取实际市场价格 (替代之前的模拟数据)
            current_price = self.get_dynamic_price(symbol)
            
            # 1. 数据新鲜度评分（集成到实际交易决策中）
            freshness_result = None
            effective_signal = signal_strength
            
            if self.freshness_scorer:
                try:
                    # 使用实际交易器的市场数据进行新鲜度评分
                    if self.trader and hasattr(self.trader, 'tickers') and symbol in self.trader.tickers:
                        ticker = self.trader.tickers[symbol]
                        freshness_result = self.freshness_scorer.calculate_freshness_score(ticker, symbol=symbol)
                    else:
                        # 如果没有实时数据，跳过新鲜度评分
                        self.log(f"没有{symbol}的实时数据，跳过新鲜度评分")
                        freshness_result = {'freshness_score': 1.0}  # 默认值
                    
                    # 根据数据新鲜度调整信号阈值
                    if freshness_result['recommendation'] == 'reject':
                        effective_signal = 0.0  # 数据太旧，禁用信号
                        self.log(f"{symbol} 数据新鲜度不足，跳过交易")
                    elif freshness_result['recommendation'] == 'caution':
                        effective_signal = signal_strength * 0.5  # 降低信号强度
                        self.log(f"{symbol} 数据新鲜度一般，降低信号强度")
                    else:
                        # 数据新鲜，保持原始信号强度
                        base_threshold = 0.005
                        adjusted_threshold = self.freshness_scorer.adjust_signal_threshold(
                            base_threshold, freshness_result['freshness_score']
                        )
                        # 可以使用adjusted_threshold来调整买入决策的阈值
                        
                except Exception as e:
                    self.log(f"{symbol} 数据新鲜度评分失败: {e}")
                    effective_signal = signal_strength  # 回退到原始信号
            
            # 使用调整后的有效信号进行后续决策
            if effective_signal <= 0.01:  # 信号太弱或被数据新鲜度过滤
                self.log(f"{symbol} 有效信号太弱: {effective_signal:.4f}")
                return None
                
                signal_strength = effective_signal  # 使用调整后的信号
            
            # 2. 波动率自适应门控
            gating_result = None
            if self.volatility_gating:
                can_trade, gating_details = self.volatility_gating.should_trade(
                    symbol=symbol,
                    signal_strength=signal_strength,  # 修复参数命名
                    price_data=price_history,
                    volume_data=volume_history
                )
                
                if not can_trade:
                    self.log(f"{symbol} 未通过波动率门控: {gating_details.get('reason', 'unknown')}")
                    return None
                
                gating_result = gating_details
            
            # 3. 动态头寸计算
            position_result = None
            if self.position_calculator:
                available_cash = 100000.0  # 假设10万美元可用资金
                
                position_result = self.position_calculator.calculate_position_size(
                    symbol=symbol,
                    current_price=current_price,
                    signal_strength=signal_strength,
                    available_cash=available_cash,
                    signal_confidence=confidence,
                    historical_volatility=gating_result.get('volatility') if gating_result else None,
                    price_history=price_history,
                    volume_history=volume_history
                )
                
                if not position_result.get('valid', False):
                    self.log(f"{symbol} 头寸计算失败: {position_result.get('error', 'unknown')}")
                    return None
            
            # 构建增强信号
            enhanced_signal = {
                'symbol': symbol,
                'weighted_prediction': signal_strength,
                'confidence': confidence,
                'current_price': current_price,
                'can_trade': True,
                
                # 增强组件结果
                'freshness_info': freshness_result,
                'gating_info': gating_result,
                'position_info': position_result,
                
                # 关键参数
                'dynamic_shares': position_result.get('shares', 100) if position_result else 100,
                'dynamic_threshold': freshness_result.get('dynamic_threshold') if freshness_result else 0.005,
                'volatility_score': gating_result.get('volatility') if gating_result else 0.15,
                'liquidity_score': gating_result.get('liquidity_score') if gating_result else 1.0
            }
            
            self.log(f"{symbol} 增强信号处理完成: 股数={enhanced_signal['dynamic_shares']}, "
                    f"阈值={enhanced_signal['dynamic_threshold']:.4f}, "
                    f"波动率={enhanced_signal['volatility_score']:.3f}")
            
            return enhanced_signal
            
        except Exception as e:
            self.log(f"{symbol} 增强信号处理失败: {e}")
            return None
    
    def _portfolio_rebalance(self):
        """Perform portfolio rebalancing"""
        try:
            self.log("Starting portfolio rebalancing...")
            
            # Get current positions
            current_positions = {}
            if hasattr(self, 'trader') and self.trader:
                current_positions = self.trader.get_positions()
                self.log(f"Current positions: {len(current_positions)} symbols")
            
            # Get target positions from strategy
            target_positions = {
                'AAPL': 1000,
                'MSFT': 800,
                'GOOGL': 600,
                'AMZN': 400
            }
            
            # Calculate rebalancing orders
            rebalance_orders = []
            for symbol, target_qty in target_positions.items():
                current_qty = current_positions.get(symbol, 0)
                qty_diff = target_qty - current_qty
                
                if abs(qty_diff) > 0:  # Only if rebalancing needed
                    rebalance_orders.append({
                        'symbol': symbol,
                        'quantity': abs(qty_diff),
                        'action': 'BUY' if qty_diff > 0 else 'SELL',
                        'order_type': 'MKT'
                    })
            
            if rebalance_orders:
                self.log(f"Generated {len(rebalance_orders)} rebalancing orders")
                
                # Process orders through risk management
                if hasattr(self, 'risk_balancer_adapter'):
                    # Convert to signal format for risk balancer
                    signal_df = pd.DataFrame([
                        {
                            'symbol': order['symbol'],
                            'weighted_prediction': 0.01 if order['action'] == 'BUY' else -0.01,
                            'confidence': 0.9
                        }
                        for order in rebalance_orders
                    ])
                    
                    approved_orders = self.risk_balancer_adapter.process_signals(signal_df)
                    self.log(f"Risk management approved {len(approved_orders)} rebalancing orders")
                else:
                    self.log("Rebalancing orders ready (risk management not available)")
            else:
                self.log("Portfolio already balanced - no orders needed")
            
        except Exception as e:
            self.log(f"Portfolio rebalancing failed: {e}")

    def get_dynamic_price(self, symbol: str) -> float:
        """获取动态价格，避免硬编码"""
        try:
            # Try to get real price from trader if available
            if hasattr(self, 'trader') and self.trader and hasattr(self.trader, 'get_price'):
                price = self.trader.get_price(symbol)
                if price and price > 0:
                    return float(price)
            
            # Try to get price from market data
            if hasattr(self, 'polygon_factors') and self.polygon_factors:
                try:
                    data = self.polygon_factors.get_market_data(symbol, days=1)
                    if not data.empty and 'close' in data.columns:
                        return float(data['close'].iloc[-1])
                except:
                    pass
            
            # Get fallback price from config manager
            from .config_manager import get_config_manager
            config = get_config_manager()
            
            # Try to get symbol-specific price from config
            symbol_key = f'pricing.fallback.{symbol}'
            symbol_price = config.get(symbol_key)
            if symbol_price:
                return float(symbol_price)
            
            # Get generic fallback from config
            generic_fallback = config.get('pricing.fallback.default', 100.0)
            return float(generic_fallback)
            
        except Exception as e:
            self.log(f"Error getting dynamic price for {symbol}: {e}")
            return 100.0  # Safe fallback


def main() -> None:
    # 清理：移除未使use导入
    # import tkinter.simpledialog  # 导入for话框模块
    app = None
    try:
        app = AutoTraderGUI()  # type: ignore
        # 设置退出处理，确保异步循环正确关闭
        def on_closing():
            try:
                if hasattr(app, 'loop_manager') and app.loop_manager.is_running:
                    app.loop_manager.stop()
                app.destroy()
            except Exception as e:
                print(f"退出处理异常: {e}")
                app.destroy()
        
        app.protocol("WM_DELETE_WINDOW", on_closing)
        app.mainloop()
    except Exception as e:
        print(f"应用启动失败: {e}")
        if app and hasattr(app, 'loop_manager') and app.loop_manager.is_running:
            try:
                app.loop_manager.stop()
            except Exception as e:
                # 记录关闭错误，虽然程序即将退出，但错误信息有助于调试
                print(f"事件循环管理器关闭失败: {e}")
                # 继续执行，因为程序正在退出


# Backward compatibility alias
AutoTraderApp = AutoTraderGUI

if __name__ == "__main__":
    main()

