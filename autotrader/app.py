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

from .ibkr_auto_trader import IbkrAutoTrader
from .engine import Engine
from .database import StockDatabase


@dataclass
class AppState:
    json_file: Optional[str] = None
    excel_file: Optional[str] = None
    sheet: Optional[str] = None
    column: Optional[str] = None
    symbols_csv: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 3130
    # 交易参数
    alloc: float = 0.03
    poll_sec: float = 10.0
    auto_sell_removed: bool = True
    fixed_qty: int = 0
    # 数据库相关
    selected_stock_list_id: Optional[int] = None
    use_database: bool = True


class AutoTraderGUI(tk.Tk):
    def __init__(self) -> None:  # type: ignore
        super().__init__()
        
        # 使用统一的配置管理器
        from autotrader.unified_config import get_unified_config
        from autotrader.event_loop_manager import get_event_loop_manager
        from autotrader.resource_monitor import get_resource_monitor
        
        self.config_manager = get_unified_config()
        self.loop_manager = get_event_loop_manager()
        self.resource_monitor = get_resource_monitor()
        
        # 启动事件循环管理器
        if not self.loop_manager.start():
            raise RuntimeError("无法启动事件循环管理器")
        
        # 启动资源监控
        self.resource_monitor.start_monitoring()
        
        # 初始化AppState使用统一配置
        conn_params = self.config_manager.get_connection_params()
        self.state = AppState(
            port=conn_params['port'],
            client_id=conn_params['client_id'],
            host=conn_params['host']
        )
        self.title("IBKR 自动交易控制台")
        self.geometry("1000x700")
        # 使用项目内固定路径的数据目录，避免当前工作目录变化导致丢失
        self.db = StockDatabase()
        # 提前初始化日志相关对象，避免在UI尚未构建完成前调用log引发属性错误
        self._log_buffer: List[str] = []
        self.txt = None  # type: ignore
        self._build_ui()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.trader: Optional[IbkrAutoTrader] = None
        self.engine: Optional[Engine] = None
        # 已改用统一配置管理器，不再需要HotConfig
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
        from autotrader.event_system import get_event_bus, GUIEventAdapter
        self.event_bus = get_event_bus()
        self.gui_adapter = GUIEventAdapter(self, self.event_bus)

    def _build_ui(self) -> None:
        frm = tk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 连接参数
        con = tk.LabelFrame(frm, text="连接设置")
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

        # 创建笔记本选项卡
        notebook = ttk.Notebook(frm)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 数据库股票管理选项卡
        db_frame = ttk.Frame(notebook)
        notebook.add(db_frame, text="数据库股票管理")
        self._build_database_tab(db_frame)
        
        # 文件导入选项卡
        file_frame = ttk.Frame(notebook)
        notebook.add(file_frame, text="文件导入")
        self._build_file_tab(file_frame)

        # 风险管理选项卡
        risk_frame = ttk.Frame(notebook)
        notebook.add(risk_frame, text="风险管理")
        self._build_risk_tab(risk_frame)

        # Polygon风控收益平衡器选项卡
        polygon_frame = ttk.Frame(notebook)
        notebook.add(polygon_frame, text="Polygon风控")
        self._build_polygon_tab(polygon_frame)

        # 策略引擎选项卡（集成模式2）
        engine_frame = ttk.Frame(notebook)
        notebook.add(engine_frame, text="策略引擎")
        self._build_engine_tab(engine_frame)

        # 直接交易选项卡（集成模式3）
        direct_frame = ttk.Frame(notebook)
        notebook.add(direct_frame, text="直接交易")
        self._build_direct_tab(direct_frame)

        # 回测分析选项卡
        backtest_frame = ttk.Frame(notebook)
        notebook.add(backtest_frame, text="回测分析")
        self._build_backtest_tab(backtest_frame)

        # 交易参数设置
        params = tk.LabelFrame(frm, text="交易参数设置")
        params.pack(fill=tk.X, pady=5)
        
        # 第一行：资金分配和轮询间隔
        tk.Label(params, text="每股资金占比").grid(row=0, column=0, padx=5, pady=5)
        self.ent_alloc = tk.Entry(params, width=8)
        self.ent_alloc.insert(0, str(self.state.alloc))
        self.ent_alloc.grid(row=0, column=1, padx=5)
        
        tk.Label(params, text="轮询间隔(秒)").grid(row=0, column=2, padx=5)
        self.ent_poll = tk.Entry(params, width=8)
        self.ent_poll.insert(0, str(self.state.poll_sec))
        self.ent_poll.grid(row=0, column=3, padx=5)
        
        tk.Label(params, text="固定股数").grid(row=0, column=4, padx=5)
        self.ent_fixed_qty = tk.Entry(params, width=8)
        self.ent_fixed_qty.insert(0, str(self.state.fixed_qty))
        self.ent_fixed_qty.grid(row=0, column=5, padx=5)
        
        # 第二行：自动清仓选项
        self.var_auto_sell = tk.BooleanVar(value=self.state.auto_sell_removed)
        tk.Checkbutton(params, text="移除股票时自动清仓", variable=self.var_auto_sell).grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        
        # 动作按钮
        act = tk.LabelFrame(frm, text="操作")
        act.pack(fill=tk.X, pady=5)
        tk.Button(act, text="测试连接", command=self._test_connection, bg="lightblue").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="断开API连接", command=self._disconnect_api, bg="#ffcccc").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="启动自动交易", command=self._start_autotrade, bg="lightgreen").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="停止交易", command=self._stop, bg="orange").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="清空日志", command=self._clear_log, bg="lightgray").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="查看账户", command=self._show_account, bg="lightyellow").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="一键运行BMA模型", command=self._run_bma_model, bg="#d8b7ff").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="打印数据库", command=self._print_database, bg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="一键删除数据库", command=self._delete_database, bg="#ff6666").pack(side=tk.RIGHT, padx=5)

        # 运行状态告示栏
        status_frame = tk.LabelFrame(frm, text="引擎运行状态")
        status_frame.pack(fill=tk.X, pady=5)
        self._build_status_panel(status_frame)
        
        # 日志
        log_frame = tk.LabelFrame(frm, text="运行日志")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.txt = tk.Text(log_frame, height=8)
        self.txt.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # 将缓冲区中的日志刷新到界面
        try:
            if getattr(self, "_log_buffer", None):
                for _line in self._log_buffer:
                    self.txt.insert(tk.END, _line + "\n")
                self.txt.see(tk.END)
                self._log_buffer.clear()
        except Exception:
            pass

    def log(self, msg: str) -> None:
        # 同时输出到控制台和GUI
        try:
            print(msg)  # 输出到终端控制台
        except UnicodeEncodeError:
            # Windows控制台中文编码问题的备选方案
            print(msg.encode('gbk', errors='ignore').decode('gbk', errors='ignore'))
        except Exception:
            # 如果控制台输出失败，至少确保GUI日志还能工作
            pass
        
        # UI尚未完成或Text尚未创建时，先写入缓冲区
        try:
            if hasattr(self, "txt") and isinstance(self.txt, tk.Text):
                self.txt.insert(tk.END, msg + "\n")
                self.txt.see(tk.END)
            else:
                # 可能在构建UI早期被调用
                if not hasattr(self, "_log_buffer"):
                    self._log_buffer = []  # type: ignore
                self._log_buffer.append(msg)  # type: ignore
        except Exception:
            # 即便日志失败也不影响主流程
            try:
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
        ttk.Label(box1, text="实时信号分配 %").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.rm_rt_alloc = ttk.Spinbox(box1, from_=0.0, to=1.0, increment=0.01, width=8)
        self.rm_rt_alloc.set(0.03)
        self.rm_rt_alloc.grid(row=0, column=5, padx=5)

        box2 = ttk.LabelFrame(frm, text="风控与资金")
        box2.pack(fill=tk.X, pady=5)
        ttk.Label(box2, text="价格下限").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.rm_price_min = ttk.Spinbox(box2, from_=0.0, to=1000.0, increment=0.5, width=8)
        self.rm_price_min.set(2.0)
        self.rm_price_min.grid(row=0, column=1, padx=5)
        ttk.Label(box2, text="价格上限").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.rm_price_max = ttk.Spinbox(box2, from_=0.0, to=5000.0, increment=1.0, width=8)
        self.rm_price_max.set(800.0)
        self.rm_price_max.grid(row=0, column=3, padx=5)
        ttk.Label(box2, text="现金预留 %").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.rm_cash_reserve = ttk.Spinbox(box2, from_=0.0, to=0.9, increment=0.01, width=8)
        self.rm_cash_reserve.set(0.15)
        self.rm_cash_reserve.grid(row=1, column=1, padx=5)
        ttk.Label(box2, text="单标的上限 %").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.rm_single_max = ttk.Spinbox(box2, from_=0.01, to=0.9, increment=0.01, width=8)
        self.rm_single_max.set(0.12)
        self.rm_single_max.grid(row=1, column=3, padx=5)
        ttk.Label(box2, text="最小下单 $").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
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
        ttk.Checkbutton(box3, text="使用ATR动态止损", variable=self.rm_use_atr).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
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
        ttk.Checkbutton(box3, text="移除平仓使用括号单(不推荐)", variable=self.rm_bracket_removed).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

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
            self.log("已加载风险配置")
        except Exception as e:
            self.log(f"加载风险配置失败: {e}")

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
                self.log("风险配置已保存到数据库")
            else:
                self.log("风险配置保存失败")
            db.close()
            
            # 同时更新统一配置管理器并持久化
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
            
            # 持久化到文件
            if self.config_manager.persist_runtime_changes():
                self.log("✅ 风险配置已持久化到配置文件")
            else:
                self.log("⚠️ 风险配置持久化失败，但已保存到数据库")
        except Exception as e:
            self.log(f"保存风险配置失败: {e}")

    def _build_polygon_tab(self, parent) -> None:
        """构建Polygon风控收益平衡器选项卡"""
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Polygon连接状态
        status_frame = ttk.LabelFrame(frm, text="Polygon连接状态")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.polygon_status_label = tk.Label(status_frame, text="状态: 检查中...", fg="gray")
        self.polygon_status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        ttk.Button(status_frame, text="刷新状态", command=self._update_polygon_status).pack(side=tk.RIGHT, padx=10, pady=5)

        # Polygon因子控制
        factor_frame = ttk.LabelFrame(frm, text="Polygon因子控制")
        factor_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(factor_frame, text="启用Polygon因子", command=self._enable_polygon_factors).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(factor_frame, text="清理因子缓存", command=self._clear_polygon_cache).grid(row=0, column=1, padx=5, pady=5)

        # 风控收益平衡器控制
        balancer_frame = ttk.LabelFrame(frm, text="风控收益平衡器控制")
        balancer_frame.pack(fill=tk.X, pady=5)
        
        # 一键开启/关闭
        control_row = ttk.Frame(balancer_frame)
        control_row.pack(fill=tk.X, padx=5, pady=5)
        
        self.polygon_balancer_var = tk.BooleanVar()
        self.polygon_balancer_check = ttk.Checkbutton(
            control_row, 
            text="启用风控收益平衡器", 
            variable=self.polygon_balancer_var,
            command=self._toggle_polygon_balancer
        )
        self.polygon_balancer_check.pack(side=tk.LEFT)
        
        ttk.Button(control_row, text="打开配置面板", command=self._open_balancer_config).pack(side=tk.RIGHT, padx=5)

        # 统计信息显示
        stats_frame = ttk.LabelFrame(frm, text="统计信息")
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.polygon_stats_text = tk.Text(stats_frame, height=12, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.polygon_stats_text.yview)
        self.polygon_stats_text.configure(yscrollcommand=scrollbar.set)
        
        self.polygon_stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 定时更新状态
        self._update_polygon_status()
        self.after(5000, self._schedule_polygon_update)  # 每5秒更新一次

    def _enable_polygon_factors(self):
        """启用Polygon因子"""
        try:
            if hasattr(self, 'trader') and self.trader:
                self.trader.enable_polygon_factors()
                self.log("Polygon因子已启用")
            else:
                self.log("请先连接交易系统")
        except Exception as e:
            self.log(f"启用Polygon因子失败: {e}")

    def _clear_polygon_cache(self):
        """清理Polygon缓存"""
        try:
            if hasattr(self, 'trader') and self.trader:
                self.trader.clear_polygon_cache()
                self.log("Polygon缓存已清理")
            else:
                self.log("请先连接交易系统")
        except Exception as e:
            self.log(f"清理Polygon缓存失败: {e}")

    def _toggle_polygon_balancer(self):
        """切换风控收益平衡器状态"""
        try:
            if hasattr(self, 'trader') and self.trader:
                if self.polygon_balancer_var.get():
                    self.trader.enable_polygon_risk_balancer()
                    self.log("风控收益平衡器已启用")
                else:
                    self.trader.disable_polygon_risk_balancer()
                    self.log("风控收益平衡器已禁用")
            else:
                self.log("请先连接交易系统")
                self.polygon_balancer_var.set(False)
        except Exception as e:
            self.log(f"切换风控收益平衡器状态失败: {e}")
            self.polygon_balancer_var.set(False)

    def _open_balancer_config(self):
        """打开风控收益平衡器配置面板"""
        try:
            # 导入GUI面板
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            from risk_balancer_gui_panel import create_standalone_gui
            
            # 在新线程中打开GUI，避免阻塞主界面
            import threading
            gui_thread = threading.Thread(target=create_standalone_gui, daemon=True)
            gui_thread.start()
            
            self.log("风控收益平衡器配置面板已打开")
            
        except Exception as e:
            self.log(f"打开配置面板失败: {e}")

    def _update_polygon_status(self):
        """更新Polygon状态显示"""
        try:
            if hasattr(self, 'trader') and self.trader:
                # 检查Polygon连接状态
                polygon_enabled = hasattr(self.trader, 'polygon_enabled') and self.trader.polygon_enabled
                balancer_enabled = hasattr(self.trader, 'polygon_risk_balancer_enabled') and self.trader.polygon_risk_balancer_enabled
                
                if polygon_enabled:
                    status_text = "状态: Polygon已连接"
                    status_color = "green"
                else:
                    status_text = "状态: Polygon未连接"
                    status_color = "red"
                
                self.polygon_status_label.config(text=status_text, fg=status_color)
                self.polygon_balancer_var.set(balancer_enabled)
                
                # 更新统计信息
                stats = self.trader.get_polygon_stats()
                if stats:
                    stats_text = "Polygon统计信息:\n"
                    stats_text += f"  启用状态: {'是' if stats.get('enabled', False) else '否'}\n"
                    stats_text += f"  风控平衡器: {'是' if stats.get('risk_balancer_enabled', False) else '否'}\n"
                    stats_text += f"  缓存大小: {stats.get('cache_size', 0)}\n"
                    stats_text += f"  总计算次数: {stats.get('total_calculations', 0)}\n"
                    stats_text += f"  成功次数: {stats.get('successful_calculations', 0)}\n"
                    stats_text += f"  失败次数: {stats.get('failed_calculations', 0)}\n"
                    stats_text += f"  缓存命中: {stats.get('cache_hits', 0)}\n"
                    
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
                    self.polygon_stats_text.insert(1.0, "暂无统计信息")
                    self.polygon_stats_text.config(state=tk.DISABLED)
            else:
                self.polygon_status_label.config(text="状态: 未连接交易系统", fg="gray")
                
        except Exception as e:
            self.polygon_status_label.config(text=f"状态: 检查失败 ({e})", fg="red")

    def _schedule_polygon_update(self):
        """定时更新Polygon状态"""
        self._update_polygon_status()
        self.after(5000, self._schedule_polygon_update)  # 每5秒更新一次

    def _build_engine_tab(self, parent) -> None:
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        box = ttk.LabelFrame(frm, text="策略引擎控制")
        box.pack(fill=tk.X, pady=8)

        ttk.Button(box, text="启动引擎(连接/订阅)", command=self._start_engine).grid(row=0, column=0, padx=6, pady=6)
        ttk.Button(box, text="运行一次信号与交易", command=self._engine_once).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(box, text="停止引擎", command=self._stop_engine_mode).grid(row=0, column=2, padx=6, pady=6)

        tip = ttk.Label(frm, text="说明: 引擎使用统一配置管理器扫描 universe，计算多因子信号并下单。")
        tip.pack(anchor=tk.W, pady=6)

    def _build_direct_tab(self, parent) -> None:
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 行1：基本参数
        row1 = ttk.LabelFrame(frm, text="下单参数")
        row1.pack(fill=tk.X, pady=6)
        ttk.Label(row1, text="标的").grid(row=0, column=0, padx=5, pady=5)
        self.d_sym = ttk.Entry(row1, width=12); self.d_sym.grid(row=0, column=1, padx=5)
        ttk.Label(row1, text="数量").grid(row=0, column=2, padx=5)
        self.d_qty = ttk.Entry(row1, width=10); self.d_qty.insert(0, "100"); self.d_qty.grid(row=0, column=3, padx=5)
        ttk.Label(row1, text="限价").grid(row=0, column=4, padx=5)
        self.d_px = ttk.Entry(row1, width=10); self.d_px.grid(row=0, column=5, padx=5)

        # 行2：基本按钮
        row2 = ttk.LabelFrame(frm, text="基础下单")
        row2.pack(fill=tk.X, pady=6)
        ttk.Button(row2, text="市价买入", command=lambda: self._direct_market("BUY")).grid(row=0, column=0, padx=6, pady=6)
        ttk.Button(row2, text="市价卖出", command=lambda: self._direct_market("SELL")).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(row2, text="限价买入", command=lambda: self._direct_limit("BUY")).grid(row=0, column=2, padx=6, pady=6)
        ttk.Button(row2, text="限价卖出", command=lambda: self._direct_limit("SELL")).grid(row=0, column=3, padx=6, pady=6)

        # 行3：括号单
        row3 = ttk.LabelFrame(frm, text="括号单")
        row3.pack(fill=tk.X, pady=6)
        ttk.Label(row3, text="止损% ").grid(row=0, column=0, padx=5)
        self.d_stop = ttk.Entry(row3, width=8); self.d_stop.insert(0, "2.0"); self.d_stop.grid(row=0, column=1)
        ttk.Label(row3, text="止盈% ").grid(row=0, column=2, padx=5)
        self.d_tp = ttk.Entry(row3, width=8); self.d_tp.insert(0, "5.0"); self.d_tp.grid(row=0, column=3)
        ttk.Button(row3, text="市价括号单(买)", command=lambda: self._direct_bracket("BUY")).grid(row=0, column=4, padx=6, pady=6)
        ttk.Button(row3, text="市价括号单(卖)", command=lambda: self._direct_bracket("SELL")).grid(row=0, column=5, padx=6, pady=6)

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
            # 立即在主线程提示，避免“无反应”感受
            self.log(f"准备启动引擎(连接/订阅)... Host={self.state.host} Port={self.state.port} ClientId={self.state.client_id}")
            loop = self._ensure_loop()
            async def _run():
                try:
                    # 线程安全日志
                    try:
                        self.after(0, lambda: self.log(
                            f"启动引擎参数: Host={self.state.host}, Port={self.state.port}, ClientID={self.state.client_id}"))
                    except Exception:
                        pass
                    # 启动前先断开现有连接，避免clientId占用
                    if self.trader and getattr(self.trader, 'ib', None) and self.trader.ib.isConnected():
                        try:
                            await self.trader.close()
                            try:
                                self.after(0, lambda: self.log("已断开之前的API连接"))
                            except Exception:
                                pass
                        except Exception:
                            pass
                    # 创建并连接交易器，使用统一配置
                    self.trader = IbkrAutoTrader(config_manager=self.config_manager)
                    # 注册到资源监控
                    self.resource_monitor.register_connection(self.trader)
                    
                    # 让 Engine 统一负责 connect 与订阅，使用统一配置
                    self.engine = Engine(self.config_manager, self.trader)
                    await self.engine.start()
                    try:
                        self.after(0, lambda: self.log("策略引擎已启动并完成订阅"))
                        self.after(0, lambda: self._update_signal_status("引擎已启动", "green"))
                    except Exception:
                        pass
                except Exception as e:
                    error_msg = str(e)
                    try:
                        self.after(0, lambda e_msg=error_msg: self.log(f"策略引擎启动失败: {e_msg}"))
                    except Exception:
                        print(f"策略引擎启动失败: {e}")  # 降级日志
            # 使用线程安全的事件循环管理器（非阻塞）
            try:
                task_id = self.loop_manager.submit_coroutine_nowait(_run())
                self.after(0, lambda: self.log(f"策略引擎任务已提交 (ID: {task_id[:8]}...)"))
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda e_msg=error_msg: self.log(f"策略引擎启动失败: {e_msg}"))
        except Exception as e:
            self.log(f"启动引擎错误: {e}")

    def _engine_once(self) -> None:
        try:
            if not self.engine:
                self.log("请先启动引擎")
                return
            # 使用非阻塞提交避免GUI卡死
            if hasattr(self, 'loop_manager') and self.loop_manager.is_running():
                task_id = self.loop_manager.submit_coroutine_nowait(self.engine.on_signal_and_trade())
                self.log(f"信号交易已提交，任务ID: {task_id}")
            else:
                self.log("事件循环未运行，无法执行信号交易")
            self.log("已触发一次信号与交易")
            self._update_signal_status("执行交易信号", "blue")
        except Exception as e:
            self.log(f"运行引擎一次失败: {e}")

    def _stop_engine_mode(self) -> None:
        try:
            self.log("策略引擎停止：可通过停止交易按钮一并断开连接与任务")
            self._update_signal_status("已停止", "red")
        except Exception as e:
            self.log(f"停止引擎失败: {e}")

    def _direct_market(self, side: str) -> None:
        try:
            sym = (self.d_sym.get() or "").strip().upper()
            qty = int(self.d_qty.get().strip())
            if not sym or qty <= 0:
                messagebox.showwarning("警告", "请输入有效的标的与数量")
                return
            loop = self._ensure_loop()
            async def _run():
                try:
                    if not self.trader:
                        self.trader = IbkrAutoTrader(config_manager=self.config_manager)
                        await self.trader.connect()
                    await self.trader.place_market_order(sym, side, qty)
                    self.log(f"已提交市价单: {side} {qty} {sym}")
                except Exception as e:
                    self.log(f"市价单失败: {e}")
            # 使用非阻塞提交避免GUI卡死
            if hasattr(self, 'loop_manager') and self.loop_manager.is_running():
                task_id = self.loop_manager.submit_coroutine_nowait(_run())
                self.log(f"下单任务已提交，任务ID: {task_id}")
            else:
                self.log("事件循环未运行，无法执行下单操作")
        except Exception as e:
            self.log(f"市价下单错误: {e}")

    def _direct_limit(self, side: str) -> None:
        try:
            sym = (self.d_sym.get() or "").strip().upper()
            qty = int(self.d_qty.get().strip())
            px_str = (self.d_px.get() or "").strip()
            if not sym or qty <= 0 or not px_str:
                messagebox.showwarning("警告", "请输入标的/数量/限价")
                return
            px = float(px_str)
            loop = self._ensure_loop()
            async def _run():
                try:
                    if not self.trader:
                        self.trader = IbkrAutoTrader(config_manager=self.config_manager)
                        await self.trader.connect()
                    await self.trader.place_limit_order(sym, side, qty, px)
                    self.log(f"已提交限价单: {side} {qty} {sym} @ {px}")
                except Exception as e:
                    self.log(f"限价单失败: {e}")
            # 使用非阻塞提交避免GUI卡死
            if hasattr(self, 'loop_manager') and self.loop_manager.is_running():
                task_id = self.loop_manager.submit_coroutine_nowait(_run())
                self.log(f"下单任务已提交，任务ID: {task_id}")
            else:
                self.log("事件循环未运行，无法执行下单操作")
        except Exception as e:
            self.log(f"限价下单错误: {e}")

    def _direct_bracket(self, side: str) -> None:
        try:
            sym = (self.d_sym.get() or "").strip().upper()
            qty = int(self.d_qty.get().strip())
            stop_pct = float((self.d_stop.get() or "2.0").strip())/100.0
            tp_pct = float((self.d_tp.get() or "5.0").strip())/100.0
            if not sym or qty <= 0:
                messagebox.showwarning("警告", "请输入标的与数量")
                return
            loop = self._ensure_loop()
            async def _run():
                try:
                    if not self.trader:
                        self.trader = IbkrAutoTrader(config_manager=self.config_manager)
                        await self.trader.connect()
                    await self.trader.place_market_order_with_bracket(sym, side, qty, stop_pct=stop_pct, target_pct=tp_pct)
                    self.log(f"已提交括号单: {side} {qty} {sym} (止损{stop_pct*100:.1f}%, 止盈{tp_pct*100:.1f}%)")
                except Exception as e:
                    self.log(f"括号单失败: {e}")
            # 使用非阻塞提交避免GUI卡死
            if hasattr(self, 'loop_manager') and self.loop_manager.is_running():
                task_id = self.loop_manager.submit_coroutine_nowait(_run())
                self.log(f"下单任务已提交，任务ID: {task_id}")
            else:
                self.log("事件循环未运行，无法执行下单操作")
        except Exception as e:
            self.log(f"括号单错误: {e}")

    def _direct_algo(self, side: str) -> None:
        try:
            sym = (self.d_sym.get() or "").strip().upper()
            qty = int(self.d_qty.get().strip())
            algo = (self.d_algo.get() or "TWAP").strip().upper()
            dur_min = int((self.d_dur.get() or "30").strip())
            if not sym or qty <= 0:
                messagebox.showwarning("警告", "请输入标的与数量")
                return
            loop = self._ensure_loop()
            async def _run():
                try:
                    if not self.trader:
                        self.trader = IbkrAutoTrader(config_manager=self.config_manager)
                        await self.trader.connect()
                    await self.trader.execute_large_order(sym, side, qty, algorithm=algo, duration_minutes=dur_min)
                    self.log(f"已提交大单执行: {algo} {side} {qty} {sym} / {dur_min}min")
                except Exception as e:
                    self.log(f"大单执行失败: {e}")
            # 使用非阻塞提交避免GUI卡死
            if hasattr(self, 'loop_manager') and self.loop_manager.is_running():
                task_id = self.loop_manager.submit_coroutine_nowait(_run())
                self.log(f"下单任务已提交，任务ID: {task_id}")
            else:
                self.log("事件循环未运行，无法执行下单操作")
        except Exception as e:
            self.log(f"大单执行错误: {e}")

    def _delete_database(self) -> None:
        """一键删除数据库文件（含确认与重建）"""
        try:
            import os
            db_path = getattr(self.db, 'db_path', None)
            if not db_path:
                messagebox.showerror("错误", "未找到数据库路径")
                return
            
            if not os.path.exists(db_path):
                messagebox.showinfo("提示", "数据库文件不存在，无需删除")
                return
            
            confirm = messagebox.askyesno(
                "确认删除",
                f"将删除数据库文件:\n{db_path}\n\n此操作不可恢复，是否继续？"
            )
            if not confirm:
                return
            
            # 关闭连接再删除
            try:
                self.db.close()
            except Exception:
                pass
            
            os.remove(db_path)
            self.log(f"已删除数据库: {db_path}")
            
            # 重新初始化数据库并刷新UI
            self.db = StockDatabase()
            self._refresh_stock_lists()
            self._refresh_configs()
            messagebox.showinfo("完成", "数据库已删除并重建为空库")
        
        except Exception as e:
            self.log(f"删除数据库失败: {e}")
            messagebox.showerror("错误", f"删除数据库失败: {e}")

    def _print_database(self) -> None:
        """打印当前数据库内容到日志（全局tickers、股票列表、选中列表、交易配置）。"""
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
                self.log("全局 tickers: 无")

            # 股票列表概览
            try:
                lists = self.db.get_stock_lists()
            except Exception:
                lists = []
            if lists:
                summary = ", ".join([f"{it['name']}({it.get('stock_count', 0)})" for it in lists])
                self.log(f"股票列表 {len(lists)} 个: {summary}")
            else:
                self.log("股票列表: 无")

            # 当前选中列表的明细
            try:
                if self.state.selected_stock_list_id:
                    rows = self.db.get_stocks_in_list(self.state.selected_stock_list_id)
                    syms = [r.get('symbol') for r in rows]
                    preview = ", ".join(syms[:200]) + ("..." if len(syms) > 200 else "")
                    self.log(f"当前列表 {self.stock_list_var.get()} 共 {len(syms)}: {preview}")
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
                self.log("交易配置: 无")

        except Exception as e:
            self.log(f"打印数据库失败: {e}")

    def _build_database_tab(self, parent):
        """构建数据库股票管理选项卡"""
        # 左侧：全局交易股票（仅显示会被交易的全局tickers）
        left_frame = tk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        stock_frame = tk.LabelFrame(left_frame, text="交易股票（全局tickers）")
        stock_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建Treeview，仅显示symbol与added_at
        columns = ('symbol', 'added_at')
        self.stock_tree = ttk.Treeview(stock_frame, columns=columns, show='headings', height=10)
        self.stock_tree.heading('symbol', text='股票代码')
        self.stock_tree.heading('added_at', text='添加时间')
        self.stock_tree.column('symbol', width=100)
        self.stock_tree.column('added_at', width=150)
        
        # 滚动条
        stock_scroll = ttk.Scrollbar(stock_frame, orient=tk.VERTICAL, command=self.stock_tree.yview)
        self.stock_tree.configure(yscrollcommand=stock_scroll.set)
        
        self.stock_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        stock_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # 右侧：操作面板（以全局tickers为主）
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
        
        # 批量导入到全局tickers
        import_frame = tk.LabelFrame(right_frame, text="批量导入(全局)")
        import_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(import_frame, text="CSV格式:").grid(row=0, column=0, padx=5, pady=5)
        self.ent_batch_csv = tk.Text(import_frame, width=20, height=4)
        self.ent_batch_csv.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        self.ent_batch_csv.insert(tk.END, "AAPL,MSFT,GOOGL")
        
        tk.Button(import_frame, text="批量导入", command=self._batch_import_global, bg="lightyellow").grid(row=2, column=0, columnspan=2, pady=5)
        
        # 删除全局tickers中的股票
        delete_frame = tk.LabelFrame(right_frame, text="删除交易股票(全局)")
        delete_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(delete_frame, text="删除选中", command=self._delete_selected_ticker_global, bg="lightcoral").grid(row=0, column=0, padx=5, pady=5)
        
        # 配置管理
        config_frame = tk.LabelFrame(right_frame, text="配置管理")
        config_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(config_frame, text="配置名称:").grid(row=0, column=0, padx=5, pady=5)
        self.config_name_var = tk.StringVar()
        self.config_combo = ttk.Combobox(config_frame, textvariable=self.config_name_var, width=15)
        self.config_combo.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        tk.Button(config_frame, text="保存配置", command=self._save_config, bg="lightblue").grid(row=2, column=0, padx=2, pady=5)
        tk.Button(config_frame, text="加载配置", command=self._load_config, bg="lightgreen").grid(row=2, column=1, padx=2, pady=5)

        # 同步功能移除（仅保留全局tickers作为唯一交易源）
        
        # 初始化数据
        self._refresh_global_tickers_table()
        self._refresh_configs()

    def _build_file_tab(self, parent):
        """构建文件导入选项卡"""
        # 股票输入
        wl = tk.LabelFrame(parent, text="股票列表（三选一或组合）")
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
        
        # 导入选项
        import_options = tk.LabelFrame(parent, text="文件导入选项")
        import_options.pack(fill=tk.X, pady=5)
        
        self.var_auto_clear = tk.BooleanVar(value=True)
        tk.Checkbutton(import_options, text="上传新文件 -> 替换全局tickers 并可选清仓被移除标的", 
                      variable=self.var_auto_clear).pack(anchor=tk.W, padx=5, pady=5)
        
        tk.Button(import_options, text="导入到数据库（替换全局tickers）", 
                 command=self._import_file_to_database, bg="orange").pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(import_options, text="导入到数据库（追加到全局tickers）", 
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
            # 注意：此线程内禁止直接调用 Tk 方法，需使用 self.after 进入主线程
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
                # 直接置位就绪事件（此刻loop已创建），避免等待超时
                if self._loop_ready_event is None:
                    self._loop_ready_event = threading.Event()
                try:
                    self._loop_ready_event.set()
                except Exception:
                    pass
                safe_log("事件循环已创建并即将启动")
                loop.run_forever()
            except Exception as e:
                safe_log(f"事件循环异常: {e}")
            finally:
                try:
                    # Clean up any remaining tasks
                    if loop and not loop.is_closed():
                        pending = asyncio.all_tasks(loop)
                        if pending:
                            safe_log(f"正在清理 {len(pending)} 个未完成任务...")
                            for task in pending:
                                task.cancel()
                            # Wait a bit for tasks to cancel
                            try:
                                loop.run_until_complete(
                                    asyncio.wait(pending, timeout=3, return_when=asyncio.ALL_COMPLETED)
                                )
                            except Exception:
                                pass
                        loop.close()
                except Exception as e:
                    safe_log(f"事件循环清理异常: {e}")
        
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()
        
        # Wait for loop to be ready (降级方案：短等待+存在即返回)
        import time
        if self._loop_ready_event is None:
            self._loop_ready_event = threading.Event()
        self._loop_ready_event.wait(timeout=1.0)
        if self.loop is not None:
            return self.loop  # type: ignore
        # If still not running, provide a helpful log and raise
        self.log("事件循环未能在预期时间内启动，请重试'测试连接'或'启动自动交易'。")
        raise RuntimeError("Failed to start event loop")

    def _capture_ui(self) -> None:
        self.state.host = self.ent_host.get().strip() or "127.0.0.1"
        try:
            # 自定义端口与clientId：完全尊重用户输入
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
            messagebox.showerror("参数错误", "端口/ClientId必须是整数，资金占比/轮询间隔必须是数字")
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"参数捕获失败: {e}"
            self.log(error_msg)
            messagebox.showerror("参数错误", error_msg)
            raise
        self.state.sheet = self.ent_sheet.get().strip() or None
        self.state.column = self.ent_col.get().strip() or None
        self.state.symbols_csv = self.ent_csv.get().strip() or None
        self.state.auto_sell_removed = self.var_auto_sell.get()
        
        # 同时更新统一配置管理器
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
            if hasattr(self, 'loop_manager') and self.loop_manager.is_running():
                # 使用无等待提交避免阻塞主线程
                task_id = self.loop_manager.submit_coroutine_nowait(coro)
                self.log(f"{operation_name}已提交，任务ID: {task_id}")
                return task_id
            else:
                # 回退到独立线程
                import asyncio
                thread_name = f"{operation_name}Thread"
                threading.Thread(
                    target=lambda: asyncio.run(coro), 
                    daemon=True,
                    name=thread_name
                ).start()
                self.log(f"{operation_name}已在后台线程启动")
                return None
        except Exception as e:
            self.log(f"{operation_name}启动失败: {e}")
            return None

    def _test_connection(self) -> None:
        try:
            self._capture_ui()
            self.log(f"正在测试连接... Host={self.state.host} Port={self.state.port} ClientId={self.state.client_id}")
            
            async def _run():
                try:
                    # 显示实际使用的连接参数
                    self.log(f"连接参数: Host={self.state.host}, Port={self.state.port}, ClientID={self.state.client_id}")
                    # 启动前先断开现有连接，避免clientId占用
                    if self.trader and getattr(self.trader, 'ib', None) and self.trader.ib.isConnected():
                        try:
                            await self.trader.close()
                            self.log("已断开之前的API连接")
                        except Exception:
                            pass
                    self.trader = IbkrAutoTrader(config_manager=self.config_manager)
                    await self.trader.connect()
                    self.log("[OK] 连接成功")
                except Exception as e:
                    self.log(f"[FAIL] 连接失败: {e}")
            
            # 使用非阻塞异步执行，避免GUI卡死
            def _async_test():
                try:
                    if hasattr(self, 'loop_manager') and self.loop_manager.is_running():
                        # 使用无等待提交避免阻塞主线程
                        task_id = self.loop_manager.submit_coroutine_nowait(_run())
                        self.log(f"连接测试已提交，任务ID: {task_id}")
                    else:
                        # 回退到独立线程
                        import asyncio
                        threading.Thread(
                            target=lambda: asyncio.run(_run()), 
                            daemon=True,
                            name="ConnectionTest"
                        ).start()
                        self.log("连接测试已在后台线程启动")
                except Exception as e:
                    self.log(f"连接测试启动失败: {e}")
            
            _async_test()
            
        except Exception as e:
            self.log(f"测试连接错误: {e}")
            messagebox.showerror("错误", f"测试连接失败: {e}")

    def _start_autotrade(self) -> None:
        try:
            self._capture_ui()
            self.log(f"正在启动自动交易（策略引擎模式）... Host={self.state.host} Port={self.state.port} ClientId={self.state.client_id}")

            async def _run():
                try:
                    # 显示实际使用的连接参数
                    self.log(f"启动引擎参数: Host={self.state.host}, Port={self.state.port}, ClientID={self.state.client_id}")
                    # 1) 准备 Trader 连接
                    # 启动前先断开现有连接，避免clientId占用
                    if self.trader and getattr(self.trader, 'ib', None) and self.trader.ib.isConnected():
                        try:
                            await self.trader.close()
                            self.log("已断开之前的API连接")
                        except Exception:
                            pass
                    if not self.trader:
                        self.trader = IbkrAutoTrader(config_manager=self.config_manager)
                    await self.trader.connect()

                    # 2) 准备 Engine 与 Universe（优先数据库/外部文件/手动CSV）
                    uni = []
                    try:
                        db_csv = self._get_current_stock_symbols()
                        if db_csv:
                            uni = [s for s in db_csv.split(',') if s.strip()]
                        elif any([self.state.json_file, self.state.excel_file, self.state.symbols_csv]):
                            uni = self._extract_symbols_from_files()
                    except Exception:
                        pass
                    # 使用统一配置管理器
                    cfg = self.config_manager
                    if uni:
                        cfg.set_runtime("scanner.universe", uni)
                        self.log(f"策略引擎使用自定义Universe: {len(uni)} 只标的")

                    if not self.engine:
                        self.engine = Engine(cfg, self.trader)
                    await self.engine.start()

                    # 3) 周期性执行信号→风控→下单（完整增强策略）
                    self.log(f"策略循环启动: 间隔={self.state.poll_sec}s")

                    async def _engine_loop():
                        try:
                            while True:
                                await self.engine.on_signal_and_trade()
                                await asyncio.sleep(max(1.0, float(self.state.poll_sec)))
                        except asyncio.CancelledError:
                            return
                        except Exception as e:
                            self.log(f"策略循环异常: {e}")

                    # 在事件循环中创建任务并保存引用
                    self._engine_loop_task = asyncio.create_task(_engine_loop())
                    self.log("策略引擎已启动并进入循环")
                    self._update_signal_status("循环运行中", "green")
                except Exception as e:
                    self.log(f"自动交易启动失败: {e}")

            # 使用非阻塞异步执行，避免GUI卡死
            def _async_start():
                try:
                    if hasattr(self, 'loop_manager') and self.loop_manager.is_running():
                        # 使用无等待提交避免阻塞主线程
                        task_id = self.loop_manager.submit_coroutine_nowait(_run())
                        self.log(f"自动交易启动已提交，任务ID: {task_id}")
                    else:
                        # 回退到独立线程
                        import asyncio
                        threading.Thread(
                            target=lambda: asyncio.run(_run()), 
                            daemon=True,
                            name="AutoTradeStart"
                        ).start()
                        self.log("自动交易启动已在后台线程启动")
                except Exception as e:
                    self.log(f"自动交易启动失败: {e}")
            
            _async_start()

        except Exception as e:
            self.log(f"启动自动交易错误: {e}")
            messagebox.showerror("错误", f"启动失败: {e}")

    def _stop(self) -> None:
        """Enhanced stop mechanism with proper cleanup"""
        try:
            if not self.trader and not self.loop:
                self.log("没有活动的交易连接")
                return
                
            self.log("正在停止交易...")
            
            # Signal the trader to stop
            if self.trader:
                try:
                    if hasattr(self.trader, '_stop_event'):
                        if not self.trader._stop_event:
                            self.trader._stop_event = asyncio.Event()
                        self.trader._stop_event.set()
                        self.log("已发送停止信号到交易器")
                except Exception as e:
                    self.log(f"发送停止信号失败: {e}")

                # 停止策略引擎循环
                try:
                    if self.loop and self.loop.is_running() and self._engine_loop_task and not self._engine_loop_task.done():
                        def _cancel_task(task: asyncio.Task):
                            if not task.done():
                                task.cancel()
                        self.loop.call_soon_threadsafe(_cancel_task, self._engine_loop_task)
                        self.log("已请求停止策略引擎循环")
                        self._update_signal_status("循环已停止", "red")
                except Exception as e:
                    self.log(f"停止策略循环失败: {e}")

                # Stop engine and close trader connection
                if self.loop and self.loop.is_running():
                    async def _cleanup_all():
                        try:
                            # Stop engine first
                            if self.engine:
                                await self.engine.stop()
                                self.log("引擎已停止")
                                self.engine = None
                            
                            # Then close trader connection
                            if self.trader:
                                await self.trader.close()
                                self.log("交易连接已关闭")
                                self.trader = None
                        except Exception as e:
                            self.log(f"停止引擎/交易器失败: {e}")
                            
                    self.loop_manager.submit_coroutine_nowait(_cleanup_all())
                    self.log("清理任务已提交到后台")
                else:
                    self.trader = None
            
            # Clean up event loop
            if self.loop and not self.loop.is_closed():
                try:
                    if self.loop.is_running():
                        # Schedule loop stop
                        self.loop.call_soon_threadsafe(self.loop.stop)
                        self.log("已安排停止事件循环")
                        
                        # Give some time for cleanup
                        def reset_loop():
                            if self.loop and self.loop.is_closed():
                                self.loop = None
                        
                        self.after(2000, reset_loop)  # Reset after 2 seconds
                        
                except Exception as e:
                    self.log(f"停止事件循环失败: {e}")
            
            self.log("停止操作已完成")
                
        except Exception as e:
            self.log(f"停止交易错误: {e}")
            messagebox.showerror("错误", f"停止失败: {e}")

    def _disconnect_api(self) -> None:
        """一键断开API连接（不影响引擎结构，清理clientId占用）"""
        try:
            if not self.trader:
                self.log("无活动API连接")
                return
            self.log("正在断开API连接...")
            if self.loop and self.loop.is_running():
                # 先在线程安全地立即断开底层IB连接，避免clientId占用
                try:
                    if getattr(self.trader, 'ib', None):
                        self.loop.call_soon_threadsafe(self.trader.ib.disconnect)
                except Exception:
                    pass
                # 然后进行完整清理，并等待结果以反馈日志
                async def _do_close():
                    try:
                        await self.trader.close()
                        self.log("API连接已断开")
                    except Exception as e:
                        self.log(f"断开API失败: {e}")
                try:
                    self.loop_manager.submit_coroutine_nowait(_do_close())
                    self.log("关闭任务已提交到后台")
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
                self.log("API连接已断开(无事件循环)")
            # 置空 trader，释放clientId
            self.trader = None
            # 更新状态显示
            try:
                self._update_status()
                self._update_signal_status("已断开", "red")
            except Exception:
                pass
            try:
                # 即刻反馈
                messagebox.showinfo("提示", "API连接已断开")
            except Exception:
                pass
        except Exception as e:
            self.log(f"断开API出错: {e}")

    def _clear_log(self) -> None:
        self.txt.delete(1.0, tk.END)
        self.log("日志已清空")

    def _show_account(self) -> None:
        try:
            if not self.trader:
                self.log("请先连接IBKR")
                return
                
            self.log("正在获取账户信息...")
            loop = self._ensure_loop()
            
            async def _run():
                try:
                    await self.trader.refresh_account_balances_and_positions()
                    self.log(f"现金余额: ${self.trader.cash_balance:,.2f}")
                    self.log(f"账户净值: ${self.trader.net_liq:,.2f}")
                    self.log(f"持仓数量: {len(self.trader.positions)} 只股票")
                    for symbol, qty in self.trader.positions.items():
                        if qty != 0:
                            self.log(f"  {symbol}: {qty} 股")
                except Exception as e:
                    self.log(f"获取账户信息失败: {e}")
                    
            # 使用非阻塞提交避免GUI卡死
            if hasattr(self, 'loop_manager') and self.loop_manager.is_running():
                task_id = self.loop_manager.submit_coroutine_nowait(_run())
                self.log(f"下单任务已提交，任务ID: {task_id}")
            else:
                self.log("事件循环未运行，无法执行下单操作")
            
        except Exception as e:
            self.log(f"查看账户错误: {e}")

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
            self.log(f"刷新股票列表失败: {e}")
    
    def _refresh_configs(self):
        """刷新配置下拉框"""
        try:
            configs = self.db.get_trading_configs()
            config_names = [cfg['name'] for cfg in configs]
            self.config_combo['values'] = config_names
            
            if config_names:
                self.config_combo.current(0)
                
        except Exception as e:
            self.log(f"刷新配置失败: {e}")
    
    # ===== 全局tickers视图与操作（唯一交易源） =====
    def _refresh_global_tickers_table(self) -> None:
        """刷新全局tickers在表格中的显示"""
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
            self.log(f"刷新交易股票失败: {e}")
    
    def _add_ticker_global(self) -> None:
        """添加到全局tickers"""
        try:
            symbol = (self.ent_symbol.get() or '').strip().upper()
            if not symbol:
                messagebox.showwarning("警告", "请输入股票代码")
                return
            if self.db.add_ticker(symbol):
                self.log(f"已添加到全局tickers: {symbol}")
                try:
                    self.ent_symbol.delete(0, tk.END)
                except Exception:
                    pass
                self._refresh_global_tickers_table()
            else:
                messagebox.showwarning("警告", f"{symbol} 已存在")
        except Exception as e:
            self.log(f"添加全局ticker失败: {e}")
            messagebox.showerror("错误", f"添加失败: {e}")
    
    def _batch_import_global(self) -> None:
        """批量导入到全局tickers"""
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
            self.log(f"批量导入(全局)完成: 成功 {success}，失败 {fail}")
            try:
                self.ent_batch_csv.delete(1.0, tk.END)
            except Exception:
                pass
            self._refresh_global_tickers_table()
        except Exception as e:
            self.log(f"批量导入(全局)失败: {e}")
            messagebox.showerror("错误", f"批量导入失败: {e}")
    
    def _delete_selected_ticker_global(self) -> None:
        """从全局tickers删除选中的股票，并触发自动清仓。"""
        try:
            selected_items = self.stock_tree.selection()
            if not selected_items:
                messagebox.showwarning("警告", "请先选择要删除的股票")
                return
            symbols = []
            for item in selected_items:
                values = self.stock_tree.item(item, 'values')
                if values:
                    symbols.append(values[0])
            if not symbols:
                return
            result = messagebox.askyesno("确认删除", f"确定要从全局tickers删除：\n{', '.join(symbols)}")
            if not result:
                return
            removed = []
            for symbol in symbols:
                if self.db.remove_ticker(symbol):
                    removed.append(symbol)
            self.log(f"已从全局tickers删除 {len(removed)} 只: {', '.join(removed) if removed else ''}")
            self._refresh_global_tickers_table()

            # 触发自动清仓（市价卖出被删除的标的的现有持仓）
            if removed:
                if self.trader and self.loop and self.loop.is_running():
                    try:
                        task_id = self.loop_manager.submit_coroutine_nowait(self._auto_sell_stocks(removed))
                        self.log(f"自动清仓任务已提交 (ID: {task_id[:8]}...)")
                    except Exception as e:
                        self.log(f"触发自动清仓失败: {e}")
                else:
                    self.log("当前未连接交易或事件循环未运行，无法自动清仓。稍后连接后可在文件导入页用替换功能清仓。")
        except Exception as e:
            self.log(f"删除全局ticker失败: {e}")
            messagebox.showerror("错误", f"删除失败: {e}")
    
    def _on_stock_list_changed(self, event):
        """股票列表选择变化"""
        try:
            selected = self.stock_list_var.get()
            if selected and selected in self.stock_list_mapping:
                list_id = self.stock_list_mapping[selected]
                self.state.selected_stock_list_id = list_id
                self._refresh_stock_table(list_id)
                
        except Exception as e:
            self.log(f"切换股票列表失败: {e}")
    
    def _refresh_stock_table(self, list_id):
        """刷新股票表格"""
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
            self.log(f"刷新股票表格失败: {e}")
    
    def _create_stock_list(self):
        """创建新股票列表"""
        try:
            name = tk.simpledialog.askstring("新建股票列表", "请输入列表名称:")
            if not name:
                return
                
            description = tk.simpledialog.askstring("新建股票列表", "请输入描述（可选）:") or ""
            
            list_id = self.db.create_stock_list(name, description)
            self.log(f"成功创建股票列表: {name}")
            self._refresh_stock_lists()
            
        except ValueError as e:
            messagebox.showerror("错误", str(e))
        except Exception as e:
            self.log(f"创建股票列表失败: {e}")
            messagebox.showerror("错误", f"创建失败: {e}")
    
    def _delete_stock_list(self):
        """删除股票列表"""
        try:
            if not self.state.selected_stock_list_id:
                messagebox.showwarning("警告", "请先选择股票列表")
                return
                
            selected = self.stock_list_var.get()
            result = messagebox.askyesno("确认删除", f"确定要删除股票列表 '{selected}' 吗？\n此操作将删除列表中的所有股票！")
            
            if result:
                if self.db.delete_stock_list(self.state.selected_stock_list_id):
                    self.log(f"成功删除股票列表: {selected}")
                    self._refresh_stock_lists()
                else:
                    messagebox.showerror("错误", "删除失败")
                    
        except Exception as e:
            self.log(f"删除股票列表失败: {e}")
            messagebox.showerror("错误", f"删除失败: {e}")
    
    def _add_stock(self):
        """已废弃（列表模式移除）"""
        messagebox.showinfo("提示", "此功能已由‘添加交易股票(全局)’替代")
    
    def _batch_import(self):
        """已废弃（列表模式移除）"""
        messagebox.showinfo("提示", "此功能已由‘批量导入(全局)’替代")
    
    def _delete_selected_stock(self):
        """已废弃（列表模式移除）"""
        messagebox.showinfo("提示", "此功能已由‘删除交易股票(全局)’替代")

    def _sync_global_to_current_list_replace(self):
        """将全局tickers替换写入当前选中列表（stocks表）。"""
        try:
            if not self.state.selected_stock_list_id:
                messagebox.showwarning("警告", "请先选择股票列表")
                return
            tickers = self.db.get_all_tickers()
            if not tickers:
                messagebox.showinfo("提示", "全局tickers为空。请先在‘文件导入’页导入或追加股票。")
                return
            ok = messagebox.askyesno(
                "确认同步",
                f"将用全局tickers({len(tickers)}只)替换当前列表的股票，是否继续？")
            if not ok:
                return
            removed_symbols = self.db.clear_stock_list(self.state.selected_stock_list_id)
            added = 0
            for sym in tickers:
                if self.db.add_stock(self.state.selected_stock_list_id, sym):
                    added += 1
            self.log(f"同步完成：清空原有 {len(removed_symbols)} 只，写入 {added} 只")
            self._refresh_stock_table(self.state.selected_stock_list_id)
            self._refresh_stock_lists()
        except Exception as e:
            self.log(f"全局→列表同步失败: {e}")
            messagebox.showerror("错误", f"同步失败: {e}")

    def _sync_current_list_to_global_replace(self):
        """将当前选中列表替换写入全局tickers（可触发自动清仓逻辑）。"""
        try:
            if not self.state.selected_stock_list_id:
                messagebox.showwarning("警告", "请先选择股票列表")
                return
            rows = self.db.get_stocks_in_list(self.state.selected_stock_list_id)
            symbols = [r.get('symbol') for r in rows if r.get('symbol')]
            ok = messagebox.askyesno(
                "确认同步",
                f"将用当前列表({len(symbols)}只)替换全局tickers，是否继续？\n可在‘文件导入’页勾选‘自动清仓’控制是否清仓被移除标的。")
            if not ok:
                return
            removed_before, success, fail = self.db.replace_all_tickers(symbols)
            self.log(f"列表→全局同步完成：移除 {len(removed_before)}，写入成功 {success}，失败 {fail}")
            # 根据勾选项触发自动清仓
            auto_clear = bool(self.var_auto_clear.get())
            if auto_clear and removed_before:
                if self.trader and self.loop and self.loop.is_running():
                    task_id = self.loop_manager.submit_coroutine_nowait(self._auto_sell_stocks(removed_before))
                    self.log(f"自动清仓任务已提交 (ID: {task_id[:8]}...)")
                else:
                    self.log("检测到被移除标的，但当前未连接交易或事件循环未运行，跳过自动清仓。")
        except Exception as e:
            self.log(f"列表→全局同步失败: {e}")
            messagebox.showerror("错误", f"同步失败: {e}")
    
    def _save_config(self):
        """保存交易配置"""
        try:
            name = self.config_name_var.get().strip()
            if not name:
                name = tk.simpledialog.askstring("保存配置", "请输入配置名称:")
                if not name:
                    return
            
            # 获取当前UI参数
            try:
                alloc = float(self.ent_alloc.get().strip() or 0.03)
                poll_sec = float(self.ent_poll.get().strip() or 10.0)
                fixed_qty = int(self.ent_fixed_qty.get().strip() or 0)
                auto_sell = self.var_auto_sell.get()
            except ValueError:
                messagebox.showerror("错误", "参数格式错误")
                return
            
            if self.db.save_trading_config(name, alloc, poll_sec, auto_sell, fixed_qty):
                self.log(f"成功保存配置到数据库: {name}")
                self._refresh_configs()
                self.config_name_var.set(name)
                
                # 同时更新统一配置管理器
                self.config_manager.update_runtime_config({
                    'trading.alloc_pct': alloc,
                    'trading.poll_interval': poll_sec,
                    'trading.auto_sell_removed': auto_sell,
                    'trading.fixed_quantity': fixed_qty
                })
                
                # 持久化到文件
                if self.config_manager.persist_runtime_changes():
                    self.log("✅ 交易配置已持久化到配置文件")
                else:
                    self.log("⚠️ 交易配置持久化失败，但已保存到数据库")
            else:
                messagebox.showerror("错误", "保存配置失败")
                
        except Exception as e:
            self.log(f"保存配置失败: {e}")
            messagebox.showerror("错误", f"保存失败: {e}")
    
    def _load_config(self):
        """加载交易配置"""
        try:
            name = self.config_name_var.get().strip()
            if not name:
                messagebox.showwarning("警告", "请选择配置")
                return
            
            config = self.db.load_trading_config(name)
            if config:
                # 更新UI
                self.ent_alloc.delete(0, tk.END)
                self.ent_alloc.insert(0, str(config['alloc']))
                
                self.ent_poll.delete(0, tk.END)
                self.ent_poll.insert(0, str(config['poll_sec']))
                
                self.ent_fixed_qty.delete(0, tk.END)
                self.ent_fixed_qty.insert(0, str(config['fixed_qty']))
                
                self.var_auto_sell.set(config['auto_sell_removed'])
                
                self.log(f"成功加载配置: {name}")
            else:
                messagebox.showerror("错误", "加载配置失败")
                
        except Exception as e:
            self.log(f"加载配置失败: {e}")
            messagebox.showerror("错误", f"加载失败: {e}")

    def _get_current_stock_symbols(self) -> str:
        """获取当前数据库中的股票代码（作为存在性检查用）。"""
        try:
            tickers = self.db.get_all_tickers()
            return ",".join(tickers)
        except Exception as e:
            self.log(f"获取股票列表失败: {e}")
            return ""

    async def _auto_sell_stocks(self, symbols_to_sell: List[str]):
        """自动清仓指定的股票"""
        if not symbols_to_sell:
            return
            
        try:
            if not self.trader:
                self.log("未连接交易接口，无法自动清仓")
                return
                
            self.log(f"开始自动清仓 {len(symbols_to_sell)} 只股票: {', '.join(symbols_to_sell)}")
            
            for symbol in symbols_to_sell:
                try:
                    # 获取当前持仓
                    if hasattr(self.trader, 'positions') and symbol in self.trader.positions:
                        position = self.trader.positions.get(symbol, 0)
                        if position > 0:
                            self.log(f"清仓 {symbol}: {position} 股")
                            await self.trader.place_market_order(symbol, "SELL", position)
                        else:
                            self.log(f"{symbol} 无持仓或已清仓")
                    else:
                        self.log(f"无法获取 {symbol} 持仓信息")
                        
                except Exception as e:
                    self.log(f"清仓 {symbol} 失败: {e}")
                    
        except Exception as e:
            self.log(f"自动清仓失败: {e}")

    def _import_file_to_database(self):
        """将文件内容导入到数据库（替换模式） -> 作用于全局 tickers 表"""
        try:
            # 同步最新的表单输入（sheet/column/手动CSV）
            self._capture_ui()
            # 获取要导入的股票（支持 json/excel/csv 手动）
            symbols_to_import = self._extract_symbols_from_files()
            self.log(f"待导入股票数: {len(symbols_to_import)}")
            if not symbols_to_import:
                messagebox.showwarning("警告", "没有找到要导入的股票")
                return
            
            # 确认对话框
            auto_clear = self.var_auto_clear.get()
            
            if auto_clear:
                msg = f"确定要替换全局tickers吗？\n\n操作内容：\n1. 自动清仓不再存在的股票\n2. 清空并导入新股票：{symbols_to_import[:50]}{'...' if len(symbols_to_import) > 50 else ''}\n\n此操作不可撤销！"
            else:
                msg = f"确定要替换全局tickers吗？\n\n操作内容：\n1. 清空并导入新股票（不清仓）\n2. 新股票：{symbols_to_import[:50]}{'...' if len(symbols_to_import) > 50 else ''}\n\n此操作不可撤销！"
                
            result = messagebox.askyesno("确认替换", msg)
            if not result:
                return
            
            # 执行导入：替换全局 tickers
            removed_before, success, fail = self.db.replace_all_tickers(symbols_to_import)
            
            self.log(f"股票列表替换完成:")
            self.log(f"  删除: {len(removed_before)} 只股票")
            self.log(f"  导入: 成功 {success} 只，失败 {fail} 只")

            # 即时打印当前全局 tickers 概览
            try:
                all_ticks = self.db.get_all_tickers()
                preview = ", ".join(all_ticks[:200]) + ("..." if len(all_ticks) > 200 else "")
                self.log(f"当前全局 tickers 共 {len(all_ticks)}: {preview}")
                try:
                    messagebox.showinfo("导入完成", f"当前全局 tickers 共 {len(all_ticks)} 条。")
                except Exception:
                    pass
            except Exception as e:
                self.log(f"读取全局tickers失败: {e}")
            
            # 如果启用自动清仓且交易器已连接且事件循环已在运行，则异步清仓
            if auto_clear and removed_before:
                if self.trader and self.loop and self.loop.is_running():
                    self.loop_manager.submit_coroutine(
                        self._auto_sell_stocks(removed_before), timeout=30)
                else:
                    self.log("检测到移除的股票，但当前未连接交易或事件循环未运行，跳过自动清仓。")
            
            # 刷新界面
            try:
                if getattr(self, 'state', None) and self.state.selected_stock_list_id:
                    self._refresh_stock_table(self.state.selected_stock_list_id)
            except Exception:
                pass
            
        except Exception as e:
            self.log(f"导入失败: {e}")
            messagebox.showerror("错误", f"导入失败: {e}")

    def _append_file_to_database(self):
        """将文件内容导入到数据库（追加模式） -> 作用于全局 tickers 表"""
        try:
            # 同步最新的表单输入
            self._capture_ui()
            # 获取要导入的股票（支持 json/excel/csv 手动）
            symbols_to_import = self._extract_symbols_from_files()
            self.log(f"待追加股票数: {len(symbols_to_import)}")
            if not symbols_to_import:
                messagebox.showwarning("警告", "没有找到要导入的股票")
                return
            
            # 确认对话框
            msg = f"确定要向全局tickers追加股票吗？\n\n将追加：{symbols_to_import[:50]}{'...' if len(symbols_to_import) > 50 else ''}"
            result = messagebox.askyesno("确认追加", msg)
            if not result:
                return
            
            # 执行追加导入到全局 tickers
            success, fail = 0, 0
            for s in symbols_to_import:
                if self.db.add_ticker(s):
                    success += 1
                else:
                    fail += 1
            
            self.log(f"股票追加完成: 成功 {success} 只，失败 {fail} 只")

            # 即时打印当前全局 tickers 概览
            try:
                all_ticks = self.db.get_all_tickers()
                preview = ", ".join(all_ticks[:200]) + ("..." if len(all_ticks) > 200 else "")
                self.log(f"当前全局 tickers 共 {len(all_ticks)}: {preview}")
                try:
                    messagebox.showinfo("追加完成", f"当前全局 tickers 共 {len(all_ticks)} 条。")
                except Exception:
                    pass
            except Exception as e:
                self.log(f"读取全局tickers失败: {e}")
            
            # 刷新界面
            try:
                if getattr(self, 'state', None) and self.state.selected_stock_list_id:
                    self._refresh_stock_table(self.state.selected_stock_list_id)
            except Exception:
                pass
            
        except Exception as e:
            self.log(f"追加导入失败: {e}")
            messagebox.showerror("错误", f"追加导入失败: {e}")

    def _extract_symbols_from_files(self) -> List[str]:
        """从JSON/Excel/CSV文件中提取股票代码（返回去重后的列表）"""
        try:
            symbols = []
            
            # 从JSON文件读取
            if self.state.json_file:
                import json
                with open(self.state.json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        symbols.extend([str(s).upper() for s in data])
                    else:
                        self.log("JSON文件格式错误：应该是股票代码数组")
            
            # 从Excel文件读取
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
                    self.log("缺少pandas库，无法读取Excel文件")
                except Exception as e:
                    self.log(f"读取Excel文件失败: {e}")
            
            # 从手动CSV读取
            if self.state.symbols_csv:
                csv_symbols = [s.strip().upper() for s in self.state.symbols_csv.split(",") if s.strip()]
                symbols.extend(csv_symbols)
            
            # 去重并返回
            unique_symbols = list(dict.fromkeys(symbols))  # 保持顺序的去重
            return unique_symbols
            
        except Exception as e:
            self.log(f"提取股票代码失败: {e}")
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
            self.log("正在关闭应用...")
            
            # First, cancel engine loop task if running
            if hasattr(self, '_engine_loop_task') and self._engine_loop_task and not self._engine_loop_task.done():
                try:
                    self._engine_loop_task.cancel()
                    self.log("已取消策略引擎循环任务")
                except Exception as e:
                    self.log(f"取消策略引擎循环失败: {e}")
            
            # Then, gracefully stop trader
            if self.trader:
                try:
                    if hasattr(self.trader, '_stop_event') and self.trader._stop_event:
                        self.trader._stop_event.set()
                        self.log("已设置交易器停止信号")
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
                                    self.log("引擎已停止")
                                
                                # Then close trader connection
                                if self.trader:
                                    await self.trader.close()
                                    self.log("交易器连接已关闭")
                            except Exception as e:
                                self.log(f"停止引擎/交易器失败: {e}")
                        
                        try:
                            self.loop_manager.submit_coroutine_nowait(_cleanup_all())
                            self.log("清理任务已提交到后台")
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
                            self.log(f"事件循环清理失败: {e}")
                    
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
                        self.log("资源监控已停止")
                    except Exception as e:
                        self.log(f"停止资源监控失败: {e}")
                    
                    # 停止事件循环管理器
                    try:
                        self.loop_manager.stop()
                        self.log("事件循环管理器已停止")
                    except Exception as e:
                        self.log(f"停止事件循环管理器失败: {e}")
                    
                    # 停止事件总线
                    try:
                        from autotrader.event_system import shutdown_event_bus
                        shutdown_event_bus()
                        self.log("事件总线已停止")
                    except Exception as e:
                        self.log(f"停止事件总线失败: {e}")
                    
                    # 保存配置变更到文件（持久化）
                    try:
                        if hasattr(self, 'config_manager'):
                            self.config_manager.persist_runtime_changes()
                            self.log("配置已自动保存")
                    except Exception as e:
                        self.log(f"自动保存配置失败: {e}")
                    
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
        """一键启动 BMA 增强模型：默认全量股票、回看最近5年、目标期=下一周"""
        try:
            # 计算5年窗口
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')

            # 默认运行 Ultra Enhanced，引入原版股票池与两阶段训练能力
            ultra_script = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, '量化模型_bma_ultra_enhanced.py'))
            script_path = ultra_script if os.path.exists(ultra_script) else os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, '量化模型_bma_enhanced.py'))
            if not os.path.exists(script_path):
                messagebox.showerror("错误", f"未找到量化模型脚本: {script_path}")
                return

            self.log(f"[BMA] 启动BMA增强模型: {start_date} -> {end_date} (默认全股票池)")

            # 使用性能优化器替代subprocess
            async def _runner_optimized():
                try:
                    # 标记模型开始训练
                    self._model_training = True
                    self._model_trained = False
                    self.after(0, lambda: self.log("[BMA] 开始优化执行..."))
                    self.after(0, lambda: self.log("[BMA] 注意：GUI应保持响应状态"))
                    
                    # 使用性能优化器
                    from .performance_optimizer import get_performance_optimizer
                    optimizer = get_performance_optimizer()
                    
                    # 定义进度回调
                    def progress_callback(result):
                        for line in result.output:
                            if line.strip():
                                self.after(0, lambda m=line: self.log(m))
                    
                    # 优化执行BMA模型
                    # Ultra Enhanced 支持参数：--tickers-file stocks.txt --tickers-limit 50
                    extra_args = []
                    if script_path.endswith('量化模型_bma_ultra_enhanced.py'):
                        # 小样本先测50只，随后脚本内部自动全量
                        extra_args = ['--tickers-file', 'stocks.txt', '--tickers-limit', '4000']

                    result = await optimizer.optimize_bma_execution(
                        script_path, start_date, end_date, progress_callback, extra_args=extra_args
                    )
                    
                    # 更新模型状态
                    self._model_training = False
                    self._model_trained = result.success
                    
                    if result.success:
                        self.after(0, lambda: self.log(f"[BMA] ✅ 运行完成 (耗时: {result.execution_time:.2f}s)"))
                        if result.cache_key:
                            self.after(0, lambda: self.log("[BMA] 📋 使用缓存优化"))
                        
                        # 显示性能统计
                        stats = optimizer.get_performance_stats()
                        speedup = stats['optimization_stats'].get('average_speedup', 1.0)
                        if speedup > 1.0:
                            self.after(0, lambda s=speedup: self.log(f"[BMA] 🚀 性能提升: {s:.1f}x"))
                    else:
                        error_msg = result.error if result.error else "未知错误"
                        self.after(0, lambda msg=error_msg: self.log(f"[BMA] ❌ 运行失败: {msg}"))
                        self.after(0, lambda msg=error_msg: messagebox.showwarning("BMA运行", f"BMA模型运行失败: {msg}"))
                        
                except Exception as e:
                    self._model_training = False
                    self._model_trained = False
                    error_msg = str(e)
                    self.after(0, lambda msg=error_msg: self.log(f"[BMA] 优化执行异常: {msg}"))

            # 在事件循环中运行优化的执行器
            def _start_optimized():
                try:
                    # 在事件循环中创建任务（非阻塞）
                    if hasattr(self, 'loop_manager') and self.loop_manager.is_running():
                        # 使用非阻塞方式提交协程
                        task_id = self.loop_manager.submit_coroutine_nowait(_runner_optimized())
                        self.log(f"[BMA] 任务已提交到事件循环 (ID: {task_id[:8]}...)")
                    else:
                        # 回退到线程执行
                        import asyncio
                        threading.Thread(
                            target=lambda: asyncio.run(_runner_optimized()), 
                            daemon=True
                        ).start()
                        self.log("[BMA] 使用后台线程执行")
                except Exception as e:
                    self.log(f"[BMA] 启动优化执行失败: {e}")
                    # 回退到原始方法（已移除subprocess部分）
                    self._model_training = False
                    self._model_trained = False

            _start_optimized()

        except Exception as e:
            self.log(f"[BMA] 启动失败: {e}")
            messagebox.showerror("错误", f"启动BMA失败: {e}")

    def _build_backtest_tab(self, parent) -> None:
        """构建回测分析选项卡"""
        # 回测类型选择
        backtest_type_frame = tk.LabelFrame(parent, text="回测类型")
        backtest_type_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 回测类型选择变量
        self.backtest_type = tk.StringVar(value="autotrader")
        
        # AutoTrader BMA 回测
        tk.Radiobutton(
            backtest_type_frame, 
            text="AutoTrader BMA 回测", 
            variable=self.backtest_type, 
            value="autotrader"
        ).pack(anchor=tk.W, padx=10, pady=2)
        
        # 周频 BMA 回测
        tk.Radiobutton(
            backtest_type_frame, 
            text="周频 BMA 回测", 
            variable=self.backtest_type, 
            value="weekly"
        ).pack(anchor=tk.W, padx=10, pady=2)
        
        # 回测参数配置
        config_frame = tk.LabelFrame(parent, text="回测参数配置")
        config_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 第一行：日期范围
        row1 = tk.Frame(config_frame)
        row1.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(row1, text="开始日期:").pack(side=tk.LEFT)
        self.ent_bt_start_date = tk.Entry(row1, width=12)
        self.ent_bt_start_date.insert(0, "2022-01-01")
        self.ent_bt_start_date.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row1, text="结束日期:").pack(side=tk.LEFT)
        self.ent_bt_end_date = tk.Entry(row1, width=12)
        self.ent_bt_end_date.insert(0, "2023-12-31")
        self.ent_bt_end_date.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row1, text="初始资金:").pack(side=tk.LEFT)
        self.ent_bt_capital = tk.Entry(row1, width=10)
        self.ent_bt_capital.insert(0, "100000")
        self.ent_bt_capital.pack(side=tk.LEFT, padx=5)
        
        # 第二行：策略参数
        row2 = tk.Frame(config_frame)
        row2.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(row2, text="最大持仓:").pack(side=tk.LEFT)
        self.ent_bt_max_positions = tk.Entry(row2, width=8)
        self.ent_bt_max_positions.insert(0, "20")
        self.ent_bt_max_positions.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row2, text="调仓频率:").pack(side=tk.LEFT)
        self.cb_bt_rebalance = ttk.Combobox(row2, values=["daily", "weekly"], width=8)
        self.cb_bt_rebalance.set("weekly")
        self.cb_bt_rebalance.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row2, text="手续费率:").pack(side=tk.LEFT)
        self.ent_bt_commission = tk.Entry(row2, width=8)
        self.ent_bt_commission.insert(0, "0.001")
        self.ent_bt_commission.pack(side=tk.LEFT, padx=5)
        
        # 第三行：BMA 模型参数
        row3 = tk.Frame(config_frame)
        row3.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(row3, text="模型重训周期:").pack(side=tk.LEFT)
        self.ent_bt_retrain_freq = tk.Entry(row3, width=8)
        self.ent_bt_retrain_freq.insert(0, "4")
        self.ent_bt_retrain_freq.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row3, text="预测周期:").pack(side=tk.LEFT)
        self.ent_bt_prediction_horizon = tk.Entry(row3, width=8)
        self.ent_bt_prediction_horizon.insert(0, "5")
        self.ent_bt_prediction_horizon.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row3, text="止损比例:").pack(side=tk.LEFT)
        self.ent_bt_stop_loss = tk.Entry(row3, width=8)
        self.ent_bt_stop_loss.insert(0, "0.08")
        self.ent_bt_stop_loss.pack(side=tk.LEFT, padx=5)
        
        # 第四行：风险控制参数
        row4 = tk.Frame(config_frame)
        row4.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(row4, text="最大仓位权重:").pack(side=tk.LEFT)
        self.ent_bt_max_weight = tk.Entry(row4, width=8)
        self.ent_bt_max_weight.insert(0, "0.15")
        self.ent_bt_max_weight.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row4, text="止盈比例:").pack(side=tk.LEFT)
        self.ent_bt_take_profit = tk.Entry(row4, width=8)
        self.ent_bt_take_profit.insert(0, "0.20")
        self.ent_bt_take_profit.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row4, text="滑点率:").pack(side=tk.LEFT)
        self.ent_bt_slippage = tk.Entry(row4, width=8)
        self.ent_bt_slippage.insert(0, "0.002")
        self.ent_bt_slippage.pack(side=tk.LEFT, padx=5)
        
        # 输出设置
        output_frame = tk.LabelFrame(parent, text="输出设置")
        output_frame.pack(fill=tk.X, padx=5, pady=5)
        
        row5 = tk.Frame(output_frame)
        row5.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(row5, text="输出目录:").pack(side=tk.LEFT)
        self.ent_bt_output_dir = tk.Entry(row5, width=30)
        self.ent_bt_output_dir.insert(0, "./backtest_results")
        self.ent_bt_output_dir.pack(side=tk.LEFT, padx=5)
        
        tk.Button(row5, text="浏览", command=self._browse_backtest_output_dir).pack(side=tk.LEFT, padx=5)
        
        # 选项
        options_frame = tk.Frame(output_frame)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.var_bt_export_excel = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="导出Excel报告", variable=self.var_bt_export_excel).pack(side=tk.LEFT, padx=10)
        
        self.var_bt_show_plots = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="显示图表", variable=self.var_bt_show_plots).pack(side=tk.LEFT, padx=10)
        
        # 操作按钮
        action_frame = tk.LabelFrame(parent, text="操作")
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        button_frame = tk.Frame(action_frame)
        button_frame.pack(pady=10)
        
        # 运行单个回测
        tk.Button(
            button_frame, 
            text="运行回测", 
            command=self._run_single_backtest,
            bg="lightgreen", 
            font=("Arial", 10, "bold"),
            width=15
        ).pack(side=tk.LEFT, padx=10)
        
        # 运行策略对比
        tk.Button(
            button_frame, 
            text="策略对比", 
            command=self._run_strategy_comparison,
            bg="lightblue", 
            font=("Arial", 10, "bold"),
            width=15
        ).pack(side=tk.LEFT, padx=10)
        
        # 快速回测（预设参数）
        tk.Button(
            button_frame, 
            text="快速回测", 
            command=self._run_quick_backtest,
            bg="orange", 
            font=("Arial", 10, "bold"),
            width=15
        ).pack(side=tk.LEFT, padx=10)
        
        # 回测状态显示
        status_frame = tk.LabelFrame(parent, text="回测状态")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 进度条
        self.bt_progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.bt_progress.pack(fill=tk.X, padx=5, pady=5)
        
        # 状态文本
        self.bt_status_text = tk.Text(status_frame, height=8, wrap=tk.WORD)
        bt_scrollbar = tk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.bt_status_text.yview)
        self.bt_status_text.configure(yscrollcommand=bt_scrollbar.set)
        self.bt_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        bt_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _browse_backtest_output_dir(self):
        """浏览回测输出目录"""
        directory = filedialog.askdirectory(title="选择回测结果输出目录")
        if directory:
            self.ent_bt_output_dir.delete(0, tk.END)
            self.ent_bt_output_dir.insert(0, directory)
    
    def _run_single_backtest(self):
        """运行单个回测"""
        try:
            # 获取参数
            backtest_type = self.backtest_type.get()
            
            # 验证参数
            start_date = self.ent_bt_start_date.get()
            end_date = self.ent_bt_end_date.get()
            
            # 验证日期格式
            from datetime import datetime
            try:
                datetime.strptime(start_date, '%Y-%m-%d')
                datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                messagebox.showerror("错误", "日期格式错误，请使用 YYYY-MM-DD 格式")
                return
            
            # 显示进度
            self.bt_progress.start()
            self._update_backtest_status("开始回测...")
            
            # 在新线程中运行回测
            threading.Thread(
                target=self._execute_backtest_thread,
                args=(backtest_type,),
                daemon=True
            ).start()
            
        except Exception as e:
            self.bt_progress.stop()
            self._update_backtest_status(f"回测启动失败: {e}")
            messagebox.showerror("错误", f"回测启动失败: {e}")
    
    def _run_strategy_comparison(self):
        """运行策略对比"""
        try:
            self.bt_progress.start()
            self._update_backtest_status("开始策略对比回测...")
            
            # 在新线程中运行策略对比
            threading.Thread(
                target=self._execute_strategy_comparison_thread,
                daemon=True
            ).start()
            
        except Exception as e:
            self.bt_progress.stop()
            self._update_backtest_status(f"策略对比启动失败: {e}")
            messagebox.showerror("错误", f"策略对比启动失败: {e}")
    
    def _run_quick_backtest(self):
        """快速回测（使用预设参数）"""
        try:
            # 设置快速回测的预设参数
            self.ent_bt_start_date.delete(0, tk.END)
            self.ent_bt_start_date.insert(0, "2023-01-01")
            
            self.ent_bt_end_date.delete(0, tk.END)  
            self.ent_bt_end_date.insert(0, "2023-12-31")
            
            self.ent_bt_capital.delete(0, tk.END)
            self.ent_bt_capital.insert(0, "50000")
            
            self.ent_bt_max_positions.delete(0, tk.END)
            self.ent_bt_max_positions.insert(0, "10")
            
            # 运行回测
            self._run_single_backtest()
            
        except Exception as e:
            messagebox.showerror("错误", f"快速回测失败: {e}")
    
    def _execute_backtest_thread(self, backtest_type):
        """在线程中执行回测"""
        try:
            if backtest_type == "autotrader":
                self._run_autotrader_backtest()
            elif backtest_type == "weekly":
                self._run_weekly_backtest()
                
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda msg=error_msg: self._update_backtest_status(f"回测执行失败: {msg}"))
            self.after(0, lambda msg=error_msg: messagebox.showerror("错误", f"回测执行失败: {msg}"))
        finally:
            self.after(0, lambda: self.bt_progress.stop())
    
    def _execute_strategy_comparison_thread(self):
        """在线程中执行策略对比"""
        try:
            # 修复：使用backtest_engine中的回测功能（run_backtest已合并到backtest_engine）
            from autotrader.backtest_engine import run_preset_backtests
            
            self.after(0, lambda: self._update_backtest_status("开始执行策略对比..."))
            
            # 运行预设策略对比
            run_preset_backtests()
            
            self.after(0, lambda: self._update_backtest_status("策略对比完成！结果已保存到 ./strategy_comparison.csv"))
            self.after(0, lambda: messagebox.showinfo("完成", "策略对比回测完成！\n结果已保存到当前目录"))
            
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda msg=error_msg: self._update_backtest_status(f"策略对比失败: {msg}"))
            self.after(0, lambda msg=error_msg: messagebox.showerror("错误", f"策略对比失败: {msg}"))
        finally:
            self.after(0, lambda: self.bt_progress.stop())
    
    def _run_autotrader_backtest(self):
        """运行 AutoTrader BMA 回测"""
        try:
            from autotrader.backtest_engine import AutoTraderBacktestEngine, BacktestConfig
            from autotrader.backtest_analyzer import analyze_backtest_results
            
            self.after(0, lambda: self._update_backtest_status("创建 AutoTrader 回测配置..."))
            
            # 构建配置
            config = BacktestConfig(
                start_date=self.ent_bt_start_date.get(),
                end_date=self.ent_bt_end_date.get(),
                initial_capital=float(self.ent_bt_capital.get()),
                rebalance_freq=self.cb_bt_rebalance.get(),
                max_positions=int(self.ent_bt_max_positions.get()),
                commission_rate=float(self.ent_bt_commission.get()),
                slippage_rate=float(self.ent_bt_slippage.get()),
                use_bma_model=True,
                model_retrain_freq=int(self.ent_bt_retrain_freq.get()),
                prediction_horizon=int(self.ent_bt_prediction_horizon.get()),
                max_position_weight=float(self.ent_bt_max_weight.get()),
                stop_loss_pct=float(self.ent_bt_stop_loss.get()),
                take_profit_pct=float(self.ent_bt_take_profit.get())
            )
            
            self.after(0, lambda: self._update_backtest_status("初始化回测引擎..."))
            
            # 创建回测引擎
            engine = AutoTraderBacktestEngine(config)
            
            self.after(0, lambda: self._update_backtest_status("执行回测..."))
            
            # 运行回测
                                # 回测功能已整合到backtest_engine.py
            from .backtest_engine import run_backtest_with_config
            results = run_backtest_with_config(config)
            
            if results:
                self.after(0, lambda: self._update_backtest_status("生成分析报告..."))
                
                # 生成分析报告
                output_dir = self.ent_bt_output_dir.get()
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                analyzer = analyze_backtest_results(results, output_dir)
                
                # 显示结果摘要
                summary = f"""
AutoTrader BMA 回测完成！

回测期间: {results['period']['start_date']} -> {results['period']['end_date']}
总收益率: {results['returns']['total_return']:.2%}
年化收益率: {results['returns']['annual_return']:.2%}
夏普比率: {results['returns']['sharpe_ratio']:.3f}
最大回撤: {results['returns']['max_drawdown']:.2%}
胜率: {results['returns']['win_rate']:.2%}
交易次数: {results['trading']['total_trades']}
最终资产: ${results['portfolio']['final_value']:,.2f}

报告已保存到: {output_dir}
                """
                
                self.after(0, lambda: self._update_backtest_status(summary))
                self.after(0, lambda s=summary: messagebox.showinfo("回测完成", f"AutoTrader BMA 回测完成！\n\n{s}"))
                
            else:
                self.after(0, lambda: self._update_backtest_status("回测失败：无结果数据"))
                
        except ImportError as e:
            error_msg = str(e)
            self.after(0, lambda msg=error_msg: self._update_backtest_status(f"导入回测模块失败: {msg}"))
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda msg=error_msg: self._update_backtest_status(f"AutoTrader 回测失败: {msg}"))
            import traceback
            traceback.print_exc()
    
    def _run_weekly_backtest(self):
        """运行周频 BMA 回测（内置引擎，无外部脚本依赖）"""
        try:
            from autotrader.backtest_engine import BacktestConfig, run_backtest_with_config
            from autotrader.backtest_analyzer import analyze_backtest_results

            self.after(0, lambda: self._update_backtest_status("创建周频回测配置..."))

            # 使用与AutoTrader相同的引擎，设置周频调仓
            config = BacktestConfig(
                start_date=self.ent_bt_start_date.get(),
                end_date=self.ent_bt_end_date.get(),
                initial_capital=float(self.ent_bt_capital.get()),
                rebalance_freq="weekly",
                max_positions=int(self.ent_bt_max_positions.get()),
                commission_rate=float(self.ent_bt_commission.get()),
                slippage_rate=float(self.ent_bt_slippage.get()),
                use_bma_model=True,
                model_retrain_freq=int(self.ent_bt_retrain_freq.get()),
                prediction_horizon=int(self.ent_bt_prediction_horizon.get()),
                max_position_weight=float(self.ent_bt_max_weight.get()),
                stop_loss_pct=float(self.ent_bt_stop_loss.get()),
                take_profit_pct=float(self.ent_bt_take_profit.get())
            )

            self.after(0, lambda: self._update_backtest_status("执行周频回测..."))

            results = run_backtest_with_config(config)

            if results:
                # 生成分析报告
                output_dir = self.ent_bt_output_dir.get()
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                analyze_backtest_results(results, output_dir)

                summary = f"""
周频 BMA 回测完成！

回测期间: {results['period']['start_date']} -> {results['period']['end_date']}
总收益率: {results['returns']['total_return']:.2%}
年化收益率: {results['returns']['annual_return']:.2%}
夏普比率: {results['returns']['sharpe_ratio']:.3f}
最大回撤: {results['returns']['max_drawdown']:.2%}
胜率: {results['returns']['win_rate']:.2%}
交易次数: {results['trading']['total_trades']}
最终资产: ${results['portfolio']['final_value']:,.2f}

报告已保存到: {output_dir}
                """

                self.after(0, lambda: self._update_backtest_status(summary))
                self.after(0, lambda s=summary: messagebox.showinfo("回测完成", f"周频 BMA 回测完成！\n\n{s}"))
            else:
                self.after(0, lambda: self._update_backtest_status("周频回测失败：无结果数据"))

        except ImportError as e:
            error_msg = str(e)
            self.after(0, lambda msg=error_msg: self._update_backtest_status(f"导入回测模块失败: {msg}"))
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda msg=error_msg: self._update_backtest_status(f"周频回测失败: {msg}"))
            import traceback
            traceback.print_exc()
    
    def _update_backtest_status(self, message):
        """更新回测状态"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.bt_status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.bt_status_text.see(tk.END)
        self.update_idletasks()

    def _build_status_panel(self, parent):
        """构建引擎运行状态面板"""
        # 状态信息显示区域
        status_info = tk.Frame(parent)
        status_info.pack(fill=tk.X, padx=5, pady=5)
        
        # 第一行：连接状态和引擎状态
        row1 = tk.Frame(status_info)
        row1.pack(fill=tk.X, pady=2)
        
        tk.Label(row1, text="连接状态:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        self.lbl_connection_status = tk.Label(row1, text="未连接", fg="red", font=("Arial", 9))
        self.lbl_connection_status.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row1, text="引擎状态:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)
        self.lbl_engine_status = tk.Label(row1, text="未启动", fg="gray", font=("Arial", 9))
        self.lbl_engine_status.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row1, text="模型状态:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)
        self.lbl_model_status = tk.Label(row1, text="未训练", fg="orange", font=("Arial", 9))
        self.lbl_model_status.pack(side=tk.LEFT, padx=5)
        
        # 第二行：账户信息和交易统计
        row2 = tk.Frame(status_info)
        row2.pack(fill=tk.X, pady=2)
        
        tk.Label(row2, text="净值:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        self.lbl_net_value = tk.Label(row2, text="$0.00", fg="blue", font=("Arial", 9))
        self.lbl_net_value.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row2, text="账户ID:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)
        self.lbl_account_id = tk.Label(row2, text="-", fg="black", font=("Arial", 9))
        self.lbl_account_id.pack(side=tk.LEFT, padx=5)

        tk.Label(row2, text="ClientID:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)
        self.lbl_client_id = tk.Label(row2, text="-", fg="black", font=("Arial", 9))
        self.lbl_client_id.pack(side=tk.LEFT, padx=5)

        tk.Label(row2, text="持仓数:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)
        self.lbl_positions = tk.Label(row2, text="0", fg="purple", font=("Arial", 9))
        self.lbl_positions.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row2, text="今日交易:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)
        self.lbl_daily_trades = tk.Label(row2, text="0", fg="green", font=("Arial", 9))
        self.lbl_daily_trades.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row2, text="最后更新:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)
        self.lbl_last_update = tk.Label(row2, text="未开始", fg="gray", font=("Arial", 9))
        self.lbl_last_update.pack(side=tk.LEFT, padx=5)
        
        # 第三行：操作统计和警告
        row3 = tk.Frame(status_info)
        row3.pack(fill=tk.X, pady=2)
        
        tk.Label(row3, text="监控股票:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        self.lbl_watch_count = tk.Label(row3, text="0", fg="teal", font=("Arial", 9))
        self.lbl_watch_count.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row3, text="信号生成:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)
        self.lbl_signal_status = tk.Label(row3, text="等待中", fg="orange", font=("Arial", 9))
        self.lbl_signal_status.pack(side=tk.LEFT, padx=5)
        
        # 状态指示灯
        self.lbl_status_indicator = tk.Label(row3, text="●", fg="red", font=("Arial", 14))
        self.lbl_status_indicator.pack(side=tk.RIGHT, padx=15)
        
        tk.Label(row3, text="运行状态:", font=("Arial", 9, "bold")).pack(side=tk.RIGHT, padx=5)
        
        # 启动状态更新定时器
        self._start_status_monitor()
    
    def _start_status_monitor(self):
        """启动状态监控定时器"""
        self._update_status()
        # 每2秒更新一次状态
        self.after(2000, self._start_status_monitor)
    
    def _update_status(self):
        """更新状态显示"""
        try:
            # 更新连接状态
            if self.trader and hasattr(self.trader, 'ib') and self.trader.ib.isConnected():
                self.lbl_connection_status.config(text="已连接", fg="green")
            else:
                self.lbl_connection_status.config(text="未连接", fg="red")
            
            # 更新引擎状态
            if self.engine:
                if hasattr(self, '_engine_loop_task') and self._engine_loop_task and not self._engine_loop_task.done():
                    self.lbl_engine_status.config(text="运行中", fg="green")
                    self.lbl_status_indicator.config(fg="green")
                else:
                    self.lbl_engine_status.config(text="已启动", fg="blue")
                    self.lbl_status_indicator.config(fg="blue")
            else:
                self.lbl_engine_status.config(text="未启动", fg="gray")
                self.lbl_status_indicator.config(fg="red")
            
            # 更新账户信息
            if self.trader and hasattr(self.trader, 'net_liq'):
                # 使用缓存避免短期为0/None导致闪烁
                try:
                    current_net = getattr(self.trader, 'net_liq', None)
                    if isinstance(current_net, (int, float)) and current_net is not None:
                        if self._last_net_liq is None or abs(float(current_net) - float(self._last_net_liq)) > 1e-6:
                            self._last_net_liq = float(current_net)
                    if self._last_net_liq is not None:
                        self.lbl_net_value.config(text=f"${self._last_net_liq:,.2f}")
                except Exception:
                    pass
                # 更新账户ID与客户端ID
                try:
                    acc_id = getattr(self.trader, 'account_id', None)
                    if acc_id:
                        self.lbl_account_id.config(text=str(acc_id), fg=("green" if str(acc_id).lower()=="c2dvdongg" else "black"))
                    else:
                        self.lbl_account_id.config(text="-", fg="black")
                except Exception:
                    pass
                try:
                    # 与当前配置的 client_id 对齐，而不是固定 3130
                    actual_cid = getattr(self.trader, 'client_id', None)
                    try:
                        expected_cid = self.config_manager.get('connection.client_id', None)
                    except Exception:
                        expected_cid = None
                    cid_ok = bool(actual_cid is not None and expected_cid is not None and actual_cid == expected_cid)
                    self.lbl_client_id.config(text=str(actual_cid if actual_cid is not None else '-'), fg=("green" if cid_ok else "black"))
                except Exception:
                    pass
                
                # 更新持仓数
                position_count = len(getattr(self.trader, 'positions', {}))
                self.lbl_positions.config(text=str(position_count))
            
            # 更新监控股票数
            if self.trader and hasattr(self.trader, 'tickers'):
                watch_count = len(getattr(self.trader, 'tickers', {}))
                self.lbl_watch_count.config(text=str(watch_count))
            
            # 更新最后更新时间
            current_time = datetime.now().strftime("%H:%M:%S")
            self.lbl_last_update.config(text=current_time)
            
            # 检查模型状态（如果有相关属性）
            if hasattr(self, '_model_training') and self._model_training:
                self.lbl_model_status.config(text="训练中", fg="blue")
            elif hasattr(self, '_model_trained') and self._model_trained:
                self.lbl_model_status.config(text="已训练", fg="green")
            else:
                self.lbl_model_status.config(text="未训练", fg="orange")
                
        except Exception as e:
            # 状态更新失败不应该影响主程序
            pass
    
    def _update_signal_status(self, status_text, color="black"):
        """更新信号状态"""
        try:
            self.lbl_signal_status.config(text=status_text, fg=color)
        except:
            pass
    
    def _update_daily_trades(self, count):
        """更新今日交易次数"""
        try:
            self.lbl_daily_trades.config(text=str(count))
        except:
            pass


def main() -> None:
    # 清理：移除未使用的导入
    # import tkinter.simpledialog  # 导入对话框模块
    app = AutoTraderGUI()  # type: ignore
    app.mainloop()


if __name__ == "__main__":
    main()

