from __future__ import annotations

import asyncio
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import dataclass
from typing import Optional, List
import os
import sys
import subprocess
from datetime import datetime, timedelta

from .ibkr_auto_trader import IbkrAutoTrader
from .engine import Engine
from .config import HotConfig
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
    client_id: int = 123
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
        self.state = AppState()
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
        self.hot_config: Optional[HotConfig] = HotConfig()
        self._loop_thread: Optional[threading.Thread] = None
        self._engine_loop_task: Optional[asyncio.Task] = None
        
        # Ensure proper cleanup on window close
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

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

        # 策略引擎选项卡（集成模式2）
        engine_frame = ttk.Frame(notebook)
        notebook.add(engine_frame, text="策略引擎")
        self._build_engine_tab(engine_frame)

        # 直接交易选项卡（集成模式3）
        direct_frame = ttk.Frame(notebook)
        notebook.add(direct_frame, text="直接交易")
        self._build_direct_tab(direct_frame)

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
        tk.Button(act, text="启动自动交易", command=self._start_autotrade, bg="lightgreen").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="停止交易", command=self._stop, bg="orange").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="清空日志", command=self._clear_log, bg="lightgray").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="查看账户", command=self._show_account, bg="lightyellow").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="一键运行BMA模型", command=self._run_bma_model, bg="#d8b7ff").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="打印数据库", command=self._print_database, bg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(act, text="一键删除数据库", command=self._delete_database, bg="#ff6666").pack(side=tk.RIGHT, padx=5)

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
                self.log("风险配置已保存")
            else:
                self.log("风险配置保存失败")
        except Exception as e:
            self.log(f"保存风险配置失败: {e}")

    def _build_engine_tab(self, parent) -> None:
        frm = ttk.Frame(parent)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        box = ttk.LabelFrame(frm, text="策略引擎控制")
        box.pack(fill=tk.X, pady=8)

        ttk.Button(box, text="启动引擎(连接/订阅)", command=self._start_engine).grid(row=0, column=0, padx=6, pady=6)
        ttk.Button(box, text="运行一次信号与交易", command=self._engine_once).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(box, text="停止引擎", command=self._stop_engine_mode).grid(row=0, column=2, padx=6, pady=6)

        tip = ttk.Label(frm, text="说明: 引擎使用统一配置 HotConfig 扫描 universe，计算多因子信号并下单。")
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
            loop = self._ensure_loop()
            async def _run():
                try:
                    if not self.trader:
                        self.trader = IbkrAutoTrader(self.state.host, self.state.port, self.state.client_id, use_delayed_if_no_realtime=True)
                    await self.trader.connect()
                    self.engine = Engine(self.hot_config or HotConfig(), self.trader)
                    await self.engine.start()
                    self.log("策略引擎已启动并完成订阅")
                except Exception as e:
                    self.log(f"策略引擎启动失败: {e}")
            asyncio.run_coroutine_threadsafe(_run(), loop)
        except Exception as e:
            self.log(f"启动引擎错误: {e}")

    def _engine_once(self) -> None:
        try:
            if not self.engine:
                self.log("请先启动引擎")
                return
            loop = self._ensure_loop()
            asyncio.run_coroutine_threadsafe(self.engine.on_signal_and_trade(), loop)
            self.log("已触发一次信号与交易")
        except Exception as e:
            self.log(f"运行引擎一次失败: {e}")

    def _stop_engine_mode(self) -> None:
        try:
            self.log("策略引擎停止：可通过停止交易按钮一并断开连接与任务")
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
                        self.trader = IbkrAutoTrader(self.state.host, self.state.port, self.state.client_id, use_delayed_if_no_realtime=True)
                        await self.trader.connect()
                    await self.trader.place_market_order(sym, side, qty)
                    self.log(f"已提交市价单: {side} {qty} {sym}")
                except Exception as e:
                    self.log(f"市价单失败: {e}")
            asyncio.run_coroutine_threadsafe(_run(), loop)
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
                        self.trader = IbkrAutoTrader(self.state.host, self.state.port, self.state.client_id, use_delayed_if_no_realtime=True)
                        await self.trader.connect()
                    await self.trader.place_limit_order(sym, side, qty, px)
                    self.log(f"已提交限价单: {side} {qty} {sym} @ {px}")
                except Exception as e:
                    self.log(f"限价单失败: {e}")
            asyncio.run_coroutine_threadsafe(_run(), loop)
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
                        self.trader = IbkrAutoTrader(self.state.host, self.state.port, self.state.client_id, use_delayed_if_no_realtime=True)
                        await self.trader.connect()
                    await self.trader.place_market_order_with_bracket(sym, side, qty, stop_pct=stop_pct, target_pct=tp_pct)
                    self.log(f"已提交括号单: {side} {qty} {sym} (止损{stop_pct*100:.1f}%, 止盈{tp_pct*100:.1f}%)")
                except Exception as e:
                    self.log(f"括号单失败: {e}")
            asyncio.run_coroutine_threadsafe(_run(), loop)
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
                        self.trader = IbkrAutoTrader(self.state.host, self.state.port, self.state.client_id, use_delayed_if_no_realtime=True)
                        await self.trader.connect()
                    await self.trader.execute_large_order(sym, side, qty, algorithm=algo, duration_minutes=dur_min)
                    self.log(f"已提交大单执行: {algo} {side} {qty} {sym} / {dur_min}min")
                except Exception as e:
                    self.log(f"大单执行失败: {e}")
            asyncio.run_coroutine_threadsafe(_run(), loop)
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
        # 左侧：股票列表
        left_frame = tk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 股票列表选择
        list_frame = tk.LabelFrame(left_frame, text="股票列表")
        list_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(list_frame, text="选择列表:").grid(row=0, column=0, padx=5, pady=5)
        self.stock_list_var = tk.StringVar()
        self.stock_list_combo = ttk.Combobox(list_frame, textvariable=self.stock_list_var, state="readonly", width=20)
        self.stock_list_combo.grid(row=0, column=1, padx=5, pady=5)
        self.stock_list_combo.bind("<<ComboboxSelected>>", self._on_stock_list_changed)
        
        tk.Button(list_frame, text="新建列表", command=self._create_stock_list, bg="lightblue").grid(row=0, column=2, padx=5)
        tk.Button(list_frame, text="删除列表", command=self._delete_stock_list, bg="lightcoral").grid(row=0, column=3, padx=5)
        tk.Button(list_frame, text="刷新", command=self._refresh_stock_lists, bg="lightgray").grid(row=0, column=4, padx=5)
        
        # 股票表格
        stock_frame = tk.LabelFrame(left_frame, text="股票列表")
        stock_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建Treeview
        columns = ('symbol', 'name', 'added_at')
        self.stock_tree = ttk.Treeview(stock_frame, columns=columns, show='headings', height=10)
        self.stock_tree.heading('symbol', text='股票代码')
        self.stock_tree.heading('name', text='公司名称')
        self.stock_tree.heading('added_at', text='添加时间')
        self.stock_tree.column('symbol', width=100)
        self.stock_tree.column('name', width=200)
        self.stock_tree.column('added_at', width=150)
        
        # 滚动条
        stock_scroll = ttk.Scrollbar(stock_frame, orient=tk.VERTICAL, command=self.stock_tree.yview)
        self.stock_tree.configure(yscrollcommand=stock_scroll.set)
        
        self.stock_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        stock_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # 右侧：操作面板
        right_frame = tk.Frame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # 添加股票
        add_frame = tk.LabelFrame(right_frame, text="添加股票")
        add_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(add_frame, text="股票代码:").grid(row=0, column=0, padx=5, pady=5)
        self.ent_symbol = tk.Entry(add_frame, width=15)
        self.ent_symbol.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(add_frame, text="公司名称:").grid(row=1, column=0, padx=5, pady=5)
        self.ent_stock_name = tk.Entry(add_frame, width=15)
        self.ent_stock_name.grid(row=1, column=1, padx=5, pady=5)
        
        tk.Button(add_frame, text="添加股票", command=self._add_stock, bg="lightgreen").grid(row=2, column=0, columnspan=2, pady=5)
        
        # 批量导入
        import_frame = tk.LabelFrame(right_frame, text="批量导入")
        import_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(import_frame, text="CSV格式:").grid(row=0, column=0, padx=5, pady=5)
        self.ent_batch_csv = tk.Text(import_frame, width=20, height=4)
        self.ent_batch_csv.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        self.ent_batch_csv.insert(tk.END, "AAPL,MSFT,GOOGL")
        
        tk.Button(import_frame, text="批量导入", command=self._batch_import, bg="lightyellow").grid(row=2, column=0, columnspan=2, pady=5)
        
        # 删除股票
        delete_frame = tk.LabelFrame(right_frame, text="删除股票")
        delete_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(delete_frame, text="删除选中", command=self._delete_selected_stock, bg="lightcoral").grid(row=0, column=0, padx=5, pady=5)
        
        # 配置管理
        config_frame = tk.LabelFrame(right_frame, text="配置管理")
        config_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(config_frame, text="配置名称:").grid(row=0, column=0, padx=5, pady=5)
        self.config_name_var = tk.StringVar()
        self.config_combo = ttk.Combobox(config_frame, textvariable=self.config_name_var, width=15)
        self.config_combo.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        tk.Button(config_frame, text="保存配置", command=self._save_config, bg="lightblue").grid(row=2, column=0, padx=2, pady=5)
        tk.Button(config_frame, text="加载配置", command=self._load_config, bg="lightgreen").grid(row=2, column=1, padx=2, pady=5)
        
        # 初始化数据
        self._refresh_stock_lists()
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
                safe_log("事件循环已启动")
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
        
        # Wait for loop to be ready with timeout; log warning instead of hard failing silently for user
        max_wait = 60  # 6 seconds max
        for _ in range(max_wait):
            if self.loop and self.loop.is_running():
                return self.loop
            import time
            time.sleep(0.1)
        # If still not running, provide a helpful log and raise
        self.log("事件循环未能在预期时间内启动，请重试‘测试连接’或‘启动自动交易’。")
        raise RuntimeError("Failed to start event loop")
            
        return self.loop

    def _capture_ui(self) -> None:
        self.state.host = self.ent_host.get().strip() or "127.0.0.1"
        try:
            self.state.port = int(self.ent_port.get().strip())
            self.state.client_id = int(self.ent_cid.get().strip())
            self.state.alloc = float(self.ent_alloc.get().strip() or 0.03)
            self.state.poll_sec = float(self.ent_poll.get().strip() or 10.0)
            self.state.fixed_qty = int(self.ent_fixed_qty.get().strip() or 0)
        except Exception:
            messagebox.showerror("参数错误", "端口/ClientId必须是整数，资金占比/轮询间隔必须是数字")
            raise
        self.state.sheet = self.ent_sheet.get().strip() or None
        self.state.column = self.ent_col.get().strip() or None
        self.state.symbols_csv = self.ent_csv.get().strip() or None
        self.state.auto_sell_removed = self.var_auto_sell.get()

    def _test_connection(self) -> None:
        try:
            self._capture_ui()
            self.log("正在测试连接...")
            loop = self._ensure_loop()

            async def _run():
                try:
                    self.trader = IbkrAutoTrader(self.state.host, self.state.port, self.state.client_id, use_delayed_if_no_realtime=True)
                    await self.trader.connect()
                    self.log("[OK] 连接成功")
                except Exception as e:
                    self.log(f"[FAIL] 连接失败: {e}")
            
            future = asyncio.run_coroutine_threadsafe(_run(), loop)
            # 不等待结果，避免阻塞GUI
            
        except Exception as e:
            self.log(f"测试连接错误: {e}")
            messagebox.showerror("错误", f"测试连接失败: {e}")

    def _start_autotrade(self) -> None:
        try:
            self._capture_ui()
            self.log("正在启动自动交易（策略引擎模式）...")
            loop = self._ensure_loop()

            async def _run():
                try:
                    # 1) 准备 Trader 连接
                    if not self.trader:
                        self.trader = IbkrAutoTrader(self.state.host, self.state.port, self.state.client_id, use_delayed_if_no_realtime=True)
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
                    cfg = self.hot_config or HotConfig()
                    if uni:
                        cfg.get()["CONFIG"]["scanner"]["universe"] = uni
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
                except Exception as e:
                    self.log(f"自动交易启动失败: {e}")

            asyncio.run_coroutine_threadsafe(_run(), loop)

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
                except Exception as e:
                    self.log(f"停止策略循环失败: {e}")

                # Close trader connection
                if self.loop and self.loop.is_running():
                    async def _cleanup_trader():
                        try:
                            await self.trader.close()
                            self.log("交易连接已关闭")
                        except Exception as e:
                            self.log(f"关闭交易连接失败: {e}")
                        finally:
                            self.trader = None
                            
                    asyncio.run_coroutine_threadsafe(_cleanup_trader(), self.loop)
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
                    
            future = asyncio.run_coroutine_threadsafe(_run(), loop)
            
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
        """添加股票"""
        try:
            if not self.state.selected_stock_list_id:
                messagebox.showwarning("警告", "请先选择股票列表")
                return
                
            symbol = self.ent_symbol.get().strip().upper()
            name = self.ent_stock_name.get().strip()
            
            if not symbol:
                messagebox.showwarning("警告", "请输入股票代码")
                return
                
            if self.db.add_stock(self.state.selected_stock_list_id, symbol, name):
                self.log(f"成功添加股票: {symbol}")
                self.ent_symbol.delete(0, tk.END)
                self.ent_stock_name.delete(0, tk.END)
                self._refresh_stock_table(self.state.selected_stock_list_id)
                self._refresh_stock_lists()  # 更新股票数量
            else:
                messagebox.showwarning("警告", f"股票 {symbol} 已在列表中")
                
        except Exception as e:
            self.log(f"添加股票失败: {e}")
            messagebox.showerror("错误", f"添加失败: {e}")
    
    def _batch_import(self):
        """批量导入股票"""
        try:
            if not self.state.selected_stock_list_id:
                messagebox.showwarning("警告", "请先选择股票列表")
                return
                
            csv_text = self.ent_batch_csv.get(1.0, tk.END).strip()
            if not csv_text:
                messagebox.showwarning("警告", "请输入股票代码")
                return
                
            success, fail = self.db.import_from_csv(self.state.selected_stock_list_id, csv_text)
            
            self.log(f"批量导入完成: 成功 {success} 个，失败 {fail} 个")
            self.ent_batch_csv.delete(1.0, tk.END)
            self._refresh_stock_table(self.state.selected_stock_list_id)
            self._refresh_stock_lists()
            
        except Exception as e:
            self.log(f"批量导入失败: {e}")
            messagebox.showerror("错误", f"批量导入失败: {e}")
    
    def _delete_selected_stock(self):
        """删除选中的股票"""
        try:
            if not self.state.selected_stock_list_id:
                messagebox.showwarning("警告", "请先选择股票列表")
                return
                
            selected_items = self.stock_tree.selection()
            if not selected_items:
                messagebox.showwarning("警告", "请先选择要删除的股票")
                return
                
            symbols = []
            for item in selected_items:
                values = self.stock_tree.item(item, 'values')
                symbols.append(values[0])
            
            result = messagebox.askyesno("确认删除", f"确定要删除这些股票吗？\n{', '.join(symbols)}")
            
            if result:
                success_count = 0
                for symbol in symbols:
                    if self.db.remove_stock(self.state.selected_stock_list_id, symbol):
                        success_count += 1
                        
                self.log(f"成功删除 {success_count} 只股票")
                self._refresh_stock_table(self.state.selected_stock_list_id)
                self._refresh_stock_lists()
                
        except Exception as e:
            self.log(f"删除股票失败: {e}")
            messagebox.showerror("错误", f"删除失败: {e}")
    
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
                self.log(f"成功保存配置: {name}")
                self._refresh_configs()
                self.config_name_var.set(name)
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
                    asyncio.run_coroutine_threadsafe(
                        self._auto_sell_stocks(removed_before), self.loop)
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


    def _on_closing(self) -> None:
        """Enhanced cleanup when closing the application with proper resource management"""
        try:
            self.log("正在关闭应用...")
            
            # First, gracefully stop trader
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
                    # Close trader connection if exists
                    if self.trader and self.loop and self.loop.is_running():
                        async def _cleanup_trader():
                            try:
                                await self.trader.close()
                                self.log("交易器连接已关闭")
                            except Exception as e:
                                self.log(f"交易器关闭失败: {e}")
                        
                        try:
                            future = asyncio.run_coroutine_threadsafe(_cleanup_trader(), self.loop)
                            future.result(timeout=2.0)  # 2 second timeout
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

            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, '量化模型_bma_enhanced.py'))
            if not os.path.exists(script_path):
                messagebox.showerror("错误", f"未找到量化模型脚本: {script_path}")
                return

            self.log(f"[BMA] 启动BMA增强模型: {start_date} -> {end_date} (默认全股票池)")

            def _runner():
                try:
                    cmd = [sys.executable, script_path, '--start-date', start_date, '--end-date', end_date]
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        bufsize=1,
                        universal_newlines=True,
                        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
                    )
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        line = line.rstrip('\n')
                        if line:
                            try:
                                self.after(0, lambda m=line: self.log(m))
                            except Exception:
                                pass
                    code = proc.wait()
                    self.after(0, lambda: self.log(f"[BMA] 运行完成，退出码={code}"))
                    if code != 0:
                        self.after(0, lambda: messagebox.showwarning("BMA运行", f"BMA模型运行异常（退出码 {code}），请查看日志"))
                except Exception as e:
                    self.after(0, lambda: self.log(f"[BMA] 运行失败: {e}"))

            threading.Thread(target=_runner, daemon=True).start()

        except Exception as e:
            self.log(f"[BMA] 启动失败: {e}")
            messagebox.showerror("错误", f"启动BMA失败: {e}")


def main() -> None:
    import tkinter.simpledialog  # 导入对话框模块
    app = AutoTraderGUI()  # type: ignore
    app.mainloop()


if __name__ == "__main__":
    main()

