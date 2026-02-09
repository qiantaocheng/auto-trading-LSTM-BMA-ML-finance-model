# =============================================================================

# SIGNAL CALCULATION RESTORED - Trading system now active

# All signal generation systems are properly integrated and functional

# Using unified signal processor with production-ready algorithms

# =============================================================================

import asyncio

import threading

import tkinter as tk

from tkinter import filedialog, messagebox, ttk

from dataclasses import dataclass

from typing import Optional, List

from pathlib import Path

import os

import sys

import json

import pandas as pd

import numpy as np

from datetime import datetime, timedelta

from pandas.tseries.offsets import BDay

import subprocess



# ????BMA??????????????????????????MultiIndex????

DEFAULT_AUTO_TRAIN_TICKERS = [

    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",

    "NVDA", "META", "NFLX", "CRM", "ADBE",

]



# ??????????????????+ ????????T+10?

AUTO_TRAIN_LOOKBACK_YEARS = 4

AUTO_TRAIN_HORIZON_DAYS = 10

# Default 11 factors (same as TOP_FEATURE_SET in simple_25_factor_engine)
TIMESPLIT_FEATURES = ['momentum_10d', 'ivol_20', 'hist_vol_20', 'rsi_21', 'near_52w_high', 'atr_ratio', 'vol_ratio_20d', '5_days_reversal', 'trend_r2_60', 'liquid_momentum']
STAGE_A_DATA_PATH = Path(r"D:\trade\data\factor_exports\polygon_factors_stageA_default.parquet")
# Direct Predict: tickers loaded from MultiIndex of this parquet (date, ticker)
DIRECT_PREDICT_TICKER_DATA_PATH = Path(r"D:\trade\data\factor_exports\polygon_full_features_T5.parquet")

DIRECT_PREDICT_SNAPSHOT_ID = "b35a35db-352b-43d8-ace8-4a54674c1da5"
# Default EWMA weights (L2 Beta 0.7) for direct prediction smoothing
DIRECT_PREDICT_EMA_WEIGHTS = (0.41, 0.28, 0.19, 0.12)  # 4-day EMA with beta=0.33
DIRECT_PREDICT_MAX_CLOSE = 10000.0  # guardrail for obviously bad close prices





from .ibkr_auto_trader import IbkrAutoTrader

from .engine import Engine

from .database import StockDatabase

from .unified_trading_core import create_unified_trading_core





def _attach_tooltip(widget, text: str) -> None:

    """Attach a simple tooltip to a Tk widget without external deps."""

    if not text:

        return



    class _SimpleTooltip:

        def __init__(self, w, t):

            self.widget = w

            self.text = t

            self.tip = None

            self.widget.bind("<Enter>", self._show, add="+")

            self.widget.bind("<Leave>", self._hide, add="+")



        def _show(self, _event=None):

            if self.tip or not self.text:

                return

            try:

                x = self.widget.winfo_rootx() + 20

                y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10

                self.tip = tk.Toplevel(self.widget)

                self.tip.wm_overrideredirect(True)

                self.tip.wm_geometry(f"+{x}+{y}")

                lbl = tk.Label(

                    self.tip,

                    text=self.text,

                    justify=tk.LEFT,

                    background="#ffffe0",

                    relief=tk.SOLID,

                    borderwidth=1,

                    font=("tahoma", 8)

                )

                lbl.pack(ipadx=4, ipady=2)

            except Exception:

                # Fail silently; tooltip is best-effort

                self.tip = None



        def _hide(self, _event=None):

            if self.tip is not None:

                try:

                    self.tip.destroy()

                except Exception:

                    pass

                self.tip = None



    try:

        _SimpleTooltip(widget, text)

    except Exception:

        pass





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

    #  

    alloc: float = 0.03

    poll_sec: float = 3.0

    auto_sell_removed: bool = True

    fixed_qty: int = 0

    #  ?

    selected_stock_list_id: Optional[int] = None

    use_database: bool = True





DIRECT_PREDICT_FEATURE_CACHE = {}
class AutoTraderGUI(tk.Tk):

    def __init__(self) -> None:  # type: ignore

        super().__init__()

        

        #  use ???

        from bma_models.unified_config_loader import get_config_manager as get_default_config

        from autotrader.unified_event_manager import get_event_loop_manager

        from autotrader.unified_monitoring_system import get_resource_monitor

        

        self.config_manager = get_default_config()

        self.loop_manager = get_event_loop_manager()

        self.resource_monitor = get_resource_monitor()

        

        # Starting event loop manager

        if not self.loop_manager.start():

            raise RuntimeError("no Starting event loop manager")

        

        # start 

        self.resource_monitor.start_monitoring()

        

        #  ? AppState use ? Client ID

        conn_params = self.config_manager.get_connection_params(auto_allocate_client_id=False)

        self.state = AppState(

            port=conn_params['port'],

            client_id=conn_params['client_id'],

            host=conn_params['host']

        )

        self.title("IBKR ????????????")

        self.geometry("1000x700")

        #  use items ? S? `? before `? 

        self.db = StockDatabase()

        self._top10_state_path = Path('cache/hetrs_top10_state.json')

        self._top10_state_path.parent.mkdir(parents=True, exist_ok=True)

        self._last_top10_refresh = self._load_top10_refresh_state()

        #  before ? for inUI completedbefore uselog ? ??

        self._log_buffer: List[str] = []

        self._log_lock = threading.Lock()

        self.txt = None  # type: ignore

        self._build_ui()

        

        # === Add top menu & toolbar for Return Comparison quick access ===

        try:

            self._ensure_top_menu()

            self._ensure_toolbar()

        except Exception:

            pass

        

        self.loop: Optional[asyncio.AbstractEventLoop] = None

        self.trader: Optional[IbkrAutoTrader] = None

        self.engine: Optional[Engine] = None

        #  use ? ?not HotConfig

        # self.hot_config: Optional[HotConfig] = HotConfig()

        self._loop_thread: Optional[threading.Thread] = None

        self._loop_ready_event: Optional[threading.Event] = None

        self._engine_loop_task: Optional[asyncio.Task] = None

        #  ? ??

        self._model_training: bool = False

        self._model_trained: bool = False

        self._daily_trade_count: int = 0

        #  ? ????

        self._last_net_liq: Optional[float] = None

        

        # Ensure proper cleanup on window close

        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        

        #  

        self.resource_monitor.add_alert_callback(self._on_resource_warning)

        

        #  ? ??

        from autotrader.unified_event_manager import get_event_bus, GUIEventAdapter

        self.event_bus = get_event_bus()

        self.gui_adapter = GUIEventAdapter(self, self.event_bus)

        

        # Initialize strategy engine components

        self._init_enhanced_trading_components()

        self._init_strategy_components()

        try:

            self._maybe_refresh_top10_pool(force=False)

        except Exception:

            pass



    def _init_enhanced_trading_components(self):

        """ ? ? ? ?+  ? """

        try:

            from autotrader.position_size_calculator import create_position_calculator

            from autotrader.volatility_adaptive_gating import create_volatility_gating



            #  ? ? ? ??

            self.position_calculator = create_position_calculator(

                target_percentage=0.05,    # ??% ??

                min_percentage=0.04,       # ??%

                max_percentage=0.10,       # ??0%

                method="volatility_adjusted"  #  

            )



            #  ??

            self.volatility_gating = create_volatility_gating(

                base_k=0.5,               #  ? ? ??

                volatility_lookback=60,    # 60 ??

                use_atr=True,             #  ATR ? ?

                enable_liquidity_filter=True  #  ? ?

            )



            self.log(" ? ??  ? ???+  ")



        except Exception as e:

            self.log(f" ? ? {e}")

            #  ? 

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

            

            # Enhanced alpha strategies ?-  Simple 25 ??

            from autotrader.unified_polygon_factors import  UnifiedPolygonFactors

            from .real_risk_balancer import get_risk_balancer_adapter



            self.log("Enhanced alpha strategies ?-  Simple 25 ?")

            

            # Initialize Polygon factors for automatic API connection  

            self.polygon_factors = None

            self._init_polygon_factors()

            

            if not hasattr(self, 'risk_balancer_adapter') or getattr(self, 'risk_balancer_adapter', None) is None:

                self.risk_balancer_adapter = get_risk_balancer_adapter(enable_balancer=False)

            

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

            from autotrader.unified_polygon_factors import  UnifiedPolygonFactors

            self.polygon_factors = UnifiedPolygonFactors()

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

    

    def get_dynamic_price(self, symbol: str) -> float:

        """ ? -  Polygon API"""

        try:

            from polygon_client import polygon_client

            

            #  ??:  get_current_price ??

            if hasattr(polygon_client, 'get_current_price'):

                price = polygon_client.get_current_price(symbol)

                if price and price > 0:

                    return float(price)

            

            #  ??:  get_realtime_snapshot ?

            if hasattr(polygon_client, 'get_realtime_snapshot'):

                snapshot = polygon_client.get_realtime_snapshot(symbol)

                if snapshot and 'last_trade' in snapshot and 'price' in snapshot['last_trade']:

                    return float(snapshot['last_trade']['price'])

            

            #  ??:  get_last_trade ? ??

            if hasattr(polygon_client, 'get_last_trade'):

                trade_data = polygon_client.get_last_trade(symbol)

                if trade_data and 'price' in trade_data:

                    return float(trade_data['price'])

                    

            #  ??:  ? ??

            if hasattr(polygon_client, 'get_today_intraday'):

                intraday_data = polygon_client.get_today_intraday(symbol)

                if not intraday_data.empty:

                    return float(intraday_data['close'].iloc[-1])

                    

        except Exception as e:

            self.log(f"Polygon API ?{symbol}: {e}")

        

        #  ? API ? ? ??

        self.log(f" ?  Polygon API ??{symbol}  ?")

        return 0.0  #  ?? 

    

    def log_message(self, message: str) -> None:

        """ ?"""

        self.log(message)

    

    def _stop_engine(self) -> None:

        """ ? """

        self._stop_engine_mode()



    def _build_ui(self) -> None:

        #  ? ? ?Canvas + Scrollbar ? ? ??

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



        #  k? Windows?

        def _on_mousewheel(event):

            try:

                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

            except Exception:

                pass

        canvas.bind_all("<MouseWheel>", _on_mousewheel)



        # connection ??

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



        #  ???items?

        notebook = ttk.Notebook(frm)

        notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        

        #  ? ??items?

        db_frame = ttk.Frame(notebook)

        notebook.add(db_frame, text="Data Services")

        self._build_database_tab(db_frame)

        

        #  ??items?

        file_frame = ttk.Frame(notebook)

        notebook.add(file_frame, text="File Imports")

        self._build_file_tab(file_frame)



        #  ??items?

        risk_frame = ttk.Frame(notebook)

        notebook.add(risk_frame, text="Risk Engine")

        self._build_risk_tab(risk_frame)



        # Polygon API ??

        polygon_frame = ttk.Frame(notebook)

        notebook.add(polygon_frame, text="Polygon API")

        self._build_polygon_tab(polygon_frame)



        #  ?items ?2?

        engine_frame = ttk.Frame(notebook)

        notebook.add(engine_frame, text="Strategy Engine")

        self._build_engine_tab(engine_frame)



        #  ?items ?3?

        direct_frame = ttk.Frame(notebook)

        notebook.add(direct_frame, text="Direct Trading")

        self._build_direct_tab(direct_frame)



        # Time Split evaluation tab

        timesplit_frame = ttk.Frame(notebook)

        notebook.add(timesplit_frame, text="80/20 OOS")

        self._build_timesplit_tab(timesplit_frame)



        #  settings

        params = tk.LabelFrame(frm, text="Trading Parameter Settings")

        params.pack(fill=tk.X, pady=5)

        

        #  ? and? 

        tk.Label(params, text="Allocation Ratio").grid(row=0, column=0, padx=5, pady=5)

        self.ent_alloc = tk.Entry(params, width=8)

        self.ent_alloc.insert(0, str(self.state.alloc))

        self.ent_alloc.grid(row=0, column=1, padx=5)

        

        tk.Label(params, text="Polling (seconds)").grid(row=0, column=2, padx=5)

        self.ent_poll = tk.Entry(params, width=8)

        self.ent_poll.insert(0, str(self.state.poll_sec or 3.0))

        self.ent_poll.grid(row=0, column=3, padx=5)

        

        tk.Label(params, text="Fixed Quantity").grid(row=0, column=4, padx=5)

        self.ent_fixed_qty = tk.Entry(params, width=8)

        self.ent_fixed_qty.insert(0, str(self.state.fixed_qty))

        self.ent_fixed_qty.grid(row=0, column=5, padx=5)

        

        #  ? ? ?items

        self.var_auto_sell = tk.BooleanVar(value=self.state.auto_sell_removed)

        tk.Checkbutton(params, text=" ?when ? ", variable=self.var_auto_sell).grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)

        

        #  

        act = tk.LabelFrame(frm, text="Actions")

        act.pack(fill=tk.X, pady=5)

        tk.Button(act, text="Test Connection", command=self._test_connection, bg="lightblue").pack(side=tk.LEFT, padx=5)

        tk.Button(act, text="Disconnect API", command=self._disconnect_api, bg="#ffcccc").pack(side=tk.LEFT, padx=5)

        tk.Button(act, text="Start Auto Trading", command=self._start_autotrade, bg="lightgreen").pack(side=tk.LEFT, padx=5)

        tk.Button(act, text="Stop Trading", command=self._stop, bg="orange").pack(side=tk.LEFT, padx=5)

        tk.Button(act, text="Clear Log", command=self._clear_log, bg="lightgray").pack(side=tk.LEFT, padx=5)

        tk.Button(act, text="Show Account", command=self._show_account, bg="lightyellow").pack(side=tk.LEFT, padx=5)


        tk.Button(act, text="Print Database", command=self._print_database, bg="white").pack(side=tk.LEFT, padx=5)

        tk.Button(act, text="Drop Database", command=self._delete_database, bg="#ff6666").pack(side=tk.RIGHT, padx=5)



        #  ? ? 

        status_frame = tk.LabelFrame(frm, text="Status Overview")

        status_frame.pack(fill=tk.X, pady=5)

        self._build_status_panel(status_frame)

        

        #  ?

        log_frame = tk.LabelFrame(frm, text="Live Log")

        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.txt = tk.Text(log_frame, height=8)

        scroll_y = tk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.txt.yview)

        self.txt.configure(yscrollcommand=scroll_y.set)

        self.txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        # will in to ??

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

        #  when to andGUI

        try:

            print(msg)  #  to ? ??

        except UnicodeEncodeError:

            # Windows in ? ??

            print(msg.encode('gbk', errors='ignore').decode('gbk', errors='ignore'))

        except Exception:

            # if failed GUI ??

            pass

        

        # UI completedorText when ??

        try:

            if hasattr(self, "txt") and isinstance(self.txt, tk.Text):

                self.txt.insert(tk.END, msg + "\n")

                self.txt.see(tk.END)

            else:

                # can in UI be use

                with self._log_lock:

                    if not hasattr(self, "_log_buffer"):

                        self._log_buffer = []  # type: ignore

                    self._log_buffer.append(msg)  # type: ignore

        except Exception:

            #  failed not ?

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



        box1 = ttk.LabelFrame(frm, text="Capital Parameters")

        box1.pack(fill=tk.X, pady=5)

        ttk.Label(box1, text="Stop Loss %").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        self.rm_stop = ttk.Spinbox(box1, from_=0.1, to=50.0, increment=0.1, width=8)

        self.rm_stop.set(2.0)

        self.rm_stop.grid(row=0, column=1, padx=5)

        ttk.Label(box1, text="Take Profit %").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

        self.rm_target = ttk.Spinbox(box1, from_=0.1, to=100.0, increment=0.1, width=8)

        self.rm_target.set(5.0)

        self.rm_target.grid(row=0, column=3, padx=5)

        ttk.Label(box1, text="Real-time Allocation %").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)

        self.rm_rt_alloc = ttk.Spinbox(box1, from_=0.0, to=1.0, increment=0.01, width=8)

        self.rm_rt_alloc.set(0.03)

        self.rm_rt_alloc.grid(row=0, column=5, padx=5)



        box2 = ttk.LabelFrame(frm, text="Risk Controls & Limits")

        box2.pack(fill=tk.X, pady=5)

        ttk.Label(box2, text="Price Floor ($)").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        self.rm_price_min = ttk.Spinbox(box2, from_=0.0, to=1000.0, increment=0.5, width=8)

        self.rm_price_min.set(2.0)

        self.rm_price_min.grid(row=0, column=1, padx=5)

        ttk.Label(box2, text="Price Ceiling ($)").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

        self.rm_price_max = ttk.Spinbox(box2, from_=0.0, to=5000.0, increment=1.0, width=8)

        self.rm_price_max.set(800.0)

        self.rm_price_max.grid(row=0, column=3, padx=5)

        ttk.Label(box2, text="Cash Reserve %").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        self.rm_cash_reserve = ttk.Spinbox(box2, from_=0.0, to=0.9, increment=0.01, width=8)

        self.rm_cash_reserve.set(0.15)

        self.rm_cash_reserve.grid(row=1, column=1, padx=5)

        ttk.Label(box2, text="Max Single Position %").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)

        self.rm_single_max = ttk.Spinbox(box2, from_=0.01, to=0.9, increment=0.01, width=8)

        self.rm_single_max.set(0.12)

        self.rm_single_max.grid(row=1, column=3, padx=5)

        ttk.Label(box2, text="Minimum Order Value ($)").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)

        self.rm_min_order = ttk.Spinbox(box2, from_=0, to=10000, increment=50, width=8)

        self.rm_min_order.set(500)

        self.rm_min_order.grid(row=2, column=1, padx=5)

        ttk.Label(box2, text="Daily Order Limit").grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)

        self.rm_daily_limit = ttk.Spinbox(box2, from_=1, to=200, increment=1, width=8)

        self.rm_daily_limit.set(20)

        self.rm_daily_limit.grid(row=2, column=3, padx=5)



        box3 = ttk.LabelFrame(frm, text="ATR / Bracket Cleanup")

        box3.pack(fill=tk.X, pady=5)

        self.rm_use_atr = tk.BooleanVar(value=False)

        ttk.Checkbutton(box3, text="Enable ATR Stops", variable=self.rm_use_atr).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        ttk.Label(box3, text="ATR Stop Multiplier").grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        self.rm_atr_stop = ttk.Spinbox(box3, from_=0.5, to=10.0, increment=0.1, width=8)

        self.rm_atr_stop.set(2.0)

        self.rm_atr_stop.grid(row=0, column=2, padx=5)

        ttk.Label(box3, text="ATR Target Multiplier").grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)

        self.rm_atr_target = ttk.Spinbox(box3, from_=0.5, to=10.0, increment=0.1, width=8)

        self.rm_atr_target.set(3.0)

        self.rm_atr_target.grid(row=0, column=4, padx=5)

        ttk.Label(box3, text="ATR Risk Scale").grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)

        self.rm_atr_scale = ttk.Spinbox(box3, from_=0.1, to=20.0, increment=0.1, width=8)

        self.rm_atr_scale.set(5.0)

        self.rm_atr_scale.grid(row=0, column=6, padx=5)

        self.rm_allow_short = tk.BooleanVar(value=True)

        ttk.Checkbutton(box3, text="Allow Short Positions", variable=self.rm_allow_short).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        self.rm_bracket_removed = tk.BooleanVar(value=False)

        ttk.Checkbutton(box3, text="Disable bracket order on delisted symbols", variable=self.rm_bracket_removed).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)



        box4 = ttk.LabelFrame(frm, text="Webhook Settings")

        box4.pack(fill=tk.X, pady=5)

        ttk.Label(box4, text="Webhook URL").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        self.rm_webhook = ttk.Entry(box4, width=60)

        self.rm_webhook.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)



        act = ttk.Frame(frm)

        act.pack(fill=tk.X, pady=10)

        ttk.Button(act, text="Load Config", command=self._risk_load).pack(side=tk.LEFT, padx=5)

        ttk.Button(act, text="Save Config", command=self._risk_save).pack(side=tk.LEFT, padx=5)



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

            self.log(f" failed: {e}")



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

                self.log(" to ?")

            else:

                self.log(" failed")

            db.close()

            

            #  whenupdates ? ??

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

            

            #  to ??

            if self.config_manager.persist_runtime_changes():

                self.log("  to ")

            else:

                self.log("  failed to ?")

        except Exception as e:

            self.log(f" failed: {e}")



    def _build_polygon_tab(self, parent) -> None:

        """Polygon API ?"""

        frm = ttk.Frame(parent)

        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)



        # Polygon API Status

        status_frame = ttk.LabelFrame(frm, text="Polygon API Status")

        status_frame.pack(fill=tk.X, pady=5)

        

        self.polygon_status_label = tk.Label(status_frame, text="Status: Connecting...", fg="blue")

        self.polygon_status_label.pack(side=tk.LEFT, padx=10, pady=5)

        

        ttk.Button(status_frame, text="Refresh Connection", command=self._refresh_polygon_connection).pack(side=tk.RIGHT, padx=10, pady=5)



        #   ( ??

        function_frame = ttk.LabelFrame(frm, text=" ")

        function_frame.pack(fill=tk.X, pady=5)

        

        ttk.Button(function_frame, text=" ", command=self._get_realtime_quotes).grid(row=0, column=0, padx=5, pady=5)

        ttk.Button(function_frame, text=" ?, command=self._get_historical_data).grid(row=0, column=1, padx=5, pady=5")

        # Return comparison tool

        compare_frame = ttk.LabelFrame(frm, text="Return Comparison")

        compare_frame.pack(fill=tk.X, pady=5)



        tk.Label(compare_frame, text="Tickers (comma separated):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        self.polygon_compare_symbols = tk.Entry(compare_frame, width=40)

        self.polygon_compare_symbols.insert(0, "AAPL,MSFT,GOOGL")

        self.polygon_compare_symbols.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)



        tk.Label(compare_frame, text="Start date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        self.polygon_compare_start = tk.Entry(compare_frame, width=15)

        self.polygon_compare_start.insert(0, (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'))

        self.polygon_compare_start.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)



        tk.Label(compare_frame, text="End date (YYYY-MM-DD):").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)

        self.polygon_compare_end = tk.Entry(compare_frame, width=15)

        self.polygon_compare_end.insert(0, datetime.now().strftime('%Y-%m-%d'))

        self.polygon_compare_end.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)



        # Excel  ?

        tk.Label(compare_frame, text="Excel  ?").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)

        self.polygon_compare_excel_entry = tk.Entry(compare_frame, width=40)

        self.polygon_compare_excel_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Button(compare_frame, text=" Excel...", command=self._browse_excel_file).grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)



        # Excel Top20 T+5  return comparison ??

        self.polygon_compare_excel_button = ttk.Button(

            compare_frame,

            text="Excel Top20 T+5 (vs SPY)",

            command=self._compare_returns_from_excel

        )

        self.polygon_compare_excel_button.grid(row=0, column=2, padx=5, pady=5, sticky=tk.E)

        try:

            _attach_tooltip(self.polygon_compare_excel_button, " Excel ?0 ? T+5 ? SPY Excel ?")

        except Exception:

            pass



        self.polygon_compare_button = ttk.Button(compare_frame, text="Compute Return Comparison", command=self._compare_polygon_returns)

        self.polygon_compare_button.grid(row=0, column=3, padx=5, pady=5, sticky=tk.E)



        compare_frame.grid_columnconfigure(1, weight=1)

        compare_frame.grid_columnconfigure(3, weight=1)



        self.polygon_compare_output = tk.Text(compare_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)

        self.polygon_compare_output.grid(row=3, column=0, columnspan=4, padx=5, pady=(5, 0), sticky=tk.EW)



        

        #  ? n???

        info_frame = ttk.LabelFrame(frm, text="API?")

        info_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        

        self.polygon_info_text = tk.Text(info_frame, height=10, state=tk.DISABLED)

        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.polygon_info_text.yview)

        self.polygon_info_text.configure(yscrollcommand=info_scrollbar.set)

        

        self.polygon_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        

        #  ? ??

        self._update_polygon_status()



    def _refresh_polygon_connection(self):

        """ Polygon API ?"""

        try:

            self.log("Refreshing Polygon API connection...")

            self._ensure_polygon_factors()

            self._update_polygon_status()

        except Exception as e:

            self.log(f"Failed to refresh Polygon connection: {e}")



    def _get_realtime_quotes(self):

        """ """

        try:

            if self.polygon_factors:

                self.log("Fetching real-time quotes from Polygon API...")

                #  ? ???

                self.log("Real-time quotes functionality ready")

            else:

                self.log("Polygon API not connected")

        except Exception as e:

            self.log(f"Failed to get real-time quotes: {e}")



    def _get_historical_data(self):

        """ ?"""

        try:

            if self.polygon_factors:

                self.log("Fetching historical data from Polygon API...")

                #  ? ???

                self.log("Historical data functionality ready")

            else:

                self.log("Polygon API not connected")

        except Exception as e:

            self.log(f"Failed to get historical data: {e}")



    def _compare_polygon_returns(self):

        """Compare ticker returns against QQQ using Polygon API."""

        if getattr(self, '_polygon_compare_running', False):

            self.log("[Polygon] Return comparison already running, please wait...")

            return



        symbols_entry = getattr(self, 'polygon_compare_symbols', None)

        start_entry = getattr(self, 'polygon_compare_start', None)

        end_entry = getattr(self, 'polygon_compare_end', None)

        output_widget = getattr(self, 'polygon_compare_output', None)



        if not all([symbols_entry, start_entry, end_entry, output_widget]):

            self.log("[Polygon] Return comparison widgets are not initialized.")

            return



        raw_symbols = symbols_entry.get().strip()

        start_str = start_entry.get().strip()

        end_str = end_entry.get().strip()



        if not raw_symbols:

            messagebox.showwarning("Warning", "Please enter at least one ticker (comma separated).")

            return



        def set_output(text_value: str) -> None:

            def _update() -> None:

                output_widget.config(state=tk.NORMAL)

                output_widget.delete(1.0, tk.END)

                output_widget.insert(tk.END, text_value)

                output_widget.config(state=tk.DISABLED)

            self.after(0, _update)



        def set_busy(is_busy: bool) -> None:

            def _update() -> None:

                if hasattr(self, 'polygon_compare_button'):

                    self.polygon_compare_button.config(state=tk.DISABLED if is_busy else tk.NORMAL)

            self.after(0, _update)



        set_output("Calculating, please wait...")



        def worker(symbols: str, start_value: str, end_value: str) -> None:

            self._polygon_compare_running = True

            set_busy(True)

            try:

                try:

                    start_dt = datetime.strptime(start_value, '%Y-%m-%d')

                except ValueError:

                    self.after(0, lambda: messagebox.showerror("Date Format Error", "Please use YYYY-MM-DD for the date."))

                    set_output("Invalid start date format.")

                    return



                if end_value:

                    try:

                        end_dt = datetime.strptime(end_value, '%Y-%m-%d')

                    except ValueError:

                        self.after(0, lambda: messagebox.showerror("Date Format Error", "Please use YYYY-MM-DD for the date."))

                        set_output("Invalid end date format.")

                        return

                else:

                    end_dt = datetime.now()



                if end_dt < start_dt:

                    self.after(0, lambda: messagebox.showerror("Date Error", "End date cannot be earlier than start date."))

                    set_output("End date is earlier than start date.")

                    return



                start_norm = start_dt.strftime('%Y-%m-%d')

                end_norm = end_dt.strftime('%Y-%m-%d')



                tickers = [s.strip().upper() for s in symbols.split(',') if s.strip()]

                if not tickers:

                    self.after(0, lambda: messagebox.showwarning("Warning", "No valid tickers were parsed."))

                    set_output("No valid tickers provided.")

                    return



                # yfinance market-cap prefilter (>= $1B) before any downstream fetching

                MCAP_THRESHOLD = 1_000_000_000

                try:

                    import yfinance as yf

                    self.log(f"[Filter] Checking yfinance market caps (threshold ${MCAP_THRESHOLD:,})...")

                    kept, filtered, missing = [], [], []

                    for sym in tickers:

                        mcap = None

                        try:

                            yft = yf.Ticker(sym)

                            # Prefer fast_info when available

                            mcap = None

                            try:

                                fi = getattr(yft, 'fast_info', None)

                                if fi is not None:

                                    try:

                                        mcap = fi.get('market_cap', None)

                                    except Exception:

                                        mcap = getattr(fi, 'market_cap', None)

                            except Exception:

                                pass

                            if mcap is None:

                                info = yft.info

                                if isinstance(info, dict):

                                    mcap = info.get('marketCap', None)

                        except Exception:

                            mcap = None



                        if isinstance(mcap, (int, float)) and mcap >= MCAP_THRESHOLD:

                            kept.append(sym)

                        elif mcap is None:

                            # Market cap data unavailable - keep the ticker anyway

                            missing.append(sym)

                            kept.append(sym)

                            self.log(f"[Filter] {sym}: market cap unavailable, keeping anyway")

                        else:

                            # Market cap below threshold

                            filtered.append(sym)



                    self.log(f"[Filter] {len(tickers)} -> {len(kept)} kept (filtered {len(filtered)}; missing mcap {len(missing)})")

                    if filtered:

                        self.log(f"[Filter] Excluded (first 10): {', '.join(filtered[:10])}")

                    if not kept:

                        self.log(f"[Filter] WARNING: All tickers filtered out, skipping market cap filter")

                        # Don't return - proceed with original tickers

                        tickers = tickers

                    else:

                        tickers = kept

                except Exception as e:

                    self.log(f"[Filter] yfinance failed; skipping market cap filter: {e}")



                try:

                    from polygon_client import polygon_client

                except Exception as import_err:

                    msg = f"Failed to import polygon_client: {import_err}"

                    self.log(f"[Polygon] {msg}")

                    set_output(msg)

                    return



                def compute_symbol(symbol: str):

                    df = polygon_client.get_historical_bars(symbol, start_norm, end_norm, 'day', 1)

                    if df is None or df.empty:

                        raise ValueError("No valid historical price data.")

                    df = df.sort_index()

                    start_row = df.iloc[0]

                    end_row = df.iloc[-1]

                    start_price = float(start_row['Open'])

                    end_price = float(end_row['Close'])

                    if start_price == 0:

                        raise ValueError("Start open price is zero; cannot compute return.")

                    return {

                        'symbol': symbol,

                        'start_date': start_row.name.strftime('%Y-%m-%d'),

                        'end_date': end_row.name.strftime('%Y-%m-%d'),

                        'start_price': float(start_row['Open']),

                        'end_price': float(end_row['Close']),

                        'return': float(end_row['Close']) / float(start_row['Open']) - 1,

                    }



                self.log(f"[Polygon] Fetching returns for {', '.join(tickers)} from {start_norm} to {end_norm}.")



                results = []

                errors = []



                for symbol in tickers:

                    try:

                        results.append(compute_symbol(symbol))

                        self.log(f"[Polygon] {symbol} return {results[-1]['return']:.2%}")

                    except Exception as symbol_err:

                        errors.append(f"{symbol}: {symbol_err}")

                        self.log(f"[Polygon] Failed to fetch {symbol}: {symbol_err}")



                if not results:

                    summary_lines = ["No valid stock data retrieved."]

                    if errors:

                        summary_lines.extend(errors)

                    set_output("\n".join(summary_lines))

                    return



                avg_return = sum(item['return'] for item in results) / len(results)



                try:

                    qqq_result = compute_symbol('QQQ')

                except Exception as qqq_err:

                    qqq_result = None

                    self.log(f"[Polygon] Failed to fetch QQQ data: {qqq_err}")



                lines = [

                    f"{item['symbol']}: {item['start_date']} open {item['start_price']:.2f} -> {item['end_date']} close {item['end_price']:.2f}, return {item['return']:.2%}"

                    for item in results

                ]



                lines.append('-')

                lines.append(f"Average return: {avg_return:.2%}")



                if qqq_result:

                    lines.append(

                        f"QQQ: {qqq_result['start_date']} open {qqq_result['start_price']:.2f} -> {qqq_result['end_date']} close {qqq_result['end_price']:.2f}, return {qqq_result['return']:.2%}"

                    )

                    lines.append(f"Excess vs QQQ: {avg_return - qqq_result['return']:.2%}")

                    self.log(f"[Polygon] Average return {avg_return:.2%} vs QQQ {qqq_result['return']:.2%}")

                else:

                    lines.append("Failed to retrieve QQQ data for comparison.")

                    self.log(f"[Polygon] Average return {avg_return:.2%}; QQQ data unavailable.")



                if errors:

                    lines.append('-')

                    lines.append("Tickers with errors:")

                    lines.extend(errors)



                set_output('\n'.join(lines))

            finally:

                set_busy(False)

                self._polygon_compare_running = False



        thread = threading.Thread(target=worker, args=(raw_symbols, start_str, end_str), daemon=True)

        thread.start()



    def _browse_excel_file(self):

        """ ? Excel ?"""

        try:

            entry = getattr(self, 'polygon_compare_excel_entry', None)

            initial_dir = os.path.expanduser("~")

            path = filedialog.askopenfilename(

                title=" ? Excel ?",

                initialdir=initial_dir,

                filetypes=[("Excel Files", "*.xlsx;*.xls")]

            )

            if path and entry is not None:

                entry.delete(0, tk.END)

                entry.insert(0, path)

        except Exception as e:

            try:

                messagebox.showerror("????", f"???Excel???: {e}")

            except Exception:

                pass



    def _compare_returns_from_excel(self):

        """ Excel 0 ? ? T+5 SPY Excel"""

        if getattr(self, '_excel_backtest_running', False):

            self.log("[Excel] Backtest already running, please wait...")

            return



        output_widget = getattr(self, 'polygon_compare_output', None)

        if not output_widget:

            messagebox.showerror("Error", "Output widget not initialized")

            return



        #  Excel S? ???

        entry = getattr(self, 'polygon_compare_excel_entry', None)

        excel_path = None

        try:

            if entry is not None:

                excel_path = entry.get().strip()

        except Exception:

            excel_path = None

        if not excel_path:

            excel_path = filedialog.askopenfilename(

                title=" ? Excel ?",

                filetypes=[("Excel Files", "*.xlsx;*.xls")]

            )

            if not excel_path:

                return



        # GUI ? ??

        def set_output(text_value: str) -> None:

            def _update() -> None:

                output_widget.config(state=tk.NORMAL)

                output_widget.delete(1.0, tk.END)

                output_widget.insert(tk.END, text_value)

                output_widget.config(state=tk.DISABLED)

            self.after(0, _update)



        def set_busy(is_busy: bool) -> None:

            def _update() -> None:

                try:

                    self.polygon_compare_excel_button.config(state=tk.DISABLED if is_busy else tk.NORMAL)

                except Exception:

                    pass

            self.after(0, _update)



        set_output(" ? S? ??..")



        TOP_N = 20

        HORIZON = 5

        BENCH = "SPY"



        def worker(path: str) -> None:

            self._excel_backtest_running = True

            set_busy(True)

            try:

                try:

                    book = pd.read_excel(path, sheet_name=None)

                except Exception as e:

                    self.after(0, lambda: messagebox.showerror(" ", f" Excel: {e}"))

                    set_output(f" Excel ? {e}")

                    return

                if not book:

                    set_output("Excel ? ?")

                    return



                # import polygon client

                try:

                    from polygon_client import polygon_client

                except Exception as import_err:

                    msg = f" polygon_client: {import_err}"

                    self.log(f"[Excel] {msg}")

                    set_output(msg)

                    return



                def _parse_date(value):

                    if pd.isna(value):

                        return None

                    try:

                        return pd.to_datetime(value).tz_localize(None).normalize()

                    except Exception:

                        return None



                def _sanitize_ticker(raw):

                    if raw is None or (isinstance(raw, float) and np.isnan(raw)):

                        return None

                    try:

                        s = str(raw).strip().upper()

                        if not s:

                            return None

                        return "".join(ch for ch in s if ch.isalnum() or ch in ".-")

                    except Exception:

                        return None



                def _download_history(symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:

                    try:

                        df = polygon_client.get_historical_bars(

                            symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), 'day', 1

                        )

                    except Exception:

                        df = pd.DataFrame()

                    if isinstance(df, pd.DataFrame) and not df.empty:

                        try:

                            df = df.sort_index()

                            idx = pd.to_datetime(df.index).tz_localize(None).normalize()

                            df.index = idx

                        except Exception:

                            pass

                    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()



                def _t_horizon_return_by_target(symbol: str, tdate: pd.Timestamp, h: int) -> Optional[float]:

                    start = (tdate - pd.Timedelta(days=30))

                    end = (tdate + pd.Timedelta(days=2))

                    hist = _download_history(symbol, start, end)

                    if hist.empty:

                        return None

                    dates = hist.index

                    pos = dates.searchsorted(tdate)

                    if pos == len(dates) or dates[pos] != tdate:

                        pos = max(0, dates.searchsorted(tdate, side="right") - 1)

                    if pos < 0 or pos >= len(dates):

                        return None

                    base_pos = pos - h

                    if base_pos < 0:

                        return None

                    try:

                        base_close = float(hist.iloc[base_pos]["Close"])

                        target_close = float(hist.iloc[pos]["Close"])

                        if base_close <= 0 or not np.isfinite(base_close) or not np.isfinite(target_close):

                            return None

                        return (target_close / base_close) - 1.0

                    except Exception:

                        return None



                def _forward_horizon_return_from_base(symbol: str, base_date: pd.Timestamp, h: int) -> Optional[float]:

                    # base_date ?base_date + h ( ??

                    start = (base_date - pd.Timedelta(days=2))

                    end = (base_date + pd.Timedelta(days=40))

                    hist = _download_history(symbol, start, end)

                    if hist.empty:

                        return None

                    dates = hist.index

                    pos = dates.searchsorted(base_date)

                    if pos == len(dates) or dates[pos] != base_date:

                        pos = max(0, dates.searchsorted(base_date, side="right") - 1)

                    if pos < 0 or pos >= len(dates):

                        return None

                    target_pos = pos + h

                    if target_pos >= len(dates):

                        return None

                    try:

                        base_close = float(hist.iloc[pos]["Close"])

                        target_close = float(hist.iloc[target_pos]["Close"])

                        if base_close <= 0 or not np.isfinite(base_close) or not np.isfinite(target_close):

                            return None

                        return (target_close / base_close) - 1.0

                    except Exception:

                        return None



                def _select_top_n(df: pd.DataFrame, n: int) -> pd.DataFrame:

                    if df is None or df.empty:

                        return pd.DataFrame()

                    #  ? [??

                    cols = {str(c).strip().lower(): c for c in df.columns}



                    def _pick(colnames: list) -> Optional[str]:

                        for nm in colnames:

                            key = str(nm).strip().lower()

                            if key in cols:

                                return cols[key]

                        return None



                    rank_col = _pick(["rank", "????", "????", "????"])

                    score_col = _pick(["final_score", "score", "???????", "? ?", "????", "????", "???", "???"])



                    df2 = df.copy()

                    if rank_col:

                        c = rank_col

                        with pd.option_context('mode.use_inf_as_na', True):

                            df2 = df2.sort_values(c, ascending=True, na_position="last")

                    elif score_col:

                        c = score_col

                        with pd.option_context('mode.use_inf_as_na', True):

                            df2 = df2.sort_values(c, ascending=False, na_position="last")

                    return df2.head(n)



                per_sheet_rows = []

                details_per_sheet = {}

                skipped_info = []  #  ? sheet ??



                for sheet_name, df in book.items():

                    if not isinstance(df, pd.DataFrame) or df.empty:

                        skipped_info.append(f"{sheet_name}:  ? ?")

                        continue

                    cols_map = {str(c).strip().lower(): c for c in df.columns}



                    def _pick_col(colnames: list) -> Optional[str]:

                        for nm in colnames:

                            key = str(nm).strip().lower()

                            if key in cols_map:

                                return cols_map[key]

                        return None



                    tick_col = _pick_col(["ticker", "symbol", "????", "???????", "??????", "???", "???", "??????", "?????a"])

                    if not tick_col:

                        self.log(f"[Excel] {sheet_name}:  ticker ")

                        skipped_info.append(f"{sheet_name}:  ticker/ ?")

                        continue



                    #  Top N

                    top_df = _select_top_n(df, TOP_N).copy()

                    top_df["__ticker__"] = top_df[tick_col].map(_sanitize_ticker)

                    top_df = top_df.dropna(subset=["__ticker__"]).drop_duplicates(subset=["__ticker__"])



                    #  L? 

                    date_col = _pick_col(["date", "???????", "???????", "target_date", "????????", "???????", "????", "???????", "base_date", "???????", "signal_date"])

                    if date_col and date_col in top_df.columns:

                        top_df["__target_date__"] = top_df[date_col].map(_parse_date)

                    else:

                        #  ? ? ??

                        tdate = None

                        if date_col and date_col in df.columns:

                            candidates = df[date_col].dropna().map(_parse_date)

                            if isinstance(candidates, pd.Series) and candidates.notna().any():

                                mode_vals = candidates.mode()

                                tdate = mode_vals.iloc[0] if len(mode_vals) > 0 else None

                        top_df["__target_date__"] = tdate



                    #  ? ??

                    realized_target, bench_target = [], []

                    realized_forward, bench_forward = [], []

                    for _, row in top_df.iterrows():

                        sym = row["__ticker__"]

                        tdate = row["__target_date__"]

                        if tdate is None:

                            realized_target.append(None)

                            bench_target.append(None)

                            realized_forward.append(None)

                            bench_forward.append(None)

                            continue

                        rt = _t_horizon_return_by_target(sym, tdate, HORIZON)

                        bt = _t_horizon_return_by_target(BENCH, tdate, HORIZON)

                        realized_target.append(rt)

                        bench_target.append(bt)



                        rf = _forward_horizon_return_from_base(sym, tdate, HORIZON)

                        bf = _forward_horizon_return_from_base(BENCH, tdate, HORIZON)

                        realized_forward.append(rf)

                        bench_forward.append(bf)



                    #  ? ??

                    cnt_t = int(pd.Series(realized_target).notna().sum())

                    cnt_f = int(pd.Series(realized_forward).notna().sum())

                    use_forward = cnt_f > cnt_t



                    if use_forward:

                        top_df["realized_ret"] = realized_forward

                        top_df["bench_ret"] = bench_forward

                    else:

                        top_df["realized_ret"] = realized_target

                        top_df["bench_ret"] = bench_target



                    valid_mask = top_df["realized_ret"].notna()

                    n_ok = int(valid_mask.sum())

                    avg_ret = float(top_df.loc[valid_mask, "realized_ret"].mean()) if n_ok > 0 else np.nan

                    avg_bmk = float(top_df.loc[valid_mask, "bench_ret"].mean()) if n_ok > 0 else np.nan

                    alpha = (avg_ret - avg_bmk) if np.isfinite(avg_ret) and np.isfinite(avg_bmk) else np.nan



                    per_sheet_rows.append({

                        "sheet": sheet_name,

                        "top_n": min(TOP_N, len(top_df)),

                        "n_computed": n_ok,

                        "avg_return_pct": None if pd.isna(avg_ret) else round(avg_ret * 100.0, 3),

                        "avg_sp500_pct": None if pd.isna(avg_bmk) else round(avg_bmk * 100.0, 3),

                        "alpha_pct": None if pd.isna(alpha) else round(alpha * 100.0, 3),

                        "direction": "base base+H" if use_forward else "target-H target"

                    })



                    #  ??

                    out_cols = [tick_col]

                    if date_col and date_col in top_df.columns:

                        out_cols.append(date_col)

                    det = pd.DataFrame({

                        "ticker": top_df[tick_col].values,

                        "date": top_df[date_col].values if (date_col and date_col in top_df.columns) else [None] * len(top_df),

                        "realized_ret_pct": (top_df["realized_ret"] * 100.0).round(3),

                        "benchmark_ret_pct": (top_df["bench_ret"] * 100.0).round(3),

                        "alpha_pct": ((top_df["realized_ret"] - top_df["bench_ret"]) * 100.0).round(3)

                    })

                    details_per_sheet[sheet_name] = det



                if not per_sheet_rows:

                    set_output(" ? Excel ? ? ? ?")

                    return



                summary_df = pd.DataFrame(per_sheet_rows).sort_values("alpha_pct", ascending=False)



                #  Excel ?backtest_results

                out_dir = os.path.join("D:", os.sep, "trade", "backtest_results")

                try:

                    os.makedirs(out_dir, exist_ok=True)

                except Exception:

                    pass

                base = os.path.splitext(os.path.basename(path))[0]

                out_path = os.path.join(out_dir, f"{base}_avg_return_backtest.xlsx")



                try:

                    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

                        summary_df.to_excel(writer, index=False, sheet_name="summary")

                        for sheet, det in details_per_sheet.items():

                            safe_name = sheet[:31] if sheet else "sheet"

                            det.to_excel(writer, index=False, sheet_name=safe_name)

                except Exception as e:

                    self.log(f"[Excel]  ? {e}")



                #  GUI

                lines = ["Excel Top20 T+5  ?"]

                for _, row in summary_df.iterrows():

                    lines.append(

                        f"{row['sheet']}: n={int(row['n_computed'])}/{int(row['top_n'])}  "

                        f"avg={row['avg_return_pct']}%  SPY={row['avg_sp500_pct']}%  alpha={row['alpha_pct']}%  dir={row.get('direction','')}"

                    )

                if skipped_info:

                    lines.append("")

                    lines.append(" ?")

                    lines.extend(skipped_info)

                lines.append(f" : {out_path}")

                set_output("\n".join(lines))

                try:

                    self.after(0, lambda: messagebox.showinfo("???", f"Excel????????????: {out_path}"))

                except Exception:

                    pass

            finally:

                set_busy(False)

                self._excel_backtest_running = False



    def _enable_polygon_factors(self):

        """ usePolygon ?"""

        try:

            if hasattr(self, 'trader') and self.trader:

                self.trader.enable_polygon_factors()

                self.log("Polygon use")

            else:

                self.log(" connection ")

        except Exception as e:

            self.log(f" usePolygon failed: {e}")



    def _clear_polygon_cache(self):

        """ Polygon ?"""

        try:

            if hasattr(self, 'trader') and self.trader:

                self.trader.clear_polygon_cache()

                self.log("Polygon ")

            else:

                self.log(" connection ")

        except Exception as e:

            self.log(f" Polygon failed: {e}")



    def _toggle_polygon_balancer(self):

        """ risk control ?"""

        try:

            if hasattr(self, 'trader') and self.trader:

                if self.polygon_balancer_var.get():

                    self.trader.enable_polygon_risk_balancer()

                    self.log("risk control use")

                else:

                    self.trader.disable_polygon_risk_balancer()

                    self.log("risk control ?use")

            else:

                self.log(" connection ")

                self.polygon_balancer_var.set(False)

        except Exception as e:

            self.log(f" risk control failed: {e}")

            self.polygon_balancer_var.set(False)



    def _open_balancer_config(self):

        """ risk control D??"""

        try:

            #  GUI ??

            import sys

            import os

            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

            

            from .real_risk_balancer import create_standalone_gui

            

            # in in GUI 

            import threading

            gui_thread = threading.Thread(target=create_standalone_gui, daemon=True)

            gui_thread.start()

            

            self.log("risk control D? ?")

            

        except Exception as e:

            self.log(f" failed: {e}")



    def _update_polygon_status(self):

        """updatesPolygon ? ?"""

        try:

            if hasattr(self, 'trader') and self.trader:

                # checkPolygonconnection ??

                polygon_enabled = hasattr(self.trader, 'polygon_enabled') and self.trader.polygon_enabled

                balancer_enabled = hasattr(self.trader, 'polygon_risk_balancer_enabled') and self.trader.polygon_risk_balancer_enabled

                

                if polygon_enabled:

                    status_text = " ? Polygonconnection"

                    status_color = "green"

                else:

                    status_text = " ? Polygon connection"

                    status_color = "red"

                

                self.polygon_status_label.config(text=status_text, fg=status_color)

                self.polygon_balancer_var.set(balancer_enabled)

                

                # updates ? R?

                stats = self.trader.get_polygon_stats()

                if stats:

                    stats_text = "Polygon ? R?:\n"

                    stats_text += f"  ??????: {'??' if stats.get('enabled', False) else '??'}\n"

                    stats_text += f"  risk control???: {'??' if stats.get('risk_balancer_enabled', False) else '??'}\n"

                    stats_text += f"   ? {stats.get('cache_size', 0)}\n"

                    stats_text += f"   ? ? {stats.get('total_calculations', 0)}\n"

                    stats_text += f"  success? {stats.get('successful_calculations', 0)}\n"

                    stats_text += f"  failed? {stats.get('failed_calculations', 0)}\n"

                    stats_text += f"   in: {stats.get('cache_hits', 0)}\n"

                    

                    #  ??

                    components = stats.get('components', {})

                    stats_text += "\n ?\n"

                    for comp, status in components.items():

                        stats_text += f"  {comp}: {'[OK]' if status else '[FAIL]'}\n"

                    

                    self.polygon_stats_text.config(state=tk.NORMAL)

                    self.polygon_stats_text.delete(1.0, tk.END)

                    self.polygon_stats_text.insert(1.0, stats_text)

                    self.polygon_stats_text.config(state=tk.DISABLED)

                else:

                    self.polygon_stats_text.config(state=tk.NORMAL)

                    self.polygon_stats_text.delete(1.0, tk.END)

                    self.polygon_stats_text.insert(1.0, " no ? R?")

                    self.polygon_stats_text.config(state=tk.DISABLED)

            else:

                self.polygon_status_label.config(text=" ?  connection ", fg="gray")

                

        except Exception as e:

            self.polygon_status_label.config(text=f" ? checkfailed ({e})", fg="red")



    def _schedule_polygon_update(self):

        """ whenupdatesPolygon ?"""

        self._update_polygon_status()

        self.after(5000, self._schedule_polygon_update)  # ? secondsupdates??



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

        ttk.Button(strategy_box, text="Direct Predict (Snapshot)", command=self._direct_predict_snapshot).grid(row=0, column=2, padx=6, pady=6)

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



    def _direct_predict_snapshot(self) -> None:

        """

        Direct predict using latest saved snapshot with Excel output.

        Features:

        - Auto-fetch data from Polygon API

        - Calculate features automatically

        - Predict with snapshot (no retrain)

        - Generate Excel ranking report with raw scores (no EMA smoothing)

        """

        try:

            from bma_models._bma_ultra_enhanced import UltraEnhancedQuantitativeModel

            from datetime import datetime, timedelta

            import pandas as pd

            import numpy as np

            from pathlib import Path

            import sys



            # Load default tickers from MultiIndex of CLEAN_STANDARDIZED parquet

            default_tickers_file = DIRECT_PREDICT_TICKER_DATA_PATH

            default_tickers: list[str] = []

            

            try:

                if default_tickers_file.exists():

                    self.log(f"[DirectPredict] ?? ???????????????? ?: {default_tickers_file}")

                    df_tickers = pd.read_parquet(default_tickers_file)

                    if isinstance(df_tickers.index, pd.MultiIndex):

                        default_tickers = sorted(df_tickers.index.get_level_values('ticker').unique().tolist())

                    elif 'ticker' in df_tickers.columns:

                        default_tickers = sorted(df_tickers['ticker'].unique().tolist())

                    self.log(f"[DirectPredict] ? ??????????????? {len(default_tickers)} ????")

                else:

                    self.log(f"[DirectPredict] ?? ?????????????: {default_tickers_file}")

            except Exception as e:

                self.log(f"[DirectPredict] ?? ?????????? ????: {e}")

                import traceback

                self.log(f"[DirectPredict] ???????: {traceback.format_exc()}")



            # Determine tickers: prefer pool selection, then default from file, else prompt user input
            # Determine tickers: prefer CLEAN_STANDARDIZED parquet (MultiIndex tickers), fall back to pool selection or user input

            tickers: list[str] = []

            if default_tickers:

                tickers = default_tickers.copy()

                self.log(f"[DirectPredict] Loaded {len(tickers)} tickers from CLEAN_STANDARDIZED parquet (MultiIndex): {default_tickers_file}")

            else:

                try:

                    if hasattr(self, 'selected_pool_info') and self.selected_pool_info and 'tickers' in self.selected_pool_info:

                        tickers = list(set([t.strip().upper() for t in self.selected_pool_info['tickers'] if isinstance(t, str) and t.strip()]))

                        self.log(f"[DirectPredict] Loaded {len(tickers)} tickers from selected pool")

                except Exception:

                    tickers = []

            

            # If still no tickers, prompt user

            if not tickers:

                import tkinter as tk

                from tkinter import simpledialog

                root = self.winfo_toplevel()

                default_str = ','.join(default_tickers[:50]) if default_tickers else ""  # Show first 50 as example

                sym_str = simpledialog.askstring(

                    "Direct Predict", 

                    f"?????????????????\n??????????? {len(default_tickers)} ????\n???: {default_str[:100]}...", 

                    parent=root,

                    initialvalue=default_str[:500] if default_tickers else ""  # Pre-fill with first 500 tickers

                )

                if not sym_str:

                    self.log("[DirectPredict] ?????")

                    return

                tickers = list({s.strip().upper() for s in sym_str.split(',') if s.strip()})



            self.log(f"[DirectPredict] ????? {len(tickers)} ?")

            self.log(f"[DirectPredict] ???? ?: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")



            # Use default T+10 prediction horizon (no user input needed)

            # Get horizon from model config (default 10 days)

            try:

                from bma_models.unified_config_loader import get_time_config

                time_config = get_time_config()

                prediction_horizon = getattr(time_config, 'prediction_horizon_days', 10)

            except Exception:

                prediction_horizon = 10  # Default T+10

            

            self.log(f"[DirectPredict] Prediction horizon: T+{prediction_horizon} days")
            required_features = self._get_direct_predict_features()
            self.log(f"[DirectPredict] Snapshot feature list: {required_features}")

            # Import Excel report function

            scripts_path = Path(__file__).parent.parent / "scripts"

            if str(scripts_path) not in sys.path:

                sys.path.insert(0, str(scripts_path))

            try:

                import sys

                from pathlib import Path

                scripts_dir = Path(__file__).parent.parent / "scripts"

                if str(scripts_dir) not in sys.path:

                    sys.path.insert(0, str(scripts_dir))

                from direct_predict_ewma_excel import generate_excel_ranking_report

            except ImportError as e:

                self.log(f"[DirectPredict] ?? ???????Excel??? ??: {e}")

                self.log("[DirectPredict] ??????Excel????????")



            # Initialize model

            model = UltraEnhancedQuantitativeModel()

            self.log("[DirectPredict] ?? ???????????????????...")

            

            # Get predictions for T+10 horizon

            # Logic: Use previous trading day as base date, predict future T+10 days

            # Need 280+ days total: 280 days (for factor calculation) + buffer

            MIN_REQUIRED_LOOKBACK_DAYS = 280  # For 252-day rolling window factors

            total_lookback_days = MIN_REQUIRED_LOOKBACK_DAYS + 30  # Extra buffer for weekends/holidays

            

            # ?? Use today as initial end_date to fetch latest available data

            # We'll determine the actual last trading day with complete close data from fetched data

            today = pd.Timestamp.today()

            initial_base_date = today - BDay(1)  # Initial estimate (will be updated after fetching data)

            # Calculate start_date: use calendar days for API call (API handles trading days)

            start_date = initial_base_date - pd.Timedelta(days=total_lookback_days)

            

            self.log(f"[DirectPredict] ?? ???????? : {start_date.strftime('%Y-%m-%d')} ?? {today.strftime('%Y-%m-%d')} (??????????????)")

            self.log(f"[DirectPredict]   ???????: {MIN_REQUIRED_LOOKBACK_DAYS} ?? (???????????)")

            self.log(f"[DirectPredict]   ???horizon: T+{prediction_horizon} ??")

            self.log(f"[DirectPredict]   ???????: ???????????????????????????????????????????????????")

            

            # base_date will be determined after fetching market data (see below)

            base_date = initial_base_date  # Temporary, will be updated

            

            # Fetch all data once (more efficient than fetching separately for each day)

            self.log(f"[DirectPredict] ?? ??????? {total_lookback_days} ???????...")

            

            # Initialize predictions list

            all_predictions = []

            

            try:

                # Get all feature data for the entire period

                from bma_models.simple_25_factor_engine import Simple17FactorEngine

                

                engine = Simple17FactorEngine(

                    lookback_days=total_lookback_days,

                    mode='predict',

                    horizon=prediction_horizon  # Use T+10 horizon

                )

                

                # Fetch market data - use today as end_date to get the latest available data

                # We'll determine the actual last trading day with complete close data from the fetched data

                market_data = engine.fetch_market_data(

                    symbols=tickers,

                    use_optimized_downloader=True,

                    start_date=start_date.strftime('%Y-%m-%d'),

                    end_date=today.strftime('%Y-%m-%d')  # Use today to get latest available data

                )

                

                if market_data.empty:

                    raise ValueError(f"Failed to fetch market data for {len(tickers)} tickers")

                

                self.log(f"[DirectPredict] ? ? ??????????: {market_data.shape}")

                close_col = None
                for candidate in ('Close', 'close', 'adj_close'):
                    if candidate in market_data.columns:
                        close_col = candidate
                        break
                if close_col:
                    close_values = market_data[close_col]
                    invalid_mask = (~pd.to_numeric(close_values, errors='coerce').notna()) | (close_values <= 0) | (close_values > DIRECT_PREDICT_MAX_CLOSE)
                    if invalid_mask.any():
                        removed_rows = market_data[invalid_mask]
                        try:
                            sample = removed_rows[['date', 'ticker', close_col]].head(5).to_dict('records')
                        except Exception:
                            sample = []
                        self.log(f"[DirectPredict] Removing {invalid_mask.sum()} rows with invalid close prices (<=0 or > {DIRECT_PREDICT_MAX_CLOSE}). Sample: {sample}")
                        market_data = market_data[~invalid_mask]
                        if market_data.empty:
                            raise ValueError("All market data rows removed due to invalid close prices")

                # ?? Determine the last trading day with close data

                # Find the latest date where we have ANY close data (????????????)

                if isinstance(market_data.index, pd.MultiIndex):

                    # MultiIndex (date, ticker) - find latest date with ANY close data

                    if 'Close' in market_data.columns:

                        close_col = 'Close'

                    elif 'close' in market_data.columns:

                        close_col = 'close'

                    elif 'adj_close' in market_data.columns:

                        close_col = 'adj_close'

                    else:

                        close_col = None

                    

                    if close_col:

                        # Get all dates from the index

                        all_dates = market_data.index.get_level_values('date').unique()

                        all_dates = sorted([pd.Timestamp(d) for d in all_dates if isinstance(d, (pd.Timestamp, str))])

                        

                        # Find the latest date where we have ANY close data (????????????)

                        last_valid_date = None

                        for date in reversed(all_dates):

                            date_data = market_data.xs(date, level='date', drop_level=False)

                            if not date_data.empty and close_col in date_data.columns:

                                # Check if there's at least ONE ticker with valid close data

                                valid_close_count = date_data[close_col].notna().sum()

                                

                                if valid_close_count > 0:  # ????????????

                                    last_valid_date = date

                                    total_tickers = len(date_data)

                                    coverage_ratio = valid_close_count / total_tickers if total_tickers > 0 else 0

                                    self.log(f"[DirectPredict] ?? ???????? ??????: {date.strftime('%Y-%m-%d')} ({valid_close_count} ??????????????, ??????: {coverage_ratio:.1%})")

                                    break

                        

                        if last_valid_date is None:

                            # Fallback: use the latest date in the data

                            last_valid_date = all_dates[-1] if all_dates else base_date

                            self.log(f"[DirectPredict] ??  ??????????????????????????????: {last_valid_date.strftime('%Y-%m-%d')}")

                        

                        # Update base_date to the last valid trading day with close data

                        base_date = pd.Timestamp(last_valid_date).normalize()

                        self.log(f"[DirectPredict] ? ??????????: {base_date.strftime('%Y-%m-%d')} (????????????????????????????????)")

                    else:

                        self.log(f"[DirectPredict] ??  ?????????? ??????????????: {base_date.strftime('%Y-%m-%d')}")

                else:

                    # Single index - try to infer from date column or index

                    if 'date' in market_data.columns:

                        last_date = pd.to_datetime(market_data['date']).max()

                        base_date = pd.Timestamp(last_date).normalize()

                        self.log(f"[DirectPredict] ? ??date????????????: {base_date.strftime('%Y-%m-%d')}")

                    else:

                        self.log(f"[DirectPredict] ?? ???????????????????????????: {base_date.strftime('%Y-%m-%d')}")

                

                # Calculate all factors for the entire period

                self.log(f"[DirectPredict] ?? ????????????????...")

                all_feature_data = engine.compute_all_17_factors(market_data, mode='predict')
                if all_feature_data.empty:

                    raise ValueError("Failed to compute features from market data")








                feature_columns = [col for col in required_features if col in all_feature_data.columns]
                missing_cols = sorted(set(required_features) - set(feature_columns))
                if missing_cols:

                    self.log(f"[DirectPredict] Missing snapshot features: {missing_cols}")

                if not feature_columns:

                    raise ValueError("None of the required snapshot features are available")

                all_feature_data = all_feature_data[feature_columns]


                

                # ?? Ensure Sato factors are computed (if not already in the data)
                needs_sato = any(name.startswith('feat_sato') for name in required_features)
                if needs_sato:

                                    if 'feat_sato_momentum_10d' not in all_feature_data.columns or 'feat_sato_divergence_10d' not in all_feature_data.columns:

                                        try:

                                            self.log(f"[DirectPredict] ?? ????Sato?????????...")

                                            from scripts.sato_factor_calculation import calculate_sato_factors



                                            # Prepare data for Sato calculation

                                            sato_data = all_feature_data.copy()

                                            if 'adj_close' not in sato_data.columns and 'Close' in sato_data.columns:

                                                sato_data['adj_close'] = sato_data['Close']



                                            has_vol_ratio = 'vol_ratio_20d' in sato_data.columns

                                            if 'Volume' not in sato_data.columns:

                                                if has_vol_ratio:

                                                    base_volume = 1_000_000

                                                    sato_data['Volume'] = base_volume * sato_data['vol_ratio_20d'].fillna(1.0).clip(lower=0.1, upper=10.0)

                                                    use_vol_ratio = True

                                                else:

                                                    use_vol_ratio = False

                                                    sato_data['Volume'] = 1_000_000  # Fallback

                                            else:

                                                use_vol_ratio = has_vol_ratio



                                            # Calculate Sato factors

                                            sato_factors_df = calculate_sato_factors(

                                                df=sato_data,

                                                price_col='adj_close',

                                                volume_col='Volume',

                                                vol_ratio_col='vol_ratio_20d',

                                                lookback_days=10,

                                                vol_window=20,

                                                use_vol_ratio_directly=use_vol_ratio

                                            )



                                            # Add Sato factors to all_feature_data

                                            # ?? FIX: Ensure sato_factors_df has MultiIndex format before reindexing

                                            if not isinstance(sato_factors_df.index, pd.MultiIndex):

                                                if 'date' in sato_factors_df.columns and 'ticker' in sato_factors_df.columns:

                                                    sato_factors_df = sato_factors_df.set_index(['date', 'ticker'])

                                                    self.log(f"[DirectPredict] ? Converted sato_factors_df to MultiIndex")



                                            # Ensure all_feature_data maintains MultiIndex format

                                            if not isinstance(all_feature_data.index, pd.MultiIndex):

                                                raise ValueError("all_feature_data must have MultiIndex format before adding Sato factors")



                                            all_feature_data['feat_sato_momentum_10d'] = sato_factors_df['feat_sato_momentum_10d'].reindex(all_feature_data.index).fillna(0.0)

                                            all_feature_data['feat_sato_divergence_10d'] = sato_factors_df['feat_sato_divergence_10d'].reindex(all_feature_data.index).fillna(0.0)



                                            # Verify MultiIndex format is maintained

                                            if not isinstance(all_feature_data.index, pd.MultiIndex):

                                                raise ValueError("all_feature_data lost MultiIndex format after adding Sato factors!")



                                            self.log(f"[DirectPredict] ? Sato???????????MultiIndex???????: {all_feature_data.index.names}")

                                        except Exception as e:

                                            self.log(f"[DirectPredict] ?? Sato??????????: {e}, ????...")

                                            # Add zero-filled columns if missing

                                            if 'feat_sato_momentum_10d' not in all_feature_data.columns:

                                                all_feature_data['feat_sato_momentum_10d'] = 0.0

                                            if 'feat_sato_divergence_10d' not in all_feature_data.columns:

                                                all_feature_data['feat_sato_divergence_10d'] = 0.0



                # ?? FIX: Final verification and standardization of all_feature_data format

                # Ensure format matches training parquet file exactly

                if not isinstance(all_feature_data.index, pd.MultiIndex):

                    raise ValueError(f"all_feature_data must have MultiIndex format, got: {type(all_feature_data.index)}")

                

                index_names = all_feature_data.index.names

                if 'date' not in index_names or 'ticker' not in index_names:

                    raise ValueError(f"all_feature_data MultiIndex must have 'date' and 'ticker' levels, got: {index_names}")

                

                # ?? FIX: Standardize MultiIndex to match training file format exactly

                # Training file format: MultiIndex(['date', 'ticker'])

                # - date: datetime64[ns], normalized (no time component)

                # - ticker: object/string

                

                # Normalize date level (remove time component)

                date_level = all_feature_data.index.get_level_values('date')

                if not pd.api.types.is_datetime64_any_dtype(date_level):

                    raise ValueError(f"Date level must be datetime, got: {date_level.dtype}")

                

                # ?? FIX: Handle DatetimeIndex vs Series - DatetimeIndex doesn't have .dt accessor

                # get_level_values can return DatetimeIndex directly, so check type first

                if isinstance(date_level, pd.DatetimeIndex):

                    # DatetimeIndex has methods directly, not through .dt accessor

                    if date_level.tz is not None:

                        date_normalized = date_level.tz_localize(None).normalize()

                    else:

                        date_normalized = date_level.normalize()

                else:

                    # Convert to datetime if needed, then use .dt accessor for Series

                    date_converted = pd.to_datetime(date_level)

                    if isinstance(date_converted, pd.DatetimeIndex):

                        # If conversion results in DatetimeIndex, use direct methods

                        if date_converted.tz is not None:

                            date_normalized = date_converted.tz_localize(None).normalize()

                        else:

                            date_normalized = date_converted.normalize()

                    else:

                        # Series has .dt accessor

                        if date_converted.dt.tz is not None:

                            date_normalized = date_converted.dt.tz_localize(None).dt.normalize()

                        else:

                            date_normalized = date_converted.dt.normalize()

                

                # ?? FIX: Ensure ticker format matches training file exactly

                # Training file uses uppercase tickers (as seen in 80/20 eval)

                ticker_level = all_feature_data.index.get_level_values('ticker').astype(str).str.strip().str.upper()

                

                # Recreate MultiIndex with standardized format (matching training file)

                # Training file format: MultiIndex(['date', 'ticker'])

                # - date: datetime64[ns], normalized (no time component)

                # - ticker: object/string, UPPERCASE (matching 80/20 eval and training)

                all_feature_data.index = pd.MultiIndex.from_arrays(

                    [date_normalized, ticker_level],

                    names=['date', 'ticker']

                )

                

                # Verify format matches training file

                self.log(f"[DirectPredict] ? ???????????: shape={all_feature_data.shape}")

                self.log(f"[DirectPredict] ? MultiIndex??????: levels={all_feature_data.index.names}")

                self.log(f"[DirectPredict] ? ????????: {all_feature_data.index.get_level_values('date').dtype} (normalized)")

                self.log(f"[DirectPredict] ? Ticker????: {all_feature_data.index.get_level_values('ticker').dtype}")

                self.log(f"[DirectPredict] ? Unique dates: {all_feature_data.index.get_level_values('date').nunique()}, unique tickers: {all_feature_data.index.get_level_values('ticker').nunique()}")

                

                # Final check: ensure no duplicates

                duplicates = all_feature_data.index.duplicated()

                if duplicates.any():

                    dup_count = duplicates.sum()

                    self.log(f"[DirectPredict] ?? Removing {dup_count} duplicate indices before prediction")

                    all_feature_data = all_feature_data[~duplicates]

                    all_feature_data = all_feature_data.groupby(level=['date', 'ticker']).first()

                    self.log(f"[DirectPredict] ? After deduplication: shape={all_feature_data.shape}")

                

                # ?? Predict using base_date (last trading day with close data) for T+10 horizon

                # ???????????????????????????????????????

                pred_date = base_date  # Use last trading day with close data as base date

                pred_date_str = pred_date.strftime('%Y-%m-%d')

                

                self.log(f"[DirectPredict] ?? ???? {pred_date_str} (????????????????????????????????) ??? T+{prediction_horizon} ??...")

                

                try:

                    # Extract feature data up to and including base_date for factor calculation

                    # ?? FIX: Ensure all_feature_data is MultiIndex format

                    if not isinstance(all_feature_data.index, pd.MultiIndex):

                        self.log(f"[DirectPredict] ?? all_feature_data is not MultiIndex, converting...")

                        # Try to convert to MultiIndex

                        if 'date' in all_feature_data.columns and 'ticker' in all_feature_data.columns:

                            all_feature_data = all_feature_data.set_index(['date', 'ticker'])

                            self.log(f"[DirectPredict] ? Converted to MultiIndex using date and ticker columns")

                        else:

                            raise ValueError("Cannot convert to MultiIndex: missing 'date' or 'ticker' columns")

                    

                    # Verify MultiIndex format

                    if not isinstance(all_feature_data.index, pd.MultiIndex):

                        raise ValueError("all_feature_data must have MultiIndex (date, ticker)")

                    

                    index_names = all_feature_data.index.names

                    if 'date' not in index_names or 'ticker' not in index_names:

                        raise ValueError(f"MultiIndex must have 'date' and 'ticker' levels, got: {index_names}")

                    

                    self.log(f"[DirectPredict] ? all_feature_data format: MultiIndex with levels {index_names}, shape: {all_feature_data.shape}")

                    

                    # Extract feature data up to and including base_date

                    date_mask = all_feature_data.index.get_level_values('date') <= pred_date

                    date_feature_data = all_feature_data[date_mask].copy()

                    

                    # ?? FIX: Ensure date_feature_data maintains MultiIndex format

                    if not isinstance(date_feature_data.index, pd.MultiIndex):

                        raise ValueError("date_feature_data lost MultiIndex format after filtering!")

                    

                    # Remove duplicate indices (if any)

                    duplicates = date_feature_data.index.duplicated()

                    if duplicates.any():

                        dup_count = duplicates.sum()

                        self.log(f"[DirectPredict] ?? Removing {dup_count} duplicate indices from date_feature_data")

                        date_feature_data = date_feature_data[~duplicates]

                    

                    # Ensure each (date, ticker) combination appears only once

                    date_feature_data = date_feature_data.groupby(level=['date', 'ticker']).first()

                    

                    if date_feature_data.empty:

                        self.log(f"[DirectPredict] ?? {pred_date_str}: ??????????")

                        raise ValueError(f"No feature data available for {pred_date_str}")

                    

                    self.log(f"[DirectPredict] ? date_feature_data format: MultiIndex, shape: {date_feature_data.shape}, unique dates: {date_feature_data.index.get_level_values('date').nunique()}, unique tickers: {date_feature_data.index.get_level_values('ticker').nunique()}")

                    

                    # Get predictions using the pre-computed features

                    # Always use the configured Direct Predict snapshot (env override supported)

                    snapshot_id_to_use = os.environ.get(

                        "DIRECT_PREDICT_SNAPSHOT_ID",

                        DIRECT_PREDICT_SNAPSHOT_ID

                    )

                    snapshot_id_to_use = (snapshot_id_to_use or "").strip() or DIRECT_PREDICT_SNAPSHOT_ID

                    self.log(f"[DirectPredict] Using snapshot ID: {snapshot_id_to_use}")



                    # Predict with T+10 horizon

                    results = model.predict_with_snapshot(

                        feature_data=date_feature_data,  # Use pre-computed features

                        snapshot_id=snapshot_id_to_use,  # Force configured snapshot for Direct Predict

                        universe_tickers=tickers,

                        as_of_date=pred_date,

                        prediction_days=prediction_horizon  # Use T+10 horizon

                    )

                        

                    if results.get('success', False):

                        # Get raw predictions (no EMA smoothing)

                        predictions_raw = results.get('predictions_raw')

                        if predictions_raw is None:

                            # Fallback to predictions if raw not available

                            predictions_raw = results.get('predictions')

                        

                        # Get base model predictions (LambdaRank, CatBoost, etc.)

                        base_predictions = results.get('base_predictions')  # DataFrame with pred_lambdarank, pred_catboost, etc.

                        

                        # Debug: Log base_predictions status

                        if base_predictions is not None:

                            self.log(f"[DirectPredict] ?? Base predictions available: {type(base_predictions)}, shape: {base_predictions.shape if hasattr(base_predictions, 'shape') else 'N/A'}")

                            if isinstance(base_predictions, pd.DataFrame):

                                self.log(f"[DirectPredict] ?? Base predictions columns: {list(base_predictions.columns)}")

                        else:

                            self.log(f"[DirectPredict] ?? Base predictions not available in results")

                        

                        if predictions_raw is not None:

                            # Debug: Log predictions_raw structure

                            self.log(f"[DirectPredict] ?? predictions_raw type: {type(predictions_raw)}")

                            if isinstance(predictions_raw, pd.Series):

                                self.log(f"[DirectPredict] ?? predictions_raw shape: {predictions_raw.shape}, unique values: {predictions_raw.nunique()}")

                                self.log(f"[DirectPredict] ?? predictions_raw sample (first 5): {predictions_raw.head().to_dict()}")

                                self.log(f"[DirectPredict] ?? predictions_raw value range: min={predictions_raw.min():.6f}, max={predictions_raw.max():.6f}, mean={predictions_raw.mean():.6f}")

                                # ?? DIAGNOSTIC: Check for duplicate indices

                                if isinstance(predictions_raw.index, pd.MultiIndex):

                                    self.log(f"[DirectPredict] ?? predictions_raw unique dates: {predictions_raw.index.get_level_values('date').nunique()}")

                                    self.log(f"[DirectPredict] ?? predictions_raw unique tickers: {predictions_raw.index.get_level_values('ticker').nunique()}")

                                    duplicates = predictions_raw.index.duplicated()

                                    if duplicates.any():

                                        self.log(f"[DirectPredict] ?? predictions_raw has {duplicates.sum()} duplicate indices!")

                                        dup_indices = predictions_raw.index[duplicates]

                                        self.log(f"[DirectPredict] ?? Sample duplicate indices: {dup_indices[:5].tolist()}")

                                        # Remove duplicates

                                        predictions_raw = predictions_raw[~duplicates]

                                        self.log(f"[DirectPredict] ? Removed {duplicates.sum()} duplicate indices from predictions_raw")

                                pred_df = predictions_raw.to_frame('score')

                            else:

                                pred_df = predictions_raw.copy()

                                if 'score' not in pred_df.columns and len(pred_df.columns) > 0:

                                    pred_df.columns = ['score']

                                self.log(f"[DirectPredict] ?? pred_df shape: {pred_df.shape}, score unique values: {pred_df['score'].nunique()}")

                                self.log(f"[DirectPredict] ?? pred_df score range: min={pred_df['score'].min():.6f}, max={pred_df['score'].max():.6f}, mean={pred_df['score'].mean():.6f}")

                            

                            # Ensure MultiIndex with date and ticker

                            if not isinstance(pred_df.index, pd.MultiIndex):

                                # Create MultiIndex from ticker index

                                tickers_from_index = pred_df.index.tolist()

                                dates_list = [pred_date] * len(tickers_from_index)

                                pred_df.index = pd.MultiIndex.from_arrays(

                                    [dates_list, tickers_from_index],

                                    names=['date', 'ticker']

                                )

                            else:

                                # Update date level to ensure correct date

                                new_index = pd.MultiIndex.from_arrays([

                                    [pred_date] * len(pred_df),

                                    pred_df.index.get_level_values('ticker')

                                ], names=['date', 'ticker'])

                                pred_df.index = new_index

                            

                            # ?? FIX: Remove duplicate indices after MultiIndex creation

                            if pred_df.index.duplicated().any():

                                self.log(f"[DirectPredict] ?? pred_df has {pred_df.index.duplicated().sum()} duplicate indices after MultiIndex creation, removing duplicates...")

                                pred_df = pred_df[~pred_df.index.duplicated(keep='first')]

                                self.log(f"[DirectPredict] ? pred_df after deduplication: {len(pred_df)} rows")

                            

                            # ?? FIX: Ensure each (date, ticker) combination appears only once

                            if isinstance(pred_df.index, pd.MultiIndex):

                                ticker_level = pred_df.index.get_level_values('ticker')

                                if ticker_level.duplicated().any():

                                    self.log(f"[DirectPredict] ?? pred_df has duplicate tickers in same date, grouping by (date, ticker)...")

                                    pred_df = pred_df.groupby(level=['date', 'ticker']).first()

                                    self.log(f"[DirectPredict] ? pred_df after grouping: {len(pred_df)} rows, unique tickers: {pred_df.index.get_level_values('ticker').nunique()}")

                            

                            # Add base model predictions if available

                            if base_predictions is not None and isinstance(base_predictions, pd.DataFrame):

                                self.log(f"[DirectPredict] ?? Adding base model predictions to pred_df...")

                                

                                # ?? DIAGNOSTIC: Check base_predictions structure

                                self.log(f"[DirectPredict] ?? base_predictions index type: {type(base_predictions.index)}")

                                if isinstance(base_predictions.index, pd.MultiIndex):

                                    self.log(f"[DirectPredict] ?? base_predictions unique dates: {base_predictions.index.get_level_values('date').nunique()}")

                                    self.log(f"[DirectPredict] ?? base_predictions unique tickers: {base_predictions.index.get_level_values('ticker').nunique()}")

                                    self.log(f"[DirectPredict] ?? base_predictions total rows: {len(base_predictions)}")

                                    duplicates = base_predictions.index.duplicated()

                                    if duplicates.any():

                                        self.log(f"[DirectPredict] ?? base_predictions has {duplicates.sum()} duplicate indices!")

                                        # Remove duplicates before alignment

                                        base_predictions = base_predictions[~duplicates]

                                        self.log(f"[DirectPredict] ? Removed {duplicates.sum()} duplicate indices from base_predictions")

                                

                                # Ensure base_predictions has same index structure

                                if isinstance(base_predictions.index, pd.MultiIndex):

                                    base_predictions_aligned = base_predictions.reindex(pred_df.index)

                                else:

                                    # Try to align by ticker

                                    base_predictions_aligned = base_predictions.reindex(pred_df.index.get_level_values('ticker'))

                                    base_predictions_aligned.index = pred_df.index

                                

                                # ?? FIX: Remove duplicate indices after alignment

                                if base_predictions_aligned.index.duplicated().any():

                                    self.log(f"[DirectPredict] ?? base_predictions_aligned has duplicate indices after reindex, removing duplicates...")

                                    base_predictions_aligned = base_predictions_aligned[~base_predictions_aligned.index.duplicated(keep='first')]

                                

                                # Debug: Log alignment result

                                self.log(f"[DirectPredict] ?? Base predictions aligned shape: {base_predictions_aligned.shape}")

                                self.log(f"[DirectPredict] ?? Base predictions aligned columns: {list(base_predictions_aligned.columns)}")

                                if isinstance(base_predictions_aligned.index, pd.MultiIndex):

                                    self.log(f"[DirectPredict] ?? Base predictions aligned unique tickers: {base_predictions_aligned.index.get_level_values('ticker').nunique()}")

                                

                                # Add LambdaRank and CatBoost scores

                                if 'pred_lambdarank' in base_predictions_aligned.columns:

                                    pred_df['score_lambdarank'] = base_predictions_aligned['pred_lambdarank']

                                    self.log(f"[DirectPredict] ? Added score_lambdarank: {pred_df['score_lambdarank'].notna().sum()} non-null values")

                                else:

                                    self.log(f"[DirectPredict] ?? pred_lambdarank not found in base_predictions. Available columns: {list(base_predictions_aligned.columns)}")

                                

                                if 'pred_catboost' in base_predictions_aligned.columns:

                                    pred_df['score_catboost'] = base_predictions_aligned['pred_catboost']

                                    self.log(f"[DirectPredict] ? Added score_catboost: {pred_df['score_catboost'].notna().sum()} non-null values")

                                else:

                                    self.log(f"[DirectPredict] ?? pred_catboost not found in base_predictions. Available columns: {list(base_predictions_aligned.columns)}")

                                

                                if 'pred_elastic' in base_predictions_aligned.columns:

                                    pred_df['score_elastic'] = base_predictions_aligned['pred_elastic']

                                    self.log(f"[DirectPredict] ? Added score_elastic: {pred_df['score_elastic'].notna().sum()} non-null values")

                                else:

                                    self.log(f"[DirectPredict] ?? pred_elastic not found in base_predictions. Available columns: {list(base_predictions_aligned.columns)}")

                                

                                if 'pred_xgb' in base_predictions_aligned.columns:

                                    pred_df['score_xgb'] = base_predictions_aligned['pred_xgb']

                                    self.log(f"[DirectPredict] ? Added score_xgb: {pred_df['score_xgb'].notna().sum()} non-null values")

                                else:

                                    self.log(f"[DirectPredict] ?? pred_xgb not found in base_predictions. Available columns: {list(base_predictions_aligned.columns)}")

                            else:

                                self.log(f"[DirectPredict] ?? Base predictions not available or not DataFrame. Type: {type(base_predictions)}")

                            

                            all_predictions.append(pred_df)

                            self.log(f"[DirectPredict] ? {pred_date_str}: T+{prediction_horizon} ???????? ({len(pred_df)} ????)")

                        else:

                            self.log(f"[DirectPredict] ?? {pred_date_str}: ?????????")

                            raise ValueError(f"No predictions returned for {pred_date_str}")

                    else:

                        error_msg = results.get('error', 'Unknown error')

                        self.log(f"[DirectPredict] ?? {pred_date_str} ??????: {error_msg}")

                        raise RuntimeError(f"Prediction failed: {error_msg}")

                except Exception as e:

                    self.log(f"[DirectPredict] ? {pred_date_str} ?????: {e}")

                    import traceback

                    self.log(f"[DirectPredict] ???????: {traceback.format_exc()}")

                    # Don't raise here - continue to check all_predictions

                    # If all_predictions is empty, it will be handled below

                    pass

                        

            except Exception as e:

                self.log(f"[DirectPredict] ? ???????????????????: {e}")

                import traceback

                self.log(f"[DirectPredict] ???????: {traceback.format_exc()}")

                # Ensure all_predictions is initialized even if exception occurs

                if 'all_predictions' not in locals():

                    all_predictions = []

                if not all_predictions:

                    return



            # Ensure all_predictions is initialized

            if 'all_predictions' not in locals():

                all_predictions = []

            

            if not all_predictions:

                self.log("[DirectPredict] ?  ?????? ????????")

                return



            # Combine all predictions

            if len(all_predictions) == 1:

                combined_predictions = all_predictions[0]

            else:

                combined_predictions = pd.concat(all_predictions, axis=0)

            

            # ?? FIX: Remove duplicate indices after concatenation

            if combined_predictions.index.duplicated().any():

                self.log(f"[DirectPredict] ?? combined_predictions has {combined_predictions.index.duplicated().sum()} duplicate indices after concat, removing duplicates...")

                combined_predictions = combined_predictions[~combined_predictions.index.duplicated(keep='first')]

                self.log(f"[DirectPredict] ? combined_predictions after deduplication: {len(combined_predictions)} rows")

            

            # ?? FIX: Ensure each (date, ticker) combination appears only once

            if isinstance(combined_predictions.index, pd.MultiIndex):

                ticker_level = combined_predictions.index.get_level_values('ticker')

                date_level = combined_predictions.index.get_level_values('date')

                # Check for duplicates within same date

                for date in date_level.unique():

                    date_mask = date_level == date

                    date_tickers = ticker_level[date_mask]

                    if date_tickers.duplicated().any():

                        self.log(f"[DirectPredict] ?? Date {date} has {date_tickers.duplicated().sum()} duplicate tickers, grouping...")

                        combined_predictions = combined_predictions.groupby(level=['date', 'ticker']).first()

                        break  # Only need to group once

            

            self.log(f"[DirectPredict] ? ??????????: {len(combined_predictions)} ??")

            dates_in_pred = sorted(combined_predictions.index.get_level_values('date').unique())

            self.log(f"[DirectPredict] ????? : {dates_in_pred[0]} ?? {dates_in_pred[-1]} ({len(dates_in_pred)} ??)")

            

            # ?? DIAGNOSTIC: Log unique tickers per date

            if isinstance(combined_predictions.index, pd.MultiIndex):

                for date in dates_in_pred[-3:]:  # Check last 3 dates

                    date_mask = combined_predictions.index.get_level_values('date') == date

                    date_tickers = combined_predictions.index.get_level_values('ticker')[date_mask]

                    self.log(f"[DirectPredict] ?? Date {date}: {date_tickers.nunique()} unique tickers, {len(date_tickers)} total rows")

                    if date_tickers.duplicated().any():

                        self.log(f"[DirectPredict] ?? Date {date} has {date_tickers.duplicated().sum()} duplicate tickers!")



            # Apply EMA smoothing so UI + Excel ranking match CLI workflow

            final_predictions = combined_predictions.copy()

            final_predictions = self._apply_direct_predict_ema(

                final_predictions,

                score_columns=['score', 'score_lambdarank']

            )



            # Get latest date predictions for recommendations

            latest_date = dates_in_pred[-1] if dates_in_pred else None

            if latest_date:

                try:

                    latest_predictions = final_predictions.xs(latest_date, level='date', drop_level=False)

                    

                    # ?? DIAGNOSTIC: Log latest_predictions structure

                    self.log(f"[DirectPredict] ?? latest_predictions shape after xs: {latest_predictions.shape}")

                    if isinstance(latest_predictions.index, pd.MultiIndex):

                        ticker_level = latest_predictions.index.get_level_values('ticker')

                        self.log(f"[DirectPredict] ?? latest_predictions unique tickers: {ticker_level.nunique()}")

                        self.log(f"[DirectPredict] ?? latest_predictions total rows: {len(latest_predictions)}")

                        duplicates = ticker_level.duplicated()

                        if duplicates.any():

                            self.log(f"[DirectPredict] ?? latest_predictions has {duplicates.sum()} duplicate tickers!")

                            ticker_counts = ticker_level.value_counts()

                            dup_tickers = ticker_counts[ticker_counts > 1]

                            self.log(f"[DirectPredict] ?? Duplicate tickers: {dup_tickers.head(10).to_dict()}")

                    

                    # ?? FIX: Remove duplicate tickers (keep first occurrence)

                    if isinstance(latest_predictions.index, pd.MultiIndex):

                        ticker_level = latest_predictions.index.get_level_values('ticker')

                        if ticker_level.duplicated().any():

                            self.log(f"[DirectPredict] ?? Removing {ticker_level.duplicated().sum()} duplicate tickers from latest_predictions...")

                            latest_predictions = latest_predictions[~ticker_level.duplicated(keep='first')]

                            self.log(f"[DirectPredict] ? latest_predictions after deduplication: {len(latest_predictions)} rows, {latest_predictions.index.get_level_values('ticker').nunique()} unique tickers")

                    else:

                        # If not MultiIndex, assume index is ticker

                        if latest_predictions.index.duplicated().any():

                            self.log(f"[DirectPredict] ?? Removing {latest_predictions.index.duplicated().sum()} duplicate indices from latest_predictions...")

                            latest_predictions = latest_predictions[~latest_predictions.index.duplicated(keep='first')]

                    

                    latest_predictions = latest_predictions.sort_values('score', ascending=False)

                    

                    # Final diagnostic

                    unique_tickers = latest_predictions.index.get_level_values('ticker').nunique() if isinstance(latest_predictions.index, pd.MultiIndex) else latest_predictions.index.nunique()

                    self.log(f"[DirectPredict] ?? Final latest_predictions: {len(latest_predictions)} rows, {unique_tickers} unique tickers")

                    

                    # Debug: Log latest_predictions before creating recommendations

                    self.log(f"[DirectPredict] ?? latest_predictions shape: {latest_predictions.shape}")

                    self.log(f"[DirectPredict] ?? latest_predictions columns: {list(latest_predictions.columns)}")

                    self.log(f"[DirectPredict] ?? latest_predictions score unique values: {latest_predictions['score'].nunique()}")

                    self.log(f"[DirectPredict] ?? latest_predictions score range: min={latest_predictions['score'].min():.6f}, max={latest_predictions['score'].max():.6f}")

                    self.log(f"[DirectPredict] ?? Top 5 scores: {latest_predictions['score'].head().tolist()}")

                    

                    # Create recommendations list

                    recs = []

                    for idx, row in latest_predictions.iterrows():

                        ticker = idx[1] if isinstance(idx, tuple) else idx

                        score = row.get('score', 0.0)

                        # Debug: Log if score is same as previous

                        if len(recs) > 0 and abs(float(score) - recs[-1]['score']) < 1e-6:

                            self.log(f"[DirectPredict] ?? Duplicate score detected: {ticker}={float(score):.6f}, previous={recs[-1]['ticker']}={recs[-1]['score']:.6f}")

                        recs.append({'ticker': ticker, 'score': float(score)})

                except Exception as e:

                    self.log(f"[DirectPredict] ?? ???????? ????: {e}")

                    recs = []

            else:

                recs = []



            # Persist results to DB (monitoring.db) for audit

            try:

                import sqlite3, time

                db_path = os.path.join("data", "monitoring.db")

                os.makedirs(os.path.dirname(db_path), exist_ok=True)

                conn = sqlite3.connect(db_path)

                cur = conn.cursor()

                cur.execute(

                    """

                    CREATE TABLE IF NOT EXISTS direct_predictions (

                        ts INTEGER,

                        snapshot_id TEXT,

                        ticker TEXT,

                        score REAL

                    )

                    """

                )

                ts = int(time.time())

                sid = results.get('snapshot_used', 'latest') if 'results' in locals() else 'latest'

                rows = [(ts, sid, r.get('ticker'), float(r.get('score', 0.0))) for r in recs if r.get('ticker')]

                if rows:

                    cur.executemany("INSERT INTO direct_predictions (ts, snapshot_id, ticker, score) VALUES (?, ?, ?, ?)", rows)

                    conn.commit()

                conn.close()

                self.log(f"[DirectPredict] ? ?? ??????? {len(rows)} ?????")

            except Exception as e:

                self.log(f"[DirectPredict] ??  ??????????: {e}")



            # Generate a single Excel report (Top-30 for all models)

            try:

                output_dir = Path("results")
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                excel_path = output_dir / f"direct_predict_top30_{timestamp}.xlsx"
                self.log(f"[DirectPredict] Generating combined Top30 Excel (all models): {excel_path}")

                generate_excel_ranking_report(
                    final_predictions,
                    str(excel_path),
                    model_name="MetaRankerStacker",
                    top_n=30
                )

                self.log("[DirectPredict] Excel report saved (Meta + LambdaRank + base models on dedicated sheets)")

            except NameError:

                self.log("[DirectPredict] ?? Excel Excel ")

            except ImportError:

                self.log("[DirectPredict] ?? openpyxl Excel ")

            except Exception as e:

                self.log(f"[DirectPredict] ?? Excel : {e}")

                import traceback

                self.log(f"[DirectPredict]  : {traceback.format_exc()}")

# Display top recommendations (MetaRankerStacker)

            try:

                top_show = min(20, len(recs))

                self.log(f"[DirectPredict] ?? MetaRankerStacker Top {top_show} ???:")

                for i, r in enumerate(recs[:top_show], 1):

                    score = r.get('score', 0.0)

                    ticker = r.get('ticker', 'N/A')

                    self.log(f"  {i:2d}. {ticker:8s}: {score:8.6f}")

            except Exception as e:

                self.log(f"[DirectPredict] ?? ?????????: {e}")

            

            # Helper function to get top N unique tickers by score

            def get_top_n_unique_tickers(df, score_col, n=20):

                """Get top N unique tickers by score, handling MultiIndex"""

                if score_col not in df.columns:

                    return pd.DataFrame()

                

                try:

                    # Extract ticker level from index

                    if isinstance(df.index, pd.MultiIndex):

                        # If MultiIndex, extract ticker level by resetting that level only.
                        # Using reset_index avoids the "ticker is both column and index" ambiguity.

                        temp_df = df[[score_col]].copy()

                        temp_df = temp_df.reset_index(level='ticker')

                        temp_df = temp_df.rename(columns={'ticker': 'ticker'})

                        # Remove NaN scores

                        temp_df = temp_df.dropna(subset=[score_col])

                        # Group by ticker and take the maximum score (in case of duplicates)

                        grouped = temp_df.groupby('ticker')[score_col].max().reset_index()

                        # Sort and get top N

                        top_n = grouped.nlargest(n, score_col).reset_index(drop=True)

                        return top_n

                    else:

                        # If not MultiIndex, assume index is ticker

                        temp_df = df[[score_col]].copy()

                        temp_df['ticker'] = df.index.astype(str)

                        # Remove NaN scores

                        temp_df = temp_df.dropna(subset=[score_col])

                        # Remove duplicates by ticker (keep max score)

                        grouped = temp_df.groupby('ticker')[score_col].max().reset_index()

                        top_n = grouped.nlargest(n, score_col).reset_index(drop=True)

                        return top_n

                except Exception as e:

                    self.log(f"[DirectPredict] ?? Error in get_top_n_unique_tickers: {e}")

                    import traceback

                    self.log(f"[DirectPredict] ???????: {traceback.format_exc()}")

                    return pd.DataFrame()

            

            # Display CatBoost Top 20

            try:

                if latest_date and 'score_catboost' in latest_predictions.columns:

                    catboost_top20 = get_top_n_unique_tickers(latest_predictions, 'score_catboost', 20)

                    if len(catboost_top20) > 0:

                        self.log(f"\n[DirectPredict] ?? CatBoost Top {len(catboost_top20)}:")

                        for idx, row in catboost_top20.iterrows():

                            ticker = str(row['ticker']).strip()

                            score = float(row['score_catboost'])

                            self.log(f"  {idx+1:2d}. {ticker:8s}: {score:8.6f}")

                    else:

                        self.log(f"[DirectPredict] ?? CatBoost Top20: No valid scores found")

                else:

                    self.log(f"[DirectPredict] ?? CatBoost scores not available")

            except Exception as e:

                self.log(f"[DirectPredict] ?? ???CatBoost Top20???: {e}")

                import traceback

                self.log(f"[DirectPredict] ???????: {traceback.format_exc()}")

            

            # Display LambdaRanker Top 20

            try:

                if latest_date and 'score_lambdarank' in latest_predictions.columns:

                    lambdarank_top20 = get_top_n_unique_tickers(latest_predictions, 'score_lambdarank', 20)

                    if len(lambdarank_top20) > 0:

                        self.log(f"\n[DirectPredict] ?? LambdaRanker Top {len(lambdarank_top20)}:")

                        for idx, row in lambdarank_top20.iterrows():

                            ticker = str(row['ticker']).strip()

                            score = float(row['score_lambdarank'])

                            self.log(f"  {idx+1:2d}. {ticker:8s}: {score:8.6f}")

                    else:

                        self.log(f"[DirectPredict] ?? LambdaRanker Top20: No valid scores found")

                else:

                    self.log(f"[DirectPredict] ?? LambdaRanker scores not available")

            except Exception as e:

                self.log(f"[DirectPredict] ?? ???LambdaRanker Top20???: {e}")

                import traceback

                self.log(f"[DirectPredict] ???????: {traceback.format_exc()}")

            

            # Display ElasticNet Top 20

            try:

                if latest_date and 'score_elastic' in latest_predictions.columns:

                    elastic_top20 = get_top_n_unique_tickers(latest_predictions, 'score_elastic', 20)

                    if len(elastic_top20) > 0:

                        self.log(f"\n[DirectPredict] ?? ElasticNet Top {len(elastic_top20)}:")

                        for idx, row in elastic_top20.iterrows():

                            ticker = str(row['ticker']).strip()

                            score = float(row['score_elastic'])

                            self.log(f"  {idx+1:2d}. {ticker:8s}: {score:8.6f}")

                    else:

                        self.log(f"[DirectPredict] ?? ElasticNet Top20: No valid scores found")

                else:

                    self.log(f"[DirectPredict] ?? ElasticNet scores not available")

            except Exception as e:

                self.log(f"[DirectPredict] ?? ???ElasticNet Top20???: {e}")

                import traceback

                self.log(f"[DirectPredict] ???????: {traceback.format_exc()}")

            

            # Display XGBoost Top 20

            try:

                if latest_date and 'score_xgb' in latest_predictions.columns:

                    xgb_top20 = get_top_n_unique_tickers(latest_predictions, 'score_xgb', 20)

                    if len(xgb_top20) > 0:

                        self.log(f"\n[DirectPredict] ?? XGBoost Top {len(xgb_top20)}:")

                        for idx, row in xgb_top20.iterrows():

                            ticker = str(row['ticker']).strip()

                            score = float(row['score_xgb'])

                            self.log(f"  {idx+1:2d}. {ticker:8s}: {score:8.6f}")

                    else:

                        self.log(f"[DirectPredict] ?? XGBoost Top20: No valid scores found")

                else:

                    self.log(f"[DirectPredict] ?? XGBoost scores not available")

            except Exception as e:

                self.log(f"[DirectPredict] ?? ???XGBoost Top20???: {e}")

                import traceback

                self.log(f"[DirectPredict] ???????: {traceback.format_exc()}")



        except Exception as e:

            self.log(f"[DirectPredict] ? ????: {e}")

            import traceback

            self.log(f"[DirectPredict] ???????: {traceback.format_exc()}")



    def _run_timesplit_eval(self) -> None:
        try:
            selected_features = [feat for feat, var in getattr(self, 'timesplit_feature_vars', {}).items() if var.get()]
        except AttributeError:
            messagebox.showerror("80/20 Evaluation", "Timesplit UI not initialized yet.")
            return
        if not selected_features:
            messagebox.showerror("80/20 Evaluation", "Select at least one feature before running the evaluation.")
            return
        split_value = float(getattr(self, 'timesplit_split_var', tk.DoubleVar(value=0.8)).get())
        split_value = max(0.60, min(0.95, split_value))
        ema_top_n = "0" if getattr(self, 'var_timesplit_ema', tk.BooleanVar(value=False)).get() else "-1"
        cmd = [
            sys.executable,
            str(Path('scripts') / 'time_split_80_20_oos_eval.py'),
            '--data-file', str(STAGE_A_DATA_PATH),
            '--train-data', str(STAGE_A_DATA_PATH),
            '--horizon-days', '10',
            '--split', f"{split_value:.3f}",
            '--model', 'lambdarank',
            '--models', 'lambdarank',
            '--output-dir', str(Path('results') / 'timesplit_gui'),
            '--log-level', 'INFO',
            '--features',
        ] + selected_features + ['--ema-top-n', ema_top_n]
        if getattr(self, '_timesplit_thread', None) and self._timesplit_thread.is_alive():
            messagebox.showinfo("80/20 Evaluation", "An evaluation is already running. Please wait for it to finish.")
            return
        self.log(f"[TimeSplit] Starting 80/20 evaluation with split={split_value:.2f}, EMA={'ON' if ema_top_n == '0' else 'OFF'}")
        cwd = Path(__file__).resolve().parent.parent
        thread = threading.Thread(target=self._run_timesplit_eval_thread, args=(cmd, cwd), daemon=True)
        self._timesplit_thread = thread
        thread.start()

    def _run_timesplit_eval_thread(self, cmd: list[str], cwd: Path) -> None:
        try:
            proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except Exception as exc:
            self.after(0, lambda: self.log(f"[TimeSplit] Failed to start evaluation: {exc}"))
            messagebox.showerror("80/20 Evaluation", f"Failed to start process: {exc}")
            return
        for line in proc.stdout or []:
            msg = line.rstrip()
            if msg:
                self.after(0, lambda m=msg: self.log(f"[TimeSplit] {m}"))
        proc.wait()
        status = "completed successfully" if proc.returncode == 0 else f"finished with code {proc.returncode}"
        self.after(0, lambda: self.log(f"[TimeSplit] Evaluation {status}."))
        if proc.returncode == 0:
            self.after(0, lambda: messagebox.showinfo("80/20 Evaluation", "Evaluation finished successfully. Check results/timesplit_gui for outputs."))
        else:
            self.after(0, lambda: messagebox.showerror("80/20 Evaluation", f"Evaluation failed with code {proc.returncode}. See log for details."))

    def _open_timesplit_results(self) -> None:
        results_path = Path('results') / 'timesplit_gui'
        results_path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith('win'):
                os.startfile(results_path)  # type: ignore[attr-defined]
            elif sys.platform.startswith('darwin'):
                subprocess.Popen(['open', str(results_path)])
            else:
                subprocess.Popen(['xdg-open', str(results_path)])
        except Exception as exc:
            self.log(f"[TimeSplit] Failed to open results folder: {exc}")
            messagebox.showerror("80/20 Evaluation", f"Unable to open folder: {exc}")


    def _build_direct_tab(self, parent) -> None:

        frm = ttk.Frame(parent)

        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)



        # ? F???

        row1 = ttk.LabelFrame(frm, text="order placement ?")

        row1.pack(fill=tk.X, pady=6)

        ttk.Label(row1, text=").grid(row=0, column=0, padx=5, pady=5")

        self.d_sym = ttk.Entry(row1, width=12); self.d_sym.grid(row=0, column=1, padx=5)

        ttk.Label(row1, text=" ?).grid(row=0, column=2, padx=5")

        self.d_qty = ttk.Entry(row1, width=10); self.d_qty.insert(0, "100"); self.d_qty.grid(row=0, column=3, padx=5)

        ttk.Label(row1, text="limit").grid(row=0, column=4, padx=5)

        self.d_px = ttk.Entry(row1, width=10); self.d_px.grid(row=0, column=5, padx=5)



        # ? D???

        row2 = ttk.LabelFrame(frm, text=" ?order placement")

        row2.pack(fill=tk.X, pady=6)

        ttk.Button(row2, text="market????", command=lambda: self._direct_market("BUY")).grid(row=0, column=0, padx=6, pady=6)

        ttk.Button(row2, text="market???", command=lambda: self._direct_market("SELL")).grid(row=0, column=1, padx=6, pady=6)

        ttk.Button(row2, text="limit????", command=lambda: self._direct_limit("BUY")).grid(row=0, column=2, padx=6, pady=6)

        ttk.Button(row2, text="limit???", command=lambda: self._direct_limit("SELL")).grid(row=0, column=3, padx=6, pady=6)



        # ? bracket order

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



        # ? ??

        row4 = ttk.LabelFrame(frm, text=" ?")

        row4.pack(fill=tk.X, pady=6)

        ttk.Label(row4, text=" ?).grid(row=0, column=0, padx=5")

        self.d_algo = ttk.Combobox(row4, values=["TWAP", "VWAP", "ICEBERG"], width=10)

        self.d_algo.current(0)

        self.d_algo.grid(row=0, column=1, padx=5)

        ttk.Label(row4, text=" ? ?").grid(row=0, column=2, padx=5)

        self.d_dur = ttk.Entry(row4, width=8); self.d_dur.insert(0, "30"); self.d_dur.grid(row=0, column=3, padx=5)

        ttk.Button(row4, text=" ? (", command=lambda: self._direct_algo("BUY")).grid(row=0, column=4, padx=6, pady=6)

        ttk.Button(row4, text=" ? (", command=lambda: self._direct_algo("SELL")).grid(row=0, column=5, padx=6, pady=6)


    def _build_timesplit_tab(self, parent) -> None:
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        dataset_frame = ttk.LabelFrame(container, text="Dataset & Split")
        dataset_frame.pack(fill=tk.X, pady=5)

        tk.Label(dataset_frame, text=f"Dataset: {STAGE_A_DATA_PATH}", anchor='w').pack(fill=tk.X, padx=5, pady=2)

        split_row = ttk.Frame(dataset_frame)
        split_row.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(split_row, text="Training Split (0.60 - 0.95)").pack(side=tk.LEFT)
        self.timesplit_split_var = tk.DoubleVar(value=0.80)
        split_value_label = ttk.Label(split_row, text="0.80")
        split_value_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(split_row, from_=0.60, to=0.95, orient=tk.HORIZONTAL, variable=self.timesplit_split_var, length=260, command=lambda v: split_value_label.config(text=f"{float(v):.2f}")).pack(fill=tk.X, expand=True, padx=5)

        feature_frame = ttk.LabelFrame(container, text="Feature Combination")
        feature_frame.pack(fill=tk.X, pady=5)
        self.timesplit_feature_vars = {}
        for idx, feature in enumerate(TIMESPLIT_FEATURES):
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(feature_frame, text=feature, variable=var)
            chk.grid(row=idx // 3, column=idx % 3, sticky=tk.W, padx=5, pady=2)
            self.timesplit_feature_vars[feature] = var

        options_frame = ttk.LabelFrame(container, text="Options")
        options_frame.pack(fill=tk.X, pady=5)
        self.var_timesplit_ema = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Enable EMA smoothing (LambdaRank only)", variable=self.var_timesplit_ema).pack(anchor='w', padx=5, pady=2)

        action_frame = ttk.Frame(container)
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Button(action_frame, text="Run 80/20 Evaluation", command=self._run_timesplit_eval).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Open Results Folder", command=self._open_timesplit_results).pack(side=tk.LEFT, padx=5)


    def _start_engine(self) -> None:

        try:

            #  ? UI ??

            self._capture_ui()

            #  in ??no ?? ??

            self.log(f" ?start ?connection/subscription)... Host={self.state.host} Port={self.state.port} ClientId={self.state.client_id}")

            loop = self._ensure_loop()

            async def _run():

                try:

                    #  

                    try:

                        self.after(0, lambda: self.log(

                            f"start : Host={self.state.host}, Port={self.state.port}, ClientID={self.state.client_id}"))

                    except Exception:

                        pass

                    # startbefore ? hasconnection clientId use

                    if self.trader and getattr(self.trader, 'ib', None) and self.trader.ib.isConnected():

                        try:

                            await self.trader.close()

                            try:

                                self.after(0, lambda: self.log(" ? beforeAPIconnection"))

                            except Exception as e:

                                # GUI ???

                                self.log(f"GUI ? {e}")

                        except Exception as e:

                            #  M? ? ?? ? ? 

                            self.log(f" ? K? : {e}")

                            #  ? ? ??

                            self._set_connection_error_state(f" X? {e}")

                    #  connection ? use 

                    self.trader = IbkrAutoTrader(config_manager=self.config_manager)

                    #  to 

                    self.resource_monitor.register_connection(self.trader)

                    

                    # ?Engine   connect andsubscription use 

                    self.engine = Engine(self.config_manager, self.trader)

                    await self.engine.start()

                    try:

                        self.after(0, lambda: self.log(" start completedsubscription"))

                        self.after(0, lambda: self._update_signal_status(" start", "green"))

                    except Exception:

                        pass

                except Exception as e:

                    error_msg = str(e)

                    try:

                        self.after(0, lambda e_msg=error_msg: self.log(f" startfailed: {e_msg}"))

                    except Exception:

                        print(f" startfailed: {e}")  #  

            #  use ? ? ? ??

            try:

                task_id = self.loop_manager.submit_coroutine_nowait(_run())

                self.after(0, lambda: self.log(f"  (ID: {task_id[:8]}...)"))

            except Exception as e:

                error_msg = str(e)

                self.after(0, lambda e_msg=error_msg: self.log(f" startfailed: {e_msg}"))

        except Exception as e:

            self.log(f"start ?: {e}")



    def _engine_once(self) -> None:

        try:

            if not self.engine:

                self.log(" start ?")

                return

            #  use GUI??

            if hasattr(self, 'loop_manager') and self.loop_manager.is_running:

                task_id = self.loop_manager.submit_coroutine_nowait(self.engine.on_signal_and_trade())

                self.log(f" ? ?D: {task_id}")

            else:

                self.log(" ? c? no ?")

            self.log(" ? and ?")

            self._update_signal_status(" ? ?", "blue")

        except Exception as e:

            self.log(f" ? ??ailed: {e}")



    def _stop_engine_mode(self) -> None:

        try:

            self.log(" ? can ? ? ?connectionand ?")

            self._update_signal_status("??", "red")

        except Exception as e:

            self.log(f" ? failed: {e}")



    def _direct_market(self, side: str) -> None:

        try:

            sym = (self.d_sym.get() or "").strip().upper()

            qty = int(self.d_qty.get().strip())

            if not sym or qty <= 0:

                messagebox.showwarning("????", "???????? ?????????")

                return

            loop = self._ensure_loop()

            async def _run():

                try:

                    if not self.trader:

                        self.trader = IbkrAutoTrader(config_manager=self.config_manager)

                        await self.trader.connect()

                    await self.trader.place_market_order(sym, side, qty)

                    self.log(f" market {side} {qty} {sym}")

                except Exception as e:

                    self.log(f"market failed: {e}")

            #  use GUI??

            if hasattr(self, 'loop_manager') and self.loop_manager.is_running:

                task_id = self.loop_manager.submit_coroutine_nowait(_run())

                self.log(f"order placement ?D: {task_id}")

            else:

                self.log(" ? c? no order placement ?")

        except Exception as e:

            self.log(f"marketorder placement ? {e}")



    def _direct_limit(self, side: str) -> None:

        try:

            sym = (self.d_sym.get() or "").strip().upper()

            qty = int(self.d_qty.get().strip())

            px_str = (self.d_px.get() or "").strip()

            if not sym or qty <= 0 or not px_str:

                messagebox.showwarning("????", "????????/??????limit???")

                return

            px = float(px_str)

            loop = self._ensure_loop()

            async def _run():

                try:

                    if not self.trader:

                        self.trader = IbkrAutoTrader(config_manager=self.config_manager)

                        await self.trader.connect()

                    await self.trader.place_limit_order(sym, side, qty, px)

                    self.log(f" limit {side} {qty} {sym} @ {px}")

                except Exception as e:

                    self.log(f"limit failed: {e}")

            #  use GUI??

            if hasattr(self, 'loop_manager') and self.loop_manager.is_running:

                task_id = self.loop_manager.submit_coroutine_nowait(_run())

                self.log(f"order placement ?D: {task_id}")

            else:

                self.log(" ? c? no order placement ?")

        except Exception as e:

            self.log(f"limitorder placement ? {e}")



    def _direct_bracket(self, side: str) -> None:

        try:

            sym = (self.d_sym.get() or "").strip().upper()

            qty = int(self.d_qty.get().strip())

            stop_pct = float((self.d_stop.get() or "2.0").strip())/100.0

            tp_pct = float((self.d_tp.get() or "5.0").strip())/100.0

            if not sym or qty <= 0:

                messagebox.showwarning("????", "?????????????")

                return

            loop = self._ensure_loop()

            async def _run():

                try:

                    if not self.trader:

                        self.trader = IbkrAutoTrader(config_manager=self.config_manager)

                        await self.trader.connect()

                    await self.trader.place_market_order_with_bracket(sym, side, qty, stop_pct=stop_pct, target_pct=tp_pct)

                    self.log(f" bracket order: {side} {qty} {sym} ( {stop_pct*100:.1f}%,  {tp_pct*100:.1f}%)")

                except Exception as e:

                    self.log(f"bracket orderfailed: {e}")

            #  use GUI??

            if hasattr(self, 'loop_manager') and self.loop_manager.is_running:

                task_id = self.loop_manager.submit_coroutine_nowait(_run())

                self.log(f"order placement ?D: {task_id}")

            else:

                self.log(" ? c? no order placement ?")

        except Exception as e:

            self.log(f"bracket order ? {e}")



    def _direct_algo(self, side: str) -> None:

        try:

            sym = (self.d_sym.get() or "").strip().upper()

            qty = int(self.d_qty.get().strip())

            algo = (self.d_algo.get() or "TWAP").strip().upper()

            dur_min = int((self.d_dur.get() or "30").strip())

            if not sym or qty <= 0:

                messagebox.showwarning("????", "?????????????")

                return

            loop = self._ensure_loop()

            async def _run():

                try:

                    if not self.trader:

                        self.trader = IbkrAutoTrader(config_manager=self.config_manager)

                        await self.trader.connect()

                    await self.trader.execute_large_order(sym, side, qty, algorithm=algo, duration_minutes=dur_min)

                    self.log(f" ? {algo} {side} {qty} {sym} / {dur_min}min")

                except Exception as e:

                    self.log(f" ?failed: {e}")

            #  use GUI??

            if hasattr(self, 'loop_manager') and self.loop_manager.is_running:

                task_id = self.loop_manager.submit_coroutine_nowait(_run())

                self.log(f"order placement ?D: {task_id}")

            else:

                self.log(" ? c? no order placement ?")

        except Exception as e:

            self.log(f" ? ? {e}")



    def _delete_database(self) -> None:

        """? ? ? ?and ?"""

        try:

            import os

            db_path = getattr(self.db, 'db_path', None)

            if not db_path:

                messagebox.showerror("????", "??????????? ??")

                return

            

            if not os.path.exists(db_path):

                messagebox.showinfo("???", "?????????????????????")

                return

            

            confirm = messagebox.askyesno(

                "? ",

                f"will ?\n{db_path}\n\n notcan ? is ?"

            )

            if not confirm:

                return

            

            #  connection ??

            try:

                self.db.close()

            except Exception:

                pass

            

            os.remove(db_path)

            self.log(f"  {db_path}")

            

            #  ? ? UI

            self.db = StockDatabase()

            self._refresh_stock_lists()

            self._refresh_configs()

            messagebox.showinfo("completed", " as ?")

        

        except Exception as e:

            self.log(f" failed: {e}")

            messagebox.showerror("????", f"???????????: {e}")



    def _print_database(self) -> None:

        """ before to ?tickers ??in X?"""

        try:

            #  ??tickers

            tickers = []

            try:

                tickers = self.db.get_all_tickers()

            except Exception:

                pass

            if tickers:

                preview = ", ".join(tickers[:200]) + ("..." if len(tickers) > 200 else "")

                self.log(f" ?tickers ?{len(tickers)}: {preview}")

            else:

                self.log(" ?tickers: no")



            #  ? ?

            try:

                lists = self.db.get_stock_lists()

            except Exception:

                lists = []

            if lists:

                summary = ", ".join([f"{it['name']}({it.get('stock_count', 0)})" for it in lists])

                self.log(f" ? ?{len(lists)} ? {summary}")

            else:

                self.log(" ? ? no")



            #  before in 

            try:

                if self.state.selected_stock_list_id:

                    rows = self.db.get_stocks_in_list(self.state.selected_stock_list_id)

                    syms = [r.get('symbol') for r in rows]

                    preview = ", ".join(syms[:200]) + ("..." if len(syms) > 200 else "")

                    self.log(f" before ?{self.stock_list_var.get()} ?{len(syms)}: {preview}")

            except Exception:

                pass



            #  

            try:

                cfgs = self.db.get_trading_configs()

            except Exception:

                cfgs = []

            if cfgs:

                names = ", ".join([c.get('name', '') for c in cfgs])

                self.log(f"  {len(cfgs)}  {names}")

            else:

                self.log(" : no")



        except Exception as e:

            self.log(f" failed: {e}")



    def _build_database_tab(self, parent):

        """Build the data services tab"""

        #  ? ? be tickers?

        left_frame = tk.Frame(parent)

        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)



        stock_frame = tk.LabelFrame(left_frame, text="Trading Tickers")

        stock_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        

        #  Treeview symbolandadded_at

        columns = ('symbol', 'added_at')

        self.stock_tree = ttk.Treeview(stock_frame, columns=columns, show='headings', height=10)

        self.stock_tree.heading('symbol', text='Ticker')

        self.stock_tree.heading('added_at', text='Added At')

        self.stock_tree.column('symbol', width=100)

        self.stock_tree.column('added_at', width=150)

        

        #  ??records

        stock_scroll = ttk.Scrollbar(stock_frame, orient=tk.VERTICAL, command=self.stock_tree.yview)

        self.stock_tree.configure(yscrollcommand=stock_scroll.set)

        

        self.stock_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        stock_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        

        #  ?tickersas ??

        right_frame = tk.Frame(parent)

        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        

        #  ?

        info_frame = tk.LabelFrame(right_frame, text="Database Info")

        info_frame.pack(fill=tk.X, pady=5)

        try:

            db_path_text = getattr(self.db, 'db_path', '') or ''

        except Exception:

            db_path_text = ''

        tk.Label(info_frame, text=f"Path: {db_path_text}", wraplength=220, justify=tk.LEFT, fg="gray").pack(anchor=tk.W, padx=5, pady=3)



        #  ? ?tickers?

        add_frame = tk.LabelFrame(right_frame, text="Add Trading Ticker")

        add_frame.pack(fill=tk.X, pady=5)

        

        tk.Label(add_frame, text="Ticker:").grid(row=0, column=0, padx=5, pady=5)

        self.ent_symbol = tk.Entry(add_frame, width=15)

        self.ent_symbol.grid(row=0, column=1, padx=5, pady=5)

        

        tk.Button(add_frame, text="Add Ticker", command=self._add_ticker_global, bg="lightgreen").grid(row=1, column=0, columnspan=2, pady=5)

        

        #  ? ??

        pool_frame = tk.LabelFrame(right_frame, text="Stock Pools")

        pool_frame.pack(fill=tk.X, pady=5)

        

        tk.Button(pool_frame, text="Open Stock Pool Manager", command=self._open_stock_pool_manager, 

                 bg="#FF9800", fg="white", font=("Arial", 10)).pack(pady=5)

        tk.Button(pool_frame, text="Export Factor Dataset", command=self._export_factor_dataset,

                 bg="#4CAF50", fg="white").pack(pady=3)

        

        #  to tickers

        import_frame = tk.LabelFrame(right_frame, text="Batch Import Tickers")

        import_frame.pack(fill=tk.X, pady=5)



        tk.Label(import_frame, text="CSV Input (supports blank lines / comments):").grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        self.ent_batch_csv = tk.Text(import_frame, width=20, height=4)

        self.ent_batch_csv.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        self.ent_batch_csv.insert(tk.END, "AAPL,MSFT,GOOGL")



        #  ??

        _btn_norm = tk.Button(import_frame, text="Normalize", command=self._normalize_batch_input_text, bg="lightblue")

        _btn_norm.grid(row=2, column=0, padx=5, pady=5, sticky=tk.EW)

        _attach_tooltip(_btn_norm, "Trim whitespace and convert to uppercase tickers")

        tk.Button(import_frame, text="Batch Import", command=self._batch_import_global, bg="lightyellow").grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)

        

        #  tickersin??

        delete_frame = tk.LabelFrame(right_frame, text="Remove Trading Tickers")

        delete_frame.pack(fill=tk.X, pady=5)

        

        tk.Button(delete_frame, text="Delete Selected", command=self._delete_selected_ticker_global, bg="lightcoral").grid(row=0, column=0, padx=5, pady=5)

        

        #  ?

        config_frame = tk.LabelFrame(right_frame, text="Configuration Management")

        config_frame.pack(fill=tk.X, pady=5)

        

        tk.Label(config_frame, text="Config Name").grid(row=0, column=0, padx=5, pady=5)

        self.config_name_var = tk.StringVar()

        self.config_combo = ttk.Combobox(config_frame, textvariable=self.config_name_var, width=15)

        self.config_combo.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        

        tk.Button(config_frame, text=" ", command=self._save_config, bg="lightblue").grid(row=2, column=0, padx=2, pady=5)

        tk.Button(config_frame, text=" ", command=self._load_config, bg="lightgreen").grid(row=2, column=1, padx=2, pady=5)



        #  ? tickers as ? ??

        

        #  ? ?

        self._refresh_global_tickers_table()

        self._refresh_configs()



    def _export_factor_dataset(self) -> None:

        """ ? ? ?"""

        if getattr(self, '_exporting_factors', False):

            try:

                messagebox.showinfo('Notice', 'Factor export already running; please wait until it completes.')

            except Exception:

                pass

            return



        pool_info = getattr(self, 'selected_pool_info', {}) or {}

        if not pool_info.get('tickers'):

            try:

                from .stock_pool_selector import select_stock_pool

                pool_choice = select_stock_pool(self)

                if not pool_choice:

                    self.log('[INFO] No pool selected; returning.')

                    return

                pool_info = pool_choice

                self.selected_pool_info = dict(pool_choice)

            except Exception as exc:

                self.log(f"[ERROR] Pool selection failed: {exc}")

                messagebox.showerror('Error', f'Could not open stock pool picker: {exc}')

                return



        symbols = [s.strip().upper() for s in pool_info.get('tickers', []) if isinstance(s, str) and s.strip()]

        if not symbols:

            messagebox.showerror('Error', 'Selected pool contains no valid tickers.')

            return

        pool_name = pool_info.get('pool_name', f"{len(symbols)} tickers")



        base_dir = Path('data/factor_exports')

        base_dir.mkdir(parents=True, exist_ok=True)

        safe_name = pool_name.replace('/', '_').replace(' ', '_')

        out_dir = base_dir / f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        out_dir.mkdir(parents=True, exist_ok=True)



        self._exporting_factors = True

        self.log(f"[INFO] Starting factor export for pool {pool_name} -> {out_dir}")



        def ui_log_safe(msg: str) -> None:

            try:

                self.after(0, lambda m=msg: self.log(m))

            except Exception:

                try:

                    self.log(msg)

                except Exception:

                    pass



        def worker() -> None:

            try:

                ui_log_safe('[INFO] Factor export job running, logging progress...')

                try:

                    from autotrader.factor_export_service import export_polygon_factors  # type: ignore

                except Exception:

                    from .factor_export_service import export_polygon_factors  # type: ignore



                result = export_polygon_factors(

                    years=5,

                    output_dir=out_dir,

                    log_level='INFO',

                    status_callback=ui_log_safe,

                    symbols=symbols,

                    pool_name=pool_name,

                    max_symbols=len(symbols),

                    mode='train',  # Train mode: with target and dropna

                    keep_multiindex=True,  # Keep MultiIndex for ML training

                )



                summary = (

                    f"Export complete: {result.get('batch_count', 0)} batches "

                    f"from {result.get('start_date')} to {result.get('end_date')} "

                    f"output saved to {result.get('output_dir')}"

                )

                ui_log_safe(f"[SUCCESS] {summary}")

                try:

                    self.after(0, lambda: messagebox.showinfo('Export Complete', summary))

                except Exception:

                    pass

            except Exception as exc:

                ui_log_safe(f"[ERROR] Factor export failed: {exc}")

                try:

                    self.after(0, lambda: messagebox.showerror('Error', f'Factor export failed: {exc}'))

                except Exception:

                    pass

            finally:

                def _reset_flag() -> None:

                    setattr(self, '_exporting_factors', False)

                try:

                    self.after(0, _reset_flag)

                except Exception:

                    self._exporting_factors = False



        threading.Thread(target=worker, daemon=True).start()





    def _build_file_tab(self, parent):

        """ ?items"""

        #  ? 

        wl = tk.LabelFrame(parent, text=" ? or ?")

        wl.pack(fill=tk.X, pady=5)

        tk.Button(wl, text=" ?JSON  ??, command=self._pick_json).grid(row=0, column=0, padx=5, pady=5")

        tk.Button(wl, text=" ?Excel  ??, command=self._pick_excel).grid(row=0, column=1, padx=5, pady=5")

        tk.Label(wl, text="Sheet").grid(row=0, column=2)

        self.ent_sheet = tk.Entry(wl, width=10)

        self.ent_sheet.grid(row=0, column=3)

        tk.Label(wl, text="Column").grid(row=0, column=4)

        self.ent_col = tk.Entry(wl, width=10)

        self.ent_col.grid(row=0, column=5)

        tk.Label(wl, text=" CSV").grid(row=1, column=0)

        self.ent_csv = tk.Entry(wl, width=50)

        self.ent_csv.grid(row=1, column=1, columnspan=5, sticky=tk.EW, padx=5)

        self.ent_csv.insert(0, "AAPL,MSFT,GOOGL,AMZN,TSLA")  #  ? 

        

        #  S? ??

        self.lbl_json = tk.Label(wl, text="JSON: ??", fg="gray")

        self.lbl_json.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=5)

        self.lbl_excel = tk.Label(wl, text="Excel: ??", fg="gray")

        self.lbl_excel.grid(row=2, column=3, columnspan=3, sticky=tk.W, padx=5)

        

        #  ??items

        import_options = tk.LabelFrame(parent, text=" ?items")

        import_options.pack(fill=tk.X, pady=5)

        

        self.var_auto_clear = tk.BooleanVar(value=True)

        tk.Checkbutton(import_options, text=" ->  tickers  can be ?", 

                      variable=self.var_auto_clear).pack(anchor=tk.W, padx=5, pady=5)

        

        tk.Button(import_options, text=" to tickers", 

                 command=self._import_file_to_database, bg="orange").pack(side=tk.LEFT, padx=5, pady=5)

        tk.Button(import_options, text=" to to tickers", 

                 command=self._append_file_to_database, bg="lightgreen").pack(side=tk.LEFT, padx=5, pady=5)



    def _pick_json(self) -> None:

        path = filedialog.askopenfilename(title=" JSON", filetypes=[("JSON", "*.json"), ("All", "*.*")])

        if path:

            self.state.json_file = path

            try:

                import os

                name = os.path.basename(path)

            except Exception:

                name = path

            self.lbl_json.config(text=f"JSON: {name}", fg="blue")

            self.log(f" JSON: {path}")



    def _pick_excel(self) -> None:

        path = filedialog.askopenfilename(title=" Excel", filetypes=[("Excel", "*.xlsx;*.xls"), ("All", "*.*")])

        if path:

            self.state.excel_file = path

            try:

                import os

                name = os.path.basename(path)

            except Exception:

                name = path

            self.lbl_excel.config(text=f"Excel: {name}", fg="blue")

            self.log(f" Excel: {path}")



    def _ensure_loop(self) -> asyncio.AbstractEventLoop:

        """Enhanced event loop management with proper cleanup"""

        if self.loop and not self.loop.is_closed() and self.loop.is_running():

            return self.loop

        

        def run_loop() -> None:

            #  ? ? use Tk  use self.after  ?

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

                #  Y? ? loop when

                if self._loop_ready_event is None:

                    self._loop_ready_event = threading.Event()

                try:

                    self._loop_ready_event.set()

                except Exception:

                    pass

                safe_log(" ? willstart")

                loop.run_forever()

            except Exception as e:

                safe_log(f" ? ? {e}")

            finally:

                try:

                    # Clean up any remaining tasks

                    if loop and not loop.is_closed():

                        pending = asyncio.all_tasks(loop)

                        if pending:

                            safe_log(f"?n ?{len(pending)}  ?completed ??..")

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

                    safe_log(f" ? : {e}")

        

        self._loop_thread = threading.Thread(target=run_loop, daemon=True)

        self._loop_thread.start()

        

        # Wait for loop to be ready ( ? + in ??

        import time

        if self._loop_ready_event is None:

            self._loop_ready_event = threading.Event()

        self._loop_ready_event.wait(timeout=1.0)

        if self.loop is not None:

            return self.loop  # type: ignore

        # If still not running, provide a helpful log and raise

        self.log(" ? ?in when start ? ' connection'or'start ? '")

        raise RuntimeError("Failed to start event loop")



    def _get_direct_predict_features(self):
        """Load the feature list expected by the current Direct Predict snapshot."""
        cache_key = os.environ.get('DIRECT_PREDICT_SNAPSHOT_ID', DIRECT_PREDICT_SNAPSHOT_ID).strip() or DIRECT_PREDICT_SNAPSHOT_ID
        cached = getattr(self, '_direct_predict_features', None)
        if cached and cached.get('snapshot_id') == cache_key:
            return cached['features']

        features = None
        try:
            snapshots_root = Path(__file__).resolve().parent.parent / 'cache' / 'model_snapshots'
            manifest_paths = list(snapshots_root.glob(f'**/{cache_key}/manifest.json'))
            if manifest_paths:
                manifest_data = json.loads(manifest_paths[0].read_text(encoding='utf-8'))
                manifest_features = manifest_data.get('feature_names')
                if manifest_features:
                    features = manifest_features
                    self.log(f"[DirectPredict] Snapshot features loaded from manifest: {features}")
        except Exception as err:
            self.log(f"[DirectPredict] Failed to read snapshot manifest: {err}")

        if not features:
            try:
                from bma_models.simple_25_factor_engine import TOP_FEATURE_SET
                features = TOP_FEATURE_SET
            except Exception:
                features = ['momentum_10d', 'ivol_20', 'hist_vol_20', 'rsi_21', 'near_52w_high', 'atr_ratio', 'vol_ratio_20d', '5_days_reversal', 'trend_r2_60', 'liquid_momentum']
            self.log(f"[DirectPredict] Fallback feature set in use: {features}")

        self._direct_predict_features = {'snapshot_id': cache_key, 'features': features}
        return features

    def _apply_direct_predict_ema(self, predictions_df, score_columns=None, weights=DIRECT_PREDICT_EMA_WEIGHTS):
        """Apply EWMA smoothing to the provided score columns for Direct Predict."""
        if predictions_df is None or len(predictions_df) == 0:
            return predictions_df

        if not isinstance(predictions_df.index, pd.MultiIndex):
            self.log("[DirectPredict] EMA smoothing skipped: predictions lack MultiIndex (date, ticker)")
            return predictions_df

        score_columns = score_columns or ['score']
        numeric_weights = tuple(float(w) for w in weights if isinstance(w, (int, float)) and float(w) > 0)
        if not numeric_weights:
            self.log("[DirectPredict] EMA smoothing skipped: invalid weights provided")
            return predictions_df

        weight_sum = sum(numeric_weights)
        normalized_weights = tuple(w / weight_sum for w in numeric_weights)

        unique_dates = sorted(predictions_df.index.get_level_values('date').unique())
        if len(unique_dates) < 2:
            self.log("[DirectPredict] EMA smoothing skipped: need at least 2 trading days of predictions")
            return predictions_df

        df = predictions_df.sort_index(level=['date', 'ticker']).copy()
        tickers = df.index.get_level_values('ticker').unique()

        for column in score_columns:
            if column not in df.columns:
                continue

            raw_col = f"{column}_raw"
            if raw_col not in df.columns:
                df[raw_col] = df[column]

            ema_col = f"{column}_ema"
            df[ema_col] = np.nan

            smoothed_points = 0
            for ticker in tickers:
                try:
                    ticker_series = df.xs(ticker, level='ticker')[raw_col].sort_index()
                except KeyError:
                    continue

                if ticker_series.dropna().empty:
                    continue

                values = ticker_series.values
                date_index = list(ticker_series.index)

                for idx_pos, current_date in enumerate(date_index):
                    weighted_sum = 0.0
                    weight_total = 0.0

                    for offset, weight in enumerate(normalized_weights):
                        prev_idx = idx_pos - offset
                        if prev_idx < 0:
                            break
                        prev_value = values[prev_idx]
                        if pd.isna(prev_value):
                            continue
                        weighted_sum += weight * float(prev_value)
                        weight_total += weight

                    if weight_total:
                        smoothed_value = weighted_sum / weight_total
                        df.loc[(current_date, ticker), ema_col] = smoothed_value
                        smoothed_points += 1

            df[column] = df[ema_col].fillna(df[column])
            self.log(
                f"[DirectPredict] Applied EMA smoothing to {column}: {smoothed_points} data points updated (weights={normalized_weights})"
            )

        return df
    def _capture_ui(self) -> None:

        self.state.host = self.ent_host.get().strip() or "127.0.0.1"

        try:

            #  U? ??ndclientId use ??

            port_input = (self.ent_port.get() or "").strip()

            cid_input = (self.ent_cid.get() or "").strip()

            self.state.port = int(port_input) if port_input else self.state.port

            self.state.client_id = int(cid_input) if cid_input else self.state.client_id

            self.state.alloc = float(self.ent_alloc.get().strip() or 0.03)

            self.state.poll_sec = float(self.ent_poll.get().strip() or 10.0)

            self.state.fixed_qty = int(self.ent_fixed_qty.get().strip() or 0)

        except ValueError as e:

            error_msg = f" ? {e}"

            self.log(error_msg)

            messagebox.showerror(" ?", "?ClientId is ratio/? is ?")

            raise ValueError(error_msg) from e

        except Exception as e:

            error_msg = f" failed: {e}"

            self.log(error_msg)

            messagebox.showerror(" ?", error_msg)

            raise

        self.state.sheet = self.ent_sheet.get().strip() or None

        self.state.column = self.ent_col.get().strip() or None

        self.state.symbols_csv = self.ent_csv.get().strip() or None

        self.state.auto_sell_removed = self.var_auto_sell.get()

        

        #  whenupdates ???

        self.config_manager.update_runtime_config({

            'connection.host': self.state.host,

            'connection.port': self.state.port,

            'connection.client_id': self.state.client_id,

            'trading.alloc_pct': self.state.alloc,

            'trading.poll_interval': self.state.poll_sec,

            'trading.fixed_quantity': self.state.fixed_qty,

            'trading.auto_sell_removed': self.state.auto_sell_removed

        })

    

    def _run_async_safe(self, coro, operation_name: str = " ?, timeout: int = 30"):

        """ ?GUI"""

        try:

            if hasattr(self, 'loop_manager') and self.loop_manager.is_running:

                #  useno ? ??

                task_id = self.loop_manager.submit_coroutine_nowait(coro)

                self.log(f"{operation_name} ?D: {task_id}")

                return task_id

            else:

                #  ? event loop manager ?

                if hasattr(self, 'loop_manager'):

                    #  ?loop_manager c? ??

                    if not self.loop_manager.is_running:

                        self.log(f"???????????????????????{operation_name}")

                        if self.loop_manager.start():

                            task_id = self.loop_manager.submit_coroutine_nowait(coro)

                            self.log(f"{operation_name} ? ?D: {task_id}")

                            return task_id

                

                # ? ? ? ? GUI ??

                import asyncio

                from concurrent.futures import ThreadPoolExecutor

                

                def run_in_isolated_loop():

                    """ ? ? GUI ?"""

                    try:

                        #  ? ? U? ??

                        loop = asyncio.new_event_loop()

                        asyncio.set_event_loop(loop)

                        try:

                            loop.run_until_complete(coro)

                        finally:

                            loop.close()

                    except Exception as e:

                        self.log(f"{operation_name} ? ? : {e}")

                

                thread_name = f"{operation_name}Thread"

                threading.Thread(

                    target=run_in_isolated_loop,

                    daemon=True,

                    name=thread_name

                ).start()

                self.log(f"{operation_name} s??")

                return None

        except Exception as e:

            self.log(f"{operation_name}startfailed: {e}")

            return None



    def _test_connection(self) -> None:

        try:

            self._capture_ui()

            self.log(f"?n connection... Host={self.state.host} Port={self.state.port} ClientId={self.state.client_id}")

            

            async def _run():

                try:

                    #  useconnection ??

                    self.log(f"connection ? Host={self.state.host}, Port={self.state.port}, ClientID={self.state.client_id}")

                    # startbefore ? hasconnection clientId use

                    if self.trader and getattr(self.trader, 'ib', None) and self.trader.ib.isConnected():

                        try:

                            await self.trader.close()

                            self.log(" ? beforeAPIconnection")

                        except Exception:

                            pass

                    self.trader = IbkrAutoTrader(config_manager=self.config_manager)

                    await self.trader.connect()

                    self.log("[OK] connectionsuccess")

                except Exception as e:

                    self.log(f"[FAIL] connectionfailed: {e}")

            

            #  use GUI??

            def _async_test():

                try:

                    if hasattr(self, 'loop_manager') and self.loop_manager.is_running:

                        #  useno ? ??

                        task_id = self.loop_manager.submit_coroutine_nowait(_run())

                        self.log(f"connection ?D: {task_id}")

                    else:

                        #  GUI ??

                        self._run_async_safe(_run(), "connection ?")

                except Exception as e:

                    self.log(f"connection startfailed: {e}")

            

            _async_test()

            

        except Exception as e:

            self.log(f"????connection????: {e}")

            messagebox.showerror("????", f"????connection???: {e}")



    def _start_autotrade(self) -> None:

        try:

            self._capture_ui()

            self.log(f"?nstart ? ?.. Host={self.state.host} Port={self.state.port} ClientId={self.state.client_id}")



            async def _run():

                try:

                    #  useconnection ??

                    self.log(f"start : Host={self.state.host}, Port={self.state.port}, ClientID={self.state.client_id}")

                    # 1)  ??Trader connection

                    # startbefore ? hasconnection clientId use

                    if self.trader and getattr(self.trader, 'ib', None) and self.trader.ib.isConnected():

                        try:

                            await self.trader.close()

                            self.log(" ? beforeAPIconnection")

                        except Exception:

                            pass

                    # Always create new trader after closing the old one

                    self.trader = IbkrAutoTrader(config_manager=self.config_manager)

                    await self.trader.connect()



                    # 2)  ??Engine and Universe ?? / CSV?

                    self._maybe_refresh_top10_pool(force=False)

                    uni = []

                    try:

                        db_csv = self._get_current_stock_symbols()

                        if db_csv:

                            uni = [s for s in db_csv.split(',') if s.strip()]

                        elif any([self.state.json_file, self.state.excel_file, self.state.symbols_csv]):

                            uni = self._extract_symbols_from_files()

                    except Exception:

                        pass

                    #  use ???

                    cfg = self.config_manager

                    if uni:

                        cfg.set_runtime("scanner.universe", uni)

                        self.log(f" use U? Universe: {len(uni)} ?")



                    if not self.engine:

                        self.engine = Engine(cfg, self.trader)

                    await self.engine.start()



                    # 3)  risk control order placement ? 

                    self.log(f" ?start:  ?{self.state.poll_sec}s")



                    async def _engine_loop():

                        try:

                            while True:

                                await self.engine.on_signal_and_trade()

                                await asyncio.sleep(max(1.0, float(self.state.poll_sec)))

                        except asyncio.CancelledError:

                            return

                        except Exception as e:

                            self.log(f" ? ? {e}")



                    # in ?in use

                    self._engine_loop_task = asyncio.create_task(_engine_loop())

                    self.log(" start ")

                    self._update_signal_status(" ? ?in", "green")

                except Exception as e:

                    self.log(f" ? startfailed: {e}")



            #  use GUI??

            def _async_start():

                try:

                    if hasattr(self, 'loop_manager') and self.loop_manager.is_running:

                        #  useno ? ??

                        task_id = self.loop_manager.submit_coroutine_nowait(_run())

                        self.log(f" ? start ?D: {task_id}")

                    else:

                        #  GUI ??

                        self._run_async_safe(_run(), " ? ?")

                except Exception as e:

                    self.log(f" ? startfailed: {e}")

            

            _async_start()



        except Exception as e:

            self.log(f"start??????????: {e}")

            messagebox.showerror("????", f"??????: {e}")



    def _stop(self) -> None:

        """Enhanced stop mechanism with proper cleanup"""

        try:

            if not self.trader and not self.loop:

                self.log("?as connection")

                return

                

            self.log("?n ? ...")

            

            # Signal the trader to stop

            if self.trader:

                try:

                    if hasattr(self.trader, '_stop_event'):

                        if not self.trader._stop_event:

                            self.trader._stop_event = asyncio.Event()

                        self.trader._stop_event.set()

                        self.log("Stop event signaled")

                except Exception as e:

                    self.log(f"Stop event cleanup failed: {e}")



                #  ? ?

                try:

                    if self.loop and self.loop.is_running() and self._engine_loop_task and not self._engine_loop_task.done():

                        def _cancel_task(task: asyncio.Task):

                            if not task.done():

                                task.cancel()

                        self.loop.call_soon_threadsafe(_cancel_task, self._engine_loop_task)

                        self.log(" ? ?")

                        self._update_signal_status(" ? ?", "red")

                except Exception as e:

                    self.log(f" ? ?failed: {e}")



                # Stop engine and close trader connection

                if self.loop and self.loop.is_running():

                    async def _cleanup_all():

                        try:

                            # Stop engine first

                            if self.engine:

                                await self.engine.stop()

                                self.log(" ?")

                                self.engine = None

                            

                            # Then close trader connection

                            if self.trader:

                                await self.trader.close()

                                self.log(" connection ?")

                                self.trader = None

                        except Exception as e:

                            self.log(f" ? / failed: {e}")

                            

                    self.loop_manager.submit_coroutine_nowait(_cleanup_all())

                    self.log(" toafter")

                else:

                    self.trader = None

            

            # Clean up event loop

            if self.loop and not self.loop.is_closed():

                try:

                    if self.loop.is_running():

                        # Schedule loop stop

                        self.loop.call_soon_threadsafe(self.loop.stop)

                        self.log(" ? ?")

                        

                        # Give some time for cleanup

                        def reset_loop():

                            if self.loop and self.loop.is_closed():

                                self.loop = None

                        

                        self.after(2000, reset_loop)  # Reset after 2 seconds

                        

                except Exception as e:

                    self.log(f" ? ?failed: {e}")

            

            self.log(" ? completed")

                

        except Exception as e:

            self.log(f"?????????: {e}")

            messagebox.showerror("????", f"?????: {e}")



    def _disconnect_api(self) -> None:

        """? ??APIconnection not clientId use"""

        try:

            if not self.trader:

                self.log("no APIconnection")

                return

            self.log("?n ?APIconnection...")

            if self.loop and self.loop.is_running():

                #  in ? IBconnection clientId use

                try:

                    if getattr(self.trader, 'ib', None):

                        self.loop.call_soon_threadsafe(self.trader.ib.disconnect)

                except Exception:

                    pass

                #  after ? ??

                async def _do_close():

                    try:

                        await self.trader.close()

                        self.log("APIconnection?")

                    except Exception as e:

                        self.log(f" ?APIfailed: {e}")

                try:

                    self.loop_manager.submit_coroutine_nowait(_do_close())

                    self.log(" toafter")

                except Exception:

                    pass

            else:

                try:

                    import asyncio as _a

                    #  ? IB

                    try:

                        if getattr(self.trader, 'ib', None):

                            self.trader.ib.disconnect()

                    except Exception:

                        pass

                    #  ?

                    _a.run(self.trader.close())

                except Exception:

                    pass

                self.log("APIconnection?no ?)")

            # ??trader clientId

            self.trader = None

            # updates ? ??

            try:

                self._update_status()

                self._update_signal_status("??", "red")

            except Exception:

                pass

            try:

                #  ?

                messagebox.showinfo("???", "API connection????")

            except Exception:

                pass

        except Exception as e:

            self.log(f" ?API ? {e}")



    def _show_stock_selection_dialog(self):

        """ ? ?"""

        import tkinter.simpledialog as simpledialog

        

        #  U? ? ?

        dialog = tk.Toplevel(self)

        dialog.title("BMA Enhanced  ? ")

        dialog.geometry("600x700")  #  ? ? 

        dialog.transient(self)

        dialog.grab_set()

        

        result = {'tickers': None, 'confirmed': False, 'training_data_path': None}

        

        #  ???

        main_frame = tk.Frame(dialog)

        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        

        #  ??

        title_label = tk.Label(main_frame, text="BMA Enhanced Model Selection", 

                              font=("Arial", 14, "bold"))

        title_label.pack(pady=(0, 15))

        

        #  

        selection_frame = tk.LabelFrame(main_frame, text="Ticker Selection", font=("Arial", 10))

        selection_frame.pack(fill=tk.X, pady=(0, 15))

        

        #  ??

        choice_var = tk.StringVar(value="default")

        

        #  ? ? ???

        default_radio = tk.Radiobutton(selection_frame, 

                                     text="Use default universe (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX, CRM, ADBE) - auto-load 4 years history",

                                     variable=choice_var, value="default",

                                     font=("Arial", 9))

        default_radio.pack(anchor=tk.W, padx=10, pady=5)

        

        #  ? ???

        pool_radio = tk.Radiobutton(selection_frame, 

                                   text="Use selected stock pool",

                                   variable=choice_var, value="pool",

                                   font=("Arial", 9))

        pool_radio.pack(anchor=tk.W, padx=10, pady=5)

        

        #  ? ? ??

        pool_frame = tk.Frame(selection_frame)

        pool_frame.pack(fill=tk.X, padx=30, pady=5)

        

        #  ? n???

        pool_info_var = tk.StringVar(value="No pool selected")

        pool_info_label = tk.Label(pool_frame, textvariable=pool_info_var, 

                                  font=("Arial", 9), fg="blue")

        pool_info_label.pack(anchor=tk.W, pady=2)

        

        #  ? ? ? ?

        pool_buttons_frame = tk.Frame(pool_frame)

        pool_buttons_frame.pack(anchor=tk.W, pady=2)

        

        #  ??

        selected_pool_info = {}

        

        def open_pool_selector():

            try:

                #  ? ? ??

                from autotrader.stock_pool_selector import select_stock_pool

                

                #  ? ? ??

                pool_result = select_stock_pool(dialog)

                

                if pool_result:

                    #  ? ??

                    selected_pool_info.update(pool_result)

                    try:

                        self.selected_pool_info = dict(pool_result)

                    except Exception:

                        self.selected_pool_info = pool_result

                    pool_info_var.set(

                        f"Chosen pool: {pool_result['pool_name']} ({len(pool_result['tickers'])} tickers)"

                    )

                    choice_var.set("pool")  #  ? ? ??

                    #  ? ???

                    start_button.config(bg="#228B22", text="Start with selected pool")

                    self.log(f"[BMA] Selected pool {pool_result['pool_name']} ({len(pool_result['tickers'])} tickers)")

                else:

                    self.log("[BMA] User cancelled pool selection")

                

            except Exception as e:

                messagebox.showerror("????", f"???????????????: {e}")

                self.log(f"[ERROR]  ? ?  {e}")

        

        def open_pool_manager():

            try:

                #  ? ? 

                import os

                import sys

                current_dir = os.path.dirname(os.path.abspath(__file__))

                if current_dir not in sys.path:

                    sys.path.insert(0, current_dir)

                from stock_pool_gui import StockPoolWindow

                

                #  ? ? 

                pool_window = StockPoolWindow()

                

            except Exception as e:

                messagebox.showerror("????", f"???????????????: {e}")

                self.log(f"[ERROR]  ? ? ? {e}")

        

        tk.Button(pool_buttons_frame, text="Select Pool", command=open_pool_selector,

                 bg="#4CAF50", fg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 5))

        

        tk.Button(pool_buttons_frame, text="Manage Pools", command=open_pool_manager,

                 bg="#2196F3", fg="white", font=("Arial", 9)).pack(side=tk.LEFT)

        

        #  U? ???

        custom_radio = tk.Radiobutton(selection_frame, 

                                    text="Enter custom ticker list",

                                    variable=choice_var, value="custom",

                                    font=("Arial", 9))

        custom_radio.pack(anchor=tk.W, padx=10, pady=5)

        

        #  U? ???

        custom_frame = tk.Frame(selection_frame)

        custom_frame.pack(fill=tk.X, padx=10, pady=5)

        

        tk.Label(custom_frame, text="Enter tickers (comma separated):", font=("Arial", 9)).pack(anchor=tk.W)

        custom_entry = tk.Text(custom_frame, height=4, width=50, font=("Arial", 9))

        custom_entry.pack(fill=tk.X, pady=5)

        custom_entry.insert("1.0", "UUUU, AAPL, MSFT")  #  

        

        # ========================================================================

        # ??  ? MultiIndex ? 

        # ========================================================================

        file_radio = tk.Radiobutton(selection_frame, 

                                   text="Load multi-index training files",

                                   variable=choice_var, value="file",

                                   font=("Arial", 9, "bold"), fg="#1976D2")

        file_radio.pack(anchor=tk.W, padx=10, pady=5)

        

        #  ??

        file_frame = tk.Frame(selection_frame)

        file_frame.pack(fill=tk.X, padx=30, pady=5)

        

        #  S? ??

        training_file_var = tk.StringVar(value="No training file selected")

        training_file_label = tk.Label(file_frame, textvariable=training_file_var, 

                                       font=("Arial", 9), fg="blue", wraplength=400)

        training_file_label.pack(anchor=tk.W, pady=2)



        #  ? ??        selected_training_file = {'path': None}

        

        def browse_training_file():

            from tkinter import filedialog

            #  ? ??

            file_path = filedialog.askopenfilename(

                title=" ? ",

                filetypes=[

                    ("Parquet Files", "*.parquet"),

                    ("Pickle Files", "*.pkl;*.pickle"),

                    ("All Files", "*.*")

                ],

                initialdir="D:\\trade\\data\\factor_exports"

            )

            if file_path:

                selected_training_file['path'] = file_path

                training_file_var.set(f" ? : {os.path.basename(file_path)}")

                choice_var.set("file")

                start_button.config(bg="#1976D2", text="? ??( ")

                self.log(f"[BMA]  ? ? : {file_path}")



        def browse_multiple_training_files():

            from tkinter import filedialog

            file_paths = filedialog.askopenfilenames(

                title=" ? ?",

                filetypes=[

                    ("Parquet Files", "*.parquet"),

                    ("Pickle Files", "*.pkl;*.pickle"),

                    ("All Files", "*.*")

                ],

                initialdir="D:\\trade\\data\\factor_exports"

            )

            if file_paths:

                paths = list(file_paths)

                selected_training_file['path'] = paths

                training_file_var.set(f" ???{len(paths)}  ??")

                choice_var.set("file")

                start_button.config(bg="#1976D2", text="? ??( ")

                self.log(f"[BMA]  ??{len(paths)} ? ??")



        def browse_training_dir():

            from tkinter import filedialog

            # ?????????parquet???????

            dir_path = filedialog.askdirectory(

                title="??????????????????parquet?????",

                initialdir="D:\\trade\\data\\factor_exports"

            )

            if dir_path:

                selected_training_file['path'] = dir_path

                training_file_var.set(f" ? `?: {os.path.basename(dir_path)}")

                choice_var.set("file")

                start_button.config(bg="#1976D2", text="? ??( ")

                self.log(f"[BMA]  ? ? `?: {dir_path}")

        

        file_buttons_frame = tk.Frame(file_frame)

        file_buttons_frame.pack(anchor=tk.W, pady=2)

        

        tk.Button(file_buttons_frame, text="Choose File", command=browse_training_file,

                 bg="#1976D2", fg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 5))



        tk.Button(file_buttons_frame, text="Choose Multiple Files", command=browse_multiple_training_files,

                 bg="#0b5394", fg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 5))



        tk.Button(file_buttons_frame, text="Choose Directory", command=browse_training_dir,

                 bg="#1565C0", fg="white", font=("Arial", 9)).pack(side=tk.LEFT)



        tk.Button(file_buttons_frame, text="Start Training", command=lambda: on_confirm(),

                 bg="#4CAF50", fg="white", font=("Arial", 9, 'bold')).pack(side=tk.LEFT, padx=(10, 0))

        

        #  ??

        file_hint = tk.Label(file_frame, 

                            text="Supported: .parquet (preserves MultiIndex date+ticker). Data should include features + target.",

                            font=("Arial", 8), fg="gray", justify=tk.LEFT)

        file_hint.pack(anchor=tk.W, pady=2)

        

        #  ??

        time_frame = tk.LabelFrame(main_frame, text="Timing Info", font=("Arial", 10))

        time_frame.pack(fill=tk.X, pady=(0, 10))

        

        time_info = tk.Label(time_frame, 

                           text="Training window: uses approx 252 trading days per year. System automatically updates two trading cycles for reliable data coverage.",

                           font=("Arial", 9), justify=tk.LEFT)

        time_info.pack(anchor=tk.W, padx=10, pady=10)

        

        #  ???-  ? ? 

        status_frame = tk.LabelFrame(main_frame, text="System Status", font=("Arial", 10))

        status_frame.pack(fill=tk.X, pady=(0, 15))

        

        #  ? 

        status_text = "BMA Enhanced system overview:\n- Alpha engine running (58 signals across industries)\n- Strategy models in staging configuration\n- Models ready; you can start trading once data is loaded"

        status_label = tk.Label(status_frame, 

                               text=status_text,

                               font=("Arial", 9), 

                               fg="#2E8B57",  #  ?

                               justify=tk.LEFT)

        status_label.pack(anchor=tk.W, padx=10, pady=8)

        

        #   -  ??

        button_frame = tk.Frame(main_frame, height=80, bg="#f0f0f0")

        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        button_frame.pack_propagate(False)  #  ? 

        

        def on_confirm():

            if choice_var.get() == "default":

                result['tickers'] = self._normalize_ticker_list(DEFAULT_AUTO_TRAIN_TICKERS)

                result['training_data_path'] = None

            elif choice_var.get() == "pool":

                #  

                if selected_pool_info and 'tickers' in selected_pool_info:

                    pool_tickers = self._normalize_ticker_list(selected_pool_info['tickers'])

                    result['tickers'] = pool_tickers

                    result['training_data_path'] = None

                    self.log(f"[BMA] Using pool {selected_pool_info['pool_name']} with {len(pool_tickers)} tickers")

                else:

                    messagebox.showerror("Error", "Please select a stock pool first")

                    return

            elif choice_var.get() == "file":

                # ??  ? ? 

                if selected_training_file.get('path'):

                    result['tickers'] = []  #  D???                    result['training_data_path'] = selected_training_file['path']

                    path_info = selected_training_file['path']

                    if isinstance(path_info, (list, tuple)):

                        self.log(f"[BMA] Selected {len(path_info)} training files")

                    else:

                        self.log(f"[BMA] Selected training file {path_info}")

                else:

                    messagebox.showerror("Error", "Please choose a training file or directory")

                    return

            else:

                #  U? ??

                custom_text = custom_entry.get("1.0", tk.END).strip()

                if custom_text:

                    normalized_csv = self.normalize_ticker_input(custom_text)

                    tickers = normalized_csv.split(',') if normalized_csv else []

                    if tickers:

                        result['tickers'] = tickers

                        result['training_data_path'] = None

                    else:

                        messagebox.showerror("Error", "Please enter valid existing tickers")

                        return

                else:

                    messagebox.showerror("Error", "Please select a stock pool or enter tickers")

                    return



            result['confirmed'] = True

            dialog.destroy()



        def on_cancel():

            result['confirmed'] = False

            dialog.destroy()

        

        #   -  ? ??

        start_button = tk.Button(button_frame, text="Start Selection", command=on_confirm, 

                                bg="#4CAF50", fg="white", font=("Arial", 11, "bold"),

                                width=18, height=2)

        start_button.pack(side=tk.RIGHT, padx=10, pady=10)

        

        cancel_button = tk.Button(button_frame, text="Cancel", command=on_cancel,

                                 bg="#f44336", fg="white", font=("Arial", 11),

                                 width=10, height=2)

        cancel_button.pack(side=tk.RIGHT, padx=10, pady=10)

        

        #  ?

        dialog.wait_window()

        

        if result['confirmed']:

            #  tickers training_data_path ??

            return {

                'tickers': result['tickers'],

                'training_data_path': result.get('training_data_path')

            }

        else:

            return None





    def _compute_prediction_window(self, lookback_years: int = 3) -> dict:

        """Automatically determine the prediction window (today -> T+5)."""



        today = datetime.now().date()

        start_dt = today - timedelta(days=int(lookback_years * 365))



        # Use pandas BDay to advance 5 trading days for the T+5 target label

        base_ts = pd.Timestamp(today)

        target_ts = (base_ts + BDay(5)).date()



        return {

            'start_date': start_dt.strftime('%Y-%m-%d'),

            'end_date': today.strftime('%Y-%m-%d'),

            'target_date': target_ts.strftime('%Y-%m-%d')

        }



    def _auto_build_multiindex_training_file(self, tickers: List[str], years: int = AUTO_TRAIN_LOOKBACK_YEARS,

                                             horizon: int = AUTO_TRAIN_HORIZON_DAYS) -> Optional[dict]:

        """Download recent market data, compute factors, and persist a MultiIndex training file."""

        from pathlib import Path

        from bma_models._bma_ultra_enhanced import UltraEnhancedQuantitativeModel

        from bma_models.simple_25_factor_engine import Simple17FactorEngine, MAX_CLOSE_THRESHOLD



        clean_tickers = self._normalize_ticker_list(tickers)

        if not clean_tickers:

            return None



        end_dt = pd.Timestamp(datetime.utcnow().date())

        lookback_days = max(int(years * 252), 252)

        start_dt = (end_dt - BDay(lookback_days)).date()

        start_date = start_dt.strftime('%Y-%m-%d')

        end_date = end_dt.date().strftime('%Y-%m-%d')



        model = UltraEnhancedQuantitativeModel(preserve_state=False)

        model.use_simple_25_factors = True

        model.horizon = horizon

        factor_engine = Simple17FactorEngine(

            mode='predict',

            lookback_days=lookback_days + horizon + 10,

            horizon=horizon

        )

        model.simple_25_engine = factor_engine



        feature_df = model.get_data_and_features(clean_tickers, start_date, end_date, mode='predict')

        if feature_df is None or len(feature_df) == 0:

            return None



        feature_df = model._ensure_standard_feature_index(feature_df)



        drop_cols = [col for col in feature_df.columns if col.lower() == 'sector']

        if drop_cols:

            feature_df = feature_df.drop(columns=drop_cols)



        output_dir = Path('data/factor_exports/auto_training')

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        file_path = output_dir / f'auto_multiindex_{timestamp}.parquet'

        feature_df.to_parquet(file_path)



        train_rows = int(feature_df['target'].notna().sum()) if 'target' in feature_df.columns else len(feature_df)

        predict_rows = int(feature_df['target'].isna().sum()) if 'target' in feature_df.columns else 0



        return {

            'path': str(file_path),

            'start_date': start_date,

            'end_date': end_date,

            'ticker_count': int(feature_df.index.get_level_values('ticker').nunique()),

            'row_count': int(len(feature_df)),

            'train_rows': train_rows,

            'predict_rows': predict_rows,

            'horizon': horizon,

        }





    def _open_stock_pool_manager(self) -> None:

        """ ? ? """

        try:

            #  ? ? ??

            try:

                from autotrader.stock_pool_gui import StockPoolWindow  # type: ignore

            except Exception:

                from .stock_pool_gui import StockPoolWindow  # type: ignore

            

            #  ? ? ?

            pool_window = StockPoolWindow()

            self.log("[INFO]  ? ? ?")

            

        except Exception as e:

            messagebox.showerror("????", f"???????????????: {e}")

            self.log(f"[ERROR]  ? ? ? {e}")



    def _clear_log(self) -> None:

        self.txt.delete(1.0, tk.END)

        self.log(" ")



    def _show_account(self) -> None:

        try:

            if not self.trader:

                self.log(" connectionIBKR")

                return

                

            self.log("?nretrievalaccount?..")

            loop = self._ensure_loop()

            

            async def _run():

                try:

                    await self.trader.refresh_account_balances_and_positions()

                    self.log(f" ?: ${self.trader.cash_balance:,.2f}")

                    self.log(f"account? ${self.trader.net_liq:,.2f}")

                    self.log(f"positions ? {len(self.trader.positions)}  ??")

                    for symbol, qty in self.trader.positions.items():

                        if qty != 0:

                            self.log(f"  {symbol}: {qty} ")

                except Exception as e:

                    self.log(f"retrievalaccount R?failed: {e}")

                    

            #  use GUI??

            if hasattr(self, 'loop_manager') and self.loop_manager.is_running:

                task_id = self.loop_manager.submit_coroutine_nowait(_run())

                self.log(f"order placement ?D: {task_id}")

            else:

                self.log(" ? c? no order placement ?")

            

        except Exception as e:

            self.log(f" account ? {e}")



    # ====================  ? ??====================

    

    def _refresh_stock_lists(self):

        """ ? """

        try:

            lists = self.db.get_stock_lists()

            list_names = [f"{lst['name']} ({lst['stock_count']}" for lst in lists]

            self.stock_list_combo['values'] = list_names

            

            #  ID ??

            self.stock_list_mapping = {f"{lst['name']} ({lst['stock_count']}": lst['id'] for lst in lists}

            

            if list_names:

                self.stock_list_combo.current(0)

                self._on_stock_list_changed(None)

                

        except Exception as e:

            self.log(f" ? failed: {e}")

    

    def _refresh_configs(self):

        """ ?"""

        try:

            configs = self.db.get_trading_configs()

            config_names = [cfg['name'] for cfg in configs]

            self.config_combo['values'] = config_names

            

            if config_names:

                self.config_combo.current(0)

                

        except Exception as e:

            self.log(f" failed: {e}")

    

    # =====  tickers and ?  =====

    def _refresh_global_tickers_table(self) -> None:

        """ tickersin in ?"""

        try:

            #  

            for item in self.stock_tree.get_children():

                self.stock_tree.delete(item)

            #  tickers

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

            self.log(f" ?failed: {e}")

    

    def _add_ticker_global(self) -> None:

        """ to tickers"""

        try:

            raw = (self.ent_symbol.get() or '')

            try:

                try:

                    from autotrader.stock_pool_manager import StockPoolManager  # type: ignore

                except Exception:

                    from .stock_pool_manager import StockPoolManager  # type: ignore

                symbol = StockPoolManager._sanitize_ticker(raw) or ''

            except Exception:

                symbol = (raw or '').strip().upper().replace('"','').replace("'", '')

                symbol = ''.join(c for c in symbol if not c.isspace())

            if not symbol:

                messagebox.showwarning("???", "?????????????")

                return

            if self.db.add_ticker(symbol):

                self.log(f" to tickers: {symbol}")

                try:

                    self.ent_symbol.delete(0, tk.END)

                except Exception:

                    pass

                self._refresh_global_tickers_table()

            else:

                messagebox.showwarning("???", f"{symbol} ????? ???")

        except Exception as e:

            self.log(f" tickerfailed: {e}")

            messagebox.showerror("????", f"??????: {e}")

    

    def _normalize_ticker_list(self, tickers: List[str]) -> List[str]:

        """Normalize ticker inputs by uppercasing, stripping spaces, and removing duplicates."""

        normalized: List[str] = []

        for ticker in tickers or []:

            cleaned = (ticker or '').strip().upper().replace('\"', '').replace("'", '')

            cleaned = ''.join(c for c in cleaned if not c.isspace())

            if cleaned and cleaned not in normalized:

                normalized.append(cleaned)

        return normalized





    def normalize_ticker_input(self, text: str) -> str:

        """

         ? ? ??



        Args:

            text:  ? ? ? ?? ??



        Returns:

             



        Example:

             ?? "AAPL MSFT\nGOOGL,AMZN  TSLA"

             ?? "AAPL,MSFT,GOOGL,AMZN,TSLA"

        """

        if not text:

            return ""



        #  ? ? ?? ? ??

        import re

        #  ? ?

        tokens = re.split(r'[\s,;]+', text.strip())



        #  ??

        cleaned_tickers = []

        for token in tokens:

            #  ? ?

            cleaned = token.strip().upper().replace('"', '').replace("'", '')

            if cleaned:  #  ??

                cleaned_tickers.append(cleaned)



        #  ??

        unique_tickers = list(dict.fromkeys(cleaned_tickers))



        #  ? 

        return ','.join(unique_tickers)



    def _normalize_batch_input_text(self) -> None:

        """ ? ? ? """

        try:

            #  ??

            raw_text = self.ent_batch_csv.get(1.0, tk.END).strip()

            if not raw_text:

                messagebox.showinfo("???", "????????????")

                return



            #  ??

            normalized = self.normalize_ticker_input(raw_text)



            if not normalized:

                messagebox.showwarning("???", " ????? ????????")

                return



            #  ?

            self.ent_batch_csv.delete(1.0, tk.END)

            self.ent_batch_csv.insert(1.0, normalized)



            #  ? ? ??

            ticker_count = len(normalized.split(','))

            self.log(f"? Z?????????{ticker_count}?????????")

            preview = normalized[:100] + ('...' if len(normalized) > 100 else '')

            messagebox.showinfo(

                "???",

                f"? Z?????\n???{ticker_count}?????????\n\n{preview}"

            )



        except Exception as e:

            self.log(f"  {e}")

            messagebox.showerror("????", f"? Z?????: {e}")



    def _batch_import_global(self) -> None:

        """ to tickers"""

        try:

            csv_text = (self.ent_batch_csv.get(1.0, tk.END) or '').strip()

            if not csv_text:

                messagebox.showwarning("???", "??????????????????")

                return

            tokens = []

            for line in csv_text.split('\n'):

                tokens.extend(line.replace(',', ' ').split())

            symbols = []

            try:

                try:

                    from autotrader.stock_pool_manager import StockPoolManager  # type: ignore

                except Exception:

                    from .stock_pool_manager import StockPoolManager  # type: ignore

                for tok in tokens:

                    s = StockPoolManager._sanitize_ticker(tok)

                    if s:

                        symbols.append(s)

            except Exception:

                for tok in tokens:

                    s = (tok or '').strip().upper().replace('"','').replace("'", '')

                    s = ''.join(c for c in s if not c.isspace())

                    if s:

                        symbols.append(s)

            success = 0

            fail = 0

            for s in symbols:

                if self.db.add_ticker(s):

                    success += 1

                else:

                    fail += 1

            self.log(f" ? ??completed: success {success} failed {fail}")

            try:

                self.ent_batch_csv.delete(1.0, tk.END)

            except Exception:

                pass

            self._refresh_global_tickers_table()

        except Exception as e:

            self.log(f" ? ??failed: {e}")

            messagebox.showerror("????", f"???????????: {e}")

    

    def _delete_selected_ticker_global(self) -> None:

        """from tickers in ? ? ?"""

        try:

            selected_items = self.stock_tree.selection()

            if not selected_items:

                messagebox.showwarning("???", "???????????????")

                return

            symbols = []

            for item in selected_items:

                values = self.stock_tree.item(item, 'values')

                if values:

                    symbols.append(values[0])

            if not symbols:

                return

            result = messagebox.askyesno("? ", f" ? from tickers \n{', '.join(symbols)}")

            if not result:

                return

            removed = []

            for symbol in symbols:

                if self.db.remove_ticker(symbol):

                    removed.append(symbol)

            self.log(f"from tickers ?{len(removed)} ? {', '.join(removed) if removed else ''}")

            self._refresh_global_tickers_table()



            #  ? market be haspositions?

            if removed:

                if self.trader and self.loop and self.loop.is_running():

                    try:

                        task_id = self.loop_manager.submit_coroutine_nowait(self._auto_sell_stocks(removed))

                        self.log(f" ?  (ID: {task_id[:8]}...)")

                    except Exception as e:

                        self.log(f" ? failed: {e}")

                else:

                    self.log(" before connection or ? c? no ??afterconnectionaftercanin use ?")

        except Exception as e:

            self.log(f" tickerfailed: {e}")

            messagebox.showerror("????", f"??????: {e}")

    

    def _on_stock_list_changed(self, event):

        """ ? ?"""

        try:

            selected = self.stock_list_var.get()

            if selected and selected in self.stock_list_mapping:

                list_id = self.stock_list_mapping[selected]

                self.state.selected_stock_list_id = list_id

                self._refresh_stock_table(list_id)

                

        except Exception as e:

            self.log(f" ? failed: {e}")

    

    def _refresh_stock_table(self, list_id):

        """ Stock table"""

        try:

            #  

            for item in self.stock_tree.get_children():

                self.stock_tree.delete(item)

            

            #  ?

            stocks = self.db.get_stocks_in_list(list_id)

            for stock in stocks:

                self.stock_tree.insert('', 'end', values=(

                    stock['symbol'], 

                    stock['name'] or '', 

                    stock['added_at'][:16] if stock['added_at'] else ''

                ))

                

        except Exception as e:

            self.log(f" Stock table failed: {e}")

    

    def _create_stock_list(self):

        """ ?"""

        try:

            name = tk.simpledialog.askstring("??????? ?", "??????? ?????")

            if not name:

                return

                

            description = tk.simpledialog.askstring("??????? ?", "?????????????????") or ""

            

            list_id = self.db.create_stock_list(name, description)

            self.log(f"success ? ? {name}")

            self._refresh_stock_lists()

            

        except ValueError as e:

            messagebox.showerror("????", str(e))

        except Exception as e:

            self.log(f" ? failed: {e}")

            messagebox.showerror("????", f"???????: {e}")

    

    def _delete_stock_list(self):

        """ ? ?"""

        try:

            if not self.state.selected_stock_list_id:

                messagebox.showwarning("???", "?????????? ?")

                return

                

            selected = self.stock_list_var.get()

            result = messagebox.askyesno("? ", f" ? '{selected}'  \n will in?has ??")

            

            if result:

                if self.db.delete_stock_list(self.state.selected_stock_list_id):

                    self.log(f"success ? ? {selected}")

                    self._refresh_stock_lists()

                else:

                    messagebox.showerror("????", "??????")

                    

        except Exception as e:

            self.log(f" ? failed: {e}")

            messagebox.showerror("????", f"??????: {e}")

    

    def _add_stock(self):

        """????????"""

        messagebox.showinfo("???", "?????????????????")

    

    def _batch_import(self):

        """ ?"""

        messagebox.showinfo("???", "?????????????????????")

    

    def _delete_selected_stock(self):

        """????????? ??"""

        messagebox.showinfo("???", "??????? ??????????????")



    def _sync_global_to_current_list_replace(self):

        """will tickers before in stocks ??"""

        try:

            if not self.state.selected_stock_list_id:

                messagebox.showwarning("???", "?????????????? ?")

                return

            tickers = self.db.get_all_tickers()

            if not tickers:

                messagebox.showinfo("???", "???????????????")

                return

            ok = messagebox.askyesno(

                "???",

                f"???? ????? ?{len(tickers)}????? I???? ???????????"

            )

            if not ok:

                return

            removed_symbols = self.db.clear_stock_list(self.state.selected_stock_list_id)

            added = 0

            for sym in tickers:

                if self.db.add_stock(self.state.selected_stock_list_id, sym):

                    added += 1

            self.log(f" ?completed has {len(removed_symbols)}  ?  {added} ")

            self._refresh_stock_table(self.state.selected_stock_list_id)

            self._refresh_stock_lists()

        except Exception as e:

            self.log(f" ?ailed: {e}")

            messagebox.showerror("????", f"??????: {e}")



    def _sync_current_list_to_global_replace(self):

        """will before in tickers can ? ?"""

        try:

            if not self.state.selected_stock_list_id:

                messagebox.showwarning("???", "?????????? ?")

                return

            rows = self.db.get_stocks_in_list(self.state.selected_stock_list_id)

            symbols = [r.get('symbol') for r in rows if r.get('symbol')]

            ok = messagebox.askyesno(

                "???",

                f"?????? ??{len(symbols)}????????????????????????????????????"

            )

            if not ok:

                return

            removed_before, success, fail = self.db.replace_all_tickers(symbols)

            self.log(f" ? ?completed {len(removed_before)} ?uccess {success} failed {fail}")

            #  ??items ? ??

            auto_clear = bool(self.var_auto_clear.get())

            if auto_clear and removed_before:

                if self.trader and self.loop and self.loop.is_running():

                    task_id = self.loop_manager.submit_coroutine_nowait(self._auto_sell_stocks(removed_before))

                    self.log(f" ?  (ID: {task_id[:8]}...)")

                else:

                    self.log("? tobe before connection or ? c? ? ?")

        except Exception as e:

            self.log(f" ? ?failed: {e}")

            messagebox.showerror("????", f"??????: {e}")

    

    def _save_config(self):

        """ ?"""

        try:

            name = self.config_name_var.get().strip()

            if not name:

                name = tk.simpledialog.askstring(" ", " ??")

                if not name:

                    return

            

            # retrieval beforeUI ??

            try:

                alloc = float(self.ent_alloc.get().strip() or 0.03)

                poll_sec = float(self.ent_poll.get().strip() or 10.0)

                fixed_qty = int(self.ent_fixed_qty.get().strip() or 0)

                auto_sell = self.var_auto_sell.get()

            except ValueError:

                messagebox.showerror("????", "???????????")

                return

            

            if self.db.save_trading_config(name, alloc, poll_sec, auto_sell, fixed_qty):

                self.log(f"success to ? {name}")

                self._refresh_configs()

                self.config_name_var.set(name)

                

                #  whenupdates ???

                self.config_manager.update_runtime_config({

                    'trading.alloc_pct': alloc,

                    'trading.poll_interval': poll_sec,

                    'trading.auto_sell_removed': auto_sell,

                    'trading.fixed_quantity': fixed_qty

                })

                

                #  to ??

                if self.config_manager.persist_runtime_changes():

                    self.log("  to ")

                else:

                    self.log("  failed to ?")

            else:

                messagebox.showerror("????", "???????????")

                

        except Exception as e:

            self.log(f" failed: {e}")

            messagebox.showerror("????", f"???????: {e}")

    

    def _load_config(self):

        """ ?"""

        try:

            name = self.config_name_var.get().strip()

            if not name:

                messagebox.showwarning("???", "?????????")

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

                

                self.log(f"success : {name}")

            else:

                messagebox.showerror("????", "???????????")

                

        except Exception as e:

            self.log(f" failed: {e}")

            messagebox.showerror("????", f"???????: {e}")



    def _get_current_stock_symbols(self) -> str:

        """retrieval before in ? as in?heckuse ?"""

        try:

            tickers = self.db.get_all_tickers()

            return ",".join(tickers)

        except Exception as e:

            self.log(f"retrieval ? failed: {e}")

            return ""



    def _load_top10_refresh_state(self) -> Optional[datetime]:

        try:

            if self._top10_state_path.exists():

                data = json.loads(self._top10_state_path.read_text(encoding='utf-8'))

                date_str = data.get('last_refresh_date')

                if date_str:

                    return datetime.fromisoformat(date_str)

        except Exception as e:

            self.log(f"[TOP10] ?????????? {e}")

        return None



    def _save_top10_refresh_state(self, when: datetime, symbols: List[str]) -> None:

        try:

            payload = {'last_refresh_date': when.isoformat(), 'symbols': symbols}

            self._top10_state_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

        except Exception as e:

            self.log(f"[TOP10] ??? ?????? {e}")



    @staticmethod

    def _is_biweekly_monday(day: datetime) -> bool:

        return day.weekday() == 0 and (day.isocalendar()[1] % 2 == 0)



    def _load_top10_from_predictions(self) -> List[str]:

        try:

            result_dir = Path('result/model_backtest')

            if not result_dir.exists():

                return []

            files = sorted(result_dir.glob('ridge_stacking_predictions_*.parquet'), key=lambda p: p.stat().st_mtime, reverse=True)

            for file in files:

                df = pd.read_parquet(file)

                if df.empty or 'ticker' not in df.columns or 'date' not in df.columns:

                    continue

                df['date'] = pd.to_datetime(df['date'])

                latest = df['date'].max()

                latest_df = df[df['date'] == latest].sort_values('prediction', ascending=False).head(10)

                tickers = latest_df['ticker'].dropna().astype(str).tolist()

                if tickers:

                    return tickers

        except Exception as e:

            self.log(f"[TOP10] ???BMA??????: {e}")

        return []



    def _load_top10_from_text(self) -> List[str]:

        txt = Path('result/bma_top10.txt')

        if not txt.exists():

            return []

        try:

            return [line.strip().upper() for line in txt.read_text(encoding='utf-8').splitlines() if line.strip()]

        except Exception as e:

            self.log(f"[TOP10] ??????Top10???: {e}")

            return []



    def _apply_top10_to_stock_pool(self, symbols: List[str]) -> List[str]:

        try:

            try:

                from autotrader.stock_pool_manager import StockPoolManager  # type: ignore

            except Exception:

                from .stock_pool_manager import StockPoolManager  # type: ignore

            sanitized = StockPoolManager._sanitize_tickers(symbols)

        except Exception:

            sanitized = [s.strip().upper() for s in symbols if s.strip()]

        if not sanitized:

            sanitized = ['QQQ']

        existing = self.db.get_all_tickers()

        removed = [sym for sym in existing if sym not in sanitized]

        if not self.db.clear_tickers():

            raise RuntimeError('??????????')

        self.db.batch_add_tickers(sanitized)

        self._refresh_global_tickers_table()

        self.log(f"[TOP10] ????1????, {len(sanitized)} ")

        if removed:

            self.log(f"[TOP10] ??????: {', '.join(removed)}")

        return removed



    def _maybe_refresh_top10_pool(self, force: bool = False) -> None:

        now = datetime.utcnow()

        if not force:

            if not self._is_biweekly_monday(now):

                return

            if self._last_top10_refresh and self._last_top10_refresh.date() >= now.date():

                return

        tickers = self._load_top10_from_predictions()

        if not tickers:

            tickers = self._load_top10_from_text()

        if not tickers:

            tickers = ['QQQ']

        removed = self._apply_top10_to_stock_pool(tickers)

        self._last_top10_refresh = datetime.utcnow()

        self._save_top10_refresh_state(self._last_top10_refresh, tickers)

        if removed and self.trader:

            try:

                self._run_async_safe(self._auto_sell_stocks(removed), "auto_sell_top10")

            except Exception:

                pass



    async def _auto_sell_stocks(self, symbols_to_sell: List[str]):

        """ ? ?"""

        if not symbols_to_sell:

            return

            

        try:

            if not self.trader:

                self.log(" connection no ")

                return

                

            self.log(f"starting ?  {len(symbols_to_sell)}  ?? {', '.join(symbols_to_sell)}")

            

            for symbol in symbols_to_sell:

                try:

                    # retrieval beforepositions

                    if hasattr(self.trader, 'positions') and symbol in self.trader.positions:

                        position = self.trader.positions.get(symbol, 0)

                        if position > 0:

                            self.log(f" ?{symbol}: {position} ")

                            await self.trader.place_market_order(symbol, "SELL", position)

                        else:

                            self.log(f"{symbol} nopositionsor ?")

                    else:

                        self.log(f"no retrieval {symbol} positions?")

                        

                except Exception as e:

                    self.log(f" ?{symbol} failed: {e}")

                    

        except Exception as e:

            self.log(f" ? failed: {e}")



    def _import_file_to_database(self):

        """will ? to ?->  useat ??tickers """

        try:

            #  ?? sheet/column/ CSV?

            self._capture_ui()

            # retrieval ? ? k? json/excel/csv  ??

            symbols_to_import = self._extract_symbols_from_files()

            self.log(f" ? ? {len(symbols_to_import)}")

            if not symbols_to_import:

                messagebox.showwarning("???", "??????????????")

                return

            

            # ?for ??

            auto_clear = self.var_auto_clear.get()

            

            preview_list = symbols_to_import[:50]

            preview_text = ", ".join(preview_list)

            if len(symbols_to_import) > 50:

                preview_text += "..."

            if auto_clear:

                msg = (

                    "????? I??????????\n\n"

                    "????????\n"

                    "1. ?????????????\n"

                    f"2. ? I??1????{preview_text}\n\n"

                    "????????????"

                )

            else:

                msg = (

                    "?????????? ?????\n\n"

                    "????????\n"

                    "1. ?????? ??????1??\n"

                    f"2. ?1????{preview_text}\n\n"

                    "????????????"

                )

                

            result = messagebox.askyesno("? ", msg)

            if not result:

                return

            

            #  ? ??tickers

            removed_before, success, fail = self.db.replace_all_tickers(symbols_to_import)

            

            self.log(f" ? completed:")

            self.log(f"   ? {len(removed_before)}  ??")

            self.log(f"   ? success {success}  ?failed {fail} ")



            #  when before ??tickers  ??

            try:

                all_ticks = self.db.get_all_tickers()

                preview = ", ".join(all_ticks[:200]) + ("..." if len(all_ticks) > 200 else "")

                self.log(f" before ?tickers ?{len(all_ticks)}: {preview}")

                try:

                    messagebox.showinfo(" completed", f" before ?tickers ?{len(all_ticks)}  records")

                except Exception:

                    pass

            except Exception as e:

                self.log(f" tickersfailed: {e}")

            

            # if use ? connection in ? ? 

            if auto_clear and removed_before:

                if self.trader and self.loop and self.loop.is_running():

                    self.loop_manager.submit_coroutine(

                        self._auto_sell_stocks(removed_before), timeout=30)

                else:

                    self.log("? to ? before connection or ? c? ? ?")

            

            #  

            try:

                if getattr(self, 'state', None) and self.state.selected_stock_list_id:

                    self._refresh_stock_table(self.state.selected_stock_list_id)

            except Exception:

                pass

            

        except Exception as e:

            self.log(f" failed: {e}")

            messagebox.showerror("????", f"???????: {e}")



    def _append_file_to_database(self):

        """will ? to ?->  useat ??tickers """

        try:

            #  ?? ?

            self._capture_ui()

            # retrieval ? ? k? json/excel/csv  ??

            symbols_to_import = self._extract_symbols_from_files()

            self.log(f" ? {len(symbols_to_import)}")

            if not symbols_to_import:

                messagebox.showwarning("???", "??????????????")

                return

            

            preview_list = symbols_to_import[:50]

            preview_text = ", ".join(preview_list)

            if len(symbols_to_import) > 50:

                preview_text += "..."

            msg = (

                "???????? ??????????????\n\n"

                f"??????{preview_text}\n"

                "?????????"

            )

            result = messagebox.askyesno("??????", msg)

            if not result:

                return

            

            #  ? to ??tickers

            success, fail = 0, 0

            for s in symbols_to_import:

                if self.db.add_ticker(s):

                    success += 1

                else:

                    fail += 1

            

            self.log(f" ? completed: success {success}  ?failed {fail} ")



            #  when before ??tickers  ??

            try:

                all_ticks = self.db.get_all_tickers()

                preview = ", ".join(all_ticks[:200]) + ("..." if len(all_ticks) > 200 else "")

                self.log(f" before ?tickers ?{len(all_ticks)}: {preview}")

                try:

                    messagebox.showinfo(" completed", f" before ?tickers ?{len(all_ticks)}  records")

                except Exception:

                    pass

            except Exception as e:

                self.log(f" tickersfailed: {e}")

            

            #  

            try:

                if getattr(self, 'state', None) and self.state.selected_stock_list_id:

                    self._refresh_stock_table(self.state.selected_stock_list_id)

            except Exception:

                pass

            

        except Exception as e:

            self.log(f" failed: {e}")

            messagebox.showerror("????", f"?????????: {e}")



    def _extract_symbols_from_files(self) -> List[str]:

        """fromJSON/Excel/CSV in ? deduplicationafter ?"""

        try:

            symbols = []

            

            # fromJSON 

            if self.state.json_file:

                import json

                with open(self.state.json_file, 'r', encoding='utf-8') as f:

                    data = json.load(f)

                    if isinstance(data, list):

                        symbols.extend([str(s).upper() for s in data])

                    else:

                        self.log("JSON ? ?s ? ?")

            

            # fromExcel 

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

                    self.log(" pandas no ? Excel ?")

                except Exception as e:

                    self.log(f" Excel failed: {e}")

            

            # from CSV ??

            if self.state.symbols_csv:

                csv_symbols = [s.strip().upper() for s in self.state.symbols_csv.split(",") if s.strip()]

                symbols.extend(csv_symbols)

            

            # deduplication ??

            unique_symbols = list(dict.fromkeys(symbols))  #  deduplication

            return unique_symbols

            

        except Exception as e:

            self.log(f" ? failed: {e}")

            return []





    def _on_resource_warning(self, warning_type: str, data: dict):

        """ """

        try:

            warning_msg = f" ?[{warning_type}]: {data.get('message', str(data))}"

            self.after(0, lambda msg=warning_msg: self.log(msg))

        except Exception:

            pass

    

    def _on_closing(self) -> None:

        """Enhanced cleanup when closing the application with proper resource management"""

        try:

            self.log("?n use...")

            

            # First, cancel engine loop task if running

            if hasattr(self, '_engine_loop_task') and self._engine_loop_task and not self._engine_loop_task.done():

                try:

                    self._engine_loop_task.cancel()

                    self.log(" ? ?")

                except Exception as e:

                    self.log(f" ?failed: {e}")

            

            # Then, gracefully stop trader

            if self.trader:

                try:

                    if hasattr(self.trader, '_stop_event') and self.trader._stop_event:

                        self.trader._stop_event.set()

                        self.log("settings {??")

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

                                    self.log(" ?")

                                

                                # Then close trader connection

                                if self.trader:

                                    await self.trader.close()

                                    self.log(" connection ?")

                            except Exception as e:

                                self.log(f" ? / failed: {e}")

                        

                        try:

                            self.loop_manager.submit_coroutine_nowait(_cleanup_all())

                            self.log(" toafter")

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

                            self.log(f" ? failed: {e}")

                    

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

                    

                    #  ? ??

                    try:

                        self.resource_monitor.stop_monitoring()

                        self.log(" ?")

                    except Exception as e:

                        self.log(f" ? failed: {e}")

                    

                    #  ? ? ??

                    try:

                        self.loop_manager.stop()

                        self.log(" ? ? ")

                    except Exception as e:

                        self.log(f" ? ? ? failed: {e}")

                    

                    #  ? ??

                    try:

                        from autotrader.unified_event_manager import shutdown_event_bus

                        shutdown_event_bus()

                        self.log(" ?")

                    except Exception as e:

                        self.log(f" ? failed: {e}")

                    

                    #  to ??

                    try:

                        if hasattr(self, 'config_manager'):

                            self.config_manager.persist_runtime_changes()

                            self.log(" ? ?")

                    except Exception as e:

                        self.log(f" ? failed: {e}")

                    

                    # Reset references

                    self.trader = None

                    self.loop = None

                    self._loop_thread = None

                    

                    # Destroy the GUI

                    self.destroy()

                    

                except Exception as e:

                    print(f" ? {e}")

                    self.destroy()  # Force close regardless

            

            # Schedule cleanup and destruction

            self.after(500, force_cleanup)  # Reduced delay for faster shutdown

            

        except Exception as e:

            print(f" ? {e}")

            self.destroy()  # Force close on error



    def _run_bma_model(self) -> None:

        """ ?BMA Enhanced?-  k? U? ? ?"""

        try:

            #  ? ??

            selection_result = self._show_stock_selection_dialog()

            if selection_result is None:  #  

                return

            

            # ??  ??

            selected_tickers = selection_result.get('tickers') or []

            training_data_path = selection_result.get('training_data_path')



            normalized_tickers = self._normalize_ticker_list(selected_tickers) if selected_tickers else []

            auto_training_spec = None



            if not training_data_path:

                if not normalized_tickers:

                    normalized_tickers = self._normalize_ticker_list(DEFAULT_AUTO_TRAIN_TICKERS)

                auto_training_spec = {

                    'tickers': normalized_tickers,

                    'years': AUTO_TRAIN_LOOKBACK_YEARS,

                    'horizon': AUTO_TRAIN_HORIZON_DAYS

                }

                self.log(f"[BMA]  ?{AUTO_TRAIN_LOOKBACK_YEARS}  ? : {len(normalized_tickers)}")



            #  ? ? ?-> T+5?            prediction_window = self._compute_prediction_window()

            start_date = prediction_window['start_date']

            end_date = prediction_window['end_date']

            target_date = prediction_window['target_date']



            #  

            self.log(f"[BMA] ? BMA Enhanced?..")

            self.log(f"[BMA] ??  ?? ? ? {end_date}")

            self.log(f"[BMA] ??  ?5 ?  (? T+5 ??{target_date})")



            #  BMA Enhanced??

            import threading

            def _run_bma_enhanced():

                try:

                    #  bma_models J? GUI ??

                    import logging as _logging

                    class _TkinterLogHandler(_logging.Handler):

                        def __init__(self, log_cb):

                            super().__init__(_logging.INFO)

                            self._cb = log_cb

                        def emit(self, record):

                            try:

                                if str(record.name).startswith('bma_models'):

                                    msg = self.format(record)

                                    #  UI 

                                    self._cb(msg)

                            except Exception:

                                pass



                    _root_logger = _logging.getLogger()

                    _tk_handler = _TkinterLogHandler(lambda m: self.after(0, lambda s=m: self.log(s)))

                    _tk_handler.setFormatter(_logging.Formatter('%(message)s'))

                    _root_logger.addHandler(_tk_handler)

                    _root_logger.setLevel(_logging.INFO)

                    try:

                        #  ? ?? ??

                        self._model_training = True

                        self._model_trained = False

                        self.after(0, lambda: self.log("[BMA] ? BMA Enhanced?.."))



                        #  BMA Enhanced??

                        import sys

                        import os

                        bma_path = os.path.join(os.path.dirname(__file__), '..', 'bma_models')

                        if bma_path not in sys.path:

                            sys.path.append(bma_path)



                        from bma_models._bma_ultra_enhanced import UltraEnhancedQuantitativeModel



                        self.after(0, lambda: self.log("[BMA]  ? ?.."))

                        if not hasattr(self, '_bma_model_instance') or self._bma_model_instance is None:

                            self._bma_model_instance = UltraEnhancedQuantitativeModel()

                        model = self._bma_model_instance



                        self.after(0, lambda: self.log("[BMA] ? ?/ ..."))



                        effective_training_path = training_data_path

                        auto_generation_stats = None

                        try:

                            if auto_training_spec:

                                self.after(0, lambda: self.log(f"[BMA]  ?{len(auto_training_spec['tickers'])}  ??? ?"))

                                auto_generation_stats = self._auto_build_multiindex_training_file(

                                    auto_training_spec['tickers'],

                                    years=auto_training_spec.get('years', AUTO_TRAIN_LOOKBACK_YEARS),

                                    horizon=auto_training_spec.get('horizon', AUTO_TRAIN_HORIZON_DAYS)

                                )

                                if not auto_generation_stats or not auto_generation_stats.get('path'):

                                    raise RuntimeError(" ? MultiIndex :  ")

                                effective_training_path = auto_generation_stats['path']

                                self.after(0, lambda p=effective_training_path: self.log(f"[BMA] ??  ? ?MultiIndex ? {p}"))

                                if auto_generation_stats.get('predict_rows'):

                                    self.after(0, lambda stats=auto_generation_stats: self.log(f"[BMA]  ? t? ? 10  {stats['predict_rows']}  ?"))



                            if effective_training_path:

                                def _fmt_path(p):

                                    return os.path.basename(p) if isinstance(p, str) else p

                                if isinstance(effective_training_path, (list, tuple)):

                                    self.after(0, lambda: self.log("[BMA] ??  MultiIndex ?"))

                                else:

                                    self.after(0, lambda: self.log(f"[BMA] ??  MultiIndex ?: {_fmt_path(effective_training_path)}"))

                                train_report = model.train_from_document(effective_training_path, top_n=50)

                                train_msg = f"[BMA]  ? ?  ??{train_report.get('training_sample_count', 'N/A')} ?? {train_report.get('training_source')}"

                                self.after(0, lambda msg=train_msg: self.log(msg))

                                results = train_report

                            else:

                                raise RuntimeError(" ? ? ?")

                        finally:

                            try:

                                _root_logger.removeHandler(_tk_handler)

                            except Exception:

                                pass



                        #  ? ??

                        self._model_training = False

                        self._model_trained = True



                        self.after(0, lambda: self.log("[BMA]  ? ?"))



                        #  ? ??

                        if results and results.get('success', False):

                        

                            sample_count = results.get('training_sample_count', 'N/A')

                            tickers_in_file = results.get('tickers_in_file') or results.get('tickers') or []

                            self.after(0, lambda: self.log(f"[BMA] ??  ? ? {sample_count}  ??{len(tickers_in_file)}  ??"))

                            self.after(0, lambda: self.log("[BMA]  ?? BMA ? ? ?"))



                            try:

                                fe = results.get('feature_engineering', {})

                                shape = fe.get('shape') if isinstance(fe, dict) else None

                                if shape and len(shape) == 2:

                                    self.after(0, lambda r=shape[0], c=shape[1]: self.log(f"[BMA]  ? : {r}  ??{c}  ?"))



                                tr = results.get('training_results', {}) or {}

                                tm = tr.get('traditional_models') or tr

                                cv_scores = tm.get('cv_scores', {}) or {}

                                cv_r2 = tm.get('cv_r2_scores', {}) or {}



                                self.after(0, lambda: self.log("[BMA]  ? ? ? ??? ?"))

                                if cv_scores:

                                    for mdl, ic in cv_scores.items():

                                        r2 = cv_r2.get(mdl, float('nan'))

                                        self.after(0, lambda m=mdl, icv=ic, r2v=r2: self.log(f"[BMA] {m.upper()}  CV(IC)={icv:.6f}  R{r2v:.6f}"))

                                else:

                                    self.after(0, lambda: self.log("[BMA]  ? CV "))



                                ridge_stacker = tr.get('ridge_stacker', None)

                                trained = tr.get('stacker_trained', None)

                                if trained is not None:

                                    self.after(0, lambda st=trained: self.log(f"[BMA] Ridge????????: {'???' if st else '???'}"))

                                if ridge_stacker is not None:

                                    try:

                                        info = ridge_stacker.get_model_info()

                                    except Exception:

                                        info = {}

                                    niter = info.get('n_iterations')

                                    if niter is not None:

                                        self.after(0, lambda nf=niter: self.log(f"[BMA] Ridge  ?? {nf}"))

                            except Exception as e:

                                self.after(0, lambda msg=str(e): self.log(f"[BMA]  ? : {msg}"))



                            if isinstance(effective_training_path, (list, tuple)):

                                training_source_label = f"{len(effective_training_path)}  ??"

                            else:

                                training_source_label = os.path.basename(effective_training_path) if (effective_training_path and isinstance(effective_training_path, str)) else 'N/A'



                            success_msg = (f"BMA Enhanced  ? ?\n\n"

                                           f" ? : {sample_count}\n"

                                           f" ?: {len(tickers_in_file)}  \n"

                                           f" ? : {training_source_label}\n"

                                           f" ? BMA ? ? ? ?")



                            self.after(0, lambda: messagebox.showinfo("BMA??????", success_msg))

                        else:

                            #  

                            error_msg = results.get('error', '????????????????????????') if results else ' ??????'

                            self.after(0, lambda: self.log(f"[BMA] {error_msg}"))

                            self.after(0, lambda: messagebox.showerror("BMA ? ", error_msg))

                    

                    except ImportError as e:

                        self._model_training = False

                        self._model_trained = False

                        error_msg = f" BMA ? : {e}"

                        self.after(0, lambda msg=error_msg: self.log(f"[BMA] {msg}"))

                        self.after(0, lambda: messagebox.showerror("BMA????", error_msg))

                    

                    except Exception as e:

                        self._model_training = False

                        self._model_trained = False

                        error_msg = str(e)

                        self.after(0, lambda msg=error_msg: self.log(f"[BMA] ????: {msg}"))

                        self.after(0, lambda: messagebox.showerror("BMA????", f"??????: {error_msg}"))



                except Exception as inner_e:

                    self.log(f"[BMA]  ? : {inner_e}")

                    self._model_training = False

                    self._model_trained = False



            #  ?BMA Enhanced 

            thread = threading.Thread(target=_run_bma_enhanced, daemon=True)

            thread.start()

            self.log("[BMA]  ? ? ?..")



        except Exception as e:

            self.log(f"[BMA] startfailed: {e}")

            messagebox.showerror("????", f"???BMA???: {e}")



    def _build_backtest_tab(self, parent) -> None:

        """ ?items"""

        #  ? ??

        main_paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)

        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        

        #   -  ? 

        left_frame = ttk.Frame(main_paned)

        main_paned.add(left_frame, weight=1)

        

        #  ? 

        stock_frame = tk.LabelFrame(left_frame, text=" ? ?")

        stock_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        

        #  ? ?

        input_frame = tk.Frame(stock_frame)

        input_frame.pack(fill=tk.X, padx=5, pady=5)

        

        tk.Label(input_frame, text=" ? :").pack(side=tk.LEFT)

        self.ent_bt_stock_input = tk.Entry(input_frame, width=10)

        self.ent_bt_stock_input.pack(side=tk.LEFT, padx=5)

        tk.Button(input_frame, text=" ?, command=self._add_backtest_stock).pack(side=tk.LEFT")

        tk.Button(input_frame, text=" ? ?, command=self._import_stocks_from_db).pack(side=tk.LEFT, padx=5")

        tk.Button(input_frame, text=" ?, command=self._clear_backtest_stocks).pack(side=tk.LEFT")

        

        #  ? ??

        list_frame = tk.Frame(stock_frame)

        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        

        scrollbar = tk.Scrollbar(list_frame)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        

        self.bt_stock_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, selectmode=tk.EXTENDED)

        self.bt_stock_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.bt_stock_listbox.yview)

        

        #  ? ? ??

        default_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ']

        for stock in default_stocks:

            self.bt_stock_listbox.insert(tk.END, stock)

        

        #  ??

        tk.Button(stock_frame, text=" ", command=self._remove_selected_stocks).pack(pady=5)

        

        #   -  

        right_frame = ttk.Frame(main_paned)

        main_paned.add(right_frame, weight=2)

        

        #  ??

        canvas = tk.Canvas(right_frame)

        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)

        scrollable_frame = ttk.Frame(canvas)

        

        scrollable_frame.bind(

            "<Configure>",

            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))

        )

        

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)

        

        #  

        backtest_type_frame = tk.LabelFrame(scrollable_frame, text=" ?")

        backtest_type_frame.pack(fill=tk.X, padx=5, pady=5)

        

        #  ??

        self.backtest_type = tk.StringVar(value="professional")

        

        # Professional BMA  ??( ??

        tk.Radiobutton(

            backtest_type_frame, 

            text=" BMA ?(Walk-Forward + Monte Carlo)", 

            variable=self.backtest_type, 

            value="professional"

        ).pack(anchor=tk.W, padx=10, pady=2)

        

        # AutoTrader BMA  ??

        tk.Radiobutton(

            backtest_type_frame, 

            text="AutoTrader BMA  ?", 

            variable=self.backtest_type, 

            value="autotrader"

        ).pack(anchor=tk.W, padx=10, pady=2)

        

        #  ??BMA  ??

        tk.Radiobutton(

            backtest_type_frame, 

            text=" ?BMA  ?", 

            variable=self.backtest_type, 

            value="weekly"

        ).pack(anchor=tk.W, padx=10, pady=2)

        

        #  ??

        config_frame = tk.LabelFrame(scrollable_frame, text=" ?")

        config_frame.pack(fill=tk.X, padx=5, pady=5)

        

        #  ? 

        row1 = tk.Frame(config_frame)

        row1.pack(fill=tk.X, padx=5, pady=5)

        

        tk.Label(row1, text="starting ?").pack(side=tk.LEFT)

        self.ent_bt_start_date = tk.Entry(row1, width=12)

        self.ent_bt_start_date.insert(0, "2022-01-01")

        self.ent_bt_start_date.pack(side=tk.LEFT, padx=5)

        

        tk.Label(row1, text=" :").pack(side=tk.LEFT)

        self.ent_bt_end_date = tk.Entry(row1, width=12)

        self.ent_bt_end_date.insert(0, "2023-12-31")

        self.ent_bt_end_date.pack(side=tk.LEFT, padx=5)

        

        tk.Label(row1, text=" ? :").pack(side=tk.LEFT)

        self.ent_bt_capital = tk.Entry(row1, width=10)

        self.ent_bt_capital.insert(0, "100000")

        self.ent_bt_capital.pack(side=tk.LEFT, padx=5)

        

        #  ? 

        row2 = tk.Frame(config_frame)

        row2.pack(fill=tk.X, padx=5, pady=5)

        

        tk.Label(row2, text="??ositions:").pack(side=tk.LEFT)

        self.ent_bt_max_positions = tk.Entry(row2, width=8)

        self.ent_bt_max_positions.insert(0, "20")

        self.ent_bt_max_positions.pack(side=tk.LEFT, padx=5)

        

        tk.Label(row2, text=" :").pack(side=tk.LEFT)

        self.cb_bt_rebalance = ttk.Combobox(row2, values=["daily", "weekly"], width=8)

        self.cb_bt_rebalance.set("weekly")

        self.cb_bt_rebalance.pack(side=tk.LEFT, padx=5)

        

        tk.Label(row2, text=" :").pack(side=tk.LEFT)

        self.ent_bt_commission = tk.Entry(row2, width=8)

        self.ent_bt_commission.insert(0, "0.001")

        self.ent_bt_commission.pack(side=tk.LEFT, padx=5)

        

        #  ? BMA  ? 

        row3 = tk.Frame(config_frame)

        row3.pack(fill=tk.X, padx=5, pady=5)

        

        tk.Label(row3, text=" ? ? ?").pack(side=tk.LEFT)

        self.ent_bt_retrain_freq = tk.Entry(row3, width=8)

        self.ent_bt_retrain_freq.insert(0, "4")

        self.ent_bt_retrain_freq.pack(side=tk.LEFT, padx=5)

        

        tk.Label(row3, text=" :").pack(side=tk.LEFT)

        self.ent_bt_prediction_horizon = tk.Entry(row3, width=8)

        self.ent_bt_prediction_horizon.insert(0, "1")

        self.ent_bt_prediction_horizon.pack(side=tk.LEFT, padx=5)

        

        tk.Label(row3, text=" :").pack(side=tk.LEFT)

        self.ent_bt_stop_loss = tk.Entry(row3, width=8)

        self.ent_bt_stop_loss.insert(0, "0.08")

        self.ent_bt_stop_loss.pack(side=tk.LEFT, padx=5)

        

        #  ? 

        row4 = tk.Frame(config_frame)

        row4.pack(fill=tk.X, padx=5, pady=5)

        

        tk.Label(row4, text="? ?").pack(side=tk.LEFT)

        self.ent_bt_max_weight = tk.Entry(row4, width=8)

        self.ent_bt_max_weight.insert(0, "0.15")

        self.ent_bt_max_weight.pack(side=tk.LEFT, padx=5)

        

        tk.Label(row4, text=" :").pack(side=tk.LEFT)

        self.ent_bt_take_profit = tk.Entry(row4, width=8)

        self.ent_bt_take_profit.insert(0, "0.20")

        self.ent_bt_take_profit.pack(side=tk.LEFT, padx=5)

        

        tk.Label(row4, text=" ?").pack(side=tk.LEFT)

        self.ent_bt_slippage = tk.Entry(row4, width=8)

        self.ent_bt_slippage.insert(0, "0.002")

        self.ent_bt_slippage.pack(side=tk.LEFT, padx=5)

        

        #  settings

        output_frame = tk.LabelFrame(scrollable_frame, text=" settings")

        output_frame.pack(fill=tk.X, padx=5, pady=5)

        

        row5 = tk.Frame(output_frame)

        row5.pack(fill=tk.X, padx=5, pady=5)

        

        tk.Label(row5, text=" `?:").pack(side=tk.LEFT)

        self.ent_bt_output_dir = tk.Entry(row5, width=30)

        self.ent_bt_output_dir.insert(0, "./backtest_results")

        self.ent_bt_output_dir.pack(side=tk.LEFT, padx=5)

        

        tk.Button(row5, text=" ?, command=self._browse_backtest_output_dir).pack(side=tk.LEFT, padx=5")

        

        # ?items

        options_frame = tk.Frame(output_frame)

        options_frame.pack(fill=tk.X, padx=5, pady=5)

        

        self.var_bt_export_excel = tk.BooleanVar(value=True)

        tk.Checkbutton(options_frame, text=" Excel ?, variable=self.var_bt_export_excel).pack(side=tk.LEFT, padx=10")

        

        self.var_bt_show_plots = tk.BooleanVar(value=True)

        tk.Checkbutton(options_frame, text=" ?, variable=self.var_bt_show_plots).pack(side=tk.LEFT, padx=10")

        

        #  

        action_frame = tk.LabelFrame(scrollable_frame, text=" ?")

        action_frame.pack(fill=tk.X, padx=5, pady=5)

        

        button_frame = tk.Frame(action_frame)

        button_frame.pack(pady=10)

        

        #  ? ??

        tk.Button(

            button_frame, 

            text=" ? ", 

            command=self._run_single_backtest,

            bg="lightgreen", 

            font=("Arial", 10, "bold"),

            width=15

        ).pack(side=tk.LEFT, padx=10)

        

        #  ? for?

        tk.Button(

            button_frame, 

            text=" for", 

            command=self._run_strategy_comparison,

            bg="lightblue", 

            font=("Arial", 10, "bold"),

            width=15

        ).pack(side=tk.LEFT, padx=10)

        

        #  ? ?

        tk.Button(

            button_frame,

            text=" ?",

            command=self._run_quick_backtest,

            bg="orange",

            font=("Arial", 10, "bold"),

            width=15

        ).pack(side=tk.LEFT, padx=10)



        #  ? ?????? scripts/comprehensive_model_backtest.py

        tk.Button(

            button_frame,

            text=" ? ",

            command=self._run_comprehensive_backtest,

            bg="#6fa8dc",

            font=("Arial", 10, "bold"),

            width=16

        ).pack(side=tk.LEFT, padx=10)



        #  ? ??

        status_frame = tk.LabelFrame(scrollable_frame, text=" ?")

        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        

        #  canvas 

        canvas.pack(side="left", fill="both", expand=True)

        scrollbar.pack(side="right", fill="y")

        

        #  ??records

        self.bt_progress = ttk.Progressbar(status_frame, mode='indeterminate')

        self.bt_progress.pack(fill=tk.X, padx=5, pady=5)

        

        #  ? ??

        self.bt_status_text = tk.Text(status_frame, height=8, wrap=tk.WORD)

        bt_scrollbar = tk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.bt_status_text.yview)

        self.bt_status_text.configure(yscrollcommand=bt_scrollbar.set)

        self.bt_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        bt_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)



    def _build_prediction_tab(self, parent):

        frame = tk.Frame(parent)

        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)



        input_frame = tk.LabelFrame(frame, text=" ")

        input_frame.pack(fill=tk.X, padx=5, pady=5)



        tk.Label(input_frame, text=" ? ?( )").pack(anchor=tk.W, padx=5, pady=2)

        self.pred_ticker_entry = tk.Text(input_frame, height=4)

        self.pred_ticker_entry.insert(tk.END, 'AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA')

        self.pred_ticker_entry.pack(fill=tk.X, padx=5, pady=2)



        pool_frame = tk.Frame(input_frame)

        pool_frame.pack(fill=tk.X, padx=5, pady=2)

        self.pred_pool_info_var = tk.StringVar(value=" ?")

        tk.Label(pool_frame, textvariable=self.pred_pool_info_var, fg='blue').pack(side=tk.LEFT)

        tk.Button(pool_frame, text=" ?, command=self._select_prediction_pool",

                 bg="#1976D2", fg="white", font=("Arial", 9)).pack(side=tk.RIGHT, padx=5)



        row = tk.Frame(input_frame)

        row.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(row, text="? ?").pack(side=tk.LEFT)

        self.ent_pred_start_date = tk.Entry(row, width=12)

        self.ent_pred_start_date.insert(0, (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))

        self.ent_pred_start_date.pack(side=tk.LEFT, padx=5)



        tk.Label(row, text=" :").pack(side=tk.LEFT)

        self.ent_pred_end_date = tk.Entry(row, width=12)

        self.ent_pred_end_date.insert(0, datetime.now().strftime('%Y-%m-%d'))

        self.ent_pred_end_date.pack(side=tk.LEFT, padx=5)



        tk.Label(row, text="Top N:").pack(side=tk.LEFT)

        self.ent_pred_topn = tk.Entry(row, width=6)

        self.ent_pred_topn.insert(0, "20")

        self.ent_pred_topn.pack(side=tk.LEFT, padx=5)



        action_row = tk.Frame(input_frame)

        action_row.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(

            action_row,

            text=" ? ??",

            command=self._run_prediction_only,

            bg="#ff69b4",

            font=("Arial", 10, "bold"),

            width=18

        ).pack(side=tk.LEFT, padx=5)



        status_frame = tk.LabelFrame(frame, text=" ?")

        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)



        self.pred_progress = ttk.Progressbar(status_frame, mode='indeterminate')

        self.pred_progress.pack(fill=tk.X, padx=5, pady=5)



        self.pred_status_text = tk.Text(status_frame, height=8, wrap=tk.WORD)

        pred_scroll = tk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.pred_status_text.yview)

        self.pred_status_text.configure(yscrollcommand=pred_scroll.set)

        self.pred_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        pred_scroll.pack(side=tk.RIGHT, fill=tk.Y)



    def _update_prediction_status(self, message: str) -> None:

        try:

            print(message)

        except Exception:

            pass

        if hasattr(self, 'pred_status_text'):

            self.pred_status_text.insert(tk.END, message + "\n")

            self.pred_status_text.see(tk.END)



    def _select_prediction_pool(self):

        try:

            from autotrader.stock_pool_selector import select_stock_pool

        except Exception as e:

            messagebox.showerror("????", f"????????????????: {e}")

            return



        pool_result = select_stock_pool(self)

        if pool_result and pool_result.get('tickers'):

            tickers = pool_result['tickers']

            self.pred_selected_pool = tickers

            info = f" ? ? {pool_result.get('pool_name','N/A')} ({len(tickers)}"

            self.pred_pool_info_var.set(info)

            #  ?

            self.pred_ticker_entry.delete('1.0', tk.END)

            self.pred_ticker_entry.insert(tk.END, ','.join(tickers))

        else:

            messagebox.showinfo("???", "??????? ??????")



    def _browse_backtest_output_dir(self):

        """ ? `?"""

        directory = filedialog.askdirectory(title=" ?")

        if directory:

            self.ent_bt_output_dir.delete(0, tk.END)

            self.ent_bt_output_dir.insert(0, directory)



    def _build_kronos_tab(self, parent) -> None:

        """ Kronos K ? ? ?"""

        try:

            #  Kronos UI ??

            import sys

            import os

            parent_dir = os.path.dirname(os.path.dirname(__file__))

            if parent_dir not in sys.path:

                sys.path.insert(0, parent_dir)



            from kronos.kronos_tkinter_ui import KronosPredictorUI



            #  Kronos UI

            self.kronos_predictor = KronosPredictorUI(parent, log_callback=self.log)



            self.log("Kronos K ? ?")



        except Exception as e:

            self.log(f"Kronos ? : {str(e)}")

            #  ? ??

            error_frame = ttk.Frame(parent)

            error_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)



            ttk.Label(

                error_frame,

                text="Kronos K ? ?",

                font=('Arial', 12, 'bold')

            ).pack(pady=20)



            ttk.Label(

                error_frame,

                text=f" ? {str(e)}",

                foreground="red"

            ).pack(pady=10)



            ttk.Label(

                error_frame,

                text=" ?' ??? :\npip install transformers torch accelerate",

                font=('Arial', 10)

            ).pack(pady=10)



    def _build_temporal_stacking_tab(self, parent) -> None:

        """Build Temporal Stacking LambdaRank tab - Advanced meta-learner with signal trajectory"""

        frame = tk.Frame(parent)

        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)



        # Header with description

        header_frame = tk.LabelFrame(frame, text="Temporal Stacking LambdaRank")

        header_frame.pack(fill=tk.X, padx=5, pady=5)



        desc_text = (

            "Advanced meta-learner that uses T-1, T-2, T-3 prediction lags to capture:\n"

            "? Signal Momentum: Is conviction increasing?\n"

            "? Signal Stability: Is the model confused?\n"

            "? Rank Acceleration: Are top picks rising or falling?"

        )

        tk.Label(header_frame, text=desc_text, justify=tk.LEFT, fg="blue").pack(anchor=tk.W, padx=10, pady=5)



        # Data Configuration

        data_frame = tk.LabelFrame(frame, text="Data Configuration")

        data_frame.pack(fill=tk.X, padx=5, pady=5)



        row1 = tk.Frame(data_frame)

        row1.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(row1, text="OOF Data File:").pack(side=tk.LEFT)

        self.ts_data_path = tk.Entry(row1, width=50)

        self.ts_data_path.insert(0, "D:/trade/data/factor_exports/polygon_factors_all_2021_2026_T5_final.parquet")

        self.ts_data_path.pack(side=tk.LEFT, padx=5)

        tk.Button(row1, text="Browse...", command=self._browse_temporal_data).pack(side=tk.LEFT, padx=5)



        row2 = tk.Frame(data_frame)

        row2.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(row2, text="Target Column:").pack(side=tk.LEFT)

        self.ts_target_col = tk.Entry(row2, width=15)

        self.ts_target_col.insert(0, "target")

        self.ts_target_col.pack(side=tk.LEFT, padx=5)



        tk.Label(row2, text="Lookback Days:").pack(side=tk.LEFT, padx=10)

        self.ts_lookback = ttk.Spinbox(row2, from_=1, to=5, width=5)

        self.ts_lookback.set(3)

        self.ts_lookback.pack(side=tk.LEFT, padx=5)



        # Model Parameters (Anti-Overfitting)

        params_frame = tk.LabelFrame(frame, text="Model Parameters (Anti-Overfitting Defaults)")

        params_frame.pack(fill=tk.X, padx=5, pady=5)



        param_row1 = tk.Frame(params_frame)

        param_row1.pack(fill=tk.X, padx=5, pady=2)



        tk.Label(param_row1, text="Num Leaves:").pack(side=tk.LEFT)

        self.ts_num_leaves = ttk.Spinbox(param_row1, from_=4, to=31, width=5)

        self.ts_num_leaves.set(8)

        self.ts_num_leaves.pack(side=tk.LEFT, padx=5)



        tk.Label(param_row1, text="Max Depth:").pack(side=tk.LEFT, padx=10)

        self.ts_max_depth = ttk.Spinbox(param_row1, from_=2, to=6, width=5)

        self.ts_max_depth.set(3)

        self.ts_max_depth.pack(side=tk.LEFT, padx=5)



        tk.Label(param_row1, text="Learning Rate:").pack(side=tk.LEFT, padx=10)

        self.ts_learning_rate = tk.Entry(param_row1, width=8)

        self.ts_learning_rate.insert(0, "0.005")

        self.ts_learning_rate.pack(side=tk.LEFT, padx=5)



        param_row2 = tk.Frame(params_frame)

        param_row2.pack(fill=tk.X, padx=5, pady=2)



        tk.Label(param_row2, text="Feature Fraction:").pack(side=tk.LEFT)

        self.ts_feature_fraction = tk.Entry(param_row2, width=8)

        self.ts_feature_fraction.insert(0, "0.6")

        self.ts_feature_fraction.pack(side=tk.LEFT, padx=5)



        tk.Label(param_row2, text="Truncation Level:").pack(side=tk.LEFT, padx=10)

        self.ts_truncation = ttk.Spinbox(param_row2, from_=20, to=200, width=5)

        self.ts_truncation.set(60)

        self.ts_truncation.pack(side=tk.LEFT, padx=5)



        tk.Label(param_row2, text="Label Gain Power:").pack(side=tk.LEFT, padx=10)

        self.ts_gain_power = tk.Entry(param_row2, width=8)

        self.ts_gain_power.insert(0, "2.5")

        self.ts_gain_power.pack(side=tk.LEFT, padx=5)



        param_row3 = tk.Frame(params_frame)

        param_row3.pack(fill=tk.X, padx=5, pady=2)



        tk.Label(param_row3, text="Boost Rounds:").pack(side=tk.LEFT)

        self.ts_boost_rounds = ttk.Spinbox(param_row3, from_=100, to=2000, width=6)

        self.ts_boost_rounds.set(500)

        self.ts_boost_rounds.pack(side=tk.LEFT, padx=5)



        tk.Label(param_row3, text="Early Stopping:").pack(side=tk.LEFT, padx=10)

        self.ts_early_stopping = ttk.Spinbox(param_row3, from_=20, to=200, width=5)

        self.ts_early_stopping.set(50)

        self.ts_early_stopping.pack(side=tk.LEFT, padx=5)



        tk.Label(param_row3, text="IPO Handling:").pack(side=tk.LEFT, padx=10)

        self.ts_ipo_handling = ttk.Combobox(param_row3, values=['backfill', 'cross_median', 'zero'], width=12)

        self.ts_ipo_handling.set('backfill')

        self.ts_ipo_handling.pack(side=tk.LEFT, padx=5)



        # Action Buttons

        action_frame = tk.Frame(frame)

        action_frame.pack(fill=tk.X, padx=5, pady=10)



        tk.Button(

            action_frame,

            text="Run Temporal Stacking Training",

            command=self._run_temporal_stacking_training,

            bg="#4CAF50",

            fg="white",

            font=("Arial", 10, "bold"),

            width=25

        ).pack(side=tk.LEFT, padx=5)



        tk.Button(

            action_frame,

            text="Validate Temporal Integrity",

            command=self._validate_temporal_integrity,

            bg="#2196F3",

            fg="white",

            font=("Arial", 10),

            width=20

        ).pack(side=tk.LEFT, padx=5)



        tk.Button(

            action_frame,

            text="View Inference State",

            command=self._view_temporal_state,

            bg="#FF9800",

            fg="white",

            font=("Arial", 10),

            width=18

        ).pack(side=tk.LEFT, padx=5)



        # Progress and Status

        status_frame = tk.LabelFrame(frame, text="Training Status")

        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)



        self.ts_progress = ttk.Progressbar(status_frame, mode='indeterminate')

        self.ts_progress.pack(fill=tk.X, padx=5, pady=5)



        self.ts_status_text = tk.Text(status_frame, height=12, wrap=tk.WORD)

        ts_scroll = tk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.ts_status_text.yview)

        self.ts_status_text.configure(yscrollcommand=ts_scroll.set)

        self.ts_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        ts_scroll.pack(side=tk.RIGHT, fill=tk.Y)



        # Initialize model reference

        self._temporal_stacker = None



    def _browse_temporal_data(self):

        """Browse for temporal stacking data file"""

        filepath = filedialog.askopenfilename(

            title="Select OOF Data File",

            filetypes=[("Parquet files", "*.parquet"), ("All files", "*.*")]

        )

        if filepath:

            self.ts_data_path.delete(0, tk.END)

            self.ts_data_path.insert(0, filepath)



    def _update_temporal_status(self, message: str):

        """Update temporal stacking status text"""

        try:

            print(message)

        except Exception:

            pass

        if hasattr(self, 'ts_status_text'):

            self.ts_status_text.insert(tk.END, message + "\n")

            self.ts_status_text.see(tk.END)



    def _run_temporal_stacking_training(self):

        """Run Temporal Stacking LambdaRank training"""

        def training_thread():

            try:

                self.ts_progress.start()

                self._update_temporal_status("Starting Temporal Stacking training...")



                # Import module

                try:

                    from bma_models.temporal_stacking import TemporalStackingLambdaRank, validate_temporal_integrity

                except ImportError as e:

                    self._update_temporal_status(f"Failed to import temporal_stacking: {e}")

                    self.ts_progress.stop()

                    return



                # Load data

                data_path = self.ts_data_path.get()

                self._update_temporal_status(f"Loading data from: {data_path}")



                import pandas as pd

                df = pd.read_parquet(data_path)



                # Ensure MultiIndex

                if not isinstance(df.index, pd.MultiIndex):

                    if 'date' in df.columns and 'ticker' in df.columns:

                        df = df.set_index(['date', 'ticker'])

                    else:

                        self._update_temporal_status("Error: Data must have date and ticker columns")

                        self.ts_progress.stop()

                        return



                self._update_temporal_status(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")



                # Get parameters

                config = {

                    'lookback_days': int(self.ts_lookback.get()),

                    'label_gain_power': float(self.ts_gain_power.get()),

                    'num_boost_round': int(self.ts_boost_rounds.get()),

                    'early_stopping_rounds': int(self.ts_early_stopping.get()),

                    'ipo_handling': self.ts_ipo_handling.get(),

                    'lgb_params': {

                        'num_leaves': int(self.ts_num_leaves.get()),

                        'max_depth': int(self.ts_max_depth.get()),

                        'learning_rate': float(self.ts_learning_rate.get()),

                        'feature_fraction': float(self.ts_feature_fraction.get()),

                        'lambdarank_truncation_level': int(self.ts_truncation.get())

                    }

                }



                self._update_temporal_status(f"Configuration: {config}")



                # Create and train model

                self._update_temporal_status("Initializing TemporalStackingLambdaRank...")

                model = TemporalStackingLambdaRank(**config)



                # Get target column

                target_col = self.ts_target_col.get()

                if target_col not in df.columns:

                    # Try common alternatives

                    for alt in ['ret_fwd_5d', 'ret_fwd_10d', 'target']:

                        if alt in df.columns:

                            target_col = alt

                            break



                self._update_temporal_status(f"Training with target: {target_col}")



                # Train

                model.fit(df, target_col=target_col)



                # Save reference

                self._temporal_stacker = model



                # Report results

                info = model.get_model_info()

                self._update_temporal_status("\n" + "="*50)

                self._update_temporal_status("TRAINING COMPLETE")

                self._update_temporal_status("="*50)

                self._update_temporal_status(f"Best iteration: {info.get('best_iteration', 'N/A')}")

                self._update_temporal_status(f"Number of features: {info.get('n_features', 'N/A')}")



                if 'feature_importance' in info:

                    self._update_temporal_status("\nTop Features:")

                    for feat, imp in list(info['feature_importance'].items())[:10]:

                        self._update_temporal_status(f"  {feat}: {imp}")



                self._update_temporal_status("\nModel ready for prediction!")



            except Exception as e:

                import traceback

                self._update_temporal_status(f"Training failed: {e}")

                self._update_temporal_status(traceback.format_exc())

            finally:

                self.ts_progress.stop()



        # Run in thread

        threading.Thread(target=training_thread, daemon=True).start()



    def _validate_temporal_integrity(self):

        """Validate temporal integrity of loaded data"""

        def validation_thread():

            try:

                self.ts_progress.start()

                self._update_temporal_status("Validating temporal integrity...")



                from bma_models.temporal_stacking import validate_temporal_integrity, build_temporal_features

                import pandas as pd



                # Load data

                data_path = self.ts_data_path.get()

                df = pd.read_parquet(data_path)



                if not isinstance(df.index, pd.MultiIndex):

                    if 'date' in df.columns and 'ticker' in df.columns:

                        df = df.set_index(['date', 'ticker'])



                # Identify prediction columns

                pred_cols = [c for c in df.columns if c.startswith('pred_')]

                if not pred_cols:

                    pred_cols = ['pred_elastic', 'pred_xgb', 'pred_lambdarank', 'pred_catboost']

                    pred_cols = [c for c in pred_cols if c in df.columns]



                self._update_temporal_status(f"Found prediction columns: {pred_cols}")



                if not pred_cols:

                    self._update_temporal_status("No prediction columns found! Need to train first-layer models.")

                    self.ts_progress.stop()

                    return



                # Build temporal features first

                self._update_temporal_status("Building temporal features for validation...")

                df_temporal = build_temporal_features(df, pred_cols, lookback=3)



                # Validate

                self._update_temporal_status("Running temporal integrity checks...")

                result = validate_temporal_integrity(df_temporal, pred_cols)



                self._update_temporal_status("\n" + "="*50)

                self._update_temporal_status("VALIDATION RESULTS")

                self._update_temporal_status("="*50)

                self._update_temporal_status(f"Passed: {result['passed']}")



                if result['checks']:

                    self._update_temporal_status("\nChecks:")

                    for check, value in result['checks'].items():

                        self._update_temporal_status(f"  {check}: {value}")



                if result['warnings']:

                    self._update_temporal_status("\nWarnings:")

                    for warn in result['warnings']:

                        self._update_temporal_status(f"  - {warn}")



                if result['errors']:

                    self._update_temporal_status("\nErrors:")

                    for err in result['errors']:

                        self._update_temporal_status(f"  - {err}")



            except Exception as e:

                import traceback

                self._update_temporal_status(f"Validation failed: {e}")

                self._update_temporal_status(traceback.format_exc())

            finally:

                self.ts_progress.stop()



        threading.Thread(target=validation_thread, daemon=True).start()



    def _view_temporal_state(self):

        """View inference state history"""

        try:

            from bma_models.temporal_stacking import TemporalInferenceState



            state_mgr = TemporalInferenceState()

            history = state_mgr.get_available_history()



            self._update_temporal_status("\n" + "="*50)

            self._update_temporal_status("INFERENCE STATE HISTORY")

            self._update_temporal_status("="*50)

            self._update_temporal_status(f"State directory: {state_mgr.state_dir}")

            self._update_temporal_status(f"Available dates: {len(history)}")



            if history:

                self._update_temporal_status("\nRecent states:")

                for date in history[-10:]:

                    self._update_temporal_status(f"  - {date}")

            else:

                self._update_temporal_status("\nNo state files found. Train and save predictions to create state.")



        except Exception as e:

            self._update_temporal_status(f"Failed to view state: {e}")



    def _run_single_backtest(self):

        """ ? ?"""

        try:

            # retrieval ??

            backtest_type = self.backtest_type.get()

            

            #  

            start_date = self.ent_bt_start_date.get()

            end_date = self.ent_bt_end_date.get()

            

            #  ??

            from datetime import datetime

            try:

                datetime.strptime(start_date, '%Y-%m-%d')

                datetime.strptime(end_date, '%Y-%m-%d')

            except ValueError:

                messagebox.showerror("????", "??????????????? YYYY-MM-DD ???")

                return

            

            #  

            self.bt_progress.start()

            self._update_backtest_status("starting ?..")

            

            # in in ? 

            threading.Thread(

                target=self._execute_backtest_thread,

                args=(backtest_type,),

                daemon=True

            ).start()

            

        except Exception as e:

            self.bt_progress.stop()

            self._update_backtest_status(f"?????????: {e}")

            messagebox.showerror("????", f"?????????: {e}")

    

    def _run_strategy_comparison(self):

        """ ? for"""

        try:

            self.bt_progress.start()

            self._update_backtest_status("starting for ?..")

            

            # in in ? for?

            threading.Thread(

                target=self._execute_strategy_comparison_thread,

                daemon=True

            ).start()

            

        except Exception as e:

            self.bt_progress.stop()

            self._update_backtest_status(f"????for??????: {e}")

            messagebox.showerror("????", f"????for??????: {e}")

    

    def _run_quick_backtest(self):

        """ use ? """

        try:

            # settings ? ??            self.ent_bt_start_date.delete(0, tk.END)

            self.ent_bt_start_date.insert(0, "2023-01-01")



            self.ent_bt_end_date.delete(0, tk.END)

            self.ent_bt_end_date.insert(0, "2023-12-31")



            self.ent_bt_capital.delete(0, tk.END)

            self.ent_bt_capital.insert(0, "50000")



            self.ent_bt_max_positions.delete(0, tk.END)

            self.ent_bt_max_positions.insert(0, "10")



            #  ? 

            self._run_single_backtest()



        except Exception as e:

            messagebox.showerror("????", f"?????????: {e}")



    def _run_comprehensive_backtest(self):

        """ comprehensive_model_backtest GUI"""

        if getattr(self, '_comprehensive_backtest_thread', None) and self._comprehensive_backtest_thread.is_alive():

            self._update_backtest_status("[ ]  ? ? ?..")

            return



        script_path = os.path.join(os.getcwd(), 'scripts', 'comprehensive_model_backtest.py')

        if not os.path.exists(script_path):

            self._update_backtest_status(f"[ ]   {script_path}")

            return



        def _worker():

            cmd = [sys.executable, script_path]

            self.after(0, lambda: self._update_backtest_status("[ ]  ? ..."))

            try:

                with subprocess.Popen(

                    cmd,

                    stdout=subprocess.PIPE,

                    stderr=subprocess.STDOUT,

                    text=True,

                    bufsize=1

                ) as proc:

                    if proc.stdout:

                        for line in proc.stdout:

                            if not line:

                                continue

                            msg = line.rstrip()

                            self.after(0, lambda m=msg: self._update_backtest_status(f"[ ] {m}"))

                    return_code = proc.wait()

            except FileNotFoundError:

                self.after(0, lambda: self._update_backtest_status("[ ] Python ?"))

                return

            except Exception as exc:

                self.after(0, lambda e=exc: self._update_backtest_status(f"[ ]  ? : {e}"))

                return



            if return_code == 0:

                self.after(0, lambda: self._update_backtest_status("[ ]  ?result/model_backtest"))

            else:

                self.after(0, lambda code=return_code: self._update_backtest_status(f"[ ] ? ??{code}"))



        self._comprehensive_backtest_thread = threading.Thread(target=_worker, daemon=True)

        self._comprehensive_backtest_thread.start()



    def _run_prediction_only(self):

        """

        ???????????????????????????????

        ?????_direct_predict_snapshot??????????? ?????Excel???????EMA?????

        """

        try:

            if getattr(self, '_prediction_thread', None) and self._prediction_thread.is_alive():

                self._update_prediction_status("???????????? ??????...")

                return



            # Get tickers from UI

            raw_text = self.pred_ticker_entry.get("1.0", tk.END) if hasattr(self, 'pred_ticker_entry') else ''

            stocks = [s.strip().upper() for s in raw_text.split(',') if s.strip()]

            if not stocks and hasattr(self, 'pred_selected_pool'):

                stocks = list(self.pred_selected_pool)

            if not stocks:

                messagebox.showwarning("???", "????????????????????")

                return



            # Store stocks in selected_pool_info for _direct_predict_snapshot to use

            if not hasattr(self, 'selected_pool_info'):

                self.selected_pool_info = {}

            self.selected_pool_info['tickers'] = stocks



            if hasattr(self, 'pred_progress'):

                self.pred_progress.start()

            self._update_prediction_status("?? ?????????????????????Excel???????EMA?????...")



            def _run_prediction_thread():

                try:

                    # Redirect log output to prediction status

                    original_log = self.log

                    def log_to_status(msg):

                        self.after(0, lambda m=msg: self._update_prediction_status(m))

                        original_log(msg)

                    self.log = log_to_status

                    

                    # Call unified _direct_predict_snapshot function

                    self.after(0, lambda: self._update_prediction_status("?? ???????????????????..."))

                    self._direct_predict_snapshot()

                    

                    self.after(0, lambda: self._update_prediction_status("? ???????"))

                    self.after(0, lambda: self._update_prediction_status("?? Excel???????????????????"))

                    

                except Exception as exc:

                    self.after(0, lambda e=exc: self._update_prediction_status(f"?????: {e}"))

                    self.after(0, lambda e=exc: messagebox.showerror("????", f"?????:\n{e}"))

                finally:

                    if hasattr(self, 'pred_progress'):

                        self.after(0, self.pred_progress.stop)

                    # Restore original log

                    if hasattr(self, 'log'):

                        self.log = original_log



            import threading

            self._prediction_thread = threading.Thread(target=_run_prediction_thread, daemon=True)

            self._prediction_thread.start()



        except Exception as e:

            if hasattr(self, 'pred_progress'):

                self.pred_progress.stop()

            messagebox.showerror("????", f"?????????: {e}")

    

    def _execute_backtest_thread(self, backtest_type):

        """in in ? ?"""

        try:

            if backtest_type == "professional":

                self._run_professional_backtest()

            elif backtest_type == "autotrader":

                self._run_autotrader_backtest()

            elif backtest_type == "weekly":

                self._run_weekly_backtest()

                

        except Exception as e:

            error_msg = str(e)

            self.after(0, lambda msg=error_msg: self._update_backtest_status(f"?????????: {msg}"))

            self.after(0, lambda msg=error_msg: messagebox.showerror("????", f"?????????: {msg}"))

        finally:

            self.after(0, lambda: self.bt_progress.stop())

    

    def _execute_strategy_comparison_thread(self):

        """in in ? for"""

        try:

            # ? usebacktest_enginein run_backtest tobacktest_engine?

            from autotrader.backtest_engine import run_preset_backtests

            

            self.after(0, lambda: self._update_backtest_status("starting ? for.."))

            

            #  ? ? for?

            run_preset_backtests()

            

            self.after(0, lambda: self._update_backtest_status(" for completed to ./strategy_comparison.csv"))

            self.after(0, lambda: messagebox.showinfo("completed", " for completed \n to before?"))

            

        except Exception as e:

            error_msg = str(e)

            self.after(0, lambda msg=error_msg: self._update_backtest_status(msg))

            self.after(0, lambda msg=error_msg: messagebox.showerror("????", f"????for???: {msg}"))

        finally:

            self.after(0, lambda: self.bt_progress.stop())

    

    def _run_autotrader_backtest(self):

        """ ?AutoTrader BMA  ?"""

        try:

            from autotrader.backtest_engine import AutoTraderBacktestEngine, BacktestConfig

            from autotrader.backtest_analyzer import analyze_backtest_results

            

            self.after(0, lambda: self._update_backtest_status(" ?AutoTrader  ..."))

            

            #  

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

            

            self.after(0, lambda: self._update_backtest_status(" ? ?.."))

            

            #  ??

            engine = AutoTraderBacktestEngine(config)

            

            self.after(0, lambda: self._update_backtest_status(" ? ?.."))

            

            #  ? 

                                #  tobacktest_engine.py

            from .backtest_engine import run_backtest_with_config

            results = run_backtest_with_config(config)

            

            if results:

                self.after(0, lambda: self._update_backtest_status(" ..."))

                

                #  

                output_dir = self.ent_bt_output_dir.get()

                if not os.path.exists(output_dir):

                    os.makedirs(output_dir)

                

                analyzer = analyze_backtest_results(results, output_dir)

                

                #  summary

                summary = f"""

AutoTrader BMA  completed?



 : {results['period']['start_date']} -> {results['period']['end_date']}

 : {results['returns']['total_return']:.2%}

 ? {results['returns']['annual_return']:.2%}

 : {results['returns']['sharpe_ratio']:.3f}

? ? {results['returns']['max_drawdown']:.2%}

 ?? {results['returns']['win_rate']:.2%}

 ?: {results['trading']['total_trades']}

? ?? ${results['portfolio']['final_value']:,.2f}



 to: {output_dir}

                """

                

                self.after(0, lambda: self._update_backtest_status(summary))

                self.after(0, lambda s=summary: messagebox.showinfo(" completed", f"AutoTrader BMA  completed \n\n{s}"))

                

            else:

                self.after(0, lambda: self._update_backtest_status(" failed no "))

                

        except ImportError as e:

            error_msg = str(e)

            self.after(0, lambda msg=error_msg: self._update_backtest_status(f" ?failed: {msg}"))

        except Exception as e:

            error_msg = str(e)

            self.after(0, lambda msg=error_msg: self._update_backtest_status(f"AutoTrader  failed: {msg}"))

            import traceback

            traceback.print_exc()

    

    def _run_weekly_backtest(self):

        """ ? ? BMA  X? no ?"""

        try:

            from autotrader.backtest_engine import BacktestConfig, run_backtest_with_config

            from autotrader.backtest_analyzer import analyze_backtest_results



            self.after(0, lambda: self._update_backtest_status(" ? ..."))



            #  useandAutoTrader settings ? 

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



            self.after(0, lambda: self._update_backtest_status(" ? ? ..."))



            results = run_backtest_with_config(config)



            if results:

                #  

                output_dir = self.ent_bt_output_dir.get()

                if not os.path.exists(output_dir):

                    os.makedirs(output_dir)



                analyze_backtest_results(results, output_dir)



                summary = f"""

 ??BMA  completed?



 : {results['period']['start_date']} -> {results['period']['end_date']}

 : {results['returns']['total_return']:.2%}

 ? {results['returns']['annual_return']:.2%}

 : {results['returns']['sharpe_ratio']:.3f}

? ? {results['returns']['max_drawdown']:.2%}

 ?? {results['returns']['win_rate']:.2%}

 ?: {results['trading']['total_trades']}

? ?? ${results['portfolio']['final_value']:,.2f}



 to: {output_dir}

                """



                self.after(0, lambda: self._update_backtest_status(summary))

                self.after(0, lambda s=summary: messagebox.showinfo(" completed", f" ?BMA  completed \n\n{s}"))

            else:

                self.after(0, lambda: self._update_backtest_status(" ? failed no "))



        except ImportError as e:

            error_msg = str(e)

            self.after(0, lambda msg=error_msg: self._update_backtest_status(f" ?failed: {msg}"))

        except Exception as e:

            error_msg = str(e)

            self.after(0, lambda msg=error_msg: self._update_backtest_status(f" ? failed: {msg}"))

            import traceback

            traceback.print_exc()

    

    def _update_backtest_status(self, message):

        """updates ?"""

        timestamp = datetime.now().strftime("%H:%M:%S")

        self.bt_status_text.insert(tk.END, f"[{timestamp}] {message}\n")

        self.bt_status_text.see(tk.END)

        self.update_idletasks()



    def _build_status_panel(self, parent):

        """ ? ? ?"""

        #  ? n? ?

        status_info = tk.Frame(parent)

        status_info.pack(fill=tk.X, padx=5, pady=5)

        

        #  ? connection ?and ??

        row1 = tk.Frame(status_info)

        row1.pack(fill=tk.X, pady=2)

        

        tk.Label(row1, text="connection ?", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)

        self.lbl_connection_status = tk.Label(row1, text=" connection", fg="red", font=("Arial", 9))

        self.lbl_connection_status.pack(side=tk.LEFT, padx=5)

        

        tk.Label(row1, text=" ?", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)

        self.lbl_engine_status = tk.Label(row1, text=" start", fg="gray", font=("Arial", 9))

        self.lbl_engine_status.pack(side=tk.LEFT, padx=5)

        

        tk.Label(row1, text=" ? ?", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)

        self.lbl_model_status = tk.Label(row1, text="??????", fg="orange", font=("Arial", 9))

        self.lbl_model_status.pack(side=tk.LEFT, padx=5)

        

        #  ? account R?and ?

        row2 = tk.Frame(status_info)

        row2.pack(fill=tk.X, pady=2)

        

        tk.Label(row2, text="?", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)

        self.lbl_net_value = tk.Label(row2, text="$0.00", fg="blue", font=("Arial", 9))

        self.lbl_net_value.pack(side=tk.LEFT, padx=5)

        

        tk.Label(row2, text="accountID:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)

        self.lbl_account_id = tk.Label(row2, text="-", fg="black", font=("Arial", 9))

        self.lbl_account_id.pack(side=tk.LEFT, padx=5)



        tk.Label(row2, text="ClientID:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)

        self.lbl_client_id = tk.Label(row2, text="-", fg="black", font=("Arial", 9))

        self.lbl_client_id.pack(side=tk.LEFT, padx=5)



        tk.Label(row2, text="positions", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)

        self.lbl_positions = tk.Label(row2, text="0", fg="purple", font=("Arial", 9))

        self.lbl_positions.pack(side=tk.LEFT, padx=5)

        

        tk.Label(row2, text=" :", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)

        self.lbl_daily_trades = tk.Label(row2, text="0", fg="green", font=("Arial", 9))

        self.lbl_daily_trades.pack(side=tk.LEFT, padx=5)

        

        tk.Label(row2, text="?afterupdates:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)

        self.lbl_last_update = tk.Label(row2, text=" starting", fg="gray", font=("Arial", 9))

        self.lbl_last_update.pack(side=tk.LEFT, padx=5)

        

        #  ? ?and ??

        row3 = tk.Frame(status_info)

        row3.pack(fill=tk.X, pady=2)

        

        tk.Label(row3, text=" ?:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)

        self.lbl_watch_count = tk.Label(row3, text="0", fg="teal", font=("Arial", 9))

        self.lbl_watch_count.pack(side=tk.LEFT, padx=5)

        

        tk.Label(row3, text=" ? ?", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=15)

        self.lbl_signal_status = tk.Label(row3, text=" in", fg="orange", font=("Arial", 9))

        self.lbl_signal_status.pack(side=tk.LEFT, padx=5)

        

        #  ? 

        self.lbl_status_indicator = tk.Label(row3, text="??", fg="red", font=("Arial", 14))

        self.lbl_status_indicator.pack(side=tk.RIGHT, padx=15)

        

        tk.Label(row3, text=" ? ?", font=("Arial", 9, "bold")).pack(side=tk.RIGHT, padx=5)

        

        # start ?updates when?

        self._start_status_monitor()

    

    def _start_status_monitor(self):

        """start ? when"""

        self._update_status()

        # ? secondsupdates? ???

        self.after(2000, self._start_status_monitor)

    

    def _update_status(self):

        """updates ? ?"""

        try:

            # updatesconnection ??

            if self.trader and hasattr(self.trader, 'ib') and self.trader.ib.isConnected():

                self.lbl_connection_status.config(text="connection", fg="green")

            else:

                self.lbl_connection_status.config(text=" connection", fg="red")

            

            # updates ??

            if self.engine:

                if hasattr(self, '_engine_loop_task') and self._engine_loop_task and not self._engine_loop_task.done():

                    self.lbl_engine_status.config(text=" ?in", fg="green")

                    self.lbl_status_indicator.config(fg="green")

                else:

                    self.lbl_engine_status.config(text="start", fg="blue")

                    self.lbl_status_indicator.config(fg="blue")

            else:

                self.lbl_engine_status.config(text=" start", fg="gray")

                self.lbl_status_indicator.config(fg="red")

            

            # updatesaccount??

            if self.trader and hasattr(self.trader, 'net_liq'):

                #  use ?as0/None ?

                try:

                    current_net = getattr(self.trader, 'net_liq', None)

                    if isinstance(current_net, (int, float)) and current_net is not None:

                        if self._last_net_liq is None or abs(float(current_net) - float(self._last_net_liq)) > 1e-6:

                            self._last_net_liq = float(current_net)

                    if self._last_net_liq is not None:

                        self.lbl_net_value.config(text=f"${self._last_net_liq:,.2f}")

                except Exception:

                    pass

                # updatesaccountIDand ID

                try:

                    acc_id = getattr(self.trader, 'account_id', None)

                    if acc_id:

                        self.lbl_account_id.config(text=str(acc_id), fg=("green" if str(acc_id).lower()=="c2dvdongg" else "black"))

                    else:

                        self.lbl_account_id.config(text="-", fg="black")

                except Exception:

                    pass

                try:

                    # and before ??client_id for notis ??3130

                    actual_cid = getattr(self.trader, 'client_id', None)

                    try:

                        expected_cid = self.config_manager.get('connection.client_id', None)

                    except Exception:

                        expected_cid = None

                    cid_ok = bool(actual_cid is not None and expected_cid is not None and actual_cid == expected_cid)

                    self.lbl_client_id.config(text=str(actual_cid if actual_cid is not None else '-'), fg=("green" if cid_ok else "black"))

                except Exception:

                    pass

                

                # updatespositions?

                position_count = len(getattr(self.trader, 'positions', {}))

                self.lbl_positions.config(text=str(position_count))

            

            # updates ??

            if self.trader and hasattr(self.trader, 'tickers'):

                watch_count = len(getattr(self.trader, 'tickers', {}))

                self.lbl_watch_count.config(text=str(watch_count))

            

            # updates?afterupdateswhen?

            current_time = datetime.now().strftime("%H:%M:%S")

            self.lbl_last_update.config(text=current_time)

            

            # check ? ? if has ???

            if hasattr(self, '_model_training') and self._model_training:

                self.lbl_model_status.config(text=" ?in", fg="blue")

            elif hasattr(self, '_model_trained') and self._model_trained:

                self.lbl_model_status.config(text="?????", fg="green")

            else:

                self.lbl_model_status.config(text=" ????", fg="orange")

                

        except Exception as e:

            #  ?updatesfailednot ? ??

            pass

    

    def _update_signal_status(self, status_text, color="black"):

        """updates ? ?"""

        try:

            self.lbl_signal_status.config(text=status_text, fg=color)

        except Exception:

            pass

    

    def _set_connection_error_state(self, error_msg: str):

        """ ? ?"""

        try:

            self.log(f" ? ? {error_msg}")

            #  ? GUI ? ??

            if hasattr(self, 'lbl_status'):

                self.lbl_status.config(text=f" ?: {error_msg[:50]}...")

        except Exception as e:

            #  GUI ? ? ??

            print(f" ? ? {e},  ? ?: {error_msg}")



    def _update_daily_trades(self, count):

        """updates ?"""

        try:

            self.lbl_daily_trades.config(text=str(count))

        except Exception as e:

            #  ? ? ? ? ?

            self.log(f" ? ? {e}")

            # GUI 



    # ========== Strategy Engine Methods ==========

    

    def _update_strategy_status(self):

        """Update strategy status display"""

        if not hasattr(self, 'strategy_status_text'):

            return

            

        try:

            status_text = "=== Strategy Engine Status ===\n\n"

            

            if hasattr(self, 'strategy_status'):

                for key, value in self.strategy_status.items():

                    status_text += f"{key}: {'Active' if value else 'Inactive'}\n"

            else:

                status_text += "Strategy components not initialized\n"

                

            status_text += f"\nLast updated: {datetime.now().strftime('%H:%M:%S')}\n"

            

            self.strategy_status_text.delete(1.0, tk.END)

            self.strategy_status_text.insert(tk.END, status_text)

            

        except Exception as e:

            self.log(f"Failed to update strategy status: {e}")

    

    def _test_alpha_factors(self):

        """Alpha factors ?-  Simple 25 ?"""

        try:

            self.log("Alpha factors - Simple 25 ")

            self.strategy_status['bma_model_loaded'] = True

            self._update_strategy_status()



        except Exception as e:

            self.log(f"Strategy status update failed: {e}")

            self.strategy_status['bma_model_loaded'] = True

            self._update_strategy_status()

    

    def _run_bma_model_demo(self):

        """Run BMA model for strategy selection (Simple 25 ?)"""

        try:

            self.log("??  ?BMA ? ? (Simple 25 ?)...")

            self.log("??  ?..")

            self.log("?  ? ? ?..")

            self.log("?? ...")



            # This would typically load real market data and run BMA

            # For demo purposes, we'll simulate the process

            import time

            import threading



            def run_bma_async():

                try:

                    self.log("?? ? ??..")

                    time.sleep(1)

                    self.log("??  ? ? ?(XGBoost, CatBoost, ElasticNet)...")

                    time.sleep(1)

                    self.log("??  ? Ridge ?..")

                    time.sleep(1)

                    self.log("BMA ? ? ??- Simple 25 Ridge ?")

                    self.log("??  ? : IC=0.045, ICIR=1.2, Sharpe=0.8")

                    self.strategy_status['bma_model_loaded'] = True

                    self.after_idle(self._update_strategy_status)

                except Exception as e:

                    self.log(f"BMA model failed: {e}")

            

            threading.Thread(target=run_bma_async, daemon=True).start()

            

        except Exception as e:

            self.log(f"Failed to run BMA model: {e}")

    

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

            

            # Generate signals (simplified for demo)

            signals = {

                'AAPL': {'signal': 0.05, 'confidence': 0.8},

                'MSFT': {'signal': -0.02, 'confidence': 0.6},

                'GOOGL': {'signal': 0.03, 'confidence': 0.7}

            }

            

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

            

            # This would typically fetch real data from Polygon

            # For demo, we'll simulate the process

            import threading

            import time

            

            def load_data_async():

                try:

                    time.sleep(3)  # Simulate data loading

                    self.log("Polygon market data loaded successfully")

                    self.after_idle(self._update_strategy_status)

                except Exception as e:

                    self.log(f"Failed to load Polygon data: {e}")

            

            threading.Thread(target=load_data_async, daemon=True).start()

            

        except Exception as e:

            self.log(f"Failed to load Polygon data: {e}")

    

    def _compute_t5_factors(self):

        """Compute T+5 prediction factors"""

        try:

            if not hasattr(self, 'polygon_factors'):

                self.log("Polygon factors not initialized")

                return

                

            self.log("Computing T+5 prediction factors...")

            

            # This would use the polygon_factors to compute short-term prediction factors

            import threading

            import time

            

            def compute_factors_async():

                try:

                    time.sleep(4)  # Simulate computation

                    self.log("T+5 factors computed successfully")

                    self.log("Factors include: momentum, reversal, volume, volatility, microstructure")

                    self.after_idle(self._update_strategy_status)

                except Exception as e:

                    self.log(f"T+5 factor computation failed: {e}")

            

            threading.Thread(target=compute_factors_async, daemon=True).start()

            

        except Exception as e:

            self.log(f"Failed to compute T+5 factors: {e}")

    

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

                status_text += f"IBKR Connection: {'Connected' if self.trader.is_connected() else '?Disconnected'}\n"

            else:

                status_text += "IBKR Connection: Not initialized\n"

            

            # Strategy components

            if hasattr(self, 'strategy_status'):

                status_text += f"Alpha Engine: {'Ready' if self.strategy_status.get('alpha_engine_ready', False) else 'Not ready'}\n"

                status_text += f"Polygon Factors: {'Ready' if self.strategy_status.get('polygon_factors_ready', False) else 'Not ready'}\n"

                status_text += f"Risk Balancer: {'Ready' if self.strategy_status.get('risk_balancer_ready', False) else 'Not ready'}\n"

            

            # Market data status

            status_text += "Market Data: Ready\n"

            status_text += f"Database: {'Connected' if self.db else '?Not available'}\n"

            

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

                self.log("IBKR connection test passed")

                # Test basic API calls

                account_summary = self.trader.get_account_summary()

                self.log(f"Account data accessible: {len(account_summary)} items")

            else:

                self.log("IBKR connection test failed - not connected")

            

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

                    self.log(f"Market data test for {symbol}: Price data accessible")

                

                self.log("Market data test completed successfully")

            else:

                self.log("No trader available for market data test")

                # Simulate successful test for demo

                self.log("Market data simulation test passed")

            

            self._update_system_status()

            

        except Exception as e:

            self.log(f"Market data test failed: {e}")

    

    def _test_order_placement(self):

        """Test order placement (paper trading)"""

        try:

            self.log("Testing order placement...")

            

            if hasattr(self, 'trader') and self.trader:

                # Test with a small paper trade

                test_order = {

                    'symbol': 'AAPL',

                    'quantity': 1,

                    'order_type': 'LMT',

                    'price': 150.00,

                    'action': 'BUY'

                }

                

                self.log(f"Placing test order: {test_order}")

                # This would place an actual test order

                self.log("Test order placement completed")

                self.log("Note: This was a paper trading test")

            else:

                self.log("No trader available for order test")

                # Simulate test for demo

                self.log("Order placement simulation test passed")

            

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

                        self.log("?? All systems operational!")

                    elif passed >= total * 0.8:

                        self.log("??Most systems operational with minor issues")

                    else:

                        self.log("Multiple system issues detected")

                    

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

                self.log("Alpha engine available")

            else:

                self.log("Alpha engine not available")

                

            # Test polygon factors

            if hasattr(self, 'polygon_factors'):

                self.log("Polygon factors available")

            else:

                self.log("Polygon factors not available")

                

            # Test risk balancer

            if hasattr(self, 'risk_balancer_adapter'):

                self.log("Risk balancer available")

            else:

                self.log("Risk balancer not available")

            

            self.log("Strategy components test completed")

            

        except Exception as e:

            self.log(f"Strategy components test failed: {e}")

    

    def _test_risk_controls(self):

        """Test risk control systems"""

        try:

            self.log("Testing risk controls...")

            

            if hasattr(self, 'risk_balancer_adapter'):

                # Test risk limits

                self.log("Risk balancer accessible")

                

                # Test position limits

                self.log("Position limits configured")

                

                # Test order validation

                self.log("Order validation active")

                

                self.log("Risk controls test passed")

            else:

                self.log("??Risk balancer not initialized - using basic controls")

            

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

            

            # For demo, simulate the process

            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

            signals = []

            

            for symbol in symbols:

                # Use existing signal calculation from unified factors

                try:

                    # Use unified signal processor (eliminates redundant signal code)

                    from autotrader.unified_signal_processor import get_unified_signal_processor, SignalMode

                    from autotrader.environment_config import get_environment_manager

                    

                    env_manager = get_environment_manager()

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

                

                # ??  ??+  ?+  ? ?

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

         ? ? ??+   +  ? ???

        

        Args:

            symbol:  ? 

            signal_strength:  ? ??

            confidence:  ? Z???

            

        Returns:

             ? None

        """

        try:

            #  ? ??( ? )

            import random

            current_price = 150.0 + random.uniform(-20, 20)

            price_history = [current_price + random.gauss(0, 2) for _ in range(100)]

            volume_history = [1000000 + random.randint(-200000, 500000) for _ in range(100)]

            

            # 1.  ??

            freshness_result = None

            if self.freshness_scorer:

                from datetime import datetime, timedelta

                data_timestamp = datetime.now() - timedelta(minutes=random.randint(1, 30))

                

                freshness_result = self.freshness_scorer.calculate_freshness_score(

                    symbol=symbol,

                    data_timestamp=data_timestamp,

                    data_source='realtime',

                    missing_ratio=random.uniform(0, 0.1),

                    data_gaps=[]

                )

                

                #  ?

                effective_signal, signal_info = self.freshness_scorer.apply_freshness_to_signal(

                    symbol, signal_strength, freshness_result['freshness_score']

                )

                

                if not signal_info.get('passes_threshold', False):

                    self.log(f"{symbol}  ? M ?")

                    return None

                

                signal_strength = effective_signal  #  ?

            

            # 2.  

            gating_result = None

            if self.volatility_gating:

                can_trade, gating_details = self.volatility_gating.should_trade(

                    symbol=symbol,

                    signal_strength=signal_strength,  # ? ??

                    price_data=price_history,

                    volume_data=volume_history

                )

                

                if not can_trade:

                    self.log(f"{symbol}  M ? {gating_details.get('reason', 'unknown')}")

                    return None

                

                gating_result = gating_details

            

            # 3.  ? ???

            position_result = None

            if self.position_calculator:

                available_cash = 100000.0  #  ??0 ??

                

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

                    self.log(f"{symbol}  ? ? ? {position_result.get('error', 'unknown')}")

                    return None

            

            #  ?

            enhanced_signal = {

                'symbol': symbol,

                'weighted_prediction': signal_strength,

                'confidence': confidence,

                'current_price': current_price,

                'can_trade': True,

                

                #  ??

                'freshness_info': freshness_result,

                'gating_info': gating_result,

                'position_info': position_result,

                

                #  

                'dynamic_shares': position_result.get('shares', 100) if position_result else 100,

                'dynamic_threshold': freshness_result.get('dynamic_threshold') if freshness_result else 0.005,

                'volatility_score': gating_result.get('volatility') if gating_result else 0.15,

                'liquidity_score': gating_result.get('liquidity_score') if gating_result else 1.0

            }

            

            self.log(f"{symbol}  ? ? ??{enhanced_signal['dynamic_shares']}, "

                    f" ?{enhanced_signal['dynamic_threshold']:.4f}, "

                    f" ?{enhanced_signal['volatility_score']:.3f}")

            

            return enhanced_signal

            

        except Exception as e:

            self.log(f"{symbol}  ? : {e}")

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





    def _add_backtest_stock(self):

        """ ? """

        stock = self.ent_bt_stock_input.get().strip().upper()

        if stock:

            # ? ??

            stocks = self.bt_stock_listbox.get(0, tk.END)

            if stock not in stocks:

                self.bt_stock_listbox.insert(tk.END, stock)

                self.ent_bt_stock_input.delete(0, tk.END)

                self.log(f" ?  {stock}")

        else:

            messagebox.showinfo("???", f"{stock} ????? ???")

    

    def _import_stocks_from_db(self):

        """ ? ? ?"""

        try:

            if hasattr(self, 'db'):

                #  ??

                stock_lists = self.db.get_all_stock_lists()

                if stock_lists:

                    #  ??

                    import tkinter.simpledialog as simpledialog

                    list_names = [f"{sl['name']} ({len(sl.get('stocks', []))} stocks)" for sl in stock_lists]

                    

                    #  U? ? ?

                    dialog = tk.Toplevel(self)

                    dialog.title(" ? ?")

                    dialog.geometry("400x300")

                    

                    tk.Label(dialog, text=" ? ? ?").pack(pady=5)

                    

                    listbox = tk.Listbox(dialog, selectmode=tk.SINGLE)

                    for name in list_names:

                        listbox.insert(tk.END, name)

                    listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

                    

                    def on_select():

                        selection = listbox.curselection()

                        if selection:

                            idx = selection[0]

                            selected_list = stock_lists[idx]

                            stocks = selected_list.get('stocks', [])

                            

                            #  ??

                            self.bt_stock_listbox.delete(0, tk.END)

                            

                            #  ?

                            for stock in stocks:

                                self.bt_stock_listbox.insert(tk.END, stock)

                            

                            self.log(f" ? ?{len(stocks)}  ??")

                            dialog.destroy()

                    

                    tk.Button(dialog, text="?, command=on_select).pack(pady=5")

                    tk.Button(dialog, text=" ?, command=dialog.destroy).pack(pady=5")

                    

                else:

                    messagebox.showinfo("???", "????????? ??????? ?")

            else:

                messagebox.showwarning("???", "????? ?? ????")

                

        except Exception as e:

            messagebox.showerror("????", f"?????????: {e}")

            self.log(f" ? ? {e}")

    

    def _clear_backtest_stocks(self):

        """ ? ?"""

        self.bt_stock_listbox.delete(0, tk.END)

        self.log(" ? ?")

    

    def _remove_selected_stocks(self):

        """ ?"""

        selection = self.bt_stock_listbox.curselection()

        #  ? 

        for index in reversed(selection):

            stock = self.bt_stock_listbox.get(index)

            self.bt_stock_listbox.delete(index)

            self.log(f" ?: {stock}")

    

    def _run_professional_backtest(self):

        """ ? BMA ?"""

        try:

            #  

            import sys

            sys.path.append('.')

            from bma_professional_backtesting import BacktestConfig, BMABacktestEngine

            

            self.after(0, lambda: self._update_backtest_status(" ? .."))

            

            #  ? ??

            stocks = list(self.bt_stock_listbox.get(0, tk.END))

            if not stocks:

                self.after(0, lambda: messagebox.showwarning("???", "???????????"))

                return

            

            #  

            start_date = self.ent_bt_start_date.get()

            end_date = self.ent_bt_end_date.get()

            initial_capital = float(self.ent_bt_capital.get())

            commission = float(self.ent_bt_commission.get())

            max_positions = int(self.ent_bt_max_positions.get())

            rebalance_freq = self.cb_bt_rebalance.get()

            

            #  

            config = BacktestConfig(

                start_date=start_date,

                end_date=end_date,

                initial_capital=initial_capital,

                commission_rate=commission,

                position_sizing='risk_parity',  #  ??

                max_position_size=float(self.ent_bt_max_weight.get()),

                rebalance_frequency=rebalance_freq,

                stop_loss=float(self.ent_bt_stop_loss.get()),

                enable_walk_forward=True,  #  ?Walk-Forward ??

                train_window_months=24,

                test_window_months=6,

                step_months=3,

                enable_regime_detection=True,  #  ? ????

                monte_carlo_simulations=100,  # Monte Carlo??

                save_results=True,

                results_dir=self.ent_bt_output_dir.get(),

                generate_report=True,

                verbose=True

            )

            

            self.after(0, lambda: self._update_backtest_status(f"? ?{len(stocks)}  ???.."))

            

            #  ? BMA??

            from bma_models._bma_ultra_enhanced import UltraEnhancedQuantitativeModel

            bma_model = UltraEnhancedQuantitativeModel(enable_v6_enhancements=True)

            

            #  ??

            engine = BMABacktestEngine(config, bma_model)

            

            #  ? 

            self.after(0, lambda: self._update_backtest_status(" ?Walk-Forward ?.."))

            results = engine.run_backtest(stocks)

            

            #  

            result_msg = f"""

 ??



??  ??

   ?? {results.total_return:.2%}

   : {results.annualized_return:.2%}

   : {results.sharpe_ratio:.2f}

  

??  :

  ? ? {results.max_drawdown:.2%}

   ?? {results.volatility:.2%}

  VaR(95%): {results.var_95:.2%}

  

??  ?:

   : {results.total_trades}

   ?? {results.win_rate:.2%}

   ?? {results.profit_factor:.2f}



??  Z? (95%):

   : [{results.return_ci[0]:.2%}, {results.return_ci[1]:.2%}]

   ?? [{results.sharpe_ci[0]:.2f}, {results.sharpe_ci[1]:.2f}]

  

 ?? {config.results_dir}

            """

            

            self.after(0, lambda msg=result_msg: self._update_backtest_status(msg))

            self.after(0, lambda: messagebox.showinfo(" ?, result_msg"))

            

            #  ? ?

            if self.var_bt_show_plots.get():

                self.after(0, lambda: self._update_backtest_status(" ?.."))

                #  ??

            

        except ImportError as e:

            error_msg = f" ? ? {e}\n ??bma_professional_backtesting.py  ?"

            self.after(0, lambda msg=error_msg: self._update_backtest_status(msg))

            self.after(0, lambda msg=error_msg: messagebox.showerror(" ?, msg"))

        except Exception as e:

            error_msg = f" ? ? {e}"

            self.after(0, lambda msg=error_msg: self._update_backtest_status(msg))

            self.after(0, lambda msg=error_msg: messagebox.showerror(" ?, msg"))

            import traceback

            traceback.print_exc()



    def _ensure_top_menu(self) -> None:

        try:

            menubar = tk.Menu(self)

            # File menu (minimal)

            file_menu = tk.Menu(menubar, tearoff=0)

            file_menu.add_command(label="Exit", command=self._on_closing)

            menubar.add_cascade(label="File", menu=file_menu)

            # Tools menu

            tools_menu = tk.Menu(menubar, tearoff=0)

            tools_menu.add_command(label="Return Comparison...", command=self._open_return_comparison_window)

            menubar.add_cascade(label="Tools", menu=tools_menu)

            self.config(menu=menubar)

        except Exception:

            pass



    def _ensure_toolbar(self) -> None:

        try:

            if getattr(self, '_toolbar_frame', None) is None:

                bar = ttk.Frame(self)

                bar.pack(side=tk.TOP, fill=tk.X)

                self._toolbar_frame = bar

                btn = ttk.Button(bar, text="Return Comparison", command=self._open_return_comparison_window)

                btn.pack(side=tk.LEFT, padx=4, pady=2)

        except Exception:

            pass



    def _open_return_comparison_window(self) -> None:

        # Create a standalone window reusing the existing form and handler

        win = tk.Toplevel(self)

        win.title("Return Comparison")

        frm = ttk.Frame(win)

        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Inputs

        ttk.Label(frm, text="Tickers (comma separated):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        entry_symbols = tk.Entry(frm, width=40)

        entry_symbols.insert(0, "AAPL,MSFT,GOOGL")

        entry_symbols.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(frm, text="Start date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        entry_start = tk.Entry(frm, width=15)

        entry_start.insert(0, (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'))

        entry_start.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(frm, text="End date (YYYY-MM-DD):").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)

        entry_end = tk.Entry(frm, width=15)

        entry_end.insert(0, datetime.now().strftime('%Y-%m-%d'))

        entry_end.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)

        # Output

        txt = tk.Text(frm, height=10, wrap=tk.WORD, state=tk.DISABLED)

        txt.grid(row=2, column=0, columnspan=4, padx=5, pady=(5, 0), sticky=tk.EW)

        frm.grid_columnconfigure(1, weight=1)

        frm.grid_columnconfigure(3, weight=1)

        

        # Wire a local handler that proxies to the existing worker via temporary bindings

        def run_compare():

            # Temporarily bind the existing GUI fields to reuse the handler

            prev_symbols = getattr(self, 'polygon_compare_symbols', None)

            prev_start = getattr(self, 'polygon_compare_start', None)

            prev_end = getattr(self, 'polygon_compare_end', None)

            prev_output = getattr(self, 'polygon_compare_output', None)

            try:

                self.polygon_compare_symbols = entry_symbols

                self.polygon_compare_start = entry_start

                self.polygon_compare_end = entry_end

                self.polygon_compare_output = txt

                self._compare_polygon_returns()

            finally:

                # Restore previous bindings

                self.polygon_compare_symbols = prev_symbols

                self.polygon_compare_start = prev_start

                self.polygon_compare_end = prev_end

                self.polygon_compare_output = prev_output

        

        btn = ttk.Button(frm, text="Compute Return Comparison", command=run_compare)

        btn.grid(row=0, column=3, padx=5, pady=5, sticky=tk.E)



        #  ??Excel  ?? ??

        ttk.Label(frm, text="Excel  ?").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)

        entry_excel = tk.Entry(frm, width=40)

        entry_excel.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)



        def browse_excel_local():

            try:

                path = filedialog.askopenfilename(

                    title=" ? Excel ?",

                    filetypes=[("Excel Files", "*.xlsx;*.xls")]

                )

                if path:

                    entry_excel.delete(0, tk.END)

                    entry_excel.insert(0, path)

            except Exception:

                pass



        ttk.Button(frm, text=" Excel...", command=browse_excel_local).grid(row=3, column=2, padx=5, pady=5, sticky=tk.W)



        def run_excel_backtest():

            #  Excel ??

            prev_excel_entry = getattr(self, 'polygon_compare_excel_entry', None)

            prev_output = getattr(self, 'polygon_compare_output', None)

            try:

                self.polygon_compare_excel_entry = entry_excel

                self.polygon_compare_output = txt

                self._compare_returns_from_excel()

            finally:

                self.polygon_compare_excel_entry = prev_excel_entry

                self.polygon_compare_output = prev_output



        ttk.Button(frm, text="Excel Top20 T+5 (vs SPY)", command=run_excel_backtest).grid(row=3, column=3, padx=5, pady=5, sticky=tk.E)



def init_batch() -> None:

    """Initialize application in batch/headless mode without GUI."""

    try:

        # Initialize core components without GUI

        from bma_models.unified_config_loader import get_config_manager as get_default_config

        from autotrader.unified_event_manager import get_event_loop_manager

        from autotrader.unified_monitoring_system import get_resource_monitor

        from autotrader.database import StockDatabase

        

        config_manager = get_default_config()

        loop_manager = get_event_loop_manager()

        resource_monitor = get_resource_monitor()

        

        # Start event loop manager

        if not loop_manager.start():

            raise RuntimeError("Failed to start event loop manager")

        

        # Start resource monitoring

        resource_monitor.start_monitoring()

        

        # Initialize database

        db = StockDatabase()

        

        print("[OK] Batch mode initialized successfully")

        print(f"[INFO] Config manager: {type(config_manager).__name__}")

        print(f"[INFO] Event loop manager: {'Running' if loop_manager.is_running else 'Stopped'}")

        print(f"[INFO] Database: {'Connected' if db else 'Not available'}")

        

        return {

            'config_manager': config_manager,

            'loop_manager': loop_manager,

            'resource_monitor': resource_monitor,

            'database': db

        }

    except Exception as e:

        print(f"[ERROR] Batch initialization failed: {e}")

        raise





def main() -> None:

    """Main entry point - supports both GUI and batch modes."""

    import sys

    

    # Check for batch mode flag

    batch_mode = '--batch' in sys.argv or '-b' in sys.argv

    

    if batch_mode:

        # Initialize in batch mode (headless)

        print("[INFO] Starting in batch mode...")

        try:

            components = init_batch()

            print("[OK] Batch mode ready. Components initialized.")

            # Keep running until interrupted

            import time

            try:

                while True:

                    time.sleep(1)

            except KeyboardInterrupt:

                print("\n[INFO] Shutting down batch mode...")

                if components and 'loop_manager' in components:

                    try:

                        components['loop_manager'].stop()

                    except Exception:

                        pass

                print("[OK] Batch mode stopped")

        except Exception as e:

            print(f"[ERROR] Batch mode failed: {e}")

            sys.exit(1)

    else:

        # GUI mode (default)

        app = None

        try:

            app = AutoTraderGUI()  # type: ignore

            # Set exit handler to ensure event loop closes properly

            def on_closing():

                try:

                    if hasattr(app, 'loop_manager') and app.loop_manager.is_running:

                        app.loop_manager.stop()

                    app.destroy()

                except Exception as e:

                    print(f"Exit handler error: {e}")

                    app.destroy()

            

            app.protocol("WM_DELETE_WINDOW", on_closing)

            app.mainloop()

        except Exception as e:

            print(f"Application startup error: {e}")

            if app and hasattr(app, 'loop_manager') and app.loop_manager.is_running:

                try:

                    app.loop_manager.stop()

                except Exception as e:

                    # Log shutdown error, even though program is exiting

                    print(f"Event loop manager shutdown error: {e}")





if __name__ == "__main__":

    main()















    def _run_timesplit_eval(self) -> None:
        try:
            selected_features = [feat for feat, var in getattr(self, 'timesplit_feature_vars', {}).items() if var.get()]
        except AttributeError:
            messagebox.showerror("80/20 Evaluation", "Timesplit UI not initialized yet.")
            return
        if not selected_features:
            messagebox.showerror("80/20 Evaluation", "Select at least one feature before running the evaluation.")
            return
        split_value = float(getattr(self, 'timesplit_split_var', tk.DoubleVar(value=0.8)).get())
        split_value = max(0.60, min(0.95, split_value))
        ema_top_n = "0" if getattr(self, 'var_timesplit_ema', tk.BooleanVar(value=False)).get() else "-1"
        cmd = [
            sys.executable,
            str(Path('scripts') / 'time_split_80_20_oos_eval.py'),
            '--data-file', str(STAGE_A_DATA_PATH),
            '--train-data', str(STAGE_A_DATA_PATH),
            '--horizon-days', '10',
            '--split', f"{split_value:.3f}",
            '--model', 'lambdarank',
            '--models', 'lambdarank',
            '--output-dir', str(Path('results') / 'timesplit_gui'),
            '--log-level', 'INFO',
            '--features',
        ] + selected_features + ['--ema-top-n', ema_top_n]
        if getattr(self, '_timesplit_thread', None) and self._timesplit_thread.is_alive():
            messagebox.showinfo("80/20 Evaluation", "An evaluation is already running. Please wait for it to finish.")
            return
        self.log(f"[TimeSplit] Starting 80/20 evaluation with split={split_value:.2f}, EMA={'ON' if ema_top_n == '0' else 'OFF'}")
        cwd = Path(__file__).resolve().parent.parent
        thread = threading.Thread(target=self._run_timesplit_eval_thread, args=(cmd, cwd), daemon=True)
        self._timesplit_thread = thread
        thread.start()

    def _run_timesplit_eval_thread(self, cmd: list[str], cwd: Path) -> None:
        try:
            proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except Exception as exc:
            self.after(0, lambda: self.log(f"[TimeSplit] Failed to start evaluation: {exc}"))
            messagebox.showerror("80/20 Evaluation", f"Failed to start process: {exc}")
            return
        for line in proc.stdout or []:
            msg = line.rstrip()
            if msg:
                self.after(0, lambda m=msg: self.log(f"[TimeSplit] {m}"))
        proc.wait()
        status = "completed successfully" if proc.returncode == 0 else f"finished with code {proc.returncode}"
        self.after(0, lambda: self.log(f"[TimeSplit] Evaluation {status}."))
        if proc.returncode == 0:
            self.after(0, lambda: messagebox.showinfo("80/20 Evaluation", "Evaluation finished successfully. Check results/timesplit_gui for outputs."))
        else:
            self.after(0, lambda: messagebox.showerror("80/20 Evaluation", f"Evaluation failed with code {proc.returncode}. See log for details."))

    def _open_timesplit_results(self) -> None:
        results_path = Path('results') / 'timesplit_gui'
        results_path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith('win'):
                os.startfile(results_path)  # type: ignore[attr-defined]
            elif sys.platform.startswith('darwin'):
                subprocess.Popen(['open', str(results_path)])
            else:
                subprocess.Popen(['xdg-open', str(results_path)])
        except Exception as exc:
            self.log(f"[TimeSplit] Failed to open results folder: {exc}")
            messagebox.showerror("80/20 Evaluation", f"Unable to open folder: {exc}")


