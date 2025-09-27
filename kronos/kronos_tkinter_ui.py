import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import json
import os

from .kronos_model import KronosModelWrapper, KronosConfig
from .utils import prepare_kline_data, format_prediction_results

logger = logging.getLogger(__name__)

class KronosPredictorUI:
    def __init__(self, parent_frame, log_callback=None):
        self.parent = parent_frame
        self.log_callback = log_callback or print
        self.model_wrapper = None
        self.last_predictions = None
        self.last_symbol = None
        self.prediction_thread = None

        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Label(main_frame, text="🔮 Kronos K线预测模型", font=('Arial', 14, 'bold'))
        header.pack(pady=(0, 10))

        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="预测参数", padding="10")
        input_frame.pack(fill=tk.X, pady=5)

        # Row 1: Symbol and Model Size
        row1 = ttk.Frame(input_frame)
        row1.pack(fill=tk.X, pady=5)

        ttk.Label(row1, text="股票代码:").pack(side=tk.LEFT, padx=5)
        self.symbol_entry = ttk.Entry(row1, width=15)
        self.symbol_entry.insert(0, "AAPL")
        self.symbol_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="模型大小:").pack(side=tk.LEFT, padx=5)
        self.model_size = ttk.Combobox(row1, width=10, state="readonly")
        self.model_size['values'] = ('large', 'base')
        self.model_size.current(1)  # Default to 'base'
        self.model_size.pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text="预测长度:").pack(side=tk.LEFT, padx=5)
        self.pred_len = ttk.Spinbox(row1, from_=5, to=120, increment=5, width=10)
        self.pred_len.set(30)
        self.pred_len.pack(side=tk.LEFT, padx=5)

        # Row 2: Period, Interval, Temperature
        row2 = ttk.Frame(input_frame)
        row2.pack(fill=tk.X, pady=5)

        ttk.Label(row2, text="历史周期:").pack(side=tk.LEFT, padx=5)
        self.period = ttk.Combobox(row2, width=10, state="readonly")
        self.period['values'] = ('1mo', '3mo', '6mo', '1y', '2y', '5y')
        self.period.current(1)  # Default to '3mo'
        self.period.pack(side=tk.LEFT, padx=5)

        ttk.Label(row2, text="时间间隔:").pack(side=tk.LEFT, padx=5)
        self.interval = ttk.Combobox(row2, width=10, state="readonly")
        self.interval['values'] = ('1m', '5m', '15m', '30m', '1h', '1d', '1wk')
        self.interval.current(5)  # Default to '1d'
        self.interval.pack(side=tk.LEFT, padx=5)

        ttk.Label(row2, text="Temperature:").pack(side=tk.LEFT, padx=5)
        self.temperature = ttk.Scale(row2, from_=0.1, to=2.0, orient=tk.HORIZONTAL, length=100)
        self.temperature.set(0.7)
        self.temperature.pack(side=tk.LEFT, padx=5)
        self.temp_label = ttk.Label(row2, text="0.7")
        self.temp_label.pack(side=tk.LEFT, padx=2)

        def update_temp_label(event=None):
            self.temp_label.config(text=f"{self.temperature.get():.1f}")
        self.temperature.configure(command=update_temp_label)

        # Control buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.predict_btn = ttk.Button(
            button_frame,
            text="🚀 生成预测",
            command=self._run_prediction
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)

        self.export_btn = ttk.Button(
            button_frame,
            text="💾 导出结果",
            command=self._export_predictions,
            state=tk.DISABLED
        )
        self.export_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(
            button_frame,
            text="🗑️ 清除结果",
            command=self._clear_results
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)

        # Status label
        self.status_label = ttk.Label(main_frame, text="就绪 (Polygon-only, US stocks)", foreground="green")
        self.status_label.pack(pady=5)

        # Results section with tabs
        results_notebook = ttk.Notebook(main_frame)
        results_notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Predictions tab
        pred_frame = ttk.Frame(results_notebook)
        results_notebook.add(pred_frame, text="预测结果")

        # Create Treeview for predictions
        columns = ('Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change%')
        self.pred_tree = ttk.Treeview(pred_frame, columns=columns, show='headings', height=15)

        for col in columns:
            self.pred_tree.heading(col, text=col)
            self.pred_tree.column(col, width=100)

        # Scrollbars for treeview
        pred_scroll_y = ttk.Scrollbar(pred_frame, orient=tk.VERTICAL, command=self.pred_tree.yview)
        pred_scroll_x = ttk.Scrollbar(pred_frame, orient=tk.HORIZONTAL, command=self.pred_tree.xview)
        self.pred_tree.configure(yscrollcommand=pred_scroll_y.set, xscrollcommand=pred_scroll_x.set)

        self.pred_tree.grid(row=0, column=0, sticky='nsew')
        pred_scroll_y.grid(row=0, column=1, sticky='ns')
        pred_scroll_x.grid(row=1, column=0, sticky='ew')

        pred_frame.grid_columnconfigure(0, weight=1)
        pred_frame.grid_rowconfigure(0, weight=1)

        # Statistics tab
        stats_frame = ttk.Frame(results_notebook)
        results_notebook.add(stats_frame, text="统计信息")

        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=15, width=60, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Log tab
        log_frame = ttk.Frame(results_notebook)
        results_notebook.add(log_frame, text="运行日志")

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=60, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _run_prediction(self):
        if self.prediction_thread and self.prediction_thread.is_alive():
            messagebox.showwarning("警告", "预测正在进行中，请等待...")
            return

        self.prediction_thread = threading.Thread(target=self._prediction_worker)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()

    def _prediction_worker(self):
        try:
            # Update UI state
            self.parent.after(0, self._set_running_state)

            # Get parameters
            symbol = self.symbol_entry.get().upper()
            model_size = self.model_size.get()
            pred_len = int(self.pred_len.get())
            period = self.period.get()
            interval = self.interval.get()
            temperature = float(self.temperature.get())

            self._log(f"开始预测 {symbol}...")

            # Initialize model if needed
            if self.model_wrapper is None or self.model_wrapper.config.model_size != model_size:
                self._log(f"初始化 {model_size} 模型...")
                config = KronosConfig(
                    model_size=model_size,
                    temperature=temperature,
                    pred_len=pred_len
                )
                self.model_wrapper = KronosModelWrapper(config)
            else:
                self.model_wrapper.config.temperature = temperature
                self.model_wrapper.config.pred_len = pred_len

            # Load model if not loaded
            if not self.model_wrapper.is_loaded:
                self._log("加载模型...")
                if not self.model_wrapper.load_model():
                    raise Exception("模型加载失败")

            # Fetch data
            self._log(f"获取 {symbol} 历史数据 (Polygon-only, 美股)...")
            df = prepare_kline_data(symbol, period, interval)
            if df is None or df.empty:
                raise Exception(f"无法获取 {symbol} 的Polygon数据（仅支持美股，且必须可从Polygon获取）")

            # Make predictions
            self._log(f"生成 {pred_len} 个预测 (数据频率 {interval})...")
            result = self.model_wrapper.predict(
                data=df[['open','high','low','close','volume']].values,
                timestamps=df.index.to_list(),
                pred_len=pred_len
            )

            if result['status'] != 'success':
                raise Exception(result.get('error', '预测失败'))

            # Format results
            predictions = result['predictions']
            try:
                # If model path returned DataFrame-like from original predictor
                pred_df = format_prediction_results(
                    predictions,
                    base_timestamp=df.index[-1],
                    interval=interval
                )
            except Exception:
                # Fallback path: predictions is numpy array
                pred_df = format_prediction_results(
                    predictions,
                    base_timestamp=df.index[-1],
                    interval=interval
                )

            self.last_predictions = pred_df
            self.last_symbol = symbol

            # Update UI with results
            self.parent.after(0, self._display_results, df, pred_df, symbol)

        except Exception as e:
            error_msg = str(e)
            self._log(f"错误: {error_msg}")
            self.parent.after(0, lambda: messagebox.showerror("预测错误", error_msg))
        finally:
            self.parent.after(0, self._set_ready_state)

    def _display_results(self, historical, predictions, symbol):
        # Clear previous results
        for item in self.pred_tree.get_children():
            self.pred_tree.delete(item)

        # Add predictions to tree
        for idx, row in predictions.iterrows():
            change = (row['close'] - predictions['close'].iloc[0]) / predictions['close'].iloc[0] * 100
            self.pred_tree.insert('', 'end', values=(
                idx.strftime('%Y-%m-%d %H:%M'),
                f"{row['open']:.2f}",
                f"{row['high']:.2f}",
                f"{row['low']:.2f}",
                f"{row['close']:.2f}",
                f"{row['volume']:.0f}",
                f"{change:.2f}%"
            ))

        # Update statistics
        self._update_statistics(historical, predictions, symbol)

        # Enable export button
        self.export_btn.config(state=tk.NORMAL)

        self._log(f"✅ 成功生成 {len(predictions)} 个预测")

        # Auto-save graphs to graph/ directory
        try:
            self._save_graphs(historical, predictions, symbol)
        except Exception as e:
            self._log(f"保存图表失败: {e}")

    def _update_statistics(self, historical, predictions, symbol):
        self.stats_text.delete('1.0', tk.END)

        stats_text = f"""
=== {symbol} 预测统计 ===

历史数据:
- 数据点数: {len(historical)}
- 起始日期: {historical.index[0].strftime('%Y-%m-%d')}
- 结束日期: {historical.index[-1].strftime('%Y-%m-%d')}
- 最后收盘价: ${historical['close'].iloc[-1]:.2f}

预测结果:
- 预测数量: {len(predictions)}
- 首个预测收盘价: ${predictions['close'].iloc[0]:.2f}
- 最后预测收盘价: ${predictions['close'].iloc[-1]:.2f}
- 总变化: {(predictions['close'].iloc[-1] - predictions['close'].iloc[0]) / predictions['close'].iloc[0] * 100:.2f}%

价格统计:
- 平均收盘价: ${predictions['close'].mean():.2f}
- 标准差: ${predictions['close'].std():.2f}
- 最高价: ${predictions['high'].max():.2f}
- 最低价: ${predictions['low'].min():.2f}
- 价格范围: ${predictions['high'].max() - predictions['low'].min():.2f}

成交量统计:
- 平均成交量: {predictions['volume'].mean():,.0f}
- 最大成交量: {predictions['volume'].max():,.0f}
- 最小成交量: {predictions['volume'].min():,.0f}
"""
        self.stats_text.insert('1.0', stats_text)

    def _export_predictions(self):
        if self.last_predictions is None:
            messagebox.showwarning("警告", "没有可导出的预测结果")
            return

        try:
            from tkinter import filedialog
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"kronos_{self.last_symbol}_{timestamp}.csv"
            )

            if filename:
                self.last_predictions.to_csv(filename)
                messagebox.showinfo("成功", f"预测结果已导出到:\n{filename}")
                self._log(f"导出成功: {filename}")
        except Exception as e:
            messagebox.showerror("导出错误", str(e))

    def _clear_results(self):
        for item in self.pred_tree.get_children():
            self.pred_tree.delete(item)
        self.stats_text.delete('1.0', tk.END)
        self.log_text.delete('1.0', tk.END)
        self.last_predictions = None
        self.last_symbol = None
        self.export_btn.config(state=tk.DISABLED)
        self.status_label.config(text="就绪", foreground="green")
        self._log("结果已清除")

    def _set_running_state(self):
        self.predict_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="运行中...", foreground="orange")

    def _set_ready_state(self):
        self.predict_btn.config(state=tk.NORMAL)
        self.progress.stop()
        self.status_label.config(text="就绪", foreground="green")

    def _log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"

        # Add to log text widget
        self.log_text.insert(tk.END, log_msg + "\n")
        self.log_text.see(tk.END)

        # Also use callback if provided
        if self.log_callback:
            self.log_callback(f"[Kronos] {message}")

    def _save_graphs(self, historical: pd.DataFrame, predictions: pd.DataFrame, symbol: str) -> None:
        """Generate and save Kronos graphs into graph/ folder (auto-created)."""
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        # Determine output directory at project root
        project_root = os.path.dirname(os.path.dirname(__file__))
        graph_dir = os.path.join(project_root, 'graph')
        os.makedirs(graph_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Try to infer interval from historical index frequency if possible
        interval_hint = 'custom'
        try:
            inferred = pd.infer_freq(historical.index)
            interval_hint = inferred or interval_hint
        except Exception:
            pass

        # 1) Main analysis chart (historical vs predicted OHLCV)
        from .utils import visualize_predictions
        fig_main = visualize_predictions(historical, predictions, title=f"Kronos - {symbol} ({interval_hint})")['figure']
        main_path = os.path.join(graph_dir, f'kronos_{symbol}_{interval_hint}_{timestamp}_analysis.html')
        fig_main.write_html(main_path)

        # 2) Closing price comparison
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(
            x=historical.index,
            y=historical['close'],
            name='Historical Close',
            line=dict(color='blue', width=2)
        ))
        fig_compare.add_trace(go.Scatter(
            x=predictions.index,
            y=predictions['close'],
            name='Predicted Close',
            line=dict(color='red', width=2, dash='dash')
        ))
        # Simple confidence band using predicted std
        std_dev = float(predictions['close'].std() * 0.1) if len(predictions) else 0.0
        if std_dev > 0:
            upper_band = predictions['close'] + std_dev
            lower_band = predictions['close'] - std_dev
            fig_compare.add_trace(go.Scatter(x=predictions.index, y=upper_band, fill=None, mode='lines', line_color='rgba(255,0,0,0)', showlegend=False))
            fig_compare.add_trace(go.Scatter(x=predictions.index, y=lower_band, fill='tonexty', mode='lines', line_color='rgba(255,0,0,0)', name='Confidence Band', fillcolor='rgba(255,0,0,0.2)'))
        fig_compare.update_layout(title=f'{symbol} - Historical vs Predicted Close', xaxis_title='Date', yaxis_title='Price', height=500, hovermode='x unified')
        compare_path = os.path.join(graph_dir, f'kronos_{symbol}_{interval_hint}_{timestamp}_comparison.html')
        fig_compare.write_html(compare_path)

        # 3) Performance metrics (distribution, cumulative, volatility, momentum)
        hist_returns = historical['close'].pct_change().dropna()
        pred_returns = predictions['close'].pct_change().dropna()
        fig_metrics = make_subplots(rows=2, cols=2, subplot_titles=('Daily Returns Distribution', 'Cumulative Returns', 'Volatility (5-step rolling)', 'Price Momentum'))
        # Distribution
        fig_metrics.add_trace(go.Histogram(x=hist_returns, name='Historical Returns', opacity=0.7, nbinsx=30), row=1, col=1)
        # Cumulative
        cumulative_hist = (1 + hist_returns).cumprod()
        fig_metrics.add_trace(go.Scatter(x=cumulative_hist.index, y=cumulative_hist, name='Historical Cumulative', line=dict(color='blue')), row=1, col=2)
        # Volatility
        volatility = hist_returns.rolling(window=5).std()
        fig_metrics.add_trace(go.Scatter(x=volatility.index, y=volatility, name='5-step Volatility', line=dict(color='orange')), row=2, col=1)
        # Momentum
        momentum = historical['close'].diff(5)
        fig_metrics.add_trace(go.Scatter(x=momentum.index, y=momentum, name='5-step Momentum', line=dict(color='green')), row=2, col=2)
        fig_metrics.update_layout(height=600, title_text=f"Performance Metrics - {symbol}", showlegend=True)
        metrics_path = os.path.join(graph_dir, f'kronos_{symbol}_{interval_hint}_{timestamp}_metrics.html')
        fig_metrics.write_html(metrics_path)

        self._log(f"图表已保存: {os.path.basename(main_path)}, {os.path.basename(compare_path)}, {os.path.basename(metrics_path)}")