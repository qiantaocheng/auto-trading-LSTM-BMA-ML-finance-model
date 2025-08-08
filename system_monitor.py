#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统监控器 - 监视交易软件运行状态
当软件开始运行时自动启动监控，负责观察系统状态和日志输出
"""

import os
import sys
import time
import logging
import threading
import psutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import queue
import subprocess
from dataclasses import dataclass, field
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    active_connections: int = 0
    trading_active: bool = False
    model_running: bool = False
    error_count: int = 0
    warning_count: int = 0


class LogMonitor:
    """日志监控器"""
    
    def __init__(self, log_files: List[str]):
        self.log_files = log_files
        self.logger = logging.getLogger(__name__)
        self.log_queue = queue.Queue()
        self.is_monitoring = False
        self.monitor_threads = []
        
        # 错误模式匹配
        self.error_patterns = [
            'ERROR', 'CRITICAL', 'FAILED', 'EXCEPTION', 'TIMEOUT',
            '❌', 'Connection lost', 'Order rejected', 'Risk limit exceeded'
        ]
        
        self.warning_patterns = [
            'WARNING', 'WARN', '⚠️', 'Retry', 'Reconnecting',
            'Position size alert', 'Market data delayed'
        ]
        
        # 统计信息
        self.stats = {
            'total_lines': 0,
            'error_count': 0,
            'warning_count': 0,
            'last_activity': None
        }
    
    def start_monitoring(self):
        """开始监控日志文件"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        for log_file in self.log_files:
            if os.path.exists(log_file):
                thread = threading.Thread(
                    target=self._monitor_log_file,
                    args=(log_file,),
                    daemon=True
                )
                thread.start()
                self.monitor_threads.append(thread)
                self.logger.info(f"📄 开始监控日志文件: {log_file}")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        self.logger.info("🛑 日志监控已停止")
    
    def _monitor_log_file(self, log_file: str):
        """监控单个日志文件"""
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                # 移动到文件末尾
                f.seek(0, 2)
                
                while self.is_monitoring:
                    line = f.readline()
                    if line:
                        self._process_log_line(line.strip(), log_file)
                        self.stats['total_lines'] += 1
                        self.stats['last_activity'] = datetime.now()
                    else:
                        time.sleep(0.1)  # 等待新内容
                        
        except Exception as e:
            self.logger.error(f"❌ 监控日志文件 {log_file} 失败: {e}")
    
    def _process_log_line(self, line: str, source_file: str):
        """处理日志行"""
        try:
            # 检查错误模式
            is_error = any(pattern in line for pattern in self.error_patterns)
            is_warning = any(pattern in line for pattern in self.warning_patterns)
            
            if is_error:
                self.stats['error_count'] += 1
                level = 'ERROR'
            elif is_warning:
                self.stats['warning_count'] += 1
                level = 'WARNING'
            else:
                level = 'INFO'
            
            # 添加到队列
            log_entry = {
                'timestamp': datetime.now(),
                'level': level,
                'message': line,
                'source': os.path.basename(source_file)
            }
            
            self.log_queue.put(log_entry)
            
        except Exception as e:
            self.logger.error(f"❌ 处理日志行失败: {e}")
    
    def get_recent_logs(self, count: int = 100) -> List[Dict]:
        """获取最近的日志"""
        logs = []
        temp_queue = queue.Queue()
        
        # 从队列中取出所有日志
        while not self.log_queue.empty() and len(logs) < count:
            try:
                log_entry = self.log_queue.get_nowait()
                logs.append(log_entry)
                temp_queue.put(log_entry)
            except queue.Empty:
                break
        
        # 将日志放回队列
        while not temp_queue.empty():
            self.log_queue.put(temp_queue.get())
        
        return logs[-count:]


class ProcessMonitor:
    """进程监控器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitored_processes = {}
        self.system_metrics_history = []
        
    def add_process(self, name: str, pattern: str):
        """添加要监控的进程"""
        self.monitored_processes[name] = {
            'pattern': pattern,
            'pid': None,
            'process': None,
            'active': False,
            'start_time': None,
            'cpu_history': [],
            'memory_history': []
        }
    
    def update_processes(self):
        """更新进程状态"""
        for name, info in self.monitored_processes.items():
            try:
                # 查找进程
                found_process = None
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if info['pattern'] in ' '.join(proc.info['cmdline'] or []):
                            found_process = proc
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if found_process:
                    if info['pid'] != found_process.pid:
                        # 新进程
                        info['pid'] = found_process.pid
                        info['process'] = found_process
                        info['active'] = True
                        info['start_time'] = datetime.now()
                        self.logger.info(f"🚀 检测到进程启动: {name} (PID: {found_process.pid})")
                    
                    # 更新进程指标
                    try:
                        cpu_percent = found_process.cpu_percent()
                        memory_info = found_process.memory_info()
                        memory_mb = memory_info.rss / 1024 / 1024
                        
                        info['cpu_history'].append(cpu_percent)
                        info['memory_history'].append(memory_mb)
                        
                        # 保持历史数据大小
                        if len(info['cpu_history']) > 100:
                            info['cpu_history'] = info['cpu_history'][-100:]
                        if len(info['memory_history']) > 100:
                            info['memory_history'] = info['memory_history'][-100:]
                            
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                else:
                    if info['active']:
                        # 进程已停止
                        info['active'] = False
                        info['pid'] = None
                        info['process'] = None
                        self.logger.info(f"⏹️ 检测到进程停止: {name}")
                
            except Exception as e:
                self.logger.error(f"❌ 更新进程 {name} 状态失败: {e}")
    
    def get_system_metrics(self) -> SystemMetrics:
        """获取系统指标"""
        try:
            # CPU和内存
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # 磁盘使用
            disk = psutil.disk_usage('/')
            
            # 网络
            network = psutil.net_io_counters()
            
            # 网络连接数
            connections = len(psutil.net_connections())
            
            # 检查交易相关进程
            trading_active = any(
                info['active'] for info in self.monitored_processes.values()
                if 'trading' in info['pattern'].lower()
            )
            
            model_running = any(
                info['active'] for info in self.monitored_processes.values()
                if any(keyword in info['pattern'].lower() for keyword in ['bma', 'lstm', 'model'])
            )
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                disk_usage_percent=disk.percent,
                network_sent_mb=network.bytes_sent / 1024 / 1024,
                network_recv_mb=network.bytes_recv / 1024 / 1024,
                active_connections=connections,
                trading_active=trading_active,
                model_running=model_running
            )
            
            # 保存历史数据
            self.system_metrics_history.append(metrics)
            if len(self.system_metrics_history) > 1000:
                self.system_metrics_history = self.system_metrics_history[-1000:]
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ 获取系统指标失败: {e}")
            return SystemMetrics()


class MonitoringGUI:
    """监控界面"""
    
    def __init__(self, log_monitor: LogMonitor, process_monitor: ProcessMonitor):
        self.log_monitor = log_monitor
        self.process_monitor = process_monitor
        self.root = None
        self.is_running = False
        
        # GUI组件
        self.log_text = None
        self.status_labels = {}
        self.chart_canvas = None
        self.figure = None
        
    def create_gui(self):
        """创建GUI界面"""
        self.root = tk.Tk()
        self.root.title("交易系统监控器")
        self.root.geometry("1200x800")
        
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建笔记本控件
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 系统状态页面
        self._create_status_tab(notebook)
        
        # 日志监控页面
        self._create_logs_tab(notebook)
        
        # 性能监控页面
        self._create_performance_tab(notebook)
        
        # 进程管理页面
        self._create_process_tab(notebook)
        
        # 启动更新循环
        self._start_update_loop()
    
    def _create_status_tab(self, notebook):
        """创建状态页面"""
        status_frame = ttk.Frame(notebook)
        notebook.add(status_frame, text="系统状态")
        
        # 状态指示器
        status_info_frame = ttk.LabelFrame(status_frame, text="系统状态", padding=10)
        status_info_frame.pack(fill=tk.X, pady=5)
        
        # 创建状态标签
        status_items = [
            ('trading_system', '交易系统'),
            ('bma_model', 'BMA模型'),
            ('lstm_model', 'LSTM模型'),
            ('connection', 'IBKR连接'),
            ('data_feed', '数据源'),
            ('risk_monitor', '风险监控')
        ]
        
        for i, (key, label) in enumerate(status_items):
            row = i // 3
            col = i % 3
            
            frame = ttk.Frame(status_info_frame)
            frame.grid(row=row, column=col, padx=10, pady=5, sticky=tk.W)
            
            ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT)
            status_label = ttk.Label(frame, text="●", foreground="red", font=("Arial", 16))
            status_label.pack(side=tk.LEFT, padx=5)
            
            self.status_labels[key] = status_label
        
        # 系统指标
        metrics_frame = ttk.LabelFrame(status_frame, text="系统指标", padding=10)
        metrics_frame.pack(fill=tk.X, pady=5)
        
        metrics_items = [
            ('cpu', 'CPU使用率'),
            ('memory', '内存使用率'),
            ('disk', '磁盘使用率'),
            ('network', '网络连接'),
            ('errors', '错误计数'),
            ('warnings', '警告计数')
        ]
        
        for i, (key, label) in enumerate(metrics_items):
            row = i // 3
            col = i % 3
            
            frame = ttk.Frame(metrics_frame)
            frame.grid(row=row, column=col, padx=10, pady=5, sticky=tk.W)
            
            ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT)
            metric_label = ttk.Label(frame, text="N/A", font=("Arial", 10, "bold"))
            metric_label.pack(side=tk.LEFT, padx=5)
            
            self.status_labels[f"metric_{key}"] = metric_label
    
    def _create_logs_tab(self, notebook):
        """创建日志页面"""
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="实时日志")
        
        # 日志显示区域
        self.log_text = scrolledtext.ScrolledText(
            logs_frame, 
            wrap=tk.WORD, 
            width=100, 
            height=30,
            font=("Consolas", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 控制按钮
        button_frame = ttk.Frame(logs_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="清空日志", command=self._clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="保存日志", command=self._save_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="暂停/继续", command=self._toggle_log_update).pack(side=tk.LEFT, padx=5)
        
        # 日志级别过滤
        ttk.Label(button_frame, text="级别过滤:").pack(side=tk.LEFT, padx=(20, 5))
        self.log_level_var = tk.StringVar(value="ALL")
        level_combo = ttk.Combobox(button_frame, textvariable=self.log_level_var, 
                                  values=["ALL", "ERROR", "WARNING", "INFO"], width=8)
        level_combo.pack(side=tk.LEFT, padx=5)
    
    def _create_performance_tab(self, notebook):
        """创建性能页面"""
        perf_frame = ttk.Frame(notebook)
        notebook.add(perf_frame, text="性能监控")
        
        # 创建图表
        self.figure = plt.Figure(figsize=(12, 8), dpi=100)
        self.chart_canvas = FigureCanvasTkAgg(self.figure, perf_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建子图
        self.cpu_ax = self.figure.add_subplot(2, 2, 1)
        self.memory_ax = self.figure.add_subplot(2, 2, 2)
        self.network_ax = self.figure.add_subplot(2, 2, 3)
        self.process_ax = self.figure.add_subplot(2, 2, 4)
        
        self.figure.tight_layout()
    
    def _create_process_tab(self, notebook):
        """创建进程页面"""
        process_frame = ttk.Frame(notebook)
        notebook.add(process_frame, text="进程管理")
        
        # 进程列表
        process_tree = ttk.Treeview(process_frame, columns=('pid', 'status', 'cpu', 'memory', 'start_time'), show='tree headings')
        
        process_tree.heading('#0', text='进程名称')
        process_tree.heading('pid', text='PID')
        process_tree.heading('status', text='状态')
        process_tree.heading('cpu', text='CPU%')
        process_tree.heading('memory', text='内存(MB)')
        process_tree.heading('start_time', text='启动时间')
        
        process_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.process_tree = process_tree
    
    def _start_update_loop(self):
        """启动更新循环"""
        self.is_running = True
        self._update_gui()
    
    def _update_gui(self):
        """更新GUI界面"""
        if not self.is_running:
            return
        
        try:
            # 更新系统状态
            self._update_status()
            
            # 更新日志
            self._update_logs()
            
            # 更新性能图表
            self._update_performance_charts()
            
            # 更新进程列表
            self._update_process_list()
            
        except Exception as e:
            print(f"GUI更新错误: {e}")
        
        # 每秒更新一次
        self.root.after(1000, self._update_gui)
    
    def _update_status(self):
        """更新状态指示器"""
        try:
            # 获取系统指标
            metrics = self.process_monitor.get_system_metrics()
            
            # 更新状态灯
            status_map = {
                'trading_system': metrics.trading_active,
                'bma_model': metrics.model_running,
                'lstm_model': metrics.model_running,
                'connection': metrics.active_connections > 0,
                'data_feed': True,  # 简化处理
                'risk_monitor': metrics.trading_active
            }
            
            for key, status in status_map.items():
                if key in self.status_labels:
                    color = "green" if status else "red"
                    self.status_labels[key].config(foreground=color)
            
            # 更新指标
            self.status_labels['metric_cpu'].config(text=f"{metrics.cpu_percent:.1f}%")
            self.status_labels['metric_memory'].config(text=f"{metrics.memory_percent:.1f}%")
            self.status_labels['metric_disk'].config(text=f"{metrics.disk_usage_percent:.1f}%")
            self.status_labels['metric_network'].config(text=f"{metrics.active_connections}")
            self.status_labels['metric_errors'].config(text=f"{self.log_monitor.stats['error_count']}")
            self.status_labels['metric_warnings'].config(text=f"{self.log_monitor.stats['warning_count']}")
            
        except Exception as e:
            print(f"状态更新错误: {e}")
    
    def _update_logs(self):
        """更新日志显示"""
        try:
            recent_logs = self.log_monitor.get_recent_logs(50)
            
            # 获取当前过滤级别
            filter_level = self.log_level_var.get()
            
            # 过滤日志
            if filter_level != "ALL":
                recent_logs = [log for log in recent_logs if log['level'] == filter_level]
            
            # 限制显示的日志数量
            if len(recent_logs) > 20:
                recent_logs = recent_logs[-20:]
            
            # 更新显示
            for log in recent_logs:
                timestamp = log['timestamp'].strftime('%H:%M:%S')
                level = log['level']
                message = log['message']
                source = log['source']
                
                # 颜色编码
                if level == 'ERROR':
                    color = 'red'
                elif level == 'WARNING':
                    color = 'orange'
                else:
                    color = 'black'
                
                log_line = f"[{timestamp}] [{level}] [{source}] {message}\n"
                
                # 插入日志
                self.log_text.insert(tk.END, log_line)
                self.log_text.see(tk.END)
                
                # 限制文本框内容长度
                if int(self.log_text.index(tk.END).split('.')[0]) > 1000:
                    self.log_text.delete(1.0, "100.0")
            
        except Exception as e:
            print(f"日志更新错误: {e}")
    
    def _update_performance_charts(self):
        """更新性能图表"""
        try:
            if len(self.process_monitor.system_metrics_history) == 0:
                return
            
            # 获取最近的数据
            recent_metrics = self.process_monitor.system_metrics_history[-60:]  # 最近60个数据点
            
            timestamps = [m.timestamp for m in recent_metrics]
            cpu_data = [m.cpu_percent for m in recent_metrics]
            memory_data = [m.memory_percent for m in recent_metrics]
            network_sent = [m.network_sent_mb for m in recent_metrics]
            network_recv = [m.network_recv_mb for m in recent_metrics]
            
            # 清除旧图表
            self.cpu_ax.clear()
            self.memory_ax.clear()
            self.network_ax.clear()
            self.process_ax.clear()
            
            # CPU图表
            self.cpu_ax.plot(timestamps, cpu_data, 'b-', label='CPU %')
            self.cpu_ax.set_title('CPU使用率')
            self.cpu_ax.set_ylabel('百分比')
            self.cpu_ax.grid(True)
            
            # 内存图表
            self.memory_ax.plot(timestamps, memory_data, 'r-', label='Memory %')
            self.memory_ax.set_title('内存使用率')
            self.memory_ax.set_ylabel('百分比')
            self.memory_ax.grid(True)
            
            # 网络图表
            self.network_ax.plot(timestamps, network_sent, 'g-', label='发送')
            self.network_ax.plot(timestamps, network_recv, 'm-', label='接收')
            self.network_ax.set_title('网络流量')
            self.network_ax.set_ylabel('MB')
            self.network_ax.legend()
            self.network_ax.grid(True)
            
            # 进程状态图表
            active_processes = len([p for p in self.process_monitor.monitored_processes.values() if p['active']])
            total_processes = len(self.process_monitor.monitored_processes)
            
            labels = ['活跃', '非活跃']
            sizes = [active_processes, total_processes - active_processes]
            colors = ['green', 'red']
            
            self.process_ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            self.process_ax.set_title('进程状态')
            
            # 更新图表
            self.chart_canvas.draw()
            
        except Exception as e:
            print(f"图表更新错误: {e}")
    
    def _update_process_list(self):
        """更新进程列表"""
        try:
            # 清空现有项
            for item in self.process_tree.get_children():
                self.process_tree.delete(item)
            
            # 添加进程信息
            for name, info in self.process_monitor.monitored_processes.items():
                status = "运行中" if info['active'] else "已停止"
                pid = info['pid'] if info['pid'] else "N/A"
                
                cpu_usage = f"{info['cpu_history'][-1]:.1f}" if info['cpu_history'] else "0.0"
                memory_usage = f"{info['memory_history'][-1]:.1f}" if info['memory_history'] else "0.0"
                start_time = info['start_time'].strftime('%H:%M:%S') if info['start_time'] else "N/A"
                
                self.process_tree.insert('', tk.END, text=name, values=(
                    pid, status, cpu_usage, memory_usage, start_time
                ))
                
        except Exception as e:
            print(f"进程列表更新错误: {e}")
    
    def _clear_logs(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
    
    def _save_logs(self):
        """保存日志"""
        try:
            logs_content = self.log_text.get(1.0, tk.END)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"monitor_logs_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(logs_content)
            
            print(f"日志已保存到: {filename}")
            
        except Exception as e:
            print(f"保存日志失败: {e}")
    
    def _toggle_log_update(self):
        """切换日志更新状态"""
        # 简化实现
        pass
    
    def run(self):
        """运行GUI"""
        self.create_gui()
        self.root.mainloop()
    
    def stop(self):
        """停止GUI"""
        self.is_running = False
        if self.root:
            self.root.quit()


class SystemMonitor:
    """系统监控器主类"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 日志文件配置
        log_files = self.config.get('log_files', [
            'logs/integrated_system.log',
            'logs/trading_manager.log',
            'logs/model_validator.log',
            'logs/stock_universe.log'
        ])
        
        # 创建监控组件
        self.log_monitor = LogMonitor(log_files)
        self.process_monitor = ProcessMonitor()
        
        # 添加要监控的进程
        self.process_monitor.add_process('Trading System', 'integrated_trading_system.py')
        self.process_monitor.add_process('BMA Model', 'bma')
        self.process_monitor.add_process('LSTM Model', 'lstm')
        self.process_monitor.add_process('Stock Universe', 'stock_universe_manager.py')
        
        # GUI
        self.gui = MonitoringGUI(self.log_monitor, self.process_monitor)
        
        # 更新线程
        self.update_thread = None
        self.is_running = False
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动日志监控
        self.log_monitor.start_monitoring()
        
        # 启动进程更新线程
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self.logger.info("🚀 系统监控器已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        
        # 停止日志监控
        self.log_monitor.stop_monitoring()
        
        # 停止GUI
        self.gui.stop()
        
        self.logger.info("🛑 系统监控器已停止")
    
    def _update_loop(self):
        """更新循环"""
        while self.is_running:
            try:
                # 更新进程状态
                self.process_monitor.update_processes()
                
                # 获取系统指标
                self.process_monitor.get_system_metrics()
                
                time.sleep(1)  # 每秒更新一次
                
            except Exception as e:
                self.logger.error(f"❌ 更新循环错误: {e}")
                time.sleep(5)
    
    def run_gui(self):
        """运行GUI界面"""
        self.start_monitoring()
        self.gui.run()


def setup_logging():
    """设置日志"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"system_monitor_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 系统监控器启动")
    
    try:
        # 配置
        config = {
            'log_files': [
                'logs/integrated_system.log',
                'logs/trading_manager.log', 
                'logs/model_validator.log',
                'logs/stock_universe.log',
                'logs/auto_update.log'
            ]
        }
        
        # 创建监控器
        monitor = SystemMonitor(config)
        
        # 运行GUI
        monitor.run_gui()
        
    except KeyboardInterrupt:
        logger.info("👋 用户中断")
    except Exception as e:
        logger.error(f"❌ 程序异常: {e}")
    
    logger.info("🛑 系统监控器已退出")


if __name__ == "__main__":
    main()