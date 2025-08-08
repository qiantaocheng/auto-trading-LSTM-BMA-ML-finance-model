#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时监控仪表板
提供交易系统的实时P&L、风险指标和系统状态监控
"""

import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import os
import sqlite3
from flask import Flask, render_template_string, jsonify, request
import plotly.graph_objs as go
import plotly.utils


@dataclass
class DashboardMetrics:
    """仪表板指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 组合指标
    portfolio_value: float = 0.0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # 风险指标
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    var_1d: float = 0.0
    beta: float = 0.0
    
    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # 持仓信息
    positions_count: int = 0
    long_positions: int = 0
    short_positions: int = 0
    largest_position_pct: float = 0.0
    
    # 系统状态
    connection_status: str = "disconnected"
    trading_active: bool = False
    emergency_stop: bool = False
    last_update: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': self.portfolio_value,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'var_1d': self.var_1d,
            'beta': self.beta,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'positions_count': self.positions_count,
            'long_positions': self.long_positions,
            'short_positions': self.short_positions,
            'largest_position_pct': self.largest_position_pct,
            'connection_status': self.connection_status,
            'trading_active': self.trading_active,
            'emergency_stop': self.emergency_stop,
            'last_update': self.last_update.isoformat()
        }


class DataCollector:
    """数据收集器 - 从各个组件收集监控数据"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 数据源引用
        self.risk_manager = None
        self.order_manager = None
        self.connection_manager = None
        self.data_processor = None
        
        # 历史数据存储
        self.metrics_history = deque(maxlen=10000)  # 保存最近10000个数据点
        self.pnl_history = deque(maxlen=252)  # 1年交易日
        self.trade_history = []
        
        # 数据库
        self.db_file = config.get('db_file', 'monitoring/dashboard_data.db')
        self._init_database()
        
        # 更新频率
        self.update_interval = config.get('update_interval_seconds', 5)
        self.is_collecting = False
        self.collection_thread = None
    
    def set_components(self, risk_manager=None, order_manager=None, 
                      connection_manager=None, data_processor=None):
        """设置组件引用"""
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.connection_manager = connection_manager
        self.data_processor = data_processor
    
    def start_collection(self):
        """开始数据收集"""
        if self.collection_thread and self.collection_thread.is_alive():
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        self.logger.info("Data collection started")
    
    def stop_collection(self):
        """停止数据收集"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
    
    def _collection_loop(self):
        """数据收集循环"""
        while self.is_collecting:
            try:
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)
                
                # 保存到数据库
                self._save_metrics_to_db(metrics)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in data collection: {e}")
                time.sleep(self.update_interval)
    
    def _collect_current_metrics(self) -> DashboardMetrics:
        """收集当前指标"""
        metrics = DashboardMetrics()
        
        try:
            # 从风险管理器收集数据
            if self.risk_manager:
                risk_status = self.risk_manager.get_risk_status()
                metrics.portfolio_value = risk_status.get('portfolio_value', 0.0)
                metrics.total_pnl = risk_status.get('portfolio_pnl', 0.0)
                metrics.daily_pnl = risk_status.get('daily_pnl', 0.0)
                metrics.max_drawdown = risk_status.get('max_drawdown', 0.0)
                metrics.positions_count = risk_status.get('positions_count', 0)
                metrics.emergency_stop = risk_status.get('emergency_stop', False)
                metrics.trading_active = not risk_status.get('trading_suspended', True)
                
                # 计算持仓分布
                positions = getattr(self.risk_manager, 'current_positions', {})
                long_count = sum(1 for pos in positions.values() if pos.quantity > 0)
                short_count = sum(1 for pos in positions.values() if pos.quantity < 0)
                metrics.long_positions = long_count
                metrics.short_positions = short_count
                
                # 最大持仓占比
                if positions:
                    max_weight = max(abs(pos.portfolio_weight) for pos in positions.values())
                    metrics.largest_position_pct = max_weight * 100
            
            # 从订单管理器收集数据
            if self.order_manager:
                order_stats = self.order_manager.get_order_statistics()
                metrics.total_trades = order_stats.get('total_orders', 0)
                
                # 计算胜率
                filled_orders = self.order_manager.get_filled_orders()
                if filled_orders:
                    winning_orders = [o for o in filled_orders 
                                    if hasattr(o, 'realized_pnl') and o.realized_pnl > 0]
                    losing_orders = [o for o in filled_orders 
                                   if hasattr(o, 'realized_pnl') and o.realized_pnl < 0]
                    
                    metrics.winning_trades = len(winning_orders)
                    metrics.losing_trades = len(losing_orders)
                    
                    if metrics.total_trades > 0:
                        metrics.win_rate = (metrics.winning_trades / metrics.total_trades) * 100
                    
                    if winning_orders:
                        metrics.avg_win = np.mean([o.realized_pnl for o in winning_orders])
                    if losing_orders:
                        metrics.avg_loss = np.mean([o.realized_pnl for o in losing_orders])
            
            # 从连接管理器收集数据
            if self.connection_manager:
                conn_status = self.connection_manager.get_status()
                metrics.connection_status = "connected" if conn_status.get('connected', False) else "disconnected"
            
            # 计算技术指标
            self._calculate_technical_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return metrics
    
    def _calculate_technical_metrics(self, metrics: DashboardMetrics):
        """计算技术指标"""
        try:
            # 添加到PnL历史
            if metrics.total_pnl != 0:
                self.pnl_history.append(metrics.total_pnl)
            
            if len(self.pnl_history) > 30:  # 至少30个数据点
                pnl_array = np.array(list(self.pnl_history))
                
                # 计算收益率
                returns = np.diff(pnl_array) / pnl_array[:-1]
                returns = returns[~np.isnan(returns)]  # 移除NaN
                
                if len(returns) > 0:
                    # Sharpe比率 (假设无风险利率为0)
                    if np.std(returns) > 0:
                        metrics.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
                    
                    # VaR (95%置信度)
                    if len(returns) >= 10:
                        metrics.var_1d = np.percentile(returns, 5) * metrics.portfolio_value
                    
                    # Beta (相对于基准，这里简化处理)
                    # 实际应用中需要基准指数数据
                    metrics.beta = 1.0  # 简化为1
            
        except Exception as e:
            self.logger.error(f"Error calculating technical metrics: {e}")
    
    def get_latest_metrics(self) -> Optional[DashboardMetrics]:
        """获取最新指标"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_historical_data(self, hours: int = 24) -> List[DashboardMetrics]:
        """获取历史数据"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def _init_database(self):
        """初始化数据库"""
        try:
            os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
            
            conn = sqlite3.connect(self.db_file)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    portfolio_value REAL,
                    total_pnl REAL,
                    daily_pnl REAL,
                    max_drawdown REAL,
                    positions_count INTEGER,
                    win_rate REAL,
                    connection_status TEXT,
                    trading_active BOOLEAN,
                    emergency_stop BOOLEAN
                )
            ''')
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    def _save_metrics_to_db(self, metrics: DashboardMetrics):
        """保存指标到数据库"""
        try:
            conn = sqlite3.connect(self.db_file)
            conn.execute('''
                INSERT INTO metrics (
                    timestamp, portfolio_value, total_pnl, daily_pnl, max_drawdown,
                    positions_count, win_rate, connection_status, trading_active, emergency_stop
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.portfolio_value,
                metrics.total_pnl,
                metrics.daily_pnl,
                metrics.max_drawdown,
                metrics.positions_count,
                metrics.win_rate,
                metrics.connection_status,
                metrics.trading_active,
                metrics.emergency_stop
            ))
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving metrics to database: {e}")


class WebDashboard:
    """Web仪表板"""
    
    def __init__(self, data_collector: DataCollector, config: Dict[str, Any] = None):
        self.data_collector = data_collector
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Flask应用
        self.app = Flask(__name__)
        self.app.secret_key = config.get('secret_key', 'your-secret-key-here')
        
        # 配置
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 5000)
        self.debug = config.get('debug', False)
        
        # 设置路由
        self._setup_routes()
        
        # 服务器线程
        self.server_thread = None
        self.is_running = False
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.route('/')
        def dashboard():
            """主仪表板页面"""
            return render_template_string(DASHBOARD_HTML_TEMPLATE)
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """获取当前指标"""
            metrics = self.data_collector.get_latest_metrics()
            if metrics:
                return jsonify(metrics.to_dict())
            return jsonify({'error': 'No data available'})
        
        @self.app.route('/api/historical')
        def get_historical():
            """获取历史数据"""
            hours = request.args.get('hours', 24, type=int)
            data = self.data_collector.get_historical_data(hours)
            return jsonify([m.to_dict() for m in data])
        
        @self.app.route('/api/charts/pnl')
        def get_pnl_chart():
            """获取PnL图表数据"""
            data = self.data_collector.get_historical_data(24)
            
            timestamps = [m.timestamp.isoformat() for m in data]
            total_pnl = [m.total_pnl for m in data]
            daily_pnl = [m.daily_pnl for m in data]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timestamps, y=total_pnl, name='Total P&L', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=timestamps, y=daily_pnl, name='Daily P&L', line=dict(color='green')))
            
            fig.update_layout(
                title='P&L Trend',
                xaxis_title='Time',
                yaxis_title='P&L ($)',
                height=400
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        @self.app.route('/api/charts/drawdown')
        def get_drawdown_chart():
            """获取回撤图表"""
            data = self.data_collector.get_historical_data(24)
            
            timestamps = [m.timestamp.isoformat() for m in data]
            drawdowns = [m.max_drawdown for m in data]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timestamps, y=drawdowns, name='Max Drawdown', 
                                   fill='tonexty', line=dict(color='red')))
            
            fig.update_layout(
                title='Drawdown Analysis',
                xaxis_title='Time',
                yaxis_title='Drawdown (%)',
                height=400
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        @self.app.route('/api/charts/positions')
        def get_positions_chart():
            """获取持仓分布图表"""
            metrics = self.data_collector.get_latest_metrics()
            if not metrics:
                return jsonify({'error': 'No data available'})
            
            labels = ['Long', 'Short']
            values = [metrics.long_positions, metrics.short_positions]
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
            fig.update_layout(title='Position Distribution', height=400)
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def start_server(self):
        """启动Web服务器"""
        if self.server_thread and self.server_thread.is_alive():
            return
        
        self.is_running = True
        self.server_thread = threading.Thread(
            target=lambda: self.app.run(host=self.host, port=self.port, debug=self.debug, use_reloader=False),
            daemon=True
        )
        self.server_thread.start()
        self.logger.info(f"Dashboard server started at http://{self.host}:{self.port}")
    
    def stop_server(self):
        """停止Web服务器"""
        self.is_running = False
        # Flask服务器停止需要外部信号，这里只是标记


# HTML模板
DASHBOARD_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .neutral { color: #34495e; }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-connected { background-color: #27ae60; }
        .status-disconnected { background-color: #e74c3c; }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .refresh-button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin-bottom: 20px;
        }
        .refresh-button:hover {
            background-color: #2980b9;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Trading System Dashboard</h1>
        <p>Real-time monitoring of P&L, risk metrics, and system status</p>
        <button class="refresh-button" onclick="refreshData()">Refresh Data</button>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value neutral" id="portfolio-value">$0</div>
            <div class="metric-label">Portfolio Value</div>
        </div>
        <div class="metric-card">
            <div class="metric-value neutral" id="total-pnl">$0</div>
            <div class="metric-label">Total P&L</div>
        </div>
        <div class="metric-card">
            <div class="metric-value neutral" id="daily-pnl">$0</div>
            <div class="metric-label">Daily P&L</div>
        </div>
        <div class="metric-card">
            <div class="metric-value neutral" id="max-drawdown">0%</div>
            <div class="metric-label">Max Drawdown</div>
        </div>
        <div class="metric-card">
            <div class="metric-value neutral" id="win-rate">0%</div>
            <div class="metric-label">Win Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value neutral" id="positions-count">0</div>
            <div class="metric-label">Open Positions</div>
        </div>
        <div class="metric-card">
            <div class="metric-value neutral" id="connection-status">
                <span class="status-indicator status-disconnected"></span>Disconnected
            </div>
            <div class="metric-label">Connection Status</div>
        </div>
        <div class="metric-card">
            <div class="metric-value neutral" id="trading-status">Inactive</div>
            <div class="metric-label">Trading Status</div>
        </div>
    </div>

    <div class="charts-grid">
        <div class="chart-container">
            <div id="pnl-chart"></div>
        </div>
        <div class="chart-container">
            <div id="drawdown-chart"></div>
        </div>
        <div class="chart-container">
            <div id="positions-chart"></div>
        </div>
    </div>

    <script>
        function formatCurrency(value) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(value);
        }

        function formatPercentage(value) {
            return (value).toFixed(2) + '%';
        }

        function updateMetrics(data) {
            // Update metric values
            document.getElementById('portfolio-value').textContent = formatCurrency(data.portfolio_value);
            document.getElementById('total-pnl').textContent = formatCurrency(data.total_pnl);
            document.getElementById('daily-pnl').textContent = formatCurrency(data.daily_pnl);
            document.getElementById('max-drawdown').textContent = formatPercentage(data.max_drawdown);
            document.getElementById('win-rate').textContent = formatPercentage(data.win_rate);
            document.getElementById('positions-count').textContent = data.positions_count;

            // Update colors based on values
            const totalPnlElement = document.getElementById('total-pnl');
            totalPnlElement.className = 'metric-value ' + (data.total_pnl >= 0 ? 'positive' : 'negative');

            const dailyPnlElement = document.getElementById('daily-pnl');
            dailyPnlElement.className = 'metric-value ' + (data.daily_pnl >= 0 ? 'positive' : 'negative');

            // Update connection status
            const connectionElement = document.getElementById('connection-status');
            const statusIndicator = connectionElement.querySelector('.status-indicator');
            if (data.connection_status === 'connected') {
                connectionElement.innerHTML = '<span class="status-indicator status-connected"></span>Connected';
                statusIndicator.className = 'status-indicator status-connected';
            } else {
                connectionElement.innerHTML = '<span class="status-indicator status-disconnected"></span>Disconnected';
                statusIndicator.className = 'status-indicator status-disconnected';
            }

            // Update trading status
            const tradingElement = document.getElementById('trading-status');
            tradingElement.textContent = data.trading_active ? 'Active' : 'Inactive';
            tradingElement.className = 'metric-value ' + (data.trading_active ? 'positive' : 'negative');
        }

        function loadCharts() {
            // Load P&L chart
            fetch('/api/charts/pnl')
                .then(response => response.json())
                .then(data => {
                    Plotly.newPlot('pnl-chart', data.data, data.layout);
                });

            // Load Drawdown chart
            fetch('/api/charts/drawdown')
                .then(response => response.json())
                .then(data => {
                    Plotly.newPlot('drawdown-chart', data.data, data.layout);
                });

            // Load Positions chart
            fetch('/api/charts/positions')
                .then(response => response.json())
                .then(data => {
                    Plotly.newPlot('positions-chart', data.data, data.layout);
                });
        }

        function refreshData() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        console.error('Error loading metrics:', data.error);
                        return;
                    }
                    updateMetrics(data);
                })
                .catch(error => {
                    console.error('Error fetching metrics:', error);
                });

            loadCharts();
        }

        // Initial load
        refreshData();

        // Auto refresh every 5 seconds
        setInterval(refreshData, 5000);
    </script>
</body>
</html>
'''


class MonitoringSystem:
    """完整的监控系统"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        collector_config = config.get('collector', {})
        self.data_collector = DataCollector(collector_config)
        
        dashboard_config = config.get('dashboard', {})
        self.web_dashboard = WebDashboard(self.data_collector, dashboard_config)
        
        # 状态
        self.is_running = False
    
    def set_trading_components(self, risk_manager=None, order_manager=None, 
                             connection_manager=None, data_processor=None):
        """设置交易系统组件"""
        self.data_collector.set_components(
            risk_manager=risk_manager,
            order_manager=order_manager,
            connection_manager=connection_manager,
            data_processor=data_processor
        )
    
    def start(self):
        """启动监控系统"""
        if self.is_running:
            return
        
        self.data_collector.start_collection()
        self.web_dashboard.start_server()
        
        self.is_running = True
        self.logger.info("Monitoring system started")
    
    def stop(self):
        """停止监控系统"""
        if not self.is_running:
            return
        
        self.data_collector.stop_collection()
        self.web_dashboard.stop_server()
        
        self.is_running = False
        self.logger.info("Monitoring system stopped")
    
    def get_dashboard_url(self) -> str:
        """获取仪表板URL"""
        return f"http://{self.web_dashboard.host}:{self.web_dashboard.port}"


# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 配置
    config = {
        'collector': {
            'db_file': 'monitoring/dashboard_data.db',
            'update_interval_seconds': 5
        },
        'dashboard': {
            'host': '127.0.0.1',
            'port': 5000,
            'debug': False,
            'secret_key': 'your-secret-key-here'
        }
    }
    
    # 创建监控系统
    monitoring = MonitoringSystem(config)
    
    # 启动系统
    monitoring.start()
    
    print(f"Dashboard available at: {monitoring.get_dashboard_url()}")
    
    try:
        # 保持运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitoring.stop()
        print("Monitoring system stopped")