#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风控收益平衡器GUI控制面板
提供一键启用/禁用和参数配置界面
"""

import sys
import os
import logging
from typing import Dict, Any
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
from datetime import datetime
import threading

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from ibkr_risk_balancer_adapter import (
        get_risk_balancer_adapter, 
        enable_risk_balancer,
        disable_risk_balancer,
        is_risk_balancer_enabled
    )
    from risk_reward_balancer_integrated import Config, RealtimeGuards, SizingConfig, DegradePolicy, ThrottleConfig
except ImportError as e:
    print(f"导入风控模块失败: {e}")

logger = logging.getLogger(__name__)

class RiskBalancerGUIPanel:
    """风控收益平衡器GUI控制面板"""
    
    def __init__(self, parent=None):
        """
        初始化GUI面板
        
        Args:
            parent: 父窗口，如果为None则创建独立窗口
        """
        self.parent = parent
        self.adapter = None
        self.config_file = "config/risk_balancer_config.json"
        
        # 创建窗口
        if parent:
            self.window = ttk.Frame(parent)
        else:
            self.window = tk.Tk()
            self.window.title("风控收益平衡器控制面板")
            self.window.geometry("800x600")
        
        # 配置变量
        self.enabled_var = tk.BooleanVar()
        self.config_vars = {}
        
        # 状态变量
        self.status_text = ""
        self.last_update = datetime.now()
        
        self._create_widgets()
        self._load_config()
        self._initialize_adapter()
        self._update_status()
        
        # 启动状态更新线程
        self._start_update_thread()
    
    def _create_widgets(self):
        """创建GUI组件"""
        # 主框架
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 1. 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 启用/禁用控制
        ttk.Label(control_frame, text="风控收益平衡器:").pack(side=tk.LEFT)
        
        enable_btn = ttk.Button(control_frame, text="启用", 
                               command=self._enable_balancer, width=10)
        enable_btn.pack(side=tk.LEFT, padx=(10, 5))
        
        disable_btn = ttk.Button(control_frame, text="禁用", 
                                command=self._disable_balancer, width=10)
        disable_btn.pack(side=tk.LEFT, padx=5)
        
        # 状态显示
        self.status_label = ttk.Label(control_frame, text="状态: 未知", 
                                     foreground="gray")
        self.status_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # 2. 配置面板
        config_frame = ttk.LabelFrame(main_frame, text="配置参数", padding=10)
        config_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建配置选项卡
        notebook = ttk.Notebook(config_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 基础配置选项卡
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text="基础配置")
        self._create_basic_config(basic_frame)
        
        # 风控参数选项卡
        risk_frame = ttk.Frame(notebook)
        notebook.add(risk_frame, text="风控参数")
        self._create_risk_config(risk_frame)
        
        # 仓位管理选项卡
        sizing_frame = ttk.Frame(notebook)
        notebook.add(sizing_frame, text="仓位管理")
        self._create_sizing_config(sizing_frame)
        
        # 节流设置选项卡
        throttle_frame = ttk.Frame(notebook)
        notebook.add(throttle_frame, text="节流设置")
        self._create_throttle_config(throttle_frame)
        
        # 3. 操作按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="保存配置", 
                  command=self._save_config).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(button_frame, text="重载配置", 
                  command=self._load_config).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="重置配置", 
                  command=self._reset_config).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="清理缓存", 
                  command=self._clear_cache).pack(side=tk.LEFT, padx=5)
        
        # 4. 状态和日志
        log_frame = ttk.LabelFrame(main_frame, text="状态信息", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # 统计信息
        stats_frame = ttk.Frame(log_frame)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=4, state=tk.DISABLED)
        self.stats_text.pack(fill=tk.X)
        
        # 日志显示
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def _create_basic_config(self, parent):
        """创建基础配置界面"""
        # 流动性筛选
        liquidity_frame = ttk.LabelFrame(parent, text="流动性筛选", padding=10)
        liquidity_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 最小价格
        ttk.Label(liquidity_frame, text="最小价格 ($):").grid(row=0, column=0, sticky=tk.W)
        self.config_vars['min_price'] = tk.DoubleVar(value=5.0)
        ttk.Entry(liquidity_frame, textvariable=self.config_vars['min_price'], width=15).grid(row=0, column=1, padx=(10, 0))
        
        # 最小成交金额
        ttk.Label(liquidity_frame, text="最小成交金额 ($):").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.config_vars['min_adv_usd'] = tk.DoubleVar(value=500000.0)
        ttk.Entry(liquidity_frame, textvariable=self.config_vars['min_adv_usd'], width=15).grid(row=1, column=1, padx=(10, 0), pady=(5, 0))
        
        # 最大价差
        ttk.Label(liquidity_frame, text="最大价差 (bps):").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.config_vars['max_median_spread_bps'] = tk.DoubleVar(value=50.0)
        ttk.Entry(liquidity_frame, textvariable=self.config_vars['max_median_spread_bps'], width=15).grid(row=2, column=1, padx=(10, 0), pady=(5, 0))
        
        # 信号门槛
        signal_frame = ttk.LabelFrame(parent, text="信号门槛", padding=10)
        signal_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 最小alpha
        ttk.Label(signal_frame, text="最小Alpha (bps):").grid(row=0, column=0, sticky=tk.W)
        self.config_vars['min_alpha_bps'] = tk.DoubleVar(value=50.0)
        ttk.Entry(signal_frame, textvariable=self.config_vars['min_alpha_bps'], width=15).grid(row=0, column=1, padx=(10, 0))
        
        # Alpha vs sigma倍数
        ttk.Label(signal_frame, text="Alpha vs Sigma倍数:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.config_vars['min_alpha_vs_15m_sigma'] = tk.DoubleVar(value=2.0)
        ttk.Entry(signal_frame, textvariable=self.config_vars['min_alpha_vs_15m_sigma'], width=15).grid(row=1, column=1, padx=(10, 0), pady=(5, 0))
        
        # 组合权重
        portfolio_frame = ttk.LabelFrame(parent, text="组合权重", padding=10)
        portfolio_frame.pack(fill=tk.X)
        
        # Top-N提升
        ttk.Label(portfolio_frame, text="Top-N提升数量:").grid(row=0, column=0, sticky=tk.W)
        self.config_vars['top_n_boost'] = tk.IntVar(value=10)
        ttk.Entry(portfolio_frame, textvariable=self.config_vars['top_n_boost'], width=15).grid(row=0, column=1, padx=(10, 0))
        
        # 提升倍数
        ttk.Label(portfolio_frame, text="提升倍数:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.config_vars['top_n_boost_multiplier'] = tk.DoubleVar(value=1.25)
        ttk.Entry(portfolio_frame, textvariable=self.config_vars['top_n_boost_multiplier'], width=15).grid(row=1, column=1, padx=(10, 0), pady=(5, 0))
    
    def _create_risk_config(self, parent):
        """创建风控参数界面"""
        # 价差控制
        spread_frame = ttk.LabelFrame(parent, text="价差控制", padding=10)
        spread_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(spread_frame, text="大盘股最大价差 (bps):").grid(row=0, column=0, sticky=tk.W)
        self.config_vars['max_spread_bps_largecap'] = tk.DoubleVar(value=30.0)
        ttk.Entry(spread_frame, textvariable=self.config_vars['max_spread_bps_largecap'], width=15).grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(spread_frame, text="小盘股最大价差 (bps):").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.config_vars['max_spread_bps_smallcap'] = tk.DoubleVar(value=80.0)
        ttk.Entry(spread_frame, textvariable=self.config_vars['max_spread_bps_smallcap'], width=15).grid(row=1, column=1, padx=(10, 0), pady=(5, 0))
        
        # 风险控制系数
        risk_frame = ttk.LabelFrame(parent, text="风险控制系数", padding=10)
        risk_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(risk_frame, text="价差容忍倍数:").grid(row=0, column=0, sticky=tk.W)
        self.config_vars['k_spread'] = tk.DoubleVar(value=1.5)
        ttk.Entry(risk_frame, textvariable=self.config_vars['k_spread'], width=15).grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(risk_frame, text="ATR容忍倍数:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.config_vars['k_atr'] = tk.DoubleVar(value=0.5)
        ttk.Entry(risk_frame, textvariable=self.config_vars['k_atr'], width=15).grid(row=1, column=1, padx=(10, 0), pady=(5, 0))
        
        # 深度要求
        depth_frame = ttk.LabelFrame(parent, text="深度要求", padding=10)
        depth_frame.pack(fill=tk.X)
        
        ttk.Label(depth_frame, text="最小美元深度:").grid(row=0, column=0, sticky=tk.W)
        self.config_vars['min_dollar_depth'] = tk.DoubleVar(value=5000.0)
        ttk.Entry(depth_frame, textvariable=self.config_vars['min_dollar_depth'], width=15).grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(depth_frame, text="最小手数:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.config_vars['min_lot'] = tk.IntVar(value=100)
        ttk.Entry(depth_frame, textvariable=self.config_vars['min_lot'], width=15).grid(row=1, column=1, padx=(10, 0), pady=(5, 0))
    
    def _create_sizing_config(self, parent):
        """创建仓位管理界面"""
        # 权重控制
        weight_frame = ttk.LabelFrame(parent, text="权重控制", padding=10)
        weight_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(weight_frame, text="单票最大权重:").grid(row=0, column=0, sticky=tk.W)
        self.config_vars['max_weight'] = tk.DoubleVar(value=0.03)
        ttk.Entry(weight_frame, textvariable=self.config_vars['max_weight'], width=15).grid(row=0, column=1, padx=(10, 0))
        
        # 订单大小控制
        order_frame = ttk.LabelFrame(parent, text="订单大小控制", padding=10)
        order_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(order_frame, text="最大ADV百分比:").grid(row=0, column=0, sticky=tk.W)
        self.config_vars['max_child_adv_pct'] = tk.DoubleVar(value=0.10)
        ttk.Entry(order_frame, textvariable=self.config_vars['max_child_adv_pct'], width=15).grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(order_frame, text="最大盘口百分比:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.config_vars['max_child_book_pct'] = tk.DoubleVar(value=0.10)
        ttk.Entry(order_frame, textvariable=self.config_vars['max_child_book_pct'], width=15).grid(row=1, column=1, padx=(10, 0), pady=(5, 0))
        
        ttk.Label(order_frame, text="最小交易股数:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.config_vars['min_child_shares'] = tk.IntVar(value=50)
        ttk.Entry(order_frame, textvariable=self.config_vars['min_child_shares'], width=15).grid(row=2, column=1, padx=(10, 0), pady=(5, 0))
        
        # 降级策略
        degrade_frame = ttk.LabelFrame(parent, text="降级策略", padding=10)
        degrade_frame.pack(fill=tk.X)
        
        ttk.Label(degrade_frame, text="价差过宽缩减比例:").grid(row=0, column=0, sticky=tk.W)
        self.config_vars['on_wide_spread_shrink_to_pct'] = tk.DoubleVar(value=0.5)
        ttk.Entry(degrade_frame, textvariable=self.config_vars['on_wide_spread_shrink_to_pct'], width=15).grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(degrade_frame, text="流动性不足缩减比例:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.config_vars['on_thin_liquidity_shrink_to_pct'] = tk.DoubleVar(value=0.3)
        ttk.Entry(degrade_frame, textvariable=self.config_vars['on_thin_liquidity_shrink_to_pct'], width=15).grid(row=1, column=1, padx=(10, 0), pady=(5, 0))
        
        self.config_vars['on_price_drift_reject'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(degrade_frame, text="价格漂移时拒绝交易", 
                       variable=self.config_vars['on_price_drift_reject']).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
    
    def _create_throttle_config(self, parent):
        """创建节流设置界面"""
        # 频率控制
        freq_frame = ttk.LabelFrame(parent, text="频率控制", padding=10)
        freq_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(freq_frame, text="同标的最小间隔 (秒):").grid(row=0, column=0, sticky=tk.W)
        self.config_vars['min_interval_seconds'] = tk.IntVar(value=300)
        ttk.Entry(freq_frame, textvariable=self.config_vars['min_interval_seconds'], width=15).grid(row=0, column=1, padx=(10, 0))
        
        ttk.Label(freq_frame, text="每分钟最大订单数:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.config_vars['max_orders_per_minute'] = tk.IntVar(value=10)
        ttk.Entry(freq_frame, textvariable=self.config_vars['max_orders_per_minute'], width=15).grid(row=1, column=1, padx=(10, 0), pady=(5, 0))
    
    def _initialize_adapter(self):
        """初始化适配器"""
        try:
            self.adapter = get_risk_balancer_adapter()
            self._log("适配器初始化成功")
        except Exception as e:
            self._log(f"适配器初始化失败: {e}")
    
    def _enable_balancer(self):
        """启用风控收益平衡器"""
        try:
            enable_risk_balancer()
            self._log("风控收益平衡器已启用")
            self._update_status()
        except Exception as e:
            messagebox.showerror("错误", f"启用失败: {e}")
    
    def _disable_balancer(self):
        """禁用风控收益平衡器"""
        try:
            disable_risk_balancer()
            self._log("风控收益平衡器已禁用")
            self._update_status()
        except Exception as e:
            messagebox.showerror("错误", f"禁用失败: {e}")
    
    def _save_config(self):
        """保存配置"""
        try:
            # 构建配置字典
            config_dict = {}
            for key, var in self.config_vars.items():
                config_dict[key] = var.get()
            
            # 保存到文件
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            # 更新适配器配置
            if self.adapter:
                self.adapter.update_balancer_config(config_dict)
            
            self._log("配置保存成功")
            messagebox.showinfo("成功", "配置已保存")
            
        except Exception as e:
            error_msg = f"保存配置失败: {e}"
            self._log(error_msg)
            messagebox.showerror("错误", error_msg)
    
    def _load_config(self):
        """加载配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                
                # 更新GUI变量
                for key, value in config_dict.items():
                    if key in self.config_vars:
                        self.config_vars[key].set(value)
                
                self._log("配置加载成功")
            else:
                self._log("配置文件不存在，使用默认配置")
                
        except Exception as e:
            error_msg = f"加载配置失败: {e}"
            self._log(error_msg)
            messagebox.showerror("错误", error_msg)
    
    def _reset_config(self):
        """重置配置为默认值"""
        try:
            # 重置所有配置变量为默认值
            defaults = {
                'min_price': 5.0,
                'min_adv_usd': 500000.0,
                'max_median_spread_bps': 50.0,
                'min_alpha_bps': 50.0,
                'min_alpha_vs_15m_sigma': 2.0,
                'top_n_boost': 10,
                'top_n_boost_multiplier': 1.25,
                'max_spread_bps_largecap': 30.0,
                'max_spread_bps_smallcap': 80.0,
                'k_spread': 1.5,
                'k_atr': 0.5,
                'min_dollar_depth': 5000.0,
                'min_lot': 100,
                'max_weight': 0.03,
                'max_child_adv_pct': 0.10,
                'max_child_book_pct': 0.10,
                'min_child_shares': 50,
                'on_wide_spread_shrink_to_pct': 0.5,
                'on_thin_liquidity_shrink_to_pct': 0.3,
                'on_price_drift_reject': True,
                'min_interval_seconds': 300,
                'max_orders_per_minute': 10
            }
            
            for key, default_value in defaults.items():
                if key in self.config_vars:
                    self.config_vars[key].set(default_value)
            
            self._log("配置已重置为默认值")
            messagebox.showinfo("成功", "配置已重置")
            
        except Exception as e:
            error_msg = f"重置配置失败: {e}"
            self._log(error_msg)
            messagebox.showerror("错误", error_msg)
    
    def _clear_cache(self):
        """清理缓存"""
        try:
            if self.adapter:
                self.adapter.clear_balancer_cache()
            self._log("缓存已清理")
            messagebox.showinfo("成功", "缓存已清理")
        except Exception as e:
            error_msg = f"清理缓存失败: {e}"
            self._log(error_msg)
            messagebox.showerror("错误", error_msg)
    
    def _update_status(self):
        """更新状态显示"""
        try:
            is_enabled = is_risk_balancer_enabled()
            
            if is_enabled:
                status_text = "状态: 已启用"
                status_color = "green"
            else:
                status_text = "状态: 已禁用"
                status_color = "red"
            
            self.status_label.config(text=status_text, foreground=status_color)
            
            # 更新统计信息
            if self.adapter:
                stats = self.adapter.get_balancer_stats()
                if stats:
                    stats_text = f"""统计信息:
总信号数: {stats.get('total_signals', 0)}
通过数: {stats.get('approved', 0)}
降级数: {stats.get('degraded', 0)} 
拒绝数: {stats.get('rejected', 0)}
订单数: {stats.get('orders_sent', 0)}
最后重置: {stats.get('last_reset', 'N/A')}"""
                    
                    self.stats_text.config(state=tk.NORMAL)
                    self.stats_text.delete(1.0, tk.END)
                    self.stats_text.insert(1.0, stats_text)
                    self.stats_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self._log(f"更新状态失败: {e}")
    
    def _log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        
        # 限制日志行数
        lines = int(self.log_text.index(tk.END).split('.')[0])
        if lines > 100:
            self.log_text.delete(1.0, "2.0")
    
    def _start_update_thread(self):
        """启动状态更新线程"""
        def update_loop():
            while True:
                try:
                    # 每5秒更新一次状态
                    import time
                    time.sleep(5)
                    
                    # 在主线程中更新GUI
                    if self.window:
                        self.window.after(0, self._update_status)
                        
                except Exception as e:
                    print(f"状态更新线程出错: {e}")
                    break
        
        update_thread = threading.Thread(target=update_loop, daemon=True)
        update_thread.start()
    
    def get_frame(self):
        """获取框架组件，用于嵌入到其他窗口"""
        return self.window

def create_standalone_gui():
    """创建独立的GUI窗口"""
    panel = RiskBalancerGUIPanel()
    panel.window.mainloop()

def create_embedded_panel(parent):
    """创建嵌入式面板"""
    return RiskBalancerGUIPanel(parent)

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建独立GUI
    create_standalone_gui()