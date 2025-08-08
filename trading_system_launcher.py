#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易系统启动器
整合LSTM、BMA模型训练和自动交易管理
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from datetime import datetime

class TradingSystemLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("量化交易系统启动器")
        self.root.geometry("800x600")
        
        # 进程状态
        self.lstm_process = None
        self.bma_process = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="量化交易系统启动器", font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 30))
        
        # 模型训练区域
        model_frame = ttk.LabelFrame(main_frame, text="模型训练", padding="15")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        model_frame.columnconfigure(1, weight=1)
        
        # LSTM模型
        ttk.Label(model_frame, text="LSTM多日预测模型:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Button(model_frame, text="训练LSTM模型", 
                  command=self.run_lstm_training, width=20).grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        self.lstm_status = ttk.Label(model_frame, text="未启动", foreground="gray")
        self.lstm_status.grid(row=0, column=2, sticky=tk.W, padx=(10, 0), pady=5)
        
        # BMA模型
        ttk.Label(model_frame, text="BMA增强版模型:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Button(model_frame, text="训练BMA模型", 
                  command=self.run_bma_training, width=20).grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        self.bma_status = ttk.Label(model_frame, text="未启动", foreground="gray")
        self.bma_status.grid(row=1, column=2, sticky=tk.W, padx=(10, 0), pady=5)
        
        # 分隔线
        ttk.Separator(main_frame, orient='horizontal').grid(row=2, column=0, sticky=(tk.W, tk.E), pady=20)
        
        # 自动交易区域
        trading_frame = ttk.LabelFrame(main_frame, text="自动交易管理", padding="15")
        trading_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # 自动交易管理器
        ttk.Label(trading_frame, text="启动独立的自动交易管理器:").grid(row=0, column=0, sticky=tk.W, pady=10)
        ttk.Button(trading_frame, text="启动交易管理器", 
                  command=self.launch_trading_manager, width=20).grid(row=1, column=0, sticky=tk.W, pady=5)
        
        # 说明文本
        info_text = """
        自动交易管理器功能：
        • 从LSTM/BMA输出文件选择股票
        • 手动编辑交易股票列表
        • 启动/停止自动交易
        • 股票退出时自动清仓
        """
        ttk.Label(trading_frame, text=info_text, justify=tk.LEFT, foreground="blue").grid(row=2, column=0, sticky=tk.W, pady=10)
        
        # 分隔线
        ttk.Separator(main_frame, orient='horizontal').grid(row=4, column=0, sticky=(tk.W, tk.E), pady=20)
        
        # 日志显示区域
        log_frame = ttk.LabelFrame(main_frame, text="运行日志", padding="10")
        log_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # 日志文本框
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 控制按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, pady=10)
        
        ttk.Button(button_frame, text="清空日志", 
                  command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="打开结果文件夹", 
                  command=self.open_result_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="退出", 
                  command=self.quit_application).pack(side=tk.LEFT, padx=5)
        
        # 配置主框架网格权重
        main_frame.rowconfigure(5, weight=1)
    
    def log_message(self, message: str):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        self.log_text.insert(tk.END, log_entry + "\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def run_lstm_training(self):
        """运行LSTM模型训练"""
        def train_lstm():
            try:
                self.lstm_status.config(text="运行中...", foreground="orange")
                self.log_message("开始LSTM模型训练...")
                
                # 运行LSTM训练
                self.lstm_process = subprocess.Popen(
                    [sys.executable, "lstm_multi_day_enhanced.py"],
                    cwd=os.getcwd(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8'
                )
                
                stdout, stderr = self.lstm_process.communicate()
                
                if self.lstm_process.returncode == 0:
                    self.lstm_status.config(text="完成", foreground="green")
                    self.log_message("LSTM模型训练完成")
                    if stdout:
                        self.log_message(f"LSTM输出: {stdout[-200:]}...")  # 显示最后200字符
                else:
                    self.lstm_status.config(text="失败", foreground="red")
                    self.log_message(f"LSTM模型训练失败: {stderr}")
                
            except Exception as e:
                self.lstm_status.config(text="错误", foreground="red")
                self.log_message(f"LSTM训练错误: {e}")
            finally:
                self.lstm_process = None
        
        if self.lstm_process and self.lstm_process.poll() is None:
            messagebox.showwarning("警告", "LSTM模型训练正在进行中")
            return
        
        # 在新线程中运行
        threading.Thread(target=train_lstm, daemon=True).start()
    
    def run_bma_training(self):
        """运行BMA模型训练"""
        def train_bma():
            try:
                self.bma_status.config(text="运行中...", foreground="orange")
                self.log_message("开始BMA模型训练...")
                
                # 运行BMA训练
                self.bma_process = subprocess.Popen(
                    [sys.executable, "量化模型_bma_enhanced.py", "--top-n", "10"],
                    cwd=os.getcwd(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8'
                )
                
                stdout, stderr = self.bma_process.communicate()
                
                if self.bma_process.returncode == 0:
                    self.bma_status.config(text="完成", foreground="green")
                    self.log_message("BMA模型训练完成")
                    if stdout:
                        self.log_message(f"BMA输出: {stdout[-200:]}...")  # 显示最后200字符
                else:
                    self.bma_status.config(text="失败", foreground="red")
                    self.log_message(f"BMA模型训练失败: {stderr}")
                
            except Exception as e:
                self.bma_status.config(text="错误", foreground="red")
                self.log_message(f"BMA训练错误: {e}")
            finally:
                self.bma_process = None
        
        if self.bma_process and self.bma_process.poll() is None:
            messagebox.showwarning("警告", "BMA模型训练正在进行中")
            return
        
        # 在新线程中运行
        threading.Thread(target=train_bma, daemon=True).start()
    
    def launch_trading_manager(self):
        """启动交易管理器"""
        try:
            # 启动独立的交易管理器
    # Updated to launch the unified QuantitativeTradingManager
    subprocess.Popen([sys.executable, "quantitative_trading_manager.py"])
            self.log_message("启动自动交易管理器")
            
        except Exception as e:
            self.log_message(f"启动交易管理器失败: {e}")
            messagebox.showerror("错误", f"启动交易管理器失败: {e}")
    
    def clear_log(self):
        """清空日志"""
        self.log_text.delete(1.0, tk.END)
    
    def open_result_folder(self):
        """打开结果文件夹"""
        try:
            result_dir = "result"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            # 根据操作系统打开文件夹
            if sys.platform.startswith('win'):
                os.startfile(result_dir)
            elif sys.platform.startswith('darwin'):
                subprocess.call(['open', result_dir])
            else:
                subprocess.call(['xdg-open', result_dir])
                
            self.log_message("打开结果文件夹")
            
        except Exception as e:
            self.log_message(f"打开结果文件夹失败: {e}")
            messagebox.showerror("错误", f"打开结果文件夹失败: {e}")
    
    def quit_application(self):
        """退出应用程序"""
        # 终止正在运行的进程
        if self.lstm_process and self.lstm_process.poll() is None:
            self.lstm_process.terminate()
        
        if self.bma_process and self.bma_process.poll() is None:
            self.bma_process.terminate()
        
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """运行应用程序"""
        self.log_message("量化交易系统启动器已启动")
        self.log_message("使用说明：")
        self.log_message("1. 先运行LSTM或BMA模型生成股票预测")
        self.log_message("2. 启动交易管理器选择股票并开始自动交易")
        self.log_message("3. 所有输出文件将保存到result文件夹")
        
        # 设置关闭窗口事件
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)
        
        self.root.mainloop()

def main():
    """主函数"""
    try:
        app = TradingSystemLauncher()
        app.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()