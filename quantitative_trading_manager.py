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
from plyer import notification

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
    3. 定时任务 - 每月第一天中午12点自动运行
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
            'result_directory': 'result'
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
                logging.FileHandler(log_dir / f"trading_manager_{datetime.now().strftime('%Y%m%d')}.log"),
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
        
        # 添加每月第一天中午12点的定时任务
        self.scheduler.add_job(
            func=self.auto_run_analysis,
            trigger=CronTrigger(day=1, hour=12, minute=0),
            id='monthly_analysis',
            name='月度量化分析',
            replace_existing=True
        )
        
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
        
        # 量化模型按钮（增强版）
        self.quant_button_frame = self.create_quagsire_button(
            buttons_container, 
            "量化分析模型",
            self.show_date_selection_dialog,
            0, 0
        )
        
        # 启动回测按钮
        self.backtest_button_frame = self.create_quagsire_button(
            buttons_container,
            "启动回测分析", 
            self.run_backtest_analysis,
            0, 1
        )
        
        # ML回测按钮
        self.ml_backtest_button_frame = self.create_quagsire_button(
            buttons_container,
            "ML滚动回测",
            self.run_ml_backtest,
            0, 2
        )
        
        # 配置按钮列权重
        for i in range(3):
            buttons_container.columnconfigure(i, weight=1)
        
        # 配置按钮行权重
        for i in range(1):
            buttons_container.rowconfigure(i, weight=1)
        
        # 添加快捷操作按钮
        quick_frame = ttk.Frame(button_frame)
        quick_frame.grid(row=1, column=0, columnspan=3, pady=(10, 0), sticky=(tk.W, tk.E))
        
        ttk.Button(quick_frame, text=" 打开结果文件夹", 
                   command=self.open_result_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text=" 查看历史记录", 
                   command=self.show_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(quick_frame, text=" 设置", 
                   command=self.show_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(quick_frame, text=" 实时监控", 
                   command=self.show_monitoring).pack(side=tk.LEFT, padx=5)
    
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
        if now.day == 1 and now.hour < 12:
            next_run = now.replace(hour=12, minute=0, second=0, microsecond=0)
        else:
            # 下个月第一天12点
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
    
    def run_ml_backtest(self):
        """运行ML滚动回测"""
        # 提示用户选择文件
        info_msg = ("机器学习滚动回测需要基于之前的量化分析结果进行。\n\n"
                   "请选择一个量化分析结果文件：\n"
                   "• 通常位于 result/ 文件夹中\n"
                   "• 文件名如：quantitative_analysis_*.xlsx\n"
                   "• 建议选择最新的分析结果")
        
        messagebox.showinfo("选择分析文件", info_msg)
        
        # 首先让用户选择分析结果文件
        file_path = filedialog.askopenfilename(
            title="选择量化分析结果文件 - ML滚动回测",
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
                self.update_status("正在启动ML回测...", 10)
                self.ml_backtest_button.config(state='disabled')
                
                start_time = time.time()
                
                # 运行ML回测，传递选中的文件路径
                process = subprocess.Popen(
                    [sys.executable, "ml_rolling_backtest_clean.py", file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='gbk',  # 使用GBK编码处理中文
                    cwd=os.getcwd()
                )
                
                self.update_status("ML回测运行中...", 50)
                
                stdout, stderr = process.communicate()
                duration = time.time() - start_time
                
                if process.returncode == 0:
                    self.update_status("ML回测完成", 100)
                    
                    # 查找生成的文件
                    result_files = self.find_latest_result_files("clean_backtest_stats_")
                    
                    # 保存到数据库
                    self.save_analysis_result("ML回测", result_files[0] if result_files else "",
                                            duration, stdout)
                    
                    self.show_notification("任务完成", f"ML回测分析完成\n耗时: {duration:.1f}秒")
                    self.load_recent_results()
                    
                else:
                    # 截断错误信息，避免过长
                    short_error = stderr[:150] + "..." if len(stderr) > 150 else stderr
                    error_msg = f"ML回测运行失败\n错误信息: {short_error}"
                    self.update_status("ML回测运行失败", 0)
                    self.show_notification("任务失败", error_msg)
                    self.logger.error(f"ML回测运行失败\n完整错误信息: {stderr}")
                    
            except Exception as e:
                error_msg = f"启动ML回测失败: {e}"
                self.update_status(error_msg, 0)
                self.show_notification("错误", error_msg)
                self.logger.error(error_msg)
                
            finally:
                self.ml_backtest_button.config(state='normal')
                self.update_status("就绪", 0)
        
        threading.Thread(target=run_in_thread, daemon=True).start()
    
    def find_latest_result_files(self, prefix):
        """查找最新的结果文件"""
        result_files = []
        
        # 检查当前目录
        for file in Path('.').glob(f"{prefix}*.xlsx"):
            result_files.append(str(file))
        
        # 检查result目录
        result_dir = Path(self.config['result_directory'])
        if result_dir.exists():
            for file in result_dir.glob(f"{prefix}*.xlsx"):
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
                if '共分析' in line and '只股票' in line:
                    # 提取股票数量
                    import re
                    match = re.search(r'(\d+)只股票', line)
                    if match:
                        stock_count = int(match.group(1))
                elif 'BUY:' in line and 'HOLD:' in line and 'SELL:' in line:
                    # 提取评级分布
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
                    # 提取平均评分
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
        self.logger.info("开始自动运行月度分析")
        
        def auto_run_thread():
            try:
                # 显示通知
                self.show_notification("定时任务", "开始执行月度量化分析", timeout=5)
                
                # 依次运行三个分析
                self.run_quantitative_model()
                time.sleep(30)  # 等待第一个任务完成
                
                self.run_backtest_analysis()
                time.sleep(30)  # 等待第二个任务完成
                
                self.run_ml_backtest()
                
                # 完成通知
                self.show_notification("定时任务完成", 
                                     f"月度量化分析已完成\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                
                # 更新下次运行时间
                self.schedule_status_var.set(f"下次自动运行: {self.get_next_run_time()}")
                
            except Exception as e:
                error_msg = f"自动分析失败: {e}"
                self.logger.error(error_msg)
                self.show_notification("定时任务失败", error_msg)
        
        threading.Thread(target=auto_run_thread, daemon=True).start()
    
    def open_result_folder(self):
        """打开结果文件夹"""
        result_path = Path(self.config['result_directory'])
        if not result_path.exists():
            result_path.mkdir()
        
        try:
            os.startfile(str(result_path))
        except:
            webbrowser.open(f"file://{result_path.absolute()}")
    
    def show_settings(self):
        """显示设置窗口"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("设置")
        settings_window.geometry("400x300")
        settings_window.resizable(False, False)
        
        # 设置选项
        ttk.Label(settings_window, text="应用设置", font=('Microsoft YaHei', 12, 'bold')).pack(pady=10)
        
        # 自动运行选项
        auto_run_var = tk.BooleanVar(value=self.config['auto_run'])
        ttk.Checkbutton(settings_window, text="启用月度自动分析", 
                       variable=auto_run_var).pack(anchor='w', padx=20, pady=5)
        
        # 通知选项
        notifications_var = tk.BooleanVar(value=self.config['notifications'])
        ttk.Checkbutton(settings_window, text="启用系统通知", 
                       variable=notifications_var).pack(anchor='w', padx=20, pady=5)
        
        # 日志级别
        ttk.Label(settings_window, text="日志级别:").pack(anchor='w', padx=20, pady=(10, 0))
        log_level_var = tk.StringVar(value=self.config['log_level'])
        log_level_combo = ttk.Combobox(settings_window, textvariable=log_level_var,
                                      values=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
        log_level_combo.pack(anchor='w', padx=20, pady=5)
        
        # 保存按钮
        def save_settings():
            self.config['auto_run'] = auto_run_var.get()
            self.config['notifications'] = notifications_var.get()
            self.config['log_level'] = log_level_var.get()
            
            # 保存配置到文件
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("设置", "设置已保存")
            settings_window.destroy()
        
        ttk.Button(settings_window, text="保存设置", command=save_settings).pack(pady=20)
    
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
    
    def show_monitoring(self):
        """显示实时监控窗口"""
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
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    recent_logs = ''.join(lines[-50:])  # 最近50行
                    log_text.config(state='normal')
                    log_text.insert(tk.END, recent_logs)
                    log_text.config(state='disabled')
                    log_text.see(tk.END)
        except Exception as e:
            self.logger.error(f"读取日志失败: {e}")
    
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
    
    def show_date_selection_dialog(self):
        """显示量化分析参数设置对话框"""
        self._show_model_dialog("enhanced")
    
    def _show_model_dialog(self, model_type):
        """通用模型参数设置对话框"""
        dialog = tk.Toplevel(self.root)
        
        # 设置标题和说明
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
            
            # 运行增强版量化模型
            self.run_enhanced_model_with_dates(start_date, end_date, custom_ticker_file)
            
        except Exception as e:
            self.logger.error(f"日期确认失败: {e}")
            messagebox.showerror("错误", f"日期设置失败: {e}")
    
    def run_enhanced_model_with_dates(self, start_date, end_date, ticker_file=None):
        """运行增强版量化模型"""
        def run_analysis():
            try:
                start_time = time.time()
                self.show_notification("量化分析", "正在启动量化分析模型...")
                
                # 构建命令
                cmd = [sys.executable, "量化模型_enhanced.py"]
                
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
                
                if result.returncode == 0:
                    self.show_notification("分析完成", "量化分析模型分析已完成！")
                    self.logger.info("量化模型执行成功")
                    
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
                    
            except FileNotFoundError:
                self.show_notification("文件不存在", "找不到量化模型文件 (量化模型_enhanced.py)")
                self.logger.error("量化模型文件不存在")
            except Exception as e:
                error_msg = str(e)[:150]
                self.show_notification("分析错误", f"执行错误: {error_msg}")
                self.logger.error(f"量化模型执行异常: {e}")
        
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
        
        default_info = ttk.Label(self.default_info_frame, 
                                text="默认股票池包含357只精选股票，涵盖科技、金融、医疗等多个行业\n推荐用于全面的市场分析",
                                font=('Microsoft YaHei', 9),
                                foreground='gray')
        default_info.pack()
        
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
        
        # 初始状态
        self.on_stock_mode_change(dialog)
    
    def on_stock_mode_change(self, dialog):
        """股票池模式切换"""
        mode = self.stock_mode_var.get()
        
        # 隐藏所有框架
        self.default_info_frame.pack_forget()
        self.edit_default_frame.pack_forget()
        self.custom_frame.pack_forget()
        
        if mode == "default":
            # 使用默认股票池
            self.default_info_frame.pack(fill='x', pady=(0, 10))
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
        input_dialog.geometry("300x150")
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
            if ticker and ticker not in self.edited_default_list:
                self.edited_default_list.append(ticker)
                self.update_default_preview_tree()
                self.default_count_label.config(text=f"默认股票池: {len(self.edited_default_list)} 只")
                input_dialog.destroy()
                messagebox.showinfo("添加成功", f"已将 {ticker} 添加到默认股票池")
            elif ticker in self.edited_default_list:
                messagebox.showinfo("重复股票", f"股票代码 {ticker} 已在默认池中")
            else:
                messagebox.showwarning("输入错误", "请输入有效的股票代码")
        
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
• 每月第一天中午12点自动运行所有分析
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
                    with open(Path("logs", log_file_var.get()), 'r', encoding='utf-8') as f:
                        content = f.read()
                        log_text.delete(1.0, tk.END)
                        log_text.insert(tk.END, content)
                        log_text.see(tk.END)
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