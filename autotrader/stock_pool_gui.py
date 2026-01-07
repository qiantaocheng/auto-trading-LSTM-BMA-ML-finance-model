#!/usr/bin/env python3
"""
股票池管理GUI组件
提供独立的股票池管理窗口
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from typing import List, Optional, Callable
import logging
from autotrader.stock_pool_manager import StockPoolManager

logger = logging.getLogger(__name__)

class StockPoolWindow:
    """股票池管理窗口"""
    
    def __init__(self, parent=None, on_pool_selected: Callable = None):
        """
        初始化股票池管理窗口
        
        Args:
            parent: 父窗口
            on_pool_selected: 选择股票池后的回调函数
        """
        self.parent = parent
        self.on_pool_selected = on_pool_selected
        self.pool_manager = StockPoolManager()
        
        # 创建独立窗口
        self.window = tk.Toplevel(parent) if parent else tk.Tk()
        self.window.title("股票池管理器")
        self.window.geometry("900x600")
        
        # 设置窗口图标和样式
        self.setup_styles()
        
        # 当前选中的股票池
        self.current_pool_id = None
        self.current_tickers = []
        
        # 创建界面
        self.create_widgets()
        
        # 加载股票池列表
        self.refresh_pools()
        
        # 绑定窗口关闭事件
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置颜色
        self.colors = {
            'bg': '#f0f0f0',
            'fg': '#333333',
            'select_bg': '#0078d4',
            'select_fg': 'white',
            'button_bg': '#0078d4',
            'button_fg': 'white',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545'
        }
        
        self.window.configure(bg=self.colors['bg'])
    
    def create_widgets(self):
        """创建界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧：股票池列表
        self.create_pool_list_panel(main_frame)
        
        # 中间：股票列表
        self.create_ticker_panel(main_frame)
        
        # 右侧：操作面板
        self.create_action_panel(main_frame)
        
        # 底部：状态栏
        self.create_status_bar()
    
    def create_pool_list_panel(self, parent):
        """创建股票池列表面板"""
        # 左侧框架
        left_frame = ttk.LabelFrame(parent, text="股票池列表", padding="5")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # 搜索框
        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(search_frame, text="搜索:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', lambda *args: self.filter_pools())
        ttk.Entry(search_frame, textvariable=self.search_var, width=20).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 股票池列表
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建Treeview
        self.pool_tree = ttk.Treeview(
            list_frame, 
            columns=('stocks', 'tags'),
            show='tree headings',
            height=15
        )
        
        self.pool_tree.heading('#0', text='名称')
        self.pool_tree.heading('stocks', text='股票数')
        self.pool_tree.heading('tags', text='标签')
        
        self.pool_tree.column('#0', width=150)
        self.pool_tree.column('stocks', width=60)
        self.pool_tree.column('tags', width=100)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.pool_tree.yview)
        self.pool_tree.configure(yscrollcommand=scrollbar.set)
        
        self.pool_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定选择事件
        self.pool_tree.bind('<<TreeviewSelect>>', self.on_pool_selected_event)
        
        # 按钮框架
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="新建", command=self.create_new_pool).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="复制", command=self.duplicate_pool).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="删除", command=self.delete_pool).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="刷新", command=self.refresh_pools).pack(
            side=tk.LEFT, padx=2)
    
    def create_ticker_panel(self, parent):
        """创建股票列表面板"""
        # 中间框架
        middle_frame = ttk.LabelFrame(parent, text="股票列表", padding="5")
        middle_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # 股票信息
        info_frame = ttk.Frame(middle_frame)
        info_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.pool_info_label = ttk.Label(info_frame, text="请选择一个股票池")
        self.pool_info_label.pack(side=tk.LEFT)
        
        # 股票列表框
        list_frame = ttk.Frame(middle_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建Listbox
        self.ticker_listbox = tk.Listbox(
            list_frame,
            selectmode=tk.EXTENDED,
            height=20
        )
        
        # 滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.ticker_listbox.yview)
        self.ticker_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.ticker_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 编辑框架
        edit_frame = ttk.Frame(middle_frame)
        edit_frame.pack(fill=tk.X, pady=5)
        
        # 添加股票输入框
        ttk.Label(edit_frame, text="添加:").pack(side=tk.LEFT)
        self.add_ticker_var = tk.StringVar()
        self.add_ticker_entry = ttk.Entry(edit_frame, textvariable=self.add_ticker_var, width=10)
        self.add_ticker_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(edit_frame, text="添加", command=self.add_ticker).pack(side=tk.LEFT, padx=2)
        ttk.Button(edit_frame, text="删除选中", command=self.remove_selected_tickers).pack(side=tk.LEFT, padx=2)
        ttk.Button(edit_frame, text="清空", command=self.clear_tickers).pack(side=tk.LEFT, padx=2)
        
        # 绑定回车键
        self.add_ticker_entry.bind('<Return>', lambda e: self.add_ticker())
    
    def create_action_panel(self, parent):
        """创建操作面板"""
        # 右侧框架
        right_frame = ttk.LabelFrame(parent, text="操作", padding="5")
        right_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        
        # 股票池详情
        detail_frame = ttk.LabelFrame(right_frame, text="股票池详情", padding="5")
        detail_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 名称
        ttk.Label(detail_frame, text="名称:").grid(row=0, column=0, sticky="w", pady=2)
        self.pool_name_var = tk.StringVar()
        self.pool_name_entry = ttk.Entry(detail_frame, textvariable=self.pool_name_var, width=20)
        self.pool_name_entry.grid(row=0, column=1, sticky="ew", pady=2)
        
        # 描述
        ttk.Label(detail_frame, text="描述:").grid(row=1, column=0, sticky="nw", pady=2)
        self.pool_desc_text = tk.Text(detail_frame, height=3, width=20)
        self.pool_desc_text.grid(row=1, column=1, sticky="ew", pady=2)
        
        # 标签
        ttk.Label(detail_frame, text="标签:").grid(row=2, column=0, sticky="w", pady=2)
        self.pool_tags_var = tk.StringVar()
        ttk.Entry(detail_frame, textvariable=self.pool_tags_var, width=20).grid(
            row=2, column=1, sticky="ew", pady=2)
        
        # 保存按钮
        ttk.Button(detail_frame, text="保存更改", command=self.save_pool_changes).grid(
            row=3, column=0, columnspan=2, pady=10)
        
        # 快速操作
        quick_frame = ttk.LabelFrame(right_frame, text="快速操作", padding="5")
        quick_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(quick_frame, text="导入文件", command=self.import_pool_file, width=15).pack(pady=2)
        ttk.Button(quick_frame, text="导出文件", command=self.export_pool_file, width=15).pack(pady=2)
        ttk.Button(quick_frame, text="批量添加", command=self.batch_add_tickers, width=15).pack(pady=2)
        
        # BMA训练集成
        bma_frame = ttk.LabelFrame(right_frame, text="BMA训练", padding="5")
        bma_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            bma_frame, 
            text="使用此股票池训练", 
            command=self.use_for_bma_training,
            width=15
        ).pack(pady=5)
        
        self.bma_status_label = ttk.Label(bma_frame, text="未选择股票池")
        self.bma_status_label.pack()
        
        # 统计信息
        stats_frame = ttk.LabelFrame(right_frame, text="统计", padding="5")
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.stats_text = tk.Text(stats_frame, height=5, width=25)
        self.stats_text.pack()
        self.stats_text.config(state=tk.DISABLED)
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = ttk.Label(
            self.window, 
            text="就绪", 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def refresh_pools(self):
        """刷新股票池列表"""
        try:
            # 清空列表
            for item in self.pool_tree.get_children():
                self.pool_tree.delete(item)
            
            # 获取所有股票池
            pools = self.pool_manager.get_all_pools()
            
            for pool in pools:
                tickers = json.loads(pool['tickers'])
                tags = json.loads(pool['tags']) if pool['tags'] else []
                
                self.pool_tree.insert(
                    '', 'end',
                    iid=pool['id'],
                    text=pool['pool_name'],
                    values=(len(tickers), ', '.join(tags[:3]))
                )
            
            self.update_status(f"已加载 {len(pools)} 个股票池")
            
        except Exception as e:
            logger.error(f"刷新股票池失败: {e}")
            self.show_error(f"刷新失败: {e}")
    
    def on_pool_selected_event(self, event):
        """股票池选择事件处理"""
        selection = self.pool_tree.selection()
        if selection:
            pool_id = int(selection[0])
            self.load_pool_details(pool_id)
    
    def load_pool_details(self, pool_id: int):
        """加载股票池详情"""
        try:
            pool = self.pool_manager.get_pool_by_id(pool_id)
            if not pool:
                return
            
            self.current_pool_id = pool_id
            
            # 更新详情面板
            self.pool_name_var.set(pool['pool_name'])
            self.pool_desc_text.delete('1.0', tk.END)
            self.pool_desc_text.insert('1.0', pool['description'] or '')
            
            tags = json.loads(pool['tags']) if pool['tags'] else []
            self.pool_tags_var.set(', '.join(tags))
            
            # 更新股票列表
            tickers = json.loads(pool['tickers'])
            self.current_tickers = tickers
            self.update_ticker_list(tickers)
            
            # 更新信息标签
            self.pool_info_label.config(
                text=f"{pool['pool_name']} - {len(tickers)}只股票"
            )
            
            # 更新统计信息
            self.update_statistics(pool, tickers)
            
            # 更新BMA状态
            self.bma_status_label.config(
                text=f"已选择: {pool['pool_name']}\n{len(tickers)}只股票"
            )
            
            self.update_status(f"已加载股票池: {pool['pool_name']}")
            
        except Exception as e:
            logger.error(f"加载股票池详情失败: {e}")
            self.show_error(f"加载失败: {e}")
    
    def update_ticker_list(self, tickers: List[str]):
        """更新股票列表显示"""
        self.ticker_listbox.delete(0, tk.END)
        for ticker in sorted(tickers):
            self.ticker_listbox.insert(tk.END, ticker)
    
    def update_statistics(self, pool: dict, tickers: List[str]):
        """更新统计信息"""
        stats = f"股票总数: {len(tickers)}\n"
        stats += f"创建时间: {pool['created_at'][:10]}\n"
        stats += f"更新时间: {pool['updated_at'][:10]}\n"
        
        # 按交易所分组
        nasdaq = sum(1 for t in tickers if '.' not in t)
        nyse = len(tickers) - nasdaq
        stats += f"NASDAQ: {nasdaq}, NYSE: {nyse}"
        
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert('1.0', stats)
        self.stats_text.config(state=tk.DISABLED)
    
    def create_new_pool(self):
        """创建新股票池"""
        dialog = NewPoolDialog(self.window)
        self.window.wait_window(dialog.dialog)
        
        if dialog.result:
            try:
                pool_id = self.pool_manager.create_pool(
                    pool_name=dialog.result['name'],
                    tickers=dialog.result['tickers'],
                    description=dialog.result['description'],
                    tags=dialog.result['tags']
                )
                self.refresh_pools()
                self.update_status(f"创建股票池成功: {dialog.result['name']}")
                
                # 选中新创建的股票池
                self.pool_tree.selection_set(str(pool_id))
                self.load_pool_details(pool_id)
                
            except Exception as e:
                self.show_error(f"创建失败: {e}")
    
    def duplicate_pool(self):
        """复制当前股票池"""
        if not self.current_pool_id:
            self.show_warning("请先选择一个股票池")
            return
        
        try:
            pool = self.pool_manager.get_pool_by_id(self.current_pool_id)
            new_name = f"{pool['pool_name']}_副本"
            
            pool_id = self.pool_manager.create_pool(
                pool_name=new_name,
                tickers=json.loads(pool['tickers']),
                description=pool['description'],
                tags=json.loads(pool['tags']) if pool['tags'] else []
            )
            
            self.refresh_pools()
            self.update_status(f"复制股票池成功: {new_name}")
            
        except Exception as e:
            self.show_error(f"复制失败: {e}")
    
    def delete_pool(self):
        """删除当前股票池"""
        if not self.current_pool_id:
            self.show_warning("请先选择一个股票池")
            return
        
        pool = self.pool_manager.get_pool_by_id(self.current_pool_id)
        if messagebox.askyesno("确认删除", f"确定要删除股票池 '{pool['pool_name']}' 吗？"):
            try:
                self.pool_manager.delete_pool(self.current_pool_id)
                self.refresh_pools()
                self.current_pool_id = None
                self.ticker_listbox.delete(0, tk.END)
                self.update_status(f"已删除股票池: {pool['pool_name']}")
                
            except Exception as e:
                self.show_error(f"删除失败: {e}")
    
    def save_pool_changes(self):
        """保存股票池更改"""
        if not self.current_pool_id:
            self.show_warning("请先选择一个股票池")
            return
        
        try:
            # 获取标签
            tags_str = self.pool_tags_var.get()
            tags = [t.strip() for t in tags_str.split(',') if t.strip()]
            
            # 更新股票池
            self.pool_manager.update_pool(
                self.current_pool_id,
                pool_name=self.pool_name_var.get(),
                description=self.pool_desc_text.get('1.0', tk.END).strip(),
                tags=tags,
                tickers=self.current_tickers
            )
            
            self.refresh_pools()
            self.update_status("保存成功")
            
        except Exception as e:
            self.show_error(f"保存失败: {e}")
    
    def add_ticker(self):
        """添加股票"""
        raw = self.add_ticker_var.get()
        try:
            from .stock_pool_manager import StockPoolManager
            ticker = StockPoolManager._sanitize_ticker(raw) or ''
        except Exception:
            ticker = (raw or '').upper().strip().replace('"','').replace("'",'')
            ticker = ''.join(c for c in ticker if not c.isspace())
        if not ticker:
            return
        
        if ticker not in self.current_tickers:
            self.current_tickers.append(ticker)
            self.update_ticker_list(self.current_tickers)
            self.add_ticker_var.set('')
            self.update_status(f"已添加: {ticker}")
        else:
            self.show_warning(f"{ticker} 已存在")
    
    def remove_selected_tickers(self):
        """删除选中的股票"""
        selected_indices = self.ticker_listbox.curselection()
        if not selected_indices:
            return
        
        # 获取选中的股票
        selected_tickers = [self.ticker_listbox.get(i) for i in selected_indices]
        
        # 从列表中移除
        self.current_tickers = [t for t in self.current_tickers if t not in selected_tickers]
        self.update_ticker_list(self.current_tickers)
        
        self.update_status(f"已删除 {len(selected_tickers)} 只股票")
    
    def clear_tickers(self):
        """清空股票列表"""
        if messagebox.askyesno("确认清空", "确定要清空所有股票吗？"):
            self.current_tickers = []
            self.update_ticker_list(self.current_tickers)
            self.update_status("已清空股票列表")
    
    def batch_add_tickers(self):
        """批量添加股票"""
        dialog = BatchAddDialog(self.window)
        self.window.wait_window(dialog.dialog)
        
        if dialog.result:
            new_tickers = [t for t in dialog.result if t not in self.current_tickers]
            self.current_tickers.extend(new_tickers)
            self.update_ticker_list(self.current_tickers)
            self.update_status(f"批量添加了 {len(new_tickers)} 只股票")
    
    def import_pool_file(self):
        """导入股票池文件"""
        filepath = filedialog.askopenfilename(
            title="选择导入文件",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if filepath:
            try:
                pool_id = self.pool_manager.import_pool(filepath)
                self.refresh_pools()
                self.update_status(f"导入成功")
                
            except Exception as e:
                self.show_error(f"导入失败: {e}")
    
    def export_pool_file(self):
        """导出股票池文件"""
        if not self.current_pool_id:
            self.show_warning("请先选择一个股票池")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="选择导出位置",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if filepath:
            try:
                self.pool_manager.export_pool(self.current_pool_id, filepath)
                self.update_status(f"导出成功: {filepath}")
                
            except Exception as e:
                self.show_error(f"导出失败: {e}")
    
    def use_for_bma_training(self):
        """使用当前股票池进行BMA训练"""
        if not self.current_pool_id:
            self.show_warning("请先选择一个股票池")
            return
        
        pool = self.pool_manager.get_pool_by_id(self.current_pool_id)
        tickers = json.loads(pool['tickers'])
        
        if len(tickers) < 10:
            self.show_warning(f"股票数量不足！至少需要10只股票，当前只有{len(tickers)}只")
            return
        
        # 调用回调函数
        if self.on_pool_selected:
            self.on_pool_selected({
                'pool_id': self.current_pool_id,
                'pool_name': pool['pool_name'],
                'tickers': tickers
            })
            self.update_status(f"已选择股票池用于BMA训练: {pool['pool_name']}")
        else:
            self.show_info(f"已选择股票池: {pool['pool_name']}\n包含{len(tickers)}只股票")
    
    def filter_pools(self):
        """过滤股票池列表"""
        keyword = self.search_var.get()
        if not keyword:
            self.refresh_pools()
            return
        
        try:
            pools = self.pool_manager.search_pools(keyword=keyword)
            
            # 清空列表
            for item in self.pool_tree.get_children():
                self.pool_tree.delete(item)
            
            # 显示搜索结果
            for pool in pools:
                tickers = json.loads(pool['tickers'])
                tags = json.loads(pool['tags']) if pool['tags'] else []
                
                self.pool_tree.insert(
                    '', 'end',
                    iid=pool['id'],
                    text=pool['pool_name'],
                    values=(len(tickers), ', '.join(tags[:3]))
                )
            
            self.update_status(f"找到 {len(pools)} 个匹配的股票池")
            
        except Exception as e:
            logger.error(f"搜索股票池失败: {e}")
    
    def update_status(self, message: str):
        """更新状态栏"""
        self.status_bar.config(text=message)
    
    def show_error(self, message: str):
        """显示错误消息"""
        messagebox.showerror("错误", message)
        self.update_status(f"错误: {message}")
    
    def show_warning(self, message: str):
        """显示警告消息"""
        messagebox.showwarning("警告", message)
        self.update_status(f"警告: {message}")
    
    def show_info(self, message: str):
        """显示信息消息"""
        messagebox.showinfo("信息", message)
    
    def on_closing(self):
        """窗口关闭事件"""
        self.window.destroy()


class NewPoolDialog:
    """新建股票池对话框"""
    
    def __init__(self, parent):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("新建股票池")
        self.dialog.geometry("400x300")
        
        # 名称
        ttk.Label(self.dialog, text="名称:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(self.dialog, textvariable=self.name_var, width=30).grid(
            row=0, column=1, padx=10, pady=5)
        
        # 描述
        ttk.Label(self.dialog, text="描述:").grid(row=1, column=0, sticky="nw", padx=10, pady=5)
        self.desc_text = tk.Text(self.dialog, height=3, width=30)
        self.desc_text.grid(row=1, column=1, padx=10, pady=5)
        
        # 股票列表
        ttk.Label(self.dialog, text="股票:\n(逗号或空格分隔)").grid(
            row=2, column=0, sticky="nw", padx=10, pady=5)
        self.tickers_text = tk.Text(self.dialog, height=5, width=30)
        self.tickers_text.grid(row=2, column=1, padx=10, pady=5)
        
        # 标签
        ttk.Label(self.dialog, text="标签:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.tags_var = tk.StringVar()
        ttk.Entry(self.dialog, textvariable=self.tags_var, width=30).grid(
            row=3, column=1, padx=10, pady=5)
        
        # 按钮
        button_frame = ttk.Frame(self.dialog)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="创建", command=self.create).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=self.cancel).pack(side=tk.LEFT, padx=5)
    
    def create(self):
        """创建股票池"""
        name = self.name_var.get().strip()
        
        # 解析股票列表（先解析股票，若名称为空则根据股票自动生成）
        tickers_text = self.tickers_text.get('1.0', tk.END).strip()
        tickers = []
        seen = set()
        for token in tickers_text.replace(',', ' ').split():
            try:
                from .stock_pool_manager import StockPoolManager
                t = StockPoolManager._sanitize_ticker(token)
            except Exception:
                t = (token or '').upper().strip().replace('"','').replace("'",'')
                t = ''.join(c for c in t if not c.isspace())
            if t and t not in seen:
                tickers.append(t)
                seen.add(t)
        
        # 若名称为空且存在有效股票，自动生成名称
        if not name and len(tickers) > 0:
            from datetime import datetime
            preview = '-'.join(tickers[:3])
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f"自定义股票池_{preview}_{ts}"
        
        # 校验名称
        if not name:
            messagebox.showwarning("警告", "请输入股票池名称")
            return
        
        # 校验股票
        if len(tickers) == 0:
            messagebox.showwarning("警告", "请输入股票代码")
            return
        
        # 解析标签
        tags_text = self.tags_var.get()
        tags = [t.strip() for t in tags_text.split(',') if t.strip()]
        
        self.result = {
            'name': name,
            'description': self.desc_text.get('1.0', tk.END).strip(),
            'tickers': tickers,
            'tags': tags
        }
        
        self.dialog.destroy()
    
    def cancel(self):
        """取消"""
        self.dialog.destroy()


class BatchAddDialog:
    """批量添加股票对话框"""
    
    def __init__(self, parent):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("批量添加股票")
        self.dialog.geometry("400x300")
        
        ttk.Label(self.dialog, text="输入股票代码（每行一个或用逗号/空格分隔）:").pack(pady=10)
        
        # 文本框
        text_frame = ttk.Frame(self.dialog)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        self.text = tk.Text(text_frame)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 按钮
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="添加", command=self.add).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=self.cancel).pack(side=tk.LEFT, padx=5)
    
    def add(self):
        """添加股票"""
        text = self.text.get('1.0', tk.END).strip()
        tickers = []
        
        # 解析输入
        for line in text.split('\n'):
            for token in line.replace(',', ' ').split():
                try:
                    from .stock_pool_manager import StockPoolManager
                    t = StockPoolManager._sanitize_ticker(token)
                except Exception:
                    t = (token or '').upper().strip().replace('"','').replace("'",'')
                    t = ''.join(c for c in t if not c.isspace())
                if t:
                    tickers.append(t)
        
        # 去重并保持输入顺序
        seen = set()
        ordered = []
        for t in tickers:
            if t not in seen:
                ordered.append(t)
                seen.add(t)
        self.result = ordered
        self.dialog.destroy()
    
    def cancel(self):
        """取消"""
        self.dialog.destroy()


# 测试窗口
if __name__ == "__main__":
    def on_pool_selected(pool_info):
        print(f"选择了股票池: {pool_info}")
    
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    window = StockPoolWindow(parent=root, on_pool_selected=on_pool_selected)
    root.mainloop()