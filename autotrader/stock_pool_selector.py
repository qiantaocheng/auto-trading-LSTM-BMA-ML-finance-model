#!/usr/bin/env python3
"""
股票池选择对话框
专门用于BMA训练中选择股票池的模态对话框
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
from typing import Optional, Dict, List
import logging
from stock_pool_manager import StockPoolManager

logger = logging.getLogger(__name__)

class StockPoolSelectorDialog:
    """股票池选择对话框"""
    
    def __init__(self, parent):
        """
        初始化股票池选择对话框
        
        Args:
            parent: 父窗口
        """
        self.parent = parent
        self.result = None
        self.pool_manager = StockPoolManager()
        
        # 创建模态对话框
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("选择股票池")
        self.dialog.geometry("600x500")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # 居中显示
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (500 // 2)
        self.dialog.geometry(f"600x500+{x}+{y}")
        
        # 防止调整大小
        self.dialog.resizable(False, False)
        
        # 当前选中的股票池
        self.selected_pool_id = None
        self.selected_pool_data = None
        
        # 创建界面
        self.create_widgets()
        
        # 加载股票池列表
        self.refresh_pools()
        
        # 绑定关闭事件
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
    
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = tk.Frame(self.dialog, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 标题
        title_label = tk.Label(
            main_frame, 
            text="选择用于BMA训练的股票池", 
            font=("Arial", 14, "bold"),
            bg='white'
        )
        title_label.pack(pady=(0, 15))
        
        # 搜索框
        search_frame = tk.Frame(main_frame, bg='white')
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(search_frame, text="搜索:", font=("Arial", 10), bg='white').pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', lambda *args: self.filter_pools())
        search_entry = tk.Entry(search_frame, textvariable=self.search_var, width=30, font=("Arial", 10))
        search_entry.pack(side=tk.LEFT, padx=5)
        
        # 股票池列表框架
        list_frame = tk.LabelFrame(main_frame, text="可用股票池", font=("Arial", 10), bg='white')
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # 创建Treeview显示股票池
        columns = ('stocks', 'description', 'tags')
        self.pool_tree = ttk.Treeview(
            list_frame,
            columns=columns,
            show='tree headings',
            height=12
        )
        
        # 设置列标题和宽度
        self.pool_tree.heading('#0', text='股票池名称')
        self.pool_tree.heading('stocks', text='股票数')
        self.pool_tree.heading('description', text='描述')
        self.pool_tree.heading('tags', text='标签')
        
        self.pool_tree.column('#0', width=150)
        self.pool_tree.column('stocks', width=80)
        self.pool_tree.column('description', width=200)
        self.pool_tree.column('tags', width=120)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.pool_tree.yview)
        self.pool_tree.configure(yscrollcommand=scrollbar.set)
        
        self.pool_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # 绑定选择事件
        self.pool_tree.bind('<<TreeviewSelect>>', self.on_pool_select)
        self.pool_tree.bind('<Double-1>', self.on_pool_double_click)
        
        # 股票池详情框架
        detail_frame = tk.LabelFrame(main_frame, text="股票池详情", font=("Arial", 10), bg='white')
        detail_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 详情文本显示
        self.detail_text = tk.Text(
            detail_frame,
            height=4,
            width=60,
            font=("Arial", 9),
            bg='#f8f9fa',
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.detail_text.pack(padx=10, pady=5)
        
        # 按钮框架
        button_frame = tk.Frame(main_frame, bg='white')
        button_frame.pack(fill=tk.X, pady=10)
        
        # 管理股票池按钮（左侧）
        manage_button = tk.Button(
            button_frame,
            text="管理股票池",
            command=self.open_pool_manager,
            bg="#FF9800",
            fg="white",
            font=("Arial", 10),
            width=12
        )
        manage_button.pack(side=tk.LEFT)
        
        # 确认和取消按钮（右侧）
        cancel_button = tk.Button(
            button_frame,
            text="取消",
            command=self.on_cancel,
            bg="#f44336",
            fg="white",
            font=("Arial", 11),
            width=10,
            height=2
        )
        cancel_button.pack(side=tk.RIGHT, padx=5)
        
        self.confirm_button = tk.Button(
            button_frame,
            text="确认选择",
            command=self.on_confirm,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11, "bold"),
            width=12,
            height=2,
            state=tk.DISABLED  # 初始禁用
        )
        self.confirm_button.pack(side=tk.RIGHT, padx=5)
        
        # 说明文本
        info_label = tk.Label(
            main_frame,
            text="提示：BMA训练需要至少10只股票。双击股票池名称可快速选择。",
            font=("Arial", 9),
            fg="gray",
            bg='white'
        )
        info_label.pack(pady=(10, 0))
    
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
                
                # 根据股票数量设置颜色
                if len(tickers) >= 20:
                    tag_color = "excellent"
                elif len(tickers) >= 10:
                    tag_color = "good"
                else:
                    tag_color = "insufficient"
                
                item_id = self.pool_tree.insert(
                    '', 'end',
                    iid=str(pool['id']),
                    text=pool['pool_name'],
                    values=(
                        len(tickers),
                        (pool['description'] or '')[:40] + ('...' if len(pool['description'] or '') > 40 else ''),
                        ', '.join(tags[:2]) + ('...' if len(tags) > 2 else '')
                    ),
                    tags=(tag_color,)
                )
            
            # 设置标签颜色
            self.pool_tree.tag_configure("excellent", background="#e8f5e8")  # 绿色
            self.pool_tree.tag_configure("good", background="#fff3cd")       # 黄色
            self.pool_tree.tag_configure("insufficient", background="#f8d7da") # 红色
            
        except Exception as e:
            logger.error(f"刷新股票池列表失败: {e}")
            messagebox.showerror("错误", f"加载股票池失败: {e}")
    
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
                    iid=str(pool['id']),
                    text=pool['pool_name'],
                    values=(
                        len(tickers),
                        (pool['description'] or '')[:40],
                        ', '.join(tags[:2])
                    )
                )
        except Exception as e:
            logger.error(f"搜索股票池失败: {e}")
    
    def on_pool_select(self, event):
        """股票池选择事件"""
        selection = self.pool_tree.selection()
        if selection:
            pool_id = int(selection[0])
            self.load_pool_details(pool_id)
    
    def on_pool_double_click(self, event):
        """双击股票池事件 - 直接确认选择"""
        selection = self.pool_tree.selection()
        if selection:
            pool_id = int(selection[0])
            self.load_pool_details(pool_id)
            if self.selected_pool_data and len(self.selected_pool_data['tickers']) >= 10:
                self.on_confirm()
            else:
                messagebox.showwarning(
                    "股票数量不足", 
                    f"该股票池只有{len(self.selected_pool_data['tickers']) if self.selected_pool_data else 0}只股票，"
                    f"BMA训练至少需要10只股票。"
                )
    
    def load_pool_details(self, pool_id: int):
        """加载股票池详情"""
        try:
            pool = self.pool_manager.get_pool_by_id(pool_id)
            if not pool:
                return
            
            self.selected_pool_id = pool_id
            tickers = json.loads(pool['tickers'])
            tags = json.loads(pool['tags']) if pool['tags'] else []
            
            self.selected_pool_data = {
                'pool_id': pool_id,
                'pool_name': pool['pool_name'],
                'tickers': tickers,
                'description': pool['description'],
                'tags': tags
            }
            
            # 更新详情显示
            detail_text = f"股票池：{pool['pool_name']}\n"
            detail_text += f"股票数量：{len(tickers)} 只\n"
            detail_text += f"描述：{pool['description'] or '无'}\n"
            detail_text += f"股票列表：{', '.join(tickers[:15])}"
            if len(tickers) > 15:
                detail_text += f" ... (共{len(tickers)}只)"
            
            self.detail_text.config(state=tk.NORMAL)
            self.detail_text.delete(1.0, tk.END)
            self.detail_text.insert(1.0, detail_text)
            self.detail_text.config(state=tk.DISABLED)
            
            # 启用/禁用确认按钮
            if len(tickers) >= 10:
                self.confirm_button.config(state=tk.NORMAL)
                self.confirm_button.config(bg="#4CAF50")
            else:
                self.confirm_button.config(state=tk.DISABLED)
                self.confirm_button.config(bg="#cccccc")
                
        except Exception as e:
            logger.error(f"加载股票池详情失败: {e}")
            messagebox.showerror("错误", f"加载详情失败: {e}")
    
    def open_pool_manager(self):
        """打开股票池管理器"""
        try:
            from stock_pool_gui import StockPoolWindow
            
            # 创建股票池管理窗口
            pool_window = StockPoolWindow()
            
            # 等待管理窗口关闭后刷新列表
            self.dialog.after(1000, self.refresh_pools)
            
        except Exception as e:
            messagebox.showerror("错误", f"打开股票池管理器失败: {e}")
    
    def on_confirm(self):
        """确认选择"""
        if not self.selected_pool_data:
            messagebox.showwarning("警告", "请先选择一个股票池")
            return
        
        if len(self.selected_pool_data['tickers']) < 10:
            if not messagebox.askyesno(
                "股票数量不足",
                f"该股票池只有{len(self.selected_pool_data['tickers'])}只股票，"
                f"BMA训练建议至少10只股票。\n\n是否继续使用？"
            ):
                return
        
        self.result = self.selected_pool_data
        self.dialog.destroy()
    
    def on_cancel(self):
        """取消选择"""
        self.result = None
        self.dialog.destroy()
    
    def show(self):
        """显示对话框并返回结果"""
        # 等待对话框关闭
        self.dialog.wait_window()
        return self.result


# 便捷函数
def select_stock_pool(parent) -> Optional[Dict]:
    """
    显示股票池选择对话框
    
    Args:
        parent: 父窗口
        
    Returns:
        选中的股票池信息字典，如果取消则返回None
    """
    dialog = StockPoolSelectorDialog(parent)
    return dialog.show()


# 测试代码
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    result = select_stock_pool(root)
    if result:
        print(f"选择了股票池: {result['pool_name']}")
        print(f"包含股票: {len(result['tickers'])}只")
        print(f"股票列表: {result['tickers']}")
    else:
        print("取消选择")
    
    root.mainloop()