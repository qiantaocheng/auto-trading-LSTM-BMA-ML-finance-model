#!/usr/bin/env python3
"""
编码问题修复工具
统一处理中文字符和日志编码问题
"""

import os
import sys
import logging
import locale
from typing import Optional
import warnings

class EncodingFixer:
    """编码修复器"""
    
    def __init__(self):
        self.original_encoding = sys.stdout.encoding
        self.fixed = False
    
    def fix_system_encoding(self):
        """修复系统编码设置"""
        # 设置环境变量
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # Windows特殊处理
        if sys.platform.startswith('win'):
            try:
                # 安全地设置控制台代码页为UTF-8
                import subprocess
                try:
                    subprocess.run(['chcp', '65001'], 
                                 capture_output=True, 
                                 check=False, 
                                 timeout=5.0,
                                 creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
                except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
                    # 如果subprocess失败，尝试使用ctypes（更安全的方法）
                    try:
                        import ctypes
                        ctypes.windll.kernel32.SetConsoleCP(65001)
                        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
                    except Exception:
                        pass  # 静默失败，不影响核心功能
                
                # 设置locale
                try:
                    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
                except (locale.Error, OSError):
                    try:
                        locale.setlocale(locale.LC_ALL, 'Chinese_China.UTF-8')
                    except (locale.Error, OSError):
                        try:
                            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
                        except (locale.Error, OSError):
                            pass  # 使用系统默认
                        
            except Exception:
                pass
        
        # Unix/Linux系统
        else:
            try:
                locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
            except (locale.Error, OSError):
                try:
                    locale.setlocale(locale.LC_ALL, 'C.UTF-8')
                except (locale.Error, OSError):
                    pass
        
        self.fixed = True
    
    def setup_safe_logging(self, log_file: Optional[str] = None):
        """设置安全的日志记录"""
        # 创建自定义格式化器，替换不安全字符
        class SafeFormatter(logging.Formatter):
            # 类级别的安全替换字典，避免每次日志都重新创建
            _SAFE_REPLACEMENTS = {
                '✅': '[OK]',
                '❌': '[FAIL]',
                '⚠️': '[WARN]',
                '🚀': '[START]',
                '📊': '[DATA]',
                '💰': '[MONEY]',
                '🔄': '[REFRESH]',
                '📈': '[UP]',
                '📉': '[DOWN]',
                '🎯': '[TARGET]',
                '🔧': '[FIX]',
                '📝': '[LOG]',
                '🧠': '[AI]',
                '⏰': '[TIME]',
                '🔥': '[HOT]',
                '💡': '[IDEA]',
                '🛡️': '[SAFE]',
                '⚡': '[FAST]',
            }
            
            def format(self, record):
                # 获取原始消息
                msg = super().format(record)
                
                # 使用类级别字典进行替换，提升性能
                for emoji, replacement in self._SAFE_REPLACEMENTS.items():
                    msg = msg.replace(emoji, replacement)
                
                return msg
        
        # 配置根日志记录器
        root_logger = logging.getLogger()
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 设置日志级别
        root_logger.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = SafeFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # 设置控制台编码
        if hasattr(console_handler.stream, 'reconfigure'):
            try:
                console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
            except (AttributeError, OSError, TypeError):
                pass
        
        root_logger.addHandler(console_handler)
        
        # 文件处理器（如果指定了文件路径）
        if log_file:
            try:
                # 确保日志目录存在
                log_dir = os.path.dirname(log_file)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                
                file_handler = logging.FileHandler(
                    log_file, 
                    mode='a', 
                    encoding='utf-8',
                    errors='replace'
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.DEBUG)
                root_logger.addHandler(file_handler)
                
            except Exception as e:
                print(f"Warning: Could not set up file logging: {e}")
    
    def safe_print(self, message: str, file=None):
        """安全打印，避免编码错误"""
        if file is None:
            file = sys.stdout
        
        try:
            print(message, file=file)
        except UnicodeEncodeError:
            # 替换不能编码的字符
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            print(safe_message, file=file)
        except (UnicodeError, OSError, IOError) as e:
            # 最后的备用方案
            try:
                print(repr(message), file=file)
            except Exception:
                # If even repr fails, just give up gracefully
                pass
    
    def fix_pandas_display(self):
        """修复pandas显示编码问题"""
        try:
            import pandas as pd
            
            # 设置pandas显示选项
            pd.set_option('display.unicode.east_asian_width', True)
            pd.set_option('display.unicode.ambiguous_as_wide', True)
            
            # 限制显示行列数，避免大量输出
            pd.set_option('display.max_rows', 50)
            pd.set_option('display.max_columns', 20)
            pd.set_option('display.width', 120)
            
        except ImportError:
            pass
    
    def fix_matplotlib_fonts(self):
        """修复matplotlib中文字体问题"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            
            # 设置matplotlib后端
            mpl.use('Agg')  # 非交互式后端
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
        except ImportError:
            pass
    
    def apply_all_fixes(self, log_file: Optional[str] = None):
        """应用所有编码修复"""
        self.fix_system_encoding()
        self.setup_safe_logging(log_file)
        self.fix_pandas_display()
        self.fix_matplotlib_fonts()
        
        # 抑制一些编码相关的警告
        warnings.filterwarnings('ignore', category=UnicodeWarning)
        
        return True


def safe_str(obj) -> str:
    """安全转换为字符串"""
    try:
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        else:
            return str(obj)
    except (UnicodeError, AttributeError, TypeError):
        return repr(obj)


def safe_format(template: str, *args, **kwargs) -> str:
    """安全格式化字符串"""
    try:
        # 转换所有参数为安全字符串
        safe_args = [safe_str(arg) for arg in args]
        safe_kwargs = {k: safe_str(v) for k, v in kwargs.items()}
        
        return template.format(*safe_args, **safe_kwargs)
    except (ValueError, KeyError, TypeError) as e:
        return f"Format error: {repr(template)} - {e}"


class SafeLogger:
    """安全日志记录器包装器"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def _safe_log(self, level: int, message: str, *args, **kwargs):
        """安全记录日志"""
        try:
            if args or kwargs:
                message = safe_format(message, *args, **kwargs)
            
            # 移除或替换可能有问题的字符
            safe_message = safe_str(message)
            
            self.logger.log(level, safe_message)
            
        except Exception as e:
            # 如果日志记录失败，至少尝试打印到控制台
            try:
                print(f"Logging error: {e}, Original message: {repr(message)}")
            except Exception:
                # Absolute fallback - do nothing to avoid infinite loops
                pass
    
    def debug(self, message: str, *args, **kwargs):
        self._safe_log(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        self._safe_log(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        self._safe_log(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        self._safe_log(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        self._safe_log(logging.CRITICAL, message, *args, **kwargs)


# 全局编码修复器
_global_encoding_fixer = None

def get_encoding_fixer() -> EncodingFixer:
    """获取全局编码修复器"""
    global _global_encoding_fixer
    if _global_encoding_fixer is None:
        _global_encoding_fixer = EncodingFixer()
    return _global_encoding_fixer


def apply_encoding_fixes(log_file: Optional[str] = None):
    """应用所有编码修复（快捷函数）"""
    fixer = get_encoding_fixer()
    return fixer.apply_all_fixes(log_file)


def get_safe_logger(name: str) -> SafeLogger:
    """获取安全日志记录器"""
    return SafeLogger(name)


# 在导入时自动应用基本修复
if not getattr(sys, '_encoding_fixed', False):
    try:
        apply_encoding_fixes()
        sys._encoding_fixed = True
    except Exception:
        pass  # 静默失败，避免影响其他功能