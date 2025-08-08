#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每周一自动运行脚本
启动模型验证器的调度器，负责每周一自动运行BMA和LSTM模型结合
"""

import os
import sys
import logging
import time
import signal
from datetime import datetime
from pathlib import Path

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_validator import ModelValidator, WeeklyScheduler


def setup_logging():
    """设置日志"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"weekly_runner_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class WeeklyAutoRunner:
    """每周自动运行管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validator = None
        self.scheduler = None
        self.is_running = False
        
    def initialize(self):
        """初始化组件"""
        try:
            # 创建模型验证器
            self.validator = ModelValidator()
            self.logger.info("[OK] 模型验证器创建成功")
            
            # 创建调度器
            self.scheduler = WeeklyScheduler(self.validator)
            self.logger.info("[OK] 每周调度器创建成功")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] 初始化失败: {e}")
            return False
    
    def start(self):
        """启动自动运行器"""
        if not self.initialize():
            return False
        
        try:
            # 启动调度器
            self.scheduler.start_scheduler()
            self.is_running = True
            
            self.logger.info("=" * 60)
            self.logger.info("每周自动运行器已启动")
            self.logger.info("=" * 60)
            self.logger.info("调度计划:")
            self.logger.info("  - 每周一 09:00: 运行模型验证和结合")
            self.logger.info("  - 每天   10:00: 检查新模型文件")
            self.logger.info("程序将持续运行，按 Ctrl+C 停止")
            self.logger.info("=" * 60)
            
            # 先运行一次立即验证
            self.logger.info("执行立即验证...")
            report = self.validator.run_validation()
            
            if report['combined']['success']:
                top_stocks = [item['ticker'] for item in report['combined']['recommendations'][:5]]
                self.logger.info(f"[OK] 立即验证成功，Top5: {top_stocks}")
            else:
                self.logger.warning("[WARNING] 立即验证完成但存在问题")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] 启动失败: {e}")
            return False
    
    def stop(self):
        """停止自动运行器"""
        if self.scheduler:
            self.scheduler.stop_scheduler()
        
        self.is_running = False
        self.logger.info("[OK] 每周自动运行器已停止")
    
    def run(self):
        """运行主循环"""
        if not self.start():
            return 1
        
        try:
            # 持续运行
            while self.is_running:
                time.sleep(10)  # 每10秒检查一次
                
        except KeyboardInterrupt:
            self.logger.info("[INFO] 用户中断程序")
        except Exception as e:
            self.logger.error(f"[ERROR] 运行异常: {e}")
            return 1
        finally:
            self.stop()
        
        return 0


def signal_handler(signum, frame):
    """信号处理器"""
    print("\n[INFO] 接收到停止信号，正在关闭...")
    global runner
    if runner:
        runner.stop()
    sys.exit(0)


def main():
    """主函数"""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("每周一自动运行器启动")
    logger.info("=" * 60)
    
    global runner
    runner = WeeklyAutoRunner()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        exit_code = runner.run()
        logger.info(f"程序结束，退出代码: {exit_code}")
        return exit_code
        
    except Exception as e:
        logger.error(f"[ERROR] 程序异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)