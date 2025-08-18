#!/usr/bin/env python3
"""
训练进度监控器
实时跟踪和显示训练进度，预测完成时间
"""

import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class TrainingStage:
    """训练阶段信息"""
    name: str
    total_items: int
    completed_items: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None
    
    @property
    def progress_percent(self) -> float:
        if self.total_items == 0:
            return 100.0
        return (self.completed_items / self.total_items) * 100
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.time()
        return end_time - self.start_time
    
    @property
    def items_per_second(self) -> float:
        duration = self.duration_seconds
        if duration <= 0:
            return 0.0
        return self.completed_items / duration
    
    @property
    def estimated_remaining_seconds(self) -> float:
        rate = self.items_per_second
        if rate <= 0:
            return 0.0
        remaining_items = self.total_items - self.completed_items
        return remaining_items / rate

@dataclass 
class OverallProgress:
    """总体进度信息"""
    total_stages: int
    completed_stages: int = 0
    current_stage: Optional[str] = None
    start_time: Optional[float] = None
    total_estimated_duration: float = 0.0
    
    @property
    def progress_percent(self) -> float:
        if self.total_stages == 0:
            return 100.0
        return (self.completed_stages / self.total_stages) * 100
    
    @property
    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

class TrainingProgressMonitor:
    """训练进度监控器"""
    
    def __init__(self, 
                 save_dir: str = "logs/progress",
                 update_interval: float = 5.0,
                 enable_auto_save: bool = True):
        """
        初始化训练进度监控器
        
        Args:
            save_dir: 进度保存目录
            update_interval: 更新间隔(秒)
            enable_auto_save: 启用自动保存
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.update_interval = update_interval
        self.enable_auto_save = enable_auto_save
        
        # 进度状态
        self.stages: Dict[str, TrainingStage] = {}
        self.stage_order: List[str] = []
        self.overall_progress = OverallProgress(total_stages=0)
        self.is_training = False
        
        # 监控线程
        self.monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # 回调函数
        self.progress_callbacks: List[Callable] = []
        self.completion_callbacks: List[Callable] = []
        
        # 统计信息
        self.last_update_time = time.time()
        self.update_count = 0
        
    def register_progress_callback(self, callback: Callable):
        """注册进度更新回调"""
        self.progress_callbacks.append(callback)
    
    def register_completion_callback(self, callback: Callable):
        """注册完成回调"""
        self.completion_callbacks.append(callback)
    
    def add_stage(self, stage_name: str, total_items: int):
        """添加训练阶段"""
        stage = TrainingStage(
            name=stage_name,
            total_items=total_items
        )
        self.stages[stage_name] = stage
        self.stage_order.append(stage_name)
        self.overall_progress.total_stages = len(self.stages)
        
        logger.debug(f"添加训练阶段: {stage_name} ({total_items} 项)")
    
    def start_training(self):
        """开始训练"""
        self.is_training = True
        self.overall_progress.start_time = time.time()
        
        # 启动监控线程
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self._stop_event.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
        
        logger.info("训练进度监控已启动")
    
    def start_stage(self, stage_name: str):
        """开始阶段"""
        if stage_name not in self.stages:
            logger.warning(f"未知阶段: {stage_name}")
            return
        
        stage = self.stages[stage_name]
        stage.status = "running"
        stage.start_time = time.time()
        stage.completed_items = 0
        
        self.overall_progress.current_stage = stage_name
        
        logger.info(f"开始阶段: {stage_name}")
        self._trigger_progress_callbacks()
    
    def update_stage_progress(self, stage_name: str, completed_items: int, error_message: Optional[str] = None):
        """更新阶段进度"""
        if stage_name not in self.stages:
            logger.warning(f"未知阶段: {stage_name}")
            return
        
        stage = self.stages[stage_name]
        stage.completed_items = min(completed_items, stage.total_items)
        
        if error_message:
            stage.error_message = error_message
            stage.status = "failed"
        
        self._trigger_progress_callbacks()
    
    def complete_stage(self, stage_name: str, success: bool = True, error_message: Optional[str] = None):
        """完成阶段"""
        if stage_name not in self.stages:
            logger.warning(f"未知阶段: {stage_name}")
            return
        
        stage = self.stages[stage_name]
        stage.end_time = time.time()
        stage.completed_items = stage.total_items
        
        if success:
            stage.status = "completed"
            self.overall_progress.completed_stages += 1
        else:
            stage.status = "failed"
            stage.error_message = error_message
        
        logger.info(f"阶段完成: {stage_name} ({'成功' if success else '失败'})")
        self._trigger_progress_callbacks()
    
    def complete_training(self, success: bool = True):
        """完成训练"""
        self.is_training = False
        
        # 停止监控线程
        self._stop_event.set()
        
        # 触发完成回调
        for callback in self.completion_callbacks:
            try:
                callback(success, self.get_summary())
            except Exception as e:
                logger.error(f"完成回调错误: {e}")
        
        logger.info(f"训练完成: {'成功' if success else '失败'}")
    
    def _trigger_progress_callbacks(self):
        """触发进度更新回调"""
        summary = self.get_summary()
        
        for callback in self.progress_callbacks:
            try:
                callback(summary)
            except Exception as e:
                logger.error(f"进度回调错误: {e}")
    
    def _monitor_loop(self):
        """监控循环"""
        while not self._stop_event.wait(self.update_interval):
            if not self.is_training:
                break
            
            try:
                self.update_count += 1
                self.last_update_time = time.time()
                
                # 自动保存进度
                if self.enable_auto_save and self.update_count % 5 == 0:
                    self.save_progress()
                
                # 显示进度摘要
                if self.update_count % 2 == 0:  # 每10秒显示一次
                    summary = self.get_summary()
                    logger.info(self._format_progress_message(summary))
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
    
    def _format_progress_message(self, summary: Dict[str, Any]) -> str:
        """格式化进度消息"""
        overall = summary['overall_progress']
        current_stage_info = summary.get('current_stage_detail')
        
        message_parts = [
            f"总进度: {overall['progress_percent']:.1f}%",
            f"已完成阶段: {overall['completed_stages']}/{overall['total_stages']}"
        ]
        
        if current_stage_info:
            stage_progress = current_stage_info['progress_percent']
            remaining_time = current_stage_info['estimated_remaining_seconds']
            
            message_parts.extend([
                f"当前阶段: {current_stage_info['name']} ({stage_progress:.1f}%)",
                f"预计剩余: {self._format_duration(remaining_time)}"
            ])
        
        return " | ".join(message_parts)
    
    def _format_duration(self, seconds: float) -> str:
        """格式化时间长度"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.1f}分钟"
        else:
            return f"{seconds/3600:.1f}小时"
    
    def get_summary(self) -> Dict[str, Any]:
        """获取进度摘要"""
        current_stage_name = self.overall_progress.current_stage
        current_stage_detail = None
        
        if current_stage_name and current_stage_name in self.stages:
            stage = self.stages[current_stage_name]
            current_stage_detail = {
                'name': stage.name,
                'progress_percent': stage.progress_percent,
                'completed_items': stage.completed_items,
                'total_items': stage.total_items,
                'duration_seconds': stage.duration_seconds,
                'items_per_second': stage.items_per_second,
                'estimated_remaining_seconds': stage.estimated_remaining_seconds,
                'status': stage.status,
                'error_message': stage.error_message
            }
        
        # 计算总体预计完成时间
        total_estimated_remaining = 0.0
        for stage_name in self.stage_order:
            stage = self.stages[stage_name]
            if stage.status in ['pending', 'running']:
                if stage.status == 'running':
                    total_estimated_remaining += stage.estimated_remaining_seconds
                else:
                    # 对于未开始的阶段，使用平均估算
                    avg_rate = self._get_average_processing_rate()
                    if avg_rate > 0:
                        total_estimated_remaining += stage.total_items / avg_rate
        
        return {
            'overall_progress': {
                'progress_percent': self.overall_progress.progress_percent,
                'completed_stages': self.overall_progress.completed_stages,
                'total_stages': self.overall_progress.total_stages,
                'elapsed_time': self.overall_progress.elapsed_time,
                'estimated_remaining_seconds': total_estimated_remaining
            },
            'current_stage_detail': current_stage_detail,
            'stages': {name: {
                'name': stage.name,
                'progress_percent': stage.progress_percent,
                'status': stage.status,
                'duration_seconds': stage.duration_seconds,
                'items_per_second': stage.items_per_second
            } for name, stage in self.stages.items()},
            'timestamp': datetime.now().isoformat(),
            'is_training': self.is_training
        }
    
    def _get_average_processing_rate(self) -> float:
        """获取平均处理速率"""
        completed_stages = [stage for stage in self.stages.values() if stage.status == 'completed']
        
        if not completed_stages:
            return 0.0
        
        total_rate = sum(stage.items_per_second for stage in completed_stages)
        return total_rate / len(completed_stages)
    
    def save_progress(self, filename: Optional[str] = None):
        """保存进度到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_progress_{timestamp}.json"
        
        filepath = self.save_dir / filename
        
        try:
            summary = self.get_summary()
            
            # 添加详细的阶段信息
            detailed_stages = {}
            for name, stage in self.stages.items():
                detailed_stages[name] = asdict(stage)
            
            save_data = {
                'summary': summary,
                'detailed_stages': detailed_stages,
                'stage_order': self.stage_order,
                'overall_progress': asdict(self.overall_progress)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"进度已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存进度失败: {e}")
    
    def load_progress(self, filename: str):
        """从文件加载进度"""
        filepath = self.save_dir / filename
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 恢复阶段信息
            if 'detailed_stages' in data:
                for name, stage_data in data['detailed_stages'].items():
                    stage = TrainingStage(**stage_data)
                    self.stages[name] = stage
            
            # 恢复阶段顺序
            if 'stage_order' in data:
                self.stage_order = data['stage_order']
            
            # 恢复总体进度
            if 'overall_progress' in data:
                self.overall_progress = OverallProgress(**data['overall_progress'])
            
            logger.info(f"进度已加载: {filepath}")
            
        except Exception as e:
            logger.error(f"加载进度失败: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        completed_stages = [s for s in self.stages.values() if s.status == 'completed']
        
        if not completed_stages:
            return {'message': 'No completed stages yet'}
        
        # 计算统计信息
        durations = [s.duration_seconds for s in completed_stages]
        rates = [s.items_per_second for s in completed_stages]
        
        return {
            'completed_stages_count': len(completed_stages),
            'total_elapsed_time': sum(durations),
            'average_stage_duration': sum(durations) / len(durations),
            'average_processing_rate': sum(rates) / len(rates),
            'fastest_stage': max(completed_stages, key=lambda s: s.items_per_second).name,
            'slowest_stage': min(completed_stages, key=lambda s: s.items_per_second).name
        }


def create_progress_monitor(**kwargs) -> TrainingProgressMonitor:
    """创建训练进度监控器的工厂函数"""
    return TrainingProgressMonitor(**kwargs)