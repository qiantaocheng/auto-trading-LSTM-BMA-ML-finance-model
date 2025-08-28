#!/usr/bin/env python3
"""
Task Lifecycle Manager Compatibility Module
Provides basic task management functionality
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class TaskState(Enum):
    """Task execution states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskInfo:
    """Task information and metadata"""
    task_id: str
    name: str
    state: TaskState = TaskState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Any = None

class TaskLifecycleManager:
    """
    Basic task lifecycle management
    Provides task tracking and execution monitoring
    """
    
    def __init__(self):
        self.tasks: Dict[str, TaskInfo] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.logger = logger
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'average_execution_time': 0.0
        }
        
        logger.info("TaskLifecycleManager initialized")
    
    def create_task_id(self, name: str) -> str:
        """Generate a unique task ID"""
        timestamp = int(time.time() * 1000)
        return f"{name}_{timestamp}_{len(self.tasks)}"
    
    def register_task(self, name: str, task_id: str = None) -> str:
        """Register a new task"""
        if not task_id:
            task_id = self.create_task_id(name)
        
        task_info = TaskInfo(task_id=task_id, name=name)
        self.tasks[task_id] = task_info
        self.stats['total_tasks'] += 1
        
        logger.debug(f"Registered task: {task_id} ({name})")
        return task_id
    
    def start_task(self, task_id: str) -> bool:
        """Mark a task as started"""
        if task_id in self.tasks:
            self.tasks[task_id].state = TaskState.RUNNING
            self.tasks[task_id].started_at = time.time()
            logger.debug(f"Started task: {task_id}")
            return True
        return False
    
    def complete_task(self, task_id: str, result: Any = None) -> bool:
        """Mark a task as completed"""
        if task_id in self.tasks:
            task_info = self.tasks[task_id]
            task_info.state = TaskState.COMPLETED
            task_info.completed_at = time.time()
            task_info.result = result
            
            # Update statistics
            self.stats['completed_tasks'] += 1
            if task_info.started_at:
                execution_time = task_info.completed_at - task_info.started_at
                self._update_average_execution_time(execution_time)
            
            # Clean up active task reference
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
                
            logger.debug(f"Completed task: {task_id}")
            return True
        return False
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark a task as failed"""
        if task_id in self.tasks:
            task_info = self.tasks[task_id]
            task_info.state = TaskState.FAILED
            task_info.completed_at = time.time()
            task_info.error = error
            
            # Update statistics
            self.stats['failed_tasks'] += 1
            
            # Clean up active task reference
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
                
            logger.warning(f"Failed task: {task_id} - {error}")
            return True
        return False
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        if task_id in self.tasks:
            task_info = self.tasks[task_id]
            task_info.state = TaskState.CANCELLED
            task_info.completed_at = time.time()
            
            # Cancel active asyncio task if it exists
            if task_id in self.active_tasks:
                asyncio_task = self.active_tasks[task_id]
                if not asyncio_task.done():
                    asyncio_task.cancel()
                del self.active_tasks[task_id]
            
            # Update statistics
            self.stats['cancelled_tasks'] += 1
            
            logger.info(f"Cancelled task: {task_id}")
            return True
        return False
    
    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """Get task information"""
        return self.tasks.get(task_id)
    
    def get_active_tasks(self) -> List[TaskInfo]:
        """Get all active (running) tasks"""
        return [task for task in self.tasks.values() if task.state == TaskState.RUNNING]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get task execution statistics"""
        return self.stats.copy()
    
    def cleanup_completed_tasks(self, max_age_seconds: float = 3600) -> int:
        """Clean up old completed tasks"""
        now = time.time()
        to_remove = []
        
        for task_id, task_info in self.tasks.items():
            if (task_info.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED] and
                task_info.completed_at and 
                now - task_info.completed_at > max_age_seconds):
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.tasks[task_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old tasks")
        
        return len(to_remove)
    
    def _update_average_execution_time(self, execution_time: float):
        """Update average execution time statistics"""
        current_avg = self.stats['average_execution_time']
        completed_count = self.stats['completed_tasks']
        
        if completed_count == 1:
            self.stats['average_execution_time'] = execution_time
        else:
            # Weighted average
            self.stats['average_execution_time'] = (
                (current_avg * (completed_count - 1) + execution_time) / completed_count
            )

# Global task manager instance
_task_manager = None

def get_task_manager() -> TaskLifecycleManager:
    """Get the global task manager instance"""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskLifecycleManager()
    return _task_manager

def create_task_manager() -> TaskLifecycleManager:
    """Create a new task manager instance"""
    return TaskLifecycleManager()