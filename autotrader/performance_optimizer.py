#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化模块 - 替换subprocess调useand提升计算性能
提供直接模块导入、异步计算and缓存机制
"""

import asyncio
import importlib
import sys
import time
import logging
from typing import Optional, Dict, Any, Callable, List, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import threading
import weakref
import pickle
import hashlib
import json

@dataclass
class ModelResult:
    """模型运行结果"""
    success: bool
    execution_time: float
    output: List[str]
    error: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None
    cache_key: Optional[str] = None

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache", max_size: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.memory_cache: Dict[str, Any] = {}
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger("CacheManager")
    
    def _generate_cache_key(self, script_path: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        key_data = {
            'script': str(script_path),
            'params': sorted(params.items()),
            'timestamp': int(time.time() / 3600)  # 按小when分组
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[ModelResult]:
        """retrieval缓存"""
        # 先check内存缓存
        if cache_key in self.memory_cache:
            self.logger.debug(f"内存缓存命in: {cache_key}")
            return self.memory_cache[cache_key]
        
        # check磁盘缓存
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                
                # 加载to内存缓存
                self.memory_cache[cache_key] = result
                self.logger.debug(f"磁盘缓存命in: {cache_key}")
                return result
                
            except Exception as e:
                self.logger.warning(f"加载缓存failed {cache_key}: {e}")
        
        return None
    
    def set(self, cache_key: str, result: ModelResult):
        """settings缓存"""
        # settings内存缓存
        self.memory_cache[cache_key] = result
        
        # settings磁盘缓存
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            # updates元数据
            self.cache_metadata[cache_key] = {
                'created': time.time(),
                'size': cache_file.stat().st_size,
                'execution_time': result.execution_time
            }
            
            self.logger.debug(f"缓存保存: {cache_key}")
            
        except Exception as e:
            self.logger.warning(f"保存缓存failed {cache_key}: {e}")
        
        # 清理过期缓存
        self._cleanup_cache()
    
    def _cleanup_cache(self):
        """清理过期缓存"""
        if len(self.memory_cache) > self.max_size:
            # 按创建when间排序，删除最老
            sorted_keys = sorted(
                self.cache_metadata.keys(),
                key=lambda k: self.cache_metadata[k]['created']
            )
            
            for key in sorted_keys[:len(self.memory_cache) - self.max_size]:
                self.memory_cache.pop(key, None)
                cache_file = self.cache_dir / f"{key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                self.cache_metadata.pop(key, None)

class DirectModuleExecutor:
    """直接模块执行器 - 替代subprocess"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("DirectModuleExecutor")
        self.cache_manager = CacheManager()
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.process_pool = ProcessPoolExecutor(max_workers=1)
    
    async def execute_bma_model(self, script_path: str, start_date: str, end_date: str,
                               use_cache: bool = True, use_async: bool = True, extra_args: Optional[List[str]] = None, 
                               progress_callback: Optional[Callable] = None) -> ModelResult:
        """执行BMA模型（优化版）"""
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'script_path': script_path,
            'extra_args': extra_args or [],
            'progress_callback': progress_callback
        }
        
        # check缓存
        cache_key = None
        if use_cache:
            # as缓存生成键when排除progress_callback（函数not能序列化）
            cache_params = {k: v for k, v in params.items() if k != 'progress_callback'}
            cache_key = self.cache_manager._generate_cache_key(script_path, cache_params)
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                self.logger.info(f"使use缓存结果: {cache_key}")
                return cached_result
        
        # 执行模型
        start_time = time.time()
        
        try:
            if use_async:
                # 异步执行（in线程池in）
                result = await self._execute_model_async(script_path, params)
            else:
                # 同步执行
                result = await self._execute_model_direct(script_path, params)
            
            execution_time = time.time() - start_time
            
            # 创建结果for象
            model_result = ModelResult(
                success=result['success'],
                execution_time=execution_time,
                output=result['output'],
                error=result.get('error'),
                result_data=result.get('result_data'),
                cache_key=cache_key
            )
            
            # 保存to缓存
            if use_cache and model_result.success:
                self.cache_manager.set(cache_key, model_result)
            
            return model_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"模型执行异常: {e}")
            
            return ModelResult(
                success=False,
                execution_time=execution_time,
                output=[],
                error=str(e)
            )
    
    async def _execute_model_direct(self, script_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """直接模块导入执行"""
        try:
            # will脚本路径转换as模块路径
            script_path_obj = Path(script_path)
            if not script_path_obj.exists():
                raise FileNotFoundError(f"脚本文件not存in: {script_path}")
            
            # 动态导入模块
            module_name = script_path_obj.stem
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"no法加载模块: {script_path}")
            
            module = importlib.util.module_from_spec(spec)
            
            # settings模块参数
            old_argv = sys.argv
            try:
                # 模拟命令行参数
                sys.argv = [
                    script_path,
                    '--start-date', params['start_date'],
                    '--end-date', params['end_date']
                ]
                
                # 捕获输出
                output_lines = []
                
                # 重定to输出
                import io
                import contextlib
                
                output_buffer = io.StringIO()
                
                with contextlib.redirect_stdout(output_buffer):
                    with contextlib.redirect_stderr(output_buffer):
                        # 执行模块
                        spec.loader.exec_module(module)
                
                # retrieval输出
                output_content = output_buffer.getvalue()
                if output_content:
                    output_lines = output_content.strip().split('\n')
                
                # 尝试retrieval结果数据
                result_data = {}
                if hasattr(module, 'get_results'):
                    result_data = module.get_results()
                elif hasattr(module, 'results'):
                    result_data = module.results
                
                return {
                    'success': True,
                    'output': output_lines,
                    'result_data': result_data
                }
                
            finally:
                sys.argv = old_argv
            
        except Exception as e:
            self.logger.error(f"直接执行模块failed: {e}")
            return {
                'success': False,
                'output': [],
                'error': str(e)
            }
    
    async def _execute_model_async(self, script_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """异步执行模型"""
        loop = asyncio.get_event_loop()
        
        # in线程池in执行，避免阻塞事件循环
        result = await loop.run_in_executor(
            self.thread_pool,
            self._execute_model_sync,
            script_path,
            params
        )
        
        return result
    
    def _execute_model_sync(self, script_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """同步执行模型（in线程池in调use）"""
        try:
            # 使use更安全方式导入and执行
            import runpy
            
            # settings环境变量传递参数
            import os
            old_env = dict(os.environ)
            
            try:
                extra_args = params.get('extra_args', [])
                
                # if果has额外参数，使usesubprocess执行；否则使userunpy（更快）
                if extra_args:
                    # 使usesubprocess执行脚本以支持命令行参数andreal-time输出
                    import subprocess
                    
                    # 构建命令（显式传递日期范围and额外参数）
                    cmd = [
                        sys.executable,
                        script_path,
                        '--start-date', params['start_date'],
                        '--end-date', params['end_date']
                    ] + extra_args
                    
                    # settings环境变量
                    env = os.environ.copy()
                    env.update({
                        'BMA_START_DATE': params['start_date'],
                        'BMA_END_DATE': params['end_date']
                    })
                    
                    # retrieval进度回调
                    progress_callback = params.get('progress_callback')
                    
                    # start进程，real-time读取输出
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=env,
                        bufsize=1,  # 行缓冲
                        universal_newlines=True
                    )
                    
                    output_lines = []
                    error_lines = []
                    
                    # real-time读取输出
                    import select
                    import threading
                    
                    def read_output(pipe, lines_list, is_error=False):
                        try:
                            for line in iter(pipe.readline, ''):
                                if line:
                                    line = line.rstrip()
                                    lines_list.append(line)
                                    
                                    # 调use进度回调
                                    if progress_callback and line.strip():
                                        from dataclasses import dataclass
                                        from typing import List
                                        
                                        @dataclass
                                        class ProgressResult:
                                            output: List[str]
                                        
                                        progress_result = ProgressResult(output=[line])
                                        try:
                                            progress_callback(progress_result)
                                        except Exception as e:
                                            print(f"进度回调错误: {e}")
                        except Exception as e:
                            print(f"读取输出错误: {e}")
                        finally:
                            pipe.close()
                    
                    # start输出读取线程
                    stdout_thread = threading.Thread(target=read_output, args=(process.stdout, output_lines, False))
                    stderr_thread = threading.Thread(target=read_output, args=(process.stderr, error_lines, True))
                    
                    stdout_thread.start()
                    stderr_thread.start()
                    
                    # 等待进程completed
                    return_code = process.wait()
                    
                    # 等待输出读取线程completed
                    stdout_thread.join(timeout=5)
                    stderr_thread.join(timeout=5)
                    
                    output_content = '\n'.join(output_lines)
                    error_content = '\n'.join(error_lines)
                    
                    if return_code != 0:
                        raise RuntimeError(f"脚本执行failed (返回码: {return_code}): {error_content}")
                        
                else:
                    # 没has额外参数，使userunpy（更快）
                    os.environ['BMA_START_DATE'] = params['start_date']
                    os.environ['BMA_END_DATE'] = params['end_date']
                    
                    # 捕获输出
                    import io
                    import contextlib
                    
                    output_buffer = io.StringIO()
                    error_buffer = io.StringIO()
                    
                    with contextlib.redirect_stdout(output_buffer):
                        with contextlib.redirect_stderr(error_buffer):
                            # 使userunpy执行脚本
                            runpy.run_path(script_path, run_name='__main__')
                    
                    # retrieval输出
                    output_content = output_buffer.getvalue()
                    error_content = error_buffer.getvalue()
                
                output_lines = []
                if output_content:
                    output_lines.extend(output_content.strip().split('\n'))
                if error_content:
                    output_lines.extend(['STDERR:'] + error_content.strip().split('\n'))
                
                return {
                    'success': True,
                    'output': output_lines,
                    'result_data': {}
                }
                
            finally:
                # 恢复环境变量
                os.environ.clear()
                os.environ.update(old_env)
            
        except Exception as e:
            return {
                'success': False,
                'output': [f"执行异常: {str(e)}"],
                'error': str(e)
            }
    
    async def shutdown(self):
        """关闭执行器"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("PerformanceOptimizer")
        self.executor = DirectModuleExecutor(logger)
        
        # 性能统计
        self.stats = {
            'subprocess_calls_replaced': 0,
            'cache_hits': 0,
            'total_execution_time_saved': 0.0,
            'average_speedup': 0.0
        }
    
    async def optimize_bma_execution(self, script_path: str, start_date: str, end_date: str,
                                   progress_callback: Optional[Callable] = None, extra_args: Optional[List[str]] = None) -> ModelResult:
        """优化BMA模型执行"""
        self.logger.info(f"优化执行BMA模型: {start_date} -> {end_date}")
        
        # 记录替换subprocess调use
        self.stats['subprocess_calls_replaced'] += 1
        
        # 执行优化模型
        result = await self.executor.execute_bma_model(
            script_path, start_date, end_date, 
            use_cache=True, use_async=True, extra_args=extra_args,
            progress_callback=progress_callback
        )
        
        # updates统计
        if result.cache_key:
            self.stats['cache_hits'] += 1
        
        # 估算节省when间（相比subprocess）
        estimated_subprocess_time = result.execution_time * 2.5  # 估算subprocess比直接执行慢2.5倍
        time_saved = estimated_subprocess_time - result.execution_time
        self.stats['total_execution_time_saved'] += time_saved
        
        # 计算平均加速比
        if self.stats['subprocess_calls_replaced'] > 0:
            self.stats['average_speedup'] = (
                self.stats['total_execution_time_saved'] / 
                self.stats['subprocess_calls_replaced']
            ) + 1.0
        
        # 执行进度回调
        if progress_callback:
            try:
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(result)
                else:
                    progress_callback(result)
            except Exception as e:
                self.logger.warning(f"进度回调执行failed: {e}")
        
        self.logger.info(f"模型执行completed: {'success' if result.success else 'failed'}, "
                        f"耗when: {result.execution_time:.2f}s")
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """retrieval性能统计"""
        return {
            'optimization_stats': self.stats.copy(),
            'cache_stats': {
                'memory_cache_size': len(self.executor.cache_manager.memory_cache),
                'disk_cache_dir': str(self.executor.cache_manager.cache_dir),
                'cache_metadata_count': len(self.executor.cache_manager.cache_metadata)
            },
            'executor_stats': {
                'thread_pool_active': self.executor.thread_pool._threads,
                'process_pool_active': getattr(self.executor.process_pool, '_processes', 0)
            }
        }
    
    async def clear_cache(self):
        """清理缓存"""
        try:
            # 清理内存缓存
            self.executor.cache_manager.memory_cache.clear()
            
            # 清理磁盘缓存
            import shutil
            if self.executor.cache_manager.cache_dir.exists():
                shutil.rmtree(self.executor.cache_manager.cache_dir)
                self.executor.cache_manager.cache_dir.mkdir(exist_ok=True)
            
            # 清理元数据
            self.executor.cache_manager.cache_metadata.clear()
            
            self.logger.info("缓存清理")
            
        except Exception as e:
            self.logger.error(f"清理缓存failed: {e}")
    
    async def shutdown(self):
        """关闭优化器"""
        await self.executor.shutdown()

# 全局实例
_global_performance_optimizer: Optional[PerformanceOptimizer] = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """retrieval全局性能优化器"""
    global _global_performance_optimizer
    if _global_performance_optimizer is None:
        _global_performance_optimizer = PerformanceOptimizer()
    return _global_performance_optimizer
