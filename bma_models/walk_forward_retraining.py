#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-Forward重训练系统
实现滚动窗口训练、run_id生成、热身期和强制重训间隔
"""

import numpy as np
import pandas as pd
import hashlib
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardConfig:
    """Walk-Forward配置"""
    train_window_months: int = 24  # 训练窗口(月) - 修改为2年
    step_size_days: int = 30  # 步长(天)，月度重训
    warmup_periods: int = 3  # 热身期数量
    force_refit_days: int = 90  # 强制重训间隔(天)
    min_train_samples: int = 1000  # 最小训练样本数
    max_lookback_days: int = 730  # 最大回看天数(2年) - 与训练窗口匹配
    window_type: str = 'rolling'  # 'rolling' 或 'expanding'
    enable_version_control: bool = True
    cache_models: bool = True
    
@dataclass  
class RunConfig:
    """运行配置"""
    run_id: str
    code_hash: str
    config_hash: str
    data_slice_hash: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    model_version: str
    creation_timestamp: str
    
class WalkForwardRetrainingSystem:
    """Walk-Forward重训练系统"""
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.cache_dir = Path("cache/walk_forward")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache = {}
        self.run_history = []
        
    def generate_run_id(self, 
                       code_content: str,
                       config_dict: Dict,
                       data_slice: pd.DataFrame,
                       train_start: str,
                       train_end: str) -> RunConfig:
        """
        生成run_id = code_hash + config_hash + data_slice_hash
        """
        # 1. 代码hash
        code_hash = hashlib.md5(code_content.encode('utf-8')).hexdigest()[:8]
        
        # 2. 配置hash
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]
        
        # 3. 数据切片hash
        data_str = f"{len(data_slice)}_{train_start}_{train_end}"
        if not data_slice.empty:
            # 添加数据的统计特征
            numeric_cols = data_slice.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data_stats = data_slice[numeric_cols].describe().values.flatten()
                data_str += f"_{hash(tuple(data_stats)) % 1000000}"
        
        data_slice_hash = hashlib.md5(data_str.encode('utf-8')).hexdigest()[:8]
        
        # 4. 生成run_id
        run_id = f"{code_hash}_{config_hash}_{data_slice_hash}"
        
        # 5. 创建运行配置
        run_config = RunConfig(
            run_id=run_id,
            code_hash=code_hash,
            config_hash=config_hash,
            data_slice_hash=data_slice_hash,
            train_start=train_start,
            train_end=train_end,
            test_start=train_end,  # 测试开始时间等于训练结束时间
            test_end=train_end,    # 将由调用者更新
            model_version="1.0.0",
            creation_timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"生成run_id: {run_id}")
        logger.info(f"  - 代码hash: {code_hash}")
        logger.info(f"  - 配置hash: {config_hash}")  
        logger.info(f"  - 数据hash: {data_slice_hash}")
        
        return run_config
    
    def should_retrain(self, 
                      last_train_date: Optional[str],
                      current_date: str,
                      force_retrain: bool = False) -> Tuple[bool, str]:
        """
        判断是否需要重训练
        
        Returns:
            (should_retrain, reason)
        """
        if force_retrain:
            return True, "强制重训练"
            
        if last_train_date is None:
            return True, "首次训练"
            
        # 计算天数差异
        last_date = pd.to_datetime(last_train_date).date()
        curr_date = pd.to_datetime(current_date).date()
        days_diff = (curr_date - last_date).days
        
        # 检查步长
        if days_diff >= self.config.step_size_days:
            return True, f"达到步长阈值({days_diff} >= {self.config.step_size_days}天)"
            
        # 检查强制重训间隔
        if days_diff >= self.config.force_refit_days:
            return True, f"达到强制重训间隔({days_diff} >= {self.config.force_refit_days}天)"
            
        return False, f"无需重训练(距离上次训练{days_diff}天)"
    
    def create_training_windows(self, 
                              data: pd.DataFrame,
                              date_column: str = 'date',
                              current_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        创建Walk-Forward训练窗口
        
        Returns:
            List of training windows with train/test splits
        """
        if current_date is None:
            current_date = datetime.now().strftime('%Y-%m-%d')
            
        data = data.sort_values(date_column)
        dates = pd.to_datetime(data[date_column])
        
        # 计算窗口参数
        train_window = timedelta(days=self.config.train_window_months * 30)
        step_size = timedelta(days=self.config.step_size_days)
        
        # 起始日期：当前日期往前推最大回看天数
        start_date = pd.to_datetime(current_date) - timedelta(days=self.config.max_lookback_days)
        end_date = pd.to_datetime(current_date)
        
        windows = []
        current_window_end = end_date
        
        # 🔒 安全机制：防止无限循环
        max_iterations = 100  # 最多100次迭代
        iteration_count = 0
        
        while current_window_end >= start_date + train_window and iteration_count < max_iterations:
            iteration_count += 1
            # 计算训练窗口
            if self.config.window_type == 'rolling':
                train_start = current_window_end - train_window
            else:  # expanding
                train_start = start_date
                
            train_end = current_window_end - step_size  # 留出gap
            
            # 测试窗口(下一个周期)
            test_start = current_window_end
            test_end = min(current_window_end + step_size, end_date)
            
            # 过滤数据
            train_mask = (dates >= train_start) & (dates <= train_end)
            test_mask = (dates >= test_start) & (dates <= test_end)
            
            train_samples = train_mask.sum()
            test_samples = test_mask.sum()
            
            # 检查样本数量
            if train_samples >= self.config.min_train_samples and test_samples > 0:
                window = {
                    'train_start': train_start.strftime('%Y-%m-%d'),
                    'train_end': train_end.strftime('%Y-%m-%d'), 
                    'test_start': test_start.strftime('%Y-%m-%d'),
                    'test_end': test_end.strftime('%Y-%m-%d'),
                    'train_samples': train_samples,
                    'test_samples': test_samples,
                    'train_mask': train_mask,
                    'test_mask': test_mask,
                    'window_id': f"wf_{train_start.strftime('%Y%m%d')}_{train_end.strftime('%Y%m%d')}"
                }
                windows.append(window)
            
            # 🔥 关键修复：更新循环变量，避免无限循环
            current_window_end = current_window_end - step_size
        
        # 🔒 安全检查：如果达到最大迭代次数，记录警告
        if iteration_count >= max_iterations:
            logger.warning(f"Walk-Forward窗口创建达到最大迭代次数({max_iterations})，可能存在配置问题")
                
        # 只保留最近的几个窗口(避免过度计算)
        max_windows = max(5, self.config.warmup_periods + 2)
        windows = sorted(windows, key=lambda x: x['train_end'])[-max_windows:]
        
        logger.info(f"创建{len(windows)}个Walk-Forward训练窗口")
        for i, w in enumerate(windows):
            logger.info(f"  窗口{i+1}: {w['train_start']} → {w['train_end']} "
                       f"(训练{w['train_samples']}, 测试{w['test_samples']})")
        
        return windows
    
    def is_warmup_complete(self, window_index: int) -> bool:
        """检查热身期是否完成"""
        return window_index >= self.config.warmup_periods
    
    def save_run_config(self, run_config: RunConfig, results: Optional[Dict] = None):
        """保存运行配置和结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存配置
        config_file = self.cache_dir / f"run_config_{run_config.run_id}_{timestamp}.json"
        config_data = asdict(run_config)
        if results:
            config_data['results'] = results
            
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False, default=str)
            
        # 更新历史记录
        self.run_history.append({
            'run_id': run_config.run_id,
            'timestamp': timestamp,
            'config_file': str(config_file)
        })
        
        logger.info(f"运行配置已保存: {config_file}")
        
    def load_cached_model(self, run_id: str) -> Optional[Any]:
        """加载缓存的模型"""
        if not self.config.cache_models:
            return None
            
        cache_file = self.cache_dir / f"model_{run_id}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"加载缓存模型: {cache_file}")
                return model
            except Exception as e:
                logger.warning(f"加载缓存模型失败: {e}")
                
        return None
        
    def cache_model(self, run_id: str, model: Any):
        """缓存模型"""
        if not self.config.cache_models:
            return
            
        cache_file = self.cache_dir / f"model_{run_id}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"模型已缓存: {cache_file}")
        except Exception as e:
            logger.warning(f"缓存模型失败: {e}")
    
    def cleanup_old_cache(self, keep_latest: int = 10):
        """清理旧缓存文件"""
        try:
            # 清理模型缓存
            model_files = list(self.cache_dir.glob("model_*.pkl"))
            if len(model_files) > keep_latest:
                # 按修改时间排序，保留最新的
                model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                for old_file in model_files[keep_latest:]:
                    old_file.unlink()
                    logger.info(f"清理旧模型缓存: {old_file}")
                    
            # 清理配置文件
            config_files = list(self.cache_dir.glob("run_config_*.json"))
            if len(config_files) > keep_latest * 2:
                config_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                for old_file in config_files[keep_latest * 2:]:
                    old_file.unlink()
                    
        except Exception as e:
            logger.warning(f"清理缓存失败: {e}")

def create_walk_forward_system(config: Optional[WalkForwardConfig] = None) -> WalkForwardRetrainingSystem:
    """创建Walk-Forward重训练系统"""
    if config is None:
        config = WalkForwardConfig()
    return WalkForwardRetrainingSystem(config)

# 🔥 集成到BMA系统的接口函数
def integrate_walk_forward_to_bma(data: pd.DataFrame,
                                 code_content: str,
                                 config_dict: Dict,
                                 current_date: Optional[str] = None) -> Dict[str, Any]:
    """
    集成Walk-Forward系统到BMA
    
    Args:
        data: 训练数据
        code_content: 代码内容(用于生成hash)
        config_dict: 配置字典
        current_date: 当前日期
    
    Returns:
        Walk-Forward结果字典
    """
    # 创建系统
    wf_config = WalkForwardConfig(
        train_window_months=config_dict.get('train_window_months', 24),
        step_size_days=config_dict.get('step_size_days', 30),
        force_refit_days=config_dict.get('force_refit_days', 90)
    )
    
    wf_system = create_walk_forward_system(wf_config)
    
    try:
        # 创建训练窗口
        windows = wf_system.create_training_windows(data, current_date=current_date)
        
        if not windows:
            logger.warning("无法创建有效的训练窗口")
            return {'success': False, 'error': '无有效窗口'}
        
        # 获取最新窗口
        latest_window = windows[-1]
        
        # 生成run_id
        train_data = data[latest_window['train_mask']]
        run_config = wf_system.generate_run_id(
            code_content=code_content,
            config_dict=config_dict,
            data_slice=train_data,
            train_start=latest_window['train_start'],
            train_end=latest_window['train_end']
        )
        
        # 更新测试时间
        run_config.test_start = latest_window['test_start']
        run_config.test_end = latest_window['test_end']
        
        # 检查是否需要重训练
        last_train_date = None  # 从历史记录中获取
        should_retrain, reason = wf_system.should_retrain(
            last_train_date, latest_window['train_end']
        )
        
        # 检查热身期
        warmup_complete = wf_system.is_warmup_complete(len(windows) - 1)
        
        return {
            'success': True,
            'run_config': asdict(run_config),
            'windows': windows,
            'latest_window': latest_window,
            'should_retrain': should_retrain,
            'retrain_reason': reason,
            'warmup_complete': warmup_complete,
            'total_windows': len(windows)
        }
        
    except Exception as e:
        logger.error(f"Walk-Forward集成失败: {e}")
        return {'success': False, 'error': str(e)}