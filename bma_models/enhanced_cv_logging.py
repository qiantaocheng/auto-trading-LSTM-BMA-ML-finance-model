#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced CV Logging System
增强的交叉验证日志系统
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class EnhancedCVLogger:
    """增强的CV日志记录器"""
    
    def __init__(self):
        """初始化CV日志记录器"""
        self.cv_logs = []
        self.fold_details = {}
        
    def log_cv_fold(self, fold_id: str, train_size: int, test_size: int, 
                   train_dates: Optional[tuple] = None, test_dates: Optional[tuple] = None,
                   gap_days: Optional[int] = None, embargo_satisfied: bool = True):
        """记录CV fold信息"""
        fold_info = {
            'fold_id': fold_id,
            'train_size': train_size,
            'test_size': test_size,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'gap_days': gap_days,
            'embargo_satisfied': embargo_satisfied,
            'timestamp': datetime.now()
        }
        
        self.fold_details[fold_id] = fold_info
        
        logger.info(f"Fold {fold_id}: 训练样本{train_size}, 测试样本{test_size}")
        if gap_days is not None:
            logger.info(f"时间间隔 {gap_days}天 {'[OK]' if embargo_satisfied else '[WARN]'}")
    
    def log_oof_status(self, ticker: str, oof_predictions: pd.Series, status: str = "success"):
        """记录OOF状态"""
        logger.info(f"OOF状态 - {ticker}: {status}, 预测数量: {len(oof_predictions)}")
        
    def get_cv_summary(self) -> Dict[str, Any]:
        """获取CV总结"""
        return {
            'total_folds': len(self.fold_details),
            'fold_details': self.fold_details,
            'logs_count': len(self.cv_logs)
        }