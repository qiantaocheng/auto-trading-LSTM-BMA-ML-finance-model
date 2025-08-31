#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传统ML训练头 - 重定向到新的模块化实现
为了向后兼容，保留此文件并重定向到新的traditional_ml_head.py
"""

import warnings
warnings.filterwarnings('ignore')

# 导入新的传统ML训练头实现
from .traditional_ml_head import TraditionalMLHead

# 保持向后兼容的别名
EnhancedMLTrainer = TraditionalMLHead

# 弃用警告
import logging
logger = logging.getLogger(__name__)
logger.warning("enhanced_ml_trainer.py已弃用，请使用traditional_ml_head.py")


if __name__ == "__main__":
    print("此文件已弃用，请使用traditional_ml_head.py")
    
    # 重定向到新的测试
    from .traditional_ml_head import TraditionalMLHead
    
    print("重定向到新的传统ML训练头测试...")
    
    # 创建模拟数据
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n_samples = 500
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    X = pd.DataFrame(np.random.randn(n_samples, 5), 
                     columns=[f'feature_{i}' for i in range(5)])
    y = pd.Series(np.random.randn(n_samples))
    tickers = pd.Series(['AAPL'] * n_samples)
    
    # 创建模拟CV工厂
    def mock_cv_factory(dates_input):
        def cv_splitter(X_input, y_input):
            n = len(X_input)
            splits = []
            for i in range(2):  # 2折CV
                train_size = int(n * 0.8)
                test_start = train_size + i * 50
                test_end = min(test_start + 100, n)
                if test_end > test_start:
                    splits.append((list(range(train_size)), list(range(test_start, test_end))))
            return splits
        return cv_splitter
    
    # 创建训练头并测试
    trainer = TraditionalMLHead(enable_hyperparam_opt=False)
    result = trainer.fit(X, y, dates, tickers, mock_cv_factory)
    
    print(f"✅ 新版本测试成功: {result.get('metadata', {}).get('training_head', 'unknown')}")
    print(f"模型数量: {len(result.get('models', {}))}")
    print(f"CV指标: {result.get('cv', {}).get('avg_ic', 'N/A')}")