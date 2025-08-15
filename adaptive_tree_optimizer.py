#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应加树优化器 - 基于验证集提升斜率的智能加树策略
实现第二层优化：仅对高提升斜率股票增加树的数量
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdaptiveTreeOptimizer:
    """
    自适应加树优化器
    基于验证集性能提升斜率决定是否增加树的数量
    """
    
    def __init__(self, 
                 slope_threshold_ic: float = 0.002,     # IC提升斜率阈值
                 slope_threshold_mse: float = 0.01,     # MSE下降斜率阈值(1%)
                 tree_increment: int = 20,              # 每次增加的树数量
                 top_percentile: float = 0.2,           # 选择前20%的股票
                 max_trees_xgb: int = 150,              # XGB最大树数
                 max_trees_lgb: int = 150,              # LightGBM最大树数
                 max_trees_rf: int = 200):              # RandomForest最大树数
        
        self.slope_threshold_ic = slope_threshold_ic
        self.slope_threshold_mse = slope_threshold_mse
        self.tree_increment = tree_increment
        self.top_percentile = top_percentile
        self.max_trees_xgb = max_trees_xgb
        self.max_trees_lgb = max_trees_lgb
        self.max_trees_rf = max_trees_rf
        
        # 斜率计算历史
        self.slope_history = {}
        
    def calculate_performance_slope(self, 
                                  performance_history: List[float], 
                                  tree_counts: List[int]) -> float:
        """
        计算性能提升斜率
        
        Args:
            performance_history: 性能指标历史 (IC 或 MSE)
            tree_counts: 对应的树数量
            
        Returns:
            斜率值 (Δ性能 / Δ树数)
        """
        if len(performance_history) < 2:
            return 0.0
        
        # 计算最近几个点的斜率
        if len(performance_history) >= 3:
            # 使用最近3个点计算斜率
            recent_perf = performance_history[-3:]
            recent_trees = tree_counts[-3:]
        else:
            recent_perf = performance_history
            recent_trees = tree_counts
        
        # 线性回归计算斜率
        x = np.array(recent_trees)
        y = np.array(recent_perf)
        
        if len(x) < 2:
            return 0.0
        
        # 简单斜率计算: (y2 - y1) / (x2 - x1)
        slope = (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0.0
        
        return slope
    
    def should_add_trees(self, 
                        stock_id: str,
                        current_performance: Dict[str, float],
                        current_tree_count: int,
                        model_type: str) -> bool:
        """
        判断是否应该为该股票增加树数量
        
        Args:
            stock_id: 股票代码
            current_performance: 当前性能指标 {'ic': value, 'mse': value}
            current_tree_count: 当前树数量
            model_type: 模型类型 ('xgboost', 'lightgbm', 'rf')
            
        Returns:
            是否应该增加树
        """
        # 检查是否已达到最大树数
        max_trees = {
            'xgboost': self.max_trees_xgb,
            'lightgbm': self.max_trees_lgb,
            'rf': self.max_trees_rf
        }.get(model_type, 150)
        
        if current_tree_count >= max_trees:
            return False
        
        # 初始化股票历史记录
        key = f"{stock_id}_{model_type}"
        if key not in self.slope_history:
            self.slope_history[key] = {
                'ic_history': [],
                'mse_history': [],
                'tree_counts': [],
                'timestamps': []
            }
        
        history = self.slope_history[key]
        
        # 记录当前性能
        history['ic_history'].append(current_performance.get('ic', 0.0))
        history['mse_history'].append(current_performance.get('mse', 1.0))
        history['tree_counts'].append(current_tree_count)
        history['timestamps'].append(time.time())
        
        # 需要至少2个数据点才能计算斜率
        if len(history['ic_history']) < 2:
            return True  # 初始阶段默认加树
        
        # 计算IC和MSE的斜率
        ic_slope = self.calculate_performance_slope(
            history['ic_history'], history['tree_counts']
        )
        
        # MSE斜率需要取负值(MSE下降是好的)
        mse_slope = -self.calculate_performance_slope(
            history['mse_history'], history['tree_counts']
        )
        
        # 判断是否满足加树条件
        ic_condition = ic_slope >= self.slope_threshold_ic
        mse_condition = mse_slope >= self.slope_threshold_mse
        
        logger.debug(f"{stock_id} {model_type}: IC斜率={ic_slope:.4f}, MSE斜率={mse_slope:.4f}")
        
        return ic_condition or mse_condition
    
    def adaptive_train_xgboost(self, 
                              X: pd.DataFrame, 
                              y: pd.Series, 
                              stock_id: str,
                              base_params: Dict = None) -> Tuple[Any, Dict]:
        """
        自适应训练XGBoost模型
        使用较大学习率 + 早停 + 动态加树
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        # 基础参数
        if base_params is None:
            base_params = {
                'max_depth': 4,
                'learning_rate': 0.3,  # 较大学习率
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'tree_method': 'hist',
                'random_state': 42,
                'n_jobs': 1
            }
        
        # 初始树数量
        current_trees = 50
        performance_history = []
        
        # 时序交叉验证
        tscv = TimeSeriesSplit(n_splits=3)
        
        best_model = None
        best_performance = {'ic': -np.inf, 'mse': np.inf}
        
        while current_trees <= self.max_trees_xgb:
            # 训练当前树数量的模型
            params = base_params.copy()
            params['n_estimators'] = current_trees
            params['early_stopping_rounds'] = min(15, current_trees // 3)
            
            model = xgb.XGBRegressor(**params)
            
            # 交叉验证评估
            cv_scores = {'ic': [], 'mse': []}
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 训练模型
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # 预测和评估
                y_pred = model.predict(X_val)
                
                # 计算IC (信息系数)
                ic = np.corrcoef(y_val, y_pred)[0, 1] if len(y_val) > 1 else 0.0
                ic = ic if not np.isnan(ic) else 0.0
                
                # 计算MSE
                mse = mean_squared_error(y_val, y_pred)
                
                cv_scores['ic'].append(ic)
                cv_scores['mse'].append(mse)
            
            # 计算平均性能
            avg_ic = np.mean(cv_scores['ic'])
            avg_mse = np.mean(cv_scores['mse'])
            
            current_performance = {'ic': avg_ic, 'mse': avg_mse}
            performance_history.append(current_performance)
            
            # 更新最佳模型
            if avg_ic > best_performance['ic']:
                best_performance = current_performance
                # 重新训练最佳模型
                best_model = xgb.XGBRegressor(**params)
                best_model.fit(X, y)
            
            logger.info(f"{stock_id} XGBoost {current_trees}棵树: IC={avg_ic:.4f}, MSE={avg_mse:.4f}")
            
            # 判断是否继续加树
            if not self.should_add_trees(stock_id, current_performance, current_trees, 'xgboost'):
                logger.info(f"{stock_id} XGBoost停止加树于{current_trees}棵")
                break
            
            current_trees += self.tree_increment
        
        return best_model, best_performance
    
    def adaptive_train_lightgbm(self, 
                               X: pd.DataFrame, 
                               y: pd.Series, 
                               stock_id: str,
                               base_params: Dict = None) -> Tuple[Any, Dict]:
        """
        自适应训练LightGBM模型
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        # 基础参数
        if base_params is None:
            base_params = {
                'max_depth': 5,
                'learning_rate': 0.3,  # 较大学习率
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'min_data_in_leaf': 50,
                'force_row_wise': True,
                'random_state': 42,
                'verbose': -1,
                'n_jobs': 1
            }
        
        # 初始树数量
        current_trees = 50
        performance_history = []
        
        # 时序交叉验证
        tscv = TimeSeriesSplit(n_splits=3)
        
        best_model = None
        best_performance = {'ic': -np.inf, 'mse': np.inf}
        
        while current_trees <= self.max_trees_lgb:
            # 训练当前树数量的模型
            params = base_params.copy()
            params['n_estimators'] = current_trees
            params['early_stopping_rounds'] = min(15, current_trees // 3)
            
            model = lgb.LGBMRegressor(**params)
            
            # 交叉验证评估
            cv_scores = {'ic': [], 'mse': []}
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 训练模型
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
                )
                
                # 预测和评估
                y_pred = model.predict(X_val)
                
                # 计算IC
                ic = np.corrcoef(y_val, y_pred)[0, 1] if len(y_val) > 1 else 0.0
                ic = ic if not np.isnan(ic) else 0.0
                
                # 计算MSE
                mse = mean_squared_error(y_val, y_pred)
                
                cv_scores['ic'].append(ic)
                cv_scores['mse'].append(mse)
            
            # 计算平均性能
            avg_ic = np.mean(cv_scores['ic'])
            avg_mse = np.mean(cv_scores['mse'])
            
            current_performance = {'ic': avg_ic, 'mse': avg_mse}
            performance_history.append(current_performance)
            
            # 更新最佳模型
            if avg_ic > best_performance['ic']:
                best_performance = current_performance
                # 重新训练最佳模型
                best_model = lgb.LGBMRegressor(**params)
                best_model.fit(X, y)
            
            logger.info(f"{stock_id} LightGBM {current_trees}棵树: IC={avg_ic:.4f}, MSE={avg_mse:.4f}")
            
            # 判断是否继续加树
            if not self.should_add_trees(stock_id, current_performance, current_trees, 'lightgbm'):
                logger.info(f"{stock_id} LightGBM停止加树于{current_trees}棵")
                break
            
            current_trees += self.tree_increment
        
        return best_model, best_performance
    
    def adaptive_train_random_forest(self, 
                                   X: pd.DataFrame, 
                                   y: pd.Series, 
                                   stock_id: str,
                                   base_params: Dict = None) -> Tuple[Any, Dict]:
        """
        自适应训练RandomForest模型
        使用 oob_score=True + warm_start=True 逐批加树
        """
        # 基础参数
        if base_params is None:
            base_params = {
                'max_depth': 10,
                'max_features': 0.8,
                'min_samples_leaf': 10,
                'max_samples': 0.8,
                'bootstrap': True,
                'oob_score': True,   # 启用OOB评分
                'warm_start': True,  # 启用增量训练
                'n_jobs': 1,
                'random_state': 42
            }
        
        # 初始树数量
        current_trees = 50
        
        # 创建模型
        params = base_params.copy()
        params['n_estimators'] = current_trees
        model = RandomForestRegressor(**params)
        
        # 训练初始模型
        model.fit(X, y)
        
        best_oob_score = model.oob_score_
        best_trees = current_trees
        
        logger.info(f"{stock_id} RandomForest {current_trees}棵树: OOB分数={best_oob_score:.4f}")
        
        # 逐步增加树的数量
        while current_trees < self.max_trees_rf:
            current_trees += self.tree_increment
            
            # 增加树的数量
            model.n_estimators = current_trees
            model.fit(X, y)  # warm_start=True 会增量训练
            
            current_oob_score = model.oob_score_
            
            logger.info(f"{stock_id} RandomForest {current_trees}棵树: OOB分数={current_oob_score:.4f}")
            
            # 计算性能指标用于斜率判断
            # 使用OOB预测计算IC和MSE
            if hasattr(model, 'oob_prediction_'):
                oob_pred = model.oob_prediction_
                
                # 计算IC
                ic = np.corrcoef(y, oob_pred)[0, 1] if len(y) > 1 else 0.0
                ic = ic if not np.isnan(ic) else 0.0
                
                # 计算MSE
                mse = mean_squared_error(y, oob_pred)
                
                current_performance = {'ic': ic, 'mse': mse}
                
                # 判断是否继续加树
                if not self.should_add_trees(stock_id, current_performance, current_trees, 'rf'):
                    logger.info(f"{stock_id} RandomForest停止加树于{current_trees}棵")
                    break
            
            # 如果OOB分数没有提升，也考虑停止
            if current_oob_score <= best_oob_score:
                consecutive_no_improvement = getattr(model, '_consecutive_no_improvement', 0) + 1
                model._consecutive_no_improvement = consecutive_no_improvement
                
                if consecutive_no_improvement >= 3:  # 连续3次无改善则停止
                    logger.info(f"{stock_id} RandomForest因OOB无改善停止于{current_trees}棵")
                    break
            else:
                best_oob_score = current_oob_score
                best_trees = current_trees
                model._consecutive_no_improvement = 0
        
        # 创建最终模型
        final_params = base_params.copy()
        final_params['n_estimators'] = best_trees
        final_params['warm_start'] = False  # 最终训练时关闭warm_start
        final_model = RandomForestRegressor(**final_params)
        final_model.fit(X, y)
        
        performance = {'ic': 0.0, 'mse': 0.0, 'oob_score': best_oob_score}
        
        return final_model, performance
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化过程摘要"""
        summary = {
            'total_stocks_processed': len(set(key.split('_')[0] for key in self.slope_history.keys())),
            'slope_history': self.slope_history,
            'settings': {
                'slope_threshold_ic': self.slope_threshold_ic,
                'slope_threshold_mse': self.slope_threshold_mse,
                'tree_increment': self.tree_increment,
                'top_percentile': self.top_percentile,
                'max_trees': {
                    'xgboost': self.max_trees_xgb,
                    'lightgbm': self.max_trees_lgb,
                    'rf': self.max_trees_rf
                }
            }
        }
        
        return summary


def demo_adaptive_optimization():
    """演示自适应加树优化"""
    print("=== 自适应加树优化演示 ===")
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randn(n_samples))
    
    # 创建优化器
    optimizer = AdaptiveTreeOptimizer(
        slope_threshold_ic=0.001,
        slope_threshold_mse=0.005,
        tree_increment=20,
        max_trees_xgb=120,
        max_trees_lgb=120,
        max_trees_rf=150
    )
    
    # 测试XGBoost优化
    if XGBOOST_AVAILABLE:
        print("\n--- XGBoost自适应优化 ---")
        xgb_model, xgb_perf = optimizer.adaptive_train_xgboost(X, y, "TEST_STOCK")
        print(f"XGBoost最终性能: {xgb_perf}")
    
    # 测试LightGBM优化
    if LIGHTGBM_AVAILABLE:
        print("\n--- LightGBM自适应优化 ---")
        lgb_model, lgb_perf = optimizer.adaptive_train_lightgbm(X, y, "TEST_STOCK")
        print(f"LightGBM最终性能: {lgb_perf}")
    
    # 测试RandomForest优化
    print("\n--- RandomForest自适应优化 ---")
    rf_model, rf_perf = optimizer.adaptive_train_random_forest(X, y, "TEST_STOCK")
    print(f"RandomForest最终性能: {rf_perf}")
    
    # 获取优化摘要
    summary = optimizer.get_optimization_summary()
    print(f"\n优化摘要: 处理了{summary['total_stocks_processed']}只股票")
    
    return optimizer


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 运行演示
    optimizer = demo_adaptive_optimization()
    print("\n自适应加树优化演示完成!")