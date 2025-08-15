#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMA轻量化配置 - 针对3800只股票的高效训练
基于你的优化建议实现的完整解决方案
"""

import pandas as pd
import numpy as np
import logging
import time
import gc
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression

# 高级模型导入
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

class LightweightBMAConfig:
    """
    轻量化BMA配置类
    针对3800只股票优化的模型参数
    """
    
    def __init__(self, 
                 target_time_per_stock: float = 3.0,  # 目标：每只股票3秒
                 feature_dimensions: int = 15,         # 特征维度
                 sample_size: int = 252,              # 样本量（约1年）
                 memory_limit_gb: float = 8.0):       # 内存限制
        
        self.target_time_per_stock = target_time_per_stock
        self.feature_dimensions = feature_dimensions
        self.sample_size = sample_size
        self.memory_limit_gb = memory_limit_gb
        
        # 基于目标时间和样本量自动调整参数
        self._calculate_optimal_params()
        
    def _calculate_optimal_params(self):
        """基于约束条件计算最优参数"""
        
        # 基于样本量调整CV折数
        if self.sample_size < 100:
            self.cv_folds = 2
        elif self.sample_size < 200:
            self.cv_folds = 3
        else:
            self.cv_folds = 3  # 最多3折，保证速度
        
        # 基于特征维度调整采样率
        if self.feature_dimensions <= 10:
            self.feature_sample_rate = 0.9
        elif self.feature_dimensions <= 20:
            self.feature_sample_rate = 0.8
        else:
            self.feature_sample_rate = 0.7
        
        # 基于目标时间调整估计器数量
        if self.target_time_per_stock <= 2:
            self.estimator_scale = 0.6  # 超快模式
        elif self.target_time_per_stock <= 5:
            self.estimator_scale = 0.8  # 快速模式
        else:
            self.estimator_scale = 1.0  # 标准模式
    
    def get_random_forest_config(self) -> Dict:
        """获取轻量化RandomForest配置"""
        base_estimators = 80
        n_estimators = max(30, int(base_estimators * self.estimator_scale))
        
        return {
            'n_estimators': n_estimators,
            'max_depth': 8,  # 从无限制改为8
            'max_features': self.feature_sample_rate,  # 特征采样
            'min_samples_leaf': 15,  # 增加叶子节点最小样本
            'min_samples_split': 30,  # 增加分割最小样本
            'max_samples': 0.8,  # 样本采样
            'bootstrap': True,
            'n_jobs': 2,  # 限制并行度，避免内存争用
            'random_state': 42,
            'warm_start': False  # 确保不累积内存
        }
    
    def get_xgboost_config(self) -> Dict:
        """获取轻量化XGBoost配置"""
        if not XGBOOST_AVAILABLE:
            return {}
        
        base_estimators = 60
        n_estimators = max(30, int(base_estimators * self.estimator_scale))
        
        return {
            # 核心参数
            'n_estimators': n_estimators,
            'max_depth': 3,  # 从6减少到3
            'learning_rate': 0.2,  # 从0.1增加到0.2
            
            # 采样参数
            'subsample': 0.8,
            'colsample_bytree': self.feature_sample_rate,
            'colsample_bylevel': 0.9,
            
            # 正则化参数
            'reg_alpha': 0.1,  # L1正则
            'reg_lambda': 1.0,  # L2正则
            'min_child_weight': 5,  # 增加最小子权重
            
            # 性能优化
            'tree_method': 'hist',  # 使用histogram算法
            'max_bin': 64,  # 减少bin数量
            'grow_policy': 'lossguide',  # 使用loss导向增长
            
            # 早停和验证
            'early_stopping_rounds': 15,  # 早停轮数
            'eval_metric': 'rmse',
            
            # 其他
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': 2  # 限制并行度
        }
    
    def get_lightgbm_config(self) -> Dict:
        """获取轻量化LightGBM配置"""
        if not LIGHTGBM_AVAILABLE:
            return {}
        
        base_estimators = 60
        n_estimators = max(30, int(base_estimators * self.estimator_scale))
        
        return {
            # 核心参数
            'n_estimators': n_estimators,
            'learning_rate': 0.2,  # 从0.1增加到0.2
            'max_depth': 4,  # 从6减少到4
            'num_leaves': 15,  # 严格控制叶子数 (< 2^max_depth)
            
            # 采样参数
            'feature_fraction': self.feature_sample_rate,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'subsample_for_bin': 100000,  # 减少采样数
            
            # 正则化参数
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'min_data_in_leaf': 50,  # 增加叶子最小数据量
            'min_child_samples': 30,
            
            # 性能优化
            'max_bin': 63,  # 减少bin数量
            'bin_construct_sample_cnt': 50000,
            
            # 早停
            'early_stopping_rounds': 15,
            
            # 其他
            'random_state': 42,
            'verbose': -1,
            'n_jobs': 2,
            'force_row_wise': True  # 强制行优先，节省内存
        }
    
    def get_ridge_config(self) -> Dict:
        """获取轻量化Ridge配置"""
        return {
            'alpha': 2.0,  # 适度增加正则化
            'fit_intercept': True,
            'copy_X': False,  # 节省内存
            'max_iter': 1000,  # 减少最大迭代
            'tol': 1e-3,  # 放宽收敛条件
            'solver': 'auto',
            'random_state': 42
        }
    
    def get_elasticnet_config(self) -> Dict:
        """获取轻量化ElasticNet配置"""
        return {
            'alpha': 0.15,  # 适度增加正则化
            'l1_ratio': 0.2,  # L1和L2混合
            'fit_intercept': True,
            'precompute': False,  # 不预计算，节省内存
            'max_iter': 1500,  # 从5000减少到1500
            'tol': 1e-3,  # 从1e-4放宽到1e-3
            'warm_start': False,
            'positive': False,
            'random_state': 42,
            'copy_X': False,  # 节省内存
            'selection': 'cyclic'
        }
    
    def get_cv_config(self) -> Dict:
        """获取交叉验证配置"""
        return {
            'n_splits': self.cv_folds,
            'test_size': None,
            'gap': 0,
            'max_train_size': None
        }
    
    def get_feature_selection_config(self) -> Dict:
        """获取特征选择配置"""
        max_features = min(self.feature_dimensions, 12)  # 最多保留12个特征
        
        return {
            'k': max_features,
            'score_func': f_regression
        }


class LightweightBMATrainer:
    """
    轻量化BMA训练器
    针对大规模股票池优化
    """
    
    def __init__(self, config: LightweightBMAConfig = None):
        self.config = config or LightweightBMAConfig()
        self.training_stats = {
            'total_time': 0.0,
            'models_trained': 0,
            'avg_time_per_model': 0.0,
            'memory_usage_mb': 0.0
        }
        
    def create_lightweight_models(self) -> Dict:
        """创建轻量化模型集合"""
        models = {}
        
        # 1. Ridge (最快的基础模型)
        models['Ridge'] = Ridge(**self.config.get_ridge_config())
        
        # 2. ElasticNet (快速线性模型)
        models['ElasticNet'] = ElasticNet(**self.config.get_elasticnet_config())
        
        # 3. RandomForest (轻量化树模型)
        models['RandomForest'] = RandomForestRegressor(**self.config.get_random_forest_config())
        
        # 4. XGBoost (如果可用)
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBRegressor(**self.config.get_xgboost_config())
        
        # 5. LightGBM (如果可用)
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMRegressor(**self.config.get_lightgbm_config())
        
        logger.info(f"创建了 {len(models)} 个轻量化模型")
        return models
    
    def train_single_stock_model(self, X: pd.DataFrame, y: pd.Series, 
                                stock_id: str = None) -> Tuple[Dict, Dict, float]:
        """
        训练单只股票的轻量化模型
        
        Returns:
            (trained_models, model_weights, training_time)
        """
        start_time = time.time()
        
        if stock_id:
            logger.debug(f"训练股票 {stock_id} 的模型...")
        
        # 数据预处理
        X_processed, y_processed = self._preprocess_data(X, y)
        
        if len(X_processed) < 30:  # 最小样本要求
            logger.warning(f"样本不足: {len(X_processed)} < 30")
            return {}, {}, 0.0
        
        # 创建模型
        models = self.create_lightweight_models()
        trained_models = {}
        model_weights = {}
        
        # 交叉验证配置
        cv_config = self.config.get_cv_config()
        tscv = TimeSeriesSplit(**cv_config)
        
        # 训练每个模型
        for model_name, model in models.items():
            try:
                model_start_time = time.time()
                
                # 针对XGBoost和LightGBM的早停训练
                if model_name in ['XGBoost', 'LightGBM']:
                    trained_model, score = self._train_with_early_stopping(
                        model, X_processed, y_processed, tscv
                    )
                else:
                    trained_model, score = self._train_regular_model(
                        model, X_processed, y_processed, tscv
                    )
                
                if trained_model is not None:
                    trained_models[model_name] = trained_model
                    model_weights[model_name] = max(0.01, score)  # 避免负权重
                
                model_time = time.time() - model_start_time
                logger.debug(f"  {model_name}: {model_time:.2f}s, score={score:.4f}")
                
            except Exception as e:
                logger.warning(f"模型 {model_name} 训练失败: {e}")
                continue
        
        # 权重归一化
        total_weight = sum(model_weights.values())
        if total_weight > 0:
            model_weights = {k: v/total_weight for k, v in model_weights.items()}
        
        training_time = time.time() - start_time
        
        # 更新统计
        self.training_stats['total_time'] += training_time
        self.training_stats['models_trained'] += 1
        self.training_stats['avg_time_per_model'] = (
            self.training_stats['total_time'] / self.training_stats['models_trained']
        )
        
        return trained_models, model_weights, training_time
    
    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """数据预处理"""
        # 特征选择
        if len(X.columns) > self.config.feature_dimensions:
            feature_config = self.config.get_feature_selection_config()
            selector = SelectKBest(**feature_config)
            X_selected = selector.fit_transform(
                X.fillna(0).replace([np.inf, -np.inf], 0), y
            )
        else:
            X_selected = X.fillna(0).replace([np.inf, -np.inf], 0).values
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # 清理目标变量
        y_clean = y.fillna(0).replace([np.inf, -np.inf], 0).values
        
        return X_scaled, y_clean
    
    def _train_with_early_stopping(self, model, X: np.ndarray, y: np.ndarray, 
                                  tscv: TimeSeriesSplit) -> Tuple[object, float]:
        """使用早停训练XGBoost/LightGBM"""
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 克隆模型
            model_clone = type(model)(**model.get_params())
            
            # 设置早停
            if hasattr(model_clone, 'fit'):
                if 'XGB' in str(type(model_clone)):
                    model_clone.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                elif 'LGBM' in str(type(model_clone)):
                    model_clone.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
                    )
                else:
                    model_clone.fit(X_train, y_train)
                
                # 评估
                y_pred = model_clone.predict(X_val)
                score = r2_score(y_val, y_pred)
                scores.append(score)
        
        # 最终训练
        final_model = type(model)(**model.get_params())
        final_model.fit(X, y)
        
        avg_score = np.mean(scores) if scores else 0.0
        return final_model, avg_score
    
    def _train_regular_model(self, model, X: np.ndarray, y: np.ndarray, 
                           tscv: TimeSeriesSplit) -> Tuple[object, float]:
        """训练常规模型"""
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 克隆并训练模型
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train, y_train)
            
            # 评估
            y_pred = model_clone.predict(X_val)
            score = r2_score(y_val, y_pred)
            scores.append(score)
        
        # 最终训练
        final_model = type(model)(**model.get_params())
        final_model.fit(X, y)
        
        avg_score = np.mean(scores) if scores else 0.0
        return final_model, avg_score
    
    def estimate_total_time(self, num_stocks: int) -> Dict[str, float]:
        """估算总训练时间"""
        if self.training_stats['models_trained'] == 0:
            # 基于配置估算
            estimated_time_per_stock = self.config.target_time_per_stock
        else:
            # 基于实际统计
            estimated_time_per_stock = self.training_stats['avg_time_per_model']
        
        total_estimated_time = num_stocks * estimated_time_per_stock
        
        return {
            'time_per_stock_seconds': estimated_time_per_stock,
            'total_time_seconds': total_estimated_time,
            'total_time_hours': total_estimated_time / 3600,
            'estimated_completion': pd.Timestamp.now() + pd.Timedelta(seconds=total_estimated_time)
        }
    
    def get_memory_usage(self) -> float:
        """获取当前内存使用量(MB)"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def cleanup_memory(self):
        """清理内存"""
        gc.collect()
        self.training_stats['memory_usage_mb'] = self.get_memory_usage()


def create_production_config(target_time_per_stock: float = 3.0) -> LightweightBMAConfig:
    """
    创建生产环境配置
    
    Args:
        target_time_per_stock: 目标每只股票训练时间(秒)
    """
    return LightweightBMAConfig(
        target_time_per_stock=target_time_per_stock,
        feature_dimensions=12,  # 精简特征
        sample_size=252,        # 1年数据
        memory_limit_gb=8.0     # 8GB内存限制
    )


def benchmark_configuration():
    """基准测试配置性能"""
    print("=" * 60)
    print("BMA轻量化配置基准测试")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 252
    n_features = 15
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randn(n_samples))
    
    # 测试不同配置
    configs = [
        ("极速模式", LightweightBMAConfig(target_time_per_stock=1.0)),
        ("快速模式", LightweightBMAConfig(target_time_per_stock=3.0)),
        ("标准模式", LightweightBMAConfig(target_time_per_stock=5.0))
    ]
    
    results = []
    
    for config_name, config in configs:
        print(f"\n测试配置: {config_name}")
        
        trainer = LightweightBMATrainer(config)
        
        start_time = time.time()
        models, weights, training_time = trainer.train_single_stock_model(X, y, "TEST")
        
        results.append({
            'config': config_name,
            'training_time': training_time,
            'models_count': len(models),
            'memory_mb': trainer.get_memory_usage()
        })
        
        print(f"  训练时间: {training_time:.2f}秒")
        print(f"  模型数量: {len(models)}")
        print(f"  内存使用: {trainer.get_memory_usage():.1f}MB")
        
        # 估算3800只股票的时间
        estimation = trainer.estimate_total_time(3800)
        print(f"  3800只股票预计: {estimation['total_time_hours']:.1f}小时")
    
    return results


if __name__ == "__main__":
    # 运行基准测试
    benchmark_results = benchmark_configuration()
    
    print("\n" + "=" * 60)
    print("推荐配置")
    print("=" * 60)
    
    # 创建推荐配置
    config = create_production_config(target_time_per_stock=3.0)
    
    print("针对3800只股票的推荐配置:")
    print(f"- RandomForest: {config.get_random_forest_config()}")
    print(f"- XGBoost: {config.get_xgboost_config()}")
    print(f"- LightGBM: {config.get_lightgbm_config()}")
    print(f"- Ridge: {config.get_ridge_config()}")
    print(f"- ElasticNet: {config.get_elasticnet_config()}")
    
    # 时间估算
    trainer = LightweightBMATrainer(config)
    estimation = trainer.estimate_total_time(3800)
    print(f"\n预计总训练时间: {estimation['total_time_hours']:.1f}小时")
    print(f"预计完成时间: {estimation['estimated_completion']}")