#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML超参数优化模块
为BMA Ultra Enhanced模型提供自动超参数调优功能
支持贝叶斯优化、网格搜索和随机搜索策略
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import json
import time
from pathlib import Path
from sklearn.model_selection import cross_val_score, ParameterGrid
from sklearn.metrics import make_scorer

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """超参数优化配置"""
    strategy: str = "bayesian"  # "bayesian", "grid", "random"
    n_calls: int = 50  # 贝叶斯优化调用次数
    n_random_starts: int = 10  # 随机起始点数量
    cv_folds: int = 5  # 交叉验证折数
    scoring_metric: str = "neg_mean_squared_error"
    n_jobs: int = -1  # 并行作业数
    random_state: int = 42
    cache_dir: str = "cache/hyperopt"
    save_results: bool = True
    early_stopping_rounds: int = 10  # 早停轮数
    target_score_improvement: float = 0.001  # 目标分数改进阈值

# 兼容性别名
@dataclass  
class HyperparameterConfig(OptimizationConfig):
    """超参数优化配置"""
    strategy: str = "bayesian"  # "bayesian", "grid", "random"
    n_calls: int = 50  # 贝叶斯优化调用次数
    n_random_starts: int = 10  # 随机起始点数量
    cv_folds: int = 5  # 交叉验证折数
    scoring_metric: str = "neg_mean_squared_error"
    n_jobs: int = -1  # 并行作业数
    random_state: int = 42
    cache_dir: str = "cache/hyperopt"
    save_results: bool = True
    early_stopping_rounds: int = 10  # 早停轮数
    target_score_improvement: float = 0.001  # 目标分数改进阈值

@dataclass 
class HyperparameterSpace:
    """超参数空间定义"""
    param_name: str
    param_type: str  # "real", "integer", "categorical"
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    categories: Optional[List[Any]] = None
    default: Optional[Any] = None

class MLHyperparameterOptimizer:
    """ML超参数优化器"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        
        # 创建缓存目录
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 优化历史记录
        self.optimization_history = []
        self.best_params = None
        self.best_score = None
        
        logger.info(f"ML超参数优化器初始化 - 策略: {self.config.strategy}")
        if not BAYESIAN_OPT_AVAILABLE and self.config.strategy == "bayesian":
            logger.warning("scikit-optimize未安装，回退到网格搜索")
            self.config.strategy = "grid"
    
    def define_search_space(self, model_type: str) -> List[HyperparameterSpace]:
        """定义不同模型类型的搜索空间"""
        
        if model_type.lower() == "lightgbm":
            return [
                HyperparameterSpace("n_estimators", "integer", 50, 500, default=100),
                HyperparameterSpace("learning_rate", "real", 0.01, 0.3, default=0.1),
                HyperparameterSpace("max_depth", "integer", 3, 15, default=6),
                HyperparameterSpace("num_leaves", "integer", 10, 300, default=31),
                HyperparameterSpace("min_child_samples", "integer", 5, 100, default=20),
                HyperparameterSpace("subsample", "real", 0.6, 1.0, default=1.0),
                HyperparameterSpace("colsample_bytree", "real", 0.6, 1.0, default=1.0),
                HyperparameterSpace("reg_alpha", "real", 0.0, 10.0, default=0.0),
                HyperparameterSpace("reg_lambda", "real", 0.0, 10.0, default=0.0)
            ]
        
        elif model_type.lower() == "xgboost":
            return [
                HyperparameterSpace("n_estimators", "integer", 50, 500, default=100),
                HyperparameterSpace("learning_rate", "real", 0.01, 0.3, default=0.1),
                HyperparameterSpace("max_depth", "integer", 3, 12, default=6),
                HyperparameterSpace("min_child_weight", "integer", 1, 10, default=1),
                HyperparameterSpace("subsample", "real", 0.6, 1.0, default=1.0),
                HyperparameterSpace("colsample_bytree", "real", 0.6, 1.0, default=1.0),
                HyperparameterSpace("gamma", "real", 0.0, 5.0, default=0.0),
                HyperparameterSpace("reg_alpha", "real", 0.0, 10.0, default=0.0),
                HyperparameterSpace("reg_lambda", "real", 0.0, 10.0, default=1.0)
            ]
        
        elif model_type.lower() == "randomforest":
            return [
                HyperparameterSpace("n_estimators", "integer", 50, 300, default=100),
                HyperparameterSpace("max_depth", "integer", 5, 30, default=10),
                HyperparameterSpace("min_samples_split", "integer", 2, 20, default=2),
                HyperparameterSpace("min_samples_leaf", "integer", 1, 10, default=1),
                HyperparameterSpace("max_features", "categorical", 
                                  categories=["sqrt", "log2", None], default="sqrt"),
                HyperparameterSpace("bootstrap", "categorical", 
                                  categories=[True, False], default=True)
            ]
        
        elif model_type.lower() == "ridge":
            return [
                HyperparameterSpace("alpha", "real", 0.1, 100.0, default=1.0),
                HyperparameterSpace("fit_intercept", "categorical", 
                                  categories=[True, False], default=True),
                HyperparameterSpace("solver", "categorical",
                                  categories=["auto", "svd", "cholesky", "lsqr"], default="auto")
            ]
        
        else:
            logger.warning(f"未知模型类型: {model_type}，使用默认搜索空间")
            return [
                HyperparameterSpace("learning_rate", "real", 0.01, 0.3, default=0.1),
                HyperparameterSpace("n_estimators", "integer", 50, 300, default=100)
            ]
    
    def optimize_hyperparameters(self, 
                                model, 
                                X: pd.DataFrame, 
                                y: pd.Series,
                                search_space: Optional[List[HyperparameterSpace]] = None,
                                model_type: str = "lightgbm") -> Dict[str, Any]:
        """
        执行超参数优化
        
        Args:
            model: 待优化的模型对象
            X: 特征数据
            y: 目标变量
            search_space: 自定义搜索空间
            model_type: 模型类型
            
        Returns:
            包含最优参数和性能的字典
        """
        logger.info(f"开始{model_type}模型超参数优化")
        
        # 定义搜索空间
        if search_space is None:
            search_space = self.define_search_space(model_type)
        
        # 根据策略选择优化方法
        if self.config.strategy == "bayesian" and BAYESIAN_OPT_AVAILABLE:
            return self._bayesian_optimization(model, X, y, search_space)
        elif self.config.strategy == "grid":
            return self._grid_search_optimization(model, X, y, search_space)
        elif self.config.strategy == "random":
            return self._random_search_optimization(model, X, y, search_space)
        else:
            logger.error(f"不支持的优化策略: {self.config.strategy}")
            return {"best_params": {}, "best_score": 0.0, "optimization_history": []}
    
    def _bayesian_optimization(self, model, X, y, search_space) -> Dict[str, Any]:
        """贝叶斯优化实现"""
        logger.info("执行贝叶斯超参数优化")
        
        # 构建skopt搜索空间
        skopt_space = []
        param_names = []
        
        for param_space in search_space:
            param_names.append(param_space.param_name)
            
            if param_space.param_type == "real":
                skopt_space.append(Real(param_space.low, param_space.high, name=param_space.param_name))
            elif param_space.param_type == "integer":
                skopt_space.append(Integer(param_space.low, param_space.high, name=param_space.param_name))
            elif param_space.param_type == "categorical":
                skopt_space.append(Categorical(param_space.categories, name=param_space.param_name))
        
        # 定义目标函数
        def objective_function(params):
            param_dict = dict(zip(param_names, params))
            
            # 设置模型参数
            model.set_params(**param_dict)
            
            # 交叉验证评估
            try:
                scores = cross_val_score(
                    model, X, y, 
                    cv=self.config.cv_folds, 
                    scoring=self.config.scoring_metric,
                    n_jobs=self.config.n_jobs
                )
                mean_score = np.mean(scores)
                
                # 记录历史
                self.optimization_history.append({
                    "params": param_dict.copy(),
                    "score": mean_score,
                    "scores": scores.tolist(),
                    "timestamp": time.time()
                })
                
                # 贝叶斯优化最小化，所以返回负分数
                return -mean_score
                
            except Exception as e:
                logger.error(f"参数评估失败: {param_dict}, 错误: {e}")
                return 0.0  # 返回较差的分数
        
        # 执行贝叶斯优化
        try:
            result = gp_minimize(
                func=objective_function,
                dimensions=skopt_space,
                n_calls=self.config.n_calls,
                n_random_starts=self.config.n_random_starts,
                random_state=self.config.random_state
            )
            
            # 提取最优参数
            best_params = dict(zip(param_names, result.x))
            best_score = -result.fun
            
            logger.info(f"贝叶斯优化完成 - 最优分数: {best_score:.4f}")
            logger.info(f"最优参数: {best_params}")
            
            self.best_params = best_params
            self.best_score = best_score
            
            return {
                "best_params": best_params,
                "best_score": best_score,
                "optimization_history": self.optimization_history,
                "n_evaluations": len(self.optimization_history)
            }
            
        except Exception as e:
            logger.error(f"贝叶斯优化失败: {e}")
            return self._fallback_to_default_params(search_space)
    
    def _grid_search_optimization(self, model, X, y, search_space) -> Dict[str, Any]:
        """网格搜索优化实现"""
        logger.info("执行网格搜索超参数优化")
        
        # 构建参数网格 - 限制组合数量避免过度搜索
        param_grid = {}
        for param_space in search_space:
            if param_space.param_type == "real":
                # 实数参数：创建10个均匀分布的点
                param_grid[param_space.param_name] = np.linspace(
                    param_space.low, param_space.high, 10
                ).tolist()
            elif param_space.param_type == "integer":
                # 整数参数：创建有限的取值点
                n_points = min(10, param_space.high - param_space.low + 1)
                param_grid[param_space.param_name] = np.linspace(
                    param_space.low, param_space.high, n_points, dtype=int
                ).tolist()
            elif param_space.param_type == "categorical":
                param_grid[param_space.param_name] = param_space.categories
        
        # 限制搜索组合数量
        grid = ParameterGrid(param_grid)
        total_combinations = len(grid)
        
        if total_combinations > 500:  # 限制搜索空间大小
            logger.warning(f"网格搜索空间过大 ({total_combinations}), 将随机采样500个组合")
            import random
            random.seed(self.config.random_state)
            grid = random.sample(list(grid), 500)
        
        best_score = float('-inf')
        best_params = {}
        
        logger.info(f"网格搜索将评估 {len(grid)} 个参数组合")
        
        for i, params in enumerate(grid):
            try:
                # 设置模型参数
                model.set_params(**params)
                
                # 交叉验证评估
                scores = cross_val_score(
                    model, X, y,
                    cv=self.config.cv_folds,
                    scoring=self.config.scoring_metric,
                    n_jobs=self.config.n_jobs
                )
                mean_score = np.mean(scores)
                
                # 记录历史
                self.optimization_history.append({
                    "params": params.copy(),
                    "score": mean_score,
                    "scores": scores.tolist(),
                    "timestamp": time.time()
                })
                
                # 更新最优结果
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()
                
                if (i + 1) % 10 == 0:
                    logger.info(f"网格搜索进度: {i+1}/{len(grid)}, 当前最优分数: {best_score:.4f}")
                    
            except Exception as e:
                logger.error(f"参数评估失败: {params}, 错误: {e}")
                continue
        
        logger.info(f"网格搜索完成 - 最优分数: {best_score:.4f}")
        logger.info(f"最优参数: {best_params}")
        
        self.best_params = best_params
        self.best_score = best_score
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "optimization_history": self.optimization_history,
            "n_evaluations": len(self.optimization_history)
        }
    
    def _random_search_optimization(self, model, X, y, search_space) -> Dict[str, Any]:
        """随机搜索优化实现"""
        logger.info("执行随机搜索超参数优化")
        
        np.random.seed(self.config.random_state)
        best_score = float('-inf')
        best_params = {}
        
        for i in range(self.config.n_calls):
            # 随机生成参数
            random_params = {}
            for param_space in search_space:
                if param_space.param_type == "real":
                    random_params[param_space.param_name] = np.random.uniform(
                        param_space.low, param_space.high
                    )
                elif param_space.param_type == "integer":
                    random_params[param_space.param_name] = np.random.randint(
                        param_space.low, param_space.high + 1
                    )
                elif param_space.param_type == "categorical":
                    random_params[param_space.param_name] = np.random.choice(
                        param_space.categories
                    )
            
            try:
                # 设置模型参数
                model.set_params(**random_params)
                
                # 交叉验证评估
                scores = cross_val_score(
                    model, X, y,
                    cv=self.config.cv_folds,
                    scoring=self.config.scoring_metric,
                    n_jobs=self.config.n_jobs
                )
                mean_score = np.mean(scores)
                
                # 记录历史
                self.optimization_history.append({
                    "params": random_params.copy(),
                    "score": mean_score,
                    "scores": scores.tolist(),
                    "timestamp": time.time()
                })
                
                # 更新最优结果
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = random_params.copy()
                
                if (i + 1) % 10 == 0:
                    logger.info(f"随机搜索进度: {i+1}/{self.config.n_calls}, 当前最优分数: {best_score:.4f}")
                    
            except Exception as e:
                logger.error(f"参数评估失败: {random_params}, 错误: {e}")
                continue
        
        logger.info(f"随机搜索完成 - 最优分数: {best_score:.4f}")
        logger.info(f"最优参数: {best_params}")
        
        self.best_params = best_params
        self.best_score = best_score
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "optimization_history": self.optimization_history,
            "n_evaluations": len(self.optimization_history)
        }
    
    def _fallback_to_default_params(self, search_space) -> Dict[str, Any]:
        """回退到默认参数"""
        logger.warning("优化失败，使用默认参数")
        
        default_params = {}
        for param_space in search_space:
            if param_space.default is not None:
                default_params[param_space.param_name] = param_space.default
        
        return {
            "best_params": default_params,
            "best_score": 0.0,
            "optimization_history": [],
            "n_evaluations": 0
        }
    
    def save_optimization_results(self, results: Dict[str, Any], filename: str = None):
        """保存优化结果"""
        if not self.config.save_results:
            return
            
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"hyperopt_results_{timestamp}.json"
        
        filepath = self.cache_dir / filename
        
        try:
            # 确保结果可序列化
            serializable_results = {
                "best_params": results["best_params"],
                "best_score": float(results["best_score"]),
                "n_evaluations": results["n_evaluations"],
                "config": {
                    "strategy": self.config.strategy,
                    "n_calls": self.config.n_calls,
                    "cv_folds": self.config.cv_folds,
                    "scoring_metric": self.config.scoring_metric
                },
                "optimization_history": [
                    {
                        "params": hist["params"],
                        "score": float(hist["score"]),
                        "timestamp": hist["timestamp"]
                    }
                    for hist in results["optimization_history"]
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"优化结果已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存优化结果失败: {e}")
    
    def load_optimization_results(self, filename: str) -> Optional[Dict[str, Any]]:
        """加载已保存的优化结果"""
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            logger.warning(f"优化结果文件不存在: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            logger.info(f"优化结果已加载: {filepath}")
            return results
            
        except Exception as e:
            logger.error(f"加载优化结果失败: {e}")
            return None

# 便捷函数
def optimize_model_hyperparameters(model, X, y, model_type="lightgbm", 
                                 strategy="bayesian", n_calls=50) -> Dict[str, Any]:
    """
    便捷函数：优化模型超参数
    
    Args:
        model: 待优化的模型
        X: 特征数据  
        y: 目标变量
        model_type: 模型类型
        strategy: 优化策略
        n_calls: 优化调用次数
        
    Returns:
        优化结果字典
    """
    config = OptimizationConfig(
        strategy=strategy,
        n_calls=n_calls
    )
    
    optimizer = MLHyperparameterOptimizer(config)
    results = optimizer.optimize_hyperparameters(model, X, y, model_type=model_type)
    optimizer.save_optimization_results(results)
    
    return results

if __name__ == "__main__":
    # 测试超参数优化器
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    
    # 生成测试数据
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y, name='target')
    
    # 测试随机森林优化
    model = RandomForestRegressor(random_state=42)
    results = optimize_model_hyperparameters(
        model, X, y, 
        model_type="randomforest",
        strategy="grid", 
        n_calls=20
    )
    
    print(f"最优参数: {results['best_params']}")
    print(f"最优分数: {results['best_score']:.4f}")
    print(f"评估次数: {results['n_evaluations']}")