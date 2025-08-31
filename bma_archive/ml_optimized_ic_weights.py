"""
机器学习优化IC权重系统
=======================
使用LightGBM和神经网络动态优化因子权重
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class MLOptimizationConfig:
    """机器学习优化配置"""
    
    # 模型选择
    USE_ENSEMBLE: bool = True  # 使用集成学习
    
    # 集成权重
    MODEL_WEIGHTS = {
        'lightgbm': 0.4,    # LightGBM权重
        'lasso': 0.3,       # LASSO权重  
        'ridge': 0.2,       # Ridge权重
        'rf': 0.1,          # 随机森林权重
    }
    
    # LightGBM参数
    LGB_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 100,
        'early_stopping_rounds': 10,
        'lambda_l1': 0.1,  # L1正则化
        'lambda_l2': 0.1,  # L2正则化
    }
    
    # LASSO参数
    LASSO_ALPHA: float = 0.001
    
    # Ridge参数
    RIDGE_ALPHA: float = 1.0
    
    # 训练配置
    VALIDATION_SPLIT: float = 0.2
    RETRAIN_FREQUENCY: int = 20  # 每20天重训练
    MIN_TRAINING_SAMPLES: int = 500
    
    # IC优化目标
    OPTIMIZE_FOR: str = 'sharpe'  # 'ic', 'sharpe', 'calmar'


class MLOptimizedICWeights:
    """
    使用机器学习优化IC权重
    核心思想：学习最优权重组合以最大化预测性能
    """
    
    def __init__(self, config: Optional[MLOptimizationConfig] = None):
        self.config = config or MLOptimizationConfig()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.last_train_date = None
        self.weight_history = []
        
    def optimize_weights(self, 
                        factors: pd.DataFrame,
                        returns: pd.Series,
                        current_ic_weights: Dict[str, float]) -> Dict[str, float]:
        """
        使用ML优化因子权重
        
        Args:
            factors: 因子数据
            returns: 未来收益
            current_ic_weights: 当前IC权重
            
        Returns:
            优化后的权重
        """
        logger.info("开始ML权重优化")
        
        # 检查是否需要重训练
        if self._should_retrain():
            self._train_models(factors, returns)
        
        # 预测最优权重
        optimized_weights = self._predict_weights(factors, current_ic_weights)
        
        # 应用约束
        optimized_weights = self._apply_constraints(optimized_weights)
        
        # 记录权重历史
        self.weight_history.append({
            'timestamp': datetime.now(),
            'weights': optimized_weights,
            'ic_weights': current_ic_weights
        })
        
        return optimized_weights
    
    def _train_models(self, factors: pd.DataFrame, returns: pd.Series):
        """训练优化模型"""
        logger.info("训练ML权重优化模型")
        
        # 准备训练数据
        X, y, feature_names = self._prepare_training_data(factors, returns)
        
        if len(X) < self.config.MIN_TRAINING_SAMPLES:
            logger.warning(f"训练样本不足: {len(X)} < {self.config.MIN_TRAINING_SAMPLES}")
            return
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler
        
        # 划分训练/验证集
        split_idx = int(len(X) * (1 - self.config.VALIDATION_SPLIT))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 训练各个模型
        if self.config.USE_ENSEMBLE:
            # 1. LightGBM
            self._train_lightgbm(X_train, y_train, X_val, y_val, feature_names)
            
            # 2. LASSO
            self._train_lasso(X_train, y_train)
            
            # 3. Ridge
            self._train_ridge(X_train, y_train)
            
            # 4. Random Forest
            self._train_random_forest(X_train, y_train)
        else:
            # 只训练LightGBM
            self._train_lightgbm(X_train, y_train, X_val, y_val, feature_names)
        
        self.last_train_date = datetime.now()
        logger.info("ML模型训练完成")
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val, feature_names):
        """训练LightGBM模型"""
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            self.config.LGB_PARAMS,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        self.models['lightgbm'] = model
        
        # 记录特征重要性
        importance = model.feature_importance(importance_type='gain')
        self.feature_importance['lightgbm'] = dict(zip(feature_names, importance))
        
        # 打印Top特征
        top_features = sorted(self.feature_importance['lightgbm'].items(), 
                            key=lambda x: x[1], reverse=True)[:10]
        logger.info("LightGBM Top 10特征:")
        for feat, imp in top_features:
            logger.info(f"  {feat}: {imp:.2f}")
    
    def _train_lasso(self, X_train, y_train):
        """训练LASSO模型"""
        model = Lasso(alpha=self.config.LASSO_ALPHA, max_iter=1000)
        model.fit(X_train, y_train)
        self.models['lasso'] = model
        
        # 记录非零系数
        non_zero = np.sum(model.coef_ != 0)
        logger.info(f"LASSO选择了 {non_zero} 个特征")
    
    def _train_ridge(self, X_train, y_train):
        """训练Ridge模型"""
        model = Ridge(alpha=self.config.RIDGE_ALPHA)
        model.fit(X_train, y_train)
        self.models['ridge'] = model
    
    def _train_random_forest(self, X_train, y_train):
        """训练随机森林模型"""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['rf'] = model
    
    def _predict_weights(self, factors: pd.DataFrame, current_ic_weights: Dict[str, float]) -> Dict[str, float]:
        """预测优化权重"""
        if not self.models:
            logger.warning("模型未训练，返回原始IC权重")
            return current_ic_weights
        
        # 准备预测数据
        X = self._prepare_prediction_features(factors, current_ic_weights)
        
        if 'main' in self.scalers:
            X_scaled = self.scalers['main'].transform(X)
        else:
            X_scaled = X
        
        # 集成预测
        predictions = {}
        
        if self.config.USE_ENSEMBLE:
            for model_name, weight in self.config.MODEL_WEIGHTS.items():
                if model_name in self.models:
                    if model_name == 'lightgbm':
                        pred = self.models[model_name].predict(X_scaled, num_iteration=self.models[model_name].best_iteration)
                    else:
                        pred = self.models[model_name].predict(X_scaled)
                    
                    predictions[model_name] = pred * weight
            
            # 加权平均
            final_prediction = np.sum(list(predictions.values()), axis=0)
        else:
            # 单模型预测
            if 'lightgbm' in self.models:
                final_prediction = self.models['lightgbm'].predict(X_scaled)
            else:
                return current_ic_weights
        
        # 转换为权重字典
        optimized_weights = self._convert_to_weight_dict(final_prediction, factors.columns)
        
        return optimized_weights
    
    def _prepare_training_data(self, factors: pd.DataFrame, returns: pd.Series) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """准备训练数据"""
        # 特征工程
        features = []
        feature_names = []
        
        # 1. 原始因子值
        for col in factors.columns:
            features.append(factors[col].values)
            feature_names.append(f'factor_{col}')
        
        # 2. 因子间相关性
        corr_matrix = factors.corr()
        for i, col1 in enumerate(factors.columns):
            for j, col2 in enumerate(factors.columns):
                if i < j:
                    features.append(np.full(len(factors), corr_matrix.loc[col1, col2]))
                    feature_names.append(f'corr_{col1}_{col2}')
        
        # 3. 因子动量（变化率）
        for col in factors.columns:
            momentum = factors[col].pct_change().fillna(0).values
            features.append(momentum)
            feature_names.append(f'momentum_{col}')
        
        # 4. 因子波动率
        for col in factors.columns:
            volatility = factors[col].rolling(20).std().fillna(0).values
            features.append(volatility)
            feature_names.append(f'vol_{col}')
        
        # 组合特征
        X = np.column_stack(features)
        y = returns.values
        
        # 移除NaN
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y, feature_names
    
    def _prepare_prediction_features(self, factors: pd.DataFrame, current_ic_weights: Dict[str, float]) -> np.ndarray:
        """准备预测特征"""
        # 使用最新数据点
        latest_factors = factors.iloc[-1] if len(factors) > 0 else factors
        
        features = []
        
        # 添加所有训练时使用的特征
        for col in factors.columns:
            features.append(latest_factors[col] if col in latest_factors else 0)
        
        # 添加相关性特征
        corr_matrix = factors.corr()
        for i, col1 in enumerate(factors.columns):
            for j, col2 in enumerate(factors.columns):
                if i < j:
                    features.append(corr_matrix.loc[col1, col2] if col1 in corr_matrix.index and col2 in corr_matrix.columns else 0)
        
        # 添加动量特征
        for col in factors.columns:
            if len(factors) > 1:
                momentum = (factors[col].iloc[-1] - factors[col].iloc[-2]) / (factors[col].iloc[-2] + 1e-8)
            else:
                momentum = 0
            features.append(momentum)
        
        # 添加波动率特征
        for col in factors.columns:
            if len(factors) > 20:
                volatility = factors[col].tail(20).std()
            else:
                volatility = factors[col].std() if len(factors) > 1 else 0
            features.append(volatility)
        
        return np.array(features).reshape(1, -1)
    
    def _convert_to_weight_dict(self, predictions: np.ndarray, factor_names: List[str]) -> Dict[str, float]:
        """转换预测值为权重字典"""
        # 确保权重为正
        weights = np.abs(predictions)
        
        # 归一化
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(factor_names)) / len(factor_names)
        
        # 创建权重字典
        weight_dict = {}
        for i, name in enumerate(factor_names):
            if i < len(weights):
                weight_dict[name] = float(weights[i] if len(weights.shape) == 1 else weights[0, i])
            else:
                weight_dict[name] = 0.0
        
        return weight_dict
    
    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """应用权重约束"""
        # 1. 确保权重在合理范围
        MIN_WEIGHT = 0.01
        MAX_WEIGHT = 0.20
        
        constrained = {}
        for factor, weight in weights.items():
            constrained[factor] = np.clip(weight, MIN_WEIGHT, MAX_WEIGHT)
        
        # 2. 重新归一化
        total = sum(constrained.values())
        if total > 0:
            for factor in constrained:
                constrained[factor] /= total
        
        return constrained
    
    def _should_retrain(self) -> bool:
        """判断是否需要重训练"""
        if self.last_train_date is None:
            return True
        
        days_since_train = (datetime.now() - self.last_train_date).days
        return days_since_train >= self.config.RETRAIN_FREQUENCY
    
    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        if not self.feature_importance:
            return pd.DataFrame()
        
        # 汇总所有模型的特征重要性
        importance_data = []
        
        for model_name, importance_dict in self.feature_importance.items():
            for feature, importance in importance_dict.items():
                importance_data.append({
                    'model': model_name,
                    'feature': feature,
                    'importance': importance
                })
        
        df = pd.DataFrame(importance_data)
        
        # 计算平均重要性
        avg_importance = df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        return avg_importance.to_frame()
    
    def save_models(self, path: str):
        """保存模型"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'config': self.config,
            'last_train_date': self.last_train_date
        }
        
        joblib.dump(model_data, path)
        logger.info(f"模型已保存到 {path}")
    
    def load_models(self, path: str):
        """加载模型"""
        model_data = joblib.load(path)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_importance = model_data['feature_importance']
        self.config = model_data['config']
        self.last_train_date = model_data['last_train_date']
        
        logger.info(f"模型已从 {path} 加载")