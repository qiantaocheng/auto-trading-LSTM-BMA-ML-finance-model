#!/usr/bin/env python3
"""
BMA权重稳健化系统
实现基于标准化+Softmax和信息准则的稳健权重计算
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from scipy import stats
from scipy.special import softmax
import warnings

logger = logging.getLogger(__name__)


@dataclass
class RobustWeightingConfig:
    """稳健权重配置"""
    
    # 方案A: 标准化 + Softmax参数
    ic_weight: float = 0.6                    # IC权重系数 β₁
    sharpe_weight: float = 0.4                # Sharpe权重系数 β₂
    sample_weight: float = 0.1                # 样本量权重系数 β₃
    
    # Softmax温度参数
    temperature: float = 0.7                  # 温度化参数 τ ∈ [0.5,1.0]
    min_weight: float = 0.03                  # 最小权重下限
    
    # 滚动窗口参数
    ic_window: int = 252                      # IC计算滚动窗口 (1年)
    sharpe_window: int = 126                  # Sharpe计算滚动窗口 (半年)
    min_samples: int = 50                     # 最小样本量要求
    
    # 方案B: 信息准则参数
    use_information_criterion: bool = False   # 是否使用信息准则
    criterion_type: str = "aic"               # 信息准则类型 ("aic", "bic", "cv_log_loss")
    cv_folds: int = 5                         # 交叉验证折数
    
    # 数据质量控制
    outlier_threshold: float = 3.0            # 异常值阈值 (标准差倍数)
    enable_outlier_filtering: bool = True     # 是否启用异常值过滤
    correlation_threshold: float = 0.95       # 相关性阈值 (去除高相关模型)


class RobustBMAWeighting:
    """
    稳健BMA权重计算系统
    
    实现两种方案:
    方案A: 标准化 + Softmax (堆叠思想)
    方案B: 信息准则/贝叶斯证据
    """
    
    def __init__(self, config: RobustWeightingConfig = None):
        self.config = config or RobustWeightingConfig()
        self.logger = logging.getLogger("RobustBMAWeighting")
        
        # 缓存历史性能数据
        self.historical_performance: Dict[str, Dict] = {}
        self.rolling_statistics: Dict[str, Dict] = {}
        
    def calculate_robust_weights(self,
                                model_performance: Dict[str, Dict],
                                predictions: Dict[str, np.ndarray],
                                targets: Optional[np.ndarray] = None) -> Tuple[Dict[str, float], Dict]:
        """
        计算稳健BMA权重
        
        Args:
            model_performance: 模型性能指标字典
            predictions: 模型预测结果字典
            targets: 目标值 (用于信息准则计算)
            
        Returns:
            (稳健权重字典, 详细信息字典)
        """
        try:
            if self.config.use_information_criterion and targets is not None:
                # 方案B: 信息准则
                return self._calculate_information_criterion_weights(
                    model_performance, predictions, targets
                )
            else:
                # 方案A: 标准化 + Softmax
                return self._calculate_softmax_weights(model_performance)
                
        except Exception as e:
            self.logger.error(f"稳健权重计算失败: {e}")
            return self._fallback_equal_weights(model_performance), {'error': str(e)}
    
    def _calculate_softmax_weights(self, model_performance: Dict[str, Dict]) -> Tuple[Dict[str, float], Dict]:
        """
        方案A: 标准化 + Softmax权重计算
        
        实现公式:
        z_IC,m = zscore(IC_m), z_SR,m = zscore(SR_m)
        η_m = β₁*z_IC,m + β₂*z_SR,m + β₃*log(N_m)
        w_m = softmax(η_m/τ)
        """
        try:
            model_names = list(model_performance.keys())
            if len(model_names) < 2:
                return self._fallback_equal_weights(model_performance), {'reason': 'insufficient_models'}
            
            # 1. 提取性能指标
            ic_scores = []
            sharpe_ratios = []
            sample_sizes = []
            
            for model_name in model_names:
                perf = model_performance[model_name]
                
                # IC评分
                ic = perf.get('ic_score', perf.get('oof_ic', 0.0))
                ic_scores.append(ic)
                
                # Sharpe比率
                sharpe = perf.get('sharpe_ratio', perf.get('oof_sharpe', 0.0))
                sharpe_ratios.append(sharpe)
                
                # 样本量
                n_samples = perf.get('n_samples', perf.get('sample_count', 100))
                sample_sizes.append(max(n_samples, 1))  # 避免log(0)
            
            # 2. 异常值过滤
            if self.config.enable_outlier_filtering:
                ic_scores, sharpe_ratios, sample_sizes = self._filter_outliers(
                    ic_scores, sharpe_ratios, sample_sizes
                )
            
            # 3. 标准化转换
            z_ic = self._robust_zscore(ic_scores)
            z_sharpe = self._robust_zscore(sharpe_ratios)
            log_samples = np.log(sample_sizes)
            z_log_samples = self._robust_zscore(log_samples)
            
            # 4. 组合评分计算
            # η_m = β₁*z_IC,m + β₂*z_SR,m + β₃*log(N_m)
            eta_scores = (
                self.config.ic_weight * z_ic +
                self.config.sharpe_weight * z_sharpe +
                self.config.sample_weight * z_log_samples
            )
            
            # 5. 温度化Softmax权重
            softmax_weights = softmax(eta_scores / self.config.temperature)
            
            # 6. 应用最小权重约束
            robust_weights = self._apply_min_weight_constraint(softmax_weights)
            
            # 7. 构建结果字典
            weight_dict = dict(zip(model_names, robust_weights))
            
            details = {
                'method': 'robust_softmax',
                'raw_scores': {
                    'ic_scores': ic_scores,
                    'sharpe_ratios': sharpe_ratios,
                    'sample_sizes': sample_sizes
                },
                'standardized_scores': {
                    'z_ic': z_ic.tolist(),
                    'z_sharpe': z_sharpe.tolist(),
                    'z_log_samples': z_log_samples.tolist()
                },
                'eta_scores': eta_scores.tolist(),
                'softmax_weights': softmax_weights.tolist(),
                'final_weights': robust_weights.tolist(),
                'temperature': self.config.temperature,
                'weight_coefficients': {
                    'ic_weight': self.config.ic_weight,
                    'sharpe_weight': self.config.sharpe_weight,
                    'sample_weight': self.config.sample_weight
                }
            }
            
            self.logger.info(f"Softmax权重计算完成: {weight_dict}")
            return weight_dict, details
            
        except Exception as e:
            self.logger.error(f"Softmax权重计算失败: {e}")
            return self._fallback_equal_weights(model_performance), {'error': str(e)}
    
    def _calculate_information_criterion_weights(self,
                                               model_performance: Dict[str, Dict],
                                               predictions: Dict[str, np.ndarray],
                                               targets: np.ndarray) -> Tuple[Dict[str, float], Dict]:
        """
        方案B: 信息准则/贝叶斯证据权重计算
        
        实现公式: w_m ∝ exp(-AIC_m/2) 或基于CV log-loss
        """
        try:
            model_names = list(predictions.keys())
            if len(model_names) < 2:
                return self._fallback_equal_weights(model_performance), {'reason': 'insufficient_models'}
            
            criterion_values = []
            
            for model_name in model_names:
                pred = predictions[model_name]
                
                if len(pred) != len(targets):
                    self.logger.warning(f"模型{model_name}预测长度不匹配，跳过")
                    criterion_values.append(np.inf)
                    continue
                
                # 计算信息准则
                if self.config.criterion_type == "aic":
                    aic = self._calculate_aic(pred, targets)
                    criterion_values.append(aic)
                elif self.config.criterion_type == "bic":
                    bic = self._calculate_bic(pred, targets, len(pred))
                    criterion_values.append(bic)
                elif self.config.criterion_type == "cv_log_loss":
                    cv_loss = self._calculate_cv_log_loss(pred, targets)
                    criterion_values.append(cv_loss)
                else:
                    # 默认使用AIC
                    aic = self._calculate_aic(pred, targets)
                    criterion_values.append(aic)
            
            # 基于信息准则计算权重
            # w_m ∝ exp(-criterion_m/2)
            criterion_values = np.array(criterion_values)
            exp_weights = np.exp(-criterion_values / 2)
            
            # 处理无穷大值
            exp_weights = np.where(np.isfinite(exp_weights), exp_weights, 0.0)
            
            # 标准化权重
            total_weight = np.sum(exp_weights)
            if total_weight > 0:
                robust_weights = exp_weights / total_weight
            else:
                robust_weights = np.ones(len(model_names)) / len(model_names)
            
            # 应用最小权重约束
            robust_weights = self._apply_min_weight_constraint(robust_weights)
            
            weight_dict = dict(zip(model_names, robust_weights))
            
            details = {
                'method': f'information_criterion_{self.config.criterion_type}',
                'criterion_values': criterion_values.tolist(),
                'exp_weights': exp_weights.tolist(),
                'final_weights': robust_weights.tolist(),
                'criterion_type': self.config.criterion_type
            }
            
            self.logger.info(f"信息准则权重计算完成: {weight_dict}")
            return weight_dict, details
            
        except Exception as e:
            self.logger.error(f"信息准则权重计算失败: {e}")
            return self._fallback_equal_weights(model_performance), {'error': str(e)}
    
    def _robust_zscore(self, values: List[float]) -> np.ndarray:
        """稳健Z-score标准化 (使用中位数和MAD)"""
        values = np.array(values)
        
        if len(values) < 2:
            return values
        
        # 使用中位数和绝对中位差(MAD)进行稳健标准化
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        
        if mad == 0:
            # 如果MAD为0，回退到标准标准化
            return stats.zscore(values, nan_policy='omit')
        
        # 稳健Z-score: (x - median) / (1.4826 * MAD)
        # 1.4826是使MAD等价于正态分布标准差的常数
        robust_zscore = (values - median) / (1.4826 * mad)
        
        return robust_zscore
    
    def _filter_outliers(self, ic_scores: List[float], sharpe_ratios: List[float], 
                        sample_sizes: List[int]) -> Tuple[List[float], List[float], List[int]]:
        """过滤异常值"""
        try:
            # 对IC和Sharpe分别进行异常值检测
            ic_array = np.array(ic_scores)
            sharpe_array = np.array(sharpe_ratios)
            
            # 使用Z-score方法检测异常值
            ic_z = np.abs(stats.zscore(ic_array, nan_policy='omit'))
            sharpe_z = np.abs(stats.zscore(sharpe_array, nan_policy='omit'))
            
            # 标记非异常值的索引
            valid_mask = (
                (ic_z < self.config.outlier_threshold) &
                (sharpe_z < self.config.outlier_threshold) &
                (~np.isnan(ic_z)) & 
                (~np.isnan(sharpe_z))
            )
            
            if np.sum(valid_mask) < 2:
                # 如果过滤后数据太少，不进行过滤
                self.logger.warning("异常值过滤会导致数据过少，跳过过滤")
                return ic_scores, sharpe_ratios, sample_sizes
            
            # 应用过滤
            filtered_ic = [ic_scores[i] for i in range(len(ic_scores)) if valid_mask[i]]
            filtered_sharpe = [sharpe_ratios[i] for i in range(len(sharpe_ratios)) if valid_mask[i]]
            filtered_samples = [sample_sizes[i] for i in range(len(sample_sizes)) if valid_mask[i]]
            
            self.logger.debug(f"异常值过滤: {len(ic_scores)} → {len(filtered_ic)} 个模型")
            return filtered_ic, filtered_sharpe, filtered_samples
            
        except Exception as e:
            self.logger.warning(f"异常值过滤失败: {e}，使用原始数据")
            return ic_scores, sharpe_ratios, sample_sizes
    
    def _apply_min_weight_constraint(self, weights: np.ndarray) -> np.ndarray:
        """应用最小权重约束并重新标准化"""
        # 应用最小权重下限
        constrained_weights = np.maximum(weights, self.config.min_weight)
        
        # 重新标准化使权重和为1
        total_weight = np.sum(constrained_weights)
        if total_weight > 0:
            constrained_weights = constrained_weights / total_weight
        
        return constrained_weights
    
    def _calculate_aic(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算AIC (Akaike Information Criterion)"""
        try:
            # 计算残差平方和
            residuals = targets - predictions
            rss = np.sum(residuals ** 2)
            
            # AIC = n * log(RSS/n) + 2*k
            # 这里假设k=1 (简单线性模型)
            n = len(targets)
            k = 1
            
            if rss <= 0 or n <= 0:
                return np.inf
            
            aic = n * np.log(rss / n) + 2 * k
            return aic
            
        except Exception as e:
            self.logger.warning(f"AIC计算失败: {e}")
            return np.inf
    
    def _calculate_bic(self, predictions: np.ndarray, targets: np.ndarray, n_params: int) -> float:
        """计算BIC (Bayesian Information Criterion)"""
        try:
            residuals = targets - predictions
            rss = np.sum(residuals ** 2)
            
            # BIC = n * log(RSS/n) + k * log(n)
            n = len(targets)
            k = n_params
            
            if rss <= 0 or n <= 0:
                return np.inf
            
            bic = n * np.log(rss / n) + k * np.log(n)
            return bic
            
        except Exception as e:
            self.logger.warning(f"BIC计算失败: {e}")
            return np.inf
    
    def _calculate_cv_log_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算交叉验证对数损失"""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import make_scorer, mean_squared_error
            
            # 重新构建简单线性回归进行CV评估
            X = predictions.reshape(-1, 1)
            y = targets
            
            model = LinearRegression()
            cv_scores = cross_val_score(
                model, X, y,
                cv=self.config.cv_folds,
                scoring=make_scorer(mean_squared_error, greater_is_better=False)
            )
            
            # 返回平均CV损失
            return -np.mean(cv_scores)
            
        except Exception as e:
            self.logger.warning(f"CV Log Loss计算失败: {e}")
            return np.inf
    
    def _fallback_equal_weights(self, model_performance: Dict[str, Dict]) -> Dict[str, float]:
        """回退到等权重"""
        model_names = list(model_performance.keys())
        n_models = len(model_names)
        
        if n_models == 0:
            return {}
        
        equal_weight = 1.0 / n_models
        return {name: equal_weight for name in model_names}
    
    def update_historical_performance(self, model_name: str, performance_metrics: Dict):
        """更新历史性能数据"""
        if model_name not in self.historical_performance:
            self.historical_performance[model_name] = []
        
        self.historical_performance[model_name].append(performance_metrics)
        
        # 保持滚动窗口大小
        max_history = max(self.config.ic_window, self.config.sharpe_window)
        if len(self.historical_performance[model_name]) > max_history:
            self.historical_performance[model_name] = (
                self.historical_performance[model_name][-max_history:]
            )
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """获取诊断信息"""
        return {
            'config': {
                'ic_weight': self.config.ic_weight,
                'sharpe_weight': self.config.sharpe_weight,
                'sample_weight': self.config.sample_weight,
                'temperature': self.config.temperature,
                'min_weight': self.config.min_weight,
                'use_information_criterion': self.config.use_information_criterion
            },
            'historical_models': list(self.historical_performance.keys()),
            'historical_count': {
                name: len(hist) for name, hist in self.historical_performance.items()
            }
        }


def create_robust_bma_weighting(ic_weight: float = 0.6,
                               sharpe_weight: float = 0.4,
                               temperature: float = 0.7,
                               min_weight: float = 0.03,
                               use_information_criterion: bool = False) -> RobustBMAWeighting:
    """
    创建稳健BMA权重系统的便捷函数
    
    Args:
        ic_weight: IC权重系数
        sharpe_weight: Sharpe权重系数  
        temperature: Softmax温度参数
        min_weight: 最小权重下限
        use_information_criterion: 是否使用信息准则
        
    Returns:
        配置好的稳健权重系统
    """
    config = RobustWeightingConfig(
        ic_weight=ic_weight,
        sharpe_weight=sharpe_weight,
        temperature=temperature,
        min_weight=min_weight,
        use_information_criterion=use_information_criterion
    )
    
    return RobustBMAWeighting(config)


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建稳健权重系统
    robust_weighting = create_robust_bma_weighting(
        ic_weight=0.6,
        sharpe_weight=0.4,
        temperature=0.7,
        min_weight=0.03
    )
    
    print("=== 稳健BMA权重系统测试 ===")
    
    # 模拟模型性能数据
    test_performance = {
        'xgboost': {
            'ic_score': 0.15,
            'sharpe_ratio': 1.2,
            'n_samples': 1000
        },
        'lightgbm': {
            'ic_score': 0.12,
            'sharpe_ratio': 0.9,
            'n_samples': 1200
        },
        'random_forest': {
            'ic_score': 0.08,
            'sharpe_ratio': 0.7,
            'n_samples': 800
        },
        'linear_model': {
            'ic_score': 0.05,
            'sharpe_ratio': 0.4,
            'n_samples': 1500
        },
        'neural_net': {
            'ic_score': 0.18,
            'sharpe_ratio': 1.5,
            'n_samples': 600
        }
    }
    
    # 计算稳健权重
    weights, details = robust_weighting.calculate_robust_weights(test_performance, {})
    
    print("方案A: 稳健Softmax权重")
    for model, weight in weights.items():
        perf = test_performance[model]
        print(f"{model:15s}: {weight:.3f} (IC={perf['ic_score']:.3f}, SR={perf['sharpe_ratio']:.1f}, N={perf['n_samples']})")
    
    print(f"\n温度参数: {details.get('temperature', 'N/A')}")
    print(f"权重系数: IC={robust_weighting.config.ic_weight}, SR={robust_weighting.config.sharpe_weight}")
    
    # 对比传统简单权重
    print("\n传统简单权重对比:")
    n_models = len(test_performance)
    for model in test_performance:
        traditional_weight = 1.0 / n_models
        print(f"{model:15s}: {traditional_weight:.3f} (等权重)")
    
    print("\n✅ 稳健BMA权重系统测试完成")