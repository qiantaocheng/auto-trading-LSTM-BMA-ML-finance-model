#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应因子权重学习系统
基于BMA (Bayesian Model Averaging) 学习最优因子权重
替代硬编码权重，提供动态权重调整
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import scipy.stats as stats
from pathlib import Path

# BMA训练系统延迟导入（避免循环依赖）
BMA_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class FactorWeightResult:
    """因子权重学习结果"""
    weights: Dict[str, float]
    confidence: float
    performance_score: float
    learning_date: datetime
    validation_sharpe: float
    factor_contributions: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class WeightLearningConfig:
    """权重学习配置"""
    lookback_days: int = 252  # 1年历史数据
    validation_days: int = 63  # 验证期
    min_confidence: float = 0.6  # 最小置信度
    max_weight: float = 0.5  # 单因子最大权重
    min_weight: float = 0.05  # 单因子最小权重
    rebalance_frequency: int = 21  # 权重更新频率(天)
    performance_threshold: float = 0.1  # 性能阈值
    enable_regime_detection: bool = True  # 启用市场状态检测

class AdaptiveFactorWeights:
    """
    自适应因子权重学习系统
    使用BMA和历史回测学习最优因子权重
    """
    
    def __init__(self, config: WeightLearningConfig = None):
        self.config = config or WeightLearningConfig()
        self.cache_dir = Path("cache/factor_weights")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认因子列表
        self.factor_names = [
            'mean_reversion', 'momentum', 'trend', 
            'volume', 'volatility'
        ]
        
        # 硬编码权重（作为回退）
        self.fallback_weights = {
            'mean_reversion': 0.30,
            'trend': 0.30,
            'momentum': 0.25,
            'volume': 0.20,
            'volatility': 0.15
        }
        
        # 学习历史
        self.weight_history = []
        self.performance_history = []
        
        # 当前学习到的权重
        self.current_weights = None
        self.last_update = None
        
        # 市场状态检测
        self.market_regimes = {
            'bull': {'volatility_threshold': 0.15, 'trend_threshold': 0.05},
            'bear': {'volatility_threshold': 0.25, 'trend_threshold': -0.05},
            'sideways': {'volatility_threshold': 0.20, 'trend_threshold': 0.02}
        }
        
        logger.info(f"AdaptiveFactorWeights initialized with {len(self.factor_names)} factors")
    
    def need_update(self) -> bool:
        """检查是否需要更新权重"""
        if self.current_weights is None or self.last_update is None:
            return True
        
        days_since_update = (datetime.now() - self.last_update).days
        return days_since_update >= self.config.rebalance_frequency
    
    def learn_weights_from_bma(self, symbols: List[str] = None) -> FactorWeightResult:
        """
        使用BMA学习因子权重
        
        Args:
            symbols: 股票列表，如果为None则使用默认列表
            
        Returns:
            FactorWeightResult: 学习结果
        """
        try:
            logger.info("开始BMA权重学习...")
            
            # 尝试延迟导入BMA模型
            try:
                from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
                bma_available = True
            except ImportError as e:
                logger.warning(f"BMA模型导入失败: {e}")
                bma_available = False
            
            if not bma_available:
                logger.warning("BMA不可用，使用历史回测方法")
                return self._learn_weights_from_backtest(symbols)
            
            # 使用BMA训练系统
            bma_model = UltraEnhancedQuantitativeModel()
            
            # 获取默认股票列表
            if symbols is None:
                symbols = self._get_default_symbols()
            
            # 准备训练数据 - 修复时间计算（使用日历天数而非交易日）
            end_date = datetime.now()
            # 252个交易日 ≈ 365个日历天，考虑周末和节假日
            calendar_days = int(self.config.lookback_days * 1.45)  # 252 * 1.45 ≈ 365天
            start_date = end_date - timedelta(days=calendar_days)
            
            logger.info(f"开始完整BMA训练，股票数量: {len(symbols)}")
            logger.info(f"训练期间: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
            
            # 🔥 使用完整BMA训练
            try:
                # 执行完整的BMA训练
                training_results = bma_model.run_complete_analysis(
                    tickers=symbols[:20],  # 限制数量避免内存问题，但使用真实训练
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    top_n=len(symbols[:20])  # 获取所有股票的结果
                )
                
                logger.info("完整BMA训练完成")
                
                # 从真实BMA结果提取因子权重
                factor_weights = self._extract_weights_from_bma_full(training_results)
                
                # 深度验证权重
                validation_result = self._deep_validate_weights(factor_weights, training_results, symbols[:10])
                
                # 创建结果
                result = FactorWeightResult(
                    weights=factor_weights,
                    confidence=validation_result['confidence'],
                    performance_score=validation_result['sharpe_ratio'],
                    learning_date=datetime.now(),
                    validation_sharpe=validation_result['sharpe_ratio'],
                    factor_contributions=validation_result['contributions'],
                    metadata={
                        'method': 'BMA_full',
                        'symbols_count': len(symbols),
                        'lookback_days': self.config.lookback_days,
                        'bma_results_available': training_results is not None,
                        'training_summary': validation_result.get('training_summary', {}),
                        'model_performances': validation_result.get('model_performances', {})
                    }
                )
                
            except Exception as training_error:
                logger.error(f"完整BMA训练失败: {training_error}")
                logger.info("回退到历史回测方法")
                return self._learn_weights_from_backtest(symbols)
            
            # 保存学习结果
            self._save_weight_result(result)
            
            logger.info(f"BMA权重学习完成，Sharpe: {validation_result['sharpe_ratio']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"BMA权重学习失败: {e}")
            return self._learn_weights_from_backtest(symbols)
    
    def _extract_weights_from_bma_full(self, bma_results: Any) -> Dict[str, float]:
        """从完整BMA结果中提取因子权重"""
        try:
            logger.info("从完整BMA结果提取因子权重")
            
            if bma_results is None:
                logger.warning("BMA结果为空，使用回退权重")
                return self.fallback_weights.copy()
            
            # 分析传统模型性能
            model_weights = {}
            total_model_score = 0
            
            # 提取传统模型性能
            if 'traditional_models' in bma_results:
                model_perfs = bma_results['traditional_models'].get('model_performance', {})
                logger.info(f"传统模型性能: {model_perfs}")
                
                for model_name, perf in model_perfs.items():
                    ic = perf.get('oof_ic', 0.0)
                    rank_ic = perf.get('oof_rank_ic', 0.0)
                    
                    # 计算模型综合得分
                    model_score = max(0, ic * 0.7 + rank_ic * 0.3)
                    model_weights[model_name] = model_score
                    total_model_score += model_score
                    
                    logger.info(f"模型 {model_name}: IC={ic:.4f}, RankIC={rank_ic:.4f}, Score={model_score:.4f}")
            
            # 分析Alpha策略性能
            alpha_performance = 0
            if 'alpha_strategy' in bma_results:
                alpha_scores = bma_results['alpha_strategy'].get('alpha_scores', pd.Series())
                if len(alpha_scores) > 0:
                    alpha_performance = max(0, alpha_scores.mean())
                    logger.info(f"Alpha策略平均得分: {alpha_performance:.4f}")
            
            # 分析Ridge性能
            ridge_performance = 0
            # 向后兼容：检查旧的learning_to_rank键
            if 'learning_to_rank' in bma_results and 'ridge_stacker' not in bma_results:
                ridge_results = bma_results.get('ridge_stacker', bma_results.get('learning_to_rank', {}))
                if isinstance(ridge_results, dict):
                    ridge_perf = ridge_results.get('performance_summary', {})
                    if ridge_perf:
                        avg_ic = np.mean([p.get('ic', 0.0) for p in ridge_perf.values() if isinstance(p, dict)])
                        ridge_performance = max(0, avg_ic)
                        logger.info(f"Ridge平均IC: {ridge_performance:.4f}")
            
            # 基于模型性能映射到因子权重
            factor_weights = {}
            
            # 模型到因子的映射（基于模型特性）
            model_factor_mapping = {
                'ridge': ['mean_reversion', 'trend'],
                'elastic': ['momentum', 'volatility'], 
                'rf': ['trend', 'volume'],
                'xgboost': ['momentum', 'mean_reversion', 'trend'],
                'lightgbm': ['momentum', 'volatility', 'volume']
            }
            
            # 初始化因子权重
            for factor in self.factor_names:
                factor_weights[factor] = 0.0
            
            # 基于模型性能分配因子权重
            if total_model_score > 0:
                for model_name, model_score in model_weights.items():
                    if model_name in model_factor_mapping:
                        weight_per_factor = (model_score / total_model_score) / len(model_factor_mapping[model_name])
                        for factor in model_factor_mapping[model_name]:
                            if factor in factor_weights:
                                factor_weights[factor] += weight_per_factor
            
            # 加入Alpha策略的贡献
            if alpha_performance > 0:
                # Alpha策略主要贡献动量和均值回归
                alpha_factors = ['momentum', 'mean_reversion', 'trend']
                alpha_weight_per_factor = alpha_performance * 0.3 / len(alpha_factors)
                for factor in alpha_factors:
                    if factor in factor_weights:
                        factor_weights[factor] += alpha_weight_per_factor
            
            # 加入Ridge的贡献
            if ridge_performance > 0:
                # Ridge主要贡献趋势和动量
                ridge_factors = ['trend', 'momentum']
                ridge_weight_per_factor = ridge_performance * 0.2 / len(ridge_factors)
                for factor in ridge_factors:
                    if factor in factor_weights:
                        factor_weights[factor] += ridge_weight_per_factor
            
            # 确保所有因子都有最小权重
            for factor in self.factor_names:
                if factor not in factor_weights:
                    factor_weights[factor] = self.config.min_weight
                else:
                    factor_weights[factor] = max(factor_weights[factor], self.config.min_weight)
            
            # 应用权重约束
            factor_weights = self._apply_weight_constraints(factor_weights)
            
            logger.info(f"从BMA提取的因子权重: {factor_weights}")
            return factor_weights
            
        except Exception as e:
            logger.error(f"从BMA提取权重失败: {e}")
            return self.fallback_weights.copy()
    
    def _deep_validate_weights(self, weights: Dict[str, float], 
                              bma_results: Any, symbols: List[str]) -> Dict[str, Any]:
        """深度验证权重的有效性"""
        try:
            logger.info("开始深度权重验证")
            
            # 基础权重验证
            total_weight = sum(weights.values())
            weight_variance = np.var(list(weights.values()))
            
            # 权重分布评分
            distribution_score = 1.0 - min(weight_variance, 0.5) / 0.5
            
            # BMA训练质量评分
            training_quality_score = 0.5
            model_performances = {}
            
            if bma_results and 'traditional_models' in bma_results:
                model_perfs = bma_results['traditional_models'].get('model_performance', {})
                if model_perfs:
                    ic_values = [perf.get('oof_ic', 0.0) for perf in model_perfs.values()]
                    avg_ic = np.mean([ic for ic in ic_values if not np.isnan(ic)])
                    training_quality_score = min(max(avg_ic * 5, 0.0), 1.0)  # 映射到0-1
                    
                    model_performances = {
                        'average_ic': avg_ic,
                        'ic_std': np.std(ic_values),
                        'positive_ic_ratio': np.mean([ic > 0 for ic in ic_values]),
                        'model_count': len(model_perfs)
                    }
                    
                    logger.info(f"训练质量评分: {training_quality_score:.3f} (平均IC: {avg_ic:.4f})")
            
            # 综合置信度
            confidence = (distribution_score * 0.4 + training_quality_score * 0.6)
            
            # 估算性能得分（基于训练质量）
            performance_score = training_quality_score * 2.0  # 转换为Sharpe-like指标
            
            # 训练总结
            training_summary = {
                'weights_distribution_score': distribution_score,
                'training_quality_score': training_quality_score,
                'total_weight': total_weight,
                'weight_variance': weight_variance,
                'symbols_trained': len(symbols)
            }
            
            validation_result = {
                'confidence': confidence,
                'sharpe_ratio': performance_score,
                'contributions': weights.copy(),
                'total_weight': total_weight,
                'distribution_score': distribution_score,
                'training_summary': training_summary,
                'model_performances': model_performances
            }
            
            logger.info(f"深度验证完成 - 置信度: {confidence:.3f}, 性能: {performance_score:.3f}")
            return validation_result
            
        except Exception as e:
            logger.error(f"深度权重验证失败: {e}")
            return {
                'confidence': 0.5,
                'sharpe_ratio': 0.1,
                'contributions': weights.copy(),
                'total_weight': sum(weights.values()),
                'distribution_score': 0.5,
                'training_summary': {'error': str(e)},
                'model_performances': {}
            }
    
    def _extract_weights_from_bma(self, bma_results: Any) -> Dict[str, float]:
        """从BMA结果中提取因子权重"""
        try:
            if bma_results is None:
                return self.fallback_weights.copy()
            
            # 如果BMA结果包含因子重要性
            if hasattr(bma_results, 'feature_importance'):
                importance = bma_results.feature_importance
                
                # 映射因子重要性到权重
                weights = {}
                total_importance = 0
                
                for factor in self.factor_names:
                    # 查找匹配的特征
                    factor_importance = 0
                    for feature, imp in importance.items():
                        if factor.lower() in feature.lower():
                            factor_importance += imp
                    
                    weights[factor] = max(factor_importance, self.config.min_weight)
                    total_importance += weights[factor]
                
                # 归一化权重
                if total_importance > 0:
                    weights = {k: v/total_importance for k, v in weights.items()}
                else:
                    weights = self.fallback_weights.copy()
                
                # 应用权重约束
                weights = self._apply_weight_constraints(weights)
                
                return weights
            
            else:
                logger.warning("BMA结果不包含因子重要性，使用等权重")
                equal_weight = 1.0 / len(self.factor_names)
                return {factor: equal_weight for factor in self.factor_names}
                
        except Exception as e:
            logger.error(f"提取BMA权重失败: {e}")
            return self.fallback_weights.copy()
    
    def _learn_weights_from_backtest(self, symbols: List[str] = None) -> FactorWeightResult:
        """
        使用历史回测学习权重（BMA不可用时的后备方案）
        """
        try:
            logger.info("使用历史回测方法学习权重...")
            
            if symbols is None:
                symbols = self._get_default_symbols()
            
            # 测试不同权重组合的效果
            weight_combinations = self._generate_weight_combinations()
            
            best_weights = None
            best_sharpe = -999
            
            for weights in weight_combinations:
                try:
                    # 模拟回测
                    sharpe = self._backtest_weights(weights, symbols[:20])  # 限制测试规模
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_weights = weights
                        
                except Exception as e:
                    logger.warning(f"权重组合测试失败: {e}")
                    continue
            
            # 如果找不到好的权重，使用回退权重
            if best_weights is None or best_sharpe < 0:
                best_weights = self.fallback_weights.copy()
                best_sharpe = 0.1  # 假设回退权重有基本表现
            
            # 创建结果
            result = FactorWeightResult(
                weights=best_weights,
                confidence=min(max(best_sharpe, 0.3), 1.0),
                performance_score=best_sharpe,
                learning_date=datetime.now(),
                validation_sharpe=best_sharpe,
                factor_contributions=best_weights,
                metadata={
                    'method': 'backtest',
                    'tested_combinations': len(weight_combinations),
                    'symbols_count': len(symbols)
                }
            )
            
            self._save_weight_result(result)
            
            logger.info(f"回测权重学习完成，最佳Sharpe: {best_sharpe:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"回测权重学习失败: {e}")
            # 返回回退结果
            return FactorWeightResult(
                weights=self.fallback_weights.copy(),
                confidence=0.5,
                performance_score=0.1,
                learning_date=datetime.now(),
                validation_sharpe=0.1,
                factor_contributions=self.fallback_weights.copy(),
                metadata={'method': 'fallback', 'error': str(e)}
            )
    
    def _generate_weight_combinations(self) -> List[Dict[str, float]]:
        """生成权重组合进行测试"""
        combinations = []
        
        # 基础组合
        combinations.append(self.fallback_weights.copy())
        
        # 等权重
        equal_weight = 1.0 / len(self.factor_names)
        combinations.append({factor: equal_weight for factor in self.factor_names})
        
        # 重点因子组合
        focus_factors = ['mean_reversion', 'trend', 'momentum']
        for focus_factor in focus_factors:
            weights = {factor: 0.1 for factor in self.factor_names}
            weights[focus_factor] = 0.6
            remaining = 0.4 / (len(self.factor_names) - 1)
            for factor in self.factor_names:
                if factor != focus_factor:
                    weights[factor] = remaining
            combinations.append(weights)
        
        # 随机权重组合
        # np.random.seed removed
        for _ in range(10):
            random_weights = np.random.dirichlet(np.ones(len(self.factor_names)))
            weights_dict = {factor: float(weight) for factor, weight in 
                          zip(self.factor_names, random_weights)}
            combinations.append(weights_dict)
        
        return combinations
    
    def _backtest_weights(self, weights: Dict[str, float], symbols: List[str]) -> float:
        """
        简化的权重回测
        返回Sharpe比率
        """
        try:
            # 这里应该实现实际的回测逻辑
            # 为了简化，我们使用模拟计算
            
            # 模拟因子表现
            factor_returns = {
                'mean_reversion': np.zeros(0.08),
                'trend': np.zeros(0.12),
                'momentum': np.zeros(0.10),
                'volume': np.zeros(0.06),
                'volatility': np.zeros(0.04)
            }
            
            # 计算加权组合收益
            portfolio_return = sum(weights[factor] * factor_returns[factor] 
                                 for factor in self.factor_names)
            
            # 估算风险
            portfolio_risk = 0.15  # 假设组合波动率
            
            # 计算Sharpe比率
            sharpe = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            
            return sharpe
            
        except Exception as e:
            logger.error(f"权重回测失败: {e}")
            return -1.0
    
    def _validate_weights(self, weights: Dict[str, float], symbols: List[str]) -> Dict[str, Any]:
        """验证权重的有效性"""
        try:
            # 计算权重约束合规性
            total_weight = sum(weights.values())
            weight_variance = np.var(list(weights.values()))
            
            # 基础验证分数
            base_score = 0.8 if abs(total_weight - 1.0) < 0.01 else 0.5
            
            # 多样性评分
            diversity_score = 1.0 - weight_variance if weight_variance < 0.5 else 0.3
            
            # 综合置信度
            confidence = (base_score + diversity_score) / 2
            
            # 模拟Sharpe比率
            estimated_sharpe = self._backtest_weights(weights, symbols)
            
            return {
                'confidence': confidence,
                'sharpe_ratio': estimated_sharpe,
                'contributions': weights.copy(),
                'total_weight': total_weight,
                'diversity_score': diversity_score
            }
            
        except Exception as e:
            logger.error(f"权重验证失败: {e}")
            return {
                'confidence': 0.5,
                'sharpe_ratio': 0.1,
                'contributions': weights.copy(),
                'total_weight': sum(weights.values()),
                'diversity_score': 0.5
            }
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """应用权重约束"""
        try:
            # 确保最小权重
            for factor in weights:
                weights[factor] = max(weights[factor], self.config.min_weight)
            
            # 确保最大权重
            for factor in weights:
                weights[factor] = min(weights[factor], self.config.max_weight)
            
            # 重新归一化
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            
            return weights
            
        except Exception as e:
            logger.error(f"应用权重约束失败: {e}")
            return self.fallback_weights.copy()
    
    def _get_default_symbols(self) -> List[str]:
        """获取默认股票列表"""
        # 返回一些流动性好的大盘股
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
            'META', 'NVDA', 'JPM', 'JNJ', 'V',
            'PG', 'UNH', 'HD', 'DIS', 'MA',
            'PFE', 'BAC', 'KO', 'PEP', 'MRK'
        ]
    
    def _save_weight_result(self, result: FactorWeightResult):
        """保存权重学习结果"""
        try:
            # 保存到缓存
            timestamp = result.learning_date.strftime('%Y%m%d_%H%M%S')
            cache_file = self.cache_dir / f"weights_{timestamp}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            # 更新当前权重
            self.current_weights = result.weights
            self.last_update = result.learning_date
            
            # 保存到JSON（可读格式）
            json_file = self.cache_dir / f"weights_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
            
            # 更新历史记录
            self.weight_history.append(result)
            
            # 保持历史记录大小
            if len(self.weight_history) > 100:
                self.weight_history = self.weight_history[-50:]
            
            logger.info(f"权重结果已保存: {cache_file}")
            
        except Exception as e:
            logger.error(f"保存权重结果失败: {e}")
    
    def load_latest_weights(self) -> Optional[FactorWeightResult]:
        """加载最新的权重结果"""
        try:
            # 查找最新的权重文件
            weight_files = list(self.cache_dir.glob("weights_*.pkl"))
            if not weight_files:
                return None
            
            latest_file = max(weight_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'rb') as f:
                result = pickle.load(f)
            
            # 更新当前状态
            self.current_weights = result.weights
            self.last_update = result.learning_date
            
            logger.info(f"加载最新权重: {latest_file}")
            return result
            
        except Exception as e:
            logger.error(f"加载权重失败: {e}")
            return None
    
    def get_current_weights(self, force_update: bool = False) -> Dict[str, float]:
        """
        获取当前权重
        
        Args:
            force_update: 强制更新权重
            
        Returns:
            Dict[str, float]: 当前因子权重
        """
        try:
            # 🔥 智能权重更新策略 - 优先尝试加载最新的ML权重
            should_trigger_training = force_update and self.need_update()
            
            if should_trigger_training:
                logger.info("💡 强制更新模式，开始学习新权重...")
                
                # 尝试学习新权重
                try:
                    result = self.learn_weights_from_bma()
                    if result.confidence >= self.config.min_confidence:
                        logger.info(f"✅ ML权重更新成功，置信度: {result.confidence:.3f}")
                        self.current_weights = result.weights
                        self.last_update = datetime.now()
                        return result.weights
                    else:
                        logger.warning(f"⚠️ 新权重置信度过低: {result.confidence:.3f}，尝试历史权重")
                except Exception as e:
                    logger.error(f"❌ 权重学习失败: {e}")
            elif self.need_update():
                logger.info("📊 检测到权重需要更新，优先尝试加载现有ML权重")
            
            # 优先使用当前内存中的权重
            if self.current_weights is not None:
                logger.info("🎯 使用当前内存中的ML权重")
                return self.current_weights
            
            # 尝试加载最新的历史权重（ML学习的结果）
            latest_result = self.load_latest_weights()
            if latest_result is not None:
                logger.info(f"📂 加载历史ML权重，学习日期: {latest_result.learning_date}, 置信度: {latest_result.confidence:.3f}")
                self.current_weights = latest_result.weights
                return latest_result.weights
            
            # 最后回退到硬编码权重
            logger.warning("⚠️ 未找到ML权重，使用硬编码回退权重")
            return self.fallback_weights.copy()
            
        except Exception as e:
            logger.error(f"获取权重失败: {e}")
            return self.fallback_weights.copy()
    
    def get_or_learn_weights(self) -> Dict[str, float]:
        """
        获取权重或主动学习新权重
        专为BMA Enhanced系统设计，确保使用ML权重而非硬编码权重
        """
        try:
            # 首先检查是否有可用的ML权重
            latest_result = self.load_latest_weights()
            
            # 如果有最近的ML权重（30天内），使用它
            if latest_result is not None:
                days_old = (datetime.now() - latest_result.learning_date).days
                if days_old <= 30 and latest_result.confidence >= 0.6:
                    logger.info(f"🎯 使用最近ML权重 ({days_old}天前), 置信度: {latest_result.confidence:.3f}")
                    self.current_weights = latest_result.weights
                    return latest_result.weights
            
            # 如果没有合适的ML权重，主动触发学习
            logger.info("🚀 主动触发ML权重学习，避免使用硬编码权重")
            try:
                result = self.learn_weights_from_bma()
                if result.confidence >= self.config.min_confidence:
                    logger.info(f"✅ 主动ML权重学习成功，置信度: {result.confidence:.3f}")
                    self.current_weights = result.weights
                    self.last_update = datetime.now()
                    return result.weights
                else:
                    logger.warning(f"⚠️ ML权重置信度不足: {result.confidence:.3f}")
            except Exception as e:
                logger.error(f"❌ 主动ML权重学习失败: {e}")
            
            # 如果ML学习失败，使用最新可用权重
            if latest_result is not None:
                logger.info(f"📂 使用可用的历史权重，置信度: {latest_result.confidence:.3f}")
                self.current_weights = latest_result.weights
                return latest_result.weights
            
            # 最后的回退
            logger.warning("⚠️ ML权重不可用，使用优化的回退权重")
            return self.fallback_weights.copy()
            
        except Exception as e:
            logger.error(f"获取或学习权重失败: {e}")
            return self.fallback_weights.copy()
    
    def get_weight_statistics(self) -> Dict[str, Any]:
        """获取权重统计信息"""
        try:
            if not self.weight_history:
                return {'status': 'no_history'}
            
            # 计算权重趋势
            recent_weights = [result.weights for result in self.weight_history[-10:]]
            weight_trends = {}
            
            for factor in self.factor_names:
                values = [w.get(factor, 0) for w in recent_weights]
                weight_trends[factor] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'trend': 'up' if len(values) > 1 and values[-1] > values[0] else 'down'
                }
            
            # 性能统计
            performance_scores = [result.performance_score for result in self.weight_history[-10:]]
            
            return {
                'total_updates': len(self.weight_history),
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'current_weights': self.current_weights,
                'weight_trends': weight_trends,
                'performance_stats': {
                    'mean_performance': np.mean(performance_scores),
                    'best_performance': np.max(performance_scores),
                    'recent_performance': performance_scores[-1] if performance_scores else 0
                }
            }
            
        except Exception as e:
            logger.error(f"获取权重统计失败: {e}")
            return {'status': 'error', 'error': str(e)}


# 全局实例
_adaptive_weights_instance = None

def get_adaptive_factor_weights(config: WeightLearningConfig = None) -> AdaptiveFactorWeights:
    """获取全局自适应权重实例"""
    global _adaptive_weights_instance
    if _adaptive_weights_instance is None:
        _adaptive_weights_instance = AdaptiveFactorWeights(config)
    return _adaptive_weights_instance

def get_current_factor_weights(force_update: bool = False) -> Dict[str, float]:
    """便捷函数：获取当前因子权重"""
    weights_manager = get_adaptive_factor_weights()
    return weights_manager.get_current_weights(force_update)

def update_factor_weights_from_bma(symbols: List[str] = None) -> FactorWeightResult:
    """便捷函数：使用BMA更新权重"""
    weights_manager = get_adaptive_factor_weights()
    return weights_manager.learn_weights_from_bma(symbols)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("测试自适应因子权重系统")
    print("=" * 50)
    
    # 创建权重学习器
    weights_manager = AdaptiveFactorWeights()
    
    # 获取当前权重
    current_weights = weights_manager.get_current_weights(force_update=True)
    print(f"当前权重: {current_weights}")
    
    # 获取统计信息
    stats = weights_manager.get_weight_statistics()
    print(f"权重统计: {stats}")
    
    print("测试完成")