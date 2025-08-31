"""
高级Alpha系统集成
==================
整合所有专业功能的完整Alpha系统
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

# 导入所有高级模块
from professional_factor_library import ProfessionalFactorCalculator, FactorDecayConfig
from ml_optimized_ic_weights import MLOptimizedICWeights, MLOptimizationConfig  
from factor_orthogonalization import FactorOrthogonalizer, adaptive_orthogonalization
from realtime_performance_monitor import RealtimePerformanceMonitor, AlertThresholds
from alpha_ic_weighted_processor import ICWeightedAlphaProcessor, ICWeightedConfig

logger = logging.getLogger(__name__)


class AdvancedAlphaSystem:
    """
    高级Alpha系统
    集成所有专业量化功能
    """
    
    def __init__(self):
        # 初始化各个组件
        self.factor_calculator = ProfessionalFactorCalculator()
        self.ml_optimizer = MLOptimizedICWeights()
        self.orthogonalizer = FactorOrthogonalizer(method='sequential')
        self.performance_monitor = RealtimePerformanceMonitor()
        self.ic_processor = ICWeightedAlphaProcessor()
        
        # 配置
        self.decay_config = FactorDecayConfig()
        
        # 状态
        self.is_initialized = False
        self.last_update = None
        
        logger.info("高级Alpha系统初始化完成")
    
    def process_complete_pipeline(self, 
                                 raw_data: pd.DataFrame,
                                 returns: pd.Series,
                                 market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        完整处理流程
        
        Pipeline:
        1. 计算专业因子（Fama-French + Barra）
        2. 应用因子衰减
        3. 因子正交化（剔除共线性）
        4. IC加权处理
        5. ML优化权重
        6. 实时性能监控
        """
        logger.info("="*60)
        logger.info("开始高级Alpha处理流程")
        logger.info("="*60)
        
        # Step 1: 计算专业因子
        logger.info("\n[Step 1/6] 计算专业因子（Fama-French + Barra）")
        professional_factors = self.factor_calculator.calculate_all_factors(raw_data)
        logger.info(f"  生成 {professional_factors.shape[1]} 个专业因子")
        
        # 显示因子统计
        factor_stats = self.factor_calculator.get_factor_statistics(professional_factors)
        logger.info("  Top因子（按夏普比率）:")
        top_factors = factor_stats.nlargest(5, 'sharpe')
        for factor in top_factors.index:
            logger.info(f"    {factor}: Sharpe={top_factors.loc[factor, 'sharpe']:.2f}")
        
        # Step 2: 应用因子衰减机制
        logger.info("\n[Step 2/6] 应用因子衰减机制")
        decayed_factors = self._apply_advanced_decay(professional_factors)
        logger.info("  因子衰减已应用:")
        for factor in ['news_sentiment', 'price_momentum', 'value_composite']:
            if factor in professional_factors.columns:
                halflife = self.decay_config.FACTOR_HALFLIFE.get(factor, self.decay_config.DEFAULT_HALFLIFE)
                logger.info(f"    {factor}: 半衰期={halflife}天")
        
        # Step 3: 因子正交化
        logger.info("\n[Step 3/6] 因子正交化（剔除共线性）")
        
        # 检测市场状态
        market_regime = self._detect_market_regime(market_data) if market_data is not None else 'normal'
        logger.info(f"  市场状态: {market_regime}")
        
        # 自适应正交化
        orthogonal_factors = adaptive_orthogonalization(decayed_factors, market_regime)
        
        # 获取正交化报告
        orth_report = self.orthogonalizer.get_orthogonalization_report()
        if not orth_report.empty:
            logger.info("  相关性降低Top因子:")
            for i, row in orth_report.head(3).iterrows():
                logger.info(f"    {row['factor']}: {row['corr_reduction']:.1f}%降低")
        
        # Step 4: IC加权处理
        logger.info("\n[Step 4/6] IC加权因子选择和组合")
        ic_weighted_features = self.ic_processor.process_alpha_factors(
            orthogonal_factors,
            returns,
            market_data
        )
        logger.info(f"  IC加权后特征数: {ic_weighted_features.shape[1]}")
        
        # Step 5: ML优化权重
        logger.info("\n[Step 5/6] 机器学习优化权重")
        
        # 获取当前IC权重
        current_ic_weights = self._get_current_ic_weights(ic_weighted_features, returns)
        
        # ML优化
        optimized_weights = self.ml_optimizer.optimize_weights(
            ic_weighted_features,
            returns,
            current_ic_weights
        )
        
        # 应用优化权重
        final_features = self._apply_optimized_weights(ic_weighted_features, optimized_weights)
        
        # 显示权重变化
        logger.info("  权重优化结果:")
        weight_changes = self._compare_weights(current_ic_weights, optimized_weights)
        for factor, change in list(weight_changes.items())[:5]:
            logger.info(f"    {factor}: {change:+.2%}变化")
        
        # Step 6: 性能监控
        logger.info("\n[Step 6/6] 实时性能监控")
        self.performance_monitor.update_metrics(
            predictions=final_features,
            actual_returns=returns,
            factor_data=orthogonal_factors
        )
        
        # 获取性能摘要
        perf_summary = self.performance_monitor.get_performance_summary()
        if perf_summary:
            current = perf_summary.get('current', {})
            logger.info("  当前性能指标:")
            logger.info(f"    Rank IC: {current.get('rank_ic', 0):.4f}")
            logger.info(f"    Sharpe Ratio: {current.get('sharpe_ratio', 0):.2f}")
            logger.info(f"    Max Drawdown: {current.get('max_drawdown', 0):.1%}")
            logger.info(f"    Turnover: {current.get('turnover', 0):.1%}")
        
        # 更新状态
        self.last_update = datetime.now()
        self.is_initialized = True
        
        logger.info("\n" + "="*60)
        logger.info("高级Alpha处理完成")
        logger.info(f"最终特征数: {final_features.shape[1]}")
        logger.info("="*60)
        
        return final_features
    
    def _apply_advanced_decay(self, factors: pd.DataFrame) -> pd.DataFrame:
        """应用高级衰减机制"""
        decayed = factors.copy()
        
        for col in factors.columns:
            if col in self.decay_config.FACTOR_HALFLIFE:
                halflife = self.decay_config.FACTOR_HALFLIFE[col]
                decay_rate = np.log(2) / halflife
                
                # 应用指数衰减权重到历史数据
                for i in range(len(factors)):
                    days_ago = len(factors) - i - 1
                    if days_ago > 0:
                        decay_weight = np.exp(-decay_rate * days_ago)
                        decayed.iloc[i, factors.columns.get_loc(col)] *= decay_weight
        
        return decayed
    
    def _detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """检测市场状态"""
        if market_data.empty:
            return 'normal'
        
        # 计算市场指标
        returns = market_data['close'].pct_change() if 'close' in market_data.columns else market_data.iloc[:, 0].pct_change()
        
        recent_return = returns.tail(20).mean()
        recent_vol = returns.tail(20).std()
        hist_vol = returns.tail(60).std() if len(returns) > 60 else recent_vol
        
        # 判断市场状态
        if recent_vol > hist_vol * 1.5:
            return 'crisis' if recent_return < -0.002 else 'high_vol'
        elif recent_return > 0.001:
            return 'bull'
        elif recent_return < -0.001:
            return 'bear'
        else:
            return 'normal'
    
    def _get_current_ic_weights(self, features: pd.DataFrame, returns: pd.Series) -> Dict[str, float]:
        """获取当前IC权重"""
        ic_weights = {}
        
        for col in features.columns:
            try:
                from scipy import stats
                ic, _ = stats.spearmanr(features[col], returns)
                ic_weights[col] = abs(ic) if not np.isnan(ic) else 0
            except:
                ic_weights[col] = 0
        
        # 归一化
        total = sum(ic_weights.values())
        if total > 0:
            for col in ic_weights:
                ic_weights[col] /= total
        
        return ic_weights
    
    def _apply_optimized_weights(self, features: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """应用优化后的权重"""
        weighted_features = pd.DataFrame(index=features.index)
        
        for col in features.columns:
            weight = weights.get(col, 1.0 / len(features.columns))
            weighted_features[col] = features[col] * weight
        
        return weighted_features
    
    def _compare_weights(self, original: Dict[str, float], optimized: Dict[str, float]) -> Dict[str, float]:
        """比较权重变化"""
        changes = {}
        
        for factor in original:
            if factor in optimized:
                changes[factor] = optimized[factor] - original[factor]
        
        # 按变化幅度排序
        changes = dict(sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True))
        
        return changes
    
    def get_system_report(self) -> Dict[str, Any]:
        """获取系统完整报告"""
        report = {
            'system_status': {
                'initialized': self.is_initialized,
                'last_update': self.last_update.isoformat() if self.last_update else None,
            },
            'factor_statistics': self.factor_calculator.get_factor_statistics(pd.DataFrame()) if self.is_initialized else {},
            'ml_feature_importance': self.ml_optimizer.get_feature_importance().to_dict() if self.is_initialized else {},
            'performance_summary': self.performance_monitor.get_performance_summary(),
            'orthogonalization_report': self.orthogonalizer.get_orthogonalization_report().to_dict() if self.is_initialized else {}
        }
        
        return report


# 使用示例
def demo_advanced_system():
    """演示高级Alpha系统"""
    
    # 创建测试数据
    n_samples = 500
    n_stocks = 100
    
    # 原始数据
    raw_data = pd.DataFrame({
        'close': np.zeros(n_samples) * 10 + 100,
        'volume': np.random.randint(1000000, 10000000, n_samples),
        'market_cap': 0.0,
        'book_to_market': 0.0,
        'roe': 0.0,
        'asset_growth': 0.0,
        'debt_to_equity': 0.0,
        'earnings': 0.0,
        'price': 0.0
    })
    
    # 未来收益
    returns = pd.Series(np.zeros(n_samples) * 0.02)
    
    # 市场数据
    market_data = pd.DataFrame({
        'close': np.zeros(n_samples) * 5 + 3000
    })
    
    # 创建系统
    system = AdvancedAlphaSystem()
    
    # 运行完整流程
    final_features = system.process_complete_pipeline(
        raw_data=raw_data,
        returns=returns,
        market_data=market_data
    )
    
    # 获取报告
    report = system.get_system_report()
    
    print("\n系统报告摘要:")
    print(f"系统状态: {'已初始化' if report['system_status']['initialized'] else '未初始化'}")
    print(f"性能监控: {report['performance_summary']}")
    
    return system, final_features


if __name__ == "__main__":
    demo_advanced_system()