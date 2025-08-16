"""
Enhanced Target Engineering Module for BMA Ultra Enhanced
目标与标签重构：降噪 Alpha 增益的核心模块

功能：
1. Triple-Barrier Labeling: 先判定胜率，再估幅度，最后合成期望收益
2. 超额收益计算：对基准/行业中位数的相对表现
3. Meta-Labeling Framework: 分离概率预测和幅度预测
4. 自适应样本权重：基于信息量而非简单的波动率倒数

作者：Claude Code Assistant
日期：2025-08-16
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# 配置日志 - 修复编码问题
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LabelType(Enum):
    """标签类型枚举"""
    WIN_RATE = "win_rate"          # 胜率标签 (0/1)
    MAGNITUDE = "magnitude"        # 幅度标签 (连续值)
    EXPECTED_RETURN = "expected"   # 期望收益 (胜率 * 幅度)

@dataclass
class TripleBarrierConfig:
    """Triple Barrier配置"""
    profit_threshold: float = 0.02      # 止盈阈值 (2%)
    loss_threshold: float = -0.015      # 止损阈值 (-1.5%)
    holding_periods: List[int] = None   # 持有期 [1, 3, 5, 10, 15]日
    vertical_barrier_ratio: float = 0.5 # 垂直屏障比例（防止数据泄露）
    
    def __post_init__(self):
        if self.holding_periods is None:
            self.holding_periods = [1, 3, 5, 10, 15]

@dataclass 
class MetaLabelingConfig:
    """Meta-Labeling配置"""
    win_rate_model: str = "logistic"    # 胜率模型：logistic/tree
    magnitude_model: str = "quantile"   # 幅度模型：quantile/huber
    ensemble_method: str = "multiply"   # 合成方法：multiply/weighted
    cv_folds: int = 3                   # 交叉验证折数

class EnhancedTargetEngineer:
    """增强目标工程师"""
    
    def __init__(self, 
                 barrier_config: TripleBarrierConfig = None,
                 meta_config: MetaLabelingConfig = None):
        """
        初始化目标工程师
        
        Args:
            barrier_config: Triple Barrier配置
            meta_config: Meta-Labeling配置
        """
        self.barrier_config = barrier_config or TripleBarrierConfig()
        self.meta_config = meta_config or MetaLabelingConfig()
        
        logger.info("Enhanced Target Engineer initialized")
        logger.info(f"Barrier config: profit={self.barrier_config.profit_threshold:.3f}, "
                   f"loss={self.barrier_config.loss_threshold:.3f}")
        
    def compute_excess_returns(self, 
                             price_data: pd.DataFrame,
                             benchmark_data: Optional[pd.DataFrame] = None,
                             sector_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        计算超额收益（相对基准/行业中位数）
        
        Args:
            price_data: 价格数据 (date, ticker, close)
            benchmark_data: 基准数据 (date, benchmark_close)
            sector_data: 行业数据 (ticker, sector)
            
        Returns:
            超额收益数据框
        """
        logger.info("Computing excess returns...")
        
        # 计算个股收益率
        price_pivot = price_data.pivot(index='date', columns='ticker', values='close')
        stock_returns = price_pivot.pct_change()
        
        excess_returns = stock_returns.copy()
        
        # 方法1: 相对基准的超额收益
        if benchmark_data is not None:
            benchmark_returns = benchmark_data.set_index('date')['benchmark_close'].pct_change()
            # 广播减法：每只股票减去基准收益
            for col in excess_returns.columns:
                excess_returns[col] = stock_returns[col] - benchmark_returns
            logger.info("Using benchmark excess returns")
        
        # 方法2: 相对行业中位数的超额收益  
        elif sector_data is not None:
            sector_mapping = sector_data.set_index('ticker')['sector'].to_dict()
            
            for date in excess_returns.index:
                date_returns = stock_returns.loc[date]
                
                # 按行业计算中位数
                sector_medians = {}
                for ticker in date_returns.index:
                    if pd.notna(date_returns[ticker]) and ticker in sector_mapping:
                        sector = sector_mapping[ticker]
                        if sector not in sector_medians:
                            # 计算该行业当日收益中位数
                            sector_tickers = [t for t, s in sector_mapping.items() if s == sector]
                            sector_returns = date_returns[sector_tickers].dropna()
                            if len(sector_returns) >= 3:  # 至少3只股票
                                sector_medians[sector] = sector_returns.median()
                            else:
                                sector_medians[sector] = 0.0  # 回退到绝对收益
                
                # 计算超额收益
                for ticker in date_returns.index:
                    if ticker in sector_mapping and sector_mapping[ticker] in sector_medians:
                        sector_median = sector_medians[sector_mapping[ticker]]
                        excess_returns.loc[date, ticker] = date_returns[ticker] - sector_median
            
            logger.info("Using sector median excess returns")
        
        # 方法3: 相对市场中位数的超额收益（默认）
        else:
            for date in excess_returns.index:
                date_returns = stock_returns.loc[date].dropna()
                if len(date_returns) >= 10:  # 至少10只股票
                    market_median = date_returns.median()
                    excess_returns.loc[date] = stock_returns.loc[date] - market_median
            logger.info("Using market median excess returns")
        
        # 转换回长格式
        excess_long = excess_returns.reset_index().melt(
            id_vars=['date'], var_name='ticker', value_name='excess_return'
        ).dropna()
        
        logger.info(f"Excess returns computed, samples: {len(excess_long)}")
        return excess_long
    
    def apply_triple_barrier_labeling(self, 
                                    price_data: pd.DataFrame,
                                    excess_returns: pd.DataFrame) -> pd.DataFrame:
        """
        应用Triple Barrier标签方法
        
        Args:
            price_data: 价格数据 (date, ticker, close)  
            excess_returns: 超额收益数据 (date, ticker, excess_return)
            
        Returns:
            包含Triple Barrier标签的数据框
        """
        logger.info("Applying Triple Barrier labeling...")
        
        results = []
        
        # 按股票分组处理
        for ticker in price_data['ticker'].unique():
            ticker_prices = price_data[price_data['ticker'] == ticker].copy()
            ticker_returns = excess_returns[excess_returns['ticker'] == ticker].copy()
            
            if len(ticker_prices) < 20:  # 数据太少跳过
                continue
                
            ticker_prices = ticker_prices.sort_values('date').reset_index(drop=True)
            ticker_returns = ticker_returns.sort_values('date').reset_index(drop=True)
            
            # 合并价格和收益数据
            merged = pd.merge(ticker_prices, ticker_returns, on=['date', 'ticker'], how='inner')
            merged = merged.sort_values('date').reset_index(drop=True)
            
            # 为每个观察点计算Triple Barrier标签
            for i in range(len(merged) - max(self.barrier_config.holding_periods)):
                current_date = merged.loc[i, 'date']
                current_price = merged.loc[i, 'close']
                
                # 计算不同持有期的标签
                for holding_period in self.barrier_config.holding_periods:
                    if i + holding_period >= len(merged):
                        continue
                        
                    # 获取持有期内的价格路径
                    end_idx = min(i + holding_period, len(merged) - 1)
                    price_path = merged.loc[i:end_idx, 'close'].values
                    returns_path = merged.loc[i+1:end_idx+1, 'excess_return'].values
                    
                    # 计算累积超额收益
                    if len(returns_path) > 0:
                        cumulative_return = np.sum(returns_path)
                    else:
                        cumulative_return = 0.0
                    
                    # Triple Barrier逻辑
                    hit_profit = cumulative_return >= self.barrier_config.profit_threshold
                    hit_loss = cumulative_return <= self.barrier_config.loss_threshold
                    
                    # 标签生成
                    if hit_profit:
                        win_label = 1
                        magnitude = cumulative_return
                        barrier_type = "profit"
                    elif hit_loss:
                        win_label = 0  
                        magnitude = cumulative_return
                        barrier_type = "loss"
                    else:
                        # 到期未触及屏障
                        win_label = 1 if cumulative_return > 0 else 0
                        magnitude = cumulative_return
                        barrier_type = "time"
                    
                    # 期望收益 = 胜率指示 * 幅度
                    expected_return = win_label * abs(magnitude) if win_label else magnitude
                    
                    results.append({
                        'date': current_date,
                        'ticker': ticker,
                        'holding_period': holding_period,
                        'win_rate_label': win_label,
                        'magnitude_label': magnitude,
                        'expected_return_label': expected_return,
                        'cumulative_return': cumulative_return,
                        'barrier_type': barrier_type
                    })
        
        result_df = pd.DataFrame(results)
        logger.info(f"Triple Barrier labeling complete, generated {len(result_df)} samples")
        
        if len(result_df) > 0:
            # 统计各类标签分布
            win_rate = result_df['win_rate_label'].mean()
            avg_magnitude = result_df['magnitude_label'].mean()
            barrier_dist = result_df['barrier_type'].value_counts()
            
            logger.info(f"Label stats - Win rate: {win_rate:.3f}, Avg magnitude: {avg_magnitude:.4f}")
            logger.info(f"Barrier distribution: {barrier_dist.to_dict()}")
        
        return result_df
    
    def compute_information_weighted_samples(self, 
                                           features: pd.DataFrame,
                                           labels: pd.DataFrame) -> np.ndarray:
        """
        计算基于信息量的样本权重（替代简单的波动率倒数）
        
        Args:
            features: 特征数据
            labels: 标签数据
            
        Returns:
            样本权重数组
        """
        logger.info("Computing information-weighted samples...")
        
        if len(features) != len(labels):
            raise ValueError(f"特征和标签长度不匹配: {len(features)} vs {len(labels)}")
        
        n_samples = len(features)
        weights = np.ones(n_samples)
        
        try:
            # Method 1: Label variance-based weights (high variance = high information)
            if 'magnitude_label' in labels.columns:
                magnitude_vol = labels['magnitude_label'].rolling(20, min_periods=5).std()
                magnitude_vol = magnitude_vol.fillna(magnitude_vol.median())
                
                # Info weights = normalized variance (not inverse)
                info_weights = magnitude_vol / magnitude_vol.median()
                info_weights = np.clip(info_weights, 0.5, 3.0)  # Limit weight range
                weights *= info_weights
            
            # Method 2: Barrier trigger type-based weights
            if 'barrier_type' in labels.columns:
                type_weights = labels['barrier_type'].map({
                    'profit': 1.2,    # Take profit: high weight (clear signal)
                    'loss': 1.1,      # Stop loss: medium-high weight (risk signal)
                    'time': 0.8       # Time expiry: low weight (ambiguous signal)
                }).fillna(1.0)
                weights *= type_weights
            
            # Method 3: Feature stability-based weights  
            numeric_features = features.select_dtypes(include=[np.number])
            if len(numeric_features.columns) > 0:
                # Compute rolling feature stability (high stability = high weight)
                feature_stability = numeric_features.rolling(10, min_periods=3).std().mean(axis=1)
                feature_stability = feature_stability.fillna(feature_stability.median())
                
                # Stability weights (stable features get higher weights)
                stability_weights = 1 / (1 + feature_stability / feature_stability.median())
                stability_weights = np.clip(stability_weights, 0.7, 1.5)
                weights *= stability_weights
            
            # Normalize weights
            weights = weights / np.median(weights)
            weights = np.clip(weights, 0.1, 5.0)  # Prevent extreme weights
            
            logger.info(f"Info weight stats: mean={np.mean(weights):.3f}, "
                       f"std={np.std(weights):.3f}, "
                       f"range=[{np.min(weights):.3f}, {np.max(weights):.3f}]")
            
        except Exception as e:
            logger.warning(f"Info weight computation failed, using equal weights: {e}")
            weights = np.ones(n_samples)
        
        return weights
    
    def create_meta_labeling_targets(self, 
                                   triple_barrier_labels: pd.DataFrame,
                                   target_type: LabelType = LabelType.EXPECTED_RETURN) -> Tuple[pd.Series, np.ndarray]:
        """
        创建Meta-Labeling目标和样本权重
        
        Args:
            triple_barrier_labels: Triple Barrier标签数据
            target_type: 目标类型（胜率/幅度/期望收益）
            
        Returns:
            (目标序列, 样本权重)
        """
        logger.info(f"Creating Meta-Labeling targets: {target_type.value}")
        
        if len(triple_barrier_labels) == 0:
            raise ValueError("Triple Barrier labels are empty")
        
        # 选择目标列
        if target_type == LabelType.WIN_RATE:
            target_col = 'win_rate_label'
        elif target_type == LabelType.MAGNITUDE:
            target_col = 'magnitude_label'
        else:  # EXPECTED_RETURN
            target_col = 'expected_return_label'
        
        if target_col not in triple_barrier_labels.columns:
            raise ValueError(f"Target column {target_col} does not exist")
        
        # 创建目标序列
        targets = triple_barrier_labels[target_col].copy()
        
        # 异常值处理
        if target_type != LabelType.WIN_RATE:  # 连续值需要去极值
            q99 = targets.quantile(0.99)
            q01 = targets.quantile(0.01)
            targets = np.clip(targets, q01, q99)
        
        # 计算信息权重（基于多个维度）
        feature_data = triple_barrier_labels[['date', 'ticker', 'holding_period']].copy()
        weights = self.compute_information_weighted_samples(feature_data, triple_barrier_labels)
        
        logger.info(f"Meta-Labeling targets created: {len(targets)} samples")
        logger.info(f"Target stats: mean={targets.mean():.4f}, std={targets.std():.4f}")
        
        return targets, weights
    
    def generate_enhanced_targets(self, 
                                price_data: pd.DataFrame,
                                benchmark_data: Optional[pd.DataFrame] = None,
                                sector_data: Optional[pd.DataFrame] = None,
                                target_type: LabelType = LabelType.EXPECTED_RETURN) -> Dict[str, Union[pd.Series, np.ndarray]]:
        """
        生成增强目标和权重（完整流程）
        
        Args:
            price_data: 价格数据 (date, ticker, close)
            benchmark_data: 基准数据 (可选)
            sector_data: 行业数据 (可选)
            target_type: 目标类型
            
        Returns:
            包含目标、权重等的字典
        """
        logger.info("Generating enhanced targets - full pipeline started")
        
        try:
            # Step 1: 计算超额收益
            excess_returns = self.compute_excess_returns(
                price_data, benchmark_data, sector_data
            )
            
            # Step 2: 应用Triple Barrier标签
            triple_labels = self.apply_triple_barrier_labeling(
                price_data, excess_returns
            )
            
            if len(triple_labels) == 0:
                raise ValueError("Triple Barrier label generation failed")
            
            # Step 3: 创建Meta-Labeling目标
            targets, weights = self.create_meta_labeling_targets(
                triple_labels, target_type
            )
            
            # Step 4: 返回完整结果
            result = {
                'targets': targets,
                'sample_weights': weights,
                'excess_returns': excess_returns,
                'triple_labels': triple_labels,
                'meta_info': {
                    'target_type': target_type.value,
                    'n_samples': len(targets),
                    'win_rate': triple_labels['win_rate_label'].mean(),
                    'avg_magnitude': triple_labels['magnitude_label'].mean(),
                    'weight_range': (weights.min(), weights.max())
                }
            }
            
            logger.info("Enhanced target generation completed successfully")
            logger.info(f"Samples: {len(targets)}, Win rate: {result['meta_info']['win_rate']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced target generation failed: {e}")
            raise

def create_enhanced_target_module(barrier_config: TripleBarrierConfig = None,
                                meta_config: MetaLabelingConfig = None) -> EnhancedTargetEngineer:
    """
    创建增强目标工程模块的便捷函数
    
    Args:
        barrier_config: Triple Barrier配置
        meta_config: Meta-Labeling配置
        
    Returns:
        EnhancedTargetEngineer实例
    """
    return EnhancedTargetEngineer(barrier_config, meta_config)

# 示例用法
if __name__ == "__main__":
    # 创建目标工程器
    target_engineer = create_enhanced_target_module()
    
    # 模拟数据
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    price_data = []
    for ticker in tickers:
        for date in dates:
            price = 100 + np.random.randn() * 2  # 模拟价格
            price_data.append({'date': date, 'ticker': ticker, 'close': price})
    
    price_df = pd.DataFrame(price_data)
    
    # 生成增强目标
    result = target_engineer.generate_enhanced_targets(
        price_data=price_df,
        target_type=LabelType.EXPECTED_RETURN
    )
    
    print("Enhanced Target Engineering test completed")
    print(f"Number of targets: {len(result['targets'])}")
    print(f"Weight range: {result['meta_info']['weight_range']}")