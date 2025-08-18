"""
数据与目标对齐 + 延迟闸门传递
===============================

统一使用复权价，强制t-1→t+H对齐，延迟分级→执行限制
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta, timezone
from enum import Enum

logger = logging.getLogger(__name__)

def make_timezone_aware(dt: datetime, default_tz: timezone = timezone.utc) -> datetime:
    """确保datetime对象具有时区信息"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=default_tz)
    return dt

def utc_now() -> datetime:
    """返回UTC当前时间"""
    return datetime.now(timezone.utc)

class DataQuality(Enum):
    """数据质量等级（数值越小质量越好）"""
    FRESH = 1                          # 实时或<5分钟
    USABLE_DELAYED = 2                 # 5-30分钟延迟  
    STALE = 3                          # 30分钟-2小时延迟
    EXPIRED = 4                        # >2小时延迟

class ExecutionPermission(Enum):
    """执行权限等级"""
    FULL_EXECUTION = "FULL_EXECUTION"        # 完全执行权限
    LIMITED_EXECUTION = "LIMITED_EXECUTION"  # 限制执行（仅限价+低参与率）
    NO_EXECUTION = "NO_EXECUTION"           # 禁止执行

class DataAlignmentEngine:
    """数据对齐引擎"""
    
    def __init__(self, target_horizon_days: int = 5, 
                 feature_lag_days: int = 1,
                 min_history_days: int = 252):
        self.target_horizon = target_horizon_days
        self.feature_lag = feature_lag_days  
        self.min_history = min_history_days
        
    def create_aligned_targets(self, adj_close: pd.DataFrame,
                             horizon_days: Optional[int] = None) -> pd.DataFrame:
        """
        创建对齐的前瞻收益目标
        
        Args:
            adj_close: 复权收盘价 (DataFrame: date x symbol)
            horizon_days: 前瞻天数，默认使用配置值
            
        Returns:
            DataFrame: 对齐的前瞻收益
        """
        H = horizon_days or self.target_horizon
        
        logger.info(f"创建{H}天前瞻收益目标")
        
        # 计算前瞻收益：t+H相对于t的收益
        forward_returns = adj_close.shift(-H) / adj_close - 1
        
        # 移除最后H天（无法计算前瞻收益）
        aligned_targets = forward_returns.iloc[:-H]
        
        logger.info(f"生成目标标签: {aligned_targets.shape[0]}行 x {aligned_targets.shape[1]}列")
        
        return aligned_targets
    
    def align_features_to_targets(self, features: pd.DataFrame,
                                targets: pd.DataFrame,
                                lag_days: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        将特征对齐到目标（确保t-1→t+H对齐）
        
        Args:
            features: 特征数据 
            targets: 目标数据
            lag_days: 特征滞后天数
            
        Returns:
            Tuple: (对齐的特征, 对齐的目标)
        """
        lag = lag_days or self.feature_lag
        
        logger.info(f"特征-目标对齐，特征滞后{lag}天")
        
        # 特征向前滞后lag天，确保使用t-lag的特征预测t的目标
        lagged_features = features.shift(lag)
        
        # 找到共同的日期范围
        common_dates = lagged_features.index.intersection(targets.index)
        
        if len(common_dates) < self.min_history:
            logger.warning(f"对齐后样本不足: {len(common_dates)} < {self.min_history}")
            
        # 对齐到共同日期
        aligned_features = lagged_features.loc[common_dates]
        aligned_targets = targets.loc[common_dates]
        
        logger.info(f"对齐完成: {len(common_dates)}个交易日")
        
        return aligned_features, aligned_targets
    
    def validate_temporal_integrity(self, features: pd.DataFrame, 
                                  targets: pd.DataFrame) -> Dict[str, bool]:
        """验证时间完整性，确保无未来信息泄露"""
        
        validation_results = {
            'no_future_leakage': True,
            'sufficient_history': True,
            'consistent_frequency': True,
            'monotonic_index': True
        }
        
        # 检查1：特征日期应早于或等于目标日期  
        if not features.index.max() <= targets.index.max():
            validation_results['no_future_leakage'] = False
            logger.error("检测到未来信息泄露：特征日期晚于目标日期")
            
        # 检查2：足够的历史数据
        if len(features) < self.min_history or len(targets) < self.min_history:
            validation_results['sufficient_history'] = False
            logger.warning(f"历史数据不足: features={len(features)}, targets={len(targets)}")
            
        # 检查3：一致的频率
        feature_freq = pd.infer_freq(features.index)
        target_freq = pd.infer_freq(targets.index)
        if feature_freq != target_freq:
            validation_results['consistent_frequency'] = False
            logger.warning(f"频率不一致: features={feature_freq}, targets={target_freq}")
            
        # 检查4：单调递增的索引
        if not features.index.is_monotonic_increasing or not targets.index.is_monotonic_increasing:
            validation_results['monotonic_index'] = False
            logger.error("索引不是单调递增")
            
        return validation_results
    
    def create_multiindex_features(self, feature_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        创建MultiIndex特征矩阵
        
        Args:
            feature_dict: {feature_name: DataFrame(date x symbol)}
            
        Returns:
            DataFrame: MultiIndex((date, symbol) x features)
        """
        all_features = []
        
        for feature_name, feature_df in feature_dict.items():
            # 将DataFrame重塑为MultiIndex格式
            stacked = feature_df.stack()
            stacked.name = feature_name
            all_features.append(stacked)
            
        if not all_features:
            return pd.DataFrame()
            
        # 合并所有特征
        feature_matrix = pd.concat(all_features, axis=1)
        feature_matrix.index.names = ['date', 'symbol']
        
        logger.info(f"MultiIndex特征矩阵: {feature_matrix.shape}")
        
        return feature_matrix

class DelayGatekeeper:
    """延迟数据门控器"""
    
    def __init__(self):
        self.data_quality_cache = {}
        self.execution_rules = {
            DataQuality.FRESH: ExecutionPermission.FULL_EXECUTION,
            DataQuality.USABLE_DELAYED: ExecutionPermission.LIMITED_EXECUTION,  
            DataQuality.STALE: ExecutionPermission.NO_EXECUTION,
            DataQuality.EXPIRED: ExecutionPermission.NO_EXECUTION
        }
        
    def assess_data_freshness(self, data_timestamp: datetime, 
                            current_time: Optional[datetime] = None) -> DataQuality:
        """评估数据新鲜度"""
        if current_time is None:
            # 使用UTC时间避免时区问题
            current_time = datetime.now(timezone.utc)
        
        # 确保两个时间戳都有时区信息
        if data_timestamp.tzinfo is None:
            # 假设naive时间戳为UTC
            data_timestamp = data_timestamp.replace(tzinfo=timezone.utc)
            logger.debug("数据时间戳缺少时区信息，假设为UTC")
        
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
            logger.debug("当前时间戳缺少时区信息，假设为UTC")
        
        # 将时间戳转换为UTC进行比较
        data_timestamp_utc = data_timestamp.astimezone(timezone.utc)
        current_time_utc = current_time.astimezone(timezone.utc)
            
        age = current_time_utc - data_timestamp_utc
        age_minutes = age.total_seconds() / 60
        
        if age_minutes <= 5:
            return DataQuality.FRESH
        elif age_minutes <= 30:
            return DataQuality.USABLE_DELAYED
        elif age_minutes <= 120:  # 2小时
            return DataQuality.STALE
        else:
            return DataQuality.EXPIRED
    
    def get_execution_permission(self, symbol: str, 
                               data_timestamps: Dict[str, datetime],
                               current_time: Optional[datetime] = None) -> Tuple[ExecutionPermission, Dict]:
        """
        获取执行权限
        
        Args:
            symbol: 股票代码
            data_timestamps: {data_type: timestamp}
            current_time: 当前时间
            
        Returns:
            Tuple: (执行权限, 详细信息)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
            
        # 评估各数据源的新鲜度
        data_quality_scores = {}
        for data_type, timestamp in data_timestamps.items():
            quality = self.assess_data_freshness(timestamp, current_time)
            data_quality_scores[data_type] = quality
            
        # 取最差的数据质量作为整体评估（数值最大的）
        worst_quality = max(data_quality_scores.values(), key=lambda x: x.value)
        
        # 获取执行权限
        execution_permission = self.execution_rules.get(worst_quality, ExecutionPermission.NO_EXECUTION)
        
        # 构建详细信息
        detail_info = {
            'symbol': symbol,
            'overall_quality': worst_quality.value,
            'execution_permission': execution_permission.value,
            'data_quality_breakdown': {k: v.value for k, v in data_quality_scores.items()},
            'assessment_time': current_time,
            'oldest_data_age_minutes': max([
                (current_time - ts).total_seconds() / 60 
                for ts in data_timestamps.values()
            ])
        }
        
        return execution_permission, detail_info
    
    def get_execution_constraints(self, permission: ExecutionPermission) -> Dict:
        """根据执行权限获取约束条件"""
        
        if permission == ExecutionPermission.FULL_EXECUTION:
            return {
                'order_types_allowed': ['MARKET', 'LIMIT', 'STOP'],
                'max_participation_rate': 0.10,
                'max_spread_bps': 50,
                'priority': 'NORMAL'
            }
            
        elif permission == ExecutionPermission.LIMITED_EXECUTION:
            return {
                'order_types_allowed': ['LIMIT'],  # 仅限价单
                'max_participation_rate': 0.05,   # 降低参与率
                'max_spread_bps': 20,             # 更严格的价差限制
                'priority': 'LOW',
                'require_manual_review': False
            }
            
        elif permission == ExecutionPermission.NO_EXECUTION:
            return {
                'order_types_allowed': [],
                'max_participation_rate': 0.0,
                'execution_blocked': True,
                'block_reason': 'Data too stale for safe execution'
            }
            
        return {}

class IntegratedDataPipeline:
    """集成数据管线 - 对齐 + 门控"""
    
    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            'target_horizon_days': 5,
            'feature_lag_days': 1, 
            'min_history_days': 252,
            'enable_data_validation': True,
            'enable_delay_gating': True,
            'max_allowed_delay_minutes': 30
        }
        
        self.config = {**default_config, **(config or {})}
        self.alignment_engine = DataAlignmentEngine(
            target_horizon_days=self.config['target_horizon_days'],
            feature_lag_days=self.config['feature_lag_days'],
            min_history_days=self.config['min_history_days']
        )
        self.delay_gatekeeper = DelayGatekeeper()
        
    def process_training_data(self, adj_close: pd.DataFrame,
                            feature_dict: Dict[str, pd.DataFrame],
                            validate_integrity: bool = True) -> Dict:
        """
        处理训练数据（完整对齐 + 验证）
        
        Args:
            adj_close: 复权收盘价
            feature_dict: 特征字典 {name: DataFrame}
            validate_integrity: 是否验证完整性
            
        Returns:
            Dict: 处理结果
        """
        logger.info("开始处理训练数据")
        
        # 1. 创建对齐的目标
        targets = self.alignment_engine.create_aligned_targets(adj_close)
        
        # 2. 创建MultiIndex特征矩阵
        features = self.alignment_engine.create_multiindex_features(feature_dict)
        
        # 3. 特征-目标对齐
        if not features.empty and not targets.empty:
            # 转换targets为MultiIndex格式以便对齐
            targets_stacked = targets.stack()
            targets_stacked.index.names = ['date', 'symbol']
            
            # 对齐
            aligned_features, aligned_targets = self.alignment_engine.align_features_to_targets(
                features, targets_stacked
            )
        else:
            aligned_features = features
            aligned_targets = targets.stack() if not targets.empty else pd.Series()
            
        # 4. 数据完整性验证
        validation_results = {}
        if validate_integrity and self.config['enable_data_validation']:
            # 将MultiIndex转回DataFrame进行验证
            features_df = aligned_features.unstack('symbol') if not aligned_features.empty else pd.DataFrame()
            targets_df = aligned_targets.unstack('symbol') if not aligned_targets.empty else pd.DataFrame()
            
            validation_results = self.alignment_engine.validate_temporal_integrity(
                features_df, targets_df
            )
            
            if not all(validation_results.values()):
                logger.warning(f"数据完整性验证失败: {validation_results}")
                
        result = {
            'features': aligned_features,
            'targets': aligned_targets,
            'validation': validation_results,
            'config': self.config,
            'processing_timestamp': datetime.now(),
            'samples': len(aligned_targets) if not aligned_targets.empty else 0
        }
        
        logger.info(f"训练数据处理完成: {result['samples']}个样本")
        
        return result
    
    def process_realtime_signal(self, symbol: str, feature_values: Dict,
                               data_timestamps: Dict[str, datetime],
                               current_price: float) -> Dict:
        """
        处理实时信号（包含延迟门控）
        
        Args:
            symbol: 股票代码
            feature_values: 特征值字典
            data_timestamps: 数据时间戳字典
            current_price: 当前价格
            
        Returns:
            Dict: 信号处理结果
        """
        logger.debug(f"处理{symbol}实时信号")
        
        result = {
            'symbol': symbol,
            'features': feature_values,
            'current_price': current_price,
            'signal_timestamp': datetime.now(),
            'execution_permission': ExecutionPermission.NO_EXECUTION.value,
            'constraints': {},
            'quality_detail': {}
        }
        
        # 延迟门控检查
        if self.config['enable_delay_gating']:
            permission, detail_info = self.delay_gatekeeper.get_execution_permission(
                symbol, data_timestamps
            )
            
            constraints = self.delay_gatekeeper.get_execution_constraints(permission)
            
            result.update({
                'execution_permission': permission.value,
                'constraints': constraints,
                'quality_detail': detail_info
            })
            
            # 如果数据过于陈旧，直接拒绝
            oldest_age = detail_info.get('oldest_data_age_minutes', 0)
            if oldest_age > self.config['max_allowed_delay_minutes']:
                result['execution_blocked'] = True
                result['block_reason'] = f"Data age {oldest_age:.1f}min exceeds limit {self.config['max_allowed_delay_minutes']}min"
                logger.warning(f"{symbol}: {result['block_reason']}")
                
        return result
    
    def integrate_with_ibkr_trader(self, signal_result: Dict) -> Dict:
        """
        与IBKR交易器集成
        
        Args:
            signal_result: process_realtime_signal的输出
            
        Returns:
            Dict: 适配后的执行参数
        """
        execution_params = {
            'symbol': signal_result['symbol'],
            'current_price': signal_result['current_price'],
            'data_quality': signal_result['quality_detail'].get('overall_quality', 'UNKNOWN'),
            'can_execute': signal_result['execution_permission'] != ExecutionPermission.NO_EXECUTION.value
        }
        
        # 根据权限设置执行参数
        if signal_result['execution_permission'] == ExecutionPermission.FULL_EXECUTION.value:
            execution_params.update({
                'order_types': ['MARKET', 'LIMIT'],
                'max_participation': 0.10,
                'urgency': 'NORMAL'
            })
            
        elif signal_result['execution_permission'] == ExecutionPermission.LIMITED_EXECUTION.value:
            execution_params.update({
                'order_types': ['LIMIT'],  # 仅限价
                'max_participation': 0.05,  # 低参与率
                'urgency': 'LOW',
                'additional_constraints': {
                    'max_spread_bps': 20,
                    'require_conservative_pricing': True
                }
            })
            
        else:  # NO_EXECUTION
            execution_params.update({
                'order_types': [],
                'execution_blocked': True,
                'block_reason': signal_result.get('block_reason', 'Data quality insufficient')
            })
            
        return execution_params

# 工厂函数和集成示例
def create_integrated_data_pipeline(config: Optional[Dict] = None) -> IntegratedDataPipeline:
    """创建集成数据管线"""
    return IntegratedDataPipeline(config)

def example_integration_with_existing_code():
    """与现有代码集成示例"""
    
    # 配置
    config = {
        'target_horizon_days': 5,
        'feature_lag_days': 1,
        'enable_delay_gating': True,
        'max_allowed_delay_minutes': 30
    }
    
    # 创建管线
    pipeline = create_integrated_data_pipeline(config)
    
    # 训练阶段使用
    # result = pipeline.process_training_data(adj_close, feature_dict)
    # features = result['features']
    # targets = result['targets']
    
    # 实时信号阶段使用
    # signal_result = pipeline.process_realtime_signal(
    #     symbol="AAPL",
    #     feature_values={'momentum': 0.05, 'value': -0.02},
    #     data_timestamps={'price': datetime.now() - timedelta(minutes=10), 
    #                     'fundamental': datetime.now() - timedelta(hours=1)},
    #     current_price=150.0
    # )
    
    # 与IBKR交易器集成
    # execution_params = pipeline.integrate_with_ibkr_trader(signal_result)
    # 然后传给: trader.plan_and_place_with_rr(**execution_params)
    
    return pipeline

# 辅助函数：快速验证现有数据
def quick_data_audit(adj_close: pd.DataFrame, features: pd.DataFrame) -> Dict:
    """快速数据审计"""
    audit_result = {
        'price_data': {
            'shape': adj_close.shape,
            'date_range': (adj_close.index.min(), adj_close.index.max()),
            'missing_ratio': adj_close.isna().sum().sum() / adj_close.size,
            'zero_prices': (adj_close <= 0).sum().sum()
        },
        'feature_data': {
            'shape': features.shape,
            'missing_ratio': features.isna().sum().sum() / features.size,
            'infinite_values': np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
        },
        'alignment_check': {
            'overlapping_dates': len(adj_close.index.intersection(features.index)),
            'price_frequency': pd.infer_freq(adj_close.index),
            'feature_frequency': pd.infer_freq(features.index)
        }
    }
    
    return audit_result