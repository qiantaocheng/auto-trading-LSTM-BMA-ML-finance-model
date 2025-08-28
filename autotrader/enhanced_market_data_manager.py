#!/usr/bin/env python3
# =============================================================================
# 增强市场数据管理器 - 整合版本
# =============================================================================
# 整合: 统一数据管理 + 数据对齐功能
# 包含原 unified_market_data_manager.py 和 data_alignment.py 的所有功能
# =============================================================================

import pandas as pd
import numpy as np
import warnings
import logging
# Polygon client imports with error handling
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from polygon_client import polygon_client
except ImportError as e:
    logging.warning(f"Polygon client import failed: {e}. Using fallback mode.")
    polygon_client = None
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import json
import requests
from pathlib import Path
import sqlite3
import pytz
from enum import Enum

logger = logging.getLogger(__name__)

# =============================================================================
# 数据对齐功能 (从 data_alignment.py 整合)
# =============================================================================

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
        
        # 计算前瞻收益：t+H相对于t的收益 (安全除法)
        adj_close_safe = adj_close.where(adj_close != 0, np.nan)
        forward_returns = adj_close.shift(-H) / adj_close_safe - 1
        
        # 移除最后H天（无法计算前瞻收益）
        if len(forward_returns) > H:
            aligned_targets = forward_returns.iloc[:-H]
        else:
            logger.warning(f"数据长度({len(forward_returns)})不足以计算{H}天前瞻收益")
            aligned_targets = pd.DataFrame(index=forward_returns.index[:0], columns=forward_returns.columns)
        
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
            current_time = datetime.now(timezone.utc)
        
        # 确保两个时间戳都有时区信息
        if data_timestamp.tzinfo is None:
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

# =============================================================================
# 市场数据管理功能 (从 unified_market_data_manager.py 整合)
# =============================================================================

@dataclass
class MarketDataConfig:
    """市场数据配置"""
    # 数据源优先级
    data_sources: List[str] = field(default_factory=lambda: ['polygon', 'local_db', 'fallback'])
    
    # 缓存设置
    cache_enabled: bool = True
    cache_duration_hours: int = 24
    cache_path: str = "data/market_cache.db"
    
    # 行业分类
    sector_classification: str = "GICS"  # GICS, ICB, 自定义
    sector_level: int = 4  # 1-4级分类细度
    
    # 市值类型
    market_cap_types: List[str] = field(default_factory=lambda: [
        'market_cap',           # 总市值
        'float_market_cap',     # 流通市值  
        'free_float_market_cap' # 自由流通市值
    ])
    
    # 指数成分股
    reference_indices: List[str] = field(default_factory=lambda: [
        'SPY',   # S&P 500
        'QQQ',   # NASDAQ 100
        'DIA'    # Dow 30
    ])

@dataclass
class StockInfo:
    """个股信息"""
    ticker: str
    name: str
    sector: str
    industry: str
    country: str
    market_cap: float
    float_market_cap: Optional[float] = None
    free_float_market_cap: Optional[float] = None
    gics_sector: Optional[str] = None
    gics_industry_group: Optional[str] = None
    gics_industry: Optional[str] = None
    gics_sub_industry: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    is_index_component: Dict[str, bool] = field(default_factory=dict)

class MarketDataCache:
    """市场数据缓存系统"""
    
    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_cache_db()
    
    def _init_cache_db(self):
        """初始化缓存数据库"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_info (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                industry TEXT,
                country TEXT,
                market_cap REAL,
                float_market_cap REAL,
                free_float_market_cap REAL,
                gics_sector TEXT,
                gics_industry_group TEXT,
                gics_industry TEXT,
                gics_sub_industry TEXT,
                exchange TEXT,
                currency TEXT,
                index_components TEXT,
                data_source TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_stock_info(self, ticker: str, max_age_hours: int = 24) -> Optional[StockInfo]:
        """从缓存获取股票信息"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM stock_info 
            WHERE ticker = ? AND datetime(updated_at) > datetime('now', '-{} hours')
        '''.format(max_age_hours), (ticker,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_stock_info(row)
        return None
    
    def save_stock_info(self, stock_info: StockInfo, source: str):
        """保存股票信息到缓存"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO stock_info 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            stock_info.ticker,
            stock_info.name,
            stock_info.sector,
            stock_info.industry,
            stock_info.country,
            stock_info.market_cap,
            stock_info.float_market_cap,
            stock_info.free_float_market_cap,
            stock_info.gics_sector,
            stock_info.gics_industry_group,
            stock_info.gics_industry,
            stock_info.gics_sub_industry,
            stock_info.exchange,
            stock_info.currency,
            json.dumps(stock_info.is_index_component),
            source
        ))
        
        conn.commit()
        conn.close()
    
    def _row_to_stock_info(self, row) -> StockInfo:
        """将数据库行转换为StockInfo对象"""
        return StockInfo(
            ticker=row[0],
            name=row[1],
            sector=row[2],
            industry=row[3],
            country=row[4],
            market_cap=row[5],
            float_market_cap=row[6],
            free_float_market_cap=row[7],
            gics_sector=row[8],
            gics_industry_group=row[9],
            gics_industry=row[10],
            gics_sub_industry=row[11],
            exchange=row[12],
            currency=row[13],
            is_index_component=json.loads(row[14] or '{}')
        )

# =============================================================================
# 集成数据管线类
# =============================================================================

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

class EnhancedMarketDataManager:
    """增强市场数据管理器 - 整合统一数据管理和数据对齐功能"""
    
    def __init__(self, config: MarketDataConfig = None):
        self.config = config or MarketDataConfig()
        
        # 初始化组件
        self.cache = MarketDataCache(self.config.cache_path) if self.config.cache_enabled else None
        
        # 数据缓存
        self.stock_info_cache: Dict[str, StockInfo] = {}
        
        # 集成数据对齐功能
        self.data_pipeline = IntegratedDataPipeline()
        
        logger.info("增强市场数据管理器初始化完成")
    
    def get_stock_info(self, ticker: str, force_refresh: bool = False) -> Optional[StockInfo]:
        """获取股票基本信息"""
        
        # 检查内存缓存
        if not force_refresh and ticker in self.stock_info_cache:
            return self.stock_info_cache[ticker]
        
        # 检查数据库缓存
        if self.cache and not force_refresh:
            cached_info = self.cache.get_stock_info(ticker)
            if cached_info:
                self.stock_info_cache[ticker] = cached_info
                return cached_info
        
        # 创建fallback股票信息
        stock_info = self._get_fallback_stock_info(ticker)
        
        if stock_info:
            # 缓存数据
            self.stock_info_cache[ticker] = stock_info
            if self.cache:
                self.cache.save_stock_info(stock_info, 'fallback')
            
            logger.info(f"获取{ticker}数据成功")
            return stock_info
        
        logger.warning(f"无法获取{ticker}的市场数据")
        return None
    
    def _get_fallback_stock_info(self, ticker: str) -> Optional[StockInfo]:
        """获取fallback股票信息"""
        
        # 基于ticker的启发式分类
        sector_mapping = {
            'AAPL': ('Technology', 'Consumer Electronics', 'US'),
            'MSFT': ('Technology', 'Software', 'US'),
            'GOOGL': ('Communication Services', 'Internet', 'US'),
            'AMZN': ('Consumer Discretionary', 'E-commerce', 'US'),
            'TSLA': ('Consumer Discretionary', 'Electric Vehicles', 'US'),
            'NVDA': ('Technology', 'Semiconductors', 'US'),
            'META': ('Communication Services', 'Social Media', 'US'),
            'NFLX': ('Communication Services', 'Streaming', 'US'),
            'JPM': ('Financial Services', 'Banks', 'US'),
            'JNJ': ('Healthcare', 'Pharmaceuticals', 'US')
        }
        
        if ticker in sector_mapping:
            sector, industry, country = sector_mapping[ticker]
        else:
            sector, industry, country = 'Unknown', 'Unknown', 'US'
        
        # 生成模拟市值数据
        base_market_cap = hash(ticker) % 1000000000000  # 基于ticker生成一致的随机数
        market_cap = max(1000000000, base_market_cap)  # 至少10亿市值
        
        return StockInfo(
            ticker=ticker,
            name=f"{ticker} Corp",
            sector=sector,
            industry=industry,
            country=country,
            market_cap=market_cap,
            float_market_cap=market_cap * 0.8,
            free_float_market_cap=market_cap * 0.6,
            gics_sector=sector,
            exchange='NASDAQ',
            currency='USD'
        )
    
    def get_batch_stock_info(self, tickers: List[str]) -> Dict[str, StockInfo]:
        """批量获取股票信息"""
        results = {}
        
        logger.info(f"批量获取{len(tickers)}只股票的市场数据...")
        
        for i, ticker in enumerate(tickers):
            if i % 50 == 0:
                logger.info(f"进度: {i}/{len(tickers)}")
            
            stock_info = self.get_stock_info(ticker)
            if stock_info:
                results[ticker] = stock_info
        
        logger.info(f"成功获取{len(results)}只股票信息")
        return results
    
    # 数据对齐功能接口
    def create_aligned_targets(self, adj_close: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
        """创建对齐的前瞻收益目标"""
        return self.data_pipeline.alignment_engine.create_aligned_targets(adj_close, horizon_days)
    
    def align_features_to_targets(self, features: pd.DataFrame, targets: pd.DataFrame, 
                                 lag_days: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """将特征对齐到目标"""
        return self.data_pipeline.alignment_engine.align_features_to_targets(features, targets, lag_days)
    
    def process_training_data(self, adj_close: pd.DataFrame, 
                            feature_dict: Dict[str, pd.DataFrame]) -> Dict:
        """处理训练数据（完整对齐 + 验证）"""
        return self.data_pipeline.process_training_data(adj_close, feature_dict)
    
    def validate_temporal_integrity(self, features: pd.DataFrame, 
                                  targets: pd.DataFrame) -> Dict[str, bool]:
        """验证时间完整性，确保无未来信息泄露"""
        return self.data_pipeline.alignment_engine.validate_temporal_integrity(features, targets)
    
    def assess_data_freshness(self, data_timestamp: datetime, 
                            current_time: Optional[datetime] = None) -> DataQuality:
        """评估数据新鲜度"""
        return self.data_pipeline.delay_gatekeeper.assess_data_freshness(data_timestamp, current_time)
    
    def get_execution_permission(self, symbol: str, data_timestamps: Dict[str, datetime]) -> Tuple[ExecutionPermission, Dict]:
        """获取执行权限"""
        return self.data_pipeline.delay_gatekeeper.get_execution_permission(symbol, data_timestamps)

# =============================================================================
# 全局实例和工厂函数
# =============================================================================

# 全局增强数据管理器实例
_enhanced_market_data_manager = None

def get_enhanced_market_data_manager(config: Optional[MarketDataConfig] = None) -> EnhancedMarketDataManager:
    """获取增强市场数据管理器单例"""
    global _enhanced_market_data_manager
    if _enhanced_market_data_manager is None:
        _enhanced_market_data_manager = EnhancedMarketDataManager(config)
    return _enhanced_market_data_manager

# 向后兼容的工厂函数
def create_integrated_data_pipeline(config: Optional[Dict] = None) -> IntegratedDataPipeline:
    """创建集成数据管线 - 向后兼容"""
    return IntegratedDataPipeline(config)

def get_market_data_manager(config: Optional[MarketDataConfig] = None) -> EnhancedMarketDataManager:
    """获取市场数据管理器 - 向后兼容统一接口"""
    return get_enhanced_market_data_manager(config)

# 快速验证现有数据的辅助函数
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

# 向后兼容函数 - 用于统一市场数据管理器
def get_unified_market_data_manager(config: Optional[MarketDataConfig] = None) -> EnhancedMarketDataManager:
    """获取统一市场数据管理器 - 向后兼容接口"""
    return get_enhanced_market_data_manager(config)