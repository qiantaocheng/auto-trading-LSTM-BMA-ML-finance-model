#!/usr/bin/env python3
"""
统一量化核心 - 整合所有量化模块功能
替代多个重复的风险管理、数据处理、优化器模块
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from threading import RLock
import threading
from enum import Enum

# 科学计算库
from scipy import optimize
from scipy.stats import norm, spearmanr
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.linalg import inv, cholesky
from sklearn.linear_model import HuberRegressor
from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

# 配置日志
logger = logging.getLogger(__name__)


# =============================================================================
# 核心数据结构
# =============================================================================

class RiskLevel(Enum):
    """风险等级"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class FactorData:
    """因子数据"""
    timestamp: datetime
    ticker: str
    factors: Dict[str, float] = field(default_factory=dict)
    returns: Optional[float] = None
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    is_valid: bool = True


@dataclass
class RiskMetrics:
    """风险指标"""
    volatility: float = 0.0
    var_95: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    beta: float = 1.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0


@dataclass
class PortfolioOptimizationResult:
    """投资组合优化结果"""
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    optimization_status: str
    risk_metrics: RiskMetrics
    constraints_satisfied: bool = True
    optimization_time: float = 0.0


@dataclass
class MarketDataCache:
    """市场数据缓存"""
    prices: Dict[str, pd.DataFrame] = field(default_factory=dict)
    factors: Dict[str, pd.DataFrame] = field(default_factory=dict)
    last_update: Dict[str, datetime] = field(default_factory=dict)
    cache_ttl: int = 300  # 5分钟TTL


# =============================================================================
# 统一量化核心类
# =============================================================================

class UnifiedQuantCore:
    """统一量化核心 - 集成所有量化功能"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("UnifiedQuantCore")
        
        # 线程安全 - 使用单一锁避免死锁
        self._master_lock = RLock()
        
        # 为了向后兼容保留原有属性名，但都指向同一个锁
        self.data_lock = self._master_lock
        self.optimization_lock = self._master_lock
        
        # 锁顺序记录（调试用）
        self._lock_debug = False
        self._lock_owner = None
        
        # 数据存储
        self.market_data_cache = MarketDataCache()
        self.factor_data: Dict[str, FactorData] = {}
        self.risk_models: Dict[str, Any] = {}
        
        # 配置参数
        self.risk_aversion = self.config.get('risk_aversion', 5.0)
        self.max_position = self.config.get('max_position', 0.03)
        self.max_turnover = self.config.get('max_turnover', 0.10)
        self.turnover_penalty = self.config.get('turnover_penalty', 1.0)
        
        # 风险模型配置
        self.factor_groups = {
            'size': ['log_market_cap', 'market_cap_rank'],
            'value': ['pe_ratio', 'pb_ratio', 'ev_ebitda'],
            'momentum': ['return_1m', 'return_3m', 'return_6m'],
            'quality': ['roe', 'roa', 'debt_to_equity'],
            'volatility': ['volatility_20d', 'volatility_60d'],
            'liquidity': ['volume_20d', 'dollar_volume']
        }
        
        # 基准和约束
        self.benchmark_weights: Optional[Dict[str, float]] = None
        self.sector_constraints: Dict[str, float] = {}
        self.position_constraints: Dict[str, Tuple[float, float]] = {}
        
        self.logger.info("统一量化核心初始化完成")
    
    def _acquire_lock_debug(self, operation: str):
        """调试锁获取"""
        if self._lock_debug:
            thread_id = threading.current_thread().ident
            self.logger.debug(f"Thread {thread_id} acquiring lock for: {operation}")
            if self._lock_owner and self._lock_owner != thread_id:
                self.logger.warning(f"Potential lock contention: {thread_id} waiting for {self._lock_owner}")
    
    def _release_lock_debug(self, operation: str):
        """调试锁释放"""
        if self._lock_debug:
            thread_id = threading.current_thread().ident
            self.logger.debug(f"Thread {thread_id} released lock for: {operation}")
    
    def enable_lock_debug(self):
        """启用锁调试"""
        self._lock_debug = True
        self.logger.info("Lock debugging enabled")
    
    def disable_lock_debug(self):
        """禁用锁调试"""
        self._lock_debug = False
        self.logger.info("Lock debugging disabled")
    
    def safe_lock(self, operation: str):
        """安全锁上下文管理器"""
        class SafeLockContext:
            def __init__(self, core, op):
                self.core = core
                self.operation = op
                
            def __enter__(self):
                self.core._acquire_lock_debug(self.operation)
                self.core._master_lock.acquire()
                self.core._lock_owner = threading.current_thread().ident
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.core._lock_owner = None
                self.core._master_lock.release()
                self.core._release_lock_debug(self.operation)
                
        return SafeLockContext(self, operation)
    
    # =============================================================================
    # 数据管理功能 (替代 unified_market_data_manager.py + enhanced_data_processing.py)
    # =============================================================================
    
    def load_market_data(self, tickers: List[str], start_date: str, end_date: str,
                        data_source: str = "polygon") -> Dict[str, pd.DataFrame]:
        """加载市场数据"""
        try:
            with self.safe_lock("load_market_data"):
                # 检查缓存
                cached_data = {}
                missing_tickers = []
                
                for ticker in tickers:
                    if (ticker in self.market_data_cache.prices and
                        ticker in self.market_data_cache.last_update):
                        
                        last_update = self.market_data_cache.last_update[ticker]
                        if (datetime.now() - last_update).seconds < self.market_data_cache.cache_ttl:
                            cached_data[ticker] = self.market_data_cache.prices[ticker]
                        else:
                            missing_tickers.append(ticker)
                    else:
                        missing_tickers.append(ticker)
                
                # 加载缺失数据
                if missing_tickers:
                    self.logger.info(f"加载 {len(missing_tickers)} 只股票的市场数据")
                    
                    if data_source == "polygon":
                        new_data = self._load_polygon_data(missing_tickers, start_date, end_date)
                    else:
                        raise RuntimeError(f"实时交易系统无法获取 {missing_tickers} 的市场数据，请检查数据源连接")
                    
                    # 验证数据真实性
                    simulated_count = 0
                    for ticker, df in new_data.items():
                        if not df.empty and 'is_real_data' in df.columns and not df['is_real_data'].iloc[0]:
                            simulated_count += 1
                    
                    if simulated_count > 0:
                        warning_msg = f"WARNING: {simulated_count}/{len(new_data)} 只股票使用模拟数据!"
                        self.logger.warning(warning_msg)
                        
                        # 检查是否为生产环境
                        if self.config.get('environment') == 'production':
                            self.logger.critical(f"生产环境检测到模拟数据使用: {list(new_data.keys())}")
                    
                    # 更新缓存
                    for ticker, df in new_data.items():
                        self.market_data_cache.prices[ticker] = df
                        self.market_data_cache.last_update[ticker] = datetime.now()
                    
                    cached_data.update(new_data)
                
                self.logger.info(f"成功加载 {len(cached_data)} 只股票的市场数据")
                return cached_data
                
        except Exception as e:
            self.logger.error(f"加载市场数据失败: {e}")
            return {}
    
    def _load_polygon_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """从Polygon加载数据"""
        try:
            from polygon_client import polygon_client, download
            
            data = {}
            for ticker in tickers:
                try:
                    df = download(ticker, start_date, end_date)
                    if not df.empty:
                        data[ticker] = df
                except Exception as e:
                    self.logger.warning(f"加载{ticker}数据失败: {e}")
                    continue
            
            return data
            
        except ImportError:
            self.logger.error("Polygon客户端不可用，实时交易系统无法继续")
            raise RuntimeError("实时交易系统需要Polygon数据源，请检查配置和网络连接")
    
    # _load_fallback_data 函数已移除 - 实时交易系统不应使用模拟数据
    
    def create_factors(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """创建因子数据"""
        try:
            all_factors = []
            
            for ticker, df in market_data.items():
                if df.empty:
                    continue
                
                # 计算因子
                factor_df = self._calculate_factors(df.copy(), ticker)
                all_factors.append(factor_df)
            
            if all_factors:
                combined_factors = pd.concat(all_factors, ignore_index=True)
                
                # 按ticker分组返回
                factor_dict = {}
                for ticker in combined_factors['ticker'].unique():
                    factor_dict[ticker] = combined_factors[combined_factors['ticker'] == ticker].copy()
                
                self.logger.info(f"成功创建 {len(factor_dict)} 只股票的因子数据")
                return factor_dict
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"创建因子数据失败: {e}")
            return {}
    
    def _calculate_factors(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """计算单只股票的因子"""
        try:
            # 确保数据按日期排序
            df = df.sort_index()
            
            # 基础价格因子
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # 动量因子
            df['return_1m'] = df['close'] / df['close'].shift(20) - 1
            df['return_3m'] = df['close'] / df['close'].shift(60) - 1
            df['return_6m'] = df['close'] / df['close'].shift(120) - 1
            
            # 波动率因子
            df['volatility_20d'] = df['log_returns'].rolling(20).std() * np.sqrt(252)
            df['volatility_60d'] = df['log_returns'].rolling(60).std() * np.sqrt(252)
            
            # 技术指标
            df['ma_20'] = df['close'].rolling(20).mean()
            df['ma_50'] = df['close'].rolling(50).mean()
            df['price_to_ma20'] = df['close'] / df['ma_20']
            df['price_to_ma50'] = df['close'] / df['ma_50']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # 成交量因子
            if 'volume' in df.columns:
                df['volume_ma_20'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma_20']
                df['dollar_volume'] = df['close'] * df['volume']
                df['volume_20d'] = df['volume'].rolling(20).mean()
            
            # 添加元数据
            df['ticker'] = ticker
            
            # 重置索引，保留日期列
            df = df.reset_index()
            if 'index' in df.columns:
                df = df.rename(columns={'index': 'date'})
            
            # 清理无效数据
            df = df.dropna(subset=['returns', 'volatility_20d'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"计算{ticker}因子失败: {e}")
            return pd.DataFrame()
    
    # =============================================================================
    # 风险管理功能 (替代所有风险模块)
    # =============================================================================
    
    def estimate_risk_model(self, factor_data: Dict[str, pd.DataFrame],
                          method: str = "barra") -> Dict[str, Any]:
        """估计风险模型"""
        try:
            if method == "barra":
                return self._estimate_barra_model(factor_data)
            elif method == "statistical":
                return self._estimate_statistical_model(factor_data)
            else:
                return self._estimate_simple_model(factor_data)
                
        except Exception as e:
            self.logger.error(f"估计风险模型失败: {e}")
            return {}
    
    def _estimate_barra_model(self, factor_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Barra风格风险模型"""
        try:
            # 合并所有因子数据
            all_data = []
            for ticker, df in factor_data.items():
                if not df.empty:
                    all_data.append(df)
            
            if not all_data:
                return {}
            
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # 选择风险因子
            risk_factors = [
                'return_1m', 'return_3m', 'volatility_20d', 'volatility_60d',
                'price_to_ma20', 'rsi_14', 'volume_ratio'
            ]
            
            # 确保因子存在
            available_factors = [f for f in risk_factors if f in combined_data.columns]
            
            if len(available_factors) < 3:
                self.logger.warning("可用因子数量不足，使用简化模型")
                return self._estimate_simple_model(factor_data)
            
            # 准备回归数据
            valid_data = combined_data.dropna(subset=['returns'] + available_factors)
            
            if len(valid_data) < 100:
                self.logger.warning("有效数据点不足，使用简化模型")
                return self._estimate_simple_model(factor_data)
            
            X = valid_data[available_factors].values
            y = valid_data['returns'].values
            
            # 标准化因子
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Huber回归估计因子载荷
            huber = HuberRegressor(epsilon=1.35, max_iter=1000)
            huber.fit(X_scaled, y)
            
            factor_loadings = pd.DataFrame({
                'factor': available_factors,
                'loading': huber.coef_
            })
            
            # 估计因子协方差矩阵
            factor_returns = pd.DataFrame(X_scaled, columns=available_factors)
            
            # Ledoit-Wolf收缩估计
            lw = LedoitWolf()
            factor_cov = lw.fit(factor_returns).covariance_
            
            # 计算特异风险
            predicted_returns = huber.predict(X_scaled)
            residuals = y - predicted_returns
            
            # 按股票分组计算特异风险
            valid_data = valid_data.copy()
            valid_data['residuals'] = residuals
            specific_risk = valid_data.groupby('ticker')['residuals'].var()
            
            risk_model = {
                'type': 'barra',
                'factor_loadings': factor_loadings,
                'factor_covariance': factor_cov,
                'specific_risk': specific_risk.to_dict(),
                'available_factors': available_factors,
                'scaler': scaler,
                'r_squared': huber.score(X_scaled, y),
                'estimation_date': datetime.now()
            }
            
            self.logger.info(f"Barra风险模型估计完成，R²={risk_model['r_squared']:.3f}")
            return risk_model
            
        except Exception as e:
            self.logger.error(f"Barra模型估计失败: {e}")
            return self._estimate_simple_model(factor_data)
    
    def _estimate_statistical_model(self, factor_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """统计风险模型"""
        try:
            # 构建收益率矩阵
            returns_data = {}
            
            for ticker, df in factor_data.items():
                if not df.empty and 'returns' in df.columns:
                    clean_returns = df['returns'].dropna()
                    if len(clean_returns) >= 20:  # 至少20个观测
                        returns_data[ticker] = clean_returns
            
            if len(returns_data) < 2:
                return {}
            
            # 创建收益率矩阵
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 20:
                return {}
            
            # 使用Ledoit-Wolf收缩估计协方差矩阵
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns_df).covariance_
            
            # 计算相关矩阵
            corr_matrix = pd.DataFrame(cov_matrix, 
                                     index=returns_df.columns, 
                                     columns=returns_df.columns).corr()
            
            # 计算个股波动率
            volatilities = returns_df.std() * np.sqrt(252)
            
            risk_model = {
                'type': 'statistical',
                'covariance_matrix': cov_matrix,
                'correlation_matrix': corr_matrix.values,
                'volatilities': volatilities.to_dict(),
                'tickers': list(returns_df.columns),
                'estimation_window': len(returns_df),
                'estimation_date': datetime.now()
            }
            
            self.logger.info(f"统计风险模型估计完成，覆盖{len(returns_data)}只股票")
            return risk_model
            
        except Exception as e:
            self.logger.error(f"统计模型估计失败: {e}")
            return {}
    
    def _estimate_simple_model(self, factor_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """简化风险模型"""
        try:
            volatilities = {}
            correlations = {}
            
            for ticker, df in factor_data.items():
                if not df.empty and 'returns' in df.columns:
                    returns = df['returns'].dropna()
                    if len(returns) >= 10:
                        vol = returns.std() * np.sqrt(252)
                        volatilities[ticker] = vol
            
            # 假设股票间相关性
            tickers = list(volatilities.keys())
            n = len(tickers)
            
            if n >= 2:
                # 简单相关矩阵（假设0.3的平均相关性）
                corr_matrix = np.full((n, n), 0.3)
                np.fill_diagonal(corr_matrix, 1.0)
                
                correlations = pd.DataFrame(corr_matrix, index=tickers, columns=tickers)
            
            risk_model = {
                'type': 'simple',
                'volatilities': volatilities,
                'correlation_matrix': correlations.values if not correlations.empty else None,
                'tickers': tickers,
                'estimation_date': datetime.now()
            }
            
            self.logger.info(f"简化风险模型创建完成，覆盖{len(volatilities)}只股票")
            return risk_model
            
        except Exception as e:
            self.logger.error(f"简化模型创建失败: {e}")
            return {}
    
    def calculate_portfolio_risk(self, weights: Dict[str, float],
                               risk_model: Dict[str, Any]) -> RiskMetrics:
        """计算投资组合风险"""
        try:
            if risk_model.get('type') == 'barra':
                return self._calculate_barra_risk(weights, risk_model)
            elif risk_model.get('type') == 'statistical':
                return self._calculate_statistical_risk(weights, risk_model)
            else:
                return self._calculate_simple_risk(weights, risk_model)
                
        except Exception as e:
            self.logger.error(f"计算投资组合风险失败: {e}")
            return RiskMetrics()
    
    def _calculate_barra_risk(self, weights: Dict[str, float], risk_model: Dict[str, Any]) -> RiskMetrics:
        """使用Barra模型计算风险"""
        try:
            # 提取模型组件
            factor_cov = risk_model.get('factor_covariance')
            specific_risk = risk_model.get('specific_risk', {})
            
            if factor_cov is None:
                return RiskMetrics()
            
            # 计算投资组合层面的因子暴露
            # 这里简化处理，实际应该根据因子载荷计算
            
            # 估算投资组合波动率
            portfolio_vol = 0.0
            total_weight = sum(abs(w) for w in weights.values())
            
            if total_weight > 0:
                # 简化的风险计算
                weighted_var = 0.0
                for ticker, weight in weights.items():
                    specific_var = specific_risk.get(ticker, 0.01) ** 2
                    weighted_var += (weight ** 2) * specific_var
                
                portfolio_vol = np.sqrt(weighted_var) * np.sqrt(252)
            
            return RiskMetrics(
                volatility=portfolio_vol,
                var_95=portfolio_vol * 1.645,  # 正态分布95%分位数
                expected_shortfall=portfolio_vol * 2.06,  # 近似ES
                beta=1.0,  # 需要基准数据计算
                tracking_error=portfolio_vol * 0.5  # 估算
            )
            
        except Exception as e:
            self.logger.error(f"Barra风险计算失败: {e}")
            return RiskMetrics()
    
    def _calculate_statistical_risk(self, weights: Dict[str, float], risk_model: Dict[str, Any]) -> RiskMetrics:
        """使用统计模型计算风险"""
        try:
            cov_matrix = risk_model.get('covariance_matrix')
            tickers = risk_model.get('tickers', [])
            
            if cov_matrix is None or not tickers:
                return RiskMetrics()
            
            # 构建权重向量
            weight_vector = np.zeros(len(tickers))
            for i, ticker in enumerate(tickers):
                weight_vector[i] = weights.get(ticker, 0.0)
            
            # 计算投资组合方差
            portfolio_variance = np.dot(weight_vector, np.dot(cov_matrix, weight_vector))
            portfolio_vol = np.sqrt(portfolio_variance) * np.sqrt(252)
            
            return RiskMetrics(
                volatility=portfolio_vol,
                var_95=portfolio_vol * 1.645,
                expected_shortfall=portfolio_vol * 2.06,
                beta=1.0,
                tracking_error=portfolio_vol * 0.3
            )
            
        except Exception as e:
            self.logger.error(f"统计风险计算失败: {e}")
            return RiskMetrics()
    
    def _calculate_simple_risk(self, weights: Dict[str, float], risk_model: Dict[str, Any]) -> RiskMetrics:
        """使用简化模型计算风险"""
        try:
            volatilities = risk_model.get('volatilities', {})
            
            # 简化的组合波动率计算
            weighted_vol = 0.0
            total_weight = 0.0
            
            for ticker, weight in weights.items():
                if ticker in volatilities:
                    weighted_vol += abs(weight) * volatilities[ticker]
                    total_weight += abs(weight)
            
            if total_weight > 0:
                avg_vol = weighted_vol / total_weight
                # 考虑分散化效应
                portfolio_vol = avg_vol * np.sqrt(len(weights)) * 0.7  # 分散化折扣
            else:
                portfolio_vol = 0.0
            
            return RiskMetrics(
                volatility=portfolio_vol,
                var_95=portfolio_vol * 1.645,
                expected_shortfall=portfolio_vol * 2.06,
                beta=1.0,
                tracking_error=portfolio_vol * 0.4
            )
            
        except Exception as e:
            self.logger.error(f"简化风险计算失败: {e}")
            return RiskMetrics()
    
    # =============================================================================
    # 投资组合优化功能 (替代所有优化器)
    # =============================================================================
    
    def optimize_portfolio(self, expected_returns: Dict[str, float],
                         risk_model: Dict[str, Any],
                         current_weights: Optional[Dict[str, float]] = None,
                         constraints: Optional[Dict[str, Any]] = None) -> PortfolioOptimizationResult:
        """投资组合优化"""
        
        start_time = datetime.now()
        
        try:
            with self.safe_lock("optimize_portfolio"):
                # 准备数据
                tickers = list(expected_returns.keys())
                n_assets = len(tickers)
                
                if n_assets == 0:
                    return PortfolioOptimizationResult(
                        weights={},
                        expected_return=0.0,
                        expected_risk=0.0,
                        sharpe_ratio=0.0,
                        optimization_status="ERROR: No assets",
                        risk_metrics=RiskMetrics()
                    )
                
                # 构建期望收益向量
                mu = np.array([expected_returns[ticker] for ticker in tickers])
                
                # 构建协方差矩阵
                if risk_model.get('type') == 'statistical':
                    cov_matrix = risk_model.get('covariance_matrix')
                    if cov_matrix is None:
                        cov_matrix = np.eye(n_assets) * 0.01  # 默认协方差
                else:
                    # 简化协方差矩阵构建
                    volatilities = risk_model.get('volatilities', {})
                    cov_matrix = np.eye(n_assets)
                    
                    for i, ticker in enumerate(tickers):
                        vol = volatilities.get(ticker, 0.2)
                        cov_matrix[i, i] = vol ** 2
                    
                    # 添加相关性
                    corr = 0.3  # 假设平均相关性
                    for i in range(n_assets):
                        for j in range(i + 1, n_assets):
                            cov_val = corr * np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])
                            cov_matrix[i, j] = cov_val
                            cov_matrix[j, i] = cov_val
                
                # 处理当前权重
                current_w = np.zeros(n_assets)
                if current_weights:
                    for i, ticker in enumerate(tickers):
                        current_w[i] = current_weights.get(ticker, 0.0)
                
                # 设置约束
                constraints_list = []
                bounds = []
                
                # 权重和约束
                constraints_list.append({
                    'type': 'eq',
                    'fun': lambda x: np.sum(x) - 1.0
                })
                
                # 位置约束
                max_pos = self.max_position
                if constraints and 'max_position' in constraints:
                    max_pos = constraints['max_position']
                
                for i in range(n_assets):
                    bounds.append((-max_pos, max_pos))  # 允许做空
                
                # 换手率约束
                if current_weights and self.max_turnover > 0:
                    def turnover_constraint(x):
                        turnover = np.sum(np.abs(x - current_w))
                        return self.max_turnover - turnover
                    
                    constraints_list.append({
                        'type': 'ineq',
                        'fun': turnover_constraint
                    })
                
                # 目标函数：最大化效用（收益 - 风险惩罚 - 换手率惩罚）
                def objective(x):
                    portfolio_return = np.dot(x, mu)
                    portfolio_risk = np.sqrt(np.dot(x, np.dot(cov_matrix, x)))
                    
                    utility = portfolio_return - 0.5 * self.risk_aversion * (portfolio_risk ** 2)
                    
                    # 换手率惩罚
                    if current_weights and self.turnover_penalty > 0:
                        turnover = np.sum(np.abs(x - current_w))
                        utility -= self.turnover_penalty * turnover
                    
                    return -utility  # 最小化负效用
                
                # 初始解
                x0 = np.ones(n_assets) / n_assets
                if current_weights:
                    x0 = current_w.copy()
                
                # 优化
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints_list,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )
                
                # 处理结果
                if result.success:
                    optimal_weights = result.x
                    
                    # 计算组合指标
                    portfolio_return = np.dot(optimal_weights, mu)
                    portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
                    sharpe_ratio = portfolio_return / max(portfolio_risk, 1e-8)
                    
                    # 构建权重字典
                    weights_dict = {ticker: float(optimal_weights[i]) for i, ticker in enumerate(tickers)}
                    
                    # 计算风险指标
                    risk_metrics = self.calculate_portfolio_risk(weights_dict, risk_model)
                    
                    optimization_time = (datetime.now() - start_time).total_seconds()
                    
                    result_obj = PortfolioOptimizationResult(
                        weights=weights_dict,
                        expected_return=float(portfolio_return),
                        expected_risk=float(portfolio_risk),
                        sharpe_ratio=float(sharpe_ratio),
                        optimization_status="SUCCESS",
                        risk_metrics=risk_metrics,
                        constraints_satisfied=True,
                        optimization_time=optimization_time
                    )
                    
                    self.logger.info(f"投资组合优化成功，夏普比率: {sharpe_ratio:.3f}")
                    return result_obj
                
                else:
                    self.logger.error(f"优化失败: {result.message}")
                    
                    # 返回等权重组合
                    equal_weights = {ticker: 1.0 / n_assets for ticker in tickers}
                    portfolio_return = np.mean(list(expected_returns.values()))
                    portfolio_risk = 0.2  # 估算
                    
                    return PortfolioOptimizationResult(
                        weights=equal_weights,
                        expected_return=portfolio_return,
                        expected_risk=portfolio_risk,
                        sharpe_ratio=portfolio_return / portfolio_risk,
                        optimization_status=f"FALLBACK: {result.message}",
                        risk_metrics=RiskMetrics(volatility=portfolio_risk),
                        constraints_satisfied=False,
                        optimization_time=(datetime.now() - start_time).total_seconds()
                    )
                
        except Exception as e:
            self.logger.error(f"投资组合优化失败: {e}")
            
            # 返回空结果
            return PortfolioOptimizationResult(
                weights={},
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                optimization_status=f"ERROR: {str(e)}",
                risk_metrics=RiskMetrics(),
                constraints_satisfied=False,
                optimization_time=(datetime.now() - start_time).total_seconds()
            )
    
    # =============================================================================
    # 集成接口
    # =============================================================================
    
    def run_full_analysis(self, tickers: List[str], start_date: str, end_date: str,
                         current_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """运行完整分析流程"""
        try:
            self.logger.info(f"开始完整分析，股票数量: {len(tickers)}")
            
            # 1. 加载市场数据
            market_data = self.load_market_data(tickers, start_date, end_date)
            if not market_data:
                return {'error': '无法加载市场数据'}
            
            # 2. 创建因子
            factor_data = self.create_factors(market_data)
            if not factor_data:
                return {'error': '无法创建因子数据'}
            
            # 3. 估计风险模型
            risk_model = self.estimate_risk_model(factor_data, method="barra")
            if not risk_model:
                return {'error': '无法估计风险模型'}
            
            # 4. 计算期望收益（简化版本）
            expected_returns = {}
            for ticker, df in factor_data.items():
                if not df.empty and 'returns' in df.columns:
                    recent_returns = df['returns'].dropna().tail(20)
                    if len(recent_returns) > 0:
                        expected_returns[ticker] = float(recent_returns.mean() * 252)  # 年化
            
            if not expected_returns:
                return {'error': '无法计算期望收益'}
            
            # 5. 优化投资组合
            optimization_result = self.optimize_portfolio(
                expected_returns=expected_returns,
                risk_model=risk_model,
                current_weights=current_weights
            )
            
            # 6. 返回综合结果
            result = {
                'analysis_date': datetime.now().isoformat(),
                'universe_size': len(tickers),
                'data_coverage': len(market_data),
                'factor_coverage': len(factor_data),
                'risk_model': {
                    'type': risk_model.get('type'),
                    'estimation_date': risk_model.get('estimation_date', datetime.now()).isoformat()
                },
                'expected_returns': expected_returns,
                'optimization_result': {
                    'weights': optimization_result.weights,
                    'expected_return': optimization_result.expected_return,
                    'expected_risk': optimization_result.expected_risk,
                    'sharpe_ratio': optimization_result.sharpe_ratio,
                    'status': optimization_result.optimization_status,
                    'optimization_time': optimization_result.optimization_time
                },
                'risk_metrics': {
                    'volatility': optimization_result.risk_metrics.volatility,
                    'var_95': optimization_result.risk_metrics.var_95,
                    'sharpe_ratio': optimization_result.risk_metrics.sharpe_ratio
                }
            }
            
            self.logger.info(f"完整分析完成，夏普比率: {optimization_result.sharpe_ratio:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"完整分析失败: {e}")
            return {'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'core_status': 'RUNNING',
            'cached_data_count': len(self.market_data_cache.prices),
            'factor_data_count': len(self.factor_data),
            'risk_models_count': len(self.risk_models),
            'config': {
                'risk_aversion': self.risk_aversion,
                'max_position': self.max_position,
                'max_turnover': self.max_turnover
            }
        }


# =============================================================================
# 工厂函数
# =============================================================================

def create_unified_quant_core(config: Dict[str, Any] = None) -> UnifiedQuantCore:
    """创建统一量化核心实例"""
    return UnifiedQuantCore(config)


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建量化核心
    core = create_unified_quant_core({
        'risk_aversion': 3.0,
        'max_position': 0.05,
        'max_turnover': 0.15
    })
    
    print("=== 统一量化核心测试 ===")
    
    # 测试股票池
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # 运行完整分析
    result = core.run_full_analysis(
        tickers=tickers,
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    if 'error' in result:
        print(f"分析失败: {result['error']}")
    else:
        print(f"分析成功，覆盖 {result['data_coverage']} 只股票")
        print(f"优化状态: {result['optimization_result']['status']}")
        print(f"组合夏普比率: {result['optimization_result']['sharpe_ratio']:.3f}")
        print(f"预期年化收益: {result['optimization_result']['expected_return']:.1%}")
        print(f"预期年化风险: {result['optimization_result']['expected_risk']:.1%}")
        
        # 显示前5大持仓
        weights = result['optimization_result']['weights']
        top_positions = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        print("前5大持仓:")
        for ticker, weight in top_positions:
            print(f"  {ticker}: {weight:.1%}")
    
    # 系统状态
    status = core.get_status()
    print(f"系统状态: {status}")
    
    print("统一量化核心测试完成")