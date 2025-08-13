#!/usr/bin/env python3
"""
高级投资组合优化模块
实现期望收益-风险联合优化、换手率控制、风险预算等高级技术
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.linalg import inv, cholesky
from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedPortfolioOptimizer:
    """高级投资组合优化器"""
    
    def __init__(self,
                 risk_aversion: float = 5.0,
                 turnover_penalty: float = 1.0,
                 max_turnover: float = 0.10,
                 max_position: float = 0.03,
                 max_sector_exposure: float = 0.15,
                 max_country_exposure: float = 0.20,
                 min_liquidity_rank: float = 0.3,
                 covariance_method: str = "ledoit_wolf"):
        """
        初始化投资组合优化器
        
        Args:
            risk_aversion: 风险厌恶系数
            turnover_penalty: 换手率惩罚系数
            max_turnover: 最大换手率
            max_position: 单个持仓上限
            max_sector_exposure: 行业敞口上限
            max_country_exposure: 国家敞口上限
            min_liquidity_rank: 最小流动性排名
            covariance_method: 协方差估计方法
        """
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty
        self.max_turnover = max_turnover
        self.max_position = max_position
        self.max_sector_exposure = max_sector_exposure
        self.max_country_exposure = max_country_exposure
        self.min_liquidity_rank = min_liquidity_rank
        self.covariance_method = covariance_method
        
        # 历史记录
        self.optimization_history = []
        self.performance_history = []
        
        # 风险模型
        self.risk_model = None
        self.factor_exposures = None
        
        logger.info(f"投资组合优化器初始化完成，风险厌恶: {risk_aversion}")
    
    def _regularize_covariance_matrix(self, cov_matrix: np.ndarray, 
                                     min_eigenvalue: float = 1e-6) -> np.ndarray:
        """
        正则化协方差矩阵以确保数值稳定性
        
        Args:
            cov_matrix: 原始协方差矩阵
            min_eigenvalue: 最小特征值
            
        Returns:
            正则化后的协方差矩阵
        """
        try:
            # 确保对称性
            cov_symmetric = (cov_matrix + cov_matrix.T) / 2
            
            # 特征值分解
            eigenvals, eigenvecs = np.linalg.eigh(cov_symmetric)
            
            # 处理负特征值和过小特征值
            eigenvals_reg = np.maximum(eigenvals, min_eigenvalue)
            
            # 重构矩阵
            cov_regularized = eigenvecs @ np.diag(eigenvals_reg) @ eigenvecs.T
            
            # 添加对角正则化
            reg_factor = np.trace(cov_regularized) / len(cov_regularized) * 1e-6
            cov_regularized += np.eye(len(cov_regularized)) * reg_factor
            
            logger.info(f"协方差矩阵正则化完成，条件数: {np.linalg.cond(cov_regularized):.2f}")
            
            return cov_regularized
            
        except Exception as e:
            logger.warning(f"协方差矩阵正则化失败: {e}，使用对角正则化")
            # 回退到简单的对角正则化
            diagonal_reg = np.eye(len(cov_matrix)) * np.trace(cov_matrix) / len(cov_matrix) * 0.01
            return cov_matrix + diagonal_reg
    
    def estimate_covariance_matrix(self, returns: pd.DataFrame, 
                                  factor_returns: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        估计协方差矩阵
        
        Args:
            returns: 股票收益率矩阵
            factor_returns: 因子收益率矩阵（可选）
            
        Returns:
            协方差矩阵
        """
        logger.info(f"估计协方差矩阵，方法: {self.covariance_method}")
        
        # 去除缺失值
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 10:
            logger.warning("历史数据不足，使用单位矩阵")
            n_assets = len(returns.columns)
            return np.eye(n_assets) * 0.01
        
        try:
            if self.covariance_method == "ledoit_wolf":
                # Ledoit-Wolf收缩估计
                lw = LedoitWolf()
                cov_arr = lw.fit(returns_clean).covariance_
                
            elif self.covariance_method == "oas":
                # Oracle Approximating Shrinkage
                oas = OAS()
                cov_arr = oas.fit(returns_clean).covariance_
                
            elif self.covariance_method == "empirical":
                # 经验协方差
                emp = EmpiricalCovariance()
                cov_arr = emp.fit(returns_clean).covariance_
                
            elif self.covariance_method == "factor_model":
                # 因子模型协方差
                if factor_returns is not None:
                    cov_arr = self._estimate_factor_covariance(returns_clean, factor_returns)
                else:
                    # 降级到Ledoit-Wolf
                    lw = LedoitWolf()
                    cov_arr = lw.fit(returns_clean).covariance_
                    
            elif self.covariance_method == "robust":
                # 稳健协方差估计
                cov_arr = self._estimate_robust_covariance(returns_clean)
                
            else:
                # 默认使用样本协方差
                cov_arr = returns_clean.cov().values
            
            # 确保协方差矩阵正定
            cov_arr = self._ensure_positive_definite(cov_arr)
            cov_matrix = pd.DataFrame(cov_arr, index=returns_clean.columns, columns=returns_clean.columns)
            
            logger.info(f"协方差矩阵估计完成，条件数: {np.linalg.cond(cov_matrix):.2f}")
            
            return cov_matrix
            
        except Exception as e:
            logger.error(f"协方差估计失败: {e}")
            n_assets = len(returns.columns)
            eye_arr = np.eye(n_assets) * 0.01
            return pd.DataFrame(eye_arr, index=returns.columns, columns=returns.columns)
    
    def _estimate_factor_covariance(self, returns: pd.DataFrame, 
                                   factor_returns: pd.DataFrame) -> np.ndarray:
        """因子模型协方差估计"""
        logger.info("使用因子模型估计协方差")
        
        # 确保时间对齐
        common_dates = returns.index.intersection(factor_returns.index)
        if len(common_dates) < 10:
            raise ValueError("因子数据和收益数据时间对齐不足")
        
        returns_aligned = returns.loc[common_dates]
        factors_aligned = factor_returns.loc[common_dates]
        
        n_assets = len(returns.columns)
        n_factors = len(factor_returns.columns)
        
        # 因子暴露矩阵
        factor_exposures = np.zeros((n_assets, n_factors))
        specific_risks = np.zeros(n_assets)
        
        for i, asset in enumerate(returns.columns):
            try:
                # 对每个资产进行因子回归
                y = returns_aligned[asset].dropna()
                X = factors_aligned.loc[y.index]
                
                if len(y) < max(10, n_factors + 1):
                    continue
                
                # 添加常数项
                X_with_const = np.column_stack([np.ones(len(X)), X.values])
                
                # 线性回归
                beta = np.linalg.lstsq(X_with_const, y.values, rcond=None)[0]
                factor_exposures[i, :] = beta[1:]  # 排除常数项
                
                # 计算残差方差（特异风险）
                predicted = X_with_const @ beta
                residuals = y.values - predicted
                specific_risks[i] = np.var(residuals, ddof=n_factors+1)
                
            except Exception as e:
                logger.warning(f"资产{asset}因子回归失败: {e}")
                specific_risks[i] = 0.01  # 默认特异风险
        
        # 因子协方差矩阵
        factor_cov = factors_aligned.cov().values
        
        # 总协方差 = B * F * B' + D
        # B: 因子暴露矩阵, F: 因子协方差, D: 特异风险对角矩阵
        factor_component = factor_exposures @ factor_cov @ factor_exposures.T
        specific_component = np.diag(specific_risks)
        
        total_cov = factor_component + specific_component
        
        # 保存因子暴露（用于风险归因）
        self.factor_exposures = pd.DataFrame(
            factor_exposures, 
            index=returns.columns, 
            columns=factor_returns.columns
        )
        
        return total_cov
    
    def _estimate_robust_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """稳健协方差估计"""
        logger.info("使用稳健方法估计协方差")
        
        # 使用Winsorized returns减少极值影响
        def winsorize(series, lower=0.05, upper=0.95):
            lower_val = series.quantile(lower)
            upper_val = series.quantile(upper)
            return series.clip(lower_val, upper_val)
        
        returns_winsorized = returns.apply(winsorize)
        
        # 使用MAD (Median Absolute Deviation) 进行标准化
        def mad_standardize(series):
            median = series.median()
            mad = (series - median).abs().median()
            return (series - median) / (mad * 1.4826)  # 1.4826使MAD等价于正态分布的标准差
        
        returns_robust = returns_winsorized.apply(mad_standardize)
        
        # 计算Spearman相关矩阵
        corr_matrix = returns_robust.corr(method='spearman')
        
        # 估计波动率（使用MAD）
        volatilities = returns_winsorized.apply(lambda x: x.std())
        
        # 构建协方差矩阵
        vol_matrix = np.outer(volatilities, volatilities)
        cov_matrix = corr_matrix.values * vol_matrix
        
        return cov_matrix
    
    def _ensure_positive_definite(self, cov_matrix: np.ndarray, 
                                 min_eigenvalue: float = 1e-8) -> np.ndarray:
        """确保协方差矩阵正定"""
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # 将负特征值和过小特征值调整为最小值
        eigenvals = np.maximum(eigenvals, min_eigenvalue)
        
        # 重构矩阵
        cov_matrix_pd = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return cov_matrix_pd

    def _project_to_capped_simplex(self, weights: np.ndarray, cap: float, target_sum: float = 1.0,
                                   max_iter: int = 100, tol: float = 1e-9) -> np.ndarray:
        """将权重投影到 {w_i in [0, cap], sum w_i = target_sum} 的集合。
        使用对阈值 tau 的二分搜索：w' = clip(w - tau, 0, cap)，使 sum w' = target_sum。
        """
        n = len(weights)
        if n == 0:
            return weights
        # 初始区间：tau 低界与高界
        lo = -cap
        hi = cap
        for _ in range(max_iter):
            tau = (lo + hi) / 2.0
            w = np.clip(weights - tau, 0.0, cap)
            s = w.sum()
            if abs(s - target_sum) < tol:
                return w
            if s > target_sum:
                lo = tau
            else:
                hi = tau
        # 兜底返回最后一次
        return np.clip(weights - (lo + hi) / 2.0, 0.0, cap)
    
    def optimize_portfolio(self, 
                          expected_returns: pd.Series,
                          covariance_matrix: np.ndarray,
                          current_weights: Optional[pd.Series] = None,
                          universe_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        投资组合优化主函数
        
        Args:
            expected_returns: 预期收益率
            covariance_matrix: 协方差矩阵
            current_weights: 当前权重
            universe_data: 股票池数据（包含行业、国家等信息）
            
        Returns:
            优化结果字典
        """
        logger.info("开始投资组合优化")
        
        n_assets = len(expected_returns)
        asset_names = expected_returns.index.tolist()
        
        # 初始权重
        if current_weights is None:
            current_weights = pd.Series(0.0, index=asset_names)
        else:
            current_weights = current_weights.reindex(asset_names, fill_value=0.0)
        
        # 数值稳定性：规范化输入
        mu = expected_returns.values
        sigma = covariance_matrix
        
        # 确保协方差矩阵正定
        sigma_regularized = self._regularize_covariance_matrix(sigma)
        
        # 缩放预期收益以提高数值稳定性
        mu_scaled = mu / np.std(mu) if np.std(mu) > 1e-8 else mu
        
        # 目标函数：最大化效用 U = w'μ - λ/2 * w'Σw - η * ||w - w_prev||²
        def objective(weights):
            weights = np.asarray(weights).flatten()
            w_prev = current_weights.values
            
            # 预期收益
            portfolio_return = np.dot(mu_scaled, weights)
            
            # 风险（二次型）
            portfolio_risk = 0.5 * np.dot(weights, np.dot(sigma_regularized, weights))
            
            # 换手率惩罚（使用L2范数，更平滑）
            turnover_penalty = 0.5 * self.turnover_penalty * np.sum((weights - w_prev) ** 2)
            
            # 返回负效用（因为minimize）
            utility = portfolio_return - self.risk_aversion * portfolio_risk - turnover_penalty
            return -utility
        
        # 目标函数的梯度
        def objective_gradient(weights):
            weights = np.asarray(weights).flatten()
            w_prev = current_weights.values
            
            # 收益梯度
            return_grad = mu_scaled
            
            # 风险梯度
            risk_grad = self.risk_aversion * np.dot(sigma_regularized, weights)
            
            # 换手率梯度（L2）
            turnover_grad = self.turnover_penalty * (weights - w_prev)
            
            return -(return_grad - risk_grad - turnover_grad)
        
        # 约束条件
        constraints = []
        
        # 1. 权重和为1（允许小幅偏差）
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0,
            'jac': lambda w: np.ones(n_assets)
        })
        
        # 2. 换手率约束
        if self.max_turnover > 0:
            def turnover_constraint(weights):
                return self.max_turnover - np.sum(np.abs(weights - current_weights.values))
            
            constraints.append({
                'type': 'ineq',
                'fun': turnover_constraint
            })
        
        # 3. 行业暴露约束
        if universe_data is not None and 'SECTOR' in universe_data.columns:
            sector_constraints = self._create_sector_constraints(universe_data, asset_names)
            constraints.extend(sector_constraints)
        
        # 4. 国家暴露约束
        if universe_data is not None and 'COUNTRY' in universe_data.columns:
            country_constraints = self._create_country_constraints(universe_data, asset_names)
            constraints.extend(country_constraints)
        
        # 5. 流动性约束
        if universe_data is not None and 'liquidity_rank' in universe_data.columns:
            liquidity_constraint = self._create_liquidity_constraint(universe_data, asset_names)
            if liquidity_constraint:
                constraints.append(liquidity_constraint)
        
        # 变量边界：单个持仓限制（做多策略）
        bounds = Bounds(
            lb=np.full(n_assets, 0.0),  # 不允许做空
            ub=np.full(n_assets, self.max_position)
        )
        
        # 智能初始猜测
        def generate_initial_guess():
            # 方法1: 基于当前权重
            if np.sum(current_weights.values) > 0.01:
                return np.maximum(current_weights.values, 0.0)
            
            # 方法2: 基于预期收益的风险调整权重
            positive_returns = np.maximum(mu_scaled, 0)
            if np.sum(positive_returns) > 1e-8:
                # 简单的风险调整：收益/波动率的权重
                vol_adj_returns = positive_returns / (np.diag(sigma_regularized) ** 0.5 + 1e-8)
                weights = vol_adj_returns / np.sum(vol_adj_returns)
                return np.minimum(weights, self.max_position)
            
            # 方法3: 等权重
            return np.full(n_assets, 1.0 / n_assets)
        
        x0 = generate_initial_guess()
        
        # 确保初始猜测满足约束
        x0 = np.maximum(x0, 0.0)
        x0 = np.minimum(x0, self.max_position)
        x0 = x0 / np.sum(x0)  # 标准化
        
        # 执行优化
        try:
            # 固定一种优化器：SLSQP 宽容差，较低 maxiter
            # 在此之前做 top-K 裁剪以减少维度、提升稳定性
            top_k = min(20, n_assets)
            if n_assets > top_k:
                top_idx = np.argsort(-mu)[:top_k]
                asset_names = [asset_names[i] for i in top_idx]
                mu = mu[top_idx]
                mu_scaled = mu_scaled[top_idx]
                sigma_regularized = sigma_regularized[np.ix_(top_idx, top_idx)]
                current_weights = current_weights.reindex(asset_names, fill_value=0.0)
                bounds = Bounds(lb=np.full(top_k, 0.0), ub=np.full(top_k, self.max_position))
                n_assets = top_k
            optimization_configs = [
                {
                    'method': 'SLSQP',
                    'options': {
                        'ftol': 1e-2,    # 降低收敛要求
                        'eps': 1e-4,     # 降低梯度精度要求
                        'disp': False,
                        'maxiter': 200   # 降低最大迭代次数
                    }
                }
            ]
            
            result = None
            best_result = None
            best_objective_value = np.inf
            
            for i, config in enumerate(optimization_configs):
                try:
                    logger.info(f"尝试优化配置 {i+1}/{len(optimization_configs)}: {config['method']}")
                    
                    # 根据方法选择是否使用梯度
                    use_gradient = config['method'] in ['SLSQP', 'L-BFGS-B']
                    
                    current_result = minimize(
                        fun=objective,
                        x0=x0.copy(),
                        jac=objective_gradient if use_gradient else None,
                        bounds=bounds,
                        constraints=constraints,
                        **config
                    )
                    
                    # 评估结果质量
                    if hasattr(current_result, 'fun') and current_result.fun < best_objective_value:
                        best_result = current_result
                        best_objective_value = current_result.fun
                    
                    if current_result.success:
                        result = current_result
                        logger.info(f"优化成功，方法: {config['method']}, 目标值: {current_result.fun:.6f}")
                        break
                    else:
                        logger.warning(f"配置{i+1}失败: {current_result.message}")
                        
                except Exception as e:
                    logger.warning(f"配置{i+1}异常: {e}")
                    continue
            
            # 如果没有成功的结果，使用最好的失败结果
            if result is None:
                if best_result is not None:
                    logger.warning(f"所有配置都未成功，使用最佳结果: {best_objective_value:.6f}")
                    result = best_result
                else:
                    logger.error("所有优化配置都失败")
                    result = type('MockResult', (), {
                        'success': False, 
                        'message': 'All optimization configurations failed',
                        'x': x0,
                        'fun': objective(x0)
                    })()
            
            # 处理优化结果（无论是否success）
            optimal_weights = pd.Series(result.x, index=asset_names)
            
            # 后处理：确保权重合理
            # 1) 小权重置零（提高稀疏性）
            weight_threshold = 0.001
            optimal_weights[optimal_weights < weight_threshold] = 0.0

            # 2) 如果求解失败或结果不佳，使用启发式构造一个多样化的解
            if (not result.success) or optimal_weights.sum() == 0 or np.isclose(optimal_weights.std(), 0):
                logger.warning("优化器未返回可用解，使用启发式替代方案")
                
                # 基于信号排序的分层权重分配
                signal_ranks = np.argsort(-mu_scaled)  # 信号从高到低排序
                n_assets_to_use = min(len(asset_names), max(5, int(len(asset_names) * 0.8)))
                
                # 创建分层权重：前20%获得更高权重
                w0 = np.zeros(len(asset_names))
                for i, rank_idx in enumerate(signal_ranks[:n_assets_to_use]):
                    if i < n_assets_to_use // 5:  # 前20%
                        w0[rank_idx] = 0.12  # 高权重
                    elif i < n_assets_to_use // 2:  # 中间30%
                        w0[rank_idx] = 0.08  # 中等权重
                    else:  # 其余50%
                        w0[rank_idx] = 0.04  # 低权重
                
                # 确保不超过max_position并归一化
                w0 = np.minimum(w0, self.max_position)
                if w0.sum() > 0:
                    w0 = w0 / w0.sum()
                else:
                    # 兜底：等权重配置
                    w0 = np.full(len(asset_names), 1.0 / len(asset_names))
                    w0 = np.minimum(w0, self.max_position)
                    w0 = w0 / w0.sum()
                
                optimal_weights = pd.Series(w0, index=asset_names)
            else:
                # 3) 正式解：投影到 {0<=w<=cap, sum w = 1}
                w_vec = optimal_weights.values
                w_proj = self._project_to_capped_simplex(w_vec, cap=self.max_position, target_sum=1.0)
                optimal_weights = pd.Series(w_proj, index=asset_names)
            
            # 计算优化后的组合特征
            portfolio_metrics = self._calculate_portfolio_metrics(
                optimal_weights, expected_returns, covariance_matrix, current_weights
            )
            
            # 构造结果
            is_successful = result.success or (best_result is not None and hasattr(result, 'fun'))
            optimization_result = {
                'success': is_successful,
                'optimal_weights': optimal_weights,
                'portfolio_metrics': portfolio_metrics,
                'optimization_info': {
                    'objective_value': -result.fun if hasattr(result, 'fun') else None,
                    'iterations': getattr(result, 'nit', None),
                    'function_evaluations': getattr(result, 'nfev', None),
                    'optimization_message': getattr(result, 'message', 'Unknown'),
                    'method_used': 'Multiple attempts' if best_result else 'Initial guess'
                }
            }
            
            if is_successful:
                logger.info(f"优化完成，目标函数值: {-result.fun:.6f}, 活跃持仓: {(optimal_weights > 0.001).sum()}")
            else:
                logger.warning(f"优化未达到最优，使用当前最佳结果: {result.message}")
                optimization_result['error_message'] = result.message
        
        except Exception as e:
            logger.error(f"优化过程异常: {e}")
            optimization_result = {
                'success': False,
                'optimal_weights': current_weights,
                'error_message': str(e)
            }
        
        # 记录优化历史
        self.optimization_history.append(optimization_result)
        
        return optimization_result
    
    def _create_sector_constraints(self, universe_data: pd.DataFrame, 
                                  asset_names: List[str]) -> List[Dict]:
        """创建行业暴露约束"""
        constraints = []
        
        # 对齐数据
        universe_aligned = universe_data.reindex(asset_names, fill_value='Unknown')
        
        # 获取所有行业
        sectors = universe_aligned['SECTOR'].unique()
        
        for sector in sectors:
            if sector == 'Unknown':
                continue
                
            sector_mask = (universe_aligned['SECTOR'] == sector).values.astype(float)
            
            # 行业暴露不超过限制
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, mask=sector_mask: self.max_sector_exposure - np.abs(np.sum(w * mask)),
                'jac': lambda w, mask=sector_mask: -np.sign(np.sum(w * mask)) * mask
            })
        
        return constraints
    
    def _create_country_constraints(self, universe_data: pd.DataFrame, 
                                   asset_names: List[str]) -> List[Dict]:
        """创建国家暴露约束"""
        constraints = []
        
        universe_aligned = universe_data.reindex(asset_names, fill_value='Unknown')
        countries = universe_aligned['COUNTRY'].unique()
        
        for country in countries:
            if country == 'Unknown':
                continue
                
            country_mask = (universe_aligned['COUNTRY'] == country).values.astype(float)
            
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, mask=country_mask: self.max_country_exposure - np.abs(np.sum(w * mask)),
                'jac': lambda w, mask=country_mask: -np.sign(np.sum(w * mask)) * mask
            })
        
        return constraints
    
    def _create_liquidity_constraint(self, universe_data: pd.DataFrame, 
                                    asset_names: List[str]) -> Optional[Dict]:
        """创建流动性约束"""
        universe_aligned = universe_data.reindex(asset_names, fill_value=0.0)
        
        if 'liquidity_rank' not in universe_aligned.columns:
            return None
        
        liquidity_ranks = universe_aligned['liquidity_rank'].values
        
        # 低流动性股票的权重限制更严格
        def liquidity_constraint_func(weights):
            low_liquidity_mask = liquidity_ranks < self.min_liquidity_rank
            low_liquidity_exposure = np.sum(np.abs(weights[low_liquidity_mask]))
            # 低流动性股票总暴露不超过5%
            return 0.05 - low_liquidity_exposure
        
        return {
            'type': 'ineq',
            'fun': liquidity_constraint_func
        }
    
    def _calculate_portfolio_metrics(self, weights: pd.Series, 
                                    expected_returns: pd.Series,
                                    covariance_matrix: np.ndarray,
                                    current_weights: pd.Series) -> Dict[str, float]:
        """计算投资组合指标"""
        w = weights.values
        
        metrics = {}
        
        # 预期收益和风险
        metrics['expected_return'] = expected_returns.values @ w
        metrics['expected_volatility'] = np.sqrt(w.T @ covariance_matrix @ w)
        metrics['sharpe_ratio'] = metrics['expected_return'] / (metrics['expected_volatility'] + 1e-12)
        
        # 换手率
        metrics['turnover'] = np.sum(np.abs(w - current_weights.values))
        
        # 集中度指标
        metrics['concentration_hhi'] = np.sum(w**2)  # HHI指数
        metrics['effective_positions'] = 1.0 / metrics['concentration_hhi']
        metrics['max_weight'] = np.max(np.abs(w))
        
        # 多空比例
        long_weights = w[w > 0]
        short_weights = w[w < 0]
        metrics['long_exposure'] = np.sum(long_weights)
        metrics['short_exposure'] = np.sum(np.abs(short_weights))
        metrics['net_exposure'] = metrics['long_exposure'] - metrics['short_exposure']
        metrics['gross_exposure'] = metrics['long_exposure'] + metrics['short_exposure']
        
        # 有效持仓数
        active_positions = np.sum(np.abs(w) > 0.001)
        metrics['active_positions'] = active_positions
        
        return metrics
    
    def risk_attribution(self, weights: pd.Series, 
                         covariance_matrix) -> Dict[str, Any]:
        """风险归因分析"""
        logger.info("进行风险归因分析")
        
        # 支持DataFrame或ndarray协方差
        if hasattr(covariance_matrix, 'values'):
            sigma = covariance_matrix.values
        else:
            sigma = np.asarray(covariance_matrix)
        w = weights.values.reshape(-1, 1)
        
        # 总风险
        total_var = float((w.T @ sigma @ w).squeeze())
        total_risk = float(np.sqrt(max(total_var, 0.0)))
        
        # 边际风险贡献 (MCTR)
        denom = total_risk if total_risk > 1e-12 else 1e-12
        mctr = (sigma @ w).flatten() / denom
        
        # 组件风险贡献 (CTR)
        ctr = weights.values * mctr
        
        # 百分比风险贡献
        pct_ctr = ctr / (total_var if total_var > 1e-12 else 1e-12) * 100
        
        attribution_result = {
            'total_risk': total_risk,
            'marginal_contributions': pd.Series(mctr, index=weights.index),
            'component_contributions': pd.Series(ctr, index=weights.index),
            'percentage_contributions': pd.Series(pct_ctr, index=weights.index)
        }
        
        # 如果有因子暴露信息，进行因子风险归因
        if self.factor_exposures is not None:
            factor_attribution = self._factor_risk_attribution(weights, covariance_matrix)
            attribution_result['factor_attribution'] = factor_attribution
        
        return attribution_result
    
    def _factor_risk_attribution(self, weights: pd.Series, 
                                covariance_matrix: np.ndarray) -> Dict[str, float]:
        """因子风险归因"""
        if self.factor_exposures is None:
            return {}
        
        # 对齐权重和因子暴露
        aligned_weights = weights.reindex(self.factor_exposures.index, fill_value=0.0)
        
        # 组合在各因子上的暴露
        portfolio_exposures = self.factor_exposures.T @ aligned_weights
        
        # 如果有因子协方差矩阵，计算因子风险贡献
        factor_risk_contrib = {}
        for factor in self.factor_exposures.columns:
            factor_exposure = portfolio_exposures[factor]
            # 简化的因子风险贡献（需要因子协方差矩阵来精确计算）
            factor_risk_contrib[factor] = factor_exposure**2
        
        return factor_risk_contrib
    
    def transaction_cost_analysis(self, current_weights: pd.Series, 
                                 target_weights: pd.Series,
                                 market_data: pd.DataFrame) -> Dict[str, float]:
        """交易成本分析"""
        logger.info("进行交易成本分析")
        
        trades = target_weights - current_weights
        abs_trades = trades.abs()
        
        cost_analysis = {
            'total_turnover': abs_trades.sum(),
            'total_trades': (abs_trades > 0.001).sum(),
            'avg_trade_size': abs_trades[abs_trades > 0.001].mean() if (abs_trades > 0.001).sum() > 0 else 0.0
        }
        
        # 如果有市场数据，估算具体成本
        if market_data is not None:
            # 流动性成本（基于bid-ask spread）
            if 'bid_ask_spread' in market_data.columns:
                spread_costs = abs_trades * market_data.get('bid_ask_spread', 0.001)
                cost_analysis['spread_cost'] = spread_costs.sum()
            
            # 市场冲击成本（基于成交量）
            if 'avg_daily_volume' in market_data.columns:
                # 简化的市场冲击模型：成本与交易量的平方根成正比
                volume_ratios = abs_trades / (market_data.get('avg_daily_volume', 1e6) + 1e-9)
                impact_costs = 0.1 * np.sqrt(volume_ratios) * abs_trades  # 0.1为冲击系数
                cost_analysis['impact_cost'] = impact_costs.sum()
            
            # 总预估成本
            total_cost = cost_analysis.get('spread_cost', 0) + cost_analysis.get('impact_cost', 0)
            cost_analysis['total_estimated_cost'] = total_cost
            cost_analysis['cost_per_turnover'] = total_cost / (cost_analysis['total_turnover'] + 1e-12)
        
        return cost_analysis
    
    def stress_test(self, weights: pd.Series, 
                   covariance_matrix: np.ndarray,
                   stress_scenarios: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """压力测试"""
        logger.info("进行投资组合压力测试")
        
        stress_results = {}
        
        for scenario_name, shock_vector in stress_scenarios.items():
            # 确保冲击向量长度匹配
            if len(shock_vector) != len(weights):
                logger.warning(f"场景{scenario_name}的冲击向量长度不匹配，跳过")
                continue
            
            # 计算场景下的投资组合收益
            portfolio_shock = weights.values @ shock_vector
            
            # 计算VaR (Value at Risk)
            portfolio_vol = np.sqrt(weights.values.T @ covariance_matrix @ weights.values)
            var_95 = -1.645 * portfolio_vol  # 95% VaR（假设正态分布）
            var_99 = -2.326 * portfolio_vol  # 99% VaR
            
            stress_results[scenario_name] = {
                'portfolio_return': portfolio_shock,
                'var_95': var_95,
                'var_99': var_99,
                'stress_ratio': portfolio_shock / var_95 if var_95 != 0 else 0.0
            }
        
        return stress_results
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化总结"""
        if not self.optimization_history:
            return {'message': '暂无优化历史'}
        
        recent_optimization = self.optimization_history[-1]
        
        summary = {
            'total_optimizations': len(self.optimization_history),
            'recent_success': recent_optimization.get('success', False),
            'recent_metrics': recent_optimization.get('portfolio_metrics', {}),
            'average_turnover': np.mean([opt.get('portfolio_metrics', {}).get('turnover', 0) 
                                       for opt in self.optimization_history if opt.get('success', False)]),
            'average_concentration': np.mean([opt.get('portfolio_metrics', {}).get('concentration_hhi', 0) 
                                            for opt in self.optimization_history if opt.get('success', False)])
        }
        
        return summary


# ============ 辅助函数 ============

def create_stress_scenarios(asset_names: List[str]) -> Dict[str, np.ndarray]:
    """创建压力测试场景"""
    n_assets = len(asset_names)
    
    scenarios = {
        'market_crash': np.full(n_assets, -0.20),  # 市场崩盘：所有资产-20%
        'sector_rotation': np.random.choice([-0.10, 0.10], n_assets),  # 行业轮动
        'volatility_spike': np.random.normal(0, 0.15, n_assets),  # 波动率激增
        'liquidity_crunch': np.random.choice([-0.05, -0.15], n_assets, p=[0.8, 0.2]),  # 流动性危机
    }
    
    return scenarios


# ============ 测试代码 ============

def test_portfolio_optimizer():
    """测试投资组合优化器"""
    
    # 生成模拟数据
    np.random.seed(42)
    n_assets = 50
    n_days = 252
    
    # 生成资产名称
    asset_names = [f'STOCK_{i:03d}' for i in range(n_assets)]
    
    # 生成预期收益率
    expected_returns = pd.Series(
        np.random.normal(0.08, 0.05, n_assets),
        index=asset_names
    )
    
    # 生成历史收益率用于协方差估计
    returns_data = np.random.multivariate_normal(
        mean=expected_returns.values / 252,
        cov=np.eye(n_assets) * 0.0001,
        size=n_days
    )
    returns_df = pd.DataFrame(returns_data, columns=asset_names)
    
    # 生成股票池数据
    universe_data = pd.DataFrame({
        'SECTOR': np.random.choice(['TECH', 'FINANCE', 'ENERGY', 'HEALTH'], n_assets),
        'COUNTRY': np.random.choice(['US', 'EU', 'ASIA'], n_assets),
        'liquidity_rank': np.random.uniform(0, 1, n_assets)
    }, index=asset_names)
    
    # 初始化优化器
    optimizer = AdvancedPortfolioOptimizer(
        risk_aversion=5.0,
        turnover_penalty=1.0,
        max_turnover=0.15,
        max_position=0.05,
        covariance_method="ledoit_wolf"
    )
    
    # 估计协方差矩阵
    logger.info("估计协方差矩阵")
    cov_matrix = optimizer.estimate_covariance_matrix(returns_df)
    print(f"协方差矩阵形状: {cov_matrix.shape}")
    
    # 优化投资组合
    logger.info("优化投资组合")
    optimization_result = optimizer.optimize_portfolio(
        expected_returns=expected_returns,
        covariance_matrix=cov_matrix,
        current_weights=None,
        universe_data=universe_data
    )
    
    if optimization_result['success']:
        optimal_weights = optimization_result['optimal_weights']
        portfolio_metrics = optimization_result['portfolio_metrics']
        
        print(f"优化成功!")
        print(f"预期收益: {portfolio_metrics['expected_return']:.4f}")
        print(f"预期波动率: {portfolio_metrics['expected_volatility']:.4f}")
        print(f"夏普比率: {portfolio_metrics['sharpe_ratio']:.4f}")
        print(f"换手率: {portfolio_metrics['turnover']:.4f}")
        print(f"有效持仓数: {portfolio_metrics['effective_positions']:.1f}")
        
        # 显示主要持仓
        top_positions = optimal_weights[optimal_weights.abs() > 0.01].sort_values(key=abs, ascending=False)
        print(f"\n主要持仓 (>1%):")
        for asset, weight in top_positions.head(10).items():
            print(f"  {asset}: {weight:.3f}")
        
        # 风险归因
        logger.info("风险归因分析")
        risk_attribution = optimizer.risk_attribution(optimal_weights, cov_matrix)
        print(f"\n总风险: {risk_attribution['total_risk']:.4f}")
        
        # 主要风险贡献者
        top_risk_contributors = risk_attribution['percentage_contributions'].abs().sort_values(ascending=False)
        print(f"主要风险贡献者:")
        for asset, contrib in top_risk_contributors.head(5).items():
            print(f"  {asset}: {contrib:.2f}%")
        
        # 压力测试
        logger.info("压力测试")
        stress_scenarios = create_stress_scenarios(asset_names)
        stress_results = optimizer.stress_test(optimal_weights, cov_matrix, stress_scenarios)
        
        print(f"\n压力测试结果:")
        for scenario, results in stress_results.items():
            print(f"  {scenario}: 收益={results['portfolio_return']:.4f}, "
                  f"VaR95={results['var_95']:.4f}")
        
        # 交易成本分析
        logger.info("交易成本分析")
        current_weights = pd.Series(0.0, index=asset_names)  # 假设从空仓开始
        cost_analysis = optimizer.transaction_cost_analysis(
            current_weights, optimal_weights, universe_data
        )
        
        print(f"\n交易成本分析:")
        print(f"  总换手率: {cost_analysis['total_turnover']:.4f}")
        print(f"  交易笔数: {cost_analysis['total_trades']}")
        print(f"  平均交易规模: {cost_analysis['avg_trade_size']:.4f}")
        
    else:
        print(f"优化失败: {optimization_result.get('error_message', '未知错误')}")
    
    # 优化总结
    summary = optimizer.get_optimization_summary()
    print(f"\n优化总结: {summary}")


if __name__ == "__main__":
    test_portfolio_optimizer()
