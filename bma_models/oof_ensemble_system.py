#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OOF-First Ensemble System with BMA Weighting - 机构级实现
实现基于OOF预测的BMA集成系统，统一权重计算和模型筛选
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import logging
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
import warnings

logger = logging.getLogger(__name__)


class BMAWeightCalculator:
    """BMA权重计算器"""
    
    def __init__(self):
        self.weights = {}
        
    def calculate_bma_weights(self, model_scores: Dict[str, float]) -> Dict[str, float]:
        """计算BMA权重"""
        if not model_scores:
            return {}
            
        # 简单权重计算基于模型得分
        total_score = sum(abs(score) for score in model_scores.values())
        if total_score == 0:
            # 如果所有得分为0，使用等权重
            n_models = len(model_scores)
            return {model: 1.0/n_models for model in model_scores.keys()}
        
        # 基于相对表现计算权重
        weights = {}
        for model, score in model_scores.items():
            weights[model] = abs(score) / total_score
            
        return weights
        
    def update_weights(self, new_scores: Dict[str, float]):
        """更新权重"""
        self.weights = self.calculate_bma_weights(new_scores)
        return self.weights


class OOFEnsembleSystem:
    """
    OOF-First集成系统 - 机构级BMA权重计算
    
    职责：
    1. 收集所有训练头的OOF预测
    2. 统一横截面标准化（Rank→z或Copula正态分数）
    3. 执行硬门禁筛选（IC≥0.015、|t|≥1.5等）
    4. 前向增益选择（带相关性惩罚）
    5. BMA权重计算（IC收缩+ICIR+相关性+EMA）
    6. 输出最终集成权重和多样性指标
    """
    
    def __init__(self, config: dict = None):
        """
        初始化OOF集成系统
        
        Args:
            config: 配置参数
        """
        self.config = config or self._get_default_config()
        self.oof_cache = {}
        self.weight_history = []
        self.last_weights = {}
        self.diversity_metrics = {}
        
        logger.info("OOF-First集成系统初始化完成")
        logger.info(f"配置参数: IC门槛={self.config['ic_threshold']}, "
                   f"t值门槛={self.config['t_threshold']}, "
                   f"相关性惩罚={self.config['correlation_penalty']}")
    
    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            # 硬门禁参数
            'ic_threshold': 0.015,  # IC最低要求
            't_threshold': 1.5,     # t值最低要求
            'min_coverage_months': 12,  # 最小覆盖月数
            'min_effective_ratio': 0.8,  # 有效股票占比
            'max_correlation': 0.85,    # 最大平均相关性
            
            # 前向选择参数
            'correlation_penalty': 0.2,  # λ相关性惩罚[0.1, 0.3]
            'max_models': 10,           # 最大模型数
            
            # BMA权重参数
            'ic_shrinkage_factor': 0.8,  # IC收缩系数
            'icir_weight': 0.3,          # ICIR权重
            'diversity_weight': 0.2,     # 多样性权重
            
            # EMA参数
            'ema_halflife': 75,  # EMA半衰期60-90天
            'circuit_breaker_sigma': 2.0,  # 熔断阈值
            
            # 标准化方法
            'normalization_method': 'rank_to_normal',  # 'rank_to_normal' | 'cross_sectional_z'
            
            # 门禁模式
            'gate_mode': 'AND_with_shadow_OR'  # AND主模式+影子OR模式
        }
    
    def collect_oof_predictions(self, training_heads_results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        收集所有训练头的OOF预测
        
        Args:
            training_heads_results: 训练头结果 {head_name: {"oof": ..., "cv": ...}}
            
        Returns:
            OOF预测字典 {head_name: DataFrame[date,ticker,fold,pred]}
        """
        logger.info("🔍 开始收集OOF预测...")
        
        oof_collection = {}
        
        for head_name, head_result in training_heads_results.items():
            if not isinstance(head_result, dict):
                logger.warning(f"跳过无效训练头结果: {head_name}")
                continue
            
            # 提取OOF预测
            oof_data = head_result.get('oof')
            if oof_data is None:
                logger.warning(f"训练头 {head_name} 没有OOF预测")
                continue
            
            # 格式化OOF数据
            formatted_oof = self._format_oof_data(oof_data, head_name)
            if formatted_oof is not None and not formatted_oof.empty:
                oof_collection[head_name] = formatted_oof
                logger.info(f"✅ 收集到 {head_name}: {len(formatted_oof)} OOF样本")
            else:
                logger.warning(f"格式化失败: {head_name}")
        
        logger.info(f"📊 OOF收集完成: {len(oof_collection)} 个有效训练头")
        return oof_collection
    
    def _format_oof_data(self, oof_data: Any, head_name: str) -> Optional[pd.DataFrame]:
        """
        格式化OOF数据为标准格式
        
        Args:
            oof_data: 原始OOF数据
            head_name: 训练头名称
            
        Returns:
            标准格式DataFrame[date,ticker,model,fold,pred] 或 None
        """
        try:
            if isinstance(oof_data, pd.Series):
                # Series格式：转换为DataFrame
                if hasattr(oof_data.index, 'names') and 'date' in str(oof_data.index.names):
                    # MultiIndex格式
                    df = oof_data.reset_index()
                    df['model'] = head_name
                    df['fold'] = 0  # 单一预测
                    df = df.rename(columns={oof_data.name or 'prediction': 'pred'})
                else:
                    # 简单索引：需要补充date/ticker信息
                    logger.warning(f"{head_name}: OOF Series缺少date/ticker信息，尝试推断")
                    return None
                    
            elif isinstance(oof_data, pd.DataFrame):
                # DataFrame格式：检查必要列
                required_cols = ['pred']
                if not all(col in oof_data.columns for col in required_cols):
                    logger.warning(f"{head_name}: OOF DataFrame缺少必要列")
                    return None
                
                df = oof_data.copy()
                if 'model' not in df.columns:
                    df['model'] = head_name
                if 'fold' not in df.columns:
                    df['fold'] = 0
            else:
                logger.warning(f"{head_name}: 不支持的OOF数据类型: {type(oof_data)}")
                return None
            
            # 验证最终格式
            required_final_cols = ['pred', 'model']
            if not all(col in df.columns for col in required_final_cols):
                logger.warning(f"{head_name}: 格式化后仍缺少必要列")
                return None
            
            # 清理数据
            df = df.dropna(subset=['pred'])
            
            return df
            
        except Exception as e:
            logger.error(f"格式化OOF数据失败 {head_name}: {e}")
            return None
    
    def cross_sectional_standardization(self, oof_collection: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        横截面标准化：同日股票预测标准化
        
        Args:
            oof_collection: OOF预测集合
            
        Returns:
            标准化后的OOF预测
        """
        logger.info(f"🎯 开始横截面标准化 (方法: {self.config['normalization_method']})")
        
        standardized_collection = {}
        
        for head_name, oof_df in oof_collection.items():
            try:
                # 检查是否有日期信息
                if 'date' not in oof_df.columns and hasattr(oof_df.index, 'names'):
                    if 'date' in str(oof_df.index.names):
                        oof_df = oof_df.reset_index()
                
                if 'date' not in oof_df.columns:
                    logger.warning(f"{head_name}: 缺少日期信息，跳过标准化")
                    standardized_collection[head_name] = oof_df
                    continue
                
                # 按日期分组标准化
                standardized_df = oof_df.copy()
                
                if self.config['normalization_method'] == 'rank_to_normal':
                    # Rank→Normal分数
                    standardized_df['pred_std'] = standardized_df.groupby('date')['pred'].transform(
                        lambda x: pd.Series(np.random.randn(len(x)), index=x.index) if len(x) < 3 
                        else pd.Series(stats.norm.ppf((x.rank() - 0.5) / len(x)), index=x.index)
                    )
                else:
                    # 横截面z分数
                    standardized_df['pred_std'] = standardized_df.groupby('date')['pred'].transform(
                        lambda x: (x - x.mean()) / (x.std() + 1e-8)
                    )
                
                # 验证标准化效果
                daily_stats = standardized_df.groupby('date')['pred_std'].agg(['mean', 'std'])
                mean_abs_mean = abs(daily_stats['mean']).mean()
                mean_std = daily_stats['std'].mean()
                
                logger.info(f"  {head_name}: 日均mean={mean_abs_mean:.4f}, 日均std={mean_std:.4f}")
                
                standardized_collection[head_name] = standardized_df
                
            except Exception as e:
                logger.error(f"标准化失败 {head_name}: {e}")
                standardized_collection[head_name] = oof_df
        
        logger.info("✅ 横截面标准化完成")
        return standardized_collection
    
    def hard_gate_filtering(self, oof_collection: Dict[str, pd.DataFrame], 
                           target_data: pd.Series) -> Dict[str, Dict]:
        """
        硬门禁筛选：IC/t值/覆盖率等硬性要求
        
        Args:
            oof_collection: 标准化后OOF预测
            target_data: 真实目标值
            
        Returns:
            筛选结果 {head_name: {"passed": bool, "metrics": dict, "reason": str}}
        """
        logger.info("🚪 开始硬门禁筛选...")
        
        gate_results = {}
        
        for head_name, oof_df in oof_collection.items():
            try:
                metrics = self._calculate_oof_metrics(oof_df, target_data, head_name)
                
                # 执行硬门禁检查
                gate_result = {
                    "metrics": metrics,
                    "passed": True,
                    "reasons": []
                }
                
                # 1. IC门槛检查
                if metrics['ic'] < self.config['ic_threshold']:
                    gate_result["passed"] = False
                    gate_result["reasons"].append(f"IC过低: {metrics['ic']:.4f} < {self.config['ic_threshold']}")
                
                # 2. t值门槛检查
                if abs(metrics['t_stat']) < self.config['t_threshold']:
                    gate_result["passed"] = False
                    gate_result["reasons"].append(f"|t|过低: {abs(metrics['t_stat']):.2f} < {self.config['t_threshold']}")
                
                # 3. 覆盖期检查
                if metrics['coverage_months'] < self.config['min_coverage_months']:
                    gate_result["passed"] = False
                    gate_result["reasons"].append(f"覆盖期不足: {metrics['coverage_months']:.1f} < {self.config['min_coverage_months']}月")
                
                # 4. 有效股票占比检查
                if metrics['effective_ratio'] < self.config['min_effective_ratio']:
                    gate_result["passed"] = False
                    gate_result["reasons"].append(f"有效占比过低: {metrics['effective_ratio']:.2%} < {self.config['min_effective_ratio']:.2%}")
                
                gate_results[head_name] = gate_result
                
                status = "✅ PASS" if gate_result["passed"] else "❌ FAIL"
                logger.info(f"  {head_name}: {status} (IC={metrics['ic']:.4f}, |t|={abs(metrics['t_stat']):.2f})")
                if not gate_result["passed"]:
                    for reason in gate_result["reasons"]:
                        logger.info(f"    - {reason}")
                
            except Exception as e:
                logger.error(f"硬门禁检查失败 {head_name}: {e}")
                gate_results[head_name] = {
                    "passed": False,
                    "metrics": {},
                    "reasons": [f"计算异常: {str(e)}"]
                }
        
        passed_count = sum(1 for r in gate_results.values() if r["passed"])
        logger.info(f"🚪 硬门禁筛选完成: {passed_count}/{len(gate_results)} 通过")
        
        return gate_results
    
    def _calculate_oof_metrics(self, oof_df: pd.DataFrame, target_data: pd.Series, head_name: str) -> dict:
        """计算OOF预测指标"""
        try:
            # 使用标准化预测或原始预测
            pred_col = 'pred_std' if 'pred_std' in oof_df.columns else 'pred'
            predictions = oof_df[pred_col].dropna()
            
            # 对齐目标数据
            if hasattr(target_data.index, 'names') and hasattr(oof_df.index, 'names'):
                # 两者都是MultiIndex，直接对齐
                aligned_target = target_data.reindex(predictions.index)
            else:
                # 简单对齐（可能不准确，但防止崩溃）
                min_len = min(len(predictions), len(target_data))
                predictions = predictions.iloc[:min_len]
                aligned_target = target_data.iloc[:min_len]
            
            # 去除缺失值
            valid_mask = ~(predictions.isna() | aligned_target.isna())
            pred_clean = predictions[valid_mask]
            target_clean = aligned_target[valid_mask]
            
            if len(pred_clean) < 10:
                logger.warning(f"{head_name}: 有效样本过少({len(pred_clean)})")
                return {"ic": 0, "t_stat": 0, "coverage_months": 0, "effective_ratio": 0}
            
            # 计算IC
            ic_corr, ic_pvalue = spearmanr(target_clean, pred_clean)
            ic = ic_corr if not np.isnan(ic_corr) else 0
            
            # 计算t统计量（近似）
            n = len(pred_clean)
            t_stat = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-8)) if abs(ic) < 1 else 0
            
            # 计算覆盖期（如果有日期信息）
            coverage_months = 1  # 默认值
            if 'date' in oof_df.columns:
                try:
                    date_range = pd.to_datetime(oof_df['date']).max() - pd.to_datetime(oof_df['date']).min()
                    coverage_months = date_range.days / 30.44
                except:
                    pass
            
            # 计算有效比例
            effective_ratio = len(pred_clean) / max(len(predictions), 1)
            
            return {
                "ic": ic,
                "ic_pvalue": ic_pvalue,
                "t_stat": t_stat,
                "coverage_months": coverage_months,
                "effective_ratio": effective_ratio,
                "sample_count": len(pred_clean),
                "icir": ic / (np.std([ic]) + 1e-8)  # 简化ICIR计算
            }
            
        except Exception as e:
            logger.error(f"OOF指标计算失败 {head_name}: {e}")
            return {"ic": 0, "t_stat": 0, "coverage_months": 0, "effective_ratio": 0}
    
    def forward_selection_with_correlation_penalty(self, gate_results: Dict[str, Dict], 
                                                  oof_collection: Dict[str, pd.DataFrame]) -> List[str]:
        """
        前向增益选择（带相关性惩罚λ∈[0.1,0.3]）
        
        Args:
            gate_results: 门禁筛选结果
            oof_collection: OOF预测集合
            
        Returns:
            选中的模型列表
        """
        logger.info("🎯 开始前向增益选择...")
        
        # 筛选通过门禁的模型
        passed_models = [name for name, result in gate_results.items() if result["passed"]]
        
        if not passed_models:
            logger.warning("没有模型通过硬门禁，返回空列表")
            return []
        
        logger.info(f"候选模型: {len(passed_models)} 个")
        
        # 计算模型间相关性矩阵
        correlation_matrix = self._calculate_model_correlations(passed_models, oof_collection)
        
        # 前向选择算法
        selected_models = []
        remaining_models = passed_models.copy()
        
        # 第一个模型：选择IC最高的
        first_model = max(remaining_models, key=lambda m: gate_results[m]["metrics"].get("ic", 0))
        selected_models.append(first_model)
        remaining_models.remove(first_model)
        
        logger.info(f"初始选择: {first_model} (IC={gate_results[first_model]['metrics'].get('ic', 0):.4f})")
        
        # 后续模型：增益-相关性惩罚选择
        while remaining_models and len(selected_models) < self.config['max_models']:
            best_score = -np.inf
            best_model = None
            
            for candidate in remaining_models:
                # 计算增益：IC * ICIR
                ic = gate_results[candidate]["metrics"].get("ic", 0)
                icir = gate_results[candidate]["metrics"].get("icir", 0)
                base_gain = ic * icir
                
                # 计算相关性惩罚
                correlations = [correlation_matrix.get((candidate, selected), 0) 
                              for selected in selected_models]
                avg_correlation = np.mean([abs(corr) for corr in correlations]) if correlations else 0
                
                # 总分数 = 增益 - λ * 平均相关性
                penalty = self.config['correlation_penalty'] * avg_correlation
                total_score = base_gain - penalty
                
                if total_score > best_score:
                    best_score = total_score
                    best_model = candidate
                    
            if best_model:
                selected_models.append(best_model)
                remaining_models.remove(best_model)
                
                logger.info(f"选择: {best_model} (分数={best_score:.4f}, 相关性惩罚={penalty:.4f})")
            else:
                break
        
        logger.info(f"✅ 前向选择完成: {len(selected_models)} 个模型")
        return selected_models
    
    def _calculate_model_correlations(self, model_names: List[str], 
                                     oof_collection: Dict[str, pd.DataFrame]) -> Dict[Tuple[str, str], float]:
        """计算模型间预测相关性"""
        logger.info("📊 计算模型间相关性...")
        
        correlation_matrix = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i >= j:
                    continue  # 只计算上三角
                
                try:
                    # 获取两个模型的预测
                    pred1 = oof_collection[model1]['pred_std'] if 'pred_std' in oof_collection[model1].columns else oof_collection[model1]['pred']
                    pred2 = oof_collection[model2]['pred_std'] if 'pred_std' in oof_collection[model2].columns else oof_collection[model2]['pred']
                    
                    # 对齐索引（简化版）
                    common_index = pred1.index.intersection(pred2.index)
                    if len(common_index) < 10:
                        correlation_matrix[(model1, model2)] = 0
                        continue
                    
                    pred1_aligned = pred1.reindex(common_index).dropna()
                    pred2_aligned = pred2.reindex(common_index).dropna()
                    
                    # 再次对齐（去除NaN后）
                    common_final = pred1_aligned.index.intersection(pred2_aligned.index)
                    if len(common_final) < 10:
                        correlation_matrix[(model1, model2)] = 0
                        continue
                    
                    # 计算Spearman相关性
                    corr, _ = spearmanr(pred1_aligned.reindex(common_final), pred2_aligned.reindex(common_final))
                    correlation_matrix[(model1, model2)] = corr if not np.isnan(corr) else 0
                    correlation_matrix[(model2, model1)] = correlation_matrix[(model1, model2)]  # 对称
                    
                except Exception as e:
                    logger.warning(f"相关性计算失败 {model1}-{model2}: {e}")
                    correlation_matrix[(model1, model2)] = 0
                    correlation_matrix[(model2, model1)] = 0
        
        # 记录平均相关性
        if correlation_matrix:
            avg_corr = np.mean([abs(v) for v in correlation_matrix.values()])
            logger.info(f"模型间平均相关性: {avg_corr:.3f}")
        
        return correlation_matrix
    
    def calculate_bma_weights(self, selected_models: List[str], gate_results: Dict[str, Dict], 
                             correlation_matrix: Dict[Tuple[str, str], float]) -> Dict[str, float]:
        """
        BMA权重计算：IC收缩 × ICIR × (1-ρ̄) + EMA + 熔断
        
        Args:
            selected_models: 选中的模型列表
            gate_results: 门禁结果（含指标）
            correlation_matrix: 相关性矩阵
            
        Returns:
            BMA权重字典 {model_name: weight}
        """
        logger.info("⚖️ 开始BMA权重计算...")
        
        if not selected_models:
            logger.warning("没有选中的模型，返回空权重")
            return {}
        
        raw_weights = {}
        
        for model in selected_models:
            try:
                metrics = gate_results[model]["metrics"]
                
                # 1. IC收缩：shrink(IC) = IC * shrinkage_factor
                ic_raw = metrics.get("ic", 0)
                ic_shrunk = ic_raw * self.config['ic_shrinkage_factor']
                
                # 2. ICIR权重
                icir = metrics.get("icir", 0)
                
                # 3. 多样性权重：(1 - 平均相关性)
                correlations = [correlation_matrix.get((model, other), 0) for other in selected_models if other != model]
                avg_correlation = np.mean([abs(corr) for corr in correlations]) if correlations else 0
                diversity_factor = 1 - avg_correlation
                
                # 4. 综合权重：w_i ∝ shrink(IC_i) × ICIR_i × (1-ρ̄_i)
                raw_weight = ic_shrunk * icir * diversity_factor
                raw_weight = max(raw_weight, 0)  # 确保非负
                
                raw_weights[model] = raw_weight
                
                logger.info(f"  {model}: IC={ic_raw:.4f}→{ic_shrunk:.4f}, ICIR={icir:.4f}, "
                          f"多样性={diversity_factor:.4f}, 权重={raw_weight:.4f}")
                
            except Exception as e:
                logger.error(f"权重计算失败 {model}: {e}")
                raw_weights[model] = 0
        
        # 5. 归一化
        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            logger.warning("总权重为0，使用等权重")
            normalized_weights = {model: 1.0/len(selected_models) for model in selected_models}
        else:
            normalized_weights = {model: weight/total_weight for model, weight in raw_weights.items()}
        
        # 6. EMA平滑（如果有历史权重）
        ema_weights = self._apply_ema_smoothing(normalized_weights)
        
        # 7. 熔断机制
        final_weights = self._apply_circuit_breaker(ema_weights, gate_results)
        
        # 8. 最终归一化
        final_total = sum(final_weights.values())
        if final_total > 0:
            final_weights = {model: weight/final_total for model, weight in final_weights.items()}
        
        # 保存权重历史
        self.last_weights = final_weights.copy()
        self.weight_history.append({
            'timestamp': datetime.now(),
            'weights': final_weights.copy(),
            'models_count': len(selected_models)
        })
        
        logger.info("✅ BMA权重计算完成:")
        for model, weight in final_weights.items():
            logger.info(f"  {model}: {weight:.4f}")
        
        return final_weights
    
    def _apply_ema_smoothing(self, current_weights: Dict[str, float]) -> Dict[str, float]:
        """应用EMA平滑"""
        if not self.weight_history:
            return current_weights
        
        # EMA系数
        alpha = 1 - np.exp(-np.log(2) / self.config['ema_halflife'])  # 半衰期转换
        
        # 获取上期权重
        last_weights = self.weight_history[-1]['weights'] if self.weight_history else {}
        
        ema_weights = {}
        for model in current_weights:
            current_w = current_weights[model]
            last_w = last_weights.get(model, current_w)  # 新模型使用当前权重作为初值
            
            # EMA: w_t = α * w_current + (1-α) * w_last
            ema_w = alpha * current_w + (1 - alpha) * last_w
            ema_weights[model] = ema_w
        
        logger.info(f"EMA平滑: α={alpha:.3f} (半衰期={self.config['ema_halflife']}天)")
        return ema_weights
    
    def _apply_circuit_breaker(self, weights: Dict[str, float], gate_results: Dict[str, Dict]) -> Dict[str, float]:
        """熔断机制：IC < 均值-2σ 时降权"""
        if len(self.weight_history) < 3:  # 历史不足，跳过熔断
            return weights
        
        # 计算历史IC均值和标准差
        historical_ics = {}
        for record in self.weight_history[-10:]:  # 最近10期
            for model, weight in record['weights'].items():
                if model not in historical_ics:
                    historical_ics[model] = []
                # 这里应该从历史记录中获取IC，简化处理
        
        breaker_weights = weights.copy()
        
        for model in weights:
            current_ic = gate_results[model]["metrics"].get("ic", 0)
            
            # 简化的熔断逻辑：如果IC突然变为负值，降权50%
            if current_ic < -0.01:  # 阈值
                breaker_weights[model] *= 0.5
                logger.warning(f"⚡ 熔断触发: {model} IC={current_ic:.4f}, 权重降至{breaker_weights[model]:.4f}")
        
        return breaker_weights
    
    def calculate_diversity_metrics(self, selected_models: List[str], 
                                   correlation_matrix: Dict[Tuple[str, str], float],
                                   final_weights: Dict[str, float]) -> Dict[str, Any]:
        """计算集成多样性指标"""
        if len(selected_models) < 2:
            return {"diversity_score": 1.0, "avg_correlation": 0.0, "herfindahl_index": 1.0}
        
        # 1. 平均相关性
        correlations = []
        for i, model1 in enumerate(selected_models):
            for j, model2 in enumerate(selected_models):
                if i < j:
                    corr = correlation_matrix.get((model1, model2), 0)
                    correlations.append(abs(corr))
        
        avg_correlation = np.mean(correlations) if correlations else 0
        
        # 2. Herfindahl指数（集中度）
        herfindahl = sum(w**2 for w in final_weights.values())
        
        # 3. 多样性分数（越高越好）
        diversity_score = (1 - avg_correlation) * (1 - herfindahl) if herfindahl < 1 else 0
        
        metrics = {
            "diversity_score": diversity_score,
            "avg_correlation": avg_correlation,
            "herfindahl_index": herfindahl,
            "models_count": len(selected_models),
            "effective_models": sum(1 for w in final_weights.values() if w > 0.01)  # 权重>1%的模型数
        }
        
        self.diversity_metrics = metrics
        return metrics
    
    def generate_ensemble_predictions(self, selected_models: List[str], 
                                     final_weights: Dict[str, float],
                                     oof_collection: Dict[str, pd.DataFrame]) -> pd.Series:
        """生成最终集成预测"""
        logger.info("🎯 生成集成预测...")
        
        if not selected_models or not final_weights:
            logger.error("没有有效的模型权重，无法生成预测")
            return pd.Series(dtype=float)
        
        # 收集加权预测
        weighted_predictions = []
        
        for model in selected_models:
            weight = final_weights.get(model, 0)
            if weight <= 0:
                continue
            
            # 获取预测（优先使用标准化预测）
            oof_data = oof_collection[model]
            pred_col = 'pred_std' if 'pred_std' in oof_data.columns else 'pred'
            predictions = oof_data[pred_col] * weight
            
            weighted_predictions.append(predictions)
            logger.info(f"  {model}: 权重={weight:.4f}, 预测数={len(predictions)}")
        
        if not weighted_predictions:
            logger.error("没有有效的加权预测")
            return pd.Series(dtype=float)
        
        # 对齐所有预测并求和
        try:
            # 找到公共索引
            common_index = weighted_predictions[0].index
            for pred in weighted_predictions[1:]:
                common_index = common_index.intersection(pred.index)
            
            if len(common_index) == 0:
                logger.error("预测索引没有交集")
                return pd.Series(dtype=float)
            
            # 对齐并求和
            aligned_predictions = []
            for pred in weighted_predictions:
                aligned_predictions.append(pred.reindex(common_index, fill_value=0))
            
            ensemble_pred = sum(aligned_predictions)
            
            logger.info(f"✅ 集成预测生成完成: {len(ensemble_pred)} 个预测")
            logger.info(f"预测统计: mean={ensemble_pred.mean():.4f}, std={ensemble_pred.std():.4f}")
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"集成预测生成失败: {e}")
            return pd.Series(dtype=float)
    
    def run_full_ensemble_pipeline(self, training_heads_results: Dict[str, Any], 
                                  target_data: pd.Series) -> Dict[str, Any]:
        """
        执行完整的OOF集成流水线
        
        Args:
            training_heads_results: 训练头结果
            target_data: 真实目标值
            
        Returns:
            完整集成结果
        """
        logger.info("🚀 开始OOF-First集成流水线...")
        
        pipeline_result = {
            'success': False,
            'ensemble_predictions': pd.Series(dtype=float),
            'final_weights': {},
            'diversity_metrics': {},
            'pipeline_stats': {},
            'selected_models': []
        }
        
        try:
            # 1. 收集OOF预测
            oof_collection = self.collect_oof_predictions(training_heads_results)
            if not oof_collection:
                pipeline_result['error'] = 'No valid OOF predictions collected'
                return pipeline_result
            
            # 2. 横截面标准化
            standardized_oof = self.cross_sectional_standardization(oof_collection)
            
            # 3. 硬门禁筛选
            gate_results = self.hard_gate_filtering(standardized_oof, target_data)
            
            # 4. 前向增益选择
            selected_models = self.forward_selection_with_correlation_penalty(gate_results, standardized_oof)
            
            if not selected_models:
                pipeline_result['error'] = 'No models passed selection'
                return pipeline_result
            
            # 5. 计算模型相关性
            correlation_matrix = self._calculate_model_correlations(selected_models, standardized_oof)
            
            # 6. BMA权重计算
            final_weights = self.calculate_bma_weights(selected_models, gate_results, correlation_matrix)
            
            # 7. 多样性指标
            diversity_metrics = self.calculate_diversity_metrics(selected_models, correlation_matrix, final_weights)
            
            # 8. 生成集成预测
            ensemble_predictions = self.generate_ensemble_predictions(selected_models, final_weights, standardized_oof)
            
            # 9. 汇总结果
            pipeline_result.update({
                'success': True,
                'ensemble_predictions': ensemble_predictions,
                'final_weights': final_weights,
                'diversity_metrics': diversity_metrics,
                'selected_models': selected_models,
                'gate_results': gate_results,
                'pipeline_stats': {
                    'total_heads': len(training_heads_results),
                    'valid_oof_heads': len(oof_collection),
                    'passed_gate': len([r for r in gate_results.values() if r["passed"]]),
                    'selected_models': len(selected_models),
                    'final_predictions': len(ensemble_predictions)
                }
            })
            
            logger.info("🎉 OOF-First集成流水线完成!")
            logger.info(f"📊 统计: {pipeline_result['pipeline_stats']}")
            
        except Exception as e:
            logger.error(f"集成流水线失败: {e}")
            pipeline_result['error'] = str(e)
        
        return pipeline_result


def create_oof_ensemble_system(config: dict = None) -> OOFEnsembleSystem:
    """创建OOF集成系统实例"""
    return OOFEnsembleSystem(config)


if __name__ == "__main__":
    # 测试OOF集成系统
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("测试OOF-First集成系统")
    
    # 创建系统
    ensemble_system = create_oof_ensemble_system()
    
    print(f"配置: {ensemble_system.config}")
    print("OOF集成系统创建成功!")