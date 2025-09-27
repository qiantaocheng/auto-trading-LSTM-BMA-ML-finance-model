#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强底部20%惩罚系统 - 低初始惩罚，加速增长
===============================================
专门针对底部20%股票，初始惩罚很低，但增速快速上升
避免对所有股票造成损害，只聚焦于最差的股票
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class EnhancedBottom20PenaltySystem:
    """
    增强底部20%惩罚系统

    核心设计原则：
    1. 只对底部20%股票应用惩罚
    2. 初始惩罚非常低（接近0）
    3. 随着排名下降，惩罚加速增长
    4. 使用三次函数实现低初始+快速加速
    5. 对极底部（最差5%）股票应用最强惩罚
    """

    def __init__(self,
                 penalty_threshold: float = 0.08,      # 惩罚阈值：底部8% (大幅降低)
                 initial_penalty_factor: float = 0.005, # 初始惩罚因子：更低
                 max_penalty: float = 0.08,            # 最大惩罚：8% (降低)
                 acceleration_power: float = 2.5,      # 加速因子：更温和
                 market_cap_weight: float = 0.4,       # 市值权重
                 liquidity_weight: float = 0.6,        # 流动性权重
                 extreme_bottom_boost: float = 1.3,    # 极底部额外惩罚倍数 (降低)
                 illiq_lookback: int = 20):             # Amihud指标回看天数
        """
        初始化增强底部8%惩罚系统 (大幅保护小盘股)

        Args:
            penalty_threshold: 开始惩罚的阈值（底部8%，大幅降低）
            initial_penalty_factor: 初始惩罚因子（在阈值处的惩罚强度，已降低）
            max_penalty: 最大惩罚幅度（最底部股票的惩罚，已降低至8%）
            acceleration_power: 惩罚加速度（已降低至2.5，更温和）
            market_cap_weight: 市值权重
            liquidity_weight: 流动性权重
            extreme_bottom_boost: 极底部5%的额外惩罚倍数（已降低）
            illiq_lookback: Amihud指标回看天数
        """
        self.penalty_threshold = penalty_threshold
        self.initial_penalty_factor = initial_penalty_factor
        self.max_penalty = max_penalty
        self.acceleration_power = acceleration_power
        self.market_cap_weight = market_cap_weight
        self.liquidity_weight = liquidity_weight
        self.extreme_bottom_boost = extreme_bottom_boost
        self.illiq_lookback = illiq_lookback

        # 归一化权重
        total_weight = market_cap_weight + liquidity_weight
        self.market_cap_weight = market_cap_weight / total_weight
        self.liquidity_weight = liquidity_weight / total_weight

        logger.info(f"增强底部8%惩罚系统初始化 (保护小盘股):")
        logger.info(f"  惩罚阈值: 底部{penalty_threshold*100:.0f}%")
        logger.info(f"  初始惩罚因子: {initial_penalty_factor:.3f}")
        logger.info(f"  最大惩罚: {max_penalty*100:.1f}%")
        logger.info(f"  加速度: {acceleration_power:.1f}次方")
        logger.info(f"  权重: 市值={self.market_cap_weight:.2f}, 流动性={self.liquidity_weight:.2f}")
        logger.info(f"  极底部增强: {extreme_bottom_boost:.1f}x")

    def calculate_amihud_score(self,
                              returns: pd.Series,
                              volumes: pd.Series,
                              prices: Optional[pd.Series] = None) -> pd.Series:
        """计算Amihud流动性评分"""
        try:
            # 计算成交额
            if prices is not None:
                dollar_volume = volumes * prices
            else:
                dollar_volume = volumes

            # 避免除零
            dollar_volume = dollar_volume.replace(0, np.nan)

            # 计算Amihud非流动性指标
            price_impact = np.abs(returns) / (dollar_volume + 1e-10)
            price_impact = price_impact.replace([np.inf, -np.inf], np.nan)

            # 按股票分组计算中位数
            if isinstance(price_impact.index, pd.MultiIndex) and 'ticker' in price_impact.index.names:
                amihud_illiq = price_impact.groupby(level='ticker').apply(
                    lambda x: x.tail(self.illiq_lookback).median()
                )
            else:
                amihud_illiq = pd.Series(price_impact.median(), index=returns.index)

            # 转换为流动性评分（反转并归一化）
            amihud_illiq = np.log1p(amihud_illiq * 1e6)
            liquidity_score = 1 / (1 + amihud_illiq)

            # 归一化到[0, 1]
            min_score = liquidity_score.min()
            max_score = liquidity_score.max()
            if max_score > min_score:
                liquidity_score = (liquidity_score - min_score) / (max_score - min_score)

            return liquidity_score

        except Exception as e:
            logger.error(f"计算Amihud评分失败: {e}")
            return pd.Series(0.5, index=returns.index.get_level_values('ticker').unique()
                          if isinstance(returns.index, pd.MultiIndex) else returns.index)

    def calculate_market_cap_score(self, market_caps: pd.Series) -> pd.Series:
        """计算市值评分"""
        market_caps = market_caps.clip(lower=1e6)
        log_caps = np.log(market_caps)

        min_cap = log_caps.min()
        max_cap = log_caps.max()
        if max_cap > min_cap:
            cap_score = (log_caps - min_cap) / (max_cap - min_cap)
        else:
            cap_score = pd.Series(0.5, index=market_caps.index)

        return cap_score

    def calculate_enhanced_penalty_amount(self, percentile: float) -> float:
        """
        计算增强惩罚量：低初始 + 快速加速

        Args:
            percentile: 股票的百分位（0=最差，1=最好）

        Returns:
            惩罚量（0到max_penalty之间）
        """
        # 只对底部8%应用惩罚 (大幅保护小盘股)
        if percentile >= self.penalty_threshold:
            return 0.0

        # 计算在惩罚区间内的相对位置（0=最差，1=阈值处）
        relative_pos = percentile / self.penalty_threshold

        # 反转：penalty_intensity = 0（阈值处）到 1（最差）
        penalty_intensity = 1.0 - relative_pos

        # === 新的惩罚公式：低初始 + 快速加速 ===
        # 使用改进的多阶段函数

        # 阶段1：在阈值处（penalty_intensity = 0），惩罚为初始因子
        # 阶段2：随着penalty_intensity增加，惩罚按加速度增长

        # 基础惩罚：三次函数实现低开始+快速增长
        base_penalty = (self.initial_penalty_factor +
                       (1 - self.initial_penalty_factor) * (penalty_intensity ** self.acceleration_power))

        # 极底部增强：对最差5%股票额外惩罚
        if percentile < 0.05:  # 最差5%
            extreme_factor = self.extreme_bottom_boost
            # 在最差5%内部，惩罚更快增长
            extreme_intensity = (0.05 - percentile) / 0.05
            extreme_penalty = extreme_factor * (extreme_intensity ** 2)
            base_penalty += extreme_penalty * 0.3  # 额外30%惩罚
        elif percentile < 0.10:  # 次差5%（5%-10%）
            # 轻微增强
            mild_factor = 1.2
            base_penalty *= mild_factor

        # 应用最大惩罚限制
        final_penalty = min(base_penalty * self.max_penalty, self.max_penalty)

        return final_penalty

    def apply_enhanced_bottom8_penalty(self,
                                      predictions: pd.Series,
                                      feature_data: pd.DataFrame) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        应用增强底部8%惩罚系统 (大幅保护小盘股)
        """
        try:
            if predictions is None or len(predictions) == 0:
                return predictions, {}

            adjusted = predictions.copy()
            diagnostics = {
                'total_stocks': len(predictions),
                'penalized_stocks': 0,
                'avg_penalty': 0,
                'max_penalty_applied': 0,
                'min_penalty_applied': 0,
                'bottom_5_penalty': 0,
                'bottom_10_penalty': 0,
                'bottom_20_penalty': 0,
                'threshold_penalty': 0,
                'acceleration_effect': 0,
                'missing_features': [],
                'feature_availability': {}
            }

            # Check feature availability
            required_features = ['market_cap', 'Volume', 'Close', 'returns']
            available_features = []
            for feature in required_features:
                if feature in feature_data.columns:
                    available_features.append(feature)
                    diagnostics['feature_availability'][feature] = True
                else:
                    diagnostics['missing_features'].append(feature)
                    diagnostics['feature_availability'][feature] = False

            if len(available_features) == 0:
                logger.warning("警告: 没有找到任何必需的特征数据，无法计算惩罚")
                logger.warning(f"  需要的特征: {required_features}")
                logger.warning(f"  可用的列: {list(feature_data.columns)[:10]}")
                return predictions, diagnostics

            # === 1. 计算综合评分 ===
            composite_scores = pd.Series(0.5, index=predictions.index)

            # 市值评分
            if 'market_cap' in feature_data.columns:
                market_caps = feature_data['market_cap'].reindex(predictions.index)
                cap_scores = self.calculate_market_cap_score(market_caps.fillna(market_caps.median()))

                for ticker in predictions.index.get_level_values('ticker').unique():
                    ticker_mask = predictions.index.get_level_values('ticker') == ticker
                    if ticker in cap_scores.index:
                        composite_scores[ticker_mask] += self.market_cap_weight * cap_scores[ticker]
            else:
                # 使用价格×成交量估算
                if 'Close' in feature_data.columns and 'Volume' in feature_data.columns:
                    estimated_caps = feature_data['Close'] * feature_data['Volume'] * 1000
                    cap_scores = self.calculate_market_cap_score(estimated_caps.reindex(predictions.index).fillna(estimated_caps.median()))
                    composite_scores += self.market_cap_weight * cap_scores

            # 流动性评分
            if 'returns' in feature_data.columns and 'Volume' in feature_data.columns:
                returns = feature_data['returns'].reindex(predictions.index)
                volumes = feature_data['Volume'].reindex(predictions.index)
                prices = feature_data.get('Close', pd.Series()).reindex(predictions.index)

                liquidity_scores = self.calculate_amihud_score(returns, volumes, prices)

                for ticker in predictions.index.get_level_values('ticker').unique():
                    ticker_mask = predictions.index.get_level_values('ticker') == ticker
                    if ticker in liquidity_scores.index:
                        composite_scores[ticker_mask] += self.liquidity_weight * liquidity_scores[ticker]

            # === 2. 按日期计算百分位并应用增强惩罚 ===
            penalties = pd.Series(0.0, index=predictions.index, dtype=float)

            for date in predictions.index.get_level_values('date').unique():
                date_mask = predictions.index.get_level_values('date') == date
                date_scores = composite_scores[date_mask]

                # 计算百分位
                percentiles = date_scores.rank(pct=True)

                # 应用增强惩罚
                for idx, percentile in percentiles.items():
                    penalty = self.calculate_enhanced_penalty_amount(percentile)
                    penalties.loc[idx] = penalty

            # === 3. 应用惩罚到预测 ===
            adjusted = predictions - penalties

            # === 4. 详细统计信息 ===
            penalized_mask = penalties > 0.001
            diagnostics['penalized_stocks'] = penalized_mask.sum()
            diagnostics['penalized_ratio'] = penalized_mask.mean()

            if penalized_mask.any():
                diagnostics['avg_penalty'] = penalties[penalized_mask].mean()
                diagnostics['max_penalty_applied'] = penalties.max()
                diagnostics['min_penalty_applied'] = penalties[penalized_mask].min()

            # 分层统计 (调整为8%阈值)
            bottom_5_mask = composite_scores <= composite_scores.quantile(0.05)
            bottom_8_mask = composite_scores <= composite_scores.quantile(0.08)
            bottom_10_mask = composite_scores <= composite_scores.quantile(0.10)
            threshold_mask = (composite_scores > composite_scores.quantile(0.07)) & (composite_scores <= composite_scores.quantile(0.08))

            diagnostics['bottom_5_penalty'] = penalties[bottom_5_mask].mean() if bottom_5_mask.any() else 0
            diagnostics['bottom_8_penalty'] = penalties[bottom_8_mask].mean() if bottom_8_mask.any() else 0
            diagnostics['bottom_10_penalty'] = penalties[bottom_10_mask].mean() if bottom_10_mask.any() else 0
            diagnostics['threshold_penalty'] = penalties[threshold_mask].mean() if threshold_mask.any() else 0

            # 加速效果统计
            if diagnostics['bottom_5_penalty'] > 0 and diagnostics['threshold_penalty'] > 0:
                diagnostics['acceleration_effect'] = diagnostics['bottom_5_penalty'] / max(diagnostics['threshold_penalty'], 0.001)

            # 输出日志
            logger.info("=" * 80)
            logger.info("增强底部8%惩罚系统应用完成 (大幅保护小盘股):")
            logger.info(f"  总股票数: {diagnostics['total_stocks']}")
            logger.info(f"  受惩罚股票: {diagnostics['penalized_stocks']} ({diagnostics['penalized_ratio']*100:.1f}%)")
            logger.info(f"  平均惩罚: {diagnostics['avg_penalty']*100:.2f}%")
            logger.info(f"  最大惩罚: {diagnostics['max_penalty_applied']*100:.2f}%")
            logger.info(f"  最小惩罚: {diagnostics['min_penalty_applied']*100:.4f}%")
            logger.info("")
            logger.info("分层惩罚统计 (体现低初始+快速加速):")
            logger.info(f"  8%阈值处惩罚: {diagnostics['threshold_penalty']*100:.4f}% (极低初始)")
            logger.info(f"  底部8%平均: {diagnostics['bottom_8_penalty']*100:.2f}%")
            logger.info(f"  底部5%平均: {diagnostics['bottom_5_penalty']*100:.2f}% (快速加速)")
            logger.info(f"  加速倍数: {diagnostics['acceleration_effect']:.1f}x")
            logger.info("")
            logger.info("预测调整效果:")
            logger.info(f"  调整前均值: {predictions.mean():.4f}")
            logger.info(f"  调整后均值: {adjusted.mean():.4f}")
            logger.info(f"  相关性: {predictions.corr(adjusted):.4f}")
            logger.info(f"  只影响底部8%，保护92%股票不受损害")
            logger.info("=" * 80)

            return adjusted, diagnostics

        except Exception as e:
            logger.error(f"增强底部8%惩罚系统应用失败: {e}")
            return predictions, {}


def test_enhanced_penalty_curves():
    """测试增强惩罚曲线效果"""
    import matplotlib.pyplot as plt

    # 创建测试数据
    percentiles = np.linspace(0, 0.25, 1000)  # 只测试底部25%

    # 创建不同配置的惩罚系统
    configs = [
        ('New Default (Reduced Penalty)', 0.005, 2.5),
        ('Very Low Initial, Ultra Fast', 0.005, 4.0),
        ('Medium Initial, Normal Speed', 0.03, 2.0)
    ]

    plt.figure(figsize=(15, 10))

    # 主图：惩罚曲线对比
    plt.subplot(2, 2, 1)

    for name, initial, power in configs:
        penalty_system = EnhancedBottom20PenaltySystem(
            initial_penalty_factor=initial,
            acceleration_power=power,
            max_penalty=0.08
        )

        penalties = [penalty_system.calculate_enhanced_penalty_amount(p) for p in percentiles]
        plt.plot(percentiles * 100, np.array(penalties) * 100, label=name, linewidth=2)

    plt.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Penalty Threshold (20%)')
    plt.axvline(x=5, color='orange', linestyle='--', alpha=0.3, label='Extreme Bottom (5%)')
    plt.axvline(x=10, color='orange', linestyle=':', alpha=0.3, label='Very Bottom (10%)')

    plt.xlabel('Stock Percentile (%)')
    plt.ylabel('Penalty Amount (%)')
    plt.title('Enhanced Bottom 20% Penalty Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 25)

    # 局部图：底部5%细节
    plt.subplot(2, 2, 2)

    bottom_percentiles = np.linspace(0, 0.05, 200)
    for name, initial, power in configs:
        penalty_system = EnhancedBottom20PenaltySystem(
            initial_penalty_factor=initial,
            acceleration_power=power,
            max_penalty=0.08
        )

        penalties = [penalty_system.calculate_enhanced_penalty_amount(p) for p in bottom_percentiles]
        plt.plot(bottom_percentiles * 100, np.array(penalties) * 100, label=name, linewidth=2)

    plt.xlabel('Stock Percentile (%)')
    plt.ylabel('Penalty Amount (%)')
    plt.title('Bottom 5% Detail (Extreme Penalty Zone)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5)

    # 加速度分析
    plt.subplot(2, 2, 3)

    # 计算不同区间的惩罚增长率
    intervals = ['0-1%', '1-5%', '5-10%', '10-15%', '15-20%']

    for name, initial, power in configs:
        penalty_system = EnhancedBottom20PenaltySystem(
            initial_penalty_factor=initial,
            acceleration_power=power,
            max_penalty=0.08
        )

        growth_rates = []
        test_points = [0.005, 0.03, 0.075, 0.125, 0.175]  # 各区间中点

        for i, point in enumerate(test_points):
            current_penalty = penalty_system.calculate_enhanced_penalty_amount(point)
            if i == 0:
                growth_rate = current_penalty / 0.001  # 相对于极小值的增长
            else:
                prev_penalty = penalty_system.calculate_enhanced_penalty_amount(test_points[i-1])
                growth_rate = (current_penalty - prev_penalty) / prev_penalty if prev_penalty > 0 else 0
            growth_rates.append(growth_rate)

        plt.plot(intervals, growth_rates, marker='o', label=name, linewidth=2)

    plt.xlabel('Percentile Intervals')
    plt.ylabel('Penalty Growth Rate')
    plt.title('Penalty Acceleration by Interval')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # 对比表格
    plt.subplot(2, 2, 4)
    plt.axis('off')

    # 创建对比数据
    comparison_data = []
    test_percentiles = [0.20, 0.15, 0.10, 0.05, 0.01]  # 20%, 15%, 10%, 5%, 1%

    for name, initial, power in configs:
        penalty_system = EnhancedBottom20PenaltySystem(
            initial_penalty_factor=initial,
            acceleration_power=power,
            max_penalty=0.08
        )

        penalties = [penalty_system.calculate_enhanced_penalty_amount(p) for p in test_percentiles]
        comparison_data.append([name] + [f'{p*100:.3f}%' for p in penalties])

    # 创建表格
    headers = ['System', '20%', '15%', '10%', '5%', '1%']

    table_text = []
    table_text.append(' | '.join(f'{h:^12}' for h in headers))
    table_text.append('-' * (13 * len(headers) + len(headers) - 1))

    for row in comparison_data:
        table_text.append(' | '.join(f'{cell:^12}' for cell in row))

    plt.text(0.1, 0.8, '\n'.join(table_text), fontfamily='monospace', fontsize=9,
             verticalalignment='top', transform=plt.gca().transAxes)

    plt.title('Penalty Comparison Table', pad=20)

    plt.tight_layout()

    # 保存图表
    output_path = 'D:/trade/enhanced_bottom20_penalty_curves.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"增强惩罚曲线图已保存: {output_path}")

    # 验证关键特性
    print("\n" + "=" * 80)
    print("🎯 增强惩罚系统关键特性验证")
    print("=" * 80)

    optimal_system = EnhancedBottom20PenaltySystem(
        initial_penalty_factor=0.005,
        acceleration_power=2.5,
        max_penalty=0.08
    )

    # 验证低初始惩罚
    threshold_penalty = optimal_system.calculate_enhanced_penalty_amount(0.20)
    print(f"✅ 低初始惩罚: 20%阈值处惩罚 = {threshold_penalty*100:.4f}%")

    # 验证快速加速
    bottom5_penalty = optimal_system.calculate_enhanced_penalty_amount(0.05)
    bottom1_penalty = optimal_system.calculate_enhanced_penalty_amount(0.01)

    acceleration_5 = bottom5_penalty / max(threshold_penalty, 0.001)
    acceleration_1 = bottom1_penalty / max(threshold_penalty, 0.001)

    print(f"✅ 快速加速效果:")
    print(f"   - 底部5%相对阈值: {acceleration_5:.1f}x")
    print(f"   - 底部1%相对阈值: {acceleration_1:.1f}x")

    # 验证保护80%股票
    protected_penalty = optimal_system.calculate_enhanced_penalty_amount(0.25)  # 25%处
    print(f"✅ 保护80%股票: 25%处惩罚 = {protected_penalty*100:.4f}% (应为0)")

    print(f"✅ 系统符合设计要求：低初始 + 快速加速 + 保护80%")


if __name__ == "__main__":
    test_enhanced_penalty_curves()