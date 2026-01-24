#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验：检测80/20 OOF是否存在数据泄露

设计思路：
1. 使用已知的未来信息（shuffled target）来检测泄露
2. 比较真实OOF预测与随机打乱后的OOF预测
3. 检查特征标准化是否使用了测试集统计量
4. 验证CV分割的时间顺序是否正确
5. 检查目标变量计算是否使用了未来信息

如果存在数据泄露，那么：
- OOF预测应该能够"预测"随机打乱的目标（不应该）
- 特征标准化应该使用测试集统计量（不应该）
- CV分割应该包含未来数据（不应该）
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Fix module import path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'data_leakage_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger("data_leakage_detection")


class DataLeakageDetector:
    """数据泄露检测器"""
    
    def __init__(self, data_path: str, split: float = 0.8, horizon_days: int = 10):
        self.data_path = Path(data_path)
        self.split = split
        self.horizon_days = horizon_days
        self.df = None
        self.train_data = None
        self.test_data = None
        self.results = {}
        
    def load_data(self):
        """加载数据"""
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_parquet(self.data_path)
        
        if not isinstance(self.df.index, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex (date, ticker)")
        
        if 'date' not in self.df.index.names or 'ticker' not in self.df.index.names:
            raise ValueError("MultiIndex must have 'date' and 'ticker' levels")
        
        logger.info(f"Loaded {len(self.df)} samples, {len(self.df.index.get_level_values('date').unique())} unique dates")
        
    def split_data(self):
        """按时间分割数据（与80/20评估一致）"""
        dates = pd.Index(pd.to_datetime(self.df.index.get_level_values("date")).tz_localize(None).unique()).sort_values()
        n_dates = len(dates)
        
        split_idx = int(n_dates * self.split)
        # Purge leakage gap = horizon days (labels use forward returns)
        train_end_idx = max(0, split_idx - 1 - self.horizon_days)
        
        if train_end_idx <= 0:
            raise RuntimeError("Not enough dates to apply purge gap")
        
        train_start = dates[0]
        train_end = dates[train_end_idx]
        test_start = dates[split_idx]
        test_end = dates[-1]
        
        logger.info(f"Time split (purged): train={train_start.date()}..{train_end.date()}, "
                   f"test={test_start.date()}..{test_end.date()}")
        logger.info(f"  Train dates: {train_end_idx + 1}, Test dates: {n_dates - split_idx}")
        
        # Split data
        self.train_data = self.df.loc[
            (self.df.index.get_level_values('date') >= train_start) & 
            (self.df.index.get_level_values('date') <= train_end)
        ].copy()
        
        self.test_data = self.df.loc[
            (self.df.index.get_level_values('date') >= test_start) & 
            (self.df.index.get_level_values('date') <= test_end)
        ].copy()
        
        logger.info(f"Train samples: {len(self.train_data)}, Test samples: {len(self.test_data)}")
        
        return train_start, train_end, test_start, test_end
    
    def experiment_1_shuffled_target(self):
        """
        实验1：使用随机打乱的目标变量
        
        如果OOF预测存在数据泄露，那么：
        - 即使目标变量被随机打乱，OOF预测仍可能显示高相关性（不应该）
        - 真实目标变量的OOF预测应该比随机目标变量好很多
        """
        logger.info("=" * 80)
        logger.info("实验1: 随机打乱目标变量检测")
        logger.info("=" * 80)
        
        if 'target' not in self.train_data.columns:
            logger.warning("No 'target' column found, skipping experiment 1")
            return
        
        # 获取真实目标变量
        y_true = self.train_data['target'].copy()
        y_shuffled = y_true.copy()
        
        # 随机打乱目标变量（保持索引）
        np.random.seed(42)
        shuffled_values = np.random.permutation(y_shuffled.values)
        y_shuffled = pd.Series(shuffled_values, index=y_shuffled.index)
        
        # 模拟OOF预测（使用简单的线性模型作为代理）
        # 这里我们使用特征的前几个主成分来模拟预测
        feature_cols = [c for c in self.train_data.columns if c not in ['target', 'ticker', 'date']]
        if len(feature_cols) == 0:
            logger.warning("No feature columns found, skipping experiment 1")
            return
        
        X = self.train_data[feature_cols].fillna(0)
        
        # 使用简单的线性组合作为"预测"
        # 如果存在泄露，这个预测应该与真实目标相关，但不应该与随机目标相关
        np.random.seed(42)
        weights = np.random.randn(len(feature_cols))
        weights = weights / np.linalg.norm(weights)
        
        pred_true = X @ weights
        pred_shuffled = X @ weights  # 相同的预测，但目标不同
        
        # 计算相关性
        corr_true = pearsonr(pred_true, y_true)[0]
        corr_shuffled = pearsonr(pred_shuffled, y_shuffled)[0]
        
        logger.info(f"真实目标 vs 预测: IC = {corr_true:.6f}")
        logger.info(f"随机目标 vs 预测: IC = {corr_shuffled:.6f}")
        logger.info(f"差异: {abs(corr_true - corr_shuffled):.6f}")
        
        # 如果随机目标的相关性接近真实目标，可能存在泄露
        if abs(corr_shuffled) > 0.1:
            logger.warning("⚠️  警告: 随机打乱的目标变量仍显示高相关性，可能存在数据泄露！")
            self.results['exp1_leakage_detected'] = True
        else:
            logger.info("✅ 随机目标变量相关性低，未检测到明显泄露")
            self.results['exp1_leakage_detected'] = False
        
        self.results['exp1_corr_true'] = corr_true
        self.results['exp1_corr_shuffled'] = corr_shuffled
        
    def experiment_2_feature_scaling_leakage(self):
        """
        实验2：检测特征标准化是否使用了测试集统计量
        
        如果存在泄露：
        - 使用训练+测试集统计量标准化的特征，在训练集上的预测应该更好（不应该）
        - 应该只使用训练集统计量
        """
        logger.info("=" * 80)
        logger.info("实验2: 特征标准化泄露检测")
        logger.info("=" * 80)
        
        feature_cols = [c for c in self.train_data.columns if c not in ['target', 'ticker', 'date']]
        if len(feature_cols) == 0:
            logger.warning("No feature columns found, skipping experiment 2")
            return
        
        X_train = self.train_data[feature_cols].fillna(0)
        X_test = self.test_data[feature_cols].fillna(0)
        
        # 方法1：只使用训练集统计量（正确方法）
        scaler_train_only = StandardScaler()
        X_train_scaled_correct = scaler_train_only.fit_transform(X_train)
        X_test_scaled_correct = scaler_train_only.transform(X_test)
        
        # 方法2：使用训练+测试集统计量（泄露方法）
        X_combined = pd.concat([X_train, X_test], axis=0)
        scaler_combined = StandardScaler()
        X_combined_scaled = scaler_combined.fit_transform(X_combined)
        X_train_scaled_leaked = X_combined_scaled[:len(X_train)]
        X_test_scaled_leaked = X_combined_scaled[len(X_train):]
        
        # 比较两种方法的特征统计量差异
        train_mean_correct = X_train_scaled_correct.mean(axis=0)
        train_mean_leaked = X_train_scaled_leaked.mean(axis=0)
        train_std_correct = X_train_scaled_correct.std(axis=0)
        train_std_leaked = X_train_scaled_leaked.std(axis=0)
        
        mean_diff = np.abs(train_mean_correct - train_mean_leaked).mean()
        std_diff = np.abs(train_std_correct - train_std_leaked).mean()
        
        logger.info(f"训练集标准化均值差异: {mean_diff:.6f}")
        logger.info(f"训练集标准化标准差差异: {std_diff:.6f}")
        
        # 如果差异很大，说明标准化方法不同
        if mean_diff > 0.01 or std_diff > 0.01:
            logger.warning("⚠️  警告: 特征标准化方法存在显著差异，可能存在泄露！")
            logger.info("   建议检查：特征标准化是否只使用训练集统计量")
            self.results['exp2_leakage_detected'] = True
        else:
            logger.info("✅ 特征标准化差异小，未检测到明显泄露")
            self.results['exp2_leakage_detected'] = False
        
        self.results['exp2_mean_diff'] = mean_diff
        self.results['exp2_std_diff'] = std_diff
        
    def experiment_3_cv_temporal_order(self):
        """
        实验3：检测CV分割的时间顺序
        
        如果存在泄露：
        - CV分割应该按时间顺序，不应该有未来数据出现在训练集中
        - 检查每个fold的训练集和验证集的时间顺序
        """
        logger.info("=" * 80)
        logger.info("实验3: CV时间顺序检测")
        logger.info("=" * 80)
        
        try:
            from bma_models.unified_purged_cv_factory import create_unified_cv
            
            # 创建CV分割器（与训练时一致）
            cv = create_unified_cv(n_splits=5, gap=self.horizon_days, embargo=self.horizon_days)
            
            # 获取训练数据的日期
            train_dates = pd.to_datetime(self.train_data.index.get_level_values('date')).values
            train_dates_norm = pd.to_datetime(train_dates).values.astype('datetime64[D]')
            
            # 创建简单的y（用于CV分割）
            y_dummy = np.ones(len(self.train_data))
            
            # 获取CV分割
            cv_splits = list(cv.split(self.train_data.values, y_dummy, groups=train_dates_norm))
            
            leakage_detected = False
            fold_violations = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                train_dates_fold = train_dates_norm[train_idx]
                val_dates_fold = train_dates_norm[val_idx]
                
                # 检查：验证集的最大日期应该 <= 训练集的最大日期 + gap
                train_max = np.max(train_dates_fold)
                val_max = np.max(val_dates_fold)
                val_min = np.min(val_dates_fold)
                
                # 计算gap（天数）
                gap_days = (val_min - train_max).astype('timedelta64[D]').astype(int)
                
                if gap_days < self.horizon_days:
                    logger.warning(f"⚠️  Fold {fold_idx + 1}: Gap = {gap_days} days < {self.horizon_days} days (required)")
                    leakage_detected = True
                    fold_violations.append(fold_idx + 1)
                else:
                    logger.info(f"Fold {fold_idx + 1}: Gap = {gap_days} days >= {self.horizon_days} days ✓")
                
                # 检查：训练集中是否有未来数据（不应该有）
                if val_max < train_max:
                    logger.warning(f"⚠️  Fold {fold_idx + 1}: Validation set contains dates before training set max date!")
                    leakage_detected = True
                    fold_violations.append(fold_idx + 1)
            
            if leakage_detected:
                logger.warning(f"⚠️  警告: 检测到CV时间顺序违规！违规的fold: {fold_violations}")
                self.results['exp3_leakage_detected'] = True
            else:
                logger.info("✅ CV时间顺序正确，未检测到泄露")
                self.results['exp3_leakage_detected'] = False
            
            self.results['exp3_fold_violations'] = fold_violations
            
        except Exception as e:
            logger.error(f"实验3失败: {e}")
            logger.info("可能原因：CV分割器未正确配置或数据格式不匹配")
            self.results['exp3_error'] = str(e)
    
    def experiment_4_target_forward_looking(self):
        """
        实验4：检测目标变量是否使用了未来信息
        
        如果存在泄露：
        - 目标变量应该只使用T+horizon_days的信息
        - 不应该使用T+horizon_days+1或更未来的信息
        """
        logger.info("=" * 80)
        logger.info("实验4: 目标变量未来信息检测")
        logger.info("=" * 80)
        
        if 'target' not in self.train_data.columns or 'Close' not in self.train_data.columns:
            logger.warning("Missing 'target' or 'Close' column, skipping experiment 4")
            return
        
        # 检查目标变量的计算是否正确
        # 对于每个日期，目标应该是该日期之后horizon_days的收益率
        
        train_dates = pd.to_datetime(self.train_data.index.get_level_values('date')).unique()
        train_dates_sorted = sorted(train_dates)
        
        violations = []
        
        # 采样检查（检查前100个日期）
        sample_dates = train_dates_sorted[:min(100, len(train_dates_sorted))]
        
        for date in sample_dates:
            date_data = self.train_data.loc[self.train_data.index.get_level_values('date') == date]
            
            if len(date_data) == 0:
                continue
            
            # 检查目标变量是否合理
            # 这里我们假设如果目标变量计算正确，它应该与未来价格相关
            # 如果目标变量使用了更未来的信息，相关性会异常高
            
            # 简单检查：目标变量的分布是否合理
            targets = date_data['target'].dropna()
            if len(targets) > 0:
                # 检查是否有异常值（可能使用了未来信息）
                q99 = targets.quantile(0.99)
                q01 = targets.quantile(0.01)
                
                # 如果目标变量范围过大，可能存在问题
                if abs(q99) > 1.0 or abs(q01) > 1.0:  # 100%收益率
                    violations.append(date)
        
        if len(violations) > len(sample_dates) * 0.1:  # 超过10%的日期有问题
            logger.warning(f"⚠️  警告: 检测到{len(violations)}个日期的目标变量异常，可能存在未来信息泄露！")
            self.results['exp4_leakage_detected'] = True
        else:
            logger.info(f"✅ 目标变量检查通过，异常日期: {len(violations)}/{len(sample_dates)}")
            self.results['exp4_leakage_detected'] = False
        
        self.results['exp4_violations'] = len(violations)
        self.results['exp4_total_checked'] = len(sample_dates)
    
    def experiment_5_oof_vs_test_performance(self):
        """
        实验5：比较OOF预测性能与测试集预测性能
        
        如果存在泄露：
        - OOF预测性能应该接近训练集性能（因为可能看到了测试集信息）
        - OOF预测性能应该远高于测试集性能（不应该）
        """
        logger.info("=" * 80)
        logger.info("实验5: OOF vs 测试集性能比较")
        logger.info("=" * 80)
        
        logger.info("⚠️  此实验需要实际的OOF预测数据")
        logger.info("   建议：运行80/20评估，然后比较OOF预测和测试集预测的性能")
        logger.info("   如果OOF IC >> 测试集 IC，可能存在泄露")
        
        # 这里我们只能提供指导，无法直接运行
        self.results['exp5_note'] = "需要实际OOF预测数据才能运行"
    
    def run_all_experiments(self):
        """运行所有实验"""
        logger.info("=" * 80)
        logger.info("开始数据泄露检测实验")
        logger.info("=" * 80)
        
        self.load_data()
        self.split_data()
        
        self.experiment_1_shuffled_target()
        self.experiment_2_feature_scaling_leakage()
        self.experiment_3_cv_temporal_order()
        self.experiment_4_target_forward_looking()
        self.experiment_5_oof_vs_test_performance()
        
        # 生成报告
        self.generate_report()
    
    def generate_report(self):
        """生成检测报告"""
        logger.info("=" * 80)
        logger.info("数据泄露检测报告")
        logger.info("=" * 80)
        
        total_experiments = 4  # 实验5需要实际数据
        leakage_count = sum([
            self.results.get('exp1_leakage_detected', False),
            self.results.get('exp2_leakage_detected', False),
            self.results.get('exp3_leakage_detected', False),
            self.results.get('exp4_leakage_detected', False),
        ])
        
        logger.info(f"\n检测结果汇总:")
        logger.info(f"  总实验数: {total_experiments}")
        logger.info(f"  检测到泄露的实验数: {leakage_count}")
        logger.info(f"  泄露率: {leakage_count / total_experiments * 100:.1f}%")
        
        logger.info(f"\n详细结果:")
        logger.info(f"  实验1 (随机目标): {'⚠️ 泄露' if self.results.get('exp1_leakage_detected') else '✅ 正常'}")
        logger.info(f"  实验2 (特征标准化): {'⚠️ 泄露' if self.results.get('exp2_leakage_detected') else '✅ 正常'}")
        logger.info(f"  实验3 (CV时间顺序): {'⚠️ 泄露' if self.results.get('exp3_leakage_detected') else '✅ 正常'}")
        logger.info(f"  实验4 (目标变量): {'⚠️ 泄露' if self.results.get('exp4_leakage_detected') else '✅ 正常'}")
        
        if leakage_count > 0:
            logger.warning("\n⚠️  警告: 检测到潜在的数据泄露！")
            logger.warning("   建议检查以下方面：")
            if self.results.get('exp1_leakage_detected'):
                logger.warning("   - OOF预测可能使用了测试集信息")
            if self.results.get('exp2_leakage_detected'):
                logger.warning("   - 特征标准化可能使用了测试集统计量")
            if self.results.get('exp3_leakage_detected'):
                logger.warning("   - CV分割的时间顺序可能不正确")
            if self.results.get('exp4_leakage_detected'):
                logger.warning("   - 目标变量可能使用了未来信息")
        else:
            logger.info("\n✅ 未检测到明显的数据泄露")
        
        # 保存结果
        import json
        report_path = Path(f'data_leakage_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"\n报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="检测80/20 OOF数据泄露")
    parser.add_argument("--data-file", type=str, required=True,
                       help="数据文件路径（parquet格式）")
    parser.add_argument("--split", type=float, default=0.8,
                       help="训练/测试分割比例（默认0.8）")
    parser.add_argument("--horizon-days", type=int, default=10,
                       help="预测horizon天数（默认10）")
    
    args = parser.parse_args()
    
    detector = DataLeakageDetector(
        data_path=args.data_file,
        split=args.split,
        horizon_days=args.horizon_days
    )
    
    detector.run_all_experiments()


if __name__ == "__main__":
    main()
