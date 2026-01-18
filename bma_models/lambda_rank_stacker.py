#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LambdaRank Stacker - 专门优化排序的二层模型

核心特性：
- 使用LightGBM的LambdaRank目标，专门优化NDCG@K
- 连续目标 → 组内分位数 → 整数等级转换
- 交易日作为排序组，符合实际选股场景
- 与Ridge Stacker并行训练，互为补充

设计理念：
Ridge回归 -> 连续预测，保留刻度信息
LambdaRank -> 排序优化，提升Top-K性能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# 导入PurgedCV防止数据泄露
try:
    from bma_models.unified_purged_cv_factory import create_unified_cv
    PURGED_CV_AVAILABLE = True
except ImportError:
    PURGED_CV_AVAILABLE = False

logger = logging.getLogger(__name__)

class LambdaRankStacker:
    """
    LambdaRank排序模型 - 直接使用Alpha Factors优化排序

    核心优势：
    - 直接使用原始Alpha factors训练，不依赖第一层OOF
    - 专门优化排序指标（NDCG@K）
    - 组内等级转换，适合Top-K选股
    - 与Stacking模型形成互补融合
    """

    def __init__(self,
                 base_cols: Tuple[str, ...] = None,  # 将自动使用alpha factor columns
                 n_quantiles: int = 128,  # 固定档位数量（64或128）
                 winsorize_quantiles: Tuple[float, float] = (0.01, 0.99),  # 异常值截断
                 label_gain_power: float = 1.0,  # 标签增益幂次（1.0=线性，1.5=强化前排）
                 lgb_params: Optional[Dict[str, Any]] = None,
                 num_boost_round: int = 260,  # Updated: 260
                 early_stopping_rounds: int = 60,  # Updated: 60
                 use_purged_cv: bool = True,  # 强制使用PurgedCV防止数据泄露（当internal CV启用时）
                 use_internal_cv: bool = True,  # 是否在fit内部执行PurgedCV（外层已有CV时可禁用以避免fold-in-fold）
                 cv_n_splits: int = 6,        # 🔥 CV折数（T+5: 6折，提高数据利用率）
                 cv_gap_days: int = 5,        # 🔥 T+5预测：gap=5（与horizon对齐）
                 cv_embargo_days: int = 5,    # 🔥 T+5预测：embargo=5（与horizon对齐）
                 random_state: int = 42):
        """
        初始化LambdaRank排序模型

        Args:
            base_cols: 特征列名（None时自动检测alpha factor列）
            n_quantiles: 固定档位数量（64/128，稳定标签构造）
            winsorize_quantiles: 异常值截断分位数（防极值影响）
            label_gain_power: 标签增益幂次（1.0=线性，1.5=强化高档位）
            lgb_params: LightGBM参数
            num_boost_round: 训练轮数
            early_stopping_rounds: 早停轮数（增强防过拟合）
            use_purged_cv: 是否使用PurgedCV（强烈推荐True）
            cv_n_splits: 交叉验证折数（T+5: 6折）
            cv_gap_days: CV间隙天数（T+5: 5天，防数据泄露）
            cv_embargo_days: CV禁运天数（T+5: 5天，防前视偏误）
            random_state: 随机种子
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for LambdaRankStacker")
        if use_internal_cv and not use_purged_cv:
            raise ValueError("When internal CV is enabled, purged CV must be enabled for T+5 training.")
        if not PURGED_CV_AVAILABLE:
            raise RuntimeError("Unified Purged CV factory is unavailable. Install the required components to enable T+5 training.")
        if use_internal_cv and (cv_n_splits, cv_gap_days, cv_embargo_days) != (6, 5, 5):
            raise ValueError("LambdaRankStacker enforces T+5 CV settings when internal CV is enabled: splits=6, gap=5, embargo=5.")

        # base_cols在fit时会自动检测为alpha factor columns
        self.base_cols = base_cols
        self._alpha_factor_cols = None  # 存储实际使用的alpha factor列
        self.n_quantiles = n_quantiles
        self.winsorize_quantiles = winsorize_quantiles
        self.label_gain_power = label_gain_power
        self.num_boost_round = num_boost_round
        # 训练过小样本时限制，防止 best_iteration 过高
        self.early_stopping_rounds = max(early_stopping_rounds, 100)
        self.use_internal_cv = bool(use_internal_cv)
        self.use_purged_cv = bool(use_purged_cv)
        self.cv_n_splits = 6  # 🔥 使用固定的n_splits（T+5: 6）
        self.cv_gap_days = 5
        self.cv_embargo_days = 5
        self.random_state = random_state

        # 生成label_gain序列（支持幂次增强）
        if label_gain_power == 1.0:
            self.label_gain = list(range(n_quantiles))  # 线性增益: [0,1,2,...,63]
        else:
            # 幂次增益强化高档位: [(i/N)^power * N for i in range(N)]
            self.label_gain = [(i / (n_quantiles - 1)) ** label_gain_power * (n_quantiles - 1)
                              for i in range(n_quantiles)]

        # 专业级LambdaRank参数（使用传入的lgb_params覆盖默认值，确保YAML配置生效）
        # 默认值仅作为fallback，实际值应从YAML配置传入
        default_lgb_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [10, 30],  # Default: Top-10/30 (can be overridden by lgb_params)
            'label_gain': self.label_gain,  # 关键：固定档位增益
            'num_leaves': 127,  # Default (can be overridden by lgb_params from YAML)
            'max_depth': 6,   # Default (can be overridden by lgb_params from YAML)
            'learning_rate': 0.03,  # Default (can be overridden by lgb_params from YAML)
            'feature_fraction': 0.85,  # Default (can be overridden by lgb_params from YAML)
            'bagging_fraction': 0.8,  # Default
            'bagging_freq': 1,  # 每轮都bagging
            'min_data_in_leaf': 380,  # Updated: 380
            'lambda_l1': 0.0,  # Disabled L1 regularization
            'lambda_l2': 10.0,  # Updated: 10.0
            'lambdarank_truncation_level': 650,  # Updated: 650
            'sigmoid': 1.2,  # Sigmoid参数
            'verbose': -1,
            'random_state': random_state,
            'force_col_wise': True
        }
        
        # 先设置默认值
        self.lgb_params = default_lgb_params.copy()
        
        # 然后用传入的lgb_params覆盖（确保YAML配置生效）
        if lgb_params:
            self.lgb_params.update(lgb_params)

        # 模型状态
        self.model = None
        self.scaler = None
        self.fitted_ = False
        self._oof_predictions = None  # OOF预测（防数据泄漏）
        self._oof_index = None  # OOF索引（用于对齐）

        logger.info("🏆 LambdaRank 排序模型初始化完成")
        logger.info(f"   特征模式: {'Alpha Factors' if self.base_cols is None else 'Custom'}")
        logger.info(f"   分位数等级: {self.n_quantiles}")
        logger.info(f"   Label gain power: {self.label_gain_power}")
        logger.info(f"   NDCG评估: {self.lgb_params['ndcg_eval_at']}")
        logger.info(f"   训练轮数: {self.num_boost_round}, 早停: {self.early_stopping_rounds}")
        logger.info(f"   模型容量: num_leaves={self.lgb_params.get('num_leaves')}, max_depth={self.lgb_params.get('max_depth')}, min_data_in_leaf={self.lgb_params.get('min_data_in_leaf')}")
        logger.info(f"   采样: feature_fraction={self.lgb_params.get('feature_fraction')}, bagging_fraction={self.lgb_params.get('bagging_fraction')}")
        logger.info(f"   LambdaRank: truncation_level={self.lgb_params.get('lambdarank_truncation_level')}, sigmoid={self.lgb_params.get('sigmoid')}")
        logger.info(f"   内部CV: {'启用' if self.use_internal_cv else '禁用'}")
        if self.use_internal_cv and self.use_purged_cv:
            logger.info(f"   CV参数: splits={self.cv_n_splits}, gap={self.cv_gap_days}天, embargo={self.cv_embargo_days}天")

    def _convert_to_rank_labels(self, df: pd.DataFrame, target_col: str = 'ret_fwd_10d') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        将连续目标变量转换为稳定的固定档位等级（64/128档软离散）

        核心改进：
        1. 异常值截断 (winsorize) 提升鲁棒性
        2. rank(pct=True) → floor(rank_pct * N) 固定档位
        3. 避免pd.qcut的bin合并和不稳定问题
        4. 完全确定性的等级分配

        Args:
            df: 包含目标变量的DataFrame
            target_col: 目标变量列名

        Returns:
            处理后的DataFrame，转换报告
        """
        logger.info(f"🔄 开始稳定标签构造: {target_col} → 固定{self.n_quantiles}档软离散")

        df_processed = df.copy()

        def _stable_group_rank_transform(group):
            """稳定的组内固定档位转换"""
            target_values = group[target_col].dropna()
            if len(target_values) <= 1:
                # 单个样本组，设为中位等级
                group[f'{target_col}_rank'] = self.n_quantiles // 2
                return group

            # 步骤1: Winsorize异常值截断（提升鲁棒性）
            lower_q, upper_q = self.winsorize_quantiles
            lower_bound = target_values.quantile(lower_q)
            upper_bound = target_values.quantile(upper_q)
            target_winsorized = target_values.clip(lower_bound, upper_bound)

            # 步骤2: 组内百分位排序 [0, 1]
            rank_pct = target_winsorized.rank(pct=True, method='average')

            # 步骤3: 固定档位映射 [0, N-1]
            # rank_pct ∈ (0, 1] → label ∈ [0, N-1]
            labels = np.floor(rank_pct * self.n_quantiles).astype(int)
            labels[labels == self.n_quantiles] = self.n_quantiles - 1  # 处理rank_pct=1的边界情况

            # 步骤4: 映射回原DataFrame结构
            full_ranks = pd.Series(self.n_quantiles // 2, index=group.index)  # 默认中位等级
            full_ranks.loc[target_values.index] = labels
            group[f'{target_col}_rank'] = full_ranks.astype(int)

            return group

        # 按交易日分组进行稳定等级转换（避免列名/索引歧义，使用索引层名）
        df_processed = df_processed.groupby(level='date', group_keys=False).apply(_stable_group_rank_transform)

        # 验证转换结果
        rank_col = f'{target_col}_rank'
        unique_ranks = df_processed[rank_col].nunique()
        rank_distribution = df_processed[rank_col].value_counts().sort_index()

        # 检查异常值截断效果
        original_values = df[target_col].dropna()
        winsorized_count = 0
        if len(original_values) > 0:
            lower_q, upper_q = self.winsorize_quantiles
            lower_bound = original_values.quantile(lower_q)
            upper_bound = original_values.quantile(upper_q)
            winsorized_count = ((original_values < lower_bound) | (original_values > upper_bound)).sum()

        logger.info(f"✅ 稳定标签构造完成: 固定{unique_ranks}档位")
        logger.info(f"   异常值截断: {winsorized_count}/{len(original_values)} ({winsorized_count/len(original_values)*100:.1f}%)")
        logger.info(f"   等级分布: {dict(list(rank_distribution.items())[:5])}...")

        conversion_report = {
            'n_quantiles_configured': self.n_quantiles,
            'n_quantiles_used': unique_ranks,
            'rank_distribution': dict(rank_distribution),
            'winsorized_count': winsorized_count,
            'winsorized_rate': winsorized_count / len(original_values) if len(original_values) > 0 else 0.0,
            'conversion_coverage': 1.0 - df_processed[rank_col].isna().mean(),
            'label_gain_type': 'linear' if self.label_gain_power == 1.0 else f'power_{self.label_gain_power}'
        }

        return df_processed, conversion_report

    def fit(self, df: pd.DataFrame, target_col: str = 'ret_fwd_10d', alpha_factors: pd.DataFrame = None) -> 'LambdaRankStacker':
        """
        训练LambdaRank模型

        Args:
            df: 训练数据，必须包含MultiIndex(date, ticker)和target
            target_col: 目标变量列名
            alpha_factors: Alpha因子DataFrame（如果为None，将从df中自动检测）

        Returns:
            self
        """
        logger.info("🚀 开始训练LambdaRank排序模型（使用Alpha Factors）...")

        # 🔧 统一输入处理：确保使用检测到的MultiIndex
        # 验证输入格式
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("DataFrame必须有MultiIndex(date, ticker)")
        
        # 🔧 验证MultiIndex格式正确（date, ticker）
        if df.index.names != ['date', 'ticker']:
            logger.warning(f"MultiIndex名称不匹配: {df.index.names}，期望: ['date', 'ticker']")
            # 尝试修复：如果只有两层，重命名
            if df.index.nlevels == 2:
                df.index.names = ['date', 'ticker']
                logger.info("✅ 已修复MultiIndex名称")
            else:
                raise ValueError(f"MultiIndex格式不正确: names={df.index.names}, levels={df.index.nlevels}")

        if target_col not in df.columns:
            # 严格模式：不进行任何回退，要求上游明确提供正确的目标列
            raise ValueError(f"目标变量 {target_col} 不存在")

        # 🔧 统一输入处理：与其他模型（ElasticNet/XGBoost/CatBoost）保持一致
        # 优先级：base_cols > alpha_factors > 自动检测
        if self.base_cols is not None and len(self.base_cols) > 0:
            # 使用指定的列（与其他模型一致的特征选择）
            self._alpha_factor_cols = [col for col in self.base_cols if col in df.columns]
            missing_cols = [col for col in self.base_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"特征列不存在: {missing_cols}")
            logger.info(f"   使用指定的特征列: {len(self._alpha_factor_cols)}个因子（与其他模型一致）")
        elif alpha_factors is not None:
            # 使用提供的alpha factors
            if not isinstance(alpha_factors, pd.DataFrame):
                raise ValueError("alpha_factors必须是pandas DataFrame")

            self._alpha_factor_cols = [col for col in alpha_factors.columns if col != target_col]

            if len(self._alpha_factor_cols) == 0:
                raise ValueError("alpha_factors中没有找到有效的特征列")

            # 确保索引对齐
            try:
                df = pd.concat([df[[target_col]], alpha_factors[self._alpha_factor_cols]], axis=1)
            except Exception as e:
                raise ValueError(f"无法合并DataFrame和alpha_factors: {e}")

            logger.info(f"   使用提供的Alpha Factors: {len(self._alpha_factor_cols)}个因子")
        else:
            # 自动检测alpha factor列（排除target和pred_开头的列）
            exclude_patterns = [target_col, 'pred_', 'lambda_', 'ridge_', 'final_', 'rank', 'weight']
            self._alpha_factor_cols = [col for col in df.columns
                                      if not any(pattern in col.lower() for pattern in exclude_patterns)]
            logger.info(f"   自动检测到{len(self._alpha_factor_cols)}个Alpha Factors")
            logger.info(f"   前5个因子: {self._alpha_factor_cols[:5]}")

        # 更新base_cols为实际使用的列
        self.base_cols = tuple(self._alpha_factor_cols)

        # 🔧 统一输入处理：确保使用与其他模型相同的样本和特征
        # 注意：上游训练循环已经处理了NaN和索引对齐，这里只做最小必要的验证
        
        # 转换为组内等级标签
        df_processed, conversion_report = self._convert_to_rank_labels(df, target_col)
        rank_col = f'{target_col}_rank'

        # 准备特征和标签（使用指定的特征列，确保与其他模型一致）
        # 确保特征列顺序与base_cols一致
        feature_cols = [col for col in self._alpha_factor_cols if col in df_processed.columns]
        if len(feature_cols) != len(self._alpha_factor_cols):
            missing = set(self._alpha_factor_cols) - set(feature_cols)
            raise ValueError(f"特征列缺失: {missing}")
        
        X = df_processed[feature_cols].values
        y = df_processed[rank_col].values

        # 准备分组信息（每个交易日为一个组）
        date_index = df_processed.index.get_level_values('date')
        unique_dates = date_index.unique()
        group_sizes = [len(df_processed.loc[date]) for date in unique_dates]

        logger.info(f"   训练样本: {len(X)}")
        logger.info(f"   特征维度: {X.shape[1]} (特征列: {feature_cols[:3]}...)")
        logger.info(f"   交易日组数: {len(group_sizes)}")
        logger.info(f"   平均组大小: {np.mean(group_sizes):.1f}")

        # 🔧 统一NaN处理：与其他模型保持一致
        # 上游训练循环已经处理了NaN，但这里仍需要处理可能的NaN（来自标签转换）
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        # 记录过滤的样本数（用于调试）
        if not valid_mask.all():
            logger.info(f"   过滤NaN样本: {len(X)} -> {len(X_valid)} ({len(X_valid)/len(X)*100:.1f}%)")

        # 小样本自适应：放宽最小样本限制并动态调整LightGBM参数
        min_required = max(30, self.cv_n_splits * 2)
        if len(X_valid) < min_required:
            logger.warning(
                f"有效训练样本过少: {len(X_valid)} < {min_required}，启用小样本自适应参数以继续训练"
            )
            small_n = int(len(X_valid))
            # 动态降低复杂度，避免叶子样本要求过高导致训练失败
            self.lgb_params['min_data_in_leaf'] = max(1, small_n // 5)
            self.lgb_params['num_leaves'] = min(self.lgb_params.get('num_leaves', 31), max(7, small_n // 2))
            self.lgb_params['max_depth'] = min(self.lgb_params.get('max_depth', 6), 6)
            self.lgb_params['learning_rate'] = min(self.lgb_params.get('learning_rate', 0.1), 0.1)
            # 缩短训练轮数以防过拟合和过长训练
            self.num_boost_round = min(self.num_boost_round, 100)
        elif len(X_valid) < 200:
            # 中等小样本的温和自适应
            small_n = int(len(X_valid))
            self.lgb_params['min_data_in_leaf'] = max(5, min(self.lgb_params.get('min_data_in_leaf', 50), small_n // 4))

        # 重新计算组大小（基于有效样本）
        df_valid = df_processed.iloc[valid_mask]
        valid_date_index = df_valid.index.get_level_values('date')
        valid_unique_dates = valid_date_index.unique()
        valid_group_sizes = [len(df_valid.loc[date]) for date in valid_unique_dates]

        logger.info(f"   有效样本: {len(X_valid)} ({len(X_valid)/len(X)*100:.1f}%)")

        # 特征标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_valid)

        logger.info(f"   特征标准化: 均值={X_scaled.mean(axis=0)[:3]}, 标准差={X_scaled.std(axis=0)[:3]}")

        if self.use_internal_cv:
            logger.info("🏋️ 开始PurgedCV LambdaRank训练（防数据泄露）...")
            self.model = self._train_with_purged_cv(
                X_scaled, y_valid, df_valid, valid_group_sizes
            )
        else:
            logger.info("🏋️ 内部CV已禁用：在外层CV的训练子集上进行全量训练（无内部分折）")
            # 构建LightGBM数据集（将所有样本视为一个大组；排序学习仍可运行，但主要依赖外层CV评估）
            # 更严谨做法是按date分组构建单个训练数据的group数组
            dates = df_valid.index.get_level_values('date')
            unique_dates = dates.unique()
            train_group_sizes = [len(dates[dates == d]) for d in unique_dates]
            train_data = lgb.Dataset(
                X_scaled, label=y_valid, group=train_group_sizes,
                feature_name=[f'f_{i}' for i in range(X_scaled.shape[1])]
            )
            callbacks = [lgb.log_evaluation(period=0)]
            if self.early_stopping_rounds > 0:
                # 无验证集时不使用早停
                pass
            logger.info(f"🔧 [LambdaRank最终训练] 使用参数: feature_fraction={self.lgb_params.get('feature_fraction')}, min_data_in_leaf={self.lgb_params.get('min_data_in_leaf')}, lambdarank_truncation_level={self.lgb_params.get('lambdarank_truncation_level')}, label_gain_power={self.label_gain_power}")
            self.model = lgb.train(
                self.lgb_params,
                train_data,
                num_boost_round=self.num_boost_round,
                valid_sets=[],
                callbacks=callbacks
            )

        # 训练后评估
        train_predictions = self.model.predict(X_scaled)

        # 计算NDCG指标 - 扩展K值以适应2600股票宇宙
        ndcg_scores = {}
        # 分层评估：核心选股 + 投资组合构建 + 风险分散
        k_values = [5, 10, 20, 50, 100]  # 最大到100，考虑计算效率
        for k in k_values:
            if k <= max(valid_group_sizes):
                ndcg_scores[f'NDCG@{k}'] = self._calculate_ndcg(y_valid, train_predictions, valid_group_sizes, k)

        self.fitted_ = True

        logger.info("✅ LambdaRank排序模型训练完成（基于Alpha Factors）")
        logger.info(f"   最佳迭代: {self.model.best_iteration}")
        logger.info(f"   NDCG指标: {ndcg_scores}")
        logger.info(f"   特征重要性: {dict(zip(self.base_cols, self.model.feature_importance()[:len(self.base_cols)]))}")

        return self

    def _train_with_purged_cv(self, X_scaled: np.ndarray, y_valid: np.ndarray,
                             df_valid: pd.DataFrame, group_sizes: list) -> lgb.Booster:
        """
        使用PurgedCV训练LambdaRank模型（防止数据泄露）

        Args:
            X_scaled: 标准化后的特征
            y_valid: 有效的目标变量（等级）
            df_valid: 有效的DataFrame（包含日期索引）
            group_sizes: 每个交易日的组大小

        Returns:
            训练好的LightGBM模型
        """
        # 创建PurgedCV
        cv_splitter = create_unified_cv(
            n_splits=self.cv_n_splits,
            gap=self.cv_gap_days,
            embargo=self.cv_embargo_days
        )

        # 获取日期序列用于CV分割
        dates = df_valid.index.get_level_values('date')
        unique_dates = sorted(dates.unique())

        logger.info(f"   PurgedCV: {self.cv_n_splits}折, gap={self.cv_gap_days}天, embargo={self.cv_embargo_days}天")
        logger.info(f"   数据时间范围: {unique_dates[0]} ~ {unique_dates[-1]} ({len(unique_dates)}天)")

        # 执行CV训练
        cv_models = []
        cv_scores = []
        oof_predictions = np.zeros(len(X_scaled))  # 初始化OOF数组（防数据泄漏）

        # 为CV创建日期索引映射
        date_to_idx = {date: i for i, date in enumerate(unique_dates)}
        sample_date_indices = [date_to_idx[date] for date in dates]

        cv_splits = list(cv_splitter.split(X_scaled, y_valid, groups=sample_date_indices))
        if not cv_splits:
            raise RuntimeError('PurgedCV did not yield any splits for LambdaRankStacker.')
        logger.info(f"   成功生成{len(cv_splits)}个CV分割")

        # 遍历CV分割进行训练
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            logger.info(f"   CV Fold {fold_idx + 1}/{len(cv_splits)}: 训练={len(train_idx)}, 验证={len(val_idx)}")

            # 分割训练和验证数据
            X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
            y_train_fold, y_val_fold = y_valid[train_idx], y_valid[val_idx]

            # 重新计算训练集的组大小
            train_dates = dates[train_idx]
            train_unique_dates = train_dates.unique()
            train_group_sizes = [len(train_dates[train_dates == date]) for date in train_unique_dates]

            # 重新计算验证集的组大小
            val_dates = dates[val_idx]
            val_unique_dates = val_dates.unique()
            val_group_sizes = [len(val_dates[val_dates == date]) for date in val_unique_dates]

            if len(train_group_sizes) == 0 or len(val_group_sizes) == 0:
                logger.warning(f"   CV Fold {fold_idx + 1}: 训练或验证集为空，跳过")
                continue

            # 创建LightGBM数据集
            train_data = lgb.Dataset(
                X_train_fold, label=y_train_fold, group=train_group_sizes,
                feature_name=[f'f_{i}' for i in range(X_scaled.shape[1])]
            )
            val_data = lgb.Dataset(
                X_val_fold, label=y_val_fold, group=val_group_sizes,
                reference=train_data
            )

            # 训练模型
            callbacks = [lgb.log_evaluation(period=0)]  # 静默训练
            if self.early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(self.early_stopping_rounds))

            logger.info(f"🔧 [LambdaRank CV Fold {fold_idx+1}] 使用参数: feature_fraction={self.lgb_params.get('feature_fraction')}, min_data_in_leaf={self.lgb_params.get('min_data_in_leaf')}, lambdarank_truncation_level={self.lgb_params.get('lambdarank_truncation_level')}, label_gain_power={self.label_gain_power}")
            model = lgb.train(
                self.lgb_params,
                train_data,
                num_boost_round=self.num_boost_round,
                valid_sets=[val_data],
                valid_names=['val'],
                callbacks=callbacks
            )

            cv_models.append(model)

            # 计算验证集NDCG - 使用NDCG@50作为主要CV指标
            val_pred = model.predict(X_val_fold)

            # 保存OOF预测（关键：防止数据泄漏）
            oof_predictions[val_idx] = val_pred
            if len(val_group_sizes) > 0:
                # 根据数据量选择合适的主要评估指标
                max_group_size = max(val_group_sizes)
                primary_k = min(50, max_group_size) if max_group_size >= 50 else min(20, max_group_size)

                ndcg_score = self._calculate_ndcg(y_val_fold, val_pred, val_group_sizes, primary_k)
                cv_scores.append(ndcg_score)

                # 多层次评估报告
                if max_group_size >= 50:
                    ndcg5 = self._calculate_ndcg(y_val_fold, val_pred, val_group_sizes, 5)
                    ndcg20 = self._calculate_ndcg(y_val_fold, val_pred, val_group_sizes, 20)
                    logger.info(f"   CV Fold {fold_idx + 1}: NDCG@5={ndcg5:.4f}, @20={ndcg20:.4f}, @50={ndcg_score:.4f}")
                else:
                    ndcg5 = self._calculate_ndcg(y_val_fold, val_pred, val_group_sizes, 5)
                    logger.info(f"   CV Fold {fold_idx + 1}: NDCG@5={ndcg5:.4f}, @{primary_k}={ndcg_score:.4f}")

        if cv_models:
            primary_k_desc = "50" if max([max(fold_sizes) for fold_sizes in [val_group_sizes] if fold_sizes]) >= 50 else "20"
            logger.info(f"   CV平均NDCG@{primary_k_desc}: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

            # 保存OOF预测和索引（用于后续融合）
            self._oof_predictions = oof_predictions
            self._oof_index = pd.Series(dates).reset_index(drop=True)  # 保存原始索引
            logger.info(f"   ✓ OOF预测已生成: {len(oof_predictions)} 个样本")

            # 返回最后一个模型（见过最多数据）
            return cv_models[-1]
        else:
            raise RuntimeError("所有CV fold都失败，无法训练模型")

    def get_oof_predictions(self, df: pd.DataFrame) -> pd.Series:
        """
        获取OOF预测（Out-of-Fold predictions）

        重要：这是真正的OOF预测，每个样本只被未见过它的模型预测，防止数据泄漏。

        Args:
            df: 原始训练数据（用于提取MultiIndex）

        Returns:
            OOF预测Series（带MultiIndex: date, ticker）

        Raises:
            RuntimeError: 如果OOF预测未生成（模型未使用CV训练）
            ValueError: 如果df没有MultiIndex或索引长度不匹配
        """
        if self._oof_predictions is None:
            raise RuntimeError("OOF预测未生成，可能模型未使用CV训练")

        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("df必须有MultiIndex(date, ticker)")

        if len(self._oof_predictions) != len(df):
            raise ValueError(
                f"OOF预测长度({len(self._oof_predictions)})与df长度({len(df)})不匹配"
            )

        # 创建Series（使用df的MultiIndex）
        oof_series = pd.Series(
            self._oof_predictions,
            index=df.index,
            name='lambda_oof'
        )

        logger.info(f"✓ 返回Lambda OOF预测: {len(oof_series)} 个样本")
        return oof_series

    def predict(self, df: pd.DataFrame, alpha_factors: pd.DataFrame = None) -> pd.DataFrame:
        """
        使用LambdaRank模型预测

        Args:
            df: 预测数据
            alpha_factors: Alpha因子DataFrame（如果为None，将从df中提取）

        Returns:
            包含预测分数的DataFrame
        """
        if not self.fitted_:
            raise RuntimeError("模型未训练，请先调用fit()")

        logger.info("📊 LambdaRank排序模型开始预测（基于Alpha Factors）...")

        # 验证输入并确保列顺序一致
        if alpha_factors is not None:
            # 使用提供的alpha factors
            df_clean = alpha_factors.copy()
        else:
            df_clean = df.copy()

        # 使用训练时的alpha factor列
        if self._alpha_factor_cols is None:
            raise RuntimeError("Alpha factor列未设置，请先训练模型")

        # 检查必需的列
        missing_cols = [col for col in self._alpha_factor_cols if col not in df_clean.columns]
        if missing_cols:
            logger.warning(f"缺少{len(missing_cols)}个Alpha factor列: {missing_cols[:5]}")
            # 尝试容错处理
            available_cols = [col for col in self._alpha_factor_cols if col in df_clean.columns]
            if len(available_cols) < len(self._alpha_factor_cols) * 0.5:
                raise ValueError(f"可用Alpha factor列太少: {len(available_cols)}/{len(self._alpha_factor_cols)}")
            X = df_clean[available_cols].values
            logger.info(f"   使用{len(available_cols)}个可用Alpha factors进行预测")
        else:
            # 提取特征
            X = df_clean[list(self._alpha_factor_cols)].values

        # 处理NaN
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]

        if len(X_valid) == 0:
            raise ValueError("所有样本都包含NaN，无法预测")

        # 使用训练时拟合的标准化器
        X_scaled = self.scaler.transform(X_valid)

        logger.info(f"   特征提取: {len(X_valid)} 有效样本")

        # 预测
        raw_predictions = self.model.predict(X_scaled)

        # 创建完整预测数组
        full_predictions = np.full(len(X), np.nan)
        full_predictions[valid_mask] = raw_predictions

        # 构建结果DataFrame
        result = df_clean.copy()
        result['lambda_score'] = full_predictions

        # 按日期计算排名（LambdaRank的核心输出）
        def _rank_by_date(group):
            scores = group['lambda_score']
            valid_scores = scores.dropna()
            if len(valid_scores) <= 1:
                return pd.Series(np.nan, index=scores.index)

            ranks = valid_scores.rank(method='average', ascending=False)
            full_ranks = pd.Series(np.nan, index=scores.index)
            full_ranks.loc[valid_scores.index] = ranks
            return full_ranks

        # 🔧 FIX: 保持正确的索引顺序，避免groupby.apply产生多层索引
        ranked_series = result.groupby(level='date')['lambda_score'].rank(method='average', ascending=False)
        result['lambda_rank'] = ranked_series

        # 计算组内百分位（用于后续Copula正态化）
        pct_series = result.groupby(level='date')['lambda_score'].rank(pct=True)
        result['lambda_pct'] = pct_series

        logger.info(f"✅ LambdaRank预测完成: 覆盖率={valid_mask.mean():.1%}")

        return result[['lambda_score', 'lambda_rank', 'lambda_pct']]

    def _calculate_ndcg(self, y_true: np.ndarray, y_pred: np.ndarray, group_sizes: list, k: int) -> float:
        """计算NDCG@K指标"""
        try:
            from sklearn.metrics import ndcg_score

            # 将预测分组
            start_idx = 0
            ndcg_scores = []

            for group_size in group_sizes:
                if group_size < k:
                    continue

                end_idx = start_idx + group_size
                group_true = y_true[start_idx:end_idx]
                group_pred = y_pred[start_idx:end_idx]

                # 计算NDCG@K
                ndcg = ndcg_score(
                    group_true.reshape(1, -1),
                    group_pred.reshape(1, -1),
                    k=k
                )
                ndcg_scores.append(ndcg)

                start_idx = end_idx

            return np.mean(ndcg_scores) if ndcg_scores else 0.0

        except ImportError:
            # Fallback：简单的排序相关性
            return 0.0

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.fitted_:
            return {'fitted': False}

        return {
            'fitted': True,
            'model_type': 'LambdaRank',
            'best_iteration': self.model.best_iteration,
            'feature_importance': dict(zip(self.base_cols, self.model.feature_importance()[:len(self.base_cols)])),
            'n_quantiles': self.n_quantiles,
            'lgb_params': self.lgb_params
        }
