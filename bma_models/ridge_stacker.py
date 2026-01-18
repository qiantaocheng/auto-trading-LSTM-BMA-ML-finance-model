#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ridge Regression Stacker - 替换LTR Isotonic Stacker
简洁高效的二层线性回归模型，直接优化连续收益率
✅ 增强CV验证：使用时间序列验证防止过拟合
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

# 🔥 导入统一PurgedCV，防止数据泄露
try:
    from bma_models.unified_purged_cv_factory import create_unified_cv
    from bma_models.unified_config_loader import get_time_config
    PURGED_CV_AVAILABLE = True
except ImportError:
    PURGED_CV_AVAILABLE = False

logger = logging.getLogger(__name__)

class RidgeStacker:
    """
    Ridge回归二层Stacker - 简洁替代LTR

    核心优势：
    - 直接优化连续收益率，无信息损失
    - 线性模型，解释性强
    - 训练快速，稳定性好
    - 自动特征标准化
    - ✅ 时间序列CV：防止过拟合，提升泛化性能
    """

    def __init__(self,
                 base_cols: Tuple[str, ...] = ('pred_catboost', 'pred_elastic', 'pred_xgb', 'pred_lightgbm_ranker'),
                 alpha: float = 1.0,
                 fit_intercept: bool = False,
                 solver: str = "auto",
                 tol: float = 1e-6,
                 auto_tune_alpha: bool = False,
                 alpha_grid: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0),
                 use_cv: bool = True,
                 use_purged_cv: bool = True,  # 🔥 使用PurgedCV防止数据泄露
                 cv_splits: int = 6,  # 🔥 T+5: 6折CV
                 cv_gap_days: int = 5,  # 🔥 T+5: gap=5
                 cv_embargo_days: int = 5,  # 🔥 T+5: embargo=5
                 cv_test_size: float = 0.2,
                 use_lambda_percentile: bool = True,  # 新增：使用Lambda percentile特征
                 # ---- Direction calibration & feature augmentation (new) ----
                 direction_calibration: bool = False,  # 🔥 DISABLED: IC signs reverse across regime changes
                 direction_calibration_min_n: int = 30,
                 add_rank_features: bool = True,
                 random_state: int = 42,
                 # ---- Nested CV & constraints (new) ----
                 use_nested_cv: bool = False,
                 nested_outer_splits: int = 4,
                 nested_gap_days: int = 5,
                 nested_embargo_days: int = 5,
                 aggregate_alpha: str = 'median',  # 'median' | 'ir_weighted'
                 use_convex_constraint: bool = True,  # 非负且和为1
                 # 🔥 新增: 极端target过滤
                 filter_extreme_targets: bool = True,  # 是否过滤极端target
                 extreme_lower_pct: float = 0.5,       # 过滤下界百分位 (移除WW -88%等)
                 extreme_upper_pct: float = 99.5,      # 过滤上界百分位 (移除WW 9900%等)
                 **kwargs):
        """
        初始化Ridge Stacker

        Args:
            base_cols: 第一层模型预测列名
            alpha: Ridge正则化强度 (默认1.0，简洁版)
            fit_intercept: 是否拟合截距 (默认False，因为已做z-score)
            solver: 求解器 (默认auto)
            tol: 收敛容差 (默认1e-6)
            auto_tune_alpha: 是否自动调参 (默认False，保持简洁)
            alpha_grid: 调参网格 (默认[0.5,1,2,3,5,8])
            use_cv: 是否使用交叉验证 (默认True)
            use_purged_cv: 🔥 是否使用PurgedCV (默认True，防止数据泄露)
            cv_splits: CV折数 (默认6，T+5优化)
            cv_gap_days: 🔥 CV gap天数 (默认5，T+5)
            cv_embargo_days: 🔥 CV embargo天数 (默认5，T+5)
            cv_test_size: 每折验证集比例 (默认0.2)
            use_lambda_percentile: 是否使用Lambda percentile特征 (默认True)
            random_state: 随机种子
        """
        if not use_cv:
            raise ValueError("RidgeStacker requires cross-validation; disabling CV violates the enforced T+5 protocol.")
        if not use_purged_cv:
            raise ValueError("RidgeStacker requires purged CV to protect the T+5 horizon; do not disable it.")
        if not PURGED_CV_AVAILABLE:
            raise RuntimeError("Unified Purged CV factory is unavailable. Install the required components to enable T+5 training.")
        if (cv_splits, cv_gap_days, cv_embargo_days) != (6, 5, 5):
            raise ValueError("RidgeStacker enforces T+5 CV settings: splits=6, gap=5, embargo=5.")

        self.base_cols = base_cols
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.tol = tol
        self.auto_tune_alpha = auto_tune_alpha
        self.alpha_grid = alpha_grid
        self.use_cv = True
        self.use_purged_cv = True
        self.cv_splits = 6
        self.cv_gap_days = 5
        self.cv_embargo_days = 5
        self.cv_test_size = cv_test_size
        self.use_lambda_percentile = use_lambda_percentile
        self.direction_calibration = bool(direction_calibration)
        self.direction_calibration_min_n = int(direction_calibration_min_n)
        self.add_rank_features = bool(add_rank_features)
        self.random_state = random_state
        self.actual_feature_cols_ = None  # 🔧 训练时实际使用的特征列（Critical Fix）
        self.direction_sign_map_ = {}  # col -> +1 / -1
        self.direction_ic_mean_ = {}   # col -> mean per-date RankIC (train window)
        self.output_sign_ = 1.0        # final output direction (score sign)

        # Nested CV & constraints
        self.use_nested_cv = use_nested_cv
        self.nested_outer_splits = nested_outer_splits
        self.nested_gap_days = nested_gap_days
        self.nested_embargo_days = nested_embargo_days
        self.aggregate_alpha = aggregate_alpha
        self.use_convex_constraint = use_convex_constraint

        # 🔥 极端值过滤
        self.filter_extreme_targets = filter_extreme_targets
        self.extreme_lower_pct = extreme_lower_pct
        self.extreme_upper_pct = extreme_upper_pct

        # 模型组件
        self.ridge_model = None
        self.scaler = None
        self.feature_importance_ = None
        self.fitted_ = False

        # 调参相关
        self.best_alpha_ = alpha
        self.alpha_scores_ = {}

        # 训练统计
        self.train_score_ = None
        self.feature_names_ = None

        logger.info(f"✅ Ridge Stacker 初始化完成 (Percentile增强版)")
        logger.info(f"   基础特征: {self.base_cols}")
        logger.info(f"   Lambda Percentile: {'启用' if self.use_lambda_percentile else '禁用'}")
        logger.info(f"   方向校正(IC<0翻转): {'启用' if self.direction_calibration else '禁用'} (min_n={self.direction_calibration_min_n})")
        logger.info(f"   Rank特征增强(截面rank_pct): {'启用' if self.add_rank_features else '禁用'}")
        logger.info(f"   正则化强度α: {self.alpha}")
        logger.info(f"   极端target过滤: {'启用' if self.filter_extreme_targets else '禁用'} "
                   f"(阈值: {self.extreme_lower_pct}%-{self.extreme_upper_pct}%)")
        logger.info(f"   拟合截距: {self.fit_intercept} (已做z-score)")
        logger.info(f"   求解器: {self.solver}, 容差: {self.tol}")
        logger.info(f"   自动调参: {self.auto_tune_alpha}")
        logger.info(f"   使用CV: {self.use_cv}, 折数: {self.cv_splits}")
        logger.info(f"   嵌套CV: {'启用' if self.use_nested_cv else '禁用'} (outer_splits={self.nested_outer_splits})")
        logger.info(f"   特征标准化: 横截面z-score")

    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证输入数据格式并确保列顺序一致"""
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("输入数据必须具有MultiIndex(date, ticker)")

        if df.index.names != ['date', 'ticker']:
            raise ValueError(f"Index names必须是['date', 'ticker'], 实际: {df.index.names}")

        missing_cols = [col for col in self.base_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")

        # 🔧 关键修复：确保列顺序与训练时完全一致
        return df[list(self.base_cols) + [col for col in df.columns if col not in self.base_cols]]

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备特征和标签（含方向校正 + rank增强）"""
        if not isinstance(df.index, pd.MultiIndex) or df.index.names != ['date', 'ticker']:
            raise ValueError("RidgeStacker expects MultiIndex(date, ticker)")

        # ---- label ----
        label_cols = [col for col in df.columns if col.startswith('ret_fwd')]
        if not label_cols:
            raise ValueError("未找到标签列 (ret_fwd_*)")
        label_col = label_cols[0]

        # ---- raw base features ----
        base_feature_cols = list(self.base_cols)
        if self.use_lambda_percentile and 'lambda_percentile' in df.columns:
            base_feature_cols.append('lambda_percentile')

        # Work on a copy to avoid mutating caller data
        feat = df[base_feature_cols + [label_col]].copy()

        # Drop NaNs early for stable IC estimation & fit
        valid_mask = ~feat.isna().any(axis=1)
        feat = feat.loc[valid_mask].copy()
        if feat.empty:
            raise ValueError("No valid rows after dropping NaNs for ridge stacker features/label.")

        # ---- direction calibration: ensure higher score => higher return ----
        if self.direction_calibration:
            # Compute per-date RankIC means for each base column
            self.direction_sign_map_ = {}
            self.direction_ic_mean_ = {}
            for col in base_feature_cols:
                # per-date Spearman
                ics = []
                for _, g in feat[[col, label_col]].groupby(level='date'):
                    if len(g) < self.direction_calibration_min_n:
                        continue
                    ic = spearmanr(g[col].values, g[label_col].values)[0]
                    if ic is None or np.isnan(ic):
                        continue
                    ics.append(float(ic))
                if len(ics) == 0:
                    # fallback to global Spearman
                    ic_global = spearmanr(feat[col].values, feat[label_col].values)[0]
                    ic_mean = float(ic_global) if ic_global is not None and not np.isnan(ic_global) else 0.0
                else:
                    ic_mean = float(np.nanmean(ics))
                sign = -1.0 if ic_mean < 0 else 1.0
                self.direction_sign_map_[col] = float(sign)
                self.direction_ic_mean_[col] = float(ic_mean)
                if sign < 0:
                    feat[col] = -feat[col]
            neg = [k for k, v in self.direction_sign_map_.items() if v < 0]
            logger.info(f"🧭 方向校正完成: flipped={neg} (n={len(neg)})")

        # ---- feature augmentation: cross-sectional ranks ----
        feature_cols = list(base_feature_cols)
        if self.add_rank_features:
            for col in base_feature_cols:
                rank_col = f"{col}__rank_pct"
                feat[rank_col] = feat.groupby(level='date')[col].rank(pct=True)
                feature_cols.append(rank_col)

        # 🔧 Critical Fix: 保存实际使用的特征列（仅在训练时，即首次调用）
        if self.actual_feature_cols_ is None:
            self.actual_feature_cols_ = list(feature_cols)
            logger.info(f"🔧 保存实际特征列: {self.actual_feature_cols_}")

        X = feat[feature_cols].values
        y = feat[label_col].values

        logger.info(f"   原始样本: {len(df)}")
        logger.info(f"   有效样本: {len(feat)}")
        logger.info(f"   数据覆盖率: {len(feat)/max(len(df),1)*100:.1f}%")

        return X, y

    def _winsorize_labels(self, y: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0) -> np.ndarray:
        """Winsorize标签，处理极端值"""
        lower_bound = np.percentile(y, lower_pct)
        upper_bound = np.percentile(y, upper_pct)
        y_winsorized = np.clip(y, lower_bound, upper_bound)

        n_clipped = np.sum((y != y_winsorized))
        if n_clipped > 0:
            logger.info(f"   Winsorize: {n_clipped}/{len(y)} ({n_clipped/len(y)*100:.1f}%) 样本被裁剪")

        return y_winsorized

    def _filter_extreme_targets(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        lower_pct: float = 0.5,
        upper_pct: float = 99.5
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        过滤target极端值的样本，完全移除而不是winsorize

        用于移除类似WW这种极端outliers (9900%涨幅或-88%跌幅)，
        防止模型训练被极端值扭曲。

        Parameters:
        -----------
        df : pd.DataFrame
            原始DataFrame（含MultiIndex）
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            target数组
        lower_pct : float
            下界百分位数（默认0.5%，移除最极端下跌）
        upper_pct : float
            上界百分位数（默认99.5%，移除最极端上涨）

        Returns:
        --------
        df_filtered : pd.DataFrame
            过滤后的DataFrame
        X_filtered : np.ndarray
            过滤后的特征矩阵
        y_filtered : np.ndarray
            过滤后的target数组

        Examples:
        ---------
        >>> # 移除WW这种9900%的极端涨幅和-88%的极端跌幅
        >>> df_filt, X_filt, y_filt = self._filter_extreme_targets(
        ...     df, X, y, lower_pct=0.5, upper_pct=99.5
        ... )
        """
        # 计算阈值
        lower_bound = np.percentile(y, lower_pct)
        upper_bound = np.percentile(y, upper_pct)

        # 创建mask: 保留在阈值范围内的样本
        mask = (y >= lower_bound) & (y <= upper_bound)

        # 过滤
        df_filtered = df.iloc[mask].copy()
        X_filtered = X[mask]
        y_filtered = y[mask]

        # 统计
        n_removed = np.sum(~mask)
        n_total = len(y)
        pct_removed = n_removed / n_total * 100

        logger.info(f"🔍 过滤极端target:")
        logger.info(f"   阈值范围: [{lower_bound*100:.2f}%, {upper_bound*100:.2f}%]")
        logger.info(f"   移除样本: {n_removed}/{n_total} ({pct_removed:.2f}%)")

        if n_removed > 0:
            # 显示被移除的极端值统计
            extreme_targets = y[~mask]
            extreme_min = extreme_targets.min() * 100
            extreme_max = extreme_targets.max() * 100
            logger.info(f"   极端值范围: [{extreme_min:.2f}%, {extreme_max:.2f}%]")

            # 显示最极端的几个
            sorted_extremes = np.sort(extreme_targets)
            n_show = min(5, len(sorted_extremes))
            if n_show > 0:
                bottom_extremes = sorted_extremes[:n_show] * 100
                top_extremes = sorted_extremes[-n_show:] * 100
                logger.info(f"   最低{n_show}个: [{', '.join([f'{x:.2f}%' for x in bottom_extremes])}]")
                logger.info(f"   最高{n_show}个: [{', '.join([f'{x:.2f}%' for x in top_extremes])}]")

        return df_filtered, X_filtered, y_filtered

    def _calculate_rank_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算RankIC"""
        try:
            return spearmanr(y_true, y_pred)[0]
        except:
            return 0.0

    def _create_purged_cv_split(self, dates: pd.Series):
        """🔥 使用统一的 PurgedCV 分割，防止数据泄露"""
        if not self.use_purged_cv:
            raise RuntimeError('PurgedCV is mandatory for RidgeStacker T+5 training.')

        cv = create_unified_cv(
            n_splits=self.cv_splits,
            gap=self.cv_gap_days,
            embargo=self.cv_embargo_days
        )

        unique_dates = sorted(dates.unique())
        if not unique_dates:
            raise ValueError('PurgedCV requires non-empty date information.')
        date_to_idx = {date: i for i, date in enumerate(unique_dates)}
        groups = dates.map(date_to_idx).values

        logger.info(f"✅ 使用PurgedCV: splits={self.cv_splits}, gap={self.cv_gap_days}, embargo={self.cv_embargo_days}")
        logger.info(f"   时间范围: {unique_dates[0]} ~ {unique_dates[-1]} ({len(unique_dates)}天)")

        return cv.split(X=np.zeros((len(dates), 1)), y=None, groups=groups)

    def _create_purged_cv_split_params(self, dates: pd.Series, n_splits: int, gap: int, embargo: int):
        """根据指定参数创建PurgedCV分割。"""
        if not PURGED_CV_AVAILABLE:
            raise RuntimeError('Unified Purged CV factory unavailable.')
        cv = create_unified_cv(n_splits=n_splits, gap=gap, embargo=embargo)
        unique_dates = sorted(dates.unique())
        if not unique_dates:
            raise ValueError('PurgedCV requires non-empty date information.')
        date_to_idx = {date: i for i, date in enumerate(unique_dates)}
        groups = dates.map(date_to_idx).values
        logger.info(f"✅ 使用PurgedCV(n_splits={n_splits}, gap={gap}, embargo={embargo})")
        return cv.split(X=np.zeros((len(dates), 1)), y=None, groups=groups)

    def _auto_tune_alpha(self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame, dates: pd.Series = None) -> float:
        """
        自动调参选择最优alpha - 增强CV版本
        """
        if not self.auto_tune_alpha:
            return self.alpha

        logger.info(f"🎯 开始自动调参，网格: {self.alpha_grid}")

        if not self.use_cv:
            raise RuntimeError("RidgeStacker requires PurgedCV-based CV for T+5 training.")

        return self._auto_tune_alpha_with_cv(X, y, dates)

    def _auto_tune_alpha_with_cv(self, X: np.ndarray, y: np.ndarray, dates: pd.Series = None) -> float:
        """
        使用CV进行调参（严格PurgedCV）
        """
        if dates is None:
            raise ValueError("PurgedCV tuning requires aligned date information.")
        if not self.use_purged_cv:
            raise RuntimeError("PurgedCV must be enabled for alpha tuning.")

        logger.info(f"   使用PurgedCV，折数: {self.cv_splits}")

        best_alpha = self.alpha
        best_score = -999

        cv_splits = list(self._create_purged_cv_split(dates))

        for alpha in self.alpha_grid:
            cv_scores = []

            for fold, (train_idx, test_idx) in enumerate(cv_splits):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = Ridge(
                    alpha=alpha,
                    fit_intercept=self.fit_intercept,
                    solver=self.solver,
                    tol=self.tol,
                    random_state=self.random_state
                )
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                rank_ic = self._calculate_rank_ic(y_test, y_pred)
                cv_scores.append(rank_ic)

            avg_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            self.alpha_scores_[alpha] = avg_score

            logger.info(f"   α={alpha}: CV RankIC={avg_score:.4f} (±{std_score:.4f})")

            if avg_score > best_score:
                best_score = avg_score
                best_alpha = alpha

        logger.info(f"✅ 最优α: {best_alpha}, CV RankIC: {best_score:.4f}")
        return best_alpha

    def _project_to_simplex(self, w: np.ndarray) -> np.ndarray:
        """将向量投影到概率单纯形：w>=0, sum(w)=1。"""
        if w.ndim != 1:
            w = w.flatten()
        # Algorithm from: Efficient Projections onto the l1-Ball for Learning in High Dimensions (Duchi et al.)
        u = np.sort(np.maximum(w, 0))[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]
        if len(rho) == 0:
            # all zeros, return uniform
            m = len(w)
            return np.full(m, 1.0 / m)
        rho = rho[-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        w_proj = np.maximum(w - theta, 0)
        s = w_proj.sum()
        if s <= 0:
            m = len(w)
            return np.full(m, 1.0 / m)
        return w_proj / s

    def _fit_ridge_with_constraint(self, X: np.ndarray, y: np.ndarray, alpha: float, fit_intercept: bool = False) -> Tuple[np.ndarray, float]:
        """拟合Ridge并对权重投影到非负且和为1的单纯形。"""
        model = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            solver=self.solver,
            tol=self.tol,
            random_state=self.random_state
        )
        model.fit(X, y)
        w = np.asarray(model.coef_).flatten()
        b = float(model.intercept_) if fit_intercept else 0.0
        if self.use_convex_constraint:
            w = self._project_to_simplex(np.maximum(w, 0))
            b = 0.0  # 保持可解释性：凸组合通常不需要截距
        return w, b

    def _inner_tune_alpha(self, X_train: np.ndarray, y_train: np.ndarray, dates_train: pd.Series) -> float:
        """在外层训练段上进行内层PurgedCV调参。"""
        best_alpha = None
        best_score = -1e9
        # dates_train 可能是 Series/Index/ndarray，统一为Series
        if not isinstance(dates_train, pd.Series):
            dates_train = pd.Series(np.asarray(dates_train))
        inner_splits = list(self._create_purged_cv_split_params(dates_train, self.cv_splits, self.cv_gap_days, self.cv_embargo_days))
        for alpha in self.alpha_grid:
            scores = []
            for (tr_idx, te_idx) in inner_splits:
                # 使用仅基于训练的标准化，避免泄露
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train[tr_idx])
                X_te = scaler.transform(X_train[te_idx])
                w, b = self._fit_ridge_with_constraint(X_tr, y_train[tr_idx], alpha, fit_intercept=self.fit_intercept)
                y_hat = X_te @ w + b
                scores.append(self._calculate_rank_ic(y_train[te_idx], y_hat))
            avg = float(np.nanmean(scores)) if scores else -1e9
            self.alpha_scores_[alpha] = avg
            if avg > best_score:
                best_score = avg
                best_alpha = alpha
        return best_alpha if best_alpha is not None else self.alpha

    def _nested_cv_oof(self, X: np.ndarray, y: np.ndarray, dates: pd.Series) -> Tuple[np.ndarray, List[float], List[float]]:
        """
        外层PurgedCV生成OOF预测；内层在外层训练段时间CV选择alpha。
        返回: y_oof_pred, alphas_per_fold, ic_per_fold
        """
        n = len(y)
        y_oof = np.full(n, np.nan, dtype=float)
        alphas = []
        ics = []
        outer_splits = list(self._create_purged_cv_split_params(dates, self.nested_outer_splits, self.nested_gap_days, self.nested_embargo_days))
        for k, (train_idx, test_idx) in enumerate(outer_splits):
            # 内层调参（dates 可能是 DatetimeIndex/Series，统一转换为numpy数组再按位置索引）
            dates_array = np.asarray(dates)
            dates_train = pd.Series(dates_array[train_idx])
            alpha_k = self._inner_tune_alpha(X[train_idx], y[train_idx], dates_train)
            alphas.append(alpha_k)
            # 基于训练段拟合，并在测试段预测
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            w, b = self._fit_ridge_with_constraint(X_tr, y[train_idx], alpha_k, fit_intercept=self.fit_intercept)
            y_hat = X_te @ w + b
            y_oof[test_idx] = y_hat
            ic_k = self._calculate_rank_ic(y[test_idx], y_hat)
            ics.append(float(ic_k))
            logger.info(f"[NestedCV] outer_fold={k+1}/{len(outer_splits)} alpha={alpha_k} OOS RankIC={ic_k:.4f}")
        return y_oof, alphas, ics

    def fit(self, df: pd.DataFrame, max_train_to_today: bool = True) -> "RidgeStacker":
        """
        训练Ridge二层Stacker。输入为包含第一层预测与标签的DataFrame。

        必需列: self.base_cols 中的列，以及一个以 'ret_fwd' 开头的标签列。
        索引: 需要MultiIndex(date, ticker)。
        """
        df_validated = self._validate_input(df)
        X, y = self._prepare_features(df_validated)

        # 🔥 新增: 先过滤极端target样本（如WW的9900%涨幅）
        if self.filter_extreme_targets:
            df_filtered, X_filtered, y_filtered = self._filter_extreme_targets(
                df_validated,
                X,
                y,
                lower_pct=self.extreme_lower_pct,
                upper_pct=self.extreme_upper_pct
            )
        else:
            df_filtered, X_filtered, y_filtered = df_validated, X, y
            logger.info("⚠️ 极端target过滤已禁用")

        # 标签Winsorize，增强稳健性（在过滤后的数据上操作）
        y_proc = self._winsorize_labels(y_filtered, lower_pct=1.0, upper_pct=99.0)

        # 特征标准化（使用过滤后的数据）
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_filtered)

        # 使用过滤后的dates
        dates = df_filtered.index.get_level_values('date')

        if self.use_nested_cv:
            # 嵌套CV：生成OOF并聚合alpha，然后在全量上再训练一次
            logger.info("🚀 使用严谨嵌套CV训练二层 (外层滚动 + 内层PurgedCV调参)")
            # 使用未缩放X进行嵌套CV，内部各fold单独标准化
            y_oof, alphas, ics = self._nested_cv_oof(X_filtered, y_proc, dates)
            # 选择聚合alpha
            if self.aggregate_alpha == 'ir_weighted' and len(ics) > 1 and np.nanstd(ics) > 0:
                weights = np.maximum(np.array(ics), 0)
                if weights.sum() == 0:
                    self.best_alpha_ = float(np.median(alphas))
                else:
                    self.best_alpha_ = float(np.average(alphas, weights=weights))
            else:
                self.best_alpha_ = float(np.median(alphas)) if len(alphas) > 0 else self.alpha
            logger.info(f"[NestedCV] 聚合alpha={self.best_alpha_} (strategy={self.aggregate_alpha})")
            # 在全量上拟合最终模型（先全量标准化，再拟合+投影）
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_filtered)
            w, b = self._fit_ridge_with_constraint(X_scaled, y_proc, self.best_alpha_, fit_intercept=self.fit_intercept)
            # 存入sklearn Ridge对象以保持接口一致
            self.ridge_model = Ridge(alpha=self.best_alpha_, fit_intercept=False)
            # 确保coef_为1D，避免predict输出(n,1)造成下游"Data must be 1-dimensional"错误
            self.ridge_model.coef_ = w.reshape(-1)
            self.ridge_model.intercept_ = 0.0
            # 显式设置特征维度，提升sklearn一致性
            try:
                self.ridge_model.n_features_in_ = X_scaled.shape[1]
            except Exception:
                pass
        else:
            # 可选：自动调参（单层CV）
            if self.auto_tune_alpha:
                self.best_alpha_ = self._auto_tune_alpha(X_scaled, y_proc, df_validated, dates)
            else:
                self.best_alpha_ = self.alpha

            # 使用Ridge训练
            self.ridge_model = Ridge(
                alpha=self.best_alpha_,
                fit_intercept=self.fit_intercept,
                solver=self.solver,
                tol=self.tol,
                random_state=self.random_state
            )
            self.ridge_model.fit(X_scaled, y_proc)
            # 训练后进行凸约束投影（可选）
            if self.use_convex_constraint:
                w = np.asarray(self.ridge_model.coef_).flatten()
                w = self._project_to_simplex(np.maximum(w, 0))
                self.ridge_model.coef_ = w.reshape(-1)
                if self.fit_intercept:
                    self.ridge_model.intercept_ = 0.0
            try:
                self.ridge_model.n_features_in_ = X_scaled.shape[1]
            except Exception:
                pass

        # 训练分数与特征重要性
        y_pred_train = self.ridge_model.predict(X_scaled)
        train_ic = float(self._calculate_rank_ic(y_proc, y_pred_train))
        # Final direction calibration: ensure Ridge score higher => higher return on train window.
        # This is robust even if base preds look aligned but the learned combination is inverted.
        self.output_sign_ = -1.0 if (self.direction_calibration and train_ic < 0) else 1.0
        if self.output_sign_ < 0:
            y_pred_train = -y_pred_train
            train_ic = float(self._calculate_rank_ic(y_proc, y_pred_train))
            logger.info("🧭 Ridge输出方向校正: output_sign=-1 (train RankIC flipped to %.4f)", train_ic)
        self.train_score_ = train_ic
        self.feature_names_ = list(self.actual_feature_cols_ or self.base_cols)
        # 线性模型的重要性可用系数绝对值
        try:
            coefs = np.asarray(self.ridge_model.coef_).flatten()
            self.feature_importance_ = {name: float(abs(w)) for name, w in zip(self.feature_names_, coefs)}
        except Exception:
            self.feature_importance_ = None

        self.fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用训练好的RidgeStacker进行预测。
        返回包含'score'列的DataFrame（索引与输入对齐）。
        """
        if not self.fitted_ or self.ridge_model is None or self.scaler is None:
            raise RuntimeError("RidgeStacker not fitted")

        # 确保列顺序与训练一致；允许缺失标签列
        if not isinstance(df.index, pd.MultiIndex) or df.index.names != ['date', 'ticker']:
            raise ValueError("预测数据必须具有MultiIndex(date, ticker)")

        # Build required features (supports derived cols like "__rank_pct")
        if not isinstance(df.index, pd.MultiIndex) or df.index.names != ['date', 'ticker']:
            raise ValueError("预测数据必须具有MultiIndex(date, ticker)")

        feature_cols = list(self.actual_feature_cols_ or self.base_cols)
        base_needed = list(self.base_cols)
        if self.use_lambda_percentile and 'lambda_percentile' in feature_cols and 'lambda_percentile' not in base_needed:
            base_needed.append('lambda_percentile')

        missing_base = [c for c in base_needed if c not in df.columns]
        if missing_base:
            raise ValueError(f"预测缺少必需基础特征列: {missing_base}")

        feat = df[base_needed].copy()

        # Apply persisted direction sign-map if available
        if getattr(self, 'direction_sign_map_', None):
            for col, s in self.direction_sign_map_.items():
                if col in feat.columns and float(s) < 0:
                    feat[col] = -feat[col]

        # Generate derived features if required by trained model
        for col in feature_cols:
            if col in feat.columns:
                continue
            if col.endswith("__rank_pct"):
                raw = col.replace("__rank_pct", "")
                if raw not in feat.columns:
                    raise ValueError(f"无法生成派生特征 {col}: 缺少原始列 {raw}")
                feat[col] = feat.groupby(level='date')[raw].rank(pct=True)
                continue
            raise ValueError(f"预测缺少必需特征列: {col}")

        X = feat[feature_cols].values
        X_scaled = self.scaler.transform(X)
        y_pred = self.ridge_model.predict(X_scaled)
        if float(getattr(self, "output_sign_", 1.0) or 1.0) < 0:
            y_pred = -y_pred
        return pd.DataFrame({'score': y_pred}, index=df.index)

    def replace_ewa_in_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compatibility shim for the legacy EWA interface used in the pipeline."""
        if not self.fitted_ or self.ridge_model is None or self.scaler is None:
            raise RuntimeError('RidgeStacker not fitted; call fit before replace_ewa_in_pipeline.')
        if not isinstance(df, pd.DataFrame):
            raise TypeError('replace_ewa_in_pipeline expects a pandas DataFrame.')
        if df.empty:
            raise ValueError('replace_ewa_in_pipeline received an empty DataFrame.')
        if not isinstance(df.index, pd.MultiIndex) or df.index.names != ['date', 'ticker']:
            raise ValueError('Input to replace_ewa_in_pipeline must have MultiIndex(date, ticker).')

        feature_cols = list(self.actual_feature_cols_ or self.base_cols)
        sanitized = df.copy()
        label_cols = [col for col in sanitized.columns if col.startswith('ret_fwd')]
        if label_cols:
            sanitized = sanitized.drop(columns=label_cols)

        missing = [col for col in feature_cols if col not in sanitized.columns]
        if missing:
            raise ValueError(f'replace_ewa_in_pipeline is missing required features: {missing}')

        sanitized = sanitized[feature_cols]
        valid_mask = ~sanitized.isna().any(axis=1)
        if not valid_mask.any():
            raise ValueError('replace_ewa_in_pipeline found no valid rows after dropping NaNs.')

        if not valid_mask.all():
            logger.debug('replace_ewa_in_pipeline dropping %d rows with NaN features', (~valid_mask).sum())

        valid_features = sanitized.loc[valid_mask]
        predictions = self.predict(valid_features)

        if valid_mask.all():
            return predictions

        full_output = pd.DataFrame(index=sanitized.index, columns=predictions.columns, dtype=float)
        full_output.loc[valid_mask] = predictions.values
        return full_output

    def get_model_info(self) -> Dict[str, any]:
        return {
            'alpha': float(self.best_alpha_),
            'use_lambda_percentile': bool(self.use_lambda_percentile),
            'direction_calibration': bool(self.direction_calibration),
            'direction_sign_map': dict(getattr(self, 'direction_sign_map_', {}) or {}),
            'direction_ic_mean': dict(getattr(self, 'direction_ic_mean_', {}) or {}),
            'output_sign': float(getattr(self, 'output_sign_', 1.0) or 1.0),
            'add_rank_features': bool(self.add_rank_features),
            'train_rank_ic': float(self.train_score_) if self.train_score_ is not None else None,
            'feature_importance': self.feature_importance_,
            'features': list(self.feature_names_ or []),
            'n_iterations': 1
        }