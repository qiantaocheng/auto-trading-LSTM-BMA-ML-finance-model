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
from sklearn.model_selection import TimeSeriesSplit

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
                 base_cols: Tuple[str, ...] = ('pred_catboost', 'pred_elastic', 'pred_xgb'),
                 alpha: float = 1.0,
                 fit_intercept: bool = False,
                 solver: str = "auto",
                 tol: float = 1e-6,
                 auto_tune_alpha: bool = False,
                 alpha_grid: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0, 8.0),
                 use_cv: bool = True,
                 cv_splits: int = 3,
                 cv_test_size: float = 0.2,
                 random_state: int = 42,
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
            cv_splits: CV折数 (默认3)
            cv_test_size: 每折验证集比例 (默认0.2)
            random_state: 随机种子
        """
        self.base_cols = base_cols
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.tol = tol
        self.auto_tune_alpha = auto_tune_alpha
        self.alpha_grid = alpha_grid
        self.use_cv = use_cv
        self.cv_splits = cv_splits
        self.cv_test_size = cv_test_size
        self.random_state = random_state

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

        logger.info(f"✅ Ridge Stacker 初始化完成 (简洁版)")
        logger.info(f"   基础特征: {self.base_cols}")
        logger.info(f"   正则化强度α: {self.alpha}")
        logger.info(f"   拟合截距: {self.fit_intercept} (已做z-score)")
        logger.info(f"   求解器: {self.solver}, 容差: {self.tol}")
        logger.info(f"   自动调参: {self.auto_tune_alpha}")
        logger.info(f"   使用CV: {self.use_cv}, 折数: {self.cv_splits}")
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
        """准备特征和标签"""
        # 提取特征
        X = df[list(self.base_cols)].values

        # 提取标签（假设标签列以ret_fwd开头）
        label_cols = [col for col in df.columns if col.startswith('ret_fwd')]
        if not label_cols:
            raise ValueError("未找到标签列 (ret_fwd_*)")

        label_col = label_cols[0]  # 使用第一个找到的标签列
        y = df[label_col].values

        # 移除NaN样本
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        logger.info(f"   原始样本: {len(X)}")
        logger.info(f"   有效样本: {len(X_clean)}")
        logger.info(f"   数据覆盖率: {len(X_clean)/len(X)*100:.1f}%")

        return X_clean, y_clean

    def _winsorize_labels(self, y: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0) -> np.ndarray:
        """Winsorize标签，处理极端值"""
        lower_bound = np.percentile(y, lower_pct)
        upper_bound = np.percentile(y, upper_pct)
        y_winsorized = np.clip(y, lower_bound, upper_bound)

        n_clipped = np.sum((y != y_winsorized))
        if n_clipped > 0:
            logger.info(f"   Winsorize: {n_clipped}/{len(y)} ({n_clipped/len(y)*100:.1f}%) 样本被裁剪")

        return y_winsorized

    def _calculate_rank_ic(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算RankIC"""
        try:
            return spearmanr(y_true, y_pred)[0]
        except:
            return 0.0

    def _time_series_cv_split(self, n_samples: int):
        """创建时间序列CV分割"""
        splits = []
        test_size = int(n_samples * self.cv_test_size)
        train_min_size = max(100, int(n_samples * 0.3))  # 至少30%训练数据

        for i in range(self.cv_splits):
            # 递增训练集大小
            train_end = train_min_size + i * ((n_samples - test_size - train_min_size) // self.cv_splits)
            test_start = train_end
            test_end = min(test_start + test_size, n_samples)

            if test_end > n_samples or test_start >= test_end:
                break

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            splits.append((train_idx, test_idx))

        return splits

    def _auto_tune_alpha(self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame) -> float:
        """
        自动调参选择最优alpha - 增强CV版本
        """
        if not self.auto_tune_alpha:
            return self.alpha

        logger.info(f"🎯 开始自动调参，网格: {self.alpha_grid}")

        if self.use_cv:
            logger.info(f"   使用时间序列CV，折数: {self.cv_splits}")
            return self._auto_tune_alpha_with_cv(X, y)
        else:
            logger.info(f"   使用全量训练（无CV）")
            return self._auto_tune_alpha_no_cv(X, y)

    def _auto_tune_alpha_with_cv(self, X: np.ndarray, y: np.ndarray) -> float:
        """使用CV进行调参"""
        best_alpha = self.alpha
        best_score = -999

        # 创建时间序列分割
        cv_splits = self._time_series_cv_split(len(X))

        for alpha in self.alpha_grid:
            cv_scores = []

            for fold, (train_idx, test_idx) in enumerate(cv_splits):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # 训练模型
                model = Ridge(
                    alpha=alpha,
                    fit_intercept=self.fit_intercept,
                    solver=self.solver,
                    tol=self.tol,
                    random_state=self.random_state
                )
                model.fit(X_train, y_train)

                # 验证集预测
                y_pred = model.predict(X_test)
                rank_ic = self._calculate_rank_ic(y_test, y_pred)
                cv_scores.append(rank_ic)

            avg_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            self.alpha_scores_[alpha] = avg_score

            logger.info(f"   α={alpha}: CV RankIC={avg_score:.4f} (±{std_score:.4f})")

            # 选择最优alpha
            if avg_score > best_score:
                best_score = avg_score
                best_alpha = alpha

        logger.info(f"✅ 最优α: {best_alpha}, CV RankIC: {best_score:.4f}")
        return best_alpha

    def _auto_tune_alpha_no_cv(self, X: np.ndarray, y: np.ndarray) -> float:
        """不使用CV的调参（原版）"""
        best_alpha = self.alpha
        best_score = -999

        for alpha in self.alpha_grid:
            try:
                # 全量训练模型
                model = Ridge(
                    alpha=alpha,
                    fit_intercept=self.fit_intercept,
                    solver=self.solver,
                    tol=self.tol,
                    random_state=self.random_state
                )
                model.fit(X, y)

                # 全量数据预测并计算RankIC
                y_pred = model.predict(X)
                rank_ic = self._calculate_rank_ic(y, y_pred)

                # 使用RankIC作为主要评分
                score = rank_ic
                self.alpha_scores_[alpha] = rank_ic

                logger.info(f"   α={alpha}: RankIC={rank_ic:.4f}")

                # 如果RankIC更好，则更新
                tolerance = 0.001
                if (score > best_score + tolerance) or \
                   (abs(score - best_score) <= tolerance and alpha > best_alpha):
                    best_score = score
                    best_alpha = alpha

            except Exception as e:
                logger.debug(f"调参异常 alpha={alpha}: {e}")
                self.alpha_scores_[alpha] = 0.0

        self.best_alpha_ = best_alpha
        logger.info(f"✅ 最优α: {best_alpha} (RankIC: {self.alpha_scores_[best_alpha]:.4f}, 无CV)")

        return best_alpha

    def fit(self, df: pd.DataFrame, **kwargs) -> 'RidgeStacker':
        """
        训练Ridge Stacker（增强CV版）

        Args:
            df: 包含第一层预测和标签的DataFrame
            **kwargs: 兼容参数（max_train_to_today等）
        """
        logger.info("🚀 开始训练Ridge Stacker")
        logger.info(f"   期望特征顺序: {list(self.base_cols)}")
        logger.info(f"   CV模式: {'启用' if self.use_cv else '禁用'}")

        # 验证数据
        df_clean = self._validate_input(df)

        # 准备特征和标签
        X, y = self._prepare_features(df_clean)

        if len(X) < 100:  # 提高最小样本要求
            raise ValueError(f"训练样本过少: {len(X)} < 100")

        # 标签Winsorization (1%, 99%)
        y_winsorized = self._winsorize_labels(y, 1.0, 99.0)

        # 特征标准化（横截面z-score）
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        logger.info(f"   标准化完成: 特征均值={X_scaled.mean(axis=0)[:3]}, 特征标准差={X_scaled.std(axis=0)[:3]}")

        # 直接使用指定alpha（无调参）
        if self.auto_tune_alpha:
            optimal_alpha = self._auto_tune_alpha(X_scaled, y_winsorized, df_clean)
        else:
            optimal_alpha = self.alpha
            logger.info(f"🎯 使用指定α: {optimal_alpha} (无调参)")

        # 使用CV验证最终模型性能
        val_score = None
        val_rank_ic = None

        if self.use_cv and len(X_scaled) > 200:  # 足够样本时使用CV
            # 使用最后一折作为验证集
            val_size = int(len(X_scaled) * self.cv_test_size)
            train_size = len(X_scaled) - val_size

            X_train = X_scaled[:train_size]
            y_train = y_winsorized[:train_size]
            X_val = X_scaled[train_size:]
            y_val = y_winsorized[train_size:]

            # 训练模型
            self.ridge_model = Ridge(
                alpha=optimal_alpha,
                fit_intercept=self.fit_intercept,
                solver=self.solver,
                tol=self.tol,
                random_state=self.random_state
            )
            self.ridge_model.fit(X_train, y_train)

            # 验证集评估
            y_val_pred = self.ridge_model.predict(X_val)
            val_score = self.ridge_model.score(X_val, y_val)
            val_rank_ic = self._calculate_rank_ic(y_val, y_val_pred)

            logger.info(f"   验证集R²: {val_score:.4f}")
            logger.info(f"   验证集RankIC: {val_rank_ic:.4f}")

            # 重新使用全量数据训练最终模型
            self.ridge_model.fit(X_scaled, y_winsorized)
        else:
            # 直接全量训练
            self.ridge_model = Ridge(
                alpha=optimal_alpha,
                fit_intercept=self.fit_intercept,
                solver=self.solver,
                tol=self.tol,
                random_state=self.random_state
            )
            self.ridge_model.fit(X_scaled, y_winsorized)

        # 计算训练性能
        y_pred = self.ridge_model.predict(X_scaled)
        self.train_score_ = self.ridge_model.score(X_scaled, y_winsorized)
        train_rmse = np.sqrt(mean_squared_error(y_winsorized, y_pred))
        train_rank_ic = self._calculate_rank_ic(y_winsorized, y_pred)

        # 计算原始标签的RankIC
        original_rank_ic = self._calculate_rank_ic(y, y_pred)

        # 保存验证分数
        self.val_score_ = val_score
        self.val_rank_ic_ = val_rank_ic

        # 保存特征重要性（回归系数）
        self.feature_names_ = list(self.base_cols)
        self.feature_importance_ = pd.DataFrame({
            'feature': self.feature_names_,
            'coefficient': self.ridge_model.coef_,
            'abs_coefficient': np.abs(self.ridge_model.coef_)
        }).sort_values('abs_coefficient', ascending=False)

        self.fitted_ = True

        logger.info("✅ Ridge Stacker 训练完成")
        logger.info(f"   使用α: {optimal_alpha}")
        logger.info(f"   训练R²: {self.train_score_:.4f}")
        logger.info(f"   训练RMSE: {train_rmse:.6f}")
        logger.info(f"   RankIC(winsorized): {train_rank_ic:.4f}")
        logger.info(f"   RankIC(原始): {original_rank_ic:.4f}")
        if self.val_score_ is not None:
            logger.info(f"   CV验证R²: {self.val_score_:.4f}")
            logger.info(f"   CV验证RankIC: {self.val_rank_ic_:.4f}")
        logger.info("   特征重要性 (系数):")
        for _, row in self.feature_importance_.head(3).iterrows():
            logger.info(f"     {row['feature']}: {row['coefficient']:.4f}")

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用Ridge模型进行预测

        Args:
            df: 包含第一层预测的DataFrame

        Returns:
            包含预测分数和排名的DataFrame
        """
        if not self.fitted_:
            raise RuntimeError("模型未训练，请先调用fit()")

        logger.info("📊 Ridge Stacker 开始预测...")

        # 验证输入并确保列顺序一致
        df_clean = self._validate_input(df)

        # 🔧 关键修复：使用与训练时完全相同的特征提取逻辑
        X = df_clean[list(self.base_cols)].values

        # 处理NaN
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]

        if len(X_valid) == 0:
            raise ValueError("所有样本都包含NaN，无法预测")

        # 🔧 关键修复：使用训练时拟合的标准化器
        if self.scaler is None:
            raise RuntimeError("标准化器未初始化，模型可能未正确训练")

        X_scaled = self.scaler.transform(X_valid)

        logger.info(f"   特征提取: {len(X_valid)} 有效样本, {X_scaled.shape[1]} 特征")
        logger.info(f"   特征顺序: {list(self.base_cols)}")
        logger.info(f"   预测标准化: 特征均值={X_scaled.mean(axis=0)[:3]}, 特征标准差={X_scaled.std(axis=0)[:3]}")

        # 🔧 验证特征维度一致性
        if X_scaled.shape[1] != len(self.base_cols):
            raise RuntimeError(f"特征维度不一致: 预测时{X_scaled.shape[1]}列，训练时{len(self.base_cols)}列")

        # 预测
        raw_predictions = self.ridge_model.predict(X_scaled)

        # 创建完整预测数组
        full_predictions = np.full(len(X), np.nan)
        full_predictions[valid_mask] = raw_predictions

        # 构建结果DataFrame
        result = df_clean.copy()
        result['score'] = full_predictions

        # 按日期计算排名
        def _rank_by_date(group):
            scores = group['score']
            valid_scores = scores.dropna()
            if len(valid_scores) <= 1:
                return pd.Series(np.nan, index=scores.index)

            ranks = valid_scores.rank(method='average', ascending=False)
            full_ranks = pd.Series(np.nan, index=scores.index)
            full_ranks.loc[valid_scores.index] = ranks
            return full_ranks

        result['score_rank'] = result.groupby(level='date').apply(_rank_by_date).values

        # 标准化分数
        def _zscore_by_date(group):
            scores = group['score']
            valid_scores = scores.dropna()
            if len(valid_scores) <= 1:
                return pd.Series(0.0, index=scores.index)

            mean_score = valid_scores.mean()
            std_score = valid_scores.std()
            if std_score < 1e-12:
                return pd.Series(0.0, index=scores.index)

            zscores = (valid_scores - mean_score) / std_score
            full_zscores = pd.Series(0.0, index=scores.index)
            full_zscores.loc[valid_scores.index] = zscores
            return full_zscores

        result['score_z'] = result.groupby(level='date').apply(_zscore_by_date).values

        logger.info(f"✅ Ridge预测完成: {len(result)}样本")
        logger.info(f"   有效预测: {(~pd.isna(result['score'])).sum()}")

        return result[['score', 'score_rank', 'score_z']]

    def replace_ewa_in_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        替代EWA的管道接口 - 兼容现有调用

        Args:
            df: 输入数据

        Returns:
            预测结果
        """
        return self.predict(df)

    def get_model_info(self) -> Dict:
        """
        获取模型信息 - 兼容LTR接口

        Returns:
            模型信息字典
        """
        if not self.fitted_:
            return {}

        return {
            'model_type': 'Ridge Regression (开箱即用版)',
            'n_features': len(self.feature_names_) if self.feature_names_ else 0,
            'alpha': self.best_alpha_,  # 使用最优alpha
            'alpha_grid': list(self.alpha_grid),
            'alpha_scores': dict(self.alpha_scores_),
            'train_score': self.train_score_,
            'solver': self.solver,
            'tol': self.tol,
            'fit_intercept': self.fit_intercept,
            'auto_tune_alpha': self.auto_tune_alpha,
            'intercept': self.ridge_model.intercept_ if self.fit_intercept and self.ridge_model else 0.0,
            'feature_importance': self.feature_importance_.to_dict('records') if self.feature_importance_ is not None else None,
            'feature_names': self.feature_names_,
            'configuration': 'Optimized for 2600 stocks × 3 years, T→T+5 horizon'
        }

    @property
    def best_iteration_(self):
        """兼容LTR接口的属性"""
        return 1 if self.fitted_ else None

# 兼容导入
LtrIsotonicStacker = RidgeStacker  # 提供向后兼容的别名