"""
LTR (LambdaRank) + Isotonic Regression 二层 Stacking 模型
替换原有的 EWA (指数加权平均) 方案，提供更优的排序和校准能力

Author: BMA Trading System
Date: 2025-01-16
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy.stats import rankdata, spearmanr
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """确保数据按 (date, ticker) 排序"""
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("df index must be MultiIndex[(date,ticker)]")
    return df.sort_index(level=['date','ticker'])


def _group_sizes_by_date(df: pd.DataFrame) -> List[int]:
    """
    以 date 为 query 生成 LightGBM 的 group
    依赖 df 已按 (date,ticker) 排序！
    """
    return [len(g) for _, g in df.groupby(level='date', sort=False)]


def _winsorize_by_date(s: pd.Series, limits=(0.01, 0.99)) -> pd.Series:
    """逐日分位裁剪（更稳健）"""
    def _w(x):
        if len(x) < 2:
            return x
        lo, hi = x.quantile(limits[0]), x.quantile(limits[1])
        return x.clip(lo, hi)
    return s.groupby(level='date').apply(_w)


def _zscore_by_date(s: pd.Series) -> pd.Series:
    """逐日标准化"""
    def _z(x):
        if len(x) < 2:
            return x
        mu, sd = x.mean(), x.std(ddof=0)
        return (x - mu) / (sd if sd > 1e-12 else 1.0)
    return s.groupby(level='date').apply(_z)


def _demean_by_group(df_feat: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """按行业等分类在截面去均值：X := X - group_mean(X)"""
    def _demean(group):
        return group - group.mean()
    return df_feat.groupby([df_feat.index.get_level_values('date'), df_feat[group_col]]).transform(_demean)


def _neutralize(df: pd.DataFrame, cols: List[str], cfg: Optional[Dict] = None) -> pd.DataFrame:
    """
    简版中性化：优先按 'by' 列去均值，可选再对 beta 做线性回归残差
    cfg 示例: {'by':['sector'], 'beta_col':'beta'}
    """
    out = df.copy()
    if cfg and 'by' in cfg:
        for gcol in cfg['by']:
            if gcol not in out.columns:
                continue
            # 按组去均值
            for c in cols:
                temp_df = out[[c, gcol]].copy()
                temp_df.columns = ['_v', gcol]
                out[c] = _demean_by_group(temp_df, gcol)['_v']
    return out


def _spearman_ic_eval(preds: np.ndarray, dataset: lgb.Dataset):
    """自定义评估：Spearman IC（逐日计算后取均值）"""
    y = dataset.get_label()
    groups = dataset.get_group()

    ic_list = []
    start = 0
    for g in groups:
        end = start + int(g)
        y_g = y[start:end]
        p_g = preds[start:end]

        if len(y_g) > 1:
            # Spearman：对 y 与 p 分别取秩相关
            r_y = rankdata(y_g, method='average')
            r_p = rankdata(p_g, method='average')
            # 皮尔逊相关系数（对秩）
            ic = np.corrcoef(r_y, r_p)[0,1] if len(r_y) > 1 else 0.0
        else:
            ic = 0.0

        ic_list.append(ic)
        start = end

    ic_mean = float(np.mean(ic_list)) if ic_list else 0.0
    # LightGBM 需要 (名称, 值, 越大越好)
    return ('spearman_ic', ic_mean, True)


def make_purged_splits(dates_sorted: np.ndarray, n_splits=5, embargo=10) -> List[Tuple]:
    """
    时序CV（带 purge + embargo）的折生成器
    dates_sorted: 升序的不重复交易日数组
    返回 [(train_date_idx, valid_date_idx), ...]，并自动在每折间隔 embargo 天
    """
    n = len(dates_sorted)
    fold_size = n // (n_splits + 1)  # 留出末段做测试/留白
    splits = []

    for k in range(n_splits):
        train_end = fold_size * (k + 1)
        valid_start = train_end + embargo
        valid_end = min(valid_start + fold_size, n)

        if valid_end <= valid_start:
            break

        train_idx = np.arange(0, train_end)   # [0, train_end)
        valid_idx = np.arange(valid_start, valid_end)
        splits.append((train_idx, valid_idx))

    return splits


class LtrIsotonicStacker:
    """
    LambdaRank + 全局 Isotonic 校准的二层 Stacking 模型
    用于替换原有的 EWA 方案，提供更优的 T+10 预测能力
    """

    def __init__(self,
                 base_cols=('pred_catboost','pred_elastic','pred_xgb'),
                 horizon=10,
                 winsor_limits=(0.01, 0.99),
                 do_zscore=True,
                 neutralize_cfg=None,
                 lgbm_params=None,
                 n_splits=5,
                 embargo=10,
                 random_state=42):
        """
        初始化 LTR Stacker

        Args:
            base_cols: 一层模型预测列名
            horizon: 预测期限（天）
            winsor_limits: 极值处理分位数
            do_zscore: 是否做Z-score标准化
            neutralize_cfg: 中性化配置
            lgbm_params: LightGBM参数
            n_splits: CV折数
            embargo: 时间间隔天数
            random_state: 随机种子
        """
        self.base_cols_ = list(base_cols)
        self.horizon_ = int(horizon)
        self.winsor_limits_ = winsor_limits
        self.do_zscore_ = do_zscore
        self.neutralize_cfg_ = neutralize_cfg or {}
        self.n_splits_ = n_splits
        self.embargo_ = embargo
        self.random_state_ = random_state

        # 默认的 LambdaRank 参数（可再调）
        self.lgbm_params_ = lgbm_params or dict(
            objective='lambdarank',
            boosting_type='gbdt',
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_data_in_leaf=100,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            metric='ndcg',
            lambda_l1=0.1,
            lambda_l2=0.1,
            verbosity=-1,
            n_estimators=3000,
            importance_type='gain'
        )

        self.ranker_ = None
        self.calibrator_ = None
        self.fitted_ = False
        self._col_cache_ = None  # 记录训练期的列顺序/名称
        self.feature_importance_ = None
        self.cv_scores_ = []
        self.oof_predictions_ = None
        self.oof_targets_ = None

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一预处理：对训练或推理数据都可用"""
        df = _ensure_sorted(df.copy())

        # 检查必要列
        use_cols = [c for c in self.base_cols_ if c in df.columns]
        if len(use_cols) != len(self.base_cols_):
            miss = set(self.base_cols_) - set(use_cols)
            logger.warning(f"缺少一层列：{miss}，尝试使用可用列")
            if len(use_cols) == 0:
                raise ValueError("没有找到任何一层预测列")

        X = df[use_cols].copy()

        # 逐日 winsorize
        for c in use_cols:
            X[c] = _winsorize_by_date(X[c], self.winsor_limits_)

        # （可选）逐日 zscore
        if self.do_zscore_:
            for c in use_cols:
                X[c] = _zscore_by_date(X[c])

        # （可选）中性化
        if self.neutralize_cfg_:
            neutralize_cols = self.neutralize_cfg_.get('by', [])
            for col in neutralize_cols:
                if col in df.columns:
                    X[col] = df[col]
            X = _neutralize(X, cols=use_cols, cfg=self.neutralize_cfg_)
            X = X[use_cols]  # 只保留特征列

        # 合并处理后的特征回原数据
        out = df.copy()
        for c in use_cols:
            out[c] = X[c]

        return out

    def fit(self, df: pd.DataFrame) -> "LtrIsotonicStacker":
        """
        训练 LTR + Isotonic 模型

        Args:
            df: 包含一层预测和标签的数据，MultiIndex[(date,ticker)]

        Returns:
            self
        """
        logger.info("🚀 开始训练 LTR + Isotonic Stacker")

        df = self._preprocess(df)

        # 检查标签列
        label_col = None
        for col in ['ret_fwd_10d', 'target', 'returns_10d', 'label']:
            if col in df.columns:
                label_col = col
                break

        if label_col is None:
            raise ValueError("训练期需要标签列 (ret_fwd_10d/target/returns_10d/label)")

        logger.info(f"使用标签列: {label_col}")

        # 标签也裁剪稳健些（避免极端收益主导 NDCG）
        y = _winsorize_by_date(df[label_col], self.winsor_limits_)

        # 生成时序折（按日期）
        unique_dates = df.index.get_level_values('date').unique().sort_values().values
        splits = make_purged_splits(unique_dates, n_splits=self.n_splits_, embargo=self.embargo_)

        logger.info(f"生成 {len(splits)} 个时序CV折")

        # 收集 OOF 预测用于全局 Isotonic 校准
        oof_preds = []
        oof_y = []
        self.cv_scores_ = []

        # 确定实际使用的特征列
        actual_base_cols = [c for c in self.base_cols_ if c in df.columns]

        for fold_idx, (tr_idx, va_idx) in enumerate(splits):
            dates_tr = unique_dates[tr_idx]
            dates_va = unique_dates[va_idx]

            df_tr = df.loc[df.index.get_level_values('date').isin(dates_tr)]
            df_va = df.loc[df.index.get_level_values('date').isin(dates_va)]

            logger.info(f"Fold {fold_idx+1}/{len(splits)}: 训练 {len(df_tr)} 样本, 验证 {len(df_va)} 样本")

            X_tr = df_tr[actual_base_cols].values
            y_tr = y.loc[df_tr.index].values
            grp_tr = _group_sizes_by_date(df_tr)

            X_va = df_va[actual_base_cols].values
            y_va = y.loc[df_va.index].values
            grp_va = _group_sizes_by_date(df_va)

            # 创建 LightGBM 数据集
            dtrain = lgb.Dataset(X_tr, label=y_tr, group=grp_tr, free_raw_data=False)
            dvalid = lgb.Dataset(X_va, label=y_va, group=grp_va, free_raw_data=False)

            # 训练 ranker
            ranker = lgb.LGBMRanker(**self.lgbm_params_, random_state=self.random_state_)
            ranker.fit(
                X_tr, y_tr,
                group=grp_tr,
                eval_set=[(X_va, y_va)],
                eval_group=[grp_va],
                eval_metric=[_spearman_ic_eval, 'ndcg'],
                callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False),
                          lgb.log_evaluation(period=0)]
            )

            # 验证集预测（OOF）
            va_pred = ranker.predict(X_va, num_iteration=ranker.best_iteration_)
            oof_preds.append(pd.Series(va_pred, index=df_va.index))
            oof_y.append(pd.Series(y_va, index=df_va.index))

            # 计算验证分数
            ic_score = spearmanr(va_pred, y_va)[0]
            self.cv_scores_.append(ic_score)
            logger.info(f"Fold {fold_idx+1} IC: {ic_score:.4f}")

        # 合并OOF预测
        oof_preds = pd.concat(oof_preds).sort_index()
        oof_y = pd.concat(oof_y).sort_index()

        self.oof_predictions_ = oof_preds
        self.oof_targets_ = oof_y

        mean_ic = np.mean(self.cv_scores_)
        logger.info(f"📊 CV平均IC: {mean_ic:.4f} (std: {np.std(self.cv_scores_):.4f})")

        # 训练全局 Isotonic（保持单调、校正刻度）
        logger.info("训练全局 Isotonic 校准器...")
        self.calibrator_ = IsotonicRegression(out_of_bounds='clip')
        self.calibrator_.fit(oof_preds.values, oof_y.values)

        # 最终在全样本重训 ranker
        logger.info("在全样本重训最终 ranker...")
        X_all = df[actual_base_cols].values
        y_all = y.loc[df.index].values
        grp_all = _group_sizes_by_date(df)

        dtrain_all = lgb.Dataset(X_all, label=y_all, group=grp_all, free_raw_data=False)

        # 使用更保守的参数进行最终训练
        final_params = self.lgbm_params_.copy()
        final_params['n_estimators'] = min(final_params.get('n_estimators', 3000),
                                          int(np.mean([r.best_iteration_ for r in [ranker]])) + 500)

        self.ranker_ = lgb.LGBMRanker(**final_params, random_state=self.random_state_)
        self.ranker_.fit(
            X_all, y_all,
            group=grp_all,
            eval_set=[(X_all, y_all)],
            eval_group=[grp_all],
            eval_metric=[_spearman_ic_eval, 'ndcg'],
            callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False),
                      lgb.log_evaluation(period=0)]
        )

        # 保存特征重要性
        self.feature_importance_ = pd.DataFrame({
            'feature': actual_base_cols,
            'importance': self.ranker_.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("特征重要性:")
        for _, row in self.feature_importance_.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.0f}")

        self._col_cache_ = list(actual_base_cols)
        self.fitted_ = True

        logger.info("✅ LTR + Isotonic Stacker 训练完成")
        return self

    def predict(self, df_today: pd.DataFrame) -> pd.DataFrame:
        """
        对数据进行预测

        Args:
            df_today: 包含一层预测的数据，MultiIndex[(date,ticker)]

        Returns:
            包含预测分数的 DataFrame
        """
        if not self.fitted_:
            raise RuntimeError("请先 fit() 再 predict()")

        # 允许批量多日推理
        df_today = self._preprocess(df_today)

        # 使用训练时确定的特征列
        X = df_today[self._col_cache_].values

        # LTR 预测
        raw = self.ranker_.predict(X, num_iteration=self.ranker_.best_iteration_)

        # 全局单调校准
        cal = self.calibrator_.transform(raw)

        out = df_today.copy()
        out['score_raw'] = raw
        out['score'] = cal

        # 可选：输出日内rank / z
        def _rank(x):
            return pd.Series(rankdata(x, method='average'), index=x.index)

        out['score_rank'] = out.groupby(level='date')['score'].transform(_rank)
        out['score_z'] = out.groupby(level='date')['score'].transform(
            lambda x: (x-x.mean())/(x.std(ddof=0)+1e-12)
        )

        return out[['score_raw','score','score_rank','score_z']]

    def replace_ewa_in_pipeline(self, df_today: pd.DataFrame) -> pd.DataFrame:
        """
        作为"EWA替换件"的薄封装：返回一列 final score（已校准）
        原来拿 EWA 分数喂组合/回测的地方，直接换这列即可

        Args:
            df_today: 包含一层预测的数据

        Returns:
            单列 DataFrame，包含最终分数
        """
        scores = self.predict(df_today)
        return scores[['score']]

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息用于报告"""
        if not self.fitted_:
            return {'fitted': False}

        return {
            'fitted': True,
            'model_type': 'LTR + Isotonic Calibration',
            'base_features': self._col_cache_,
            'cv_mean_ic': np.mean(self.cv_scores_) if self.cv_scores_ else None,
            'cv_std_ic': np.std(self.cv_scores_) if self.cv_scores_ else None,
            'n_iterations': self.ranker_.best_iteration_ if self.ranker_ else None,
            'feature_importance': self.feature_importance_.to_dict() if self.feature_importance_ is not None else None,
            'calibrator_fitted': self.calibrator_ is not None,
            'horizon': self.horizon_
        }