#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Model Backtest - 完整的模型回测分析
================================================================
使用已训练模型在聚合因子数据上做滚动预测，分别统计每个模型的性能

任务：
1. 加载最新模型快照 (ElasticNet, XGBoost, CatBoost, LambdaRank, Ridge Stacking)
2. 在因子数据上每周做一次预测
3. 确保时间正确分割（防止信息泄漏）
4. 分别统计每个模型的性能：
   - IC & Rank IC
   - MSE/MAE/R²
   - Top 20% / Bottom 20% 分组收益
5. 对比Nasdaq的T+10收益
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_FACTORS_FILE = "data/factor_exports/factors/factors_all.parquet"
DEFAULT_FACTORS_DIR = str(Path(DEFAULT_FACTORS_FILE).parent)


class ComprehensiveModelBacktest:
    """完整的模型回测分析"""

    def __init__(
        self,
        data_dir: str = DEFAULT_FACTORS_DIR,
        snapshot_id: Optional[str] = None,
        data_file: Optional[str] = DEFAULT_FACTORS_FILE,
        tickers_file: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        allow_insample_backtest: bool = False,
        load_catboost: bool = False,
    ):
        """
        初始化回测引擎

        Args:
            data_dir: 因子数据所在目录（若 data_file 未指定则用于manifest加载）
            snapshot_id: 指定要加载的snapshot ID (None = 加载最新)
        """
        self.data_dir = data_dir
        self.data_file = data_file
        self.tickers_file = tickers_file
        self.models = {}
        self.ridge_stacker = None
        self.lambda_rank_stacker = None
        self.lambda_percentile_transformer = None
        self.snapshot_id = snapshot_id  # Store the requested snapshot_id
        self._user_start_date = self._parse_date(start_date)
        self._user_end_date = self._parse_date(end_date)
        self._resolved_eval_start: Optional[pd.Timestamp] = None
        self._resolved_eval_end: Optional[pd.Timestamp] = None
        self._resolved_eval_start_source: Optional[str] = None
        self._eval_window_initialized = False
        self._allow_insample_backtest = bool(allow_insample_backtest)
        # Must be set before _load_models() is called.
        self._load_catboost = bool(load_catboost)
        self._target_horizon_days = 10
        self._rebalance_mode = "horizon"
        self.training_start_date: Optional[pd.Timestamp] = None
        self.training_end_date: Optional[pd.Timestamp] = None

        # Feature mappings for each model
        self.model_features = {}

        # Kronos trade filter (optional; used only when requested in backtest)
        self._kronos_service = None
        self._kronos_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}  # (YYYY-MM-DD, TICKER) -> {pass, t5_ret, t0_price}
        self._yf_hist_cache: Dict[str, pd.DataFrame] = {}  # TICKER -> full daily OHLCV history (yfinance)

        # 加载模型
        self._load_models()
        self._extract_model_features()

    @staticmethod
    def _load_tickers_file(file_path: str) -> List[str]:
        """Load tickers (one per line; supports comma/space separated; ignores # comments)."""
        if not file_path:
            return []
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        tickers: List[str] = []
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p for token in line.split(",") for p in token.split()]
                for p in parts:
                    t = str(p).strip().strip("'\"").upper()
                    if t:
                        tickers.append(t)
        # de-dupe keep order
        tickers = list(dict.fromkeys(tickers))
        return tickers

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[pd.Timestamp]:
        """Parse CLI/manifest dates into normalized timestamps."""
        if date_str is None:
            return None
        if isinstance(date_str, str) and not date_str.strip():
            return None
        try:
            return pd.to_datetime(date_str).tz_localize(None).normalize()
        except Exception as exc:
            raise ValueError(f"Invalid date value: {date_str}" ) from exc

    def _extract_training_date_range(self, manifest: Dict[str, Any]) -> None:
        """Read training start/end dates from snapshot manifest if available."""
        try:
            metadata = (manifest or {}).get('metadata') or {}
            training_range = metadata.get('training_date_range') or {}
            start = training_range.get('start_date')
            end = training_range.get('end_date')
            if start:
                try:
                    self.training_start_date = self._parse_date(start)
                except ValueError as exc:
                    logger.warning(f"?? [MODEL] Failed to parse training start date {start}: {exc}")
            if end:
                try:
                    self.training_end_date = self._parse_date(end)
                except ValueError as exc:
                    logger.warning(f"?? [MODEL] Failed to parse training end date {end}: {exc}")
            if self.training_start_date or self.training_end_date:
                start_label = self.training_start_date.date() if self.training_start_date is not None else 'unknown'
                end_label = self.training_end_date.date() if self.training_end_date is not None else 'unknown'
                logger.info(f"?? [MODEL] Training window recorded in snapshot: {start_label} -> {end_label}")
        except Exception as exc:
            logger.debug(f"Training range metadata unavailable: {exc}")

    def _resolve_eval_window(self) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        if self._eval_window_initialized:
            return self._resolved_eval_start, self._resolved_eval_end
        start = self._user_start_date
        source = 'user' if start is not None else None
        if start is None and self.training_end_date is not None:
            try:
                gap_days = int(getattr(self, '_target_horizon_days', 10) or 10)
            except Exception:
                gap_days = 10
            gap_days = max(gap_days, 1)
            start = self.training_end_date + pd.Timedelta(days=gap_days)
            source = 'training_end'
        self._resolved_eval_start = start
        self._resolved_eval_end = self._user_end_date
        self._resolved_eval_start_source = source
        self._eval_window_initialized = True
        return self._resolved_eval_start, self._resolved_eval_end

    def _log_eval_span(self, date_index: pd.Index, count: int, note: str = '') -> None:
        if count == 0:
            return
        dates = pd.to_datetime(date_index).tz_localize(None).normalize()
        start_label = dates.min().date()
        end_label = dates.max().date()
        note_str = f" {note}" if note else ''
        logger.info(f"?? [EVAL WINDOW] Dates: {start_label} -> {end_label} ({count} rows){note_str}")

    def _filter_date_window(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restrict dataset to the resolved evaluation window (if any)."""
        if df is None or len(df) == 0:
            return df
        dates = pd.to_datetime(df.index.get_level_values('date')).tz_localize(None).normalize()
        start, end = self._resolve_eval_window()
        if start is None and end is None:
            self._log_eval_span(dates, len(df))
            return df
        mask = np.ones(len(df), dtype=bool)
        if start is not None:
            mask &= dates >= start
        if end is not None:
            mask &= dates <= end
        if not mask.any():
            if (
                start is not None
                and self._resolved_eval_start_source == 'training_end'
                and self._allow_insample_backtest
            ):
                logger.warning(
                    "?? [EVAL WINDOW] No rows after training end %s. allow_insample_backtest=True so running in-sample.",
                    start.date(),
                )
                # Disable previously resolved window to avoid repeated warnings
                self._resolved_eval_start = None
                self._resolved_eval_start_source = None
                self._eval_window_initialized = True
                self._log_eval_span(dates, len(df), note='(in-sample)')
                return df
            raise ValueError(
                "No data available inside the requested evaluation window. "
                "Consider adjusting --start-date/--end-date or pass --allow-insample to override."
            )
        filtered = df[mask]
        if len(filtered) != len(df):
            span_desc = [start.date() if start is not None else '-inf', end.date() if end is not None else '+inf']
            logger.info(
                "?? [EVAL WINDOW] Restricting data to %s -> %s (%d -> %d rows)",
                span_desc[0],
                span_desc[1],
                len(df),
                len(filtered),
            )
        self._log_eval_span(
            filtered.index.get_level_values('date'),
            len(filtered),
        )
        return filtered

    @staticmethod
    def _standardize_multiindex(data: pd.DataFrame) -> pd.DataFrame:
        """Ensure the concatenated dataset uses a clean MultiIndex(date, ticker)."""
        if data is None or len(data) == 0:
            return data

        if isinstance(data.index, pd.MultiIndex):
            level_names = [name.lower() if isinstance(name, str) else '' for name in data.index.names]
            if 'date' in level_names and ('ticker' in level_names or 'symbol' in level_names):
                date_level = level_names.index('date')
                other_level = 1 - date_level
                dates = pd.to_datetime(data.index.get_level_values(date_level)).tz_localize(None).normalize()
                tickers = data.index.get_level_values(other_level).astype(str).str.upper().str.strip()
            else:
                # Reset and rebuild if names are wrong
                df_reset = data.reset_index()
                if 'date' not in df_reset.columns or not any(col in df_reset.columns for col in ['ticker', 'symbol']):
                    raise ValueError('MultiIndex缺少 date/ticker 信息，无法标准化')
                df_reset['date'] = pd.to_datetime(df_reset['date']).dt.tz_localize(None).dt.normalize()
                ticker_col = 'ticker' if 'ticker' in df_reset.columns else 'symbol'
                df_reset['ticker'] = df_reset[ticker_col].astype(str).str.upper().str.strip()
                data = df_reset.drop(columns=[col for col in ['symbol'] if col in df_reset.columns])
                dates = df_reset['date']
                tickers = df_reset['ticker']
        else:
            if {'date', 'ticker'}.issubset(data.columns) or {'date', 'symbol'}.issubset(data.columns):
                df_reset = data.reset_index(drop=True)
                df_reset['date'] = pd.to_datetime(df_reset['date']).dt.tz_localize(None).dt.normalize()
                ticker_col = 'ticker' if 'ticker' in df_reset.columns else 'symbol'
                df_reset['ticker'] = df_reset[ticker_col].astype(str).str.upper().str.strip()
                data = df_reset
                dates = df_reset['date']
                tickers = df_reset['ticker']
            else:
                raise ValueError('数据缺少 date/ticker 列，无法构建MultiIndex')

        standardized = data.copy()
        standardized.index = pd.MultiIndex.from_arrays([dates, tickers], names=['date', 'ticker'])
        standardized = standardized[~standardized.index.duplicated(keep='last')]
        standardized = standardized.sort_index()
        return standardized

    @staticmethod
    def _prepare_feature_matrix(date_data: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Prepare features for a single cross-section without introducing ad-hoc zeros."""
        X = date_data[feature_cols].copy()
        # Remove obvious non-numeric columns
        for drop_col in ['target', 'Close', 'ret_fwd_5d']:
            X = X.drop(columns=[drop_col], errors='ignore')

        # Replace inf/-inf with NaN then fill using cross-sectional medians
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        if X.isna().any().any():
            medians = X.median(axis=0, skipna=True)
            X = X.fillna(medians)
            # Fallback for columns that were entirely NaN
            X = X.fillna(0)

        return X

    def _load_models(self):
        """加载模型快照 (最新或指定ID)"""
        logger.info("=" * 80)
        if self.snapshot_id:
            logger.info(f"📦 加载指定模型快照: {self.snapshot_id}")
        else:
            logger.info("📦 加载最新模型快照")
        logger.info("=" * 80)

        try:
            from bma_models.model_registry import load_manifest, load_models_from_snapshot

            # Load specified snapshot or latest
            manifest = load_manifest(snapshot_id=self.snapshot_id)
            self.snapshot_id = manifest['snapshot_id']  # Update with actual loaded ID
            self._extract_training_date_range(manifest)

            logger.info(f"快照ID: {self.snapshot_id}")
            logger.info(f"创建时间: {datetime.fromtimestamp(manifest['created_at']).strftime('%Y-%m-%d %H:%M:%S')}")

            # Load all models
            # CatBoost can be heavy/fragile to import in some environments; default is disabled.
            # Certain snapshots may require 'pred_catboost' as a ridge stacking base column; in that case,
            # callers can set `self._load_catboost = True` (or pass `--load-catboost` via CLI if available).
            loaded = load_models_from_snapshot(self.snapshot_id, load_catboost=bool(getattr(self, "_load_catboost", False)))

            self.models = loaded['models']
            self.ridge_stacker = loaded.get('ridge_stacker')
            self.lambda_rank_stacker = loaded.get('lambda_rank_stacker')
            self.lambda_percentile_transformer = loaded.get('lambda_percentile_transformer')

            # Count loaded models
            model_names = []
            if 'elastic_net' in self.models and self.models['elastic_net'] is not None:
                model_names.append("ElasticNet")
            if 'xgboost' in self.models and self.models['xgboost'] is not None:
                model_names.append("XGBoost")
            if 'catboost' in self.models and self.models['catboost'] is not None:
                model_names.append("CatBoost")
            if self.lambda_rank_stacker is not None:
                model_names.append("LambdaRank")
            if self.ridge_stacker is not None:
                model_names.append("Ridge Stacking")

            logger.info(f"✅ 成功加载 {len(model_names)} 个模型: {', '.join(model_names)}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _extract_model_features(self):
        """提取每个模型期望的特征列表"""
        logger.info("提取模型特征信息...")

        # ElasticNet
        if 'elastic_net' in self.models and self.models['elastic_net'] is not None:
            en = self.models['elastic_net']
            if hasattr(en, 'feature_names_in_'):
                self.model_features['elastic_net'] = list(en.feature_names_in_)
                logger.info(f"  ElasticNet: {len(self.model_features['elastic_net'])} features")

        # XGBoost
        if 'xgboost' in self.models and self.models['xgboost'] is not None:
            xgb = self.models['xgboost']
            if hasattr(xgb, 'get_booster'):
                booster = xgb.get_booster()
                self.model_features['xgboost'] = booster.feature_names
                logger.info(f"  XGBoost: {len(self.model_features['xgboost'])} features")

        # CatBoost
        if 'catboost' in self.models and self.models['catboost'] is not None:
            cat = self.models['catboost']
            if hasattr(cat, 'feature_names_'):
                self.model_features['catboost'] = list(cat.feature_names_)
                logger.info(f"  CatBoost: {len(self.model_features['catboost'])} features")

        # LambdaRank - use its own alpha_factor_cols (16 features)
        if self.lambda_rank_stacker is not None:
            if hasattr(self.lambda_rank_stacker, '_alpha_factor_cols') and self.lambda_rank_stacker._alpha_factor_cols:
                self.model_features['lambdarank'] = list(self.lambda_rank_stacker._alpha_factor_cols)
                logger.info(f"  LambdaRank: {len(self.model_features['lambdarank'])} features")
            elif hasattr(self.lambda_rank_stacker, 'base_cols') and self.lambda_rank_stacker.base_cols:
                self.model_features['lambdarank'] = list(self.lambda_rank_stacker.base_cols)
                logger.info(f"  LambdaRank: {len(self.model_features['lambdarank'])} features (from base_cols)")

    def load_factor_data(self) -> pd.DataFrame:
        """
        加载因子数据（单文件或目录）

        Returns:
            合并后的完整数据集
        """
        # Fast path: load a single parquet file (used for tuning / quick backtests)
        if self.data_file:
            logger.info(f"📊 加载数据文件: {self.data_file}")
            single = pd.read_parquet(self.data_file)
            single = self._standardize_multiindex(single)
            logger.info(f"✅ 总数据: {single.shape}")
            logger.info(f"   日期范围: {single.index.get_level_values('date').min()} 至 {single.index.get_level_values('date').max()}")
            logger.info(f"   股票数量: {single.index.get_level_values('ticker').nunique()}")
            single = self._apply_universe_filter(single)
            single = self._filter_date_window(single)
            return single

        # Backward-compatible: allow data_dir to be a single parquet
        if isinstance(self.data_dir, str) and self.data_dir.lower().endswith(".parquet") and os.path.exists(self.data_dir):
            logger.info(f"📊 加载数据文件: {self.data_dir}")
            single = pd.read_parquet(self.data_dir)
            single = self._standardize_multiindex(single)
            logger.info(f"✅ 总数据: {single.shape}")
            logger.info(f"   日期范围: {single.index.get_level_values('date').min()} 至 {single.index.get_level_values('date').max()}")
            logger.info(f"   股票数量: {single.index.get_level_values('ticker').nunique()}")
            single = self._apply_universe_filter(single)
            single = self._filter_date_window(single)
            return single

            logger.info("📊 加载因子数据...")

        # Read manifest
        manifest_path = os.path.join(self.data_dir, "manifest.parquet")
        manifest = pd.read_parquet(manifest_path)

        logger.info(f"发现 {len(manifest)} 个批次文件")

        # Load all batches
        all_data = []
        for idx, row in manifest.iterrows():
            batch_id = row['batch_id']
            # Prefer explicit path from manifest (supports custom shard names like factors_batch_XXXX.parquet)
            batch_file = None
            try:
                mf = row.get('file', None)
                if isinstance(mf, str) and mf.strip():
                    candidate = mf
                    # If manifest stores relative paths, resolve relative to repo root and to data_dir
                    if os.path.exists(candidate):
                        batch_file = candidate
                    else:
                        cand2 = os.path.join(self.data_dir, os.path.basename(candidate))
                        if os.path.exists(cand2):
                            batch_file = cand2
                        else:
                            cand3 = os.path.join(os.getcwd(), candidate)
                            if os.path.exists(cand3):
                                batch_file = cand3
            except Exception:
                batch_file = None

            # Backward-compatible fallback naming
            if not batch_file:
                legacy = os.path.join(self.data_dir, f"polygon_factors_batch_{batch_id:04d}.parquet")
                if os.path.exists(legacy):
                    batch_file = legacy
                else:
                    alt = os.path.join(self.data_dir, f"factors_batch_{batch_id:04d}.parquet")
                    if os.path.exists(alt):
                        batch_file = alt

            if batch_file and os.path.exists(batch_file):
                batch_data = pd.read_parquet(batch_file)
                all_data.append(batch_data)
                logger.info(f"  ✅ Batch {batch_id}: {batch_data.shape} ({os.path.basename(batch_file)})")

        if not all_data:
            raise ValueError("没有可用的因子数据")

        # Concatenate all batches and normalize index
        full_data = pd.concat(all_data, axis=0)
        full_data = self._standardize_multiindex(full_data)

        logger.info(f"✅ 总数据: {full_data.shape}")
        logger.info(f"   日期范围: {full_data.index.get_level_values('date').min()} 至 {full_data.index.get_level_values('date').max()}")
        logger.info(f"   股票数量: {full_data.index.get_level_values('ticker').nunique()}")

        filtered = self._apply_universe_filter(full_data)
        filtered = self._filter_date_window(filtered)
        return filtered

    def _apply_universe_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter MultiIndex(date,ticker) to tickers_file universe if provided."""
        if df is None or len(df) == 0:
            return df
        if not self.tickers_file:
            return df
        try:
            tickers = self._load_tickers_file(self.tickers_file)
            if not tickers:
                return df
            before = int(df.index.get_level_values("ticker").nunique())
            keep = set(tickers)
            mask = df.index.get_level_values("ticker").astype(str).str.upper().str.strip().isin(keep)
            out = df.loc[mask].copy()
            after = int(out.index.get_level_values("ticker").nunique()) if len(out) else 0
            logger.info(f"📌 [UNIVERSE] 回测股票池过滤: {before} → {after} (file={self.tickers_file})")
            return out
        except Exception as e:
            logger.warning(f"⚠️ [UNIVERSE] 过滤失败，使用原始数据: {e}")
            return df

    def get_weekly_dates(self, data: pd.DataFrame) -> List[pd.Timestamp]:
        """
        获取每周的预测日期（每周一）

        Args:
            data: 完整数据集

        Returns:
            每周一的日期列表
        """
        all_dates = data.index.get_level_values('date').unique().sort_values()

        # Convert to datetime
        all_dates = pd.to_datetime(all_dates)

        # Get weekly dates (Monday of each week)
        weekly_dates = []
        current_week = None

        for date in all_dates:
            week = date.isocalendar()[1]  # Week number
            year = date.year
            week_key = (year, week)

            if week_key != current_week:
                weekly_dates.append(date)
                current_week = week_key

        logger.info(f"📅 生成 {len(weekly_dates)} 个每周预测日期")

        return weekly_dates

    def get_rebalance_dates(
        self,
        data: pd.DataFrame,
        rebalance_mode: str = "horizon",
        target_horizon_days: int = 10,
    ) -> List[pd.Timestamp]:
        """
        Get rebalance dates for rolling prediction.

        Why:
        - Our `actual` return uses `target`, which is computed as T+H forward return.
        - If we rebalance weekly while H=10 trading days, returns overlap (double-count / autocorrelate).
        - Using rebalance_mode='horizon' fixes overlap by stepping every H trading days.

        Modes:
        - weekly: first trading day of each ISO week (legacy behavior)
        - horizon: every `target_horizon_days` trading days (non-overlapping vs target horizon)
        """
        mode = (rebalance_mode or "horizon").strip().lower()
        if mode == "weekly":
            return self.get_weekly_dates(data)

        # horizon mode (default)
        all_dates = data.index.get_level_values("date").unique().sort_values()
        all_dates = pd.to_datetime(all_dates)

        try:
            step = int(target_horizon_days)
        except Exception:
            step = 10
        if step <= 0:
            step = 10

        rebalance_dates = list(all_dates[::step])
        logger.info(f"📅 生成 {len(rebalance_dates)} 个回测调仓日期 (mode={mode}, step={step} trading days)")
        return rebalance_dates

    def predict_single_model(self,
                            model_name: str,
                            model,
                            X: pd.DataFrame,
                            required_features: List[str] = None,
                            pred_date: pd.Timestamp = None) -> pd.Series:
        """
        使用单个模型进行预测

        Args:
            model_name: 模型名称
            model: 模型对象
            X: 特征矩阵（包含所有可用特征）
            required_features: 模型需要的特征列表
            pred_date: 预测日期（LambdaRank需要用于重建MultiIndex）

        Returns:
            预测结果 (pd.Series)
        """
        try:
            if model is None:
                logger.warning(f"⚠️ {model_name} 模型为空")
                return None

            # Select only required features
            if required_features is not None:
                missing_cols = [col for col in required_features if col not in X.columns]
                if missing_cols:
                    logger.warning(f"⚠️ {model_name} 缺少特征列: {missing_cols[:5]}")
                    return None
                X_model = X[required_features].copy()
            else:
                X_model = X.copy()

            # Make prediction
            if model_name == 'lambdarank':
                # LambdaRank needs MultiIndex (date, ticker) for groupby operations
                if pred_date is not None:
                    # Reconstruct MultiIndex
                    X_model.index = pd.MultiIndex.from_arrays(
                        [[pred_date] * len(X_model), X_model.index],
                        names=['date', 'ticker']
                    )

                # LambdaRank.predict returns DataFrame with lambda_score, lambda_rank, lambda_pct
                pred_result = model.predict(X_model)

                if isinstance(pred_result, pd.DataFrame) and 'lambda_score' in pred_result.columns:
                    predictions = pred_result['lambda_score']
                elif isinstance(pred_result, pd.Series):
                    predictions = pred_result
                else:
                    predictions = pd.Series(pred_result, index=X_model.index)
            else:
                raw_pred = model.predict(X_model)
                predictions = pd.Series(raw_pred, index=X_model.index)

            # Ensure index matches
            pred_series = predictions.copy()
            pred_series.index = X.index
            pred_series.name = model_name

            return pred_series

        except Exception as e:
            logger.error(f"❌ {model_name} 预测失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def rolling_prediction(self, data: pd.DataFrame, max_weeks: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        滚动预测：在每个时间点使用可用数据进行预测

        Args:
            data: 完整数据集（包含target）

        Returns:
            每个模型的预测结果字典 {model_name: predictions_df}
        """
        logger.info("=" * 80)
        logger.info("🔮 开始滚动预测")
        logger.info("=" * 80)

        # Get rebalance dates (default: horizon-step to avoid overlapping T+H targets)
        rebalance_mode = getattr(self, "_rebalance_mode", "horizon")
        target_horizon_days = int(getattr(self, "_target_horizon_days", 10) or 10)
        weekly_dates = self.get_rebalance_dates(
            data,
            rebalance_mode=str(rebalance_mode),
            target_horizon_days=target_horizon_days,
        )
        if max_weeks is not None:
            try:
                max_weeks_int = int(max_weeks)
            except Exception:
                max_weeks_int = None
            if max_weeks_int is not None and max_weeks_int > 0 and len(weekly_dates) > max_weeks_int:
                weekly_dates = weekly_dates[-max_weeks_int:]
                logger.info(f"⏱️ 回测加速: 仅使用最后 {max_weeks_int} 个weekly dates用于滚动预测")

        # Get all available features from data (exclude labels/price columns)
        exclude_cols = {'target', 'Close', 'ret_fwd_5d'}
        all_feature_cols = [col for col in data.columns if col not in exclude_cols]
        logger.info(f"数据包含 {len(all_feature_cols)} 个特征")

        # Initialize results storage
        all_predictions = {
            'elastic_net': [],
            'xgboost': [],
            'catboost': [],
            'lambdarank': [],
            'ridge_stacking': []
        }

        # Rolling prediction
        for i, pred_date in enumerate(weekly_dates):
            # Get features for this exact date (for prediction)
            try:
                # Use xs to cross-section by date (drop_level=True to get ticker index only)
                date_data = data.xs(pred_date, level='date', drop_level=True)
            except KeyError:
                # This date might not exist in the index
                continue

            if len(date_data) == 0:
                continue

            # Prepare features - use production-style handling
            X = self._prepare_feature_matrix(date_data, all_feature_cols)
            if X is None or X.empty:
                continue

            # Get actual target (align with filtered tickers)
            if 'target' in date_data.columns:
                actual_target = date_data.loc[X.index, 'target']
            else:
                actual_target = pd.Series(np.nan, index=X.index)

            tickers = X.index.tolist()

            # Predict with each model (using model-specific features)
            # 1. ElasticNet
            if 'elastic_net' in self.models and self.models['elastic_net'] is not None:
                required_features = self.model_features.get('elastic_net')
                pred = self.predict_single_model('elastic_net', self.models['elastic_net'], X, required_features)
                if pred is not None:
                    pred_df = pd.DataFrame({
                        'date': pred_date,
                        'ticker': tickers,
                        'prediction': pred.values,
                        'actual': actual_target.values
                    })
                    all_predictions['elastic_net'].append(pred_df)

            # 2. XGBoost
            if 'xgboost' in self.models and self.models['xgboost'] is not None:
                required_features = self.model_features.get('xgboost')
                pred = self.predict_single_model('xgboost', self.models['xgboost'], X, required_features)
                if pred is not None:
                    pred_df = pd.DataFrame({
                        'date': pred_date,
                        'ticker': tickers,
                        'prediction': pred.values,
                        'actual': actual_target.values
                    })
                    all_predictions['xgboost'].append(pred_df)

            # 3. CatBoost
            if 'catboost' in self.models and self.models['catboost'] is not None:
                required_features = self.model_features.get('catboost')
                pred = self.predict_single_model('catboost', self.models['catboost'], X, required_features)
                if pred is not None:
                    pred_df = pd.DataFrame({
                        'date': pred_date,
                        'ticker': tickers,
                        'prediction': pred.values,
                        'actual': actual_target.values
                    })
                    all_predictions['catboost'].append(pred_df)

            # 4. LambdaRank
            if self.lambda_rank_stacker is not None:
                required_features = self.model_features.get('lambdarank')
                pred = self.predict_single_model('lambdarank', self.lambda_rank_stacker, X, required_features, pred_date=pred_date)
                if pred is not None:
                    pred_df = pd.DataFrame({
                        'date': pred_date,
                        'ticker': tickers,
                        'prediction': pred.values,
                        'actual': actual_target.values
                    })
                    all_predictions['lambdarank'].append(pred_df)

            # 5. Ridge Stacking (combine first 4 models)
            if self.ridge_stacker is not None:
                # Create stacking features
                stacking_features = pd.DataFrame(index=X.index)

                # Note: RidgeStacker expects specific column names from training
                if 'elastic_net' in self.models and self.models['elastic_net'] is not None:
                    en_features = self.model_features.get('elastic_net')
                    en_pred = self.predict_single_model('elastic_net', self.models['elastic_net'], X, en_features)
                    if en_pred is not None:
                        stacking_features['pred_elastic'] = en_pred  # Ridge expects 'pred_elastic'

                if 'xgboost' in self.models and self.models['xgboost'] is not None:
                    xgb_features = self.model_features.get('xgboost')
                    xgb_pred = self.predict_single_model('xgboost', self.models['xgboost'], X, xgb_features)
                    if xgb_pred is not None:
                        stacking_features['pred_xgb'] = xgb_pred  # Ridge expects 'pred_xgb'

                if 'catboost' in self.models and self.models['catboost'] is not None:
                    cat_features = self.model_features.get('catboost')
                    cat_pred = self.predict_single_model('catboost', self.models['catboost'], X, cat_features)
                    if cat_pred is not None:
                        stacking_features['pred_catboost'] = cat_pred  # Ridge expects 'pred_catboost'

                # Optional LambdaRank input if RidgeStacker was trained with it
                try:
                    ridge_expected = list(getattr(self.ridge_stacker, 'actual_feature_cols_', None) or getattr(self.ridge_stacker, 'base_cols', []) or [])
                except Exception:
                    ridge_expected = []
                if 'pred_lambdarank' in ridge_expected and self.lambda_rank_stacker is not None:
                    lambda_features = self.model_features.get('lambdarank')
                    lambda_pred = self.predict_single_model('lambdarank', self.lambda_rank_stacker, X, lambda_features, pred_date=pred_date)
                    if lambda_pred is not None:
                        stacking_features['pred_lambdarank'] = lambda_pred

                if len(stacking_features.columns) > 0:
                    # RidgeStacker expects MultiIndex (date, ticker)
                    stacking_features_with_date = stacking_features.copy()
                    stacking_features_with_date.index = pd.MultiIndex.from_arrays(
                        [[pred_date] * len(tickers), tickers],
                        names=['date', 'ticker']
                    )

                    # Ensure all required cols exist (fill missing with 0.0 so ridge can still run)
                    try:
                        expected_cols = list(getattr(self.ridge_stacker, 'actual_feature_cols_', None) or getattr(self.ridge_stacker, 'base_cols', []) or [])
                    except Exception:
                        expected_cols = []
                    for col in expected_cols:
                        if col not in stacking_features_with_date.columns:
                            stacking_features_with_date[col] = 0.0

                    ridge_pred = self.ridge_stacker.predict(stacking_features_with_date)

                    # Ensure ridge_pred is 1D
                    if isinstance(ridge_pred, pd.DataFrame):
                        ridge_pred = ridge_pred.values.ravel()
                    elif isinstance(ridge_pred, pd.Series):
                        ridge_pred = ridge_pred.values
                    elif isinstance(ridge_pred, np.ndarray) and ridge_pred.ndim > 1:
                        ridge_pred = ridge_pred.ravel()

                    pred_df = pd.DataFrame({
                        'date': pred_date,
                        'ticker': tickers,
                        'prediction': ridge_pred,
                        'actual': actual_target.values
                    })
                    all_predictions['ridge_stacking'].append(pred_df)

            if (i + 1) % 10 == 0:
                logger.info(f"  进度: {i+1}/{len(weekly_dates)} ({(i+1)/len(weekly_dates)*100:.1f}%)")

        # Concatenate all predictions
        results = {}
        for model_name, pred_list in all_predictions.items():
            if len(pred_list) > 0:
                results[model_name] = pd.concat(pred_list, axis=0, ignore_index=True)
                logger.info(f"✅ {model_name}: {len(results[model_name])} 条预测")

        logger.info("=" * 80)
        return results

    def calculate_metrics(self, predictions: pd.DataFrame) -> Dict:
        """
        计算性能指标

        Args:
            predictions: 预测结果 DataFrame (date, ticker, prediction, actual)

        Returns:
            性能指标字典
        """
        # Remove NaN
        valid_data = predictions.dropna(subset=['prediction', 'actual'])

        if len(valid_data) == 0:
            logger.warning("⚠️ 没有有效的预测数据")
            return {}

        y_true = valid_data['actual'].values
        y_pred = valid_data['prediction'].values

        # Calculate metrics
        metrics = {}

        # IC (Pearson correlation)
        ic, ic_pval = pearsonr(y_pred, y_true)
        metrics['IC'] = ic
        metrics['IC_pvalue'] = ic_pval

        # Rank IC (Spearman correlation)
        rank_ic, rank_ic_pval = spearmanr(y_pred, y_true)
        metrics['Rank_IC'] = rank_ic
        metrics['Rank_IC_pvalue'] = rank_ic_pval

        # MSE, MAE, R²
        metrics['MSE'] = mean_squared_error(y_true, y_pred)
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)
        metrics['R2'] = r2_score(y_true, y_pred)

        return metrics

    def calculate_group_returns(
        self,
        predictions: pd.DataFrame,
        quantile: float = 0.2,
        top_n: Optional[int] = 30,
        bottom_n: Optional[int] = 30,
        cost_bps: float = 0.0,
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        计算分组收益

        - 默认使用固定Top/Bottom N（Top 30 / Bottom 30）
        - 若 top_n/bottom_n 为 None，则回退使用 quantile（Top/Bottom 分位数）

        Args:
            predictions: 预测结果 DataFrame
            quantile: 分位数（仅当 top_n/bottom_n 为 None 时生效）
            top_n: 固定Top N（默认30）
            bottom_n: 固定Bottom N（默认30）

        Returns:
            分组收益字典
        """
        results = []
        cost_bps = float(cost_bps or 0.0)
        cost_rate = cost_bps / 10000.0

        # Turnover-based transaction costs for the long Top-N basket (equal weight).
        # We iterate dates in sorted order to ensure turnover is computed sequentially.
        prev_long: Dict[str, float] = {}

        for date in sorted(pd.to_datetime(predictions["date"]).unique()):
            date_group = predictions[predictions["date"] == date]
            # Remove NaN
            valid_group = date_group.dropna(subset=['prediction', 'actual'])

            if len(valid_group) < 10:  # Need minimum stocks
                continue

            # Sort by prediction
            sorted_group = valid_group.sort_values('prediction', ascending=False)

            # Determine group sizes
            if top_n is None:
                n_top = max(1, int(len(sorted_group) * quantile))
            else:
                n_top = max(1, min(int(top_n), len(sorted_group)))
            if bottom_n is None:
                n_bottom = max(1, int(len(sorted_group) * quantile))
            else:
                n_bottom = max(1, min(int(bottom_n), len(sorted_group)))

            # Get top and bottom groups
            top_group = sorted_group.head(n_top)
            bottom_group = sorted_group.tail(n_bottom)

            # Calculate average returns
            top_return = top_group['actual'].mean()
            bottom_return = bottom_group['actual'].mean()
            all_return = sorted_group['actual'].mean()

            # Costs: apply only to the long Top-N basket.
            tickers = top_group["ticker"].astype(str).str.upper().str.strip().tolist()
            n = len(tickers)
            if n > 0:
                w = 1.0 / float(n)
                new_long = {t: w for t in tickers}
                union = set(prev_long.keys()) | set(new_long.keys())
                turnover = float(sum(abs(new_long.get(t, 0.0) - prev_long.get(t, 0.0)) for t in union))
            else:
                new_long = {}
                turnover = 0.0
            cost = float(turnover * cost_rate) if cost_rate > 0 else 0.0
            top_return_net = float(top_return) - cost if pd.notna(top_return) else np.nan
            prev_long = new_long

            results.append({
                'date': date,
                'top_return': top_return,
                'top_return_net': top_return_net,
                'bottom_return': bottom_return,
                'all_return': all_return,
                'long_short': top_return - bottom_return,
                'top_turnover': turnover,
                'top_cost': cost,
                'n_stocks': len(valid_group),
                'top_n': n_top,
                'bottom_n': n_bottom,
                'quantile': quantile if (top_n is None or bottom_n is None) else np.nan,
            })

        if len(results) == 0:
            return {}, pd.DataFrame()

        results_df = pd.DataFrame(results)

        # Calculate summary statistics
        # Sharpe conventions:
        # - For rebalance_mode='horizon', dates are spaced ~target_horizon_days trading days (non-overlapping).
        # - We annualize per-period Sharpe using sqrt(periods_per_year), where periods_per_year ≈ 252/target_horizon_days.
        def _annualized_sharpe(r: pd.Series, periods_per_year: float) -> float:
            rr = pd.to_numeric(r, errors="coerce").dropna()
            if rr.empty:
                return float("nan")
            mu = float(rr.mean())
            sd = float(rr.std())
            if sd <= 0:
                return float("nan")
            return float((mu / sd) * np.sqrt(periods_per_year))

        if str(getattr(self, "_rebalance_mode", "horizon")) == "horizon":
            h = float(getattr(self, "_target_horizon_days", 10) or 10)
            periods_per_year = float(252.0 / max(1.0, h))
        else:
            periods_per_year = 52.0

        summary = {
            'avg_top_return': results_df['top_return'].mean(),
            'avg_top_return_net': results_df['top_return_net'].mean() if 'top_return_net' in results_df.columns else np.nan,
            'avg_bottom_return': results_df['bottom_return'].mean(),
            'avg_all_return': results_df['all_return'].mean(),
            'avg_long_short': results_df['long_short'].mean(),
            'long_short_sharpe': results_df['long_short'].mean() / results_df['long_short'].std() if results_df['long_short'].std() > 0 else 0,
            'win_rate': (results_df['long_short'] > 0).sum() / len(results_df),
            # Long-only metrics (做多 Top-N) — this is what you want when not trading Bottom-N.
            'top_sharpe': _annualized_sharpe(results_df['top_return'], periods_per_year),
            'top_sharpe_net': _annualized_sharpe(results_df['top_return_net'], periods_per_year) if 'top_return_net' in results_df.columns else np.nan,
            'top_win_rate': float((results_df['top_return'] > 0).mean()),
            'top_win_rate_net': float((results_df['top_return_net'] > 0).mean()) if 'top_return_net' in results_df.columns else np.nan,
            'avg_top_turnover': results_df['top_turnover'].mean() if 'top_turnover' in results_df.columns else np.nan,
            'avg_top_cost': results_df['top_cost'].mean() if 'top_cost' in results_df.columns else np.nan,
            'cost_bps': cost_bps,
        }

        return summary, results_df

    @staticmethod
    def _make_rank_buckets(max_rank: int = 150, step: int = 10) -> List[Tuple[int, int]]:
        """
        Build 1-based inclusive rank buckets.
        Example: max_rank=30, step=10 -> [(1,10),(11,20),(21,30)]
        """
        try:
            max_rank_i = int(max_rank)
        except Exception:
            max_rank_i = 150
        try:
            step_i = int(step)
        except Exception:
            step_i = 10
        max_rank_i = max(1, max_rank_i)
        step_i = max(1, step_i)

        buckets: List[Tuple[int, int]] = []
        start = 1
        while start <= max_rank_i:
            end = min(start + step_i - 1, max_rank_i)
            buckets.append((start, end))
            start = end + 1
        return buckets

    def calculate_bucket_returns(
        self,
        predictions: pd.DataFrame,
        top_buckets: List[Tuple[int, int]] | None = None,
        bottom_buckets: List[Tuple[int, int]] | None = None,
        cost_bps: float = 0.0,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Calculate bucketed average returns based on predicted rank per date.

        Buckets are 1-based inclusive ranges.
        Example top_buckets=[(1,10),(11,20)] means take mean(actual) for ranks 1-10 and 11-20.
        """
        top_buckets = top_buckets or self._make_rank_buckets(max_rank=150, step=10)
        bottom_buckets = bottom_buckets or self._make_rank_buckets(max_rank=150, step=10)  # bottom ranks counted from the end
        cost_bps = float(cost_bps or 0.0)
        cost_rate = cost_bps / 10000.0

        # Per-bucket turnover tracking (each bucket treated as a separate equal-weight portfolio)
        prev_top_bucket: Dict[Tuple[int, int], Dict[str, float]] = {tuple(b): {} for b in top_buckets}

        rows: List[Dict[str, Any]] = []
        for date, date_group in predictions.groupby("date"):
            valid = date_group.dropna(subset=["prediction", "actual"])
            if len(valid) < 20:
                continue

            sorted_group = valid.sort_values("prediction", ascending=False).reset_index(drop=True)
            n = len(sorted_group)

            row: Dict[str, Any] = {"date": pd.to_datetime(date), "n_stocks": n}

            # Top buckets (1-based rank from best prediction)
            for a, b in top_buckets:
                if a <= n:
                    aa = max(1, a)
                    bb = min(b, n)
                    s = sorted_group.iloc[aa - 1 : bb]["actual"]
                    row[f"top_{aa}_{bb}_return"] = float(s.mean()) if len(s) else np.nan
                    # Net-of-cost return for this bucket portfolio
                    tickers = sorted_group.iloc[aa - 1 : bb]["ticker"].astype(str).str.upper().str.strip().tolist()
                    m = len(tickers)
                    if m > 0:
                        w = 1.0 / float(m)
                        new_w = {t: w for t in tickers}
                        prev_w = prev_top_bucket.get((a, b), {})
                        union = set(prev_w.keys()) | set(new_w.keys())
                        turnover = float(sum(abs(new_w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in union))
                    else:
                        new_w = {}
                        turnover = 0.0
                    cost = float(turnover * cost_rate) if cost_rate > 0 else 0.0
                    gross = row[f"top_{aa}_{bb}_return"]
                    row[f"top_{aa}_{bb}_return_net"] = float(gross) - cost if pd.notna(gross) else np.nan
                    row[f"top_{aa}_{bb}_turnover"] = turnover
                    row[f"top_{aa}_{bb}_cost"] = cost
                    prev_top_bucket[(a, b)] = new_w
                else:
                    row[f"top_{a}_{b}_return"] = np.nan
                    row[f"top_{a}_{b}_return_net"] = np.nan
                    row[f"top_{a}_{b}_turnover"] = np.nan
                    row[f"top_{a}_{b}_cost"] = np.nan

            # Bottom buckets (1-based rank from worst prediction)
            for a, b in bottom_buckets:
                if a <= n:
                    aa = max(1, a)
                    bb = min(b, n)
                    start = max(0, n - bb)
                    end = n - (aa - 1)
                    s = sorted_group.iloc[start:end]["actual"]
                    row[f"bottom_{aa}_{bb}_return"] = float(s.mean()) if len(s) else np.nan
                else:
                    row[f"bottom_{a}_{b}_return"] = np.nan

            rows.append(row)

        if not rows:
            return {}, pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("date")

        summary: Dict[str, Any] = {}
        for col in df.columns:
            if col.endswith("_return") and col != "all_return":
                if df[col].notna().any():
                    summary[f"avg_{col}"] = float(df[col].mean())
                else:
                    summary[f"avg_{col}"] = float("nan")
            if col.endswith("_return_net"):
                if df[col].notna().any():
                    summary[f"avg_{col}"] = float(df[col].mean())
                else:
                    summary[f"avg_{col}"] = float("nan")
        summary["cost_bps"] = cost_bps
        return summary, df

    def _get_kronos_service(self):
        """Lazy-load KronosService to avoid heavy init unless enabled."""
        if self._kronos_service is not None:
            return self._kronos_service
        try:
            from kronos.kronos_service import KronosService
            svc = KronosService()
            ok = svc.initialize(model_size="base")
            if not ok:
                logger.warning("⚠️ Kronos initialize failed; Kronos filter will be disabled")
                self._kronos_service = None
                return None
            self._kronos_service = svc
            return self._kronos_service
        except Exception as e:
            logger.warning(f"⚠️ Kronos import/init failed: {e}")
            self._kronos_service = None
            return None

    def _kronos_pass_for(self, ticker: str, as_of_date: pd.Timestamp, min_price: float = 10.0) -> Dict[str, Any]:
        """Compute Kronos pass for (ticker, date) with caching, using yfinance history up to as_of_date."""
        key = (pd.to_datetime(as_of_date).strftime("%Y-%m-%d"), str(ticker).upper())
        if key in self._kronos_cache:
            return self._kronos_cache[key]

        svc = self._get_kronos_service()
        if svc is None:
            out = {"kronos_pass": False, "t5_return_pct": None, "t0_price": None}
            self._kronos_cache[key] = out
            return out

        try:
            hist_full = self._yf_hist_cache.get(key[1])
            hist_slice = None
            if hist_full is not None and not hist_full.empty:
                # Slice up to as_of_date (leak-free) and keep ~1y worth of daily bars
                try:
                    hist_slice = hist_full[hist_full.index <= pd.to_datetime(as_of_date)].tail(260)
                except Exception:
                    hist_slice = hist_full.tail(260)

            res = svc.predict_stock(
                symbol=key[1],
                period="1y",
                interval="1d",
                pred_len=5,
                model_size="base",
                temperature=0.1,
                end_date=pd.to_datetime(as_of_date).to_pydatetime(),
                historical_df=hist_slice,
            )
            if res.get("status") != "success":
                out = {"kronos_pass": False, "t5_return_pct": None, "t0_price": None}
                self._kronos_cache[key] = out
                return out

            hist = res["historical_data"]
            cur_px = float(hist["close"].iloc[-1])
            pred_df = res["predictions"]
            if isinstance(pred_df, pd.DataFrame):
                t5_px = float(pred_df["close"].iloc[4]) if "close" in pred_df.columns else float(pred_df.iloc[4, -1])
            else:
                t5_px = float("nan")
            t5_ret = (t5_px - cur_px) / cur_px if cur_px and np.isfinite(cur_px) else float("nan")

            passed = bool(np.isfinite(t5_ret) and (t5_ret > 0) and (cur_px > float(min_price)))
            out = {"kronos_pass": passed, "t5_return_pct": (t5_ret * 100.0) if np.isfinite(t5_ret) else None, "t0_price": cur_px}
            self._kronos_cache[key] = out
            return out
        except Exception:
            out = {"kronos_pass": False, "t5_return_pct": None, "t0_price": None}
            self._kronos_cache[key] = out
            return out

    def _prefetch_yfinance_history(
        self,
        tickers: List[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        interval: str = "1d",
    ) -> None:
        """Prefetch yfinance OHLCV history once per ticker to avoid per-week network calls."""
        try:
            import yfinance as yf
        except Exception as e:
            logger.warning(f"⚠️ yfinance not available for prefetch: {e}")
            return

        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        end_plus = end + timedelta(days=1)

        tickers_norm = sorted({str(t).upper().strip() for t in tickers if str(t).strip()})
        need = [t for t in tickers_norm if t not in self._yf_hist_cache]
        if not need:
            return

        logger.info(f"[Kronos] Prefetch yfinance history for {len(need)} tickers (start={start.date()}, end={end.date()}, interval={interval})...")

        for t in need:
            try:
                raw = yf.download(
                    tickers=t,
                    start=start,
                    end=end_plus,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                    group_by="column",
                    threads=False,
                )
                if raw is None or raw.empty:
                    continue
                col_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
                df = raw.rename(columns=col_map).copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                    df = df.rename(columns=col_map)
                req = ["open", "high", "low", "close", "volume"]
                if any(c not in df.columns for c in req):
                    continue
                df = df[req]
                try:
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                except Exception:
                    df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                # PIT-safe: only forward-fill (past information). Never backfill here.
                df = df.replace([np.inf, -np.inf], np.nan).ffill()
                self._yf_hist_cache[t] = df
            except Exception:
                continue

    def calculate_topn_kronos_returns(
        self,
        predictions: pd.DataFrame,
        top_n: int = 20,
        min_price: float = 10.0,
        apply_kronos: bool = True,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Compute fixed Top-N returns per date, and (optionally) Kronos-filtered Top-N returns."""
        # Prefetch yfinance once for all tickers that ever appear in Top-N across dates (major speedup).
        if apply_kronos:
            try:
                grouped = predictions.dropna(subset=["prediction", "actual"]).groupby("date")
                top_tickers: List[str] = []
                dates: List[pd.Timestamp] = []
                for d, g in grouped:
                    g2 = g.sort_values("prediction", ascending=False)
                    n = min(int(top_n), len(g2))
                    if n <= 0:
                        continue
                    top_tickers.extend(g2.head(n)["ticker"].astype(str).tolist())
                    dates.append(pd.to_datetime(d))
                if dates and top_tickers:
                    start = min(dates) - timedelta(days=400)
                    end = max(dates)
                    self._prefetch_yfinance_history(top_tickers, start=start, end=end, interval="1d")
            except Exception:
                pass

        rows: List[Dict[str, Any]] = []
        for date, date_group in predictions.groupby("date"):
            valid = date_group.dropna(subset=["prediction", "actual"])
            if len(valid) < 5:
                continue
            sorted_group = valid.sort_values("prediction", ascending=False)
            n = min(int(top_n), len(sorted_group))
            top = sorted_group.head(n).copy()
            base_ret = float(top["actual"].mean())

            kronos_ret = np.nan
            n_pass = 0
            if apply_kronos:
                passes = []
                for t in top["ticker"].astype(str).tolist():
                    k = self._kronos_pass_for(ticker=t, as_of_date=pd.to_datetime(date), min_price=min_price)
                    passes.append(bool(k.get("kronos_pass", False)))
                top["kronos_pass"] = passes
                passed = top[top["kronos_pass"] == True]
                n_pass = int(len(passed))
                if n_pass > 0:
                    kronos_ret = float(passed["actual"].mean())

            rows.append(
                {
                    "date": pd.to_datetime(date),
                    "topn": n,
                    "topn_return": base_ret,
                    "kronos_topn_return": kronos_ret,
                    "kronos_n_pass": n_pass,
                }
            )

        if not rows:
            return {}, pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("date")
        summary = {
            "avg_topn_return": float(df["topn_return"].mean()),
            "avg_kronos_topn_return": float(df["kronos_topn_return"].dropna().mean()) if df["kronos_topn_return"].notna().any() else float("nan"),
            "avg_kronos_pass": float(df["kronos_n_pass"].mean()),
            "kronos_pass_rate": float(df["kronos_n_pass"].sum() / df["topn"].sum()) if df["topn"].sum() > 0 else float("nan"),
        }
        return summary, df

    @staticmethod
    def _zscore_cross_section(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Cross-sectional z-score (single date slice)."""
        x = df[cols].copy()
        x = x.replace([np.inf, -np.inf], np.nan)
        mu = x.mean(axis=0, skipna=True)
        sd = x.std(axis=0, skipna=True).replace(0, np.nan)
        return (x - mu) / sd

    def calculate_topn_feature_filtered_returns(
        self,
        predictions: pd.DataFrame,
        full_data: pd.DataFrame,
        top_n: int = 30,
        apply_filter: bool = True,
        refill: bool = True,
        vol_feature: str = "hist_vol_40d",
        vol_z_max: float = 2.0,
        near_high_feature: str = "near_52w_high",
        near_high_z_min: float = -1.5,
        squeeze_feature: str = "bollinger_squeeze",
        squeeze_z_max: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Fixed Top-N mean return per date, and (optionally) feature-filtered Top-N mean return.

        Evidence-based filter (from "high-rank big-loser" analysis):
          - drop names with extremely high volatility *and* far from 52w high.
          - optionally also drop extreme squeeze regimes.

        Filtering is applied to the long picks only and can refill using next-ranked names.
        Z-scores are computed cross-sectionally on the SAME date (leak-free).
        """
        req_cols = [vol_feature, near_high_feature]
        if squeeze_z_max is not None:
            req_cols.append(squeeze_feature)

        rows: List[Dict[str, Any]] = []
        for date, date_group in predictions.groupby("date"):
            valid = date_group.dropna(subset=["prediction", "actual"])
            if len(valid) < 50:
                continue

            # Same-day universe slice for z-scoring
            try:
                day = full_data.xs(pd.to_datetime(date), level="date", drop_level=True)
            except KeyError:
                continue
            if any(c not in day.columns for c in req_cols):
                # Can't apply filter; still compute baseline
                day = None

            sorted_group = valid.sort_values("prediction", ascending=False)
            n = min(int(top_n), len(sorted_group))
            base_top = sorted_group.head(n).copy()
            base_ret = float(base_top["actual"].mean())
            base_tickers = base_top["ticker"].astype(str).str.upper().str.strip().tolist()
            base_set = set(base_tickers)

            filt_ret = np.nan
            kept = np.nan
            dropped = np.nan

            if apply_filter and day is not None:
                z = self._zscore_cross_section(day, req_cols)

                def _passes(t: str) -> bool:
                    t = str(t).upper().strip()
                    if t not in z.index:
                        return False
                    vz = z.at[t, vol_feature]
                    nh = z.at[t, near_high_feature]
                    if pd.notna(vz) and pd.notna(nh):
                        if (float(vz) > float(vol_z_max)) and (float(nh) < float(near_high_z_min)):
                            return False
                    if squeeze_z_max is not None:
                        sz = z.at[t, squeeze_feature]
                        if pd.notna(sz) and float(sz) > float(squeeze_z_max):
                            return False
                    return True

                selected: List[str] = []
                scan = sorted_group["ticker"].astype(str).tolist() if refill else base_top["ticker"].astype(str).tolist()
                for t in scan:
                    if _passes(t):
                        selected.append(str(t).upper().strip())
                        if len(selected) >= n:
                            break

                if selected:
                    # Preserve rank order; take first N selected
                    ranked = sorted_group.copy()
                    ranked["ticker"] = ranked["ticker"].astype(str).str.upper().str.strip()
                    filt_sel = ranked[ranked["ticker"].isin(selected)].head(n)
                    if len(filt_sel) > 0:
                        filt_ret = float(filt_sel["actual"].mean())
                        sel_set = set(filt_sel["ticker"].astype(str).str.upper().str.strip().tolist())
                        kept_from_base = float(len(base_set.intersection(sel_set)))
                        replaced_from_base = float(n - kept_from_base)
                        kept = kept_from_base
                        dropped = replaced_from_base

            rows.append(
                {
                    "date": pd.to_datetime(date),
                    "topn": n,
                    "topn_return": base_ret,
                    "feature_filtered_topn_return": filt_ret,
                    "feature_filtered_kept": kept,
                    "feature_filtered_dropped": dropped,
                }
            )

        if not rows:
            return {}, pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("date")
        summary = {
            "avg_topn_return": float(df["topn_return"].mean()),
            "avg_feature_filtered_topn_return": float(df["feature_filtered_topn_return"].dropna().mean())
            if df["feature_filtered_topn_return"].notna().any()
            else float("nan"),
            "avg_feature_filtered_kept": float(df["feature_filtered_kept"].dropna().mean())
            if df["feature_filtered_kept"].notna().any()
            else float("nan"),
            "avg_feature_filtered_dropped": float(df["feature_filtered_dropped"].dropna().mean())
            if df["feature_filtered_dropped"].notna().any()
            else float("nan"),
        }
        return summary, df

    def generate_report(self, all_results: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        生成性能对比报告

        Args:
            all_results: 所有模型的预测结果

        Returns:
            性能对比表格
        """
        logger.info("=" * 80)
        logger.info("📊 生成性能报告")
        logger.info("=" * 80)

        report_rows = []
        full_data_for_filter = getattr(self, "_full_data_for_analysis", None)

        for model_name, predictions in all_results.items():
            logger.info(f"\n分析 {model_name}...")

            cost_bps = float(getattr(self, "_cost_bps", 0.0) or 0.0)

            # Calculate metrics
            metrics = self.calculate_metrics(predictions)

            # Calculate group returns (fixed Top/Bottom 30 by default)
            group_returns, weekly_details = self.calculate_group_returns(predictions, top_n=30, bottom_n=30, cost_bps=cost_bps)

            # Bucket returns: every 10 ranks from Top 1-150 and Bottom 1-150
            top10_buckets = self._make_rank_buckets(max_rank=150, step=10)
            bottom10_buckets = self._make_rank_buckets(max_rank=150, step=10)
            bucket_summary, bucket_weekly = self.calculate_bucket_returns(
                predictions,
                top_buckets=top10_buckets,
                bottom_buckets=bottom10_buckets,
                cost_bps=cost_bps,
            )

            # Optional: Kronos trade-filter analysis (ONLY meaningful for trading signal; default use ridge_stacking)
            kronos_summary = {}
            kronos_weekly = None
            try:
                if getattr(self, "_apply_kronos_filter", False) and model_name == "ridge_stacking":
                    kronos_summary, kronos_weekly = self.calculate_topn_kronos_returns(
                        predictions,
                        top_n=getattr(self, "_kronos_top_n", 20),
                        min_price=getattr(self, "_kronos_min_price", 10.0),
                        apply_kronos=True,
                    )
            except Exception as e:
                logger.warning(f"⚠️ Kronos trade-filter metrics failed: {e}")
                kronos_summary = {}
                kronos_weekly = None

            # Optional: evidence-based feature filter (ONLY meaningful for trading; default use ridge_stacking)
            featflt_summary = {}
            featflt_weekly = None
            try:
                if (
                    getattr(self, "_apply_feature_filter", False)
                    and model_name == "ridge_stacking"
                    and full_data_for_filter is not None
                ):
                    featflt_summary, featflt_weekly = self.calculate_topn_feature_filtered_returns(
                        predictions=predictions,
                        full_data=full_data_for_filter,
                        top_n=int(getattr(self, "_feature_filter_top_n", 30)),
                        apply_filter=True,
                        refill=bool(getattr(self, "_feature_filter_refill", True)),
                        vol_feature=str(getattr(self, "_feature_filter_vol_feature", "hist_vol_40d")),
                        vol_z_max=float(getattr(self, "_feature_filter_vol_z_max", 2.0)),
                        near_high_feature=str(getattr(self, "_feature_filter_near_high_feature", "near_52w_high")),
                        near_high_z_min=float(getattr(self, "_feature_filter_near_high_z_min", -1.5)),
                        squeeze_feature=str(getattr(self, "_feature_filter_squeeze_feature", "bollinger_squeeze")),
                        squeeze_z_max=getattr(self, "_feature_filter_squeeze_z_max", None),
                    )
            except Exception as e:
                logger.warning(f"⚠️ Feature-filter metrics failed: {e}")
                featflt_summary = {}
                featflt_weekly = None

            # Combine into report row
            row = {
                'Model': model_name,
                'N_Predictions': len(predictions),
                **metrics,
                **group_returns,
                **bucket_summary,
                **({} if not kronos_summary else {
                    "top20_fixed_avg_return": kronos_summary.get("avg_topn_return", np.nan),
                    "top20_kronos_avg_return": kronos_summary.get("avg_kronos_topn_return", np.nan),
                    "top20_kronos_avg_pass": kronos_summary.get("avg_kronos_pass", np.nan),
                    "top20_kronos_pass_rate": kronos_summary.get("kronos_pass_rate", np.nan),
                }),
                **({} if not featflt_summary else {
                    "feature_filter_topn_fixed_avg_return": featflt_summary.get("avg_topn_return", np.nan),
                    "feature_filter_topn_filtered_avg_return": featflt_summary.get("avg_feature_filtered_topn_return", np.nan),
                    "feature_filter_avg_kept": featflt_summary.get("avg_feature_filtered_kept", np.nan),
                    "feature_filter_avg_dropped": featflt_summary.get("avg_feature_filtered_dropped", np.nan),
                }),
            }

            report_rows.append(row)

            # Print summary
            if metrics:
                logger.info(f"  IC: {metrics.get('IC', np.nan):.4f} (p={metrics.get('IC_pvalue', np.nan):.4f})")
                logger.info(f"  Rank IC: {metrics.get('Rank_IC', np.nan):.4f} (p={metrics.get('Rank_IC_pvalue', np.nan):.4f})")
                logger.info(f"  MSE: {metrics.get('MSE', np.nan):.6f}, MAE: {metrics.get('MAE', np.nan):.6f}, R²: {metrics.get('R2', np.nan):.4f}")

            if group_returns:
                logger.info(f"  Top 30 Avg Return: {group_returns.get('avg_top_return', np.nan):.4f}%")
                if 'top_sharpe' in group_returns:
                    logger.info(f"  Top 30 Sharpe (annualized): {group_returns.get('top_sharpe', np.nan):.4f}")
                if 'top_sharpe_net' in group_returns:
                    logger.info(f"  Top 30 Sharpe Net (annualized): {group_returns.get('top_sharpe_net', np.nan):.4f}")
                if 'top_win_rate_net' in group_returns:
                    logger.info(f"  Top 30 Win Rate Net: {group_returns.get('top_win_rate_net', np.nan):.2%}")
                logger.info(f"  Bottom 30 Avg Return: {group_returns.get('avg_bottom_return', np.nan):.4f}%")
                logger.info(f"  Long-Short Return: {group_returns.get('avg_long_short', np.nan):.4f}%")
                logger.info(f"  Long-Short Sharpe: {group_returns.get('long_short_sharpe', np.nan):.4f}")
                logger.info(f"  Win Rate: {group_returns.get('win_rate', np.nan):.2%}")

            # Bucket summaries (print a compact view)
            if bucket_summary:
                # Keep console output short; full bucket set is in the CSV.
                def _g(k: str):
                    return bucket_summary.get(k, np.nan)
                logger.info("  Bucket Avg Returns (sample):")
                logger.info(
                    f"    Top 1-10: {_g('avg_top_1_10_return'):.4f}% | Top 11-20: {_g('avg_top_11_20_return'):.4f}% | "
                    f"... | Top 141-150: {_g('avg_top_141_150_return'):.4f}%"
                )
                logger.info(
                    f"    Bottom 1-10: {_g('avg_bottom_1_10_return'):.4f}% | Bottom 11-20: {_g('avg_bottom_11_20_return'):.4f}% | "
                    f"... | Bottom 141-150: {_g('avg_bottom_141_150_return'):.4f}%"
                )

            if kronos_summary:
                logger.info("  [Kronos@Top20] Trade-filter impact (fixed top-20 tickers):")
                logger.info(f"    Top20 Avg Return (no Kronos): {kronos_summary.get('avg_topn_return', np.nan):.4f}%")
                logger.info(f"    Top20 Avg Return (Kronos pass): {kronos_summary.get('avg_kronos_topn_return', np.nan):.4f}%")
                logger.info(f"    Avg pass count: {kronos_summary.get('avg_kronos_pass', np.nan):.2f} / 20")
                logger.info(f"    Pass rate: {kronos_summary.get('kronos_pass_rate', np.nan):.2%}")

            if featflt_summary:
                nff = int(getattr(self, "_feature_filter_top_n", 30))
                logger.info("  [FeatureFilter] Trade-filter impact (refill to fixed Top-N):")
                logger.info(f"    Top{nff} Avg Return (no filter): {featflt_summary.get('avg_topn_return', np.nan):.4f}%")
                logger.info(f"    Top{nff} Avg Return (filtered):  {featflt_summary.get('avg_feature_filtered_topn_return', np.nan):.4f}%")
                logger.info(f"    Avg kept: {featflt_summary.get('avg_feature_filtered_kept', np.nan):.2f} | Avg dropped: {featflt_summary.get('avg_feature_filtered_dropped', np.nan):.2f}")

        report_df = pd.DataFrame(report_rows)

        logger.info("=" * 80)
        logger.info("📋 最终性能对比表格:")
        logger.info("\n" + report_df.to_string())
        logger.info("=" * 80)

        # Store weekly details for each model
        weekly_details_dict = {}
        for model_name, predictions in all_results.items():
            cost_bps = float(getattr(self, "_cost_bps", 0.0) or 0.0)
            _, weekly_df = self.calculate_group_returns(predictions, top_n=30, bottom_n=30, cost_bps=cost_bps)
            weekly_details_dict[model_name] = weekly_df

            # Add bucket weekly details per model
            try:
                top10_buckets = self._make_rank_buckets(max_rank=150, step=10)
                bottom10_buckets = self._make_rank_buckets(max_rank=150, step=10)
                _, bucket_weekly_df = self.calculate_bucket_returns(
                    predictions,
                    top_buckets=top10_buckets,
                    bottom_buckets=bottom10_buckets,
                    cost_bps=cost_bps,
                )
                weekly_details_dict[f"{model_name}_bucket_returns"] = bucket_weekly_df
            except Exception:
                pass

        # Store Kronos weekly details separately (only for ridge_stacking)
        try:
            if getattr(self, "_apply_kronos_filter", False) and "ridge_stacking" in all_results:
                _, kronos_weekly_df = self.calculate_topn_kronos_returns(
                    all_results["ridge_stacking"],
                    top_n=getattr(self, "_kronos_top_n", 20),
                    min_price=getattr(self, "_kronos_min_price", 10.0),
                    apply_kronos=True,
                )
                weekly_details_dict["ridge_stacking_kronos_top20"] = kronos_weekly_df
        except Exception:
            pass

        # Store feature-filter weekly details separately (only for ridge_stacking)
        try:
            if (
                getattr(self, "_apply_feature_filter", False)
                and "ridge_stacking" in all_results
                and full_data_for_filter is not None
            ):
                _, ff_weekly_df = self.calculate_topn_feature_filtered_returns(
                    predictions=all_results["ridge_stacking"],
                    full_data=full_data_for_filter,
                    top_n=int(getattr(self, "_feature_filter_top_n", 30)),
                    apply_filter=True,
                    refill=bool(getattr(self, "_feature_filter_refill", True)),
                    vol_feature=str(getattr(self, "_feature_filter_vol_feature", "hist_vol_40d")),
                    vol_z_max=float(getattr(self, "_feature_filter_vol_z_max", 2.0)),
                    near_high_feature=str(getattr(self, "_feature_filter_near_high_feature", "near_52w_high")),
                    near_high_z_min=float(getattr(self, "_feature_filter_near_high_z_min", -1.5)),
                    squeeze_feature=str(getattr(self, "_feature_filter_squeeze_feature", "bollinger_squeeze")),
                    squeeze_z_max=getattr(self, "_feature_filter_squeeze_z_max", None),
                )
                weekly_details_dict["ridge_stacking_feature_filter_topn"] = ff_weekly_df
        except Exception:
            pass
        
        
        return report_df, weekly_details_dict

    def run_backtest(
        self,
        max_weeks: Optional[int] = None,
        apply_kronos_filter: bool = False,
        kronos_top_n: int = 30,
        kronos_min_price: float = 10.0,
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        运行完整回测

        Returns:
            (all_results, report_df)
        """
        # Store Kronos config for report generation
        self._apply_kronos_filter = bool(apply_kronos_filter)
        self._kronos_top_n = int(kronos_top_n)
        self._kronos_min_price = float(kronos_min_price)

        # Load data
        data = self.load_factor_data()
        # Cache for any report-time analyses that need the full cross-section features (e.g., feature filter)
        self._full_data_for_analysis = data

        # Rolling prediction
        all_results = self.rolling_prediction(data, max_weeks=max_weeks)

        # Generate report
        report_df, weekly_details_dict = self.generate_report(all_results)

        return all_results, report_df, weekly_details_dict


def main():
    """主函数"""
    import argparse
    logger.info("=" * 80)
    logger.info("🚀 Comprehensive Model Backtest - 完整模型回测分析")
    logger.info("=" * 80)

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=DEFAULT_FACTORS_DIR)
    ap.add_argument("--data-file", type=str, default=DEFAULT_FACTORS_FILE)
    ap.add_argument("--snapshot-id", type=str, default=None)
    ap.add_argument("--max-weeks", type=int, default=None)
    ap.add_argument("--tickers-file", type=str, default=None, help="Restrict backtest universe to tickers in this file (e.g., NASDAQ-only)")
    ap.add_argument("--rebalance-mode", type=str, default="horizon", choices=["horizon", "weekly"],
                    help="How often to rebalance dates for rolling prediction. 'horizon' avoids overlap vs target horizon (default).")
    ap.add_argument("--target-horizon-days", type=int, default=10,
                    help="Target horizon in TRADING DAYS used in factor 'target' (default 10). Used only for rebalance-mode=horizon.")
    ap.add_argument("--start-date", type=str, default=None, help="First date (inclusive, YYYY-MM-DD) to include in evaluation. Defaults to training_end + horizon when snapshot metadata is available.")
    ap.add_argument("--end-date", type=str, default=None, help="Last date (inclusive, YYYY-MM-DD) to include in evaluation.")
    ap.add_argument("--allow-insample", action="store_true", help="Allow evaluating dates that overlap the training window when no out-of-sample data exists (NOT recommended).")
    ap.add_argument("--apply-kronos", action="store_true", help="Apply Kronos trade filter on Top-N picks (ridge_stacking) and report impact")
    ap.add_argument("--kronos-top-n", type=int, default=30)
    ap.add_argument("--kronos-min-price", type=float, default=10.0)
    ap.add_argument("--apply-feature-filter", action="store_true", help="Apply evidence-based feature filter on Top-N long picks (ridge_stacking) and report impact")
    ap.add_argument("--feature-filter-top-n", type=int, default=30)
    ap.add_argument("--feature-filter-refill", action="store_true", help="Refill to keep fixed Top-N after filtering (recommended)")
    ap.add_argument("--feature-filter-vol-z-max", type=float, default=2.0, help="Drop if vol_z > this AND near_high_z < near_high_z_min")
    ap.add_argument("--feature-filter-near-high-z-min", type=float, default=-1.5)
    ap.add_argument("--feature-filter-squeeze-z-max", type=float, default=None, help="Optional: also drop if squeeze_z > this")
    ap.add_argument("--output-dir", type=str, default="result/model_backtest")
    ap.add_argument("--cost-bps", type=float, default=0.0, help="Transaction cost (basis points) applied each rebalance as: turnover * cost_bps/1e4")
    ap.add_argument("--load-catboost", action="store_true", help="Load CatBoost model from snapshot (required for catboost metrics and for ridge_stacking if ridge base_cols include pred_catboost).")
    args = ap.parse_args()

    # Initialize backtest engine
    backtest = ComprehensiveModelBacktest(
        data_dir=args.data_dir,
        snapshot_id=args.snapshot_id,
        data_file=args.data_file,
        tickers_file=args.tickers_file,
        start_date=args.start_date,
        end_date=args.end_date,
        allow_insample_backtest=args.allow_insample,
        load_catboost=bool(args.load_catboost),
    )
    backtest._rebalance_mode = args.rebalance_mode
    backtest._target_horizon_days = int(args.target_horizon_days)
    backtest._cost_bps = float(args.cost_bps or 0.0)
    # NOTE: load_catboost is handled in __init__ so models are loaded correctly.
    # Feature filter config (evidence-based)
    backtest._apply_feature_filter = bool(args.apply_feature_filter)
    backtest._feature_filter_top_n = int(args.feature_filter_top_n)
    backtest._feature_filter_refill = bool(args.feature_filter_refill)
    backtest._feature_filter_vol_feature = "hist_vol_40d"
    backtest._feature_filter_near_high_feature = "near_52w_high"
    backtest._feature_filter_squeeze_feature = "bollinger_squeeze"
    backtest._feature_filter_vol_z_max = float(args.feature_filter_vol_z_max)
    backtest._feature_filter_near_high_z_min = float(args.feature_filter_near_high_z_min)
    backtest._feature_filter_squeeze_z_max = args.feature_filter_squeeze_z_max

    # Run backtest
    all_results, report_df, weekly_details_dict = backtest.run_backtest(
        max_weeks=args.max_weeks,
        apply_kronos_filter=args.apply_kronos,
        kronos_top_n=args.kronos_top_n,
        kronos_min_price=args.kronos_min_price,
    )

    # Save results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save report
    report_path = os.path.join(output_dir, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    report_df.to_csv(report_path, index=False)
    logger.info(f"📄 性能报告已保存: {report_path}")

    # Save weekly details for each model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for model_name, weekly_df in weekly_details_dict.items():
        weekly_path = os.path.join(output_dir, f"{model_name}_weekly_returns_{timestamp}.csv")
        weekly_df.to_csv(weekly_path, index=False)
        logger.info(f"📊 {model_name} 每周收益已保存: {weekly_path}")
    
    # Save detailed predictions
    for model_name, predictions in all_results.items():
        pred_path = os.path.join(output_dir, f"{model_name}_predictions_{timestamp}.parquet")
        predictions.to_parquet(pred_path, index=False)
        logger.info(f"📄 {model_name} 预测结果已保存: {pred_path}")

    logger.info("=" * 80)
    logger.info("✅ 回测完成！")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
