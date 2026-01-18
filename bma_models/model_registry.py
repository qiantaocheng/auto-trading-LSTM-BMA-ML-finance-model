#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Registry - Save/Load trained models and parameters for inference without retraining.

Artifacts stored:
- First layer models: ElasticNet (joblib .pkl), XGBoost (.json), CatBoost (.cbm)
- LambdaRankStacker: LightGBM booster (.txt) + StandardScaler (.pkl) + feature columns (.json)
- RidgeStacker: Ridge model (.pkl) + StandardScaler (.pkl) + base_cols (.json)
- Blender/gate metadata (.json)
- Snapshot manifest (.json)

Also records metadata into a SQLite DB at data/model_registry.db (auto-created).
"""

from __future__ import annotations

import os
import json
import uuid
import time
import sqlite3
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _ensure_sqlite_schema(db_path: str) -> None:
    _ensure_dir(os.path.dirname(db_path))
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_snapshots (
                id TEXT PRIMARY KEY,
                created_at INTEGER,
                tag TEXT,
                manifest_path TEXT,
                details_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_params (
                snapshot_id TEXT,
                component TEXT,
                param_key TEXT,
                param_value TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


@dataclass
class SnapshotPaths:
    root_dir: str
    elastic_net_pkl: Optional[str]
    xgb_json: Optional[str]
    catboost_cbm: Optional[str]
    lightgbm_ranker_pkl: Optional[str]
    lambdarank_txt: Optional[str]
    lambdarank_scaler_pkl: Optional[str]
    lambdarank_meta_json: Optional[str]
    ridge_model_pkl: Optional[str]
    ridge_scaler_pkl: Optional[str]
    ridge_meta_json: Optional[str]
    meta_ranker_txt: Optional[str]  # MetaRankerStacker LightGBM model
    meta_ranker_scaler_pkl: Optional[str]  # MetaRankerStacker scaler
    meta_ranker_meta_json: Optional[str]  # MetaRankerStacker metadata
    blender_meta_json: Optional[str]
    dual_head_fusion_pkl: Optional[str]  # 新增：双头融合模型
    dual_head_fusion_meta_json: Optional[str]  # 新增：双头融合元数据
    lambda_percentile_meta_json: Optional[str]
    manifest_json: str


def _save_xgb(model, out_path: str) -> None:
    try:
        # XGBRegressor supports .save_model for JSON
        model.save_model(out_path)
    except Exception as e:
        logger.error(f"保存XGBoost模型失败: {e}")
        raise


def _save_catboost(model, out_path: str) -> None:
    try:
        model.save_model(out_path)
    except Exception as e:
        logger.error(f"保存CatBoost模型失败: {e}")
        raise


def _save_lgb_booster(booster, out_path: str) -> None:
    try:
        booster.save_model(out_path)
    except Exception as e:
        logger.error(f"保存LightGBM Booster失败: {e}")
        raise


def _default_snapshot_dir() -> str:
    return os.path.join("cache", "model_snapshots", time.strftime("%Y%m%d"))


def save_model_snapshot(
    training_results: Dict[str, Any],
    ridge_stacker: Any = None,
    lambda_rank_stacker: Any = None,
    rank_aware_blender: Any = None,
    dual_head_fusion_manager: Any = None,  # 新增：双头融合管理器
    lambda_percentile_transformer: Any = None,
    tag: str = "default",
    snapshot_dir: Optional[str] = None,
    sqlite_path: str = os.path.join("data", "model_registry.db"),
) -> str:
    """
    Save trained models and parameters for future inference without retraining.

    Args:
        training_results: dict containing first layer models and metadata
            Expected layout: {
              'models': {
                 'elastic_net': {'model': ElasticNet},
                 'xgboost': {'model': XGBRegressor},
                 'catboost': {'model': CatBoostRegressor},
                 'lightgbm_ranker': {'model': LGBMRegressor},
                 'lambdarank': {'model': LambdaRankStacker} (optional)
              },
              'feature_names': [...]
            }
        ridge_stacker: trained RidgeStacker instance (optional)
        lambda_rank_stacker: trained LambdaRankStacker instance (optional, if not in training_results)
        rank_aware_blender: RankAwareBlender instance for gate metadata (optional)
        tag: friendly tag for snapshot
        snapshot_dir: directory to store artifacts (default under cache/model_snapshots)
        sqlite_path: sqlite DB path to record manifest

    Returns:
        snapshot_id (str)
    """
    models = training_results.get('models', {}) or training_results.get('traditional_models', {}).get('models', {})
    feature_names = training_results.get('feature_names') or training_results.get('traditional_models', {}).get('feature_names')
    feature_names_by_model = (
        training_results.get('feature_names_by_model')
        or training_results.get('traditional_models', {}).get('feature_names_by_model')
        or {}
    )

    if not models:
        raise ValueError("training_results缺少'models'，无法导出快照")

    snapshot_id = str(uuid.uuid4())
    root_dir = snapshot_dir or _default_snapshot_dir()
    root_dir = os.path.join(root_dir, snapshot_id)
    _ensure_dir(root_dir)

    paths = SnapshotPaths(
        root_dir=root_dir,
        elastic_net_pkl=None,
        xgb_json=None,
        catboost_cbm=None,
        lightgbm_ranker_pkl=None,
        lambdarank_txt=None,
        lambdarank_scaler_pkl=None,
        lambdarank_meta_json=None,
        ridge_model_pkl=None,
        ridge_scaler_pkl=None,
        ridge_meta_json=None,
        meta_ranker_txt=None,
        meta_ranker_scaler_pkl=None,
        meta_ranker_meta_json=None,
        blender_meta_json=None,
        dual_head_fusion_pkl=None,
        dual_head_fusion_meta_json=None,
        lambda_percentile_meta_json=None,
        manifest_json=os.path.join(root_dir, "manifest.json"),
    )

    # First layer models
    # ElasticNet
    try:
        if 'elastic_net' in models and models['elastic_net'].get('model') is not None:
            enet = models['elastic_net']['model']
            paths.elastic_net_pkl = os.path.join(root_dir, 'elastic_net.pkl')
            joblib.dump(enet, paths.elastic_net_pkl)
            logger.info(f"✅ ElasticNet模型已保存: {paths.elastic_net_pkl}")
    except Exception as e:
        logger.warning(f"导出ElasticNet失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # XGBoost
    try:
        if 'xgboost' in models and models['xgboost'].get('model') is not None:
            xgb_model = models['xgboost']['model']
            paths.xgb_json = os.path.join(root_dir, 'xgboost.json')
            _save_xgb(xgb_model, paths.xgb_json)
            logger.info(f"✅ XGBoost模型已保存: {paths.xgb_json}")
    except Exception as e:
        logger.warning(f"导出XGBoost失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # CatBoost
    try:
        if 'catboost' in models and models['catboost'].get('model') is not None:
            cat_model = models['catboost']['model']
            paths.catboost_cbm = os.path.join(root_dir, 'catboost.cbm')
            _save_catboost(cat_model, paths.catboost_cbm)
            logger.info(f"✅ CatBoost模型已保存: {paths.catboost_cbm}")
    except Exception as e:
        logger.warning(f"导出CatBoost失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # LightGBM ranker
    try:
        if 'lightgbm_ranker' in models and models['lightgbm_ranker'].get('model') is not None:
            lgbm_model = models['lightgbm_ranker']['model']
            paths.lightgbm_ranker_pkl = os.path.join(root_dir, 'lightgbm_ranker.pkl')
            joblib.dump(lgbm_model, paths.lightgbm_ranker_pkl)
            logger.info(f"✅ LightGBM ranker已保存: {paths.lightgbm_ranker_pkl}")
    except Exception as e:
        logger.warning(f"导出LightGBM ranker失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # LambdaRankStacker
    try:
        ltr = None
        if lambda_rank_stacker is not None:
            ltr = lambda_rank_stacker
        elif 'lambdarank' in models and models['lambdarank'].get('model') is not None:
            ltr = models['lambdarank']['model']

        if ltr is not None and getattr(ltr, 'model', None) is not None:
            paths.lambdarank_txt = os.path.join(root_dir, 'lambdarank_lgb.txt')
            _save_lgb_booster(ltr.model, paths.lambdarank_txt)
            # scaler
            if getattr(ltr, 'scaler', None) is not None:
                paths.lambdarank_scaler_pkl = os.path.join(root_dir, 'lambdarank_scaler.pkl')
                joblib.dump(ltr.scaler, paths.lambdarank_scaler_pkl)
            # base_cols/meta
            ltr_meta = {
                'base_cols': list(getattr(ltr, 'base_cols', []) or []),
                'n_quantiles': getattr(ltr, 'n_quantiles', None),
                'winsorize_quantiles': getattr(ltr, 'winsorize_quantiles', None),
                'label_gain_power': getattr(ltr, 'label_gain_power', None),
                'cv_n_splits': getattr(ltr, 'cv_n_splits', None),
                'cv_gap_days': getattr(ltr, 'cv_gap_days', None),
                'cv_embargo_days': getattr(ltr, 'cv_embargo_days', None),
                'random_state': getattr(ltr, 'random_state', None),
            }
            paths.lambdarank_meta_json = os.path.join(root_dir, 'lambdarank_meta.json')
            with open(paths.lambdarank_meta_json, 'w', encoding='utf-8') as f:
                json.dump(ltr_meta, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ LambdaRankStacker已保存: {paths.lambdarank_txt}")
    except Exception as e:
        logger.warning(f"导出LambdaRank失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # MetaRankerStacker only (RidgeStacker has been completely replaced)
    try:
        if ridge_stacker is not None:
            # Check if it's MetaRankerStacker (has lightgbm_model)
            is_meta_ranker = hasattr(ridge_stacker, 'lightgbm_model') and getattr(ridge_stacker, 'lightgbm_model', None) is not None
            is_ridge = hasattr(ridge_stacker, 'ridge_model') and getattr(ridge_stacker, 'ridge_model', None) is not None
            
            if is_ridge:
                logger.error("❌ RidgeStacker is no longer supported. Please use MetaRankerStacker instead.")
                raise ValueError("RidgeStacker has been completely replaced by MetaRankerStacker. Please retrain your model.")
            
            if is_meta_ranker:
                # 🔧 Save MetaRankerStacker
                import lightgbm as lgb
                logger.info(f"[SNAPSHOT] 🔧 开始保存MetaRankerStacker...")
                
                # 保存LightGBM模型
                paths.meta_ranker_txt = os.path.join(root_dir, 'meta_ranker.txt')
                try:
                    ridge_stacker.lightgbm_model.save_model(str(paths.meta_ranker_txt))
                    logger.info(f"✅ [SNAPSHOT] LightGBM模型已保存: {paths.meta_ranker_txt}")
                except Exception as e:
                    logger.error(f"❌ [SNAPSHOT] 保存LightGBM模型失败: {e}")
                    raise
                
                # 保存scaler
                if getattr(ridge_stacker, 'scaler', None) is not None:
                    paths.meta_ranker_scaler_pkl = os.path.join(root_dir, 'meta_ranker_scaler.pkl')
                    try:
                        joblib.dump(ridge_stacker.scaler, paths.meta_ranker_scaler_pkl)
                        logger.info(f"✅ [SNAPSHOT] Scaler已保存: {paths.meta_ranker_scaler_pkl}")
                    except Exception as e:
                        logger.warning(f"⚠️  [SNAPSHOT] 保存Scaler失败: {e}")
                else:
                    logger.warning("[SNAPSHOT] ⚠️  MetaRankerStacker没有scaler，跳过保存")
                
                # 🔧 保存完整的元数据（包含所有必要参数）
                meta_ranker_meta = {
                    'base_cols': list(getattr(ridge_stacker, 'base_cols', []) or []),
                    'n_quantiles': getattr(ridge_stacker, 'n_quantiles', 64),
                    'label_gain_power': getattr(ridge_stacker, 'label_gain_power', 2.2),
                    'num_boost_round': getattr(ridge_stacker, 'num_boost_round', 180),
                    'early_stopping_rounds': getattr(ridge_stacker, 'early_stopping_rounds', 60),
                    'lgb_params': getattr(ridge_stacker, 'lgb_params', {}),
                    'actual_feature_cols': list(getattr(ridge_stacker, 'actual_feature_cols_', []) or []),
                    'model_type': 'MetaRankerStacker',
                    'fitted': getattr(ridge_stacker, 'fitted_', False),
                }
                paths.meta_ranker_meta_json = os.path.join(root_dir, 'meta_ranker_meta.json')
                try:
                    with open(paths.meta_ranker_meta_json, 'w', encoding='utf-8') as f:
                        json.dump(meta_ranker_meta, f, ensure_ascii=False, indent=2)
                    logger.info(f"✅ [SNAPSHOT] MetaRankerStacker元数据已保存: {paths.meta_ranker_meta_json}")
                except Exception as e:
                    logger.error(f"❌ [SNAPSHOT] 保存MetaRankerStacker元数据失败: {e}")
                    raise
                
                # Also save as ridge_meta.json for backward compatibility
                ridge_meta = {
                    'base_cols': list(getattr(ridge_stacker, 'base_cols', []) or []),
                    'actual_feature_cols': list(getattr(ridge_stacker, 'actual_feature_cols_', []) or []),
                    'model_type': 'MetaRankerStacker',
                }
                paths.ridge_meta_json = os.path.join(root_dir, 'ridge_meta.json')
                with open(paths.ridge_meta_json, 'w', encoding='utf-8') as f:
                    json.dump(ridge_meta, f, ensure_ascii=False, indent=2)
            elif is_ridge:
                # Save RidgeStacker (backward compatibility)
                paths.ridge_model_pkl = os.path.join(root_dir, 'ridge_model.pkl')
                joblib.dump(ridge_stacker.ridge_model, paths.ridge_model_pkl)
                if getattr(ridge_stacker, 'scaler', None) is not None:
                    paths.ridge_scaler_pkl = os.path.join(root_dir, 'ridge_scaler.pkl')
                    joblib.dump(ridge_stacker.scaler, paths.ridge_scaler_pkl)
                ridge_meta = {
                    'base_cols': list(getattr(ridge_stacker, 'base_cols', []) or []),
                    'alpha': getattr(ridge_stacker, 'best_alpha_', getattr(ridge_stacker, 'alpha', None)),
                    'fit_intercept': getattr(ridge_stacker, 'fit_intercept', False),
                    'solver': getattr(ridge_stacker, 'solver', 'auto'),
                    'tol': getattr(ridge_stacker, 'tol', 1e-6),
                    'feature_names': list(getattr(ridge_stacker, 'feature_names_', []) or []),
                    'actual_feature_cols': list(getattr(ridge_stacker, 'actual_feature_cols_', []) or []),
                    'direction_calibration': bool(getattr(ridge_stacker, 'direction_calibration', False)),
                    'direction_calibration_min_n': int(getattr(ridge_stacker, 'direction_calibration_min_n', 0) or 0),
                    'direction_sign_map': dict(getattr(ridge_stacker, 'direction_sign_map_', {}) or {}),
                    'direction_ic_mean': dict(getattr(ridge_stacker, 'direction_ic_mean_', {}) or {}),
                    'output_sign': float(getattr(ridge_stacker, 'output_sign_', 1.0) or 1.0),
                    'add_rank_features': bool(getattr(ridge_stacker, 'add_rank_features', False)),
                    'model_type': 'RidgeStacker',
                }
                paths.ridge_meta_json = os.path.join(root_dir, 'ridge_meta.json')
                with open(paths.ridge_meta_json, 'w', encoding='utf-8') as f:
                    json.dump(ridge_meta, f, ensure_ascii=False, indent=2)
                logger.info(f"✅ RidgeStacker已保存: {paths.ridge_model_pkl}")
    except Exception as e:
        logger.warning(f"导出Stacker失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # Blender info
    try:
        if rank_aware_blender is not None:
            blender_info = rank_aware_blender.get_blender_info() if hasattr(rank_aware_blender, 'get_blender_info') else {}
            paths.blender_meta_json = os.path.join(root_dir, 'blender_meta.json')
            with open(paths.blender_meta_json, 'w', encoding='utf-8') as f:
                json.dump(blender_info, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"导出Blender参数失败: {e}")

    # Skip saving Lambda Percentile Transformer meta (no longer used)

    # Dual Head Fusion Manager (optional)
    try:
        if dual_head_fusion_manager is not None and getattr(dual_head_fusion_manager, 'fitted_', False):
            # 保存融合模型本身
            if getattr(dual_head_fusion_manager, 'fusion_model', None) is not None:
                paths.dual_head_fusion_pkl = os.path.join(root_dir, 'dual_head_fusion.pkl')
                joblib.dump(dual_head_fusion_manager.fusion_model, paths.dual_head_fusion_pkl)

                # 保存融合模型元数据
                fusion_info = dual_head_fusion_manager.fusion_model.get_model_info()
                fusion_meta = {
                    'enable_fusion': dual_head_fusion_manager.enable_fusion,
                    'alpha': dual_head_fusion_manager.alpha,
                    'beta': dual_head_fusion_manager.beta,
                    'auto_tune': dual_head_fusion_manager.auto_tune,
                    'fitted': dual_head_fusion_manager.fitted_,
                    'fusion_model_info': fusion_info
                }
                paths.dual_head_fusion_meta_json = os.path.join(root_dir, 'dual_head_fusion_meta.json')
                with open(paths.dual_head_fusion_meta_json, 'w', encoding='utf-8') as f:
                    json.dump(fusion_meta, f, ensure_ascii=False, indent=2)

                logger.info(f"✅ Dual Head Fusion Manager已保存: {paths.dual_head_fusion_pkl}")
    except Exception as e:
        logger.warning(f"导出Dual Head Fusion Manager失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # =====================
    # Export model weights
    # =====================
    def _safe_write_json(obj: Dict[str, Any], filename: str) -> Optional[str]:
        try:
            out_path = os.path.join(root_dir, filename)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            return out_path
        except Exception as e:
            logger.warning(f"导出权重失败[{filename}]: {e}")
            return None

    weights_paths: Dict[str, Optional[str]] = {}

    try:
        # ElasticNet coefficients
        if 'elastic_net' in models and models['elastic_net'].get('model') is not None:
            enet = models['elastic_net']['model']
            # 兼容 numpy 标量/数组
            feat_names_raw = feature_names if feature_names is not None else getattr(enet, 'feature_names_in_', None)
            feat_names = list(feat_names_raw) if feat_names_raw is not None else []

            if hasattr(enet, 'coef_'):
                raw_coefs = getattr(enet, 'coef_', None)
                if raw_coefs is not None:
                    try:
                        import numpy as _np
                        coefs = [float(_np.asarray(c).item()) for c in _np.ravel(raw_coefs)]
                    except Exception:
                        coefs = list(map(float, raw_coefs))

                    if feat_names and len(feat_names) == len(coefs):
                        weights = dict(zip(feat_names, coefs))
                    else:
                        weights = {f'f{i}': coefs[i] for i in range(len(coefs))}
                    weights_paths['elastic_net'] = _safe_write_json(weights, 'weights_elastic_net.json')
    except Exception as e:
        logger.warning(f"导出ElasticNet权重失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    try:
        # XGBoost feature importance
        if 'xgboost' in models and models['xgboost'].get('model') is not None:
            xgb_model = models['xgboost']['model']
            booster = getattr(xgb_model, 'get_booster', lambda: None)()
            if booster is not None:
                try:
                    names = getattr(booster, 'feature_names', None)
                except Exception:
                    names = None
                gain = booster.get_score(importance_type='gain') or {}
                weight = booster.get_score(importance_type='weight') or {}
                # Map fN->name if available
                if names and len(names) > 0:
                    mapped = {}
                    for k, v in gain.items():
                        try:
                            idx = int(k[1:]) if k.startswith('f') else None
                            mapped[names[idx] if idx is not None and idx < len(names) else k] = float(v)
                        except Exception:
                            mapped[k] = float(v)
                    gain = mapped
                weights_paths['xgboost_gain'] = _safe_write_json(gain, 'weights_xgboost_gain.json')
                weights_paths['xgboost_weight'] = _safe_write_json(weight, 'weights_xgboost_weight.json')
    except Exception as e:
        logger.warning(f"导出XGBoost权重失败: {e}")

    try:
        # CatBoost feature importance
        if 'catboost' in models and models['catboost'].get('model') is not None:
            cat_model = models['catboost']['model']
            if hasattr(cat_model, 'get_feature_importance'):
                import numpy as _np  # local import
                imps = list(map(float, cat_model.get_feature_importance()))
                fn = feature_names or getattr(cat_model, 'feature_names_', [])
                if fn and len(fn) == len(imps):
                    weights = dict(zip(fn, imps))
                else:
                    weights = {f'f{i}': imps[i] for i in range(len(imps))}
                weights_paths['catboost'] = _safe_write_json(weights, 'weights_catboost.json')
    except Exception as e:
        logger.warning(f"导出CatBoost权重失败: {e}")

    try:
        # LightGBM ranker feature importance
        if 'lightgbm_ranker' in models and models['lightgbm_ranker'].get('model') is not None:
            lgbm_model = models['lightgbm_ranker']['model']
            try:
                import numpy as _np  # local import
                imps = list(map(float, getattr(lgbm_model, 'feature_importances_', [])))
                fn = feature_names or getattr(lgbm_model, 'feature_name_', []) or getattr(lgbm_model, 'feature_names_in_', [])
                if fn and len(fn) == len(imps):
                    weights = dict(zip(fn, imps))
                else:
                    weights = {f'feat_{i}': float(imps[i]) for i in range(len(imps))}
                weights_paths['lightgbm_ranker'] = _safe_write_json(weights, 'weights_lightgbm_ranker.json')
            except Exception as e:
                logger.warning(f"导出LightGBM ranker权重失败: {e}")
    except Exception as e:
        logger.warning(f"导出LightGBM ranker权重失败: {e}")

    try:
        # LambdaRank (LightGBM) feature importance
        ltr = None
        if lambda_rank_stacker is not None:
            ltr = lambda_rank_stacker
        elif 'lambdarank' in models and models['lambdarank'].get('model') is not None:
            ltr = models['lambdarank']['model']
        if ltr is not None and getattr(ltr, 'model', None) is not None:
            booster = ltr.model
            try:
                fn = booster.feature_name() or []
            except Exception:
                fn = []
            try:
                gain = list(map(float, booster.feature_importance(importance_type='gain')))
            except Exception:
                gain = []
            if fn and len(fn) == len(gain):
                weights = dict(zip(fn, gain))
            else:
                weights = {f'f{i}': gain[i] for i in range(len(gain))}
            weights_paths['lambdarank_gain'] = _safe_write_json(weights, 'weights_lambdarank_gain.json')
    except Exception as e:
        logger.warning(f"导出LambdaRank权重失败: {e}")

    try:
        # Ridge stacking coefficients or MetaRankerStacker feature importance
        if ridge_stacker is not None:
            # Check if it's MetaRankerStacker or RidgeStacker
            is_meta_ranker = hasattr(ridge_stacker, 'lightgbm_model') and getattr(ridge_stacker, 'lightgbm_model', None) is not None
            is_ridge = hasattr(ridge_stacker, 'ridge_model') and getattr(ridge_stacker, 'ridge_model', None) is not None
            
            if is_meta_ranker:
                # MetaRankerStacker: Use feature importance from LightGBM
                try:
                    booster = ridge_stacker.lightgbm_model
                    feat_cols = list(getattr(ridge_stacker, 'actual_feature_cols_', []) or getattr(ridge_stacker, 'base_cols', []) or [])
                    try:
                        fn = booster.feature_name() or []
                        gain = list(map(float, booster.feature_importance(importance_type='gain')))
                        if fn and len(fn) == len(gain):
                            weights = dict(zip(fn, gain))
                        elif feat_cols and len(feat_cols) == len(gain):
                            weights = dict(zip(feat_cols, gain))
                        else:
                            weights = {f'f{i}': gain[i] for i in range(len(gain))}
                    except Exception:
                        weights = {col: 1.0 for col in feat_cols} if feat_cols else {}
                    weights_paths['ridge_stacking'] = _safe_write_json(weights, 'weights_ridge_stacking.json')
                except Exception as e:
                    logger.warning(f"导出MetaRankerStacker权重失败: {e}")
            elif is_ridge:
                # RidgeStacker: Use coefficients
                coefs = list(map(float, getattr(ridge_stacker.ridge_model, 'coef_', [])))
                feat_cols = list(getattr(ridge_stacker, 'actual_feature_cols_', []) or getattr(ridge_stacker, 'feature_names_', []) or [])
                if feat_cols and len(feat_cols) == len(coefs):
                    weights = dict(zip(feat_cols, coefs))
                else:
                    weights = {f'x{i}': coefs[i] for i in range(len(coefs))}
                weights_paths['ridge_stacking'] = _safe_write_json(weights, 'weights_ridge_stacking.json')
    except Exception as e:
        logger.warning(f"导出Stacker权重失败: {e}")

    # Manifest
    manifest_metadata: Dict[str, Any] = {}
    training_metadata = None
    if isinstance(training_results, dict):
        training_metadata = training_results.get('training_metadata')
    if training_metadata:
        manifest_metadata['training_date_range'] = {
            'start_date': training_metadata.get('actual_start'),
            'end_date': training_metadata.get('actual_end'),
            'requested_start_date': training_metadata.get('requested_start'),
            'requested_end_date': training_metadata.get('requested_end'),
            'coverage_days': training_metadata.get('coverage_days'),
            'coverage_years': training_metadata.get('coverage_years'),
            'expected_days': training_metadata.get('expected_days'),
            'expected_years': training_metadata.get('expected_years'),
            'start_gap_days': training_metadata.get('start_gap_days'),
            'end_gap_days': training_metadata.get('end_gap_days'),
            'tolerance_days': training_metadata.get('tolerance_days'),
            'uses_full_requested_range': training_metadata.get('uses_full_requested_range'),
        }
        manifest_metadata['sample_count'] = training_metadata.get('sample_count')
        manifest_metadata['feature_count'] = training_metadata.get('feature_count')
        manifest_metadata['ticker_count'] = training_metadata.get('unique_tickers')
        manifest_metadata['unique_dates'] = training_metadata.get('unique_dates')
        manifest_metadata['requested_ticker_count'] = training_metadata.get('requested_ticker_count')
        manifest_metadata['coverage_ratio'] = training_metadata.get('coverage_ratio')
        manifest_metadata['actual_ticker_coverage_ratio'] = training_metadata.get('actual_ticker_coverage_ratio')
        manifest_metadata['requested_span_label'] = training_metadata.get('requested_span_label')
        manifest_metadata['actual_span_label'] = training_metadata.get('actual_span_label')
    manifest = {
        'snapshot_id': snapshot_id,
        'created_at': int(time.time()),
        'tag': tag,
        'feature_names': list(feature_names) if feature_names else None,
        'feature_names_by_model': feature_names_by_model if feature_names_by_model else None,
        'paths': asdict(paths),
    }
    if manifest_metadata:
        manifest['metadata'] = manifest_metadata

    with open(paths.manifest_json, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Record to DB
    _ensure_sqlite_schema(sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO model_snapshots (id, created_at, tag, manifest_path, details_json) VALUES (?, ?, ?, ?, ?)",
            (
                snapshot_id,
                manifest['created_at'],
                tag,
                paths.manifest_json,
                json.dumps({
                    'elastic_net_pkl': paths.elastic_net_pkl,
                    'xgb_json': paths.xgb_json,
                    'catboost_cbm': paths.catboost_cbm,
                    'lightgbm_ranker_pkl': paths.lightgbm_ranker_pkl,
                    'lambdarank_txt': paths.lambdarank_txt,
                    'ridge_model_pkl': paths.ridge_model_pkl,
                    'meta_ranker_txt': paths.meta_ranker_txt,
                    'meta_ranker_scaler_pkl': paths.meta_ranker_scaler_pkl,
                    'meta_ranker_meta_json': paths.meta_ranker_meta_json,
                }, ensure_ascii=False),
            ),
        )
        # Insert params for queryability
        def _insert_params(component: str, meta_path: Optional[str]) -> None:
            if not meta_path or not os.path.isfile(meta_path):
                return
            try:
                with open(meta_path, 'r', encoding='utf-8') as mf:
                    meta = json.load(mf)
                for k, v in meta.items():
                    cur.execute(
                        "INSERT INTO model_params (snapshot_id, component, param_key, param_value) VALUES (?, ?, ?, ?)",
                        (snapshot_id, component, str(k), json.dumps(v, ensure_ascii=False)),
                    )
            except Exception as e:
                logger.warning(f"写入参数失败[{component}]: {e}")

        _insert_params('ridge', paths.ridge_meta_json)
        _insert_params('lambdarank', paths.lambdarank_meta_json)
        _insert_params('blender', paths.blender_meta_json)
        conn.commit()
    finally:
        conn.close()

    # Summary of saved models
    saved_models = []
    if paths.elastic_net_pkl and os.path.exists(paths.elastic_net_pkl):
        saved_models.append("ElasticNet")
    if paths.xgb_json and os.path.exists(paths.xgb_json):
        saved_models.append("XGBoost")
    if paths.catboost_cbm and os.path.exists(paths.catboost_cbm):
        saved_models.append("CatBoost")
    if paths.lightgbm_ranker_pkl and os.path.exists(paths.lightgbm_ranker_pkl):
        saved_models.append("LightGBM Ranker")
    if paths.lambdarank_txt and os.path.exists(paths.lambdarank_txt):
        saved_models.append("LambdaRank")
    if paths.ridge_model_pkl and os.path.exists(paths.ridge_model_pkl):
        saved_models.append("Ridge")

    logger.info("=" * 80)
    logger.info(f"✅ [SNAPSHOT] 模型快照已保存")
    logger.info(f"快照ID: {snapshot_id}")
    logger.info(f"标签: {tag}")
    logger.info(f"保存路径: {root_dir}")
    logger.info(f"成功保存 {len(saved_models)}/6 个模型: {', '.join(saved_models)}")
    if len(saved_models) < 6:
        missing = set(['ElasticNet', 'XGBoost', 'CatBoost', 'LightGBM Ranker', 'LambdaRank', 'Ridge']) - set(saved_models)
        logger.warning(f"⚠️  未保存的模型: {', '.join(missing)}")
    logger.info(f"清单文件: {paths.manifest_json}")
    logger.info("=" * 80)

    return snapshot_id


def load_manifest(snapshot_id: Optional[str] = None, sqlite_path: str = os.path.join("data", "model_registry.db")) -> Dict[str, Any]:
    """Load manifest for a snapshot (latest if snapshot_id is None)."""
    _ensure_sqlite_schema(sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    try:
        cur = conn.cursor()
        if snapshot_id is None:
            cur.execute("SELECT id, manifest_path FROM model_snapshots ORDER BY created_at DESC LIMIT 1")
        else:
            cur.execute("SELECT id, manifest_path FROM model_snapshots WHERE id=? LIMIT 1", (snapshot_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError("未找到可用的模型快照，请先导出快照")
        _, manifest_path = row
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    finally:
        conn.close()


def list_snapshots(sqlite_path: str = os.path.join("data", "model_registry.db")) -> Tuple[Tuple[str, int, str], ...]:
    """Return list of (id, created_at, tag)."""
    _ensure_sqlite_schema(sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, created_at, tag FROM model_snapshots ORDER BY created_at DESC")
        return tuple(cur.fetchall())
    finally:
        conn.close()



# =====================
# Weights Loading & Summary Helpers
# =====================
def load_weights_from_snapshot(snapshot_id: Optional[str] = None,
                               sqlite_path: str = os.path.join("data", "model_registry.db")) -> Dict[str, Any]:
    """
    Load saved model weights/feature importances and stacking coefficients from a snapshot.

    Returns a dict like:
    {
      'paths': {...},
      'elastic_net': {...},
      'xgboost_gain': {...},
      'xgboost_weight': {...},
      'catboost': {...},
      'lightgbm_ranker': {...},
      'lambdarank_gain': {...},
      'ridge_stacking': {...},
      'lambda_percentile_meta': {...}
    }
    """
    try:
        manifest = load_manifest(snapshot_id, sqlite_path=sqlite_path)
        paths = manifest.get('paths', {}) or {}
        root_dir = paths.get('root_dir') or os.path.dirname(paths.get('manifest_json', ''))

        def _load_json_if_exists(filename: str) -> Optional[Dict[str, Any]]:
            try:
                fpath = os.path.join(root_dir, filename)
                if fpath and os.path.isfile(fpath):
                    with open(fpath, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"读取权重失败[{filename}]: {e}")
            return None

        result = {'paths': paths}
        result['elastic_net'] = _load_json_if_exists('weights_elastic_net.json')
        result['xgboost_gain'] = _load_json_if_exists('weights_xgboost_gain.json')
        result['xgboost_weight'] = _load_json_if_exists('weights_xgboost_weight.json')
        result['catboost'] = _load_json_if_exists('weights_catboost.json')
        result['lightgbm_ranker'] = _load_json_if_exists('weights_lightgbm_ranker.json')
        result['lambdarank_gain'] = _load_json_if_exists('weights_lambdarank_gain.json')
        result['ridge_stacking'] = _load_json_if_exists('weights_ridge_stacking.json')
        result['lambda_percentile_meta'] = _load_json_if_exists('lambda_percentile_meta.json')

        return result
    except Exception as e:
        logger.error(f"加载快照权重失败: {e}")
        return {}


def load_models_from_snapshot(
    snapshot_id: Optional[str] = None,
    sqlite_path: str = os.path.join("data", "model_registry.db"),
    *,
    load_catboost: bool = False,
) -> Dict[str, Any]:
    """
    Load all trained models from a snapshot.

    Returns:
    {
        'models': {
            'elastic_net': ElasticNet model,
            'xgboost': XGBoost model,
            'catboost': CatBoost model,
            'lightgbm_ranker': LightGBM ranker
        },
        'ridge_stacker': RidgeStacker instance,
        'lambda_rank_stacker': LambdaRankStacker instance,
        'lambda_percentile_transformer': LambdaPercentileTransformer instance
    }
    """
    manifest = load_manifest(snapshot_id, sqlite_path=sqlite_path)
    paths_dict = manifest.get('paths', {})

    result = {
        'models': {},
        'ridge_stacker': None,
        'lambda_rank_stacker': None,
        'lambda_percentile_transformer': None
    }

    # Load ElasticNet
    if paths_dict.get('elastic_net_pkl') and os.path.exists(paths_dict['elastic_net_pkl']):
        try:
            result['models']['elastic_net'] = joblib.load(paths_dict['elastic_net_pkl'])
            logger.info(f"✅ ElasticNet loaded from {paths_dict['elastic_net_pkl']}")
        except Exception as e:
            logger.warning(f"ElasticNet加载失败: {e}")

    # Load XGBoost
    if paths_dict.get('xgb_json') and os.path.exists(paths_dict['xgb_json']):
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(paths_dict['xgb_json'])
            result['models']['xgboost'] = xgb_model
            logger.info(f"✅ XGBoost loaded from {paths_dict['xgb_json']}")
        except Exception as e:
            logger.warning(f"XGBoost加载失败: {e}")

    # Load CatBoost (optional; disabled by default)
    #
    # Rationale:
    # - CatBoost import can pull IPython/Jupyter widget deps and can be slow / fail in some environments.
    # - Many research/backtest workflows do not require CatBoost.
    if load_catboost and paths_dict.get('catboost_cbm') and os.path.exists(paths_dict['catboost_cbm']):
        try:
            from catboost import CatBoostRegressor  # type: ignore
            cat_model = CatBoostRegressor()
            cat_model.load_model(paths_dict['catboost_cbm'])
            result['models']['catboost'] = cat_model
            logger.info(f"✅ CatBoost loaded from {paths_dict['catboost_cbm']}")
        except Exception as e:
            logger.warning(f"CatBoost加载失败 (skipped): {e}")

    # Load LightGBM ranker
    if paths_dict.get('lightgbm_ranker_pkl') and os.path.exists(paths_dict['lightgbm_ranker_pkl']):
        try:
            result['models']['lightgbm_ranker'] = joblib.load(paths_dict['lightgbm_ranker_pkl'])
            logger.info(f"✅ LightGBM ranker loaded from {paths_dict['lightgbm_ranker_pkl']}")
        except Exception as e:
            logger.warning(f"LightGBM ranker加载失败: {e}")

    # Load LambdaRankStacker
    if paths_dict.get('lambdarank_txt') and os.path.exists(paths_dict['lambdarank_txt']):
        try:
            import lightgbm as lgb
            from bma_models.lambda_rank_stacker import LambdaRankStacker

            # Load metadata
            meta = {}
            if paths_dict.get('lambdarank_meta_json') and os.path.exists(paths_dict['lambdarank_meta_json']):
                with open(paths_dict['lambdarank_meta_json'], 'r') as f:
                    meta = json.load(f)

            # Create LambdaRankStacker instance
            ltr = LambdaRankStacker(
                n_quantiles=meta.get('n_quantiles', 10),
                winsorize_quantiles=meta.get('winsorize_quantiles', (0.01, 0.99)),
                label_gain_power=meta.get('label_gain_power', 2.0)
            )

            # Load booster
            ltr.model = lgb.Booster(model_file=paths_dict['lambdarank_txt'])
            ltr.base_cols = meta.get('base_cols', [])
            
            # Set alpha factor columns (required for prediction)
            ltr._alpha_factor_cols = list(ltr.base_cols) if ltr.base_cols else None

            # Load scaler if exists
            if paths_dict.get('lambdarank_scaler_pkl') and os.path.exists(paths_dict['lambdarank_scaler_pkl']):
                ltr.scaler = joblib.load(paths_dict['lambdarank_scaler_pkl'])

            # Set fitted flag
            ltr.fitted_ = True

            result['lambda_rank_stacker'] = ltr
            logger.info(f"✅ LambdaRankStacker loaded from {paths_dict['lambdarank_txt']}")
        except Exception as e:
            logger.warning(f"LambdaRankStacker加载失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    # Load MetaRankerStacker (priority) or RidgeStacker (fallback)
    # Try MetaRankerStacker first
    if paths_dict.get('meta_ranker_txt') and os.path.exists(paths_dict['meta_ranker_txt']):
        try:
            import lightgbm as lgb
            from bma_models.meta_ranker_stacker import MetaRankerStacker

            # Load metadata
            meta = {}
            if paths_dict.get('meta_ranker_meta_json') and os.path.exists(paths_dict['meta_ranker_meta_json']):
                with open(paths_dict['meta_ranker_meta_json'], 'r') as f:
                    meta = json.load(f)
            elif paths_dict.get('ridge_meta_json') and os.path.exists(paths_dict['ridge_meta_json']):
                # Fallback to ridge_meta.json for backward compatibility
                with open(paths_dict['ridge_meta_json'], 'r') as f:
                    meta = json.load(f)

            # Create MetaRankerStacker instance
            meta_ranker = MetaRankerStacker(
                base_cols=tuple(meta.get('base_cols', [])),
                n_quantiles=meta.get('n_quantiles', 64),
                label_gain_power=meta.get('label_gain_power', 2.2),
                num_boost_round=meta.get('num_boost_round', 180),
                lgb_params=meta.get('lgb_params', {}),
                use_purged_cv=True,
                use_internal_cv=True,
                random_state=42
            )

            # Load LightGBM model
            meta_ranker.lightgbm_model = lgb.Booster(model_file=paths_dict['meta_ranker_txt'])
            meta_ranker.base_cols = tuple(meta.get('base_cols', []))
            meta_ranker.actual_feature_cols_ = meta.get('actual_feature_cols', [])

            # Load scaler if exists
            if paths_dict.get('meta_ranker_scaler_pkl') and os.path.exists(paths_dict['meta_ranker_scaler_pkl']):
                meta_ranker.scaler = joblib.load(paths_dict['meta_ranker_scaler_pkl'])

            # Set fitted flag
            meta_ranker.fitted_ = True

            result['ridge_stacker'] = meta_ranker  # Use same key for compatibility (backward compatibility)
            logger.info(f"✅ MetaRankerStacker loaded from {paths_dict['meta_ranker_txt']}")
        except Exception as e:
            logger.error(f"❌ MetaRankerStacker加载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Cannot load MetaRankerStacker from snapshot. This snapshot may be corrupted or incomplete. Error: {e}")
    
    # RidgeStacker has been completely replaced by MetaRankerStacker
    # No fallback to RidgeStacker - if MetaRankerStacker is not found, raise an error
    elif paths_dict.get('ridge_model_pkl') and os.path.exists(paths_dict['ridge_model_pkl']):
        logger.error("❌ Found old RidgeStacker snapshot but RidgeStacker has been replaced by MetaRankerStacker.")
        logger.error("   Please retrain the model to generate a new snapshot with MetaRankerStacker.")
        raise RuntimeError("RidgeStacker is no longer supported. Please retrain the model to use MetaRankerStacker.")

    # Skip loading Lambda Percentile Transformer (no longer used)

    return result


def summarize_weights(snapshot_id: Optional[str] = None,
                      top_n: int = 10,
                      sqlite_path: str = os.path.join("data", "model_registry.db")) -> Dict[str, Any]:
    """
    Summarize and return top-N weights/importances and stacking coefficients for quick inspection.
    """
    weights = load_weights_from_snapshot(snapshot_id, sqlite_path=sqlite_path)

    def _topn(d: Optional[Dict[str, Any]], n: int) -> list:
        if not d:
            return []
        try:
            items = [(k, float(v)) for k, v in d.items()]
            items.sort(key=lambda x: abs(x[1]), reverse=True)
            return items[:n]
        except Exception:
            return []

    summary = {
        'elastic_net_top': _topn(weights.get('elastic_net'), top_n),
        'xgboost_gain_top': _topn(weights.get('xgboost_gain'), top_n),
        'catboost_top': _topn(weights.get('catboost'), top_n),
        'lightgbm_ranker_top': _topn(weights.get('lightgbm_ranker'), top_n),
        'lambdarank_gain_top': _topn(weights.get('lambdarank_gain'), top_n),
        'ridge_stacking': weights.get('ridge_stacking') or {},
        'lambda_percentile_meta': weights.get('lambda_percentile_meta') or {}
    }

    try:
        logger.info("=== Snapshot Weights Summary ===")
        def _print_block(title: str, pairs: list):
            logger.info(title)
            for k, v in pairs:
                logger.info(f"  {k}: {v:.6f}")

        if summary['elastic_net_top']:
            _print_block("ElasticNet (top)", summary['elastic_net_top'])
        if summary['xgboost_gain_top']:
            _print_block("XGBoost gain (top)", summary['xgboost_gain_top'])
        if summary['catboost_top']:
            _print_block("CatBoost (top)", summary['catboost_top'])
        if summary['lightgbm_ranker_top']:
            _print_block("LightGBM ranker (top)", summary['lightgbm_ranker_top'])
        if summary['lambdarank_gain_top']:
            _print_block("LambdaRank gain (top)", summary['lambdarank_gain_top'])
        if summary['ridge_stacking']:
            rs = [(k, float(v)) for k, v in summary['ridge_stacking'].items()]
            rs.sort(key=lambda x: abs(x[1]), reverse=True)
            _print_block("Ridge stacking coefficients", rs)
    except Exception:
        pass

    return summary
