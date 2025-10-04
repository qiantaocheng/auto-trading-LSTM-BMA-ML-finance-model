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
    lambdarank_txt: Optional[str]
    lambdarank_scaler_pkl: Optional[str]
    lambdarank_meta_json: Optional[str]
    ridge_model_pkl: Optional[str]
    ridge_scaler_pkl: Optional[str]
    ridge_meta_json: Optional[str]
    blender_meta_json: Optional[str]
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
        lambdarank_txt=None,
        lambdarank_scaler_pkl=None,
        lambdarank_meta_json=None,
        ridge_model_pkl=None,
        ridge_scaler_pkl=None,
        ridge_meta_json=None,
        blender_meta_json=None,
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

    # RidgeStacker
    try:
        if ridge_stacker is not None and getattr(ridge_stacker, 'ridge_model', None) is not None:
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
            }
            paths.ridge_meta_json = os.path.join(root_dir, 'ridge_meta.json')
            with open(paths.ridge_meta_json, 'w', encoding='utf-8') as f:
                json.dump(ridge_meta, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ RidgeStacker已保存: {paths.ridge_model_pkl}")
    except Exception as e:
        logger.warning(f"导出RidgeStacker失败: {e}")
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

    # Lambda Percentile Transformer meta (optional)
    try:
        if lambda_percentile_transformer is not None and bool(getattr(lambda_percentile_transformer, 'fitted_', False)):
            # Helper to safely extract scalar from potentially numpy value
            def _safe_float(val, default):
                if val is None:
                    return default
                try:
                    if isinstance(val, np.ndarray):
                        return float(val.item()) if val.size == 1 else default
                    return float(val)
                except (ValueError, TypeError):
                    return default

            oof_mean_raw = getattr(lambda_percentile_transformer, 'oof_mean_', None)
            oof_std_raw = getattr(lambda_percentile_transformer, 'oof_std_', None)
            oof_quantiles_raw = getattr(lambda_percentile_transformer, 'oof_quantiles_', None)

            lpt_meta = {
                'method': getattr(lambda_percentile_transformer, 'method', 'quantile'),
                'oof_mean': _safe_float(oof_mean_raw, 0.0),
                'oof_std': _safe_float(oof_std_raw, 1.0),
                'oof_quantiles': [float(x) for x in (oof_quantiles_raw if oof_quantiles_raw is not None else [])],
            }
            paths.lambda_percentile_meta_json = os.path.join(root_dir, 'lambda_percentile_meta.json')
            with open(paths.lambda_percentile_meta_json, 'w', encoding='utf-8') as f:
                json.dump(lpt_meta, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Lambda Percentile Transformer已保存: {paths.lambda_percentile_meta_json}")
    except Exception as e:
        logger.warning(f"导出Lambda Percentile Transformer失败: {e}")
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
        # Ridge stacking coefficients（包含lambda_percentile时也会自然记录）
        if ridge_stacker is not None and getattr(ridge_stacker, 'ridge_model', None) is not None:
            coefs = list(map(float, getattr(ridge_stacker.ridge_model, 'coef_', [])))
            feat_cols = list(getattr(ridge_stacker, 'actual_feature_cols_', []) or getattr(ridge_stacker, 'feature_names_', []) or [])
            if feat_cols and len(feat_cols) == len(coefs):
                weights = dict(zip(feat_cols, coefs))
            else:
                weights = {f'x{i}': coefs[i] for i in range(len(coefs))}
            weights_paths['ridge_stacking'] = _safe_write_json(weights, 'weights_ridge_stacking.json')
    except Exception as e:
        logger.warning(f"导出Ridge权重失败: {e}")

    # Manifest
    manifest = {
        'snapshot_id': snapshot_id,
        'created_at': int(time.time()),
        'tag': tag,
        'feature_names': list(feature_names) if feature_names else None,
        'paths': asdict(paths),
    }

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
                    'lambdarank_txt': paths.lambdarank_txt,
                    'ridge_model_pkl': paths.ridge_model_pkl,
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
    if paths.lambdarank_txt and os.path.exists(paths.lambdarank_txt):
        saved_models.append("LambdaRank")
    if paths.ridge_model_pkl and os.path.exists(paths.ridge_model_pkl):
        saved_models.append("Ridge")

    logger.info("=" * 80)
    logger.info(f"✅ [SNAPSHOT] 模型快照已保存")
    logger.info(f"快照ID: {snapshot_id}")
    logger.info(f"标签: {tag}")
    logger.info(f"保存路径: {root_dir}")
    logger.info(f"成功保存 {len(saved_models)}/5 个模型: {', '.join(saved_models)}")
    if len(saved_models) < 5:
        missing = set(['ElasticNet', 'XGBoost', 'CatBoost', 'LambdaRank', 'Ridge']) - set(saved_models)
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
        result['lambdarank_gain'] = _load_json_if_exists('weights_lambdarank_gain.json')
        result['ridge_stacking'] = _load_json_if_exists('weights_ridge_stacking.json')
        result['lambda_percentile_meta'] = _load_json_if_exists('lambda_percentile_meta.json')

        return result
    except Exception as e:
        logger.error(f"加载快照权重失败: {e}")
        return {}


def load_models_from_snapshot(snapshot_id: Optional[str] = None,
                             sqlite_path: str = os.path.join("data", "model_registry.db")) -> Dict[str, Any]:
    """
    Load all trained models from a snapshot.

    Returns:
    {
        'models': {
            'elastic_net': ElasticNet model,
            'xgboost': XGBoost model,
            'catboost': CatBoost model
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

    # Load CatBoost
    if paths_dict.get('catboost_cbm') and os.path.exists(paths_dict['catboost_cbm']):
        try:
            from catboost import CatBoostRegressor
            cat_model = CatBoostRegressor()
            cat_model.load_model(paths_dict['catboost_cbm'])
            result['models']['catboost'] = cat_model
            logger.info(f"✅ CatBoost loaded from {paths_dict['catboost_cbm']}")
        except Exception as e:
            logger.warning(f"CatBoost加载失败: {e}")

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

            # Load scaler if exists
            if paths_dict.get('lambdarank_scaler_pkl') and os.path.exists(paths_dict['lambdarank_scaler_pkl']):
                ltr.scaler = joblib.load(paths_dict['lambdarank_scaler_pkl'])

            result['lambda_rank_stacker'] = ltr
            logger.info(f"✅ LambdaRankStacker loaded from {paths_dict['lambdarank_txt']}")
        except Exception as e:
            logger.warning(f"LambdaRankStacker加载失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    # Load RidgeStacker
    if paths_dict.get('ridge_model_pkl') and os.path.exists(paths_dict['ridge_model_pkl']):
        try:
            from bma_models.ridge_stacker import RidgeStacker

            # Load metadata
            meta = {}
            if paths_dict.get('ridge_meta_json') and os.path.exists(paths_dict['ridge_meta_json']):
                with open(paths_dict['ridge_meta_json'], 'r') as f:
                    meta = json.load(f)

            # Create RidgeStacker instance
            ridge = RidgeStacker(
                alpha=meta.get('alpha', 1.0),
                fit_intercept=meta.get('fit_intercept', False)
            )

            # Load ridge model
            ridge.ridge_model = joblib.load(paths_dict['ridge_model_pkl'])
            ridge.base_cols = meta.get('base_cols', [])
            ridge.feature_names_ = meta.get('feature_names', [])
            ridge.actual_feature_cols_ = meta.get('actual_feature_cols', [])

            # Load scaler if exists
            if paths_dict.get('ridge_scaler_pkl') and os.path.exists(paths_dict['ridge_scaler_pkl']):
                ridge.scaler = joblib.load(paths_dict['ridge_scaler_pkl'])

            result['ridge_stacker'] = ridge
            logger.info(f"✅ RidgeStacker loaded from {paths_dict['ridge_model_pkl']}")
        except Exception as e:
            logger.warning(f"RidgeStacker加载失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    # Load Lambda Percentile Transformer
    if paths_dict.get('lambda_percentile_meta_json') and os.path.exists(paths_dict['lambda_percentile_meta_json']):
        try:
            from bma_models.lambda_percentile_transformer import LambdaPercentileTransformer

            with open(paths_dict['lambda_percentile_meta_json'], 'r') as f:
                meta = json.load(f)

            lpt = LambdaPercentileTransformer(method=meta.get('method', 'quantile'))
            lpt.oof_mean_ = meta.get('oof_mean', 0.0)
            lpt.oof_std_ = meta.get('oof_std', 1.0)
            lpt.oof_quantiles_ = np.array(meta.get('oof_quantiles', []))
            lpt.fitted_ = True

            result['lambda_percentile_transformer'] = lpt
            logger.info(f"✅ Lambda Percentile Transformer loaded")
        except Exception as e:
            logger.warning(f"Lambda Percentile Transformer加载失败: {e}")

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
        if summary['lambdarank_gain_top']:
            _print_block("LambdaRank gain (top)", summary['lambdarank_gain_top'])
        if summary['ridge_stacking']:
            rs = [(k, float(v)) for k, v in summary['ridge_stacking'].items()]
            rs.sort(key=lambda x: abs(x[1]), reverse=True)
            _print_block("Ridge stacking coefficients", rs)
    except Exception:
        pass

    return summary
