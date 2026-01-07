#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LambdaRank feature + parameter search (connected to ComprehensiveModelBacktest).

Design (practical + reproducible):
1) Exhaustive feature search over ALL combinations of non-compulsory factors
   (compulsory factors are always included).
   - Uses fixed parameters (params_mode=default) to isolate feature effects.
2) Parameter grid search on the best feature set from (1).
   - Uses full PARAM_GRIDS['lambdarank'] from scripts/full_grid_search.py by default.

Outputs:
- <output_dir>/lambdarank_feature_search.csv
- <output_dir>/lambdarank_param_search.csv
- <output_dir>/best_lambdarank.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Ensure project root is on sys.path when running as a script
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.full_grid_search import get_param_combinations, run_single_training, run_backtest_for_snapshot


COMPULSORY = [
    "obv_divergence",
    "ivol_20",
    "rsi_21",
    "near_52w_high",
    "trend_r2_60",
]

# Default: treat every other alpha factor in the training parquet (excluding Close/target) as optional.
DEFAULT_OPTIONAL = [
    "liquid_momentum",
    "rsrs_beta_18",
    "ret_skew_20d",
    "blowoff_ratio",
    "hist_vol_40d",
    "atr_ratio",
    "bollinger_squeeze",
    "vol_ratio_20d",
    "price_ma60_deviation",
    "making_new_low_5d",
]


def _all_optional_subsets(optional: List[str]) -> List[List[str]]:
    """Return all subsets (including empty) of optional features."""
    out: List[List[str]] = []
    for r in range(len(optional) + 1):
        for combo in itertools.combinations(optional, r):
            out.append(list(combo))
    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_feature_search(
    *,
    data_file: str,
    data_dir: str,
    base_config: str,
    snapshot_dir: str,
    output_dir: str,
    optional_features: List[str],
    max_feature_combos: int | None,
    backtest_max_weeks: int | None,
    backtest_data_file: str | None,
) -> pd.DataFrame:
    """
    Exhaustive feature search for LambdaRank.
    Uses params_mode=default by passing empty params {} to isolate feature impact.
    """
    combos = _all_optional_subsets(optional_features)
    if max_feature_combos is not None:
        combos = combos[:max_feature_combos]

    rows: List[Dict[str, Any]] = []
    for i, opt in enumerate(combos, start=1):
        feature_list = COMPULSORY + opt
        feature_set_name = "compulsory_only" if len(opt) == 0 else f"optional_{len(opt)}"

        snapshot_id, train_ok = run_single_training(
            model_name="lambdarank",
            params={},  # default params from unified_config.yaml
            data_file=data_file,
            base_config=base_config,
            snapshot_dir=snapshot_dir,
            feature_list=feature_list,  # will be applied via BMA_FEATURE_OVERRIDES to lambdarank only
        )

        top20 = None
        bt_ok = False
        if train_ok and snapshot_id:
            top20, _metrics, bt_ok = run_backtest_for_snapshot(
                model_name="lambdarank",
                snapshot_id=snapshot_id,
                data_dir=data_dir,
                feature_list=None,  # snapshot already contains feature_names_by_model
                max_weeks=backtest_max_weeks,
                data_file=backtest_data_file,
            )

        rows.append(
            {
                "stage": "feature",
                "combination_id": i,
                "params": json.dumps({}),
                "feature_set": feature_set_name,
                "feature_list": json.dumps(feature_list),
                "snapshot_id": snapshot_id,
                "top20_avg_return": top20,
                "train_success": bool(train_ok),
                "backtest_success": bool(bt_ok),
            }
        )

        if i % 25 == 0:
            df = pd.DataFrame(rows)
            df.to_csv(Path(output_dir) / "lambdarank_feature_search_intermediate.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / "lambdarank_feature_search.csv", index=False)
    return df


def run_param_search(
    *,
    data_file: str,
    data_dir: str,
    base_config: str,
    snapshot_dir: str,
    output_dir: str,
    feature_list: List[str],
    max_param_combos: int | None,
    backtest_max_weeks: int | None,
    backtest_data_file: str | None,
) -> pd.DataFrame:
    """
    Parameter grid search for LambdaRank on a fixed feature set.
    """
    params_list = get_param_combinations("lambdarank")
    if max_param_combos is not None:
        params_list = params_list[:max_param_combos]

    rows: List[Dict[str, Any]] = []
    for i, params in enumerate(params_list, start=1):
        snapshot_id, train_ok = run_single_training(
            model_name="lambdarank",
            params=params,
            data_file=data_file,
            base_config=base_config,
            snapshot_dir=snapshot_dir,
            feature_list=feature_list,
        )

        top20 = None
        bt_ok = False
        if train_ok and snapshot_id:
            top20, _metrics, bt_ok = run_backtest_for_snapshot(
                model_name="lambdarank",
                snapshot_id=snapshot_id,
                data_dir=data_dir,
                feature_list=None,
                max_weeks=backtest_max_weeks,
                data_file=backtest_data_file,
            )

        rows.append(
            {
                "stage": "params",
                "combination_id": i,
                "params": json.dumps(params),
                "feature_set": "fixed_best",
                "feature_list": json.dumps(feature_list),
                "snapshot_id": snapshot_id,
                "top20_avg_return": top20,
                "train_success": bool(train_ok),
                "backtest_success": bool(bt_ok),
            }
        )

        if i % 50 == 0:
            df = pd.DataFrame(rows)
            df.to_csv(Path(output_dir) / "lambdarank_param_search_intermediate.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / "lambdarank_param_search.csv", index=False)
    return df


def _pick_best(df: pd.DataFrame) -> Dict[str, Any]:
    df_ok = df[df["train_success"].astype(bool) & df["backtest_success"].astype(bool)].copy()
    df_ok = df_ok.dropna(subset=["top20_avg_return"])
    if df_ok.empty:
        return {}
    best = df_ok.sort_values("top20_avg_return", ascending=False).iloc[0].to_dict()
    # decode json fields for convenience
    try:
        best["params_dict"] = json.loads(best.get("params") or "{}")
    except Exception:
        best["params_dict"] = {}
    try:
        best["feature_list_decoded"] = json.loads(best.get("feature_list") or "[]")
    except Exception:
        best["feature_list_decoded"] = []
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--base-config", default="bma_models/unified_config.yaml")
    ap.add_argument("--snapshot-dir", default="cache/grid_search_snapshots")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-feature-combos", type=int, default=None)
    ap.add_argument("--max-param-combos", type=int, default=None)
    ap.add_argument("--backtest-max-weeks", type=int, default=52, help="Speed: limit rolling backtest to last N weeks during tuning (full backtest is expensive). Use 0/None for full.")
    ap.add_argument("--backtest-data-file", type=str, default=None, help="Optional: backtest on a single parquet (much faster than loading full allfac directory). Default: use --data-file.")
    ap.add_argument("--optional-features", type=str, default=None, help="JSON array override for optional features")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    _ensure_dir(out_dir)
    _ensure_dir(Path(args.snapshot_dir))

    optional = DEFAULT_OPTIONAL
    if args.optional_features:
        optional = json.loads(args.optional_features)
        optional = list(map(str, optional))

    backtest_max_weeks = args.backtest_max_weeks if args.backtest_max_weeks and args.backtest_max_weeks > 0 else None
    backtest_data_file = args.backtest_data_file or args.data_file

    # Stage 1: exhaustive feature combos
    feat_df = run_feature_search(
        data_file=args.data_file,
        data_dir=args.data_dir,
        base_config=args.base_config,
        snapshot_dir=args.snapshot_dir,
        output_dir=str(out_dir),
        optional_features=optional,
        max_feature_combos=args.max_feature_combos,
        backtest_max_weeks=backtest_max_weeks,
        backtest_data_file=backtest_data_file,
    )
    best_feat = _pick_best(feat_df)
    if not best_feat:
        raise SystemExit("No successful feature-search runs; cannot proceed to param search.")

    best_feature_list = best_feat["feature_list_decoded"]

    # Stage 2: params on best features
    param_df = run_param_search(
        data_file=args.data_file,
        data_dir=args.data_dir,
        base_config=args.base_config,
        snapshot_dir=args.snapshot_dir,
        output_dir=str(out_dir),
        feature_list=best_feature_list,
        max_param_combos=args.max_param_combos,
        backtest_max_weeks=backtest_max_weeks,
        backtest_data_file=backtest_data_file,
    )
    best_params = _pick_best(param_df)

    best_all = {
        "best_feature_run": best_feat,
        "best_param_run": best_params,
    }
    with open(out_dir / "best_lambdarank.json", "w", encoding="utf-8") as f:
        json.dump(best_all, f, ensure_ascii=False, indent=2)

    print(json.dumps(best_all, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


