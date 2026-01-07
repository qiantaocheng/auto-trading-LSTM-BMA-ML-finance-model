#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T+10 Feature-Combination Grid Search (All Models)
=================================================

Reality check:
- 13 features => 2^13 = 8192 subsets.
- Training + backtesting every subset for every model is usually too slow.

So this script supports a 2-stage approach:
1) Stage-1 (fast): score ALL subsets using RankIC summary + correlation penalty.
2) Stage-2 (slow): run real train+backtest only for top-N subsets per model.

You can still "run all" in the sense of evaluating all subsets in Stage-1, then
only spending training time on the best candidates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import sys
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.full_grid_search import run_backtest_for_snapshot

logger = logging.getLogger(__name__)


DEFAULT_CANDIDATES_FILE = "results/t10_feature_selection/selected_features.txt"
DEFAULT_IC_SUMMARY = "results/t10_feature_selection/feature_ic_summary.csv"
DEFAULT_CORR = "results/t10_feature_selection/feature_corr.csv"
DEFAULT_DATA_FILE = "data/factor_exports/factors/factors_all.parquet"
DEFAULT_DATA_DIR = "data/factor_exports/factors"

# T+10 compulsory features (should always be present)
COMPULSORY_T10 = ["obv_divergence", "ivol_20", "rsi_21", "near_52w_high", "trend_r2_60"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", type=str, default=DEFAULT_DATA_FILE)
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--candidates-file", type=str, default=DEFAULT_CANDIDATES_FILE)
    p.add_argument("--ic-summary", type=str, default=DEFAULT_IC_SUMMARY)
    p.add_argument("--corr-matrix", type=str, default=DEFAULT_CORR)
    p.add_argument("--stage1-file", type=str, default=None, help="Optional: use an existing stage1_ranked_combos.csv")
    p.add_argument("--models", nargs="+", default=["elastic_net", "xgboost", "catboost", "lambdarank"])
    p.add_argument("--k-min", type=int, default=8, help="Min subset size (includes compulsory).")
    p.add_argument("--k-max", type=int, default=13, help="Max subset size (includes compulsory).")
    p.add_argument("--no-compulsory", action="store_true", help="Do NOT force COMPULSORY_T10 into every subset.")
    p.add_argument("--max-combos", type=int, default=0, help="Cap combos for stage-2 (0 = no cap). Stage-1 can still rank all.")
    p.add_argument("--train-top-per-model", type=int, default=50, help="Stage-2: train+backtest top-N subsets per model.")
    p.add_argument("--stage1-only", action="store_true", help="Only rank all subsets; do not train.")
    p.add_argument("--score-corr-penalty", type=float, default=0.25, help="Penalty weight for correlated pairs.")
    p.add_argument("--corr-threshold", type=float, default=0.90, help="Correlation threshold for penalty.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-weeks", type=int, default=260)
    p.add_argument("--output-dir", type=str, default="results/t10_feature_combo_grid")
    return p.parse_args()


def load_candidates(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")
    feats: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        feats.append(s)
    # de-dupe preserve order
    out: List[str] = []
    seen = set()
    for f in feats:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def enumerate_feature_sets(candidates: List[str], k_min: int, k_max: int, seed: int, max_combos: int) -> List[List[str]]:
    # Ensure compulsory always included
    base = [f for f in COMPULSORY_T10 if f in candidates]
    rest = [f for f in candidates if f not in base]
    if len(base) < len(COMPULSORY_T10):
        missing = [f for f in COMPULSORY_T10 if f not in candidates]
        logger.warning(f"Compulsory features missing from candidates file: {missing} (will proceed)")

    combos: List[List[str]] = []
    for k in range(max(len(base), k_min), min(k_max, len(base) + len(rest)) + 1):
        need = k - len(base)
        for extra in itertools.combinations(rest, need):
            feats = list(dict.fromkeys(base + list(extra)))
            combos.append(feats)

    if not combos:
        raise RuntimeError("No feature combos generated; check k-min/k-max and candidates list.")

    # Stable sort by length then lexicographically (helps reproducibility)
    combos.sort(key=lambda x: (len(x), ",".join(x)))
    return combos


def _load_rankic_weights(ic_summary_csv: str) -> Dict[str, float]:
    df = pd.read_csv(ic_summary_csv)
    # Some runs may have index column; support both formats
    if "feature" in df.columns:
        feat_col = "feature"
    elif df.columns[0] not in ("n_dates", "mean_rank_ic"):
        feat_col = df.columns[0]
    else:
        raise ValueError(f"Cannot infer feature column from {ic_summary_csv}")

    if "abs_mean_rank_ic" in df.columns:
        wcol = "abs_mean_rank_ic"
    elif "mean_rank_ic" in df.columns:
        wcol = "mean_rank_ic"
    else:
        raise ValueError(f"IC summary missing mean_rank_ic columns: {ic_summary_csv}")

    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        f = str(r[feat_col])
        w = float(r[wcol])
        out[f] = abs(w)
    return out


def _load_corr(corr_csv: str) -> pd.DataFrame:
    c = pd.read_csv(corr_csv, index_col=0)
    # ensure square
    return c


def score_subset(
    subset: List[str],
    w: Dict[str, float],
    corr: pd.DataFrame,
    corr_threshold: float,
    corr_penalty: float,
) -> float:
    # base score = sum abs RankIC weights
    base = 0.0
    for f in subset:
        base += float(w.get(f, 0.0))

    # penalty for highly correlated pairs above threshold
    pen = 0.0
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            a, b = subset[i], subset[j]
            if a in corr.index and b in corr.columns:
                v = float(corr.loc[a, b])
                av = abs(v)
                if av >= corr_threshold:
                    pen += (av - corr_threshold)
    return base - corr_penalty * pen


def train_one(model: str, params_json: str, data_file: str, feature_list: List[str], output_dir: Path) -> str:
    """Train a single model via train_single_model.py, returns snapshot id."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_id = output_dir / "snapshot_id.txt"
    cmd = [
        os.environ.get("PYTHON", "python"),
        "scripts/train_single_model.py",
        "--model",
        model,
        "--params",
        params_json,
        "--data-file",
        data_file,
        "--base-config",
        "bma_models/unified_config.yaml",
        "--snapshot-dir",
        "cache/grid_search_snapshots",
        "--output-file",
        str(out_id),
        "--feature-list",
        json.dumps(feature_list),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"train failed for {model}: {proc.stderr[-2000:]}")
    sid = out_id.read_text(encoding="utf-8").strip()
    if not sid:
        raise RuntimeError(f"snapshot id missing for {model}")
    return sid


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    candidates = load_candidates(args.candidates_file)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stage1_path = Path(args.stage1_file) if args.stage1_file else (out_root / "stage1_ranked_combos.csv")

    # Stage-1: rank combos (or load existing)
    if args.stage1_file and stage1_path.exists():
        stage1_df = pd.read_csv(stage1_path)
        scored = [(float(r["score"]), json.loads(r["features"])) for _, r in stage1_df.iterrows()]
        logger.info(f"LOADED {stage1_path} (ranked {len(scored)} combos)")
    else:
        # If user wants truly all subsets, disable compulsory enforcement.
        global COMPULSORY_T10
        if args.no_compulsory:
            COMPULSORY_T10 = []
        all_combos = enumerate_feature_sets(candidates, args.k_min, args.k_max, args.seed, max_combos=0)

        w = _load_rankic_weights(args.ic_summary)
        corr = _load_corr(args.corr_matrix)
        scored = []
        for feats in all_combos:
            s = score_subset(
                feats,
                w=w,
                corr=corr,
                corr_threshold=float(args.corr_threshold),
                corr_penalty=float(args.score_corr_penalty),
            )
            scored.append((s, feats))
        scored.sort(key=lambda x: x[0], reverse=True)

        pd.DataFrame(
            [{"score": sc, "k": len(fs), "features": json.dumps(fs)} for sc, fs in scored]
        ).to_csv(stage1_path, index=False)
        logger.info(f"WROTE {stage1_path} (ranked {len(scored)} combos)")

    if args.stage1_only:
        logger.info("stage1-only: skipping training.")
        return

    # Stage-2: take top-N (optionally cap by max-combos)
    top_n = int(args.train_top_per_model)
    combos = [fs for _, fs in scored[:top_n]]
    if args.max_combos and len(combos) > int(args.max_combos):
        combos = combos[: int(args.max_combos)]

    logger.info(f"Candidates={len(candidates)} | ranked_combos={len(scored)} | stage2_combos={len(combos)} | models={args.models}")

    best_rows: List[Dict[str, object]] = []

    # Default params: keep base config; empty override
    params_json = "{}"

    for model in args.models:
        rows: List[Dict[str, object]] = []
        logger.info(f"=== MODEL {model} ===")

        for i, feat_list in enumerate(combos, start=1):
            run_tag = f"{model}_k{len(feat_list)}_{i:04d}"
            run_dir = out_root / "runs" / run_tag
            try:
                # Training will always include compulsory factors; ensure we explicitly include them for reproducibility.
                if not args.no_compulsory:
                    feat_list = list(dict.fromkeys(list(feat_list) + COMPULSORY_T10))
                sid = train_one(model, params_json, args.data_file, feat_list, run_dir)
                score, full_metrics, ok = run_backtest_for_snapshot(
                    model_name=model,
                    snapshot_id=sid,
                    data_dir=args.data_dir,
                    data_file=args.data_file,
                    feature_list=None,  # training already enforced subset; avoid global whitelist here
                    max_weeks=args.max_weeks,
                )
                row = {
                    "model": model,
                    "run": run_tag,
                    "snapshot_id": sid,
                    "k": len(feat_list),
                    "features": json.dumps(feat_list),
                    "avg_top_return": score,
                    "ok": bool(ok),
                }
                rows.append(row)
                logger.info(f"[{model}] {i}/{len(combos)} k={len(feat_list)} avg_top_return={score:.4f}")
            except Exception as e:
                rows.append(
                    {
                        "model": model,
                        "run": run_tag,
                        "snapshot_id": "",
                        "k": len(feat_list),
                        "features": json.dumps(feat_list),
                        "avg_top_return": float("nan"),
                        "ok": False,
                        "error": str(e),
                    }
                )
                logger.warning(f"[{model}] {i}/{len(combos)} FAILED: {e}")

        df = pd.DataFrame(rows)
        out_csv = out_root / f"results_{model}.csv"
        df.to_csv(out_csv, index=False)
        logger.info(f"WROTE {out_csv}")

        best = df[df["ok"] == True].sort_values("avg_top_return", ascending=False).head(1)
        if len(best):
            best_rows.append(best.iloc[0].to_dict())

    best_df = pd.DataFrame(best_rows).sort_values("avg_top_return", ascending=False)
    best_path = out_root / "best_summary.csv"
    best_df.to_csv(best_path, index=False)
    logger.info(f"WROTE {best_path}")


if __name__ == "__main__":
    main()


