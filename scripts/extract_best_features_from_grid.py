#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract best feature combination per model from a grid-search run.

Input:  combo_results.csv produced by scripts/t10_feature_combo_grid_search_parallel_models.py
Output: best_features_per_model.json + best_features_per_model.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--combo-results",
        type=str,
        default="results/t10_feature_combo_grid_parallel_run500_no_overlap/combo_results.csv",
        help="Path to combo_results.csv",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/t10_optimized_all_models",
        help="Directory to write best_features_per_model.{json,csv}",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="avg_top_return",
        help="Metric column used to pick best combo per model (default avg_top_return).",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=["elastic_net", "xgboost", "catboost", "lambdarank"],
        help="Model names to extract (default: elastic_net xgboost catboost lambdarank).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    combo_results_path = Path(args.combo_results)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not combo_results_path.exists():
        raise FileNotFoundError(f"combo_results.csv not found: {combo_results_path}")

    df = pd.read_csv(combo_results_path)
    if df.empty:
        raise ValueError(f"combo_results.csv is empty: {combo_results_path}")

    metric = str(args.metric).strip()
    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' not found in {combo_results_path}. Columns={list(df.columns)}")

    df = df.copy()
    df["model"] = df["model"].astype(str).str.strip().str.lower()
    df = df[df["model"].isin([m.lower() for m in args.models])]
    if "error" in df.columns:
        df = df[df["error"].isna()]

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    if df.empty:
        raise ValueError("No valid rows found after filtering (check models / error column / metric).")

    best = (
        df.sort_values(["model", metric], ascending=[True, False])
        .groupby("model", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    best_records: List[Dict[str, Any]] = []
    best_features: Dict[str, List[str]] = {}
    for _, r in best.iterrows():
        m = str(r["model"]).strip().lower()
        features = json.loads(r["features"]) if isinstance(r.get("features"), str) else []
        features = [str(x) for x in features]
        best_features[m] = features
        best_records.append(
            {
                "model": m,
                "metric": metric,
                "metric_value": float(r[metric]),
                "combo": str(r.get("combo", "")),
                "k": int(r.get("k", len(features))) if pd.notna(r.get("k", None)) else len(features),
                "snapshot_id": str(r.get("snapshot_id", "")),
                "features": json.dumps(features, ensure_ascii=False),
            }
        )

    # Stable artifacts
    json_path = out_dir / "best_features_per_model.json"
    csv_path = out_dir / "best_features_per_model.csv"
    json_path.write_text(json.dumps(best_features, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(best_records).to_csv(csv_path, index=False, encoding="utf-8")

    print("✅ Best feature combo per model")
    print(best[["model", metric, "combo", "k"]].to_string(index=False))
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


