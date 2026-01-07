"""
Apply best grid-search hyperparameters to the production default config (unified_config.yaml).

Why this exists
---------------
The repo's `bma_models/unified_config.yaml` is heavily commented. Using PyYAML to rewrite it
would destroy comments/formatting. This script performs **targeted in-place edits** inside
`training.base_models.<model>` blocks while preserving the rest of the file.

Typical usage
-------------
1) After a grid search:
   python scripts/apply_best_grid_params.py --results-dir results/feature_grid_20251208_full --write

2) Dry run (print what would change):
   python scripts/apply_best_grid_params.py --results-dir results/feature_grid_20251208_full

Notes
-----
- Picks the row with max `top20_avg_return` among rows where train_success/backtest_success are True.
- If `<model>_grid_search_final.csv` is missing, it can optionally fall back to intermediate via --allow-intermediate.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MODEL_FILES = {
    "elastic_net": ("elastic_net_grid_search_final.csv", "elastic_net_grid_search_intermediate.csv"),
    "xgboost": ("xgboost_grid_search_final.csv", "xgboost_grid_search_intermediate.csv"),
    "catboost": ("catboost_grid_search_final.csv", "catboost_grid_search_intermediate.csv"),
    "lambdarank": ("lambdarank_grid_search_final.csv", "lambdarank_grid_search_intermediate.csv"),
}

# Which columns from the grid CSV should be pushed into unified_config.yaml for each model
MODEL_PARAM_COLUMNS = {
    "elastic_net": ("alpha", "l1_ratio"),
    "xgboost": ("n_estimators", "max_depth", "learning_rate", "min_child_weight"),
    "catboost": ("iterations", "depth", "learning_rate", "subsample"),
    # These correspond to UnifiedTrainingConfig._LAMBDA_RANK_CONFIG keys
    "lambdarank": ("num_boost_round", "learning_rate", "num_leaves", "max_depth", "lambda_l2"),
}

# Tie-breakers (prefer smaller = faster), applied only when metric ties within tolerance
MODEL_TIEBREAK_KEYS = {
    "xgboost": ("n_estimators",),
    "catboost": ("iterations",),
    "lambdarank": ("num_boost_round",),
}


def _parse_scalar(s: str) -> Any:
    s = s.strip()
    if s == "":
        return s
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    # int first (avoid treating "50" as float)
    try:
        if re.fullmatch(r"[-+]?\d+", s):
            return int(s)
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        return s


def _yaml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # compact but stable
        return format(v, ".12g")
    # strings: quote only if needed
    s = str(v)
    if re.search(r"[\s:#\[\]\{\},]", s):
        return f"\"{s.replace('\"', '\\\"')}\""
    return s


def _read_best_row(csv_path: Path, metric: str, require_success: bool = True) -> Dict[str, str]:
    best_row: Optional[Dict[str, str]] = None
    best_metric: Optional[float] = None
    tol = 1e-12

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if require_success:
                if row.get("train_success") != "True" or row.get("backtest_success") != "True":
                    continue
            try:
                m = float(row[metric])
            except Exception:
                continue

            if best_metric is None or m > best_metric + tol:
                best_metric = m
                best_row = row
            elif best_metric is not None and abs(m - best_metric) <= tol and best_row is not None:
                # tie-break: prefer smaller compute where possible
                model = row.get("model") or ""
                keys = MODEL_TIEBREAK_KEYS.get(model, ())
                if keys:
                    def _key_tuple(r: Dict[str, str]) -> Tuple:
                        out: List[Any] = []
                        for k in keys:
                            out.append(_parse_scalar(r.get(k, "")))
                        return tuple(out)
                    if _key_tuple(row) < _key_tuple(best_row):
                        best_row = row

    if best_row is None:
        raise RuntimeError(f"No valid rows found in {csv_path} for metric={metric}")
    return best_row


def _pick_csv_path(results_dir: Path, model: str, allow_intermediate: bool) -> Optional[Path]:
    final_name, intermediate_name = MODEL_FILES[model]
    final_path = results_dir / final_name
    if final_path.exists():
        return final_path
    if allow_intermediate:
        inter_path = results_dir / intermediate_name
        if inter_path.exists():
            return inter_path
    return None


def _find_block(lines: List[str], model: str) -> Tuple[int, int]:
    """
    Find the [start, end) line indices for:
      training -> base_models -> <model>:

    Returns:
      (start_idx, end_idx) where start_idx points to the '<model>:' line.
    """
    # We assume indentation for base_models blocks is:
    # training: (0) -> base_models: (2) -> model: (4) -> params: (6)
    base_models_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s{2}base_models:\s*$", line):
            base_models_idx = i
            break
    if base_models_idx is None:
        raise RuntimeError("Could not find 'training: ... base_models:' block in unified_config.yaml")

    # find model start after base_models
    model_pat = re.compile(rf"^\s{{4}}{re.escape(model)}:\s*$")
    start = None
    for j in range(base_models_idx + 1, len(lines)):
        if model_pat.match(lines[j]):
            start = j
            break
        # stop if we exit base_models (indent <= 2, non-empty)
        if re.match(r"^\s{0,2}\S", lines[j]):
            break
    if start is None:
        raise KeyError(f"Model block not found in config: training.base_models.{model}")

    # end at next model (indent 4) or leaving base_models
    end = len(lines)
    for k in range(start + 1, len(lines)):
        if re.match(r"^\s{4}\S", lines[k]):
            end = k
            break
        if re.match(r"^\s{0,2}\S", lines[k]):
            end = k
            break
    return start, end


def _upsert_kv_in_block(lines: List[str], start: int, end: int, updates: Dict[str, Any]) -> List[str]:
    block = lines[start:end]
    key_line_pat = {k: re.compile(rf"^(\s{{6}}{re.escape(k)}:\s*)(.*?)(\s*)(#.*)?$") for k in updates}

    present = {k: False for k in updates}
    new_block: List[str] = []
    for line in block:
        replaced = False
        for k, pat in key_line_pat.items():
            m = pat.match(line)
            if m:
                prefix, _old_val, mid_ws, comment = m.group(1), m.group(2), m.group(3), m.group(4)
                comment = comment or ""
                new_val = _yaml_scalar(updates[k])
                new_block.append(f"{prefix}{new_val}{mid_ws}{comment}".rstrip() + "\n")
                present[k] = True
                replaced = True
                break
        if not replaced:
            new_block.append(line)

    # insert any missing keys after 'enable:' if present, else right after model header
    insert_at = None
    for idx, line in enumerate(new_block):
        if re.match(r"^\s{6}enable:\s*", line):
            insert_at = idx + 1
            break
    if insert_at is None:
        insert_at = 1 if len(new_block) > 1 else len(new_block)

    inserts: List[str] = []
    for k, v in updates.items():
        if not present[k]:
            inserts.append(f"      {k}: {_yaml_scalar(v)}\n")
    if inserts:
        new_block = new_block[:insert_at] + inserts + new_block[insert_at:]

    return lines[:start] + new_block + lines[end:]


def _ensure_model_block_exists(lines: List[str], model: str) -> List[str]:
    """
    If training.base_models.<model> does not exist, append an empty block after the last model block.
    """
    try:
        _find_block(lines, model)
        return lines
    except KeyError:
        pass

    # find base_models start and the next section after it
    base_models_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s{2}base_models:\s*$", line):
            base_models_idx = i
            break
    if base_models_idx is None:
        raise RuntimeError("Could not find base_models in config")

    # insert before leaving base_models (first non-empty line with indent <=2 after base_models)
    insert_pos = None
    for j in range(base_models_idx + 1, len(lines)):
        if re.match(r"^\s{0,2}\S", lines[j]):
            insert_pos = j
            break
    if insert_pos is None:
        insert_pos = len(lines)

    block = [
        "\n",
        f"    {model}:\n",
        "      enable: true\n",
    ]
    return lines[:insert_pos] + block + lines[insert_pos:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, required=True, help="Directory containing *_grid_search_{final,intermediate}.csv files")
    ap.add_argument("--config-file", type=str, default="bma_models/unified_config.yaml", help="Path to unified_config.yaml")
    ap.add_argument("--metric", type=str, default="top20_avg_return", help="Metric column to maximize")
    ap.add_argument("--models", nargs="*", default=["elastic_net", "xgboost", "catboost", "lambdarank"], help="Models to apply")
    ap.add_argument("--allow-intermediate", action="store_true", help="Use *_intermediate.csv when *_final.csv is missing")
    ap.add_argument("--write", action="store_true", help="Actually write changes to config-file (otherwise dry-run)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    cfg_path = Path(args.config_file)

    if not results_dir.exists():
        raise SystemExit(f"results-dir not found: {results_dir}")
    if not cfg_path.exists():
        raise SystemExit(f"config-file not found: {cfg_path}")

    per_model_updates: Dict[str, Dict[str, Any]] = {}
    per_model_metrics: Dict[str, float] = {}
    for model in args.models:
        if model not in MODEL_FILES:
            print(f"[WARN] unknown model '{model}', skipping")
            continue
        csv_path = _pick_csv_path(results_dir, model, allow_intermediate=args.allow_intermediate)
        if csv_path is None:
            print(f"[WARN] no grid CSV found for {model} in {results_dir} (final missing; intermediate allowed={args.allow_intermediate})")
            continue
        best = _read_best_row(csv_path, metric=args.metric, require_success=True)
        updates: Dict[str, Any] = {}
        for col in MODEL_PARAM_COLUMNS.get(model, ()):
            if col in best and best[col] != "":
                updates[col] = _parse_scalar(best[col])
        if not updates:
            print(f"[WARN] no parameter columns found for {model} in {csv_path.name}")
            continue
        per_model_updates[model] = updates
        per_model_metrics[model] = float(best[args.metric])

    if not per_model_updates:
        print("[ERROR] no updates computed; nothing to do")
        return 2

    print("Planned updates (maximize metric: %s):" % args.metric)
    for m in sorted(per_model_updates.keys()):
        print(f"  - {m}: {per_model_updates[m]}  ({args.metric}={per_model_metrics[m]:.12g})")

    if not args.write:
        print("\nDry-run only. Re-run with --write to update the config.")
        return 0

    # read config and apply updates
    text = cfg_path.read_text(encoding="utf-8")
    lines = [ln if ln.endswith("\n") else (ln + "\n") for ln in text.splitlines()]

    # backup
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = cfg_path.with_suffix(cfg_path.suffix + f".bak_{ts}")
    shutil.copy2(cfg_path, backup)
    print(f"\nBackup written: {backup}")

    for model, updates in per_model_updates.items():
        lines = _ensure_model_block_exists(lines, model)
        start, end = _find_block(lines, model)
        lines = _upsert_kv_in_block(lines, start, end, updates)

    cfg_path.write_text("".join(lines), encoding="utf-8")
    print(f"Updated: {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


