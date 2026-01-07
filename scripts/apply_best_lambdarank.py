#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply best LambdaRank params + feature subset (optional-only) to production defaults.

Inputs:
- results/<run_dir>/best_lambdarank.json (from scripts/lambdarank_feature_param_search.py)

Outputs:
- Updates bma_models/unified_config.yaml (in-place, preserves most formatting)
- Updates bma_models/量化模型_bma_ultra_enhanced.py default first_layer_feature_overrides['lambdarank']
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


COMPULSORY = [
    "obv_divergence",
    "ivol_20",
    "rsi_21",
    "near_52w_high",
    "trend_r2_60",
]


def _yaml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return format(v, ".12g")
    s = str(v)
    if re.search(r"[\s:#\[\]\{\},]", s):
        return f"\"{s.replace('\"', '\\\"')}\""
    return s


def _find_block(lines: List[str], model: str) -> Tuple[int, int]:
    """
    Find the [start, end) line indices for:
      training -> base_models -> <model>:
    """
    base_models_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s{2}base_models:\s*$", line):
            base_models_idx = i
            break
    if base_models_idx is None:
        raise RuntimeError("Could not find 'training: ... base_models:' block in unified_config.yaml")

    model_pat = re.compile(rf"^\s{{4}}{re.escape(model)}:\s*$")
    start = None
    for j in range(base_models_idx + 1, len(lines)):
        if model_pat.match(lines[j]):
            start = j
            break
        if re.match(r"^\s{0,2}\S", lines[j]):
            break
    if start is None:
        raise KeyError(f"Model block not found in config: training.base_models.{model}")

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

    # Insert missing keys after enable: if present, else after model header
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


def _update_model_default_overrides(model_py: Path, optional_only: List[str], write: bool) -> None:
    text = model_py.read_text(encoding="utf-8")
    # Replace the 'lambdarank': None entry inside first_layer_feature_overrides dict.
    pat = re.compile(r"(\n\s*'lambdarank'\s*:\s*)(None|\[[^\]]*\])(\s*,\s*\n)")
    repl = r"\1" + json.dumps(optional_only, ensure_ascii=False) + r"\3"
    if not pat.search(text):
        raise RuntimeError("Could not find first_layer_feature_overrides['lambdarank'] entry to update")
    new_text = pat.sub(repl, text, count=1)
    if write:
        model_py.write_text(new_text, encoding="utf-8")
    else:
        print("[DRY RUN] Would update `first_layer_feature_overrides['lambdarank']` in", str(model_py))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-json", required=True, help="Path to best_lambdarank.json")
    ap.add_argument("--config", default="bma_models/unified_config.yaml")
    ap.add_argument("--model-py", default="bma_models/量化模型_bma_ultra_enhanced.py")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    best_path = Path(args.best_json)
    payload = json.loads(best_path.read_text(encoding="utf-8"))
    best_param_run = payload.get("best_param_run") or {}
    params = best_param_run.get("params_dict") or {}
    feature_list = best_param_run.get("feature_list_decoded") or []

    if not params or not feature_list:
        raise SystemExit("best_param_run missing params_dict or feature_list_decoded; cannot apply.")

    optional_only = [f for f in feature_list if f not in set(COMPULSORY)]

    # Update unified_config.yaml block
    config_path = Path(args.config)
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    start, end = _find_block(lines, "lambdarank")

    updates = {
        "num_boost_round": params["num_boost_round"],
        "learning_rate": params["learning_rate"],
        "num_leaves": params["num_leaves"],
        "max_depth": params["max_depth"],
        "lambda_l2": params["lambda_l2"],
    }

    new_lines = _upsert_kv_in_block(lines, start, end, updates)
    if args.write:
        config_path.write_text("".join(new_lines), encoding="utf-8")
        print("Updated", str(config_path))
    else:
        print("[DRY RUN] Would update", str(config_path), "with:", updates)

    # Update model default overrides
    _update_model_default_overrides(Path(args.model_py), optional_only=optional_only, write=args.write)
    if args.write:
        print("Updated", str(args.model_py))
        print("LambdaRank optional feature override:", optional_only)


if __name__ == "__main__":
    main()


