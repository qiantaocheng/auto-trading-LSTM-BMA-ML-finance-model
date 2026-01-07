#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight grid search runner (base models first, ridge separate)
=================================================================

设计目标
- 复用现有 `full_grid_search.py` 中的训练/回测/参数网格逻辑
- 默认只跑四个基模型（elastic_net, xgboost, catboost, lambdarank）
- 如果命令同时包含 ridge 与其他模型，会跳过 ridge 并提示；仅当 --models 里只含 ridge 时才会运行二层 ridge 网格搜索
- 评分逻辑与现有实现一致：回测只取目标模型行的 avg_top_return
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# 确保可以通过 scripts.* 导入同目录下模块
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 复用已有实现，避免复制逻辑
from scripts.full_grid_search import (
    run_single_training,
    run_backtest_for_snapshot,
    get_param_combinations,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def grid_search_single_model(
    model_name: str,
    data_file: str,
    data_dir: str,
    base_config: str,
    snapshot_dir: str,
    output_dir: str,
    max_combos: int | None = None,
    feature_sets: list[list[str] | None] | None = None,
    params_mode: str = "grid",
) -> pd.DataFrame:
    """对单个模型执行完整网格搜索（训练 + 回测），可遍历多组特征子集。"""
    feature_sets = feature_sets or [None]
    all_results = []

    for fset in feature_sets:
        # None -> all features; [] -> compulsory-only; [..] -> compulsory + selected optional
        if fset is None:
            fset_label = "all_features"
        elif len(fset) == 0:
            fset_label = "compulsory_only"
        else:
            fset_label = "|".join(fset)

        if params_mode == "default":
            param_combos = [{}]  # use unified_config defaults (already tuned) and only test feature subsets
        else:
            param_combos = get_param_combinations(model_name)
        if max_combos is not None:
            param_combos = param_combos[:max_combos]
        results = []
        total = len(param_combos)

        logger.info("=" * 80)
        logger.info(f"🔍 Start grid search: {model_name} ({total} combos) | feature_set={fset_label}")
        logger.info("=" * 80)

        for idx, params in enumerate(param_combos, 1):
            logger.info(f"[{idx}/{total}] {model_name} params = {params} | feature_set={fset_label}")

            snapshot_id, train_success = run_single_training(
                model_name=model_name,
                params=params,
                data_file=data_file,
                base_config=base_config,
                snapshot_dir=snapshot_dir,
                feature_list=fset,
            )

            if not train_success or not snapshot_id:
                logger.error(f"❌ Training failed for combo #{idx}")
                results.append(
                    {
                        "model": model_name,
                        "combination_id": idx,
                        "params": params,
                        **params,
                        "feature_set": fset_label,
                        "feature_list": fset if fset else [],
                        "snapshot_id": None,
                        "top20_avg_return": float("nan"),
                        "train_success": False,
                        "backtest_success": False,
                    }
                )
                continue

            top_ret, full_metrics, backtest_success = run_backtest_for_snapshot(
                model_name=model_name,
                snapshot_id=snapshot_id,
                data_dir=data_dir,
                feature_list=fset,
                data_file=data_file,
            )

            results.append(
                {
                    "model": model_name,
                    "combination_id": idx,
                    "params": params,
                    **params,
                    "feature_set": fset_label,
                    "feature_list": fset if fset else [],
                    "snapshot_id": snapshot_id,
                    "top20_avg_return": top_ret,
                    "train_success": train_success,
                    "backtest_success": backtest_success,
                }
            )

            # 写入中间结果（累积当前feature_set）
            inter_path = Path(output_dir) / f"{model_name}_grid_search_intermediate.csv"
            out_df = pd.DataFrame(results)
            # IMPORTANT: `results` already contains *all* rows for the current run,
            # so we should overwrite instead of appending; otherwise rows duplicate
            # and the intermediate file can blow up quadratically.
            out_df.to_csv(inter_path, index=False)
            logger.info(f"💾 Intermediate saved: {inter_path}")

        all_results.extend(results)

    # 最终结果排序并落盘
    final_df = pd.DataFrame(all_results)
    if not final_df.empty:
        final_df = final_df.sort_values("top20_avg_return", ascending=False)
    final_path = Path(output_dir) / f"{model_name}_grid_search_final.csv"
    final_df.to_csv(final_path, index=False)
    logger.info(f"✅ Finished {model_name}, best={final_df['top20_avg_return'].max() if not final_df.empty else 'nan'}")
    logger.info(f"Results saved: {final_path}")
    return final_df


def main():
    parser = argparse.ArgumentParser(description="Lightweight grid search runner")
    parser.add_argument("--data-file", required=True, help="训练数据文件（MultiIndex parquet/csv）")
    parser.add_argument("--data-dir", default="data/factor_exports/factors", help="回测数据目录")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["elastic_net", "xgboost", "catboost", "lambdarank", "ridge"],
        default=["elastic_net", "xgboost", "catboost", "lambdarank"],
        help="默认仅四个基模型；ridge 需单独运行",
    )
    parser.add_argument(
        "--base-config",
        default="bma_models/unified_config.yaml",
        help="基础配置文件路径（可被临时覆盖）",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="cache/grid_search_snapshots",
        help="训练快照输出目录",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=None,
        help="每个模型最多测试多少个超参组合（仅用于smoke test，不传则全量网格）",
    )
    parser.add_argument(
        "--feature-combos",
        type=str,
        default=None,
        help="JSON数组，指定特征子集列表；每个子集为特征名数组。若为空则使用全部特征。",
    )
    parser.add_argument(
        "--feature-combos-file",
        type=str,
        default=None,
        help="特征子集JSON文件路径（内容为JSON数组）。优先级高于 --feature-combos。",
    )
    parser.add_argument(
        "--params-mode",
        choices=["grid", "default"],
        default="grid",
        help="grid=全量超参网格；default=只用unified_config默认参数（用于大规模特征组合测试）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse feature combos
    feature_sets = [None]
    feature_json_text = None
    if args.feature_combos_file:
        try:
            feature_json_text = Path(args.feature_combos_file).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read --feature-combos-file, fallback to all features. Error: {e}")
            feature_json_text = None
    elif args.feature_combos:
        feature_json_text = args.feature_combos

    if feature_json_text:
        try:
            import json

            parsed = json.loads(feature_json_text)
            if isinstance(parsed, list):
                feature_sets = []
                for item in parsed:
                    if item is None:
                        feature_sets.append(None)
                    elif isinstance(item, list):
                        feature_sets.append([str(x) for x in item])
            logger.info(f"[FEATURE] Using feature subsets: {feature_sets}")
        except Exception as e:
            logger.warning(f"Failed to parse --feature-combos, fallback to all features. Error: {e}")
            feature_sets = [None]

    results: Dict[str, pd.DataFrame] = {}

    for model in args.models:
        # 只有在单独运行 ridge 时才执行；否则跳过
        if model == "ridge" and len(args.models) > 1:
            logger.warning("Skip ridge in mixed run. Run `--models ridge` alone after base models are tuned.")
            continue

        df = grid_search_single_model(
            model_name=model,
            data_file=args.data_file,
            data_dir=args.data_dir,
            base_config=args.base_config,
            snapshot_dir=args.snapshot_dir,
            output_dir=str(output_dir),
            max_combos=args.max_combos,
            feature_sets=feature_sets,
            params_mode=args.params_mode,
        )
        results[model] = df

    # 合并输出
    if results:
        combined = pd.concat(results.values(), ignore_index=True)
        combined_path = output_dir / "all_models_grid_search_results.csv"
        combined.to_csv(combined_path, index=False)
        logger.info(f"📦 Combined results saved: {combined_path}")

    logger.info("🏁 Grid search finished.")


if __name__ == "__main__":
    main()

