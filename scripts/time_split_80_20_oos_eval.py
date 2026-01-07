#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
80/20 time-split train/test on MultiIndex(date,ticker) factors.

Design:
  - Split by unique dates (sorted)
  - Train on first 80% dates, BUT purge a gap = horizon_days to avoid label leakage
    (because target at date t uses forward returns through t+horizon_days)
  - Evaluate on last 20% dates using ComprehensiveModelBacktest with start/end window
  - Report Top-20 expected return for ridge_stacking on the test window
  - Produce per-period and cumulative plots vs NASDAQ proxy (QQQ via yfinance fallback)

Outputs (under output-dir/run_<ts>/):
  - snapshot_id.txt
  - report_df.csv
  - ridge_top20_timeseries.csv
  - top20_vs_qqq.png
  - top20_vs_qqq_cumulative.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", type=str, default="data/factor_exports/factors", help="Parquet shards dir or a single parquet file.")
    p.add_argument("--data-dir", type=str, default="data/factor_exports/factors")
    p.add_argument("--data-file", type=str, default="data/factor_exports/factors/factors_all.parquet")
    p.add_argument("--horizon-days", type=int, default=10)
    p.add_argument("--split", type=float, default=0.8, help="Train split fraction by time (default 0.8).")
    p.add_argument("--model", type=str, default="ridge_stacking", help="Primary model for legacy single-model TopN plot exports.")
    p.add_argument("--models", nargs="+", default=None, help="If provided, export TopN OOS plots/metrics for these models (e.g. elastic_net xgboost catboost lambdarank ridge_stacking). If omitted, uses --model only.")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--rebalance-mode", type=str, default="horizon", choices=["horizon", "weekly"])
    p.add_argument("--max-weeks", type=int, default=260)
    p.add_argument("--cost-bps", type=float, default=0.0, help="Transaction cost (bps) applied each rebalance as: turnover * cost_bps/1e4.")
    p.add_argument("--benchmark", type=str, default="QQQ")
    p.add_argument("--ridge-base-cols", nargs="+", default=None, help="Override RidgeStacker base_cols for this run (e.g. pred_catboost pred_elastic pred_xgb pred_lambdarank).")
    p.add_argument("--output-dir", type=str, default="results/t10_time_split_80_20")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _compute_benchmark_tplus_from_yfinance(bench: str, rebalance_dates: pd.Series, horizon_days: int, logger: logging.Logger) -> pd.Series:
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        logger.warning("yfinance not available: %s", e)
        return pd.Series(dtype=float)

    dates = pd.to_datetime(rebalance_dates).dropna().sort_values()
    if len(dates) == 0:
        return pd.Series(dtype=float)

    start = (dates.min() - pd.Timedelta(days=30)).date().isoformat()
    end = (dates.max() + pd.Timedelta(days=30)).date().isoformat()
    logger.info("Fetching benchmark %s via yfinance (%s -> %s)...", bench, start, end)
    px = yf.download(
        tickers=bench,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if px is None or px.empty:
        return pd.Series(dtype=float)

    close = px["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    trading_days = close.index

    def _ret(d: pd.Timestamp) -> float:
        base_candidates = trading_days[trading_days <= d]
        if len(base_candidates) == 0:
            return float("nan")
        base = pd.Timestamp(base_candidates[-1])
        base_pos = int(trading_days.get_indexer([base])[0])
        tgt_pos = base_pos + int(horizon_days)
        if tgt_pos >= len(trading_days):
            return float("nan")
        tgt = pd.Timestamp(trading_days[tgt_pos])
        b = float(close.loc[base])
        t = float(close.loc[tgt])
        return (t - b) / b if b else float("nan")

    out = pd.Series({_d: _ret(pd.Timestamp(_d)) for _d in dates})
    out.index = pd.to_datetime(out.index)
    out.name = "benchmark_return"
    return out


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _write_model_topn_vs_benchmark(
    *,
    run_dir: Path,
    bt,
    model_name: str,
    preds: pd.DataFrame,
    top_n: int,
    horizon: int,
    bench: str,
    bench_ret: pd.Series,
    cost_bps: float,
    logger: logging.Logger,
) -> dict:
    """Write per-model TopN time series + plots and return a small summary dict (percent units)."""
    group_summary, group_ts = bt.calculate_group_returns(preds, top_n=top_n, bottom_n=top_n, cost_bps=cost_bps)
    if group_ts.empty:
        raise RuntimeError(f"Group return time series is empty on test window for model={model_name}")

    cols = ["date", "top_return"]
    if "top_return_net" in group_ts.columns:
        cols.append("top_return_net")
    if "top_turnover" in group_ts.columns:
        cols.append("top_turnover")
    if "top_cost" in group_ts.columns:
        cols.append("top_cost")

    out = group_ts[cols].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.sort_values("date")
    out["benchmark_return"] = out["date"].map(lambda d: float(bench_ret.get(d, float("nan"))) if hasattr(bench_ret, "get") else float("nan"))

    # Convert to percent
    out["top_return"] = pd.to_numeric(out["top_return"], errors="coerce") * 100.0
    if "top_return_net" in out.columns:
        out["top_return_net"] = pd.to_numeric(out["top_return_net"], errors="coerce") * 100.0
    out["benchmark_return"] = pd.to_numeric(out["benchmark_return"], errors="coerce") * 100.0

    # Cumulative (compounded)
    def _cum_pct(s_pct: pd.Series) -> pd.Series:
        r = pd.to_numeric(s_pct, errors="coerce").fillna(0.0) / 100.0
        return (1.0 + r).cumprod() - 1.0

    out["cum_top_return"] = _cum_pct(out["top_return"]) * 100.0
    if "top_return_net" in out.columns:
        out["cum_top_return_net"] = _cum_pct(out["top_return_net"]) * 100.0
    out["cum_benchmark_return"] = _cum_pct(out["benchmark_return"]) * 100.0

    # Save time series
    ts_path = run_dir / f"{model_name}_top{top_n}_timeseries.csv"
    out.to_csv(ts_path, index=False, encoding="utf-8")

    # Plots (period)
    plt.figure(figsize=(14, 7))
    plt.plot(out["date"], out["top_return"], label=f"{model_name} Top{top_n} (gross, period)", linewidth=1.6)
    if "top_return_net" in out.columns and cost_bps > 0:
        plt.plot(out["date"], out["top_return_net"], label=f"{model_name} Top{top_n} (net {cost_bps:g}bp, period)", linewidth=1.9)
    plt.plot(out["date"], out["benchmark_return"], label=f"{bench} (period)", linewidth=2.0, linestyle="--", color="black")
    plt.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
    plt.title(f"OOS period returns (Top{top_n}, T+{horizon}) vs {bench} — model={model_name} — cost={cost_bps:g}bp")
    plt.xlabel("Rebalance date (test window)")
    plt.ylabel("Return over next horizon (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / f"{model_name}_top{top_n}_vs_{bench.lower()}.png", dpi=160)
    plt.close()

    # Plots (cumulative)
    plt.figure(figsize=(14, 7))
    plt.plot(out["date"], out["cum_top_return"], label=f"{model_name} Top{top_n} (gross, cum)", linewidth=1.6)
    if "cum_top_return_net" in out.columns and cost_bps > 0:
        plt.plot(out["date"], out["cum_top_return_net"], label=f"{model_name} Top{top_n} (net {cost_bps:g}bp, cum)", linewidth=2.0)
    plt.plot(out["date"], out["cum_benchmark_return"], label=f"{bench} (cum)", linewidth=2.2, linestyle="--", color="black")
    plt.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
    plt.title(f"OOS cumulative return (Top{top_n}, T+{horizon}) vs {bench} — model={model_name} — cost={cost_bps:g}bp")
    plt.xlabel("Rebalance date (test window)")
    plt.ylabel("Cumulative return (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / f"{model_name}_top{top_n}_vs_{bench.lower()}_cumulative.png", dpi=160)
    plt.close()

    logger.info("[%s] OOS Top%d avg return gross (%%): %.6f", model_name, top_n, float(group_summary.get("avg_top_return", float("nan"))) * 100.0)
    if cost_bps > 0 and "avg_top_return_net" in group_summary:
        logger.info("[%s] OOS Top%d avg return net (%%): %.6f", model_name, top_n, float(group_summary.get("avg_top_return_net", float("nan"))) * 100.0)

    return {
        "model": model_name,
        "top_n": top_n,
        "avg_top_return_pct": _safe_float(pd.to_numeric(out["top_return"], errors="coerce").mean()),
        "avg_top_return_net_pct": _safe_float(pd.to_numeric(out["top_return_net"], errors="coerce").mean()) if "top_return_net" in out.columns else float("nan"),
        "avg_benchmark_return_pct": _safe_float(pd.to_numeric(out["benchmark_return"], errors="coerce").mean()),
        "end_cum_top_return_pct": _safe_float(out["cum_top_return"].iloc[-1]) if len(out) else float("nan"),
        "end_cum_top_return_net_pct": _safe_float(out["cum_top_return_net"].iloc[-1]) if "cum_top_return_net" in out.columns and len(out) else float("nan"),
        "end_cum_benchmark_return_pct": _safe_float(out["cum_benchmark_return"].iloc[-1]) if len(out) else float("nan"),
        "timeseries_csv": str(ts_path).replace("\\", "/"),
    }


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("time_split_80_20")

    # Make imports work on Windows
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scripts"))

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    horizon = int(args.horizon_days)
    split = float(args.split)
    if not (0.5 < split < 0.95):
        raise ValueError("--split must be in (0.5, 0.95) for a meaningful train/test split")

    from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
    from scripts.comprehensive_model_backtest import ComprehensiveModelBacktest

    # Load data once to compute date split
    logger.info("Loading data to compute 80/20 time split...")
    tmp_bt = ComprehensiveModelBacktest(
        data_dir=str(Path(args.data_dir)),
        snapshot_id=None,
        data_file=str(Path(args.data_file)),
    )
    tmp_bt._rebalance_mode = args.rebalance_mode
    tmp_bt._target_horizon_days = horizon
    df = tmp_bt.load_factor_data()
    if not isinstance(df.index, pd.MultiIndex) or "date" not in df.index.names:
        raise RuntimeError("Expected MultiIndex with 'date' level in factors dataset.")

    dates = pd.Index(pd.to_datetime(df.index.get_level_values("date")).tz_localize(None).unique()).sort_values()
    n_dates = len(dates)
    if n_dates < 200:
        logger.warning("Only %d unique dates detected; 80/20 split may be noisy.", n_dates)

    split_idx = int(n_dates * split)
    # Purge leakage gap = horizon days (labels use forward returns)
    train_end_idx = max(0, split_idx - 1 - horizon)
    if train_end_idx <= 0:
        raise RuntimeError("Not enough dates to apply purge gap; reduce horizon or use more history.")

    train_start = dates[0]
    train_end = dates[train_end_idx]
    test_start = dates[split_idx]
    test_end = dates[-1]

    logger.info("Time split (purged): train=%s..%s, test=%s..%s (dates=%d, split=%.2f, gap=%d)",
                train_start.date(), train_end.date(), test_start.date(), test_end.date(), n_dates, split, horizon)
    logger.info("Costs: cost_bps=%.4f (applied on test backtest only)", float(args.cost_bps or 0.0))

    # Train snapshot on train window only
    # Optional: override RidgeStacker base_cols for this run by writing a temp unified_config
    # and setting BMA_TEMP_CONFIG_PATH (respected by UnifiedTrainingConfig).
    import os
    temp_cfg_path = None
    try:
        if args.ridge_base_cols:
            try:
                import yaml  # type: ignore

                base_cfg = Path("bma_models/unified_config.yaml")
                cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8")) or {}
                training_cfg = cfg.setdefault("training", {})
                ridge_cfg = training_cfg.setdefault("ridge_stacker", {})
                ridge_cfg["base_cols"] = [str(x) for x in args.ridge_base_cols]

                temp_cfg_path = run_dir / "unified_config_override.yaml"
                temp_cfg_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
                os.environ["BMA_TEMP_CONFIG_PATH"] = str(temp_cfg_path)
                logger.info("🔧 Overriding RidgeStacker base_cols: %s", ridge_cfg["base_cols"])
                logger.info("🔧 Using temp config: %s", temp_cfg_path)
            except Exception as e:
                logger.warning("Failed to apply --ridge-base-cols override; proceeding with default config. err=%s", e)

        model = UltraEnhancedQuantitativeModel()
        train_res = model.train_from_document(
            training_data_path=str(Path(args.train_data)),
            top_n=50,
            start_date=str(train_start.date()),
            end_date=str(train_end.date()),
        )
    finally:
        os.environ.pop("BMA_TEMP_CONFIG_PATH", None)
    if not train_res.get("success", False):
        raise RuntimeError(f"Training failed: {train_res.get('error')}")

    snapshot_id = getattr(model, "active_snapshot_id", None) or train_res.get("snapshot_id")
    if not snapshot_id:
        raise RuntimeError("Training succeeded but did not yield snapshot_id.")

    # If --ridge-base-cols provided, refit RidgeStacker on the training OOF stacker_data with the requested base_cols,
    # then save a NEW snapshot (reusing the already-trained base models) so backtest truly reflects the new Ridge design.
    if args.ridge_base_cols:
        try:
            stacker_data = getattr(model, "_last_stacker_data", None)
            if stacker_data is None or not isinstance(stacker_data, pd.DataFrame) or stacker_data.empty:
                raise RuntimeError("Missing model._last_stacker_data; cannot refit RidgeStacker with custom base_cols.")

            want_cols = [str(c) for c in args.ridge_base_cols]
            missing = [c for c in want_cols if c not in stacker_data.columns]
            if missing:
                raise RuntimeError(f"stacker_data missing required columns for ridge_base_cols: {missing}. Available={list(stacker_data.columns)[:30]}")

            from bma_models.ridge_stacker import RidgeStacker
            from bma_models.model_registry import load_models_from_snapshot, save_model_snapshot

            loaded = load_models_from_snapshot(str(snapshot_id), load_catboost=True)
            base_models = loaded.get("models") or {}
            lambda_model = loaded.get("lambda_rank_stacker") or getattr(model, "lambda_rank_stacker", None)
            lambda_pct = loaded.get("lambda_percentile_transformer") or getattr(model, "lambda_percentile_transformer", None)

            # Prepare snapshot payload in the format expected by save_model_snapshot
            formatted_models = {
                "elastic_net": {"model": base_models.get("elastic_net")},
                "xgboost": {"model": base_models.get("xgboost")},
                "catboost": {"model": base_models.get("catboost")},
            }

            ridge_alpha = 100.0
            try:
                # best-effort: read alpha from current config override if present
                import yaml  # type: ignore
                if temp_cfg_path and Path(temp_cfg_path).exists():
                    cfg = yaml.safe_load(Path(temp_cfg_path).read_text(encoding="utf-8")) or {}
                    ridge_alpha = float((cfg.get("training", {}) or {}).get("ridge_stacker", {}).get("alpha", ridge_alpha))
            except Exception:
                pass

            logger.info("🔧 Re-fitting RidgeStacker on OOF stacker_data with base_cols=%s (alpha=%s)", want_cols, ridge_alpha)
            ridge = RidgeStacker(base_cols=tuple(want_cols), alpha=float(ridge_alpha))
            ridge.fit(stacker_data, max_train_to_today=True)

            tag = f"time_split_ridge_basecols_{'-'.join(want_cols)}"
            snapshot_id = save_model_snapshot(
                training_results={"models": formatted_models},
                ridge_stacker=ridge,
                lambda_rank_stacker=lambda_model,
                rank_aware_blender=None,
                lambda_percentile_transformer=lambda_pct,
                tag=tag,
            )
            logger.info("✅ New snapshot with updated RidgeStacker saved: %s", snapshot_id)
        except Exception as e:
            logger.warning("Failed to refit/save RidgeStacker with --ridge-base-cols; falling back to training snapshot. err=%s", e)

    (run_dir / "snapshot_id.txt").write_text(str(snapshot_id), encoding="utf-8")
    logger.info("Snapshot: %s", snapshot_id)

    # Decide whether to load CatBoost for backtest based on ridge base columns.
    # If ridge uses 'pred_catboost', backtest must load catboost model to compute ridge_stacking.
    load_catboost_for_ridge = False
    try:
        from bma_models.model_registry import load_manifest
        import json as _json

        manifest = load_manifest(str(snapshot_id))
        ridge_meta_path = (manifest.get("paths") or {}).get("ridge_meta_json")
        if ridge_meta_path and Path(ridge_meta_path).exists():
            meta = _json.loads(Path(ridge_meta_path).read_text(encoding="utf-8"))
            base_cols = list(meta.get("base_cols") or [])
            load_catboost_for_ridge = "pred_catboost" in set(map(str, base_cols))
    except Exception as e:
        logger.warning("Could not inspect ridge_meta.json to decide CatBoost loading: %s", e)

    # Evaluate on test window only
    bt = ComprehensiveModelBacktest(
        data_dir=str(Path(args.data_dir)),
        snapshot_id=str(snapshot_id),
        data_file=str(Path(args.data_file)),
        start_date=str(test_start.date()),
        end_date=str(test_end.date()),
        allow_insample_backtest=False,
        load_catboost=bool(load_catboost_for_ridge),
    )
    bt._rebalance_mode = args.rebalance_mode
    bt._target_horizon_days = horizon
    bt._cost_bps = float(args.cost_bps or 0.0)
    # NOTE: load_catboost is handled in ComprehensiveModelBacktest.__init__ so models are loaded correctly.

    all_results, report_df, _weekly_details = bt.run_backtest(max_weeks=int(args.max_weeks))
    report_df.to_csv(run_dir / "report_df.csv", index=False, encoding="utf-8")

    # Benchmark returns (yfinance fallback) computed once on test-window rebalance dates of primary model.
    primary_model = str(args.model).strip()
    if primary_model not in all_results or all_results[primary_model].empty:
        raise RuntimeError(f"Model '{primary_model}' missing/empty in backtest results: {list(all_results.keys())}")

    top_n = int(args.top_n)
    bench = str(args.benchmark).upper().strip()
    # Use primary model's dates to fetch benchmark
    _preds_for_bench = all_results[primary_model]
    _tmp_summary, _tmp_ts = bt.calculate_group_returns(_preds_for_bench, top_n=top_n, bottom_n=top_n, cost_bps=float(args.cost_bps or 0.0))
    if _tmp_ts.empty:
        raise RuntimeError("Group return time series is empty on test window for benchmark date extraction.")
    bench_ret = _compute_benchmark_tplus_from_yfinance(bench, _tmp_ts["date"], horizon, logger)

    # Export TopN vs benchmark for multiple models (if provided), else only for --model
    models_to_export = [primary_model] if not args.models else [str(m).strip() for m in args.models if str(m).strip()]
    summaries = []
    for m in models_to_export:
        if m not in all_results or all_results[m].empty:
            logger.warning("Skipping model=%s (missing/empty in all_results). Available=%s", m, list(all_results.keys()))
            continue
        summaries.append(
            _write_model_topn_vs_benchmark(
                run_dir=run_dir,
                bt=bt,
                model_name=m,
                preds=all_results[m],
                top_n=top_n,
                horizon=horizon,
                bench=bench,
                bench_ret=bench_ret,
                cost_bps=float(args.cost_bps or 0.0),
                logger=logger,
            )
        )

    # Keep legacy filename for ridge_stacking for backward compatibility (if present)
    for s in summaries:
        if s.get("model") == "ridge_stacking":
            try:
                src = run_dir / f"ridge_stacking_top{top_n}_timeseries.csv"
                if src.exists():
                    src.replace(run_dir / "ridge_top20_timeseries.csv")
            except Exception:
                pass

    # Write explicit OOS metrics (so downstream summaries don't need to recompute)
    # NOTE: All metrics here are computed only from saved test-window series in this script.
    metrics = {
        "snapshot_id": str(snapshot_id),
        "model": primary_model,
        "top_n": top_n,
        "horizon_days": horizon,
        "split": split,
        "train_start": str(train_start.date()),
        "train_end": str(train_end.date()),
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
        "rebalance_mode": str(args.rebalance_mode),
        "max_weeks": int(args.max_weeks),
        "benchmark": bench,
        "cost_bps": _safe_float(args.cost_bps or 0.0),
        "n_test_rebalances": int(_tmp_ts.shape[0]),
    }

    # Add predictive + backtest summary metrics from ComprehensiveModelBacktest report_df (test window).
    # This is the authoritative source for IC/RankIC/MSE/MAE/R2 and (gross/net) avg_top_return at Top-30.
    try:
        if isinstance(report_df, pd.DataFrame) and "Model" in report_df.columns:
            rr = report_df.loc[report_df["Model"].astype(str) == str(primary_model)].head(1)
            if not rr.empty:
                row = rr.iloc[0].to_dict()
                metrics.update(
                    {
                        # predictive metrics (unitless / model scale)
                        "IC": _safe_float(row.get("IC")),
                        "IC_pvalue": _safe_float(row.get("IC_pvalue")),
                        "Rank_IC": _safe_float(row.get("Rank_IC")),
                        "Rank_IC_pvalue": _safe_float(row.get("Rank_IC_pvalue")),
                        "MSE": _safe_float(row.get("MSE")),
                        "MAE": _safe_float(row.get("MAE")),
                        "R2": _safe_float(row.get("R2")),
                        # backtest summary (Top-30 portfolio, per rebalance period; return units)
                        "avg_top_return": _safe_float(row.get("avg_top_return")),
                        "avg_top_return_net": _safe_float(row.get("avg_top_return_net")),
                        "avg_top_turnover": _safe_float(row.get("avg_top_turnover")),
                        "avg_top_cost": _safe_float(row.get("avg_top_cost")),
                        "win_rate": _safe_float(row.get("win_rate")),
                        "long_short_sharpe": _safe_float(row.get("long_short_sharpe")),
                    }
                )
    except Exception as e:
        logger.warning("Could not merge report_df metrics into oos_metrics: %s", e)
    # Add per-model TopN summaries to oos_metrics_all_models.csv (percent units) and keep oos_metrics.csv for primary model.
    if summaries:
        all_oos = pd.DataFrame(summaries)
        all_oos.to_csv(run_dir / "oos_topn_vs_benchmark_all_models.csv", index=False, encoding="utf-8")

    (run_dir / "oos_metrics.json").write_text(pd.Series(metrics).to_json(indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(run_dir / "oos_metrics.csv", index=False, encoding="utf-8")

    # NOTE: plotting is handled per-model in _write_model_topn_vs_benchmark().
    # Avoid duplicate single-model plotting here (which previously referenced an out-of-scope `out`).

    # NOTE: per-model OOS return logging is handled in _write_model_topn_vs_benchmark().
    logger.info("Saved outputs: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


