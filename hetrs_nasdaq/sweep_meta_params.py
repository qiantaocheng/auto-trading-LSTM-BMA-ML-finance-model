from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hetrs_nasdaq.backtest import CPCVConfig
from hetrs_nasdaq.backtest_v2 import V31Config, run_cpcv_meta_backtest
from hetrs_nasdaq.repro import set_global_seed


@dataclass(frozen=True)
class SweepGrid:
    primary_thresholds: tuple[float, ...]
    meta_prob_thresholds: tuple[float, ...]


def plot_heatmap(
    pivot: pd.DataFrame,
    title: str,
    out_path: str,
    fmt: str = ".3f",
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    x = pivot.columns.values
    y = pivot.index.values
    z = pivot.values.astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(z, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(x)))
    ax.set_yticks(np.arange(len(y)))
    ax.set_xticklabels([str(v) for v in x])
    ax.set_yticklabels([str(v) for v in y])
    ax.set_xlabel("meta_prob_threshold")
    ax.set_ylabel("primary_threshold")
    ax.set_title(title)

    # annotate
    for i in range(len(y)):
        for j in range(len(x)):
            val = z[i, j]
            if np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8, color="white")

    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep meta-label parameters (HETRS-NASDAQ v3.1).")
    p.add_argument("--in", dest="inp", required=True, help="Input parquet (features + tft preds)")
    p.add_argument("--outdir", required=True, help="Output directory for sweep artifacts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-groups", type=int, default=6)
    p.add_argument("--k-test-groups", type=int, default=2)
    p.add_argument("--purge", type=int, default=5)
    p.add_argument("--embargo", type=float, default=0.01)
    p.add_argument(
        "--primary-thresholds",
        type=str,
        default="0.001,0.002,0.003,0.004",
        help="Comma-separated list",
    )
    p.add_argument(
        "--meta-prob-thresholds",
        type=str,
        default="0.50,0.55,0.60,0.65,0.70,0.75",
        help="Comma-separated list",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    set_global_seed(args.seed)

    df = pd.read_parquet(args.inp).sort_index()
    cfg = CPCVConfig(
        n_groups=int(args.n_groups),
        k_test_groups=int(args.k_test_groups),
        purge=int(args.purge),
        embargo=float(args.embargo),
    )

    primary_thresholds = tuple(float(x.strip()) for x in args.primary_thresholds.split(",") if x.strip())
    meta_prob_thresholds = tuple(float(x.strip()) for x in args.meta_prob_thresholds.split(",") if x.strip())
    grid = SweepGrid(primary_thresholds=primary_thresholds, meta_prob_thresholds=meta_prob_thresholds)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pt in grid.primary_thresholds:
        for mp in grid.meta_prob_thresholds:
            run_dir = outdir / f"pt_{pt:.4f}_mp_{mp:.2f}"
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics = run_cpcv_meta_backtest(
                df,
                outdir=str(run_dir),
                cfg=cfg,
                v31=V31Config(primary_threshold=float(pt), meta_prob_threshold=float(mp), seed=int(args.seed)),
            )
            rows.append({"primary_threshold": pt, "meta_prob_threshold": mp, **metrics})

    res = pd.DataFrame(rows).sort_values(["primary_threshold", "meta_prob_threshold"])
    res.to_csv(outdir / "sweep_results.csv", index=False)

    # Best configs
    best_sharpe = res.loc[res["meta_sharpe"].idxmax()].to_dict()
    best_return = res.loc[res["meta_cum_return"].idxmax()].to_dict()
    best_dd = res.loc[res["meta_max_dd"].idxmax()].to_dict()  # less negative -> higher
    (outdir / "best_configs.json").write_text(
        json_dumps(
            {
                "best_meta_sharpe": best_sharpe,
                "best_meta_cum_return": best_return,
                "best_meta_max_dd": best_dd,
            }
        )
    )

    # Heatmaps
    for col, title, fname in [
        ("meta_sharpe", "Meta-TFT Sharpe", "heatmap_meta_sharpe.png"),
        ("meta_cum_return", "Meta-TFT Cumulative Return", "heatmap_meta_cum_return.png"),
        ("meta_max_dd", "Meta-TFT Max Drawdown (higher is better)", "heatmap_meta_max_dd.png"),
    ]:
        piv = res.pivot(index="primary_threshold", columns="meta_prob_threshold", values=col)
        plot_heatmap(piv, title=title, out_path=str(outdir / fname), fmt=".3f")

    print(f"[sweep_meta_params] saved: {outdir}")
    return 0


def json_dumps(obj: object) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    raise SystemExit(main())


