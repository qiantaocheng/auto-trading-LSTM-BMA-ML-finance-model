#!/usr/bin/env python3
"""Aggregate all LambdaRank top5-15 data from results/5fold_ticker_cv into one summary and one combined CSV."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CV_DIR = ROOT / "results" / "5fold_ticker_cv"
OUT_DIR = CV_DIR

def main():
    files = sorted(CV_DIR.glob("run_*/fold_*/run_*/lambdarank_top5_15_rebalance10d_accumulated.csv"))
    rows = []
    combined = []
    for f in files:
        # run_20260131_185204/fold_4/run_20260131_201122/lambdarank_top5_15...
        parts = f.relative_to(CV_DIR).parts
        run_id = parts[0]
        fold_id = parts[1]
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        n_periods = len(df)
        final_acc_return = df["acc_return"].iloc[-1] if n_periods else float("nan")
        min_dd = df["drawdown"].min() if "drawdown" in df.columns else float("nan")
        mean_period_ret = df["top_gross_return"].mean() * 100 if n_periods else float("nan")
        rows.append({
            "run_id": run_id,
            "fold": fold_id,
            "final_acc_return_pct": final_acc_return * 100,
            "min_drawdown_pct": min_dd,
            "n_periods": n_periods,
            "mean_period_return_pct": mean_period_ret,
        })
        df = df.copy()
        df["run_id"] = run_id
        df["fold"] = fold_id
        combined.append(df)
    summary = pd.DataFrame(rows)
    out_summary = OUT_DIR / "lambdarank_top5_15_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"Wrote {out_summary} ({len(rows)} rows)")
    comb = pd.concat(combined, ignore_index=True)
    out_comb = OUT_DIR / "lambdarank_top5_15_all_folds.csv"
    comb.to_csv(out_comb, index=False)
    print(f"Wrote {out_comb} ({len(comb)} rows)")
    # Print summary table
    print("\nLambdaRank Top 5-15 累计收益 (by run & fold):")
    print(summary.to_string(index=False))
    print("\nCross-fold averages by run:")
    by_run = summary.groupby("run_id").agg({
        "final_acc_return_pct": "mean",
        "min_drawdown_pct": "mean",
        "n_periods": "mean",
        "mean_period_return_pct": "mean",
    }).round(4)
    print(by_run.to_string())

if __name__ == "__main__":
    main()
