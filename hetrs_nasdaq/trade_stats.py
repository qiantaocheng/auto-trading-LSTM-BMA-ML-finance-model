from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeStats:
    rows: int
    entries_total: int
    exits_total: int
    flips: int
    long_entries: int
    long_exits: int
    short_entries: int
    short_exits: int
    est_trades_per_year: float
    avg_hold_days: float
    median_hold_days: float


def compute_trade_stats_from_positions(pos: pd.Series) -> TradeStats:
    pos = pos.fillna(0.0).astype(float)
    ch = pos.diff().fillna(pos)

    # A "flip" is going from +1 to -1 or -1 to +1 in one step
    flips = int(((pos.shift(1).fillna(0.0) != 0.0) & (pos != 0.0) & (np.sign(pos) != np.sign(pos.shift(1).fillna(0.0)))).sum())

    entries = (ch != 0) & (pos != 0)
    exits = (ch != 0) & (pos == 0)

    long_entries = int(((ch != 0) & (pos > 0)).sum())
    short_entries = int(((ch != 0) & (pos < 0)).sum())

    prev = pos.shift(1).fillna(0.0)
    long_exits = int(((ch != 0) & (prev > 0) & (pos <= 0)).sum())
    short_exits = int(((ch != 0) & (prev < 0) & (pos >= 0)).sum())

    n_entries = int(entries.sum())
    n_exits = int(exits.sum())
    n_days = int(len(pos))
    years = n_days / 252.0
    trades_per_year = float(n_entries / years) if years > 0 else float("nan")

    # Holding period stats: pair each entry timestamp to the next exit timestamp
    entry_idx = pos.index[entries]
    exit_idx = pos.index[exits]
    hold_lengths = []
    for t in entry_idx:
        nxt = exit_idx[exit_idx > t]
        if len(nxt) == 0:
            break
        hold_lengths.append((nxt[0] - t).days)

    avg_hold = float(np.mean(hold_lengths)) if hold_lengths else float("nan")
    med_hold = float(np.median(hold_lengths)) if hold_lengths else float("nan")

    return TradeStats(
        rows=n_days,
        entries_total=n_entries,
        exits_total=n_exits,
        flips=flips,
        long_entries=long_entries,
        long_exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        est_trades_per_year=trades_per_year,
        avg_hold_days=avg_hold,
        median_hold_days=med_hold,
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute buy/sell frequency from a backtest timeseries.")
    p.add_argument(
        "--csv",
        required=True,
        help="Path to meta_timeseries.csv or naive_timeseries.csv (must contain 'position' column).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path, parse_dates=["Unnamed: 0"]).rename(columns={"Unnamed: 0": "date"}).set_index("date")
    if "position" not in df.columns:
        raise ValueError("CSV must contain a 'position' column.")
    s = compute_trade_stats_from_positions(df["position"])
    print(
        f"rows={s.rows} entries={s.entries_total} exits={s.exits_total} flips={s.flips} "
        f"long_entries={s.long_entries} short_entries={s.short_entries} "
        f"trades_per_year≈{s.est_trades_per_year:.2f} "
        f"avg_hold_days≈{s.avg_hold_days:.1f} median_hold_days≈{s.median_hold_days:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


