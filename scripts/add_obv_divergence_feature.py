#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add obv_divergence column to subset parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def add_obv_divergence(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    df = pd.read_parquet(input_path)

    if 'obv_momentum_40d' not in df.columns:
        raise KeyError("Column 'obv_momentum_40d' is required but missing from dataset.")

    df = df.copy()
    df['obv_divergence'] = df['obv_momentum_40d']

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    print(f"[OK] Saved subset with obv_divergence: {output_path}")
    print(f"      Shape: {df.shape}, MultiIndex: {isinstance(df.index, pd.MultiIndex)}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    default_input = project_root / "data" / "factor_exports" / "polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet"
    default_output = default_input.with_name(default_input.stem + "_with_obv_divergence.parquet")

    parser = argparse.ArgumentParser(description="Add obv_divergence feature to subset parquet")
    parser.add_argument('--input', type=Path, default=default_input)
    parser.add_argument('--output', type=Path, default=default_output)
    args = parser.parse_args()

    add_obv_divergence(args.input, args.output)


if __name__ == '__main__':
    main()
