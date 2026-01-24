#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""80/20 evaluation comparing runs with and without obv_divergence."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DEFAULT = PROJECT_ROOT / "data" / "factor_exports" / "polygon_factors_all_filtered_clean_final_v2.parquet"
RECALC_PATH = PROJECT_ROOT / "data" / "factor_exports" / "polygon_factors_all_filtered_clean_final_v2_recalculated.parquet"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "obv_divergence_8020_comparison"
FACTOR_ENGINE = PROJECT_ROOT / "bma_models" / "simple_25_factor_engine.py"


def choose_data_file() -> Path:
    if RECALC_PATH.exists():
        return RECALC_PATH
    return DATA_DEFAULT


def load_dataframe(data_file: Path) -> pd.DataFrame:
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    df = pd.read_parquet(data_file)
    if not isinstance(df.index, pd.MultiIndex) or 'ticker' not in df.index.names:
        raise ValueError("Dataset must have a MultiIndex with 'ticker'")
    required = {'target', 'Close', 'obv_divergence'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    return df


def filter_dataframe(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        mask = df.index.get_level_values('ticker').isin(tickers)
        return df.loc[mask]
    raise ValueError("Dataset index must be MultiIndex")


def load_metrics_from_report(report_csv: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not report_csv.exists():
        return metrics
    try:
        report_df = pd.read_csv(report_csv)
    except Exception as err:  # pragma: no cover
        print(f"[WARN] Failed to read {report_csv}: {err}")
        return metrics
    if report_df.empty:
        return metrics
    row = None
    if 'Model' in report_df.columns:
        ridge = report_df[report_df['Model'] == 'ridge_stacking']
        if not ridge.empty:
            row = ridge.iloc[0]
    if row is None:
        row = report_df.iloc[0]
    def assign(col: str, key: str, scale: float = 1.0) -> None:
        if col in row and pd.notna(row[col]):
            metrics[key] = float(row[col]) * scale
    assign('IC', 'ic')
    assign('Rank_IC', 'rank_ic')
    if 'win_rate' in row and pd.notna(row['win_rate']):
        val = float(row['win_rate'])
        metrics['win_rate'] = val * 100.0 if val < 1.0 else val
    assign('avg_top_return', 'avg_return', scale=100.0)
    return metrics


def find_report_dir(run_output_dir: Path) -> Path:
    csv_path = run_output_dir / 'report_df.csv'
    if csv_path.exists():
        return run_output_dir
    nested = sorted(
        [p for p in run_output_dir.glob('run_*') if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for cand in nested:
        if (cand / 'report_df.csv').exists():
            return cand
    return run_output_dir


def run_eval(label: str, df_filtered: pd.DataFrame, include_obv: bool) -> Dict[str, Any]:
    print("=" * 80)
    print(f"80/20 evaluation: {label.upper()}")
    print("=" * 80)

    backup_file = None
    modified = False
    if not include_obv:
        backup_file = FACTOR_ENGINE.with_suffix('.py.backup_obv_compare')
        FACTOR_ENGINE.replace(backup_file)
        text = backup_file.read_text(encoding='utf-8')
        text = text.replace("'obv_divergence',  # OBV divergence", "# 'obv_divergence',  # temporarily removed")
        FACTOR_ENGINE.write_text(text, encoding='utf-8')
        modified = True
        print("[INFO] Temporarily removed obv_divergence from factor list")

    temp_file = OUTPUT_ROOT / f"temp_data_{label}.parquet"
    df_filtered.to_parquet(temp_file)
    run_output_dir = OUTPUT_ROOT / f"run_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    eval_cmd = [
        sys.executable,
        "scripts/time_split_80_20_oos_eval.py",
        "--data-file", str(temp_file),
        "--horizon-days", "10",
        "--split", "0.8",
        "--top-n", "20",
        "--log-level", "INFO",
        "--output-dir", str(run_output_dir),
    ]
    print("Command:", ' '.join(eval_cmd))

    start_time = time.time()
    try:
        result = subprocess.run(
            eval_cmd,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        elapsed = time.time() - start_time
        report_dir = find_report_dir(run_output_dir)
        metrics = load_metrics_from_report(report_dir / 'report_df.csv')
        return {
            'success': True,
            'elapsed_time': elapsed,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'output_dir': str(report_dir),
            'metrics': metrics,
        }
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - start_time
        return {
            'success': False,
            'elapsed_time': elapsed,
            'stdout': exc.stdout,
            'stderr': exc.stderr,
        }
    finally:
        if temp_file.exists():
            temp_file.unlink()
        if modified:
            FACTOR_ENGINE.unlink()
            backup_file.rename(FACTOR_ENGINE)


def main() -> None:
    data_file = choose_data_file()
    print("Data file:", data_file)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(data_file)
    tickers = sorted(df.index.get_level_values('ticker').unique())
    print(f"Total tickers: {len(tickers)} (using full universe)")
    df_filtered = filter_dataframe(df, tickers)
    print("Filtered shape:", df_filtered.shape)

    with_results = run_eval('with_obv', df_filtered, include_obv=True)
    without_results = run_eval('without_obv', df_filtered, include_obv=False)

    comparison = {
        'timestamp': datetime.now().isoformat(),
        'tickers_used': len(tickers),
        'with_obv_divergence': with_results,
        'without_obv_divergence': without_results,
    }
    diff = {}
    for key in ('ic', 'rank_ic', 'win_rate', 'avg_return'):
        w = with_results.get('metrics', {}).get(key)
        wo = without_results.get('metrics', {}).get(key)
        if w is not None and wo is not None:
            diff[key] = w - wo
    comparison['difference'] = diff

    output_file = OUTPUT_ROOT / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding='utf-8')
    print("Saved comparison:", output_file)


if __name__ == '__main__':
    main()
