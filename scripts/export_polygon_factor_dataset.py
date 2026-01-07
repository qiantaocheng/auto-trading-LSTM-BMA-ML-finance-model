#!/usr/bin/env python3
"""CLI wrapper for exporting polygon factor dataset."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Ensure repo root is on sys.path before importing local modules
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from autotrader.factor_export_service import export_polygon_factors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export polygon factor dataset for ML training')
    parser.add_argument('--max-symbols', type=int, default=2600)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--years', type=int, default=5)
    parser.add_argument('--output', type=str, default='data/factor_exports')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'])
    parser.add_argument('--horizon', type=int, default=5, help='Forward return horizon (e.g., 5 or 10)')
    parser.add_argument('--symbols-file', type=str, default=None, help='Text file with one ticker per line')
    parser.add_argument('--symbols-manifest', type=str, default=None, help='manifest.parquet with a symbols column')
    parser.add_argument('--log-level', type=str, default='INFO')
    return parser.parse_args()

def _load_symbols_from_file(path: str) -> List[str]:
    syms: List[str] = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            syms.append(s.upper())
    # Preserve order, drop dupes
    out: List[str] = []
    seen = set()
    for s in syms:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

def _load_symbols_from_manifest(path: str) -> List[str]:
    import pandas as pd
    import numpy as np

    m = pd.read_parquet(path)
    if 'symbols' not in m.columns:
        raise ValueError(f"manifest missing 'symbols' column: {path}")
    syms: List[str] = []
    for lst in m['symbols'].tolist():
        if isinstance(lst, (list, tuple, np.ndarray, pd.Series)):
            syms.extend([str(x).upper() for x in list(lst) if str(x).strip()])
        elif isinstance(lst, str):
            # Fallback for stringified lists: "['A', 'AA', ...]"
            cleaned = lst.strip()
            cleaned = cleaned.strip('[]')
            for tok in cleaned.replace("'", "").replace('"', '').split(','):
                t = tok.strip().upper()
                if t:
                    syms.append(t)
    # Preserve order, drop dupes
    out: List[str] = []
    seen = set()
    for s in syms:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format='%(asctime)s %(levelname)s %(name)s - %(message)s')

    symbols: Optional[List[str]] = None
    if args.symbols_file:
        symbols = _load_symbols_from_file(args.symbols_file)
    elif args.symbols_manifest:
        symbols = _load_symbols_from_manifest(args.symbols_manifest)

    export_polygon_factors(
        max_symbols=args.max_symbols,
        batch_size=args.batch_size,
        start_date=args.start_date,
        end_date=args.end_date,
        years=args.years,
        output_dir=args.output,
        log_level=args.log_level,
        mode=args.mode,
        horizon=args.horizon,
        symbols=symbols,
    )


if __name__ == '__main__':
    main()
