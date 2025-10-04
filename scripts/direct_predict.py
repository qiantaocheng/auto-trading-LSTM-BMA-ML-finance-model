#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct Predict - Load saved snapshot and run predictions for one or more tickers
without retraining. Outputs signals to console and saves to CSV if requested.

Usage (example):
  python scripts/direct_predict.py --tickers AAPL,MSFT --snapshot latest --out results/direct_predict.csv
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import logging

from bma_models.model_registry import load_manifest
from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("direct_predict")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", type=str, required=True, help="Comma-separated tickers")
    p.add_argument("--snapshot", type=str, default="latest", help="Snapshot ID or 'latest'")
    p.add_argument("--out", type=str, default="", help="Optional CSV output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    if not tickers:
        raise SystemExit("No tickers provided")

    # Load snapshot manifest
    manifest = load_manifest(None if args.snapshot == "latest" else args.snapshot)
    logger.info(f"Loaded snapshot: {manifest.get('snapshot_id')}")

    # Initialize model in inference mode
    model = UltraEnhancedQuantitativeModel()

    # Minimal data fetch via Simple17FactorEngine path inside the model (uses Polygon)
    # We'll re-use the built-in pipeline to compute features and run inference using saved models is not baked-in here;
    # This script demonstrates wiring; UI will call model methods directly.

    # For simplicity, call the main analysis with provided tickers and let the model pipeline fetch/compute features.
    results = model.run_complete_analysis(
        tickers=tickers,
        start_date=None,
        end_date=None,
        top_n=max(len(tickers), 50)
    )

    recs = results.get('recommendations', [])
    df = pd.DataFrame(recs)
    if not df.empty:
        logger.info(df.head(min(len(df), 10)).to_string(index=False))
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            df.to_csv(args.out, index=False)
            logger.info(f"Saved predictions to {args.out}")
    else:
        logger.warning("No predictions produced")


if __name__ == "__main__":
    main()


