#!/usr/bin/env python3
"""\
Utility to validate Polygon-backed Simple25 factor coverage and optionally run the
BMA Ultra analysis pipeline on live data.

Usage examples:
  python scripts/test_simple25_polygon.py --tickers AAPL MSFT NVDA --start-date 2024-06-01 --end-date 2024-09-30
  python scripts/test_simple25_polygon.py --tickers-file bma_models/default_tickers.txt --count 25 --run-analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

# Ensure repository root is on sys.path so bma_models can be imported when the
# default working directory is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bma_models.simple_25_factor_engine import (  # type: ignore
    T10_ALPHA_FACTORS,
    Simple17FactorEngine,
)
from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel  # type: ignore
REQUIRED_FACTORS = list(T10_ALPHA_FACTORS)

def load_tickers_from_file(path: Path, limit: int | None = None) -> List[str]:
    """Load tickers (one per line) from the given file."""
    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")

    tickers: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tickers.append(line)
            if limit and len(tickers) >= limit:
                break
    return tickers


def determine_tickers(args: argparse.Namespace) -> List[str]:
    if args.tickers:
        return list(dict.fromkeys(args.tickers))  # preserve order, remove duplicates

    tickers_file = Path(args.tickers_file)
    limit = args.count if args.count and args.count > 0 else None
    tickers = load_tickers_from_file(tickers_file, limit)
    if not tickers:
        raise ValueError(f"No tickers loaded from {tickers_file}")
    return tickers


def fetch_and_compute_factors(
    tickers: Sequence[str],
    start_date: str | None,
    end_date: str | None,
    lookback_days: int,
    skip_standardization: bool,
    enable_sentiment: bool,
) -> dict:
    engine = Simple17FactorEngine(
        lookback_days=lookback_days,
        enable_sentiment=enable_sentiment,
        skip_cross_sectional_standardization=skip_standardization,
    )

    market_data = engine.fetch_market_data(
        symbols=list(tickers),
        start_date=start_date,
        end_date=end_date,
    )
    if market_data.empty:
        raise RuntimeError("Polygon market data fetch returned an empty DataFrame")

    factors = engine.compute_all_17_factors(market_data)
    if factors.empty:
        raise RuntimeError("Factor computation returned an empty DataFrame")

    missing = sorted(set(REQUIRED_FACTORS) - set(factors.columns))
    extras = sorted(set(factors.columns) - set(REQUIRED_FACTORS + ["Close"]))

    summary = {
        "market_data_shape": market_data.shape,
        "factor_shape": factors.shape,
        "start": factors.index.get_level_values(0).min().isoformat() if isinstance(factors.index, pd.MultiIndex) else None,
        "end": factors.index.get_level_values(0).max().isoformat() if isinstance(factors.index, pd.MultiIndex) else None,
        "missing_factors": missing,
        "extra_columns": extras,
    }

    return {
        "engine": engine,
        "market_data": market_data,
        "factors": factors,
        "summary": summary,
    }


def maybe_save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path)


def run_analysis(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    top_n: int,
    config_path: str,
) -> dict:
    model = UltraEnhancedQuantitativeModel(config_path=config_path)
    model.enable_simple_25_factors(True)
    results = model.run_complete_analysis(
        tickers=list(tickers),
        start_date=start_date,
        end_date=end_date,
        top_n=top_n,
    )
    return results


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple25 Polygon factor validation utility")
    parser.add_argument("--tickers", nargs="*", help="Explicit list of tickers to use")
    parser.add_argument(
        "--tickers-file",
        default=str(REPO_ROOT / "bma_models" / "default_tickers.txt"),
        help="Path to ticker universe file (default: bma_models/default_tickers.txt)",
    )
    parser.add_argument("--count", type=int, default=10, help="Number of tickers to sample from the file")
    parser.add_argument("--start-date", default=None, help="Start date (YYYY-MM-DD). Defaults to lookback window.")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--lookback", type=int, default=252, help="Lookback window in days when dates are omitted")
    parser.add_argument("--skip-standardization", action="store_true", help="Skip cross-sectional standardization (prediction mode)")
    parser.add_argument("--enable-sentiment", action="store_true", help="Compute Polygon+FinBERT sentiment features if available")
    parser.add_argument("--save", type=str, help="Optional path to save computed factors (CSV or Parquet)")
    parser.add_argument("--summary", action="store_true", help="Only print JSON summary of factor coverage")
    parser.add_argument("--run-analysis", action="store_true", help="Run UltraEnhancedQuantitativeModel.run_complete_analysis afterwards")
    parser.add_argument("--analysis-top-n", type=int, default=20, help="Top-N recommendations for the analysis stage")
    parser.add_argument("--config", type=str, default="alphas_config.yaml", help="Path to unified model config file for analysis")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])

    tickers = determine_tickers(args)
    print(f"Selected {len(tickers)} tickers: {tickers[:10]}{'...' if len(tickers) > 10 else ''}")

    payload = fetch_and_compute_factors(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=args.lookback,
        skip_standardization=args.skip_standardization,
        enable_sentiment=args.enable_sentiment,
    )

    summary = payload["summary"]
    if args.summary:
        print(json.dumps(summary, indent=2))
    else:
        print("\n=== Polygon Factor Fetch Summary ===")
        print(f"Market data shape: {summary['market_data_shape']}")
        print(f"Factor matrix shape: {summary['factor_shape']}")
        print(f"Date range: {summary['start']} -> {summary['end']}")
        print(f"Missing factors: {summary['missing_factors'] or 'None'}")
        print(f"Extra columns: {summary['extra_columns'] or 'None'}")

    if args.save:
        output_path = Path(args.save)
        maybe_save_dataframe(payload["factors"], output_path)
        print(f"Saved factors to {output_path}")

    if args.run_analysis:
        if not args.start_date or not args.end_date:
            raise ValueError("start-date and end-date are required when --run-analysis is set")
        print("\n=== Running complete BMA Ultra analysis ===")
        try:
            results = run_analysis(
                tickers=tickers,
                start_date=args.start_date,
                end_date=args.end_date,
                top_n=args.analysis_top_n,
                config_path=args.config,
            )
            print("Analysis success:" if results.get("success") else "Analysis failed!")
            if results.get("success"):
                print(json.dumps({k: v for k, v in results.items() if k not in {"feature_engineering", "predictions"}}, indent=2, default=str))
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[ERROR] run_complete_analysis failed: {exc}")
            raise


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
