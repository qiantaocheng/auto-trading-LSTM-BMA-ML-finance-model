#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter tickers by Polygon market cap and save formatted list.
Optimized with concurrent processing and rate limiting."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import requests

POLYGON_BASE = "https://api.polygon.io"

# Global rate limiter
_rate_lock = Lock()
_last_request_time = 0.0
_rate_limit_delay = 0.2  # 200ms between requests (5 requests/second max)


def _rate_limited_request():
    """Ensure minimum delay between API requests."""
    global _last_request_time
    with _rate_lock:
        elapsed = time.time() - _last_request_time
        if elapsed < _rate_limit_delay:
            time.sleep(_rate_limit_delay - elapsed)
        _last_request_time = time.time()


def _get_json(url: str, params: Optional[dict] = None, timeout: int = 30, max_retries: int = 3) -> dict:
    """Simple GET with retry/backoff for rate limits."""
    last_exc: Optional[requests.HTTPError] = None
    for attempt in range(max_retries):
        _rate_limited_request()  # Rate limiting before each request
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = 0.8 * (2 ** attempt)
            time.sleep(wait)
            last_exc = requests.HTTPError(resp.text, response=resp)
            continue
        resp.raise_for_status()
        return resp.json()
    if last_exc:
        raise last_exc
    resp.raise_for_status()
    return {}


def fetch_market_cap(api_key: str, ticker: str) -> Optional[float]:
    """Fetch market cap for a single ticker via /v3/reference/tickers/{ticker}."""
    url = f"{POLYGON_BASE}/v3/reference/tickers/{ticker}"
    payload = _get_json(url, params={'apiKey': api_key})
    results = payload.get('results') or {}
    mc = results.get('market_cap')
    if mc is None:
        return None
    try:
        return float(mc)
    except (TypeError, ValueError):
        return None


def load_tickers(file_path: Path, limit: Optional[int] = None) -> List[str]:
    tickers: List[str] = []
    with file_path.open('r', encoding='utf-8') as fh:
        for line in fh:
            symbol = line.strip().upper()
            if not symbol:
                continue
            tickers.append(symbol)
            if limit and len(tickers) >= limit:
                break
    return tickers


def _process_single_ticker(args: Tuple[str, str, float, int, int, bool]) -> Tuple[str, Dict[str, Optional[float]], bool]:
    """Process a single ticker (for concurrent execution)."""
    ticker, api_key, min_market_cap, idx, total, verbose = args
    info = {'market_cap': None}
    passed = False
    
    try:
        mc = fetch_market_cap(api_key, ticker)
    except Exception as exc:
        info['error'] = str(exc)
        if verbose:
            print(f"[{idx}/{total}] {ticker}: error {exc}")
        return ticker, info, passed

    info['market_cap'] = mc
    if mc is not None and mc >= min_market_cap:
        passed = True
        if verbose:
            print(f"[{idx}/{total}] {ticker}: market cap {mc:,.0f} ✔")
    else:
        if verbose:
            print(f"[{idx}/{total}] {ticker}: market cap {mc} ✘")
    
    return ticker, info, passed


def filter_tickers_polygon(
    tickers: List[str],
    api_key: str,
    min_market_cap: float,
    verbose: bool = True,
    max_workers: int = 5,
    checkpoint_file: Optional[Path] = None,
) -> Tuple[List[str], Dict[str, Dict[str, Optional[float]]]]:
    """
    Filter tickers with concurrent processing and checkpoint support.
    
    Args:
        tickers: List of ticker symbols
        api_key: Polygon API key
        min_market_cap: Minimum market cap threshold
        verbose: Print progress
        max_workers: Number of concurrent threads (default: 5)
        checkpoint_file: Optional path to save/load progress
    """
    passed: List[str] = []
    detail: Dict[str, Dict[str, Optional[float]]] = {}
    
    # Load checkpoint if exists
    processed_tickers = set()
    if checkpoint_file and checkpoint_file.exists():
        try:
            checkpoint_data = json.loads(checkpoint_file.read_text(encoding='utf-8'))
            processed_tickers = set(checkpoint_data.get('processed', []))
            passed = checkpoint_data.get('passed', [])
            detail = checkpoint_data.get('detail', {})
            print(f"[Checkpoint] Loaded: {len(processed_tickers)} tickers already processed")
        except Exception as e:
            print(f"[Warn] Failed to load checkpoint: {e}")
    
    # Filter out already processed tickers
    remaining_tickers = [(t, i+1) for i, t in enumerate(tickers) if t not in processed_tickers]
    
    if not remaining_tickers:
        print("[Done] All tickers already processed!")
        return passed, detail
    
    print(f"[Start] Processing {len(remaining_tickers)} tickers with {max_workers} workers...")
    start_time = time.time()
    
    # Prepare arguments for concurrent processing
    tasks = [
        (ticker, api_key, min_market_cap, idx, len(tickers), verbose)
        for ticker, idx in remaining_tickers
    ]
    
    # Process with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_ticker, task): task[0] for task in tasks}
        
        completed = 0
        checkpoint_interval = 100  # Save checkpoint every 100 tickers
        
        for future in as_completed(futures):
            ticker = futures[future]
            completed += 1
            
            try:
                ticker_result, info, ticker_passed = future.result()
                detail[ticker_result] = info
                if ticker_passed:
                    passed.append(ticker_result)
            except Exception as e:
                detail[ticker] = {'error': str(e), 'market_cap': None}
                if verbose:
                    print(f"❌ {ticker}: Exception {e}")
            
            # Save checkpoint periodically
            if checkpoint_file and completed % checkpoint_interval == 0:
                checkpoint_data = {
                    'processed': list(processed_tickers) + [t for t, _ in remaining_tickers[:completed]],
                    'passed': passed,
                    'detail': detail,
                }
                checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2), encoding='utf-8')
                if verbose:
                    print(f"[Checkpoint] Saved: {completed}/{len(remaining_tickers)}")
    
    # Final checkpoint save
    if checkpoint_file:
        checkpoint_data = {
            'processed': list(set(tickers)),
            'passed': passed,
            'detail': detail,
        }
        checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2), encoding='utf-8')
    
    elapsed = time.time() - start_time
    print(f"[Done] Completed in {elapsed:.1f}s ({elapsed/len(remaining_tickers):.2f}s per ticker)")
    
    return passed, detail


def write_list_file(tickers: List[str], path: Path) -> None:
    formatted = ", ".join(f"'{ticker}'" for ticker in tickers)
    path.write_text(formatted, encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Filter tickers by Polygon market cap (optimized with concurrency).')
    parser.add_argument('--input', type=Path, default=Path('us_all_tickers.txt'))
    parser.add_argument('--output', type=Path, default=Path('filtered_tickers_over_100m.txt'))
    parser.add_argument('--detail-output', type=Path, default=None)
    parser.add_argument('--api-key', type=str, default=os.environ.get('POLYGON_API_KEY'))
    parser.add_argument('--min-market-cap', type=float, default=100_000_000)
    parser.add_argument('--max-tickers', type=int, default=None, help='Optional limit for tickers processed (for testing).')
    parser.add_argument('--max-workers', type=int, default=5, help='Number of concurrent threads (default: 5)')
    parser.add_argument('--checkpoint', type=Path, default=None, help='Checkpoint file for progress saving/resuming')
    parser.add_argument('--no-verbose', action='store_true')
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit('Polygon API key required via --api-key or POLYGON_API_KEY env var')
    if not args.input.exists():
        raise SystemExit(f'Input file not found: {args.input}')

    tickers = load_tickers(args.input, limit=args.max_tickers)
    print(f"[Info] Loaded {len(tickers)} tickers from {args.input}")

    # Auto-generate checkpoint file if not specified
    checkpoint_file = args.checkpoint
    if checkpoint_file is None and args.max_tickers is None:
        checkpoint_file = args.output.parent / f"{args.output.stem}_checkpoint.json"

    passed, detail = filter_tickers_polygon(
        tickers=tickers,
        api_key=args.api_key,
        min_market_cap=args.min_market_cap,
        verbose=not args.no_verbose,
        max_workers=args.max_workers,
        checkpoint_file=checkpoint_file,
    )
    print(f"[Result] Tickers passing market_cap >= {args.min_market_cap:.0f}: {len(passed)}/{len(tickers)}")

    write_list_file(passed, args.output)
    print(f"[Saved] Formatted list to {args.output}")

    if args.detail_output:
        args.detail_output.write_text(json.dumps(detail, indent=2), encoding='utf-8')
        print(f"[Saved] Detail JSON to {args.detail_output}")


if __name__ == '__main__':
    main()
