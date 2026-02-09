#!/usr/bin/env python
"""Earnings Surprise Scanner for TraderApp.

FAST approach: bulk-fetch ALL recent news (no per-ticker filter), filter locally
against our ticker universe for earnings beat keywords, then concurrent gap calc.

Output: JSON array on stdout.
Progress: JSON lines on stderr.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def emit_progress(step: str, progress: int, detail: str = "") -> None:
    msg = json.dumps({"step": step, "progress": progress, "detail": detail})
    print(msg, file=sys.stderr, flush=True)


# Keywords indicating earnings beat
BEAT_KEYWORDS = [
    "beat expectations", "beat estimates", "beats expectations", "beats estimates",
    "topped expectations", "topped estimates", "exceeded expectations", "exceeded estimates",
    "surpassed expectations", "surpassed estimates", "above expectations", "above estimates",
    "better than expected", "better-than-expected", "earnings beat", "earnings surprise",
    "strong earnings", "strong results", "blowout quarter", "record revenue",
    "raised guidance", "raises guidance", "upbeat", "outperformed",
]

# Keywords indicating earnings miss
MISS_KEYWORDS = [
    "missed expectations", "missed estimates", "misses expectations", "misses estimates",
    "below expectations", "below estimates", "fell short", "falls short",
    "worse than expected", "worse-than-expected", "earnings miss", "weak earnings",
    "weak results", "disappointing", "lowered guidance", "lowers guidance",
    "cut guidance", "cuts guidance", "underperformed",
]

# Event words that MUST appear in the TITLE to confirm it's a relevant earnings/corporate event
# This filters out generic market roundup articles that mention tickers in passing
TITLE_EVENT_WORDS = [
    "earnings", "outlook", "guidance", "results", "revenue", "eps",
    "dividend", "sec", "lawsuit", "profit", "quarterly", "fiscal",
    "beat", "miss", "surprise", "report",
]

# Keywords indicating the article is about earnings (checked in title + description)
EARNINGS_KEYWORDS = [
    "earnings", "quarterly results", "quarter results", "fiscal quarter",
    "revenue", "profit", "eps", "earnings per share",
    "financial results", "quarterly report", "q1", "q2", "q3", "q4",
]


def is_earnings_article(title: str, description: str) -> bool:
    # Title MUST contain at least one event word (strict filter)
    title_lower = title.lower()
    if not any(w in title_lower for w in TITLE_EVENT_WORDS):
        return False
    # Then also check combined text for earnings keywords
    text = (title + " " + description).lower()
    return any(kw in text for kw in EARNINGS_KEYWORDS)


def get_surprise_direction(title: str, description: str) -> str | None:
    text = (title + " " + description).lower()
    has_beat = any(kw in text for kw in BEAT_KEYWORDS)
    has_miss = any(kw in text for kw in MISS_KEYWORDS)
    if has_beat and not has_miss:
        return "beat"
    if has_miss and not has_beat:
        return "miss"
    return None


def fetch_all_news_bulk(api_key: str, since_date: str, max_pages: int = 50) -> list[dict]:
    """Fetch ALL recent news from Polygon in bulk (no ticker filter).

    Uses pagination with limit=1000 per page. Much faster than per-ticker queries.
    """
    all_articles = []
    url = (
        f"https://api.polygon.io/v2/reference/news"
        f"?published_utc.gte={since_date}"
        f"&limit=1000"
        f"&order=desc"
        f"&apiKey={api_key}"
    )

    for page in range(max_pages):
        if not url:
            break

        emit_progress("Fetching news", 10 + int(50 * page / max_pages),
                       f"Page {page + 1}, {len(all_articles)} articles so far")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TraderApp/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("results", [])
                if not results:
                    break
                all_articles.extend(results)

                # Polygon pagination: next_url contains the full URL for the next page
                next_url = data.get("next_url", "")
                if next_url:
                    # next_url doesn't include apiKey, append it
                    if "apiKey=" not in next_url:
                        sep = "&" if "?" in next_url else "?"
                        next_url = f"{next_url}{sep}apiKey={api_key}"
                    url = next_url
                else:
                    break
        except Exception as e:
            emit_progress("Fetch warning", 0, f"Page {page + 1} failed: {e}")
            break

        # Brief pause between pages to avoid rate limit
        time.sleep(0.25)

    return all_articles


def fetch_prev_close(ticker: str, api_key: str) -> float | None:
    """Get previous close price from Polygon."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={api_key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TraderApp/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            results = data.get("results", [])
            if results:
                return results[0].get("c", 0)
    except Exception:
        pass
    return None


def fetch_daily_bar(ticker: str, date_str: str, api_key: str) -> dict | None:
    """Get daily OHLCV bar for a specific date."""
    url = (
        f"https://api.polygon.io/v1/open-close/{ticker}/{date_str}"
        f"?adjusted=true&apiKey={api_key}"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TraderApp/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") == "OK":
                return data
    except Exception:
        pass
    return None


def calc_gap_for_item(item: dict, api_key: str) -> None:
    """Calculate price gap for a single beat ticker (called concurrently)."""
    prev = fetch_prev_close(item["ticker"], api_key)
    bar = fetch_daily_bar(item["ticker"], item["published"], api_key)
    if prev and bar and prev > 0:
        open_price = bar.get("open", 0)
        if open_price > 0:
            item["gap_pct"] = round((open_price - prev) / prev * 100, 2)


def main():
    parser = argparse.ArgumentParser(description="Earnings Surprise Scanner")
    parser.add_argument("--polygon-key", type=str, required=True)
    parser.add_argument("--parquet-path", type=str, required=True)
    parser.add_argument("--lookback-days", type=int, default=2,
                        help="How many days back to scan for earnings news (default=2 for T+0/T-1)")
    parser.add_argument("--max-tickers", type=int, default=0,
                        help="Limit ticker universe size (0 = no limit, for testing)")
    args = parser.parse_args()

    api_key = args.polygon_key
    parquet_path = args.parquet_path
    lookback = args.lookback_days

    # 1. Read ticker universe from parquet
    emit_progress("Reading tickers", 5, f"Loading {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path, columns=["Close"])  # Only need index
        all_tickers = sorted(df.index.get_level_values("ticker").unique().tolist())
        if args.max_tickers > 0:
            all_tickers = all_tickers[:args.max_tickers]
        ticker_universe = set(all_tickers)
    except Exception as e:
        print(json.dumps({"error": f"Failed to read parquet: {e}"}))
        sys.exit(1)

    emit_progress("Tickers loaded", 8, f"{len(ticker_universe)} tickers in universe")

    since_date = (datetime.now(tz=None) - timedelta(days=lookback)).strftime("%Y-%m-%d")

    # 2. Bulk fetch ALL news (no per-ticker filter) — ~10-20 requests vs 3200+
    all_articles = fetch_all_news_bulk(api_key, since_date)
    emit_progress("News fetched", 60, f"{len(all_articles)} total articles")

    # 3. Filter locally: match against our universe + earnings keywords
    results = {}  # ticker -> result dict (one per ticker)
    for article in all_articles:
        # Each article may have multiple tickers
        article_tickers = article.get("tickers", [])
        if not article_tickers:
            continue

        title = article.get("title", "")
        desc = article.get("description", "") or ""

        if not is_earnings_article(title, desc):
            continue

        direction = get_surprise_direction(title, desc)
        if direction is None:
            continue

        pub_date = article.get("published_utc", "")[:10]

        for ticker in article_tickers:
            ticker = ticker.upper()
            if ticker in ticker_universe and ticker not in results:
                results[ticker] = {
                    "ticker": ticker,
                    "direction": direction,
                    "title": title[:200],
                    "published": pub_date,
                    "gap_pct": 0.0,
                }

    emit_progress("Filtering done", 70, f"{len(results)} earnings surprises matched")

    # 4. Calculate price gaps for beat tickers — concurrent for speed
    beat_items = [r for r in results.values() if r["direction"] == "beat"]
    if beat_items:
        emit_progress("Calculating gaps", 75, f"{len(beat_items)} beat tickers")
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(calc_gap_for_item, item, api_key): item
                for item in beat_items
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                if done % 5 == 0:
                    pct = 75 + int(20 * done / len(beat_items))
                    emit_progress("Calculating gaps", pct, f"{done}/{len(beat_items)}")
                try:
                    future.result()
                except Exception:
                    pass

    emit_progress("Complete", 100, f"{len(beat_items)} beat results")

    # Output only beat results (sorted by gap)
    beat_items.sort(key=lambda x: x["gap_pct"], reverse=True)
    print(json.dumps(beat_items, ensure_ascii=False))


if __name__ == "__main__":
    main()
