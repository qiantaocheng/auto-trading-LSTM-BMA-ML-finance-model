#!/usr/bin/env python
"""Minervini Trend Template + News Catalyst Backtest.

3-layer stock filter with strict time-leakage prevention:
  Layer 1: Activity filter (price < $100, volume > 50K, RVOL > 1.5, daily gain > 2%)
  Layer 2: Minervini 8-criteria trend template (MA alignment, 52w range, RS > 70)
  Layer 3: Positive news catalyst from Polygon API (earnings beat, FDA, contracts, etc.)

Signal at end of day D -> Buy at Open D+1 -> Sell at Open D+6 (5 trading days).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_OHLCV_PATH = Path("D:/trade/data/raw_ohlcv/polygon_raw_ohlcv_2021_2026.parquet")
RESULT_DIR = Path("D:/trade/result/minervini_news_backtest")
NEWS_CACHE_DIR = RESULT_DIR / "news_cache"
SPY_CACHE_PATH = RESULT_DIR / "spy_daily.csv"

INITIAL_CAPITAL = 100_000.0
HOLD_DAYS = 5
COST_BPS = 10
RISK_FREE_RATE = 0.04
START_DATE = "2022-03-01"
END_DATE = "2025-12-31"

# Layer 1 thresholds
PRICE_MAX = 100.0
VOLUME_MIN = 50_000
RVOL_MIN = 1.5
DAILY_RETURN_MIN = 0.02

# Layer 2 thresholds
RS_LOOKBACK = 126  # 6 months trading days
RS_THRESHOLD = 70
SMA200_RISE_LOOKBACK = 22  # ~1 month
WEEK_52_LOOKBACK = 252

# Layer 3: positive catalyst keywords (lowercased for matching)
POSITIVE_KEYWORDS = [
    "beats expectations", "beat expectations", "beats estimates", "beat estimates",
    "topped expectations", "exceeded expectations", "better than expected",
    "earnings beat", "revenue beat",
    "raises guidance", "raised guidance", "raises outlook", "raises forecast",
    "upbeat guidance", "strong results", "record revenue", "record earnings",
    "blowout quarter", "outperformed", "surpassed estimates",
    "above expectations", "above estimates",
    "turns profitable", "achieves profitability", "first profitable quarter",
    "positive ebitda", "cash flow positive",
    "wins contract", "awarded contract", "receives order", "major order",
    "supply agreement", "partnership agreement", "strategic partnership",
    "long-term agreement",
    "government contract", "dod contract", "nasa contract",
    "grant awarded", "funding awarded",
    "launches new product", "product approval",
    "receives fda approval", "fda approval", "fda approves", "fda clearance",
    "breakthrough", "successful trial", "milestone achieved",
    "technology breakthrough",
    "phase 2 success", "phase 3 success", "clinical success",
    "trial meets endpoint",
    "share buyback", "repurchase program",
    "dividend increase", "special dividend",
    "acquisition", "to acquire", "merger", "m&a deal",
    "joint venture", "collaboration agreement", "licensing agreement",
    "to be acquired", "takeover bid", "buyout offer",
    "strategic investment", "private placement", "pipe investment",
    "analyst upgrade", "upgraded to buy", "initiated coverage with buy",
    "price target raised",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Load OHLCV data
# ---------------------------------------------------------------------------
def load_ohlcv(path: Path) -> pd.DataFrame:
    """Load raw OHLCV parquet, normalize dates, set MultiIndex [date, ticker]."""
    log("  Reading parquet...")
    df = pd.read_parquet(path)
    # Normalize date (strip 05:00:00 UTC offset)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index(["date", "ticker"]).sort_index()
    # Drop rows with zero/nan Close or Volume
    df = df[(df["Close"] > 0) & (df["Volume"] > 0)]
    log(f"  Loaded {len(df):,} rows, "
        f"{df.index.get_level_values('ticker').nunique()} tickers, "
        f"{df.index.get_level_values('date').nunique()} dates")
    return df


# ---------------------------------------------------------------------------
# 2. Fetch SPY data from Polygon API
# ---------------------------------------------------------------------------
def fetch_spy_data(api_key: str) -> pd.DataFrame:
    """Fetch SPY daily bars from Polygon API, cache locally."""
    if SPY_CACHE_PATH.exists():
        df = pd.read_csv(SPY_CACHE_PATH, parse_dates=["date"], index_col="date")
        if len(df) > 500:
            log(f"  SPY from cache: {len(df)} bars")
            return df

    log("  Fetching SPY from Polygon API...")
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2020-01-01/2026-02-01"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "TraderApp/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
        results = data.get("results", [])

    rows = []
    for bar in results:
        ts = bar.get("t", 0)
        dt = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        rows.append({
            "date": dt,
            "Open": bar.get("o", 0),
            "High": bar.get("h", 0),
            "Low": bar.get("l", 0),
            "Close": bar.get("c", 0),
            "Volume": bar.get("v", 0),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPY_CACHE_PATH)
    log(f"  SPY: {len(df)} bars cached")
    return df


# ---------------------------------------------------------------------------
# 3. Compute technical indicators (vectorized, no leakage)
# ---------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    """Add all indicator columns. All lookbacks end at day D (inclusive)."""
    log("  Computing daily returns...")
    g = df.groupby("ticker", sort=False)

    # Daily return: Close[D] / Close[D-1] - 1
    df["daily_return"] = g["Close"].pct_change()

    log("  Computing SMAs (50/150/200)...")
    df["sma50"] = g["Close"].transform(lambda x: x.rolling(50, min_periods=45).mean())
    df["sma150"] = g["Close"].transform(lambda x: x.rolling(150, min_periods=140).mean())
    df["sma200"] = g["Close"].transform(lambda x: x.rolling(200, min_periods=190).mean())

    # SMA200 from 22 trading days ago (for rising check)
    df["sma200_22d_ago"] = g["sma200"].shift(SMA200_RISE_LOOKBACK)

    log("  Computing RVOL (20d baseline, shifted to exclude day D)...")
    # CRITICAL: .shift(1) ensures day D volume is NOT in the baseline
    df["vol_20d_avg"] = g["Volume"].transform(
        lambda x: x.rolling(20, min_periods=15).mean().shift(1)
    )
    df["rvol"] = df["Volume"] / df["vol_20d_avg"]

    log("  Computing 52-week high/low...")
    df["high_52w"] = g["Close"].transform(
        lambda x: x.rolling(WEEK_52_LOOKBACK, min_periods=200).max()
    )
    df["low_52w"] = g["Close"].transform(
        lambda x: x.rolling(WEEK_52_LOOKBACK, min_periods=200).min()
    )

    log("  Computing RS score (6-month return, cross-sectional percentile)...")
    # 6-month return: Close[D] / Close[D-126] - 1
    df["close_126d_ago"] = g["Close"].shift(RS_LOOKBACK)
    df["ret_6m"] = df["Close"] / df["close_126d_ago"] - 1

    # Cross-sectional percentile rank on each date
    df["rs_score"] = df.groupby(level="date")["ret_6m"].rank(pct=True) * 100

    # Forward 5-day return for filter analysis (NOT used in signals — analysis only)
    df["fwd_5d_open"] = g["Open"].shift(-1)  # D+1 Open
    df["fwd_6d_open"] = g["Open"].shift(-6)  # D+6 Open (5 trading days after entry)
    df["fwd_return"] = df["fwd_6d_open"] / df["fwd_5d_open"] - 1

    log(f"  Indicators computed for {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# 4. NYSE trading calendar
# ---------------------------------------------------------------------------
def get_nyse_trading_days(start: str, end: str) -> list[pd.Timestamp]:
    nyse = mcal.get_calendar("XNYS")
    schedule = nyse.schedule(start_date=start, end_date=end)
    return list(schedule.index)


# ---------------------------------------------------------------------------
# 5. Polygon news fetch with month-by-month caching
# ---------------------------------------------------------------------------
def fetch_news_month(year: int, month: int, api_key: str) -> list[dict]:
    """Fetch all news for a month from Polygon, with local JSON cache."""
    cache_file = NEWS_CACHE_DIR / f"{year:04d}-{month:02d}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            log(f"    [cache] {year}-{month:02d}: {len(data)} articles")
            return data

    start = f"{year:04d}-{month:02d}-01"
    if month == 12:
        end = f"{year + 1:04d}-01-01"
    else:
        end = f"{year:04d}-{month + 1:02d}-01"

    all_articles: list[dict] = []
    url = (
        f"https://api.polygon.io/v2/reference/news"
        f"?published_utc.gte={start}"
        f"&published_utc.lt={end}"
        f"&limit=1000"
        f"&order=asc"
        f"&apiKey={api_key}"
    )

    for page in range(200):
        if not url:
            break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TraderApp/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("results", [])
                if not results:
                    break
                all_articles.extend(results)

                next_url = data.get("next_url", "")
                if next_url:
                    if "apiKey=" not in next_url:
                        sep = "&" if "?" in next_url else "?"
                        next_url = f"{next_url}{sep}apiKey={api_key}"
                    url = next_url
                else:
                    break
        except Exception as e:
            log(f"    Warning: page {page + 1} failed: {e}")
            break
        time.sleep(0.3)

    NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(all_articles, f)
    log(f"    [fetch] {year}-{month:02d}: {len(all_articles)} articles cached")
    return all_articles


def fetch_all_news(api_key: str, start_date: str, end_date: str) -> dict[str, list[dict]]:
    """Fetch news for the full backtest range, indexed by published date."""
    sd = datetime.strptime(start_date, "%Y-%m-%d")
    ed = datetime.strptime(end_date, "%Y-%m-%d")

    # Go back a few days to cover potential edge cases
    sd_fetch = sd - timedelta(days=5)

    all_articles: list[dict] = []
    current = sd_fetch
    while current <= ed:
        y, m = current.year, current.month
        log(f"  Fetching news for {y}-{m:02d}...")
        month_articles = fetch_news_month(y, m, api_key)
        all_articles.extend(month_articles)
        if m == 12:
            current = datetime(y + 1, 1, 1)
        else:
            current = datetime(y, m + 1, 1)

    # Index by published date (YYYY-MM-DD)
    news_by_date: dict[str, list[dict]] = defaultdict(list)
    for article in all_articles:
        pub = article.get("published_utc", "")
        pub_date = pub[:10]
        if pub_date:
            # Only keep articles published before market close (21:00 UTC ~ 4-5 PM ET)
            pub_time = pub[11:19] if len(pub) > 19 else "00:00:00"
            if pub_time <= "21:00:00":
                news_by_date[pub_date].append(article)

    log(f"  Total: {sum(len(v) for v in news_by_date.values())} articles "
        f"across {len(news_by_date)} dates (pre-close only)")
    return news_by_date


# ---------------------------------------------------------------------------
# 6. Signal generation: 3-layer filtering
# ---------------------------------------------------------------------------
def apply_layer1(df: pd.DataFrame) -> pd.Series:
    """Layer 1: Activity filter. Returns boolean mask."""
    return (
        (df["Close"] < PRICE_MAX) &
        (df["Volume"] > VOLUME_MIN) &
        (df["rvol"] > RVOL_MIN) &
        (df["daily_return"] > DAILY_RETURN_MIN)
    )


def apply_layer2(df: pd.DataFrame) -> pd.Series:
    """Layer 2: Minervini 8-criteria trend template. Returns boolean mask."""
    c1 = (df["Close"] > df["sma150"]) & (df["Close"] > df["sma200"])
    c2 = df["sma150"] > df["sma200"]
    c3 = df["sma200"] > df["sma200_22d_ago"]
    c4 = (df["sma50"] > df["sma150"]) & (df["sma50"] > df["sma200"])
    c5 = df["Close"] > df["sma50"]
    c6 = df["Close"] > (1.3 * df["low_52w"])
    c7 = df["Close"] > (0.75 * df["high_52w"])
    c8 = df["rs_score"] > RS_THRESHOLD
    return c1 & c2 & c3 & c4 & c5 & c6 & c7 & c8


def has_positive_catalyst(ticker: str, articles: list[dict]) -> tuple[bool, str]:
    """Check if any article on this date mentions the ticker with a positive keyword."""
    for article in articles:
        article_tickers = [t.upper() for t in article.get("tickers", [])]
        if ticker.upper() not in article_tickers:
            continue
        title = (article.get("title", "") or "").lower()
        desc = (article.get("description", "") or "").lower()
        combined = title + " " + desc
        for kw in POSITIVE_KEYWORDS:
            if kw in combined:
                return True, (article.get("title", "") or "")[:200]
    return False, ""


def generate_signals(
    df: pd.DataFrame,
    news_by_date: dict[str, list[dict]],
    trading_days: list[pd.Timestamp],
    warmup_end: pd.Timestamp,
) -> tuple[list[dict], dict]:
    """Generate signals for each trading day, applying all 3 filter layers.

    Returns (signals, filter_stats) where filter_stats tracks counts and
    average forward returns at each filter stage for analysis.
    """
    signals: list[dict] = []
    filter_stats = {
        "universe": {"count": 0, "fwd_returns": []},
        "layer1": {"count": 0, "fwd_returns": []},
        "layer2": {"count": 0, "fwd_returns": []},
        "layer3": {"count": 0, "fwd_returns": []},
    }

    # Pre-compute layer masks on entire df (vectorized)
    log("  Computing Layer 1 & 2 masks (vectorized)...")
    mask_l1 = apply_layer1(df)
    mask_l2 = apply_layer2(df)

    # Get available dates in data
    available_dates = set(df.index.get_level_values("date").unique())

    n_days = 0
    for td in trading_days:
        td_norm = pd.Timestamp(td).normalize()
        if td_norm <= warmup_end or td_norm not in available_dates:
            continue

        date_str = td_norm.strftime("%Y-%m-%d")
        n_days += 1

        try:
            day_df = df.loc[td_norm]
        except KeyError:
            continue

        n_universe = len(day_df)
        filter_stats["universe"]["count"] += n_universe

        # Layer 1
        l1_mask = mask_l1.loc[td_norm] if td_norm in mask_l1.index.get_level_values(0) else pd.Series(dtype=bool)
        if isinstance(l1_mask, pd.Series):
            l1_tickers = l1_mask[l1_mask].index.tolist()
        else:
            l1_tickers = []

        filter_stats["layer1"]["count"] += len(l1_tickers)

        # Collect fwd returns for Layer 1 passers
        for t in l1_tickers:
            try:
                fwd = day_df.loc[t, "fwd_return"]
                if pd.notna(fwd):
                    filter_stats["layer1"]["fwd_returns"].append(fwd)
            except (KeyError, TypeError):
                pass

        # Layer 2 (only among Layer 1 passers)
        l2_tickers = []
        for t in l1_tickers:
            try:
                if mask_l2.loc[(td_norm, t)]:
                    l2_tickers.append(t)
            except (KeyError, TypeError):
                continue

        filter_stats["layer2"]["count"] += len(l2_tickers)
        for t in l2_tickers:
            try:
                fwd = day_df.loc[t, "fwd_return"]
                if pd.notna(fwd):
                    filter_stats["layer2"]["fwd_returns"].append(fwd)
            except (KeyError, TypeError):
                pass

        # Layer 3: news catalyst (only among Layer 2 passers)
        articles = news_by_date.get(date_str, [])
        for t in l2_tickers:
            has_news, headline = has_positive_catalyst(t, articles)
            if has_news:
                filter_stats["layer3"]["count"] += 1
                try:
                    fwd = day_df.loc[t, "fwd_return"]
                    if pd.notna(fwd):
                        filter_stats["layer3"]["fwd_returns"].append(fwd)
                except (KeyError, TypeError):
                    pass

                signals.append({
                    "signal_date": date_str,
                    "ticker": t,
                    "headline": headline,
                })

        if n_days % 100 == 0:
            log(f"    Processed {n_days} days, {len(signals)} signals so far...")

    log(f"  Done: {n_days} days scanned, {len(signals)} total signals")
    return signals, filter_stats


# ---------------------------------------------------------------------------
# 7. Backtest engine
# ---------------------------------------------------------------------------
def run_backtest(
    signals: list[dict],
    df: pd.DataFrame,
    trading_days: list[pd.Timestamp],
    hold_days: int = HOLD_DAYS,
    cost_bps: int = COST_BPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fixed-hold backtest. Entry at D+1 Open, exit at D+6 Open."""
    # Build price lookup: {(date_str, ticker): {"Open": ..., "Close": ...}}
    log("  Building price lookup...")
    price_lookup: dict[tuple[str, str], dict] = {}
    for (dt, ticker), row in df[["Open", "Close"]].iterrows():
        ds = dt.strftime("%Y-%m-%d")
        price_lookup[(ds, ticker)] = {"Open": row["Open"], "Close": row["Close"]}

    # Trading day index for offset computation
    td_strs = [td.strftime("%Y-%m-%d") for td in trading_days]
    td_idx = {ds: i for i, ds in enumerate(td_strs)}

    # Group signals by signal_date
    signals_by_date: dict[str, list[dict]] = defaultdict(list)
    for sig in signals:
        signals_by_date[sig["signal_date"]].append(sig)

    open_positions: list[dict] = []
    completed_trades: list[dict] = []
    cash = INITIAL_CAPITAL
    cost_mult = cost_bps / 10_000.0
    daily_equity: list[dict] = []
    open_tickers: set[str] = set()

    for day_str in td_strs:
        # 1. Check exits — sell at Open on exit date
        still_open = []
        for pos in open_positions:
            if day_str >= pos["exit_date"]:
                bar = price_lookup.get((day_str, pos["ticker"]))
                if bar and bar["Open"] > 0:
                    exit_price = bar["Open"]
                    sell_proceeds = pos["shares"] * exit_price * (1 - cost_mult)
                    pnl = sell_proceeds - pos["dollars"]
                    ret = pnl / pos["dollars"] if pos["dollars"] > 0 else 0
                    completed_trades.append({
                        "ticker": pos["ticker"],
                        "signal_date": pos["signal_date"],
                        "entry_date": pos["entry_date"],
                        "exit_date": day_str,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "shares": pos["shares"],
                        "dollars_in": pos["dollars"],
                        "dollars_out": sell_proceeds,
                        "pnl": pnl,
                        "return_pct": ret * 100,
                        "headline": pos["headline"],
                    })
                    cash += sell_proceeds
                    open_tickers.discard(pos["ticker"])
                else:
                    still_open.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2. New entries — signals from previous day, buy at today's Open
        # Find yesterday's signal date
        idx = td_idx.get(day_str)
        if idx is not None and idx > 0:
            prev_day = td_strs[idx - 1]
            day_signals = signals_by_date.get(prev_day, [])

            # Filter out tickers with open positions (no-overlap rule)
            valid_signals = []
            for sig in day_signals:
                ticker = sig["ticker"]
                if ticker in open_tickers:
                    continue
                bar = price_lookup.get((day_str, ticker))
                if bar and bar["Open"] > 0:
                    valid_signals.append((sig, bar["Open"]))

            if valid_signals and cash > 100:
                per_stock = cash / len(valid_signals)
                if per_stock > 50:
                    # Compute exit date: hold_days trading days after entry
                    exit_idx = idx + hold_days
                    if exit_idx < len(td_strs):
                        exit_date_str = td_strs[exit_idx]
                    else:
                        exit_date_str = td_strs[-1]

                    for sig, open_price in valid_signals:
                        buy_cost = per_stock * (1 + cost_mult)
                        if buy_cost > cash:
                            buy_cost = cash
                        shares = (buy_cost / (1 + cost_mult)) / open_price
                        open_positions.append({
                            "ticker": sig["ticker"],
                            "signal_date": sig["signal_date"],
                            "entry_date": day_str,
                            "entry_price": open_price,
                            "shares": shares,
                            "exit_date": exit_date_str,
                            "dollars": buy_cost,
                            "headline": sig.get("headline", ""),
                        })
                        cash -= buy_cost
                        open_tickers.add(sig["ticker"])

        # 3. Daily mark-to-market
        position_value = 0.0
        for pos in open_positions:
            bar = price_lookup.get((day_str, pos["ticker"]))
            if bar:
                position_value += pos["shares"] * bar["Close"]
            else:
                position_value += pos["shares"] * pos["entry_price"]

        daily_equity.append({
            "date": day_str,
            "equity": cash + position_value,
            "cash": cash,
            "position_value": position_value,
            "n_positions": len(open_positions),
        })

    trades_df = pd.DataFrame(completed_trades)
    equity_df = pd.DataFrame(daily_equity)
    if not equity_df.empty:
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df = equity_df.set_index("date")

    return trades_df, equity_df


# ---------------------------------------------------------------------------
# Metrics & analytics
# ---------------------------------------------------------------------------
def compute_metrics(equity_series: pd.Series, label: str = "Strategy") -> dict:
    returns = equity_series.pct_change().dropna()
    n_days = len(returns)
    ann_factor = 252

    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    years = n_days / ann_factor
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    excess_returns = returns - RISK_FREE_RATE / ann_factor
    sharpe = (excess_returns.mean() / returns.std() * np.sqrt(ann_factor)) if returns.std() > 0 else 0

    downside = returns[returns < 0]
    sortino = (excess_returns.mean() / downside.std() * np.sqrt(ann_factor)) if len(downside) > 0 and downside.std() > 0 else 0

    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        "label": label,
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd_pct": max_dd * 100,
        "calmar": calmar,
        "ann_vol_pct": returns.std() * np.sqrt(ann_factor) * 100,
        "n_days": n_days,
    }


def annual_breakdown(trades_df: pd.DataFrame) -> str:
    if trades_df.empty:
        return "  No trades"
    df = trades_df.copy()
    df["year"] = pd.to_datetime(df["entry_date"]).dt.year
    lines = []
    for year, group in df.groupby("year"):
        n = len(group)
        wins = (group["pnl"] > 0).sum()
        avg_ret = group["return_pct"].mean()
        total_pnl = group["pnl"].sum()
        win_rate = wins / n * 100 if n > 0 else 0
        lines.append(
            f"  {year}: {n:4d} trades, win={win_rate:5.1f}%, "
            f"avg_ret={avg_ret:+6.2f}%, total_pnl=${total_pnl:+,.0f}"
        )
    return "\n".join(lines)


def format_filter_analysis(filter_stats: dict, n_days: int) -> str:
    """Format filter contribution analysis."""
    lines = ["Filter Contribution Analysis", "=" * 50]

    for stage, label in [
        ("layer1", "Layer 1 (Activity: Price/Vol/RVOL/Return)"),
        ("layer2", "Layer 2 (Minervini 8-Criteria Trend)"),
        ("layer3", "Layer 3 (Positive News Catalyst)"),
    ]:
        stats = filter_stats[stage]
        count = stats["count"]
        fwd = stats["fwd_returns"]
        avg_per_day = count / n_days if n_days > 0 else 0
        avg_fwd = np.mean(fwd) * 100 if fwd else float("nan")
        med_fwd = np.median(fwd) * 100 if fwd else float("nan")
        win_rate = (np.sum(np.array(fwd) > 0) / len(fwd) * 100) if fwd else float("nan")

        lines.append(f"\n{label}:")
        lines.append(f"  Total passers:   {count:,}")
        lines.append(f"  Avg per day:     {avg_per_day:.1f}")
        lines.append(f"  Avg fwd 5d ret:  {avg_fwd:+.2f}%")
        lines.append(f"  Med fwd 5d ret:  {med_fwd:+.2f}%")
        lines.append(f"  Win rate (5d):   {win_rate:.1f}%")
        lines.append(f"  N observations:  {len(fwd)}")

    return "\n".join(lines)


def compute_spy_benchmark(spy_df: pd.DataFrame, equity_df: pd.DataFrame) -> pd.Series | None:
    """Compute SPY buy-and-hold equity curve aligned to backtest dates."""
    if spy_df is None or spy_df.empty:
        return None

    first_date = equity_df.index[0]
    spy_df_norm = spy_df.copy()
    spy_df_norm.index = spy_df_norm.index.normalize()

    if first_date not in spy_df_norm.index:
        # Find nearest date
        mask = spy_df_norm.index >= first_date
        if not mask.any():
            return None
        first_date = spy_df_norm.index[mask][0]

    spy_entry_price = spy_df_norm.loc[first_date, "Open"]
    spy_shares = INITIAL_CAPITAL / spy_entry_price

    spy_equity = []
    last_val = INITIAL_CAPITAL
    for date in equity_df.index:
        date_norm = date.normalize()
        if date_norm in spy_df_norm.index:
            val = spy_shares * spy_df_norm.loc[date_norm, "Close"]
            last_val = val
        spy_equity.append(last_val)

    return pd.Series(spy_equity, index=equity_df.index, name="SPY B&H")


def plot_results(
    equity_df: pd.DataFrame,
    spy_equity: pd.Series | None,
    metrics: dict,
    spy_metrics: dict | None,
    trades_df: pd.DataFrame,
    start_date: str,
    end_date: str,
):
    """Generate and save equity curve and positions plots."""
    # Equity curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1],
                                    gridspec_kw={"hspace": 0.3})

    ax1.plot(equity_df.index, equity_df["equity"],
             label=f"Strategy (Sharpe={metrics['sharpe']:.2f})",
             linewidth=1.5, color="blue")
    if spy_equity is not None and spy_metrics is not None:
        ax1.plot(spy_equity.index, spy_equity.values,
                 label=f"SPY B&H (Sharpe={spy_metrics['sharpe']:.2f})",
                 linewidth=1.5, color="gray", alpha=0.7)

    ax1.set_title(
        f"Minervini Trend + News Catalyst Backtest ({start_date} -> {end_date})\n"
        f"CAGR={metrics['cagr_pct']:+.2f}%, Sharpe={metrics['sharpe']:.3f}, "
        f"MaxDD={metrics['max_dd_pct']:.2f}%",
        fontsize=12,
    )
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Drawdown
    eq = equity_df["equity"]
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    ax2.fill_between(equity_df.index, dd, 0, alpha=0.4, color="red")
    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("DD %")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = RESULT_DIR / "equity_curve.png"
    fig.savefig(plot_path, dpi=150)
    log(f"  Saved: {plot_path}")
    plt.close(fig)

    # Positions over time
    fig2, ax3 = plt.subplots(figsize=(14, 4))
    ax3.fill_between(equity_df.index, equity_df["n_positions"], alpha=0.4, color="orange")
    ax3.set_title("Open Positions Over Time")
    ax3.set_ylabel("# Positions")
    ax3.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(RESULT_DIR / "positions.png", dpi=150)
    plt.close(fig2)

    # Monthly returns heatmap
    if not trades_df.empty:
        df_t = trades_df.copy()
        df_t["entry_dt"] = pd.to_datetime(df_t["entry_date"])
        df_t["year"] = df_t["entry_dt"].dt.year
        df_t["month"] = df_t["entry_dt"].dt.month

        pivot = df_t.groupby(["year", "month"])["return_pct"].mean().unstack(fill_value=0)

        fig3, ax4 = plt.subplots(figsize=(14, 6))
        im = ax4.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                        vmin=-5, vmax=5)
        ax4.set_xticks(range(len(pivot.columns)))
        ax4.set_xticklabels([f"{m:02d}" for m in pivot.columns])
        ax4.set_yticks(range(len(pivot.index)))
        ax4.set_yticklabels(pivot.index)
        ax4.set_xlabel("Month")
        ax4.set_ylabel("Year")
        ax4.set_title("Average Trade Return (%) by Month")
        plt.colorbar(im, ax=ax4, label="Avg Return %")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if val != 0:
                    ax4.text(j, i, f"{val:.1f}", ha="center", va="center",
                             fontsize=8, color="black")

        fig3.tight_layout()
        fig3.savefig(RESULT_DIR / "monthly_returns.png", dpi=150)
        plt.close(fig3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Minervini Trend + News Catalyst Backtest")
    parser.add_argument("--polygon-key", type=str, required=True)
    parser.add_argument("--start", type=str, default=START_DATE)
    parser.add_argument("--end", type=str, default=END_DATE)
    parser.add_argument("--hold-days", type=int, default=HOLD_DAYS)
    parser.add_argument("--cost-bps", type=int, default=COST_BPS)
    args = parser.parse_args()

    hold_days = args.hold_days
    cost_bps = args.cost_bps
    api_key = args.polygon_key
    start_date = args.start
    end_date = args.end

    log("=" * 60)
    log("Minervini Trend + News Catalyst Backtest")
    log(f"Period: {start_date} -> {end_date}")
    log(f"Capital: ${INITIAL_CAPITAL:,.0f}")
    log(f"Hold: {hold_days} trading days, Cost: {cost_bps} bps")
    log("=" * 60)

    # [1/7] Load OHLCV
    log("\n[1/7] Loading raw OHLCV data...")
    ohlcv = load_ohlcv(RAW_OHLCV_PATH)

    # [2/7] Fetch SPY
    log("\n[2/7] Fetching SPY benchmark data...")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    spy_df = fetch_spy_data(api_key)

    # [3/7] Compute indicators
    log("\n[3/7] Computing technical indicators (vectorized)...")
    ohlcv = compute_indicators(ohlcv, spy_df["Close"])

    # Drop warmup rows (need 252 days for 52w lookback)
    all_dates = sorted(ohlcv.index.get_level_values("date").unique())
    if len(all_dates) > WEEK_52_LOOKBACK:
        warmup_end = all_dates[WEEK_52_LOOKBACK]
    else:
        warmup_end = all_dates[-1]
    log(f"  Warmup period ends at: {warmup_end.strftime('%Y-%m-%d')}")

    # [4/7] NYSE calendar
    log("\n[4/7] Building NYSE trading calendar...")
    trading_days = get_nyse_trading_days(start_date, end_date)
    log(f"  {len(trading_days)} trading days in [{start_date}, {end_date}]")

    # [5/7] Fetch news
    log("\n[5/7] Fetching news from Polygon (month-by-month caching)...")
    news_by_date = fetch_all_news(api_key, start_date, end_date)

    # [6/7] Generate signals
    log("\n[6/7] Generating signals (Layer 1 -> Layer 2 -> Layer 3)...")
    signals, filter_stats = generate_signals(ohlcv, news_by_date, trading_days, warmup_end)
    log(f"  Total signals: {len(signals)}")

    if not signals:
        log("\nERROR: No signals generated. Check data/filters.")
        sys.exit(1)

    # Show signal count by month
    sig_by_month: dict[str, int] = defaultdict(int)
    for s in signals:
        sig_by_month[s["signal_date"][:7]] += 1
    for ym in sorted(sig_by_month):
        log(f"    {ym}: {sig_by_month[ym]} signals")

    # [7/7] Run backtest
    log("\n[7/7] Running backtest simulation...")
    trades_df, equity_df = run_backtest(signals, ohlcv, trading_days,
                                        hold_days=hold_days, cost_bps=cost_bps)

    if equity_df.empty:
        log("ERROR: No equity data generated")
        sys.exit(1)

    # Compute metrics
    metrics = compute_metrics(equity_df["equity"], "Minervini+News")
    spy_equity = compute_spy_benchmark(spy_df, equity_df)
    spy_metrics = compute_metrics(spy_equity, "SPY B&H") if spy_equity is not None else None

    # -----------------------------------------------------------------------
    # Output results
    # -----------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("RESULTS")
    log("=" * 60)

    summary = []
    summary.append(f"Backtest Period:   {start_date} -> {end_date}")
    summary.append(f"Initial Capital:   ${INITIAL_CAPITAL:,.0f}")
    summary.append(f"Hold Period:       {hold_days} trading days")
    summary.append(f"Transaction Cost:  {cost_bps} bps round-trip")
    summary.append(f"Trading Days:      {len(trading_days)}")
    summary.append(f"Total Signals:     {len(signals)}")
    summary.append(f"Completed Trades:  {len(trades_df)}")
    summary.append("")

    def fmt(m: dict) -> str:
        return (
            f"  CAGR:     {m['cagr_pct']:+.2f}%\n"
            f"  Sharpe:   {m['sharpe']:.3f}  (rf={RISK_FREE_RATE:.0%})\n"
            f"  Sortino:  {m['sortino']:.3f}\n"
            f"  MaxDD:    {m['max_dd_pct']:.2f}%\n"
            f"  Calmar:   {m['calmar']:.3f}\n"
            f"  Ann Vol:  {m['ann_vol_pct']:.2f}%\n"
            f"  Total:    {m['total_return_pct']:+.2f}%"
        )

    summary.append(f"--- {metrics['label']} ---")
    summary.append(fmt(metrics))
    summary.append("")

    if spy_metrics:
        summary.append(f"--- {spy_metrics['label']} ---")
        summary.append(fmt(spy_metrics))
        summary.append("")

    if not trades_df.empty:
        wins = (trades_df["pnl"] > 0).sum()
        total = len(trades_df)
        summary.append(f"Win Rate:       {wins}/{total} = {wins / total * 100:.1f}%")
        summary.append(f"Avg Return:     {trades_df['return_pct'].mean():+.2f}%")
        summary.append(f"Median Return:  {trades_df['return_pct'].median():+.2f}%")
        summary.append(f"Best Trade:     {trades_df['return_pct'].max():+.2f}%")
        summary.append(f"Worst Trade:    {trades_df['return_pct'].min():+.2f}%")
        summary.append(f"Total P&L:      ${trades_df['pnl'].sum():+,.0f}")
        summary.append("")
        summary.append("Annual Breakdown:")
        summary.append(annual_breakdown(trades_df))

    summary_text = "\n".join(summary)
    print(summary_text)

    # Compute backtest days for filter analysis
    effective_days = len([td for td in trading_days if pd.Timestamp(td).normalize() > warmup_end])
    filter_text = format_filter_analysis(filter_stats, effective_days)
    print(f"\n{filter_text}")

    # Save files
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULT_DIR / "summary.txt", "w") as f:
        f.write(summary_text)
    log(f"\nSaved: {RESULT_DIR / 'summary.txt'}")

    with open(RESULT_DIR / "filter_analysis.txt", "w") as f:
        f.write(filter_text)
    log(f"Saved: {RESULT_DIR / 'filter_analysis.txt'}")

    if not trades_df.empty:
        trades_df.to_csv(RESULT_DIR / "trades.csv", index=False)
        log(f"Saved: {RESULT_DIR / 'trades.csv'}")

    # Plots
    plot_results(equity_df, spy_equity, metrics, spy_metrics, trades_df, start_date, end_date)

    log("\nDone!")


if __name__ == "__main__":
    main()
