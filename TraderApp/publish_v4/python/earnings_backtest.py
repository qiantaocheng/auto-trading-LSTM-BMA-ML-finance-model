#!/usr/bin/env python
"""3-Year Earnings Beat Strategy Backtest.

Uses the EXACT same filtering logic as earnings_scanner.py:
- Same keyword lists (BEAT_KEYWORDS, MISS_KEYWORDS, TITLE_EVENT_WORDS, EARNINGS_KEYWORDS)
- Same is_earnings_article() and get_surprise_direction() functions
- T-1 news → T open buy, T+10 trading days close sell

Data sourced from Polygon.io with month-by-month caching.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

# ---------------------------------------------------------------------------
# Import filtering logic from earnings_scanner.py (exact same keywords/funcs)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from earnings_scanner import (
    BEAT_KEYWORDS,
    MISS_KEYWORDS,
    TITLE_EVENT_WORDS,
    EARNINGS_KEYWORDS,
    is_earnings_article,
    get_surprise_direction,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULT_DIR = SCRIPT_DIR.parent / "result"
NEWS_CACHE_DIR = RESULT_DIR / "news_cache"
PRICE_CACHE_DIR = RESULT_DIR / "price_cache"
INITIAL_CAPITAL = 100_000.0
EARNINGS_ALLOC = 0.10  # 10% of capital
ENTRY_DELAY = 0  # skip first N trading days, buy at T+ENTRY_DELAY open
HOLD_DAYS = 10  # hold for N trading days after entry (exit at T+ENTRY_DELAY+HOLD_DAYS close)
COST_BPS = 10  # 10 bps round-trip
RISK_FREE_RATE = 0.04
START_DATE = "2023-02-10"
END_DATE = "2026-02-10"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# NYSE trading calendar
# ---------------------------------------------------------------------------
def get_nyse_trading_days(start: str, end: str) -> list[pd.Timestamp]:
    nyse = mcal.get_calendar("XNYS")
    schedule = nyse.schedule(start_date=start, end_date=end)
    return list(schedule.index)


# ---------------------------------------------------------------------------
# Polygon news fetch with month-by-month caching
# ---------------------------------------------------------------------------
def fetch_news_month(year: int, month: int, api_key: str) -> list[dict]:
    """Fetch all news for a given month from Polygon, with local JSON cache."""
    cache_file = NEWS_CACHE_DIR / f"{year:04d}-{month:02d}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            log(f"    [cache] {len(data)} articles from cache")
            return data

    # Date range for this month
    start = f"{year:04d}-{month:02d}-01"
    if month == 12:
        end = f"{year + 1:04d}-01-01"
    else:
        end = f"{year:04d}-{month + 1:02d}-01"

    all_articles = []
    url = (
        f"https://api.polygon.io/v2/reference/news"
        f"?published_utc.gte={start}"
        f"&published_utc.lt={end}"
        f"&limit=1000"
        f"&order=asc"
        f"&apiKey={api_key}"
    )

    for page in range(200):  # safety limit
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
                log(f"    page {page + 1}: +{len(results)} = {len(all_articles)} total")

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
        time.sleep(0.3)  # rate limit padding

    # Cache to disk
    NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(all_articles, f)
    log(f"    -> cached {len(all_articles)} articles")

    return all_articles


def fetch_all_news(api_key: str, start_date: str, end_date: str) -> dict[str, list[dict]]:
    """Fetch news for entire date range, organized by published date."""
    sd = datetime.strptime(start_date, "%Y-%m-%d")
    ed = datetime.strptime(end_date, "%Y-%m-%d")

    # We need news from T-1 before start_date, so go back a few days
    sd_fetch = sd - timedelta(days=5)

    all_articles = []
    current = sd_fetch
    while current <= ed:
        y, m = current.year, current.month
        log(f"  Fetching news for {y}-{m:02d}...")
        month_articles = fetch_news_month(y, m, api_key)
        all_articles.extend(month_articles)

        # Next month
        if m == 12:
            current = datetime(y + 1, 1, 1)
        else:
            current = datetime(y, m + 1, 1)

    # Index by published date
    news_by_date: dict[str, list[dict]] = defaultdict(list)
    for article in all_articles:
        pub_date = article.get("published_utc", "")[:10]
        if pub_date:
            news_by_date[pub_date].append(article)

    log(f"Total: {len(all_articles)} articles across {len(news_by_date)} dates")
    return news_by_date


# ---------------------------------------------------------------------------
# Polygon price fetch with per-ticker caching
# ---------------------------------------------------------------------------
def fetch_ticker_prices(ticker: str, api_key: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch daily OHLCV for a ticker from Polygon, with CSV cache."""
    cache_file = PRICE_CACHE_DIR / f"{ticker}.csv"
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, parse_dates=["date"], index_col="date")
            if len(df) > 0:
                return df
        except Exception:
            pass

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TraderApp/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            results = data.get("results", [])
            if not results:
                return None

            rows = []
            for bar in results:
                ts = bar.get("t", 0)
                dt = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
                rows.append({
                    "date": dt,
                    "open": bar.get("o", 0),
                    "high": bar.get("h", 0),
                    "low": bar.get("l", 0),
                    "close": bar.get("c", 0),
                    "volume": bar.get("v", 0),
                })

            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

            PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_file)
            return df
    except Exception:
        return None


def fetch_prices_batch(tickers: list[str], api_key: str, start: str, end: str) -> dict[str, pd.DataFrame]:
    """Fetch prices for multiple tickers concurrently."""
    # Check which tickers already have cache
    need_fetch = []
    cached = {}
    for t in tickers:
        cache_file = PRICE_CACHE_DIR / f"{t}.csv"
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, parse_dates=["date"], index_col="date")
                if len(df) > 0:
                    cached[t] = df
                    continue
            except Exception:
                pass
        need_fetch.append(t)

    log(f"  {len(cached)} tickers from cache, {len(need_fetch)} need API fetch")

    prices = dict(cached)
    if not need_fetch:
        return prices

    total = len(need_fetch)
    done = 0

    def _fetch(t):
        return t, fetch_ticker_prices(t, api_key, start, end)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch, t): t for t in need_fetch}
        for future in as_completed(futures):
            done += 1
            if done % 25 == 0 or done == total:
                log(f"  Prices: {done}/{total} fetched")
            try:
                ticker, df = future.result()
                if df is not None and len(df) > 0:
                    prices[ticker] = df
            except Exception:
                pass
            time.sleep(0.1)

    return prices


# ---------------------------------------------------------------------------
# Signal generation: scan news for earnings beats
# ---------------------------------------------------------------------------
def generate_signals(
    news_by_date: dict[str, list[dict]],
    trading_days: list[pd.Timestamp],
    ticker_universe: set[str],
) -> list[dict]:
    """For each trading day T, look at T-1 news for earnings beats."""
    signals = []

    for trade_day in trading_days:
        trade_date_str = trade_day.strftime("%Y-%m-%d")

        # T-1 = previous calendar day
        t_minus_1 = (trade_day - timedelta(days=1)).strftime("%Y-%m-%d")

        articles = news_by_date.get(t_minus_1, [])
        if not articles:
            continue

        seen_tickers = set()
        for article in articles:
            article_tickers = article.get("tickers", [])
            if not article_tickers:
                continue

            title = article.get("title", "")
            desc = article.get("description", "") or ""

            if not is_earnings_article(title, desc):
                continue

            direction = get_surprise_direction(title, desc)
            if direction != "beat":
                continue

            for ticker in article_tickers:
                ticker = ticker.upper()
                if ticker in ticker_universe and ticker not in seen_tickers:
                    seen_tickers.add(ticker)
                    signals.append({
                        "trade_date": trade_date_str,
                        "ticker": ticker,
                        "news_date": t_minus_1,
                        "title": title[:200],
                    })

    return signals


# ---------------------------------------------------------------------------
# Backtest simulation
# ---------------------------------------------------------------------------
def run_backtest(
    signals: list[dict],
    prices: dict[str, pd.DataFrame],
    trading_days: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the backtest simulation."""
    # Pre-compute price lookup as dict of {ticker: {date_str: row}}
    price_lookup: dict[str, dict[str, dict]] = {}
    for ticker, df in prices.items():
        lookup = {}
        for idx, row in df.iterrows():
            ds = idx.strftime("%Y-%m-%d")
            lookup[ds] = {"open": row["open"], "close": row["close"]}
        price_lookup[ticker] = lookup

    # Group signals by trade date (original signal day)
    signals_by_date = defaultdict(list)
    for sig in signals:
        signals_by_date[sig["trade_date"]].append(sig)

    # Pre-compute: for each signal, schedule delayed entry at T+ENTRY_DELAY
    # pending_entries: {entry_date_str: [(sig, ...)]}
    pending_entries: dict[str, list[dict]] = defaultdict(list)
    td_strs = [td.strftime("%Y-%m-%d") for td in trading_days]
    td_to_idx = {s: i for i, s in enumerate(td_strs)}

    for sig_date_str, sigs in signals_by_date.items():
        sig_idx = td_to_idx.get(sig_date_str)
        if sig_idx is None:
            continue
        entry_idx = sig_idx + ENTRY_DELAY
        if entry_idx >= len(trading_days):
            continue
        entry_date_str = td_strs[entry_idx]
        for sig in sigs:
            pending_entries[entry_date_str].append(sig)

    open_positions = []
    completed_trades = []
    earnings_cash = INITIAL_CAPITAL * EARNINGS_ALLOC
    cost_mult = COST_BPS / 10000.0

    daily_equity = []

    for day_idx, trade_day in enumerate(trading_days):
        date_str = trade_day.strftime("%Y-%m-%d")

        # 1. Check for exits — sell at close on exit date
        still_open = []
        for pos in open_positions:
            if date_str >= pos["exit_date"]:
                lk = price_lookup.get(pos["ticker"], {})
                bar = lk.get(date_str)
                if bar:
                    exit_price = bar["close"]
                    sell_proceeds = pos["shares"] * exit_price * (1 - cost_mult)
                    pnl = sell_proceeds - pos["dollars"]
                    ret = pnl / pos["dollars"] if pos["dollars"] > 0 else 0

                    completed_trades.append({
                        "ticker": pos["ticker"],
                        "entry_date": pos["entry_date"],
                        "exit_date": date_str,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "shares": pos["shares"],
                        "dollars_in": pos["dollars"],
                        "dollars_out": sell_proceeds,
                        "pnl": pnl,
                        "return_pct": ret * 100,
                    })
                    earnings_cash += sell_proceeds
                else:
                    still_open.append(pos)
                    continue
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2. Delayed entries — buy at open on T+ENTRY_DELAY
        day_entries = pending_entries.get(date_str, [])
        valid_signals = []
        seen = set()
        for sig in day_entries:
            ticker = sig["ticker"]
            if ticker in seen:
                continue
            seen.add(ticker)
            lk = price_lookup.get(ticker, {})
            bar = lk.get(date_str)
            if bar and bar["open"] > 0:
                valid_signals.append((sig, bar["open"]))

        if valid_signals and earnings_cash > 100:
            per_stock = earnings_cash / len(valid_signals)
            if per_stock > 50:
                exit_idx = day_idx + HOLD_DAYS
                if exit_idx < len(trading_days):
                    exit_td = trading_days[exit_idx]
                else:
                    exit_td = trading_days[-1]
                exit_date_str = exit_td.strftime("%Y-%m-%d")

                for sig, open_price in valid_signals:
                    buy_cost = per_stock * (1 + cost_mult)
                    shares = per_stock / open_price
                    open_positions.append({
                        "ticker": sig["ticker"],
                        "entry_date": date_str,
                        "entry_price": open_price,
                        "shares": shares,
                        "exit_date": exit_date_str,
                        "dollars": buy_cost,
                    })
                    earnings_cash -= buy_cost

        # 3. Daily mark-to-market
        position_value = 0.0
        for pos in open_positions:
            lk = price_lookup.get(pos["ticker"], {})
            bar = lk.get(date_str)
            if bar:
                position_value += pos["shares"] * bar["close"]
            else:
                position_value += pos["shares"] * pos["entry_price"]

        total_equity = (INITIAL_CAPITAL * (1 - EARNINGS_ALLOC)) + earnings_cash + position_value
        daily_equity.append({
            "date": date_str,
            "equity": total_equity,
            "earnings_cash": earnings_cash,
            "position_value": position_value,
            "n_positions": len(open_positions),
        })

    trades_df = pd.DataFrame(completed_trades)
    equity_df = pd.DataFrame(daily_equity)
    equity_df["date"] = pd.to_datetime(equity_df["date"])
    equity_df = equity_df.set_index("date")

    return trades_df, equity_df


# ---------------------------------------------------------------------------
# SPY benchmark
# ---------------------------------------------------------------------------
def compute_spy_benchmark(prices: dict[str, pd.DataFrame], equity_df: pd.DataFrame) -> pd.Series | None:
    spy_df = prices.get("SPY")
    if spy_df is None:
        return None

    first_date = equity_df.index[0].strftime("%Y-%m-%d")
    spy_lookup = {}
    for idx, row in spy_df.iterrows():
        spy_lookup[idx.strftime("%Y-%m-%d")] = row

    spy_first = spy_lookup.get(first_date)
    if spy_first is None:
        return None

    spy_entry = spy_first["open"]
    spy_shares = (INITIAL_CAPITAL * EARNINGS_ALLOC) / spy_entry
    base_capital = INITIAL_CAPITAL * (1 - EARNINGS_ALLOC)

    spy_equity = []
    last_val = INITIAL_CAPITAL
    for date in equity_df.index:
        date_str = date.strftime("%Y-%m-%d")
        bar = spy_lookup.get(date_str)
        if bar is not None:
            val = base_capital + spy_shares * bar["close"]
            last_val = val
        else:
            val = last_val
        spy_equity.append(val)

    return pd.Series(spy_equity, index=equity_df.index, name="SPY")


# ---------------------------------------------------------------------------
# Metrics
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
        return "No trades"

    trades_df = trades_df.copy()
    trades_df["year"] = pd.to_datetime(trades_df["entry_date"]).dt.year

    lines = []
    for year, group in trades_df.groupby("year"):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Earnings Beat 3-Year Backtest")
    parser.add_argument("--polygon-key", type=str, required=True)
    parser.add_argument("--parquet-path", type=str, required=True)
    parser.add_argument("--start", type=str, default=START_DATE)
    parser.add_argument("--end", type=str, default=END_DATE)
    args = parser.parse_args()

    api_key = args.polygon_key
    start_date = args.start
    end_date = args.end

    log("=" * 60)
    log("Earnings Beat Strategy Backtest")
    log(f"Period: {start_date} -> {end_date}")
    log(f"Capital: ${INITIAL_CAPITAL:,.0f} (earnings alloc: {EARNINGS_ALLOC:.0%})")
    log(f"Entry: T+{ENTRY_DELAY} open, Exit: T+{ENTRY_DELAY + HOLD_DAYS} close ({HOLD_DAYS}d hold), Cost: {COST_BPS} bps")
    log("=" * 60)

    # 1. Load ticker universe
    log("\n[1/6] Loading ticker universe...")
    try:
        df = pd.read_parquet(args.parquet_path, columns=["Close"])
        ticker_universe = set(df.index.get_level_values("ticker").unique().tolist())
        log(f"  {len(ticker_universe)} tickers in universe")
    except Exception as e:
        log(f"ERROR: Failed to read parquet: {e}")
        sys.exit(1)

    # 2. Get NYSE trading days
    log("\n[2/6] Building NYSE trading day calendar...")
    trading_days = get_nyse_trading_days(start_date, end_date)
    log(f"  {len(trading_days)} trading days")

    # 3. Fetch all news
    log("\n[3/6] Fetching news from Polygon (month-by-month with caching)...")
    news_by_date = fetch_all_news(api_key, start_date, end_date)

    # 4. Generate signals
    log("\n[4/6] Generating earnings beat signals...")
    signals = generate_signals(news_by_date, trading_days, ticker_universe)
    log(f"  {len(signals)} total beat signals")

    # Unique tickers that need prices
    signal_tickers = sorted({s["ticker"] for s in signals})
    if "SPY" not in signal_tickers:
        signal_tickers.append("SPY")
    log(f"  {len(signal_tickers)} unique tickers need price data")

    # Show signal count by month
    sig_by_month = defaultdict(int)
    for s in signals:
        sig_by_month[s["trade_date"][:7]] += 1
    for ym in sorted(sig_by_month):
        log(f"    {ym}: {sig_by_month[ym]} signals")

    # 5. Fetch prices
    log("\n[5/6] Fetching price data (with caching)...")
    price_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    price_end = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")
    prices = fetch_prices_batch(signal_tickers, api_key, price_start, price_end)
    log(f"  Got prices for {len(prices)} tickers")

    # 6. Run backtest
    log("\n[6/6] Running backtest simulation...")
    trades_df, equity_df = run_backtest(signals, prices, trading_days)

    if equity_df.empty:
        log("ERROR: No equity data generated")
        sys.exit(1)

    # Compute metrics
    metrics = compute_metrics(equity_df["equity"], "Earnings Beat")

    spy_equity = compute_spy_benchmark(prices, equity_df)
    spy_metrics = None
    if spy_equity is not None:
        spy_metrics = compute_metrics(spy_equity, "SPY (10% alloc)")

    # ---------------------------------------------------------------------------
    # Output results
    # ---------------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("RESULTS")
    log("=" * 60)

    summary_lines = []
    summary_lines.append(f"Backtest Period: {start_date} -> {end_date}")
    summary_lines.append(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    summary_lines.append(f"Earnings Allocation: {EARNINGS_ALLOC:.0%} (${INITIAL_CAPITAL * EARNINGS_ALLOC:,.0f})")
    summary_lines.append(f"Entry: T+{ENTRY_DELAY} open, Exit: T+{ENTRY_DELAY + HOLD_DAYS} close ({HOLD_DAYS}d hold)")
    summary_lines.append(f"Transaction Cost: {COST_BPS} bps round-trip")
    summary_lines.append(f"Trading Days: {len(trading_days)}")
    summary_lines.append(f"Total Signals: {len(signals)}")
    summary_lines.append(f"Completed Trades: {len(trades_df)}")
    summary_lines.append("")

    def fmt_metrics(m):
        return (
            f"  CAGR:     {m['cagr_pct']:+.2f}%\n"
            f"  Sharpe:   {m['sharpe']:.3f}  (rf={RISK_FREE_RATE:.0%})\n"
            f"  Sortino:  {m['sortino']:.3f}\n"
            f"  MaxDD:    {m['max_dd_pct']:.2f}%\n"
            f"  Calmar:   {m['calmar']:.3f}\n"
            f"  Ann Vol:  {m['ann_vol_pct']:.2f}%\n"
            f"  Total:    {m['total_return_pct']:+.2f}%"
        )

    summary_lines.append(f"--- {metrics['label']} ---")
    summary_lines.append(fmt_metrics(metrics))
    summary_lines.append("")

    if spy_metrics:
        summary_lines.append(f"--- {spy_metrics['label']} ---")
        summary_lines.append(fmt_metrics(spy_metrics))
        summary_lines.append("")

    if not trades_df.empty:
        wins = (trades_df["pnl"] > 0).sum()
        total = len(trades_df)
        summary_lines.append(f"Win Rate: {wins}/{total} = {wins / total * 100:.1f}%")
        summary_lines.append(f"Avg Return: {trades_df['return_pct'].mean():+.2f}%")
        summary_lines.append(f"Median Return: {trades_df['return_pct'].median():+.2f}%")
        summary_lines.append(f"Best Trade: {trades_df['return_pct'].max():+.2f}%")
        summary_lines.append(f"Worst Trade: {trades_df['return_pct'].min():+.2f}%")
        summary_lines.append(f"Total P&L: ${trades_df['pnl'].sum():+,.0f}")
        summary_lines.append("")
        summary_lines.append("Annual Breakdown:")
        summary_lines.append(annual_breakdown(trades_df))

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Save files
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULT_DIR / "earnings_backtest_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    log(f"\nSaved: {summary_path}")

    if not trades_df.empty:
        trades_path = RESULT_DIR / "earnings_backtest_trades.csv"
        trades_df.to_csv(trades_path, index=False)
        log(f"Saved: {trades_path}")

    # Plot equity curve
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(equity_df.index, equity_df["equity"], label="Earnings Beat", linewidth=1.5, color="blue")
    if spy_equity is not None:
        ax.plot(spy_equity.index, spy_equity.values, label="SPY (10% alloc)", linewidth=1.5, color="gray", alpha=0.7)

    ax.set_title(
        f"Earnings Beat Strategy Backtest ({start_date} -> {end_date})\n"
        f"CAGR={metrics['cagr_pct']:+.2f}%, Sharpe={metrics['sharpe']:.3f}, MaxDD={metrics['max_dd_pct']:.2f}%",
        fontsize=12,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()

    plot_path = RESULT_DIR / "earnings_backtest_equity.png"
    fig.savefig(plot_path, dpi=150)
    log(f"Saved: {plot_path}")

    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.fill_between(equity_df.index, equity_df["n_positions"], alpha=0.4, color="orange")
    ax2.set_title("Open Positions Over Time")
    ax2.set_ylabel("# Positions")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    pos_plot_path = RESULT_DIR / "earnings_backtest_positions.png"
    fig2.savefig(pos_plot_path, dpi=150)
    log(f"Saved: {pos_plot_path}")

    log("\nDone!")


if __name__ == "__main__":
    main()
