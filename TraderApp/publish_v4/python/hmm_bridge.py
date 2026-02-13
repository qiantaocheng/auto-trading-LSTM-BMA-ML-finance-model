#!/usr/bin/env python
"""HMM Risk Assessment Bridge for TraderApp.

Fetches SPY daily data from Polygon API, trains a 3-state Gaussian HMM,
computes p_crisis / risk_gate, applies hysteresis for crisis mode detection,
and persists state across runs.

Output: single JSON object on stdout.
Progress: JSON lines on stderr.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # D:\trade
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BMA_DIR = ROOT / "bma_models"
if str(BMA_DIR) not in sys.path:
    sys.path.insert(0, str(BMA_DIR))

STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "hmm_state.json"
_STATE_DB_PATH = None  # Set by --state-db arg; if set, use SQLite instead of JSON file

# --- HMM Parameters ---
N_STATES = 3
HMM_WINDOW = 1000       # training window in trading days
FETCH_DAYS = 1400        # fetch extra days to ensure 1000 after dropna
EMA_SPAN = 4             # EMA span for p_crisis smoothing
GAMMA = 2                # risk_gate exponent
RISK_GATE_MIN = 0.05     # below this, treat as 0

# Hysteresis thresholds
CRISIS_ENTER_THRESH = 0.70
CRISIS_EXIT_THRESH = 0.40
CRISIS_CONFIRM_DAYS = 2
SAFE_CONFIRM_DAYS = 2
COOLDOWN_DAYS = 3


def emit_progress(step: str, progress: int, detail: str = "") -> None:
    msg = json.dumps({"step": step, "progress": progress, "detail": detail})
    print(msg, file=sys.stderr, flush=True)


_DEFAULT_STATE = {
    "p_crisis_history": [],
    "crisis_mode": False,
    "crisis_confirm_days": 0,
    "safe_confirm_days": 0,
    "cooldown_remaining": 0,
    "last_run_date": None,
    "rebalance_day_counter": 0,
}

_HMM_STATE_KEYS = [
    "hmm_p_crisis_history",
    "hmm_crisis_mode",
    "hmm_crisis_confirm_days",
    "hmm_safe_confirm_days",
    "hmm_cooldown_remaining",
    "hmm_last_run_date",
    "hmm_rebalance_day_counter",
]


def _load_state_from_db(db_path: str) -> dict:
    """Load HMM state from SQLite etf_rotation_state table."""
    import sqlite3
    state = dict(_DEFAULT_STATE)
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        cursor = conn.execute(
            "SELECT key, value FROM etf_rotation_state WHERE key LIKE 'hmm_%'")
        rows = {r[0]: r[1] for r in cursor.fetchall()}
        conn.close()

        if "hmm_p_crisis_history" in rows:
            try:
                state["p_crisis_history"] = json.loads(rows["hmm_p_crisis_history"])
            except Exception:
                pass
        if "hmm_crisis_mode" in rows:
            state["crisis_mode"] = rows["hmm_crisis_mode"] == "True"
        if "hmm_crisis_confirm_days" in rows:
            try:
                state["crisis_confirm_days"] = int(rows["hmm_crisis_confirm_days"])
            except (ValueError, TypeError):
                pass
        if "hmm_safe_confirm_days" in rows:
            try:
                state["safe_confirm_days"] = int(rows["hmm_safe_confirm_days"])
            except (ValueError, TypeError):
                pass
        if "hmm_cooldown_remaining" in rows:
            try:
                state["cooldown_remaining"] = int(rows["hmm_cooldown_remaining"])
            except (ValueError, TypeError):
                pass
        if "hmm_last_run_date" in rows:
            state["last_run_date"] = rows["hmm_last_run_date"] or None
        if "hmm_rebalance_day_counter" in rows:
            try:
                state["rebalance_day_counter"] = int(rows["hmm_rebalance_day_counter"])
            except (ValueError, TypeError):
                pass
    except Exception as e:
        print(f"DB state load failed, using defaults: {e}", file=sys.stderr)
    return state


def _save_state_to_db(db_path: str, state: dict) -> None:
    """Save HMM state to SQLite etf_rotation_state table."""
    import sqlite3
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        pairs = [
            ("hmm_p_crisis_history", json.dumps(state.get("p_crisis_history", []))),
            ("hmm_crisis_mode", str(state.get("crisis_mode", False))),
            ("hmm_crisis_confirm_days", str(state.get("crisis_confirm_days", 0))),
            ("hmm_safe_confirm_days", str(state.get("safe_confirm_days", 0))),
            ("hmm_cooldown_remaining", str(state.get("cooldown_remaining", 0))),
            ("hmm_last_run_date", state.get("last_run_date") or ""),
            ("hmm_rebalance_day_counter", str(state.get("rebalance_day_counter", 0))),
        ]
        for key, value in pairs:
            conn.execute(
                "INSERT OR REPLACE INTO etf_rotation_state (key, value) VALUES (?, ?)",
                (key, value))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB state save failed: {e}", file=sys.stderr)


def load_state() -> dict:
    if _STATE_DB_PATH:
        return _load_state_from_db(_STATE_DB_PATH)
    # Fallback to JSON file
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return dict(_DEFAULT_STATE)


def save_state(state: dict) -> None:
    if _STATE_DB_PATH:
        _save_state_to_db(_STATE_DB_PATH, state)
        return
    # Fallback to JSON file
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _resolve_api_key(env_key: str) -> str:
    """Resolve Polygon API key: env var → api_config module → fallback."""
    if env_key:
        return env_key
    try:
        from api_config import POLYGON_API_KEY
        return POLYGON_API_KEY
    except Exception:
        return ""


def _is_nyse_holiday(d: datetime) -> bool:
    """Check if a date is an NYSE holiday. Matches C# IsNyseHoliday."""
    year = d.year
    # Fixed holidays (adjusted for weekends)
    def adjust_weekend(dt):
        if dt.weekday() == 5:  # Saturday → Friday
            return dt - timedelta(days=1)
        if dt.weekday() == 6:  # Sunday → Monday
            return dt + timedelta(days=1)
        return dt

    def nth_weekday(y, month, dow, n):
        """nth occurrence of dow (0=Mon) in month."""
        from calendar import monthcalendar
        cal = monthcalendar(y, month)
        days = [week[dow] for week in cal if week[dow] != 0]
        return datetime(y, month, days[n - 1])

    def last_weekday(y, month, dow):
        from calendar import monthcalendar
        cal = monthcalendar(y, month)
        days = [week[dow] for week in cal if week[dow] != 0]
        return datetime(y, month, days[-1])

    def easter_sunday(y):
        a, b, c = y % 19, y // 100, y % 100
        d, e = b // 4, b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i, k = c // 4, c % 4
        l_ = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l_) // 451
        month = (h + l_ - 7 * m + 114) // 31
        day = ((h + l_ - 7 * m + 114) % 31) + 1
        return datetime(y, month, day)

    holidays = [
        adjust_weekend(datetime(year, 1, 1)),       # New Year
        adjust_weekend(datetime(year, 6, 19)),       # Juneteenth
        adjust_weekend(datetime(year, 7, 4)),        # Independence Day
        adjust_weekend(datetime(year, 12, 25)),      # Christmas
        nth_weekday(year, 1, 0, 3),                  # MLK Day (3rd Monday Jan)
        nth_weekday(year, 2, 0, 3),                  # Presidents Day (3rd Monday Feb)
        last_weekday(year, 5, 0),                    # Memorial Day (last Monday May)
        nth_weekday(year, 9, 0, 1),                  # Labor Day (1st Monday Sep)
        nth_weekday(year, 11, 3, 4),                 # Thanksgiving (4th Thursday Nov)
        easter_sunday(year) - timedelta(days=2),     # Good Friday
    ]
    return d.replace(hour=0, minute=0, second=0, microsecond=0) in [
        h.replace(hour=0, minute=0, second=0, microsecond=0) for h in holidays
    ]


def _is_trading_day(d: datetime) -> bool:
    """Check if date is an NYSE trading day (not weekend, not holiday)."""
    if d.weekday() >= 5:
        return False
    return not _is_nyse_holiday(d)


def get_last_closed_trading_day() -> datetime:
    """Get the last fully closed trading day (to avoid lookahead bias).

    Rules:
    - If run before 4:00 PM ET today: use previous trading day's close
    - If run after 4:00 PM ET today: use today's close (if today is trading day)
    - Skip weekends and NYSE holidays
    """
    import pytz
    et_tz = pytz.timezone("US/Eastern")
    now_et = datetime.now(et_tz)

    # Market closes at 4:00 PM ET
    market_close_today = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    # If before market close, use previous day
    if now_et < market_close_today:
        current = now_et - timedelta(days=1)
    else:
        current = now_et

    # Skip back to last trading day (weekends + NYSE holidays)
    while not _is_trading_day(current):
        current -= timedelta(days=1)

    return current.replace(hour=0, minute=0, second=0, microsecond=0)


def fetch_spy_data(api_key: str) -> pd.DataFrame:
    """Fetch SPY daily data from Polygon API.

    CRITICAL: Only fetches data up to last fully closed trading day
    to prevent lookahead bias.
    """
    last_closed = get_last_closed_trading_day()
    end_date = last_closed.strftime("%Y-%m-%d")
    start_date = (last_closed - timedelta(days=FETCH_DAYS)).strftime("%Y-%m-%d")

    emit_progress("Fetching SPY data", 10,
                  f"Last {FETCH_DAYS} days ending {end_date} (last closed trading day)")

    from polygon_client import PolygonClient
    client = PolygonClient(api_key=api_key)

    df = client.get_historical_bars("SPY", start_date, end_date, "day", 1)
    if df.empty:
        raise RuntimeError("Failed to fetch SPY data from Polygon API")

    # Additional safety: filter out any bars after last_closed (should not happen but be defensive)
    # Convert last_closed to naive datetime for comparison (polygon data is timezone-naive)
    last_closed_naive = pd.Timestamp(last_closed.replace(tzinfo=None))
    df = df[df.index <= last_closed_naive]

    emit_progress("SPY data fetched", 30, f"{len(df)} trading days up to {end_date}")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute HMM features: mkt_logret_1d and mkt_vol_10d.

    CRITICAL: vol_10d uses past 10 days (strict), no lookahead.
    """
    emit_progress("Computing features", 40, "log returns + 10d rolling vol")

    close = df["Close"].astype(float)
    # Log return: today's return = log(close_t / close_{t-1})
    logret = np.log(close / close.shift(1))

    # 10-day rolling volatility: uses past 10 days INCLUDING today
    # This is correct: vol at day t uses [t-9, t-8, ..., t-1, t]
    vol10 = logret.rolling(window=10, min_periods=10).std()

    features = pd.DataFrame({
        "mkt_logret_1d": logret,
        "mkt_vol_10d": vol10,
    }, index=df.index)

    # Drop NaN rows (first row for logret, first 9 rows for vol10)
    features = features.dropna()
    emit_progress("Features computed", 45, f"{len(features)} valid rows after NaN drop")
    return features


def train_hmm(features: pd.DataFrame) -> tuple:
    """Train 3-state GaussianHMM on last HMM_WINDOW days, return model + state labels + scaler.

    CRITICAL IMPROVEMENTS:
    1. Standardize features to prevent scale dominance
    2. Stable state labeling by mean volatility (highest vol = CRISIS)
    3. Return scaler for consistent transform during prediction
    """
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    emit_progress("Training HMM", 50, f"{N_STATES} states, {HMM_WINDOW}-day window")

    # Use last HMM_WINDOW days
    if len(features) > HMM_WINDOW:
        train_data = features.iloc[-HMM_WINDOW:]
    else:
        train_data = features

    # Standardize features (critical for HMM convergence)
    scaler = StandardScaler()
    X_raw = train_data[["mkt_logret_1d", "mkt_vol_10d"]].values
    X = scaler.fit_transform(X_raw)

    emit_progress("Training HMM", 55, f"Features standardized: mean={X.mean(axis=0)}, std={X.std(axis=0)}")

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=200,
        random_state=42,
        tol=1e-4,
    )

    # Suppress stdout during fit (hmmlearn may print warnings)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)

    # STABLE state labeling: sort by mean volatility in ORIGINAL scale
    # Composite key: vol is primary; |logret| breaks ties within ~1pp vol band
    # Weight 0.1 ensures |logret| range [0,0.05] contributes at most 0.5pp to sort key
    means_original = scaler.inverse_transform(model.means_)
    mean_vols = means_original[:, 1]  # column 1 = mkt_vol_10d in original scale
    mean_abs_logret = np.abs(means_original[:, 0])  # col0 = logret magnitude
    composite_key = mean_vols + mean_abs_logret * 0.1
    state_order = np.argsort(composite_key)
    label_map = {
        int(state_order[0]): "SAFE",
        int(state_order[1]): "MID",
        int(state_order[2]): "CRISIS",
    }
    crisis_state_idx = int(state_order[2])

    emit_progress("HMM trained", 65,
                  f"States: SAFE={state_order[0]} MID={state_order[1]} CRISIS={state_order[2]} | "
                  f"mean vols: {dict(zip(['SAFE','MID','CRISIS'], mean_vols[state_order]))}")

    return model, label_map, crisis_state_idx, scaler


def get_posterior_crisis_prob(model, features: pd.DataFrame, crisis_state_idx: int, scaler) -> float:
    """Get posterior probability of crisis state for the latest day.

    CRITICAL: Must use the same scaler from training to transform features.
    """
    emit_progress("Computing posteriors", 70, "Latest day crisis probability")

    X_raw = features[["mkt_logret_1d", "mkt_vol_10d"]].values
    X = scaler.transform(X_raw)  # Apply same standardization as training
    posteriors = model.predict_proba(X)
    p_crisis = float(posteriors[-1, crisis_state_idx])
    return p_crisis


def compute_risk_gate(p_crisis_smooth: float) -> float:
    """risk_gate = (1 - p_crisis_smooth) ^ gamma, floored at RISK_GATE_MIN."""
    rg = (1.0 - p_crisis_smooth) ** GAMMA
    if rg < RISK_GATE_MIN:
        rg = 0.0
    return round(rg, 6)


def apply_hysteresis(state: dict, p_crisis_smooth: float, trading_date_str: str) -> dict:
    """Apply hysteresis logic for crisis mode transitions.

    CRITICAL: Only processes on NEW TRADING DAYS (not calendar days).
    Caller must ensure trading_date_str is the latest SPY data date, not calendar date.
    This ensures weekends/holidays don't cause spurious counter increments.
    """
    crisis_mode = state.get("crisis_mode", False)
    crisis_confirm = state.get("crisis_confirm_days", 0)
    safe_confirm = state.get("safe_confirm_days", 0)
    cooldown = state.get("cooldown_remaining", 0)
    last_run = state.get("last_run_date")

    # Only process once per trading day (not calendar day)
    if last_run == trading_date_str:
        return state

    # Decrement cooldown (only on new trading days)
    if cooldown > 0:
        cooldown -= 1

    if not crisis_mode:
        # Check for crisis entry
        if p_crisis_smooth >= CRISIS_ENTER_THRESH and cooldown == 0:
            crisis_confirm += 1
            if crisis_confirm >= CRISIS_CONFIRM_DAYS:
                crisis_mode = True
                crisis_confirm = 0
                safe_confirm = 0
        else:
            crisis_confirm = 0
    else:
        # Check for crisis exit
        if p_crisis_smooth <= CRISIS_EXIT_THRESH:
            safe_confirm += 1
            if safe_confirm >= SAFE_CONFIRM_DAYS:
                crisis_mode = False
                safe_confirm = 0
                crisis_confirm = 0
                cooldown = COOLDOWN_DAYS
        else:
            safe_confirm = 0

    state["crisis_mode"] = crisis_mode
    state["crisis_confirm_days"] = crisis_confirm
    state["safe_confirm_days"] = safe_confirm
    state["cooldown_remaining"] = cooldown

    return state


def get_current_hmm_state(model, features: pd.DataFrame, label_map: dict, scaler) -> str:
    """Get the HMM state label for the latest day.

    CRITICAL: Must use the same scaler from training to transform features.
    """
    X_raw = features[["mkt_logret_1d", "mkt_vol_10d"]].values
    X = scaler.transform(X_raw)  # Apply same standardization as training
    states = model.predict(X)
    latest_state = int(states[-1])
    return label_map.get(latest_state, "UNKNOWN")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="HMM Risk Assessment Bridge")
    parser.add_argument("--reset-counter", action="store_true",
                        help="Reset rebalance_day_counter to 0 and exit")
    parser.add_argument("--polygon-key", type=str, default="",
                        help="Polygon API key (overrides env var and api_config)")
    parser.add_argument("--state-db", type=str, default="",
                        help="SQLite DB path for state persistence (uses etf_rotation_state table)")
    args = parser.parse_args()

    global _STATE_DB_PATH
    if args.state_db and os.path.exists(args.state_db):
        _STATE_DB_PATH = args.state_db

    # Quick command: reset rebalance counter
    if args.reset_counter:
        state = load_state()
        state["rebalance_day_counter"] = 0
        save_state(state)
        print(json.dumps({"reset": True, "rebalance_day_counter": 0}))
        return

    try:
        # Priority: CLI arg → env var → api_config module
        api_key = args.polygon_key or _resolve_api_key(os.environ.get("POLYGON_API_KEY", ""))
        if not api_key:
            print(json.dumps({"error": "No Polygon API key found (env var or api_config.py)"}))
            sys.exit(1)

        emit_progress("Starting HMM assessment", 5, "")

        # 1. Fetch SPY data
        spy_df = fetch_spy_data(api_key)

        # 2. Compute features
        features = compute_features(spy_df)
        if len(features) < 100:
            print(json.dumps({"error": f"Insufficient data: only {len(features)} days"}))
            sys.exit(1)

        # 3. Train HMM
        model, label_map, crisis_state_idx, scaler = train_hmm(features)

        # 4. Get posterior crisis probability
        p_crisis = get_posterior_crisis_prob(model, features, crisis_state_idx, scaler)

        # 5. Load state, compute EMA smoothing
        emit_progress("Computing risk gate", 75, "EMA smoothing + hysteresis")
        state = load_state()

        # Append to history (keep last 30 entries)
        history = state.get("p_crisis_history", [])
        history.append(p_crisis)
        if len(history) > 30:
            history = history[-30:]
        state["p_crisis_history"] = history

        # EMA smoothing
        series = pd.Series(history)
        p_crisis_smooth = float(series.ewm(span=EMA_SPAN, adjust=False).mean().iloc[-1])

        # 6. Risk gate
        risk_gate = compute_risk_gate(p_crisis_smooth)

        # 7. Hysteresis
        today_str = datetime.now().strftime("%Y-%m-%d")
        features_date = features.index[-1].strftime("%Y-%m-%d") if hasattr(features.index[-1], "strftime") else str(features.index[-1])
        state = apply_hysteresis(state, p_crisis_smooth, features_date)

        # 8. Increment rebalance counter (ONLY on NEW TRADING DAYS, PAUSE during crisis)
        # CRITICAL: Only increment when features_date is newer than last_run_date
        # This ensures weekends/holidays don't increment the counter
        # During crisis mode: pause normal 5-day rotation (counter frozen)
        last_run = state.get("last_run_date")
        rebal_counter = state.get("rebalance_day_counter", 0)

        if last_run != features_date:
            if not state["crisis_mode"]:
                # Normal mode: increment counter
                rebal_counter += 1
                state["rebalance_day_counter"] = rebal_counter
                emit_progress("Rebalance counter", 78, f"Incremented to {rebal_counter} (new trading day)")
            else:
                # Crisis mode: freeze counter (no normal rebalancing during crisis)
                emit_progress("Rebalance counter", 78, f"Frozen at {rebal_counter} (crisis mode active)")
        else:
            emit_progress("Rebalance counter", 78, f"No change: {rebal_counter} (same trading day)")

        state["last_run_date"] = features_date

        # 9. Get current HMM state label
        hmm_state = get_current_hmm_state(model, features, label_map, scaler)

        # 10. Save state
        save_state(state)

        emit_progress("Complete", 100, f"state={hmm_state} risk_gate={risk_gate:.4f}")

        # 10b. Compute SPY MA200 cap for direct prediction strategy
        spy_close = spy_df["close"] if "close" in spy_df.columns else spy_df["Close"]
        spy_ma200 = spy_close.rolling(200).mean()
        latest_spy = float(spy_close.iloc[-1])
        latest_ma200 = float(spy_ma200.iloc[-1]) if pd.notna(spy_ma200.iloc[-1]) else 0.0
        if latest_ma200 > 0:
            if latest_spy < latest_ma200 * 0.95:
                spy_ma200_cap = 0.30
            elif latest_spy < latest_ma200:
                spy_ma200_cap = 0.60
            else:
                spy_ma200_cap = 1.0
        else:
            spy_ma200_cap = 1.0  # not enough data for MA200
        emit_progress("MA200 check", 90,
                       f"SPY={latest_spy:.2f} MA200={latest_ma200:.2f} cap={spy_ma200_cap:.2f}")

        # 11. Output result
        training_days = min(len(features), HMM_WINDOW)
        result = {
            "p_crisis": round(p_crisis, 6),
            "p_crisis_smooth": round(p_crisis_smooth, 6),
            "risk_gate": risk_gate,
            "hmm_state": hmm_state,
            "crisis_mode": state["crisis_mode"],
            "crisis_confirm_days": state["crisis_confirm_days"],
            "safe_confirm_days": state["safe_confirm_days"],
            "cooldown_remaining": state["cooldown_remaining"],
            "rebalance_day_counter": rebal_counter,
            "training_days": training_days,
            "features_date": features.index[-1].strftime("%Y-%m-%d") if hasattr(features.index[-1], "strftime") else str(features.index[-1]),
            "spy_ma200_cap": spy_ma200_cap,
            "spy_price": round(latest_spy, 2),
            "spy_ma200": round(latest_ma200, 2),
        }
        print(json.dumps(result, ensure_ascii=False))

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
