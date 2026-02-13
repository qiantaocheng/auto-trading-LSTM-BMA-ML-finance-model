#!/usr/bin/env python3
"""Regime Probabilistic Investigation — HMM + Markov Transition Matrix.

Compares rule-based V7 state machine with data-driven HMM regimes.
Analyzes transition probabilities, state durations, and soft sizing.

Usage:
    python scripts/regime_probabilistic_investigation.py
    python scripts/regime_probabilistic_investigation.py --hmm-states 4
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ═══════════════════════════════════════════════════════════════════════
# Part A: Historical Reconstruction of V7 Rule-Based States
# ═══════════════════════════════════════════════════════════════════════

# V7 thresholds (from EtfRotationSchedulerService.cs)
VIX_CRISIS_THRESHOLD = 40.0
VIX_CRISIS_EXTREME = 50.0
DD_CRISIS_THRESHOLD = -0.10
DD_RISK_OFF_THRESHOLD = -0.06
DD_RECOVERY_THRESHOLD = -0.04
VIX_RECOVERY_THRESHOLD = 20.0
SPY_BELOW_MA200_CONFIRM = 3
RISK_OFF_CONFIRM_DAYS = 2
CRISIS_COOLDOWN_DAYS = 10
SOFT_EXIT_VIX = 32.0
SOFT_EXIT_CONFIRM = 3
SOFT_PLUS_VIX = 25.0
SOFT_PLUS_CONFIRM = 3
RECOVERY_CONFIRM_DAYS = 5


def compute_spy_ma200(close: pd.Series) -> pd.Series:
    """200-day simple moving average of SPY close."""
    return close.rolling(200, min_periods=200).mean()


def compute_spy_drawdown(close: pd.Series) -> pd.Series:
    """Running drawdown from rolling peak."""
    peak = close.expanding().max()
    return (close - peak) / peak


def reconstruct_v7_states(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct V7 risk state machine from SPY + VIX data.

    Matches EtfRotationSchedulerService.cs logic as closely as possible.
    Uses SPY drawdown as proxy for portfolio drawdown.

    Returns DataFrame with columns: date, state, sub_state, exposure_cap.
    """
    close = df['close']
    ma200 = compute_spy_ma200(close)
    dd = compute_spy_drawdown(close)
    vix = df['vix_close']

    records = []
    state = "RISK_ON"
    sub_state = "HARD"
    spy_below_days = 0
    risk_off_confirm = 0
    soft_exit_counter = 0
    soft_plus_counter = 0
    recovery_confirm = 0
    cooldown = 0
    transitions_this_cycle = 0

    for i, date in enumerate(df.index):
        if pd.isna(ma200.iloc[i]):
            records.append({'date': date, 'state': 'WARMUP', 'sub_state': '',
                            'exposure_cap': 1.0, 'vix': vix.iloc[i],
                            'spy_vs_ma200': np.nan, 'drawdown': dd.iloc[i]})
            continue

        spy_dev = (close.iloc[i] - ma200.iloc[i]) / ma200.iloc[i]
        spy_below = close.iloc[i] < ma200.iloc[i]
        v = vix.iloc[i] if not pd.isna(vix.iloc[i]) else 20.0
        d = dd.iloc[i]

        old_state = state
        new_state = state
        reason = ""

        # Track SPY below MA200
        if spy_below:
            spy_below_days += 1
        else:
            spy_below_days = 0

        # Cooldown
        if cooldown > 0:
            cooldown -= 1

        # ── Layer 1: CRISIS triggers ──
        if v >= VIX_CRISIS_THRESHOLD and spy_below:
            new_state = "CRISIS"
            reason = f"VIX≥{VIX_CRISIS_THRESHOLD}+SPY<MA200"
        elif v >= VIX_CRISIS_EXTREME:
            new_state = "CRISIS"
            reason = f"VIX≥{VIX_CRISIS_EXTREME}"
        elif d <= DD_CRISIS_THRESHOLD:
            new_state = "CRISIS"
            reason = f"DD≥10%"

        # ── Layer 2: RISK_OFF ──
        if new_state != "CRISIS":
            if spy_below and d <= DD_RISK_OFF_THRESHOLD:
                risk_off_confirm += 1
                if risk_off_confirm >= RISK_OFF_CONFIRM_DAYS and old_state != "CRISIS":
                    new_state = "RISK_OFF"
            else:
                risk_off_confirm = 0

            if spy_below_days >= SPY_BELOW_MA200_CONFIRM and new_state not in ("CRISIS", "RISK_OFF"):
                new_state = "RISK_OFF"

        # ── Soft CRISIS exit logic ──
        if old_state == "CRISIS" and new_state == "CRISIS":
            if sub_state == "HARD":
                if v < SOFT_EXIT_VIX:
                    soft_exit_counter += 1
                    if soft_exit_counter >= SOFT_EXIT_CONFIRM:
                        sub_state = "SOFT"
                        soft_exit_counter = 0
                else:
                    soft_exit_counter = 0
            elif sub_state == "SOFT":
                if v >= VIX_CRISIS_THRESHOLD:
                    sub_state = "HARD"
                    soft_plus_counter = 0
                elif not spy_below and v < SOFT_PLUS_VIX:
                    soft_plus_counter += 1
                    if soft_plus_counter >= SOFT_PLUS_CONFIRM:
                        sub_state = "SOFT_PLUS"
                        soft_plus_counter = 0
                else:
                    soft_plus_counter = 0
            elif sub_state == "SOFT_PLUS":
                if v >= VIX_CRISIS_THRESHOLD:
                    sub_state = "HARD"

        # ── Recovery logic ──
        recovery_ok = (d > DD_RECOVERY_THRESHOLD and not spy_below
                       and v < VIX_RECOVERY_THRESHOLD)
        if recovery_ok and old_state in ("RISK_OFF", "CRISIS", "RECOVERY_RAMP"):
            recovery_confirm += 1
            if recovery_confirm >= RECOVERY_CONFIRM_DAYS:
                if old_state in ("RISK_OFF", "CRISIS"):
                    new_state = "RECOVERY_RAMP"
                    recovery_confirm = 0
                elif old_state == "RECOVERY_RAMP":
                    new_state = "RISK_ON"
                    recovery_confirm = 0
        elif not recovery_ok and old_state != "RECOVERY_RAMP":
            recovery_confirm = 0

        # ── Apply transition ──
        if new_state != old_state:
            state = new_state
            if new_state == "CRISIS":
                sub_state = "HARD"
                cooldown = CRISIS_COOLDOWN_DAYS
                soft_exit_counter = 0
                soft_plus_counter = 0
            elif new_state == "RISK_ON":
                sub_state = ""
            elif new_state == "RISK_OFF":
                sub_state = ""
            elif new_state == "RECOVERY_RAMP":
                sub_state = ""

        # Compute exposure cap
        if state == "CRISIS":
            if sub_state == "HARD":
                cap = 0.30
            elif sub_state == "SOFT":
                cap = 0.55
            else:
                cap = 0.80
        elif state == "RISK_OFF":
            if d <= -0.08:
                cap = 0.70
            else:
                cap = 0.80
        elif state == "RECOVERY_RAMP":
            cap = 0.80  # simplified — actual ramps +5-10%/day
        else:
            cap = 1.0

        records.append({
            'date': date, 'state': state, 'sub_state': sub_state,
            'exposure_cap': cap, 'vix': v, 'spy_vs_ma200': spy_dev,
            'drawdown': d,
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════
# Part B: HMM Training and Comparison
# ═══════════════════════════════════════════════════════════════════════

def prepare_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare HMM observables from regime_features data.

    Observables:
      1. SPY daily log return
      2. 10-day rolling vol (Parkinson or from close)
      3. VIX level (z-scored)
      4. QQQ/TLT ratio z-score (risk premium proxy, replaces credit spread)
    """
    close = df['close']
    logret = np.log(close / close.shift(1))
    vol_10d = logret.rolling(10, min_periods=10).std()

    features = pd.DataFrame({
        'logret': logret,
        'vol_10d': vol_10d,
        'vix_z': df['vix_z_score'],
        'risk_prem_z': df['qqq_tlt_ratio_z'],
    }, index=df.index).dropna()

    return features


def train_hmm_nstates(features: pd.DataFrame, n_states: int,
                      seed: int = 42) -> Tuple:
    """Train Gaussian HMM with n_states on the features.

    Returns (model, state_labels, scaler, state_sequence).
    States labeled by mean volatility: lowest=SAFE, highest=CRISIS.
    """
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=300,
        random_state=seed,
        tol=1e-5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)

    # Label states by mean vol (column 1 = vol_10d in scaled space)
    means_orig = scaler.inverse_transform(model.means_)
    mean_vols = means_orig[:, 1]
    order = np.argsort(mean_vols)

    if n_states == 2:
        label_names = ["SAFE", "CRISIS"]
    elif n_states == 3:
        label_names = ["SAFE", "MID", "CRISIS"]
    elif n_states == 4:
        label_names = ["SAFE", "MID_LOW", "MID_HIGH", "CRISIS"]
    else:
        label_names = [f"S{i}" for i in range(n_states)]

    label_map = {int(order[i]): label_names[i] for i in range(n_states)}

    # Decode most likely state sequence
    state_seq = model.predict(X)
    labeled_seq = [label_map[s] for s in state_seq]

    # Posterior probabilities
    posteriors = model.predict_proba(X)

    return model, label_map, scaler, labeled_seq, posteriors, means_orig[order]


def compare_hmm_vs_rulebased(hmm_labels: List[str], rule_states: pd.Series,
                              dates: pd.DatetimeIndex) -> Dict:
    """Compare HMM regime labels with rule-based V7 states.

    Maps HMM states to simplified {SAFE→RISK_ON, MID→RISK_ON, CRISIS→not-RISK_ON}
    for comparison.
    """
    # Simplify rule-based states to binary: risk_on vs not_risk_on
    rule_binary = rule_states.map(lambda s: 'RISK_ON' if s == 'RISK_ON' else 'NOT_RISK_ON')

    # Map HMM states: CRISIS → NOT_RISK_ON, everything else → RISK_ON
    hmm_binary = ['NOT_RISK_ON' if 'CRISIS' in h else 'RISK_ON' for h in hmm_labels]

    # Agreement rate
    agree = sum(1 for a, b in zip(hmm_binary, rule_binary) if a == b)
    total = len(hmm_binary)

    # Find divergence dates
    divergences = []
    for i, (hb, rb, d) in enumerate(zip(hmm_binary, rule_binary, dates)):
        if hb != rb:
            divergences.append({
                'date': d,
                'hmm': hmm_labels[i],
                'rule': rule_states.iloc[i],
                'hmm_simple': hb,
                'rule_simple': rb,
            })

    # Confusion matrix
    confusion = defaultdict(int)
    for h, r in zip(hmm_binary, rule_binary):
        confusion[(h, r)] += 1

    return {
        'agreement_rate': agree / total if total > 0 else 0,
        'n_agree': agree,
        'n_total': total,
        'n_diverge': len(divergences),
        'confusion': dict(confusion),
        'divergences': divergences[:20],  # first 20
    }


# ═══════════════════════════════════════════════════════════════════════
# Part C: Markov Transition Matrix
# ═══════════════════════════════════════════════════════════════════════

def compute_transition_matrix(state_sequence: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """Compute 1-step transition probability matrix from state sequence.

    Returns:
      - transition_matrix: DataFrame with P(next_state | current_state)
      - counts: raw transition counts
    """
    states = sorted(set(state_sequence))
    n = len(states)
    state_idx = {s: i for i, s in enumerate(states)}
    counts = np.zeros((n, n), dtype=int)

    for i in range(len(state_sequence) - 1):
        curr = state_idx[state_sequence[i]]
        nxt = state_idx[state_sequence[i + 1]]
        counts[curr, nxt] += 1

    # Normalize rows
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div by zero
    probs = counts / row_sums

    tm = pd.DataFrame(probs, index=states, columns=states)
    count_dict = {(states[i], states[j]): int(counts[i, j])
                  for i in range(n) for j in range(n)}

    return tm, count_dict


def compute_state_durations(state_sequence: List[str]) -> Dict[str, Dict]:
    """Compute duration statistics for each state.

    Returns dict of {state: {mean, median, std, min, max, n_episodes, durations}}.
    """
    if not state_sequence:
        return {}

    durations = defaultdict(list)
    current = state_sequence[0]
    run_len = 1

    for i in range(1, len(state_sequence)):
        if state_sequence[i] == current:
            run_len += 1
        else:
            durations[current].append(run_len)
            current = state_sequence[i]
            run_len = 1
    durations[current].append(run_len)  # last run

    stats = {}
    for state, durs in durations.items():
        arr = np.array(durs)
        stats[state] = {
            'mean_duration': float(np.mean(arr)),
            'median_duration': float(np.median(arr)),
            'std_duration': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            'min_duration': int(np.min(arr)),
            'max_duration': int(np.max(arr)),
            'n_episodes': len(durs),
            'total_days': int(np.sum(arr)),
            'durations': durs,
        }
    return stats


def conditional_persistence(state_sequence: List[str], max_lag: int = 20) -> Dict:
    """Compute P(still in state S at t+k | entered state S at t=0) for k=1..max_lag.

    Shows how persistence varies with time-in-state.
    """
    states = sorted(set(state_sequence))
    result = {}

    for target_state in states:
        # Find entry points (transitions INTO this state)
        entries = []
        for i in range(1, len(state_sequence)):
            if state_sequence[i] == target_state and state_sequence[i - 1] != target_state:
                entries.append(i)

        if not entries:
            continue

        persistence = []
        for lag in range(1, max_lag + 1):
            still_in = 0
            total = 0
            for entry in entries:
                if entry + lag < len(state_sequence):
                    total += 1
                    if state_sequence[entry + lag] == target_state:
                        still_in += 1
            if total > 0:
                persistence.append({
                    'lag': lag,
                    'p_persist': still_in / total,
                    'n_obs': total,
                })
        result[target_state] = persistence

    return result


# ═══════════════════════════════════════════════════════════════════════
# Part D: Soft Position Sizing Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_soft_sizing(df: pd.DataFrame, posteriors: np.ndarray,
                          hmm_labels: List[str], label_map: Dict,
                          hmm_dates: pd.DatetimeIndex) -> Dict:
    """Evaluate soft position sizing using HMM probability.

    Compares:
      1. Binary rule-based: 100% RISK_ON, 30% CRISIS
      2. Binary HMM: same thresholds using HMM crisis probability
      3. Soft HMM: exposure = (1 - p_crisis)^2 (continuous)

    Uses SPY returns as portfolio proxy.
    """
    # Find crisis state index in label_map
    crisis_idx = None
    for raw_idx, label in label_map.items():
        if 'CRISIS' in label:
            crisis_idx = raw_idx
            break

    if crisis_idx is None:
        return {'error': 'No CRISIS state found in HMM'}

    p_crisis = posteriors[:, crisis_idx]

    # Align with SPY returns
    spy_close = df['close'].reindex(hmm_dates)
    spy_ret = spy_close.pct_change().fillna(0).values

    # Strategy 1: Always 100% exposure (buy-and-hold SPY)
    bh_equity = np.cumprod(1 + spy_ret)

    # Strategy 2: Binary rule-based sizing (from V7 reconstruction)
    # Need the rule states aligned
    # We'll just use p_crisis-based strategies

    # Strategy 3: Hard threshold on p_crisis (like hysteresis)
    hard_exposure = np.where(p_crisis >= 0.70, 0.30, 1.0)
    hard_ret = spy_ret * hard_exposure
    hard_equity = np.cumprod(1 + hard_ret)

    # Strategy 4: Soft sizing — exposure = (1 - p_crisis)^2
    soft_exposure = (1 - p_crisis) ** 2
    soft_exposure = np.clip(soft_exposure, 0.05, 1.0)
    soft_ret = spy_ret * soft_exposure
    soft_equity = np.cumprod(1 + soft_ret)

    # Strategy 5: EMA-smoothed soft sizing (span=4)
    p_smooth = pd.Series(p_crisis).ewm(span=4, adjust=False).mean().values
    smooth_exposure = (1 - p_smooth) ** 2
    smooth_exposure = np.clip(smooth_exposure, 0.05, 1.0)
    smooth_ret = spy_ret * smooth_exposure
    smooth_equity = np.cumprod(1 + smooth_ret)

    def _stats(rets: np.ndarray, label: str, exposure: np.ndarray) -> Dict:
        cum = np.cumprod(1 + rets)
        n = len(rets)
        if n < 2:
            return {'label': label, 'error': 'insufficient data'}
        mean_r = float(np.mean(rets))
        std_r = float(np.std(rets, ddof=1))
        sharpe = float(mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0
        peak = np.maximum.accumulate(cum)
        max_dd = float(((cum - peak) / peak).min())
        n_years = n / 252.0
        cagr = float(cum[-1] ** (1.0 / n_years) - 1) if cum[-1] > 0 and n_years > 0 else 0
        calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0
        return {
            'label': label, 'sharpe': round(sharpe, 3), 'cagr': round(cagr, 4),
            'max_dd': round(max_dd, 4), 'calmar': round(calmar, 3),
            'mean_exposure': round(float(np.mean(exposure)), 3),
        }

    # Strategy 6: Combined p_crisis + p_mid for broader risk signal
    mid_idx = None
    for raw_idx, label in label_map.items():
        if label == 'MID':
            mid_idx = raw_idx
            break
    if mid_idx is not None:
        p_risk = p_crisis + 0.5 * posteriors[:, mid_idx]  # crisis fully, mid half
        p_risk_smooth = pd.Series(p_risk).ewm(span=4, adjust=False).mean().values
        combo_exposure = (1 - p_risk_smooth) ** 2
        combo_exposure = np.clip(combo_exposure, 0.05, 1.0)
        combo_ret = spy_ret * combo_exposure
    else:
        combo_exposure = smooth_exposure
        combo_ret = smooth_ret

    strategies = {
        'buy_hold': _stats(spy_ret, 'Buy & Hold SPY', np.ones(len(spy_ret))),
        'hard_threshold': _stats(hard_ret, 'Hard p_crisis>=0.70 -> 30%', hard_exposure),
        'soft_sizing': _stats(soft_ret, 'Soft (1-p_crisis)^2', soft_exposure),
        'smooth_sizing': _stats(smooth_ret, 'Smooth EMA(4) (1-p)^2', smooth_exposure),
        'combo_sizing': _stats(combo_ret, 'Combo (crisis+0.5*mid)', combo_exposure),
    }

    return strategies


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Regime Probabilistic Investigation')
    parser.add_argument('--data-file', type=Path,
                        default=Path('data/regime_features.parquet'))
    parser.add_argument('--hmm-states', type=int, nargs='+', default=[2, 3, 4],
                        help='Number of HMM states to test')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/regime_investigation'))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Regime Probabilistic Investigation — HMM + Transition Matrix")
    print("=" * 70)

    # ── Load data ──
    print(f"\nLoading: {args.data_file}")
    df = pd.read_parquet(args.data_file)
    print(f"  Shape: {df.shape}")
    print(f"  Dates: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")

    # ══════════════════════════════════════════════════════════════════
    # PART A: Rule-Based V7 State Reconstruction
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(" PART A: V7 Rule-Based State Reconstruction")
    print(f"{'='*60}")

    rule_df = reconstruct_v7_states(df)

    # Filter out warmup period
    valid = rule_df[rule_df['state'] != 'WARMUP']
    print(f"  Valid days (after MA200 warmup): {len(valid)}")

    state_counts = valid['state'].value_counts()
    print(f"\n  State distribution:")
    for state, count in state_counts.items():
        pct = count / len(valid) * 100
        print(f"    {state:20s}: {count:4d} days ({pct:5.1f}%)")

    # Combined state (CRISIS sub-states)
    valid_copy = valid.copy()
    valid_copy['full_state'] = valid_copy.apply(
        lambda r: f"CRISIS_{r['sub_state']}" if r['state'] == 'CRISIS' and r['sub_state']
        else r['state'], axis=1)
    full_counts = valid_copy['full_state'].value_counts()
    print(f"\n  Full state distribution (with CRISIS sub-states):")
    for state, count in full_counts.items():
        pct = count / len(valid_copy) * 100
        print(f"    {state:20s}: {count:4d} days ({pct:5.1f}%)")

    # ── Transition Matrix (rule-based) ──
    print(f"\n  --- Transition Matrix (Rule-Based, 4-state) ---")
    rule_states_list = valid['state'].tolist()
    tm, counts = compute_transition_matrix(rule_states_list)
    print(f"\n  Transition probabilities P(row -> col):")
    print(tm.to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"\n  Raw transition counts:")
    for (s1, s2), cnt in sorted(counts.items()):
        if cnt > 0:
            print(f"    {s1:20s} -> {s2:20s}: {cnt:4d}")

    # ── Full state transition matrix (with CRISIS sub-states) ──
    print(f"\n  --- Transition Matrix (Full 6-state) ---")
    full_states_list = valid_copy['full_state'].tolist()
    tm_full, counts_full = compute_transition_matrix(full_states_list)
    print(f"\n  Transition probabilities:")
    print(tm_full.to_string(float_format=lambda x: f"{x:.3f}"))

    # ── State Durations ──
    print(f"\n  --- State Durations (4-state) ---")
    dur_stats = compute_state_durations(rule_states_list)
    for state, stats in sorted(dur_stats.items()):
        print(f"\n  {state}:")
        print(f"    Episodes: {stats['n_episodes']}, Total days: {stats['total_days']}")
        print(f"    Mean: {stats['mean_duration']:.1f}d, Median: {stats['median_duration']:.0f}d, "
              f"Std: {stats['std_duration']:.1f}d")
        print(f"    Range: [{stats['min_duration']}, {stats['max_duration']}]d")
        print(f"    Durations: {stats['durations']}")

    # ── Conditional Persistence ──
    print(f"\n  --- Conditional Persistence P(still in state at t+k | entered at t) ---")
    persistence = conditional_persistence(rule_states_list, max_lag=20)
    for state, persis in sorted(persistence.items()):
        if persis:
            vals = [f"t+{p['lag']}={p['p_persist']:.2f}({p['n_obs']})" for p in persis[:10]]
            print(f"\n  {state}: {', '.join(vals)}")

    # ══════════════════════════════════════════════════════════════════
    # PART B: HMM Training and Comparison
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(" PART B: HMM Training (Gaussian, full covariance)")
    print(f"{'='*60}")

    hmm_features = prepare_hmm_features(df)
    print(f"  HMM features shape: {hmm_features.shape}")
    print(f"  Features: {list(hmm_features.columns)}")

    # Align HMM dates with rule-based dates
    hmm_dates = hmm_features.index
    rule_aligned = valid[valid['date'].isin(hmm_dates)].set_index('date')

    hmm_results = {}
    for n_states in args.hmm_states:
        print(f"\n  --- {n_states}-State HMM ---")
        model, label_map, scaler, labels, posteriors, means = train_hmm_nstates(
            hmm_features, n_states)

        # Model characteristics
        print(f"  Converged: {model.monitor_.converged}")
        print(f"  Log-likelihood: {model.score(scaler.transform(hmm_features.values)):.1f}")
        print(f"  BIC approx: {-2*model.score(scaler.transform(hmm_features.values))*len(hmm_features) + n_states*(n_states+4+4*4)*np.log(len(hmm_features)):.0f}")

        print(f"\n  State means (original scale, sorted by vol ascending):")
        # means is already sorted by vol ascending (SAFE -> ... -> CRISIS)
        # label_names[i] directly corresponds to means[i]
        if n_states == 2:
            ordered_names = ["SAFE", "CRISIS"]
        elif n_states == 3:
            ordered_names = ["SAFE", "MID", "CRISIS"]
        elif n_states == 4:
            ordered_names = ["SAFE", "MID_LOW", "MID_HIGH", "CRISIS"]
        else:
            ordered_names = [f"S{i}" for i in range(n_states)]
        for i in range(n_states):
            m = means[i]
            print(f"    {ordered_names[i]:12s}: logret={m[0]:.5f}, vol_10d={m[1]:.5f}, "
                  f"vix_z={m[2]:.3f}, risk_prem_z={m[3]:.3f}")

        # State distribution
        label_counts = Counter(labels)
        print(f"\n  HMM state distribution:")
        for state, count in sorted(label_counts.items()):
            pct = count / len(labels) * 100
            print(f"    {state:10s}: {count:4d} days ({pct:5.1f}%)")

        # Compare with rule-based
        common_dates = hmm_dates.intersection(rule_aligned.index)
        if len(common_dates) > 0:
            hmm_on_common = [labels[list(hmm_dates).index(d)] for d in common_dates]
            rule_on_common = rule_aligned.loc[common_dates, 'state']

            comparison = compare_hmm_vs_rulebased(hmm_on_common, rule_on_common, common_dates)
            print(f"\n  HMM vs Rule-Based comparison:")
            print(f"    Agreement: {comparison['agreement_rate']:.1%} "
                  f"({comparison['n_agree']}/{comparison['n_total']})")
            print(f"    Divergences: {comparison['n_diverge']}")
            print(f"    Confusion (HMM_simple, Rule_simple) -> count:")
            for (h, r), cnt in sorted(comparison['confusion'].items()):
                print(f"      ({h}, {r}): {cnt}")

            if comparison['divergences']:
                print(f"\n    Notable divergences (first 10):")
                for div in comparison['divergences'][:10]:
                    print(f"      {div['date'].date()}: HMM={div['hmm']:10s} vs Rule={div['rule']:15s}")

        # HMM transition matrix
        print(f"\n  HMM Transition Matrix:")
        hmm_tm, _ = compute_transition_matrix(labels)
        print(hmm_tm.to_string(float_format=lambda x: f"{x:.3f}"))

        # HMM state durations
        hmm_dur = compute_state_durations(labels)
        print(f"\n  HMM State Durations:")
        for state, stats in sorted(hmm_dur.items()):
            print(f"    {state:10s}: mean={stats['mean_duration']:.1f}d, "
                  f"median={stats['median_duration']:.0f}d, "
                  f"episodes={stats['n_episodes']}, range=[{stats['min_duration']},{stats['max_duration']}]")

        hmm_results[f'HMM_{n_states}'] = {
            'n_states': n_states,
            'label_map': {str(k): v for k, v in label_map.items()},
            'state_distribution': dict(label_counts),
            'agreement_rate': comparison['agreement_rate'] if len(common_dates) > 0 else None,
            'durations': {s: {k: v for k, v in d.items() if k != 'durations'}
                          for s, d in hmm_dur.items()},
        }

    # ══════════════════════════════════════════════════════════════════
    # PART C: Soft Position Sizing Evaluation
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(" PART C: Soft Position Sizing (3-State HMM)")
    print(f"{'='*60}")

    # Use the 3-state HMM for sizing evaluation
    model3, lmap3, scaler3, labels3, post3, means3 = train_hmm_nstates(hmm_features, 3)
    sizing = evaluate_soft_sizing(df, post3, labels3, lmap3, hmm_dates)

    print(f"\n  {'Strategy':<35s} {'Sharpe':>7s} {'CAGR':>8s} {'MaxDD':>8s} {'Calmar':>8s} {'AvgExp':>8s}")
    print(f"  {'-'*35} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for key, s in sizing.items():
        if 'error' in s:
            continue
        print(f"  {s['label']:<35s} {s['sharpe']:7.3f} {s['cagr']:8.2%} "
              f"{s['max_dd']:8.2%} {s['calmar']:8.3f} {s['mean_exposure']:8.1%}")

    # ══════════════════════════════════════════════════════════════════
    # PART D: Transition Asymmetry Analysis
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(" PART D: Transition Asymmetry & Insights")
    print(f"{'='*60}")

    # Key asymmetries
    print(f"\n  Key findings:")
    if 'RISK_ON' in tm.index and 'RISK_OFF' in tm.columns:
        p_on_to_off = tm.loc['RISK_ON', 'RISK_OFF']
        print(f"    P(RISK_ON -> RISK_OFF):    {p_on_to_off:.4f} (per-day probability)")
        if 'RISK_OFF' in tm.index and 'RISK_ON' in tm.columns:
            p_off_to_on = tm.loc['RISK_OFF', 'RISK_ON']
            print(f"    P(RISK_OFF -> RISK_ON):    {p_off_to_on:.4f}")
            if p_on_to_off > 0:
                ratio = p_off_to_on / p_on_to_off
                print(f"    Asymmetry ratio (exit/enter): {ratio:.2f}x")

    if 'RISK_ON' in tm.index:
        p_self = tm.loc['RISK_ON', 'RISK_ON']
        expected_dur = 1 / (1 - p_self) if p_self < 1 else float('inf')
        print(f"    RISK_ON self-loop:          {p_self:.4f} (expected duration: {expected_dur:.0f}d)")

    if 'CRISIS' in tm.index:
        p_self = tm.loc['CRISIS', 'CRISIS']
        expected_dur = 1 / (1 - p_self) if p_self < 1 else float('inf')
        print(f"    CRISIS self-loop:           {p_self:.4f} (expected duration: {expected_dur:.0f}d)")

    # Stationary distribution
    print(f"\n  Stationary distribution (rule-based):")
    try:
        tm_arr = tm.values
        # Solve π = π × P
        eigenvalues, eigenvectors = np.linalg.eig(tm_arr.T)
        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        for state, prob in zip(tm.index, stationary):
            print(f"    {state:20s}: {prob:.4f}")
    except Exception as e:
        print(f"    (Could not compute: {e})")

    # ══════════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════════
    rule_df.to_csv(args.output_dir / 'v7_states_reconstructed.csv', index=False)

    summary = {
        'rule_based': {
            'state_distribution': state_counts.to_dict(),
            'full_state_distribution': full_counts.to_dict(),
            'transition_matrix_4state': tm.to_dict(),
            'durations': {s: {k: v for k, v in d.items() if k != 'durations'}
                          for s, d in dur_stats.items()},
        },
        'hmm': hmm_results,
        'soft_sizing': sizing,
    }

    def _json_safe(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, pd.Timestamp):
            return str(obj.date())
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, default=_json_safe), encoding='utf-8')

    print(f"\n  Results saved to {args.output_dir}/")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
