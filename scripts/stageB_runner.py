import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_PATH = Path(r"D:\trade\data\factor_exports\polygon_factors_subset_1_5.parquet")
RESULTS_ROOT = Path(r"D:\trade\results\t10_time_split_80_20_final")
STAGE_DIR = RESULTS_ROOT / "stageB_stageA_runs"
HISTORY_PATH = RESULTS_ROOT / "feature_history.csv"
FRAMEWORK_SCRIPT = Path(r"D:\trade\scripts\feature_search_framework.py")
LOG_PATH = RESULTS_ROOT / "feature_search_activity.log"

BASE_FEATURES = ['hist_vol_20','momentum_10d','trend_r2_60','near_52w_high','rsi_21','vol_ratio_20d','5_days_reversal','alpha_linreg_corr_20d']
COMBOS = [
    ("combo_A0_base", []),
    ("combo_A0_obv_divergence", ['obv_divergence']),
    ("combo_A0_obv_momentum", ['obv_momentum_60d']),
    ("combo_A0_liquid_momentum", ['liquid_momentum']),
    ("combo_A0_momentum_60d", ['momentum_60d']),
    ("combo_A0_price_ma60", [])
]

BASE_CMD = [
    "python", str(Path("scripts") / "time_split_80_20_oos_eval.py"),
    "--train-data", str(DATA_PATH),
    "--data-file", str(DATA_PATH),
    "--horizon-days", "10",
    "--split", "0.8",
    "--models", "elastic_net", "xgboost", "catboost", "lightgbm_ranker", "lambdarank", "ridge_stacking",
    "--model", "ridge_stacking",
    "--top-n", "20",
    "--cost-bps", "0",
    "--benchmark", "QQQ",
    "--max-weeks", "260",
    "--hac-method", "newey-west",
    "--hac-lag", "20",
    "--log-level", "INFO"
]

REWARD_WEIGHTS = {'top_1_10': 1.0, 'top_11_20': -0.2, 'top_21_30': -1.0}

def log(msg):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with LOG_PATH.open('a', encoding='utf-8') as fh:
        fh.write(f"[{stamp}] {msg}\n")
    print(msg)

def compute_reward(row):
    return row['avg_top_1_10_return'] * REWARD_WEIGHTS['top_1_10'] - 0.2 * row['avg_top_11_20_return'] - row['avg_top_21_30_return']

def append_history(label, run_dir, features):
    entries = []
    for model in ('lambdarank','ridge_stacking'):
        summary = run_dir / f"{model}_bucket_summary.csv"
        row = pd.read_csv(summary).iloc[0]
        entries.append({
            'run': label,
            'model': model,
            'top_1_10': row['avg_top_1_10_return'],
            'top_11_20': row['avg_top_11_20_return'],
            'top_21_30': row['avg_top_21_30_return'],
            'reward': compute_reward(row),
            'features': json.dumps(features)
        })
    hist = pd.read_csv(HISTORY_PATH) if HISTORY_PATH.exists() else pd.DataFrame(columns=entries[0].keys())
    hist = pd.concat([hist, pd.DataFrame(entries)], ignore_index=True)
    hist.to_csv(HISTORY_PATH, index=False)
    log(f"History updated for {label}")

def run_combo(name, extra_feats):
    features = sorted(set(BASE_FEATURES + extra_feats))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parent = STAGE_DIR / f"{name}_{timestamp}"
    parent.mkdir(parents=True, exist_ok=True)
    cmd = BASE_CMD + ["--features", *features, "--output-dir", str(parent)]
    log(f"Launching StageB {name} with features: {', '.join(features)}")
    subprocess.run(cmd, check=True)
    run_dirs = [p for p in parent.iterdir() if p.is_dir()]
    final_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
    append_history(f"stageB_{name}", final_dir, features)
    subprocess.run(["python", str(FRAMEWORK_SCRIPT)], check=True)
    log(f"StageB {name} completed -> {final_dir}")

if __name__ == '__main__':
    for combo_name, combo in COMBOS:
        try:
            run_combo(combo_name, combo)
        except subprocess.CalledProcessError as exc:
            log(f"Command failed for {combo_name}: {exc}")
            sys.exit(exc.returncode)
        except Exception as exc:
            log(f"Error in {combo_name}: {exc}")
            sys.exit(1)
