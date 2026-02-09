import ast
import pandas as pd
import numpy as np
from pathlib import Path
import math

OUTPUT_DIR = Path(r"D:\trade\results\t10_time_split_80_20_final")
HISTORY_PATH = OUTPUT_DIR / 'feature_history.csv'
REWARD_WEIGHTS = {'top_1_10': 1.0, 'top_11_20': -0.2, 'top_21_30': -1.0}
BASE_DELTA_THRESHOLD = 0.0015
SOFTMAX_TAU = 0.25
FAMILY_MAP = {
    'momentum': {'momentum_10d','momentum_60d','liquid_momentum'},
    'volatility': {'hist_vol_20','ivol_20','atr_ratio'},
    'obv': {'obv_momentum_60d','obv_divergence'},
}
TOTAL_FEATURE_TARGET = 14

if not HISTORY_PATH.exists():
    raise SystemExit('feature_history.csv not found - log runs first')

raw_df = pd.read_csv(HISTORY_PATH)
if raw_df.empty:
    raise SystemExit('feature_history.csv is empty')

# ensure features column parsed
raw_df['features'] = raw_df['features'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

lambda_df = raw_df[raw_df['model'] == 'lambdarank'].copy()
if lambda_df.empty:
    raise SystemExit('No LambdaRank rows in history')

feature_set = set()
for lst in lambda_df['features']:
    feature_set.update(lst)

feature_stats = []
for feat in sorted(feature_set):
    mask = lambda_df['features'].apply(lambda feats, f=feat: f in feats)
    included = lambda_df[mask]
    excluded = lambda_df[~mask]
    inclusion_rate = mask.mean()
    avg_reward = included['reward'].mean() if not included.empty else np.nan
    avg_top1 = included['top_1_10'].mean() if not included.empty else np.nan
    absent_top1 = excluded['top_1_10'].mean() if not excluded.empty else np.nan
    delta_top1 = (avg_top1 - absent_top1) if (not included.empty and not excluded.empty) else np.nan
    feature_stats.append({
        'feature': feat,
        'included_runs': len(included),
        'excluded_runs': len(excluded),
        'inclusion_rate': inclusion_rate,
        'avg_reward': avg_reward,
        'avg_top_1_10': avg_top1,
        'delta_top_1_10': delta_top1,
    })

feature_df = pd.DataFrame(feature_stats)
reward_mean = feature_df['avg_reward'].mean()
reward_std = feature_df['avg_reward'].std(ddof=1)
inclusion_mean = feature_df['inclusion_rate'].mean()
inclusion_std = feature_df['inclusion_rate'].std(ddof=1)

def zscore(value, mean, std):
    if std is None or std <= 1e-9 or pd.isna(value):
        return 0.0
    return (value - mean) / std

feature_df['reward_z'] = feature_df['avg_reward'].apply(lambda v: zscore(v, reward_mean, reward_std))
feature_df['inclusion_z'] = feature_df['inclusion_rate'].apply(lambda v: zscore(v, inclusion_mean, inclusion_std))
feature_df['usefulness_score'] = feature_df['reward_z'] + 0.6 * feature_df['inclusion_z']

if feature_df['usefulness_score'].isna().all():
    feature_df['usefulness_score'] = 0.0

logits = feature_df['usefulness_score'] / max(SOFTMAX_TAU, 1e-6)
exp_logits = np.exp(logits - logits.max())
feature_df['selection_probability'] = exp_logits / exp_logits.sum()

base_mask = (feature_df['delta_top_1_10'] >= BASE_DELTA_THRESHOLD) & (feature_df['inclusion_rate'] >= 0.6)
noise_mask = (feature_df['delta_top_1_10'] <= 0)
feature_df['classification'] = np.where(base_mask, 'base', np.where(noise_mask, 'noise', 'conditional'))

feature_df = feature_df.sort_values('usefulness_score', ascending=False)
feature_df.to_csv(OUTPUT_DIR / 'feature_usefulness.csv', index=False)

base_features = feature_df[feature_df['classification'] == 'base']['feature'].tolist()
conditional_features = feature_df[feature_df['classification'] == 'conditional']['feature'].tolist()
noise_features = feature_df[feature_df['classification'] == 'noise']['feature'].tolist()

TOTAL_FEATURE_TARGET = min(16, max(12, len(base_features) + 4))

def family_counts(current):
    counts = {fam: 0 for fam in FAMILY_MAP}
    for feat in current:
        for fam, members in FAMILY_MAP.items():
            if feat in members:
                counts[fam] += 1
    return counts

def can_add(feat, counts):
    for fam, members in FAMILY_MAP.items():
        if feat in members:
            limit = 2 if fam in ['momentum','volatility','obv'] else None
            if limit is not None and counts[fam] >= limit:
                return False
    return True

recommended = []
base_set = set(base_features)
base_counts = family_counts(base_set)

non_noise = feature_df[feature_df['classification'] != 'noise']
for variant in range(3):
    combo = set(base_set)
    counts = base_counts.copy()
    ordered = non_noise.sort_values('selection_probability', ascending=False)
    for _, row in ordered.iterrows():
        feat = row['feature']
        if feat in combo:
            continue
        if len(combo) >= TOTAL_FEATURE_TARGET:
            break
        if can_add(feat, counts):
            combo.add(feat)
            for fam, members in FAMILY_MAP.items():
                if feat in members:
                    counts[fam] += 1
                    break
    recommended.append(sorted(combo))
    non_noise = non_noise.sample(frac=1, random_state=variant)

md_lines = ['# Feature Search Status','', '## Base Features', ', '.join(base_features) or 'None', '', '## Conditional Feature Pool', ', '.join(conditional_features) or 'None', '', '## Noise / Excluded Features', ', '.join(noise_features) or 'None', '', '## Feature Usefulness Table', '', feature_df.to_markdown(index=False), '', '## Recommended Feature Combinations']
for idx, combo in enumerate(recommended, 1):
    md_lines.append(f"### Combo {idx}")
    md_lines.append(', '.join(combo))
    md_lines.append('')

status_path = OUTPUT_DIR / 'feature_search_status.md'
status_path.write_text('\n'.join(md_lines))
print(f'Updated {OUTPUT_DIR / 'feature_usefulness.csv'} and {status_path}')
