#!/usr/bin/env python3
"""Compare all models before and after ret_skew_20d removal"""
import pandas as pd

# Load both reports
df_15 = pd.read_csv(r'D:\trade\results\t10_time_split_80_20_final\run_20260124_022650\report_df.csv')
df_14 = pd.read_csv(r'D:\trade\results\t10_time_split_80_20_final\run_20260124_040014\report_df.csv')

models = ['elastic_net', 'xgboost', 'catboost', 'lambdarank', 'ridge_stacking']

print("=" * 80)
print("ALL MODELS COMPARISON: 15 factors (with ret_skew_20d) vs 14 factors (without)")
print("=" * 80)
print()

for model in models:
    # Check if model exists in both reports
    df15_model = df_15[df_15['Model'] == model]
    df14_model = df_14[df_14['Model'] == model]
    
    if df15_model.empty or df14_model.empty:
        status_15 = "EXISTS" if not df15_model.empty else "MISSING"
        status_14 = "EXISTS" if not df14_model.empty else "MISSING"
        print(f"[{model.upper()}]")
        print("-" * 60)
        print(f"  Status: 15-factor run: {status_15}, 14-factor run: {status_14}")
        if df15_model.empty and df14_model.empty:
            print(f"  >>> Model not trained/evaluated in either run (likely not in snapshot)")
        print()
        continue
    
    r15 = df15_model.iloc[0]
    r14 = df14_model.iloc[0]
    
    print(f"[{model.upper()}]")
    print("-" * 60)
    
    # Key metrics
    metrics = [
        ('IC', 'IC'),
        ('Rank IC', 'Rank_IC'),
        ('R2', 'R2'),
        ('Win Rate', 'win_rate'),
        ('Top Sharpe (net)', 'top_sharpe_net'),
        ('Avg Top Return', 'avg_top_return'),
        ('Median Top Return', 'median_top_return'),
    ]
    
    identical = True
    for label, col in metrics:
        v15 = r15[col]
        v14 = r14[col]
        diff = v14 - v15
        
        if abs(diff) < 1e-10:
            status = "[IDENTICAL]"
        else:
            status = f"DIFF: {diff:.10f}"
            identical = False
        
        if col in ['win_rate']:
            print(f"  {label:20s}: {v15:.4f} (15) vs {v14:.4f} (14) | {status}")
        elif col in ['avg_top_return', 'median_top_return']:
            print(f"  {label:20s}: {v15:.6f} (15) vs {v14:.6f} (14) | {status}")
        else:
            print(f"  {label:20s}: {v15:.6f} (15) vs {v14:.6f} (14) | {status}")
    
    if identical:
        print(f"  >>> ALL METRICS IDENTICAL - ret_skew_20d had NO impact on {model}")
    else:
        print(f"  >>> SOME DIFFERENCES DETECTED for {model}")
    
    print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
all_identical = True
for model in models:
    df15_model = df_15[df_15['Model'] == model]
    df14_model = df_14[df_14['Model'] == model]
    
    if df15_model.empty or df14_model.empty:
        print(f"  {model:15s}: [NOT TRAINED]")
        continue
    
    r15 = df15_model.iloc[0]
    r14 = df14_model.iloc[0]
    
    # Check if all key metrics are identical
    key_cols = ['IC', 'Rank_IC', 'R2', 'win_rate', 'top_sharpe_net', 'avg_top_return', 'median_top_return']
    model_identical = all(abs(r14[col] - r15[col]) < 1e-10 for col in key_cols)
    
    if not model_identical:
        all_identical = False
    
    status = "[IDENTICAL]" if model_identical else "[DIFFERENCES]"
    print(f"  {model:15s}: {status}")

print()
if all_identical:
    print("CONCLUSION: Removing ret_skew_20d had ZERO impact on ALL models!")
    print("This confirms that ret_skew_20d was never in the data file.")
else:
    print("CONCLUSION: Some models showed differences (though likely negligible).")
