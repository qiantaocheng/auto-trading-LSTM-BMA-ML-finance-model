# Sato Factor Evaluation Fix

## 🔧 Problem
The `results/sato` directory was empty because the evaluation was not run with the correct output directory.

## ✅ Solution
Created `scripts/run_sato_evaluation.py` to properly run the 80/20 evaluation with Sato factors and save results to `results/sato/`.

## 🚀 How to Run

### Option 1: Use the convenience script (Recommended)
```bash
python scripts/run_sato_evaluation.py
```

This script will:
1. Create `results/sato/` directory if it doesn't exist
2. Run 80/20 time split evaluation with Sato factors
3. Save all results to `results/sato/run_<timestamp>/`

### Option 2: Run directly with correct output directory
```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
  --horizon-days 10 \
  --split 0.8 \
  --models catboost lambdarank ridge_stacking \
  --top-n 20 \
  --output-dir "results/sato" \
  --log-level INFO
```

## 📊 Expected Output

After the evaluation completes, you should see results in:
- `results/sato/run_<timestamp>/snapshot_id.txt`
- `results/sato/run_<timestamp>/report_df.csv`
- `results/sato/run_<timestamp>/*_bucket_summary.csv`
- `results/sato/run_<timestamp>/*_bucket_returns.csv`
- `results/sato/run_<timestamp>/*.png` (plots)

## ⏱️ Evaluation Status

The evaluation is currently running in the background. Check progress by:
1. Monitoring the terminal output
2. Checking for new files in `results/sato/`
3. Looking for completion messages

## 🔍 Verify Results

Once complete, verify results exist:
```powershell
Get-ChildItem "results\sato" -Recurse -File | Sort-Object LastWriteTime -Descending
```
