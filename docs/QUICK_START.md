# Quick Start Guide - 80/20 Timesplit Evaluation

## Standard Command

The recommended command to run 80/20 timesplit evaluation:

```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "data/factor_exports/polygon_factors_all_filtered.parquet" \
  --horizon-days 10 \
  --split 0.8 \
  --models elastic_net xgboost catboost lightgbm_ranker lambdarank ridge_stacking \
  --model ridge_stacking \
  --top-n 20 \
  --cost-bps 10 \
  --benchmark QQQ \
  --output-dir "results/t10_time_split_80_20_new_params" \
  --log-level INFO
```

## What This Does

1. **Loads Data**: Reads MultiIndex parquet file (`data/factor_exports/polygon_factors_all_filtered.parquet`)
2. **Time Split**: 
   - Training: First 80% of dates
   - Testing: Last 20% of dates
   - Purge gap: 10 days (prevents label leakage)
3. **Training Phase**: Trains all models on training window
   - ElasticNet, XGBoost, CatBoost, LightGBM Ranker, LambdaRank (first layer)
   - MetaRankerStacker (second layer, legacy name: ridge_stacking)
4. **Testing Phase**: Evaluates models on test window (OOS)
5. **Generates Outputs**: Metrics, plots, and reports

## Output Location

Results are saved to:
```
results/t10_time_split_80_20_new_params/run_YYYYMMDD_HHMMSS/
├── snapshot_id.txt                    # Model snapshot ID
├── report_df.csv                      # Performance summary
├── oos_metrics.csv                    # Out-of-sample metrics
├── ridge_stacking_top20_vs_qqq.png    # Top-20 vs QQQ plot
├── ridge_stacking_top20_vs_qqq_cumulative.png
└── ... (other model outputs)
```

## Key Parameters

- `--data-file`: Path to MultiIndex parquet file (date, ticker format)
- `--horizon-days 10`: T+10 prediction horizon
- `--split 0.8`: 80/20 train/test split
- `--models`: List of models to evaluate
- `--model ridge_stacking`: Primary model (uses MetaRankerStacker internally)
- `--top-n 20`: Evaluate Top-20 stocks
- `--cost-bps 10`: Transaction cost (10 basis points)
- `--benchmark QQQ`: Benchmark ticker
- `--output-dir`: Output directory for results
- `--log-level INFO`: Logging verbosity

## Prediction Only Mode

If you already have a trained snapshot and want to run predictions only:

```bash
python scripts/time_split_80_20_oos_eval.py \
  --snapshot-id <your_snapshot_id> \
  --data-file "data/factor_exports/polygon_factors_all_filtered.parquet" \
  --horizon-days 10 \
  --split 0.8 \
  --models elastic_net xgboost catboost lightgbm_ranker lambdarank ridge_stacking \
  --model ridge_stacking \
  --top-n 20 \
  --output-dir "results/prediction_<timestamp>"
```

## Notes

- The script uses **MetaRankerStacker** internally (not RidgeStacker, despite legacy naming)
- All models are trained with PurgedCV to prevent temporal leakage
- Results include IC, Rank IC, NDCG@10, NDCG@30, and bucket returns
- Transaction costs are applied at each rebalance
