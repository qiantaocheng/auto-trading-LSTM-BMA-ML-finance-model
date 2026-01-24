# Bucket Returns Table - 80/20 Time Split Evaluation

## 📊 Bucket Returns Summary (Previous Run - Without Sato Factors)

### CatBoost Model

| Bucket | Avg Return | Median Return | Avg Return (vs Median) | Median Return (vs Median) |
|--------|------------|--------------|------------------------|--------------------------|
| **Top 1-10** | 2.18% | 1.88% | 1.07% | 0.33% |
| **Top 5-15** | 1.55% | 1.83% | 0.74% | 0.44% |
| **Top 11-20** | 1.48% | 1.90% | 0.47% | 0.87% |
| **Top 21-30** | 2.70% | 3.38% | 1.50% | 1.52% |
| **Bottom 1-10** | -0.77% | 0.51% | -0.27% | 0.00% |
| **Bottom 11-20** | -0.42% | 0.16% | -0.27% | 0.00% |
| **Bottom 21-30** | -0.26% | 0.27% | -0.14% | 0.09% |

**Key Metrics:**
- Top 20 Avg Return: 2.12% (net: 2.02%)
- Top 20 Median Return: 1.43% (net: 1.33%)
- Sharpe Ratio (Net): 1.54
- Win Rate: 64%

---

### LambdaRank Model

| Bucket | Avg Return | Median Return | Avg Return (vs Median) | Median Return (vs Median) |
|--------|------------|--------------|------------------------|--------------------------|
| **Top 1-10** | 1.92% | 2.34% | 0.79% | 0.70% |
| **Top 5-15** | 2.48% | 2.91% | 1.52% | 1.76% |
| **Top 11-20** | 2.13% | 2.79% | 1.03% | 1.30% |
| **Top 21-30** | 1.29% | 1.48% | 0.39% | 0.39% |
| **Bottom 1-10** | -0.04% | 0.11% | -0.04% | 0.00% |
| **Bottom 11-20** | -0.06% | 0.03% | -0.06% | 0.00% |
| **Bottom 21-30** | 0.07% | 0.19% | 0.00% | 0.09% |

**Key Metrics:**
- Top 20 Avg Return: 1.78% (net: 1.77%)
- Top 20 Median Return: 1.75% (net: 1.75%)
- Sharpe Ratio (Net): 0.69
- Win Rate: 56%

---

### Ridge Stacking Model

| Bucket | Avg Return | Median Return | Avg Return (vs Median) | Median Return (vs Median) |
|--------|------------|--------------|------------------------|--------------------------|
| **Top 1-10** | 0.97% | 1.03% | -0.02% | 0.10% |
| **Top 5-15** | 1.20% | 2.36% | 0.09% | 0.73% |
| **Top 11-20** | 0.80% | 1.26% | -0.46% | -0.16% |
| **Top 21-30** | 1.03% | 0.97% | -0.51% | -0.12% |
| **Bottom 1-10** | 0.34% | 0.66% | 0.00% | 0.00% |
| **Bottom 11-20** | 0.78% | 0.84% | 0.00% | 0.00% |
| **Bottom 21-30** | -0.55% | -0.41% | -0.11% | 0.07% |

**Key Metrics:**
- Top 20 Avg Return: 0.93% (net: 0.78%)
- Top 20 Median Return: 0.38% (net: 0.38%)
- Sharpe Ratio (Net): 0.24
- Win Rate: 48%

---

## 📈 Detailed Bucket Returns from report_df.csv

### All Models Comparison

| Model | Top 1-10 Avg | Top 1-10 Median | Top 5-15 Avg | Top 5-15 Median | Top 11-20 Avg | Top 11-20 Median | Top 21-30 Avg | Top 21-30 Median |
|------|--------------|-----------------|--------------|-----------------|---------------|------------------|---------------|------------------|
| **CatBoost** | 2.18% | 1.88% | 1.55% | 1.83% | 1.48% | 1.90% | 2.70% | 3.38% |
| **LambdaRank** | 1.92% | 2.34% | 2.48% | 2.91% | 2.13% | 2.79% | 1.29% | 1.48% |
| **Ridge Stacking** | 0.97% | 1.03% | 1.20% | 2.36% | 0.80% | 1.26% | 1.03% | 0.97% |
| **XGBoost** | 0.78% | 0.73% | 0.73% | 1.49% | 0.25% | 0.56% | 0.60% | 0.66% |
| **ElasticNet** | 0.34% | 0.69% | -0.04% | 0.27% | 0.05% | 0.35% | 0.49% | 0.60% |

### Bottom Buckets Comparison

| Model | Bottom 1-10 Avg | Bottom 1-10 Median | Bottom 11-20 Avg | Bottom 11-20 Median | Bottom 21-30 Avg | Bottom 21-30 Median |
|------|-----------------|---------------------|------------------|---------------------|------------------|---------------------|
| **CatBoost** | -0.77% | 0.51% | -0.42% | 0.16% | -0.26% | 0.27% |
| **LambdaRank** | -0.04% | 0.11% | -0.06% | 0.03% | 0.07% | 0.19% |
| **Ridge Stacking** | 0.34% | 0.66% | 0.78% | 0.84% | -0.55% | -0.41% |
| **XGBoost** | -0.16% | 0.00% | 0.00% | 0.00% | 0.01% | 0.00% |
| **ElasticNet** | -0.13% | -0.27% | -0.02% | -0.02% | -0.02% | -0.02% |

---

## 🔍 Key Observations

### Top Buckets
1. **CatBoost** shows strong performance in Top 1-10 (2.18%) and Top 21-30 (2.70%)
2. **LambdaRank** performs best in Top 5-15 (2.48%) and Top 11-20 (2.13%)
3. **Ridge Stacking** shows moderate performance across all top buckets

### Bottom Buckets
1. **CatBoost** shows good separation with negative returns in bottom buckets
2. **LambdaRank** has minimal negative returns, suggesting less clear separation
3. **Ridge Stacking** shows mixed results with some positive returns in bottom buckets

### Overall Performance
- **Best Top 20 Performance**: CatBoost (2.12% avg return, 1.54 Sharpe)
- **Best IC**: LambdaRank (0.033)
- **Best Win Rate**: CatBoost (64%)

---

## 📝 Note

These results are from the **previous 80/20 evaluation without Sato factors** (run_20260120_041850).

**New results with Sato factors** will be available in:
- `results/t10_time_split_80_20_sato/run_<timestamp>/`

The new bucket tables will show the impact of including Sato factors in the models.
