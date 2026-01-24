# OBV_DIVERGENCE 80/20 时间分割对比评估

## 脚本说明

`compare_obv_divergence_8020_split.py` - 使用 80/20 时间分割评估对比 obv_divergence 因子的影响

## 功能

1. **数据采样**: 从 MultiIndex DataFrame 中采样 1/5 (20%) 的 tickers
2. **80/20 时间分割**: 
   - 训练集: 前 80% 的日期
   - 测试集: 后 20% 的日期
3. **对比实验**:
   - 实验1: 包含 obv_divergence 因子
   - 实验2: 不包含 obv_divergence 因子（临时修改因子文件）
4. **结果对比**: 提取并对比关键指标（IC, Rank IC, Win Rate, Avg Return）

## 使用方法

```bash
python scripts/compare_obv_divergence_8020_split.py
```

## 数据要求

- 数据文件: `data/factor_exports/polygon_factors_all_filtered_clean_final_v2.parquet`
- 或子集文件: `data/factor_exports/polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet`
- 数据格式: MultiIndex(date, ticker)

## 输出

- 结果目录: `results/obv_divergence_8020_comparison/`
- 对比文件: `comparison_YYYYMMDD_HHMMSS.json`

## 对比指标

- **IC (Information Coefficient)**: 信息系数
- **Rank IC**: 排序信息系数
- **Win Rate**: 胜率
- **Avg Return**: 平均收益率
- **时间差异**: 训练时间对比

## 注意事项

1. 脚本会临时修改 `bma_models/simple_25_factor_engine.py` 来移除 obv_divergence
2. 修改后会自动恢复原文件
3. 如果脚本异常退出，请手动检查并恢复因子文件

## 示例输出

```json
{
  "timestamp": "2026-01-24T...",
  "tickers_used": 50,
  "with_obv_divergence": {
    "success": true,
    "elapsed_time_minutes": 15.5,
    "metrics": {
      "ic": 0.023,
      "rank_ic": 0.031,
      "win_rate": 52.5
    }
  },
  "without_obv_divergence": {
    "success": true,
    "elapsed_time_minutes": 14.8,
    "metrics": {
      "ic": 0.019,
      "rank_ic": 0.028,
      "win_rate": 51.2
    }
  },
  "difference": {
    "ic": 0.004,
    "rank_ic": 0.003,
    "win_rate": 1.3
  }
}
```
