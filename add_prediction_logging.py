# -*- coding: utf-8 -*-
from pathlib import Path

path = Path('bma_models/量化模型_bma_ultra_enhanced.py')
lines = path.read_text(encoding='utf-8').split('\n')
result = []
inserted = False
for line in lines:
    result.append(line)
    if 'first_layer_preds = pd.DataFrame(index=feature_data.index)' in line and not inserted:
        result.append('            logger.info("[预测] 初始化第一层预测容器")')
        inserted = True
    if "logger.info(f\"标准化预测完成"" in line and inserted:
        # insert stats after standardized predictions filled
        result.append('                    for col in first_layer_preds.columns:\n                        col_std = first_layer_preds[col].std(skipna=True)\n                        col_mean = first_layer_preds[col].mean(skipna=True)\n                        logger.info(f"[预测] 第一层列 {col}: mean={col_mean:.6f}, std={col_std:.6f}")')
        inserted = False

path.write_text('\n'.join(result), encoding='utf-8')
