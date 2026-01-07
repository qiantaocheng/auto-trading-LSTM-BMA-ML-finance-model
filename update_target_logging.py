# -*- coding: utf-8 -*-
from pathlib import Path

path = Path('bma_models/量化模型_bma_ultra_enhanced.py')
text = path.read_text(encoding='utf-8')
old = "                      logger.info(f\"[二层] 使用增强版对齐器成功对齐: {alignment_report}\")\n\n                  except Exception as e:\n                      logger.warning(f\"[二层] ⚠️ 所有对齐器失败，使用基础回退: {e}\")\n\n                      # 基础回退：手动构建stacker_data"
new = """                      logger.info(f"[二层] 使用增强版对齐器成功对齐: {alignment_report}")\n\n                      if 'ret_fwd_10d' in stacker_data.columns:\n                          target_stats = stacker_data['ret_fwd_10d'].describe()\n                          nonzero = int((stacker_data['ret_fwd_10d'] != 0).sum())\n                          logger.info("[二层] Ridge目标统计: "
                                      f"mean={target_stats['mean']:.6f}, "
                                      f"std={target_stats['std']:.6f}, "
                                      f"min={target_stats['min']:.6f}, "
                                      f"max={target_stats['max']:.6f}, "
                                      f"nonzero={nonzero}/{len(stacker_data)}")\n\n                  except Exception as e:\n                      logger.warning(f"[二层] ⚠️ 所有对齐器失败，使用基础回退: {e}")\n\n                      # 基础回退：手动构建stacker_data"""

if old not in text:
    raise SystemExit('target logging anchor not found')

text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
