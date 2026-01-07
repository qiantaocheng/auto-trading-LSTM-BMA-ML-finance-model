# -*- coding: utf-8 -*-
from pathlib import Path
text = Path('bma_models/量化模型_bma_ultra_enhanced.py').read_text(encoding='utf-8')
needle = "logger.info(f\"[二层] 使用增强版对齐器成功对齐: {alignment_report}\")"
idx = text.find(needle)
print(idx)
print(text[idx:idx+200])
