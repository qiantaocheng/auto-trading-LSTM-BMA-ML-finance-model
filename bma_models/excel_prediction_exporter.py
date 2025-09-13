#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMA预测结果Excel输出模块 - 统一版本
重定向到 CorrectedPredictionExporter，旧版本已废弃
"""

import logging
logger = logging.getLogger(__name__)

# [DEPRECATED NOTICE]
logger.warning("🔄 excel_prediction_exporter.py is DEPRECATED - redirecting to CorrectedPredictionExporter")
logger.warning("📋 Please use 'from bma_models.corrected_prediction_exporter import CorrectedPredictionExporter' directly")

# 统一导入 CorrectedPredictionExporter
try:
    from bma_models.corrected_prediction_exporter import CorrectedPredictionExporter

    # 向后兼容别名
    BMAExcelExporter = CorrectedPredictionExporter

    def export_bma_predictions_to_excel(predictions, dates, tickers, model_info, output_dir="result", filename=None):
        """向后兼容的导出函数"""
        exporter = CorrectedPredictionExporter()
        return exporter.export_predictions(
            predictions=predictions,
            dates=dates,
            tickers=tickers,
            model_info=model_info,
            filename=filename
        )

    logger.info("✅ Unified to CorrectedPredictionExporter successfully")

except ImportError as e:
    logger.error(f"❌ Failed to import CorrectedPredictionExporter: {e}")
    # 紧急回退到 fixed 版本
    try:
        from bma_models.excel_prediction_exporter_fixed import (
            BMAExcelExporterFixed as BMAExcelExporter,
            export_bma_predictions_fixed as export_bma_predictions_to_excel
        )
        logger.warning("⚠️ Fallback to excel_prediction_exporter_fixed")
    except ImportError:
        logger.critical("❌ All Excel exporters failed to import!")
        raise

# 导出统一接口
__all__ = ['BMAExcelExporter', 'export_bma_predictions_to_excel', 'CorrectedPredictionExporter']