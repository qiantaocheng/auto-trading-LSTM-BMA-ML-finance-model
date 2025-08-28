
# BMA预测校准器适配器
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression

class PredictionCalibratorAdapter:
    """预测校准器适配器"""
    
    def __init__(self):
        self.logger = logging.getLogger('prediction_calibrator')
        self.calibrators = {}
        self.calibration_history = {}
        
    def calibrate_predictions(self, predictions: np.ndarray, 
                            actuals: np.ndarray = None, 
                            model_name: str = "default") -> np.ndarray:
        """校准预测值"""
        try:
            if model_name not in self.calibrators and actuals is not None:
                # 首次校准：训练校准器
                self._train_calibrator(predictions, actuals, model_name)
            
            if model_name in self.calibrators:
                # 应用校准
                calibrated = self.calibrators[model_name].transform(predictions.reshape(-1, 1)).flatten()
                return calibrated
            else:
                # 无校准器时返回原预测
                return predictions
                
        except Exception as e:
            self.logger.error(f"预测校准错误: {e}")
            return predictions
    
    def _train_calibrator(self, predictions: np.ndarray, actuals: np.ndarray, model_name: str):
        """训练校准器"""
        try:
            if len(predictions) >= 10:  # 需要足够的样本
                # 使用等渗回归进行校准
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(predictions, actuals)
                self.calibrators[model_name] = calibrator
                
                # 记录校准历史
                self.calibration_history[model_name] = {
                    'sample_count': len(predictions),
                    'train_time': np.datetime64('now'),
                    'calibrated': True
                }
                
                self.logger.info(f"模型 {model_name} 校准器训练完成，样本数: {len(predictions)}")
            
        except Exception as e:
            self.logger.error(f"校准器训练错误: {e}")
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """获取校准状态"""
        return {
            'calibrated_models': list(self.calibrators.keys()),
            'calibration_history': self.calibration_history,
            'status': 'active'
        }

# 全局校准器实例
_calibrator = None

def get_prediction_calibrator():
    """获取预测校准器实例"""
    global _calibrator
    if _calibrator is None:
        _calibrator = PredictionCalibratorAdapter()
    return _calibrator
