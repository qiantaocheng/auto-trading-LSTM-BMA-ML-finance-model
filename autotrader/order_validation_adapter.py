
# 订单验证适配器
from typing import Dict, Any, Tuple, Optional
import logging

class OrderValidationAdapter:
    """订单验证适配器"""
    
    def __init__(self):
        self.logger = logging.getLogger('order_validator')
        self.min_order_value = 1000.0
        self.max_position_pct = 0.15
        
    def validate_order(self, order_data: Dict[str, Any]) -> Tuple[bool, str]:
        """验证订单"""
        try:
            # 基本字段检查
            required_fields = ['symbol', 'action', 'quantity', 'price']
            for field in required_fields:
                if field not in order_data:
                    return False, f"缺少必需字段: {field}"
            
            # 数量检查
            quantity = order_data.get('quantity', 0)
            if quantity <= 0:
                return False, "订单数量必须大于0"
                
            # 价格检查  
            price = order_data.get('price', 0)
            if price <= 0:
                return False, "订单价格必须大于0"
                
            # 订单价值检查
            order_value = quantity * price
            if order_value < self.min_order_value:
                return False, f"订单价值 ${order_value:.2f} 低于最小值 ${self.min_order_value}"
                
            return True, "订单验证通过"
            
        except Exception as e:
            self.logger.error(f"订单验证错误: {e}")
            return False, f"验证异常: {e}"
    
    def validate_position_size(self, symbol: str, quantity: int, 
                             portfolio_value: float) -> Tuple[bool, str]:
        """验证仓位大小"""
        try:
            # 这里需要获取当前价格，简化处理
            estimated_position_value = quantity * 100  # 简化估算
            position_pct = estimated_position_value / portfolio_value
            
            if position_pct > self.max_position_pct:
                return False, f"仓位占比 {position_pct:.2%} 超过最大限制 {self.max_position_pct:.2%}"
                
            return True, "仓位验证通过"
            
        except Exception as e:
            return False, f"仓位验证错误: {e}"

# 全局验证器实例
_order_validator = None

def get_order_validator():
    """获取订单验证器实例"""
    global _order_validator
    if _order_validator is None:
        _order_validator = OrderValidationAdapter()
    return _order_validator
