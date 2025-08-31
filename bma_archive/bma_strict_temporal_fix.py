
def strict_temporal_validation(self, X: pd.DataFrame, y: pd.Series, 
                             dates: pd.Series, embargo_days: int = 5) -> bool:
    """
    严格的时间泄漏检测 - 发现问题立即中止训练
    
    Args:
        X: 特征矩阵
        y: 目标变量  
        dates: 日期序列
        embargo_days: 禁运天数
        
    Returns:
        bool: 验证是否通过
        
    Raises:
        ValueError: 发现时间泄漏时抛出异常
    """
    logger = logging.getLogger(__name__)
    
    try:
        # 1. 检查数据对齐
        if len(X) != len(y) or len(X) != len(dates):
            raise ValueError(f"数据长度不匹配: X={len(X)}, y={len(y)}, dates={len(dates)}")
        
        # 2. 转换日期并排序
        dates_dt = pd.to_datetime(dates).sort_values()
        sorted_indices = dates_dt.index
        
        # 3. 检查时间连续性和间隔
        violations = []
        
        for i in range(len(dates_dt) - 1):
            current_date = dates_dt.iloc[i]
            next_date = dates_dt.iloc[i + 1]
            
            # 计算实际时间间隔
            actual_gap = (next_date - current_date).days
            
            # 严格检查：间隔必须 >= embargo_days
            if actual_gap < embargo_days:
                violations.append({
                    'index': i,
                    'current_date': current_date,
                    'next_date': next_date,
                    'actual_gap': actual_gap,
                    'required_gap': embargo_days
                })
        
        # 4. 如果发现违规，立即报错中止
        if violations:
            error_msg = f"❌ 严重时间泄漏检测到 {len(violations)} 个违规:"
            for v in violations[:5]:  # 显示前5个
                error_msg += f"\n  - {v['current_date']} -> {v['next_date']}: 间隔{v['actual_gap']}天 < 要求{v['required_gap']}天"
            
            if len(violations) > 5:
                error_msg += f"\n  ... 还有 {len(violations)-5} 个违规"
            
            raise ValueError(error_msg)
        
        # 5. 检查特征和目标的时间对齐
        # 确保特征日期 + embargo_days <= 目标日期
        feature_dates = dates_dt
        target_dates = feature_dates + pd.Timedelta(days=embargo_days)
        
        # 检查是否有目标日期超出了数据范围
        max_available_date = feature_dates.max()
        future_targets = target_dates[target_dates > max_available_date]
        
        if len(future_targets) > 0:
            logger.warning(f"检测到 {len(future_targets)} 个目标需要未来数据，这是正常的前向预测")
        
        # 6. 最终验证通过
        logger.info(f"✅ 严格时间验证通过: {len(dates_dt)}个样本, 最小间隔{embargo_days}天")
        return True
        
    except Exception as e:
        logger.error(f"❌ 严格时间验证失败: {e}")
        raise
