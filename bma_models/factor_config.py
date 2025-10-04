"""
统一因子配置
定义所有模型训练使用的标准因子集
"""

# Simple25FactorEngine生成的完整因子集 (15个因子)
# 这些因子已经过测试验证，可以正常计算且质量良好
COMPLETE_FACTOR_SET = [
    # ==================== 核心动量因子 (1) ====================
    'momentum_10d_ex1',          # 10日动量

    # ==================== 高Alpha因子 (4) ====================
    'near_52w_high',             # 接近52周高点的程度
    'reversal_1d',               # 1日均值回归
    'rel_volume_spike',          # 相对成交量激增
    'mom_accel_5_2',             # 动量加速度（5日vs2日） ✓ 已验证

    # ==================== 技术指标 (2) ====================
    'rsi_7',                     # 7日RSI指标 ✓ 已修复（从rsi改名）
    'bollinger_squeeze',         # 布林带挤压

    # ==================== 成交量因子 (1) ====================
    'obv_momentum',              # OBV动量

    # ==================== 波动率因子 (2) ====================
    'atr_ratio',                 # ATR比率（5日/20日）
    'ivol_60d',                  # 60日特质波动率

    # ==================== 流动性因子 (1) ====================
    'liquidity_factor',          # 流动性因子（成交量相对均值）

    # ==================== 价格效率因子 (1) ====================
    'price_efficiency_5d',       # 5日价格效率 ✓ 已验证

    # ==================== 行为因子 (3) ====================
    # 注：需要180+天数据才能达到>80%覆盖率
    'overnight_intraday_gap',    # 隔夜-日内收益差 (83.7% @ 180d, 88.6% @ 365d)
    'max_lottery_factor',        # 最大彩票因子 (83.7% @ 180d, 88.6% @ 365d)
    'streak_reversal'            # 连续反转因子 (86.4% @ 30d, 83.3% @ 180d, 68.6% @ 365d)
]

# 因子总数
FACTOR_COUNT = len(COMPLETE_FACTOR_SET)  # 15

# 旧因子名称映射（向后兼容）
LEGACY_FACTOR_MAPPING = {
    'momentum_10d': 'momentum_10d_ex1',
    'reversal_5d': 'reversal_1d',
    'mom_accel_10_5': 'mom_accel_5_2',
    'price_efficiency_10d': 'price_efficiency_5d',
    'rsi': 'rsi_7'  # 监控器使用的旧名称
}

# 因子分类（用于分析和可视化）
FACTOR_CATEGORIES = {
    'momentum': ['momentum_10d_ex1', 'mom_accel_5_2', 'obv_momentum'],
    'mean_reversion': ['reversal_1d', 'rsi_7', 'bollinger_squeeze'],
    'volume': ['rel_volume_spike', 'liquidity_factor'],
    'volatility': ['atr_ratio', 'ivol_60d'],
    'behavioral': ['near_52w_high', 'overnight_intraday_gap', 'max_lottery_factor', 'streak_reversal'],
    'efficiency': ['price_efficiency_5d']
}

# 因子描述
FACTOR_DESCRIPTIONS = {
    'momentum_10d_ex1': '10日价格动量',
    'near_52w_high': '当前价格接近52周最高价的程度',
    'reversal_1d': '1日价格反转（均值回归）',
    'rel_volume_spike': '成交量相对20日最大值的激增程度',
    'mom_accel_5_2': '动量加速度：2日动量 - 5日动量',
    'rsi_7': '7日相对强弱指标（标准化到[-1,1]）',
    'bollinger_squeeze': '布林带挤压程度（波动率收窄）',
    'obv_momentum': '能量潮指标的10日动量',
    'atr_ratio': 'ATR比率：5日ATR / 20日ATR - 1',
    'ivol_60d': '60日特质波动率（相对市场的特异波动）',
    'liquidity_factor': '成交量相对20日均值的偏离度',
    'price_efficiency_5d': '5日价格效率：净收益 / 路径长度',
    'overnight_intraday_gap': '隔夜收益与日内收益的20日累积差异',
    'max_lottery_factor': '20日窗口内的最大单日收益率',
    'streak_reversal': '连续上涨/下跌反转信号（相对市场，带阈值，上限5天）'
}

# T+1预测目标配置
TARGET_CONFIG = {
    'name': 'ret_fwd_1d',
    'description': 'T+1日收益率',
    'formula': '(Close_{t+1} - Close_t) / Close_t'
}

def validate_factor_data(factor_data, required_factors=None):
    """
    验证因子数据是否包含所有必需因子

    Args:
        factor_data: pd.DataFrame，因子数据
        required_factors: list，必需因子列表，默认使用COMPLETE_FACTOR_SET

    Returns:
        dict: 验证结果
    """
    if required_factors is None:
        required_factors = COMPLETE_FACTOR_SET

    available_factors = [f for f in required_factors if f in factor_data.columns]
    missing_factors = [f for f in required_factors if f not in factor_data.columns]

    validation_result = {
        'is_valid': len(missing_factors) == 0,
        'required_count': len(required_factors),
        'available_count': len(available_factors),
        'missing_count': len(missing_factors),
        'available_factors': available_factors,
        'missing_factors': missing_factors,
        'coverage': len(available_factors) / len(required_factors) * 100 if required_factors else 0
    }

    return validation_result

def get_factor_subset(categories=None):
    """
    获取特定类别的因子子集

    Args:
        categories: list of str，因子类别列表

    Returns:
        list: 因子名称列表
    """
    if categories is None:
        return COMPLETE_FACTOR_SET.copy()

    factors = []
    for category in categories:
        if category in FACTOR_CATEGORIES:
            factors.extend(FACTOR_CATEGORIES[category])

    # 去重并保持原始顺序
    seen = set()
    ordered_factors = []
    for f in COMPLETE_FACTOR_SET:
        if f in factors and f not in seen:
            seen.add(f)
            ordered_factors.append(f)

    return ordered_factors

def print_factor_summary():
    """打印因子配置摘要"""
    print("=" * 80)
    print("标准因子配置摘要")
    print("=" * 80)
    print(f"总因子数: {FACTOR_COUNT}")
    print(f"\n因子列表:")
    for i, factor in enumerate(COMPLETE_FACTOR_SET, 1):
        category = [cat for cat, factors in FACTOR_CATEGORIES.items() if factor in factors]
        category_str = f"[{category[0]}]" if category else "[未分类]"
        desc = FACTOR_DESCRIPTIONS.get(factor, "无描述")
        print(f"  {i:2d}. {factor:30s} {category_str:15s} - {desc}")

    print(f"\n因子分类统计:")
    for category, factors in FACTOR_CATEGORIES.items():
        print(f"  {category:15s}: {len(factors):2d} 个因子")

    print(f"\n预测目标:")
    print(f"  名称: {TARGET_CONFIG['name']}")
    print(f"  描述: {TARGET_CONFIG['description']}")
    print(f"  公式: {TARGET_CONFIG['formula']}")
    print("=" * 80)

if __name__ == "__main__":
    print_factor_summary()
