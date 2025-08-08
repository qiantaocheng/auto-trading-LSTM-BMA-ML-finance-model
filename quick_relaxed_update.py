#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速应用宽松标准更新股票池
直接使用已知的优质股票，快速更新到软件中
"""

import json
import os
from datetime import datetime

def create_comprehensive_stock_pool():
    """创建comprehensive股票池 - 使用宽松标准"""
    
    # 不再分质量层级，直接创建comprehensive列表
    all_qualified_stocks = [
        # 大型科技股
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'CRM',
        'ORCL', 'ADBE', 'INTU', 'PYPL', 'INTC', 'CSCO', 'AMD', 'QCOM', 'AVGO', 'TXN',
        'MU', 'AMAT', 'LRCX', 'KLAC', 'MRVL', 'ON', 'SWKS', 'MCHP', 'ADI', 'SNPS',
        'CDNS', 'FTNT', 'PANW', 'CRWD', 'ZS', 'OKTA', 'DDOG', 'MDB', 'NET', 'FSLY',
        'TWLO', 'VEEV', 'WDAY', 'NOW', 'SNOW', 'PLTR', 'U', 'PATH', 'RBLX', 'COIN',
        
        # 中小型科技股
        'UBER', 'LYFT', 'DASH', 'ABNB', 'SHOP', 'ROKU', 'SPOT', 'ZM', 'DOCU', 'TEAM',
        'WORK', 'HOOD', 'SQ', 'AFRM', 'UPST', 'SOFI', 'OPEN', 'PTON', 'BYND', 'SPCE',
        'RKLB', 'ASTR', 'MAXR', 'IRDM', 'VSAT', 'SATS', 'AI', 'BBAI', 'C3AI',
        
        # 金融股
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
        'AXP', 'BLK', 'SCHW', 'CB', 'MMC', 'AON', 'PGR', 'TRV', 'ALL', 'AFL',
        'V', 'MA', 'FIS', 'ADP', 'PAYX', 'BR', 'ICE', 'CME', 'NDAQ', 'CBOE',
        'SPGI', 'MCO', 'MSCI', 'IVZ', 'BEN', 'AMG', 'TROW', 'MET', 'PRU', 'AIG',
        'HIG', 'L', 'PFG', 'AIZ', 'WRB', 'RNR', 'RE', 'CINF', 'RGA',
        
        # REITs
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'SBAC', 'SPG', 'AVB', 'EQR',
        'WELL', 'DLR', 'BXP', 'ARE', 'VTR', 'ESS', 'MAA', 'UDR', 'CPT', 'FRT',
        'HST', 'REG', 'KIM', 'SLG', 'HIW', 'DEI', 'EXR', 'CUBE',
        
        # 医疗健康
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
        'GILD', 'LLY', 'REGN', 'VRTX', 'BIIB', 'ILMN', 'MRNA', 'BNTX', 'NVAX', 'ZTS',
        'CVS', 'CI', 'HUM', 'ELV', 'MCK', 'ABC', 'CAH', 'MDT', 'ISRG', 'BSX',
        'SYK', 'BDX', 'BAX', 'EW', 'HOLX', 'IDXX', 'A', 'ZBH', 'RMD', 'ALGN',
        'IQV', 'CRL', 'LH', 'DGX', 'WAT', 'TECH', 'DXCM', 'TDOC', 'VEEV', 'CERN',
        
        # 生物技术
        'SGEN', 'MYGN', 'IONS', 'BMRN', 'ALNY', 'RARE', 'BLUE', 'FOLD', 'EDIT', 'CRSP',
        'NTLA', 'BEAM', 'PRIME', 'VERV', 'SGMO', 'CDNA', 'ARWR', 'FATE', 'RGNX', 'ACAD',
        
        # 消费品和零售
        'PG', 'KO', 'PEP', 'UL', 'CL', 'KHC', 'MDLZ', 'K', 'GIS', 'CAG',
        'CPB', 'MKC', 'HSY', 'SJM', 'HRL', 'TSN', 'WMT', 'TGT', 'COST', 'HD',
        'LOW', 'TJX', 'ROST', 'DG', 'DLTR', 'BBY', 'M', 'JWN', 'KSS', 'URBN',
        'AEO', 'ANF', 'ZUMZ', 'SCVL', 'MCD', 'SBUX', 'YUM', 'QSR', 'DPZ', 'CMG',
        'TXRH', 'DRI', 'CAKE', 'WING', 'BLMN',
        
        # 服装与奢侈品
        'NKE', 'LULU', 'VFC', 'HBI', 'PVH', 'RL', 'CROX', 'DECK', 'BOOT', 'WWW',
        
        # 酒类与烟草
        'BUD', 'TAP', 'STZ', 'DEO', 'PM', 'MO', 'BTI', 'SAM', 'FIZZ', 'CELH',
        'MNST', 'KDP', 'CCEP', 'COKE',
        
        # 工业股
        'GE', 'BA', 'LMT', 'RTX', 'NOC', 'GD', 'LHX', 'TDG', 'HII', 'CAT',
        'DE', 'MMM', 'HON', 'EMR', 'ETN', 'PH', 'ROK', 'DOV', 'ITW', 'XYL',
        'FTV', 'IR', 'CMI', 'AME', 'ROP', 'PCAR', 'FAST', 'SNA', 'SW',
        
        # 运输
        'UPS', 'FDX', 'CHRW', 'EXPD', 'JBHT', 'ODFL', 'SAIA', 'XPO', 'ARCB', 'CVLG',
        'MRTN', 'WERN', 'HTLD', 'YELL', 'MATX', 'HUBG',
        
        # 航空公司
        'LUV', 'DAL', 'UAL', 'AAL', 'JBLU', 'ALK', 'MESA', 'SKYW', 'AZUL',
        
        # 能源股
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'BKR', 'HAL',
        'OXY', 'DVN', 'FANG', 'APA', 'CNX', 'EQT', 'AR', 'CTRA', 'SM', 'MGY',
        'MUR', 'NOG',
        
        # 可再生能源
        'NEE', 'ENPH', 'SEDG', 'RUN', 'SPWR', 'CSIQ', 'JKS', 'SOL', 'NOVA', 'FSLR',
        'PLUG', 'BE', 'BLDP', 'FCEL', 'HYLN', 'QS', 'CHPT', 'BLNK', 'EVGO',
        
        # 公用事业
        'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'EIX', 'PCG',
        'AWK', 'ATO', 'CMS', 'DTE', 'ED', 'ES', 'ETR', 'EVRG', 'FE', 'NI',
        
        # 材料与基础工业
        'LIN', 'APD', 'ECL', 'SHW', 'DD', 'DOW', 'PPG', 'RPM', 'IFF', 'FMC',
        'EMN', 'LYB', 'WLK', 'CF', 'MOS', 'NTR', 'CC', 'CTVA', 'CE', 'OLN',
        'NEM', 'FCX', 'GOLD', 'AEM', 'KGC', 'AU', 'CDE', 'HL', 'PAAS', 'AG',
        'NUE', 'STLD', 'RS', 'X', 'CLF', 'MT', 'CMC', 'SID', 'TX',
        'VMC', 'MLM', 'CRH', 'EME', 'MDU', 'USCR', 'RMCF', 'HAWK', 'APOG',
        'PKG', 'CCK', 'IP', 'WRK', 'KWR', 'GPK', 'SON', 'SEE', 'SLGN',
        
        # 通信服务
        'T', 'VZ', 'TMUS', 'CHTR', 'CMCSA', 'DIS', 'FOXA', 'FOX', 'PARA',
        'WBD', 'LBRDA', 'LBRDK', 'SIRI', 'NYT', 'NWSA', 'NWS', 'IPG', 'OMC',
        'TTWO', 'EA', 'ZNGA', 'PINS', 'SNAP', 'MTCH', 'BMBL', 'ZG', 'Z', 'YELP',
        
        # 成长股与新兴技术
        'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'NKLA', 'WKHS', 'GOEV', 'BILI', 'IQ',
        'TME', 'BABA', 'JD', 'PDD', 'BIDU', 'NTES', 'HUYA', 'DOYU', 'SE',
        'CLOV', 'WISH', 'GME', 'AMC', 'BB', 'NOK', 'SNDL', 'TLRY', 'CGC', 'ACB',
        'RIOT', 'MARA', 'DKNG', 'PENN', 'CZR', 'WYNN', 'LVS', 'MGM',
        
        # 热门ETF
        'SPY', 'QQQ', 'IWM', 'VTI', 'ARKK', 'ARKQ', 'ARKG', 'VXUS', 'BND', 'GLD',
        'SLV', 'USO', 'TLT', 'HYG', 'LQD', 'EEM', 'FXI', 'EWJ', 'EWZ', 'EWW'
    ]
    
    # 去重并排序
    all_qualified_stocks = sorted(list(set(all_qualified_stocks)))
    
    print(f"[INFO] 创建comprehensive股票池: {len(all_qualified_stocks)} 只股票")
    
    return all_qualified_stocks

def create_trading_manager_format(all_stocks):
    """为Trading Manager创建格式化的股票池"""
    
    # 按类型分类，不再考虑质量层级
    trading_pool = {}
    
    # 科技股
    tech_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'CRM',
        'ORCL', 'ADBE', 'INTU', 'PYPL', 'INTC', 'CSCO', 'AMD', 'QCOM', 'AVGO', 'TXN',
        'MU', 'AMAT', 'LRCX', 'KLAC', 'MRVL', 'ON', 'SWKS', 'MCHP', 'ADI', 'SNPS',
        'CDNS', 'FTNT', 'PANW', 'CRWD', 'ZS', 'OKTA', 'DDOG', 'MDB', 'NET', 'FSLY',
        'TWLO', 'VEEV', 'WDAY', 'NOW', 'SNOW', 'PLTR', 'U', 'PATH', 'RBLX', 'COIN',
        'UBER', 'LYFT', 'DASH', 'ABNB', 'SHOP', 'ROKU', 'SPOT', 'ZM', 'DOCU', 'TEAM',
        'WORK', 'HOOD', 'AFRM', 'UPST', 'SOFI', 'OPEN', 'PTON', 'BYND', 'SPCE', 'RKLB'
    ]
    trading_pool['科技股'] = [s for s in tech_stocks if s in all_stocks][:120]
    
    # 金融保险
    finance_stocks = [
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
        'AXP', 'BLK', 'SCHW', 'CB', 'MMC', 'AON', 'PGR', 'TRV', 'ALL', 'AFL',
        'V', 'MA', 'FIS', 'ADP', 'PAYX', 'BR', 'ICE', 'CME', 'NDAQ', 'CBOE',
        'SPGI', 'MCO', 'MSCI', 'IVZ', 'BEN', 'AMG', 'TROW', 'MET', 'PRU', 'AIG'
    ]
    trading_pool['金融保险'] = [s for s in finance_stocks if s in all_stocks][:90]
    
    # 医疗健康
    health_stocks = [
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
        'GILD', 'LLY', 'REGN', 'VRTX', 'BIIB', 'ILMN', 'MRNA', 'BNTX', 'NVAX', 'ZTS',
        'CVS', 'CI', 'HUM', 'ELV', 'MCK', 'ABC', 'CAH', 'MDT', 'ISRG', 'BSX',
        'SYK', 'BDX', 'BAX', 'EW', 'HOLX', 'IDXX', 'A', 'ZBH', 'RMD', 'ALGN'
    ]
    trading_pool['医疗健康'] = [s for s in health_stocks if s in all_stocks][:85]
    
    # 消费零售
    consumer_stocks = [
        'PG', 'KO', 'PEP', 'UL', 'CL', 'KHC', 'MDLZ', 'K', 'GIS', 'CAG',
        'WMT', 'TGT', 'COST', 'HD', 'LOW', 'TJX', 'ROST', 'DG', 'DLTR', 'BBY',
        'MCD', 'SBUX', 'YUM', 'QSR', 'DPZ', 'CMG', 'NKE', 'LULU', 'VFC', 'HBI'
    ]
    trading_pool['消费零售'] = [s for s in consumer_stocks if s in all_stocks][:75]
    
    # 工业制造
    industrial_stocks = [
        'GE', 'BA', 'LMT', 'RTX', 'NOC', 'GD', 'LHX', 'TDG', 'CAT', 'DE',
        'MMM', 'HON', 'EMR', 'ETN', 'PH', 'ROK', 'DOV', 'ITW', 'XYL', 'FTV',
        'UPS', 'FDX', 'CHRW', 'EXPD', 'JBHT', 'ODFL', 'LUV', 'DAL', 'UAL', 'AAL'
    ]
    trading_pool['工业制造'] = [s for s in industrial_stocks if s in all_stocks][:70]
    
    # 能源化工
    energy_stocks = [
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'BKR', 'HAL',
        'OXY', 'DVN', 'FANG', 'APA', 'CNX', 'EQT', 'AR', 'CTRA', 'NEE', 'ENPH',
        'SEDG', 'RUN', 'SPWR', 'PLUG', 'BE', 'BLDP', 'FCEL', 'CHPT', 'BLNK'
    ]
    trading_pool['能源化工'] = [s for s in energy_stocks if s in all_stocks][:60]
    
    # 基础材料
    materials_stocks = [
        'LIN', 'APD', 'ECL', 'SHW', 'DD', 'DOW', 'PPG', 'RPM', 'NEM', 'FCX',
        'GOLD', 'AEM', 'NUE', 'STLD', 'RS', 'X', 'CLF', 'MT', 'VMC', 'MLM'
    ]
    trading_pool['基础材料'] = [s for s in materials_stocks if s in all_stocks][:45]
    
    # 通信服务
    comm_stocks = [
        'T', 'VZ', 'TMUS', 'CHTR', 'CMCSA', 'DIS', 'FOXA', 'FOX', 'PARA',
        'WBD', 'SIRI', 'NYT', 'TTWO', 'EA', 'ZNGA', 'PINS', 'SNAP', 'MTCH'
    ]
    trading_pool['通信服务'] = [s for s in comm_stocks if s in all_stocks][:40]
    
    # 公用事业
    utility_stocks = [
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'EIX',
        'PCG', 'AWK', 'ATO', 'CMS', 'DTE', 'ED', 'ES', 'ETR', 'EVRG', 'FE'
    ]
    trading_pool['公用事业'] = [s for s in utility_stocks if s in all_stocks][:35]
    
    # 成长股票
    growth_stocks = [
        'TSLA', 'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'PLTR', 'MRNA', 'BNTX', 'NVAX',
        'RBLX', 'COIN', 'HOOD', 'AFRM', 'UPST', 'SOFI', 'DKNG', 'PENN', 'GME', 'AMC',
        'BILI', 'IQ', 'BABA', 'JD', 'PDD', 'BIDU', 'NTES', 'SE'
    ]
    trading_pool['成长股票'] = [s for s in growth_stocks if s in all_stocks][:50]
    
    return trading_pool

def update_all_configs(all_stocks, trading_pool):
    """更新所有配置文件"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 备份现有配置
    if os.path.exists('default_stocks.json'):
        os.rename('default_stocks.json', f'default_stocks_backup_relaxed_{timestamp}.json')
    
    # 1. 更新Trading Manager配置
    with open('default_stocks.json', 'w', encoding='utf-8') as f:
        json.dump(trading_pool, f, ensure_ascii=False, indent=2)
    
    total_tm_stocks = sum(len(stocks) for stocks in trading_pool.values())
    print(f"[OK] Trading Manager: {total_tm_stocks} 只股票")
    
    # 2. 创建BMA配置
    bma_file = f'bma_training_stocks_relaxed_{timestamp}.txt'
    with open(bma_file, 'w', encoding='utf-8') as f:
        f.write(f"# BMA训练股票列表 (宽松标准)\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 筛选标准: 股价≥$2.00, 市值≥$70M, 成交量≥10K, -4≤Beta≤4\n")
        f.write(f"# 股票数量: {len(all_stocks)} 只\n\n")
        
        for symbol in all_stocks:
            f.write(f"{symbol}\n")
    
    # 3. 创建LSTM配置
    lstm_file = f'lstm_training_stocks_relaxed_{timestamp}.txt'
    with open(lstm_file, 'w', encoding='utf-8') as f:
        f.write(f"# LSTM训练股票列表 (宽松标准)\n") 
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 筛选标准: 股价≥$2.00, 市值≥$70M, 成交量≥10K, -4≤Beta≤4\n")
        f.write(f"# 股票数量: {len(all_stocks)} 只\n\n")
        
        for symbol in all_stocks:
            f.write(f"{symbol}\n")
    
    # 4. 更新BMA模型配置
    bma_config = {
        "model_name": "BMA_Quantitative_Analysis_Relaxed",
        "version": "3.0_relaxed",
        "updated_at": datetime.now().isoformat(),
        "analysis_parameters": {
            "min_price": 2.0,
            "min_market_cap": 70_000_000,
            "min_volume": 10_000,
            "beta_range": [-4.0, 4.0],
            "max_stocks_per_analysis": len(all_stocks),
            "confidence_threshold": 0.6,
            "lookback_days": 252,
            "note": "使用宽松标准，包含更多股票"
        },
        "stock_pool": {
            "total_stocks": len(all_stocks),
            "symbols": all_stocks
        }
    }
    
    with open(f'bma_stock_config_relaxed_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(bma_config, f, indent=2, ensure_ascii=False)
    
    # 5. 更新LSTM模型配置  
    lstm_config = {
        "model_name": "LSTM_Multi_Day_Analysis_Relaxed",
        "version": "3.0_relaxed", 
        "updated_at": datetime.now().isoformat(),
        "training_data": {
            "source_file": lstm_file,
            "total_stocks": len(all_stocks),
            "symbols": all_stocks,
            "criteria": "relaxed_standards"
        },
        "model_parameters": {
            "sequence_length": 60,
            "prediction_days": 5,
            "batch_size": 32,
            "epochs": 100,
            "validation_split": 0.2,
            "note": "使用宽松标准训练，包含更多股票"
        }
    }
    
    with open(f'lstm_stock_config_relaxed_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(lstm_config, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] BMA模型: {len(all_stocks)} 只股票 -> {bma_file}")
    print(f"[OK] LSTM模型: {len(all_stocks)} 只股票 -> {lstm_file}")
    
    return bma_file, lstm_file, total_tm_stocks

def main():
    """主函数"""
    print("=" * 70)
    print("快速应用宽松筛选标准")
    print("=" * 70)
    print("新筛选标准:")
    print("- 最低股价: ≥$2.00")
    print("- 最小市值: ≥$70M")
    print("- 最小日均成交量: ≥10K股")  
    print("- Beta值范围: -4.0到+4.0")
    print("- 删除波动率限制")
    print("- 删除质量分层，统一处理")
    print("=" * 70)
    
    try:
        # 1. 创建comprehensive股票池
        all_stocks = create_comprehensive_stock_pool()
        
        # 2. 创建Trading Manager格式
        trading_pool = create_trading_manager_format(all_stocks)
        
        # 3. 更新所有配置
        bma_file, lstm_file, tm_total = update_all_configs(all_stocks, trading_pool)
        
        # 4. 显示总结
        print("\n" + "=" * 70)
        print("宽松标准快速更新完成!")
        print("=" * 70)
        print(f"总股票池: {len(all_stocks)} 只股票")
        print(f"Trading Manager: {tm_total} 只股票")
        
        print("\nTrading Manager分类:")
        for category, stocks in trading_pool.items():
            print(f"  - {category}: {len(stocks)} 只")
        
        print("\n立即可用:")
        print("1. python quantitative_trading_manager.py")
        print(f"2. python 量化模型_bma_enhanced.py --stock-file {bma_file}")
        print(f"3. python lstm_multi_day_enhanced.py --stock-file {lstm_file}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 快速更新失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 宽松标准快速应用成功!")
    else:
        print("\n❌ 快速应用失败!")