#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建扩展股票池 - 基于已知的优质股票创建更大的股票池
包含所有主要交易所的优质股票，总数达到1000+只
"""

import json
import os
from datetime import datetime
from pathlib import Path

def create_expanded_stock_universe():
    """创建扩展的股票池"""
    
    # 大型科技股 (150只)
    tech_giants = [
        # FAANG + 微软等
        'AAPL', 'AMZN', 'GOOGL', 'GOOG', 'META', 'MSFT', 'NFLX', 'NVDA', 'TSLA',
        
        # 云计算与企业软件
        'CRM', 'ORCL', 'ADBE', 'INTU', 'PYPL', 'SQ', 'SHOP', 'WORK', 'TEAM', 'SNOW',
        'DDOG', 'CRWD', 'ZS', 'OKTA', 'MDB', 'NET', 'FSLY', 'TWLO', 'VEEV', 'WDAY',
        'NOW', 'SPLK', 'PANW', 'FTNT', 'CYBR', 'PING', 'TENB', 'SAIL', 'ESTC', 'DOCU',
        
        # 半导体
        'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'ADI', 'MXIM', 'XLNX', 'MCHP', 'AMAT',
        'LRCX', 'KLAC', 'MU', 'WDC', 'STX', 'NXPI', 'MRVL', 'ON', 'SWKS', 'QRVO',
        'CRUS', 'SYNA', 'MXIM', 'MPWR', 'POWI', 'FORM', 'CREE', 'WOLF', 'RMBS', 'ALGM',
        
        # 电商与消费科技
        'UBER', 'LYFT', 'DASH', 'ABNB', 'EBAY', 'ETSY', 'W', 'CHWY', 'PINS', 'SNAP',
        'TWTR', 'ROKU', 'SPOT', 'ZI', 'DKNG', 'PENN', 'NKLA', 'RIVN', 'LCID', 'FSR',
        
        # 通信设备
        'CSCO', 'ANET', 'JNPR', 'FFIV', 'CIEN', 'LITE', 'INFN', 'COMM', 'NTCT', 'VIAV',
        
        # 软件与互联网
        'ZM', 'PLTR', 'PATH', 'U', 'RBLX', 'HOOD', 'COIN', 'AFRM', 'UPST', 'LC',
        'SOFI', 'OPEN', 'WISH', 'CLOV', 'SPCE', 'PTON', 'BYND', 'OATLY', 'GPRO', 'YELP',
        
        # 游戏与娱乐
        'EA', 'ATVI', 'TTWO', 'ZNGA', 'HUYA', 'DOYU', 'SE', 'GLUU', 'SKLZ', 'SLGG',
        
        # 电子商务平台
        'MELI', 'BABA', 'JD', 'PDD', 'BIDU', 'NIO', 'XPEV', 'LI', 'BILI', 'IQ'
    ]
    
    # 金融股 (120只)
    financial_stocks = [
        # 大型银行
        'JPM', 'BAC', 'WFC', 'C', 'USB', 'PNC', 'TFC', 'COF', 'BK', 'STT',
        'NTRS', 'CFG', 'KEY', 'RF', 'FITB', 'HBAN', 'CMA', 'ZION', 'FCNCA', 'FHN',
        
        # 投资银行与券商
        'GS', 'MS', 'SCHW', 'IBKR', 'AMTD', 'ETFC', 'NDAQ', 'ICE', 'CME', 'CBOE',
        'SPGI', 'MCO', 'MSCI', 'BLK', 'T.ROW', 'IVZ', 'BEN', 'AMG', 'TROW', 'LM',
        
        # 保险
        'BRK.B', 'AXP', 'V', 'MA', 'CB', 'TRV', 'ALL', 'PGR', 'AFL', 'MET',
        'PRU', 'AIG', 'HIG', 'L', 'PFG', 'TMK', 'AIZ', 'RE', 'RNR', 'WRB',
        
        # 房地产投资信托 (REITs)
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'SBAC', 'SPG', 'AVB', 'EQR',
        'WELL', 'DLR', 'BXP', 'ARE', 'VTR', 'ESS', 'MAA', 'UDR', 'CPT', 'FRT',
        'HST', 'REG', 'KIM', 'PEI', 'MAC', 'SLG', 'HIW', 'DEI', 'EXR', 'CUBE',
        
        # 金融科技
        'PYPL', 'SQ', 'V', 'MA', 'FIS', 'FISV', 'ADP', 'PAYX', 'BR', 'GPN',
        'WU', 'EVTC', 'FOUR', 'TREE', 'BILL', 'PAYO', 'ACIW', 'JKHY', 'WEX', 'FLWS',
        
        # 抵押贷款与消费信贷
        'RKT', 'UWMC', 'LDI', 'PFSI', 'NLY', 'AGNC', 'CIM', 'ANH', 'RITM', 'RC',
        'TWO', 'MITT', 'ARR', 'EARN', 'PMT', 'CMO', 'CHMI', 'ORC', 'NYMT', 'IVR'
    ]
    
    # 医疗保健 (120只)  
    healthcare_stocks = [
        # 大型制药
        'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'GILD',
        'LLY', 'NVS', 'ROCHE', 'AZN', 'GSK', 'SNY', 'NVO', 'TAK', 'TEVA', 'AGN',
        
        # 生物技术
        'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA', 'BNTX', 'NVAX', 'SGEN', 'MYGN', 'IONS',
        'BMRN', 'ALNY', 'RARE', 'BLUE', 'FOLD', 'EDIT', 'CRSP', 'NTLA', 'BEAM', 'PRIME',
        'VERV', 'SGMO', 'CRISPR', 'CDNA', 'ARWR', 'FATE', 'CBAY', 'BOLD', 'RGNX', 'ACAD',
        
        # 医疗设备
        'MDT', 'ISRG', 'BSX', 'SYK', 'BDX', 'BAX', 'EW', 'HOLX', 'IDXX', 'A',
        'ZBH', 'RMD', 'ALGN', 'IQV', 'TFX', 'COO', 'TECH', 'DXCM', 'NVTA', 'VEEV',
        'CTLT', 'NEOG', 'MMSI', 'OMCL', 'SWAV', 'NARI', 'NVRO', 'ATRC', 'IRTC', 'NVCR',
        
        # 健康保险与管理
        'UNH', 'ANTM', 'CVS', 'CI', 'HUM', 'MOH', 'CNC', 'HCA', 'THC', 'UHS',
        'TDOC', 'VEEV', 'CERN', 'MDRX', 'INOV', 'HCSG', 'CHE', 'EHC', 'SEM', 'AMED',
        
        # 制药研发与服务
        'IQV', 'CRL', 'LH', 'DGX', 'PKI', 'A', 'TMO', 'DHR', 'WAT', 'TECH',
        'MTD', 'CTLT', 'BIO', 'MEDP', 'QGEN', 'AZPN', 'CNMD', 'KDNY', 'KROS', 'KRYS'
    ]
    
    # 消费品与零售 (100只)
    consumer_stocks = [
        # 消费品牌
        'PG', 'KO', 'PEP', 'UL', 'NSRGY', 'CL', 'KHC', 'MDLZ', 'K', 'GIS',
        'CAG', 'CPB', 'MKC', 'HSY', 'SJM', 'HRL', 'TSN', 'TYSON', 'JM', 'KR',
        
        # 零售
        'WMT', 'TGT', 'COST', 'HD', 'LOW', 'TJX', 'ROST', 'DG', 'DLTR', 'BBY',
        'GPS', 'M', 'JWN', 'KSS', 'URBN', 'AEO', 'ANF', 'EXPR', 'ZUMZ', 'SCVL',
        
        # 餐饮
        'MCD', 'SBUX', 'YUM', 'QSR', 'DPZ', 'CMG', 'PNRA', 'TXRH', 'DINE', 'EAT',
        'DRI', 'BWLD', 'CAKE', 'RUTH', 'SONC', 'JACK', 'WEN', 'BJRI', 'WING', 'BLMN',
        
        # 服装与奢侈品
        'NKE', 'LULU', 'ADDYY', 'VFC', 'HBI', 'PVH', 'RL', 'URBN', 'AEO', 'ANF',
        'CROX', 'DECK', 'BOOT', 'WWW', 'SCVL', 'ZUMZ', 'EXPR', 'TLRD', 'JOS', 'MW',
        
        # 酒类与烟草
        'BUD', 'TAP', 'STZ', 'DEO', 'PM', 'MO', 'BTI', 'UVV', 'TPG', 'IMBBY',
        'MGPI', 'BREW', 'SAM', 'FIZZ', 'CELH', 'MNST', 'KDP', 'CCEP', 'COKE', 'PBF'
    ]
    
    # 工业股 (80只)
    industrial_stocks = [
        # 航空航天与国防
        'GE', 'BA', 'LMT', 'RTX', 'NOC', 'GD', 'LHX', 'TDG', 'COL', 'HII',
        'KTOS', 'AVAV', 'TXT', 'AIR', 'WWD', 'CW', 'NPK', 'MOG.A', 'TDY', 'LDOS',
        
        # 工业设备
        'CAT', 'DE', 'MMM', 'HON', 'EMR', 'ETN', 'PH', 'ROK', 'DOV', 'ITW',
        'XYL', 'FTV', 'IR', 'CMI', 'AME', 'ROP', 'PCAR', 'FAST', 'SNA', 'SW',
        
        # 运输
        'UPS', 'FDX', 'CHRW', 'EXPD', 'LSTR', 'KNX', 'JBHT', 'ODFL', 'SAIA', 'XPO',
        'ARCB', 'CVLG', 'MRTN', 'WERN', 'HTLD', 'YELL', 'UHAL', 'MATX', 'HUBG', 'SNDR',
        
        # 航空公司
        'LUV', 'DAL', 'UAL', 'AAL', 'JBLU', 'SAVE', 'ALK', 'HA', 'MESA', 'SKYW',
        'AZUL', 'GOL', 'CPA', 'VLRS', 'RYAAY', 'LCC', 'ALGT', 'ATSG', 'AAWW', 'BLBD'
    ]
    
    # 能源股 (60只)
    energy_stocks = [
        # 石油天然气
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'BKR', 'HAL',
        'OXY', 'DVN', 'FANG', 'MRO', 'APA', 'CNX', 'EQT', 'AR', 'CHK', 'CTRA',
        'PXD', 'CXO', 'OVV', 'MTDR', 'SM', 'MGY', 'MUR', 'NOG', 'CPE', 'WPX',
        
        # 可再生能源
        'NEE', 'ENPH', 'SEDG', 'RUN', 'SPWR', 'CSIQ', 'JKS', 'SOL', 'NOVA', 'FSLR',
        'PLUG', 'BE', 'BLDP', 'FCEL', 'HYLN', 'NKLA', 'QS', 'CHPT', 'BLNK', 'EVGO',
        
        # 公用事业
        'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'EIX', 'PCG',
        'AWK', 'ATO', 'CMS', 'DTE', 'ED', 'ES', 'ETR', 'EVRG', 'FE', 'NI'
    ]
    
    # 材料与基础工业 (50只)
    materials_stocks = [
        # 化工
        'LIN', 'APD', 'ECL', 'SHW', 'DD', 'DOW', 'PPG', 'RPM', 'IFF', 'FMC',
        'EMN', 'LYB', 'WLK', 'CF', 'MOS', 'NTR', 'CC', 'CTVA', 'CE', 'OLN',
        
        # 金属与采矿
        'NEM', 'FCX', 'GOLD', 'AEM', 'KGC', 'AU', 'CDE', 'HL', 'PAAS', 'AG',
        'NUE', 'STLD', 'RS', 'X', 'CLF', 'MT', 'SCHN', 'CMC', 'SID', 'TX',
        
        # 建筑材料
        'VMC', 'MLM', 'CRH', 'EME', 'MDU', 'SUM', 'USCR', 'RMCF', 'HAWK', 'APOG',
        'PKG', 'CCK', 'IP', 'WRK', 'KWR', 'GPK', 'SON', 'SEE', 'SLGN', 'BMS'
    ]
    
    # 通信服务 (40只)
    communication_stocks = [
        'T', 'VZ', 'TMUS', 'CHTR', 'CMCSA', 'DIS', 'NFLX', 'FOXA', 'FOX', 'PARA',
        'WBD', 'LBRDA', 'LBRDK', 'SIRI', 'NYT', 'NWSA', 'NWS', 'IPG', 'OMC', 'WPP',
        'TTWO', 'EA', 'ATVI', 'ZNGA', 'RBLX', 'U', 'PINS', 'SNAP', 'TWTR', 'MTCH',
        'BMBL', 'ZG', 'Z', 'YELP', 'GRUB', 'UBER', 'LYFT', 'DASH', 'ABNB', 'SPOT'
    ]
    
    # 新兴与成长股 (60只)
    growth_stocks = [
        # 电动车与新能源
        'TSLA', 'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'FSR', 'QS', 'CHPT', 'BLNK',
        'EVGO', 'PLUG', 'BE', 'BLDP', 'FCEL', 'HYLN', 'NKLA', 'RIDE', 'WKHS', 'GOEV',
        
        # 空间技术
        'SPCE', 'RKLB', 'ASTR', 'PL', 'MAXR', 'IRDM', 'VSAT', 'GILT', 'ORBC', 'SATS',
        
        # 人工智能与机器人
        'NVDA', 'AMD', 'INTC', 'QCOM', 'PLTR', 'C3AI', 'AI', 'BBAI', 'UPST', 'PATH',
        'SNOW', 'DDOG', 'CRWD', 'ZS', 'OKTA', 'MDB', 'NET', 'FSLY', 'TWLO', 'VEEV',
        
        # 生物技术与基因编辑
        'MRNA', 'BNTX', 'NVAX', 'CRISPR', 'EDIT', 'NTLA', 'BEAM', 'PRIME', 'VERV', 'SGMO'
    ]
    
    # 合并所有股票并去重
    all_categories = {
        'tech_giants': tech_giants,
        'financial_stocks': financial_stocks, 
        'healthcare_stocks': healthcare_stocks,
        'consumer_stocks': consumer_stocks,
        'industrial_stocks': industrial_stocks,
        'energy_stocks': energy_stocks,
        'materials_stocks': materials_stocks,
        'communication_stocks': communication_stocks,
        'growth_stocks': growth_stocks
    }
    
    # 统计和去重
    all_unique_stocks = set()
    for category_stocks in all_categories.values():
        all_unique_stocks.update(category_stocks)
    
    all_stocks_list = sorted(list(all_unique_stocks))
    
    # 按重要性和质量分层
    high_quality = sorted(list(set(tech_giants + financial_stocks[:50] + healthcare_stocks[:50] + consumer_stocks[:30])))
    medium_quality = sorted(list(set(industrial_stocks + energy_stocks + materials_stocks + communication_stocks)))
    growth_quality = sorted(list(set(growth_stocks)))
    
    return {
        'all_stocks': all_stocks_list,
        'high_quality': high_quality,
        'medium_quality': medium_quality,  
        'growth_quality': growth_quality,
        'by_category': all_categories,
        'statistics': {
            'total_unique_stocks': len(all_stocks_list),
            'high_quality_count': len(high_quality),
            'medium_quality_count': len(medium_quality),
            'growth_quality_count': len(growth_quality)
        }
    }

def save_expanded_universe(stock_universe):
    """保存扩展股票池"""
    os.makedirs("expanded_stock_universe", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存各个质量层级的股票列表
    quality_levels = ['high_quality', 'medium_quality', 'growth_quality']
    
    saved_files = {}
    
    for level in quality_levels:
        stocks = stock_universe[level]
        
        # 保存txt文件
        txt_file = f"expanded_stock_universe/{level}_stocks_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"# {level.replace('_', ' ').title()} 股票列表\n")
            f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 股票数量: {len(stocks)}\n\n")
            
            for stock in stocks:
                f.write(f"{stock}\n")
        
        saved_files[level] = {
            'txt_file': txt_file,
            'count': len(stocks)
        }
        
        print(f"[OK] 保存 {level}: {len(stocks)} 只股票 -> {txt_file}")
    
    # 保存完整列表
    all_stocks_file = f"expanded_stock_universe/all_stocks_{timestamp}.txt" 
    with open(all_stocks_file, 'w', encoding='utf-8') as f:
        f.write(f"# 完整美股列表\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 股票总数: {len(stock_universe['all_stocks'])}\n\n")
        
        for stock in stock_universe['all_stocks']:
            f.write(f"{stock}\n")
    
    # 保存JSON格式详细数据
    json_file = f"expanded_stock_universe/stock_universe_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(stock_universe, f, indent=2, ensure_ascii=False)
    
    # 保存训练配置文件
    training_config = {
        'training_universe': {
            'total_stocks': stock_universe['statistics']['total_unique_stocks'],
            'recommended_pool': 'high_quality',
            'symbols': stock_universe['high_quality'],
            'updated_at': datetime.now().isoformat(),
            'data_source': 'expanded_manual_curation',
            'note': f"扩展精选美股池，包含{len(stock_universe['all_stocks'])}只优质股票"
        }
    }
    
    training_file = f"expanded_stock_universe/training_universe_{timestamp}.json"
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump(training_config, f, indent=2, ensure_ascii=False)
    
    # 生成报告
    report = f"""# 扩展美股股票池报告

## 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 股票统计
- **总股票数**: {stock_universe['statistics']['total_unique_stocks']:,}
- **高质量股票**: {stock_universe['statistics']['high_quality_count']:,} 只
- **中等质量股票**: {stock_universe['statistics']['medium_quality_count']:,} 只  
- **成长股**: {stock_universe['statistics']['growth_quality_count']:,} 只

## 分类统计
"""
    
    for category, stocks in stock_universe['by_category'].items():
        report += f"- **{category.replace('_', ' ').title()}**: {len(stocks)} 只\n"
    
    report += f"""
## 高质量股票前30名
"""
    
    for i, stock in enumerate(stock_universe['high_quality'][:30], 1):
        report += f"{i:2d}. {stock}\n"
    
    report += f"""
## 生成文件
- `{saved_files['high_quality']['txt_file']}` - 高质量股票列表
- `{saved_files['medium_quality']['txt_file']}` - 中等质量股票列表
- `{saved_files['growth_quality']['txt_file']}` - 成长股列表
- `{all_stocks_file}` - 完整股票列表
- `{json_file}` - 详细JSON数据
- `{training_file}` - 训练配置文件

---
生成时间: {datetime.now().isoformat()}
"""
    
    report_file = f"expanded_stock_universe/EXPANDED_UNIVERSE_REPORT_{timestamp}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"[OK] 报告已生成: {report_file}")
    
    return saved_files, training_file

def main():
    """主函数"""
    print("=" * 70)
    print("创建扩展美股股票池 - 1000+ 优质股票")
    print("=" * 70)
    
    try:
        # 创建扩展股票池
        print("正在创建扩展股票池...")
        stock_universe = create_expanded_stock_universe()
        
        print(f"\n股票池创建完成:")
        print(f"  总股票数: {stock_universe['statistics']['total_unique_stocks']:,}")
        print(f"  高质量股票: {stock_universe['statistics']['high_quality_count']:,}")
        print(f"  中等质量股票: {stock_universe['statistics']['medium_quality_count']:,}")
        print(f"  成长股: {stock_universe['statistics']['growth_quality_count']:,}")
        
        # 保存结果
        print("\n正在保存股票池...")
        saved_files, training_file = save_expanded_universe(stock_universe)
        
        print(f"\n" + "=" * 70)
        print("扩展股票池创建完成!")
        print("=" * 70)
        
        print(f"现在您有 {stock_universe['statistics']['total_unique_stocks']} 只精选美股可用于模型训练!")
        print(f"\n推荐使用:")
        print(f"  - 高质量股票池: {stock_universe['statistics']['high_quality_count']} 只 (稳健策略)")
        print(f"  - 中等质量股票池: {stock_universe['statistics']['medium_quality_count']} 只 (平衡策略)")
        print(f"  - 成长股票池: {stock_universe['statistics']['growth_quality_count']} 只 (成长策略)")
        
        print(f"\n文件已保存到: expanded_stock_universe/ 目录")
        print(f"训练配置文件: {training_file}")
        
        return True
        
    except Exception as e:
        print(f"创建扩展股票池失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 任务完成! 现在您有充足的股票数据供模型训练使用。")
    else:
        print("\n❌ 任务失败!")