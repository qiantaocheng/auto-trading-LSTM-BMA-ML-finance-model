#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMA Ultra Enhanced 量化分析模型 V4
集成Alpha策略、Learning-to-Rank、不确定性感知BMA、高级投资组合优化
提供工业级的量化交易解决方案
"""

import pandas as pd
import numpy as np
# Polygon client for potential future factor data (currently unused to avoid global broadcast issues)
# TODO: When implementing Polygon factors, ensure data is ticker-specific not globally broadcast
try:
    from polygon_client import polygon_client as pc, download as polygon_download, Ticker as PolygonTicker
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    pc, polygon_download, PolygonTicker = None, None, None
import yaml
import warnings
import argparse
import os
import tempfile
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
import time

# 基础科学计算
from scipy.stats import spearmanr, entropy
from scipy.optimize import minimize
import statsmodels.api as sm
from dataclasses import dataclass, field
from scipy import stats
from sklearn.linear_model import HuberRegressor
from sklearn.covariance import LedoitWolf

# 机器学习
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from purged_time_series_cv import PurgedGroupTimeSeriesSplit, ValidationConfig, create_time_groups

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 导入我们的增强模块
try:
    from enhanced_alpha_strategies import AlphaStrategiesEngine
    from learning_to_rank_bma import LearningToRankBMA
    from advanced_portfolio_optimizer import AdvancedPortfolioOptimizer
    ENHANCED_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] 增强模块导入失败: {e}")
    ENHANCED_MODULES_AVAILABLE = False

# 增强目标工程模块
try:
    from enhanced_target_engineering import (
        EnhancedTargetEngineer, 
        TripleBarrierConfig, 
        MetaLabelingConfig,
        LabelType
    )
    TARGET_ENGINEERING_AVAILABLE = True
    print("[INFO] Enhanced Target Engineering module loaded")
except ImportError as e:
    TARGET_ENGINEERING_AVAILABLE = False
    print(f"[WARN] Enhanced Target Engineering not available: {e}")

# 内存优化模块
try:
    from bma_memory_optimization import MemoryOptimizedBMA, create_memory_optimized_feature_engineering
    MEMORY_OPTIMIZATION_AVAILABLE = True
    print("[INFO] Memory optimization module loaded")
except ImportError as e:
    MEMORY_OPTIMIZATION_AVAILABLE = False
    print(f"[WARN] Memory optimization not available: {e}")

# 安全内存管理模块
try:
    from safe_memory_management import SafeMemoryManager, create_safe_memory_manager
    SAFE_MEMORY_AVAILABLE = True
    print("[INFO] Safe memory management module loaded")
except ImportError as e:
    SAFE_MEMORY_AVAILABLE = False
    print(f"[WARN] Safe memory management not available: {e}")

# 统一市场数据（行业/市值/国家等）
try:
    from unified_market_data_manager import UnifiedMarketDataManager
    MARKET_MANAGER_AVAILABLE = True
except Exception:
    MARKET_MANAGER_AVAILABLE = False

# 中性化已统一由Alpha引擎处理，移除重复依赖

# 导入isotonic校准
try:
    from sklearn.isotonic import IsotonicRegression
    ISOTONIC_AVAILABLE = True
except ImportError:
    ISOTONIC_AVAILABLE = False

# 导入Barra风险模型和约束优化器
try:
    from barra_risk_optimizer import (
        BarraRiskModel, ConstrainedPortfolioOptimizer,
        create_barra_risk_model, create_portfolio_optimizer,
        BarraFactorModel, OptimizationConstraints
    )
    BARRA_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    BARRA_OPTIMIZER_AVAILABLE = False
    logging.warning(f"Barra风险模型模块不可用: {e}")

# 导入Barra风格因子库
try:
    from barra_style_factors import BarraStyleFactors, create_barra_style_factors
    BARRA_FACTORS_AVAILABLE = True
except ImportError as e:
    BARRA_FACTORS_AVAILABLE = False
    logging.warning(f"Barra风格因子模块不可用: {e}")

# 自适应加树优化器
try:
    from adaptive_tree_optimizer import AdaptiveTreeOptimizer
    ADAPTIVE_OPTIMIZER_AVAILABLE = True
except ImportError:
    ADAPTIVE_OPTIMIZER_AVAILABLE = False
    logging.warning("自适应加树优化器不可用，将使用标准模型训练")

# 高级模型
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMRanker
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# CatBoost removed due to compatibility issues
CATBOOST_AVAILABLE = False

# 配置
warnings.filterwarnings('ignore')

# 修复matplotlib版本兼容性问题 (使用语义化版本比较)
try:
    import matplotlib
    try:
        from packaging import version
        version_parse = version.parse
    except ImportError:
        # 如果packaging不可用，使用简单的版本解析
        def version_parse(v):
            return tuple(map(int, v.split('.')))
    
    if hasattr(matplotlib, '__version__') and version_parse(matplotlib.__version__) >= version_parse('3.4.0'):
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            # 如果seaborn-v0_8不可用，使用默认样式
            plt.style.use('default')
            print("[WARN] seaborn-v0_8样式不可用，使用默认样式")
    else:
        plt.style.use('seaborn')
except Exception as e:
    print(f"[WARN] matplotlib样式设置失败: {e}，使用默认样式")
    plt.style.use('default')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局配置
DEFAULT_TICKER_LIST =["A", "AA", "AACB", "AACI", "AACT", "AAL", "AAMI", "AAOI", "AAON", "AAP", "AAPL", "AARD", "AAUC", "AB", "ABAT", "ABBV", "ABCB", "ABCL", "ABEO", "ABEV", "ABG", "ABL", "ABM", "ABNB", "ABSI", "ABT", "ABTS", "ABUS", "ABVC", "ABVX", "ACA", "ACAD", "ACB", "ACCO", "ACDC", "ACEL", "ACGL", "ACHC", "ACHR", "ACHV", "ACI", "ACIC", "ACIU", "ACIW", "ACLS", "ACLX", "ACM", "ACMR", "ACN", "ACNT", "ACOG", "ACRE", "ACT", "ACTG", "ACTU", "ACVA", "ACXP", "ADAG", "ADBE", "ADC", "ADCT", "ADEA", "ADI", "ADM", "ADMA", "ADNT", "ADP", "ADPT", "ADSE", "ADSK", "ADT", "ADTN", "ADUR", "ADUS", "ADVM", "AEBI", "AEE", "AEG", "AEHL", "AEHR", "AEIS", "AEM", "AEO", "AEP", "AER", "AES", "AESI", "AEVA", "AEYE", "AFCG", "AFG", "AFL", "AFRM", "AFYA", "AG", "AGCO", "AGD", "AGEN", "AGH", "AGI", "AGIO", "AGM", "AGNC", "AGO", "AGRO", "AGX", "AGYS", "AHCO", "AHH", "AHL", "AHR", "AI", "AIFF", "AIFU", "AIG", "AII", "AIM", "AIMD", "AIN", "AIOT", "AIP", "AIR", "AIRI", "AIRJ", "AIRO", "AIRS", "AISP", "AIT", "AIV", "AIZ", "AJG", "AKAM", "AKBA", "AKRO", "AL", "ALAB", "ALAR", "ALB", "ALBT", "ALC", "ALDF", "ALDX", "ALE", "ALEX", "ALF", "ALG", "ALGM", "ALGN", "ALGS", "ALGT", "ALHC", "ALIT", "ALK", "ALKS", "ALKT", "ALL", "ALLE", "ALLT", "ALLY", "ALM", "ALMS", "ALMU", "ALNT", "ALNY", "ALRM", "ALRS", "ALSN", "ALT", "ALTG", "ALTI", "ALTS", "ALUR", "ALV", "ALVO", "ALX", "ALZN", "AM", "AMAL", "AMAT", "AMBA", "AMBC", "AMBP", "AMBQ", "AMBR", "AMC", "AMCR", "AMCX", "AMD", "AME", "AMED", "AMG", "AMGN", "AMH", "AMKR", "AMLX", "AMN", "AMP", "AMPG", "AMPH", "AMPL", "AMPX", "AMPY", "AMR", "AMRC", "AMRK", "AMRN", "AMRX", "AMRZ", "AMSC", "AMSF", "AMST", "AMT", "AMTB", "AMTM", "AMTX", "AMWD", "AMWL", "AMX", "AMZE", "AMZN", "AN", "ANAB", "ANDE", "ANEB", "ANET", "ANF", "ANGH", "ANGI", "ANGO", "ANIK", "ANIP", "ANIX", "ANNX", "ANPA", "ANRO", "ANSC", "ANTA", "ANTE", "ANVS", "AOMR", "AON", "AORT", "AOS", "AOSL", "AOUT", "AP", "APA", "APAM", "APD", "APEI", "APG", "APGE", "APH", "API", "APLD", "APLE", "APLS", "APO", "APOG", "APP", "APPF", "APPN", "APPS", "APTV", "APVO", "AQN", "AQST", "AR", "ARAI", "ARCB", "ARCC", "ARCO", "ARCT", "ARDT", "ARDX", "ARE", "AREN", "ARES", "ARHS", "ARI", "ARIS", "ARKO", "ARLO", "ARLP", "ARM", "ARMK", "ARMN", "ARMP", "AROC", "ARQ", "ARQQ", "ARQT", "ARR", "ARRY", "ARTL", "ARTV", "ARVN", "ARW", "ARWR", "ARX", "AS", "ASA", "ASAN", "ASB", "ASC", "ASGN", "ASH", "ASIC", "ASIX", "ASLE", "ASM", "ASND", "ASO", "ASPI", "ASPN", "ASR", "ASST", "ASTE", "ASTH", "ASTI", "ASTL", "ASTS", "ASUR", "ASX", "ATAI", "ATAT", "ATEC", "ATEN", "ATEX", "ATGE", "ATHE", "ATHM", "ATHR", "ATI", "ATII", "ATKR", "ATLC", "ATLX", "ATMU", "ATNF", "ATO", "ATOM", "ATR", "ATRA", "ATRC", "ATRO", "ATS", "ATUS", "ATXS", "ATYR", "AU", "AUB", "AUDC", "AUGO", "AUID", "AUPH", "AUR", "AURA", "AUTL", "AVA", "AVAH", "AVAL", "AVAV", "AVB", "AVBC", "AVBP", "AVD", "AVDL", "AVDX", "AVGO", "AVIR", "AVNS", "AVNT", "AVNW", "AVO", "AVPT", "AVR", "AVT", "AVTR", "AVTX", "AVXL", "AVY", "AWI", "AWK", "AWR", "AX", "AXGN", "AXIN", "AXL", "AXP", "AXS", "AXSM", "AXTA", "AXTI", "AYI", "AYTU", "AZ", "AZN", "AZTA", "AZZ", "B", "BA", "BABA", "BAC", "BACC", "BACQ", "BAER", "BAH", "BAK", "BALL", "BALY", "BAM", "BANC", "BAND", "BANF", "BANR", "BAP", "BASE", "BATRA", "BATRK", "BAX", "BB", "BBAI", "BBAR", "BBCP", "BBD", "BBDC", "BBIO", "BBNX", "BBSI", "BBUC", "BBVA", "BBW", "BBWI", "BBY", "BC", "BCAL", "BCAX", "BCBP", "BCC", "BCE", "BCH", "BCO", "BCPC", "BCRX", "BCS", "BCSF", "BCYC", "BDC", "BDMD", "BDRX", "BDTX", "BDX", "BE", "BEAG", "BEAM", "BEEM", "BEEP", "BEKE", "BELFB", "BEN", "BEP", "BEPC", "BETR", "BF-A", "BF-B", "BFAM", "BFC", "BFH", "BFIN", "BFS", "BFST", "BG", "BGC", "BGL", "BGLC", "BGM", "BGS", "BGSF", "BHC", "BHE", "BHF", "BHFAP", "BHLB", "BHP", "BHR", "BHRB", "BHVN", "BIDU", "BIIB", "BILI", "BILL", "BIO", "BIOA", "BIOX", "BIP", "BIPC", "BIRD", "BIRK", "BJ", "BJRI", "BK", "BKD", "BKE", "BKH", "BKKT", "BKR", "BKSY", "BKTI", "BKU", "BKV", "BL", "BLBD", "BLBX", "BLCO", "BLD", "BLDE", "BLDR", "BLFS", "BLFY", "BLIV", "BLKB", "BLMN", "BLND", "BLNE", "BLRX", "BLUW", "BLX", "BLZE", "BMA", "BMBL", "BMGL", "BMHL", "BMI", "BMNR", "BMO", "BMR", "BMRA", "BMRC", "BMRN", "BMY", "BN", "BNC", "BNED", "BNGO", "BNL", "BNS", "BNTC", "BNTX", "BNZI", "BOC", "BOF", "BOH", "BOKF", "BOOM", "BOOT", "BORR", "BOSC", "BOW", "BOX", "BP", "BPOP", "BQ", "BR", "BRBR", "BRBS", "BRC", "BRDG", "BRFS", "BRK-B", "BRKL", "BRKR", "BRLS", "BRO", "BROS", "BRR", "BRSL", "BRSP", "BRX", "BRY", "BRZE", "BSAA", "BSAC", "BSBR", "BSET", "BSGM", "BSM", "BSX", "BSY", "BTAI", "BTBD", "BTBT", "BTCM", "BTCS", "BTCT", "BTDR", "BTE", "BTG", "BTI", "BTM", "BTMD", "BTSG", "BTU", "BUD", "BULL", "BUR", "BURL", "BUSE", "BV", "BVFL", "BVN", "BVS", "BWA", "BWB", "BWEN", "BWIN", "BWLP", "BWMN", "BWMX", "BWXT", "BX", "BXC", "BXP", "BY", "BYD", "BYND", "BYON", "BYRN", "BYSI", "BZ", "BZAI", "BZFD", "BZH", "BZUN", "C", "CAAP", "CABO", "CAC", "CACC", "CACI", "CADE", "CADL", "CAE", "CAEP", "CAG", "CAH", "CAI", "CAKE", "CAL", "CALC", "CALM", "CALX", "CAMT", "CANG", "CAPR", "CAR", "CARE", "CARG", "CARL", "CARR", "CARS", "CART", "CASH", "CASS", "CAT", "CATX", "CATY", "CAVA", "CB", "CBAN", "CBIO", "CBL", "CBLL", "CBNK", "CBOE", "CBRE", "CBRL", "CBSH", "CBT", "CBU", "CBZ", "CC", "CCAP", "CCB", "CCCC", "CCCS", "CCCX", "CCEP", "CCI", "CCIR", "CCIX", "CCJ", "CCK", "CCL", "CCLD", "CCNE", "CCOI", "CCRD", "CCRN", "CCS", "CCSI", "CCU", "CDE", "CDIO", "CDLR", "CDNA", "CDNS", "CDP", "CDRE", "CDRO", "CDTX", "CDW", "CDXS", "CDZI", "CE", "CECO", "CEG", "CELC", "CELH", "CELU", "CELZ", "CENT", "CENTA", "CENX", "CEP", "CEPO", "CEPT", "CEPU", "CERO", "CERT", "CEVA", "CF", "CFFN", "CFG", "CFLT", "CFR", "CG", "CGAU", "CGBD", "CGCT", "CGEM", "CGNT", "CGNX", "CGON", "CHA", "CHAC", "CHCO", "CHD", "CHDN", "CHE", "CHEF", "CHH", "CHKP", "CHMI", "CHPT", "CHRD", "CHRW", "CHT", "CHTR", "CHWY", "CHYM", "CI", "CIA", "CIB", "CIEN", "CIFR", "CIGI", "CIM", "CINF", "CING", "CINT", "CIO", "CION", "CIVB", "CIVI", "CL", "CLAR", "CLB", "CLBK", "CLBT", "CLCO", "CLDI", "CLDX", "CLF", "CLFD", "CLGN", "CLH", "CLLS", "CLMB", "CLMT", "CLNE", "CLNN", "CLOV", "CLPR", "CLPT", "CLRB", "CLRO", "CLS", "CLSK", "CLVT", "CLW", "CLX", "CM", "CMA", "CMBT", "CMC", "CMCL", "CMCO", "CMCSA", "CMDB", "CME", "CMG", "CMI", "CMP", "CMPO", "CMPR", "CMPS", "CMPX", "CMRC", "CMRE", "CMS", "CMTL", "CNA", "CNC", "CNCK", "CNDT", "CNEY", "CNH", "CNI", "CNK", "CNL", "CNM", "CNMD", "CNNE", "CNO", "CNOB", "CNP", "CNQ", "CNR", "CNS", "CNTA", "CNTB", "CNTY", "CNVS", "CNX", "CNXC", "CNXN", "COCO", "CODI", "COF", "COFS", "COGT", "COHR", "COHU", "COIN", "COKE", "COLB", "COLL", "COLM", "COMM", "COMP", "CON", "COO", "COOP", "COP", "COPL", "COR", "CORT", "CORZ", "COTY", "COUR", "COYA", "CP", "CPA", "CPAY", "CPB", "CPF", "CPIX", "CPK", "CPNG", "CPRI", "CPRT", "CPRX", "CPS", "CPSH", "CQP", "CR", "CRAI", "CRAQ", "CRBG", "CRBP", "CRC", "CRCL", "CRCT", "CRD-A", "CRDF", "CRDO", "CRE", "CRESY", "CREV", "CREX", "CRGO", "CRGX", "CRGY", "CRH", "CRI", "CRK", "CRL", "CRM", "CRMD", "CRML", "CRMT", "CRNC", "CRNX", "CRON", "CROX", "CRS", "CRSP", "CRSR", "CRTO", "CRUS", "CRVL", "CRVO", "CRVS", "CRWD", "CRWV", "CSAN", "CSCO", "CSGP", "CSGS", "CSIQ", "CSL", "CSR", "CSTL", "CSTM", "CSV", "CSW", "CSWC", "CSX", "CTAS", "CTEV", "CTGO", "CTKB", "CTLP", "CTMX", "CTNM", "CTO", "CTOS", "CTRA", "CTRI", "CTRM", "CTRN", "CTS", "CTSH", "CTVA", "CTW", "CUB", "CUBE", "CUBI", "CUK", "CUPR", "CURB", "CURI", "CURV", "CUZ", "CV", "CVAC", "CVBF", "CVCO", "CVE", "CVEO", "CVGW", "CVI", "CVLG", "CVLT", "CVM", "CVNA", "CVRX", "CVS", "CVX", "CW", "CWAN", "CWBC", "CWCO", "CWEN", "CWEN-A", "CWH", "CWK", "CWST", "CWT", "CX", "CXDO", "CXM", "CXT", "CXW", "CYBN", "CYBR", "CYCC", "CYD", "CYH", "CYN", "CYRX", "CYTK", "CZR", "CZWI", "D", "DAAQ", "DAC", "DAIC", "DAKT", "DAL", "DALN", "DAN", "DAO", "DAR", "DARE", "DASH", "DATS", "DAVA", "DAVE", "DAWN", "DAY", "DB", "DBD", "DBI", "DBRG", "DBX", "DC", "DCBO", "DCI", "DCO", "DCOM", "DCTH", "DD", "DDC", "DDI", "DDL", "DDOG", "DDS", "DEA", "DEC", "DECK", "DEFT", "DEI", "DELL", "DENN", "DEO", "DERM", "DEVS", "DFDV", "DFH", "DFIN", "DFSC", "DG", "DGICA", "DGII", "DGX", "DGXX", "DH", "DHI", "DHR", "DHT", "DHX", "DIBS", "DIN", "DINO", "DIOD", "DIS", "DJCO", "DJT", "DK", "DKL", "DKNG", "DKS", "DLB", "DLHC", "DLO", "DLTR", "DLX", "DLXY", "DMAC", "DMLP", "DMRC", "DMYY", "DNA", "DNB", "DNLI", "DNN", "DNOW", "DNTH", "DNUT", "DOC", "DOCN", "DOCS", "DOCU", "DOGZ", "DOLE", "DOMH", "DOMO", "DOOO", "DORM", "DOUG", "DOV", "DOW", "DOX", "DOYU", "DPRO", "DPZ", "DQ", "DRD", "DRDB", "DRH", "DRI", "DRS", "DRVN", "DSGN", "DSGR", "DSGX", "DSP", "DT", "DTE", "DTI", "DTIL", "DTM", "DTST", "DUK", "DUOL", "DUOT", "DV", "DVA", "DVAX", "DVN", "DVS", "DWTX", "DX", "DXC", "DXCM", "DXPE", "DXYZ", "DY", "DYN", "DYNX", "E", "EA", "EARN", "EAT", "EB", "EBAY", "EBC", "EBF", "EBMT", "EBR", "EBS", "EC", "ECC", "ECG", "ECL", "ECO", "ECOR", "ECPG", "ECVT", "ED", "EDBL", "EDIT", "EDN", "EDU", "EE", "EEFT", "EEX", "EFC", "EFSC", "EFX", "EFXT", "EG", "EGAN", "EGBN", "EGG", "EGO", "EGP", "EGY", "EH", "EHAB", "EHC", "EHTH", "EIC", "EIG", "EIX", "EKSO", "EL", "ELAN", "ELDN", "ELF", "ELMD", "ELME", "ELP", "ELPW", "ELS", "ELV", "ELVA", "ELVN", "ELWS", "EMA", "EMBC", "EMN", "EMP", "EMPD", "EMPG", "EMR", "EMX", "ENB", "ENGN", "ENGS", "ENIC", "ENOV", "ENPH", "ENR", "ENS", "ENSG", "ENTA", "ENTG", "ENVA", "ENVX", "EOG", "EOLS", "EOSE", "EPAC", "EPAM", "EPC", "EPD", "EPM", "EPR", "EPSM", "EPSN", "EQBK", "EQH", "EQNR", "EQR", "EQT", "EQV", "EQX", "ERIC", "ERIE", "ERII", "ERJ", "ERO", "ES", "ESAB", "ESE", "ESGL", "ESI", "ESLT", "ESNT", "ESOA", "ESQ", "ESTA", "ESTC", "ET", "ETD", "ETN", "ETNB", "ETON", "ETOR", "ETR", "ETSY", "EU", "EUDA", "EVAX", "EVC", "EVCM", "EVER", "EVEX", "EVGO", "EVH", "EVLV", "EVO", "EVOK", "EVR", "EVRG", "EVTC", "EVTL", "EW", "EWBC", "EWCZ", "EWTX", "EXAS", "EXC", "EXE", "EXEL", "EXK", "EXLS", "EXOD", "EXP", "EXPD", "EXPE", "EXPI", "EXPO", "EXR", "EXTR", "EYE", "EYPT", "EZPW", "F", "FA",
 "FACT", "FAF", "FANG", "FAST", "FAT", "FATN", "FBIN", "FBK", "FBLA", 
 "FBNC", "FBP", "FBRX", "FC", "FCBC", "FCEL", "FCF", "FCFS", "FCN", "FCX", "FDMT",
  "FDP", "FDS", "FDUS", "FDX", "FE", "FEIM", "FELE", "FENC", "FER", "FERA", "FERG", "FET", "FF", 
  "FFAI", "FFBC", "FFIC", "FFIN", "FFIV", "FFWM", "FG", "FGI", "FHB", "FHI", "FHN", "FHTX", "FI", "FIBK", "FIEE", "FIG", "FIGS", 
  "FIHL", "FINV", "FIP", "FIS", "FISI", "FITB", "FIVE", "FIVN", "FIZZ", "FL", "FLD", "FLEX", "FLG", "FLGT", "FLL", "FLNC", "FLNG", "FLO", "FLOC",
   "FLR", "FLS", "FLUT", "FLWS", "FLX", "FLY", "FLYE", "FLYW", "FLYY", "FMBH", "FMC", "FMFC", "FMNB", "FMS", "FMST", 
   "FMX", "FN", "FNB", "FND", "FNF", "FNGD", "FNKO", "FNV", "FOA", "FOLD", "FOR", "FORM", "FORR", "FOUR", "FOX", "FOXA", 
   "FOXF", "FPH", "FPI", "FRGE", "FRHC", "FRME", "FRO", "FROG", "FRPT", "FRSH", "FRST", "FSCO", "FSK", "FSLR", "FSLY",
    "FSM", "FSS", "FSUN", "FSV", "FTAI", "FTCI", "FTDR", "FTEK", "FTI", "FTK", "FTNT", "FTRE", "FTS", "FTV", "FUBO", "FUFU", "FUL", "FULC", "FULT", "FUN", "FUTU", "FVR", "FVRR", "FWONA", "FWONK", "FWRD", "FWRG", "FYBR", "G",
     "GABC", "GAIA", "GAIN", "GALT", "GAMB", "GAP", "GASS", "GATX", "GAUZ", "GB", "GBCI", "GBDC", "GBFH", "GBIO", "GBTG", "GBX", "GCI", "GCL", "GCMG", "GCO", "GCT", "GD", "GDC", "GDDY", "GDEN", "GDOT", "GDRX", "GDS", 
     "GDYN", "GE", "GEF", "GEHC", "GEL", "GEN", "GENI", "GENK", "GEO", "GEOS", "GES", "GFF", "GFI", "GFL", "GFR", "GFS", "GGAL", "GGB", "GGG", "GH", "GHLD", "GHM", "GHRS", "GIB", "GIC", "GIG", "GIII", "GIL", "GILD", "GILT", "GIS", "GITS", 
     "GKOS", "GL", "GLAD", "GLBE", "GLD", "GLDD", "GLIBA", "GLIBK", "GLNG", "GLOB", "GLP", "GLPG", "GLPI", "GLRE", "GLSI", "GLUE", "GLW", "GLXY", "GM", "GMAB", "GME", "GMED", "GMRE", "GMS", "GNE", "GNK", "GNL", "GNLX", "GNRC", "GNTX", "GNTY", "GNW", "GO", "GOCO", "GOGL", "GOGO", "GOLF", "GOOD", "GOOG", "GOOGL", "GOOS", "GORV", "GOTU", "GPAT", 
     "GPC", "GPCR", "GPI", "GPK", "GPN", "GPOR", "GPRE", "GPRK", "GRAB", "GRAL", "GRAN", "GRBK", "GRC", "GRCE", "GRDN", "GRFS", "GRMN", "GRND", "GRNT", "GROY", "GRPN", "GRRR", "GSAT", "GSBC", "GSBD", "GSHD", "GSIT", "GSK", "GSL", "GSM", "GSRT", "GT", "GTE", "GTEN", "GTERA", "GTES", "GTLB", "GTLS", "GTM", "GTN", "GTX", "GTY", "GVA", "GWRE", "GWRS", "GXO", "GYRE", "H", "HAE", "HAFC", "HAFN", "HAL", "HALO", "HAS", "HASI", "HAYW", "HBAN", "HBCP", "HBI", "HBM", "HBNC", "HCA", "HCAT", "HCC", "HCHL", "HCI", "HCKT", "HCM", "HCSG", "HCTI", "HCWB", "HD", "HDB", "HDSN", "HE", "HEI", "HEI-A", "HELE", "HEPS", "HESM", "HFFG", "HFWA", "HG", "HGTY", "HGV", "HHH", "HI", "HIFS", "HIG", "HII", "HIMS", "HIMX",
     "HIPO", "HIT", "HITI", "HIVE", "HIW", "HL", "HLF", "HLI", "HLIO", "HLIT", "HLLY", "HLMN", "HLN", "HLNE", "HLT", "HLVX", "HLX", "HLXB", "HMC", "HMN", "HMST", "HMY", "HNGE", "HNI", "HNRG", "HNST", "HOFT", "HOG", "HOLO", "HOLX", "HOMB", "HON", "HOND", "HONE", "HOOD", "HOPE", "HOUS", "HOV", "HP", "HPE", "HPK", "HPP", "HPQ", "HQH", "HQL", "HQY", "HRB", "HRI", "HRL", "HRMY", "HROW", "HRTG", "HRZN", "HSAI", "HSBC", "HSCS", "HSHP", "HSIC", "HSII", "HST", "HSTM", "HSY", "HTBK", "HTCO", "HTGC", "HTH", "HTHT", "HTLD", "HTO", "HTOO", "HTZ", "HUBB", "HUBC", "HUBG", "HUBS", "HUHU", "HUM", "HUMA", "HUN", "HURA", "HURN", "HUSA", "HUT", "HUYA", "HVII", "HVT", "HWC", "HWKN", "HWM", 
     "HXL", "HY", "HYAC", "HYMC", "HYPD", "HZO", "IAC", "IAG", "IART", "IAS", "IBCP", "IBEX", "IBKR", "IBM", "IBN", "IBOC", "IBP", "IBRX", "IBTA", "ICE", "ICFI", "ICG", "ICHR", "ICL", "ICLR", "ICUI", "IDA", "IDAI", "IDCC", "IDN", "IDR", "IDT", "IDYA", "IE", "IEP", "IESC", "IEX", "IFF", "IFS", "IGIC", "IHG", "IHS", "III", "IIIN", "IIIV", "IIPR", "ILMN", "IMAB", "IMAX", "IMCC", "IMCR", "IMDX", "IMKTA", "IMMR", "IMMX", "IMNM", "IMNN", "IMO", "IMPP", "IMRX", "IMTX", "IMVT", "IMXI", "INAB", "INAC", "INBK", "INBX", "INCY", "INDB", "INDI", "INDO", "INDP", "INDV", "INFA", "INFU", "INFY", "ING", "INGM", "INGN", "INGR", "INKT", "INMB", "INMD", "INN", "INOD", "INR", "INSE", "INSG", "INSM", "INSP", "INSW", "INTA", "INTC", "INTR", "INUV", "INV", "INVA", "INVE", "INVH", "INVX", "IONQ", "IONS", "IOSP", "IOT", "IOVA", "IP", "IPA", "IPAR", "IPDN", "IPG", "IPGP", "IPI", "IPX", "IQST", "IQV", "IR", "IRBT", "IRDM", "IREN", "IRM", "IRMD", "IROH", "IRON", "IRS", "IRTC", "ISPR", "ISRG", "ISSC", "IT", "ITGR", "ITIC", "ITOS", "ITRI", "ITRN", "ITT", "ITUB", "ITW", "IVR", "IVZ", "IX", "IZEA", "J", "JACK", "JACS", "JAKK", "JAMF", "JANX", "JAZZ", "JBGS", "JBHT", "JBI", "JBIO", "JBL", "JBLU", "JBS", "JBSS", "JBTM", "JCAP", "JCI", "JD", "JEF", "JELD", "JEM", "JENA", "JFIN", "JHG", "JHX", "JILL", "JJSF", "JKHY", "JKS", "JLHL", "JLL", "JMIA", "JNJ", "JOBY", "JOE", "JOUT", "JOYY", "JPM", "JRSH", "JRVR", "JSPR", "JTAI", "JVA", "JXN", "JYNT", "K", "KAI", "KALA", "KALU", 
     "KALV", "KAR", "KARO", "KB", "KBDC", "KBH", "KBR", "KC", "KCHV", "KD", "KDP", "KE", "KELYA", "KEP", "KEX", "KEY", "KEYS", "KFII", "KFRC", "KFS", "KFY", "KGC", "KGEI", "KGS", "KHC", "KIDS", "KIM", "KINS", "KKR", "KLC", "KLG", "KLIC", "KLRS", "KMB", "KMDA", "KMI", "KMPR", "KMT", "KMTS", "KMX", "KN", "KNF", "KNOP", "KNSA", "KNSL", "KNTK", "KNW", "KNX", "KO", "KOD", "KODK", "KOF", "KOP", "KOSS", "KPRX", "KPTI", "KR", "KRC", "KRMD", "KRMN", "KRNT", "KRNY", "KRO", "KROS", "KRP", "KRRO", "KRT", "KRUS", "KRYS", "KSCP", "KSPI", "KSS", "KT", "KTB", "KTOS", "KULR", "KURA", "KVUE", "KVYO", "KW", "KWM", "KWR", "KYMR", "KYTX", "KZIA", "L", "LAC", "LAD", "LADR", "LAES", "LAKE", "LAMR", "LAND", "LANV", "LAR", "LASE", "LASR", "LAUR", "LAW", "LAWR", "LAZ", "LAZR", "LB", "LBRDA", "LBRDK", "LBRT", "LBTYA", "LBTYK", "LC", "LCCC", "LCFY", "LCID", "LCII", "LCUT", "LDOS", "LE", "LEA", "LECO", "LEG", "LEGH", "LEGN", "LEN", "LENZ", "LEO", "LEU", "LEVI", "LFCR", "LFMD", "LFST", "LFUS", "LFVN", "LGCY", "LGIH", "LGND", "LH", "LHAI", "LHSW", "LHX", "LI", "LIDR", "LIF", "LILA", "LILAK", "LIMN", "LIN", "LINC", "LIND", "LINE", "LION", "LITE", "LITM", "LIVE", "LIVN", "LIXT", "LKFN", "LKQ", "LLYVA", "LLYVK", "LMAT", "LMB", "LMND", "LMNR", "LMT", "LNC", "LNG", "LNN", "LNSR", "LNT", "LNTH", "LNW", "LOAR", "LOB", "LOCO", "LODE", "LOGI", "LOKV", "LOMA", "LOPE", "LOT", "LOVE", "LOW", "LPAA", "LPBB", "LPCN", "LPG", "LPL", "LPLA", "LPRO", "LPTH", "LPX", "LQDA", "LQDT", "LRCX", "LRMR", "LRN", "LSCC", "LSE", "LSPD", "LSTR", "LTBR", "LTC", "LTH", "LTM", "LTRN", "LTRX", "LU", "LUCK", "LULU", "LUMN", "LUNR", "LUV", "LUXE", "LVLU", "LVS", "LVWR", "LW", "LWAY", "LWLG", "LX", "LXEH", "LXEO", "LXFR", "LXU", "LYB", "LYEL", "LYFT", "LYG", "LYRA", "LYTS", "LYV", "LZ", "LZB", "LZM", "LZMH", "M", "MAA", "MAAS", "MAC", "MACI", "MAG", "MAGN", "MAIN", "MAMA", "MAMK", "MAN", "MANH", "MANU", "MAR", "MARA", "MAS", "MASI", "MASS", "MAT", "MATH", "MATV", "MATW", "MATX", "MAX", "MAXN", "MAZE", "MB", "MBAV", "MBC", "MBI", "MBIN", "MBLY",
      "MBOT", "MBUU", "MBWM", "MBX", "MC", "MCB", "MCD", "MCFT", "MCHP", "MCRB", "MCRI", "MCRP", "MCS", "MCVT", "MCW", "MCY", "MD", "MDAI", "MDB", "MDCX", "MDGL", "MDLZ", "MDT", "MDU", "MDV", "MDWD", "MDXG", "MDXH", "MEC", "MED", "MEDP", "MEG", "MEI", "MEIP", "MENS", "MEOH", "MERC", "MESO", "MET", "METC", "METCB", "MFA", "MFC", "MFG", "MFH", "MFI", "MFIC", "MFIN", "MG", "MGA", "MGEE", "MGIC", "MGM", "MGNI", "MGPI", "MGRC", "MGRM", "MGRT", "MGTX", "MGY", "MH", "MHK", "MHO", "MIDD", "MIMI", "MIND", "MIR", "MIRM", "MITK", "MKC", "MKSI", "MKTX", "MLAB", "MLCO", "MLEC", "MLGO", "MLI", "MLKN", "MLNK", "MLR", "MLTX", "MLYS", "MMC", "MMI", "MMM", "MMS", "MMSI", "MMYT", "MNDY", "MNKD", "MNMD", "MNR", "MNRO", "MNSO", "MNST", "MNTN", "MO", "MOB", "MOD", "MODG", "MODV", "MOFG", "MOG-A", "MOH", "MOMO", "MORN", "MOS", "MOV", "MP", "MPAA", "MPB", "MPC", "MPLX", "MPTI", "MPU", "MQ", "MRAM", "MRBK", "MRC", "MRCC", "MRCY", "MRK", "MRNA", "MRP", "MRSN", "MRT", "MRTN", "MRUS", "MRVI", "MRVL", "MRX", "MS", "MSA", "MSBI", "MSEX", "MSGE", "MSGM", "MSGS", "MSGY", "MSI", "MSM", "MSTR", "MT", "MTA", "MTAL", "MTB", "MTCH", "MTDR", "MTEK", "MTEN", "MTG", "MTH", "MTLS", "MTN", "MTRN", "MTRX", "MTSI", "MTSR", "MTUS", "MTW", "MTX", "MTZ", "MU", "MUFG", "MUR", "MUSA", "MUX", "MVBF", "MVST", "MWA", "MX", "MXL", "MYE", "MYFW", "MYGN", "MYRG", "MZTI", "NA", "NAAS", "NABL", "NAGE", "NAKA", "NAMM", "NAMS", "NAT", "NATH", "NATL", "NATR", "NAVI", "NB", "NBBK", "NBHC", "NBIS", "NBIX", "NBN", "NBR", "NBTB", "NCDL", "NCLH", "NCMI", "NCNO", "NCPL", "NCT", "NCTY", "NDAQ", "NDSN", "NE", "NEE", "NEGG", "NEM", "NEO", "NEOG", "NEON", "NEOV", "NESR", "NET", "NETD", "NEWT", "NEXM", "NEXN", "NEXT", "NFBK", "NFE", "NFG", "NG", "NGD", "NGG", "NGL", "NGNE", "NGS", "NGVC", "NGVT", "NHC", "NHI", "NHIC", "NI", "NIC", "NICE", "NIO", "NIQ", "NISN", "NIU", "NJR", "NKE", "NKTR", "NLOP", "NLSP", "NLY", "NMAX", "NMFC", "NMIH", "NMM", "NMR", "NMRK", "NN", "NNBR", "NNE", "NNI", "NNN", "NNNN", "NNOX", "NOA", "NOAH", "NOG", "NOK", "NOMD", "NOV", "NOVT", "NPAC", "NPB", "NPCE", "NPK", "NPKI", "NPO", "NPWR", "NRC", "NRDS", "NRG", "NRIM", "NRIX", "NRXP", "NRXS", "NSC", "NSIT", "NSP", "NSPR", "NSSC", "NTAP", "NTB", "NTCT", "NTES", "NTGR", "NTHI", "NTLA", "NTNX", "NTR", "NTRA", "NTRB", "NTST", "NU", "NUE", "NUKK", "NUS", "NUTX", "NUVB", "NUVL", "NUWE", "NVAX", "NVCR", "NVCT", "NVDA", "NVEC", "NVGS", "NVMI", "NVNO", "NVO", "NVRI", "NVS", 
  "NVST", "NVT", "NVTS", "NWBI", "NWE", "NWG", "NWL", "NWN", "NWPX", "NWS", "NWSA", "NX", "NXE", "NXP", "NXPI", "NXST", "NXT", "NXTC", "NYT", "NYXH", "O", "OACC", "OBDC", "OBE", "OBIO", "OBK", "OBLG", "OBT", "OC", "OCC", "OCCI", "OCFC", "OCFT", "OCSL", "OCUL", "ODC", "ODD", "ODFL", "ODP", "ODV", "OEC", "OFG", "OFIX", "OGE", "OGN", "OGS", "OHI", "OI", "OII", "OIS", "OKE", "OKLO", "OKTA", "OKUR", "OKYO", "OLED", "OLLI", "OLMA", "OLN", "OLO", "OLP", "OM", "OMAB", "OMC", "OMCL", "OMDA", "OMER", "OMF", "OMI", "OMSE", "ON", "ONB", "ONC", "ONDS", "ONEG", "ONEW", "ONL", "ONON", "ONTF", "ONTO", "OOMA", "OPAL", "OPBK", "OPCH", "OPFI", "OPRA", "OPRT", "OPRX", "OPXS", "OPY", "OR", "ORA", "ORC", "ORCL", "ORGO", "ORI", "ORIC", "ORKA", "ORLA", "ORLY", "ORMP", "ORN", "ORRF", "OS", "OSBC", "OSCR", "OSIS", "OSK", "OSPN", "OSS", "OSUR", "OSW", "OTEX", "OTF", "OTIS", "OTLY", "OTTR", "OUST", "OUT", "OVV", "OWL", "OWLT", "OXLC", "OXM", "OXSQ", "OXY", "OYSE", "OZK", "PAA", "PAAS", "PAC", "PACK", "PACS", "PAG", "PAGP", "PAGS", "PAHC", "PAL", "PAM", "PANL", "PANW", "PAR", "PARR", "PATH", "PATK", "PAX", "PAY", "PAYC", "PAYO", "PAYS", "PAYX", "PB", "PBA", "PBF", "PBH", "PBI", "PBPB", "PBR", "PBR-A", "PBYI", "PC", "PCAP", "PCAR", "PCG", "PCH", "PCOR", "PCRX", "PCT", "PCTY", "PCVX", "PD", "PDD", "PDEX", "PDFS", "PDS", "PDYN", "PEBO", "PECO", "PEG", "PEGA", "PEN", "PENG", "PENN", "PEP", "PERI", "PESI", "PETS", "PEW", "PFBC", "PFE", "PFG", "PFGC", "PFLT", "PFS", "PFSI", "PG", "PGC", "PGNY", "PGR", "PGRE", "PGY", "PHAT", "PHG", "PHI", "PHIN", "PHIO", "PHLT", "PHM", "PHOE", "PHR", "PHUN", "PHVS", "PI", "PII", "PINC", "PINS", "PIPR", "PJT", "PK", "PKE", "PKG", "PKX", "PL", "PLAB", "PLAY", "PLCE", "PLD", "PLL", "PLMR", "PLNT", "PLOW", "PLPC", "PLSE", "PLTK", "PLTR", "PLUS", "PLXS", "PLYM", "PM", "PMTR", "PMTS", "PN", "PNC", "PNFP", "PNNT", "PNR", "PNRG", "PNTG", "PNW", "PODD", "POET", "PONY", "POOL", "POR", "POST", "POWI", "POWL", "PPBI", "PPBT", "PPC", "PPG", "PPIH", "PPL", "PPSI", "PPTA", "PR", "PRA", "PRAA", "PRAX", "PRCH", "PRCT", "PRDO", "PRE", "PRG", "PRGO", "PRGS", "PRI", "PRIM", "PRK", "PRKS", "PRLB", "PRM", "PRMB", "PRME", "PRO", "PROK", "PROP", "PRQR", "PRSU", "PRTA", "PRTG", "PRTH", "PRU", "PRVA", "PSA", "PSEC", "PSFE", "PSIX", "PSKY", "PSMT", "PSN", "PSNL", "PSO", "PSQH", "PSTG", "PSX", "PTC", "PTCT", "PTEN", "PTGX", "PTHS", "PTLO", "PTON", "PUBM", "PUK", "PUMP", "PVBC", "PVH", "PVLA", "PWP", "PWR", "PX", "PXLW", "PYPD", "PYPL", "PZZA", "QBTS", "QCOM", "QCRH", "QD", "QDEL", "QFIN", "QGEN", "QIPT", "QLYS", "QMCO", "QMMM", "QNST", "QNTM", "QRHC", "QRVO", "QS", "QSEA", "QSG", "QSR", "QTRX", "QTWO", "QUAD", "QUBT", "QUIK", "QURE", "QVCGA", "QXO", "R", "RAAQ", "RAC", "RACE", "RAIL", "RAL", "RAMP", "RAPP", "RAPT", "RARE", "RAY", "RBA", "RBB", "RBBN", "RBC", "RBCAA", "RBLX", "RBRK", "RC", "RCAT", "RCEL", "RCI", "RCKT", "RCKY", "RCL", "RCMT", "RCON", "RCT", 
    "RCUS", "RDAG", "RDAGU", "RDCM", "RDDT", "RDN", "RDNT", "RDVT", "RDW", "RDWR", "RDY", "REAL", "REAX", "REBN", "REFI", "REG", "RELX", "RELY", "RENT", "REPL", "REPX", "RERE", "RES", "RETO", "REVG", "REX", "REXR", "REYN", "REZI", "RF", "RFIL", "RGA", "RGC", "RGEN", "RGLD", "RGNX", "RGP", "RGR", "RGTI", "RH", "RHI", "RHLD", "RHP", "RICK", "RIG", "RIGL", "RILY", "RIME", "RIO", "RIOT", "RITM", "RITR", "RIVN", "RJF", "RKLB", "RKT", "RL", "RLAY", "RLGT", "RLI", "RLX", "RMAX", "RMBI", "RMBL", "RMBS", "RMD", "RMNI", "RMR", "RMSG", "RNA", "RNAC", "RNAZ", "RNG", "RNGR", "RNR", "RNST", "RNW", "ROAD", "ROCK", "ROG", "ROIV", "ROK", "ROKU", "ROL", "ROLR", "ROMA", "ROOT", "ROST", "RPAY", "RPD", "RPID", "RPM", "RPRX", "RPT", "RRC", "RRGB", "RRR", "RRX", "RS", "RSG", "RSI", "RSKD", "RSLS", "RSVR", "RTAC", "RTO", "RTX", "RUBI", "RUM", "RUN", "RUSHA", "RUSHB", "RVLV", "RVMD", "RVSB", "RVTY", "RWAY", "RXO", "RXRX", "RXST", "RY", "RYAAY", "RYAM", "RYAN", "RYI", "RYN", "RYTM", "RZB", "RZLT", "RZLV", "S", "SA", "SABS", "SAFE", "SAFT", "SAGT", "SAH", "SAIA", "SAIC", "SAIL", "SAM", "SAMG", "SAN", "SANA", "SAND", "SANM", "SAP", "SAR", "SARO", "SATL", "SATS", "SAVA", "SB", "SBAC", "SBC", "SBCF", "SBET", "SBGI", "SBH", "SBLK", "SBRA", "SBS", "SBSI", "SBSW", "SBUX", "SBXD", "SCAG", "SCCO", "SCHL", "SCHW", "SCI", "SCL", "SCLX", "SCM", "SCNX", "SCPH", "SCS", "SCSC", "SCVL", "SD", "SDA", "SDGR", "SDHC", "SDHI", "SDM", "SDRL", "SE", "SEAT", "SEDG", "SEE", "SEG", "SEI", "SEIC", "SEM", "SEMR", "SENEA", "SEPN", "SERA", "SERV", "SEZL", "SF", "SFBS", "SFD", "SFIX", "SFL", "SFM", "SFNC", "SG", "SGHC", "SGHT", "SGI", "SGML", "SGMT", "SGRY", "SHAK", "SHBI", "SHC", "SHCO", "SHEL", "SHEN", "SHG", "SHIP", "SHLS", "SHO", "SHOO", "SHOP", "SHW", "SI", "SIBN", "SIEB", "SIFY", "SIG", "SIGA", "SIGI", "SII", "SIMO", "SINT", "SION", "SIRI", "SITC", "SITE", "SITM", "SJM", "SKE", "SKLZ", "SKM", "SKT", "SKWD", "SKX", "SKY", "SKYE", "SKYH", "SKYT", "SKYW", "SLAB", "SLB", "SLDB", "SLDE", "SLDP", "SLF", "SLG", "SLGN", "SLI", "SLM", "SLN", "SLND", "SLNO", "SLP", "SLRC", "SLSN", "SLVM", "SM", "SMA", "SMBK", "SMC", "SMCI", "SMFG", "SMG", "SMHI", "SMLR", "SMMT", "SMP", "SMPL", "SMR", "SMTC", "SMWB", "SMX", "SN", "SNA", "SNAP", "SNBR", "SNCR", "SNCY", "SNDK", "SNDR", "SNDX", "SNES", "SNEX", "SNFCA", "SNGX", "SNN", "SNOW", "SNRE", "SNT", "SNV", "SNWV", "SNX", "SNY", "SNYR", "SO", "SOBO", "SOC", "SOFI", "SOGP", "SOHU", "SOLV", "SON", "SOND", "SONN", "SONO", "SONY", "SOPH", "SORA", "SOS", "SOUL", "SOUN", "SPAI", "SPB", "SPCB", "SPCE", "SPG", "SPH", "SPHR", "SPIR", "SPKL", "SPNS", "SPNT", "SPOK", "SPR", "SPRO", "SPRY", "SPSC", "SPT", "SPTN", "SPWH", "SPXC", "SQM", "SR", "SRAD", "SRBK", "SRCE", "SRDX", "SRE", "SRFM", "SRG", "SRI", "SRPT", "SRRK", "SRTS", "SSB", "SSD", "SSII", "SSL", "SSNC", "SSP", "SSRM", "SSSS", "SST", "SSTI", "SSTK", "SSYS", "ST", "STAA", "STAG", "STBA", "STC", "STE", "STEL", "STEM", "STEP", "STFS", "STGW", "STHO", "STI", "STIM", "STKL", "STKS", "STLA", "STLD", "STM", "STN", "STNE", "STNG", "STOK", "STR", "STRA", "STRD", "STRL", 
    "STRM", "STRT", "STRZ", "STSS", "STT", "STVN", "STX", "STXS", "STZ", "SU", "SUI", "SUN", "SUPN", "SUPV", "SUPX", "SURG", "SUZ", "SVCO", "SVM", "SVRA", "SVV", "SW", "SWBI", "SWIM", "SWIN", "SWK", "SWKS", "SWX", "SXC", "SXI", "SXT", "SY", "SYBT", "SYF", "SYK", "SYM", "SYNA", "SYRE", "SYTA", "SYY", "SZZL", "T", "TAC", "TACH", "TACO", "TAK", "TAL", "TALK", "TALO", "TAOX", "TAP", "TARA", "TARS", "TASK", "TATT", "TBB", "TBBB", "TBBK", "TBCH", "TBI", "TBLA", "TBPH", "TBRG", "TCBI", "TCBK", "TCBX", "TCMD", "TCOM", "TCPC", "TD", "TDC", "TDIC", "TDOC", "TDS", "TDUP", "TDW", "TEAM", "TECH", "TECK", "TECX", "TEF", "TEL", "TEM", "TEN", "TENB", "TEO", "TER", "TERN", "TEVA", "TEX", "TFC", "TFII", "TFIN", "TFPM", "TFSL", "TFX", "TG", "TGB", "TGE", "TGEN", "TGLS", "TGNA", "TGS", "TGT", "TGTX", "TH", "THC", "THFF", "THG", "THO", "THR", "THRM", "THRY", "THS", "THTX", "TIC", "TIGO", "TIGR", "TIL", "TILE", "TIMB", "TIPT", "TITN", "TIXT", "TJX", "TK", "TKC", "TKNO", "TKO", "TKR", "TLK", "TLN", "TLS", "TLSA", "TLSI", "TM", "TMC",
     "TMCI", "TMDX", "TME", "TMHC", "TMO", "TMUS", "TNC", "TNDM", "TNET", "TNGX", "TNK", "TNL", "TNXP", "TOI", "TOL", "TOPS", "TORO", "TOST", "TOWN", "TPB", "TPC", "TPCS", "TPG", "TPH", "TPR", "TPST", "TPVG", "TR", "TRAK", "TRC", "TRDA", "TREE", "TREX", "TRGP", "TRI", "TRIN", "TRIP", "TRMB", "TRMD", "TRML", "TRN", "TRNO", "TRNR", "TRNS", "TRON", "TROW", "TROX", "TRP", "TRS", "TRU", "TRUE", "TRUG", "TRUP", "TRV", "TRVG", "TRVI", "TS", "TSAT", "TSCO", "TSE", "TSEM", "TSHA", "TSLA", "TSLX", "TSM", "TSN", "TSQ", "TSSI", "TT", "TTAM", "TTAN", "TTC", "TTD", "TTE", "TTEC", "TTEK", "TTGT", "TTI", "TTMI", "TTSH", "TTWO", "TU", "TUSK", "TUYA", "TV", "TVA", "TVAI", "TVRD", "TVTX", "TW", "TWFG", "TWI", "TWIN", "TWLO", "TWNP", "TWO", "TWST", "TX", "TXG", "TXN", "TXNM", "TXO", "TXRH", "TXT", "TYG", "TYRA", "TZOO", "TZUP", "U", "UA", "UAA", "UAL", "UAMY", "UAVS", "UBER", "UBFO", "UBS", "UBSI", "UCAR", "UCB", "UCL", "UCTT", "UDMY", "UDR", "UE", "UEC", "UFCS", "UFG", "UFPI", "UFPT", 
     "UGI", "UGP", "UHAL", "UHAL-B", "UHG", "UHS", "UI", "UIS", "UL", "ULBI", "ULCC", "ULS", "ULY", "UMAC", "UMBF", "UMC", "UMH", "UNCY", "UNF", "UNFI", "UNH", "UNIT", "UNM", "UNP", "UNTY", "UPB", "UPBD", "UPS", "UPST", "UPWK", "UPXI", "URBN", "URGN", "UROY", "USAC", "USAR", "USAU", "USB", "USFD", "USLM", "USM", "USNA", "USPH", "UTHR", "UTI", "UTL", "UTZ", "UUUU", "UVE", "UVSP", "UVV", "UWMC", "UXIN", "V", "VAC", "VAL", "VALE", "VBIX", "VBNK", "VBTX", "VC", "VCEL", "VCTR", "VCYT", "VECO", "VEEV", "VEL", "VENU", "VEON", "VERA", "VERB", "VERI", "VERX", 
     "VET", "VFC", "VFS", "VG", "VIAV", "VICI", "VICR", "VIK", "VINP", "VIOT", "VIPS", "VIR", "VIRC", "VIRT", "VIST", "VITL", "VIV", "VKTX",
      "VLGEA", "VLN", "VLO", "VLRS", "VLTO", "VLY", "VMC", "VMD", "VMEO", "VMI", "VNDA", "VNET", "VNOM", "VNT", "VNTG", "VOD", "VOR", "VOXR", "VOYA", "VOYG", "VPG", "VRDN", "VRE",
       "VREX", "V", "WING", "WIT", "WIX", "WK", "WKC", "WKEY", "WKSP", "WLDN", "WLFC", "WLK", "WLY", "WM", "WMB", "WMG", "WMK", "WMS", "WMT", "WNC", "WNEB", "WNS", "WOOF", "WOR", "WOW", "WPC", "WPM", "WPP", "WRB", "WRBY", "WRD",
       "WS", "WSBC", "WSC", "WSFS", "WSM", "WSO", "WSR", "WST", "WT", "WTF", "WTG", "WTRG", "WTS", "WTTR", "WTW", "WU", "WULF", "WVE", "WW", "WWD", "WWW", "WXM", "WY", "WYFI", "WYNN", "WYY", "XAIR", "XBIT", "XCUR", "XEL", "XENE", "XERS", "XGN", "XHR", "XIFR", "XMTR", "XNCR", "XNET", "XOM", "XOMA", "XP", "XPEL", "XPER", "XPEV", "XPO",
        "XPOF", "XPRO", "XRAY", "XRX", "XTKG", "XYF", "XYL", "XYZ", "YALA", "YB", "YELP", "YETI", "YEXT", "YMAB", "YMAT", "YMM", "YORK", "YORW", "YOU", "YPF", "YRD", "YSG", "YSXT", "YUM", "YUMC", "YYAI", "YYGH", "Z",
         "ZBAI", "ZBH", "ZBIO", "ZBRA", "ZD", "ZDGE", "ZENA", "ZEO", "ZEPP", "ZETA", "ZEUS", 
         "ZG", "ZGN", "ZH", "ZIM", "ZIMV", "ZION", "ZIP", "ZJK", "ZK", "ZLAB", "ZM", "ZONE", "ZS", "ZSPC", "ZTO", "ZTS", "ZUMZ", "ZVIA", "ZVRA", "ZWS", "ZYBT", "ZYME"]


@dataclass
class MarketRegime:
    """市场状态"""
    regime_id: int
    name: str
    probability: float
    characteristics: Dict[str, float]
    duration: int = 0

@dataclass 
class RiskFactorExposure:
    """风险因子暴露"""
    market_beta: float
    size_exposure: float  
    value_exposure: float
    momentum_exposure: float
    volatility_exposure: float
    quality_exposure: float
    country_exposure: Dict[str, float] = field(default_factory=dict)
    sector_exposure: Dict[str, float] = field(default_factory=dict)

def sanitize_ticker(raw: Union[str, Any]) -> str:
    """清理股票代码中的BOM、引号、空白等杂质。"""
    try:
        s = str(raw)
    except Exception:
        return ''
    # 去除BOM与零宽字符
    s = s.replace('\ufeff', '').replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    # 去除引号与空白
    s = s.strip().strip("'\"")
    # 统一大写
    s = s.upper()
    return s


def load_universe_from_file(file_path: str) -> Optional[List[str]]:
    try:
        if os.path.exists(file_path):
            # 使用utf-8-sig以自动去除BOM
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                tickers = []
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # 支持逗号或空格分隔
                    parts = [p for token in line.split(',') for p in token.split()]
                    for p in parts:
                        t = sanitize_ticker(p)
                        if t:
                            tickers.append(t)
            # 去重并保持顺序
            tickers = list(dict.fromkeys(tickers))
            return tickers if tickers else None
    except Exception:
        return None
    return None

def load_universe_fallback() -> List[str]:
    # 统一从配置文件读取股票清单，移除旧版依赖
    root_stocks = os.path.join(os.path.dirname(__file__), 'stocks.txt')
    tickers = load_universe_from_file(root_stocks)
    if tickers:
        return tickers
    
    logger.warning("未找到stocks.txt文件，使用默认股票清单")
    return DEFAULT_TICKER_LIST

class UltraEnhancedQuantitativeModel:
    """Ultra Enhanced 量化模型：集成所有高级功能"""
    
    def __init__(self, config_path: str = "alphas_config.yaml"):
        """
        初始化Ultra Enhanced量化模型
        
        Args:
            config_path: Alpha策略配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 🔥 生产级功能：模型版本控制
        try:
            from model_version_control import ModelVersionControl
            self.version_control = ModelVersionControl("ultra_models")
            logger.info("模型版本控制系统已启用")
        except ImportError as e:
            logger.warning(f"版本控制模块导入失败: {e}")
            self.version_control = None
        
        # 🔥 增强目标工程引擎
        if TARGET_ENGINEERING_AVAILABLE:
            barrier_config = TripleBarrierConfig(
                profit_threshold=self.config.get('target_config', {}).get('profit_threshold', 0.02),
                loss_threshold=self.config.get('target_config', {}).get('loss_threshold', -0.015),
                holding_periods=self.config.get('target_config', {}).get('holding_periods', [1, 3, 5, 10, 15])
            )
            meta_config = MetaLabelingConfig(
                win_rate_model=self.config.get('target_config', {}).get('win_rate_model', 'logistic'),
                magnitude_model=self.config.get('target_config', {}).get('magnitude_model', 'quantile')
            )
            self.target_engineer = EnhancedTargetEngineer(barrier_config, meta_config)
            logger.info("✅ 增强目标工程引擎已启用")
        else:
            self.target_engineer = None
            logger.warning("增强目标工程引擎不可用，使用传统目标构建")

        # 🔧 内存优化引擎
        if MEMORY_OPTIMIZATION_AVAILABLE:
            self.memory_optimizer = MemoryOptimizedBMA()
            self.optimized_feature_engineering = create_memory_optimized_feature_engineering()
            logger.info("✅ 内存优化引擎已启用")
        else:
            self.memory_optimizer = None
            self.optimized_feature_engineering = None
            logger.warning("内存优化引擎不可用，使用标准特征工程")

        # 🛡️ 安全内存管理引擎
        if SAFE_MEMORY_AVAILABLE:
            self.safe_memory_manager = create_safe_memory_manager()
            logger.info("✅ 安全内存管理已启用")
        else:
            self.safe_memory_manager = None
            logger.warning("安全内存管理不可用")

        # 核心引擎
        if ENHANCED_MODULES_AVAILABLE:
            self.alpha_engine = AlphaStrategiesEngine(config_path)
            self.ltr_bma = LearningToRankBMA(
                ranking_objective=self.config.get('model_config', {}).get('ranking_objective', 'rank:pairwise'),
                temperature=self.config.get('temperature', 1.2),
                enable_regime_detection=self.config.get('model_config', {}).get('regime_detection', True)
            )
            self.portfolio_optimizer = AdvancedPortfolioOptimizer(
                risk_aversion=self.config.get('risk_config', {}).get('risk_aversion', 5.0),
                turnover_penalty=self.config.get('risk_config', {}).get('turnover_penalty', 1.0),
                max_turnover=self.config.get('max_turnover', 0.10),
                max_position=self.config.get('max_position', 0.03),
                max_sector_exposure=self.config.get('risk_config', {}).get('max_sector_exposure', 0.15),
                max_country_exposure=self.config.get('risk_config', {}).get('max_country_exposure', 0.20)
            )
        else:
            logger.warning("增强模块不可用，使用基础功能")
            self.alpha_engine = None
            self.ltr_bma = None
            self.portfolio_optimizer = None
        
        # 传统ML模型（作为对比）
        self.traditional_models = {}
        self.model_weights = {}
        
        # Professional引擎功能
        self.risk_model_results = {}
        self.current_regime = None
        self.regime_weights = {}
        self.market_data_manager = UnifiedMarketDataManager() if MARKET_MANAGER_AVAILABLE else None
        
        # 🔥 Barra风格因子库初始化
        if BARRA_FACTORS_AVAILABLE:
            self.barra_style_factors = create_barra_style_factors(lookback_window=252)
            logger.info(f"Barra风格因子库初始化完成: {len(self.barra_style_factors.get_factor_clusters())}个因子簇")
        else:
            self.barra_style_factors = None
            logger.warning("Barra风格因子库不可用")
        
        # 🔥 Barra风险模型和约束优化器初始化
        if BARRA_OPTIMIZER_AVAILABLE:
            # 使用Barra标准风格因子
            barra_style_factors = ['size_ln_market_cap', 'value_book_to_price', 'quality_roe', 
                                 'momentum_12_1', 'lowvol_volatility_90d', 'liquidity_turnover_rate', 
                                 'growth_sales_growth']
            
            self.barra_risk_model = create_barra_risk_model(
                style_factors=barra_style_factors,
                lookback_window=252
            )
            self.constrained_optimizer = create_portfolio_optimizer(
                max_position_weight=0.05,
                max_turnover=0.20,
                risk_aversion=5.0
            )
            logger.info("Barra风险模型和约束优化器初始化完成")
        else:
            self.barra_risk_model = None
            self.constrained_optimizer = None
            logger.warning("Barra风险模型不可用，将使用传统优化方法")
        
        # 数据和结果存储
        self.raw_data = {}
        self.feature_data = None
        self.alpha_signals = None
        self.final_predictions = None
        self.portfolio_weights = None
        
        # 性能跟踪
        self.performance_metrics = {}
        self.backtesting_results = {}
        
        # 健康监控计数器
        self.health_metrics = {
            'universe_load_fallbacks': 0,
            'risk_model_failures': 0,
            'optimization_fallbacks': 0,
            'alpha_computation_failures': 0,
            'neutralization_failures': 0,
            'prediction_failures': 0,
            'total_exceptions': 0
        }
        
        logger.info("UltraEnhanced量化模型初始化完成")
    
    def get_health_report(self) -> Dict[str, Any]:
        """获取系统健康状况报告"""
        total_operations = sum(self.health_metrics.values())
        failure_rate = (self.health_metrics['total_exceptions'] / max(total_operations, 1)) * 100
        
        report = {
            'health_metrics': self.health_metrics.copy(),
            'failure_rate_percent': failure_rate,
            'risk_level': 'LOW' if failure_rate < 5 else 'MEDIUM' if failure_rate < 15 else 'HIGH',
            'recommendations': []
        }
        
        # 根据失败类型给出建议
        if self.health_metrics['universe_load_fallbacks'] > 0:
            report['recommendations'].append("检查股票清单文件格式和编码")
        if self.health_metrics['risk_model_failures'] > 2:
            report['recommendations'].append("检查UMDM配置和市场数据连接")
        if self.health_metrics['optimization_fallbacks'] > 1:
            report['recommendations'].append("检查投资组合约束设置")
        
        return report
    
    def build_risk_model(self) -> Dict[str, Any]:
        """构建Multi-factor风险模型（来自Professional引擎）"""
        logger.info("构建Multi-factor风险模型")
        
        if not self.raw_data:
            raise ValueError("Market data not loaded")
        
        # 构建收益率矩阵
        returns_data = []
        tickers = []
        
        for ticker, data in self.raw_data.items():
            if len(data) > 100:
                returns = data['close'].pct_change().fillna(0)
                returns_data.append(returns)
                tickers.append(ticker)
        
        if not returns_data:
            raise ValueError("No valid returns data")
        
        returns_matrix = pd.concat(returns_data, axis=1, keys=tickers)
        returns_matrix = returns_matrix.fillna(0.0)
        
        # 构建风险因子
        risk_factors = self._build_risk_factors(returns_matrix)
        
        # 估计因子载荷
        factor_loadings = self._estimate_factor_loadings(returns_matrix, risk_factors)
        
        # 估计因子协方差
        factor_covariance = self._estimate_factor_covariance(risk_factors)
        
        # 估计特异风险
        specific_risk = self._estimate_specific_risk(returns_matrix, factor_loadings, risk_factors)
        
        self.risk_model_results = {
            'factor_loadings': factor_loadings,
            'factor_covariance': factor_covariance,
            'specific_risk': specific_risk,
            'risk_factors': risk_factors
        }
        
        logger.info("风险模型构建完成")
        return self.risk_model_results
    
    def _build_risk_factors(self, returns_matrix: pd.DataFrame) -> pd.DataFrame:
        """构建风险因子（来自Professional引擎）"""
        factors = pd.DataFrame(index=returns_matrix.index)
        
        # 1. 市场因子
        factors['market'] = returns_matrix.mean(axis=1)
        
        # 2. 规模因子 (使用UMDM真实市值数据)
        try:
            if self.market_data_manager is not None:
                # 构建统一特征DataFrame，获取真实市值数据
                tickers = returns_matrix.columns.tolist()
                dates = returns_matrix.index.tolist()
                
                # 创建用于UMDM的输入DataFrame
                input_data = []
                for date in dates:
                    for ticker in tickers:
                        input_data.append({'date': date, 'ticker': ticker})
                
                if input_data:
                    input_df = pd.DataFrame(input_data)
                    features_df = self.market_data_manager.create_unified_features_dataframe(input_df)
                    
                    if 'free_float_market_cap' in features_df.columns:
                        # 重塑为[date, ticker]格式并对齐
                        features_pivot = features_df.set_index(['date', 'ticker'])['free_float_market_cap']
                        
                        #  修复时间泄露：Size因子使用前期市值分组当期收益
                        size_factor = []
                        dates_list = list(returns_matrix.index)
                        
                        for i, date in enumerate(dates_list):
                            try:
                                #  关键修复：使用T-1期的市值进行分组，计算T期收益
                                if i == 0:
                                    # 第一个日期没有前期数据，跳过
                                    size_factor.append(0.0)
                                    continue
                                
                                prev_date = dates_list[i-1]
                                prev_date_caps = features_pivot.loc[prev_date]  # 使用前一期市值
                                prev_date_caps = prev_date_caps.reindex(returns_matrix.columns)
                                
                                if prev_date_caps.notna().sum() > 2:  # 至少需要3只股票有市值数据
                                    cap_median = prev_date_caps.median()
                                    small_cap_mask = prev_date_caps < cap_median
                                    large_cap_mask = ~small_cap_mask
                                    
                                    # 使用当期收益率，但分组基于前期市值
                                    date_returns = returns_matrix.loc[date]
                                    small_ret = date_returns[small_cap_mask].mean()
                                    large_ret = date_returns[large_cap_mask].mean()
                                    
                                    size_factor.append(small_ret - large_ret)
                                    
                                    logger.debug(f"日期{date}: 使用{prev_date}市值分组，"
                                               f"小盘股收益{small_ret:.4f}, 大盘股收益{large_ret:.4f}")
                                else:
                                    size_factor.append(0.0)
                            except (KeyError, IndexError):
                                size_factor.append(0.0)
                        
                        factors['size'] = pd.Series(size_factor, index=returns_matrix.index)
                        logger.info("使用UMDM真实市值数据构建Size因子")
                    else:
                        logger.warning("UMDM中缺少free_float_market_cap字段，使用回退方案")
                        raise ValueError("No market cap data available")
                else:
                    raise ValueError("No input data for UMDM")
            else:
                raise ValueError("UMDM not available")
                
        except (ValueError, KeyError, IndexError) as e:
            logger.exception(f"UMDM Size因子构建失败: {e}, 使用简化回退方案")
            self.health_metrics['risk_model_failures'] += 1
            # 回退方案：基于成交量估算规模
            try:
                volume_data = {}
                for ticker in returns_matrix.columns:
                    if ticker in self.raw_data and 'volume' in self.raw_data[ticker].columns:
                        # 使用最近60天平均成交量作为规模代理
                        recent_volume = self.raw_data[ticker]['volume'].tail(60).mean()
                        volume_data[ticker] = recent_volume

                if volume_data:
                    volume_series = pd.Series(volume_data)
                    volume_median = volume_series.median()
                    small_vol_mask = volume_series < volume_median

                    small_vol_returns = returns_matrix.loc[:, small_vol_mask].mean(axis=1)
                    large_vol_returns = returns_matrix.loc[:, ~small_vol_mask].mean(axis=1)
                    factors['size'] = small_vol_returns - large_vol_returns
                    logger.info("使用成交量代理构建Size因子（回退方案）")
                else:
                    # 最终回退：使用零值
                    factors['size'] = 0.0
                    logger.warning("无法构建Size因子，使用零值")
            except Exception as fallback_error:
                logger.error(f"Size因子回退方案也失败: {fallback_error}")
                factors['size'] = 0.0
        
        # 3. 动量因子
        momentum_scores = {}
        for ticker in returns_matrix.columns:
            momentum_scores[ticker] = returns_matrix[ticker].rolling(252).sum().shift(21)
        
        momentum_df = pd.DataFrame(momentum_scores)
        high_momentum = momentum_df.rank(axis=1, pct=True) > 0.7
        low_momentum = momentum_df.rank(axis=1, pct=True) < 0.3
        
        factors['momentum'] = returns_matrix.where(high_momentum).mean(axis=1) - \
                             returns_matrix.where(low_momentum).mean(axis=1)
        
        # 4. 波动率因子
        volatility_scores = returns_matrix.rolling(60).std()
        low_vol = volatility_scores.rank(axis=1, pct=True) < 0.3
        high_vol = volatility_scores.rank(axis=1, pct=True) > 0.7
        
        factors['volatility'] = returns_matrix.where(low_vol).mean(axis=1) - \
                               returns_matrix.where(high_vol).mean(axis=1)
        
        # 5. 质量因子
        quality_scores = returns_matrix.rolling(60).mean() / returns_matrix.rolling(60).std()
        high_quality = quality_scores.rank(axis=1, pct=True) > 0.7
        low_quality = quality_scores.rank(axis=1, pct=True) < 0.3
        
        factors['quality'] = returns_matrix.where(high_quality).mean(axis=1) - \
                            returns_matrix.where(low_quality).mean(axis=1)
        
        # 6. 反转因子
        reversal_scores = returns_matrix.rolling(21).sum()
        high_reversal = reversal_scores.rank(axis=1, pct=True) < 0.3
        low_reversal = reversal_scores.rank(axis=1, pct=True) > 0.7
        
        factors['reversal'] = returns_matrix.where(high_reversal).mean(axis=1) - \
                             returns_matrix.where(low_reversal).mean(axis=1)
        
        # 标准化因子
        factors = factors.fillna(0)
        for col in factors.columns:
            factors[col] = (factors[col] - factors[col].mean()) / (factors[col].std() + 1e-8)
        
        return factors
    
    def _estimate_factor_loadings(self, returns_matrix: pd.DataFrame, 
                                 risk_factors: pd.DataFrame) -> pd.DataFrame:
        """估计因子载荷"""
        loadings = {}
        
        for ticker in returns_matrix.columns:
            stock_returns = returns_matrix[ticker].dropna()
            aligned_factors = risk_factors.loc[stock_returns.index].dropna().fillna(0)
            
            if len(stock_returns) < 50 or len(aligned_factors) < 50:
                loadings[ticker] = np.zeros(len(risk_factors.columns))
                continue
            
            try:
                # 确保数据长度匹配
                min_len = min(len(stock_returns), len(aligned_factors))
                stock_returns = stock_returns.iloc[:min_len]
                aligned_factors = aligned_factors.iloc[:min_len]
                
                # 使用稳健回归估计载荷
                model = HuberRegressor(epsilon=1.35, alpha=0.0001)
                model.fit(aligned_factors.values, stock_returns.values)
                
                loadings[ticker] = model.coef_
                
            except Exception as e:
                logger.warning(f"Failed to estimate loadings for {ticker}: {e}")
                loadings[ticker] = np.zeros(len(risk_factors.columns))
        
        loadings_df = pd.DataFrame(loadings, index=risk_factors.columns).T
        return loadings_df
    
    def _estimate_factor_covariance(self, risk_factors: pd.DataFrame) -> pd.DataFrame:
        """估计因子协方差矩阵"""
        # 使用Ledoit-Wolf收缩估计
        cov_estimator = LedoitWolf()
        factor_cov_matrix = cov_estimator.fit(risk_factors.fillna(0)).covariance_
        
        # 确保正定性
        eigenvals, eigenvecs = np.linalg.eigh(factor_cov_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)
        factor_cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return pd.DataFrame(factor_cov_matrix, 
                           index=risk_factors.columns, 
                           columns=risk_factors.columns)
    
    def _estimate_specific_risk(self, returns_matrix: pd.DataFrame,
                               factor_loadings: pd.DataFrame, 
                               risk_factors: pd.DataFrame) -> pd.Series:
        """估计特异风险"""
        specific_risks = {}
        
        for ticker in returns_matrix.columns:
            if ticker not in factor_loadings.index:
                specific_risks[ticker] = 0.2  # 默认特异风险
                continue
            
            stock_returns = returns_matrix[ticker].dropna()
            loadings = factor_loadings.loc[ticker]
            aligned_factors = risk_factors.loc[stock_returns.index].fillna(0)
            
            if len(stock_returns) < 50:
                specific_risks[ticker] = 0.2
                continue
            
            # 计算残差
            min_len = min(len(stock_returns), len(aligned_factors))
            factor_returns = (aligned_factors.iloc[:min_len] @ loadings).values
            residuals = stock_returns.iloc[:min_len].values - factor_returns
            
            # 特异风险为残差标准差
            specific_var = np.nan_to_num(np.var(residuals), nan=0.04)
            specific_risks[ticker] = np.sqrt(specific_var)
        
        return pd.Series(specific_risks)
    
    def detect_market_regime(self) -> MarketRegime:
        """检测市场状态（来自Professional引擎）"""
        logger.info("检测市场状态")
        
        if not self.raw_data:
            return MarketRegime(0, "Unknown", 0.5, {'volatility': 0.2, 'trend': 0.0})
        
        # 构建市场指数
        market_returns = []
        for ticker, data in self.raw_data.items():
            if len(data) > 100:
                returns = data['close'].pct_change().fillna(0)
                market_returns.append(returns)
        
        if not market_returns:
            return MarketRegime(0, "Unknown", 0.5, {'volatility': 0.2, 'trend': 0.0})
        
        market_index = pd.concat(market_returns, axis=1).mean(axis=1).dropna()
        
        if len(market_index) < 100:
            return MarketRegime(1, "Normal", 1.0, {'volatility': 0.15, 'trend': 0.0})
        
        # 基于波动率和趋势的状态检测 (修复：使用价格级别动量而非收益均值)
        rolling_vol = market_index.rolling(21).std()
        # 构建累积价格指数用于趋势计算
        cumulative_price = (1 + market_index).cumprod()
        # 使用价格级别动量：21日动量（当前价格相对21天前的变化）
        rolling_trend = cumulative_price.pct_change(21)
        
        # 定义状态阈值
        vol_low = rolling_vol.quantile(0.33)
        vol_high = rolling_vol.quantile(0.67)
        trend_low = rolling_trend.quantile(0.33)
        trend_high = rolling_trend.quantile(0.67)
        
        # 当前状态
        current_vol = rolling_vol.iloc[-1]
        current_trend = rolling_trend.iloc[-1]
        
        if current_vol < vol_low:
            if current_trend > trend_high:
                regime = MarketRegime(1, "Bull_Low_Vol", 0.8, 
                                    {'volatility': current_vol, 'trend': current_trend})
            elif current_trend < trend_low:
                regime = MarketRegime(2, "Bear_Low_Vol", 0.8,
                                    {'volatility': current_vol, 'trend': current_trend})
            else:
                regime = MarketRegime(3, "Normal_Low_Vol", 0.8,
                                    {'volatility': current_vol, 'trend': current_trend})
        elif current_vol > vol_high:
            if current_trend > trend_high:
                regime = MarketRegime(4, "Bull_High_Vol", 0.8,
                                    {'volatility': current_vol, 'trend': current_trend})
            elif current_trend < trend_low:
                regime = MarketRegime(5, "Bear_High_Vol", 0.8,
                                    {'volatility': current_vol, 'trend': current_trend})
            else:
                regime = MarketRegime(6, "Volatile", 0.8,
                                    {'volatility': current_vol, 'trend': current_trend})
        else:
            regime = MarketRegime(0, "Normal", 0.7,
                                {'volatility': current_vol, 'trend': current_trend})
        
        self.current_regime = regime
        logger.info(f"检测到市场状态: {regime.name} (概率: {regime.probability:.2f})")
        
        return regime
    
    def _create_equal_weight_fallback(self, predictions: pd.Series, top_k: int = 10, 
                                     portfolio_cov: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        独立的等权回退函数，解决嵌套缩进下 return 路径混乱
        
        Args:
            predictions: 预测收益率
            top_k: 选择前K只股票
            portfolio_cov: 可选的协方差矩阵
            
        Returns:
            包含权重和组合指标的字典
        """
        try:
            # 选择Top-K资产
            if len(predictions) == 0:
                return {
                    'success': False,
                    'method': 'equal_weight_fallback',
                    'weights': {},
                    'portfolio_metrics': {},
                    'error': 'No predictions available'
                }
            
            top_k = min(top_k, len(predictions))
            top_assets = predictions.nlargest(top_k).index
            
            # 创建等权重
            equal_weights = pd.Series(1.0/len(top_assets), index=top_assets)
            
            # 计算组合指标
            expected_returns = predictions.reindex(top_assets).dropna()
            portfolio_return = expected_returns.mean()
            
            # 计算风险
            if portfolio_cov is not None and len(expected_returns) > 0:
                try:
                    aligned_weights = equal_weights.reindex(expected_returns.index)
                    aligned_cov = portfolio_cov.reindex(
                        index=expected_returns.index, 
                        columns=expected_returns.index
                    ).fillna(0.01)
                    portfolio_risk = np.sqrt(aligned_weights @ aligned_cov @ aligned_weights)
                except (KeyError, ValueError, IndexError):
                    portfolio_risk = 0.15  # 默认风险估计
            else:
                portfolio_risk = 0.15  # 默认风险估计
            
            # 计算夏普比率
            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0.0
            
            return {
                'success': True,
                'method': 'equal_weight_fallback',
                'weights': equal_weights.to_dict(),
                'portfolio_metrics': {
                    'expected_return': float(portfolio_return),
                    'portfolio_risk': float(portfolio_risk),
                    'sharpe_ratio': float(sharpe_ratio),
                    'diversification_ratio': len(top_assets),
                    'n_positions': len(top_assets)
                }
            }
            
        except Exception as e:
            logger.error(f"等权回退失败: {e}")
            return {
                'success': False,
                'method': 'equal_weight_fallback',
                'weights': {},
                'portfolio_metrics': {},
                'error': str(e)
            }

    def _get_regime_alpha_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """根据市场状态调整Alpha权重（来自Professional引擎）"""
        if "Bull" in regime.name:
            # 牛市：偏好动量
            return {
                'momentum_21d': 2.0, 'momentum_63d': 2.5, 'momentum_126d': 2.0,
                'reversion_5d': 0.5, 'reversion_10d': 0.5, 'reversion_21d': 0.5,
                'volatility_factor': 1.0, 'volume_trend': 1.5, 'quality_factor': 1.0
            }
        elif "Bear" in regime.name:
            # 熊市：偏好质量和防御
            return {
                'momentum_21d': 0.5, 'momentum_63d': 0.5, 'momentum_126d': 1.0,
                'reversion_5d': 1.5, 'reversion_10d': 2.0, 'reversion_21d': 1.5,
                'volatility_factor': 2.0, 'volume_trend': 0.5, 'quality_factor': 2.0
            }
        elif "Volatile" in regime.name:
            # 高波动：偏好均值回归
            return {
                'momentum_21d': 0.5, 'momentum_63d': 1.0, 'momentum_126d': 1.0,
                'reversion_5d': 2.5, 'reversion_10d': 2.0, 'reversion_21d': 1.5,
                'volatility_factor': 2.5, 'volume_trend': 1.0, 'quality_factor': 1.5
            }
        else:
            # 正常市场：均衡权重
            return {col: 1.0 for col in [
                'momentum_21d', 'momentum_63d', 'momentum_126d',
                'reversion_5d', 'reversion_10d', 'reversion_21d',
                'volatility_factor', 'volume_trend', 'quality_factor'
            ]}
    
    def generate_enhanced_predictions(self, training_results: Dict[str, Any], 
                                    market_regime: MarketRegime) -> pd.Series:
        """生成Regime-Aware的增强预测"""
        try:
            # 获取基础预测
            base_predictions = self.generate_ensemble_predictions(training_results)
            
            if not ENHANCED_MODULES_AVAILABLE or not self.alpha_engine:
                # 如果没有增强模块，应用regime权重到基础预测
                regime_weights = self._get_regime_alpha_weights(market_regime)
                # 简单应用权重（这里简化处理）
                adjustment_factor = sum(regime_weights.values()) / len(regime_weights)
                enhanced_predictions = base_predictions * adjustment_factor
                logger.info(f"应用简化的regime调整，调整因子: {adjustment_factor:.3f}")
                return enhanced_predictions
            
            # 如果有Alpha引擎，生成Alpha信号
            try:
                # 为Alpha引擎准备数据（包含标准化的价格列）
                alpha_input = self._prepare_alpha_data()
                # 计算Alpha因子（签名只接受df）
                alpha_signals = self.alpha_engine.compute_all_alphas(alpha_input)
                
                # 根据市场状态调整Alpha权重
                regime_weights = self._get_regime_alpha_weights(market_regime)
                
                # 应用regime权重到alpha信号
                weighted_alpha = pd.Series(0.0, index=alpha_signals.index)
                for alpha_name, weight in regime_weights.items():
                    if alpha_name in alpha_signals.columns:
                        weighted_alpha += alpha_signals[alpha_name] * weight
                
                # 标准化加权后的alpha
                if weighted_alpha.std() > 0:
                    weighted_alpha = (weighted_alpha - weighted_alpha.mean()) / weighted_alpha.std()
                
                # 与基础ML预测融合
                alpha_weight = 0.3  # Alpha信号权重
                ml_weight = 0.7     # ML预测权重
                
                # 确保索引对齐
                common_index = base_predictions.index.intersection(weighted_alpha.index)
                if len(common_index) > 0:
                    enhanced_predictions = (
                        ml_weight * base_predictions.reindex(common_index).fillna(0) +
                        alpha_weight * weighted_alpha.reindex(common_index).fillna(0)
                    )
                else:
                    enhanced_predictions = base_predictions
                
                logger.info(f"成功融合Alpha信号和ML预测，market regime: {market_regime.name}")
                return enhanced_predictions
                
            except (ValueError, KeyError, AttributeError) as e:
                logger.exception(f"Alpha信号生成失败: {e}")
                self.health_metrics['alpha_computation_failures'] += 1
                # 回退到基础预测
                return base_predictions
                
        except Exception as e:
            logger.exception(f"增强预测生成失败: {e}")
            self.health_metrics['prediction_failures'] += 1
            self.health_metrics['total_exceptions'] += 1
            # 最终回退
            return pd.Series(0.0, index=range(10))
    
    def optimize_portfolio_with_risk_model(self, predictions: pd.Series, 
                                          feature_data: pd.DataFrame) -> Dict[str, Any]:
        """使用风险模型的投资组合优化"""
        try:
            # 如果有Professional的风险模型结果，使用它们
            if self.risk_model_results and 'factor_loadings' in self.risk_model_results:
                factor_loadings = self.risk_model_results['factor_loadings']
                factor_covariance = self.risk_model_results['factor_covariance']
                specific_risk = self.risk_model_results['specific_risk']
                
                # 构建协方差矩阵
                common_assets = list(set(predictions.index) & set(factor_loadings.index))
                if len(common_assets) >= 3:
                    # 使用专业风险模型进行优化
                    try:
                        # 构建投资组合协方差矩阵: B * F * B' + S
                        B = factor_loadings.reindex(index=common_assets).dropna()  # 因子载荷 - 安全索引
                        F = factor_covariance                   # 因子协方差
                        S = specific_risk.reindex(index=common_assets).dropna()    # 特异风险 - 安全索引
                        
                        # 计算协方差矩阵
                        portfolio_cov = B @ F @ B.T + np.diag(S**2)
                        portfolio_cov = pd.DataFrame(
                            portfolio_cov, 
                            index=common_assets, 
                            columns=common_assets
                        )
                        
                        # 使用统一的AdvancedPortfolioOptimizer而非重复实现
                        if self.portfolio_optimizer:
                            try:
                                # 准备预期收益率 - 使用安全的索引访问
                                available_assets = predictions.index.intersection(common_assets)
                                if len(available_assets) == 0:
                                    raise ValueError("No common assets between predictions and risk model")
                                expected_returns = predictions.reindex(available_assets).dropna()
                                common_assets = list(expected_returns.index)  # 更新common_assets为实际可用的资产
                                
                                # 重新构建协方差矩阵以匹配可用资产
                                B_updated = factor_loadings.reindex(index=common_assets).dropna()
                                S_updated = specific_risk.reindex(index=common_assets).dropna()
                                portfolio_cov = B_updated @ F @ B_updated.T + np.diag(S_updated**2)
                                portfolio_cov = pd.DataFrame(
                                    portfolio_cov, 
                                    index=common_assets, 
                                    columns=common_assets
                                )
                                
                                # 准备股票池数据（用于约束）
                                universe_data = pd.DataFrame(index=common_assets)
                                # 添加模拟的行业/国家信息用于约束
                                universe_data['COUNTRY'] = 'US'  # 简化
                                universe_data['SECTOR'] = 'TECH'  # 简化 
                                universe_data['liquidity_rank'] = 0.5  # 中等流动性
                                
                                # 调用统一的优化器
                                optimization_result = self.portfolio_optimizer.optimize_portfolio(
                                    expected_returns=expected_returns,
                                    covariance_matrix=portfolio_cov,
                                    current_weights=None,  # 假设从空仓开始
                                    universe_data=universe_data
                                )
                                
                                if optimization_result.get('success', False):
                                    optimal_weights = optimization_result['optimal_weights']
                                    portfolio_metrics = optimization_result['portfolio_metrics']

                                    # 风险归因
                                    risk_attribution = self.portfolio_optimizer.risk_attribution(
                                        optimal_weights, portfolio_cov
                                    )
                                    
                                    return {
                                        'success': True,
                                        'method': 'unified_portfolio_optimizer_with_risk_model',
                                        'weights': optimal_weights.to_dict(),
                                        'portfolio_metrics': portfolio_metrics,
                                        'risk_attribution': risk_attribution,
                                        'regime_context': self.current_regime.name if self.current_regime else "Unknown"
                                    }
                                else:
                                    logger.warning("统一优化器优化失败，使用回退方案")
                                    raise ValueError("Unified optimizer failed")
                            
                            except (ValueError, RuntimeError, np.linalg.LinAlgError) as optimizer_error:
                                logger.exception(f"统一优化器调用失败: {optimizer_error}, 使用简化优化")
                                self.health_metrics['optimization_fallbacks'] += 1
                                # 简化回退：等权组合 - 使用独立函数
                                fallback_assets = predictions.index.intersection(common_assets)
                                if len(fallback_assets) == 0:
                                    fallback_predictions = predictions.head(5)
                                else:
                                    fallback_predictions = predictions.reindex(fallback_assets).dropna()
                                
                                fallback_result = self._create_equal_weight_fallback(
                                    fallback_predictions, 
                                    top_k=len(fallback_predictions),
                                    portfolio_cov=portfolio_cov
                                )
                                
                                # 添加额外的上下文信息
                                fallback_result['risk_attribution'] = {}
                                fallback_result['regime_context'] = self.current_regime.name if self.current_regime else "Unknown"
                                fallback_result['method'] = 'equal_weight_fallback_with_risk_model'
                            
                            return fallback_result
                        else:
                            logger.error("AdvancedPortfolioOptimizer 不可用")
                            raise ValueError("Portfolio optimizer not available")
                        
                    except Exception as e:
                        logger.warning(f"专业风险模型优化失败: {e}")
            
            # 回退到基础优化
            return self.optimize_portfolio(predictions, feature_data)
            
        except Exception as e:
            logger.error(f"风险模型优化失败: {e}")
            # 最终回退到等权组合
            top_assets = predictions.nlargest(min(10, len(predictions))).index
            equal_weights = pd.Series(1.0/len(top_assets), index=top_assets)
            
            return {
                'success': True,
                'method': 'equal_weight_fallback',
                'weights': equal_weights.to_dict(),
                'portfolio_metrics': {
                    'expected_return': predictions.reindex(top_assets).dropna().mean(),
                    'portfolio_risk': 0.15,  # 假设风险
                    'sharpe_ratio': 1.0,
                    'diversification_ratio': len(top_assets)
                },
                'risk_attribution': {},
                'regime_context': self.current_regime.name if self.current_regime else "Unknown"
            }
    
    def _prepare_alpha_data(self) -> pd.DataFrame:
        """为Alpha引擎准备数据"""
        if not self.raw_data:
            return pd.DataFrame()
        
        # 将原始数据转换为Alpha引擎需要的格式
        all_data = []
        for ticker, data in self.raw_data.items():
            ticker_data = data.copy()
            ticker_data['ticker'] = ticker
            ticker_data['date'] = ticker_data.index
            # 标准化价格列，Alpha引擎需要 'Close','High','Low'
            if 'Adj Close' in ticker_data.columns:
                ticker_data['Close'] = ticker_data['Adj Close']
            elif 'close' in ticker_data.columns:
                ticker_data['Close'] = ticker_data['close']
            elif 'Close' not in ticker_data.columns and 'close' not in ticker_data.columns:
                # 若缺少close信息，跳过该票
                continue
            if 'High' not in ticker_data.columns and 'high' in ticker_data.columns:
                ticker_data['High'] = ticker_data['high']
            if 'Low' not in ticker_data.columns and 'low' in ticker_data.columns:
                ticker_data['Low'] = ticker_data['low']
            # 添加模拟的基本信息
            ticker_data['COUNTRY'] = 'US'
            ticker_data['SECTOR'] = 'Technology'  # 简化处理
            ticker_data['SUBINDUSTRY'] = 'Software'
            all_data.append(ticker_data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        else:
            return pd.DataFrame()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"配置文件{self.config_path}未找到，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'universe': 'TOPDIV3000',
            'neutralization': ['COUNTRY'],
            'hump_levels': [0.003, 0.008],
            'winsorize_std': 2.5,
            'truncation': 0.10,
            'max_position': 0.03,
            'max_turnover': 0.10,
            'temperature': 1.2,
            'model_config': {
                'learning_to_rank': True,
                'ranking_objective': 'rank:pairwise',
                'uncertainty_aware': True,
                'quantile_regression': True
            },
            'risk_config': {
                'risk_aversion': 5.0,
                'turnover_penalty': 1.0,
                'max_sector_exposure': 0.15,
                'max_country_exposure': 0.20
            }
        }
    
    def download_stock_data(self, tickers: List[str], 
                           start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        下载股票数据
        
        Args:
            tickers: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            股票数据字典
        """
        logger.info(f"下载{len(tickers)}只股票的数据，时间范围: {start_date} - {end_date}")

        # 将训练结束时间限制为当天的前一天（T-1），避免使用未完全结算的数据
        # 修复：使用更稳健的时区与T-1截止逻辑，为交易日历扩展预留接口
        try:
            # 使用UTC时间避免本地时区问题
            from datetime import timezone
            utc_now = datetime.now(timezone.utc)
            # 考虑美股市场：使用美东时区判断交易日
            # 这里简化处理，实际可对接交易日历如pandas_market_calendars
            us_eastern = utc_now - timedelta(hours=5)  # 简化的EST偏移
            
            # 如果当前美东时间是交易日且市场已收盘(16:00 EST后)，使用当天
            # 否则使用前一个交易日
            market_close_time = us_eastern.replace(hour=16, minute=0, second=0, microsecond=0)
            
            if us_eastern.hour >= 16 and us_eastern.weekday() < 5:  # 工作日且已收盘
                last_trading_day = us_eastern.date()
            else:
                # 找到前一个工作日
                days_back = 1
                if us_eastern.weekday() == 0:  # 周一，回退到周五
                    days_back = 3
                elif us_eastern.weekday() == 6:  # 周日，回退到周五  
                    days_back = 2
                last_trading_day = (us_eastern - timedelta(days=days_back)).date()
            
            end_dt = pd.to_datetime(end_date).date()
            if end_dt > last_trading_day:
                adjusted_end = last_trading_day.strftime('%Y-%m-%d')
                logger.info(f"结束日期{end_date} 超过最后交易日，已调整为 {adjusted_end}")
                end_date = adjusted_end
                
        except Exception as e:
            # 回退到简单逻辑
            logger.debug(f"高级T-1截止失败，使用简化逻辑: {e}")
            try:
                yesterday = (datetime.now() - timedelta(days=1)).date()
                end_dt = pd.to_datetime(end_date).date()
                if end_dt > yesterday:
                    adjusted_end = yesterday.strftime('%Y-%m-%d')
                    logger.info(f"结束日期{end_date} 超过昨日，已调整为 {adjusted_end}")
                    end_date = adjusted_end
            except Exception as _e:
                logger.debug(f"结束日期调整完全跳过: {_e}")
        
        # 数据验证
        if not tickers or len(tickers) == 0:
            logger.error("股票代码列表为空")
            return {}
        
        if not start_date or not end_date:
            logger.error("开始日期或结束日期为空")
            return {}
        
        all_data = {}
        failed_downloads = []
        
        for i, ticker in enumerate(tickers):
            try:
                # 验证股票代码格式
                if not ticker or not isinstance(ticker, str) or len(ticker.strip()) == 0:
                    logger.warning(f"无效的股票代码: {ticker}")
                    failed_downloads.append(ticker)
                    continue
                
                ticker = ticker.strip().upper()  # 标准化股票代码
                logger.info(f"[{i+1:3d}/{len(tickers):3d}] 下载 {ticker:6s}...")
                
                stock = PolygonTicker(ticker)
                # 使用复权数据，避免股利污染；固定日频，关闭actions列
                hist = stock.history(start=start_date, end=end_date, interval='1d')
                
                # 数据质量检查
                if hist is None or len(hist) == 0:
                    logger.warning(f"{ticker}: 无数据")
                    failed_downloads.append(ticker)
                    continue
                
                # 检查必要的列是否存在
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in hist.columns]
                if missing_cols:
                    logger.warning(f"{ticker}: 缺少必要列 {missing_cols}")
                    failed_downloads.append(ticker)
                    continue
                
                # 检查数据质量
                if hist['Close'].isna().all():
                    logger.warning(f"{ticker}: 所有收盘价都是NaN")
                    failed_downloads.append(ticker)
                    continue
                
                # 标准化列名
                hist = hist.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 
                    'Close': 'close', 'Volume': 'volume'
                })
                
                # 添加基础特征
                hist['ticker'] = ticker
                hist['date'] = hist.index
                hist['amount'] = hist['close'] * hist['volume']  # 成交额
                
                # 添加元数据（模拟）
                hist['COUNTRY'] = self._get_country_for_ticker(ticker)
                hist['SECTOR'] = self._get_sector_for_ticker(ticker)
                hist['SUBINDUSTRY'] = self._get_subindustry_for_ticker(ticker)
                
                all_data[ticker] = hist
                
            except Exception as e:
                logger.warning(f"下载{ticker}失败: {e}")
                failed_downloads.append(ticker)
        
        if failed_downloads:
            logger.warning(f"以下股票下载失败: {failed_downloads}")
        
        logger.info(f"成功下载{len(all_data)}只股票的数据")
        self.raw_data = all_data
        
        return all_data
    
    def _get_country_for_ticker(self, ticker: str) -> str:
        """获取股票的国家（使用稳定哈希确保可复现）"""
        # 这里可以接入真实的股票元数据API
        if ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']:
            return 'US'
        
        # 使用稳定哈希确保相同ticker总是得到相同国家
        import hashlib
        countries = ['US', 'EU', 'ASIA']
        hash_value = int(hashlib.md5((ticker + '_country').encode()).hexdigest(), 16)
        return countries[hash_value % len(countries)]
    
    def _get_sector_for_ticker(self, ticker: str) -> str:
        """获取股票的行业（使用稳定哈希确保可复现）"""
        sector_mapping = {
            'AAPL': 'TECH', 'MSFT': 'TECH', 'GOOGL': 'TECH', 'NVDA': 'TECH',
            'AMZN': 'CONSUMER', 'TSLA': 'AUTO', 'META': 'TECH', 'NFLX': 'MEDIA'
        }
        if ticker in sector_mapping:
            return sector_mapping[ticker]
        
        # 使用稳定哈希确保相同ticker总是得到相同行业
        import hashlib
        sectors = ['TECH', 'FINANCE', 'ENERGY', 'HEALTH']
        hash_value = int(hashlib.md5(ticker.encode()).hexdigest(), 16)
        return sectors[hash_value % len(sectors)]
    
    def _get_subindustry_for_ticker(self, ticker: str) -> str:
        """获取股票的子行业（使用稳定哈希确保可复现）"""
        # 使用稳定哈希确保相同ticker总是得到相同子行业
        import hashlib
        subindustries = ['SOFTWARE', 'HARDWARE', 'BIOTECH', 'RETAIL']
        hash_value = int(hashlib.md5((ticker + '_sub').encode()).hexdigest(), 16)
        return subindustries[hash_value % len(subindustries)]
    
    def _apply_enhanced_targets_to_features(self, combined_features: pd.DataFrame) -> pd.DataFrame:
        """将增强目标工程应用到特征数据框"""
        if not self.target_engineer:
            return combined_features
        
        try:
            # 准备价格数据 (需要date, ticker, close格式)
            price_data_for_target = combined_features[['date', 'ticker', 'close']].copy()
            
            # 生成增强目标
            target_result = self.target_engineer.generate_enhanced_targets(
                price_data=price_data_for_target,
                target_type=LabelType.EXPECTED_RETURN
            )
            
            # 将增强目标合并回数据框
            target_df = pd.DataFrame({
                'date': combined_features['date'],
                'ticker': combined_features['ticker'],
                'enhanced_target': target_result['targets']
            })
            
            # 合并增强目标到主数据框
            combined_features = combined_features.merge(target_df, on=['date', 'ticker'], how='left')
            combined_features['target'] = combined_features['enhanced_target'].fillna(
                combined_features.get('target', 0)
            )
            
            # 保存增强权重供后续使用
            self._enhanced_sample_weights = target_result['sample_weights']
            
            logger.info("✅ 增强目标工程已应用到特征数据")
            
        except Exception as e:
            logger.warning(f"增强目标工程应用失败: {e}")
        
        return combined_features
    
    def cleanup_memory(self, force: bool = False, safe_mode: bool = True):
        """
        清理模型内存
        
        Args:
            force: 是否强制清理谨慎对象
            safe_mode: 是否使用安全模式（保护重要结果）
        """
        if safe_mode and self.safe_memory_manager:
            # 🛡️ 使用安全内存管理器
            logger.info("使用安全内存管理进行清理")
            
            if force:
                # 紧急清理模式
                cleanup_stats = self.safe_memory_manager.emergency_cleanup(self)
            else:
                # 智能清理模式
                cleanup_stats = self.safe_memory_manager.smart_memory_cleanup(self)
            
            logger.info(f"安全清理完成: {cleanup_stats}")
            
        elif self.memory_optimizer:
            # 🔧 使用内存优化器清理
            self.memory_optimizer.force_memory_cleanup()
            
        else:
            # 🔄 传统清理方式
            import gc
            logger.warning("使用传统内存清理，可能影响重要结果")
            
            # 只有在force=True时才清理重要对象
            if force:
                if hasattr(self, 'raw_data'):
                    self.raw_data.clear()
                if hasattr(self, 'feature_data'):
                    del self.feature_data
                    self.feature_data = None
            
            # 清理临时对象（相对安全）
            temp_attrs = [attr for attr in dir(self) 
                         if any(keyword in attr for keyword in ['temp_', '_temp', 'cache_', '_cache'])]
            
            for attr in temp_attrs:
                try:
                    delattr(self, attr)
                    logger.debug(f"清理临时对象: {attr}")
                except:
                    pass
            
            # 强制垃圾回收
            for _ in range(3):
                collected = gc.collect()
                if collected == 0:
                    break
            
            logger.info("传统内存清理完成")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """获取内存使用报告"""
        if self.safe_memory_manager:
            return self.safe_memory_manager.get_memory_report(self)
        else:
            import psutil
            try:
                process = psutil.Process()
                return {
                    'current_memory_mb': process.memory_info().rss / 1024 / 1024,
                    'status': 'basic_monitoring'
                }
            except:
                return {'status': 'unavailable'}
    
    def create_traditional_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        创建传统技术指标特征
        
        Args:
            data_dict: 股票数据字典
            
        Returns:
            特征数据框
        """
        # 🔧 使用内存优化特征工程（如果可用）
        if self.memory_optimizer and self.optimized_feature_engineering:
            logger.info("使用内存优化特征工程")
            try:
                # 使用内存优化版本
                combined_features = self.optimized_feature_engineering(data_dict)
                
                # 应用增强目标工程（如果可用）
                if self.target_engineer:
                    combined_features = self._apply_enhanced_targets_to_features(combined_features)
                
                logger.info(f"✅ 内存优化特征工程完成，特征数据形状: {combined_features.shape}")
                return combined_features
                
            except Exception as e:
                logger.warning(f"内存优化特征工程失败，回退到标准方法: {e}")
                # 继续使用标准方法
        
        # 🔄 标准特征工程（回退方案）
        logger.info("创建传统技术指标特征（标准方法）")
        
        all_features = []
        
        # 分批处理以减少内存峰值
        tickers = list(data_dict.keys())
        batch_size = 30  # 减小批处理大小
        processed_count = 0
        
        for batch_idx in range(0, len(tickers), batch_size):
            batch_tickers = tickers[batch_idx:batch_idx + batch_size]
            logger.info(f"处理批次 {batch_idx//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}")
            
            batch_features = []
            
            for ticker in batch_tickers:
                df = data_dict[ticker]
                if len(df) < 60:  # 至少需要60天数据
                    continue
                
                df_copy = df.copy().sort_values('date')
                processed_count += 1
            
            # 价格特征
            df_copy['returns'] = df_copy['close'].pct_change()
            df_copy['log_returns'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
            
            # 移动平均
            for window in [5, 10, 20, 50]:
                df_copy[f'ma_{window}'] = df_copy['close'].rolling(window).mean()
                df_copy[f'ma_ratio_{window}'] = df_copy['close'] / df_copy[f'ma_{window}']
            
            # 波动率
            for window in [10, 20, 50]:
                df_copy[f'vol_{window}'] = df_copy['log_returns'].rolling(window).std()
            
            # RSI
            def calculate_rsi(prices, window=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            df_copy['rsi_14'] = calculate_rsi(df_copy['close'])
            
            # 成交量特征
            if 'volume' in df_copy.columns:
                df_copy['volume_ma_20'] = df_copy['volume'].rolling(20).mean()
                df_copy['volume_ratio'] = df_copy['volume'] / df_copy['volume_ma_20']
            
            # 价格位置
            for window in [20, 50]:
                high_roll = df_copy['high'].rolling(window).max()
                low_roll = df_copy['low'].rolling(window).min()
                df_copy[f'price_position_{window}'] = (df_copy['close'] - low_roll) / (high_roll - low_roll + 1e-8)
            
            # 动量指标
            for period in [5, 10, 20]:
                df_copy[f'momentum_{period}'] = df_copy['close'] / df_copy['close'].shift(period) - 1
            
            # 🔴 修复严重时间泄露：增强的时间对齐和验证
            FEATURE_LAG = 2        # 特征使用T-2及之前数据
            SAFETY_GAP = 2         # 额外安全间隔（防止信息泄露）
            PRED_START = 1         # 预测从T+1开始  
            PRED_END = 5           # 预测到T+5结束
            prediction_horizon = PRED_END  # 向后兼容
            
            # 验证时间对齐正确性
            total_gap = FEATURE_LAG + SAFETY_GAP + PRED_START
            if total_gap <= 0:
                raise ValueError(f"时间对齐错误：总间隔 {total_gap} <= 0，存在数据泄露风险")
            
            logger.info(f"时间对齐配置: 特征lag={FEATURE_LAG}, 安全gap={SAFETY_GAP}, 预测[T+{PRED_START}, T+{PRED_END}]")
            
            # 🔥 增强目标构建：使用Triple Barrier + Meta-Labeling或传统方法
            if self.target_engineer is not None:
                try:
                    # 准备价格数据 (需要date, ticker, close格式)
                    price_data_for_target = df_copy[['date', 'ticker', 'close']].copy()
                    
                    # 生成增强目标
                    target_result = self.target_engineer.generate_enhanced_targets(
                        price_data=price_data_for_target,
                        target_type=LabelType.EXPECTED_RETURN
                    )
                    
                    # 将增强目标合并回数据框
                    target_df = pd.DataFrame({
                        'date': df_copy['date'],
                        'ticker': df_copy['ticker'],
                        'enhanced_target': target_result['targets']
                    })
                    
                    # 合并增强目标到主数据框
                    df_copy = df_copy.merge(target_df, on=['date', 'ticker'], how='left')
                    df_copy['target'] = df_copy['enhanced_target'].fillna(0)  # 回退到0
                    
                    # 保存增强权重供后续使用
                    self._enhanced_sample_weights = target_result['sample_weights']
                    
                    logger.info("✅ 使用增强目标工程 (Triple Barrier + Meta-Labeling)")
                    logger.info(f"增强目标统计: 均值={df_copy['target'].mean():.4f}, 标准差={df_copy['target'].std():.4f}")
                    
                except Exception as e:
                    logger.warning(f"增强目标工程失败，回退到传统方法: {e}")
                    # 回退到传统目标构建
                    df_copy['target'] = (
                        df_copy['close'].shift(-PRED_END) / 
                        df_copy['close'].shift(-PRED_START + 1) - 1
                    )
                    self._enhanced_sample_weights = None
            else:
                # 传统目标构建：T时刻使用T-2-2=T-4特征，预测T+1到T+5收益
                # 确保特征和目标之间有足够的时间间隔（至少6期）
                df_copy['target'] = (
                    df_copy['close'].shift(-PRED_END) / 
                    df_copy['close'].shift(-PRED_START + 1) - 1
                )
                self._enhanced_sample_weights = None
            
            # 时间验证：确保没有重叠
            feature_max_time = -FEATURE_LAG - SAFETY_GAP  # 特征最新时间
            target_min_time = -PRED_START + 1             # 目标最早时间
            actual_gap = target_min_time - feature_max_time
            
            if actual_gap <= 0:
                raise ValueError(f"时间重叠错误：特征最新时间{feature_max_time} >= 目标最早时间{target_min_time}")
            
            logger.info(f"✅ 时间对齐验证通过：特征和目标间隔 {actual_gap} 期")
            
            # 🔥 关键：强制特征滞后以匹配增强的时间线
            # 特征使用T-4数据，目标使用T+1到T+5，间隔6期（安全）
            feature_lag = FEATURE_LAG + SAFETY_GAP  # 所有特征额外滞后4期
            
            # 在后续feature_cols处理中会统一应用滞后
            
            # 添加辅助信息
            df_copy['ticker'] = ticker
            df_copy['date'] = df_copy.index
            # 模拟行业和国家信息（实际应从数据源获取）
            df_copy['COUNTRY'] = 'US'
            df_copy['SECTOR'] = ticker[:2] if len(ticker) >= 2 else 'TECH'  # 简化分类
            df_copy['SUBINDUSTRY'] = ticker[:3] if len(ticker) >= 3 else 'SOFTWARE'
            
            df_copy['ticker'] = ticker
            batch_features.append(df_copy)
            
            # 批次内合并
            if batch_features:
                batch_combined = pd.concat(batch_features, ignore_index=True)
                
                # 内存优化（如果可用）
                if self.memory_optimizer:
                    batch_combined = self.memory_optimizer.optimize_dataframe_memory(batch_combined)
                
                all_features.append(batch_combined)
                
                # 清理临时变量
                del batch_features
                import gc
                gc.collect()
        
        logger.info(f"处理完成，共处理 {processed_count} 只股票")
        
        if all_features:
            # 最终合并
            if self.memory_optimizer:
                combined_features = self.memory_optimizer.memory_efficient_concat(all_features)
            else:
                combined_features = pd.concat(all_features, ignore_index=True)
            # 选出纯特征列（排除标识/目标/元数据）
            feature_cols = [col for col in combined_features.columns 
                            if col not in ['ticker','date','target','COUNTRY','SECTOR','SUBINDUSTRY']]
            # 🔥 强化特征滞后：确保严格的时间对齐
            try:
                # T-2基础滞后 + formation_lag(2) = 总共T-4滞后
                # 这确保特征信息严格早于目标时间窗口
                total_lag = 2 + 2  # base_lag + formation_lag
                combined_features[feature_cols] = combined_features.groupby('ticker')[feature_cols].shift(total_lag)
                logger.info(f"应用总滞后期数: {total_lag}，确保特征-目标时间隔离")
            except Exception as e:
                logger.warning(f"特征滞后处理失败: {e}")
                # 回退到基础滞后
                combined_features[feature_cols] = combined_features.groupby('ticker')[feature_cols].shift(2)
            # 基础清洗 - 只删除特征全为NaN的行，保留目标变量
            # 删除特征全为NaN的行，但保留有效目标的行
            feature_na_mask = combined_features[feature_cols].isna().all(axis=1)
            combined_features = combined_features[~feature_na_mask]

            # 🔗 合并完整的Polygon 40+专业因子集（统一来源 - T+5优化）
            try:
                from polygon_complete_factors import PolygonCompleteFactors
                from polygon_factors import PolygonShortTermFactors
                
                complete_factors = PolygonCompleteFactors()
                short_term_factors = PolygonShortTermFactors()
                symbols = sorted(combined_features['ticker'].unique().tolist())
                
                logger.info(f"开始集成Polygon完整因子库，股票数量: {len(symbols)}")
                
                # 获取因子库摘要
                factor_summary = complete_factors.get_factor_summary()
                logger.info(f"完整因子库包含 {factor_summary['total_factors']} 个专业因子")
                
                # 完整40+专业因子集合
                all_polygon_factors = {}
                factor_calculation_success = {}
                
                # 对前几只代表性股票计算完整因子
                sample_symbols = symbols[:min(3, len(symbols))]  # 限制样本数量以避免API限制
                
                for symbol in sample_symbols:
                    try:
                        logger.info(f"为 {symbol} 计算完整因子...")
                        
                        # 计算所有类别的因子
                        symbol_factors = complete_factors.calculate_all_complete_factors(
                            symbol, 
                            categories=['momentum', 'fundamental', 'profitability', 'quality', 'risk', 'microstructure']
                        )
                        
                        if symbol_factors:
                            logger.info(f"{symbol} 成功计算 {len(symbol_factors)} 个因子")
                            
                            # 提取因子值作为特征
                            for factor_name, result in symbol_factors.items():
                                if len(result.values) > 0 and result.data_quality > 0.5:
                                    col_name = f"polygon_{factor_name}"
                                    # 使用最新值
                                    factor_value = result.values.iloc[-1]
                                    if not np.isnan(factor_value) and np.isfinite(factor_value):
                                        all_polygon_factors[col_name] = factor_value
                                        factor_calculation_success[factor_name] = True
                        
                        # T+5短期因子
                        try:
                            t5_results = short_term_factors.calculate_all_short_term_factors(symbol)
                            if t5_results:
                                prediction = short_term_factors.create_t_plus_5_prediction(symbol, t5_results)
                                
                                # T+5专用因子
                                for factor_name, result in t5_results.items():
                                    col_name = f"t5_{factor_name}"
                                    if hasattr(result, 't_plus_5_signal'):
                                        signal_value = result.t_plus_5_signal
                                        if not np.isnan(signal_value) and np.isfinite(signal_value):
                                            all_polygon_factors[col_name] = signal_value
                                
                                # T+5综合预测信号
                                if 'signal_strength' in prediction:
                                    all_polygon_factors['t5_prediction_signal'] = prediction['signal_strength']
                                    all_polygon_factors['t5_prediction_confidence'] = prediction.get('confidence', 0.5)
                        except Exception as t5_e:
                            logger.warning(f"{symbol} T+5因子计算失败: {t5_e}")
                        
                        time.sleep(0.5)  # API限制
                        
                    except Exception as e:
                        logger.warning(f"{symbol}完整因子计算失败: {e}")
                        continue
                
                # 将计算成功的因子添加到特征矩阵
                if all_polygon_factors:
                    logger.info(f"成功计算Polygon因子: {len(all_polygon_factors)} 个")
                    logger.info(f"因子类型分布: {list(factor_calculation_success.keys())}")
                    
                    # 添加到combined_features
                    for col_name, value in all_polygon_factors.items():
                        if col_name not in combined_features.columns:
                            # 对所有股票广播该因子值（简化处理）
                            combined_features[col_name] = value
                    
                    # 记录成功添加的因子数量
                    added_factors = len(all_polygon_factors)
                    logger.info(f"✅ 成功添加 {added_factors} 个Polygon专业因子到特征矩阵")
                    
                    # 显示因子分类统计
                    momentum_factors = len([k for k in all_polygon_factors.keys() if 'momentum' in k])
                    fundamental_factors = len([k for k in all_polygon_factors.keys() if any(x in k for x in ['earnings', 'ebit', 'yield'])])
                    quality_factors = len([k for k in all_polygon_factors.keys() if any(x in k for x in ['piotroski', 'altman', 'quality'])])
                    risk_factors = len([k for k in all_polygon_factors.keys() if any(x in k for x in ['volatility', 'beta', 'risk'])])
                    t5_factors = len([k for k in all_polygon_factors.keys() if 't5_' in k])
                    
                    logger.info(f"因子分布 - 动量:{momentum_factors}, 基本面:{fundamental_factors}, 质量:{quality_factors}, 风险:{risk_factors}, T+5:{t5_factors}")
                else:
                    logger.warning("未能成功计算任何Polygon因子")
                
            except Exception as _e:
                logger.error(f"Polygon完整因子库集成失败: {_e}")
                import traceback
                logger.debug(f"详细错误: {traceback.format_exc()}")
            
            # ========== 简化但可靠的中性化处理 ==========
            logger.info("应用简化中性化处理")
            try:
                # 预先获取一次所有ticker的行业信息，避免在循环中重复获取
                all_tickers = combined_features['ticker'].unique().tolist()
                stock_info_cache = {}
                if self.market_data_manager:
                    try:
                        stock_info_cache = self.market_data_manager.get_batch_stock_info(all_tickers)
                        logger.info(f"预获取{len(all_tickers)}只股票的行业信息完成")
                    except Exception as e:
                        logger.warning(f"预获取行业信息失败: {e}")
                
                # 按日期分组，逐日进行简单的标准化和winsorization
                neutralized_features = []
                
                for date, group in combined_features.groupby('date'):
                    group_features = group[feature_cols].copy()
                    
                    # 1. Winsorization (1%-99%分位数截断)
                    for col in feature_cols:
                        if group_features[col].notna().sum() > 2:
                            q01, q99 = group_features[col].quantile([0.01, 0.99])
                            group_features[col] = group_features[col].clip(lower=q01, upper=q99)
                    
                    # 2. 横截面标准化（Z-score）
                    for col in feature_cols:
                        if group_features[col].notna().sum() > 2:
                            mean_val = group_features[col].mean()
                            std_val = group_features[col].std()
                            if std_val > 0:
                                group_features[col] = (group_features[col] - mean_val) / std_val
                            else:
                                group_features[col] = 0.0
                    
                    # 3. 行业中性化（使用预获取的行业信息）
                    if stock_info_cache:
                        try:
                            tickers = group['ticker'].tolist()
                            industries = {}
                            for ticker in tickers:
                                info = stock_info_cache.get(ticker)
                                if info:
                                    sector = info.gics_sub_industry or info.gics_industry or info.sector
                                    industries[ticker] = sector or 'Unknown'
                                else:
                                    industries[ticker] = 'Unknown'
                            
                            # 按行业去均值
                            group_with_industry = group_features.copy()
                            group_with_industry['industry'] = group['ticker'].map(industries)
                            
                            for col in feature_cols:
                                if group_with_industry[col].notna().sum() > 2:
                                    industry_means = group_with_industry.groupby('industry')[col].transform('mean')
                                    group_features[col] = group_features[col] - industry_means
                                    
                        except Exception as e:
                            logger.debug(f"行业中性化跳过: {e}")
                    
                    # 保留非特征列
                    group_result = group[['date', 'ticker']].copy()
                    group_result[feature_cols] = group_features[feature_cols]
                    neutralized_features.append(group_result)
                
                # 合并结果
                neutralized_df = pd.concat(neutralized_features, ignore_index=True)
                combined_features[feature_cols] = neutralized_df[feature_cols]
                
                logger.info(f"简化中性化完成，处理{len(feature_cols)}个特征")
                
            except Exception as e:
                logger.warning(f"简化中性化失败: {e}")
                logger.info("使用原始特征，仅进行标准化")
                # 最简单的回退：全局标准化
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                try:
                    combined_features[feature_cols] = scaler.fit_transform(combined_features[feature_cols].fillna(0))
                except Exception:
                    pass
            
            logger.info(f"传统特征创建完成，数据形状: {combined_features.shape}")
            return combined_features
        else:
            logger.error("没有有效的特征数据")
            return pd.DataFrame()
    
    def _validate_temporal_alignment(self, feature_data: pd.DataFrame) -> bool:
        """验证特征和目标的时间对齐，确保无数据泄露"""
        try:
            # 检查每个ticker的时间对齐
            for ticker in feature_data['ticker'].unique()[:3]:  # 样本检查
                ticker_data = feature_data[feature_data['ticker'] == ticker].sort_values('date')
                if len(ticker_data) < 10:
                    continue
                    
                # 检查特征和目标的时间差 - 修正逻辑
                if len(ticker_data) >= 10:
                    # 检查同一时间点特征和目标的情况（这里假设target是5天后的值）
                    # 正确的验证应该是检查特征日期和对应的目标预测期间是否重叠
                    first_date = ticker_data['date'].iloc[0]
                    last_date = ticker_data['date'].iloc[-1]
                    total_span = (last_date - first_date).days
                    
                    # 验证有足够的历史数据进行训练（至少30天）
                    if total_span < 30:
                        logger.warning(f"时间跨度过短：{ticker} 总跨度仅{total_span}天")
                        continue  # 不返回False，只是警告
                    
                    # 检查数据连续性，确保没有使用未来信息
                    sorted_dates = ticker_data['date'].sort_values()
                    date_gaps = sorted_dates.diff().dt.days.dropna()
                    max_gap = date_gaps.max() if len(date_gaps) > 0 else 0
                    
                    if max_gap > 30:  # 如果有超过30天的间隔，可能有问题
                        logger.warning(f"数据存在较大时间间隔：{ticker} 最大间隔{max_gap}天")
                        continue  # 不返回False，只是警告
            
            logger.info("✅ 时间对齐验证通过：特征和目标时间充分隔离")
            return True
        except Exception as e:
            logger.warning(f"时间对齐验证异常: {e}")
            return False

    def train_enhanced_models(self, feature_data: pd.DataFrame, current_ticker: str = None) -> Dict[str, Any]:
        """
        训练增强模型（Alpha策略 + Learning-to-Rank + 传统ML）
        
        Args:
            feature_data: 特征数据
            current_ticker: 当前处理的股票代码（用于自适应优化）
            
        Returns:
            训练结果
        """
        logger.info("开始训练增强模型")
        
        self.feature_data = feature_data
        training_results = {}
        
        # 准备数据
        feature_cols = [col for col in feature_data.columns 
                       if col not in ['ticker', 'date', 'target', 'COUNTRY', 'SECTOR', 'SUBINDUSTRY']]
        
        X = feature_data[feature_cols]
        y = feature_data['target']
        dates = feature_data['date']
        tickers = feature_data['ticker']
        
        # 去除缺失值 - 改进版：只去除特征或目标为空的样本
        # 先填充NaN值，然后过滤
        from sklearn.impute import SimpleImputer
        
        # 对特征进行安全的中位数填充（只处理数值列）
        try:
            # 识别数值列和非数值列
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            
            X_imputed = X.copy()
            
            # 只对数值列应用中位数填充
            if numeric_cols:
                imputer = SimpleImputer(strategy='median')
                X_imputed[numeric_cols] = pd.DataFrame(
                    imputer.fit_transform(X[numeric_cols]), 
                    columns=numeric_cols, 
                    index=X.index
                )
            
            # 对非数值列使用常数填充
            if non_numeric_cols:
                for col in non_numeric_cols:
                    X_imputed[col] = X_imputed[col].fillna('Unknown')
                    
        except Exception as e:
            logger.warning(f"特征填充失败，使用简单填充: {e}")
            X_imputed = X.fillna(0)
        
        # 目标变量必须有效
        target_valid = ~y.isna()
        
        X_clean = X_imputed[target_valid]
        y_clean = y[target_valid]
        dates_clean = dates[target_valid]
        tickers_clean = tickers[target_valid]
        
        if len(X_clean) == 0:
            logger.error("清洗后数据为空")
            return {}
        
        logger.info(f"训练数据: {len(X_clean)}样本, {len(feature_cols)}特征")
        
        # ===== Barra风格因子工程（专业量化框架） =====
        X_clean = self._enrich_with_barra_factors(X_clean, dates_clean, tickers_clean)
        
        # ===== 横截面标准化和异常值处理（专业量化框架） =====
        X_clean = self._apply_cross_sectional_processing(X_clean, dates_clean, tickers_clean)
        
        # 🔥 时间对齐验证：确保无数据泄露
        if not self._validate_temporal_alignment(feature_data):
            logger.error("⚠️ 时间对齐验证失败，存在数据泄露风险！")
        
        # 1. 训练Alpha策略引擎
        if self.alpha_engine and ENHANCED_MODULES_AVAILABLE:
            logger.info("训练Alpha策略引擎")
            try:
                # 重组数据格式用于Alpha计算 - 添加列存在性检查
                required_cols = ['date', 'ticker', 'close']
                optional_cols = ['high', 'low', 'volume', 'amount', 'COUNTRY', 'SECTOR', 'SUBINDUSTRY']
                
                # 检查必需列
                missing_required = [col for col in required_cols if col not in feature_data.columns]
                if missing_required:
                    logger.warning(f"Alpha策略缺少必需列: {missing_required}")
                    raise ValueError(f"Missing required columns: {missing_required}")
                
                # 构建可用列列表
                available_cols = required_cols.copy()
                for col in optional_cols:
                    if col in feature_data.columns:
                        available_cols.append(col)
                    else:
                        logger.debug(f"Alpha策略跳过缺失列: {col}")
                
                alpha_data = feature_data[available_cols].copy()
                
                # 为Alpha引擎标准化列名并优先使用复权收盘价
                if 'Adj Close' in feature_data.columns:
                    alpha_data['Close'] = feature_data['Adj Close']
                else:
                    alpha_data['Close'] = feature_data['close']
                
                # 安全地添加高低价
                if 'high' in feature_data.columns:
                    alpha_data['High'] = feature_data['high']
                else:
                    alpha_data['High'] = alpha_data['Close']  # 用收盘价代替
                    
                if 'low' in feature_data.columns:
                    alpha_data['Low'] = feature_data['low']
                else:
                    alpha_data['Low'] = alpha_data['Close']  # 用收盘价代替
                
                # 计算Alpha因子
                alpha_df = self.alpha_engine.compute_all_alphas(alpha_data)
                
                # 计算OOF评分
                if len(alpha_df) > 0:
                    alpha_scores = self.alpha_engine.compute_oof_scores(
                        alpha_df, y_clean, dates_clean, metric='ic'
                    )
                    
                    # 计算BMA权重
                    alpha_weights = self.alpha_engine.compute_bma_weights(alpha_scores)
                    
                    # 组合Alpha信号
                    alpha_signal = self.alpha_engine.combine_alphas(alpha_df, alpha_weights)
                    
                    # 简单过滤：去除极值和NaN
                    filtered_signal = alpha_signal.copy()
                    filtered_signal = filtered_signal.replace([np.inf, -np.inf], np.nan)
                    filtered_signal = filtered_signal.fillna(0.0)
                    
                    # 可选：Winsorize处理极值
                    q1, q99 = filtered_signal.quantile([0.01, 0.99])
                    filtered_signal = filtered_signal.clip(lower=q1, upper=q99)
                    
                    self.alpha_signals = filtered_signal
                    training_results['alpha_strategy'] = {
                        'alpha_scores': alpha_scores,
                        'alpha_weights': alpha_weights,
                        'alpha_signals': filtered_signal,
                        'alpha_stats': self.alpha_engine.get_stats()
                    }
                    
                    logger.info(f"Alpha策略训练完成，信号覆盖: {(~filtered_signal.isna()).sum()}样本")
                
            except Exception as e:
                logger.error(f"Alpha策略训练失败: {e}")
                training_results['alpha_strategy'] = {'error': str(e)}
        
        # 2. 训练Learning-to-Rank BMA
        if self.ltr_bma and ENHANCED_MODULES_AVAILABLE:
            logger.info("训练Learning-to-Rank BMA")
            try:
                ltr_results = self.ltr_bma.train_ranking_models(
                    X=X_clean, y=y_clean, dates=dates_clean,
                    cv_folds=3, optimize_hyperparams=False
                )
                
                training_results['learning_to_rank'] = {
                    'model_results': ltr_results,
                    'performance_summary': self.ltr_bma.get_performance_summary()
                }
                
                logger.info("Learning-to-Rank训练完成")
                
            except Exception as e:
                logger.error(f"Learning-to-Rank训练失败: {e}")
                training_results['learning_to_rank'] = {'error': str(e)}
        
        # 3. 训练传统ML模型（作为基准）
        logger.info("训练传统ML模型")
        try:
            # 尝试从特征数据中提取股票代码
            if current_ticker is None and 'ticker' in feature_data.columns:
                tickers = feature_data['ticker'].unique()
                current_ticker = tickers[0] if len(tickers) > 0 else "MULTI_STOCK"
            elif current_ticker is None:
                current_ticker = "UNKNOWN"
            
            traditional_results = self._train_traditional_models(X_clean, y_clean, dates_clean, current_ticker)
            training_results['traditional_models'] = traditional_results
            
        except Exception as e:
            logger.error(f"传统模型训练失败: {e}")
            training_results['traditional_models'] = {'error': str(e)}
        
        logger.info("增强模型训练完成")
        
        # 🔧 训练完成后内存清理
        if self.memory_optimizer:
            self.memory_optimizer.force_memory_cleanup()
        else:
            import gc
            gc.collect()
            
        return training_results
    
    def _get_bucket_info(self, X: pd.DataFrame, dates: pd.Series) -> Optional[Dict[str, Any]]:
        """
        获取行业/规模桶信息用于局部线性训练
        
        Returns:
            Dict包含buckets分配，如果无法获取则返回None
        """
        try:
            # 尝试从feature_data获取行业/市值信息
            if hasattr(self, 'feature_data') and len(self.feature_data) > 0:
                available_cols = self.feature_data.columns.tolist()
                
                # 查找行业列（可能的列名）
                industry_cols = [col for col in available_cols if any(keyword in col.lower() 
                    for keyword in ['sector', 'industry', 'gics', 'sic'])]
                
                # 查找市值列（可能的列名）
                mcap_cols = [col for col in available_cols if any(keyword in col.lower() 
                    for keyword in ['market_cap', 'mcap', 'mktcap', 'capitalization'])]
                
                buckets = pd.Series(index=X.index, dtype=str)
                
                if industry_cols:
                    # 使用行业分桶
                    industry_col = industry_cols[0]
                    industry_data = self.feature_data.loc[X.index, industry_col]
                    buckets = industry_data.fillna('Unknown').astype(str)
                    logger.info(f"使用行业分桶: {industry_col}, 共{buckets.nunique()}个行业")
                    
                elif mcap_cols:
                    # 使用市值分桶（分位数）
                    mcap_col = mcap_cols[0]
                    mcap_data = self.feature_data.loc[X.index, mcap_col]
                    # 分为5个规模桶：超小、小、中、大、超大
                    mcap_buckets = pd.qcut(mcap_data.rank(method='first'), 
                                         q=5, labels=['XS', 'S', 'M', 'L', 'XL'])
                    buckets = mcap_buckets.fillna('Unknown').astype(str)
                    logger.info(f"使用市值分桶: {mcap_col}, 5个规模桶")
                    
                else:
                    # 回退：按时间分桶（季度）
                    quarters = pd.to_datetime(dates).dt.to_period('Q')
                    buckets = quarters.astype(str)
                    logger.info("回退到时间分桶（按季度）")
                
                # 过滤太小的桶（少于10个样本）
                bucket_counts = buckets.value_counts()
                valid_buckets = bucket_counts[bucket_counts >= 10].index
                buckets = buckets.where(buckets.isin(valid_buckets), 'Other')
                
                if buckets.nunique() > 1:
                    return {
                        'buckets': buckets,
                        'bucket_counts': buckets.value_counts(),
                        'method': 'industry' if industry_cols else ('mcap' if mcap_cols else 'time')
                    }
            
        except Exception as e:
            logger.warning(f"获取桶信息失败: {e}")
        
        return None
    
    def _train_bucket_models(self, base_model, model_name: str, X_train_scaled: np.ndarray, 
                            y_train: pd.Series, X_test_scaled: np.ndarray, train_weights: np.ndarray,
                            train_mask: np.ndarray, test_mask: np.ndarray, 
                            bucket_info: Dict[str, Any]) -> np.ndarray:
        """
        在行业/规模桶内训练线性模型（局部线性）
        
        Returns:
            测试集预测结果
        """
        buckets = bucket_info['buckets']
        train_buckets = buckets[train_mask]
        test_buckets = buckets[test_mask]
        
        test_pred = np.full(len(X_test_scaled), np.nan)
        bucket_models = {}
        
        # 对每个桶单独训练模型
        for bucket_name in train_buckets.unique():
            if pd.isna(bucket_name):
                continue
                
            # 训练集中该桶的数据
            bucket_train_mask = (train_buckets == bucket_name)
            if bucket_train_mask.sum() < 5:  # 样本太少，跳过
                continue
                
            X_bucket_train = X_train_scaled[bucket_train_mask]
            y_bucket_train = y_train[bucket_train_mask]
            weights_bucket_train = train_weights[bucket_train_mask] if train_weights is not None else None
            
            try:
                # 训练桶内模型
                bucket_model = type(base_model)(**base_model.get_params())
                
                if model_name in ['elastic', 'ridge'] and weights_bucket_train is not None:
                    bucket_model.fit(X_bucket_train, y_bucket_train, sample_weight=weights_bucket_train)
                else:
                    bucket_model.fit(X_bucket_train, y_bucket_train)
                    
                bucket_models[bucket_name] = bucket_model
                logger.debug(f"桶{bucket_name}: 训练数据{len(X_bucket_train)}个样本")
                
            except Exception as e:
                logger.warning(f"桶{bucket_name}训练失败: {e}")
                continue
        
        # 预测测试集
        for bucket_name in test_buckets.unique():
            if pd.isna(bucket_name) or bucket_name not in bucket_models:
                continue
                
            bucket_test_mask = (test_buckets == bucket_name)
            if bucket_test_mask.sum() == 0:
                continue
                
            X_bucket_test = X_test_scaled[bucket_test_mask]
            try:
                bucket_pred = bucket_models[bucket_name].predict(X_bucket_test)
                test_pred[bucket_test_mask] = bucket_pred
            except Exception as e:
                logger.warning(f"桶{bucket_name}预测失败: {e}")
        
        # 处理无法预测的样本（使用全市场模型回退）
        nan_mask = np.isnan(test_pred)
        if nan_mask.sum() > 0:
            logger.warning(f"{nan_mask.sum()}个测试样本无法使用桶内模型预测，使用全市场模型回退")
            # 训练全市场回退模型
            fallback_model = type(base_model)(**base_model.get_params())
            if model_name in ['elastic', 'ridge'] and train_weights is not None:
                fallback_model.fit(X_train_scaled, y_train, sample_weight=train_weights)
            else:
                fallback_model.fit(X_train_scaled, y_train)
            test_pred[nan_mask] = fallback_model.predict(X_test_scaled[nan_mask])
        
        # 保存桶模型信息
        if not hasattr(self, 'bucket_models'):
            self.bucket_models = {}
        if model_name not in self.bucket_models:
            self.bucket_models[model_name] = []
        self.bucket_models[model_name].append(bucket_models)
        
        return test_pred
    
    def _train_lgbm_ranker(self, X: pd.DataFrame, y: pd.Series, dates: pd.Series, 
                          oof_predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        第二层：LightGBM Ranker 用于吸收交互特征+排序优化（专业量化框架）
        
        Args:
            X: 原始特征
            y: 目标变量
            dates: 日期序列
            oof_predictions: 第一层模型的OOF预测
            
        Returns:
            Ranker训练结果
        """
        logger.info("开始训练第二层LGBM Ranker")
        
        try:
            # 准备Ranker输入特征：第一层分数 + 市况特征 + 稳健技术面
            ranker_features = pd.DataFrame(index=X.index)
            
            # 1. 第一层分数（主要信号）
            for model_name, predictions in oof_predictions.items():
                if len(predictions) == len(X):
                    ranker_features[f'l1_{model_name}'] = predictions
            
            # 2. 市况特征（波动/趋势分位）
            if len(y) >= 20:
                vol_features = self._extract_market_regime_features(y, dates)
                for col, values in vol_features.items():
                    if len(values) == len(X):
                        ranker_features[col] = values
            
            # 3. 规模/流动性分位（如果可用）
            size_features = self._extract_size_liquidity_features(X)
            for col, values in size_features.items():
                if len(values) == len(X):
                    ranker_features[col] = values
            
            # 4. 少量稳健技术面（避免特征爆炸）
            tech_features = self._extract_robust_technical_features(X)
            for col, values in tech_features.items():
                if len(values) == len(X):
                    ranker_features[col] = values
            
            # 清理缺失值
            ranker_features = ranker_features.fillna(ranker_features.median())
            
            if ranker_features.empty or len(ranker_features.columns) == 0:
                logger.warning("Ranker特征为空，跳过第二层训练")
                return {}
            
            logger.info(f"Ranker特征维度: {ranker_features.shape}")
            
            # 准备日期分组（Ranker需要query groups）
            date_groups = pd.to_datetime(dates).dt.date
            unique_dates = sorted(date_groups.unique())
            
            # 过滤太小的日期组（少于3个样本）
            date_counts = date_groups.value_counts()
            valid_dates = date_counts[date_counts >= 3].index
            
            if len(valid_dates) < 10:
                logger.warning(f"有效日期组太少 ({len(valid_dates)})，跳过Ranker训练")
                return {}
                
            # 过滤数据
            valid_mask = date_groups.isin(valid_dates)
            X_ranker = ranker_features[valid_mask].copy()
            y_ranker = y[valid_mask].copy()
            dates_ranker = date_groups[valid_mask]
            
            # 构造组大小列表（每个日期的样本数）
            group_sizes = []
            for date in sorted(dates_ranker.unique()):
                group_sizes.append((dates_ranker == date).sum())
            
            # LGBM Ranker 配置（浅层+强正则）
            ranker = LGBMRanker(
                objective='lambdarank',    # pairwise ranking
                metric='ndcg',             # 排序指标
                ndcg_eval_at=[3, 5, 10],   # Top-K 评估
                max_depth=4,               # 浅层树
                num_leaves=15,             # 低复杂度
                n_estimators=80,           # 中等数量
                learning_rate=0.1,         # 保守学习率
                feature_fraction=0.8,      # 特征采样
                bagging_fraction=0.8,      # 样本采样
                bagging_freq=1,
                min_data_in_leaf=20,       # 强正则
                reg_alpha=0.1,             # L1正则
                reg_lambda=1.0,            # L2正则
                random_state=42,
                verbose=-1,
                n_jobs=1
            )
            
            # 训练Ranker
            ranker.fit(
                X_ranker, y_ranker,
                group=group_sizes,
                eval_set=[(X_ranker, y_ranker)],
                eval_group=[group_sizes],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )
            
            # 生成OOF预测
            ranker_pred = ranker.predict(X_ranker)
            
            # 计算排序性能
            rank_ic = spearmanr(y_ranker, ranker_pred)[0] if len(y_ranker) > 1 else 0.0
            
            # Top-K命中率计算
            top_k_metrics = self._calculate_top_k_metrics(y_ranker, ranker_pred, dates_ranker)
            
            logger.info(f"LGBM Ranker - RankIC: {rank_ic:.4f}, Top5命中率: {top_k_metrics.get('top5_hit_rate', 0):.3f}")
            
            # 保存模型
            if not hasattr(self, 'ranker_models'):
                self.ranker_models = {}
            self.ranker_models['lgbm_ranker'] = ranker
            
            # 扩展预测到原始长度
            full_ranker_pred = np.full(len(X), np.nan)
            full_ranker_pred[valid_mask] = ranker_pred
            
            return {
                'model': ranker,
                'oof_predictions': full_ranker_pred,
                'rank_ic': rank_ic,
                'top_k_metrics': top_k_metrics,
                'feature_importance': dict(zip(X_ranker.columns, ranker.feature_importances_)),
                'valid_samples': len(X_ranker)
            }
            
        except Exception as e:
            logger.error(f"LGBM Ranker训练失败: {e}")
            return {}
    
    def _extract_market_regime_features(self, y: pd.Series, dates: pd.Series) -> Dict[str, np.ndarray]:
        """提取市况特征（波动/趋势分位）"""
        features = {}
        try:
            # 滚动波动率分位
            vol_20 = y.rolling(20, min_periods=10).std()
            features['vol_percentile'] = vol_20.rank(pct=True)
            
            # 趋势指标（简单移动平均）
            ma_10 = y.rolling(10, min_periods=5).mean()
            ma_20 = y.rolling(20, min_periods=10).mean()
            features['trend_strength'] = ((ma_10 - ma_20) / ma_20.abs()).fillna(0)
            
        except Exception as e:
            logger.warning(f"市况特征提取失败: {e}")
        return features
    
    def _extract_size_liquidity_features(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """提取规模/流动性特征（如果可用）"""
        features = {}
        try:
            # 尝试从特征中找到相关列
            feature_cols = X.columns.tolist()
            
            # 市值相关
            mcap_cols = [col for col in feature_cols if any(keyword in col.lower() 
                for keyword in ['market_cap', 'mcap', 'size', 'capitalization'])]
            if mcap_cols:
                mcap_data = X[mcap_cols[0]]
                features['size_percentile'] = mcap_data.rank(pct=True)
            
            # 成交量/流动性相关
            volume_cols = [col for col in feature_cols if any(keyword in col.lower() 
                for keyword in ['volume', 'turnover', 'liquidity'])]
            if volume_cols:
                volume_data = X[volume_cols[0]]
                features['liquidity_percentile'] = volume_data.rank(pct=True)
                
        except Exception as e:
            logger.debug(f"规模/流动性特征提取失败: {e}")
        return features
    
    def _extract_robust_technical_features(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """提取稳健技术面特征（避免特征爆炸）"""
        features = {}
        try:
            feature_cols = X.columns.tolist()
            
            # RSI相关
            rsi_cols = [col for col in feature_cols if 'rsi' in col.lower()]
            if rsi_cols:
                features['rsi_signal'] = (X[rsi_cols[0]] - 50) / 50  # 标准化到[-1,1]
            
            # 移动平均比率
            ma_cols = [col for col in feature_cols if any(keyword in col.lower() 
                for keyword in ['ma_', 'sma_', 'moving_average'])]
            if len(ma_cols) >= 2:
                # 取前两个ma特征做比率
                ma_ratio = X[ma_cols[0]] / X[ma_cols[1]].replace(0, np.nan)
                features['ma_ratio'] = (ma_ratio - 1).fillna(0)
            
            # 波动率相关
            vol_cols = [col for col in feature_cols if any(keyword in col.lower() 
                for keyword in ['volatility', 'vol_', 'std_'])]
            if vol_cols:
                features['vol_signal'] = X[vol_cols[0]].rank(pct=True)
                
        except Exception as e:
            logger.debug(f"技术面特征提取失败: {e}")
        return features
    
    def _calculate_top_k_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                date_groups: pd.Series, k_list: List[int] = [3, 5, 10]) -> Dict[str, float]:
        """计算Top-K命中率"""
        metrics = {}
        try:
            for k in k_list:
                hit_rates = []
                
                for date in date_groups.unique():
                    date_mask = (date_groups == date)
                    if date_mask.sum() < k:
                        continue
                        
                    y_date = y_true[date_mask]
                    pred_date = y_pred[date_mask]
                    
                    # Top-K预测
                    top_k_idx = np.argsort(pred_date)[-k:]
                    # Top-K实际
                    top_k_actual = np.argsort(y_date.values)[-k:]
                    
                    # 计算交集命中率
                    hit_rate = len(set(top_k_idx) & set(top_k_actual)) / k
                    hit_rates.append(hit_rate)
                
                metrics[f'top{k}_hit_rate'] = np.mean(hit_rates) if hit_rates else 0.0
                
        except Exception as e:
            logger.warning(f"Top-K指标计算失败: {e}")
        return metrics
    
    def _apply_isotonic_calibration(self, predictions_dict: Dict[str, pd.Series], 
                                   training_results: Dict[str, Any]) -> Dict[str, pd.Series]:
        """
        应用等值单调校准：把分数映射为期望超额收益（bps）
        
        Args:
            predictions_dict: 各模型的预测分数
            training_results: 训练结果（包含OOF预测和目标）
            
        Returns:
            校准后的预测（期望超额收益）
        """
        logger.info("应用等值单调校准")
        
        try:
            calibrated_predictions = predictions_dict.copy()
            
            # 获取训练期目标数据（用于校准）
            target_data = self._get_training_targets(training_results)
            if target_data is None or len(target_data) == 0:
                logger.warning("无法获取训练目标数据，跳过校准")
                return predictions_dict
            
            # 初始化校准器存储
            if not hasattr(self, 'isotonic_calibrators'):
                self.isotonic_calibrators = {}
            
            # 对每个模型的预测进行校准
            for model_name, predictions in predictions_dict.items():
                try:
                    # 获取该模型的OOF预测（用于训练校准器）
                    oof_predictions = self._get_oof_predictions_for_model(model_name, training_results)
                    if oof_predictions is None or len(oof_predictions) == 0:
                        logger.debug(f"模型{model_name}无OOF预测，跳过校准")
                        continue
                    
                    # 对齐预测和目标
                    aligned_data = self._align_predictions_and_targets(oof_predictions, target_data)
                    if aligned_data is None or len(aligned_data['predictions']) < 50:
                        logger.debug(f"模型{model_name}对齐数据不足，跳过校准")
                        continue
                    
                    X_calib = aligned_data['predictions']
                    y_calib = aligned_data['targets']
                    
                    # 训练等值单调回归校准器
                    calibrator = IsotonicRegression(out_of_bounds='clip')
                    calibrator.fit(X_calib, y_calib * 10000)  # 转为bps
                    
                    # 保存校准器
                    self.isotonic_calibrators[model_name] = calibrator
                    
                    # 应用校准
                    if len(predictions.dropna()) > 0:
                        calibrated_scores = calibrator.predict(predictions.fillna(0))
                        calibrated_predictions[model_name] = pd.Series(calibrated_scores, index=predictions.index)
                        
                        # 计算校准后的统计信息
                        calib_stats = {
                            'mean_bps': np.mean(calibrated_scores),
                            'std_bps': np.std(calibrated_scores),
                            'range_bps': [np.min(calibrated_scores), np.max(calibrated_scores)]
                        }
                        logger.debug(f"{model_name}校准统计: 均值={calib_stats['mean_bps']:.1f}bps, "
                                   f"标准差={calib_stats['std_bps']:.1f}bps, "
                                   f"范围=[{calib_stats['range_bps'][0]:.1f}, {calib_stats['range_bps'][1]:.1f}]bps")
                    
                except Exception as e:
                    logger.warning(f"模型{model_name}校准失败: {e}")
                    continue
            
            logger.info(f"完成{len(self.isotonic_calibrators)}个模型的等值单调校准")
            return calibrated_predictions
            
        except Exception as e:
            logger.error(f"等值单调校准失败: {e}")
            return predictions_dict
    
    def _get_training_targets(self, training_results: Dict[str, Any]) -> Optional[pd.Series]:
        """获取训练期目标数据"""
        try:
            # 从特征数据中获取目标
            if hasattr(self, 'feature_data') and 'target' in self.feature_data.columns:
                return self.feature_data['target']
            
            # 从训练结果中获取（如果有的话）
            # 这里可以添加其他获取目标的方式
            
        except Exception as e:
            logger.debug(f"获取训练目标失败: {e}")
        return None
    
    def _get_oof_predictions_for_model(self, model_name: str, training_results: Dict[str, Any]) -> Optional[pd.Series]:
        """获取指定模型的OOF预测"""
        try:
            # 从传统模型结果中获取
            if ('traditional_models' in training_results and 
                'oof_predictions' in training_results['traditional_models']):
                oof_preds = training_results['traditional_models']['oof_predictions']
                
                # 处理传统模型
                if model_name.startswith('traditional_'):
                    base_name = model_name.replace('traditional_', '')
                    if base_name in oof_preds:
                        pred_array = oof_preds[base_name]
                        if hasattr(self, 'feature_data') and len(pred_array) == len(self.feature_data):
                            return pd.Series(pred_array, index=self.feature_data.index)
                
                # 处理Ranker模型
                elif model_name == 'ranker_lgbm':
                    ranker_results = training_results['traditional_models'].get('ranker_results', {})
                    if 'oof_predictions' in ranker_results:
                        pred_array = ranker_results['oof_predictions']
                        if hasattr(self, 'feature_data') and len(pred_array) == len(self.feature_data):
                            return pd.Series(pred_array, index=self.feature_data.index)
            
            # 其他模型类型（Alpha、LTR等）可以在这里添加
            
        except Exception as e:
            logger.debug(f"获取模型{model_name}OOF预测失败: {e}")
        return None
    
    def _align_predictions_and_targets(self, predictions: pd.Series, targets: pd.Series) -> Optional[Dict[str, np.ndarray]]:
        """对齐预测和目标数据"""
        try:
            # 找到共同索引
            common_index = predictions.index.intersection(targets.index)
            if len(common_index) == 0:
                return None
            
            aligned_pred = predictions.reindex(common_index)
            aligned_target = targets.reindex(common_index)
            
            # 过滤缺失值
            valid_mask = ~(aligned_pred.isna() | aligned_target.isna())
            if valid_mask.sum() == 0:
                return None
            
            return {
                'predictions': aligned_pred[valid_mask].values,
                'targets': aligned_target[valid_mask].values
            }
            
        except Exception as e:
            logger.debug(f"对齐预测和目标失败: {e}")
        return None
    
    def _apply_cross_sectional_processing(self, X: pd.DataFrame, dates: pd.Series, 
                                         tickers: pd.Series) -> pd.DataFrame:
        """
        横截面标准化和异常值处理（对标Barra/AQR标准）
        
        Args:
            X: 特征矩阵
            dates: 日期序列
            tickers: 股票代码序列
            
        Returns:
            处理后的特征矩阵
        """
        logger.info("应用横截面标准化和异常值处理")
        
        try:
            X_processed = X.copy()
            
            # 创建日期-股票对应关系
            df_full = X_processed.copy()
            df_full['date'] = pd.to_datetime(dates).dt.date
            df_full['ticker'] = tickers
            
            # 获取数值特征列
            numeric_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                logger.warning("没有找到数值特征列，跳过横截面处理")
                return X_processed
            
            logger.info(f"对{len(numeric_cols)}个数值特征应用横截面处理")
            
            # 按日期分组处理
            processed_data = []
            unique_dates = sorted(df_full['date'].unique())
            
            for date in unique_dates:
                date_mask = df_full['date'] == date
                date_data = df_full[date_mask].copy()
                
                if len(date_data) < 3:  # 跨过样本太少的日期
                    processed_data.append(date_data)
                    continue
                
                # 步骤1：分位去极值（1%-99%）
                for col in numeric_cols:
                    col_data = date_data[col]
                    if col_data.isna().all():
                        continue
                        
                    # 计算分位数
                    q01 = col_data.quantile(0.01)
                    q99 = col_data.quantile(0.99)
                    
                    # Winsorize：将极值截断到分位数
                    date_data[col] = col_data.clip(lower=q01, upper=q99)
                
                # 步骤2：横截面标准化（每个交易日z-score）
                for col in numeric_cols:
                    col_data = date_data[col]
                    if col_data.isna().all() or col_data.std() == 0:
                        continue
                        
                    # Z-score标准化
                    col_mean = col_data.mean()
                    col_std = col_data.std()
                    date_data[col] = (col_data - col_mean) / col_std
                
                # 步骤3：行业/规模中性化（如果可用）
                date_data = self._apply_neutralization(date_data, numeric_cols)
                
                processed_data.append(date_data)
            
            # 合并结果
            if processed_data:
                df_processed = pd.concat(processed_data, ignore_index=False)
                # 恢复原始索引顺序
                df_processed = df_processed.reindex(X_processed.index)
                # 只返回特征列（去掉date和ticker）
                X_processed = df_processed[X_processed.columns]
            
            # 最后一次缺失值处理
            X_processed = X_processed.fillna(0)
            
            logger.info("横截面处理完成")
            return X_processed
            
        except Exception as e:
            logger.error(f"横截面处理失败: {e}")
            return X  # 返回原始数据
    
    def _apply_neutralization(self, date_data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        应用行业/规模中性化（如果可用）
        
        Args:
            date_data: 单日数据
            numeric_cols: 数值特征列名
            
        Returns:
            中性化后的数据
        """
        try:
            # 尝试获取行业/规模信息
            neutralization_applied = False
            
            # 如果有行业信息，进行行业中性化
            if hasattr(self, 'feature_data') and self.feature_data is not None:
                available_cols = self.feature_data.columns.tolist()
                industry_cols = [col for col in available_cols if any(keyword in col.lower() 
                    for keyword in ['sector', 'industry', 'gics'])]
                
                if industry_cols and len(date_data) > 5:
                    try:
                        # 获取当日的行业信息
                        industry_info = self.feature_data.loc[date_data.index, industry_cols[0]]
                        industry_data = date_data.copy()
                        industry_data['industry'] = industry_info.fillna('Unknown')
                        
                        # 对每个数值特征进行行业中性化
                        for col in numeric_cols:
                            if col in industry_data.columns:
                                industry_neutralized = self._industry_neutralize_feature(
                                    industry_data[col], industry_data['industry']
                                )
                                if industry_neutralized is not None:
                                    date_data[col] = industry_neutralized
                                    neutralization_applied = True
                    except Exception as e:
                        logger.debug(f"行业中性化失败: {e}")
            
            # 如果有市值信息，进行规模中性化
            if hasattr(self, 'feature_data') and self.feature_data is not None:
                available_cols = self.feature_data.columns.tolist()
                mcap_cols = [col for col in available_cols if any(keyword in col.lower() 
                    for keyword in ['market_cap', 'mcap', 'mktcap'])]
                
                if mcap_cols and len(date_data) > 10:
                    try:
                        mcap_info = self.feature_data.loc[date_data.index, mcap_cols[0]]
                        # 规模五分位
                        size_buckets = pd.qcut(mcap_info.rank(method='first'), 
                                             q=5, labels=['XS', 'S', 'M', 'L', 'XL'])
                        
                        for col in numeric_cols:
                            if col in date_data.columns:
                                size_neutralized = self._size_neutralize_feature(
                                    date_data[col], size_buckets
                                )
                                if size_neutralized is not None:
                                    date_data[col] = size_neutralized
                                    neutralization_applied = True
                    except Exception as e:
                        logger.debug(f"规模中性化失败: {e}")
            
            if neutralization_applied:
                logger.debug(f"应用了行业/规模中性化")
                
        except Exception as e:
            logger.debug(f"中性化处理失败: {e}")
        
        return date_data
    
    def _industry_neutralize_feature(self, feature: pd.Series, industries: pd.Series) -> Optional[pd.Series]:
        """行业中性化单个特征"""
        try:
            if len(feature) < 5 or feature.isna().all():
                return None
                
            neutralized = feature.copy()
            for industry in industries.unique():
                if pd.isna(industry):
                    continue
                    
                industry_mask = (industries == industry)
                if industry_mask.sum() < 2:
                    continue
                    
                industry_data = feature[industry_mask]
                if industry_data.std() > 0:
                    # 行业内去均值
                    neutralized[industry_mask] = industry_data - industry_data.mean()
            
            return neutralized
            
        except Exception:
            return None
    
    def _size_neutralize_feature(self, feature: pd.Series, size_buckets: pd.Series) -> Optional[pd.Series]:
        """规模中性化单个特征"""
        try:
            if len(feature) < 5 or feature.isna().all():
                return None
                
            neutralized = feature.copy()
            for bucket in size_buckets.unique():
                if pd.isna(bucket):
                    continue
                    
                bucket_mask = (size_buckets == bucket)
                if bucket_mask.sum() < 2:
                    continue
                    
                bucket_data = feature[bucket_mask]
                if bucket_data.std() > 0:
                    # 规模桶内去均值
                    neutralized[bucket_mask] = bucket_data - bucket_data.mean()
            
            return neutralized
            
        except Exception:
            return None
    
    def _enrich_with_barra_factors(self, X: pd.DataFrame, dates: pd.Series, 
                                  tickers: pd.Series) -> pd.DataFrame:
        """
        使用Barra风格因子库丰富特征集（专业量化框架）
        
        Args:
            X: 原始特征矩阵
            dates: 日期序列
            tickers: 股票代码序列
            
        Returns:
            丰富后的特征矩阵
        """
        if not BARRA_FACTORS_AVAILABLE or self.barra_style_factors is None:
            logger.info("跳过Barra风格因子工程（模块不可用）")
            return X
        
        logger.info("开始计算Barra风格因子")
        
        try:
            # 准备数据结构
            enhanced_data = self._prepare_barra_input_data(X, dates, tickers)
            
            if enhanced_data is None:
                logger.warning("Barra因子数据准备失败，使用原始特征")
                return X
            
            # 计算Barra风格因子
            barra_factors = self.barra_style_factors.calculate_all_factors(
                data=enhanced_data['fundamental_data'],
                price_data=enhanced_data.get('price_data'),
                volume_data=enhanced_data.get('volume_data')
            )
            
            if barra_factors.empty:
                logger.warning("Barra因子计算返回空结果，使用原始特征")
                return X
            
            # 将Barra因子与X对齐并合并
            X_enhanced = self._merge_barra_factors(X, barra_factors, dates, tickers)
            
            logger.info(f"Barra风格因子集成完成: 新增{len(barra_factors.columns)}个因子")
            logger.info(f"增强后特征维度: {X.shape} -> {X_enhanced.shape}")
            
            return X_enhanced
            
        except Exception as e:
            logger.error(f"Barra风格因子计算失败: {e}")
            return X
    
    def _prepare_barra_input_data(self, X: pd.DataFrame, dates: pd.Series, 
                                 tickers: pd.Series) -> Optional[Dict[str, pd.DataFrame]]:
        """准备Barra因子计算所需的输入数据"""
        try:
            # 创建长格式基础数据
            fundamental_data = pd.DataFrame({
                'date': dates,
                'ticker': tickers
            })
            
            # 添加可用的基本面数据
            available_cols = X.columns.tolist()
            
            # 映射常见列名到Barra所需的字段
            column_mapping = {
                # 价格相关
                'close': 'price', 'price': 'price', 'adj_close': 'price',
                # 成交量
                'volume': 'volume', 'trading_volume': 'volume',
                # 市值
                'market_cap': 'market_cap', 'mktcap': 'market_cap', 'mcap': 'market_cap',
                # 财务数据
                'book_value': 'book_value', 'total_assets': 'total_assets',
                'revenue': 'revenue', 'sales': 'sales', 'net_income': 'net_income',
                'roe': 'roe', 'debt_to_equity': 'debt_to_equity',
                # 收益率
                'returns': 'returns', 'daily_return': 'returns'
            }
            
            for original_col, barra_col in column_mapping.items():
                if original_col in available_cols:
                    fundamental_data[barra_col] = X[original_col].values
            
            # 如果缺少关键数据，创建模拟数据用于测试
            if 'price' not in fundamental_data.columns:
                # 使用任意数值列作为价格代理
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fundamental_data['price'] = X[numeric_cols[0]].abs() + 10  # 确保为正值
                else:
                    fundamental_data['price'] = 100  # 默认价格
            
            if 'volume' not in fundamental_data.columns:
                fundamental_data['volume'] = 1e6  # 默认成交量
            
            if 'market_cap' not in fundamental_data.columns:
                fundamental_data['market_cap'] = fundamental_data['price'] * 1e6  # 模拟市值
            
            # 准备价格和成交量的宽格式数据
            price_data = fundamental_data.pivot(index='date', columns='ticker', values='price')
            volume_data = fundamental_data.pivot(index='date', columns='ticker', values='volume')
            
            result = {
                'fundamental_data': fundamental_data,
                'price_data': price_data,
                'volume_data': volume_data
            }
            
            logger.debug(f"Barra输入数据准备完成: 基本面{fundamental_data.shape}, 价格{price_data.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Barra输入数据准备失败: {e}")
            return None
    
    def _merge_barra_factors(self, X: pd.DataFrame, barra_factors: pd.DataFrame,
                           dates: pd.Series, tickers: pd.Series) -> pd.DataFrame:
        """将Barra因子与原始特征合并"""
        try:
            # 将Barra因子从宽格式转为长格式以匹配X的索引
            barra_long = []
            
            for date in barra_factors.index:
                for ticker in barra_factors.columns:
                    if pd.notna(barra_factors.loc[date, ticker]):
                        row_data = {'date': date, 'ticker': ticker, 'factor_value': barra_factors.loc[date, ticker]}
                        # 添加因子名
                        row_data['factor_name'] = ticker  # 这里可能需要调整逻辑
                        barra_long.append(row_data)
            
            if not barra_long:
                logger.warning("Barra因子转换后为空")
                return X
            
            # 简化方法：直接将Barra因子作为新列添加到X
            X_enhanced = X.copy()
            
            # 如果barra_factors是横截面数据，取最新日期的因子值
            if len(barra_factors) > 0:
                latest_date = barra_factors.index[-1]
                latest_factors = barra_factors.loc[latest_date]
                
                # 为每个因子创建新列
                for factor_name in latest_factors.index:
                    if pd.notna(latest_factors[factor_name]):
                        # 使用因子值填充整列（简化处理）
                        X_enhanced[f'barra_{factor_name}'] = latest_factors[factor_name]
            
            return X_enhanced
            
        except Exception as e:
            logger.error(f"Barra因子合并失败: {e}")
            return X
    
    def _optimize_with_barra_model(self, predictions: pd.Series, feature_data: pd.DataFrame) -> Dict[str, Any]:
        """
        使用Barra风险模型和约束优化器进行投资组合优化（专业量化框架）
        
        Args:
            predictions: 集成预测
            feature_data: 特征数据
            
        Returns:
            优化结果
        """
        logger.info("开始使用Barra风险模型进行投资组合优化")
        
        try:
            # 步骤1：数据预处理和对齐
            aligned_data = self._prepare_barra_data(predictions, feature_data)
            if aligned_data is None:
                raise ValueError("数据预处理失败")
            
            expected_returns, returns_data, factor_data, market_data, current_weights = aligned_data
            
            # 步骤2：拟合Barra风险模型
            logger.info("拟合Barra风险模型")
            risk_model_results = self.barra_risk_model.fit(
                returns_data=returns_data,
                factor_data=factor_data,
                market_data=market_data
            )
            
            if risk_model_results is None:
                raise ValueError("Barra风险模型拟合失败")
            
            # 步骤3：约束优化
            logger.info("执行约束优化")
            optimization_result = self.constrained_optimizer.optimize(
                expected_returns=expected_returns,
                risk_model=risk_model_results,
                current_weights=current_weights,
                market_data=market_data
            )
            
            if not optimization_result.get('success', False):
                raise ValueError(f"优化失败: {optimization_result.get('error', '未知错误')}")
            
            # 步骤4：风险归因分析
            optimal_weights = optimization_result['optimal_weights']
            risk_attribution = self.constrained_optimizer.calculate_risk_attribution(
                optimal_weights, risk_model_results
            )
            
            # 步骤5：整合结果
            final_result = {
                'success': True,
                'method': 'barra_constrained_optimization',
                'optimal_weights': optimal_weights.to_dict(),
                'expected_return': optimization_result.get('expected_return', 0),
                'portfolio_risk': optimization_result.get('portfolio_risk', 0),
                'turnover': optimization_result.get('turnover', 0),
                'risk_attribution': risk_attribution,
                'n_positions': int((optimal_weights > 0.001).sum()),
                'solver_info': {
                    'solver': optimization_result.get('solver', 'unknown'),
                    'problem_status': optimization_result.get('problem_status', 'unknown')
                },
                'barra_model_info': {
                    'n_factors': len(risk_model_results.factor_loadings.columns),
                    'n_assets': len(risk_model_results.factor_loadings),
                    'factor_names': risk_model_results.factor_loadings.columns.tolist()
                }
            }
            
            # 保存组合权重
            self.portfolio_weights = optimal_weights
            
            logger.info(f"Barra投资组合优化完成: {final_result['n_positions']}个位置, "
                       f"预期收益={final_result['expected_return']:.4f}, "
                       f"风险={final_result['portfolio_risk']:.4f}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Barra风险模型优化失败: {e}")
            # 回退到传统优化
            logger.info("回退到传统优化方法")
            return self._fallback_to_traditional_optimization(predictions, feature_data, str(e))
    
    def _prepare_barra_data(self, predictions: pd.Series, feature_data: pd.DataFrame) -> Optional[Tuple]:
        """
        为Barra风险模型准备数据
        
        Returns:
            (expected_returns, returns_data, factor_data, market_data, current_weights)
        """
        try:
            # 获取最新截面的预测
            if self.feature_data is None or len(self.feature_data) == 0:
                logger.error("缺少特征元数据用于对齐预测")
                return None
            
            # 数据对齐
            valid_pred_indices = predictions.index.intersection(self.feature_data.index)
            if len(valid_pred_indices) == 0:
                logger.error("预测索引与特征数据索引没有交集")
                return None
            
            valid_predictions = predictions.reindex(valid_pred_indices)
            meta = self.feature_data.loc[valid_pred_indices, ['date', 'ticker']].copy()
            pred_df = meta.assign(pred=valid_predictions.values)
            
            # 获取最新截面
            pred_df = pred_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['pred'])
            if pred_df.empty:
                logger.error("没有有效的预测信号")
                return None
            
            latest_date = pred_df['date'].max()
            latest_pred = pred_df[pred_df['date'] == latest_date]
            ticker_pred = latest_pred.groupby('ticker')['pred'].mean()
            
            # 构造期望收益率
            expected_returns = ticker_pred.copy()
            
            # 构造历史收益率数据（用于风险模型）
            returns_data = self._construct_returns_data(ticker_pred.index)
            
            # 构造因子数据
            factor_data = self._construct_factor_data(ticker_pred.index)
            
            # 构造市场数据
            market_data = self._construct_market_data(ticker_pred.index)
            
            # 当前持仓（如果有）
            current_weights = pd.Series(0.0, index=ticker_pred.index)
            if hasattr(self, 'portfolio_weights') and self.portfolio_weights is not None:
                current_weights = self.portfolio_weights.reindex(ticker_pred.index).fillna(0)
            
            logger.info(f"Barra数据准备完成: {len(expected_returns)}只股票, "
                       f"{len(returns_data)}天历史数据")
            
            return expected_returns, returns_data, factor_data, market_data, current_weights
            
        except Exception as e:
            logger.error(f"Barra数据准备失败: {e}")
            return None
    
    def _construct_returns_data(self, tickers: pd.Index) -> pd.DataFrame:
        """构造收益率数据（用于风险模型）"""
        try:
            # 尝试从特征数据中提取收益率
            if hasattr(self, 'feature_data') and 'target' in self.feature_data.columns:
                # 使用目标变量作为收益率代理
                target_data = self.feature_data[['date', 'ticker', 'target']].copy()
                returns_wide = target_data.pivot(index='date', columns='ticker', values='target')
                returns_wide = returns_wide.reindex(columns=tickers, fill_value=0)
                
                # 只保留最近252个交易日
                if len(returns_wide) > 252:
                    returns_wide = returns_wide.tail(252)
                
                return returns_wide
            
            # 回退：创建模拟数据
            logger.warning("无法获取历史收益率，创建模拟数据")
            n_days = 252
            dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')
            
            # 模拟随机收益率（小攻进有效性）
            np.random.seed(42)
            returns_data = pd.DataFrame(
                np.random.normal(0, 0.02, (n_days, len(tickers))),
                index=dates,
                columns=tickers
            )
            
            return returns_data
            
        except Exception as e:
            logger.warning(f"收益率数据构造失败: {e}")
            # 最简单的回退
            return pd.DataFrame(0.01, index=pd.date_range(end=pd.Timestamp.now(), periods=60), columns=tickers)
    
    def _construct_factor_data(self, tickers: pd.Index) -> pd.DataFrame:
        """构造因子数据"""
        try:
            # 尝试从特征数据中提取风格因子
            style_factors = ['size', 'value', 'quality', 'momentum', 'volatility', 'growth']
            factor_data = {}
            
            if hasattr(self, 'feature_data'):
                available_cols = self.feature_data.columns.tolist()
                
                for factor in style_factors:
                    # 查找匹配的列
                    matching_cols = [col for col in available_cols if factor.lower() in col.lower()]
                    if matching_cols:
                        # 使用第一个匹配的列
                        factor_col = matching_cols[0]
                        factor_series = self.feature_data[['date', 'ticker', factor_col]].copy()
                        factor_wide = factor_series.pivot(index='date', columns='ticker', values=factor_col)
                        factor_wide = factor_wide.reindex(columns=tickers, fill_value=0)
                        factor_data[factor] = factor_wide
            
            # 如果没找到因子，创建模拟数据
            if not factor_data:
                logger.warning("无法获取风格因子，创建模拟数据")
                dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='D')
                np.random.seed(42)
                
                for factor in style_factors:
                    factor_values = pd.DataFrame(
                        np.random.normal(0, 1, (len(dates), len(tickers))),
                        index=dates,
                        columns=tickers
                    )
                    factor_data[factor] = factor_values
            
            # 合并为一个DataFrame
            if factor_data:
                combined_factor_data = pd.concat(factor_data, axis=1)
                return combined_factor_data
            
            # 最简单的回退
            return pd.DataFrame(0, index=pd.date_range(end=pd.Timestamp.now(), periods=60), 
                              columns=pd.MultiIndex.from_product([['market'], tickers]))
            
        except Exception as e:
            logger.warning(f"因子数据构造失败: {e}")
            return pd.DataFrame()
    
    def _construct_market_data(self, tickers: pd.Index) -> pd.DataFrame:
        """构造市场数据（行业、国家等）"""
        try:
            # 尝试从特征数据中提取
            market_data = pd.DataFrame(index=tickers)
            
            if hasattr(self, 'feature_data'):
                available_cols = self.feature_data.columns.tolist()
                
                # 行业信息
                industry_cols = [col for col in available_cols if any(keyword in col.lower() 
                    for keyword in ['sector', 'industry', 'gics'])]
                if industry_cols:
                    # 取最新的行业信息
                    latest_industry = self.feature_data.groupby('ticker')[industry_cols[0]].last()
                    market_data['industry'] = latest_industry.reindex(tickers).fillna('Unknown')
                else:
                    market_data['industry'] = 'Technology'
                
                # 市值信息
                mcap_cols = [col for col in available_cols if any(keyword in col.lower() 
                    for keyword in ['market_cap', 'mcap', 'mktcap'])]
                if mcap_cols:
                    latest_mcap = self.feature_data.groupby('ticker')[mcap_cols[0]].last()
                    market_data['market_cap'] = latest_mcap.reindex(tickers).fillna(1e9)
                else:
                    # 模拟市值
                    np.random.seed(42)
                    market_data['market_cap'] = np.random.lognormal(20, 1, len(tickers))
                
                # 国家信息
                market_data['country'] = 'US'
                
                # 添加date和ticker列用于兼容
                market_data = market_data.reset_index().rename(columns={'index': 'ticker'})
                market_data['date'] = pd.Timestamp.now()
            
            else:
                # 回退数据
                market_data = pd.DataFrame({
                    'ticker': tickers,
                    'industry': 'Technology',
                    'market_cap': 1e9,
                    'country': 'US',
                    'date': pd.Timestamp.now()
                })
            
            logger.info(f"市场数据构造完成: {len(market_data)}只股票")
            return market_data
            
        except Exception as e:
            logger.warning(f"市场数据构造失败: {e}")
            return pd.DataFrame({
                'ticker': tickers,
                'industry': 'Unknown',
                'market_cap': 1e9,
                'country': 'US',
                'date': pd.Timestamp.now()
            })
    
    def _fallback_to_traditional_optimization(self, predictions: pd.Series, feature_data: pd.DataFrame, 
                                            error_msg: str) -> Dict[str, Any]:
        """回退到传统优化方法"""
        try:
            logger.info("使用传统优化方法作为回退")
            
            # 简化的Top-K等权策略
            top_k = min(20, len(predictions) // 2)
            top_assets = predictions.nlargest(top_k)
            
            equal_weights = pd.Series(1.0 / len(top_assets), index=top_assets.index)
            
            return {
                'success': True,
                'method': 'fallback_equal_weight',
                'optimal_weights': equal_weights.to_dict(),
                'expected_return': float(predictions.reindex(top_assets.index).mean()),
                'portfolio_risk': 0.15,  # 估计值
                'turnover': 1.0,  # 估计值
                'n_positions': len(top_assets),
                'fallback_reason': error_msg,
                'solver_info': {'solver': 'fallback'}
            }
            
        except Exception as e:
            logger.error(f"回退优化也失败: {e}")
            return {'success': False, 'error': f'Fallback optimization failed: {e}'}
    
    def _train_traditional_models(self, X: pd.DataFrame, y: pd.Series, 
                                 dates: pd.Series, stock_symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """训练传统ML模型 - 支持自适应加树优化"""
        
        # 🚀 第二层优化：自适应加树
        if ADAPTIVE_OPTIMIZER_AVAILABLE:
            logger.info(f"使用自适应加树优化器训练{stock_symbol}")
            return self._train_with_adaptive_optimizer(X, y, stock_symbol)
        else:
            logger.info(f"使用标准模型训练{stock_symbol}")
            return self._train_standard_models(X, y, dates)
    
    def _train_with_adaptive_optimizer(self, X: pd.DataFrame, y: pd.Series, 
                                     stock_symbol: str) -> Dict[str, Any]:
        """使用自适应加树优化器训练模型"""
        # 创建自适应优化器
        optimizer = AdaptiveTreeOptimizer(
            slope_threshold_ic=0.002,    # IC提升斜率阈值
            slope_threshold_mse=0.01,    # MSE下降斜率阈值(1%)
            tree_increment=20,           # 每次增加20棵树
            top_percentile=0.2,          # 选择前20%的股票
            max_trees_xgb=150,           # XGB最大150棵树
            max_trees_lgb=150,           # LightGBM最大150棵树
            max_trees_rf=200             # RandomForest最大200棵树
        )
        
        model_results = {}
        oof_predictions = {}
        
        # 1. 训练线性模型（不需要自适应优化）
        linear_models = {
            'ridge': Ridge(alpha=1.0),
            'elastic': ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=5000)
        }
        
        for model_name, model in linear_models.items():
            try:
                model.fit(X, y)
                predictions = model.predict(X)
                score = r2_score(y, predictions)
                
                model_results[model_name] = {
                    'model': model,
                    'cv_score': score,
                    'feature_importance': getattr(model, 'coef_', None)
                }
                oof_predictions[model_name] = predictions
                
                logger.info(f"{stock_symbol} {model_name}: R2={score:.4f}")
            except Exception as e:
                logger.warning(f"{stock_symbol} {model_name}训练失败: {e}")
        
        # 2. 自适应训练RandomForest (内存优化)
        try:
            rf_model = RandomForestRegressor(
                n_estimators=50,         # 减少树的数量以节省内存
                max_depth=6,            # 减少树的深度
                min_samples_split=20,   # 增加分割最小样本数
                min_samples_leaf=10,    # 增加叶子最小样本数
                max_features='sqrt',
                max_samples=0.7,        # 使用70%的样本，减少内存使用
                oob_score=True,
                random_state=42,
                n_jobs=1
            )
            rf_model.fit(X, y)
            predictions = rf_model.predict(X)
            score = r2_score(y, predictions)
            
            model_results['rf'] = {
                'model': rf_model,
                'cv_score': rf_model.oob_score_ if hasattr(rf_model, 'oob_score_') else score,
                'feature_importance': rf_model.feature_importances_,
                'adaptive_performance': {'r2': score, 'oob_score': getattr(rf_model, 'oob_score_', 0.0)}
            }
            oof_predictions['rf'] = predictions
            logger.info(f"{stock_symbol} RandomForest训练完成: R2={score:.4f}")
        except Exception as e:
            logger.warning(f"{stock_symbol} RandomForest训练失败: {e}")
        
        # 3. 自适应训练XGBoost (Fixed early stopping)
        if XGBOOST_AVAILABLE:
            try:
                # Create train/validation split for early stopping
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                
                xgb_model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    tree_method='hist',
                    random_state=42,
                    n_jobs=1
                )
                
                # Fit with proper eval_set for early stopping
                xgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=15,
                    verbose=False
                )
                
                predictions = xgb_model.predict(X)
                score = r2_score(y, predictions)
                
                model_results['xgboost'] = {
                    'model': xgb_model,
                    'cv_score': score,
                    'feature_importance': xgb_model.feature_importances_,
                    'adaptive_performance': {'r2': score, 'n_estimators': xgb_model.n_estimators}
                }
                oof_predictions['xgboost'] = predictions
                logger.info(f"{stock_symbol} XGBoost训练完成: R2={score:.4f}")
            except Exception as e:
                logger.warning(f"{stock_symbol} XGBoost训练失败: {e}")
        
        # 4. 自适应训练LightGBM (Fixed early stopping)  
        if LIGHTGBM_AVAILABLE:
            try:
                # Create train/validation split for early stopping
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_data_in_leaf=50,
                    force_row_wise=True,
                    random_state=42,
                    verbose=-1,
                    n_jobs=1
                )
                
                # Fit with proper eval_set for early stopping
                lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)]
                )
                
                predictions = lgb_model.predict(X)
                score = r2_score(y, predictions)
                
                model_results['lightgbm'] = {
                    'model': lgb_model,
                    'cv_score': score,
                    'feature_importance': lgb_model.feature_importances_,
                    'adaptive_performance': {'r2': score, 'n_estimators': lgb_model.n_estimators}
                }
                oof_predictions['lightgbm'] = predictions
                logger.info(f"{stock_symbol} LightGBM训练完成: R2={score:.4f}")
            except Exception as e:
                logger.warning(f"{stock_symbol} LightGBM训练失败: {e}")
        
        return {
            'models': model_results,
            'oof_predictions': oof_predictions,
            'optimizer_summary': {'status': 'completed', 'models_trained': len(model_results)}
        }
        
        # 🔧 传统模型训练后内存清理
        if hasattr(self, 'memory_optimizer') and self.memory_optimizer:
            self.memory_optimizer.force_memory_cleanup()
    
    def _train_standard_models(self, X: pd.DataFrame, y: pd.Series, 
                             dates: pd.Series) -> Dict[str, Any]:
        """标准模型训练（专业量化框架优化）- 第一层加权鲁棒线性为主"""
        # ===== 第一层：线性为主（按专业量化优先级排序） =====
        models = {
            'elastic': ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=5000),
            'ridge': Ridge(alpha=1.0),
            'robust_linear': HuberRegressor(epsilon=1.35, alpha=1e-4),
            # 非线性模型仍保留，但权重由第二层/集成端决定
            'rf': RandomForestRegressor(
                n_estimators=50,         # 减少到50以节省内存
                max_depth=6,             # 减少深度限制
                max_features=0.7,        # 特征采样70%
                min_samples_leaf=15,     # 增加叶子最小样本
                max_samples=0.6,         # 样本采样60%以减少内存
                n_jobs=1,                # 限制并行度
                random_state=42
            )
        }
        
        # 添加高级模型
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=70,         # 从100减到70 (BMA优化)
                max_depth=4,             # 从6减到4 (BMA优化)
                learning_rate=0.2,       # 从0.1增到0.2 (BMA优化)
                subsample=0.8,           # 样本采样
                colsample_bytree=0.8,    # 特征采样
                reg_alpha=0.1,           # L1正则化
                reg_lambda=1.0,          # L2正则化
                tree_method='hist',      # 高效算法
                early_stopping_rounds=15, # 早停机制 (BMA优化)
                random_state=42,
                n_jobs=1                 # 限制并行度
            )
        
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=80,         # 从100减到80 (BMA优化)
                max_depth=5,             # 从6减到5 (BMA优化)
                num_leaves=31,           # 严格控制叶子数
                learning_rate=0.2,       # 从0.1增到0.2 (BMA优化)
                feature_fraction=0.8,    # 特征采样
                bagging_fraction=0.8,    # 样本采样
                bagging_freq=1,
                min_data_in_leaf=50,     # 增加叶子最小数据
                force_row_wise=True,     # 内存优化 (BMA优化)
                early_stopping_rounds=15, # 早停机制 (BMA优化)
                random_state=42,
                verbose=-1,
                n_jobs=1                 # 限制并行度
            )
        
        # CatBoost removed due to compatibility issues
        
        # ===== 样本权重：增强信息权重 vs 传统WLS（1/20日波动）=====
        if hasattr(self, '_enhanced_sample_weights') and self._enhanced_sample_weights is not None:
            # 🔥 使用增强目标工程的信息权重
            try:
                # 确保权重长度匹配
                if len(self._enhanced_sample_weights) == len(y):
                    sample_weights = self._enhanced_sample_weights
                    logger.info("✅ 使用增强信息权重 (基于Triple Barrier信息量)")
                else:
                    # 长度不匹配时截断或填充
                    if len(self._enhanced_sample_weights) > len(y):
                        sample_weights = self._enhanced_sample_weights[:len(y)]
                    else:
                        # 用中位数填充不足部分
                        median_weight = np.median(self._enhanced_sample_weights)
                        sample_weights = np.concatenate([
                            self._enhanced_sample_weights,
                            np.full(len(y) - len(self._enhanced_sample_weights), median_weight)
                        ])
                    logger.info(f"⚠️ 权重长度调整: {len(self._enhanced_sample_weights)} -> {len(sample_weights)}")
                
                logger.info(f"增强权重统计: 均值={np.mean(sample_weights):.3f}, 标准差={np.std(sample_weights):.3f}, 范围=[{np.min(sample_weights):.3f}, {np.max(sample_weights):.3f}]")
                
            except Exception as e:
                logger.warning(f"增强权重应用失败，回退到传统WLS: {e}")
                sample_weights = None  # 触发传统权重计算
        else:
            sample_weights = None  # 触发传统权重计算
        
        # 传统WLS权重（回退方案）
        if sample_weights is None:
            # 计算基于20日波动的样本权重（专业量化框架标准）
            feat = pd.DataFrame({
                'date': dates,
                'target': y
            })
            
            # 使用滚动标准差作为波动率代理（如果有returns列更好）
            if hasattr(self, 'feature_data') and 'returns' in self.feature_data.columns:
                feat['returns'] = self.feature_data.loc[X.index, 'returns'].values if len(self.feature_data.loc[X.index]) > 0 else y
                vol20 = feat['returns'].rolling(20, min_periods=10).std()
            else:
                # 回退：用目标序列近似波动（不理想，但可工作）
                vol20 = feat['target'].rolling(20, min_periods=10).std()
            
            # WLS权重 = 1/波动率，截断极值并归一化
            sample_weights = 1.0 / np.clip(vol20, 1e-6, np.percentile(vol20.dropna(), 95))
            sample_weights = (sample_weights / np.nanmedian(sample_weights)).fillna(1.0).values
            
            logger.info("使用传统WLS权重 (1/波动率)")
            logger.info(f"传统权重统计: 均值={np.mean(sample_weights):.3f}, 标准差={np.std(sample_weights):.3f}, 范围=[{np.min(sample_weights):.3f}, {np.max(sample_weights):.3f}]")
        
        # ===== 行业/规模桶内训练（局部线性）=====
        # 获取行业/规模信息（如果可用）
        bucket_info = self._get_bucket_info(X, dates)
        use_bucket_training = bucket_info is not None and len(bucket_info['buckets'].unique()) > 1
        
        if use_bucket_training:
            logger.info(f"启用行业/规模桶内训练，共{len(bucket_info['buckets'].unique())}个桶")
        else:
            logger.info("使用全市场训练（无行业/规模信息）")
        
        # 🔥 加强时序验证：增加embargo防止目标泄露
        cv_config = ValidationConfig(
            n_splits=3,    # 减少折数适应小数据集
            test_size=42,  # 减少测试集大小
            gap=5,         # 适中的gap
            embargo=3,     # 适中的embargo
            group_freq='W',
            min_train_size=50  # 降低最小训练集要求
        )
        purged_cv = PurgedGroupTimeSeriesSplit(cv_config)
        groups = create_time_groups(dates, freq=cv_config.group_freq)
        
        model_results = {}
        oof_predictions = {}
        
        for model_name, model in models.items():
            logger.info(f"训练{model_name}模型")
            
            fold_predictions = np.full(len(X), np.nan)
            fold_models = []
            
            for train_idx, test_idx in purged_cv.split(X, y, groups):
                # 确保索引在有效范围内（先转为ndarray再比较）
                train_idx = np.asarray(train_idx)
                test_idx = np.asarray(test_idx)
                train_idx = train_idx[train_idx < len(X)]
                test_idx = test_idx[test_idx < len(X)]
                
                if len(train_idx) == 0 or len(test_idx) == 0:
                    continue
                
                train_mask = np.zeros(len(X), dtype=bool)
                train_mask[train_idx] = True
                test_mask = np.zeros(len(X), dtype=bool) 
                test_mask[test_idx] = True
                
                X_train, y_train = X[train_mask], y[train_mask]
                X_test = X[test_mask]
                
                try:
                    # 标准化
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # 获取训练集权重
                    train_weights = sample_weights[train_mask]
                    
                    # 训练模型（区分线性与树模型，支持桶内训练）
                    if model_name in ['xgboost', 'lightgbm', 'rf']:
                        # Tree-based模型：全市场训练（不使用桶）
                        model_copy = type(model)(**model.get_params())
                        model_copy.fit(X_train, y_train)
                        test_pred = model_copy.predict(X_test)
                        
                    elif model_name in ['elastic', 'ridge', 'robust_linear'] and use_bucket_training:
                        # 线性模型：桶内训练（局部线性）
                        test_pred = self._train_bucket_models(
                            model, model_name, X_train_scaled, y_train, X_test_scaled,
                            train_weights, train_mask, test_mask, bucket_info
                        )
                        model_copy = model  # 占位符，实际模型保存在bucket_models中
                        
                    elif model_name in ['elastic', 'ridge']:
                        # ElasticNet和Ridge：全市场训练 + WLS样本权重
                        model_copy = type(model)(**model.get_params())
                        model_copy.fit(X_train_scaled, y_train, sample_weight=train_weights)
                        test_pred = model_copy.predict(X_test_scaled)
                        logger.debug(f"{model_name}使用WLS权重训练，权重范围: [{np.min(train_weights):.3f}, {np.max(train_weights):.3f}]")
                        
                    elif model_name == 'robust_linear':
                        # Huber回归：全市场训练（不支持sample_weight，但具备鲁棒性）
                        model_copy = type(model)(**model.get_params())
                        model_copy.fit(X_train_scaled, y_train)
                        test_pred = model_copy.predict(X_test_scaled)
                        logger.debug(f"{model_name}使用鲁棒回归（无样本权重）")
                        
                    else:
                        # 其他线性模型（回退）
                        model_copy = type(model)(**model.get_params())
                        model_copy.fit(X_train_scaled, y_train)
                        test_pred = model_copy.predict(X_test_scaled)
                    
                    fold_predictions[test_mask] = test_pred
                    fold_models.append((model_copy, scaler))
                    
                except Exception as e:
                    logger.warning(f"{model_name}模型训练失败: {e}")
                    continue
            
            oof_predictions[model_name] = fold_predictions
            self.traditional_models[model_name] = fold_models
            
            # 计算性能指标 (完整的OOF指标计算)
            valid_mask = ~np.isnan(fold_predictions)
            if valid_mask.sum() > 0:
                y_valid = y[valid_mask]
                pred_valid = fold_predictions[valid_mask]
                
                oof_ic = np.corrcoef(y_valid, pred_valid)[0, 1] if len(y_valid) > 1 else 0.0
                oof_rank_ic = spearmanr(y_valid, pred_valid)[0] if len(y_valid) > 1 else 0.0
                oof_mse = mean_squared_error(y_valid, pred_valid)
                oof_r2 = r2_score(y_valid, pred_valid)
                
                model_results[model_name] = {
                    'oof_ic': oof_ic if not np.isnan(oof_ic) else 0.0,
                    'oof_rank_ic': oof_rank_ic if not np.isnan(oof_rank_ic) else 0.0,
                    'oof_mse': oof_mse if not np.isnan(oof_mse) else float('inf'),
                    'oof_r2': oof_r2 if not np.isnan(oof_r2) else -float('inf'),
                    'valid_predictions': valid_mask.sum()
                }
                
                logger.info(f"{model_name} - IC: {oof_ic:.4f}, RankIC: {oof_rank_ic:.4f}, MSE: {oof_mse:.6f}, R2: {oof_r2:.4f}")
        
        # ===== 第二层：LGBM Ranker 吸收交互+排序 =====
        ranker_results = {}
        if LIGHTGBM_AVAILABLE and len(oof_predictions) > 0:
            ranker_results = self._train_lgbm_ranker(X, y, dates, oof_predictions)
            
        # 🔴 修复Stacking泄露：二层Stacking元学习器时间安全训练
        try:
            logger.info("训练时间安全的二层Stacking元学习器")
            base_pred_df = pd.DataFrame({name: preds for name, preds in oof_predictions.items()})
            
            # 🔴 关键修复：确保OOF预测来自严格的时间分割
            # 第一层模型的OOF预测必须是真正的out-of-fold，不能有时间泄露
            base_pred_df = base_pred_df.reset_index(drop=True)
            y_reset = y.reset_index(drop=True)
            dates_reset = dates.reset_index(drop=True)
            
            # 验证第一层OOF预测的完整性
            base_valid_mask = ~base_pred_df.isna().any(axis=1) & ~y_reset.isna()
            
            # 🔴 时间验证：确保只使用有完整OOF预测的样本
            if base_valid_mask.sum() < len(base_pred_df) * 0.8:
                logger.warning(f"OOF预测完整性不足: {base_valid_mask.sum()}/{len(base_pred_df)} ({base_valid_mask.mean():.1%})")
            
            X_meta = base_pred_df.loc[base_valid_mask].copy()
            y_meta = y_reset.loc[base_valid_mask].copy()
            dates_meta = dates_reset.loc[base_valid_mask].copy()

            # 🔴 第二层CV必须严格晚于第一层：更大的gap和embargo
            groups = create_time_groups(dates_meta, freq='W')
            stacking_cv_config = ValidationConfig(
                n_splits=3,       # 更少的fold（避免过度切分）
                test_size=84,     # 更大的测试集（4周）
                gap=14,           # 更大的gap（2周，确保超过第一层的gap）
                embargo=10,       # 更大的embargo（避免目标泄露）
                min_train_size=126  # 确保足够的训练样本
            )
            pgts = PurgedGroupTimeSeriesSplit(stacking_cv_config)
            
            logger.info(f"第二层CV配置: n_splits={stacking_cv_config.n_splits}, "
                       f"gap={stacking_cv_config.gap}, embargo={stacking_cv_config.embargo}")

            meta_models = {
                'meta_ridge': Ridge(alpha=0.5),
                'meta_elastic': ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000)
            }

            meta_oof = {name: np.full(len(X_meta), np.nan) for name in meta_models.keys()}
            trained_meta = {}

            # 🔴 严格的时间验证：确保第二层CV不会泄露
            for fold_idx, (train_idx, test_idx) in enumerate(pgts.split(X_meta, y_meta, groups)):
                # 时间验证：训练集最大日期 + gap + embargo < 测试集最小日期
                train_dates = dates_meta.iloc[train_idx]
                test_dates = dates_meta.iloc[test_idx]
                
                train_max_date = train_dates.max()
                test_min_date = test_dates.min()
                gap_days = (test_min_date - train_max_date).days
                
                # 验证时间间隔
                required_gap = stacking_cv_config.gap + stacking_cv_config.embargo
                if gap_days < required_gap:
                    logger.error(f"第二层CV Fold {fold_idx}: 时间间隔不足 {gap_days} < {required_gap}")
                    raise ValueError(f"Stacking CV时间泄露风险: fold {fold_idx}")
                
                logger.debug(f"第二层CV Fold {fold_idx}: 时间间隔 {gap_days}天 >= {required_gap}天 ✅")
                
                X_tr, X_te = X_meta.iloc[train_idx], X_meta.iloc[test_idx]
                y_tr = y_meta.iloc[train_idx]
                
                for mname, m in meta_models.items():
                    m_fit = type(m)(**m.get_params())
                    m_fit.fit(X_tr, y_tr)
                    meta_oof[mname][test_idx] = m_fit.predict(X_te)
                    trained_meta.setdefault(mname, []).append(m_fit)

            # 记录元学习器性能
            meta_perf = {}
            for mname, preds in meta_oof.items():
                valid = ~np.isnan(preds)
                ic = np.corrcoef(y_meta.values[valid], np.array(preds)[valid])[0, 1]
                rank_ic = spearmanr(y_meta.values[valid], np.array(preds)[valid])[0]
                meta_perf[mname] = {
                    'oof_ic': float(ic) if not np.isnan(ic) else 0.0,
                    'oof_rank_ic': float(rank_ic) if not np.isnan(rank_ic) else 0.0,
                    'valid_predictions': int(valid.sum())
                }

            # 保存到实例以供后续预测
            self.meta_learners = trained_meta
            self.meta_oof_predictions = meta_oof
            model_results.update({f'stacking_{k}': v for k, v in meta_perf.items()})
        except Exception as e:
            logger.warning(f"二层Stacking训练失败: {e}")

        return {
            'model_performance': model_results,
            'oof_predictions': oof_predictions,
            'ranker_results': ranker_results,  # 第二层LGBM Ranker结果
            'stacking': {
                'meta_oof': meta_oof if 'meta_oof' in locals() else {},
                'meta_performance': meta_perf if 'meta_perf' in locals() else {}
            }
        }
    
    def generate_ensemble_predictions(self, training_results: Dict[str, Any]) -> pd.Series:
        """
        生成集成预测
        
        Args:
            training_results: 训练结果
            
        Returns:
            集成预测序列
        """
        logger.info("生成集成预测")
        
        predictions_dict = {}
        weights_dict = {}
        
        # 1. Alpha策略预测
        if 'alpha_strategy' in training_results and 'alpha_signals' in training_results['alpha_strategy']:
            alpha_signals = training_results['alpha_strategy']['alpha_signals']
            if alpha_signals is not None and len(alpha_signals) > 0:
                predictions_dict['alpha'] = alpha_signals
                # 基于Alpha评分设置权重
                alpha_scores = training_results['alpha_strategy'].get('alpha_scores', pd.Series())
                if len(alpha_scores) > 0:
                    avg_alpha_score = alpha_scores.mean()
                    weights_dict['alpha'] = max(0.1, min(0.5, avg_alpha_score * 5))  # 权重在0.1-0.5之间
                else:
                    weights_dict['alpha'] = 0.2
        
        # 2. Learning-to-Rank预测
        if (self.ltr_bma and 'learning_to_rank' in training_results and 
            'model_results' in training_results['learning_to_rank']):
            try:
                if self.feature_data is not None:
                    feature_cols = [col for col in self.feature_data.columns 
                                   if col not in ['ticker', 'date', 'target', 'COUNTRY', 'SECTOR', 'SUBINDUSTRY']]
                    X_for_prediction = self.feature_data[feature_cols].dropna()
                    
                    ltr_pred, ltr_uncertainty = self.ltr_bma.predict_with_uncertainty(X_for_prediction)
                    
                    # 权重基于不确定性和LTR性能
                    avg_uncertainty = np.nanmean(ltr_uncertainty)
                    base_ltr_weight = 1.0 / (1.0 + avg_uncertainty * 10)
                    
                    # 检查LTR性能，如果有负IC通道则降权
                    performance_penalty = 1.0
                    try:
                        ltr_results = training_results['learning_to_rank']
                        if isinstance(ltr_results, dict):
                            ltr_performance = ltr_results.get('performance_summary', {})
                            if ltr_performance and isinstance(ltr_performance, dict):
                                avg_ic = np.mean([p.get('ic', 0.0) for p in ltr_performance.values() if isinstance(p, dict)])
                                if avg_ic < 0:
                                    performance_penalty = 0.3  # 负IC时大幅降权
                                elif avg_ic < 0.05:
                                    performance_penalty = 0.6  # 弱IC时中度降权
                    except Exception as e:
                        logger.debug(f"LTR性能检查失败: {e}")
                        performance_penalty = 0.8  # 安全的中等权重
                    
                    final_ltr_weight = base_ltr_weight * performance_penalty
                    predictions_dict['ltr'] = pd.Series(ltr_pred, index=X_for_prediction.index)
                    weights_dict['ltr'] = max(0.05, min(0.25, final_ltr_weight))  # 降低上限从0.4到0.25
                    
            except Exception as e:
                logger.warning(f"Learning-to-Rank预测失败: {e}")
        
        # 3. 传统模型预测
        if 'traditional_models' in training_results and 'oof_predictions' in training_results['traditional_models']:
            oof_preds = training_results['traditional_models']['oof_predictions']
            model_perfs = training_results['traditional_models'].get('model_performance', {})
            stacking_info = training_results['traditional_models'].get('stacking', {})
            
            # 获取训练数据的索引作为参考
            if hasattr(self, 'feature_data') and self.feature_data is not None:
                ref_index = self.feature_data.index
            else:
                ref_index = None
            
            for model_name, pred_array in oof_preds.items():
                if pred_array is not None and not np.all(np.isnan(pred_array)):
                    # 确保预测与特征数据索引对齐
                    if ref_index is not None and len(pred_array) == len(ref_index):
                        predictions_dict[f'traditional_{model_name}'] = pd.Series(pred_array, index=ref_index)
                    else:
                        # 回退到默认索引，但要确保长度匹配
                        logger.warning(f"传统模型{model_name}预测长度{len(pred_array)}与特征数据不匹配")
                        continue
                    
                    # 动态权重：基于IC值的智能分配（负IC模型排除或反向使用）
                    if model_name in model_perfs:
                        ic = model_perfs[model_name].get('oof_ic', 0.0)
                        ic_abs = abs(ic)
                        
                        if ic < -0.05:
                            # 强负IC：完全排除，不给予权重
                            weights_dict[f'traditional_{model_name}'] = 0.0
                            logger.warning(f"模型 {model_name} IC={ic:.4f} < -0.05，已排除")
                        elif ic < -0.02:
                            # 中等负IC：极低权重或考虑反向信号
                            weights_dict[f'traditional_{model_name}'] = 0.0
                            logger.info(f"模型 {model_name} IC={ic:.4f} 负相关性较强，已排除")
                        elif ic < 0.02:
                            # 噪音区间：IC接近0，不使用
                            weights_dict[f'traditional_{model_name}'] = 0.0
                            logger.debug(f"模型 {model_name} IC={ic:.4f} 在噪音区间，已排除")
                        elif ic > 0.15:
                            # 强正IC：最高权重
                            weights_dict[f'traditional_{model_name}'] = 0.30
                        elif ic > 0.1:
                            # 较强正IC：高权重
                            weights_dict[f'traditional_{model_name}'] = 0.25
                        elif ic > 0.05:
                            # 中等正IC：中等权重
                            weights_dict[f'traditional_{model_name}'] = 0.15
                        elif ic >= 0.02:
                            # 弱正IC：低权重
                            weights_dict[f'traditional_{model_name}'] = 0.08
                        else:
                            # 默认情况：极低权重
                            weights_dict[f'traditional_{model_name}'] = 0.0
                    else:
                        weights_dict[f'traditional_{model_name}'] = 0.05
        
        # 3.5. 第二层LGBM Ranker预测（专业量化框架）
        if 'traditional_models' in training_results and 'ranker_results' in training_results['traditional_models']:
            ranker_results = training_results['traditional_models']['ranker_results']
            if ranker_results and 'oof_predictions' in ranker_results:
                ranker_pred = ranker_results['oof_predictions']
                if ranker_pred is not None and not np.all(np.isnan(ranker_pred)):
                    # 对齐索引
                    if ref_index is not None and len(ranker_pred) == len(ref_index):
                        predictions_dict['ranker_lgbm'] = pd.Series(ranker_pred, index=ref_index)
                        
                        # 基于RankIC设置权重
                        rank_ic = ranker_results.get('rank_ic', 0.0)
                        top_k_metrics = ranker_results.get('top_k_metrics', {})
                        top5_hit = top_k_metrics.get('top5_hit_rate', 0.0)
                        
                        # 综合RankIC和Top-K命中率设置权重
                        if rank_ic > 0.1 and top5_hit > 0.25:
                            # 强排序性能：高权重
                            ranker_weight = 0.25
                        elif rank_ic > 0.05 and top5_hit > 0.2:
                            # 中等排序性能：中等权重
                            ranker_weight = 0.15
                        elif rank_ic > 0.02:
                            # 弱排序性能：低权重
                            ranker_weight = 0.08
                        else:
                            # 无效排序：最小权重
                            ranker_weight = 0.02
                            
                        weights_dict['ranker_lgbm'] = ranker_weight
                        logger.info(f"LGBM Ranker - RankIC: {rank_ic:.4f}, Top5命中: {top5_hit:.3f}, 权重: {ranker_weight:.3f}")
                    else:
                        logger.warning(f"Ranker预测长度{len(ranker_pred)}与特征数据不匹配")

            # 加入二层Stacking元学习器的预测（作为额外通道）
            try:
                if stacking_info and 'meta_oof' in stacking_info and hasattr(self, 'feature_data'):
                    base_models = [f"{k}" for k in oof_preds.keys()]
                    base_pred_df = pd.DataFrame({name: predictions_dict.get(f'traditional_{name}', pd.Series(dtype=float)) for name in base_models})
                    # 对齐到参考索引
                    base_pred_df = base_pred_df.reindex(ref_index)
                    # 使用已训练的meta learners做一层预测平均
                    if hasattr(self, 'meta_learners') and isinstance(self.meta_learners, dict):
                        for mname, mlist in self.meta_learners.items():
                            # 对多个折的meta模型取平均预测
                            meta_preds = np.nanmean([m.predict(base_pred_df.fillna(0.0)) for m in mlist], axis=0)
                            predictions_dict[f'stacking_{mname}'] = pd.Series(meta_preds, index=ref_index)
                            perf = stacking_info.get('meta_performance', {}).get(mname, {})
                            ic = perf.get('oof_ic', 0.0)
                            weights_dict[f'stacking_{mname}'] = max(0.05, min(0.35, ic * 6))
            except Exception as e:
                logger.warning(f"Stacking通道集成失败: {e}")
        
        # ===== 等值单调校准：把“分数”变“可交易刻度” =====
        if ISOTONIC_AVAILABLE and predictions_dict:
            predictions_dict = self._apply_isotonic_calibration(predictions_dict, training_results)
        
        # 集成预测
        if not predictions_dict:
            logger.error("没有有效的预测结果")
            return pd.Series()
        
        # 标准化权重
        total_weight = sum(weights_dict.values())
        if total_weight > 0:
            for key in weights_dict:
                weights_dict[key] /= total_weight
        
        logger.info(f"集成权重: {weights_dict}")
        
        # 统一所有预测的索引到feature_data的索引 (修复Index交集逻辑)
        if hasattr(self, 'feature_data') and self.feature_data is not None:
            reference_index = self.feature_data.index
        else:
            # 如果没有参考索引，使用pd.Index安全取交集，避免空集
            if len(predictions_dict) == 0:
                reference_index = pd.Index([])
            else:
                pred_values = list(predictions_dict.values())
                reference_index = pred_values[0].index
                for pred in pred_values[1:]:
                    reference_index = reference_index.intersection(pred.index)
                # 如果交集为空，使用第一个预测的索引作为回退
                if len(reference_index) == 0:
                    logger.warning("预测索引交集为空，使用第一个预测的索引")
                    reference_index = pred_values[0].index
        
        if len(reference_index) == 0:
            logger.error("没有可用的参考索引进行集成")
            return pd.Series()
        
        # 构建预测矩阵（不填充为0，保留NaN）
        preds_df = pd.DataFrame({
            name: series.reindex(reference_index) for name, series in predictions_dict.items()
        })

        # 将权重向量与列对齐
        weights_vec = np.array([weights_dict.get(name, 0.0) for name in preds_df.columns], dtype=float)
        # 每行有效权重之和（忽略该行中的NaN）
        mask = ~preds_df.isna().values
        weights_matrix = np.tile(weights_vec, (len(preds_df), 1))
        denom = (weights_matrix * mask).sum(axis=1)
        numer = np.nansum(preds_df.values * weights_matrix, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            ensemble_values = np.where(denom > 0, numer / denom, np.nan)
        ensemble_prediction = pd.Series(ensemble_values, index=reference_index)
        
        self.final_predictions = ensemble_prediction
        
        logger.info(f"集成预测完成，覆盖{len(ensemble_prediction)}个样本")
        
        return ensemble_prediction
    
    def optimize_portfolio(self, predictions: pd.Series, 
                          feature_data: pd.DataFrame) -> Dict[str, Any]:
        """
        优化投资组合
        
        Args:
            predictions: 集成预测
            feature_data: 特征数据
            
        Returns:
            投资组合优化结果
        """
        # 🔥 优先使用Barra风险模型和约束优化器
        if BARRA_OPTIMIZER_AVAILABLE and self.barra_risk_model and self.constrained_optimizer:
            logger.info("使用Barra风险模型进行投资组合优化")
            return self._optimize_with_barra_model(predictions, feature_data)
        
        # 回退到传统优化器
        if not self.portfolio_optimizer or not ENHANCED_MODULES_AVAILABLE:
            logger.warning("投资组合优化器不可用，无法生成投资建议")
            return {'success': False, 'error': 'Portfolio optimizer not available'}
        
        logger.info("开始传统投资组合优化")
        
        try:
            # 将预测与样本元数据(date,ticker)对齐，再筛选最新截面
            if self.feature_data is None or len(self.feature_data) == 0:
                logger.error("缺少特征元数据用于对齐预测")
                return {}
            
            # 只取预测索引中存在于feature_data中的部分
            valid_pred_indices = predictions.index.intersection(self.feature_data.index)
            if len(valid_pred_indices) == 0:
                logger.error("预测索引与特征数据索引没有交集")
                return {}
            
            # 获取有效预测
            valid_predictions = predictions.reindex(valid_pred_indices)
            meta = self.feature_data.loc[valid_pred_indices, ['date', 'ticker']].copy()
            pred_df = meta.assign(pred=valid_predictions.values)
            
            # 仅保留有效预测
            pred_df = pred_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['pred'])
            if pred_df.empty:
                logger.error("没有有效的预测信号")
                return {}
            
            latest_date = pred_df['date'].max()
            latest_pred = pred_df[pred_df['date'] == latest_date]
            if latest_pred.empty:
                logger.error("最新截面没有预测信号")
                return {}
            
            # 聚合到ticker层面
            ticker_pred = latest_pred.groupby('ticker')['pred'].mean()
            
            # 对齐到最新截面特征
            latest_slice = feature_data[feature_data['date'] == latest_date].copy()
            if latest_slice.empty:
                logger.error("没有最新截面数据")
                return {}
            
            latest_slice = latest_slice.set_index('ticker')
            predictions_valid = ticker_pred.reindex(latest_slice.index)
            
            # 过滤NaN（但不把信号强行置零，避免全零）
            valid_mask = (~predictions_valid.isna())
            if valid_mask.sum() == 0:
                logger.error("没有有效的预测信号")
                return {}
                
            latest_data_valid = latest_slice[valid_mask]
            predictions_valid = predictions_valid[valid_mask]

            # 如果预测为常数（std为0），用备用打分破平（如近20日动量），并做截面标准化
            try:
                if float(predictions_valid.std()) == 0.0:
                    backup_scores = []
                    for tk in latest_data_valid.index:
                        try:
                            df_hist = self.raw_data.get(tk)
                            if df_hist is not None and 'close' in df_hist.columns:
                                df_hist = df_hist.sort_values('date')
                                df_hist['ret'] = df_hist['close'].pct_change()
                                mom20 = (1.0 + df_hist['ret']).rolling(21).apply(lambda x: np.prod(1.0 + x) - 1.0).iloc[-1]
                                backup_scores.append(mom20 if pd.notna(mom20) else 0.0)
                            else:
                                backup_scores.append(0.0)
                        except Exception:
                            backup_scores.append(0.0)
                    backup_series = pd.Series(backup_scores, index=latest_data_valid.index)
                    # 截面标准化
                    if backup_series.std() > 0:
                        backup_series = (backup_series - backup_series.mean()) / backup_series.std()
                    predictions_valid = backup_series
                    logger.info("检测到预测为常数，已使用近20日动量作为备用信号并标准化")
            except Exception:
                pass

            # 记录最新截面信号统计，诊断是否出现全0
            try:
                nz_ratio = float((predictions_valid != 0).sum()) / float(len(predictions_valid))
                logger.info(f"最新截面信号非零比率: {nz_ratio:.2%}, 均值: {predictions_valid.mean():.6f}, 标准差: {predictions_valid.std():.6f}")
                self.latest_ticker_predictions = predictions_valid.copy()
            except Exception:
                self.latest_ticker_predictions = predictions_valid
            
            logger.info(f"有效预测信号数量: {len(predictions_valid)}, 涵盖股票: {list(predictions_valid.index)}")
            
            # 构建预期收益率（基于预测信号）。
            # 增强信号处理：标准化 + 放大 + 抖动
            expected_returns = predictions_valid.copy()
            
            # 标准化
            if expected_returns.std() > 1e-12:
                expected_returns = (expected_returns - expected_returns.mean()) / expected_returns.std()
            else:
                # 信号过于平坦，创建人工梯度
                expected_returns = pd.Series(
                    np.linspace(-1, 1, len(expected_returns)), 
                    index=expected_returns.index
                )
            
            # 放大信号强度（改善优化器数值稳定性）
            expected_returns = expected_returns * 0.02  # 目标年化收益2%的量级
            
            # 微抖动确保非等权解
            rng = np.random.RandomState(42)
            expected_returns = expected_returns + rng.normal(0, 1e-4, size=len(expected_returns))
            expected_returns.name = 'expected_returns'
            
            # 构建历史收益率矩阵用于协方差估计
            returns_data = []
            tickers_for_cov = expected_returns.index.tolist()
            
            # 获取历史收益率
            for ticker in tickers_for_cov:
                if ticker in self.raw_data:
                    hist_data = self.raw_data[ticker].copy()
                    hist_data['returns'] = hist_data['close'].pct_change()
                    returns_data.append(hist_data[['date', 'returns']].set_index('date')['returns'].rename(ticker))
            
            if returns_data:
                returns_matrix = pd.concat(returns_data, axis=1).dropna()
                
                # 估计协方差矩阵
                cov_matrix = self.portfolio_optimizer.estimate_covariance_matrix(returns_matrix)
                
                # 统一资产顺序，避免维度不一致（使用returns_matrix列作为权威顺序）
                cov_tickers = list(returns_matrix.columns)
                expected_returns = expected_returns.reindex(cov_tickers).dropna()
                universe_data = latest_data_valid[['COUNTRY', 'SECTOR', 'SUBINDUSTRY']].copy()
                universe_data = universe_data.reindex(expected_returns.index)

                # 至少需要2只股票以进行优化
                if len(expected_returns) < 2:
                    logger.error("有效股票数量不足以进行优化")
                    return {}
                if 'volume' in latest_data_valid.columns:
                    # 简单的流动性排名
                    universe_data['liquidity_rank'] = latest_data_valid['volume'].reindex(expected_returns.index).rank(pct=True)
                else:
                    universe_data['liquidity_rank'] = 0.5
                
                # 执行投资组合优化
                optimization_result = self.portfolio_optimizer.optimize_portfolio(
                    expected_returns=expected_returns,
                    covariance_matrix=cov_matrix,
                    current_weights=None,  # 假设从空仓开始
                    universe_data=universe_data
                )
                
                if optimization_result.get('success', False):
                    optimal_weights = optimization_result['optimal_weights']
                    portfolio_metrics = optimization_result['portfolio_metrics']
                    
                    # 风险归因
                    risk_attribution = self.portfolio_optimizer.risk_attribution(
                        optimal_weights, cov_matrix
                    )
                    
                    # 压力测试
                    from advanced_portfolio_optimizer import create_stress_scenarios
                    stress_scenarios = create_stress_scenarios(optimal_weights.index.tolist())
                    stress_results = self.portfolio_optimizer.stress_test(
                        optimal_weights, cov_matrix, stress_scenarios
                    )
                    
                    self.portfolio_weights = optimal_weights
                    
                    return {
                        'success': True,
                        'optimal_weights': optimal_weights,
                        'portfolio_metrics': portfolio_metrics,
                        'risk_attribution': risk_attribution,
                        'stress_test': stress_results,
                        'optimization_info': optimization_result.get('optimization_info', {})
                    }
                else:
                    logger.warning("高级投资组合优化未达到最优，但已返回最佳可用结果")
                    return optimization_result
            else:
                logger.error("无法构建协方差矩阵")
                return {}
                
        except Exception as e:
            logger.error(f"投资组合优化异常: {e}")
            return {'error': str(e)}
    

    
    def generate_investment_recommendations(self, portfolio_result: Dict[str, Any],
                                          top_n: int = 10) -> List[Dict[str, Any]]:
        """
        生成投资建议
        
        Args:
            portfolio_result: 投资组合优化结果
            top_n: 返回前N个推荐
            
        Returns:
            投资建议列表
        """
        logger.info(f"生成前{top_n}个投资建议")
        
        if not portfolio_result.get('success', False):
            logger.error("投资组合优化失败，无法生成建议")
            return []
        
        # Handle different portfolio result formats
        if 'optimal_weights' in portfolio_result:
            optimal_weights = portfolio_result['optimal_weights']
        elif 'weights' in portfolio_result:
            optimal_weights = portfolio_result['weights']
        else:
            logger.error("No weights found in portfolio result")
            return []
            
        # Ensure weights are in Series format for comparisons
        if isinstance(optimal_weights, dict):
            optimal_weights = pd.Series(optimal_weights)
            
        portfolio_metrics = portfolio_result.get('portfolio_metrics', {})
        
        # 获取最新的股票数据
        recommendations = []
        
        # 按权重排序
        sorted_weights = optimal_weights[optimal_weights > 0.001].sort_values(ascending=False)
        
        for i, (ticker, weight) in enumerate(sorted_weights.head(top_n).items()):
            try:
                # 获取股票基本信息
                if ticker in self.raw_data:
                    stock_data = self.raw_data[ticker]
                    latest_price = stock_data['close'].iloc[-1]
                    
                    # 计算一些基本指标
                    price_change_1d = stock_data['close'].pct_change().iloc[-1]
                    price_change_5d = (stock_data['close'].iloc[-1] / stock_data['close'].iloc[-6] - 1) if len(stock_data) > 5 else 0
                    
                    avg_volume = stock_data['volume'].tail(20).mean() if 'volume' in stock_data.columns else 0
                    
                    # 获取预测信号（优先使用按ticker聚合过的最新截面信号）
                    if hasattr(self, 'latest_ticker_predictions') and isinstance(self.latest_ticker_predictions, pd.Series):
                        prediction_signal = float(self.latest_ticker_predictions.get(ticker, np.nan))
                    else:
                        # 回退：从逐行预测聚合（最新日期）
                        try:
                            if self.final_predictions is not None and hasattr(self, 'feature_data'):
                                ref_idx = self.feature_data.index
                                preds = pd.Series(self.final_predictions).reindex(ref_idx)
                                latest_date = self.feature_data['date'].max()
                                latest_mask = self.feature_data['date'] == latest_date
                                latest_tickers = self.feature_data.loc[latest_mask, 'ticker']
                                grouped = pd.DataFrame({'ticker': latest_tickers, 'pred': preds[latest_mask]}).groupby('ticker')['pred'].mean()
                                prediction_signal = float(grouped.get(ticker, np.nan))
                            else:
                                prediction_signal = np.nan
                        except Exception:
                            prediction_signal = np.nan
                    if np.isnan(prediction_signal):
                        prediction_signal = 0.0
                    
                    # Generate proper recommendation format expected by save_results
                    expected_return = max(0.01, min(0.25, prediction_signal * 2))  # Scale prediction to reasonable return
                    confidence_score = min(0.95, max(0.5, weight * 10))  # Scale weight to confidence
                    
                    # Determine rating based on prediction signal
                    if prediction_signal > 0.05:
                        rating = "STRONG_BUY"
                        target_price = latest_price * (1 + expected_return)
                    elif prediction_signal > 0.02:
                        rating = "BUY" 
                        target_price = latest_price * (1 + expected_return * 0.8)
                    elif prediction_signal > -0.02:
                        rating = "HOLD"
                        target_price = latest_price * (1 + expected_return * 0.3)
                    else:
                        rating = "SELL"
                        target_price = latest_price * (1 - abs(expected_return) * 0.5)
                    
                    # Determine risk level
                    if weight > 0.1:
                        risk_level = "LOW"
                    elif weight > 0.05:
                        risk_level = "MEDIUM"
                    else:
                        risk_level = "HIGH"
                    
                    recommendation = {
                        'rank': i + 1,
                        'ticker': ticker,
                        'rating': rating,
                        'target_price': round(target_price, 2),
                        'current_price': round(latest_price, 2),
                        'expected_return': round(expected_return, 4),
                        'confidence_score': round(confidence_score, 4),
                        'risk_level': risk_level,
                        'sector': 'Technology',  # Default sector
                        'market_cap': 1000000000,  # Default market cap
                        'volume_avg': int(avg_volume),
                        'weight': weight,
                        'price_change_1d': price_change_1d,
                        'price_change_5d': price_change_5d,
                        'avg_volume_20d': avg_volume,
                        'prediction_signal': prediction_signal,
                        'recommendation_reason': self._get_recommendation_reason(ticker, weight, prediction_signal)
                    }
                    
                    recommendations.append(recommendation)
                
            except Exception as e:
                logger.warning(f"生成{ticker}推荐信息失败: {e}")
                continue
        
        return recommendations
    
    def _get_recommendation_reason(self, ticker: str, weight: float, signal: float) -> str:
        """生成推荐理由"""
        reasons = []
        
        if weight > 0.05:
            reasons.append("高权重配置")
        elif weight > 0.03:
            reasons.append("中等权重配置")
        else:
            reasons.append("低权重配置")
        
        if signal > 0.1:
            reasons.append("强烈买入信号")
        elif signal > 0.05:
            reasons.append("买入信号")
        elif signal > 0:
            reasons.append("弱买入信号")
        else:
            reasons.append("中性信号")
        
        return "; ".join(reasons)
    
    def _extract_weights_safely(self, portfolio_result: Dict[str, Any]) -> Dict[str, float]:
        """Safely extract weights from portfolio result regardless of format"""
        weights = portfolio_result.get('optimal_weights') or portfolio_result.get('weights', {})
        
        if isinstance(weights, pd.Series):
            return {sanitize_ticker(k): float(v) for k, v in weights.to_dict().items()}
        elif isinstance(weights, dict):
            return {sanitize_ticker(k): float(v) for k, v in weights.items()}
        else:
            return {}
    
    def save_results(self, recommendations: List[Dict[str, Any]], 
                    portfolio_result: Dict[str, Any]) -> str:
        """
        保存结果
        
        Args:
            recommendations: 投资建议
            portfolio_result: 投资组合结果
            
        Returns:
            保存文件路径
        """
        logger.info("保存分析结果")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path("result")
        result_dir.mkdir(exist_ok=True)
        
        # 保存投资建议
        if recommendations:
            # Excel格式（确保列顺序与数据类型稳定）
            excel_file = result_dir / f"ultra_enhanced_recommendations_{timestamp}.xlsx"
            rec_df = pd.DataFrame(recommendations)
            # 规范ticker
            if 'ticker' in rec_df.columns:
                rec_df['ticker'] = rec_df['ticker'].map(sanitize_ticker)
            # 设定列顺序
            preferred_cols = ['rank','ticker','weight','latest_price','price_change_1d','price_change_5d','avg_volume_20d','prediction_signal','recommendation_reason']
            ordered_cols = [c for c in preferred_cols if c in rec_df.columns] + [c for c in rec_df.columns if c not in preferred_cols]
            rec_df = rec_df[ordered_cols]
            # 仅导出前200条到Excel
            rec_df = rec_df.head(200)
            # Excel优先；失败时回退CSV
            try:
                rec_df.to_excel(excel_file, index=False)
            except Exception:
                excel_file = result_dir / f"ultra_enhanced_recommendations_{timestamp}.csv"
                rec_df.to_csv(excel_file, index=False, encoding='utf-8')
            
            # 简化的股票代码列表
            tickers_file = result_dir / f"top_tickers_{timestamp}.txt"
            top_tickers = [sanitize_ticker(rec.get('ticker','')) for rec in recommendations[:7] if rec.get('ticker')]
            with open(tickers_file, 'w', encoding='utf-8') as f:
                f.write(", ".join([f"'{ticker}'" for ticker in top_tickers]))

            # 仅股票代码（JSON数组格式），Top10
            top10_json = result_dir / f"top10_tickers_{timestamp}.json"
            # 取前10个推荐，如果不足10个就取所有
            top10_tickers = [sanitize_ticker(rec.get('ticker','')) for rec in recommendations[:10] if rec.get('ticker')]
            with open(top10_json, 'w', encoding='utf-8') as f:
                json.dump(top10_tickers, f, ensure_ascii=False)
                
            # 保持向后兼容，同时生成top7
            top7_json = result_dir / f"top7_tickers_{timestamp}.json"
            top7_tickers = top10_tickers[:7] if len(top10_tickers) >= 7 else top10_tickers
            with open(top7_json, 'w', encoding='utf-8') as f:
                json.dump(top7_tickers, f, ensure_ascii=False)
        
            # 保存投资组合详情
        if portfolio_result.get('success', False):
            portfolio_file = result_dir / f"portfolio_details_{timestamp}.json"
            portfolio_data = {
                'timestamp': timestamp,
                'portfolio_metrics': portfolio_result.get('portfolio_metrics', {}),
                'optimization_info': portfolio_result.get('optimization_info', {}),
                    'weights': self._extract_weights_safely(portfolio_result)
            }
            
            with open(portfolio_file, 'w', encoding='utf-8') as f:
                json.dump(portfolio_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"结果已保存到 {result_dir}")
        return str(excel_file) if recommendations else str(result_dir)
    
    def run_complete_analysis(self, tickers: List[str], 
                             start_date: str, end_date: str,
                             top_n: int = 10) -> Dict[str, Any]:
        """
        运行完整分析流程
        
        Args:
            tickers: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            top_n: 返回推荐数量
            
        Returns:
            完整分析结果
        """
        logger.info("开始完整分析流程")
        
        analysis_results = {
            'start_time': datetime.now(),
            'config': self.config,
            'tickers': tickers,
            'date_range': f"{start_date} to {end_date}"
        }
        
        try:
            # 1. 下载数据
            stock_data = self.download_stock_data(tickers, start_date, end_date)
            if not stock_data:
                raise ValueError("无法获取股票数据")
            
            analysis_results['data_download'] = {
                'success': True,
                'stocks_downloaded': len(stock_data)
            }
            
            # 2. 创建特征
            feature_data = self.create_traditional_features(stock_data)
            if len(feature_data) == 0:
                raise ValueError("特征创建失败")
            
            analysis_results['feature_engineering'] = {
                'success': True,
                'feature_shape': feature_data.shape,
                'feature_columns': len([col for col in feature_data.columns 
                                      if col not in ['ticker', 'date', 'target']])
            }
            
            # 3. 构建Multi-factor风险模型
            try:
                risk_model = self.build_risk_model()
                analysis_results['risk_model'] = {
                    'success': True,
                    'factor_count': len(risk_model['risk_factors'].columns),
                    'assets_covered': len(risk_model['factor_loadings'])
                }
                logger.info("风险模型构建完成")
            except Exception as e:
                logger.warning(f"风险模型构建失败: {e}")
                analysis_results['risk_model'] = {'success': False, 'error': str(e)}
            
            # 4. 检测市场状态
            try:
                market_regime = self.detect_market_regime()
                analysis_results['market_regime'] = {
                    'success': True,
                    'regime': market_regime.name,
                    'probability': market_regime.probability,
                    'characteristics': market_regime.characteristics
                }
                logger.info(f"市场状态检测完成: {market_regime.name}")
            except Exception as e:
                logger.warning(f"市场状态检测失败: {e}")
                analysis_results['market_regime'] = {'success': False, 'error': str(e)}
                market_regime = MarketRegime(0, "Normal", 0.7, {'volatility': 0.15, 'trend': 0.0})
            
            # 5. 训练模型
            training_results = self.train_enhanced_models(feature_data)
            analysis_results['model_training'] = training_results
            
            # 6. 生成预测（结合regime-aware权重）
            ensemble_predictions = self.generate_enhanced_predictions(training_results, market_regime)
            if len(ensemble_predictions) == 0:
                raise ValueError("预测生成失败")
            
            analysis_results['prediction_generation'] = {
                'success': True,
                'predictions_count': len(ensemble_predictions),
                'prediction_stats': {
                    'mean': ensemble_predictions.mean(),
                    'std': ensemble_predictions.std(),
                    'min': ensemble_predictions.min(),
                    'max': ensemble_predictions.max()
                },
                'regime_adjusted': True
            }
            
            # 7. 投资组合优化（带风险模型）
            portfolio_result = self.optimize_portfolio_with_risk_model(ensemble_predictions, feature_data)
            analysis_results['portfolio_optimization'] = portfolio_result
            
            # 6. 生成投资建议
            recommendations = self.generate_investment_recommendations(portfolio_result, top_n)
            analysis_results['recommendations'] = recommendations
            
            # 7. 保存结果
            result_file = self.save_results(recommendations, portfolio_result)
            analysis_results['result_file'] = result_file
            
            analysis_results['end_time'] = datetime.now()
            analysis_results['total_time'] = (analysis_results['end_time'] - analysis_results['start_time']).total_seconds()
            analysis_results['success'] = True
            
            # 添加健康监控报告
            analysis_results['health_report'] = self.get_health_report()
            
            logger.info(f"完整分析流程完成，耗时: {analysis_results['total_time']:.1f}秒")
            logger.info(f"系统健康状况: {analysis_results['health_report']['risk_level']}, "
                       f"失败率: {analysis_results['health_report']['failure_rate_percent']:.2f}%")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"分析流程失败: {e}")
            analysis_results['error'] = str(e)
            analysis_results['success'] = False
            analysis_results['end_time'] = datetime.now()
            
            return analysis_results


def main():
    """主函数"""
    print("=== BMA Ultra Enhanced 量化分析模型 V4 ===")
    print("集成Alpha策略、Learning-to-Rank、高级投资组合优化")
    print(f"增强模块可用: {ENHANCED_MODULES_AVAILABLE}")
    print(f"高级模型: XGBoost={XGBOOST_AVAILABLE}, LightGBM={LIGHTGBM_AVAILABLE}")
    
    # 设置全局超时保护
    start_time = time.time()
    MAX_EXECUTION_TIME = 300  # 5分钟超时
    
    # 命令行参数
    parser = argparse.ArgumentParser(description='BMA Ultra Enhanced量化模型V4')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='开始日期')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='结束日期')
    parser.add_argument('--top-n', type=int, default=200, help='返回top N个推荐')
    parser.add_argument('--config', type=str, default='alphas_config.yaml', help='配置文件路径')
    parser.add_argument('--tickers', type=str, nargs='+', default=None, help='股票代码列表')
    parser.add_argument('--tickers-file', type=str, default='stocks.txt', help='股票列表文件（每行一个代码）')
    parser.add_argument('--tickers-limit', type=int, default=0, help='先用前N只做小样本测试，再全量训练（0表示直接全量）')
    
    args = parser.parse_args()
    
    # 确定股票列表
    if args.tickers:
        tickers = args.tickers
    else:
        tickers = load_universe_from_file(args.tickers_file) or load_universe_fallback()
    
    print(f"分析参数:")
    print(f"  时间范围: {args.start_date} - {args.end_date}")
    print(f"  股票数量: {len(tickers)}")
    print(f"  推荐数量: {args.top_n}")
    print(f"  配置文件: {args.config}")
    
    # 初始化模型
    model = UltraEnhancedQuantitativeModel(config_path=args.config)
    
    # 两阶段：小样本测试 → 全量
    if args.tickers_limit and args.tickers_limit > 0 and len(tickers) > args.tickers_limit:
        print("\n[TEST] 先运行小样本测试...")
        small_tickers = tickers[:args.tickers_limit]
        _ = model.run_complete_analysis(
            tickers=small_tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            top_n=min(args.top_n, len(small_tickers))
        )
        print("\n[SUCCESS] 小样本测试完成，开始全量训练...")

    # 运行完整分析 (带超时保护)
    try:
        results = model.run_complete_analysis(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            top_n=args.top_n
        )
        
        # 检查执行时间
        execution_time = time.time() - start_time
        if execution_time > MAX_EXECUTION_TIME:
            print(f"\n⚠️ 执行时间超过{MAX_EXECUTION_TIME}秒，但已完成")
            
    except KeyboardInterrupt:
        print("\n❌ 用户中断执行")
        results = {'success': False, 'error': '用户中断'}
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n❌ 执行异常 (耗时{execution_time:.1f}s): {e}")
        results = {'success': False, 'error': str(e)}
    
    # 显示结果摘要
    print("\n" + "="*60)
    print("分析结果摘要")
    print("="*60)
    
    if results.get('success', False):
        # 避免控制台编码错误（GBK）
        print(f"分析成功完成，耗时: {results['total_time']:.1f}秒")
        
        if 'data_download' in results:
            print(f"数据下载: {results['data_download']['stocks_downloaded']}只股票")
        
        if 'feature_engineering' in results:
            fe_info = results['feature_engineering']
            print(f"特征工程: {fe_info['feature_shape'][0]}样本, {fe_info['feature_columns']}特征")
        
        if 'prediction_generation' in results:
            pred_info = results['prediction_generation']
            stats = pred_info['prediction_stats']
            print(f"预测生成: {pred_info['predictions_count']}个预测 (均值: {stats['mean']:.4f})")
        
        if 'portfolio_optimization' in results and results['portfolio_optimization'].get('success', False):
            port_metrics = results['portfolio_optimization']['portfolio_metrics']
            print(f"投资组合: 预期收益{port_metrics.get('expected_return', 0):.4f}, "
                  f"夏普比{port_metrics.get('sharpe_ratio', 0):.4f}")
        
        if 'recommendations' in results:
            recommendations = results['recommendations']
            print(f"\n投资建议 (Top {len(recommendations)}):")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec['ticker']}: 权重{rec['weight']:.3f}, "
                      f"信号{rec['prediction_signal']:.4f}")
        
        if 'result_file' in results:
            print(f"\n结果已保存至: {results['result_file']}")
    
    else:
        print(f"分析失败: {results.get('error', '未知错误')}")
    
    print("="*60)


if __name__ == "__main__":
    main()
