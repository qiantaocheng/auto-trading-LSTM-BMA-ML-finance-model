    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMA增强版量化分析模型 V3
使用贝叶斯模型平均(BMA) + Shrinkage替代传统Stacking
提供更稳定、理论自洽的集成学习方案
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import argparse
import os
import tempfile
from pathlib import Path
import schedule
import time
import threading
import json
from typing import List, Optional, Dict
from scipy.stats import spearmanr, entropy
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit, KFold, GroupKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# GPU加速优化（如果可用）
try:
    import cupy as cp
    import cudf
    GPU_ACCELERATION = True
    print("[BMA GPU] GPU加速可用 (CuPy/cuDF)")
except ImportError:
    GPU_ACCELERATION = False
    print("[BMA GPU] GPU加速不可用，使用CPU计算")

# 尝试导入RAPIDS加速的sklearn
try:
    from cuml import Ridge as cuRidge
    from cuml import RandomForestRegressor as cuRandomForest
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    CUML_AVAILABLE = True
    print("[BMA GPU] RAPIDS cuML可用")
except ImportError:
    CUML_AVAILABLE = False
    print("[BMA GPU] RAPIDS cuML不可用")

# 尝试导入高级模型
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# 尝试导入TensorFlow/Keras用于CNN模型 - 已禁用
# try:
#     import tensorflow as tf
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, BatchNormalization
#     from tensorflow.keras.optimizers import Adam
#     from tensorflow.keras.callbacks import EarlyStopping
#     from tensorflow.keras.regularizers import l2
#     from sklearn.base import BaseEstimator, RegressorMixin
#     TENSORFLOW_AVAILABLE = True
#     print("[BMA CNN] TensorFlow/Keras可用，支持CNN模型")
# except ImportError:
#     TENSORFLOW_AVAILABLE = False
#     print("[BMA CNN] TensorFlow/Keras不可用，跳过CNN模型")

# 禁用CNN模型
TENSORFLOW_AVAILABLE = False
print("[BMA CNN] CNN模型已手动禁用，专注于传统机器学习模型")

# 禁用警告
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
# =============================
# 专业回测/避免泄露：带Embargo的时间序列分割
# =============================
class PurgedTimeSeriesSplit:
    """简单的时间序列分割，支持embargo（防止相邻泄露）。

    参数
    - n_splits: 折数
    - embargo: 每次切分在验证集两侧留出的样本比例（0~0.2 常用）
    """
    def __init__(self, n_splits: int = 5, embargo: float = 0.01):
        self.n_splits = max(2, int(n_splits))
        self.embargo = max(0.0, float(embargo))

    def split(self, X):
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        for i in range(self.n_splits):
            start_val = i * fold_size
            end_val = n_samples if i == self.n_splits - 1 else (i + 1) * fold_size

            # 计算embargo范围
            emb = int(self.embargo * n_samples)
            train_end = max(0, start_val - emb)
            train_start = 0
            # 训练集为 [0, train_end) 与 (end_val+emb, n)
            val_idx = np.arange(start_val, end_val)
            left_train = np.arange(train_start, train_end)
            right_start = min(n_samples, end_val + emb)
            right_train = np.arange(right_start, n_samples)
            train_idx = np.concatenate([left_train, right_train])
            yield train_idx, val_idx

# 默认股票池（从GUI的完整默认股票池继承）
= ["A", "AA", "AACB", "AACI", "AACT", "AAL", "AAMI", "AAOI", "AAON", "AAP", "AAPL", "AARD", "AAUC", "AB", "ABAT", "ABBV", "ABCB", "ABCL", "ABEO", "ABEV", "ABG", "ABL", "ABM", "ABNB", "ABSI", "ABT", "ABTS", "ABUS", "ABVC", "ABVX", "ACA", "ACAD", "ACB", "ACCO", "ACDC", "ACEL", "ACGL", "ACHC", "ACHR", "ACHV", "ACI", "ACIC", "ACIU", "ACIW", "ACLS", "ACLX", "ACM", "ACMR", "ACN", "ACNT", "ACOG", "ACRE", "ACT", "ACTG", "ACTU", "ACVA", "ACXP", "ADAG", "ADBE", "ADC", "ADCT", "ADEA", "ADI", "ADM", "ADMA", "ADNT", "ADP", "ADPT", "ADSE", "ADSK", "ADT", "ADTN", "ADUR", "ADUS", "ADVM", "AEBI", "AEE", "AEG", "AEHL", "AEHR", "AEIS", "AEM", "AEO", "AEP", "AER", "AES", "AESI", "AEVA", "AEYE", "AFCG", "AFG", "AFL", "AFRM", "AFYA", "AG", "AGCO", "AGD", "AGEN", "AGH", "AGI", "AGIO", "AGM", "AGNC", "AGO", "AGRO", "AGX", "AGYS", "AHCO", "AHH", "AHL", "AHR", "AI", "AIFF", "AIFU", "AIG", "AII", "AIM", "AIMD", "AIN", "AIOT", "AIP", "AIR", "AIRI", "AIRJ", "AIRO", "AIRS", "AISP", "AIT", "AIV", "AIZ", "AJG", "AKAM", "AKBA", "AKRO", "AL", "ALAB", "ALAR", "ALB", "ALBT", "ALC", "ALDF", "ALDX", "ALE", "ALEX", "ALF", "ALG", "ALGM", "ALGN", "ALGS", "ALGT", "ALHC", "ALIT", "ALK", "ALKS", "ALKT", "ALL", "ALLE", "ALLT", "ALLY", "ALM", "ALMS", "ALMU", "ALNT", "ALNY", "ALRM", "ALRS", "ALSN", "ALT", "ALTG", "ALTI", "ALTS", "ALUR", "ALV", "ALVO", "ALX", "ALZN", "AM", "AMAL", "AMAT", "AMBA", "AMBC", "AMBP", "AMBQ", "AMBR", "AMC", "AMCR", "AMCX", "AMD", "AME", "AMED", "AMG", "AMGN", "AMH", "AMKR", "AMLX", "AMN", "AMP", "AMPG", "AMPH", "AMPL", "AMPX", "AMPY", "AMR", "AMRC", "AMRK", "AMRN", "AMRX", "AMRZ", "AMSC", "AMSF", "AMST", "AMT", "AMTB", "AMTM", "AMTX", "AMWD", "AMWL", "AMX", "AMZE", "AMZN", "AN", "ANAB", "ANDE", "ANEB", "ANET", "ANF", "ANGH", "ANGI", "ANGO", "ANIK", "ANIP", "ANIX", "ANNX", "ANPA", "ANRO", "ANSC", "ANTA", "ANTE", "ANVS", "AOMR", "AON", "AORT", "AOS", "AOSL", "AOUT", "AP", "APA", "APAM", "APD", "APEI", "APG", "APGE", "APH", "API", "APLD", "APLE", "APLS", "APO", "APOG", "APP", "APPF", "APPN", "APPS", "APTV", "APVO", "AQN", "AQST", "AR", "ARAI", "ARCB", "ARCC", "ARCO", "ARCT", "ARDT", "ARDX", "ARE", "AREN", "ARES", "ARHS", "ARI", "ARIS", "ARKO", "ARLO", "ARLP", "ARM", "ARMK", "ARMN", "ARMP", "AROC", "ARQ", "ARQQ", "ARQT", "ARR", "ARRY", "ARTL", "ARTV", "ARVN", "ARW", "ARWR", "ARX", "AS", "ASA", "ASAN", "ASB", "ASC", "ASGN", "ASH", "ASIC", "ASIX", "ASLE", "ASM", "ASND", "ASO", "ASPI", "ASPN", "ASR", "ASST", "ASTE", "ASTH", "ASTI", "ASTL", "ASTS", "ASUR", "ASX", "ATAI", "ATAT", "ATEC", "ATEN", "ATEX", "ATGE", "ATHE", "ATHM", "ATHR", "ATI", "ATII", "ATKR", "ATLC", "ATLX", "ATMU", "ATNF", "ATO", "ATOM", "ATR", "ATRA", "ATRC", "ATRO", "ATS", "ATUS", "ATXS", "ATYR", "AU", "AUB", "AUDC", "AUGO", "AUID", "AUPH", "AUR", "AURA", "AUTL", "AVA", "AVAH", "AVAL", "AVAV", "AVB", "AVBC", "AVBP", "AVD", "AVDL", "AVDX", "AVGO", "AVIR", "AVNS", "AVNT", "AVNW", "AVO", "AVPT", "AVR", "AVT", "AVTR", "AVTX", "AVXL", "AVY", "AWI", "AWK", "AWR", "AX", "AXGN", "AXIN", "AXL", "AXP", "AXS", "AXSM", "AXTA", "AXTI", "AYI", "AYTU", "AZ", "AZN", "AZTA", "AZZ", "B", "BA", "BABA", "BAC", "BACC", "BACQ", "BAER", "BAH", "BAK", "BALL", "BALY", "BAM", "BANC", "BAND", "BANF", "BANR", "BAP", "BASE", "BATRA", "BATRK", "BAX", "BB", "BBAI", "BBAR", "BBCP", "BBD", "BBDC", "BBIO", "BBNX", "BBSI", "BBUC", "BBVA", "BBW", "BBWI", "BBY", "BC", "BCAL", "BCAX", "BCBP", "BCC", "BCE", "BCH", "BCO", "BCPC", "BCRX", "BCS", "BCSF", "BCYC", "BDC", "BDMD", "BDRX", "BDTX", "BDX", "BE", "BEAG", "BEAM", "BEEM", "BEEP", "BEKE", "BELFB", "BEN", "BEP", "BEPC", "BETR", "BF-A", "BF-B", "BFAM", "BFC", "BFH", "BFIN", "BFS", "BFST", "BG", "BGC", "BGL", "BGLC", "BGM", "BGS", "BGSF", "BHC", "BHE", "BHF", "BHFAP", "BHLB", "BHP", "BHR", "BHRB", "BHVN", "BIDU", "BIIB", "BILI", "BILL", "BIO", "BIOA", "BIOX", "BIP", "BIPC", "BIRD", "BIRK", "BJ", "BJRI", "BK", "BKD", "BKE", "BKH", "BKKT", "BKR", "BKSY", "BKTI", "BKU", "BKV", "BL", "BLBD", "BLBX", "BLCO", "BLD", "BLDE", "BLDR", "BLFS", "BLFY", "BLIV", "BLKB", "BLMN", "BLND", "BLNE", "BLRX", "BLUW", "BLX", "BLZE", "BMA", "BMBL", "BMGL", "BMHL", "BMI", "BMNR", "BMO", "BMR", "BMRA", "BMRC", "BMRN", "BMY", "BN", "BNC", "BNED", "BNGO", "BNL", "BNS", "BNTC", "BNTX", "BNZI", "BOC", "BOF", "BOH", "BOKF", "BOOM", "BOOT", "BORR", "BOSC", "BOW", "BOX", "BP", "BPOP", "BQ", "BR", "BRBR", "BRBS", "BRC", "BRDG", "BRFS", "BRK-B", "BRKL", "BRKR", "BRLS", "BRO", "BROS", "BRR", "BRSL", "BRSP", "BRX", "BRY", "BRZE", "BSAA", "BSAC", "BSBR", "BSET", "BSGM", "BSM", "BSX", "BSY", "BTAI", "BTBD", "BTBT", "BTCM", "BTCS", "BTCT", "BTDR", "BTE", "BTG", "BTI", "BTM", "BTMD", "BTSG", "BTU", "BUD", "BULL", "BUR", "BURL", "BUSE", "BV", "BVFL", "BVN", "BVS", "BWA", "BWB", "BWEN", "BWIN", "BWLP", "BWMN", "BWMX", "BWXT", "BX", "BXC", "BXP", "BY", "BYD", "BYND", "BYON", "BYRN", "BYSI", "BZ", "BZAI", "BZFD", "BZH", "BZUN", "C", "CAAP", "CABO", "CAC", "CACC", "CACI", "CADE", "CADL", "CAE", "CAEP", "CAG", "CAH", "CAI", "CAKE", "CAL", "CALC", "CALM", "CALX", "CAMT", "CANG", "CAPR", "CAR", "CARE", "CARG", "CARL", "CARR", "CARS", "CART", "CASH", "CASS", "CAT", "CATX", "CATY", "CAVA", "CB", "CBAN", "CBIO", "CBL", "CBLL", "CBNK", "CBOE", "CBRE", "CBRL", "CBSH", "CBT", "CBU", "CBZ", "CC", "CCAP", "CCB", "CCCC", "CCCS", "CCCX", "CCEP", "CCI", "CCIR", "CCIX", "CCJ", "CCK", "CCL", "CCLD", "CCNE", "CCOI", "CCRD", "CCRN", "CCS", "CCSI", "CCU", "CDE", "CDIO", "CDLR", "CDNA", "CDNS", "CDP", "CDRE", "CDRO", "CDTX", "CDW", "CDXS", "CDZI", "CE", "CECO", "CEG", "CELC", "CELH", "CELU", "CELZ", "CENT", "CENTA", "CENX", "CEP", "CEPO", "CEPT", "CEPU", "CERO", "CERT", "CEVA", "CF", "CFFN", "CFG", "CFLT", "CFR", "CG", "CGAU", "CGBD", "CGCT", "CGEM", "CGNT", "CGNX", "CGON", "CHA", "CHAC", "CHCO", "CHD", "CHDN", "CHE", "CHEF", "CHH", "CHKP", "CHMI", "CHPT", "CHRD", "CHRW", "CHT", "CHTR", "CHWY", "CHYM", "CI", "CIA", "CIB", "CIEN", "CIFR", "CIGI", "CIM", "CINF", "CING", "CINT", "CIO", "CION", "CIVB", "CIVI", "CL", "CLAR", "CLB", "CLBK", "CLBT", "CLCO", "CLDI", "CLDX", "CLF", "CLFD", "CLGN", "CLH", "CLLS", "CLMB", "CLMT", "CLNE", "CLNN", "CLOV", "CLPR", "CLPT", "CLRB", "CLRO", "CLS", "CLSK", "CLVT", "CLW", "CLX", "CM", "CMA", "CMBT", "CMC", "CMCL", "CMCO", "CMCSA", "CMDB", "CME", "CMG", "CMI", "CMP", "CMPO", "CMPR", "CMPS", "CMPX", "CMRC", "CMRE", "CMS", "CMTL", "CNA", "CNC", "CNCK", "CNDT", "CNEY", "CNH", "CNI", "CNK", "CNL", "CNM", "CNMD", "CNNE", "CNO", "CNOB", "CNP", "CNQ", "CNR", "CNS", "CNTA", "CNTB", "CNTY", "CNVS", "CNX", "CNXC", "CNXN", "COCO", "CODI", "COF", "COFS", "COGT", "COHR", "COHU", "COIN", "COKE", "COLB", "COLL", "COLM", "COMM", "COMP", "CON", "COO", "COOP", "COP", "COPL", "COR", "CORT", "CORZ", "COTY", "COUR", "COYA", "CP", "CPA", "CPAY", "CPB", "CPF", "CPIX", "CPK", "CPNG", "CPRI", "CPRT", "CPRX", "CPS", "CPSH", "CQP", "CR", "CRAI", "CRAQ", "CRBG", "CRBP", "CRC", "CRCL", "CRCT", "CRD-A", "CRDF", "CRDO", "CRE", "CRESY", "CREV", "CREX", "CRGO", "CRGX", "CRGY", "CRH", "CRI", "CRK", "CRL", "CRM", "CRMD", "CRML", "CRMT", "CRNC", "CRNX", "CRON", "CROX", "CRS", "CRSP", "CRSR", "CRTO", "CRUS", "CRVL", "CRVO", "CRVS", "CRWD", "CRWV", "CSAN", "CSCO", "CSGP", "CSGS", "CSIQ", "CSL", "CSR", "CSTL", "CSTM", "CSV", "CSW", "CSWC", "CSX", "CTAS", "CTEV", "CTGO", "CTKB", "CTLP", "CTMX", "CTNM", "CTO", "CTOS", "CTRA", "CTRI", "CTRM", "CTRN", "CTS", "CTSH", "CTVA", "CTW", "CUB", "CUBE", "CUBI", "CUK", "CUPR", "CURB", "CURI", "CURV", "CUZ", "CV", "CVAC", "CVBF", "CVCO", "CVE", "CVEO", "CVGW", "CVI", "CVLG", "CVLT", "CVM", "CVNA", "CVRX", "CVS", "CVX", "CW", "CWAN", "CWBC", "CWCO", "CWEN", "CWEN-A", "CWH", "CWK", "CWST", "CWT", "CX", "CXDO", "CXM", "CXT", "CXW", "CYBN", "CYBR", "CYCC", "CYD", "CYH", "CYN", "CYRX", "CYTK", "CZR", "CZWI", "D", "DAAQ", "DAC", "DAIC", "DAKT", "DAL", "DALN", "DAN", "DAO", "DAR", "DARE", "DASH", "DATS", "DAVA", "DAVE", "DAWN", "DAY", "DB", "DBD", "DBI", "DBRG", "DBX", "DC", "DCBO", "DCI", "DCO", "DCOM", "DCTH", "DD", "DDC", "DDI", "DDL", "DDOG", "DDS", "DEA", "DEC", "DECK", "DEFT", "DEI", "DELL", "DENN", "DEO", "DERM", "DEVS", "DFDV", "DFH", "DFIN", "DFSC", "DG", "DGICA", "DGII", "DGX", "DGXX", "DH", "DHI", "DHR", "DHT", "DHX", "DIBS", "DIN", "DINO", "DIOD", "DIS", "DJCO", "DJT", "DK", "DKL", "DKNG", "DKS", "DLB", "DLHC", "DLO", "DLTR", "DLX", "DLXY", "DMAC", "DMLP", "DMRC", "DMYY", "DNA", "DNB", "DNLI", "DNN", "DNOW", "DNTH", "DNUT", "DOC", "DOCN", "DOCS", "DOCU", "DOGZ", "DOLE", "DOMH", "DOMO", "DOOO", "DORM", "DOUG", "DOV", "DOW", "DOX", "DOYU", "DPRO", "DPZ", "DQ", "DRD", "DRDB", "DRH", "DRI", "DRS", "DRVN", "DSGN", "DSGR", "DSGX", "DSP", "DT", "DTE", "DTI", "DTIL", "DTM", "DTST", "DUK", "DUOL", "DUOT", "DV", "DVA", "DVAX", "DVN", "DVS", "DWTX", "DX", "DXC", "DXCM", "DXPE", "DXYZ", "DY", "DYN", "DYNX", "E", "EA", "EARN", "EAT", "EB", "EBAY", "EBC", "EBF", "EBMT", "EBR", "EBS", "EC", "ECC", "ECG", "ECL", "ECO", "ECOR", "ECPG", "ECVT", "ED", "EDBL", "EDIT", "EDN", "EDU", "EE", "EEFT", "EEX", "EFC", "EFSC", "EFX", "EFXT", "EG", "EGAN", "EGBN", "EGG", "EGO", "EGP", "EGY", "EH", "EHAB", "EHC", "EHTH", "EIC", "EIG", "EIX", "EKSO", "EL", "ELAN", "ELDN", "ELF", "ELMD", "ELME", "ELP", "ELPW", "ELS", "ELV", "ELVA", "ELVN", "ELWS", "EMA", "EMBC", "EMN", "EMP", "EMPD", "EMPG", "EMR", "EMX", "ENB", "ENGN", "ENGS", "ENIC", "ENOV", "ENPH", "ENR", "ENS", "ENSG", "ENTA", "ENTG", "ENVA", "ENVX", "EOG", "EOLS", "EOSE", "EPAC", "EPAM", "EPC", "EPD", "EPM", "EPR", "EPSM", "EPSN", "EQBK", "EQH", "EQNR", "EQR", "EQT", "EQV", "EQX", "ERIC", "ERIE", "ERII", "ERJ", "ERO", "ES", "ESAB", "ESE", "ESGL", "ESI", "ESLT", "ESNT", "ESOA", "ESQ", "ESTA", "ESTC", "ET", "ETD", "ETN", "ETNB", "ETON", "ETOR", "ETR", "ETSY", "EU", "EUDA", "EVAX", "EVC", "EVCM", "EVER", "EVEX", "EVGO", "EVH", "EVLV", "EVO", "EVOK", "EVR", "EVRG", "EVTC", "EVTL", "EW", "EWBC", "EWCZ", "EWTX", "EXAS", "EXC", "EXE", "EXEL", "EXK", "EXLS", "EXOD", "EXP", "EXPD", "EXPE", "EXPI", "EXPO", "EXR", "EXTR", "EYE", "EYPT", "EZPW", "F", "FA",
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
    "STRM", "STRT", "STRZ", "STSS", "STT", "STVN", "STX", "STXS", "STZ", "SU", "SUI", "SUN", "SUPN", "SUPV", "SUPX", "SURG", "SUZ", "SVCO", "SVM", "SVRA", "SVV", "SW", "SWBI", "SWIM", "SWIN", "SWK", "SWKS", "SWX", "SXC", "SXI", "SXT", "SY", "SYBT", "SYF", "SYK", "SYM", "SYNA", "SYRE", "SYTA", "SYY", "SZZL", "T", "TAC", "TACH", "TACO", "TAK", "TAL", "TALK", "TALO", "TAOX", "TAP", "TARA", "TARS", "TASK", "TATT", "TBB", "TBBB", "TBBK", "TBCH", "TBI", "TBLA", "TBPH", "TBRG", "TCBI", "TCBK", "TCBX", "TCMD", "TCOM", "TCPC", "TD", "TDC", "TDIC", "TDOC", "TDS", "TDUP", "TDW", "TEAM", "TECH", "TECK", "TECX", "TEF", "TEL", "TEM", "TEN", "TENB", "TEO", "TER", "TERN", "TEVA", "TEX", "TFC", "TFII", "TFIN", "TFPM", "TFSL", "TFX", "TG", "TGB", "TGE", "TGEN", "TGLS", "TGNA", "TGS", "TGT", "TGTX", "TH", "THC", "THFF", "THG", "THO", "THR", "THRM", "THRY", "THS", "THTX", "TIC", "TIGO", "TIGR", "TIL", "TILE", "TIMB", "TIPT", "TITN", "TIXT", "TJX", "TK", "TKC", "TKNO", "TKO", "TKR", "TLK", "TLN", "TLS", "TLSA", "TLSI", "TM", "TMC", "TMCI", "TMDX", "TME", "TMHC", "TMO", "TMUS", "TNC", "TNDM", "TNET", "TNGX", "TNK", "TNL", "TNXP", "TOI", "TOL", "TOPS", "TORO", "TOST", "TOWN", "TPB", "TPC", "TPCS", "TPG", "TPH", "TPR", "TPST", "TPVG", "TR", "TRAK", "TRC", "TRDA", "TREE", "TREX", "TRGP", "TRI", "TRIN", "TRIP", "TRMB", "TRMD", "TRML", "TRN", "TRNO", "TRNR", "TRNS", "TRON", "TROW", "TROX", "TRP", "TRS", "TRU", "TRUE", "TRUG", "TRUP", "TRV", "TRVG", "TRVI", "TS", "TSAT", "TSCO", "TSE", "TSEM", "TSHA", "TSLA", "TSLX", "TSM", "TSN", "TSQ", "TSSI", "TT", "TTAM", "TTAN", "TTC", "TTD", "TTE", "TTEC", "TTEK", "TTGT", "TTI", "TTMI", "TTSH", "TTWO", "TU", "TUSK", "TUYA", "TV", "TVA", "TVAI", "TVRD", "TVTX", "TW", "TWFG", "TWI", "TWIN", "TWLO", "TWNP", "TWO", "TWST", "TX", "TXG", "TXN", "TXNM", "TXO", "TXRH", "TXT", "TYG", "TYRA", "TZOO", "TZUP", "U", "UA", "UAA", "UAL", "UAMY", "UAVS", "UBER", "UBFO", "UBS", "UBSI", "UCAR", "UCB", "UCL", "UCTT", "UDMY", "UDR", "UE", "UEC", "UFCS", "UFG", "UFPI", "UFPT", "UGI", "UGP", "UHAL", "UHAL-B", "UHG", "UHS", "UI", "UIS", "UL", "ULBI", "ULCC", "ULS", "ULY", "UMAC", "UMBF", "UMC", "UMH", "UNCY", "UNF", "UNFI", "UNH", "UNIT", "UNM", "UNP", "UNTY", "UPB", "UPBD", "UPS", "UPST", "UPWK", "UPXI", "URBN", "URGN", "UROY", "USAC", "USAR", "USAU", "USB", "USFD", "USLM", "USM", "USNA", "USPH", "UTHR", "UTI", "UTL", "UTZ", "UUUU", "UVE", "UVSP", "UVV", "UWMC", "UXIN", "V", "VAC", "VAL", "VALE", "VBIX", "VBNK", "VBTX", "VC", "VCEL", "VCTR", "VCYT", "VECO", "VEEV", "VEL", "VENU", "VEON", "VERA", "VERB", "VERI", "VERX", "VET", "VFC", "VFS", "VG", "VIAV", "VICI", "VICR", "VIK", "VINP", "VIOT", "VIPS", "VIR", "VIRC", "VIRT", "VIST", "VITL", "VIV", "VKTX", "VLGEA", "VLN", "VLO", "VLRS", "VLTO", "VLY", "VMC", "VMD", "VMEO", "VMI", "VNDA", "VNET", "VNOM", "VNT", "VNTG", "VOD", "VOR", "VOXR", "VOYA", "VOYG", "VPG", "VRDN", "VRE", "VREX", "V"WING", "WIT", "WIX", "WK", "WKC", "WKEY", "WKSP", "WLDN", "WLFC", "WLK", "WLY", "WM", "WMB", "WMG", "WMK", "WMS", "WMT", "WNC", "WNEB", "WNS", "WOOF", "WOR", "WOW", "WPC", "WPM", "WPP", "WRB", "WRBY", "WRD", "WRLD", "WS", "WSBC", "WSC", "WSFS", "WSM", "WSO", "WSR", "WST", "WT", "WTF", "WTG", "WTRG", "WTS", "WTTR", "WTW", "WU", "WULF", "WVE", "WW", "WWD", "WWW", "WXM", "WY", "WYFI", "WYNN", "WYY", "XAIR", "XBIT", "XCUR", "XEL", "XENE", "XERS", "XGN", "XHR", "XIFR", "XMTR", "XNCR", "XNET", "XOM", "XOMA", "XP", "XPEL", "XPER", "XPEV", "XPO", "XPOF", "XPRO", "XRAY", "XRX", "XTKG", "XYF", "XYL", "XYZ", "YALA", "YB", "YELP", "YETI", "YEXT", "YMAB", "YMAT", "YMM", "YORK", "YORW", "YOU", "YPF", "YRD", "YSG", "YSXT", "YUM", "YUMC", "YYAI", "YYGH", "Z", "ZBAI", "ZBH", "ZBIO", "ZBRA", "ZD", "ZDGE", "ZENA", "ZEO", "ZEPP", "ZETA", "ZEUS", "ZG", "ZGN", "ZH", "ZIM", "ZIMV", "ZION", "ZIP", "ZJK", "ZK", "ZLAB", "ZM", "ZONE", "ZS", "ZSPC", "ZTO", "ZTS", "ZUMZ", "ZVIA", "ZVRA", "ZWS", "ZYBT", "ZYME"]
]


# 去重处理
ticker_list = list(dict.fromkeys(ticker_list))


# CNN模型已禁用 - 注释掉CNN类定义
# class CNNRegressor(BaseEstimator, RegressorMixin):
#     """CNN回归器，与scikit-learn兼容的包装器"""
#     
#     def __init__(self, input_dim=None, sequence_length=20, epochs=50, batch_size=32, 
#                  learning_rate=0.001, dropout_rate=0.3, verbose=0):
#         self.input_dim = input_dim
#         self.sequence_length = sequence_length
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.dropout_rate = dropout_rate
#         self.verbose = verbose
#         self.model = None
#         self.scaler = StandardScaler()
# CNN模型方法已禁用
#         
#     def _create_cnn_model(self, input_shape):
#         """创建CNN模型"""
#         if not TENSORFLOW_AVAILABLE:
#             raise ImportError("TensorFlow不可用，无法创建CNN模型")
#             
#         model = Sequential([
#             # 第一层卷积
#             Conv1D(filters=64, kernel_size=3, activation='relu', 
#                    input_shape=input_shape, padding='same'),
#             BatchNormalization(),
#             MaxPooling1D(pool_size=2),
#             Dropout(self.dropout_rate),
#             
#             # 第二层卷积
#             Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
#             BatchNormalization(),
#             MaxPooling1D(pool_size=2),
#             Dropout(self.dropout_rate),
#             
#             # 第三层卷积
#             Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'),
#             BatchNormalization(),
#             
#             # 展平并全连接
#             Flatten(),
#             Dense(100, activation='relu', kernel_regularizer=l2(0.001)),
#             Dropout(self.dropout_rate),
#             Dense(50, activation='relu', kernel_regularizer=l2(0.001)),
#             Dropout(self.dropout_rate),
#             Dense(1, activation='linear')  # 回归输出
#         ])
#         
#         model.compile(
#             optimizer=Adam(learning_rate=self.learning_rate),
#             loss='mse',
#             metrics=['mae']
#         )
#         
#         return model
#     
#     def _prepare_sequences(self, X):
#         """将特征数据转换为时序数据"""
#         if len(X.shape) == 2:
#             n_samples, n_features = X.shape
#             
#             # 如果样本数量少于序列长度，重复数据
#             if n_samples < self.sequence_length:
#                 X_repeated = np.tile(X, (self.sequence_length // n_samples + 1, 1))
#                 X = X_repeated[:self.sequence_length]
#                 n_samples = self.sequence_length
#             
#             # 创建滑动窗口序列
#             sequences = []
#             for i in range(n_samples - self.sequence_length + 1):
#                 sequences.append(X[i:i + self.sequence_length])
#             
#             if not sequences:
#                 # 如果无法创建序列，则使用重复的单个样本
#                 sequences = [np.tile(X[-1], (self.sequence_length, 1))]
#             
#             return np.array(sequences)
#         else:
#             return X
#     
#     def fit(self, X, y):
#         """训练CNN模型"""
#         if not TENSORFLOW_AVAILABLE:
#             print("[CNN] TensorFlow不可用，跳过CNN训练")
#             return self
#             
#         # 标准化特征
#         X_scaled = self.scaler.fit_transform(X)
#         
#         # 准备序列数据
#         X_seq = self._prepare_sequences(X_scaled)
#         
#         # 调整y的长度以匹配序列数据
#         if len(y) > len(X_seq):
#             y = y[-len(X_seq):]
#         elif len(y) < len(X_seq):
#             y = np.concatenate([y, np.repeat(y[-1], len(X_seq) - len(y))])
#         
#         # 确定输入维度
#         if self.input_dim is None:
#             self.input_dim = X_seq.shape[-1]
#         
#         # 创建模型
#         input_shape = (self.sequence_length, self.input_dim)
#         self.model = self._create_cnn_model(input_shape)
#         
#         # 训练模型
#         early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
#         
#         try:
#             self.model.fit(
#                 X_seq, y,
#                 epochs=self.epochs,
#                 batch_size=self.batch_size,
#                 verbose=self.verbose,
#                 callbacks=[early_stopping],
#                 validation_split=0.2
#             )
#         except Exception as e:
#             print(f"[CNN] 训练失败: {e}")
#             # 创建一个简单的模型作为备用
#             self.model = None
#         
#         return self
#     
#     def predict(self, X):
#         """CNN预测"""
#         if not TENSORFLOW_AVAILABLE or self.model is None:
#             # 返回零预测作为备用
#             return np.zeros(len(X))
#         
#         try:
#             # 标准化特征
#             X_scaled = self.scaler.transform(X)
#             
#             # 准备序列数据
#             X_seq = self._prepare_sequences(X_scaled)
#             
#             # 预测
#             predictions = self.model.predict(X_seq, verbose=0)
#             
#             # 如果序列数量与原始样本数量不匹配，调整输出
#             if len(predictions) != len(X):
#                 if len(predictions) < len(X):
#                     # 如果预测数量不足，重复最后一个预测
#                     last_pred = predictions[-1] if len(predictions) > 0 else 0
#                     predictions = np.concatenate([
#                         predictions.flatten(),
#                         np.repeat(last_pred, len(X) - len(predictions))
#                     ])
#                 else:
#                     # 如果预测数量过多，截取
#                     predictions = predictions[:len(X)]
#             
#             return predictions.flatten()
#             
#         except Exception as e:
#             print(f"[CNN] 预测失败: {e}")
#             return np.zeros(len(X))


class ICFactorSelector(BaseEstimator, TransformerMixin):
    """基于IC的因子选择器（保持兼容性）"""
    
    def __init__(self, ic_threshold=0.01):
        self.ic_threshold = ic_threshold
        self.selected_features_ = None
        self.feature_names_ = None
        
    def fit(self, X, y):
        selected = []
        
        if isinstance(X, np.ndarray):
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            self.feature_names_ = feature_names
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
            self.feature_names_ = X_df.columns.tolist()
        
        for i, feature_name in enumerate(X_df.columns):
            try:
                factor_values = X_df.iloc[:, i]
                ic = self._calculate_ic_safe(factor_values, y)
                
                if abs(ic) > self.ic_threshold:
                    selected.append(feature_name)
                    
            except Exception:
                continue
        
        self.selected_features_ = selected
        print(f"[IC SELECTOR] 从 {len(X_df.columns)} 个因子中选择了 {len(selected)} 个")
        
        return self
    
    def transform(self, X):
        if self.selected_features_ is None:
            raise ValueError("Must fit before transform")
        
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names_)
        else:
            X_df = X.copy()
        
        available_features = [f for f in self.selected_features_ if f in X_df.columns]
        
        # 如果没有选择到特征，则至少保留3个原始特征避免维度不匹配
        if not available_features:
            print(f"[IC SELECTOR WARNING] 没有选择到有效特征，使用前3个原始特征")
            return X_df.iloc[:, :min(3, X_df.shape[1])].values
        
        result = X_df[available_features].values
        print(f"[IC SELECTOR] 转换后特征数: {result.shape[1]}")
        return result
    
    def _calculate_ic_safe(self, factor_data, returns):
        try:
            factor_array = np.array(factor_data).flatten()
            returns_array = np.array(returns).flatten()
            
            min_len = min(len(factor_array), len(returns_array))
            factor_array = factor_array[:min_len]
            returns_array = returns_array[:min_len]
            
            mask = ~(np.isnan(factor_array) | np.isnan(returns_array))
            clean_factor = factor_array[mask]
            clean_returns = returns_array[mask]
            
            if len(clean_factor) < 10:
                return 0.0
                
            if np.std(clean_factor) == 0 or np.std(clean_returns) == 0:
                return 0.0
            
            ic, _ = spearmanr(clean_factor, clean_returns)
            return ic if not np.isnan(ic) else 0.0
            
        except Exception:
            return 0.0


class BayesianModelAveraging:
    """
    贝叶斯模型平均 + Shrinkage
    替代原有的Stacking方法
    """
    
    def __init__(self, alpha_prior=1.5, shrinkage_factor=0.15, min_weight_threshold=0.02,
                 model_class_priors: Optional[Dict[str, float]] = None,
                 enable_meta_learner: bool = True,
                 meta_learner_type: str = 'ridge'):
        self.alpha_prior = alpha_prior
        self.shrinkage_factor = shrinkage_factor
        self.min_weight_threshold = min_weight_threshold
        # 先验：突出 LightGBM / XGBoost，弱化传统线性/随机森林/猫
        self.model_class_priors = model_class_priors or {
            'LightGBM': 0.35,
            'XGBoost': 0.35,
            'Ridge': 0.10,
            'ElasticNet': 0.10,
            'CatBoost': 0.05,
        }
        self.models = {}
        self.posterior_weights = {}
        self.model_likelihoods = {}
        self.training_history = []
        # 二层融合（Ridge/ElasticNet）
        self.enable_meta_learner = enable_meta_learner
        self.meta_learner_type = meta_learner_type
        self.meta_model = None
        self.meta_feature_names: List[str] = []
        
    def fit(self, X, y, models_dict):
        """训练BMA ensemble"""
        print(f"[BMA] 开始训练贝叶斯模型平均ensemble...")
        print(f"[BMA] 模型数量: {len(models_dict)}")
        print(f"[BMA] 先验参数α: {self.alpha_prior}")
        print(f"[BMA] 收缩因子: {self.shrinkage_factor}")
        
        self.models = models_dict
        n_models = len(models_dict)
        
        # 时序交叉验证计算似然（加入embargo，防泄露）
        tscv = PurgedTimeSeriesSplit(n_splits=5, embargo=0.01)
        model_scores = {}
        
        for name, model in models_dict.items():
            fold_likelihoods = []
            fold_r2_scores = []
            
            try:
                for train_idx, val_idx in tscv.split(X):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                    
                    model.fit(X_train_fold, y_train_fold)
                    y_pred_fold = model.predict(X_val_fold)
                    
                    mse = mean_squared_error(y_val_fold, y_pred_fold)
                    r2 = r2_score(y_val_fold, y_pred_fold)
                    
                    # 对数似然（假设高斯噪声）；可加入时间衰减权重
                    if mse > 1e-10:
                        base_ll = -0.5 * len(y_val_fold) * np.log(2 * np.pi * mse) - 0.5 * len(y_val_fold)
                    else:
                        base_ll = 1000
                    # 最近数据更高权重（线性递增）
                    recency = np.linspace(0.5, 1.5, len(y_val_fold)).mean()
                    likelihood = base_ll * recency
                    
                    fold_likelihoods.append(likelihood)
                    fold_r2_scores.append(r2)
                
                avg_likelihood = np.mean(fold_likelihoods)
                avg_r2 = np.mean(fold_r2_scores)
                
                model_scores[name] = {
                    'likelihood': avg_likelihood,
                    'r2': avg_r2,
                    'std_r2': np.std(fold_r2_scores)
                }
                
                print(f"[BMA] {name}: 似然={avg_likelihood:.2f}, R²={avg_r2:.4f}±{np.std(fold_r2_scores):.4f}")
                
            except Exception as e:
                print(f"[BMA ERROR] {name} CV失败: {e}")
                model_scores[name] = {
                    'likelihood': -np.inf,
                    'r2': -np.inf,
                    'std_r2': np.inf
                }
        
        # 保存似然分数
        self.model_likelihoods = model_scores
        
        # 计算后验权重
        self._calculate_posterior_weights()
        
        # 在全部数据上重新训练所有模型
        print(f"[BMA] 在全部数据上重新训练模型...")
        for name, model in self.models.items():
            try:
                model.fit(X, y)
                print(f"[BMA] {name} 最终训练完成")
            except Exception as e:
                print(f"[BMA ERROR] {name} 最终训练失败: {e}")
        
        # 训练二层融合模型（使用OOF预测）
        if self.enable_meta_learner:
            try:
                self._train_meta_learner(X, y)
                print(f"[BMA META] 二层融合模型训练完成: {self.meta_learner_type}")
            except Exception as e:
                print(f"[BMA META ERROR] 二层融合训练失败: {e}")

        print(f"[BMA] 训练完成！")
        return self
    
    def _calculate_posterior_weights(self):
        """计算后验权重"""
        model_names = list(self.models.keys())
        n_models = len(model_names)
        
        # 提取似然值和R²
        likelihoods = np.array([self.model_likelihoods[name]['likelihood'] for name in model_names])
        r2_scores = np.array([self.model_likelihoods[name]['r2'] for name in model_names])
        
        # 处理无效值
        finite_mask = np.isfinite(likelihoods) & np.isfinite(r2_scores)
        if not finite_mask.any():
            print(f"[BMA WARNING] 所有模型无效，使用均等权重")
            weights = np.ones(n_models) / n_models
        else:
            # 过滤无效模型
            valid_likelihoods = likelihoods[finite_mask]
            valid_r2 = r2_scores[finite_mask]
            
            # 综合评分：似然 + R²表现 + 最近期表现强化
            combined_scores = 0.6 * valid_likelihoods + 0.35 * valid_r2 * 1000
            
            # 数值稳定的softmax
            exp_scores = np.exp(combined_scores - combined_scores.max())
            raw_weights_valid = exp_scores / exp_scores.sum()
            
            # 恢复到完整权重数组
            raw_weights = np.zeros(n_models)
            raw_weights[finite_mask] = raw_weights_valid
            
            # 先验权重：强调 LightGBM/XGBoost，其它均匀分布
            simplicity_bias = np.ones(n_models) / n_models
            prior_vec = np.array([
                self.model_class_priors.get(name, simplicity_bias[i])
                for i, name in enumerate(model_names)
            ], dtype=float)
            if prior_vec.sum() <= 0:
                prior_vec = simplicity_bias
            prior_vec = prior_vec / prior_vec.sum()
            
            # 后验权重 = 收缩版本
            weights = ((1 - self.shrinkage_factor) * raw_weights +
                       self.shrinkage_factor * prior_vec)
        
        # 最小权重阈值
        weights = np.maximum(weights, self.min_weight_threshold)
        weights = weights / weights.sum()
        
        self.posterior_weights = dict(zip(model_names, weights))
        
        print(f"[BMA WEIGHTS] 后验权重分配:")
        for name, weight in self.posterior_weights.items():
            print(f"  {name}: {weight:.4f}")
        
        # 权重熵（多样性指标）
        entropy_val = entropy(list(self.posterior_weights.values()))
        max_entropy = np.log(n_models)
        normalized_entropy = entropy_val / max_entropy
        print(f"[BMA] 权重熵: {entropy_val:.4f} (归一化: {normalized_entropy:.4f})")
        
        return self.posterior_weights
    
    def predict(self, X):
        """BMA预测 - 优化版本，跳过有问题的模型"""
        if not self.posterior_weights:
            raise ValueError("模型未训练，请先调用fit()")
        
        predictions = {}
        valid_weights = {}
        total_failed_weight = 0.0
        
        # 获取每个模型的预测
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
                valid_weights[name] = self.posterior_weights[name]
            except Exception as e:
                print(f"[BMA PREDICT ERROR] {name}: {e}")
                total_failed_weight += self.posterior_weights[name]
                # 跳过失败的模型
        
        # 如果没有任何模型成功，返回零预测
        if not predictions:
            print("[BMA WARNING] 所有模型预测失败，返回零预测")
            return np.zeros(len(X))
        
        # 重新归一化有效模型的权重
        if total_failed_weight > 0:
            remaining_weight = 1.0 - total_failed_weight
            if remaining_weight > 0:
                for name in valid_weights:
                    valid_weights[name] = valid_weights[name] / remaining_weight
            else:
                # 如果失败权重太大，平均分配
                uniform_weight = 1.0 / len(valid_weights)
                for name in valid_weights:
                    valid_weights[name] = uniform_weight
        
        # 加权平均（作为基础特征）
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            weight = valid_weights[name]
            ensemble_pred += weight * pred

        # 如果有二层融合，使用元学习器进行最终预测
        if self.enable_meta_learner and self.meta_model is not None:
            # 构造特征矩阵：优先使用 XGBoost/LightGBM，再附加 BMA 加权结果
            feature_cols = []
            # 选择用于融合的模型
            preferred = [m for m in ["XGBoost", "LightGBM"] if m in predictions]
            used_names = preferred if preferred else list(predictions.keys())
            for name in used_names:
                feature_cols.append(predictions[name].reshape(-1, 1))
            feature_cols.append(ensemble_pred.reshape(-1, 1))  # 加入BMA加权
            meta_X = np.hstack(feature_cols)
            return self.meta_model.predict(meta_X)
        
        return ensemble_pred

    def _train_meta_learner(self, X, y):
        """使用时序OOF预测训练二层融合模型（Ridge/ElasticNet）"""
        model_names = list(self.models.keys())
        # 仅用核心GBDT模型作为特征，若不可用则退回全部
        core = [m for m in ["XGBoost", "LightGBM"] if m in self.models]
        selected = core if core else model_names
        self.meta_feature_names = selected + ["BMA_weighted"]

        tscv = TimeSeriesSplit(n_splits=5)
        oof_pred_matrix = np.full((len(y), len(selected)), np.nan, dtype=float)
        # 为计算 BMA 加权的OOF，需要存储每个模型的OOF
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            for j, name in enumerate(selected):
                base_model = self.models[name]
                try:
                    m = clone(base_model)
                    m.fit(X_train, y_train)
                    oof_pred_matrix[val_idx, j] = m.predict(X_val)
                except Exception as e:
                    print(f"[BMA META OOF] {name} 第{fold_idx+1}折失败: {e}")
                    # 留下 NaN，后续填补

        # 将 NaN 用列均值填补
        col_means = np.nanmean(oof_pred_matrix, axis=0)
        inds = np.where(np.isnan(oof_pred_matrix))
        oof_pred_matrix[inds] = np.take(col_means, inds[1])

        # 计算OOF层面的BMA加权作为额外特征
        # 对齐选择的模型权重
        weights = np.array([self.posterior_weights.get(name, 0.0) for name in selected], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights) / max(1, len(weights))
        else:
            weights = weights / weights.sum()
        bma_oof = (oof_pred_matrix * weights.reshape(1, -1)).sum(axis=1).reshape(-1, 1)
        meta_X = np.hstack([oof_pred_matrix, bma_oof])

        # 训练元学习器
        if self.meta_learner_type == 'elasticnet':
            self.meta_model = ElasticNet(alpha=0.05, l1_ratio=0.2)
        else:
            self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(meta_X, y)


class QuantitativeModel:
    """增强版量化分析模型 - 使用BMA替代Stacking"""
    
    def __init__(self, progress_callback=None):
        self.bma_model = None
        self.pipelines = {}
        self.factor_ic_scores = {}
        self.model_scores = {}
        self.training_feature_columns = []
        self.progress_callback = progress_callback  # 添加进度回调函数
        self.prediction_horizon_days = None  # 当前预测周期（天）
    
    def download_data(self, tickers, start_date, end_date):
        """下载股票数据（保持原有接口）"""
        print(f"[DATA] 下载 {len(tickers)} 只股票的数据...")
        print(f"[PROGRESS] 开始时间: {datetime.now().strftime('%H:%M:%S')}")
        
        # 进度监控初始化
        if self.progress_callback:
            self.progress_callback.set_stage("数据下载", len(tickers))
        
        data = {}
        success_count = 0
        failed_count = 0
        
        for i, ticker in enumerate(tickers, 1):
            try:
                # 更新进度监控
                if self.progress_callback:
                    self.progress_callback.update_progress(f"正在下载 {ticker}", i-1)
                
                print(f"[{i:3d}/{len(tickers):3d}] 下载 {ticker:6s}...", end=" ")
                stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                
                if len(stock_data) > 0:
                    if isinstance(stock_data.columns, pd.MultiIndex):
                        stock_data.columns = stock_data.columns.droplevel(1)
                    data[ticker] = stock_data
                    success_count += 1
                    print(f"成功 {len(stock_data)} 天数据")
                else:
                    failed_count += 1
                    print(f"❌ 无数据")
                    
            except Exception as e:
                failed_count += 1
                print(f"❌ 失败: {str(e)[:30]}...")
                continue
        
        # 最终进度更新
        if self.progress_callback:
            self.progress_callback.update_progress("数据下载完成", len(tickers))
        
        print(f"[PROGRESS] 结束时间: {datetime.now().strftime('%H:%M:%S')}")
        print(f"[SUMMARY] 成功: {success_count}, 失败: {failed_count}, 总计: {len(tickers)}")
        
        if data:
            print(f"[SUCCESS] 成功下载 {len(data)} 只股票数据")
        else:
            print(f"[ERROR] 没有成功下载任何数据")
        
        return data
    
    def calculate_technical_indicators(self, data):
        """计算技术指标（周度预测增强：加入更快的均线/交叉信号，RSI缩短窗口）"""
        indicators = {}
        
        # 移动平均线
        indicators['sma_5'] = data['Close'].rolling(window=5).mean()
        indicators['sma_10'] = data['Close'].rolling(window=10).mean()
        indicators['sma_20'] = data['Close'].rolling(window=20).mean()
        indicators['sma_50'] = data['Close'].rolling(window=50).mean()
        
        # 指数移动平均（新增更快的EMA及交叉差）
        indicators['ema_5'] = data['Close'].ewm(span=5).mean()
        indicators['ema_10'] = data['Close'].ewm(span=10).mean()
        indicators['ema_12'] = data['Close'].ewm(span=12).mean()
        indicators['ema_26'] = data['Close'].ewm(span=26).mean()
        indicators['ema_cross_5_10'] = indicators['ema_5'] - indicators['ema_10']
        
        # MACD
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # RSI（缩短窗口提升对未来5日的敏感度）
        indicators['rsi'] = self.calculate_rsi(data['Close'], period=10)

        # 随机指标 Stochastic (5,3)
        highest_high_5 = data['High'].rolling(window=5).max()
        lowest_low_5 = data['Low'].rolling(window=5).min()
        stoch_k = (data['Close'] - lowest_low_5) / (highest_high_5 - lowest_low_5 + 1e-9) * 100.0
        indicators['stoch_k_5_3'] = stoch_k.rolling(window=3).mean()
        indicators['stoch_d_5_3'] = indicators['stoch_k_5_3'].rolling(window=3).mean()
        
        # 布林带
        bb_middle = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        indicators['bollinger_upper'] = bb_middle + (bb_std * 2)
        indicators['bollinger_lower'] = bb_middle - (bb_std * 2)
        indicators['bollinger_width'] = indicators['bollinger_upper'] - indicators['bollinger_lower']
        indicators['bollinger_position'] = (data['Close'] - indicators['bollinger_lower']) / indicators['bollinger_width']
        
        return indicators
    
    def calculate_rsi(self, prices, period=10):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_cci(self, data, period=20):
        """计算CCI (Commodity Channel Index)"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad_tp = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=False)
        return (typical_price - sma_tp) / (0.015 * mad_tp)
    
    def calculate_factors(self, data):
        """计算所有因子（周度预测：加强5-10日窗口）"""
        factors = {}
        
        # 基础技术指标
        tech_indicators = self.calculate_technical_indicators(data)
        factors.update(tech_indicators)
        
        # 价格因子 - 强化 5/10 日
        factors['price_momentum_5'] = data['Close'].pct_change(5)
        factors['price_momentum_10'] = data['Close'].pct_change(10)
        factors['price_mom_diff_5_10'] = factors['price_momentum_5'] - factors['price_momentum_10']
        
        # 价格加速度和反转（5 vs 10）
        factors['price_acceleration'] = data['Close'].pct_change(5) - data['Close'].pct_change(10)
        factors['price_reversal'] = -data['Close'].pct_change(2)
        
        # 成交量因子 - 使用10天聚合
        factors['volume_sma_10'] = data['Volume'].rolling(window=10).mean()
        factors['volume_ratio'] = data['Volume'] / factors['volume_sma_10']
        factors['volume_momentum'] = data['Volume'].pct_change(10)
        
        # 价格-成交量因子 - 使用10天聚合
        factors['price_volume'] = data['Close'] * data['Volume']
        factors['money_flow'] = (data['Close'] * data['Volume']).rolling(10).mean() / data['Close'].rolling(10).mean()
        
        # 波动率因子 - 使用10天聚合
        factors['volatility_10'] = data['Close'].pct_change().rolling(window=10).std()

        # ATR(5) 真实波动幅度
        tr1 = data['High'] - data['Low']
        tr2 = (data['High'] - data['Close'].shift(1)).abs()
        tr3 = (data['Low'] - data['Close'].shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        factors['atr_5'] = true_range.rolling(window=5).mean()
        
        # 相对强弱因子 - 使用10天聚合
        factors['rs_vs_sma'] = data['Close'] / data['Close'].rolling(10).mean() - 1
        factors['high_low_ratio'] = data['High'] / data['Low'] - 1
        factors['close_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        factors['rolling_max_5'] = data['Close'].rolling(window=5).max()
        factors['rolling_min_5'] = data['Close'].rolling(window=5).min()
        factors['breakout_5'] = data['Close'] / factors['rolling_max_5'] - 1
        factors['pullback_5'] = data['Close'] / factors['rolling_min_5'] - 1

        # 5日收益z分数（短期均值回归/过热程度）
        daily_ret = data['Close'].pct_change()
        mean_5 = daily_ret.rolling(window=5).mean()
        std_5 = daily_ret.rolling(window=5).std()
        factors['ret_5_zscore'] = (mean_5 / (std_5 + 1e-9))
        
        # 市场情绪因子
        factors['gap_ratio'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        factors['intraday_return'] = (data['Close'] - data['Open']) / data['Open']
        factors['overnight_return'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        
        # 高级因子 - 使用10天聚合
        factors['rolling_sharpe'] = (data['Close'].pct_change().rolling(10).mean() / 
                                   data['Close'].pct_change().rolling(10).std())
        
        # CCI (Commodity Channel Index) - 商品通道指数
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp_10 = typical_price.rolling(window=10).mean()
        mad_tp_10 = typical_price.rolling(window=10).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=False)
        factors['cci_10'] = (typical_price - sma_tp_10) / (0.015 * mad_tp_10)
        
        # CCI衍生因子
        factors['cci_20'] = self._calculate_cci(data, period=20)
        factors['cci_normalized'] = factors['cci_10'] / 100.0  # 标准化CCI
        factors['cci_momentum'] = factors['cci_10'].diff(5)  # CCI动量
        factors['cci_cross_signal'] = np.where(
            (factors['cci_10'] > 100) & (factors['cci_10'].shift(1) <= 100), 1,
            np.where((factors['cci_10'] < -100) & (factors['cci_10'].shift(1) >= -100), -1, 0)
        )  # CCI交叉信号
        
        # 转换为DataFrame并清理
        factors_df = pd.DataFrame(factors, index=data.index)
        factors_df = factors_df.replace([np.inf, -np.inf], np.nan)
        
        return factors_df
    
    def calculate_ic(self, factor_data, returns):
        """计算IC"""
        try:
            factor_array = np.array(factor_data).flatten()
            returns_array = np.array(returns).flatten()
            
            min_len = min(len(factor_array), len(returns_array))
            factor_array = factor_array[:min_len]
            returns_array = returns_array[:min_len]
            
            mask = ~(np.isnan(factor_array) | np.isnan(returns_array))
            clean_factor = factor_array[mask]
            clean_returns = returns_array[mask]
            
            if len(clean_factor) < 10:
                return 0.0
                
            if np.std(clean_factor) == 0 or np.std(clean_returns) == 0:
                return 0.0
            
            ic, _ = spearmanr(clean_factor, clean_returns)
            return ic if not np.isnan(ic) else 0.0
            
        except Exception:
            return 0.0
    
    def prepare_ml_data_with_time_series(self, all_data, target_period=5):
        """准备时序安全的机器学习数据（默认预测未来一周/5个交易日的收益）"""
        print(f"[TIME-SERIES PREP] 准备时序安全的机器学习数据...")
        # 记录当前预测周期（用于后续动态特征筛选等参数）
        try:
            self.prediction_horizon_days = int(target_period)
        except Exception:
            self.prediction_horizon_days = 5
        
        all_factor_data = []
        
        print(f"[GLOBAL FACTORS] 计算全局因子池...")
        
        for ticker, data in all_data.items():
            try:
                print(f"[GLOBAL FACTORS] 处理 {ticker}...")
                
                # 计算因子
                factors = self.calculate_factors(data)
                # 计算目标变量(未来一周/5个交易日收益率)
                target = data['Close'].pct_change(target_period).shift(-target_period)
                
                # 对齐数据
                aligned_data = pd.concat([factors, target.rename('target')], axis=1)
                aligned_data = aligned_data.dropna()
                
                if len(aligned_data) == 0:
                    print(f"[WARNING] {ticker} 没有有效的对齐数据")
                    continue
                
                # 添加股票和日期信息
                aligned_data['ticker'] = ticker
                aligned_data['date'] = aligned_data.index
                aligned_data = aligned_data.reset_index(drop=True)
                
                all_factor_data.append(aligned_data)
                
                print(f"[FACTORS] 计算了 {len(factors.columns)} 个有效因子")
                
            except Exception as e:
                print(f"[GLOBAL FACTORS ERROR] {ticker}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        if not all_factor_data:
            raise ValueError("没有有效的因子数据")
        
        # 合并所有数据并按日期全局排序
        combined_data = pd.concat(all_factor_data, ignore_index=True)
        combined_data = combined_data.sort_values('date').reset_index(drop=True)
        
        # 分离特征和目标
        feature_columns = [col for col in combined_data.columns 
                          if col not in ['target', 'ticker', 'date']]
        
        X = combined_data[feature_columns]
        y = combined_data['target']
        dates = combined_data['date']
        tickers = combined_data['ticker']
        
        print(f"[TIME-SERIES PREP] 总共准备了 {len(X)} 个样本，{len(feature_columns)} 个因子")
        print(f"[TIME-SERIES PREP] 时间范围: {dates.min()} 到 {dates.max()}")
        print(f"[TIME-SERIES PREP] 包含 {len(tickers.unique())} 只股票")
        
        return X, y, tickers, dates
    
    def train_models_with_bma(self, X, y, enable_hyperopt=True):
        """使用BMA替代Stacking训练模型"""
        print(f"[BMA] 开始使用BMA替代Stacking训练模型...")
        print(f"[PROGRESS] 开始时间: {datetime.now().strftime('%H:%M:%S')}")
        
        # 保存训练时的特征列顺序（原始特征，未经过Pipeline处理）
        self.training_feature_columns = X.columns.tolist()
        print(f"[FEATURE CONSISTENCY] 保存训练特征列: {len(self.training_feature_columns)} 个")
        
        # 训练前清空并延后记录pipelines（在构建base_models后统一保存）
        self.training_pipelines = {}
        
        # 智能数据处理Pipeline步骤
        # 注意：树模型对标准化/线性降维不敏感，拆分线性与树模型的Pipeline
        # 周期越短，提升IC阈值以过滤噪声
        ic_thr = 0.02 if (getattr(self, 'prediction_horizon_days', 5) <= 5) else 0.01
        linear_steps = [
            ('advanced_imputer', self._get_advanced_imputer()),
            ('ic_selector', ICFactorSelector(ic_threshold=ic_thr)),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95))
        ]
        tree_steps = [
            ('advanced_imputer', self._get_advanced_imputer()),
            ('ic_selector', ICFactorSelector(ic_threshold=ic_thr))
        ]
        
        # 基学习器配置（以树模型为主，线性作二层/稳健化）
        base_models = {}
        
        # Ridge
        ridge_pipeline = Pipeline(linear_steps + [('model', Ridge(alpha=1.0))])
        base_models['Ridge'] = ridge_pipeline
        
        # ElasticNet  
        elastic_pipeline = Pipeline(linear_steps + [('model', ElasticNet(alpha=0.1, l1_ratio=0.5))])
        base_models['ElasticNet'] = elastic_pipeline
        
        # 添加高级模型（LightGBM / XGBoost 为主力）
        if XGBOOST_AVAILABLE:
            xgb_pipeline = Pipeline(tree_steps + [('model', xgb.XGBRegressor(
                random_state=42,
                n_estimators=600,
                max_depth=7,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                tree_method='hist',
                n_jobs=-1
            ))])
            base_models['XGBoost'] = xgb_pipeline
        
        if LIGHTGBM_AVAILABLE:
            lgb_pipeline = Pipeline(tree_steps + [('model', lgb.LGBMRegressor(
                random_state=42,
                n_estimators=1000,
                max_depth=-1,
                learning_rate=0.025,
                num_leaves=96,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                min_child_samples=20,
                verbose=-1,
                n_jobs=-1
            ))])
            base_models['LightGBM'] = lgb_pipeline
        
        # 禁用单独决策树、随机森林、CatBoost 作为主力；若仍需可改为较小权重先验
        
        # CNN模型已禁用
        # if TENSORFLOW_AVAILABLE:
        #     try:
        #         # CNN模型不需要PCA，因为它可以处理高维数据
        #         cnn_pipeline_steps = [
        #             ('imputer', SimpleImputer(strategy='median')),
        #             ('scaler', StandardScaler())
        #         ]
        #         cnn_pipeline = Pipeline(cnn_pipeline_steps + [('model', CNNRegressor(
        #             sequence_length=min(20, len(X) // 5),  # 动态调整序列长度
        #             epochs=30,  # 减少训练时间
        #             batch_size=32,
        #             learning_rate=0.001,
        #             dropout_rate=0.3,
        #             verbose=0
        #         ))])
        #         base_models['CNN'] = cnn_pipeline
        #         print(f"[BMA CNN] CNN模型已添加到BMA集成")
        #     except Exception as e:
        #         print(f"[BMA CNN] CNN模型创建失败: {e}")
        print(f"[BMA] CNN模型已手动禁用，仅使用传统机器学习模型")
        
        print(f"[BMA] 创建了 {len(base_models)} 个基础模型")
        print(f"[BMA] 模型列表: {', '.join(base_models.keys())}")
        
        # 创建BMA集成器（偏向树模型）
        self.bma_model = BayesianModelAveraging(
            alpha_prior=1.5,
            shrinkage_factor=0.15,
            min_weight_threshold=0.02,
            model_class_priors={'LightGBM': 0.5, 'XGBoost': 0.5},
            enable_meta_learner=True,
            meta_learner_type='ridge'
        )
        # 记录pipelines，确保预测阶段一致性
        self.training_pipelines = {name: pipe for name, pipe in base_models.items()}
        
        try:
            # 训练BMA
            self.bma_model.fit(X.values, y.values, base_models)
            
            # 评估性能
            y_pred = self.bma_model.predict(X.values)
            train_r2 = r2_score(y.values, y_pred)
            train_mse = mean_squared_error(y.values, y_pred)
            
            print(f"[BMA] BMA训练完成")
            print(f"[BMA] 训练R²: {train_r2:.4f}, MSE: {train_mse:.6f}")
            print(f"[PROGRESS] 结束时间: {datetime.now().strftime('%H:%M:%S')}")
            
            # 计算因子IC得分
            print(f"[BMA] 计算全局因子IC得分...")
            try:
                ic_scores = {}
                for i, col in enumerate(X.columns):
                    factor_values = X.iloc[:, i]
                    ic = self.calculate_ic(factor_values, y)
                    if not np.isnan(ic) and ic != 0.0:
                        ic_scores[col] = ic
                
                self.factor_ic_scores = ic_scores
                effective_factors = len([ic for ic in ic_scores.values() if abs(ic) > 0.05])
                print(f"[BMA] 计算了 {len(ic_scores)} 个因子的IC，其中 {effective_factors} 个有效因子")
            except Exception as e:
                print(f"[BMA] IC计算失败: {e}")
                self.factor_ic_scores = {}
            
            return {'BMA': {'train_r2': train_r2, 'train_mse': train_mse}}
            
        except Exception as e:
            print(f"[BMA ERROR] BMA训练失败: {e}")
            return {}
    
    def predict_with_bma(self, X):
        """使用BMA模型进行预测"""
        if hasattr(self, 'bma_model') and self.bma_model is not None:
            try:
                # 确保预测数据的特征列与训练时一致
                if hasattr(self, 'training_feature_columns'):
                    # 检查是否有缺失的列
                    missing_columns = set(self.training_feature_columns) - set(X.columns)
                    if missing_columns:
                        print(f"[FEATURE CONSISTENCY] 添加缺失列: {missing_columns}")
                        # 为缺失的列添加默认值0
                        for col in missing_columns:
                            X[col] = 0
                    
                    # 检查是否有多余的列
                    extra_columns = set(X.columns) - set(self.training_feature_columns)
                    if extra_columns:
                        print(f"[FEATURE CONSISTENCY] 移除多余列: {extra_columns}")
                        X = X.drop(columns=list(extra_columns))
                    
                    # 重新排列列顺序以匹配训练时的顺序
                    X = X.reindex(columns=self.training_feature_columns)
                    print(f"[FEATURE CONSISTENCY] 确保特征列一致性: {len(X.columns)} 个特征")
                
                # 确保数据类型一致
                X = X.astype(float)
                
                prediction = self.bma_model.predict(X.values)
                return prediction
            except Exception as e:
                print(f"[BMA PREDICT ERROR] {e}")
                # 添加更详细的错误信息
                print(f"[DEBUG] X shape: {X.shape}, X columns: {list(X.columns)}")
                if hasattr(self, 'training_feature_columns'):
                    print(f"[DEBUG] Training columns: {self.training_feature_columns}")
                return None
        else:
            return None
    
    def generate_recommendations(self, all_data, top_n=None):
        """生成投资建议（使用动态分位数阈值）"""
        print(f"[RECOMMENDATIONS] 生成BMA投资建议...")
        
        # 首先收集所有预测收益率
        all_predictions = []
        valid_data = []
        
        for ticker, data in all_data.items():
            try:
                # 计算最新因子
                factors = self.calculate_factors(data)
                
                # 获取最新的非NaN因子值
                latest_factors = {}
                for factor_name, factor_data in factors.items():
                    if isinstance(factor_data, pd.Series) and len(factor_data) > 0:
                        latest_value = factor_data.iloc[-1]
                        if not pd.isna(latest_value) and not np.isinf(latest_value):
                            latest_factors[factor_name] = latest_value
                
                if not latest_factors:
                    print(f"[REC WARNING] {ticker}: 没有有效的因子数据")
                    continue
                
                # 创建特征向量
                base_df = pd.DataFrame([latest_factors])
                
                # 使用BMA预测
                if hasattr(self, 'bma_model') and self.bma_model is not None:
                    prediction = self.predict_with_bma(base_df)
                    if prediction is not None:
                        predicted_return = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
                        all_predictions.append(predicted_return)
                        
                        # 获取当前价格和基本信息
                        current_price = data['Close'].iloc[-1]
                        volume = data['Volume'].iloc[-1]
                        
                        valid_data.append({
                            'ticker': ticker,
                            'predicted_return': predicted_return,
                            'current_price': current_price,
                            'volume': volume,
                            'data': data,
                            'latest_factors': latest_factors
                        })
                    else:
                        continue
                else:
                    continue
                    
            except Exception as e:
                print(f"[REC ERROR] {ticker}: {e}")
                continue
        
        if not all_predictions:
            print("[REC ERROR] 没有有效的预测结果")
            return []
        
        # 计算动态阈值（分位数）
        all_predictions = np.array(all_predictions)
        buy_threshold = np.percentile(all_predictions, 70)  # 前30%为BUY
        sell_threshold = np.percentile(all_predictions, 30)  # 后30%为SELL
        
        print(f"[THRESHOLDS] BUY阈值: {buy_threshold:.4f}, SELL阈值: {sell_threshold:.4f}")
        print(f"[THRESHOLDS] 预测收益范围: {all_predictions.min():.4f} 到 {all_predictions.max():.4f}")
        
        # 生成最终推荐
        recommendations = []
        for item in valid_data:
            try:
                predicted_return = item['predicted_return']
                ticker = item['ticker']
                current_price = item['current_price']
                volume = item['volume']
                
                # 基于分位数的动态评级
                if predicted_return >= buy_threshold:
                    rating = 'BUY'
                elif predicted_return <= sell_threshold:
                    rating = 'SELL'
                else:
                    rating = 'HOLD'
                
                recommendations.append({
                    'ticker': ticker,
                    'current_price': current_price,
                    'predicted_return': predicted_return,
                    'rating': rating,
                    'volume': volume,
                    'factors_count': len(item.get('latest_factors', {}))
                })
                
            except Exception as e:
                print(f"[REC ERROR] {ticker}: {e}")
                continue
        
        # 按预测收益率排序
        recommendations = sorted(recommendations, key=lambda x: x['predicted_return'], reverse=True)
        
        # 限制返回数量
        if top_n is not None:
            recommendations = recommendations[:top_n]
        
        print(f"[RECOMMENDATIONS] 生成了 {len(recommendations)} 个BMA建议")
        
        return recommendations
    
    def _get_advanced_imputer(self):
        """获取智能数据插值器"""
        try:
            from advanced_data_imputation import DataImputationAdapter
            return DataImputationAdapter()
        except ImportError:
            print("[WARNING] 智能数据处理模块不可用，使用简单插值器")
            return SimpleImputer(strategy='median')
    
    def save_results_to_excel(self, recommendations, excel_file):
        """保存结果到指定Excel文件"""
        if not recommendations:
            print("[SAVE] 没有建议可保存")
            return None
        
        try:
            # 转换为DataFrame
            df = pd.DataFrame(recommendations)
            
            # 保存到Excel
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='BMA推荐', index=False)
                
                # 如果有模型评分信息，也保存
                if hasattr(self, 'bma_model') and self.bma_model and self.bma_model.posterior_weights:
                    weights_df = pd.DataFrame([
                        {'模型': name, '权重': weight}
                        for name, weight in self.bma_model.posterior_weights.items()
                    ])
                    weights_df.to_excel(writer, sheet_name='模型权重', index=False)
            
            print(f"[SAVE] 结果已保存到: {excel_file}")
            return excel_file
            
        except Exception as e:
            print(f"[SAVE ERROR] 保存失败: {e}")
            return None

    def save_results(self, recommendations, factor_ic_scores):
        """保存分析结果到Excel（保持原有接口）"""
        if not recommendations:
            print("[SAVE] 没有建议可保存")
            return None
        
        # 创建结果目录
        os.makedirs('result', exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = np.random.randint(100, 999)
        random_suffix2 = np.random.randint(100, 999)
        excel_file = f'result/bma_quantitative_analysis_{timestamp}_{random_suffix}_{random_suffix2}.xlsx'
        
        try:
            # 转换为DataFrame
            recommendations_df = pd.DataFrame(recommendations)
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 主要建议
                recommendations_df.to_excel(writer, sheet_name='BMA投资建议', index=False)
                
                # 因子IC分析
                ic_df = pd.DataFrame(list(factor_ic_scores.items()), 
                                   columns=['Factor', 'IC_Score'])
                ic_df = ic_df.sort_values('IC_Score', key=abs, ascending=False)
                ic_df.to_excel(writer, sheet_name='因子IC评分', index=False)
                
                # 模型权重信息
                if hasattr(self, 'bma_model') and self.bma_model is not None:
                    model_info = []
                    for model_name, weight in self.bma_model.posterior_weights.items():
                        model_info.append({
                            'Model': model_name,
                            'Posterior_Weight': weight,
                            'Weight_Percentage': f"{weight*100:.2f}%",
                            'Type': 'BMA_Ensemble'
                        })
                    
                    model_df = pd.DataFrame(model_info)
                    model_df.to_excel(writer, sheet_name='BMA模型权重', index=False)
            
            print(f"[SAVE] BMA结果已保存到: {excel_file}")
            
            # 打印总结
            print(f"\n=== BMA增强版量化分析结果总结 ===")
            print(f"分析时间: {timestamp}")
            print(f"股票总数: {len(recommendations_df)}")
            print(f"BUY推荐: {len(recommendations_df[recommendations_df['rating'] == 'BUY'])}")
            print(f"HOLD推荐: {len(recommendations_df[recommendations_df['rating'] == 'HOLD'])}")
            print(f"SELL推荐: {len(recommendations_df[recommendations_df['rating'] == 'SELL'])}")
            print(f"平均预测收益率: {recommendations_df['predicted_return'].mean():.4f}")
            print(f"有效因子数量: {len([ic for ic in factor_ic_scores.values() if abs(ic) > 0.05])}")
            
            if hasattr(self, 'bma_model') and self.bma_model is not None:
                print(f"使用BMA集成: 替代Stacking")
                print(f"BMA权重分配:")
                for model_name, weight in self.bma_model.posterior_weights.items():
                    print(f"  {model_name}: {weight:.4f}")
            
            # 新增：选出得分最高的10只股票用于IBKR自动交易
            top_10_stocks = self.generate_top_10_list(recommendations_df)
            self.save_top_10_for_ibkr(top_10_stocks, timestamp)
            
            return excel_file
        
        except Exception as e:
            print(f"[SAVE ERROR] 保存失败: {e}")
            return None
    
    def generate_top_10_list(self, recommendations):
        """生成得分最高的10只股票列表，用于IBKR自动交易"""
        try:
            # 如果输入是列表，转换为DataFrame
            if isinstance(recommendations, list):
                recommendations_df = pd.DataFrame(recommendations)
            else:
                recommendations_df = recommendations.copy()
            
            # 确保有predicted_return列，并创建final_score列（使用predicted_return作为评分标准）
            if 'predicted_return' not in recommendations_df.columns:
                print("[TOP10 ERROR] 缺少predicted_return列")
                return []
            
            # 创建final_score列（基于predicted_return）
            recommendations_df['final_score'] = recommendations_df['predicted_return']
            
            # 按predicted_return排序，选出前10名
            top_10_df = recommendations_df.nlargest(10, 'predicted_return')
            
            # 创建股票信息列表
            top_10_stocks = []
            for _, row in top_10_df.iterrows():
                stock_info = {
                    'ticker': row['ticker'],
                    'final_score': row['final_score'],
                    'predicted_return': row['predicted_return'],
                    'rating': row['rating'],
                    'recommendation': row.get('recommendation', row['rating'])  # 使用rating作为fallback
                }
                top_10_stocks.append(stock_info)
            
            print(f"\n=== IBKR自动交易股票列表 ===")
            print(f"选出得分最高的10只股票:")
            for i, stock in enumerate(top_10_stocks, 1):
                print(f"{i:2d}. {stock['ticker']:6s} | 得分: {stock['final_score']:.4f} | "
                      f"预测收益: {stock['predicted_return']:.2%} | 评级: {stock['rating']}")
            
            return top_10_stocks
            
        except Exception as e:
            print(f"[TOP10 ERROR] 生成Top10列表失败: {e}")
            return []
    
    def save_top_10_for_ibkr(self, top_10_stocks, timestamp):
        """保存Top10股票列表，用于IBKR API自动交易"""
        try:
            # 创建结果目录
            result_dir = "result"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            # 保存为JSON文件（便于API读取）
            import json
            json_file = os.path.join(result_dir, f"bma_top_10_stocks_{timestamp}.json")
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(top_10_stocks, f, ensure_ascii=False, indent=2)
            
            # 保存为CSV文件（便于查看）
            csv_file = os.path.join(result_dir, f"bma_top_10_stocks_{timestamp}.csv")
            top_10_df = pd.DataFrame(top_10_stocks)
            top_10_df.to_csv(csv_file, index=False, encoding='utf-8')
            
            # 保存为纯文本文件（便于API直接读取）
            txt_file = os.path.join(result_dir, f"bma_top_10_tickers_{timestamp}.txt")
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"# IBKR自动交易股票列表 - 生成时间: {timestamp}\n")
                f.write(f"# 格式: 股票代码,得分,预测收益率,评级\n")
                for stock in top_10_stocks:
                    f.write(f"{stock['ticker']},{stock['final_score']:.4f},"
                           f"{stock['predicted_return']:.4f},{stock['rating']}\n")
            
            print(f"[BMA] Top10股票列表已保存:")
            print(f"  JSON: {json_file}")
            print(f"  CSV:  {csv_file}")
            print(f"  TXT:  {txt_file}")
            
            # 返回股票代码列表（便于直接使用）
            ticker_list = [stock['ticker'] for stock in top_10_stocks]
            return ticker_list
            
        except Exception as e:
            print(f"[IBKR ERROR] 保存Top10列表失败: {e}")
            return []


def main():
    """主函数（保持原有接口）"""
    print("=== BMA增强版量化分析模型启动 V3 ===")
    print(f"使用贝叶斯模型平均(BMA)替代Stacking，提供更稳定的预测性能")
    print(f"使用高级模型: XGBoost={XGBOOST_AVAILABLE}, LightGBM={LIGHTGBM_AVAILABLE}, CatBoost={CATBOOST_AVAILABLE}")
    
    # 命令行参数解析（保持兼容性）
    parser = argparse.ArgumentParser(description='BMA增强版量化分析模型V3')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--top-n', type=int, default=None, help='返回top N个推荐')
    parser.add_argument('--ticker-file', type=str, default=None, help='股票列表文件 (每行一个ticker)')
    
    args = parser.parse_args()
    
    # 处理股票列表
    final_ticker_list = ticker_list
    if args.ticker_file and os.path.exists(args.ticker_file):
        try:
            with open(args.ticker_file, 'r', encoding='utf-8') as f:
                file_tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
            if file_tickers:
                final_ticker_list = file_tickers
                print(f"[TICKER FILE] 从文件加载了 {len(final_ticker_list)} 只股票")
        except Exception as e:
            print(f"[TICKER FILE ERROR] 读取股票文件失败: {e}，使用默认股票池")
    
    print(f"[DATE] 分析时间范围: {args.start_date} 到 {args.end_date}")
    print(f"[FINAL] 最终股票池: {len(final_ticker_list)} 只股票")
    print(f"[STOCKS] 当前考虑的股票: {', '.join(final_ticker_list[:25])}{'...' if len(final_ticker_list) > 25 else ''}")
    
    # 初始化模型
    model = QuantitativeModel()
    
    try:
        # 下载数据
        all_data = model.download_data(final_ticker_list, args.start_date, args.end_date)
        
        if not all_data:
            print("[ERROR] 没有可用数据，程序退出")
            return
        
        # 准备机器学习数据
        X, y, stock_names, dates = model.prepare_ml_data_with_time_series(all_data, target_period=5)
        
        # 使用BMA训练（替代Stacking）
        print(f"[TRAINING] 数据量: {len(X)} 个样本，使用BMA替代Stacking训练")
        model_scores = model.train_models_with_bma(X, y)
        
        # 生成投资建议
        recommendations = model.generate_recommendations(all_data, top_n=args.top_n)
        
        # 保存结果
        result_file = model.save_results(recommendations, model.factor_ic_scores)
        
        # 生成Top10列表用于IBKR自动交易
        if recommendations:
            top_10_stocks = model.generate_top_10_list(recommendations)
            if top_10_stocks:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model.save_top_10_for_ibkr(top_10_stocks, timestamp)
        
        if result_file:
            print(f"\n=== BMA分析完成 ===")
            print(f"结果文件: {result_file}")
            print(f"\n优势说明:")
            print(f"✓ 使用BMA替代Stacking，避免过拟合")
            print(f"✓ 贝叶斯理论基础，权重分配更合理")
            print(f"✓ Shrinkage机制，提高稳定性")
            print(f"✓ 保持原有接口，无缝替换")
        else:
            print(f"[ERROR] 结果保存失败")
            
    except Exception as e:
        print(f"[MAIN ERROR] {e}")
        import traceback
        traceback.print_exc()


class WeeklyBMAScheduler:
    """BMA模型周调度器"""
    
    def __init__(self, config_file='bma_weekly_config.json'):
        self.config_file = config_file
        self.config = self.load_config()
        self.is_running = False
        self.scheduler_thread = None
        
    def load_config(self):
        """加载配置文件"""
        default_config = {
            "enabled": True,
            "run_day": "monday",  # 每周一运行
            "run_time": "09:00",  # 运行时间
            "stock_count": -1,  # 分析股票数量 (-1为使用所有股票)
            "min_price_threshold": null,  # 关闭价格下限过滤
            "max_price_threshold": null,  # 关闭价格上限过滤
            "days_history": 1825,  # 历史数据天数(5年)
            "top_n": 20,  # Top N 推荐
            "output_dir": "weekly_bma_results",  # 输出目录
            "json_output": "weekly_bma_trading.json",  # 交易用JSON文件
            "manual_stocks": [],  # 手动添加的股票列表
            "exclude_stocks": []  # 排除的股票列表
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    print(f"[SCHEDULER] 配置文件已加载: {self.config_file}")
            except Exception as e:
                print(f"[SCHEDULER] 配置文件加载失败: {e}，使用默认配置")
        else:
            # 创建默认配置文件
            self.save_config(default_config)
            print(f"[SCHEDULER] 创建默认配置文件: {self.config_file}")
        
        return default_config
    
    def save_config(self, config=None):
        """保存配置文件"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"[SCHEDULER] 配置已保存: {self.config_file}")
        except Exception as e:
            print(f"[SCHEDULER] 配置保存失败: {e}")
    
    def filter_stocks_by_price(self, ticker_list):
        """根据价格阈值过滤股票"""
        if not self.config.get('min_price_threshold') and not self.config.get('max_price_threshold'):
            return ticker_list
        
        filtered_stocks = []
        min_price = self.config.get('min_price_threshold', 0)
        max_price = self.config.get('max_price_threshold', float('inf'))
        
        print(f"[SCHEDULER] 价格过滤: ${min_price} - ${max_price}")
        
        # 批量获取当前价格
        try:
            batch_size = 20
            for i in range(0, len(ticker_list), batch_size):
                batch = ticker_list[i:i+batch_size]
                batch_str = ' '.join(batch)
                
                try:
                    data = yf.download(batch_str, period='1d', interval='1d', 
                                     group_by='ticker', progress=False)
                    
                    for ticker in batch:
                        try:
                            if len(batch) == 1:
                                price = data['Close'].iloc[-1]
                            else:
                                price = data[ticker]['Close'].iloc[-1]
                            
                            if pd.notna(price) and min_price <= price <= max_price:
                                filtered_stocks.append(ticker)
                                
                        except (KeyError, IndexError, AttributeError):
                            continue
                            
                except Exception as e:
                    print(f"[SCHEDULER] 批次价格获取失败: {e}")
                    # 如果批量失败，添加整个批次（保守做法）
                    filtered_stocks.extend(batch)
                    
                time.sleep(0.1)  # 避免API限制
                
        except Exception as e:
            print(f"[SCHEDULER] 价格过滤失败: {e}，返回原始列表")
            return ticker_list
        
        print(f"[SCHEDULER] 价格过滤后: {len(filtered_stocks)} 只股票")
        return filtered_stocks
    
    def prepare_stock_list(self):
        """准备股票列表"""
        # 基础股票池
        stock_count = self.config.get('stock_count', -1)
        if stock_count is None or stock_count == -1 or stock_count > len(ticker_list):
            base_stocks = ticker_list
        else:
            base_stocks = ticker_list[:int(stock_count)]
        
        # 添加手动选择的股票
        manual_stocks = self.config.get('manual_stocks', [])
        if manual_stocks:
            base_stocks.extend(manual_stocks)
            print(f"[SCHEDULER] 添加手动股票: {len(manual_stocks)} 只")
        
        # 排除指定股票
        exclude_stocks = self.config.get('exclude_stocks', [])
        if exclude_stocks:
            base_stocks = [s for s in base_stocks if s not in exclude_stocks]
            print(f"[SCHEDULER] 排除股票: {len(exclude_stocks)} 只")
        
        # 去重
        final_stocks = list(dict.fromkeys(base_stocks))
        
        # 价格过滤
        final_stocks = self.filter_stocks_by_price(final_stocks)
        
        return final_stocks
    
    def run_weekly_analysis(self):
        """运行周度BMA分析"""
        try:
            print(f"\n{'='*80}")
            print(f"周度BMA分析开始 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")
            
            # 创建输出目录
            output_dir = self.config.get('output_dir', 'weekly_bma_results')
            os.makedirs(output_dir, exist_ok=True)
            
            # 准备股票列表
            stock_list = self.prepare_stock_list()
            print(f"[WEEKLY BMA] 最终股票列表: {len(stock_list)} 只")
            
            # 初始化模型
            model = QuantitativeModel()
            
            # 下载数据
            print(f"[WEEKLY BMA] 开始下载数据...")
            start_date = (datetime.now() - timedelta(days=self.config.get('days_history', 1825))).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            all_data = model.download_data(stock_list, start_date, end_date)
            
            if not all_data:
                print("[WEEKLY BMA ERROR] 没有可用数据")
                return None
            
            # 准备ML数据
            X, y, stock_names, dates = model.prepare_ml_data_with_time_series(all_data, target_period=5)
            
            # 训练BMA模型
            print(f"[WEEKLY BMA] 使用BMA训练模型...")
            model_scores = model.train_models_with_bma(X, y)
            
            # 生成推荐
            recommendations = model.generate_recommendations(all_data, top_n=self.config.get('top_n', 20))
            
            # 保存Excel结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            excel_file = os.path.join(output_dir, f'weekly_bma_analysis_{timestamp}.xlsx')
            model.save_results_to_excel(recommendations, excel_file)
            
            # 保存交易用JSON文件
            json_file = self.config.get('json_output', 'weekly_bma_trading.json')
            self.save_trading_json(recommendations, json_file)
            
            print(f"\n周度BMA分析完成!")
            print(f"Excel结果: {excel_file}")
            print(f"交易JSON: {json_file}")
            print(f"推荐数量: {len(recommendations)}")
            
            return json_file
            
        except Exception as e:
            print(f"[WEEKLY BMA ERROR] 分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_trading_json(self, recommendations, json_file):
        """保存交易用JSON文件"""
        try:
            trading_data = {
                "timestamp": datetime.now().isoformat(),
                "config": self.config,
                "recommendations": []
            }
            
            for rec in recommendations:
                trading_rec = {
                    "ticker": rec.get('ticker'),
                    "rating": rec.get('rating'),
                    "predicted_return": rec.get('predicted_return', 0),
                    "confidence": rec.get('confidence', 0),
                    "current_price": rec.get('current_price', 0),
                    "target_price": rec.get('target_price', 0),
                    "stop_loss": rec.get('stop_loss', 0),
                    "position_size": rec.get('position_size', 0),
                    "ranking": rec.get('ranking', 0)
                }
                trading_data["recommendations"].append(trading_rec)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(trading_data, f, indent=2, ensure_ascii=False)
            
            print(f"[SCHEDULER] 交易JSON已保存: {json_file}")
            
        except Exception as e:
            print(f"[SCHEDULER] 交易JSON保存失败: {e}")
    
    def start_scheduler(self):
        """启动调度器"""
        if not self.config.get('enabled', True):
            print("[SCHEDULER] 调度器已禁用")
            return
        
        print(f"[SCHEDULER] 启动周度BMA调度器...")
        print(f"[SCHEDULER] 运行时间: 每周{self.config.get('run_day', 'monday')} {self.config.get('run_time', '09:00')}")
        
        # 设置调度
        run_day = self.config.get('run_day', 'monday').lower()
        run_time = self.config.get('run_time', '09:00')
        
        getattr(schedule.every(), run_day).at(run_time).do(self.run_weekly_analysis)
        
        # 启动调度线程
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        print(f"[SCHEDULER] 调度器已启动")
    
    def stop_scheduler(self):
        """停止调度器"""
        self.is_running = False
        schedule.clear()
        print(f"[SCHEDULER] 调度器已停止")
    
    def _scheduler_loop(self):
        """调度器循环"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
    
    def run_once(self):
        """立即运行一次分析（用于测试）"""
        print("[SCHEDULER] 立即运行BMA分析...")
        return self.run_weekly_analysis()
    
    def update_config(self, **kwargs):
        """更新配置"""
        self.config.update(kwargs)
        self.save_config()
        print(f"[SCHEDULER] 配置已更新")


def main_with_scheduler():
    """带调度器的主函数"""
    parser = argparse.ArgumentParser(description='BMA增强版量化分析模型（支持周调度）')
    parser.add_argument('--mode', choices=['once', 'schedule', 'config'], default='once',
                       help='运行模式: once=单次运行, schedule=启动调度器, config=配置管理')
    parser.add_argument('--config-file', default='bma_weekly_config.json', help='配置文件路径')
    parser.add_argument('--min-price', type=float, help='最低价格阈值(美元)')
    parser.add_argument('--max-price', type=float, help='最高价格阈值(美元)')
    parser.add_argument('--add-stock', action='append', help='添加手动股票')
    parser.add_argument('--remove-stock', action='append', help='移除股票')
    
    args = parser.parse_args()
    
    # 创建调度器
    scheduler = WeeklyBMAScheduler(args.config_file)
    
    # 更新配置（如果有命令行参数）
    config_updates = {}
    if args.min_price is not None:
        config_updates['min_price_threshold'] = args.min_price
    if args.max_price is not None:
        config_updates['max_price_threshold'] = args.max_price
    if args.add_stock:
        current_manual = scheduler.config.get('manual_stocks', [])
        config_updates['manual_stocks'] = list(set(current_manual + args.add_stock))
    if args.remove_stock:
        current_exclude = scheduler.config.get('exclude_stocks', [])
        config_updates['exclude_stocks'] = list(set(current_exclude + args.remove_stock))
    
    if config_updates:
        scheduler.update_config(**config_updates)
    
    if args.mode == 'once':
        # 单次运行
        result = scheduler.run_once()
        if result:
            print(f"\n分析完成，交易文件: {result}")
        else:
            print(f"\n分析失败")
    
    elif args.mode == 'schedule':
        # 启动调度器
        scheduler.start_scheduler()
        print(f"\n调度器运行中，按 Ctrl+C 停止...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            scheduler.stop_scheduler()
            print(f"\n调度器已停止")
    
    elif args.mode == 'config':
        # 配置管理
        print(f"\n当前配置:")
        for key, value in scheduler.config.items():
            print(f"  {key}: {value}")
        print(f"\n配置文件: {args.config_file}")


if __name__ == "__main__":
    # 检查是否有调度器参数
    import sys
    if len(sys.argv) > 1 and any(arg in ['--mode', 'schedule', 'config'] for arg in sys.argv):
        main_with_scheduler()
    else:
        main()
