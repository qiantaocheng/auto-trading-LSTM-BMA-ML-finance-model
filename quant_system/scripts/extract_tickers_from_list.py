"""
Extract all tickers from the categorized list provided by user.
"""

# All tickers organized by sector and volatility
TICKERS_BY_SECTOR = {
    "Information Technology": {
        "Low Volatility": {
            "Large": ["IBM", "CSCO", "ORCL"],
            "Mid": ["TDY", "DOX", "OTEX"],
            "Small": ["PLUS", "CTS", "BMBL"]
        },
        "Medium Volatility": {
            "Large": ["AAPL", "MSFT", "ACN"],
            "Mid": ["TRMB", "GEN", "FIVN"],
            "Small": ["EXTR", "CALX", "SGH"]
        },
        "High Volatility": {
            "Large": ["NVDA", "AMD", "SMCI"],
            "Mid": ["APP", "AI", "GTLB"],
            "Small": ["IONQ", "RGTI", "QBTS"]
        }
    },
    "Healthcare": {
        "Low Volatility": {
            "Large": ["JNJ", "UNH", "MRK"],
            "Mid": ["CHX", "OGN", "PRGO"],
            "Small": ["USPH", "HSTM", "EGRX"]
        },
        "Medium Volatility": {
            "Large": ["ABT", "PFE", "TMO"],
            "Mid": ["EXEL", "HALO", "NBIX"],
            "Small": ["AMN", "NEOG", "PCRX"]
        },
        "High Volatility": {
            "Large": ["LLY", "MRNA", "VRTX"],
            "Mid": ["CRSP", "NTLA", "BEAM"],
            "Small": ["SAVA", "BLUE", "SES"]
        }
    },
    "Financials": {
        "Low Volatility": {
            "Large": ["BRK.B", "JPM", "MMC"],
            "Mid": ["CINF", "BRO", "FAF"],
            "Small": ["ESGR", "BANR", "HTBI"]
        },
        "Medium Volatility": {
            "Large": ["BAC", "GS", "MS"],
            "Mid": ["KEY", "FITB", "ZION"],
            "Small": ["HOPE", "NBN", "CUBI"]
        },
        "High Volatility": {
            "Large": ["COIN", "KKR", "APO"],
            "Mid": ["UPST", "LC", "SOFI"],
            "Small": ["MARA", "RIOT", "OPEN"]
        }
    },
    "Consumer Discretionary": {
        "Low Volatility": {
            "Large": ["MCD", "SBUX", "TJX"],
            "Mid": ["HRB", "GHC", "SCI"],
            "Small": ["STRA", "LOPE", "UNFI"]
        },
        "Medium Volatility": {
            "Large": ["AMZN", "NKE", "HD"],
            "Mid": ["YETI", "CROX", "FIVE"],
            "Small": ["BOOT", "PLCE", "WWW"]
        },
        "High Volatility": {
            "Large": ["TSLA", "RCL", "CCL"],
            "Mid": ["CVNA", "CHWY", "PTON"],
            "Small": ["LUMN", "FUBO", "GME"]
        }
    },
    "Communication Services": {
        "Low Volatility": {
            "Large": ["TMUS", "CMCSA", "VZ"],
            "Mid": ["NYT", "IPG", "OMC"],
            "Small": ["IDT", "SPOK", "CNSL"]
        },
        "Medium Volatility": {
            "Large": ["GOOGL", "META", "NFLX"],
            "Mid": ["FWONA", "ZG", "YELP"],
            "Small": ["TRIP", "GOGO", "LILA"]
        },
        "High Volatility": {
            "Large": ["DASH", "RBLX", "SNAP"],
            "Mid": ["RDDT", "BMBL", "MTCH"],
            "Small": ["GREE", "WBD", "AMC"]
        }
    },
    "Industrials": {
        "Low Volatility": {
            "Large": ["RTX", "LMT", "WM"],
            "Mid": ["LII", "NDSN", "TKR"],
            "Small": ["ABM", "HNI", "MLI"]
        },
        "Medium Volatility": {
            "Large": ["CAT", "GE", "UNP"],
            "Mid": ["XPO", "SAIA", "GXO"],
            "Small": ["WNC", "GMS", "BLUE"]
        },
        "High Volatility": {
            "Large": ["UBER", "BA", "DAL"],
            "Mid": ["RKLB", "JOBY", "ACHR"],
            "Small": ["SPCE", "PLUG", "BLDP"]
        }
    },
    "Energy": {
        "Low Volatility": {
            "Large": ["XOM", "CVX", "SLB"],
            "Mid": ["OGE", "NFG", "UGI"],
            "Small": ["CLMT", "MMLP", "GLP"]
        },
        "Medium Volatility": {
            "Large": ["EOG", "COP", "MPC"],
            "Mid": ["MTDR", "CHRD", "MRO"],
            "Small": ["CDEV", "TALO", "LPI"]
        },
        "High Volatility": {
            "Large": ["OXY", "APA", "DVN"],
            "Mid": ["AR", "RRC", "SWN"],
            "Small": ["TELL", "RIG", "NINE"]
        }
    },
    "Utilities": {
        "Low Volatility": {
            "Large": ["DUK", "SO", "AEP"],
            "Mid": ["IDA", "OGE", "PNW"],
            "Small": ["YORW", "MSEX", "CWT"]
        },
        "Medium Volatility": {
            "Large": ["NEE", "EXC", "ED"],
            "Mid": ["NI", "AEE", "LNT"],
            "Small": ["OTTR", "CPK", "UNIT"]
        },
        "High Volatility": {
            "Large": ["VST", "PCG", "CEG"],
            "Mid": ["NRG", "CWEN", "AGR"],
            "Small": ["MWA", "ARTNA", "GNE"]
        }
    },
    "Materials": {
        "Low Volatility": {
            "Large": ["LIN", "APD", "SHW"],
            "Mid": ["RPM", "SON", "ATR"],
            "Small": ["KWR", "SCL", "MYE"]
        },
        "Medium Volatility": {
            "Large": ["FCX", "NEM", "DOW"],
            "Mid": ["AA", "X", "STLD"],
            "Small": ["CDE", "HL", "KGC"]
        },
        "High Volatility": {
            "Large": ["ALB", "SQM", "SCCO"],
            "Mid": ["MP", "LAC", "WOLF"],
            "Small": ["PLL", "IPI", "LITH"]
        }
    }
}


def get_all_tickers():
    """Extract all unique tickers from the categorized list."""
    all_tickers = set()
    
    for sector, volatilities in TICKERS_BY_SECTOR.items():
        for vol_level, sizes in volatilities.items():
            for size, tickers in sizes.items():
                all_tickers.update(tickers)
    
    return sorted(list(all_tickers))


def get_tickers_by_sector():
    """Get tickers organized by sector."""
    result = {}
    for sector, volatilities in TICKERS_BY_SECTOR.items():
        sector_tickers = set()
        for vol_level, sizes in volatilities.items():
            for size, tickers in sizes.items():
                sector_tickers.update(tickers)
        result[sector] = sorted(list(sector_tickers))
    return result


if __name__ == "__main__":
    all_tickers = get_all_tickers()
    print(f"Total unique tickers: {len(all_tickers)}")
    print("\nAll tickers:")
    print(", ".join(all_tickers))
    
    print("\n\nBy sector:")
    by_sector = get_tickers_by_sector()
    for sector, tickers in by_sector.items():
        print(f"\n{sector}: {len(tickers)} tickers")
        print(f"  {', '.join(tickers)}")
