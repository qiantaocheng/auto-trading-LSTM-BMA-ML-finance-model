"""
API Configuration - DELAYED DATA MODE
"""

# Polygon.io API - DELAYED DATA SUBSCRIPTION
POLYGON_API_KEY = "FExbaO1xdmrV6f6p3zHCxk8IArjeowQ1"  # Your actual delayed data API key

# REMOVED: AlphaVantage API (not needed)
# REMOVED: Yahoo Finance (not using external free sources)

# Data source configuration - POLYGON ONLY
USE_POLYGON_ONLY = True
USE_DELAYED_DATA_MODE = True
DISABLE_REALTIME_CALLS = True

# Data source priority - POLYGON DELAYED DATA ONLY
DATA_SOURCE_PRIORITY = [
    "polygon_delayed"  # Only use Polygon with delayed data settings
]

# Delayed data settings
DELAYED_DATA_CONFIG = {
    "max_data_delay_hours": 24,
    "use_previous_close_fallback": True,
    "ignore_realtime_403_errors": True,
    "prefer_historical_data": True,
    "skip_intraday_requests": True
}

# Cache settings - extended for delayed data
ENABLE_CACHE = True
CACHE_DIR = "D:/trade/data_cache"
CACHE_EXPIRY_HOURS = 12  # Shorter cache for delayed data
