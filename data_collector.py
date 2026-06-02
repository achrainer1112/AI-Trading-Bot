"""
AI Trading Bot - Marktdaten-Collector (robust)
================================================
Sammelt Marktdaten und stellt sicher, dass jeder Eintrag ein Dict ist.
"""

import warnings
warnings.filterwarnings("ignore", category=Warning, module="yfinance")
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow is deprecated.*")

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from logger import log
from config import (
    FULL_WATCHLIST, PRICE_HISTORY_DAYS,
    SHORT_WINDOW, MEDIUM_WINDOW, LONG_WINDOW, EXTRA_LONG_WINDOW,
)

try:
    import yfinance as yf
except ImportError:
    log.error("yfinance nicht installiert")
    raise

# ... der Rest deiner `data_collector.py` bleibt unverändert ...
