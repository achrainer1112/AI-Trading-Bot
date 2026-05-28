"""
AI Trading Bot - Konfigurationsdatei (Erweitert)
=================================================
PHASE 4A: Live Trading Migration + Erweiterte Risikoparameter

Neue Parameter:
  - MAX_DAILY_LOSS_PCT (für Emergency Cash Mode / Circuit Breaker)
  - CAPITAL_ROTATION_MIN_SCORE_DIFF
  - CAPITAL_ROTATION_MAX_PER_RUN
  - DYNAMIC_POSITION_SIZING_ENABLED
  - VOLATILITY_TARGET (für dynamisches Sizing)
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# TRADING MODUS – PHASE 4A: LIVE
# ─────────────────────────────────────────────
TRADING_MODE = os.getenv("TRADING_MODE", "LIVE")

# ─────────────────────────────────────────────
# LIVE TRADING GATE
# ─────────────────────────────────────────────
ALLOW_LIVE_TRADING: bool = os.getenv("ALLOW_LIVE_TRADING", "false").strip().lower() == "true"

# ─────────────────────────────────────────────
# DRAWDOWN KILL-SWITCH
# ─────────────────────────────────────────────
KILL_SWITCH_DRAWDOWN_PCT: float = float(os.getenv("KILL_SWITCH_DRAWDOWN_PCT", "0.08"))

# ─────────────────────────────────────────────
# API KEYS
# ─────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets")

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ─────────────────────────────────────────────
# RISIKOPROFILE (erweitert)
# ─────────────────────────────────────────────
class RiskProfile(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


RISK_SETTINGS = {
    RiskProfile.CONSERVATIVE: {
        "max_position_pct": 0.15,
        "min_cash_pct": 0.20,
        "max_trades_per_run": 3,
        "stop_loss_pct": 0.05,
        "confidence_threshold": 0.70,
        "max_sector_exposure": 0.35,
        "max_factor_exposure": 0.50,
        "min_drift_to_rebalance": 0.03,
        "min_trade_value": 200.0,
        "asset_cooldown_days": 3,
        "max_daily_turnover": 0.15,
        "var_confidence": 0.95,
        "max_portfolio_var": 0.03,
        "volatility_target": 0.10,
        "max_drawdown_trigger": 0.15,
        "min_buy_score": 65,
        # NEU: Circuit Breaker / Emergency Cash Mode
        "max_daily_loss_pct": 0.03,      # 3% Tagesverlust löst Emergency Cash aus
        "max_intraday_loss_pct": 0.05,   # 5% Intraday-Verlust löst CB1 aus
        "vix_panic_threshold": 30.0,     # VIX über 30 löst DE_RISK_50 aus
    },
    RiskProfile.BALANCED: {
        "max_position_pct": 0.20,
        "min_cash_pct": 0.10,
        "max_trades_per_run": 5,
        "stop_loss_pct": 0.08,
        "confidence_threshold": 0.60,
        "max_sector_exposure": 0.45,
        "max_factor_exposure": 0.60,
        "min_drift_to_rebalance": 0.02,
        "min_trade_value": 150.0,
        "asset_cooldown_days": 2,
        "max_daily_turnover": 0.25,
        "var_confidence": 0.95,
        "max_portfolio_var": 0.05,
        "volatility_target": 0.15,
        "max_drawdown_trigger": 0.20,
        "min_buy_score": 60,
        # NEU
        "max_daily_loss_pct": 0.05,
        "max_intraday_loss_pct": 0.07,
        "vix_panic_threshold": 35.0,
    },
    RiskProfile.AGGRESSIVE: {
        "max_position_pct": 0.30,
        "min_cash_pct": 0.05,
        "max_trades_per_run": 10,
        "stop_loss_pct": 0.12,
        "confidence_threshold": 0.50,
        "max_sector_exposure": 0.55,
        "max_factor_exposure": 0.70,
        "min_drift_to_rebalance": 0.015,
        "min_trade_value": 100.0,
        "asset_cooldown_days": 1,
        "max_daily_turnover": 0.40,
        "var_confidence": 0.95,
        "max_portfolio_var": 0.08,
        "volatility_target": 0.25,
        "max_drawdown_trigger": 0.30,
        "min_buy_score": 55,
        # NEU
        "max_daily_loss_pct": 0.08,
        "max_intraday_loss_pct": 0.10,
        "vix_panic_threshold": 40.0,
    },
}

ACTIVE_RISK_PROFILE = RiskProfile.BALANCED

# ─────────────────────────────────────────────
# CAPITAL ROTATION (neu)
# ─────────────────────────────────────────────
CAPITAL_ROTATION_ENABLED = True
CAPITAL_ROTATION_MIN_SCORE_DIFF = 15.0      # Mindest-Score-Differenz für Rotation
CAPITAL_ROTATION_MAX_PER_RUN = 2            # Max. Rotationen pro Run
CAPITAL_ROTATION_MIN_VALUE_USD = 50.0      # Mindestwert für Rotation
CAPITAL_ROTATION_MIN_HOLD_DAYS = 5          # Position muss mind. 5 Tage gehalten sein

# ─────────────────────────────────────────────
# DYNAMIC POSITION SIZING (neu)
# ─────────────────────────────────────────────
DYNAMIC_POSITION_SIZING_ENABLED = True
VOLATILITY_TARGET = 0.15                    # Zielvolatilität für Skalierung
MAX_VOLATILITY_FACTOR = 2.0
MIN_VOLATILITY_FACTOR = 0.5

# ─────────────────────────────────────────────
# SCORE GUARDRAILS (neu)
# ─────────────────────────────────────────────
SCORE_GUARDRAIL_STRICT = True
SCORE_MIN_FOR_BUY = 50                      # Absolutes Minimum
SCORE_MIN_FOR_BUY_BEAR = 65                 # Höheres Minimum im Bärenmarkt
SCORE_BUY_WITHOUT_GUARDRAIL = 60            # Ohne Guardrail-Override

# ─────────────────────────────────────────────
# WATCHLIST & ETFs
# ─────────────────────────────────────────────
ETF_WATCHLIST = ["SPY", "QQQ", "VT"]
SECTOR_ETFS = ["XLV", "XLF", "XLE", "XLK"]
STOCK_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "JPM", "V", "MA", "AMD",
]
FULL_WATCHLIST = ETF_WATCHLIST + SECTOR_ETFS + STOCK_WATCHLIST

SECTOR_CLASSIFICATION = {
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "AMD": "tech",
    "QQQ": "tech", "XLK": "tech", "GOOGL": "consumer", "AMZN": "consumer",
    "V": "financial", "MA": "financial", "JPM": "financial",
    "XLV": "healthcare", "XLF": "financial", "XLE": "energy",
    "SPY": "diversified", "VT": "diversified",
}

FACTOR_CLASSIFICATION = {
    "QQQ": "tech", "XLK": "tech", "XLV": "healthcare", "XLF": "financial",
    "XLE": "energy", "SPY": "diversified", "VT": "diversified",
}

CORRELATION_GROUPS = [
    ["SPY", "QQQ", "XLK", "VT", "AAPL", "MSFT", "AMZN"],
    ["QQQ", "XLK", "AAPL", "MSFT", "NVDA", "AMD"],
    ["XLF", "JPM", "V", "MA"],
]

ETF_SECTOR_WEIGHTS = {
    "QQQ": {"tech": 0.70, "diversified": 0.30},
    "XLK": {"tech": 0.90, "diversified": 0.10},
    "SPY": {"diversified": 1.00},
    "VT":  {"diversified": 1.00},
    "XLV": {"healthcare": 0.80, "diversified": 0.20},
    "XLF": {"financial": 0.80, "diversified": 0.20},
    "XLE": {"energy": 0.80, "diversified": 0.20},
}
ETF_FACTOR_WEIGHTS = ETF_SECTOR_WEIGHTS

# ─────────────────────────────────────────────
# PORTFOLIO EINSTELLUNGEN
# ─────────────────────────────────────────────
PORTFOLIO_FILE = "portfolio.json"
TRADE_LOG_FILE = "logs/trades.json"
PORTFOLIO_HISTORY_FILE = "logs/portfolio_history.json"
LOG_DIR = "logs"

INITIAL_CAPITAL = 100_000.0
MIN_ORDER_VALUE = 10.0

# ─────────────────────────────────────────────
# SCORE ENGINE
# ─────────────────────────────────────────────
DEFAULT_MIN_BUY_SCORE = 60
SCORE_TOP_K_CANDIDATES = 8
LLM_SCORE_OVERRIDE_LIMIT = 15

# ─────────────────────────────────────────────
# REBALANCING / TRADE FRICTION
# ─────────────────────────────────────────────
TRADE_FRICTION_PCT = 0.001
TAX_ESTIMATE_PCT = 0.0
DEFAULT_ASSET_COOLDOWN_DAYS = 2
COOLDOWN_FILE = "logs/trade_cooldowns.json"

# ─────────────────────────────────────────────
# PERFORMANCE TRACKING
# ─────────────────────────────────────────────
PERFORMANCE_FILE = "logs/performance_stats.json"
SIGNAL_STATS_FILE = "logs/signal_stats.json"

# ─────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────
SCHEDULE_INTERVAL = "daily"
SCHEDULE_TIME = "09:35"
SCHEDULE_WEEKDAY = 1
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16

# ─────────────────────────────────────────────
# DATEN
# ─────────────────────────────────────────────
PRICE_HISTORY_DAYS = 120
SHORT_WINDOW = 7
MEDIUM_WINDOW = 20
LONG_WINDOW = 50
EXTRA_LONG_WINDOW = 90
MAX_NEWS_ARTICLES = 20
NEWS_TOPICS = [
    "inflation", "interest rates", "recession", "tech stocks",
    "energy prices", "federal reserve", "earnings", "GDP"
]

# ─────────────────────────────────────────────
# BACKTESTING
# ─────────────────────────────────────────────
BACKTEST_START_DATE = "2022-01-01"
BACKTEST_END_DATE = "2024-01-01"
BACKTEST_INITIAL_CAPITAL = 100_000.0
BACKTEST_COMMISSION = 0.001

# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
DASHBOARD_PORT = 8501
ENABLE_DASHBOARD = True
