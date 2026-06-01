"""
AI Trading Bot - Konfigurationsdatei (final)
=============================================
Zentraler Konfigurationsort für alle Parameter.
"""

import os
from enum import Enum
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# TRADING MODUS
# ─────────────────────────────────────────────
TRADING_MODE = os.getenv("TRADING_MODE", "LIVE")           # "PAPER", "LIVE", "DRY"
ALLOW_LIVE_TRADING: bool = os.getenv("ALLOW_LIVE_TRADING", "false").strip().lower() == "true"
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
# RISIKOPROFILE (für RiskManager, VaR, Stop-Loss etc.)
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
        "max_daily_loss_pct": 0.03,
        "max_intraday_loss_pct": 0.05,
        "vix_panic_threshold": 30.0,
    },
    RiskProfile.BALANCED: {
        "max_position_pct": 0.20,
        "min_cash_pct": 0.10,
        "max_trades_per_run": 10,
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
        "max_daily_loss_pct": 0.05,
        "max_intraday_loss_pct": 0.07,
        "vix_panic_threshold": 35.0,
    },
    RiskProfile.AGGRESSIVE: {
        "max_position_pct": 0.30,
        "min_cash_pct": 0.05,
        "max_trades_per_run": 15,
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
        "max_daily_loss_pct": 0.08,
        "max_intraday_loss_pct": 0.10,
        "vix_panic_threshold": 40.0,
    },
}

ACTIVE_RISK_PROFILE = RiskProfile.BALANCED

# ─────────────────────────────────────────────
# TOP-N SCORE PORTFOLIO OPTIMIERUNG (KERNLOGIK)
# ─────────────────────────────────────────────
MIN_SCORE_FOR_BUY = 50          # Assets unter diesem Score werden nicht gehalten (sofort verkauft)
MAX_POSITION_COUNT = 8          # Maximal Anzahl von Positionen im Portfolio
MAX_POSITION_PCT = 0.20         # Maximales Gewicht pro Asset (20%)
MIN_CASH_PCT = 0.10             # Mindestens 10% Cash (wird immer gehalten)
MIN_TRADE_VALUE = 10.0          # Absolute Untergrenze für Order (Alpaca fractional shares)

# ─────────────────────────────────────────────
# TRADE EXECUTION (wird von trade_executor.py benötigt)
# ─────────────────────────────────────────────
MIN_ORDER_VALUE = 10.0          # Mindestordervolumen in USD (Alpaca fractional shares)

# ─────────────────────────────────────────────
# REGIME-AWARE CONFIDENCE THRESHOLDS (für RiskManager, falls verwendet)
# ─────────────────────────────────────────────
REGIME_CONFIDENCE_THRESHOLDS = {
    "BULL": {"buy_threshold": 0.55, "sell_threshold": 0.70},
    "SIDEWAYS": {"buy_threshold": 0.65, "sell_threshold": 0.65},
    "BEAR": {"buy_threshold": 0.75, "sell_threshold": 0.55},
}

# ─────────────────────────────────────────────
# VOLATILITÄTS-MULTIPLIER (nicht aktiv im reinen Score-Modus, aber für Risiko)
# ─────────────────────────────────────────────
VOLATILITY_MULTIPLIERS = {
    "very_low": {"max_vol": 15.0, "multiplier": 1.2},
    "low":      {"max_vol": 25.0, "multiplier": 1.0},
    "medium":   {"max_vol": 40.0, "multiplier": 0.7},
    "high":     {"max_vol": 100.0, "multiplier": 0.4},
}

MOMENTUM_BOOST_ENABLED = False   # Im reinen Score-Modus nicht benötigt
MOMENTUM_STRENGTH_THRESHOLD = 10.0

# ─────────────────────────────────────────────
# CVAR RISK MANAGEMENT (optional, für zusätzliche Absicherung)
# ─────────────────────────────────────────────
CVAR_LIMIT_PCT = 0.05
CVAR_CONFIDENCE_LEVEL = 0.95
CVAR_LOOKBACK_DAYS = 252

# ─────────────────────────────────────────────
# ZOMBIE-LOGIK
# ─────────────────────────────────────────────
ZOMBIE_POSITION_THRESHOLD = 50.0      # USD
ZOMBIE_MIN_AGE_DAYS = 7               # Tage, bevor eine Position als Zombie gilt

# ─────────────────────────────────────────────
# COOLDOWN (verhindert zu häufiges Traden gleicher Assets)
# ─────────────────────────────────────────────
DEFAULT_ASSET_COOLDOWN_DAYS = 2
COOLDOWN_FILE = "logs/trade_cooldowns.json"

# ─────────────────────────────────────────────
# WATCHLIST & ASSET KLASSIFIKATION
# ─────────────────────────────────────────────
ETF_WATCHLIST = ["SPY", "QQQ", "VT"]
SECTOR_ETFS = ["XLV", "XLF", "XLE", "XLK"]
STOCK_WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "V", "MA", "AMD"]
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
# PORTFOLIO & LOGGING
# ─────────────────────────────────────────────
PORTFOLIO_FILE = "portfolio.json"
TRADE_LOG_FILE = "logs/trades.json"
PORTFOLIO_HISTORY_FILE = "logs/portfolio_history.json"
LOG_DIR = "logs"

INITIAL_CAPITAL = 100_000.0       # nur als Fallback, wird bei Sync überschrieben
TRADE_FRICTION_PCT = 0.001

# ─────────────────────────────────────────────
# SCHEDULER (für automatische Runs)
# ─────────────────────────────────────────────
SCHEDULE_INTERVAL = "daily"
SCHEDULE_TIME = "09:35"           # 09:35 Eastern Time (nach Marktöffnung)
SCHEDULE_WEEKDAY = 1
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16

# ─────────────────────────────────────────────
# DATEN & BACKTEST
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

BACKTEST_START_DATE = "2022-01-01"
BACKTEST_END_DATE = "2024-01-01"
BACKTEST_INITIAL_CAPITAL = 100_000.0
BACKTEST_COMMISSION = 0.001

# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
DASHBOARD_PORT = 8501
ENABLE_DASHBOARD = True
