"""
AI Trading Bot - Utilities (FIXED VERSION)
==========================================
"""

import json
import hashlib
from datetime import datetime, time
from typing import Dict, Any, Optional, List, Set, Tuple
import pytz
import numpy as np

from logger import log

# ─────────────────────────────────────────────────────────────
# JSON HELPERS (MÜSSEN VOR KLASSEN STEHEN!)
# ─────────────────────────────────────────────────────────────

def load_json_file(filepath: str, default: Any = None) -> Any:
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return default if default is not None else {}
    except json.JSONDecodeError:
        return default if default is not None else {}


def save_json_file(filepath: str, data: Any):
    import os
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────
# MARKET TIME
# ─────────────────────────────────────────────────────────────

EASTERN = pytz.timezone("America/New_York")

def is_market_open() -> bool:
    now = datetime.now(EASTERN)
    if now.weekday() >= 5:
        return False
    return time(9, 30) <= now.time() <= time(16, 0)


def market_status() -> str:
    now = datetime.now(EASTERN)
    if now.weekday() >= 5:
        return "WEEKEND_CLOSED"
    t = now.time()
    if t < time(9, 30):
        return "PRE_MARKET"
    if t <= time(16, 0):
        return "OPEN"
    return "AFTER_HOURS"


# ─────────────────────────────────────────────────────────────
# ZOMBIE SYSTEM
# ─────────────────────────────────────────────────────────────

ZOMBIE_POSITION_THRESHOLD = 50.0


class ZombieRegistry:
    REGISTRY_FILE = "logs/zombie_registry.json"

    STATUS_PENDING = "PENDING"
    STATUS_SELL_EXECUTED = "SELL_EXECUTED"
    STATUS_SELL_SKIPPED = "SELL_SKIPPED"

    def __init__(self):
        import os
        os.makedirs("logs", exist_ok=True)
        self._data = self._load()

    def _load(self):
        return load_json_file(self.REGISTRY_FILE, default={})

    def _persist(self):
        save_json_file(self.REGISTRY_FILE, self._data)

    def mark_zombie(self, ticker: str, market_value: float, reason: str = ""):
        if ticker not in self._data:
            self._data[ticker] = {
                "status": self.STATUS_PENDING,
                "market_value": market_value,
                "reason": reason,
                "marked_at": datetime.now().isoformat(),
            }
            self._persist()

    def get_status(self, ticker: str):
        return self._data.get(ticker, {}).get("status")

    def is_buy_blocked(self, ticker: str) -> bool:
        return ticker in self._data


# SINGLETON (WICHTIG!)
zombie_registry = ZombieRegistry()


# ─────────────────────────────────────────────────────────────
# ZOMBIE SELL ORDERS
# ─────────────────────────────────────────────────────────────

def build_zombie_sell_orders(zombie_tickers, positions):
    orders = []

    for ticker in zombie_tickers:
        pos = positions.get(ticker)
        if not pos:
            continue

        orders.append({
            "ticker": ticker,
            "action": "SELL",
            "target_allocation": 0.0,
            "confidence": 1.0,
            "reason": "Zombie liquidation",
            "risk_approved": True,
            "zombie_cleanup": True,
        })

    return orders


def find_zombie_positions(positions, threshold=ZOMBIE_POSITION_THRESHOLD):
    zombies = []

    for ticker, pos in positions.items():
        value = pos.get("market_value", 0)
        if 0 < value < threshold:
            zombies.append(ticker)
            zombie_registry.mark_zombie(ticker, value, "below threshold")

    return zombies


# ─────────────────────────────────────────────────────────────
# TRADE ID / DEDUP
# ─────────────────────────────────────────────────────────────

def generate_trade_id(ticker: str, action: str, value: float):
    raw = f"{ticker}_{action}_{round(value,2)}_{datetime.now().date()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def generate_decision_uid(ticker: str, action: str, index: int):
    base = f"D{index + 1}_{ticker}_{action}"
    return hashlib.sha256(base.encode()).hexdigest()[:10]


def ensure_decision_ids(decisions: List[Dict]) -> List[Dict]:
    for idx, d in enumerate(decisions):
        if not d.get("decision_id"):
            d["decision_id"] = generate_decision_uid(
                str(d.get("ticker", "UNK")).upper(),
                str(d.get("action", "HOLD")).upper(),
                idx,
            )
    return decisions


def normalize_ai_decisions(
    decisions: List[Dict],
    positions: Dict[str, Dict],
    total_value: float,
    market_data: Dict[str, Dict],
    correlation_groups: List[List[str]],
) -> (List[Dict], List[str]):
    """Normalize raw AI decisions before final risk validation."""
    warnings: List[str] = []
    decisions = ensure_decision_ids([dict(d) for d in decisions])

    current_allocs = {
        ticker: pos.get("market_value", 0.0) / total_value if total_value else 0.0
        for ticker, pos in positions.items()
    }

    for decision in decisions:
        ticker = decision.get("ticker")
        action = decision.get("action", "HOLD")
        target_alloc = float(decision.get("target_allocation", 0.0))
        current_alloc = current_allocs.get(ticker, 0.0)

        if action == "BUY":
            if current_alloc >= target_alloc - 1e-6:
                note = (
                    f"{ticker}: current allocation {current_alloc:.1%} >= target {target_alloc:.1%} "
                    "→ BUY converted to HOLD"
                )
                decision["action"] = "HOLD"
                decision["risk_approved"] = False
                decision["reason"] = f"{decision.get('reason', '')} [NORMALIZATION: {note}]"
                warnings.append(note)
            elif target_alloc - current_alloc < 0.01:
                note = (
                    f"{ticker}: allocation drift {target_alloc - current_alloc:.1%} < 1% "
                    "→ BUY converted to HOLD"
                )
                decision["action"] = "HOLD"
                decision["risk_approved"] = False
                decision["reason"] = f"{decision.get('reason', '')} [NORMALIZATION: {note}]"
                warnings.append(note)

        elif action == "SELL" and current_alloc <= 0:
            note = f"{ticker}: no current position → SELL converted to HOLD"
            decision["action"] = "HOLD"
            decision["risk_approved"] = False
            decision["reason"] = f"{decision.get('reason', '')} [NORMALIZATION: {note}]"
            warnings.append(note)

    for group in correlation_groups:
        group_buys = [d for d in decisions if d.get("action") == "BUY" and d.get("ticker") in group]
        if len(group_buys) <= 1:
            continue

        group_buys.sort(
            key=lambda d: (d.get("confidence", 0.0), d.get("target_allocation", 0.0)),
            reverse=True,
        )
        leader = group_buys[0]
        for other in group_buys[1:]:
            if other["confidence"] + 0.05 < leader["confidence"]:
                note = (
                    f"{other['ticker']}: correlated with {leader['ticker']} BUY → downgraded to HOLD"
                )
                other["action"] = "HOLD"
                other["risk_approved"] = False
                other["reason"] = f"{other.get('reason', '')} [NORMALIZATION: {note}]"
                warnings.append(note)

    return decisions, warnings


class TradeDeduplicator:
    FILE = "logs/trade_ids.json"

    def __init__(self):
        self.ids = set(load_json_file(self.FILE, []))

    def is_duplicate(self, tid: str):
        return tid in self.ids

    def mark(self, tid: str):
        self.ids.add(tid)
        save_json_file(self.FILE, list(self.ids))

    def mark_executed(self, trade_id: str):
        self.mark(trade_id)


# ─────────────────────────────────────────────────────────────
# STATISTICS (FIXED IMPORTS)
# ─────────────────────────────────────────────────────────────

def calculate_sharpe_ratio(returns, risk_free_rate=0.05):
    if len(returns) < 2:
        return 0.0

    arr = np.array(returns)
    excess = arr - risk_free_rate / 252

    if excess.std() == 0:
        return 0.0

    return float(np.sqrt(252) * excess.mean() / excess.std())


def calculate_volatility(returns):
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    return float(np.std(arr, ddof=1) * np.sqrt(252))


def calculate_beta_alpha(returns, benchmark_returns, risk_free_rate=0.05):
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0, 0.0

    arr = np.array(returns)
    bench = np.array(benchmark_returns)
    if len(arr) != len(bench):
        min_len = min(len(arr), len(bench))
        arr = arr[-min_len:]
        bench = bench[-min_len:]

    cov = np.cov(arr, bench, ddof=1)
    if cov.shape != (2, 2) or cov[1, 1] == 0:
        return 0.0, 0.0

    beta = float(cov[0, 1] / cov[1, 1])
    alpha = float(arr.mean() - beta * bench.mean() - risk_free_rate / 252)
    return beta, alpha


def calculate_profit_factor(returns):
    wins = sum(r for r in returns if r > 0)
    losses = -sum(r for r in returns if r < 0)
    if losses == 0:
        return float('inf') if wins > 0 else 0.0
    return float(wins / losses)


def calculate_max_drawdown(values):
    if len(values) < 2:
        return 0.0

    peak = values[0]
    max_dd = 0.0

    for v in values:
        peak = max(peak, v)
        dd = (v - peak) / peak
        max_dd = min(max_dd, dd)

    return float(max_dd)


# ─────────────────────────────────────────────────────────────
# FORMAT HELPERS
# ─────────────────────────────────────────────────────────────

def format_currency(x: float) -> str:
    return f"${x:,.2f}"


def pct_change(old, new):
    if old == 0:
        return 0
    return (new - old) / abs(old) * 100

# ─────────────────────────────────────────────────────────────
# COOLDOWN MANAGER
# ─────────────────────────────────────────────────────────────

from datetime import date as _date
import json as _json

class CooldownManager:
    """
    Verwaltet Cooldowns pro Asset.
    Verhindert zu häufige Trades auf demselben Ticker.
    """

    def __init__(self, cooldown_days: int = 2, filepath: str = "logs/trade_cooldowns.json"):
        self.cooldown_days = cooldown_days
        self.filepath = filepath
        self._data: Dict[str, str] = self._load()

    def _load(self) -> Dict[str, str]:
        try:
            with open(self.filepath) as f:
                return _json.load(f)
        except Exception:
            return {}

    def _save(self):
        try:
            import os
            os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
            with open(self.filepath, "w") as f:
                _json.dump(self._data, f, indent=2)
        except Exception as e:
            log.warning(f"Cooldown save failed: {e}")

    def is_on_cooldown(self, ticker: str) -> bool:
        last_str = self._data.get(ticker)
        if not last_str:
            return False
        try:
            from datetime import datetime as _dt
            last = _dt.fromisoformat(last_str).date()
            delta = (_date.today() - last).days
            return delta < self.cooldown_days
        except Exception:
            return False

    def register_trade(self, ticker: str):
        self._data[ticker] = datetime.now().isoformat()
        self._save()

    def filter_decisions(self, decisions: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """Filtert Entscheidungen, die sich noch im Cooldown befinden."""
        filtered = []
        blocked = []
        for d in decisions:
            ticker = d.get("ticker")
            action = d.get("action")
            # Cooldown gilt nur für BUY, nicht für SELL/Stop-Loss/Zombie
            if action == "BUY" and self.is_on_cooldown(ticker):
                blocked.append(f"{ticker}: BUY blocked (cooldown active)")
                d = dict(d, action="HOLD", risk_approved=False,
                         reason=f"{d.get('reason','')} [COOLDOWN]")
            filtered.append(d)
        return filtered, blocked

    def register_executed_trades(self, executed_trades: List[Dict]):
        for t in executed_trades:
            if t.get("status") == "EXECUTED" and t.get("ticker"):
                self.register_trade(t["ticker"])


# Global singleton
cooldown_manager = CooldownManager()