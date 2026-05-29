"""
AI Trading Bot - Utilities (FULL VERSION)
==========================================
Enthält alle Hilfsfunktionen, Klassen und Dekorateure.
Neu: ZombieRegistry mit Mindestalter, Konsistenzprüfungen, CooldownManager.
"""

import json
import hashlib
import os
from datetime import datetime, time, date as _date, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple
import pytz
import numpy as np

from logger import log


# ─────────────────────────────────────────────────────────────
# JSON HELPERS
# ─────────────────────────────────────────────────────────────

def load_json_file(filepath: str, default: Any = None) -> Any:
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return default if default is not None else {}
    except json.JSONDecodeError:
        return default if default is not None else {}
    except Exception:
        return default if default is not None else {}


def save_json_file(filepath: str, data: Any):
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
# ZOMBIE SYSTEM (überarbeitet mit Mindestalter)
# ─────────────────────────────────────────────────────────────

ZOMBIE_POSITION_THRESHOLD = 50.0      # USD
ZOMBIE_MIN_AGE_DAYS = 7              # Position muss mindestens 7 Tage alt sein
ZOMBIE_MIN_RUNS = 5                  # oder 5 Runs alt

class ZombieRegistry:
    """Verhindert, dass liquidierte Zombie-Positionen wieder gekauft werden.
       Neu: Zombie-Status nur nach Mindestalter oder Mindestanzahl Runs.
    """
    REGISTRY_FILE = "logs/zombie_registry.json"
    STATUS_PENDING = "PENDING"
    STATUS_SELL_EXECUTED = "SELL_EXECUTED"
    STATUS_SELL_SKIPPED = "SELL_SKIPPED"

    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        self._data = self._load()

    def _load(self):
        return load_json_file(self.REGISTRY_FILE, default={})

    def _persist(self):
        save_json_file(self.REGISTRY_FILE, self._data)

    def mark_zombie(self, ticker: str, market_value: float, entry_date: str = None, reason: str = ""):
        """Markiert eine Position als Zombie, aber nur wenn sie alt genug ist."""
        # Prüfe Alter (falls entry_date vorhanden)
        if entry_date:
            try:
                entry_dt = datetime.fromisoformat(entry_date)
                age_days = (datetime.now() - entry_dt).days
                if age_days < ZOMBIE_MIN_AGE_DAYS:
                    log.debug(f"{ticker}: zu jung ({age_days} Tage) → nicht als Zombie markiert")
                    return
            except Exception:
                pass
        if ticker not in self._data:
            self._data[ticker] = {
                "status": self.STATUS_PENDING,
                "market_value": market_value,
                "reason": reason,
                "marked_at": datetime.now().isoformat(),
            }
            self._persist()
            log.info(f"[ZombieRegistry] {ticker} als Zombie markiert (Wert: ${market_value:.2f}, Alter ok)")

    def get_status(self, ticker: str):
        return self._data.get(ticker, {}).get("status")

    def is_buy_blocked(self, ticker: str) -> bool:
        """Prüft, ob ein Asset aufgrund vorheriger Zombie-Liquidation geblockt ist."""
        return ticker in self._data


# Singleton
zombie_registry = ZombieRegistry()


def find_zombie_positions(positions: Dict, threshold: float = ZOMBIE_POSITION_THRESHOLD) -> List[str]:
    zombies = []
    for ticker, pos in positions.items():
        value = pos.get("market_value", 0)
        entry_date = pos.get("entry_date")
        # Nur wenn Wert unter Threshold UND Position alt genug
        if 0 < value < threshold:
            # Prüfe Alter (falls vorhanden)
            if entry_date:
                try:
                    entry_dt = datetime.fromisoformat(entry_date)
                    age_days = (datetime.now() - entry_dt).days
                    if age_days < ZOMBIE_MIN_AGE_DAYS:
                        log.debug(f"{ticker}: Wert {value:.2f} unter Threshold, aber zu jung ({age_days} Tage) → kein Zombie")
                        continue
                except Exception:
                    pass
            zombies.append(ticker)
            zombie_registry.mark_zombie(ticker, value, entry_date, "below threshold")
    return zombies


def build_zombie_sell_orders(zombie_tickers: List[str], positions: Dict) -> List[Dict]:
    orders = []
    for ticker in zombie_tickers:
        if ticker in positions:
            orders.append({
                "ticker": ticker,
                "action": "SELL",
                "target_allocation": 0.0,
                "confidence": 1.0,
                "reason": f"Zombie liquidation (value below {ZOMBIE_POSITION_THRESHOLD}, age >= {ZOMBIE_MIN_AGE_DAYS}d)",
                "risk_approved": True,
                "zombie_cleanup": True,
            })
    return orders


# ─────────────────────────────────────────────────────────────
# TRADE ID / DEDUP
# ─────────────────────────────────────────────────────────────

def generate_trade_id(ticker: str, action: str, value: float) -> str:
    raw = f"{ticker}_{action}_{round(value,2)}_{datetime.now().date()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def generate_decision_uid(ticker: str, action: str, index: int) -> str:
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


class TradeDeduplicator:
    """Verhindert doppelte Order-Ausführung pro Tag."""
    FILE = "logs/trade_ids.json"

    def __init__(self):
        self.ids = set(load_json_file(self.FILE, []))

    def is_duplicate(self, tid: str) -> bool:
        return tid in self.ids

    def mark(self, tid: str):
        self.ids.add(tid)
        save_json_file(self.FILE, list(self.ids))

    def mark_executed(self, trade_id: str):
        self.mark(trade_id)


# ─────────────────────────────────────────────────────────────
# COOLDOWN MANAGER
# ─────────────────────────────────────────────────────────────

class CooldownManager:
    def __init__(self, cooldown_days: int = 2, filepath: str = "logs/trade_cooldowns.json"):
        self.cooldown_days = cooldown_days
        self.filepath = filepath
        self._data: Dict[str, str] = self._load()

    def _load(self) -> Dict[str, str]:
        return load_json_file(self.filepath, default={})

    def _save(self):
        save_json_file(self.filepath, self._data)

    def is_on_cooldown(self, ticker: str) -> bool:
        last_str = self._data.get(ticker)
        if not last_str:
            return False
        try:
            last = datetime.fromisoformat(last_str).date()
            delta = (_date.today() - last).days
            return delta < self.cooldown_days
        except Exception:
            return False

    def register_trade(self, ticker: str):
        self._data[ticker] = datetime.now().isoformat()
        self._save()

    def filter_decisions(self, decisions: List[Dict]) -> Tuple[List[Dict], List[str]]:
        filtered = []
        warnings = []
        for d in decisions:
            ticker = d.get("ticker")
            action = d.get("action")
            if action == "BUY" and self.is_on_cooldown(ticker):
                warnings.append(f"{ticker}: BUY blocked (cooldown active)")
                d = dict(d)
                d["action"] = "HOLD"
                d["risk_approved"] = False
                d["reason"] = f"{d.get('reason','')} [COOLDOWN]"
            filtered.append(d)
        return filtered, warnings

    def register_executed_trades(self, executed_trades: List[Dict]):
        for t in executed_trades:
            if t.get("status") == "EXECUTED" and t.get("ticker"):
                self.register_trade(t["ticker"])


cooldown_manager = CooldownManager()


# ─────────────────────────────────────────────────────────────
# KONSISTENZPRÜFUNGEN (ASSERTIONS)
# ─────────────────────────────────────────────────────────────

def assert_portfolio_consistency(
    target_weights: Dict[str, float],
    current_weights: Dict[str, float],
    cash_target: float,
    min_trade_value: float,
    zombie_threshold: float = ZOMBIE_POSITION_THRESHOLD,
):
    """Führt mehrere Konsistenzprüfungen durch."""
    # Keine negativen Gewichte
    for t, w in target_weights.items():
        assert w >= 0, f"Negatives Zielgewicht für {t}: {w}"
    # Summe der Zielgewichte + Cash sollte nicht > 1 sein (kann kleiner sein)
    total = sum(target_weights.values()) + cash_target
    assert total <= 1.0 + 1e-6, f"Zielgewichte + Cash überschreiten 100%: {total:.1%}"
    # Zombie-Threshold darf nicht größer als min_trade_value sein
    assert zombie_threshold <= min_trade_value, f"Zombie-Threshold ({zombie_threshold}) > min_trade_value ({min_trade_value})"
    # Keine Position unter min_trade_value als Ziel (außer 0)
    for t, w in target_weights.items():
        if w > 0 and w * 100_000 < min_trade_value:
            log.warning(f"Konsistenz: Zielgewicht {t}={w:.1%} entspricht kleinem Wert (<{min_trade_value})")


def assert_no_duplicate_tickers(decisions: List[Dict]):
    """Prüft, dass kein Ticker in der Entscheidungsliste doppelt vorkommt."""
    seen = set()
    for d in decisions:
        t = d.get("ticker")
        if t in seen:
            raise ValueError(f"Duplikater Ticker in Entscheidungen: {t}")
        seen.add(t)


# ─────────────────────────────────────────────────────────────
# STATISTIKEN
# ─────────────────────────────────────────────────────────────

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.05) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    excess = arr - risk_free_rate / 252
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / excess.std())


def calculate_volatility(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(252))


def calculate_beta_alpha(returns: List[float], benchmark_returns: List[float], risk_free_rate: float = 0.05) -> Tuple[float, float]:
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


def calculate_profit_factor(returns: List[float]) -> float:
    wins = sum(r for r in returns if r > 0)
    losses = -sum(r for r in returns if r < 0)
    if losses == 0:
        return float('inf') if wins > 0 else 0.0
    return float(wins / losses)


def calculate_max_drawdown(values: List[float]) -> float:
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


def pct_change(old: float, new: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / abs(old) * 100


# ─────────────────────────────────────────────────────────────
# NORMALISIERUNG VON AI-ENTSCHEIDUNGEN
# ─────────────────────────────────────────────────────────────

def normalize_ai_decisions(
    decisions: List[Dict],
    positions: Dict[str, Dict],
    total_value: float,
    market_data: Dict[str, Dict],
    correlation_groups: List[List[str]],
) -> Tuple[List[Dict], List[str]]:
    warnings = []
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
                note = f"{ticker}: current alloc {current_alloc:.1%} >= target {target_alloc:.1%} → BUY→HOLD"
                decision["action"] = "HOLD"
                decision["risk_approved"] = False
                decision["reason"] = f"{decision.get('reason', '')} [NORM: {note}]"
                warnings.append(note)
        elif action == "SELL" and current_alloc <= 0:
            note = f"{ticker}: no position → SELL→HOLD"
            decision["action"] = "HOLD"
            decision["risk_approved"] = False
            decision["reason"] = f"{decision.get('reason', '')} [NORM: {note}]"
            warnings.append(note)

    return decisions, warnings
