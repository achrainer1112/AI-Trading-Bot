"""
AI Trading Bot - Utilities (FULL VERSION mit Zombie-Mindestalter & Konsistenzprüfungen)
==================================================
Enthält alle Hilfsfunktionen, Klassen und Dekorateure:

- JSON-Helper
- Marktzeit-Funktionen
- ZombieRegistry (verhindert Wiederholungskäufe nach Liquidation, mit Mindestalter)
- CooldownManager (verhindert zu häufige Trades pro Asset)
- TradeDeduplicator (doppelte Order-IDs)
- Statistik-Funktionen (Sharpe, Max Drawdown, etc.)
- Formatierungen
- Normalisierungsfunktionen für AI-Entscheidungen
- Konsistenzprüfungen (Assertions für Portfolio)
- Portfolio-Report Helper
"""

import json
import hashlib
import os
from datetime import datetime, time, date as _date
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
# ZOMBIE SYSTEM (mit Mindestalter)
# ─────────────────────────────────────────────────────────────

ZOMBIE_POSITION_THRESHOLD = 50.0          # USD
ZOMBIE_MIN_AGE_DAYS = 7                  # Mindestalter in Tagen
ZOMBIE_MIN_RUNS = 5                      # Mindestanzahl Runs (Alternative)

class ZombieRegistry:
    """Verhindert, dass liquidierte Zombie-Positionen wieder gekauft werden. Berücksichtigt Mindestalter."""
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

    def mark_zombie(self, ticker: str, market_value: float, reason: str = ""):
        if ticker not in self._data:
            self._data[ticker] = {
                "status": self.STATUS_PENDING,
                "market_value": market_value,
                "reason": reason,
                "marked_at": datetime.now().isoformat(),
                "age_days": 0,
            }
            self._persist()
            log.info(f"[ZombieRegistry] {ticker} als Zombie markiert (Wert: ${market_value:.2f})")

    def get_status(self, ticker: str):
        return self._data.get(ticker, {}).get("status")

    def is_buy_blocked(self, ticker: str) -> bool:
        """Prüft, ob ein Asset aufgrund vorheriger Zombie-Liquidation geblockt ist."""
        return ticker in self._data


# Singleton
zombie_registry = ZombieRegistry()


def find_zombie_positions(
    positions: Dict,
    threshold: float = ZOMBIE_POSITION_THRESHOLD,
    min_age_days: int = ZOMBIE_MIN_AGE_DAYS,
) -> List[str]:
    """
    Findet Zombie-Positionen: solche mit Marktwert < threshold UND Alter >= min_age_days.
    Vermeidet frisch gekaufte Positionen.
    """
    zombies = []
    today = datetime.now()
    for ticker, pos in positions.items():
        market_value = pos.get("market_value", 0)
        if market_value <= 0 or market_value >= threshold:
            continue
        # Alter berechnen
        entry_date_str = pos.get("entry_date")
        age_days = 0
        if entry_date_str:
            try:
                entry_dt = datetime.fromisoformat(entry_date_str)
                age_days = max(0, (today - entry_dt).days)
            except Exception:
                age_days = 0
        if age_days < min_age_days:
            log.debug(f"{ticker}: Wert ${market_value:.2f} unter Threshold, aber erst {age_days} Tage alt -> kein Zombie")
            continue
        zombies.append(ticker)
        zombie_registry.mark_zombie(ticker, market_value, f"below threshold after {age_days} days")
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
                "reason": f"Zombie liquidation (age >= {ZOMBIE_MIN_AGE_DAYS}d, value low)",
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
# STATISTICS
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
            elif target_alloc - current_alloc < 0.01:
                note = f"{ticker}: drift {target_alloc-current_alloc:.1%} < 1% → BUY→HOLD"
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

    # Korrelations-Filter
    for group in correlation_groups:
        group_buys = [d for d in decisions if d.get("action") == "BUY" and d.get("ticker") in group]
        if len(group_buys) <= 1:
            continue
        group_buys.sort(key=lambda d: (d.get("confidence", 0.0), d.get("target_allocation", 0.0)), reverse=True)
        leader = group_buys[0]
        for other in group_buys[1:]:
            if other["confidence"] + 0.05 < leader["confidence"]:
                note = f"{other['ticker']}: correlated with {leader['ticker']} BUY → downgraded to HOLD"
                other["action"] = "HOLD"
                other["risk_approved"] = False
                other["reason"] = f"{other.get('reason', '')} [NORM: {note}]"
                warnings.append(note)

    return decisions, warnings


# ─────────────────────────────────────────────────────────────
# KONSISTENZPRÜFUNGEN (ASSERTIONS)
# ─────────────────────────────────────────────────────────────

def assert_portfolio_consistency(
    target_weights: Dict[str, float],
    current_weights: Dict[str, float],
    cash_target: float,
    min_trade_value: float,
    zombie_threshold: float,
) -> None:
    """
    Führt mehrere Konsistenzprüfungen durch:
    - Keine negativen Zielgewichte
    - Summe der Zielgewichte + Cash ≈ 1 (Toleranz 1%)
    - Keine Zielgewichte unter der Mindestordergröße (wenn aktuell nicht gehalten)
    - Zombie-Threshold darf nicht größer sein als min_trade_value
    """
    # 1. Negative Gewichte
    for ticker, w in target_weights.items():
        assert w >= 0, f"Negative target weight for {ticker}: {w:.2%}"
    
    # 2. Summe + Cash ≈ 1
    total_target = sum(target_weights.values())
    total_with_cash = total_target + cash_target
    assert abs(total_with_cash - 1.0) < 0.01, f"Portfolio sum + cash = {total_with_cash:.1%} != 100%"
    
    # 3. Mindestordergröße für neue Käufe (wenn aktuell nicht gehalten)
    for ticker, target in target_weights.items():
        if target > 0 and current_weights.get(ticker, 0) == 0:
            # Neukauf – muss über min_trade_value liegen (relativ zum Portfolio)
            # Da wir keine Portfolio-Größe haben, prüfen wir nur, ob target > 0
            # Eine detaillierte Prüfung erfordert die Portfolio-Größe – hier nur Warnung
            if target < 0.005:  # 0.5% Minimalgewicht für Neukäufe
                log.warning(f"Neukauf {ticker} mit sehr geringem Zielgewicht {target:.1%} – könnte unter Mindestordergröße fallen.")
    
    # 4. Zombie-Threshold vs min_trade_value
    assert zombie_threshold <= min_trade_value, \
        f"Zombie threshold (${zombie_threshold:.2f}) > min_trade_value (${min_trade_value:.2f}) – führt zu Konflikten"
    
    log.debug("Portfolio consistency checks passed.")


def assert_no_duplicate_tickers(decisions: List[Dict]) -> None:
    """Prüft, ob ein Ticker mehrfach in Entscheidungen vorkommt."""
    seen = set()
    for d in decisions:
        ticker = d.get("ticker")
        if ticker in seen:
            raise ValueError(f"Duplicate ticker in decisions: {ticker}")
        seen.add(ticker)


# ─────────────────────────────────────────────────────────────
# PORTFOLIO REPORT HELPER (für main)
# ─────────────────────────────────────────────────────────────

def format_portfolio_report(current_weights: Dict, target_weights: Dict, scores: Dict) -> str:
    """Erstellt einen formatierten Portfolio-Report als String."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("📊 PORTFOLIO REPORT")
    lines.append("=" * 70)
    lines.append("\nAKTUELLES PORTFOLIO:")
    lines.append(f"{'Ticker':<8} {'Gewicht':>10} {'Score':>6}")
    for ticker, w in sorted(current_weights.items(), key=lambda x: -x[1])[:10]:
        score = scores.get(ticker, 0)
        lines.append(f"{ticker:<8} {w:>9.1%} {score:>6.0f}")
    lines.append("\nZIELPORTFOLIO (optimiert):")
    lines.append(f"{'Ticker':<8} {'Gewicht':>10} {'Score':>6}")
    for ticker, w in sorted(target_weights.items(), key=lambda x: -x[1])[:10]:
        score = scores.get(ticker, 0)
        lines.append(f"{ticker:<8} {w:>9.1%} {score:>6.0f}")
    lines.append("\nDIFFERENZEN (Ziel - Aktuell > 2%):")
    for ticker, target in target_weights.items():
        current = current_weights.get(ticker, 0.0)
        diff = target - current
        if abs(diff) > 0.02:
            action = "AUFSTOCKEN" if diff > 0 else "REDUZIEREN"
            lines.append(f"{ticker:<8} {action:<10} {diff:>+7.1%}")
    lines.append("=" * 70)
    return "\n".join(lines)
