"""
AI Trading Bot - Portfolio Manager
====================================
Verwaltet das lokale Portfolio: Positionen, Cash, Rebalancing-Berechnung.
Speichert und lädt Daten aus portfolio.json.

Modi:
- DRY:   Liest portfolio.json nur, schreibt NIE (read-only)
- PAPER: Synchronisiert Positionen direkt von Alpaca (immer aktuell)
- REAL:  Wie PAPER, aber mit echtem Geld

FIXES (Production-grade):
  FIX 1 – SINGLE SOURCE OF TRUTH: Alpaca ist Master, keine eigene Positionsschätzung
  FIX 2 – PRE-TRADE VALIDATION: qty_available vom Broker, Cash final gecheckt
  FIX 3 – SOFT REBALANCING: 2% Drift-Schwelle, kein Hard Reset auf 0%
  FIX 4 – 2-PHASE EXECUTION: sync_from_broker() nach SELL-Phase aufrufen
  NEU  – Dynamische Mindestordergröße für kleine Konten (5% des Portfolios, min $10)
"""

import copy
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

from logger import log, trade_logger
from config import PORTFOLIO_FILE, INITIAL_CAPITAL
from utils import save_json_file, load_json_file, format_currency, pct_change


@dataclass
class Position:
    """Repräsentiert eine einzelne Portfolio-Position."""
    ticker: str
    quantity: float
    avg_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    allocation_pct: float
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


class PortfolioManager:
    """
    Verwaltet das lokale Portfolio.

    mode="DRY"   -> portfolio.json nur lesen, nie schreiben
    mode="PAPER" -> Positionen von Alpaca laden (live sync)
    mode="REAL"  -> Positionen von Alpaca laden (live sync, echtes Geld)
    """

    # FIX 3: Soft Rebalancing – nur handeln wenn Allokations-Drift > Schwelle
    DRIFT_THRESHOLD = 0.02  # 2% – vermeidet Stress-Trading bei kleinen Abweichungen

    def __init__(
        self,
        portfolio_file: str = PORTFOLIO_FILE,
        initial_capital: float = INITIAL_CAPITAL,
        mode: str = "DRY",
        alpaca_api=None,
    ):
        self.portfolio_file = portfolio_file
        self.initial_capital = initial_capital
        self.mode = mode.upper()
        self.alpaca_api = alpaca_api
        self.positions: Dict[str, Dict] = {}
        self.cash: float = initial_capital
        self.read_only = (self.mode == "DRY")

        if self.mode in ("PAPER", "REAL", "LIVE") and self.alpaca_api:
            try:
                self._load_from_alpaca()
            except Exception as e:
                log.warning(f"Alpaca init fehlgeschlagen -> fallback local: {e}")
                self.load()
        else:
            self.load()

    # --- Laden & Speichern ---------------------------------------------------

    def load(self):
        """Ladet Portfolio aus JSON-Datei (DRY Modus: nur lesen)."""
        data = load_json_file(self.portfolio_file, default=None)
        if data is None:
            log.info(f"Kein Portfolio gefunden, starte mit ${self.initial_capital:,.0f} Cash.")
            self.positions = {}
            self.cash = self.initial_capital
            if not self.read_only:
                self.save()
        else:
            self.positions = data.get("positions", {})
            self.cash = data.get("cash", self.initial_capital)

            if self.read_only:
                log.info("[DRY MODE] Portfolio geladen (read-only)")
            mode_hint = " [READ-ONLY, keine Aenderungen]" if self.read_only else ""
            log.info(f"Portfolio geladen: {len(self.positions)} Positionen, "
                     f"{format_currency(self.cash)} Cash{mode_hint}.")

    def save(self):
        if self.read_only:
            log.debug("DRY MODE: save skipped")
            return

        save_json_file(self.portfolio_file, {
            "cash": self.cash,
            "positions": self.positions,
            "last_updated": datetime.now().isoformat()
        })

    def _load_from_alpaca(self):
        """
        FIX 1 – SINGLE SOURCE OF TRUTH:
        Laedt Positionen und Cash direkt von Alpaca.
        """
        try:
            account = self.alpaca_api.get_account()
            self.cash = float(account.cash)

            local_data = load_json_file(self.portfolio_file, default=None)
            if local_data and "initial_capital" in local_data:
                self.initial_capital = local_data["initial_capital"]
            else:
                self.initial_capital = float(account.portfolio_value)

            positions = self.alpaca_api.list_positions()
            self.positions = {}
            for pos in positions:
                ticker = pos.symbol
                qty_available = getattr(pos, "qty_available", None)
                if qty_available is None:
                    qty_available = pos.qty
                qty_available = float(qty_available)
                qty_total = float(pos.qty)
                avg_price = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                market_value = float(pos.market_value)
                cost_basis = float(pos.cost_basis)
                unrealized_pnl = float(pos.unrealized_pl)
                unrealized_pnl_pct = float(pos.unrealized_plpc) * 100

                self.positions[ticker] = {
                    "ticker": ticker,
                    "quantity": qty_total,
                    "qty_available": qty_available,
                    "avg_price": avg_price,
                    "current_price": current_price,
                    "market_value": market_value,
                    "cost_basis": cost_basis,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": unrealized_pnl_pct,
                    "entry_date": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                }

            log.info(
                f"[BROKER SYNC] Portfolio von Alpaca geladen: "
                f"{len(self.positions)} Positionen, {format_currency(self.cash)} Cash."
            )
            self.save()

        except Exception as e:
            log.warning(f"Alpaca Portfolio-Sync fehlgeschlagen: {e} – lade lokales Portfolio.")
            self.load()

    def sync_from_broker(self) -> bool:
        """Expliziter Broker-Sync nach der SELL-Phase."""
        if not self.alpaca_api or self.mode not in ("PAPER", "REAL", "LIVE"):
            log.debug("Broker-Sync nicht noetig (DRY Modus).")
            return False
        try:
            time.sleep(2.0)
            self._load_from_alpaca()
            log.info("[BROKER SYNC] Portfolio nach SELL-Phase resynchronisiert.")
            return True
        except Exception as e:
            log.warning(f"Broker-Sync fehlgeschlagen: {e}")
            return False

    # --- Portfolio-Zustand ---------------------------------------------------

    def update_prices(self, market_data: Dict[str, Dict]):
        """Aktualisiert aktuelle Preise aller Positionen aus Marktdaten."""
        for ticker, pos in self.positions.items():
            if ticker in market_data:
                price = market_data[ticker].get("current_price")
                if price:
                    pos["current_price"] = price
                    pos["market_value"] = pos["quantity"] * price
                    pos["unrealized_pnl"] = pos["market_value"] - pos["cost_basis"]
                    pos["unrealized_pnl_pct"] = pct_change(pos["cost_basis"], pos["market_value"])
                    pos["last_updated"] = datetime.now().isoformat()
        log.debug("Portfoliopreise aktualisiert.")

    def get_total_value(self) -> float:
        invested = sum(p.get("market_value", 0) for p in self.positions.values())
        return self.cash + invested

    def get_invested_value(self) -> float:
        return sum(p.get("market_value", 0) for p in self.positions.values())

    def get_allocations(self) -> Dict[str, float]:
        total = self.get_total_value()
        if total == 0:
            return {}
        allocs = {"CASH": self.cash / total}
        for ticker, pos in self.positions.items():
            allocs[ticker] = pos.get("market_value", 0) / total
        return allocs

    def get_summary(self) -> Dict:
        total = self.get_total_value()
        invested = self.get_invested_value()
        total_pnl = sum(p.get("unrealized_pnl", 0) for p in self.positions.values())
        pnl_pct = pct_change(self.initial_capital, total) if total > 0 else 0
        allocs = self.get_allocations()
        for ticker, pos in self.positions.items():
            pos["allocation_pct"] = round(allocs.get(ticker, 0.0) * 100, 2)
        return {
            "total_value": round(total, 2),
            "cash": round(self.cash, 2),
            "cash_pct": round(self.cash / total * 100, 2) if total else 0,
            "invested": round(invested, 2),
            "initial_capital": self.initial_capital,
            "unrealized_pnl": round(total_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "n_positions": len(self.positions),
            "positions": self.positions,
            "mode": self.mode,
        }

    # ─── TRAILING STOP SYSTEM ──────────────────────────────────────────────────
    
    def get_trailing_stop_triggers(
        self,
        market_data: Dict[str, Dict],
        trail_pct: float = 0.12,
    ) -> List[Dict]:
        """
        Trailing Stop Engine: Generiert automatische Sell-Signale bei Profit-Takes.
        Bedingung: unrealized_gain > 8% und current_price < high_52w * (1 - trail_pct)
        """
        trailing_sells = []
        for ticker, pos in self.positions.items():
            if pos.get("quantity", 0) <= 0:
                continue
            unrealized_pnl_pct = pos.get("unrealized_pnl_pct", 0.0)
            if unrealized_pnl_pct <= 0.08:
                continue
            ticker_data = market_data.get(ticker, {})
            high_52w = ticker_data.get("high_52w")
            current_price = ticker_data.get("current_price") or pos.get("current_price", 0)
            if high_52w is None or current_price <= 0:
                continue
            trail_level = high_52w * (1 - trail_pct)
            if current_price < trail_level:
                trailing_sells.append({
                    "ticker": ticker,
                    "action": "SELL",
                    "target_allocation": 0.0,
                    "confidence": 0.95,
                    "reason": f"Trailing Stop: {unrealized_pnl_pct:+.1%} gain, price ${current_price:.2f} < trail ${trail_level:.2f}",
                    "risk_approved": True,
                    "trailing_stop": True,
                })
                log.info(f"  Trailing Stop Trigger: {ticker} | Gain: {unrealized_pnl_pct:+.1%}")
        return trailing_sells

    # ─── STALE POSITION DETECTION ─────────────────────────────────────────────
    def stale_position_flag(
        self,
        stale_gain_low: float = -0.02,
        stale_gain_high: float = 0.02,
        min_hold_days: int = 15,
    ) -> Dict[str, Dict]:
        """Markiert Positionen als 'stale' (nur Flag, kein erzwungener Verkauf)."""
        stale_flags = {}
        today = datetime.now()
        for ticker, pos in self.positions.items():
            pnl_pct_raw = pos.get("unrealized_pnl_pct", 0.0)
            pnl_pct = pnl_pct_raw / 100.0 if abs(pnl_pct_raw) > 1.0 else pnl_pct_raw
            hold_days = 0
            entry_date_str = pos.get("entry_date")
            if entry_date_str:
                try:
                    entry_dt = datetime.fromisoformat(entry_date_str)
                    hold_days = max(0, (today - entry_dt).days)
                except Exception:
                    hold_days = 0
            is_stale = (stale_gain_low <= pnl_pct <= stale_gain_high and hold_days >= min_hold_days)
            stale_flags[ticker] = {
                "stale": is_stale,
                "pnl_pct": round(pnl_pct * 100, 2),
                "hold_days": hold_days,
            }
            if is_stale:
                log.info(f"[STALE FLAG] {ticker}: {pnl_pct:+.1%} unrealized, {hold_days}d held")
        return stale_flags

    def print_summary(self):
        total = self.get_total_value()
        log.info("=" * 60)
        log.info(f"PORTFOLIO ({self.mode})")
        log.info(f"  Gesamtwert:  {format_currency(total)}")
        log.info(f"  Cash:        {format_currency(self.cash)} ({self.cash/total*100:.1f}%)")
        log.info(f"  P&L:         {format_currency(total - self.initial_capital)} "
                 f"({pct_change(self.initial_capital, total):+.2f}%)")
        log.info(f"  Positionen:  {len(self.positions)}")
        for ticker, pos in self.positions.items():
            alloc = pos.get("market_value", 0) / total * 100 if total else 0
            log.info(
                f"    {ticker:<6} {pos.get('quantity', 0):.4f} Stk @ "
                f"${pos.get('avg_price', 0):.2f} -> "
                f"{format_currency(pos.get('market_value', 0))} | "
                f"({alloc:.1f}%) | "
                f"P&L: {format_currency(pos.get('unrealized_pnl', 0))}"
            )
        log.info("=" * 60)

    # --- Trade-Ausfuehrung (lokal) -------------------------------------------

    def apply_trade(self, ticker: str, action: str, quantity: float, price: float):
        if self.read_only:
            log.info(f"[DRY] Simuliert: {action} {ticker} {quantity:.4f} Stk @ ${price:.2f}")
            return True

        trade_value = quantity * price

        if action == "BUY":
            if trade_value > self.cash:
                log.warning(f"BUY {ticker} blockiert: benötigt {format_currency(trade_value)}, verfügbar {format_currency(self.cash)}")
                return False

            from config import RISK_SETTINGS, ACTIVE_RISK_PROFILE
            min_cash_pct = RISK_SETTINGS[ACTIVE_RISK_PROFILE]["min_cash_pct"]
            total_value = self.get_total_value()
            min_cash_required = total_value * min_cash_pct
            cash_after_trade = self.cash - trade_value
            if cash_after_trade < min_cash_required:
                max_spendable = self.cash - min_cash_required
                if max_spendable < 100.0:
                    log.warning(f"BUY {ticker} blockiert: Cash würde unter Minimum fallen")
                    return False
                quantity = round(max_spendable / price, 6)
                trade_value = quantity * price
                log.info(f"BUY {ticker} auf max. {format_currency(trade_value)} reduziert")

            self.cash -= trade_value
            if ticker in self.positions:
                pos = self.positions[ticker]
                old_qty = pos["quantity"]
                old_cost = pos["cost_basis"]
                new_qty = old_qty + quantity
                pos["quantity"] = new_qty
                pos["qty_available"] = new_qty
                pos["cost_basis"] = old_cost + trade_value
                pos["avg_price"] = pos["cost_basis"] / new_qty
                pos["current_price"] = price
                pos["market_value"] = new_qty * price
            else:
                self.positions[ticker] = {
                    "ticker": ticker,
                    "quantity": quantity,
                    "qty_available": quantity,
                    "avg_price": price,
                    "current_price": price,
                    "market_value": trade_value,
                    "cost_basis": trade_value,
                    "unrealized_pnl": 0.0,
                    "unrealized_pnl_pct": 0.0,
                    "entry_date": datetime.now().isoformat(),
                }
            log.info(f"Position {ticker} GEKAUFT: {quantity:.6f} Stk @ ${price:.2f}")

        elif action == "SELL":
            if ticker not in self.positions:
                log.warning(f"Kann {ticker} nicht verkaufen – keine Position.")
                return False
            pos = self.positions[ticker]
            available = pos.get("qty_available", pos["quantity"])
            sell_qty = round(min(quantity, available), 6)
            if sell_qty <= 0:
                log.warning(f"SELL {ticker}: keine verfügbare Menge")
                return False
            sell_value = sell_qty * price
            self.cash += sell_value
            pos["quantity"] -= sell_qty
            pos["qty_available"] = round(max(0.0, available - sell_qty), 6)
            pos["cost_basis"] = pos["avg_price"] * pos["quantity"]
            pos["current_price"] = price
            pos["market_value"] = pos["quantity"] * price
            from utils import ZOMBIE_POSITION_THRESHOLD
            residual_value = pos["quantity"] * pos["current_price"]
            if pos["quantity"] <= 0.0001 or residual_value < ZOMBIE_POSITION_THRESHOLD:
                del self.positions[ticker]
                log.info(f"Position {ticker} vollständig geschlossen.")
            else:
                log.info(f"Position {ticker} reduziert: {sell_qty:.6f} Stk verkauft @ ${price:.2f}")

        self.save()
        return True

    # --- Rebalancing-Berechnung (verbessert) ---------------------------------

    def calculate_rebalancing_trades(
        self,
        target_allocations: Dict[str, float],
        current_prices: Dict[str, float],
        decisions_map: Dict[str, Dict] = None,
        min_trade_value: float = 10.0,   # jetzt nur noch "execution safety floor"
    ) -> List[Dict]:
        """
        Berechnet Trades basierend auf Zielallokationen.
        min_trade_value ist nur ein Sicherheits-Floor, nicht die primäre Logik.
        """
        decisions_map = decisions_map or {}
        total_value = self.get_total_value()
        current_allocs = self.get_allocations()
        trades = []

        # Dynamische Mindestordergröße als Prozentsatz (0.5% des Portfolios, aber mindestens 10 USD)
        min_trade_abs = max(10.0, total_value * 0.005)

        for ticker, target_alloc in target_allocations.items():
            if ticker == "CASH":
                continue

            price = current_prices.get(ticker)
            if not price or price <= 0:
                log.warning(f"Kein Preis für {ticker}, überspringe.")
                continue

            current_value = self.positions.get(ticker, {}).get("market_value", 0)
            target_value = total_value * target_alloc
            diff_value = target_value - current_value

            if abs(diff_value) < min_trade_abs:
                # nur loggen, nicht ignorieren – aber trotzdem Trade generieren,
                # wenn die relative Abweichung signifikant ist (> 2% des Portfolios)
                if abs(target_alloc - current_allocs.get(ticker, 0)) < 0.02:
                    continue

            action = "BUY" if diff_value > 0 else "SELL"
            quantity = abs(diff_value) / price
            actual_value = abs(diff_value)

            trades.append({
                "ticker": ticker,
                "action": action,
                "quantity": round(quantity, 6),
                "price": price,
                "value": round(actual_value, 2),
                "current_alloc": round(current_allocs.get(ticker, 0), 4),
                "target_alloc": round(target_alloc, 4),
                "diff_value": round(diff_value, 2),
                "ai_action": decisions_map.get(ticker, {}).get("action", action),
                "ai_reason": decisions_map.get(ticker, {}).get("reason", ""),
                "ai_confidence": decisions_map.get(ticker, {}).get("confidence", 0.7),
                "decision_id": decisions_map.get(ticker, {}).get("decision_id"),
            })

        # SELLs zuerst
        trades.sort(key=lambda t: 0 if t["action"] == "SELL" else 1)
        return trades

    def simulate_trade_plan(self, trades: List[Dict], current_prices: Dict[str, float], min_cash_pct: float = 0.0) -> Dict:
        working_positions = copy.deepcopy(self.positions)
        cash = self.cash
        total_value = cash + sum(p.get("market_value", 0) for p in working_positions.values())
        min_cash = max(0.0, total_value * min_cash_pct)

        for trade in sorted(trades, key=lambda x: 0 if x["action"] == "SELL" else 1):
            ticker = trade["ticker"]
            price = current_prices.get(ticker, trade.get("price", 0))
            if price <= 0:
                continue
            if trade["action"] == "SELL":
                pos = working_positions.get(ticker)
                if not pos:
                    continue
                quantity = min(trade.get("quantity", 0), pos.get("quantity", 0))
                if quantity <= 0:
                    continue
                value = round(quantity * price, 2)
                cash += value
                pos["quantity"] = round(pos.get("quantity", 0) - quantity, 6)
                pos["current_price"] = price
                pos["market_value"] = round(pos["quantity"] * price, 2)
                pos["cost_basis"] = round(pos.get("avg_price", 0) * pos["quantity"], 2)
                if pos["quantity"] <= 0 or pos["market_value"] < 1.0:
                    working_positions.pop(ticker, None)
            else:  # BUY
                spendable = max(0.0, cash - min_cash)
                order_value = min(trade.get("value", 0), spendable)
                if order_value < 1.0:
                    continue
                quantity = round(order_value / price, 6)
                if quantity <= 0:
                    continue
                value = round(quantity * price, 2)
                cash -= value
                pos = working_positions.get(ticker)
                if pos:
                    cost_basis = pos.get("cost_basis", 0) + value
                    quantity_total = pos.get("quantity", 0) + quantity
                    pos.update({
                        "quantity": quantity_total, "qty_available": quantity_total,
                        "cost_basis": round(cost_basis, 2), "avg_price": round(cost_basis / quantity_total, 6),
                        "current_price": price, "market_value": round(quantity_total * price, 2),
                    })
                else:
                    working_positions[ticker] = {
                        "ticker": ticker, "quantity": quantity, "qty_available": quantity, "avg_price": price,
                        "current_price": price, "market_value": round(value, 2), "cost_basis": round(value, 2),
                        "unrealized_pnl": 0.0, "unrealized_pnl_pct": 0.0, "entry_date": datetime.now().isoformat(),
                    }

        invested = sum(p.get("market_value", 0) for p in working_positions.values())
        total_after = round(cash + invested, 2)
        pnl_pct = pct_change(self.initial_capital, total_after) if total_after else 0
        return {
            "total_value": total_after, "cash": round(cash, 2), "cash_pct": round(cash / total_after * 100, 2) if total_after else 0,
            "invested": round(invested, 2), "initial_capital": self.initial_capital,
            "unrealized_pnl": round(sum(p.get("market_value", 0) - p.get("cost_basis", 0) for p in working_positions.values()), 2),
            "pnl_pct": round(pnl_pct, 2), "n_positions": len(working_positions), "positions": working_positions,
            "mode": self.mode, "assumed_min_cash_pct": min_cash_pct,
        }
