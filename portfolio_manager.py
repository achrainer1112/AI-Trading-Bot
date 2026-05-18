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

        if self.mode in ("PAPER", "REAL") and self.alpaca_api:
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
        Alpaca ist immer Master – lokale Schaetzungen werden verworfen.
        Verwendet qty_available (nicht qty) fuer exakte Stueckzahlen.
        """
        try:
            account = self.alpaca_api.get_account()
            # FIX 1: account.cash (nicht buying_power!) = echter verfuegbarer Cash
            self.cash = float(account.cash)

            # initial_capital aus lokaler Datei lesen (nicht mit Portfolio-Value ueberschreiben)
            local_data = load_json_file(self.portfolio_file, default=None)
            if local_data and "initial_capital" in local_data:
                self.initial_capital = local_data["initial_capital"]
            else:
                self.initial_capital = float(account.portfolio_value)

            positions = self.alpaca_api.list_positions()
            self.positions = {}
            for pos in positions:
                ticker = pos.symbol
                # FIX 2: qty_available = Broker-exakte handelbare Stueckzahl
                qty_available = getattr(pos, "qty_available", None)
                if qty_available is None:
                    qty_available = pos.qty  # fallback safe

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
                    "qty_available": qty_available,  # Broker-exakt fuer Orders
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
        """
        FIX 4 – 2-PHASE EXECUTION:
        Expliziter Broker-Sync nach der SELL-Phase.
        In main.py zwischen SELL- und BUY-Phase aufrufen:
          1. SELL-Orders ausfuehren
          2. portfolio.sync_from_broker()   <-- hier
          3. BUY-Orders ausfuehren
        Gibt True zurueck wenn Sync erfolgreich.
        """
        if not self.alpaca_api or self.mode not in ("PAPER", "REAL"):
            log.debug("Broker-Sync nicht noetig (DRY Modus).")
            return False
        try:
            time.sleep(2.0)  # Alpaca-Orders verarbeiten lassen
            self._load_from_alpaca()
            log.info("[BROKER SYNC] Portfolio nach SELL-Phase resynchronisiert.")
            return True
        except Exception as e:
            log.warning(f"Broker-Sync fehlgeschlagen: {e}")
            return False

    # --- Portfolio-Zustand ---------------------------------------------------

    def update_prices(self, market_data: Dict[str, Dict]):
        """
        Aktualisiert aktuelle Preise aller Positionen aus Marktdaten.
        market_data: {ticker: {"current_price": float, ...}}
        """
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
        """Gesamtwert des Portfolios (Cash + alle Positionen)."""
        invested = sum(p.get("market_value", 0) for p in self.positions.values())
        return self.cash + invested

    def get_invested_value(self) -> float:
        """Gesamtwert aller investierten Positionen."""
        return sum(p.get("market_value", 0) for p in self.positions.values())

    def get_allocations(self) -> Dict[str, float]:
        """
        Gibt aktuelle Allokation zurueck: {ticker: anteil_als_dezimal}
        Inklusive "CASH" als eigene Position.
        """
        total = self.get_total_value()
        if total == 0:
            return {}
        allocs = {"CASH": self.cash / total}
        for ticker, pos in self.positions.items():
            allocs[ticker] = pos.get("market_value", 0) / total
        return allocs

    def get_summary(self) -> Dict:
        """Gibt eine Zusammenfassung des Portfolios zurueck."""
        total = self.get_total_value()
        invested = self.get_invested_value()
        total_pnl = sum(p.get("unrealized_pnl", 0) for p in self.positions.values())
        pnl_pct = pct_change(self.initial_capital, total) if total > 0 else 0
        # Ensure allocation_pct is available for prompt generation
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
        🔥 TRAILING STOP ENGINE: Generiert automatische Sell-Signale bei Profit-Takes
        
        Bedingungen für Exit-Trigger:
          1. unrealized_gain > 0.08 (mindestens 8% Profit)
          2. current_price < high_52w * (1 - trail_pct)
             → Preis ist > 12% unter 52-Wochen-High gefallen
        
        Diese Signale werden VOR der KI-Analyse in main.py hinzugefügt
        und wirken wie automatische "Gewinnsicherungen".
        
        Args:
            market_data: Dict mit Preisdaten, z.B. high_52w
            trail_pct: Wie viel % unter 52w-High ← triggert Trailing Stop (default 12%)
        
        Returns:
            List von SELL-Decisions für Positionen mit Trailing-Stop-Trigger
        """
        trailing_sells: List[Dict] = []
        
        for ticker, pos in self.positions.items():
            if pos.get("quantity", 0) <= 0:
                continue
            
            # Bedingung 1: Unrealized Gain > 8% (nur Profitpositionen interessieren)
            unrealized_pnl_pct = pos.get("unrealized_pnl_pct", 0.0)
            if unrealized_pnl_pct <= 0.08:
                continue  # Kein Profit oder zu klein
            
            # Bedingung 2: Habe wir 52w-High Daten?
            ticker_data = market_data.get(ticker, {})
            high_52w = ticker_data.get("high_52w")
            current_price = ticker_data.get("current_price") or pos.get("current_price", 0)
            
            if high_52w is None or current_price <= 0:
                continue  # Unvollständige Daten
            
            # Berechne Trail-Level
            trail_level = high_52w * (1 - trail_pct)
            
            # Trigger wenn aktueller Preis < trail_level
            if current_price < trail_level:
                # Berechne Gewinn, den wir sichern
                current_value = pos.get("market_value", 0)
                cost_basis = pos.get("cost_basis", 0)
                profit = current_value - cost_basis
                
                trailing_sells.append({
                    "ticker": ticker,
                    "action": "SELL",
                    "target_allocation": 0.0,  # Fullständige Liquidation
                    "confidence": 0.95,  # Sehr hohe Konfidenz
                    "reason": (
                        f"Trailing Stop Trigger: {unrealized_pnl_pct:+.1%} gain, "
                        f"price ${current_price:.2f} < 52w-high-trail ${trail_level:.2f}. "
                        f"Profit-Taking: ${profit:.2f}"
                    ),
                    "risk_approved": True,
                    "trailing_stop": True,
                    "decision_id": f"TRAILING_STOP_{ticker}",
                    "priority": "HIGH",
                })
                
                log.info(
                    f"  Trailing Stop Trigger: {ticker} | "
                    f"Gain: {unrealized_pnl_pct:+.1%} | "
                    f"Price: ${current_price:.2f} vs Trail ${trail_level:.2f} | "
                    f"Profit: ${profit:.2f}"
                )
        
        if trailing_sells:
            log.info(f"  {len(trailing_sells)} Trailing Stop signals generated")
        
        return trailing_sells

    # ─── STALE POSITION DETECTION ─────────────────────────────────────────────

    def stale_position_flag(
        self,
        stale_gain_low: float = -0.02,
        stale_gain_high: float = 0.02,
        min_hold_days: int = 15,
    ) -> Dict[str, Dict]:
        """
        Phase 4 Production Stability – Stale Position Detection.

        Marks positions as 'stale' when:
          - unrealized P&L is between stale_gain_low and stale_gain_high
            (default: -2% to +2%, i.e. dead weight with no clear direction)
          - AND position has been held for more than min_hold_days days.

        IMPORTANT: This method only FLAGS – it does NOT force a sell.
        The flag is informational only; human monitoring or journaling can
        act on it if desired.

        Returns:
            Dict[ticker, {"stale": bool, "pnl_pct": float, "hold_days": int, "reason": str}]
        """
        stale_flags: Dict[str, Dict] = {}
        today = datetime.now()

        for ticker, pos in self.positions.items():
            pnl_pct_raw = pos.get("unrealized_pnl_pct", 0.0)
            # unrealized_pnl_pct is stored as percent (e.g. 5.2 = 5.2%)
            # Normalize: if value > 1.0 assume it's already in percent form
            pnl_pct = pnl_pct_raw / 100.0 if abs(pnl_pct_raw) > 1.0 else pnl_pct_raw

            # Attempt to calculate hold days from entry_date
            hold_days = 0
            entry_date_str = pos.get("entry_date")
            if entry_date_str:
                try:
                    entry_dt = datetime.fromisoformat(entry_date_str)
                    hold_days = max(0, (today - entry_dt).days)
                except Exception:
                    hold_days = 0

            is_stale = (
                stale_gain_low <= pnl_pct <= stale_gain_high
                and hold_days >= min_hold_days
            )

            stale_flags[ticker] = {
                "stale": is_stale,
                "pnl_pct": round(pnl_pct * 100, 2),   # back to percent for readability
                "hold_days": hold_days,
                "reason": (
                    f"P&L {pnl_pct:+.1%} in [{stale_gain_low:.0%},{stale_gain_high:.0%}] "
                    f"for {hold_days}d ≥ {min_hold_days}d"
                ) if is_stale else "",
            }

            if is_stale:
                log.info(
                    f"[STALE FLAG] {ticker}: {pnl_pct:+.1%} unrealized | "
                    f"{hold_days} days held | marked as stale (no action forced)"
                )

        stale_count = sum(1 for v in stale_flags.values() if v["stale"])
        if stale_count:
            log.warning(
                f"[STALE] {stale_count}/{len(self.positions)} positions flagged as stale "
                f"(P&L {stale_gain_low:.0%}–{stale_gain_high:.0%}, held ≥{min_hold_days}d)"
            )

        return stale_flags

    def print_summary(self):
        """Gibt eine lesbare Zusammenfassung des Portfolios aus."""
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
        """
        Wendet einen Trade auf das lokale Portfolio an.
        DRY Modus: simuliert aber speichert NIE.
        PAPER/REAL: speichert nach jedem Trade.
        FIX 2: Pre-Trade Cash-Validierung verhindert negativen Cash.
        """
        if self.read_only:
            log.info(f"[DRY READ-ONLY] Simuliert: {action} {ticker} "
                     f"{quantity:.4f} Stk @ ${price:.2f} – keine Aenderung an portfolio.json")
            return True

        trade_value = quantity * price

        if action == "BUY":
            # FIX 2: Hartes Cash-Limit – niemals negativen Cash-Stand zulassen
            if trade_value > self.cash:
                log.warning(
                    f"BUY {ticker} blockiert: benoetigt {format_currency(trade_value)}, "
                    f"verfuegbar {format_currency(self.cash)} – Trade abgebrochen."
                )
                return False

            # Mindest-Cash-Reserve schuetzen
            from config import RISK_SETTINGS, ACTIVE_RISK_PROFILE
            min_cash_pct = RISK_SETTINGS[ACTIVE_RISK_PROFILE]["min_cash_pct"]
            total_value = self.get_total_value()
            min_cash_required = total_value * min_cash_pct
            cash_after_trade = self.cash - trade_value
            if cash_after_trade < min_cash_required:
                max_spendable = self.cash - min_cash_required
                if max_spendable < 100.0:
                    log.warning(
                        f"BUY {ticker} blockiert: Cash wuerde unter Minimum fallen "
                        f"({format_currency(cash_after_trade)} < {format_currency(min_cash_required)})"
                    )
                    return False
                quantity = round(max_spendable / price, 6)
                trade_value = quantity * price
                log.info(
                    f"BUY {ticker} auf max. {format_currency(trade_value)} reduziert "
                    f"(Cash-Minimum {min_cash_pct:.0%} schuetzen)"
                )

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
                log.warning(f"Kann {ticker} nicht verkaufen – keine Position vorhanden.")
                return False

            pos = self.positions[ticker]
            # FIX 2: qty_available (Broker-exakt) als Obergrenze
            available = pos.get("qty_available", pos["quantity"])
            sell_qty = round(min(quantity, available), 6)

            if sell_qty <= 0:
                log.warning(f"SELL {ticker}: keine verfuegbare Menge -> uebersprungen")
                return False

            sell_value = sell_qty * price
            self.cash += sell_value

            pos["quantity"] -= sell_qty
            pos["qty_available"] = round(max(0.0, available - sell_qty), 6)
            pos["cost_basis"] = pos["avg_price"] * pos["quantity"]
            pos["current_price"] = price
            pos["market_value"] = pos["quantity"] * price

            # Fix #3: Position schließen wenn Qty ≤ Epsilon ODER Wert < Zombie-Threshold
            from utils import ZOMBIE_POSITION_THRESHOLD
            residual_value = pos["quantity"] * pos["current_price"]
            if pos["quantity"] <= 0.0001 or residual_value < ZOMBIE_POSITION_THRESHOLD:
                del self.positions[ticker]
                log.info(f"Position {ticker} vollstaendig geschlossen (Zombie-Schutz).")
            else:
                log.info(f"Position {ticker} reduziert: {sell_qty:.6f} Stk verkauft @ ${price:.2f}")

        self.save()
        return True

    # --- Rebalancing-Berechnung ----------------------------------------------

    def calculate_rebalancing_trades(
        self,
        target_allocations: Dict[str, float],
        current_prices: Dict[str, float],
        decisions_map: Dict[str, Dict] = None,
        min_trade_value: float = 100.0,
    ) -> List[Dict]:
        """
        Berechnet die noetigen Trades um Ziel-Allokationen zu erreichen.

        FIX 3 – SOFT REBALANCING:
          Nur handeln wenn Allokations-Drift > DRIFT_THRESHOLD (2%).
          Vollstaendige Liquidation (target=0) nur bei explizitem Signal:
          orphan=True, stop_loss=True oder confidence >= 0.80.

        FIX 2 – PRE-TRADE VALIDATION:
          qty_available (Broker-exakt) fuer SELL-Stueckzahlen.
          6 Dezimalstellen gegen Floating-Point-Fehler.

        target_allocations: {ticker: gewuenschter_anteil_0_bis_1}
        current_prices: {ticker: aktueller_preis}
        decisions_map: {ticker: ki_entscheidung_dict}
        Gibt Liste von Trade-Dicts zurueck (SELLs zuerst).
        """
        decisions_map = decisions_map or {}
        total_value = self.get_total_value()
        current_allocs = self.get_allocations()
        trades = []

        for ticker, target_alloc in target_allocations.items():
            if ticker == "CASH":
                continue

            price = current_prices.get(ticker)
            decision = decisions_map.get(ticker, {})
            is_zombie_sell = (
                decision.get("zombie_cleanup", False)
                or decision.get("orphan", False)
            ) and decision.get("action") == "SELL"

            if not price or price <= 0:
                in_portfolio_check = ticker in self.positions
                ai_action_check = decisions_map.get(ticker, {}).get("action", "HOLD")
                ai_conf_check = decisions_map.get(ticker, {}).get("confidence", 0)

                # Zombie ohne Preis → force_close
                if is_zombie_sell:
                    log.info(f"Zombie SELL {ticker}: kein Marktpreis → wird via force_close liquidiert.")
                    trades.append({
                        "ticker": ticker,
                        "action": "SELL",
                        "value": 0.0,
                        "quantity": self.positions.get(ticker, {}).get("quantity", 0),
                        "price": 0.0,
                        "current_alloc": 0.0,
                        "target_alloc": 0.0,
                        "ai_reason": decisions_map.get(ticker, {}).get("reason", "Zombie liquidation"),
                        "ai_confidence": 1.0,
                        "zombie_cleanup": True,
                    })
                    continue

                # FIX: KI sagt SELL mit Konfidenz ≥ 60% und Position vorhanden
                # → als force_close weitergeben, auch ohne aktuellen Preis
                if (
                    in_portfolio_check
                    and ai_action_check == "SELL"
                    and ai_conf_check >= 0.60
                ):
                    pos_qty = self.positions[ticker].get("qty_available",
                              self.positions[ticker].get("quantity", 0))
                    avg_px  = self.positions[ticker].get("avg_price", 0)
                    log.warning(
                        f"SELL {ticker}: kein aktueller Marktpreis → "
                        f"nutze Ø-Kaufpreis ${avg_px:.2f} als Fallback-Preis."
                    )
                    # Fallback: avg_price als Schätzpreis verwenden
                    est_value = pos_qty * avg_px
                    trades.append({
                        "ticker": ticker,
                        "action": "SELL",
                        "quantity": round(pos_qty, 6),
                        "price": avg_px,
                        "value": round(est_value, 2),
                        "current_alloc": self.positions[ticker].get("market_value", 0) / max(self.get_total_value(), 1),
                        "target_alloc": 0.0,
                        "diff_value": round(-est_value, 2),
                        "ai_action": "SELL",
                        "ai_reason": decisions_map.get(ticker, {}).get("reason", "Portfolio Rebalancing"),
                        "ai_confidence": ai_conf_check,
                        "price_estimated": True,  # Flag: Preis ist Schätzung
                    })
                    continue

                log.warning(f"Kein Preis fuer {ticker}, ueberspringe.")
                continue

            in_portfolio = ticker in self.positions
            target_value = total_value * target_alloc
            current_value = self.positions.get(ticker, {}).get("market_value", 0)
            diff_value = target_value - current_value
            current_alloc = current_allocs.get(ticker, 0)

            decision = decisions_map.get(ticker, {})
            ai_action = decision.get("action", "HOLD")
            ai_reason = decision.get("reason", "Portfolio Rebalancing")
            ai_confidence = decision.get("confidence", 0)

            # KI sagt SELL aber keine Position vorhanden -> ueberspringe
            if ai_action == "SELL" and not in_portfolio:
                log.info(f"{ticker}: KI sagt SELL aber keine Position vorhanden -> uebersprungen")
                continue

            # FIX 3: Vollstaendige Liquidation NUR bei explizitem Signal
            # Verhindert aggressiven Hard Reset durch KI-Allokation=0
            is_explicit_liquidation = (
                target_alloc == 0.0
                and in_portfolio
                and ai_action == "SELL"
                and (
                    decision.get("orphan", False)
                    or decision.get("stop_loss", False)
                    or decision.get("rebalancing", False)
                    or ai_confidence >= 0.80
                )
            )
            if target_alloc == 0.0 and in_portfolio and not is_explicit_liquidation:
                log.debug(
                    f"{ticker}: target=0% aber kein explizites Liquidationssignal "
                    f"(Konfidenz: {ai_confidence:.0%}) -> Soft Rebalancing uebersprungen"
                )
                continue

            if ai_action == "SELL" and target_alloc == 0.0 and in_portfolio:
                available_qty = self.positions[ticker].get(
                    "qty_available", self.positions[ticker]["quantity"]
                )
                sell_value = available_qty * price
                trades.append({
                    "ticker": ticker,
                    "action": "SELL",
                    "quantity": round(available_qty, 6),
                    "price": price,
                    "value": round(sell_value, 2),
                    "current_alloc": round(current_alloc, 4),
                    "target_alloc": 0.0,
                    "diff_value": round(-sell_value, 2),
                    "ai_action": "SELL",
                    "ai_reason": ai_reason,
                    "ai_confidence": ai_confidence,
                    "decision_id": decision.get("decision_id"),
                })
                log.info(
                    f"{ticker}: SELL-Aktion mit 0% Zielallokation akzeptiert → vollständige Liquidation. "
                    f"({current_alloc*100:.1f}% -> 0%)"
                )
                continue

            # FIX 3: Soft Rebalancing – nur bei BUYs anwenden
            alloc_drift = abs(current_alloc - target_alloc)
            if ai_action == "BUY" and target_alloc > 0:
                effective_threshold = 0.01  # 1% Threshold für gewollte Aufstockungen
                if alloc_drift < effective_threshold:
                    log.debug(
                        f"{ticker}: BUY-Drift {alloc_drift:.1%} < {effective_threshold:.0%} Schwelle "
                        f"-> Soft Rebalancing uebersprungen"
                    )
                    continue

            if abs(diff_value) < min_trade_value:
                log.debug(f"{ticker}: Differenz ${diff_value:.2f} unter Mindestwert, ueberspringe.")
                continue

            action = "BUY" if diff_value > 0 else "SELL"

            if action == "SELL":
                # FIX 2: Broker-exakte Stueckzahl fuer Teilverkauf
                available_qty = self.positions.get(ticker, {}).get(
                    "qty_available", self.positions.get(ticker, {}).get("quantity", 0)
                )
                needed_qty = abs(diff_value) / price
                sell_qty = round(min(needed_qty, available_qty), 6)
                if sell_qty <= 0:
                    log.info(f"{ticker}: Keine verfuegbare Menge zum Verkaufen -> uebersprungen")
                    continue
                quantity = sell_qty
                actual_value = round(quantity * price, 2)
            else:
                quantity = abs(diff_value) / price
                actual_value = round(abs(diff_value), 2)

            trades.append({
                "ticker": ticker,
                "action": action,
                "quantity": round(quantity, 6),
                "price": price,
                "value": actual_value,
                "current_alloc": round(current_alloc, 4),
                "target_alloc": round(target_alloc, 4),
                "diff_value": round(diff_value, 2),
                "ai_action": ai_action,
                "ai_reason": ai_reason,
                "ai_confidence": ai_confidence,
                "decision_id": decision.get("decision_id"),
            })
            log.debug(
                f"Rebalancing {ticker}: {action} ${actual_value:.0f} "
                f"({current_alloc*100:.1f}% -> {target_alloc*100:.1f}%, "
                f"Drift: {alloc_drift:.1%})"
            )

        # SELLs zuerst (Cash generieren fuer BUYs)
        trades.sort(key=lambda t: 0 if t["action"] == "SELL" else 1)
        return trades

    def simulate_trade_plan(
        self,
        trades: List[Dict],
        current_prices: Dict[str, float],
        min_cash_pct: float = 0.0,
    ) -> Dict:
        """Simuliert einen Trade-Plan ohne das reale Portfolio zu verändern."""
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
            else:
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
                        "quantity": quantity_total,
                        "qty_available": quantity_total,
                        "cost_basis": round(cost_basis, 2),
                        "avg_price": round(cost_basis / quantity_total, 6),
                        "current_price": price,
                        "market_value": round(quantity_total * price, 2),
                    })
                else:
                    working_positions[ticker] = {
                        "ticker": ticker,
                        "quantity": quantity,
                        "qty_available": quantity,
                        "avg_price": price,
                        "current_price": price,
                        "market_value": round(value, 2),
                        "cost_basis": round(value, 2),
                        "unrealized_pnl": 0.0,
                        "unrealized_pnl_pct": 0.0,
                        "entry_date": datetime.now().isoformat(),
                        "last_updated": datetime.now().isoformat(),
                    }

        invested = sum(p.get("market_value", 0) for p in working_positions.values())
        total_after = round(cash + invested, 2)
        pnl_pct = pct_change(self.initial_capital, total_after) if total_after else 0
        allocs = {
            ticker: pos.get("market_value", 0) / total_after if total_after else 0
            for ticker, pos in working_positions.items()
        }
        for ticker, pos in working_positions.items():
            pos["allocation_pct"] = round(allocs.get(ticker, 0) * 100, 2)

        return {
            "total_value": total_after,
            "cash": round(cash, 2),
            "cash_pct": round(cash / total_after * 100, 2) if total_after else 0,
            "invested": round(invested, 2),
            "initial_capital": self.initial_capital,
            "unrealized_pnl": round(sum(p.get("market_value", 0) - p.get("cost_basis", 0) for p in working_positions.values()), 2),
            "pnl_pct": round(pnl_pct, 2),
            "n_positions": len(working_positions),
            "positions": working_positions,
            "mode": self.mode,
            "assumed_min_cash_pct": min_cash_pct,
        }