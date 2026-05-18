"""
AI Trading Bot - Risk Manager (Production)
============================================
CHANGES:
  Fix #1  – Globaler Final-Check: Cash >= Minimum nach ALLEN Entscheidungen
             (bereits implementiert – nicht angetastet)
  Fix #2  – KI-Input Filter: SELL nur für gehaltene Assets
  Fix #4  – ZombieExclusionList: Forced-Exit Assets können kein BUY erhalten
  Fix #5  – FinalDecisionResolver: jedes Asset genau EINEN finalen Status pro Run
  Fix #7  – Risk Manager als harte letzte Instanz (Hard Gate)
  Fix #12 – State: Risk-Entscheidungen werden persistent geloggt

Invarianten (nach validate_decisions):
  1. Jedes Asset hat GENAU EINEN Status: BUY | SELL | HOLD | SKIPPED | REJECTED
  2. Kein Asset mit ForcedExit (Zombie, StopLoss, ForcedRebalancing) bekommt BUY
  3. Cash >= min_cash_pct (Hard Gate Fix #1)
  4. SELL ohne Position → gefiltert
  5. Alle Zwischenzustände (z.B. SELL + HOLD für selben Ticker) → aufgelöst
"""

from typing import Dict, List, Optional, Set, Tuple
from logger import log
from config import (
    ACTIVE_RISK_PROFILE,
    RISK_SETTINGS,
    RiskProfile,
    SECTOR_CLASSIFICATION,
    FACTOR_CLASSIFICATION,
    ETF_SECTOR_WEIGHTS,
    ETF_FACTOR_WEIGHTS,
)
from utils import format_currency, zombie_registry, ensure_decision_ids

ETF_OVERLAP_GROUPS = [
    {"SPY", "VTI", "IVV", "VOO"},
    {"QQQ", "QQQM"},
]

TOP_N_BUYS = 5

# ─────────────────────────────────────────────────────────────
# CIRCUIT BREAKER SYSTEM
# ─────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Portfolio-level stress detection and automatic trading halts.
    
    Triggers:
      - CB1: Intraday loss > 5% → STOP_ALL_BUYS (allow SELL only)
      - CB2: Portfolio drawdown > 10% → REDUCE_EXPOSURE (50% smaller sizes)
      - CB3: VIX spike > 35 → DE_RISK_50 (close 50% of long positions)
      - CB4: Market halted → HALT_ALL (no trades)
    """
    
    def __init__(self):
        self.triggered = False
        self.reason = None
        self.action = None
    
    def check_all(self, portfolio: Dict, market_data: Dict) -> bool:
        """
        Check all circuit breaker conditions.
        
        Returns True if ANY breaker is triggered.
        """
        # CB1: Intraday Loss
        intraday_loss = market_data.get('intraday_loss_pct', 0.0)
        if intraday_loss < -0.05:
            self.triggered = True
            self.reason = f"Intraday loss: {intraday_loss:.2%} < -5%"
            self.action = 'STOP_ALL_BUYS'
            log.critical(f"🛑 CIRCUIT BREAKER 1 (Intraday Loss): {self.reason}")
            return True
        
        # CB2: Portfolio Drawdown
        drawdown = portfolio.get('drawdown_pct', 0.0)
        if drawdown < -0.10:
            self.triggered = True
            self.reason = f"Portfolio drawdown: {drawdown:.2%} < -10%"
            self.action = 'REDUCE_EXPOSURE'
            log.critical(f"🛑 CIRCUIT BREAKER 2 (Drawdown): {self.reason}")
            return True
        
        # CB3: VIX Spike
        vix = market_data.get('vix', 20.0)
        if vix > 35:
            self.triggered = True
            self.reason = f"VIX spike: {vix:.1f} > 35"
            self.action = 'DE_RISK_50'
            log.critical(f"🛑 CIRCUIT BREAKER 3 (VIX Spike): {self.reason}")
            return True
        
        # CB4: Market Halted
        if market_data.get('market_halted', False):
            self.triggered = True
            self.reason = "Market circuit breaker triggered"
            self.action = 'HALT_ALL'
            log.critical(f"🛑 CIRCUIT BREAKER 4 (Market Halt): {self.reason}")
            return True
        
        # All clear
        self.triggered = False
        self.action = None
        return False
    
    def get_action(self) -> str:
        """Get the action to take when breaker is triggered."""
        return self.action or 'OK'


# ─────────────────────────────────────────────────────────────



FINAL_STATUS_SELL      = "SELL"
FINAL_STATUS_BUY       = "BUY"
FINAL_STATUS_HOLD      = "HOLD"
FINAL_STATUS_SKIPPED   = "SKIPPED"
FINAL_STATUS_REJECTED  = "REJECTED"


class FinalDecisionResolver:
    """
    Fix #5 – Mutual Exclusive Decision Resolver.

    Problem: Ein Asset kann nach dem Validierungsdurchlauf in mehreren
    Entscheidungslisten auftauchen:
      - SELL (Zombie) + HOLD (KI)   → Konflikt
      - BUY (KI) + HOLD (limitiert) → Konflikt
      - SELL (StopLoss) + BUY (KI)  → Kritischer Konflikt

    Auflösungsregeln (Priorität absteigend):
      1. FORCED_EXIT beats everything
         (zombie_cleanup, stop_loss, forced_rebalancing → immer SELL)
      2. SELL beats HOLD
         (explizit beschlossener Verkauf schlägt neutrales Halten)
      3. BUY beats HOLD
         (genehmigter Kauf hat mehr Information als neutrales Halten)
      4. Wenn mehrere SELL → letzter (späterer im Stream) gewinnt
         (Rebalancing-SELL kann StopLoss überschreiben wäre falsch → StopLoss gewinnt)
         Spezialfall: stop_loss > rebalancing > zombie > normaler SELL

    Ergebnis: Exakt eine Decision pro Ticker. Keine Duplikate. Kein Widerspruch.
    """

    # Forced-Exit-Marker – diese Flags haben höchste Priorität
    FORCED_EXIT_MARKERS = ("stop_loss", "zombie_cleanup", "forced_rebalancing")

    # SELL-Typ-Priorität (höher = wichtiger)
    SELL_PRIORITY = {
        "stop_loss":          100,
        "zombie_cleanup":      90,
        "forced_rebalancing":  80,
        "rebalancing":         70,
        "normal":              50,
    }

    def resolve(self, decisions: List[Dict]) -> List[Dict]:
        """
        Nimmt eine Liste von Entscheidungen (kann Duplikate enthalten)
        und gibt eine Liste zurück bei der jeder Ticker genau einmal vorkommt.

        Reihenfolge der Eingabe ist irrelevant – Prioritätsregeln gelten.
        """
        # Schritt 1: Nach Ticker gruppieren
        by_ticker: Dict[str, List[Dict]] = {}
        for d in decisions:
            t = d.get("ticker", "")
            if not t:
                continue
            by_ticker.setdefault(t, []).append(d)

        resolved: List[Dict] = []
        for ticker, candidates in by_ticker.items():
            if len(candidates) == 1:
                resolved.append(candidates[0])
                continue

            # Mehrere Einträge für denselben Ticker → auflösen
            winner = self._pick_winner(ticker, candidates)
            losers = [c for c in candidates if c is not winner]

            if losers:
                loser_actions = [f"{l.get('action')} ({self._sell_type(l)})" for l in losers]
                log.info(
                    f"[DecisionResolver] {ticker}: {len(candidates)} Konflikte aufgelöst "
                    f"→ {winner.get('action')} gewinnt "
                    f"(verworfen: {', '.join(loser_actions)})"
                )

            resolved.append(winner)

        return resolved

    def _pick_winner(self, ticker: str, candidates: List[Dict]) -> Dict:
        """
        Wählt die dominierende Entscheidung nach Prioritätsregeln.
        """
        # Trenne nach Aktion
        forced_exits = [
            c for c in candidates
            if any(c.get(m, False) for m in self.FORCED_EXIT_MARKERS)
        ]
        sells = [c for c in candidates if c.get("action") == "SELL"]
        buys  = [c for c in candidates if c.get("action") == "BUY"
                 and c.get("risk_approved", False)]
        holds = [c for c in candidates if c.get("action") == "HOLD"]

        # Regel 1: Forced-Exit hat absolute Priorität
        if forced_exits:
            # Unter forced exits: stop_loss > zombie > forced_rebalancing
            forced_exits.sort(
                key=lambda c: self._forced_exit_priority(c),
                reverse=True,
            )
            winner = forced_exits[0]
            # Sicherstellen dass action=SELL und risk_approved=True
            winner["action"]       = "SELL"
            winner["risk_approved"] = True
            return winner

        # Regel 2: Approved SELL beats HOLD
        approved_sells = [c for c in sells if c.get("risk_approved", False)]
        if approved_sells:
            # Höchste SELL-Priorität gewinnt
            approved_sells.sort(
                key=lambda c: self._sell_type_priority(c),
                reverse=True,
            )
            return approved_sells[0]

        # Regel 3: Approved BUY beats HOLD
        if buys:
            # Höchste Konfidenz gewinnt
            buys.sort(key=lambda c: c.get("confidence", 0), reverse=True)
            return buys[0]

        # Regel 4: HOLD (nimm das letzte/neueste)
        if holds:
            return holds[-1]

        # Fallback: ersten nehmen
        log.warning(
            f"[DecisionResolver] {ticker}: Unerwarteter Fall, "
            f"nehme ersten Kandidaten. Aktionen: "
            f"{[c.get('action') for c in candidates]}"
        )
        return candidates[0]

    def _forced_exit_priority(self, d: Dict) -> int:
        if d.get("stop_loss"):
            return 100
        if d.get("zombie_cleanup"):
            return 90
        if d.get("forced_rebalancing"):
            return 80
        return 0

    def _sell_type_priority(self, d: Dict) -> int:
        return self.SELL_PRIORITY.get(self._sell_type(d), 50)

    @staticmethod
    def _sell_type(d: Dict) -> str:
        if d.get("stop_loss"):
            return "stop_loss"
        if d.get("zombie_cleanup"):
            return "zombie_cleanup"
        if d.get("forced_rebalancing"):
            return "forced_rebalancing"
        if d.get("rebalancing"):
            return "rebalancing"
        return "normal"


class RiskManager:

    def __init__(self, risk_profile: RiskProfile = None):
        self.risk_profile    = risk_profile or ACTIVE_RISK_PROFILE
        self.settings        = RISK_SETTINGS[self.risk_profile].copy()
        self._base_settings  = RISK_SETTINGS[self.risk_profile].copy()
        self._resolver       = FinalDecisionResolver()
        self.regime_state    = None
        log.info(f"Risk Manager initialisiert: Profil={self.risk_profile.value.upper()}")
        log.info(
            f"  Max Position: {self.settings['max_position_pct']*100:.0f}% | "
            f"Min Cash: {self.settings['min_cash_pct']*100:.0f}% | "
            f"Stop-Loss: {self.settings['stop_loss_pct']*100:.0f}% | "
            f"Top-N BUYs: {TOP_N_BUYS}"
        )

    # ─── ATR-BASED DYNAMIC STOP-LOSS ──────────────────────────────────────────
    
    def calculate_dynamic_stop_loss(
        self,
        ticker: str,
        avg_price: float,
        market_data: Dict[str, Dict],
        atr_multiplier: float = 2.5,
    ) -> float:
        """
        🔥 VOLATILITY ENGINE: ATR-based dynamic stop loss
        
        Berechnung:
          1. annual_vol aus market_data[ticker]["volatility_annual"]
          2. daily_vol = annual_vol / sqrt(252)
          3. ATR ≈ avg_price * daily_vol
          4. dynamic_stop = avg_price - (ATR * atr_multiplier)
          5. fixed_stop = avg_price * (1 - stop_loss_pct)
          6. return max(dynamic_stop, fixed_stop)  ← Fixed floor bleibt MINIMUM
        
        Args:
            ticker: Ticker symbol
            avg_price: Durchschnittskaufpreis
            market_data: Dict mit Volatilitätsdaten
            atr_multiplier: wie viele ATRs unter Kaufpreis (default 2.5)
        
        Returns:
            stop_loss_price (höher = strengere Stop)
        """
        import math
        
        # Hole Annual-Volatilität aus market_data
        ticker_data = market_data.get(ticker, {})
        annual_vol = ticker_data.get("volatility_annual", 0.20)  # fallback 20%
        
        # Berechne tägliche Volatilität
        daily_vol = annual_vol / math.sqrt(252)
        
        # ATR-Approximation: avg_price * daily_vol
        atr_approx = avg_price * daily_vol
        
        # Dynamischer Stop
        dynamic_stop = avg_price - (atr_approx * atr_multiplier)
        
        # Fixed Stop Floor (WICHTIG: bleibt MINIMUM)
        fixed_stop_pct = self.settings.get("stop_loss_pct", 0.08)
        fixed_stop = avg_price * (1 - fixed_stop_pct)
        
        # Finale Regel: max(dynamic, fixed) - fixed bleibt Floor
        final_stop = max(dynamic_stop, fixed_stop)
        
        log.debug(
            f"[ATR STOP] {ticker}: "
            f"avg={avg_price:.2f} | annual_vol={annual_vol:.0%} | "
            f"daily_vol={daily_vol:.2%} | ATR={atr_approx:.2f} | "
            f"dynamic_stop={dynamic_stop:.2f} | fixed_stop={fixed_stop:.2f} | "
            f"final={final_stop:.2f}"
        )
        
        return final_stop

    # ─── VOLATILITY-ADJUSTED POSITION SIZING ──────────────────────────────────
    
    def volatility_adjusted_allocation(
        self,
        base_allocation: float,
        asset_volatility_pct: float,
        target_volatility: float = 0.15,
    ) -> float:
        """
        🔥 SIZING ENGINE: Volatility-adjusted allocation
        
        Idea: Assets mit höherer Volatilität bekommen kleinere Positionen.
        
        Regeln:
          1. scale = target_volatility / asset_volatility
          2. Clamp: 0.3 ≤ scale ≤ 2.0 (nicht zu extrem)
          3. allocation = base_allocation * scale
          4. allocation = min(allocation, max_position_pct)
        
        Args:
            base_allocation: Basis-Allokation (von KI empfohlen)
            asset_volatility_pct: Volatilität des Assets (z.B. 0.25 = 25%)
            target_volatility: Portfolio target volatility (default 15%)
        
        Returns:
            adjusted_allocation (clamped to reasonable bounds)
        """
        if asset_volatility_pct <= 0:
            asset_volatility_pct = 0.15  # fallback
        
        # Scale-Faktor basierend auf Volatilitätsverhältnis
        scale = target_volatility / asset_volatility_pct
        
        # Clamp zwischen 0.3 und 2.0 – verhindert extreme Anpassungen
        scale = max(0.3, min(2.0, scale))
        
        # Berechne adjusted allocation
        adjusted = base_allocation * scale
        
        # Clampe auf max_position von Profil
        max_position = self.settings.get("max_position_pct", 0.25)
        adjusted = min(adjusted, max_position)
        
        log.debug(
            f"[VOL ADJUST] "
            f"base={base_allocation:.0%} | asset_vol={asset_volatility_pct:.0%} | "
            f"scale={scale:.2f} | adjusted={adjusted:.0%} | max={max_position:.0%}"
        )
        
        return adjusted

    def apply_regime(self, regime_state) -> None:
        """
        Passt Risk-Settings basierend auf dem Markt-Regime an.
        BULL → unverändert | BEAR → konservativer | SIDEWAYS → weniger Trades
        """
        from market_regime import apply_regime_to_risk_settings
        self.regime_state = regime_state
        self.settings = apply_regime_to_risk_settings(self._base_settings, regime_state)

    def _sector_of(self, ticker: str) -> str:
        return SECTOR_CLASSIFICATION.get(ticker, "diversified")

    def _factor_of(self, ticker: str) -> str:
        return FACTOR_CLASSIFICATION.get(ticker, self._sector_of(ticker))

    def _weighted_exposure(
        self,
        ticker: str,
        alloc: float,
        mapping: Dict[str, Dict[str, float]],
        default_label: str,
    ) -> Dict[str, float]:
        exposure: Dict[str, float] = {}
        weights = mapping.get(ticker)
        if weights:
            for label, weight in weights.items():
                exposure[label] = exposure.get(label, 0.0) + alloc * weight
        else:
            exposure[default_label] = exposure.get(default_label, 0.0) + alloc
        return exposure

    def _sector_exposure(self, positions: Dict, total_value: float) -> Dict[str, float]:
        exposure: Dict[str, float] = {}
        for ticker, pos in positions.items():
            alloc = pos.get("market_value", 0) / total_value if total_value > 0 else 0
            sector = self._sector_of(ticker)
            weighted = self._weighted_exposure(ticker, alloc, ETF_SECTOR_WEIGHTS, sector)
            for label, value in weighted.items():
                exposure[label] = exposure.get(label, 0.0) + value
        return exposure

    def _factor_exposure(self, positions: Dict, total_value: float) -> Dict[str, float]:
        exposure: Dict[str, float] = {}
        for ticker, pos in positions.items():
            alloc = pos.get("market_value", 0) / total_value if total_value > 0 else 0
            factor = self._factor_of(ticker)
            weighted = self._weighted_exposure(ticker, alloc, ETF_FACTOR_WEIGHTS, factor)
            for label, value in weighted.items():
                exposure[label] = exposure.get(label, 0.0) + value
        return exposure

    def _apply_sector_limit(self, decision: Dict, positions: Dict, total_value: float, current_sectors: Dict[str, float], max_sector: float) -> Tuple[Dict, Optional[str]]:
        ticker = decision.get("ticker", "")
        if decision.get("action") != "BUY":
            return decision, None

        sector = self._sector_of(ticker)
        current_alloc = positions.get(ticker, {}).get("market_value", 0) / total_value if total_value > 0 else 0
        current_sector = current_sectors.get(sector, 0.0)
        proposed = decision.get("target_allocation", 0)
        if proposed <= current_alloc:
            return decision, None

        max_add = max(0.0, max_sector - current_sector)
        max_target = current_alloc + max_add

        if max_add <= 0:
            msg = (
                f"{ticker}: BUY blockiert, {sector.upper()}-Sektorlimit ({max_sector:.0%}) erreicht."
            )
            decision = dict(decision)
            decision["action"] = "HOLD"
            decision["risk_approved"] = False
            decision["reason"] += f" [SECTOR LIMIT: {sector}]"
            return decision, msg

        if proposed > max_target:
            if max_target - current_alloc < 0.01:
                msg = (
                    f"{ticker}: BUY reduziert, da {sector.upper()}-Sektorlimit ({max_sector:.0%}) sonst verletzt würde."
                )
                decision = dict(decision)
                decision["action"] = "HOLD"
                decision["risk_approved"] = False
                decision["reason"] += f" [SECTOR LIMIT: {sector}]"
                return decision, msg

            old_target = proposed
            decision = dict(decision)
            decision["target_allocation"] = round(max_target, 4)
            msg = (
                f"{ticker}: Target reduziert wegen {sector.upper()}-Exposure. "
                f"{old_target:.0%} -> {decision['target_allocation']:.0%}."
            )
            decision["reason"] += f" [SECTOR LIMIT: {sector}]"
            return decision, msg

        return decision, None

    def _apply_factor_limit(self, decision: Dict, positions: Dict, total_value: float, current_factors: Dict[str, float], max_factor: float) -> Tuple[Dict, Optional[str]]:
        ticker = decision.get("ticker", "")
        if decision.get("action") != "BUY":
            return decision, None

        factor = self._factor_of(ticker)
        current_alloc = positions.get(ticker, {}).get("market_value", 0) / total_value if total_value > 0 else 0
        current_factor = current_factors.get(factor, 0.0)
        proposed = decision.get("target_allocation", 0)
        if proposed <= current_alloc:
            return decision, None

        max_add = max(0.0, max_factor - current_factor)
        max_target = current_alloc + max_add

        if max_add <= 0:
            msg = (
                f"{ticker}: BUY blockiert, {factor.upper()}-Faktorlimit ({max_factor:.0%}) erreicht."
            )
            decision = dict(decision)
            decision["action"] = "HOLD"
            decision["risk_approved"] = False
            decision["reason"] += f" [FACTOR LIMIT: {factor}]"
            return decision, msg

        if proposed > max_target:
            if max_target - current_alloc < 0.01:
                msg = (
                    f"{ticker}: BUY reduziert, da {factor.upper()}-Faktorlimit ({max_factor:.0%}) sonst verletzt würde."
                )
                decision = dict(decision)
                decision["action"] = "HOLD"
                decision["risk_approved"] = False
                decision["reason"] += f" [FACTOR LIMIT: {factor}]"
                return decision, msg

            old_target = proposed
            decision = dict(decision)
            decision["target_allocation"] = round(max_target, 4)
            msg = (
                f"{ticker}: Target reduziert wegen {factor.upper()}-Exposure. "
                f"{old_target:.0%} -> {decision['target_allocation']:.0%}."
            )
            decision["reason"] += f" [FACTOR LIMIT: {factor}]"
            return decision, msg

        return decision, None

    # ─── ETF-Overlap ──────────────────────────────────────────────────────────

    def check_etf_overlap(self, decisions: List[Dict], positions: Dict) -> List[str]:
        """Warnt bei stark überlappenden ETFs im Portfolio."""
        warnings     = []
        buy_tickers  = {d["ticker"] for d in decisions if d.get("action") == "BUY"}
        held_tickers = set(positions.keys())
        all_active   = buy_tickers | held_tickers
        for group in ETF_OVERLAP_GROUPS:
            active = all_active & group
            if len(active) >= 2:
                msg = (
                    f"ETF-Überschneidung: {', '.join(sorted(active))} "
                    f"haben starke Überschneidungen – Diversifikation prüfen!"
                )
                warnings.append(msg)
                log.warning(f"  {msg}")
        return warnings

    # ─── Fix #2: SELL-Filter für nicht gehaltene Assets ───────────────────────

    def filter_invalid_sells(
        self,
        decisions: List[Dict],
        positions: Dict,
    ) -> Tuple[List[Dict], List[str]]:
        """
        Fix #2/4: Entfernt SELL-Entscheidungen für Assets ohne Position.
        Ausnahme: zombie_cleanup und stop_loss dürfen trotzdem durch –
        sie werden im nächsten Schritt sauber behandelt.
        """
        warnings = []
        filtered = []
        for d in decisions:
            if d.get("action") == "SELL" and d["ticker"] not in positions:
                # Zombie/StopLoss-SELLs ohne Position sind sinnlos, aber harmlos
                is_special = d.get("zombie_cleanup") or d.get("stop_loss")
                if is_special:
                    log.debug(
                        f"{d['ticker']}: Spezial-SELL (zombie/stop) ohne Position "
                        f"→ übersprungen"
                    )
                else:
                    msg = (
                        f"{d['ticker']}: SELL für nicht gehaltene Position "
                        f"→ gefiltert (Fix #4)"
                    )
                    log.info(f"  {msg}")
                    warnings.append(msg)
                continue
            filtered.append(d)
        return filtered, warnings

    # ─── Fix #4: ZombieExclusionList – BUY für Forced-Exit Assets blockieren ─

    def apply_zombie_exclusion(
        self,
        decisions: List[Dict],
        forced_exit_tickers: Set[str],
    ) -> Tuple[List[Dict], List[str]]:
        """
        Fix #4 – Zombie Exclusion List.

        Wenn ein Asset in diesem Run einen Forced-Exit erhält
        (zombie_cleanup, stop_loss, forced_rebalancing), dann darf es
        KEINEN BUY erhalten. NIEMALS. Forced Exit > KI-Entscheidung.

        Zusätzlich: ZombieRegistry blockiert BUYs aus vorherigen Runs.

        Gibt die gefilterte Liste und Warnungen zurück.
        """
        if not forced_exit_tickers:
            return decisions, []

        warnings: List[str] = []
        result: List[Dict]  = []

        for d in decisions:
            ticker = d.get("ticker", "")
            action = d.get("action", "")

            # Prüfe ob BUY für ein Forced-Exit-Asset versucht wird
            if action == "BUY":
                # Aktueller Run: Forced-Exit in diesem Run
                if ticker in forced_exit_tickers:
                    msg = (
                        f"{ticker}: BUY blockiert – Asset hat Forced-Exit "
                        f"in diesem Run (zombie/stop-loss/forced-rebalancing). "
                        f"Forced Exit > KI-Entscheidung."
                    )
                    warnings.append(msg)
                    log.warning(f"  ⛔ [Fix #4] {msg}")
                    d = dict(d)  # Kopie um Original nicht zu mutieren
                    d["action"]        = "HOLD"
                    d["risk_approved"] = False
                    d["reason"]        = d.get("reason", "") + " [FIX #4: Forced-Exit blockiert BUY]"
                    result.append(d)
                    continue

                # Vorherige Runs: ZombieRegistry blockiert BUY
                if zombie_registry.is_buy_blocked(ticker):
                    status = zombie_registry.get_status(ticker)
                    msg = (
                        f"{ticker}: BUY blockiert – Asset in ZombieRegistry "
                        f"(Status: {status}). Zombie-State ist final."
                    )
                    warnings.append(msg)
                    log.warning(f"  ⛔ [Fix #4] {msg}")
                    d = dict(d)
                    d["action"]        = "HOLD"
                    d["risk_approved"] = False
                    d["reason"]        = d.get("reason", "") + f" [FIX #4: Zombie blockiert BUY ({status})]"
                    result.append(d)
                    continue

            result.append(d)

        return result, warnings

    # ─── Rebalancing ──────────────────────────────────────────────────────────

    def generate_rebalancing_decisions(
        self,
        decisions: List[Dict],
        portfolio_summary: Dict,
    ) -> List[Dict]:
        """
        Fügt SELL-Entscheidungen hinzu für Positionen über ihrer Ziel-Allokation.

        Wichtig: Rebalancing wird NUR ausgelöst wenn die KI explizit eine
        niedrigere Zielallokation per BUY oder SELL angegeben hat.
        HOLD-Entscheidungen (="nichts tun") lösen KEIN Rebalancing aus —
        fehlende Erwähnung eines Tickers ebenfalls nicht.
        """
        total_value = portfolio_summary.get("total_value", 1)
        positions   = portfolio_summary.get("positions", {})

        # Nur BUY und SELL-Entscheidungen mit expliziter Zielallokation berücksichtigen
        # HOLD bedeutet "Status quo beibehalten" — kein Rebalancing-Signal
        target_map = {
            d["ticker"]: d.get("target_allocation", 0)
            for d in decisions
            if d.get("action") in ("BUY", "SELL")
        }
        rebalance_sells = []

        for ticker, pos in positions.items():
            current_alloc = pos.get("market_value", 0) / total_value
            target_alloc  = target_map.get(ticker)
            # Kein explizites Ziel von der KI → Position unangetastet lassen
            if target_alloc is None:
                continue
            if current_alloc > target_alloc + 0.03:
                already_sell = any(
                    d["ticker"] == ticker and d["action"] == "SELL"
                    for d in decisions
                )
                if not already_sell:
                    rebalance_sells.append({
                        "ticker":            ticker,
                        "action":            "SELL",
                        "target_allocation": target_alloc,
                        "confidence":        0.80,
                        "reason": (
                            f"Rebalancing: {current_alloc:.0%} -> {target_alloc:.0%} "
                            f"(+{(current_alloc - target_alloc)*100:.0f}% über Ziel)"
                        ),
                        "risk_approved": True,
                        "rebalancing":   True,
                        "decision_id":     f"REBALANCE_{ticker}",
                    })
                    log.info(
                        f"  Rebalancing-SELL: {ticker} | "
                        f"Aktuell: {current_alloc:.0%} -> Ziel: {target_alloc:.0%}"
                    )
        if rebalance_sells:
            log.info(f"  {len(rebalance_sells)} Rebalancing-Verkäufe hinzugefügt.")
        return rebalance_sells

    def generate_sector_rebalancing_sells(
        self,
        portfolio_summary: Dict,
        max_sector: float,
    ) -> List[Dict]:
        """Generates forced SELLs when current sector exposure already exceeds limits."""
        total_value = portfolio_summary.get("total_value", 1)
        positions = portfolio_summary.get("positions", {})
        sector_exposure = self._sector_exposure(positions, total_value)
        forced_sells: List[Dict] = []

        for sector, exposure in sector_exposure.items():
            if exposure <= max_sector:
                continue

            excess = exposure - max_sector * 0.9
            if excess <= 0:
                continue

            remaining = excess * total_value
            sector_positions = [
                (ticker, pos)
                for ticker, pos in positions.items()
                if self._sector_of(ticker) == sector and pos.get("market_value", 0) > 0
            ]
            sector_positions.sort(key=lambda item: item[1].get("market_value", 0))

            for ticker, pos in sector_positions:
                if remaining <= 0:
                    break
                current_alloc = pos.get("market_value", 0) / total_value if total_value > 0 else 0
                if current_alloc <= 0:
                    continue

                target_alloc = max(0.0, current_alloc - min(current_alloc, remaining / total_value))
                sell_value = max(0.0, current_alloc - target_alloc) * total_value
                if sell_value < 1:
                    continue

                forced_sells.append({
                    "ticker": ticker,
                    "action": "SELL",
                    "target_allocation": round(target_alloc, 4),
                    "confidence": 1.0,
                    "reason": (
                        f"Forced sector rebalance: {sector} exposure {exposure:.0%} > {max_sector:.0%}."
                    ),
                    "risk_approved": True,
                    "forced_rebalancing": True,
                    "rebalancing": True,
                    "priority": "CRITICAL",
                    "decision_id": f"SECTOR_REBALANCE_{ticker}",
                })
                remaining -= sell_value

        if forced_sells:
            sectors = sorted({self._sector_of(s['ticker']) for s in forced_sells})
            log.warning(
                f"Sector-Exposure über Limit: {', '.join(sectors)}. Forced sells generated."
            )
        return forced_sells

    # ─── Haupt-Validierung ────────────────────────────────────────────────────

    def validate_decisions(
        self,
        decisions: List[Dict],
        portfolio_summary: Dict,
        market_data: Dict[str, Dict],
    ) -> Tuple[List[Dict], List[str]]:
        """
        Prüft und korrigiert alle KI-Entscheidungen.

        Reihenfolge (Fix #7 – Risk Manager als letzte Instanz):
          -1. CIRCUIT BREAKERS: Prüfe Portfolio-Stress (Intraday Loss, Drawdown, VIX)
          0. Fix #4: SELL-Filter für nicht gehaltene Assets
          1. ETF-Overlap Detection
          2. Rebalancing SELLs hinzufügen
          3. Forced-Exit-Ticker sammeln (zombie, stop_loss, forced_rebal)
          4. Fix #4: Zombie-Exclusion – BUY für Forced-Exit-Assets blockieren
          5. HOLDs immer durch
          6. SELLs (nie durch Max-Trades blockiert)
          7. Top-N BUYs auswählen
          8. BUYs sequenziell mit Cash-Update
          9. Fix #1: GLOBALER FINAL-CHECK Cash >= Minimum (Hard Gate)
         10. Fix #5: Final Decision Resolver – exakt 1 Status pro Ticker
        """
        warnings     = []
        validated    = []
        max_pos        = self.settings["max_position_pct"]
        min_cash       = self.settings["min_cash_pct"]
        conf_threshold = self.settings["confidence_threshold"]
        max_sector     = self.settings.get("max_sector_exposure", 0.45)
        max_factor     = self.settings.get("max_factor_exposure", 0.60)
        max_trades     = self.settings.get("max_trades_per_run", TOP_N_BUYS)

        total_value  = portfolio_summary.get("total_value", 1)
        running_cash = portfolio_summary.get("cash", 0)
        positions    = portfolio_summary.get("positions", {})

        decisions = ensure_decision_ids(decisions)

        # ═══════════════════════════════════════════════════════════════
        # SCHRITT -1: CIRCUIT BREAKER CHECK (PORTFOLIO STRESS)
        # ═══════════════════════════════════════════════════════════════
        breaker = CircuitBreaker()
        if breaker.check_all(portfolio_summary, market_data):
            log.warning(f"Circuit Breaker Action: {breaker.get_action()}")
            
            if breaker.action == 'STOP_ALL_BUYS':
                # Filter out all BUYs, keep SELLs and HOLDs
                decisions = [d for d in decisions if d.get('action') != 'BUY']
                warnings.append(f"🛑 Circuit Breaker: STOP_ALL_BUYS. {breaker.reason}")
            
            elif breaker.action == 'REDUCE_EXPOSURE':
                # Reduce all BUY sizes by 50%
                for d in decisions:
                    if d.get('action') == 'BUY':
                        d['target_allocation'] *= 0.5
                warnings.append(f"🛑 Circuit Breaker: REDUCE_EXPOSURE. {breaker.reason}")
            
            elif breaker.action == 'DE_RISK_50':
                # Generate forced SELL signals for 50% of long positions
                forced_sells = []
                positions_list = sorted(positions.items(), key=lambda x: x[1].get('market_value', 0))
                total_sell_value = total_value * 0.5
                current_sell_value = 0
                
                for ticker, pos in positions_list:
                    if current_sell_value >= total_sell_value:
                        break
                    if pos.get('quantity', 0) > 0:
                        sell_decision = {
                            'ticker': ticker,
                            'action': 'SELL',
                            'target_allocation': 0.0,
                            'confidence': 1.0,
                            'reason': '🛑 Circuit Breaker: DE_RISK_50',
                            'priority': 'CRITICAL'
                        }
                        forced_sells.append(sell_decision)
                        current_sell_value += pos.get('market_value', 0)
                
                decisions.extend(forced_sells)
                warnings.append(f"🛑 Circuit Breaker: DE_RISK_50 (generated {len(forced_sells)} forced SELLs). {breaker.reason}")
            
            elif breaker.action == 'HALT_ALL':
                # No trades at all
                decisions = []
                warnings.append(f"🛑 Circuit Breaker: HALT_ALL (all trades blocked). {breaker.reason}")

        # Schritt 0: Ungültige SELLs rausfiltern
        decisions, sell_filter_warnings = self.filter_invalid_sells(decisions, positions)
        warnings.extend(sell_filter_warnings)

        # Schritt 1: ETF-Overlap
        overlap_warnings = self.check_etf_overlap(decisions, positions)
        warnings.extend(overlap_warnings)

        # Schritt 2: Rebalancing SELLs
        rebalance_sells = self.generate_rebalancing_decisions(decisions, portfolio_summary)
        decisions = rebalance_sells + decisions

        # Schritt 2b: Sector-Exposure über Limit? Forced Rebalance SELLs generieren
        sector_rebalance_sells = self.generate_sector_rebalancing_sells(portfolio_summary, max_sector)
        if sector_rebalance_sells:
            log.info(f"  Sector-Rebalance: {len(sector_rebalance_sells)} SELL(s) für überlimitierte Sektoren hinzugefügt.")
            decisions = sector_rebalance_sells + decisions

        # Schritt 3: Forced-Exit-Ticker sammeln (Fix #4)
        # Diese Assets dürfen in diesem Run KEIN BUY erhalten
        forced_exit_tickers: Set[str] = {
            d["ticker"]
            for d in decisions
            if (
                d.get("action") == "SELL"
                and (
                    d.get("zombie_cleanup", False)
                    or d.get("stop_loss", False)
                    or d.get("forced_rebalancing", False)
                )
            )
        }
        if forced_exit_tickers:
            log.info(
                f"  Forced-Exit-Ticker (Fix #4): {', '.join(sorted(forced_exit_tickers))}"
            )

        # Schritt 4: Zombie-Exclusion anwenden (Fix #4)
        decisions, exclusion_warnings = self.apply_zombie_exclusion(
            decisions, forced_exit_tickers
        )
        warnings.extend(exclusion_warnings)

        hold_decisions = [d for d in decisions if d.get("action") == "HOLD"]
        sell_decisions = [d for d in decisions if d.get("action") == "SELL"]
        buy_decisions  = [d for d in decisions if d.get("action") == "BUY"]

        sector_exposure = self._sector_exposure(positions, total_value)
        factor_exposure = self._factor_exposure(positions, total_value)

        log.info(
            f"Risikoprüfung: {len(decisions)} Entscheidungen "
            f"({len(sell_decisions)} SELL, {len(buy_decisions)} BUY, "
            f"{len(hold_decisions)} HOLD) | "
            f"Cash: {format_currency(running_cash)}"
        )

        # Schritt 5: HOLDs immer durch
        for d in hold_decisions:
            d["risk_approved"] = True
            validated.append(d)

        # Schritt 6: SELLs – nie durch Max-Trades blockiert
        for d in sell_decisions:
            ticker   = d.get("ticker", "?")
            conf     = d.get("confidence", 0)
            is_rebal  = d.get("rebalancing", False)
            is_zombie = d.get("zombie_cleanup", False)
            is_stop   = d.get("stop_loss", False)
            is_forced = d.get("forced_rebalancing", False)

            # Confidence-Check: Spezialfälle immer durchlassen
            if not (is_rebal or is_zombie or is_stop or is_forced) and conf < conf_threshold:
                d = dict(d)
                d["action"]  = "HOLD"
                d["reason"] += f" [RISK: Konfidenz {conf:.0%} zu niedrig]"
                d["risk_approved"] = False
                validated.append(d)
                warnings.append(f"{ticker}: Konfidenz {conf:.0%} → HOLD")
                continue

            target_alloc     = d.get("target_allocation", 0)
            pos_market_value = positions.get(ticker, {}).get("market_value", 0)
            sell_value = (
                pos_market_value if target_alloc == 0.0
                else max(0, (pos_market_value / total_value - target_alloc) * total_value)
            )

            if sell_value > 0:
                running_cash += sell_value
            d["risk_approved"] = True
            validated.append(d)
            log.debug(
                f"  SELL {ticker} | Frei: {format_currency(sell_value)} | "
                f"Cash: {format_currency(running_cash)}"
            )

        # Schritt 7: Top-N BUYs
        qualified          = [d for d in buy_decisions if d.get("confidence", 0) >= conf_threshold]
        below_threshold    = [d for d in buy_decisions if d.get("confidence", 0) < conf_threshold]
        qualified.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        max_trades = self.settings.get("max_trades_per_run", TOP_N_BUYS)
        top_buys  = qualified[:max_trades]
        skip_buys = qualified[max_trades:]

        for d in below_threshold:
            d = dict(d)
            d["action"]  = "HOLD"
            d["reason"] += f" [RISK: Konfidenz {d['confidence']:.0%} zu niedrig]"
            d["risk_approved"] = False
            validated.append(d)
            warnings.append(f"{d['ticker']}: Konfidenz zu niedrig → HOLD")

        for d in skip_buys:
            d = dict(d)
            d["action"]  = "HOLD"
            d["reason"] += f" [RISK: Max-Trades ({max_trades}) Limit]"
            d["risk_approved"] = False
            validated.append(d)
            warnings.append(f"{d['ticker']}: Max-Trades ({max_trades}) Limit → HOLD")

        if skip_buys:
            log.info(f"  Max-Trades {max_trades}: {len(top_buys)} ausgewählt, {len(skip_buys)} übersprungen")

        # Schritt 8: BUYs sequenziell
        executed_buys = 0
        for d in top_buys:
            ticker       = d.get("ticker", "?")
            target_alloc = d.get("target_allocation", 0)

            d, sector_msg = self._apply_sector_limit(
                d, positions, total_value, sector_exposure, max_sector
            )
            if sector_msg:
                warnings.append(sector_msg)
                if d.get("action") == "HOLD":
                    validated.append(d)
                    continue
                target_alloc = d.get("target_allocation", 0)

            d, factor_msg = self._apply_factor_limit(
                d, positions, total_value, factor_exposure, max_factor
            )
            if factor_msg:
                warnings.append(factor_msg)
                if d.get("action") == "HOLD":
                    validated.append(d)
                    continue
                target_alloc = d.get("target_allocation", 0)

            if target_alloc > max_pos:
                d["target_allocation"] = max_pos
                target_alloc = max_pos

            current_value = positions.get(ticker, {}).get("market_value", 0)
            buy_cost      = max(0, total_value * target_alloc - current_value)
            cash_after    = running_cash - buy_cost
            cash_pct      = cash_after / total_value if total_value > 0 else 0

            if cash_pct < min_cash:
                max_spend = running_cash - (total_value * min_cash)
                if max_spend >= 500:
                    reduced = (current_value + max_spend) / total_value
                    d["target_allocation"] = round(reduced, 4)
                    running_cash -= max_spend
                    msg = f"{ticker}: Teilkauf auf {reduced:.0%} ({format_currency(max_spend)})"
                    warnings.append(msg)
                    log.info(f"  {msg}")
                    d["risk_approved"] = True
                    validated.append(d)
                    executed_buys += 1
                else:
                    d = dict(d)
                    d["action"]  = "HOLD"
                    d["reason"] += " [RISK: Cash reicht nicht]"
                    d["risk_approved"] = False
                    validated.append(d)
                    warnings.append(f"{ticker}: Cash reicht nicht → HOLD")
                continue

            running_cash -= buy_cost
            executed_buys += 1
            d["risk_approved"] = True
            validated.append(d)

            # Aktualisiere Sektor- und Faktor-Exposure für nachfolgende Kaufentscheidungen
            sector = self._sector_of(ticker)
            factor = self._factor_of(ticker)
            current_alloc = positions.get(ticker, {}).get("market_value", 0) / total_value if total_value > 0 else 0
            new_alloc = d.get("target_allocation", 0)
            exposure_delta = max(0.0, new_alloc - current_alloc)
            sector_exposure[sector] = sector_exposure.get(sector, 0.0) + exposure_delta
            factor_exposure[factor] = factor_exposure.get(factor, 0.0) + exposure_delta

        # Schritt 9: GLOBALER FINAL-CHECK mit Auto-Rebalancing (Fix #1 – unverändert)
        validated, final_warnings = self._enforce_cash_invariant(
            validated=validated,
            total_value=total_value,
            projected_cash=running_cash,
            min_cash_pct=min_cash,
            positions=positions,
            market_data=market_data,
        )
        warnings.extend(final_warnings)

        # Schritt 10: Fix #5 – Final Decision Resolver
        # Hier werden alle Konflikte (SELL + HOLD, BUY + HOLD, etc.) aufgelöst.
        # Jeder Ticker erhält genau einen finalen Status.
        count_before = len(validated)
        validated = self._resolver.resolve(validated)
        count_after = len(validated)

        if count_before != count_after:
            log.info(
                f"[DecisionResolver] {count_before} → {count_after} Entscheidungen "
                f"({count_before - count_after} Duplikate aufgelöst)"
            )

        for d in validated:
            if d.get("action") in ("BUY", "SELL") and d.get("risk_approved", False):
                d["status"] = "APPROVED"
            elif not d.get("risk_approved", True):
                d["status"] = "BLOCKED"
            else:
                d["status"] = "HOLD"

        # Finale Ausgabe
        final_sells = sum(1 for d in validated if d.get("action") == "SELL")
        final_buys  = sum(1 for d in validated if d.get("action") == "BUY" and d.get("risk_approved"))
        final_holds = sum(1 for d in validated if d.get("action") == "HOLD")

        log.info(
            f"Risikoprüfung abgeschlossen | "
            f"FINAL: {final_sells} SELL | {final_buys} BUY | {final_holds} HOLD | "
            f"Cash: {format_currency(running_cash)} ({running_cash/total_value:.0%})"
        )
        if warnings:
            warnings = list(dict.fromkeys(warnings))
            for w in warnings:
                log.info(f"  ⚠ {w}")

        # Integritätsprüfung: kein Ticker darf doppelt vorkommen
        self._assert_no_duplicates(validated)

        return validated, warnings

    def _assert_no_duplicates(self, decisions: List[Dict]):
        """
        Defensive: Prüft dass kein Ticker doppelt vorkommt.
        Wirft keinen Fehler, loggt aber einen ERROR wenn Invariante verletzt.
        """
        seen: Set[str] = set()
        for d in decisions:
            t = d.get("ticker", "")
            if t in seen:
                log.error(
                    f"[INVARIANTE VERLETZT] Ticker {t} kommt mehrfach vor! "
                    f"Dies ist ein Bug – bitte melden."
                )
            seen.add(t)

    # ─── Fix #1: Cash-Invariante (unverändert, nur Zombie-Marker ergänzt) ─────

    def _enforce_cash_invariant(
        self,
        validated: List[Dict],
        total_value: float,
        projected_cash: float,
        min_cash_pct: float,
        positions: Dict,
        market_data: Dict[str, Dict],
    ) -> Tuple[List[Dict], List[str]]:
        """
        PHASE-1 FIX #1 – Aktives Auto-Rebalancing wenn Cash < Minimum.
        (Vollständig aus der vorherigen Implementierung übernommen.)

        Stufe 1: BUY-Orders canceln (niedrigste Konfidenz zuerst)
        Stufe 2: Aktive SELL-Orders generieren
        """
        warnings: List[str] = []
        min_cash_abs = total_value * min_cash_pct

        if projected_cash >= min_cash_abs:
            return validated, warnings

        shortfall = min_cash_abs - projected_cash
        log.warning(
            f"[HARD GATE] Cash-Invariante verletzt! "
            f"Projiziert: {format_currency(projected_cash)} | "
            f"Minimum: {format_currency(min_cash_abs)} | "
            f"Fehlbetrag: {format_currency(shortfall)}"
        )

        # Stufe 1: BUYs canceln
        approved_buys = [
            d for d in validated
            if d.get("action") == "BUY" and d.get("risk_approved", False)
        ]
        approved_buys.sort(key=lambda x: x.get("confidence", 0))

        recovered = projected_cash
        for d in approved_buys:
            if recovered >= min_cash_abs:
                break
            ticker   = d["ticker"]
            buy_cost = max(0.0, total_value * d.get("target_allocation", 0)
                          - positions.get(ticker, {}).get("market_value", 0))
            d["action"]       = "HOLD"
            d["reason"]      += " [HARD GATE: BUY gecancelt für Cash-Reserve]"
            d["risk_approved"] = False
            recovered += buy_cost
            msg = (
                f"{ticker}: BUY gecancelt → Cash freigegeben "
                f"~{format_currency(buy_cost)} | "
                f"Projiziert: {format_currency(recovered)}"
            )
            warnings.append(msg)
            log.warning(f"  ⬆ {msg}")

        if recovered >= min_cash_abs:
            log.info(
                f"[HARD GATE] Stufe 1 erfolgreich: "
                f"Cash nach BUY-Cancel: {format_currency(recovered)} "
                f">= Minimum {format_currency(min_cash_abs)}"
            )
            return validated, warnings

        # Stufe 2: Aktive SELL-Orders generieren
        already_selling = {
            d["ticker"] for d in validated
            if d.get("action") == "SELL" and d.get("risk_approved", False)
        }

        target_alloc_map: Dict[str, float] = {
            d["ticker"]: d.get("target_allocation", 0.0)
            for d in validated
            if d.get("action") in ("BUY", "HOLD", "SELL")
        }

        candidates: List[Dict] = []
        for ticker, pos in positions.items():
            if ticker in already_selling:
                continue
            current_alloc = pos.get("market_value", 0) / total_value if total_value > 0 else 0
            target_alloc  = target_alloc_map.get(ticker, current_alloc)
            market_val    = pos.get("market_value", 0)
            if market_val <= 0:
                continue

            md = market_data.get(ticker, {})
            return_30d = md.get("return_30d", 0) or 0.0
            confidence = next(
                (d.get("confidence", 0.5) for d in validated if d["ticker"] == ticker),
                0.5,
            )
            etf_overlap = any(
                ticker in grp and any(
                    t in positions for t in grp if t != ticker
                )
                for grp in ETF_OVERLAP_GROUPS
            )

            overweight_score  = max(0.0, (current_alloc - target_alloc) / 0.05) * 4.0
            momentum_score    = max(0.0, -return_30d / 5.0) * 2.0
            confidence_score  = (1.0 - confidence) * 1.0
            overlap_score     = 1.0 if etf_overlap else 0.0
            priority_score    = overweight_score + momentum_score + confidence_score + overlap_score

            candidates.append({
                "ticker":        ticker,
                "market_value":  market_val,
                "current_alloc": current_alloc,
                "target_alloc":  target_alloc,
                "priority":      priority_score,
                "return_30d":    return_30d,
                "confidence":    confidence,
                "etf_overlap":   etf_overlap,
            })

        candidates.sort(key=lambda x: (-x["priority"], x["market_value"]))

        still_needed = min_cash_abs - recovered
        rebalancing_sells_generated: List[Dict] = []

        for cand in candidates:
            if still_needed <= 0:
                break

            ticker     = cand["ticker"]
            market_val = cand["market_value"]
            sell_amount = min(market_val, still_needed * 1.05)

            if sell_amount < 100.0:
                log.debug(
                    f"[HARD GATE] {ticker}: Sell-Amount ${sell_amount:.2f} "
                    f"zu klein, übersprungen"
                )
                continue

            price = market_data.get(ticker, {}).get("current_price", 0)
            if price <= 0:
                log.warning(f"[HARD GATE] {ticker}: kein Preis verfügbar – SELL übersprungen")
                continue

            new_target = max(0.0, (market_val - sell_amount) / total_value)

            rebalancing_sell = {
                "ticker":            ticker,
                "action":            "SELL",
                "target_allocation": round(new_target, 4),
                "confidence":        1.0,
                "reason": (
                    f"[AUTO-REBALANCING] Cash unter Minimum – "
                    f"freizusetzen: {format_currency(sell_amount)} | "
                    f"Prio-Score: {cand['priority']:.1f} | "
                    f"30d: {cand['return_30d']:+.1f}%"
                ),
                "risk_approved":      True,
                "forced_rebalancing": True,
                "rebalancing":        True,
                "stop_loss":          False,
                "zombie_cleanup":     False,
            }
            rebalancing_sells_generated.append(rebalancing_sell)
            recovered    += sell_amount
            still_needed -= sell_amount

            msg = (
                f"{ticker}: AUTO-REBALANCING SELL {format_currency(sell_amount)} | "
                f"Prio: {cand['priority']:.1f} | "
                f"30d: {cand['return_30d']:+.1f}% | "
                f"Cash danach: {format_currency(recovered)}"
            )
            warnings.append(msg)
            log.warning(f"  💰 {msg}")

        if rebalancing_sells_generated:
            validated = rebalancing_sells_generated + validated
            log.info(
                f"[HARD GATE] {len(rebalancing_sells_generated)} Auto-Rebalancing SELL(s) "
                f"generiert | Cash projiziert: {format_currency(recovered)}"
            )

        if recovered < min_cash_abs:
            remaining_buys = [
                d for d in validated
                if d.get("action") == "BUY" and d.get("risk_approved", False)
            ]
            for d in remaining_buys:
                d["action"]       = "HOLD"
                d["reason"]      += " [HARD GATE: Notfall-Cancel]"
                d["risk_approved"] = False
                msg = f"{d['ticker']}: Notfall-BUY-Cancel (Cash-Invariante)"
                warnings.append(msg)
                log.error(f"  🚨 {msg}")

            log.error(
                f"[HARD GATE] Cash-Invariante nach allen Maßnahmen immer noch verletzt: "
                f"{format_currency(recovered)} < {format_currency(min_cash_abs)}. "
                f"Nächster Run wird nach Broker-Sync korrekt sein."
            )
        else:
            log.info(
                f"[HARD GATE] ✅ Cash-Invariante erfüllt: "
                f"{format_currency(recovered)} >= {format_currency(min_cash_abs)}"
            )

        return validated, warnings

    # ─── Stop-Loss ────────────────────────────────────────────────────────────

    def check_stop_loss(self, positions: Dict, market_data: Dict) -> List[Dict]:
        """Prüft alle Positionen auf Stop-Loss."""
        stop_loss_pct    = self.settings["stop_loss_pct"]
        stop_loss_orders = []
        for ticker, pos in positions.items():
            avg_price     = pos.get("avg_price", 0)
            current_price = market_data.get(ticker, {}).get("current_price", 0)
            if avg_price <= 0 or current_price <= 0:
                continue
            loss_pct = (current_price - avg_price) / avg_price
            if loss_pct < -stop_loss_pct:
                log.warning(
                    f"STOP-LOSS: {ticker} | "
                    f"Kauf: ${avg_price:.2f} | Aktuell: ${current_price:.2f} | "
                    f"Verlust: {loss_pct:.1%}"
                )
                stop_loss_orders.append({
                    "ticker":            ticker,
                    "action":            "SELL",
                    "target_allocation": 0.0,
                    "confidence":        1.0,
                    "reason":            f"Stop-Loss: {loss_pct:.1%} Verlust",
                    "risk_approved":     True,
                    "stop_loss":         True,
                    "decision_id":       f"STOPLOSS_{ticker}",
                })
        return stop_loss_orders

    def get_risk_summary(self) -> Dict:
        return {
            "profile":              self.risk_profile.value,
            "max_position_pct":     self.settings["max_position_pct"],
            "min_cash_pct":         self.settings["min_cash_pct"],
            "stop_loss_pct":        self.settings["stop_loss_pct"],
            "max_trades_per_run":   self.settings["max_trades_per_run"],
            "confidence_threshold": self.settings["confidence_threshold"],
            "top_n_buys":           self.settings.get("max_trades_per_run", TOP_N_BUYS),
        }
    # ── NEW: Portfolio VaR & Volatility Targeting ──────────────────────────

    def calculate_portfolio_var(
        self,
        positions: Dict[str, Dict],
        market_data: Dict[str, Dict],
        total_value: float,
        confidence: float = 0.95,
    ) -> Dict:
        """
        Berechnet parametrisches Portfolio VaR (1-Tag, normal. Verteilung).
        Returns dict mit var_pct, var_usd, component_vars.
        """
        import numpy as np
        from scipy.stats import norm

        if total_value <= 0 or not positions:
            return {"var_pct": 0.0, "var_usd": 0.0, "component_vars": {}}

        weights = []
        vols = []
        tickers = []
        for ticker, pos in positions.items():
            alloc = pos.get("market_value", 0) / total_value
            vol_annual = (market_data.get(ticker, {}).get("volatility_annual_pct") or 20.0) / 100
            vol_daily = vol_annual / (252 ** 0.5)
            weights.append(alloc)
            vols.append(vol_daily)
            tickers.append(ticker)

        if not weights:
            return {"var_pct": 0.0, "var_usd": 0.0, "component_vars": {}}

        w = np.array(weights)
        v = np.array(vols)

        # Simplified: assume zero correlation (conservative approximation)
        portfolio_vol = float(np.sqrt(np.sum((w * v) ** 2)))

        z_score = norm.ppf(confidence)
        var_pct = portfolio_vol * z_score
        var_usd = var_pct * total_value

        component_vars = {
            t: round(w[i] * v[i] * z_score, 4)
            for i, t in enumerate(tickers)
        }

        return {
            "var_pct": round(var_pct, 4),
            "var_usd": round(var_usd, 2),
            "component_vars": component_vars,
            "portfolio_vol_daily": round(portfolio_vol, 4),
        }

    def check_var_limit(
        self,
        positions: Dict[str, Dict],
        market_data: Dict[str, Dict],
        total_value: float,
    ) -> Tuple[bool, str]:
        """
        Prüft ob Portfolio-VaR innerhalb des konfigurierten Limits liegt.
        Gibt (ok, message) zurück.
        """
        max_var = self.settings.get("max_portfolio_var", 0.05)
        var_data = self.calculate_portfolio_var(
            positions, market_data, total_value,
            confidence=self.settings.get("var_confidence", 0.95)
        )
        var_pct = var_data.get("var_pct", 0)
        if var_pct > max_var:
            msg = f"Portfolio VaR {var_pct:.1%} > max {max_var:.1%} – Reduktion empfohlen"
            log.warning(f"[VaR] {msg}")
            return False, msg
        return True, f"VaR OK: {var_pct:.1%} <= {max_var:.1%}"

    def apply_volatility_targeting(
        self,
        decisions: List[Dict],
        market_data: Dict[str, Dict],
        portfolio_vol: float,
    ) -> Tuple[List[Dict], List[str]]:
        """
        Skaliert BUY-Allokationen wenn Portfolio-Volatilität über dem Ziel liegt.
        """
        target_vol = self.settings.get("volatility_target", 0.15)
        warnings = []

        if portfolio_vol <= 0 or portfolio_vol <= target_vol:
            return decisions, warnings

        scale_factor = target_vol / portfolio_vol
        scale_factor = max(0.5, min(1.0, scale_factor))  # cap at 50% reduction

        if scale_factor < 0.95:
            msg = f"Volatility targeting: portfolio_vol={portfolio_vol:.1%} > target={target_vol:.1%} → scale BUYs by {scale_factor:.0%}"
            log.info(f"[VolTarget] {msg}")
            warnings.append(msg)

            for d in decisions:
                if d.get("action") == "BUY" and not d.get("stop_loss") and not d.get("zombie_cleanup"):
                    original = d.get("target_allocation", 0)
                    d["target_allocation"] = round(original * scale_factor, 4)

        return decisions, warnings

    def check_drawdown_protection(
        self,
        portfolio_history: List[Dict],
        current_value: float,
    ) -> Tuple[bool, float]:
        """
        Erkennt signifikanten Drawdown und aktiviert defensive Mode.
        Returns (drawdown_triggered, drawdown_pct).
        """
        if not portfolio_history or len(portfolio_history) < 2:
            return False, 0.0

        peak = max(
            entry.get("portfolio_after", {}).get("total_value", 0)
            for entry in portfolio_history
        )
        if peak <= 0:
            return False, 0.0

        drawdown = (current_value - peak) / peak
        max_trigger = self.settings.get("max_drawdown_trigger", 0.20)

        if drawdown < -max_trigger:
            log.warning(
                f"[DrawdownProtection] Drawdown {drawdown:.1%} > trigger {-max_trigger:.1%} "
                f"– defensive mode empfohlen"
            )
            return True, float(drawdown)

        return False, float(drawdown)

    def check_daily_turnover(
        self,
        planned_trades: List[Dict],
        total_value: float,
    ) -> Tuple[List[Dict], List[str]]:
        """
        Begrenzt täglichen Portfolio-Turnover auf konfigurierten Maximalwert.
        """
        max_turnover = self.settings.get("max_daily_turnover", 0.25)
        max_value = total_value * max_turnover
        warnings = []
        approved = []
        cumulative = 0.0

        for trade in planned_trades:
            trade_value = trade.get("value", 0)
            if cumulative + trade_value > max_value:
                msg = f"{trade['ticker']}: Turnover limit erreicht ({cumulative/total_value:.1%} >= {max_turnover:.1%})"
                warnings.append(msg)
                log.info(f"[TurnoverLimit] {msg}")
                # Convert to HOLD instead of dropping
                trade = dict(trade, action="HOLD", skip_reason="turnover_limit")
            else:
                cumulative += trade_value
            approved.append(trade)

        return approved, warnings

    def calculate_correlation_adjusted_exposure(
        self,
        positions: Dict[str, Dict],
        total_value: float,
    ) -> Dict[str, float]:
        """
        Berechnet echte Gesamt-Exposure unter Berücksichtigung von ETF-Overlap.
        QQQ + XLK + NVDA + AMD + AAPL → kumulierte Tech-Exposure.
        """
        from config import SECTOR_CLASSIFICATION, ETF_SECTOR_WEIGHTS
        exposure: Dict[str, float] = {}

        for ticker, pos in positions.items():
            alloc = pos.get("market_value", 0) / total_value if total_value > 0 else 0

            # ETF: verwende intern gewichtete Sektoren
            if ticker in ETF_SECTOR_WEIGHTS:
                for sector, weight in ETF_SECTOR_WEIGHTS[ticker].items():
                    exposure[sector] = exposure.get(sector, 0) + alloc * weight
            else:
                sector = SECTOR_CLASSIFICATION.get(ticker, "other")
                exposure[sector] = exposure.get(sector, 0) + alloc

        return {k: round(v, 4) for k, v in exposure.items()}