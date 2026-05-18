"""
AI Trading Bot - Decision Optimization Filter
===============================================
Transforms raw AI decisions into high-quality, actionable trades.

Philosophy: NEVER default to inactivity. Instead:
  - Adjust allocations
  - Downgrade BUY → smaller BUY
  - Convert blocked SELL → PARTIAL REDUCE
  - Always keep top signals alive

Key Rules:
  1. No full inactivity in BULL regime
  2. Overbought (RSI > 75) → reduce size, don't block
  3. High confidence (≥80%) → allow slight overweight
  4. Small drift (<1.5%) + high confidence → micro-adjustment
  5. Blocked SELL → partial reduce (-25%) instead of HOLD
  6. Always keep top 2–3 strongest signals active
  7. BULL override: bias toward action, deploy capital
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from logger import log


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

RSI_OVERBOUGHT = 75
RSI_OVERSOLD = 30
HIGH_CONFIDENCE = 0.80
SMALL_CONFIDENCE = 0.50
DRIFT_THRESHOLD = 0.015          # 1.5% – below this → normally HOLD
OVERWEIGHT_ALLOWANCE = 0.02      # +2% overweight for high-confidence
OVERBOUGHT_SIZE_FACTOR = 0.50    # Halve allocation if RSI > 75
PARTIAL_SELL_FACTOR = 0.25       # 25% position reduce for blocked SELLs
MIN_TRADE_ALLOCATION = 0.005     # 0.5% minimum meaningful trade
TOP_SIGNAL_COUNT = 3             # Always keep top N signals active
CORRELATED_BUY_GROUPS = [
    ["SPY", "QQQ", "XLK", "VT", "AAPL", "MSFT", "AMZN"],
    ["QQQ", "XLK", "AAPL", "MSFT", "NVDA", "AMD"],
]


# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    """Result of the decision filter for a single ticker."""
    ticker: str
    original_action: str
    final_action: str
    original_allocation: float
    final_allocation: float
    confidence: float
    reason: str
    adjustment_notes: List[str] = field(default_factory=list)
    risk_approved: bool = True
    priority_score: float = 0.0
    decision_id: str = ""

    @property
    def was_modified(self) -> bool:
        return (
            self.original_action != self.final_action
            or abs(self.original_allocation - self.final_allocation) > 0.001
        )


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _regime_is_bull(regime_state) -> bool:
    """Returns True if market regime is bullish."""
    if regime_state is None:
        return False
    regime = getattr(regime_state, "regime", None)
    if regime is None:
        regime = regime_state.get("regime", "")
    if isinstance(regime, Enum):
        regime = regime.name
    return str(regime).upper() == "BULL"


def _get_rsi(market_data: Dict, ticker: str) -> Optional[float]:
    """Safely extract RSI from market data dict."""
    return market_data.get(ticker, {}).get("rsi_14")


def _get_current_alloc(positions: Dict, ticker: str, total_value: float) -> float:
    """Current allocation % for a ticker."""
    if total_value <= 0:
        return 0.0
    pos = positions.get(ticker, {})
    return pos.get("market_value", 0.0) / total_value


def _priority_score(decision: Dict, market_data: Dict) -> float:
    """
    Composite score for ranking signals by strength.
    Higher = more important to keep active.
    """
    confidence = decision.get("confidence", 0.5)
    action = decision.get("action", "HOLD")
    ticker = decision.get("ticker", "")
    rsi = _get_rsi(market_data, ticker)

    score = confidence * 100

    # Bonus for strong directional signals
    if action == "BUY":
        score += 10
        if rsi and rsi < RSI_OVERSOLD:
            score += 15  # Oversold BUY = very strong
    elif action == "SELL":
        score += 8
        if rsi and rsi > RSI_OVERBOUGHT:
            score += 12  # Overbought SELL = strong

    return round(score, 2)


# ─────────────────────────────────────────────────────────────
# CORE FILTER
# ─────────────────────────────────────────────────────────────

class DecisionFilter:
    """
    Decision optimization layer.
    Improves raw AI decisions rather than removing them.
    """

    def __init__(
        self,
        risk_settings: Dict,
        positions: Dict,
        total_value: float,
        market_data: Dict,
        regime_state=None,
    ):
        self.risk_settings = risk_settings
        self.positions = positions
        self.total_value = total_value
        self.market_data = market_data
        self.regime_state = regime_state
        self.is_bull = _regime_is_bull(regime_state)
        self.max_position_pct = risk_settings.get("max_position_pct", 0.20)
        self.min_cash_pct = risk_settings.get("min_cash_pct", 0.10)

        log.debug(
            f"DecisionFilter init | Bull={self.is_bull} | "
            f"max_pos={self.max_position_pct:.0%} | "
            f"min_cash={self.min_cash_pct:.0%}"
        )

    # ─────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ─────────────────────────────────────────────────────────

    def filter(self, decisions: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Main filter method.

        Args:
            decisions: Raw AI decisions list. Each dict has keys:
                       ticker, action, target_allocation, confidence,
                       reason, risk_approved.

        Returns:
            (optimized_decisions, warnings)
        """
        if not decisions:
            return [], []

        warnings: List[str] = []

        # Step 1: Score and rank all decisions
        scored = self._score_decisions(decisions)

        # Step 1.5: Reduce redundant correlated BUYs
        scored = self._reduce_correlated_buys(scored, warnings)

        # Step 1.7: Check sector concentration risk
        scored = self._check_concentration_risk(scored, warnings)

        # Step 2: Process each decision individually
        results: List[FilterResult] = []
        for decision in scored:
            result = self._process_decision(decision, warnings)
            results.append(result)

        # Step 3: BULL regime enforcement – ensure at least 1–2 actions
        results = self._enforce_bull_activity(results, warnings)

        # Step 4: Priority enforcement – keep top signals active
        results = self._enforce_top_signals(results, warnings)

        # Step 5: Convert FilterResults back to decision dicts
        optimized = self._to_decisions(results)

        # Summary log
        buys  = sum(1 for d in optimized if d["action"] == "BUY")
        sells = sum(1 for d in optimized if d["action"] == "SELL")
        holds = sum(1 for d in optimized if d["action"] == "HOLD")
        modified = sum(1 for r in results if r.was_modified)

        log.info(
            f"[DecisionFilter] {len(decisions)} → {len(optimized)} decisions | "
            f"BUY={buys} SELL={sells} HOLD={holds} | "
            f"Modified={modified} | Bull={self.is_bull}"
        )

        return optimized, warnings

    # ─────────────────────────────────────────────────────────
    # SCORING
    # ─────────────────────────────────────────────────────────

    def _score_decisions(self, decisions: List[Dict]) -> List[Dict]:
        """Attach priority score to each decision, sort descending."""
        for d in decisions:
            d["_priority"] = _priority_score(d, self.market_data)
        return sorted(decisions, key=lambda x: x["_priority"], reverse=True)

    def _reduce_correlated_buys(self, decisions: List[Dict], warnings: List[str]) -> List[Dict]:
        """Downgrade redundant correlated BUY signals to HOLD when a leader exists."""
        for group in CORRELATED_BUY_GROUPS:
            buys = [d for d in decisions if d.get("action") == "BUY" and d.get("ticker") in group]
            if len(buys) <= 1:
                continue

            buys.sort(key=lambda d: (d.get("confidence", 0.0), d.get("_priority", 0)), reverse=True)
            leader = buys[0]
            for other in buys[1:]:
                if other["confidence"] + 0.10 < leader["confidence"]:
                    note = (
                        f"{other['ticker']}: correlated with {leader['ticker']} BUY -> downgraded to HOLD"
                    )
                    other["action"] = "HOLD"
                    other["target_allocation"] = other.get("target_allocation", 0.0)
                    other["risk_approved"] = False
                    other["reason"] = f"{other.get('reason','')} [FILTER: {note}]"
                    warnings.append(note)
        return decisions

    def _check_concentration_risk(self, decisions: List[Dict], warnings: List[str]) -> List[Dict]:
        """
        🔥 SECTOR CONCENTRATION CONTROL
        
        Rule: Max 3 BUYs per Sektor
        
        Wenn überschritten:
          - Sortiere BUYs nach confidence (ascending = schwächste zuerst)
          - Konvertiere schwächste BUYs zu HOLD
          - NICHT löschen, nur action ändern
        
        Sectoren werden ermittelt aus SECTOR_CLASSIFICATION config.
        """
        from config import SECTOR_CLASSIFICATION
        
        # Gruppiere BUYs nach Sektor
        buy_decisions = [d for d in decisions if d.get("action") == "BUY"]
        if len(buy_decisions) <= 3:
            return decisions  # Kein Problem
        
        # Gruppiere nach Sektor
        sector_buys: Dict[str, List[Dict]] = {}
        for decision in buy_decisions:
            ticker = decision.get("ticker", "")
            sector = SECTOR_CLASSIFICATION.get(ticker, "diversified")
            sector_buys.setdefault(sector, []).append(decision)
        
        # Prüfe Limit pro Sektor (max 3)
        max_buys_per_sector = 3
        sectors_over_limit = {s: bs for s, bs in sector_buys.items() if len(bs) > max_buys_per_sector}
        
        if not sectors_over_limit:
            return decisions  # Alle Sektoren okay
        
        # Für jeden Sektor über Limit: schwächste BUYs zu HOLD konvertieren
        for sector, buys_in_sector in sectors_over_limit.items():
            excess_count = len(buys_in_sector) - max_buys_per_sector
            
            # Sortiere nach confidence (ascending = niedrigste zuerst = schwächste)
            buys_in_sector.sort(key=lambda d: d.get("confidence", 0.0))
            
            # Konvertiere die schwächsten zu HOLD
            for i in range(excess_count):
                weak_buy = buys_in_sector[i]
                ticker = weak_buy.get("ticker", "")
                conf = weak_buy.get("confidence", 0.0)
                
                weak_buy["action"] = "HOLD"
                weak_buy["risk_approved"] = False
                weak_buy["reason"] = (
                    f"{weak_buy.get('reason', '')} "
                    f"[CONCENTRATION LIMIT: {sector} sector limit ({max_buys_per_sector} BUYs max)]"
                )
                
                note = (
                    f"{ticker}: Sector concentration limit → BUY→HOLD "
                    f"({sector} has {len(buys_in_sector)} BUYs, max {max_buys_per_sector})"
                )
                warnings.append(note)
                log.info(f"  Concentration: {note}")
        
        return decisions

    # ─────────────────────────────────────────────────────────
    # SINGLE DECISION PROCESSING
    # ─────────────────────────────────────────────────────────

    def _process_decision(self, decision: Dict, warnings: List[str]) -> FilterResult:
        ticker = decision.get("ticker", "?")
        action = decision.get("action", "HOLD")
        target_alloc = float(decision.get("target_allocation", 0.0))
        confidence = float(decision.get("confidence", 0.5))
        reason = decision.get("reason", "")
        priority = decision.get("_priority", 0.0)

        notes: List[str] = []
        final_action = action
        final_alloc = target_alloc

        current_alloc = _get_current_alloc(self.positions, ticker, self.total_value)
        rsi = _get_rsi(self.market_data, ticker)
        drift = abs(target_alloc - current_alloc)

        if action == "BUY":
            final_action, final_alloc, buy_notes = self._optimize_buy(
                ticker, target_alloc, current_alloc, confidence, drift, rsi, warnings
            )
            notes.extend(buy_notes)

        elif action == "SELL":
            final_action, final_alloc, sell_notes = self._optimize_sell(
                ticker, target_alloc, current_alloc, confidence, rsi, warnings
            )
            notes.extend(sell_notes)

        elif action == "HOLD":
            final_action, final_alloc, hold_notes = self._optimize_hold(
                ticker, target_alloc, current_alloc, confidence, drift, rsi, warnings
            )
            notes.extend(hold_notes)

        return FilterResult(
            ticker=ticker,
            original_action=action,
            final_action=final_action,
            original_allocation=target_alloc,
            final_allocation=final_alloc,
            confidence=confidence,
            reason=reason,
            adjustment_notes=notes,
            risk_approved=True,
            priority_score=priority,
            decision_id=decision.get("decision_id", ""),
        )

    # ─────────────────────────────────────────────────────────
    # BUY OPTIMIZATION
    # ─────────────────────────────────────────────────────────

    def _optimize_buy(
        self,
        ticker: str,
        target_alloc: float,
        current_alloc: float,
        confidence: float,
        drift: float,
        rsi: Optional[float],
        warnings: List[str],
    ) -> Tuple[str, float, List[str]]:
        notes = []
        final_alloc = target_alloc

        # Rule 1: Avoid BUYs if target allocation is already met
        if target_alloc <= current_alloc + 1e-6:
            notes.append(
                f"{ticker}: current allocation {current_alloc:.1%} already meets target {target_alloc:.1%} -> HOLD"
            )
            return "HOLD", current_alloc, notes

        # Rule 2: Overbought → reduce size instead of blocking
        if rsi and rsi > RSI_OVERBOUGHT:
            original = final_alloc
            final_alloc = current_alloc + (final_alloc - current_alloc) * OVERBOUGHT_SIZE_FACTOR
            final_alloc = max(final_alloc, current_alloc + MIN_TRADE_ALLOCATION)
            note = f"{ticker}: RSI={rsi:.0f} overbought → BUY halved ({original:.1%}→{final_alloc:.1%})"
            notes.append(note)
            warnings.append(note)

        # Rule 3: High confidence → allow slight overweight
        if confidence >= HIGH_CONFIDENCE and final_alloc <= self.max_position_pct:
            cap = self.max_position_pct + OVERWEIGHT_ALLOWANCE
            final_alloc = min(final_alloc, cap)
            if final_alloc > self.max_position_pct:
                notes.append(f"{ticker}: High confidence {confidence:.0%} → +2% overweight allowed")

        # Rule 4: Small drift + high confidence → allow micro-adjustment
        if drift < DRIFT_THRESHOLD and confidence >= HIGH_CONFIDENCE:
            micro_alloc = current_alloc + MIN_TRADE_ALLOCATION * 3  # 1.5% micro
            final_alloc = max(final_alloc, micro_alloc)
            notes.append(f"{ticker}: Small drift but high conf {confidence:.0%} → micro-BUY")

        # ── HARD LIMIT: absolute max_position_pct (Phase 4 Production Safety) ──
        # No exception allowed – neither by AI decision, regime, nor confidence.
        final_alloc = self.max_position_cap_hard_limit(ticker, final_alloc, notes)

        # Cap at max position (legacy soft cap, now superseded by hard limit above)
        if final_alloc > self.max_position_pct + OVERWEIGHT_ALLOWANCE:
            final_alloc = self.max_position_pct + OVERWEIGHT_ALLOWANCE
            notes.append(f"{ticker}: Capped at max position {final_alloc:.1%}")

        # Ensure meaningful trade size
        if final_alloc - current_alloc < MIN_TRADE_ALLOCATION:
            if drift < DRIFT_THRESHOLD and confidence < HIGH_CONFIDENCE:
                notes.append(f"{ticker}: BUY → HOLD (drift too small, low confidence)")
                return "HOLD", current_alloc, notes
            final_alloc = current_alloc + MIN_TRADE_ALLOCATION

        return "BUY", round(final_alloc, 4), notes

    # ─────────────────────────────────────────────────────────
    # PHASE 4 PRODUCTION SAFETY: HARD POSITION CAP
    # ─────────────────────────────────────────────────────────

    def max_position_cap_hard_limit(
        self,
        ticker: str,
        proposed_alloc: float,
        notes: List[str],
    ) -> float:
        """
        FINAL SAFETY LAYER – Phase 4 Production Hardening.

        Ensures no position ever exceeds max_position_pct from config.
        This cap is ABSOLUTE: no exception is granted by AI reasoning,
        regime state, confidence level, or any other factor.

        If the proposed allocation exceeds the cap:
          → Silently clamp to max_position_pct
          → Log the adjustment
          → NEVER block or reject the trade (just reduce size)

        Returns the (possibly clamped) allocation.
        """
        if proposed_alloc > self.max_position_pct:
            note = (
                f"{ticker}: HARD CAP enforced "
                f"({proposed_alloc:.1%} → {self.max_position_pct:.1%}, "
                f"max_position_pct={self.max_position_pct:.1%})"
            )
            notes.append(note)
            log.warning(f"[HARD CAP] {note}")
            return self.max_position_pct
        return proposed_alloc

    # ─────────────────────────────────────────────────────────
    # SELL OPTIMIZATION
    # ─────────────────────────────────────────────────────────

    def _optimize_sell(
        self,
        ticker: str,
        target_alloc: float,
        current_alloc: float,
        confidence: float,
        rsi: Optional[float],
        warnings: List[str],
    ) -> Tuple[str, float, List[str]]:
        notes = []
        final_alloc = target_alloc

        if current_alloc <= 0:
            notes.append(f"{ticker}: no current position -> SELL converted to HOLD")
            return "HOLD", current_alloc, notes

        # Rule 5: If SELL would be very aggressive → partial reduce first
        reduction = current_alloc - target_alloc

        if reduction > self.max_position_pct * 0.5 and confidence < HIGH_CONFIDENCE:
            # Large sell, low confidence → only do partial
            partial_alloc = current_alloc - (reduction * PARTIAL_SELL_FACTOR)
            note = (
                f"{ticker}: Large SELL downgraded to PARTIAL REDUCE "
                f"({current_alloc:.1%}→{partial_alloc:.1%}) confidence={confidence:.0%}"
            )
            notes.append(note)
            warnings.append(note)
            final_alloc = round(partial_alloc, 4)

        # RSI oversold → soften sell
        if rsi and rsi < RSI_OVERSOLD and confidence < HIGH_CONFIDENCE:
            softened = current_alloc - (reduction * PARTIAL_SELL_FACTOR)
            note = f"{ticker}: RSI={rsi:.0f} oversold → SELL softened to partial ({softened:.1%})"
            notes.append(note)
            warnings.append(note)
            final_alloc = round(max(final_alloc, softened), 4)

        # Ensure sell makes sense (not selling to below 0)
        final_alloc = max(0.0, final_alloc)

        # If reduction becomes negligible → HOLD
        actual_reduction = current_alloc - final_alloc
        if actual_reduction < MIN_TRADE_ALLOCATION and target_alloc > 0:
            notes.append(f"{ticker}: SELL reduction negligible → HOLD")
            return "HOLD", current_alloc, notes

        return "SELL", final_alloc, notes

    # ─────────────────────────────────────────────────────────
    # HOLD OPTIMIZATION
    # ─────────────────────────────────────────────────────────

    def _optimize_hold(
        self,
        ticker: str,
        target_alloc: float,
        current_alloc: float,
        confidence: float,
        drift: float,
        rsi: Optional[float],
        warnings: List[str],
    ) -> Tuple[str, float, List[str]]:
        notes = []

        # Rule 4: High confidence even on HOLD → micro-adjustment BUY
        if confidence >= HIGH_CONFIDENCE and self.is_bull:
            micro_alloc = current_alloc + MIN_TRADE_ALLOCATION * 3
            if micro_alloc <= self.max_position_pct + OVERWEIGHT_ALLOWANCE:
                note = f"{ticker}: HOLD upgraded to micro-BUY (conf={confidence:.0%}, bull regime)"
                notes.append(note)
                return "BUY", round(micro_alloc, 4), notes

        # Otherwise stay HOLD
        notes.append(f"{ticker}: HOLD confirmed (drift={drift:.1%}, conf={confidence:.0%})")
        return "HOLD", current_alloc, notes

    # ─────────────────────────────────────────────────────────
    # BULL REGIME ENFORCEMENT
    # ─────────────────────────────────────────────────────────

    def _enforce_bull_activity(
        self, results: List[FilterResult], warnings: List[str]
    ) -> List[FilterResult]:
        """Rule 1 + 7: In BULL regime, ensure at least 1–2 active trades."""
        if not self.is_bull:
            return results

        active = [r for r in results if r.final_action in ("BUY", "SELL")]

        if len(active) >= 1:
            return results  # Already have activity

        # No active trades in BULL → force top signals into small BUYs
        buys_by_priority = sorted(
            [r for r in results if r.original_action == "BUY"],
            key=lambda x: x.priority_score,
            reverse=True,
        )

        activated = 0
        for result in buys_by_priority[:2]:
            current_alloc = _get_current_alloc(self.positions, result.ticker, self.total_value)
            new_alloc = current_alloc + MIN_TRADE_ALLOCATION * 4  # 2% deployment
            new_alloc = min(new_alloc, self.max_position_pct)  # hard cap

            note = (
                f"{result.ticker}: BULL override → forced small BUY "
                f"({current_alloc:.1%}→{new_alloc:.1%})"
            )
            result.final_action = "BUY"
            result.final_allocation = round(new_alloc, 4)
            result.adjustment_notes.append(note)
            warnings.append(note)
            activated += 1

        if activated:
            log.info(f"[DecisionFilter] BULL override: activated {activated} small BUY(s)")

        return results

    # ─────────────────────────────────────────────────────────
    # TOP SIGNAL ENFORCEMENT
    # ─────────────────────────────────────────────────────────

    def _enforce_top_signals(
        self, results: List[FilterResult], warnings: List[str]
    ) -> List[FilterResult]:
        """Rule 6: Always keep top N strongest signals active (not HOLDed away)."""
        top_results = sorted(results, key=lambda x: x.priority_score, reverse=True)
        top_n = top_results[:TOP_SIGNAL_COUNT]

        for result in top_n:
            if result.original_action in ("BUY", "SELL") and result.final_action == "HOLD":
                # This strong signal was silenced → restore with reduced size
                current_alloc = _get_current_alloc(
                    self.positions, result.ticker, self.total_value
                )

                if result.original_action == "BUY":
                    restored_alloc = current_alloc + MIN_TRADE_ALLOCATION * 4
                    restored_alloc = min(restored_alloc, self.max_position_pct)  # hard cap
                    note = (
                        f"{result.ticker}: Top signal ({result.priority_score:.0f}) "
                        f"restored from HOLD → small BUY ({restored_alloc:.1%})"
                    )
                    result.final_action = "BUY"
                    result.final_allocation = round(restored_alloc, 4)
                else:
                    # SELL restored as partial reduce
                    restored_alloc = current_alloc * (1 - PARTIAL_SELL_FACTOR)
                    note = (
                        f"{result.ticker}: Top SELL signal ({result.priority_score:.0f}) "
                        f"restored as partial reduce ({current_alloc:.1%}→{restored_alloc:.1%})"
                    )
                    result.final_action = "SELL"
                    result.final_allocation = round(restored_alloc, 4)

                result.adjustment_notes.append(note)
                warnings.append(note)

        return results

    # ─────────────────────────────────────────────────────────
    # OUTPUT CONVERSION
    # ─────────────────────────────────────────────────────────

    def _to_decisions(self, results: List[FilterResult]) -> List[Dict]:
        """Convert FilterResult list back to decision dicts."""
        output = []
        for r in results:
            d = {
                "ticker": r.ticker,
                "action": r.final_action,
                "target_allocation": r.final_allocation,
                "confidence": r.confidence,
                "reason": r.reason,
                "risk_approved": r.risk_approved,
                "decision_id": r.decision_id,
                # Metadata for journal/debugging
                "_original_action": r.original_action,
                "_original_allocation": r.original_allocation,
                "_filter_notes": r.adjustment_notes,
                "_priority": r.priority_score,
                "_modified": r.was_modified,
            }
            output.append(d)
        return output


# ─────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION (used by ai_analysis.py / main.py)
# ─────────────────────────────────────────────────────────────

def apply_decision_filter(
    decisions: List[Dict],
    risk_settings: Dict,
    positions: Dict,
    total_value: float,
    market_data: Dict,
    regime_state=None,
) -> Tuple[List[Dict], List[str]]:
    """
    Drop-in replacement for the old filter_decisions() calls.

    Usage in ai_analysis.py / portfolio_manager.py:
        from decision_filter import apply_decision_filter
        decisions, warnings = apply_decision_filter(
            raw_decisions, risk_settings, positions, total_value, market_data, regime_state
        )

    Returns:
        (optimized_decisions, adjustment_warnings)
    """
    f = DecisionFilter(
        risk_settings=risk_settings,
        positions=positions,
        total_value=total_value,
        market_data=market_data,
        regime_state=regime_state,
    )
    return f.filter(decisions)