"""
portfolio_rebalancer.py – Integration der Rebalancing Engine V2 in den Trading Bot
==================================================================================
Vermittelt zwischen RebalancerV2, PortfolioManager und RiskManager.
Wandelt die optimierten Allokationen in handelbare Entscheidungen um.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from logger import log
from config import (
    ACTIVE_RISK_PROFILE,
    RISK_SETTINGS,
    REBALANCING_ENGINE_ENABLED,
    REBALANCING_MIN_DRIFT,
    REBALANCING_MAX_TRADES,
    TRANSACTION_COST_MODEL,
)
from rebalancer_v2 import RebalancerV2, AssetEdge, SwapCandidate


@dataclass
class RebalancingDecision:
    """Eine handelbare Entscheidung aus der Rebalancing Engine."""
    ticker: str
    action: str          # BUY, SELL, HOLD
    target_weight: float
    confidence: float
    reason: str
    expected_improvement: float
    transaction_cost: float


class PortfolioRebalancer:
    """
    Integriert die PortfolioRebalancingEngine V2 in den Bot-Workflow.
    """

    def __init__(self, risk_profile=None):
        self.risk_profile = risk_profile or ACTIVE_RISK_PROFILE
        self.engine = RebalancerV2(risk_profile)
        self.settings = RISK_SETTINGS[self.risk_profile]
        self.enabled = REBALANCING_ENGINE_ENABLED
        self.min_drift = REBALANCING_MIN_DRIFT
        self.max_trades = REBALANCING_MAX_TRADES

    def optimize_portfolio(
        self,
        scores: Dict[str, float],           # ticker -> quality score (0-100)
        market_data: Dict[str, Dict],       # market_data vom DataCollector (enthält momentum, volatility)
        current_weights: Dict[str, float],  # aktuelle Allokationen
        cash: float,
        total_value: float,
        regime_state,                       # von MarketRegimeDetector
    ) -> Tuple[List[RebalancingDecision], Dict[str, float], float, str]:
        """
        Hauptmethode: Ruft die Rebalancing Engine V2 auf und wandelt die Ergebnisse um.
        Returns: (decisions, target_weights, cash_target, rationale)
        """
        if not self.enabled:
            log.debug("Portfolio Rebalancing Engine deaktiviert (config)")
            return [], current_weights, cash / total_value if total_value > 0 else 0, ""

        # Extrahiere Momentum, Volatilität und Konfidenz aus market_data / scores
        # Wir brauchen confidences – für jetzt: leite aus Score ab (Score/100)
        momentums = {}
        volatilities = {}
        confidences = {}
        for ticker, data in market_data.items():
            # Momentum: 20d return (falls vorhanden)
            mom = data.get("return_20d")
            momentums[ticker] = mom if mom is not None else 0.0
            # Volatilität (annualisiert)
            vol = data.get("volatility_annual_pct")
            volatilities[ticker] = vol if vol is not None else 20.0
            # Konfidenz aus Score (vereinfacht)
            sc = scores.get(ticker, 50.0)
            confidences[ticker] = sc / 100.0

        # Regime als String
        regime_str = "SIDEWAYS"
        if regime_state is not None:
            regime_val = getattr(regime_state, 'regime', None)
            if regime_val is not None:
                if hasattr(regime_val, 'value'):
                    regime_str = regime_val.value.upper()
                else:
                    regime_str = str(regime_val).upper()
            else:
                regime_str = getattr(regime_state, 'label', 'SIDEWAYS').upper()

        # Rebalancing V2 durchführen
        decisions_raw, target_weights, cash_target, rationale = self.engine.rebalance(
            scores=scores,
            confidences=confidences,
            momentums=momentums,
            volatilities=volatilities,
            current_weights=current_weights,
            cash=cash,
            total_value=total_value,
            regime=regime_str,
        )

        # In RebalancingDecision-Objekte umwandeln
        decisions = []
        for d in decisions_raw:
            decision = RebalancingDecision(
                ticker=d["ticker"],
                action=d["action"],
                target_weight=d["target_allocation"],
                confidence=d["confidence"],
                reason=d["reason"],
                expected_improvement=0.0,  # V2 liefert noch keine einzelne Verbesserung pro Trade
                transaction_cost=0.0,
            )
            decisions.append(decision)

        # Auf max_trades begrenzen (Priorität nach action? SELLs zuerst)
        sell_decisions = [d for d in decisions if d.action == "SELL"]
        buy_decisions = [d for d in decisions if d.action == "BUY"]
        # Kapazität: max_trades insgesamt
        total_planned = len(sell_decisions) + len(buy_decisions)
        if total_planned > self.max_trades:
            # Bevorzuge SELLs (machen Cash frei), dann BUYs nach Konfidenz
            buy_decisions.sort(key=lambda x: x.confidence, reverse=True)
            keep_buys = buy_decisions[:max(0, self.max_trades - len(sell_decisions))]
            decisions = sell_decisions + keep_buys

        log.info(f"PortfolioRebalancer V2: {len(decisions)} Trades generiert (SELL={len(sell_decisions)}, BUY={len(buy_decisions)})")
        if rationale:
            log.info(f"  Rationale: {rationale[:200]}")

        return decisions, target_weights, cash_target, rationale

    def generate_prompt_section(self, decisions: List[RebalancingDecision], target_weights: Dict[str, float], cash_target: float, rationale: str) -> str:
        """Erzeugt einen Abschnitt für den LLM-Prompt, der die Rebalancing-Überlegungen erklärt."""
        if not decisions:
            return "Portfolio Rebalancing Engine V2: Keine vorteilhaften Swaps identifiziert."

        lines = [
            "=== PORTFOLIO REBALANCING ENGINE V2 (Edge-aware) ===",
            f"Cash target: {cash_target:.1%}",
            "",
            "Recommended trades:"
        ]
        for d in decisions:
            lines.append(f"  {d.action} {d.ticker}: target weight {d.target_weight:.1%} (conf: {d.confidence:.0%}) – {d.reason}")
        lines.append(f"\nRationale: {rationale[:300]}")
        lines.append("Diese Vorschläge basieren auf Edge-Differenz, Transaktionskosten und Turnover-Limit.")
        return "\n".join(lines)

    def should_rebalance(self, current_weights: Dict[str, float], target_weights: Dict[str, float]) -> bool:
        """
        Prüft, ob die aktuelle Abweichung groß genug ist, um überhaupt zu rebalancen.
        """
        if not target_weights:
            return False
        total_drift = 0.0
        for ticker, target in target_weights.items():
            current = current_weights.get(ticker, 0.0)
            total_drift += abs(target - current)
        avg_drift = total_drift / max(1, len(target_weights))
        log.debug(f"Rebalancing Drift: {avg_drift:.2%} (threshold {self.min_drift:.2%})")
        return avg_drift >= self.min_drift
