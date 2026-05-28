"""
portfolio_rebalancer.py – Integration der Rebalancing Engine in den Trading Bot
================================================================================
Vermittelt zwischen RebalancingEngine, PortfolioManager und RiskManager.
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
from rebalancing_engine import PortfolioRebalancingEngine, TradeSuggestion, RebalancingProposal


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
    Integriert die PortfolioRebalancingEngine in den Bot-Workflow.
    """

    def __init__(self, risk_profile=None):
        self.risk_profile = risk_profile or ACTIVE_RISK_PROFILE
        self.engine = PortfolioRebalancingEngine(risk_profile)
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
        Hauptmethode: Ruft die Rebalancing Engine auf und wandelt die Ergebnisse um.
        Returns: (decisions, target_weights, cash_target, rationale)
        """
        if not self.enabled:
            log.debug("Portfolio Rebalancing Engine deaktiviert (config)")
            return [], current_weights, cash / total_value if total_value > 0 else 0, ""

        # Extrahiere Momentum und Volatilität aus market_data
        momentum = {}
        volatility = {}
        for ticker, data in market_data.items():
            # Momentum: 20d return (falls vorhanden)
            mom = data.get("return_20d")
            if mom is not None:
                momentum[ticker] = mom
            else:
                momentum[ticker] = 0.0
            # Volatilität (annualisiert)
            vol = data.get("volatility_annual_pct")
            if vol is not None:
                volatility[ticker] = vol
            else:
                volatility[ticker] = 20.0  # Fallback

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

        # Market Volatilität für Kosten-Schätzung (SPY oder Default)
        market_vol = volatility.get("SPY", 15.0) / 100.0

        # Rebalancing durchführen
        proposal = self.engine.optimize(
            scores=scores,
            momentum=momentum,
            volatility=volatility,
            current_weights=current_weights,
            cash=cash,
            total_value=total_value,
            regime=regime_str,
            sector_map=None,  # wird intern aus config geladen
            correlation_groups=None,
            market_volatility=market_vol,
        )

        # In handelbare Entscheidungen umwandeln
        decisions = []
        for trade in proposal.trades:
            # Nur Trades mit Netto-Nutzen > Mindestschwelle (bereits im Vorschlag gefiltert)
            action = trade.action
            if action == "SWAP_OUT":
                # SWAP_OUT wird separat behandelt; wir ignorieren hier und lassen SWAP_IN das BUY sein
                continue
            if action == "SWAP_IN":
                action = "BUY"
            target_weight = proposal.final_allocations.get(trade.ticker, 0.0)
            decision = RebalancingDecision(
                ticker=trade.ticker,
                action=action,
                target_weight=target_weight,
                confidence=min(0.95, 0.5 + (trade.expected_improvement / 100.0)),
                reason=trade.reason,
                expected_improvement=trade.expected_improvement,
                transaction_cost=trade.transaction_cost,
            )
            decisions.append(decision)

        # Auf max_trades begrenzen (nach Netto-Nutzen)
        decisions.sort(key=lambda x: x.expected_improvement - x.transaction_cost, reverse=True)
        decisions = decisions[:self.max_trades]

        log.info(f"PortfolioRebalancer: {len(decisions)} Trades generiert, "
                 f"Nettoverbesserung {proposal.net_improvement:.2f}%, "
                 f"Kosten {proposal.total_transaction_cost:.2f}%")

        return decisions, proposal.final_allocations, proposal.cash_target, proposal.rationale

    def generate_prompt_section(self, proposal: RebalancingProposal) -> str:
        """Erzeugt einen Abschnitt für den LLM-Prompt, der die Rebalancing-Überlegungen erklärt."""
        if not proposal.trades:
            return "Portfolio Rebalancing Engine: Keine vorteilhaften Trades identifiziert."

        lines = [
            "=== PORTFOLIO REBALANCING ENGINE ===",
            f"Portfolio Quality before: {proposal.portfolio_quality_before:.2%}",
            f"Portfolio Quality after:  {proposal.portfolio_quality_after:.2%}",
            f"Net improvement: {proposal.net_improvement:.2%}",
            f"Transaction cost impact: {proposal.total_transaction_cost:.2%}",
            "",
            "Recommended trades:"
        ]
        for t in proposal.trades:
            lines.append(f"  {t.action} {t.ticker}: {t.delta_weight:+.1%} weight, "
                         f"benefit {t.expected_improvement:.2f}%, cost {t.transaction_cost:.2f}%")
        lines.append(f"\nRationale: {proposal.rationale}")
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
        # Durchschnittliche Drift über alle Assets
        avg_drift = total_drift / max(1, len(target_weights))
        log.debug(f"Rebalancing Drift: {avg_drift:.2%} (threshold {self.min_drift:.2%})")
        return avg_drift >= self.min_drift
