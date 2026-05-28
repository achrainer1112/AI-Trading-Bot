"""
portfolio_rebalancer.py – Integration des Continuous Portfolio Optimizer
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass

from logger import log
from config import ACTIVE_RISK_PROFILE, REBALANCING_ENGINE_ENABLED, REBALANCING_MAX_TRADES
from cpo_engine import ContinuousPortfolioOptimizer


@dataclass
class RebalancingDecision:
    ticker: str
    action: str
    target_weight: float
    confidence: float
    reason: str


class PortfolioRebalancer:
    def __init__(self, risk_profile=None):
        self.risk_profile = risk_profile or ACTIVE_RISK_PROFILE
        self.engine = ContinuousPortfolioOptimizer(risk_profile)
        self.enabled = REBALANCING_ENGINE_ENABLED
        self.max_trades = REBALANCING_MAX_TRADES

    def optimize_portfolio(
        self,
        scores: Dict[str, float],
        market_data: Dict[str, Dict],
        current_weights: Dict[str, float],
        cash: float,
        total_value: float,
        regime_state,
    ) -> Tuple[List[RebalancingDecision], Dict[str, float], float, str]:
        if not self.enabled:
            return [], current_weights, cash / max(total_value, 1), ""

        # Extrahiere benötigte Felder aus market_data
        confidences = {}
        momentums = {}
        volatilities = {}
        for ticker, data in market_data.items():
            # Confidenz aus Score (vereinfacht)
            sc = scores.get(ticker, 50.0)
            confidences[ticker] = sc / 100.0
            momentums[ticker] = data.get("return_20d", 0.0)
            volatilities[ticker] = data.get("volatility_annual_pct", 20.0)

        # Regime als String
        regime_str = "SIDEWAYS"
        if regime_state is not None:
            regime_val = getattr(regime_state, 'regime', None)
            if regime_val is not None:
                regime_str = regime_val.value.upper() if hasattr(regime_val, 'value') else str(regime_val).upper()
            else:
                regime_str = getattr(regime_state, 'label', 'SIDEWAYS').upper()

        # CPO aufrufen
        target_weights, cash_target, buy_cluster, sell_cluster, rationale = self.engine.optimize(
            scores=scores,
            confidences=confidences,
            momentums=momentums,
            volatilities=volatilities,
            current_weights=current_weights,
            regime=regime_str,
            portfolio_value=total_value,
        )

        # Erzeuge RebalancingDecision-Liste aus Differenz
        decisions = []
        for ticker, target in target_weights.items():
            current = current_weights.get(ticker, 0.0)
            delta = target - current
            if abs(delta) < 0.01:   # unter 1% ignorieren
                continue
            action = "BUY" if delta > 0 else "SELL"
            conf = confidences.get(ticker, 0.5)
            decisions.append(RebalancingDecision(
                ticker=ticker,
                action=action,
                target_weight=target,
                confidence=conf,
                reason=f"Target {target:.1%} vs current {current:.1%}",
            ))

        # Begrenzen auf max_trades (optional)
        decisions = decisions[:self.max_trades]

        log.info(f"PortfolioRebalancer (CPO): {len(decisions)} Trades, Cash-Ziel {cash_target:.1%}")
        return decisions, target_weights, cash_target, rationale
