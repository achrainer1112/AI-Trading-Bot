"""
portfolio_rebalancer.py – Integration der Rebalancing Engine V3 in den Trading Bot
==================================================================================
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass

from logger import log
from config import (
    ACTIVE_RISK_PROFILE,
    REBALANCING_ENGINE_ENABLED,
    REBALANCING_MAX_TRADES,
)
from rebalancer_v3 import RebalancerV3


@dataclass
class RebalancingDecision:
    ticker: str
    action: str          # BUY, SELL, HOLD
    target_weight: float
    confidence: float
    reason: str


class PortfolioRebalancer:
    def __init__(self, risk_profile=None):
        self.risk_profile = risk_profile or ACTIVE_RISK_PROFILE
        self.engine = RebalancerV3(risk_profile)
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

        # Extrahiere Momentum, Volatilität, Konfidenz
        momentums = {}
        volatilities = {}
        confidences = {}
        for ticker, data in market_data.items():
            momentums[ticker] = data.get("return_20d", 0.0)
            volatilities[ticker] = data.get("volatility_annual_pct", 20.0)
            # Konfidenz aus Score (vereinfacht, kann auch aus AI kommen)
            sc = scores.get(ticker, 50.0)
            confidences[ticker] = sc / 100.0

        # Regime als String
        regime_str = "SIDEWAYS"
        if regime_state is not None:
            regime_val = getattr(regime_state, 'regime', None)
            if regime_val is not None:
                regime_str = regime_val.value.upper() if hasattr(regime_val, 'value') else str(regime_val).upper()
            else:
                regime_str = getattr(regime_state, 'label', 'SIDEWAYS').upper()

        trades_raw, target_weights, cash_target, rationale = self.engine.rebalance(
            scores=scores,
            confidences=confidences,
            momentums=momentums,
            volatilities=volatilities,
            current_weights=current_weights,
            cash=cash,
            total_value=total_value,
            regime=regime_str,
        )

        # Begrenzen auf max_trades
        trades_raw = trades_raw[:self.max_trades]

        decisions = []
        for t in trades_raw:
            decisions.append(RebalancingDecision(
                ticker=t["ticker"],
                action=t["action"],
                target_weight=t["target_allocation"],
                confidence=t.get("confidence", 0.7),
                reason=t.get("reason", ""),
            ))

        log.info(f"PortfolioRebalancer V3: {len(decisions)} Trades (max {self.max_trades})")
        return decisions, target_weights, cash_target, rationale
