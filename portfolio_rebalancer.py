"""
portfolio_rebalancer.py – Integration der Portfolio-Optimierung
================================================================
Vermittelt zwischen ContinuousPortfolioOptimizer (CPO) und dem Trading Bot.
Wandelt die optimierten Zielallokationen in handelbare Entscheidungen um.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from logger import log
from config import (
    ACTIVE_RISK_PROFILE,
    REBALANCING_ENGINE_ENABLED,
    REBALANCING_MAX_TRADES,
    SECTOR_CLASSIFICATION,
)
from cpo_engine import ContinuousPortfolioOptimizer


@dataclass
class RebalancingDecision:
    ticker: str
    action: str          # BUY, SELL, HOLD
    target_weight: float
    confidence: float
    reason: str


class PortfolioRebalancer:
    """
    Integriert den Continuous Portfolio Optimizer (CPO) in den Bot-Workflow.
    Kann optional CVaR-basierte Nachjustierung vornehmen, falls risk_manager übergeben wird.
    """

    def __init__(self, risk_profile=None, cvar_manager=None):
        self.risk_profile = risk_profile or ACTIVE_RISK_PROFILE
        self.engine = ContinuousPortfolioOptimizer(risk_profile)
        self.enabled = REBALANCING_ENGINE_ENABLED
        self.max_trades = REBALANCING_MAX_TRADES
        self.cvar_manager = cvar_manager  # optional für CVaR-basiertes Sizing

    def optimize_portfolio(
        self,
        scores: Dict[str, float],
        market_data: Dict[str, Dict],
        current_weights: Dict[str, float],
        cash: float,
        total_value: float,
        regime_state,
        historical_returns: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[List[RebalancingDecision], Dict[str, float], float, str]:
        """
        Hauptmethode: Ruft CPO auf und erzeugt handelbare Entscheidungen.
        Optional: CVaR-basierte Anpassung der Zielgewichte (falls historical_returns gegeben).
        """
        if not self.enabled:
            return [], current_weights, cash / max(total_value, 1), ""

        # Extrahiere Momentum, Volatilität, Konfidenz aus market_data
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

        # Optional: CVaR-basierte Nachjustierung der Zielgewichte
        if self.cvar_manager is not None and historical_returns is not None:
            try:
                # Baue temporäre Positionsstruktur für CVaR-Berechnung
                temp_positions = {t: {"market_value": w * total_value} for t, w in current_weights.items() if w > 0}
                for ticker, target in target_weights.items():
                    if target > 0:
                        temp_positions[ticker] = {"market_value": target * total_value}
                # Berechne marginale Beiträge und adjustiere Zielgewichte
                marginal = self.cvar_manager.marginal_cvar_contribution(temp_positions, historical_returns)
                if marginal:
                    # Gewichte proportional zum inversen marginalen Beitrag
                    inv = {t: 1.0 / max(marginal.get(t, 0.05), 0.01) for t in target_weights}
                    total_inv = sum(inv.values())
                    if total_inv > 0:
                        investable = 1.0 - cash_target
                        for t in target_weights:
                            new_weight = (inv[t] / total_inv) * investable
                            target_weights[t] = min(new_weight, self.engine.max_position_pct)
                    log.info("CVaR-basiertes Position Sizing angewendet")
            except Exception as e:
                log.warning(f"CVaR-Nachjustierung fehlgeschlagen: {e}")

        # Trade-Intents aus Differenz generieren
        decisions = []
        for ticker, target in target_weights.items():
            current = current_weights.get(ticker, 0.0)
            delta = target - current
            if abs(delta) < 0.01:  # unter 1% ignorieren
                continue
            action = "BUY" if delta > 0 else "SELL"
            conf = confidences.get(ticker, 0.5)
            decisions.append(RebalancingDecision(
                ticker=ticker,
                action=action,
                target_weight=target,
                confidence=conf,
                reason=f"CPO: target {target:.1%} vs current {current:.1%} ({regime_str})",
            ))

        # Begrenze auf max_trades (bevorzuge SELLs, dann BUYs mit höchster Confidence)
        sells = [d for d in decisions if d.action == "SELL"]
        buys = [d for d in decisions if d.action == "BUY"]
        buys.sort(key=lambda x: x.confidence, reverse=True)
        limited = sells + buys[:max(0, self.max_trades - len(sells))]
        if len(limited) < len(decisions):
            log.info(f"Limited trades from {len(decisions)} to {len(limited)} (max_trades={self.max_trades})")

        log.info(f"PortfolioRebalancer: {len(limited)} Trades, Cash-Ziel {cash_target:.1%}")
        return limited, target_weights, cash_target, rationale
