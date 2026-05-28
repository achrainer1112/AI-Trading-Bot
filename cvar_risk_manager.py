"""
cvar_risk_manager.py – Conditional Value at Risk (CVaR) Portfolio Risk Management
==================================================================================
Berechnet CVaR des Portfolios und erzwingt Risikolimits vor der Trade-Ausführung.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from logger import log
from utils import format_currency


class CVaRRiskManager:
    """
    Portfolio-Risikomanagement basierend auf Conditional Value at Risk (Expected Shortfall).
    """

    def __init__(
        self,
        cvar_limit_pct: float = 0.05,
        confidence_level: float = 0.95,
        lookback_days: int = 252,
    ):
        self.cvar_limit_pct = cvar_limit_pct
        self.confidence_level = confidence_level
        self.lookback_days = lookback_days
        self._last_cvar_state = None

    def calculate_portfolio_returns(
        self,
        positions: Dict[str, Dict],
        historical_returns: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Berechnet historische Portfolio-Renditen aus den einzelnen Asset-Renditen.
        """
        if not positions or not historical_returns:
            return np.array([])

        # Gemeinsame Länge aller Reihen (kürzeste)
        min_len = min(len(r) for r in historical_returns.values() if len(r) > 0)
        if min_len < 2:
            return np.array([])

        # Gewichte basierend auf aktuellen Marktwerten
        total_value = sum(p.get("market_value", 0) for p in positions.values())
        if total_value <= 0:
            return np.array([])

        weights = {}
        for ticker, pos in positions.items():
            if ticker in historical_returns:
                weights[ticker] = pos.get("market_value", 0) / total_value

        # Portfolio-Renditen berechnen
        portfolio_returns = np.zeros(min_len)
        for ticker, w in weights.items():
            rets = historical_returns[ticker][-min_len:]
            portfolio_returns += w * rets
        return portfolio_returns

    def calculate_cvar(
        self,
        portfolio_returns: np.ndarray,
        confidence_level: float = None,
    ) -> Tuple[float, float]:
        """
        Berechnet VaR und CVaR (Expected Shortfall) aus historischen Renditen.
        Returns: (VaR, CVaR)
        """
        if len(portfolio_returns) < 2:
            return 0.0, 0.0

        conf = confidence_level or self.confidence_level
        var = np.percentile(portfolio_returns, (1 - conf) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return float(var), float(cvar)

    def marginal_risk_contribution(
        self,
        positions: Dict[str, Dict],
        historical_returns: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Berechnet den marginalen Risikobeitrag jedes Assets zum CVaR.
        Returns: {ticker: marginal_contribution_in_percent}
        """
        if not positions or not historical_returns:
            return {}

        total_value = sum(p.get("market_value", 0) for p in positions.values())
        if total_value <= 0:
            return {}

        portfolio_returns = self.calculate_portfolio_returns(positions, historical_returns)
        if len(portfolio_returns) < 2:
            return {}

        var, cvar = self.calculate_cvar(portfolio_returns)
        if cvar == 0:
            return {}

        contributions = {}
        for ticker, pos in positions.items():
            if ticker not in historical_returns:
                continue
            weight = pos.get("market_value", 0) / total_value
            asset_returns = historical_returns[ticker][-len(portfolio_returns):]
            # Kovarianz-basierte Approximation der marginalen Beitrags (vereinfacht)
            # Besser: Simulation mit und ohne Asset
            # Hier: Kovarianz mit Portfolio, gewichtet
            cov = np.cov(asset_returns, portfolio_returns)[0, 1]
            if cov != 0:
                marginal = weight * cov / np.var(portfolio_returns)
            else:
                marginal = weight
            contributions[ticker] = float(marginal)
        return contributions

    def evaluate_risk_state(
        self,
        portfolio_returns: np.ndarray,
    ) -> Dict:
        """
        Bewertet den aktuellen Risikozustand des Portfolios.
        """
        if len(portfolio_returns) < 2:
            return {
                "var_95": 0.0,
                "cvar_95": 0.0,
                "cvar_pct": 0.0,
                "limit_pct": self.cvar_limit_pct,
                "breach": False,
                "reduction_factor": 1.0,
            }

        var, cvar = self.calculate_cvar(portfolio_returns)
        cvar_pct = abs(cvar)  # CVaR als positive Zahl (Verlust)
        breach = cvar_pct > self.cvar_limit_pct
        reduction_factor = min(1.0, self.cvar_limit_pct / cvar_pct) if breach else 1.0

        state = {
            "var_95": var,
            "cvar_95": cvar,
            "cvar_pct": cvar_pct,
            "limit_pct": self.cvar_limit_pct,
            "breach": breach,
            "reduction_factor": reduction_factor,
        }
        self._last_cvar_state = state
        return state

    def apply_risk_mitigation(
        self,
        target_weights: Dict[str, float],
        positions: Dict[str, Dict],
        historical_returns: Dict[str, np.ndarray],
        cvar_state: Dict,
        volatility_data: Dict[str, float] = None,
        correlation_clusters: List[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Reduziert Zielgewichte proportional zum CVaR-Überschuss.
        Berücksichtigt marginale Risikobeiträge und Cluster-Konzentration.
        """
        if not cvar_state.get("breach", False):
            return target_weights

        reduction_factor = cvar_state.get("reduction_factor", 1.0)
        if reduction_factor >= 0.99:
            return target_weights

        log.warning(f"CVaR-Breach: {cvar_state['cvar_pct']:.2%} > limit {cvar_state['limit_pct']:.2%}, reduction factor {reduction_factor:.2f}")

        # Berechne marginale Risikobeiträge
        contributions = self.marginal_risk_contribution(positions, historical_returns)
        if not contributions:
            return target_weights

        # Zusätzliche Faktoren: Volatilität, Cluster
        cluster_weights = defaultdict(float)
        if correlation_clusters:
            for ticker, w in target_weights.items():
                for cluster in correlation_clusters:
                    if ticker in cluster:
                        cluster_weights[tuple(cluster)] += w
                        break

        adjusted_weights = {}
        for ticker, target in target_weights.items():
            # Basis-Reduktion
            factor = reduction_factor
            # Höhere Reduktion für hohen marginalen Beitrag
            marginal = contributions.get(ticker, 0.5)
            if marginal > 0.7:
                factor *= 0.8
            # Höhere Reduktion für volatile Assets
            if volatility_data and ticker in volatility_data:
                vol = volatility_data[ticker]
                if vol > 40:
                    factor *= 0.7
                elif vol > 30:
                    factor *= 0.85
            # Cluster-Reduktion
            for cluster in correlation_clusters or []:
                if ticker in cluster:
                    cluster_weight = cluster_weights.get(tuple(cluster), 0)
                    if cluster_weight > 0.35:  # Cluster über 35%
                        factor *= 0.8
                    break
            adjusted_weights[ticker] = target * factor

        # Normalisierung auf ursprüngliche Summe (Cash bleibt gleich)
        total_adj = sum(adjusted_weights.values())
        investable_before = sum(target_weights.values())
        if investable_before > 0:
            scale = investable_before / total_adj if total_adj > 0 else 1.0
            adjusted_weights = {t: w * scale for t, w in adjusted_weights.items()}

        return adjusted_weights

    def filter_trades_by_cvar(
        self,
        trades: List[Dict],
        cvar_state: Dict,
        positions: Dict[str, Dict],
        historical_returns: Dict[str, np.ndarray],
        volatility_data: Dict[str, float] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filtert Trades basierend auf CVaR-Breach.
        Returns: (allowed_trades, blocked_buys)
        """
        if not cvar_state.get("breach", False):
            return trades, []

        blocked_buys = []
        allowed_trades = []

        # Berechne marginalen Risikobeitrag für Buy-Vorschläge (vereinfacht)
        for trade in trades:
            if trade.get("action") == "BUY":
                ticker = trade.get("ticker")
                # Prüfe, ob der Kauf das Risiko erhöhen würde
                if volatility_data and ticker in volatility_data:
                    vol = volatility_data[ticker]
                    if vol > 30:  # hohe Vola = risikoreich
                        blocked_buys.append(trade)
                        log.info(f"CVaR: BUY {ticker} blockiert (hohe Vola {vol:.0f}%)")
                        continue
                allowed_trades.append(trade)
            else:
                # SELLs sind immer erlaubt (reduzieren Risiko)
                allowed_trades.append(trade)

        return allowed_trades, blocked_buys
