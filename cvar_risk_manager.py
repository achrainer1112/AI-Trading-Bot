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

# ========== NEUE METHODEN IN CVaRRiskManager ==========

    def marginal_cvar_contribution(self, positions: Dict[str, Dict], historical_returns: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Berechnet den marginalen Beitrag jedes Assets zum Portfolio-CVaR.
        Returns: {ticker: contribution_in_percent_of_total_cvar}
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
            # Kovarianz-basierte Approximation des marginalen Beitrags
            cov = np.cov(asset_returns, portfolio_returns)[0, 1]
            marginal_beta = cov / np.var(portfolio_returns) if np.var(portfolio_returns) != 0 else 0
            # Marginaler Beitrag = Gewicht * marginal_beta * (CVaR / VaR?) Vereinfacht: weight * marginal_beta
            marginal_contrib = weight * marginal_beta
            contributions[ticker] = float(marginal_contrib)
        # Normalisieren auf 1 (Summe = 1)
        total = sum(contributions.values())
        if total > 0:
            contributions = {t: v / total for t, v in contributions.items()}
        return contributions

    def simulate_trade_impact(self, current_positions: Dict[str, Dict], proposed_trade: Dict,
                              historical_returns: Dict[str, np.ndarray]) -> Dict:
        """
        Simuliert die Auswirkung eines einzelnen Trades auf den Portfolio-CVaR.
        proposed_trade: {"ticker": "AAPL", "action": "BUY", "value_usd": 1000}
        Returns: {"cvar_before": float, "cvar_after": float, "cvar_change": float, "breach": bool}
        """
        if not current_positions or not historical_returns:
            return {"cvar_before": 0, "cvar_after": 0, "cvar_change": 0, "breach": False}
        total_value_before = sum(p.get("market_value", 0) for p in current_positions.values())
        if total_value_before <= 0:
            return {"cvar_before": 0, "cvar_after": 0, "cvar_change": 0, "breach": False}
        # Portfolio-Renditen vorher
        port_returns_before = self.calculate_portfolio_returns(current_positions, historical_returns)
        _, cvar_before = self.calculate_cvar(port_returns_before)
        cvar_before = abs(cvar_before)

        # Simuliere neues Portfolio
        import copy
        new_positions = copy.deepcopy(current_positions)
        ticker = proposed_trade["ticker"]
        action = proposed_trade["action"]
        value = proposed_trade.get("value_usd", 0)
        if action == "BUY":
            if ticker in new_positions:
                new_positions[ticker]["market_value"] += value
            else:
                new_positions[ticker] = {"market_value": value}
        elif action == "SELL":
            if ticker in new_positions:
                new_positions[ticker]["market_value"] = max(0, new_positions[ticker].get("market_value", 0) - value)
                if new_positions[ticker]["market_value"] == 0:
                    del new_positions[ticker]
        # Portfolio-Renditen nachher
        port_returns_after = self.calculate_portfolio_returns(new_positions, historical_returns)
        _, cvar_after = self.calculate_cvar(port_returns_after)
        cvar_after = abs(cvar_after)

        return {
            "cvar_before": cvar_before,
            "cvar_after": cvar_after,
            "cvar_change": cvar_after - cvar_before,
            "breach": cvar_after > self.cvar_limit_pct,
        }

    def cvar_adjusted_allocation(self, base_allocation: float, ticker: str,
                                 positions: Dict[str, Dict], historical_returns: Dict[str, np.ndarray]) -> float:
        """
        Berechnet eine CVaR-bereinigte Zielallokation.
        Prinzip: Allokation ~ (Expected_Return / Marginal_CVaR_Contribution)
        Hier verwenden wir den Score als Proxy für Expected Return.
        """
        if not positions or not historical_returns or ticker not in historical_returns:
            return base_allocation
        contributions = self.marginal_cvar_contribution(positions, historical_returns)
        marginal = contributions.get(ticker, 0.05)  # Fallback
        if marginal <= 0.01:
            marginal = 0.01
        # Je kleiner der marginale Beitrag, desto höher die erlaubte Allokation
        # Skalierungsfaktor: 1 / marginal (normiert)
        scale = 1.0 / marginal
        # Begrenzung auf sinnvollen Bereich (0.5 ... 2.0)
        scale = max(0.5, min(2.0, scale))
        adjusted = base_allocation * scale
        # Max Position Cap beachten
        max_pos = self.settings.get("max_position_pct", 0.20) if hasattr(self, 'settings') else 0.20
        return min(adjusted, max_pos)
