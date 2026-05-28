"""
rebalancer_v2.py – Rank-Based Portfolio Rebalancer v2
======================================================
Portfolio-first Optimierung mit Edge-basierter Swap-Logik, Confidence-Faktor,
Core/Tactical Unterscheidung und Turnover Control.

Keine starren Top-N-Regeln. Entscheidungen basieren auf strukturellen Verbesserungen
des Gesamtportfolios unter Berücksichtigung von Transaktionskosten.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from logger import log
from config import (
    ACTIVE_RISK_PROFILE,
    RISK_SETTINGS,
    SECTOR_CLASSIFICATION,
    CORRELATION_GROUPS,
    TRADE_FRICTION_PCT,
)


@dataclass
class AssetEdge:
    """Bewertung eines Assets im Portfolio-Kontext (Edge-basiert)."""
    ticker: str
    base_score: float                # 0-100 Roh-Score
    confidence: float                # 0-1
    momentum: float                  # 20d Return (%)
    volatility: float                # annualisiert (%)
    regime_fit: float                # -1 bis +1 (wie gut zum aktuellen Regime)
    diversification_benefit: float   # 0-1 (Korrelationsvorteil)
    drawdown_risk: float             # 0-1 (je höher desto riskanter)
    current_weight: float            # aktuelle Allokation
    optimal_weight: float            # wird berechnet
    role: str                        # "core" oder "tactical"
    
    @property
    def raw_edge(self) -> float:
        """
        Roh-Edge ohne Konfidenz.
        Kombiniert Score, Momentum, Volatilität, Regime-Fit, Diversifikation.
        """
        # Score normalisiert 0-1
        score_norm = self.base_score / 100.0
        # Momentum Edge: positive Returns geben Bonus, negative malus
        momentum_edge = max(-0.5, min(0.5, self.momentum / 20.0))
        # Volatilität: niedrige Vola ist positiv (Stabilität), hohe negativ
        vol_edge = max(-0.3, min(0.3, (15.0 - self.volatility) / 50.0))
        # Regime-Fit (direkt)
        # Diversifikation: je höher desto besser für das Portfolio
        # Drawdown-Risiko: höherer Wert senkt Edge
        risk_penalty = self.drawdown_risk * 0.2
        # Gewichtung
        edge = (score_norm * 0.4 +
                momentum_edge * 0.25 +
                vol_edge * 0.1 +
                self.regime_fit * 0.15 +
                self.diversification_benefit * 0.1 -
                risk_penalty)
        # Clamp auf -1..1
        return max(-1.0, min(1.0, edge))
    
    @property
    def confidence_factor(self) -> float:
        """Konfidenz als Verstärker, nie als Killer."""
        if self.confidence >= 0.75:
            return 1.0
        elif self.confidence >= 0.60:
            return 0.85
        elif self.confidence >= 0.50:
            return 0.70
        else:
            return 0.50   # nicht blockieren, nur reduzieren
    
    @property
    def adjusted_edge(self) -> float:
        """Edge × Confidence-Faktor."""
        return self.raw_edge * self.confidence_factor


@dataclass
class SwapCandidate:
    """Möglicher Austausch (SELL eines Bestands, BUY eines neuen Assets)."""
    sell_ticker: str
    buy_ticker: str
    sell_edge: float
    buy_edge: float
    edge_gain: float          # buy_edge - sell_edge
    transaction_cost: float   # in % des Portfoliowerts
    risk_increase: float      # zusätzliches Risiko (0-1)
    net_benefit: float        # edge_gain - cost - risk_penalty
    sell_role: str
    buy_role: str


class RebalancerV2:
    """
    Portfolio Rebalancer mit Edge-basierter Logik, Core/Tactical Unterscheidung,
    Turnover Control und Confidenz-Faktor.
    """

    def __init__(self, risk_profile=None):
        self.risk_profile = risk_profile or ACTIVE_RISK_PROFILE
        self.settings = RISK_SETTINGS[self.risk_profile].copy()
        self.max_position_pct = self.settings.get("max_position_pct", 0.20)
        self.min_cash_pct = self.settings.get("min_cash_pct", 0.10)
        self.max_turnover_per_run = 0.20          # 20% maximaler Umschlag pro Run
        self.min_edge_threshold = 0.12            # Mindest-Edge-Gewinn für Swap
        self.transaction_cost_bps = TRADE_FRICTION_PCT * 100  # in Basispunkten
        
        # Core Holdings (nicht leicht zu verkaufen)
        self.core_tickers = {"SPY", "VT", "QQQ", "IVV", "VOO", "VTI"}
        
    def calculate_asset_edge(
        self,
        ticker: str,
        score: float,
        confidence: float,
        momentum: float,
        volatility: float,
        regime: str,                # "BULL", "BEAR", "SIDEWAYS"
        current_weight: float,
        portfolio_weights: Dict[str, float],
        correlation_matrix: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> AssetEdge:
        """
        Berechnet den vollständigen Edge für ein Asset.
        """
        # Regime-Fit
        if regime == "BULL":
            regime_fit = 0.5 if ticker in self.core_tickers else 0.3
            # Tech und Momentum profitieren im Bullenmarkt
            if ticker in {"QQQ", "XLK", "NVDA", "AMD", "AAPL", "MSFT"}:
                regime_fit += 0.3
        elif regime == "BEAR":
            regime_fit = -0.3 if ticker in self.core_tickers else -0.5
            # Defensive Sektoren
            if ticker in {"XLV", "XLU", "GLD", "BND"}:
                regime_fit = 0.2
        else:  # SIDEWAYS
            regime_fit = 0.0
        
        regime_fit = max(-1.0, min(1.0, regime_fit))
        
        # Diversifikationsvorteil (vereinfacht: je weniger Korrelation mit SPY, desto besser)
        # Hier als Platzhalter: 0.5 für die meisten, höher für defensive Assets
        if ticker in {"XLV", "XLU", "GLD"}:
            diversification_benefit = 0.7
        elif ticker in self.core_tickers:
            diversification_benefit = 0.3
        else:
            diversification_benefit = 0.5
        
        # Drawdown-Risiko (vereinfacht: hohe Volatilität erhöht Risiko)
        drawdown_risk = min(0.8, max(0.1, volatility / 100.0))
        
        # Rolle: Core oder Tactical
        role = "core" if ticker in self.core_tickers else "tactical"
        
        # Optimale Gewichtung später berechnen
        optimal_weight = current_weight
        
        return AssetEdge(
            ticker=ticker,
            base_score=score,
            confidence=confidence,
            momentum=momentum,
            volatility=volatility,
            regime_fit=regime_fit,
            diversification_benefit=diversification_benefit,
            drawdown_risk=drawdown_risk,
            current_weight=current_weight,
            optimal_weight=optimal_weight,
            role=role,
        )
    
    def compute_optimal_weights(
        self,
        edges: Dict[str, AssetEdge],
        total_investable: float,
    ) -> Dict[str, float]:
        """
        Berechnet Zielgewichte basierend auf adjusted_edge (Softmax-ähnlich).
        Core Holdings erhalten Grundgewichtung.
        """
        if not edges:
            return {}
        
        # Trenne Core und Tactical
        core_edges = {t: e for t, e in edges.items() if e.role == "core"}
        tactical_edges = {t: e for t, e in edges.items() if e.role == "tactical"}
        
        # Core: Mindestens 30% des investierbaren Kapitals (wenn vorhanden)
        core_weight_total = min(0.5, len(core_edges) * 0.15) if core_edges else 0.0
        
        # Verteilung auf Core nach Edge
        core_weights = {}
        if core_edges:
            total_core_edge = sum(e.adjusted_edge + 0.5 for e in core_edges.values())
            if total_core_edge > 0:
                for t, e in core_edges.items():
                    raw = (e.adjusted_edge + 0.5) / total_core_edge * core_weight_total
                    core_weights[t] = min(self.max_position_pct, raw)
        
        # Rest für Tactical
        tactical_budget = total_investable - sum(core_weights.values())
        if tactical_budget > 0 and tactical_edges:
            total_tactical_edge = sum(e.adjusted_edge + 0.2 for e in tactical_edges.values())
            if total_tactical_edge > 0:
                for t, e in tactical_edges.items():
                    raw = (e.adjusted_edge + 0.2) / total_tactical_edge * tactical_budget
                    tactical_weights[t] = min(self.max_position_pct, raw)
            else:
                tactical_weights = {t: 0.0 for t in tactical_edges}
        else:
            tactical_weights = {}
        
        return {**core_weights, **tactical_weights}
    
    def identify_swap_candidates(
        self,
        current_edges: Dict[str, AssetEdge],
        optimal_weights: Dict[str, float],
        total_value: float,
        regime: str,
    ) -> List[SwapCandidate]:
        """
        Identifiziert vielversprechende Swaps: Verkauf eines unterdurchschnittlichen
        Assets, Kauf eines überdurchschnittlichen Assets.
        """
        # Berechne Portfolio-Durchschnitts-Edge (gewichtet)
        total_weight = sum(e.current_weight for e in current_edges.values())
        if total_weight == 0:
            return []
        avg_edge = sum(e.current_weight * e.adjusted_edge for e in current_edges.values()) / total_weight
        
        # Assets mit niedrigerem Edge als Durchschnitt (potenzielle SELLs)
        weak_assets = [
            (t, e) for t, e in current_edges.items()
            if e.current_weight > 0.01 and e.adjusted_edge < avg_edge - 0.1
        ]
        # Assets mit höherem Edge (potenzielle BUYs)
        strong_assets = [
            (t, e) for t, e in current_edges.items()
            if e.adjusted_edge > avg_edge + 0.15 and (optimal_weights.get(t, 0) > e.current_weight or e.current_weight == 0)
        ]
        
        # Zusätzlich: Assets die in optimal_weights vorkommen, aber nicht gehalten werden
        for t, target in optimal_weights.items():
            if t not in current_edges and target > 0.01:
                # Simuliere Edge für dieses Asset (muss vorher berechnet sein)
                # Hier wird erwartet, dass edges bereits alle Kandidaten enthält
                pass
        
        swaps = []
        for sell_t, sell_e in weak_assets:
            # Core Holdings nur bei extremem Edge-Verlust
            if sell_e.role == "core" and sell_e.adjusted_edge > avg_edge - 0.2:
                continue
            
            for buy_t, buy_e in strong_assets:
                if buy_t == sell_t:
                    continue
                # Kein Swap zwischen Core und Core (sinnlos)
                if sell_e.role == "core" and buy_e.role == "core":
                    continue
                
                edge_gain = buy_e.adjusted_edge - sell_e.adjusted_edge
                if edge_gain < self.min_edge_threshold:
                    continue
                
                # Transaktionskosten (2x Spread, da Verkauf + Kauf)
                transaction_cost = 2 * TRADE_FRICTION_PCT
                # Risikozuschlag: höhere Volatilität des neuen Assets
                risk_increase = max(0, (buy_e.volatility - sell_e.volatility) / 100.0)
                risk_penalty = risk_increase * 0.1
                
                net_benefit = edge_gain - transaction_cost - risk_penalty
                
                if net_benefit > 0:
                    swaps.append(SwapCandidate(
                        sell_ticker=sell_t,
                        buy_ticker=buy_t,
                        sell_edge=sell_e.adjusted_edge,
                        buy_edge=buy_e.adjusted_edge,
                        edge_gain=edge_gain,
                        transaction_cost=transaction_cost,
                        risk_increase=risk_increase,
                        net_benefit=net_benefit,
                        sell_role=sell_e.role,
                        buy_role=buy_e.role,
                    ))
        
        # Sortiere nach Netto-Nutzen
        swaps.sort(key=lambda x: x.net_benefit, reverse=True)
        return swaps
    
    def apply_turnover_limit(
        self,
        swaps: List[SwapCandidate],
        current_weights: Dict[str, float],
        total_value: float,
    ) -> List[SwapCandidate]:
        """
        Begrenzt den maximalen Umschlag pro Run.
        """
        if not swaps:
            return swaps
        
        max_turnover_value = total_value * self.max_turnover_per_run
        turnover_used = 0.0
        approved = []
        
        for swap in swaps:
            sell_value = current_weights.get(swap.sell_ticker, 0.0) * total_value
            if turnover_used + sell_value > max_turnover_value:
                log.info(f"Turnover limit reached, skipping swap {swap.sell_ticker} -> {swap.buy_ticker}")
                continue
            approved.append(swap)
            turnover_used += sell_value
        
        return approved
    
    def generate_decisions(
        self,
        swaps: List[SwapCandidate],
        optimal_weights: Dict[str, float],
        current_weights: Dict[str, float],
        edges: Dict[str, AssetEdge],
    ) -> List[Dict]:
        """
        Erzeugt finale Handelsentscheidungen (BUY, SELL, HOLD) basierend auf Swaps.
        """
        decisions = []
        
        # Zuerst alle SELLs aus Swaps
        sell_tickers = {s.sell_ticker for s in swaps}
        for ticker in sell_tickers:
            target_weight = 0.0  # bei Swap wird komplett verkauft
            decision = {
                "ticker": ticker,
                "action": "SELL",
                "target_allocation": target_weight,
                "confidence": 1.0,
                "reason": f"Swap out: low edge ({edges[ticker].adjusted_edge:.2f})",
                "risk_approved": True,
                "rebalancing_engine": True,
            }
            decisions.append(decision)
        
        # Dann BUYs aus Swaps (kombiniere doppelte Käufe)
        buy_map = {}
        for swap in swaps:
            ticker = swap.buy_ticker
            target_weight = optimal_weights.get(ticker, 0.0)
            if target_weight == 0:
                target_weight = min(self.max_position_pct, current_weights.get(ticker, 0.0) + 0.05)
            if ticker in buy_map:
                buy_map[ticker] = max(buy_map[ticker], target_weight)
            else:
                buy_map[ticker] = target_weight
        
        for ticker, target in buy_map.items():
            decision = {
                "ticker": ticker,
                "action": "BUY",
                "target_allocation": target,
                "confidence": edges[ticker].confidence if ticker in edges else 0.7,
                "reason": f"Swap in: high edge ({edges[ticker].adjusted_edge:.2f})",
                "risk_approved": True,
                "rebalancing_engine": True,
            }
            decisions.append(decision)
        
        return decisions
    
    def rebalance(
        self,
        scores: Dict[str, float],          # ticker -> base score 0-100
        confidences: Dict[str, float],     # ticker -> confidence 0-1
        momentums: Dict[str, float],       # ticker -> momentum (%)
        volatilities: Dict[str, float],    # ticker -> volatility (%)
        current_weights: Dict[str, float],
        cash: float,
        total_value: float,
        regime: str,                       # "BULL", "BEAR", "SIDEWAYS"
    ) -> Tuple[List[Dict], Dict[str, float], float, str]:
        """
        Hauptmethode: Führt das gesamte Rebalancing durch.
        Returns: (decisions, target_weights, cash_target, rationale)
        """
        # Schritt 1: Asset-Edges berechnen (für alle Assets in scores)
        edges = {}
        all_tickers = set(scores.keys()) | set(current_weights.keys())
        portfolio_weights = current_weights.copy()
        
        for ticker in all_tickers:
            score = scores.get(ticker, 50.0)
            confidence = confidences.get(ticker, 0.5)
            momentum = momentums.get(ticker, 0.0)
            volatility = volatilities.get(ticker, 20.0)
            current_weight = current_weights.get(ticker, 0.0)
            
            edge = self.calculate_asset_edge(
                ticker=ticker,
                score=score,
                confidence=confidence,
                momentum=momentum,
                volatility=volatility,
                regime=regime,
                current_weight=current_weight,
                portfolio_weights=portfolio_weights,
            )
            edges[ticker] = edge
        
        # Schritt 2: Optimale Gewichte berechnen
        investable_budget = 1.0 - self.min_cash_pct
        optimal_weights = self.compute_optimal_weights(edges, investable_budget)
        
        # Schritt 3: Swap-Kandidaten identifizieren
        swaps = self.identify_swap_candidates(edges, optimal_weights, total_value, regime)
        
        # Schritt 4: Turnover-Limit anwenden
        swaps = self.apply_turnover_limit(swaps, current_weights, total_value)
        
        # Schritt 5: Entscheidungen generieren
        decisions = self.generate_decisions(swaps, optimal_weights, current_weights, edges)
        
        # Schritt 6: Zielallokationen und Cash-Ziel
        target_weights = current_weights.copy()
        for d in decisions:
            if d["action"] == "BUY":
                target_weights[d["ticker"]] = d["target_allocation"]
            elif d["action"] == "SELL":
                target_weights[d["ticker"]] = 0.0
        
        total_target = sum(target_weights.values())
        cash_target = max(self.min_cash_pct, 1.0 - total_target)
        
        # Rationale
        if swaps:
            rationale = f"Swapped {len(swaps)} positions: " + ", ".join([f"{s.sell_ticker}→{s.buy_ticker}" for s in swaps[:3]])
        else:
            rationale = "No beneficial swaps identified within edge and turnover constraints."
        
        log.info(f"RebalancerV2: {len(decisions)} decisions, cash target {cash_target:.1%}")
        return decisions, target_weights, cash_target, rationale
