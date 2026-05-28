"""
rebalancer_v3.py – Rank-based Portfolio Rebalancer v3 (Clean Institutional Version)
====================================================================================
Ziel: Optimale Portfolio-Allokation unter Berücksichtigung von Edge, Risiko und Regime.
Keine Einzel-Signal-Jagd. Nur Target-Weights und Swaps.

Kernprinzipien:
- Portfolio ist ein geschlossenes System (Summe 100%).
- Ranking nur als Input, nicht als Output (keine Top-N Auswahl).
- Edge Differential entscheidet über Swaps.
- Confidence beeinflusst nicht direkt die Position Size, sondern nur, ob neue Position eröffnet wird.
- Trade Generation nur als Differenz: trade = target_weight - current_weight.
- Minimalismus: max. 3–5 meaningful reallocations pro Run, keine kosmetischen Anpassungen.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from logger import log
from config import (
    ACTIVE_RISK_PROFILE,
    RISK_SETTINGS,
    SECTOR_CLASSIFICATION,
    TRADE_FRICTION_PCT,
)


@dataclass
class AssetEdge:
    ticker: str
    raw_score: float               # 0-100
    confidence: float              # 0-1 (separat)
    momentum: float                # 20d Return (%)
    volatility: float              # annualisiert (%)
    regime_fit: float              # -1 bis +1
    diversification_benefit: float # 0-1
    current_weight: float
    is_core: bool                  # True für SPY, QQQ, VT etc.
    
    @property
    def edge(self) -> float:
        """
        Edge berechnet aus Score (normalisiert), Momentum, Regime-Fit.
        Bereich ca. -1 .. +1.
        """
        score_norm = self.raw_score / 100.0
        momentum_norm = max(-1, min(1, self.momentum / 20.0))
        edge = score_norm * 0.5 + momentum_norm * 0.3 + self.regime_fit * 0.2
        return max(-1.0, min(1.0, edge))
    
    @property
    def can_open_position(self) -> bool:
        """Confidence < 55% -> keine neue Position eröffnen (nur HOLD/REDUCE)."""
        return self.confidence >= 0.55


class RebalancerV3:
    """
    Portfolio Rebalancer nach institutionellen Regeln.
    """

    def __init__(self, risk_profile=None):
        self.risk_profile = risk_profile or ACTIVE_RISK_PROFILE
        self.settings = RISK_SETTINGS[self.risk_profile].copy()
        self.max_position_pct = self.settings.get("max_position_pct", 0.20)
        
        # Regime-abhängige Cash-Minima
        self.cash_target_by_regime = {
            "BULL": 0.08,      # 8% (5-12% Band)
            "BEAR": 0.15,      # 15% (10-25%)
            "SIDEWAYS": 0.10,  # 10% (8-15%)
        }
        
        # Schwellwerte für Edge Differential
        self.EDGE_IGNORE = 0.10
        self.EDGE_WATCH = 0.25
        self.EDGE_ACTIVE_SWAP = 0.40
        
        # Regime-Multiplier für Position Sizing
        self.regime_multipliers = {
            "BULL": {"momentum": 1.25, "defensive": 0.85},
            "BEAR": {"momentum": 0.60, "defensive": 1.40},
            "SIDEWAYS": {"momentum": 1.00, "defensive": 1.00},
        }
        
        # Core Holdings (nicht leicht tauschen)
        self.core_tickers = {"SPY", "VT", "QQQ", "IVV", "VOO", "VTI"}
        
        # Maximal sinnvolle Allokationsänderungen pro Run
        self.max_reallocations = 4   # max. 4 Swaps/Shifts

    def compute_target_weights(
        self,
        edges: Dict[str, AssetEdge],
        regime: str,
    ) -> Dict[str, float]:
        """
        Berechnet Zielgewichte proportional zur Edge, unter Berücksichtigung von:
        - Regime-Multiplier (Momentum/Defensive)
        - Diversifikation
        - Confidence (nur ob neue Position erlaubt ist)
        """
        # 1. Roh-Ranking nach Edge
        sorted_by_edge = sorted(edges.items(), key=lambda x: x[1].edge, reverse=True)
        
        # 2. Anwendung Regime-Multiplier auf die Edge (für Gewichtung)
        weighted_edges = {}
        for ticker, e in edges.items():
            if e.is_core:
                multiplier = 1.0
            else:
                # Bestimme Momentum vs Defensive (einfache Heuristik)
                if e.momentum > 5 and regime == "BULL":
                    multiplier = self.regime_multipliers[regime]["momentum"]
                elif e.momentum < -5 and regime == "BEAR":
                    multiplier = self.regime_multipliers[regime]["defensive"]
                else:
                    multiplier = 1.0
            weighted_edge = e.edge * multiplier
            weighted_edges[ticker] = max(0.05, weighted_edge)   # minimale Edge für Gewicht
        
        # 3. Softmax-artige Gewichtung
        total_edge = sum(weighted_edges.values())
        if total_edge <= 0:
            return {t: 0.0 for t in edges}
        
        raw_weights = {t: weighted_edges[t] / total_edge for t in edges}
        
        # 4. Anwenden von Confidence: wenn confidence < 0.55, kann das Asset keine neue Position eröffnen
        # D.h. wenn current_weight == 0, dann bleibt es 0.
        for ticker, e in edges.items():
            if e.current_weight == 0 and not e.can_open_position:
                raw_weights[ticker] = 0.0
        
        # 5. Neu normalisieren
        total = sum(raw_weights.values())
        if total > 0:
            raw_weights = {t: w / total for t, w in raw_weights.items()}
        
        # 6. Caps pro Position (max_position_pct)
        for t in raw_weights:
            raw_weights[t] = min(raw_weights[t], self.max_position_pct)
        
        return raw_weights

    def apply_regime_cash(self, regime: str, target_weights: Dict[str, float]) -> Dict[str, float]:
        """Stellt sicher, dass Cash-Minimum eingehalten wird."""
        cash_target = self.cash_target_by_regime.get(regime, 0.10)
        total_invested = sum(target_weights.values())
        if total_invested > (1 - cash_target):
            scale = (1 - cash_target) / total_invested
            target_weights = {t: w * scale for t, w in target_weights.items()}
        return target_weights

    def identify_swaps(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        edges: Dict[str, AssetEdge],
        regime: str,
    ) -> List[Tuple[str, str, float]]:
        """
        Identifiziert sinnvolle Swaps basierend auf Edge Differential.
        Rückgabe: Liste von (sell_ticker, buy_ticker, edge_gain)
        """
        # Nur Assets mit signifikanter Differenz
        overweights = [(t, w) for t, w in current_weights.items() if w > target_weights.get(t, 0) + 0.02]
        underweights = [(t, w) for t, w in target_weights.items() if w > current_weights.get(t, 0) + 0.02]
        
        # Sortiere Overweights nach niedrigster Edge, Underweights nach höchster Edge
        overweights.sort(key=lambda x: edges[x[0]].edge)
        underweights.sort(key=lambda x: edges[x[0]].edge, reverse=True)
        
        swaps = []
        for sell_t, sell_w in overweights:
            # Core Holdings nicht leichtfertig tauschen
            if edges[sell_t].is_core:
                continue
            for buy_t, buy_w in underweights:
                if buy_t == sell_t:
                    continue
                # Edge Differential
                edge_gain = edges[buy_t].edge - edges[sell_t].edge
                if edge_gain < self.EDGE_ACTIVE_SWAP:
                    continue
                # Verschiebung sollte > 3% Portfolio sein
                shift = min(buy_w, sell_w)
                if shift < 0.03:
                    continue
                swaps.append((sell_t, buy_t, edge_gain))
                # Nach einem Swap diesen buy_t aus der Liste entfernen (1:1 Swap)
                underweights = [u for u in underweights if u[0] != buy_t]
                if len(swaps) >= self.max_reallocations:
                    break
            if len(swaps) >= self.max_reallocations:
                break
        
        return swaps

    def generate_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        swaps: List[Tuple[str, str, float]],
        edges: Dict[str, AssetEdge],   # edges wird benötigt für confidence
    ) -> List[Dict]:
        """
        Generiert Trades als Differenz zwischen Ziel und aktuell.
        Beachtet Swaps (override target_weights).
        """
        # Zuerst Swaps anwenden: setze sell_target auf 0, buy_target auf gewünschten Wert
        adjusted_target = target_weights.copy()
        for sell_t, buy_t, _ in swaps:
            adjusted_target[sell_t] = 0.0
            # Erhöhe Ziel für buy_t (max. max_position_pct)
            new_buy_target = min(adjusted_target.get(buy_t, 0) + current_weights.get(sell_t, 0), self.max_position_pct)
            adjusted_target[buy_t] = new_buy_target
        
        # Berechne Deltas
        trades = []
        for ticker, target in adjusted_target.items():
            current = current_weights.get(ticker, 0.0)
            delta = target - current
            if abs(delta) < 0.02:   # 2% Mindeständerung
                continue
            action = "BUY" if delta > 0 else "SELL"
            # Confidence aus edges, falls vorhanden
            confidence = edges[ticker].confidence if ticker in edges else 0.7
            trades.append({
                "ticker": ticker,
                "action": action,
                "target_allocation": target,
                "confidence": confidence,
                "reason": f"Target weight {target:.1%} vs current {current:.1%}",
            })
        return trades

    def rebalance(
        self,
        scores: Dict[str, float],
        confidences: Dict[str, float],
        momentums: Dict[str, float],
        volatilities: Dict[str, float],
        current_weights: Dict[str, float],
        cash: float,
        total_value: float,
        regime: str,
    ) -> Tuple[List[Dict], Dict[str, float], float, str]:
        """
        Hauptmethode: Führt das Rebalancing durch.
        Returns: (trades, target_weights, cash_target, rationale)
        """
        # 1. AssetEdge-Objekte erstellen
        edges = {}
        for ticker in set(scores.keys()) | set(current_weights.keys()):
            score = scores.get(ticker, 50.0)
            conf = confidences.get(ticker, 0.5)
            mom = momentums.get(ticker, 0.0)
            vol = volatilities.get(ticker, 20.0)
            current = current_weights.get(ticker, 0.0)
            is_core = ticker in self.core_tickers
            
            # Regime-Fit (vereinfacht)
            if regime == "BULL":
                if ticker in self.core_tickers:
                    regime_fit = 0.2
                elif mom > 5:
                    regime_fit = 0.5
                else:
                    regime_fit = -0.2
            elif regime == "BEAR":
                if ticker in self.core_tickers:
                    regime_fit = 0.1
                elif mom < -5:
                    regime_fit = -0.4
                else:
                    regime_fit = 0.3
            else:  # SIDEWAYS
                regime_fit = 0.0
            
            # Diversifikationsvorteil (vereinfacht)
            divers = 0.5 if ticker in self.core_tickers else 0.3
            
            edges[ticker] = AssetEdge(
                ticker=ticker,
                raw_score=score,
                confidence=conf,
                momentum=mom,
                volatility=vol,
                regime_fit=regime_fit,
                diversification_benefit=divers,
                current_weight=current,
                is_core=is_core,
            )
        
        # 2. Zielgewichte berechnen
        target_weights = self.compute_target_weights(edges, regime)
        target_weights = self.apply_regime_cash(regime, target_weights)
        
        # 3. Swaps identifizieren (Edge-basiert)
        swaps = self.identify_swaps(current_weights, target_weights, edges, regime)
        
        # 4. Trades generieren (edges wird übergeben)
        trades = self.generate_trades(current_weights, target_weights, swaps, edges)
        
        # 5. Cash-Ziel
        total_target = sum(target_weights.values())
        cash_target = max(self.cash_target_by_regime.get(regime, 0.10), 1.0 - total_target)
        
        # 6. Rationale
        if swaps:
            rationale = f"{len(swaps)} swaps: " + ", ".join([f"{s}→{b}" for s,b,_ in swaps[:3]])
        else:
            rationale = "Keine vorteilhaften Swaps identifiziert."
        
        log.info(f"RebalancerV3: {len(trades)} Trades, Cash-Ziel {cash_target:.1%}, Swaps: {len(swaps)}")
        return trades, target_weights, cash_target, rationale
