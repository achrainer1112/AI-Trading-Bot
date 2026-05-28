"""
rebalancer_v3.py – Rank-based Portfolio Rebalancer v3
mit Dynamic Position Sizing & Confidence Weighting
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

from logger import log
from config import ACTIVE_RISK_PROFILE, RISK_SETTINGS


@dataclass
class AssetEdge:
    ticker: str
    raw_score: float
    confidence: float
    momentum: float
    volatility: float
    regime_fit: float
    diversification_benefit: float
    current_weight: float
    is_core: bool

    @property
    def edge(self) -> float:
        score_norm = self.raw_score / 100.0
        momentum_norm = max(-1, min(1, self.momentum / 20.0))
        return max(-1.0, min(1.0, score_norm * 0.5 + momentum_norm * 0.3 + self.regime_fit * 0.2))

    @property
    def confidence_weight(self) -> float:
        if self.confidence < 0.55:
            return 0.3
        elif self.confidence < 0.65:
            return 0.7
        elif self.confidence < 0.80:
            return 1.0
        else:
            return 1.2


class RebalancerV3:
    def __init__(self, risk_profile=None):
        self.risk_profile = risk_profile or ACTIVE_RISK_PROFILE
        self.settings = RISK_SETTINGS[self.risk_profile].copy()
        self.max_position_pct = self.settings.get("max_position_pct", 0.20)
        self.cash_target_by_regime = {"BULL": 0.08, "BEAR": 0.15, "SIDEWAYS": 0.10}
        self.EDGE_ACTIVE_SWAP = 0.40
        self.core_tickers = {"SPY", "VT", "QQQ", "IVV", "VOO", "VTI"}
        self.max_reallocations = 4

    def compute_target_weights(self, edges: Dict[str, AssetEdge], regime: str, portfolio_value: float) -> Dict[str, float]:
        # 1. Basis-Ranking (Edge)
        total_edge = sum(max(0.05, e.edge) for e in edges.values())
        if total_edge <= 0:
            return {t: 0.0 for t in edges}
        raw = {t: max(0.05, e.edge) / total_edge for t, e in edges.items()}

        # 2. Confidence Weighting
        for t, e in edges.items():
            raw[t] *= e.confidence_weight

        # 3. Volatility Factor
        for t, e in edges.items():
            raw[t] *= 1.0 / (1.0 + e.volatility / 100.0)

        # 4. Portfolio Factor (small accounts scale)
        portfolio_factor = math.log(max(1, portfolio_value) / 1000 + 1)
        portfolio_factor = max(0.5, min(1.5, portfolio_factor))
        for t in raw:
            raw[t] *= portfolio_factor

        # 5. Neu normalisieren
        total = sum(raw.values())
        if total > 0:
            raw = {t: w / total for t, w in raw.items()}

        # 6. Caps
        for t in raw:
            raw[t] = min(raw[t], self.max_position_pct)
        return raw

    def apply_regime_cash(self, regime: str, target_weights: Dict[str, float]) -> Dict[str, float]:
        cash_target = self.cash_target_by_regime.get(regime, 0.10)
        total_invested = sum(target_weights.values())
        if total_invested > (1 - cash_target):
            scale = (1 - cash_target) / total_invested
            target_weights = {t: w * scale for t, w in target_weights.items()}
        return target_weights

    def identify_swaps(self, current_weights: Dict[str, float], target_weights: Dict[str, float],
                       edges: Dict[str, AssetEdge], regime: str) -> List[Tuple[str, str, float]]:
        overweights = [(t, w) for t, w in current_weights.items() if w > target_weights.get(t, 0) + 0.02]
        underweights = [(t, w) for t, w in target_weights.items() if w > current_weights.get(t, 0) + 0.02]
        overweights.sort(key=lambda x: edges[x[0]].edge)
        underweights.sort(key=lambda x: edges[x[0]].edge, reverse=True)

        swaps = []
        for sell_t, sell_w in overweights:
            if edges[sell_t].is_core:
                continue
            for buy_t, buy_w in underweights:
                if buy_t == sell_t:
                    continue
                edge_gain = edges[buy_t].edge - edges[sell_t].edge
                if edge_gain < self.EDGE_ACTIVE_SWAP:
                    continue
                shift = min(buy_w, sell_w)
                if shift < 0.03:
                    continue
                swaps.append((sell_t, buy_t, edge_gain))
                underweights = [u for u in underweights if u[0] != buy_t]
                if len(swaps) >= self.max_reallocations:
                    break
            if len(swaps) >= self.max_reallocations:
                break
        return swaps

    def generate_trades(self, current_weights: Dict[str, float], target_weights: Dict[str, float],
                        swaps: List[Tuple[str, str, float]], edges: Dict[str, AssetEdge]) -> List[Dict]:
        adjusted_target = target_weights.copy()
        for sell_t, buy_t, _ in swaps:
            adjusted_target[sell_t] = 0.0
            new_buy_target = min(adjusted_target.get(buy_t, 0) + current_weights.get(sell_t, 0), self.max_position_pct)
            adjusted_target[buy_t] = new_buy_target

        trades = []
        for ticker, target in adjusted_target.items():
            current = current_weights.get(ticker, 0.0)
            delta = target - current
            if abs(delta) < 0.02:
                continue
            action = "BUY" if delta > 0 else "SELL"
            confidence = edges[ticker].confidence if ticker in edges else 0.7
            trades.append({
                "ticker": ticker,
                "action": action,
                "target_allocation": target,
                "confidence": confidence,
                "reason": f"Target {target:.1%} vs current {current:.1%}",
            })
        return trades

    def rebalance(self, scores: Dict[str, float], confidences: Dict[str, float],
                  momentums: Dict[str, float], volatilities: Dict[str, float],
                  current_weights: Dict[str, float], cash: float, total_value: float,
                  regime: str) -> Tuple[List[Dict], Dict[str, float], float, str]:
        # Build edges
        edges = {}
        all_tickers = set(scores.keys()) | set(current_weights.keys())
        for ticker in all_tickers:
            score = scores.get(ticker, 50.0)
            conf = confidences.get(ticker, 0.5)
            mom = momentums.get(ticker, 0.0)
            vol = volatilities.get(ticker, 20.0)
            current = current_weights.get(ticker, 0.0)
            is_core = ticker in self.core_tickers

            if regime == "BULL":
                regime_fit = 0.2 if is_core else (0.5 if mom > 5 else -0.2)
            elif regime == "BEAR":
                regime_fit = 0.1 if is_core else (0.3 if mom < -5 else -0.4)
            else:
                regime_fit = 0.0

            edges[ticker] = AssetEdge(
                ticker=ticker, raw_score=score, confidence=conf, momentum=mom,
                volatility=vol, regime_fit=regime_fit, diversification_benefit=0.4,
                current_weight=current, is_core=is_core,
            )

        target_weights = self.compute_target_weights(edges, regime, total_value)
        target_weights = self.apply_regime_cash(regime, target_weights)
        swaps = self.identify_swaps(current_weights, target_weights, edges, regime)
        trades = self.generate_trades(current_weights, target_weights, swaps, edges)

        total_target = sum(target_weights.values())
        cash_target = max(self.cash_target_by_regime.get(regime, 0.10), 1.0 - total_target)

        rationale = f"{len(swaps)} swaps: " + ", ".join([f"{s}→{b}" for s, b, _ in swaps[:3]]) if swaps else "Keine Swaps"
        log.info(f"RebalancerV3: {len(trades)} Trades, Cash {cash_target:.1%}, Swaps {len(swaps)}")
        return trades, target_weights, cash_target, rationale
