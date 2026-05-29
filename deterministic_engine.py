"""
deterministic_engine.py – Deterministische Portfolio-Optimierungs-Engine
=======================================================================
Strikte Hierarchie: Hard Constraints → Objective → Trade Generation → Swap Matching → Priority → Execution Filter.

Ausgabe: SELL list, BUY list, SWAP pairs, kurze deterministische Begründung pro Trade.
Keine narrativen Texte, keine Spekulation.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from logger import log
from config import ACTIVE_RISK_PROFILE, RISK_SETTINGS


@dataclass
class TradeSuggestion:
    ticker: str
    action: str          # "BUY", "SELL", "SWAP_IN", "SWAP_OUT"
    weight_delta: float  # positive = kaufen, negativ = verkaufen (für SWAP: delta des betroffenen Assets)
    reason: str          # kurze deterministische Begründung


class DeterministicPortfolioOptimizer:
    """
    Deterministische Engine, die strikt nach Regeln arbeitet.
    """

    def __init__(self, risk_profile=None):
        self.risk_profile = risk_profile or ACTIVE_RISK_PROFILE
        self.settings = RISK_SETTINGS[self.risk_profile]
        self.max_position_pct = self.settings.get("max_position_pct", 0.20)
        self.min_cash_pct = self.settings.get("min_cash_pct", 0.10)
        self.min_trade_improvement = 0.01  # 1% Gewichtsdifferenz als Schwelle
        self.score_valid_threshold = 30.0   # Scores unter 30 werden ignoriert

    def optimize(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        scores: Dict[str, float],
        total_value: float,
    ) -> Tuple[List[TradeSuggestion], List[TradeSuggestion], List[Tuple[str, str, float]]]:
        """
        Hauptmethode.

        Input:
          current_weights: aktuelle Allokationen (ticker -> float)
          target_weights: Zielallokationen (von CPO oder eigenem Modell)
          scores: Gesamtscores (ticker -> 0-100)
          total_value: Portfolio-Wert (für absolute Prüfungen, optional)

        Returns:
          sell_list, buy_list, swap_pairs
        """

        # 1. Hard Constraints: Prüfe, ob Scoring-System gültig ist
        if not scores or max(scores.values()) < self.score_valid_threshold:
            log.warning("DeterministicEngine: Scoring ungültig – keine Trades erzeugt.")
            return [], [], []

        # 2. Zielgewichte validieren: keine leeren oder ungültigen Werte
        validated_target = self._apply_hard_constraints(target_weights, current_weights)

        # 3. Objective: Minimiere Distanz (current, target)
        deltas = self._compute_deltas(current_weights, validated_target)

        # 4. Trade Generation: SELL candidates (delta < -Schwelle), BUY candidates (delta > Schwelle)
        sell_candidates = [(t, d) for t, d in deltas.items() if d < -self.min_trade_improvement]
        buy_candidates = [(t, d) for t, d in deltas.items() if d > self.min_trade_improvement]

        # 5. Sortieren nach Gewichtsdifferenz-Größe (Priority Rule 1)
        sell_candidates.sort(key=lambda x: x[1])       # negativ, also je kleiner desto stärkerer Sell
        buy_candidates.sort(key=lambda x: x[1], reverse=True)

        # 6. Optional: Risk reduction benefit (Priority 2) – hier vereinfacht durch Score-Bonus
        sell_candidates = self._apply_risk_score_priority(sell_candidates, scores, is_sell=True)
        buy_candidates = self._apply_risk_score_priority(buy_candidates, scores, is_sell=False)

        # 7. Swap Logic: Matche SELL und BUY, um cash-neutral oder cash-positive zu sein
        sell_list, buy_list, swaps = self._match_swaps(sell_candidates, buy_candidates, current_weights, target_weights)

        # 8. Execution Filter: Nur Trades mit erwarteter Verbesserung > Schwelle (bereits oben)
        return sell_list, buy_list, swaps

    def _apply_hard_constraints(self, target_weights: Dict[str, float], current_weights: Dict[str, float]) -> Dict[str, float]:
        """Wendet Positionscaps und Cash-Minimum an."""
        # Cap pro Position
        capped = {t: min(w, self.max_position_pct) for t, w in target_weights.items()}
        # Stelle sicher, dass Cash nicht unter Minimum fällt
        total_invested = sum(capped.values())
        max_investable = 1.0 - self.min_cash_pct
        if total_invested > max_investable:
            scale = max_investable / total_invested
            capped = {t: w * scale for t, w in capped.items()}
        return capped

    def _compute_deltas(self, current: Dict[str, float], target: Dict[str, float]) -> Dict[str, float]:
        """Berechnet delta = target - current für alle Assets."""
        all_tickers = set(current.keys()) | set(target.keys())
        deltas = {}
        for t in all_tickers:
            cur = current.get(t, 0.0)
            tar = target.get(t, 0.0)
            deltas[t] = tar - cur
        return deltas

    def _apply_risk_score_priority(self, candidates: List[Tuple[str, float]], scores: Dict[str, float], is_sell: bool) -> List[Tuple[str, float]]:
        """
        Sortiert Kandidaten nach kombinierter Dringlichkeit:
        - Für SELLs: je niedriger der Score, desto höhere Priorität.
        - Für BUYs: je höher der Score, desto höhere Priorität.
        """
        if is_sell:
            candidates.sort(key=lambda x: (scores.get(x[0], 50), x[1]))
        else:
            candidates.sort(key=lambda x: (-scores.get(x[0], 50), -x[1]))
        return candidates

    def _match_swaps(
        self,
        sell_candidates: List[Tuple[str, float]],
        buy_candidates: List[Tuple[str, float]],
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> Tuple[List[TradeSuggestion], List[TradeSuggestion], List[Tuple[str, str, float]]]:
        """
        Erzeugt SWAP-Paare, wo möglich.
        Returns: (sell_list, buy_list, swap_pairs)
        """
        sells = list(sell_candidates)
        buys = list(buy_candidates)

        swaps = []
        matched_sell = set()
        matched_buy = set()

        for i, (sell_t, sell_delta) in enumerate(sells):
            if sell_t in matched_sell:
                continue
            best_buy_idx = -1
            best_buy_delta = 0
            for j, (buy_t, buy_delta) in enumerate(buys):
                if buy_t in matched_buy or buy_t == sell_t:
                    continue
                swap_amount = min(abs(sell_delta), buy_delta)
                if swap_amount > self.min_trade_improvement:
                    best_buy_idx = j
                    best_buy_delta = buy_delta
                    break
            if best_buy_idx >= 0:
                buy_t, buy_delta = buys[best_buy_idx]
                swap_amount = min(abs(sell_delta), buy_delta)
                swaps.append((sell_t, buy_t, swap_amount))
                matched_sell.add(sell_t)
                matched_buy.add(buy_t)
                # Delta reduzieren
                new_sell_delta = sell_delta + swap_amount
                new_buy_delta = buy_delta - swap_amount
                if abs(new_sell_delta) > self.min_trade_improvement:
                    sells[i] = (sell_t, new_sell_delta)
                if new_buy_delta > self.min_trade_improvement:
                    buys[best_buy_idx] = (buy_t, new_buy_delta)
            else:
                pass

        sell_list = []
        for i, (t, d) in enumerate(sells):
            if t not in matched_sell and d < -self.min_trade_improvement:
                sell_list.append(TradeSuggestion(
                    ticker=t,
                    action="SELL",
                    weight_delta=d,
                    reason=f"overweight by {abs(d):.1%}, target {target_weights.get(t, 0):.1%}"
                ))

        buy_list = []
        for j, (t, d) in enumerate(buys):
            if t not in matched_buy and d > self.min_trade_improvement:
                buy_list.append(TradeSuggestion(
                    ticker=t,
                    action="BUY",
                    weight_delta=d,
                    reason=f"underweight by {d:.1%}, target {target_weights.get(t, 0):.1%}"
                ))

        return sell_list, buy_list, swaps

    def generate_swap_trades(self, swap_pairs: List[Tuple[str, str, float]]) -> List[TradeSuggestion]:
        """Wandelt Swap-Paare in TradeSuggestion-Objekte um."""
        trades = []
        for sell_t, buy_t, amount in swap_pairs:
            trades.append(TradeSuggestion(
                ticker=sell_t,
                action="SWAP_OUT",
                weight_delta=-amount,
                reason=f"swap out to {buy_t}, amount {amount:.1%}"
            ))
            trades.append(TradeSuggestion(
                ticker=buy_t,
                action="SWAP_IN",
                weight_delta=amount,
                reason=f"swap in from {sell_t}, amount {amount:.1%}"
            ))
        return trades
