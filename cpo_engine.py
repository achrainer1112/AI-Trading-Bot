"""
cpo_engine.py – Continuous Portfolio Optimizer (Top-N Auswahl)
===============================================================
- Assets werden nach Score sortiert (höchster zuerst)
- Nur die besten N Assets (mit Score >= MIN_SCORE) werden gehalten
- Positionsgrößen sind proportional zu (Score - MIN_SCORE + 1)
- Nicht gehaltene Assets werden verkauft, neue gekauft
- Cash wenn weniger als N Assets über Mindestscore
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from logger import log
from config import (
    ACTIVE_RISK_PROFILE,
    RISK_SETTINGS,
    MIN_SCORE_FOR_BUY,      # z.B. 50
    MAX_POSITION_COUNT,     # neu: maximale Anzahl Positionen (z.B. 8)
    MAX_POSITION_PCT,       # 20%
    MIN_CASH_PCT,           # 10%
)


@dataclass
class AssetData:
    ticker: str
    score: float
    current_weight: float


class ContinuousPortfolioOptimizer:
    def __init__(self, risk_profile=None):
        self.risk_profile = risk_profile or ACTIVE_RISK_PROFILE
        self.settings = RISK_SETTINGS[self.risk_profile]
        self.min_score = MIN_SCORE_FOR_BUY
        self.max_positions = MAX_POSITION_COUNT
        self.max_position_pct = self.settings.get("max_position_pct", MAX_POSITION_PCT)
        self.min_cash_pct = self.settings.get("min_cash_pct", MIN_CASH_PCT)

    def compute_target_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Berechnet Zielgewichte:
        - Filtere Assets mit Score >= min_score
        - Sortiere absteigend nach Score
        - Nimm die besten max_positions Assets
        - Gewichte proportional zu (score - min_score + 1), normalisiert auf investierbares Kapital
        """
        # Filter
        qualified = [(t, s) for t, s in scores.items() if s >= self.min_score]
        if not qualified:
            log.info(f"Kein Asset mit Score >= {self.min_score} – 100% Cash")
            return {}

        # Sortieren und Top-N
        qualified.sort(key=lambda x: x[1], reverse=True)
        top_assets = qualified[:self.max_positions]

        # Berechne Rohgewichte (linear: je höher Score, desto mehr)
        base = {t: max(1.0, s - self.min_score + 1) for t, s in top_assets}
        total_base = sum(base.values())
        investable = 1.0 - self.min_cash_pct
        raw_weights = {t: (w / total_base) * investable for t, w in base.items()}

        # Caps pro Position
        capped = {}
        for t, w in raw_weights.items():
            capped[t] = min(w, self.max_position_pct)
        total_capped = sum(capped.values())

        # Normalisierung (falls nach Caps investable überschritten)
        if total_capped > investable:
            scale = investable / total_capped
            capped = {t: w * scale for t, w in capped.items()}
        return capped

    def generate_trades(self, current_weights: Dict[str, float], target_weights: Dict[str, float]) -> Tuple[List[Dict], Dict[str, float], float]:
        """
        Erzeugt BUY/SELL Vorschläge aus Differenz.
        """
        trades = []
        final_target = target_weights.copy()
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())

        for t in all_tickers:
            if t not in final_target:
                final_target[t] = 0.0

        for ticker, target in final_target.items():
            current = current_weights.get(ticker, 0.0)
            delta = target - current
            if abs(delta) < 0.005:   # unter 0.5% ignorieren (kleiner als min_trade_value)
                continue
            action = "BUY" if delta > 0 else "SELL"
            trades.append({
                "ticker": ticker,
                "action": action,
                "target_weight": target,
                "confidence": 0.85,
                "reason": f"Top-{self.max_positions} Score-basiert: Ziel {target:.1%} vs aktuell {current:.1%}",
            })
        cash_target = 1.0 - sum(final_target.values())
        return trades, final_target, cash_target

    def optimize(
        self,
        scores: Dict[str, float],
        current_weights: Dict[str, float],
        total_value: float,
    ) -> Tuple[List[Dict], Dict[str, float], float, str]:
        """
        Hauptmethode.
        """
        target_weights = self.compute_target_weights(scores)
        trades, final_target, cash_target = self.generate_trades(current_weights, target_weights)
        if not target_weights:
            rationale = f"Kein Asset erreicht Mindestscore {self.min_score} – Portfolio wird zu {cash_target:.1%} Cash gehalten."
        else:
            rationale = f"Portfolio hält die besten {len(target_weights)} Assets (min Score {self.min_score})"
        return trades, final_target, cash_target, rationale
