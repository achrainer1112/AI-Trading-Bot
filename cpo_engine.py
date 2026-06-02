"""
cpo_engine.py – Continuous Portfolio Optimizer (Score-basiert)
===============================================================
Berechnet Zielportfolio basierend auf Score-Ranking, ohne KI-Confidence.
"""

from typing import Dict, List, Tuple

from logger import log
from config import (
    MIN_SCORE_FOR_BUY,
    MAX_POSITION_COUNT,
    MAX_POSITION_PCT,
    MIN_CASH_PCT,
)


class ContinuousPortfolioOptimizer:
    def __init__(self, risk_profile=None):
        self.min_score = MIN_SCORE_FOR_BUY
        self.max_positions = MAX_POSITION_COUNT
        self.max_position_pct = MAX_POSITION_PCT
        self.min_cash_pct = MIN_CASH_PCT

    def optimize(
        self,
        scores: Dict[str, float],
        confidences: Dict[str, float] = None,   # wird ignoriert
        momentums: Dict[str, float] = None,     # wird ignoriert
        volatilities: Dict[str, float] = None,  # wird ignoriert
        current_weights: Dict[str, float] = None,
        regime: str = "BULL",
        portfolio_value: float = 0,
    ) -> Tuple[Dict[str, float], float, Dict[str, List[str]], Dict[str, List[str]], str]:
        """
        Berechnet Zielgewichte:
        - Nur Assets mit Score >= MIN_SCORE_FOR_BUY werden berücksichtigt.
        - Maximal MAX_POSITION_COUNT Assets.
        - Gewichte proportional zum Score, gedeckelt auf MAX_POSITION_PCT.
        - Cash-Reserve = MIN_CASH_PCT.
        """
        # 1. Filtere und sortiere
        qualified = [(t, s) for t, s in scores.items() if s >= self.min_score]
        if not qualified:
            log.info(f"Kein Asset mit Score >= {self.min_score} – 100% Cash")
            return {}, 1.0, {}, {}, "Keine Assets erreichen Mindestscore"

        qualified.sort(key=lambda x: x[1], reverse=True)
        top_assets = qualified[:self.max_positions]

        # 2. Rohgewichte proportional zu Score
        total_score = sum(s for _, s in top_assets)
        investable = 1.0 - self.min_cash_pct
        raw_weights = {t: (s / total_score) * investable for t, s in top_assets}

        # 3. Caps anwenden
        capped_weights = {}
        for t, w in raw_weights.items():
            capped_weights[t] = min(w, self.max_position_pct)

        # 4. Normalisierung (falls durch Caps investable überschritten)
        total_capped = sum(capped_weights.values())
        if total_capped > investable:
            scale = investable / total_capped
            capped_weights = {t: w * scale for t, w in capped_weights.items()}

        # 5. Cash-Ziel
        cash_target = 1.0 - sum(capped_weights.values())
        cash_target = max(self.min_cash_pct, cash_target)

        # Dummy-Cluster für Kompatibilität (nicht verwendet)
        buy_cluster = {t: [t] for t in capped_weights}
        sell_cluster = {}
        rationale = f"Score-basiert: {len(capped_weights)} Assets (Score >= {self.min_score})"

        return capped_weights, cash_target, buy_cluster, sell_cluster, rationale
