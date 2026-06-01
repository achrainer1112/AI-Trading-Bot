"""
cpo_engine.py – Continuous Portfolio Optimizer (score-basiert, ohne Confidence)
===============================================================================
- Nur Assets mit Score >= MIN_SCORE werden in Betracht gezogen.
- Zielgewichte sind proportional zu (Score - MIN_SCORE + 1).
- Wenn keine Assets über MIN_SCORE, wird 100% Cash gehalten.
- Positionsgrößen werden durch max_position_pct und min_trade_value begrenzt.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

from logger import log
from config import (
    ACTIVE_RISK_PROFILE,
    RISK_SETTINGS,
    SECTOR_CLASSIFICATION,
    MIN_SCORE_FOR_BUY,        # neu: in config.py definieren, z.B. 50
    MAX_POSITION_PCT,         # 20% aus config
    MIN_CASH_PCT,             # 10% aus config
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
        self.min_score = MIN_SCORE_FOR_BUY   # aus config, z.B. 50
        self.max_position_pct = self.settings.get("max_position_pct", 0.20)
        self.min_cash_pct = self.settings.get("min_cash_pct", 0.10)

    def compute_target_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Berechnet Zielgewichte proportional zu (score - min_score + 1) für alle Assets
        mit score >= min_score. Wenn keine Assets, wird leeres Dict zurückgegeben.
        """
        # Filtere Assets über Mindestscore
        qualified = {t: s for t, s in scores.items() if s >= self.min_score}
        if not qualified:
            log.info(f"Kein Asset mit Score >= {self.min_score} – Portfolio wird zu 100% Cash")
            return {}

        # Berechne Rohgewichte als Score-Differenz (linear)
        base = {t: max(1.0, s - self.min_score + 1) for t, s in qualified.items()}
        total_base = sum(base.values())
        investable = 1.0 - self.min_cash_pct   # z.B. 90% investierbar, 10% Cash
        raw_weights = {t: (w / total_base) * investable for t, w in base.items()}

        # Begrenze auf maximale Positionsgröße
        capped = {}
        for t, w in raw_weights.items():
            capped[t] = min(w, self.max_position_pct)
        total_capped = sum(capped.values())

        # Normalisierung (falls nach Caps investable überschritten wird)
        if total_capped > investable:
            scale = investable / total_capped
            capped = {t: w * scale for t, w in capped.items()}
        return capped

    def generate_trades(self, current_weights: Dict[str, float], target_weights: Dict[str, float],
                        total_value: float) -> Tuple[List[Dict], Dict[str, float], float]:
        """
        Erzeugt BUY/SELL Vorschläge aus Differenz von Ziel- und aktuellen Gewichten.
        Returns: (trades, final_target_weights, cash_target)
        """
        trades = []
        final_target = target_weights.copy()

        # Bestimme die endgültigen Zielgewichte (nicht im Ziel enthaltene Assets werden auf 0 gesetzt)
        all_tickers = set(current_weights.keys()) | set(target_weights.keys())
        for t in all_tickers:
            if t not in final_target:
                final_target[t] = 0.0

        # Erzeuge Trades, wenn absolute Differenz > 1% (oder individuelle Mindestgrenze)
        for ticker, target in final_target.items():
            current = current_weights.get(ticker, 0.0)
            delta = target - current
            if abs(delta) < 0.01:   # unter 1% ignorieren
                continue
            action = "BUY" if delta > 0 else "SELL"
            trades.append({
                "ticker": ticker,
                "action": action,
                "target_weight": target,
                "confidence": 0.8,   # CPO hat immer hohe Konfidenz
                "reason": f"Score-basiertes Zielgewicht {target:.1%} vs aktuell {current:.1%}",
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
        Returns: (trades, target_weights, cash_target, rationale)
        """
        target_weights = self.compute_target_weights(scores)
        trades, final_target, cash_target = self.generate_trades(current_weights, target_weights, total_value)
        if not target_weights:
            rationale = f"Kein Asset erreicht Mindestscore {self.min_score} – Portfolio wird zu {cash_target:.1%} Cash gehalten."
        else:
            rationale = f"Portfolio optimiert auf {len(target_weights)} Assets (Mindestscore {self.min_score})"
        return trades, final_target, cash_target, rationale
