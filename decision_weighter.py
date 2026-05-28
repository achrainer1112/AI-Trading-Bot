"""
decision_weighter.py – Decision Weighting Engine
=================================================
Fusioniert verschiedene Signalquellen (Quant, AI, Risk, CPO) zu einer finalen
Entscheidung unter regimespezifischer Gewichtung und Hard Constraints.

Kernprinzipien:
- Keine Einzelsignal-Jagd
- Regime-adaptive Gewichtung
- Risk Engine hat immer Veto-Recht
- Konfliktlösung durch gewichtete Fusion
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from logger import log


@dataclass
class WeightedSignal:
    ticker: str
    quant_score: float      # 0-100
    ai_confidence: float    # 0-100
    risk_score: float       # -1 (riskant) bis +1 (sicher)
    cpo_score: float        # -1 (übergewichtet) bis +1 (untergewichtet)
    current_weight: float
    target_weight: float    # vom CPO
    risk_approved: bool     # Risk Engine erlaubt Trade?

    def normalized_quant(self) -> float:
        return (self.quant_score - 50) / 50.0

    def normalized_ai(self) -> float:
        return (self.ai_confidence - 50) / 50.0

    def normalized_cpo(self) -> float:
        # Zielabweichung: positiv = untergewichtet (Kaufbedarf)
        diff = (self.target_weight - self.current_weight) * 5  # Skalierung
        return max(-1.0, min(1.0, diff))


class DecisionWeighter:
    """
    Kombiniert mehrere Signale zu einer finalen Aktion.
    """

    # Regime-abhängige Gewichte
    WEIGHTS = {
        "BULL": {
            "quant": 0.45,
            "ai": 0.20,
            "risk": 0.20,
            "cpo": 0.15,
        },
        "BEAR": {
            "risk": 0.40,
            "quant": 0.25,
            "ai": 0.20,
            "cpo": 0.15,
        },
        "SIDEWAYS": {
            "ai": 0.30,
            "quant": 0.30,
            "risk": 0.25,
            "cpo": 0.15,
        },
    }

    def __init__(self, default_regime: str = "SIDEWAYS"):
        self.default_regime = default_regime

    def get_regime(self, regime_state) -> str:
        if regime_state is None:
            return self.default_regime
        regime_val = getattr(regime_state, 'regime', None)
        if regime_val is None:
            return self.default_regime
        if hasattr(regime_val, 'value'):
            return regime_val.value.upper()
        return str(regime_val).upper()

    def compute_final_score_and_action(
        self,
        signal: WeightedSignal,
        regime: str,
        high_volatility_stress: bool = False,
    ) -> Tuple[float, str, float]:
        """
        Berechnet finalen Score, Aktion und Konfidenz für ein Asset.
        Returns: (final_score, action, final_confidence)
        """
        weights = self.WEIGHTS.get(regime, self.WEIGHTS["SIDEWAYS"])

        # Normierte Einzelscores
        q = signal.normalized_quant()
        a = signal.normalized_ai()
        r = signal.risk_score
        c = signal.normalized_cpo()

        # Gewichtete Summe (nur wenn risk_approved=True, sonst Force HOLD)
        if not signal.risk_approved:
            log.debug(f"{signal.ticker}: Risk Engine blockiert → HOLD")
            return 0.0, "HOLD", 0.0

        final_score = (
            q * weights.get("quant", 0) +
            a * weights.get("ai", 0) +
            r * weights.get("risk", 0) +
            c * weights.get("cpo", 0)
        )

        # Volatilitätsstress: BUY Confidence reduzieren
        if high_volatility_stress and final_score > 0:
            final_score *= 0.8
            log.debug(f"{signal.ticker}: Volatilitätsstress → BUY Score reduziert")

        # Aktion basierend auf final_score
        if final_score > 0.35:
            action = "BUY"
        elif final_score < -0.35:
            action = "SELL"
        else:
            action = "HOLD"

        # Konfidenz: gewichtete Übereinstimmung (Standardabweichung der Scores)
        scores = [q, a, r, c]
        # Gewichtete Standardabweichung (niedrige Abweichung = hohe Konfidenz)
        weighted_mean = final_score
        weighted_var = sum(
            weights.get(k, 0) * (s - weighted_mean) ** 2
            for k, s in zip(["quant", "ai", "risk", "cpo"], scores)
        )
        weighted_std = weighted_var ** 0.5
        # Konfidenz: 1 - min(1, weighted_std) (std=0 -> 1, std=1 -> 0)
        confidence = 1.0 - min(1.0, weighted_std)
        # Zusätzlich: bei starker Disagreement (alle Scores unterschiedlich) reduzieren
        if max(scores) - min(scores) > 1.2:
            confidence *= 0.7

        return final_score, action, confidence

    def process_assets(
        self,
        signals: List[WeightedSignal],
        regime_state,
        high_volatility_stress: bool = False,
    ) -> List[Dict]:
        """
        Verarbeitet eine Liste von Assets und gibt finale Entscheidungen zurück.
        """
        regime = self.get_regime(regime_state)
        decisions = []

        for s in signals:
            final_score, action, confidence = self.compute_final_score_and_action(
                s, regime, high_volatility_stress
            )
            decisions.append({
                "ticker": s.ticker,
                "action": action,
                "confidence": round(confidence, 3),
                "final_score": round(final_score, 3),
                "target_weight": s.target_weight if action in ("BUY", "SELL") else s.current_weight,
                "reason": f"{regime} regime | Quant={s.quant_score:.0f} AI={s.ai_confidence:.0f} Risk={s.risk_score:.2f} CPO={s.cpo_score:.2f}",
            })
        return decisions
