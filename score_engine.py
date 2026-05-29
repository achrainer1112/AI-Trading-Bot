"""
AI Trading Bot - Quantitative Score Engine (erweitert)
=======================================================
Berechnet für jedes Asset einen Gesamtscore aus:
- Quant Score (technisch/fundamental)
- AI Score (LLM-Marktanalyse)
- Momentum Score (Trendstärke)
- Sentiment Score (optional, aus News)
- Risk Penalty (Volatilität, Drawdown, etc.)

Gibt pro Asset einen FINAL_SCORE (0-100) und ein detailliertes Breakdown.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from logger import log
from config import SECTOR_CLASSIFICATION


# ─────────────────────────────────────────────────────────────
# GEWICHTE FÜR GESAMTSCORE
# ─────────────────────────────────────────────────────────────

FINAL_SCORE_WEIGHTS = {
    "quant": 0.45,
    "ai": 0.20,
    "momentum": 0.20,
    "sentiment": 0.05,
    "risk_penalty": 0.10,   # negative Komponente (Abzug)
}


@dataclass
class ScoreBreakdown:
    """Vollständige Score-Aufschlüsselung für ein Asset."""
    ticker: str
    total_score: float              # 0-100
    quant_score: float = 0.0
    ai_score: float = 0.0
    momentum_score: float = 0.0
    sentiment_score: float = 0.0
    risk_penalty: float = 0.0       # Abzug in Punkten (0-30)
    
    # Rohdaten
    rsi: Optional[float] = None
    momentum_20d: Optional[float] = None
    volatility_annual: Optional[float] = None
    current_price: Optional[float] = None
    current_alloc: float = 0.0
    regime_fit: float = 0.0
    ai_confidence: float = 0.5
    
    # Zusätzliche Metriken
    sma_distance_pct: Optional[float] = None
    relative_strength: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "total_score": round(self.total_score, 1),
            "quant_score": round(self.quant_score, 1),
            "ai_score": round(self.ai_score, 1),
            "momentum_score": round(self.momentum_score, 1),
            "sentiment_score": round(self.sentiment_score, 1),
            "risk_penalty": round(self.risk_penalty, 1),
            "metrics": {
                "rsi": self.rsi,
                "momentum_20d": self.momentum_20d,
                "volatility": self.volatility_annual,
                "current_price": self.current_price,
                "current_alloc": round(self.current_alloc, 4),
                "regime_fit": round(self.regime_fit, 2),
                "sma_distance_pct": self.sma_distance_pct,
                "relative_strength": self.relative_strength,
            }
        }
    
    def to_llm_summary(self) -> str:
        parts = [f"{self.ticker}: {self.total_score:.0f}"]
        if self.momentum_20d is not None:
            parts.append(f"Mom={self.momentum_20d:+.1f}%")
        if self.volatility_annual is not None:
            parts.append(f"Vola={self.volatility_annual:.0f}%")
        if self.current_alloc > 0:
            parts.append(f"Alloc={self.current_alloc:.1%}")
        return " | ".join(parts)
    
    @property
    def confidence(self) -> float:
        """Konfidenz als Hilfsgröße (für Kompatibilität)."""
        return min(1.0, self.total_score / 100.0)


class ScoreEngine:
    """
    Berechnet Gesamtscores für alle Assets basierend auf:
    - Quant Score (vorhandene ScoreEngine)
    - AI Score (aus LLM)
    - Momentum (20d return)
    - Sentiment (optional)
    - Risk Penalty (Volatilität, etc.)
    """

    def __init__(self, positions: Dict[str, Dict] = None, total_value: float = 100_000.0):
        self.positions = positions or {}
        self.total_value = total_value

    def score_all(
        self,
        market_data: Dict[str, Dict],
        regime_state=None,
        ai_scores: Dict[str, float] = None,
        sentiment_scores: Dict[str, float] = None,
    ) -> Dict[str, ScoreBreakdown]:
        """
        Berechnet Gesamtscores für alle Assets.
        
        Args:
            market_data: Dict mit Marktdaten (enthält returns_20d, volatility, rsi, etc.)
            regime_state: Marktregime für Fit-Anpassung
            ai_scores: Dict mit AI-Konfidenz pro Ticker (0-1)
            sentiment_scores: Dict mit Sentiment-Score (-1 bis +1)
        """
        results = {}
        for ticker, data in market_data.items():
            try:
                sb = self._calculate_score(ticker, data, regime_state, ai_scores, sentiment_scores)
                results[ticker] = sb
            except Exception as e:
                log.warning(f"ScoreEngine Fehler bei {ticker}: {e}")
        return results

    def _calculate_score(
        self,
        ticker: str,
        data: Dict,
        regime_state=None,
        ai_scores: Dict[str, float] = None,
        sentiment_scores: Dict[str, float] = None,
    ) -> ScoreBreakdown:
        # ── Rohdaten ─────────────────────────────────────────
        current_price = data.get("current_price") or 0.0
        rsi = data.get("rsi_14")
        return_20d = data.get("return_20d") or data.get("return_30d") or 0.0
        volatility = data.get("volatility_annual_pct") or 20.0
        sma_distance = data.get("sma_distance_pct")
        relative_strength = data.get("relative_strength_vs_spy")
        
        # Aktuelle Allokation
        pos = self.positions.get(ticker, {})
        current_alloc = pos.get("market_value", 0.0) / self.total_value if self.total_value > 0 else 0.0
        
        # ── Komponenten ──────────────────────────────────────
        # 1. Quant Score (bestehend, 0-100)
        quant_score = self._quant_score(data)
        
        # 2. AI Score (aus LLM-Konfidenz, 0-100)
        ai_conf = (ai_scores or {}).get(ticker, 0.5)
        ai_score = ai_conf * 100
        
        # 3. Momentum Score (0-100)
        momentum_score = self._momentum_score(return_20d)
        
        # 4. Sentiment Score (0-100)
        sentiment = (sentiment_scores or {}).get(ticker, 0.0)
        sentiment_score = (sentiment + 1) * 50  # -1..+1 -> 0..100
        
        # 5. Risk Penalty (Abzug, 0-30)
        risk_penalty = self._risk_penalty(volatility, rsi, current_alloc)
        
        # ── Gesamtscore ──────────────────────────────────────
        total = (FINAL_SCORE_WEIGHTS["quant"] * quant_score +
                 FINAL_SCORE_WEIGHTS["ai"] * ai_score +
                 FINAL_SCORE_WEIGHTS["momentum"] * momentum_score +
                 FINAL_SCORE_WEIGHTS["sentiment"] * sentiment_score) - risk_penalty
        total = max(0.0, min(100.0, total))
        
        # Regime-Fit (für Logging)
        regime_fit = self._regime_fit(ticker, regime_state, return_20d)
        
        return ScoreBreakdown(
            ticker=ticker,
            total_score=round(total, 1),
            quant_score=round(quant_score, 1),
            ai_score=round(ai_score, 1),
            momentum_score=round(momentum_score, 1),
            sentiment_score=round(sentiment_score, 1),
            risk_penalty=round(risk_penalty, 1),
            rsi=rsi,
            momentum_20d=return_20d,
            volatility_annual=volatility,
            current_price=current_price,
            current_alloc=current_alloc,
            regime_fit=regime_fit,
            sma_distance_pct=sma_distance,
            relative_strength=relative_strength,
            ai_confidence=ai_conf,
        )
    
    def _quant_score(self, data: Dict) -> float:
        """Berechnet Quant-Score (0-100) aus vorhandenen Metriken."""
        # Vereinfachte Replikation der alten ScoreEngine-Logik
        rsi = data.get("rsi_14", 50)
        return_20d = data.get("return_20d", 0) or 0
        volatility = data.get("volatility_annual_pct", 20)
        sma_dist = data.get("sma_distance_pct", 0) or 0
        
        score = 50.0
        # RSI
        if rsi is not None:
            if rsi < 30:
                score += 15
            elif rsi > 70:
                score -= 15
        # Momentum
        score += return_20d * 1.5
        # Volatility Penalty
        score -= (volatility - 20) * 0.5
        # SMA distance
        if sma_dist > 5:
            score += 10
        elif sma_dist < -5:
            score -= 10
        return max(0.0, min(100.0, score))
    
    def _momentum_score(self, ret_20d: float) -> float:
        """Berechnet Momentum-Score (0-100)."""
        if ret_20d >= 20:
            return 100
        if ret_20d <= -10:
            return 15
        return max(0.0, min(100.0, 50 + ret_20d * 2.5))
    
    def _risk_penalty(self, volatility: float, rsi: Optional[float], current_alloc: float) -> float:
        """Berechnet Risikoabzug (0-30 Punkte)."""
        penalty = 0.0
        # Volatilitätsstrafe
        if volatility > 40:
            penalty += 15
        elif volatility > 30:
            penalty += 8
        # RSI-Überkauft-Strafe
        if rsi is not None and rsi > 80:
            penalty += 10
        elif rsi is not None and rsi > 75:
            penalty += 5
        # Übergewichtungsstrafe (nicht core)
        if current_alloc > 0.20:
            penalty += (current_alloc - 0.20) * 100
        return min(30.0, penalty)
    
    def _regime_fit(self, ticker: str, regime_state, momentum: float) -> float:
        """Berechnet Regime-Fit (-1 bis +1)."""
        if regime_state is None:
            return 0.0
        regime = getattr(regime_state, 'regime', None)
        if regime is None:
            return 0.0
        regime_str = regime.value if hasattr(regime, 'value') else str(regime)
        # Core Holdings immer guten Fit
        if ticker in {"SPY", "VT", "QQQ"}:
            return 0.5
        if regime_str.lower() == "bull":
            return 0.5 if momentum > 0 else -0.2
        elif regime_str.lower() == "bear":
            return 0.3 if ticker in {"XLV", "XLU", "GLD"} else -0.3
        else:
            return 0.0


def rank_assets(scores: Dict[str, ScoreBreakdown]) -> List[ScoreBreakdown]:
    """Gibt alle Assets absteigend nach total_score sortiert zurück."""
    return sorted(scores.values(), key=lambda x: x.total_score, reverse=True)


def build_score_prompt_section(scores: Dict[str, ScoreBreakdown]) -> str:
    """Erstellt den Score-Abschnitt für den LLM-Prompt."""
    if not scores:
        return ""
    lines = ["=== QUANTITATIVE SCORES (deterministisch, vor KI-Interpretation) ==="]
    lines.append(f"{'Ticker':<8} {'Score':>6} {'Signal':<12} {'RSI':>5} {'Mom20d':>8} {'SMA50dist':>10} {'Vola':>7} {'RS':>6} {'MACD':>8} {'BB':>5} {'Alloc':>7}")
    lines.append("-" * 80)
    for ticker, sb in sorted(scores.items(), key=lambda x: x[1].total_score, reverse=True):
        rsi_str = f"{sb.rsi:.0f}" if sb.rsi is not None else "n/a"
        mom_str = f"{sb.momentum_20d:+.1f}%" if sb.momentum_20d is not None else "n/a"
        dist_str = f"{sb.sma_distance_pct:+.1f}%" if sb.sma_distance_pct is not None else "n/a"
        vola_str = f"{sb.volatility_annual:.0f}%" if sb.volatility_annual is not None else "n/a"
        rs_str = f"{sb.relative_strength:.2f}" if sb.relative_strength is not None else "n/a"
        macd_str = f"{sb.macd_histogram:+.4f}" if sb.macd_histogram is not None else "n/a"
        bb_str = f"{sb.bb_position:.2f}" if sb.bb_position is not None else "n/a"
        alloc_str = f"{sb.current_alloc:.1%}"
        lines.append(
            f"{ticker:<8} {sb.total_score:>6.1f} {sb.signal:<12} "
            f"{rsi_str:>5} {mom_str:>8} {dist_str:>10} {vola_str:>7} {rs_str:>6} {macd_str:>8} {bb_str:>5} {alloc_str:>7}"
        )
    lines.append("")
    lines.append("Scoring-Legende: >75=STRONG_BUY | 60-75=BUY | 45-60=HOLD | 30-45=REDUCE | <30=SELL")
    lines.append("Das LLM soll diese Scores berücksichtigen und nur bei starken Makro-Gründen abweichen.")
    return "\n".join(lines)


def rank_candidates(
    scores: Dict[str, ScoreBreakdown],
    min_score: float = 60.0,
    top_k: int = 8,
) -> List[ScoreBreakdown]:
    """Gibt die Top-K Assets mit Score >= min_score zurück, sortiert absteigend."""
    candidates = [sb for sb in scores.values() if sb.total_score >= min_score]
    candidates.sort(key=lambda x: x.total_score, reverse=True)
    return candidates[:top_k]
