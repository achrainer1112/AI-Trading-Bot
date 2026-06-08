"""
AI Trading Bot - Quantitative Score Engine (überarbeitet)
===========================================================
Berechnet einen robusten Score (0-100) aus:

- Trend (SMA20/SMA50/SMA200)          30%
- Gewichtetes Momentum (20d/60d/120d)  25%
- Relative Stärke vs. SPY              15%
- Volatilitätsstrafe                  10%
- RSI (Überkauft/Überverkauft)        10%
- Preis-Lage (52-Wochen-High)         10%

Keine doppelte Momentum-Gewichtung mehr.
Relative Stärke jetzt fester Bestandteil.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from logger import log
from config import SECTOR_CLASSIFICATION


# ========== Konstanten ==========
SCORE_STRONG_BUY = 75
SCORE_BUY = 60
SCORE_HOLD = 45
SCORE_REDUCE = 30

# Neue Gewichte
WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.25,
    "relative_strength": 0.15,
    "volatility": 0.10,      # negative Komponente
    "rsi": 0.10,
    "price_position": 0.10,
}


@dataclass
class ScoreBreakdown:
    """Vollständige Score-Aufschlüsselung für ein Asset."""
    ticker: str
    total_score: float
    signal: str
    recommended_action: str

    # Komponenten (0-100)
    trend_score: float = 0.0
    momentum_score: float = 0.0
    relative_strength_score: float = 0.0
    volatility_penalty: float = 0.0
    rsi_score: float = 0.0
    price_position_score: float = 0.0

    # Rohdaten für Logging / LLM
    rsi: Optional[float] = None
    momentum_20d: Optional[float] = None
    momentum_60d: Optional[float] = None
    momentum_120d: Optional[float] = None
    volatility_annual: Optional[float] = None
    current_price: Optional[float] = None
    current_alloc: float = 0.0
    sma_distance_pct: Optional[float] = None
    relative_strength: Optional[float] = None
    price_52w_high_pct: Optional[float] = None

    @property
    def confidence(self) -> float:
        return min(1.0, self.total_score / 100.0)

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "total_score": round(self.total_score, 1),
            "signal": self.signal,
            "recommended_action": self.recommended_action,
            "components": {
                "trend_score": round(self.trend_score, 1),
                "momentum_score": round(self.momentum_score, 1),
                "relative_strength_score": round(self.relative_strength_score, 1),
                "volatility_penalty": round(self.volatility_penalty, 1),
                "rsi_score": round(self.rsi_score, 1),
                "price_position_score": round(self.price_position_score, 1),
            },
            "metrics": {
                "rsi": self.rsi,
                "momentum_20d": self.momentum_20d,
                "momentum_60d": self.momentum_60d,
                "momentum_120d": self.momentum_120d,
                "volatility_annual": self.volatility_annual,
                "relative_strength": self.relative_strength,
                "current_price": self.current_price,
                "current_alloc": round(self.current_alloc, 4),
                "sma_distance_pct": self.sma_distance_pct,
                "price_52w_high_pct": self.price_52w_high_pct,
            }
        }

    def to_llm_summary(self) -> str:
        parts = [f"{self.ticker}: {self.total_score:.0f}"]
        if self.momentum_20d is not None:
            parts.append(f"Mom20d={self.momentum_20d:+.1f}%")
        if self.relative_strength is not None:
            parts.append(f"RS={self.relative_strength:.2f}")
        if self.volatility_annual is not None:
            parts.append(f"Vola={self.volatility_annual:.0f}%")
        if self.current_alloc > 0:
            parts.append(f"Alloc={self.current_alloc:.1%}")
        return " | ".join(parts)


class ScoreEngine:
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
        results = {}
        ai_scores = ai_scores or {}
        sentiment_scores = sentiment_scores or {}
        for ticker, data in market_data.items():
            if not isinstance(data, dict):
                log.debug(f"Überspringe {ticker}: data ist {type(data)}")
                dummy = ScoreBreakdown(ticker=ticker, total_score=50.0, signal="HOLD", recommended_action="HOLD")
                results[ticker] = dummy
                continue
            try:
                sb = self._calculate_score(ticker, data, regime_state)
                results[ticker] = sb
            except Exception as e:
                log.warning(f"ScoreEngine Fehler bei {ticker}: {e}")
                dummy = ScoreBreakdown(ticker=ticker, total_score=50.0, signal="HOLD", recommended_action="HOLD")
                results[ticker] = dummy
        return results

    def _calculate_score(
        self,
        ticker: str,
        data: Dict,
        regime_state=None,
    ) -> ScoreBreakdown:
        # ─── Rohdaten extrahieren ──────────────────────────────
        current_price = data.get("current_price", 0.0) or 0.0
        rsi = data.get("rsi_14")
        volatility = data.get("volatility_annual_pct") or 20.0

        # Returns (verschiedene Zeitfenster)
        return_20d = data.get("return_20d") or data.get("return_30d") or 0.0
        return_60d = data.get("return_60d") or 0.0
        return_120d = data.get("return_90d") or 0.0  # 90d als Proxy für 120d

        # SMAs
        sma20 = data.get("sma_20")
        sma50 = data.get("sma_50")
        sma200 = data.get("sma_200")
        sma_distance_pct = data.get("sma_distance_pct")  # Abstand zu SMA50

        # 52w-High
        high_52w = data.get("high_52w")
        price_52w_high_pct = None
        if high_52w and current_price > 0:
            price_52w_high_pct = (current_price / high_52w) * 100

        # Relative Stärke vs. SPY (20d)
        relative_strength = data.get("relative_strength_vs_spy")

        # Aktuelle Allokation
        pos = self.positions.get(ticker, {})
        current_alloc = pos.get("market_value", 0.0) / self.total_value if self.total_value > 0 else 0.0

        # ─── Komponenten (jeweils 0-100) ───────────────────────
        trend_score = self._trend_score(current_price, sma20, sma50, sma200, sma_distance_pct)
        momentum_score = self._momentum_score(return_20d, return_60d, return_120d)
        rs_score = self._relative_strength_score(relative_strength)
        volatility_penalty = self._volatility_penalty(volatility)  # Abzug (0-30)
        rsi_score = self._rsi_score(rsi)
        price_pos_score = self._price_position_score(price_52w_high_pct)

        # ─── Gewichteter Gesamtscore ───────────────────────────
        total = (WEIGHTS["trend"] * trend_score +
                 WEIGHTS["momentum"] * momentum_score +
                 WEIGHTS["relative_strength"] * rs_score -
                 WEIGHTS["volatility"] * volatility_penalty +
                 WEIGHTS["rsi"] * rsi_score +
                 WEIGHTS["price_position"] * price_pos_score)
        total = max(0.0, min(100.0, total))

        # ─── Signal & Aktion ───────────────────────────────────
        if total >= SCORE_STRONG_BUY:
            signal = "STRONG_BUY"
            action = "BUY"
        elif total >= SCORE_BUY:
            signal = "BUY"
            action = "BUY"
        elif total >= SCORE_HOLD:
            signal = "HOLD"
            action = "HOLD"
        elif total >= SCORE_REDUCE:
            signal = "REDUCE"
            action = "SELL"
        else:
            signal = "SELL"
            action = "SELL"

        return ScoreBreakdown(
            ticker=ticker,
            total_score=round(total, 1),
            signal=signal,
            recommended_action=action,
            trend_score=round(trend_score, 1),
            momentum_score=round(momentum_score, 1),
            relative_strength_score=round(rs_score, 1),
            volatility_penalty=round(volatility_penalty, 1),
            rsi_score=round(rsi_score, 1),
            price_position_score=round(price_pos_score, 1),
            rsi=rsi,
            momentum_20d=return_20d,
            momentum_60d=return_60d,
            momentum_120d=return_120d,
            volatility_annual=volatility,
            current_price=current_price,
            current_alloc=current_alloc,
            sma_distance_pct=sma_distance_pct,
            relative_strength=relative_strength,
            price_52w_high_pct=price_52w_high_pct,
        )

    # ─── Komponentenberechnungen ───────────────────────────────

    def _trend_score(self, price: float, sma20, sma50, sma200, sma_dist) -> float:
        """Trend-Score basierend auf SMA-Strukturen."""
        score = 50.0
        # Preise über SMAs
        if price and sma20:
            score += 10 if price > sma20 else -10
        if price and sma50:
            score += 15 if price > sma50 else -15
        if price and sma200:
            score += 15 if price > sma200 else -15
        # SMA-Distanz (wie bisher)
        if sma_dist is not None:
            if 0 < sma_dist < 5:
                score += 12
            elif 5 <= sma_dist < 15:
                score += 8
            elif sma_dist >= 15:
                score -= 8
            elif -5 <= sma_dist <= 0:
                score -= 4
            elif sma_dist < -5:
                score -= 12
        return max(0.0, min(100.0, score))

    def _momentum_score(self, ret_20d: float, ret_60d: float, ret_120d: float) -> float:
        """Gewichteter Momentum-Score über 20/60/120 Tage."""
        weighted = 0.5 * ret_20d + 0.3 * ret_60d + 0.2 * ret_120d
        # Linearer Zusammenhang: 0% -> 50, +20% -> 100, -20% -> 0
        score = 50.0 + weighted * 2.5
        return max(0.0, min(100.0, score))

    def _relative_strength_score(self, rs: Optional[float]) -> float:
        """Relativer Stärke-Score: RS > 1,0 = outperformance."""
        if rs is None:
            return 50.0
        # RS = asset_return / spy_return (z. B. 1,15 = 15% besser)
        # Score linear: 1,0 -> 50, 1,3 -> 100, 0,7 -> 0
        score = (rs - 0.7) / 0.6 * 100
        return max(0.0, min(100.0, score))

    def _volatility_penalty(self, vol: float) -> float:
        """Rückgabe eines Abzugs (0-30), je höher die Vola, desto höher der Abzug."""
        if vol < 15:
            return 0.0
        elif vol < 25:
            return (vol - 15) / 10 * 5   # 0-5 Punkte
        elif vol < 40:
            return 5.0 + (vol - 25) / 15 * 10  # 5-15 Punkte
        else:
            return 15.0 + min(15.0, (vol - 40) / 60 * 15)  # 15-30 Punkte

    def _rsi_score(self, rsi: Optional[float]) -> float:
        if rsi is None:
            return 50.0
        # RSI 50 -> 50, <30 -> 80 (oversold bullish), >70 -> 20 (overbought bearish)
        if rsi < 30:
            return 80.0
        elif rsi < 40:
            return 70.0
        elif rsi <= 60:
            return 50.0
        elif rsi <= 70:
            return 30.0
        else:
            return 20.0

    def _price_position_score(self, pct_of_high: Optional[float]) -> float:
        """Je näher am 52w-High, desto höher der Score (Trendstärke)."""
        if pct_of_high is None:
            return 50.0
        # 100% = 100 Punkte, 80% = 50 Punkte, 60% = 0 Punkte
        score = (pct_of_high - 60) / 40 * 100
        return max(0.0, min(100.0, score))


# ========== Hilfsfunktionen für Prompt und Ranking ==========
def rank_candidates(scores: Dict[str, ScoreBreakdown], min_score: float = SCORE_BUY, top_k: int = 8) -> List[ScoreBreakdown]:
    candidates = [sb for sb in scores.values() if sb.total_score >= min_score and sb.recommended_action == "BUY"]
    candidates.sort(key=lambda x: x.total_score, reverse=True)
    return candidates[:top_k]


def build_score_prompt_section(scores: Dict[str, ScoreBreakdown]) -> str:
    if not scores:
        return ""
    lines = ["=== QUANTITATIVE SCORES (überarbeitet) ==="]
    lines.append(f"{'Ticker':<8} {'Score':>6} {'Signal':<12} {'RSI':>5} {'Mom20d':>8} {'RS':>6} {'Vola':>7} {'Alloc':>7}")
    lines.append("-" * 65)
    for ticker, sb in sorted(scores.items(), key=lambda x: x[1].total_score, reverse=True):
        rsi_str = f"{sb.rsi:.0f}" if sb.rsi is not None else "n/a"
        mom_str = f"{sb.momentum_20d:+.1f}%" if sb.momentum_20d is not None else "n/a"
        rs_str = f"{sb.relative_strength:.2f}" if sb.relative_strength is not None else "n/a"
        vola_str = f"{sb.volatility_annual:.0f}%" if sb.volatility_annual is not None else "n/a"
        alloc_str = f"{sb.current_alloc:.1%}"
        lines.append(f"{ticker:<8} {sb.total_score:>6.1f} {sb.signal:<12} {rsi_str:>5} {mom_str:>8} {rs_str:>6} {vola_str:>7} {alloc_str:>7}")
    lines.append("")
    lines.append("Legende: >75=STRONG_BUY | 60-75=BUY | 45-60=HOLD | 30-45=REDUCE | <30=SELL")
    return "\n".join(lines)


# ========== Fallback PortfolioOptimizer (für AI-Analysis, bleibt kompatibel) ==========
@dataclass
class PortfolioAllocation:
    target_allocations: Dict[str, float]
    recommended_sells: List[str]
    cash_target: float
    rationale: Dict[str, str]

    def summary(self) -> str:
        lines = [f"Portfolio Allocation ({len(self.target_allocations)} positions, cash={self.cash_target:.1%}):"]
        for t, a in sorted(self.target_allocations.items(), key=lambda x: -x[1]):
            lines.append(f"  BUY  {t}: {a:.1%}")
        for t in self.recommended_sells:
            lines.append(f"  SELL {t}")
        return "\n".join(lines)


class PortfolioOptimizer:
    def __init__(self, sector_map=None, correlation_groups=None, risk_settings=None):
        from config import SECTOR_CLASSIFICATION, CORRELATION_GROUPS, RISK_SETTINGS, ACTIVE_RISK_PROFILE
        self.sector_map = sector_map or SECTOR_CLASSIFICATION
        self.correlation_groups = correlation_groups or CORRELATION_GROUPS
        self.risk_settings = risk_settings or RISK_SETTINGS[ACTIVE_RISK_PROFILE]
        self.max_position_pct = self.risk_settings.get("max_position_pct", 0.20)
        self.min_cash_pct = self.risk_settings.get("min_cash_pct", 0.10)
        self.max_sector_pct = self.risk_settings.get("max_sector_exposure", 0.45)
        self.min_score_for_buy = self.risk_settings.get("min_buy_score", 60)

    def optimize(self, scores: Dict[str, ScoreBreakdown], current_positions=None, total_value=100_000.0,
                 regime_state=None, market_confidence=0.5, portfolio_risk_high=False) -> PortfolioAllocation:
        # Einfache Fallback-Implementierung (wird normalerweise nicht verwendet)
        candidates = [(t, sb.total_score) for t, sb in scores.items() if sb.total_score >= self.min_score_for_buy]
        candidates.sort(key=lambda x: x[1], reverse=True)
        target = {}
        investable = 1.0 - self.min_cash_pct
        if candidates:
            per_asset = investable / min(len(candidates), 5)
            for ticker, _ in candidates[:5]:
                target[ticker] = min(per_asset, self.max_position_pct)
        total_target = sum(target.values())
        if total_target > investable:
            scale = investable / total_target
            target = {t: w * scale for t, w in target.items()}
        cash_target = 1.0 - sum(target.values())
        return PortfolioAllocation(
            target_allocations=target,
            recommended_sells=[],
            cash_target=max(cash_target, self.min_cash_pct),
            rationale={t: f"Score={scores[t].total_score:.0f}" for t in target}
        )

    def build_prompt_section(self, scores: Dict[str, ScoreBreakdown], allocation: PortfolioAllocation) -> str:
        lines = ["=== PORTFOLIO OPTIMIZER VORSCHLAG (Fallback) ==="]
        lines.append(f"Cash reserve: {allocation.cash_target:.1%}")
        lines.append("")
        lines.append(f"{'Ticker':<8} {'TargetAlloc':>11}")
        lines.append("-" * 25)
        for ticker, alloc in sorted(allocation.target_allocations.items(), key=lambda x: -x[1]):
            lines.append(f"{ticker:<8} {alloc:>10.1%}")
        return "\n".join(lines)
