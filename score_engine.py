"""
AI Trading Bot - Quantitative Score Engine (Type Safe)
=======================================================
Sichere Extraktion von Daten – nur Dictionaries werden verarbeitet,
alle anderen Typen werden durch Dummy-Scores ersetzt (ohne Warnungen im Normalbetrieb).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from logger import log
from config import SECTOR_CLASSIFICATION, CORRELATION_GROUPS, RISK_SETTINGS, ACTIVE_RISK_PROFILE

SCORE_STRONG_BUY = 75
SCORE_BUY = 60
SCORE_HOLD = 45
SCORE_REDUCE = 30


@dataclass
class ScoreBreakdown:
    ticker: str
    total_score: float
    signal: str = "HOLD"
    recommended_action: str = "HOLD"
    quant_score: float = 0.0
    ai_score: float = 0.0
    momentum_score: float = 0.0
    sentiment_score: float = 0.0
    risk_penalty: float = 0.0
    rsi: Optional[float] = None
    momentum_20d: Optional[float] = None
    volatility_annual: Optional[float] = None
    current_price: Optional[float] = None
    current_alloc: float = 0.0
    regime_fit: float = 0.0
    sma_distance_pct: Optional[float] = None
    relative_strength: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_position: Optional[float] = None
    sma20: Optional[float] = None
    sma50: Optional[float] = None

    @property
    def confidence(self) -> float:
        return min(1.0, self.total_score / 100.0)

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "total_score": round(self.total_score, 1),
            "signal": self.signal,
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


class ScoreEngine:
    def __init__(self, positions: Dict[str, Dict] = None, total_value: float = 100_000.0):
        self.positions = positions or {}
        self.total_value = total_value

    def score_all(self, market_data: Dict, regime_state=None, ai_scores=None, sentiment_scores=None) -> Dict[str, ScoreBreakdown]:
        """
        Gibt für jeden Ticker einen ScoreBreakdown zurück.
        Bei ungültigen Daten (kein dict) wird ein Dummy-Score (50) zurückgegeben.
        Keine lauten Warnungen im Normalbetrieb – nur Debug-Logs.
        """

        if not isinstance(ai_scores, dict):
            ai_scores = {}
        if not isinstance(sentiment_scores, dict):
            sentiment_scores = {}
        
        results = {}
        ai_scores = ai_scores or {}
        sentiment_scores = sentiment_scores or {}
        for ticker, data in market_data.items():
            # TYPE SAFETY: Nur Dictionaries verarbeiten
            if not isinstance(data, dict):
                log.debug(f"ScoreEngine: Überspringe {ticker}, data ist {type(data)} (kein Dict) – verwende Dummy")
                dummy = ScoreBreakdown(ticker=ticker, total_score=50.0, signal="HOLD", recommended_action="HOLD")
                results[ticker] = dummy
                continue
            try:
                sb = self._calculate_score(ticker, data, regime_state, ai_scores, sentiment_scores)
                results[ticker] = sb
            except Exception as e:
                log.warning(f"ScoreEngine Fehler bei {ticker}: {e} – verwende Dummy")
                dummy = ScoreBreakdown(ticker=ticker, total_score=50.0, signal="HOLD", recommended_action="HOLD")
                results[ticker] = dummy
        return results

    def _calculate_score(
        self,
        ticker: str,
        data: Dict,
        regime_state=None,
        ai_scores: Dict[str, float] = None,
        sentiment_scores: Dict[str, float] = None,
    ) -> ScoreBreakdown:
        # Sichere Extraktion mit Defaults
        current_price = data.get("current_price", 0.0) or 0.0
        rsi = data.get("rsi_14")
        return_20d = data.get("return_20d") or data.get("return_30d") or 0.0
        volatility = data.get("volatility_annual_pct") or 20.0
        sma_dist = data.get("sma_distance_pct")
        rs = data.get("relative_strength_vs_spy")
        macd_hist = data.get("macd_histogram")
        bb_pos = data.get("bb_position")
        sma20 = data.get("sma_20")
        sma50 = data.get("sma_50")

        pos = self.positions.get(ticker, {})
        current_alloc = pos.get("market_value", 0.0) / self.total_value if self.total_value > 0 else 0.0

        quant_score = self._quant_score(data)
        ai_conf = ai_scores.get(ticker, 0.5)
        ai_score = ai_conf * 100
        momentum_score = self._momentum_score(return_20d)
        sentiment = sentiment_scores.get(ticker, 0.0)
        sentiment_score = (sentiment + 1) * 50
        risk_penalty = self._risk_penalty(volatility, rsi, current_alloc)

        total = quant_score * 0.45 + ai_score * 0.20 + momentum_score * 0.20 + sentiment_score * 0.05 - risk_penalty
        total = max(0.0, min(100.0, total))

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

        regime_fit = self._regime_fit(ticker, regime_state, return_20d)

        return ScoreBreakdown(
            ticker=ticker,
            total_score=round(total, 1),
            signal=signal,
            recommended_action=action,
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
            sma_distance_pct=sma_dist,
            relative_strength=rs,
            macd_histogram=macd_hist,
            bb_position=bb_pos,
            sma20=sma20,
            sma50=sma50,
        )

    def _quant_score(self, data: Dict) -> float:
        rsi = data.get("rsi_14", 50) or 50
        return_20d = data.get("return_20d", 0) or 0
        volatility = data.get("volatility_annual_pct", 20) or 20
        sma_dist = data.get("sma_distance_pct", 0) or 0
        score = 50.0
        if isinstance(rsi, (int, float)):
            if rsi < 30:
                score += 15
            elif rsi > 70:
                score -= 15
        score += return_20d * 1.5
        score -= (volatility - 20) * 0.5
        if sma_dist > 5:
            score += 10
        elif sma_dist < -5:
            score -= 10
        return max(0.0, min(100.0, score))

    def _momentum_score(self, ret_20d: float) -> float:
        if ret_20d >= 20:
            return 100
        if ret_20d <= -10:
            return 15
        return max(0.0, min(100.0, 50 + ret_20d * 2.5))

    def _risk_penalty(self, volatility: float, rsi: Optional[float], current_alloc: float) -> float:
        penalty = 0.0
        if volatility > 40:
            penalty += 15
        elif volatility > 30:
            penalty += 8
        if rsi is not None:
            if rsi > 80:
                penalty += 10
            elif rsi > 75:
                penalty += 5
        if current_alloc > 0.20:
            penalty += (current_alloc - 0.20) * 100
        return min(30.0, penalty)

    def _regime_fit(self, ticker: str, regime_state, momentum: float) -> float:
        if regime_state is None:
            return 0.0
        regime = getattr(regime_state, 'regime', None)
        if regime is None:
            return 0.0
        regime_str = regime.value if hasattr(regime, 'value') else str(regime)
        if ticker in {"SPY", "VT", "QQQ"}:
            return 0.5
        if regime_str.lower() == "bull":
            return 0.5 if momentum > 0 else -0.2
        elif regime_str.lower() == "bear":
            return 0.3 if ticker in {"XLV", "XLU", "GLD"} else -0.3
        return 0.0


def rank_candidates(scores: Dict[str, ScoreBreakdown], min_score: float = SCORE_BUY, top_k: int = 8) -> List[ScoreBreakdown]:
    candidates = [sb for sb in scores.values() if sb.total_score >= min_score and sb.recommended_action == "BUY"]
    candidates.sort(key=lambda x: x.total_score, reverse=True)
    return candidates[:top_k]


def build_score_prompt_section(scores: Dict[str, ScoreBreakdown]) -> str:
    if not scores:
        return ""
    lines = ["=== QUANTITATIVE SCORES ==="]
    lines.append(f"{'Ticker':<8} {'Score':>6} {'Signal':<12} {'RSI':>5} {'Mom20d':>8} {'Vola':>7} {'Alloc':>7}")
    lines.append("-" * 60)
    for ticker, sb in sorted(scores.items(), key=lambda x: x[1].total_score, reverse=True):
        rsi_str = f"{sb.rsi:.0f}" if sb.rsi is not None else "n/a"
        mom_str = f"{sb.momentum_20d:+.1f}%" if sb.momentum_20d is not None else "n/a"
        vola_str = f"{sb.volatility_annual:.0f}%" if sb.volatility_annual is not None else "n/a"
        alloc_str = f"{sb.current_alloc:.1%}"
        lines.append(f"{ticker:<8} {sb.total_score:>6.1f} {sb.signal:<12} {rsi_str:>5} {mom_str:>8} {vola_str:>7} {alloc_str:>7}")
    lines.append("")
    lines.append("Legende: >75=STRONG_BUY | 60-75=BUY | 45-60=HOLD | 30-45=REDUCE | <30=SELL")
    return "\n".join(lines)


@dataclass
class PortfolioAllocation:
    target_allocations: Dict[str, float]
    recommended_sells: List[str]
    cash_target: float
    rationale: Dict[str, str]

    def summary(self) -> str:
        lines = [f"Portfolio Allocation ({len(self.target_allocations)} positions, cash={self.cash_target:.1%}):"]
        for t, a in sorted(self.target_allocations.items(), key=lambda x: -x[1]):
            lines.append(f"  BUY  {t}: {a:.1%}  — {self.rationale.get(t,'')}")
        for t in self.recommended_sells:
            lines.append(f"  SELL {t}")
        return "\n".join(lines)


class PortfolioOptimizer:
    def __init__(self, sector_map=None, correlation_groups=None, risk_settings=None):
        self.sector_map = sector_map or SECTOR_CLASSIFICATION
        self.correlation_groups = correlation_groups or CORRELATION_GROUPS
        self.risk_settings = risk_settings or RISK_SETTINGS[ACTIVE_RISK_PROFILE]
        self.max_position_pct = self.risk_settings.get("max_position_pct", 0.20)
        self.min_cash_pct = self.risk_settings.get("min_cash_pct", 0.10)
        self.max_sector_pct = self.risk_settings.get("max_sector_exposure", 0.45)
        self.min_score_for_buy = self.risk_settings.get("min_buy_score", 60)

    def optimize(self, scores: Dict[str, ScoreBreakdown], current_positions=None, total_value=100_000.0,
                 regime_state=None, market_confidence=0.5, portfolio_risk_high=False) -> PortfolioAllocation:
        candidates = []
        for ticker, sb in scores.items():
            if sb.total_score >= self.min_score_for_buy:
                candidates.append((ticker, sb.total_score))
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
