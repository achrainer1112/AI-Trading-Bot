"""
AI Trading Bot - Quantitative Score Engine
============================================
Deterministisches Scoring-System für alle Assets.

Berechnet einen Gesamtscore 0-100 aus mehreren Komponenten:
  - Trend Score      (SMA-Abstand, Preis > SMA)
  - Momentum Score   (20d / 30d Momentum)
  - Volatility Penalty (hohe Vola = Abzug)
  - Sentiment Score  (News-Sentiment, wenn verfügbar)
  - Relative Strength (vs SPY)
  - Regime Alignment (Regime passt zur Aktion?)
  - Overbought Penalty (RSI > 70)
  - Oversold Bonus   (RSI < 30)

Schwellenwerte:
  > 75  → Strong BUY
  60-75 → BUY
  45-60 → HOLD
  30-45 → REDUCE
  < 30  → SELL
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import math

from logger import log


# ─────────────────────────────────────────────────────────────
# SCORE THRESHOLDS
# ─────────────────────────────────────────────────────────────

SCORE_STRONG_BUY = 75
SCORE_BUY        = 60
SCORE_HOLD       = 45
SCORE_REDUCE     = 30
# < 30 → SELL


# ─────────────────────────────────────────────────────────────
# COMPONENT WEIGHTS
# ─────────────────────────────────────────────────────────────

WEIGHTS = {
    "trend":            0.18,
    "momentum":         0.20,
    "macd":             0.10,
    "volatility":       0.10,  # negative component (penalty)
    "relative_strength":0.20,
    "rsi":              0.12,
    "regime":           0.10,
}


@dataclass
class ScoreBreakdown:
    """Vollständige Score-Aufschlüsselung für ein Asset."""
    ticker: str
    total_score: float           # 0-100
    signal: str                  # STRONG_BUY / BUY / HOLD / REDUCE / SELL
    recommended_action: str      # BUY / SELL / HOLD

    # Komponenten (jeweils 0-100 vor Gewichtung)
    trend_score: float = 0.0
    momentum_score: float = 0.0
    macd_score: float = 50.0
    volatility_penalty: float = 0.0
    relative_strength_score: float = 0.0
    rsi_score: float = 0.0
    regime_score: float = 0.0

    # Rohdaten für LLM-Prompt
    rsi: Optional[float] = None
    momentum_20d: Optional[float] = None
    sma20: Optional[float] = None
    sma50: Optional[float] = None
    sma_distance_pct: Optional[float] = None   # (price - SMA50) / SMA50 * 100
    volatility_annual: Optional[float] = None
    relative_strength: Optional[float] = None   # return vs SPY return (same period)
    current_price: Optional[float] = None
    current_alloc: float = 0.0
    # MACD / Bollinger
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_position: Optional[float] = None

    # Confidence (0-1) abgeleitet vom Score
    @property
    def confidence(self) -> float:
        return min(1.0, max(0.0, self.total_score / 100.0))

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "total_score": round(self.total_score, 1),
            "signal": self.signal,
            "recommended_action": self.recommended_action,
            "confidence": round(self.confidence, 3),
            "components": {
                "trend_score": round(self.trend_score, 1),
                "momentum_score": round(self.momentum_score, 1),
                "macd_score": round(self.macd_score, 1),
                "volatility_penalty": round(self.volatility_penalty, 1),
                "relative_strength_score": round(self.relative_strength_score, 1),
                "rsi_score": round(self.rsi_score, 1),
                "regime_score": round(self.regime_score, 1),
            },
            "metrics": {
                "rsi": self.rsi,
                "momentum_20d": self.momentum_20d,
                "sma20": self.sma20,
                "sma50": self.sma50,
                "sma_distance_pct": round(self.sma_distance_pct, 2) if self.sma_distance_pct is not None else None,
                "volatility_annual": self.volatility_annual,
                "relative_strength": round(self.relative_strength, 3) if self.relative_strength is not None else None,
                "macd_signal": self.macd_signal,
                "macd_histogram": self.macd_histogram,
                "bb_position": self.bb_position,
                "current_price": self.current_price,
                "current_alloc": round(self.current_alloc, 4),
            },
        }

    def to_llm_summary(self) -> str:
        """Kompakte Darstellung für LLM-Prompt."""
        parts = [f"{self.ticker}: Score={self.total_score:.0f} ({self.signal})"]
        if self.rsi is not None:
            parts.append(f"RSI={self.rsi:.0f}")
        if self.momentum_20d is not None:
            parts.append(f"Mom20d={self.momentum_20d:+.1f}%")
        if self.sma_distance_pct is not None:
            parts.append(f"SMA50dist={self.sma_distance_pct:+.1f}%")
        if self.volatility_annual is not None:
            parts.append(f"Vola={self.volatility_annual:.0f}%")
        if self.relative_strength is not None:
            parts.append(f"RS={self.relative_strength:.2f}")
        if self.macd_histogram is not None:
            parts.append(f"MACD_hist={self.macd_histogram:+.4f}")
        if self.bb_position is not None:
            parts.append(f"BBpos={self.bb_position:.2f}")
        if self.current_alloc > 0:
            parts.append(f"Alloc={self.current_alloc:.1%}")
        return " | ".join(parts)


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _sigmoid_score(x: float, center: float = 0.0, scale: float = 5.0) -> float:
    """Maps any real number to 0-100 via sigmoid, centered at `center`."""
    z = (x - center) / scale
    return 100.0 / (1.0 + math.exp(-z))


class ScoreEngine:
    """
    Deterministisches Scoring-System.

    Nimmt market_data (von DataCollector) + optionalen Regime-State
    und gibt für jeden Ticker einen ScoreBreakdown zurück.
    """

    def __init__(self, positions: Dict[str, Dict] = None, total_value: float = 100_000.0):
        self.positions = positions or {}
        self.total_value = total_value

    def score_all(
        self,
        market_data: Dict[str, Dict],
        regime_state=None,
        spy_return_20d: Optional[float] = None,
    ) -> Dict[str, ScoreBreakdown]:
        """Score alle Assets in market_data. Gibt {ticker: ScoreBreakdown}."""
        # SPY als Benchmark
        spy_data = market_data.get("SPY", {})
        spy_ret_20d = spy_return_20d or spy_data.get("return_30d")  # approximation

        results: Dict[str, ScoreBreakdown] = {}
        for ticker, data in market_data.items():
            try:
                sb = self.score_ticker(ticker, data, regime_state, spy_ret_20d)
                results[ticker] = sb
            except Exception as e:
                log.warning(f"[ScoreEngine] Fehler bei {ticker}: {e}")
        return results

    def score_ticker(
        self,
        ticker: str,
        data: Dict,
        regime_state=None,
        spy_return_20d: Optional[float] = None,
    ) -> ScoreBreakdown:
        """Score einen einzelnen Ticker."""
        # ── Rohdaten extrahieren ────────────────────────────────
        current_price = data.get("current_price") or 0.0
        rsi = data.get("rsi_14")
        return_7d  = data.get("return_7d") or 0.0
        return_20d = data.get("return_20d") or data.get("return_30d") or 0.0  # best available
        return_30d = data.get("return_30d") or 0.0
        volatility = data.get("volatility_annual_pct") or 20.0
        sma20 = data.get("sma_20") or data.get("sma_7")   # fallback
        sma50 = data.get("sma_50") or data.get("sma_30")   # fallback
        above_sma30 = data.get("above_sma_30")
        above_sma90 = data.get("above_sma_90")
        # MACD / Bollinger
        macd_signal = data.get("macd_signal")
        macd_hist = data.get("macd_histogram")
        bb_position = data.get("bb_position")

        # Current allocation
        pos = self.positions.get(ticker, {})
        current_alloc = pos.get("market_value", 0.0) / self.total_value if self.total_value > 0 else 0.0

        # SMA distance
        sma_distance_pct = None
        if current_price and sma50:
            sma_distance_pct = (current_price - sma50) / sma50 * 100

        # Relative strength vs SPY
        relative_strength = None
        if spy_return_20d is not None and spy_return_20d != 0:
            relative_strength = (1 + return_20d / 100) / (1 + spy_return_20d / 100)
        elif spy_return_20d == 0 and return_20d != 0:
            relative_strength = None  # undefined

        # ── Score-Komponenten ───────────────────────────────────

        # 1. Trend Score (0-100): Preis über SMAs → bullisch
        trend_score = self._trend_score(current_price, sma20, sma50, above_sma30, above_sma90, bb_position, sma_distance_pct)

        # 2. Momentum Score (0-100)
        momentum_score = self._momentum_score(return_20d, return_30d, return_7d)

        # 3. Volatility Penalty (0-100): hohe Vola → niedrig → Abzug
        volatility_score = self._volatility_score(volatility)

        # 4. RSI Score (0-100): normal RSI = hoch, überkauft/überverkauft = angepasst
        rsi_score = self._rsi_score(rsi)

        # 5. Relative Strength Score (0-100)
        rs_score = self._relative_strength_score(relative_strength)

        # 6. Regime Alignment Score (0-100)
        regime_score = self._regime_score(regime_state)

        # 7. MACD Score (0-100) — histogram-based evaluation
        macd_score = self._macd_score(macd_hist, current_price)

        # ── Gewichteter Gesamtscore ─────────────────────────────
        total = (
            WEIGHTS["trend"]             * trend_score
            + WEIGHTS["momentum"]        * momentum_score
            + WEIGHTS.get("macd", 0.0)    * macd_score
            - WEIGHTS["volatility"]      * (100 - volatility_score)   # Penalty
            + WEIGHTS["relative_strength"] * rs_score
            + WEIGHTS["rsi"]             * rsi_score
            + WEIGHTS["regime"]          * regime_score
        )
        total = _clamp(total, 0, 100)

        # ── Signal & Action ─────────────────────────────────────
        signal, recommended_action = self._classify(total)

        return ScoreBreakdown(
            ticker=ticker,
            total_score=round(total, 1),
            signal=signal,
            recommended_action=recommended_action,
            trend_score=round(trend_score, 1),
            momentum_score=round(momentum_score, 1),
            macd_score=round(macd_score, 1),
            macd_signal=round(macd_signal, 6) if macd_signal is not None else None,
            macd_histogram=round(macd_hist, 6) if macd_hist is not None else None,
            volatility_penalty=round(100 - volatility_score, 1),
            relative_strength_score=round(rs_score, 1),
            rsi_score=round(rsi_score, 1),
            regime_score=round(regime_score, 1),
            rsi=round(rsi, 1) if rsi is not None else None,
            momentum_20d=round(return_20d, 2),
            sma20=round(sma20, 2) if sma20 else None,
            sma50=round(sma50, 2) if sma50 else None,
            sma_distance_pct=round(sma_distance_pct, 2) if sma_distance_pct is not None else None,
            volatility_annual=round(volatility, 1),
            relative_strength=round(relative_strength, 3) if relative_strength is not None else None,
            current_price=round(current_price, 2),
            current_alloc=round(current_alloc, 4),
            bb_position=round(bb_position, 4) if bb_position is not None else None,
        )

    # ── Component Calculators ───────────────────────────────────────────────

    def _trend_score(
        self,
        price: float,
        sma20: Optional[float],
        sma50: Optional[float],
        above_sma30: Optional[bool],
        above_sma90: Optional[bool],
        bb_position: Optional[float],
        sma_dist: Optional[float],
    ) -> float:
        score = 50.0  # neutral base

        # Price vs SMA crossovers
        if price and sma20:
            if price > sma20:
                score += 10
            else:
                score -= 10

        if price and sma50:
            if price > sma50:
                score += 15
            else:
                score -= 15

        # SMA distance: slight positive = bullish, too much = overbought
        if sma_dist is not None:
            if 0 < sma_dist < 5:
                score += 12   # mildly above SMA50
            elif 5 <= sma_dist < 15:
                score += 8    # moderately above
            elif sma_dist >= 15:
                score -= 8    # too far above = overextended
            elif -5 <= sma_dist <= 0:
                score -= 4    # slightly below SMA50
            elif sma_dist < -5:
                score -= 12   # below SMA50

        # SMA90 alignment
        if above_sma90 is True:
            score += 8
        elif above_sma90 is False:
            score -= 8

        # Bollinger position influence (BB position: 0.0 bottom, 1.0 top)
        if bb_position is not None:
            try:
                if bb_position < 0.15:
                    score += 12
                elif 0.15 <= bb_position < 0.30:
                    score += 6
                elif 0.70 <= bb_position < 0.85:
                    score -= 5
                elif bb_position >= 0.85:
                    score -= 10
            except Exception:
                pass

        return _clamp(score)

    def _momentum_score(self, ret_20d: float, ret_30d: float, ret_7d: float) -> float:
        """Linear momentum mapping for broader differentiation.

        Very strong momentum (>15%) is rewarded with 80+ scores,
        while weak/negative momentum is pushed toward the lower bands.
        """
        weighted = 0.5 * ret_20d + 0.35 * ret_30d + 0.15 * ret_7d
        if weighted >= 20.0:
            return 100.0
        if weighted <= -10.0:
            return 15.0
        return _clamp(50.0 + weighted * 2.5)

    def _volatility_score(self, vol: float) -> float:
        """High volatility → low score (risk penalty). Returns 0-100 where 100 = low vol."""
        if vol < 10:
            return 95.0
        elif vol < 20:
            return _clamp(92.0 - (vol - 10) * 1.5)
        elif vol < 35:
            return _clamp(77.0 - (vol - 20) * 1.2)
        elif vol < 50:
            return _clamp(59.0 - (vol - 35) * 1.2)
        else:
            return _clamp(40.0 - (vol - 50) * 0.7)

    def _rsi_score(self, rsi: Optional[float]) -> float:
        """
        Normal range (40-60) = neutral (50).
        Oversold (<30) = slightly bullish bonus (65).
        Overbought (>70) = penalty (30).
        """
        if rsi is None:
            return 50.0
        if rsi < 20:
            return 70.0   # Very oversold – high reversal potential
        elif rsi < 30:
            return 65.0
        elif rsi < 40:
            return 58.0
        elif rsi <= 60:
            return 52.0   # Normal range
        elif rsi <= 70:
            return 42.0   # Getting stretched
        elif rsi <= 80:
            return 28.0   # Overbought penalty
        else:
            return 15.0   # Extremely overbought

    def _macd_score(self, macd_hist: Optional[float], price: float) -> float:
        """
        Histogram-based MACD evaluation.
        Maps the MACD histogram (price units) to a 0-100 score by
        normalizing to price and using sensible cutoffs. Positive
        histogram => bullish, negative => bearish.
        """
        if macd_hist is None or price is None or price <= 0:
            return 50.0

        # Normalize to percent of price
        hist_pct = (macd_hist / price) * 100.0

        # Heuristic thresholds (percent of price)
        if hist_pct >= 0.5:
            return 95.0
        if hist_pct >= 0.2:
            return 80.0
        if hist_pct >= 0.05:
            return 65.0
        if hist_pct > -0.05:
            return 55.0
        if hist_pct > -0.2:
            return 45.0
        if hist_pct > -0.5:
            return 30.0
        return 15.0

    def _relative_strength_score(self, rs: Optional[float]) -> float:
        """RS > 1.0 means outperforming SPY → bonus. RS < 1.0 → penalty."""
        if rs is None:
            return 50.0
        score = 50.0 + (rs - 1.0) * 120.0
        return _clamp(score)

    def _regime_score(self, regime_state) -> float:
        """
        BULL regime → slightly bullish bias (60).
        BEAR regime → defensive bias (35).
        SIDEWAYS    → neutral (50).
        None        → neutral (50).
        """
        if regime_state is None:
            return 50.0
        try:
            from market_regime import Regime
            r = getattr(regime_state, "regime", None)
            if r == Regime.BULL:
                return 70.0
            elif r == Regime.BEAR:
                return 30.0
            else:
                return 50.0
        except Exception:
            return 50.0

    def _classify(self, score: float) -> Tuple[str, str]:
        """Maps score to signal string and recommended action."""
        if score >= SCORE_STRONG_BUY:
            return "STRONG_BUY", "BUY"
        elif score >= SCORE_BUY:
            return "BUY", "BUY"
        elif score >= SCORE_HOLD:
            return "HOLD", "HOLD"
        elif score >= SCORE_REDUCE:
            return "REDUCE", "SELL"
        else:
            return "SELL", "SELL"


def rank_candidates(
    scores: Dict[str, ScoreBreakdown],
    min_score: float = SCORE_BUY,
    top_k: int = 8,
) -> List[ScoreBreakdown]:
    """
    Gibt die Top-K BUY-Kandidaten nach Score sortiert zurück.
    Nur Assets mit score >= min_score werden berücksichtigt.
    """
    candidates = [
        sb for sb in scores.values()
        if sb.total_score >= min_score and sb.recommended_action == "BUY"
    ]
    candidates.sort(key=lambda x: x.total_score, reverse=True)
    return candidates[:top_k]


class PortfolioOptimizer:
    """
    Portfolio-level optimizer: takes individual scores and constructs
    a diversified target allocation across all assets.

    Philosophy:
      - Think in terms of PORTFOLIO CONSTRUCTION, not single-stock signals.
      - Rank assets by risk-adjusted score (score / volatility).
      - Respect sector diversification limits.
      - Spread capital across 3-8 positions.
      - Never concentrate >20% in one asset, maintain >10% cash reserve.
      - Use correlation awareness to avoid stacking similar ETFs/stocks.
    """

    MAX_POSITION_PCT   = 0.20
    MIN_CASH_PCT       = 0.10
    MAX_SECTOR_PCT     = 0.45
    TARGET_MIN_POS     = 3
    TARGET_MAX_POS     = 8
    MIN_SCORE_FOR_BUY  = 50  

    def __init__(
        self,
        sector_map: Dict[str, str] = None,
        correlation_groups: List[List[str]] = None,
        risk_settings: Dict = None,
    ):
        self.sector_map = sector_map or {}
        self.correlation_groups = correlation_groups or []
        if risk_settings:
            self.MAX_POSITION_PCT  = risk_settings.get("max_position_pct",  self.MAX_POSITION_PCT)
            self.MIN_CASH_PCT      = risk_settings.get("min_cash_pct",      self.MIN_CASH_PCT)
            self.MAX_SECTOR_PCT    = risk_settings.get("max_sector_exposure",self.MAX_SECTOR_PCT)
            self.MIN_SCORE_FOR_BUY = risk_settings.get("min_buy_score",     self.MIN_SCORE_FOR_BUY)


    def optimize(
        self,
        scores: Dict[str, "ScoreBreakdown"],
        current_positions: Dict[str, Dict] = None,
        total_value: float = 100_000.0,
    ) -> "PortfolioAllocation":
        """
        Build an optimal target portfolio allocation using risk-adjusted scoring
        and diversification constraints. This is the core portfolio construction engine.
        """
        current_positions = current_positions or {}

        # 1. Compute risk-adjusted score = score / volatility (higher is better)
        candidates = []
        for ticker, sb in scores.items():
            vol = max(sb.volatility_annual or 20.0, 5.0)  # mind. 5% Vola
            risk_adj = sb.total_score / vol  # direktes Verhältnis
            candidates.append((ticker, sb, risk_adj))

        # 2. Sort by risk-adjusted score descending
        candidates.sort(key=lambda x: x[2], reverse=True)

        # 3. Apply correlation filtering: keep best from each correlated group
        filtered = self._filter_correlated(candidates)

        # 4. Apply sector cap
        filtered = self._filter_sector_cap(filtered)

        # 5. Limit to TARGET_MAX_POS (max 8)
        filtered = filtered[:self.TARGET_MAX_POS]

        # 6. Compute target allocations using risk-adjusted score weighting
        target_allocations = {}
        rationale = {}
        investable_budget = 1.0 - self.MIN_CASH_PCT

        if filtered:
            total_ra = sum(ra for _, _, ra in filtered)
            for ticker, sb, ra in filtered:
                raw_alloc = (ra / total_ra) * investable_budget if total_ra > 0 else investable_budget / len(filtered)
                # Cap per-position
                alloc = min(raw_alloc, self.MAX_POSITION_PCT)
                target_allocations[ticker] = round(alloc, 4)
                rationale[ticker] = (
                    f"RiskAdj={ra:.2f} | Score={sb.total_score:.0f} | Vola={sb.volatility_annual:.0f}% | "
                    f"Sector={self.sector_map.get(ticker, 'other')}"
                )

        # Normalize if total exceeds investable budget
        total_alloc = sum(target_allocations.values())
        if total_alloc > investable_budget:
            scale = investable_budget / total_alloc
            target_allocations = {t: round(a * scale, 4) for t, a in target_allocations.items()}

        # 7. Identify sells: current positions not in target and score < HOLD
        recommended_sells = []
        for ticker, pos in current_positions.items():
            if pos.get("market_value", 0) <= 0:
                continue
            sb = scores.get(ticker)
            if sb and sb.total_score < SCORE_HOLD:
                recommended_sells.append(ticker)
            elif ticker not in target_allocations and (sb is None or sb.total_score < self.MIN_SCORE_FOR_BUY):
                recommended_sells.append(ticker)

        cash_target = round(1.0 - sum(target_allocations.values()), 4)
        cash_target = max(cash_target, self.MIN_CASH_PCT)

        return PortfolioAllocation(
            target_allocations=target_allocations,
            recommended_sells=recommended_sells,
            cash_target=cash_target,
            rationale=rationale,
        )

    def _filter_correlated(self, candidates: List) -> List:
        """Keep only the highest-scoring asset from each correlated group."""
        selected = []
        blocked: set = set()
        for ticker, sb, ra in candidates:
            if ticker in blocked:
                continue
            selected.append((ticker, sb, ra))
            # Block lower-scoring members of same correlation group
            for group in self.correlation_groups:
                if ticker in group:
                    for peer in group:
                        if peer != ticker and peer not in blocked:
                            # Only block if current ticker significantly outperforms
                            peer_ra = next((r for t, _, r in candidates if t == peer), None)
                            if peer_ra is not None and ra > peer_ra * 1.05:
                                blocked.add(peer)
        return selected

    def _filter_sector_cap(self, candidates: List) -> List:
        """Prune candidates that would push sector above MAX_SECTOR_PCT."""
        sector_alloc: Dict[str, float] = {}
        result = []
        investable = 1.0 - self.MIN_CASH_PCT
        n = len(candidates)
        if n == 0:
            return result
        avg_alloc = min(investable / n, self.MAX_POSITION_PCT)

        for ticker, sb, ra in candidates:
            sector = self.sector_map.get(ticker, "other")
            current_sector_alloc = sector_alloc.get(sector, 0.0)
            if current_sector_alloc + avg_alloc > self.MAX_SECTOR_PCT + 0.01:
                continue  # would exceed sector cap – skip
            sector_alloc[sector] = current_sector_alloc + avg_alloc
            result.append((ticker, sb, ra))
        return result

    def build_prompt_section(
        self,
        scores: Dict[str, "ScoreBreakdown"],
        allocation: "PortfolioAllocation",
    ) -> str:
        """Build the portfolio-optimizer section for the LLM prompt."""
        lines = ["=== PORTFOLIO OPTIMIZER (Score-Weighted Risk-Adjusted Allocation) ==="]
        lines.append(f"Target positions: {len(allocation.target_allocations)} | Cash reserve: {allocation.cash_target:.1%}")
        lines.append("")
        lines.append(f"{'Ticker':<8} {'TargetAlloc':>11} {'Score':>7} {'RiskAdj':>9} {'Sector':<14} {'Rationale'}")
        lines.append("-" * 90)

        # BUY (target > 0)
        for ticker, alloc in sorted(allocation.target_allocations.items(), key=lambda x: -x[1]):
            sb = scores.get(ticker)
            score_str = f"{sb.total_score:.0f}" if sb else "n/a"
            rat = allocation.rationale.get(ticker, "")
            ra_str = ""
            for part in rat.split("|"):
                if "RiskAdj" in part:
                    ra_str = part.strip()
            sector = self.sector_map.get(ticker, "other")
            lines.append(f"{ticker:<8} {alloc:>10.1%}   {score_str:>7}   {ra_str:>9}   {sector:<14} →BUY")

        # SELLS
        for ticker in allocation.recommended_sells:
            sb = scores.get(ticker)
            score_str = f"{sb.total_score:.0f}" if sb else "n/a"
            lines.append(f"{ticker:<8} {'0%':>10}   {score_str:>7}   {'':>9}   {self.sector_map.get(ticker,'other'):<14} →SELL")

        lines.append("")
        lines.append(
            "PORTFOLIO CONSTRUCTION RULE: The LLM must allocate capital across the portfolio "
            "above — not pick isolated winners. Validate sector balance and correlation. "
            "Only deviate from suggested allocations if macro/news data provides strong justification."
        )
        return "\n".join(lines)


@dataclass
class PortfolioAllocation:
    """Output of PortfolioOptimizer.optimize()."""
    target_allocations: Dict[str, float]   # ticker → target fraction of portfolio
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
