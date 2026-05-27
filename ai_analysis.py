"""
AI Trading Bot - KI-Analyse (LLM als INTERPRETATION-LAYER)
==================================================
Neue Pipeline:
  Indicators → ScoreEngine (IMMUTABLE) → LLM Interpretation → Risk Engine → Execution

Das LLM (DECOUPLED):
  - Empfängt deterministische Scores für jeden Ticker (IMMUTABLE)
  - Interpretiert und erklärt die Scores (context nur)
  - DARF KEINE Scores adjustieren (llm_score_adj immer = 0)
  - Liefert STRUKTURIERTE JSON-Entscheidungen mit Metrik-Feldern
  - Darf keine komplett irrationalen Entscheidungen erzeugen
  - Narrativer Text ist OPTIONAL, Metriken sind PFLICHT

Ausgabeformat je Entscheidung:
{
  "ticker": "NVDA",
  "action": "BUY",
  "target_allocation": 0.12,
  "confidence": 0.82,
  "quant_score": 78,       <- deterministischer Score (UNVERÄNDLICH)
  "llm_score_adj": 0,      <- LLM-Adjustierungen sind NICHT MEHR AKTIVIERT
  "reasoning": {
    "momentum_20d": 14.2,
    "relative_strength": 1.31,
    "rsi": 64,
    "sma_distance_pct": 5.2,
    "volatility": 28.1,
    "regime": "BULL",
    "current_alloc": 0.08
  },
  "reason": "Optional kurze narrative Begründung (max 20 Wörter)"
}
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from logger import log
from config import (
    OPENAI_API_KEY, OPENAI_MODEL,
    ACTIVE_RISK_PROFILE, RISK_SETTINGS,
    SECTOR_CLASSIFICATION, FACTOR_CLASSIFICATION, CORRELATION_GROUPS,
    LLM_SCORE_OVERRIDE_LIMIT, SCORE_TOP_K_CANDIDATES,
)
from utils import ensure_decision_ids
from score_engine import (
    ScoreEngine, ScoreBreakdown, build_score_prompt_section,
    rank_candidates, SCORE_BUY,
    PortfolioOptimizer, PortfolioAllocation,
)

try:
    from openai import OpenAI
except ImportError:
    log.error("openai nicht installiert. Führe aus: pip install openai")
    raise


SYSTEM_PROMPT = """Du bist ein quantitativer Portfolio-Manager. Deine Aufgabe ist es, ein optimal diversifiziertes Portfolio zu konstruieren – nicht einzelne Aktien zu picken.

PORTFOLIO-KONSTRUKTIONSPRINZIPIEN:
1. Bewerte ALLE Assets relativ zueinander (nicht isoliert).
2. Das Ziel ist ein Portfolio mit 3–8 Positionen, das risikoadjustierte Rendite maximiert.
3. Keine Überkonzentration: max. 20% pro Asset, max. 45% pro Sektor.
4. Mindestens 10% Cash-Reserve.
5. Nutze Korrelationsbewusstsein: Vermeide stark korrelierte Assets gleichzeitig (z.B. QQQ und XLK).
6. Weise jedem Asset eine Zielallokation zu (target_allocation). Nicht nur BUY/SELL.
7. Die quantitativen Scores sind bereits berechnet (FINAL, nicht anpassbar). llm_score_adj muss 0 sein.
8. Du darfst nur dann von der vorgeschlagenen Portfolio-Optimizer-Allokation abweichen, wenn starke makroökonomische Nachrichten dies rechtfertigen.

JSON-FORMAT (PFLICHT):
{
  "decisions": [
    {
      "ticker": "NVDA",
      "action": "BUY",   // oder SELL, HOLD
      "target_allocation": 0.12,
      "confidence": 0.82,
      "quant_score": 78,
      "llm_score_adj": 0,
      "reasoning": {
        "momentum_20d": 14.2,
        "relative_strength": 1.31,
        "rsi": 64,
        "sma_distance_pct": 5.2,
        "volatility": 28.1,
        "regime": "BULL",
        "current_alloc": 0.08,
        "portfolio_role": "Momentum leader, diversifies tech exposure"
      },
      "reason": "Top risk-adjusted score; anchors tech allocation at 12%"
    }
  ],
  "portfolio_rationale": "1-2 sentences: why THIS portfolio mix makes sense as a whole",
  "market_outlook": "Brief market outlook",
  "risk_assessment": "Portfolio-level risk commentary",
  "feedback_learnings": "What you learned from past decisions"
}

WICHTIG: Denke in Portfolio-Allokationen."""


class AIAnalyzer:
    """
    KI-Analyse-Modul: LLM als INTERPRETATION-LAYER (nicht Score-Adjustment).
    """

    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.model = OPENAI_MODEL
        self.client = None
        self.risk_settings = RISK_SETTINGS[ACTIVE_RISK_PROFILE]

        # Portfolio-level optimizer (score-weighted, diversified allocation)
        self.portfolio_optimizer = PortfolioOptimizer(
            sector_map=SECTOR_CLASSIFICATION,
            correlation_groups=CORRELATION_GROUPS,
            risk_settings=self.risk_settings,
        )

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            log.info(f"OpenAI Client initialisiert (Modell: {self.model})")
        else:
            log.warning("Kein OpenAI API Key! Nutze regelbasierte Fallback-Analyse.")

    def build_feedback_section(self, journal_entries: List[Dict], market_data: Dict[str, Dict]) -> str:
        if not journal_entries:
            return ""

        lines = ["=== FEEDBACK: DEINE LETZTEN ENTSCHEIDUNGEN & OUTCOME ===", ""]
        outcome_count = 0

        for entry in journal_entries[-10:]:
            date = entry.get("date", "?")
            decisions = entry.get("ai_decisions", [])
            if not decisions:
                continue

            actionable = [d for d in decisions if d.get("action") in ("BUY", "SELL")]
            if not actionable:
                continue

            lines.append(f"Run vom {date}:")
            for d in actionable[:6]:
                ticker = d.get("ticker", "?")
                action = d.get("action", "?")
                confidence = d.get("confidence_pct", 0)
                reason = d.get("reason", "")
                quant_score = d.get("quant_score", "?")

                outcome_str = ""
                current_data = market_data.get(ticker, {})
                return_since = current_data.get("return_20d") or current_data.get("return_7d")

                if return_since is not None:
                    outcome_count += 1
                    if action == "BUY":
                        verdict = "✅ RICHTIG" if return_since > 1.0 else ("❌ FALSCH" if return_since < -1.0 else "➡ NEUTRAL")
                    else:
                        verdict = "✅ RICHTIG" if return_since < -1.0 else ("❌ FALSCH" if return_since > 1.0 else "➡ NEUTRAL")
                    outcome_str = f" | 20d seitdem: {return_since:+.1f}% → {verdict}"

                lines.append(
                    f"  {action} {ticker} | Score: {quant_score} | "
                    f"Konfidenz: {confidence:.0f}% | {reason[:50]}{outcome_str}"
                )
            lines.append("")

        if outcome_count == 0:
            return ""

        lines.append("→ Lerne daraus: Welche Scores haben sich bewahrheitet?")
        lines.append("")
        return "\n".join(lines)

    def build_prompt(
        self,
        portfolio_summary: Dict,
        market_data: Dict[str, Dict],
        news_text: str,
        watchlist: List[str],
        scores: Dict[str, ScoreBreakdown],
        journal_entries: List[Dict] = None,
        regime_state=None,
        portfolio_allocation=None,  # PortfolioAllocation from PortfolioOptimizer
    ) -> str:
        prompt_parts = []

        # 0. Feedback (unverändert)
        if journal_entries:
            feedback = self.build_feedback_section(journal_entries, market_data)
            if feedback:
                prompt_parts.append(feedback)

        # 1. Datum & Profil
        prompt_parts.append(f"DATUM: {datetime.now().strftime('%Y-%m-%d')}")
        prompt_parts.append(f"RISIKOPROFIL: {ACTIVE_RISK_PROFILE.value.upper()}")
        
        # 2. Markt-Regime
        if regime_state is not None:
            prompt_parts.append("\n=== MARKT-REGIME ===")
            prompt_parts.append(f"Regime: {regime_state.label} (Konfidenz: {regime_state.confidence:.0%})")

        # 3. Portfolio Optimizer Output (CORE – zuerst!)
        if portfolio_allocation is not None:
            prompt_parts.append(
                "\n" + self.portfolio_optimizer.build_prompt_section(scores, portfolio_allocation)
            )
        else:
            # Fallback: selbst berechnen
            portfolio_allocation = self.portfolio_optimizer.optimize(
                scores=scores,
                current_positions=portfolio_summary.get("positions", {}),
                total_value=portfolio_summary.get("total_value", 100_000),
            )
            prompt_parts.append(
                "\n" + self.portfolio_optimizer.build_prompt_section(scores, portfolio_allocation)
            )

        # 4. Quantitative Scores
        if scores:
            prompt_parts.append("\n" + build_score_prompt_section(scores))

        # 5. Portfolio-Status
        prompt_parts.append("\n=== AKTUELLES PORTFOLIO ===")
        prompt_parts.append(f"Gesamtwert: ${portfolio_summary.get('total_value', 0):,.0f}")
        prompt_parts.append(f"Cash: {portfolio_summary.get('cash_pct', 0):.1f}%")
        
        # 6. Top-Kandidaten (nur als Referenz, nicht als Aufforderung zum Picken)
        candidates = rank_candidates(scores, min_score=self.risk_settings.get("min_buy_score", 60), top_k=5)
        if candidates:
            prompt_parts.append("\n=== HOCH BEWERTETE ASSETS (Referenz) ===")
            for c in candidates:
                prompt_parts.append(f"  {c.to_llm_summary()}")

        # 7. News
        prompt_parts.append("\n=== AKTUELLE FINANZNACHRICHTEN ===")
        prompt_parts.append(news_text[:2500])

        # 8. Aufgabe – Betonung der Portfolio-Optimierung
        prompt_parts.append("\n=== AUFGABE ===")
        prompt_parts.append(
            "Erstelle ein PORTFOLIO basierend auf dem obigen Portfolio-Optimizer-Vorschlag.\n"
            "Du darfst die vorgeschlagenen Allokationen nur aus wichtigen Gründen anpassen (z.B. starke Nachrichten).\n"
            "Denke in Gesamtallokationen, nicht in einzelnen Trades. Antworte NUR mit JSON."
        )

        return "\n".join(prompt_parts)

    def analyze(
        self,
        portfolio_summary: Dict,
        market_data: Dict[str, Dict],
        news_text: str,
        watchlist: List[str],
        journal_entries: List[Dict] = None,
        regime_state=None,
    ) -> Dict:
        """
        Hauptfunktion: Score Engine → LLM Meta-Layer → validierte Entscheidungen.
        """
        # 1. Quantitative Scores berechnen (deterministisch)
        positions = portfolio_summary.get("positions", {})
        total_value = portfolio_summary.get("total_value", 100_000)
        score_engine = ScoreEngine(positions=positions, total_value=total_value)

        spy_return = market_data.get("SPY", {}).get("return_20d")
        scores = score_engine.score_all(market_data, regime_state, spy_return)

        log.info(f"ScoreEngine: {len(scores)} Assets bewertet")
        # Log top 5
        top = sorted(scores.values(), key=lambda x: x.total_score, reverse=True)[:5]
        for s in top:
            log.debug(f"  {s.ticker}: {s.total_score:.0f} ({s.signal}) | RSI={s.rsi} Mom={s.momentum_20d}")

        # 2. Portfolio-level optimization (score-weighted, diversified)
        portfolio_allocation = self.portfolio_optimizer.optimize(
            scores=scores,
            current_positions=positions,
            total_value=total_value,
        )
        log.info(
            f"PortfolioOptimizer: {len(portfolio_allocation.target_allocations)} BUY targets | "
            f"{len(portfolio_allocation.recommended_sells)} sells | cash={portfolio_allocation.cash_target:.1%}"
        )

        if not self.client:
            log.warning("Kein OpenAI Client – nutze Score-basierte Fallback-Analyse.")
            return self._score_based_fallback(scores, market_data, portfolio_summary, watchlist, portfolio_allocation)

        prompt = self.build_prompt(
            portfolio_summary, market_data, news_text, watchlist, scores,
            journal_entries, regime_state=regime_state,
            portfolio_allocation=portfolio_allocation,
        )

        log.info(f"Sende Anfrage an OpenAI ({self.model})...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0,
                max_tokens=2500,
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content
            result = self._parse_response(raw_text, scores, portfolio_allocation)

            # Post-filter: nur erlaubte Ticker
            allowed = set(watchlist)
            before = len(result.get("decisions", []))
            result["decisions"] = [d for d in result.get("decisions", []) if d["ticker"] in allowed]
            filtered = before - len(result["decisions"])
            if filtered > 0:
                log.warning(f"{filtered} KI-Entscheidungen für nicht-erlaubte Ticker gefiltert.")

            # Inject scores into decisions für Logging/Journal
            for d in result["decisions"]:
                ticker = d.get("ticker")
                if ticker in scores:
                    sb = scores[ticker]
                    d.setdefault("quant_score", sb.total_score)
                    # Ensure reasoning has real metrics
                    if not d.get("reasoning"):
                        d["reasoning"] = self._build_reasoning_from_score(sb)

            log.info(
                f"KI-Analyse: {len(result.get('decisions', []))} Entscheidungen | "
                f"{result.get('market_outlook', '')[:80]}"
            )
            # Attach scores dict to result for downstream use
            result["scores"] = {t: sb.to_dict() for t, sb in scores.items()}
            return result

        except Exception as e:
            log.error(f"OpenAI API Fehler: {e}")
            log.warning("Fallback auf Score-basierte Analyse.")
            return self._score_based_fallback(scores, market_data, portfolio_summary, watchlist)

    def _build_reasoning_from_score(self, sb: ScoreBreakdown) -> Dict:
        """Erstellt reasoning-Dict aus ScoreBreakdown."""
        return {
            "momentum_20d": sb.momentum_20d,
            "relative_strength": sb.relative_strength,
            "rsi": sb.rsi,
            "sma_distance_pct": sb.sma_distance_pct,
            "volatility": sb.volatility_annual,
            "current_alloc": sb.current_alloc,
        }

    def _parse_response(self, raw_text: str, scores: Dict[str, ScoreBreakdown], portfolio_allocation=None) -> Dict:
        """Parst und validiert die JSON-Antwort. Enforces score-based guardrails."""
        try:
            clean = re.sub(r"```json\s*|\s*```", "", raw_text).strip()
            data = json.loads(clean)

            if "decisions" not in data:
                raise ValueError("Kein 'decisions' Feld")

            valid_decisions = []
            min_buy_score = self.risk_settings.get("min_buy_score", 60)

            # Prepare optimizer ticker set for override
            optimizer_tickers = set()
            if portfolio_allocation is not None:
                optimizer_tickers = set(portfolio_allocation.target_allocations.keys())

            for d in data["decisions"]:
                missing = [k for k in ("ticker", "action", "target_allocation", "confidence") if k not in d]
                if missing:
                    log.warning(f"Entscheidung übersprungen (fehlende Felder {missing}): {d}")
                    continue

                d["ticker"] = str(d["ticker"]).strip().upper()
                d["action"] = str(d["action"]).upper()
                if d["action"] not in ("BUY", "SELL", "HOLD"):
                    d["action"] = "HOLD"
                d["target_allocation"] = max(0.0, min(1.0, float(d["target_allocation"])))
                d["confidence"] = max(0.0, min(1.0, float(d["confidence"])))
                d["reason"] = str(d.get("reason", ""))[:200]

                ticker = d["ticker"]
                quant_score = float(d.get("quant_score", 0))
                
                if quant_score == 0 and ticker in scores:
                    quant_score = scores[ticker].total_score
                    d["quant_score"] = quant_score
                
                d["llm_score_adj"] = 0.0
                effective_score = quant_score
                
                # Guardrail: BUY blockieren wenn Score zu niedrig – AUSSER wenn vom PortfolioOptimizer vorgeschlagen
                is_optimizer_suggestion = ticker in optimizer_tickers
                if d["action"] == "BUY" and effective_score < min_buy_score and not is_optimizer_suggestion:
                    log.warning(
                        f"[GUARDRAIL] {ticker}: BUY blockiert "
                        f"(effective_score={effective_score:.1f} < min={min_buy_score})"
                    )
                    d["action"] = "HOLD"
                    d["reason"] = f"{d['reason']} [GUARDRAIL: score too low]"
                elif d["action"] == "BUY" and effective_score < min_buy_score and is_optimizer_suggestion:
                    log.info(
                        f"[GUARDRAIL OVERRIDE] {ticker}: PortfolioOptimizer BUY akzeptiert "
                        f"trotz Score {effective_score:.1f} < {min_buy_score}"
                    )

                if d["action"] == "SELL" and d.get("target_allocation", 0) > 0.5:
                    d["target_allocation"] = 0.0
                    log.warning(f"{ticker}: SELL mit hoher target_allocation → korrigiert auf 0")

                if "reasoning" not in d or not isinstance(d.get("reasoning"), dict):
                    if ticker in scores:
                        d["reasoning"] = self._build_reasoning_from_score(scores[ticker])
                    else:
                        d["reasoning"] = {}

                valid_decisions.append(d)

            data["decisions"] = ensure_decision_ids(valid_decisions)
            data["market_outlook"] = str(data.get("market_outlook", "No outlook"))[:500]
            data["risk_assessment"] = str(data.get("risk_assessment", ""))[:500]
            data["feedback_learnings"] = str(data.get("feedback_learnings", ""))[:300]
            data["portfolio_rationale"] = str(data.get("portfolio_rationale", ""))[:500]
            return data

        except json.JSONDecodeError as e:
            log.error(f"JSON-Parse-Fehler: {e}")
            return {"decisions": [], "market_outlook": "Parse error", "risk_assessment": ""}

    def _score_based_fallback(
        self,
        scores: Dict[str, ScoreBreakdown],
        market_data: Dict[str, Dict],
        portfolio_summary: Dict,
        watchlist: List[str],
        portfolio_allocation=None,
    ) -> Dict:
        """
        Portfolio-aware fallback without LLM. Uses PortfolioOptimizer allocation directly.
        """
        log.info("Score-basierter Fallback: Nutze PortfolioOptimizer-Allokation direkt.")
        positions = portfolio_summary.get("positions", {})
        total_value = portfolio_summary.get("total_value", 100_000)

        if portfolio_allocation is None:
            portfolio_allocation = self.portfolio_optimizer.optimize(
                scores=scores,
                current_positions=positions,
                total_value=total_value,
            )

        decisions = []
        # Für jeden Ticker in der Watchlist (oder in Scores) Entscheidung treffen
        all_tickers = set(watchlist) | set(positions.keys())
        for ticker in all_tickers:
            sb = scores.get(ticker)
            if not sb:
                # Kein Score, aber Position? Dann HOLD mit aktueller Allokation
                current_alloc = positions.get(ticker, {}).get("market_value", 0) / total_value if total_value else 0
                decisions.append({
                    "ticker": ticker,
                    "action": "HOLD",
                    "target_allocation": current_alloc,
                    "confidence": 0.5,
                    "quant_score": 0,
                    "llm_score_adj": 0,
                    "reasoning": {},
                    "reason": "No score available – holding position",
                })
                continue

            target_alloc = portfolio_allocation.target_allocations.get(ticker)
            if target_alloc is not None:
                action = "BUY"
                reason = f"Portfolio-optimized: {portfolio_allocation.rationale.get(ticker, '')}"
            elif ticker in portfolio_allocation.recommended_sells:
                action = "SELL"
                target_alloc = 0.0
                reason = f"Score below hold threshold ({sb.total_score:.0f})"
            else:
                action = "HOLD"
                target_alloc = sb.current_alloc
                reason = f"Score-based HOLD ({sb.total_score:.0f})"

            decisions.append({
                "ticker": ticker,
                "action": action,
                "target_allocation": target_alloc,
                "confidence": sb.confidence,
                "quant_score": sb.total_score,
                "llm_score_adj": 0,
                "reasoning": self._build_reasoning_from_score(sb),
                "reason": reason,
            })

        return {
            "decisions": decisions,
            "portfolio_rationale": portfolio_allocation.summary(),
            "market_outlook": "Fallback: portfolio-optimized score-based analysis (no LLM).",
            "risk_assessment": "Using PortfolioOptimizer allocation (risk-adjusted, diversified).",
            "scores": {t: sb.to_dict() for t, sb in scores.items()},
        }
