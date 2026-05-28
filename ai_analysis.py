"""
AI Trading Bot - KI-Analyse (LLM als INTERPRETATION-LAYER)
==================================================
Neue Pipeline: ScoreEngine → PortfolioRebalancingEngine → LLM Interpretation → Risk Engine → Execution

Das LLM (DECOUPLED):
  - Empfängt deterministische Scores + Rebalancing-Vorschläge
  - Interpretiert und erklärt die Portfolio-Optimierung
  - DARF KEINE Scores adjustieren (llm_score_adj immer = 0)
  - Liefert STRUKTURIERTE JSON-Entscheidungen
  - Darf nicht von optimaler Allokation abweichen ohne starke Gründe
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
5. Nutze Korrelationsbewusstsein: Vermeide stark korrelierte Assets gleichzeitig.
6. Weise jedem Asset eine Zielallokation zu (target_allocation). Nicht nur BUY/SELL.
7. Die quantitativen Scores sind bereits berechnet (FINAL, nicht anpassbar). llm_score_adj muss 0 sein.
8. Du erhältst einen Portfolio-Optimizer-Vorschlag (Rebalancing Engine). Diese Allokation ist bereits kosten- und risikooptimiert.
9. Du darfst nur dann von diesem Vorschlag abweichen, wenn starke makroökonomische Nachrichten dies rechtfertigen.

JSON-FORMAT (PFLICHT):
{
  "decisions": [
    {
      "ticker": "NVDA",
      "action": "BUY",
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

WICHTIG: Denke in Portfolio-Allokationen. Jede Entscheidung muss zum Gesamtportfolio passen. Der Rebalancing-Vorschlag ist deine Baseline – weiche nur ab, wenn starke Gründe vorliegen."""


class AIAnalyzer:
    """
    KI-Analyse-Modul: LLM als INTERPRETATION-LAYER (nicht Score-Adjustment).
    Verwendet den Portfolio-Optimizer (oder Rebalancing Engine) als Basis.
    """

    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.model = OPENAI_MODEL
        self.client = None
        self.risk_settings = RISK_SETTINGS[ACTIVE_RISK_PROFILE]

        # Fallback: PortfolioOptimizer (wird nur verwendet, wenn keine Rebalancing Engine aktiv)
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
        portfolio_allocation=None,  # Kann PortfolioAllocation oder MockAllocation sein
    ) -> str:
        prompt_parts = []

        # 0. Feedback
        if journal_entries:
            feedback = self.build_feedback_section(journal_entries, market_data)
            if feedback:
                prompt_parts.append(feedback)

        # 1. Datum & Profil
        prompt_parts.append(f"DATUM: {datetime.now().strftime('%Y-%m-%d')}")
        prompt_parts.append(f"RISIKOPROFIL: {ACTIVE_RISK_PROFILE.value.upper()}")
        prompt_parts.append(f"MIN_BUY_SCORE: {self.risk_settings.get('min_buy_score', 60)}")

        # 2. Regime
        if regime_state is not None:
            prompt_parts.append("\n=== MARKT-REGIME ===")
            prompt_parts.append(f"Regime: {regime_state.label} (Konfidenz: {regime_state.confidence:.0%})")
            prompt_parts.append(f"Beschreibung: {regime_state.description}")
            if regime_state.vix is not None:
                prompt_parts.append(f"VIX: {regime_state.vix:.1f}")
            if regime_state.spy_return_20d is not None:
                prompt_parts.append(f"SPY 20d Return: {regime_state.spy_return_20d:+.1f}%")

            from market_regime import Regime
            if regime_state.regime == Regime.BEAR:
                prompt_parts.append("REGIME-ANWEISUNG: BEAR. Bevorzuge Defensive/Cash. Hohe Anforderungen für BUY.")
            elif regime_state.regime == Regime.SIDEWAYS:
                prompt_parts.append("REGIME-ANWEISUNG: SIDEWAYS. Reduziere Trades stark, nur >0.70 Konfidenz.")
            else:
                prompt_parts.append("REGIME-ANWEISUNG: BULL. Momentum-Strategien funktionieren. Risiko beachten.")

        # 3. Portfolio Optimizer / Rebalancing Engine Output (CORE)
        if portfolio_allocation is not None:
            prompt_parts.append("\n=== PORTFOLIO REBALANCING ENGINE VORSCHLAG (Baseline) ===")
            prompt_parts.append(f"Target positions: {len(portfolio_allocation.target_allocations)}")
            prompt_parts.append(f"Cash target: {portfolio_allocation.cash_target:.1%}")
            if hasattr(portfolio_allocation, 'rationale') and portfolio_allocation.rationale:
                # rationale kann ein String oder ein Dict sein
                rationale_text = portfolio_allocation.rationale if isinstance(portfolio_allocation.rationale, str) else str(portfolio_allocation.rationale)
                prompt_parts.append(f"Rationale: {rationale_text}")
            prompt_parts.append("")
            prompt_parts.append(f"{'Ticker':<8} {'TargetAlloc':>11}")
            prompt_parts.append("-" * 25)
            for ticker, alloc in sorted(portfolio_allocation.target_allocations.items(), key=lambda x: -x[1]):
                prompt_parts.append(f"{ticker:<8} {alloc:>10.1%}")
            prompt_parts.append("")
            prompt_parts.append("Diese Allokation ist bereits kosten- und risikooptimiert. Weiche nur bei starken Gründen ab.")
        else:
            # Fallback: eigenen Optimizer verwenden
            log.warning("Kein portfolio_allocation in build_prompt – verwende Fallback-Optimizer")
            if scores:
                current_positions = portfolio_summary.get("positions", {})
                total_val = portfolio_summary.get("total_value", 100_000)
                market_confidence = getattr(regime_state, 'confidence', 0.5) if regime_state else 0.5
                tech_exposure = 0.0
                for ticker, pos in current_positions.items():
                    sector = SECTOR_CLASSIFICATION.get(ticker, "other")
                    if sector == "tech":
                        tech_exposure += pos.get("market_value", 0)
                tech_exposure = tech_exposure / total_val if total_val > 0 else 0
                portfolio_risk_high = tech_exposure > 0.5
                fallback_allocation = self.portfolio_optimizer.optimize(
                    scores=scores,
                    current_positions=current_positions,
                    total_value=total_val,
                    regime_state=regime_state,
                    market_confidence=market_confidence,
                    portfolio_risk_high=portfolio_risk_high,
                )
                prompt_parts.append(self.portfolio_optimizer.build_prompt_section(scores, fallback_allocation))

        # 4. Quantitative Scores
        if scores:
            prompt_parts.append("\n" + build_score_prompt_section(scores))

        # 5. Portfolio-Status
        prompt_parts.append("\n=== AKTUELLES PORTFOLIO ===")
        prompt_parts.append(f"Gesamtwert: ${portfolio_summary.get('total_value', 0):,.0f}")
        prompt_parts.append(f"Cash: ${portfolio_summary.get('cash', 0):,.0f} ({portfolio_summary.get('cash_pct', 0):.1f}%)")
        prompt_parts.append(f"P&L gesamt: {portfolio_summary.get('pnl_pct', 0):+.2f}%")

        positions = portfolio_summary.get("positions", {})
        if positions:
            prompt_parts.append("Positionen:")
            total_val = portfolio_summary.get("total_value", 1)
            for ticker, pos in positions.items():
                alloc = pos.get("market_value", 0) / total_val
                prompt_parts.append(
                    f"  {ticker}: {alloc:.1%} | P&L: {pos.get('unrealized_pnl_pct', 0):+.1f}%"
                )

        # 5b. Top Candidates (Referenz)
        candidates = rank_candidates(scores, min_score=self.risk_settings.get("min_buy_score", 60), top_k=SCORE_TOP_K_CANDIDATES)
        if candidates:
            prompt_parts.append("\n=== HOCH BEWERTETE ASSETS (Referenz) ===")
            for c in candidates:
                prompt_parts.append(f"  {c.to_llm_summary()}")

        # 6. News
        prompt_parts.append("\n=== AKTUELLE FINANZNACHRICHTEN ===")
        prompt_parts.append(news_text[:2500])

        # 7. Aufgabe
        tickers_str = ", ".join(watchlist[:15])
        prompt_parts.append("\n=== AUFGABE ===")
        prompt_parts.append(
            f"Erstelle ein PORTFOLIO basierend auf dem obigen Rebalancing-Vorschlag.\n"
            f"Du darfst die vorgeschlagenen Allokationen nur aus wichtigen Gründen anpassen (z.B. starke Nachrichten).\n"
            f"Denke in Gesamtallokationen. Antworte NUR mit JSON."
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
        portfolio_allocation=None,  # Wird von main.py übergeben (MockAllocation oder PortfolioAllocation)
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
        top = sorted(scores.values(), key=lambda x: x.total_score, reverse=True)[:5]
        for s in top:
            log.debug(f"  {s.ticker}: {s.total_score:.0f} ({s.signal}) | RSI={s.rsi} Mom={s.momentum_20d}")

        # 2. Falls kein portfolio_allocation übergeben wurde, berechne Fallback
        if portfolio_allocation is None:
            log.warning("Kein portfolio_allocation in analyze – verwende Fallback-Optimizer")
            # Fallback mit Regime-Parametern
            market_confidence = getattr(regime_state, 'confidence', 0.5) if regime_state else 0.5
            from config import SECTOR_CLASSIFICATION
            tech_exposure = 0.0
            for ticker, pos in positions.items():
                sector = SECTOR_CLASSIFICATION.get(ticker, "other")
                if sector == "tech":
                    tech_exposure += pos.get("market_value", 0)
            tech_exposure = tech_exposure / total_value if total_value > 0 else 0
            portfolio_risk_high = tech_exposure > 0.5
            portfolio_allocation = self.portfolio_optimizer.optimize(
                scores=scores,
                current_positions=positions,
                total_value=total_value,
                regime_state=regime_state,
                market_confidence=market_confidence,
                portfolio_risk_high=portfolio_risk_high,
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

            for d in result["decisions"]:
                ticker = d.get("ticker")
                if ticker in scores:
                    sb = scores[ticker]
                    d.setdefault("quant_score", sb.total_score)
                    if not d.get("reasoning"):
                        d["reasoning"] = self._build_reasoning_from_score(sb)

            log.info(
                f"KI-Analyse: {len(result.get('decisions', []))} Entscheidungen | "
                f"{result.get('market_outlook', '')[:80]}"
            )
            result["scores"] = {t: sb.to_dict() for t, sb in scores.items()}
            return result

        except Exception as e:
            log.error(f"OpenAI API Fehler: {e}")
            log.warning("Fallback auf Score-basierte Analyse.")
            return self._score_based_fallback(scores, market_data, portfolio_summary, watchlist, portfolio_allocation)

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
                # 🔥 FIX: quant_score kann None sein
                quant_score_raw = d.get("quant_score")
                if quant_score_raw is None:
                    quant_score_raw = 0
                try:
                    quant_score = float(quant_score_raw)
                except (TypeError, ValueError):
                    quant_score = 0.0

                if quant_score == 0 and ticker in scores:
                    quant_score = scores[ticker].total_score
                    d["quant_score"] = quant_score

                d["llm_score_adj"] = 0.0
                effective_score = quant_score

                # Guardrail: BUY blockieren wenn Score zu niedrig – AUSSER wenn vom PortfolioOptimizer/Rebalancing vorgeschlagen
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
        Portfolio-aware fallback ohne LLM. Verwendet PortfolioOptimizer-Allokation direkt.
        """
        log.info("Score-basierter Fallback: nutzt PortfolioOptimizer-Allokation direkt")
        positions = portfolio_summary.get("positions", {})
        total_value = portfolio_summary.get("total_value", 100_000)

        if portfolio_allocation is None:
            # Fallback: Optimizer ohne Regime-Parameter aufrufen
            portfolio_allocation = self.portfolio_optimizer.optimize(
                scores=scores,
                current_positions=positions,
                total_value=total_value,
                regime_state=None,
                market_confidence=0.5,
                portfolio_risk_high=False,
            )

        decisions = []
        all_tickers = set(watchlist) | set(positions.keys())
        for ticker in all_tickers:
            sb = scores.get(ticker)
            if not sb:
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
                # 🔥 FIX: rationale kann ein String oder ein Dict sein
                rationale_value = ""
                if hasattr(portfolio_allocation, 'rationale'):
                    if isinstance(portfolio_allocation.rationale, dict):
                        rationale_value = portfolio_allocation.rationale.get(ticker, "")
                    elif isinstance(portfolio_allocation.rationale, str):
                        rationale_value = portfolio_allocation.rationale
                reason = f"Portfolio-optimized: {rationale_value}" if rationale_value else "Portfolio-optimized (Rebalancing Engine)"
            elif ticker in getattr(portfolio_allocation, 'recommended_sells', []):
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
            "portfolio_rationale": getattr(portfolio_allocation, 'summary', lambda: "Fallback allocation")(),
            "market_outlook": "Fallback: portfolio-optimized score-based analysis (no LLM).",
            "risk_assessment": "Using PortfolioOptimizer allocation (risk-adjusted, diversified).",
            "scores": {t: sb.to_dict() for t, sb in scores.items()},
        }
