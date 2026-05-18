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
from score_engine import ScoreEngine, ScoreBreakdown, build_score_prompt_section, rank_candidates, SCORE_BUY

try:
    from openai import OpenAI
except ImportError:
    log.error("openai nicht installiert. Führe aus: pip install openai")
    raise


SYSTEM_PROMPT = """Du bist ein quantitativer Portfolio-Manager. Du erhältst deterministische Metriken und Scores für alle Assets.

DEINE ROLLE: Interpretation-Layer (NICHT Score-Adjustment)
  - Die quantitativen Scores sind bereits berechnet (deterministisch) und FINAL
  - DEIN JOB: Erklären, WARUM diese Scores aktuell sind
  - DARF NICHT: Scores adjustieren (llm_score_adj muss IMMER 0 sein)
  - DARF NICHT: Irrationale Entscheidungen erzeugen (z.B. Score=20 als STRONG_BUY)
  - Narrative Begründungen sind optional, Metrik-Felder sind PFLICHT

ENTSCHEIDUNGSREGELN (Score-based, NO LLM adjustment):
  - BUY: quant_score >= 60 (Minimum)
  - SELL: quant_score < 45
  - HOLD: alles dazwischen
  - Keine extremen Umschichtungen ohne klare Datenbasis
  - Sektordiversifikation beachten (nicht zu viel Tech)
  - Antwort NUR als JSON-Objekt, kein erklärender Text

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
        "current_alloc": 0.08
      },
      "reason": "Strong momentum, RS > 1.3, confirmed by score"
    }
  ],
  "market_outlook": "Brief market outlook (1-2 sentences)",
  "risk_assessment": "Brief risk assessment",
  "feedback_learnings": "What you learned from past decisions"
}"""


class AIAnalyzer:
    """
    KI-Analyse-Modul: LLM als INTERPRETATION-LAYER (nicht Score-Adjustment).
    """

    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.model = OPENAI_MODEL
        self.client = None
        self.risk_settings = RISK_SETTINGS[ACTIVE_RISK_PROFILE]

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

        # 3. Quantitative Scores (Kern des neuen Systems)
        if scores:
            prompt_parts.append("\n" + build_score_prompt_section(scores))

        # 4. Portfolio-Status
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

        # 4b. Portfolio Risk Context (Phase 4 Production Stability)
        prompt_parts.append("\n=== PORTFOLIO RISIKEN & KONZENTRATION ===")
        if positions:
            total_val = portfolio_summary.get("total_value", 1)
            _profile = RISK_SETTINGS[ACTIVE_RISK_PROFILE]
            max_pos_pct = _profile.get("max_position_pct", 0.20)
            max_sector_pct = _profile.get("max_sector_exposure", 0.45)

            # Position concentration warnings
            overweight = [
                f"{t}: {pos.get('market_value',0)/total_val:.1%}"
                for t, pos in positions.items()
                if pos.get("market_value", 0) / total_val > max_pos_pct * 0.85
            ]
            if overweight:
                prompt_parts.append(f"⚠ Near position cap ({max_pos_pct:.0%}): {', '.join(overweight)}")

            # Sector concentration
            sector_vals: Dict[str, float] = {}
            for t, pos in positions.items():
                sector = SECTOR_CLASSIFICATION.get(t, "other")
                sector_vals[sector] = sector_vals.get(sector, 0) + pos.get("market_value", 0)
            for sector, val in sorted(sector_vals.items(), key=lambda x: -x[1]):
                pct = val / total_val
                flag = "⚠ " if pct > max_sector_pct * 0.85 else "  "
                prompt_parts.append(f"{flag}{sector}: {pct:.1%}")
        else:
            prompt_parts.append("  Keine offenen Positionen – Cash-only")

        # 4c. Regime & Stability Status
        if regime_state is not None:
            prompt_parts.append(f"\nRegime-Stabilität: {getattr(regime_state, 'stability', 'unbekannt')}")
            prompt_parts.append(
                f"Regime seit letztem Signal: {getattr(regime_state, 'label', '?')} "
                f"(Konfidenz: {getattr(regime_state, 'confidence', 0):.0%})"
            )

        # 5. Top Candidates summary (for LLM to focus on)
        candidates = rank_candidates(scores, min_score=self.risk_settings.get("min_buy_score", 60), top_k=SCORE_TOP_K_CANDIDATES)
        if candidates:
            prompt_parts.append("\n=== TOP BUY-KANDIDATEN (nach quantitativem Score) ===")
            for c in candidates:
                prompt_parts.append(f"  {c.to_llm_summary()}")

        # 6. News
        prompt_parts.append("\n=== AKTUELLE FINANZNACHRICHTEN ===")
        prompt_parts.append(news_text[:2500])

        # 7. Aufgabe
        tickers_str = ", ".join(watchlist[:15])
        prompt_parts.append("\n=== AUFGABE ===")
        prompt_parts.append(
            f"Erstelle Entscheidungen für diese Ticker: {tickers_str}\n"
            f"Verwende die quantitativen Scores als FINAL DECISION BASIS (KEINE Adjustments). "
            f"Deine Aufgabe: Erkläre, WARUM diese Scores aktuell sind. "
            f"Fülle das 'reasoning' Feld mit echten Metrikwerten aus den Scores. "
            f"Antworte NUR mit JSON."
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

        if not self.client:
            log.warning("Kein OpenAI Client – nutze Score-basierte Fallback-Analyse.")
            return self._score_based_fallback(scores, market_data, portfolio_summary, watchlist)

        prompt = self.build_prompt(
            portfolio_summary, market_data, news_text, watchlist, scores,
            journal_entries, regime_state=regime_state,
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
            result = self._parse_response(raw_text, scores)

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

    def _parse_response(self, raw_text: str, scores: Dict[str, ScoreBreakdown]) -> Dict:
        """Parst und validiert die JSON-Antwort. Enforces score-based guardrails."""
        try:
            clean = re.sub(r"```json\s*|\s*```", "", raw_text).strip()
            data = json.loads(clean)

            if "decisions" not in data:
                raise ValueError("Kein 'decisions' Feld")

            valid_decisions = []
            min_buy_score = self.risk_settings.get("min_buy_score", 60)

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

                # ── LLM-DECOUPLING: Quant Score is immutable ──
                # LLM can provide narrative context, but NOT score adjustments.
                # effective_score = quant_score (always, no LLM adjustment)
                
                ticker = d["ticker"]
                quant_score = float(d.get("quant_score", 0))
                
                # If quant_score missing, use score engine result
                if quant_score == 0 and ticker in scores:
                    quant_score = scores[ticker].total_score
                    d["quant_score"] = quant_score
                
                # LLM adjustments are IGNORED (decoupled)
                d["llm_score_adj"] = 0.0  # Always zero (not used)
                effective_score = quant_score  # NEVER adjusted by LLM
                
                # Guardrail: prevent BUY below minimum score (based on pure quant)
                if d["action"] == "BUY" and effective_score < min_buy_score:
                    log.warning(
                        f"[GUARDRAIL] {ticker}: BUY blockiert "
                        f"(effective_score={effective_score:.1f} < min={min_buy_score})"
                    )
                    d["action"] = "HOLD"
                    d["reason"] = f"{d['reason']} [GUARDRAIL: score too low]"

                # Guardrail: SELL with high confidence score stays SELL
                if d["action"] == "SELL" and d.get("target_allocation", 0) > 0.5:
                    d["target_allocation"] = 0.0
                    log.warning(f"{ticker}: SELL mit hoher target_allocation → korrigiert auf 0")

                # Ensure reasoning has metrics
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
    ) -> Dict:
        """
        Rein score-basierter Fallback ohne LLM.
        Deterministisch – keine Zufallselemente.
        """
        log.info("Score-basierter Fallback: verwendet quantitative Scores direkt")
        decisions = []
        min_buy_score = self.risk_settings.get("min_buy_score", 60)

        for ticker in watchlist:
            sb = scores.get(ticker)
            if not sb:
                continue

            action = sb.recommended_action
            score = sb.total_score

            # Allocations basieren auf Score-Stärke
            if action == "BUY":
                if score >= 80:
                    target_alloc = 0.12
                elif score >= 70:
                    target_alloc = 0.09
                else:
                    target_alloc = 0.06
            elif action == "SELL":
                target_alloc = 0.0
            else:
                target_alloc = scores[ticker].current_alloc  # HOLD = maintain

            decisions.append({
                "ticker": ticker,
                "action": action,
                "target_allocation": target_alloc,
                "confidence": sb.confidence,
                "quant_score": score,
                "llm_score_adj": 0,
                "reasoning": self._build_reasoning_from_score(sb),
                "reason": f"Score-based: {sb.signal} ({score:.0f}pts)",
            })

        return {
            "decisions": decisions,
            "market_outlook": "Fallback: score-based analysis (no LLM).",
            "risk_assessment": "Using quantitative scores only.",
            "scores": {t: sb.to_dict() for t, sb in scores.items()},
        }