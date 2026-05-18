"""
AI Trading Bot – Erweiterte Markt-Regime-Detektion
====================================================
Ergänzt den bestehenden MarketRegimeDetector um Leading Indicators,
die Stress früher ankündigen als VIX oder SMAs.

Warum Leading statt Lagging?
  VIX und SMAs reagieren auf das was bereits passiert ist.
  Bis VIX=30 erreicht wird, bist du oft 10–15% im Drawdown.
  Leading Indicators drehen häufig 2–6 Wochen früher.

Neue Indikatoren:
  1. Yield Spread (10Y minus 2Y Treasury)
     → Invertierte Kurve = Rezessionssignal (oft 6–18 Monate vorlaufend)
     → Quelle: ^TNX (10Y) und ^IRX (3M) via yfinance

  2. Credit Spread Proxy (HYG/LQD Ratio)
     → HYG = High-Yield-ETF, LQD = Investment-Grade-ETF
     → Ratio fällt = Kreditmarkt stresst → oft 3–8 Wochen vor Aktienmärkten

  3. Put/Call Ratio auf SPY-Optionen
     → Über 1.0 = überdurchschnittlich viele Absicherungen → Angst steigt
     → Quelle: VIX-ähnlich via yfinance (^PCCE als Proxy)

VERWENDUNG:
  Ersetze MarketRegimeDetector durch EnhancedMarketRegimeDetector in main.py:

    # main.py, Schritt 0:
    # ALT:
    from market_regime import MarketRegimeDetector
    regime_detector = MarketRegimeDetector(watchlist=FULL_WATCHLIST)

    # NEU:
    from market_regime_enhanced import EnhancedMarketRegimeDetector
    regime_detector = EnhancedMarketRegimeDetector(watchlist=FULL_WATCHLIST)

  Die detect()-Methode hat die gleiche Signatur und gibt denselben
  RegimeState zurück – vollständig rückwärtskompatibel.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from logger import log
from market_regime import (
    MarketRegimeDetector, RegimeState, Regime,
    apply_regime_to_risk_settings,
)

try:
    import yfinance as yf
except ImportError:
    log.error("yfinance nicht installiert.")
    raise


# ─────────────────────────────────────────────────────────────
# INDIKATOREN-KONFIGURATION
# ─────────────────────────────────────────────────────────────

# Yield Spread: 10Y minus 2Y (oder 3M als Proxy)
YIELD_10Y_TICKER  = "^TNX"   # 10-Jahres-Treasury (in %)
YIELD_2Y_TICKER   = "^IRX"   # 3-Monats-Treasury (Proxy für kurzfristigen Zins)

# Credit Spread Proxy
HYG_TICKER = "HYG"   # iShares iBoxx High Yield Corporate Bond ETF
LQD_TICKER = "LQD"   # iShares iBoxx Investment Grade Corporate Bond ETF

# Put/Call Ratio Proxy
PCCE_TICKER = "^PCCE"  # CBOE Equity Put/Call Ratio

# Schwellwerte
YIELD_INVERSION_THRESHOLD = 0.0   # 10Y - 2Y < 0 = invertierte Kurve
YIELD_FLAT_THRESHOLD      = 0.5   # 10Y - 2Y < 0.5% = flache Kurve (Warnsignal)
CREDIT_SPREAD_STRESS      = 0.95  # HYG/LQD Ratio unter diesem Wert = Stress
CREDIT_SPREAD_PANIC       = 0.90  # Unter diesem Wert = erhöhte Panik
PCCE_ELEVATED             = 0.80  # Put/Call Ratio über 0.8 = erhöhte Absicherung
PCCE_PANIC                = 1.10  # Über 1.1 = Panik-Absicherungen


# ─────────────────────────────────────────────────────────────
# ERWEITERTER DETEKTOR
# ─────────────────────────────────────────────────────────────

class EnhancedMarketRegimeDetector(MarketRegimeDetector):
    """
    Erweitert MarketRegimeDetector um drei Leading Indicators:
    Yield Spread, Credit Spreads und Put/Call Ratio.

    Vollständig rückwärtskompatibel: detect() hat dieselbe Signatur.
    Zusätzliche Daten werden im RegimeState.description vermerkt.
    """

    def __init__(self, watchlist: List[str] = None):
        super().__init__(watchlist)
        self._indicator_cache: Dict = {}

    # ─────────────────────────────────────────────────────────
    # ÖFFENTLICHE API
    # ─────────────────────────────────────────────────────────

    def detect(self) -> RegimeState:
        """
        Überschreibt MarketRegimeDetector.detect().
        Führt Basis-Detektion durch und ergänzt Leading Indicators.
        """
        # 1. Basis-Detektion (VIX + SPY-Trend + Marktbreite)
        base_state = super().detect()
        log.info("Erweitere Regime-Detektion um Leading Indicators...")

        # 2. Leading Indicators laden
        yield_data  = self._fetch_yield_spread()
        credit_data = self._fetch_credit_spread()
        pcce_data   = self._fetch_put_call_ratio()
        self._indicator_cache = {
            "yield": yield_data,
            "credit": credit_data,
            "pcce": pcce_data,
        }

        # 3. Leading-Scores berechnen
        leading_scores: List[float] = []
        leading_weights: List[float] = []
        leading_notes: List[str]     = []

        # Yield Spread
        yield_score, yield_note = self._score_yield_spread(yield_data)
        if yield_score is not None:
            leading_scores.append(yield_score)
            leading_weights.append(1.5)   # Mittelgewicht (langer Vorlauf, aber verzögerte Wirkung)
            leading_notes.append(yield_note)
            log.debug(f"  Yield Spread: {yield_note} → Score={yield_score:+.2f}")

        # Credit Spread
        credit_score, credit_note = self._score_credit_spread(credit_data)
        if credit_score is not None:
            leading_scores.append(credit_score)
            leading_weights.append(2.0)   # Stärkstes Leading Signal
            leading_notes.append(credit_note)
            log.debug(f"  Credit Spread: {credit_note} → Score={credit_score:+.2f}")

        # Put/Call Ratio
        pcce_score, pcce_note = self._score_put_call_ratio(pcce_data)
        if pcce_score is not None:
            leading_scores.append(pcce_score)
            leading_weights.append(1.0)   # Kurzfristiger Indikator, geringeres Gewicht
            leading_notes.append(pcce_note)
            log.debug(f"  Put/Call Ratio: {pcce_note} → Score={pcce_score:+.2f}")

        if not leading_scores:
            log.warning("  Keine Leading-Indicator-Daten verfügbar – nutze Basis-Regime.")
            final_state = base_state
        else:
            # 4. Leading-Score zusammenfassen
            leading_composite = float(
                np.average(leading_scores, weights=leading_weights)
            )
            log.info(f"  Leading-Composite-Score: {leading_composite:+.2f}")

            # 5. Basis-Regime mit Leading-Signal kombinieren
            final_state = self._merge_signals(base_state, leading_composite, leading_notes)
        
        # 🔥 APPLY HYSTERESIS FILTER – Verhindere Regime-Whipsaws
        from market_regime import RegimeHysteresisFilter
        hysteresis_filter = RegimeHysteresisFilter()
        final_state = hysteresis_filter.filter(final_state)
        
        return final_state

    def get_indicator_snapshot(self) -> Dict:
        """
        Gibt den letzten Indikator-Snapshot zurück (nach detect() verfügbar).
        Nützlich für Dashboard und Logging.
        """
        return self._indicator_cache.copy()

    # ─────────────────────────────────────────────────────────
    # LEADING INDICATOR 1: YIELD SPREAD
    # ─────────────────────────────────────────────────────────

    def _fetch_yield_spread(self) -> Dict:
        """
        Lädt 10Y und 3M-Treasury-Renditen und berechnet den Spread.
        Invertierter Spread (10Y < 3M) = historisch zuverlässiges Rezessionssignal.

        Rückgabe: {
            "spread":      float (10Y - 3M in Prozentpunkten),
            "yield_10y":   float,
            "yield_3m":    float,
            "inverted":    bool,
            "flat":        bool,
            "trend_30d":   float (Spread-Veränderung über 30 Tage),
        }
        """
        result = {}
        try:
            df_10y = yf.download(YIELD_10Y_TICKER, period="60d", progress=False, auto_adjust=True)
            df_3m  = yf.download(YIELD_2Y_TICKER,  period="60d", progress=False, auto_adjust=True)

            if df_10y.empty or df_3m.empty:
                log.debug("  Yield-Daten nicht verfügbar.")
                return result

            for df in [df_10y, df_3m]:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

            close_10y = df_10y["Close"].dropna()
            close_3m  = df_3m["Close"].dropna()

            # Gemeinsamer Index
            common = close_10y.index.intersection(close_3m.index)
            if len(common) < 2:
                return result

            spread_series = close_10y.loc[common] - close_3m.loc[common]
            current_spread = float(spread_series.iloc[-1])
            y10 = float(close_10y.loc[common].iloc[-1])
            y3m = float(close_3m.loc[common].iloc[-1])

            # 30-Tage-Trend des Spreads
            trend_30d = None
            if len(spread_series) >= 20:
                trend_30d = float(spread_series.iloc[-1] - spread_series.iloc[-20])

            result = {
                "spread":    round(current_spread, 3),
                "yield_10y": round(y10, 3),
                "yield_3m":  round(y3m, 3),
                "inverted":  current_spread < YIELD_INVERSION_THRESHOLD,
                "flat":      current_spread < YIELD_FLAT_THRESHOLD,
                "trend_30d": round(trend_30d, 3) if trend_30d is not None else None,
            }
            log.info(
                f"  Yield Spread: 10Y={y10:.2f}% | 3M={y3m:.2f}% | "
                f"Spread={current_spread:+.3f}pp"
                f"{' [INVERTIERT!]' if result['inverted'] else ''}"
            )

        except Exception as e:
            log.warning(f"  Yield-Spread-Abruf fehlgeschlagen: {e}")

        return result

    def _score_yield_spread(self, data: Dict) -> Tuple[Optional[float], str]:
        """
        Konvertiert Yield-Spread-Daten in Score (-1 bis +1) + Beschreibung.

        Interpretation:
          Spread > 1.5pp  → normaler, gesunder Markt → +1.0
          Spread 0.5–1.5  → leicht abflachend → 0.0 bis +0.5
          Spread 0–0.5    → flache Kurve, Vorsicht → -0.3
          Spread < 0      → invertiert, Rezessionswarnung → -0.8
          Spread stark negativ (< -0.5) → starkes Rezessionssignal → -1.0
        """
        if not data or "spread" not in data:
            return None, ""

        spread = data["spread"]
        inverted = data.get("inverted", False)
        trend_30d = data.get("trend_30d")

        if spread > 1.5:
            score = 1.0
            note  = f"Yield-Spread={spread:+.2f}pp: normal (bullish)"
        elif spread > 0.5:
            score = 0.3 + (spread - 0.5) / 1.0 * 0.5   # 0.3 bis 0.8
            note  = f"Yield-Spread={spread:+.2f}pp: leicht abflachend"
        elif spread >= 0:
            score = -0.3
            note  = f"Yield-Spread={spread:+.2f}pp: flache Kurve (Warnung)"
        elif spread >= -0.5:
            score = -0.7
            note  = f"Yield-Spread={spread:+.2f}pp: INVERTIERT – Rezessionswarnung"
        else:
            score = -1.0
            note  = f"Yield-Spread={spread:+.2f}pp: STARK INVERTIERT – hohes Rezessionsrisiko"

        # Trend-Modifikation: Spread der sich schnell verschlechtert = zusätzliche Warnung
        if trend_30d is not None and trend_30d < -0.3:
            score = max(-1.0, score - 0.2)
            note += f" (↓ Trend: {trend_30d:+.2f}pp/30d)"

        return float(score), note

    # ─────────────────────────────────────────────────────────
    # LEADING INDICATOR 2: CREDIT SPREAD (HYG/LQD)
    # ─────────────────────────────────────────────────────────

    def _fetch_credit_spread(self) -> Dict:
        """
        Berechnet den Credit-Spread-Proxy via HYG/LQD-Ratio.

        HYG (High Yield) fällt schneller als LQD (Investment Grade) wenn
        Kreditmarkt stresst → Ratio sinkt → Leading Signal für Aktien.

        Rückgabe: {
            "ratio":         float (HYG/LQD normalisiert),
            "ratio_raw":     float,
            "ratio_30d_avg": float,
            "stress":        bool,
            "panic":         bool,
            "trend_14d":     float,
        }
        """
        result = {}
        try:
            df_hyg = yf.download(HYG_TICKER, period="90d", progress=False, auto_adjust=True)
            df_lqd = yf.download(LQD_TICKER, period="90d", progress=False, auto_adjust=True)

            if df_hyg.empty or df_lqd.empty:
                log.debug("  Credit-Spread-Daten nicht verfügbar.")
                return result

            for df in [df_hyg, df_lqd]:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

            close_hyg = df_hyg["Close"].dropna()
            close_lqd = df_lqd["Close"].dropna()
            common    = close_hyg.index.intersection(close_lqd.index)

            if len(common) < 10:
                return result

            ratio_series = close_hyg.loc[common] / close_lqd.loc[common]

            # Normalisierung: Ratio relativ zum 60-Tage-Durchschnitt
            rolling_avg = ratio_series.rolling(60, min_periods=20).mean()
            normalized  = ratio_series / rolling_avg

            current_ratio     = float(ratio_series.iloc[-1])
            current_normalized = float(normalized.iloc[-1]) if not normalized.empty else 1.0
            avg_30d           = float(ratio_series.tail(30).mean())

            trend_14d = None
            if len(ratio_series) >= 14:
                trend_14d = float(
                    (ratio_series.iloc[-1] - ratio_series.iloc[-14]) / ratio_series.iloc[-14] * 100
                )

            result = {
                "ratio":         round(current_normalized, 4),
                "ratio_raw":     round(current_ratio, 4),
                "ratio_30d_avg": round(avg_30d, 4),
                "stress":        current_normalized < CREDIT_SPREAD_STRESS,
                "panic":         current_normalized < CREDIT_SPREAD_PANIC,
                "trend_14d":     round(trend_14d, 2) if trend_14d is not None else None,
            }
            log.info(
                f"  Credit Spread (HYG/LQD): ratio={current_normalized:.3f}"
                f"{' [STRESS]' if result['stress'] else ''}"
                f"{' [PANIK]' if result['panic'] else ''}"
            )

        except Exception as e:
            log.warning(f"  Credit-Spread-Abruf fehlgeschlagen: {e}")

        return result

    def _score_credit_spread(self, data: Dict) -> Tuple[Optional[float], str]:
        """
        Konvertiert Credit-Spread-Daten in Score (-1 bis +1).

        Normalisiertes Ratio (relativ zu 60-Tage-Avg):
          > 1.02  → Kreditmarkt entspannt → +0.8
          0.98–1.02 → normal → 0.0
          0.95–0.98 → leichter Stress → -0.4
          < 0.95  → deutlicher Stress → -0.8
          < 0.90  → Panik → -1.0
        """
        if not data or "ratio" not in data:
            return None, ""

        ratio    = data["ratio"]
        trend_14 = data.get("trend_14d")

        if ratio > 1.02:
            score = 0.8
            note  = f"Credit-Spread={ratio:.3f}: Kreditmarkt entspannt (bullish)"
        elif ratio >= 0.98:
            score = 0.0
            note  = f"Credit-Spread={ratio:.3f}: normal"
        elif ratio >= 0.95:
            score = -0.5
            note  = f"Credit-Spread={ratio:.3f}: leichter Kreditstress (Vorsicht)"
        elif ratio >= 0.90:
            score = -0.8
            note  = f"Credit-Spread={ratio:.3f}: KREDITSTRESS (Warnung)"
        else:
            score = -1.0
            note  = f"Credit-Spread={ratio:.3f}: KREDIT-PANIK (stark bearish)"

        # Trend-Modifikation
        if trend_14 is not None and trend_14 < -2.0:
            score = max(-1.0, score - 0.15)
            note += f" (schnell fallend: {trend_14:+.1f}%/14d)"

        return float(score), note

    # ─────────────────────────────────────────────────────────
    # LEADING INDICATOR 3: PUT/CALL RATIO
    # ─────────────────────────────────────────────────────────

    def _fetch_put_call_ratio(self) -> Dict:
        """
        Lädt den CBOE Equity Put/Call Ratio (^PCCE).
        Hohe Ratio = viele Absicherungen = Angst im Markt.

        Hinweis: ^PCCE ist kurzfristiger als Yield- oder Credit-Spreads.
        Werte über 1.0 sind selten und zeigen extreme Absicherung.

        Rückgabe: {
            "pcce":          float (aktueller Wert),
            "pcce_5d_avg":   float,
            "pcce_20d_avg":  float,
            "elevated":      bool,
            "panic":         bool,
        }
        """
        result = {}
        try:
            import contextlib, io
            # ^PCCE liefert bei Yahoo Finance häufig 404 → stderr unterdrücken
            with contextlib.redirect_stderr(io.StringIO()):
                df = yf.download(PCCE_TICKER, period="30d", progress=False, auto_adjust=True)

            if df.empty:
                log.debug("  Put/Call-Ratio (^PCCE) nicht verfügbar – wird übersprungen.")
                return result

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            close = df["Close"].dropna()
            if len(close) < 3:
                return result

            current   = float(close.iloc[-1])
            avg_5d    = float(close.tail(5).mean())
            avg_20d   = float(close.tail(20).mean()) if len(close) >= 20 else avg_5d

            result = {
                "pcce":         round(current, 3),
                "pcce_5d_avg":  round(avg_5d, 3),
                "pcce_20d_avg": round(avg_20d, 3),
                "elevated":     current > PCCE_ELEVATED,
                "panic":        current > PCCE_PANIC,
            }
            log.info(
                f"  Put/Call Ratio (PCCE): {current:.3f} (5d-Avg: {avg_5d:.3f})"
                f"{' [erhöht]' if result['elevated'] else ''}"
                f"{' [PANIK]' if result['panic'] else ''}"
            )

        except Exception as e:
            # ^PCCE wird von Yahoo Finance aktuell nicht unterstützt (404)
            # → als debug loggen damit der Run sauber bleibt
            log.debug(f"  Put/Call-Ratio (^PCCE) nicht verfügbar: {e}")

        return result

    def _score_put_call_ratio(self, data: Dict) -> Tuple[Optional[float], str]:
        """
        Konvertiert Put/Call Ratio in Score (-1 bis +1).

        Niedrige Put/Call = Selbstzufriedenheit (contrarian: leicht bearish)
        Hohe Put/Call = Angst/Absicherung (contrarian: leicht bullish wg. Mean Reversion)

        Achtung: Put/Call Ratio ist contrarian ANDERS als Yield/Credit!
        - Sehr hohe Ratio (Panik) → Markt oft überverkauft → kurzfristig bullish
        - Aber anhaltend hohe Ratio + BEAR-Markt → bearish bestätigt
        Daher: Wir nutzen ihn als STRESS-Indikator, nicht rein contrarian.
        """
        if not data or "pcce" not in data:
            return None, ""

        pcce     = data["pcce"]
        avg_5d   = data.get("pcce_5d_avg", pcce)
        avg_20d  = data.get("pcce_20d_avg", pcce)

        # Relative Position: aktuell vs. 20-Tage-Schnitt
        rel = pcce / avg_20d if avg_20d > 0 else 1.0

        if pcce < 0.55:
            # Sehr geringe Absicherung = Euphorie = leicht bearish (Contrarian)
            score = -0.2
            note  = f"Put/Call={pcce:.2f}: Geringe Absicherung (Euphorie-Warnung)"
        elif pcce < 0.70:
            score = 0.2
            note  = f"Put/Call={pcce:.2f}: normal-niedrig"
        elif pcce < PCCE_ELEVATED:
            score = 0.0
            note  = f"Put/Call={pcce:.2f}: normal"
        elif pcce < PCCE_PANIC:
            # Erhöhte Absicherung = wachsende Angst = bearish (im Trend)
            score = -0.4
            note  = f"Put/Call={pcce:.2f}: erhöhte Absicherung (Stresssignal)"
        else:
            # Panik-Absicherung = extremer Stress (bearish im laufenden Markt)
            score = -0.7
            note  = f"Put/Call={pcce:.2f}: PANIK-Absicherung (extremer Stress)"

        # Spike-Warnung: Stark angestiegener PCCE als kurzfristiger Alarm
        if rel > 1.3 and pcce > PCCE_ELEVATED:
            score = max(-1.0, score - 0.2)
            note += f" (↑ Spike: +{(rel-1)*100:.0f}% über 20d-Avg)"

        return float(score), note

    # ─────────────────────────────────────────────────────────
    # SIGNAL-KOMBINATION
    # ─────────────────────────────────────────────────────────

    def _merge_signals(
        self,
        base_state: RegimeState,
        leading_composite: float,
        leading_notes: List[str],
    ) -> RegimeState:
        """
        Kombiniert Basis-Regime mit Leading-Composite-Score.

        Prinzip:
          - Leading-Score allein ändert das Regime NICHT (zu viele Fehlsignale)
          - Aber er VERSTÄRKT oder ABSCHWÄCHT das Basis-Signal
          - Leading-Score < -0.5 bei BULL → Early-Warning-Flag setzen
          - Leading-Score > +0.5 bei BEAR → Erholung möglich, etwas entspannen

        Das finale Regime bleibt beim Basis-Regime, aber:
          - Confidence wird angepasst
          - Description wird um Leading-Infos ergänzt
          - Bei starkem Widerspruch: Regime-Override möglich
        """
        base_regime      = base_state.regime
        base_confidence  = base_state.confidence
        new_regime       = base_regime
        new_confidence   = base_confidence
        conf_delta_add   = 0.0
        description_ext  = ""

        leading_info = " | ".join(leading_notes[:3]) if leading_notes else ""

        # ── Fall 1: Leading warnt bei BULL ────────────────────────────────────
        # Leading-Score deutlich negativ, aber Basis ist BULL
        # → Frühwarnung einbauen: Confidence senken, mehr Cash-Reserve
        if base_regime == Regime.BULL and leading_composite < -0.4:
            severity = abs(leading_composite + 0.4) / 0.6   # 0 bis 1
            new_confidence = max(0.40, base_confidence - 0.20 * severity)
            conf_delta_add = 0.08 * severity  # Leicht erhöhte Schwelle für Trades

            if leading_composite < -0.7:
                # Starke Leading-Warnung: Frühzeitig auf SIDEWAYS herunterstufen
                new_regime     = Regime.SIDEWAYS
                new_confidence = 0.55
                conf_delta_add = 0.12
                description_ext = (
                    f" | ⚠ FRÜHWARNUNG: Leading Indicators deuten auf Stress hin "
                    f"(Score={leading_composite:+.2f}). Regime auf SIDEWAYS vorgezogen."
                )
                log.warning(
                    f"  LEADING EARLY WARNING: BULL→SIDEWAYS Override "
                    f"(Leading-Score={leading_composite:+.2f})"
                )
            else:
                description_ext = (
                    f" | ⚡ Early-Warning: Leading Indikatoren leicht negativ "
                    f"(Score={leading_composite:+.2f}), erhöhte Vorsicht."
                )
                log.info(
                    f"  EARLY WARNING eingebaut (Leading={leading_composite:+.2f}, "
                    f"Confidence {base_confidence:.0%}→{new_confidence:.0%})"
                )

        # ── Fall 2: Leading bestätigt BEAR ────────────────────────────────────
        elif base_regime == Regime.BEAR and leading_composite < -0.3:
            # Beide Signale negativ → erhöhe Confidence, strengere Limits
            new_confidence = min(0.95, base_confidence + 0.08)
            conf_delta_add = 0.05
            description_ext = (
                f" | Leading Indicators BESTÄTIGEN Abwärtstrend "
                f"(Score={leading_composite:+.2f})."
            )
            log.info(f"  BEAR bestätigt durch Leading Indicators.")

        # ── Fall 3: Leading kontraindiziert BEAR ──────────────────────────────
        elif base_regime == Regime.BEAR and leading_composite > 0.3:
            # Kreditmarkt entspannt, Kurven normal → BEAR evtl. übertrieben
            new_confidence = max(0.45, base_confidence - 0.15)
            description_ext = (
                f" | Leading Indicators WIDERSPRECHEN BEAR-Signal "
                f"(Score={leading_composite:+.2f}). Mögliche Erholung."
            )
            log.info(
                f"  BEAR-Signal geschwächt: Leading Indicators positiv "
                f"(Score={leading_composite:+.2f})"
            )

        # ── Fall 4: Leading bestätigt BULL ────────────────────────────────────
        elif base_regime == Regime.BULL and leading_composite > 0.4:
            new_confidence = min(0.95, base_confidence + 0.05)
            description_ext = f" | Leading Indicators bestätigen Trend (Score={leading_composite:+.2f})."

        # ── Finalen State zusammenbauen ───────────────────────────────────────
        # Baue auf dem Basis-State auf, überschreibe nur was sich geändert hat
        final_description = (
            base_state.description
            + description_ext
            + (f"\n  Leading: {leading_info}" if leading_info else "")
        )

        from dataclasses import replace
        final_state = replace(
            base_state,
            regime=new_regime,
            confidence=round(new_confidence, 3),
            description=final_description,
        )

        # Conf-Delta kombinieren (Basis-Delta + Leading-Delta)
        final_state.confidence_threshold_delta = round(
            base_state.confidence_threshold_delta + conf_delta_add, 3
        )

        log.info(
            f"  Finales Regime: {final_state.label} "
            f"(Basis: {base_regime.value.upper()} | "
            f"Leading: {leading_composite:+.2f} | "
            f"Confidence: {base_state.confidence:.0%}→{new_confidence:.0%})"
        )

        return final_state


# ─────────────────────────────────────────────────────────────
# STANDALONE: QUICK-CHECK
# ─────────────────────────────────────────────────────────────

def run_indicator_check():
    """
    Standalone: Gibt einen schnellen Überblick über alle Leading Indicators.
    Zum manuellen Prüfen: python market_regime_enhanced.py
    """
    from config import FULL_WATCHLIST

    print("\n" + "=" * 65)
    print("  LEADING INDICATOR CHECK")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    detector = EnhancedMarketRegimeDetector(watchlist=FULL_WATCHLIST)
    state    = detector.detect()

    print(f"\n{state.summary()}")

    cache = detector.get_indicator_snapshot()

    if cache.get("yield"):
        y = cache["yield"]
        print(f"\nYield Spread Detail:")
        print(f"  10Y: {y.get('yield_10y', 'n/a')}%  |  3M: {y.get('yield_3m', 'n/a')}%")
        print(f"  Spread: {y.get('spread', 'n/a'):+.3f}pp"
              f"{'  ← INVERTIERT!' if y.get('inverted') else ''}")
        if y.get("trend_30d") is not None:
            print(f"  30d-Trend: {y['trend_30d']:+.3f}pp")

    if cache.get("credit"):
        c = cache["credit"]
        print(f"\nCredit Spread (HYG/LQD):")
        print(f"  Ratio: {c.get('ratio', 'n/a'):.3f} (raw: {c.get('ratio_raw', 'n/a'):.4f})")
        print(f"  14d-Trend: {c.get('trend_14d', 'n/a')}%"
              f"{'  ← STRESS' if c.get('stress') else ''}")

    if cache.get("pcce"):
        p = cache["pcce"]
        print(f"\nPut/Call Ratio (PCCE):")
        print(f"  Aktuell: {p.get('pcce', 'n/a'):.3f}  |  5d-Avg: {p.get('pcce_5d_avg', 'n/a'):.3f}")
        print(f"  20d-Avg: {p.get('pcce_20d_avg', 'n/a'):.3f}"
              f"{'  ← ERHÖHT' if p.get('elevated') else ''}")

    print("\n" + "=" * 65)
    return state


if __name__ == "__main__":
    run_indicator_check()