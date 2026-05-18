"""
AI Trading Bot - Markt-Regime-Detektor
========================================
Erkennt automatisch das aktuelle Marktregime zu Beginn jedes Runs.

Regime-Kategorien:
  BULL     – Trendmarkt aufwärts, Momentum-Strategien funktionieren gut
  BEAR     – Trendmarkt abwärts, defensiv agieren
  SIDEWAYS – Choppy/trendlos, Momentum-Signale sind größtenteils Fehlsignale

Indikatoren:
  - VIX-Level           → Volatilitätsregime
  - SPY-Trendrichtung   → SMA20 vs SMA50 Crossover
  - Marktbreite         → Anteil Assets über 50-Tage-Linie
  - Momentum-Score      → Gewichteter Return über 20/60 Tage

Auswirkungen je Regime:
  BULL     → normales Verhalten (Profil-Settings gelten unverändert)
  BEAR     → max_position_pct −40%, confidence_threshold +15%, nur defensive ETFs preferred
  SIDEWAYS → max_trades_per_run halbiert, confidence_threshold +10%, mehr Cash-Reserve
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from logger import log

try:
    import yfinance as yf
except ImportError:
    log.error("yfinance nicht installiert.")
    raise


# ─────────────────────────────────────────────────────────────
# REGIME ENUM & DATACLASS
# ─────────────────────────────────────────────────────────────

class Regime(Enum):
    BULL     = "bull"
    BEAR     = "bear"
    SIDEWAYS = "sideways"


@dataclass
class RegimeState:
    """Vollständiger Regime-Snapshot für einen Run."""
    regime:           Regime
    confidence:       float          # 0.0–1.0 wie sicher ist die Klassifikation

    # Rohdaten der Indikatoren
    vix:              Optional[float] = None
    spy_above_sma20:  Optional[bool]  = None
    spy_above_sma50:  Optional[bool]  = None
    spy_return_20d:   Optional[float] = None
    spy_return_60d:   Optional[float] = None
    market_breadth:   Optional[float] = None   # Anteil Assets über SMA50 (0–1)
    momentum_score:   Optional[float] = None   # gewichteter Score (−1 bis +1)

    # Override-Felder für Risk-Manager
    max_position_override:      Optional[float] = None   # None = kein Override
    min_cash_override:          Optional[float] = None
    confidence_threshold_delta: float = 0.0              # Addiert auf Profil-Schwellwert
    max_trades_multiplier:      float = 1.0              # 1.0 = unverändert

    # Beschreibung
    description:      str = ""
    detected_at:      str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def label(self) -> str:
        return self.regime.value.upper()

    def summary(self) -> str:
        lines = [
            f"Markt-Regime: {self.label} (Konfidenz: {self.confidence:.0%})",
            f"  VIX: {self.vix:.1f}" if self.vix else "  VIX: n/a",
            f"  SPY > SMA20: {self.spy_above_sma20} | SPY > SMA50: {self.spy_above_sma50}",
            f"  SPY Return 20d: {self.spy_return_20d:+.1f}%" if self.spy_return_20d is not None else "  SPY Return: n/a",
            f"  Marktbreite (>SMA50): {self.market_breadth:.0%}" if self.market_breadth is not None else "  Marktbreite: n/a",
            f"  Momentum-Score: {self.momentum_score:+.2f}" if self.momentum_score is not None else "  Momentum-Score: n/a",
            f"  → {self.description}",
        ]
        if self.max_position_override is not None:
            lines.append(f"  Override max_position: {self.max_position_override:.0%}")
        if self.min_cash_override is not None:
            lines.append(f"  Override min_cash: {self.min_cash_override:.0%}")
        if self.confidence_threshold_delta != 0.0:
            lines.append(f"  Confidence-Threshold Δ: {self.confidence_threshold_delta:+.0%}")
        if self.max_trades_multiplier != 1.0:
            lines.append(f"  Max-Trades Faktor: {self.max_trades_multiplier:.1f}x")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "regime":                   self.regime.value,
            "confidence":               round(self.confidence, 3),
            "vix":                      self.vix,
            "spy_above_sma20":          self.spy_above_sma20,
            "spy_above_sma50":          self.spy_above_sma50,
            "spy_return_20d":           self.spy_return_20d,
            "spy_return_60d":           self.spy_return_60d,
            "market_breadth":           self.market_breadth,
            "momentum_score":           self.momentum_score,
            "max_position_override":    self.max_position_override,
            "min_cash_override":        self.min_cash_override,
            "confidence_threshold_delta": self.confidence_threshold_delta,
            "max_trades_multiplier":    self.max_trades_multiplier,
            "description":              self.description,
            "detected_at":              self.detected_at,
        }


# ─────────────────────────────────────────────────────────────
# REGIME HYSTERESIS FILTER
# ─────────────────────────────────────────────────────────────

class RegimeHysteresisFilter:
    """
    🔥 REGIME STABILITY LAYER: Verhindert Whipsaws bei Regime-Übergängen
    
    Problem: Regime-Wechsel können aus Noise entstehen → führen zu falschen
    Risk-Settings Änderungen → unnötige Drawdowns
    
    Lösung: HYSTERESIS FILTER – Neues Regime muss 2x hintereinander erkannt
    werden bevor es ein etabliertes neues Regime wird.
    
    Verhalten:
      - Run 1: Detect BEAR → Regime bleibt BULL (old), aber speichere "BEAR_detected_1x"
      - Run 2: Detect BEAR nochmal → OK, wechsel zu BEAR (new)
      - Run 3: Detect BULL → Zurück zu BULL nur wenn nochmal in Run 4
      
    Speicherung: Letzte 10 Regime-States in JSON persistent.
    """
    
    REGIME_HISTORY_FILE = "logs/regime_history.json"
    MAX_HISTORY = 10
    
    def __init__(self):
        self.regime_history: List[Dict] = self._load_history()
        self.stable_regime: Optional[Regime] = self._get_last_stable_regime()
        log.info(
            f"[RegimeHysteresis] Initialized. "
            f"Stable regime: {self.stable_regime.value if self.stable_regime else 'NONE'}"
        )
    
    def _load_history(self) -> List[Dict]:
        """Lade die letzten 10 Regime-States aus JSON."""
        from utils import load_json_file
        history = load_json_file(self.REGIME_HISTORY_FILE, default=[])
        # Sort by detected_at DESC (neueste zuerst)
        if isinstance(history, list):
            return sorted(history, key=lambda x: x.get("detected_at", ""), reverse=True)[:self.MAX_HISTORY]
        return []
    
    def _save_history(self):
        """Speichere Regime-History in JSON."""
        from utils import save_json_file
        save_json_file(self.REGIME_HISTORY_FILE, self.regime_history)
    
    def _get_last_stable_regime(self) -> Optional[Regime]:
        """Lese das zuletzt stabile Regime aus History."""
        if not self.regime_history:
            return Regime.BULL  # Fallback zu BULL
        
        last_regime_str = self.regime_history[0].get("regime")
        if last_regime_str:
            try:
                return Regime(last_regime_str)
            except:
                return Regime.BULL
        return Regime.BULL
    
    def filter(self, detected_regime: RegimeState) -> RegimeState:
        """
        Filtern Sie das erkannte Regime durch Hysteresis-Logik.
        
        Returns: Gefilterte RegimeState (entweder neu erkannt 2x oder stable)
        """
        detected_regime_enum = detected_regime.regime
        
        # Ist dies eine WIEDERHOLUNG des letzten erkannten Regimes?
        if self.regime_history:
            last_detected = self.regime_history[0].get("regime")
            if last_detected == detected_regime_enum.value:
                # ✅ CONFIRMED: Regime wurde 2x hintereinander erkannt
                self.stable_regime = detected_regime_enum
                self.regime_history.insert(0, {
                    "regime": detected_regime_enum.value,
                    "confidence": detected_regime.confidence,
                    "detected_at": datetime.now().isoformat(),
                    "status": "CONFIRMED_2X",
                })
                self._trim_and_save_history()
                log.info(
                    f"[RegimeHysteresis] ✅ REGIME CONFIRMED: {detected_regime_enum.value.upper()} "
                    f"detected 2x in a row (confidence: {detected_regime.confidence:.0%})"
                )
                return detected_regime
            else:
                # ❌ DIFFERENT: Regime hat sich gewechselt
                # Speichere die neue Detektion, aber behalte altes stabiles Regime
                self.regime_history.insert(0, {
                    "regime": detected_regime_enum.value,
                    "confidence": detected_regime.confidence,
                    "detected_at": datetime.now().isoformat(),
                    "status": "PENDING_CONFIRMATION",
                })
                self._trim_and_save_history()
                
                log.warning(
                    f"[RegimeHysteresis] ⏳ REGIME TRANSITION PENDING: "
                    f"Detected {detected_regime_enum.value.upper()} "
                    f"but keeping {self.stable_regime.value.upper()} until confirmed again"
                )
                
                # Return den stabilen Regime statt des neuen
                detected_regime.regime = self.stable_regime
                return detected_regime
        else:
            # Erste Run – speichere und behalte
            self.stable_regime = detected_regime_enum
            self.regime_history.insert(0, {
                "regime": detected_regime_enum.value,
                "confidence": detected_regime.confidence,
                "detected_at": datetime.now().isoformat(),
                "status": "FIRST_DETECTION",
            })
            self._trim_and_save_history()
            return detected_regime
    
    def _trim_and_save_history(self):
        """Behalte nur die letzten N Einträge."""
        self.regime_history = self.regime_history[:self.MAX_HISTORY]
        self._save_history()

# ─────────────────────────────────────────────────────────────
# DETECTOR
# ─────────────────────────────────────────────────────────────

class MarketRegimeDetector:
    """
    Erkennt das aktuelle Marktregime anhand mehrerer Indikatoren.

    Scoring-System:
      Jeder Indikator trägt einen Score bei (positiv = bullish, negativ = bearish).
      Finaler Score → Regime-Klassifikation + Confidence.
    """

    # VIX-Schwellwerte
    VIX_LOW     = 15.0   # Ruhiger Markt
    VIX_MEDIUM  = 20.0   # Erhöhte Vorsicht
    VIX_HIGH    = 30.0   # Angstmodus → BEAR-Signal

    # Trendwechsel (Crossover-Puffer)
    SMA_BUFFER  = 0.005  # 0.5% Puffer um False-Crossovers zu vermeiden

    def __init__(self, watchlist: List[str] = None):
        """
        watchlist: Assets zur Berechnung der Marktbreite.
        Wenn None, werden nur SPY/QQQ-Signale genutzt.
        """
        self.watchlist = watchlist or []

    # ─────────────────────────────
    # PUBLIC: DETECT
    # ─────────────────────────────

    def detect(self) -> RegimeState:
        """
        Hauptfunktion: Erkennt das aktuelle Regime.
        Gibt einen vollständigen RegimeState zurück.
        """
        log.info("Markt-Regime-Detektion startet...")

        scores: List[float] = []   # −1 (bearish) bis +1 (bullish)
        weights: List[float] = []

        # ── 1. VIX ───────────────────────────────────────────
        vix = self._fetch_vix()
        vix_score = self._score_vix(vix)
        if vix_score is not None:
            scores.append(vix_score)
            weights.append(2.0)   # VIX hat doppeltes Gewicht
            log.debug(f"  VIX={vix:.1f} → Score={vix_score:+.2f}")

        # ── 2. SPY-Trend ──────────────────────────────────────
        spy_data = self._fetch_price_series("SPY", days=80)
        spy_above_sma20 = spy_above_sma50 = spy_return_20d = spy_return_60d = None
        trend_score = None

        if spy_data is not None and len(spy_data) >= 60:
            sma20 = float(spy_data.tail(20).mean())
            sma50 = float(spy_data.tail(50).mean())
            current = float(spy_data.iloc[-1])

            spy_above_sma20 = current > sma20 * (1 + self.SMA_BUFFER)
            spy_above_sma50 = current > sma50 * (1 + self.SMA_BUFFER)

            spy_return_20d = float((current / spy_data.iloc[-20] - 1) * 100) if len(spy_data) >= 20 else None
            spy_return_60d = float((current / spy_data.iloc[-60] - 1) * 100) if len(spy_data) >= 60 else None

            # Trend-Score
            trend_score = 0.0
            if spy_above_sma20:
                trend_score += 0.4
            else:
                trend_score -= 0.4
            if spy_above_sma50:
                trend_score += 0.6
            else:
                trend_score -= 0.6

            scores.append(trend_score)
            weights.append(2.5)
            log.debug(f"  SPY >SMA20={spy_above_sma20} >SMA50={spy_above_sma50} → Score={trend_score:+.2f}")

        # ── 3. Momentum (SPY Returns) ─────────────────────────
        momentum_score = None
        if spy_return_20d is not None and spy_return_60d is not None:
            mom_20 = np.clip(spy_return_20d / 10.0, -1.0, 1.0)   # ±10% → ±1
            mom_60 = np.clip(spy_return_60d / 15.0, -1.0, 1.0)   # ±15% → ±1
            momentum_score = float(0.4 * mom_20 + 0.6 * mom_60)
            scores.append(momentum_score)
            weights.append(1.5)
            log.debug(f"  Momentum 20d={spy_return_20d:+.1f}% 60d={spy_return_60d:+.1f}% → Score={momentum_score:+.2f}")

        # ── 4. Marktbreite ────────────────────────────────────
        breadth = self._calculate_market_breadth()
        breadth_score = None
        if breadth is not None:
            breadth_score = (breadth - 0.5) * 2.0   # 50% Breite → 0, 100% → +1, 0% → -1
            scores.append(breadth_score)
            weights.append(1.5)
            log.debug(f"  Marktbreite={breadth:.0%} → Score={breadth_score:+.2f}")

        # ── 5. QQQ-Divergenz (Tech-Sentiment) ─────────────────
        qqq_data = self._fetch_price_series("QQQ", days=30)
        if qqq_data is not None and len(qqq_data) >= 20:
            qqq_sma20  = float(qqq_data.tail(20).mean())
            qqq_curr   = float(qqq_data.iloc[-1])
            qqq_score  = 0.5 if qqq_curr > qqq_sma20 else -0.5
            scores.append(qqq_score)
            weights.append(1.0)
            log.debug(f"  QQQ>SMA20={qqq_curr > qqq_sma20} → Score={qqq_score:+.2f}")

        # ── Gewichteten Gesamt-Score berechnen ─────────────────
        if not scores:
            log.warning("Keine Indikatoren verfügbar – Regime: SIDEWAYS (Fallback)")
            return self._build_state(
                Regime.SIDEWAYS, 0.3,
                vix=vix,
                description="Keine Indikatordaten – konservativer Fallback",
            )

        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        log.debug(f"  Gesamt-Score: {weighted_score:+.3f} (aus {len(scores)} Indikatoren)")

        # ── Regime klassifizieren ──────────────────────────────
        regime, confidence, description = self._classify(weighted_score, vix)

        state = self._build_state(
            regime=regime,
            confidence=confidence,
            vix=vix,
            spy_above_sma20=spy_above_sma20,
            spy_above_sma50=spy_above_sma50,
            spy_return_20d=spy_return_20d,
            spy_return_60d=spy_return_60d,
            market_breadth=breadth,
            momentum_score=momentum_score,
            description=description,
        )

        log.info(state.summary())
        return state

    # ─────────────────────────────
    # KLASSIFIKATION
    # ─────────────────────────────

    def _classify(
        self,
        score: float,
        vix: Optional[float],
    ) -> Tuple[Regime, float, str]:
        """
        Klassifiziert den Score in ein Regime.

        Score-Bereiche:
          > +0.35  → BULL
          < -0.35  → BEAR
          sonst    → SIDEWAYS

        VIX kann das Regime überschreiben (Panik-Regime).
        """
        # VIX-Override: Extreme Angst → immer BEAR, egal was Score sagt
        if vix is not None and vix >= self.VIX_HIGH:
            confidence = min(0.95, 0.6 + (vix - self.VIX_HIGH) / 20.0)
            return (
                Regime.BEAR,
                confidence,
                f"VIX={vix:.1f} über Panik-Schwelle ({self.VIX_HIGH}) – BEAR Override aktiv",
            )

        # Normaler Score-Pfad
        if score > 0.35:
            # Confidence: linearer Anstieg von 0.55 bis 0.95
            confidence = min(0.95, 0.55 + (score - 0.35) * 1.0)
            if vix is not None and vix < self.VIX_LOW:
                confidence = min(0.95, confidence + 0.05)   # Ruhiger Markt stärkt BULL
            return (
                Regime.BULL,
                confidence,
                f"Score={score:+.2f}: Klarer Aufwärtstrend, Momentum positiv",
            )

        elif score < -0.35:
            confidence = min(0.95, 0.55 + (-score - 0.35) * 1.0)
            return (
                Regime.BEAR,
                confidence,
                f"Score={score:+.2f}: Abwärtstrend erkannt, defensiv agieren",
            )

        else:
            # SIDEWAYS: Je näher an 0, desto sicherer
            confidence = min(0.90, 0.50 + (0.35 - abs(score)) * 1.4)
            vix_str = f", VIX={vix:.1f}" if vix else ""
            return (
                Regime.SIDEWAYS,
                confidence,
                f"Score={score:+.2f}: Kein klarer Trend{vix_str} – trendloser Markt",
            )

    # ─────────────────────────────
    # OVERRIDE-SETTINGS
    # ─────────────────────────────

    def _build_state(
        self,
        regime: Regime,
        confidence: float,
        vix: Optional[float] = None,
        spy_above_sma20: Optional[bool] = None,
        spy_above_sma50: Optional[bool] = None,
        spy_return_20d: Optional[float] = None,
        spy_return_60d: Optional[float] = None,
        market_breadth: Optional[float] = None,
        momentum_score: Optional[float] = None,
        description: str = "",
    ) -> RegimeState:
        """
        Erstellt den RegimeState inkl. Risk-Override-Felder.

        BULL     → keine Overrides (Profil-Settings gelten)
        BEAR     → konservativere Limits, höhere Confidence-Schwelle
        SIDEWAYS → weniger Trades, höhere Confidence-Schwelle
        """
        state = RegimeState(
            regime=regime,
            confidence=confidence,
            vix=vix,
            spy_above_sma20=spy_above_sma20,
            spy_above_sma50=spy_above_sma50,
            spy_return_20d=spy_return_20d,
            spy_return_60d=spy_return_60d,
            market_breadth=market_breadth,
            momentum_score=momentum_score,
            description=description,
        )

        if regime == Regime.BEAR:
            # BEAR: Maximalposition −40%, Cash-Reserve +15%, Confidence +15%, Trades −30%
            state.max_position_override      = None   # Wir skalieren relativ (im Risk-Manager)
            state.min_cash_override          = None
            state.confidence_threshold_delta = +0.15
            state.max_trades_multiplier      = 0.70

        elif regime == Regime.SIDEWAYS:
            # SIDEWAYS: Confidence +10%, Trades halbiert
            state.confidence_threshold_delta = +0.10
            state.max_trades_multiplier      = 0.50

        # BULL: alles auf Default (0 Delta, 1.0 Multiplier)

        return state

    # ─────────────────────────────
    # DATENQUELLEN
    # ─────────────────────────────

    def _score_vix(self, vix: Optional[float]) -> Optional[float]:
        """
        Konvertiert den VIX-Wert in einen Score zwischen -1 (bearish) und +1 (bullish).

        VIX < 15  → ruhiger Markt    → +1.0 (sehr bullish)
        VIX 15-20 → normaler Bereich → 0.0 bis +1.0
        VIX 20-30 → erhöhte Angst   → 0.0 bis -1.0
        VIX > 30  → Panik           → -1.0 (sehr bearish)
        """
        if vix is None:
            return None
        if vix <= self.VIX_LOW:
            return 1.0
        elif vix <= self.VIX_MEDIUM:
            # Linearer Übergang von +1 bis 0
            return 1.0 - (vix - self.VIX_LOW) / (self.VIX_MEDIUM - self.VIX_LOW)
        elif vix <= self.VIX_HIGH:
            # Linearer Übergang von 0 bis -1
            return -((vix - self.VIX_MEDIUM) / (self.VIX_HIGH - self.VIX_MEDIUM))
        else:
            return -1.0

    def _fetch_vix(self) -> Optional[float]:
        """Lädt aktuellen VIX-Stand von Yahoo Finance (Ticker: ^VIX)."""
        try:
            df = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return float(df["Close"].iloc[-1])
        except Exception as e:
            log.warning(f"VIX-Abruf fehlgeschlagen: {e}")
            return None

    def _fetch_price_series(self, ticker: str, days: int = 80) -> Optional[pd.Series]:
        """Lädt Closing-Preisserie für einen Ticker."""
        try:
            end   = datetime.today()
            start = end - timedelta(days=days + 15)
            df    = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return df["Close"].dropna()
        except Exception as e:
            log.warning(f"Preisdaten für {ticker} fehlgeschlagen: {e}")
            return None

    def _calculate_market_breadth(self) -> Optional[float]:
        """
        Berechnet Marktbreite: Anteil der Watchlist-Assets über ihrer 50-Tage-Linie.
        Gibt float zwischen 0 (alle darunter) und 1 (alle darüber) zurück.
        Gibt None zurück wenn zu wenig Daten.
        """
        if len(self.watchlist) < 3:
            return None

        above_count = 0
        total_count = 0

        # Max 15 Ticker für Performance (kein API-Spam)
        sample = self.watchlist[:15]

        for ticker in sample:
            series = self._fetch_price_series(ticker, days=65)
            if series is None or len(series) < 50:
                continue
            sma50   = float(series.tail(50).mean())
            current = float(series.iloc[-1])
            total_count += 1
            if current > sma50:
                above_count += 1

        if total_count < 2:
            return None

        return above_count / total_count


# ─────────────────────────────────────────────────────────────
# REGIME → RISK-SETTINGS ANPASSEN
# ─────────────────────────────────────────────────────────────

def apply_regime_to_risk_settings(
    base_settings: Dict,
    regime_state: RegimeState,
) -> Dict:
    """
    Passt die Risiko-Settings basierend auf dem erkannten Regime an.
    Gibt ein neues Settings-Dict zurück (das originale bleibt unverändert).

    Aufgerufen vom RiskManager zu Beginn jedes Runs.
    """
    settings = base_settings.copy()

    if regime_state.regime == Regime.BEAR:
        # Max-Position 40% reduzieren
        settings["max_position_pct"] = round(
            settings["max_position_pct"] * 0.60, 3
        )
        # Cash-Reserve um 10 Prozentpunkte erhöhen
        settings["min_cash_pct"] = min(
            0.50, settings["min_cash_pct"] + 0.10
        )
        # Confidence-Schwelle anheben
        settings["confidence_threshold"] = min(
            0.95,
            settings["confidence_threshold"] + regime_state.confidence_threshold_delta
        )
        # Trades reduzieren
        settings["max_trades_per_run"] = max(
            1, int(settings["max_trades_per_run"] * regime_state.max_trades_multiplier)
        )

    elif regime_state.regime == Regime.SIDEWAYS:
        # Confidence-Schwelle leicht anheben
        settings["confidence_threshold"] = min(
            0.95,
            settings["confidence_threshold"] + regime_state.confidence_threshold_delta
        )
        # Trades halbieren
        settings["max_trades_per_run"] = max(
            1, int(settings["max_trades_per_run"] * regime_state.max_trades_multiplier)
        )

    # BULL: Settings unverändert

    log.info(
        f"Risk-Settings nach Regime-Anpassung ({regime_state.label}): "
        f"max_pos={settings['max_position_pct']:.0%} | "
        f"min_cash={settings['min_cash_pct']:.0%} | "
        f"conf_threshold={settings['confidence_threshold']:.0%} | "
        f"max_trades={settings['max_trades_per_run']}"
    )
    return settings