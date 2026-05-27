"""
capital_rotator.py – Intelligente Kapitalrotation
==================================================
Wenn ein neues Asset deutlich besser bewertet ist, aber nicht genug Cash verfügbar ist,
prüft das System automatisch, ob eine bestehende Position ersetzt werden sollte.

Regeln:
  - Nur rotieren, wenn Score-Differenz >= 15
  - Oder alte Position negatives Momentum & neue Position positives Momentum
  - Maximal 1–2 Rotationen pro Run
  - Cooldown berücksichtigen
  - Keine Rotation wenn die bestehende Position kürzlich gekauft wurde
  - Keine Rotation bei kleinen Beträgen (unter min_rotation_value)
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from logger import log
from score_engine import ScoreBreakdown


class CapitalRotator:
    """
    Entscheidet, ob ein neues Asset eine bestehende Position verdrängen soll,
    wenn nicht genug Cash vorhanden ist.
    """

    def __init__(
        self,
        min_score_diff: float = 15.0,
        max_rotations_per_run: int = 2,
        min_rotation_value_usd: float = 100.0,
        min_hold_days_before_rotation: int = 5,
    ):
        self.min_score_diff = min_score_diff
        self.max_rotations_per_run = max_rotations_per_run
        self.min_rotation_value_usd = min_rotation_value_usd
        self.min_hold_days_before_rotation = min_hold_days_before_rotation

    def find_rotation_candidates(
        self,
        new_buy_tickers: List[str],
        new_scores: Dict[str, ScoreBreakdown],
        current_positions: Dict[str, Dict],
        total_value: float,
        market_data: Dict,
        regime_state=None,
        cooldown_tickers: set = None,
    ) -> List[Tuple[str, str, float, str]]:
        """
        Für jeden neuen BUY-Kandidaten prüfen, ob er eine bestehende Position ersetzen sollte.
        Returns: Liste von (sell_ticker, buy_ticker, score_diff, reason)
        """
        if not new_buy_tickers or not current_positions:
            return []

        rotations = []
        cooldown_tickers = cooldown_tickers or set()

        # Bestehende Positionen nach Score sortieren (schlechteste zuerst)
        existing_scores = []
        for ticker, pos in current_positions.items():
            if pos.get("market_value", 0) <= 0:
                continue
            # Cooldown-Check: kürzlich gekaufte Positionen nicht rotieren
            entry_date = pos.get("entry_date")
            if entry_date:
                try:
                    entry_dt = datetime.fromisoformat(entry_date)
                    if (datetime.now() - entry_dt).days < self.min_hold_days_before_rotation:
                        log.debug(f"{ticker}: zu jung (<{self.min_hold_days_before_rotation}d), überspringe Rotation")
                        continue
                except Exception:
                    pass

            sb = new_scores.get(ticker)
            if sb is None:
                continue
            existing_scores.append((
                ticker,
                sb.total_score,
                pos.get("market_value", 0),
                sb.momentum_20d or 0,
                sb.current_price or 0,
                pos.get("entry_date", "")
            ))
        existing_scores.sort(key=lambda x: x[1])  # aufsteigend = schlechtester Score zuerst

        if not existing_scores:
            return []

        for buy_ticker in new_buy_tickers:
            if buy_ticker in current_positions:
                continue  # bereits gehalten
            buy_sb = new_scores.get(buy_ticker)
            if not buy_sb:
                continue

            # Prüfe, ob Kauf überhaupt sinnvoll (Score ausreichend)
            if buy_sb.total_score < 50:
                continue

            for (sell_ticker, sell_score, sell_value, sell_momentum, sell_price, sell_entry) in existing_scores:
                if sell_ticker == buy_ticker:
                    continue
                if sell_value < self.min_rotation_value_usd:
                    continue

                score_diff = buy_sb.total_score - sell_score
                rotation_reason = None

                # Hauptbedingung: Score-Differenz groß genug
                if score_diff >= self.min_score_diff:
                    rotation_reason = f"ScoreDiff={score_diff:.0f}"
                # Alternative: alte Position negatives Momentum, neue positives Momentum
                elif sell_momentum < 0 and (buy_sb.momentum_20d or 0) > 0:
                    rotation_reason = f"Momentum: {sell_ticker} {sell_momentum:+.1f}% → {buy_ticker} {buy_sb.momentum_20d:+.1f}%"
                # Regime BULL zusätzlich erlaubt kleinere Differenz
                elif regime_state and getattr(regime_state, 'regime', None) == Regime.BULL and score_diff >= 10:
                    rotation_reason = f"BULL Regime + ScoreDiff={score_diff:.0f}"
                else:
                    continue

                rotations.append((sell_ticker, buy_ticker, score_diff, rotation_reason))
                break  # für diesen buy_ticker nur eine Rotation vorschlagen

        # Nach Score-Differenz sortieren (größte zuerst), limitieren
        rotations.sort(key=lambda x: x[2], reverse=True)
        rotations = rotations[:self.max_rotations_per_run]

        # Cooldown-Filter (vermeide Rotation von Ticker, der gerade erst verkauft wurde)
        filtered = []
        for sell_ticker, buy_ticker, diff, reason in rotations:
            if sell_ticker in cooldown_tickers:
                log.debug(f"{sell_ticker} im Cooldown, überspringe Rotation")
                continue
            filtered.append((sell_ticker, buy_ticker, diff, reason))

        if filtered:
            log.info(f"Capital Rotator: {len(filtered)} Rotation(en) vorgeschlagen")
            for sell, buy, diff, reason in filtered:
                log.info(f"  → {sell} → {buy} (Diff: {diff:.0f}, {reason})")

        return filtered


# Hilfsimport für Regime – muss später aufgelöst werden (vermeidet zirkulären Import)
try:
    from market_regime import Regime
except ImportError:
    class Regime:
        BULL = "bull"
