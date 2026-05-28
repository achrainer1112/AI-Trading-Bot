"""
cpo_engine.py – Continuous Portfolio Optimizer (CPO)
======================================================
Zentrales Portfolio-Optimierungsmodul, das keine Einzeltrades generiert,
sondern ein optimiertes Zielportfolio + aggregierte Trade-Intents.

Ausgabe:
- target_weights (pro Asset)
- buy_cluster / sell_cluster (optional nach Sektor gruppiert)
- cash_target
- risk_summary
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

from logger import log
from config import ACTIVE_RISK_PROFILE, RISK_SETTINGS, SECTOR_CLASSIFICATION


@dataclass
class AssetData:
    ticker: str
    score: float          # 0-100
    confidence: float     # 0-1
    volatility: float     # annualisiert (%)
    momentum: float       # 20d return (%)
    current_weight: float
    sector: str
    is_core: bool = False


class ContinuousPortfolioOptimizer:
    """
    Portfolio-Optimierer nach CPO-Prinzipien.
    """

    def __init__(self, risk_profile=None):
        self.risk_profile = risk_profile or ACTIVE_RISK_PROFILE
        self.settings = RISK_SETTINGS[self.risk_profile].copy()
        self.max_position_pct = self.settings.get("max_position_pct", 0.20)
        self.core_tickers = {"SPY", "VT", "QQQ", "IVV", "VOO", "VTI"}

        # Cash-Ziele je Regime (als Bereich, wir nehmen Mittelwert)
        self.cash_target_by_regime = {
            "BULL": 0.10,       # 5-15%
            "SIDEWAYS": 0.15,   # 10-25%
            "BEAR": 0.30,       # 20-40%
        }

        # Regime-Multiplier für Risikobereitschaft
        self.regime_multiplier = {
            "BULL": 1.2,
            "SIDEWAYS": 1.0,
            "BEAR": 0.7,
        }

        # Sektor-Cluster für Aggregation (optional)
        self.sector_clusters = {
            "tech": ["QQQ", "XLK", "AAPL", "MSFT", "NVDA", "AMD"],
            "financial": ["XLF", "JPM", "V", "MA"],
            "healthcare": ["XLV"],
            "energy": ["XLE"],
            "defensive": ["SPY", "VT"],
        }

    def compute_target_weights(
        self,
        assets: Dict[str, AssetData],
        regime: str,
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Berechnet Zielgewichte basierend auf:
        - Base Score
        - Confidence Multiplier (linear: confidence/100)
        - Regime Multiplier
        - Volatility Penalty (1 / (1 + vol/100))
        - Portfolio Smoothing (log)
        """
        # Roh-Scores
        raw = {}
        for ticker, a in assets.items():
            # Base Score
            base = a.score / 100.0
            # Confidence Multiplier (direkt proportional)
            conf_mult = a.confidence
            # Regime Multiplier (für alle Assets gleich)
            regime_mult = self.regime_multiplier.get(regime, 1.0)
            # Volatility Penalty (höhere Vola -> niedrigeres Gewicht)
            vol_penalty = 1.0 / (1.0 + a.volatility / 100.0)
            # Core Bonus (geringer Bonus für Core Holdings)
            core_bonus = 1.05 if a.is_core else 1.0

            weighted_score = base * conf_mult * regime_mult * vol_penalty * core_bonus
            raw[ticker] = max(0.01, weighted_score)

        # Normalisierung auf investierbares Kapital
        total_raw = sum(raw.values())
        if total_raw == 0:
            return {t: 0.0 for t in assets}

        investable = 1.0 - self.cash_target_by_regime.get(regime, 0.10)
        target = {t: (w / total_raw) * investable for t, w in raw.items()}

        # Cap pro Position
        for t in target:
            target[t] = min(target[t], self.max_position_pct)

        # Smoothing: Verhindere extreme Ausreißer (optional)
        for t in target:
            current = assets[t].current_weight
            target[t] = 0.7 * target[t] + 0.3 * current  # Glättung

        # Zweite Normalisierung nach Smoothing
        total_target = sum(target.values())
        if total_target > investable:
            scale = investable / total_target
            target = {t: w * scale for t, w in target.items()}

        return target

    def determine_clusters(
        self,
        trades: List[Dict],
    ) -> Dict[str, List[str]]:
        """
        Gruppiert Trades in Cluster (z. B. TECH_CLUSTER, DEFENSIVE_CLUSTER).
        """
        if not trades:
            return {}

        # Mapping Ticker -> Sektor (aus config)
        sector_of = SECTOR_CLASSIFICATION
        clusters = {}

        for t in trades:
            ticker = t["ticker"]
            sector = sector_of.get(ticker, "other")
            cluster_name = f"{sector.upper()}_CLUSTER"
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(ticker)

        return clusters

    def optimize(
        self,
        scores: Dict[str, float],
        confidences: Dict[str, float],
        momentums: Dict[str, float],
        volatilities: Dict[str, float],
        current_weights: Dict[str, float],
        regime: str,
        portfolio_value: float,
    ) -> Tuple[Dict[str, float], float, Dict[str, List[str]], Dict[str, List[str]], str]:
        """
        Hauptmethode.
        Returns:
            target_weights, cash_target, buy_cluster, sell_cluster, rationale
        """
        # 1. AssetData-Objekte erstellen
        assets = {}
        all_tickers = set(scores.keys()) | set(current_weights.keys())
        for ticker in all_tickers:
            score = scores.get(ticker, 50.0)
            conf = confidences.get(ticker, 0.5)
            mom = momentums.get(ticker, 0.0)
            vol = volatilities.get(ticker, 20.0)
            current = current_weights.get(ticker, 0.0)
            sector = SECTOR_CLASSIFICATION.get(ticker, "other")
            is_core = ticker in self.core_tickers
            assets[ticker] = AssetData(
                ticker=ticker, score=score, confidence=conf,
                volatility=vol, momentum=mom, current_weight=current,
                sector=sector, is_core=is_core,
            )

        # 2. Zielgewichte berechnen
        target_weights = self.compute_target_weights(assets, regime, portfolio_value)

        # 3. Cash-Ziel
        cash_target = max(self.cash_target_by_regime.get(regime, 0.10), 1.0 - sum(target_weights.values()))

        # 4. Trade-Intents (Differenzen)
        buy_tickers = []
        sell_tickers = []
        for ticker, target in target_weights.items():
            current = current_weights.get(ticker, 0.0)
            if target > current + 0.01:   # >1% Aufstockung
                buy_tickers.append(ticker)
            elif target < current - 0.01: # >1% Reduktion
                sell_tickers.append(ticker)

        # 5. Cluster bilden (optional)
        buy_cluster = self.determine_clusters([{"ticker": t} for t in buy_tickers])
        sell_cluster = self.determine_clusters([{"ticker": t} for t in sell_tickers])

        # 6. Rationale
        rationale = f"CPO: Regime {regime}, Cash-Ziel {cash_target:.1%}, "
        rationale += f"{len(buy_tickers)} Buy-Intents, {len(sell_tickers)} Sell-Intents"

        log.info(f"CPO: Target weights for {len(target_weights)} assets, Cash {cash_target:.1%}")
        return target_weights, cash_target, buy_cluster, sell_cluster, rationale
