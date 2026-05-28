"""
cpo_engine.py – Continuous Portfolio Optimizer (CPO)
======================================================
Regime-aware, volatilitätsadjustiertes Position Sizing mit Momentum-Boosting
und Korrelations-Cluster Exposure Control.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

from logger import log
from config import (
    ACTIVE_RISK_PROFILE, RISK_SETTINGS, SECTOR_CLASSIFICATION,
    VOLATILITY_MULTIPLIERS, MOMENTUM_BOOST_ENABLED, MOMENTUM_BOOST_FACTOR,
    MOMENTUM_STRENGTH_THRESHOLD, CORRELATION_CLUSTERS, MAX_CLUSTER_EXPOSURE,
    CASH_TARGET_BY_REGIME,
)


@dataclass
class AssetData:
    ticker: str
    score: float
    confidence: float
    volatility: float
    momentum: float
    current_weight: float
    sector: str
    is_core: bool = False


class ContinuousPortfolioOptimizer:
    def __init__(self, risk_profile=None):
        self.risk_profile = risk_profile or ACTIVE_RISK_PROFILE
        self.settings = RISK_SETTINGS[self.risk_profile].copy()
        self.max_position_pct = self.settings.get("max_position_pct", 0.20)
        self.core_tickers = {"SPY", "VT", "QQQ", "IVV", "VOO", "VTI"}
        self.regime_multiplier = {"BULL": 1.2, "SIDEWAYS": 1.0, "BEAR": 0.7}
        self.cash_target_by_regime = CASH_TARGET_BY_REGIME

    def _get_volatility_multiplier(self, vol: float) -> float:
        for level, cfg in VOLATILITY_MULTIPLIERS.items():
            if vol <= cfg["max_vol"]:
                return cfg["multiplier"]
        return 0.4

    def _apply_cluster_caps(self, target_weights: Dict[str, float]) -> Dict[str, float]:
        """Reduziert Exposure in Korrelations-Clustern, die das Limit überschreiten."""
        result = target_weights.copy()
        for cluster in CORRELATION_CLUSTERS:
            name = cluster["name"]
            tickers = cluster["tickers"]
            exposure = sum(result.get(t, 0.0) for t in tickers)
            max_exp = MAX_CLUSTER_EXPOSURE.get(name, 0.40)
            if exposure > max_exp:
                scale = max_exp / exposure
                for t in tickers:
                    if t in result:
                        result[t] *= scale
        return result

    def compute_target_weights(
        self,
        assets: Dict[str, AssetData],
        regime: str,
        portfolio_value: float,
    ) -> Dict[str, float]:
        investable = 1.0 - self.cash_target_by_regime.get(regime, 0.10)
        raw = {}

        for ticker, a in assets.items():
            base = a.score / 100.0
            conf_mult = a.confidence
            regime_mult = self.regime_multiplier.get(regime, 1.0)
            vol_mult = self._get_volatility_multiplier(a.volatility)

            # Momentum-Boosting nur im Bullenmarkt
            momentum_boost = 1.0
            if regime == "BULL" and MOMENTUM_BOOST_ENABLED and a.momentum > MOMENTUM_STRENGTH_THRESHOLD:
                boost = 1.0 + (a.momentum / 100.0) * 0.5
                momentum_boost = min(MOMENTUM_BOOST_FACTOR, boost)

            core_bonus = 1.05 if a.is_core else 1.0

            weighted = base * conf_mult * regime_mult * vol_mult * momentum_boost * core_bonus
            raw[ticker] = max(0.01, weighted)

        total_raw = sum(raw.values())
        if total_raw == 0:
            return {t: 0.0 for t in assets}

        target = {t: (w / total_raw) * investable for t, w in raw.items()}

        # Einzelne Positionscaps
        for t in target:
            target[t] = min(target[t], self.max_position_pct)

        # Cluster-Caps
        target = self._apply_cluster_caps(target)

        # Glättung mit aktuellen Gewichten (verhindert Overfitting)
        for t in target:
            current = assets[t].current_weight
            target[t] = 0.7 * target[t] + 0.3 * current

        # Zweite Normalisierung
        total_target = sum(target.values())
        if total_target > investable:
            scale = investable / total_target
            target = {t: w * scale for t, w in target.items()}

        return target

    def determine_clusters(self, tickers: List[str]) -> Dict[str, List[str]]:
        """Gruppiert Ticker in Sektor-Cluster (für Output)."""
        clusters = {}
        for t in tickers:
            sector = SECTOR_CLASSIFICATION.get(t, "other")
            cluster_name = f"{sector.upper()}_CLUSTER"
            clusters.setdefault(cluster_name, []).append(t)
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
        Returns:
            target_weights, cash_target, buy_cluster, sell_cluster, rationale
        """
        # AssetData-Objekte erstellen
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
                sector=sector, is_core=is_core
            )

        target_weights = self.compute_target_weights(assets, regime, portfolio_value)
        cash_target = max(self.cash_target_by_regime.get(regime, 0.10), 1.0 - sum(target_weights.values()))

        # Trade-Intents ableiten
        buy_tickers = []
        sell_tickers = []
        for t, target in target_weights.items():
            current = current_weights.get(t, 0.0)
            if target > current + 0.01:
                buy_tickers.append(t)
            elif target < current - 0.01:
                sell_tickers.append(t)

        buy_cluster = self.determine_clusters(buy_tickers)
        sell_cluster = self.determine_clusters(sell_tickers)

        rationale = f"CPO: Regime {regime}, Cash-Ziel {cash_target:.1%}, " \
                    f"{len(buy_tickers)} Buy-Intents, {len(sell_tickers)} Sell-Intents"

        log.info(f"CPO: {len(target_weights)} Assets, Cash {cash_target:.1%}, "
                 f"Boosts: {MOMENTUM_BOOST_ENABLED and regime=='BULL'}")
        return target_weights, cash_target, buy_cluster, sell_cluster, rationale
