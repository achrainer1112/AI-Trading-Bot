"""
rebalancing_engine.py – Portfolio Rebalancing Engine
=====================================================
Optimiert das gesamte Portfolio basierend auf relativer Asset-Qualität,
Allokationseffizienz und Transaktionskostenbewusstsein.

Kernprinzipien:
- Keine isolierten Einzelsignale
- Ganzheitliche Portfolio-Optimierung
- Transaktionskosten rechtfertigen jeden Trade
- Swap-Logik bei Cash-Mangel
- Keine starren Ranking-Regeln ("Top 3 kaufen, Bottom 3 verkaufen")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from logger import log
from config import (
    ACTIVE_RISK_PROFILE,
    RISK_SETTINGS,
    SECTOR_CLASSIFICATION,
    CORRELATION_GROUPS,
    TRADE_FRICTION_PCT,
)


@dataclass
class AssetEvaluation:
    """Bewertung eines einzelnen Assets im Portfolio-Kontext."""
    ticker: str
    score: float                    # Qualitäts-Score 0-100
    current_weight: float           # aktuelle Allokation (0-1)
    optimal_weight: float           # implizierte optimale Allokation (wird berechnet)
    underweight: float              # weight deficit (optimal - current, negativ = overweight)
    momentum: float                 # Momentum (z.B. 20d Return)
    volatility: float               # Volatilität (annualisiert)
    sector: str
    transaction_cost_estimate: float  # geschätzte Kosten für einen Trade (in % des Werts)
    is_held: bool = False

    @property
    def quality_score_normalized(self) -> float:
        """Score normalisiert auf 0-1 für Berechnungen."""
        return self.score / 100.0

    @property
    def efficiency_score(self) -> float:
        """
        Effizienz = Qualität / (Volatilität + 0.1) * (1 + Momentum-Bonus)
        Höher = besser
        """
        vol_adj = max(0.05, self.volatility / 100.0 if self.volatility else 0.2)
        momentum_bonus = 1.0 + max(0, self.momentum / 100.0) if self.momentum else 1.0
        return (self.quality_score_normalized / vol_adj) * momentum_bonus


@dataclass
class TradeSuggestion:
    """Einzelne Trade-Empfehlung."""
    ticker: str
    action: str  # BUY, SELL, SWAP_OUT, SWAP_IN
    delta_weight: float  # Änderung der Allokation (positiv = kaufen, negativ = verkaufen)
    expected_improvement: float  # erwartete Verbesserung der Portfolio-Qualität (in %)
    transaction_cost: float  # geschätzte Kosten (in % des Portfoliowerts)
    net_benefit: float  # improvement - cost
    reason: str


@dataclass
class RebalancingProposal:
    """Komplette Rebalancing-Empfehlung."""
    trades: List[TradeSuggestion]
    final_allocations: Dict[str, float]  # Zielallokationen nach Rebalancing
    cash_target: float
    portfolio_quality_before: float
    portfolio_quality_after: float
    total_transaction_cost: float
    net_improvement: float
    rationale: str


class PortfolioRebalancingEngine:
    """
    Haupt-Engine für ganzheitliche Portfolio-Optimierung.
    """

    def __init__(self, risk_profile=None):
        self.risk_profile = risk_profile or ACTIVE_RISK_PROFILE
        self.risk_settings = RISK_SETTINGS[self.risk_profile].copy()
        self.max_position_pct = self.risk_settings.get("max_position_pct", 0.20)
        self.min_cash_pct = self.risk_settings.get("min_cash_pct", 0.10)
        self.max_turnover = self.risk_settings.get("max_daily_turnover", 0.25)
        self.min_trade_improvement = 0.005  # 0.5% Mindestverbesserung für einen Trade

    def optimize(
        self,
        scores: Dict[str, float],           # ticker -> quality score (0-100)
        momentum: Dict[str, float],         # ticker -> momentum (z.B. 20d return %)
        volatility: Dict[str, float],       # ticker -> annualized volatility (%)
        current_weights: Dict[str, float],  # ticker -> aktuelle Allokation (0-1)
        cash: float,                        # absoluter Cash-Betrag
        total_value: float,                 # gesamter Portfoliowert
        regime: str,                        # "BULL", "BEAR", "SIDEWAYS"
        sector_map: Dict[str, str] = None,
        correlation_groups: List[List[str]] = None,
        market_volatility: float = 0.15,    # Marktvolatilität für Kosten-Schätzung
    ) -> RebalancingProposal:
        """
        Hauptmethode: Berechnet optimale Portfolio-Umschichtung.
        """
        sector_map = sector_map or SECTOR_CLASSIFICATION
        correlation_groups = correlation_groups or []

        # 1. Asset-Bewertungen erstellen
        assets = self._evaluate_assets(
            scores, momentum, volatility, current_weights, cash, total_value,
            sector_map, market_volatility
        )

        # 2. Optimale Gewichte berechnen (Quality-Weighted mit Constraints)
        optimal_weights = self._compute_optimal_weights(assets, regime, total_value, sector_map, correlation_groups)

        # 3. Portfolio-Qualität vorher berechnen
        quality_before = self._portfolio_quality(assets, current_weights)

        # 4. Trade-Ideen generieren (Differenz zwischen optimal und aktuell)
        trade_ideas = self._generate_trade_ideas(assets, optimal_weights)

        # 5. Transaktionskosten und Netto-Nutzen berechnen
        trade_ideas = self._evaluate_trade_costs(trade_ideas, assets, total_value, market_volatility)

        # 6. Trades priorisieren und nach Netto-Nutzen filtern
        valid_trades = [t for t in trade_ideas if t.net_benefit > self.min_trade_improvement]

        # 7. Cash-Constraint und Swap-Logik anwenden
        final_trades, final_weights, cash_target = self._apply_cash_constraints(
            valid_trades, assets, optimal_weights, total_value, cash
        )

        # 8. Portfolio-Qualität nachher berechnen
        quality_after = self._portfolio_quality(assets, final_weights)

        total_cost = sum(t.transaction_cost for t in final_trades)
        net_improvement = quality_after - quality_before

        # 9. Rationale generieren
        rationale = self._generate_rationale(assets, optimal_weights, final_trades, net_improvement)

        return RebalancingProposal(
            trades=final_trades,
            final_allocations=final_weights,
            cash_target=cash_target,
            portfolio_quality_before=quality_before,
            portfolio_quality_after=quality_after,
            total_transaction_cost=total_cost,
            net_improvement=net_improvement,
            rationale=rationale,
        )

    def _evaluate_assets(
        self,
        scores: Dict[str, float],
        momentum: Dict[str, float],
        volatility: Dict[str, float],
        current_weights: Dict[str, float],
        cash: float,
        total_value: float,
        sector_map: Dict[str, str],
        market_volatility: float,
    ) -> Dict[str, AssetEvaluation]:
        """Erstellt AssetEvaluation-Objekte für alle relevanten Ticker."""
        assets = {}
        cash_weight = cash / total_value if total_value > 0 else 0

        # Bestehende Positionen
        for ticker, weight in current_weights.items():
            if weight <= 0:
                continue
            assets[ticker] = AssetEvaluation(
                ticker=ticker,
                score=scores.get(ticker, 50.0),
                current_weight=weight,
                optimal_weight=weight,  # vorläufig
                underweight=0.0,
                momentum=momentum.get(ticker, 0.0),
                volatility=volatility.get(ticker, 20.0),
                sector=sector_map.get(ticker, "other"),
                transaction_cost_estimate=self._estimate_transaction_cost(ticker, market_volatility),
                is_held=True,
            )

        # Neue Kandidaten (in Scores, aber nicht im Portfolio)
        for ticker, score in scores.items():
            if ticker not in assets and score >= 40:  # Mindestqualität für Betrachtung
                assets[ticker] = AssetEvaluation(
                    ticker=ticker,
                    score=score,
                    current_weight=0.0,
                    optimal_weight=0.0,
                    underweight=0.0,
                    momentum=momentum.get(ticker, 0.0),
                    volatility=volatility.get(ticker, 20.0),
                    sector=sector_map.get(ticker, "other"),
                    transaction_cost_estimate=self._estimate_transaction_cost(ticker, market_volatility),
                    is_held=False,
                )

        return assets

    def _estimate_transaction_cost(self, ticker: str, market_volatility: float) -> float:
        """
        Schätzt Transaktionskosten für einen Ticker (in % des Ordervolumens).
        Basis: Spread + Slippage + Markteinfluss.
        """
        # Vereinfacht: Grundkosten basierend auf Volatilität und Liquiditätsannahme
        # Bei kleinen Konten höhere relative Kosten
        base_cost = TRADE_FRICTION_PCT  # aus config, z.B. 0.001 (10bps)
        # Zuschlag für höhere Volatilität
        vol_adjustment = max(0.0, (market_volatility - 0.10) * 0.5)
        return base_cost + vol_adjustment

    def _compute_optimal_weights(
        self,
        assets: Dict[str, AssetEvaluation],
        regime: str,
        total_value: float,
        sector_map: Dict[str, str],
        correlation_groups: List[List[str]],
    ) -> Dict[str, float]:
        """
        Berechnet optimale Gewichte basierend auf relativer Qualität,
        unter Berücksichtigung von Regime, Sektorlimits und Korrelation.
        """
        # 1. Roh-Score: Effizienz (Qualität/Volatilität * Momentum-Bonus)
        efficiencies = {t: a.efficiency_score for t, a in assets.items()}
        total_eff = sum(efficiencies.values())
        if total_eff <= 0:
            return {t: 0.0 for t in assets}

        # 2. Roh-Gewichte (proportional zur Effizienz)
        raw_weights = {t: e / total_eff for t, e in efficiencies.items()}

        # 3. Regime-Anpassung
        regime_factor = 1.0
        if regime == "BEAR":
            regime_factor = 0.5  # reduzierte Risikobereitschaft
        elif regime == "BULL":
            regime_factor = 1.2  # erhöhte Risikobereitschaft

        # 4. Max-Position-Limit
        max_weight = self.max_position_pct
        min_weight = 0.0

        # 5. Iterative Anpassung (vereinfachte Wasserfüll-Methode)
        adjusted = {t: min(w * regime_factor, max_weight) for t, w in raw_weights.items()}
        total_adj = sum(adjusted.values())
        investable = 1.0 - self.min_cash_pct
        if total_adj > investable:
            scale = investable / total_adj
            adjusted = {t: w * scale for t, w in adjusted.items()}
        elif total_adj < investable:
            # Verteile Rest proportional auf bestehende
            rest = investable - total_adj
            if adjusted:
                for t in adjusted:
                    adjusted[t] += rest * (efficiencies[t] / total_eff)

        # 6. Sektor-Limits anwenden (vereinfacht)
        sector_limits = {}
        for t, a in assets.items():
            sector = a.sector
            sector_limits[sector] = sector_limits.get(sector, 0.0) + adjusted.get(t, 0.0)
        max_sector_pct = self.risk_settings.get("max_sector_exposure", 0.45)
        over_limit_sectors = [s for s, w in sector_limits.items() if w > max_sector_pct]
        if over_limit_sectors:
            log.info(f"Sektor-Limit Überschreitung: {over_limit_sectors} – reduziere Gewichte")
            for s in over_limit_sectors:
                scale = max_sector_pct / sector_limits[s]
                for t, a in assets.items():
                    if a.sector == s:
                        adjusted[t] *= scale
            # Erneute Normalisierung (vereinfacht)
            total_adj = sum(adjusted.values())
            if total_adj > investable:
                scale = investable / total_adj
                adjusted = {t: w * scale for t, w in adjusted.items()}

        return adjusted

    def _portfolio_quality(self, assets: Dict[str, AssetEvaluation], weights: Dict[str, float]) -> float:
        """Berechnet die gewichtete durchschnittliche Portfolio-Qualität."""
        total_quality = 0.0
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return 0.0
        for ticker, weight in weights.items():
            if ticker in assets:
                total_quality += weight * assets[ticker].quality_score_normalized
        return total_quality / total_weight

    def _generate_trade_ideas(
        self,
        assets: Dict[str, AssetEvaluation],
        optimal_weights: Dict[str, float],
    ) -> List[TradeSuggestion]:
        """Generiert Trade-Ideen aus der Differenz zwischen optimal und aktuell."""
        trades = []
        for ticker, asset in assets.items():
            current = asset.current_weight
            target = optimal_weights.get(ticker, 0.0)
            delta = target - current
            if abs(delta) < 0.001:  # weniger als 0.1% -> ignorieren
                continue

            action = "BUY" if delta > 0 else "SELL"
            # Erwartete Verbesserung: Qualität des Assets * delta (vereinfacht)
            improvement = abs(delta) * (asset.quality_score_normalized - 0.5) * 100  # in Prozentpunkten
            trades.append(TradeSuggestion(
                ticker=ticker,
                action=action,
                delta_weight=delta,
                expected_improvement=improvement,
                transaction_cost=0.0,  # wird später berechnet
                net_benefit=0.0,
                reason=f"Optimal weight {target:.1%} vs current {current:.1%}",
            ))
        return trades

    def _evaluate_trade_costs(
        self,
        trades: List[TradeSuggestion],
        assets: Dict[str, AssetEvaluation],
        total_value: float,
        market_volatility: float,
    ) -> List[TradeSuggestion]:
        """Berechnet Transaktionskosten und Netto-Nutzen für jeden Trade."""
        for t in trades:
            asset = assets.get(t.ticker)
            if not asset:
                cost_rate = self._estimate_transaction_cost(t.ticker, market_volatility)
            else:
                cost_rate = asset.transaction_cost_estimate
            # Kosten in Prozent des Portfoliowerts: delta_weight * cost_rate
            t.transaction_cost = abs(t.delta_weight) * cost_rate * 100  # in Prozentpunkten
            t.net_benefit = t.expected_improvement - t.transaction_cost
        return trades

    def _apply_cash_constraints(
        self,
        trades: List[TradeSuggestion],
        assets: Dict[str, AssetEvaluation],
        optimal_weights: Dict[str, float],
        total_value: float,
        cash: float,
    ) -> Tuple[List[TradeSuggestion], Dict[str, float], float]:
        """
        Wendet Cash-Constraints an, führt Swap-Logik durch und gibt finales Portfolio zurück.
        """
        # Sortiere Trades nach Netto-Nutzen (höchster zuerst)
        trades.sort(key=lambda x: x.net_benefit, reverse=True)

        # Separiere BUYs und SELLs
        sells = [t for t in trades if t.action == "SELL"]
        buys = [t for t in trades if t.action == "BUY"]

        # Berechne freigesetztes Cash durch SELLs
        cash_released = sum(abs(t.delta_weight) * total_value for t in sells)
        cash_available = cash + cash_released
        cash_reserve = total_value * self.min_cash_pct
        spendable = max(0, cash_available - cash_reserve)

        # Kaufe nur so viel, wie spendable erlaubt
        total_buy_delta = sum(t.delta_weight for t in buys)  # positive Summe
        if total_buy_delta * total_value > spendable:
            # Reduziere proportionale Käufe
            scale = spendable / (total_buy_delta * total_value) if total_buy_delta > 0 else 0
            scale = min(1.0, scale)
            for t in buys:
                t.delta_weight *= scale
                t.expected_improvement *= scale
                t.net_benefit = t.expected_improvement - t.transaction_cost

        # Entferne Trades mit net_benefit <= 0
        final_trades = [t for t in trades if t.net_benefit > self.min_trade_improvement]

        # Berechne finale Gewichte
        final_weights = {}
        for ticker, asset in assets.items():
            final_weights[ticker] = asset.current_weight
        for t in final_trades:
            final_weights[t.ticker] = final_weights.get(t.ticker, 0.0) + t.delta_weight
            final_weights[t.ticker] = max(0.0, final_weights[t.ticker])

        # Cash-Ziel
        total_invested = sum(final_weights.values())
        cash_target = max(self.min_cash_pct, 1.0 - total_invested)

        return final_trades, final_weights, cash_target

    def _generate_rationale(
        self,
        assets: Dict[str, AssetEvaluation],
        optimal_weights: Dict[str, float],
        trades: List[TradeSuggestion],
        net_improvement: float,
    ) -> str:
        """Erzeugt eine lesbare Begründung für die vorgeschlagenen Änderungen."""
        if not trades:
            return "No trades: transaction costs exceed expected improvement."

        lines = [f"Portfolio optimization yields {net_improvement:.2f}% net quality improvement."]
        for t in trades:
            if t.action == "BUY":
                lines.append(f"BUY {t.ticker}: +{t.delta_weight:.1%} weight, net benefit {t.net_benefit:.2f}%")
            elif t.action == "SELL":
                lines.append(f"SELL {t.ticker}: {t.delta_weight:.1%} weight, net benefit {t.net_benefit:.2f}%")
        lines.append(f"Transaction cost impact: {sum(t.transaction_cost for t in trades):.2f}%")
        return " ".join(lines)
