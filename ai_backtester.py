"""
AI Trading Bot – KI-Backtester
================================
Testet die ECHTE KI-Strategie mit historischen Daten.
Im Gegensatz zum regelbasierten Backtester wird hier die tatsächliche
AIAnalyzer.analyze()-Funktion aufgerufen – inklusive aller Prompts,
Regime-Logik und Risk-Manager-Regeln.

KOSTEN-WARNUNG:
  Jeder simulierte Run kostet einen OpenAI API-Call.
  Beispiel: 2 Jahre wöchentlich = ~104 Calls ≈ $2–5 bei gpt-4o-mini.
  Mit --dry-run werden Calls durch die regelbasierte Fallback-Analyse ersetzt.

VERWENDUNG:
  python ai_backtester.py                        # 2-Jahres-Backtest mit KI
  python ai_backtester.py --dry-run              # Ohne OpenAI (Fallback-Logik)
  python ai_backtester.py --start 2023-01-01 --end 2024-01-01
  python ai_backtester.py --weekly --capital 50000
  python ai_backtester.py --tickers SPY QQQ AAPL MSFT --weekly

UNTERSCHIED ZUM ALTEN BACKTESTER:
  backtester.py       → testet SMA20/SMA60 Crossover-Strategie
  ai_backtester.py    → testet was du wirklich handelst (KI + Regime + RiskManager)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from logger import log
from config import (
    BACKTEST_START_DATE, BACKTEST_END_DATE,
    BACKTEST_INITIAL_CAPITAL, BACKTEST_COMMISSION,
    ACTIVE_RISK_PROFILE, RISK_SETTINGS, FULL_WATCHLIST,
    RiskProfile,
)
from ai_analysis import AIAnalyzer
from risk_manager import RiskManager
from score_engine import ScoreEngine
from market_regime import MarketRegimeDetector, RegimeState, Regime
from utils import calculate_sharpe_ratio, calculate_max_drawdown, format_currency

try:
    import yfinance as yf
except ImportError:
    log.error("yfinance nicht installiert: pip install yfinance")
    raise


# ─────────────────────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────────────────────

# Wie oft pro Woche simuliert der Bot einen Entscheidungs-Run?
# "weekly" = 1x pro Woche (Mo), "daily" = jeden Handelstag (sehr teuer!)
DEFAULT_FREQUENCY = "weekly"

# Mindest-Order in USD (wie in config.py)
MIN_ORDER_VALUE = 100.0

# Backtest-Konfiguration
USE_LLM_IN_BACKTEST = False
WALK_FORWARD_MODE = False
LLM_TIMEOUT_SECONDS = 20
MAX_API_RETRIES = 1
ENABLE_SCORE_PARALLELISM = False
BENCHMARK_TICKERS = ["SPY", "QQQ"]


# ─────────────────────────────────────────────────────────────
# SIMULIERTER PORTFOLIO-STATE
# ─────────────────────────────────────────────────────────────

@dataclass
class SimPortfolio:
    """
    Leichtgewichtiger Portfolio-State für den Backtest.
    Keine Broker-Anbindung, kein Disk-I/O – alles im Speicher.
    """
    cash: float
    initial_capital: float
    positions: Dict[str, Dict] = field(default_factory=dict)
    # positions[ticker] = {"qty": float, "avg_price": float}

    def total_value(self, prices: Dict[str, float]) -> float:
        invested = sum(
            self.positions[t]["qty"] * prices.get(t, 0)
            for t in self.positions
        )
        return self.cash + invested

    def market_value(self, ticker: str, prices: Dict[str, float]) -> float:
        pos = self.positions.get(ticker)
        if not pos:
            return 0.0
        return pos["qty"] * prices.get(ticker, 0)

    def allocation(self, ticker: str, prices: Dict[str, float]) -> float:
        tv = self.total_value(prices)
        if tv <= 0:
            return 0.0
        return self.market_value(ticker, prices) / tv

    def as_summary(self, prices: Dict[str, float]) -> Dict:
        """Portfolio-Summary im Format das AIAnalyzer.build_prompt() erwartet."""
        tv = self.total_value(prices)
        invested = tv - self.cash
        pnl = tv - self.initial_capital
        pnl_pct = pnl / self.initial_capital * 100 if self.initial_capital else 0

        pos_summary = {}
        for ticker, pos in self.positions.items():
            price = prices.get(ticker, pos["avg_price"])
            mv = pos["qty"] * price
            unrealized = mv - pos["qty"] * pos["avg_price"]
            pos_summary[ticker] = {
                "quantity":          pos["qty"],
                "avg_price":         pos["avg_price"],
                "current_price":     price,
                "market_value":      mv,
                "unrealized_pnl":    unrealized,
                "unrealized_pnl_pct": (unrealized / (pos["qty"] * pos["avg_price"]) * 100
                                       if pos["qty"] * pos["avg_price"] > 0 else 0),
            }

        return {
            "total_value":   tv,
            "cash":          self.cash,
            "cash_pct":      self.cash / tv if tv > 0 else 1.0,
            "invested":      invested,
            "initial_capital": self.initial_capital,
            "pnl":           pnl,
            "pnl_pct":       pnl_pct,
            "n_positions":   len(self.positions),
            "positions":     pos_summary,
        }


# ─────────────────────────────────────────────────────────────
# HISTORISCHER MARKT-DATEN-PROVIDER
# ─────────────────────────────────────────────────────────────

class HistoricalDataProvider:
    """
    Lädt historische OHLCV-Daten einmalig und liefert dann
    datumsgenaue Snapshots für die Simulation.
    """

    def __init__(
        self,
        trade_tickers: List[str],
        start_date: str,
        end_date: str,
        benchmark_tickers: Optional[List[str]] = None,
    ):
        self.trade_tickers = trade_tickers
        self.benchmark_tickers = benchmark_tickers or []
        self.tickers = list(dict.fromkeys([
            "^VIX",
            *self.trade_tickers,
            *self.benchmark_tickers,
        ]))

        # Extra Puffer für Indikatoren (SMA90 braucht 90 Tage Vorlauf)
        extended_start = (
            datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=120)
        ).strftime("%Y-%m-%d")

        self.start_date = start_date
        self.end_date = end_date
        self.price_data: Dict[str, pd.Series] = {}   # ticker → Close-Serie
        self._metrics_cache: Dict[pd.Timestamp, Dict[str, Dict]] = {}
        self._load(extended_start)

    def _load(self, extended_start: str):
        log.info(f"Lade historische Preisdaten für {len(self.tickers)} Ticker...")
        loaded = 0
        for ticker in self.tickers:
            try:
                df = yf.download(
                    ticker,
                    start=extended_start,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True,
                )
                if df.empty:
                    log.warning(f"  Keine Daten: {ticker}")
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                self.price_data[ticker] = df["Close"].dropna()
                loaded += 1
            except Exception as e:
                log.warning(f"  Fehler bei {ticker}: {e}")

        log.info(f"  {loaded}/{len(self.tickers)} Ticker geladen.")

    def trading_dates(self) -> List[pd.Timestamp]:
        """Alle Handelstage im Backtest-Zeitraum (sortiert)."""
        all_dates: set = set()
        for ticker in self.trade_tickers:
            series = self.price_data.get(ticker)
            if series is None:
                continue
            all_dates.update(series.index)
        return sorted(
            d for d in all_dates
            if self.start_date <= str(d.date()) <= self.end_date
        )

    def prices_at(self, date: pd.Timestamp) -> Dict[str, float]:
        """Schlusskurse an einem bestimmten Datum."""
        result = {}
        for ticker, series in self.price_data.items():
            if date in series.index:
                p = float(series[date])
                if p > 0:
                    result[ticker] = p
        return result

    def market_metrics_at(self, ticker: str, date: pd.Timestamp) -> Dict:
        """
        Berechnet Marktmetriken für einen Ticker am Backtest-Datum.
        Repliziert MarketDataCollector.calculate_metrics() mit historischen Daten.
        Nur Daten die VOR 'date' lagen werden genutzt (kein Look-Ahead!).
        """
        series = self.price_data.get(ticker)
        if series is None:
            return {}

        # Nur Daten bis zum Datum (exklusiv future)
        mask   = series.index <= date
        close  = series[mask]

        if len(close) < 10:
            return {}

        current_price = float(close.iloc[-1])

        def safe_ret(n):
            if len(close) > n:
                return float((close.iloc[-1] / close.iloc[-n - 1] - 1) * 100)
            return None

        daily_rets = close.pct_change().dropna()
        volatility = float(daily_rets.std() * np.sqrt(252) * 100) if len(daily_rets) >= 5 else None

        sma7  = float(close.tail(7).mean())
        sma30 = float(close.tail(30).mean()) if len(close) >= 30 else None
        sma90 = float(close.tail(90).mean()) if len(close) >= 90 else None

        # RSI (14)
        rsi = None
        if len(close) >= 15:
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            rsi_s = 100 - (100 / (1 + rs))
            rsi   = float(rsi_s.iloc[-1]) if not rsi_s.empty else None

        return {
            "ticker":                ticker,
            "current_price":         current_price,
            "return_7d":             safe_ret(7),
            "return_30d":            safe_ret(30),
            "return_90d":            safe_ret(90),
            "volatility_annual_pct": round(volatility, 2) if volatility else None,
            "sma_7":                 round(sma7, 2),
            "sma_30":                round(sma30, 2) if sma30 else None,
            "sma_90":                round(sma90, 2) if sma90 else None,
            "rsi_14":                round(rsi, 2)  if rsi   else None,
            "above_sma_30":          current_price > sma30 if sma30 else None,
            "above_sma_90":          current_price > sma90 if sma90 else None,
            "data_points":           len(close),
            "last_updated":          date.isoformat(),
        }

    def all_metrics_at(self, date: pd.Timestamp) -> Dict[str, Dict]:
        """Metriken für alle verfügbaren Ticker an einem Datum."""
        if date in self._metrics_cache:
            return self._metrics_cache[date]

        metrics = {
            t: m
            for t in self.price_data
            if (m := self.market_metrics_at(t, date))
        }
        self._metrics_cache[date] = metrics
        return metrics

    def spy_close_at(self, date: pd.Timestamp) -> Optional[float]:
        """SPY-Kurs an einem Datum (für simulierte Regime-Detektion)."""
        series = self.price_data.get("SPY")
        if series is None:
            return None
        mask = series.index <= date
        s    = series[mask]
        return float(s.iloc[-1]) if not s.empty else None


# ─────────────────────────────────────────────────────────────
# HISTORISCHE REGIME-DETEKTION
# ─────────────────────────────────────────────────────────────

def detect_regime_at(
    data_provider: HistoricalDataProvider,
    date: pd.Timestamp,
) -> Optional[RegimeState]:
    """
    Simuliert MarketRegimeDetector.detect() für ein historisches Datum.
    Nutzt nur Daten bis 'date' (kein Look-Ahead).

    Vereinfachte Variante: SPY-SMA-Crossover + einfacher Momentum-Score.
    Der echte VIX-Abruf wird durch die historischen VIX-Daten ersetzt.
    """
    spy_series = data_provider.price_data.get("SPY")
    vix_series = data_provider.price_data.get("^VIX")

    scores: List[float] = []
    weights: List[float] = []

    # VIX
    vix = None
    if vix_series is not None:
        mask = vix_series.index <= date
        vix_s = vix_series[mask]
        if not vix_s.empty:
            vix = float(vix_s.iloc[-1])
            if vix <= 15:
                vix_score = 1.0
            elif vix <= 20:
                vix_score = 1.0 - (vix - 15) / 5
            elif vix <= 30:
                vix_score = -((vix - 20) / 10)
            else:
                vix_score = -1.0
            scores.append(vix_score)
            weights.append(2.0)

    # SPY-Trend
    spy_return_20d = spy_return_60d = None
    trend_score    = None
    spy_above_sma20 = spy_above_sma50 = None

    if spy_series is not None:
        mask    = spy_series.index <= date
        spy_s   = spy_series[mask]
        if len(spy_s) >= 60:
            sma20   = float(spy_s.tail(20).mean())
            sma50   = float(spy_s.tail(50).mean())
            current = float(spy_s.iloc[-1])
            spy_above_sma20 = current > sma20 * 1.005
            spy_above_sma50 = current > sma50 * 1.005
            spy_return_20d  = float((current / spy_s.iloc[-20] - 1) * 100)
            spy_return_60d  = float((current / spy_s.iloc[-60] - 1) * 100)

            trend_score = 0.0
            trend_score += 0.4 if spy_above_sma20 else -0.4
            trend_score += 0.6 if spy_above_sma50 else -0.6
            scores.append(trend_score)
            weights.append(2.5)

    # Momentum
    momentum_score = None
    if spy_return_20d is not None and spy_return_60d is not None:
        mom_20 = float(np.clip(spy_return_20d / 10.0, -1.0, 1.0))
        mom_60 = float(np.clip(spy_return_60d / 15.0, -1.0, 1.0))
        momentum_score = 0.4 * mom_20 + 0.6 * mom_60
        scores.append(momentum_score)
        weights.append(1.5)

    if not scores:
        return None

    score = float(np.average(scores, weights=weights))

    if score > 0.35:
        regime     = Regime.BULL
        confidence = min(0.92, 0.60 + (score - 0.35) * 0.8)
        desc       = f"Score={score:+.2f}: Aufwärtstrend"
        conf_delta = 0.0
        trade_mult = 1.0
    elif score < -0.35:
        regime     = Regime.BEAR
        confidence = min(0.92, 0.60 + (-score - 0.35) * 0.8)
        desc       = f"Score={score:+.2f}: Abwärtstrend"
        conf_delta = +0.15
        trade_mult = 0.70
    else:
        regime     = Regime.SIDEWAYS
        confidence = min(0.90, 0.50 + (0.35 - abs(score)) * 1.4)
        desc       = f"Score={score:+.2f}: Kein klarer Trend"
        conf_delta = +0.10
        trade_mult = 0.50

    state = RegimeState(
        regime=regime,
        confidence=confidence,
        vix=vix,
        spy_above_sma20=spy_above_sma20,
        spy_above_sma50=spy_above_sma50,
        spy_return_20d=spy_return_20d,
        spy_return_60d=spy_return_60d,
        momentum_score=momentum_score,
        description=desc,
        detected_at=date.isoformat(),
    )
    state.confidence_threshold_delta = conf_delta
    state.max_trades_multiplier      = trade_mult
    return state


def deterministic_decision_engine(
    scores: Dict[str, "ScoreBreakdown"],
    regime_state: Optional[RegimeState],
    risk_settings: Dict,
) -> List[Dict]:
    """
    Deterministische Entscheidungspipeline für den Backtest.

    Diese Funktion ersetzt die LLM-Interpretation im Standardmodus
    und wird bei LLM-Fehlern als Fallback verwendet.
    """
    decisions: List[Dict] = []
    min_buy_score = max(65, int(risk_settings.get("min_buy_score", 65)))
    hold_threshold = 50
    max_pos_pct = risk_settings.get("max_position_pct", 0.12)

    for ticker, sb in scores.items():
        score = float(sb.total_score)
        current_alloc = sb.current_alloc

        if score >= min_buy_score:
            if score >= 85:
                target_alloc = min(max_pos_pct, 0.15)
            elif score >= 75:
                target_alloc = min(max_pos_pct, 0.12)
            else:
                target_alloc = min(max_pos_pct, 0.08)
            action = "BUY"
        elif score >= hold_threshold:
            action = "HOLD"
            target_alloc = current_alloc
        else:
            action = "SELL" if current_alloc > 0 else "HOLD"
            target_alloc = 0.0 if current_alloc > 0 else current_alloc

        decisions.append({
            "ticker": ticker,
            "action": action,
            "target_allocation": round(target_alloc, 4),
            "confidence": round(min(1.0, max(0.0, score / 100.0)), 3),
            "quant_score": round(score, 1),
            "llm_score_adj": 0.0,
            "reasoning": {
                "regime": regime_state.label if regime_state is not None else "UNKNOWN",
                "score": round(score, 1),
                "current_alloc": round(current_alloc, 4),
            },
            "reason": (
                f"Deterministic rule: score={score:.1f}, "
                f"action={action}, target_alloc={target_alloc:.0%}"
            ),
        })

    return decisions


# ─────────────────────────────────────────────────────────────
# HAUPT-BACKTESTER
# ─────────────────────────────────────────────────────────────

class AIBacktester:
    """
    Backtester der die echte KI-Entscheidungslogik simuliert.

    Ablauf pro simuliertem Run (wöchentlich/täglich):
      1. Historische Marktdaten für dieses Datum laden (kein Look-Ahead)
      2. Regime-Detektion mit historischen Daten
      3. AI-Analyse aufrufen (echt oder Fallback)
      4. Risk-Manager-Regeln anwenden
      5. Trades simulieren (mit Transaktionskosten)
      6. Portfolio-State aktualisieren
    """

    def __init__(
        self,
        tickers: List[str] = None,
        start_date: str = BACKTEST_START_DATE,
        end_date: str   = BACKTEST_END_DATE,
        initial_capital: float = BACKTEST_INITIAL_CAPITAL,
        commission: float = BACKTEST_COMMISSION,
        frequency: str = DEFAULT_FREQUENCY,   # "weekly" oder "daily"
        risk_profile: RiskProfile = ACTIVE_RISK_PROFILE,
        use_ai: bool = USE_LLM_IN_BACKTEST,
        walk_forward: bool = WALK_FORWARD_MODE,
        use_multiprocessing: bool = ENABLE_SCORE_PARALLELISM,
        verbose: bool = False,
    ):
        self.trade_tickers    = tickers or FULL_WATCHLIST
        self.start_date      = start_date
        self.end_date        = end_date
        self.initial_capital = initial_capital
        self.commission      = commission
        self.frequency       = frequency
        self.risk_profile    = risk_profile
        self.use_ai          = use_ai
        self.walk_forward    = walk_forward
        self.use_multiprocessing = use_multiprocessing
        self.verbose         = verbose
        self.benchmark_tickers = [t for t in BENCHMARK_TICKERS if t not in self.trade_tickers]

        # Basis-Ticker für den Backtest laden, zusätzlich Benchmark-Assets.
        self.loaded_tickers = list(dict.fromkeys([
            "^VIX",
            *self.trade_tickers,
            *self.benchmark_tickers,
        ]))

        # Module initialisieren
        self.ai_analyzer  = AIAnalyzer()
        self.risk_manager = RiskManager(risk_profile)
        self.data_provider: Optional[HistoricalDataProvider] = None

        log.info(f"KI-Backtester initialisiert | {start_date} → {end_date}")
        log.info(f"  Frequenz: {frequency} | Kapital: {format_currency(initial_capital)}")
        log.info(f"  KI-Modus: {'AKTIV (OpenAI)' if use_ai else 'FALLBACK (regelbasiert)'}")
        log.info(f"  Handelbare Ticker: {len(self.trade_tickers)} | Benchmarks: {', '.join(self.benchmark_tickers)}")

    def run(self) -> Dict:
        """
        Führt den KI-Backtest durch.
        Gibt ein vollständiges Ergebnis-Dictionary zurück.
        """
        log.info("=" * 60)
        log.info("KI-BACKTEST GESTARTET")
        log.info("=" * 60)

        # Daten laden
        self.data_provider = HistoricalDataProvider(
            trade_tickers=self.trade_tickers,
            start_date=self.start_date,
            end_date=self.end_date,
            benchmark_tickers=self.benchmark_tickers,
        )

        if self.walk_forward:
            return self._run_walk_forward()
        return self._run_backtest()

    def _run_backtest(self) -> Dict:
        all_dates = self.data_provider.trading_dates()
        run_dates = self._select_run_dates(all_dates)
        value_series = []
        daily_returns = []
        cash_exposure = []
        all_trades = []
        run_logs = []

        portfolio = SimPortfolio(
            cash=self.initial_capital,
            initial_capital=self.initial_capital,
        )

        prev_value = self.initial_capital
        api_calls = 0

        log.info(f"Gesamt Handelstage: {len(all_dates)} | Simulations-Runs: {len(run_dates)}")

        for date in all_dates:
            prices = self.data_provider.prices_at(date)
            if not prices:
                continue

            tv = portfolio.total_value(prices)
            value_series.append((date, tv))
            cash_exposure.append(portfolio.cash / tv if tv > 0 else 1.0)

            if prev_value > 0:
                daily_returns.append((tv - prev_value) / prev_value)
            prev_value = tv

            if date in run_dates:
                try:
                    trades, n_api = self._simulate_run(date, portfolio, prices)
                except Exception as e:
                    log.error(f"Backtest iteration failed: {e}")
                    continue
                all_trades.extend(trades)
                api_calls += n_api
                run_logs.append({
                    "date": str(date.date()),
                    "value": round(tv, 2),
                    "trades": len(trades),
                })

        if not value_series:
            log.error("Keine Daten für Backtest.")
            return {}

        final_prices = self.data_provider.prices_at(all_dates[-1])
        final_value = portfolio.total_value(final_prices)
        pv = [v for _, v in value_series]

        total_return = (final_value - self.initial_capital) / self.initial_capital
        trading_days = len(pv)
        ann_return = self._annualized_return(self.initial_capital, final_value, trading_days)
        max_dd = calculate_max_drawdown(pv)
        sharpe = calculate_sharpe_ratio(daily_returns)

        trade_stats = self._calculate_trade_statistics(all_trades, value_series, cash_exposure)
        benchmark_scores = self._benchmark_metrics()
        sector_exposure = self.risk_manager._sector_exposure(portfolio.positions, final_value) if final_value > 0 else {}

        result = {
            "strategy": "AI" if self.use_ai else "AI-Fallback",
            "risk_profile": self.risk_profile.value,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "frequency": self.frequency,
            "initial_capital": self.initial_capital,
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return * 100, 2),
            "annualized_return_pct": round(ann_return * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "total_trades": len(all_trades),
            "buy_trades": sum(1 for t in all_trades if t["action"] == "BUY"),
            "sell_trades": sum(1 for t in all_trades if t["action"] == "SELL"),
            "simulation_runs": len(run_dates),
            "api_calls_made": api_calls,
            "trading_days": trading_days,
            "run_log": run_logs,
            "trades": all_trades,
            "benchmarks": benchmark_scores,
            "sector_exposure": sector_exposure,
            **trade_stats,
        }

        self._print_results(result)
        return result

    def _run_walk_forward(self) -> Dict:
        windows = self._build_walk_forward_windows()
        if not windows:
            log.warning("Walk-forward kann nicht ausgeführt werden: nicht genügend Daten.")
            return self._run_backtest()

        results = []
        for idx, window in enumerate(windows, start=1):
            log.info(
                f"Walk-Forward Window {idx}: "
                f"Train {window['train_start']}→{window['train_end']} | "
                f"Test {window['test_start']}→{window['test_end']}"
            )
            period_result = self._run_period(window)
            results.append(period_result)

        aggregated = {
            "strategy": "AI-WalkForward" if self.use_ai else "AI-Fallback-WalkForward",
            "risk_profile": self.risk_profile.value,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "windows": results,
            "window_count": len(results),
        }
        return aggregated

    def _run_period(self, window: Dict) -> Dict:
        period_start = datetime.strptime(window["test_start"], "%Y-%m-%d")
        period_end = datetime.strptime(window["test_end"], "%Y-%m-%d")
        all_dates = [d for d in self.data_provider.trading_dates() if period_start <= d <= period_end]
        run_dates = self._select_run_dates(all_dates)

        value_series = []
        daily_returns = []
        cash_exposure = []
        all_trades = []

        portfolio = SimPortfolio(
            cash=self.initial_capital,
            initial_capital=self.initial_capital,
        )

        prev_value = self.initial_capital
        api_calls = 0

        for date in all_dates:
            prices = self.data_provider.prices_at(date)
            if not prices:
                continue

            tv = portfolio.total_value(prices)
            value_series.append((date, tv))
            cash_exposure.append(portfolio.cash / tv if tv > 0 else 1.0)

            if prev_value > 0:
                daily_returns.append((tv - prev_value) / prev_value)
            prev_value = tv

            if date in run_dates:
                try:
                    trades, n_api = self._simulate_run(date, portfolio, prices)
                except Exception as e:
                    log.error(f"Walk-Forward iteration failed: {e}")
                    continue
                all_trades.extend(trades)
                api_calls += n_api

        final_prices = self.data_provider.prices_at(all_dates[-1]) if all_dates else {}
        final_value = portfolio.total_value(final_prices)
        pv = [v for _, v in value_series]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        trading_days = len(pv)
        ann_return = self._annualized_return(self.initial_capital, final_value, trading_days)
        max_dd = calculate_max_drawdown(pv)
        sharpe = calculate_sharpe_ratio(daily_returns)
        trade_stats = self._calculate_trade_statistics(all_trades, value_series, cash_exposure)
        benchmark_scores = self._benchmark_metrics(window["test_start"], window["test_end"])
        sector_exposure = self.risk_manager._sector_exposure(portfolio.positions, final_value) if final_value > 0 else {}

        return {
            "window_name": window["name"],
            "train_start": window["train_start"],
            "train_end": window["train_end"],
            "test_start": window["test_start"],
            "test_end": window["test_end"],
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return * 100, 2),
            "annualized_return_pct": round(ann_return * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "api_calls_made": api_calls,
            "total_trades": len(all_trades),
            "buy_trades": sum(1 for t in all_trades if t["action"] == "BUY"),
            "sell_trades": sum(1 for t in all_trades if t["action"] == "SELL"),
            "trading_days": trading_days,
            "benchmarks": benchmark_scores,
            "sector_exposure": sector_exposure,
            **trade_stats,
        }

    def _build_walk_forward_windows(self) -> List[Dict]:
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        windows = []
        current_train_start = start

        while True:
            train_end = current_train_start + pd.DateOffset(months=6)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=3)
            if test_end > end:
                test_end = end
            if test_start >= end or test_start >= test_end:
                break

            windows.append({
                "name": f"WF_{len(windows)+1}",
                "train_start": current_train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
            })
            current_train_start = test_end
            if current_train_start >= end:
                break

        return windows

    def _score_all(
        self,
        positions: Dict[str, Dict],
        total_value: float,
        market_data: Dict[str, Dict],
        regime_state,
        spy_return_20d: Optional[float],
    ) -> Dict[str, "ScoreBreakdown"]:
        if not self.use_multiprocessing:
            engine = ScoreEngine(positions=positions, total_value=total_value)
            return engine.score_all(market_data, regime_state, spy_return_20d)

        args = [
            (ticker, data, positions, total_value, regime_state, spy_return_20d)
            for ticker, data in market_data.items()
        ]
        results: Dict[str, "ScoreBreakdown"] = {}
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as executor:
            futures = [executor.submit(self._score_ticker_worker, arg) for arg in args]
            for future in futures:
                try:
                    ticker, sb = future.result(timeout=15)
                    results[ticker] = sb
                except FuturesTimeoutError:
                    log.warning("Score-Berechnung Timeout für einen Ticker – verwende Default-Score.")
                except Exception as e:
                    log.warning(f"Score-Berechnung Fehler: {e}")
        return results

    @staticmethod
    def _score_ticker_worker(args):
        ticker, data, positions, total_value, regime_state, spy_return_20d = args
        engine = ScoreEngine(positions=positions, total_value=total_value)
        return ticker, engine.score_ticker(ticker, data, regime_state, spy_return_20d)

    def _safe_ai_decision(
        self,
        date: pd.Timestamp,
        portfolio_summary: Dict,
        market_data: Dict[str, Dict],
        regime_state,
        scores: Dict[str, "ScoreBreakdown"],
    ) -> Tuple[List[Dict], int]:
        api_calls = 0
        watchlist = [t for t in self.trade_tickers if t in market_data and t != "^VIX"]
        if not self.use_ai or not self.ai_analyzer.client:
            filtered_scores = {t: scores[t] for t in watchlist if t in scores}
            return deterministic_decision_engine(filtered_scores, regime_state, self.risk_manager.settings), api_calls

        api_calls = 1
        try:
            ai_result = self.ai_analyzer.analyze(
                portfolio_summary=portfolio_summary,
                market_data=market_data,
                news_text=f"[Historische Simulation – {date.date()} – Keine Echtzeit-News]",
                watchlist=watchlist,
                journal_entries=None,
                regime_state=regime_state,
            )
            if not self._is_valid_ai_decision_result(ai_result):
                raise ValueError("Ungültiges LLM-Ergebnis")
            decisions = ai_result.get("decisions", [])
            if not decisions:
                raise ValueError("Leere LLM-Entscheidungsliste")
            return decisions, api_calls
        except Exception as e:
            log.warning("LLM failed → deterministic fallback activated")
            log.warning(f"  Grund: {e}")
            filtered_scores = {t: scores[t] for t in watchlist if t in scores}
            return deterministic_decision_engine(filtered_scores, regime_state, self.risk_manager.settings), api_calls

    def _is_valid_ai_decision_result(self, ai_result: Dict) -> bool:
        if not isinstance(ai_result, dict):
            return False
        decisions = ai_result.get("decisions")
        if not isinstance(decisions, list):
            return False
        for decision in decisions:
            if not isinstance(decision, dict):
                return False
            if "ticker" not in decision or "action" not in decision:
                return False
        return True

    def _benchmark_metrics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Dict]:
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        result: Dict[str, Dict] = {}
        for ticker in BENCHMARK_TICKERS:
            series = self.data_provider.price_data.get(ticker)
            if series is None or series.empty:
                continue
            try:
                start = series[series.index >= pd.Timestamp(start_date)].iloc[0]
                end = series[series.index <= pd.Timestamp(end_date)].iloc[-1]
            except Exception:
                continue
            total_return = (end - start) / start if start else 0.0
            days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days or 1
            result[ticker] = {
                "start_price": round(float(start), 2),
                "end_price": round(float(end), 2),
                "total_return_pct": round(total_return * 100, 2),
                "annualized_return_pct": round(((1 + total_return) ** (365 / days) - 1) * 100, 2),
            }
        return result

    def _annualized_return(self, start_value: float, end_value: float, days: int) -> float:
        if days <= 0 or start_value <= 0:
            return 0.0
        total_return = (end_value - start_value) / start_value
        return (1 + total_return) ** (252 / days) - 1

    def _calculate_trade_statistics(
        self,
        trades: List[Dict],
        value_series: List[tuple],
        cash_exposure: List[float],
    ) -> Dict:
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        holding_periods: List[int] = []
        turnover_value = 0.0
        buy_lots = {}

        for trade in sorted(trades, key=lambda x: x.get("date", "")):
            ticker = trade.get("ticker")
            action = trade.get("action")
            qty = trade.get("qty", 0)
            price = trade.get("price", 0)
            fee = trade.get("fee", 0)
            trade_value = abs(trade.get("value", 0))
            turnover_value += trade_value

            if action == "BUY" and qty > 0:
                buy_lots.setdefault(ticker, []).append({
                    "qty": qty,
                    "price": price,
                    "date": datetime.fromisoformat(trade.get("date")),
                })
            elif action == "SELL" and qty > 0:
                remaining = qty
                sell_date = datetime.fromisoformat(trade.get("date"))
                while remaining > 1e-9 and buy_lots.get(ticker):
                    lot = buy_lots[ticker][0]
                    matched = min(remaining, lot["qty"])
                    profit = (price - lot["price"]) * matched - fee * (matched / qty if qty else 0)
                    if profit >= 0:
                        gross_profit += profit
                        wins += 1
                    else:
                        gross_loss += abs(profit)
                        losses += 1
                    holding_periods.append((sell_date - lot["date"]).days)
                    lot["qty"] -= matched
                    if lot["qty"] <= 1e-9:
                        buy_lots[ticker].pop(0)
                    remaining -= matched

        total_trades = wins + losses
        win_rate = (wins / total_trades) if total_trades else 0.0
        profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else None
        avg_holding = round(sum(holding_periods) / len(holding_periods), 1) if holding_periods else 0.0
        avg_portfolio_value = sum(v for _, v in value_series) / max(len(value_series), 1)
        turnover_pct = round((turnover_value / avg_portfolio_value) * 100, 2) if avg_portfolio_value else 0.0
        volatility = round(np.std([r for r in np.diff([v for _, v in value_series]) / np.array([v for _, v in value_series][:-1]) if len(value_series) > 1]) * np.sqrt(252) * 100, 2) if len(value_series) > 1 else 0.0
        avg_cash = round(np.mean(cash_exposure) * 100, 2) if cash_exposure else 0.0

        return {
            "win_rate_pct": round(win_rate * 100, 2),
            "profit_factor": profit_factor,
            "volatility_pct": volatility,
            "cash_exposure_pct": avg_cash,
            "turnover_pct": turnover_pct,
            "avg_holding_days": avg_holding,
        }

    def _simulate_run(
        self,
        date: pd.Timestamp,
        portfolio: SimPortfolio,
        prices: Dict[str, float],
    ) -> Tuple[List[Dict], int]:
        """
        Simuliert einen einzelnen Trading-Run an einem historischen Datum.
        Gibt (ausgeführte Trades, Anzahl API-Calls) zurück.
        """
        # 1. Markt-Metriken für dieses Datum (kein Look-Ahead)
        market_data = self.data_provider.all_metrics_at(date)
        if not market_data:
            return [], 0

        # 2. Regime-Detektion (historisch)
        regime_state = detect_regime_at(self.data_provider, date)
        if regime_state:
            self.risk_manager.apply_regime(regime_state)

        # 3. Portfolio-Summary für KI-Prompt
        portfolio_summary = portfolio.as_summary(prices)

        # 4. Deterministische Score-Berechnung
        positions = portfolio_summary.get("positions", {})
        scores = self._score_all(
            positions=positions,
            total_value=portfolio_summary.get("total_value", 0),
            market_data=market_data,
            regime_state=regime_state,
            spy_return_20d=market_data.get("SPY", {}).get("return_20d"),
        )

        decisions, api_calls = self._safe_ai_decision(
            date=date,
            portfolio_summary=portfolio_summary,
            market_data=market_data,
            regime_state=regime_state,
            scores=scores,
        )

        if not decisions:
            return [], api_calls

        # 5. Risk-Manager: Entscheidungen validieren
        validated, _ = self.risk_manager.validate_decisions(
            decisions=decisions,
            portfolio_summary=portfolio_summary,
            market_data=market_data,
        )

        # 6. Trades simulieren
        executed = self._execute_decisions(validated, portfolio, prices, date)
        return executed, api_calls

    def _execute_decisions(
        self,
        decisions: List[Dict],
        portfolio: SimPortfolio,
        prices: Dict[str, float],
        date: pd.Timestamp,
    ) -> List[Dict]:
        """
        Führt validierte Entscheidungen als simulierte Trades aus.
        Berücksichtigt Transaktionskosten (Kommission).
        """
        executed = []
        total_value = portfolio.total_value(prices)

        # Sells zuerst (Cash freimachen)
        for d in decisions:
            if d.get("action") != "SELL" or not d.get("risk_approved", True):
                continue
            ticker = d["ticker"]
            if ticker not in portfolio.positions:
                continue
            price = prices.get(ticker)
            if not price:
                continue

            pos = portfolio.positions[ticker]
            sell_value = pos["qty"] * price
            fee = sell_value * self.commission

            portfolio.cash += sell_value - fee
            del portfolio.positions[ticker]

            executed.append({
                "date":      str(date.date()),
                "ticker":    ticker,
                "action":    "SELL",
                "qty":       pos["qty"],
                "price":     price,
                "value":     round(sell_value, 2),
                "fee":       round(fee, 4),
                "reason":    d.get("reason", ""),
                "confidence": d.get("confidence", 0),
            })

        # Buys
        settings     = RISK_SETTINGS[self.risk_profile]
        max_pos_pct  = self.risk_manager.settings["max_position_pct"]
        min_cash_pct = self.risk_manager.settings["min_cash_pct"]
        total_value  = portfolio.total_value(prices)  # nach Sells neu berechnen
        min_cash_abs = total_value * min_cash_pct

        for d in decisions:
            if d.get("action") != "BUY" or not d.get("risk_approved", True):
                continue
            ticker = d["ticker"]
            price  = prices.get(ticker)
            if not price or price <= 0:
                continue

            target_alloc  = min(d.get("target_allocation", 0), max_pos_pct)
            target_value  = total_value * target_alloc
            current_value = portfolio.market_value(ticker, prices)
            buy_value     = target_value - current_value

            # Cash-Schutz
            spendable = max(0, portfolio.cash - min_cash_abs)
            buy_value = min(buy_value, spendable * 0.95)

            if buy_value < MIN_ORDER_VALUE:
                continue

            fee    = buy_value * self.commission
            shares = (buy_value - fee) / price
            portfolio.cash -= buy_value

            if ticker in portfolio.positions:
                old = portfolio.positions[ticker]
                new_qty     = old["qty"] + shares
                new_avg     = (old["qty"] * old["avg_price"] + buy_value) / new_qty
                portfolio.positions[ticker] = {"qty": new_qty, "avg_price": new_avg}
            else:
                portfolio.positions[ticker] = {"qty": shares, "avg_price": price}

            executed.append({
                "date":      str(date.date()),
                "ticker":    ticker,
                "action":    "BUY",
                "qty":       round(shares, 6),
                "price":     price,
                "value":     round(buy_value, 2),
                "fee":       round(fee, 4),
                "reason":    d.get("reason", ""),
                "confidence": d.get("confidence", 0),
            })

        return executed

    def _select_run_dates(self, all_dates: List[pd.Timestamp]) -> set:
        """Wählt Dates für Simulations-Runs basierend auf der Frequenz."""
        run_set = set()
        if self.frequency == "weekly":
            # Erster Handelstag jeder Woche (Montag oder nächster Tag)
            seen_weeks = set()
            for d in all_dates:
                week_key = (d.year, d.isocalendar()[1])
                if week_key not in seen_weeks:
                    seen_weeks.add(week_key)
                    run_set.add(d)
        else:  # daily
            run_set = set(all_dates)
        return run_set

    def _print_results(self, result: Dict):
        log.info("\n" + "=" * 60)
        log.info("KI-BACKTEST ERGEBNISSE")
        log.info("=" * 60)
        log.info(f"  Strategie:          {result['strategy']}")
        log.info(f"  Zeitraum:           {result['start_date']} → {result['end_date']}")
        log.info(f"  Startkapital:       {format_currency(result['initial_capital'])}")
        log.info(f"  Endwert:            {format_currency(result['final_value'])}")
        log.info(f"  Gesamtrendite:      {result['total_return_pct']:+.2f}%")
        log.info(f"  Ann. Rendite:       {result['annualized_return_pct']:+.2f}%")
        log.info(f"  Max Drawdown:       {result['max_drawdown_pct']:.2f}%")
        log.info(f"  Sharpe Ratio:       {result['sharpe_ratio']:.3f}")
        log.info(f"  Win Rate:           {result.get('win_rate_pct', 0):.2f}%")
        log.info(f"  Profit Factor:      {result.get('profit_factor', 0) or 0:.2f}")
        log.info(f"  Volatility:         {result.get('volatility_pct', 0):.2f}%")
        log.info(f"  Cash Exposure:      {result.get('cash_exposure_pct', 0):.2f}%")
        log.info(f"  Turnover:           {result.get('turnover_pct', 0):.2f}%")
        log.info(f"  Avg. Haltedauer:    {result.get('avg_holding_days', 0):.1f} Tage")
        log.info(f"  Trades gesamt:      {result['total_trades']} "
                 f"(B: {result['buy_trades']} / S: {result['sell_trades']})")
        log.info(f"  Simulations-Runs:   {result['simulation_runs']}")
        if result['api_calls_made'] > 0:
            log.info(f"  OpenAI API-Calls:   {result['api_calls_made']} "
                     f"(≈${result['api_calls_made'] * 0.05:.2f} bei gpt-4o-mini)")
        log.info("=" * 60)

    def save_results(self, result: Dict, output_path: str = "logs/ai_backtest_result.json"):
        """Speichert Backtest-Ergebnisse als JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log.info(f"Ergebnisse gespeichert: {output_path}")

    def compare_with_momentum(self) -> Dict:
        """
        Vergleicht KI-Strategie mit der regelbasierten Momentum-Baseline.
        Führt beide Backtests durch und gibt Vergleich zurück.
        """
        from backtester import Backtester as MomentumBacktester

        log.info("\nVergleich: KI vs. Momentum-Strategie")
        log.info("-" * 40)

        # KI-Ergebnis (dieser Backtester)
        ai_result = self.run()

        # Momentum-Baseline (alter Backtester)
        momentum_bt = MomentumBacktester(
            tickers=[t for t in self.trade_tickers if t != "^VIX"],
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            commission=self.commission,
        )
        momentum_result = momentum_bt.run(self.risk_profile)

        comparison = {
            "KI-Strategie": {
                "total_return_pct":      ai_result.get("total_return_pct"),
                "annualized_return_pct": ai_result.get("annualized_return_pct"),
                "max_drawdown_pct":      ai_result.get("max_drawdown_pct"),
                "sharpe_ratio":          ai_result.get("sharpe_ratio"),
                "total_trades":          ai_result.get("total_trades"),
            },
            "Momentum-Baseline": {
                "total_return_pct":      momentum_result.get("total_return_pct"),
                "annualized_return_pct": momentum_result.get("annualized_return_pct"),
                "max_drawdown_pct":      momentum_result.get("max_drawdown_pct"),
                "sharpe_ratio":          momentum_result.get("sharpe_ratio"),
                "total_trades":          momentum_result.get("total_trades"),
            },
        }

        log.info("\nVERGLEICHSERGEBNIS:")
        log.info(f"{'Kennzahl':<25} {'KI':>14} {'Momentum':>14}")
        log.info("-" * 55)
        for key in ["total_return_pct", "annualized_return_pct", "max_drawdown_pct", "sharpe_ratio", "total_trades"]:
            ki_val  = comparison["KI-Strategie"][key]
            mo_val  = comparison["Momentum-Baseline"][key]
            ki_str  = f"{ki_val:>14.2f}" if isinstance(ki_val, float) else f"{str(ki_val):>14}"
            mo_str  = f"{mo_val:>14.2f}" if isinstance(mo_val, float) else f"{str(mo_val):>14}"
            log.info(f"{key:<25} {ki_str} {mo_str}")

        return comparison


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KI-Backtester: Testet die echte AI-Strategie mit historischen Daten."
    )
    parser.add_argument("--start",    default=BACKTEST_START_DATE,  help="Startdatum YYYY-MM-DD")
    parser.add_argument("--end",      default=BACKTEST_END_DATE,    help="Enddatum YYYY-MM-DD")
    parser.add_argument("--capital",  type=float, default=BACKTEST_INITIAL_CAPITAL)
    parser.add_argument("--weekly",   action="store_true", help="Wöchentliche Runs (Standard)")
    parser.add_argument("--daily",    action="store_true", help="Tägliche Runs (teuer!)")
    parser.add_argument("--dry-run",  action="store_true", dest="dry_run",
                        help="Ohne OpenAI (regelbasierte Fallback-Analyse)")
    parser.add_argument("--use-llm", action="store_true",
                        help="OpenAI-Backtest aktivieren (nur wenn API-Key gesetzt)")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Aktiviert rollierendes Walk-Forward-Testing")
    parser.add_argument("--parallel", action="store_true",
                        help="Optional: Score-Berechnung parallelisieren")
    parser.add_argument("--compare",  action="store_true",
                        help="KI vs. Momentum-Baseline vergleichen")
    parser.add_argument("--tickers",  nargs="+", default=None,
                        help="Ticker-Liste (Standard: FULL_WATCHLIST aus config.py)")
    parser.add_argument("--output",   default="logs/ai_backtest_result.json")
    args = parser.parse_args()

    frequency = "daily" if args.daily else "weekly"
    use_ai = False
    if args.use_llm and not args.dry_run:
        from config import OPENAI_API_KEY
        if OPENAI_API_KEY:
            use_ai = True
        else:
            log.warning("OPENAI_API_KEY nicht gesetzt – verwende Fallback-Modus.")

    bt = AIBacktester(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        frequency=frequency,
        use_ai=use_ai,
        walk_forward=args.walk_forward,
        use_multiprocessing=args.parallel,
    )

    if args.compare:
        bt.compare_with_momentum()
    else:
        result = bt.run()
        bt.save_results(result, args.output)


if __name__ == "__main__":
    main()