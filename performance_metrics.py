"""
performance_metrics.py – Professionelle Performance-Analytics
=============================================================
Berechnet alle gängigen Kennzahlen für Portfolio- und Strategie-Evaluation.

Enthält:
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
  - Max Drawdown
  - Alpha / Beta
  - Win Rate / Profit Factor
  - Rolling Volatility
  - Information Ratio
  - Treynor Ratio
  - Capture Ratios (Up/Down)
  - VaR / CVaR
  - Regime-basierte Performance
  - Trade Attribution
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from scipy import stats

from logger import log


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Berechnet die Sharpe Ratio.
    returns: Liste der täglichen/periodischen Renditen (als Dezimalzahlen, z.B. 0.01 für 1%)
    """
    if len(returns) < 2:
        return 0.0
    ret_array = np.array(returns)
    excess = ret_array - risk_free_rate / periods_per_year
    if np.std(excess) == 0:
        return 0.0
    return np.sqrt(periods_per_year) * np.mean(excess) / np.std(excess)


def calculate_sortino_ratio(
    returns: List[float],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Berechnet die Sortino Ratio (nur Downside-Volatilität).
    """
    if len(returns) < 2:
        return 0.0
    ret_array = np.array(returns)
    excess = ret_array - risk_free_rate / periods_per_year
    downside = np.std(excess[excess < 0]) if np.any(excess < 0) else 1e-6
    return np.sqrt(periods_per_year) * np.mean(excess) / downside


def calculate_calmar_ratio(
    returns: List[float],
    periods_per_year: int = 252
) -> float:
    """
    Berechnet die Calmar Ratio (annualisierte Rendite / Max Drawdown).
    """
    if len(returns) < 2:
        return 0.0
    annual_return = (1 + np.mean(returns)) ** periods_per_year - 1
    max_dd = calculate_max_drawdown_from_returns(returns)
    if max_dd == 0:
        return 0.0
    return annual_return / abs(max_dd)


def calculate_max_drawdown_from_values(values: List[float]) -> float:
    """
    Berechnet maximalen Drawdown aus einer Reihe von Portfolio-Werten.
    """
    if len(values) < 2:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def calculate_max_drawdown_from_returns(returns: List[float]) -> float:
    """
    Berechnet maximalen Drawdown aus einer Reihe von Renditen.
    """
    if len(returns) < 2:
        return 0.0
    cumulative = np.cumprod(1 + np.array(returns))
    return calculate_max_drawdown_from_values(cumulative.tolist())


def calculate_alpha_beta(
    returns: List[float],
    benchmark_returns: List[float],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> Tuple[float, float]:
    """
    Berechnet Alpha und Beta relativ zu einem Benchmark.
    returns: Portfolio-Renditen
    benchmark_returns: Benchmark-Renditen (gleiche Länge)
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0, 0.0
    ret_arr = np.array(returns)
    bench_arr = np.array(benchmark_returns)
    # Gleiche Länge sicherstellen
    min_len = min(len(ret_arr), len(bench_arr))
    ret_arr = ret_arr[-min_len:]
    bench_arr = bench_arr[-min_len:]
    # Kovarianz und Varianz
    cov = np.cov(ret_arr, bench_arr)[0, 1]
    var_bench = np.var(bench_arr)
    beta = cov / var_bench if var_bench != 0 else 0.0
    # Annualisierte Renditen
    ret_ann = (1 + np.mean(ret_arr)) ** periods_per_year - 1
    bench_ann = (1 + np.mean(bench_arr)) ** periods_per_year - 1
    rf_ann = risk_free_rate
    alpha = (ret_ann - rf_ann) - beta * (bench_ann - rf_ann)
    return alpha, beta


def calculate_win_rate(returns: List[float]) -> float:
    """Berechnet die prozentuale Anzahl positiver Perioden."""
    if not returns:
        return 0.0
    wins = sum(1 for r in returns if r > 0)
    return wins / len(returns)


def calculate_profit_factor(returns: List[float]) -> float:
    """Berechnet Profit Factor (Summe Gewinne / Summe Verluste)."""
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def calculate_rolling_volatility(
    returns: List[float],
    window: int = 20,
    periods_per_year: int = 252
) -> List[float]:
    """
    Berechnet rollierende annualisierte Volatilität.
    """
    if len(returns) < window:
        return [0.0] * len(returns)
    result = []
    for i in range(window, len(returns) + 1):
        window_rets = returns[i - window:i]
        vol = np.std(window_rets) * np.sqrt(periods_per_year)
        result.append(vol)
    # Auffüllen am Anfang
    padding = [0.0] * (window - 1)
    return padding + result


def calculate_information_ratio(
    returns: List[float],
    benchmark_returns: List[float],
    periods_per_year: int = 252
) -> float:
    """
    Berechnet Information Ratio (aktive Rendite / Tracking Error).
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    min_len = min(len(returns), len(benchmark_returns))
    ret_arr = np.array(returns[-min_len:])
    bench_arr = np.array(benchmark_returns[-min_len:])
    active = ret_arr - bench_arr
    if np.std(active) == 0:
        return 0.0
    return np.sqrt(periods_per_year) * np.mean(active) / np.std(active)


def calculate_treynor_ratio(
    returns: List[float],
    benchmark_returns: List[float],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Berechnet Treynor Ratio (Überschussrendite pro Beta).
    """
    alpha, beta = calculate_alpha_beta(returns, benchmark_returns, risk_free_rate, periods_per_year)
    if beta == 0:
        return 0.0
    ret_ann = (1 + np.mean(returns)) ** periods_per_year - 1
    rf_ann = risk_free_rate
    return (ret_ann - rf_ann) / beta


def calculate_capture_ratios(
    returns: List[float],
    benchmark_returns: List[float]
) -> Dict[str, float]:
    """
    Berechnet Up- und Down-Capture Ratios.
    Up-Capture: Rendite in Up-Märkten relativ zum Benchmark.
    Down-Capture: Rendite in Down-Märkten relativ zum Benchmark.
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return {"up_capture": 1.0, "down_capture": 1.0}
    min_len = min(len(returns), len(benchmark_returns))
    ret_arr = np.array(returns[-min_len:])
    bench_arr = np.array(benchmark_returns[-min_len:])
    # Up-Märkte: Benchmark positiv
    up_mask = bench_arr > 0
    down_mask = bench_arr < 0
    up_capture = np.mean(ret_arr[up_mask]) / np.mean(bench_arr[up_mask]) if np.any(up_mask) else 1.0
    down_capture = np.mean(ret_arr[down_mask]) / np.mean(bench_arr[down_mask]) if np.any(down_mask) else 1.0
    return {"up_capture": up_capture, "down_capture": down_capture}


def calculate_var(returns: List[float], confidence: float = 0.95) -> float:
    """
    Berechnet Value at Risk (parametrisch, Normalverteilung).
    """
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns)
    z = stats.norm.ppf(1 - confidence)
    return mu + z * sigma


def calculate_cvar(returns: List[float], confidence: float = 0.95) -> float:
    """
    Berechnet Conditional Value at Risk (Expected Shortfall).
    """
    if len(returns) < 2:
        return 0.0
    var = calculate_var(returns, confidence)
    # CVaR ist der Durchschnitt der Renditen unterhalb von VaR
    below_var = [r for r in returns if r <= var]
    if not below_var:
        return var
    return np.mean(below_var)


def calculate_trade_attribution(
    trades: List[Dict],
    initial_capital: float = 100000.0
) -> Dict:
    """
    Berechnet Trade-Attribution: Gewinne/Verluste pro Trade, beste/schlechteste Trades,
    durchschnittliche Haltedauer, Sektor-Attribution.
    """
    if not trades:
        return {"total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                "avg_profit": 0.0, "avg_loss": 0.0, "best_trade": 0.0, "worst_trade": 0.0}

    profits = []
    for trade in trades:
        if trade.get("action") == "SELL" and trade.get("fill_qty") and trade.get("fill_price"):
            # Vereinfacht: Gewinn aus Verkauf (müsste mit Kaufpreis verrechnet werden)
            # Hier nur Platzhalter – für echte Attribution braucht man vollständige Order-Historie
            pass

    # Platzhalter: Rückgabe von Basisstatistiken
    winning = [t for t in trades if t.get("pnl", 0) > 0]
    losing = [t for t in trades if t.get("pnl", 0) < 0]
    return {
        "total_trades": len(trades),
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "win_rate": len(winning) / len(trades) if trades else 0,
        "avg_profit": np.mean([t.get("pnl", 0) for t in winning]) if winning else 0,
        "avg_loss": np.mean([t.get("pnl", 0) for t in losing]) if losing else 0,
    }


def regime_based_performance(
    returns: List[float],
    regimes: List[str],
    regime_labels: List[str] = None
) -> Dict[str, Dict]:
    """
    Berechnet Performance-Kennzahlen pro Marktregime.
    returns: Liste der Renditen
    regimes: Liste der Regime-Bezeichnungen (gleiche Länge)
    """
    if len(returns) != len(regimes) or len(returns) == 0:
        return {}
    df = pd.DataFrame({"return": returns, "regime": regimes})
    results = {}
    for regime in df["regime"].unique():
        regime_returns = df[df["regime"] == regime]["return"].tolist()
        results[regime] = {
            "count": len(regime_returns),
            "total_return": (1 + np.mean(regime_returns)) ** len(regime_returns) - 1 if regime_returns else 0,
            "avg_return": np.mean(regime_returns) if regime_returns else 0,
            "volatility": np.std(regime_returns) if len(regime_returns) > 1 else 0,
            "sharpe": calculate_sharpe_ratio(regime_returns),
            "max_drawdown": calculate_max_drawdown_from_returns(regime_returns),
        }
    return results


def comprehensive_performance_report(
    returns: List[float],
    benchmark_returns: Optional[List[float]] = None,
    values: Optional[List[float]] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> Dict:
    """
    Erstellt einen umfassenden Performance-Bericht mit allen Kennzahlen.
    """
    if not returns:
        return {"error": "Keine Renditedaten"}

    report = {
        "total_return_pct": (np.prod(1 + np.array(returns)) - 1) * 100,
        "annualized_return_pct": ((1 + np.mean(returns)) ** periods_per_year - 1) * 100,
        "volatility_annual_pct": np.std(returns) * np.sqrt(periods_per_year) * 100,
        "sharpe_ratio": calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        "calmar_ratio": calculate_calmar_ratio(returns, periods_per_year),
        "win_rate": calculate_win_rate(returns),
        "profit_factor": calculate_profit_factor(returns),
        "max_drawdown_pct": calculate_max_drawdown_from_returns(returns) * 100,
    }

    if values:
        report["max_drawdown_from_values_pct"] = calculate_max_drawdown_from_values(values) * 100

    if benchmark_returns:
        alpha, beta = calculate_alpha_beta(returns, benchmark_returns, risk_free_rate, periods_per_year)
        report["alpha"] = alpha * 100  # in Prozent
        report["beta"] = beta
        report["information_ratio"] = calculate_information_ratio(returns, benchmark_returns, periods_per_year)
        report["treynor_ratio"] = calculate_treynor_ratio(returns, benchmark_returns, risk_free_rate, periods_per_year)
        capture = calculate_capture_ratios(returns, benchmark_returns)
        report["up_capture"] = capture["up_capture"]
        report["down_capture"] = capture["down_capture"]

    return report
