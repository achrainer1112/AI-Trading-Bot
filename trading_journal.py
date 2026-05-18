"""
AI Trading Bot - Trading Journal
==================================
Transparentes Protokoll aller KI-Entscheidungen und Trades.
Speichert WARUM jede Entscheidung getroffen wurde.

Gespeichert in: logs/journal.json
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from logger import log
from utils import (
    save_json_file,
    load_json_file,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_volatility,
    calculate_beta_alpha,
    calculate_profit_factor,
)

# Robust import of format_currency – never crash the journal on a missing utility
try:
    from utils import format_currency
except ImportError:
    def format_currency(x: float) -> str:  # type: ignore[misc]
        return f"${x:,.2f}"

JOURNAL_FILE = "logs/journal.json"


class TradingJournal:
    """
    Schreibt für jeden Run einen vollständigen Eintrag:
    - Marktausblick der KI
    - Jede Entscheidung mit Begründung, Konfidenz, Aktion
    - Ausgeführte Trades mit Grund
    - Portfolio-Snapshot vorher/nachher
    """

    def __init__(self):
        Path("logs").mkdir(exist_ok=True)

    def log_run(
        self,
        market_outlook: str,
        risk_assessment: str,
        ai_signals: List[Dict],
        final_decisions: List[Dict],
        simulated_trades: List[Dict],
        executed_trades: List[Dict],
        portfolio_before: Dict,
        portfolio_after: Dict,
        portfolio_projection: Dict = None,
        risk_warnings: List[str] = None,
        mode: str = "DRY",
        feedback_learnings: str = "",
        regime_state=None,
        market_data: Dict[str, Dict] = None,
        execution_mode: str = "SIMULATED",
        market_closed: bool = False,
        debug: bool = False,
    ):
        """Speichert einen kompletten Trading-Run im Journal."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "mode": mode,

            # Markt-Regime
            "market_regime": regime_state.to_dict() if regime_state is not None else None,

            # KI Markteinschätzung
            "market_outlook": market_outlook,
            "risk_assessment": risk_assessment,
            "feedback_learnings": feedback_learnings,

            # Score Breakdown + Decision Trace (structured logging)
            "score_breakdown": self._build_score_breakdown(ai_signals, market_data),
            "decision_trace": self._build_decision_trace(ai_signals, final_decisions),

            # Portfolio Vorher
            "portfolio_before": {
                "total_value": portfolio_before.get("total_value", 0),
                "cash": portfolio_before.get("cash", 0),
                "cash_pct": portfolio_before.get("cash_pct", 0),
                "n_positions": portfolio_before.get("n_positions", 0),
                "pnl_pct": portfolio_before.get("pnl_pct", 0),
            },

            # Alle KI-Entscheidungen mit Begründung
            "ai_signals": [
                {
                    "ticker": d.get("ticker"),
                    "action": d.get("action"),
                    "target_allocation_pct": round(d.get("target_allocation", 0) * 100, 1),
                    "confidence_pct": round(d.get("confidence", 0) * 100, 1),
                    "reason": d.get("reason", ""),
                    "decision_id": d.get("decision_id"),
                    "filter_notes": d.get("_filter_notes", []),
                    "original_action": d.get("_original_action", d.get("action")),
                }
                for d in ai_signals
            ] if debug else [],

            "approved_trades": [
                {
                    "ticker": d.get("ticker"),
                    "action": d.get("action"),
                    "target_allocation_pct": round(d.get("target_allocation", 0) * 100, 1),
                    "confidence_pct": round(d.get("confidence", 0) * 100, 1),
                    "reason": d.get("reason", ""),
                    "decision_id": d.get("decision_id"),
                    "status": d.get("status"),
                }
                for d in final_decisions
                if d.get("action") in ("BUY", "SELL") and d.get("risk_approved", False)
            ],

            "blocked_trades": [
                {
                    "ticker": d.get("ticker"),
                    "action": d.get("action"),
                    "target_allocation_pct": round(d.get("target_allocation", 0) * 100, 1),
                    "confidence_pct": round(d.get("confidence", 0) * 100, 1),
                    "reason": d.get("reason", ""),
                    "decision_id": d.get("decision_id"),
                    "status": d.get("status"),
                }
                for d in final_decisions
                if not d.get("risk_approved", True)
            ],

            "simulated_trades": [
                {
                    "ticker": t.get("ticker"),
                    "action": t.get("action"),
                    "value_usd": t.get("value", 0),
                    "quantity": t.get("quantity", 0),
                    "price": t.get("price", 0),
                    "decision_id": t.get("decision_id"),
                    "ai_reason": t.get("ai_reason", ""),
                    "ai_confidence_pct": round(t.get("ai_confidence", 0) * 100, 1),
                    "from_alloc_pct": round(t.get("current_alloc", 0) * 100, 1),
                    "to_alloc_pct": round(t.get("target_alloc", 0) * 100, 1),
                    "status": t.get("status", "SIMULATED"),
                }
                for t in simulated_trades
            ],

            "risk_warnings": risk_warnings,

            # Ausgeführte Trades
            "executed_trades": [
                {
                    "ticker": t.get("ticker"),
                    "action": t.get("action"),

                    # FIX → echte Broker-Fills statt falscher Keys
                    "value_usd": t.get("fill_value", t.get("planned_value", 0)),
                    "quantity": t.get("fill_qty", t.get("planned_qty", 0)),
                    "price": t.get("fill_price", t.get("price", 0)),
                    "decision_id": t.get("decision_id"),

                    "ai_reason": t.get("reason", t.get("ai_reason", "Portfolio Rebalancing")),
                    "ai_confidence_pct": round(t.get("ai_confidence", 0) * 100, 1),

                    "from_alloc_pct": round(t.get("current_alloc", 0) * 100, 1),
                    "to_alloc_pct": round(t.get("target_alloc", 0) * 100, 1),
                    "status": t.get("status", "EXECUTED"),
                }
                for t in executed_trades
            ],

            # Portfolio Nachher
            "portfolio_after": {
                "total_value": portfolio_after.get("total_value", 0),
                "cash": portfolio_after.get("cash", 0),
                "cash_pct": portfolio_after.get("cash_pct", 0),
                "n_positions": portfolio_after.get("n_positions", 0),
                "pnl_pct": portfolio_after.get("pnl_pct", 0),
            },

            "benchmark": {
                "spy_return_7d": market_data.get("SPY", {}).get("return_7d") if market_data else None,
                "spy_return_30d": market_data.get("SPY", {}).get("return_30d") if market_data else None,
                "spy_volatility": market_data.get("SPY", {}).get("volatility_annual_pct") if market_data else None,
            },

            "portfolio_projection": {
                "total_value": portfolio_projection.get("total_value", 0) if portfolio_projection else None,
                "cash": portfolio_projection.get("cash", 0) if portfolio_projection else None,
                "cash_pct": portfolio_projection.get("cash_pct", 0) if portfolio_projection else None,
                "n_positions": portfolio_projection.get("n_positions", 0) if portfolio_projection else None,
                "pnl_pct": portfolio_projection.get("pnl_pct", 0) if portfolio_projection else None,
            } if portfolio_projection else None,

            "pending_orders": [
                d.get("ticker") for d in final_decisions if d.get("pending_order")
            ],

            "execution_mode": execution_mode,
            "market_closed": market_closed,
            "performance_metrics": {},
            "trades_executed": len(executed_trades),
            "simulated_trades_count": len(simulated_trades),
            "approved_trades_count": len([d for d in final_decisions if d.get("action") in ("BUY", "SELL") and d.get("risk_approved", False)]),
            "blocked_trades_count": len([d for d in final_decisions if not d.get("risk_approved", True)]),
        }

        # In Journal speichern
        journal = load_json_file(JOURNAL_FILE, default=[])
        journal.append(entry)
        # Max 365 Einträge behalten
        if len(journal) > 365:
            journal = journal[-365:]
        metrics = self._calculate_performance_metrics(journal)
        entry["performance_metrics"] = metrics
        journal[-1] = entry
        save_json_file(JOURNAL_FILE, journal)
        save_json_file("logs/performance_metrics.json", metrics)

        # Lesbares Summary im Log ausgeben
        self._print_journal_entry(entry)

    def _calculate_performance_metrics(self, history: List[Dict]) -> Dict:
        values = [entry.get("portfolio_after", {}).get("total_value") for entry in history]
        returns = []
        benchmark_returns = []
        wins = 0
        losses = 0

        for i in range(1, len(history)):
            prev = history[i - 1].get("portfolio_after", {}).get("total_value", 0)
            curr = history[i].get("portfolio_after", {}).get("total_value", 0)
            if prev and curr is not None:
                run_return = (curr - prev) / prev
                returns.append(run_return)
                if run_return > 0:
                    wins += 1
                elif run_return < 0:
                    losses += 1

            spy_ret = history[i].get("benchmark", {}).get("spy_return_7d")
            if spy_ret is not None:
                benchmark_returns.append(spy_ret / 100)

        win_rate = float(wins / len(returns)) if returns else 0.0
        profit_factor = calculate_profit_factor(returns)
        volatility = calculate_volatility(returns)
        sharpe = calculate_sharpe_ratio(returns)
        beta, alpha = calculate_beta_alpha(returns, benchmark_returns)
        max_drawdown = calculate_max_drawdown(values)

        return {
            "run_count": len(history),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float('inf') else float('inf'),
            "volatility": round(volatility, 4),
            "sharpe_ratio": round(sharpe, 4),
            "beta": round(beta, 4),
            "alpha": round(alpha, 4),
            "max_drawdown": round(max_drawdown, 4),
            "total_return": round((values[-1] / values[0] - 1) if values and values[0] else 0.0, 4),
        }

    def _print_journal_entry(self, entry: Dict):
        """Gibt einen lesbaren Journal-Eintrag aus."""
        log.info("=" * 70)
        log.info(f"TRADING JOURNAL – {entry['date']} {entry['time']}")
        log.info(f"Modus: {entry['mode']}")

        # Markt-Regime anzeigen
        regime = entry.get("market_regime")
        if regime:
            regime_label = regime.get("regime", "?").upper()
            regime_conf  = regime.get("confidence", 0)
            vix_str      = f" | VIX={regime['vix']:.1f}" if regime.get("vix") else ""
            log.info(f"Markt-Regime: {regime_label} ({regime_conf:.0%}){vix_str}")
            log.info(f"  {regime.get('description', '')}")

        log.info("")
        log.info(f"KI MARKTAUSBLICK: {entry['market_outlook']}")
        log.info(f"RISIKOEINSCHÄTZUNG: {entry['risk_assessment']}")
        if entry.get("execution_mode"):
            log.info(
                f"Execution mode: {entry['execution_mode']} | "
                f"Market closed: {entry.get('market_closed', False)}"
            )
        if entry.get("portfolio_projection"):
            proj = entry["portfolio_projection"]
            log.info(
                f"Projected portfolio after simulated trades: "
                f"{format_currency(proj.get('total_value', 0))} | "
                f"Cash: {format_currency(proj.get('cash', 0))} | "
                f"PnL: {proj.get('pnl_pct', 0):+.2f}%"
            )
        if entry.get("feedback_learnings"):
            log.info(f"KI-LEARNINGS: {entry['feedback_learnings']}")
        log.info("")

        # AI-Signale
        if entry.get("ai_signals"):
            log.info("── AI SIGNALS ──")
            for d in entry["ai_signals"]:
                log.info(
                    f"  {d['action']:<5} {d['ticker']:<6} | "
                    f"Ziel: {d['target_allocation_pct']:.0f}% | "
                    f"Konfidenz: {d['confidence_pct']:.0f}% | "
                    f"ID: {d.get('decision_id', '')}"
                )
                if d.get("filter_notes"):
                    for note in d["filter_notes"]:
                        log.info(f"         Note: {note}")
                log.info(f"         Grund: {d['reason']}")

        # Genehmigte und blockierte Entscheidungen
        if entry.get("approved_trades"):
            log.info("\n── APPROVED TRADES ──")
            for d in entry["approved_trades"]:
                log.info(
                    f"  {d['action']:<5} {d['ticker']:<6} | "
                    f"Ziel: {d['target_allocation_pct']:.0f}% | "
                    f"Konfidenz: {d['confidence_pct']:.0f}% | "
                    f"ID: {d.get('decision_id', '')}"
                )
                log.info(f"         Grund: {d['reason']}")

        if entry.get("blocked_trades"):
            log.info("\n── BLOCKED TRADES ──")
            for d in entry["blocked_trades"]:
                log.info(
                    f"  {d['action']:<5} {d['ticker']:<6} | "
                    f"Status: {d.get('status', 'BLOCKED')} | "
                    f"Ziel: {d['target_allocation_pct']:.0f}% | "
                    f"Konfidenz: {d['confidence_pct']:.0f}%"
                )
                log.info(f"         Grund: {d['reason']}")

        if entry.get("simulated_trades"):
            log.info("\n── SIMULATED TRADES ──")
            for t in entry["simulated_trades"]:
                log.info(
                    f"  {t['action']:<5} {t['ticker']:<6} | "
                    f"Wert: ${t['value_usd']:,.0f} | "
                    f"ID: {t.get('decision_id', '')} | "
                    f"Ziel: {t['to_alloc_pct']:.0f}%"
                )

        if entry.get("executed_trades"):
            log.info("\n── EXECUTED TRADES ──")
            for t in entry["executed_trades"]:
                log.info(
                    f"  {t['action']:<5} {t['ticker']:<6} | "
                    f"Wert: ${t['value_usd']:,.0f} | "
                    f"Status: {t.get('status', '')} | "
                    f"ID: {t.get('decision_id', '')}"
                )
        else:
            log.info("\n── KEINE TRADES AUSGEFÜHRT ──")

        if entry["risk_warnings"]:
            log.info("")
            log.info("── RISIKOPRÜFUNG KORREKTUREN ──")
            for w in entry["risk_warnings"]:
                log.info(f"  ⚠ {w}")

        log.info("")
        log.info(f"PORTFOLIO: ${entry['portfolio_after']['total_value']:,.2f} | "
                 f"P&L: {entry['portfolio_after']['pnl_pct']:+.2f}% | "
                 f"Cash: {entry['portfolio_after']['cash_pct']:.1f}%")
        log.info("=" * 70)

    # ── Structured Logging Helpers ──────────────────────────────────────────

    def _build_score_breakdown(self, ai_signals: List[Dict], market_data: Dict) -> List[Dict]:
        """
        Extrahiert Score-Aufschlüsselung aus KI-Signalen für persistentes Logging.
        Enthält quant_score, llm_adj, reasoning-Metriken.
        """
        rows = []
        for d in ai_signals:
            ticker = d.get("ticker", "")
            mkt = (market_data or {}).get(ticker, {})
            rows.append({
                "ticker": ticker,
                "action": d.get("action"),
                "quant_score": d.get("quant_score"),
                "llm_score_adj": d.get("llm_score_adj", 0),
                "effective_score": (d.get("quant_score") or 0) + (d.get("llm_score_adj") or 0),
                "confidence": d.get("confidence"),
                "reasoning": d.get("reasoning", {}),
                # Zusätzliche Marktmetriken direkt aus market_data
                "rsi": mkt.get("rsi_14"),
                "momentum_20d": mkt.get("return_20d"),
                "sma_distance_pct": mkt.get("sma_distance_pct"),
                "volatility": mkt.get("volatility_annual_pct"),
                "relative_strength": mkt.get("relative_strength_vs_spy"),
            })
        return rows

    def _build_decision_trace(self, ai_signals: List[Dict], final_decisions: List[Dict]) -> List[Dict]:
        """
        Decision Trace: Verfolgt jede Entscheidung von AI-Signal bis zur finalen Validierung.
        Zeigt was geblockt/modifiziert wurde und warum.
        """
        final_map = {d.get("ticker"): d for d in final_decisions}
        trace = []
        for sig in ai_signals:
            ticker = sig.get("ticker")
            final = final_map.get(ticker, {})
            trace.append({
                "ticker": ticker,
                "ai_action": sig.get("action"),
                "ai_allocation": sig.get("target_allocation"),
                "ai_confidence": sig.get("confidence"),
                "quant_score": sig.get("quant_score"),
                "final_action": final.get("action"),
                "final_allocation": final.get("target_allocation"),
                "risk_approved": final.get("risk_approved", False),
                "status": final.get("status", "UNKNOWN"),
                "modifications": [
                    note for note in [
                        final.get("reason", "") if final.get("action") != sig.get("action") else None
                    ] if note
                ],
            })
        return trace

    def get_history(self) -> List[Dict]:
        """Gibt gesamte Journal-Historie zurück."""
        return load_json_file(JOURNAL_FILE, default=[])

    def print_history(self, last_n: int = 5):
        """Gibt die letzten N Journal-Einträge aus."""
        history = self.get_history()
        if not history:
            log.info("Journal ist leer – noch keine Runs.")
            return
        for entry in history[-last_n:]:
            self._print_journal_entry(entry)


# Singleton
journal = TradingJournal()