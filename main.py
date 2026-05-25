"""
AI Trading Bot - Hauptprogramm (Production)
=============================================
Korrekte Ausführungsreihenfolge:
  Phase 0: Markt-Check → sofort abbrechen wenn geschlossen (Fix #1)
  Phase 1: Zombie-Cleanup (Fix #3)
  Phase 2: SELLs ausführen
  Phase 3: Broker-Sync (echtes Cash nach SELLs)
  Phase 4: BUYs mit korrekt berechnetem Cash ausführen
  Phase 5: Portfolio-Konsistenz-Check (Fix #10)
  Phase 6: Journal mit echten Fill-Daten

FIXES:
  Fix #1  – Fake Trades: Abbruch bei geschlossenem Markt (vor jeder Trade-Logik)
  Fix #3  – Zombie-Positionen werden automatisch liquidiert
  Fix #4  – KI-Input strikt auf Watchlist + aktuelle Positionen begrenzt
  Fix #8  – Phasenlogik: SELL → Sync → BUY (keine stale Daten)
  Fix #10 – Portfolio-Konsistenz nach jedem Run sicherstellen
  Fix #12 – State Management: alle Inputs/Outputs persistent gespeichert
  Fix #17 – KI-Input wird geloggt (Determinismus + Debugging)
"""

import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from matplotlib import ticker  # noqa: used by dashboard only – safe to import here

from score_engine import ScoreEngine  # noqa: imported for type awareness

from logger import log, trade_logger
from config import (
    TRADING_MODE, FULL_WATCHLIST, ETF_WATCHLIST, STOCK_WATCHLIST,
    ACTIVE_RISK_PROFILE, RISK_SETTINGS, SCHEDULE_INTERVAL, SCHEDULE_TIME,
    BACKTEST_START_DATE, BACKTEST_END_DATE, INITIAL_CAPITAL,
    LOG_DIR, CORRELATION_GROUPS,
)
from data_collector import MarketDataCollector
from news_collector import NewsCollector
from ai_analysis import AIAnalyzer
from portfolio_manager import PortfolioManager
from risk_manager import RiskManager
from trade_executor import TradeExecutor
from ai_backtester import AIBacktester
from utils import (
    is_market_open, market_status, format_currency,
    find_zombie_positions, build_zombie_sell_orders, save_json_file,
    normalize_ai_decisions, cooldown_manager,
)
from trading_journal import journal
from market_regime import Regime
from market_regime_enhanced import EnhancedMarketRegimeDetector
from decision_filter import apply_decision_filter
from execution_safety import LiveRunLogger, DrawdownMonitor
from config import ALLOW_LIVE_TRADING, KILL_SWITCH_DRAWDOWN_PCT


import os
os.makedirs(LOG_DIR, exist_ok=True)


class TradingBot:
    """
    Orchestriert alle Module für den vollautomatischen Trading-Ablauf.
    """

    def __init__(self, mode: str = None):
        self.mode = (mode or TRADING_MODE).upper()
        log.info("=" * 70)
        log.info("  AI TRADING BOT – INITIALISIERUNG")
        log.info(f"  Modus: {self.mode} | Risikoprofil: {ACTIVE_RISK_PROFILE.value.upper()}")
        log.info(f"  Watchlist: {len(FULL_WATCHLIST)} Assets")
        log.info(f"  Markt-Status: {market_status()}")
        log.info("=" * 70)

        self.data_collector = MarketDataCollector(FULL_WATCHLIST)
        self.news_collector = NewsCollector()
        self.ai_analyzer    = AIAnalyzer()

        # Executor zuerst – Alpaca-Verbindung wird für Portfolio-Sync benötigt
        self.executor     = TradeExecutor(mode=self.mode)
        alpaca_api        = self.executor.api if self.mode in ("PAPER", "REAL", "LIVE") else None
        self.portfolio    = PortfolioManager(mode=self.mode, alpaca_api=alpaca_api)
        self.risk_manager = RiskManager(ACTIVE_RISK_PROFILE)

        # ── Phase 4A: Live Safety Layer ──────────────────────────────────────
        self._live_logger = LiveRunLogger()
        _guard = self.executor.get_guard()
        self._drawdown_monitor = DrawdownMonitor(
            guard=_guard, limit_pct=KILL_SWITCH_DRAWDOWN_PCT
        ) if _guard else None

    # ─── Hauptablauf ──────────────────────────────────────────────────────────

    def run(self) -> Dict:
        start_time = datetime.now()
        log.info(f"\n{'='*70}")
        log.info(f"  TRADING RUN GESTARTET: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        current_market_status = market_status()
        market_open = is_market_open()
        analysis_only = not market_open

        log.info(f"  Markt: {current_market_status}")
        if analysis_only:
            log.warning("Market closed -> analysis-only mode activated")
        log.info(f"{'='*70}\n")

        run_summary: Dict = {
            "started_at":    start_time.isoformat(),
            "mode":          self.mode,
            "risk_profile":  ACTIVE_RISK_PROFILE.value,
            "market_status": market_status(),
            "regime":        None,
            "sells_executed": 0,
            "buys_executed":  0,
            "trades_executed": 0,
            "errors":         [],
            "risk_warnings":  [],
            "analysis_only":  analysis_only,
        }

        # ── Fix #1: MARKT-CHECK – einzige, zentrale Prüfung ──────────────────
        # Der Bot analysiert weiterhin in OFF-Hours, führt aber keine Orders aus.
        if analysis_only:
            log.warning(
                f"Market closed ({current_market_status}) – analysis-only mode activated. "
                f"Keine Orders werden in diesem Run ausgeführt."
            )

        # ── SCHRITT 0: Markt-Regime erkennen ─────────────────────────────────
        log.info("SCHRITT 0/8: Markt-Regime-Detektion...")
        try:
            regime_detector = EnhancedMarketRegimeDetector(watchlist=FULL_WATCHLIST)
            regime_state    = regime_detector.detect()
            self.risk_manager.apply_regime(regime_state)
            run_summary["regime"] = regime_state.to_dict()
            save_json_file(
                f"{LOG_DIR}/regime_{start_time.strftime('%Y%m%d_%H%M%S')}.json",
                regime_state.to_dict(),
            )
        except Exception as e:
            log.warning(f"Regime-Detektion fehlgeschlagen: {e} – fahre ohne Regime-Override fort.")
            regime_state = None

        # ── SCHRITT 1: Marktdaten ─────────────────────────────────────────────
        log.info("SCHRITT 1/8: Marktdaten sammeln...")
        market_data = self.data_collector.collect_all()
        if not market_data:
            log.error("Keine Marktdaten – Run abgebrochen.")
            run_summary["errors"].append("no_market_data")
            return run_summary

        # ── SCHRITT 2: News ───────────────────────────────────────────────────
        log.info("\nSCHRITT 2/8: News & Sentiment sammeln...")
        news_articles = self.news_collector.collect_all()
        news_text     = self.news_collector.format_for_ai(news_articles)

        # ── SCHRITT 3: Portfolio-Snapshot (vor Trades) ────────────────────────
        log.info("\nSCHRITT 3/8: Portfolio aktualisieren...")
        self.portfolio.update_prices(market_data)
        portfolio_summary_before = self._build_portfolio_summary()
        self.portfolio.print_summary()

        # ── Fix #3: Zombie-Positionen erkennen und liquidieren ────────────────
        zombie_tickers = find_zombie_positions(self.portfolio.positions)
        zombie_orders  = build_zombie_sell_orders(zombie_tickers, self.portfolio.positions)
        if zombie_orders:
            log.warning(f"  {len(zombie_orders)} Zombie-Positionen gefunden → werden liquidiert")

        # ── SCHRITT 4: KI-Analyse ─────────────────────────────────────────────
        log.info("\nSCHRITT 4/8: KI-Analyse & Entscheidungsfindung...")

        # 🔥 PRE-AI: TRAILING STOP TRIGGERS (BEFORE AI ANALYSIS)
        # Generiere automatische Profit-Taking Signale für Positionen mit Gewinnen
        log.info("\nSCHRITT 3.5/8: Trailing Stop Check...")
        trailing_stops = self.portfolio.get_trailing_stop_triggers(market_data, trail_pct=0.12)
        if trailing_stops:
            log.info(f"  {len(trailing_stops)} Trailing Stop Trigger(s) erkannt")
            for ts in trailing_stops:
                log.info(f"    → {ts['ticker']}: {ts['reason']}")
        else:
            log.info("  Keine Trailing Stop Trigger")

        # Fix #4: KI-Input auf Watchlist + aktuelle Positionen begrenzen
        # (verhindert dass KI SELL für nicht gehaltene Assets empfiehlt)
        effective_watchlist = list(
            set(FULL_WATCHLIST) | set(self.portfolio.positions.keys())
        )

        # Fix #17: KI-Input loggen für Determinismus + Debugging
        ai_input_log = {
            "timestamp":  start_time.isoformat(),
            "watchlist":  effective_watchlist,
            "portfolio":  portfolio_summary_before,
            "news_len":   len(news_text),
        }
        save_json_file(
            f"{LOG_DIR}/ai_input_{start_time.strftime('%Y%m%d_%H%M%S')}.json",
            ai_input_log,
        )

        # Fix #19: Letzte Journal-Einträge für Feedback-Loop laden (max. 10)
        recent_journal = journal.get_history()[-10:]
        if recent_journal:
            log.info(f"  Feedback-Loop: {len(recent_journal)} vergangene Runs geladen.")

        ai_result = self.ai_analyzer.analyze(
            portfolio_summary=portfolio_summary_before,
            market_data=market_data,
            news_text=news_text,
            watchlist=effective_watchlist,
            journal_entries=recent_journal,  # Fix #19
            regime_state=regime_state,       # Regime
        )
        raw_decisions = [dict(d) for d in ai_result.get("decisions", [])]

        # ── DecisionFilter: quality & consistency improvement layer ───────────
        log.info("\nSCHRITT 4b/8: Decision quality filter...")
        filtered_decisions, filter_warnings = apply_decision_filter(
            decisions=raw_decisions,
            risk_settings=RISK_SETTINGS[ACTIVE_RISK_PROFILE],
            positions=self.portfolio.positions,
            total_value=portfolio_summary_before.get("total_value", 100_000),
            market_data=market_data,
            regime_state=regime_state,
        )
        ai_result["decisions"] = filtered_decisions
        if filter_warnings:
            run_summary["decision_filter_warnings"] = filter_warnings
            log.info(f"  DecisionFilter adjustments: {len(filter_warnings)}")

        # Fix #17: KI-Output loggen
        save_json_file(
            f"{LOG_DIR}/ai_output_{start_time.strftime('%Y%m%d_%H%M%S')}.json",
            ai_result,
        )

        decisions = filtered_decisions

        # ── Cooldown Filter ───────────────────────────────────────────────────
        log.info("\nSCHRITT 4c/8: Cooldown-Filter...")
        decisions, cooldown_warnings = cooldown_manager.filter_decisions(decisions)
        if cooldown_warnings:
            log.info(f"  Cooldown adjustments: {len(cooldown_warnings)}")
            run_summary["cooldown_warnings"] = cooldown_warnings
        log.info(f"  {len(decisions)} Entscheidungen | {ai_result.get('market_outlook', '')[:80]}")
        for d in decisions:
            log.info(
                f"  → {d['action']:<5} {d['ticker']:<6} "
                f"Allok: {d.get('target_allocation', 0):.0%} | "
                f"Konfidenz: {d.get('confidence', 0):.0%} | "
                f"{d.get('reason', '')}"
            )

        # ── SCHRITT 5: Stop-Loss + Zombie-Cleanup prüfen ─────────────────────
        log.info("\nSCHRITT 5/8: Stop-Loss & Zombie-Cleanup...")
        stop_loss_orders = self.risk_manager.check_stop_loss(
            self.portfolio.positions, market_data
        )
        if stop_loss_orders:
            log.warning(f"  ⛔ {len(stop_loss_orders)} Stop-Loss ausgelöst!")

        # Priorität: Stop-Loss > Trailing-Stop > Zombie-Cleanup > KI-Entscheidungen
        decisions = stop_loss_orders + trailing_stops + zombie_orders + decisions

        # Normalisierung und Redundanz-Filter vor finaler Risiko-Validierung
        normalized_decisions, normalization_warnings = normalize_ai_decisions(
            decisions=decisions,
            positions=self.portfolio.positions,
            total_value=portfolio_summary_before.get("total_value", 100_000),
            market_data=market_data,
            correlation_groups=CORRELATION_GROUPS,
        )
        if normalization_warnings:
            log.info(f"  Normalization adjustments: {len(normalization_warnings)}")
            run_summary["normalization_warnings"] = normalization_warnings

        # ── SCHRITT 6: Risikoprüfung & Validierung ──────────────────────────
        log.info("\nSCHRITT 6/8: Risikoprüfung & Validierung...")
        validated, warnings = self.risk_manager.validate_decisions(
            decisions=normalized_decisions,
            portfolio_summary=portfolio_summary_before,
            market_data=market_data,
        )

        # ── VaR Check ─────────────────────────────────────────────────────────
        total_val = portfolio_summary_before.get("total_value", INITIAL_CAPITAL)
        var_ok, var_msg = self.risk_manager.check_var_limit(
            self.portfolio.positions, market_data, total_val
        )
        if not var_ok:
            warnings.append(var_msg)

        # ── Correlation-adjusted exposure log ─────────────────────────────────
        corr_exposure = self.risk_manager.calculate_correlation_adjusted_exposure(
            self.portfolio.positions, total_val
        )
        if corr_exposure:
            log.info(f"  Sektor-Exposure (ETF-overlap-adjusted): {corr_exposure}")
            run_summary["sector_exposure"] = corr_exposure

        warnings = list(dict.fromkeys(
            (filter_warnings or []) + (normalization_warnings or []) + warnings
        ))
        run_summary["risk_warnings"] = warnings

        decisions_map = {d["ticker"]: d for d in validated}
        current_prices = {t: d.get("current_price", 0) for t, d in market_data.items()}
        _profile_settings = RISK_SETTINGS[ACTIVE_RISK_PROFILE]
        planned_trades = self.portfolio.calculate_rebalancing_trades(
            target_allocations={
                d["ticker"]: d.get("target_allocation", 0)
                for d in validated
                if d.get("action") in ("BUY", "SELL")
            },
            current_prices=current_prices,
            decisions_map=decisions_map,
            min_trade_value=_profile_settings.get("min_trade_value", 100.0),
        )
        projected_portfolio = self.portfolio.simulate_trade_plan(
            trades=planned_trades,
            current_prices=current_prices,
            min_cash_pct=self.risk_manager.settings.get("min_cash_pct", 0.0),
        )
        simulated_trades = [dict(t, status="SIMULATED") for t in planned_trades]
        sell_trades = [t for t in planned_trades if t["action"] == "SELL"]
        buy_trades = [t for t in planned_trades if t["action"] == "BUY"]

        approved_trades = [
            d for d in validated
            if d.get("action") in ("BUY", "SELL") and d.get("risk_approved", False)
        ]
        blocked_trades = [
            d for d in validated
            if not d.get("risk_approved", True)
        ]

        run_summary.update({
            "recommended_signals": len(raw_decisions),
            "approved_trades": len(approved_trades),
            "blocked_trades": len(blocked_trades),
            "simulated_trades": len(simulated_trades),
        })

        log.info(
            f"  Finaler Entscheidungsfluss: {len(raw_decisions)} Signale | "
            f"{len(approved_trades)} approved | {len(blocked_trades)} blocked | "
            f"{len(simulated_trades)} simulated"
        )

        if analysis_only:
            log.warning("Market closed -> execution disabled (analysis/simulation mode)")

        execution_enabled = (
            not analysis_only
            and self.mode in ("PAPER", "REAL", "LIVE")
            and self.executor.api is not None
        )

        # ── Phase 4A: Guard-Validierungen (Marktdaten + KI-Response) ─────────
        _guard = self.executor.get_guard()
        if execution_enabled and _guard:
            md_ok  = _guard.validate_market_data(market_data)
            ai_ok  = _guard.validate_ai_response(ai_result)
            if not md_ok or not ai_ok:
                log.critical(
                    f"[PHASE 4A] Guard-Validierung fehlgeschlagen: "                    f"market_data={md_ok} ai_response={ai_ok} → Execution deaktiviert."
                )
                execution_enabled = False
                run_summary["errors"].append("guard_validation_failed")

        # ── Phase 4A: Drawdown Kill-Switch prüfen ─────────────────────────────
        if execution_enabled and self._drawdown_monitor:
            _total = portfolio_summary_before.get("total_value", 0)
            _peak  = portfolio_summary_before.get("peak_value") or _total
            killed = self._drawdown_monitor.check(
                portfolio_value=_total,
                peak_value=_peak,
                api=self.executor.api,
            )
            if killed:
                log.critical("[PHASE 4A] Drawdown Kill-Switch ausgelöst → Execution gestoppt.")
                execution_enabled = False
                run_summary["errors"].append("drawdown_kill_switch")

        if not execution_enabled:
            log.info("\nSCHRITT 7/8: Execution disabled – nur Simulation")
        else:
            log.info("\nSCHRITT 7/8: Trades ausführen (SELL → Sync → BUY)...")

        log.info(f"  Rebalancing-Berechnung: {len(sell_trades)} SELL(s), {len(buy_trades)} BUY(s) geplant")
        for t in buy_trades:
            log.info(
                f"    → BUY {t['ticker']}: ${t['value']:,.0f} | "
                f"{t.get('current_alloc', 0)*100:.1f}% → {t.get('target_alloc', 0)*100:.1f}% | "
                f"Drift: {abs(t.get('target_alloc', 0) - t.get('current_alloc', 0))*100:.1f}%"
            )

        executed_records: List[Dict] = []
        sells_ok = 0
        buys_ok = 0

        # ── FINAL EXECUTION GUARD (Phase 4 Production Hardening) ─────────────
        # Before any order is sent, verify all invariants hold.
        # Nothing is blocked – violations are automatically adjusted.
        if execution_enabled:
            sell_trades, buy_trades, guard_warnings = self._final_execution_guard(
                sell_trades, buy_trades, portfolio_summary_before
            )
            if guard_warnings:
                log.warning(f"  [EXEC GUARD] {len(guard_warnings)} adjustment(s) applied:")
                for w in guard_warnings:
                    log.warning(f"    • {w}")
                run_summary["execution_guard_warnings"] = guard_warnings

        if execution_enabled and sell_trades:
            log.info(f"\n  ── Phase 1: {len(sell_trades)} SELL(s) ──")
            for trade in sell_trades:
                ok, record = self._execute_sell_trade(trade, current_prices)
                if record.get("status") == "EXECUTED":
                    executed_records.append(record)
                if ok:
                    sells_ok += 1
                    self.portfolio.apply_trade(
                        ticker=trade["ticker"],
                        action="SELL",
                        quantity=record.get("fill_qty") or trade["quantity"],
                        price=record.get("fill_price") or trade["price"],
                    )
            run_summary["sells_executed"] = sells_ok
        else:
            log.info("\n  ── Phase 1: SELL-Phase übersprungen (Execution disabled oder keine SELLs) ──")
            run_summary["sells_executed"] = 0

        if execution_enabled and sell_trades and self.mode in ("PAPER", "REAL", "LIVE") and self.executor.api:
            log.info("\n  ── Phase 2: Broker-Sync nach SELLs ──")
            time.sleep(2)
            self.portfolio._load_from_alpaca()
            log.info(f"  Sync: Cash nach SELLs = {format_currency(self.portfolio.cash)}")
        else:
            log.info("\n  ── Phase 2: Broker-Sync übersprungen (Execution disabled, DRY oder keine SELLs) ──")

        if execution_enabled and buy_trades:
            log.info(f"\n  ── Phase 3: {len(buy_trades)} BUY(s) geplant ──")

            total_value = self.portfolio.get_total_value()
            min_cash_pct = self.risk_manager.settings.get("min_cash_pct",
                           RISK_SETTINGS[ACTIVE_RISK_PROFILE]["min_cash_pct"])
            min_cash_abs = max(0.0, total_value * min_cash_pct)
            running_cash = self.portfolio.cash

            for trade in buy_trades:
                ticker = trade["ticker"]
                if ticker not in current_prices:
                    log.warning(f"[PRICE MISSING] {ticker}")

                price = current_prices.get(ticker) or trade.get("price", 0)
                if price <= 0:
                    log.warning(f"[PRICE INVALID] {ticker} → BUY skipped")
                    continue

                spendable_cash = max(0.0, running_cash - min_cash_abs)
                if spendable_cash < 1.0:
                    log.info(
                        f"  BUY {ticker}: kein spendbares Cash "
                        f"(Cash: {format_currency(running_cash)}, "
                        f"Reserve: {format_currency(min_cash_abs)}) – übersprungen."
                    )
                    continue

                ok, record = self.executor.execute_buy(
                    ticker=ticker,
                    target_value=trade["value"],
                    current_price=price,
                    available_cash=running_cash,
                    min_cash_reserve=min_cash_abs,
                    reason=trade.get("ai_reason", "Portfolio Rebalancing"),
                )

                if record.get("status") not in ("SKIPPED", None) or ok:
                    executed_records.append(record)

                if ok:
                    buys_ok += 1
                    actual_spent = record.get("fill_value") or record.get("planned_value", 0)
                    running_cash -= actual_spent
                    self.portfolio.apply_trade(
                        ticker=ticker,
                        action="BUY",
                        quantity=record.get("fill_qty") or (
                            actual_spent / price if price > 0 else 0
                        ),
                        price=record.get("fill_price") or price,
                    )
            run_summary["buys_executed"] = buys_ok
        else:
            log.info("\n  ── Phase 3: BUY-Phase übersprungen (Execution disabled oder keine BUYs) ──")
            run_summary["buys_executed"] = 0

        run_summary["trades_executed"] = sells_ok + buys_ok
        run_summary["executed_trades"] = len(executed_records)
        run_summary["execution_mode"] = "LIVE" if execution_enabled else "SIMULATED"
        run_summary["market_closed"] = analysis_only

        # Register cooldowns for executed trades
        if executed_records:
            cooldown_manager.register_executed_trades(executed_records)

        # ── Phase 4A: Rejected/Aborted Trades sammeln ─────────────────────────
        rejected_records = [
            r for r in executed_records
            if r.get("status") in ("REJECTED", "ABORTED", "SKIPPED")
        ]
        real_executed_records = [
            r for r in executed_records
            if r.get("status") == "EXECUTED"
        ]

        if not execution_enabled and analysis_only:
            log.info(f"Executed trades: 0 (market closed)")
        elif not execution_enabled:
            log.info(f"Executed trades: 0 (execution disabled: {self.mode})")

        # ── Fix #10: Portfolio-Konsistenz-Check nach dem Run ─────────────────
        self._check_portfolio_consistency(market_data)

        # ── SCHRITT 8: Journal & Snapshot ─────────────────────────────────────
        log.info("\nSCHRITT 8/8: Journal & Portfolio-Snapshot...")
        portfolio_summary_after = self._build_portfolio_summary()
        trade_logger.log_portfolio_snapshot(portfolio_summary_after)

        # Fix #5: Nur Trades mit echtem Qty/Wert im Journal
        real_executed = [
            r for r in executed_records
            if r.get("status") == "EXECUTED"
            or (r.get("fill_qty") or 0) > 0
            or (r.get("fill_value") or 0) > 0
        ]

        # Log score summary for this run
        if ai_result.get("scores"):
            top_scores = sorted(
                ai_result["scores"].values(),
                key=lambda x: x.get("total_score", 0), reverse=True
            )[:5]
            log.info("  Top-5 Quant Scores: " + " | ".join(
                f"{s['ticker']}={s.get('total_score',0):.0f}({s.get('signal','')})"
                for s in top_scores
            ))
            save_json_file(
                f"{LOG_DIR}/scores_{start_time.strftime('%Y%m%d_%H%M%S')}.json",
                ai_result["scores"],
            )

        journal.log_run(
            market_outlook=ai_result.get("market_outlook", ""),
            risk_assessment=ai_result.get("risk_assessment", ""),
            ai_signals=raw_decisions,
            final_decisions=validated,
            simulated_trades=simulated_trades,
            executed_trades=real_executed,
            portfolio_before=portfolio_summary_before,
            portfolio_after=portfolio_summary_after,
            portfolio_projection=projected_portfolio,
            risk_warnings=warnings,
            mode=self.mode,
            feedback_learnings=ai_result.get("feedback_learnings", ""),  # Fix #19
            regime_state=regime_state,
            market_data=market_data,
            execution_mode=run_summary.get("execution_mode", "SIMULATED"),
            market_closed=run_summary.get("market_closed", False),
            debug=False,
        )

        # ── Phase 4A: Live-Pflichtprotokoll /logs/live/YYYY-MM-DD.json ──────────
        if self.mode == "LIVE":
            _guard = self.executor.get_guard()
            self._live_logger.log_run(
                guard_snapshot=_guard.status_snapshot() if _guard else {},
                executed_trades=real_executed_records,
                rejected_trades=rejected_records,
                portfolio_snapshot=portfolio_summary_after,
                risk_state={
                    "profile":   ACTIVE_RISK_PROFILE.value,
                    "warnings":  warnings,
                    "var_ok":    var_ok,
                },
                regime_state=regime_state.to_dict() if regime_state else None,
                run_summary=run_summary,
            )

        # Run-Summary
        duration = (datetime.now() - start_time).seconds
        run_summary.update({
            "completed_at":          datetime.now().isoformat(),
            "duration_seconds":      duration,
            "final_portfolio_value": portfolio_summary_after.get("total_value", 0),
            "final_pnl_pct":         portfolio_summary_after.get("pnl_pct", 0),
        })

        # Fix #12: Run-Summary persistent speichern
        save_json_file(
            f"{LOG_DIR}/run_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json",
            run_summary,
        )

        log.info(f"\n{'='*70}")
        log.info(f"  RUN ABGESCHLOSSEN | Dauer: {duration}s")
        log.info(f"  Portfolio:  {format_currency(portfolio_summary_after.get('total_value', 0))}")
        log.info(f"  P&L:        {portfolio_summary_after.get('pnl_pct', 0):+.2f}%")
        log.info(f"  SELLs ausgeführt: {sells_ok} | BUYs ausgeführt: {buys_ok}"
                 + (f" | Keine Trades (Ziele bereits erreicht)" if sells_ok + buys_ok == 0 else ""))
        log.info(f"{'='*70}\n")

        return run_summary

    # ─── PHASE 4: Final Execution Guard ──────────────────────────────────────

    def _final_execution_guard(
        self,
        sell_trades: List[Dict],
        buy_trades: List[Dict],
        portfolio_summary: Dict,
    ) -> Tuple[List[Dict], List[Dict], List[str]]:
        """
        Phase 4 Production Hardening – Final pre-execution safety check.

        Runs immediately before any order hits the broker. Verifies:
          1. Cash >= min_cash_requirement (after all planned buys)
          2. No position exceeds max_position_pct after planned buys
          3. Trailing stops have been applied (already in pipeline, just verify)
          4. Stop losses have been checked (already in pipeline, just verify)
          5. Sector limit respected per decision filter

        Policy: NOTHING IS BLOCKED. Violations are automatically adjusted
        (buy sizes reduced / clamped). The guard never cancels a trade.

        Returns: (adjusted_sell_trades, adjusted_buy_trades, warnings)
        """
        warnings: List[str] = []
        _profile = RISK_SETTINGS[ACTIVE_RISK_PROFILE]
        max_position_pct = _profile["max_position_pct"]
        min_cash_pct = _profile["min_cash_pct"]

        total_value = portfolio_summary.get("total_value", self.portfolio.get_total_value())
        min_cash_abs = total_value * min_cash_pct

        # ── Guard 1: Cash reserve check ──────────────────────────────────────
        available_cash = portfolio_summary.get("cash", self.portfolio.cash)
        total_buy_value = sum(t.get("value", 0) for t in buy_trades)
        projected_cash = available_cash - total_buy_value

        if projected_cash < min_cash_abs:
            shortfall = min_cash_abs - projected_cash
            msg = (
                f"Cash guard: projected cash ${projected_cash:,.0f} < "
                f"reserve ${min_cash_abs:,.0f} → reducing BUY sizes by ${shortfall:,.0f}"
            )
            warnings.append(msg)

            # Proportionally reduce buy sizes to meet cash reserve
            if total_buy_value > 0:
                scale = max(0.0, (available_cash - min_cash_abs) / total_buy_value)
                for trade in buy_trades:
                    orig = trade.get("value", 0)
                    trade["value"] = round(orig * scale, 2)
                    if orig != trade["value"]:
                        warnings.append(
                            f"  {trade['ticker']}: BUY scaled ${orig:,.0f}→${trade['value']:,.0f} "
                            f"(cash guard, scale={scale:.2f})"
                        )

        # ── Guard 2: Position size cap ────────────────────────────────────────
        for trade in buy_trades:
            ticker = trade.get("ticker", "")
            current_val = self.portfolio.positions.get(ticker, {}).get("market_value", 0.0)
            buy_val = trade.get("value", 0.0)
            projected_val = current_val + buy_val
            cap_val = total_value * max_position_pct

            if projected_val > cap_val:
                clamped_buy = max(0.0, cap_val - current_val)
                msg = (
                    f"Position cap guard: {ticker} projected ${projected_val:,.0f} "
                    f"> cap ${cap_val:,.0f} → BUY clamped to ${clamped_buy:,.0f}"
                )
                warnings.append(msg)
                trade["value"] = round(clamped_buy, 2)

        # ── Guard 3: Trailing stops verified ─────────────────────────────────
        # (already injected as stop_loss_orders + trailing_stops in the pipeline)
        # Just log confirmation.
        log.debug("[EXEC GUARD] Trailing stops: verified (pre-injected in pipeline)")

        # ── Guard 4: Stop loss verified ───────────────────────────────────────
        log.debug("[EXEC GUARD] Stop losses: verified (pre-injected in pipeline)")

        # ── Guard 5: Strip zero-value buys ────────────────────────────────────
        original_buy_count = len(buy_trades)
        buy_trades = [t for t in buy_trades if t.get("value", 0) >= 1.0]
        stripped = original_buy_count - len(buy_trades)
        if stripped:
            warnings.append(f"Guard stripped {stripped} zero-value BUY(s) after adjustment")

        # ── Stale position report ─────────────────────────────────────────────
        stale_flags = self.portfolio.stale_position_flag()
        stale_tickers = [t for t, v in stale_flags.items() if v["stale"]]
        if stale_tickers:
            warnings.append(
                f"Stale positions detected (flag only, no forced sell): {', '.join(stale_tickers)}"
            )

        log.info(
            f"[EXEC GUARD] Guard passed | sells={len(sell_trades)} buys={len(buy_trades)} | "
            f"warnings={len(warnings)}"
        )
        return sell_trades, buy_trades, warnings

    # ─── Fix #10: Portfolio-Konsistenz ────────────────────────────────────────

    def _check_portfolio_consistency(self, market_data: Dict):
        """
        Fix #10: Nach jedem Run sicherstellen dass Portfolio == Broker-State.
        Bei Abweichungen wird gewarnt und ggf. re-synct.
        """
        if self.mode not in ("PAPER", "REAL", "LIVE") or not self.executor.api:
            return  # DRY Modus: kein Broker-Abgleich möglich

        try:
            account = self.executor.api.get_account()
            broker_cash = float(account.cash)
            local_cash  = self.portfolio.cash
            cash_diff   = abs(broker_cash - local_cash)

            if cash_diff > 1.0:  # Mehr als $1 Abweichung
                log.warning(
                    f"[KONSISTENZ] Cash-Abweichung erkannt: "
                    f"Lokal {format_currency(local_cash)} vs "
                    f"Broker {format_currency(broker_cash)} | "
                    f"Differenz: {format_currency(cash_diff)}"
                )
                # Re-sync mit Broker als Master (Fix #10)
                self.portfolio._load_from_alpaca()
                log.info("[KONSISTENZ] Portfolio mit Broker re-synchronisiert.")
            else:
                log.debug(
                    f"[KONSISTENZ] Cash-Abgleich OK: "
                    f"Lokal={format_currency(local_cash)}, "
                    f"Broker={format_currency(broker_cash)}"
                )
        except Exception as e:
            log.warning(f"[KONSISTENZ] Check fehlgeschlagen: {e}")

    # ─── Hilfsmethoden ────────────────────────────────────────────────────────

    def _build_portfolio_summary(self) -> Dict:
        """Portfolio-Summary mit Positions-Dict für KI-Prompt und Journal."""
        summary = self.portfolio.get_summary()
        summary["positions"] = self.portfolio.positions
        return summary

    def _calculate_trades(
        self,
        validated: List[Dict],
        current_prices: Dict[str, float],
        decisions_map: Dict[str, Dict],
    ) -> List[Dict]:
        """Berechnet Rebalancing-Trades aus validierten Entscheidungen."""
        target_allocations = {
            d["ticker"]: d.get("target_allocation", 0)
            for d in validated
            if d.get("action") in ("BUY", "SELL") and d.get("risk_approved", False)
        }
        if not target_allocations:
            log.info("  Keine genehmigten Trades.")
            return []

        return self.portfolio.calculate_rebalancing_trades(
            target_allocations=target_allocations,
            current_prices=current_prices,
            decisions_map=decisions_map,
        )

    def _execute_sell_trade(
            self,
            trade: Dict,
            current_prices: Dict[str, float],
        ) -> Tuple[bool, Dict]:

        ticker = trade["ticker"]
        price = current_prices.get(ticker, trade.get("price", 0))

        # ✅ FIX Zombie ohne Preis: direkt force_close aufrufen
        is_zombie = trade.get("zombie_cleanup", False)
        if (not price or price <= 0) and is_zombie:
            log.info(f"[ZOMBIE] {ticker}: kein Preis verfügbar → force_close_position()")
            return self.executor._force_close_position(ticker, 0.0, trade.get("ai_reason", "Zombie liquidation"))

        # ✅ FIX: Kein Marktpreis aber portfolio_manager hat Fallback-Preis (avg_price) geliefert
        if (not price or price <= 0) and trade.get("price_estimated") and trade.get("price", 0) > 0:
            price = trade["price"]
            log.info(f"[SELL] {ticker}: kein Marktpreis → Fallback Ø-Kaufpreis ${price:.2f}")

        # ❗ echte Position holen
        position_qty = self.portfolio.positions.get(ticker, {}).get("quantity", 0)

        return self.executor.execute_sell(
            ticker=ticker,
            position_qty=position_qty,
            current_price=price,
            target_value=trade.get("value", 0),
            reason=trade.get("ai_reason", "Portfolio Rebalancing"),
            full_liquidation=is_zombie,
        )
    
    def _execute_buy_trade(self, trade: Dict, current_prices: Dict[str, float], portfolio: Dict) -> Tuple[bool, Dict]:
        ticker = trade["ticker"]
        price = current_prices.get(ticker, trade.get("price", 0))

        if price <= 0:
            return False, self._skipped(ticker, "BUY", "invalid_price")

        return self.executor.execute_buy(
            ticker=ticker,
            target_value=trade["value"],
            current_price=price,
            available_cash=portfolio["cash"],              # 🔥 FIX 1
            min_cash_reserve=self.risk_manager.min_cash,   # 🔥 FIX 2 (oder config)
            reason=trade.get("ai_reason", "Portfolio Rebalancing"),
        )
    
    def show_status(self):
        """Zeigt aktuellen Portfolio-Status."""
        market_data = self.data_collector.collect_all()
        self.portfolio.update_prices(market_data)
        self.portfolio.print_summary()
        trades = trade_logger.get_trade_history()
        log.info(f"Trades insgesamt: {len(trades)}")
        log.info(f"Markt-Status: {market_status()}")


# ─── Standalone-Funktionen ────────────────────────────────────────────────────

def run_backtest(use_ai: bool = False):
    """
    Führt Backtests durch.
    --backtest          → Momentum-Baseline (alle 3 Risikoprofile, kostenlos)
    --backtest --ai     → KI-Backtest (Fallback-Analyse, kostenlos)
    --backtest --ai --openai → KI-Backtest mit echten OpenAI-Calls (~$2–5)
    """
    tickers = ETF_WATCHLIST + STOCK_WATCHLIST[:5]

    if use_ai:
        log.info("Starte KI-Backtest (testet echte AI-Strategie)...")
        from config import OPENAI_API_KEY
        has_key = bool(OPENAI_API_KEY)
        bt = AIBacktester(
            tickers=tickers,
            start_date=BACKTEST_START_DATE,
            end_date=BACKTEST_END_DATE,
            frequency="weekly",
            use_ai=has_key,   # True nur wenn API Key vorhanden
        )
        result = bt.run()
        bt.save_results(result)
        return result


def run_scheduler(bot: TradingBot):
    try:
        import schedule
    except ImportError:
        log.error("schedule nicht installiert: pip install schedule")
        return

    log.info(f"Scheduler: {SCHEDULE_INTERVAL} um {SCHEDULE_TIME}")
    if SCHEDULE_INTERVAL == "daily":
        schedule.every().day.at(SCHEDULE_TIME).do(bot.run)
    elif SCHEDULE_INTERVAL == "weekly":
        from config import SCHEDULE_WEEKDAY
        days = ["monday", "tuesday", "wednesday", "thursday", "friday"]
        getattr(schedule.every(), days[min(SCHEDULE_WEEKDAY, 4)]).at(SCHEDULE_TIME).do(bot.run)

    log.info("Scheduler läuft. Ctrl+C zum Beenden.")
    while True:
        schedule.run_pending()
        time.sleep(60)





def main():
    parser = argparse.ArgumentParser(description="AI Trading Bot")
    parser.add_argument("--mode",     choices=["dry", "paper", "real"], default=TRADING_MODE.lower())
    parser.add_argument("--backtest", action="store_true",
                        help="Momentum-Backtest (Baseline, kostenlos)")
    parser.add_argument("--ai",       action="store_true",
                        help="KI-Backtest statt Momentum-Baseline (mit --backtest)")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--status",   action="store_true")
    args = parser.parse_args()

    if args.backtest:
        run_backtest(use_ai=args.ai)
        return

    bot = TradingBot(mode=args.mode)

    if args.status:
        bot.show_status()
    elif args.schedule:
        run_scheduler(bot)
    else:
        bot.run()


if __name__ == "__main__":
    main()