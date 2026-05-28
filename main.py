"""
AI Trading Bot - Hauptprogramm (Production) mit Portfolio Rebalancing Engine
==============================================================
Integriert die neue Portfolio Rebalancing Engine für ganzheitliche,
transaktionskostenbewusste Portfolio-Optimierung.
"""

import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from logger import log, trade_logger
from config import (
    TRADING_MODE, FULL_WATCHLIST, ETF_WATCHLIST, STOCK_WATCHLIST,
    ACTIVE_RISK_PROFILE, RISK_SETTINGS, SCHEDULE_INTERVAL, SCHEDULE_TIME,
    BACKTEST_START_DATE, BACKTEST_END_DATE, INITIAL_CAPITAL,
    LOG_DIR, CORRELATION_GROUPS, REBALANCING_ENGINE_ENABLED,
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
from market_regime_enhanced import EnhancedMarketRegimeDetector
from decision_filter import apply_decision_filter
from execution_safety import LiveRunLogger, DrawdownMonitor
from config import ALLOW_LIVE_TRADING, KILL_SWITCH_DRAWDOWN_PCT
from capital_rotator import CapitalRotator
from portfolio_rebalancer import PortfolioRebalancer, RebalancingDecision

import os
os.makedirs(LOG_DIR, exist_ok=True)


class TradingBot:
    """
    Orchestriert alle Module für den vollautomatischen Trading-Ablauf.
    Verwendet jetzt die Portfolio Rebalancing Engine als zentrale Optimierungsinstanz.
    """

    def __init__(self, mode: str = None):
        self.mode = (mode or TRADING_MODE).upper()
        log.info("=" * 70)
        log.info("  AI TRADING BOT – INITIALISIERUNG")
        log.info(f"  Modus: {self.mode} | Risikoprofil: {ACTIVE_RISK_PROFILE.value.upper()}")
        log.info(f"  Watchlist: {len(FULL_WATCHLIST)} Assets")
        log.info(f"  Markt-Status: {market_status()}")
        log.info(f"  Rebalancing Engine: {'AKTIV' if REBALANCING_ENGINE_ENABLED else 'INAKTIV'}")
        log.info("=" * 70)

        self.data_collector = MarketDataCollector(FULL_WATCHLIST)
        self.news_collector = NewsCollector()
        self.ai_analyzer = AIAnalyzer()

        self.executor = TradeExecutor(mode=self.mode)
        alpaca_api = self.executor.api if self.mode in ("PAPER", "REAL", "LIVE") else None
        self.portfolio = PortfolioManager(mode=self.mode, alpaca_api=alpaca_api)
        self.risk_manager = RiskManager(ACTIVE_RISK_PROFILE)

        # Neue Portfolio Rebalancing Engine
        self.rebalancer = PortfolioRebalancer(ACTIVE_RISK_PROFILE)

        self._live_logger = LiveRunLogger()
        _guard = self.executor.get_guard()
        self._drawdown_monitor = DrawdownMonitor(guard=_guard, limit_pct=KILL_SWITCH_DRAWDOWN_PCT) if _guard else None

    # ─── Hauptablauf ──────────────────────────────────────────────────────────

    def run(self) -> Dict:
        start_time = datetime.now()
        log.info(f"\n{'='*70}")
        log.info(f"  TRADING RUN GESTARTET: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        market_open = is_market_open()
        analysis_only = not market_open

        log.info(f"  Markt: {market_status()}")
        if analysis_only:
            log.warning("Market closed -> analysis-only mode activated")
        log.info(f"{'='*70}\n")

        run_summary = {
            "started_at": start_time.isoformat(),
            "mode": self.mode,
            "risk_profile": ACTIVE_RISK_PROFILE.value,
            "market_status": market_status(),
            "regime": None,
            "sells_executed": 0,
            "buys_executed": 0,
            "trades_executed": 0,
            "errors": [],
            "risk_warnings": [],
            "analysis_only": analysis_only,
        }

        # SCHRITT 0: Markt-Regime erkennen
        log.info("SCHRITT 0/8: Markt-Regime-Detektion...")
        try:
            regime_detector = EnhancedMarketRegimeDetector(watchlist=FULL_WATCHLIST)
            regime_state = regime_detector.detect()
            self.risk_manager.apply_regime(regime_state)
            run_summary["regime"] = regime_state.to_dict()
            save_json_file(f"{LOG_DIR}/regime_{start_time.strftime('%Y%m%d_%H%M%S')}.json", regime_state.to_dict())
        except Exception as e:
            log.warning(f"Regime-Detektion fehlgeschlagen: {e} – fahre ohne Regime-Override fort.")
            regime_state = None

        # SCHRITT 1: Marktdaten
        log.info("SCHRITT 1/8: Marktdaten sammeln...")
        market_data = self.data_collector.collect_all()
        if not market_data:
            log.error("Keine Marktdaten – Run abgebrochen.")
            run_summary["errors"].append("no_market_data")
            return run_summary

        # SCHRITT 2: News
        log.info("\nSCHRITT 2/8: News & Sentiment sammeln...")
        news_articles = self.news_collector.collect_all()
        news_text = self.news_collector.format_for_ai(news_articles)

        # SCHRITT 3: Portfolio-Snapshot
        log.info("\nSCHRITT 3/8: Portfolio aktualisieren...")
        self.portfolio.update_prices(market_data)
        portfolio_summary_before = self._build_portfolio_summary()
        self.portfolio.print_summary()

        # Zombie-Positionen
        zombie_tickers = find_zombie_positions(self.portfolio.positions)
        zombie_orders = build_zombie_sell_orders(zombie_tickers, self.portfolio.positions)

        # SCHRITT 4: KI-Analyse & Entscheidungsfindung
        log.info("\nSCHRITT 4/8: KI-Analyse & Entscheidungsfindung...")
        log.info("\nSCHRITT 3.5/8: Trailing Stop Check...")
        trailing_stops = self.portfolio.get_trailing_stop_triggers(market_data, trail_pct=0.12)
        if trailing_stops:
            log.info(f"  {len(trailing_stops)} Trailing Stop Trigger(s) erkannt")

        effective_watchlist = list(set(FULL_WATCHLIST) | set(self.portfolio.positions.keys()))

        # ScoreEngine für Scores, Momentum, Volatilität
        from score_engine import ScoreEngine
        score_engine = ScoreEngine(positions=self.portfolio.positions, total_value=portfolio_summary_before.get("total_value", 100_000))
        spy_return = market_data.get("SPY", {}).get("return_20d")
        scores_obj = score_engine.score_all(market_data, regime_state, spy_return)
        scores = {ticker: sb.total_score for ticker, sb in scores_obj.items()}
        momentum = {ticker: sb.momentum_20d or 0.0 for ticker, sb in scores_obj.items()}
        volatility = {ticker: sb.volatility_annual or 20.0 for ticker, sb in scores_obj.items()}
        current_weights = self.portfolio.get_allocations()
        total_value = portfolio_summary_before.get("total_value", self.portfolio.get_total_value())
        cash = portfolio_summary_before.get("cash", 0.0)

        # Portfolio Rebalancing Engine (neu)
        rebalancing_decisions, target_weights, cash_target, rebalance_rationale = self.rebalancer.optimize_portfolio(
            scores=scores,
            market_data=market_data,
            current_weights=current_weights,
            cash=cash,
            total_value=total_value,
            regime_state=regime_state,
        )

        # Konvertiere Rebalancing-Entscheidungen in das alte Decision-Format für Kompatibilität
        rebalancing_actions = []
        for d in rebalancing_decisions:
            rebalancing_actions.append({
                "ticker": d.ticker,
                "action": d.action,
                "target_allocation": d.target_weight,
                "confidence": d.confidence,
                "reason": d.reason,
                "risk_approved": True,
                "rebalancing_engine": True,
            })

        # KI-Analyse (nutzt die Rebalancing-Vorschläge als Basis)
        ai_input_log = {"timestamp": start_time.isoformat(), "watchlist": effective_watchlist, "portfolio": portfolio_summary_before}
        save_json_file(f"{LOG_DIR}/ai_input_{start_time.strftime('%Y%m%d_%H%M%S')}.json", ai_input_log)

        recent_journal = journal.get_history()[-10:]
        if recent_journal:
            log.info(f"  Feedback-Loop: {len(recent_journal)} vergangene Runs geladen.")

        # Wir übergeben die Rebalancing-Zielallokationen an die KI als "portfolio_allocation"
        # Dazu bauen wir ein einfaches Dict (mock des PortfolioAllocation)
        class MockAllocation:
            def __init__(self, targets, cash_t, rationale):
                self.target_allocations = targets
                self.cash_target = cash_t
                self.rationale = rationale
                self.recommended_sells = []
        mock_allocation = MockAllocation(target_weights, cash_target, rebalance_rationale)

        ai_result = self.ai_analyzer.analyze(
            portfolio_summary=portfolio_summary_before,
            market_data=market_data,
            news_text=news_text,
            watchlist=effective_watchlist,
            journal_entries=recent_journal,
            regime_state=regime_state,
            portfolio_allocation=mock_allocation,
        )
        raw_decisions = [dict(d) for d in ai_result.get("decisions", [])]

        # Mische KI-Entscheidungen mit den Rebalancing-Entscheidungen (Rebalancing hat höhere Priorität)
        decision_map = {d["ticker"]: d for d in raw_decisions}
        for rd in rebalancing_actions:
            ticker = rd["ticker"]
            if ticker in decision_map:
                log.debug(f"Überschreibe KI-Entscheidung für {ticker} mit Rebalancing-Engine: {rd['action']}")
            decision_map[ticker] = rd
        merged_decisions = list(decision_map.values())

        # DecisionFilter
        log.info("\nSCHRITT 4b/8: Decision quality filter...")
        filtered_decisions, filter_warnings = apply_decision_filter(
            decisions=merged_decisions,
            risk_settings=RISK_SETTINGS[ACTIVE_RISK_PROFILE],
            positions=self.portfolio.positions,
            total_value=portfolio_summary_before.get("total_value", 100_000),
            market_data=market_data,
            regime_state=regime_state,
        )
        if filter_warnings:
            run_summary["decision_filter_warnings"] = filter_warnings

        save_json_file(f"{LOG_DIR}/ai_output_{start_time.strftime('%Y%m%d_%H%M%S')}.json", ai_result)

        decisions = filtered_decisions

        # Cooldown-Filter
        log.info("\nSCHRITT 4c/8: Cooldown-Filter...")
        decisions, cooldown_warnings = cooldown_manager.filter_decisions(decisions)
        if cooldown_warnings:
            run_summary["cooldown_warnings"] = cooldown_warnings
        log.info(f"  {len(decisions)} Entscheidungen | {ai_result.get('market_outlook', '')[:80]}")
        for d in decisions:
            log.info(f"  → {d['action']:<5} {d['ticker']:<6} Allok: {d.get('target_allocation', 0):.0%} | Konfidenz: {d.get('confidence', 0):.0%}")

        # SCHRITT 5: Stop-Loss & Zombie-Cleanup
        log.info("\nSCHRITT 5/8: Stop-Loss & Zombie-Cleanup...")
        stop_loss_orders = self.risk_manager.check_stop_loss(self.portfolio.positions, market_data)
        decisions = stop_loss_orders + trailing_stops + zombie_orders + decisions

        normalized_decisions, norm_warnings = normalize_ai_decisions(
            decisions=decisions,
            positions=self.portfolio.positions,
            total_value=portfolio_summary_before.get("total_value", 100_000),
            market_data=market_data,
            correlation_groups=CORRELATION_GROUPS,
        )
        if norm_warnings:
            run_summary["normalization_warnings"] = norm_warnings

        # SCHRITT 6: Risikoprüfung & Validierung
        log.info("\nSCHRITT 6/8: Risikoprüfung & Validierung...")
        validated, risk_warnings = self.risk_manager.validate_decisions(
            decisions=normalized_decisions,
            portfolio_summary=portfolio_summary_before,
            market_data=market_data,
        )
        run_summary["risk_warnings"] = risk_warnings

        decisions_map = {d["ticker"]: d for d in validated}
        current_prices = {t: d.get("current_price", 0) for t, d in market_data.items()}
        _profile_settings = RISK_SETTINGS[ACTIVE_RISK_PROFILE]
        total_value = portfolio_summary_before.get("total_value", self.portfolio.get_total_value())
        min_trade_value = _profile_settings.get("min_trade_value", 100.0)
        if total_value < 10000:
            dynamic_min = max(10.0, total_value * 0.05)
            if dynamic_min < min_trade_value:
                log.info(f"Small account: lowering min_trade_value from ${min_trade_value:.2f} to ${dynamic_min:.2f}")
                min_trade_value = dynamic_min

        # Rebalancing-Berechnung basierend auf validierten Entscheidungen
        planned_trades = self.portfolio.calculate_rebalancing_trades(
            target_allocations={
                d["ticker"]: d.get("target_allocation", 0)
                for d in validated
                if d.get("action") in ("BUY", "SELL")
            },
            current_prices=current_prices,
            decisions_map=decisions_map,
            min_trade_value=min_trade_value,
        )

        # ===== CAPITAL ROTATION (optional) =====
        approved_buys = [d for d in validated if d.get("action") == "BUY" and d.get("risk_approved")]
        if approved_buys and total_value > 2000:
            rotator = CapitalRotator()
            cooldown_set = set(cooldown_manager._data.keys()) if hasattr(cooldown_manager, '_data') else set()
            rotation_candidates = rotator.find_rotation_candidates(
                new_buy_tickers=[d['ticker'] for d in approved_buys],
                new_scores=scores,
                current_positions=self.portfolio.positions,
                total_value=total_value,
                market_data=market_data,
                regime_state=regime_state,
                cooldown_tickers=cooldown_set,
            )
            for sell_ticker, buy_ticker, diff, reason in rotation_candidates:
                forced_sell = {
                    "ticker": sell_ticker,
                    "action": "SELL",
                    "target_allocation": 0.0,
                    "confidence": 1.0,
                    "reason": f"Capital rotation: replaced by {buy_ticker} ({reason})",
                    "risk_approved": True,
                    "forced_rebalancing": True,
                }
                validated.append(forced_sell)
                decisions_map[sell_ticker] = forced_sell
                log.info(f"  Capital Rotation: {sell_ticker} → {buy_ticker} (Diff: {diff:.0f}, {reason})")
            if rotation_candidates:
                planned_trades = self.portfolio.calculate_rebalancing_trades(
                    target_allocations={
                        d["ticker"]: d.get("target_allocation", 0)
                        for d in validated
                        if d.get("action") in ("BUY", "SELL")
                    },
                    current_prices=current_prices,
                    decisions_map=decisions_map,
                    min_trade_value=min_trade_value,
                )

        sell_trades = [t for t in planned_trades if t["action"] == "SELL"]
        buy_trades = [t for t in planned_trades if t["action"] == "BUY"]

        # Debug-Ausgaben
        log.info(f"🔍 DEBUG: analysis_only={analysis_only}, mode={self.mode}, has_api={self.executor.api is not None}")
        log.info(f"🔍 DEBUG: sell_trades count={len(sell_trades)}, buy_trades count={len(buy_trades)}")

        execution_enabled = (
            not analysis_only
            and self.mode in ("PAPER", "REAL", "LIVE")
            and self.executor.api is not None
        )

        if execution_enabled:
            _guard = self.executor.get_guard()
            if _guard:
                md_ok = _guard.validate_market_data(market_data)
                ai_ok = _guard.validate_ai_response(ai_result)
                log.info(f"🔍 DEBUG: guard market_data_ok={md_ok}, ai_response_ok={ai_ok}")
                if not md_ok or not ai_ok:
                    log.critical(f"Guard-Validierung fehlgeschlagen -> Execution deaktiviert.")
                    execution_enabled = False
                    run_summary["errors"].append("guard_validation_failed")
            if execution_enabled and self._drawdown_monitor:
                _total = portfolio_summary_before.get("total_value", 0)
                _peak = portfolio_summary_before.get("peak_value") or _total
                if self._drawdown_monitor.check(portfolio_value=_total, peak_value=_peak, api=self.executor.api):
                    log.critical("Drawdown Kill-Switch ausgelöst -> Execution gestoppt.")
                    execution_enabled = False
                    run_summary["errors"].append("drawdown_kill_switch")

        if execution_enabled:
            sell_trades, buy_trades, guard_warnings = self._final_execution_guard(
                sell_trades, buy_trades, portfolio_summary_before
            )
            if guard_warnings:
                run_summary["execution_guard_warnings"] = guard_warnings

        if not execution_enabled:
            log.info("\nSCHRITT 7/8: Execution disabled – nur Simulation")
        else:
            log.info("\nSCHRITT 7/8: Trades ausführen (SELL → Sync → BUY)...")

        executed_records = []
        sells_ok = 0
        buys_ok = 0

        # Phase 1: SELLs
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
            log.info("\n  ── Phase 1: SELL-Phase übersprungen")

        # Phase 2: Broker-Sync
        if execution_enabled and sell_trades and self.mode in ("PAPER", "REAL", "LIVE") and self.executor.api:
            log.info("\n  ── Phase 2: Broker-Sync nach SELLs ──")
            time.sleep(2)
            self.portfolio._load_from_alpaca()
            log.info(f"  Sync: Cash nach SELLs = {format_currency(self.portfolio.cash)}")
        else:
            log.info("\n  ── Phase 2: Broker-Sync übersprungen")

        # Phase 3: BUYs
        if execution_enabled and buy_trades:
            log.info(f"\n  ── Phase 3: {len(buy_trades)} BUY(s) geplant ──")
            total_value = self.portfolio.get_total_value()
            min_cash_pct = self.risk_manager.settings.get("min_cash_pct", RISK_SETTINGS[ACTIVE_RISK_PROFILE]["min_cash_pct"])
            min_cash_abs = max(0.0, total_value * min_cash_pct)
            running_cash = self.portfolio.cash

            for trade in buy_trades:
                ticker = trade["ticker"]
                price = current_prices.get(ticker) or trade.get("price", 0)
                if price <= 0:
                    log.warning(f"[PRICE INVALID] {ticker} → BUY skipped")
                    continue
                spendable_cash = max(0.0, running_cash - min_cash_abs)
                if spendable_cash < 1.0:
                    log.info(f"  BUY {ticker}: kein spendbares Cash – übersprungen")
                    continue
                log.info(f"🔍 DEBUG: Executing BUY {ticker} target_value={trade.get('value')}, spendable={spendable_cash}")
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
                        quantity=record.get("fill_qty") or (actual_spent / price if price > 0 else 0),
                        price=record.get("fill_price") or price,
                    )
            run_summary["buys_executed"] = buys_ok
        else:
            log.info("\n  ── Phase 3: BUY-Phase übersprungen")

        run_summary["trades_executed"] = sells_ok + buys_ok
        run_summary["execution_mode"] = "LIVE" if execution_enabled else "SIMULATED"
        run_summary["market_closed"] = analysis_only

        if executed_records:
            cooldown_manager.register_executed_trades(executed_records)

        # SCHRITT 8: Journal & Snapshot
        log.info("\nSCHRITT 8/8: Journal & Portfolio-Snapshot...")
        self._check_portfolio_consistency(market_data)
        portfolio_summary_after = self._build_portfolio_summary()
        trade_logger.log_portfolio_snapshot(portfolio_summary_after)

        real_executed = [r for r in executed_records if r.get("status") == "EXECUTED" or (r.get("fill_qty") or 0) > 0]

        if scores_obj:
            top_scores = sorted(scores_obj.values(), key=lambda x: x.total_score, reverse=True)[:5]
            log.info("  Top-5 Quant Scores: " + " | ".join(f"{s.ticker}={s.total_score:.0f}({s.signal})" for s in top_scores))
            save_json_file(f"{LOG_DIR}/scores_{start_time.strftime('%Y%m%d_%H%M%S')}.json", {t: sb.to_dict() for t, sb in scores_obj.items()})

        journal.log_run(
            market_outlook=ai_result.get("market_outlook", ""),
            risk_assessment=ai_result.get("risk_assessment", ""),
            ai_signals=raw_decisions,
            final_decisions=validated,
            simulated_trades=planned_trades,
            executed_trades=real_executed,
            portfolio_before=portfolio_summary_before,
            portfolio_after=portfolio_summary_after,
            portfolio_projection=None,
            risk_warnings=risk_warnings,
            mode=self.mode,
            feedback_learnings=ai_result.get("feedback_learnings", ""),
            regime_state=regime_state,
            market_data=market_data,
            execution_mode=run_summary.get("execution_mode", "SIMULATED"),
            market_closed=run_summary.get("market_closed", False),
        )

        duration = (datetime.now() - start_time).seconds
        run_summary.update({
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": duration,
            "final_portfolio_value": portfolio_summary_after.get("total_value", 0),
            "final_pnl_pct": portfolio_summary_after.get("pnl_pct", 0),
        })
        save_json_file(f"{LOG_DIR}/run_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json", run_summary)

        log.info(f"\n{'='*70}")
        log.info(f"  RUN ABGESCHLOSSEN | Dauer: {duration}s")
        log.info(f"  Portfolio:  {format_currency(portfolio_summary_after.get('total_value', 0))}")
        log.info(f"  P&L:        {portfolio_summary_after.get('pnl_pct', 0):+.2f}%")
        log.info(f"  SELLs ausgeführt: {sells_ok} | BUYs ausgeführt: {buys_ok}")
        log.info(f"{'='*70}\n")

        return run_summary

    # ─── Final Execution Guard ────────────────────────────────────────────────
    def _final_execution_guard(self, sell_trades: List[Dict], buy_trades: List[Dict], portfolio_summary: Dict) -> Tuple[List[Dict], List[Dict], List[str]]:
        warnings = []
        _profile = RISK_SETTINGS[ACTIVE_RISK_PROFILE]
        max_position_pct = _profile["max_position_pct"]
        min_cash_pct = _profile["min_cash_pct"]
        total_value = portfolio_summary.get("total_value", self.portfolio.get_total_value())
        min_cash_abs = total_value * min_cash_pct

        available_cash = portfolio_summary.get("cash", self.portfolio.cash)
        total_buy_value = sum(t.get("value", 0) for t in buy_trades)
        projected_cash = available_cash - total_buy_value

        if projected_cash < min_cash_abs:
            shortfall = min_cash_abs - projected_cash
            warnings.append(f"Cash guard: projected cash ${projected_cash:,.0f} < reserve ${min_cash_abs:,.0f} → reducing BUY sizes")
            if total_buy_value > 0:
                scale = max(0.0, (available_cash - min_cash_abs) / total_buy_value)
                for trade in buy_trades:
                    orig = trade.get("value", 0)
                    trade["value"] = round(orig * scale, 2)

        for trade in buy_trades:
            ticker = trade.get("ticker", "")
            current_val = self.portfolio.positions.get(ticker, {}).get("market_value", 0.0)
            buy_val = trade.get("value", 0.0)
            projected_val = current_val + buy_val
            cap_val = total_value * max_position_pct
            if projected_val > cap_val:
                clamped_buy = max(0.0, cap_val - current_val)
                warnings.append(f"Position cap guard: {ticker} projected ${projected_val:,.0f} > cap ${cap_val:,.0f} → clamped to ${clamped_buy:,.0f}")
                trade["value"] = round(clamped_buy, 2)

        original_buy_count = len(buy_trades)
        buy_trades = [t for t in buy_trades if t.get("value", 0) >= 1.0]
        if len(buy_trades) != original_buy_count:
            warnings.append(f"Guard stripped {original_buy_count - len(buy_trades)} zero-value BUY(s)")

        return sell_trades, buy_trades, warnings

    # ─── Portfolio-Konsistenz ────────────────────────────────────────────────
    def _check_portfolio_consistency(self, market_data: Dict):
        if self.mode not in ("PAPER", "REAL", "LIVE") or not self.executor.api:
            return
        try:
            account = self.executor.api.get_account()
            broker_cash = float(account.cash)
            local_cash = self.portfolio.cash
            cash_diff = abs(broker_cash - local_cash)
            if cash_diff > 1.0:
                log.warning(f"[KONSISTENZ] Cash-Abweichung: lokal {format_currency(local_cash)} vs Broker {format_currency(broker_cash)}")
                self.portfolio._load_from_alpaca()
                log.info("[KONSISTENZ] Portfolio mit Broker re-synchronisiert.")
        except Exception as e:
            log.warning(f"[KONSISTENZ] Check fehlgeschlagen: {e}")

    def _build_portfolio_summary(self) -> Dict:
        summary = self.portfolio.get_summary()
        summary["positions"] = self.portfolio.positions
        return summary

    def _execute_sell_trade(self, trade: Dict, current_prices: Dict[str, float]) -> Tuple[bool, Dict]:
        ticker = trade["ticker"]
        price = current_prices.get(ticker, trade.get("price", 0))
        is_zombie = trade.get("zombie_cleanup", False)
        if (not price or price <= 0) and is_zombie:
            log.info(f"[ZOMBIE] {ticker}: kein Preis → force_close_position()")
            return self.executor._force_close_position(ticker, 0.0, trade.get("ai_reason", "Zombie liquidation"))
        if (not price or price <= 0) and trade.get("price_estimated") and trade.get("price", 0) > 0:
            price = trade["price"]
            log.info(f"[SELL] {ticker}: kein Marktpreis → Fallback Ø-Kaufpreis ${price:.2f}")
        position_qty = self.portfolio.positions.get(ticker, {}).get("quantity", 0)
        return self.executor.execute_sell(
            ticker=ticker,
            position_qty=position_qty,
            current_price=price,
            target_value=trade.get("value", 0),
            reason=trade.get("ai_reason", "Portfolio Rebalancing"),
            full_liquidation=is_zombie,
        )

    def show_status(self):
        market_data = self.data_collector.collect_all()
        self.portfolio.update_prices(market_data)
        self.portfolio.print_summary()
        trades = trade_logger.get_trade_history()
        log.info(f"Trades insgesamt: {len(trades)}")
        log.info(f"Markt-Status: {market_status()}")


# ─── Standalone-Funktionen ────────────────────────────────────────────────────
def run_backtest(use_ai: bool = False):
    tickers = ETF_WATCHLIST + STOCK_WATCHLIST[:5]
    if use_ai:
        log.info("Starte KI-Backtest...")
        from config import OPENAI_API_KEY
        has_key = bool(OPENAI_API_KEY)
        bt = AIBacktester(tickers=tickers, start_date=BACKTEST_START_DATE, end_date=BACKTEST_END_DATE, frequency="weekly", use_ai=has_key)
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
    parser.add_argument("--mode", choices=["dry", "paper", "real", "live"], default=TRADING_MODE.lower())
    parser.add_argument("--backtest", action="store_true", help="Momentum-Backtest (Baseline)")
    parser.add_argument("--ai", action="store_true", help="KI-Backtest statt Momentum-Baseline (mit --backtest)")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--status", action="store_true")
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
