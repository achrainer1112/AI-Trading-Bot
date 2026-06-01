"""
AI Trading Bot - Hauptprogramm (Top-N Score-basierte Portfolio-Optimierung)
===========================================================================
- Berechnet Scores für alle Assets (quantitativ, ohne KI)
- Wählt die besten N Assets (z.B. 5) mit Score >= MIN_SCORE aus
- Zielgewichte proportional zu Score (höherer Score = größere Position)
- Trades werden ausschließlich aus Ziel- vs. Ist-Gewichten generiert
- Keine KI-Entscheidungen mehr, keine Swap-Schwellen, keine Capital Rotation
"""

import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

from logger import log, trade_logger
from config import (
    TRADING_MODE, FULL_WATCHLIST, ETF_WATCHLIST, STOCK_WATCHLIST,
    ACTIVE_RISK_PROFILE, RISK_SETTINGS, SCHEDULE_INTERVAL, SCHEDULE_TIME,
    BACKTEST_START_DATE, BACKTEST_END_DATE, LOG_DIR, CORRELATION_GROUPS,
    CVAR_LOOKBACK_DAYS, ZOMBIE_POSITION_THRESHOLD, ZOMBIE_MIN_AGE_DAYS,
    MIN_SCORE_FOR_BUY, MAX_POSITION_COUNT, MAX_POSITION_PCT, MIN_CASH_PCT,
)
from data_collector import MarketDataCollector
from news_collector import NewsCollector
from portfolio_manager import PortfolioManager
from risk_manager import RiskManager
from trade_executor import TradeExecutor
from ai_backtester import AIBacktester
from utils import (
    is_market_open, market_status, format_currency,
    find_zombie_positions, build_zombie_sell_orders, save_json_file,
    cooldown_manager,
    assert_portfolio_consistency, assert_no_duplicate_tickers,
)
from trading_journal import journal
from market_regime_enhanced import EnhancedMarketRegimeDetector
from execution_safety import LiveRunLogger, DrawdownMonitor
from config import ALLOW_LIVE_TRADING, KILL_SWITCH_DRAWDOWN_PCT
from order_aggregator import aggregate_trades
from score_engine import ScoreEngine

import os
os.makedirs(LOG_DIR, exist_ok=True)


class TradingBot:
    def __init__(self, mode: str = None):
        self.mode = (mode or TRADING_MODE).upper()
        log.info("=" * 70)
        log.info("  AI TRADING BOT – INITIALISIERUNG (Top-N Score-Modus)")
        log.info(f"  Modus: {self.mode} | Risikoprofil: {ACTIVE_RISK_PROFILE.value.upper()}")
        log.info(f"  Watchlist: {len(FULL_WATCHLIST)} Assets")
        log.info(f"  Max Positionen: {MAX_POSITION_COUNT} | Min Score: {MIN_SCORE_FOR_BUY}")
        log.info(f"  Max Position Size: {MAX_POSITION_PCT:.0%} | Min Cash: {MIN_CASH_PCT:.0%}")
        log.info(f"  Markt-Status: {market_status()}")
        log.info("=" * 70)

        self.data_collector = MarketDataCollector(FULL_WATCHLIST)
        self.news_collector = NewsCollector()   # wird nur für Analysen genutzt, nicht für Entscheidungen
        self.executor = TradeExecutor(mode=self.mode)
        alpaca_api = self.executor.api if self.mode in ("PAPER", "REAL", "LIVE") else None
        self.portfolio = PortfolioManager(mode=self.mode, alpaca_api=alpaca_api)
        self.risk_manager = RiskManager(ACTIVE_RISK_PROFILE)

        self._live_logger = LiveRunLogger()
        _guard = self.executor.get_guard()
        self._drawdown_monitor = DrawdownMonitor(guard=_guard, limit_pct=KILL_SWITCH_DRAWDOWN_PCT) if _guard else None

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

        # SCHRITT 0: Markt-Regime (optional für Risikomanagement)
        log.info("SCHRITT 0/8: Markt-Regime-Detektion...")
        try:
            regime_detector = EnhancedMarketRegimeDetector(watchlist=FULL_WATCHLIST)
            regime_state = regime_detector.detect()
            self.risk_manager.apply_regime(regime_state)
            run_summary["regime"] = regime_state.to_dict()
            save_json_file(f"{LOG_DIR}/regime_{start_time.strftime('%Y%m%d_%H%M%S')}.json", regime_state.to_dict())
        except Exception as e:
            log.warning(f"Regime-Detektion fehlgeschlagen: {e}")
            regime_state = None

        # SCHRITT 1: Marktdaten
        log.info("SCHRITT 1/8: Marktdaten sammeln...")
        market_data = self.data_collector.collect_all()
        if not market_data:
            log.error("Keine Marktdaten – Run abgebrochen.")
            run_summary["errors"].append("no_market_data")
            return run_summary

        # Historische Renditen für CVaR (optional)
        historical_returns = None
        if hasattr(self.data_collector, 'get_historical_returns'):
            try:
                all_tickers = list(set(self.portfolio.positions.keys()) | set(FULL_WATCHLIST))
                historical_returns = self.data_collector.get_historical_returns(all_tickers, days=CVAR_LOOKBACK_DAYS)
            except Exception as e:
                log.warning(f"Historische Renditen nicht verfügbar: {e}")

        # SCHRITT 2: News (nur für Journal, nicht für Entscheidungen)
        log.info("\nSCHRITT 2/8: News & Sentiment sammeln...")
        news_articles = self.news_collector.collect_all()
        news_text = self.news_collector.format_for_ai(news_articles)

        # SCHRITT 3: Portfolio-Snapshot
        log.info("\nSCHRITT 3/8: Portfolio aktualisieren...")
        self.portfolio.update_prices(market_data)
        portfolio_summary_before = self._build_portfolio_summary()
        self.portfolio.print_summary()

        # Zombies (Altersprüfung)
        zombie_tickers = find_zombie_positions(
            self.portfolio.positions,
            threshold=ZOMBIE_POSITION_THRESHOLD,
            min_age_days=ZOMBIE_MIN_AGE_DAYS
        )
        zombie_orders = build_zombie_sell_orders(zombie_tickers, self.portfolio.positions)

        # SCHRITT 4: Scores berechnen (Quantitativ, ohne KI)
        log.info("\nSCHRITT 4/8: Score-Berechnung...")
        score_engine = ScoreEngine(
            positions=self.portfolio.positions,
            total_value=portfolio_summary_before.get("total_value", 100_000)
        )
        # Keine KI- oder Sentiment-Scores, nur quantitativ
        scores_obj = score_engine.score_all(
            market_data=market_data,
            regime_state=regime_state,
            ai_scores={},
            sentiment_scores={},
        )
        scores = {ticker: sb.total_score for ticker, sb in scores_obj.items()}
        current_weights = self.portfolio.get_allocations()
        total_value = portfolio_summary_before.get("total_value", self.portfolio.get_total_value())

        # SCHRITT 5: Top-N Portfolio-Optimierung (CPO)
        log.info("\nSCHRITT 5/8: Portfolio-Optimierung (Top-N Score-basiert)...")
        # Nur Assets mit Score >= MIN_SCORE_FOR_BUY in Betracht ziehen
        qualified_scores = {t: s for t, s in scores.items() if s >= MIN_SCORE_FOR_BUY}
        # Sortieren und Top MAX_POSITION_COUNT auswählen
        sorted_assets = sorted(qualified_scores.items(), key=lambda x: x[1], reverse=True)
        top_assets = sorted_assets[:MAX_POSITION_COUNT]

        # Zielgewichte proportional zu Score
        investable = 1.0 - MIN_CASH_PCT
        if top_assets:
            # Basisgewicht = Score (oder Score - MIN_SCORE + 1)
            base = {t: max(1.0, s - MIN_SCORE_FOR_BUY + 1) for t, s in top_assets}
            total_base = sum(base.values())
            raw_targets = {t: (w / total_base) * investable for t, w in base.items()}
            # Caps
            target_weights = {}
            for t, w in raw_targets.items():
                target_weights[t] = min(w, MAX_POSITION_PCT)
            # Nach Caps neu normalisieren
            total_capped = sum(target_weights.values())
            if total_capped > investable:
                scale = investable / total_capped
                target_weights = {t: w * scale for t, w in target_weights.items()}
            cash_target = 1.0 - sum(target_weights.values())
        else:
            target_weights = {}
            cash_target = 1.0

        log.info(f"Zielportfolio: {len(target_weights)} Assets, Cash-Ziel {cash_target:.1%}")
        for t, w in target_weights.items():
            log.info(f"  {t}: {w:.1%} (Score {scores[t]:.0f})")

        # SCHRITT 6: Trades aus Differenz generieren
        trades = []
        for ticker, target in target_weights.items():
            current = current_weights.get(ticker, 0.0)
            delta = target - current
            if abs(delta) < 0.005:   # 0.5% Mindeständerung
                continue
            action = "BUY" if delta > 0 else "SELL"
            trades.append({
                "ticker": ticker,
                "action": action,
                "target_allocation": target,
                "confidence": 0.9,
                "reason": f"Top-N Score: Ziel {target:.1%} vs aktuell {current:.1%}",
                "risk_approved": True,
            })
        # Verkäufe für Assets, die nicht mehr in Top-N sind
        for ticker, current in current_weights.items():
            if ticker not in target_weights and current > 0:
                trades.append({
                    "ticker": ticker,
                    "action": "SELL",
                    "target_allocation": 0.0,
                    "confidence": 0.9,
                    "reason": f"Asset nicht mehr in Top-{MAX_POSITION_COUNT} (Score {scores.get(ticker, 0):.0f} < {MIN_SCORE_FOR_BUY})",
                    "risk_approved": True,
                })

        # Zombie-Orders hinzufügen (überschreiben keine anderen Entscheidungen)
        trades.extend(zombie_orders)

        # Cooldown-Filter
        trades, cooldown_warnings = cooldown_manager.filter_decisions(trades)
        if cooldown_warnings:
            run_summary["cooldown_warnings"] = cooldown_warnings

        # SCHRITT 7: Risikoprüfung & Validierung
        log.info("\nSCHRITT 7/8: Risikoprüfung & Validierung...")
        validated, risk_warnings = self.risk_manager.validate_decisions(
            decisions=trades,
            portfolio_summary=portfolio_summary_before,
            market_data=market_data,
            historical_returns=historical_returns,
        )
        run_summary["risk_warnings"] = risk_warnings

        # Rebalancing-Trades berechnen (aus validierten Entscheidungen)
        decisions_map = {d["ticker"]: d for d in validated}
        current_prices = {t: d.get("current_price", 0) for t, d in market_data.items()}
        _profile_settings = RISK_SETTINGS[ACTIVE_RISK_PROFILE]
        min_trade_value = _profile_settings.get("min_trade_value", 100.0)
        if total_value < 10000:
            dynamic_min = max(10.0, total_value * 0.05)
            if dynamic_min < min_trade_value:
                log.info(f"Small account: lowering min_trade_value from ${min_trade_value:.2f} to ${dynamic_min:.2f}")
                min_trade_value = dynamic_min

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

        # Trade Aggregation
        agg_input = [{
            "ticker": t["ticker"],
            "action": t["action"],
            "target_allocation": t["target_alloc"],
            "confidence": t.get("ai_confidence", 0.9),
        } for t in planned_trades]
        aggregated = aggregate_trades(agg_input)
        aggregated_trades = []
        for agg in aggregated:
            price = current_prices.get(agg["ticker"])
            if not price:
                continue
            aggregated_trades.append({
                "ticker": agg["ticker"],
                "action": agg["action"],
                "target_alloc": agg["target_allocation"],
                "value": agg["target_allocation"] * total_value,
                "quantity": agg["target_allocation"] * total_value / price,
                "price": price,
                "current_alloc": current_weights.get(agg["ticker"], 0.0),
                "ai_reason": agg.get("reason", "Top-N Score Rebalancing"),
                "ai_confidence": agg["confidence"],
            })
        sell_trades = [t for t in aggregated_trades if t["action"] == "SELL"]
        buy_trades = [t for t in aggregated_trades if t["action"] == "BUY"]

        # Execution Guard
        execution_enabled = (
            not analysis_only
            and self.mode in ("PAPER", "REAL", "LIVE")
            and self.executor.api is not None
        )
        if execution_enabled:
            _guard = self.executor.get_guard()
            if _guard:
                # Erzeuge Dummy-ai_result für Guard-Validierung
                ai_result = {"decisions": trades}
                md_ok = _guard.validate_market_data(market_data)
                ai_ok = _guard.validate_ai_response(ai_result)
                if not md_ok or not ai_ok:
                    log.critical("Guard-Validierung fehlgeschlagen -> Execution deaktiviert.")
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
            sell_trades, buy_trades, guard_warnings = self._final_execution_guard(sell_trades, buy_trades, portfolio_summary_before)
            if guard_warnings:
                run_summary["execution_guard_warnings"] = guard_warnings

        if not execution_enabled:
            log.info("\nSCHRITT 8/8: Execution disabled – nur Simulation")
        else:
            log.info("\nSCHRITT 8/8: Trades ausführen (SELL → Sync → BUY)...")

        executed_records = []
        sells_ok = 0
        buys_ok = 0

        # SELLs
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

        # Broker-Sync
        if execution_enabled and sell_trades and self.mode in ("PAPER", "REAL", "LIVE") and self.executor.api:
            log.info("\n  ── Phase 2: Broker-Sync nach SELLs ──")
            time.sleep(2)
            self.portfolio._load_from_alpaca()
            log.info(f"  Sync: Cash nach SELLs = {format_currency(self.portfolio.cash)}")
        else:
            log.info("\n  ── Phase 2: Broker-Sync übersprungen")

        # BUYs
        if execution_enabled and buy_trades:
            log.info(f"\n  ── Phase 3: {len(buy_trades)} BUY(s) geplant ──")
            total_value = self.portfolio.get_total_value()
            min_cash_pct = self.risk_manager.settings.get("min_cash_pct", MIN_CASH_PCT)
            min_cash_abs = max(0.0, total_value * min_cash_pct)
            running_cash = self.portfolio.cash
            for trade in buy_trades:
                ticker = trade["ticker"]
                price = current_prices.get(ticker) or trade.get("price", 0)
                if price <= 0:
                    continue
                spendable_cash = max(0.0, running_cash - min_cash_abs)
                if spendable_cash < 1.0:
                    continue
                ok, record = self.executor.execute_buy(
                    ticker=ticker,
                    target_value=trade["value"],
                    current_price=price,
                    available_cash=running_cash,
                    min_cash_reserve=min_cash_abs,
                    reason=trade.get("ai_reason", "Top-N Score Rebalancing"),
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

        run_summary["trades_executed"] = sells_ok + buys_ok
        run_summary["execution_mode"] = "LIVE" if execution_enabled else "SIMULATED"
        run_summary["market_closed"] = analysis_only

        if executed_records:
            cooldown_manager.register_executed_trades(executed_records)

        # JOURNAL
        log.info("\nSCHRITT 9/8: Journal & Portfolio-Snapshot...")
        self._check_portfolio_consistency(market_data)
        portfolio_summary_after = self._build_portfolio_summary()
        trade_logger.log_portfolio_snapshot(portfolio_summary_after)

        real_executed = [r for r in executed_records if r.get("status") == "EXECUTED" or (r.get("fill_qty") or 0) > 0]

        self._print_portfolio_report(current_weights, target_weights, scores)

        journal.log_run(
            market_outlook="Top-N Score-basierte Optimierung",
            risk_assessment="System hält die besten Assets basierend auf quantitativem Score",
            ai_signals=[],   # keine KI-Signale mehr
            final_decisions=validated,
            simulated_trades=planned_trades,
            executed_trades=real_executed,
            portfolio_before=portfolio_summary_before,
            portfolio_after=portfolio_summary_after,
            portfolio_projection=None,
            risk_warnings=risk_warnings,
            mode=self.mode,
            feedback_learnings="",
            regime_state=regime_state,
            market_data=market_data,
            execution_mode=run_summary.get("execution_mode", "SIMULATED"),
            market_closed=run_summary.get("market_closed", False),
            risk_manager=self.risk_manager,
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

    # Hilfsmethoden (unverändert)
    def _final_execution_guard(self, sell_trades, buy_trades, portfolio_summary):
        warnings = []
        _profile = RISK_SETTINGS[ACTIVE_RISK_PROFILE]
        max_position_pct = _profile.get("max_position_pct", MAX_POSITION_PCT)
        min_cash_pct = _profile.get("min_cash_pct", MIN_CASH_PCT)
        total_value = portfolio_summary.get("total_value", self.portfolio.get_total_value())
        min_cash_abs = total_value * min_cash_pct
        available_cash = portfolio_summary.get("cash", self.portfolio.cash)
        total_buy_value = sum(t.get("value", 0) for t in buy_trades)
        projected_cash = available_cash - total_buy_value
        if projected_cash < min_cash_abs:
            scale = max(0.0, (available_cash - min_cash_abs) / total_buy_value) if total_buy_value > 0 else 0
            for trade in buy_trades:
                trade["value"] = round(trade.get("value", 0) * scale, 2)
        for trade in buy_trades:
            ticker = trade.get("ticker", "")
            current_val = self.portfolio.positions.get(ticker, {}).get("market_value", 0.0)
            buy_val = trade.get("value", 0.0)
            projected_val = current_val + buy_val
            cap_val = total_value * max_position_pct
            if projected_val > cap_val:
                clamped = max(0.0, cap_val - current_val)
                trade["value"] = round(clamped, 2)
                warnings.append(f"Cap guard: {ticker} {current_val:,.0f}→{projected_val:,.0f} > cap {cap_val:,.0f}")
        buy_trades = [t for t in buy_trades if t.get("value", 0) >= 1.0]
        return sell_trades, buy_trades, warnings

    def _check_portfolio_consistency(self, market_data):
        if self.mode not in ("PAPER", "REAL", "LIVE") or not self.executor.api:
            return
        try:
            account = self.executor.api.get_account()
            broker_cash = float(account.cash)
            local_cash = self.portfolio.cash
            if abs(broker_cash - local_cash) > 1.0:
                log.warning(f"[KONSISTENZ] Cash-Abweichung: lokal {format_currency(local_cash)} vs Broker {format_currency(broker_cash)}")
                self.portfolio._load_from_alpaca()
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
            return self.executor._force_close_position(ticker, 0.0, trade.get("ai_reason", "Zombie liquidation"))
        if (not price or price <= 0) and trade.get("price_estimated") and trade.get("price", 0) > 0:
            price = trade["price"]
        position_qty = self.portfolio.positions.get(ticker, {}).get("quantity", 0)
        return self.executor.execute_sell(
            ticker=ticker,
            position_qty=position_qty,
            current_price=price,
            target_value=trade.get("value", 0),
            reason=trade.get("ai_reason", "Top-N Score Rebalancing"),
            full_liquidation=is_zombie,
        )

    def _print_portfolio_report(self, current_weights: Dict, target_weights: Dict, scores: Dict):
        log.info("\n" + "=" * 70)
        log.info("📊 PORTFOLIO REPORT (Top-N Score-basiert)")
        log.info("=" * 70)
        log.info("\nAKTUELLES PORTFOLIO:")
        log.info(f"{'Ticker':<8} {'Gewicht':>10} {'Score':>6}")
        for ticker, w in sorted(current_weights.items(), key=lambda x: -x[1])[:10]:
            score = scores.get(ticker, 0)
            log.info(f"{ticker:<8} {w:>9.1%} {score:>6.0f}")
        log.info("\nZIELPORTFOLIO (Top-N):")
        log.info(f"{'Ticker':<8} {'Gewicht':>10} {'Score':>6}")
        for ticker, w in sorted(target_weights.items(), key=lambda x: -x[1])[:10]:
            score = scores.get(ticker, 0)
            log.info(f"{ticker:<8} {w:>9.1%} {score:>6.0f}")
        log.info("\nDIFFERENZEN (Ziel - Aktuell > 2%):")
        for ticker, target in target_weights.items():
            current = current_weights.get(ticker, 0.0)
            diff = target - current
            if abs(diff) > 0.02:
                action = "AUFSTOCKEN" if diff > 0 else "REDUZIEREN"
                log.info(f"{ticker:<8} {action:<10} {diff:>+7.1%}")
        log.info("=" * 70)

    def show_status(self):
        market_data = self.data_collector.collect_all()
        self.portfolio.update_prices(market_data)
        self.portfolio.print_summary()
        trades = trade_logger.get_trade_history()
        log.info(f"Trades insgesamt: {len(trades)}")
        log.info(f"Markt-Status: {market_status()}")


def run_backtest(use_ai: bool = False):
    tickers = ETF_WATCHLIST + STOCK_WATCHLIST[:5]
    if use_ai:
        from config import OPENAI_API_KEY
        bt = AIBacktester(tickers=tickers, start_date=BACKTEST_START_DATE, end_date=BACKTEST_END_DATE, frequency="weekly", use_ai=bool(OPENAI_API_KEY))
        bt.run()
        bt.save_results(bt.run())


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
        days = ["monday", "tuesday", "wednesday", "thursday", "friday"]
        getattr(schedule.every(), days[min(SCHEDULE_WEEKDAY, 4)]).at(SCHEDULE_TIME).do(bot.run)
    while True:
        schedule.run_pending()
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="AI Trading Bot (Top-N Score)")
    parser.add_argument("--mode", choices=["dry", "paper", "real", "live"], default=TRADING_MODE.lower())
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--ai", action="store_true")
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
