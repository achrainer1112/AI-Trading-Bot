"""
AI Trading Bot - Hauptprogramm (Top-N Score-Modus, Zweiphasen)
===========================================================================
- Verwendet bestehenden CPO (cpo_engine) und PortfolioRebalancer
- Führt SELLs zuerst aus → Broker‑Sync → dann BUYs mit aktualisiertem Cash
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
    assert_portfolio_consistency, assert_no_duplicate_tickers,
)
from trading_journal import journal
from market_regime_enhanced import EnhancedMarketRegimeDetector
from execution_safety import LiveRunLogger, DrawdownMonitor
from config import ALLOW_LIVE_TRADING, KILL_SWITCH_DRAWDOWN_PCT
from capital_rotator import CapitalRotator
from portfolio_rebalancer import PortfolioRebalancer, RebalancingDecision
from decision_weighter import DecisionWeighter, WeightedSignal
from order_aggregator import aggregate_trades
from score_engine import ScoreEngine, rank_candidates

import os
os.makedirs(LOG_DIR, exist_ok=True)


class TradingBot:
    def __init__(self, mode: str = None):
        self.mode = (mode or TRADING_MODE).upper()
        log.info("=" * 70)
        log.info("  AI TRADING BOT – INITIALISIERUNG (Top-N Score-Modus, Zweiphasen)")
        log.info(f"  Modus: {self.mode} | Risikoprofil: {ACTIVE_RISK_PROFILE.value.upper()}")
        log.info(f"  Watchlist: {len(FULL_WATCHLIST)} Assets")
        log.info(f"  Max Positionen: 8 | Min Score: 50")
        log.info(f"  Max Position Size: 20% | Min Cash: 10%")
        log.info(f"  Markt-Status: {market_status()}")
        log.info("=" * 70)

        self.data_collector = MarketDataCollector(FULL_WATCHLIST)
        self.news_collector = NewsCollector()
        self.ai_analyzer = AIAnalyzer()

        self.executor = TradeExecutor(mode=self.mode)
        alpaca_api = self.executor.api if self.mode in ("PAPER", "REAL", "LIVE") else None
        self.portfolio = PortfolioManager(mode=self.mode, alpaca_api=alpaca_api)
        self.risk_manager = RiskManager(ACTIVE_RISK_PROFILE)

        self.rebalancer = PortfolioRebalancer(
            risk_profile=ACTIVE_RISK_PROFILE,
            cvar_manager=self.risk_manager.cvar_manager if hasattr(self.risk_manager, 'cvar_manager') else None
        )
        self.decision_weighter = DecisionWeighter()

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

        # SCHRITT 0: Markt-Regime
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

        # SCHRITT 2: News
        log.info("\nSCHRITT 2/8: News & Sentiment sammeln...")
        news_articles = self.news_collector.collect_all()
        news_text = self.news_collector.format_for_ai(news_articles)

        # SCHRITT 3: Portfolio-Snapshot
        log.info("\nSCHRITT 3/8: Portfolio aktualisieren...")
        self.portfolio.update_prices(market_data)
        portfolio_summary_before = self._build_portfolio_summary()
        self.portfolio.print_summary()

        zombie_tickers = find_zombie_positions(
            self.portfolio.positions,
            threshold=ZOMBIE_POSITION_THRESHOLD,
            min_age_days=ZOMBIE_MIN_AGE_DAYS
        )
        zombie_orders = build_zombie_sell_orders(zombie_tickers, self.portfolio.positions)

        # SCHRITT 4: KI-Analyse & Rebalancing
        log.info("\nSCHRITT 4/8: KI-Analyse & Entscheidungsfindung...")
        trailing_stops = self.portfolio.get_trailing_stop_triggers(market_data, trail_pct=0.12)
        if trailing_stops:
            log.info(f"  {len(trailing_stops)} Trailing Stop Trigger(s) erkannt")

        effective_watchlist = list(set(FULL_WATCHLIST) | set(self.portfolio.positions.keys()))

        # ScoreEngine
        score_engine = ScoreEngine(
            positions=self.portfolio.positions,
            total_value=portfolio_summary_before.get("total_value", 100_000)
        )
        scores_obj = score_engine.score_all(
            market_data=market_data,
            regime_state=regime_state,
            ai_scores={},
            sentiment_scores={},
        )
        scores = {ticker: sb.total_score for ticker, sb in scores_obj.items()}
        momentums = {ticker: sb.momentum_20d or 0.0 for ticker, sb in scores_obj.items()}
        volatilities = {ticker: sb.volatility_annual or 20.0 for ticker, sb in scores_obj.items()}
        current_weights = self.portfolio.get_allocations()
        total_value = portfolio_summary_before.get("total_value", self.portfolio.get_total_value())
        cash = portfolio_summary_before.get("cash", 0.0)

        # ========== REBALANCING: ZIELPORTFOLIO BERECHNEN ==========
        log.info("\nSCHRITT 5/8: Zielportfolio berechnen (CPO)...")
        rebalancing_decisions, target_weights, cash_target, rebalance_rationale = self.rebalancer.optimize_portfolio(
            scores=scores,
            market_data=market_data,
            current_weights=current_weights,
            cash=cash,
            total_value=total_value,
            regime_state=regime_state,
        )

        # ========== TRADE-ENTSCHEIDUNGEN AUS REBALANCING EXTRAHIEREN ==========
        # Die `rebalancing_decisions` sind bereits `RebalancingDecision` Objekte mit Aktion und Zielgewicht.
        # Wir wandeln sie in das interne Dict-Format um.
        rebalance_trades = []
        for rd in rebalancing_decisions:
            rebalance_trades.append({
                "ticker": rd.ticker,
                "action": rd.action,
                "target_allocation": rd.target_weight,
                "confidence": rd.confidence,
                "reason": rd.reason,
                "risk_approved": True,
            })

        # Weitere Entscheidungen: KI, DecisionWeighter (optional) – hier vereinfacht,
        # da wir nur die rebalancing_trades verwenden wollen. Aber für Kompatibilität:
        # (Der bisherige Code mit AI und DecisionWeighter wird auskommentiert, da wir nur CPO nutzen)
        ai_result = {"market_outlook": "CPO-basierte Optimierung", "risk_assessment": "Score-basiertes Rebalancing"}
        raw_ai_decisions = []
        # merged_decisions = []  # nicht benötigt

        # Wir verwenden ausschließlich die Trades aus dem Rebalancer
        decisions = rebalance_trades

        # Stop-Loss & Zombie hinzufügen
        stop_loss_orders = self.risk_manager.check_stop_loss(self.portfolio.positions, market_data)
        decisions = stop_loss_orders + trailing_stops + zombie_orders + decisions

        # Cooldown-Filter
        decisions, cooldown_warnings = cooldown_manager.filter_decisions(decisions)
        if cooldown_warnings:
            run_summary["cooldown_warnings"] = cooldown_warnings

        # ========== ZWEIPHASEN-AUSFÜHRUNG ==========
        execution_enabled = (
            not analysis_only
            and self.mode in ("PAPER", "REAL", "LIVE")
            and self.executor.api is not None
        )

        if not execution_enabled:
            log.info("SCHRITT 7/8: Execution disabled – nur Simulation")
            validated, risk_warnings = self.risk_manager.validate_decisions(
                decisions=decisions,
                portfolio_summary=portfolio_summary_before,
                market_data=market_data,
                historical_returns=historical_returns,
            )
            run_summary["risk_warnings"] = risk_warnings
            sells_executed = 0
            buys_executed = 0
        else:
            # Phase 1: SELLs sammeln
            sell_candidates = [d for d in decisions if d.get("action") == "SELL"]
            buy_candidates = [d for d in decisions if d.get("action") == "BUY"]

            log.info(f"\n  ── Phase 1: {len(sell_candidates)} SELL(s) ──")
            if sell_candidates:
                validated_sells, sell_warnings = self.risk_manager.validate_decisions(
                    decisions=sell_candidates,
                    portfolio_summary=portfolio_summary_before,
                    market_data=market_data,
                    historical_returns=historical_returns,
                )
                run_summary["risk_warnings"].extend(sell_warnings)

                sells_executed = 0
                for d in validated_sells:
                    if d.get("action") != "SELL" or not d.get("risk_approved"):
                        continue
                    price = current_prices.get(d["ticker"], 0)
                    if price <= 0:
                        continue
                    ok, record = self._execute_sell_trade(d, current_prices)
                    if ok and record.get("status") == "EXECUTED":
                        sells_executed += 1
                        self.portfolio.apply_trade(
                            ticker=d["ticker"],
                            action="SELL",
                            quantity=record.get("fill_qty") or d.get("quantity", 0),
                            price=record.get("fill_price") or price,
                        )
                run_summary["sells_executed"] = sells_executed

                # Broker-Sync nach SELLs
                if sells_executed > 0:
                    log.info("\n  ── Phase 2: Broker-Sync nach SELLs ──")
                    time.sleep(2)
                    self.portfolio._load_from_alpaca()
                    log.info(f"  Sync: Cash nach SELLs = {format_currency(self.portfolio.cash)}")
                    # Aktualisiere Portfolio für BUYs
                    portfolio_summary_before = self._build_portfolio_summary()
                    total_value = portfolio_summary_before.get("total_value", self.portfolio.get_total_value())
                    current_weights = self.portfolio.get_allocations()
            else:
                log.info("  Keine SELLs – überspringe")
                sells_executed = 0

            # Phase 3: BUYs mit aktualisiertem Cash
            if buy_candidates:
                log.info(f"\n  ── Phase 3: {len(buy_candidates)} BUY(s) mit aktualisiertem Cash ──")
                # Aktualisiere Portfolio für BUYs
                portfolio_summary_for_buys = self._build_portfolio_summary()
                validated_buys, buy_warnings = self.risk_manager.validate_decisions(
                    decisions=buy_candidates,
                    portfolio_summary=portfolio_summary_for_buys,
                    market_data=market_data,
                    historical_returns=historical_returns,
                )
                run_summary["risk_warnings"].extend(buy_warnings)

                total_value = portfolio_summary_for_buys.get("total_value", self.portfolio.get_total_value())
                min_cash_pct = self.risk_manager.settings.get("min_cash_pct", 0.10)
                min_cash_abs = max(0.0, total_value * min_cash_pct)
                running_cash = self.portfolio.cash
                current_prices = {t: d.get("current_price", 0) for t, d in market_data.items()}

                buys_executed = 0
                for d in validated_buys:
                    if d.get("action") != "BUY" or not d.get("risk_approved"):
                        continue
                    ticker = d["ticker"]
                    target_alloc = d.get("target_allocation", 0)
                    current_val = self.portfolio.positions.get(ticker, {}).get("market_value", 0)
                    buy_cost = max(0, total_value * target_alloc - current_val)
                    price = current_prices.get(ticker, 0)
                    if price <= 0:
                        continue
                    spendable = max(0.0, running_cash - min_cash_abs)
                    if spendable < buy_cost:
                        buy_cost = spendable
                    if buy_cost < 1.0:
                        continue
                    ok, record = self.executor.execute_buy(
                        ticker=ticker,
                        target_value=buy_cost,
                        current_price=price,
                        available_cash=running_cash,
                        min_cash_reserve=min_cash_abs,
                        reason=d.get("reason", "CPO Rebalancing"),
                    )
                    if ok and record.get("status") == "EXECUTED":
                        buys_executed += 1
                        actual_spent = record.get("fill_value") or buy_cost
                        running_cash -= actual_spent
                        self.portfolio.apply_trade(
                            ticker=ticker,
                            action="BUY",
                            quantity=record.get("fill_qty") or (actual_spent / price if price > 0 else 0),
                            price=record.get("fill_price") or price,
                        )
                run_summary["buys_executed"] = buys_executed
            else:
                log.info("  Keine BUYs")
                buys_executed = 0

        # Abschließende Portfolio-Konsistenz
        self._check_portfolio_consistency(market_data)
        portfolio_summary_after = self._build_portfolio_summary()
        trade_logger.log_portfolio_snapshot(portfolio_summary_after)

        # Portfolio-Report
        self._print_portfolio_report(current_weights, target_weights, scores)

        # Journal
        journal.log_run(
            market_outlook=ai_result.get("market_outlook", "CPO-basierte Optimierung"),
            risk_assessment=ai_result.get("risk_assessment", ""),
            ai_signals=raw_ai_decisions,
            final_decisions=decisions,
            simulated_trades=[],
            executed_trades=[],
            portfolio_before=portfolio_summary_before,
            portfolio_after=portfolio_summary_after,
            portfolio_projection=None,
            risk_warnings=run_summary.get("risk_warnings", []),
            mode=self.mode,
            feedback_learnings="",
            regime_state=regime_state,
            market_data=market_data,
            execution_mode="LIVE" if execution_enabled else "SIMULATED",
            market_closed=analysis_only,
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
        log.info(f"  SELLs ausgeführt: {run_summary.get('sells_executed', 0)} | BUYs ausgeführt: {run_summary.get('buys_executed', 0)}")
        log.info(f"{'='*70}\n")

        return run_summary

    # Hilfsmethoden (unverändert)
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
            return self.executor._force_close_position(ticker, 0.0, trade.get("reason", "Zombie liquidation"))
        if (not price or price <= 0) and trade.get("price_estimated") and trade.get("price", 0) > 0:
            price = trade["price"]
        position_qty = self.portfolio.positions.get(ticker, {}).get("quantity", 0)
        return self.executor.execute_sell(
            ticker=ticker,
            position_qty=position_qty,
            current_price=price,
            target_value=0.0,
            reason=trade.get("reason", "CPO Rebalancing"),
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
        log.info("\nZIELPORTFOLIO (CPO):")
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
    parser = argparse.ArgumentParser(description="AI Trading Bot (Top-N Score, Zweiphasen)")
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
