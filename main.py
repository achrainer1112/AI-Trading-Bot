"""
AI Trading Bot - Hauptprogramm (Top-N Score-Modus, strikt sequenziell)
========================================================================
- Berechnet Zielportfolio basierend auf Score-Ranking (Top N, Mindestscore)
- Erzeugt SELLs (vollständig + Rebalancing) und BUYs aus Ziel-Ist-Differenz
- Führt SELLs zuerst aus, synchronisiert Cash, führt dann BUYs aus
- Keine Vermischung von SELL- und BUY-Validierung
- Risk Manager bleibt vollständig erhalten, wird aber sequenziell angewendet
"""

import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow is deprecated.*")

from logger import log, trade_logger
from config import (
    TRADING_MODE, FULL_WATCHLIST, ETF_WATCHLIST, STOCK_WATCHLIST,
    ACTIVE_RISK_PROFILE, RISK_SETTINGS, SCHEDULE_INTERVAL, SCHEDULE_TIME,
    BACKTEST_START_DATE, BACKTEST_END_DATE, LOG_DIR, CORRELATION_GROUPS,
    CVAR_LOOKBACK_DAYS, ZOMBIE_POSITION_THRESHOLD, ZOMBIE_MIN_AGE_DAYS,
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
)
from trading_journal import journal
from market_regime_enhanced import EnhancedMarketRegimeDetector
from execution_safety import LiveRunLogger, DrawdownMonitor
from config import ALLOW_LIVE_TRADING, KILL_SWITCH_DRAWDOWN_PCT
from score_engine import ScoreEngine

import os
os.makedirs(LOG_DIR, exist_ok=True)


class TradingBot:
    def __init__(self, mode: str = None):
        self.mode = (mode or TRADING_MODE).upper()
        log.info("=" * 70)
        log.info("  AI TRADING BOT – STRENG SEQUENZIELL (Top-N Score-Modus)")
        log.info(f"  Modus: {self.mode} | Risikoprofil: {ACTIVE_RISK_PROFILE.value.upper()}")
        log.info(f"  Watchlist: {len(FULL_WATCHLIST)} Assets")
        log.info(f"  Max Positionen: 8 | Min Score: 50")
        log.info(f"  Max Position Size: 20% | Min Cash: 10%")
        log.info(f"  Markt-Status: {market_status()}")
        log.info("=" * 70)

        self.data_collector = MarketDataCollector(FULL_WATCHLIST)
        self.news_collector = NewsCollector()
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

        # SCHRITT 2: News (nur für Journal)
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

        # SCHRITT 4: Scores berechnen
        log.info("\nSCHRITT 4/8: Score-Berechnung...")
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
        current_weights = self.portfolio.get_allocations()
        total_value = portfolio_summary_before.get("total_value", self.portfolio.get_total_value())
        current_prices = {t: d.get("current_price", 0) for t, d in market_data.items()}

        # ========== 1. ZIELPORTFOLIO BASIERT AUF SCORES ==========
        log.info("\nSCHRITT 5/8: Zielportfolio berechnen (Top-N Score-basiert)...")
        MIN_SCORE = 50
        MAX_POSITIONS = 8
        MAX_POSITION_PCT = 0.20
        MIN_CASH_PCT = 0.10

        qualified = [(t, s) for t, s in scores.items() if s >= MIN_SCORE]
        qualified.sort(key=lambda x: x[1], reverse=True)
        top_assets = qualified[:MAX_POSITIONS]

        if top_assets:
            total_score = sum(s for _, s in top_assets)
            investable = 1.0 - MIN_CASH_PCT
            raw_weights = {t: (s / total_score) * investable for t, s in top_assets}
            # Caps anwenden
            capped_weights = {}
            for t, w in raw_weights.items():
                capped_weights[t] = min(w, MAX_POSITION_PCT)
            total_capped = sum(capped_weights.values())
            if total_capped > investable:
                scale = investable / total_capped
                capped_weights = {t: w * scale for t, w in capped_weights.items()}
            target_weights = capped_weights
            cash_target = 1.0 - sum(target_weights.values())
        else:
            target_weights = {}
            cash_target = 1.0

        log.info(f"Zielportfolio: {len(target_weights)} Assets, Cash-Ziel {cash_target:.1%}")
        for t, w in target_weights.items():
            log.info(f"  {t}: {w:.1%} (Score {scores[t]:.0f})")

        # ========== 2. TRADES AUS DIFFERENZ ABLEITEN ==========
        trades = []
        # Verkäufe für Assets, die nicht im Zielportfolio sind
        for ticker, current in current_weights.items():
            if ticker == "CASH":
                continue
            target = target_weights.get(ticker, 0.0)
            if current > target:
                trades.append({
                    "ticker": ticker,
                    "action": "SELL",
                    "target_allocation": target,
                    "confidence": 0.99,
                    "reason": f"CPO: Reduziere von {current:.1%} auf {target:.1%}",
                    "risk_approved": True,
                })
        # Käufe für Assets, die im Zielportfolio sind, aber untergewichtet
        for ticker, target in target_weights.items():
            current = current_weights.get(ticker, 0.0)
            if target > current:
                trades.append({
                    "ticker": ticker,
                    "action": "BUY",
                    "target_allocation": target,
                    "confidence": 0.99,
                    "reason": f"CPO: Erhöhe von {current:.1%} auf {target:.1%}",
                    "risk_approved": True,
                })

        # Zombie-Orders hinzufügen
        trades.extend(zombie_orders)

        # Cooldown-Filter
        trades, cooldown_warnings = cooldown_manager.filter_decisions(trades)
        if cooldown_warnings:
            run_summary["cooldown_warnings"] = cooldown_warnings

        # ========== 3. AUSFÜHRUNG: STRENG SEQUENZIELL ==========
        execution_enabled = (
            not analysis_only
            and self.mode in ("PAPER", "REAL", "LIVE")
            and self.executor.api is not None
        )

        if not execution_enabled:
            log.info("\nSCHRITT 6/8: Execution disabled – nur Simulation")
            # Nur Risikoprüfung für Logging (ohne Ausführung)
            validated, risk_warnings = self.risk_manager.validate_decisions(
                decisions=trades,
                portfolio_summary=portfolio_summary_before,
                market_data=market_data,
                historical_returns=historical_returns,
            )
            run_summary["risk_warnings"] = risk_warnings
            sells_executed = 0
            buys_executed = 0
        else:
            # Phase 1: SELLs ausführen
            sell_trades = [t for t in trades if t["action"] == "SELL"]
            buy_trades = [t for t in trades if t["action"] == "BUY"]

            if sell_trades:
                log.info(f"\n  ── Phase 1: {len(sell_trades)} SELL(s) ausführen ──")
                validated_sells, sell_warnings = self.risk_manager.validate_decisions(
                    decisions=sell_trades,
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

                # Broker-Sync nach SELLs (Cash aktualisieren)
                if sells_executed > 0:
                    log.info("\n  ── Phase 2: Broker-Sync nach SELLs ──")
                    time.sleep(2)
                    self.portfolio._load_from_alpaca()
                    log.info(f"  Cash nach SELLs: {format_currency(self.portfolio.cash)}")
                    # Portfolio-Zustand aktualisieren
                    portfolio_summary_after_sells = self._build_portfolio_summary()
                    total_value = portfolio_summary_after_sells.get("total_value", self.portfolio.get_total_value())
                    current_weights = self.portfolio.get_allocations()
            else:
                log.info("\n  ── Phase 1: Keine SELLs ──")
                sells_executed = 0

            # Phase 3: BUYs mit aktualisiertem Cash ausführen
            if buy_trades:
                log.info(f"\n  ── Phase 3: {len(buy_trades)} BUY(s) mit aktualisiertem Cash ──")
                # Aktualisiere Portfolio für BUYs (nach SELLs)
                portfolio_summary_for_buys = self._build_portfolio_summary()
                validated_buys, buy_warnings = self.risk_manager.validate_decisions(
                    decisions=buy_trades,
                    portfolio_summary=portfolio_summary_for_buys,
                    market_data=market_data,
                    historical_returns=historical_returns,
                )
                run_summary["risk_warnings"].extend(buy_warnings)

                total_value = portfolio_summary_for_buys.get("total_value", self.portfolio.get_total_value())
                min_cash_abs = max(0.0, total_value * MIN_CASH_PCT)
                running_cash = self.portfolio.cash

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
                log.info("\n  ── Phase 3: Keine BUYs ──")
                buys_executed = 0

        # Abschließende Portfolio-Konsistenz
        self._check_portfolio_consistency(market_data)
        portfolio_summary_after = self._build_portfolio_summary()
        trade_logger.log_portfolio_snapshot(portfolio_summary_after)

        # Portfolio-Report
        self._print_portfolio_report(current_weights, target_weights, scores)

        # Journal (vereinfacht, ohne KI)
        journal.log_run(
            market_outlook="CPO-basierte Optimierung (streng sequenziell)",
            risk_assessment="Score-basiertes Rebalancing mit sequenzieller Ausführung",
            ai_signals=[],
            final_decisions=[],
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
    parser = argparse.ArgumentParser(description="AI Trading Bot (streng sequenziell)")
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
