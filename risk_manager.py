"""
AI Trading Bot - Risk Manager (Production)
============================================
Erweiterte Risikokontrollen mit adaptiver Confidence Engine.
"""

from typing import Dict, List, Optional, Set, Tuple
from logger import log
from config import (
    ACTIVE_RISK_PROFILE,
    RISK_SETTINGS,
    RiskProfile,
    SECTOR_CLASSIFICATION,
    FACTOR_CLASSIFICATION,
    ETF_SECTOR_WEIGHTS,
    ETF_FACTOR_WEIGHTS,
    REGIME_CONFIDENCE_THRESHOLDS,
)
from utils import format_currency, zombie_registry, ensure_decision_ids

ETF_OVERLAP_GROUPS = [
    {"SPY", "VTI", "IVV", "VOO"},
    {"QQQ", "QQQM"},
]

TOP_N_BUYS = 5


class CircuitBreaker:
    """Circuit Breaker System (unverändert)"""
    def __init__(self):
        self.triggered = False
        self.reason = None
        self.action = None

    def check_all(self, portfolio: Dict, market_data: Dict) -> bool:
        intraday_loss = market_data.get('intraday_loss_pct', 0.0)
        if intraday_loss < -0.05:
            self.triggered = True
            self.reason = f"Intraday loss: {intraday_loss:.2%} < -5%"
            self.action = 'STOP_ALL_BUYS'
            log.critical(f"🛑 CIRCUIT BREAKER 1 (Intraday Loss): {self.reason}")
            return True
        drawdown = portfolio.get('drawdown_pct', 0.0)
        if drawdown < -0.10:
            self.triggered = True
            self.reason = f"Portfolio drawdown: {drawdown:.2%} < -10%"
            self.action = 'REDUCE_EXPOSURE'
            log.critical(f"🛑 CIRCUIT BREAKER 2 (Drawdown): {self.reason}")
            return True
        vix = market_data.get('vix', 20.0)
        if vix > 35:
            self.triggered = True
            self.reason = f"VIX spike: {vix:.1f} > 35"
            self.action = 'DE_RISK_50'
            log.critical(f"🛑 CIRCUIT BREAKER 3 (VIX Spike): {self.reason}")
            return True
        if market_data.get('market_halted', False):
            self.triggered = True
            self.reason = "Market circuit breaker triggered"
            self.action = 'HALT_ALL'
            log.critical(f"🛑 CIRCUIT BREAKER 4 (Market Halt): {self.reason}")
            return True
        self.triggered = False
        self.action = None
        return False

    def get_action(self) -> str:
        return self.action or 'OK'


class FinalDecisionResolver:
    """Auflösung von Konflikten (unverändert)"""
    FORCED_EXIT_MARKERS = ("stop_loss", "zombie_cleanup", "forced_rebalancing")
    SELL_PRIORITY = {
        "stop_loss": 100,
        "zombie_cleanup": 90,
        "forced_rebalancing": 80,
        "rebalancing": 70,
        "normal": 50,
    }

    def resolve(self, decisions: List[Dict]) -> List[Dict]:
        by_ticker: Dict[str, List[Dict]] = {}
        for d in decisions:
            t = d.get("ticker", "")
            if not t:
                continue
            by_ticker.setdefault(t, []).append(d)
        resolved = []
        for ticker, candidates in by_ticker.items():
            if len(candidates) == 1:
                resolved.append(candidates[0])
                continue
            winner = self._pick_winner(ticker, candidates)
            resolved.append(winner)
        return resolved

    def _pick_winner(self, ticker: str, candidates: List[Dict]) -> Dict:
        forced_exits = [c for c in candidates if any(c.get(m, False) for m in self.FORCED_EXIT_MARKERS)]
        sells = [c for c in candidates if c.get("action") == "SELL"]
        buys = [c for c in candidates if c.get("action") == "BUY" and c.get("risk_approved", False)]
        holds = [c for c in candidates if c.get("action") == "HOLD"]
        if forced_exits:
            forced_exits.sort(key=lambda c: self._forced_exit_priority(c), reverse=True)
            winner = forced_exits[0]
            winner["action"] = "SELL"
            winner["risk_approved"] = True
            return winner
        if sells:
            sells.sort(key=lambda c: self._sell_type_priority(c), reverse=True)
            return sells[0]
        if buys:
            buys.sort(key=lambda c: c.get("confidence", 0), reverse=True)
            return buys[0]
        if holds:
            return holds[-1]
        return candidates[0]

    def _forced_exit_priority(self, d: Dict) -> int:
        if d.get("stop_loss"):
            return 100
        if d.get("zombie_cleanup"):
            return 90
        if d.get("forced_rebalancing"):
            return 80
        return 0

    def _sell_type_priority(self, d: Dict) -> int:
        sell_type = "normal"
        if d.get("stop_loss"):
            sell_type = "stop_loss"
        elif d.get("zombie_cleanup"):
            sell_type = "zombie_cleanup"
        elif d.get("forced_rebalancing"):
            sell_type = "forced_rebalancing"
        elif d.get("rebalancing"):
            sell_type = "rebalancing"
        return self.SELL_PRIORITY.get(sell_type, 50)


class RiskManager:
    def __init__(self, risk_profile: RiskProfile = None):
        self.risk_profile = risk_profile or ACTIVE_RISK_PROFILE
        self.settings = RISK_SETTINGS[self.risk_profile].copy()
        self._base_settings = RISK_SETTINGS[self.risk_profile].copy()
        self._resolver = FinalDecisionResolver()
        self.regime_state = None
        log.info(f"Risk Manager initialisiert: Profil={self.risk_profile.value.upper()}")
        log.info(
            f"  Max Position: {self.settings['max_position_pct']*100:.0f}% | "
            f"Min Cash: {self.settings['min_cash_pct']*100:.0f}% | "
            f"Stop-Loss: {self.settings['stop_loss_pct']*100:.0f}% | "
            f"Max Trades: {self.settings.get('max_trades_per_run', TOP_N_BUYS)}"
        )

    # ========== ADAPTIVE CONFIDENCE ENGINE ==========
    def get_adaptive_thresholds(
        self,
        regime_state,
        vix: Optional[float] = None,
        market_momentum: float = 0.0,
        cash_pct: float = 1.0,
        invested_pct: float = 0.0,
    ) -> Dict[str, float]:
        """
        Berechnet dynamische Confidence-Thresholds basierend auf:
        - Marktregime (Basis)
        - VIX (Stressadjustment)
        - Markt-Momentum
        - Portfolio-Exposure (Cash/Investiert)
        Returns: {"buy": float, "sell": float} (als Dezimal, 0-1)
        """
        # 1. Basis aus Regime
        if regime_state is None:
            regime = "SIDEWAYS"
        else:
            regime_val = getattr(regime_state, 'regime', None)
            if regime_val is not None:
                regime = regime_val.value.upper() if hasattr(regime_val, 'value') else str(regime_val).upper()
            else:
                regime = getattr(regime_state, 'label', 'SIDEWAYS').upper()
        base = REGIME_CONFIDENCE_THRESHOLDS.get(regime, REGIME_CONFIDENCE_THRESHOLDS["SIDEWAYS"])
        buy_threshold = base["buy_threshold"]
        sell_threshold = base["sell_threshold"]

        # 2. VIX-Anpassung
        vix_adj_buy = 0.0
        vix_adj_max_trades = 0
        if vix is not None:
            if vix > 35:
                vix_adj_buy = 0.10
                vix_adj_max_trades = -50   # Halbieren
                log.info(f"VIX {vix:.1f} > 35 → BUY +10%, Max Trades halbiert")
            elif vix > 25:
                vix_adj_buy = 0.05
                vix_adj_max_trades = -20
                log.info(f"VIX {vix:.1f} > 25 → BUY +5%, Max Trades -20%")
        buy_threshold += vix_adj_buy

        # 3. Momentum-Anpassung
        momentum_adj_buy = 0.0
        momentum_adj_sell = 0.0
        if market_momentum > 0.5:   # > 50% Momentum (sehr stark)
            momentum_adj_buy = -0.03
            log.info(f"Momentum {market_momentum:.1%} > 50% → BUY -3%")
        elif market_momentum < -0.5:
            momentum_adj_sell = -0.05
            log.info(f"Momentum {market_momentum:.1%} < -50% → SELL -5%")
        buy_threshold += momentum_adj_buy
        sell_threshold += momentum_adj_sell

        # 4. Portfolio Exposure Adjustment
        if cash_pct < 0.08:   # Cash < 8%
            buy_threshold += 0.05
            log.info(f"Cash {cash_pct:.1%} < 8% → BUY +5%")
        if invested_pct > 0.90:   # >90% investiert
            # Neue BUYs nur mit >75% Confidence
            if buy_threshold < 0.75:
                buy_threshold = 0.75
            log.info(f"Investiert {invested_pct:.1%} > 90% → BUY min 75%")

        # 5. Safety Clamping
        buy_threshold = max(0.45, min(0.85, buy_threshold))
        sell_threshold = max(0.45, min(0.85, sell_threshold))

        # Logging
        log.info(f"Adaptive Thresholds: BUY={buy_threshold:.0%}, SELL={sell_threshold:.0%} "
                 f"(Regime={regime}, VIX={vix or 'n/a'}, Momentum={market_momentum:.2f})")
        return {"buy": buy_threshold, "sell": sell_threshold}

    def emergency_cash_mode(self, portfolio_summary: Dict, market_data: Dict) -> bool:
        max_daily_loss = self.settings.get("max_daily_loss_pct", 0.05)
        daily_pnl = portfolio_summary.get("daily_pnl_pct", 0.0)
        if daily_pnl < -max_daily_loss:
            log.critical(f"EMERGENCY CASH MODE: Daily loss {daily_pnl:.2%} < -{max_daily_loss:.2%}")
            return True
        return False

    def _dynamic_max_trades(self, market_data: Dict, vix_adj: int = 0) -> int:
        vix = market_data.get('vix', 20)
        base_max = self.settings.get("max_trades_per_run", TOP_N_BUYS)
        if vix > 30:
            base_max = max(1, base_max // 2)
        elif vix < 15:
            base_max = base_max
        # Anwendung der VIX-Adjustierung (z.B. -50% bei VIX>35)
        if vix_adj < 0:
            factor = 1.0 + vix_adj / 100.0
            base_max = max(1, int(base_max * factor))
        return base_max

    def calculate_dynamic_stop_loss(self, ticker: str, avg_price: float, market_data: Dict, atr_multiplier: float = 2.5) -> float:
        import math
        ticker_data = market_data.get(ticker, {})
        annual_vol = ticker_data.get("volatility_annual", 0.20)
        daily_vol = annual_vol / math.sqrt(252)
        atr_approx = avg_price * daily_vol
        dynamic_stop = avg_price - (atr_approx * atr_multiplier)
        fixed_stop_pct = self.settings.get("stop_loss_pct", 0.08)
        fixed_stop = avg_price * (1 - fixed_stop_pct)
        return max(dynamic_stop, fixed_stop)

    def volatility_adjusted_allocation(self, base_allocation: float, asset_volatility_pct: float, target_volatility: float = 0.15) -> float:
        if asset_volatility_pct <= 0:
            asset_volatility_pct = 0.15
        scale = target_volatility / asset_volatility_pct
        scale = max(0.3, min(2.0, scale))
        adjusted = base_allocation * scale
        max_position = self.settings.get("max_position_pct", 0.25)
        return min(adjusted, max_position)

    def apply_regime(self, regime_state) -> None:
        from market_regime import apply_regime_to_risk_settings
        self.regime_state = regime_state
        self.settings = apply_regime_to_risk_settings(self._base_settings, regime_state)

    def _sector_of(self, ticker: str) -> str:
        return SECTOR_CLASSIFICATION.get(ticker, "diversified")

    def _factor_of(self, ticker: str) -> str:
        return FACTOR_CLASSIFICATION.get(ticker, self._sector_of(ticker))

    def _sector_exposure(self, positions: Dict, total_value: float) -> Dict[str, float]:
        exposure = {}
        for ticker, pos in positions.items():
            alloc = pos.get("market_value", 0) / total_value if total_value > 0 else 0
            sector = self._sector_of(ticker)
            weights = ETF_SECTOR_WEIGHTS.get(ticker)
            if weights:
                for label, weight in weights.items():
                    exposure[label] = exposure.get(label, 0.0) + alloc * weight
            else:
                exposure[sector] = exposure.get(sector, 0.0) + alloc
        return exposure

    def validate_decisions(
        self,
        decisions: List[Dict],
        portfolio_summary: Dict,
        market_data: Dict[str, Dict],
    ) -> Tuple[List[Dict], List[str]]:
        warnings = []
        validated = []
        max_pos = self.settings["max_position_pct"]
        min_cash = self.settings["min_cash_pct"]
        max_sector = self.settings.get("max_sector_exposure", 0.45)
        max_factor = self.settings.get("max_factor_exposure", 0.60)

        total_value = portfolio_summary.get("total_value", 1)
        running_cash = portfolio_summary.get("cash", 0)
        positions = portfolio_summary.get("positions", {})

        decisions = ensure_decision_ids(decisions)

        # ── CIRCUIT BREAKER ──
        breaker = CircuitBreaker()
        if breaker.check_all(portfolio_summary, market_data):
            log.warning(f"Circuit Breaker Action: {breaker.get_action()}")
            if breaker.action == 'STOP_ALL_BUYS':
                decisions = [d for d in decisions if d.get('action') != 'BUY']
                warnings.append(f"🛑 Circuit Breaker: STOP_ALL_BUYS. {breaker.reason}")
            elif breaker.action == 'REDUCE_EXPOSURE':
                for d in decisions:
                    if d.get('action') == 'BUY':
                        d['target_allocation'] *= 0.5
                warnings.append(f"🛑 Circuit Breaker: REDUCE_EXPOSURE. {breaker.reason}")
            elif breaker.action == 'DE_RISK_50':
                forced_sells = []
                positions_list = sorted(positions.items(), key=lambda x: x[1].get('market_value', 0))
                total_sell_value = total_value * 0.5
                current_sell_value = 0
                for ticker, pos in positions_list:
                    if current_sell_value >= total_sell_value:
                        break
                    if pos.get('quantity', 0) > 0:
                        forced_sells.append({
                            'ticker': ticker,
                            'action': 'SELL',
                            'target_allocation': 0.0,
                            'confidence': 1.0,
                            'reason': '🛑 Circuit Breaker: DE_RISK_50',
                            'priority': 'CRITICAL'
                        })
                        current_sell_value += pos.get('market_value', 0)
                decisions.extend(forced_sells)
                warnings.append(f"🛑 Circuit Breaker: DE_RISK_50 (generated {len(forced_sells)} forced SELLs). {breaker.reason}")
            elif breaker.action == 'HALT_ALL':
                decisions = []
                warnings.append(f"🛑 Circuit Breaker: HALT_ALL (all trades blocked). {breaker.reason}")

        # ── EMERGENCY CASH MODE ──
        if self.emergency_cash_mode(portfolio_summary, market_data):
            for ticker, pos in positions.items():
                if pos.get('quantity', 0) > 0:
                    decisions.append({
                        'ticker': ticker,
                        'action': 'SELL',
                        'target_allocation': 0.0,
                        'confidence': 1.0,
                        'reason': 'EMERGENCY CASH MODE: daily loss exceeded',
                        'risk_approved': True,
                        'forced_rebalancing': True,
                    })
            warnings.append("🚨 EMERGENCY CASH MODE ACTIVATED: all positions liquidated")

        # ── ADAPTIVE CONFIDENCE THRESHOLDS ──
        # Ermittle benötigte Daten
        spy_data = market_data.get("SPY", {})
        market_momentum = spy_data.get("return_20d", 0.0) / 100.0   # als Dezimal
        vix = market_data.get("vix", None)
        cash_pct = portfolio_summary.get("cash_pct", 100.0) / 100.0
        invested_pct = 1.0 - cash_pct
        adaptive = self.get_adaptive_thresholds(
            regime_state=self.regime_state,
            vix=vix,
            market_momentum=market_momentum,
            cash_pct=cash_pct,
            invested_pct=invested_pct,
        )
        buy_conf_threshold = adaptive["buy"]
        sell_conf_threshold = adaptive["sell"]

        # Log für Journal (wird später in main.py aufgenommen)
        self._last_adaptive_log = {
            "buy_threshold": buy_conf_threshold,
            "sell_threshold": sell_conf_threshold,
            "vix": vix,
            "vix_adjustment": (buy_conf_threshold - REGIME_CONFIDENCE_THRESHOLDS.get(
                (self.regime_state.label if self.regime_state else "SIDEWAYS"), 
                REGIME_CONFIDENCE_THRESHOLDS["SIDEWAYS"]
            )["buy_threshold"]) if self.regime_state else 0,
            "momentum": market_momentum,
            "cash_pct": cash_pct,
        }

        # ─── UNGÜLTIGE SELLS FILTERN ──
        filtered_sells = []
        for d in decisions:
            if d.get("action") == "SELL" and d["ticker"] not in positions:
                is_special = d.get("zombie_cleanup") or d.get("stop_loss")
                if not is_special:
                    warnings.append(f"{d['ticker']}: SELL für nicht gehaltene Position → gefiltert")
                continue
            filtered_sells.append(d)
        decisions = filtered_sells

        # ── REBALANCING SELLS GENERIEREN ──
        rebalance_sells = self._generate_rebalancing_decisions(decisions, portfolio_summary)
        decisions = rebalance_sells + decisions

        # ── DYNAMISCHE MAX-TRADES (mit VIX-Adjustierung) ──
        vix_adj_pct = 0
        if vix is not None:
            if vix > 35:
                vix_adj_pct = -50
            elif vix > 25:
                vix_adj_pct = -20
        max_trades = self._dynamic_max_trades(market_data, vix_adj_pct)

        hold_decisions = [d for d in decisions if d.get("action") == "HOLD"]
        sell_decisions = [d for d in decisions if d.get("action") == "SELL"]
        buy_decisions = [d for d in decisions if d.get("action") == "BUY"]

        sector_exposure = self._sector_exposure(positions, total_value)

        for d in hold_decisions:
            d["risk_approved"] = True
            validated.append(d)

        for d in sell_decisions:
            ticker = d.get("ticker", "?")
            conf = d.get("confidence", 0)
            is_rebal = d.get("rebalancing", False)
            is_zombie = d.get("zombie_cleanup", False)
            is_stop = d.get("stop_loss", False)
            is_forced = d.get("forced_rebalancing", False)

            if not (is_rebal or is_zombie or is_stop or is_forced) and conf < sell_conf_threshold:
                d = dict(d)
                d["action"] = "HOLD"
                d["reason"] += f" [ADAPTIVE CONF: {conf:.0%} < SELL threshold {sell_conf_threshold:.0%}]"
                d["risk_approved"] = False
                validated.append(d)
                warnings.append(f"{ticker}: Konfidenz {conf:.0%} → HOLD")
                continue

            pos_market_value = positions.get(ticker, {}).get("market_value", 0)
            target_alloc = d.get("target_allocation", 0)
            sell_value = (pos_market_value if target_alloc == 0.0
                         else max(0, (pos_market_value / total_value - target_alloc) * total_value))
            running_cash += sell_value
            d["risk_approved"] = True
            validated.append(d)

        qualified = [d for d in buy_decisions if d.get("confidence", 0) >= buy_conf_threshold]
        qualified.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        top_buys = qualified[:max_trades]
        skip_buys = qualified[max_trades:]

        for d in skip_buys:
            d = dict(d)
            d["action"] = "HOLD"
            d["reason"] += f" [RISK: Max-Trades ({max_trades}) Limit]"
            d["risk_approved"] = False
            validated.append(d)
            warnings.append(f"{d['ticker']}: Max-Trades Limit → HOLD")

        for d in top_buys:
            ticker = d.get("ticker", "?")
            target_alloc = d.get("target_allocation", 0)
            current_value = positions.get(ticker, {}).get("market_value", 0)
            buy_cost = max(0, total_value * target_alloc - current_value)
            cash_after = running_cash - buy_cost
            cash_pct_after = cash_after / total_value if total_value > 0 else 0

            if cash_pct_after < min_cash:
                max_spend = running_cash - (total_value * min_cash)
                if max_spend >= 500:
                    reduced_target = (current_value + max_spend) / total_value
                    d["target_allocation"] = round(reduced_target, 4)
                    running_cash -= max_spend
                    warnings.append(f"{ticker}: Teilkauf auf {reduced_target:.0%} (${max_spend:,.0f})")
                    d["risk_approved"] = True
                    validated.append(d)
                else:
                    d = dict(d)
                    d["action"] = "HOLD"
                    d["reason"] += " [RISK: Cash reicht nicht]"
                    d["risk_approved"] = False
                    validated.append(d)
                    warnings.append(f"{ticker}: Cash reicht nicht → HOLD")
                continue

            running_cash -= buy_cost
            d["risk_approved"] = True
            validated.append(d)

        # Finale Cash-Invariante
        validated, cash_warnings = self._enforce_cash_invariant(
            validated, total_value, running_cash, min_cash, positions, market_data
        )
        warnings.extend(cash_warnings)

        validated = self._resolver.resolve(validated)

        for d in validated:
            if d.get("action") in ("BUY", "SELL") and d.get("risk_approved", False):
                d["status"] = "APPROVED"
            elif not d.get("risk_approved", True):
                d["status"] = "BLOCKED"
            else:
                d["status"] = "HOLD"

        log.info(
            f"Risikoprüfung abgeschlossen | "
            f"FINAL: {sum(1 for d in validated if d['action']=='SELL')} SELL | "
            f"{sum(1 for d in validated if d['action']=='BUY' and d.get('risk_approved'))} BUY | "
            f"Cash: {format_currency(running_cash)} ({running_cash/total_value:.0%})"
        )
        return validated, warnings

    # ========== HILFSMETHODEN (unverändert) ==========
    def _generate_rebalancing_decisions(self, decisions: List[Dict], portfolio_summary: Dict) -> List[Dict]:
        total_value = portfolio_summary.get("total_value", 1)
        positions = portfolio_summary.get("positions", {})
        target_map = {d["ticker"]: d.get("target_allocation", 0) for d in decisions if d.get("action") in ("BUY", "SELL")}
        rebalance_sells = []
        for ticker, pos in positions.items():
            current_alloc = pos.get("market_value", 0) / total_value
            target_alloc = target_map.get(ticker)
            if target_alloc is None:
                continue
            if current_alloc > target_alloc + 0.03:
                already_sell = any(d["ticker"] == ticker and d["action"] == "SELL" for d in decisions)
                if not already_sell:
                    rebalance_sells.append({
                        "ticker": ticker,
                        "action": "SELL",
                        "target_allocation": target_alloc,
                        "confidence": 0.80,
                        "reason": f"Rebalancing: {current_alloc:.0%} -> {target_alloc:.0%}",
                        "risk_approved": True,
                        "rebalancing": True,
                    })
        return rebalance_sells

    def _enforce_cash_invariant(
        self,
        validated: List[Dict],
        total_value: float,
        projected_cash: float,
        min_cash_pct: float,
        positions: Dict,
        market_data: Dict,
    ) -> Tuple[List[Dict], List[str]]:
        warnings = []
        min_cash_abs = total_value * min_cash_pct
        if projected_cash >= min_cash_abs:
            return validated, warnings

        shortfall = min_cash_abs - projected_cash
        log.warning(f"[HARD GATE] Cash-Invariante verletzt! Fehlbetrag: {format_currency(shortfall)}")

        approved_buys = [d for d in validated if d.get("action") == "BUY" and d.get("risk_approved")]
        approved_buys.sort(key=lambda x: x.get("confidence", 0))
        recovered = projected_cash
        for d in approved_buys:
            if recovered >= min_cash_abs:
                break
            ticker = d["ticker"]
            buy_cost = max(0.0, total_value * d.get("target_allocation", 0) - positions.get(ticker, {}).get("market_value", 0))
            d["action"] = "HOLD"
            d["reason"] += " [HARD GATE: BUY gecancelt für Cash-Reserve]"
            d["risk_approved"] = False
            recovered += buy_cost
            warnings.append(f"{ticker}: BUY gecancelt → Cash freigegeben ~{format_currency(buy_cost)}")

        if recovered >= min_cash_abs:
            return validated, warnings

        candidates = []
        for ticker, pos in positions.items():
            if any(d.get("ticker") == ticker and d.get("action") == "SELL" for d in validated):
                continue
            current_alloc = pos.get("market_value", 0) / total_value if total_value > 0 else 0
            market_val = pos.get("market_value", 0)
            if market_val <= 0:
                continue
            priority = current_alloc * (1 + max(0, -pos.get("unrealized_pnl_pct", 0) / 100))
            candidates.append({"ticker": ticker, "market_value": market_val, "priority": priority})

        candidates.sort(key=lambda x: -x["priority"])
        still_needed = min_cash_abs - recovered
        for cand in candidates:
            if still_needed <= 0:
                break
            sell_amount = min(cand["market_value"], still_needed * 1.05)
            if sell_amount < 100:
                continue
            forced_sell = {
                "ticker": cand["ticker"],
                "action": "SELL",
                "target_allocation": 0.0,
                "confidence": 1.0,
                "reason": "[AUTO-REBALANCING] Cash unter Minimum",
                "risk_approved": True,
                "forced_rebalancing": True,
            }
            validated.append(forced_sell)
            recovered += sell_amount
            still_needed -= sell_amount

        return validated, warnings

    def check_stop_loss(self, positions: Dict, market_data: Dict) -> List[Dict]:
        stop_loss_pct = self.settings["stop_loss_pct"]
        stop_loss_orders = []
        for ticker, pos in positions.items():
            avg_price = pos.get("avg_price", 0)
            current_price = market_data.get(ticker, {}).get("current_price", 0)
            if avg_price <= 0 or current_price <= 0:
                continue
            loss_pct = (current_price - avg_price) / avg_price
            if loss_pct < -stop_loss_pct:
                log.warning(f"STOP-LOSS: {ticker} | Verlust: {loss_pct:.1%}")
                stop_loss_orders.append({
                    "ticker": ticker,
                    "action": "SELL",
                    "target_allocation": 0.0,
                    "confidence": 1.0,
                    "reason": f"Stop-Loss: {loss_pct:.1%} Verlust",
                    "risk_approved": True,
                    "stop_loss": True,
                })
        return stop_loss_orders

    def calculate_portfolio_var(self, positions: Dict, market_data: Dict, total_value: float, confidence: float = 0.95) -> Dict:
        import numpy as np
        from scipy.stats import norm
        if total_value <= 0 or not positions:
            return {"var_pct": 0.0, "var_usd": 0.0, "component_vars": {}}
        weights = []
        vols = []
        tickers = []
        for ticker, pos in positions.items():
            alloc = pos.get("market_value", 0) / total_value
            vol_annual = (market_data.get(ticker, {}).get("volatility_annual_pct") or 20.0) / 100
            vol_daily = vol_annual / (252 ** 0.5)
            weights.append(alloc)
            vols.append(vol_daily)
            tickers.append(ticker)
        w = np.array(weights)
        v = np.array(vols)
        portfolio_vol = float(np.sqrt(np.sum((w * v) ** 2)))
        z_score = norm.ppf(confidence)
        var_pct = portfolio_vol * z_score
        var_usd = var_pct * total_value
        return {"var_pct": round(var_pct, 4), "var_usd": round(var_usd, 2)}

    def check_var_limit(self, positions: Dict, market_data: Dict, total_value: float) -> Tuple[bool, str]:
        max_var = self.settings.get("max_portfolio_var", 0.05)
        var_data = self.calculate_portfolio_var(positions, market_data, total_value)
        var_pct = var_data.get("var_pct", 0)
        if var_pct > max_var:
            return False, f"Portfolio VaR {var_pct:.1%} > max {max_var:.1%}"
        return True, f"VaR OK: {var_pct:.1%}"

    def get_risk_summary(self) -> Dict:
        return {
            "profile": self.risk_profile.value,
            "max_position_pct": self.settings["max_position_pct"],
            "min_cash_pct": self.settings["min_cash_pct"],
            "stop_loss_pct": self.settings["stop_loss_pct"],
            "max_trades_per_run": self.settings.get("max_trades_per_run", TOP_N_BUYS),
            "confidence_threshold": self.settings["confidence_threshold"],
        }

    def get_adaptive_log(self) -> Dict:
        """Gibt die letzten adaptiven Thresholds fürs Journal zurück."""
        return getattr(self, '_last_adaptive_log', {})
