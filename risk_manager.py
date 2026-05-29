"""
AI Trading Bot - Risk Manager (Production) – Harte Guardrails
==============================================================
Keine Overrides des Risk Layers durch AI.
Bei Score < Threshold wird BUY strikt abgelehnt.
Zusätzlich: Konzentrationslimits (max. 25% Einzelposition, CVaR > 50% reduziert automatisch).
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
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
    CVAR_LIMIT_PCT,
    CVAR_CONFIDENCE_LEVEL,
    CVAR_LOOKBACK_DAYS,
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
            log.critical(f"🛑 CIRCUIT BREAKER 1: {self.reason}")
            return True
        drawdown = portfolio.get('drawdown_pct', 0.0)
        if drawdown < -0.10:
            self.triggered = True
            self.reason = f"Portfolio drawdown: {drawdown:.2%} < -10%"
            self.action = 'REDUCE_EXPOSURE'
            log.critical(f"🛑 CIRCUIT BREAKER 2: {self.reason}")
            return True
        vix = market_data.get('vix', 20.0)
        if vix > 35:
            self.triggered = True
            self.reason = f"VIX spike: {vix:.1f} > 35"
            self.action = 'DE_RISK_50'
            log.critical(f"🛑 CIRCUIT BREAKER 3: {self.reason}")
            return True
        if market_data.get('market_halted', False):
            self.triggered = True
            self.reason = "Market circuit breaker triggered"
            self.action = 'HALT_ALL'
            log.critical(f"🛑 CIRCUIT BREAKER 4: {self.reason}")
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


class CVaRRiskManager:
    """Conditional Value at Risk (Expected Shortfall) Management"""
    def __init__(self, cvar_limit_pct: float = 0.05, confidence_level: float = 0.95, lookback_days: int = 252):
        self.cvar_limit_pct = cvar_limit_pct
        self.confidence_level = confidence_level
        self.lookback_days = lookback_days
        self._last_cvar_state = None

    def calculate_portfolio_returns(self, positions: Dict[str, Dict], historical_returns: Dict[str, np.ndarray]) -> np.ndarray:
        if not positions or not historical_returns:
            return np.array([])
        min_len = min(len(r) for r in historical_returns.values() if len(r) > 0)
        if min_len < 2:
            return np.array([])
        total_value = sum(p.get("market_value", 0) for p in positions.values())
        if total_value <= 0:
            return np.array([])
        weights = {}
        for ticker, pos in positions.items():
            if ticker in historical_returns:
                weights[ticker] = pos.get("market_value", 0) / total_value
        portfolio_returns = np.zeros(min_len)
        for ticker, w in weights.items():
            rets = historical_returns[ticker][-min_len:]
            portfolio_returns += w * rets
        return portfolio_returns

    def calculate_cvar(self, portfolio_returns: np.ndarray, confidence_level: float = None) -> Tuple[float, float]:
        if len(portfolio_returns) < 2:
            return 0.0, 0.0
        conf = confidence_level or self.confidence_level
        var = np.percentile(portfolio_returns, (1 - conf) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return float(var), float(cvar)

    def evaluate_risk_state(self, portfolio_returns: np.ndarray) -> Dict:
        if len(portfolio_returns) < 2:
            return {"var_95": 0.0, "cvar_95": 0.0, "cvar_pct": 0.0, "limit_pct": self.cvar_limit_pct, "breach": False, "reduction_factor": 1.0}
        var, cvar = self.calculate_cvar(portfolio_returns)
        cvar_pct = abs(cvar)
        breach = cvar_pct > self.cvar_limit_pct
        reduction_factor = min(1.0, self.cvar_limit_pct / cvar_pct) if breach else 1.0
        state = {"var_95": var, "cvar_95": cvar, "cvar_pct": cvar_pct, "limit_pct": self.cvar_limit_pct, "breach": breach, "reduction_factor": reduction_factor}
        self._last_cvar_state = state
        return state

    def marginal_cvar_contribution(self, positions: Dict[str, Dict], historical_returns: Dict[str, np.ndarray]) -> Dict[str, float]:
        if not positions or not historical_returns:
            return {}
        total_value = sum(p.get("market_value", 0) for p in positions.values())
        if total_value <= 0:
            return {}
        portfolio_returns = self.calculate_portfolio_returns(positions, historical_returns)
        if len(portfolio_returns) < 2:
            return {}
        var, cvar = self.calculate_cvar(portfolio_returns)
        if cvar == 0:
            return {}
        contributions = {}
        for ticker, pos in positions.items():
            if ticker not in historical_returns:
                continue
            weight = pos.get("market_value", 0) / total_value
            asset_returns = historical_returns[ticker][-len(portfolio_returns):]
            cov = np.cov(asset_returns, portfolio_returns)[0, 1]
            marginal_beta = cov / np.var(portfolio_returns) if np.var(portfolio_returns) != 0 else 0
            marginal_contrib = weight * marginal_beta
            contributions[ticker] = float(marginal_contrib)
        total = sum(contributions.values())
        if total > 0:
            contributions = {t: v / total for t, v in contributions.items()}
        return contributions

    def simulate_trade_impact(self, current_positions: Dict[str, Dict], proposed_trade: Dict,
                              historical_returns: Dict[str, np.ndarray]) -> Dict:
        if not current_positions or not historical_returns:
            return {"cvar_before": 0, "cvar_after": 0, "cvar_change": 0, "breach": False}
        total_value_before = sum(p.get("market_value", 0) for p in current_positions.values())
        if total_value_before <= 0:
            return {"cvar_before": 0, "cvar_after": 0, "cvar_change": 0, "breach": False}
        port_returns_before = self.calculate_portfolio_returns(current_positions, historical_returns)
        _, cvar_before = self.calculate_cvar(port_returns_before)
        cvar_before = abs(cvar_before)
        new_positions = {t: {"market_value": v.get("market_value", 0)} for t, v in current_positions.items()}
        ticker = proposed_trade["ticker"]
        action = proposed_trade["action"]
        value = proposed_trade.get("value_usd", 0)
        if action == "BUY":
            if ticker in new_positions:
                new_positions[ticker]["market_value"] += value
            else:
                new_positions[ticker] = {"market_value": value}
        elif action == "SELL":
            if ticker in new_positions:
                new_positions[ticker]["market_value"] = max(0, new_positions[ticker]["market_value"] - value)
                if new_positions[ticker]["market_value"] == 0:
                    del new_positions[ticker]
        port_returns_after = self.calculate_portfolio_returns(new_positions, historical_returns)
        _, cvar_after = self.calculate_cvar(port_returns_after)
        cvar_after = abs(cvar_after)
        return {
            "cvar_before": cvar_before,
            "cvar_after": cvar_after,
            "cvar_change": cvar_after - cvar_before,
            "breach": cvar_after > self.cvar_limit_pct,
        }

    def cvar_adjusted_allocation(self, base_allocation: float, ticker: str,
                                 positions: Dict[str, Dict], historical_returns: Dict[str, np.ndarray],
                                 max_position_pct: float = 0.20) -> float:
        if not positions or not historical_returns or ticker not in historical_returns:
            return base_allocation
        contributions = self.marginal_cvar_contribution(positions, historical_returns)
        marginal = contributions.get(ticker, 0.05)
        if marginal <= 0.01:
            marginal = 0.01
        scale = 1.0 / marginal
        scale = max(0.5, min(2.0, scale))
        adjusted = base_allocation * scale
        return min(adjusted, max_position_pct)

    def apply_risk_mitigation(self, target_weights: Dict[str, float], positions: Dict[str, Dict],
                              historical_returns: Dict[str, np.ndarray], cvar_state: Dict,
                              volatility_data: Dict[str, float] = None, correlation_clusters: List[List[str]] = None) -> Dict[str, float]:
        if not cvar_state.get("breach", False):
            return target_weights
        reduction_factor = cvar_state.get("reduction_factor", 1.0)
        if reduction_factor >= 0.99:
            return target_weights
        log.warning(f"CVaR-Breach: {cvar_state['cvar_pct']:.2%} > limit {cvar_state['limit_pct']:.2%}, reduction factor {reduction_factor:.2f}")
        contributions = self.marginal_cvar_contribution(positions, historical_returns)
        if not contributions:
            return target_weights
        adjusted_weights = {}
        for ticker, target in target_weights.items():
            factor = reduction_factor
            marginal = contributions.get(ticker, 0.5)
            if marginal > 0.7:
                factor *= 0.8
            if volatility_data and ticker in volatility_data:
                vol = volatility_data[ticker]
                if vol > 40:
                    factor *= 0.7
                elif vol > 30:
                    factor *= 0.85
            adjusted_weights[ticker] = target * factor
        total_adj = sum(adjusted_weights.values())
        investable_before = sum(target_weights.values())
        if investable_before > 0 and total_adj > 0:
            scale = investable_before / total_adj
            adjusted_weights = {t: w * scale for t, w in adjusted_weights.items()}
        return adjusted_weights

    def filter_trades_by_cvar(self, trades: List[Dict], cvar_state: Dict, positions: Dict[str, Dict],
                              historical_returns: Dict[str, np.ndarray], volatility_data: Dict[str, float] = None) -> Tuple[List[Dict], List[Dict]]:
        if not cvar_state.get("breach", False):
            return trades, []
        blocked_buys = []
        allowed_trades = []
        for trade in trades:
            if trade.get("action") == "BUY":
                ticker = trade.get("ticker")
                if volatility_data and ticker in volatility_data and volatility_data[ticker] > 30:
                    blocked_buys.append(trade)
                    log.info(f"CVaR: BUY {ticker} blockiert (hohe Vola {volatility_data[ticker]:.0f}%)")
                    continue
                allowed_trades.append(trade)
            else:
                allowed_trades.append(trade)
        return allowed_trades, blocked_buys


class RiskManager:
    def __init__(self, risk_profile: RiskProfile = None):
        self.risk_profile = risk_profile or ACTIVE_RISK_PROFILE
        self.settings = RISK_SETTINGS[self.risk_profile].copy()
        self._base_settings = RISK_SETTINGS[self.risk_profile].copy()
        self._resolver = FinalDecisionResolver()
        self.regime_state = None
        self.cvar_manager = CVaRRiskManager(cvar_limit_pct=CVAR_LIMIT_PCT, confidence_level=CVAR_CONFIDENCE_LEVEL, lookback_days=CVAR_LOOKBACK_DAYS)
        log.info(f"Risk Manager initialisiert: Profil={self.risk_profile.value.upper()}")
        log.info(f"  Max Position: {self.settings['max_position_pct']*100:.0f}% | Min Cash: {self.settings['min_cash_pct']*100:.0f}% | Stop-Loss: {self.settings['stop_loss_pct']*100:.0f}% | Max Trades: {self.settings.get('max_trades_per_run', TOP_N_BUYS)}")
        log.info(f"  CVaR Limit: {CVAR_LIMIT_PCT:.1%} | Confidence: {CVAR_CONFIDENCE_LEVEL:.0%}")

    # ========== ADAPTIVE THRESHOLDS (unverändert) ==========
    def get_adaptive_thresholds(self, regime_state=None, vix: Optional[float] = None, market_momentum: float = 0.0,
                                cash_pct: float = 1.0, invested_pct: float = 0.0) -> Dict[str, float]:
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
        if vix is not None:
            if vix > 35:
                buy_threshold += 0.10
                log.info(f"VIX {vix:.1f} > 35 → BUY +10%")
            elif vix > 25:
                buy_threshold += 0.05
                log.info(f"VIX {vix:.1f} > 25 → BUY +5%")
        if market_momentum > 0.5:
            buy_threshold -= 0.03
            log.info(f"Momentum {market_momentum:.1%} > 50% → BUY -3%")
        elif market_momentum < -0.5:
            sell_threshold -= 0.05
            log.info(f"Momentum {market_momentum:.1%} < -50% → SELL -5%")
        if cash_pct < 0.08:
            buy_threshold += 0.05
            log.info(f"Cash {cash_pct:.1%} < 8% → BUY +5%")
        if invested_pct > 0.90:
            buy_threshold = max(buy_threshold, 0.75)
            log.info(f"Investiert {invested_pct:.1%} > 90% → BUY min 75%")
        buy_threshold = max(0.45, min(0.85, buy_threshold))
        sell_threshold = max(0.45, min(0.85, sell_threshold))
        self._last_adaptive_log = {"buy_threshold": buy_threshold, "sell_threshold": sell_threshold, "vix": vix, "vix_adjustment": (buy_threshold - base["buy_threshold"]), "momentum": market_momentum, "cash_pct": cash_pct}
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

    # ========== HAUPTVALIDIERUNG MIT HARTEN GUARDRAILS ==========
    def validate_decisions(
        self,
        decisions: List[Dict],
        portfolio_summary: Dict,
        market_data: Dict,
        historical_returns: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[List[Dict], List[str]]:
        warnings = []
        validated = []
        max_pos = self.settings["max_position_pct"]
        min_cash = self.settings["min_cash_pct"]
        max_sector = self.settings.get("max_sector_exposure", 0.45)

        total_value = portfolio_summary.get("total_value", 1)
        running_cash = portfolio_summary.get("cash", 0)
        positions = portfolio_summary.get("positions", {})

        decisions = ensure_decision_ids(decisions)

        # ── 1. CIRCUIT BREAKER ──
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
                        forced_sells.append({'ticker': ticker, 'action': 'SELL', 'target_allocation': 0.0, 'confidence': 1.0, 'reason': '🛑 Circuit Breaker: DE_RISK_50', 'priority': 'CRITICAL'})
                        current_sell_value += pos.get('market_value', 0)
                decisions.extend(forced_sells)
                warnings.append(f"🛑 Circuit Breaker: DE_RISK_50 (generated {len(forced_sells)} forced SELLs). {breaker.reason}")
            elif breaker.action == 'HALT_ALL':
                decisions = []
                warnings.append(f"🛑 Circuit Breaker: HALT_ALL (all trades blocked). {breaker.reason}")

        # ── 2. EMERGENCY CASH MODE ──
        if self.emergency_cash_mode(portfolio_summary, market_data):
            for ticker, pos in positions.items():
                if pos.get('quantity', 0) > 0:
                    decisions.append({'ticker': ticker, 'action': 'SELL', 'target_allocation': 0.0, 'confidence': 1.0, 'reason': 'EMERGENCY CASH MODE: daily loss exceeded', 'risk_approved': True, 'forced_rebalancing': True})
            warnings.append("🚨 EMERGENCY CASH MODE ACTIVATED: all positions liquidated")

        # ── 3. ADAPTIVE CONFIDENCE THRESHOLDS ──
        spy_data = market_data.get("SPY", {})
        market_momentum = spy_data.get("return_20d", 0.0) / 100.0
        vix = market_data.get("vix", None)
        cash_pct = portfolio_summary.get("cash_pct", 100.0) / 100.0
        invested_pct = 1.0 - cash_pct
        adaptive = self.get_adaptive_thresholds(self.regime_state, vix, market_momentum, cash_pct, invested_pct)
        buy_conf_threshold = adaptive["buy"]
        sell_conf_threshold = adaptive["sell"]

        # ─── 4. FILTER UNGÜLTIGER SELLS ──
        filtered_sells = []
        for d in decisions:
            if d.get("action") == "SELL" and d["ticker"] not in positions:
                is_special = d.get("zombie_cleanup") or d.get("stop_loss")
                if not is_special:
                    warnings.append(f"{d['ticker']}: SELL für nicht gehaltene Position → gefiltert")
                continue
            filtered_sells.append(d)
        decisions = filtered_sells

        # ── 5. REBALANCING SELLS GENERIEREN ──
        rebalance_sells = self._generate_rebalancing_decisions(decisions, portfolio_summary)
        decisions = rebalance_sells + decisions

        # ── 6. CVaR-INTEGRATION (falls historische Daten vorhanden) ──
        if historical_returns is not None:
            try:
                decisions, cvar_warnings = self.apply_cvar_constraints(
                    decisions=decisions,
                    portfolio_summary=portfolio_summary,
                    market_data=market_data,
                    historical_returns=historical_returns,
                )
                warnings.extend(cvar_warnings)
            except Exception as e:
                log.warning(f"CVaR-Prüfung fehlgeschlagen: {e}")

        # ── 7. DYNAMISCHE MAX-TRADES ──
        vix_adj_pct = 0
        if vix is not None:
            if vix > 35:
                vix_adj_pct = -50
            elif vix > 25:
                vix_adj_pct = -20
        max_trades = self._dynamic_max_trades(market_data, vix_adj_pct)

        # Trenne Entscheidungen
        hold_decisions = [d for d in decisions if d.get("action") == "HOLD"]
        sell_decisions = [d for d in decisions if d.get("action") == "SELL"]
        buy_decisions = [d for d in decisions if d.get("action") == "BUY"]

        # ── 8. HARD GUARDRAIL: BUY nur wenn Score >= buy_conf_threshold (KEIN OVERRIDE!) ──
        # Kein "GUARDRAIL OVERRIDE" mehr. Wenn Score zu niedrig, wird BUY strikt abgelehnt.
        qualified_buys = []
        for d in buy_decisions:
            # Verwende entweder quant_score (falls vorhanden) oder confidence als Proxy
            score = d.get("quant_score", d.get("confidence", 0) * 100)
            if score >= buy_conf_threshold * 100:   # threshold in % umrechnen
                qualified_buys.append(d)
            else:
                d = dict(d)
                d["action"] = "HOLD"
                d["reason"] += f" [HARD GUARDRAIL: Score {score:.0f} < BUY threshold {buy_conf_threshold*100:.0f}]"
                d["risk_approved"] = False
                validated.append(d)
                warnings.append(f"{d['ticker']}: BUY blockiert (Score {score:.0f} < {buy_conf_threshold*100:.0f})")

        # Sortiere nach Konfidenz
        qualified_buys.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        top_buys = qualified_buys[:max_trades]
        skip_buys = qualified_buys[max_trades:]

        for d in skip_buys:
            d = dict(d)
            d["action"] = "HOLD"
            d["reason"] += f" [RISK: Max-Trades ({max_trades}) Limit]"
            d["risk_approved"] = False
            validated.append(d)
            warnings.append(f"{d['ticker']}: Max-Trades Limit → HOLD")

        # ── 9. CASH-VALIDIERUNG FÜR TOP BUYS ──
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

        # ── 10. HOLD- und SELL-Entscheidungen (unverändert) ──
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
            sell_value = (pos_market_value if target_alloc == 0.0 else max(0, (pos_market_value / total_value - target_alloc) * total_value))
            running_cash += sell_value
            d["risk_approved"] = True
            validated.append(d)

        # ── 11. KONZENTRATIONSLIMIT (neue Regel) ──
        # Wenn ein Asset > 25% Gewicht oder CVaR-Beitrag > 50%, erzwinge Reduktion
        cvar_contributions = {}
        if historical_returns is not None:
            try:
                cvar_contributions = self.cvar_manager.marginal_cvar_contribution(positions, historical_returns)
            except Exception:
                pass
        for d in validated:
            if d.get("action") == "BUY":
                ticker = d["ticker"]
                new_weight = d.get("target_allocation", 0)
                cvar_contrib = cvar_contributions.get(ticker, 0)
                if new_weight > 0.25 or cvar_contrib > 0.5:
                    # Reduziere Zielgewicht
                    reduced = min(0.20, new_weight * 0.7)
                    d["target_allocation"] = reduced
                    d["reason"] += f" [CONCENTRATION: Gewicht {new_weight:.1%} -> {reduced:.1%} (Limit 25% / CVaR {cvar_contrib:.0%})]"
                    log.warning(f"Konzentration reduziert: {ticker} {new_weight:.1%} -> {reduced:.1%}")

        # ── 12. CASH-INVARIANTE ──
        validated, cash_warnings = self._enforce_cash_invariant(
            validated, total_value, running_cash, min_cash, positions, market_data
        )
        warnings.extend(cash_warnings)

        # ── 13. KONFLIKTE AUFLÖSEN ──
        validated = self._resolver.resolve(validated)

        for d in validated:
            if d.get("action") in ("BUY", "SELL") and d.get("risk_approved", False):
                d["status"] = "APPROVED"
            elif not d.get("risk_approved", True):
                d["status"] = "BLOCKED"
            else:
                d["status"] = "HOLD"

        log.info(f"Risikoprüfung abgeschlossen | FINAL: {sum(1 for d in validated if d['action']=='SELL')} SELL | {sum(1 for d in validated if d['action']=='BUY' and d.get('risk_approved'))} BUY | Cash: {format_currency(running_cash)} ({running_cash/total_value:.0%})")
        return validated, warnings

    # ─── HILFSMETHODEN (unverändert) ───
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
                    rebalance_sells.append({"ticker": ticker, "action": "SELL", "target_allocation": target_alloc, "confidence": 0.80, "reason": f"Rebalancing: {current_alloc:.0%} -> {target_alloc:.0%}", "risk_approved": True, "rebalancing": True})
        return rebalance_sells

    def _enforce_cash_invariant(self, validated: List[Dict], total_value: float, projected_cash: float, min_cash_pct: float, positions: Dict, market_data: Dict) -> Tuple[List[Dict], List[str]]:
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
            forced_sell = {"ticker": cand["ticker"], "action": "SELL", "target_allocation": 0.0, "confidence": 1.0, "reason": "[AUTO-REBALANCING] Cash unter Minimum", "risk_approved": True, "forced_rebalancing": True}
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
                stop_loss_orders.append({"ticker": ticker, "action": "SELL", "target_allocation": 0.0, "confidence": 1.0, "reason": f"Stop-Loss: {loss_pct:.1%} Verlust", "risk_approved": True, "stop_loss": True})
        return stop_loss_orders

    def calculate_portfolio_var(self, positions: Dict, market_data: Dict, total_value: float, confidence: float = 0.95) -> Dict:
        from scipy.stats import norm
        if total_value <= 0 or not positions:
            return {"var_pct": 0.0, "var_usd": 0.0}
        weights, vols = [], []
        for ticker, pos in positions.items():
            alloc = pos.get("market_value", 0) / total_value
            vol_annual = (market_data.get(ticker, {}).get("volatility_annual_pct") or 20.0) / 100
            vol_daily = vol_annual / (252 ** 0.5)
            weights.append(alloc)
            vols.append(vol_daily)
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
        return {"profile": self.risk_profile.value, "max_position_pct": self.settings["max_position_pct"], "min_cash_pct": self.settings["min_cash_pct"], "stop_loss_pct": self.settings["stop_loss_pct"], "max_trades_per_run": self.settings.get("max_trades_per_run", TOP_N_BUYS), "confidence_threshold": self.settings["confidence_threshold"]}

    def get_adaptive_log(self) -> Dict:
        return getattr(self, '_last_adaptive_log', {})

    def apply_cvar_constraints(self, decisions: List[Dict], portfolio_summary: Dict, market_data: Dict, historical_returns: Dict[str, np.ndarray]) -> Tuple[List[Dict], List[str]]:
        if historical_returns is None:
            return decisions, []
        positions = portfolio_summary.get("positions", {})
        total_value = portfolio_summary.get("total_value", 1)
        if total_value <= 0:
            return decisions, []
        port_returns = self.cvar_manager.calculate_portfolio_returns(positions, historical_returns)
        cvar_state = self.cvar_manager.evaluate_risk_state(port_returns)
        if not cvar_state["breach"]:
            return decisions, []
        volatility_data = {t: d.get("volatility_annual_pct", 20.0) for t, d in market_data.items()}
        allowed, blocked = self.cvar_manager.filter_trades_by_cvar(decisions, cvar_state, positions, historical_returns, volatility_data)
        warnings = [f"CVaR constraint active: {cvar_state['cvar_pct']:.2%} > limit {cvar_state['limit_pct']:.2%}"]
        for b in blocked:
            warnings.append(f"Blocked BUY {b['ticker']} due to CVaR breach")
        return allowed, warnings
