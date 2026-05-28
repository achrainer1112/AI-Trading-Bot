"""
trade_executor.py – Phase 4A: Live Trading Safety Layer + Retry Logic
========================================================
"""

from __future__ import annotations

import time
import random
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Optional, Tuple

from logger import log, trade_logger
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
    TRADING_MODE, MIN_ORDER_VALUE,
    ALLOW_LIVE_TRADING,
    RISK_SETTINGS, ACTIVE_RISK_PROFILE,
)
from utils import (
    format_currency, is_market_open,
    generate_trade_id, TradeDeduplicator,
)
from execution_safety import (
    LiveTradingGuard,
    DrawdownMonitor,
    IdempotencyStore,
    CapitalSafetyChecker,
    DuplicateOrderError,
    CapitalSafetyError,
    ExecutionAbortError,
)

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    log.warning("alpaca-trade-api nicht installiert – nur DRY RUN möglich.")


QTY_PRECISION = Decimal("0.00000001")
MIN_QTY = Decimal("0.00000001")

STATUS_EXECUTED = "EXECUTED"
STATUS_SKIPPED = "SKIPPED"
STATUS_REJECTED = "REJECTED"
STATUS_ABORTED = "ABORTED"

_profile = RISK_SETTINGS[ACTIVE_RISK_PROFILE]


class TradeExecutor:

    def __init__(self, mode: str = None):
        self.mode = (mode or TRADING_MODE).upper()
        self.api: Optional[tradeapi.REST] = None
        self._deduplicator = TradeDeduplicator()

        self._idempotency = IdempotencyStore()
        self._capital_checker = CapitalSafetyChecker(
            min_cash_pct=_profile["min_cash_pct"],
            max_position_pct=_profile["max_position_pct"],
        )

        self._connect()
        self._init_guard()

        log.info(f"Trade Executor initialisiert | Modus: {self.mode}")

    def _connect(self):
        if not ALPACA_AVAILABLE:
            if self.mode == "LIVE":
                raise RuntimeError("LIVE-Mode erfordert alpaca-trade-api. pip install alpaca-trade-api")
            self.mode = "DRY"
            return

        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            if self.mode == "LIVE":
                raise RuntimeError("LIVE-Mode erfordert ALPACA_API_KEY + ALPACA_SECRET_KEY in .env")
            self.mode = "DRY"
            return

        try:
            self.api = tradeapi.REST(
                key_id=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY,
                base_url=ALPACA_BASE_URL,
            )
            log.info(f"[EXECUTOR] Alpaca verbunden | URL: {ALPACA_BASE_URL}")
        except Exception as e:
            if self.mode == "LIVE":
                raise RuntimeError(f"LIVE-Mode: Alpaca-Verbindung fehlgeschlagen: {e}")
            log.error(f"Alpaca Fehler: {e}")
            self.mode = "DRY"
            self.api = None

    def _init_guard(self):
        try:
            self._guard = LiveTradingGuard(
                execution_mode=self.mode,
                allow_live_trading=ALLOW_LIVE_TRADING,
            )
        except EnvironmentError as e:
            log.critical(f"[EXECUTOR] SYSTEM STOP: {e}")
            raise

        if self.mode == "LIVE":
            ok = self._guard.validate_system_health(self.api)
            if not ok:
                raise RuntimeError(f"LIVE System-Health-Check fehlgeschlagen: {self._guard.system_health}")

    def get_guard(self) -> Optional[LiveTradingGuard]:
        return self._guard

    def get_drawdown_monitor(self, limit_pct: float = 0.08) -> DrawdownMonitor:
        return DrawdownMonitor(self._guard, limit_pct=limit_pct)

    def _submit_with_retry(self, order_func, max_retries: int = 3, base_delay: float = 1.0):
        last_exception = None
        for attempt in range(max_retries):
            try:
                return order_func()
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                if 'rate limit' in error_msg or 'too many requests' in error_msg or 'timeout' in error_msg or 'connection' in error_msg:
                    wait = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    log.warning(f"Transienter Fehler (Versuch {attempt+1}/{max_retries}): {e}. Warte {wait:.1f}s")
                    time.sleep(wait)
                    continue
                else:
                    raise
        raise last_exception

    # ─────────────────────────────
    # BUY (mit Debug-Logs)
    # ─────────────────────────────
    def execute_buy(
        self,
        ticker: str,
        target_value: float,
        current_price: float,
        available_cash: float,
        min_cash_reserve: float,
        reason: str = "",
        total_portfolio_value: float = 0.0,
        current_position_value: float = 0.0,
    ) -> Tuple[bool, Dict]:

        log.info(f"🔍 [EXECUTOR] execute_buy called: {ticker}, target_value={target_value}, price={current_price}, available_cash={available_cash}")

        market_open = is_market_open()
        log.info(f"🔍 [EXECUTOR] is_market_open() = {market_open}")

        if not market_open:
            log.warning(f"Market closed: BUY {ticker} skipped")
            return False, self._skipped(ticker, "BUY", reason, "market_closed")

        if current_price <= 0:
            log.warning(f"Invalid price {current_price} for {ticker}")
            return False, self._skipped(ticker, "BUY", reason, "invalid_price")

        spendable = max(0, available_cash - min_cash_reserve)
        value = min(target_value, spendable)

        if value < MIN_ORDER_VALUE:
            log.warning(f"Value ${value:.2f} < MIN_ORDER_VALUE ${MIN_ORDER_VALUE:.2f}")
            return False, self._skipped(ticker, "BUY", reason, "insufficient_cash")

        qty = Decimal(str(value)) / Decimal(str(current_price))
        qty = qty.quantize(QTY_PRECISION, rounding=ROUND_DOWN)

        if qty <= MIN_QTY:
            log.warning(f"Qty {qty} <= MIN_QTY")
            return False, self._skipped(ticker, "BUY", reason, "qty_too_small")

        exec_id = IdempotencyStore.generate_id(ticker, "BUY", value)
        if self._idempotency.is_duplicate(exec_id):
            log.warning(f"[IDEMPOTENCY] {ticker} BUY ${value:,.0f} bereits heute ausgeführt – übersprungen.")
            return False, self._skipped(ticker, "BUY", reason, "duplicate_order")

        if self.mode == "LIVE" and total_portfolio_value > 0:
            try:
                self._capital_checker.check_buy(
                    ticker=ticker,
                    buy_value=value,
                    available_cash=available_cash,
                    total_value=total_portfolio_value,
                    current_position_value=current_position_value,
                )
            except CapitalSafetyError as e:
                log.error(str(e))
                return False, self._aborted(ticker, "BUY", reason, str(e))

        if self.mode == "LIVE":
            try:
                self._guard.assert_order_allowed()
            except ExecutionAbortError as e:
                return False, self._aborted(ticker, "BUY", reason, str(e))

        trade_id = generate_trade_id(ticker, "BUY", value)

        record = self._base_record(ticker, "BUY", float(qty), value, current_price, reason, trade_id)
        record["execution_id"] = exec_id

        log.info(f"🔍 [EXECUTOR] Dispatching BUY order for {ticker}: qty={qty}, notional={value}")
        ok, record = self._dispatch(record, qty=qty, notional=value)

        if ok:
            self._idempotency.register(exec_id)
            log.info(f"✅ [EXECUTOR] BUY order for {ticker} executed successfully")
        else:
            log.error(f"❌ [EXECUTOR] BUY order for {ticker} failed: {record.get('status')} - {record.get('reject_reason', '')}")

        return ok, record

    # ─────────────────────────────
    # SELL
    # ─────────────────────────────
    def execute_sell(
        self,
        ticker: str,
        position_qty: float,
        current_price: float,
        target_value: float,
        reason: str = "",
        full_liquidation: bool = False,
    ) -> Tuple[bool, Dict]:

        log.info(f"🔍 [EXECUTOR] execute_sell called: {ticker}, target_value={target_value}, price={current_price}")

        if not is_market_open():
            return False, self._skipped(ticker, "SELL", reason, "market_closed")

        if current_price <= 0:
            return False, self._skipped(ticker, "SELL", reason, "invalid_price")

        max_sell_qty = Decimal(str(position_qty))

        if full_liquidation:
            qty = max_sell_qty
        else:
            target_qty = Decimal(str(target_value)) / Decimal(str(current_price))
            qty = min(max_sell_qty, target_qty)

        qty = qty.quantize(QTY_PRECISION, rounding=ROUND_DOWN)

        if qty <= MIN_QTY:
            if full_liquidation:
                return self._force_close_position(ticker, current_price, reason)
            return False, self._skipped(ticker, "SELL", reason, "qty_zero")

        exec_id = IdempotencyStore.generate_id(ticker, "SELL", target_value)
        if self._idempotency.is_duplicate(exec_id):
            log.warning(f"[IDEMPOTENCY] {ticker} SELL bereits heute ausgeführt – übersprungen.")
            return False, self._skipped(ticker, "SELL", reason, "duplicate_order")

        if self.mode == "LIVE":
            try:
                self._guard.assert_order_allowed()
            except ExecutionAbortError as e:
                return False, self._aborted(ticker, "SELL", reason, str(e))

        trade_id = generate_trade_id(ticker, "SELL", target_value)

        record = self._base_record(ticker, "SELL", float(qty), target_value, current_price, reason, trade_id)
        record["execution_id"] = exec_id

        ok, record = self._dispatch(record, qty=qty, notional=float(qty) * current_price)

        if ok:
            self._idempotency.register(exec_id)

        return ok, record

    def _force_close_position(self, ticker: str, current_price: float, reason: str) -> Tuple[bool, Dict]:
        log.info(f"[FORCE CLOSE] {ticker}: Zombie-Rest via close_position() liquidieren")
        record = self._base_record(ticker, "SELL", 0.0, 0.0, current_price, reason,
                                   generate_trade_id(ticker, "FORCE_CLOSE", current_price))
        record["mode"] = self.mode
        record["force_close"] = True

        if self.mode == "DRY" or not self.api:
            record["status"] = STATUS_EXECUTED
            record["fill_qty"] = 0.0
            record["fill_price"] = current_price
            record["fill_value"] = 0.0
            log.info(f"[FORCE CLOSE] {ticker}: DRY-Mode -> als bereinigt markiert")
            self._log_trade_if_valid(record)
            return True, record

        if self.mode == "LIVE":
            try:
                self._guard.assert_order_allowed()
            except ExecutionAbortError as e:
                return False, self._aborted(ticker, "SELL", reason, str(e))

        try:
            self.api.close_position(ticker)
            record["status"] = STATUS_EXECUTED
            record["fill_qty"] = 0.0
            record["fill_price"] = current_price
            record["fill_value"] = 0.0
            log.info(f"[FORCE CLOSE] {ticker}: Position erfolgreich geschlossen")
        except Exception as e:
            log.warning(f"[FORCE CLOSE] {ticker}: close_position fehlgeschlagen: {e} -> ignoriere")
            record["status"] = STATUS_EXECUTED
            record["fill_qty"] = 0.0
            record["fill_price"] = 0.0
            record["fill_value"] = 0.0

        self._log_trade_if_valid(record)
        return True, record

    def _dispatch(self, record: Dict, qty=None, notional=None):
        record["mode"] = self.mode
        log.debug(f"[DISPATCH] {record['ticker']} {record['action']} | qty={qty} notional={notional}")

        if self.mode == "DRY":
            return self._run_dry(record, qty, notional)
        return self._run_broker(record, qty, notional)

    def _run_dry(self, record, qty, notional):
        slippage = random.uniform(-0.001, 0.001)
        price = record.get("price") or 100
        fill_price = price * (1 + slippage) if record["action"] == "BUY" else price * (1 - slippage)
        record["status"] = STATUS_EXECUTED
        record["fill_qty"] = float(qty) if qty else 0
        record["fill_price"] = round(fill_price, 4)
        record["fill_value"] = record["fill_qty"] * record["fill_price"]
        self._log_trade_if_valid(record)
        return True, record

    def _run_broker(self, record, qty, notional):
        if not self.api:
            return self._run_dry(record, qty, notional)

        def submit_order():
            if record["action"] == "BUY":
                return self.api.submit_order(
                    symbol=record["ticker"],
                    side="buy",
                    type="market",
                    time_in_force="day",
                    notional=str(round(notional, 2)),
                )
            else:
                return self.api.submit_order(
                    symbol=record["ticker"],
                    side="sell",
                    type="market",
                    time_in_force="day",
                    qty=str(qty),
                )

        try:
            log.info(f"🔍 [BROKER] Submitting {record['action']} order for {record['ticker']}, notional={notional}, qty={qty}")
            order = self._submit_with_retry(submit_order, max_retries=3)
            log.info(f"🔍 [BROKER] Order submitted: id={order.id}")
        except Exception as e:
            log.error(f"ORDER REJECTED: {record['ticker']} {record['action']} → {e}")
            record["status"] = STATUS_REJECTED
            record["reject_reason"] = str(e)
            return False, record

        record["order_id"] = str(order.id)
        filled = self._wait_for_fill(order.id)

        if filled is None:
            log.warning(f"[FILL TIMEOUT] {record['ticker']} {record['action']} | order_id={order.id} -> Status unknown")
            record["status"] = "PENDING_FILL"
            record["fill_qty"] = 0
            record["fill_price"] = 0
            record["fill_value"] = 0
            return False, record

        if filled.status == "rejected":
            log.error(f"[ORDER REJECTED] {record['ticker']} | Grund: {getattr(filled, 'failed_at', 'unknown')}")
            record["status"] = STATUS_REJECTED
            record["reject_reason"] = f"Alpaca rejected: status={filled.status}"
            return False, record

        if filled.status == "canceled":
            log.warning(f"[ORDER CANCELED] {record['ticker']}")
            record["status"] = STATUS_REJECTED
            record["reject_reason"] = "canceled"
            return False, record

        filled_qty = float(filled.filled_qty or 0)
        filled_avg = float(filled.filled_avg_price or 0)

        if filled_qty <= 0:
            log.warning(f"[PARTIAL/ZERO FILL] {record['ticker']} filled_qty=0 -> als rejected markiert")
            record["status"] = STATUS_REJECTED
            record["reject_reason"] = "zero_fill"
            return False, record

        if filled.status == "partially_filled":
            log.warning(f"[PARTIAL FILL] {record['ticker']} {record['action']} | filled={filled_qty} von geplant qty={qty}")
            record["partial_fill"] = True

        record["status"] = STATUS_EXECUTED
        record["fill_qty"] = filled_qty
        record["fill_price"] = filled_avg
        record["fill_value"] = filled_qty * filled_avg

        self._log_trade_if_valid(record)
        return True, record

    def _wait_for_fill(self, order_id: str, max_attempts: int = 20, interval: float = 1.5):
        terminal = {"filled", "rejected", "canceled", "partially_filled", "expired"}
        for attempt in range(max_attempts):
            try:
                o = self.api.get_order(order_id)
                if o.status in terminal:
                    return o
                log.debug(f"[FILL WAIT] order_id={order_id} status={o.status} (attempt {attempt+1}/{max_attempts})")
            except Exception as e:
                log.warning(f"[FILL WAIT] get_order fehlgeschlagen: {e}")
            time.sleep(interval)
        log.warning(f"[FILL TIMEOUT] order_id={order_id} nach {max_attempts} Versuchen nicht terminal")
        return None

    def _log_trade_if_valid(self, record: Dict):
        if record.get("status") != STATUS_EXECUTED:
            return
        if not record.get("fill_qty") or record["fill_qty"] <= 0:
            return
        trade_logger.log_trade(record)

    def _base_record(self, ticker, action, qty, value, price, reason, trade_id):
        return {
            "trade_id": trade_id,
            "ticker": ticker,
            "action": action,
            "planned_qty": qty,
            "planned_value": value,
            "price": price,
            "reason": reason,
            "mode": self.mode,
            "status": "pending",
            "timestamp": datetime.now().isoformat(),
            "fill_qty": None,
            "fill_price": None,
            "fill_value": None,
        }

    def _skipped(self, t, a, r, s):
        return {
            "ticker": t, "action": a, "reason": r,
            "status": STATUS_SKIPPED, "skip_reason": s,
        }

    def _aborted(self, t, a, r, abort_reason: str):
        log.error(f"[ORDER ABORTED] {t} {a} | {abort_reason}")
        return {
            "ticker": t, "action": a, "reason": r,
            "status": STATUS_ABORTED, "abort_reason": abort_reason,
        }
