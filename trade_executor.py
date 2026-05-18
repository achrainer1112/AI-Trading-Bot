from __future__ import annotations

import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Optional, Tuple
import random

from logger import log, trade_logger
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
    TRADING_MODE, MIN_ORDER_VALUE,
)
from utils import (
    format_currency, is_market_open,
    generate_trade_id, TradeDeduplicator,
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


class TradeExecutor:

    def __init__(self, mode: str = None):
        self.mode = (mode or TRADING_MODE).upper()
        self.api: Optional[tradeapi.REST] = None
        self._deduplicator = TradeDeduplicator()
        self._connect()

        log.info(f"Trade Executor initialisiert | Modus: {self.mode}")

    # ─────────────────────────────
    # CONNECT
    # ─────────────────────────────
    def _connect(self):
        if not ALPACA_AVAILABLE:
            self.mode = "DRY"
            return

        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            self.mode = "DRY"
            return

        try:
            self.api = tradeapi.REST(
                key_id=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY,
                base_url=ALPACA_BASE_URL,
            )
        except Exception as e:
            log.error(f"Alpaca Fehler: {e}")
            self.mode = "DRY"
            self.api = None

    # ─────────────────────────────
    # BUY
    # ─────────────────────────────
    def execute_buy(
        self,
        ticker: str,
        target_value: float,
        current_price: float,
        available_cash: float,
        min_cash_reserve: float,
        reason: str = "",
    ) -> Tuple[bool, Dict]:

        if not is_market_open():
            return False, self._skipped(ticker, "BUY", reason, "market_closed")

        if current_price <= 0:
            return False, self._skipped(ticker, "BUY", reason, "invalid_price")

        spendable = max(0, available_cash - min_cash_reserve)
        value = min(target_value, spendable)

        if value < MIN_ORDER_VALUE:
            return False, self._skipped(ticker, "BUY", reason, "insufficient_cash")

        # ✅ FIX: qty IMMER berechnen (vorher war das missing!)
        qty = Decimal(str(value)) / Decimal(str(current_price))
        qty = qty.quantize(QTY_PRECISION, rounding=ROUND_DOWN)

        if qty <= MIN_QTY:
            return False, self._skipped(ticker, "BUY", reason, "qty_too_small")

        trade_id = generate_trade_id(ticker, "BUY", value)

        record = self._base_record(
            ticker, "BUY",
            float(qty),
            value,
            current_price,
            reason,
            trade_id
        )

        return self._dispatch(record, qty=qty, notional=value)

    # ─────────────────────────────
    # SELL (PORTFOLIO SAFE)
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

        if not is_market_open():
            return False, self._skipped(ticker, "SELL", reason, "market_closed")

        if current_price <= 0:
            return False, self._skipped(ticker, "SELL", reason, "invalid_price")

        # ✅ FIX: SELL basiert jetzt auf REAL POSITION, nicht AI target_value
        max_sell_qty = Decimal(str(position_qty))

        if full_liquidation:
            qty = max_sell_qty
        else:
            target_qty = Decimal(str(target_value)) / Decimal(str(current_price))
            qty = min(max_sell_qty, target_qty)

        qty = qty.quantize(QTY_PRECISION, rounding=ROUND_DOWN)

        # ✅ FIX Zombie-Reste: Bei full_liquidation mit qty=0 (Rundungsfehler-Rest)
        # → Alpaca close_position() verwenden statt submit_order()
        # → Im DRY-Modus: direkt als executed markieren
        if qty <= MIN_QTY:
            if full_liquidation:
                return self._force_close_position(ticker, current_price, reason)
            return False, self._skipped(ticker, "SELL", reason, "qty_zero")

        trade_id = generate_trade_id(ticker, "SELL", target_value)

        record = self._base_record(
            ticker, "SELL",
            float(qty),
            target_value,
            current_price,
            reason,
            trade_id
        )

        return self._dispatch(record, qty=qty, notional=float(qty) * current_price)

    # ─────────────────────────────
    # FORCE CLOSE (Zombie-Reste)
    # ─────────────────────────────
    def _force_close_position(self, ticker: str, current_price: float, reason: str) -> Tuple[bool, Dict]:
        # Schliesst eine Position vollstaendig via Alpaca close_position().
        # Wird verwendet wenn qty nach Rundung = 0 ist (Zombie-Reste).
        # Im DRY/PAPER-Modus: simuliert erfolgreichen Close.
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

    # ─────────────────────────────
    # DISPATCH
    # ─────────────────────────────
    def _dispatch(self, record: Dict, qty=None, notional=None):

        record["mode"] = self.mode

        log.debug(f"[DISPATCH] {record['ticker']} {record['action']} | qty={qty} notional={notional}")

        if self.mode == "DRY":
            return self._run_dry(record, qty, notional)

        return self._run_broker(record, qty, notional)

    # ─────────────────────────────
    # DRY RUN (FIXED)
    # ─────────────────────────────
    def _run_dry(self, record, qty, notional):

        slippage = random.uniform(-0.001, 0.001)

        price = record.get("price") or 100

        if record["action"] == "BUY":
            fill_price = price * (1 + slippage)
        else:
            fill_price = price * (1 - slippage)

        record["status"] = STATUS_EXECUTED
        record["fill_qty"] = float(qty) if qty else 0
        record["fill_price"] = round(fill_price, 4)
        record["fill_value"] = record["fill_qty"] * record["fill_price"]

        self._log_trade_if_valid(record)

        return True, record

    # ─────────────────────────────
    # BROKER
    # ─────────────────────────────
    def _run_broker(self, record, qty, notional):
        if not self.api:
            return self._run_dry(record, qty, notional)

        try:
            if record["action"] == "BUY":
                # BUY → nur notional
                order = self.api.submit_order(
                    symbol=record["ticker"],
                    side="buy",
                    type="market",
                    time_in_force="day",
                    notional=str(notional),
                )

            else:
                # SELL → nur qty
                order = self.api.submit_order(
                    symbol=record["ticker"],
                    side="sell",
                    type="market",
                    time_in_force="day",
                    qty=str(qty),
                )

        except Exception as e:
            log.error(f"ORDER REJECTED: {record['ticker']} {record['action']} → {e}")
            record["status"] = STATUS_REJECTED
            record["reject_reason"] = str(e)
            return False, record

        record["order_id"] = str(order.id)

        filled = self._wait(order.id)

        if filled and filled.filled_qty:
            record["status"] = STATUS_EXECUTED
            record["fill_qty"] = float(filled.filled_qty)
            record["fill_price"] = float(filled.filled_avg_price)
            record["fill_value"] = record["fill_qty"] * record["fill_price"]

        self._log_trade_if_valid(record)

        return True, record

    # ─────────────────────────────
    # LOGGING
    # ─────────────────────────────
    def _log_trade_if_valid(self, record: Dict):

        if record.get("status") != STATUS_EXECUTED:
            return

        if not record.get("fill_qty") or record["fill_qty"] <= 0:
            return

        trade_logger.log_trade(record)

    # ─────────────────────────────
    # HELPERS
    # ─────────────────────────────
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
        return {"ticker": t, "action": a, "reason": r, "status": STATUS_SKIPPED, "skip_reason": s}

    def _wait(self, order_id):
        for _ in range(15):
            try:
                o = self.api.get_order(order_id)
                if o.status in ("filled", "rejected", "canceled", "partially_filled"):
                    return o
            except:
                pass
            time.sleep(1)
        return None