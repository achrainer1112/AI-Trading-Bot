"""
execution_safety.py - Live execution safety helpers
===============================================

Dieses Modul stellt die Klassen bereit, die für die Live-/Paper-Trade-Schutzschicht
benötigt werden. Es implementiert minimale, sichere Default-Verhalten für den
Workflow, damit die Kernlogik des Bots funktioniert.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils import load_json_file, save_json_file


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------

class DuplicateOrderError(Exception):
    pass


class CapitalSafetyError(Exception):
    pass


class ExecutionAbortError(Exception):
    pass


# -----------------------------------------------------------------------------
# Idempotency / duplicate order prevention
# -----------------------------------------------------------------------------

class IdempotencyStore:
    FILE = "logs/idempotency_ids.json"

    def __init__(self):
        os.makedirs(os.path.dirname(self.FILE) or ".", exist_ok=True)
        self.ids = set(load_json_file(self.FILE, []))

    @staticmethod
    def generate_id(ticker: str, side: str, value: float) -> str:
        return f"{side}:{ticker}:{value:.2f}"

    def is_duplicate(self, exec_id: str) -> bool:
        return exec_id in self.ids

    def register(self, exec_id: str) -> None:
        self.ids.add(exec_id)
        save_json_file(self.FILE, sorted(self.ids))


# -----------------------------------------------------------------------------
# Live trading guard
# -----------------------------------------------------------------------------

class LiveTradingGuard:
    def __init__(self, execution_mode: str, allow_live_trading: bool):
        self.execution_mode = execution_mode
        self.allow_live_trading = allow_live_trading
        self.system_health = {
            "api_connected": execution_mode != "LIVE" or allow_live_trading,
            "mode": execution_mode,
        }

        if self.execution_mode == "LIVE" and not self.allow_live_trading:
            raise EnvironmentError(
                "LIVE-Mode erfordert ALLOW_LIVE_TRADING=true. "
                "Systemhalt ausgelöst."
            )

    def validate_system_health(self, api: Optional[Any]) -> bool:
        if self.execution_mode == "LIVE":
            return api is not None
        return True

    def validate_market_data(self, market_data: Dict[str, Any]) -> bool:
        return bool(market_data)

    def validate_ai_response(self, ai_response: Dict[str, Any]) -> bool:
        return bool(ai_response)

    def assert_order_allowed(self) -> bool:
        if self.execution_mode == "LIVE" and not self.allow_live_trading:
            raise ExecutionAbortError("Live trading is not permitted by configuration.")
        return True

    def status_snapshot(self) -> Dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "allow_live_trading": self.allow_live_trading,
            "system_health": self.system_health,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


# -----------------------------------------------------------------------------
# Drawdown kill-switch
# -----------------------------------------------------------------------------

class DrawdownMonitor:
    def __init__(self, guard: Optional[LiveTradingGuard] = None, limit_pct: float = 0.08):
        self.guard = guard
        self.limit_pct = limit_pct

    def check(
        self,
        portfolio_value: float,
        peak_value: float,
        api: Optional[Any] = None,
    ) -> bool:
        if peak_value <= 0 or portfolio_value <= 0:
            return False

        current_pct = portfolio_value / peak_value
        if current_pct <= 1.0 - self.limit_pct:
            return True
        return False


# -----------------------------------------------------------------------------
# Live run logging
# -----------------------------------------------------------------------------

class LiveRunLogger:
    def __init__(self, log_dir: str = "logs/live"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_run(
        self,
        guard_snapshot: Dict[str, Any],
        executed_trades: List[Dict[str, Any]],
        rejected_trades: List[Dict[str, Any]],
        portfolio_snapshot: Dict[str, Any],
        risk_state: Dict[str, Any],
        regime_state: Optional[Dict[str, Any]],
        run_summary: Dict[str, Any],
    ) -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "guard_snapshot": guard_snapshot,
            "executed_trades": executed_trades,
            "rejected_trades": rejected_trades,
            "portfolio_snapshot": portfolio_snapshot,
            "risk_state": risk_state,
            "regime_state": regime_state,
            "run_summary": run_summary,
        }

        filename = os.path.join(
            self.log_dir,
            f"live_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
        )
        save_json_file(filename, payload)
