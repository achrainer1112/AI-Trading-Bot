"""
AI Trading Bot - Logger
========================
Zentrales Logging-System für alle Aktionen des Bots.
Schreibt in Datei und Konsole, speichert Trade-History als JSON.
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from config import LOG_DIR, TRADE_LOG_FILE, PORTFOLIO_HISTORY_FILE


def setup_logger(name: str = "ai_trader") -> logging.Logger:
    """Initialisiert den Logger mit File- und Console-Handler."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Verhindere doppelte Handler bei mehrfachem Aufruf
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler (INFO+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (DEBUG+)
    log_file = os.path.join(LOG_DIR, f"trader_{datetime.now().strftime('%Y%m%d')}.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# Globaler Logger
log = setup_logger()


class TradeLogger:
    """Persistentes Logging von Trades und Portfolio-Snapshots als JSON."""

    def __init__(self):
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        self.trade_file = TRADE_LOG_FILE
        self.portfolio_history_file = PORTFOLIO_HISTORY_FILE

    def _load_json(self, filepath: str) -> List[Dict]:
        """Lädt eine JSON-Datei oder gibt leere Liste zurück."""
        try:
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    return json.load(f)
        except Exception as e:
            log.warning(f"Konnte {filepath} nicht laden: {e}")
        return []

    def _save_json(self, filepath: str, data: List[Dict]):
        """Speichert Daten als JSON."""
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            log.error(f"Fehler beim Speichern von {filepath}: {e}")

    def log_trade(self, trade: Dict[str, Any]):
        trades = self._load_json(self.trade_file)
        trade = trade.copy()

        price = float(trade.get("price") or 0.0)

        qty = trade.get("fill_qty") or trade.get("planned_qty") or 0.0

        trade["price"] = price
        trade["logged_at"] = datetime.now().isoformat()

        trades.append(trade)
        self._save_json(self.trade_file, trades)

        log.info(
            f"[TRADE] {trade.get('action','?')} {trade.get('ticker','?')} | "
            f"Qty: {float(qty):.6f} | "
            f"Price: ${price:.2f} | "
            f"Status: {trade.get('status','?')} | "
            f"Mode: {trade.get('mode','?')}"
        )

    def log_portfolio_snapshot(self, snapshot: Dict[str, Any]):
        """Portfolio-Snapshot speichern (täglich/nach jedem Run)."""
        history = self._load_json(self.portfolio_history_file)
        snapshot["timestamp"] = datetime.now().isoformat()
        history.append(snapshot)
        # Behalte max. 365 Snapshots
        if len(history) > 365:
            history = history[-365:]
        self._save_json(self.portfolio_history_file, history)
        log.info(
            f"[PORTFOLIO] Total Value: ${snapshot.get('total_value', 0):,.2f} | "
            f"Cash: ${snapshot.get('cash', 0):,.2f} | "
            f"P&L: {snapshot.get('pnl_pct', 0):+.2f}%"
        )

    def get_trade_history(self) -> List[Dict]:
        """Alle geloggten Trades zurückgeben."""
        return self._load_json(self.trade_file)

    def get_portfolio_history(self) -> List[Dict]:
        """Portfolio-Verlauf zurückgeben."""
        return self._load_json(self.portfolio_history_file)


# Singleton TradeLogger
trade_logger = TradeLogger()