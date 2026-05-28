"""
order_aggregator.py – Aggregation von Trades pro Asset
Vermeidet fragmentierte Mini-Orders.
"""

from typing import Dict, List
from logger import log


def aggregate_trades(trades: List[Dict]) -> List[Dict]:
    """
    Gruppiert Trades nach (ticker, action) und summiert target_allocation.
    Beachtet: Bei gemischten BUY/SELL für gleichen Ticker wird NETTO berechnet.
    """
    if not trades:
        return []

    # Schritt 1: Summe pro Ticker und Aktion
    agg = {}
    for t in trades:
        key = (t["ticker"], t["action"])
        agg[key] = agg.get(key, 0.0) + t["target_allocation"]

    # Schritt 2: Netting (BUY + SELL für gleichen Ticker)
    net = {}
    for (ticker, action), alloc in agg.items():
        net[ticker] = net.get(ticker, 0.0) + (alloc if action == "BUY" else -alloc)

    # Schritt 3: Neue Trade-Liste (nur wenn net != 0)
    result = []
    for ticker, delta in net.items():
        if abs(delta) < 0.005:   # unter 0.5% ignorieren
            continue
        action = "BUY" if delta > 0 else "SELL"
        # Confidence = gewichteter Durchschnitt (vereinfacht: nehme ersten)
        confidence = next((t["confidence"] for t in trades if t["ticker"] == ticker), 0.7)
        result.append({
            "ticker": ticker,
            "action": action,
            "target_allocation": abs(delta),
            "confidence": confidence,
            "reason": f"Aggregated from {len([t for t in trades if t['ticker'] == ticker])} signals",
        })
    log.info(f"Trade aggregation: {len(trades)} → {len(result)} orders")
    return result
