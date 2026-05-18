"""
AI Trading Bot - Streamlit Dashboard
=======================================
Interaktives Dashboard für Portfolio-Monitoring.

STARTEN:
  streamlit run dashboard.py

Zeigt:
  - Portfolio-Wert Verlauf
  - Asset-Allokation (Pie Chart)
  - Gewinn/Verlust pro Position
  - Trade-History
  - Backtest-Ergebnisse
"""

import json
import os
from datetime import datetime
from typing import Dict, List

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Seitenkonfiguration
st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import aus Bot-Modulen
import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    PORTFOLIO_FILE, TRADE_LOG_FILE, PORTFOLIO_HISTORY_FILE,
    TRADING_MODE, ACTIVE_RISK_PROFILE, FULL_WATCHLIST,
    BACKTEST_START_DATE, BACKTEST_END_DATE,
)
from utils import load_json_file, format_currency


# ─── Hilfsfunktionen ─────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_portfolio() -> Dict:
    return load_json_file(PORTFOLIO_FILE, default={"positions": {}, "cash": 0})

@st.cache_data(ttl=60)
def load_portfolio_history() -> List[Dict]:
    return load_json_file(PORTFOLIO_HISTORY_FILE, default=[])

@st.cache_data(ttl=60)
def load_trade_history() -> List[Dict]:
    return load_json_file(TRADE_LOG_FILE, default=[])


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ AI Trading Bot")
    st.divider()

    mode_color = {"PAPER": "🟡", "REAL": "🔴", "DRY": "🟢"}.get(TRADING_MODE.upper(), "⚪")
    st.metric("Modus", f"{mode_color} {TRADING_MODE.upper()}")
    st.metric("Risikoprofil", ACTIVE_RISK_PROFILE.value.upper())

    st.divider()
    if st.button("🔄 Daten neu laden", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.button("▶️ Trading Run starten", use_container_width=True):
        with st.spinner("Trading Run läuft..."):
            try:
                from main import TradingBot
                bot = TradingBot()
                result = bot.run()
                st.success(f"Run abgeschlossen! Trades: {result.get('trades_executed', 0)}")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Fehler: {e}")

    st.divider()
    st.caption(f"Letztes Update: {datetime.now().strftime('%H:%M:%S')}")


# ─── HAUPTBEREICH ─────────────────────────────────────────────────────────────

st.title("📈 AI Trading Bot Dashboard")

portfolio = load_portfolio()
history = load_portfolio_history()
trades = load_trade_history()

positions = portfolio.get("positions", {})
cash = portfolio.get("cash", 0)
initial = portfolio.get("initial_capital", 100_000)
invested = sum(p.get("market_value", 0) for p in positions.values())
total_value = cash + invested
pnl = total_value - initial
pnl_pct = (pnl / initial * 100) if initial > 0 else 0

# ─── METRICS ROW ──────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("💰 Portfolio-Wert", f"${total_value:,.0f}", f"{pnl_pct:+.2f}%")
with col2:
    st.metric("💵 Cash", f"${cash:,.0f}", f"{cash/total_value*100:.1f}%" if total_value > 0 else "")
with col3:
    st.metric("📊 Investiert", f"${invested:,.0f}")
with col4:
    st.metric("📈 P&L", f"${pnl:,.0f}", f"{pnl_pct:+.2f}%",
              delta_color="normal" if pnl >= 0 else "inverse")
with col5:
    st.metric("🔢 Positionen", len(positions))

# Markt-Regime aus letztem Journal-Eintrag anzeigen
_last_journal = load_trade_history()  # Wiederverwendung – Journal separat laden wenn nötig
try:
    import json as _json, os as _os
    _journal_path = "logs/journal.json"
    if _os.path.exists(_journal_path):
        with open(_journal_path) as _f:
            _journal_data = _json.load(_f)
        if _journal_data:
            _last_regime = _journal_data[-1].get("market_regime")
            if _last_regime:
                _regime_label = _last_regime.get("regime", "?").upper()
                _regime_emoji = {"BULL": "🟢", "BEAR": "🔴", "SIDEWAYS": "🟡"}.get(_regime_label, "⚪")
                _regime_conf  = _last_regime.get("confidence", 0)
                _vix          = _last_regime.get("vix")
                _desc         = _last_regime.get("description", "")
                st.info(
                    f"{_regime_emoji} **Markt-Regime: {_regime_label}** "
                    f"(Konfidenz: {_regime_conf:.0%}"
                    f"{f' | VIX={_vix:.1f}' if _vix else ''})"
                    f"  \n{_desc}"
                )
except Exception:
    pass

st.divider()

# ─── HAUPTCHARTS ──────────────────────────────────────────────────────────────

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📉 Portfolio-Verlauf")
    if history:
        df_hist = pd.DataFrame(history)
        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
        df_hist = df_hist.sort_values("timestamp")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist["timestamp"],
            y=df_hist["total_value"],
            mode="lines+markers",
            name="Portfolio Value",
            line=dict(color="#00D4AA", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.1)",
        ))
        # Startkapital-Linie
        fig.add_hline(y=initial, line_dash="dash", line_color="gray",
                      annotation_text="Startkapital", annotation_position="right")

        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="",
            yaxis_title="USD",
            hovermode="x unified",
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
            yaxis=dict(gridcolor="rgba(128,128,128,0.2)", tickprefix="$"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Noch keine Portfolio-Historie. Führe einen Trading-Run aus.")

with col_right:
    st.subheader("🥧 Asset-Allokation")
    if positions or cash > 0:
        labels = list(positions.keys()) + ["CASH"]
        values = [p.get("market_value", 0) for p in positions.values()] + [cash]
        colors = px.colors.qualitative.Set3[:len(labels)]

        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo="label+percent",
            marker=dict(colors=colors),
        )])
        fig_pie.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="v", x=1.05),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Keine offenen Positionen.")

# ─── POSITIONEN TABELLE ────────────────────────────────────────────────────────

st.subheader("📋 Aktuelle Positionen")
if positions:
    rows = []
    for ticker, pos in positions.items():
        pnl_pos = pos.get("unrealized_pnl", 0)
        pnl_pct_pos = pos.get("unrealized_pnl_pct", 0)
        rows.append({
            "Ticker": ticker,
            "Menge": f"{pos.get('quantity', 0):.4f}",
            "Ø Kaufpreis": f"${pos.get('avg_price', 0):.2f}",
            "Aktuell": f"${pos.get('current_price', 0):.2f}",
            "Marktwert": f"${pos.get('market_value', 0):,.2f}",
            "P&L $": f"${pnl_pos:+,.2f}",
            "P&L %": f"{pnl_pct_pos:+.2f}%",
            "Allokation": f"{pos.get('market_value', 0) / total_value * 100:.1f}%" if total_value else "0%",
        })
    df_pos = pd.DataFrame(rows)

    def color_pnl(val):
        if isinstance(val, str) and val.startswith("$"):
            num = float(val.replace("$", "").replace(",", "").replace("+", ""))
            return "color: #00C851" if num >= 0 else "color: #FF4444"
        return ""

    st.dataframe(df_pos, use_container_width=True, hide_index=True)
else:
    st.info("Keine offenen Positionen. Cash: " + format_currency(cash))

# ─── TRADE HISTORY ────────────────────────────────────────────────────────────

st.subheader("📝 Trade-History")
if trades:
    df_trades = pd.DataFrame(trades[-50:])  # Letzte 50 Trades
    df_trades = df_trades.sort_values("logged_at", ascending=False) if "logged_at" in df_trades.columns else df_trades

    display_cols = ["timestamp", "ticker", "action", "quantity", "price", "value", "mode", "reason", "status"]
    available = [c for c in display_cols if c in df_trades.columns]
    st.dataframe(df_trades[available].head(20), use_container_width=True, hide_index=True)
else:
    st.info("Noch keine Trades ausgeführt.")

# ─── BACKTEST SECTION ─────────────────────────────────────────────────────────

with st.expander("🔬 Backtest ausführen", expanded=False):
    st.write(f"Zeitraum: {BACKTEST_START_DATE} → {BACKTEST_END_DATE}")
    col_bt1, col_bt2 = st.columns(2)
    with col_bt1:
        bt_tickers = st.multiselect("Ticker", FULL_WATCHLIST, default=FULL_WATCHLIST[:8])
    with col_bt2:
        bt_start = st.date_input("Start", value=pd.to_datetime(BACKTEST_START_DATE))
        bt_end = st.date_input("Ende", value=pd.to_datetime(BACKTEST_END_DATE))

    if st.button("▶️ Backtest starten"):
        with st.spinner("Backtest läuft..."):
            from backtester import Backtester
            bt = Backtester(
                tickers=bt_tickers,
                start_date=str(bt_start),
                end_date=str(bt_end),
            )
            results = bt.run_all_profiles()

            if results:
                metrics_data = []
                for profile, r in results.items():
                    if r:
                        metrics_data.append({
                            "Profil": profile.upper(),
                            "Gesamtrendite": f"{r.get('total_return_pct', 0):+.2f}%",
                            "Ann. Rendite": f"{r.get('annualized_return_pct', 0):+.2f}%",
                            "Max Drawdown": f"{r.get('max_drawdown_pct', 0):.2f}%",
                            "Sharpe Ratio": f"{r.get('sharpe_ratio', 0):.3f}",
                            "Trades": r.get("total_trades", 0),
                            "Endwert": f"${r.get('final_value', 0):,.0f}",
                        })
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

st.divider()
st.caption("AI Trading Bot | Paper Trading aktiv | Alle Trades dienen experimentellen Zwecken")