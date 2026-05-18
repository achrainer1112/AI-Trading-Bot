"""
AI Trading Bot - Marktdaten-Collector (Enhanced)
==================================================
Sammelt erweiterte Metriken für alle Watchlist-Assets:
  - SMA20, SMA50, SMA90
  - Momentum 20d (separater Wert)
  - Relative Strength vs SPY
  - Erweitertes RSI
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from logger import log
from config import (
    FULL_WATCHLIST, PRICE_HISTORY_DAYS,
    SHORT_WINDOW, MEDIUM_WINDOW, LONG_WINDOW, EXTRA_LONG_WINDOW,
)

try:
    import yfinance as yf
except ImportError:
    log.error("yfinance nicht installiert. Führe aus: pip install yfinance")
    raise


class MarketDataCollector:
    """
    Sammelt und berechnet Marktdaten für alle Assets.
    Liefert deterministische, vollständige Metriken für Score Engine und LLM.
    """

    def __init__(self, watchlist: List[str] = None):
        self.watchlist = watchlist or FULL_WATCHLIST
        self.data_cache: Dict[str, pd.DataFrame] = {}

    def fetch_price_history(self, ticker: str, days: int = PRICE_HISTORY_DAYS) -> Optional[pd.DataFrame]:
        """Lädt historische OHLCV-Daten für einen Ticker."""
        try:
            end = datetime.today()
            start = end - timedelta(days=days + 20)   # Puffer
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df is None or getattr(df, "empty", True):
                log.warning(f"Keine Daten für {ticker}")
                return None

            # Robust Column Normalization: handle MultiIndex and inconsistent column names
            df = self._normalize_ohlcv_columns(df)

            # Ensure index is datetime and sorted
            try:
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
            except Exception:
                pass

            self.data_cache[ticker] = df
            return df
        except Exception as e:
            log.error(f"Fehler beim Laden von {ticker}: {e}")
            return None

    def calculate_metrics(self, ticker: str, spy_close: Optional[pd.Series] = None) -> Dict:
        """
        Berechnet alle relevanten Metriken für einen Ticker.
        Neu: SMA20, SMA50, momentum_20d, relative_strength, sma_distance_pct.
        """
        df = self.data_cache.get(ticker)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            df = self.fetch_price_history(ticker)
        if df is None or len(df) < 10:
            return {}

        close = df["Close"]
        current_price = float(close.iloc[-1])

        def safe_return(days: int) -> Optional[float]:
            if len(close) > days:
                return float((close.iloc[-1] / close.iloc[-days - 1] - 1) * 100)
            return None

        # Returns
        return_7d  = safe_return(SHORT_WINDOW)
        return_20d = safe_return(MEDIUM_WINDOW)
        return_30d = safe_return(30)
        return_90d = safe_return(EXTRA_LONG_WINDOW)

        # Volatilität
        daily_returns = close.pct_change().dropna()
        volatility = float(daily_returns.std() * np.sqrt(252) * 100)

        # --- MACD (12,26,9) ---
        macd_line = macd_signal = macd_hist = None
        try:
            if len(close) >= 26:
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd_s = ema12 - ema26
                macd_sig = macd_s.ewm(span=9, adjust=False).mean()
                macd_line = float(macd_s.iloc[-1])
                macd_signal = float(macd_sig.iloc[-1])
                macd_hist = float(macd_line - macd_signal)
        except Exception:
            macd_line = macd_signal = macd_hist = None

        # --- Bollinger Bands (20, 2σ) ---
        bb_upper = bb_middle = bb_lower = bb_position = None
        try:
            if len(close) >= 20:
                bb_middle_s = close.rolling(window=20).mean()
                bb_std_s = close.rolling(window=20).std()
                bb_middle = float(bb_middle_s.iloc[-1])
                std = float(bb_std_s.iloc[-1])
                bb_upper = float(bb_middle + 2 * std)
                bb_lower = float(bb_middle - 2 * std)
                if bb_upper is not None and bb_lower is not None and (bb_upper - bb_lower) != 0:
                    bb_position = float((current_price - bb_lower) / (bb_upper - bb_lower))
        except Exception:
            bb_upper = bb_middle = bb_lower = bb_position = None

        # Moving Averages
        sma_20 = float(close.tail(MEDIUM_WINDOW).mean()) if len(close) >= MEDIUM_WINDOW else None
        sma_50 = float(close.tail(LONG_WINDOW).mean()) if len(close) >= LONG_WINDOW else None
        sma_90 = float(close.tail(EXTRA_LONG_WINDOW).mean()) if len(close) >= EXTRA_LONG_WINDOW else None
        # Legacy compatibility
        sma_7  = float(close.tail(SHORT_WINDOW).mean())

        # SMA distance
        sma_distance_pct = None
        if sma_50:
            sma_distance_pct = (current_price - sma_50) / sma_50 * 100

        # RSI (14)
        rsi = self._calculate_rsi(close)

        # Relative Strength vs SPY (20d)
        relative_strength = None
        if spy_close is not None and ticker != "SPY":
            try:
                # Align indices
                aligned = pd.concat([close, spy_close], axis=1, join="inner")
                aligned.columns = ["ticker", "spy"]
                aligned = aligned.dropna()
                if len(aligned) > MEDIUM_WINDOW:
                    ticker_ret = (aligned["ticker"].iloc[-1] / aligned["ticker"].iloc[-(MEDIUM_WINDOW+1)] - 1)
                    spy_ret = (aligned["spy"].iloc[-1] / aligned["spy"].iloc[-(MEDIUM_WINDOW+1)] - 1)
                    if spy_ret != 0:
                        relative_strength = float((1 + ticker_ret) / (1 + spy_ret))
            except Exception:
                pass

        # Avg Volume
        avg_volume = float(df["Volume"].tail(20).mean()) if "Volume" in df else None

        # --- Open Gap % (Open today vs prev Close) ---
        open_gap_pct = None
        try:
            if "Open" in df and "Close" in df and len(df) >= 2:
                open_today = float(df["Open"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2])
                if prev_close != 0:
                    open_gap_pct = (open_today - prev_close) / prev_close * 100
        except Exception:
            open_gap_pct = None

        # 52-week high/low
        high_52w = low_52w = None
        if len(close) >= 252:
            high_52w = float(close.tail(252).max())
            low_52w = float(close.tail(252).min())

        return {
            "ticker": ticker,
            "current_price": current_price,
            # Returns
            "return_7d": return_7d,
            "return_20d": return_20d,
            "return_30d": return_30d,
            "return_90d": return_90d,
            # Volatility
            "volatility_annual_pct": round(volatility, 2),
            # SMAs
            "sma_7": round(sma_7, 2),
            "sma_20": round(sma_20, 2) if sma_20 else None,
            "sma_50": round(sma_50, 2) if sma_50 else None,
            "sma_90": round(sma_90, 2) if sma_90 else None,
            # Legacy keys for compatibility
            "sma_30": round(sma_50, 2) if sma_50 else None,
            "sma_90_compat": round(sma_90, 2) if sma_90 else None,
            # SMA flags
            "above_sma_20": current_price > sma_20 if sma_20 else None,
            "above_sma_30": current_price > sma_50 if sma_50 else None,   # kept for compat
            "above_sma_50": current_price > sma_50 if sma_50 else None,
            "above_sma_90": current_price > sma_90 if sma_90 else None,
            "sma_distance_pct": round(sma_distance_pct, 2) if sma_distance_pct is not None else None,
            # RSI
            "rsi_14": round(rsi, 2) if rsi is not None else None,
            # Relative Strength
            "relative_strength_vs_spy": round(relative_strength, 4) if relative_strength else None,
            # Volume
            "avg_volume_20d": int(avg_volume) if avg_volume else None,
            # MACD
            "macd_line": round(macd_line, 6) if macd_line is not None else None,
            "macd_signal": round(macd_signal, 6) if macd_signal is not None else None,
            "macd_histogram": round(macd_hist, 6) if macd_hist is not None else None,
            # Bollinger Bands
            "bb_upper": round(bb_upper, 4) if bb_upper is not None else None,
            "bb_middle": round(bb_middle, 4) if bb_middle is not None else None,
            "bb_lower": round(bb_lower, 4) if bb_lower is not None else None,
            "bb_position": round(bb_position, 4) if bb_position is not None else None,
            # Open Gap
            "open_gap_pct": round(open_gap_pct, 4) if open_gap_pct is not None else None,
            # 52-week
            "high_52w": round(high_52w, 2) if high_52w else None,
            "low_52w": round(low_52w, 2) if low_52w else None,
            # Meta
            "data_points": len(close),
            "last_updated": datetime.now().isoformat(),
        }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        if len(prices) < period + 1:
            return None
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    def collect_all(self) -> Dict[str, Dict]:
        """Sammelt Marktdaten für alle Assets und berechnet Relative Strength vs SPY."""
        log.info(f"Sammle Marktdaten für {len(self.watchlist)} Assets...")

        # SPY zuerst laden (Benchmark für Relative Strength)
        spy_close: Optional[pd.Series] = None
        if "SPY" in self.watchlist:
            spy_df = self.fetch_price_history("SPY")
            if spy_df is not None:
                spy_close = spy_df["Close"]

        results = {}
        for ticker in self.watchlist:
            log.debug(f"  → Lade {ticker}")
            # Wenn SPY schon im Cache, wird es wiederverwendet
            metrics = self.calculate_metrics(ticker, spy_close=spy_close if ticker != "SPY" else None)
            if metrics:
                results[ticker] = metrics
                log.debug(
                    f"  ✓ {ticker}: ${metrics.get('current_price', 0):.2f} | "
                    f"20d: {metrics.get('return_20d', 0) or 0:+.1f}% | "
                    f"RSI: {metrics.get('rsi_14', 'n/a')} | "
                    f"RS: {metrics.get('relative_strength_vs_spy', 'n/a')}"
                )
            else:
                log.warning(f"  ✗ Keine Daten für {ticker}")

        log.info(f"Marktdaten gesammelt: {len(results)}/{len(self.watchlist)} Assets.")
        return results

    def get_current_price(self, ticker: str) -> Optional[float]:
        try:
            data = yf.Ticker(ticker).fast_info
            price = getattr(data, "last_price", None)
            if price:
                return float(price)
            df = yf.download(ticker, period="2d", progress=False, auto_adjust=True)
            if not df.empty:
                return float(df["Close"].iloc[-1])
        except Exception as e:
            log.error(f"Fehler beim Abrufen des aktuellen Preises für {ticker}: {e}")
        return None

    def _normalize_ohlcv_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize dataframe columns to standard OHLCV names.

        Handles MultiIndex columns and common variants like 'Adj Close'.
        Leaves the dataframe intact if no mapping is found for a column.
        """
        # Flatten MultiIndex if present
        try:
            if isinstance(df.columns, pd.MultiIndex):
                df = df.copy()
                df.columns = [c[0] if isinstance(c, tuple) and c[0] else "_".join([str(x) for x in c]).strip() for c in df.columns]
        except Exception:
            pass

        cols_lower = {c.lower(): c for c in df.columns}

        # Priority matches for each target column
        candidates = {
            "Open": ["open"],
            "High": ["high"],
            "Low": ["low"],
            "Close": ["adj close", "adjclose", "adj_close", "close"],
            "Volume": ["volume"],
        }

        rename_map = {}
        available = list(df.columns)
        for target, keys in candidates.items():
            found = None
            for k in keys:
                # exact match
                for col in available:
                    if col.lower() == k:
                        found = col
                        break
                if found:
                    break
            if not found:
                # contains match
                for k in keys:
                    for col in available:
                        if k in col.lower():
                            found = col
                            break
                    if found:
                        break
            if found and found != target:
                # avoid overwriting if target already present
                if target in df.columns:
                    # prefer adjusted close over plain close: if target exists and found is 'Adj Close', replace mapping
                    if "adj" in found.lower() and target == "Close":
                        rename_map[found] = target
                else:
                    rename_map[found] = target

        if rename_map:
            try:
                df = df.rename(columns=rename_map)
            except Exception:
                pass

        return df