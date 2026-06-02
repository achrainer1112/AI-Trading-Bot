"""
AI Trading Bot - Marktdaten-Collector (robust)
================================================
Sammelt Marktdaten und stellt sicher, dass jeder Eintrag ein Dict ist.
"""

import warnings
warnings.filterwarnings("ignore", category=Warning, module="yfinance")
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow is deprecated.*")

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
    def __init__(self, watchlist: List[str] = None):
        self.watchlist = watchlist or FULL_WATCHLIST
        self.data_cache: Dict[str, pd.DataFrame] = {}

    def fetch_price_history(self, ticker: str, days: int = PRICE_HISTORY_DAYS) -> Optional[pd.DataFrame]:
        try:
            end = datetime.today()
            start = end - timedelta(days=days + 120)
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df is None or df.empty:
                log.warning(f"Keine Daten für {ticker}")
                return None
            df = self._normalize_ohlcv_columns(df)
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

    def _empty_metrics(self, ticker: str) -> Dict:
        """Leeres, aber vollständiges Dict für fehlgeschlagene Ticker."""
        return {
            "ticker": ticker,
            "current_price": 0.0,
            "return_7d": None,
            "return_20d": None,
            "return_30d": None,
            "return_60d": None,
            "return_90d": None,
            "volatility_annual_pct": 20.0,
            "sma_7": None,
            "sma_20": None,
            "sma_50": None,
            "sma_90": None,
            "sma_200": None,
            "above_sma_20": None,
            "above_sma_30": None,
            "above_sma_50": None,
            "above_sma_90": None,
            "sma_distance_pct": None,
            "rsi_14": None,
            "relative_strength_vs_spy": None,
            "avg_volume_20d": None,
            "macd_line": None,
            "macd_signal": None,
            "macd_histogram": None,
            "bb_upper": None,
            "bb_middle": None,
            "bb_lower": None,
            "bb_position": None,
            "atr_14": None,
            "ema_12": None,
            "ema_26": None,
            "open_gap_pct": None,
            "high_52w": None,
            "low_52w": None,
            "data_points": 0,
            "last_updated": datetime.now().isoformat(),
        }

    def calculate_metrics(self, ticker: str, spy_close: Optional[pd.Series] = None) -> Dict:
        """
        Berechnet Metriken; gibt IMMER ein Dict zurück.
        Bei Fehlern: leeres Dict mit Nones.
        """
        df = self.data_cache.get(ticker)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            df = self.fetch_price_history(ticker)
        if df is None or len(df) < 10:
            return self._empty_metrics(ticker)

        try:
            close = df["Close"]
            high = df["High"] if "High" in df else close
            low = df["Low"] if "Low" in df else close
            current_price = float(close.iloc[-1])

            def safe_return(days: int) -> Optional[float]:
                if len(close) > days:
                    return float((close.iloc[-1] / close.iloc[-days - 1] - 1) * 100)
                return None

            return_7d = safe_return(SHORT_WINDOW)
            return_20d = safe_return(MEDIUM_WINDOW)
            return_30d = safe_return(30)
            return_60d = safe_return(60)
            return_90d = safe_return(EXTRA_LONG_WINDOW)

            daily_returns = close.pct_change().dropna()
            volatility = float(daily_returns.std() * np.sqrt(252) * 100) if len(daily_returns) >= 5 else 20.0

            # ATR
            atr_14 = None
            try:
                if len(close) >= 15:
                    high_low = high - low
                    high_close = (high - close.shift()).abs()
                    low_close = (low - close.shift()).abs()
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr_14 = float(tr.rolling(14).mean().iloc[-1])
            except Exception:
                pass

            # MACD
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
                pass

            # EMA
            ema_12 = ema_26 = None
            try:
                if len(close) >= 26:
                    ema_12 = float(close.ewm(span=12, adjust=False).mean().iloc[-1])
                    ema_26 = float(close.ewm(span=26, adjust=False).mean().iloc[-1])
            except Exception:
                pass

            # Bollinger
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
                pass

            sma_20 = float(close.tail(MEDIUM_WINDOW).mean()) if len(close) >= MEDIUM_WINDOW else None
            sma_50 = float(close.tail(LONG_WINDOW).mean()) if len(close) >= LONG_WINDOW else None
            sma_90 = float(close.tail(EXTRA_LONG_WINDOW).mean()) if len(close) >= EXTRA_LONG_WINDOW else None
            sma_200 = float(close.tail(200).mean()) if len(close) >= 200 else None
            sma_7 = float(close.tail(SHORT_WINDOW).mean()) if len(close) >= SHORT_WINDOW else current_price

            sma_distance_pct = None
            if sma_50:
                sma_distance_pct = (current_price - sma_50) / sma_50 * 100

            rsi = self._calculate_rsi(close)

            relative_strength = None
            if spy_close is not None and ticker != "SPY":
                try:
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

            avg_volume = float(df["Volume"].tail(20).mean()) if "Volume" in df else None
            open_gap_pct = None
            try:
                if "Open" in df and "Close" in df and len(df) >= 2:
                    open_today = float(df["Open"].iloc[-1])
                    prev_close = float(df["Close"].iloc[-2])
                    if prev_close != 0:
                        open_gap_pct = (open_today - prev_close) / prev_close * 100
            except Exception:
                pass

            high_52w = low_52w = None
            if len(close) >= 252:
                high_52w = float(close.tail(252).max())
                low_52w = float(close.tail(252).min())

            return {
                "ticker": ticker,
                "current_price": current_price,
                "return_7d": return_7d,
                "return_20d": return_20d,
                "return_30d": return_30d,
                "return_60d": return_60d,
                "return_90d": return_90d,
                "volatility_annual_pct": round(volatility, 2),
                "sma_7": round(sma_7, 2),
                "sma_20": round(sma_20, 2) if sma_20 else None,
                "sma_50": round(sma_50, 2) if sma_50 else None,
                "sma_90": round(sma_90, 2) if sma_90 else None,
                "sma_200": round(sma_200, 2) if sma_200 else None,
                "above_sma_20": current_price > sma_20 if sma_20 else None,
                "above_sma_30": current_price > sma_50 if sma_50 else None,
                "above_sma_50": current_price > sma_50 if sma_50 else None,
                "above_sma_90": current_price > sma_90 if sma_90 else None,
                "sma_distance_pct": round(sma_distance_pct, 2) if sma_distance_pct is not None else None,
                "rsi_14": round(rsi, 2) if rsi is not None else None,
                "relative_strength_vs_spy": round(relative_strength, 4) if relative_strength else None,
                "avg_volume_20d": int(avg_volume) if avg_volume else None,
                "macd_line": round(macd_line, 6) if macd_line is not None else None,
                "macd_signal": round(macd_signal, 6) if macd_signal is not None else None,
                "macd_histogram": round(macd_hist, 6) if macd_hist is not None else None,
                "bb_upper": round(bb_upper, 4) if bb_upper is not None else None,
                "bb_middle": round(bb_middle, 4) if bb_middle is not None else None,
                "bb_lower": round(bb_lower, 4) if bb_lower is not None else None,
                "bb_position": round(bb_position, 4) if bb_position is not None else None,
                "atr_14": round(atr_14, 4) if atr_14 is not None else None,
                "ema_12": round(ema_12, 2) if ema_12 is not None else None,
                "ema_26": round(ema_26, 2) if ema_26 is not None else None,
                "open_gap_pct": round(open_gap_pct, 4) if open_gap_pct is not None else None,
                "high_52w": round(high_52w, 2) if high_52w else None,
                "low_52w": round(low_52w, 2) if low_52w else None,
                "data_points": len(close),
                "last_updated": datetime.now().isoformat(),
            }
        except Exception as e:
            log.debug(f"Fehler bei Metrikberechnung für {ticker}: {e}")
            return self._empty_metrics(ticker)

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
        log.info(f"Sammle Marktdaten für {len(self.watchlist)} Assets...")
        spy_close = None
        if "SPY" in self.watchlist:
            spy_df = self.fetch_price_history("SPY")
            if spy_df is not None:
                spy_close = spy_df["Close"]

        results = {}
        for ticker in self.watchlist:
            log.debug(f"  → Lade {ticker}")
            metrics = self.calculate_metrics(ticker, spy_close=spy_close if ticker != "SPY" else None)
            if not isinstance(metrics, dict):
                log.warning(f"  ✗ Ungültiges Format für {ticker}: {type(metrics)} – ersetze durch leeres Dict")
                metrics = self._empty_metrics(ticker)
            results[ticker] = metrics
            if metrics.get("current_price", 0) > 0:
                log.debug(f"  ✓ {ticker}: ${metrics.get('current_price', 0):.2f}")
            else:
                log.debug(f"  ✗ Keine gültigen Daten für {ticker}")
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

    def get_historical_returns(self, tickers: List[str], days: int = 252) -> Dict[str, np.ndarray]:
        import numpy as np
        returns = {}
        for ticker in tickers:
            df = self.fetch_price_history(ticker, days=days + 10)
            if df is not None and len(df) > days:
                close = df["Close"]
                rets = close.pct_change().dropna().values[-days:]
                returns[ticker] = rets
        return returns

    def _normalize_ohlcv_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                df = df.copy()
                df.columns = [c[0] if isinstance(c, tuple) and c[0] else "_".join([str(x) for x in c]).strip() for c in df.columns]
        except Exception:
            pass
        cols_lower = {c.lower(): c for c in df.columns}
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
                for col in available:
                    if col.lower() == k:
                        found = col
                        break
                if found:
                    break
            if not found:
                for k in keys:
                    for col in available:
                        if k in col.lower():
                            found = col
                            break
                    if found:
                        break
            if found and found != target:
                if target not in df.columns or ("adj" in found.lower() and target == "Close"):
                    rename_map[found] = target
        if rename_map:
            try:
                df = df.rename(columns=rename_map)
            except Exception:
                pass
        return df
