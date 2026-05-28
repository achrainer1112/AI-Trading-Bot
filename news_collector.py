"""
AI Trading Bot - News & Sentiment Collector (Robust, ohne RSS)
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re
import requests
import time

from logger import log
from config import NEWS_API_KEY, MAX_NEWS_ARTICLES, NEWS_TOPICS, FULL_WATCHLIST

try:
    import yfinance as yf
except ImportError:
    log.error("yfinance nicht installiert.")


class NewsCollector:
    def __init__(self):
        self.news_api_key = NEWS_API_KEY
        self.collected_articles: List[Dict] = []

    # ------------------------------------------------------------
    # Yahoo Finance News (mit Retry)
    # ------------------------------------------------------------
    def fetch_yahoo_news(self, tickers: List[str] = None, max_retries: int = 2) -> List[Dict]:
        top_tickers = (tickers or FULL_WATCHLIST)[:12]
        articles = []

        for ticker in top_tickers:
            for attempt in range(max_retries):
                try:
                    t = yf.Ticker(ticker)
                    news = t.news
                    if news and len(news) > 0:
                        for item in news[:4]:
                            title = item.get("title", "")
                            if title:
                                articles.append({
                                    "title": title,
                                    "summary": item.get("summary", "") or title,
                                    "source": item.get("publisher", "Yahoo Finance"),
                                    "url": item.get("link", ""),
                                    "published": datetime.fromtimestamp(
                                        item.get("providerPublishTime", datetime.now().timestamp())
                                    ).isoformat(),
                                    "tickers": item.get("relatedTickers", [ticker]),
                                    "sentiment": self._estimate_sentiment(title, item.get("summary", "")),
                                })
                        break
                    else:
                        log.debug(f"Yahoo News für {ticker}: Versuch {attempt+1} -> keine News")
                        time.sleep(0.5)
                except Exception as e:
                    log.debug(f"Yahoo News Fehler {ticker} (Versuch {attempt+1}): {e}")
                    time.sleep(0.5)
        log.info(f"Yahoo Finance: {len(articles)} Artikel gesammelt.")
        return articles

    # ------------------------------------------------------------
    # NewsAPI (benötigt API-Key)
    # ------------------------------------------------------------
    def fetch_newsapi(self, query: str = "stock market finance") -> List[Dict]:
        if not self.news_api_key:
            log.debug("Kein NewsAPI Key konfiguriert, überspringe.")
            return []

        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(MAX_NEWS_ARTICLES, 20),
                "from": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
                "apiKey": self.news_api_key,
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            articles = []
            for item in data.get("articles", []):
                title = item.get("title")
                if not title or title == "[Removed]":
                    continue
                articles.append({
                    "title": title,
                    "summary": item.get("description", "") or "",
                    "source": item.get("source", {}).get("name", "NewsAPI"),
                    "url": item.get("url", ""),
                    "published": item.get("publishedAt", ""),
                    "tickers": self._extract_tickers_from_text(title + " " + (item.get("description") or "")),
                    "sentiment": self._estimate_sentiment(title, item.get("description") or ""),
                })
            log.info(f"NewsAPI: {len(articles)} Artikel gesammelt.")
            return articles
        except Exception as e:
            log.warning(f"NewsAPI Fehler: {e}")
            return []

    # ------------------------------------------------------------
    # Hauptsammlung
    # ------------------------------------------------------------
    def collect_all(self) -> List[Dict]:
        log.info("Sammle Finanznachrichten...")
        all_articles = []

        # 1. Yahoo News
        yahoo_news = self.fetch_yahoo_news()
        all_articles.extend(yahoo_news)

        # 2. NewsAPI (falls Key gesetzt)
        if self.news_api_key:
            macro_news = self.fetch_newsapi("stock market federal reserve inflation earnings")
            all_articles.extend(macro_news[:10])

        # 3. Fallback: simulierte News
        if not all_articles:
            log.warning("Keine News-Quellen verfügbar – verwende simulierte Marktkommentare.")
            all_articles = self._generate_fallback_news()

        # Deduplizierung
        unique_articles = self._deduplicate_articles(all_articles)

        # Nach Relevanz sortieren (Ticker-Erwähnungen zuerst)
        unique_articles.sort(key=lambda a: (len(a.get("tickers", [])) > 0, a.get("published", "")), reverse=True)

        self.collected_articles = unique_articles[:MAX_NEWS_ARTICLES]
        log.info(f"News gesammelt: {len(self.collected_articles)} einzigartige Artikel.")
        return self.collected_articles

    # ------------------------------------------------------------
    # Hilfsmethoden (alle vorhanden)
    # ------------------------------------------------------------
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        seen = {}
        unique = []
        for a in articles:
            title = a.get("title", "")
            normalized = re.sub(r'[^\w\s]', '', title.lower().strip())[:80]
            if not normalized:
                continue
            if normalized in seen:
                existing = seen[normalized]
                if len(a.get("summary", "")) > len(existing.get("summary", "")):
                    seen[normalized] = a
                continue
            seen[normalized] = a
            unique.append(a)
        return unique

    def _extract_tickers_from_text(self, text: str) -> List[str]:
        if not text:
            return []
        words = re.findall(r'\b[A-Z]{2,5}\b', text)
        tickers = [w for w in words if w in FULL_WATCHLIST]
        return list(set(tickers))[:3]

    def _estimate_sentiment(self, title: str, summary: str) -> float:
        text = (title + " " + summary).lower()
        positive_words = ["surge", "rally", "gain", "up", "positive", "bullish", "growth", "profit", "beat", "upgrade"]
        negative_words = ["drop", "fall", "down", "negative", "bearish", "loss", "miss", "downgrade", "crash", "selloff"]
        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total

    def _generate_fallback_news(self) -> List[Dict]:
        return [{
            "title": "Marktupdate: Keine aktuellen Nachrichten verfügbar",
            "summary": "Der Bot konnte keine Nachrichtenquellen erreichen. Handelsentscheidungen basieren nur auf Quant-Daten.",
            "source": "System",
            "published": datetime.now().isoformat(),
            "tickers": [],
            "sentiment": 0.0,
        }]

    def format_for_ai(self, articles: List[Dict] = None) -> str:
        articles = articles or self.collected_articles
        if not articles:
            return "Keine aktuellen Nachrichten verfügbar."
        lines = [f"AKTUELLE FINANZNACHRICHTEN ({datetime.now().strftime('%Y-%m-%d')}):", ""]
        for i, a in enumerate(articles[:12], 1):
            title = a.get("title", "")
            source = a.get("source", "Unbekannt")
            sentiment = a.get("sentiment")
            sent_str = f" [Sentiment: {sentiment:+.2f}]" if sentiment is not None else ""
            lines.append(f"{i}. [{source}]{sent_str} {title}")
            summary = a.get("summary", "")
            if summary and summary != title:
                summary_short = summary[:150] + "..." if len(summary) > 150 else summary
                lines.append(f"   → {summary_short}")
            tickers = a.get("tickers", [])
            if tickers:
                lines.append(f"   Betroffene Ticker: {', '.join(tickers[:5])}")
            lines.append("")
        return "\n".join(lines)

    def get_sentiment_score(self) -> float:
        if not self.collected_articles:
            return 0.0
        scores = [a.get("sentiment", 0.0) for a in self.collected_articles if a.get("sentiment") is not None]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
