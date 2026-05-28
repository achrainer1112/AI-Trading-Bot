"""
AI Trading Bot - News & Sentiment Collector (Robust)
=====================================================
Sammelt Finanznachrichten aus:
- Yahoo Finance (via yfinance) – oft flaky, daher mit Fallback
- NewsAPI (falls API-Key vorhanden)
- RSS-Feed (z.B. Reuters, als Backup)
- Simulierte News als letzter Fallback
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re
import requests
import feedparser
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

    def fetch_yahoo_news(self, tickers: List[str] = None, max_retries: int = 2) -> List[Dict]:
        """Versucht Yahoo News zu holen, mit Retry bei leerem Ergebnis."""
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
                        break  # erfolgreich
                    else:
                        log.debug(f"Yahoo News für {ticker}: Versuch {attempt+1} -> keine News")
                        time.sleep(0.5)
                except Exception as e:
                    log.debug(f"Yahoo News Fehler {ticker} (Versuch {attempt+1}): {e}")
                    time.sleep(0.5)
        log.info(f"Yahoo Finance: {len(articles)} Artikel gesammelt.")
        return articles

    def fetch_newsapi(self, query: str = "stock market finance") -> List[Dict]:
        """Holt News von NewsAPI.org (benötigt API Key)."""
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

    def fetch_rss_news(self, feed_url: str = "https://feeds.bloomberg.com/markets/news.rss") -> List[Dict]:
        """Holt News aus einem RSS-Feed (z.B. Bloomberg Markets)."""
        articles = []
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                title = entry.get("title", "")
                if not title:
                    continue
                articles.append({
                    "title": title,
                    "summary": entry.get("summary", "")[:300],
                    "source": feed.feed.get("title", "RSS"),
                    "url": entry.get("link", ""),
                    "published": entry.get("published", datetime.now().isoformat()),
                    "tickers": self._extract_tickers_from_text(title),
                    "sentiment": self._estimate_sentiment(title, entry.get("summary", "")),
                })
            log.info(f"RSS ({feed_url}): {len(articles)} Artikel gesammelt.")
        except Exception as e:
            log.warning(f"RSS-Feed Fehler: {e}")
        return articles

    def collect_all(self) -> List[Dict]:
        log.info("Sammle Finanznachrichten...")
        all_articles = []

        # 1. Yahoo News (versuchen, aber nicht kritisch)
        yahoo_news = self.fetch_yahoo_news()
        all_articles.extend(yahoo_news)

        # 2. NewsAPI (falls Key gesetzt)
        if self.news_api_key:
            macro_news = self.fetch_newsapi("stock market federal reserve inflation earnings")
            all_articles.extend(macro_news[:10])

        # 3. RSS-Feed als zusätzliche Quelle
        rss_news = self.fetch_rss_news()
        all_articles.extend(rss_news[:8])

        # 4. Fallback: simulierte News
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

    # Die Hilfsmethoden _deduplicate_articles, _extract_tickers_from_text,
    # _estimate_sentiment, _generate_fallback_news, format_for_ai
    # bleiben identisch zur vorherigen Version (siehe oben).
