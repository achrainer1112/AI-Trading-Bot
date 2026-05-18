"""
AI Trading Bot - News & Sentiment Collector
=============================================
Sammelt aktuelle Finanznachrichten und bereitet sie für die KI-Analyse vor.
Unterstützt: NewsAPI, Yahoo Finance News (kostenlos), RSS-Feeds.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests

from logger import log
from config import NEWS_API_KEY, MAX_NEWS_ARTICLES, NEWS_TOPICS, FULL_WATCHLIST

try:
    import yfinance as yf
except ImportError:
    log.error("yfinance nicht installiert.")


class NewsCollector:
    """
    Sammelt Finanznachrichten aus mehreren Quellen:
    1. Yahoo Finance News (kostenlos, via yfinance)
    2. NewsAPI.org (erfordert API Key)
    3. Fallback: Simulierte News für Tests
    """

    def __init__(self):
        self.news_api_key = NEWS_API_KEY
        self.collected_articles: List[Dict] = []

    def fetch_yahoo_news(self, tickers: List[str] = None) -> List[Dict]:
        """
        Holt News von Yahoo Finance für die wichtigsten Ticker.
        Kostenlos, kein API Key nötig.
        """
        articles = []
        # Nur Top-Ticker für News (zu viele Anfragen vermeiden)
        top_tickers = (tickers or FULL_WATCHLIST)[:10]

        for ticker in top_tickers:
            try:
                t = yf.Ticker(ticker)
                news = t.news
                if not news:
                    continue
                for item in news[:3]:  # Max 3 News pro Ticker
                    articles.append({
                        "title": item.get("title", ""),
                        "summary": item.get("summary", item.get("title", "")),
                        "source": item.get("publisher", "Yahoo Finance"),
                        "url": item.get("link", ""),
                        "published": datetime.fromtimestamp(
                            item.get("providerPublishTime", datetime.now().timestamp())
                        ).isoformat(),
                        "tickers": item.get("relatedTickers", [ticker]),
                    })
            except Exception as e:
                log.debug(f"Yahoo News Fehler für {ticker}: {e}")

        log.info(f"Yahoo Finance: {len(articles)} Artikel gesammelt.")
        return articles

    def fetch_newsapi(self, query: str = "stock market finance") -> List[Dict]:
        """
        Holt aktuelle Nachrichten von NewsAPI.org.
        Benötigt einen API Key (kostenloser Plan: 100 req/Tag).
        API Key auf https://newsapi.org/ holen.
        """
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
                if item.get("title") and item["title"] != "[Removed]":
                    articles.append({
                        "title": item["title"],
                        "summary": item.get("description", ""),
                        "source": item.get("source", {}).get("name", "NewsAPI"),
                        "url": item.get("url", ""),
                        "published": item.get("publishedAt", ""),
                        "tickers": [],
                    })

            log.info(f"NewsAPI: {len(articles)} Artikel gesammelt.")
            return articles

        except Exception as e:
            log.warning(f"NewsAPI Fehler: {e}")
            return []

    def get_macro_news(self) -> List[Dict]:
        """Sammelt makroökonomische News zu definierten Themen."""
        all_articles = []
        for topic in NEWS_TOPICS[:3]:  # Limitiert um API-Limits zu schonen
            articles = self.fetch_newsapi(query=f"{topic} economy market")
            all_articles.extend(articles[:3])
        return all_articles

    def collect_all(self) -> List[Dict]:
        """
        Hauptfunktion: Sammelt News aus allen verfügbaren Quellen.
        Dedupliziert und sortiert nach Relevanz.
        """
        log.info("Sammle Finanznachrichten...")
        all_articles = []

        # 1. Yahoo Finance News (immer verfügbar)
        yahoo_news = self.fetch_yahoo_news()
        all_articles.extend(yahoo_news)

        # 2. NewsAPI (wenn Key vorhanden)
        if self.news_api_key:
            macro_news = self.fetch_newsapi("stock market federal reserve inflation earnings")
            all_articles.extend(macro_news[:10])

        # Deduplizierung via Titel
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title = article.get("title", "").lower()[:60]
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)

        # Limitiere auf MAX_NEWS_ARTICLES
        self.collected_articles = unique_articles[:MAX_NEWS_ARTICLES]
        log.info(f"News gesammelt: {len(self.collected_articles)} einzigartige Artikel.")
        return self.collected_articles

    def format_for_ai(self, articles: List[Dict] = None) -> str:
        """
        Formatiert die News-Artikel als kompakten Text für den KI-Prompt.
        """
        articles = articles or self.collected_articles
        if not articles:
            return "Keine aktuellen Nachrichten verfügbar."

        lines = [f"AKTUELLE FINANZNACHRICHTEN ({datetime.now().strftime('%Y-%m-%d')}):", ""]
        for i, a in enumerate(articles[:15], 1):
            lines.append(f"{i}. [{a.get('source', 'Unbekannt')}] {a.get('title', '')}")
            if a.get("summary") and a["summary"] != a.get("title"):
                # Kürze Summary auf max 150 Zeichen
                summary = a["summary"][:150] + "..." if len(a["summary"]) > 150 else a["summary"]
                lines.append(f"   → {summary}")
            tickers = a.get("tickers", [])
            if tickers:
                lines.append(f"   Ticker: {', '.join(str(t) for t in tickers[:5])}")
            lines.append("")

        return "\n".join(lines)