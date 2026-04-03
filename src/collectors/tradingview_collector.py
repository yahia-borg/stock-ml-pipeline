"""
TradingView news collector — fetches market news and analysis headlines.

TradingView doesn't have a public API, but we can:
  1. Scrape their news headlines page (free, no API key)
  2. Use their Ideas RSS feed for community analysis
  3. Fetch ticker-specific news via their widget endpoint

Rate limit: be polite, max 1 request per 5 seconds.
"""

from __future__ import annotations

import argparse
import time as time_mod
from datetime import datetime, timezone

import pandas as pd
import requests

from src.utils.config import ROOT_DIR, get_data_config
from src.utils.logger import get_logger

log = get_logger("collectors.tradingview")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}


def fetch_tradingview_news(category: str = "stock", limit: int = 50) -> pd.DataFrame:
    """Fetch latest news from TradingView's news endpoint.

    Categories: stock, forex, crypto, commodities, bonds, economy
    """
    url = f"https://news-headlines.tradingview.com/v2/view/headlines/en"
    params = {
        "client": "web",
        "lang": "en",
        "category": category,
        "streaming": "false",
    }

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        log.warning("TradingView news fetch failed: %s", e)
        return pd.DataFrame()
    except ValueError:
        log.warning("TradingView returned non-JSON response")
        return pd.DataFrame()

    stories = data.get("items", data.get("stories", []))
    if not stories:
        log.info("TradingView: 0 headlines for category=%s", category)
        return pd.DataFrame()

    articles = []
    for story in stories[:limit]:
        title = story.get("title", "")
        if not title:
            continue

        published = story.get("published", story.get("astDescription", {}).get("publishedDateText", ""))

        articles.append({
            "source": f"tradingview_{category}",
            "headline": title,
            "summary": story.get("shortDescription", story.get("astDescription", {}).get("text", ""))[:500],
            "url": story.get("storyPath", story.get("link", "")),
            "time": published,
        })

    df = pd.DataFrame(articles)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time", "headline"])
        df = df.drop_duplicates(subset=["headline"])

    log.info("TradingView %s: %d headlines", category, len(df))
    return df


def fetch_tradingview_ideas(symbol: str = "AAPL") -> pd.DataFrame:
    """Fetch TradingView community ideas/analysis for a symbol via RSS."""
    import feedparser

    url = f"https://www.tradingview.com/feed/?symbol={symbol}"

    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries:
            articles.append({
                "source": f"tradingview_ideas_{symbol}",
                "headline": entry.get("title", "").strip(),
                "summary": (entry.get("summary", "") or "")[:500],
                "url": entry.get("link", ""),
                "time": entry.get("published", ""),
            })

        if not articles:
            return pd.DataFrame()

        df = pd.DataFrame(articles)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time", "headline"])
        log.info("TradingView ideas for %s: %d entries", symbol, len(df))
        return df

    except Exception as e:
        log.warning("TradingView ideas fetch failed for %s: %s", symbol, e)
        return pd.DataFrame()


def fetch_all_tradingview() -> pd.DataFrame:
    """Fetch news from multiple TradingView categories."""
    categories = ["stock", "forex", "commodities", "economy"]
    all_dfs = []

    for cat in categories:
        df = fetch_tradingview_news(category=cat)
        if not df.empty:
            all_dfs.append(df)
        time_mod.sleep(3)  # polite rate limiting

    # Also fetch ideas for key tickers
    key_tickers = ["SPY", "AAPL", "NVDA", "GOLD", "OIL"]
    for ticker in key_tickers:
        df = fetch_tradingview_ideas(ticker)
        if not df.empty:
            all_dfs.append(df)
        time_mod.sleep(2)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["headline"])
    log.info("TradingView total: %d unique headlines", len(combined))
    return combined


def main():
    parser = argparse.ArgumentParser(description="TradingView news collector")
    parser.add_argument("--category", default="stock", help="News category")
    args = parser.parse_args()

    df = fetch_all_tradingview()
    if df.empty:
        print("No TradingView data collected")
        return

    # Save
    cfg = get_data_config()
    out_dir = ROOT_DIR / cfg["storage"]["raw_dir"] / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = out_dir / f"tradingview_{today}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} TradingView headlines to {path}")


if __name__ == "__main__":
    main()
