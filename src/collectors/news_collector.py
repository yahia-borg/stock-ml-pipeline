"""
News collector — ingests financial headlines from RSS feeds and NewsAPI.

- RSS feeds: free, unlimited, rolling (no history)
- NewsAPI: free tier = 100 req/day, 1 month history

Headlines are stored raw. Sentiment scoring happens in Phase 2 (features/sentiment.py).
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone

import feedparser
import pandas as pd
import requests

from src.utils.config import get_data_config, ROOT_DIR
from src.utils.db import check_connection, upsert_dataframe
from src.utils.logger import get_logger

log = get_logger("news_collector")


# ── RSS Feeds ──────────────────────────────────────────────

def fetch_rss_headlines() -> pd.DataFrame:
    """Parse all configured RSS feeds and return a DataFrame of articles."""
    cfg = get_data_config()
    feeds = cfg["news"]["rss_feeds"]
    articles = []

    for source, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                published = entry.get("published", entry.get("updated", ""))
                articles.append({
                    "source": source,
                    "headline": entry.get("title", "").strip(),
                    "summary": (entry.get("summary", "") or "")[:500],
                    "url": entry.get("link", ""),
                    "published": published,
                })
            log.info("  %s: %d articles", source, len(feed.entries))
        except Exception as e:
            log.warning("  RSS feed error %s: %s", source, e)

    if not articles:
        return pd.DataFrame()

    df = pd.DataFrame(articles)
    df["time"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "headline"])
    df = df.drop(columns=["published"])

    # Deduplicate by headline
    df = df.drop_duplicates(subset=["headline"])
    log.info("RSS total: %d unique headlines", len(df))
    return df


# ── NewsAPI ────────────────────────────────────────────────

def fetch_newsapi(
    query: str = "stock market OR earnings OR federal reserve",
    days: int | None = None,
) -> pd.DataFrame:
    """Fetch headlines from NewsAPI.org (requires NEWSAPI_KEY in .env)."""
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key or api_key == "your_newsapi_key_here":
        log.warning("NEWSAPI_KEY not set — skipping NewsAPI. Get one at https://newsapi.org/register")
        return pd.DataFrame()

    cfg = get_data_config()
    days = days or cfg["news"]["newsapi"]["history_days"]

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d"),
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": api_key,
        "pageSize": 100,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            log.error("NewsAPI error: %s", data.get("message", "unknown"))
            return pd.DataFrame()

        articles = data.get("articles", [])
        if not articles:
            log.info("NewsAPI: 0 articles returned")
            return pd.DataFrame()

        rows = []
        for a in articles:
            source_name = a.get("source", {}).get("name", "newsapi")
            rows.append({
                "source": f"newsapi_{source_name}",
                "headline": (a.get("title") or "").strip(),
                "summary": (a.get("description") or "")[:500],
                "url": a.get("url", ""),
                "time": a.get("publishedAt"),
            })

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time", "headline"])
        df = df[df["headline"] != "[Removed]"]
        log.info("NewsAPI: %d articles", len(df))
        return df

    except requests.RequestException as e:
        log.error("NewsAPI request failed: %s", e)
        return pd.DataFrame()


# ── Combine & Store ────────────────────────────────────────

def collect_all_news() -> pd.DataFrame:
    """Fetch from all news sources and combine."""
    dfs = []

    log.info("Fetching RSS feeds...")
    rss = fetch_rss_headlines()
    if not rss.empty:
        dfs.append(rss)

    log.info("Fetching NewsAPI...")
    newsapi = fetch_newsapi()
    if not newsapi.empty:
        dfs.append(newsapi)

    if not dfs:
        log.warning("No news collected from any source")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["headline"])
    combined = combined.sort_values("time", ascending=False)

    log.info("Total collected: %d unique headlines", len(combined))
    return combined


def save_parquet(df: pd.DataFrame) -> None:
    """Save news to parquet with date-based filename."""
    out_dir = ROOT_DIR / get_data_config()["storage"]["raw_dir"] / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = out_dir / f"news_{today}.parquet"
    df.to_parquet(path, index=False)
    log.info("Saved %s", path.relative_to(ROOT_DIR))


def upsert_to_db(df: pd.DataFrame) -> int:
    """Insert news articles into TimescaleDB (no upsert — articles are append-only)."""
    if df.empty:
        return 0

    db_df = df[["time", "source", "headline", "summary", "url"]].copy()
    # Sentiment columns filled later by features/sentiment.py
    db_df["sentiment_finbert"] = None
    db_df["sentiment_llm"] = None
    db_df["tickers"] = None

    from sqlalchemy import text
    from src.utils.db import get_engine

    inserted = 0
    with get_engine().connect() as conn:
        for _, row in db_df.iterrows():
            # Skip if headline already exists (dedup)
            exists = conn.execute(
                text("SELECT 1 FROM news_articles WHERE headline = :headline LIMIT 1"),
                {"headline": row["headline"]},
            ).fetchone()
            if exists:
                continue

            conn.execute(
                text("""
                    INSERT INTO news_articles (time, source, headline, summary, url,
                                               sentiment_finbert, sentiment_llm, tickers)
                    VALUES (:time, :source, :headline, :summary, :url,
                            :sentiment_finbert, :sentiment_llm, :tickers)
                """),
                dict(row),
            )
            inserted += 1
        conn.commit()

    log.info("Inserted %d new articles into DB", inserted)
    return inserted


def main():
    parser = argparse.ArgumentParser(description="News headline collector")
    parser.add_argument("--source", choices=["rss", "newsapi", "all"], default="all")
    args = parser.parse_args()

    if args.source == "rss":
        df = fetch_rss_headlines()
    elif args.source == "newsapi":
        df = fetch_newsapi()
    else:
        df = collect_all_news()

    if df.empty:
        return

    save_parquet(df)

    if check_connection():
        upsert_to_db(df)


if __name__ == "__main__":
    main()
