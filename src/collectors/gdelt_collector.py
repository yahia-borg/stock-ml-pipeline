"""
GDELT collector — fetches global events with tone scoring.

Uses the GDELT 2.0 DOC API:
  - Completely free, no API key required
  - Covers 100+ languages, 250+ countries
  - Tone field: positive = bullish, negative = bearish
  - Rate limit: be reasonable (~1 req/sec)
"""

from __future__ import annotations

import argparse
import time as time_mod
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from src.utils.config import ROOT_DIR, get_data_config, get_tickers_config
from src.utils.db import check_connection
from src.utils.logger import get_logger

log = get_logger("gdelt_collector")

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_gdelt_events(
    keywords: list[str],
    days: int | None = None,
    max_records: int | None = None,
) -> pd.DataFrame:
    """Fetch articles matching keywords from GDELT DOC API.

    Args:
        keywords: Search terms (OR-joined). Use quotes for phrases.
        days: Lookback window in days.
        max_records: Max articles to return per query.
    """
    cfg = get_data_config()
    days = days or cfg["news"]["gdelt"]["lookback_days"]
    max_records = max_records or cfg["news"]["gdelt"]["max_records"]

    query = "(" + " OR ".join(f'"{k}"' for k in keywords) + ")"
    start_dt = datetime.now(timezone.utc) - timedelta(days=days)

    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max_records,
        "format": "json",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
    }

    # Retry with backoff for rate limiting
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(GDELT_DOC_URL, params=params, timeout=60)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                log.warning("GDELT rate limited (429), waiting %ds...", wait)
                time_mod.sleep(wait)
                continue
            resp.raise_for_status()
            # Check if response is actually JSON (GDELT sometimes returns HTML)
            content_type = resp.headers.get("content-type", "")
            if "json" not in content_type and not resp.text.strip().startswith("{"):
                log.warning("GDELT returned non-JSON (content-type: %s), skipping", content_type)
                return pd.DataFrame()
            data = resp.json()
            break
        except requests.RequestException as e:
            log.error("GDELT request failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time_mod.sleep(5 * (attempt + 1))
            continue
        except (ValueError, KeyError) as e:
            log.error("GDELT JSON parse error: %s", e)
            return pd.DataFrame()
    else:
        log.error("GDELT failed after %d retries", max_retries)
        return pd.DataFrame()

    articles = data.get("articles", [])
    if not articles:
        log.info("GDELT: 0 articles for query: %s", query[:80])
        return pd.DataFrame()

    df = pd.DataFrame(articles)

    # Parse datetime
    if "seendate" in df.columns:
        df["time"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ",
                                    utc=True, errors="coerce")
    else:
        df["time"] = pd.Timestamp.now(tz="UTC")

    # Parse tone (GDELT returns as string, e.g. "-2.34,1.2,3.5,...")
    # First value is the average tone
    if "tone" in df.columns:
        df["tone"] = df["tone"].apply(_parse_tone)
    else:
        df["tone"] = 0.0

    result = df[["time", "title", "url", "tone", "domain"]].copy()
    result = result.dropna(subset=["time", "title"])
    result = result.drop_duplicates(subset=["title"])

    log.info("GDELT: %d articles fetched", len(result))
    return result


def _parse_tone(tone_str) -> float:
    """Extract average tone from GDELT tone string."""
    try:
        if isinstance(tone_str, (int, float)):
            return float(tone_str)
        return float(str(tone_str).split(",")[0])
    except (ValueError, IndexError):
        return 0.0


def fetch_all_gdelt() -> pd.DataFrame:
    """Fetch GDELT events for all configured keyword groups."""
    keywords = get_tickers_config().get("gdelt_keywords", [])
    if not keywords:
        log.warning("No GDELT keywords configured in tickers.yaml")
        return pd.DataFrame()

    # Batch keywords — keep small to avoid long queries + rate limits
    batch_size = 3
    all_dfs = []

    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i + batch_size]
        log.info("Fetching GDELT batch %d/%d: %s",
                 i // batch_size + 1,
                 (len(keywords) + batch_size - 1) // batch_size,
                 [k[:30] for k in batch])

        df = fetch_gdelt_events(batch)
        if not df.empty:
            df["keywords"] = [batch] * len(df)
            all_dfs.append(df)

        # GDELT free API needs 5+ seconds between requests to avoid 429
        time_mod.sleep(5)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["title"])
    combined = combined.sort_values("time", ascending=False)

    log.info("GDELT total: %d unique articles", len(combined))
    return combined


def save_parquet(df: pd.DataFrame) -> None:
    """Save GDELT events to parquet."""
    out_dir = ROOT_DIR / get_data_config()["storage"]["raw_dir"] / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = out_dir / f"gdelt_{today}.parquet"

    # Convert list columns to string for parquet compatibility
    save_df = df.copy()
    if "keywords" in save_df.columns:
        save_df["keywords"] = save_df["keywords"].apply(
            lambda x: "|".join(x) if isinstance(x, list) else str(x)
        )

    save_df.to_parquet(path, index=False)
    log.info("Saved %s", path.relative_to(ROOT_DIR))


def upsert_to_db(df: pd.DataFrame) -> int:
    """Insert GDELT events into TimescaleDB."""
    if df.empty:
        return 0

    from sqlalchemy import text
    from src.utils.db import get_engine

    inserted = 0
    with get_engine().connect() as conn:
        for _, row in df.iterrows():
            # Dedup by title
            exists = conn.execute(
                text("SELECT 1 FROM gdelt_events WHERE title = :title LIMIT 1"),
                {"title": row["title"]},
            ).fetchone()
            if exists:
                continue

            kw = row.get("keywords", [])
            kw_array = kw if isinstance(kw, list) else []

            conn.execute(
                text("""
                    INSERT INTO gdelt_events (time, title, url, tone, domain, keywords)
                    VALUES (:time, :title, :url, :tone, :domain, :keywords)
                """),
                {
                    "time": row["time"],
                    "title": row["title"],
                    "url": row.get("url", ""),
                    "tone": row.get("tone", 0.0),
                    "domain": row.get("domain", ""),
                    "keywords": kw_array,
                },
            )
            inserted += 1
        conn.commit()

    log.info("Inserted %d new GDELT events into DB", inserted)
    return inserted


def main():
    parser = argparse.ArgumentParser(description="GDELT global events collector")
    parser.add_argument("--days", type=int, help="Override lookback days")
    parser.add_argument("--max-records", type=int, help="Override max records per query")
    args = parser.parse_args()

    df = fetch_all_gdelt()
    if df.empty:
        return

    save_parquet(df)

    if check_connection():
        upsert_to_db(df)


if __name__ == "__main__":
    main()
