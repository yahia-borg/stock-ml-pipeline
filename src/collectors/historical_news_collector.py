"""
Historical news collector — downloads FNSPID dataset from HuggingFace.

FNSPID (Financial News and Stock Price Integration Dataset):
  - 15.7M news records, 1999-2023
  - Pre-aligned to S&P 500 tickers
  - Includes headlines, article text, and sentiment
  - Published at KDD 2024

This fills the 2000-2015 gap where live news sources (RSS, NewsAPI) have no coverage.
Run once during initial backfill.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.utils.config import ROOT_DIR, get_data_config
from src.utils.db import check_connection
from src.utils.logger import get_logger

log = get_logger("historical_news")


def download_fnspid(max_rows: int | None = None) -> pd.DataFrame:
    """Download FNSPID dataset from HuggingFace.

    This is a large dataset (~15M rows). On first run it will download
    and cache locally via HuggingFace datasets library.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error(
            "HuggingFace datasets library required. Install with: "
            "pip install datasets"
        )
        return pd.DataFrame()

    cfg = get_data_config()
    dataset_name = cfg["news"]["historical_news"]["dataset"]
    log.info("Downloading %s from HuggingFace (this may take a while on first run)...",
             dataset_name)

    try:
        ds = load_dataset(dataset_name, split="train")
        df = ds.to_pandas()

        if max_rows:
            df = df.head(max_rows)

        log.info("Downloaded %d rows from FNSPID", len(df))
        return df

    except Exception as e:
        log.error("Failed to download FNSPID: %s", e)
        return pd.DataFrame()


def process_fnspid(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize FNSPID columns to match our news schema."""
    if df.empty:
        return df

    # FNSPID typical columns: Date, Title, Content, Stock, Sentiment, ...
    # Column names may vary — handle common variants
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ("date", "timestamp", "published_date"):
            col_map[col] = "time"
        elif lower in ("title", "headline"):
            col_map[col] = "headline"
        elif lower in ("content", "article", "text", "summary"):
            col_map[col] = "summary"
        elif lower in ("stock", "ticker", "symbol"):
            col_map[col] = "ticker"
        elif lower in ("sentiment", "sentiment_score"):
            col_map[col] = "sentiment_score"
        elif lower in ("source", "publisher"):
            col_map[col] = "source"
        elif lower in ("url", "link"):
            col_map[col] = "url"

    df = df.rename(columns=col_map)

    # Ensure required columns
    if "time" not in df.columns:
        log.error("No date column found in FNSPID data")
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])

    if "headline" not in df.columns:
        df["headline"] = df.get("summary", "").str[:200]

    if "source" not in df.columns:
        df["source"] = "fnspid_historical"

    if "summary" not in df.columns:
        df["summary"] = ""

    if "url" not in df.columns:
        df["url"] = ""

    df = df.drop_duplicates(subset=["headline"])

    log.info("Processed FNSPID: %d rows, date range %s to %s",
             len(df), df["time"].min().date(), df["time"].max().date())
    return df


def save_parquet(df: pd.DataFrame) -> None:
    """Save historical news in yearly parquet files."""
    out_dir = ROOT_DIR / get_data_config()["storage"]["raw_dir"] / "news"
    out_dir.mkdir(parents=True, exist_ok=True)

    df["year"] = df["time"].dt.year
    for year, group in df.groupby("year"):
        path = out_dir / f"historical_{year}.parquet"
        # Keep only our standard columns
        cols = ["time", "source", "headline", "summary", "url"]
        extra = [c for c in ["ticker", "sentiment_score"] if c in group.columns]
        group[cols + extra].to_parquet(path, index=False)
        log.info("  Saved %s (%d rows)", path.name, len(group))


def upsert_to_db(df: pd.DataFrame) -> int:
    """Insert historical news into TimescaleDB."""
    if df.empty:
        return 0

    from sqlalchemy import text
    from src.utils.db import get_engine

    db_df = df[["time", "source", "headline"]].copy()
    db_df["summary"] = df.get("summary", "").fillna("")
    db_df["url"] = df.get("url", "").fillna("")
    db_df["sentiment_finbert"] = df.get("sentiment_score")  # FNSPID pre-computed
    db_df["sentiment_llm"] = None
    # Convert ticker to array format
    if "ticker" in df.columns:
        db_df["tickers"] = df["ticker"].apply(
            lambda x: [x] if pd.notna(x) and x else None
        )
    else:
        db_df["tickers"] = None

    inserted = 0
    batch_size = 1000
    with get_engine().connect() as conn:
        for i in range(0, len(db_df), batch_size):
            batch = db_df.iloc[i:i + batch_size]
            for _, row in batch.iterrows():
                try:
                    conn.execute(
                        text("""
                            INSERT INTO news_articles
                                (time, source, headline, summary, url,
                                 sentiment_finbert, sentiment_llm, tickers)
                            VALUES (:time, :source, :headline, :summary, :url,
                                    :sentiment_finbert, :sentiment_llm, :tickers)
                            ON CONFLICT DO NOTHING
                        """),
                        dict(row),
                    )
                    inserted += 1
                except Exception:
                    continue
            conn.commit()
            if (i + batch_size) % 10000 == 0:
                log.info("  Progress: %d / %d rows", min(i + batch_size, len(db_df)), len(db_df))

    log.info("Inserted %d historical news articles into DB", inserted)
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Historical news backfill from FNSPID")
    parser.add_argument("--max-rows", type=int,
                        help="Limit rows (for testing, e.g. --max-rows 10000)")
    parser.add_argument("--skip-db", action="store_true",
                        help="Only save parquet, don't insert into DB")
    args = parser.parse_args()

    cfg = get_data_config()
    if not cfg["news"]["historical_news"].get("use_for_backfill", True):
        log.info("Historical news backfill disabled in config")
        return

    df = download_fnspid(max_rows=args.max_rows)
    if df.empty:
        return

    df = process_fnspid(df)
    if df.empty:
        return

    save_parquet(df)

    if not args.skip_db and check_connection():
        upsert_to_db(df)


if __name__ == "__main__":
    main()
