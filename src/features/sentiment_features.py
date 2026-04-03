"""
Sentiment feature aggregation — transforms per-article scores into daily model features.

Builds features from three sentiment sources:
  1. FinBERT scores (fast, per-headline)
  2. LLM scores (rich, per-headline, optional)
  3. GDELT tone (per-event, pre-computed)

Aggregation strategy:
  - Exponential decay weighting (recent headlines weighted more)
  - Multiple statistics: mean, std, min, max, count
  - Disagreement features (FinBERT vs LLM divergence = uncertainty)
  - Sentiment momentum (change vs rolling average)
  - Per-ticker and market-wide variants
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import ROOT_DIR, get_data_config
from src.utils.logger import get_logger

log = get_logger("features.sentiment_features")


def load_scored_news(source: str = "db") -> pd.DataFrame:
    """Load news articles with sentiment scores.

    Args:
        source: "db" to load from TimescaleDB, or "parquet" for local files
    """
    if source == "db":
        from src.utils.db import read_sql, check_connection
        if check_connection():
            df = read_sql("""
                SELECT time, headline, source, sentiment_finbert, sentiment_llm, tickers
                FROM news_articles
                WHERE sentiment_finbert IS NOT NULL
                ORDER BY time
            """)
            if not df.empty:
                df["time"] = pd.to_datetime(df["time"])
                log.info("Loaded %d scored articles from DB", len(df))
                return df

    # Fallback: parquet files
    cfg = get_data_config()
    news_dir = ROOT_DIR / cfg["storage"]["raw_dir"] / "news"
    parquet_files = sorted(news_dir.glob("*.parquet"))

    if not parquet_files:
        log.warning("No news data found")
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in parquet_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined["time"] = pd.to_datetime(combined["time"])

    log.info("Loaded %d articles from %d parquet files", len(combined), len(parquet_files))
    return combined


def load_gdelt_events(source: str = "db") -> pd.DataFrame:
    """Load GDELT events with tone scores."""
    if source == "db":
        from src.utils.db import read_sql, check_connection
        if check_connection():
            df = read_sql("""
                SELECT time, title, tone, domain
                FROM gdelt_events
                ORDER BY time
            """)
            if not df.empty:
                df["time"] = pd.to_datetime(df["time"])
                log.info("Loaded %d GDELT events from DB", len(df))
                return df

    # Fallback: parquet
    cfg = get_data_config()
    news_dir = ROOT_DIR / cfg["storage"]["raw_dir"] / "news"
    gdelt_files = sorted(news_dir.glob("gdelt_*.parquet"))

    if not gdelt_files:
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in gdelt_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined["time"] = pd.to_datetime(combined["time"])
    return combined


def _decay_weights(hours_ago: pd.Series, halflife_hours: float = 24) -> pd.Series:
    """Exponential decay weights — recent headlines count more."""
    decay_rate = np.log(2) / halflife_hours
    return np.exp(-decay_rate * hours_ago.clip(lower=0))


def aggregate_daily_sentiment(
    news_df: pd.DataFrame,
    score_col: str = "sentiment_finbert",
    halflife_hours: float | None = None,
) -> pd.DataFrame:
    """Aggregate per-article sentiment scores to daily features.

    Returns a DataFrame indexed by date with columns:
      - sent_{prefix}_mean: weighted mean sentiment
      - sent_{prefix}_std: standard deviation (disagreement among articles)
      - sent_{prefix}_min: most bearish headline
      - sent_{prefix}_max: most bullish headline
      - sent_{prefix}_count: number of scored articles
      - sent_{prefix}_pos_ratio: fraction of positive articles
      - sent_{prefix}_neg_ratio: fraction of negative articles
    """
    cfg = get_data_config()
    halflife = halflife_hours or cfg["features"]["sentiment"]["decay_halflife_hours"]

    if news_df.empty or score_col not in news_df.columns:
        return pd.DataFrame()

    df = news_df.dropna(subset=[score_col]).copy()
    if df.empty:
        return pd.DataFrame()

    df["date"] = df["time"].dt.date

    # Compute decay weights within each day
    df["hours_in_day"] = (
        df.groupby("date")["time"].transform("max") - df["time"]
    ).dt.total_seconds() / 3600
    df["weight"] = _decay_weights(df["hours_in_day"], halflife)

    prefix = score_col.replace("sentiment_", "sent_")

    daily = df.groupby("date").apply(
        lambda g: pd.Series({
            f"{prefix}_mean": np.average(g[score_col], weights=g["weight"]),
            f"{prefix}_std": g[score_col].std(),
            f"{prefix}_min": g[score_col].min(),
            f"{prefix}_max": g[score_col].max(),
            f"{prefix}_count": len(g),
            f"{prefix}_pos_ratio": (g[score_col] > 0.1).mean(),
            f"{prefix}_neg_ratio": (g[score_col] < -0.1).mean(),
        })
    )

    daily.index = pd.to_datetime(daily.index)
    log.info("Daily %s features: %d days", prefix, len(daily))
    return daily


def aggregate_gdelt_daily(gdelt_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate GDELT events to daily features."""
    if gdelt_df.empty or "tone" not in gdelt_df.columns:
        return pd.DataFrame()

    df = gdelt_df.copy()
    df["date"] = df["time"].dt.date

    daily = df.groupby("date").agg(
        gdelt_tone_mean=("tone", "mean"),
        gdelt_tone_std=("tone", "std"),
        gdelt_tone_min=("tone", "min"),
        gdelt_tone_max=("tone", "max"),
        gdelt_event_count=("tone", "count"),
    )

    # Tone dispersion — high dispersion = uncertainty = higher vol
    daily["gdelt_tone_dispersion"] = daily["gdelt_tone_max"] - daily["gdelt_tone_min"]

    daily.index = pd.to_datetime(daily.index)
    log.info("Daily GDELT features: %d days", len(daily))
    return daily


def build_sentiment_features() -> pd.DataFrame:
    """Build the complete daily sentiment feature matrix.

    Combines FinBERT, LLM (if available), and GDELT into one DataFrame.
    Adds derived features like momentum and FinBERT-LLM disagreement.
    """
    log.info("Building sentiment feature matrix ...")

    all_features = []

    # ── FinBERT features ──
    news = load_scored_news()
    if not news.empty and "sentiment_finbert" in news.columns:
        finbert_daily = aggregate_daily_sentiment(news, "sentiment_finbert")
        if not finbert_daily.empty:
            all_features.append(finbert_daily)

    # ── LLM features (if available) ──
    if not news.empty and "sentiment_llm" in news.columns:
        llm_daily = aggregate_daily_sentiment(news, "sentiment_llm")
        if not llm_daily.empty:
            all_features.append(llm_daily)

    # ── GDELT features ──
    gdelt = load_gdelt_events()
    if not gdelt.empty:
        gdelt_daily = aggregate_gdelt_daily(gdelt)
        if not gdelt_daily.empty:
            all_features.append(gdelt_daily)

    if not all_features:
        log.warning("No sentiment data available")
        return pd.DataFrame()

    # Combine all sources
    combined = pd.concat(all_features, axis=1)

    # ── Derived features ──

    # Sentiment momentum (5d change)
    if "sent_finbert_mean" in combined.columns:
        combined["sent_momentum_5d"] = combined["sent_finbert_mean"].diff(5)
        combined["sent_momentum_20d"] = combined["sent_finbert_mean"].diff(20)

        # Sentiment vs its rolling average (surprise)
        rolling_mean = combined["sent_finbert_mean"].rolling(20).mean()
        rolling_std = combined["sent_finbert_mean"].rolling(20).std()
        combined["sent_surprise"] = (
            (combined["sent_finbert_mean"] - rolling_mean) / rolling_std.replace(0, np.nan)
        )

    # FinBERT-LLM disagreement (uncertainty signal)
    if "sent_finbert_mean" in combined.columns and "sent_llm_mean" in combined.columns:
        combined["sent_disagreement"] = abs(
            combined["sent_finbert_mean"] - combined["sent_llm_mean"]
        )
        # When models disagree, average confidence should be lower
        combined["sent_ensemble_mean"] = (
            combined["sent_finbert_mean"] + combined["sent_llm_mean"]
        ) / 2

    # News volume anomaly (spike in article count = potential market mover)
    if "sent_finbert_count" in combined.columns:
        count = combined["sent_finbert_count"]
        count_mean = count.rolling(20).mean()
        count_std = count.rolling(20).std()
        combined["news_volume_zscore"] = (count - count_mean) / count_std.replace(0, np.nan)

    # GDELT-FinBERT alignment (both bearish = stronger signal)
    if "sent_finbert_mean" in combined.columns and "gdelt_tone_mean" in combined.columns:
        # Normalize GDELT tone to [-1, 1] range (it's typically [-10, 10])
        gdelt_normalized = combined["gdelt_tone_mean"] / 10
        combined["sent_gdelt_alignment"] = (
            np.sign(combined["sent_finbert_mean"]) == np.sign(gdelt_normalized)
        ).astype(int)

    log.info("Sentiment feature matrix: %d days, %d features",
             len(combined), len(combined.columns))
    return combined


def merge_sentiment_to_prices(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join daily sentiment features to a price DataFrame (forward-fill)."""
    if sentiment_df is None:
        sentiment_df = build_sentiment_features()

    if sentiment_df.empty:
        log.warning("No sentiment features to merge")
        return price_df

    price_df = price_df.copy()

    # Align to price index — strip timezone from both sides to avoid mismatch
    if "time" in price_df.columns:
        price_idx = pd.to_datetime(price_df["time"])
    else:
        price_idx = pd.to_datetime(price_df.index)

    if price_idx.tz is not None:
        price_idx = price_idx.tz_convert(None)

    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    if sentiment_df.index.tz is not None:
        sentiment_df.index = sentiment_df.index.tz_convert(None)

    sent_reindexed = sentiment_df.reindex(price_idx, method="ffill")

    for col in sent_reindexed.columns:
        price_df[col] = sent_reindexed[col].values

    log.info("Merged %d sentiment features into price data", len(sentiment_df.columns))
    return price_df


def main():
    """Build and save sentiment features."""
    features = build_sentiment_features()
    if features.empty:
        print("No sentiment data available. Run collectors first:")
        print("  make collect-news")
        print("  make collect-gdelt")
        print("  python -m src.features.sentiment --score-new")
        return

    # Save
    cfg = get_data_config()
    out_path = ROOT_DIR / cfg["storage"]["processed_dir"] / "sentiment_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path)
    print(f"Saved sentiment features: {out_path}")
    print(f"  {len(features)} days, {len(features.columns)} features")
    print(f"  Columns: {list(features.columns)}")


if __name__ == "__main__":
    main()
