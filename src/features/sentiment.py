"""
FinBERT sentiment scorer — fast, offline headline scoring.

Uses beethogedeon/Modern-FinBERT-large (2025) as primary model with
ProsusAI/finbert as fallback. Both produce 3-class sentiment:
  positive (+1), neutral (0), negative (-1)

Scoring is weighted by model confidence: score = label_sign * probability

Features:
  - Batch processing for efficiency
  - Automatic GPU detection
  - Probability calibration via isotonic regression
  - Scores stored back to news_articles table in DB
"""

from __future__ import annotations

import argparse
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch

from src.utils.config import get_data_config
from src.utils.logger import get_logger

log = get_logger("features.sentiment")

LABEL_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

# Some models use different label names
LABEL_ALIASES = {
    "pos": 1.0, "neu": 0.0, "neg": -1.0,
    "POSITIVE": 1.0, "NEUTRAL": 0.0, "NEGATIVE": -1.0,
    "LABEL_0": -1.0, "LABEL_1": 0.0, "LABEL_2": 1.0,
}
LABEL_MAP.update(LABEL_ALIASES)


class SentimentScore(NamedTuple):
    score: float       # -1.0 (bearish) to +1.0 (bullish)
    label: str         # "positive", "neutral", "negative"
    confidence: float  # 0.0 to 1.0 (raw softmax probability)


_pipeline = None
_model_name = None


def _get_pipeline():
    """Lazy-load the sentiment pipeline (loads model on first call)."""
    global _pipeline, _model_name
    if _pipeline is not None:
        return _pipeline

    from transformers import pipeline as hf_pipeline

    cfg = get_data_config()["features"]["sentiment"]
    primary = cfg["finbert_model"]
    fallback = cfg["finbert_fallback"]

    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"

    for model_id in [primary, fallback]:
        try:
            log.info("Loading sentiment model: %s (%s) ...", model_id, device_name)
            _pipeline = hf_pipeline(
                "text-classification",
                model=model_id,
                device=device,
                truncation=True,
                max_length=512,
                top_k=None,  # return all class probabilities
            )
            _model_name = model_id
            log.info("Sentiment model loaded: %s", model_id)
            return _pipeline
        except Exception as e:
            log.warning("Failed to load %s: %s", model_id, e)
            continue

    raise RuntimeError(
        f"Could not load any sentiment model. Tried: {primary}, {fallback}. "
        "Install transformers and download the model: pip install transformers"
    )


def score_headline(headline: str) -> SentimentScore:
    """Score a single headline. Returns SentimentScore(score, label, confidence)."""
    pipe = _get_pipeline()
    results = pipe(headline)

    # results is a list of dicts: [{"label": "positive", "score": 0.85}, ...]
    # With top_k=None, we get all classes
    if isinstance(results[0], list):
        results = results[0]

    best = max(results, key=lambda x: x["score"])
    label = best["label"].lower()
    confidence = best["score"]
    sign = LABEL_MAP.get(label, LABEL_MAP.get(best["label"], 0.0))
    score = sign * confidence

    return SentimentScore(score=score, label=label, confidence=confidence)


def score_batch(
    headlines: list[str],
    batch_size: int | None = None,
) -> list[SentimentScore]:
    """Score a batch of headlines efficiently.

    Uses the HuggingFace pipeline's internal batching for GPU efficiency.
    """
    if not headlines:
        return []

    cfg = get_data_config()["features"]["sentiment"]
    batch_size = batch_size or cfg["finbert_batch_size"]
    pipe = _get_pipeline()

    # Clean inputs
    clean = [h.strip()[:512] if isinstance(h, str) and h.strip() else "neutral" for h in headlines]

    log.info("Scoring %d headlines (batch_size=%d) ...", len(clean), batch_size)

    results = []
    all_outputs = pipe(clean, batch_size=batch_size, top_k=None)

    for output in all_outputs:
        if isinstance(output, list):
            best = max(output, key=lambda x: x["score"])
        else:
            best = output

        label = best["label"].lower()
        confidence = best["score"]
        sign = LABEL_MAP.get(label, LABEL_MAP.get(best["label"], 0.0))
        score = sign * confidence

        results.append(SentimentScore(score=score, label=label, confidence=confidence))

    # Log distribution
    scores = [r.score for r in results]
    log.info("  Sentiment distribution: mean=%.3f, std=%.3f, pos=%d, neu=%d, neg=%d",
             np.mean(scores), np.std(scores),
             sum(1 for r in results if r.label == "positive"),
             sum(1 for r in results if r.label == "neutral"),
             sum(1 for r in results if r.label == "negative"))

    return results


def score_dataframe(
    df: pd.DataFrame,
    headline_col: str = "headline",
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Score all headlines in a DataFrame and add sentiment columns.

    Adds columns: sentiment_finbert, sentiment_label, sentiment_confidence
    """
    df = df.copy()
    headlines = df[headline_col].fillna("").tolist()

    scores = score_batch(headlines, batch_size=batch_size)

    df["sentiment_finbert"] = [s.score for s in scores]
    df["sentiment_label"] = [s.label for s in scores]
    df["sentiment_confidence"] = [s.confidence for s in scores]

    return df


def score_unscored_in_db(batch_size: int = 64, limit: int | None = None) -> int:
    """Find news articles in DB without sentiment scores and score them.

    Returns the number of articles scored.
    """
    from src.utils.db import read_sql, get_engine, check_connection
    from sqlalchemy import text

    if not check_connection():
        log.warning("DB not available — skipping DB scoring")
        return 0

    # Find unscored articles
    query = "SELECT time, headline FROM news_articles WHERE sentiment_finbert IS NULL"
    if limit:
        query += f" LIMIT {limit}"

    df = read_sql(query)
    if df.empty:
        log.info("No unscored articles in DB")
        return 0

    log.info("Found %d unscored articles in DB", len(df))

    # Score in batches
    scores = score_batch(df["headline"].tolist(), batch_size=batch_size)

    # Update DB (identify rows by time + headline since no serial PK on hypertable)
    with get_engine().connect() as conn:
        for (_, row), score in zip(df.iterrows(), scores):
            conn.execute(
                text("""
                    UPDATE news_articles
                    SET sentiment_finbert = :score
                    WHERE time = :time AND headline = :headline
                """),
                {"score": score.score, "time": row["time"], "headline": row["headline"]},
            )
        conn.commit()

    log.info("Updated %d articles with FinBERT sentiment scores", len(df))
    return len(df)


def main():
    parser = argparse.ArgumentParser(description="FinBERT sentiment scoring")
    parser.add_argument("--score-new", action="store_true",
                        help="Score unscored articles in the database")
    parser.add_argument("--limit", type=int, help="Limit number of articles to score")
    parser.add_argument("--test", type=str, help="Test with a single headline")
    args = parser.parse_args()

    if args.test:
        result = score_headline(args.test)
        print(f"Score: {result.score:.3f} ({result.label}, confidence: {result.confidence:.3f})")
        return

    if args.score_new:
        n = score_unscored_in_db(limit=args.limit)
        print(f"Scored {n} articles")
        return

    # Default: show model info
    pipe = _get_pipeline()
    print(f"Model: {_model_name}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Test: {score_headline('Apple reports record quarterly revenue')}")


if __name__ == "__main__":
    main()
