"""
LLM sentiment scorer — rich, context-aware headline analysis via vLLM.

Slower than FinBERT (~100ms/headline) but provides:
  - Nuanced scoring with reasoning
  - Ticker-specific context awareness
  - Key factor extraction
  - Magnitude estimation

Best used for daily batch enrichment alongside FinBERT for real-time scoring.
Requires VLLM_BASE_URL and VLLM_MODEL in .env.
"""

from __future__ import annotations

import json
import os
from typing import NamedTuple

import pandas as pd

from src.utils.logger import get_logger

log = get_logger("features.sentiment_llm")

SENTIMENT_PROMPT = """You are a financial analyst. Rate the market impact of this headline.

Headline: {headline}
{ticker_context}

Rules:
- Return ONLY a JSON object with these exact fields:
  {{"score": float, "magnitude": str, "factors": [str], "reason": str}}
- score: -1.0 (strongly bearish) to +1.0 (strongly bullish), 0.0 = neutral
- magnitude: "high", "medium", or "low" (expected market impact size)
- factors: 1-3 key factors driving the score (e.g., "earnings beat", "rate hike")
- reason: one sentence explanation
- Consider: earnings, macro policy, geopolitics, sector trends, supply chain"""


class LLMSentimentScore(NamedTuple):
    score: float
    magnitude: str
    factors: list[str]
    reason: str
    raw_response: str


def _get_client():
    """Get OpenAI-compatible client for vLLM."""
    from openai import OpenAI

    base_url = os.environ.get("VLLM_BASE_URL")
    if not base_url or base_url == "http://localhost:8021/v1":
        log.warning("VLLM_BASE_URL not configured — LLM sentiment unavailable")
        return None, None

    model = os.environ.get("VLLM_MODEL", "")
    if not model or model == "your-model-name":
        log.warning("VLLM_MODEL not set — LLM sentiment unavailable")
        return None, None

    client = OpenAI(base_url=base_url, api_key="dummy")
    return client, model


def score_headline(
    headline: str,
    ticker: str | None = None,
) -> LLMSentimentScore:
    """Score a single headline using the LLM."""
    client, model = _get_client()
    if client is None:
        return LLMSentimentScore(
            score=0.0, magnitude="low", factors=[], reason="LLM not configured", raw_response=""
        )

    ticker_context = f"Ticker context: {ticker}" if ticker else "General market news"
    prompt = SENTIMENT_PROMPT.format(headline=headline, ticker_context=ticker_context)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)

        return LLMSentimentScore(
            score=float(data.get("score", 0.0)),
            magnitude=data.get("magnitude", "low"),
            factors=data.get("factors", []),
            reason=data.get("reason", ""),
            raw_response=raw,
        )
    except Exception as e:
        log.warning("LLM scoring failed for '%s': %s", headline[:50], e)
        return LLMSentimentScore(
            score=0.0, magnitude="low", factors=[], reason=f"error: {e}", raw_response=""
        )


def score_batch(
    headlines: list[str],
    ticker: str | None = None,
) -> list[LLMSentimentScore]:
    """Score multiple headlines sequentially (LLM is inherently sequential).

    For high throughput, consider running multiple instances or using
    vLLM's continuous batching on the server side.
    """
    client, _ = _get_client()
    if client is None:
        log.warning("LLM not configured — returning zero scores for %d headlines", len(headlines))
        return [
            LLMSentimentScore(score=0.0, magnitude="low", factors=[], reason="LLM not configured", raw_response="")
            for _ in headlines
        ]

    log.info("LLM scoring %d headlines (ticker=%s) ...", len(headlines), ticker)
    results = []
    for i, headline in enumerate(headlines):
        result = score_headline(headline, ticker=ticker)
        results.append(result)
        if (i + 1) % 50 == 0:
            log.info("  Progress: %d/%d", i + 1, len(headlines))

    scored = [r for r in results if r.score != 0.0]
    log.info("LLM scoring done: %d/%d non-zero scores", len(scored), len(results))
    return results


def score_dataframe(
    df: pd.DataFrame,
    headline_col: str = "headline",
    ticker: str | None = None,
) -> pd.DataFrame:
    """Score headlines in a DataFrame and add LLM sentiment columns."""
    df = df.copy()
    headlines = df[headline_col].fillna("").tolist()

    scores = score_batch(headlines, ticker=ticker)

    df["sentiment_llm"] = [s.score for s in scores]
    df["sentiment_llm_magnitude"] = [s.magnitude for s in scores]
    df["sentiment_llm_factors"] = [s.factors for s in scores]
    df["sentiment_llm_reason"] = [s.reason for s in scores]

    return df


def score_unscored_in_db(limit: int | None = None) -> int:
    """Score articles in DB that have FinBERT scores but no LLM scores."""
    from src.utils.db import read_sql, get_engine, check_connection
    from sqlalchemy import text

    if not check_connection():
        return 0

    client, _ = _get_client()
    if client is None:
        return 0

    query = """
        SELECT time, headline
        FROM news_articles
        WHERE sentiment_finbert IS NOT NULL
          AND sentiment_llm IS NULL
        ORDER BY time DESC
    """
    if limit:
        query += f" LIMIT {limit}"

    df = read_sql(query)
    if df.empty:
        log.info("No articles need LLM scoring")
        return 0

    log.info("LLM scoring %d articles from DB ...", len(df))
    scores = score_batch(df["headline"].tolist())

    with get_engine().connect() as conn:
        for (_, row), score in zip(df.iterrows(), scores):
            conn.execute(
                text("""
                    UPDATE news_articles SET sentiment_llm = :score
                    WHERE time = :time AND headline = :headline
                """),
                {"score": score.score, "time": row["time"], "headline": row["headline"]},
            )
        conn.commit()

    log.info("Updated %d articles with LLM sentiment", len(df))
    return len(df)
