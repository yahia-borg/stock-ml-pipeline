"""
Prediction logger — stores every prediction to DB for monitoring and evaluation.

Two operations:
  1. log_prediction() — called after each prediction, writes to `predictions` table
  2. backfill_actuals() — runs daily, fills in actual_return/actual_direction
     once ground truth prices are available
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.utils.db import get_engine, check_connection, read_sql
from src.utils.logger import get_logger

log = get_logger("inference.logger")


def log_prediction(
    ticker: str,
    model_name: str,
    horizon_days: int,
    predicted_direction: str,
    predicted_proba: dict,
    confidence: float,
    sentiment_score: float | None = None,
    regime_state: int | None = None,
    model_version: str | None = None,
    metadata: dict | None = None,
) -> bool:
    """Log a single prediction to the database."""
    if not check_connection():
        log.warning("DB not connected — prediction not logged")
        return False

    try:
        import json
        with get_engine().connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO predictions
                        (time, ticker, model_name, model_version, horizon_days,
                         predicted_direction, predicted_proba, confidence,
                         sentiment_score, regime_state, metadata)
                    VALUES
                        (:time, :ticker, :model_name, :model_version, :horizon_days,
                         :predicted_direction, :predicted_proba, :confidence,
                         :sentiment_score, :regime_state, :metadata)
                """),
                {
                    "time": datetime.now(timezone.utc),
                    "ticker": ticker,
                    "model_name": model_name,
                    "model_version": model_version,
                    "horizon_days": horizon_days,
                    "predicted_direction": predicted_direction,
                    "predicted_proba": json.dumps(predicted_proba),
                    "confidence": confidence,
                    "sentiment_score": sentiment_score,
                    "regime_state": regime_state,
                    "metadata": json.dumps(metadata) if metadata else None,
                },
            )
            conn.commit()
        return True
    except Exception as e:
        log.error("Failed to log prediction: %s", e)
        return False


def log_batch(predictions: list[dict]) -> int:
    """Log multiple predictions at once."""
    logged = 0
    for p in predictions:
        if log_prediction(**p):
            logged += 1
    log.info("Logged %d/%d predictions", logged, len(predictions))
    return logged


def backfill_actuals(lookback_days: int = 30) -> int:
    """Fill in actual returns for past predictions where ground truth is now available.

    Looks at predictions from the last `lookback_days` that don't have
    actual_return filled in yet, and computes the actual forward return
    from the prices table.
    """
    if not check_connection():
        return 0

    # Find predictions missing actuals
    df = read_sql(f"""
        SELECT p.time, p.ticker, p.horizon_days, p.predicted_direction
        FROM predictions p
        WHERE p.actual_return IS NULL
          AND p.time > NOW() - INTERVAL '{lookback_days} days'
        ORDER BY p.time
    """)

    if df.empty:
        log.info("No predictions need actual backfill")
        return 0

    log.info("Backfilling actuals for %d predictions ...", len(df))
    updated = 0

    with get_engine().connect() as conn:
        for _, row in df.iterrows():
            pred_time = pd.Timestamp(row["time"])
            horizon = int(row["horizon_days"])
            ticker = row["ticker"]

            # Get price at prediction time and at horizon
            price_at_pred = conn.execute(
                text("""
                    SELECT close FROM prices
                    WHERE ticker = :ticker AND time <= :pred_time
                    ORDER BY time DESC LIMIT 1
                """),
                {"ticker": ticker, "pred_time": pred_time},
            ).fetchone()

            future_time = pred_time + pd.Timedelta(days=horizon * 1.5)  # buffer for weekends
            price_at_horizon = conn.execute(
                text("""
                    SELECT close FROM prices
                    WHERE ticker = :ticker AND time > :pred_time
                    ORDER BY time ASC
                    OFFSET :offset LIMIT 1
                """),
                {"ticker": ticker, "pred_time": pred_time, "offset": max(0, horizon - 1)},
            ).fetchone()

            if not price_at_pred or not price_at_horizon:
                continue

            actual_return = (price_at_horizon[0] - price_at_pred[0]) / price_at_pred[0]

            # Determine direction using simple threshold
            if actual_return > 0.005:
                actual_direction = "up"
            elif actual_return < -0.005:
                actual_direction = "down"
            else:
                actual_direction = "flat"

            conn.execute(
                text("""
                    UPDATE predictions
                    SET actual_return = :actual_return, actual_direction = :actual_direction
                    WHERE time = :time AND ticker = :ticker AND horizon_days = :horizon
                      AND actual_return IS NULL
                """),
                {
                    "actual_return": actual_return,
                    "actual_direction": actual_direction,
                    "time": pred_time,
                    "ticker": ticker,
                    "horizon": horizon,
                },
            )
            updated += 1

        conn.commit()

    log.info("Backfilled actuals for %d predictions", updated)
    return updated
