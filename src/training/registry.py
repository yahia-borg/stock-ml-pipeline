"""
Model registry — tracks model versions with stage transitions.

Stages: staging -> production -> archived
Shadow mode: new model runs in parallel for N days before promotion.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.utils.db import check_connection, get_engine, read_sql
from src.utils.logger import get_logger

log = get_logger("training.registry")

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))


def register_model(
    model_name: str,
    version: str,
    metrics: dict,
    artifact_path: str,
    train_start: str | None = None,
    train_end: str | None = None,
    feature_hash: str | None = None,
) -> bool:
    """Register a new model version in staging."""
    if not check_connection():
        log.warning("DB not available — model not registered")
        return False

    from sqlalchemy import text
    try:
        with get_engine().connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO model_registry
                        (model_name, version, stage, metrics, artifact_path,
                         train_start, train_end, feature_hash, created_at)
                    VALUES
                        (:model_name, :version, 'staging', :metrics, :artifact_path,
                         :train_start, :train_end, :feature_hash, :created_at)
                    ON CONFLICT (model_name, version) DO UPDATE
                    SET metrics = :metrics, artifact_path = :artifact_path
                """),
                {
                    "model_name": model_name,
                    "version": version,
                    "metrics": json.dumps(metrics),
                    "artifact_path": artifact_path,
                    "train_start": train_start,
                    "train_end": train_end,
                    "feature_hash": feature_hash,
                    "created_at": datetime.now(timezone.utc),
                },
            )
            conn.commit()

        log.info("Registered model: %s v%s (staging)", model_name, version)
        return True
    except Exception as e:
        log.error("Failed to register model: %s", e)
        return False


def promote_to_production(model_name: str, version: str) -> bool:
    """Promote a staging model to production. Archives the current production model."""
    if not check_connection():
        return False

    from sqlalchemy import text
    try:
        with get_engine().connect() as conn:
            # Archive current production
            conn.execute(
                text("""
                    UPDATE model_registry
                    SET stage = 'archived'
                    WHERE model_name = :model_name AND stage = 'production'
                """),
                {"model_name": model_name},
            )

            # Promote new model
            conn.execute(
                text("""
                    UPDATE model_registry
                    SET stage = 'production', promoted_at = :now
                    WHERE model_name = :model_name AND version = :version
                """),
                {"model_name": model_name, "version": version, "now": datetime.now(timezone.utc)},
            )
            conn.commit()

        log.info("Promoted %s v%s to production", model_name, version)
        return True
    except Exception as e:
        log.error("Promotion failed: %s", e)
        return False


def get_production_model(model_name: str) -> dict | None:
    """Get the current production model for a given name."""
    if not check_connection():
        return None

    df = read_sql(
        "SELECT * FROM model_registry WHERE model_name = :name AND stage = 'production' LIMIT 1",
        {"name": model_name},
    )
    if df.empty:
        return None

    row = df.iloc[0].to_dict()
    if isinstance(row.get("metrics"), str):
        row["metrics"] = json.loads(row["metrics"])
    return row


def list_models(stage: str | None = None) -> list[dict]:
    """List all registered models, optionally filtered by stage."""
    if not check_connection():
        return []

    query = "SELECT * FROM model_registry"
    if stage:
        query += f" WHERE stage = '{stage}'"
    query += " ORDER BY created_at DESC"

    df = read_sql(query)
    results = []
    for _, row in df.iterrows():
        d = row.to_dict()
        if isinstance(d.get("metrics"), str):
            d["metrics"] = json.loads(d["metrics"])
        results.append(d)
    return results


def should_promote(
    candidate_metrics: dict,
    current_metrics: dict | None,
    min_improvement: float = 0.01,
) -> bool:
    """Decide if a candidate model should replace the current production model.

    Compares directional accuracy as the primary metric.
    """
    candidate_dir_acc = candidate_metrics.get("directional_accuracy", 0)

    if current_metrics is None:
        log.info("No current production model — candidate auto-promoted")
        return True

    current_dir_acc = current_metrics.get("directional_accuracy", 0)
    improvement = candidate_dir_acc - current_dir_acc

    if improvement >= min_improvement:
        log.info("Candidate BETTER: %.4f vs %.4f (improvement: +%.4f)",
                 candidate_dir_acc, current_dir_acc, improvement)
        return True
    else:
        log.info("Candidate NOT better: %.4f vs %.4f (diff: %.4f < %.4f threshold)",
                 candidate_dir_acc, current_dir_acc, improvement, min_improvement)
        return False
