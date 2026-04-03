"""
Retraining orchestrator — automated model retraining pipeline.

Triggers:
  1. Scheduled (monthly by default)
  2. Drift-triggered (performance or regime change)

Flow:
  1. Check drift → decide if retraining needed
  2. Rebuild features (if data updated)
  3. Train new model
  4. Compare with current production model
  5. If better → register as staging → shadow period → promote
  6. If not → keep current production

Usage:
  python -m src.training.retrain                  # full check + retrain if needed
  python -m src.training.retrain --force          # force retrain regardless of drift
  python -m src.training.retrain --model lgbm     # retrain specific model
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.utils.config import ROOT_DIR, get_data_config, get_model_config
from src.utils.logger import get_logger

log = get_logger("training.retrain")

MODELS_DIR = ROOT_DIR / os.environ.get("MODELS_DIR", "models")


def retrain_pipeline(
    model_type: str = "lgbm",
    horizon: int = 5,
    force: bool = False,
) -> dict:
    """Full retraining pipeline: check drift → retrain → compare → promote."""
    from src.training.drift import full_drift_check
    from src.training.train import train_baseline
    from src.training.registry import (
        register_model, get_production_model, should_promote, promote_to_production,
    )

    log.info("=" * 60)
    log.info("RETRAINING PIPELINE: %s %dd", model_type, horizon)
    log.info("=" * 60)

    result = {
        "model_type": model_type,
        "horizon": horizon,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retrained": False,
        "promoted": False,
        "reason": "",
    }

    # Step 1: Check drift
    if not force:
        report = full_drift_check()
        if not report.any_drift:
            result["reason"] = "No drift detected — skipping retraining"
            log.info(result["reason"])
            return result
        result["drift"] = {
            "performance": report.performance_drift,
            "data": report.data_drift,
            "regime": report.regime_drift,
            "recommendation": report.recommendation,
        }
        log.info("Drift detected: %s", report.recommendation)
    else:
        result["reason"] = "Forced retraining"
        log.info("Forced retraining (skipping drift check)")

    # Step 2: Train new model
    log.info("Training new %s model ...", model_type)
    train_results = train_baseline(
        model_type=model_type,
        horizon=horizon,
        save_models=True,
    )

    if not train_results or not train_results.get("fold_metrics"):
        result["reason"] = "Training failed — no metrics produced"
        log.error(result["reason"])
        return result

    result["retrained"] = True

    # Compute summary metrics
    metrics = train_results["fold_metrics"]
    new_metrics = {
        "accuracy": float(np.mean([m["accuracy"] for m in metrics])),
        "directional_accuracy": float(np.nanmean([m.get("directional_accuracy", 0) for m in metrics])),
        "macro_f1": float(np.mean([m["macro_f1"] for m in metrics])),
        "sharpe_ratio": float(np.mean([m.get("sharpe_ratio", 0) for m in metrics])),
        "n_folds": len(metrics),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    result["new_metrics"] = new_metrics
    result["version"] = version

    # Step 3: Register in staging
    artifact_path = str(MODELS_DIR / f"{model_type}_{horizon}d_results.json")
    register_model(
        model_name=f"{model_type}_{horizon}d",
        version=version,
        metrics=new_metrics,
        artifact_path=artifact_path,
    )

    # Step 4: Compare with production
    current = get_production_model(f"{model_type}_{horizon}d")
    current_metrics = current["metrics"] if current else None

    if should_promote(new_metrics, current_metrics):
        # Step 5: Promote (in production, you'd add shadow mode here)
        promote_to_production(f"{model_type}_{horizon}d", version)
        result["promoted"] = True
        result["reason"] = f"New model promoted: dir_acc={new_metrics['directional_accuracy']:.4f}"
        log.info("NEW MODEL PROMOTED to production!")
    else:
        result["reason"] = "New model not better than current production"
        log.info("Keeping current production model")

    log.info("=" * 60)
    log.info("Retraining complete: %s", result["reason"])
    log.info("=" * 60)

    # Save report
    report_path = MODELS_DIR / f"retrain_{model_type}_{horizon}d_{version}.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def retrain_all(force: bool = False) -> list[dict]:
    """Retrain all model types across all horizons."""
    horizons = get_data_config()["horizons"]
    model_types = ["lgbm", "xgboost"]

    results = []
    for model_type in model_types:
        for horizon in horizons:
            result = retrain_pipeline(model_type, horizon, force=force)
            results.append(result)

    # Summary
    retrained = sum(1 for r in results if r["retrained"])
    promoted = sum(1 for r in results if r["promoted"])
    log.info("Retraining summary: %d/%d retrained, %d promoted",
             retrained, len(results), promoted)

    return results


def main():
    parser = argparse.ArgumentParser(description="Automated retraining pipeline")
    parser.add_argument("--model", choices=["lgbm", "xgboost", "all"], default="all")
    parser.add_argument("--horizon", type=int, choices=[1, 5, 20], default=5)
    parser.add_argument("--force", action="store_true", help="Force retrain regardless of drift")
    args = parser.parse_args()

    if args.model == "all":
        retrain_all(force=args.force)
    else:
        retrain_pipeline(args.model, args.horizon, force=args.force)


if __name__ == "__main__":
    main()
