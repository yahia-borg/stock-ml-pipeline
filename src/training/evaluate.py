"""
Model evaluation — computes accuracy, directional accuracy, Sharpe, and other metrics.

Tracks both classification performance and simulated trading performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.utils.logger import get_logger

log = get_logger("training.evaluate")

DIRECTION_LABELS = ["down", "flat", "up"]


def evaluate_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None = None,
    actual_returns: np.ndarray | None = None,
    horizon: int = 5,
) -> dict:
    """Evaluate a single fold's predictions.

    Args:
        y_true: True labels (0=down, 1=flat, 2=up)
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities [n_samples, n_classes]
        actual_returns: Actual forward returns (for Sharpe calculation)
        horizon: Forecast horizon in days

    Returns:
        Dict of metric name -> value
    """
    metrics = {}

    # ── Classification metrics ──
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class metrics — use labels param to handle folds with fewer than 3 classes
    report = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2],
        target_names=DIRECTION_LABELS,
        output_dict=True,
        zero_division=0,
    )
    for label in DIRECTION_LABELS:
        if label in report:
            metrics[f"{label}_precision"] = report[label]["precision"]
            metrics[f"{label}_recall"] = report[label]["recall"]
            metrics[f"{label}_f1"] = report[label]["f1-score"]

    # ── Directional accuracy (up vs down, ignoring flat) ──
    # This is the most important metric — can we tell up from down?
    directional_mask = (y_true != 1) & (y_pred != 1)  # both not flat
    if directional_mask.sum() > 0:
        metrics["directional_accuracy"] = accuracy_score(
            y_true[directional_mask], y_pred[directional_mask]
        )
    else:
        metrics["directional_accuracy"] = np.nan

    # ── Simulated trading metrics ──
    if actual_returns is not None:
        # Simple strategy: go long when predicting up, short when predicting down, flat = no position
        positions = np.where(y_pred == 2, 1, np.where(y_pred == 0, -1, 0))
        strategy_returns = positions * actual_returns

        # Sharpe ratio (annualized)
        if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
            daily_sharpe = np.mean(strategy_returns) / np.std(strategy_returns)
            annualization = np.sqrt(252 / horizon)
            metrics["sharpe_ratio"] = daily_sharpe * annualization
        else:
            metrics["sharpe_ratio"] = 0.0

        # Cumulative return
        metrics["cumulative_return"] = np.sum(strategy_returns)

        # Max drawdown
        cum = np.cumsum(strategy_returns)
        peak = np.maximum.accumulate(cum)
        drawdown = cum - peak
        metrics["max_drawdown"] = np.min(drawdown) if len(drawdown) > 0 else 0.0

        # Win rate
        trades = strategy_returns[strategy_returns != 0]
        if len(trades) > 0:
            metrics["win_rate"] = (trades > 0).mean()
            metrics["n_trades"] = len(trades)
        else:
            metrics["win_rate"] = 0.0
            metrics["n_trades"] = 0

    # ── Confidence metrics (if probabilities available) ──
    if y_pred_proba is not None:
        max_proba = np.max(y_pred_proba, axis=1)
        metrics["avg_confidence"] = np.mean(max_proba)

        # Accuracy on high-confidence predictions (top quartile)
        high_conf_mask = max_proba >= np.percentile(max_proba, 75)
        if high_conf_mask.sum() > 0:
            metrics["high_conf_accuracy"] = accuracy_score(
                y_true[high_conf_mask], y_pred[high_conf_mask]
            )

    metrics["n_samples"] = len(y_true)
    metrics["horizon"] = horizon

    return metrics


def summarize_walk_forward(fold_metrics: list[dict]) -> pd.DataFrame:
    """Summarize metrics across all walk-forward folds.

    Returns a DataFrame with one row per fold plus a summary row.
    """
    if not fold_metrics:
        return pd.DataFrame()

    df = pd.DataFrame(fold_metrics)

    # Add summary row (mean across folds)
    summary = df.select_dtypes(include=[np.number]).mean().to_dict()
    summary["fold_id"] = "MEAN"
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    return df


def print_evaluation_report(
    fold_metrics: list[dict],
    model_name: str = "Model",
    horizon: int = 5,
) -> pd.DataFrame:
    """Print a formatted evaluation report and return summary DataFrame."""
    summary = summarize_walk_forward(fold_metrics)

    if summary.empty:
        log.warning("No fold metrics to report")
        return summary

    mean_row = summary.iloc[-1]

    log.info("=" * 60)
    log.info("%s Evaluation Report (%dd horizon)", model_name, horizon)
    log.info("=" * 60)
    log.info("Folds: %d", len(fold_metrics))
    log.info("")
    log.info("Classification:")
    log.info("  Accuracy:              %.4f", mean_row.get("accuracy", 0))
    log.info("  Directional accuracy:  %.4f", mean_row.get("directional_accuracy", 0))
    log.info("  Macro F1:              %.4f", mean_row.get("macro_f1", 0))
    log.info("")
    log.info("Per-class F1:")
    for label in DIRECTION_LABELS:
        f1 = mean_row.get(f"{label}_f1", 0)
        log.info("  %-5s F1:              %.4f", label, f1)
    log.info("")

    if "sharpe_ratio" in mean_row:
        log.info("Trading simulation:")
        log.info("  Sharpe ratio:          %.4f", mean_row.get("sharpe_ratio", 0))
        log.info("  Cumulative return:     %.2f%%", mean_row.get("cumulative_return", 0) * 100)
        log.info("  Max drawdown:          %.2f%%", mean_row.get("max_drawdown", 0) * 100)
        log.info("  Win rate:              %.2f%%", mean_row.get("win_rate", 0) * 100)
    log.info("")

    if "high_conf_accuracy" in mean_row:
        log.info("Confidence:")
        log.info("  Avg confidence:        %.4f", mean_row.get("avg_confidence", 0))
        log.info("  High-conf accuracy:    %.4f", mean_row.get("high_conf_accuracy", 0))

    log.info("=" * 60)

    return summary
