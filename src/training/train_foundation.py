"""
Foundation model evaluation — walk-forward evaluation of zero-shot models.

Unlike baseline training, foundation models need NO training.
We just run walk-forward evaluation: at each fold's test period,
feed the model the available history and evaluate its predictions.

Usage:
  python -m src.training.train_foundation                        # evaluate Chronos-2
  python -m src.training.train_foundation --model moirai2        # evaluate Moirai 2.0
  python -m src.training.train_foundation --horizon 5 --ticker AAPL
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import ROOT_DIR, get_data_config
from src.utils.logger import get_logger

log = get_logger("training.foundation")

MODELS_DIR = ROOT_DIR / os.environ.get("MODELS_DIR", "models")


def evaluate_foundation(
    model_name: str = "chronos2",
    horizon: int = 5,
    ticker: str | None = None,
    mode: str = "expanding",
) -> dict:
    """Evaluate a foundation model using walk-forward validation.

    No training occurs — the model is used zero-shot at each test fold.
    We feed it the price history up to the test period and evaluate predictions.
    """
    from src.models.foundation import get_forecaster
    from src.training.walk_forward import generate_splits
    from src.training.evaluate import evaluate_fold, print_evaluation_report
    from src.training.train import load_feature_matrix

    # Load data
    df = load_feature_matrix(ticker)
    if df.empty:
        return {}

    if "ticker" in df.columns and ticker:
        df = df[df["ticker"] == ticker].drop(columns=["ticker"])
    elif "ticker" in df.columns:
        # For foundation models, evaluate per-ticker since they use raw price series
        tickers = df["ticker"].unique()
        log.info("Multiple tickers found. Evaluating on first: %s", tickers[0])
        ticker = tickers[0]
        df = df[df["ticker"] == ticker].drop(columns=["ticker"])

    target_col = f"label_{horizon}d"
    return_col = f"fwd_return_{horizon}d"

    if target_col not in df.columns or "close" not in df.columns:
        log.error("Required columns missing. Need 'close' and '%s'", target_col)
        return {}

    # Get forecaster
    log.info("Loading %s for %dd horizon evaluation ...", model_name, horizon)
    forecaster = get_forecaster(model_name)

    # Generate walk-forward splits
    folds = generate_splits(df, mode=mode)
    log.info("Evaluating %s across %d walk-forward folds", model_name, len(folds))

    fold_metrics = []

    for fold in folds:
        # Get price history up to test start (what the model can see)
        history_mask = df.index <= fold.train_end
        prices = df.loc[history_mask, "close"]

        # Get test period labels and returns
        test_mask = (df.index >= fold.test_start) & (df.index <= fold.test_end)
        test_df = df.loc[test_mask]

        if len(test_df) == 0 or len(prices) < 60:
            continue

        y_true = test_df[target_col].dropna()
        if len(y_true) == 0:
            continue

        # Predict for each test day
        # At each test day, the model sees all history up to that day
        y_preds = []
        y_proba_list = []
        valid_indices = []

        for test_date in y_true.index:
            # History available to the model at this point
            available = df.loc[df.index <= test_date, "close"]
            if len(available) < 60:
                continue

            try:
                result = forecaster.predict_direction(available, horizon=horizon)
                y_preds.append(result["label"])
                y_proba_list.append([
                    result["probabilities"]["down"],
                    result["probabilities"]["flat"],
                    result["probabilities"]["up"],
                ])
                valid_indices.append(test_date)
            except Exception as e:
                log.debug("Prediction failed at %s: %s", test_date, e)
                continue

        if not y_preds:
            continue

        y_pred = np.array(y_preds)
        y_pred_proba = np.array(y_proba_list)
        y_true_aligned = y_true.loc[valid_indices].values

        # Actual returns for Sharpe
        actual_returns = None
        if return_col in df.columns:
            actual_returns = df.loc[valid_indices, return_col].values

        # Evaluate
        metrics = evaluate_fold(
            y_true=y_true_aligned,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            actual_returns=actual_returns,
            horizon=horizon,
        )
        metrics["fold_id"] = fold.fold_id
        metrics["test_start"] = str(fold.test_start.date())
        metrics["test_end"] = str(fold.test_end.date())
        fold_metrics.append(metrics)

        log.info("Fold %d: acc=%.4f, dir_acc=%.4f, n=%d",
                 fold.fold_id,
                 metrics["accuracy"],
                 metrics.get("directional_accuracy", 0),
                 metrics["n_samples"])

    # Summary
    summary = print_evaluation_report(
        fold_metrics, model_name=model_name.upper(), horizon=horizon
    )

    # Save results
    results = {
        "model_type": model_name,
        "horizon": horizon,
        "mode": mode,
        "ticker": ticker or "pooled",
        "n_folds": len(fold_metrics),
        "fold_metrics": fold_metrics,
        "zero_shot": True,
    }

    results_path = MODELS_DIR / f"{model_name}_{horizon}d_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s", results_path.name)

    return results


def compare_all_models(horizon: int = 5) -> pd.DataFrame:
    """Load all saved results and compare models side by side."""
    results_files = sorted(MODELS_DIR.glob(f"*_{horizon}d_results.json"))

    if not results_files:
        log.warning("No results files found for %dd horizon", horizon)
        return pd.DataFrame()

    rows = []
    for path in results_files:
        with open(path) as f:
            data = json.load(f)

        if not data.get("fold_metrics"):
            continue

        metrics = data["fold_metrics"]
        row = {
            "model": data["model_type"],
            "horizon": horizon,
            "n_folds": len(metrics),
            "accuracy": np.mean([m["accuracy"] for m in metrics]),
            "directional_acc": np.mean([m.get("directional_accuracy", 0) for m in metrics]),
            "macro_f1": np.mean([m["macro_f1"] for m in metrics]),
            "sharpe": np.mean([m.get("sharpe_ratio", 0) for m in metrics]),
            "max_drawdown": np.mean([m.get("max_drawdown", 0) for m in metrics]),
            "win_rate": np.mean([m.get("win_rate", 0) for m in metrics]),
            "zero_shot": data.get("zero_shot", False),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    comparison = pd.DataFrame(rows).sort_values("directional_acc", ascending=False)

    log.info("\n" + "=" * 80)
    log.info("MODEL COMPARISON — %dd HORIZON", horizon)
    log.info("=" * 80)
    log.info("%-15s %7s %7s %7s %7s %7s %7s %s",
             "Model", "Acc", "DirAcc", "F1", "Sharpe", "MaxDD", "WinRate", "Type")
    log.info("-" * 80)
    for _, row in comparison.iterrows():
        model_type = "zero-shot" if row["zero_shot"] else "trained"
        log.info("%-15s %7.4f %7.4f %7.4f %7.4f %7.2f%% %7.2f%% %s",
                 row["model"], row["accuracy"], row["directional_acc"],
                 row["macro_f1"], row["sharpe"],
                 row["max_drawdown"] * 100, row["win_rate"] * 100, model_type)
    log.info("=" * 80)

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate foundation models")
    parser.add_argument("--model", choices=["chronos2", "moirai2"], default="chronos2")
    parser.add_argument("--horizon", type=int, choices=[1, 5, 20], default=5)
    parser.add_argument("--ticker", help="Evaluate on specific ticker")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all models (no evaluation, just load results)")
    args = parser.parse_args()

    if args.compare:
        compare_all_models(horizon=args.horizon)
        return

    evaluate_foundation(
        model_name=args.model,
        horizon=args.horizon,
        ticker=args.ticker,
    )


if __name__ == "__main__":
    main()
