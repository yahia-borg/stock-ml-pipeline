"""
Training orchestrator — runs walk-forward training for baseline models.

Usage:
  python -m src.training.train                           # train all baselines
  python -m src.training.train --model lgbm --horizon 5  # train specific model/horizon
  python -m src.training.train --ticker AAPL             # train on specific ticker
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import ROOT_DIR, get_data_config, get_model_config
from src.utils.logger import get_logger

log = get_logger("training.train")

MODELS_DIR = ROOT_DIR / os.environ.get("MODELS_DIR", "models")


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Identify feature columns (everything except targets, metadata, raw OHLCV)."""
    exclude_prefixes = ("fwd_", "label_", "threshold_")
    exclude_exact = {
        "ticker", "time", "open", "high", "low", "close", "volume", "adj_close",
    }

    return [
        c for c in df.columns
        if c not in exclude_exact
        and not any(c.startswith(p) for p in exclude_prefixes)
    ]


def load_feature_matrix(ticker: str | None = None) -> pd.DataFrame:
    """Load the pre-built feature matrix."""
    cfg = get_data_config()
    processed_dir = ROOT_DIR / cfg["storage"]["processed_dir"]

    if ticker:
        path = processed_dir / f"{ticker}_features.parquet"
    else:
        path = processed_dir / "feature_matrix.parquet"

    if not path.exists():
        log.error("Feature matrix not found at %s. Run `make features` first.", path)
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index("time")
        df.index = pd.to_datetime(df.index)

    # Strip timezone for consistent walk-forward comparison
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    log.info("Loaded feature matrix: %d rows, %d columns from %s",
             len(df), len(df.columns), path.name)
    return df


def train_baseline(
    model_type: str = "lgbm",
    horizon: int = 5,
    ticker: str | None = None,
    mode: str = "expanding",
    save_models: bool = True,
) -> dict:
    """Train a baseline model using walk-forward validation.

    Args:
        model_type: "lgbm" or "xgboost"
        horizon: Forecast horizon in days (1, 5, or 20)
        ticker: Train on specific ticker (None = all tickers pooled)
        mode: "expanding" or "sliding" walk-forward
        save_models: Whether to save per-fold models to disk

    Returns:
        Dict with training results and metrics summary
    """
    from src.models.baseline import build_lgbm, build_xgboost, train_model, save_model, get_feature_importance
    from src.training.walk_forward import walk_forward_iterator
    from src.training.evaluate import evaluate_fold, print_evaluation_report

    # Load data
    df = load_feature_matrix(ticker)
    if df.empty:
        return {}

    # If specific ticker requested, filter. Otherwise pool all tickers.
    if "ticker" in df.columns and ticker:
        df = df[df["ticker"] == ticker].drop(columns=["ticker"])
    elif "ticker" in df.columns:
        # Pool all tickers — use integer index to handle duplicate dates
        # This gives the model cross-sectional data and more samples
        n_tickers = df["ticker"].nunique()
        df = df.sort_index()
        df = df.drop(columns=["ticker"])
        df = df.reset_index(drop=False)
        # Create unique integer index, keep time for walk-forward splitting
        df = df.rename(columns={df.columns[0]: "_time"})
        df.index = pd.RangeIndex(len(df))
        # Restore time-based index for walk-forward (use _time column)
        df = df.set_index("_time")
        df.index.name = None
        log.info("Pooled %d tickers (%d total rows)", n_tickers, len(df))

    feature_cols = _get_feature_cols(df)
    target_col = f"label_{horizon}d"
    return_col = f"fwd_return_{horizon}d"

    if target_col not in df.columns:
        log.error("Target column '%s' not found. Available: %s",
                  target_col, [c for c in df.columns if c.startswith("label_")])
        return {}

    log.info("Training %s on %dd horizon (%s mode)", model_type, horizon, mode)
    log.info("  Features: %d, Target: %s", len(feature_cols), target_col)

    # Walk-forward training
    fold_metrics = []
    all_importance = []

    # Tree models don't need normalization (invariant to monotonic transforms)
    for fold, X_train, y_train, X_test, y_test in walk_forward_iterator(
        df, feature_cols, target_col, mode=mode, normalize=None
    ):
        # Build model
        if model_type == "lgbm":
            model = build_lgbm()
        elif model_type == "xgboost":
            model = build_xgboost()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Split a validation set from the end of training for early stopping
        val_size = int(len(X_train) * 0.15)
        if val_size > 50:
            X_tr = X_train.iloc[:-val_size]
            y_tr = y_train.iloc[:-val_size]
            X_val = X_train.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]
        else:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = None, None

        # Train
        model = train_model(model, X_tr.values, y_tr.values,
                            X_val.values if X_val is not None else None,
                            y_val.values if y_val is not None else None)

        # Predict
        y_pred = model.predict(X_test.values)
        y_pred_proba = model.predict_proba(X_test.values)

        # Get actual returns for Sharpe calculation
        actual_returns = None
        if return_col in df.columns:
            test_mask = (df.index >= fold.test_start) & (df.index <= fold.test_end)
            returns_series = df.loc[test_mask, return_col].dropna()
            if len(returns_series) >= len(y_test):
                actual_returns = returns_series.iloc[:len(y_test)].values

        # Evaluate
        metrics = evaluate_fold(
            y_true=y_test.values,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            actual_returns=actual_returns,
            horizon=horizon,
        )
        metrics["fold_id"] = fold.fold_id
        metrics["train_start"] = str(fold.train_start.date())
        metrics["train_end"] = str(fold.train_end.date())
        metrics["test_start"] = str(fold.test_start.date())
        metrics["test_end"] = str(fold.test_end.date())
        fold_metrics.append(metrics)

        # Feature importance
        importance = get_feature_importance(model, feature_cols)
        all_importance.append(pd.Series(importance))

        # Save per-fold model
        if save_models:
            save_model(model, model_type, horizon, fold.fold_id)

    # Summary
    summary = print_evaluation_report(fold_metrics, model_name=model_type.upper(), horizon=horizon)

    # Aggregate feature importance across folds
    if all_importance:
        mean_importance = pd.concat(all_importance, axis=1).mean(axis=1).sort_values(ascending=False)
        log.info("Top 10 features by importance:")
        for feat, imp in mean_importance.head(10).items():
            log.info("  %-30s %.4f", feat, imp)

    # Save results
    results = {
        "model_type": model_type,
        "horizon": horizon,
        "mode": mode,
        "ticker": ticker or "pooled",
        "n_folds": len(fold_metrics),
        "fold_metrics": fold_metrics,
        "top_features": mean_importance.head(20).to_dict() if all_importance else {},
    }

    results_path = MODELS_DIR / f"{model_type}_{horizon}d_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s", results_path.name)

    return results


def train_all_baselines(ticker: str | None = None) -> dict:
    """Train all baseline models across all horizons."""
    cfg = get_data_config()
    horizons = cfg["horizons"]
    model_types = ["lgbm", "xgboost"]

    all_results = {}
    for model_type in model_types:
        for horizon in horizons:
            key = f"{model_type}_{horizon}d"
            log.info("=" * 60)
            log.info("Training: %s", key)
            log.info("=" * 60)

            results = train_baseline(
                model_type=model_type,
                horizon=horizon,
                ticker=ticker,
            )
            all_results[key] = results

    # Final comparison
    log.info("\n" + "=" * 60)
    log.info("BASELINE COMPARISON")
    log.info("=" * 60)
    log.info("%-20s %8s %8s %8s %8s", "Model", "Acc", "DirAcc", "Sharpe", "F1")
    log.info("-" * 60)
    for key, results in all_results.items():
        if results and results.get("fold_metrics"):
            metrics = results["fold_metrics"]
            mean_acc = np.mean([m["accuracy"] for m in metrics])
            mean_dir = np.mean([m.get("directional_accuracy", 0) for m in metrics])
            mean_sharpe = np.mean([m.get("sharpe_ratio", 0) for m in metrics])
            mean_f1 = np.mean([m["macro_f1"] for m in metrics])
            log.info("%-20s %8.4f %8.4f %8.4f %8.4f",
                     key, mean_acc, mean_dir, mean_sharpe, mean_f1)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--model", choices=["lgbm", "xgboost", "all"], default="all",
                        help="Model type to train")
    parser.add_argument("--horizon", type=int, choices=[1, 5, 20],
                        help="Forecast horizon (default: all)")
    parser.add_argument("--ticker", help="Train on specific ticker")
    parser.add_argument("--mode", choices=["expanding", "sliding"], default="expanding",
                        help="Walk-forward mode")
    args = parser.parse_args()

    if args.model == "all" and args.horizon is None:
        train_all_baselines(ticker=args.ticker)
    elif args.model == "all":
        for m in ["lgbm", "xgboost"]:
            train_baseline(m, args.horizon, ticker=args.ticker, mode=args.mode)
    else:
        horizons = [args.horizon] if args.horizon else get_data_config()["horizons"]
        for h in horizons:
            train_baseline(args.model, h, ticker=args.ticker, mode=args.mode)


if __name__ == "__main__":
    main()
