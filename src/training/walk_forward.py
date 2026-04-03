"""
Walk-forward validation framework — THE critical piece for financial ML.

Rules:
  - Never use random train/test splits on time series
  - Always train on past, test on future
  - Purged embargo gap between train and test (prevents label leakage)
  - Scaler fitted INSIDE each fold (prevents feature leakage)

Supports:
  - Expanding window (anchored): training set grows, uses all history
  - Sliding window: fixed-size training window, old data drops off
  - Per-fold scaler fitting and saving
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from src.utils.config import get_data_config
from src.utils.logger import get_logger

log = get_logger("training.walk_forward")


@dataclass
class WalkForwardFold:
    """A single train/test split in walk-forward validation."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    gap_start: pd.Timestamp    # embargo period start
    gap_end: pd.Timestamp      # embargo period end

    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days

    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold_id}: "
            f"train [{self.train_start.date()} → {self.train_end.date()}] "
            f"({self.train_days}d) | "
            f"gap [{self.gap_start.date()} → {self.gap_end.date()}] | "
            f"test [{self.test_start.date()} → {self.test_end.date()}] "
            f"({self.test_days}d)"
        )


def generate_splits(
    df: pd.DataFrame,
    mode: str = "expanding",
    initial_train_years: int | None = None,
    test_months: int | None = None,
    step_months: int | None = None,
    gap_days: int | None = None,
    min_test_samples: int | None = None,
    sliding_window_years: int = 5,
) -> list[WalkForwardFold]:
    """Generate walk-forward validation splits.

    Args:
        df: DataFrame with DatetimeIndex
        mode: "expanding" (anchored) or "sliding" (fixed window)
        initial_train_years: Years of data for first training fold
        test_months: Length of each test period
        step_months: How far to slide between folds
        gap_days: Purged embargo gap between train end and test start
        min_test_samples: Skip folds with fewer test samples
        sliding_window_years: Training window size for sliding mode

    Returns:
        List of WalkForwardFold objects
    """
    cfg = get_data_config()["validation"]
    initial_train_years = initial_train_years or cfg["initial_train_years"]
    test_months = test_months or cfg["test_months"]
    step_months = step_months or cfg["step_months"]
    gap_days = gap_days or cfg["gap_days"]
    min_test_samples = min_test_samples or cfg["min_test_samples"]

    # Ensure datetime index — strip timezone for consistent comparison
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)

    data_start = df.index.min()
    data_end = df.index.max()

    folds = []
    fold_id = 0
    train_end = data_start + pd.DateOffset(years=initial_train_years)

    while True:
        gap_start = train_end
        gap_end = train_end + pd.DateOffset(days=gap_days)
        test_start = gap_end
        test_end = test_start + pd.DateOffset(months=test_months)

        # Check if test period extends beyond data
        if test_start >= data_end:
            break

        test_end = min(test_end, data_end)

        # Training start depends on mode
        if mode == "expanding":
            train_start = data_start
        elif mode == "sliding":
            train_start = max(
                data_start,
                train_end - pd.DateOffset(years=sliding_window_years),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'expanding' or 'sliding'.")

        # Count test samples
        test_mask = (df.index >= test_start) & (df.index <= test_end)
        n_test = test_mask.sum()

        if n_test >= min_test_samples:
            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                gap_start=gap_start,
                gap_end=gap_end,
            )
            folds.append(fold)
            fold_id += 1
        else:
            log.debug("Skipping fold at %s: only %d test samples (need %d)",
                      test_start.date(), n_test, min_test_samples)

        # Slide forward
        train_end += pd.DateOffset(months=step_months)

    log.info("Generated %d %s walk-forward folds from %s to %s",
             len(folds), mode, data_start.date(), data_end.date())
    if folds:
        log.info("  First: %s", folds[0])
        log.info("  Last:  %s", folds[-1])

    return folds


def split_data(
    df: pd.DataFrame,
    fold: WalkForwardFold,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split data into train and test sets for a given fold.

    Returns:
        (X_train, y_train, X_test, y_test)
    """
    train_mask = (df.index >= fold.train_start) & (df.index <= fold.train_end)
    test_mask = (df.index >= fold.test_start) & (df.index <= fold.test_end)

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]
    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, target_col]

    # Drop rows where target is NaN — but KEEP rows with NaN features
    # (tree models handle NaN natively; NaN features are filled to 0 later)
    train_valid = y_train.notna()
    test_valid = y_test.notna()

    return (
        X_train.loc[train_valid],
        y_train.loc[train_valid],
        X_test.loc[test_valid],
        y_test.loc[test_valid],
    )


def fit_fold_scaler(
    X_train: pd.DataFrame,
    method: str = "percentile_rank",
) -> tuple[pd.DataFrame, object]:
    """Fit a normalizer on training data only (prevents leakage).

    Returns:
        (X_train_normalized, scaler_object)
    """
    from src.features.normalizer import RollingPercentileRank, ExpandingZScore

    if method == "percentile_rank":
        scaler = RollingPercentileRank(min_periods=min(252, len(X_train) // 2))
    elif method == "zscore":
        scaler = ExpandingZScore(min_periods=min(252, len(X_train) // 2))
    else:
        # No normalization (for tree models)
        return X_train, None

    X_normalized = scaler.transform(X_train)
    return X_normalized, scaler


def walk_forward_iterator(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    mode: str = "expanding",
    normalize: str | None = None,
) -> Iterator[tuple[WalkForwardFold, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    """Iterate through walk-forward folds, yielding split data.

    Args:
        df: Full feature matrix with DatetimeIndex
        feature_cols: List of feature column names
        target_col: Target column name
        mode: "expanding" or "sliding"
        normalize: Normalization method ("percentile_rank", "zscore", or None)

    Yields:
        (fold, X_train, y_train, X_test, y_test)
    """
    # Strip timezone for consistent comparison with fold timestamps
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)

    folds = generate_splits(df, mode=mode)

    for fold in folds:
        X_train, y_train, X_test, y_test = split_data(
            df, fold, feature_cols, target_col
        )

        if len(X_train) == 0 or len(X_test) == 0:
            log.warning("Empty split at fold %d, skipping", fold.fold_id)
            continue

        # Normalize within fold
        if normalize:
            X_train, scaler = fit_fold_scaler(X_train, method=normalize)
            if scaler:
                X_test = scaler.transform(X_test)

        # Fill any remaining NaNs from normalization
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        log.info("Fold %d: train=%d, test=%d",
                 fold.fold_id, len(X_train), len(X_test))

        yield fold, X_train, y_train, X_test, y_test
