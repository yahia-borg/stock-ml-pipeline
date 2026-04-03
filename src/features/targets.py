"""
Target engineering — creates prediction targets for the model.

Key design decisions (based on research):
  - Adaptive volatility-based thresholds (not fixed %) — stable class distribution across regimes
  - Multi-horizon: 1d, 5d, 20d forward returns
  - Dual targets: continuous returns (regression) + 3-class labels (classification)
  - Threshold scaling: k * sqrt(horizon) * rolling_std
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import get_data_config
from src.utils.logger import get_logger

log = get_logger("features.targets")

LABEL_MAP = {0: "down", 1: "flat", 2: "up"}
LABEL_MAP_INV = {"down": 0, "flat": 1, "up": 2}


def compute_forward_returns(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute forward returns for each horizon.

    These are the *actual* future returns — used as targets during training
    and for evaluating predictions after the fact.
    """
    cfg = get_data_config()
    horizons = horizons or cfg["horizons"]

    df = df.copy()
    for h in horizons:
        # Forward return: price at t+h / price at t - 1
        df[f"fwd_return_{h}d"] = df["close"].shift(-h) / df["close"] - 1
        # Forward log return (better statistical properties)
        df[f"fwd_log_return_{h}d"] = np.log(df["close"].shift(-h) / df["close"])

    log.info("Forward returns computed for horizons: %s", horizons)
    return df


def adaptive_threshold(
    returns: pd.Series,
    k: float = 0.4,
    vol_window: int = 60,
    min_vol: float = 0.001,
) -> pd.Series:
    """Compute adaptive threshold based on recent volatility.

    threshold_t = k * rolling_std(returns, vol_window)

    This ensures ~30-40% of observations fall in the 'flat' class
    regardless of market regime.
    """
    rolling_vol = returns.rolling(vol_window, min_periods=20).std()
    threshold = k * rolling_vol.clip(lower=min_vol)
    return threshold


def compute_labels(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
    k: float = 0.4,
    vol_window: int = 60,
) -> pd.DataFrame:
    """Classify forward returns into 3 classes: down (0), flat (1), up (2).

    Uses adaptive volatility thresholds scaled by sqrt(horizon):
      threshold = k * sqrt(h) * rolling_std(1d_returns, vol_window)

    This is better than fixed thresholds because:
    - In low-vol regimes: tighter threshold = more flat, less noise
    - In high-vol regimes: wider threshold = only labels big moves
    - Consistent class distribution across regimes
    """
    cfg = get_data_config()
    horizons = horizons or cfg["horizons"]

    df = df.copy()

    # Base 1-day volatility for threshold computation
    daily_returns = df["close"].pct_change()

    for h in horizons:
        fwd_col = f"fwd_return_{h}d"
        if fwd_col not in df.columns:
            continue

        # Scale threshold by sqrt(horizon) — volatility scales with sqrt(time)
        scale = np.sqrt(h)
        threshold = adaptive_threshold(daily_returns, k=k, vol_window=vol_window) * scale

        # Classify
        label_col = f"label_{h}d"
        df[label_col] = 1  # default: flat
        df.loc[df[fwd_col] > threshold, label_col] = 2   # up
        df.loc[df[fwd_col] < -threshold, label_col] = 0  # down

        # Store threshold for reference
        df[f"threshold_{h}d"] = threshold

        # Log class distribution
        if not df[label_col].isna().all():
            counts = df[label_col].value_counts(normalize=True).sort_index()
            dist_str = " | ".join(
                f"{LABEL_MAP.get(i, i)}: {v:.1%}" for i, v in counts.items()
            )
            log.info("  %dd labels: %s", h, dist_str)

    return df


def add_targets(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
    k: float = 0.4,
    vol_window: int = 60,
) -> pd.DataFrame:
    """Add both continuous returns and categorical labels as targets.

    Adds columns:
      - fwd_return_{h}d: continuous forward return (regression target)
      - fwd_log_return_{h}d: log forward return
      - label_{h}d: 3-class label (classification target): 0=down, 1=flat, 2=up
      - threshold_{h}d: adaptive threshold used for labeling
    """
    log.info("Computing targets (k=%.2f, vol_window=%d) ...", k, vol_window)

    df = compute_forward_returns(df, horizons)
    df = compute_labels(df, horizons, k=k, vol_window=vol_window)

    return df


def get_target_columns(horizons: list[int] | None = None) -> dict[str, list[str]]:
    """Return the expected target column names for given horizons."""
    cfg = get_data_config()
    horizons = horizons or cfg["horizons"]

    return {
        "regression": [f"fwd_return_{h}d" for h in horizons],
        "log_regression": [f"fwd_log_return_{h}d" for h in horizons],
        "classification": [f"label_{h}d" for h in horizons],
        "thresholds": [f"threshold_{h}d" for h in horizons],
    }
