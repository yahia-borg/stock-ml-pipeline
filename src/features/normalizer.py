"""
Feature normalization and selection pipeline.

Normalization strategy (from research):
  - Rolling percentile rank (default) — robust to outliers and fat tails
  - Expanding Z-score — for tree models where magnitude matters
  - Both fitted WITHIN walk-forward folds to prevent leakage

Feature selection (3-stage):
  1. Fast filter: mutual information / near-zero variance removal
  2. SHAP-based: LightGBM SHAP values, keep top features
  3. Stability selection: only keep features that rank highly across multiple time periods
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.utils.logger import get_logger

log = get_logger("features.normalizer")


# ── Normalization Methods ─────────────────────────────────

class RollingPercentileRank:
    """Normalize each feature to its percentile rank within a rolling/expanding window.

    This is the recommended default for financial features because:
    - Robust to outliers and fat tails
    - Enforces uniform distribution
    - Handles non-stationarity naturally
    - Works for both tree and neural net models
    """

    def __init__(self, min_periods: int = 252, expanding: bool = True):
        self.min_periods = min_periods
        self.expanding = expanding

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features to rolling percentile ranks in [0, 1]."""
        result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

        for col in df.columns:
            series = df[col]
            if self.expanding:
                result[col] = series.expanding(min_periods=self.min_periods).rank(pct=True)
            else:
                result[col] = series.rolling(
                    window=self.min_periods * 2,
                    min_periods=self.min_periods,
                ).rank(pct=True)

        log.info("Percentile rank normalization applied (%d cols, min_periods=%d)",
                 len(df.columns), self.min_periods)
        return result


class ExpandingZScore:
    """Expanding window Z-score normalization.

    Uses exponential weighting (halflife) to adapt to regime changes
    while using all available history for stability.
    """

    def __init__(self, min_periods: int = 252, halflife: int = 126):
        self.min_periods = min_periods
        self.halflife = halflife

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

        for col in df.columns:
            series = df[col]
            ewm_mean = series.ewm(halflife=self.halflife, min_periods=self.min_periods).mean()
            ewm_std = series.ewm(halflife=self.halflife, min_periods=self.min_periods).std()
            result[col] = (series - ewm_mean) / ewm_std.replace(0, np.nan)

        # Clip extreme Z-scores to ±5
        result = result.clip(-5, 5)

        log.info("Expanding Z-score normalization applied (%d cols)", len(df.columns))
        return result


# ── Feature Selection ─────────────────────────────────────

def remove_low_variance(
    df: pd.DataFrame,
    threshold: float = 0.01,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove features with near-zero variance."""
    variances = df.var()
    keep = variances[variances > threshold].index.tolist()
    removed = [c for c in df.columns if c not in keep]

    if removed:
        log.info("Removed %d low-variance features: %s", len(removed), removed[:5])
    return df[keep], removed


def remove_high_correlation(
    df: pd.DataFrame,
    threshold: float = 0.95,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove one of each pair of highly correlated features."""
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > threshold].tolist()
        if high_corr:
            to_drop.add(col)

    removed = list(to_drop)
    keep = [c for c in df.columns if c not in to_drop]

    if removed:
        log.info("Removed %d highly correlated features (r>%.2f): %s",
                 len(removed), threshold, removed[:5])
    return df[keep], removed


def mutual_info_filter(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int | None = None,
    threshold: float = 0.001,
) -> tuple[list[str], pd.Series]:
    """Filter features by mutual information with target.

    Args:
        X: Feature matrix
        y: Target labels (classification)
        top_k: Keep top K features (if None, use threshold)
        threshold: Minimum MI score to keep
    """
    # Drop rows with NaN in target
    mask = y.notna() & X.notna().all(axis=1)
    X_clean = X.loc[mask].fillna(0)
    y_clean = y.loc[mask]

    if len(X_clean) < 100:
        log.warning("Too few samples for MI filter (%d)", len(X_clean))
        return list(X.columns), pd.Series()

    mi_scores = mutual_info_classif(X_clean, y_clean, random_state=42, n_neighbors=5)
    mi = pd.Series(mi_scores, index=X_clean.columns).sort_values(ascending=False)

    if top_k:
        keep = mi.head(top_k).index.tolist()
    else:
        keep = mi[mi > threshold].index.tolist()

    log.info("MI filter: %d/%d features kept (top 5: %s)",
             len(keep), len(X.columns),
             list(mi.head(5).items()))
    return keep, mi


def shap_feature_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    top_k: int = 50,
) -> tuple[list[str], pd.Series]:
    """Select features using SHAP values from a quick LightGBM model.

    Must be called WITHIN a walk-forward fold to prevent leakage.
    """
    import shap
    from lightgbm import LGBMClassifier

    # Quick model for feature importance
    model = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        verbose=-1,
        random_state=42,
    )

    mask_train = y_train.notna()
    mask_val = X_val.notna().all(axis=1)

    X_tr = X_train.loc[mask_train].fillna(0)
    y_tr = y_train.loc[mask_train]
    X_v = X_val.loc[mask_val].fillna(0)

    model.fit(X_tr, y_tr)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_v)

    # For multi-class, shap_values is a list of arrays
    if isinstance(shap_values, list):
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    importance = pd.Series(mean_shap, index=X_v.columns).sort_values(ascending=False)
    keep = importance.head(top_k).index.tolist()

    log.info("SHAP selection: top %d features (best: %s = %.4f)",
             top_k, importance.index[0], importance.iloc[0])
    return keep, importance


def stability_selection(
    feature_ranks: list[pd.Series],
    min_appearances: int = 3,
    top_n: int = 50,
) -> list[str]:
    """Keep only features that rank in top_n across multiple time periods.

    Args:
        feature_ranks: List of pd.Series with feature importance from different periods
        min_appearances: Minimum number of periods a feature must appear in top_n
        top_n: What counts as "top" in each period
    """
    counts = {}
    for ranks in feature_ranks:
        top_features = ranks.nlargest(top_n).index.tolist()
        for f in top_features:
            counts[f] = counts.get(f, 0) + 1

    stable = [f for f, c in counts.items() if c >= min_appearances]
    stable_sorted = sorted(stable, key=lambda f: counts[f], reverse=True)

    log.info("Stability selection: %d features appear in top-%d across %d+ periods",
             len(stable_sorted), top_n, min_appearances)
    return stable_sorted


# ── Full Pipeline ─────────────────────────────────────────

def normalize_features(
    df: pd.DataFrame,
    method: str = "percentile_rank",
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Normalize all feature columns using the specified method.

    Args:
        df: Feature matrix
        method: "percentile_rank" or "zscore"
        exclude_cols: Columns to skip (e.g., targets, metadata)
    """
    exclude = set(exclude_cols or [])
    feature_cols = [c for c in df.columns if c not in exclude]
    meta_cols = [c for c in df.columns if c in exclude]

    features = df[feature_cols]

    if method == "percentile_rank":
        normalizer = RollingPercentileRank(min_periods=252)
    elif method == "zscore":
        normalizer = ExpandingZScore(min_periods=252, halflife=126)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    normalized = normalizer.transform(features)

    # Rejoin with excluded columns
    result = pd.concat([df[meta_cols], normalized], axis=1)
    return result
