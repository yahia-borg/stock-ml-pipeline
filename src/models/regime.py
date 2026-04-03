"""
Hidden Markov Model regime detector — identifies market regimes (bull/bear/sideways).

Fits a GaussianHMM on daily returns + volatility + volume ratio.
States are labeled post-hoc by mean return: highest = bull, lowest = bear, middle = sideways.

Output:
  - regime_state: integer state label (0=bear, 1=sideways, 2=bull)
  - regime_proba_*: probability of being in each state
  - regime_duration: consecutive days in current regime
  - regime_transition: 1 on days when regime changes

Used as:
  - Feature for downstream models
  - Training filter (weight recent-regime samples higher)
  - Risk gate (reduce confidence during regime transitions)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from src.utils.config import ROOT_DIR, get_model_config
from src.utils.logger import get_logger

log = get_logger("models.regime")

MODELS_DIR = ROOT_DIR / os.environ.get("MODELS_DIR", "models")
REGIME_LABELS = {0: "bear", 1: "sideways", 2: "bull"}


def _prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Extract HMM input features from price DataFrame."""
    cfg = get_model_config()["regime"]
    feature_names = cfg["features"]

    # Map config names to DataFrame columns
    feature_map = {
        "daily_return": "return_1d",
        "volatility_20d": "vol_20d",
        "volume_ratio": "vol_ratio",
    }

    cols = []
    for f in feature_names:
        col = feature_map.get(f, f)
        if col in df.columns:
            cols.append(col)
        else:
            log.warning("Regime feature '%s' (col '%s') not found in data", f, col)

    if not cols:
        raise ValueError(f"No regime features found. Need: {feature_names}")

    features = df[cols].dropna()
    return features.values, features.index


def fit_regime_model(df: pd.DataFrame) -> tuple[GaussianHMM, dict]:
    """Fit the HMM regime model on price data.

    Args:
        df: DataFrame with at minimum return_1d, vol_20d, vol_ratio columns

    Returns:
        (fitted model, state_mapping dict)
    """
    cfg = get_model_config()["regime"]

    X, idx = _prepare_features(df)
    log.info("Fitting HMM: %d observations, %d features", X.shape[0], X.shape[1])

    model = GaussianHMM(
        n_components=cfg["n_states"],
        covariance_type=cfg["covariance_type"],
        n_iter=cfg["n_iter"],
        random_state=get_model_config()["training"]["seed"],
    )

    model.fit(X)
    log.info("HMM converged: %s (score: %.2f)", model.monitor_.converged, model.score(X))

    # Decode states
    states = model.predict(X)

    # Label states by mean return: bull = highest, bear = lowest
    state_returns = {}
    return_col_idx = 0  # daily_return is first feature
    for state in range(cfg["n_states"]):
        mask = states == state
        state_returns[state] = X[mask, return_col_idx].mean()

    # Sort states by mean return
    sorted_states = sorted(state_returns.keys(), key=lambda s: state_returns[s])
    state_mapping = {}
    for new_label, old_state in enumerate(sorted_states):
        state_mapping[old_state] = new_label

    log.info("Regime states:")
    for old, new in state_mapping.items():
        label = REGIME_LABELS.get(new, str(new))
        count = (states == old).sum()
        mean_ret = state_returns[old] * 252  # annualize
        log.info("  State %d -> %s: %d days (%.1f%%), ann. return: %.1f%%",
                 old, label, count, count / len(states) * 100, mean_ret * 100)

    return model, state_mapping


def predict_regimes(
    df: pd.DataFrame,
    model: GaussianHMM,
    state_mapping: dict,
) -> pd.DataFrame:
    """Predict regime states and add regime features to DataFrame.

    Adds columns:
      - regime_state: 0=bear, 1=sideways, 2=bull
      - regime_proba_bear/sideways/bull: state probabilities
      - regime_duration: consecutive days in current regime
      - regime_transition: 1 on regime change days
    """
    df = df.copy()
    X, idx = _prepare_features(df)

    # Predict states and probabilities
    raw_states = model.predict(X)
    raw_proba = model.predict_proba(X)

    # Remap to sorted labels
    mapped_states = np.array([state_mapping.get(s, s) for s in raw_states])

    # Create a series aligned to the DataFrame index
    state_series = pd.Series(np.nan, index=df.index)
    state_series.loc[idx] = mapped_states
    df["regime_state"] = state_series.ffill().astype("Int64")

    # State probabilities
    for new_label, label_name in REGIME_LABELS.items():
        proba_series = pd.Series(np.nan, index=df.index)
        # Find which original state maps to this label
        for old_state, new_state in state_mapping.items():
            if new_state == new_label:
                proba_series.loc[idx] = raw_proba[:, old_state]
                break
        df[f"regime_proba_{label_name}"] = proba_series.ffill()

    # Regime duration (consecutive days in same state)
    state_change = df["regime_state"] != df["regime_state"].shift(1)
    df["regime_transition"] = state_change.fillna(False).astype(int)
    df["regime_duration"] = state_change.fillna(False).groupby(
        state_change.fillna(False).cumsum()
    ).cumcount() + 1

    log.info("Regime prediction complete: %d rows", len(df))
    return df


def save_model(model: GaussianHMM, state_mapping: dict, path: Path | None = None) -> Path:
    """Save fitted regime model to disk."""
    path = path or MODELS_DIR / "hmm_regime.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "state_mapping": state_mapping}, path)
    log.info("Saved regime model to %s", path)
    return path


def load_model(path: Path | None = None) -> tuple[GaussianHMM, dict]:
    """Load a previously fitted regime model."""
    path = path or MODELS_DIR / "hmm_regime.joblib"
    data = joblib.load(path)
    log.info("Loaded regime model from %s", path)
    return data["model"], data["state_mapping"]


def fit_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: fit the HMM and add regime features in one call."""
    model, mapping = fit_regime_model(df)
    save_model(model, mapping)
    return predict_regimes(df, model, mapping)


def main():
    parser = argparse.ArgumentParser(description="HMM Regime Detection")
    parser.add_argument("--ticker", default="SPY", help="Ticker to fit on (default: SPY)")
    parser.add_argument("--plot", action="store_true", help="Show regime plot")
    args = parser.parse_args()

    from src.features.pipeline import load_price_data
    from src.features.technical import add_technical_features

    # Load and prepare data
    df = load_price_data(args.ticker)
    if df.empty:
        print(f"No data for {args.ticker}. Run `make backfill` first.")
        return

    df = add_technical_features(df)
    model, mapping = fit_regime_model(df)
    save_model(model, mapping)

    df = predict_regimes(df, model, mapping)

    # Summary
    print(f"\nRegime detection for {args.ticker}:")
    print(f"  Total days: {len(df)}")
    for state, label in REGIME_LABELS.items():
        count = (df["regime_state"] == state).sum()
        pct = count / len(df) * 100
        print(f"  {label:>8}: {count:>5} days ({pct:.1f}%)")

    transitions = df["regime_transition"].sum()
    avg_duration = df.groupby(
        (df["regime_state"] != df["regime_state"].shift()).cumsum()
    )["regime_duration"].max().mean()
    print(f"  Transitions: {transitions}")
    print(f"  Avg regime duration: {avg_duration:.0f} days")


if __name__ == "__main__":
    main()
