"""
Drift detection — monitors model performance and data distribution changes.

Three types of drift:
  1. Performance drift: rolling accuracy drops below threshold
  2. Data drift: feature distributions shift (PSI - Population Stability Index)
  3. Regime drift: HMM detects state transition

When drift is detected, triggers retraining recommendation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.utils.config import get_model_config
from src.utils.db import check_connection, read_sql
from src.utils.logger import get_logger

log = get_logger("training.drift")


@dataclass
class DriftReport:
    timestamp: str
    performance_drift: bool
    data_drift: bool
    regime_drift: bool
    rolling_accuracy: float
    accuracy_threshold: float
    psi_scores: dict[str, float]
    current_regime: str | None
    previous_regime: str | None
    recommendation: str

    @property
    def any_drift(self) -> bool:
        return self.performance_drift or self.data_drift or self.regime_drift


def check_performance_drift(
    window_days: int | None = None,
    threshold: float | None = None,
) -> tuple[bool, float]:
    """Check if model accuracy has dropped below threshold over a rolling window."""
    cfg = get_model_config()["monitoring"]
    window_days = window_days or cfg["drift_window_days"]
    threshold = threshold or cfg["accuracy_threshold"]

    if not check_connection():
        return False, 0.0

    df = read_sql(f"""
        SELECT predicted_direction, actual_direction
        FROM predictions
        WHERE actual_direction IS NOT NULL
          AND time > NOW() - INTERVAL '{window_days} days'
    """)

    if len(df) < 20:
        log.info("Not enough evaluated predictions (%d) for drift check", len(df))
        return False, 0.0

    accuracy = (df["predicted_direction"] == df["actual_direction"]).mean()
    drift = accuracy < threshold

    if drift:
        log.warning("PERFORMANCE DRIFT: accuracy %.4f < threshold %.4f (window=%dd, n=%d)",
                     accuracy, threshold, window_days, len(df))
    else:
        log.info("Performance OK: accuracy %.4f >= threshold %.4f", accuracy, threshold)

    return drift, accuracy


def compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions.

    PSI < 0.1: no significant change
    PSI 0.1-0.25: moderate change
    PSI > 0.25: significant change — retrain recommended
    """
    # Bin the data
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        bins + 1,
    )

    ref_counts = np.histogram(reference.dropna(), bins=breakpoints)[0]
    cur_counts = np.histogram(current.dropna(), bins=breakpoints)[0]

    # Normalize to proportions
    ref_pct = (ref_counts + 1) / (ref_counts.sum() + bins)  # add 1 to avoid zero
    cur_pct = (cur_counts + 1) / (cur_counts.sum() + bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def check_data_drift(
    feature_cols: list[str] | None = None,
    window_days: int = 30,
    psi_threshold: float = 0.25,
) -> tuple[bool, dict[str, float]]:
    """Check if feature distributions have shifted significantly.

    Compares recent data (last window_days) against the full training history.
    """
    from src.utils.config import ROOT_DIR, get_data_config

    cfg = get_data_config()
    path = ROOT_DIR / cfg["storage"]["processed_dir"] / "feature_matrix.parquet"

    if not path.exists():
        log.warning("No feature matrix found for drift check")
        return False, {}

    df = pd.read_parquet(path)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    cutoff = df.index.max() - pd.Timedelta(days=window_days)
    reference = df[df.index < cutoff]
    current = df[df.index >= cutoff]

    if len(current) < 10 or len(reference) < 100:
        return False, {}

    # Check key features
    if feature_cols is None:
        # Use top features by variance
        numeric = df.select_dtypes(include=[np.number])
        exclude = [c for c in numeric.columns if c.startswith(("fwd_", "label_", "threshold_"))]
        candidates = [c for c in numeric.columns if c not in exclude]
        feature_cols = sorted(candidates, key=lambda c: numeric[c].var(), reverse=True)[:20]

    psi_scores = {}
    drifted = False
    for col in feature_cols:
        if col in reference.columns and col in current.columns:
            psi = compute_psi(reference[col], current[col])
            psi_scores[col] = round(psi, 4)
            if psi > psi_threshold:
                drifted = True
                log.warning("DATA DRIFT in '%s': PSI=%.4f > %.4f", col, psi, psi_threshold)

    if not drifted:
        log.info("Data drift check passed (%d features, max PSI=%.4f)",
                 len(psi_scores), max(psi_scores.values()) if psi_scores else 0)

    return drifted, psi_scores


def check_regime_drift() -> tuple[bool, str | None, str | None]:
    """Check if the HMM regime has recently changed."""
    import os
    from pathlib import Path

    models_dir = Path(os.environ.get("MODELS_DIR", "models"))
    regime_path = models_dir / "hmm_regime.joblib"

    if not regime_path.exists():
        return False, None, None

    try:
        import joblib
        from src.features.pipeline import load_price_data
        from src.features.technical import add_technical_features
        from src.models.regime import predict_regimes

        data = joblib.load(regime_path)
        model = data["model"]
        mapping = data["state_mapping"]

        df = load_price_data("SPY")
        if df.empty:
            return False, None, None

        df = add_technical_features(df)
        df = predict_regimes(df, model, mapping)

        labels = {0: "bear", 1: "sideways", 2: "bull"}
        current = labels.get(int(df["regime_state"].iloc[-1]), "unknown")
        previous = labels.get(int(df["regime_state"].iloc[-20]), "unknown") if len(df) > 20 else current

        drift = current != previous
        if drift:
            log.warning("REGIME DRIFT: %s -> %s", previous, current)
        else:
            log.info("Regime stable: %s", current)

        return drift, current, previous

    except Exception as e:
        log.warning("Regime check failed: %s", e)
        return False, None, None


def full_drift_check() -> DriftReport:
    """Run all drift checks and produce a report."""
    log.info("=" * 50)
    log.info("Running full drift check...")
    log.info("=" * 50)

    cfg = get_model_config()["monitoring"]

    perf_drift, accuracy = check_performance_drift()
    data_drift, psi_scores = check_data_drift()
    regime_drift, current_regime, prev_regime = check_regime_drift()

    # Recommendation
    if perf_drift and regime_drift:
        rec = "RETRAIN URGENTLY: performance degraded + regime change"
    elif perf_drift:
        rec = "RETRAIN: performance below threshold"
    elif regime_drift and cfg.get("regime_triggered_retrain", True):
        rec = "RETRAIN: regime transition detected"
    elif data_drift:
        rec = "MONITOR: data distribution shifted, consider retraining"
    else:
        rec = "OK: no drift detected"

    report = DriftReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        performance_drift=perf_drift,
        data_drift=data_drift,
        regime_drift=regime_drift,
        rolling_accuracy=accuracy,
        accuracy_threshold=cfg["accuracy_threshold"],
        psi_scores=psi_scores,
        current_regime=current_regime,
        previous_regime=prev_regime,
        recommendation=rec,
    )

    log.info("")
    log.info("Drift Report:")
    log.info("  Performance drift: %s (accuracy=%.4f, threshold=%.4f)",
             report.performance_drift, report.rolling_accuracy, report.accuracy_threshold)
    log.info("  Data drift:        %s (%d features checked)", report.data_drift, len(report.psi_scores))
    log.info("  Regime drift:      %s (%s -> %s)", report.regime_drift, report.previous_regime, report.current_regime)
    log.info("  Recommendation:    %s", report.recommendation)

    return report


def main():
    parser = argparse.ArgumentParser(description="Drift detection")
    parser.add_argument("--perf-only", action="store_true", help="Only check performance drift")
    parser.add_argument("--data-only", action="store_true", help="Only check data drift")
    parser.add_argument("--regime-only", action="store_true", help="Only check regime drift")
    args = parser.parse_args()

    if args.perf_only:
        drift, acc = check_performance_drift()
        print(f"Performance drift: {drift} (accuracy: {acc:.4f})")
    elif args.data_only:
        drift, scores = check_data_drift()
        print(f"Data drift: {drift}")
        for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {k}: PSI={v:.4f}")
    elif args.regime_only:
        drift, cur, prev = check_regime_drift()
        print(f"Regime drift: {drift} ({prev} -> {cur})")
    else:
        report = full_drift_check()
        print(f"\nRecommendation: {report.recommendation}")


if __name__ == "__main__":
    main()
