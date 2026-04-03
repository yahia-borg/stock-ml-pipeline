"""
Predictor service — loads trained models and generates predictions.

Serves as the backend logic for the FastAPI endpoints.
Handles model loading, caching, feature preparation, and prediction aggregation.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.config import ROOT_DIR, get_data_config, get_model_config, get_ticker_list
from src.utils.logger import get_logger

log = get_logger("inference.predictor")

MODELS_DIR = ROOT_DIR / os.environ.get("MODELS_DIR", "models")


class StockPredictor:
    """Loads all available models and provides a unified prediction interface."""

    def __init__(self):
        self._baseline_models = {}
        self._regime_model = None
        self._regime_mapping = None
        self._chronos = None
        self._ensemble = None
        self._loaded = False

    def load_models(self):
        """Load all available trained models from disk."""
        log.info("Loading models from %s ...", MODELS_DIR)

        # Baseline models (LightGBM / XGBoost)
        for model_file in MODELS_DIR.glob("*_*d.joblib"):
            name = model_file.stem  # e.g., "lgbm_5d"
            try:
                self._baseline_models[name] = joblib.load(model_file)
                log.info("  Loaded baseline: %s", name)
            except Exception as e:
                log.warning("  Failed to load %s: %s", name, e)

        # Also load latest fold models if no single model exists
        if not self._baseline_models:
            for model_file in sorted(MODELS_DIR.glob("*_*d_fold*.joblib")):
                name_parts = model_file.stem.rsplit("_fold", 1)
                name = name_parts[0]
                if name not in self._baseline_models:
                    try:
                        self._baseline_models[name] = joblib.load(model_file)
                        log.info("  Loaded baseline (fold): %s", name)
                    except Exception as e:
                        log.warning("  Failed to load %s: %s", name, e)

        # Regime model
        regime_path = MODELS_DIR / "hmm_regime.joblib"
        if regime_path.exists():
            try:
                data = joblib.load(regime_path)
                self._regime_model = data["model"]
                self._regime_mapping = data["state_mapping"]
                log.info("  Loaded regime model")
            except Exception as e:
                log.warning("  Failed to load regime: %s", e)

        # Ensemble
        ensemble_path = MODELS_DIR / "ensemble_meta.joblib"
        if ensemble_path.exists():
            try:
                from src.models.ensemble import StackingEnsemble
                self._ensemble = StackingEnsemble.load(ensemble_path)
                log.info("  Loaded ensemble")
            except Exception as e:
                log.warning("  Failed to load ensemble: %s", e)

        n_models = len(self._baseline_models) + (1 if self._regime_model else 0) + (1 if self._ensemble else 0)
        log.info("Models loaded: %d total", n_models)
        self._loaded = True

    def get_available_models(self) -> list[dict]:
        """List all loaded models with metadata."""
        models = []
        for name, model in self._baseline_models.items():
            parts = name.split("_")
            models.append({
                "name": name,
                "type": parts[0],
                "horizon": parts[1] if len(parts) > 1 else "unknown",
                "status": "loaded",
            })

        if self._regime_model:
            models.append({"name": "hmm_regime", "type": "regime", "horizon": "N/A", "status": "loaded"})
        if self._ensemble:
            models.append({"name": "ensemble", "type": "ensemble", "horizon": "5d", "status": "loaded"})

        return models

    def predict_ticker(self, ticker: str, horizon: int = 5) -> dict:
        """Generate prediction for a single ticker at a given horizon.

        Returns a dict with predictions from all available models.
        """
        if not self._loaded:
            self.load_models()

        result = {
            "ticker": ticker,
            "horizon": horizon,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predictions": {},
        }

        # Load latest features for this ticker
        features = self._load_latest_features(ticker)
        if features is None:
            result["error"] = "No feature data available"
            return result

        # Baseline model predictions
        model_key = None
        for name in [f"lgbm_{horizon}d", f"xgboost_{horizon}d"]:
            if name in self._baseline_models:
                try:
                    model = self._baseline_models[name]
                    feature_cols = self._get_feature_cols(features)
                    X = features[feature_cols].tail(1).fillna(0).values

                    pred = model.predict(X)[0]
                    proba = model.predict_proba(X)[0]

                    labels = ["down", "flat", "up"]
                    result["predictions"][name] = {
                        "direction": labels[int(pred)],
                        "label": int(pred),
                        "confidence": float(np.max(proba)),
                        "probabilities": {
                            "down": float(proba[0]),
                            "flat": float(proba[1]),
                            "up": float(proba[2]),
                        },
                    }
                except Exception as e:
                    log.warning("Prediction failed for %s on %s: %s", name, ticker, e)

        # Regime state
        if self._regime_model:
            try:
                regime_features = ["return_1d", "vol_20d", "vol_ratio"]
                available = [c for c in regime_features if c in features.columns]
                if available:
                    X_regime = features[available].dropna().tail(60).values
                    if len(X_regime) > 10:
                        states = self._regime_model.predict(X_regime)
                        current_state = self._regime_mapping.get(states[-1], states[-1])
                        regime_labels = {0: "bear", 1: "sideways", 2: "bull"}
                        result["regime"] = {
                            "state": int(current_state),
                            "label": regime_labels.get(current_state, "unknown"),
                        }
            except Exception as e:
                log.warning("Regime prediction failed: %s", e)

        # Best prediction (prefer ensemble > lgbm > xgboost)
        for pref in ["ensemble", f"lgbm_{horizon}d", f"xgboost_{horizon}d"]:
            if pref in result["predictions"]:
                result["best_prediction"] = result["predictions"][pref]
                result["best_model"] = pref
                break

        return result

    def predict_all_tickers(self, horizon: int = 5) -> list[dict]:
        """Generate predictions for all tickers."""
        tickers = get_ticker_list()
        results = []
        for ticker in tickers:
            result = self.predict_ticker(ticker, horizon)
            results.append(result)
        return results

    def _load_latest_features(self, ticker: str) -> pd.DataFrame | None:
        """Load the latest feature data for a ticker."""
        cfg = get_data_config()
        path = ROOT_DIR / cfg["storage"]["processed_dir"] / f"{ticker}_features.parquet"

        if path.exists():
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                if "time" in df.columns:
                    df = df.set_index("time")
                df.index = pd.to_datetime(df.index)
            return df

        # Try combined matrix
        combined_path = ROOT_DIR / cfg["storage"]["processed_dir"] / "feature_matrix.parquet"
        if combined_path.exists():
            df = pd.read_parquet(combined_path)
            if "ticker" in df.columns:
                df = df[df["ticker"] == ticker]
                if not df.empty:
                    return df

        return None

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Get feature columns (exclude targets, metadata, OHLCV)."""
        exclude_prefixes = ("fwd_", "label_", "threshold_")
        exclude_exact = {"ticker", "time", "open", "high", "low", "close", "volume", "adj_close"}
        return [
            c for c in df.columns
            if c not in exclude_exact
            and not any(c.startswith(p) for p in exclude_prefixes)
        ]


def get_model_results() -> list[dict]:
    """Load all model evaluation results for the dashboard."""
    results = []
    for path in sorted(MODELS_DIR.glob("*_results.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            data["file"] = path.name
            results.append(data)
        except Exception:
            continue
    return results


# Singleton instance
_predictor: StockPredictor | None = None


def get_predictor() -> StockPredictor:
    """Get or create the singleton predictor."""
    global _predictor
    if _predictor is None:
        _predictor = StockPredictor()
        _predictor.load_models()
    return _predictor
