"""
Foundation model forecasters — zero-shot time series prediction.

Supported models:
  - Amazon Chronos-2 (primary): 120M params, multivariate, covariate-aware, 300 forecasts/sec
  - Salesforce Moirai 2.0 (optional): quantile outputs, any-frequency

These models require NO training on our data — they predict out-of-the-box.
This gives us a strong baseline with zero training cost.

Models auto-download from HuggingFace on first use (~800MB for Chronos-2).
Cache location: ~/.cache/huggingface/ (override with HF_HOME env var)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.config import ROOT_DIR, get_model_config, get_data_config
from src.utils.logger import get_logger

log = get_logger("models.foundation")

MODELS_DIR = ROOT_DIR / os.environ.get("MODELS_DIR", "models")


# ── Chronos-2 ─────────────────────────────────────────────

class ChronosForecaster:
    """Amazon Chronos-2 zero-shot forecaster.

    Chronos-2 is a foundation model for time series that:
    - Works zero-shot (no fine-tuning needed)
    - Supports covariates (feed technical indicators alongside price)
    - Outputs probabilistic forecasts (quantiles)
    - Runs at 300+ forecasts/sec on GPU
    """

    def __init__(self, model_id: str | None = None, device: str = "auto"):
        cfg = get_model_config()["foundation"]["chronos2"]
        self.model_id = model_id or cfg["model_id"]
        self.prediction_length = cfg["prediction_length"]
        self.context_length = cfg["context_length"]

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._pipeline = None

    def _load(self):
        """Lazy-load the Chronos pipeline (downloads model on first call)."""
        if self._pipeline is not None:
            return

        try:
            from chronos import ChronosPipeline

            log.info("Loading Chronos-2 model: %s (device: %s) ...", self.model_id, self.device)
            log.info("  First run will download ~800MB from HuggingFace (cached after)")

            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=torch.float32,
            )
            log.info("Chronos-2 loaded successfully")

        except ImportError:
            raise ImportError(
                "chronos-forecasting not installed. Install with: "
                "pip install chronos-forecasting"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Chronos-2 ({self.model_id}): {e}")

    def predict(
        self,
        context: np.ndarray | pd.Series,
        prediction_length: int | None = None,
        num_samples: int = 20,
    ) -> dict:
        """Generate probabilistic forecast from a price/return series.

        Args:
            context: Historical values (e.g., close prices or returns).
                     Uses the last `context_length` values.
            prediction_length: Number of steps ahead to predict.
            num_samples: Number of sample paths for uncertainty estimation.

        Returns:
            Dict with keys: mean, median, quantile_10, quantile_90, samples
        """
        self._load()

        prediction_length = prediction_length or self.prediction_length

        # Convert to tensor
        if isinstance(context, pd.Series):
            context = context.dropna().values
        context = np.asarray(context, dtype=np.float32)

        # Truncate to context length
        if len(context) > self.context_length:
            context = context[-self.context_length:]

        context_tensor = torch.tensor(context).unsqueeze(0)  # [1, seq_len]

        # Generate forecast
        with torch.no_grad():
            forecast = self._pipeline.predict(
                context_tensor,
                prediction_length=prediction_length,
                num_samples=num_samples,
            )

        # forecast shape: [1, num_samples, prediction_length]
        samples = forecast[0].numpy()  # [num_samples, prediction_length]

        return {
            "mean": samples.mean(axis=0),
            "median": np.median(samples, axis=0),
            "quantile_10": np.percentile(samples, 10, axis=0),
            "quantile_90": np.percentile(samples, 90, axis=0),
            "std": samples.std(axis=0),
            "samples": samples,
        }

    def predict_direction(
        self,
        prices: np.ndarray | pd.Series,
        horizon: int = 5,
        num_samples: int = 20,
    ) -> dict:
        """Predict price direction (up/flat/down) from historical prices.

        Converts Chronos price-level forecast into directional probabilities
        using the forecast distribution.

        Args:
            prices: Historical close prices
            horizon: Days ahead to predict direction
            num_samples: Samples for probability estimation

        Returns:
            Dict with direction, probabilities, confidence, and forecast stats
        """
        forecast = self.predict(prices, prediction_length=horizon, num_samples=num_samples)

        if isinstance(prices, pd.Series):
            current_price = prices.dropna().iloc[-1]
        else:
            current_price = prices[~np.isnan(prices)][-1]

        # Get forecasted prices at the target horizon
        horizon_prices = forecast["samples"][:, -1]  # last step of each sample
        horizon_returns = (horizon_prices - current_price) / current_price

        # Adaptive threshold based on recent volatility
        if isinstance(prices, pd.Series):
            recent_returns = prices.pct_change().dropna().tail(60)
        else:
            recent_returns = np.diff(prices) / prices[:-1]
            recent_returns = recent_returns[-60:]
        vol = np.std(recent_returns) * np.sqrt(horizon)
        flat_threshold = 0.4 * vol  # same k factor as our target engineering

        # Compute directional probabilities from sample distribution
        prob_up = (horizon_returns > flat_threshold).mean()
        prob_down = (horizon_returns < -flat_threshold).mean()
        prob_flat = 1.0 - prob_up - prob_down

        probs = {"down": float(prob_down), "flat": float(prob_flat), "up": float(prob_up)}
        direction = max(probs, key=probs.get)
        confidence = probs[direction]

        # Map to label integers
        label_map = {"down": 0, "flat": 1, "up": 2}

        return {
            "direction": direction,
            "label": label_map[direction],
            "confidence": confidence,
            "probabilities": probs,
            "expected_return": float(horizon_returns.mean()),
            "forecast_std": float(horizon_returns.std()),
            "forecast_mean_price": float(forecast["mean"][-1]),
            "forecast_median_price": float(forecast["median"][-1]),
            "current_price": float(current_price),
        }

    def predict_batch(
        self,
        price_series_list: list[np.ndarray | pd.Series],
        horizon: int = 5,
        num_samples: int = 20,
    ) -> list[dict]:
        """Predict direction for multiple series (batch processing)."""
        self._load()
        results = []

        for i, prices in enumerate(price_series_list):
            try:
                result = self.predict_direction(prices, horizon=horizon, num_samples=num_samples)
                results.append(result)
            except Exception as e:
                log.warning("Chronos prediction failed for series %d: %s", i, e)
                results.append({
                    "direction": "flat", "label": 1, "confidence": 0.0,
                    "probabilities": {"down": 0.33, "flat": 0.34, "up": 0.33},
                    "expected_return": 0.0, "forecast_std": 0.0,
                })

            if (i + 1) % 50 == 0:
                log.info("  Chronos batch progress: %d/%d", i + 1, len(price_series_list))

        return results


# ── Moirai 2.0 (Optional) ─────────────────────────────────

class MoiraiForecaster:
    """Salesforce Moirai 2.0 zero-shot forecaster.

    Moirai provides quantile-based probabilistic forecasts.
    Requires the `uni2ts` package: pip install "stock-forecast[moirai]"
    """

    def __init__(self, model_id: str | None = None, device: str = "auto"):
        cfg = get_model_config()["foundation"]["moirai2"]
        self.model_id = model_id or cfg["model_id"]
        self.prediction_length = cfg["prediction_length"]
        self.context_length = cfg["context_length"]
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu"
        self._model = None

    def _load(self):
        if self._model is not None:
            return

        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

            log.info("Loading Moirai 2.0: %s ...", self.model_id)
            module = MoiraiModule.from_pretrained(self.model_id)
            self._model = MoiraiForecast(
                module=module,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                patch_size="auto",
                num_samples=100,
            )
            log.info("Moirai 2.0 loaded")
        except ImportError:
            raise ImportError(
                "uni2ts not installed. Install with: pip install uni2ts"
            )

    def predict_direction(
        self,
        prices: np.ndarray | pd.Series,
        horizon: int = 5,
    ) -> dict:
        """Predict direction using Moirai quantile forecasts."""
        self._load()

        # Moirai uses GluonTS-style datasets — simplified interface here
        if isinstance(prices, pd.Series):
            values = prices.dropna().values
        else:
            values = prices[~np.isnan(prices)]

        current_price = values[-1]

        try:
            # Use the model's predict method
            context = torch.tensor(values[-self.context_length:], dtype=torch.float32).unsqueeze(0)
            forecast = self._model(context)

            # Extract quantile predictions
            samples = forecast.numpy()[0]  # [num_samples, prediction_length]
            horizon_prices = samples[:, min(horizon - 1, samples.shape[1] - 1)]
            horizon_returns = (horizon_prices - current_price) / current_price

            vol = np.std(np.diff(values) / values[:-1]) * np.sqrt(horizon)
            threshold = 0.4 * vol

            prob_up = (horizon_returns > threshold).mean()
            prob_down = (horizon_returns < -threshold).mean()
            prob_flat = 1.0 - prob_up - prob_down

            probs = {"down": float(prob_down), "flat": float(prob_flat), "up": float(prob_up)}
            direction = max(probs, key=probs.get)

            return {
                "direction": direction,
                "label": {"down": 0, "flat": 1, "up": 2}[direction],
                "confidence": float(probs[direction]),
                "probabilities": probs,
                "expected_return": float(horizon_returns.mean()),
                "forecast_std": float(horizon_returns.std()),
            }
        except Exception as e:
            log.warning("Moirai prediction failed: %s", e)
            return {
                "direction": "flat", "label": 1, "confidence": 0.0,
                "probabilities": {"down": 0.33, "flat": 0.34, "up": 0.33},
            }


# ── Factory ────────────────────────────────────────────────

def get_forecaster(model_name: str = "chronos2") -> ChronosForecaster | MoiraiForecaster:
    """Get a foundation model forecaster by name."""
    if model_name == "chronos2":
        return ChronosForecaster()
    elif model_name == "moirai2":
        return MoiraiForecaster()
    else:
        raise ValueError(f"Unknown foundation model: {model_name}. Use 'chronos2' or 'moirai2'.")
