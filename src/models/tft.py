"""
Temporal Fusion Transformer model wrapper.

TFT is purpose-built for multi-horizon time series forecasting with:
  - Static covariates (ticker, sector)
  - Known future inputs (calendar features)
  - Unknown observed inputs (price, volume, technicals, macro, sentiment)
  - Built-in interpretability (variable selection + attention weights)

Uses pytorch-forecasting library which handles the complex data preparation.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from src.utils.config import ROOT_DIR, get_model_config
from src.utils.logger import get_logger

log = get_logger("models.tft")

MODELS_DIR = ROOT_DIR / os.environ.get("MODELS_DIR", "models")

# ── Covariate classification ──────────────────────────────
# These define which features are static, known-in-advance, or observed-only

STATIC_CATEGORICALS = ["ticker"]

# Features known at prediction time (calendar-based)
TIME_VARYING_KNOWN_CATEGORICALS = [
    "is_opex_day", "is_opex_week", "is_quad_witching", "is_day_after_opex",
    "is_quarter_end_week", "is_quarter_start_week",
    "is_pre_holiday", "is_post_holiday",
    "is_monday", "is_friday", "sell_in_may",
]

TIME_VARYING_KNOWN_REALS = [
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "days_to_opex", "days_to_quarter_end",
]

# Features we only know once the day has passed
# (everything else — technicals, macro, sentiment, prices)
# These are auto-detected as: all numeric columns NOT in known/static/target lists

# Columns to exclude from features entirely
EXCLUDE_COLS = {
    "open", "high", "low", "close", "volume", "adj_close",
    "time", "ticker",
}

# Target and forward-looking columns (never used as input)
TARGET_PREFIXES = ("fwd_", "label_", "threshold_")


def get_tft_config() -> dict:
    """Get TFT hyperparameters from model_config.yaml."""
    return get_model_config()["tft"]


def classify_features(columns: list[str]) -> dict:
    """Classify DataFrame columns into TFT covariate types.

    Returns dict with keys:
      - static_categoricals
      - time_varying_known_categoricals
      - time_varying_known_reals
      - time_varying_unknown_categoricals
      - time_varying_unknown_reals
      - targets (excluded from features)
    """
    targets = [c for c in columns if any(c.startswith(p) for p in TARGET_PREFIXES)]
    excluded = EXCLUDE_COLS | set(targets)

    known_cats = [c for c in TIME_VARYING_KNOWN_CATEGORICALS if c in columns]
    known_reals = [c for c in TIME_VARYING_KNOWN_REALS if c in columns]
    static_cats = [c for c in STATIC_CATEGORICALS if c in columns]

    known_set = set(known_cats + known_reals + static_cats)

    # Everything else that's numeric and not excluded = unknown real
    unknown_reals = [
        c for c in columns
        if c not in excluded and c not in known_set
    ]

    # Split unknown into categoricals (binary/integer flags) vs reals
    unknown_cats = []
    unknown_reals_final = []
    for col in unknown_reals:
        # Heuristic: if column name suggests binary/categorical
        if col.startswith(("is_", "regime_state", "vix_high_regime", "vix_extreme",
                           "near_52w_high", "ema_20_50_cross", "ema_50_200_cross",
                           "breadth_thrust", "rate_hiking", "rate_cutting",
                           "recession_signal", "dual_inversion")):
            unknown_cats.append(col)
        else:
            unknown_reals_final.append(col)

    result = {
        "static_categoricals": static_cats,
        "time_varying_known_categoricals": known_cats,
        "time_varying_known_reals": known_reals,
        "time_varying_unknown_categoricals": unknown_cats,
        "time_varying_unknown_reals": unknown_reals_final,
        "targets": targets,
    }

    log.info("Feature classification:")
    for key, cols in result.items():
        log.info("  %-40s %d cols", key, len(cols))

    return result


def build_tft_model(
    n_features: int,
    output_size: int = 3,
    **override_params,
):
    """Build a TFT model from config with optional parameter overrides.

    This creates the pytorch-forecasting TemporalFusionTransformer.
    Requires a TimeSeriesDataSet to be passed during actual instantiation.
    Returns config dict for deferred model creation.
    """
    cfg = get_tft_config()
    cfg.update(override_params)

    model_params = {
        "hidden_size": cfg["hidden_size"],
        "attention_head_size": cfg["attention_head_size"],
        "dropout": cfg["dropout"],
        "hidden_continuous_size": cfg["hidden_continuous_size"],
        "output_size": output_size,  # 3 classes or quantiles
        "learning_rate": cfg["learning_rate"],
        "reduce_on_plateau_patience": cfg["patience"] // 2,
        "log_interval": 10,
        "log_val_interval": 1,
    }

    log.info("TFT config: hidden=%d, heads=%d, dropout=%.2f, lr=%.4f",
             model_params["hidden_size"],
             cfg["attention_head_size"],
             model_params["dropout"],
             model_params["learning_rate"])

    return model_params


def save_tft(model, path: Path | None = None) -> Path:
    """Save TFT model checkpoint."""
    path = path or MODELS_DIR / "tft_best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    log.info("Saved TFT model: %s", path)
    return path


def load_tft(model, path: Path | None = None):
    """Load TFT model from checkpoint."""
    path = path or MODELS_DIR / "tft_best.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    log.info("Loaded TFT model: %s", path)
    return model
