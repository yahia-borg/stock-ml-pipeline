"""
TFT data preparation — converts our feature matrix into pytorch-forecasting format.

pytorch-forecasting's TimeSeriesDataSet requires a specific long-format DataFrame
with group identifiers, time index, and explicit covariate classification.

This module handles:
  - Converting our wide feature matrix into TFT-compatible format
  - Proper train/val splitting aligned with walk-forward folds
  - DataLoader creation with appropriate batching
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.models.tft import classify_features, get_tft_config
from src.utils.logger import get_logger

log = get_logger("training.tft_data")

# Sequence length: how many past days the TFT looks at
ENCODER_LENGTH = 60   # 60 trading days (~3 months)
DECODER_LENGTH = 20   # predict up to 20 days ahead


def prepare_tft_dataframe(
    df: pd.DataFrame,
    target_col: str = "fwd_return_5d",
) -> pd.DataFrame:
    """Prepare a DataFrame for TFT consumption.

    Adds required columns:
      - time_idx: integer time index (monotonically increasing per group)
      - group: group identifier (ticker)

    Fills NaN, encodes categoricals, and ensures proper types.
    """
    df = df.copy()

    # Ensure ticker column exists
    if "ticker" not in df.columns:
        df["ticker"] = "UNKNOWN"

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df.index = pd.to_datetime(df["time"])
        else:
            df.index = pd.to_datetime(df.index)

    # Create time_idx per ticker (integer sequence)
    df = df.sort_index()
    df["time_idx"] = df.groupby("ticker").cumcount()

    # Ensure target exists and is numeric
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    # Drop rows where target is NaN (end of series for forward returns)
    n_before = len(df)
    df = df.dropna(subset=[target_col])
    if len(df) < n_before:
        log.info("Dropped %d rows with NaN target", n_before - len(df))

    # Fill remaining NaN in features with 0
    feature_classification = classify_features(df.columns.tolist())
    all_feature_cols = (
        feature_classification["time_varying_known_reals"]
        + feature_classification["time_varying_unknown_reals"]
    )
    for col in all_feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float)

    # Encode categoricals as strings (pytorch-forecasting handles encoding)
    all_cat_cols = (
        feature_classification["static_categoricals"]
        + feature_classification["time_varying_known_categoricals"]
        + feature_classification["time_varying_unknown_categoricals"]
    )
    for col in all_cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(str)

    log.info("Prepared TFT DataFrame: %d rows, %d tickers, time_idx range [%d, %d]",
             len(df), df["ticker"].nunique(),
             df["time_idx"].min(), df["time_idx"].max())

    return df


def create_tft_dataset(
    df: pd.DataFrame,
    target_col: str = "fwd_return_5d",
    encoder_length: int = ENCODER_LENGTH,
    prediction_length: int = 1,
    training: bool = True,
):
    """Create a pytorch-forecasting TimeSeriesDataSet.

    Args:
        df: Prepared DataFrame (from prepare_tft_dataframe)
        target_col: Column to predict
        encoder_length: Number of past timesteps as input
        prediction_length: Number of future timesteps to predict
        training: If True, create training dataset; if False, create prediction dataset

    Returns:
        TimeSeriesDataSet instance
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    feature_classification = classify_features(df.columns.tolist())

    cfg = get_tft_config()

    # Filter to columns that actually exist in df
    known_cats = [c for c in feature_classification["time_varying_known_categoricals"] if c in df.columns]
    known_reals = [c for c in feature_classification["time_varying_known_reals"] if c in df.columns]
    unknown_cats = [c for c in feature_classification["time_varying_unknown_categoricals"] if c in df.columns]
    unknown_reals = [c for c in feature_classification["time_varying_unknown_reals"] if c in df.columns]
    static_cats = [c for c in feature_classification["static_categoricals"] if c in df.columns]

    # Limit unknown reals to avoid memory issues (top features by variance)
    if len(unknown_reals) > 80:
        variances = df[unknown_reals].var().sort_values(ascending=False)
        unknown_reals = variances.head(80).index.tolist()
        log.info("Limited unknown reals to top 80 by variance")

    dataset_kwargs = dict(
        data=df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["ticker"],
        max_encoder_length=encoder_length,
        max_prediction_length=prediction_length,
        static_categoricals=static_cats,
        time_varying_known_categoricals=known_cats,
        time_varying_known_reals=known_reals,
        time_varying_unknown_categoricals=unknown_cats,
        time_varying_unknown_reals=unknown_reals + [target_col],
        target_normalizer=GroupNormalizer(groups=["ticker"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    if training:
        dataset_kwargs["min_encoder_length"] = encoder_length // 2

    dataset = TimeSeriesDataSet(**dataset_kwargs)

    log.info("Created TFT dataset: %d samples, encoder=%d, prediction=%d",
             len(dataset), encoder_length, prediction_length)
    log.info("  Static categoricals: %d", len(static_cats))
    log.info("  Known categoricals: %d, Known reals: %d", len(known_cats), len(known_reals))
    log.info("  Unknown categoricals: %d, Unknown reals: %d", len(unknown_cats), len(unknown_reals))

    return dataset


def create_dataloaders(
    train_dataset,
    val_dataset=None,
    batch_size: int | None = None,
    num_workers: int | None = None,
):
    """Create DataLoaders from TimeSeriesDataSets."""
    cfg = get_tft_config()
    batch_size = batch_size or cfg["batch_size"]
    num_workers = num_workers or get_model_config_training().get("num_workers", 4)

    train_loader = train_dataset.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = val_dataset.to_dataloader(
            train=False,
            batch_size=batch_size * 2,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    return train_loader, val_loader


def get_model_config_training():
    from src.utils.config import get_model_config
    return get_model_config().get("training", {})


def split_train_val(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split time-ordered data into train and validation.

    Splits by time within each ticker (no random shuffling).
    """
    train_dfs = []
    val_dfs = []

    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("time_idx")
        split_idx = int(len(group) * (1 - val_fraction))
        train_dfs.append(group.iloc[:split_idx])
        val_dfs.append(group.iloc[split_idx:])

    train = pd.concat(train_dfs)
    val = pd.concat(val_dfs)

    log.info("Train/val split: %d / %d rows (%.0f%% / %.0f%%)",
             len(train), len(val),
             len(train) / (len(train) + len(val)) * 100,
             len(val) / (len(train) + len(val)) * 100)

    return train, val
