"""
TFT training loop — walk-forward training with PyTorch Lightning.

Usage:
  python -m src.training.train_tft                          # train TFT (5d horizon)
  python -m src.training.train_tft --horizon 1              # 1-day horizon
  python -m src.training.train_tft --horizon 20 --no-wandb  # no experiment tracking
  python -m src.training.train_tft --quick                  # quick run (fewer epochs, for testing)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.config import ROOT_DIR, get_model_config, get_data_config
from src.utils.logger import get_logger

log = get_logger("training.train_tft")

MODELS_DIR = ROOT_DIR / os.environ.get("MODELS_DIR", "models")


def train_tft(
    horizon: int = 5,
    ticker: str | None = None,
    use_wandb: bool = True,
    quick: bool = False,
) -> dict:
    """Train TFT using walk-forward validation.

    Args:
        horizon: Forecast horizon in days
        ticker: Train on specific ticker (None = all tickers)
        use_wandb: Enable Weights & Biases logging
        quick: Quick mode (fewer epochs, for testing)
    """
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import CrossEntropy, QuantileLoss
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    from src.training.train import load_feature_matrix
    from src.training.walk_forward import generate_splits
    from src.training.evaluate import evaluate_fold, print_evaluation_report
    from src.training.tft_data import (
        prepare_tft_dataframe, create_tft_dataset, split_train_val,
    )
    from src.models.tft import get_tft_config

    cfg = get_tft_config()
    training_cfg = get_model_config().get("training", {})

    max_epochs = 5 if quick else cfg["max_epochs"]
    patience = 3 if quick else cfg["patience"]

    # Load feature matrix
    df = load_feature_matrix(ticker)
    if df.empty:
        return {}

    target_col = f"fwd_return_{horizon}d"
    label_col = f"label_{horizon}d"

    if target_col not in df.columns:
        log.error("Target '%s' not found", target_col)
        return {}

    # Prepare for TFT
    tft_df = prepare_tft_dataframe(df, target_col=target_col)

    # Walk-forward splits
    # For TFT we use fewer, larger folds since training is expensive
    folds = generate_splits(tft_df, mode="expanding")

    # Use a subset of folds for TFT (every 3rd fold to save time)
    if not quick:
        folds = folds[::3]  # every 3rd fold
    else:
        folds = folds[-2:]  # just last 2 folds for quick mode

    log.info("Training TFT: %dd horizon, %d folds, max_epochs=%d",
             horizon, len(folds), max_epochs)

    fold_metrics = []
    best_model_path = None

    for fold in folds:
        log.info("=" * 60)
        log.info("Fold %d: train [%s → %s], test [%s → %s]",
                 fold.fold_id,
                 fold.train_start.date(), fold.train_end.date(),
                 fold.test_start.date(), fold.test_end.date())

        # Split data for this fold
        train_mask = (tft_df.index >= fold.train_start) & (tft_df.index <= fold.train_end)
        test_mask = (tft_df.index >= fold.test_start) & (tft_df.index <= fold.test_end)

        train_data = tft_df.loc[train_mask].copy()
        test_data = tft_df.loc[test_mask].copy()

        if len(train_data) < 500 or len(test_data) < 50:
            log.warning("Insufficient data for fold %d, skipping", fold.fold_id)
            continue

        # Split train into train + val
        train_split, val_split = split_train_val(train_data, val_fraction=0.15)

        try:
            # Create datasets
            train_dataset = create_tft_dataset(
                train_split, target_col=target_col, training=True
            )
            val_dataset = train_dataset.from_dataset(
                train_dataset, val_split, predict=False, stop_randomization=True
            )

            # Create dataloaders
            batch_size = 64 if quick else cfg["batch_size"]
            num_workers = 0 if quick else training_cfg.get("num_workers", 4)

            train_loader = train_dataset.to_dataloader(
                train=True, batch_size=batch_size, num_workers=num_workers,
            )
            val_loader = val_dataset.to_dataloader(
                train=False, batch_size=batch_size * 2, num_workers=num_workers,
            )

            # Build TFT model
            tft_model = TemporalFusionTransformer.from_dataset(
                train_dataset,
                hidden_size=cfg["hidden_size"],
                attention_head_size=cfg["attention_head_size"],
                dropout=cfg["dropout"],
                hidden_continuous_size=cfg["hidden_continuous_size"],
                learning_rate=cfg["learning_rate"],
                reduce_on_plateau_patience=patience // 2,
                log_interval=10,
                log_val_interval=1,
            )

            n_params = sum(p.numel() for p in tft_model.parameters())
            log.info("TFT parameters: %s", f"{n_params:,}")

            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    mode="min",
                    verbose=False,
                ),
                ModelCheckpoint(
                    dirpath=str(MODELS_DIR / "tft_checkpoints"),
                    filename=f"tft_{horizon}d_fold{fold.fold_id:03d}_{{epoch}}_{{val_loss:.4f}}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                ),
            ]

            # Logger
            logger = None
            if use_wandb:
                try:
                    from pytorch_lightning.loggers import WandbLogger
                    logger = WandbLogger(
                        project=os.environ.get("WANDB_PROJECT", "stock-forecast"),
                        name=f"tft_{horizon}d_fold{fold.fold_id}",
                        tags=["tft", f"{horizon}d"],
                    )
                except Exception:
                    log.info("W&B not available, using default logger")

            # Train
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator="auto",
                devices=1,
                gradient_clip_val=cfg["gradient_clip_val"],
                callbacks=callbacks,
                logger=logger,
                enable_progress_bar=True,
                enable_model_summary=False,
            )

            trainer.fit(tft_model, train_loader, val_loader)

            # Load best checkpoint
            best_ckpt = callbacks[1].best_model_path
            if best_ckpt:
                tft_model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
                best_model_path = best_ckpt

            # Predict on test set
            test_dataset = train_dataset.from_dataset(
                train_dataset, test_data, predict=True, stop_randomization=True
            )
            test_loader = test_dataset.to_dataloader(
                train=False, batch_size=batch_size * 2, num_workers=num_workers,
            )

            raw_predictions = tft_model.predict(
                test_loader, mode="raw", return_x=True,
            )

            # Extract predictions — TFT returns quantile forecasts
            # Convert to directional predictions
            prediction_values = raw_predictions.output["prediction"]

            # Get the median prediction (middle quantile)
            if prediction_values.dim() == 3:
                # [batch, prediction_length, quantiles]
                median_pred = prediction_values[:, 0, prediction_values.shape[2] // 2]
            else:
                median_pred = prediction_values[:, 0]

            median_pred = median_pred.cpu().numpy()

            # Get actual values from test data aligned to predictions
            test_targets = test_data[target_col].dropna().values
            test_labels = test_data[label_col].dropna().values if label_col in test_data.columns else None

            # Align lengths
            min_len = min(len(median_pred), len(test_targets))
            median_pred = median_pred[:min_len]
            test_targets_aligned = test_targets[:min_len]

            # Convert predictions to directional labels using adaptive thresholds
            recent_vol = np.std(train_data[target_col].dropna().values[-60:]) if len(train_data) > 60 else 0.01
            threshold = 0.4 * recent_vol

            y_pred = np.where(median_pred > threshold, 2,
                              np.where(median_pred < -threshold, 0, 1))

            if test_labels is not None:
                y_true = test_labels[:min_len].astype(int)
            else:
                y_true = np.where(test_targets_aligned > threshold, 2,
                                  np.where(test_targets_aligned < -threshold, 0, 1))

            # Evaluate
            metrics = evaluate_fold(
                y_true=y_true,
                y_pred=y_pred,
                actual_returns=test_targets_aligned,
                horizon=horizon,
            )
            metrics["fold_id"] = fold.fold_id
            metrics["test_start"] = str(fold.test_start.date())
            metrics["test_end"] = str(fold.test_end.date())
            metrics["best_val_loss"] = float(callbacks[1].best_model_score or 0)
            fold_metrics.append(metrics)

            log.info("Fold %d: acc=%.4f, dir_acc=%.4f, val_loss=%.4f",
                     fold.fold_id,
                     metrics["accuracy"],
                     metrics.get("directional_accuracy", 0),
                     metrics.get("best_val_loss", 0))

        except Exception as e:
            log.error("Fold %d failed: %s", fold.fold_id, e)
            import traceback
            traceback.print_exc()
            continue

        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not fold_metrics:
        log.error("No folds completed successfully")
        return {}

    # Summary
    summary = print_evaluation_report(fold_metrics, model_name="TFT", horizon=horizon)

    # Save best model to standard location
    if best_model_path:
        import shutil
        final_path = MODELS_DIR / f"tft_{horizon}d_best.ckpt"
        shutil.copy2(best_model_path, final_path)
        log.info("Best model saved: %s", final_path)

    # Save results
    results = {
        "model_type": "tft",
        "horizon": horizon,
        "ticker": ticker or "pooled",
        "n_folds": len(fold_metrics),
        "fold_metrics": fold_metrics,
        "best_model_path": str(best_model_path) if best_model_path else None,
    }

    results_path = MODELS_DIR / f"tft_{horizon}d_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results: %s", results_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Temporal Fusion Transformer")
    parser.add_argument("--horizon", type=int, choices=[1, 5, 20], default=5)
    parser.add_argument("--ticker", help="Train on specific ticker")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (5 epochs, 2 folds — for testing)")
    args = parser.parse_args()

    train_tft(
        horizon=args.horizon,
        ticker=args.ticker,
        use_wandb=not args.no_wandb,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
