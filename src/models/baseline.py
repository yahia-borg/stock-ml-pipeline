"""
Baseline models — XGBoost and LightGBM classifiers.

These are the first models to train. They:
  - Train fast (seconds to minutes)
  - Provide a concrete accuracy baseline to beat
  - Work well with tabular features out of the box
  - Support built-in feature importance (SHAP)
  - Handle missing values natively

All hyperparameters come from configs/model_config.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb

from src.utils.config import ROOT_DIR, get_model_config
from src.utils.logger import get_logger

log = get_logger("models.baseline")

MODELS_DIR = ROOT_DIR / os.environ.get("MODELS_DIR", "models")


def build_lgbm(n_classes: int = 3) -> lgb.LGBMClassifier:
    """Create a LightGBM classifier from config."""
    cfg = get_model_config()["baseline"]["lgbm"]

    params = {
        "n_estimators": cfg["n_estimators"],
        "max_depth": cfg["max_depth"],
        "learning_rate": cfg["learning_rate"],
        "num_leaves": cfg["num_leaves"],
        "subsample": cfg["subsample"],
        "colsample_bytree": cfg["colsample_bytree"],
        "min_child_samples": cfg.get("min_child_samples", 50),
        "class_weight": cfg["class_weight"],
        "reg_alpha": cfg.get("reg_alpha", 0.0),
        "reg_lambda": cfg.get("reg_lambda", 0.0),
        "random_state": cfg["random_state"],
        "verbose": -1,
        "n_jobs": -1,
    }

    if n_classes > 2:
        params["objective"] = "multiclass"
        params["num_class"] = n_classes

    return lgb.LGBMClassifier(**params)


def build_xgboost(n_classes: int = 3) -> xgb.XGBClassifier:
    """Create an XGBoost classifier from config."""
    cfg = get_model_config()["baseline"]["xgboost"]

    params = {
        "n_estimators": cfg["n_estimators"],
        "max_depth": cfg["max_depth"],
        "learning_rate": cfg["learning_rate"],
        "subsample": cfg["subsample"],
        "colsample_bytree": cfg["colsample_bytree"],
        "min_child_weight": cfg.get("min_child_weight", 20),
        "gamma": cfg.get("gamma", 0.5),
        "reg_alpha": cfg.get("reg_alpha", 0.0),
        "reg_lambda": cfg.get("reg_lambda", 1.0),
        "eval_metric": cfg["eval_metric"],
        "tree_method": cfg["tree_method"],
        "random_state": cfg["random_state"],
        "use_label_encoder": False,
        "verbosity": 0,
        "n_jobs": -1,
    }

    if n_classes > 2:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes

    return xgb.XGBClassifier(**params)


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    early_stopping_rounds: int = 50,
):
    """Train a baseline model with optional early stopping on validation set."""
    model_name = type(model).__name__

    fit_params = {}

    if X_val is not None and y_val is not None:
        if isinstance(model, lgb.LGBMClassifier):
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["callbacks"] = [
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]
        elif isinstance(model, xgb.XGBClassifier):
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

    log.info("Training %s: %d samples, %d features ...",
             model_name, X_train.shape[0], X_train.shape[1])

    model.fit(X_train, y_train, **fit_params)

    # Log training accuracy
    train_acc = (model.predict(X_train) == y_train).mean()
    log.info("  %s train accuracy: %.4f", model_name, train_acc)

    if X_val is not None:
        val_acc = (model.predict(X_val) == y_val).mean()
        log.info("  %s val accuracy: %.4f", model_name, val_acc)

    return model


def get_feature_importance(model, feature_names: list[str]) -> dict[str, float]:
    """Extract feature importance from a trained model."""
    if isinstance(model, lgb.LGBMClassifier):
        importance = model.feature_importances_
    elif isinstance(model, xgb.XGBClassifier):
        importance = model.feature_importances_
    else:
        return {}

    return dict(sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True,
    ))


def save_model(model, name: str, horizon: int, fold_id: int | None = None) -> Path:
    """Save a trained model to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if fold_id is not None:
        filename = f"{name}_{horizon}d_fold{fold_id:03d}.joblib"
    else:
        filename = f"{name}_{horizon}d.joblib"

    path = MODELS_DIR / filename
    joblib.dump(model, path)
    log.info("Saved model: %s", path.name)
    return path


def load_model(name: str, horizon: int, fold_id: int | None = None):
    """Load a trained model from disk."""
    if fold_id is not None:
        filename = f"{name}_{horizon}d_fold{fold_id:03d}.joblib"
    else:
        filename = f"{name}_{horizon}d.joblib"

    path = MODELS_DIR / filename
    model = joblib.load(path)
    log.info("Loaded model: %s", path.name)
    return model
