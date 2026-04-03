"""
Technical feature engineering — standard TA + advanced microstructure and volatility features.

Feature groups:
  1. Momentum (RSI, ROC, MACD, frog-in-the-pan)
  2. Trend (EMA, ADX, distance from highs/lows)
  3. Volatility (ATR, Bollinger, Garman-Klass, Parkinson, Yang-Zhang, vol-of-vol, variance ratio)
  4. Volume (OBV, VWAP deviation, volume Z-score, Amihud illiquidity)
  5. Microstructure (close location value, Corwin-Schultz spread, Roll spread)
  6. Returns (lagged, multi-horizon)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.utils.config import get_data_config
from src.utils.logger import get_logger

log = get_logger("features.technical")


# ── 1. Momentum ────────────────────────────────────────────

def _add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    cfg = get_data_config()["features"]["technical_indicators"]

    for p in cfg["rsi_periods"]:
        df.ta.rsi(length=p, append=True)

    for p in cfg["roc_periods"]:
        df.ta.roc(length=p, append=True)

    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    # Frog-in-the-pan momentum decomposition
    # Measures whether momentum is "continuous" (many small gains) vs "discrete" (few big jumps)
    for window in [20, 60]:
        pos_days = (df["close"].pct_change() > 0).rolling(window).sum()
        neg_days = (df["close"].pct_change() < 0).rolling(window).sum()
        total_days = pos_days + neg_days
        df[f"fip_{window}d"] = (pos_days - neg_days) / total_days.replace(0, np.nan)
        # Interaction: FIP * total return (continuous momentum is stronger)
        df[f"fip_momentum_{window}d"] = df[f"fip_{window}d"] * df["close"].pct_change(window)

    return df


# ── 2. Trend ───────────────────────────────────────────────

def _add_trend(df: pd.DataFrame) -> pd.DataFrame:
    cfg = get_data_config()["features"]["technical_indicators"]

    for p in cfg["ema_periods"]:
        df.ta.ema(length=p, append=True)

    df.ta.adx(length=14, append=True)

    # Distance from 52-week high/low
    df["dist_52w_high"] = df["close"] / df["high"].rolling(252).max() - 1
    df["dist_52w_low"] = df["close"] / df["low"].rolling(252).min() - 1

    # 52-week high proximity (George & Hwang factor — stronger than raw momentum)
    df["near_52w_high"] = (df["dist_52w_high"] > -0.05).astype(int)

    # EMA crossover signals
    if all(f"EMA_{p}" in df.columns for p in [20, 50, 200]):
        df["ema_20_50_cross"] = (df["EMA_20"] > df["EMA_50"]).astype(int)
        df["ema_50_200_cross"] = (df["EMA_50"] > df["EMA_200"]).astype(int)

    return df


# ── 3. Volatility (Advanced) ──────────────────────────────

def _add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    cfg = get_data_config()["features"]["technical_indicators"]

    # Standard
    df.ta.bbands(length=cfg["bbands_length"], append=True)
    df.ta.atr(length=cfg["atr_length"], append=True)

    # Returns for volatility calcs
    ret = df["close"].pct_change()
    for w in cfg["rolling_vol_windows"]:
        df[f"vol_{w}d"] = ret.rolling(w).std()

    # Garman-Klass volatility — most efficient OHLC estimator
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    gk_daily = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    for w in [5, 20]:
        df[f"gk_vol_{w}d"] = np.sqrt(gk_daily.rolling(w).mean() * 252)

    # Parkinson volatility — uses high-low range, 5x more efficient than close-close
    park_daily = log_hl ** 2 / (4 * np.log(2))
    for w in [5, 20]:
        df[f"parkinson_vol_{w}d"] = np.sqrt(park_daily.rolling(w).mean() * 252)

    # Yang-Zhang volatility — handles overnight jumps
    log_oc = np.log(df["open"] / df["close"].shift(1))  # overnight return
    log_oc_mean = log_oc.rolling(20).mean()
    overnight_var = ((log_oc - log_oc_mean) ** 2).rolling(20).mean()

    log_open_close = np.log(df["close"] / df["open"])
    loc_mean = log_open_close.rolling(20).mean()
    open_close_var = ((log_open_close - loc_mean) ** 2).rolling(20).mean()

    rs_var = (log_hl ** 2 - 2 * np.log(2) * log_co ** 2).rolling(20).mean()

    k = 0.34 / (1.34 + (20 + 1) / (20 - 1))
    yz_var = overnight_var + k * open_close_var + (1 - k) * rs_var
    df["yz_vol_20d"] = np.sqrt(yz_var.clip(lower=0) * 252)

    # Volatility-of-volatility (regime indicator)
    df["vol_of_vol_20d"] = df["vol_20d"].rolling(20).std()

    # Variance ratio (trending vs mean-reverting regime)
    # ratio > 1 = trending, < 1 = mean-reverting
    df["var_ratio_5_20"] = df["vol_5d"] / df["vol_20d"].replace(0, np.nan)

    return df


# ── 4. Volume ──────────────────────────────────────────────

def _add_volume(df: pd.DataFrame) -> pd.DataFrame:
    df.ta.obv(append=True)

    # Volume ratio
    vol_mean_20 = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / vol_mean_20.replace(0, np.nan)

    # Volume Z-score — detects abnormal volume days
    vol_std_60 = df["volume"].rolling(60).std()
    vol_mean_60 = df["volume"].rolling(60).mean()
    df["volume_zscore"] = (df["volume"] - vol_mean_60) / vol_std_60.replace(0, np.nan)

    # VWAP deviation (if VWAP available, otherwise approximate from typical price * volume)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).rolling(20).sum()
    cum_vol = df["volume"].rolling(20).sum()
    vwap_approx = cum_tp_vol / cum_vol.replace(0, np.nan)
    df["vwap_deviation"] = (df["close"] - vwap_approx) / vwap_approx.replace(0, np.nan)

    # OBV divergence: rank difference between price ROC and OBV ROC
    if "OBV" in df.columns:
        price_roc_20 = df["close"].pct_change(20)
        obv_roc_20 = df["OBV"].pct_change(20)
        df["obv_divergence"] = price_roc_20.rank(pct=True) - obv_roc_20.rank(pct=True)

    # Amihud illiquidity ratio — one of the most robust academic predictors
    dollar_volume = df["close"] * df["volume"]
    daily_illiq = ret_abs(df) / dollar_volume.replace(0, np.nan)
    for w in [20, 60]:
        df[f"amihud_illiq_{w}d"] = daily_illiq.rolling(w).mean()

    return df


def ret_abs(df: pd.DataFrame) -> pd.Series:
    """Absolute daily return."""
    return df["close"].pct_change().abs()


# ── 5. Microstructure ─────────────────────────────────────

def _add_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    # Close Location Value — proxy for buying/selling pressure
    # 1.0 = closed at high (buying pressure), 0.0 = closed at low (selling pressure)
    hl_range = df["high"] - df["low"]
    df["close_location_value"] = (df["close"] - df["low"]) / hl_range.replace(0, np.nan)

    # Corwin-Schultz spread estimator — bid-ask spread from daily high-low
    log_hl = np.log(df["high"] / df["low"])
    log_hl_sq = log_hl ** 2

    # Two-day high-low
    high_2d = df["high"].rolling(2).max()
    low_2d = df["low"].rolling(2).min()
    log_hl2_sq = np.log(high_2d / low_2d) ** 2

    beta = log_hl_sq.rolling(2).sum()
    gamma = log_hl2_sq

    sqrt2 = np.sqrt(2)
    sqrt3m2sqrt2 = np.sqrt(3 - 2 * sqrt2)

    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / sqrt3m2sqrt2 - np.sqrt(gamma / sqrt3m2sqrt2)
    alpha = alpha.clip(lower=0)
    df["cs_spread"] = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    # Roll spread estimator — from serial covariance of price changes
    delta_p = df["close"].diff()
    for w in [20, 60]:
        cov = delta_p.rolling(w).apply(
            lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0,
            raw=True,
        )
        # Roll spread = 2 * sqrt(-cov) when cov < 0
        df[f"roll_spread_{w}d"] = np.where(cov < 0, 2 * np.sqrt(-cov), 0)

    return df


# ── 6. Returns ─────────────────────────────────────────────

def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
    cfg = get_data_config()["features"]["technical_indicators"]
    for lag in cfg["return_lags"]:
        df[f"return_{lag}d"] = df["close"].pct_change(lag)

    # Log returns (more suitable for statistical modeling)
    df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))

    # Jegadeesh-Titman 12-1 month momentum (GKX top-3 predictor)
    # Skip the most recent month (reversal effect) and use prior 11 months
    df["momentum_12_1"] = df["close"].shift(21) / df["close"].shift(252) - 1

    return df


# ── 7. Enhanced Microstructure (Tier 1 alpha features) ────

def _add_enhanced_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """Additional microstructure features from OHLCV that proxy institutional flow."""

    # Gap signal — overnight information flow proxy
    df["gap_signal"] = df["open"] / df["close"].shift(1) - 1
    df["gap_signal_5d_avg"] = df["gap_signal"].rolling(5).mean()
    df["gap_signal_abs_20d"] = df["gap_signal"].abs().rolling(20).mean()

    # Candlestick body and shadow ratios (conviction and pressure)
    hl_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / hl_range
    df["upper_shadow_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / hl_range
    df["lower_shadow_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / hl_range

    # Signed volume accumulation — volume-weighted buying pressure
    clv = (df["close"] - df["low"]) / hl_range  # close location value
    signed_vol = clv * df["volume"]
    for w in [5, 20]:
        df[f"signed_volume_{w}d"] = signed_vol.rolling(w).sum() / df["volume"].rolling(w).sum().replace(0, np.nan)

    # Overnight return (open vs previous close) — information arrival proxy
    df["overnight_return"] = df["open"] / df["close"].shift(1) - 1
    df["intraday_return"] = df["close"] / df["open"] - 1

    # Range ratio (unusual intraday range detection)
    daily_range = (df["high"] - df["low"]) / df["close"]
    df["range_ratio_zscore"] = (daily_range - daily_range.rolling(60).mean()) / daily_range.rolling(60).std().replace(0, np.nan)

    return df


# ── Public API ─────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical features to an OHLCV DataFrame.

    Input: DataFrame with columns [open, high, low, close, volume]
    Output: Same DataFrame with ~60+ technical feature columns added
    """
    log.info("Computing technical features for %d rows...", len(df))

    df = df.copy()

    # Ensure we have return_1d for volatility calcs
    if "return_1d" not in df.columns:
        df["return_1d"] = df["close"].pct_change()

    df = _add_returns(df)
    df = _add_momentum(df)
    df = _add_trend(df)
    df = _add_volatility(df)
    df = _add_volume(df)
    df = _add_microstructure(df)
    df = _add_enhanced_microstructure(df)

    n_features = len([c for c in df.columns if c not in
                      ["time", "ticker", "open", "high", "low", "close", "volume", "adj_close"]])
    log.info("Technical features: %d columns added", n_features)
    return df
