"""
Macro feature engineering — transforms raw FRED data into model-ready features.

Handles:
  - Loading from parquet or DB
  - Forward-filling to daily frequency
  - Derived features (recession signal, rate momentum, real yield, etc.)
  - Joining macro features to per-ticker price data
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import ROOT_DIR, get_data_config
from src.utils.logger import get_logger

log = get_logger("features.macro")


def load_macro_data() -> pd.DataFrame:
    """Load raw macro data from parquet (preferred) or DB."""
    cfg = get_data_config()
    parquet_path = ROOT_DIR / cfg["storage"]["raw_dir"] / "macro" / "macro_indicators.parquet"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        log.info("Loaded macro data from parquet: %d rows", len(df))
        return df

    from src.utils.db import read_sql, check_connection
    if check_connection():
        df = read_sql("SELECT time, indicator, value FROM macro_indicators ORDER BY time")
        log.info("Loaded macro data from DB: %d rows", len(df))
        return df

    log.warning("No macro data found")
    return pd.DataFrame()


def build_macro_features() -> pd.DataFrame:
    """Build the full daily macro feature matrix.

    Returns a DataFrame indexed by date with one column per macro feature,
    resampled to daily frequency with forward-fill.
    """
    cfg = get_data_config()
    raw = load_macro_data()
    if raw.empty:
        return pd.DataFrame()

    # Pivot: date x indicator
    raw["time"] = pd.to_datetime(raw["time"])
    pivot = raw.pivot_table(index="time", columns="indicator", values="value", aggfunc="first")

    # Resample to daily + forward-fill
    freq = cfg["features"]["macro"]["resample_freq"]
    pivot = pivot.resample(freq).ffill()

    log.info("Base macro matrix: %s", pivot.shape)

    # ── Derived features ──

    # Yield curve recession signal
    if "yield_curve_10y2y" in pivot.columns:
        pivot["recession_signal"] = (pivot["yield_curve_10y2y"] < 0).astype(int)
        # Duration of inversion (consecutive days inverted)
        inverted = pivot["yield_curve_10y2y"] < 0
        pivot["inversion_duration"] = inverted.groupby((~inverted).cumsum()).cumsum()

    # Yield curve rate of change (acceleration more predictive than level)
    if "yield_curve_10y2y" in pivot.columns:
        pivot["yield_curve_roc_20d"] = pivot["yield_curve_10y2y"].diff(20)
        pivot["yield_curve_roc_60d"] = pivot["yield_curve_10y2y"].diff(60)
        # Acceleration (second derivative) — curve flattening/steepening speed
        pivot["yield_curve_acceleration"] = pivot["yield_curve_roc_20d"].diff(20)

    # Dual yield curve signal (10y-2y AND 10y-3m both inverted = stronger signal)
    if "yield_curve_10y2y" in pivot.columns and "yield_curve_10y3m" in pivot.columns:
        pivot["dual_inversion"] = (
            (pivot["yield_curve_10y2y"] < 0) & (pivot["yield_curve_10y3m"] < 0)
        ).astype(int)

    # Fed funds rate momentum
    if "fed_funds_rate" in pivot.columns:
        pivot["rate_change_1m"] = pivot["fed_funds_rate"].diff(21)
        pivot["rate_change_3m"] = pivot["fed_funds_rate"].diff(63)
        # Rate regime: hiking (>25bps change), cutting, or holding
        pivot["rate_hiking"] = (pivot["rate_change_3m"] > 0.25).astype(int)
        pivot["rate_cutting"] = (pivot["rate_change_3m"] < -0.25).astype(int)

    # Inflation momentum
    if "cpi" in pivot.columns:
        pivot["cpi_yoy"] = pivot["cpi"].pct_change(252)
        pivot["cpi_mom"] = pivot["cpi"].pct_change(21)
        # Inflation acceleration (second derivative)
        pivot["cpi_acceleration"] = pivot["cpi_yoy"].diff(63)

    # Core vs headline CPI spread (signals transitory vs persistent inflation)
    if "cpi" in pivot.columns and "core_cpi" in pivot.columns:
        cpi_yoy = pivot["cpi"].pct_change(252)
        core_yoy = pivot["core_cpi"].pct_change(252)
        pivot["cpi_core_spread"] = cpi_yoy - core_yoy

    # Real yield proxy (Fed Funds - CPI YoY)
    if "fed_funds_rate" in pivot.columns and "cpi" in pivot.columns:
        pivot["real_yield"] = pivot["fed_funds_rate"] - pivot["cpi_yoy"] * 100

    # Labor market momentum
    if "unemployment" in pivot.columns:
        pivot["unemployment_change_3m"] = pivot["unemployment"].diff(63)
        # Sahm rule: recession when 3m avg unemployment rises 0.5pp above 12m low
        unemp_3m_avg = pivot["unemployment"].rolling(63).mean()
        unemp_12m_low = pivot["unemployment"].rolling(252).min()
        pivot["sahm_indicator"] = unemp_3m_avg - unemp_12m_low

    # Initial claims momentum (leading indicator)
    if "initial_claims" in pivot.columns:
        pivot["claims_4w_avg"] = pivot["initial_claims"].rolling(28).mean()
        pivot["claims_change_mom"] = pivot["claims_4w_avg"].pct_change(21)

    # USD strength momentum
    if "usd_index" in pivot.columns:
        pivot["usd_roc_20d"] = pivot["usd_index"].pct_change(20)
        pivot["usd_roc_60d"] = pivot["usd_index"].pct_change(60)

    # Consumer sentiment
    if "consumer_sentiment" in pivot.columns:
        pivot["sentiment_change_1m"] = pivot["consumer_sentiment"].diff(21)

    # Industrial production
    if "industrial_production" in pivot.columns:
        pivot["ip_yoy"] = pivot["industrial_production"].pct_change(252)

    n_features = len(pivot.columns)
    log.info("Macro feature matrix: %d rows, %d features", len(pivot), n_features)
    return pivot


def merge_macro_to_prices(
    price_df: pd.DataFrame,
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join macro features to a price DataFrame using as-of join (forward-fill).

    This ensures no look-ahead bias: each row gets the latest macro value
    known at that date, not future values.
    """
    if macro_df is None:
        macro_df = build_macro_features()

    if macro_df.empty:
        log.warning("No macro features to merge")
        return price_df

    price_df = price_df.copy()

    # Align indexes — strip timezone from both sides
    if "time" in price_df.columns:
        price_idx = pd.to_datetime(price_df["time"])
    else:
        price_idx = pd.to_datetime(price_df.index)

    if price_idx.tz is not None:
        price_idx = price_idx.tz_convert(None)

    macro_df.index = pd.to_datetime(macro_df.index)
    if macro_df.index.tz is not None:
        macro_df.index = macro_df.index.tz_convert(None)

    # As-of merge: for each price date, get the most recent macro values
    macro_reindexed = macro_df.reindex(price_idx, method="ffill")

    # Add macro_ prefix to avoid column name conflicts
    macro_cols = {col: f"macro_{col}" for col in macro_reindexed.columns}
    macro_reindexed = macro_reindexed.rename(columns=macro_cols)

    # Merge
    for col in macro_reindexed.columns:
        price_df[col] = macro_reindexed[col].values

    log.info("Merged %d macro features into price data", len(macro_cols))
    return price_df
