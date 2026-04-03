"""
Macro data collector — fetches economic indicators from FRED.

Uses the fredapi library which supports the ALFRED vintage API,
giving us "as-reported" data to avoid look-ahead bias from retroactive revisions.

Requires FRED_API_KEY in .env (free: https://fred.stlouisfed.org/docs/api/api_key.html)
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import pandas as pd

from src.utils.config import ROOT_DIR, get_data_config, get_macro_indicators
from src.utils.db import check_connection, upsert_dataframe
from src.utils.logger import get_logger

log = get_logger("macro_collector")


def _get_fred_client():
    """Initialize FRED API client."""
    from fredapi import Fred

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key or api_key == "your_fred_api_key_here":
        raise ValueError(
            "FRED_API_KEY not set. Get a free key at: "
            "https://fred.stlouisfed.org/docs/api/api_key.html "
            "then add it to your .env file."
        )
    return Fred(api_key=api_key)


def fetch_series(
    series_id: str,
    name: str,
    start: str,
    use_vintage: bool = True,
) -> pd.DataFrame:
    """Fetch a single FRED series.

    If use_vintage=True, fetches all vintage dates so we can use the
    value that was known at each point in time (no look-ahead).
    """
    fred = _get_fred_client()

    if use_vintage:
        try:
            # get_series_all_releases gives every vintage revision
            df = fred.get_series_all_releases(series_id)
            # df has columns: date, realtime_start, value
            # For each date, take the first release (what was known at the time)
            df = df.sort_values(["date", "realtime_start"])
            df = df.drop_duplicates(subset=["date"], keep="first")
            df = df.set_index("date")[["value", "realtime_start"]]
            df = df.rename(columns={"realtime_start": "vintage_date"})
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
            log.info("  %s (%s): %d rows (vintage)", name, series_id, len(df))
            return df
        except Exception as e:
            log.warning("  Vintage fetch failed for %s, falling back to latest: %s",
                        series_id, e)

    # Fallback: latest revised values
    series = fred.get_series(series_id, observation_start=start)
    df = pd.DataFrame({"value": series})
    df.index = pd.to_datetime(df.index, utc=True)
    df["vintage_date"] = None
    log.info("  %s (%s): %d rows (latest)", name, series_id, len(df))
    return df


def _splice_usd_index(start: str, use_vintage: bool) -> pd.DataFrame:
    """Splice DTWEXB (pre-2006) + DTWEXBGS (2006+) for full USD index coverage from 2000."""
    log.info("Splicing USD index: DTWEXB (legacy) + DTWEXBGS (current) ...")

    legacy = fetch_series("DTWEXB", "usd_index_legacy", start, use_vintage=use_vintage)
    current = fetch_series("DTWEXBGS", "usd_index", start, use_vintage=use_vintage)

    if legacy.empty:
        return current

    # Find overlap period and normalize scales
    if not current.empty:
        overlap_start = current.index.min()
        overlap_mask = (legacy.index >= overlap_start) & (legacy.index <= overlap_start + pd.DateOffset(months=6))
        if overlap_mask.any():
            legacy_mean = legacy.loc[overlap_mask, "value"].mean()
            current_mean = current.loc[current.index <= overlap_start + pd.DateOffset(months=6), "value"].mean()
            if legacy_mean > 0 and current_mean > 0:
                scale_factor = current_mean / legacy_mean
                legacy["value"] = legacy["value"] * scale_factor
                log.info("  Scale factor applied: %.4f", scale_factor)

        # Use legacy for pre-overlap, current for post
        legacy_part = legacy[legacy.index < overlap_start]
        spliced = pd.concat([legacy_part, current])
    else:
        spliced = legacy

    log.info("  USD index spliced: %d rows (%s to %s)",
             len(spliced), spliced.index.min().date(), spliced.index.max().date())
    return spliced


def fetch_all_macro(start: str | None = None, use_vintage: bool | None = None) -> pd.DataFrame:
    """Fetch all configured macro indicators and combine into a single DataFrame."""
    cfg = get_data_config()
    start = start or cfg["history"]["start_date"]
    if use_vintage is None:
        use_vintage = cfg["features"]["macro"]["use_vintage_data"]

    indicators = get_macro_indicators()
    splice_usd = cfg["features"]["macro"].get("usd_index_splice", True)
    all_rows = []

    for name, series_id in indicators.items():
        # Skip legacy USD series — handled by splice
        if name == "usd_index_legacy":
            continue

        # Handle USD index splice for pre-2006 coverage
        if name == "usd_index" and splice_usd:
            try:
                df = _splice_usd_index(start, use_vintage)
                for idx, row in df.iterrows():
                    all_rows.append({
                        "time": idx,
                        "indicator": name,
                        "value": row["value"],
                        "vintage_date": row.get("vintage_date"),
                    })
                continue
            except Exception as e:
                log.warning("USD splice failed, falling back to DTWEXBGS only: %s", e)

        log.info("Fetching %s (%s) ...", name, series_id)
        try:
            df = fetch_series(series_id, name, start, use_vintage=use_vintage)

            for idx, row in df.iterrows():
                all_rows.append({
                    "time": idx,
                    "indicator": name,
                    "value": row["value"],
                    "vintage_date": row.get("vintage_date"),
                })
        except Exception as e:
            log.error("Failed to fetch %s: %s", name, e)
            continue

    if not all_rows:
        log.error("No macro data fetched!")
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)
    log.info("Total macro rows: %d across %d indicators", len(result), len(indicators))
    return result


def save_parquet(df: pd.DataFrame) -> Path:
    """Save macro data as parquet."""
    out_dir = ROOT_DIR / get_data_config()["storage"]["raw_dir"] / "macro"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "macro_indicators.parquet"
    df.to_parquet(path, index=False)
    log.info("Saved %s", path.relative_to(ROOT_DIR))
    return path


def upsert_to_db(df: pd.DataFrame) -> int:
    """Upsert macro rows into TimescaleDB."""
    if df.empty:
        return 0
    # Fill None vintage_date with a sentinel for the unique constraint
    db_df = df.copy()
    db_df["vintage_date"] = db_df["vintage_date"].fillna(date(1900, 1, 1))
    return upsert_dataframe(
        df=db_df,
        table="macro_indicators",
        conflict_columns=["time", "indicator", "vintage_date"],
    )


def build_daily_macro_features(start: str | None = None) -> pd.DataFrame:
    """Build a daily-frequency macro feature matrix (forward-filled).

    Returns a DataFrame indexed by date with one column per indicator,
    suitable for joining with price data.
    """
    cfg = get_data_config()
    start = start or cfg["history"]["start_date"]

    # Try to load from parquet first
    parquet_path = ROOT_DIR / cfg["storage"]["raw_dir"] / "macro" / "macro_indicators.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        df = fetch_all_macro(start=start)

    if df.empty:
        return pd.DataFrame()

    # Pivot to wide format: date x indicator
    df["time"] = pd.to_datetime(df["time"])
    pivot = df.pivot_table(index="time", columns="indicator", values="value", aggfunc="first")

    # Resample to daily and forward-fill (weekends, holidays)
    freq = cfg["features"]["macro"]["resample_freq"]
    pivot = pivot.resample(freq).ffill()

    # Derived features
    if "yield_curve_10y2y" in pivot.columns:
        pivot["recession_signal"] = (pivot["yield_curve_10y2y"] < 0).astype(int)

    if "fed_funds_rate" in pivot.columns:
        pivot["rate_change_1m"] = pivot["fed_funds_rate"].pct_change(21)

    if "cpi" in pivot.columns:
        pivot["cpi_yoy"] = pivot["cpi"].pct_change(252)

    log.info("Macro feature matrix: %s, %d columns", pivot.shape, len(pivot.columns))
    return pivot


def main():
    parser = argparse.ArgumentParser(description="Macro data collector (FRED)")
    parser.add_argument("--start", help="Override start date")
    parser.add_argument("--no-vintage", action="store_true",
                        help="Use latest revised data instead of as-reported vintage")
    args = parser.parse_args()

    df = fetch_all_macro(start=args.start, use_vintage=not args.no_vintage)
    if df.empty:
        return

    save_parquet(df)

    if check_connection():
        upsert_to_db(df)


if __name__ == "__main__":
    main()
