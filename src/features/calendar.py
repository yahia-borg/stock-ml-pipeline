"""
Calendar and seasonal features — market structure effects driven by dates.

Feature groups (ranked by predictive value):
  1. Options expiration (OPEX) + quad witching — strongest calendar signal
  2. Quarter-end rebalancing effects
  3. Holiday effects (pre/post)
  4. Day-of-week + month cyclical encoding
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger("features.calendar")


def _get_third_friday(year: int, month: int) -> date:
    """Get the 3rd Friday of a month (monthly options expiration)."""
    d = date(year, month, 1)
    offset = (4 - d.weekday()) % 7  # 4 = Friday
    first_friday = d.replace(day=1 + offset)
    return first_friday.replace(day=first_friday.day + 14)


def _build_opex_dates(start_year: int, end_year: int) -> dict:
    """Build a lookup of all OPEX and quad witching dates."""
    monthly_opex = set()
    quad_witching = set()

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            try:
                opex = _get_third_friday(year, month)
                monthly_opex.add(opex)
                if month in (3, 6, 9, 12):
                    quad_witching.add(opex)
            except ValueError:
                continue

    return {"monthly_opex": monthly_opex, "quad_witching": quad_witching}


def _build_fomc_dates(start_year: int, end_year: int) -> list:
    """Build approximate FOMC meeting dates (8 meetings per year).

    FOMC meets 8 times per year, roughly every 6 weeks.
    Actual dates vary but are always Tue-Wed. We approximate with
    the standard schedule months: Jan, Mar, May, Jun, Jul, Sep, Nov, Dec.
    """
    fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
    dates_list = []
    for year in range(start_year, end_year + 1):
        for month in fomc_months:
            # FOMC typically meets on the 3rd Wednesday
            try:
                d = date(year, month, 1)
                # Find first Wednesday
                offset = (2 - d.weekday()) % 7  # 2 = Wednesday
                first_wed = d.replace(day=1 + offset)
                # Third Wednesday
                fomc_day = first_wed.replace(day=first_wed.day + 14)
                dates_list.append(fomc_day)
            except ValueError:
                continue
    return sorted(dates_list)


def _get_market_holidays(dates: pd.Series) -> set:
    """Get US market holidays."""
    # Strip tz for calendar APIs that expect naive datetimes
    dmin = dates.min()
    dmax = dates.max()
    if hasattr(dmin, "tzinfo") and dmin.tzinfo is not None:
        dmin = dmin.tz_convert(None)
        dmax = dmax.tz_convert(None)

    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        start = dmin.strftime("%Y-%m-%d")
        end = dmax.strftime("%Y-%m-%d")
        holidays = nyse.holidays().holidays
        return set(pd.Timestamp(h).date() for h in holidays
                   if pd.Timestamp(start) <= pd.Timestamp(h) <= pd.Timestamp(end))
    except ImportError:
        from pandas.tseries.holiday import USFederalHolidayCalendar
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=dmin, end=dmax)
        return set(h.date() for h in holidays)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features to a time-indexed DataFrame.

    Input: DataFrame with a datetime index (or 'time' column).
    Output: Same DataFrame with calendar feature columns added.
    """
    df = df.copy()

    # Ensure we have a datetime Series (not DatetimeIndex — need .dt accessor)
    if "time" in df.columns:
        dates = pd.to_datetime(df["time"])
    else:
        dates = pd.to_datetime(df.index).to_series(index=df.index)

    date_values = dates.dt.date

    log.info("Computing calendar features for %d rows...", len(df))

    # ── 1. OPEX features (highest predictive value) ──
    start_year = dates.min().year
    end_year = dates.max().year
    opex_data = _build_opex_dates(start_year, end_year)

    df["is_opex_day"] = date_values.isin(opex_data["monthly_opex"]).astype(int)
    df["is_quad_witching"] = date_values.isin(opex_data["quad_witching"]).astype(int)

    opex_sorted = sorted(opex_data["monthly_opex"])
    df["days_to_opex"] = date_values.map(
        lambda d: _days_to_next(d, opex_sorted)
    )
    df["is_opex_week"] = (df["days_to_opex"] <= 5).astype(int)

    # Day after OPEX (historically strong reversal day)
    opex_next_day = set()
    for opex in opex_sorted:
        opex_next_day.add(opex + timedelta(days=3))  # Friday + 3 = Monday
    df["is_day_after_opex"] = date_values.isin(opex_next_day).astype(int)

    # ── 2. Quarter-end features ──
    month_in_quarter = dates.dt.month % 3
    month_in_quarter = month_in_quarter.replace(0, 3)

    # Strip timezone for period arithmetic (avoids tz-naive/aware mismatch)
    dates_naive = dates.dt.tz_convert(None) if dates.dt.tz is not None else dates
    quarter_ends = dates_naive.dt.to_period("Q").dt.end_time
    df["days_to_quarter_end"] = (quarter_ends - dates_naive).dt.days.clip(0, 30)
    df["is_quarter_end_week"] = (df["days_to_quarter_end"] <= 7).astype(int)
    df["is_quarter_start_week"] = (
        (dates.dt.day <= 7).astype(int) & (month_in_quarter == 1).astype(int)
    )

    # ── 3. Holiday effects ──
    holidays = _get_market_holidays(dates)
    if holidays:
        holiday_list = sorted(holidays)
        df["days_to_holiday"] = date_values.map(
            lambda d: _days_to_next(d, holiday_list)
        )
        df["is_pre_holiday"] = (df["days_to_holiday"] == 1).astype(int)

        holiday_next = set()
        for h in holiday_list:
            for offset in range(1, 4):
                candidate = h + timedelta(days=offset)
                holiday_next.add(candidate)
        df["is_post_holiday"] = date_values.isin(holiday_next).astype(int)
    else:
        df["is_pre_holiday"] = 0
        df["is_post_holiday"] = 0

    # ── 4. FOMC meeting calendar (Fed communication edge) ──
    fomc_dates = _build_fomc_dates(start_year, end_year)
    df["days_to_fomc"] = date_values.map(lambda d: _days_to_next(d, fomc_dates))
    df["is_fomc_week"] = (df["days_to_fomc"] <= 5).astype(int)
    # Post-FOMC drift: day after FOMC decision historically trends
    fomc_next = set()
    for fd in fomc_dates:
        fomc_next.add(fd + timedelta(days=1))
        fomc_next.add(fd + timedelta(days=2))
    df["is_post_fomc"] = date_values.isin(fomc_next).astype(int)
    # FOMC blackout period (10 days before meeting — Fed officials don't speak)
    df["is_fomc_blackout"] = (df["days_to_fomc"] <= 10).astype(int) & (df["days_to_fomc"] > 0).astype(int)

    # ── 5. Day-of-week (cyclical encoding for neural nets) ──
    dow = dates.dt.dayofweek  # 0=Monday, 4=Friday
    df["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    df["is_monday"] = (dow == 0).astype(int)
    df["is_friday"] = (dow == 4).astype(int)

    # ── 5. Month-of-year (cyclical encoding) ──
    month = dates.dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["sell_in_may"] = ((month >= 5) & (month <= 10)).astype(int)

    n_cal = len([c for c in df.columns if c.startswith(("is_", "dow_", "month_", "days_to_", "sell_", "fomc"))])
    log.info("Calendar features: %d columns added", n_cal)
    return df


def _days_to_next(d, sorted_dates: list) -> int:
    """Find calendar days until next occurrence in a sorted date list."""
    import bisect
    if isinstance(d, pd.Timestamp):
        d = d.date()
    idx = bisect.bisect_left(sorted_dates, d)
    if idx < len(sorted_dates):
        target = sorted_dates[idx]
        if isinstance(target, pd.Timestamp):
            target = target.date()
        return (target - d).days
    return 30
