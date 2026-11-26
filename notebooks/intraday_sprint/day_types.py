"""Day-type router: labels each trading date with one category known before the open.

Categories, in priority order (a date that is two things gets the first that applies):

    russell         Russell reconstitution (last Friday of June)
    msci            MSCI index review effective date
    sp_quad         S&P quarterly rebalance or quarterly options expiry ("quad witching")
    month_end       last trading day of the month
    monthly_expiry  monthly options expiry
    normal          everything else

All inputs are scheduled calendar events, published well in advance, so the label is
safe to use at any intraday checkpoint. Use it to route rows to day-type models::

    from day_types import DAY_TYPES, index_calendar, day_type
    calendar = index_calendar()
    labels = day_type(fold.test, calendar)      # Series aligned with fold.test
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REBALANCE = ROOT / "blackrock_data/rebalancing/stock_rebalance_features_2011_2025.parquet"

DAY_TYPES = ("russell", "msci", "sp_quad", "month_end", "monthly_expiry", "normal")
EVENT_TYPES = DAY_TYPES[:-1]


def index_calendar(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """One row per date: market-wide Russell, MSCI and S&P quarterly rebalance flags (0/1)."""
    con = con or duckdb.connect()
    cal = con.sql(f"""
        SELECT CAST(date AS DATE) AS date, max(russell_recon) AS russell, max(msci_review) AS msci,
               max(sp_quarterly_rebal) AS sp_rebal
        FROM read_parquet('{REBALANCE}') GROUP BY 1""").df()
    cal["date"] = pd.to_datetime(cal.date)
    return cal.set_index("date").sort_index().fillna(0)


def day_type(frame: pd.DataFrame, calendar: pd.DataFrame) -> pd.Series:
    """Category per row of ``frame``.

    ``frame`` needs ``date``, ``is_last_trading_day_month``, ``quarterly_options_expiry`` and
    ``monthly_options_expiry`` (all in the harness fold frames).
    """
    cal = calendar.reindex(frame.date).fillna(0).set_axis(frame.index)
    flag = lambda col: frame[col].fillna(False).astype(bool)
    rules = [
        ("russell", cal.russell > 0),
        ("msci", cal.msci > 0),
        ("sp_quad", (cal.sp_rebal > 0) | flag("quarterly_options_expiry")),
        ("month_end", flag("is_last_trading_day_month")),
        ("monthly_expiry", flag("monthly_options_expiry")),
    ]
    out = pd.Series("normal", index=frame.index, dtype=object)
    for name, mask in reversed(rules):  # reversed so earlier rules win
        out[mask.to_numpy()] = name
    return out
