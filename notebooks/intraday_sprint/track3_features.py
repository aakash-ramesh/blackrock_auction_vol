"""Track 3: calendar-adjusted 15:30 volume surprises.

Idea: the existing surprise, ``cp_volume_surprise20`` = today's volume by 15:30 / its
20-day average, reads a month-end or index-rebalance day as "surprisingly busy" even
when that extra volume is completely normal for that kind of day. It also inflates the
baseline for the 20 days after any special day. This module separates the expected
part from the real surprise.

For each fold year Y, using only rows dated before Y-01-01:

1. Day-type multipliers. Regress log(cp_volume_surprise20) on day-type dummies
   (month-end, quarter-end, expiry, Russell, MSCI, S&P add/delete, FOMC, ...).
   The fitted value ``cal_expected_cp`` is how much busier than usual a stock should be
   by 15:30 on a day like today. A second regression does the same for the auction's
   share of the checkpoint volume: log(auction / cp volume) - log(ratio20) gives
   ``cal_expected_ratio``.
2. Clean baselines. Divide each past day's checkpoint volume (and auction/checkpoint
   ratio) by its day-type multiplier, then take the 20 prior available days. Special
   days no longer inflate the baseline.
3. Features (all known at 15:30 on the forecast day):
   - ``cal_surprise``      log(cp volume) - expected day effect - log(clean baseline)
                           = how unusual today is *for this kind of day*
   - ``cal_expected_cp``   expected day effect on checkpoint volume
   - ``cal_expected_ratio`` expected day effect on auction / checkpoint ratio
   - ``cal_rule_correction`` log1p(cp volume x clean ratio x expected ratio uplift)
                           - log daily forecast: a calendar-aware "rule" forecast,
                           expressed as a correction to the daily anchor
   - ``cal_surprise_x_special`` the surprise on special days only (0 otherwise)
4. Projected full-day volume (v2, ``PROJECTION_FEATURES``). From past days, learn what
   fraction of a stock's full-day volume is normally done by 15:30, adjusted for the
   kind of day (a clean 20-day median, plus a day-type effect fitted on history).
   A past day's full-day volume is only known the next morning (``lag1_daily_volume``
   on the next trading day), so day t uses fractions up to day t-1 only.
   - ``cal_projected_day_surprise``   log(projected full-day volume) - log(20-day average
                                      daily volume)
   - ``cal_projected_rest_correction`` log1p(projected volume still to trade after 15:30,
                                      auction included) - log daily forecast

Multipliers are fitted on history only; baselines use only earlier days (shifted by
one). Today's auction is never used for today's features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from intraday_harness import FOMC, REBALANCE

# Day-type dummies: all known before the trading day starts.
PANEL_FLAGS = [
    "is_last_trading_day_month", "is_last_trading_day_quarter", "is_last_trading_day_year",
    "is_first_trading_day_month", "monthly_options_expiry", "quarterly_options_expiry",
    "is_monday", "is_friday", "before_long_market_gap", "after_long_market_gap",
]
EVENT_FLAGS = ["russell_recon", "msci_review", "sp500_add_day", "sp500_del_day", "fomc_day"]
DAY_TYPES = PANEL_FLAGS + EVENT_FLAGS
SPECIAL = ["is_last_trading_day_month", "quarterly_options_expiry", "russell_recon", "msci_review",
           "sp500_add_day", "sp500_del_day", "fomc_day"]
BASE_COLUMNS = ["date", "permno", "cp_available", "cp_cum_volume", "cp_volume_surprise20",
                "auction_volume", "ratio20_auction_to_checkpoint", "avg20_daily_volume", *PANEL_FLAGS]
FEATURES = ["cal_surprise", "cal_expected_cp", "cal_expected_ratio", "cal_rule_correction",
            "cal_surprise_x_special"]
PROJECTION_FEATURES = ["cal_projected_day_surprise", "cal_projected_rest_correction"]
ALL_FEATURES = FEATURES + PROJECTION_FEATURES
WINDOW = 20


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with an intercept; returns [intercept, coefficients...]."""
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta


class CalendarFeatures:
    """Builds the Track 3 columns once per fold year and serves them to the harness."""

    def __init__(self, harness, day_types=DAY_TYPES, window: int = WINDOW, suffix: str = ""):
        """``day_types``: which dummies get a multiplier. ``suffix`` is appended to the column
        names so two versions (e.g. with and without rebalance events) can be compared."""
        self.h = harness
        self.day_types = list(day_types)
        self.suffix = suffix
        self.names = [f + suffix for f in ALL_FEATURES]
        self.window = window
        self.multipliers: dict[int, pd.DataFrame] = {}
        self._cache: dict[int, pd.DataFrame] = {}
        self._panel = self._load_panel()

    # --------------------------------------------------------------- inputs
    def _load_panel(self) -> pd.DataFrame:
        panel = self.h.full_panel("2021-01-01", "2026-01-01", BASE_COLUMNS)
        if REBALANCE.is_file():
            reb = pd.read_parquet(REBALANCE, columns=["date", "permno", "russell_recon", "msci_review",
                                                     "sp500_add_day", "sp500_del_day"])
            reb["date"] = pd.to_datetime(reb["date"])
            panel = panel.merge(reb, on=["date", "permno"], how="left", validate="one_to_one")
            # Russell/MSCI are market-wide dates: fill stocks missing from the rebalance file by date.
            for col in ("russell_recon", "msci_review"):
                by_date = reb.groupby("date")[col].max()
                panel[col] = panel[col].fillna(panel["date"].map(by_date))
        else:
            for col in ("russell_recon", "msci_review", "sp500_add_day", "sp500_del_day"):
                panel[col] = 0
        fomc = set(pd.to_datetime(pd.read_csv(FOMC)["announcement_date"]).dt.normalize())
        panel["fomc_day"] = panel["date"].isin(fomc)
        for col in DAY_TYPES:
            panel[col] = panel[col].fillna(0).astype(float)
        panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)
        # Full-day volume of day t = lag1_daily_volume reported on the next market trading day
        # (early-close days included, so the next day is the true next session).
        nxt = self.h.con.sql("""
            WITH d AS (SELECT DISTINCT date FROM checkpoint_panel),
                 n AS (SELECT date, lead(date) OVER (ORDER BY date) AS next_date FROM d)
            SELECT n.date, p.permno, p.lag1_daily_volume AS day_volume
            FROM n JOIN checkpoint_panel p ON p.date = n.next_date
        """).df()
        nxt["date"] = pd.to_datetime(nxt["date"])
        panel = panel.merge(nxt, on=["date", "permno"], how="left", validate="one_to_one")
        ok = panel.cp_available.astype(bool) & (panel.cp_cum_volume > 0)
        panel["log_cp"] = np.where(ok, np.log(panel.cp_cum_volume.where(ok)), np.nan)
        panel["log_surprise"] = np.log(panel.cp_volume_surprise20.where(ok & (panel.cp_volume_surprise20 > 0)))
        ratio_ok = ok & (panel.auction_volume > 0)
        panel["log_ratio"] = np.log((panel.auction_volume / panel.cp_cum_volume).where(ratio_ok))
        panel["log_ratio_vs20"] = panel["log_ratio"] - np.log(
            panel.ratio20_auction_to_checkpoint.where(panel.ratio20_auction_to_checkpoint > 0))
        frac_ok = ok & (panel.day_volume >= panel.cp_cum_volume)
        panel["log_frac"] = np.log((panel.cp_cum_volume / panel.day_volume).where(frac_ok))
        # Raw baseline uses days before t only; day t's own fraction is known after the close.
        raw_base = panel.groupby("permno").log_frac.transform(
            lambda s: s.shift(1).rolling(self.window, min_periods=5).median())
        panel["log_frac_vs20"] = panel["log_frac"] - raw_base
        return panel

    # ---------------------------------------------------------- per fold
    def _fit_multipliers(self, year: int) -> tuple[np.ndarray, np.ndarray]:
        hist = self._panel[self._panel.date < pd.Timestamp(f"{year}-01-01")]
        x = hist[self.day_types].to_numpy(float)
        out = []
        for target in ("log_surprise", "log_ratio_vs20", "log_frac_vs20"):
            y = hist[target].to_numpy(float)
            keep = np.isfinite(y)
            beta = _ols(x[keep], y[keep])
            out.append(beta[1:])  # day effects only: a normal day has expected effect 0
        self.multipliers[year] = pd.DataFrame(
            {"cp_volume": out[0], "auction_ratio": out[1], "cp_fraction_of_day": out[2]}, index=self.day_types)
        return out[0], out[1], out[2]

    def _build(self, year: int) -> pd.DataFrame:
        cp_beta, ratio_beta, frac_beta = self._fit_multipliers(year)
        p = self._panel
        x = p[self.day_types].to_numpy(float)
        expected_cp, expected_ratio, expected_frac = x @ cp_beta, x @ ratio_beta, x @ frac_beta

        # Clean baselines: deflate each day by its own expected effect, then average the
        # previous `window` available days (shift(1) keeps today out of its own baseline).
        deflated_cp = pd.Series(np.exp(p.log_cp - expected_cp), index=p.index)
        base_cp = deflated_cp.groupby(p.permno).transform(
            lambda s: s.shift(1).rolling(self.window, min_periods=5).mean())
        deflated_ratio = pd.Series(p.log_ratio - expected_ratio, index=p.index)
        base_ratio = deflated_ratio.groupby(p.permno).transform(
            lambda s: s.shift(1).rolling(self.window, min_periods=5).median())

        surprise = p.log_cp - expected_cp - np.log(base_cp)
        special_cols = [c for c in SPECIAL if c in self.day_types]
        special = (p[special_cols].to_numpy(float).sum(axis=1) > 0)
        rule_log = np.log1p(np.exp(p.log_cp + base_ratio + expected_ratio))

        # Projected full-day volume: today's 15:30 volume / expected fraction of the day by 15:30.
        deflated_frac = pd.Series(p.log_frac - expected_frac, index=p.index)
        base_frac = deflated_frac.groupby(p.permno).transform(
            lambda s: s.shift(1).rolling(self.window, min_periods=5).median())
        log_frac_today = np.minimum(base_frac + expected_frac, 0.0)   # a fraction can't exceed 1
        log_projected = p.log_cp - log_frac_today
        rest = np.exp(log_projected) - np.exp(p.log_cp)
        projected_day_surprise = log_projected - np.log(p.avg20_daily_volume.where(p.avg20_daily_volume > 0))
        feats = pd.DataFrame({
            "date": p.date, "permno": p.permno,
            "cal_surprise": surprise,
            "cal_expected_cp": expected_cp,
            "cal_expected_ratio": expected_ratio,
            "cal_rule_log": rule_log,
            "cal_surprise_x_special": np.where(special, surprise, 0.0),
            "cal_projected_day_surprise": projected_day_surprise,
            "cal_rest_log": np.log1p(np.maximum(rest, 0.0)),
        })
        return feats

    # ------------------------------------------------------- harness hook
    def __call__(self, frame: pd.DataFrame, fold_year: int, role: str) -> pd.DataFrame:
        if fold_year not in self._cache:
            self._cache[fold_year] = self._build(fold_year)
        feats = self._cache[fold_year]
        merged = frame[["date", "permno"]].merge(feats, on=["date", "permno"], how="left", validate="one_to_one")
        merged.index = frame.index
        # Rule forecast as a correction to the daily anchor (same units as the target).
        merged["cal_rule_correction"] = merged.pop("cal_rule_log") - frame["log_daily_forecast"]
        merged["cal_projected_rest_correction"] = merged.pop("cal_rest_log") - frame["log_daily_forecast"]
        out = merged[ALL_FEATURES].replace([np.inf, -np.inf], np.nan)
        out.columns = self.names
        # Missing checkpoint or too little history -> NaN; the preprocessor median-fills
        # and adds a missing flag, and the harness falls back to the daily anchor anyway.
        return out
