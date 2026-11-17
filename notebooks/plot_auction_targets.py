"""Plot the realized ridge targets and compare their stability within stocks."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import numpy as np
import pandas as pd

from notebooks.compare_ridge_targets import BASE, OUT, TARGETS, YEARS, build_panel, transform

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import EngFormatter, MaxNLocator, PercentFormatter


LABELS = {
    "raw_volume": "Raw auction volume",
    "sqrt_volume": "Square-root volume",
    "log_volume": "Direct log volume",
    BASE: "20-day median-relative log (current)",
    "median60_log_residual": "60-day median-relative log",
    "mean20_log_residual": "20-day mean-relative log",
    "mean60_log_residual": "60-day mean-relative log",
    "adv20_log_residual": "Daily liquidity-relative log",
    "median20_share_residual": "Excess over 20-day median, shares",
    "median20_ratio": "Ratio to 20-day auction median",
    "adv20_ratio": "Ratio to average daily volume",
    "lag1_log_growth": "One-day log growth",
    "lag1_pct_growth": "One-day simple growth",
}


def balanced_panel():
    OUT.mkdir(parents=True, exist_ok=True)
    references = sorted({column for _, column, _ in TARGETS.values() if column})
    columns = ", ".join(references)
    positive_refs = " AND ".join(f"isfinite({c}) AND {c} > 0" for c in references)
    with duckdb.connect() as con:
        con.execute("SET threads=4")
        panel = build_panel(con)
        frame = con.execute(f"""
            WITH period AS (
                SELECT date, permno, auction_volume, eligible_target, {columns}
                FROM read_parquet(?) WHERE year BETWEEN ? AND ?
            ), complete AS (
                SELECT * FROM period
                WHERE eligible_target AND isfinite(auction_volume) AND {positive_refs}
            ), cohort AS (
                SELECT permno FROM complete GROUP BY permno
                HAVING count(*) = (SELECT count(DISTINCT date) FROM period)
            )
            SELECT * FROM complete SEMI JOIN cohort USING (permno)
            ORDER BY date, permno
        """, [str(panel), YEARS[0], YEARS[-1]]).df()
    return frame, dict.fromkeys(references, np.nan)


def layout(title, subtitle):
    fig, axes = plt.subplots(5, 3, figsize=(13.5, 12), layout="constrained")
    fig.suptitle(f"{title}\n{subtitle}", fontsize=15, fontweight="medium")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in list(axes.flat)[len(LABELS):]:
        ax.set_axis_off()
    return fig, list(axes.flat)


def plot_and_summarize():
    assert set(LABELS) == set(TARGETS)
    frame, fallbacks = balanced_panel()
    dates = pd.DatetimeIndex(frame.date.unique())
    stocks = frame.permno.nunique()
    assert len(frame) == len(dates) * stocks and not frame.duplicated(["date", "permno"]).any()
    assert frame.groupby("date").permno.nunique().eq(stocks).all()
    first_year, last_year = YEARS[0], YEARS[-1]
    early, late = dates.year == first_year, dates.year == last_year
    quarters = dates.to_period("Q")
    subtitle = f"{stocks:,} stocks with complete observations on all {len(dates):,} trading dates, {first_year}–{last_year}"
    time_fig, time_axes = layout("Realized auction target series", subtitle)
    spread_fig, spread_axes = layout("How within-stock spread and tails change", subtitle)
    rows = []
    for (name, label), time_ax, spread_ax in zip(LABELS.items(), time_axes, spread_axes):
        values = transform(frame, name, fallbacks).reshape(len(dates), stocks)
        assert np.isfinite(values).all(), name
        lower, median, upper = np.quantile(values, [.25, .5, .75], axis=1)
        smooth = pd.Series(median, index=dates).rolling(63, min_periods=20).median()
        color = "#bd5b25" if name == BASE else "#216c93"
        time_ax.fill_between(dates, lower, upper, color=color, alpha=.13, linewidth=0)
        time_ax.plot(dates, median, color=color, alpha=.35, linewidth=.6)
        time_ax.plot(dates, smooth, color=color, linewidth=1.8)
        time_ax.set_title(label, loc="left", fontsize=10, fontweight="medium")
        time_ax.yaxis.set_major_locator(MaxNLocator(4))
        if name == "lag1_pct_growth":
            time_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        elif name in {"raw_volume", "median20_share_residual"}:
            time_ax.yaxis.set_major_formatter(EngFormatter(unit=" shares", sep=""))

        def stock_quantiles(mask):
            return np.quantile(values[mask], [.25, .5, .75, .99], axis=0)

        q0, q1 = stock_quantiles(early), stock_quantiles(late)
        iqr0, tail0 = q0[2] - q0[0], q0[3] - q0[1]
        good = (iqr0 > 0) & (tail0 > 0)
        quarter_dates, quarter_iqr, quarter_tail = [], [], []
        for quarter in quarters.unique():
            mask = quarters == quarter
            q = stock_quantiles(mask)
            quarter_dates.append(dates[mask][-1])
            quarter_iqr.append(np.median((q[2, good] - q[0, good]) / iqr0[good]))
            quarter_tail.append(np.median((q[3, good] - q[1, good]) / tail0[good]))
        spread_ax.axhline(1, color="#7d858c", linestyle="--", linewidth=.8)
        spread_ax.plot(quarter_dates, quarter_iqr, color="#216c93", marker="o", markersize=2.5)
        spread_ax.plot(quarter_dates, quarter_tail, color="#a14935", marker="o", markersize=2.5)
        spread_ax.set_title(label, loc="left", fontsize=10, fontweight="medium")
        spread_ax.yaxis.set_major_locator(MaxNLocator(4))
        spread_ax.set_xlim(dates[0], dates[-1] + pd.Timedelta(days=15))

        def lag_corr(lag):
            left, right = values[:-lag], values[lag:]
            left = left - left.mean(axis=0)
            right = right - right.mean(axis=0)
            return (left * right).sum(axis=0) / np.sqrt((left**2).sum(axis=0) * (right**2).sum(axis=0))

        shift = (q1[1, good] - q0[1, good]) / iqr0[good]
        rows.append({
            "target": name, "stocks": stocks, "stocks_in_scale_comparison": int(good.sum()),
            "median_level_shift_in_2022_iqrs": np.median(shift),
            "median_absolute_level_shift_in_2022_iqrs": np.median(np.abs(shift)),
            "median_2025_to_2022_iqr_ratio": np.median((q1[2, good] - q1[0, good]) / iqr0[good]),
            "median_2025_to_2022_upper_tail_ratio": np.median((q1[3, good] - q1[1, good]) / tail0[good]),
            "median_lag1_correlation": np.nanmedian(lag_corr(1)),
            "median_lag20_correlation": np.nanmedian(lag_corr(20)),
            "pooled_p01": np.quantile(values, .01), "pooled_p50": np.median(values),
            "pooled_p99": np.quantile(values, .99), "pooled_max": values.max(),
        })
    time_axes[-2].text(0, .65, "Pale line: daily median across stocks\nDark line: trailing 63-session median\nBand: daily 25th–75th percentiles",
                       fontsize=10, va="center", transform=time_axes[-2].transAxes)
    spread_axes[-2].legend(handles=[
        Line2D([], [], color="#216c93", label="Middle spread (P75 − P25)"),
        Line2D([], [], color="#a14935", label="Upper tail (P99 − median)"),
        Line2D([], [], color="#7d858c", linestyle="--", label="Full-year 2022 reference")],
        loc="center left", frameon=False, fontsize=10)
    time_fig.savefig(OUT / "target_time_series.png", dpi=180)
    spread_fig.savefig(OUT / "target_spread_and_tails.png", dpi=180)
    plt.close("all")
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / "target_stability.csv", index=False)
    table = [
        "| Target | Absolute median shift / 2022 IQR | 2025 / 2022 IQR | 2025 / 2022 upper tail | Lag 1 | Lag 20 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in stats.itertuples():
        table.append(f"| {LABELS[row.target]} | {row.median_absolute_level_shift_in_2022_iqrs:.3f} | "
                     f"{row.median_2025_to_2022_iqr_ratio:.3f} | {row.median_2025_to_2022_upper_tail_ratio:.3f} | "
                     f"{row.median_lag1_correlation:+.3f} | {row.median_lag20_correlation:+.3f} |")
    (OUT / "target_stability.md").write_text(f"""# Auction target stability

The descriptive comparison uses {stocks:,} stocks with an eligible label and positive historical references on every one of the {len(dates):,} trading dates in {first_year}–{last_year}. All targets use the same cohort, with observed references throughout. The cohort favors stocks with continuous coverage over the four years. The ridge backtest uses the full eligible universe.

- [Realized target series](target_time_series.png) shows each target in its native units with its daily cohort median and middle 50% band. The dark line smooths the cohort median over 63 sessions. Vertical scales vary by target.
- [Quarterly spread and upper tails](target_spread_and_tails.png) shows each stock's quarterly interquartile range and P99-minus-median, scaled by that stock's full-year 2022 value, then takes the median across stocks. A value of one matches the 2022 reference. Each panel has its own vertical scale.
- [Full diagnostics](target_stability.csv) also records signed shifts and pooled tail quantiles. Each table entry below is a median across stocks. Absolute shifts compare stock medians between 2022 and 2025 in units of the stock's 2022 IQR. Lag correlations are computed within stock over the full four years.

{chr(10).join(table)}

These charts describe changes in level, spread and serial dependence over this period. They support visual screening for stationarity. See the [NIST stationarity reference](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc442.htm).

Reproduce with `.venv/bin/python notebooks/plot_auction_targets.py`.
""")
    print(stats.drop(columns=["stocks_in_scale_comparison"]).to_string(index=False))


if __name__ == "__main__":
    plot_and_summarize()
