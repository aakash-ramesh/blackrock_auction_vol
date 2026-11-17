"""Compare individual and aggregate auction targets with ADF and KPSS."""

from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import numpy as np
import pandas as pd
from statsmodels import __version__ as statsmodels_version
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from notebooks.compare_ridge_targets import OUT, TARGETS, YEARS, transform
from notebooks.plot_auction_targets import LABELS, balanced_panel

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, MaxNLocator, PercentFormatter


STOCKS = {47896: "JPM", 11018: "FBP", 87267: "BLK", 11267: "CATO"}
AGGREGATES = {
    "fixed_mean": "Fixed cohort mean",
    "fixed_median": "Fixed cohort median",
    "eligible_mean": "Universe mean",
    "eligible_median": "Universe median",
}
ALPHA = .05


def test_series(values, regression="c"):
    # Rescale for numerical conditioning. Both tests are invariant to this change.
    x = (values - values.mean()) / values.std()
    adf = adfuller(x, maxlag=20, autolag="BIC", regression=regression, result_object=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kp = kpss(x, regression=regression, nlags="auto", result_object=False)
        kp20 = kpss(x, regression=regression, nlags=20, result_object=False)
    return {
        "adf_stat": adf[0], "adf_p": adf[1], "adf_lag": adf[2],
        "kpss_stat": kp[0], "kpss_p": kp[1], "kpss_lag": kp[2],
        "kpss_lag20_stat": kp20[0], "kpss_lag20_p": kp20[1],
        "classification": classify(adf[1], kp[1]),
        "classification_kpss_lag20": classify(adf[1], kp20[1]),
    }


def classify(adf_p, kpss_p):
    adf_rejects, kpss_rejects = adf_p < ALPHA, kpss_p < ALPHA
    if adf_rejects and not kpss_rejects:
        return "stationary-compatible"
    if not adf_rejects and kpss_rejects:
        return "unit-root-compatible"
    return "both reject" if adf_rejects else "neither rejects"


def plot_comparison(dates, series, names, path):
    columns = list(STOCKS.values()) + ["eligible_mean", "eligible_median"]
    fig, axes = plt.subplots(len(names), len(columns), figsize=(16, 1.8 * len(names) + 1),
                             sharex=True, layout="constrained")
    fig.suptitle("Individual stocks and the daily auction target across the universe\n"
                 "Pale lines show daily observations. Dark lines show trailing 63-session medians. "
                 "Vertical scales vary by panel.", fontsize=13)
    ticks = [dates[dates.year == year][0] for year in YEARS]
    for i, name in enumerate(names):
        for j, key in enumerate(columns):
            ax = axes[i, j]
            values = series[name][key]
            smooth = pd.Series(values, index=dates).rolling(63, min_periods=20).median()
            color = "#21718b" if key in STOCKS.values() else "#9e4d26"
            ax.plot(dates, values, color=color, alpha=.38, linewidth=.55)
            ax.plot(dates, smooth, color=color, linewidth=1.4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.18)
            ax.tick_params(labelsize=7)
            ax.set_xlim(dates[0], dates[-1])
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            if name == "lag1_pct_growth":
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            elif name in {"raw_volume", "median20_share_residual"}:
                ax.yaxis.set_major_formatter(EngFormatter(sep=""))
            if i == 0:
                ax.set_title(AGGREGATES.get(key, key), fontsize=10, fontweight="bold")
            if j == 0:
                label = LABELS[name].replace(" (current)", "\n(current)")
                label = label.replace("-relative ", "-relative\n").replace("Excess over ", "Excess over\n")
                ax.set_ylabel(label, fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run():
    frame, missing_refs = balanced_panel()
    dates = pd.DatetimeIndex(frame.date.unique())
    permnos = frame.permno.drop_duplicates().to_numpy()
    assert len(frame) == len(dates) * len(permnos)
    assert not frame.duplicated(["date", "permno"]).any()
    assert np.all(frame.permno.to_numpy().reshape(len(dates), -1) == permnos)
    selected = {ticker: int(np.flatnonzero(permnos == permno).item()) for permno, ticker in STOCKS.items()}
    columns = ", ".join(missing_refs)
    with duckdb.connect() as con:
        universe = con.execute(f"""
            SELECT date, auction_volume, {columns}
            FROM read_parquet(?) WHERE year BETWEEN ? AND ? AND eligible_target
            ORDER BY date
        """, [str(OUT / "daily_panel.parquet"), YEARS[0], YEARS[-1]]).df()

    rows, aggregate_rows, summary_rows, plot_series = [], [], [], {}
    for name, label in LABELS.items():
        print(f"Testing {label}", flush=True)
        values = transform(frame, name, missing_refs).reshape(len(dates), -1)
        assert np.isfinite(values).all()
        observed = pd.Series(transform(universe, name, missing_refs), index=universe.date)
        aggregate = observed.groupby(level=0).agg(["mean", "median", "count"]).reindex(dates)
        assert np.isfinite(aggregate.to_numpy()).all() and (aggregate["count"] > 0).all()
        aggregate.insert(0, "target", name)
        aggregate["fixed_mean"] = values.mean(axis=1)
        aggregate["fixed_median"] = np.median(values, axis=1)
        aggregate_rows.append(aggregate.reset_index(names="date"))
        aggregate_series = {
            "fixed_mean": aggregate.fixed_mean.to_numpy(),
            "fixed_median": aggregate.fixed_median.to_numpy(),
            "eligible_mean": aggregate["mean"].to_numpy(),
            "eligible_median": aggregate["median"].to_numpy(),
        }
        chosen_series = {ticker: values[:, j] for ticker, j in selected.items()}
        plot_series[name] = {**chosen_series, **aggregate_series}
        stock_tests = []
        for j, permno in enumerate(permnos):
            result = test_series(values[:, j])
            stock_tests.append(result)
            rows.append({"target": name, "series": STOCKS.get(permno, str(permno)),
                         "scope": "stock", "permno": permno, "spec": "level", **result})
        for key, x in aggregate_series.items():
            rows.append({"target": name, "series": key, "scope": "aggregate",
                         "spec": "level", **test_series(x)})
        for key, x in plot_series[name].items():
            rows.append({"target": name, "series": key, "scope": "stock" if key in STOCKS.values() else "aggregate",
                         "permno": next((p for p, ticker in STOCKS.items() if ticker == key), None),
                         "spec": "linear trend", **test_series(x, regression="ct")})
        tests = pd.DataFrame(stock_tests)
        changes = np.diff(values, axis=0)
        stock_iqr = np.diff(np.quantile(changes, [.25, .75], axis=0), axis=0).ravel()
        mean_iqr = np.diff(np.quantile(np.diff(aggregate_series["fixed_mean"]), [.25, .75])).item()
        median_iqr = np.diff(np.quantile(np.diff(aggregate_series["fixed_median"]), [.25, .75])).item()
        summary_rows.append({
            "target": name, "stocks": len(permnos),
            "adf_rejection_rate": tests.adf_p.lt(ALPHA).mean(),
            "kpss_rejection_rate": tests.kpss_p.lt(ALPHA).mean(),
            "stationary_compatible_rate": tests.classification.eq("stationary-compatible").mean(),
            "stationary_compatible_rate_kpss_lag20": tests.classification_kpss_lag20.eq("stationary-compatible").mean(),
            "both_reject_rate": tests.classification.eq("both reject").mean(),
            "fixed_mean_change_iqr_to_typical_stock": mean_iqr / np.median(stock_iqr),
            "fixed_median_change_iqr_to_typical_stock": median_iqr / np.median(stock_iqr),
            "universe_min_stocks": aggregate["count"].min(), "universe_max_stocks": aggregate["count"].max(),
        })

    tests = pd.DataFrame(rows)
    tests["sessions"] = len(dates)
    summary = pd.DataFrame(summary_rows)
    tests.to_csv(OUT / "target_stationarity_tests.csv", index=False)
    summary.to_csv(OUT / "target_stationarity_summary.csv", index=False)
    pd.concat(aggregate_rows, ignore_index=True).to_csv(OUT / "target_daily_aggregates.csv", index=False)
    names = list(LABELS)
    plot_comparison(dates, plot_series, names[:7], OUT / "stock_vs_universe_targets_1.png")
    plot_comparison(dates, plot_series, names[7:], OUT / "stock_vs_universe_targets_2.png")

    levels = tests.query("scope == 'aggregate' and spec == 'level'").pivot(
        index="target", columns="series", values="classification")
    compact = {"stationary-compatible": "S", "unit-root-compatible": "U",
               "both reject": "B", "neither rejects": "I"}
    lines = [
        "| Target | Within-stock S | S with KPSS lag 20 | Fixed mean | Fixed median | Universe mean | Universe median |",
        "|---|---:|---:|:---:|:---:|:---:|:---:|",
    ]
    for row in summary.itertuples():
        outcomes = " | ".join(compact[levels.loc[row.target, key]] for key in AGGREGATES)
        lines.append(f"| {LABELS[row.target]} | {row.stationary_compatible_rate:.1%} | "
                     f"{row.stationary_compatible_rate_kpss_lag20:.1%} | {outcomes} |")
    (OUT / "stock_vs_universe_stationarity.md").write_text(f"""# Stock and universe auction target stationarity

All 13 targets are constructed within stock before aggregation. The four plotted examples are JPM, FBP, BLK and CATO from the project's representative-stock set. The fixed cohort includes {len(permnos):,} stocks with complete observations on {len(dates):,} sessions from {YEARS[0]} through {YEARS[-1]}. The universe aggregates use all eligible stocks with an observed value for each target that day, so membership can vary. The fixed-cohort aggregates hold membership constant. The means give each available stock equal weight.

- [Stock and universe plots, targets 1–7](stock_vs_universe_targets_1.png)
- [Stock and universe plots, targets 8–13](stock_vs_universe_targets_2.png)
- [Test statistics and p-values](target_stationarity_tests.csv)
- [Within-stock summary and daily change comparisons](target_stationarity_summary.csv)
- [Daily fixed and available-universe aggregates](target_daily_aggregates.csv)

## ADF and KPSS results

At the 5% threshold, **S** means ADF rejects a unit root and KPSS does not reject level stationarity. **U** is the reverse, **B** means both reject, and **I** means neither rejects. Stock percentages are descriptive counts across dependent series. They use unadjusted per-series tests and give a cohort summary. KPSS p-values at .01 or .10 are limits of the tabulated range.

{chr(10).join(lines)}

ADF includes an intercept and selects from 0 to 20 lags by BIC. KPSS uses automatic bandwidth selection and a separate sensitivity run with 20 lags. The CSV also includes intercept-plus-linear-trend specifications for the four examples and four aggregates. All tests use unsmoothed daily observations. ADF and KPSS assess unit roots and level or trend stability. The tests can disagree in the presence of breaks or other departures from their models. See the [ADF reference](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html) and [KPSS reference](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html).

The daily-change ratios compare the IQR of aggregate first differences with the median stock's first-difference IQR. Ratios below one indicate a smaller typical day-to-day range in the aggregate, in the target's native units. The four-year complete-coverage cohort favors continuously observed stocks.

Reproduce with `VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python notebooks/test_auction_target_stationarity.py` using Statsmodels {statsmodels_version}.
""")


if __name__ == "__main__":
    run()
