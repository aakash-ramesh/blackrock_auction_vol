# Stock and universe auction target stationarity

All 13 targets are constructed within stock before aggregation. The four plotted examples are JPM, FBP, BLK and CATO from the project's representative-stock set. The fixed cohort includes 1,003 stocks with complete observations on 1,003 sessions from 2022 through 2025. The universe aggregates use all eligible stocks with an observed value for each target that day, so membership can vary. The fixed-cohort aggregates hold membership constant. The means give each available stock equal weight.

- [Stock and universe plots, targets 1–7](stock_vs_universe_targets_1.png)
- [Stock and universe plots, targets 8–13](stock_vs_universe_targets_2.png)
- [Test statistics and p-values](target_stationarity_tests.csv)
- [Within-stock summary and daily change comparisons](target_stationarity_summary.csv)
- [Daily fixed and available-universe aggregates](target_daily_aggregates.csv)

## ADF and KPSS results

At the 5% threshold, **S** means ADF rejects a unit root and KPSS does not reject level stationarity. **U** is the reverse, **B** means both reject, and **I** means neither rejects. Stock percentages are descriptive counts across dependent series. They use unadjusted per-series tests and give a cohort summary. KPSS p-values at .01 or .10 are limits of the tabulated range.

| Target | Within-stock S | S with KPSS lag 20 | Fixed mean | Fixed median | Universe mean | Universe median |
|---|---:|---:|:---:|:---:|:---:|:---:|
| Raw auction volume | 23.2% | 25.6% | B | B | B | B |
| Square-root volume | 9.7% | 10.7% | B | B | B | B |
| Direct log volume | 5.9% | 7.4% | B | B | B | B |
| 20-day median-relative log (current) | 99.0% | 99.0% | S | S | S | S |
| 60-day median-relative log | 95.7% | 96.8% | S | S | S | S |
| 20-day mean-relative log | 93.3% | 93.1% | S | S | S | S |
| 60-day mean-relative log | 77.1% | 80.8% | S | S | B | S |
| Daily liquidity-relative log | 14.2% | 16.0% | B | B | B | B |
| Excess over 20-day median, shares | 90.9% | 90.7% | S | S | S | S |
| Ratio to 20-day auction median | 94.6% | 94.1% | S | S | B | S |
| Ratio to average daily volume | 40.7% | 40.5% | B | B | B | B |
| One-day log growth | 97.3% | 100.0% | S | S | S | S |
| One-day simple growth | 89.3% | 88.8% | S | S | S | S |

ADF includes an intercept and selects from 0 to 20 lags by BIC. KPSS uses automatic bandwidth selection and a separate sensitivity run with 20 lags. The CSV also includes intercept-plus-linear-trend specifications for the four examples and four aggregates. All tests use unsmoothed daily observations. ADF and KPSS assess unit roots and level or trend stability. The tests can disagree in the presence of breaks or other departures from their models. See the [ADF reference](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html) and [KPSS reference](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html).

The daily-change ratios compare the IQR of aggregate first differences with the median stock's first-difference IQR. Ratios below one indicate a smaller typical day-to-day range in the aggregate, in the target's native units. The four-year complete-coverage cohort favors continuously observed stocks.

Reproduce with `VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python notebooks/test_auction_target_stationarity.py` using Statsmodels 0.15.0.
