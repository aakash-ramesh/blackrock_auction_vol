# Auction target stability

The descriptive comparison uses 1,003 stocks with an eligible label and positive historical references on every one of the 1,003 trading dates in 2022–2025. All targets use the same cohort, with observed references throughout. The cohort favors stocks with continuous coverage over the four years. The ridge backtest uses the full eligible universe.

- [Realized target series](target_time_series.png) shows each target in its native units with its daily cohort median and middle 50% band. The dark line smooths the cohort median over 63 sessions. Vertical scales vary by target.
- [Quarterly spread and upper tails](target_spread_and_tails.png) shows each stock's quarterly interquartile range and P99-minus-median, scaled by that stock's full-year 2022 value, then takes the median across stocks. A value of one matches the 2022 reference. Each panel has its own vertical scale.
- [Full diagnostics](target_stability.csv) also records signed shifts and pooled tail quantiles. Each table entry below is a median across stocks. Absolute shifts compare stock medians between 2022 and 2025 in units of the stock's 2022 IQR. Lag correlations are computed within stock over the full four years.

| Target | Absolute median shift / 2022 IQR | 2025 / 2022 IQR | 2025 / 2022 upper tail | Lag 1 | Lag 20 |
|---|---:|---:|---:|---:|---:|
| Raw auction volume | 1.379 | 1.665 | 1.246 | +0.090 | +0.016 |
| Square-root volume | 1.220 | 1.254 | 1.037 | +0.251 | +0.098 |
| Direct log volume | 1.065 | 0.943 | 0.856 | +0.394 | +0.196 |
| 20-day median-relative log (current) | 0.037 | 0.920 | 0.833 | +0.225 | -0.055 |
| 60-day median-relative log | 0.067 | 0.936 | 0.847 | +0.250 | +0.004 |
| 20-day mean-relative log | 0.099 | 0.891 | 0.847 | +0.291 | -0.103 |
| 60-day mean-relative log | 0.168 | 0.911 | 0.838 | +0.289 | +0.029 |
| Daily liquidity-relative log | 0.545 | 0.897 | 0.850 | +0.331 | +0.043 |
| Excess over 20-day median, shares | 0.050 | 1.622 | 1.234 | +0.051 | -0.021 |
| Ratio to 20-day auction median | 0.035 | 0.916 | 0.676 | +0.054 | -0.024 |
| Ratio to average daily volume | 0.619 | 1.232 | 0.972 | +0.063 | -0.015 |
| One-day log growth | 0.031 | 0.889 | 0.863 | -0.459 | -0.019 |
| One-day simple growth | 0.030 | 0.883 | 0.729 | -0.149 | -0.012 |

These charts describe changes in level, spread and serial dependence over this period. They support visual screening for stationarity. See the [NIST stationarity reference](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc442.htm).

Reproduce with `.venv/bin/python notebooks/plot_auction_targets.py`.
